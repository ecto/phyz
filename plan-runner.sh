#!/bin/bash
# plan-runner: execute a multi-phase implementation plan with Claude Code
#
# Usage:
#   ./plan-runner.sh <plan.md> [options]
#
# Options:
#   --start N        Start from phase N (default: 1)
#   --end N          Stop after phase N (default: all)
#   --dry-run        Print prompts without executing
#   --retry N        Retry failed phases N times (default: 1)
#   --model MODEL    Claude model to use (default: sonnet)
#   --no-commit      Don't auto-commit between phases
#   --verify CMD     Verification command to run after each phase (default: from plan)
#
# Plan format (markdown):
#   # Project Title           <- used as context
#   ## Phase 1: Name          <- phase delimiter
#   Description of work...
#   ### Verify                <- optional verification block
#   ```bash
#   cargo test
#   ```
#   ## Phase 2: Name
#   ...

set -euo pipefail

# ── defaults ──
START_PHASE=1
END_PHASE=999
DRY_RUN=false
MAX_RETRIES=1
MODEL="sonnet"
AUTO_COMMIT=true
VERIFY_CMD=""
PLAN_FILE=""

# ── parse args ──
while [[ $# -gt 0 ]]; do
  case $1 in
    --start)     START_PHASE="$2"; shift 2 ;;
    --end)       END_PHASE="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --retry)     MAX_RETRIES="$2"; shift 2 ;;
    --model)     MODEL="$2"; shift 2 ;;
    --no-commit) AUTO_COMMIT=false; shift ;;
    --verify)    VERIFY_CMD="$2"; shift 2 ;;
    -h|--help)
      head -20 "$0" | tail -19
      exit 0
      ;;
    *)
      if [[ -z "$PLAN_FILE" ]]; then
        PLAN_FILE="$1"
      else
        echo "Unknown option: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$PLAN_FILE" ]]; then
  echo "Usage: $0 <plan.md> [options]" >&2
  exit 1
fi

if [[ ! -f "$PLAN_FILE" ]]; then
  echo "Plan file not found: $PLAN_FILE" >&2
  exit 1
fi

# ── parse plan into phases ──
# Uses parallel indexed arrays (bash 3.2 compatible, no associative arrays)
PHASE_NUMBERS=()
PHASE_TITLES=()
PHASE_BODIES=()

current_num=""
current_title=""
current_body=""
in_phase=false

while IFS= read -r line; do
  if [[ "$line" =~ ^##\ +[Pp]hase\ +([0-9]+):?\ *(.*) ]]; then
    # Emit previous phase
    if $in_phase && [[ -n "$current_num" ]]; then
      PHASE_NUMBERS+=("$current_num")
      PHASE_TITLES+=("$current_title")
      PHASE_BODIES+=("$current_body")
    fi
    current_num="${BASH_REMATCH[1]}"
    current_title="${BASH_REMATCH[2]}"
    current_body=""
    in_phase=true
  elif $in_phase; then
    current_body+="$line"$'\n'
  fi
done < "$PLAN_FILE"

# Emit last phase
if $in_phase && [[ -n "$current_num" ]]; then
  PHASE_NUMBERS+=("$current_num")
  PHASE_TITLES+=("$current_title")
  PHASE_BODIES+=("$current_body")
fi

# Extract verification command from phase body (```bash block under ### Verify)
extract_verify() {
  local body="$1"
  local in_verify=false
  local in_code=false
  local cmd=""

  while IFS= read -r line; do
    if [[ "$line" =~ ^###\ *[Vv]erif ]]; then
      in_verify=true
    elif $in_verify && [[ "$line" == '```bash' || "$line" == '```sh' ]]; then
      in_code=true
    elif $in_verify && $in_code && [[ "$line" == '```' ]]; then
      break
    elif $in_code; then
      cmd+="$line"$'\n'
    fi
  done <<< "$body"

  echo "$cmd"
}

TOTAL=${#PHASE_NUMBERS[@]}

if [[ $TOTAL -eq 0 ]]; then
  echo "No phases found in $PLAN_FILE" >&2
  echo "Expected format: ## Phase 1: Title" >&2
  exit 1
fi

echo "Found $TOTAL phases in $(basename "$PLAN_FILE")"
echo ""

# ── read project preamble (everything before first ## Phase) ──
PREAMBLE=""
while IFS= read -r line; do
  if [[ "$line" =~ ^##\ +[Pp]hase ]]; then
    break
  fi
  PREAMBLE+="$line"$'\n'
done < "$PLAN_FILE"

# ── log setup ──
LOG_DIR=".plan-runner"
mkdir -p "$LOG_DIR"

# ── execute phases ──
PASSED=0
FAILED=0
SKIPPED=0

for idx in $(seq 0 $((TOTAL - 1))); do
  num="${PHASE_NUMBERS[$idx]}"
  title="${PHASE_TITLES[$idx]}"
  body="${PHASE_BODIES[$idx]}"

  if [[ $num -lt $START_PHASE || $num -gt $END_PHASE ]]; then
    ((SKIPPED++))
    continue
  fi

  phase_verify=$(extract_verify "$body")

  # Use explicit verify command if provided, else from plan, else empty
  verify="${VERIFY_CMD:-$phase_verify}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Phase $num: $title"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Build the prompt
  PROMPT="You are implementing a multi-phase project. Here is the project context:

---
$PREAMBLE
---

You are now implementing Phase $num: $title

Here is the detailed specification for this phase:

$body

IMPORTANT INSTRUCTIONS:
- Implement everything described in this phase specification
- Read existing code first to understand the current state
- Follow existing patterns and conventions in the codebase
- Run tests after implementation to verify correctness
- Fix any test failures, clippy warnings, or fmt issues before finishing"

  if [[ -n "$verify" ]]; then
    PROMPT+="

After implementation, run this verification:
\`\`\`bash
$verify
\`\`\`"
  fi

  if $AUTO_COMMIT; then
    # lowercase title for commit message (bash 3.2 compatible)
    lower_title=$(echo "$title" | tr '[:upper:]' '[:lower:]')
    PROMPT+="

When everything passes, create a git commit with message: 'phase $num: $lower_title'
Stage only the files you created or modified."
  fi

  if $DRY_RUN; then
    echo "[dry-run] Would execute claude with prompt (${#PROMPT} chars)"
    echo "[dry-run] First 200 chars: ${PROMPT:0:200}..."
    echo ""
    continue
  fi

  # Execute with retries
  attempt=0
  success=false

  while [[ $attempt -lt $MAX_RETRIES ]]; do
    ((attempt++))
    echo "Attempt $attempt/$MAX_RETRIES..."

    LOG_FILE="$LOG_DIR/phase-${num}-attempt-${attempt}.log"

    if claude -p "$PROMPT" \
        --model "$MODEL" \
        --output-format text \
        --verbose 2>&1 | tee "$LOG_FILE"; then
      success=true
      break
    else
      echo "Attempt $attempt failed (exit code $?)"
      if [[ $attempt -lt $MAX_RETRIES ]]; then
        echo "Retrying..."
        sleep 2
      fi
    fi
  done

  if $success; then
    echo "Phase $num: PASS"
    ((PASSED++))
  else
    echo "Phase $num: FAIL after $MAX_RETRIES attempts"
    echo "Log: $LOG_FILE"
    ((FAILED++))
    echo ""
    echo "Stopping. Resume with: $0 $PLAN_FILE --start $num"
    break
  fi

  echo ""
done

# ── summary ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $FAILED
