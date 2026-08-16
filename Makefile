# phyz — development tasks.
#
# Benchmarks live in crates/phyz-bench. They are deliberately NOT part of the
# normal CI gate: benchmark numbers from shared CI runners are noise, and a
# noisy gate is a gate people learn to ignore.

CARGO ?= cargo
BENCH_OUT ?= bench-results.json
BENCH_MD ?= bench-results.md

.PHONY: help build test fmt clippy bench bench-quick bench-gpu bench-criterion bench-mujoco agreement

help:
	@echo "make build           — build the workspace"
	@echo "make test            — run the test suite"
	@echo "make fmt             — format"
	@echo "make clippy          — lint"
	@echo "make bench           — full benchmark suite -> $(BENCH_OUT), $(BENCH_MD)"
	@echo "make bench-quick     — fewer reps; for checking the harness, not for publishing"
	@echo "make bench-gpu       — full suite including the GPU batch sweep"
	@echo "make bench-criterion — criterion regression benches (per-commit comparison)"
	@echo "make bench-mujoco    — instructions for the optional MuJoCo/MJX harness"
	@echo "make agreement       — phyz vs MuJoCo trajectory agreement (needs \`pip install mujoco\`)"

build:
	$(CARGO) build --workspace

test:
	$(CARGO) test --workspace

fmt:
	$(CARGO) fmt --all

clippy:
	$(CARGO) clippy --workspace --all-targets

# The published run. Release profile is mandatory; the harness refuses to be
# quiet about a debug build.
bench:
	$(CARGO) run --release -p phyz-bench -- --json $(BENCH_OUT) --markdown $(BENCH_MD)

bench-quick:
	$(CARGO) run --release -p phyz-bench -- --quick --json $(BENCH_OUT) --markdown $(BENCH_MD)

# Requires a working wgpu adapter. Reports `skipped` with a reason rather than
# failing when there isn't one.
bench-gpu:
	$(CARGO) run --release -p phyz-bench --features gpu -- --json $(BENCH_OUT) --markdown $(BENCH_MD)

bench-criterion:
	$(CARGO) bench -p phyz-bench

# Cross-engine agreement: does phyz compute the same trajectory as MuJoCo?
# Separate from the timing comparison, and asking a different question.
agreement:
	$(CARGO) build --release -p phyz-bench --bin phyz-traj
	python3 crates/phyz-bench/python/mujoco_agreement.py \
	  --json agreement-results.json --markdown agreement-results.md

bench-mujoco:
	@echo "The MuJoCo / MJX comparison is a separate Python harness, kept out of the"
	@echo "Rust build on purpose. See crates/phyz-bench/python/README.md:"
	@echo ""
	@echo "  cd crates/phyz-bench/python"
	@echo "  python3 -m venv .venv && . .venv/bin/activate"
	@echo "  pip install -r requirements.txt"
	@echo "  python mujoco_bench.py --json ../../../mujoco-results.json"
