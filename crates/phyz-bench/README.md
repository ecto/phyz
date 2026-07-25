# phyz-bench

Benchmark harness for phyz. Not published to crates.io (`publish = false`).

Results live in [`BENCHMARKS.md`](../../BENCHMARKS.md) at the workspace root;
this file documents the harness itself.

## Running

```bash
make bench            # full suite -> bench-results.json + bench-results.md
make bench-quick      # fewer reps; for checking the harness, not for publishing
make bench-gpu        # includes the GPU batch sweep (needs a wgpu adapter)
make bench-criterion  # criterion regression benches
```

Or directly:

```bash
cargo run --release -p phyz-bench -- --suite single-sim --json out.json
```

Flags: `--suite NAME` (repeatable; `single-sim`, `energy`, `gradient`, `gpu`,
`rapier`), `--quick`, `--json PATH`, `--markdown PATH`.

The MuJoCo / MJX comparison is a separate Python harness — see
[`python/README.md`](python/README.md).

## Layout

| Path | What it is |
|---|---|
| `src/settings.rs` | The physics settings every engine is held to. Single source of truth. |
| `src/scenes.rs` | Scene definitions and the phyz stepping loop. |
| `src/timing.rs` | Warmup + repetition + median timing, identical for every engine. |
| `src/meta.rs` | Hardware, OS, toolchain, library versions. |
| `src/report.rs` | JSON schema and Markdown renderer. |
| `src/suites/` | The five suites. |
| `benches/single_sim.rs` | Criterion, for per-commit regression detection only. |
| `python/` | Optional MuJoCo / MJX harness. |

## Two measurement tools, on purpose

**The `phyz-bench` binary** produces the published numbers. It measures every
engine through one code path — same warmup, same repetition count, same median
statistic — and emits JSON. Cross-library comparison requires that.

**Criterion** (`cargo bench -p phyz-bench`) is for the other job: detecting a
regression between two commits on one machine, with confidence intervals. Its
output does not feed the published document.

## Rules the harness enforces

These are mechanical, not aspirational:

- **A debug build says so.** The profile is recorded in every report, and the
  binary prints a warning that debug numbers must not be published.
- **A noisy measurement is flagged.** Any row whose spread `(max−min)/median`
  exceeds 25% gets a ⚠ in the table and a per-suite warning banner. Benchmarks
  on a shared machine are noise, and noise that looks like a result is worse
  than no result.
- **A diverged simulation cannot be reported as a fast one.** Every timed
  scene is checked for a finite final state. A blown-up sim does *less* work,
  so instability would otherwise read as a speedup.
- **Every repetition starts from the same state.** Scenes are reset inside the
  timed closure; Rapier's world is rebuilt per repetition, because Rapier puts
  settled islands to sleep and a reused world would flatter it.
- **Settings travel with the number.** Every record carries the full
  `Settings` block, including the `caveats` list naming what could not be held
  equal.
- **Nothing is silently dropped.** A suite that cannot run emits `skipped` with
  a reason. A GPU batch size that fails to allocate is recorded, and the sweep
  stops rather than quietly truncating.

## Adding a scene

Add a variant to `scenes::Scene`, then handle it in `build_model`,
`initial_state`, `description`, and — if it has a Rapier equivalent —
`suites::rapier::build`. `suites::standard_scenes()` controls what appears in
the single-sim and comparison tables.

If a scene has no honest equivalent in another engine, return an `Err` from the
Rapier builder explaining why. That reason is published alongside the phyz row,
which is better than an unequal comparison or a silent gap.
