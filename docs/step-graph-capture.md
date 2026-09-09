# The step's host bill — a launch census, and what capture can still buy

*Lane: `step-graph-capture`. Measured 2026-09-09.*

## Why this was opened, and what it found instead

The lane was opened to add CUDA Graph capture to the batch-sim step, because
ipse's device-resident collector is launch-bound: on the mount stage 16x the
worlds cost 1.6x the time, and the host's `issue` time is 175 s -> 285 s per
collect from 256 to 4096 worlds, against a 5 % readback and a 0.03 % host
post-pass (ipse `docs/mount-gpu-stage.md` §6).

**phyz already has capture.** It landed in #80 and was extended in #90:
`BatchSim` captures a one-step span for `step()` and a whole control period
for `step_many(n)`, keyed on `(steps, sweeps, epoch)` so a changed sweep count
or a re-allocated buffer discards the recording, with `PHYZ_CUDA_GRAPHS=0` and
`set_graphs_enabled(false)` as the uncaptured reference. Bit-identity of a
replay against that reference is already pinned by `suite_graph_replay` in
`crates/phyz-gpu/tests/cuda_vs_cpu.rs`, including the stale-capture case.

So there was nothing to build here, and the useful question became the one
task 1 of the lane asked for anyway: **how many host calls does a step
actually cost, and which of them can a capture remove?** That is the census
below. It also relocates the blocker, which is the finding.

## The census

`cuda::census` counts what `BatchSim` issues — kernel launches, graph replays,
captures, transfers, syncs — as a thread-local tally. It sits above the
backends because `BatchSim` issues the same sequence to all of them, so a
census taken on the `cuda-host` mirror is the count the CUDA path issues, and
it needs no GPU. `examples/launch_census` prints the table.

Ant plant (9 bodies, nv 14), impulse contact at the default 16 sweeps, PD on
every revolute DOF, `control_every = 20`. Kernel launches:

| issue mode | one `step()` | period as 20x `step()` | period as `step_many(20)` |
|---|---|---|---|
| `Fused` (default) | **1** | **20** | **1** |
| `Fission` | 35 | 700 | 700 |
| `Unfused` | 35 | 700 | 700 |

`3 + 2 * sweeps = 35` is the unfused sequence: PD, a leading ABA, sixteen
[contact, ABA] pairs, integrate. A `readback_states()` costs 2 downloads and
**1 sync** on top, per readback.

**Every row is identical at 512 and at 4096 worlds.** That is the launch-bound
diagnosis stated exactly: widening the batch adds threads to a launch and
never adds a launch, so no amount of extra width amortises the host bill.
`tests/launch_census.rs` pins all three facts.

## Where the blocker actually is: the call shape, not the capture

Read the fused row again. In the default configuration a whole control period
is **one** launch if the caller asks for the period, and **twenty** if it asks
for twenty steps — same arithmetic, same bits, 20x the host bill. Capture is
not what separates those two numbers; `step_many` is.

And the collector asks for twenty steps. ipse's `GpuCollector::step_period`
loops `for _ in 0..n { sim.step() }` unless `RL_STEP_MANY=1` is set, and it is
**off by default**; `rl_task_gpu.rs` has no such switch at all and always
loops `step()`. So the launch-bound measurement in `mount-gpu-stage.md` was
taken while paying twenty host calls per control period for physics that phyz
will issue in one.

That reframes the ceiling. The three exits that doc names — "fuse / graph-
capture / several control periods per launch" — are all already implemented
in phyz. What is not done is the caller taking them:

1. **Default `RL_STEP_MANY=1`** in ipse's `GpuCollector`, and give
   `rl_task_gpu.rs` the same path. Projected: the per-period launch bill goes
   20 -> 1 in fused mode, a **20x cut in launches**, with a capture on top
   making the replay itself one `cudaGraphLaunch`.
2. **The remaining floor is the sync.** One `readback_states()` per control
   period is 1 blocking sync + 2 downloads, and a sync drains the pipeline, so
   it costs far more than its count. Once launches are 1 per period, the sync
   is the next thing in the way, and it is a collector-side question (how
   often the host must see states) rather than a phyz one.

**None of this is a speedup measured yet.** It is a count, and a count of
launches is not a duration. The multiplier is bounded above by the fraction of
the collect that is `issue` (98 % by the ipse split, once readback and the
post-pass are removed), but what a 20x launch cut actually returns has to be
timed on a device.

## What is capturable, and what is not

Nothing in the step depends on a host-side decision. Contact counts and the
active set are resolved **on device**, inside the kernels: the sweep count is
a fixed `contact_sweeps` baked into the launch, every argument that changes
between steps already lives in a device buffer, and the scalars (world count,
DOF widths, `dt`, gravity) do not change between steps. That is why the whole
sequence captures, and why the capture key only has to guard `(steps, sweeps,
epoch)`. **100 % of the step's launches are capturable**, and they already are.
There is no refactor to name here — which is itself worth recording, since the
lane brief expected a host-side active set to be the obstacle and it is not.

## Timing

Not taken. mew's 3090 was running four walk arms for the whole session, and a
timing run against a contended GPU measures the other process's kernels. A
detached runner (`~/lanes/run-step-graph-capture.sh` on mew) waits on
`/tmp/wsarms/pids` and then takes the census and `cuda_graph_bench` at
512/2048/4096/16384 worlds into `/tmp/phyzgraph-gpu/`. The graph rows of the
census table — the ones that need a device that can capture — are blank until
it runs.

## Rows

`docs/step-graph-capture-rows.txt`.
