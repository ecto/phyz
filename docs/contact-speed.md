# contact-speed — where the K1 step's time goes, and taking it back

**Lane:** `contact-speed` (2026-09-11), on `contact-audit` (phyz PR #104).
The audit's §speed: phyz steps a walking K1 in ~100 us, MuJoCo a standing K1
in 22 us, and the solve is 1.1 % of phyz's step — detection and assembly are
the rest. Cam: *"even if it's working i think they could be 10x better."*
This lane owns collision **detection** and contact **assembly**; the solver's
step (rigid impact branch, friction projection) is `contact-fixes`' and is not
touched here.

Rows: `docs/contact-speed/*.jsonl`. Harness:
`crates/phyz/examples/contact_speed.rs`. MuJoCo mirror:
`docs/contact-speed/mj_scenes.py` (run with ipse's audit venv,
`.venv-mj`, MuJoCo 3.13.0). Exactness gate:
`crates/phyz/tests/contact_speed_exact.rs`.

## Method

**The step** is `Simulator::step_with_contacts`'s, hand-rolled the way the
audit's harness is so each stage can be timed: FK, ground detection,
body-body detection, ABA (+ free-joint strip), assembly, warm start, solve,
velocity update, turn + integrate. (`Simulator` also re-runs FK at the end of
the step; the audit's harness and ipse's loops do not.)

**Scenes.**

| scene | what | steps |
|---|---|---|
| `k1u_stance` | the K1 as ipse's `StandingRig` builds it — URDF primitives, the vendor MJCF's foot pads and armature, STL-fitted boxes for the 6 mesh-only links (20 shapes, nv 28) — holding the rig's pose under vendor PD | 2000 |
| `k1u_single` | same, left leg lifted from 0.3 s (scripted hip/knee/ankle) | 1000 |
| `k1u_step` | same, alternating leg lifts at 1.25 Hz from 0.3 s | 3000 |
| `k1m_*` | the vendor MJCF as phyz-mjcf loads it (MuJoCo's own 20 geoms, 6 as `Mesh`) | as above |
| `box_rest`, `box_stack` | the audit's analytic cube at rest; a two-cube stack | 2000 |

The scripted scenes are not balanced walking; they are the contact patterns
(double support, single support, touchdown/liftoff churn) at the K1's real
shape count. Whether the robot stays up is reported (`trunk_z_end`), not
required.

**MuJoCo** runs the same scenes on the vendor `K1_22dof.xml` with the audit's
settings (elliptic cone, Newton, tol 1e-10, Euler, phyz's floor material and
solref/solimp), native PD servos with the same gains and force limits, and
the same scripted targets refreshed every 10 steps (the C loop runs the
chunk). Note MuJoCo's K1 collides **the same 20 geoms** ipse's rig does —
the 6 mesh links as convex hulls where ipse uses fitted boxes — so the shape
count is like for like.

**Timing is thread CPU time** (`CLOCK_THREAD_CPUTIME_ID`; Python
`time.thread_time`), min and median over repeated identical runs. The bench
machine is shared with other lanes (load average 60–140 on 16 cores during
this work); wall-clock numbers moved 4x run to run and are not used. CPU time
still sees cache and memory-bandwidth contention, so treat single rows as
±20 % and compare min to min.

**Allocations** are counted with a counting global allocator in the bench,
per stage.

**Anatomy.** Every 1/20th of a scene the harness freezes the state and times
the insides of detection and assembly by re-running their public pieces 50
times each: placed shapes, AABBs, sweep-and-prune, the pair filter,
narrowphase per shape-pair type (and which pairs ever produce a contact),
ground candidates; FK, CRBA, the mass inverse, Jacobians, `M^-1 J^T`, the
Delassus product. These are replicas of private code paths, so they are
attributions, not the stage totals — the stage totals come from the step.

**Exactness.** Every row carries `state_hash`, FNV-1a over the final `q`,`v`
bits. `contact_speed_exact.rs` pins the hash of every state along a mixed
shape pile (1500 steps) and the K1 stance/step (400/1500 steps) through
`Simulator::step_with_contacts`, recorded on the lane's base before any
change. A pure-speed change must keep all of them.

## Baseline (row 0, `r0_baseline.jsonl`, `mujoco.jsonl`)

Thread CPU µs/step, min of 9 runs (median in the rows).

| scene | phyz | MuJoCo 3.13 | phyz / MuJoCo |
|---|---|---|---|
| stance | 144.8 | 22.7 | 6.4x |
| single | 124.0 | 22.2 | 5.6x |
| step | 103.4 | 23.2 | 4.5x |

`k1u_stance` by stage (µs / allocations per step): FK 1.4 / 3, ground
detection 2.5 / 25, **body-body detection 50.1 / 573**, ABA 12.8 / 80,
**assembly 60.3 / 138**, warm start 0.5, solve 25.3 / 12, velocity update
1.4, integrate 0.8. **835 heap allocations per step.**

(The audit's 1.1 % solve share was the sim2 walking loop, whose solves are
cheaper than this stance's ~21 iterations; here the solve is 17 %. Detection
plus assembly is 76 % either way.)

### The waste, ranked

1. **Separated curved pairs iterated to convergence.** Broadphase passes
   ~11 shape pairs per stance step; the pair filter leaves 6, and all 6 are
   box-cylinder: each hand box against its own side's hip roll, hip yaw and
   shank cylinders. Their AABBs overlap because the arms hang beside the hips.
   None ever produces a contact in any scene. Each costs ~7.3 µs and ~95 heap
   allocations: GJK on a cylinder converges linearly to its 1e-10 relative
   tolerance, and its simplex was a `Vec` rebuilt on every Voronoi
   reduction. **44 of the 50 µs of body detection.**
2. **Dense assembly over all 28 DOFs.** A foot-corner Jacobian is nonzero on
   12 columns (the chain trunk→foot), but `M^-1 J^T`, `J M^-1 J^T` and
   `J v` summed over all 28: ~27 µs at stance.
3. **The dense mass inverse**: Gauss-Jordan on the 28x28 `M`, ~14 µs, every
   step, plus CRBA ~10 µs.
4. **Allocations**: 835 per step, 573 of them in (1).
5. Not waste, for the record: broadphase itself (sweep-and-prune over 20
   AABBs, 1.9 µs), the pair filter (0.04 µs) and ground detection (2.0 µs)
   are cheap; the pair filter already runs once per pair after broadphase and
   a static pair mask would save < 0.1 µs. The warm-start cache is 0.5 µs.

`k1m_*` (the vendor MJCF in phyz, mesh hulls as `Mesh`) are 1.7–4x slower than
`k1u_*` and **diverge to NaN in every scene** (identical `state_hash`
`7f7e827ebb41dc65`, `trunk_z_end` NaN): phyz-mjcf's K1 is not a usable
model today. Its cost is `placed_shapes` cloning every mesh's vertex list
each step (~160 µs) and cylinder-vs-mesh GJK (~130 µs). Reported, not fixed:
no consumer steps this model, and ipse's rig fits boxes to those meshes.

## Changes

| row | change | stance | single | step | exact |
|---|---|---|---|---|---|
| 0 | baseline | 144.8 | 124.0 | 103.4 | — |
| 1 | GJK stops at the margin; stack simplex | 102.9 (1.41x) | 90.5 (1.37x) | 68.4 (1.51x) | yes, every scene |
| 2 | assembly skips structurally zero Jacobian columns | 89.7 (1.15x) | 70.1 (1.29x) | 61.5 (1.11x) | yes, every scene |
| 3 | inverse on slices + flat temporaries | **slower**, 0.89–0.97x (A/B) | | | yes — not applied |
| 4 | CRBA `Sᵀf` on the stack | 1.10x (A/B) | 1.08x (A/B) | 1.08x (A/B) | yes, every scene |

**Row 1** (`r1_gjk_cutoff.jsonl`). `gjk_rot_until(cutoff)` returns `None`
once GJK's own lower bound `v·w/|v|` reaches the cutoff;
`contact_manifold_within` passes the margin, where the old code's
`distance >= margin` arm refused the pair anyway. Public `gjk_rot` is
`gjk_rot_until(∞)`. The simplex is a `[Vec3; 4]` with the same point order and
arithmetic. Body detection 50.1 → 5.1 µs, allocations 835 → 282 per step.
*Exactness argument:* the lower bound never exceeds the true distance, and the
old path's final `|v|` never falls below it, so a pair the cutoff drops is one
the old path dropped. The only gap is a true distance within ~1 ulp of the
margin, where the two float estimates could straddle it; no scene hits it
(every `state_hash` unchanged).

**Row 2** (`r2_sparse_cols.jsonl`). `M^-1 J^T`, the Delassus product, `J v_free`
and the pre-step normal speed sum over each contact's nonzero Jacobian columns
only, in the same increasing order. Each dropped term is `finite × 0 = ±0`,
which cannot change a running sum that starts at `+0` under round-to-nearest,
so the sums are bit-identical; any non-finite input keeps the dense path so
NaN propagation is unchanged. Assembly 61.2 → 48.9 µs at stance. (The anatomy
columns `asm.minv_jt`/`asm.delassus` in this and later rows time replicas of
the *old* dense loops; they are attributions of the baseline, not of the new
code.)

### Row 3 — a negative, reverted (`r3_invert.jsonl`, `ab_r2_r3_r4.txt`)

Two exact rewrites inside `assemble`: the Gauss-Jordan mass inverse on
bounds-check-free slices, skipping `a`'s already-eliminated block (exact
`+0`s, and `a` is discarded), and the per-contact temporaries (`cols`,
`M^-1 J^T`) flattened into single buffers. Every `state_hash` unchanged.

**It is slower.** The benched row said stance 1.03x, single 0.86x, step 0.90x —
mixed, and rows benched minutes apart swing with the other lanes' load, so it
was re-measured as an interleaved A/B (`docs/contact-speed/ab.py`: the
binaries alternate, 8 rounds, min and median of per-round mins):

| k1u | r2 | r3 | r2 → r3 |
|---|---|---|---|
| stance | 78.1 / 79.7 | 80.7 / 84.6 | 0.967x / 0.942x |
| single | 70.8 / 75.8 | 75.5 / 80.5 | 0.938x / 0.941x |
| step | 56.8 / 60.8 | 64.1 / 67.6 | 0.886x / 0.899x |

A 3–11 % regression, exact or not. Row 3 was never committed; the rows and
the A/B stay in the directory as the record. (Every row from here on is
judged by an interleaved A/B against its parent, not by a standalone bench.)

### Row 4 — CRBA's `Sᵀf` on the stack (`ab_r2_crba_inv.txt`)

CRBA walks every body up to the root to fill the mass matrix's off-diagonal
column. When the ancestor is a multi-DOF joint — the K1's free base, which is
every body's last ancestor — it formed `Sᵀf` as
`motion_subspace_matrix().transpose().mul_vec(&sv_to_dvec(f))`: four heap
allocations and a 6x6 product, 22 times a step. `subspace_t_force` computes
the same thing on the stack with the same entries (`S = I₆` for a free joint,
`[I₃; 0]` for a ball) and tang-la's own `j`-outer / `i`-inner accumulation
from the same `+0` (tang-la is built without `accelerate`, so `mul_vec` *is*
that loop). Other joint types keep the allocating path. Exact by construction.

Interleaved A/B, 8 rounds (min / median of per-round mins, µs):

| k1u | r2 | r2 + CRBA (row 4) | r2 + CRBA + row 3's inverse |
|---|---|---|---|
| stance | 79.0 / 82.6 | **71.6 / 77.7** (1.10x) | 81.9 / 83.6 |
| single | 71.7 / 74.4 | **66.5 / 69.6** (1.08x) | 73.0 / 76.9 |
| step | 55.6 / 60.9 | **51.7 / 54.9** (1.08x) | 60.8 / 63.6 |

Same `state_hash` in all three arms. The third arm isolates row 3's
regression: it is the inverse rewrite, not the flattened temporaries — a
slice-and-zip Gauss-Jordan that skips half of `a`'s updates is slower than
the indexed original on a 28x28 matrix. Not investigated further; the dense
inverse stays as it was.

## What `phyz-gpu` inherits

Nothing directly: every change on this lane is **CPU-only**. `phyz-gpu` runs
its own WGSL pipeline — its own ground-contact pass (plane only; its
`contact_pipeline` is deliberately not general body-body contact), its own
Delassus assembly and contact solve, and `invert_small` for the per-joint `D`
blocks in its ABA. It never calls `phyz_collision`'s GJK/manifold,
`phyz_contact::assemble` or `phyz_rigid::crba` on the host.

| change | GPU |
|---|---|
| GJK stops at the margin; stack simplex | not applicable — the GPU has no body-body narrowphase. If it ever gains one, the margin cutoff is the same exact early-out and matters more there (divergent iteration counts cost a whole warp). |
| assembly skips structurally zero Jacobian columns | not inherited. The same identity (a contact touches only its chain's DOFs, 12 of 28 on the K1) applies to the GPU's own Delassus build and is the obvious next thing to try there. |
| CRBA `Sᵀf` on the stack | not inherited (allocation removal has no GPU analogue). |

What the GPU does get is a faster CPU reference: its parity tests that step
the CPU contact path to compare against (`contact_impulse_parity`,
`multi_collision_parity`, `heightfield`) run on the code changed here.
