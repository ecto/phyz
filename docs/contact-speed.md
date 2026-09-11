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
