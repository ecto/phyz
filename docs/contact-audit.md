# contact-audit — phyz contacts against closed forms, against 21a33f91, and against MuJoCo

**Lane:** `phyz-contact-audit` (2026-09-11). Cam: *"even if it's working i think
they could be 10x better. i think we could verify with mujoco?"*
Consumer rows (K1 stance / single support / step / policy) live in ipse
`docs/phyz-contact-audit.md`; this file carries the engine side.

Rows: `docs/contact-audit/*.jsonl`. Harness: `crates/phyz/examples/contact_audit.rs`
(the `Simulator::step_with_contacts` step, hand-rolled so impulses and solver
iterations are visible; the two free-joint turn calls are the only rev shim).
MuJoCo side: ipse `scripts/contact_audit/mj_analytic.py` (MuJoCo 3.13.0, same
bodies, default solref/solimp = phyz's defaults, elliptic cone, condim 3,
Newton, tol 1e-10). "old" = 21a33f91, "new" = 68fc142 (ipse's pin since #426).

## 1. The analytic suite

0.2 m cube, 1 kg, mu 0.5 unless noted. dt 1 ms shown; 0.25 and 0.5 ms are in
the rows and change nothing below except where marked.

| scenario | exact answer | phyz old | phyz new | MuJoCo |
|---|---|---|---|---|
| (a) rest 10 s: per-corner Fn | mg/4, err 0 | 2.3e-13 N | 1.7e-13 N | 7.6e-13 N |
| (a) tangential force | 0 | 8.7e-7 N | 8.7e-7 N | 1e-13 N |
| (a) sink / drift xy | 0 / 0 | 49.5 um / 1e-14 | 49.5 um / 1e-14 | 107.8 um / 4e-15 |
| (a) solver iters mean (max) | — | 2.09 (18) | 2.09 (18) | 0.10 (1) |
| (b) incline 20 deg, 2 s | sticks | 17 um | 17 um | **2.0 mm creep** |
| (b) incline 26 deg (tan < mu) | **sticks** | **slides 0.1726 m/s2** | **slides 0.1726** | sticks (3.4 mm creep) |
| (b) incline 27.5 deg | 0.1790 m/s2 | **0.456** | **0.456** | 0.1787 |
| (b) incline 35 deg | 1.6088 m/s2 | **1.865 (+15.9 %)** | **1.865** | 1.595 (-0.9 %) |
| (c) sphere e=0.5, 0.5 m: sqrt(h1/h0) | 0.5 | 0.417 (1 ms), 0.327 (0.5), **0.040 (0.25)** | **0.498 / 0.499 / 0.499** | n/a (no e) |
| (c) sphere e=0, max / rest pen | small | 0.87 mm / 49.5 um | 0.71 mm / 49.5 um | 21.5 mm / 367 um |
| (c) box drop 0.3 m, max pen / settle | — | 0.64 mm / 0.363 s | 0.46 mm / 0.347 s | 16.0 mm / 0.446 s |
| (d) F = 2 N at COM: front / back corner | 2.9525 / 1.9525 | 2.953 / 1.952 | 2.953 / 1.952 | 2.953 / 1.952 |
| (d) tall box released at 25 / 28 deg (crit 26.57) | stays / topples | stays / topples | stays / topples | stays / topples |
| (e) yaw torque 0.5 N m (4-corner cap 0.694) | sticks | 6e-9 rad/s | 6e-9 rad/s | 0.0065 rad/s creep |
| (e) yaw torque 0.9 N m, wz at 1 s | 30.95 rad/s | 30.95 | 30.95 | 32.26 (1 ms), 29.77 (0.5) |
| (f) 2-box stack: ground / interface total | 29.43 / 19.62 N | exact / exact | exact / exact | exact / exact |
| (f) top-box drift 10 s | 0 | 9.6e-12 m | 2.9e-12 m | 2.7e-16 m |

Readings, each backed by the row above it:

1. **The bump broke no analytic scenario and fixed one.** Restitution went
   from dt-dependent and badly short (0.04–0.42 of e = 0.5) to 0.498–0.499
   at every dt. Rest, stack, load shift, tipping and torsion are identical
   between revs to the last printed digit.
2. **Sliding friction is wrong at both revs, and the error is ~15 % of the
   sliding acceleration.** The block slides at 26 deg, where Coulomb says it
   sticks, and at 35 deg it accelerates 15.9 % too fast. The effective mu is
   0.468 = 0.936 mu, and it doesn't depend on dt, angle or solver config
   (`CA_CFG=nonewton|block|perbody|gpu|reg` all reproduce it; `compat` sticks
   at 26 deg only because it creeps). §3 has the mechanism.
3. MuJoCo pays for its soft default in penetration (16–21 mm on a 0.3–0.5 m
   drop against phyz's < 1 mm) and in creep (2 mm on a 20-deg slope,
   0.0065 rad/s under a sub-capacity torque). phyz holds stiction exactly.
   Neither is "wrong": both are the documented models (phyz regularizes the
   normal row only, `ContactSolverConfig::mujoco_compat`).

## 2. What the bump changed (21a33f91..68fc142, read-only)

Four forward-dynamics changes land in `a445646` (#99) and `2206349` (#95):

- `assemble.rs`: restitution reads the **pre-step** approach speed
  (`J v`, not the free velocity with `g dt` in it), and every row whose approach
  exceeds `restitution_threshold` (0.05 m/s) is made an **impact row**:
  impedance to `IMPACT_IMPEDANCE = 0.999`, stabilization bias scaled by
  `1 - impact`. **The impact weight does not look at the material's
  restitution.** An e = 0 foot landing at > 0.1 m/s gets a rigid, bias-free row.
- `rigid/integrate.rs`: the exact free-joint turn (`strip_free_joint_coriolis`
  + `rotate_free_joint_velocities`), and `integrate_configuration` now rotates
  the displacement by the end-of-step orientation.
- `manifold.rs`: a flat face's witness point is the curved side's witness
  projected onto it (curved-vs-flat pairs only; box-on-plane and the K1 pad are
  untouched).

Attribution on the K1, from the ipse rows (box pads, e = 0):

| | 21a33f91 | 68fc142 |
|---|---|---|
| stance settle, first 0.25 s: least-loaded corner | 0.883 N | **0.0597 N** (the triage red, reproduced) |
| stance settle: non-converged steps / Fz peak | 7 / 1005 N | **15** / **2874 N** |
| stance after 1 s: corners, Fz/mg, drift | identical to 4 digits | identical |
| `WalkGait` 12 s: peak single-corner force at heel strike | 3217 N | **4541 N** (+41 %) |

Every one of those rows is an e = 0 impact: the settle drop and the heel
strike. The stance once it's settled is bit-near identical. So the rigid
impact row is what moved the triage's settle-window reds. §4's prototype
tests that directly.

## 3. The friction error: a radial clamp of an anisotropic block

Per-corner world friction on the 35-deg slide (`CA_CORNERS=1`): every corner
sits exactly on its cone (`|F_t| = 0.5 F_n`) but is rotated **±20.56 deg**
off the slide direction, symmetric in y, so the corners pinch against each
other. cos 20.56 deg = 0.936, the whole missing fraction. Total F_n = mg cos
theta exactly.

Mechanism (`convex.rs`, the staged update in `sweep`): the tangential step is
the unconstrained 2x2 solve `t = -A_tt^-1 r` followed by a **radial** clamp
to `|t| <= mu f_n`. That is the exact block minimizer only when `A_tt` is
isotropic. A box corner's tangential Delassus block has lever-arm coupling
(off-diagonal terms whose sign flips corner to corner), so the radial clamp
lands on a fixed point that is not max-dissipation. The Newton stage converges
to the same staged fixed point, which is why no config escapes it.

**The exact step** is the disc-constrained block QP:
`t = -(A_tt + k I)^-1 r`, with `k >= 0` chosen so `|t| = mu f_n`, a 1-D
secular equation (`PHYZ_TR_CLAMP=1` prototype, forward path of the PGS
sweep only):

| incline | analytic | radial clamp (shipped) | trust-region clamp |
|---|---|---|---|
| 26 deg | 0 (sticks) | 0.1726 | **0 (sticks)** |
| 27.5 deg | 0.17897 | 0.45604 | **0.17897** |
| 35 deg | 1.60884 | 1.86471 | **1.60884** |

Its cost, measured: on the stuck 26-deg block the default config goes from
0 to 206/2000 non-converged steps (statically indeterminate; the 20-deg row
was already 2000/2000 at both revs).

## 4. Ranked fixes (evidence, size, status)

1. **Impact rows only when the material can bounce** (`assemble.rs`: impact
   weight 0 when `restitution <= 0`; the `phyz-diff` adjoint mirror of
   `with_impact` needs the same line). Evidence: every e = 0 drop in the suite
   reverts bit-for-bit to 21a33f91 under `PHYZ_IMPACT_NEEDS_E=1`, and e = 0.5
   keeps 0.498–0.499 (`drops_impact_needs_e.jsonl`). The K1 rows under the
   prototype are §5. Size: ~5 lines + mirror.
2. **Trust-region friction step** (§3). Evidence: exact on the incline. Size:
   **not small.** The differential branch in `sweep`, the transpose sweep, the
   Newton `staged_residual`/regime linearization and
   `contact_solve_differential(_transpose)` all differentiate the radial
   clamp. Shipping the forward alone would desynchronize the IFT gradient from
   the forward. Left as a ranked, evidenced item; the prototype is env-gated
   and off by default.
3. **Speed is detection and assembly, not the solve.** K1 step, own loop,
   vendor gains, dt 1 ms: 100.7 us/step, of which detect 54 %, assemble 30 %,
   ABA 11 %, **solve 1.1 %**. MuJoCo 3.13 steps the same standing K1 (native PD
   actuators, 8 contacts) in **21.95 us**. The 10x is in
   `find_ground_contacts_model` + `find_contacts` (every collision shape, every
   step) and in the dense `3 x nv` Jacobians and `M^-1` in `assemble`, not in
   the convex solver.
4. **The integrator pair isn't exported to hand-rolled loops.** Since the
   bump, `integrate_configuration` rotates displacement by the end-of-step
   frame. ipse's `run.rs`/`sim2.rs`/`rl_task` loops step with `aba` +
   `assemble` + `integrate_configuration` and never call
   `strip_free_joint_coriolis`/`rotate_free_joint_velocities`, so they run a
   half of the #99 scheme `Simulator` runs. Magnitude: see the ipse doc (not
   yet measured when this was written).

## 5. The prototypes on the K1

Same binary, one phyz tree (68fc142 + the env-gated prototypes), env off / on:

| K1 row | 68fc142 (`=0`) | `PHYZ_IMPACT_NEEDS_E=1` | 21a33f91 |
|---|---|---|---|
| settle 0.25 s: least-loaded corner | 0.0597 N | **0.8834 N** | 0.8834 N |
| settle: non-converged / Fz peak | 15 / 2874 N | **7 / 1005 N** | 7 / 1005 N |
| gait: peak corner at heel strike | 4541 N | **3364 N** | 3217 N |
| gait: ticks with Fz > 2 mg | 136 | 222 | 226 |
| settled stance (after 1 s) | identical | identical | identical |

The prototype returns every e = 0 row the bump moved to the old number and
keeps the bump's restitution fix. That's the evidence for fix 1.

**On the triage's actual reds** (ipse test binaries, same tree, env gates):

| ipse test | neither | `IMPACT_NEEDS_E` | `INTEG_CURRENT` | both |
|---|---|---|---|---|
| topple_forensics x4 | red | green | red | green |
| shac two-step ratio (bar 1..5) | 9.318 | 12.020 | 11.442 | **green** |
| milestone_5 lateral placement | red | red | red | green |
| milestone_5 straight line (pin +1.6310) | +1.5574 | +1.6270 | +1.5522 | +1.6278 |

`PHYZ_INTEG_CURRENT=1` restores 21a33f91's displacement frame in
`integrate_configuration`. It isn't proposed as a fix. #99's frame is the
correct half of an exact scheme whose other half (`strip_free_joint_coriolis`
+ `rotate_free_joint_velocities`) ipse's hand-rolled loops never adopted, so
the consumer-side fix is to adopt the pair. The table's point is attribution:
the bump's K1 reds are the rigid e = 0 impact row **plus** that half-adoption,
and neither alone closes the gradient row.

**What fix 1 costs in this repo:** `phyz/tests/determinism.rs::golden_rollout_hashes`
passes at the default and fails under `PHYZ_IMPACT_NEEDS_E=1`. The goldens
roll out e = 0 impacts, so they re-pin with the change.
