# Flat-contact adjoint blow-up on K1: root cause

**Symptom.** ipse `fd_check --flat` (K1 squat, box feet, friction 0.8, timeconst 0.02,
tol 1e-10, dt 2 ms, armature on, `ipse_data` fixture joints): `convex_adjoint_gradient`
gives |dJ/dθ| 1e10–1e17 on J ≈ 0.05, and central FD is unstable too: sign flips, and it
grows ~10x per 10x smaller step. Sphere-foot fd_check passes.

**Root cause: not contact.** The fixture gives every joint `friction_loss = 0.05`.
`Joint::passive_force` applies that as `-fl·tanh(qd/1e-3)`, evaluated *explicitly*.
Near rest that acts as a viscous damper with `c = fl/1e-3 = 50 N·m·s`. On the light
joints (arms, hands, head: armature 1e-3–2e-3) the explicit step gain is
`1 − dt·c/M ≈ −70`. So the discrete step map *expands* around rest. `tanh` saturation
bounds the motion, and the coupled chain rattles chaotically at |qd| ~ 1e-3.
Measured evidence:

| measurement (window 0, 125 steps) | explicit (shipped) | `friction_loss = 0` | implicit (fix) |
|---|---|---|---|
| ‖Δv‖ after 125 steps from a 1e-9 rel. trunk-mass change | 2.8e-2 (saturated; ~1.5x/step from t≈15) | 1.7e-9 | 1.5e-9 |
| same, ground moved 5 m down (**no contacts**) | 4.4e-2 | – | – |
| adjoint vs FD(h=1e-8), horizon 10 / 40 / 125 steps | match / 1e3 vs 9e2 / 6e11 vs noise | match to 6 digits at every horizon | match to 6 digits at every horizon |

- The adjoint is *correct*. It matches FD wherever FD can resolve the derivative (H ≤ 20,
  and H = 40 at h = 1e-8). The backward pass grows by ~1.37x per step through 125 steps
  because the true linearization does. FD fails because the perturbation saturates, so
  ΔJ ≈ const ≈ 1e-4 and FD ≈ ΔJ/h, which is the observed 10x growth per decade of h.
- Contact is a bystander. All 8 corner contacts stay loaded and sticking (|f_t|/f_n ≲ 0.03).
  Every step converges at residual 1e-13–1e-17. There is no active-set switching
  (rules out (a)). There are no solver stalls (rules out (c)). The divergent eigenvector
  sits in Head_2, left_hand, Arm_2/3, not in the feet. The growth persists in free fall
  with zero contacts.
- Sphere feet pass because the sphere rig is `StandingRig` from the URDF, with no
  `friction_loss`. The flat rig is the `ipse_data` fixture, with 0.05 on every joint.
  Foot shape was confounded with the friction model.

**Secondary finding (contact, (b)-type, not the K1 cause).** A box face *pressed* into the
plane (20 N on a 1 kg box) does not converge at tol 1e-10. It stalls at ~1e-8, even with
the full 1000-iteration budget. The four corners give 8 tangential rows on 3 in-plane DOFs,
and that 5-dim null space is regularized only by `regularization = 1e-6`. The adjoint then
refuses the window, or is 0.4% off at tol 1e-7. Setting `regularization = 1e-4` (existing
knob) converges it, and the adjoint matches FD to 4e-7. A resting face, a sliding face, and
the K1 stance all converge at the default.

**Fix (opt-in, named; default physics unchanged).**
`Joint::friction_loss_integration: FrictionLossIntegration::{Explicit (default), Implicit}`.
`Model::set_friction_loss_integration` sets it on all joints. `Implicit` keeps the same
force law. It adds `dt·c(qd)` to the joint's ABA diagonal, as `dt·damping` already is,
where `c = fl·tanh(qd/s)/qd` is the secant coefficient: smooth, and monotone with no
overshoot. It also adds the same term to the contact solver's mass matrix
(`phyz_contact::assemble`), so contact impulses see `M + dt·C`. The generic (reverse-mode)
mirrors in `phyz-diff` match. Friction-only update is `v' = v/(1 + dt·c/M)`, stable for any
dt. The cost: near rest, joints creep at the tanh-regularized rate (~7e-4 rad/s here)
instead of chattering. CPU only; `phyz-gpu` stays explicit.

**Result** (ipse `fd_check --flat` with the opt-in, 2 windows x 26 params):
24/26 params per window agree at 1e-7–1e-4. The four ankle-cross/foot-link masses
(|g| ~ 1e-6, 4–7 decades below the window's largest gradient) are at 1e-3–7e-3 with the
default lane-FD smooth blocks. With `PHYZ_SMOOTH_ADJOINT=1` (reverse-mode smooth blocks)
the worst best-step rel err is **4.1e-5: PASS**. Without the opt-in it is still 1.0: FAIL.
Regression: `phyz-diff/tests/flat_contact_adjoint.rs`.
