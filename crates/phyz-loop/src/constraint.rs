//! Loop-closure constraints over a `phyz_model::Model` tree, and the linear
//! system they induce at the acceleration level.
//!
//! # Why a constraint at all
//!
//! `phyz_rigid`'s ABA is a *reduced-coordinate* solver over a kinematic tree.
//! A tree has exactly one path from the world to every body, so a closed chain
//! — four-bar, slider-crank, delta robot — has no representation in it at all.
//! The standard fix, and the one taken here, is to cut each loop: model the
//! mechanism as a spanning tree (which ABA can handle) and re-impose the cut
//! joint as an explicit constraint `c(q) = 0` on the tree's coordinates.
//!
//! # Frames and sign conventions — read this before touching the Jacobians
//!
//! Every constraint here is written as **A minus B**:
//!
//! ```text
//! c(q)  = e_A(q) - e_B(q)          (position-level error, world frame)
//! cdot  = J v,        J = J_A - J_B
//! cddot = J a + k,    k = k_A - k_B
//! ```
//!
//! so `J`, the residual and the bias term `k = Jdot v` all carry the same
//! orientation. Getting one of the three backwards is the classic way to ship
//! a constraint that quietly pushes bodies apart instead of together, and it
//! is invisible in a symmetric test case — a four-bar with equal links
//! satisfies a sign-flipped constraint just as well as a correct one. The
//! asymmetric slider-crank test in `tests/slider_crank.rs` exists partly to
//! catch that.
//!
//! `xforms[i]` from [`phyz_rigid::forward_kinematics`] is a **world-to-body**
//! Plücker transform: `pos` is the body origin *in world coordinates* but
//! `rot` maps world coordinates *into* the body frame. So the world position
//! of a body-local anchor is `body_to_world_point`, never `pos + rot * r`.
//!
//! Body velocities and accelerations from
//! [`phyz_rigid::forward_kinematics_acc`] are in each body's **own** frame.
//! They are rotated to world here, because the constraint rows are world-frame
//! rows (that is what [`phyz_rigid::point_jacobian`] produces).

use phyz_math::{DMat, DVec, Mat3, SpatialTransform, SpatialTransformExt, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{body_angular_jacobian, forward_kinematics_acc, point_jacobian};

// The body a constraint endpoint is attached to.
//
// This crate originally defined its own `Attachment` enum, structurally
// identical to `phyz_model::Attachment` — two places arriving at the same
// answer to "a body index, or the world" without knowing about each other.
// Two identical-but-incompatible types would be worse than either, so the
// shared one wins and this is a re-export: `phyz_loop::Attachment` still
// resolves, and it is now the same type the Jacobian and contact APIs take.
//
// `World` is the static inertial frame: it contributes no Jacobian columns and
// no bias, and its anchor point is read directly in world coordinates. A
// grounded mechanism (every four-bar is one) needs it.
pub use phyz_model::Attachment;

/// One endpoint of a loop-closure constraint.
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    /// Which body (or the world) the anchor is rigidly fixed to.
    pub attachment: Attachment,
    /// Anchor position in that body's frame. For [`Attachment::World`] this is
    /// already a world position.
    pub offset: Vec3,
}

impl Anchor {
    /// An anchor at `offset` in body `body`'s frame.
    pub fn body(body: usize, offset: Vec3) -> Self {
        Self {
            attachment: Attachment::Body(body),
            offset,
        }
    }

    /// An anchor at a fixed world point.
    pub fn world(point: Vec3) -> Self {
        Self {
            attachment: Attachment::World,
            offset: point,
        }
    }
}

/// What a loop-closure constraint holds fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// Three rows: the two anchor points coincide. The relative orientation of
    /// the two bodies is free — this is a ball joint closing the loop.
    Point,
    /// Six rows: the anchor points coincide *and* the relative orientation is
    /// held at the value captured when the constraint was built. A rigid weld.
    Weld,
}

/// A loop-closure constraint between two anchors.
///
/// Build with [`LoopConstraint::point`] or [`LoopConstraint::weld`]. The weld
/// constructor needs a state, because "hold the current relative orientation"
/// is the only target that does not require the caller to reason about two
/// body frames at once.
#[derive(Debug, Clone)]
pub struct LoopConstraint {
    /// Name, for diagnostics. Not used in the solve.
    pub name: String,
    /// The "A" endpoint. Residual is measured as A minus B.
    pub anchor_a: Anchor,
    /// The "B" endpoint.
    pub anchor_b: Anchor,
    /// Which rows this constraint contributes.
    pub kind: LoopKind,
    /// For [`LoopKind::Weld`]: the target relative rotation `R_A^T R_B`, where
    /// `R_X` is body X's body-to-world rotation. Unused for
    /// [`LoopKind::Point`], where it is the identity.
    ///
    /// Storing the *relative* rotation rather than two absolute ones is what
    /// makes the constraint invariant to rigidly rotating the whole mechanism.
    pub target_rel_rot: Mat3,
}

impl LoopConstraint {
    /// A 3-row point-coincidence ("ball joint") loop closure.
    pub fn point(name: &str, anchor_a: Anchor, anchor_b: Anchor) -> Self {
        Self {
            name: name.to_string(),
            anchor_a,
            anchor_b,
            kind: LoopKind::Point,
            target_rel_rot: Mat3::identity(),
        }
    }

    /// A 6-row weld loop closure, capturing the relative orientation the two
    /// bodies have in `state` as the target.
    ///
    /// Capturing rather than taking a target as an argument is deliberate: a
    /// weld is nearly always built from an assembled mechanism, and asking the
    /// caller for `R_A^T R_B` invites exactly the frame confusion this module
    /// is trying to avoid. Any *position* error present in `state` is not
    /// captured — the point rows always target coincidence — so a weld built
    /// from a misassembled state will pull the anchors together.
    pub fn weld(
        model: &Model,
        state: &State,
        name: &str,
        anchor_a: Anchor,
        anchor_b: Anchor,
    ) -> Self {
        let (xforms, _) = phyz_rigid::forward_kinematics(model, state);
        let ra = rotation_of(&xforms, anchor_a.attachment);
        let rb = rotation_of(&xforms, anchor_b.attachment);
        Self {
            name: name.to_string(),
            anchor_a,
            anchor_b,
            kind: LoopKind::Weld,
            target_rel_rot: ra.transpose() * rb,
        }
    }

    /// Number of constraint rows this contributes.
    pub fn nrows(&self) -> usize {
        match self.kind {
            LoopKind::Point => 3,
            LoopKind::Weld => 6,
        }
    }
}

/// An ordered set of loop closures.
///
/// Order is part of the contract: it fixes the row order of the assembled
/// system, and therefore the arithmetic the solver performs. A `Vec`, never a
/// map — see the determinism note in the crate docs.
#[derive(Debug, Clone, Default)]
pub struct LoopConstraintSet {
    /// The constraints, in row order.
    pub constraints: Vec<LoopConstraint>,
}

impl LoopConstraintSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a constraint, keeping insertion order.
    pub fn push(&mut self, c: LoopConstraint) -> &mut Self {
        self.constraints.push(c);
        self
    }

    /// Total number of constraint rows.
    pub fn nrows(&self) -> usize {
        self.constraints.iter().map(|c| c.nrows()).sum()
    }

    /// Whether the set holds no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

/// The linearization of a constraint set at one state.
#[derive(Debug, Clone)]
pub struct LoopSystem {
    /// `m x nv` constraint Jacobian `J`, world-frame rows, A-minus-B.
    pub jacobian: DMat,
    /// `m` position-level residual `c(q)`. Zero on a perfectly assembled
    /// mechanism; its growth over a rollout *is* the constraint drift.
    pub position_error: DVec,
    /// `m` velocity-level residual `J v`.
    pub velocity_error: DVec,
    /// `m` bias acceleration `Jdot v` — the constraint-space acceleration the
    /// mechanism has when every joint acceleration is zero. This is the term a
    /// naive implementation forgets, and forgetting it makes every rotating
    /// loop drift at a rate proportional to `omega^2`.
    pub bias: DVec,
}

impl LoopSystem {
    /// Number of constraint rows.
    pub fn nrows(&self) -> usize {
        self.jacobian.nrows()
    }

    /// Largest-magnitude entry of the position residual, in metres (and
    /// radians for weld rows).
    pub fn position_residual_inf(&self) -> f64 {
        inf_norm(&self.position_error)
    }
}

/// Assemble `J`, `c`, `J v` and `Jdot v` for `constraints` at `state`.
///
/// Cost is one [`forward_kinematics_acc`] pass plus, per constraint, the
/// ancestor walks inside [`point_jacobian`] / [`body_angular_jacobian`].
pub fn assemble(model: &Model, state: &State, constraints: &LoopConstraintSet) -> LoopSystem {
    let nv = model.nv;
    let m = constraints.nrows();

    // qdd = 0: the resulting body accelerations are exactly the velocity-
    // product terms, i.e. what `Jdot v` means. Deriving Jdot symbolically per
    // joint type would duplicate `forward_kinematics_acc`'s recursion and is
    // the kind of duplication this repo has been bitten by.
    let zero_acc = DVec::zeros(nv);
    let kin = forward_kinematics_acc(model, state, &zero_acc);

    let mut jacobian = DMat::zeros(m, nv);
    let mut position_error = DVec::zeros(m);
    let mut bias = DVec::zeros(m);

    let mut row = 0;
    for c in &constraints.constraints {
        let pa = world_point(&kin.xforms, c.anchor_a);
        let pb = world_point(&kin.xforms, c.anchor_b);

        // --- translational rows ---
        // `point_jacobian` wants the *world* point; for coincident anchors the
        // two calls would agree, but on a drifted mechanism they do not, and
        // each endpoint must be differentiated about its own current position.
        let ja = point_jac(model, &kin.xforms, c.anchor_a, pa, nv);
        let jb = point_jac(model, &kin.xforms, c.anchor_b, pb, nv);
        let ka = point_bias(&kin, c.anchor_a);
        let kb = point_bias(&kin, c.anchor_b);

        let dp = pa - pb;
        let dk = ka - kb;
        for r in 0..3 {
            for col in 0..nv {
                jacobian.set(row + r, col, ja.get(r, col) - jb.get(r, col));
            }
            position_error[row + r] = component(dp, r);
            bias[row + r] = component(dk, r);
        }
        row += 3;

        if c.kind == LoopKind::Weld {
            // --- rotational rows ---
            let ra = rotation_of(&kin.xforms, c.anchor_a.attachment);
            let rb = rotation_of(&kin.xforms, c.anchor_b.attachment);
            // Everything below is in the **world** frame, on purpose: the
            // angular Jacobian rows are world-frame, and mixing a body-frame
            // error vector with world-frame rows is a bug that looks like a
            // slightly-too-soft weld rather than like a failure.
            //
            // Where the weld says B should be: R_B_target = R_A * T.
            // Error rotation, world frame, target -> actual:
            let target_b = ra * c.target_rel_rot;
            let err = rb * target_b.transpose();
            // vee(E - E^T)/2 = sin(theta) * axis: the world rotation vector
            // taking target-B to actual-B, so its time derivative is
            // `omega_B - omega_A`. Our row convention is A minus B, hence the
            // negation below — this is precisely the sign that must match
            // `J = J_A - J_B`.
            //
            // The `sin(theta)` scaling is exact to first order and monotone in
            // |theta| up to 90 degrees. Past that it folds back toward zero and
            // stabilization would stall; documented rather than guarded,
            // because a weld 90 degrees out has already failed.
            let theta_b = Vec3::new(
                0.5 * (err.get(2, 1) - err.get(1, 2)),
                0.5 * (err.get(0, 2) - err.get(2, 0)),
                0.5 * (err.get(1, 0) - err.get(0, 1)),
            );
            let theta_w = -theta_b;

            let wa = ang_jac(model, &kin.xforms, c.anchor_a, nv);
            let wb = ang_jac(model, &kin.xforms, c.anchor_b, nv);
            let ba = ang_bias(&kin, c.anchor_a);
            let bb = ang_bias(&kin, c.anchor_b);
            let dbias = ba - bb;

            for r in 0..3 {
                for col in 0..nv {
                    jacobian.set(row + r, col, wa.get(r, col) - wb.get(r, col));
                }
                position_error[row + r] = component(theta_w, r);
                bias[row + r] = component(dbias, r);
            }
            row += 3;
        }
    }

    let velocity_error = jacobian.mul_vec(&state.v);

    LoopSystem {
        jacobian,
        position_error,
        velocity_error,
        bias,
    }
}

fn component(v: Vec3, i: usize) -> f64 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn inf_norm(v: &DVec) -> f64 {
    v.iter()
        .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc })
}

/// Body-to-world rotation of an attachment. `rot` in a Plücker transform maps
/// world into body, so the body-to-world rotation is its transpose.
fn rotation_of(xforms: &[SpatialTransform], attachment: Attachment) -> Mat3 {
    match attachment {
        Attachment::Body(b) => xforms[b].rot.transpose(),
        Attachment::World => Mat3::identity(),
    }
}

fn world_point(xforms: &[SpatialTransform], anchor: Anchor) -> Vec3 {
    match anchor.attachment {
        Attachment::Body(b) => xforms[b].body_to_world_point(anchor.offset),
        Attachment::World => anchor.offset,
    }
}

fn point_jac(
    model: &Model,
    xforms: &[SpatialTransform],
    anchor: Anchor,
    world: Vec3,
    nv: usize,
) -> DMat {
    match anchor.attachment {
        Attachment::Body(b) => point_jacobian(model, xforms, b, world),
        Attachment::World => DMat::zeros(3, nv),
    }
}

fn ang_jac(model: &Model, xforms: &[SpatialTransform], anchor: Anchor, nv: usize) -> DMat {
    match anchor.attachment {
        Attachment::Body(b) => body_angular_jacobian(model, xforms, b),
        Attachment::World => DMat::zeros(3, nv),
    }
}

/// World-frame classical acceleration of an anchor point at `qdd = 0`.
///
/// The body-frame classical acceleration of the origin is
/// `a_lin + omega x v_lin` (Featherstone's spatial acceleration is not a point
/// acceleration); carrying it out to an offset `r` adds the Euler term
/// `alpha x r` and the centripetal term `omega x (omega x r)`. Dropping the
/// centripetal term is the single most common bug in this derivation and it is
/// invisible at low speed, which is why the four-bar drift test spins.
fn point_bias(kin: &phyz_rigid::BodyKinematics, anchor: Anchor) -> Vec3 {
    match anchor.attachment {
        Attachment::World => Vec3::zeros(),
        Attachment::Body(b) => {
            let r = anchor.offset;
            let omega = kin.velocities[b].angular;
            let alpha = kin.accelerations[b].angular;
            let a_origin = kin.classical_linear_accel(b);
            let a_body = a_origin + alpha.cross(r) + omega.cross(omega.cross(r));
            kin.xforms[b].body_to_world_dir(a_body)
        }
    }
}

/// World-frame angular acceleration of an attachment at `qdd = 0`.
fn ang_bias(kin: &phyz_rigid::BodyKinematics, anchor: Anchor) -> Vec3 {
    match anchor.attachment {
        Attachment::World => Vec3::zeros(),
        Attachment::Body(b) => kin.xforms[b].body_to_world_dir(kin.accelerations[b].angular),
    }
}
