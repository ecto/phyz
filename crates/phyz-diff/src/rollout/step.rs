//! Scalar-polymorphic dynamics step: ABA + forward kinematics + per-vertex
//! ground contact + semi-implicit Euler, generic over `T: tang::Scalar`.
//!
//! Instantiated at `f64` this is the plain forward rollout; instantiated at
//! [`tang::Dual<f64>`] with one seeded input it yields an **exact** partial
//! derivative of the step through every branch the `f64` computation takes
//! (dual comparisons act on the primal). The adjoint driver in
//! [`super::adjoint`] uses those dual lanes as its per-step Jacobian oracle.
//!
//! # Scope
//!
//! **Every** joint type the model can express: revolute/hinge,
//! prismatic/slide, fixed, spherical/ball (3 DOF) and free (6 DOF). Multi-DOF
//! joints go through a generic `ndof×ndof` articulated-inertia solve and carry
//! **quaternion** configuration, so `nq ≠ nv` in general — see [`DofLayout`].
//!
//! # State layout (`nq ≠ nv`)
//!
//! Velocity coordinates are the joint's motion subspace, one per DOF. Position
//! coordinates are *not*: rotational sub-blocks of multi-DOF joints are stored
//! as unit quaternions rather than exponential coordinates, so the
//! configuration update is a Lie-group step rather than `q += dt·v`.
//!
//! | joint | `nq` | `q` layout | `nv` | `v` layout |
//! |---|---|---|---|---|
//! | revolute / prismatic | 1 | `[θ]` / `[d]` | 1 | `[θ̇]` / `[ḋ]` |
//! | fixed | 0 | — | 0 | — |
//! | spherical | 4 | `[w, x, y, z]` | 3 | body-frame `ω` |
//! | free | 7 | `[x, y, z, w, qx, qy, qz]` | 6 | body-frame `[ω; v]` |
//!
//! `[x, y, z]` of a free joint is the successor origin in **predecessor**
//! coordinates (the `pos` half of the Plücker transform), matching
//! [`tang::SpatialTransform`]; its rate is therefore `Eᵀ·v_lin`, not `v_lin`.
//! The quaternion `p` is the one with `E = R(p)`, i.e. the *coordinate*
//! transform predecessor→successor, matching the sign convention
//! `phyz_model::Joint::joint_transform_slice` uses for its exponential
//! coordinates. Since `Ė = −ω× E`, the update is
//! `p' = normalize(exp(−dt·ω) ⊗ p)`.
//!
//! This differs from `phyz_model`'s own `nq == nv` exponential-coordinate
//! packing (and from [`phyz::sim::SemiImplicitEulerSolver`]'s flat
//! `q += dt·v`, which for a free joint would add angular rates to positional
//! coordinates). Build the initial configuration with
//! [`DofLayout::neutral_q`] and index it with [`DofLayout::q_offsets`].
//!
//! # Contact model
//!
//! Contact here is a **differentiable forward model of its own**, not a
//! wrapper around the GJK/EPA narrow phase: every collision-mesh vertex below
//! the ground plane contributes an independent penalty force
//!
//! ```text
//! f_z = max(0, k·depth − c·v_z),   depth = ground_height − x_z  (> 0 inside)
//! ```
//!
//! applied at the vertex (wrench about the body origin). This is the standard
//! differentiable-simulation choice: the force is a smooth function of the
//! vertex position wherever the contact is active, and the "contact set" is
//! implicit in the smooth `max` rather than a combinatorial narrow-phase
//! output. Friction is deliberately absent from v1 (its `‖v_t‖` kink sits at
//! exactly the sticking state a resting gate converges to).

use phyz_math::DVec;
use phyz_model::{JointType, Model};
use tang::{Mat3, Quat, Scalar, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec, Vec3};

/// Number of inertia parameters per body: `[m, cx, cy, cz, Ixx, Iyy, Izz,
/// Ixy, Ixz, Iyz]` — mass, COM (body frame), inertia about the COM. The
/// packing (and the symmetric off-diagonal convention) matches vcad's
/// `BodyMassProps::scalars()` so a phyz gradient drops into the vcad seam
/// unchanged.
pub const N_INERTIA_PARAMS: usize = 10;

/// Pack a body's spatial inertia into the canonical 10-vector.
pub fn inertia_params(si: &SpatialInertia<f64>) -> [f64; N_INERTIA_PARAMS] {
    [
        si.mass,
        si.com.x,
        si.com.y,
        si.com.z,
        si.inertia.get(0, 0),
        si.inertia.get(1, 1),
        si.inertia.get(2, 2),
        si.inertia.get(0, 1),
        si.inertia.get(0, 2),
        si.inertia.get(1, 2),
    ]
}

/// Rebuild a spatial inertia from the canonical 10-vector. Each off-diagonal
/// scalar populates **both** symmetric entries, so seeding one parameter
/// differentiates with respect to the symmetric pair — the same convention a
/// central difference on the packed scalars probes.
pub fn inertia_from_params<T: Scalar>(p: &[T; N_INERTIA_PARAMS]) -> SpatialInertia<T> {
    SpatialInertia::new(
        p[0],
        Vec3::new(p[1], p[2], p[3]),
        Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
    )
}

/// A ground plane at `z = height` with a linear penalty contact law.
#[derive(Debug, Clone, Copy)]
pub struct GroundContact {
    /// World z of the plane (m).
    pub height: f64,
    /// Penalty stiffness k (N/m per vertex).
    pub stiffness: f64,
    /// Penalty damping c (N·s/m per vertex).
    pub damping: f64,
}

/// A body's collision skin: vertices in the **body frame** (m). Only the
/// vertices matter to the per-vertex penalty model — triangles are not used.
#[derive(Debug, Clone)]
pub struct CollisionMesh {
    /// Index of the body this skin is attached to.
    pub body: usize,
    /// Vertex positions in the body frame (m).
    pub vertices: Vec<Vec3<f64>>,
}

// ---------------------------------------------------------------------------
// Lifts: f64 model constants → T
// ---------------------------------------------------------------------------

pub(crate) fn lift_vec3<T: Scalar>(v: Vec3<f64>) -> Vec3<T> {
    Vec3::new(T::from_f64(v.x), T::from_f64(v.y), T::from_f64(v.z))
}

fn lift_mat3<T: Scalar>(m: &Mat3<f64>) -> Mat3<T> {
    Mat3::from_cols(
        lift_vec3(m.col(0)),
        lift_vec3(m.col(1)),
        lift_vec3(m.col(2)),
    )
}

fn lift_xform<T: Scalar>(xf: &SpatialTransform<f64>) -> SpatialTransform<T> {
    SpatialTransform::new(lift_mat3(&xf.rot), lift_vec3(xf.pos))
}

/// Lift a body's spatial inertia to `T` with zero tangents.
pub(crate) fn lift_inertia<T: Scalar>(si: &SpatialInertia<f64>) -> SpatialInertia<T> {
    SpatialInertia::new(
        T::from_f64(si.mass),
        lift_vec3(si.com),
        lift_mat3(&si.inertia),
    )
}

// ---------------------------------------------------------------------------
// Position/velocity coordinate layout
// ---------------------------------------------------------------------------

/// Largest `ndof` any joint type has (free joint).
const MAX_DOF: usize = 6;

/// Number of **position** coordinates a joint type carries in this module's
/// layout: quaternions for the rotational sub-blocks of multi-DOF joints, so
/// this is *not* `ndof()` in general. See the module docs for the packing.
pub fn joint_nq(joint_type: JointType) -> usize {
    match joint_type {
        JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => 1,
        JointType::Spherical | JointType::Ball => 4,
        JointType::Free => 7,
        JointType::Fixed => 0,
    }
}

/// The `q`/`v` coordinate split for a model under this module's layout.
///
/// `nv` and `v_offsets` agree with `Model::nv` / `Model::v_offsets` (velocity
/// coordinates are the motion subspace either way). `nq` and `q_offsets` do
/// **not** agree with the model's when any spherical or free joint is present,
/// because those carry quaternions here — always index a diff-rollout `q`
/// through this layout, never through `Model::q_offsets`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DofLayout {
    /// Total number of position coordinates.
    pub nq: usize,
    /// Total number of velocity coordinates (== `Model::nv`).
    pub nv: usize,
    /// Position offset of each joint, parallel to `Model::joints`.
    pub q_offsets: Vec<usize>,
    /// Velocity offset of each joint, parallel to `Model::joints`.
    pub v_offsets: Vec<usize>,
}

impl DofLayout {
    /// Derive the layout of a model.
    pub fn of(model: &Model) -> Self {
        let mut nq = 0;
        let mut nv = 0;
        let mut q_offsets = Vec::with_capacity(model.joints.len());
        let mut v_offsets = Vec::with_capacity(model.joints.len());
        for joint in &model.joints {
            q_offsets.push(nq);
            v_offsets.push(nv);
            nq += joint_nq(joint.joint_type);
            nv += joint.ndof();
        }
        Self {
            nq,
            nv,
            q_offsets,
            v_offsets,
        }
    }

    /// The identity configuration: zeros, except every quaternion sub-block is
    /// the identity rotation `[1, 0, 0, 0]`. Use this as the base for a `q0`
    /// and overwrite the coordinates you care about.
    pub fn neutral_q(&self, model: &Model) -> Vec<f64> {
        let mut q = vec![0.0; self.nq];
        for (j, joint) in model.joints.iter().enumerate() {
            let qi = self.q_offsets[j];
            match joint.joint_type {
                JointType::Spherical | JointType::Ball => q[qi] = 1.0,
                JointType::Free => q[qi + 3] = 1.0,
                _ => {}
            }
        }
        q
    }
}

// ---------------------------------------------------------------------------
// Generic joint helpers (all joint types)
// ---------------------------------------------------------------------------

/// Joint transform (Plücker, predecessor→successor) for `ndof`-many position
/// coordinates starting at `q[0]`. `q` must be at least `joint_nq` long.
fn joint_transform<T: Scalar>(
    joint_type: JointType,
    axis: &Vec3<T>,
    q: &[T],
) -> SpatialTransform<T> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => {
            // Rodrigues with negated angle: coordinate transform, matching
            // the concrete Joint::joint_transform_slice.
            let (s, c) = (-q[0]).sin_cos();
            let ax = tang::skew(axis);
            let rot = Mat3::identity() + ax * s + ax.mul_mat(&ax) * (T::ONE - c);
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Prismatic | JointType::Slide => {
            SpatialTransform::new(Mat3::identity(), *axis * q[0])
        }
        JointType::Spherical | JointType::Ball => {
            let rot = quat_at(&q[0..4]).normalize().to_matrix();
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Free => {
            let rot = quat_at(&q[3..7]).normalize().to_matrix();
            SpatialTransform::new(rot, Vec3::new(q[0], q[1], q[2]))
        }
        JointType::Fixed => SpatialTransform::identity(),
    }
}

/// Read a `[w, x, y, z]` quaternion out of a coordinate slice.
fn quat_at<T: Scalar>(q: &[T]) -> Quat<T> {
    Quat::new(q[0], q[1], q[2], q[3])
}

/// The single motion-subspace column of a 1-DOF joint.
fn motion_subspace<T: Scalar>(joint_type: JointType, axis: &Vec3<T>) -> SpatialVec<T> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(*axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), *axis),
        _ => unreachable!("motion_subspace is the 1-DOF path"),
    }
}

/// Cartesian basis vector `e_k`.
fn unit<T: Scalar>(k: usize) -> Vec3<T> {
    let mut e = Vec3::zero();
    match k {
        0 => e.x = T::ONE,
        1 => e.y = T::ONE,
        _ => e.z = T::ONE,
    }
    e
}

/// The motion subspace `S` (6×ndof) as columns; only the first `ndof` entries
/// of the returned array are meaningful.
fn motion_subspace_cols<T: Scalar>(
    joint_type: JointType,
    axis: &Vec3<T>,
) -> [SpatialVec<T>; MAX_DOF] {
    let mut s = [SpatialVec::zero(); MAX_DOF];
    match joint_type {
        JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
            s[0] = motion_subspace(joint_type, axis);
        }
        JointType::Spherical | JointType::Ball => {
            for (k, s_k) in s.iter_mut().enumerate().take(3) {
                *s_k = SpatialVec::new(unit(k), Vec3::zero());
            }
        }
        JointType::Free => {
            for k in 0..3 {
                s[k] = SpatialVec::new(unit(k), Vec3::zero());
                s[3 + k] = SpatialVec::new(Vec3::zero(), unit(k));
            }
        }
        JointType::Fixed => {}
    }
    s
}

/// The joint's spatial velocity `S·v` for `ndof` velocity coordinates.
fn joint_velocity<T: Scalar>(
    joint_type: JointType,
    axis: &Vec3<T>,
    v: &[T],
    ndof: usize,
) -> SpatialVec<T> {
    if ndof == 1 {
        let s = motion_subspace(joint_type, axis);
        return SpatialVec::new(s.angular * v[0], s.linear * v[0]);
    }
    let s = motion_subspace_cols(joint_type, axis);
    let mut out = SpatialVec::zero();
    for k in 0..ndof {
        out = out + SpatialVec::new(s[k].angular * v[k], s[k].linear * v[k]);
    }
    out
}

/// Solve `A·X = B` in place for a small dense system, `A` row-major `n×n` and
/// `B` row-major `n×m` (overwritten with `X`). Returns `false` — leaving `b`
/// unusable — when the matrix is numerically singular.
///
/// Gaussian elimination with partial pivoting. The pivot search and the
/// singularity test both read the **primal** (`to_f64`), so a dual-seeded
/// solve differentiates exactly the elimination order the `f64` solve takes —
/// the same discipline the rest of this module applies to its branches.
fn solve_in_place<T: Scalar>(a: &mut [T], n: usize, b: &mut [T], m: usize) -> bool {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].to_f64().abs();
        for row in (col + 1)..n {
            let mag = a[row * n + col].to_f64().abs();
            if mag > best {
                best = mag;
                piv = row;
            }
        }
        if best < 1e-20 {
            return false;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            for k in 0..m {
                b.swap(col * m + k, piv * m + k);
            }
        }
        let inv = a[col * n + col].recip();
        for row in (col + 1)..n {
            let f = a[row * n + col] * inv;
            if f.to_f64() == 0.0 {
                continue;
            }
            for k in col..n {
                let t = f * a[col * n + k];
                a[row * n + k] -= t;
            }
            for k in 0..m {
                let t = f * b[col * m + k];
                b[row * m + k] -= t;
            }
        }
    }
    // Back-substitution.
    for col in (0..n).rev() {
        let inv = a[col * n + col].recip();
        for k in 0..m {
            let mut acc = b[col * m + k];
            for j in (col + 1)..n {
                acc -= a[col * n + j] * b[j * m + k];
            }
            b[col * m + k] = acc * inv;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Generic forward kinematics
// ---------------------------------------------------------------------------

/// Generic mirror of `rigid::forward_kinematics`: world→body Plücker
/// transforms (`.pos` = body origin in world coordinates, `.rot` = world→body
/// rotation) and body-frame spatial velocities.
pub(crate) fn fk_generic<T: Scalar>(
    model: &Model,
    layout: &DofLayout,
    q: &[T],
    v: &[T],
) -> (Vec<SpatialTransform<T>>, Vec<SpatialVec<T>>) {
    let nb = model.nbodies();
    let mut xforms: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    let mut vels: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = layout.q_offsets[body.joint_idx];
        let v_idx = layout.v_offsets[body.joint_idx];
        let ndof = joint.ndof();
        let axis = lift_vec3::<T>(joint.axis);

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform(joint.joint_type, &axis, &q[q_idx..])
        };
        let x_tree = x_joint.compose(&lift_xform(&joint.parent_to_joint));

        let v_joint = if ndof == 0 {
            SpatialVec::zero()
        } else {
            joint_velocity(joint.joint_type, &axis, &v[v_idx..], ndof)
        };

        if body.parent < 0 {
            xforms[i] = x_tree;
            vels[i] = v_joint;
        } else {
            let pi = body.parent as usize;
            xforms[i] = x_tree.compose(&xforms[pi]);
            vels[i] = x_tree.apply_motion(&vels[pi]) + v_joint;
        }
    }

    (xforms, vels)
}

// ---------------------------------------------------------------------------
// Per-vertex ground contact
// ---------------------------------------------------------------------------

/// Wrench (about the body origin, **body frame**) contributed by one
/// collision-mesh vertex against the ground plane.
///
/// `x_b` is the vertex in body coordinates; `xform` the body's world→body
/// Plücker transform; `twist` the body-frame spatial velocity. Every factor
/// is generic, so seeding `x_b` gives the exact ∂(wrench)/∂(vertex) the
/// adjoint's vertex channel contracts against.
pub(crate) fn vertex_wrench<T: Scalar>(
    gc: &GroundContact,
    x_b: Vec3<T>,
    xform: &SpatialTransform<T>,
    twist: &SpatialVec<T>,
) -> SpatialVec<T> {
    let e = &xform.rot; // world → body rotation
    let x_w = xform.pos + e.transpose().mul_vec(x_b);
    let depth = T::from_f64(gc.height) - x_w.z;

    // Vertex world velocity: body-frame point velocity, rotated to world.
    let v_b = twist.linear + twist.angular.cross(x_b);
    let v_w = e.transpose().mul_vec(v_b);

    // Spring–damper along +Z, pushing the body out of the plane; clamped to
    // non-adhesive, and gated on actual penetration. Both the clamp and the
    // gate branch on the primal, so the tangent is the derivative of the
    // branch the f64 rollout takes.
    let raw = T::from_f64(gc.stiffness) * depth - T::from_f64(gc.damping) * v_w.z;
    let f_mag = T::select(depth, raw.max(T::ZERO), T::ZERO);

    let f_w = Vec3::new(T::ZERO, T::ZERO, f_mag);
    let f_b = e.mul_vec(f_w);
    SpatialVec::new(x_b.cross(f_b), f_b)
}

/// Sum the ground-contact wrenches of every collision mesh into per-body
/// external wrenches (body frame), given precomputed kinematics.
fn contact_wrenches<T: Scalar>(
    gc: &GroundContact,
    meshes: &[CollisionMesh],
    xforms: &[SpatialTransform<T>],
    vels: &[SpatialVec<T>],
    out: &mut [SpatialVec<T>],
) {
    for mesh in meshes {
        let b = mesh.body;
        for v in &mesh.vertices {
            out[b] = out[b] + vertex_wrench(gc, lift_vec3(*v), &xforms[b], &vels[b]);
        }
    }
}

// ---------------------------------------------------------------------------
// Generic ABA (single-DOF) with external wrenches
// ---------------------------------------------------------------------------

/// Generic mirror of `rigid::aba_with_external_forces` over the full joint
/// domain, with the body inertias supplied separately so a caller can seed
/// them.
///
/// Single-DOF joints keep the scalar fast path (`D` is one number); spherical
/// and free joints go through a dense `ndof×ndof` solve
/// ([`solve_in_place`]), which is what makes the articulated-inertia
/// factorisation `Iᴬ = I − U D⁻¹ Uᵀ` well-defined for them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aba_generic<T: Scalar>(
    model: &Model,
    layout: &DofLayout,
    inertias: &[SpatialInertia<T>],
    q: &[T],
    v: &[T],
    ctrl: &[T],
    ext: Option<&[SpatialVec<T>]>,
) -> Vec<T> {
    let nb = model.nbodies();
    let mut qdd = vec![T::ZERO; layout.nv];

    let mut x_tree: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    let mut vel: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut c_bias: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut p_a: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut i_a: Vec<SpatialMat<T>> = vec![SpatialMat::zero(); nb];

    // Gravity as base acceleration.
    let a0: SpatialVec<T> = SpatialVec::new(Vec3::zero(), -lift_vec3(model.gravity));

    // -- Pass 1: velocities and bias --
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = layout.q_offsets[body.joint_idx];
        let v_idx = layout.v_offsets[body.joint_idx];
        let ndof = joint.ndof();
        let axis = lift_vec3::<T>(joint.axis);

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform(joint.joint_type, &axis, &q[q_idx..])
        };
        x_tree[i] = x_joint.compose(&lift_xform(&joint.parent_to_joint));

        let v_joint = if ndof == 0 {
            SpatialVec::zero()
        } else {
            joint_velocity(joint.joint_type, &axis, &v[v_idx..], ndof)
        };

        if body.parent < 0 {
            vel[i] = v_joint;
            c_bias[i] = SpatialVec::zero();
        } else {
            let pi = body.parent as usize;
            vel[i] = x_tree[i].apply_motion(&vel[pi]) + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }

        i_a[i] = inertias[i].to_matrix();
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));
        if let Some(ext) = ext {
            p_a[i] = p_a[i] - ext[i];
        }
    }

    // -- Pass 2: articulated inertias and forces --
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = layout.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        if ndof == 0 {
            if body.parent >= 0 {
                let pi = body.parent as usize;
                let x_mot = x_tree[i].to_motion_matrix();
                i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&i_a[i]).mul_mat(&x_mot);
                p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_a[i]);
            }
            continue;
        }

        let axis = lift_vec3::<T>(joint.axis);

        if ndof > 1 {
            // Multi-DOF: U = Iᴬ S, D = Sᵀ U, u = τ − Sᵀ pᴬ, then one solve of
            // D against [u | Uᵀ] gives both D⁻¹u and D⁻¹Uᵀ.
            let s = motion_subspace_cols(joint.joint_type, &axis);
            let ia = i_a[i];
            let mut u_cols = [SpatialVec::<T>::zero(); MAX_DOF];
            for k in 0..ndof {
                u_cols[k] = ia.mul_vec(&s[k]);
            }

            let mut d = vec![T::ZERO; ndof * ndof];
            for k in 0..ndof {
                for l in 0..ndof {
                    d[k * ndof + l] = s[k].dot(&u_cols[l]);
                }
            }
            // rhs column 0 = u, columns 1..7 = Uᵀ.
            let mut rhs = vec![T::ZERO; ndof * 7];
            for k in 0..ndof {
                rhs[k * 7] =
                    ctrl[v_idx + k] - T::from_f64(joint.damping) * v[v_idx + k] - s[k].dot(&p_a[i]);
                let uk = u_cols[k].as_array();
                for c in 0..6 {
                    rhs[k * 7 + 1 + c] = uk[c];
                }
            }
            if !solve_in_place(&mut d, ndof, &mut rhs, 7) {
                continue;
            }

            if body.parent >= 0 {
                let pi = body.parent as usize;
                // Iᴬ_new = Iᴬ − Σ_k U[:,k] ⊗ (D⁻¹Uᵀ)[k,:]
                let mut ia_new = ia;
                for k in 0..ndof {
                    let y_k = SpatialVec::new(
                        Vec3::new(rhs[k * 7 + 1], rhs[k * 7 + 2], rhs[k * 7 + 3]),
                        Vec3::new(rhs[k * 7 + 4], rhs[k * 7 + 5], rhs[k * 7 + 6]),
                    );
                    ia_new = ia_new - outer_product_6(&u_cols[k], &y_k);
                }
                // pᴬ_new = pᴬ + Iᴬ_new·c + U·(D⁻¹u)
                let mut p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]);
                for k in 0..ndof {
                    p_new = p_new + u_cols[k] * rhs[k * 7];
                }

                let x_mot = x_tree[i].to_motion_matrix();
                i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
                p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_new);
            }
            continue;
        }

        let s_i = motion_subspace(joint.joint_type, &axis);
        let tau_i = ctrl[v_idx] - T::from_f64(joint.damping) * v[v_idx];

        let ia = &i_a[i];
        let u_i = tau_i - s_i.dot(&p_a[i]);
        let d_i = s_i.dot(&ia.mul_vec(&s_i));

        // Same degeneracy guard as the concrete ABA, on the primal.
        if d_i.to_f64().abs() < 1e-20 {
            continue;
        }

        let ia_s = ia.mul_vec(&s_i);
        if body.parent >= 0 {
            let pi = body.parent as usize;
            let d_inv = d_i.recip();
            let outer = outer_product_6(&ia_s, &ia_s);
            let ia_new = *ia - outer * d_inv;
            let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * (u_i * d_inv);

            let x_mot = x_tree[i].to_motion_matrix();
            i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
            p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_new);
        }
    }

    // -- Pass 3: accelerations --
    let mut acc: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = layout.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let a_parent = if body.parent < 0 {
            x_tree[i].apply_motion(&a0)
        } else {
            x_tree[i].apply_motion(&acc[body.parent as usize])
        };

        if ndof == 0 {
            acc[i] = a_parent + c_bias[i];
            continue;
        }

        let axis = lift_vec3::<T>(joint.axis);

        if ndof > 1 {
            // qdd = D⁻¹(u − Sᵀ Iᴬ (a_parent + c))
            let s = motion_subspace_cols(joint.joint_type, &axis);
            let ia = i_a[i];
            let a_total = a_parent + c_bias[i];
            let ia_a = ia.mul_vec(&a_total);

            let mut d = vec![T::ZERO; ndof * ndof];
            let mut rhs = vec![T::ZERO; ndof];
            for k in 0..ndof {
                for l in 0..ndof {
                    d[k * ndof + l] = s[k].dot(&ia.mul_vec(&s[l]));
                }
                rhs[k] = ctrl[v_idx + k]
                    - T::from_f64(joint.damping) * v[v_idx + k]
                    - s[k].dot(&p_a[i])
                    - s[k].dot(&ia_a);
            }
            if !solve_in_place(&mut d, ndof, &mut rhs, 1) {
                acc[i] = a_total;
                continue;
            }
            let mut s_qdd = SpatialVec::zero();
            for k in 0..ndof {
                qdd[v_idx + k] = rhs[k];
                s_qdd = s_qdd + SpatialVec::new(s[k].angular * rhs[k], s[k].linear * rhs[k]);
            }
            acc[i] = a_total + s_qdd;
            continue;
        }

        let s_i = motion_subspace(joint.joint_type, &axis);
        let ia = &i_a[i];
        let d_i = s_i.dot(&ia.mul_vec(&s_i));
        if d_i.to_f64().abs() < 1e-20 {
            acc[i] = a_parent + c_bias[i];
            continue;
        }

        let tau_i = ctrl[v_idx] - T::from_f64(joint.damping) * v[v_idx];
        let u_i = tau_i - s_i.dot(&p_a[i]);
        let qdd_i = (u_i - ia.mul_vec(&(a_parent + c_bias[i])).dot(&s_i)) * d_i.recip();
        qdd[v_idx] = qdd_i;
        acc[i] = a_parent + c_bias[i] + SpatialVec::new(s_i.angular * qdd_i, s_i.linear * qdd_i);
    }

    qdd
}

fn outer_product_6<T: Scalar>(a: &SpatialVec<T>, b: &SpatialVec<T>) -> SpatialMat<T> {
    fn v3_outer<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Mat3<T> {
        Mat3::new(
            a.x * b.x,
            a.x * b.y,
            a.x * b.z,
            a.y * b.x,
            a.y * b.y,
            a.y * b.z,
            a.z * b.x,
            a.z * b.y,
            a.z * b.z,
        )
    }
    SpatialMat::new(
        v3_outer(a.angular, b.angular),
        v3_outer(a.angular, b.linear),
        v3_outer(a.linear, b.angular),
        v3_outer(a.linear, b.linear),
    )
}

// ---------------------------------------------------------------------------
// Generic semi-implicit Euler step
// ---------------------------------------------------------------------------

/// The configuration update `q' = Φ(q, v')` of the semi-implicit step, split
/// out because the adjoint needs its two Jacobian blocks (`Φ_q`, `Φ_v'`)
/// separately — with quaternion coordinates neither is the identity and
/// `Φ_v'` is not even square.
///
/// Single-DOF joints get `q' = q + dt·v'`. Quaternion sub-blocks get the
/// Lie-group step `p' = normalize(exp(−dt·ω) ⊗ p)` (the sign follows
/// `Ė = −ω× E` for the coordinate transform `E = R(p)`), and a free joint's
/// translation integrates `ṙ = Eᵀ·v_lin` at the **current** orientation.
pub(crate) fn config_update_generic<T: Scalar>(
    model: &Model,
    layout: &DofLayout,
    q: &[T],
    v_next: &[T],
    dt: T,
) -> Vec<T> {
    let mut out = q.to_vec();
    for (j, joint) in model.joints.iter().enumerate() {
        let qi = layout.q_offsets[j];
        let vi = layout.v_offsets[j];
        match joint.joint_type {
            JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
                out[qi] = q[qi] + dt * v_next[vi];
            }
            JointType::Spherical | JointType::Ball => {
                let p = quat_at(&q[qi..qi + 4]);
                let omega = Vec3::new(v_next[vi], v_next[vi + 1], v_next[vi + 2]);
                write_quat(&mut out[qi..qi + 4], &rotate_step(&p, omega, dt));
            }
            JointType::Free => {
                let p = quat_at(&q[qi + 3..qi + 7]);
                let e = p.normalize().to_matrix();
                let v_lin = Vec3::new(v_next[vi + 3], v_next[vi + 4], v_next[vi + 5]);
                let dr = e.transpose().mul_vec(v_lin) * dt;
                out[qi] = q[qi] + dr.x;
                out[qi + 1] = q[qi + 1] + dr.y;
                out[qi + 2] = q[qi + 2] + dr.z;

                let omega = Vec3::new(v_next[vi], v_next[vi + 1], v_next[vi + 2]);
                write_quat(&mut out[qi + 3..qi + 7], &rotate_step(&p, omega, dt));
            }
            JointType::Fixed => {}
        }
    }
    out
}

/// `normalize(exp(−dt·ω) ⊗ p)` — one Lie-group step of the coordinate
/// quaternion under body-frame angular velocity `ω`.
fn rotate_step<T: Scalar>(p: &Quat<T>, omega: Vec3<T>, dt: T) -> Quat<T> {
    Quat::exp(&(omega * -dt)).mul(p).normalize()
}

fn write_quat<T: Scalar>(out: &mut [T], p: &Quat<T>) {
    out[0] = p.w;
    out[1] = p.v.x;
    out[2] = p.v.y;
    out[3] = p.v.z;
}

/// One semi-implicit Euler step, generic over the scalar:
///
/// ```text
/// qdd  = ABA(q, v, ctrl; π, contact(q, v, V), ext)
/// v'   = v + dt·qdd
/// q'   = Φ(q, v')
/// ```
///
/// `ext` is an additive per-body external wrench **on top of** the contact
/// forces — zero in the plain rollout, dual-seeded by the adjoint driver to
/// read out `∂qdd/∂(wrench component)` for the vertex channel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_generic<T: Scalar>(
    model: &Model,
    layout: &DofLayout,
    inertias: &[SpatialInertia<T>],
    contact: Option<(&GroundContact, &[CollisionMesh])>,
    ext: Option<&[SpatialVec<T>]>,
    q: &[T],
    v: &[T],
    ctrl: &[T],
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let nb = model.nbodies();
    let dt = T::from_f64(model.dt);

    let total_ext: Option<Vec<SpatialVec<T>>> = if contact.is_some() || ext.is_some() {
        let mut w: Vec<SpatialVec<T>> = match ext {
            Some(e) => e.to_vec(),
            None => vec![SpatialVec::zero(); nb],
        };
        if let Some((gc, meshes)) = contact {
            let (xforms, vels) = fk_generic(model, layout, q, v);
            contact_wrenches(gc, meshes, &xforms, &vels, &mut w);
        }
        Some(w)
    } else {
        None
    };

    let qdd = aba_generic(model, layout, inertias, q, v, ctrl, total_ext.as_deref());

    let v_next: Vec<T> = v.iter().zip(&qdd).map(|(&vi, &ai)| vi + dt * ai).collect();
    let q_next = config_update_generic(model, layout, q, &v_next, dt);

    (q_next, v_next, qdd)
}

/// Derive the coordinate layout and the nominal inertia parameters (one
/// 10-vector per body) of a model.
pub(crate) fn validate_and_params(model: &Model) -> (DofLayout, Vec<[f64; N_INERTIA_PARAMS]>) {
    let layout = DofLayout::of(model);
    let params = model
        .bodies
        .iter()
        .map(|b| inertia_params(&b.inertia))
        .collect();
    (layout, params)
}

/// Convenience: run the plain `f64` rollout for `steps` steps from `(q0, v0)`
/// under an open-loop control schedule, returning every state along the way
/// (`states[t] = (q_t, v_t)`, `t = 0..=steps`).
pub(crate) fn rollout_states(
    model: &Model,
    layout: &DofLayout,
    contact: Option<(&GroundContact, &[CollisionMesh])>,
    q0: &[f64],
    v0: &[f64],
    ctrl: &dyn Fn(usize) -> DVec,
    steps: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let inertias: Vec<SpatialInertia<f64>> = model.bodies.iter().map(|b| b.inertia).collect();
    let mut states = Vec::with_capacity(steps + 1);
    let mut q = q0.to_vec();
    let mut v = v0.to_vec();
    states.push((q.clone(), v.clone()));
    for t in 0..steps {
        let u = ctrl(t);
        let (qn, vn, _) = step_generic::<f64>(
            model,
            layout,
            &inertias,
            contact,
            None,
            &q,
            &v,
            u.as_slice(),
        );
        q = qn;
        v = vn;
        states.push((q.clone(), v.clone()));
    }
    states
}
