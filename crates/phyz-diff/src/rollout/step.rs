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
//! Same joint domain as `phyz-diff`'s symbolic tracer: **single-DOF joints**
//! (revolute/hinge, prismatic/slide) plus fixed. Spherical and free joints
//! need a generic `ndof×ndof` articulated-inertia solve and quaternion state
//! handling; they are deliberately out of scope (the functions panic, they do
//! not silently misdifferentiate).
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
use tang::{Mat3, Scalar, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec, Vec3};

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
// Generic joint helpers (single-DOF domain)
// ---------------------------------------------------------------------------

fn assert_single_dof(model: &Model) {
    assert_eq!(
        model.nq, model.nv,
        "diff step requires nq == nv (single-DOF joints only)"
    );
    for joint in &model.joints {
        assert!(
            matches!(
                joint.joint_type,
                JointType::Revolute
                    | JointType::Hinge
                    | JointType::Prismatic
                    | JointType::Slide
                    | JointType::Fixed
            ),
            "diff step supports revolute/prismatic/fixed joints only, found {:?}",
            joint.joint_type
        );
    }
}

fn joint_transform<T: Scalar>(joint_type: JointType, axis: &Vec3<T>, q: T) -> SpatialTransform<T> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => {
            // Rodrigues with negated angle: coordinate transform, matching
            // the concrete Joint::joint_transform_slice.
            let (s, c) = (-q).sin_cos();
            let ax = tang::skew(axis);
            let rot = Mat3::identity() + ax * s + ax.mul_mat(&ax) * (T::ONE - c);
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Prismatic | JointType::Slide => {
            SpatialTransform::new(Mat3::identity(), *axis * q)
        }
        JointType::Fixed => SpatialTransform::identity(),
        _ => unreachable!("multi-DOF joints rejected by assert_single_dof"),
    }
}

fn motion_subspace<T: Scalar>(joint_type: JointType, axis: &Vec3<T>) -> SpatialVec<T> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(*axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), *axis),
        _ => unreachable!("multi-DOF joints rejected by assert_single_dof"),
    }
}

// ---------------------------------------------------------------------------
// Generic forward kinematics
// ---------------------------------------------------------------------------

/// Generic mirror of `rigid::forward_kinematics`: world→body Plücker
/// transforms (`.pos` = body origin in world coordinates, `.rot` = world→body
/// rotation) and body-frame spatial velocities.
pub(crate) fn fk_generic<T: Scalar>(
    model: &Model,
    q: &[T],
    v: &[T],
) -> (Vec<SpatialTransform<T>>, Vec<SpatialVec<T>>) {
    let nb = model.nbodies();
    let mut xforms: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    let mut vels: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();
        let axis = lift_vec3::<T>(joint.axis);

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform(joint.joint_type, &axis, q[q_idx])
        };
        let x_tree = x_joint.compose(&lift_xform(&joint.parent_to_joint));

        let v_joint = if ndof == 0 {
            SpatialVec::zero()
        } else {
            let s = motion_subspace(joint.joint_type, &axis);
            SpatialVec::new(s.angular * v[v_idx], s.linear * v[v_idx])
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

/// Generic mirror of `rigid::aba_with_external_forces`, restricted to the
/// single-DOF joint domain, with the body inertias supplied separately so a
/// caller can seed them.
pub(crate) fn aba_generic<T: Scalar>(
    model: &Model,
    inertias: &[SpatialInertia<T>],
    q: &[T],
    v: &[T],
    ctrl: &[T],
    ext: Option<&[SpatialVec<T>]>,
) -> Vec<T> {
    let nb = model.nbodies();
    let mut qdd = vec![T::ZERO; model.nv];

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
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();
        let axis = lift_vec3::<T>(joint.axis);

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform(joint.joint_type, &axis, q[q_idx])
        };
        x_tree[i] = x_joint.compose(&lift_xform(&joint.parent_to_joint));

        let v_joint = if ndof == 0 {
            SpatialVec::zero()
        } else {
            let s = motion_subspace(joint.joint_type, &axis);
            SpatialVec::new(s.angular * v[v_idx], s.linear * v[v_idx])
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
        let v_idx = model.v_offsets[body.joint_idx];
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
        let v_idx = model.v_offsets[body.joint_idx];
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

/// Body-frame spatial velocities and accelerations at a nominal
/// `(q, v, qdd)`, mirroring passes 1 and 3 of [`aba_generic`] exactly
/// (base acceleration `-gravity`, so the accelerations are the ones an
/// inverse-dynamics pass would use).
///
/// This is the kinematic half of the recursive Newton–Euler regressor the
/// adjoint's inertia channel contracts against; it costs one `O(nb)` sweep,
/// no dual arithmetic.
pub(crate) fn nominal_motion(
    model: &Model,
    q: &[f64],
    v: &[f64],
    qdd: &[f64],
) -> (Vec<SpatialVec<f64>>, Vec<SpatialVec<f64>>) {
    let nb = model.nbodies();
    let mut vel = vec![SpatialVec::<f64>::zero(); nb];
    let mut c_bias = vec![SpatialVec::<f64>::zero(); nb];
    let mut acc = vec![SpatialVec::<f64>::zero(); nb];
    let mut x_tree = vec![SpatialTransform::<f64>::identity(); nb];

    let a0 = SpatialVec::new(Vec3::zero(), -model.gravity);

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform(joint.joint_type, &joint.axis, q[q_idx])
        };
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);

        let v_joint = if ndof == 0 {
            SpatialVec::zero()
        } else {
            let s = motion_subspace(joint.joint_type, &joint.axis);
            SpatialVec::new(s.angular * v[v_idx], s.linear * v[v_idx])
        };

        if body.parent < 0 {
            vel[i] = v_joint;
        } else {
            let pi = body.parent as usize;
            vel[i] = x_tree[i].apply_motion(&vel[pi]) + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }
    }

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let a_parent = if body.parent < 0 {
            x_tree[i].apply_motion(&a0)
        } else {
            x_tree[i].apply_motion(&acc[body.parent as usize])
        };

        acc[i] = a_parent + c_bias[i];
        if ndof > 0 {
            let s = motion_subspace(joint.joint_type, &joint.axis);
            let a_j = qdd[v_idx];
            acc[i] = acc[i] + SpatialVec::new(s.angular * a_j, s.linear * a_j);
        }
    }

    (vel, acc)
}

// ---------------------------------------------------------------------------
// Generic semi-implicit Euler step
// ---------------------------------------------------------------------------

/// One semi-implicit Euler step, generic over the scalar:
///
/// ```text
/// qdd  = ABA(q, v, ctrl; π, contact(q, v, V), ext)
/// v'   = v + dt·qdd
/// q'   = q + dt·v'
/// ```
///
/// `ext` is an additive per-body external wrench **on top of** the contact
/// forces — zero in the plain rollout, dual-seeded by the adjoint driver to
/// read out `∂qdd/∂(wrench component)` for the vertex channel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_generic<T: Scalar>(
    model: &Model,
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
            let (xforms, vels) = fk_generic(model, q, v);
            contact_wrenches(gc, meshes, &xforms, &vels, &mut w);
        }
        Some(w)
    } else {
        None
    };

    let qdd = aba_generic(model, inertias, q, v, ctrl, total_ext.as_deref());

    let v_next: Vec<T> = v.iter().zip(&qdd).map(|(&vi, &ai)| vi + dt * ai).collect();
    let q_next: Vec<T> = q
        .iter()
        .zip(&v_next)
        .map(|(&qi, &vi)| qi + dt * vi)
        .collect();

    (q_next, v_next, qdd)
}

/// Validate a model for the diff step domain and return its nominal inertia
/// parameters (one 10-vector per body).
pub(crate) fn validate_and_params(model: &Model) -> Vec<[f64; N_INERTIA_PARAMS]> {
    assert_single_dof(model);
    model
        .bodies
        .iter()
        .map(|b| inertia_params(&b.inertia))
        .collect()
}

/// A nominal trajectory: `(states, accels)` — see [`rollout_states`].
pub(crate) type RolloutTrace = (Vec<(Vec<f64>, Vec<f64>)>, Vec<Vec<f64>>);

/// Convenience: run the plain `f64` rollout for `steps` steps from `(q0, v0)`
/// under an open-loop control schedule, returning every state along the way
/// (`states[t] = (q_t, v_t)`, `t = 0..=steps`) and the nominal accelerations
/// (`accels[t] = qdd(q_t, v_t, u_t)`, length `steps`). The backward pass needs
/// the accelerations to build its inverse-dynamics regressor; recording them
/// here is free, recomputing them later is not.
pub(crate) fn rollout_states(
    model: &Model,
    contact: Option<(&GroundContact, &[CollisionMesh])>,
    q0: &[f64],
    v0: &[f64],
    ctrl: &dyn Fn(usize) -> DVec,
    steps: usize,
) -> RolloutTrace {
    let inertias: Vec<SpatialInertia<f64>> = model.bodies.iter().map(|b| b.inertia).collect();
    let mut states = Vec::with_capacity(steps + 1);
    let mut accels = Vec::with_capacity(steps);
    let mut q = q0.to_vec();
    let mut v = v0.to_vec();
    states.push((q.clone(), v.clone()));
    for t in 0..steps {
        let u = ctrl(t);
        let (qn, vn, qdd) =
            step_generic::<f64>(model, &inertias, contact, None, &q, &v, u.as_slice());
        q = qn;
        v = vn;
        states.push((q.clone(), v.clone()));
        accels.push(qdd);
    }
    (states, accels)
}
