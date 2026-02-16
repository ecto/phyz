//! Composite Rigid Body Algorithm (CRBA) — mass matrix computation.

use tau_math::{DMat, SpatialMat, SpatialTransform};
use tau_model::{Model, State};

/// Compute the joint-space mass matrix M(q) using CRBA.
///
/// Returns an nv × nv symmetric positive-definite matrix.
pub fn crba(model: &Model, state: &State) -> DMat {
    let nb = model.nbodies();
    let mut mass_matrix = DMat::zeros(model.nv, model.nv);

    // Compute tree transforms
    let mut x_tree = vec![SpatialTransform::identity(); nb];
    #[allow(clippy::needless_range_loop)]
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let q = state.q[q_idx];
        let x_joint = joint.joint_transform(q);
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);
    }

    // Composite inertias (initialized from body inertias)
    let mut i_c: Vec<SpatialMat> = model.bodies.iter().map(|b| b.inertia.to_matrix()).collect();

    // Backward pass: accumulate composite inertias
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        if body.parent >= 0 {
            let pi = body.parent as usize;
            // Transform child composite inertia to parent frame: I_parent = X^T I X
            let x_mot = x_tree[i].to_motion_matrix();
            let ic_in_parent = SpatialMat::from_mat6(x_mot.transpose() * i_c[i].data * x_mot);
            i_c[pi] = i_c[pi] + ic_in_parent;
        }
    }

    // Compute mass matrix entries
    for i in 0..nb {
        let joint_i = &model.joints[model.bodies[i].joint_idx];
        let v_i = model.v_offsets[model.bodies[i].joint_idx];
        let s_i = joint_i.motion_subspace();

        // Diagonal: S_i^T * I_c_i * S_i
        let f_i = i_c[i].mul_vec(&s_i);
        mass_matrix[(v_i, v_i)] = s_i.dot(&f_i);

        // Off-diagonal: walk up the tree
        let mut f = x_tree[i].inv_apply_force(&f_i);
        let mut j = model.bodies[i].parent;
        while j >= 0 {
            let ju = j as usize;
            let joint_j = &model.joints[model.bodies[ju].joint_idx];
            let v_j = model.v_offsets[model.bodies[ju].joint_idx];
            let s_j = joint_j.motion_subspace();

            mass_matrix[(v_i, v_j)] = s_j.dot(&f);
            mass_matrix[(v_j, v_i)] = mass_matrix[(v_i, v_j)];

            f = x_tree[ju].inv_apply_force(&f);
            j = model.bodies[ju].parent;
        }
    }

    mass_matrix
}
