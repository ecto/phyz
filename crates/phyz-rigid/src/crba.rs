//! Composite Rigid Body Algorithm (CRBA) -- mass matrix computation.

use phyz_math::{DMat, SpatialMat, SpatialTransform};
use phyz_model::{Model, State};

/// Compute the joint-space mass matrix M(q) using CRBA.
///
/// Returns an nv x nv symmetric positive-definite matrix.
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
        let ndof = joint.ndof();
        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            let q_slice = &state.q.as_slice()[q_idx..q_idx + ndof];
            joint.joint_transform_slice(q_slice)
        };
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);
    }

    // Composite inertias (initialized from body inertias)
    let mut i_c: Vec<SpatialMat> = model.bodies.iter().map(|b| b.inertia.to_matrix()).collect();

    // Backward pass: accumulate composite inertias
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        if body.parent >= 0 {
            let pi = body.parent as usize;
            let x_mot = x_tree[i].to_motion_matrix();
            let ic_in_parent = x_mot.transpose().mul_mat(&i_c[i]).mul_mat(&x_mot);
            i_c[pi] = i_c[pi] + ic_in_parent;
        }
    }

    // Compute mass matrix entries
    for i in 0..nb {
        let joint_i = &model.joints[model.bodies[i].joint_idx];
        let v_i = model.v_offsets[model.bodies[i].joint_idx];
        let ndof_i = joint_i.ndof();

        if ndof_i == 0 {
            continue;
        }

        if ndof_i == 1 {
            let s_i = joint_i.motion_subspace();
            let f_i = i_c[i].mul_vec(&s_i);
            mass_matrix[(v_i, v_i)] = s_i.dot(&f_i);

            // Off-diagonal: walk up the tree
            let mut f = x_tree[i].inv_apply_force(&f_i);
            let mut j = model.bodies[i].parent;
            while j >= 0 {
                let ju = j as usize;
                let joint_j = &model.joints[model.bodies[ju].joint_idx];
                let v_j = model.v_offsets[model.bodies[ju].joint_idx];
                let ndof_j = joint_j.ndof();

                if ndof_j == 1 {
                    let s_j = joint_j.motion_subspace();
                    mass_matrix[(v_i, v_j)] = s_j.dot(&f);
                    mass_matrix[(v_j, v_i)] = mass_matrix[(v_i, v_j)];
                } else if ndof_j > 1 {
                    let s_j = joint_j.motion_subspace_matrix();
                    let f_vec = crate::aba::sv_to_dvec(&f);
                    let block = s_j.transpose().mul_vec(&f_vec); // ndof_j x 1
                    for kj in 0..ndof_j {
                        mass_matrix[(v_i, v_j + kj)] = block[kj];
                        mass_matrix[(v_j + kj, v_i)] = block[kj];
                    }
                }

                f = x_tree[ju].inv_apply_force(&f);
                j = model.bodies[ju].parent;
            }
        } else {
            // Multi-DOF joint
            let s_i = joint_i.motion_subspace_matrix(); // 6 x ndof_i
            let ic_dmat = crate::aba::sm_to_dmat(&i_c[i]);

            // F_i = I_c * S_i  (6 x ndof_i)
            let f_i_mat = ic_dmat.mul_mat(&s_i); // 6 x ndof_i

            // Diagonal block: S_i^T * I_c * S_i  (ndof_i x ndof_i)
            let diag = s_i.transpose().mul_mat(&f_i_mat);
            for ki in 0..ndof_i {
                for kj in 0..ndof_i {
                    mass_matrix[(v_i + ki, v_i + kj)] = diag[(ki, kj)];
                }
            }

            // Off-diagonal: walk up tree, one column at a time
            for col in 0..ndof_i {
                let f_col = crate::aba::dvec_to_sv(&f_i_mat.col_vec(col));
                let mut f = x_tree[i].inv_apply_force(&f_col);
                let mut j = model.bodies[i].parent;
                while j >= 0 {
                    let ju = j as usize;
                    let joint_j = &model.joints[model.bodies[ju].joint_idx];
                    let v_j = model.v_offsets[model.bodies[ju].joint_idx];
                    let ndof_j = joint_j.ndof();

                    if ndof_j == 1 {
                        let s_j = joint_j.motion_subspace();
                        let val = s_j.dot(&f);
                        mass_matrix[(v_i + col, v_j)] = val;
                        mass_matrix[(v_j, v_i + col)] = val;
                    } else if ndof_j > 1 {
                        let s_j = joint_j.motion_subspace_matrix();
                        let f_vec = crate::aba::sv_to_dvec(&f);
                        let block = s_j.transpose().mul_vec(&f_vec);
                        for kj in 0..ndof_j {
                            mass_matrix[(v_i + col, v_j + kj)] = block[kj];
                            mass_matrix[(v_j + kj, v_i + col)] = block[kj];
                        }
                    }

                    f = x_tree[ju].inv_apply_force(&f);
                    j = model.bodies[ju].parent;
                }
            }
        }
    }

    mass_matrix
}
