//! Mobility: how many degrees of freedom a closed mechanism actually has.
//!
//! Two independent answers, which is the point — one is combinatorial and one
//! is numerical, and a disagreement between them is informative.
//!
//! - [`grubler`] is the Grübler/Kutzbach count, a closed form over link and
//!   joint counts. It is what a textbook quotes, and it is *wrong* on
//!   mechanisms with special geometry (a Sarrus linkage, a planar mechanism
//!   built from spatial joints), because it cannot see geometry at all.
//! - [`mobility`] is `nv - rank(J)` at an actual configuration. It sees the
//!   geometry, and it is configuration-dependent — at a singular pose a
//!   four-bar gains an instantaneous DOF, and this function will say so.
//!
//! The four-bar test asserts both and reports both.

use crate::solver::constraint_rank;
use phyz_math::DMat;

/// The space a mechanism moves in, i.e. the number of constraints a fully
/// rigid connection imposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobilitySpace {
    /// Planar mechanism: 3 (two translations, one rotation).
    Planar,
    /// Spatial mechanism: 6.
    Spatial,
}

impl MobilitySpace {
    /// Constraints imposed by a rigid connection in this space.
    pub fn dim(self) -> i64 {
        match self {
            MobilitySpace::Planar => 3,
            MobilitySpace::Spatial => 6,
        }
    }
}

/// Grübler/Kutzbach mobility: `d (n - 1 - j) + sum(f_i)`.
///
/// `n_links` **counts the ground as a link** — a four-bar is `n = 4`, not 3.
/// This is the convention every textbook uses and the one people get wrong.
/// `joint_dofs` has one entry per joint.
///
/// The formula assumes generic geometry. It has no way to know that four
/// parallel revolute axes make a spatial four-bar planar, so on such a
/// mechanism the spatial count comes out negative ("overconstrained") while
/// the mechanism moves perfectly well. That is a property of the formula, not
/// of the mechanism, and it is why [`mobility`] exists.
pub fn grubler(space: MobilitySpace, n_links: usize, joint_dofs: &[usize]) -> i64 {
    let d = space.dim();
    let n = n_links as i64;
    let j = joint_dofs.len() as i64;
    let f: i64 = joint_dofs.iter().map(|&x| x as i64).sum();
    d * (n - 1 - j) + f
}

/// Instantaneous mobility at a configuration: `nv - rank(J)`.
///
/// Configuration-dependent by construction. `rel_tol` is the relative singular
/// value below which a direction counts as unconstrained; `1e-9` is a
/// reasonable default for a well-scaled mechanism, and a mechanism near a
/// singularity will be sensitive to it — that sensitivity is real, not an
/// artefact.
pub fn mobility(nv: usize, jacobian: &DMat, rel_tol: f64) -> usize {
    nv.saturating_sub(constraint_rank(jacobian, rel_tol))
}
