//! Coupling definitions between different solver types.

use crate::boundary::BoundingBox;
use phyz_math::Vec3;

/// Types of physics solvers that can be coupled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverType {
    /// Rigid body dynamics (ABA/CRBA/RNEA).
    RigidBody,
    /// Material point method (MPM).
    Particle,
    /// Electromagnetic fields (FDTD).
    Electromagnetic,
    /// Molecular dynamics (MD).
    MolecularDynamics,
    /// Quantum field theory / lattice gauge.
    QuantumField,
    /// Lattice Boltzmann method.
    LatticeBoltzmann,
}

/// Force transfer mechanism between solvers.
#[derive(Clone, Debug)]
pub enum ForceTransfer {
    /// Direct velocity damping: F_a = -k (v_a - v_b).
    Direct {
        /// Damping coefficient.
        damping: f64,
    },
    /// Momentum flux transfer at specified rate.
    Flux {
        /// Transfer rate (kg·m/s² or equivalent).
        rate: f64,
    },
    /// Effective potential barrier.
    Barrier {
        /// Potential energy function (stored as parameters).
        /// In practice, this would be a function pointer or trait object,
        /// but for simplicity we use a simple spring potential here.
        stiffness: f64,
        equilibrium: f64,
    },
}

impl ForceTransfer {
    /// Compute coupling force between two objects.
    ///
    /// # Arguments
    /// * `pos_a` - Position of object in solver A
    /// * `vel_a` - Velocity of object in solver A
    /// * `pos_b` - Position of object in solver B
    /// * `vel_b` - Velocity of object in solver B
    ///
    /// # Returns
    /// Force on object A (force on B is -force_a by Newton's 3rd law)
    pub fn compute_force(&self, pos_a: &Vec3, vel_a: &Vec3, pos_b: &Vec3, vel_b: &Vec3) -> Vec3 {
        match self {
            ForceTransfer::Direct { damping } => {
                // F_a = -k (v_a - v_b)
                -damping * (vel_a - vel_b)
            }
            ForceTransfer::Flux { rate } => {
                // Momentum transfer based on velocity difference
                let rel_vel = vel_a - vel_b;
                let dir = if rel_vel.norm() > 1e-10 {
                    rel_vel.normalize()
                } else {
                    Vec3::zeros()
                };
                -rate * dir
            }
            ForceTransfer::Barrier {
                stiffness,
                equilibrium,
            } => {
                // F = -k (r - r_eq)
                let r = pos_a - pos_b;
                let dist = r.norm();
                if dist > 1e-10 {
                    let delta = dist - equilibrium;
                    -stiffness * delta * (r / dist)
                } else {
                    Vec3::zeros()
                }
            }
        }
    }
}

/// Coupling definition between two solvers.
#[derive(Clone, Debug)]
pub struct Coupling {
    /// First solver type.
    pub solver_a: SolverType,
    /// Second solver type.
    pub solver_b: SolverType,
    /// Spatial overlap region where coupling is active.
    pub overlap_region: BoundingBox,
    /// Force transfer mechanism.
    pub force_transfer: ForceTransfer,
}

impl Coupling {
    /// Create a new coupling.
    pub fn new(
        solver_a: SolverType,
        solver_b: SolverType,
        overlap_region: BoundingBox,
        force_transfer: ForceTransfer,
    ) -> Self {
        Self {
            solver_a,
            solver_b,
            overlap_region,
            force_transfer,
        }
    }

    /// Check if a position is in the coupling region.
    pub fn is_in_region(&self, pos: &Vec3) -> bool {
        self.overlap_region.contains(pos)
    }

    /// Compute coupling force for objects at given positions/velocities.
    pub fn compute_coupling_force(
        &self,
        pos_a: &Vec3,
        vel_a: &Vec3,
        pos_b: &Vec3,
        vel_b: &Vec3,
    ) -> Option<Vec3> {
        // Only apply force if at least one object is in the overlap region
        if self.is_in_region(pos_a) || self.is_in_region(pos_b) {
            Some(
                self.force_transfer
                    .compute_force(pos_a, vel_a, pos_b, vel_b),
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_direct_force_transfer() {
        let transfer = ForceTransfer::Direct { damping: 2.0 };

        let pos_a = Vec3::new(0.0, 0.0, 0.0);
        let vel_a = Vec3::new(1.0, 0.0, 0.0);
        let pos_b = Vec3::new(1.0, 0.0, 0.0);
        let vel_b = Vec3::new(0.0, 0.0, 0.0);

        let force = transfer.compute_force(&pos_a, &vel_a, &pos_b, &vel_b);

        // F = -k (v_a - v_b) = -2.0 * (1.0, 0, 0) = (-2.0, 0, 0)
        assert_relative_eq!(force.x, -2.0);
        assert_relative_eq!(force.y, 0.0);
        assert_relative_eq!(force.z, 0.0);
    }

    #[test]
    fn test_barrier_force_transfer() {
        let transfer = ForceTransfer::Barrier {
            stiffness: 10.0,
            equilibrium: 1.0,
        };

        let pos_a = Vec3::new(0.0, 0.0, 0.0);
        let vel_a = Vec3::zeros();
        let pos_b = Vec3::new(2.0, 0.0, 0.0);
        let vel_b = Vec3::zeros();

        let force = transfer.compute_force(&pos_a, &vel_a, &pos_b, &vel_b);

        // Distance = 2.0, equilibrium = 1.0, delta = 1.0
        // r = pos_a - pos_b = (-2, 0, 0), direction = (-1, 0, 0)
        // F = -k * delta * direction = -10.0 * 1.0 * (-1, 0, 0) = (10, 0, 0)
        // (pulling pos_a toward pos_b)
        assert_relative_eq!(force.x, 10.0, epsilon = 1e-6);
        assert_relative_eq!(force.y, 0.0);
        assert_relative_eq!(force.z, 0.0);
    }

    #[test]
    fn test_coupling_region() {
        let coupling = Coupling::new(
            SolverType::RigidBody,
            SolverType::Electromagnetic,
            BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            ForceTransfer::Direct { damping: 1.0 },
        );

        assert!(coupling.is_in_region(&Vec3::new(0.0, 0.0, 0.0)));
        assert!(!coupling.is_in_region(&Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_coupling_force_computation() {
        let coupling = Coupling::new(
            SolverType::RigidBody,
            SolverType::Particle,
            BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            ForceTransfer::Direct { damping: 2.0 },
        );

        // Both in region
        let pos_a = Vec3::new(0.0, 0.0, 0.0);
        let vel_a = Vec3::new(1.0, 0.0, 0.0);
        let pos_b = Vec3::new(0.5, 0.0, 0.0);
        let vel_b = Vec3::zeros();

        let force = coupling
            .compute_coupling_force(&pos_a, &vel_a, &pos_b, &vel_b)
            .unwrap();
        assert_relative_eq!(force.x, -2.0);

        // Both outside region
        let pos_a_out = Vec3::new(5.0, 0.0, 0.0);
        let pos_b_out = Vec3::new(6.0, 0.0, 0.0);

        let force_out = coupling.compute_coupling_force(&pos_a_out, &vel_a, &pos_b_out, &vel_b);
        assert!(force_out.is_none());
    }
}
