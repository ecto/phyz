//! Contact dynamics and soft contact resolution for tau physics engine.
//!
//! Implements MuJoCo-style soft contacts using penalty forces.

pub mod material;
pub mod solver;

pub use material::ContactMaterial;
pub use solver::{contact_forces, find_contacts};

use tau_collision::Collision;
use tau_math::{SpatialVec, Vec3};

/// Compute contact force for a single collision.
pub fn compute_contact_force(
    collision: &Collision,
    material: &ContactMaterial,
    velocity_i: &Vec3,
    velocity_j: &Vec3,
) -> SpatialVec {
    let depth = collision.penetration_depth;
    if depth <= 0.0 {
        return SpatialVec::zero();
    }

    let normal = collision.contact_normal;
    let rel_vel = velocity_j - velocity_i;
    let normal_vel = rel_vel.dot(&normal);

    // Penalty force: F = k * depth^p - c * v_n
    let k = material.stiffness;
    let c = material.damping;
    let p = 1.0; // Linear spring for now

    let force_magnitude = k * depth.powf(p) - c * normal_vel;
    let force_magnitude = force_magnitude.max(0.0); // No pulling

    let force = normal * force_magnitude;

    // Friction
    let tangent_vel = rel_vel - normal * normal_vel;
    let tangent_speed = tangent_vel.norm();
    let friction_force = if tangent_speed > 1e-10 {
        let tangent_dir = tangent_vel / tangent_speed;
        let friction_magnitude = (material.friction * force_magnitude).min(c * tangent_speed);
        -tangent_dir * friction_magnitude
    } else {
        Vec3::zeros()
    };

    let total_force = force + friction_force;

    // Convert to spatial force (wrench) at contact point
    // τ = r × F, where r is from body COM to contact point
    // For now, assume force applied at body origin (simplified)
    SpatialVec::from_linear_angular(total_force, Vec3::zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_force_zero_depth() {
        let collision = Collision {
            body_i: 0,
            body_j: 1,
            contact_point: Vec3::zeros(),
            contact_normal: Vec3::z(),
            penetration_depth: 0.0,
        };
        let material = ContactMaterial::default();
        let force = compute_contact_force(&collision, &material, &Vec3::zeros(), &Vec3::zeros());
        assert!(force.linear().norm() < 1e-10);
    }

    #[test]
    fn test_contact_force_penetration() {
        let collision = Collision {
            body_i: 0,
            body_j: 1,
            contact_point: Vec3::zeros(),
            contact_normal: Vec3::z(),
            penetration_depth: 0.1,
        };
        let material = ContactMaterial {
            stiffness: 1000.0,
            ..Default::default()
        };
        let force = compute_contact_force(&collision, &material, &Vec3::zeros(), &Vec3::zeros());
        assert!(force.linear().norm() > 0.0);
        assert!(force.linear().dot(&Vec3::z()) > 0.0);
    }
}
