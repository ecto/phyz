//! Contact dynamics and soft contact resolution for phyz physics engine.
//!
//! Implements MuJoCo-style soft contacts using penalty forces.

pub mod material;
pub mod solver;

pub use material::ContactMaterial;
pub use solver::{contact_forces, contact_forces_implicit, find_contacts, find_ground_contacts};

use phyz_collision::Collision;
use phyz_math::{SpatialVec, Vec3};

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
    SpatialVec::new(Vec3::zeros(), total_force)
}

/// Compute a contact force using implicit damping (and implicit stiffness)
/// based on the post-step velocity of the contact pair.
///
/// See `phyz::contact::compute_contact_force_implicit` for the derivation.
/// `mass_i` / `mass_j` are the effective contact masses of the pair; use
/// `f64::INFINITY` for the world or any body that cannot translate.
pub fn compute_contact_force_implicit(
    collision: &Collision,
    material: &ContactMaterial,
    velocity_i: &Vec3,
    velocity_j: &Vec3,
    mass_i: f64,
    mass_j: f64,
    dt: f64,
) -> SpatialVec {
    let depth = collision.penetration_depth;
    if depth <= 0.0 {
        return SpatialVec::zero();
    }

    let normal = collision.contact_normal;
    let rel_vel = velocity_j - velocity_i;
    let normal_vel = rel_vel.dot(&normal);

    let k = material.stiffness;
    let c = material.damping;

    let m_eff = if !mass_i.is_finite() && !mass_j.is_finite() {
        return SpatialVec::zero();
    } else if !mass_i.is_finite() {
        mass_j
    } else if !mass_j.is_finite() {
        mass_i
    } else if mass_i + mass_j > 0.0 {
        (mass_i * mass_j) / (mass_i + mass_j)
    } else {
        return SpatialVec::zero();
    };

    if m_eff <= 0.0 || !m_eff.is_finite() {
        return SpatialVec::zero();
    }

    // One-step Newton with implicit stiffness and damping. See
    // `phyz::contact::compute_contact_force_implicit` for the derivation.
    let denom = m_eff + dt * c + dt * dt * k;
    let force_magnitude = m_eff * (k * depth - (c + dt * k) * normal_vel) / denom;
    let force_magnitude = force_magnitude.max(0.0);

    let force = normal * force_magnitude;

    let tangent_vel = rel_vel - normal * normal_vel;
    let tangent_speed = tangent_vel.norm();
    let friction_force = if tangent_speed > 1e-10 {
        let tangent_dir = tangent_vel / tangent_speed;
        let friction_magnitude = (material.friction * force_magnitude).min(c * tangent_speed);
        -tangent_dir * friction_magnitude
    } else {
        Vec3::zeros()
    };

    SpatialVec::new(Vec3::zeros(), force + friction_force)
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
        assert!(force.linear.norm() < 1e-10);
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
        assert!(force.linear.norm() > 0.0);
        assert!(force.linear.dot(&Vec3::z()) > 0.0);
    }
}
