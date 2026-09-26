//! Ground-plane contact.
//!
//! A soft (penalty) contact against a horizontal plane. Three properties matter
//! more than sophistication here, and each one fixed a real divergence:
//!
//! 1. **Geom offsets are respected.** MJCF geoms sit where `pos`/`fromto` put
//!    them, usually well below the body origin. Testing the body origin instead
//!    makes a cheetah collide with its hips while its feet pass through the
//!    floor.
//! 2. **Shapes contact at several points.** A capsule is exactly a segment
//!    Minkowski-summed with a sphere, so a plane contact is exactly two sphere
//!    contacts at its endpoints. Collapsing it to one point puts the contact at
//!    an end cap, which gives a horizontal limb a half-metre lever arm and
//!    flings it.
//! 3. **Stiffness is derived, not tuned.** Parameters are a response *time
//!    constant* and damping ratio, following MuJoCo's `solref`, with spring
//!    constants computed per body from its mass. A fixed stiffness is stable
//!    for a torso and explosive for a toe.
//!
//! It is still a penalty model, not a constraint solver: deep penetrations
//! resolve over a few steps rather than exactly. `phyz-contact` (CPU) and
//! `phyz_gpu::ContactPipeline` (GPU) remain the path to a real solver.

use phyz_math::{SpatialTransform, SpatialTransformExt, SpatialVec, Vec3};
use phyz_model::{Geometry, Model};

/// Ground contact parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundContact {
    /// Plane height along world Z.
    pub height: f64,
    /// Contact response time constant, seconds. Must be comfortably larger
    /// than the timestep; MuJoCo's default is `0.02`.
    pub time_const: f64,
    /// Damping ratio. `1.0` is critically damped, which is what you want for a
    /// foot that should neither bounce nor sink.
    pub damp_ratio: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Fraction of the remaining penetration corrected per step (Baumgarte
    /// error-reduction parameter).
    ///
    /// Correcting the whole penetration in one step massively overshoots for a
    /// link in a kinematic chain, because the ground sees the chain's
    /// *articulated* effective mass, not the link's own. Correcting a fraction
    /// converges monotonically instead of ringing.
    pub erp: f64,
    /// Simulation timestep, used to bound the contact impulse. Must match
    /// `Model::dt`; [`crate::BatchEnv`] keeps it in sync for you.
    pub dt: f64,
}

impl Default for GroundContact {
    fn default() -> Self {
        Self {
            height: 0.0,
            time_const: 0.02,
            damp_ratio: 1.0,
            friction: 0.8,
            erp: 0.2,
            dt: 0.002,
        }
    }
}

/// The most contact points any supported shape generates (a box's 8 corners).
const MAX_POINTS: usize = 8;

/// Cap on contact points considered per body across all of its collision
/// shapes. Bodies with more shapes than this contribute their first
/// `MAX_TOTAL_POINTS` points; the alternative is a heap allocation in the
/// innermost simulation loop.
const MAX_TOTAL_POINTS: usize = 32;

/// One sphere in the sphere-decomposition of a geom, in geom-local coordinates.
#[derive(Debug, Clone, Copy, Default)]
struct ContactPoint {
    pos: Vec3,
    radius: f64,
}

impl GroundContact {
    /// Accumulate body-frame spatial forces from ground penetration.
    ///
    /// `xform[i].pos` is the body origin in world coordinates and `xform[i].rot`
    /// is the world→body rotation (see `phyz-rigid/tests/frame_conventions.rs`).
    /// `vel[i]` is the body-frame spatial velocity. Output is in the frame
    /// `aba_with_external_forces` expects.
    pub fn forces(
        &self,
        model: &Model,
        xform: &[SpatialTransform],
        vel: &[SpatialVec],
        out: &mut [SpatialVec],
    ) {
        for f in out.iter_mut() {
            *f = SpatialVec::zero();
        }

        for (i, body) in model.bodies.iter().enumerate() {
            if body.collisions.is_empty() {
                continue;
            }

            let body_to_world = xform[i].rot.transpose();

            // Gather every penetrating point across all of the body's collision
            // shapes before choosing stiffness, so a body resting on two feet is
            // held by the same total force as one balanced on one.
            let mut depth = [0.0f64; MAX_TOTAL_POINTS];
            let mut r_body = [Vec3::zeros(); MAX_TOTAL_POINTS];
            let mut active = 0usize;
            let mut total = 0usize;

            for inst in &body.collisions {
                // `origin.rot` is the body → shape transform (same convention as
                // `parent_to_joint`), so shape → body is its transpose.
                let shape_to_body = inst.origin.rot.transpose();
                let geom_to_world = body_to_world * shape_to_body;
                let geom_center = xform[i].pos + body_to_world * inst.origin.pos;

                let mut points = [ContactPoint::default(); MAX_POINTS];
                let n = contact_points(&inst.geometry, &mut points);

                for p in points.iter().take(n) {
                    if total >= MAX_TOTAL_POINTS {
                        break;
                    }
                    let p_world = geom_center + geom_to_world * p.pos;
                    let pen = self.height + p.radius - p_world.z;
                    if pen > 0.0 {
                        depth[total] = pen;
                        r_body[total] = xform[i].world_to_body_dir(
                            body_to_world * inst.origin.pos + geom_to_world * p.pos,
                        );
                        active += 1;
                    }
                    total += 1;
                }
            }
            if active == 0 {
                continue;
            }

            let m = body.inertia.mass.max(1e-6) / active as f64;
            let omega = 1.0 / self.time_const.max(1e-6);
            let k_spring = m * omega * omega;
            let c_damp = 2.0 * m * omega * self.damp_ratio;

            for idx in 0..total {
                let penetration = depth[idx];
                if penetration <= 0.0 {
                    continue;
                }
                let r = r_body[idx];

                // Velocity of the contact point: v_p = R (v_o + ω × r).
                let v_body = vel[i].linear + vel[i].angular.cross(r);
                let v_world = body_to_world * v_body;

                let spring = k_spring * penetration - c_damp * v_world.z;

                // Impulse feasibility bound: the force that arrests the approach
                // and recovers `erp` of the penetration this step. Anything
                // beyond it can only overshoot, and the overshoot is what makes
                // light distal links whip and diverge.
                let target_dv = self.erp * penetration / self.dt - v_world.z;
                let f_max = if target_dv > 0.0 {
                    m * target_dv / self.dt
                } else {
                    0.0
                };
                let normal_mag = spring.clamp(0.0, f_max);
                if normal_mag <= 0.0 {
                    continue;
                }

                let tangent = Vec3::new(v_world.x, v_world.y, 0.0);
                let speed = (tangent.x * tangent.x + tangent.y * tangent.y).sqrt();
                let friction_world = if speed > 1e-9 {
                    tangent * (-(self.friction * normal_mag) / speed)
                } else {
                    Vec3::zeros()
                };

                let f_world = Vec3::new(friction_world.x, friction_world.y, normal_mag);
                let f_body = xform[i].world_to_body_dir(f_world);
                out[i] = out[i] + SpatialVec::new(r.cross(f_body), f_body);
            }
        }
    }
}

/// Decompose a geom into spheres for plane contact, writing into `out`.
///
/// Returns how many were written. This is **exact** for spheres, capsules and
/// boxes against a plane; a cylinder is approximated by its two rim circles
/// sampled at four points each, since a disc edge has no finite sphere
/// decomposition.
fn contact_points(geom: &Geometry, out: &mut [ContactPoint; MAX_POINTS]) -> usize {
    match geom {
        Geometry::Sphere { radius } => {
            out[0] = ContactPoint {
                pos: Vec3::zeros(),
                radius: *radius,
            };
            1
        }

        // A capsule is the segment [-h, +h] along local Z swept by a sphere of
        // `radius`, so two end spheres reproduce it exactly.
        Geometry::Capsule { radius, length } => {
            let h = length * 0.5;
            out[0] = ContactPoint {
                pos: Vec3::new(0.0, 0.0, -h),
                radius: *radius,
            };
            out[1] = ContactPoint {
                pos: Vec3::new(0.0, 0.0, h),
                radius: *radius,
            };
            2
        }

        Geometry::Cylinder { radius, height } => {
            let h = height * 0.5;
            let mut n = 0;
            for &z in &[-h, h] {
                for &(x, y) in &[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    out[n] = ContactPoint {
                        pos: Vec3::new(x * radius, y * radius, z),
                        radius: 0.0,
                    };
                    n += 1;
                }
            }
            n
        }

        // A box's support against a plane is always a corner, so the eight
        // corners cover every orientation exactly.
        Geometry::Box { half_extents } => {
            let mut n = 0;
            for &sx in &[-1.0, 1.0] {
                for &sy in &[-1.0, 1.0] {
                    for &sz in &[-1.0, 1.0] {
                        out[n] = ContactPoint {
                            pos: Vec3::new(
                                sx * half_extents.x,
                                sy * half_extents.y,
                                sz * half_extents.z,
                            ),
                            radius: 0.0,
                        };
                        n += 1;
                    }
                }
            }
            n
        }

        // A plane is the ground, not something resting on it; meshes need a
        // real narrow phase.
        Geometry::Mesh { .. } | Geometry::Plane { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialInertia};
    use phyz_model::ModelBuilder;

    fn sphere_body(mass: f64, radius: f64) -> Model {
        let mut m = ModelBuilder::new()
            .add_free_body(
                "b",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(mass, Vec3::zeros(), Mat3::identity() * 0.1),
            )
            .build();
        m.bodies[0].geometry = Some(Geometry::Sphere { radius });
        m.bodies[0].collisions = vec![phyz_model::GeomInstance::centered(Geometry::Sphere {
            radius,
        })];
        m
    }

    fn at(z: f64) -> Vec<SpatialTransform> {
        vec![SpatialTransform::new(
            Mat3::identity(),
            Vec3::new(0.0, 0.0, z),
        )]
    }

    #[test]
    fn capsule_contacts_at_both_endpoints() {
        let mut out = [ContactPoint::default(); MAX_POINTS];
        let n = contact_points(
            &Geometry::Capsule {
                radius: 0.1,
                length: 1.0,
            },
            &mut out,
        );
        assert_eq!(n, 2);
        assert!((out[0].pos.z + 0.5).abs() < 1e-12);
        assert!((out[1].pos.z - 0.5).abs() < 1e-12);
    }

    /// The bug this decomposition exists to fix: a horizontal capsule resting
    /// on the ground must produce (near) zero net torque about its own centre,
    /// not a lever arm equal to half its length.
    #[test]
    fn horizontal_capsule_produces_no_net_torque() {
        let mut m = ModelBuilder::new()
            .add_free_body(
                "b",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(4.0, Vec3::zeros(), Mat3::identity() * 0.5),
            )
            .build();
        // Lay the capsule's local Z along world X.
        // Lay the capsule's local Z along world X. `origin.rot` is body→shape,
        // so the shape→body rotation goes in transposed.
        let shape_to_body = Mat3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0);
        m.bodies[0].collisions = vec![phyz_model::GeomInstance::new(
            Geometry::Capsule {
                radius: 0.05,
                length: 1.0,
            },
            SpatialTransform::new(shape_to_body.transpose(), Vec3::zeros()),
        )];

        let g = GroundContact {
            dt: 0.002,
            ..Default::default()
        };
        let mut f = vec![SpatialVec::zero()];
        g.forces(&m, &at(0.04), &[SpatialVec::zero()], &mut f);

        assert!(f[0].linear.z > 0.0, "capsule should be pushed up");
        let t = f[0].angular;
        let mag = (t.x * t.x + t.y * t.y + t.z * t.z).sqrt();
        assert!(
            mag < 1e-9,
            "symmetric contact must cancel torque, got {mag:.3e} ({t:?})"
        );
    }

    #[test]
    fn geom_offset_moves_the_contact_point() {
        let mut with_offset = sphere_body(1.0, 0.05);
        with_offset.bodies[0].collisions = vec![phyz_model::GeomInstance::new(
            Geometry::Sphere { radius: 0.05 },
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, -0.5)),
        )];
        let no_offset = sphere_body(1.0, 0.05);

        let g = GroundContact::default();
        let xf = at(0.3);
        let v = [SpatialVec::zero()];

        let mut f_off = vec![SpatialVec::zero()];
        g.forces(&with_offset, &xf, &v, &mut f_off);
        let mut f_no = vec![SpatialVec::zero()];
        g.forces(&no_offset, &xf, &v, &mut f_no);

        assert!(f_off[0].linear.z > 0.0, "offset geom must touch the ground");
        assert_eq!(f_no[0].linear.z, 0.0, "un-offset geom is above the ground");
    }

    /// Stiffness scales with mass, so heavy and light bodies reach the same
    /// resting penetration instead of the light one exploding.
    #[test]
    fn stiffness_scales_with_mass() {
        let g = GroundContact::default();
        let force_for = |mass: f64| {
            let m = sphere_body(mass, 0.1);
            let mut f = vec![SpatialVec::zero()];
            g.forces(&m, &at(0.05), &[SpatialVec::zero()], &mut f);
            f[0].linear.z
        };
        let light = force_for(1.0);
        let heavy = force_for(10.0);
        assert!((heavy / light - 10.0).abs() < 1e-9, "{light} vs {heavy}");
    }

    #[test]
    fn no_force_when_clear_of_the_ground() {
        let g = GroundContact::default();
        let m = sphere_body(1.0, 0.1);
        let mut f = vec![SpatialVec::zero()];
        g.forces(&m, &at(5.0), &[SpatialVec::zero()], &mut f);
        assert_eq!(f[0].linear.z, 0.0);
    }

    /// Normal force is never attractive, no matter how fast the body is moving
    /// away from the plane.
    #[test]
    fn normal_force_is_never_negative() {
        let g = GroundContact::default();
        let m = sphere_body(1.0, 0.1);
        let fast_up = [SpatialVec::new(Vec3::zeros(), Vec3::new(0.0, 0.0, 50.0))];
        let mut f = vec![SpatialVec::zero()];
        g.forces(&m, &at(0.05), &fast_up, &mut f);
        assert!(f[0].linear.z >= 0.0, "got {}", f[0].linear.z);
    }

    #[test]
    fn box_contacts_at_its_eight_corners() {
        let mut out = [ContactPoint::default(); MAX_POINTS];
        let n = contact_points(
            &Geometry::Box {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
            },
            &mut out,
        );
        assert_eq!(n, 8);
        assert!(out[..8].iter().all(|p| p.pos.x.abs() == 1.0));
    }
}
