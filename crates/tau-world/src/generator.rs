//! Procedural world generation for creating random articulated systems.

use rand::prelude::*;
use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::{Model, ModelBuilder};

/// Procedural world generator with seeded randomness.
pub struct WorldGenerator {
    #[allow(dead_code)]
    seed: u64,
    rng: StdRng,
}

impl WorldGenerator {
    /// Create a new world generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a random kinematic chain with n links.
    ///
    /// Each link has randomized mass and length within the given ranges.
    /// All joints are revolute joints about the Z axis.
    pub fn random_chain(
        &mut self,
        nlinkages: usize,
        mass_range: [f64; 2],
        length_range: [f64; 2],
    ) -> Model {
        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001);

        for i in 0..nlinkages {
            let parent = if i == 0 { -1 } else { (i - 1) as i32 };
            let mass = self.rng.gen_range(mass_range[0]..=mass_range[1]);
            let length = self.rng.gen_range(length_range[0]..=length_range[1]);

            let parent_to_joint = if i == 0 {
                SpatialTransform::identity()
            } else {
                // Previous link's endpoint
                SpatialTransform::translation(Vec3::new(0.0, 0.0, -length))
            };

            // Cylinder inertia: I_xx = I_yy = m*L^2/12 + m*(L/2)^2 = m*L^2/3
            // I_zz = m*r^2/2 ≈ 0 for thin rod
            let com = Vec3::new(0.0, 0.0, -length / 2.0);
            let inertia_val = mass * length * length / 3.0;
            let inertia = SpatialInertia::new(
                mass,
                com,
                Mat3::from_diagonal(&Vec3::new(inertia_val, inertia_val, 0.0)),
            );

            builder =
                builder.add_revolute_body(&format!("link{}", i), parent, parent_to_joint, inertia);
        }

        builder.build()
    }

    /// Generate a random quadruped robot.
    ///
    /// Creates a body with 4 legs, each leg having 2 revolute joints.
    /// Returns a model with 1 body + 4*2 = 8 linkages (9 bodies total including torso).
    pub fn random_quadruped(&mut self) -> Model {
        let body_mass = 5.0;
        let body_size = 0.4;
        let leg_mass = 0.5;
        let leg_length = 0.3;

        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001);

        // Torso (free-floating body)
        let torso_inertia = SpatialInertia::new(
            body_mass,
            Vec3::zeros(),
            Mat3::from_diagonal(&Vec3::new(
                body_mass * body_size * body_size / 6.0,
                body_mass * body_size * body_size / 6.0,
                body_mass * body_size * body_size / 6.0,
            )),
        );
        builder = builder.add_free_body("torso", -1, SpatialTransform::identity(), torso_inertia);

        // Four legs: each leg has 2 segments
        let leg_positions = [
            Vec3::new(body_size / 2.0, body_size / 2.0, 0.0),
            Vec3::new(body_size / 2.0, -body_size / 2.0, 0.0),
            Vec3::new(-body_size / 2.0, body_size / 2.0, 0.0),
            Vec3::new(-body_size / 2.0, -body_size / 2.0, 0.0),
        ];

        for (leg_idx, leg_pos) in leg_positions.iter().enumerate() {
            // Upper leg segment attached to torso
            let upper_leg_inertia = SpatialInertia::new(
                leg_mass,
                Vec3::new(0.0, 0.0, -leg_length / 2.0),
                Mat3::from_diagonal(&Vec3::new(
                    leg_mass * leg_length * leg_length / 12.0,
                    leg_mass * leg_length * leg_length / 12.0,
                    0.0,
                )),
            );
            let upper_joint = SpatialTransform::translation(*leg_pos);
            builder = builder.add_revolute_body(
                &format!("leg{}_upper", leg_idx),
                0, // parent is torso (body 0)
                upper_joint,
                upper_leg_inertia,
            );

            // Lower leg segment attached to upper leg
            let lower_leg_inertia = upper_leg_inertia;
            let lower_joint = SpatialTransform::translation(Vec3::new(0.0, 0.0, -leg_length));
            let upper_body_idx = 1 + leg_idx * 2;
            builder = builder.add_revolute_body(
                &format!("leg{}_lower", leg_idx),
                upper_body_idx as i32,
                lower_joint,
                lower_leg_inertia,
            );
        }

        builder.build()
    }

    /// Generate a platform with randomly placed obstacles.
    ///
    /// Creates a flat platform (fixed body) with n obstacles of varying heights.
    /// Obstacles are modeled as fixed bodies with box collision geometry.
    pub fn platform_with_obstacles(
        &mut self,
        width: f64,
        nobs: usize,
        obs_height_range: [f64; 2],
    ) -> Model {
        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001);

        // Platform (fixed body at origin)
        let platform_inertia = SpatialInertia::new(
            1000.0, // large mass for stability (won't move anyway)
            Vec3::zeros(),
            Mat3::identity(),
        );
        builder = builder.add_fixed_body(
            "platform",
            -1,
            SpatialTransform::identity(),
            platform_inertia,
        );

        // Random obstacles
        for i in 0..nobs {
            let x = self.rng.gen_range(-width / 2.0..width / 2.0);
            let y = self.rng.gen_range(-width / 2.0..width / 2.0);
            let height = self
                .rng
                .gen_range(obs_height_range[0]..=obs_height_range[1]);

            let obs_pos = Vec3::new(x, y, height / 2.0);
            let obs_inertia = SpatialInertia::new(
                10.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
            );

            builder = builder.add_fixed_body(
                &format!("obstacle{}", i),
                0, // parent is platform
                SpatialTransform::translation(obs_pos),
                obs_inertia,
            );
        }

        builder.build()
    }

    /// Generate a random tree-like structure.
    ///
    /// Creates a branching kinematic tree with random splits.
    /// Useful for testing tree traversal algorithms.
    pub fn random_tree(&mut self, depth: usize, branching_factor: usize) -> Model {
        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001);

        let mass = 1.0;
        let length = 0.5;

        // Build tree level by level
        let mut body_count: i32 = 0;
        let mut current_level = vec![(None, 0)]; // (parent_idx, level)

        for level in 0..=depth {
            let mut next_level = Vec::new();

            for (parent_opt, _) in current_level {
                let num_children = if level == 0 { 1 } else { branching_factor };

                for i in 0..num_children {
                    let parent_idx = parent_opt.unwrap_or(-1);

                    let (offset, name) = if level == 0 {
                        (SpatialTransform::identity(), "root".to_string())
                    } else {
                        let angle = 2.0 * std::f64::consts::PI * i as f64 / branching_factor as f64;
                        let offset = Vec3::new(
                            length * angle.cos() * 0.3,
                            length * angle.sin() * 0.3,
                            -length,
                        );
                        (
                            SpatialTransform::translation(offset),
                            format!("node_{}_{}", body_count, i),
                        )
                    };

                    let inertia = SpatialInertia::new(
                        mass * 0.8_f64.powi(level as i32),
                        Vec3::new(0.0, 0.0, -length / 2.0),
                        Mat3::from_diagonal(&Vec3::new(
                            0.1 * 0.8_f64.powi(level as i32),
                            0.1 * 0.8_f64.powi(level as i32),
                            0.0,
                        )),
                    );

                    builder = builder.add_revolute_body(&name, parent_idx, offset, inertia);

                    if level < depth {
                        next_level.push((Some(body_count), level + 1));
                    }

                    body_count += 1;
                }
            }

            current_level = next_level;
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_chain() {
        let mut generator = WorldGenerator::new(42);
        let model = generator.random_chain(5, [0.5, 2.0], [0.3, 1.5]);
        assert_eq!(model.nbodies(), 5);
        assert_eq!(model.nq, 5);
        assert_eq!(model.nv, 5);
    }

    #[test]
    fn test_random_quadruped() {
        let mut generator = WorldGenerator::new(42);
        let model = generator.random_quadruped();
        // 1 torso + 4 legs * 2 segments = 9 bodies
        assert_eq!(model.nbodies(), 9);
        // torso is free (6 DOF) + 8 revolute joints (1 DOF each) = 14 DOF
        assert_eq!(model.nv, 14);
    }

    #[test]
    fn test_platform_with_obstacles() {
        let mut generator = WorldGenerator::new(42);
        let model = generator.platform_with_obstacles(10.0, 5, [0.5, 2.0]);
        // 1 platform + 5 obstacles = 6 bodies
        assert_eq!(model.nbodies(), 6);
        // All fixed joints = 0 DOF
        assert_eq!(model.nv, 0);
    }

    #[test]
    fn test_random_tree() {
        let mut generator = WorldGenerator::new(42);
        let model = generator.random_tree(2, 2);
        // 1 root + 2 children + 4 grandchildren = 7 bodies
        assert_eq!(model.nbodies(), 7);
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = WorldGenerator::new(123);
        let mut gen2 = WorldGenerator::new(123);

        let m1 = gen1.random_chain(3, [0.5, 1.5], [0.5, 1.0]);
        let m2 = gen2.random_chain(3, [0.5, 1.5], [0.5, 1.0]);

        // Same seed should produce same masses
        for i in 0..3 {
            assert_eq!(m1.bodies[i].inertia.mass, m2.bodies[i].inertia.mass);
        }
    }
}
