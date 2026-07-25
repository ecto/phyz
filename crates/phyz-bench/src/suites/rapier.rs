//! Cross-library comparison against Rapier.
//!
//! Rapier is the obvious in-process comparison: same language, same process,
//! no FFI or interpreter in the measurement. The scenes here are built to
//! match [`crate::scenes`] as closely as two different formulations of rigid
//! body dynamics can be matched — same masses, same inertia tensors, same
//! link lengths, same initial angles, same timestep, same gravity, same
//! friction and restitution.
//!
//! What cannot be matched is stated in [`crate::settings::Settings::caveats`]
//! and repeated in every record's notes. In particular:
//!
//! - The pendulums use Rapier's **multibody** joint set, which like phyz's ABA
//!   is a reduced-coordinate formulation. This is the fairest available
//!   comparison.
//! - The box stack pits phyz's compliant penalty contact against Rapier's
//!   impulse solver. There is no setting that makes these the same algorithm.
//!   Read that row as "cost of each engine's own contact story on the same
//!   scene", not as a controlled experiment.
//!
//! Rapier is built here in its **f64** flavour (`rapier3d-f64`) so that
//! precision matches phyz. Stock `rapier3d` is f32 and would likely be
//! faster; that choice is documented rather than exploited in either
//! direction.

use crate::report::Suite;
use crate::timing::Budget;

const SUITE_NAME: &str = "cross-library comparison (phyz vs Rapier)";
const SUITE_DESC: &str = "Equivalent scenes, identical timestep, gravity, masses, inertias and \
     initial conditions. Rows where phyz loses are reported as-is. Rapier has \
     no gradient path, so the gradient dimension has no comparison row.";

#[cfg(not(feature = "rapier"))]
/// Run the comparison suite (disabled build: always reports skipped).
pub fn run(_budget: Budget) -> Suite {
    Suite::skipped(
        SUITE_NAME,
        SUITE_DESC,
        "built without the `rapier` feature (it is on by default; \
         `--no-default-features` disables it)",
    )
}

#[cfg(feature = "rapier")]
pub use enabled::run;

#[cfg(feature = "rapier")]
mod enabled {
    use super::*;
    use crate::report::{Metric, Record};
    use crate::scenes::{BOX_GAP, BOX_HALF_EXTENT, BOX_MASS, LINK_LENGTH, LINK_MASS, Scene};
    use crate::settings::{GRAVITY, Settings};
    use crate::suites::single_sim::{self, STEPS_PER_REP};
    use crate::timing::measure;
    use rapier3d_f64::prelude::*;

    /// A configured Rapier world, resettable between repetitions.
    struct RapierSim {
        bodies: RigidBodySet,
        colliders: ColliderSet,
        params: IntegrationParameters,
        pipeline: PhysicsPipeline,
        islands: IslandManager,
        broad_phase: DefaultBroadPhase,
        narrow_phase: NarrowPhase,
        impulse_joints: ImpulseJointSet,
        multibody_joints: MultibodyJointSet,
        ccd: CCDSolver,
        gravity: Vector,
    }

    impl RapierSim {
        fn new(settings: &Settings) -> Self {
            let params = IntegrationParameters {
                dt: settings.dt,
                // Match the solver effort we ask of every engine with the knob.
                num_solver_iterations: settings.solver_iterations,
                // Everything else stays at Rapier's own defaults: tuning them
                // toward phyz would be choosing the answer.
                ..IntegrationParameters::default()
            };
            Self {
                bodies: RigidBodySet::new(),
                colliders: ColliderSet::new(),
                params,
                pipeline: PhysicsPipeline::new(),
                islands: IslandManager::new(),
                broad_phase: DefaultBroadPhase::new(),
                narrow_phase: NarrowPhase::new(),
                impulse_joints: ImpulseJointSet::new(),
                multibody_joints: MultibodyJointSet::new(),
                ccd: CCDSolver::new(),
                gravity: Vector::new(0.0, 0.0, -GRAVITY),
            }
        }

        fn step(&mut self) {
            self.pipeline.step(
                self.gravity,
                &self.params,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd,
                &(),
                &(),
            );
        }

        fn steps(&mut self, n: usize) {
            for _ in 0..n {
                self.step();
            }
        }
    }

    /// Thin-rod mass properties matching `scenes::rod_inertia`: rod of length
    /// `LINK_LENGTH` hanging along −z from the joint.
    fn rod_mass_properties() -> MassProperties {
        let i = LINK_MASS * LINK_LENGTH * LINK_LENGTH / 12.0;
        MassProperties::new(
            Vector::new(0.0, 0.0, -LINK_LENGTH * 0.5),
            LINK_MASS,
            Vector::new(i, i, 0.0),
        )
    }

    /// Rotate `(0, 0, -LINK_LENGTH)` about +Y by `angle`.
    ///
    /// Closed form rather than a quaternion API call: the hinge axis is fixed,
    /// and this keeps the link placement provably identical to the phyz side.
    fn link_offset(angle: Real) -> Vector {
        Vector::new(-LINK_LENGTH * angle.sin(), 0.0, -LINK_LENGTH * angle.cos())
    }

    /// Build an `n`-link pendulum with Rapier multibody joints, hinged about
    /// +Y, at the same initial angles phyz uses.
    fn build_pendulum(settings: &Settings, angles: &[Real]) -> RapierSim {
        let mut sim = RapierSim::new(settings);

        let ground = sim
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(Vector::ZERO));

        let mut parent = ground;
        // Joint position and cumulative rotation as we walk down the chain.
        let mut joint_pos = Vector::ZERO;
        let mut cum_angle = 0.0;

        for (i, &angle) in angles.iter().enumerate() {
            cum_angle += angle;
            let link = sim.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(joint_pos)
                    // `rotation` takes a scaled-axis vector.
                    .rotation(Vector::Y * cum_angle)
                    // No collider on the pendulum links, so mass comes only
                    // from here and matches phyz exactly.
                    .additional_mass_properties(rod_mass_properties()),
            );

            // Anchor on the parent: the origin for the first link, the far end
            // of the previous link thereafter.
            let anchor1 = if i == 0 {
                Vector::ZERO
            } else {
                Vector::new(0.0, 0.0, -LINK_LENGTH)
            };
            let joint = RevoluteJointBuilder::new(Vector::Y)
                .local_anchor1(anchor1)
                .local_anchor2(Vector::ZERO);
            sim.multibody_joints.insert(parent, link, joint, true);

            joint_pos += link_offset(cum_angle);
            parent = link;
        }
        sim
    }

    /// Build the box stack: `n` cuboids of `BOX_MASS` over a fixed ground.
    fn build_box_stack(settings: &Settings, n: usize) -> RapierSim {
        let mut sim = RapierSim::new(settings);
        let contact = settings
            .contact
            .as_ref()
            .expect("box stack requires contact settings");

        // Ground: a large thin slab whose top face sits exactly at z = 0, so
        // the plane matches phyz's ground query.
        let ground = sim
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.0, -0.5)));
        sim.colliders.insert_with_parent(
            ColliderBuilder::cuboid(10.0, 10.0, 0.5)
                .friction(contact.friction)
                .restitution(contact.restitution),
            ground,
            &mut sim.bodies,
        );

        // Density chosen so each box masses exactly BOX_MASS, matching phyz.
        let volume = (2.0 * BOX_HALF_EXTENT).powi(3);
        let density = BOX_MASS / volume;

        for i in 0..n {
            let z = BOX_HALF_EXTENT + i as f64 * (2.0 * BOX_HALF_EXTENT + BOX_GAP);
            let body = sim
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, z)));
            sim.colliders.insert_with_parent(
                ColliderBuilder::cuboid(BOX_HALF_EXTENT, BOX_HALF_EXTENT, BOX_HALF_EXTENT)
                    .density(density)
                    .friction(contact.friction)
                    .restitution(contact.restitution),
                body,
                &mut sim.bodies,
            );
        }
        sim
    }

    /// Construct the Rapier equivalent of a phyz scene, or explain why there
    /// isn't one.
    fn build(scene: Scene, settings: &Settings) -> Result<RapierSim, String> {
        Ok(match scene {
            Scene::Pendulum => build_pendulum(settings, &[1.0]),
            Scene::DoublePendulum => build_pendulum(settings, &[1.0, 0.5]),
            Scene::BoxStack(n) => build_box_stack(settings, n),
            Scene::Ant => {
                return Err(
                    "the ant is loaded from MJCF, which Rapier does not parse in-tree; \
                     building it by hand would risk an unequal model, so no comparison row \
                     is published for this scene"
                        .into(),
                );
            }
        })
    }

    /// Number of degrees of freedom in the Rapier scene, for the table.
    fn dof(scene: Scene) -> usize {
        match scene {
            Scene::Pendulum => 1,
            Scene::DoublePendulum => 2,
            Scene::BoxStack(n) => 6 * n,
            Scene::Ant => 0,
        }
    }

    /// Run the comparison over every scene that has a Rapier equivalent.
    pub fn run(budget: Budget) -> Suite {
        let mut results = Vec::new();

        for scene in crate::suites::standard_scenes() {
            let settings = single_sim::settings_for(scene);

            // phyz side, measured identically to the single-sim suite.
            let mut phyz_record = single_sim::run_scene(scene, budget);
            let phyz_tput = phyz_record
                .timing
                .as_ref()
                .map(|t| t.throughput_per_sec)
                .unwrap_or(f64::NAN);

            let rapier_sim = match build(scene, &settings) {
                Ok(s) => s,
                Err(why) => {
                    phyz_record
                        .notes
                        .push(format!("No Rapier comparison for this scene: {why}"));
                    results.push(phyz_record);
                    continue;
                }
            };
            drop(rapier_sim); // built once above only to validate the scene

            let timing = measure(budget, STEPS_PER_REP, || {
                // Rebuild per repetition: Rapier's islands sleep, and a
                // reused world would let later repetitions measure a
                // slumbering scene.
                let mut sim = build(scene, &settings).expect("scene built above");
                sim.steps(STEPS_PER_REP as usize);
                sim.bodies.len()
            });

            let ratio = phyz_tput / timing.throughput_per_sec;
            phyz_record
                .metrics
                .push(Metric::new("phyz_vs_rapier", ratio, "×"));
            results.push(phyz_record);

            results.push(Record {
                engine: "rapier3d-f64".into(),
                scene: scene.name(),
                description: scene.description(),
                dof: Some(dof(scene)),
                batch: Some(1),
                settings: settings.clone(),
                timing: Some(timing),
                metrics: vec![Metric::new("phyz_vs_rapier", ratio, "×")],
                notes: vec![
                    "`phyz_vs_rapier` > 1 means phyz is faster; < 1 means Rapier is faster. \
                     The same value is repeated on both rows of a pair."
                        .into(),
                    "Rapier world is rebuilt for each timed repetition, because Rapier puts \
                     settled islands to sleep and a reused world would flatter it."
                        .into(),
                    "Built as `rapier3d-f64` so precision matches phyz. The stock f32 \
                     `rapier3d` build would likely be faster than these numbers."
                        .into(),
                    if scene.has_contact() {
                        "Contact scene: Rapier's impulse solver vs phyz's penalty contact. \
                         Same friction, restitution, dt and geometry; fundamentally \
                         different algorithms."
                            .into()
                    } else {
                        "Pendulum scenes use Rapier's multibody (reduced-coordinate) joint \
                         set, the closest analogue to phyz's ABA."
                            .to_string()
                    },
                ],
            });
        }

        Suite::new(SUITE_NAME, SUITE_DESC, results)
    }
}
