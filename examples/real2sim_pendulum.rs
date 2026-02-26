//! Real2Sim example: pendulum parameter estimation from trajectory data.
//!
//! This example demonstrates:
//! 1. Generating reference trajectory from a known pendulum model
//! 2. Creating observations from the trajectory
//! 3. Estimating parameters (mass, length) from observations
//! 4. Exporting the identified model to .phyz format

use phyz_format::{export_phyz, schema::PhyzSpec};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_real2sim::{
    GradientDescentOptimizer, LossWeights, Optimizer, OptimizerConfig, PhysicsParams, Trajectory,
    TrajectoryMatcher, TrajectoryObservation,
};
use phyz_rigid::{aba, forward_kinematics};

fn main() {
    println!("=== Real2Sim Pendulum Parameter Estimation ===\n");

    // Step 1: Create "ground truth" pendulum with known parameters
    let true_length = 1.0;
    let true_mass = 1.5;

    let true_model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                true_mass,
                Vec3::new(0.0, -true_length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    true_mass * true_length * true_length / 12.0,
                    0.0,
                    true_mass * true_length * true_length / 12.0,
                )),
            ),
        )
        .build();

    println!(
        "Ground truth parameters: mass = {:.2} kg, length = {:.2} m",
        true_mass, true_length
    );

    // Step 2: Generate reference trajectory by simulating the true model
    let mut reference = Trajectory::new();
    let mut state = true_model.default_state();
    state.q[0] = 0.5; // Initial angle
    state.v[0] = 0.0;

    println!("Generating reference trajectory...");
    for step in 0..100 {
        let (xforms, _) = forward_kinematics(&true_model, &state);
        state.body_xform = xforms;

        // Record observation every 10 steps
        if step % 10 == 0 {
            reference.add_observation(TrajectoryObservation {
                time: state.time,
                body_idx: 0,
                position: Some([
                    state.body_xform[0].pos.x,
                    state.body_xform[0].pos.y,
                    state.body_xform[0].pos.z,
                ]),
                orientation: None,
                joint_angles: Some(vec![state.q[0]]),
            });
        }

        // Step simulation
        let qdd = aba(&true_model, &state);
        state.v += &(&qdd * true_model.dt);
        let v_tmp = state.v.clone();
        state.q += &(&v_tmp * true_model.dt);
        state.time += true_model.dt;
    }

    println!(
        "Generated {} observations over {:.3} seconds\n",
        reference.len(),
        reference.duration
    );

    // Step 3: Create initial guess model (with wrong parameters)
    let initial_mass = 1.0;
    let initial_length = 0.8;

    let guess_model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                initial_mass,
                Vec3::new(0.0, -initial_length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    initial_mass * initial_length * initial_length / 12.0,
                    0.0,
                    initial_mass * initial_length * initial_length / 12.0,
                )),
            ),
        )
        .build();

    println!(
        "Initial guess: mass = {:.2} kg, length = {:.2} m",
        initial_mass, initial_length
    );

    // Step 4: Set up parameter optimization
    let mut params = PhysicsParams::new();
    params.set_model_param("mass", initial_mass);
    params.set_model_param("length", initial_length);

    let weights = LossWeights {
        position: 1.0,
        velocity: 0.0,
        energy: 0.0,
        joint_angle: 10.0, // Emphasize joint angle matching
    };

    let matcher = TrajectoryMatcher::with_weights(reference.clone(), true_model.dt, weights);

    // Initial loss
    let initial_loss = matcher.compute_loss(&guess_model, &params);
    println!("Initial loss: {:.6e}\n", initial_loss);

    // Step 5: Run optimization (simplified - just a few iterations for demo)
    println!("Optimizing parameters...");
    let config = OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.001,
        convergence_threshold: 1e-8,
        gradient_epsilon: 1e-6,
        print_every: 1,
    };

    let optimizer = GradientDescentOptimizer::with_config(config);
    let result = optimizer.optimize(&guess_model, &matcher, &mut params);

    println!("\nOptimization complete:");
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!("  Final loss: {:.6e}", result.final_loss);

    let final_mass = params.get_model_param("mass").unwrap();
    let final_length = params.get_model_param("length").unwrap();
    println!(
        "\nEstimated parameters: mass = {:.2} kg, length = {:.2} m",
        final_mass, final_length
    );

    // Step 6: Build final identified model
    let identified_model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                final_mass,
                Vec3::new(0.0, -final_length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    final_mass * final_length * final_length / 12.0,
                    0.0,
                    final_mass * final_length * final_length / 12.0,
                )),
            ),
        )
        .build();

    // Step 7: Export to .phyz format
    let phyz_spec = model_to_phyz_spec(&identified_model, &params);

    match export_phyz(&phyz_spec) {
        Ok(json) => {
            println!("\n.phyz format export:");
            println!("{}", json);

            // Save to file
            if let Err(e) = std::fs::write("pendulum_identified.phyz", &json) {
                eprintln!("Failed to save .phyz file: {}", e);
            } else {
                println!("\nSaved to: pendulum_identified.phyz");
            }
        }
        Err(e) => eprintln!("Export failed: {}", e),
    }
}

/// Convert model and parameters to PhyzSpec.
fn model_to_phyz_spec(model: &phyz_model::Model, params: &PhysicsParams) -> PhyzSpec {
    use phyz_format::domain::{BodySpec, Domain, DomainType, JointSpec, JointTypeSpec};
    use phyz_format::schema::{ParameterSpec, ParameterType, WorldConfig};
    use std::collections::HashMap;

    // Build domain config
    let mut bodies = vec![BodySpec {
        name: "link".to_string(),
        mass: model.bodies[0].inertia.mass,
        inertia: [
            model.bodies[0].inertia.inertia[(0, 0)],
            model.bodies[0].inertia.inertia[(1, 1)],
            model.bodies[0].inertia.inertia[(2, 2)],
            model.bodies[0].inertia.inertia[(0, 1)],
            model.bodies[0].inertia.inertia[(0, 2)],
            model.bodies[0].inertia.inertia[(1, 2)],
        ],
        center_of_mass: [
            model.bodies[0].inertia.com.x,
            model.bodies[0].inertia.com.y,
            model.bodies[0].inertia.com.z,
        ],
    }];

    let mut joints = vec![JointSpec {
        joint_type: JointTypeSpec::Revolute,
        parent: "world".to_string(),
        child: "link".to_string(),
        axis: [0.0, 0.0, 1.0],
        position: [0.0, 0.0, 0.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        limits: None,
        damping: 0.0,
    }];

    let mut rigid_config = HashMap::new();
    rigid_config.insert("bodies".to_string(), serde_json::to_value(&bodies).unwrap());
    rigid_config.insert("joints".to_string(), serde_json::to_value(&joints).unwrap());

    let mut domains = HashMap::new();
    domains.insert(
        "rigid_body".to_string(),
        Domain {
            domain_type: DomainType::RigidBodyDynamics,
            config: rigid_config,
        },
    );

    // Build parameters
    let mut parameters: HashMap<String, ParameterSpec> = HashMap::new();
    for (name, value) in &params.model_params {
        parameters.insert(
            name.clone(),
            ParameterSpec {
                param_type: ParameterType::Scalar,
                value: serde_json::json!(value),
                uncertainty: None,
            },
        );
    }

    PhyzSpec {
        version: "1.0".to_string(),
        name: "pendulum-identified".to_string(),
        description: "Pendulum model identified from trajectory data".to_string(),
        world: WorldConfig {
            gravity: [model.gravity.x, model.gravity.y, model.gravity.z],
            dt: model.dt,
            default_contact_material: Default::default(),
        },
        domains,
        couplings: vec![],
        parameters,
        importers: vec![],
    }
}
