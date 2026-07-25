//! Benchmark scenes and the phyz stepping loop used to run them.
//!
//! phyz does not ship a `Simulator` facade, so the loop here is assembled from
//! the same public pieces a user would reach for: `forward_kinematics` →
//! ground contact → `aba_with_external_forces` → semi-implicit Euler. That
//! makes the measurement representative of the library as it actually exists.

use phyz_collision::Collision;
use phyz_contact::{ContactMaterial, contact_forces, find_contacts, find_ground_contacts};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, SpatialVec, Vec3};
use phyz_model::{Body, Geometry, Model, ModelBuilder, State};
use phyz_rigid::{aba_with_external_forces, forward_kinematics};

use crate::settings::{CONTACT_DAMPING, CONTACT_FRICTION, CONTACT_STIFFNESS, GRAVITY};

/// Which scene to run. Every suite refers to scenes through this enum so the
/// phyz and Rapier sides cannot silently drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    /// Single revolute link, 1 m, 1 kg, swinging under gravity.
    Pendulum,
    /// Two 1 m / 1 kg revolute links — chaotic, and a real test of the
    /// integrator rather than of a closed-form solution.
    DoublePendulum,
    /// The MuJoCo-style ant from `models/ant.xml`: free-floating torso plus
    /// eight hinges, 14 DOF.
    Ant,
    /// `n` free boxes dropped in a column onto a ground plane.
    BoxStack(usize),
}

impl Scene {
    /// Stable identifier used as the JSON key and the table row label.
    pub fn name(&self) -> String {
        match self {
            Scene::Pendulum => "pendulum".into(),
            Scene::DoublePendulum => "double_pendulum".into(),
            Scene::Ant => "ant".into(),
            Scene::BoxStack(n) => format!("box_stack_{n}"),
        }
    }

    /// One-line description carried into the published results.
    pub fn description(&self) -> String {
        match self {
            Scene::Pendulum => "1 revolute link, 1 m, 1 kg, no contact".into(),
            Scene::DoublePendulum => "2 revolute links, 1 m / 1 kg each, no contact".into(),
            Scene::Ant => "free-floating torso + 8 hinges (14 DOF), no contact".into(),
            Scene::BoxStack(n) => {
                format!("{n} free 10 cm boxes (1 kg) stacked on a ground plane, contact enabled")
            }
        }
    }

    /// Whether this scene exercises the contact path.
    pub fn has_contact(&self) -> bool {
        matches!(self, Scene::BoxStack(_))
    }
}

/// A uniform rod of `mass` and `length` hanging along −z from its joint, with
/// the inertia tensor of a thin rod about its centre of mass.
fn rod_inertia(mass: f64, length: f64) -> SpatialInertia {
    let i = mass * length * length / 12.0;
    SpatialInertia::new(
        mass,
        Vec3::new(0.0, 0.0, -length * 0.5),
        Mat3::from_diagonal(&Vec3::new(i, i, 0.0)),
    )
}

/// Solid box inertia about its centre.
fn box_inertia(mass: f64, half: f64) -> SpatialInertia {
    let i = mass * (2.0 * half) * (2.0 * half) / 6.0;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

/// Link length shared by the pendulum scenes and their Rapier counterparts.
pub const LINK_LENGTH: f64 = 1.0;
/// Link mass shared by the pendulum scenes and their Rapier counterparts.
pub const LINK_MASS: f64 = 1.0;
/// Half-extent of a box-stack box (m).
pub const BOX_HALF_EXTENT: f64 = 0.05;
/// Mass of a box-stack box (kg).
pub const BOX_MASS: f64 = 1.0;
/// Vertical gap between boxes at t = 0 (m). Small, so the stack settles fast.
pub const BOX_GAP: f64 = 0.01;

fn gravity_vec() -> Vec3 {
    Vec3::new(0.0, 0.0, -GRAVITY)
}

/// A revolute joint about **+Y**.
///
/// `Joint::revolute` defaults to the Z axis, which with a rod hanging along −Z
/// under −Z gravity puts the centre of mass on the rotation axis: the
/// gravitational torque is identically zero and the "pendulum" never swings.
/// Hinging about Y instead gives a real pendulum with its equilibrium hanging
/// straight down.
pub fn hinge_y(parent_to_joint: SpatialTransform) -> phyz_model::Joint {
    let mut j = phyz_model::Joint::revolute(parent_to_joint);
    j.axis = Vec3::new(0.0, 1.0, 0.0);
    j
}

/// Build the phyz `Model` for a scene at the given timestep.
pub fn build_model(scene: Scene, dt: f64) -> Model {
    match scene {
        Scene::Pendulum => ModelBuilder::new()
            .gravity(gravity_vec())
            .dt(dt)
            .add_body(
                "link1",
                -1,
                hinge_y(SpatialTransform::identity()),
                rod_inertia(LINK_MASS, LINK_LENGTH),
            )
            .build(),
        Scene::DoublePendulum => ModelBuilder::new()
            .gravity(gravity_vec())
            .dt(dt)
            .add_body(
                "link1",
                -1,
                hinge_y(SpatialTransform::identity()),
                rod_inertia(LINK_MASS, LINK_LENGTH),
            )
            .add_body(
                "link2",
                0,
                hinge_y(SpatialTransform::new(
                    Mat3::identity(),
                    Vec3::new(0.0, 0.0, -LINK_LENGTH),
                )),
                rod_inertia(LINK_MASS, LINK_LENGTH),
            )
            .build(),
        Scene::Ant => {
            let mut model = load_ant();
            model.dt = dt;
            model.gravity = gravity_vec();
            model
        }
        Scene::BoxStack(n) => {
            let mut b = ModelBuilder::new().gravity(gravity_vec()).dt(dt);
            for i in 0..n {
                b = b.add_free_body_with_geometry(
                    &format!("box{i}"),
                    -1,
                    SpatialTransform::identity(),
                    box_inertia(BOX_MASS, BOX_HALF_EXTENT),
                    Body {
                        name: String::new(),
                        inertia: box_inertia(BOX_MASS, BOX_HALF_EXTENT),
                        parent: -1,
                        joint_idx: 0,
                        geometry: Some(Geometry::Box {
                            half_extents: Vec3::new(
                                BOX_HALF_EXTENT,
                                BOX_HALF_EXTENT,
                                BOX_HALF_EXTENT,
                            ),
                        }),
                    },
                );
            }
            b.build()
        }
    }
}

/// Locate and parse `models/ant.xml`.
///
/// Searched relative to the crate, then to the workspace root, so the harness
/// works from `cargo bench -p phyz-bench` and from a bare `./phyz-bench` alike.
pub fn load_ant() -> Model {
    let candidates = [
        "models/ant.xml",
        "../../models/ant.xml",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml"),
    ];
    for path in candidates {
        if let Ok(loader) = phyz_mjcf::MjcfLoader::from_file(path) {
            return loader.build_model();
        }
    }
    panic!(
        "could not find models/ant.xml (looked in {candidates:?}); run the harness from the \
         workspace root"
    );
}

/// Initial state for a scene — deliberately away from equilibrium so the
/// dynamics are actually exercised.
pub fn initial_state(scene: Scene, model: &Model) -> State {
    let mut state = model.default_state();
    match scene {
        Scene::Pendulum => state.q[0] = 1.0,
        Scene::DoublePendulum => {
            state.q[0] = 1.0;
            state.q[1] = 0.5;
        }
        Scene::Ant => {
            // Free joint q = [x, y, z, wx, wy, wz]; torso at its rest height.
            state.q[2] = 0.75;
            for i in 6..model.nq {
                state.q[i] = 0.1;
            }
        }
        Scene::BoxStack(n) => {
            for i in 0..n {
                let base = model.q_offsets[i];
                state.q[base + 2] = BOX_HALF_EXTENT + i as f64 * (2.0 * BOX_HALF_EXTENT + BOX_GAP);
            }
        }
    }
    state
}

/// Per-body collision geometry, in the shape `phyz_contact` expects.
pub fn geometries(model: &Model) -> Vec<Option<Geometry>> {
    model.bodies.iter().map(|b| b.geometry.clone()).collect()
}

/// The contact material used by every phyz contact scene.
pub fn contact_material() -> ContactMaterial {
    ContactMaterial::new(
        CONTACT_STIFFNESS,
        CONTACT_DAMPING,
        CONTACT_FRICTION,
        0.0, // restitution
    )
}

/// A reusable phyz stepper for one scene: owns the scratch buffers so the
/// timing loop measures physics rather than allocation.
pub struct PhyzSim {
    pub model: Model,
    pub state: State,
    geoms: Vec<Option<Geometry>>,
    materials: Vec<ContactMaterial>,
    contact_enabled: bool,
    scene: Scene,
}

impl PhyzSim {
    /// Build a simulator for `scene` at timestep `dt`, in its initial state.
    pub fn new(scene: Scene, dt: f64) -> Self {
        let model = build_model(scene, dt);
        let state = initial_state(scene, &model);
        let geoms = geometries(&model);
        let materials = vec![contact_material(); model.nbodies()];
        Self {
            model,
            state,
            geoms,
            materials,
            contact_enabled: scene.has_contact(),
            scene,
        }
    }

    /// Reset to the initial state without rebuilding the model.
    pub fn reset(&mut self) {
        self.state = initial_state(self.scene, &self.model);
    }

    /// Advance one timestep: kinematics, contact, ABA, semi-implicit Euler.
    pub fn step(&mut self) {
        let dt = self.model.dt;
        let (xforms, velocities) = forward_kinematics(&self.model, &self.state);
        self.state.body_xform = xforms;

        let qdd = if self.contact_enabled {
            // A real stack needs both halves of phyz's contact story: the
            // cheap plane query against the ground, and the full broad-phase +
            // GJK/EPA path between boxes. Benchmarking only the former would
            // flatter the engine.
            let mut contacts: Vec<Collision> = find_ground_contacts(&self.state, &self.geoms, 0.0);
            contacts.extend(find_contacts(&self.model, &self.state, &self.geoms));
            let forces: Vec<SpatialVec> =
                contact_forces(&contacts, &self.state, &self.materials, Some(&velocities));
            aba_with_external_forces(&self.model, &self.state, Some(&forces))
        } else {
            aba_with_external_forces(&self.model, &self.state, None)
        };

        // Semi-implicit (symplectic) Euler: velocity first, then position.
        for i in 0..self.model.nv {
            self.state.v[i] += dt * qdd[i];
        }
        for i in 0..self.model.nq {
            self.state.q[i] += dt * self.state.v[i];
        }
        self.state.time += dt;
    }

    /// Advance `n` timesteps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Whether the state is still finite.
    ///
    /// A simulation that has blown up to NaN often gets *faster* (no contacts
    /// are found, branches collapse), so every timed run is checked: an
    /// unstable scene must not be published as a fast one.
    pub fn state_is_finite(&self) -> bool {
        self.state.q.as_slice().iter().all(|x| x.is_finite())
            && self.state.v.as_slice().iter().all(|x| x.is_finite())
    }

    /// Total mechanical energy (J) of the current state.
    pub fn total_energy(&self) -> f64 {
        phyz_rigid::total_energy(&self.model, &self.state)
    }
}
