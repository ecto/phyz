//! Multi-scale coupling between different physics solvers.
//!
//! # What is wired
//!
//! The [`Solver`] trait is the abstraction the coupling layer drives: a domain
//! that can advance by `dt`, expose its state for coupling queries, accept
//! external forces or sources, and report its natural timestep.
//! [`CoupledSystem`] pairs two of them, evaluates a Lorentz handshake over an
//! overlap region, and applies it antisymmetrically while a [`FluxLedger`]
//! accounts for the momentum and energy that crossed.
//!
//! Two real solvers implement the trait today — [`RigidSolver`] over
//! `phyz-rigid`'s ABA, and [`EmSolver`] over `phyz-em`'s Yee-grid FDTD — and
//! `tests/cyclotron.rs` runs them coupled against the closed-form cyclotron
//! solution. `phyz-particle`, `phyz-md`, `phyz-lbm`, and `phyz-gravity` have no
//! adapters yet.
//!
//! # Coupled simulation
//!
//! ```no_run
//! use phyz_coupling::{BoundingBox, CoupledSystem, EmSolver, RigidSolver, Solver};
//! use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
//! use phyz_model::ModelBuilder;
//!
//! let model = ModelBuilder::new()
//!     .gravity(Vec3::zeros())
//!     .dt(5e-11)
//!     .add_free_body(
//!         "bob",
//!         -1,
//!         SpatialTransform::identity(),
//!         SpatialInertia::new(1e-9, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(1e-12, 1e-12, 1e-12))),
//!     )
//!     .build();
//!
//! let mut state = model.default_state();
//! state.q[0] = 0.4; state.q[1] = 0.4; state.q[2] = 0.4;
//! state.v[3] = 1e5; // free joint: v = [omega(3), v(3)]
//!
//! let mut matter = RigidSolver::new(model, state);
//! matter.couple_body(0, 6.28e-3); // charge, in coulombs
//!
//! let field = EmSolver::uniform_b_field(8, 0.1, Vec3::new(0.0, 0.0, 1.0));
//! let region = BoundingBox::new(Vec3::zeros(), Vec3::new(0.8, 0.8, 0.8));
//!
//! let mut sys = CoupledSystem::new(matter, field, region);
//! sys.run(20_000, 5e-11);
//!
//! // Momentum booked into each domain matches what each actually absorbed.
//! let (err_matter, err_field) = sys.absorption_error();
//! ```
//!
//! # Coupling configuration
//!
//! [`Coupling`] and [`ForceTransfer`] describe handshake regions and transfer
//! laws as data. They are configuration types: they compute forces from bare
//! position/velocity pairs and are not yet driven by [`CoupledSystem`], which
//! implements the Lorentz path directly.
//!
//! ```
//! use phyz_coupling::{Coupling, ForceTransfer, BoundingBox, SolverType};
//! use phyz_math::Vec3;
//!
//! let coupling = Coupling {
//!     solver_a: SolverType::Electromagnetic,
//!     solver_b: SolverType::RigidBody,
//!     overlap_region: BoundingBox {
//!         min: Vec3::new(-1.0, -1.0, -1.0),
//!         max: Vec3::new(1.0, 1.0, 1.0),
//!     },
//!     force_transfer: ForceTransfer::Direct { damping: 0.1 },
//! };
//!
//! let force = phyz_coupling::lorentz_force(
//!     1e-6,                      // charge (C)
//!     Vec3::new(0.0, 0.0, 0.0),
//!     Vec3::new(1.0, 0.0, 0.0),  // velocity
//!     &Vec3::new(0.0, 0.0, 1e3), // E field
//!     &Vec3::new(0.0, 1.0, 0.0), // B field
//! );
//! ```

pub mod boundary;
pub mod conserve;
pub mod coupled;
pub mod coupling;
pub mod lorentz;
pub mod solver;
pub mod solvers;
pub mod subcycling;

pub use boundary::BoundingBox;
pub use conserve::FluxLedger;
pub use coupled::{CoupledSystem, ReactionMode};
pub use coupling::{Coupling, ForceTransfer, SolverType};
pub use lorentz::{lorentz_force, magnetic_torque};
pub use solver::{CouplingSite, ExternalInput, FieldSample, Solver};
pub use solvers::{EmSolver, RigidSolver};
pub use subcycling::{SubcyclingSchedule, TimeScale};
