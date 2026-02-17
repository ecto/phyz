//! 4D Regge calculus with U(1) gauge field for Einstein-Maxwell theory.
//!
//! Discretizes general relativity on a simplicial complex (Regge calculus)
//! coupled to electromagnetism via a U(1) lattice gauge field. The combined
//! action S = S_R + α·S_M is differentiable w.r.t. all degrees of freedom:
//!
//! - **Edge lengths** {l_e}: encode the metric (gravitational DOF)
//! - **Edge phases** {θ_e}: encode the U(1) gauge field (electromagnetic DOF)
//!
//! The coupling between gravity and EM enters through the metric weights
//! in the Maxwell action, which depend on the edge lengths.
//!
//! # Architecture
//!
//! - [`complex`]: Simplicial complex data structure with full incidence relations
//! - [`geometry`]: Cayley-Menger determinants, areas, volumes, dihedral angles
//! - [`regge`]: Regge action S_R = Σ A_t δ_t and its gradient (Schläfli identity)
//! - [`gauge`]: U(1) gauge field, field strengths, Maxwell action on curved background
//! - [`action`]: Combined Einstein-Maxwell action, gradients, symmetry search utilities
//! - [`mesh`]: Mesh generation (flat hypercubic, Reissner-Nordström deformation)
//!
//! # Example
//!
//! ```
//! use phyz_regge::{mesh, action::{Fields, ActionParams, einstein_maxwell_action}};
//!
//! // Flat spacetime on a 2×2×2×2 periodic lattice.
//! let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
//! let phases = vec![0.0; complex.n_edges()];
//!
//! let fields = Fields::new(lengths, phases);
//! let params = ActionParams::default();
//! let s = einstein_maxwell_action(&complex, &fields, &params);
//!
//! // Flat vacuum → zero action.
//! assert!(s.abs() < 1e-8);
//! ```

pub mod action;
pub mod complex;
pub mod gauge;
pub mod geometry;
pub mod mesh;
pub mod regge;
pub mod richardson;
pub mod search;
pub mod symmetry;

pub use action::{ActionParams, Fields};
pub use complex::SimplicialComplex;
pub use search::{search_symmetries, SearchConfig, SearchResults};
pub use symmetry::Generator;
