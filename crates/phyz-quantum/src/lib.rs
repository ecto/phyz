#![allow(clippy::needless_range_loop)]
//! Hamiltonian lattice gauge theory on simplicial complexes.
//!
//! Implements the Kogut-Susskind Hamiltonian for U(1) gauge theory on the
//! simplicial lattices from `phyz-regge`, with triangular plaquettes and
//! 3-edge holonomies instead of the standard square plaquettes on hypercubic
//! lattices.
//!
//! # Modules
//!
//! - [`gauss_law`]: Vertex-edge adjacency, spanning tree, gauge-invariant basis
//! - [`hilbert`]: Hilbert space with fast config↔index lookup
//! - [`hamiltonian`]: Sparse Kogut-Susskind Hamiltonian construction
//! - [`diag`]: Dense eigendecomposition via tang-la
//! - [`observables`]: Wilson loops, electric field, entanglement entropy
//! - [`qubit_map`]: Qubit encoding and resource estimates
//! - [`stabilizer`]: Z₂ limit → stabilizer code parameters

pub mod diag;
pub mod gauss_law;
pub mod hamiltonian;
pub mod hilbert;
pub mod hypercubic;
pub mod jacobson;
pub mod lanczos;
pub mod observables;
pub mod qubit_map;
pub mod ryu_takayanagi;
pub mod stabilizer;
pub mod su2_quantum;
pub mod triangulated_torus;

#[cfg(feature = "gpu")]
pub mod csr;
#[cfg(feature = "gpu")]
pub mod gpu_lanczos;

pub use diag::Spectrum;
pub use hamiltonian::KSParams;
pub use hilbert::U1HilbertSpace;
