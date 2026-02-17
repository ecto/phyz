//! Physics compiler: JIT compilation of physics kernels to GPU compute shaders.
//!
//! `phyz-compile` provides a domain-specific IR for physics computations and
//! compiles them to WGSL compute shaders. Features include:
//!
//! - Physics IR with stencil operations
//! - Compilation to WGSL (WebGPU Shading Language)
//! - Kernel fusion for performance optimization
//! - Automatic differentiation (forward-mode)
//! - Scheduling hints for GPU optimization
//!
//! # Example: Heat Equation
//!
//! ```
//! use phyz_compile::{KernelBuilder, StencilBuilder, PhysicsOp, Compiler};
//!
//! // Build a heat diffusion kernel: ∂T/∂t = κ ∇²T
//! let kernel = KernelBuilder::new("heat_diffusion")
//!     .field("T", [64, 64, 64])
//!     .field("T_new", [64, 64, 64])
//!     .tile_size([8, 8, 8])
//!     .op(PhysicsOp::store(
//!         "T_new",
//!         PhysicsOp::add(
//!             PhysicsOp::load("T"),
//!             PhysicsOp::mul(
//!                 PhysicsOp::constant(0.01), // κ * dt
//!                 StencilBuilder::laplacian_3d("T", 1.0),
//!             ),
//!         ),
//!     ))
//!     .build();
//!
//! // Compile to WGSL
//! let mut compiler = Compiler::new();
//! let compiled = compiler.compile(&kernel).unwrap();
//!
//! println!("Generated WGSL:\n{}", compiled.wgsl_source);
//! ```
//!
//! # Example: Kernel Fusion
//!
//! ```
//! use phyz_compile::{KernelBuilder, PhysicsOp, FusionOptimizer};
//!
//! // Two independent kernels
//! let kernel1 = KernelBuilder::new("k1")
//!     .field("A", [64, 64, 64])
//!     .op(PhysicsOp::store(
//!         "A",
//!         PhysicsOp::mul(PhysicsOp::load("A"), PhysicsOp::constant(2.0)),
//!     ))
//!     .build();
//!
//! let kernel2 = KernelBuilder::new("k2")
//!     .field("B", [64, 64, 64])
//!     .op(PhysicsOp::store(
//!         "B",
//!         PhysicsOp::add(PhysicsOp::load("B"), PhysicsOp::constant(1.0)),
//!     ))
//!     .build();
//!
//! // Fuse them
//! if FusionOptimizer::can_fuse(&kernel1, &kernel2) {
//!     let fused = FusionOptimizer::fuse(kernel1, kernel2).unwrap();
//!     println!("Fused {} operations", fused.ops.len());
//! }
//! ```
//!
//! # Example: Automatic Differentiation
//!
//! ```
//! use phyz_compile::{KernelBuilder, PhysicsOp, AutoDiff};
//!
//! let kernel = KernelBuilder::new("forward")
//!     .field("x", [64, 64, 64])
//!     .field("y", [64, 64, 64])
//!     .op(PhysicsOp::store(
//!         "y",
//!         PhysicsOp::mul(PhysicsOp::load("x"), PhysicsOp::load("x")),
//!     ))
//!     .build();
//!
//! // Augment with forward-mode AD
//! let ad_kernel = AutoDiff::augment_forward_mode(&kernel).unwrap();
//!
//! // Now has both y and dy fields
//! assert!(ad_kernel.fields.contains_key("y"));
//! assert!(ad_kernel.fields.contains_key("dy"));
//! ```

pub mod autodiff;
pub mod builder;
pub mod compiler;
pub mod fusion;
pub mod ir;

pub use autodiff::AutoDiff;
pub use builder::{KernelBuilder, StencilBuilder};
pub use compiler::{CompiledKernel, Compiler};
pub use fusion::FusionOptimizer;
pub use ir::{BinOp, FieldMeta, FieldRef, FieldType, PhysicsOp, PhysicsProgram, ScheduleHints};
