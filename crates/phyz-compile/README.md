# phyz-compile

Physics compiler for JIT compilation of physics kernels to GPU compute shaders.

## Features

- **Physics IR**: Domain-specific intermediate representation for physics computations
- **WGSL Compilation**: Generates WebGPU Shading Language compute shaders
- **Kernel Fusion**: Automatically fuses compatible kernels to reduce memory bandwidth
- **Automatic Differentiation**: Forward-mode AD for computing gradients
- **Stencil Operations**: Built-in support for common stencil patterns (Laplacian, averaging)
- **Scheduling Hints**: Tile sizes, loop unrolling, prefetching

## Example

```rust
use phyz_compile::{KernelBuilder, StencilBuilder, PhysicsOp, Compiler};

// Build a heat diffusion kernel: ∂T/∂t = κ ∇²T
let kernel = KernelBuilder::new("heat_diffusion")
    .field("T", [64, 64, 64])
    .field("T_new", [64, 64, 64])
    .tile_size([8, 8, 8])
    .op(PhysicsOp::store(
        "T_new",
        PhysicsOp::add(
            PhysicsOp::load("T"),
            PhysicsOp::mul(
                PhysicsOp::constant(0.01), // κ * dt
                StencilBuilder::laplacian_3d("T", 1.0),
            ),
        ),
    ))
    .build();

// Compile to WGSL
let mut compiler = Compiler::new();
let compiled = compiler.compile(&kernel).unwrap();

println!("{}", compiled.wgsl_source);
```

## Kernel Fusion

Fuse independent kernels to reduce memory traffic:

```rust
use phyz_compile::{FusionOptimizer, KernelBuilder, PhysicsOp};

let kernel1 = KernelBuilder::new("diffusion")
    .field("T", [64, 64, 64])
    .op(/* ... */)
    .build();

let kernel2 = KernelBuilder::new("advection")
    .field("U", [64, 64, 64])
    .op(/* ... */)
    .build();

if FusionOptimizer::can_fuse(&kernel1, &kernel2) {
    let fused = FusionOptimizer::fuse(kernel1, kernel2).unwrap();
    // 1.5-3x speedup expected from reduced memory bandwidth
}
```

## Automatic Differentiation

Augment kernels with forward-mode AD:

```rust
use phyz_compile::{AutoDiff, KernelBuilder, PhysicsOp};

let kernel = KernelBuilder::new("forward")
    .field("x", [64, 64, 64])
    .field("y", [64, 64, 64])
    .op(PhysicsOp::store("y", PhysicsOp::mul(
        PhysicsOp::load("x"),
        PhysicsOp::load("x"),
    )))
    .build();

// Augment with derivatives (creates dx, dy fields)
let ad_kernel = AutoDiff::augment_forward_mode(&kernel).unwrap();
```

## Architecture

```
PhysicsProgram (IR)
    ↓
Compiler
    ↓
WGSL Source (Naga-compatible)
    ↓
wgpu Compute Pipeline
```

## Examples

- `heat_equation.rs`: Heat diffusion PDE compilation
- `fusion.rs`: Kernel fusion demonstration
- `autodiff.rs`: Automatic differentiation

Run with:
```bash
cargo run --example heat_equation
cargo run --example fusion
cargo run --example autodiff
```
