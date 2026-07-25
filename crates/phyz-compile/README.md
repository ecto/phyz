# phyz-compile

A physics kernel compiler: IR → fused WGSL compute shaders.

A domain-specific IR for grid-based physics, compiled to WebGPU shading
language with kernel fusion and forward-mode automatic differentiation.

| Type | Purpose |
| --- | --- |
| `KernelBuilder` | declare fields, tile size, and ops |
| `StencilBuilder` | ready-made stencils (`laplacian_3d`, gradients, ...) |
| `PhysicsOp`, `PhysicsProgram` | the IR itself |
| `Compiler`, `CompiledKernel` | IR → WGSL source + workgroup size |
| `FusionOptimizer` | merge independent kernels into one dispatch |
| `AutoDiff` | augment a kernel with tangent fields |

## Example: heat diffusion

```rust
use phyz_compile::{Compiler, KernelBuilder, PhysicsOp, StencilBuilder};

let kernel = KernelBuilder::new("heat_diffusion")
    .field("T", [64, 64, 64])
    .field("T_new", [64, 64, 64])
    .tile_size([8, 8, 8])
    .op(PhysicsOp::store(
        "T_new",
        PhysicsOp::add(
            PhysicsOp::load("T"),
            PhysicsOp::mul(
                PhysicsOp::constant(0.01),               // κ·dt
                StencilBuilder::laplacian_3d("T", 1.0),
            ),
        ),
    ))
    .build();

let compiled = Compiler::new().compile(&kernel).unwrap();
println!("{}", compiled.wgsl_source);
```

## Example: fusion

```rust
use phyz_compile::{FusionOptimizer, KernelBuilder, PhysicsOp};

# let k1 = KernelBuilder::new("k1").field("A", [64, 64, 64])
#     .op(PhysicsOp::store("A", PhysicsOp::mul(PhysicsOp::load("A"), PhysicsOp::constant(2.0)))).build();
# let k2 = KernelBuilder::new("k2").field("B", [64, 64, 64])
#     .op(PhysicsOp::store("B", PhysicsOp::add(PhysicsOp::load("B"), PhysicsOp::constant(1.0)))).build();
if FusionOptimizer::can_fuse(&k1, &k2) {
    let fused = FusionOptimizer::fuse(k1, k2).unwrap();
    // one dispatch instead of two
}
```

Compilation needs no GPU — it only generates shader source. Run
`cargo run --release -p phyz-examples --example kernel_fusion` for the full
pipeline end to end.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
