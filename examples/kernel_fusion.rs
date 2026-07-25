//! End-to-end `phyz-compile`: physics IR → kernel fusion → WGSL → autodiff.
//!
//! Walks the whole compiler pipeline on a 3D heat-diffusion stencil:
//!
//! 1. Build a stencil kernel from the physics IR (`∂T/∂t = κ ∇²T`).
//! 2. Compile it to WGSL and inspect the generated shader.
//! 3. Fuse two independent kernels into one dispatch and show the op count
//!    and the WGSL size actually shrink.
//! 4. Augment a kernel with forward-mode automatic differentiation.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p phyz-examples --example kernel_fusion
//! ```
//!
//! No GPU is required — this example only *generates* shaders, it does not
//! dispatch them.

use phyz_compile::{AutoDiff, Compiler, FusionOptimizer, KernelBuilder, PhysicsOp, StencilBuilder};

const N: usize = 64;
const GRID: [usize; 3] = [N, N, N];

/// κ · dt for the explicit heat update.
const KAPPA_DT: f64 = 0.01;

fn main() {
    println!("=== phyz-compile: physics IR → WGSL ===\n");

    // ---------------------------------------------------------------
    // 1. Build a heat-diffusion kernel: T_new = T + κ dt ∇²T
    // ---------------------------------------------------------------
    let heat = KernelBuilder::new("heat_diffusion")
        .field("T", GRID)
        .field("T_new", GRID)
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "T_new",
            PhysicsOp::add(
                PhysicsOp::load("T"),
                PhysicsOp::mul(
                    PhysicsOp::constant(KAPPA_DT),
                    StencilBuilder::laplacian_3d("T", 1.0),
                ),
            ),
        ))
        .build();

    println!("kernel     : {}", heat.name);
    println!("fields     : {}", heat.fields.len());
    println!("ops        : {}", heat.ops.len());
    println!("grid       : {}x{}x{}", GRID[0], GRID[1], GRID[2]);

    // ---------------------------------------------------------------
    // 2. Compile to WGSL
    // ---------------------------------------------------------------
    let mut compiler = Compiler::new();
    let compiled = compiler
        .compile(&heat)
        .expect("heat kernel should compile to WGSL");

    let lines = compiled.wgsl_source.lines().count();
    println!(
        "\ncompiled   : {} lines of WGSL, workgroup {:?}",
        lines, compiled.workgroup_size
    );
    println!("\n--- first 20 lines of generated WGSL ---");
    for line in compiled.wgsl_source.lines().take(20) {
        println!("  {line}");
    }
    println!("  ... ({} more lines)", lines.saturating_sub(20));

    // ---------------------------------------------------------------
    // 3. Kernel fusion: two elementwise passes become one dispatch
    // ---------------------------------------------------------------
    println!("\n=== kernel fusion ===\n");

    let scale = KernelBuilder::new("scale_a")
        .field("A", GRID)
        .op(PhysicsOp::store(
            "A",
            PhysicsOp::mul(PhysicsOp::load("A"), PhysicsOp::constant(2.0)),
        ))
        .build();

    let offset = KernelBuilder::new("offset_b")
        .field("B", GRID)
        .op(PhysicsOp::store(
            "B",
            PhysicsOp::add(PhysicsOp::load("B"), PhysicsOp::constant(1.0)),
        ))
        .build();

    let unfused_wgsl: usize = [&scale, &offset]
        .iter()
        .map(|k| {
            Compiler::new()
                .compile(k)
                .expect("kernel should compile")
                .wgsl_source
                .len()
        })
        .sum();

    if !FusionOptimizer::can_fuse(&scale, &offset) {
        println!("kernels are not fusable — nothing to do");
        return;
    }

    let ops_before = scale.ops.len() + offset.ops.len();
    let fused = FusionOptimizer::fuse(scale, offset).expect("fusion should succeed");
    let fused_wgsl = Compiler::new()
        .compile(&fused)
        .expect("fused kernel should compile")
        .wgsl_source
        .len();

    println!("before     : 2 dispatches, {ops_before} ops, {unfused_wgsl} bytes of WGSL");
    println!(
        "after      : 1 dispatch,  {} ops, {} bytes of WGSL",
        fused.ops.len(),
        fused_wgsl
    );
    println!(
        "saved      : 1 dispatch, {:.1}% WGSL",
        100.0 * (1.0 - fused_wgsl as f64 / unfused_wgsl as f64)
    );

    // ---------------------------------------------------------------
    // 4. Forward-mode autodiff on a compiled kernel
    // ---------------------------------------------------------------
    println!("\n=== forward-mode autodiff ===\n");

    let forward = KernelBuilder::new("square")
        .field("x", GRID)
        .field("y", GRID)
        .op(PhysicsOp::store(
            "y",
            PhysicsOp::mul(PhysicsOp::load("x"), PhysicsOp::load("x")),
        ))
        .build();

    let differentiated =
        AutoDiff::augment_forward_mode(&forward).expect("forward-mode AD should succeed");

    let mut names: Vec<&String> = differentiated.fields.keys().collect();
    names.sort();
    println!("y = x²  →  fields after AD: {names:?}");
    assert!(differentiated.fields.contains_key("dy"), "dy tangent field");

    let ad_compiled = Compiler::new()
        .compile(&differentiated)
        .expect("AD kernel should compile");
    println!(
        "compiled   : {} lines of WGSL computing y and dy in one pass",
        ad_compiled.wgsl_source.lines().count()
    );
}
