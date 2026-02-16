//! Heat equation kernel compilation example.
//!
//! Demonstrates compiling a heat diffusion PDE kernel to WGSL:
//! ∂T/∂t = κ ∇²T

use tau_compile::{Compiler, KernelBuilder, PhysicsOp, StencilBuilder};

fn main() {
    println!("=== Heat Equation Compiler Demo ===\n");

    // Build the heat diffusion kernel
    // ∂T/∂t = κ ∇²T
    // T_new = T + κ * dt * ∇²T

    let kappa = 1.0; // thermal diffusivity
    let dt = 0.01; // time step
    let dx = 1.0; // grid spacing

    let kernel = KernelBuilder::new("heat_diffusion")
        .field("T", [64, 64, 64])
        .field("T_new", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "T_new",
            PhysicsOp::add(
                PhysicsOp::load("T"),
                PhysicsOp::mul(
                    PhysicsOp::constant(kappa * dt),
                    StencilBuilder::laplacian_3d("T", dx),
                ),
            ),
        ))
        .build();

    println!("Kernel: {}", kernel.name);
    println!("Fields: {:?}", kernel.fields.keys());
    println!("Operations: {}", kernel.ops.len());
    println!("Tile size: {:?}\n", kernel.schedule.tile_size);

    // Compile to WGSL
    let mut compiler = Compiler::new();
    match compiler.compile(&kernel) {
        Ok(compiled) => {
            println!("=== Generated WGSL ===\n");
            println!("{}", compiled.wgsl_source);
            println!("\n=== Compilation Successful ===");
            println!("Workgroup size: {:?}", compiled.workgroup_size);
        }
        Err(e) => {
            eprintln!("Compilation error: {}", e);
            std::process::exit(1);
        }
    }
}
