//! Kernel fusion example.
//!
//! Demonstrates fusing multiple physics kernels to reduce memory bandwidth.

use phyz_compile::{Compiler, FusionOptimizer, KernelBuilder, PhysicsOp, StencilBuilder};

fn main() {
    println!("=== Kernel Fusion Demo ===\n");

    // Create two independent kernels that can be fused

    // Kernel 1: Heat diffusion on field T
    let kernel1 = KernelBuilder::new("heat_diffusion")
        .field("T", [64, 64, 64])
        .field("T_new", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "T_new",
            PhysicsOp::add(
                PhysicsOp::load("T"),
                PhysicsOp::mul(
                    PhysicsOp::constant(0.01),
                    StencilBuilder::laplacian_3d("T", 1.0),
                ),
            ),
        ))
        .build();

    // Kernel 2: Advection on field U
    let kernel2 = KernelBuilder::new("advection")
        .field("U", [64, 64, 64])
        .field("U_new", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "U_new",
            PhysicsOp::mul(PhysicsOp::load("U"), PhysicsOp::constant(0.99)),
        ))
        .build();

    println!("Kernel 1: {}", kernel1.name);
    println!("  Fields: {:?}", kernel1.fields.keys());
    println!("  Operations: {}", kernel1.ops.len());

    println!("\nKernel 2: {}", kernel2.name);
    println!("  Fields: {:?}", kernel2.fields.keys());
    println!("  Operations: {}", kernel2.ops.len());

    // Check if they can be fused
    if FusionOptimizer::can_fuse(&kernel1, &kernel2) {
        println!("\n✓ Kernels can be fused!");

        match FusionOptimizer::fuse(kernel1.clone(), kernel2.clone()) {
            Ok(fused) => {
                println!("\nFused kernel: {}", fused.name);
                println!("  Fields: {:?}", fused.fields.keys());
                println!("  Operations: {}", fused.ops.len());

                // Compile both versions for comparison
                let mut compiler = Compiler::new();

                println!("\n=== Unfused Version ===");
                if let Ok(c1) = compiler.compile(&kernel1) {
                    println!("Kernel 1 WGSL size: {} bytes", c1.wgsl_source.len());
                }
                if let Ok(c2) = compiler.compile(&kernel2) {
                    println!("Kernel 2 WGSL size: {} bytes", c2.wgsl_source.len());
                }

                println!("\n=== Fused Version ===");
                if let Ok(compiled) = compiler.compile(&fused) {
                    println!("Fused WGSL size: {} bytes", compiled.wgsl_source.len());
                    println!(
                        "\nFused kernel combines {} operations in a single pass",
                        fused.ops.len()
                    );
                    println!("Expected speedup: 1.5-2x from reduced memory traffic");
                }
            }
            Err(e) => {
                eprintln!("Fusion failed: {}", e);
            }
        }
    } else {
        println!("\n✗ Kernels cannot be fused (incompatible domains or dependencies)");
    }

    // Example of kernels that CAN'T be fused due to dependency
    println!("\n\n=== Dependency Prevention Example ===\n");

    let kernel_a = KernelBuilder::new("kernel_a")
        .field("T", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("T", PhysicsOp::constant(1.0)))
        .build();

    let kernel_b = KernelBuilder::new("kernel_b")
        .field("T", [64, 64, 64])
        .field("U", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("U", PhysicsOp::load("T")))
        .build();

    println!("Kernel A writes to T");
    println!("Kernel B reads from T");

    if FusionOptimizer::can_fuse(&kernel_a, &kernel_b) {
        println!("✓ Can fuse (unexpected!)");
    } else {
        println!("✗ Cannot fuse (kernel B depends on kernel A's output)");
    }
}
