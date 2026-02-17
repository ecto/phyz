//! Automatic differentiation example.
//!
//! Demonstrates forward-mode AD for physics kernels.

use phyz_compile::{AutoDiff, Compiler, KernelBuilder, PhysicsOp};

fn main() {
    println!("=== Automatic Differentiation Demo ===\n");

    // Create a simple kernel: y = x^2
    let kernel = KernelBuilder::new("square")
        .field("x", [64, 64, 64])
        .field("y", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "y",
            PhysicsOp::mul(PhysicsOp::load("x"), PhysicsOp::load("x")),
        ))
        .build();

    println!("Original kernel: {}", kernel.name);
    println!("  Fields: {:?}", kernel.fields.keys());
    println!("  Operation: y = x * x");

    // Augment with forward-mode AD
    match AutoDiff::augment_forward_mode(&kernel) {
        Ok(ad_kernel) => {
            println!("\nAugmented kernel: {}", ad_kernel.name);
            println!("  Fields: {:?}", ad_kernel.fields.keys());
            println!("  Operations: {} (primal + tangent)", ad_kernel.ops.len());

            // The augmented kernel computes:
            // - Primal: y = x * x
            // - Tangent: dy = x * dx + x * dx = 2 * x * dx

            println!("\nDifferentiation rules applied:");
            println!("  Primal: y = x * x");
            println!("  Tangent: dy = 2 * x * dx (product rule)");

            // Compile both versions
            let mut compiler = Compiler::new();

            println!("\n=== Original Kernel WGSL ===");
            if let Ok(compiled) = compiler.compile(&kernel) {
                println!("{}", compiled.wgsl_source);
            }

            println!("\n=== AD Kernel WGSL ===");
            if let Ok(compiled) = compiler.compile(&ad_kernel) {
                println!("{}", compiled.wgsl_source);
            }
        }
        Err(e) => {
            eprintln!("AD augmentation failed: {}", e);
        }
    }

    // More complex example: combined operations
    println!("\n\n=== Complex Example: (x + 1) / x ===\n");

    let complex_kernel = KernelBuilder::new("complex")
        .field("x", [64, 64, 64])
        .field("y", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "y",
            PhysicsOp::div(
                PhysicsOp::add(PhysicsOp::load("x"), PhysicsOp::constant(1.0)),
                PhysicsOp::load("x"),
            ),
        ))
        .build();

    println!("Original: y = (x + 1) / x");

    if let Ok(ad_kernel) = AutoDiff::augment_forward_mode(&complex_kernel) {
        println!("Augmented: computes both y and dy");
        println!("Derivative: dy = (x * dx - (x + 1) * dx) / x^2 = -dx / x^2");
        println!(
            "\nFields in augmented kernel: {:?}",
            ad_kernel.fields.keys()
        );
        println!("Total operations: {}", ad_kernel.ops.len());
    }
}
