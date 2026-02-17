//! Integration tests for phyz-compile.

use phyz_compile::{AutoDiff, Compiler, FusionOptimizer, KernelBuilder, PhysicsOp, StencilBuilder};

#[test]
fn test_simple_kernel_compilation() {
    let kernel = KernelBuilder::new("simple")
        .field("T", [32, 32, 32])
        .op(PhysicsOp::store(
            "T",
            PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0)),
        ))
        .build();

    let mut compiler = Compiler::new();
    let result = compiler.compile(&kernel);

    assert!(result.is_ok());
    let compiled = result.unwrap();

    // Check that WGSL contains expected elements
    assert!(compiled.wgsl_source.contains("@compute"));
    assert!(compiled.wgsl_source.contains("@workgroup_size"));
    assert!(compiled.wgsl_source.contains("fn simple"));
    assert!(compiled.wgsl_source.contains("var<storage, read_write> T"));
}

#[test]
fn test_heat_equation_kernel() {
    let kernel = KernelBuilder::new("heat")
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

    let mut compiler = Compiler::new();
    let result = compiler.compile(&kernel);

    assert!(result.is_ok());
    let compiled = result.unwrap();

    assert_eq!(compiled.workgroup_size, [8, 8, 8]);
    assert!(compiled.fields.contains_key("T"));
    assert!(compiled.fields.contains_key("T_new"));
}

#[test]
fn test_stencil_operations() {
    // Test laplacian stencil
    let lap = StencilBuilder::laplacian_3d("T", 1.0);

    // Should involve loads with offsets
    match lap {
        PhysicsOp::BinOp { .. } => {}
        _ => panic!("Laplacian should be a binary operation"),
    }

    // Test averaging stencil
    let avg = StencilBuilder::average_3d("U");
    match avg {
        PhysicsOp::BinOp { .. } => {}
        _ => panic!("Average should be a binary operation"),
    }
}

#[test]
fn test_kernel_fusion_success() {
    // Two independent kernels operating on different fields
    let kernel1 = KernelBuilder::new("k1")
        .field("A", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "A",
            PhysicsOp::mul(PhysicsOp::load("A"), PhysicsOp::constant(2.0)),
        ))
        .build();

    let kernel2 = KernelBuilder::new("k2")
        .field("B", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "B",
            PhysicsOp::add(PhysicsOp::load("B"), PhysicsOp::constant(1.0)),
        ))
        .build();

    assert!(FusionOptimizer::can_fuse(&kernel1, &kernel2));

    let fused = FusionOptimizer::fuse(kernel1, kernel2);
    assert!(fused.is_ok());

    let fused = fused.unwrap();
    assert_eq!(fused.ops.len(), 2);
    assert_eq!(fused.fields.len(), 2);
}

#[test]
fn test_kernel_fusion_dependency_prevention() {
    // kernel1 writes to T
    let kernel1 = KernelBuilder::new("k1")
        .field("T", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("T", PhysicsOp::constant(1.0)))
        .build();

    // kernel2 reads from T and writes to U
    let kernel2 = KernelBuilder::new("k2")
        .field("T", [64, 64, 64])
        .field("U", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("U", PhysicsOp::load("T")))
        .build();

    // Should not be fusable due to dependency
    assert!(!FusionOptimizer::can_fuse(&kernel1, &kernel2));
}

#[test]
fn test_kernel_fusion_incompatible_domains() {
    let kernel1 = KernelBuilder::new("k1")
        .field("T", [64, 64, 64])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("T", PhysicsOp::constant(1.0)))
        .build();

    let kernel2 = KernelBuilder::new("k2")
        .field("T", [128, 128, 128]) // Different size
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store("T", PhysicsOp::constant(2.0)))
        .build();

    // Should not be fusable due to incompatible domains
    assert!(!FusionOptimizer::can_fuse(&kernel1, &kernel2));
}

#[test]
fn test_autodiff_simple() {
    let kernel = KernelBuilder::new("square")
        .field("x", [64, 64, 64])
        .field("y", [64, 64, 64])
        .op(PhysicsOp::store(
            "y",
            PhysicsOp::mul(PhysicsOp::load("x"), PhysicsOp::load("x")),
        ))
        .build();

    let ad_kernel = AutoDiff::augment_forward_mode(&kernel);
    assert!(ad_kernel.is_ok());

    let ad_kernel = ad_kernel.unwrap();

    // Should have tangent fields
    assert!(ad_kernel.fields.contains_key("x"));
    assert!(ad_kernel.fields.contains_key("y"));
    assert!(ad_kernel.fields.contains_key("dx"));
    assert!(ad_kernel.fields.contains_key("dy"));

    // Should have 2x operations (primal + tangent)
    assert_eq!(ad_kernel.ops.len(), 2);
}

#[test]
fn test_autodiff_compilation() {
    let kernel = KernelBuilder::new("linear")
        .field("x", [32, 32, 32])
        .field("y", [32, 32, 32])
        .tile_size([8, 8, 8])
        .op(PhysicsOp::store(
            "y",
            PhysicsOp::add(
                PhysicsOp::mul(PhysicsOp::load("x"), PhysicsOp::constant(2.0)),
                PhysicsOp::constant(1.0),
            ),
        ))
        .build();

    let ad_kernel = AutoDiff::augment_forward_mode(&kernel).unwrap();

    let mut compiler = Compiler::new();
    let result = compiler.compile(&ad_kernel);

    assert!(result.is_ok());
    let compiled = result.unwrap();

    // Should compile successfully and contain both primal and tangent fields
    assert!(compiled.wgsl_source.contains("var<storage, read_write> x"));
    assert!(compiled.wgsl_source.contains("var<storage, read_write> y"));
    assert!(compiled.wgsl_source.contains("var<storage, read_write> dx"));
    assert!(compiled.wgsl_source.contains("var<storage, read_write> dy"));
}

#[test]
fn test_schedule_hints() {
    let kernel = KernelBuilder::new("scheduled")
        .field("T", [64, 64, 64])
        .tile_size([16, 16, 4])
        .unroll()
        .prefetch()
        .fuse_next("next_kernel")
        .build();

    assert_eq!(kernel.schedule.tile_size, [16, 16, 4]);
    assert!(kernel.schedule.loop_unroll);
    assert!(kernel.schedule.prefetch);
    assert_eq!(kernel.schedule.fuse_next.as_deref(), Some("next_kernel"));

    let mut compiler = Compiler::new();
    let compiled = compiler.compile(&kernel).unwrap();

    assert_eq!(compiled.workgroup_size, [16, 16, 4]);
}

#[test]
fn test_validation_catches_unknown_field() {
    let kernel = KernelBuilder::new("invalid")
        .field("T", [64, 64, 64])
        .op(PhysicsOp::load("unknown_field"))
        .build();

    let mut compiler = Compiler::new();
    let result = compiler.compile(&kernel);

    assert!(result.is_err());
}

#[test]
fn test_compiled_kernel_output_structure() {
    let kernel = KernelBuilder::new("test")
        .field("A", [32, 32, 32])
        .field("B", [32, 32, 32])
        .tile_size([4, 4, 4])
        .op(PhysicsOp::store(
            "B",
            PhysicsOp::add(PhysicsOp::load("A"), PhysicsOp::constant(1.0)),
        ))
        .build();

    let mut compiler = Compiler::new();
    let compiled = compiler.compile(&kernel).unwrap();

    // Verify structure
    assert_eq!(compiled.workgroup_size, [4, 4, 4]);
    assert_eq!(compiled.fields.len(), 2);

    // Verify WGSL structure
    let wgsl = &compiled.wgsl_source;

    // Should have storage bindings
    assert!(wgsl.contains("@group(0) @binding"));

    // Should have compute shader entry point
    assert!(wgsl.contains("@compute"));
    assert!(wgsl.contains("@workgroup_size(4, 4, 4)"));

    // Should have bounds checking
    assert!(wgsl.contains("if (i >= nx || j >= ny || k >= nz)"));

    // Should have the actual computation
    assert!(wgsl.contains("["));
    assert!(wgsl.contains("]"));
}
