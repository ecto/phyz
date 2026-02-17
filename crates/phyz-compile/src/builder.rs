//! Builder API for constructing physics kernels.
//!
//! Provides a fluent interface for building physics programs.

use crate::ir::{FieldType, PhysicsOp, PhysicsProgram, ScheduleHints};

/// Builder for physics kernels.
pub struct KernelBuilder {
    program: PhysicsProgram,
}

impl KernelBuilder {
    /// Create a new kernel builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            program: PhysicsProgram::new(name),
        }
    }

    /// Add a 3D field to the kernel.
    pub fn field(mut self, name: impl Into<String>, shape: [usize; 3]) -> Self {
        self.program.add_field(name, shape, FieldType::F64);
        self
    }

    /// Add a 3D field with specific type.
    pub fn field_typed(
        mut self,
        name: impl Into<String>,
        shape: [usize; 3],
        dtype: FieldType,
    ) -> Self {
        self.program.add_field(name, shape, dtype);
        self
    }

    /// Set the workgroup tile size.
    pub fn tile_size(mut self, size: [u32; 3]) -> Self {
        self.program.schedule.tile_size = size;
        self
    }

    /// Enable loop unrolling.
    pub fn unroll(mut self) -> Self {
        self.program.schedule.loop_unroll = true;
        self
    }

    /// Enable prefetching.
    pub fn prefetch(mut self) -> Self {
        self.program.schedule.prefetch = true;
        self
    }

    /// Mark this kernel for fusion with the next one.
    pub fn fuse_next(mut self, next_kernel: impl Into<String>) -> Self {
        self.program.schedule.fuse_next = Some(next_kernel.into());
        self
    }

    /// Set complete scheduling hints.
    pub fn schedule(mut self, hints: ScheduleHints) -> Self {
        self.program.schedule = hints;
        self
    }

    /// Add an operation to the kernel.
    pub fn op(mut self, operation: PhysicsOp) -> Self {
        self.program.add_op(operation);
        self
    }

    /// Build the final program.
    pub fn build(self) -> PhysicsProgram {
        self.program
    }
}

/// Stencil builder for common stencil patterns.
pub struct StencilBuilder;

impl StencilBuilder {
    /// Create a 3D Laplacian stencil operation.
    ///
    /// Computes: (T[i+1,j,k] + T[i-1,j,k] + T[i,j+1,k] + T[i,j-1,k] + T[i,j,k+1] + T[i,j,k-1] - 6*T[i,j,k]) / dx^2
    pub fn laplacian_3d(field: impl Into<String>, dx: f64) -> PhysicsOp {
        let field = field.into();

        // Sum of neighbors
        let neighbors = PhysicsOp::add(
            PhysicsOp::add(
                PhysicsOp::load_offset(&field, [1, 0, 0]),
                PhysicsOp::load_offset(&field, [-1, 0, 0]),
            ),
            PhysicsOp::add(
                PhysicsOp::add(
                    PhysicsOp::load_offset(&field, [0, 1, 0]),
                    PhysicsOp::load_offset(&field, [0, -1, 0]),
                ),
                PhysicsOp::add(
                    PhysicsOp::load_offset(&field, [0, 0, 1]),
                    PhysicsOp::load_offset(&field, [0, 0, -1]),
                ),
            ),
        );

        // Subtract 6 * center
        let center = PhysicsOp::mul(PhysicsOp::constant(6.0), PhysicsOp::load(&field));

        let laplacian = PhysicsOp::sub(neighbors, center);

        // Divide by dx^2
        PhysicsOp::div(laplacian, PhysicsOp::constant(dx * dx))
    }

    /// Create a simple averaging stencil (nearest neighbors).
    pub fn average_3d(field: impl Into<String>) -> PhysicsOp {
        let field = field.into();

        let sum = PhysicsOp::add(
            PhysicsOp::add(
                PhysicsOp::load_offset(&field, [1, 0, 0]),
                PhysicsOp::load_offset(&field, [-1, 0, 0]),
            ),
            PhysicsOp::add(
                PhysicsOp::add(
                    PhysicsOp::load_offset(&field, [0, 1, 0]),
                    PhysicsOp::load_offset(&field, [0, -1, 0]),
                ),
                PhysicsOp::add(
                    PhysicsOp::load_offset(&field, [0, 0, 1]),
                    PhysicsOp::load_offset(&field, [0, 0, -1]),
                ),
            ),
        );

        PhysicsOp::div(sum, PhysicsOp::constant(6.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_builder() {
        let prog = KernelBuilder::new("test")
            .field("T", [64, 64, 64])
            .field("U", [64, 64, 64])
            .tile_size([8, 8, 8])
            .unroll()
            .op(PhysicsOp::store(
                "U",
                PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0)),
            ))
            .build();

        assert_eq!(prog.name, "test");
        assert_eq!(prog.fields.len(), 2);
        assert_eq!(prog.ops.len(), 1);
        assert_eq!(prog.schedule.tile_size, [8, 8, 8]);
        assert!(prog.schedule.loop_unroll);
    }

    #[test]
    fn test_laplacian_stencil() {
        let lap = StencilBuilder::laplacian_3d("T", 1.0);

        // Should be a division
        match lap {
            PhysicsOp::BinOp {
                op: crate::ir::BinOp::Div,
                ..
            } => {}
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_average_stencil() {
        let avg = StencilBuilder::average_3d("T");

        // Should be a division by 6
        match avg {
            PhysicsOp::BinOp {
                op: crate::ir::BinOp::Div,
                rhs,
                ..
            } => match *rhs {
                PhysicsOp::Constant(v) => assert!((v - 6.0).abs() < 1e-10),
                _ => panic!("Expected constant 6.0"),
            },
            _ => panic!("Expected division"),
        }
    }
}
