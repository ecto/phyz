//! Kernel fusion optimizer.
//!
//! Fuses adjacent stencil operations to reduce memory bandwidth and improve performance.

use crate::ir::{PhysicsOp, PhysicsProgram, ScheduleHints};
use std::collections::HashSet;

/// Kernel fusion analyzer and optimizer.
pub struct FusionOptimizer;

impl FusionOptimizer {
    /// Analyze two programs to determine if they can be fused.
    pub fn can_fuse(prog1: &PhysicsProgram, prog2: &PhysicsProgram) -> bool {
        // Check if programs have compatible domains (same field shapes)
        if !Self::compatible_domains(prog1, prog2) {
            return false;
        }

        // Check if schedules are compatible
        if !Self::compatible_schedules(&prog1.schedule, &prog2.schedule) {
            return false;
        }

        // Check for data dependencies that would prevent fusion
        !Self::has_blocking_dependency(prog1, prog2)
    }

    /// Fuse two programs into a single program.
    pub fn fuse(prog1: PhysicsProgram, prog2: PhysicsProgram) -> Result<PhysicsProgram, String> {
        if !Self::can_fuse(&prog1, &prog2) {
            return Err("Programs cannot be fused".to_string());
        }

        let mut fused = PhysicsProgram::new(format!("{}_fused_{}", prog1.name, prog2.name));

        // Merge fields
        for (name, meta) in prog1.fields.iter().chain(prog2.fields.iter()) {
            fused.fields.insert(name.clone(), meta.clone());
        }

        // Merge operations
        fused.ops.extend(prog1.ops);
        fused.ops.extend(prog2.ops);

        // Use the schedule hints from the first program
        fused.schedule = prog1.schedule;

        Ok(fused)
    }

    fn compatible_domains(prog1: &PhysicsProgram, prog2: &PhysicsProgram) -> bool {
        // Check if all shared fields have the same shape
        for (name, meta1) in &prog1.fields {
            if let Some(meta2) = prog2.fields.get(name)
                && (meta1.shape != meta2.shape || meta1.dtype != meta2.dtype)
            {
                return false;
            }
        }
        true
    }

    fn compatible_schedules(sched1: &ScheduleHints, sched2: &ScheduleHints) -> bool {
        // For now, require identical tile sizes
        sched1.tile_size == sched2.tile_size
    }

    fn has_blocking_dependency(prog1: &PhysicsProgram, prog2: &PhysicsProgram) -> bool {
        // Collect fields written by prog1
        let writes1 = Self::collect_writes(&prog1.ops);

        // Collect fields read by prog2
        let reads2 = Self::collect_reads(&prog2.ops);

        // If prog2 reads from a field that prog1 writes, there's a dependency
        // This is a conservative check - a full analysis would check indices
        !writes1.is_disjoint(&reads2)
    }

    fn collect_writes(ops: &[PhysicsOp]) -> HashSet<String> {
        let mut writes = HashSet::new();
        for op in ops {
            Self::collect_writes_recursive(op, &mut writes);
        }
        writes
    }

    fn collect_writes_recursive(op: &PhysicsOp, writes: &mut HashSet<String>) {
        match op {
            PhysicsOp::Store(field_ref, value) => {
                writes.insert(field_ref.name.clone());
                Self::collect_writes_recursive(value, writes);
            }
            PhysicsOp::BinOp { lhs, rhs, .. } => {
                Self::collect_writes_recursive(lhs, writes);
                Self::collect_writes_recursive(rhs, writes);
            }
            PhysicsOp::StencilOp { args, .. } => {
                for arg in args {
                    Self::collect_writes_recursive(arg, writes);
                }
            }
            _ => {}
        }
    }

    fn collect_reads(ops: &[PhysicsOp]) -> HashSet<String> {
        let mut reads = HashSet::new();
        for op in ops {
            Self::collect_reads_recursive(op, &mut reads);
        }
        reads
    }

    fn collect_reads_recursive(op: &PhysicsOp, reads: &mut HashSet<String>) {
        match op {
            PhysicsOp::Load(field_ref) => {
                reads.insert(field_ref.name.clone());
            }
            PhysicsOp::Store(_, value) => {
                Self::collect_reads_recursive(value, reads);
            }
            PhysicsOp::BinOp { lhs, rhs, .. } => {
                Self::collect_reads_recursive(lhs, reads);
                Self::collect_reads_recursive(rhs, reads);
            }
            PhysicsOp::StencilOp { args, .. } => {
                for arg in args {
                    Self::collect_reads_recursive(arg, reads);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FieldType, PhysicsOp};

    #[test]
    fn test_fusable_programs() {
        // Two programs operating on same field without dependency
        let mut prog1 = PhysicsProgram::new("prog1");
        prog1.add_field("T", [64, 64, 64], FieldType::F64);
        prog1.add_field("U", [64, 64, 64], FieldType::F64);
        prog1.add_op(PhysicsOp::store(
            "U",
            PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0)),
        ));

        let mut prog2 = PhysicsProgram::new("prog2");
        prog2.add_field("V", [64, 64, 64], FieldType::F64);
        prog2.add_op(PhysicsOp::store(
            "V",
            PhysicsOp::mul(PhysicsOp::load("V"), PhysicsOp::constant(2.0)),
        ));

        assert!(FusionOptimizer::can_fuse(&prog1, &prog2));

        let fused = FusionOptimizer::fuse(prog1, prog2);
        assert!(fused.is_ok());

        let fused = fused.unwrap();
        assert_eq!(fused.ops.len(), 2);
        assert_eq!(fused.fields.len(), 3);
    }

    #[test]
    fn test_dependency_prevents_fusion() {
        // prog1 writes to T, prog2 reads from T - dependency exists
        let mut prog1 = PhysicsProgram::new("prog1");
        prog1.add_field("T", [64, 64, 64], FieldType::F64);
        prog1.add_op(PhysicsOp::store("T", PhysicsOp::constant(1.0)));

        let mut prog2 = PhysicsProgram::new("prog2");
        prog2.add_field("T", [64, 64, 64], FieldType::F64);
        prog2.add_field("U", [64, 64, 64], FieldType::F64);
        prog2.add_op(PhysicsOp::store("U", PhysicsOp::load("T")));

        // This should fail because prog2 depends on prog1's output
        assert!(!FusionOptimizer::can_fuse(&prog1, &prog2));
    }

    #[test]
    fn test_incompatible_domains() {
        let mut prog1 = PhysicsProgram::new("prog1");
        prog1.add_field("T", [64, 64, 64], FieldType::F64);

        let mut prog2 = PhysicsProgram::new("prog2");
        prog2.add_field("T", [128, 128, 128], FieldType::F64); // Different size

        assert!(!FusionOptimizer::can_fuse(&prog1, &prog2));
    }
}
