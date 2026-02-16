//! Physics IR (Intermediate Representation) for kernel compilation.
//!
//! Represents physics computations as an expression graph that can be
//! compiled to various backends (WGSL, SPIR-V, Metal).

use std::collections::HashMap;

/// Reference to a field in the physics computation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldRef {
    pub name: String,
    pub offset: [i32; 3], // [i, j, k] offset for stencil access
}

impl FieldRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            offset: [0, 0, 0],
        }
    }

    pub fn with_offset(name: impl Into<String>, offset: [i32; 3]) -> Self {
        Self {
            name: name.into(),
            offset,
        }
    }
}

/// Binary operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Physics operation in the IR.
#[derive(Debug, Clone)]
pub enum PhysicsOp {
    /// Load a value from a field
    Load(FieldRef),

    /// Store a value to a field
    Store(FieldRef, Box<PhysicsOp>),

    /// Stencil operation (named pattern)
    StencilOp { name: String, args: Vec<PhysicsOp> },

    /// Binary operation
    BinOp {
        op: BinOp,
        lhs: Box<PhysicsOp>,
        rhs: Box<PhysicsOp>,
    },

    /// Constant value
    Constant(f64),
}

impl PhysicsOp {
    /// Create a load operation
    pub fn load(field: impl Into<String>) -> Self {
        Self::Load(FieldRef::new(field))
    }

    /// Create a load with offset
    pub fn load_offset(field: impl Into<String>, offset: [i32; 3]) -> Self {
        Self::Load(FieldRef::with_offset(field, offset))
    }

    /// Create a store operation
    pub fn store(field: impl Into<String>, value: PhysicsOp) -> Self {
        Self::Store(FieldRef::new(field), Box::new(value))
    }

    /// Create a constant
    pub fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Create an addition operation (not the trait method)
    #[allow(clippy::should_implement_trait)]
    pub fn add(lhs: PhysicsOp, rhs: PhysicsOp) -> Self {
        Self::BinOp {
            op: BinOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create a subtraction operation (not the trait method)
    #[allow(clippy::should_implement_trait)]
    pub fn sub(lhs: PhysicsOp, rhs: PhysicsOp) -> Self {
        Self::BinOp {
            op: BinOp::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create a multiplication operation (not the trait method)
    #[allow(clippy::should_implement_trait)]
    pub fn mul(lhs: PhysicsOp, rhs: PhysicsOp) -> Self {
        Self::BinOp {
            op: BinOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create a division operation (not the trait method)
    #[allow(clippy::should_implement_trait)]
    pub fn div(lhs: PhysicsOp, rhs: PhysicsOp) -> Self {
        Self::BinOp {
            op: BinOp::Div,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

/// Metadata about a field in the computation.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub name: String,
    pub shape: [usize; 3],
    pub dtype: FieldType,
}

/// Field data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    F32,
    F64,
}

/// Scheduling hints for kernel optimization.
#[derive(Debug, Clone)]
pub struct ScheduleHints {
    pub tile_size: [u32; 3],
    pub loop_unroll: bool,
    pub prefetch: bool,
    pub fuse_next: Option<String>,
}

impl Default for ScheduleHints {
    fn default() -> Self {
        Self {
            tile_size: [16, 16, 16],
            loop_unroll: false,
            prefetch: false,
            fuse_next: None,
        }
    }
}

/// A complete physics program in IR form.
#[derive(Debug, Clone)]
pub struct PhysicsProgram {
    pub name: String,
    pub ops: Vec<PhysicsOp>,
    pub fields: HashMap<String, FieldMeta>,
    pub schedule: ScheduleHints,
}

impl PhysicsProgram {
    /// Create a new empty program.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ops: Vec::new(),
            fields: HashMap::new(),
            schedule: ScheduleHints::default(),
        }
    }

    /// Add a field to the program.
    pub fn add_field(&mut self, name: impl Into<String>, shape: [usize; 3], dtype: FieldType) {
        let name = name.into();
        self.fields
            .insert(name.clone(), FieldMeta { name, shape, dtype });
    }

    /// Add an operation to the program.
    pub fn add_op(&mut self, op: PhysicsOp) {
        self.ops.push(op);
    }

    /// Set scheduling hints.
    pub fn set_schedule(&mut self, schedule: ScheduleHints) {
        self.schedule = schedule;
    }

    /// Validate the program (check field references exist, etc.)
    pub fn validate(&self) -> Result<(), String> {
        for op in &self.ops {
            self.validate_op(op)?;
        }
        Ok(())
    }

    fn validate_op(&self, op: &PhysicsOp) -> Result<(), String> {
        match op {
            PhysicsOp::Load(field_ref) => {
                if !self.fields.contains_key(&field_ref.name) {
                    return Err(format!("Field '{}' not found", field_ref.name));
                }
            }
            PhysicsOp::Store(field_ref, value) => {
                if !self.fields.contains_key(&field_ref.name) {
                    return Err(format!("Field '{}' not found", field_ref.name));
                }
                self.validate_op(value)?;
            }
            PhysicsOp::StencilOp { args, .. } => {
                for arg in args {
                    self.validate_op(arg)?;
                }
            }
            PhysicsOp::BinOp { lhs, rhs, .. } => {
                self.validate_op(lhs)?;
                self.validate_op(rhs)?;
            }
            PhysicsOp::Constant(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_ref() {
        let field = FieldRef::new("T");
        assert_eq!(field.name, "T");
        assert_eq!(field.offset, [0, 0, 0]);

        let field = FieldRef::with_offset("T", [1, 0, -1]);
        assert_eq!(field.offset, [1, 0, -1]);
    }

    #[test]
    fn test_physics_op_builders() {
        let op = PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0));

        match op {
            PhysicsOp::BinOp { op: BinOp::Add, .. } => {}
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_program_validation() {
        let mut prog = PhysicsProgram::new("test");
        prog.add_field("T", [64, 64, 64], FieldType::F64);

        // Valid operation
        prog.add_op(PhysicsOp::load("T"));
        assert!(prog.validate().is_ok());

        // Invalid operation (unknown field)
        prog.add_op(PhysicsOp::load("unknown"));
        assert!(prog.validate().is_err());
    }
}
