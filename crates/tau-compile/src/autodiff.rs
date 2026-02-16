//! Automatic differentiation for physics kernels.
//!
//! Implements forward-mode AD to augment kernels with tangent computations.

use crate::ir::{BinOp, FieldMeta, PhysicsOp, PhysicsProgram};

/// Automatic differentiation transformer.
pub struct AutoDiff;

impl AutoDiff {
    /// Augment a program with forward-mode automatic differentiation.
    ///
    /// For each field F, creates a tangent field dF. The augmented program
    /// computes both primal values and derivatives.
    pub fn augment_forward_mode(program: &PhysicsProgram) -> Result<PhysicsProgram, String> {
        let mut augmented = PhysicsProgram::new(format!("{}_ad", program.name));
        augmented.schedule = program.schedule.clone();

        // Add primal and tangent fields
        for (name, meta) in &program.fields {
            // Primal field
            augmented.fields.insert(name.clone(), meta.clone());

            // Tangent field (derivative)
            let tangent_name = format!("d{}", name);
            augmented.fields.insert(
                tangent_name.clone(),
                FieldMeta {
                    name: tangent_name,
                    shape: meta.shape,
                    dtype: meta.dtype,
                },
            );
        }

        // Augment each operation
        for op in &program.ops {
            // Add primal operation
            augmented.ops.push(op.clone());

            // Add tangent operation
            let tangent_op = Self::differentiate_op(op)?;
            augmented.ops.push(tangent_op);
        }

        Ok(augmented)
    }

    /// Differentiate a single operation using forward-mode AD rules.
    fn differentiate_op(op: &PhysicsOp) -> Result<PhysicsOp, String> {
        match op {
            PhysicsOp::Load(field_ref) => {
                // dF = load(dF)
                let mut tangent_ref = field_ref.clone();
                tangent_ref.name = format!("d{}", field_ref.name);
                Ok(PhysicsOp::Load(tangent_ref))
            }

            PhysicsOp::Store(field_ref, value) => {
                // dF = d(value)
                let mut tangent_ref = field_ref.clone();
                tangent_ref.name = format!("d{}", field_ref.name);
                let tangent_value = Self::differentiate_op(value)?;
                Ok(PhysicsOp::Store(tangent_ref, Box::new(tangent_value)))
            }

            PhysicsOp::Constant(_) => {
                // d(constant) = 0
                Ok(PhysicsOp::Constant(0.0))
            }

            PhysicsOp::BinOp { op, lhs, rhs } => {
                // Apply differentiation rules
                let d_lhs = Self::differentiate_op(lhs)?;
                let d_rhs = Self::differentiate_op(rhs)?;

                match op {
                    BinOp::Add => {
                        // d(u + v) = du + dv
                        Ok(PhysicsOp::add(d_lhs, d_rhs))
                    }
                    BinOp::Sub => {
                        // d(u - v) = du - dv
                        Ok(PhysicsOp::sub(d_lhs, d_rhs))
                    }
                    BinOp::Mul => {
                        // d(u * v) = u * dv + v * du (product rule)
                        Ok(PhysicsOp::add(
                            PhysicsOp::mul(*lhs.clone(), d_rhs),
                            PhysicsOp::mul(*rhs.clone(), d_lhs),
                        ))
                    }
                    BinOp::Div => {
                        // d(u / v) = (v * du - u * dv) / v^2 (quotient rule)
                        let numerator = PhysicsOp::sub(
                            PhysicsOp::mul(*rhs.clone(), d_lhs),
                            PhysicsOp::mul(*lhs.clone(), d_rhs),
                        );
                        let denominator = PhysicsOp::mul(*rhs.clone(), *rhs.clone());
                        Ok(PhysicsOp::div(numerator, denominator))
                    }
                }
            }

            PhysicsOp::StencilOp { name, args } => {
                // Differentiate stencil operations
                let d_args: Result<Vec<_>, _> = args.iter().map(Self::differentiate_op).collect();
                Ok(PhysicsOp::StencilOp {
                    name: format!("d{}", name),
                    args: d_args?,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FieldType, PhysicsOp};

    #[test]
    fn test_differentiate_constant() {
        let op = PhysicsOp::Constant(3.14);
        let d_op = AutoDiff::differentiate_op(&op).unwrap();

        match d_op {
            PhysicsOp::Constant(v) => assert_eq!(v, 0.0),
            _ => panic!("Expected constant"),
        }
    }

    #[test]
    fn test_differentiate_addition() {
        // d(u + v) = du + dv
        let op = PhysicsOp::add(PhysicsOp::load("u"), PhysicsOp::load("v"));
        let d_op = AutoDiff::differentiate_op(&op).unwrap();

        match d_op {
            PhysicsOp::BinOp {
                op: BinOp::Add,
                lhs,
                rhs,
            } => {
                // Check that we have d(u) and d(v)
                match *lhs {
                    PhysicsOp::Load(ref field) => assert_eq!(field.name, "du"),
                    _ => panic!("Expected load of du"),
                }
                match *rhs {
                    PhysicsOp::Load(ref field) => assert_eq!(field.name, "dv"),
                    _ => panic!("Expected load of dv"),
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_differentiate_multiplication() {
        // d(u * v) = u * dv + v * du
        let op = PhysicsOp::mul(PhysicsOp::load("u"), PhysicsOp::load("v"));
        let d_op = AutoDiff::differentiate_op(&op).unwrap();

        match d_op {
            PhysicsOp::BinOp { op: BinOp::Add, .. } => {
                // Product rule produces a sum
            }
            _ => panic!("Expected addition from product rule"),
        }
    }

    #[test]
    fn test_augment_program() {
        let mut prog = PhysicsProgram::new("test");
        prog.add_field("T", [64, 64, 64], FieldType::F64);
        prog.add_op(PhysicsOp::store(
            "T",
            PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0)),
        ));

        let augmented = AutoDiff::augment_forward_mode(&prog).unwrap();

        // Should have both T and dT fields
        assert!(augmented.fields.contains_key("T"));
        assert!(augmented.fields.contains_key("dT"));

        // Should have 2x operations (primal + tangent)
        assert_eq!(augmented.ops.len(), 2);
    }
}
