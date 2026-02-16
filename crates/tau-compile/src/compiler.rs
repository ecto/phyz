//! Compiler from Physics IR to GPU compute shaders.
//!
//! Converts PhysicsProgram to WGSL compute shaders for execution on GPU.

use crate::ir::{BinOp, FieldMeta, FieldRef, FieldType, PhysicsOp, PhysicsProgram};
use std::collections::HashMap;

/// Compiled kernel ready for GPU execution.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub wgsl_source: String,
    pub workgroup_size: [u32; 3],
    pub fields: HashMap<String, FieldMeta>,
}

/// Compiler for physics kernels.
pub struct Compiler {
    next_temp_id: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Self { next_temp_id: 0 }
    }

    /// Compile a physics program to WGSL.
    pub fn compile(&mut self, program: &PhysicsProgram) -> Result<CompiledKernel, String> {
        program.validate()?;

        let mut code = String::new();

        // Generate struct definitions for fields
        self.generate_field_buffers(&mut code, &program.fields);

        // Generate main compute shader
        code.push_str("\n@compute @workgroup_size(");
        code.push_str(&program.schedule.tile_size[0].to_string());
        code.push_str(", ");
        code.push_str(&program.schedule.tile_size[1].to_string());
        code.push_str(", ");
        code.push_str(&program.schedule.tile_size[2].to_string());
        code.push_str(")\n");
        code.push_str(&format!("fn {}(\n", program.name));
        code.push_str("    @builtin(global_invocation_id) global_id: vec3<u32>,\n");
        code.push_str(") {\n");
        code.push_str("    let i = i32(global_id.x);\n");
        code.push_str("    let j = i32(global_id.y);\n");
        code.push_str("    let k = i32(global_id.z);\n\n");

        // Get grid dimensions from first field
        if let Some(field) = program.fields.values().next() {
            code.push_str(&format!("    let nx = {}i;\n", field.shape[0]));
            code.push_str(&format!("    let ny = {}i;\n", field.shape[1]));
            code.push_str(&format!("    let nz = {}i;\n\n", field.shape[2]));

            // Bounds check
            code.push_str("    if (i >= nx || j >= ny || k >= nz) {\n");
            code.push_str("        return;\n");
            code.push_str("    }\n\n");
        }

        // Generate computation
        self.next_temp_id = 0;
        for op in &program.ops {
            self.generate_op(&mut code, op, "    ")?;
        }

        code.push_str("}\n");

        Ok(CompiledKernel {
            wgsl_source: code,
            workgroup_size: program.schedule.tile_size,
            fields: program.fields.clone(),
        })
    }

    fn generate_field_buffers(&self, code: &mut String, fields: &HashMap<String, FieldMeta>) {
        for (idx, (name, meta)) in fields.iter().enumerate() {
            let wgsl_type = match meta.dtype {
                FieldType::F32 => "f32",
                FieldType::F64 => "f32", // WGSL doesn't have f64 in all contexts
            };

            code.push_str(&format!("@group(0) @binding({})\n", idx));
            code.push_str(&format!(
                "var<storage, read_write> {}: array<{}>;\n\n",
                name, wgsl_type
            ));
        }
    }

    fn generate_op(
        &mut self,
        code: &mut String,
        op: &PhysicsOp,
        indent: &str,
    ) -> Result<String, String> {
        match op {
            PhysicsOp::Load(field_ref) => {
                let idx_expr = self.compute_index(field_ref);
                Ok(format!("{}[{}]", field_ref.name, idx_expr))
            }

            PhysicsOp::Store(field_ref, value) => {
                let value_expr = self.generate_op(code, value, indent)?;
                let idx_expr = self.compute_index(field_ref);
                code.push_str(indent);
                code.push_str(&format!(
                    "{}[{}] = {};\n",
                    field_ref.name, idx_expr, value_expr
                ));
                Ok(String::new())
            }

            PhysicsOp::Constant(val) => Ok(format!("{:.10}", val)),

            PhysicsOp::BinOp { op, lhs, rhs } => {
                let lhs_expr = self.generate_op(code, lhs, indent)?;
                let rhs_expr = self.generate_op(code, rhs, indent)?;

                let op_str = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                };

                Ok(format!("({} {} {})", lhs_expr, op_str, rhs_expr))
            }

            PhysicsOp::StencilOp { name, args } => {
                // For stencil ops, generate a temporary variable
                let temp_name = format!("temp_{}", self.next_temp_id);
                self.next_temp_id += 1;

                code.push_str(indent);
                code.push_str(&format!("let {} = ", temp_name));

                // Evaluate stencil based on name
                match name.as_str() {
                    "laplacian" => {
                        // Simple 3D laplacian stencil
                        if args.is_empty() {
                            return Err("laplacian stencil requires field argument".to_string());
                        }
                        // For now, assume first arg is the field
                        // This is a simplified version - full implementation would need field analysis
                        code.push_str("0.0; // laplacian stencil\n");
                    }
                    _ => {
                        return Err(format!("Unknown stencil operation: {}", name));
                    }
                }

                Ok(temp_name)
            }
        }
    }

    fn compute_index(&self, field_ref: &FieldRef) -> String {
        let i = if field_ref.offset[0] == 0 {
            "i".to_string()
        } else if field_ref.offset[0] > 0 {
            format!("(i + {})", field_ref.offset[0])
        } else {
            format!("(i - {})", -field_ref.offset[0])
        };

        let j = if field_ref.offset[1] == 0 {
            "j".to_string()
        } else if field_ref.offset[1] > 0 {
            format!("(j + {})", field_ref.offset[1])
        } else {
            format!("(j - {})", -field_ref.offset[1])
        };

        let k = if field_ref.offset[2] == 0 {
            "k".to_string()
        } else if field_ref.offset[2] > 0 {
            format!("(k + {})", field_ref.offset[2])
        } else {
            format!("(k - {})", -field_ref.offset[2])
        };

        format!("({} + {} * nx + {} * nx * ny)", k, j, i)
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FieldType, PhysicsOp, PhysicsProgram};

    #[test]
    fn test_simple_kernel_compilation() {
        let mut prog = PhysicsProgram::new("simple");
        prog.add_field("T", [64, 64, 64], FieldType::F64);

        // T[i,j,k] = T[i,j,k] + 1.0
        prog.add_op(PhysicsOp::store(
            "T",
            PhysicsOp::add(PhysicsOp::load("T"), PhysicsOp::constant(1.0)),
        ));

        let mut compiler = Compiler::new();
        let result = compiler.compile(&prog);

        assert!(result.is_ok());
        let kernel = result.unwrap();
        assert!(kernel.wgsl_source.contains("@compute"));
        assert!(kernel.wgsl_source.contains("@workgroup_size"));
        assert!(kernel.wgsl_source.contains("fn simple"));
    }

    #[test]
    fn test_index_computation() {
        let compiler = Compiler::new();

        let field_ref = FieldRef::new("T");
        assert_eq!(
            compiler.compute_index(&field_ref),
            "(k + j * nx + i * nx * ny)"
        );

        let field_ref = FieldRef::with_offset("T", [1, 0, 0]);
        assert_eq!(
            compiler.compute_index(&field_ref),
            "(k + j * nx + (i + 1) * nx * ny)"
        );

        let field_ref = FieldRef::with_offset("T", [0, -1, 2]);
        assert_eq!(
            compiler.compute_index(&field_ref),
            "((k + 2) + (j - 1) * nx + i * nx * ny)"
        );
    }

    #[test]
    fn test_invalid_field_reference() {
        let mut prog = PhysicsProgram::new("invalid");
        prog.add_field("T", [64, 64, 64], FieldType::F64);
        prog.add_op(PhysicsOp::load("unknown"));

        let mut compiler = Compiler::new();
        let result = compiler.compile(&prog);

        assert!(result.is_err());
    }
}
