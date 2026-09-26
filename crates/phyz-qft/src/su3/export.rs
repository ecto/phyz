//! `.qcdf` field files: a one-line JSON header, then each field as
//! little-endian `f32` in lattice site order (`t + nt (x + nx (y + ny z))`).
//!
//! ```text
//! {"dims":[nt,nx,ny,nz],"a_fm":0.093,"fields":["action","topo"],"quarks":[[x,y,z],…]}\n
//! <f32 × n_sites> per field, in header order
//! ```
//!
//! Kept dependency-free so any viewer can read it; kosm's `kosm-qcd` does.

use std::io::{self, Write};
use std::path::Path;

pub struct FieldFile<'a> {
    pub dims: [usize; 4],
    /// Lattice spacing in fm, for axis labels.
    pub a_fm: f64,
    pub fields: Vec<(&'a str, &'a [f64])>,
    /// Static quark positions in lattice units (x, y, z), if any.
    pub quarks: Vec<[f64; 3]>,
}

impl FieldFile<'_> {
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let n: usize = self.dims.iter().product();
        for (name, data) in &self.fields {
            if data.len() != n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "field {name} has {} values, lattice has {n} sites",
                        data.len()
                    ),
                ));
            }
        }
        let names: Vec<String> = self.fields.iter().map(|(n, _)| format!("{n:?}")).collect();
        let quarks: Vec<String> = self
            .quarks
            .iter()
            .map(|[x, y, z]| format!("[{x},{y},{z}]"))
            .collect();
        let [nt, nx, ny, nz] = self.dims;
        let header = format!(
            "{{\"dims\":[{nt},{nx},{ny},{nz}],\"a_fm\":{},\"fields\":[{}],\"quarks\":[{}]}}\n",
            self.a_fm,
            names.join(","),
            quarks.join(",")
        );
        let mut out = io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(header.as_bytes())?;
        for (_, data) in &self.fields {
            for &v in *data {
                out.write_all(&(v as f32).to_le_bytes())?;
            }
        }
        out.flush()
    }
}
