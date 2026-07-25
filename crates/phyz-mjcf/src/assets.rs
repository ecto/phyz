//! `<asset>` records: meshes, textures, materials, height fields.
//!
//! Mesh files are loaded when they are STL or OBJ and present on disk; anything
//! else is recorded with `data: None` so the caller can see what was referenced
//! but not resolved.

use crate::attrs::Attrs;
use crate::{MjcfError, Result};
use phyz_math::Vec3;
use std::path::{Path, PathBuf};

/// Triangle soup loaded from a mesh file.
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<[usize; 3]>,
}

/// A `<mesh>` asset.
#[derive(Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    /// `file` attribute as written in the XML.
    pub file: Option<String>,
    /// `file` resolved against `meshdir`/`assetdir` and the model directory.
    pub resolved_path: Option<PathBuf>,
    pub scale: Vec3,
    /// Geometry, if the file was found and in a format we can read.
    pub data: Option<MeshData>,
    /// Why `data` is `None`, for diagnostics.
    pub load_error: Option<String>,
}

/// A `<texture>` asset. Recorded only; phyz has no renderer-side use for it yet.
#[derive(Debug, Clone)]
pub struct TextureAsset {
    pub name: Option<String>,
    pub texture_type: String,
    pub file: Option<String>,
    pub builtin: Option<String>,
    pub rgb1: Option<[f64; 3]>,
    pub rgb2: Option<[f64; 3]>,
}

/// A `<material>` asset.
#[derive(Debug, Clone)]
pub struct MaterialAsset {
    pub name: String,
    pub texture: Option<String>,
    pub rgba: Option<[f64; 4]>,
    pub specular: Option<f64>,
    pub shininess: Option<f64>,
}

/// An `<hfield>` asset. Recorded only: phyz-collision has no heightfield support.
#[derive(Debug, Clone)]
pub struct HFieldAsset {
    pub name: String,
    pub file: Option<String>,
    pub nrow: Option<usize>,
    pub ncol: Option<usize>,
    /// `size` = [radius_x, radius_y, elevation_z, base_z].
    pub size: Option<[f64; 4]>,
}

pub(crate) fn parse_mesh(attrs: &Attrs, asset_dir: Option<&Path>) -> Result<MeshAsset> {
    let file = attrs.string("file");
    let name = match attrs.string("name") {
        Some(n) => n,
        // MuJoCo defaults a mesh's name to its file stem.
        None => file
            .as_deref()
            .and_then(|f| Path::new(f).file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .ok_or_else(|| MjcfError::MissingAttribute {
                element: "mesh".to_string(),
                attribute: "name".to_string(),
            })?,
    };

    let scale = attrs.vec3_or("scale", Vec3::new(1.0, 1.0, 1.0))?;

    let resolved_path = file.as_deref().map(|f| match asset_dir {
        Some(dir) => dir.join(f),
        None => PathBuf::from(f),
    });

    let (data, load_error) = match resolved_path.as_deref() {
        None => (None, Some("mesh has no 'file' attribute".to_string())),
        Some(path) => match load_mesh_file(path) {
            Ok(mut mesh) => {
                if scale != Vec3::new(1.0, 1.0, 1.0) {
                    for v in &mut mesh.vertices {
                        *v = Vec3::new(v.x * scale.x, v.y * scale.y, v.z * scale.z);
                    }
                }
                (Some(mesh), None)
            }
            Err(e) => (None, Some(e)),
        },
    };

    Ok(MeshAsset {
        name,
        file,
        resolved_path,
        scale,
        data,
        load_error,
    })
}

pub(crate) fn parse_texture(attrs: &Attrs) -> Result<TextureAsset> {
    Ok(TextureAsset {
        name: attrs.string("name"),
        texture_type: attrs.string("type").unwrap_or_else(|| "cube".to_string()),
        file: attrs.string("file"),
        builtin: attrs.string("builtin"),
        rgb1: attrs.fixed::<3>("rgb1")?,
        rgb2: attrs.fixed::<3>("rgb2")?,
    })
}

pub(crate) fn parse_material(attrs: &Attrs) -> Result<MaterialAsset> {
    Ok(MaterialAsset {
        name: attrs.required("name")?.to_string(),
        texture: attrs.string("texture"),
        rgba: attrs.fixed::<4>("rgba")?,
        specular: attrs.f64("specular")?,
        shininess: attrs.f64("shininess")?,
    })
}

pub(crate) fn parse_hfield(attrs: &Attrs) -> Result<HFieldAsset> {
    let usize_attr = |key: &str| -> Result<Option<usize>> {
        match attrs.f64(key)? {
            None => Ok(None),
            Some(v) if v >= 0.0 && v.fract() == 0.0 => Ok(Some(v as usize)),
            Some(v) => Err(MjcfError::invalid_attr(
                "hfield",
                key,
                &v.to_string(),
                "expected a non-negative integer",
            )),
        }
    };
    Ok(HFieldAsset {
        name: attrs.required("name")?.to_string(),
        file: attrs.string("file"),
        nrow: usize_attr("nrow")?,
        ncol: usize_attr("ncol")?,
        size: attrs.fixed::<4>("size")?,
    })
}

/// Read a mesh file, dispatching on extension. Errors are returned as strings
/// because a missing or exotic mesh is recorded, not fatal.
fn load_mesh_file(path: &Path) -> std::result::Result<MeshData, String> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "stl" => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            load_stl(&bytes).map_err(|e| format!("{}: {e}", path.display()))
        }
        "obj" => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            load_obj(&text).map_err(|e| format!("{}: {e}", path.display()))
        }
        other => Err(format!(
            "unsupported mesh format '{other}' for {} (STL and OBJ are supported)",
            path.display()
        )),
    }
}

/// Parse binary or ASCII STL.
fn load_stl(bytes: &[u8]) -> std::result::Result<MeshData, String> {
    // An ASCII STL starts with "solid" — but so can a badly-written binary one, so
    // confirm with the size the binary header implies before committing.
    let looks_ascii = bytes.starts_with(b"solid");
    if bytes.len() >= 84 {
        let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if 84 + tri_count * 50 == bytes.len() {
            return load_stl_binary(bytes, tri_count);
        }
    }
    if looks_ascii {
        let text = std::str::from_utf8(bytes).map_err(|e| format!("not valid UTF-8: {e}"))?;
        return load_stl_ascii(text);
    }
    Err("file is neither a valid binary nor ASCII STL".to_string())
}

fn load_stl_binary(bytes: &[u8], tri_count: usize) -> std::result::Result<MeshData, String> {
    let mut mesh = MeshData::default();
    for i in 0..tri_count {
        let base = 84 + i * 50 + 12; // skip the per-facet normal
        let mut tri = [0usize; 3];
        for (c, slot) in tri.iter_mut().enumerate() {
            let o = base + c * 12;
            let f = |k: usize| -> f64 {
                f32::from_le_bytes([
                    bytes[o + k * 4],
                    bytes[o + k * 4 + 1],
                    bytes[o + k * 4 + 2],
                    bytes[o + k * 4 + 3],
                ]) as f64
            };
            mesh.vertices.push(Vec3::new(f(0), f(1), f(2)));
            *slot = mesh.vertices.len() - 1;
        }
        mesh.faces.push(tri);
    }
    Ok(mesh)
}

fn load_stl_ascii(text: &str) -> std::result::Result<MeshData, String> {
    let mut mesh = MeshData::default();
    let mut pending: Vec<usize> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("vertex") {
            let nums: std::result::Result<Vec<f64>, _> =
                rest.split_whitespace().map(str::parse::<f64>).collect();
            let nums = nums.map_err(|e| format!("line {}: bad vertex ({e})", lineno + 1))?;
            if nums.len() != 3 {
                return Err(format!("line {}: vertex needs 3 numbers", lineno + 1));
            }
            mesh.vertices.push(Vec3::new(nums[0], nums[1], nums[2]));
            pending.push(mesh.vertices.len() - 1);
        } else if line.starts_with("endfacet") {
            if pending.len() == 3 {
                mesh.faces.push([pending[0], pending[1], pending[2]]);
            }
            pending.clear();
        }
    }
    if mesh.faces.is_empty() {
        return Err("no facets found".to_string());
    }
    Ok(mesh)
}

/// Parse the subset of OBJ that describes geometry: `v` and `f` (triangulating
/// convex polygons by fanning).
fn load_obj(text: &str) -> std::result::Result<MeshData, String> {
    let mut mesh = MeshData::default();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let nums: std::result::Result<Vec<f64>, _> = rest
                .split_whitespace()
                .take(3)
                .map(str::parse::<f64>)
                .collect();
            let nums = nums.map_err(|e| format!("line {}: bad vertex ({e})", lineno + 1))?;
            if nums.len() != 3 {
                return Err(format!("line {}: vertex needs 3 numbers", lineno + 1));
            }
            mesh.vertices.push(Vec3::new(nums[0], nums[1], nums[2]));
        } else if let Some(rest) = line.strip_prefix("f ") {
            let mut idxs = Vec::new();
            for tok in rest.split_whitespace() {
                // "v", "v/vt", "v//vn", "v/vt/vn"
                let v = tok.split('/').next().unwrap_or(tok);
                let raw: i64 = v
                    .parse()
                    .map_err(|e| format!("line {}: bad face index '{v}' ({e})", lineno + 1))?;
                // OBJ indices are 1-based; negatives count back from the end.
                let idx = if raw > 0 {
                    (raw - 1) as usize
                } else if raw < 0 {
                    let back = (-raw) as usize;
                    mesh.vertices.len().checked_sub(back).ok_or_else(|| {
                        format!("line {}: face index {raw} out of range", lineno + 1)
                    })?
                } else {
                    return Err(format!("line {}: face index 0 is invalid", lineno + 1));
                };
                if idx >= mesh.vertices.len() {
                    return Err(format!(
                        "line {}: face index {raw} exceeds {} vertices",
                        lineno + 1,
                        mesh.vertices.len()
                    ));
                }
                idxs.push(idx);
            }
            for i in 1..idxs.len().saturating_sub(1) {
                mesh.faces.push([idxs[0], idxs[i], idxs[i + 1]]);
            }
        }
    }
    if mesh.faces.is_empty() {
        return Err("no faces found".to_string());
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_triangulates_quads() {
        let obj = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";
        let mesh = load_obj(obj).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);
    }

    #[test]
    fn obj_handles_index_forms_and_rejects_overflow() {
        let mesh = load_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2//2 -1\n").unwrap();
        assert_eq!(mesh.faces, vec![[0, 1, 2]]);
        assert!(load_obj("v 0 0 0\nf 1 2 3\n").is_err());
    }

    #[test]
    fn ascii_stl_round_trips() {
        let stl = "\
solid t
 facet normal 0 0 1
  outer loop
   vertex 0 0 0
   vertex 1 0 0
   vertex 0 1 0
  endloop
 endfacet
endsolid t
";
        let mesh = load_stl(stl.as_bytes()).unwrap();
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.vertices.len(), 3);
    }

    #[test]
    fn binary_stl_round_trips() {
        let mut bytes = vec![0u8; 80];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for f in [0.0f32, 0.0, 1.0] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for f in v {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
        let mesh = load_stl(&bytes).unwrap();
        assert_eq!(mesh.faces.len(), 1);
        assert!((mesh.vertices[1].x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn garbage_is_reported_not_panicked() {
        assert!(load_stl(b"not a mesh at all").is_err());
    }
}
