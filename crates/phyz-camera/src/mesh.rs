//! CPU triangle meshes: tessellation of `phyz_model::Geometry` and STL loading.
//!
//! Everything here is flat-shaded: each triangle carries its own three vertices
//! with the face normal duplicated onto them. That triples the vertex count of a
//! smooth surface, which at 64–128 px render targets is entirely irrelevant, and
//! it keeps depth exact for the flat primitives (boxes, planes) that the
//! analytic tests rely on.

use crate::error::{CameraError, Result};
use phyz_math::Vec3;
use phyz_model::Geometry;

/// A single vertex in mesh-local coordinates.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Position in mesh-local coordinates, metres.
    pub position: [f32; 3],
    /// Unit surface normal in mesh-local coordinates.
    pub normal: [f32; 3],
    /// Linear RGB tint in `[0, 1]`, **multiplied** by the instance albedo.
    ///
    /// White is the identity, which is what every geometric builder writes, so
    /// a mesh nobody has painted renders exactly as it did before this field
    /// existed. It carries measured appearance for meshes that have one — a
    /// scanned room whose colour comes from the camera that mapped it — where
    /// one albedo for the whole surface would flatten a garage into a grey box.
    pub color: [f32; 3],
}

/// The vertex tint that changes nothing: `albedo * WHITE == albedo`.
pub const UNPAINTED: [f32; 3] = [1.0, 1.0, 1.0];

/// A flat-shaded triangle soup in mesh-local coordinates.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    /// Vertices, three per triangle (no index buffer; see the module docs).
    pub vertices: Vec<Vertex>,
}

impl TriMesh {
    /// An empty mesh, which renders nothing.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// True if there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Append one triangle, deriving a flat normal from the winding.
    ///
    /// Vertices are expected counter-clockwise when viewed from outside the
    /// surface, which is what makes the derived normal point outward. Degenerate
    /// triangles are dropped rather than emitting a NaN normal.
    pub fn push_triangle(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        self.push_triangle_painted(a, b, c, [UNPAINTED; 3]);
    }

    /// Append one triangle carrying a colour per corner.
    ///
    /// Same winding rule and same degenerate-drop as [`Self::push_triangle`];
    /// the colours are interpolated across the face by the rasterizer, so a
    /// scanned surface reads as a gradient rather than as facets.
    pub fn push_triangle_painted(
        &mut self,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        colors: [[f32; 3]; 3],
    ) {
        let Some(n) = (b - a).cross(c - a).try_normalize() else {
            return;
        };
        let n = [n.x as f32, n.y as f32, n.z as f32];
        for (p, color) in [a, b, c].into_iter().zip(colors) {
            self.vertices.push(Vertex {
                position: [p.x as f32, p.y as f32, p.z as f32],
                normal: n,
                color,
            });
        }
    }

    /// Append a planar quad as two triangles, wound `a → b → c → d`.
    pub fn push_quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        self.push_triangle(a, b, c);
        self.push_triangle(a, c, d);
    }

    /// Scale every vertex position by a per-axis factor.
    pub fn scaled(mut self, scale: Vec3) -> Self {
        let s = [scale.x as f32, scale.y as f32, scale.z as f32];
        for v in &mut self.vertices {
            for (p, s) in v.position.iter_mut().zip(s) {
                *p *= s;
            }
            // Normals transform by the inverse transpose; for a diagonal scale
            // that is a reciprocal scale, then renormalise.
            let mut n = [0.0f32; 3];
            for ((n, &vn), s) in n.iter_mut().zip(v.normal.iter()).zip(s) {
                *n = if s != 0.0 { vn / s } else { vn };
            }
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                v.normal = [n[0] / len, n[1] / len, n[2] / len];
            }
        }
        self
    }
}

/// How finely curved primitives are tessellated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tessellation {
    /// Segments around the axis of revolution.
    pub segments: u32,
    /// Stacks from pole to pole (spheres and capsule caps).
    pub rings: u32,
    /// Half-extent of the quad standing in for an infinite `Geometry::Plane`,
    /// in metres. A plane has no finite geometry, so it is drawn as a large
    /// square centred on the shape origin; anything past this reads as "no
    /// return" rather than as ground.
    pub plane_extent: f64,
}

impl Default for Tessellation {
    fn default() -> Self {
        Self {
            segments: 32,
            rings: 16,
            plane_extent: 50.0,
        }
    }
}

/// Tessellate a `phyz_model::Geometry` into a flat-shaded mesh in shape-local
/// coordinates.
///
/// Axis conventions follow `phyz-model`: capsules and cylinders run along local
/// `Z`, boxes are centred on the origin.
pub fn tessellate(geometry: &Geometry, tess: &Tessellation) -> TriMesh {
    match geometry {
        Geometry::Sphere { radius } => sphere(*radius, tess),
        Geometry::Box { half_extents } => cuboid(*half_extents),
        Geometry::Cylinder { radius, height } => cylinder(*radius, *height, tess),
        Geometry::Capsule { radius, length } => capsule(*radius, *length, tess),
        Geometry::Plane { normal } => plane(*normal, tess.plane_extent),
        Geometry::Mesh { vertices, faces } => from_indexed(vertices, faces),
    }
}

/// Build a mesh from an explicit vertex/face list, discarding out-of-range
/// faces rather than panicking on malformed geometry.
pub fn from_indexed(vertices: &[Vec3], faces: &[[usize; 3]]) -> TriMesh {
    let mut m = TriMesh::empty();
    for f in faces {
        if f.iter().any(|&i| i >= vertices.len()) {
            continue;
        }
        m.push_triangle(vertices[f[0]], vertices[f[1]], vertices[f[2]]);
    }
    m
}

/// An axis-aligned box centred on the origin.
pub fn cuboid(half_extents: Vec3) -> TriMesh {
    let (x, y, z) = (half_extents.x, half_extents.y, half_extents.z);
    let v = |sx: f64, sy: f64, sz: f64| Vec3::new(sx * x, sy * y, sz * z);
    let mut m = TriMesh::empty();
    // +X, -X, +Y, -Y, +Z, -Z, each wound CCW from outside.
    m.push_quad(
        v(1., -1., -1.),
        v(1., 1., -1.),
        v(1., 1., 1.),
        v(1., -1., 1.),
    );
    m.push_quad(
        v(-1., 1., -1.),
        v(-1., -1., -1.),
        v(-1., -1., 1.),
        v(-1., 1., 1.),
    );
    m.push_quad(
        v(1., 1., -1.),
        v(-1., 1., -1.),
        v(-1., 1., 1.),
        v(1., 1., 1.),
    );
    m.push_quad(
        v(-1., -1., -1.),
        v(1., -1., -1.),
        v(1., -1., 1.),
        v(-1., -1., 1.),
    );
    m.push_quad(
        v(-1., -1., 1.),
        v(1., -1., 1.),
        v(1., 1., 1.),
        v(-1., 1., 1.),
    );
    m.push_quad(
        v(-1., 1., -1.),
        v(1., 1., -1.),
        v(1., -1., -1.),
        v(-1., -1., -1.),
    );
    m
}

/// A UV sphere centred on the origin.
pub fn sphere(radius: f64, tess: &Tessellation) -> TriMesh {
    let mut m = TriMesh::empty();
    let (nseg, nring) = (tess.segments.max(3), tess.rings.max(2));
    let p = |i: u32, j: u32| -> Vec3 {
        let theta = std::f64::consts::PI * j as f64 / nring as f64;
        let phi = std::f64::consts::TAU * i as f64 / nseg as f64;
        Vec3::new(
            radius * theta.sin() * phi.cos(),
            radius * theta.sin() * phi.sin(),
            radius * theta.cos(),
        )
    };
    for j in 0..nring {
        for i in 0..nseg {
            let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            m.push_triangle(a, d, c);
            m.push_triangle(a, c, b);
        }
    }
    m
}

/// A closed cylinder along local `Z`, centred on the origin.
pub fn cylinder(radius: f64, height: f64, tess: &Tessellation) -> TriMesh {
    let mut m = TriMesh::empty();
    let nseg = tess.segments.max(3);
    let hz = height / 2.0;
    let rim = |i: u32, z: f64| {
        let phi = std::f64::consts::TAU * i as f64 / nseg as f64;
        Vec3::new(radius * phi.cos(), radius * phi.sin(), z)
    };
    for i in 0..nseg {
        m.push_quad(rim(i, -hz), rim(i + 1, -hz), rim(i + 1, hz), rim(i, hz));
        m.push_triangle(Vec3::new(0.0, 0.0, hz), rim(i, hz), rim(i + 1, hz));
        m.push_triangle(Vec3::new(0.0, 0.0, -hz), rim(i + 1, -hz), rim(i, -hz));
    }
    m
}

/// A capsule along local `Z`: a cylinder of `length` capped by hemispheres of
/// `radius`, matching `phyz_model::Geometry::Capsule`.
pub fn capsule(radius: f64, length: f64, tess: &Tessellation) -> TriMesh {
    let mut m = TriMesh::empty();
    let (nseg, nring) = (tess.segments.max(3), tess.rings.max(2));
    let hz = length / 2.0;
    let rim = |i: u32, z: f64| {
        let phi = std::f64::consts::TAU * i as f64 / nseg as f64;
        Vec3::new(radius * phi.cos(), radius * phi.sin(), z)
    };
    for i in 0..nseg {
        m.push_quad(rim(i, -hz), rim(i + 1, -hz), rim(i + 1, hz), rim(i, hz));
    }
    // Hemispherical caps: half the rings of a full sphere, offset to the ends.
    let half = nring.max(2);
    let cap = |i: u32, j: u32, top: bool| -> Vec3 {
        let theta = 0.5 * std::f64::consts::PI * j as f64 / half as f64;
        let phi = std::f64::consts::TAU * i as f64 / nseg as f64;
        let z = radius * theta.sin();
        let r = radius * theta.cos();
        let (z, off) = if top { (z, hz) } else { (-z, -hz) };
        Vec3::new(r * phi.cos(), r * phi.sin(), z + off)
    };
    for top in [true, false] {
        for j in 0..half {
            for i in 0..nseg {
                let (a, b, c, d) = (
                    cap(i, j, top),
                    cap(i + 1, j, top),
                    cap(i + 1, j + 1, top),
                    cap(i, j + 1, top),
                );
                if top {
                    m.push_triangle(a, b, c);
                    m.push_triangle(a, c, d);
                } else {
                    m.push_triangle(a, d, c);
                    m.push_triangle(a, c, b);
                }
            }
        }
    }
    m
}

/// A finite square standing in for an infinite half-space with the given
/// outward `normal`, centred on the shape origin.
pub fn plane(normal: Vec3, extent: f64) -> TriMesh {
    let n = normal.try_normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    // Any vector not parallel to `n` gives a usable tangent basis.
    let seed = if n.z.abs() < 0.9 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = n.cross(seed).normalize();
    let b = n.cross(t);
    let mut m = TriMesh::empty();
    m.push_quad(
        (t * -extent) + (b * -extent),
        (t * extent) + (b * -extent),
        (t * extent) + (b * extent),
        (t * -extent) + (b * extent),
    );
    m
}

/// Load an STL file (binary or ASCII) as a flat-shaded mesh.
///
/// Normals are recomputed from the triangle winding rather than trusted from
/// the file, because a great many exporters write zero or inconsistent normals.
pub fn load_stl(path: impl AsRef<std::path::Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| CameraError::MeshIo {
        path: path.display().to_string(),
        source,
    })?;
    parse_stl(&bytes).map_err(|reason| CameraError::MeshParse {
        path: path.display().to_string(),
        reason,
    })
}

/// Load a mesh by extension. Only `.stl` is supported so far; DAE, OBJ and
/// glTF return [`CameraError::UnsupportedMeshFormat`].
pub fn load_mesh(path: impl AsRef<std::path::Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "stl" => load_stl(path),
        _ => Err(CameraError::UnsupportedMeshFormat {
            path: path.display().to_string(),
        }),
    }
}

/// Parse STL bytes, auto-detecting binary versus ASCII.
pub fn parse_stl(bytes: &[u8]) -> std::result::Result<TriMesh, String> {
    if looks_binary(bytes) {
        parse_binary_stl(bytes)
    } else {
        parse_ascii_stl(bytes)
    }
}

/// Binary STL has an exact size: an 84-byte header plus 50 bytes per triangle.
/// That is a far more reliable test than sniffing for a leading "solid", since
/// binary files are allowed to start with those five bytes too.
fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    bytes.len() == 84 + count * 50
}

fn parse_binary_stl(bytes: &[u8]) -> std::result::Result<TriMesh, String> {
    let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let mut m = TriMesh::empty();
    for t in 0..count {
        let base = 84 + t * 50;
        let f = |off: usize| {
            let s = base + off;
            f32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]]) as f64
        };
        // Bytes 0..12 are the stored face normal, deliberately ignored.
        let a = Vec3::new(f(12), f(16), f(20));
        let b = Vec3::new(f(24), f(28), f(32));
        let c = Vec3::new(f(36), f(40), f(44));
        m.push_triangle(a, b, c);
    }
    Ok(m)
}

fn parse_ascii_stl(bytes: &[u8]) -> std::result::Result<TriMesh, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("not binary STL and not valid UTF-8 ASCII STL: {e}"))?;
    let mut m = TriMesh::empty();
    let mut tri: Vec<Vec3> = Vec::with_capacity(3);
    for (lineno, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        if it.next() != Some("vertex") {
            continue;
        }
        let mut coord = [0.0f64; 3];
        for (i, slot) in coord.iter_mut().enumerate() {
            let tok = it
                .next()
                .ok_or_else(|| format!("line {}: vertex has fewer than 3 coords", lineno + 1))?;
            *slot = tok
                .parse::<f64>()
                .map_err(|e| format!("line {}: bad coord {i} `{tok}`: {e}", lineno + 1))?;
        }
        tri.push(Vec3::new(coord[0], coord[1], coord[2]));
        if tri.len() == 3 {
            m.push_triangle(tri[0], tri[1], tri[2]);
            tri.clear();
        }
    }
    if !tri.is_empty() {
        return Err(format!(
            "trailing partial facet with {} vertices",
            tri.len()
        ));
    }
    if m.is_empty() {
        return Err("no vertices found; file is neither binary nor ASCII STL".to_string());
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuboid_has_twelve_triangles_with_outward_normals() {
        let m = cuboid(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.triangle_count(), 12);
        // Every vertex normal must point away from the (origin-centred) box.
        for v in &m.vertices {
            let dot = v.position[0] * v.normal[0]
                + v.position[1] * v.normal[1]
                + v.position[2] * v.normal[2];
            assert!(dot > 0.0, "inward normal on {v:?}");
        }
    }

    #[test]
    fn sphere_vertices_sit_on_the_sphere_and_normals_point_out() {
        let m = sphere(0.5, &Tessellation::default());
        for v in &m.vertices {
            let r = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!((r - 0.5).abs() < 1e-5, "radius {r}");
            let dot = v.position[0] * v.normal[0]
                + v.position[1] * v.normal[1]
                + v.position[2] * v.normal[2];
            assert!(dot > 0.0);
        }
    }

    #[test]
    fn ascii_and_binary_stl_agree() {
        let ascii = b"solid t\nfacet normal 0 0 0\nouter loop\n\
vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let a = parse_stl(ascii).unwrap();
        assert_eq!(a.triangle_count(), 1);
        assert!((a.vertices[0].normal[2] - 1.0).abs() < 1e-6);

        let mut bin = vec![0u8; 80];
        bin.extend_from_slice(&1u32.to_le_bytes());
        for f in [
            0.0f32, 0.0, 0.0, // stored normal (ignored)
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ] {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        bin.extend_from_slice(&0u16.to_le_bytes());
        let b = parse_stl(&bin).unwrap();
        assert_eq!(b.triangle_count(), 1);
        assert_eq!(a.vertices[1].position, b.vertices[1].position);
    }

    #[test]
    fn capsule_is_radius_from_its_axis_segment() {
        let (r, len) = (0.2, 1.0);
        let m = capsule(r, len, &Tessellation::default());
        for v in &m.vertices {
            let (x, y, z) = (
                v.position[0] as f64,
                v.position[1] as f64,
                v.position[2] as f64,
            );
            // Distance to the segment from (0,0,-len/2) to (0,0,+len/2).
            let zc = z.clamp(-len / 2.0, len / 2.0);
            let d = ((x * x) + (y * y) + (z - zc).powi(2)).sqrt();
            assert!((d - r).abs() < 1e-6, "capsule vertex at distance {d}");
        }
    }
}
