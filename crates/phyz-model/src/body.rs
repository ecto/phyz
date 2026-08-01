//! Rigid body definition.

use phyz_math::{SpatialInertia, SpatialTransform};

/// A rigid body in the kinematic tree.
#[derive(Debug, Clone)]
pub struct Body {
    /// Name of the body (optional, for debugging).
    pub name: String,
    /// Spatial inertia in body-local frame.
    pub inertia: SpatialInertia,
    /// Index of the parent body (-1 for world/root).
    pub parent: i32,
    /// Index of the joint connecting this body to its parent.
    pub joint_idx: usize,
    /// Primary collision geometry, centred on the body frame (if any).
    ///
    /// This is the shape used by the contact pipeline. For sources that
    /// describe several offset shapes per body (URDF, MJCF) the full set
    /// lives in [`Body::collisions`]; `geometry` mirrors the first one whose
    /// origin is the identity, so existing single-shape consumers keep working.
    pub geometry: Option<Geometry>,
    /// All collision shapes attached to this body, each with its own offset.
    pub collisions: Vec<GeomInstance>,
    /// Visual-only shapes attached to this body (not used for contact).
    pub visuals: Vec<GeomInstance>,
}

impl Body {
    /// Create a body with no attached geometry.
    pub fn new(name: &str, inertia: SpatialInertia, parent: i32, joint_idx: usize) -> Self {
        Self {
            name: name.to_string(),
            inertia,
            parent,
            joint_idx,
            geometry: None,
            collisions: Vec::new(),
            visuals: Vec::new(),
        }
    }
}

/// A geometry placed at an offset within a body frame.
#[derive(Debug, Clone)]
pub struct GeomInstance {
    /// Optional name from the source file.
    pub name: Option<String>,
    /// Placement of the shape within the body frame.
    ///
    /// Follows the same convention as [`crate::Joint::parent_to_joint`]:
    /// `origin.pos` is the shape origin expressed in body coordinates and
    /// `origin.rot` is the coordinate transform *body → shape*, so the shape's
    /// orientation in the body frame is `origin.rot.transpose()`.
    pub origin: SpatialTransform,
    /// The shape itself.
    pub geometry: Geometry,
}

impl GeomInstance {
    /// Create a geom instance at the given offset.
    pub fn new(geometry: Geometry, origin: SpatialTransform) -> Self {
        Self {
            name: None,
            origin,
            geometry,
        }
    }

    /// Create a geom instance centred on the body frame.
    pub fn centered(geometry: Geometry) -> Self {
        Self::new(geometry, SpatialTransform::identity())
    }

    /// True if this shape sits exactly on the body frame.
    pub fn is_centered(&self) -> bool {
        const EPS: f64 = 1e-12;
        self.origin.pos.norm_sq() < EPS
            && (self.origin.rot - phyz_math::Mat3::identity()).norm_sq() < EPS
    }
}

/// Collision geometry types (mirrors `phyz-collision`'s shapes for convenience).
#[derive(Debug, Clone)]
pub enum Geometry {
    /// A ball centred on the body frame origin.
    Sphere {
        /// Ball radius, in metres.
        radius: f64,
    },
    /// A capsule (swept sphere) aligned with the body frame Z axis.
    Capsule {
        /// Radius of the swept sphere, in metres.
        radius: f64,
        /// Distance between the two cap centres, in metres.
        length: f64,
    },
    /// An axis-aligned box centred on the body frame origin.
    Box {
        /// Half-extent along each body frame axis, in metres.
        half_extents: phyz_math::Vec3,
    },
    /// A cylinder aligned with the body frame Z axis.
    Cylinder {
        /// Cylinder radius, in metres.
        radius: f64,
        /// Cylinder height along Z, in metres.
        height: f64,
    },
    /// An arbitrary triangle mesh in body frame coordinates.
    Mesh {
        /// Vertex positions in body frame.
        vertices: Vec<phyz_math::Vec3>,
        /// Triangles, as triples of indices into `vertices`.
        faces: Vec<[usize; 3]>,
    },
    /// A half-space through the body frame origin.
    Plane {
        /// Outward unit normal of the plane, in body frame.
        normal: phyz_math::Vec3,
    },
}

impl Geometry {
    /// Tight axis-aligned bounding box of this shape in body frame, as
    /// half-extents from the body origin.
    ///
    /// `None` for [`Geometry::Plane`], which is unbounded, and for a
    /// [`Geometry::Mesh`] with no vertices.
    ///
    /// Note the half-extents are measured from the *body origin*, not from
    /// the shape's own centroid, so an off-centre mesh yields a box that
    /// covers it symmetrically — correct as a conservative bound, and the
    /// reason [`Self::to_box_approximation`] is documented as conservative.
    pub fn aabb_half_extents(&self) -> Option<phyz_math::Vec3> {
        use phyz_math::Vec3;
        match self {
            Self::Sphere { radius } => Some(Vec3::new(*radius, *radius, *radius)),
            Self::Capsule { radius, length } => {
                Some(Vec3::new(*radius, *radius, radius + 0.5 * length))
            }
            Self::Box { half_extents } => Some(*half_extents),
            Self::Cylinder { radius, height } => Some(Vec3::new(*radius, *radius, 0.5 * height)),
            Self::Mesh { vertices, .. } => {
                if vertices.is_empty() {
                    return None;
                }
                let mut h = Vec3::zeros();
                for v in vertices {
                    h.x = h.x.max(v.x.abs());
                    h.y = h.y.max(v.y.abs());
                    h.z = h.z.max(v.z.abs());
                }
                Some(h)
            }
            Self::Plane { .. } => None,
        }
    }

    /// This shape as a [`Geometry::Box`] covering its bounding volume.
    ///
    /// Lossy and deliberately explicit: a caller converts because some
    /// consumer (the GPU contact pipeline, say) cannot represent the
    /// original shape, and the resulting physics differs from the exact
    /// one. Nothing in this crate performs the conversion implicitly —
    /// silently substituting a box for a mesh is the kind of approximation
    /// that only shows up as a robot standing on air.
    pub fn to_box_approximation(&self) -> Option<Self> {
        self.aabb_half_extents()
            .map(|half_extents| Self::Box { half_extents })
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::Geometry;
    use phyz_math::Vec3;

    /// A mesh's bounding box must cover every vertex. This is what makes the
    /// box a *conservative* stand-in: a GPU consumer that can only do boxes
    /// gets a shape that contacts no later than the mesh would.
    #[test]
    fn mesh_aabb_covers_every_vertex() {
        let vertices = vec![
            Vec3::new(0.1, -0.2, 0.05),
            Vec3::new(-0.3, 0.15, -0.4),
            Vec3::new(0.02, 0.02, 0.02),
        ];
        let g = Geometry::Mesh {
            vertices: vertices.clone(),
            faces: vec![[0, 1, 2]],
        };
        let h = g.aabb_half_extents().expect("mesh has vertices");
        for v in &vertices {
            assert!(
                v.x.abs() <= h.x + 1e-12,
                "vertex {v:?} outside x half-extent"
            );
            assert!(
                v.y.abs() <= h.y + 1e-12,
                "vertex {v:?} outside y half-extent"
            );
            assert!(
                v.z.abs() <= h.z + 1e-12,
                "vertex {v:?} outside z half-extent"
            );
        }
        assert_eq!(h, Vec3::new(0.3, 0.2, 0.4));
    }

    /// A capsule's bounds are the cap sphere at each end, not the segment:
    /// half-length plus radius along Z, radius across. Getting this wrong
    /// shortens a leg by its own foot radius.
    #[test]
    fn capsule_aabb_includes_the_caps() {
        let g = Geometry::Capsule {
            radius: 0.05,
            length: 0.4,
        };
        assert_eq!(g.aabb_half_extents().unwrap(), Vec3::new(0.05, 0.05, 0.25));
    }

    /// A plane is unbounded, so it has no box — and converting one must
    /// return `None` rather than a silently finite slab.
    #[test]
    fn unbounded_and_empty_shapes_have_no_box() {
        let plane = Geometry::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
        };
        assert!(plane.aabb_half_extents().is_none());
        assert!(plane.to_box_approximation().is_none());

        let empty = Geometry::Mesh {
            vertices: vec![],
            faces: vec![],
        };
        assert!(empty.to_box_approximation().is_none());
    }

    /// Converting a box is the identity — a caller that runs everything
    /// through `to_box_approximation` must not perturb shapes that were
    /// already representable.
    #[test]
    fn box_approximation_is_identity_on_a_box() {
        let g = Geometry::Box {
            half_extents: Vec3::new(0.1, 0.2, 0.3),
        };
        let Some(Geometry::Box { half_extents }) = g.to_box_approximation() else {
            panic!("box did not convert to a box");
        };
        assert_eq!(half_extents, Vec3::new(0.1, 0.2, 0.3));
    }
}
