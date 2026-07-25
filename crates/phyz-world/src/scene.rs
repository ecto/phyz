//! Static geometry that sensors can see but that is not part of the kinematic
//! tree: ground planes, walls, props.

use phyz_collision::Geometry as CollisionGeometry;
use phyz_math::{Mat3, Vec3};
use phyz_model::{Body, Geometry as ModelGeometry, Model};

/// A fixed obstacle in the world.
#[derive(Debug, Clone)]
pub struct Obstacle {
    /// Name, for debugging and for identifying sensor hits.
    pub name: String,
    /// Shape.
    pub geometry: CollisionGeometry,
    /// World position of the shape origin.
    pub pos: Vec3,
    /// Rotation mapping shape-local coordinates into world coordinates.
    pub rot: Mat3,
}

impl Obstacle {
    /// Create an axis-aligned obstacle at `pos`.
    pub fn new(name: &str, geometry: CollisionGeometry, pos: Vec3) -> Self {
        Self {
            name: name.to_string(),
            geometry,
            pos,
            rot: Mat3::identity(),
        }
    }

    /// Set the obstacle's orientation.
    pub fn with_rotation(mut self, rot: Mat3) -> Self {
        self.rot = rot;
        self
    }
}

/// The static half of what a sensor can perceive.
///
/// Kinematic bodies come from the `Model` and `State`; everything else that a
/// rangefinder or contact sensor should see lives here.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    /// Fixed obstacles.
    pub obstacles: Vec<Obstacle>,
}

impl Scene {
    /// A scene with no static geometry. Sensors still see the robot's own
    /// bodies.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Add an obstacle (builder style).
    pub fn with_obstacle(mut self, obstacle: Obstacle) -> Self {
        self.obstacles.push(obstacle);
        self
    }

    /// Add a horizontal ground plane at the given height.
    pub fn with_ground(self, height: f64) -> Self {
        self.with_obstacle(Obstacle::new(
            "ground",
            CollisionGeometry::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
            },
            Vec3::new(0.0, 0.0, height),
        ))
    }

    /// Add an obstacle.
    pub fn push(&mut self, obstacle: Obstacle) {
        self.obstacles.push(obstacle);
    }
}

/// What a sensor hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeOwner {
    /// A body in the kinematic tree.
    Body(usize),
    /// A static obstacle, by index into [`Scene::obstacles`].
    Obstacle(usize),
}

/// A collision shape placed in the world.
#[derive(Debug, Clone)]
pub struct PlacedShape {
    /// Who the shape belongs to.
    pub owner: ShapeOwner,
    /// Shape in its local frame.
    pub geometry: CollisionGeometry,
    /// World position of the shape origin.
    pub pos: Vec3,
    /// Rotation mapping shape-local coordinates into world coordinates.
    pub rot: Mat3,
}

/// Translate a model-side geometry into the collision crate's equivalent.
///
/// The two enums are structurally identical; `phyz-model` keeps its own copy to
/// avoid depending on `phyz-collision`.
pub fn to_collision_geometry(g: &ModelGeometry) -> CollisionGeometry {
    match g {
        ModelGeometry::Sphere { radius } => CollisionGeometry::Sphere { radius: *radius },
        ModelGeometry::Capsule { radius, length } => CollisionGeometry::Capsule {
            radius: *radius,
            length: *length,
        },
        ModelGeometry::Box { half_extents } => CollisionGeometry::Box {
            half_extents: *half_extents,
        },
        ModelGeometry::Cylinder { radius, height } => CollisionGeometry::Cylinder {
            radius: *radius,
            height: *height,
        },
        ModelGeometry::Mesh { vertices, faces } => CollisionGeometry::Mesh {
            vertices: vertices.clone(),
            faces: faces.clone(),
        },
        ModelGeometry::Plane { normal } => CollisionGeometry::Plane { normal: *normal },
    }
}

/// A body's collision shapes: everything in `collisions`, or the single
/// `geometry` field if the body predates the multi-shape representation.
fn body_shapes(body: &Body) -> Vec<(CollisionGeometry, phyz_math::SpatialTransform)> {
    if !body.collisions.is_empty() {
        return body
            .collisions
            .iter()
            .map(|c| (to_collision_geometry(&c.geometry), c.origin))
            .collect();
    }
    body.geometry
        .as_ref()
        .map(|g| {
            vec![(
                to_collision_geometry(g),
                phyz_math::SpatialTransform::identity(),
            )]
        })
        .unwrap_or_default()
}

/// Gather every collision shape in the world, in world coordinates.
///
/// `xforms` are the world→body Plücker transforms from forward kinematics.
/// Shapes belonging to `exclude` are skipped, which is how a sensor avoids
/// detecting the body it is mounted on.
pub fn placed_shapes(
    model: &Model,
    xforms: &[phyz_math::SpatialTransform],
    scene: &Scene,
    exclude: Option<usize>,
) -> Vec<PlacedShape> {
    let mut out = Vec::new();

    for (i, body) in model.bodies.iter().enumerate() {
        if exclude == Some(i) || i >= xforms.len() {
            continue;
        }
        // World→body has `rot` mapping world into the body frame, so the
        // body→world rotation is its transpose.
        let body_rot = xforms[i].rot.transpose();
        let body_pos = xforms[i].pos;

        for (geom, origin) in body_shapes(body) {
            // `origin.rot` is likewise a body→shape coordinate transform.
            out.push(PlacedShape {
                owner: ShapeOwner::Body(i),
                geometry: geom,
                pos: body_pos + body_rot.mul_vec(origin.pos),
                rot: body_rot.mul_mat(&origin.rot.transpose()),
            });
        }
    }

    for (i, o) in scene.obstacles.iter().enumerate() {
        out.push(PlacedShape {
            owner: ShapeOwner::Obstacle(i),
            geometry: o.geometry.clone(),
            pos: o.pos,
            rot: o.rot,
        });
    }

    out
}
