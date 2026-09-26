//! Turning a phyz model plus its per-step body transforms into draw calls.
//!
//! A [`RenderScene`] separates what is *static* (mesh geometry, in shape-local
//! coordinates) from what changes every step (one world placement per shape).
//! That split is what makes a per-step update cheap: the vertex buffers are
//! uploaded once, and each step only rewrites a small instance buffer built from
//! [`phyz_world::SensorContext::xforms`].

use crate::mesh::{Tessellation, TriMesh, tessellate};
use phyz_math::{Mat3, SpatialTransform, Vec3};
use phyz_model::{Body, GeomInstance, Model};
use phyz_world::{Scene, SensorContext};

/// One placed, shaded copy of a mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    /// Index into [`RenderScene::meshes`].
    pub mesh: usize,
    /// Rotation taking mesh-local coordinates into world coordinates.
    pub world_from_local: Mat3,
    /// Mesh origin in world coordinates.
    pub position: Vec3,
    /// Linear RGB albedo in `[0, 1]`.
    pub albedo: [f32; 3],
    /// Which body this came from, or `None` for static scene obstacles. Carried
    /// so callers can re-colour or hide the robot's own geometry.
    pub body: Option<usize>,
}

/// Which geometry on a body gets drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeometrySource {
    /// Prefer `<visual>` geometry, falling back to collision geometry (and then
    /// to the body's single `geometry`) for bodies that have no visuals. This is
    /// what a real camera sees.
    #[default]
    VisualThenCollision,
    /// Draw collision geometry only. Useful when checking that what the physics
    /// believes and what the camera sees are the same shape.
    CollisionOnly,
}

/// Knobs for building a scene.
#[derive(Debug, Clone, Default)]
pub struct SceneOptions {
    /// Which geometry to draw.
    pub source: GeometrySource,
    /// Tessellation density for curved primitives.
    pub tessellation: Tessellation,
    /// Body to leave out — typically the one the camera is mounted on, so a
    /// head-mounted camera does not stare at the inside of its own skull.
    pub exclude_body: Option<usize>,
    /// Albedo for model geometry.
    pub body_albedo: [f32; 3],
    /// Albedo for static scene obstacles.
    pub obstacle_albedo: [f32; 3],
}

impl SceneOptions {
    /// Defaults: visual geometry, default tessellation, nothing excluded, and
    /// two mid-grey albedos distinguishable in the RGB image.
    pub fn new() -> Self {
        Self {
            source: GeometrySource::default(),
            tessellation: Tessellation::default(),
            exclude_body: None,
            body_albedo: [0.70, 0.72, 0.75],
            obstacle_albedo: [0.35, 0.45, 0.35],
        }
    }

    /// Leave this body out of the render.
    pub fn exclude_body(mut self, body_idx: usize) -> Self {
        self.exclude_body = Some(body_idx);
        self
    }
}

/// Geometry to rasterize: shared meshes plus their per-step world placements.
#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    /// Mesh geometry in shape-local coordinates.
    pub meshes: Vec<TriMesh>,
    /// One entry per drawn shape.
    pub instances: Vec<Instance>,
}

impl RenderScene {
    /// An empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh, returning its index.
    pub fn add_mesh(&mut self, mesh: TriMesh) -> usize {
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    /// Place a mesh in the world.
    pub fn add_instance(&mut self, instance: Instance) {
        self.instances.push(instance);
    }

    /// Total triangles across all instances, for sizing and sanity checks.
    pub fn triangle_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.meshes[i.mesh].triangle_count())
            .sum()
    }

    /// Build a scene from a model, its world→body transforms, and the static
    /// scene.
    ///
    /// `xforms` comes from [`phyz_world::SensorContext::xforms`], so the poses
    /// are exactly the ones the rest of the sensor suite is reading — no second
    /// forward-kinematics pass and no chance of a one-step skew between what the
    /// camera renders and what the proprioception reports.
    pub fn from_model(
        model: &Model,
        xforms: &[SpatialTransform],
        scene: &Scene,
        options: &SceneOptions,
    ) -> Self {
        let mut out = Self::new();
        out.rebuild_from_model(model, xforms, scene, options);
        out
    }

    /// Same as [`Self::from_model`], reading the transforms out of a
    /// [`SensorContext`].
    pub fn from_context(ctx: &SensorContext<'_>, options: &SceneOptions) -> Self {
        Self::from_model(ctx.model, ctx.xforms(), ctx.scene, options)
    }

    /// Rebuild in place, reusing the allocation.
    pub fn rebuild_from_model(
        &mut self,
        model: &Model,
        xforms: &[SpatialTransform],
        scene: &Scene,
        options: &SceneOptions,
    ) {
        self.meshes.clear();
        self.instances.clear();

        for (i, body) in model.bodies.iter().enumerate() {
            if options.exclude_body == Some(i) || i >= xforms.len() {
                continue;
            }
            // `xforms[i].rot` maps world→body, so its transpose is body→world,
            // and `.pos` is already the body origin in world coordinates.
            let world_from_body = xforms[i].rot.transpose();
            let body_pos = xforms[i].pos;

            for geom in body_geoms(body, options.source) {
                let mesh = self.add_mesh(tessellate(&geom.geometry, &options.tessellation));
                // `geom.origin.rot` is body→shape, same convention one level in.
                self.add_instance(Instance {
                    mesh,
                    world_from_local: world_from_body.mul_mat(&geom.origin.rot.transpose()),
                    position: body_pos + world_from_body.mul_vec(geom.origin.pos),
                    albedo: options.body_albedo,
                    body: Some(i),
                });
            }
        }

        for obstacle in &scene.obstacles {
            let Some(geom) = collision_to_model_geometry(&obstacle.geometry) else {
                continue;
            };
            let mesh = self.add_mesh(tessellate(&geom, &options.tessellation));
            self.add_instance(Instance {
                mesh,
                world_from_local: obstacle.rot,
                position: obstacle.pos,
                albedo: options.obstacle_albedo,
                body: None,
            });
        }
    }
}

/// The geometry instances to draw for one body, honouring the source policy.
fn body_geoms(body: &Body, source: GeometrySource) -> Vec<GeomInstance> {
    let visuals_first = matches!(source, GeometrySource::VisualThenCollision);
    if visuals_first && !body.visuals.is_empty() {
        return body.visuals.clone();
    }
    if !body.collisions.is_empty() {
        return body.collisions.clone();
    }
    body.geometry
        .clone()
        .map(|g| vec![GeomInstance::centered(g)])
        .unwrap_or_default()
}

/// Map a `phyz-collision` shape onto the `phyz-model` shape the tessellator
/// understands. Shapes with no renderable form yield `None`.
fn collision_to_model_geometry(
    geometry: &phyz_collision::Geometry,
) -> Option<phyz_model::Geometry> {
    use phyz_collision::Geometry as C;
    use phyz_model::Geometry as M;
    Some(match geometry {
        C::Sphere { radius } => M::Sphere { radius: *radius },
        C::Box { half_extents } => M::Box {
            half_extents: *half_extents,
        },
        C::Capsule { radius, length } => M::Capsule {
            radius: *radius,
            length: *length,
        },
        C::Cylinder { radius, height } => M::Cylinder {
            radius: *radius,
            height: *height,
        },
        C::Plane { normal } => M::Plane { normal: *normal },
        C::Mesh { vertices, faces } => M::Mesh {
            vertices: vertices.clone(),
            faces: faces.clone(),
        },
        #[allow(unreachable_patterns)]
        _ => return None,
    })
}
