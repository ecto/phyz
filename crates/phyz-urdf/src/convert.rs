//! URDF → phyz `Model` conversion.
//!
//! # Conventions
//!
//! This is where importers usually go wrong, so the mapping is spelled out.
//!
//! **Poses.** A URDF `<origin xyz rpy>` describes a *pose*: `xyz` is the child
//! frame's origin expressed in the parent frame, and `rpy` is a fixed-axis
//! roll-pitch-yaw triple, i.e. `R = Rz(yaw) · Ry(pitch) · Rx(roll)`, which maps
//! child coordinates into parent coordinates.
//!
//! phyz uses Featherstone's Plücker transforms, which are *coordinate*
//! transforms in the opposite direction: `SpatialTransform { rot, pos }` has
//! `rot = R_{child←parent}` and `pos` = the child origin in parent coordinates.
//! So every URDF pose converts as `SpatialTransform::new(R.transpose(), xyz)` —
//! the transpose is the whole trick.
//!
//! **Joint axes.** URDF declares `<axis>` in the joint frame, which coincides
//! with the child link frame at zero configuration. phyz's `Joint::axis` is
//! also expressed in the joint frame, so the axis is copied straight across
//! (normalized). No rotation is applied to it.
//!
//! **Link frames.** A URDF child link frame *is* the joint frame, and a phyz
//! body frame is the joint's successor frame, so link ≡ body with no fix-up.
//!
//! **Inertia.** URDF gives the inertia tensor about the centre of mass, in the
//! frame set by `<inertial><origin>`. phyz's `SpatialInertia` also stores the
//! about-COM tensor but expressed in the *body* frame, hence `R · I · Rᵀ`.

use crate::error::{Result, UrdfError};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Body, GeomInstance, Geometry, Joint, JointType, Model, ModelBuilder};
use std::collections::HashMap;

/// How to attach the URDF root link to the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaseKind {
    /// The root link is rigidly welded to the world (the URDF default, and what
    /// you want for an arm bolted to a table).
    #[default]
    Fixed,
    /// The root link gets a 6-DOF free joint (for legged robots, drones, or any
    /// floating base).
    Floating,
}

/// Import options.
#[derive(Debug, Clone, Default)]
pub struct UrdfOptions {
    /// How the root link connects to the world.
    pub base: BaseKind,
    /// Simulation timestep to stamp on the resulting model.
    pub dt: Option<f64>,
    /// Gravity vector; defaults to the phyz standard `-Z` gravity.
    pub gravity: Option<Vec3>,
}

/// A mesh referenced by the URDF that was not converted into geometry.
///
/// phyz's `Geometry::Mesh` needs actual vertices, and URDF only supplies a file
/// path (often `package://…`, resolvable only through a ROS workspace). Rather
/// than invent a bounding box and pretend it is the robot's collision shape,
/// mesh references are surfaced here for the caller to load.
#[derive(Debug, Clone)]
pub struct MeshRef {
    /// Index of the body the mesh belongs to.
    pub body_idx: usize,
    /// Link name, for error messages.
    pub link: String,
    /// Whether this came from `<visual>` or `<collision>`.
    pub visual: bool,
    /// Raw `filename` attribute, unresolved.
    pub filename: String,
    /// Optional per-axis scale.
    pub scale: Option<Vec3>,
    /// Placement within the link frame.
    pub origin: SpatialTransform,
}

/// A URDF that has been converted into a phyz model, plus the metadata that
/// does not fit in `Model`.
#[derive(Debug, Clone)]
pub struct UrdfModel {
    /// The kinematic model. Bodies are in topological order (parents first) and
    /// carry their URDF link names.
    pub model: Model,
    /// `<robot name="...">`.
    pub robot_name: String,
    /// Meshes that were referenced but not loaded. See [`MeshRef`].
    pub mesh_refs: Vec<MeshRef>,
    /// Things that could not be represented faithfully. Non-empty is not an
    /// error, but callers doing anything safety-relevant should inspect it.
    pub warnings: Vec<String>,
}

impl UrdfModel {
    /// Body index for a URDF link name.
    pub fn body_index(&self, link: &str) -> Option<usize> {
        self.model.body_index(link)
    }

    /// Joint index for a URDF joint name.
    pub fn joint_index(&self, joint: &str) -> Option<usize> {
        self.model.joint_index(joint)
    }
}

/// Convert a parsed `urdf_rs::Robot` into a phyz model.
pub fn robot_to_model(robot: &urdf_rs::Robot, options: &UrdfOptions) -> Result<UrdfModel> {
    if robot.links.is_empty() {
        return Err(UrdfError::NoLinks {
            robot: robot.name.clone(),
        });
    }

    let mut link_by_name: HashMap<&str, &urdf_rs::Link> = HashMap::new();
    for link in &robot.links {
        if link_by_name.insert(&link.name, link).is_some() {
            return Err(UrdfError::DuplicateLink(link.name.clone()));
        }
    }

    // ── Build the link graph, validating as we go ──
    // child link -> the joint that drives it
    let mut driving_joint: HashMap<&str, &urdf_rs::Joint> = HashMap::new();
    let mut children: HashMap<&str, Vec<&urdf_rs::Joint>> = HashMap::new();

    for joint in &robot.joints {
        for link in [&joint.parent.link, &joint.child.link] {
            if !link_by_name.contains_key(link.as_str()) {
                return Err(UrdfError::UnknownLink {
                    joint: joint.name.clone(),
                    link: link.clone(),
                });
            }
        }
        if let Some(prev) = driving_joint.insert(&joint.child.link, joint) {
            return Err(UrdfError::DuplicateChild {
                link: joint.child.link.clone(),
                first: prev.name.clone(),
                second: joint.name.clone(),
            });
        }
        children.entry(&joint.parent.link).or_default().push(joint);
    }

    let roots: Vec<&str> = robot
        .links
        .iter()
        .map(|l| l.name.as_str())
        .filter(|n| !driving_joint.contains_key(n))
        .collect();

    let [root] = roots.as_slice() else {
        return Err(UrdfError::MultipleRoots(roots.len(), roots.join(", ")));
    };
    let root = *root;

    // ── Breadth-first walk so every parent is emitted before its children,
    //    which is what the Featherstone passes assume. ──
    let mut order: Vec<&str> = vec![root];
    let mut head = 0;
    while head < order.len() {
        let current = order[head];
        head += 1;
        if let Some(js) = children.get(current) {
            for j in js {
                order.push(&j.child.link);
            }
        }
    }
    if order.len() != robot.links.len() {
        return Err(UrdfError::Cycle(robot.links.len() - order.len()));
    }

    let mut warnings = Vec::new();
    let mut mesh_refs = Vec::new();
    let mut builder = ModelBuilder::new();
    if let Some(dt) = options.dt {
        builder = builder.dt(dt);
    }
    if let Some(g) = options.gravity {
        builder = builder.gravity(g);
    }

    // Link name -> phyz body index. Note that a link's body index is not its
    // position in `order` once planar joints insert helper bodies.
    let mut body_of: HashMap<&str, i32> = HashMap::new();
    let mut next_body: i32 = 0;

    for link_name in &order {
        let link = link_by_name[link_name];
        let inertia = convert_inertial(&link.inertial);

        let parent_idx;
        let joint;

        if *link_name == root {
            parent_idx = -1;
            joint = match options.base {
                BaseKind::Fixed => Joint::fixed(SpatialTransform::identity()),
                BaseKind::Floating => Joint::free(SpatialTransform::identity()),
            }
            .with_name(&format!("{link_name}_base"));
        } else {
            let uj = driving_joint[link_name];
            let origin = pose_to_transform(&uj.origin);
            parent_idx = body_of[uj.parent.link.as_str()];

            match urdf_joint_type(uj)? {
                Converted::Simple(j) => joint = j,
                Converted::Planar { u, v, normal } => {
                    // phyz has no planar joint type, so expand it into the
                    // exact equivalent: two prismatic DOFs in the plane plus a
                    // revolute DOF about its normal, linked by massless bodies.
                    warnings.push(format!(
                        "joint `{}` is planar; expanded into two prismatic bodies \
                         (`{0}_px`, `{0}_py`) and a revolute joint. Joint limits \
                         on planar joints are not representable and were dropped.",
                        uj.name
                    ));
                    let massless = SpatialInertia::new(0.0, Vec3::zeros(), Mat3::zero());

                    // First helper carries the joint origin; the rest sit on it.
                    let mut jx = Joint::prismatic(origin, u);
                    jx.name = format!("{}_px", uj.name);
                    builder =
                        builder.add_body(&format!("{link_name}_px"), parent_idx, jx, massless);
                    let px_idx = next_body;
                    next_body += 1;

                    let mut jy = Joint::prismatic(SpatialTransform::identity(), v);
                    jy.name = format!("{}_py", uj.name);
                    builder = builder.add_body(&format!("{link_name}_py"), px_idx, jy, massless);
                    let py_idx = next_body;
                    next_body += 1;

                    let mut jr = Joint::revolute(SpatialTransform::identity());
                    jr.axis = normal;
                    jr.name = uj.name.clone();
                    apply_dynamics(&mut jr, uj);

                    builder = push_link(builder, link, py_idx, jr, inertia);
                    body_of.insert(link_name, next_body);
                    collect_geometry(link, next_body as usize, &mut mesh_refs, &mut warnings);
                    next_body += 1;
                    continue;
                }
            }
        }

        builder = push_link(builder, link, parent_idx, joint, inertia);
        body_of.insert(link_name, next_body);
        collect_geometry(link, next_body as usize, &mut mesh_refs, &mut warnings);
        next_body += 1;
    }

    for joint in &robot.joints {
        if let Some(m) = &joint.mimic {
            warnings.push(format!(
                "joint `{}` mimics `{}` (multiplier {:?}, offset {:?}); \
                 phyz has no joint coupling constraint, so it was imported as an \
                 independent DOF",
                joint.name, m.joint, m.multiplier, m.offset
            ));
        }
    }

    let mut model = builder.build();
    // Attach the collision/visual shapes now that body indices are final.
    attach_geometry(&mut model, robot, &link_by_name, &order, &body_of);

    Ok(UrdfModel {
        model,
        robot_name: robot.name.clone(),
        mesh_refs,
        warnings,
    })
}

/// Add a link's body to the builder.
fn push_link(
    builder: ModelBuilder,
    link: &urdf_rs::Link,
    parent: i32,
    joint: Joint,
    inertia: SpatialInertia,
) -> ModelBuilder {
    builder.add_body(&link.name, parent, joint, inertia)
}

/// The result of mapping a URDF joint type onto phyz.
enum Converted {
    Simple(Joint),
    /// Planar joints need three phyz DOFs; carries the in-plane basis.
    Planar {
        u: Vec3,
        v: Vec3,
        normal: Vec3,
    },
}

fn urdf_joint_type(uj: &urdf_rs::Joint) -> Result<Converted> {
    use urdf_rs::JointType as T;

    let origin = pose_to_transform(&uj.origin);
    let axis = Vec3::new(uj.axis.xyz[0], uj.axis.xyz[1], uj.axis.xyz[2]);

    // Fixed and floating joints legitimately carry a zero axis.
    let needs_axis = matches!(
        uj.joint_type,
        T::Revolute | T::Continuous | T::Prismatic | T::Planar
    );
    let axis = if needs_axis {
        let n = axis.norm();
        if !n.is_finite() || n < 1e-12 {
            return Err(UrdfError::DegenerateAxis {
                joint: uj.name.clone(),
            });
        }
        axis / n
    } else {
        axis
    };

    let mut joint = match uj.joint_type {
        // `continuous` is `revolute` without limits.
        T::Revolute | T::Continuous => {
            let mut j = Joint::revolute(origin);
            j.axis = axis;
            if uj.joint_type == T::Revolute {
                j.limits = Some([uj.limit.lower, uj.limit.upper]);
            }
            j
        }
        T::Prismatic => {
            let mut j = Joint::prismatic(origin, axis);
            j.limits = Some([uj.limit.lower, uj.limit.upper]);
            j
        }
        T::Fixed => Joint::fixed(origin),
        T::Floating => Joint::free(origin),
        T::Spherical => Joint::spherical(origin),
        T::Planar => {
            let (u, v) = orthonormal_basis(&axis);
            return Ok(Converted::Planar { u, v, normal: axis });
        }
    };

    joint.name = uj.name.clone();
    apply_dynamics(&mut joint, uj);
    Ok(Converted::Simple(joint))
}

/// Copy `<dynamics>` and the effort/velocity halves of `<limit>`.
fn apply_dynamics(joint: &mut Joint, uj: &urdf_rs::Joint) {
    if let Some(d) = &uj.dynamics {
        joint.damping = d.damping;
        joint.friction = d.friction;
    }
    // `<limit>` is required for revolute/prismatic and defaults to zeros
    // otherwise; a zero effort/velocity cap means "unspecified", not "locked".
    if uj.limit.effort > 0.0 {
        joint.effort_limit = Some(uj.limit.effort);
    }
    if uj.limit.velocity > 0.0 {
        joint.velocity_limit = Some(uj.limit.velocity);
    }
}

/// Two unit vectors completing `n` into a right-handed orthonormal basis.
fn orthonormal_basis(n: &Vec3) -> (Vec3, Vec3) {
    // Pick whichever cardinal axis is least aligned with `n` to stay conditioned.
    let seed = if n.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = n.cross(seed).normalize();
    let v = n.cross(u);
    (u, v)
}

/// URDF pose → Plücker coordinate transform (see the module docs).
pub fn pose_to_transform(pose: &urdf_rs::Pose) -> SpatialTransform {
    let rot = rpy_to_matrix(pose.rpy[0], pose.rpy[1], pose.rpy[2]);
    SpatialTransform::new(
        rot.transpose(),
        Vec3::new(pose.xyz[0], pose.xyz[1], pose.xyz[2]),
    )
}

/// Fixed-axis roll-pitch-yaw → rotation matrix (child → parent).
pub fn rpy_to_matrix(roll: f64, pitch: f64, yaw: f64) -> Mat3 {
    Mat3::rotation_z(yaw)
        .mul_mat(&Mat3::rotation_y(pitch))
        .mul_mat(&Mat3::rotation_x(roll))
}

fn convert_inertial(inertial: &urdf_rs::Inertial) -> SpatialInertia {
    let i = &inertial.inertia;
    let tensor = Mat3::new(
        i.ixx, i.ixy, i.ixz, //
        i.ixy, i.iyy, i.iyz, //
        i.ixz, i.iyz, i.izz,
    );
    // Rotate the about-COM tensor from the inertial frame into the link frame.
    let r = rpy_to_matrix(
        inertial.origin.rpy[0],
        inertial.origin.rpy[1],
        inertial.origin.rpy[2],
    );
    let tensor = r.mul_mat(&tensor).mul_mat(&r.transpose());

    SpatialInertia::new(
        inertial.mass.value,
        Vec3::new(
            inertial.origin.xyz[0],
            inertial.origin.xyz[1],
            inertial.origin.xyz[2],
        ),
        tensor,
    )
}

/// Record the meshes a link refers to, and warn about them.
fn collect_geometry(
    link: &urdf_rs::Link,
    body_idx: usize,
    mesh_refs: &mut Vec<MeshRef>,
    warnings: &mut Vec<String>,
) {
    let visuals = link.visual.iter().map(|v| (true, &v.origin, &v.geometry));
    let collisions = link
        .collision
        .iter()
        .map(|c| (false, &c.origin, &c.geometry));

    for (visual, origin, geom) in visuals.chain(collisions) {
        if let urdf_rs::Geometry::Mesh { filename, scale } = geom {
            mesh_refs.push(MeshRef {
                body_idx,
                link: link.name.clone(),
                visual,
                filename: filename.clone(),
                scale: scale.as_ref().map(|s| Vec3::new(s[0], s[1], s[2])),
                origin: pose_to_transform(origin),
            });
            warnings.push(format!(
                "link `{}` uses mesh `{}` for its {}; phyz needs explicit \
                 vertices, so no geometry was created. See `UrdfModel::mesh_refs`.",
                link.name,
                filename,
                if visual { "visual" } else { "collision" }
            ));
        }
    }
}

/// Convert a URDF primitive into a phyz geometry. Meshes return `None` — they
/// are reported through [`MeshRef`] instead.
fn convert_geometry(geom: &urdf_rs::Geometry) -> Option<Geometry> {
    match geom {
        urdf_rs::Geometry::Box { size } => Some(Geometry::Box {
            // URDF gives full extents; phyz stores half-extents.
            half_extents: Vec3::new(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5),
        }),
        // Both URDF and phyz put cylinders along local Z, with `length`/`height`
        // being the full extent.
        urdf_rs::Geometry::Cylinder { radius, length } => Some(Geometry::Cylinder {
            radius: *radius,
            height: *length,
        }),
        // Capsules are a non-standard extension; `length` is the cylindrical
        // section only in both representations.
        urdf_rs::Geometry::Capsule { radius, length } => Some(Geometry::Capsule {
            radius: *radius,
            length: *length,
        }),
        urdf_rs::Geometry::Sphere { radius } => Some(Geometry::Sphere { radius: *radius }),
        urdf_rs::Geometry::Mesh { .. } => None,
    }
}

/// Populate `Body::visuals` / `Body::collisions` (and the legacy `geometry`
/// field) once body indices are known.
fn attach_geometry(
    model: &mut Model,
    _robot: &urdf_rs::Robot,
    link_by_name: &HashMap<&str, &urdf_rs::Link>,
    order: &[&str],
    body_of: &HashMap<&str, i32>,
) {
    for link_name in order {
        let Some(&idx) = body_of.get(*link_name) else {
            continue;
        };
        let link = link_by_name[*link_name];
        let body: &mut Body = &mut model.bodies[idx as usize];

        for v in &link.visual {
            if let Some(g) = convert_geometry(&v.geometry) {
                body.visuals.push(GeomInstance {
                    name: v.name.clone(),
                    origin: pose_to_transform(&v.origin),
                    geometry: g,
                });
            }
        }
        for c in &link.collision {
            if let Some(g) = convert_geometry(&c.geometry) {
                body.collisions.push(GeomInstance {
                    name: c.name.clone(),
                    origin: pose_to_transform(&c.origin),
                    geometry: g,
                });
            }
        }
        // The contact pipeline reads the single `geometry` field, which has no
        // offset, so only a shape actually centred on the body qualifies.
        body.geometry = body
            .collisions
            .iter()
            .find(|g| g.is_centered())
            .map(|g| g.geometry.clone());
    }
}

/// Number of actuated (non-fixed) DOFs, for quick sanity checks.
pub fn actuated_dofs(model: &Model) -> usize {
    model
        .joints
        .iter()
        .filter(|j| j.joint_type != JointType::Fixed)
        .map(|j| j.ndof())
        .sum()
}
