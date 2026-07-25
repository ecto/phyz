//! MJCF XML parser implementation.

use crate::attrs::Attrs;
use crate::defaults::{DefaultsManager, ROOT_CLASS};
use crate::inertia::{self, MassProps, Shape};
use crate::{MjcfError, Result};
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Actuator, Geometry, Joint, JointType, Model, ModelBuilder};
use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// An MJCF feature present in the file that this parser does not implement.
///
/// Collected rather than ignored: a model that silently loses its tendons is
/// far more damaging than one that refuses to load, and callers building RL
/// environments need to know before they train on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFeature {
    /// The XML tag that triggered it.
    pub tag: String,
    /// What the omission means for the resulting model.
    pub detail: String,
}

/// Parsed body element from MJCF.
#[derive(Debug, Clone)]
struct BodyElement {
    name: String,
    pos: Vec3,
    quat: [f64; 4],
    parent_idx: Option<usize>,
    inertial: Option<SpatialInertia>,
    joints: Vec<JointElement>,
    geoms: Vec<GeomElement>,
    /// Class inherited by descendants, from `childclass`.
    childclass: Option<String>,
}

/// Parsed joint element.
#[derive(Debug, Clone)]
struct JointElement {
    name: String,
    joint_type: JointType,
    pos: Vec3,
    axis: Vec3,
    range: Option<[f64; 2]>,
    limited: Option<bool>,
    damping: f64,
    armature: f64,
    stiffness: f64,
    spring_ref: f64,
    friction_loss: f64,
}

/// Parsed geom element.
#[derive(Debug, Clone)]
struct GeomElement {
    #[allow(dead_code)]
    name: String,
    geom_type: String,
    size: Vec<f64>,
    pos: Vec3,
    rot: Mat3,
    density: f64,
    mass: Option<f64>,
    /// Set when the geom is `contype="0" conaffinity="0"` — visual only, so it
    /// contributes inertia but never collides.
    collides: bool,
}

impl GeomElement {
    /// The inertial shape, or `None` for geoms we cannot integrate (meshes,
    /// planes — a plane has infinite extent and contributes no mass).
    fn shape(&self) -> Option<Shape> {
        let s = &self.size;
        match self.geom_type.as_str() {
            "sphere" => Some(Shape::Sphere {
                radius: *s.first()?,
            }),
            "capsule" => Some(Shape::Capsule {
                radius: *s.first()?,
                half_len: *s.get(1)?,
            }),
            "cylinder" => Some(Shape::Cylinder {
                radius: *s.first()?,
                half_height: *s.get(1)?,
            }),
            "box" => Some(Shape::Box {
                half: Vec3::new(*s.first()?, *s.get(1)?, *s.get(2)?),
            }),
            _ => None,
        }
    }

    fn mass_props(&self) -> Option<MassProps> {
        Some(inertia::mass_props(
            self.shape()?,
            self.pos,
            self.rot,
            self.density,
            self.mass,
        ))
    }
}

/// Parsed actuator element, before joint resolution.
#[derive(Debug, Clone)]
struct ActuatorElement {
    name: String,
    joint_name: String,
    gear: f64,
    ctrl_range: Option<[f64; 2]>,
    force_range: Option<[f64; 2]>,
    gain: f64,
    bias_q: f64,
    bias_v: f64,
}

/// A parsed `<sensor>` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorElement {
    /// Sensor name, or a generated one.
    pub name: String,
    /// MJCF sensor tag (`jointpos`, `framepos`, `accelerometer`, …).
    pub kind: String,
    /// The `joint`, `site`, or `body` attribute this sensor targets.
    pub target: String,
}

/// MJCF loader.
pub struct MjcfLoader {
    defaults: DefaultsManager,
    bodies: Vec<BodyElement>,
    actuators: Vec<ActuatorElement>,
    sensors: Vec<SensorElement>,
    unsupported: Vec<UnsupportedFeature>,
    gravity_vec: Vec3,
    timestep: f64,
    angle_in_degrees: bool,
    #[allow(dead_code)]
    coordinate: String,
}

impl MjcfLoader {
    /// Load MJCF from file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let xml_content = fs::read_to_string(path)?;
        Self::from_xml_str(&xml_content)
    }

    /// Load MJCF from XML string.
    pub fn from_xml_str(xml: &str) -> Result<Self> {
        let mut loader = Self {
            defaults: DefaultsManager::new(),
            bodies: Vec::new(),
            actuators: Vec::new(),
            sensors: Vec::new(),
            unsupported: Vec::new(),
            gravity_vec: Vec3::new(0.0, 0.0, -GRAVITY),
            timestep: 0.002,
            angle_in_degrees: false,
            coordinate: "local".to_string(),
        };

        loader.parse_xml(xml)?;
        Ok(loader)
    }

    /// MJCF features present in the file that this parser dropped.
    ///
    /// Check this before trusting a model for anything quantitative.
    pub fn unsupported(&self) -> &[UnsupportedFeature] {
        &self.unsupported
    }

    /// Parsed `<sensor>` entries. Not yet wired into the model — exposed so
    /// callers can see what the file asked for.
    pub fn sensors(&self) -> &[SensorElement] {
        &self.sensors
    }

    fn note_unsupported(&mut self, tag: &str, detail: &str) {
        let f = UnsupportedFeature {
            tag: tag.to_string(),
            detail: detail.to_string(),
        };
        if !self.unsupported.contains(&f) {
            self.unsupported.push(f);
        }
    }

    fn parse_xml(&mut self, xml: &str) -> Result<()> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut in_worldbody = false;
        let mut in_actuator = false;
        let mut in_sensor = false;
        let mut in_asset = false;
        let mut body_stack: Vec<usize> = Vec::new();
        // Stack of `<default>` class names; the outermost is the root class.
        let mut default_stack: Vec<String> = Vec::new();

        loop {
            let event = reader.read_event_into(&mut buf);
            match event {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    let is_empty = matches!(event, Ok(Event::Empty(_)));

                    // Inside <default>, elements *declare* defaults rather than
                    // instantiate anything.
                    if tag == "default" {
                        let a = Attrs::read(e, Default::default())?;
                        let name = a.str_or("class", ROOT_CLASS);
                        let parent = default_stack.last().cloned();
                        self.defaults.declare(&name, parent.as_deref());
                        if !is_empty {
                            default_stack.push(name);
                        }
                        buf.clear();
                        continue;
                    }
                    if let Some(class) = default_stack.last().cloned() {
                        let a = Attrs::read(e, Default::default())?;
                        self.defaults.set(&class, &tag, a.into_own());
                        buf.clear();
                        continue;
                    }

                    match tag.as_str() {
                        "compiler" => self.parse_compiler(e)?,
                        "option" => self.parse_option(e)?,
                        "worldbody" => in_worldbody = true,
                        "actuator" => in_actuator = true,
                        "sensor" => in_sensor = true,
                        "asset" => in_asset = true,
                        "mesh" | "hfield" if in_asset => {
                            self.note_unsupported(
                                &tag,
                                "mesh assets are not loaded; geoms referencing them are dropped \
                                 and contribute no inertia or collision",
                            );
                        }
                        "equality" => self.note_unsupported(
                            "equality",
                            "equality constraints are ignored; closed kinematic loops will \
                             behave as open chains",
                        ),
                        "tendon" => self.note_unsupported(
                            "tendon",
                            "tendons are ignored; coupled joints move independently",
                        ),
                        "body" if in_worldbody => {
                            let idx = self.parse_body(e, body_stack.last().copied())?;
                            body_stack.push(idx);
                            if is_empty {
                                body_stack.pop();
                            }
                        }
                        "joint" | "freejoint" if in_worldbody && !body_stack.is_empty() => {
                            let idx = *body_stack.last().unwrap();
                            self.parse_joint(e, idx, &tag)?;
                        }
                        "inertial" if in_worldbody && !body_stack.is_empty() => {
                            let idx = *body_stack.last().unwrap();
                            self.parse_inertial(e, idx)?;
                        }
                        "geom" if in_worldbody && !body_stack.is_empty() => {
                            let idx = *body_stack.last().unwrap();
                            self.parse_geom(e, idx)?;
                        }
                        "motor" | "position" | "velocity" | "general" if in_actuator => {
                            self.parse_actuator(e, &tag)?;
                        }
                        "site" if in_worldbody => {
                            // Sites carry no dynamics; only sensors reference
                            // them, and those are unresolved anyway.
                        }
                        _ if in_sensor && !tag.is_empty() && tag != "sensor" => {
                            self.parse_sensor(e, &tag)?;
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    match tag.as_str() {
                        "default" => {
                            default_stack.pop();
                        }
                        "body" if in_worldbody && default_stack.is_empty() => {
                            body_stack.pop();
                        }
                        "worldbody" => in_worldbody = false,
                        "actuator" => in_actuator = false,
                        "sensor" => in_sensor = false,
                        "asset" => in_asset = false,
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(MjcfError::XmlError(e)),
                _ => {}
            }
            buf.clear();
        }

        if self.angle_in_degrees {
            for body in &mut self.bodies {
                for joint in &mut body.joints {
                    if let Some(ref mut range) = joint.range {
                        range[0] = range[0].to_radians();
                        range[1] = range[1].to_radians();
                    }
                }
            }
            for act in &mut self.actuators {
                if let Some(ref mut range) = act.ctrl_range {
                    range[0] = range[0].to_radians();
                    range[1] = range[1].to_radians();
                }
            }
        }

        Ok(())
    }

    /// Resolve defaults for `tag` using the element's own `class` attribute if
    /// present, otherwise the enclosing body's `childclass`.
    fn attrs_for(
        &self,
        e: &quick_xml::events::BytesStart,
        tag: &str,
        body_idx: Option<usize>,
    ) -> Result<Attrs> {
        let bare = Attrs::read(e, Default::default())?;
        let explicit = bare.get("class").map(str::to_string);
        let inherited = body_idx.and_then(|i| self.inherited_class(i));
        let class = explicit.or(inherited);
        let defaults = self.defaults.resolve(tag, class.as_deref());
        Attrs::read(e, defaults)
    }

    /// Walk up the body tree for the nearest `childclass`.
    fn inherited_class(&self, body_idx: usize) -> Option<String> {
        let mut cursor = Some(body_idx);
        while let Some(i) = cursor {
            let b = &self.bodies[i];
            if let Some(c) = &b.childclass {
                return Some(c.clone());
            }
            cursor = b.parent_idx;
        }
        None
    }

    fn parse_compiler(&mut self, e: &quick_xml::events::BytesStart) -> Result<()> {
        let a = Attrs::read(e, Default::default())?;
        if let Some(angle) = a.get("angle") {
            self.angle_in_degrees = angle == "degree";
        }
        if let Some(c) = a.get("coordinate") {
            self.coordinate = c.to_string();
        }
        Ok(())
    }

    fn parse_option(&mut self, e: &quick_xml::events::BytesStart) -> Result<()> {
        let a = Attrs::read(e, Default::default())?;
        if let Some(g) = a.vec3("gravity") {
            self.gravity_vec = g;
        }
        self.timestep = a.f64_or("timestep", self.timestep);
        Ok(())
    }

    fn parse_body(
        &mut self,
        e: &quick_xml::events::BytesStart,
        parent_idx: Option<usize>,
    ) -> Result<usize> {
        let a = self.attrs_for(e, "body", parent_idx)?;
        let name = a.str_or("name", &format!("body_{}", self.bodies.len()));
        let pos = a.vec3_or("pos", Vec3::zeros());
        let quat = match a.floats("quat") {
            Some(v) if v.len() == 4 => [v[0], v[1], v[2], v[3]],
            _ => [1.0, 0.0, 0.0, 0.0],
        };
        let childclass = a.get("childclass").map(str::to_string);

        let idx = self.bodies.len();
        self.bodies.push(BodyElement {
            name,
            pos,
            quat,
            parent_idx,
            inertial: None,
            joints: Vec::new(),
            geoms: Vec::new(),
            childclass,
        });
        Ok(idx)
    }

    fn parse_joint(
        &mut self,
        e: &quick_xml::events::BytesStart,
        body_idx: usize,
        tag: &str,
    ) -> Result<()> {
        let a = self.attrs_for(e, tag, Some(body_idx))?;
        let name = a.str_or("name", &format!("joint_{body_idx}"));

        // `<freejoint>` is sugar for `<joint type="free">`.
        let joint_type = if tag == "freejoint" {
            JointType::Free
        } else {
            match a.get("type").unwrap_or("hinge") {
                "hinge" => JointType::Hinge,
                "slide" => JointType::Slide,
                "ball" => JointType::Ball,
                "free" => JointType::Free,
                other => {
                    self.note_unsupported(
                        "joint",
                        &format!("joint type '{other}' is unknown; treated as hinge"),
                    );
                    JointType::Hinge
                }
            }
        };

        let axis = a
            .vec3("axis")
            .map(|v| v.normalize())
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

        // MuJoCo only applies `range` when `limited` is true (or "auto" with a
        // range present). Honouring that avoids inventing limits.
        let limited = a.bool("limited");
        let range = match limited {
            Some(false) => None,
            _ => a.range("range"),
        };

        self.bodies[body_idx].joints.push(JointElement {
            name,
            joint_type,
            pos: a.vec3_or("pos", Vec3::zeros()),
            axis,
            range,
            limited,
            damping: a.f64_or("damping", 0.0),
            armature: a.f64_or("armature", 0.0),
            stiffness: a.f64_or("stiffness", 0.0),
            spring_ref: a.f64_or("springref", 0.0),
            friction_loss: a.f64_or("frictionloss", 0.0),
        });
        Ok(())
    }

    fn parse_inertial(&mut self, e: &quick_xml::events::BytesStart, body_idx: usize) -> Result<()> {
        let a = self.attrs_for(e, "inertial", Some(body_idx))?;
        let pos = a.vec3_or("pos", Vec3::zeros());
        let mass = a.f64_or("mass", 1.0);

        // `fullinertia` is [xx, yy, zz, xy, xz, yz].
        let tensor = match a.floats("fullinertia") {
            Some(v) if v.len() == 6 => {
                Mat3::new(v[0], v[3], v[4], v[3], v[1], v[5], v[4], v[5], v[2])
            }
            _ => {
                let d = a.vec3_or("diaginertia", Vec3::new(0.001, 0.001, 0.001));
                let local = Mat3::from_diagonal(&d);
                match a.floats("quat") {
                    Some(v) if v.len() == 4 => {
                        let r = Quat::new(v[0], v[1], v[2], v[3]).normalize().to_matrix();
                        r * local * r.transpose()
                    }
                    _ => local,
                }
            }
        };

        self.bodies[body_idx].inertial = Some(SpatialInertia::new(mass, pos, tensor));
        Ok(())
    }

    fn parse_geom(&mut self, e: &quick_xml::events::BytesStart, body_idx: usize) -> Result<()> {
        let a = self.attrs_for(e, "geom", Some(body_idx))?;
        let name = a.str_or("name", &format!("geom_{body_idx}"));
        let geom_type = a.str_or("type", "sphere");

        if geom_type == "mesh" || a.get("mesh").is_some() {
            self.note_unsupported(
                "geom",
                "mesh geoms are dropped; the body loses that collision shape and its inertia \
                 contribution",
            );
            return Ok(());
        }

        let mut size = a.floats("size").unwrap_or_else(|| vec![0.05]);
        let mut pos = a.vec3_or("pos", Vec3::zeros());
        let mut rot = match a.floats("quat") {
            Some(v) if v.len() == 4 => Quat::new(v[0], v[1], v[2], v[3]).normalize().to_matrix(),
            _ => Mat3::identity(),
        };

        // `fromto` gives the two endpoints of a capsule/cylinder axis; it
        // overrides pos/quat and supplies the half-length. Cheetah and humanoid
        // are written almost entirely this way.
        if let Some(ft) = a.floats("fromto").filter(|v| v.len() == 6) {
            {
                let p0 = Vec3::new(ft[0], ft[1], ft[2]);
                let p1 = Vec3::new(ft[3], ft[4], ft[5]);
                let axis = p1 - p0;
                let len = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
                pos = (p0 + p1) * 0.5;
                let radius = size.first().copied().unwrap_or(0.05);
                size = vec![radius, len * 0.5];
                if len > 1e-12 {
                    rot = rotation_z_to(axis / len);
                }
            }
        }

        let collides = !(a.f64_or("contype", 1.0) == 0.0 && a.f64_or("conaffinity", 1.0) == 0.0);

        self.bodies[body_idx].geoms.push(GeomElement {
            name,
            geom_type,
            size,
            pos,
            rot,
            density: a.f64_or("density", 1000.0),
            mass: a.f64("mass"),
            collides,
        });
        Ok(())
    }

    fn parse_actuator(&mut self, e: &quick_xml::events::BytesStart, tag: &str) -> Result<()> {
        let a = self.attrs_for(e, tag, None)?;
        let joint_name = a.str_or("joint", "");
        if joint_name.is_empty() {
            self.note_unsupported(
                tag,
                "actuator without a `joint` target (site/tendon actuators are not supported)",
            );
            return Ok(());
        }

        let gear = a
            .floats("gear")
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);
        let kp = a.f64_or("kp", 1.0);
        let kv = a.f64_or("kv", 0.0);

        // MuJoCo's affine law: f = gear * (gain*ctrl + bias_q*q + bias_v*v).
        let (gain, bias_q, bias_v) = match tag {
            "position" => (kp, -kp, -kv),
            "velocity" => (kv.max(1.0), 0.0, -kv.max(1.0)),
            "general" => {
                let gp = a.floats("gainprm").unwrap_or_default();
                let bp = a.floats("biasprm").unwrap_or_default();
                if !gp.is_empty() || !bp.is_empty() {
                    (
                        gp.first().copied().unwrap_or(1.0),
                        bp.get(1).copied().unwrap_or(0.0),
                        bp.get(2).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0, 0.0)
                }
            }
            _ => (1.0, 0.0, 0.0),
        };

        let name = match a.get("name") {
            Some(n) => n.to_string(),
            None => format!("{tag}_{joint_name}"),
        };

        self.actuators.push(ActuatorElement {
            name,
            joint_name,
            gear,
            ctrl_range: a.range("ctrlrange"),
            force_range: a.range("forcerange"),
            gain,
            bias_q,
            bias_v,
        });
        Ok(())
    }

    fn parse_sensor(&mut self, e: &quick_xml::events::BytesStart, tag: &str) -> Result<()> {
        let a = Attrs::read(e, Default::default())?;
        let target = a
            .get("joint")
            .or_else(|| a.get("site"))
            .or_else(|| a.get("body"))
            .or_else(|| a.get("objname"))
            .unwrap_or("")
            .to_string();
        let name = a.str_or("name", &format!("{tag}_{target}"));
        self.sensors.push(SensorElement {
            name,
            kind: tag.to_string(),
            target,
        });
        Ok(())
    }

    /// Build a phyz Model from the parsed MJCF.
    pub fn build_model(&self) -> Model {
        let mut builder = ModelBuilder::new()
            .gravity(self.gravity_vec)
            .dt(self.timestep);

        let mut body_map: HashMap<usize, i32> = HashMap::new();
        let mut next_model_idx: i32 = 0;
        let mut joint_name_map: HashMap<String, usize> = HashMap::new();

        for (body_idx, body) in self.bodies.iter().enumerate() {
            let parent = body
                .parent_idx
                .and_then(|p| body_map.get(&p).copied())
                .unwrap_or(-1);

            let quat =
                Quat::new(body.quat[0], body.quat[1], body.quat[2], body.quat[3]).normalize();
            let parent_to_body = SpatialTransform::new(quat.to_matrix(), body.pos);

            let inertia = self.body_inertia(body);

            if body.joints.is_empty() {
                builder = builder.add_fixed_body(&body.name, parent, parent_to_body, inertia);
                body_map.insert(body_idx, next_model_idx);
                next_model_idx += 1;
            } else {
                // MJCF allows several joints per body; phyz models each as its
                // own single-joint link, with all but the last massless so the
                // body's inertia is not counted more than once.
                let last = body.joints.len() - 1;
                for (joint_idx, joint_elem) in body.joints.iter().enumerate() {
                    let link_name = if body.joints.len() == 1 {
                        body.name.clone()
                    } else {
                        format!("{}_{}", body.name, joint_idx)
                    };

                    // Only the first link carries the body's placement; the
                    // rest are coincident.
                    let base = if joint_idx == 0 {
                        parent_to_body
                    } else {
                        SpatialTransform::identity()
                    };
                    let joint_offset = SpatialTransform::from_translation(joint_elem.pos);
                    let parent_to_joint = base.compose(&joint_offset);

                    let mut joint = match joint_elem.joint_type {
                        JointType::Hinge => Joint::revolute(parent_to_joint),
                        JointType::Slide => Joint::prismatic(parent_to_joint, joint_elem.axis),
                        JointType::Ball => Joint::spherical(parent_to_joint),
                        JointType::Free => Joint::free(parent_to_joint),
                        _ => Joint::revolute(parent_to_joint),
                    };
                    joint.axis = joint_elem.axis;
                    joint.name = joint_elem.name.clone();
                    joint.damping = joint_elem.damping;
                    // MuJoCo: limited="false" disables the range even if given.
                    joint.limits = match joint_elem.limited {
                        Some(false) => None,
                        _ => joint_elem.range,
                    };
                    joint.armature = joint_elem.armature;
                    joint.stiffness = joint_elem.stiffness;
                    joint.spring_ref = joint_elem.spring_ref;
                    joint.friction_loss = joint_elem.friction_loss;

                    let link_inertia = if joint_idx == last {
                        inertia
                    } else {
                        SpatialInertia::new(1e-9, Vec3::zeros(), Mat3::identity() * 1e-9)
                    };

                    joint_name_map.insert(joint_elem.name.clone(), next_model_idx as usize);
                    let link_parent = if joint_idx == 0 {
                        parent
                    } else {
                        next_model_idx - 1
                    };
                    builder = builder.add_body(&link_name, link_parent, joint, link_inertia);

                    if joint_idx == last {
                        body_map.insert(body_idx, next_model_idx);
                    }
                    next_model_idx += 1;
                }
            }
        }

        let mut model = builder.build();

        // Attach every geom to its body, split into collision and visual sets.
        // Geoms are rarely at the body origin — a `fromto` capsule sits at its
        // midpoint — so each carries its own placement.
        for (body_idx, body) in self.bodies.iter().enumerate() {
            let Some(&model_idx) = body_map.get(&body_idx) else {
                continue;
            };
            let b = &mut model.bodies[model_idx as usize];
            for geom in &body.geoms {
                let Some(geometry) = geom_to_geometry(geom) else {
                    continue;
                };
                // `GeomInstance::origin` follows the `parent_to_joint`
                // convention: `rot` is the body → shape transform, so a
                // shape→body rotation goes in transposed.
                let instance = phyz_model::GeomInstance {
                    name: Some(geom.name.clone()),
                    origin: SpatialTransform::new(geom.rot.transpose(), geom.pos),
                    geometry,
                };
                if geom.collides {
                    b.collisions.push(instance);
                } else {
                    b.visuals.push(instance);
                }
            }
            // `geometry` mirrors the first centred collision shape so existing
            // single-shape consumers keep working.
            b.geometry = b
                .collisions
                .iter()
                .find(|g| g.is_centered())
                .or_else(|| b.collisions.first())
                .map(|g| g.geometry.clone());
        }

        for act_elem in &self.actuators {
            if let Some(&joint_idx) = joint_name_map.get(&act_elem.joint_name) {
                model.actuators.push(Actuator {
                    name: act_elem.name.clone(),
                    joint_name: act_elem.joint_name.clone(),
                    joint_idx,
                    gear: act_elem.gear,
                    ctrl_range: act_elem.ctrl_range,
                    gain: act_elem.gain,
                    bias_q: act_elem.bias_q,
                    bias_v: act_elem.bias_v,
                    force_range: act_elem.force_range,
                });
            }
        }

        model
    }

    /// The body's inertia: explicit `<inertial>` if present, else derived from
    /// its geoms, else a small placeholder.
    fn body_inertia(&self, body: &BodyElement) -> SpatialInertia {
        if let Some(i) = body.inertial {
            return i;
        }
        let parts: Vec<MassProps> = body.geoms.iter().filter_map(|g| g.mass_props()).collect();
        if let Some(i) = inertia::combine(&parts) {
            return i;
        }
        SpatialInertia::new(
            1.0,
            Vec3::zeros(),
            Mat3::from_diagonal(&Vec3::new(0.001, 0.001, 0.001)),
        )
    }

    /// Joints declaring a non-zero `armature`.
    pub fn armature_joints(&self) -> Vec<String> {
        self.bodies
            .iter()
            .flat_map(|b| b.joints.iter())
            .filter(|j| j.armature != 0.0)
            .map(|j| j.name.clone())
            .collect()
    }
}

/// A rotation taking the local +Z axis onto `dir` (which must be unit length).
fn rotation_z_to(dir: Vec3) -> Mat3 {
    let z = Vec3::new(0.0, 0.0, 1.0);
    let dot = z.x * dir.x + z.y * dir.y + z.z * dir.z;
    if dot > 1.0 - 1e-12 {
        return Mat3::identity();
    }
    if dot < -1.0 + 1e-12 {
        // 180°: any axis perpendicular to Z works.
        return Mat3::from_diagonal(&Vec3::new(1.0, -1.0, -1.0));
    }
    let axis = Vec3::new(
        z.y * dir.z - z.z * dir.y,
        z.z * dir.x - z.x * dir.z,
        z.x * dir.y - z.y * dir.x,
    );
    let s = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
    Quat::from_axis_angle(axis / s, dot.acos()).to_matrix()
}

/// Convert a parsed GeomElement to a phyz_model Geometry.
///
/// Note the capsule/cylinder unit change: MJCF `size` carries the **half**
/// length, while `Geometry::Capsule::length` is the **full** length (downstream
/// contact code computes `pos.z - length * 0.5 - radius`).
fn geom_to_geometry(geom: &GeomElement) -> Option<Geometry> {
    let s = &geom.size;
    match geom.geom_type.as_str() {
        "sphere" => Some(Geometry::Sphere {
            radius: s.first().copied().unwrap_or(0.05),
        }),
        "capsule" => Some(Geometry::Capsule {
            radius: s.first().copied().unwrap_or(0.05),
            length: s.get(1).copied().unwrap_or(0.05) * 2.0,
        }),
        "cylinder" => Some(Geometry::Cylinder {
            radius: s.first().copied().unwrap_or(0.05),
            height: s.get(1).copied().unwrap_or(0.05) * 2.0,
        }),
        "box" => Some(Geometry::Box {
            half_extents: if s.len() >= 3 {
                Vec3::new(s[0], s[1], s[2])
            } else {
                Vec3::new(0.05, 0.05, 0.05)
            },
        }),
        "plane" => Some(Geometry::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mjcf() {
        let mjcf = r#"
        <mujoco>
            <option gravity="0 0 -9.81" timestep="0.001"/>
            <worldbody>
                <body name="link1" pos="0 0 0">
                    <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert_eq!(model.nbodies(), 1);
        assert_eq!(model.nv, 1);
    }

    #[test]
    fn test_multi_joint_mjcf() {
        let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="link1" pos="0 0 0">
                    <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                    <joint type="hinge" axis="0 0 1"/>
                    <body name="link2" pos="1 0 0">
                        <inertial mass="0.5" diaginertia="0.05 0.05 0.05"/>
                        <joint type="slide" axis="1 0 0"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert_eq!(model.nbodies(), 2);
        assert_eq!(model.nv, 2);
    }

    #[test]
    fn test_geom_parsing() {
        let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="ball" pos="0 0 1">
                    <joint type="free"/>
                    <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                    <geom name="ball_geom" type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert_eq!(model.nbodies(), 1);
        match &model.bodies[0].geometry {
            Some(Geometry::Sphere { radius }) => assert!((*radius - 0.1).abs() < 1e-10),
            other => panic!("expected sphere, got {other:?}"),
        }
    }

    #[test]
    fn test_actuator_parsing() {
        let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="link1"><inertial mass="1" diaginertia=".1 .1 .1"/>
                    <joint name="j1" type="hinge" axis="0 0 1"/>
                </body>
            </worldbody>
            <actuator><motor name="m1" joint="j1" gear="100" ctrlrange="-1 1"/></actuator>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert_eq!(model.actuators.len(), 1);
        assert_eq!(model.actuators[0].ctrl_range, Some([-1.0, 1.0]));
        // Motor: f = gear * ctrl
        assert!((model.actuators[0].force_at(0.5, 1.0, 2.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn position_and_velocity_actuators_use_the_affine_law() {
        let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="b"><inertial mass="1" diaginertia=".1 .1 .1"/>
                    <joint name="j1" type="hinge"/>
                </body>
                <body name="c"><inertial mass="1" diaginertia=".1 .1 .1"/>
                    <joint name="j2" type="hinge"/>
                </body>
            </worldbody>
            <actuator>
                <position name="p" joint="j1" kp="10" kv="2"/>
                <velocity name="v" joint="j2" kv="5"/>
            </actuator>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        let p = &model.actuators[0];
        // kp*(ctrl - q) - kv*v = 10*(1 - 0.5) - 2*0.25 = 4.5
        assert!((p.force_at(1.0, 0.5, 0.25) - 4.5).abs() < 1e-9);
        let v = &model.actuators[1];
        // kv*(ctrl - v) = 5*(2 - 0.5) = 7.5
        assert!((v.force_at(2.0, 0.0, 0.5) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn defaults_classes_are_applied_and_inherited() {
        let mjcf = r#"
        <mujoco>
            <default>
                <joint damping="0.5" limited="true"/>
                <geom density="500" type="capsule"/>
                <default class="stiff">
                    <joint damping="9"/>
                </default>
            </default>
            <worldbody>
                <body name="a">
                    <joint name="ja" range="-1 1"/>
                    <geom size="0.05 0.2"/>
                </body>
                <body name="b" childclass="stiff">
                    <joint name="jb" range="-1 1"/>
                    <geom size="0.05 0.2"/>
                </body>
            </worldbody>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert!(
            (model.joints[0].damping - 0.5).abs() < 1e-12,
            "root default"
        );
        assert!((model.joints[1].damping - 9.0).abs() < 1e-12, "childclass");
        // Root default supplied `type=capsule`, so the geom is a capsule with
        // full length 2 * 0.2.
        match &model.bodies[0].geometry {
            Some(Geometry::Capsule { radius, length }) => {
                assert!((*radius - 0.05).abs() < 1e-12);
                assert!((*length - 0.4).abs() < 1e-12);
            }
            other => panic!("expected capsule, got {other:?}"),
        }
    }

    #[test]
    fn inertia_is_derived_from_geoms_when_inertial_is_absent() {
        let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="a">
                    <joint type="hinge"/>
                    <geom type="sphere" size="0.5" density="1000"/>
                </body>
            </worldbody>
        </mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        let expect = 1000.0 * 4.0 / 3.0 * std::f64::consts::PI * 0.125;
        assert!(
            (model.bodies[0].inertia.mass - expect).abs() < 1e-6,
            "got {}",
            model.bodies[0].inertia.mass
        );
    }

    #[test]
    fn explicit_geom_mass_beats_density() {
        let mjcf = r#"
        <mujoco><worldbody><body name="a">
            <joint type="hinge"/>
            <geom type="box" size="1 1 1" mass="7"/>
        </body></worldbody></mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert!((model.bodies[0].inertia.mass - 7.0).abs() < 1e-12);
    }

    #[test]
    fn fromto_sets_capsule_length_and_centre() {
        let mjcf = r#"
        <mujoco><worldbody><body name="a">
            <joint type="hinge"/>
            <geom type="capsule" fromto="0 0 0  0 0 -1" size="0.05"/>
        </body></worldbody></mujoco>
        "#;
        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        let g = &loader.bodies[0].geoms[0];
        assert!((g.size[1] - 0.5).abs() < 1e-12, "half-length from fromto");
        assert!((g.pos.z + 0.5).abs() < 1e-12, "midpoint");
    }

    #[test]
    fn freejoint_tag_is_a_free_joint() {
        let mjcf = r#"
        <mujoco><worldbody><body name="a">
            <freejoint/>
            <geom type="sphere" size="0.1"/>
        </body></worldbody></mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        assert_eq!(model.nv, 6);
    }

    #[test]
    fn unsupported_features_are_reported_not_swallowed() {
        let mjcf = r#"
        <mujoco>
            <asset><mesh name="m" file="x.stl"/></asset>
            <worldbody><body name="a"><joint type="hinge"/>
                <geom type="sphere" size="0.1"/>
            </body></worldbody>
            <tendon><fixed name="t"/></tendon>
            <equality><connect body1="a" body2="a"/></equality>
        </mujoco>
        "#;
        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        let tags: Vec<&str> = loader
            .unsupported()
            .iter()
            .map(|u| u.tag.as_str())
            .collect();
        assert!(tags.contains(&"mesh"), "{tags:?}");
        assert!(tags.contains(&"tendon"), "{tags:?}");
        assert!(tags.contains(&"equality"), "{tags:?}");
    }

    #[test]
    fn sensors_are_recorded() {
        let mjcf = r#"
        <mujoco>
            <worldbody><body name="a"><joint name="j" type="hinge"/>
                <geom type="sphere" size="0.1"/></body></worldbody>
            <sensor><jointpos name="jp" joint="j"/><jointvel joint="j"/></sensor>
        </mujoco>
        "#;
        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        assert_eq!(loader.sensors().len(), 2);
        assert_eq!(loader.sensors()[0].kind, "jointpos");
        assert_eq!(loader.sensors()[0].target, "j");
    }

    #[test]
    fn visual_only_geoms_do_not_become_collision_shapes() {
        let mjcf = r#"
        <mujoco><worldbody><body name="a"><joint type="hinge"/>
            <geom name="visual" type="box" size="1 1 1" contype="0" conaffinity="0"/>
            <geom name="collider" type="sphere" size="0.2"/>
        </body></worldbody></mujoco>
        "#;
        let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
        match &model.bodies[0].geometry {
            Some(Geometry::Sphere { .. }) => {}
            other => panic!("expected the colliding sphere, got {other:?}"),
        }
    }

    #[test]
    fn test_compiler_degree() {
        let mjcf = r#"
        <mujoco>
            <compiler angle="degree"/>
            <worldbody><body name="link1">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="j1" type="hinge" axis="0 0 1" range="-90 90"/>
            </body></worldbody>
        </mujoco>
        "#;
        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        assert!(loader.angle_in_degrees);
        let model = loader.build_model();
        let range = model.joints[0].limits.unwrap();
        assert!((range[0] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_ant_model() {
        let loader = MjcfLoader::from_file("../../models/ant.xml").unwrap();
        let model = loader.build_model();
        assert_eq!(model.nbodies(), 9);
        assert_eq!(model.nv, 14);
        assert!((model.gravity.z + 9.81).abs() < 1e-10);
        assert_eq!(model.actuators.len(), 8, "ant needs 8 leg motors");
    }
}
