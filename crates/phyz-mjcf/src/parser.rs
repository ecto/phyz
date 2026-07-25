//! MJCF XML parser implementation.

use crate::assets::{
    HFieldAsset, MaterialAsset, MeshAsset, TextureAsset, parse_hfield, parse_material, parse_mesh,
    parse_texture,
};
use crate::attrs::Attrs;
use crate::defaults::{DefaultsManager, MAIN_CLASS};
use crate::orientation::{AngleConfig, parse_fromto, parse_orientation};
use crate::{MjcfError, Result};
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{
    Actuator, ActuatorType, GeomInstance, Geometry, Joint, JointType, Model, ModelBuilder,
};
use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Elements we parse for reporting but cannot yet simulate.
const RECORD_ONLY_SECTIONS: [&str; 5] = ["equality", "tendon", "sensor", "contact", "keyframe"];

/// Parsed body element from MJCF.
#[derive(Debug, Clone)]
struct BodyElement {
    name: String,
    pos: Vec3,
    quat: Quat,
    parent_idx: Option<usize>,
    inertial: Option<InertialElement>,
    joints: Vec<JointElement>,
    geoms: Vec<GeomElement>,
}

/// Parsed inertial element.
#[derive(Debug, Clone)]
struct InertialElement {
    pos: Vec3,
    mass: f64,
    inertia: Mat3,
}

/// Parsed joint element.
#[derive(Debug, Clone)]
struct JointElement {
    name: String,
    joint_type: JointType,
    pos: Vec3,
    axis: Vec3,
    range: Option<[f64; 2]>,
    damping: f64,
    armature: f64,
    stiffness: f64,
    spring_ref: f64,
    friction_loss: f64,
}

/// Parsed geom element.
#[derive(Debug, Clone)]
struct GeomElement {
    name: String,
    geom_type: String,
    size: Vec<f64>,
    /// Pose relative to the owning body.
    pos: Vec3,
    quat: Quat,
    /// Referenced `<mesh>` asset name, for `type="mesh"`.
    mesh: Option<String>,
    /// Referenced `<hfield>` asset name, for `type="hfield"`.
    hfield: Option<String>,
}

/// A `<site>`: a named massless frame. Recorded for sensors/tendons/tooling.
#[derive(Debug, Clone)]
pub struct SiteElement {
    /// Site name, or a generated one if the document omitted it.
    pub name: String,
    /// Name of the body the site is attached to.
    pub body: String,
    /// Position in the owning body's frame.
    pub pos: Vec3,
    /// Orientation in the owning body's frame.
    pub quat: Quat,
    /// `size` attribute, interpreted per `site_type`.
    pub size: Vec<f64>,
    /// `type` attribute, e.g. `"sphere"` or `"box"`.
    pub site_type: String,
}

/// Parsed actuator element (any of motor/position/velocity/general).
#[derive(Debug, Clone)]
struct ActuatorElement {
    name: String,
    joint_name: String,
    gear: f64,
    ctrl_range: Option<[f64; 2]>,
    force_range: Option<[f64; 2]>,
    actuator_type: ActuatorType,
    gain: f64,
    bias: [f64; 3],
}

/// A model feature that was parsed but is not carried into the built [`Model`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFeature {
    /// The XML element, e.g. `"equality"` or `"hfield"`.
    pub element: String,
    /// What the limitation is.
    pub detail: String,
}

/// MJCF loader.
#[derive(Debug)]
pub struct MjcfLoader {
    defaults: DefaultsManager,
    bodies: Vec<BodyElement>,
    actuators: Vec<ActuatorElement>,
    sites: Vec<SiteElement>,
    meshes: Vec<MeshAsset>,
    textures: Vec<TextureAsset>,
    materials: Vec<MaterialAsset>,
    hfields: Vec<HFieldAsset>,
    unsupported: Vec<UnsupportedFeature>,
    gravity_vec: Vec3,
    timestep: f64,
    angles: AngleConfig,
    #[allow(dead_code)]
    coordinate: String,
    /// Directory of the model file, used to resolve `<include>` and assets.
    model_dir: Option<PathBuf>,
    /// `compiler/meshdir` (or `assetdir`), relative to `model_dir`.
    meshdir: Option<String>,
}

/// Mutable state threaded through the recursive XML walk.
struct ParseCtx {
    in_worldbody: bool,
    /// Stack of (body index, class inherited from `childclass`).
    body_stack: Vec<(usize, String)>,
    /// Nesting stack of `<default>` class names.
    default_stack: Vec<String>,
    in_default: bool,
    in_actuator: bool,
    in_asset: bool,
    /// Name of the record-only section currently open, if any.
    record_section: Option<String>,
    /// Depth of nested `<include>` expansion, to bound recursion.
    include_depth: usize,
}

impl ParseCtx {
    fn new(include_depth: usize) -> Self {
        Self {
            in_worldbody: false,
            body_stack: Vec::new(),
            default_stack: Vec::new(),
            in_default: false,
            in_actuator: false,
            in_asset: false,
            record_section: None,
            include_depth,
        }
    }

    /// Class an element inherits when it names none itself: the nearest enclosing
    /// body's `childclass`, else `main`.
    fn inherited_class(&self) -> &str {
        self.body_stack
            .last()
            .map(|(_, c)| c.as_str())
            .unwrap_or(MAIN_CLASS)
    }
}

const MAX_INCLUDE_DEPTH: usize = 16;

impl MjcfLoader {
    /// Load MJCF from a file path. `<include>` and asset paths resolve relative to it.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let xml_content = fs::read_to_string(path)?;
        let mut loader = Self::empty();
        loader.model_dir = path.parent().map(Path::to_path_buf);
        loader.parse_xml(&xml_content, &mut ParseCtx::new(0))?;
        loader.finish();
        Ok(loader)
    }

    /// Load MJCF from an XML string. `<include>` resolves relative to the process
    /// working directory; use [`MjcfLoader::from_file`] when paths matter.
    pub fn from_xml_str(xml: &str) -> Result<Self> {
        let mut loader = Self::empty();
        loader.parse_xml(xml, &mut ParseCtx::new(0))?;
        loader.finish();
        Ok(loader)
    }

    fn empty() -> Self {
        Self {
            defaults: DefaultsManager::new(),
            bodies: Vec::new(),
            actuators: Vec::new(),
            sites: Vec::new(),
            meshes: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
            hfields: Vec::new(),
            unsupported: Vec::new(),
            gravity_vec: Vec3::new(0.0, 0.0, -GRAVITY),
            timestep: 0.002,
            angles: AngleConfig::default(),
            coordinate: "local".to_string(),
            model_dir: None,
            meshdir: None,
        }
    }

    /// Features that were parsed but are not represented in the built [`Model`].
    pub fn unsupported(&self) -> &[UnsupportedFeature] {
        &self.unsupported
    }

    /// `<site>` elements, in document order.
    pub fn sites(&self) -> &[SiteElement] {
        &self.sites
    }

    /// `<mesh>` assets, in document order.
    pub fn meshes(&self) -> &[MeshAsset] {
        &self.meshes
    }

    /// `<texture>` assets, in document order.
    pub fn textures(&self) -> &[TextureAsset] {
        &self.textures
    }

    /// `<material>` assets, in document order.
    pub fn materials(&self) -> &[MaterialAsset] {
        &self.materials
    }

    /// `<hfield>` assets, in document order.
    pub fn hfields(&self) -> &[HFieldAsset] {
        &self.hfields
    }

    /// The parsed default classes.
    pub fn defaults(&self) -> &DefaultsManager {
        &self.defaults
    }

    /// Whether `compiler/angle="degree"` was set.
    pub fn angle_in_degrees(&self) -> bool {
        self.angles.degrees
    }

    /// Directory that asset `file` attributes resolve against.
    fn asset_dir(&self) -> Option<PathBuf> {
        match (&self.model_dir, &self.meshdir) {
            (Some(dir), Some(sub)) => Some(dir.join(sub)),
            (Some(dir), None) => Some(dir.clone()),
            (None, Some(sub)) => Some(PathBuf::from(sub)),
            (None, None) => None,
        }
    }

    fn note_unsupported(&mut self, element: &str, detail: impl Into<String>) {
        let feature = UnsupportedFeature {
            element: element.to_string(),
            detail: detail.into(),
        };
        if !self.unsupported.contains(&feature) {
            self.unsupported.push(feature);
        }
    }

    /// Post-parse work that needs the whole document.
    fn finish(&mut self) {
        if !self.angles.degrees {
            return;
        }
        for body in &mut self.bodies {
            for joint in &mut body.joints {
                // Slide joints are lengths even when angle="degree".
                if joint.joint_type == JointType::Slide {
                    continue;
                }
                if let Some(range) = joint.range.as_mut() {
                    range[0] = range[0].to_radians();
                    range[1] = range[1].to_radians();
                }
            }
        }
    }

    fn parse_xml(&mut self, xml: &str, ctx: &mut ParseCtx) -> Result<()> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    self.handle_start(&tag, &e, ctx)?;
                }
                Ok(Event::Empty(e)) => {
                    // A self-closing element opens and closes in one event.
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    self.handle_start(&tag, &e, ctx)?;
                    self.handle_end(&tag, ctx);
                }
                Ok(Event::End(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    self.handle_end(&tag, ctx);
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(MjcfError::XmlError(e)),
                _ => {}
            }
            buf.clear();
        }
        Ok(())
    }

    fn handle_start(
        &mut self,
        tag: &str,
        e: &quick_xml::events::BytesStart,
        ctx: &mut ParseCtx,
    ) -> Result<()> {
        // Inside a <default> block, elements declare defaults rather than model content.
        if ctx.in_default && tag != "default" {
            let attrs = Attrs::from_event(tag, e)?;
            let class = ctx
                .default_stack
                .last()
                .cloned()
                .unwrap_or_else(|| MAIN_CLASS.to_string());
            self.defaults.set_element_defaults(&class, tag, attrs.raw());
            return Ok(());
        }

        // Record-only sections are noted once when opened; skip their contents.
        if ctx.record_section.is_some() {
            return Ok(());
        }

        match tag {
            "include" => {
                let attrs = Attrs::from_event(tag, e)?;
                let file = attrs.required("file")?.to_string();
                self.parse_include(&file, ctx)?;
            }
            "compiler" => self.parse_compiler(&Attrs::from_event(tag, e)?)?,
            "option" => self.parse_option(&Attrs::from_event(tag, e)?)?,
            "default" => {
                let attrs = Attrs::from_event(tag, e)?;
                let parent = ctx.default_stack.last().cloned();
                let class = attrs
                    .string("class")
                    .unwrap_or_else(|| parent.clone().unwrap_or_else(|| MAIN_CLASS.to_string()));
                self.defaults
                    .declare_class(&class, parent.as_deref().or(Some(MAIN_CLASS)));
                ctx.default_stack.push(class);
                ctx.in_default = true;
            }
            "asset" => ctx.in_asset = true,
            "mesh" if ctx.in_asset => {
                let attrs = self.resolve(tag, e, ctx)?;
                let asset_dir = self.asset_dir();
                let mesh = parse_mesh(&attrs, asset_dir.as_deref())?;
                if let Some(err) = mesh.load_error.clone() {
                    self.note_unsupported(
                        "mesh",
                        format!("mesh '{}' not loaded: {err}", mesh.name),
                    );
                }
                self.meshes.push(mesh);
            }
            "texture" if ctx.in_asset => {
                let attrs = self.resolve(tag, e, ctx)?;
                self.textures.push(parse_texture(&attrs)?);
                self.note_unsupported("texture", "textures are recorded but phyz has no renderer");
            }
            "material" if ctx.in_asset => {
                let attrs = self.resolve(tag, e, ctx)?;
                self.materials.push(parse_material(&attrs)?);
            }
            "hfield" if ctx.in_asset => {
                let attrs = self.resolve(tag, e, ctx)?;
                let hf = parse_hfield(&attrs)?;
                self.note_unsupported(
                    "hfield",
                    format!(
                        "heightfield '{}' parsed but phyz-collision has no heightfield support",
                        hf.name
                    ),
                );
                self.hfields.push(hf);
            }
            "worldbody" => ctx.in_worldbody = true,
            "actuator" => ctx.in_actuator = true,
            "body" if ctx.in_worldbody => {
                let attrs = self.resolve(tag, e, ctx)?;
                let idx = self.parse_body(&attrs, ctx.body_stack.last().map(|(i, _)| *i))?;
                // childclass propagates down the subtree unless a body overrides it.
                let childclass = match attrs.string("childclass") {
                    Some(c) => {
                        if !self.defaults.has_class(&c) {
                            return Err(MjcfError::UnknownClass {
                                element: "body".to_string(),
                                class: c,
                            });
                        }
                        c
                    }
                    None => ctx.inherited_class().to_string(),
                };
                ctx.body_stack.push((idx, childclass));
            }
            "joint" if ctx.in_worldbody && !ctx.body_stack.is_empty() => {
                let attrs = self.resolve(tag, e, ctx)?;
                let body_idx = ctx.body_stack.last().map(|(i, _)| *i).unwrap_or(0);
                self.parse_joint(&attrs, body_idx)?;
            }
            "freejoint" if ctx.in_worldbody && !ctx.body_stack.is_empty() => {
                let attrs = self.resolve(tag, e, ctx)?;
                let body_idx = ctx.body_stack.last().map(|(i, _)| *i).unwrap_or(0);
                self.parse_freejoint(&attrs, body_idx)?;
            }
            "inertial" if ctx.in_worldbody && !ctx.body_stack.is_empty() => {
                let attrs = self.resolve(tag, e, ctx)?;
                let body_idx = ctx.body_stack.last().map(|(i, _)| *i).unwrap_or(0);
                self.parse_inertial(&attrs, body_idx)?;
            }
            "geom" if ctx.in_worldbody && !ctx.body_stack.is_empty() => {
                let attrs = self.resolve(tag, e, ctx)?;
                let body_idx = ctx.body_stack.last().map(|(i, _)| *i).unwrap_or(0);
                self.parse_geom(&attrs, body_idx)?;
            }
            "site" if ctx.in_worldbody && !ctx.body_stack.is_empty() => {
                let attrs = self.resolve(tag, e, ctx)?;
                let body_idx = ctx.body_stack.last().map(|(i, _)| *i).unwrap_or(0);
                self.parse_site(&attrs, body_idx)?;
            }
            "motor" | "position" | "velocity" | "general" if ctx.in_actuator => {
                let attrs = self.resolve(tag, e, ctx)?;
                self.parse_actuator(tag, &attrs)?;
            }
            other if ctx.in_actuator && !other.is_empty() && other != "actuator" => {
                self.note_unsupported(
                    other,
                    format!("actuator type <{other}> is not supported; it is ignored"),
                );
            }
            other if RECORD_ONLY_SECTIONS.contains(&other) => {
                ctx.record_section = Some(other.to_string());
                self.note_unsupported(other, format!("<{other}> parsed but not simulated"));
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_end(&mut self, tag: &str, ctx: &mut ParseCtx) {
        match tag {
            "body" if ctx.in_worldbody => {
                ctx.body_stack.pop();
            }
            "worldbody" => ctx.in_worldbody = false,
            "actuator" => ctx.in_actuator = false,
            "asset" => ctx.in_asset = false,
            "default" => {
                ctx.default_stack.pop();
                ctx.in_default = !ctx.default_stack.is_empty();
            }
            other => {
                if ctx.record_section.as_deref() == Some(other) {
                    ctx.record_section = None;
                }
            }
        }
    }

    /// Read an element's attributes and fill in whatever its default class provides.
    fn resolve(
        &self,
        tag: &str,
        e: &quick_xml::events::BytesStart,
        ctx: &ParseCtx,
    ) -> Result<Attrs> {
        let mut attrs = Attrs::from_event(tag, e)?;
        let class = match attrs.get("class") {
            Some(c) => c.to_string(),
            None => ctx.inherited_class().to_string(),
        };
        let defaults = self.defaults.resolve(&class, tag)?;
        attrs.merge_defaults(&defaults);
        Ok(attrs)
    }

    fn parse_include(&mut self, file: &str, ctx: &mut ParseCtx) -> Result<()> {
        if ctx.include_depth >= MAX_INCLUDE_DEPTH {
            return Err(MjcfError::InvalidMjcf(format!(
                "<include> nesting exceeded {MAX_INCLUDE_DEPTH} levels at '{file}'; \
                 the includes are probably cyclic"
            )));
        }
        let path = match &self.model_dir {
            Some(dir) => dir.join(file),
            None => PathBuf::from(file),
        };
        let content = fs::read_to_string(&path).map_err(|e| {
            MjcfError::InvalidMjcf(format!("<include file=\"{file}\"> could not be read: {e}"))
        })?;

        // The included file is spliced in at this point, sharing the enclosing
        // context so it can contribute bodies to the current subtree.
        let mut nested = ParseCtx {
            in_worldbody: ctx.in_worldbody,
            body_stack: ctx.body_stack.clone(),
            default_stack: ctx.default_stack.clone(),
            in_default: ctx.in_default,
            in_actuator: ctx.in_actuator,
            in_asset: ctx.in_asset,
            record_section: ctx.record_section.clone(),
            include_depth: ctx.include_depth + 1,
        };
        self.parse_xml(&content, &mut nested)?;
        Ok(())
    }

    fn parse_compiler(&mut self, attrs: &Attrs) -> Result<()> {
        if let Some(angle) = attrs.get("angle") {
            self.angles.degrees = match angle {
                "degree" => true,
                "radian" => false,
                other => {
                    return Err(MjcfError::invalid_attr(
                        "compiler",
                        "angle",
                        other,
                        "expected 'degree' or 'radian'",
                    ));
                }
            };
        }
        if let Some(seq) = attrs.string("eulerseq") {
            if seq.chars().count() != 3 || !seq.chars().all(|c| "xyzXYZ".contains(c)) {
                return Err(MjcfError::invalid_attr(
                    "compiler",
                    "eulerseq",
                    &seq,
                    "expected exactly 3 characters from x/y/z/X/Y/Z",
                ));
            }
            self.angles.eulerseq = seq;
        }
        if let Some(c) = attrs.string("coordinate") {
            if c == "global" {
                return Err(MjcfError::Unsupported(
                    "compiler coordinate=\"global\" (only local coordinates are supported)"
                        .to_string(),
                ));
            }
            self.coordinate = c;
        }
        // meshdir wins over the more general assetdir, matching MuJoCo.
        if let Some(dir) = attrs.string("assetdir") {
            self.meshdir = Some(dir);
        }
        if let Some(dir) = attrs.string("meshdir") {
            self.meshdir = Some(dir);
        }
        Ok(())
    }

    fn parse_option(&mut self, attrs: &Attrs) -> Result<()> {
        if let Some(g) = attrs.vec3("gravity")? {
            self.gravity_vec = g;
        }
        if let Some(dt) = attrs.f64("timestep")? {
            if dt <= 0.0 {
                return Err(MjcfError::invalid_attr(
                    "option",
                    "timestep",
                    &dt.to_string(),
                    "timestep must be positive",
                ));
            }
            self.timestep = dt;
        }
        Ok(())
    }

    fn parse_body(&mut self, attrs: &Attrs, parent_idx: Option<usize>) -> Result<usize> {
        let name = attrs
            .string("name")
            .unwrap_or_else(|| format!("body_{}", self.bodies.len()));
        let pos = attrs.vec3_or("pos", Vec3::zeros())?;
        let quat = parse_orientation(attrs, &self.angles)?.unwrap_or_else(Quat::identity);

        let idx = self.bodies.len();
        self.bodies.push(BodyElement {
            name,
            pos,
            quat,
            parent_idx,
            inertial: None,
            joints: Vec::new(),
            geoms: Vec::new(),
        });
        Ok(idx)
    }

    fn parse_joint(&mut self, attrs: &Attrs, body_idx: usize) -> Result<()> {
        let name = attrs.string("name").unwrap_or_else(|| {
            format!("joint_{}_{}", body_idx, self.bodies[body_idx].joints.len())
        });

        let joint_type = match attrs.get("type").unwrap_or("hinge") {
            "hinge" => JointType::Hinge,
            "slide" => JointType::Slide,
            "ball" => JointType::Ball,
            "free" => JointType::Free,
            other => {
                return Err(MjcfError::invalid_attr(
                    "joint",
                    "type",
                    other,
                    "expected one of hinge/slide/ball/free",
                ));
            }
        };

        let pos = attrs.vec3_or("pos", Vec3::zeros())?;
        let axis = match attrs.vec3("axis")? {
            Some(a) => {
                if a.norm() < 1e-12 {
                    return Err(MjcfError::invalid_attr(
                        "joint",
                        "axis",
                        attrs.get("axis").unwrap_or_default(),
                        "joint axis is the zero vector",
                    ));
                }
                a.normalize()
            }
            None => Vec3::new(0.0, 0.0, 1.0),
        };

        // `limited` gates `range`; MuJoCo's "auto" means "limited iff range is given".
        let range = match attrs.fixed::<2>("range")? {
            Some(r) => {
                if r[0] > r[1] {
                    return Err(MjcfError::invalid_attr(
                        "joint",
                        "range",
                        attrs.get("range").unwrap_or_default(),
                        "lower limit exceeds upper limit",
                    ));
                }
                match attrs.tri_bool("limited")? {
                    Some(Some(false)) => None,
                    _ => Some(r),
                }
            }
            None => None,
        };

        self.bodies[body_idx].joints.push(JointElement {
            name,
            joint_type,
            pos,
            axis,
            range,
            damping: attrs.f64_or("damping", 0.0)?,
            armature: attrs.f64_or("armature", 0.0)?,
            stiffness: attrs.f64_or("stiffness", 0.0)?,
            spring_ref: attrs.f64_or("springref", 0.0)?,
            friction_loss: attrs.f64_or("frictionloss", 0.0)?,
        });
        Ok(())
    }

    /// `<freejoint>` is `<joint type="free">` with no axis/range/damping attributes.
    fn parse_freejoint(&mut self, attrs: &Attrs, body_idx: usize) -> Result<()> {
        let name = attrs
            .string("name")
            .unwrap_or_else(|| format!("freejoint_{body_idx}"));
        self.bodies[body_idx].joints.push(JointElement {
            name,
            joint_type: JointType::Free,
            pos: Vec3::zeros(),
            axis: Vec3::new(0.0, 0.0, 1.0),
            range: None,
            damping: 0.0,
            armature: 0.0,
            stiffness: 0.0,
            spring_ref: 0.0,
            friction_loss: 0.0,
        });
        Ok(())
    }

    fn parse_inertial(&mut self, attrs: &Attrs, body_idx: usize) -> Result<()> {
        let pos = attrs.vec3_or("pos", Vec3::zeros())?;
        let mass = attrs.f64_or("mass", 1.0)?;
        if mass < 0.0 {
            return Err(MjcfError::invalid_attr(
                "inertial",
                "mass",
                &mass.to_string(),
                "mass must be non-negative",
            ));
        }

        let inertia = if let Some(d) = attrs.vec3("diaginertia")? {
            Mat3::from_diagonal(&d)
        } else if let Some(f) = attrs.fixed::<6>("fullinertia")? {
            // MuJoCo order: M(1,1) M(2,2) M(3,3) M(1,2) M(1,3) M(2,3)
            Mat3::new(f[0], f[3], f[4], f[3], f[1], f[5], f[4], f[5], f[2])
        } else {
            Mat3::from_diagonal(&Vec3::new(0.001, 0.001, 0.001))
        };

        self.bodies[body_idx].inertial = Some(InertialElement { pos, mass, inertia });
        Ok(())
    }

    fn parse_geom(&mut self, attrs: &Attrs, body_idx: usize) -> Result<()> {
        let name = attrs
            .string("name")
            .unwrap_or_else(|| format!("geom_{}_{}", body_idx, self.bodies[body_idx].geoms.len()));
        let geom_type = attrs.string("type").unwrap_or_else(|| "sphere".to_string());
        let mut size = attrs.floats("size")?.unwrap_or_default();
        let mut pos = attrs.vec3_or("pos", Vec3::zeros())?;
        let mut quat = parse_orientation(attrs, &self.angles)?.unwrap_or_else(Quat::identity);

        // `fromto` sets position, orientation, and the length component of `size`
        // from two endpoints, overriding pos/quat. Ubiquitous for capsules.
        if let Some(ft) = attrs.fixed::<6>("fromto")? {
            let frame = parse_fromto("geom", &ft)?;
            pos = frame.center;
            quat = frame.quat;
            match size.len() {
                0 => {
                    return Err(MjcfError::InvalidMjcf(format!(
                        "<geom name=\"{name}\"> uses fromto but gives no size (radius)"
                    )));
                }
                1 => size.push(frame.half_length),
                _ => size[1] = frame.half_length,
            }
        }

        if size.is_empty() {
            // Match MuJoCo's small default rather than failing: many models rely on
            // defaults classes for size, which we have already merged in by now.
            size.push(0.05);
        }

        self.bodies[body_idx].geoms.push(GeomElement {
            name,
            geom_type,
            size,
            pos,
            quat,
            mesh: attrs.string("mesh"),
            hfield: attrs.string("hfield"),
        });
        Ok(())
    }

    fn parse_site(&mut self, attrs: &Attrs, body_idx: usize) -> Result<()> {
        let name = attrs
            .string("name")
            .unwrap_or_else(|| format!("site_{}", self.sites.len()));
        self.sites.push(SiteElement {
            name,
            body: self.bodies[body_idx].name.clone(),
            pos: attrs.vec3_or("pos", Vec3::zeros())?,
            quat: parse_orientation(attrs, &self.angles)?.unwrap_or_else(Quat::identity),
            size: attrs.floats("size")?.unwrap_or_else(|| vec![0.005]),
            site_type: attrs.string("type").unwrap_or_else(|| "sphere".to_string()),
        });
        Ok(())
    }

    /// Parse `motor`, `position`, `velocity`, or `general` into the shared affine
    /// actuator model.
    fn parse_actuator(&mut self, tag: &str, attrs: &Attrs) -> Result<()> {
        let joint_name = match attrs.string("joint") {
            Some(j) => j,
            None => {
                // Tendon/site/body transmissions exist but phyz only drives joints.
                let target = ["tendon", "site", "body", "cranksite", "slidersite"]
                    .iter()
                    .find(|k| attrs.has(k))
                    .copied();
                match target {
                    Some(kind) => {
                        self.note_unsupported(
                            tag,
                            format!(
                                "<{tag}> with a '{kind}' transmission is ignored; \
                                 only joint transmissions are supported"
                            ),
                        );
                        return Ok(());
                    }
                    None => {
                        return Err(MjcfError::MissingAttribute {
                            element: tag.to_string(),
                            attribute: "joint".to_string(),
                        });
                    }
                }
            }
        };

        let name = attrs
            .string("name")
            .unwrap_or_else(|| format!("{tag}_{joint_name}"));

        // `gear` is a 6-vector in general; for a joint transmission only the first
        // component is meaningful.
        let gear = match attrs.floats("gear")? {
            Some(g) if !g.is_empty() => g[0],
            Some(_) => 1.0,
            None => 1.0,
        };

        let range_pair = |key: &str| -> Result<Option<[f64; 2]>> {
            match attrs.fixed::<2>(key)? {
                Some(r) if r[0] > r[1] => Err(MjcfError::invalid_attr(
                    tag,
                    key,
                    attrs.get(key).unwrap_or_default(),
                    "lower limit exceeds upper limit",
                )),
                other => Ok(other),
            }
        };
        let mut ctrl_range = range_pair("ctrlrange")?;
        if matches!(attrs.tri_bool("ctrllimited")?, Some(Some(false))) {
            ctrl_range = None;
        }
        let mut force_range = range_pair("forcerange")?;
        if matches!(attrs.tri_bool("forcelimited")?, Some(Some(false))) {
            force_range = None;
        }

        let (actuator_type, gain, bias) = match tag {
            "motor" => (ActuatorType::Motor, 1.0, [0.0; 3]),
            "position" => {
                let kp = attrs.f64_or("kp", 1.0)?;
                // MuJoCo >= 2.3 supports an explicit damping term on position servos.
                let kv = attrs.f64_or("kv", 0.0)?;
                (ActuatorType::Position, kp, [0.0, -kp, -kv])
            }
            "velocity" => {
                let kv = attrs.f64_or("kv", 1.0)?;
                (ActuatorType::Velocity, kv, [0.0, 0.0, -kv])
            }
            "general" => {
                let gainprm = attrs.floats("gainprm")?.unwrap_or_else(|| vec![1.0]);
                let biasprm = attrs.floats("biasprm")?.unwrap_or_default();
                let gain = gainprm.first().copied().unwrap_or(1.0);
                let mut bias = [0.0; 3];
                for (i, slot) in bias.iter_mut().enumerate() {
                    *slot = biasprm.get(i).copied().unwrap_or(0.0);
                }
                if let Some(t) = attrs.get("gaintype")
                    && t != "fixed"
                {
                    self.note_unsupported(
                        "general",
                        format!("gaintype='{t}' is not modelled; treated as 'fixed'"),
                    );
                }
                if let Some(t) = attrs.get("biastype")
                    && t != "none"
                    && t != "affine"
                {
                    self.note_unsupported(
                        "general",
                        format!("biastype='{t}' is not modelled; treated as 'affine'"),
                    );
                }
                if let Some(t) = attrs.get("dyntype")
                    && t != "none"
                {
                    self.note_unsupported(
                        "general",
                        format!(
                            "dyntype='{t}' needs actuator state integration, which phyz \
                                 does not have; the actuator acts as if dyntype='none'"
                        ),
                    );
                }
                (ActuatorType::General, gain, bias)
            }
            other => {
                return Err(MjcfError::Unsupported(format!("actuator type <{other}>")));
            }
        };

        self.actuators.push(ActuatorElement {
            name,
            joint_name,
            gear,
            ctrl_range,
            force_range,
            actuator_type,
            gain,
            bias,
        });
        Ok(())
    }

    /// Build a phyz Model from the parsed MJCF.
    pub fn build_model(&self) -> Model {
        /// Inertia for the intermediate links of a compound joint. Not exactly
        /// zero: a strictly massless link makes the articulated-body inertia
        /// singular if it ever ends up without children.
        const EPS: f64 = 1e-9;
        let massless_link = SpatialInertia::new(
            EPS,
            Vec3::zeros(),
            Mat3::from_diagonal(&Vec3::new(EPS, EPS, EPS)),
        );

        let mut builder = ModelBuilder::new()
            .gravity(self.gravity_vec)
            .dt(self.timestep);

        // Map from body index to model body index
        let mut body_map: HashMap<usize, i32> = HashMap::new();
        let mut next_model_idx: i32 = 0;
        // Map from joint name to model joint index
        let mut joint_name_map: HashMap<String, usize> = HashMap::new();

        // Process bodies in order (assumes parent comes before child in list)
        for (body_idx, body) in self.bodies.iter().enumerate() {
            let parent = body
                .parent_idx
                .and_then(|p| body_map.get(&p).copied())
                .unwrap_or(-1);

            let parent_to_body = SpatialTransform::new(body.quat.to_matrix(), body.pos);

            let inertia = if let Some(ref inertial) = body.inertial {
                SpatialInertia::new(inertial.mass, inertial.pos, inertial.inertia)
            } else {
                // Default: 1kg point mass at origin
                SpatialInertia::new(
                    1.0,
                    Vec3::zeros(),
                    Mat3::from_diagonal(&Vec3::new(0.001, 0.001, 0.001)),
                )
            };

            // If body has no joints, add a fixed joint
            if body.joints.is_empty() {
                builder = builder.add_fixed_body(&body.name, parent, parent_to_body, inertia);
                body_map.insert(body_idx, next_model_idx);
                next_model_idx += 1;
            } else {
                // MJCF allows several joints on one body, which together form a
                // single compound joint between parent and body. Model that as a
                // serial chain of massless links, with the real inertia on the last
                // one so the body's mass is not counted more than once.
                let last_joint = body.joints.len() - 1;
                for (joint_idx, joint_elem) in body.joints.iter().enumerate() {
                    let joint_name = if body.joints.len() == 1 {
                        body.name.clone()
                    } else {
                        format!("{}_{}", body.name, joint_idx)
                    };

                    // Only the first link in the chain carries the body transform;
                    // the rest are zero-length links carrying extra DOFs.
                    let base = if joint_idx == 0 {
                        parent_to_body
                    } else {
                        SpatialTransform::identity()
                    };
                    let joint_offset = SpatialTransform::from_translation(joint_elem.pos);
                    let parent_to_joint = base.compose(&joint_offset);

                    let mut joint = match joint_elem.joint_type {
                        JointType::Hinge | JointType::Revolute => Joint::revolute(parent_to_joint),
                        JointType::Slide | JointType::Prismatic => {
                            Joint::prismatic(parent_to_joint, joint_elem.axis)
                        }
                        JointType::Ball | JointType::Spherical => Joint::spherical(parent_to_joint),
                        JointType::Free => Joint::free(parent_to_joint),
                        JointType::Fixed => Joint::revolute(parent_to_joint),
                    };

                    joint.axis = joint_elem.axis;
                    joint.damping = joint_elem.damping;
                    // `limited="false"` was already folded into `range` at parse time.
                    joint.limits = joint_elem.range;
                    joint.armature = joint_elem.armature;
                    joint.stiffness = joint_elem.stiffness;
                    joint.spring_ref = joint_elem.spring_ref;
                    joint.friction_loss = joint_elem.friction_loss;

                    let model_joint_idx = next_model_idx as usize;
                    joint_name_map.insert(joint_elem.name.clone(), model_joint_idx);

                    // Chain extra joints off the previous one so the DOFs compose.
                    let attach_to = if joint_idx == 0 {
                        parent
                    } else {
                        next_model_idx - 1
                    };
                    let link_inertia = if joint_idx == last_joint {
                        inertia
                    } else {
                        massless_link
                    };
                    builder = builder.add_body(&joint_name, attach_to, joint, link_inertia);

                    body_map.insert(body_idx, next_model_idx);
                    next_model_idx += 1;
                }
            }
        }

        let mut model = builder.build();

        // Attach geometry post-build. Every geom is carried with its own
        // body-relative pose; `geometry` mirrors the first centred shape so the
        // single-shape contact path keeps working.
        for (body_idx, body) in self.bodies.iter().enumerate() {
            let Some(&model_idx) = body_map.get(&body_idx) else {
                continue;
            };
            let target = &mut model.bodies[model_idx as usize];
            for geom in &body.geoms {
                let Some(geometry) = self.geom_to_geometry(geom) else {
                    continue;
                };
                // GeomInstance::origin.rot is the body -> shape coordinate
                // transform, i.e. the transpose of the shape's orientation in
                // the body frame.
                let origin = SpatialTransform::new(geom.quat.to_matrix().transpose(), geom.pos);
                target.collisions.push(GeomInstance {
                    name: Some(geom.name.clone()),
                    origin,
                    geometry,
                });
            }
            target.geometry = target
                .collisions
                .iter()
                .find(|g| g.is_centered())
                .map(|g| g.geometry.clone());
        }

        // Build actuators
        for act_elem in &self.actuators {
            if let Some(&joint_idx) = joint_name_map.get(&act_elem.joint_name) {
                model.actuators.push(Actuator {
                    name: act_elem.name.clone(),
                    joint_name: act_elem.joint_name.clone(),
                    joint_idx,
                    gear: act_elem.gear,
                    ctrl_range: act_elem.ctrl_range,
                    actuator_type: act_elem.actuator_type,
                    gain: act_elem.gain,
                    bias: act_elem.bias,
                    force_range: act_elem.force_range,
                });
            }
        }

        model
    }

    /// Convert a parsed GeomElement to a phyz_model Geometry.
    fn geom_to_geometry(&self, geom: &GeomElement) -> Option<Geometry> {
        match geom.geom_type.as_str() {
            "sphere" => Some(Geometry::Sphere {
                radius: geom.size.first().copied()?,
            }),
            "capsule" => Some(Geometry::Capsule {
                radius: geom.size.first().copied()?,
                // MJCF stores the half-length; phyz's Capsule takes full length.
                length: geom.size.get(1).copied().unwrap_or(0.05) * 2.0,
            }),
            "box" => {
                if geom.size.len() >= 3 {
                    Some(Geometry::Box {
                        half_extents: Vec3::new(geom.size[0], geom.size[1], geom.size[2]),
                    })
                } else {
                    None
                }
            }
            "cylinder" => Some(Geometry::Cylinder {
                radius: geom.size.first().copied()?,
                height: geom.size.get(1).copied().unwrap_or(0.05) * 2.0,
            }),
            "plane" => Some(Geometry::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
            }),
            "ellipsoid" => None,
            "mesh" => {
                let name = geom.mesh.as_deref()?;
                let asset = self.meshes.iter().find(|m| m.name == name)?;
                let data = asset.data.as_ref()?;
                Some(Geometry::Mesh {
                    vertices: data.vertices.clone(),
                    faces: data.faces.clone(),
                })
            }
            "hfield" => {
                let _ = &geom.hfield;
                None
            }
            _ => None,
        }
    }
}
