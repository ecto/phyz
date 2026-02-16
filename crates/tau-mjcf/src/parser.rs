//! MJCF XML parser implementation.

use crate::defaults::DefaultsManager;
use crate::{MjcfError, Result};
use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::{Joint, JointType, Model, ModelBuilder};

/// Parsed body element from MJCF.
#[derive(Debug, Clone)]
struct BodyElement {
    name: String,
    pos: Vec3,
    quat: [f64; 4], // [w, x, y, z]
    parent_idx: Option<usize>,
    inertial: Option<InertialElement>,
    joints: Vec<JointElement>,
}

/// Parsed inertial element.
#[derive(Debug, Clone)]
struct InertialElement {
    pos: Vec3,
    mass: f64,
    diaginertia: Vec3,
}

/// Parsed joint element.
#[derive(Debug, Clone)]
struct JointElement {
    #[allow(dead_code)]
    name: String,
    joint_type: JointType,
    pos: Vec3,
    axis: Vec3,
    range: Option<[f64; 2]>,
    damping: f64,
}

/// MJCF loader.
pub struct MjcfLoader {
    #[allow(dead_code)]
    defaults: DefaultsManager,
    bodies: Vec<BodyElement>,
    gravity_vec: Vec3,
    timestep: f64,
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
            gravity_vec: Vec3::new(0.0, 0.0, -GRAVITY),
            timestep: 0.002,
        };

        loader.parse_xml(xml)?;
        Ok(loader)
    }

    fn parse_xml(&mut self, xml: &str) -> Result<()> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut in_worldbody = false;
        let mut body_stack: Vec<usize> = Vec::new(); // Stack of body indices

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    match tag_name.as_str() {
                        "option" => {
                            self.parse_option(&e)?;
                        }
                        "worldbody" => {
                            in_worldbody = true;
                        }
                        "body" if in_worldbody => {
                            let body_idx = self.parse_body(&e, body_stack.last().copied())?;
                            body_stack.push(body_idx);
                        }
                        "joint" if in_worldbody && !body_stack.is_empty() => {
                            let body_idx = *body_stack.last().unwrap();
                            self.parse_joint(&e, body_idx)?;
                        }
                        "inertial" if in_worldbody && !body_stack.is_empty() => {
                            let body_idx = *body_stack.last().unwrap();
                            self.parse_inertial(&e, body_idx)?;
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(e)) => {
                    let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if tag_name == "body" && in_worldbody {
                        body_stack.pop();
                    } else if tag_name == "worldbody" {
                        in_worldbody = false;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(MjcfError::XmlError(e)),
                _ => {}
            }
            buf.clear();
        }

        Ok(())
    }

    fn parse_option(&mut self, e: &quick_xml::events::BytesStart) -> Result<()> {
        for attr in e.attributes() {
            let attr = attr.map_err(|e| MjcfError::InvalidMjcf(e.to_string()))?;
            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            let value = String::from_utf8_lossy(&attr.value).to_string();

            match key.as_str() {
                "gravity" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        self.gravity_vec = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
                "timestep" => {
                    self.timestep = value.parse().unwrap_or(0.002);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_body(
        &mut self,
        e: &quick_xml::events::BytesStart,
        parent_idx: Option<usize>,
    ) -> Result<usize> {
        let mut name = format!("body_{}", self.bodies.len());
        let mut pos = Vec3::zeros();
        let mut quat = [1.0, 0.0, 0.0, 0.0]; // identity quaternion

        for attr in e.attributes() {
            let attr = attr.map_err(|e| MjcfError::InvalidMjcf(e.to_string()))?;
            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            let value = String::from_utf8_lossy(&attr.value).to_string();

            match key.as_str() {
                "name" => name = value.to_string(),
                "pos" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        pos = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
                "quat" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 4 {
                        quat = [parts[0], parts[1], parts[2], parts[3]];
                    }
                }
                _ => {}
            }
        }

        let body = BodyElement {
            name,
            pos,
            quat,
            parent_idx,
            inertial: None,
            joints: Vec::new(),
        };

        let idx = self.bodies.len();
        self.bodies.push(body);
        Ok(idx)
    }

    fn parse_joint(&mut self, e: &quick_xml::events::BytesStart, body_idx: usize) -> Result<()> {
        let mut name = format!("joint_{}", body_idx);
        let mut joint_type = JointType::Hinge; // default
        let mut pos = Vec3::zeros();
        let mut axis = Vec3::new(0.0, 0.0, 1.0);
        let mut range: Option<[f64; 2]> = None;
        let mut damping = 0.0;

        for attr in e.attributes() {
            let attr = attr.map_err(|e| MjcfError::InvalidMjcf(e.to_string()))?;
            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            let value = String::from_utf8_lossy(&attr.value).to_string();

            match key.as_str() {
                "name" => name = value.to_string(),
                "type" => {
                    joint_type = match value.as_str() {
                        "hinge" => JointType::Hinge,
                        "slide" => JointType::Slide,
                        "ball" => JointType::Ball,
                        "free" => JointType::Free,
                        _ => JointType::Hinge,
                    };
                }
                "pos" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        pos = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
                "axis" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        axis = Vec3::new(parts[0], parts[1], parts[2]).normalize();
                    }
                }
                "range" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 2 {
                        range = Some([parts[0], parts[1]]);
                    }
                }
                "damping" => {
                    damping = value.parse().unwrap_or(0.0);
                }
                _ => {}
            }
        }

        let joint = JointElement {
            name,
            joint_type,
            pos,
            axis,
            range,
            damping,
        };

        self.bodies[body_idx].joints.push(joint);
        Ok(())
    }

    fn parse_inertial(&mut self, e: &quick_xml::events::BytesStart, body_idx: usize) -> Result<()> {
        let mut pos = Vec3::zeros();
        let mut mass = 1.0;
        let mut diaginertia = Vec3::new(0.001, 0.001, 0.001);

        for attr in e.attributes() {
            let attr = attr.map_err(|e| MjcfError::InvalidMjcf(e.to_string()))?;
            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            let value = String::from_utf8_lossy(&attr.value).to_string();

            match key.as_str() {
                "pos" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        pos = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
                "mass" => {
                    mass = value.parse().unwrap_or(1.0);
                }
                "diaginertia" => {
                    let parts: Vec<f64> = value
                        .split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0))
                        .collect();
                    if parts.len() == 3 {
                        diaginertia = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
                _ => {}
            }
        }

        self.bodies[body_idx].inertial = Some(InertialElement {
            pos,
            mass,
            diaginertia,
        });

        Ok(())
    }

    /// Build a tau Model from the parsed MJCF.
    pub fn build_model(&self) -> Model {
        let mut builder = ModelBuilder::new()
            .gravity(self.gravity_vec)
            .dt(self.timestep);

        // Map from body index to model body index
        let mut body_map: HashMap<usize, i32> = HashMap::new();
        let mut next_model_idx: i32 = 0;

        // Process bodies in order (assumes parent comes before child in list)
        for (body_idx, body) in self.bodies.iter().enumerate() {
            let parent = body
                .parent_idx
                .and_then(|p| body_map.get(&p).copied())
                .unwrap_or(-1);

            // Create transform from parent to this body
            let quat = tau_math::Quat::new(body.quat[0], body.quat[1], body.quat[2], body.quat[3])
                .normalize();
            let rot = quat.to_matrix();
            let parent_to_body = SpatialTransform::new(rot, body.pos);

            // Get inertia (use default if not specified)
            let inertia = if let Some(ref inertial) = body.inertial {
                SpatialInertia::new(
                    inertial.mass,
                    inertial.pos,
                    Mat3::from_diagonal(&inertial.diaginertia),
                )
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
                // Add body for each joint (MJCF allows multiple joints per body)
                for (joint_idx, joint_elem) in body.joints.iter().enumerate() {
                    let joint_name = if body.joints.len() == 1 {
                        body.name.clone()
                    } else {
                        format!("{}_{}", body.name, joint_idx)
                    };

                    // Create joint transform (joint may have its own pos offset)
                    let joint_offset = SpatialTransform::translation(joint_elem.pos);
                    let parent_to_joint = parent_to_body.compose(&joint_offset);

                    // Create joint based on type
                    let mut joint = match joint_elem.joint_type {
                        JointType::Hinge => Joint::revolute(parent_to_joint),
                        JointType::Slide => Joint::prismatic(parent_to_joint, joint_elem.axis),
                        JointType::Ball => Joint::spherical(parent_to_joint),
                        JointType::Free => Joint::free(parent_to_joint),
                        _ => Joint::revolute(parent_to_joint),
                    };

                    joint.axis = joint_elem.axis;
                    joint.damping = joint_elem.damping;
                    joint.limits = joint_elem.range;

                    builder = builder.add_body(&joint_name, parent, joint, inertia);

                    if joint_idx == 0 {
                        body_map.insert(body_idx, next_model_idx);
                    }
                    next_model_idx += 1;
                }
            }
        }

        builder.build()
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

        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        let model = loader.build_model();

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

        let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
        let model = loader.build_model();

        assert_eq!(model.nbodies(), 2);
        assert_eq!(model.nv, 2); // One revolute + one prismatic
    }
}
