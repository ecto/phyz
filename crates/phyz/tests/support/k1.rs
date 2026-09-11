//! The Booster K1 as ipse's `StandingRig` builds it, plus a scripted PD rig.
//!
//! Shared by `examples/contact_speed.rs` and `tests/contact_speed_exact.rs`
//! (`#[path]`-included; not a test target itself). Everything returns `None`
//! when the vendor assets are absent, so tests can skip instead of failing.
//! `K1_DIR` overrides the asset directory.

#![allow(dead_code)]

use phyz::phyz_math::{Mat3, SpatialTransform, SpatialTransformExt, Vec3};
use phyz::phyz_model::{GeomInstance, Geometry, JointType, Model, State};
use phyz::phyz_rigid::forward_kinematics;
use std::path::{Path, PathBuf};

/// Vendor joint order, gains, effort limits and pose: ipse's `StandingRig`
/// (vendor gains) as written to `rig.json` by the audit's `contact_audit_k1`.
pub const JOINTS: [&str; 22] = [
    "AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
    "Left_Elbow_Yaw", "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
    "Right_Elbow_Yaw", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
];
pub const KP: [f64; 22] = [
    1.6076564945148037, 4.96794603104, 14.219637448679467, 12.858821292846736, 0.5937363343523681,
    1.7562086058847597, 14.216133999885427, 12.854414564672165, 0.594081067660144, 1.75529654190604,
    604.180261289572, 500.3593451001352, 54.445200651623594, 278.2593675021229, 92.548975055296,
    90.9807259388928, 604.1759041201899, 500.3530313430554, 54.4445516726996, 278.25672367342963,
    92.548975055296, 90.9807259388928,
];
pub const EFFORT: [f64; 22] = [
    6.0, 6.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 30.0, 35.0, 20.0, 40.0, 20.0, 20.0,
    30.0, 35.0, 20.0, 40.0, 20.0, 20.0,
];
pub const Q0: [f64; 22] = [
    0.0, 0.0, 0.0, -1.4, 0.25, 0.0, 0.0, 1.4, 0.25, 0.0, -0.05, 0.0, 0.0, 0.1, -0.05, 0.0, -0.05,
    0.0, 0.0, 0.1, -0.05, 0.0,
];
pub const TRUNK_Z: f64 = 0.5532540584455157;
pub const FEET: [&str; 2] = ["left_foot_link", "right_foot_link"];

/// Leg kd = kp / 20, arm/head kd = kp / 10 (the rig's vendor gains).
pub fn kd(j: usize) -> f64 {
    if j >= 10 { KP[j] * 0.05 } else { KP[j] * 0.1 }
}

pub fn k1_dir() -> PathBuf {
    std::env::var("K1_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/cam/Developer/booster_assets/robots/K1"))
}

/// Axis-aligned bounds of a binary STL, as (half extents, centre).
pub fn stl_aabb(path: &Path) -> Option<(Vec3, Vec3)> {
    let b = std::fs::read(path).ok()?;
    let n = u32::from_le_bytes(b.get(80..84)?.try_into().ok()?) as usize;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for t in 0..n {
        let base = 84 + 50 * t + 12;
        for v in 0..3 {
            for k in 0..3 {
                let o = base + 12 * v + 4 * k;
                let x = f32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as f64;
                lo[k] = lo[k].min(x);
                hi[k] = hi[k].max(x);
            }
        }
    }
    Some((
        Vec3::new(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 0.5,
        Vec3::new(hi[0] + lo[0], hi[1] + lo[1], hi[2] + lo[2]) * 0.5,
    ))
}

/// The vendor MJCF as phyz-mjcf loads it (its own collision set).
pub fn mjcf_k1(dt: f64) -> Option<Model> {
    let path = k1_dir().join("K1_22dof.xml");
    if !path.is_file() {
        return None;
    }
    let mut m = phyz_mjcf::MjcfLoader::from_file(path).ok()?.build_model();
    m.dt = dt;
    Some(m)
}

/// The K1 as ipse's `StandingRig::build` makes it: URDF (floating joint inside
/// the file, so `BaseKind::Fixed`), vendor armature, vendor foot pad boxes,
/// and a fitted box for every link whose only collision is a mesh.
pub fn urdf_k1(dt: f64) -> Option<Model> {
    let dir = k1_dir();
    let path = dir.join("K1_22dof_floating.urdf");
    if !path.is_file() {
        return None;
    }
    let opts = phyz_urdf::UrdfOptions { dt: Some(dt), ..Default::default() };
    let u = phyz_urdf::load_file(path, &opts).ok()?;
    let mut model = u.model;
    let mj = mjcf_k1(dt)?;
    for j in &mj.joints {
        if j.armature != 0.0 {
            if let Some(k) = model.joint_index(&j.name) {
                model.joints[k].armature = j.armature;
            }
        }
    }
    for link in FEET {
        let idx = model.body_index(link)?;
        let mb = &mj.bodies[mj.body_index(link)?];
        let (half, off) = mb.collisions.iter().chain(mb.visuals.iter()).find_map(|g| match g.geometry {
            Geometry::Box { half_extents } => Some((half_extents, g.origin.pos)),
            _ => None,
        })?;
        model.bodies[idx].geometry = None;
        model.bodies[idx].collisions = vec![GeomInstance::new(
            Geometry::Box { half_extents: half },
            SpatialTransform::new(Mat3::identity(), off),
        )];
    }
    for r in u.mesh_refs.iter().filter(|r| !r.visual) {
        if FEET.contains(&r.link.as_str()) {
            continue;
        }
        let Some(idx) = model.body_index(&r.link) else { continue };
        if !model.bodies[idx].collisions.is_empty() || model.bodies[idx].geometry.is_some() {
            continue;
        }
        let Some((half, centre)) = stl_aabb(&dir.join(&r.filename)) else { continue };
        let origin = SpatialTransform::new(r.origin.rot, r.origin.body_to_world_point(centre));
        model.bodies[idx].collisions.push(GeomInstance::new(Geometry::Box { half_extents: half }, origin));
    }
    Some(model)
}

/// State indices of the vendor joints.
pub struct K1Map {
    pub q: [usize; 22],
    pub v: [usize; 22],
}

pub fn k1_map(model: &Model) -> K1Map {
    let mut q = [0; 22];
    let mut v = [0; 22];
    for (k, n) in JOINTS.iter().enumerate() {
        let j = model.joint_index(n).unwrap_or_else(|| panic!("joint {n}"));
        q[k] = model.q_offsets[j];
        v[k] = model.v_offsets[j];
    }
    K1Map { q, v }
}

/// The rig's standing pose, trunk at `TRUNK_Z` whatever offset the file put
/// on the trunk body.
pub fn k1_state(model: &Model, map: &K1Map) -> State {
    let mut s = model.default_state();
    let free = model.joints.iter().position(|j| j.joint_type == JointType::Free).expect("free base");
    let b = model.q_offsets[free];
    for k in 0..22 {
        s.q[map.q[k]] = Q0[k];
    }
    let trunk = model.body_index("Trunk").unwrap_or(0);
    let (x, _) = forward_kinematics(model, &s);
    s.q[b + 5] += TRUNK_Z - x[trunk].pos.z;
    s
}

/// The scripted PD target for vendor joint `k` at time `t`:
/// `stance` holds the pose, `single` lifts the left leg from 0.3 s,
/// `step` alternates lifts at 1.25 Hz from 0.3 s.
pub fn k1_target(script: &str, t: f64, k: usize) -> f64 {
    let base = Q0[k];
    let smooth = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    // (hip pitch, knee, ankle pitch) lift shape for leg `leg` (0 left, 1 right).
    let lift = |leg: usize, s: f64| -> f64 {
        match k.checked_sub(10 + 6 * leg) {
            Some(0) => -0.3 * s,
            Some(3) => 0.6 * s,
            Some(4) => -0.3 * s,
            _ => 0.0,
        }
    };
    match script {
        "single" => base + lift(0, smooth((t - 0.3) / 0.2)),
        "step" => {
            if t < 0.3 {
                return base;
            }
            let ph = ((t - 0.3) / 0.8).fract();
            let (leg, x) = if ph < 0.5 { (0, ph / 0.5) } else { (1, (ph - 0.5) / 0.5) };
            base + lift(leg, (std::f64::consts::PI * x).sin())
        }
        _ => base,
    }
}

/// Write the scripted PD torques into `state.ctrl` (everything else zero).
pub fn k1_ctrl(map: &K1Map, script: &str, state: &mut State) {
    for i in 0..state.ctrl.len() {
        state.ctrl[i] = 0.0;
    }
    for k in 0..22 {
        let e = k1_target(script, state.time, k) - state.q[map.q[k]];
        let v = map.v[k];
        state.ctrl[v] = (KP[k] * e - kd(k) * state.v[v]).clamp(-EFFORT[k], EFFORT[k]);
    }
}
