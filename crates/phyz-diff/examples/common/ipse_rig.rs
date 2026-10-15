//! ipse's K1 standing rig, as written by ipse-sim's `k1_rig_dump` example:
//! the plant parameters a simulator outside ipse needs to match it.

#![allow(dead_code)]

use phyz_contact::ContactMaterial;
use phyz_diff::contact_step::{ContactPoint, PdLaw};
use phyz_math::Vec3;
use phyz_model::Model;

pub struct Rig {
    pub dt: f64,
    pub shove: f64,
    pub shove_dir: String,
    /// `(name, armature)` per joint, in ipse's order.
    pub joints: Vec<(String, f64)>,
    pub kp: Vec<f64>,
    pub kd: Vec<f64>,
    pub effort: Vec<f64>,
    pub q_target: Vec<f64>,
    pub q0: Vec<f64>,
    pub v0: Vec<f64>,
    /// `(body name, half extents, offset)` per foot box.
    pub feet: Vec<(String, Vec3, Vec3)>,
    pub friction: f64,
    pub margin: f64,
    /// ipse's trajectory: `(t, q)` every 10 ms.
    pub traj: Vec<(f64, Vec<f64>)>,
}

impl Rig {
    pub fn read(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("rig file");
        let mut r = Rig {
            dt: 0.001,
            shove: 0.0,
            shove_dir: String::new(),
            joints: Vec::new(),
            kp: Vec::new(),
            kd: Vec::new(),
            effort: Vec::new(),
            q_target: Vec::new(),
            q0: Vec::new(),
            v0: Vec::new(),
            feet: Vec::new(),
            friction: 0.5,
            margin: 0.001,
            traj: Vec::new(),
        };
        let nums = |w: &[&str]| -> Vec<f64> { w.iter().map(|v| v.parse().unwrap()).collect() };
        // `key value` pairs in a struct's Debug text
        let field = |line: &str, key: &str| -> f64 {
            let at = line
                .find(&format!("{key}: "))
                .unwrap_or_else(|| panic!("{key} in {line}"))
                + key.len()
                + 2;
            line[at..]
                .split([',', ' ', '}'])
                .next()
                .unwrap()
                .parse()
                .unwrap()
        };
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w.first().copied() {
                Some("meta") => {
                    r.dt = w[2].parse().unwrap();
                    r.shove = w[10].parse().unwrap();
                    r.shove_dir = w[11].to_string();
                }
                Some("joint") => r.joints.push((w[2].to_string(), w[10].parse().unwrap())),
                Some("dof") => {
                    r.kp.push(w[3].parse().unwrap());
                    r.kd.push(w[5].parse().unwrap());
                    r.effort.push(w[7].parse().unwrap());
                }
                Some("q_target") => r.q_target = nums(&w[1..]),
                Some("q0") => r.q0 = nums(&w[1..]),
                Some("v0") => r.v0 = nums(&w[1..]),
                Some("foot") => {
                    let f = |i: usize| w[i].parse::<f64>().unwrap();
                    r.feet.push((
                        w[2].to_string(),
                        Vec3::new(f(4), f(5), f(6)),
                        Vec3::new(f(8), f(9), f(10)),
                    ));
                }
                Some("material") => {
                    r.friction = field(line, "friction");
                    r.margin = field(line, "margin");
                }
                Some("traj") => {
                    let v = nums(&w[1..]);
                    r.traj.push((v[0], v[1..].to_vec()));
                }
                _ => {}
            }
        }
        r
    }

    /// The K1 as ipse builds it: the floating URDF, vendor armature, no joint
    /// damping or actuators (the PD is traced by `ContactStep::build_pd`).
    pub fn model(&self) -> Model {
        let root = std::env::var_os("IPSE_K1_ASSETS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join("Developer/booster_assets/robots/K1")
            });
        let opts = phyz_urdf::UrdfOptions {
            base: phyz_urdf::BaseKind::Fixed,
            dt: Some(self.dt),
            gravity: None,
        };
        let mut model = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts)
            .expect("K1 URDF")
            .model;
        model.actuators.clear();
        assert_eq!(
            model.joints.len(),
            self.joints.len(),
            "joint count differs from ipse's"
        );
        for (joint, (name, armature)) in model.joints.iter_mut().zip(&self.joints) {
            assert_eq!(&joint.name, name, "joint order differs from ipse's");
            joint.armature = *armature;
            joint.damping = 0.0;
            joint.friction_loss = 0.0;
        }
        model
    }

    /// The driven DOFs (kp > 0) and their law.
    pub fn pd(&self) -> PdLaw {
        let dofs: Vec<usize> = (0..self.kp.len()).filter(|&v| self.kp[v] > 0.0).collect();
        PdLaw {
            kp: dofs.iter().map(|&v| self.kp[v]).collect(),
            kd: dofs.iter().map(|&v| self.kd[v]).collect(),
            effort: dofs.iter().map(|&v| self.effort[v]).collect(),
            dofs,
        }
    }

    /// The four bottom corners of each foot box, in the foot link's frame.
    pub fn points(&self) -> Vec<ContactPoint> {
        let model = self.model();
        let mut pts = Vec::new();
        for (name, half, off) in &self.feet {
            let body = model
                .bodies
                .iter()
                .position(|b| &b.name == name)
                .expect("foot body");
            for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                pts.push(ContactPoint {
                    body,
                    local: Vec3::new(off.x + sx * half.x, off.y + sy * half.y, off.z - half.z),
                });
            }
        }
        pts
    }

    pub fn material(&self) -> ContactMaterial {
        ContactMaterial {
            friction: self.friction,
            margin: self.margin,
            ..ContactMaterial::default()
        }
    }

    /// ipse's shove force at time `t` (trunk frame, N), over 20 ms from 2 s.
    pub fn shove_force(&self, t: f64) -> Option<[f64; 3]> {
        let (start, dur) = (2.0, 0.02);
        let dt = self.dt;
        (t >= start - 0.5 * dt && t < start + dur - 0.5 * dt).then(|| {
            let f = self.shove / dur;
            if self.shove_dir == "sagittal" {
                [f, 0.0, 0.0]
            } else {
                [0.0, f, 0.0]
            }
        })
    }
}

/// v indices of the ten DOFs ipse's CEM `LinearPolicy` commands, in its
/// action order: per side (Left, Right) hip pitch, hip roll, knee, ankle
/// pitch, ankle roll (`ipse_sim::learn::leg_act_dofs`).
pub fn cem_dofs(model: &Model) -> Vec<usize> {
    let v_of = |name: String| {
        let j = model
            .joints
            .iter()
            .position(|j| j.name == name)
            .unwrap_or_else(|| panic!("{name}"));
        model.v_offsets[j]
    };
    ["Left", "Right"]
        .iter()
        .flat_map(|s| {
            [
                "Hip_Pitch",
                "Hip_Roll",
                "Knee_Pitch",
                "Ankle_Pitch",
                "Ankle_Roll",
            ]
            .map(|p| v_of(format!("{s}_{p}")))
        })
        .collect()
}

/// `ipse_sim::learn::observe_legs`: trunk pitch and roll, base gyro, COM
/// velocity and centroidal angular momentum in the heading frame, then
/// `(q - q_target, q̇)` over [`cem_dofs`]. 31 entries.
pub fn observe_legs(
    model: &Model,
    q_target: &[f64],
    q: &[f64],
    v: &[f64],
    dofs: &[usize],
) -> Vec<f64> {
    let mut st = model.default_state();
    st.q.as_mut_slice().copy_from_slice(q);
    st.v.as_mut_slice().copy_from_slice(v);
    let (xf, vel) = phyz_rigid::forward_kinematics(model, &st);
    let trunk = model
        .bodies
        .iter()
        .position(|b| b.name == "Trunk")
        .expect("Trunk");
    let r = &xf[trunk].rot;
    let pitch = (-r[(0, 2)]).atan2(r[(2, 2)]);
    let roll = r[(1, 2)].atan2(r[(2, 2)]);
    // COM and its velocity, then the centroidal angular momentum
    let (mut com, mut cv, mut m) = (Vec3::zeros(), Vec3::zeros(), 0.0);
    for (i, b) in model.bodies.iter().enumerate() {
        if b.inertia.mass <= 0.0 {
            continue;
        }
        let rw = xf[i].rot.transpose();
        com += (xf[i].pos + rw * b.inertia.com) * b.inertia.mass;
        cv += rw * (vel[i].linear + vel[i].angular.cross(b.inertia.com)) * b.inertia.mass;
        m += b.inertia.mass;
    }
    com *= 1.0 / m;
    cv *= 1.0 / m;
    let mut cam = Vec3::zeros();
    for (i, b) in model.bodies.iter().enumerate() {
        if b.inertia.mass <= 0.0 {
            continue;
        }
        let rw = xf[i].rot.transpose();
        let ri = xf[i].pos + rw * b.inertia.com;
        let vi = rw * (vel[i].linear + vel[i].angular.cross(b.inertia.com));
        cam +=
            (ri - com).cross(vi - cv) * b.inertia.mass + rw * (b.inertia.inertia * vel[i].angular);
    }
    let rt = r.transpose();
    let (sy, cy) = rt[(1, 0)].atan2(rt[(0, 0)]).sin_cos();
    let heading = |w: Vec3| [cy * w.x + sy * w.y, -sy * w.x + cy * w.y, w.z];
    let mut obs = vec![pitch, roll, v[0], v[1], v[2]];
    obs.extend(heading(cv));
    obs.extend(heading(cam));
    // q index = v index for the K1's 1-DOF joints after the free base
    for &d in dofs {
        obs.push(q[d] - q_target[d]);
        obs.push(v[d]);
    }
    obs
}

/// ipse's CEM `LinearPolicy` action: `clamp(W obs + b, ±0.3)`.
pub fn cem_action(params: &[f64], obs: &[f64]) -> Vec<f64> {
    let (no, na) = (obs.len(), 10);
    (0..na)
        .map(|i| {
            let w = &params[i * no..(i + 1) * no];
            (w.iter().zip(obs).map(|(a, b)| a * b).sum::<f64>() + params[na * no + i])
                .clamp(-0.3, 0.3)
        })
        .collect()
}

/// A CEM policy file: one parameter per non-comment line.
pub fn read_cem(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path)
        .expect("CEM policy file")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

/// One recorded CEM row: `(t, v, obs, act)`.
pub type CemRow = (f64, Vec<f64>, Vec<f64>, Vec<f64>);

/// The recorded rows of a CEM dump.
pub fn cem_rows(path: &str) -> Vec<CemRow> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut rows: Vec<CemRow> = Vec::new();
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let nums = || {
            w[2..]
                .iter()
                .map(|v| v.parse().unwrap())
                .collect::<Vec<f64>>()
        };
        match w.first().copied() {
            Some("vel") => rows.push((w[1].parse().unwrap(), nums(), Vec::new(), Vec::new())),
            Some("obs") => rows.last_mut().unwrap().2 = nums(),
            Some("act") => rows.last_mut().unwrap().3 = nums(),
            _ => {}
        }
    }
    rows
}
