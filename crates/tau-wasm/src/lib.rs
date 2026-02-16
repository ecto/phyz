use wasm_bindgen::prelude::*;

use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::{Model, ModelBuilder, State};
use tau_rigid::{aba, forward_kinematics, total_energy};

/// A simulation instance exposed to JS.
#[wasm_bindgen]
pub struct WasmSim {
    model: Model,
    state: State,
    rk4: bool,
}

#[wasm_bindgen]
impl WasmSim {
    /// Single pendulum: revolute about Z, gravity along -Y, rod length L.
    pub fn pendulum() -> WasmSim {
        let length = 1.5;
        let mass = 1.0;

        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = 0.8;

        WasmSim {
            model,
            state,
            rk4: true,
        }
    }

    /// Double pendulum: two revolute joints, chaotic dynamics.
    pub fn double_pendulum() -> WasmSim {
        let length = 1.0;
        let mass = 1.0;

        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, -length, 0.0)),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_4 * 2.4;
        state.q[1] = std::f64::consts::FRAC_PI_4 * 1.6;

        WasmSim {
            model,
            state,
            rk4: true,
        }
    }

    /// N-link chain: n revolute bodies chained together.
    pub fn chain(n: usize) -> WasmSim {
        let link_length = 0.6;
        let link_mass = 1.0 / n as f64;

        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.0005);

        for i in 0..n {
            let parent = if i == 0 { -1 } else { (i - 1) as i32 };
            let parent_to_joint = if i == 0 {
                SpatialTransform::identity()
            } else {
                SpatialTransform::translation(Vec3::new(0.0, -link_length, 0.0))
            };

            builder = builder.add_revolute_body(
                &format!("link{i}"),
                parent,
                parent_to_joint,
                SpatialInertia::new(
                    link_mass,
                    Vec3::new(0.0, -link_length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        link_mass * link_length * link_length / 12.0,
                        0.0,
                        link_mass * link_length * link_length / 12.0,
                    )),
                ),
            );
        }

        let model = builder.build();
        let mut state = model.default_state();

        // Gentle initial displacement
        state.q[0] = 0.5;
        for i in 1..n {
            state.q[i] = 0.05;
        }

        WasmSim {
            model,
            state,
            rk4: true,
        }
    }

    /// Step the simulation forward n times.
    pub fn step_n(&mut self, n: usize) {
        let dt = self.model.dt;

        if self.rk4 {
            for _ in 0..n {
                self.rk4_step(dt);
            }
        } else {
            for _ in 0..n {
                let qdd = aba(&self.model, &self.state);
                self.state.v += &qdd * dt;
                self.state.q += &self.state.v * dt;
                self.state.time += dt;
            }
        }
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.state.time
    }

    /// Total mechanical energy.
    pub fn total_energy(&self) -> f64 {
        total_energy(&self.model, &self.state)
    }

    /// Number of bodies.
    pub fn nbodies(&self) -> usize {
        self.model.nbodies()
    }

    /// Joint angles as flat array.
    pub fn joint_angles(&self) -> Vec<f64> {
        self.state.q.as_slice().to_vec()
    }

    /// Body joint positions (origins) in world frame as flat [x0,y0,z0, x1,y1,z1, ...].
    pub fn joint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut positions = Vec::with_capacity(nb * 3);

        for xf in xforms.iter().take(nb) {
            let p = xf.pos;
            positions.push(p.x);
            positions.push(p.y);
            positions.push(p.z);
        }

        positions
    }

    /// Body endpoint positions in world frame as flat [x0,y0,z0, x1,y1,z1, ...].
    pub fn body_endpoint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut positions = Vec::with_capacity(nb * 3);

        for (i, xf) in xforms.iter().enumerate().take(nb) {
            let body = &self.model.bodies[i];
            let endpoint_body = body.inertia.com * 2.0;

            // p_world = R^T * p_body + pos
            let rotated = xf.rot.transpose() * endpoint_body;
            let p = rotated + xf.pos;
            positions.push(p.x);
            positions.push(p.y);
            positions.push(p.z);
        }

        positions
    }
}

impl WasmSim {
    fn rk4_step(&mut self, dt: f64) {
        let q0 = self.state.q.clone();
        let v0 = self.state.v.clone();

        let k1v = aba(&self.model, &self.state);
        let k1q = self.state.v.clone();

        self.state.q = &q0 + &k1q * (dt / 2.0);
        self.state.v = &v0 + &k1v * (dt / 2.0);
        let k2v = aba(&self.model, &self.state);
        let k2q = self.state.v.clone();

        self.state.q = &q0 + &k2q * (dt / 2.0);
        self.state.v = &v0 + &k2v * (dt / 2.0);
        let k3v = aba(&self.model, &self.state);
        let k3q = self.state.v.clone();

        self.state.q = &q0 + &k3q * dt;
        self.state.v = &v0 + &k3v * dt;
        let k4v = aba(&self.model, &self.state);
        let k4q = self.state.v.clone();

        self.state.q = q0 + (&k1q + &k2q * 2.0 + &k3q * 2.0 + &k4q) * (dt / 6.0);
        self.state.v = v0 + (&k1v + &k2v * 2.0 + &k3v * 2.0 + &k4v) * (dt / 6.0);
        self.state.time += dt;
    }
}
