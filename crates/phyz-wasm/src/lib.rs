use wasm_bindgen::prelude::*;

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};
use phyz_rigid::{aba, forward_kinematics, total_energy};
use phyz_diff::analytical_step_jacobians;

// ================================================================
// Articulated Body Simulation (pendulums, chains)
// ================================================================

#[wasm_bindgen]
pub struct WasmSim {
    model: Model,
    state: State,
}

#[wasm_bindgen]
impl WasmSim {
    /// Single pendulum with viscous damping.
    pub fn pendulum() -> WasmSim {
        let length = 1.5;
        let mass = 1.0;

        let mut model = ModelBuilder::new()
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

        for j in &mut model.joints {
            j.damping = 0.08;
        }

        let mut state = model.default_state();
        state.q[0] = 0.8;

        WasmSim { model, state }
    }

    /// Double pendulum with viscous damping.
    pub fn double_pendulum() -> WasmSim {
        let length = 1.0;
        let mass = 1.0;

        let mut model = ModelBuilder::new()
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

        for j in &mut model.joints {
            j.damping = 0.05;
        }

        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_4 * 2.4;
        state.q[1] = std::f64::consts::FRAC_PI_4 * 1.6;

        WasmSim { model, state }
    }

    /// N-link chain with viscous damping.
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

        let mut model = builder.build();
        for j in &mut model.joints {
            j.damping = 0.03;
        }

        let mut state = model.default_state();
        state.q[0] = 0.5;
        for i in 1..n {
            state.q[i] = 0.05;
        }

        WasmSim { model, state }
    }

    /// Step the simulation forward n times using RK4.
    pub fn step_n(&mut self, n: usize) {
        let dt = self.model.dt;
        for _ in 0..n {
            self.rk4_step(dt);
        }
    }

    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn total_energy(&self) -> f64 {
        total_energy(&self.model, &self.state)
    }

    pub fn nbodies(&self) -> usize {
        self.model.nbodies()
    }

    pub fn joint_angles(&self) -> Vec<f64> {
        self.state.q.as_slice().to_vec()
    }

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

    pub fn body_endpoint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut positions = Vec::with_capacity(nb * 3);
        for (i, xf) in xforms.iter().enumerate().take(nb) {
            let body = &self.model.bodies[i];
            let endpoint_body = body.inertia.com * 2.0;
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

// ================================================================
// Particle Contact Simulation (bouncing spheres, cascade)
// ================================================================

#[wasm_bindgen]
pub struct WasmParticleSim {
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    radii: Vec<f64>,
    masses: Vec<f64>,
    bounce: Vec<f64>,
    stiffness: f64,
    contact_damping: f64,
    friction: f64,
    n: usize,
    time: f64,
    dt: f64,
    wall_x: f64, // side wall distance (0 = no walls)
}

#[wasm_bindgen]
impl WasmParticleSim {
    /// Three spheres with different bounce coefficients.
    pub fn bouncing_spheres() -> WasmParticleSim {
        let r = 0.15;
        let m = 1.0;
        let h = 2.5;
        WasmParticleSim {
            px: vec![-0.6, 0.0, 0.6],
            py: vec![h, h, h],
            vx: vec![0.0; 3],
            vy: vec![0.0; 3],
            radii: vec![r; 3],
            masses: vec![m; 3],
            bounce: vec![0.92, 0.50, 0.05],
            stiffness: 50000.0,
            contact_damping: 200.0,
            friction: 0.4,
            n: 3,
            time: 0.0,
            dt: 0.0005,
            wall_x: 0.0,
        }
    }

    /// 20 spheres cascading into a pile.
    pub fn sphere_cascade() -> WasmParticleSim {
        let n = 20;
        let r = 0.06;
        let m = 0.3;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let vy = vec![0.0; n];

        for i in 0..n {
            let row = i / 3;
            let col = (i % 3) as f64 - 1.0;
            let x = col * 0.13 + if row % 2 == 1 { 0.04 } else { 0.0 };
            let y = 1.5 + i as f64 * 0.45;
            px.push(x);
            py.push(y);
            // Slight deterministic horizontal scatter
            let scatter = ((i as f64 * 7.3 + 2.1) % 3.0 - 1.5) * 0.08;
            vx.push(scatter);
        }

        WasmParticleSim {
            px,
            py,
            vx,
            vy,
            radii: vec![r; n],
            masses: vec![m; n],
            bounce: vec![0.15; n],
            stiffness: 40000.0,
            contact_damping: 300.0,
            friction: 0.6,
            n,
            time: 0.0,
            dt: 0.0005,
            wall_x: 0.4,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let g = -GRAVITY;
        for _ in 0..steps {
            // Gravity
            for i in 0..self.n {
                self.vy[i] += g * self.dt;
            }

            // Ground contacts
            for i in 0..self.n {
                let r = self.radii[i];
                if self.py[i] < r {
                    let depth = r - self.py[i];
                    let vn = self.vy[i];

                    if vn < -0.3 && depth > 0.001 {
                        // Bounce
                        self.vy[i] = (-vn) * self.bounce[i];
                        self.py[i] = r;
                    } else {
                        // Penalty push
                        let f = self.stiffness * depth - self.contact_damping * vn;
                        let f = f.max(0.0);
                        self.vy[i] += f / self.masses[i] * self.dt;
                    }

                    // Ground friction
                    self.vx[i] *= 1.0 - self.friction * self.dt * 5.0;

                    // Prevent sinking
                    if self.py[i] < r * 0.3 {
                        self.py[i] = r * 0.3;
                        if self.vy[i] < 0.0 {
                            self.vy[i] = 0.0;
                        }
                    }
                }
            }

            // Sphere-sphere contacts (impulse-based)
            for i in 0..self.n {
                for j in (i + 1)..self.n {
                    let dx = self.px[j] - self.px[i];
                    let dy = self.py[j] - self.py[i];
                    let dist = (dx * dx + dy * dy).sqrt();
                    let min_dist = self.radii[i] + self.radii[j];

                    if dist < min_dist && dist > 1e-10 {
                        let nx = dx / dist;
                        let ny = dy / dist;

                        let dvx = self.vx[j] - self.vx[i];
                        let dvy = self.vy[j] - self.vy[i];
                        let rel_vn = dvx * nx + dvy * ny;

                        if rel_vn < 0.0 {
                            // Equal mass impulse with restitution
                            let e = (self.bounce[i] + self.bounce[j]) * 0.5;
                            let impulse = -(1.0 + e) * rel_vn * 0.5;

                            self.vx[i] -= impulse * nx;
                            self.vy[i] -= impulse * ny;
                            self.vx[j] += impulse * nx;
                            self.vy[j] += impulse * ny;
                        }

                        // Separate overlapping spheres
                        let overlap = min_dist - dist;
                        let sep = overlap * 0.5 + 0.001;
                        self.px[i] -= sep * nx;
                        self.py[i] -= sep * ny;
                        self.px[j] += sep * nx;
                        self.py[j] += sep * ny;
                    }
                }
            }

            // Side walls
            if self.wall_x > 0.0 {
                for i in 0..self.n {
                    let r = self.radii[i];
                    if self.px[i] < -self.wall_x + r {
                        self.px[i] = -self.wall_x + r;
                        if self.vx[i] < 0.0 {
                            self.vx[i] = -self.vx[i] * 0.2;
                        }
                    }
                    if self.px[i] > self.wall_x - r {
                        self.px[i] = self.wall_x - r;
                        if self.vx[i] > 0.0 {
                            self.vx[i] = -self.vx[i] * 0.2;
                        }
                    }
                }
            }

            // Integrate positions
            for i in 0..self.n {
                self.px[i] += self.vx[i] * self.dt;
                self.py[i] += self.vy[i] * self.dt;
            }

            self.time += self.dt;
        }
    }

    /// Flat [x0,y0,z0, x1,y1,z1, ...] positions.
    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 3);
        for i in 0..self.n {
            out.push(self.px[i]);
            out.push(self.py[i]);
            out.push(0.0);
        }
        out
    }

    pub fn num_particles(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn radii(&self) -> Vec<f64> {
        self.radii.clone()
    }
}

// ================================================================
// Newton's Cradle Simulation
// ================================================================

#[wasm_bindgen]
pub struct WasmCradleSim {
    model: Model,
    state: State,
    bob_radius: f64,
    pend_length: f64,
    spacing: f64,
    n: usize,
}

#[wasm_bindgen]
impl WasmCradleSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmCradleSim {
        let n = 5;
        let length = 1.5;
        let mass = 1.0;
        let bob_radius = 0.12;
        let spacing = bob_radius * 1.98; // slightly less than 2r so resting bobs overlap

        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.0005);

        for i in 0..n {
            let offset = (i as f64 - 2.0) * spacing;
            builder = builder.add_revolute_body(
                &format!("bob{i}"),
                -1,
                SpatialTransform::translation(Vec3::new(offset, 0.0, 0.0)),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            );
        }

        let mut model = builder.build();
        for j in &mut model.joints {
            j.damping = 0.008;
        }

        let mut state = model.default_state();
        state.q[0] = 0.6; // pull first bob back ~35 degrees

        WasmCradleSim {
            model,
            state,
            bob_radius,
            pend_length: length,
            spacing,
            n,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let dt = self.model.dt;
        for _ in 0..steps {
            self.rk4_step(dt);
            self.resolve_collisions();
        }
    }

    /// Bob positions as flat [x0,y0,z0, ...].
    pub fn bob_positions(&self) -> Vec<f64> {
        let l = self.pend_length;
        let mut out = Vec::with_capacity(self.n * 3);
        for i in 0..self.n {
            let pivot_x = (i as f64 - 2.0) * self.spacing;
            let theta = self.state.q[i];
            out.push(pivot_x + l * theta.sin());
            out.push(-l * theta.cos());
            out.push(0.0);
        }
        out
    }

    /// Pivot positions as flat [x0,y0,z0, ...].
    pub fn pivot_positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 3);
        for i in 0..self.n {
            out.push((i as f64 - 2.0) * self.spacing);
            out.push(0.0);
            out.push(0.0);
        }
        out
    }

    pub fn num_bobs(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn bob_radius(&self) -> f64 {
        self.bob_radius
    }
}

impl WasmCradleSim {
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

    fn resolve_collisions(&mut self) {
        let l = self.pend_length;
        let r = self.bob_radius;

        // Multiple passes so the impulse wave propagates through the chain.
        // Process from both ends alternately for better propagation.
        for pass in 0..(self.n * 2) {
            let forward = pass % 2 == 0;
            for step in 0..(self.n - 1) {
                let i = if forward { step } else { self.n - 2 - step };
                let j = i + 1;

                let pivot_xi = (i as f64 - 2.0) * self.spacing;
                let pivot_xj = (j as f64 - 2.0) * self.spacing;

                let bxi = pivot_xi + l * self.state.q[i].sin();
                let byi = -l * self.state.q[i].cos();
                let bxj = pivot_xj + l * self.state.q[j].sin();
                let byj = -l * self.state.q[j].cos();

                let dx = bxj - bxi;
                let dy = byj - byi;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 2.0 * r + 0.001 && dist > 1e-10 {
                    // Project angular velocities to bob linear velocities
                    let vxi = self.state.v[i] * l * self.state.q[i].cos();
                    let vxj = self.state.v[j] * l * self.state.q[j].cos();
                    let nx = dx / dist;
                    let rel_vn = (vxj - vxi) * nx;

                    if rel_vn < 0.0 {
                        // Equal-mass elastic collision: swap velocities
                        let e = 0.999;
                        let vi = self.state.v[i];
                        let vj = self.state.v[j];
                        self.state.v[i] = vj * e;
                        self.state.v[j] = vi * e;
                    }

                    // Separate overlapping bobs
                    if dist < 2.0 * r {
                        let overlap = 2.0 * r - dist;
                        let sep = overlap / (2.0 * l);
                        self.state.q[i] -= sep;
                        self.state.q[j] += sep;
                    }
                }
            }
        }
    }
}

// ================================================================
// GPU Batch Simulations (CPU-backed in WASM, demonstrates concept)
// ================================================================

/// 1024 pendulums with varying lengths creating wave interference.
#[wasm_bindgen]
pub struct WasmWaveFieldSim {
    n: usize,
    q: Vec<f64>,
    v: Vec<f64>,
    freq_sq: Vec<f64>, // precomputed 1.5 * g / L per pendulum
    time: f64,
    dt: f64,
}

#[wasm_bindgen]
impl WasmWaveFieldSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmWaveFieldSim {
        let n = 1024;
        let mut q = vec![0.0; n];
        let v = vec![0.0; n];
        let mut freq_sq = vec![0.0; n];

        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let length = 0.6 + t * 0.8; // 0.6 to 1.4
            freq_sq[i] = 1.5 * GRAVITY / length; // omega^2
            q[i] = 0.5; // same initial displacement
        }

        WasmWaveFieldSim {
            n,
            q,
            v,
            freq_sq,
            time: 0.0,
            dt: 0.002,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            for i in 0..self.n {
                let qdd = -self.freq_sq[i] * self.q[i].sin();
                self.v[i] += qdd * self.dt;
                self.q[i] += self.v[i] * self.dt;
            }
            self.time += self.dt;
        }
    }

    pub fn angles(&self) -> Vec<f64> {
        self.q.clone()
    }

    pub fn velocities(&self) -> Vec<f64> {
        self.v.clone()
    }

    pub fn num_pendulums(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.time
    }
}

/// 100 double pendulums with tiny perturbations showing chaotic divergence.
#[wasm_bindgen]
pub struct WasmEnsembleSim {
    model: Model,
    states: Vec<State>,
    n: usize,
}

#[wasm_bindgen]
impl WasmEnsembleSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmEnsembleSim {
        let n = 100;
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

        let base_q0 = std::f64::consts::FRAC_PI_4 * 2.4;
        let base_q1 = std::f64::consts::FRAC_PI_4 * 1.6;

        let mut states = Vec::with_capacity(n);
        for i in 0..n {
            let mut state = model.default_state();
            let eps = (i as f64 - 50.0) * 0.0002;
            state.q[0] = base_q0 + eps;
            state.q[1] = base_q1 + eps * 0.5;
            states.push(state);
        }

        WasmEnsembleSim { model, states, n }
    }

    pub fn step_n(&mut self, steps: usize) {
        let dt = self.model.dt;
        for _ in 0..steps {
            for i in 0..self.n {
                let q0 = self.states[i].q.clone();
                let v0 = self.states[i].v.clone();

                let k1v = aba(&self.model, &self.states[i]);
                let k1q = self.states[i].v.clone();

                self.states[i].q = &q0 + &k1q * (dt / 2.0);
                self.states[i].v = &v0 + &k1v * (dt / 2.0);
                let k2v = aba(&self.model, &self.states[i]);
                let k2q = self.states[i].v.clone();

                self.states[i].q = &q0 + &k2q * (dt / 2.0);
                self.states[i].v = &v0 + &k2v * (dt / 2.0);
                let k3v = aba(&self.model, &self.states[i]);
                let k3q = self.states[i].v.clone();

                self.states[i].q = &q0 + &k3q * dt;
                self.states[i].v = &v0 + &k3v * dt;
                let k4v = aba(&self.model, &self.states[i]);
                let k4q = self.states[i].v.clone();

                self.states[i].q = q0 + (&k1q + &k2q * 2.0 + &k3q * 2.0 + &k4q) * (dt / 6.0);
                self.states[i].v = v0 + (&k1v + &k2v * 2.0 + &k3v * 2.0 + &k4v) * (dt / 6.0);
                self.states[i].time += dt;
            }
        }
    }

    /// Second bob endpoint positions as flat [x0,y0,z0, ...].
    pub fn endpoint_positions(&self) -> Vec<f64> {
        let l1 = 1.0;
        let l2 = 1.0;
        let mut out = Vec::with_capacity(self.n * 3);
        for i in 0..self.n {
            let q0 = self.states[i].q[0];
            let q1 = self.states[i].q[1];
            let abs2 = q0 + q1;
            let x = l1 * q0.sin() + l2 * abs2.sin();
            let y = -l1 * q0.cos() - l2 * abs2.cos();
            out.push(x);
            out.push(y);
            out.push(0.0);
        }
        out
    }

    pub fn num_instances(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.states[0].time
    }
}

/// 100 pendulums with different PD controllers for swing-up.
#[wasm_bindgen]
pub struct WasmPolicyGridSim {
    n: usize,
    q: Vec<f64>,
    v: Vec<f64>,
    kp: Vec<f64>,
    kd: Vec<f64>,
    freq_sq: f64, // 1.5 * g / L
    inertia: f64,
    time: f64,
    dt: f64,
}

#[wasm_bindgen]
impl WasmPolicyGridSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmPolicyGridSim {
        let n = 100;
        let length = 1.0;
        let mass = 1.0;
        let inertia = mass * length * length / 3.0;

        let mut q = vec![0.0; n];
        let v = vec![0.0; n];
        let mut kp = vec![0.0; n];
        let mut kd = vec![0.0; n];

        for i in 0..n {
            let row = i / 10;
            let col = i % 10;
            kp[i] = 1.0 + row as f64 * 2.7; // 1.0 to 25.3
            kd[i] = col as f64 * 0.55; // 0.0 to 4.95
            q[i] = std::f64::consts::PI - 0.5; // start near inverted
        }

        WasmPolicyGridSim {
            n,
            q,
            v,
            kp,
            kd,
            freq_sq: 1.5 * GRAVITY / length,
            inertia,
            time: 0.0,
            dt: 0.002,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            for i in 0..self.n {
                // PD controller: torque toward inverted (q=π)
                let ctrl = self.kp[i] * self.q[i].sin() - self.kd[i] * self.v[i];
                let gravity = -self.freq_sq * self.q[i].sin();
                let qdd = (ctrl / self.inertia) + gravity;
                self.v[i] += qdd * self.dt;
                self.q[i] += self.v[i] * self.dt;
            }
            self.time += self.dt;
        }
    }

    pub fn angles(&self) -> Vec<f64> {
        self.q.clone()
    }

    /// Reward: 1 when inverted (q=π), 0 when hanging (q=0).
    pub fn rewards(&self) -> Vec<f64> {
        self.q.iter().map(|&q| (1.0 - q.cos()) * 0.5).collect()
    }

    pub fn num_envs(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.time
    }
}

// ================================================================
// Particle Physics: DEM / MPM Concept (Phase 5: phyz-particle)
// ================================================================

#[wasm_bindgen]
pub struct WasmMpmSim {
    n: usize,
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    radius: f64,
    stiffness: f64,
    contact_damp: f64,
    friction: f64,
    wall_bounce: f64,
    sp_a: Vec<usize>,
    sp_b: Vec<usize>,
    sp_rest: Vec<f64>,
    sp_k: f64,
    x_min: f64,
    x_max: f64,
    time: f64,
    dt: f64,
}

#[wasm_bindgen]
impl WasmMpmSim {
    pub fn granular_column() -> WasmMpmSim {
        let cols = 8;
        let n = cols * 35;
        let r = 0.02;
        let spacing = r * 2.15;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        for i in 0..n {
            let col = i % cols;
            let row = i / cols;
            px.push((col as f64 - cols as f64 / 2.0 + 0.5) * spacing);
            py.push(r + row as f64 * spacing);
        }
        WasmMpmSim {
            n,
            px,
            py,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            radius: r,
            stiffness: 80000.0,
            contact_damp: 200.0,
            friction: 0.8,
            wall_bounce: 0.1,
            sp_a: vec![],
            sp_b: vec![],
            sp_rest: vec![],
            sp_k: 0.0,
            x_min: -0.8,
            x_max: 0.8,
            time: 0.0,
            dt: 0.00015,
        }
    }

    pub fn fluid_dam() -> WasmMpmSim {
        let cols = 10;
        let n = cols * 25;
        let r = 0.018;
        let spacing = r * 2.2;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        for i in 0..n {
            let col = i % cols;
            let row = i / cols;
            px.push(-0.45 + (col as f64 + 0.5) * spacing);
            py.push(r + row as f64 * spacing);
        }
        WasmMpmSim {
            n,
            px,
            py,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            radius: r,
            stiffness: 25000.0,
            contact_damp: 500.0,
            friction: 0.01,
            wall_bounce: 0.05,
            sp_a: vec![],
            sp_b: vec![],
            sp_rest: vec![],
            sp_k: 0.0,
            x_min: -0.6,
            x_max: 0.6,
            time: 0.0,
            dt: 0.00015,
        }
    }

    pub fn elastic_blob() -> WasmMpmSim {
        let r = 0.022;
        let spacing = r * 2.1;
        let blob_r = 0.2;
        let cy = 0.8;
        let mut px = vec![];
        let mut py = vec![];
        let ny = (blob_r * 2.0 / (spacing * 0.866)) as i32 + 1;
        for row in -ny..=ny {
            let y = row as f64 * spacing * 0.866;
            let xr = (blob_r * blob_r - y * y).max(0.0).sqrt();
            let nx = (xr / spacing) as i32;
            for col in -nx..=nx {
                let x = col as f64 * spacing + if row & 1 != 0 { spacing * 0.5 } else { 0.0 };
                if x * x + y * y <= blob_r * blob_r {
                    px.push(x);
                    py.push(cy + y);
                }
            }
        }
        let n = px.len();
        let mut sp_a = vec![];
        let mut sp_b = vec![];
        let mut sp_rest = vec![];
        let conn = spacing * 1.5;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = px[j] - px[i];
                let dy = py[j] - py[i];
                let d = (dx * dx + dy * dy).sqrt();
                if d < conn {
                    sp_a.push(i);
                    sp_b.push(j);
                    sp_rest.push(d);
                }
            }
        }
        WasmMpmSim {
            n,
            px,
            py,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            radius: r,
            stiffness: 40000.0,
            contact_damp: 100.0,
            friction: 0.3,
            wall_bounce: 0.4,
            sp_a,
            sp_b,
            sp_rest,
            sp_k: 8000.0,
            x_min: -0.8,
            x_max: 0.8,
            time: 0.0,
            dt: 0.00015,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let g = -GRAVITY;
        for _ in 0..steps {
            for i in 0..self.n {
                self.vy[i] += g * self.dt;
            }
            for i in 0..self.n {
                for j in (i + 1)..self.n {
                    let dx = self.px[j] - self.px[i];
                    let dy = self.py[j] - self.py[i];
                    let d2 = dx * dx + dy * dy;
                    let mind = self.radius * 2.0;
                    if d2 < mind * mind && d2 > 1e-20 {
                        let d = d2.sqrt();
                        let nx = dx / d;
                        let ny = dy / d;
                        let overlap = mind - d;
                        let dvn = (self.vx[j] - self.vx[i]) * nx + (self.vy[j] - self.vy[i]) * ny;
                        let fn_mag = (self.stiffness * overlap - self.contact_damp * dvn).max(0.0);
                        let dvt =
                            (self.vx[j] - self.vx[i]) * (-ny) + (self.vy[j] - self.vy[i]) * nx;
                        let ft_mag = if dvt.abs() > 0.001 {
                            (-self.friction * fn_mag).max(-dvt.abs() * 500.0) * dvt.signum()
                        } else {
                            0.0
                        };
                        let fx = fn_mag * nx + ft_mag * (-ny);
                        let fy = fn_mag * ny + ft_mag * nx;
                        self.vx[i] -= fx * self.dt;
                        self.vy[i] -= fy * self.dt;
                        self.vx[j] += fx * self.dt;
                        self.vy[j] += fy * self.dt;
                        let sep = overlap * 0.25;
                        self.px[i] -= sep * nx;
                        self.py[i] -= sep * ny;
                        self.px[j] += sep * nx;
                        self.py[j] += sep * ny;
                    }
                }
            }
            for s in 0..self.sp_a.len() {
                let (i, j) = (self.sp_a[s], self.sp_b[s]);
                let dx = self.px[j] - self.px[i];
                let dy = self.py[j] - self.py[i];
                let d = (dx * dx + dy * dy).sqrt();
                if d > 1e-10 {
                    let f = self.sp_k * (d - self.sp_rest[s]) / d;
                    self.vx[i] += f * dx * self.dt;
                    self.vy[i] += f * dy * self.dt;
                    self.vx[j] -= f * dx * self.dt;
                    self.vy[j] -= f * dy * self.dt;
                }
            }
            for i in 0..self.n {
                let r = self.radius;
                if self.py[i] < r {
                    self.py[i] = r;
                    if self.vy[i] < 0.0 {
                        self.vy[i] *= -self.wall_bounce;
                    }
                    self.vx[i] *= 1.0 - self.friction * self.dt * 10.0;
                }
                if self.px[i] < self.x_min + r {
                    self.px[i] = self.x_min + r;
                    if self.vx[i] < 0.0 {
                        self.vx[i] *= -self.wall_bounce;
                    }
                }
                if self.px[i] > self.x_max - r {
                    self.px[i] = self.x_max - r;
                    if self.vx[i] > 0.0 {
                        self.vx[i] *= -self.wall_bounce;
                    }
                }
            }
            for i in 0..self.n {
                self.px[i] += self.vx[i] * self.dt;
                self.py[i] += self.vy[i] * self.dt;
            }
            self.time += self.dt;
        }
    }

    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.px[i]);
            out.push(self.py[i]);
        }
        out
    }
    pub fn velocities(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.vx[i]);
            out.push(self.vy[i]);
        }
        out
    }
    pub fn num_particles(&self) -> usize {
        self.n
    }
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn particle_radius(&self) -> f64 {
        self.radius
    }
    pub fn num_springs(&self) -> usize {
        self.sp_a.len()
    }
    pub fn spring_endpoints(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.sp_a.len() * 4);
        for s in 0..self.sp_a.len() {
            let (i, j) = (self.sp_a[s], self.sp_b[s]);
            out.push(self.px[i]);
            out.push(self.py[i]);
            out.push(self.px[j]);
            out.push(self.py[j]);
        }
        out
    }
}

// ================================================================
// World Generation + Training (Phase 7: phyz-world)
// ================================================================

#[wasm_bindgen]
pub struct WasmWorldSim {
    model: Model,
    state: State,
    link_lengths: Vec<f64>,
    variant: u8,
}

#[wasm_bindgen]
impl WasmWorldSim {
    pub fn random_chain() -> WasmWorldSim {
        let lengths = vec![0.7, 0.45, 0.85, 0.5, 0.65, 0.4];
        let n = lengths.len();
        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.0005);
        for i in 0..n {
            let parent = if i == 0 { -1 } else { (i - 1) as i32 };
            let pj = if i == 0 {
                SpatialTransform::identity()
            } else {
                SpatialTransform::translation(Vec3::new(0.0, -lengths[i - 1], 0.0))
            };
            let m = 0.3 + (i as f64) * 0.1;
            let l = lengths[i];
            builder = builder.add_revolute_body(
                &format!("link{i}"),
                parent,
                pj,
                SpatialInertia::new(
                    m,
                    Vec3::new(0.0, -l / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(m * l * l / 12.0, 0.0, m * l * l / 12.0)),
                ),
            );
        }
        let mut model = builder.build();
        for j in &mut model.joints {
            j.damping = 0.04;
        }
        let mut state = model.default_state();
        let ics = [0.3, -0.2, 0.4, -0.1, 0.2, -0.15];
        for (i, &ic) in ics.iter().enumerate().take(n) {
            state.q[i] = ic;
        }
        WasmWorldSim {
            model,
            state,
            link_lengths: lengths,
            variant: 0,
        }
    }

    pub fn phase_portrait() -> WasmWorldSim {
        let l = 1.5;
        let m = 1.0;
        let mut model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pend",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    m,
                    Vec3::new(0.0, -l / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(m * l * l / 12.0, 0.0, m * l * l / 12.0)),
                ),
            )
            .build();
        model.joints[0].damping = 0.15;
        let mut state = model.default_state();
        state.q[0] = 2.8;
        state.v[0] = 1.0;
        WasmWorldSim {
            model,
            state,
            link_lengths: vec![l],
            variant: 1,
        }
    }

    pub fn tendon_actuated() -> WasmWorldSim {
        let n = 4;
        let l = 0.6;
        let m = 0.5;
        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.0005);
        for i in 0..n {
            let parent = if i == 0 { -1 } else { (i - 1) as i32 };
            let pj = if i == 0 {
                SpatialTransform::identity()
            } else {
                SpatialTransform::translation(Vec3::new(0.0, -l, 0.0))
            };
            builder = builder.add_revolute_body(
                &format!("link{i}"),
                parent,
                pj,
                SpatialInertia::new(
                    m,
                    Vec3::new(0.0, -l / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(m * l * l / 12.0, 0.0, m * l * l / 12.0)),
                ),
            );
        }
        let mut model = builder.build();
        for j in &mut model.joints {
            j.damping = 0.1;
        }
        let state = model.default_state();
        WasmWorldSim {
            model,
            state,
            link_lengths: vec![l; n],
            variant: 2,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let dt = self.model.dt;
        for _ in 0..steps {
            if self.variant == 2 {
                let t = self.state.time;
                let nv = self.model.nv;
                for i in 0..nv {
                    let freq = 0.5 + i as f64 * 0.3;
                    let phase = i as f64 * 1.2;
                    self.state.ctrl[i] =
                        3.0 * (2.0 * std::f64::consts::PI * freq * t + phase).sin();
                }
            }
            self.rk4_step(dt);
        }
    }

    pub fn body_endpoint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut out = Vec::with_capacity(nb * 3);
        for (i, xf) in xforms.iter().enumerate().take(nb) {
            let body = &self.model.bodies[i];
            let ep = body.inertia.com * 2.0;
            let p = xf.rot.transpose() * ep + xf.pos;
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
        }
        out
    }

    pub fn link_lengths(&self) -> Vec<f64> {
        self.link_lengths.clone()
    }
    pub fn nbodies(&self) -> usize {
        self.model.nbodies()
    }
    pub fn time(&self) -> f64 {
        self.state.time
    }
    pub fn q(&self) -> f64 {
        self.state.q[0]
    }
    pub fn v(&self) -> f64 {
        self.state.v[0]
    }
}

impl WasmWorldSim {
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

// ================================================================
// Electromagnetic Field Simulation (Phase 8: phyz-em)
// ================================================================

#[wasm_bindgen]
pub struct WasmEmSim {
    nx: usize,
    ny: usize,
    u: Vec<f64>,
    u_prev: Vec<f64>,
    c: f64,
    dx: f64,
    dt_sim: f64,
    time: f64,
    variant: u8,
    mask: Vec<f64>,
    damp: Vec<f64>,
}

#[wasm_bindgen]
impl WasmEmSim {
    pub fn dipole() -> WasmEmSim {
        let n = 80;
        let dx = 1.0 / n as f64;
        let c = 1.0;
        let dt = dx / (c * 2.0_f64.sqrt()) * 0.9;
        let mut damp = vec![0.0; n * n];
        let pml = 8;
        for y in 0..n {
            for x in 0..n {
                let de = x.min(n - 1 - x).min(y.min(n - 1 - y));
                if de < pml {
                    damp[y * n + x] = ((pml - de) as f64 / pml as f64).powi(2) * 5.0;
                }
            }
        }
        WasmEmSim {
            nx: n,
            ny: n,
            u: vec![0.0; n * n],
            u_prev: vec![0.0; n * n],
            c,
            dx,
            dt_sim: dt,
            time: 0.0,
            variant: 0,
            mask: vec![1.0; n * n],
            damp,
        }
    }

    pub fn waveguide() -> WasmEmSim {
        let n = 80;
        let dx = 1.0 / n as f64;
        let c = 1.0;
        let dt = dx / (c * 2.0_f64.sqrt()) * 0.9;
        let mut mask = vec![1.0; n * n];
        let mut damp = vec![0.0; n * n];
        let pml = 8;
        let ch_lo = n / 2 - 6;
        let ch_hi = n / 2 + 6;
        for y in 0..n {
            for x in 0..n {
                let idx = y * n + x;
                if x > 5 && x < n - 5 && (y <= ch_lo || y >= ch_hi) {
                    mask[idx] = 0.0;
                }
                let de = x.min(n - 1 - x).min(y.min(n - 1 - y));
                if de < pml {
                    damp[idx] = ((pml - de) as f64 / pml as f64).powi(2) * 5.0;
                }
            }
        }
        WasmEmSim {
            nx: n,
            ny: n,
            u: vec![0.0; n * n],
            u_prev: vec![0.0; n * n],
            c,
            dx,
            dt_sim: dt,
            time: 0.0,
            variant: 1,
            mask,
            damp,
        }
    }

    pub fn double_slit() -> WasmEmSim {
        let n = 80;
        let dx = 1.0 / n as f64;
        let c = 1.0;
        let dt = dx / (c * 2.0_f64.sqrt()) * 0.9;
        let mut mask = vec![1.0; n * n];
        let mut damp = vec![0.0; n * n];
        let pml = 8;
        let bx = n / 3;
        let s1lo = n / 2 - 10;
        let s1hi = n / 2 - 5;
        let s2lo = n / 2 + 5;
        let s2hi = n / 2 + 10;
        for y in 0..n {
            let in_slit = (y >= s1lo && y <= s1hi) || (y >= s2lo && y <= s2hi);
            if !in_slit {
                mask[y * n + bx] = 0.0;
                if bx + 1 < n {
                    mask[y * n + bx + 1] = 0.0;
                }
            }
        }
        for y in 0..n {
            for x in 0..n {
                let de = x.min(n - 1 - x).min(y.min(n - 1 - y));
                if de < pml {
                    damp[y * n + x] = ((pml - de) as f64 / pml as f64).powi(2) * 5.0;
                }
            }
        }
        WasmEmSim {
            nx: n,
            ny: n,
            u: vec![0.0; n * n],
            u_prev: vec![0.0; n * n],
            c,
            dx,
            dt_sim: dt,
            time: 0.0,
            variant: 2,
            mask,
            damp,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let r2 = (self.c * self.dt_sim / self.dx).powi(2);
        let (nx, ny) = (self.nx, self.ny);
        let omega = 15.0;
        for _ in 0..steps {
            let mut u_new = vec![0.0; nx * ny];
            for y in 1..(ny - 1) {
                for x in 1..(nx - 1) {
                    let idx = y * nx + x;
                    if self.mask[idx] == 0.0 {
                        continue;
                    }
                    let lap =
                        self.u[idx + 1] + self.u[idx - 1] + self.u[idx + nx] + self.u[idx - nx]
                            - 4.0 * self.u[idx];
                    let d = self.damp[idx] * self.dt_sim;
                    u_new[idx] =
                        (2.0 * self.u[idx] - self.u_prev[idx] * (1.0 - d) + r2 * lap) / (1.0 + d);
                }
            }
            match self.variant {
                0 => {
                    u_new[ny / 2 * nx + nx / 2] += 2.0 * (omega * self.time).sin();
                }
                1 => {
                    for dy in -3i32..=3 {
                        let y = (ny as i32 / 2 + dy) as usize;
                        u_new[y * nx + 3] += 1.5 * (omega * self.time).sin();
                    }
                }
                2 => {
                    for y in 1..(ny - 1) {
                        u_new[y * nx + 3] += 0.8 * (omega * self.time).sin();
                    }
                }
                _ => {}
            }
            self.u_prev = std::mem::replace(&mut self.u, u_new);
            self.time += self.dt_sim;
        }
    }

    pub fn field(&self) -> Vec<f64> {
        self.u.clone()
    }
    pub fn mask_data(&self) -> Vec<f64> {
        self.mask.clone()
    }
    pub fn grid_size(&self) -> usize {
        self.nx
    }
    pub fn time(&self) -> f64 {
        self.time
    }
}

// ================================================================
// Molecular Dynamics (Phase 9: phyz-md)
// ================================================================

#[wasm_bindgen]
pub struct WasmMdSim {
    n: usize,
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    ax: Vec<f64>,
    ay: Vec<f64>,
    epsilon: f64,
    sigma: f64,
    r_cut: f64,
    box_size: f64,
    bond_a: Vec<usize>,
    bond_b: Vec<usize>,
    bond_k: f64,
    bond_r0: f64,
    time: f64,
    dt: f64,
}

#[wasm_bindgen]
impl WasmMdSim {
    pub fn argon_gas() -> WasmMdSim {
        let n = 80;
        let sigma = 0.04;
        let box_size = 0.8;
        let cols = 8;
        let sp = box_size / (cols as f64 + 1.0);
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        for i in 0..n {
            px.push(sp * ((i % cols) as f64 + 1.0));
            py.push(sp * ((i / cols) as f64 + 1.0));
            let s = (i as f64 + 0.5) * 2.3998;
            vx.push((s * 7.31).sin() * 3.0);
            vy.push((s * 11.17).cos() * 3.0);
        }
        let mut sim = WasmMdSim {
            n,
            px,
            py,
            vx,
            vy,
            ax: vec![0.0; n],
            ay: vec![0.0; n],
            epsilon: 1.0,
            sigma,
            r_cut: 2.5 * sigma,
            box_size,
            bond_a: vec![],
            bond_b: vec![],
            bond_k: 0.0,
            bond_r0: 0.0,
            time: 0.0,
            dt: 0.0001,
        };
        sim.compute_forces();
        sim
    }

    pub fn crystal() -> WasmMdSim {
        let side = 10;
        let n = side * side;
        let sigma = 0.04;
        let eq = sigma * 1.122;
        let box_size = (side as f64 + 1.0) * eq;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        for i in 0..n {
            px.push(eq * ((i % side) as f64 + 0.5));
            py.push(eq * ((i / side) as f64 + 0.5));
            let s = (i as f64 + 0.5) * 1.7321;
            vx.push((s * 5.19).sin() * 0.3);
            vy.push((s * 9.73).cos() * 0.3);
        }
        let mut sim = WasmMdSim {
            n,
            px,
            py,
            vx,
            vy,
            ax: vec![0.0; n],
            ay: vec![0.0; n],
            epsilon: 1.0,
            sigma,
            r_cut: 2.5 * sigma,
            box_size,
            bond_a: vec![],
            bond_b: vec![],
            bond_k: 0.0,
            bond_r0: 0.0,
            time: 0.0,
            dt: 0.0001,
        };
        sim.compute_forces();
        sim
    }

    pub fn polymer() -> WasmMdSim {
        let n = 40;
        let sigma = 0.03;
        let bond_r0 = sigma * 1.5;
        let box_size = 0.8;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        for i in 0..n {
            let x = box_size / 2.0 + (i as f64 * 0.3).sin() * 0.15;
            let y = 0.1 + i as f64 * bond_r0;
            px.push(((x % box_size) + box_size) % box_size);
            py.push(((y % box_size) + box_size) % box_size);
            let s = (i as f64 + 0.5) * std::f64::consts::PI;
            vx[i] = (s * 4.33).sin() * 1.0;
            vy[i] = (s * 7.89).cos() * 1.0;
        }
        let mut bond_a = Vec::new();
        let mut bond_b = Vec::new();
        for i in 0..(n - 1) {
            bond_a.push(i);
            bond_b.push(i + 1);
        }
        let mut sim = WasmMdSim {
            n,
            px,
            py,
            vx,
            vy,
            ax: vec![0.0; n],
            ay: vec![0.0; n],
            epsilon: 1.0,
            sigma,
            r_cut: 2.5 * sigma,
            box_size,
            bond_a,
            bond_b,
            bond_k: 5000.0,
            bond_r0,
            time: 0.0,
            dt: 0.0001,
        };
        sim.compute_forces();
        sim
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            for i in 0..self.n {
                self.px[i] += self.vx[i] * self.dt + 0.5 * self.ax[i] * self.dt * self.dt;
                self.py[i] += self.vy[i] * self.dt + 0.5 * self.ay[i] * self.dt * self.dt;
                self.px[i] = ((self.px[i] % self.box_size) + self.box_size) % self.box_size;
                self.py[i] = ((self.py[i] % self.box_size) + self.box_size) % self.box_size;
            }
            let ax_old = self.ax.clone();
            let ay_old = self.ay.clone();
            self.compute_forces();
            for i in 0..self.n {
                self.vx[i] += 0.5 * (ax_old[i] + self.ax[i]) * self.dt;
                self.vy[i] += 0.5 * (ay_old[i] + self.ay[i]) * self.dt;
            }
            self.time += self.dt;
        }
    }

    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.px[i]);
            out.push(self.py[i]);
        }
        out
    }
    pub fn velocities(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.vx[i]);
            out.push(self.vy[i]);
        }
        out
    }
    pub fn num_particles(&self) -> usize {
        self.n
    }
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn box_size(&self) -> f64 {
        self.box_size
    }
    pub fn temperature(&self) -> f64 {
        let ke: f64 = (0..self.n)
            .map(|i| 0.5 * (self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i]))
            .sum();
        ke / self.n as f64
    }
    pub fn num_bonds(&self) -> usize {
        self.bond_a.len()
    }
    pub fn bond_endpoints(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.bond_a.len() * 4);
        for s in 0..self.bond_a.len() {
            let (i, j) = (self.bond_a[s], self.bond_b[s]);
            out.push(self.px[i]);
            out.push(self.py[i]);
            out.push(self.px[j]);
            out.push(self.py[j]);
        }
        out
    }
}

impl WasmMdSim {
    fn min_img(&self, mut d: f64) -> f64 {
        let h = self.box_size * 0.5;
        if d > h {
            d -= self.box_size;
        } else if d < -h {
            d += self.box_size;
        }
        d
    }
    fn compute_forces(&mut self) {
        for i in 0..self.n {
            self.ax[i] = 0.0;
            self.ay[i] = 0.0;
        }
        let s2 = self.sigma * self.sigma;
        let rc2 = self.r_cut * self.r_cut;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let dx = self.min_img(self.px[j] - self.px[i]);
                let dy = self.min_img(self.py[j] - self.py[i]);
                let r2 = dx * dx + dy * dy;
                if r2 < rc2 && r2 > 1e-20 {
                    let s2r2 = s2 / r2;
                    let s6 = s2r2 * s2r2 * s2r2;
                    let f = 24.0 * self.epsilon * (2.0 * s6 * s6 - s6) / r2;
                    self.ax[i] -= f * dx;
                    self.ay[i] -= f * dy;
                    self.ax[j] += f * dx;
                    self.ay[j] += f * dy;
                }
            }
        }
        for s in 0..self.bond_a.len() {
            let (i, j) = (self.bond_a[s], self.bond_b[s]);
            let dx = self.min_img(self.px[j] - self.px[i]);
            let dy = self.min_img(self.py[j] - self.py[i]);
            let r = (dx * dx + dy * dy).sqrt();
            if r > 1e-10 {
                let f = self.bond_k * (r - self.bond_r0) / r;
                self.ax[i] += f * dx;
                self.ay[i] += f * dy;
                self.ax[j] -= f * dx;
                self.ay[j] -= f * dy;
            }
        }
    }
}

// ================================================================
// Lattice Gauge Theory (Phase 10: phyz-qft)
// ================================================================

#[wasm_bindgen]
pub struct WasmQftSim {
    nx: usize,
    ny: usize,
    links: Vec<f64>,
    beta: f64,
    rng: u64,
    sweeps: usize,
    variant: u8,
    plaq_history: Vec<f64>,
}

#[wasm_bindgen]
impl WasmQftSim {
    pub fn u1_plaquette() -> WasmQftSim {
        let n = 16;
        let mut sim = WasmQftSim {
            nx: n,
            ny: n,
            links: vec![0.0; 2 * n * n],
            beta: 2.0,
            rng: 12345678901234567,
            sweeps: 0,
            variant: 0,
            plaq_history: Vec::new(),
        };
        sim.randomize_links();
        sim
    }

    pub fn wilson_loops() -> WasmQftSim {
        let n = 16;
        let mut sim = WasmQftSim {
            nx: n,
            ny: n,
            links: vec![0.0; 2 * n * n],
            beta: 1.5,
            rng: 98765432109876543,
            sweeps: 0,
            variant: 1,
            plaq_history: Vec::new(),
        };
        sim.randomize_links();
        sim
    }

    pub fn phase_scan() -> WasmQftSim {
        let n = 16;
        let mut sim = WasmQftSim {
            nx: n,
            ny: n,
            links: vec![0.0; 2 * n * n],
            beta: 0.5,
            rng: 55555555555555555,
            sweeps: 0,
            variant: 2,
            plaq_history: Vec::new(),
        };
        sim.randomize_links();
        sim
    }

    pub fn step_n(&mut self, metro_sweeps: usize) {
        for _ in 0..metro_sweeps {
            self.metropolis_sweep();
            self.sweeps += 1;
        }
        let avg = self.average_plaquette();
        self.plaq_history.push(avg);
        if self.variant == 2 && self.sweeps.is_multiple_of(20) {
            self.beta += 0.02;
            if self.beta > 5.0 {
                self.beta = 0.5;
                self.plaq_history.clear();
            }
        }
    }

    pub fn plaquettes(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.nx * self.ny);
        for y in 0..self.ny {
            for x in 0..self.nx {
                out.push(self.plaquette(x, y).cos());
            }
        }
        out
    }

    pub fn average_plaquette(&self) -> f64 {
        let mut sum = 0.0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                sum += self.plaquette(x, y).cos();
            }
        }
        sum / (self.nx * self.ny) as f64
    }

    pub fn wilson_loop(&self, r: usize, t: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let mut angle = 0.0;
                for dx in 0..r {
                    angle += self.lnk(0, (x + dx) % self.nx, y);
                }
                for dy in 0..t {
                    angle += self.lnk(1, (x + r) % self.nx, (y + dy) % self.ny);
                }
                for dx in 0..r {
                    angle -= self.lnk(0, (x + r - 1 - dx) % self.nx, (y + t) % self.ny);
                }
                for dy in 0..t {
                    angle -= self.lnk(1, x, (y + t - 1 - dy) % self.ny);
                }
                sum += angle.cos();
                count += 1;
            }
        }
        sum / count as f64
    }

    pub fn plaq_history(&self) -> Vec<f64> {
        self.plaq_history.clone()
    }
    pub fn lattice_size(&self) -> usize {
        self.nx
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
    pub fn time(&self) -> f64 {
        self.sweeps as f64
    }
}

impl WasmQftSim {
    fn lnk(&self, d: usize, x: usize, y: usize) -> f64 {
        self.links[d * self.ny * self.nx + y * self.nx + x]
    }
    fn set_lnk(&mut self, d: usize, x: usize, y: usize, val: f64) {
        self.links[d * self.ny * self.nx + y * self.nx + x] = val;
    }
    fn plaquette(&self, x: usize, y: usize) -> f64 {
        let xp = (x + 1) % self.nx;
        let yp = (y + 1) % self.ny;
        self.lnk(0, x, y) + self.lnk(1, xp, y) - self.lnk(0, x, yp) - self.lnk(1, x, y)
    }
    fn next_f64(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }
    fn randomize_links(&mut self) {
        let n = self.links.len();
        for i in 0..n {
            self.links[i] = (self.next_f64() - 0.5) * 2.0 * std::f64::consts::PI;
        }
    }
    fn metropolis_sweep(&mut self) {
        let pi = std::f64::consts::PI;
        for d in 0..2 {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let old = self.lnk(d, x, y);
                    let s_old = self.local_action(d, x, y);
                    let proposal = old + (self.next_f64() - 0.5) * 2.0 * pi;
                    self.set_lnk(d, x, y, proposal);
                    let s_new = self.local_action(d, x, y);
                    let ds = s_new - s_old;
                    if ds > 0.0 && self.next_f64() > (-ds).exp() {
                        self.set_lnk(d, x, y, old);
                    }
                }
            }
        }
    }
    fn local_action(&self, d: usize, x: usize, y: usize) -> f64 {
        let mut s = 0.0;
        if d == 0 {
            s += 1.0 - self.plaquette(x, y).cos();
            let ym = (y + self.ny - 1) % self.ny;
            s += 1.0 - self.plaquette(x, ym).cos();
        } else {
            s += 1.0 - self.plaquette(x, y).cos();
            let xm = (x + self.nx - 1) % self.nx;
            s += 1.0 - self.plaquette(xm, y).cos();
        }
        self.beta * s
    }
}

// ================================================================
// Lorentz Force / Charged Particle (Phase 11: phyz-coupling)
// ================================================================

#[wasm_bindgen]
pub struct WasmLorentzSim {
    px: f64,
    py: f64,
    pz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    charge: f64,
    mass: f64,
    bx: f64,
    by: f64,
    bz: f64,
    ex: f64,
    ey: f64,
    ez: f64,
    time: f64,
    dt: f64,
    trail_x: Vec<f64>,
    trail_y: Vec<f64>,
    trail_z: Vec<f64>,
    max_trail: usize,
    variant: u8,
    mirror_length: f64,
}

#[wasm_bindgen]
impl WasmLorentzSim {
    /// Helical spiral in uniform B field along z.
    pub fn spiral() -> WasmLorentzSim {
        WasmLorentzSim {
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.3,
            charge: 1.0,
            mass: 1.0,
            bx: 0.0,
            by: 0.0,
            bz: 2.0,
            ex: 0.0,
            ey: 0.0,
            ez: 0.0,
            time: 0.0,
            dt: 0.01,
            trail_x: Vec::new(),
            trail_y: Vec::new(),
            trail_z: Vec::new(),
            max_trail: 600,
            variant: 0,
            mirror_length: 0.0,
        }
    }

    /// E x B drift: crossed electric and magnetic fields.
    pub fn crossed_fields() -> WasmLorentzSim {
        WasmLorentzSim {
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            vx: 0.5,
            vy: 0.0,
            vz: 0.0,
            charge: 1.0,
            mass: 1.0,
            bx: 0.0,
            by: 0.0,
            bz: 2.0,
            ex: 0.0,
            ey: 0.5,
            ez: 0.0,
            time: 0.0,
            dt: 0.01,
            trail_x: Vec::new(),
            trail_y: Vec::new(),
            trail_z: Vec::new(),
            max_trail: 600,
            variant: 1,
            mirror_length: 0.0,
        }
    }

    /// Magnetic mirror: converging B field lines.
    pub fn magnetic_mirror() -> WasmLorentzSim {
        WasmLorentzSim {
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            vx: 0.8,
            vy: 0.0,
            vz: 0.6,
            charge: 1.0,
            mass: 1.0,
            bx: 0.0,
            by: 0.0,
            bz: 1.0,
            ex: 0.0,
            ey: 0.0,
            ez: 0.0,
            time: 0.0,
            dt: 0.008,
            trail_x: Vec::new(),
            trail_y: Vec::new(),
            trail_z: Vec::new(),
            max_trail: 800,
            variant: 2,
            mirror_length: 4.0,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let qm = self.charge / self.mass;
        for _ in 0..steps {
            // Get local B field (variant 2 has position-dependent B)
            let (lbx, lby, lbz) = if self.variant == 2 {
                // Mirror: B_z increases near ends
                let z_norm = self.pz / self.mirror_length;
                let mirror_factor = 1.0 + 3.0 * z_norm * z_norm;
                // Radial component from div B = 0
                let br = -3.0 * z_norm / self.mirror_length * self.bz;
                let r = (self.px * self.px + self.py * self.py).sqrt().max(1e-10);
                (
                    br * self.px / r,
                    br * self.py / r,
                    self.bz * mirror_factor,
                )
            } else {
                (self.bx, self.by, self.bz)
            };

            // Boris integrator
            let half_dt = self.dt * 0.5;
            // Half-step E acceleration
            self.vx += qm * self.ex * half_dt;
            self.vy += qm * self.ey * half_dt;
            self.vz += qm * self.ez * half_dt;

            // Rotation
            let tx = qm * lbx * half_dt;
            let ty = qm * lby * half_dt;
            let tz = qm * lbz * half_dt;
            let t_mag2 = tx * tx + ty * ty + tz * tz;
            let sx = 2.0 * tx / (1.0 + t_mag2);
            let sy = 2.0 * ty / (1.0 + t_mag2);
            let sz = 2.0 * tz / (1.0 + t_mag2);

            let vpx = self.vx + self.vy * tz - self.vz * ty;
            let vpy = self.vy + self.vz * tx - self.vx * tz;
            let vpz = self.vz + self.vx * ty - self.vy * tx;

            self.vx += vpy * sz - vpz * sy;
            self.vy += vpz * sx - vpx * sz;
            self.vz += vpx * sy - vpy * sx;

            // Half-step E acceleration
            self.vx += qm * self.ex * half_dt;
            self.vy += qm * self.ey * half_dt;
            self.vz += qm * self.ez * half_dt;

            // Position update
            self.px += self.vx * self.dt;
            self.py += self.vy * self.dt;
            self.pz += self.vz * self.dt;

            self.time += self.dt;
        }

        // Record trail point
        self.trail_x.push(self.px);
        self.trail_y.push(self.py);
        self.trail_z.push(self.pz);
        if self.trail_x.len() > self.max_trail {
            self.trail_x.remove(0);
            self.trail_y.remove(0);
            self.trail_z.remove(0);
        }
    }

    pub fn position(&self) -> Vec<f64> {
        vec![self.px, self.py, self.pz]
    }

    pub fn trail(&self) -> Vec<f64> {
        let n = self.trail_x.len();
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.push(self.trail_x[i]);
            out.push(self.trail_y[i]);
            out.push(self.trail_z[i]);
        }
        out
    }

    pub fn trail_len(&self) -> usize {
        self.trail_x.len()
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }
}

// ================================================================
// N-Body Gravity (Phase 12: phyz-gravity)
// ================================================================

#[wasm_bindgen]
#[allow(dead_code)]
pub struct WasmGravitySim {
    n: usize,
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    mass: Vec<f64>,
    softening: f64,
    time: f64,
    dt: f64,
    trail_x: Vec<Vec<f64>>,
    trail_y: Vec<Vec<f64>>,
    max_trail: usize,
    variant: u8,
    g_const: f64,
    gr_strength: f64,
}

#[wasm_bindgen]
impl WasmGravitySim {
    /// Two bodies in mutual orbit.
    pub fn binary_orbit() -> WasmGravitySim {
        let g: f64 = 10.0;
        let m1 = 5.0;
        let m2 = 5.0;
        let sep = 2.0;
        // Circular orbit velocity
        let v = (g * m2 / (sep * 2.0)).sqrt();
        WasmGravitySim {
            n: 2,
            px: vec![-sep / 2.0, sep / 2.0],
            py: vec![0.0, 0.0],
            vx: vec![0.0, 0.0],
            vy: vec![v, -v],
            mass: vec![m1, m2],
            softening: 0.05,
            time: 0.0,
            dt: 0.001,
            trail_x: vec![Vec::new(); 2],
            trail_y: vec![Vec::new(); 2],
            max_trail: 500,
            variant: 0,
            g_const: g,
            gr_strength: 0.0,
        }
    }

    /// 5 planets orbiting a central mass.
    pub fn solar_system() -> WasmGravitySim {
        let g: f64 = 20.0;
        let m_sun = 20.0;
        let n = 6; // sun + 5 planets
        let mut px = vec![0.0; n];
        let mut py = vec![0.0; n];
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        let mut mass = vec![0.0; n];
        mass[0] = m_sun;
        let radii = [0.8, 1.2, 1.8, 2.5, 3.3];
        let planet_mass = [0.1, 0.3, 0.5, 0.2, 0.15];
        for i in 0..5 {
            let r = radii[i];
            let angle = i as f64 * 1.2566; // ~72 degrees apart
            px[i + 1] = r * angle.cos();
            py[i + 1] = r * angle.sin();
            let v = (g * m_sun / r).sqrt();
            vx[i + 1] = -v * angle.sin();
            vy[i + 1] = v * angle.cos();
            mass[i + 1] = planet_mass[i];
        }
        WasmGravitySim {
            n,
            px,
            py,
            vx,
            vy,
            mass,
            softening: 0.05,
            time: 0.0,
            dt: 0.0005,
            trail_x: vec![Vec::new(); n],
            trail_y: vec![Vec::new(); n],
            max_trail: 600,
            variant: 1,
            g_const: g,
            gr_strength: 0.0,
        }
    }

    /// Mercury-like orbit with post-Newtonian precession.
    pub fn precession() -> WasmGravitySim {
        let g: f64 = 20.0;
        let m_sun = 20.0;
        let r = 1.5;
        // Slightly elliptical
        let v = (g * m_sun / r).sqrt() * 0.85;
        WasmGravitySim {
            n: 2,
            px: vec![0.0, r],
            py: vec![0.0, 0.0],
            vx: vec![0.0, 0.0],
            vy: vec![0.0, v],
            mass: vec![m_sun, 0.1],
            softening: 0.05,
            time: 0.0,
            dt: 0.0005,
            trail_x: vec![Vec::new(); 2],
            trail_y: vec![Vec::new(); 2],
            max_trail: 1200,
            variant: 2,
            g_const: g,
            gr_strength: 0.15, // exaggerated for visibility
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let eps2 = self.softening * self.softening;
        for _ in 0..steps {
            // Compute accelerations
            let mut ax = vec![0.0; self.n];
            let mut ay = vec![0.0; self.n];
            for i in 0..self.n {
                for j in (i + 1)..self.n {
                    let dx = self.px[j] - self.px[i];
                    let dy = self.py[j] - self.py[i];
                    let r2 = dx * dx + dy * dy + eps2;
                    let r = r2.sqrt();
                    let r3 = r * r2;
                    let mut f = self.g_const / r3;

                    // Post-Newtonian correction for precession demo
                    if self.gr_strength > 0.0 {
                        f *= 1.0 + self.gr_strength / r2;
                    }

                    ax[i] += f * self.mass[j] * dx;
                    ay[i] += f * self.mass[j] * dy;
                    ax[j] -= f * self.mass[i] * dx;
                    ay[j] -= f * self.mass[i] * dy;
                }
            }

            // Leapfrog integration
            for i in 0..self.n {
                self.vx[i] += ax[i] * self.dt;
                self.vy[i] += ay[i] * self.dt;
                self.px[i] += self.vx[i] * self.dt;
                self.py[i] += self.vy[i] * self.dt;
            }
            self.time += self.dt;
        }

        // Record trails
        for i in 0..self.n {
            self.trail_x[i].push(self.px[i]);
            self.trail_y[i].push(self.py[i]);
            if self.trail_x[i].len() > self.max_trail {
                self.trail_x[i].remove(0);
                self.trail_y[i].remove(0);
            }
        }
    }

    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.px[i]);
            out.push(self.py[i]);
        }
        out
    }

    pub fn masses(&self) -> Vec<f64> {
        self.mass.clone()
    }

    pub fn trail_for(&self, idx: usize) -> Vec<f64> {
        let n = self.trail_x[idx].len();
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.push(self.trail_x[idx][i]);
            out.push(self.trail_y[idx][i]);
        }
        out
    }

    pub fn trail_len(&self, idx: usize) -> usize {
        self.trail_x[idx].len()
    }

    pub fn num_bodies(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.time
    }
}

// ================================================================
// Guardian / Conservation Monitor (Phase 13: phyz-guardian)
// ================================================================

#[wasm_bindgen]
#[allow(dead_code)]
pub struct WasmGuardianSim {
    q: f64,
    v: f64,
    dt: f64,
    time: f64,
    length: f64,
    ke_history: Vec<f64>,
    pe_history: Vec<f64>,
    total_history: Vec<f64>,
    max_history: usize,
    variant: u8,
    // For adaptive dt demo
    q2: f64,
    v2: f64,
    dt_adaptive: f64,
    total_history2: Vec<f64>,
    // For correction demo
    correction_on: bool,
    e0: f64,
    injection_rate: f64,
}

#[wasm_bindgen]
impl WasmGuardianSim {
    /// Pendulum with energy monitoring gauge.
    pub fn energy_monitor() -> WasmGuardianSim {
        let q: f64 = 2.0;
        let length = 1.5;
        let pe = GRAVITY / length * (1.0 - q.cos());
        WasmGuardianSim {
            q,
            v: 0.0,
            dt: 0.002,
            time: 0.0,
            length,
            ke_history: Vec::new(),
            pe_history: Vec::new(),
            total_history: Vec::new(),
            max_history: 200,
            variant: 0,
            q2: 0.0,
            v2: 0.0,
            dt_adaptive: 0.0,
            total_history2: Vec::new(),
            correction_on: false,
            e0: 0.5 * 0.0 * 0.0 + pe,
            injection_rate: 0.0,
        }
    }

    /// Fixed vs adaptive timestep comparison.
    pub fn adaptive_dt() -> WasmGuardianSim {
        let q: f64 = 2.5;
        let length = 1.5;
        let pe = GRAVITY / length * (1.0 - q.cos());
        WasmGuardianSim {
            q,
            v: 0.0,
            dt: 0.015, // intentionally large fixed dt
            time: 0.0,
            length,
            ke_history: Vec::new(),
            pe_history: Vec::new(),
            total_history: Vec::new(),
            max_history: 200,
            variant: 1,
            q2: q,
            v2: 0.0,
            dt_adaptive: 0.01,
            total_history2: Vec::new(),
            correction_on: false,
            e0: pe,
            injection_rate: 0.0,
        }
    }

    /// Correction demo with energy injection and guardian fix.
    pub fn correction_demo() -> WasmGuardianSim {
        let q: f64 = 1.5;
        let length = 1.5;
        let pe = GRAVITY / length * (1.0 - q.cos());
        WasmGuardianSim {
            q,
            v: 0.0,
            dt: 0.003,
            time: 0.0,
            length,
            ke_history: Vec::new(),
            pe_history: Vec::new(),
            total_history: Vec::new(),
            max_history: 200,
            variant: 2,
            q2: 0.0,
            v2: 0.0,
            dt_adaptive: 0.0,
            total_history2: Vec::new(),
            correction_on: true,
            e0: pe,
            injection_rate: 0.005,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let omega2 = GRAVITY / self.length;
        match self.variant {
            0 => {
                // Simple symplectic Euler for energy monitor
                for _ in 0..steps {
                    let acc = -omega2 * self.q.sin();
                    self.v += acc * self.dt;
                    self.q += self.v * self.dt;
                    self.time += self.dt;
                }
            }
            1 => {
                // Fixed dt (Euler, drifts)
                for _ in 0..steps {
                    let acc = -omega2 * self.q.sin();
                    self.v += acc * self.dt;
                    self.q += self.v * self.dt;

                    // Adaptive dt (symplectic Euler with smaller step)
                    let sub = 4;
                    let adt = self.dt / sub as f64;
                    for _ in 0..sub {
                        let acc2 = -omega2 * self.q2.sin();
                        self.v2 += acc2 * adt;
                        self.q2 += self.v2 * adt;
                    }
                    self.time += self.dt;
                }
            }
            2 => {
                // With deliberate energy injection + guardian correction
                for _ in 0..steps {
                    let acc = -omega2 * self.q.sin();
                    self.v += acc * self.dt;
                    // Inject energy
                    self.v += self.injection_rate;
                    self.q += self.v * self.dt;

                    // Guardian correction: rescale velocity to maintain E0
                    if self.correction_on {
                        let pe = omega2 * (1.0 - self.q.cos());
                        let target_ke = (self.e0 - pe).max(0.0);
                        let current_ke = 0.5 * self.v * self.v;
                        if current_ke > 1e-10 {
                            let scale = (target_ke / current_ke).sqrt();
                            self.v *= scale;
                        }
                    }
                    self.time += self.dt;
                }
            }
            _ => {}
        }

        // Record energy history
        let omega2 = GRAVITY / self.length;
        let ke = 0.5 * self.v * self.v;
        let pe = omega2 * (1.0 - self.q.cos());
        self.ke_history.push(ke);
        self.pe_history.push(pe);
        self.total_history.push(ke + pe);
        if self.ke_history.len() > self.max_history {
            self.ke_history.remove(0);
            self.pe_history.remove(0);
            self.total_history.remove(0);
        }

        if self.variant == 1 {
            let ke2 = 0.5 * self.v2 * self.v2;
            let pe2 = omega2 * (1.0 - self.q2.cos());
            self.total_history2.push(ke2 + pe2);
            if self.total_history2.len() > self.max_history {
                self.total_history2.remove(0);
            }
        }
    }

    pub fn q_val(&self) -> f64 {
        self.q
    }
    pub fn v_val(&self) -> f64 {
        self.v
    }
    pub fn q2_val(&self) -> f64 {
        self.q2
    }

    pub fn ke(&self) -> f64 {
        0.5 * self.v * self.v
    }
    pub fn pe(&self) -> f64 {
        (GRAVITY / self.length) * (1.0 - self.q.cos())
    }
    pub fn total_energy(&self) -> f64 {
        self.ke() + self.pe()
    }

    pub fn ke_history(&self) -> Vec<f64> {
        self.ke_history.clone()
    }
    pub fn pe_history(&self) -> Vec<f64> {
        self.pe_history.clone()
    }
    pub fn total_history(&self) -> Vec<f64> {
        self.total_history.clone()
    }
    pub fn total_history2(&self) -> Vec<f64> {
        self.total_history2.clone()
    }
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn e0(&self) -> f64 {
        self.e0
    }
    pub fn length(&self) -> f64 {
        self.length
    }
}

// ================================================================
// Lattice Boltzmann Method (Phase 14: phyz-lbm)
// ================================================================

#[wasm_bindgen]
pub struct WasmLbmSim {
    nx: usize,
    ny: usize,
    f: Vec<f64>,
    tau: f64,
    time: f64,
    variant: u8,
    obstacle: Vec<bool>,
}

// D2Q9 directions: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)
const D2Q9_CX: [i32; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const D2Q9_CY: [i32; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const D2Q9_W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
const D2Q9_OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

#[wasm_bindgen]
impl WasmLbmSim {
    /// Lid-driven cavity flow.
    pub fn cavity_flow() -> WasmLbmSim {
        let nx = 80;
        let ny = 80;
        let n9 = nx * ny * 9;
        let mut f = vec![0.0; n9];
        // Initialize to equilibrium at rest
        for y in 0..ny {
            for x in 0..nx {
                for k in 0..9 {
                    f[(y * nx + x) * 9 + k] = D2Q9_W[k];
                }
            }
        }
        WasmLbmSim {
            nx,
            ny,
            f,
            tau: 0.56,
            time: 0.0,
            variant: 0,
            obstacle: vec![false; nx * ny],
        }
    }

    /// Poiseuille channel flow.
    pub fn channel_flow() -> WasmLbmSim {
        let nx = 120;
        let ny = 40;
        let n9 = nx * ny * 9;
        let mut f = vec![0.0; n9];
        for y in 0..ny {
            for x in 0..nx {
                for k in 0..9 {
                    f[(y * nx + x) * 9 + k] = D2Q9_W[k];
                }
            }
        }
        WasmLbmSim {
            nx,
            ny,
            f,
            tau: 0.8,
            time: 0.0,
            variant: 1,
            obstacle: vec![false; nx * ny],
        }
    }

    /// Flow around obstacle (von Karman vortex street).
    pub fn vortex_street() -> WasmLbmSim {
        let nx = 160;
        let ny = 60;
        let n9 = nx * ny * 9;
        let mut f = vec![0.0; n9];
        let mut obstacle = vec![false; nx * ny];
        // Circular obstacle
        let cx = nx / 4;
        let cy = ny / 2;
        let r = ny as f64 / 8.0;
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                let dx = x as f64 - cx as f64;
                let dy = y as f64 - cy as f64;
                if dx * dx + dy * dy < r * r {
                    obstacle[idx] = true;
                }
                // Initialize with slight rightward flow
                let ux = 0.04_f64;
                for k in 0..9 {
                    let cu = D2Q9_CX[k] as f64 * ux;
                    f[idx * 9 + k] = D2Q9_W[k] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * ux * ux);
                }
            }
        }
        WasmLbmSim {
            nx,
            ny,
            f,
            tau: 0.52,
            time: 0.0,
            variant: 2,
            obstacle,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        let (nx, ny) = (self.nx, self.ny);
        let omega = 1.0 / self.tau;

        for _ in 0..steps {
            // --- Collision ---
            let mut f_new = self.f.clone();
            for y in 0..ny {
                for x in 0..nx {
                    let idx = y * nx + x;
                    if self.obstacle[idx] {
                        continue;
                    }
                    // Compute macroscopic quantities
                    let mut rho = 0.0;
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    for k in 0..9 {
                        let fi = self.f[idx * 9 + k];
                        rho += fi;
                        ux += fi * D2Q9_CX[k] as f64;
                        uy += fi * D2Q9_CY[k] as f64;
                    }
                    if rho > 1e-10 {
                        ux /= rho;
                        uy /= rho;
                    }

                    // Body force for channel flow
                    if self.variant == 1 {
                        ux += 0.0001;
                    }

                    // BGK collision
                    let usq = ux * ux + uy * uy;
                    for k in 0..9 {
                        let cu = D2Q9_CX[k] as f64 * ux + D2Q9_CY[k] as f64 * uy;
                        let feq = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                        f_new[idx * 9 + k] =
                            self.f[idx * 9 + k] * (1.0 - omega) + feq * omega;
                    }
                }
            }

            // --- Streaming ---
            let mut f_streamed = vec![0.0; nx * ny * 9];
            for y in 0..ny {
                for x in 0..nx {
                    let idx = y * nx + x;
                    for k in 0..9 {
                        let nx2 = (x as i32 + D2Q9_CX[k] + nx as i32) as usize % nx;
                        let ny2 = (y as i32 + D2Q9_CY[k] + ny as i32) as usize % ny;
                        let dst = ny2 * nx + nx2;
                        f_streamed[dst * 9 + k] = f_new[idx * 9 + k];
                    }
                }
            }

            // --- Bounce-back on obstacles ---
            for y in 0..ny {
                for x in 0..nx {
                    let idx = y * nx + x;
                    if self.obstacle[idx] {
                        for k in 0..9 {
                            f_streamed[idx * 9 + k] = f_new[idx * 9 + D2Q9_OPP[k]];
                        }
                    }
                }
            }

            // --- Boundary conditions ---
            match self.variant {
                0 => {
                    // Cavity: top wall moves right
                    let u_lid = 0.08;
                    let y = ny - 1;
                    for x in 0..nx {
                        let idx = y * nx + x;
                        let mut rho = 0.0;
                        for k in 0..9 {
                            rho += f_streamed[idx * 9 + k];
                        }
                        // Zou-He top wall
                        for k in 0..9 {
                            let cu = D2Q9_CX[k] as f64 * u_lid;
                            let feq = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_lid * u_lid);
                            f_streamed[idx * 9 + k] = feq;
                        }
                    }
                    // No-slip on other walls
                    for x in 0..nx {
                        let idx_bot = x;
                        for k in 0..9 {
                            let feq = D2Q9_W[k];
                            f_streamed[idx_bot * 9 + k] = feq;
                        }
                    }
                    for y in 0..ny {
                        let idx_l = y * nx;
                        let idx_r = y * nx + nx - 1;
                        for k in 0..9 {
                            f_streamed[idx_l * 9 + k] = D2Q9_W[k];
                            f_streamed[idx_r * 9 + k] = D2Q9_W[k];
                        }
                    }
                }
                1 => {
                    // Channel: no-slip top/bottom walls
                    for x in 0..nx {
                        for k in 0..9 {
                            f_streamed[x * 9 + k] = D2Q9_W[k]; // bottom
                            f_streamed[((ny - 1) * nx + x) * 9 + k] = D2Q9_W[k]; // top
                        }
                    }
                }
                2 => {
                    // Vortex street: inlet/outlet
                    let u_in = 0.04;
                    for y in 1..(ny - 1) {
                        // Inlet
                        let idx = y * nx;
                        let rho = 1.0;
                        for k in 0..9 {
                            let cu = D2Q9_CX[k] as f64 * u_in;
                            f_streamed[idx * 9 + k] = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_in * u_in);
                        }
                        // Outlet: extrapolate
                        let idx_out = y * nx + nx - 1;
                        let idx_prev = y * nx + nx - 2;
                        for k in 0..9 {
                            f_streamed[idx_out * 9 + k] = f_streamed[idx_prev * 9 + k];
                        }
                    }
                    // No-slip top/bottom
                    for x in 0..nx {
                        for k in 0..9 {
                            f_streamed[x * 9 + k] = D2Q9_W[k];
                            f_streamed[((ny - 1) * nx + x) * 9 + k] = D2Q9_W[k];
                        }
                    }
                }
                _ => {}
            }

            self.f = f_streamed;
            self.time += 1.0;
        }
    }

    /// Velocity magnitude field as flat array [nx*ny].
    pub fn velocity_field(&self) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let mut out = vec![0.0; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                if self.obstacle[idx] {
                    out[idx] = -1.0; // sentinel for obstacle
                    continue;
                }
                let mut rho = 0.0;
                let mut ux = 0.0;
                let mut uy = 0.0;
                for k in 0..9 {
                    let fi = self.f[idx * 9 + k];
                    rho += fi;
                    ux += fi * D2Q9_CX[k] as f64;
                    uy += fi * D2Q9_CY[k] as f64;
                }
                if rho > 1e-10 {
                    ux /= rho;
                    uy /= rho;
                }
                out[idx] = (ux * ux + uy * uy).sqrt();
            }
        }
        out
    }

    /// Vorticity field (curl of velocity).
    pub fn vorticity_field(&self) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        // First compute velocity components
        let mut ux_field = vec![0.0; nx * ny];
        let mut uy_field = vec![0.0; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                let mut rho = 0.0;
                let mut ux = 0.0;
                let mut uy = 0.0;
                for k in 0..9 {
                    let fi = self.f[idx * 9 + k];
                    rho += fi;
                    ux += fi * D2Q9_CX[k] as f64;
                    uy += fi * D2Q9_CY[k] as f64;
                }
                if rho > 1e-10 {
                    ux /= rho;
                    uy /= rho;
                }
                ux_field[idx] = ux;
                uy_field[idx] = uy;
            }
        }
        // Compute curl
        let mut vort = vec![0.0; nx * ny];
        for y in 1..(ny - 1) {
            for x in 1..(nx - 1) {
                let idx = y * nx + x;
                let duy_dx = (uy_field[idx + 1] - uy_field[idx - 1]) * 0.5;
                let dux_dy = (ux_field[idx + nx] - ux_field[idx - nx]) * 0.5;
                vort[idx] = duy_dx - dux_dy;
            }
        }
        vort
    }

    pub fn grid_nx(&self) -> usize {
        self.nx
    }
    pub fn grid_ny(&self) -> usize {
        self.ny
    }
    pub fn time(&self) -> f64 {
        self.time
    }
}

// ================================================================
// Probabilistic Simulation (Phase 16: phyz-prob)
// ================================================================

#[wasm_bindgen]
#[allow(dead_code)]
pub struct WasmProbSim {
    n: usize,
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    time: f64,
    dt: f64,
    variant: u8,
    rng: u64,
    trail_x: Vec<Vec<f64>>,
    trail_y: Vec<Vec<f64>>,
    max_trail: usize,
}

#[wasm_bindgen]
impl WasmProbSim {
    /// Ensemble trajectories diverging from near-identical ICs.
    /// No gravity — pure spreading to show uncertainty cone.
    pub fn uncertainty_cone() -> WasmProbSim {
        let n = 60;
        let mut px = vec![0.0; n];
        let mut py = vec![0.0; n];
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        let base_v = 1.0;
        let base_angle = 0.0; // rightward
        for i in 0..n {
            // Small angular perturbation creates the cone
            let eps = (i as f64 - n as f64 / 2.0) * 0.008;
            let angle = base_angle + eps;
            // Small speed perturbation too
            let dv = (i as f64 - n as f64 / 2.0) * 0.003;
            vx.push((base_v + dv) * angle.cos());
            vy.push((base_v + dv) * angle.sin());
            px[i] = 0.0;
            py[i] = 0.0;
        }
        WasmProbSim {
            n,
            px,
            py,
            vx,
            vy,
            time: 0.0,
            dt: 0.008,
            variant: 0,
            rng: 42,
            trail_x: vec![Vec::new(); n],
            trail_y: vec![Vec::new(); n],
            max_trail: 400,
        }
    }

    /// SVGD particles converging to a target distribution (2D Gaussian mixture).
    pub fn svgd_particles() -> WasmProbSim {
        let n = 80;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        // Initialize in a grid pattern
        let side = (n as f64).sqrt() as usize;
        let mut rng: u64 = 123456789;
        for i in 0..n {
            let row = i / side;
            let col = i % side;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise_x = ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.3;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise_y = ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.3;
            px.push((col as f64 - side as f64 / 2.0) * 0.4 + noise_x);
            py.push((row as f64 - side as f64 / 2.0) * 0.4 + noise_y);
        }
        WasmProbSim {
            n,
            px,
            py,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            time: 0.0,
            dt: 0.02,
            variant: 1,
            rng,
            trail_x: vec![Vec::new(); n],
            trail_y: vec![Vec::new(); n],
            max_trail: 100,
        }
    }

    /// Ensemble of bouncing balls with parameter uncertainty.
    pub fn monte_carlo() -> WasmProbSim {
        let n = 50;
        let mut px = vec![0.0; n];
        let py = vec![1.5; n];
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        for i in 0..n {
            // Wide spread of launch angles and speeds
            let t = (i as f64 / (n - 1) as f64) - 0.5; // -0.5 to 0.5
            let angle = 0.4 + t * 1.4; // ~-0.3 to ~1.1 rad
            let speed = 3.0 + t.abs() * 2.0;
            vx.push(speed * angle.cos());
            vy.push(speed * angle.sin());
            px[i] = t * 0.1;
        }
        WasmProbSim {
            n,
            px,
            py,
            vx,
            vy,
            time: 0.0,
            dt: 0.003,
            variant: 2,
            rng: 9876543,
            trail_x: vec![Vec::new(); n],
            trail_y: vec![Vec::new(); n],
            max_trail: 400,
        }
    }

    pub fn step_n(&mut self, steps: usize) {
        match self.variant {
            0 => {
                // Free-streaming particles with slight nonlinear perturbation.
                // Periodically reset to show the spreading cone repeatedly.
                for _ in 0..steps {
                    for i in 0..self.n {
                        // Gentle nonlinear coupling to amplify divergence
                        let r2 = self.px[i] * self.px[i] + self.py[i] * self.py[i];
                        let bend = 0.0005 * r2;
                        self.vy[i] += bend * self.vx[i].signum() * self.dt;
                        self.px[i] += self.vx[i] * self.dt;
                        self.py[i] += self.vy[i] * self.dt;
                    }
                    self.time += self.dt;
                }
                // Reset when particles spread too far (cycle the demo)
                let max_x = self.px.iter().cloned().fold(0.0_f64, f64::max);
                if max_x > 5.0 {
                    let base_v = 1.0;
                    for i in 0..self.n {
                        let eps = (i as f64 - self.n as f64 / 2.0) * 0.008;
                        let dv = (i as f64 - self.n as f64 / 2.0) * 0.003;
                        self.px[i] = 0.0;
                        self.py[i] = 0.0;
                        self.vx[i] = (base_v + dv) * eps.cos();
                        self.vy[i] = (base_v + dv) * eps.sin();
                    }
                    for t in &mut self.trail_x { t.clear(); }
                    for t in &mut self.trail_y { t.clear(); }
                }
            }
            1 => {
                // SVGD: gradient descent toward Gaussian mixture + repulsion
                // Target: two Gaussians at (-1, 0) and (1, 0.5)
                let centers = [(-1.0, 0.0), (1.0, 0.5)];
                let sigma2 = 0.3;
                let h = 0.5; // kernel bandwidth

                for _ in 0..steps {
                    let mut grad_x = vec![0.0; self.n];
                    let mut grad_y = vec![0.0; self.n];

                    for i in 0..self.n {
                        // Gradient of log target
                        let mut dlp_x = 0.0;
                        let mut dlp_y = 0.0;
                        for &(cx, cy) in &centers {
                            let dx = self.px[i] - cx;
                            let dy = self.py[i] - cy;
                            let w = (-0.5 * (dx * dx + dy * dy) / sigma2).exp();
                            dlp_x -= w * dx / sigma2;
                            dlp_y -= w * dy / sigma2;
                        }

                        // SVGD kernel interactions
                        for j in 0..self.n {
                            let dx = self.px[j] - self.px[i];
                            let dy = self.py[j] - self.py[i];
                            let k = (-0.5 * (dx * dx + dy * dy) / (h * h)).exp() / self.n as f64;
                            // Kernel * grad_log_p + grad_kernel
                            grad_x[j] += k * dlp_x + k * dx / (h * h);
                            grad_y[j] += k * dlp_y + k * dy / (h * h);
                        }
                    }

                    for i in 0..self.n {
                        self.px[i] += grad_x[i] * self.dt;
                        self.py[i] += grad_y[i] * self.dt;
                    }
                    self.time += self.dt;
                }
            }
            2 => {
                // Bouncing balls with gravity — elastic enough to stay lively
                for _ in 0..steps {
                    for i in 0..self.n {
                        self.vy[i] -= GRAVITY * self.dt;
                        self.px[i] += self.vx[i] * self.dt;
                        self.py[i] += self.vy[i] * self.dt;
                        // Bounce off ground (elastic)
                        if self.py[i] < 0.0 {
                            self.py[i] = -self.py[i];
                            self.vy[i] = -self.vy[i] * 0.85;
                            self.vx[i] *= 0.98;
                        }
                        // Side walls
                        if self.px[i] < -4.0 {
                            self.px[i] = -4.0;
                            self.vx[i] = self.vx[i].abs() * 0.7;
                        }
                        if self.px[i] > 4.0 {
                            self.px[i] = 4.0;
                            self.vx[i] = -self.vx[i].abs() * 0.7;
                        }
                    }
                    self.time += self.dt;
                }
            }
            _ => {}
        }

        // Record trails
        for i in 0..self.n {
            self.trail_x[i].push(self.px[i]);
            self.trail_y[i].push(self.py[i]);
            if self.trail_x[i].len() > self.max_trail {
                self.trail_x[i].remove(0);
                self.trail_y[i].remove(0);
            }
        }
    }

    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            out.push(self.px[i]);
            out.push(self.py[i]);
        }
        out
    }

    pub fn trail_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..self.n {
            let n = self.trail_x[i].len();
            out.push(n as f64);
            for j in 0..n {
                out.push(self.trail_x[i][j]);
                out.push(self.trail_y[i][j]);
            }
        }
        out
    }

    pub fn num_particles(&self) -> usize {
        self.n
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn spread(&self) -> f64 {
        let mut cx = 0.0;
        let mut cy = 0.0;
        for i in 0..self.n {
            cx += self.px[i];
            cy += self.py[i];
        }
        cx /= self.n as f64;
        cy /= self.n as f64;
        let mut var = 0.0;
        for i in 0..self.n {
            let dx = self.px[i] - cx;
            let dy = self.py[i] - cy;
            var += dx * dx + dy * dy;
        }
        (var / self.n as f64).sqrt()
    }
}

// ================================================================
// phyz-diff: Analytical Jacobians
// ================================================================

/// Gradient descent optimization: find initial angle to hit target x position.
#[wasm_bindgen]
pub struct WasmDiffGradientSim {
    model: Model,
    state: State,
    target_x: f64,
    theta0: f64,
    lr: f64,
    loss_history: Vec<f64>,
    iteration: usize,
    length: f64,
    // Current simulation state for rendering
    render_state: State,
}

#[wasm_bindgen]
impl WasmDiffGradientSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmDiffGradientSim {
        let length = 1.5;
        let mass = 1.0;
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum", -1, SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0, 0.0, mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let theta0 = 0.2; // initial guess
        let mut state = model.default_state();
        state.q[0] = theta0;

        let render_state = state.clone();
        WasmDiffGradientSim {
            model, state, target_x: 1.2, theta0, lr: 0.005,
            loss_history: vec![], iteration: 0, length, render_state,
        }
    }

    /// Run one gradient descent step: simulate forward, compute loss gradient, update theta0.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            // Reset state with current theta0
            self.state = self.model.default_state();
            self.state.q[0] = self.theta0;

            // Simulate 500 steps forward
            let dt = self.model.dt;
            for _ in 0..500 {
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

            // Compute loss: (x_final - target_x)^2
            let x_final = self.length * self.state.q[0].sin();
            let loss = (x_final - self.target_x).powi(2);
            self.loss_history.push(loss);

            // Gradient via finite diff on loss w.r.t. theta0
            let eps = 1e-5;
            let mut sp = self.model.default_state();
            sp.q[0] = self.theta0 + eps;
            for _ in 0..500 {
                let acc = aba(&self.model, &sp);
                sp.v = &sp.v + &acc * dt;
                sp.q = &sp.q + &sp.v * dt;
            }
            let xp = self.length * sp.q[0].sin();
            let loss_p = (xp - self.target_x).powi(2);

            let mut sm = self.model.default_state();
            sm.q[0] = self.theta0 - eps;
            for _ in 0..500 {
                let acc = aba(&self.model, &sm);
                sm.v = &sm.v + &acc * dt;
                sm.q = &sm.q + &sm.v * dt;
            }
            let xm = self.length * sm.q[0].sin();
            let loss_m = (xm - self.target_x).powi(2);

            let grad = (loss_p - loss_m) / (2.0 * eps);
            self.theta0 -= self.lr * grad;
            self.theta0 = self.theta0.clamp(-std::f64::consts::PI, std::f64::consts::PI);
            self.iteration += 1;
        }

        // Update render state: simulate current theta0 with animation
        self.render_state = self.model.default_state();
        self.render_state.q[0] = self.theta0;
        let dt = self.model.dt;
        for _ in 0..200 {
            let acc = aba(&self.model, &self.render_state);
            self.render_state.v = &self.render_state.v + &acc * dt;
            self.render_state.q = &self.render_state.q + &self.render_state.v * dt;
            self.render_state.time += dt;
        }
    }

    pub fn theta0(&self) -> f64 { self.theta0 }
    pub fn target_x(&self) -> f64 { self.target_x }
    pub fn loss_history(&self) -> Vec<f64> { self.loss_history.clone() }
    pub fn iteration(&self) -> usize { self.iteration }
    pub fn current_x(&self) -> f64 { self.length * self.render_state.q[0].sin() }
    pub fn current_y(&self) -> f64 { -self.length * self.render_state.q[0].cos() }
    pub fn time(&self) -> f64 { self.render_state.time }
    pub fn current_loss(&self) -> f64 {
        self.loss_history.last().copied().unwrap_or(f64::MAX)
    }
}

/// Jacobian heatmap: pendulum simulation with live Jacobian matrix display.
#[wasm_bindgen]
pub struct WasmDiffJacobianSim {
    model: Model,
    state: State,
    length: f64,
    // Jacobian matrix entries (2x2 for single DOF: dq'/dq, dq'/dv, dv'/dq, dv'/dv)
    jac: [f64; 4],
}

#[wasm_bindgen]
impl WasmDiffJacobianSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmDiffJacobianSim {
        let length = 1.5;
        let mass = 1.0;
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum", -1, SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0, 0.0, mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = 0.8;
        WasmDiffJacobianSim { model, state, length, jac: [1.0, 0.0, 0.0, 1.0] }
    }

    pub fn step_n(&mut self, n: usize) {
        let dt = self.model.dt;
        for _ in 0..n {
            // Compute Jacobians
            let j = analytical_step_jacobians(&self.model, &self.state);
            self.jac = [j.dqnext_dq[(0, 0)], j.dqnext_dv[(0, 0)], j.dvnext_dq[(0, 0)], j.dvnext_dv[(0, 0)]];

            // Semi-implicit Euler step
            let acc = aba(&self.model, &self.state);
            self.state.v = &self.state.v + &acc * dt;
            self.state.q = &self.state.q + &self.state.v * dt;
            self.state.time += dt;
        }
    }

    pub fn angle(&self) -> f64 { self.state.q[0] }
    pub fn bob_x(&self) -> f64 { self.length * self.state.q[0].sin() }
    pub fn bob_y(&self) -> f64 { -self.length * self.state.q[0].cos() }
    pub fn jacobian(&self) -> Vec<f64> { self.jac.to_vec() }
    pub fn time(&self) -> f64 { self.state.time }
}

/// Sensitivity: two pendulums with perturbed ICs, analytical gradient prediction vs actual divergence.
#[wasm_bindgen]
pub struct WasmDiffSensitivitySim {
    model: Model,
    state_a: State,
    state_b: State,
    length: f64,
    predicted_divergence: Vec<f64>, // accumulated from Jacobians
    actual_divergence: Vec<f64>,
}

#[wasm_bindgen]
impl WasmDiffSensitivitySim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmDiffSensitivitySim {
        let length = 1.5;
        let mass = 1.0;
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum", -1, SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0, 0.0, mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let mut state_a = model.default_state();
        state_a.q[0] = 0.8;
        let mut state_b = model.default_state();
        state_b.q[0] = 0.8 + 0.01; // small perturbation

        WasmDiffSensitivitySim {
            model, state_a, state_b, length,
            predicted_divergence: vec![0.01], actual_divergence: vec![0.01],
        }
    }

    pub fn step_n(&mut self, n: usize) {
        let dt = self.model.dt;
        for _ in 0..n {
            // Compute Jacobian at state_a
            let j = analytical_step_jacobians(&self.model, &self.state_a);

            // Update predicted divergence using Jacobian
            let prev_pred = self.predicted_divergence.last().copied().unwrap_or(0.01);
            let growth_factor = j.dqnext_dq[(0, 0)].abs();
            let new_pred = prev_pred * growth_factor;
            self.predicted_divergence.push(new_pred);

            // Step both sims
            let acc_a = aba(&self.model, &self.state_a);
            self.state_a.v = &self.state_a.v + &acc_a * dt;
            self.state_a.q = &self.state_a.q + &self.state_a.v * dt;
            self.state_a.time += dt;

            let acc_b = aba(&self.model, &self.state_b);
            self.state_b.v = &self.state_b.v + &acc_b * dt;
            self.state_b.q = &self.state_b.q + &self.state_b.v * dt;
            self.state_b.time += dt;

            // Actual divergence
            let actual = (self.state_a.q[0] - self.state_b.q[0]).abs();
            self.actual_divergence.push(actual);
        }

        // Keep history bounded
        if self.predicted_divergence.len() > 600 {
            self.predicted_divergence.drain(0..self.predicted_divergence.len() - 600);
        }
        if self.actual_divergence.len() > 600 {
            self.actual_divergence.drain(0..self.actual_divergence.len() - 600);
        }
    }

    pub fn pos_a(&self) -> Vec<f64> {
        vec![self.length * self.state_a.q[0].sin(), -self.length * self.state_a.q[0].cos()]
    }
    pub fn pos_b(&self) -> Vec<f64> {
        vec![self.length * self.state_b.q[0].sin(), -self.length * self.state_b.q[0].cos()]
    }
    pub fn predicted_divergence(&self) -> Vec<f64> { self.predicted_divergence.clone() }
    pub fn actual_divergence(&self) -> Vec<f64> { self.actual_divergence.clone() }
    pub fn time(&self) -> f64 { self.state_a.time }
}

// ================================================================
// phyz-mjcf: MuJoCo XML Loading (inline, demonstrates concept)
// ================================================================

/// Ant model: multi-body spider with 8 legs.
#[wasm_bindgen]
pub struct WasmMjcfAntSim {
    model: Model,
    state: State,
}

#[wasm_bindgen]
impl WasmMjcfAntSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmMjcfAntSim {
        // Build ant-like kinematic tree: torso + 4 legs with 2 joints each
        let _torso_mass = 2.0;
        let leg_mass = 0.3;
        let hip_len = 0.4;
        let shin_len = 0.6;

        let mut builder = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001);

        // 4 legs at 45°/135°/225°/315° around torso, each has hip + shin
        let offsets = [
            Vec3::new(0.4, 0.0, 0.4),
            Vec3::new(-0.4, 0.0, 0.4),
            Vec3::new(-0.4, 0.0, -0.4),
            Vec3::new(0.4, 0.0, -0.4),
        ];
        // Rotate each hip frame around Y so leg swings outward from torso
        let azimuths = [
            std::f64::consts::FRAC_PI_4,       // +x, +z → 45°
            -std::f64::consts::FRAC_PI_4,      // -x, +z → -45°  (actually 135° but swing plane rotated)
            std::f64::consts::FRAC_PI_4,       // -x, -z → mirror of leg 0
            -std::f64::consts::FRAC_PI_4,      // +x, -z → mirror of leg 1
        ];

        for (i, (offset, &azimuth)) in offsets.iter().zip(azimuths.iter()).enumerate() {
            // Compose Y-rotation with translation so each leg's revolute Z-axis faces outward
            let mut hip_xf = SpatialTransform::rot_y(azimuth);
            hip_xf.pos = *offset;

            let hip_parent = -1i32; // all hips attached to world (torso)
            builder = builder.add_revolute_body(
                &format!("hip_{i}"), hip_parent,
                hip_xf,
                SpatialInertia::new(
                    leg_mass,
                    Vec3::new(0.0, -hip_len / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        leg_mass * hip_len * hip_len / 12.0, 0.0, leg_mass * hip_len * hip_len / 12.0,
                    )),
                ),
            );
            builder = builder.add_revolute_body(
                &format!("shin_{i}"), (i * 2) as i32,
                SpatialTransform::translation(Vec3::new(0.0, -hip_len, 0.0)),
                SpatialInertia::new(
                    leg_mass,
                    Vec3::new(0.0, -shin_len / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        leg_mass * shin_len * shin_len / 12.0, 0.0, leg_mass * shin_len * shin_len / 12.0,
                    )),
                ),
            );
        }

        let mut model = builder.build();
        for j in &mut model.joints {
            j.damping = 0.1;
        }

        let mut state = model.default_state();
        // Spread legs out a bit
        for i in 0..4 {
            state.q[i * 2] = 0.3 + (i as f64) * 0.15;
            state.q[i * 2 + 1] = -0.4;
        }

        WasmMjcfAntSim { model, state }
    }

    pub fn step_n(&mut self, n: usize) {
        let dt = self.model.dt;
        for _ in 0..n {
            let acc = aba(&self.model, &self.state);
            self.state.v = &self.state.v + &acc * dt;
            self.state.q = &self.state.q + &self.state.v * dt;
            self.state.time += dt;
        }
    }

    pub fn nbodies(&self) -> usize { self.model.nbodies() }
    pub fn joint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut positions = Vec::with_capacity(nb * 3);
        for xf in xforms.iter().take(nb) {
            positions.push(xf.pos.x);
            positions.push(xf.pos.y);
            positions.push(xf.pos.z);
        }
        positions
    }
    pub fn body_endpoint_positions(&self) -> Vec<f64> {
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let nb = self.model.nbodies();
        let mut positions = Vec::with_capacity(nb * 3);
        for (i, xf) in xforms.iter().enumerate().take(nb) {
            let body = &self.model.bodies[i];
            let ep = body.inertia.com * 2.0;
            let rotated = xf.rot.transpose() * ep;
            let p = rotated + xf.pos;
            positions.push(p.x);
            positions.push(p.y);
            positions.push(p.z);
        }
        positions
    }
    pub fn time(&self) -> f64 { self.state.time }
}

/// Cartpole: cart + inverted pendulum (classic control benchmark).
#[wasm_bindgen]
pub struct WasmMjcfCartpoleSim {
    // Simple inline cartpole (cart position x, pole angle theta)
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
    time: f64,
    dt: f64,
    cart_mass: f64,
    pole_mass: f64,
    pole_length: f64,
}

#[wasm_bindgen]
impl WasmMjcfCartpoleSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmMjcfCartpoleSim {
        WasmMjcfCartpoleSim {
            x: 0.0, x_dot: 0.0,
            theta: 0.15, theta_dot: 0.0,
            time: 0.0, dt: 0.002,
            cart_mass: 1.0, pole_mass: 0.1, pole_length: 1.0,
        }
    }

    pub fn step_n(&mut self, n: usize) {
        let g = GRAVITY;
        let mc = self.cart_mass;
        let mp = self.pole_mass;
        let l = self.pole_length;

        for _ in 0..n {
            let s = self.theta.sin();
            let c = self.theta.cos();
            let total = mc + mp;

            // Simple PD controller to keep pole upright
            let force = -5.0 * self.theta - 2.0 * self.theta_dot - 0.5 * self.x - 0.3 * self.x_dot;

            let theta_acc = (g * s - c * (force + mp * l * self.theta_dot.powi(2) * s) / total)
                / (l * (4.0 / 3.0 - mp * c * c / total));
            let x_acc = (force + mp * l * (self.theta_dot.powi(2) * s - theta_acc * c)) / total;

            self.theta_dot += theta_acc * self.dt;
            self.theta += self.theta_dot * self.dt;
            self.x_dot += x_acc * self.dt;
            self.x += self.x_dot * self.dt;
            self.time += self.dt;
        }
    }

    pub fn cart_x(&self) -> f64 { self.x }
    pub fn pole_angle(&self) -> f64 { self.theta }
    pub fn pole_tip_x(&self) -> f64 { self.x + self.pole_length * self.theta.sin() }
    pub fn pole_tip_y(&self) -> f64 { self.pole_length * self.theta.cos() }
    pub fn time(&self) -> f64 { self.time }
}

/// MJCF editor: parse a simple body description and return skeleton.
#[wasm_bindgen]
pub struct WasmMjcfEditorSim {
    // Parsed body chain: stores positions of each body
    body_x: Vec<f64>,
    body_y: Vec<f64>,
    n_bodies: usize,
}

#[wasm_bindgen]
impl WasmMjcfEditorSim {
    /// Parse a simple MJCF-like string. Format: one body per line, "name length mass".
    pub fn parse(xml: &str) -> WasmMjcfEditorSim {
        let mut lengths = Vec::new();
        let mut body_x = vec![0.0];
        let mut body_y = vec![0.0];

        // Simple parser: each line "name length" creates a body
        for line in xml.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('<') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(l) = parts[1].parse::<f64>() {
                    lengths.push(l.clamp(0.1, 3.0));
                }
            }
        }

        if lengths.is_empty() {
            lengths.push(1.0); // default single body
        }

        // Build skeleton as a chain hanging down
        let mut y = 0.0;
        for l in &lengths {
            y -= l;
            body_x.push(0.0);
            body_y.push(y);
        }

        let n_bodies = lengths.len();
        WasmMjcfEditorSim { body_x, body_y, n_bodies }
    }

    pub fn n_bodies(&self) -> usize { self.n_bodies }
    pub fn positions(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..self.body_x.len() {
            out.push(self.body_x[i]);
            out.push(self.body_y[i]);
        }
        out
    }
}

// ================================================================
// phyz-real2sim: Inverse Problems (inline optimization demos)
// ================================================================

/// Parameter fitting: gradient descent to match a reference trajectory.
#[wasm_bindgen]
pub struct WasmReal2SimFitSim {
    // Reference trajectory (generated from true params)
    ref_traj: Vec<f64>, // angles at each timestep
    // Current estimated params
    est_mass: f64,
    est_length: f64,
    // True params
    true_mass: f64,
    true_length: f64,
    // Optimization state
    loss_history: Vec<f64>,
    iteration: usize,
    lr: f64,
    dt: f64,
    n_steps: usize,
}

#[wasm_bindgen]
impl WasmReal2SimFitSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReal2SimFitSim {
        let true_mass = 1.0;
        let true_length = 1.5;
        let dt = 0.002;
        let n_steps = 300;

        // Generate reference trajectory
        let ref_traj = Self::simulate_pendulum(true_mass, true_length, 0.8, dt, n_steps);

        WasmReal2SimFitSim {
            ref_traj, est_mass: 2.0, est_length: 1.0,
            true_mass, true_length,
            loss_history: vec![], iteration: 0, lr: 0.02,
            dt, n_steps,
        }
    }

    fn simulate_pendulum(_mass: f64, length: f64, q0: f64, dt: f64, n: usize) -> Vec<f64> {
        let g = GRAVITY;
        let mut q = q0;
        let mut v = 0.0;
        let mut traj = Vec::with_capacity(n);
        for _ in 0..n {
            let acc = -g / length * q.sin();
            v += acc * dt;
            q += v * dt;
            traj.push(q);
        }
        traj
    }

    fn compute_loss(&self, mass: f64, length: f64) -> f64 {
        let traj = Self::simulate_pendulum(mass, length, 0.8, self.dt, self.n_steps);
        let mut loss = 0.0;
        for (a, b) in traj.iter().zip(self.ref_traj.iter()) {
            loss += (a - b).powi(2);
        }
        loss / self.n_steps as f64
    }

    pub fn step_n(&mut self, n: usize) {
        let eps = 1e-5;
        for _ in 0..n {
            let loss = self.compute_loss(self.est_mass, self.est_length);
            self.loss_history.push(loss);

            // Gradient w.r.t. mass
            let lp_m = self.compute_loss(self.est_mass + eps, self.est_length);
            let lm_m = self.compute_loss(self.est_mass - eps, self.est_length);
            let grad_m = (lp_m - lm_m) / (2.0 * eps);

            // Gradient w.r.t. length
            let lp_l = self.compute_loss(self.est_mass, self.est_length + eps);
            let lm_l = self.compute_loss(self.est_mass, self.est_length - eps);
            let grad_l = (lp_l - lm_l) / (2.0 * eps);

            self.est_mass -= self.lr * grad_m;
            self.est_length -= self.lr * grad_l;
            self.est_mass = self.est_mass.clamp(0.1, 5.0);
            self.est_length = self.est_length.clamp(0.3, 3.0);
            self.iteration += 1;
        }
    }

    pub fn est_mass(&self) -> f64 { self.est_mass }
    pub fn est_length(&self) -> f64 { self.est_length }
    pub fn true_mass(&self) -> f64 { self.true_mass }
    pub fn true_length(&self) -> f64 { self.true_length }
    pub fn loss_history(&self) -> Vec<f64> { self.loss_history.clone() }
    pub fn iteration(&self) -> usize { self.iteration }
    pub fn current_loss(&self) -> f64 { self.loss_history.last().copied().unwrap_or(f64::MAX) }
    pub fn time(&self) -> f64 { self.iteration as f64 }
    /// Current estimated trajectory (for overlay rendering)
    pub fn est_trajectory(&self) -> Vec<f64> {
        Self::simulate_pendulum(self.est_mass, self.est_length, 0.8, self.dt, self.n_steps)
    }
    pub fn ref_trajectory(&self) -> Vec<f64> { self.ref_traj.clone() }
}

/// Loss landscape: 2D contour of loss over (mass, length) parameter grid.
#[wasm_bindgen]
pub struct WasmReal2SimLandscapeSim {
    grid: Vec<f64>, // flattened grid_size x grid_size loss values
    grid_size: usize,
    // Gradient descent path
    path_m: Vec<f64>,
    path_l: Vec<f64>,
    est_mass: f64,
    est_length: f64,
    ref_traj: Vec<f64>,
    dt: f64,
    n_steps: usize,
    lr: f64,
    iteration: usize,
}

#[wasm_bindgen]
impl WasmReal2SimLandscapeSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReal2SimLandscapeSim {
        let dt = 0.002;
        let n_steps = 300;
        let ref_traj = WasmReal2SimFitSim::simulate_pendulum(1.0, 1.5, 0.8, dt, n_steps);

        let grid_size = 32;
        let mut grid = vec![0.0; grid_size * grid_size];

        // Compute loss over (mass, length) grid
        let m_min = 0.3;
        let m_max = 3.0;
        let l_min = 0.5;
        let l_max = 2.5;

        for i in 0..grid_size {
            for j in 0..grid_size {
                let m = m_min + (m_max - m_min) * (j as f64 / (grid_size - 1) as f64);
                let l = l_min + (l_max - l_min) * (i as f64 / (grid_size - 1) as f64);
                let traj = WasmReal2SimFitSim::simulate_pendulum(m, l, 0.8, dt, n_steps);
                let mut loss = 0.0;
                for (a, b) in traj.iter().zip(ref_traj.iter()) {
                    loss += (a - b).powi(2);
                }
                grid[i * grid_size + j] = (loss / n_steps as f64).ln().max(-10.0);
            }
        }

        WasmReal2SimLandscapeSim {
            grid, grid_size,
            path_m: vec![2.5], path_l: vec![0.8],
            est_mass: 2.5, est_length: 0.8,
            ref_traj, dt, n_steps, lr: 0.02, iteration: 0,
        }
    }

    pub fn step_n(&mut self, n: usize) {
        let eps = 1e-5;
        for _ in 0..n {
            let compute_loss = |m: f64, l: f64| -> f64 {
                let traj = WasmReal2SimFitSim::simulate_pendulum(m, l, 0.8, self.dt, self.n_steps);
                let mut loss = 0.0;
                for (a, b) in traj.iter().zip(self.ref_traj.iter()) {
                    loss += (a - b).powi(2);
                }
                loss / self.n_steps as f64
            };

            let lp_m = compute_loss(self.est_mass + eps, self.est_length);
            let lm_m = compute_loss(self.est_mass - eps, self.est_length);
            let grad_m = (lp_m - lm_m) / (2.0 * eps);

            let lp_l = compute_loss(self.est_mass, self.est_length + eps);
            let lm_l = compute_loss(self.est_mass, self.est_length - eps);
            let grad_l = (lp_l - lm_l) / (2.0 * eps);

            self.est_mass -= self.lr * grad_m;
            self.est_length -= self.lr * grad_l;
            self.est_mass = self.est_mass.clamp(0.3, 3.0);
            self.est_length = self.est_length.clamp(0.5, 2.5);

            self.path_m.push(self.est_mass);
            self.path_l.push(self.est_length);
            self.iteration += 1;
        }
    }

    pub fn grid(&self) -> Vec<f64> { self.grid.clone() }
    pub fn grid_size(&self) -> usize { self.grid_size }
    pub fn path_m(&self) -> Vec<f64> { self.path_m.clone() }
    pub fn path_l(&self) -> Vec<f64> { self.path_l.clone() }
    pub fn time(&self) -> f64 { self.iteration as f64 }
    pub fn est_mass(&self) -> f64 { self.est_mass }
    pub fn est_length(&self) -> f64 { self.est_length }
}

/// Adam vs GD: compare two optimizers on the same problem.
#[wasm_bindgen]
pub struct WasmReal2SimAdamVsGdSim {
    ref_traj: Vec<f64>,
    dt: f64,
    n_steps: usize,
    // GD state
    gd_mass: f64,
    gd_length: f64,
    gd_loss: Vec<f64>,
    // Adam state
    adam_mass: f64,
    adam_length: f64,
    adam_loss: Vec<f64>,
    adam_m: [f64; 2], // first moment
    adam_v: [f64; 2], // second moment
    adam_t: usize,
    iteration: usize,
}

#[wasm_bindgen]
impl WasmReal2SimAdamVsGdSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReal2SimAdamVsGdSim {
        let dt = 0.002;
        let n_steps = 300;
        let ref_traj = WasmReal2SimFitSim::simulate_pendulum(1.0, 1.5, 0.8, dt, n_steps);

        WasmReal2SimAdamVsGdSim {
            ref_traj, dt, n_steps,
            gd_mass: 2.5, gd_length: 0.8, gd_loss: vec![],
            adam_mass: 2.5, adam_length: 0.8, adam_loss: vec![],
            adam_m: [0.0; 2], adam_v: [0.0; 2], adam_t: 0, iteration: 0,
        }
    }

    fn compute_loss_and_grad(&self, mass: f64, length: f64) -> (f64, f64, f64) {
        let eps = 1e-5;
        let compute = |m: f64, l: f64| -> f64 {
            let traj = WasmReal2SimFitSim::simulate_pendulum(m, l, 0.8, self.dt, self.n_steps);
            let mut loss = 0.0;
            for (a, b) in traj.iter().zip(self.ref_traj.iter()) {
                loss += (a - b).powi(2);
            }
            loss / self.n_steps as f64
        };
        let loss = compute(mass, length);
        let grad_m = (compute(mass + eps, length) - compute(mass - eps, length)) / (2.0 * eps);
        let grad_l = (compute(mass, length + eps) - compute(mass, length - eps)) / (2.0 * eps);
        (loss, grad_m, grad_l)
    }

    pub fn step_n(&mut self, n: usize) {
        let lr = 0.02;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let adam_eps = 1e-8;

        for _ in 0..n {
            // GD step
            let (gd_l, gd_gm, gd_gl) = self.compute_loss_and_grad(self.gd_mass, self.gd_length);
            self.gd_loss.push(gd_l);
            self.gd_mass -= lr * gd_gm;
            self.gd_length -= lr * gd_gl;
            self.gd_mass = self.gd_mass.clamp(0.1, 5.0);
            self.gd_length = self.gd_length.clamp(0.3, 3.0);

            // Adam step
            let (adam_l, adam_gm, adam_gl) = self.compute_loss_and_grad(self.adam_mass, self.adam_length);
            self.adam_loss.push(adam_l);
            self.adam_t += 1;

            self.adam_m[0] = beta1 * self.adam_m[0] + (1.0 - beta1) * adam_gm;
            self.adam_m[1] = beta1 * self.adam_m[1] + (1.0 - beta1) * adam_gl;
            self.adam_v[0] = beta2 * self.adam_v[0] + (1.0 - beta2) * adam_gm * adam_gm;
            self.adam_v[1] = beta2 * self.adam_v[1] + (1.0 - beta2) * adam_gl * adam_gl;

            let bc1 = 1.0 - beta1.powi(self.adam_t as i32);
            let bc2 = 1.0 - beta2.powi(self.adam_t as i32);

            self.adam_mass -= lr * (self.adam_m[0] / bc1) / ((self.adam_v[0] / bc2).sqrt() + adam_eps);
            self.adam_length -= lr * (self.adam_m[1] / bc1) / ((self.adam_v[1] / bc2).sqrt() + adam_eps);
            self.adam_mass = self.adam_mass.clamp(0.1, 5.0);
            self.adam_length = self.adam_length.clamp(0.3, 3.0);

            self.iteration += 1;
        }
    }

    pub fn gd_loss(&self) -> Vec<f64> { self.gd_loss.clone() }
    pub fn adam_loss(&self) -> Vec<f64> { self.adam_loss.clone() }
    pub fn iteration(&self) -> usize { self.iteration }
    pub fn time(&self) -> f64 { self.iteration as f64 }
    pub fn gd_mass(&self) -> f64 { self.gd_mass }
    pub fn gd_length(&self) -> f64 { self.gd_length }
    pub fn adam_mass(&self) -> f64 { self.adam_mass }
    pub fn adam_length(&self) -> f64 { self.adam_length }
}

// ================================================================
// phyz-regge: Regge Calculus (GR + EM)
// ================================================================

use phyz_regge::{
    ActionParams, Fields, SimplicialComplex,
    action::einstein_maxwell_action,
    action::noether_current,
    mesh,
    symmetry,
};

/// Curvature slice: Reissner-Nordström edge lengths as color heatmap.
#[wasm_bindgen]
pub struct WasmReggeCurvatureSim {
    n: usize,
    spacing: f64,
}

#[wasm_bindgen]
impl WasmReggeCurvatureSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReggeCurvatureSim {
        WasmReggeCurvatureSim { n: 4, spacing: 1.0 }
    }

    /// Compute edge lengths for Reissner-Nordström mesh.
    /// Returns flat array of edge length values (one per edge).
    pub fn compute(&self, mass: f64, charge: f64) -> Vec<f64> {
        let r_min = (mass + (mass * mass - charge * charge).abs().sqrt()).max(0.5) * 1.1;
        let (_complex, lengths) = mesh::reissner_nordstrom(self.n, self.spacing, mass, charge, r_min);
        lengths
    }

    /// Returns number of edges for the current mesh size.
    pub fn n_edges(&self) -> usize {
        let (complex, _) = mesh::flat_hypercubic(self.n, self.spacing);
        complex.n_edges()
    }

    /// Grid size for heatmap rendering.
    pub fn grid_size(&self) -> usize {
        // Approximate: arrange edges in a square grid
        let ne = self.n_edges();
        ((ne as f64).sqrt().ceil() as usize).max(4)
    }

    pub fn time(&self) -> f64 { 0.0 }
}

/// Symmetry bars: Noether current norms for different spacetimes.
#[wasm_bindgen]
pub struct WasmReggeSymmetrySim {
    n: usize,
    spacing: f64,
}

#[wasm_bindgen]
impl WasmReggeSymmetrySim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReggeSymmetrySim {
        WasmReggeSymmetrySim { n: 3, spacing: 1.0 }
    }

    /// Compute Noether current norms for flat, RN, and Kerr spacetimes.
    /// Returns [flat_gauge, flat_translation, rn_gauge, rn_translation, kerr_gauge, kerr_translation].
    pub fn compute(&self) -> Vec<f64> {
        let params = ActionParams::default();
        let mut results = Vec::new();

        // Flat spacetime
        let (flat_complex, flat_lengths) = mesh::flat_hypercubic(self.n, self.spacing);
        let ne = flat_complex.n_edges();
        let flat_fields = Fields::new(flat_lengths, vec![0.0; ne]);
        let flat_gauge = symmetry::gauge_generator(&flat_complex, 0);
        let flat_j = noether_current(&flat_complex, &flat_fields, &flat_gauge.delta_lengths, &flat_gauge.delta_phases, &params);
        results.push(flat_j.abs());
        let flat_trans = symmetry::translation_generator(&flat_complex, &flat_fields, 0, self.n);
        let flat_tj = noether_current(&flat_complex, &flat_fields, &flat_trans.delta_lengths, &flat_trans.delta_phases, &params);
        results.push(flat_tj.abs());

        // Reissner-Nordström
        let (rn_complex, rn_lengths) = mesh::reissner_nordstrom(self.n, self.spacing, 1.0, 0.3, 2.5);
        let rn_ne = rn_complex.n_edges();
        let rn_fields = Fields::new(rn_lengths, vec![0.0; rn_ne]);
        let rn_gauge = symmetry::gauge_generator(&rn_complex, 0);
        let rn_j = noether_current(&rn_complex, &rn_fields, &rn_gauge.delta_lengths, &rn_gauge.delta_phases, &params);
        results.push(rn_j.abs());
        let rn_trans = symmetry::translation_generator(&rn_complex, &rn_fields, 0, self.n);
        let rn_tj = noether_current(&rn_complex, &rn_fields, &rn_trans.delta_lengths, &rn_trans.delta_phases, &params);
        results.push(rn_tj.abs());

        // Kerr
        let (kerr_complex, kerr_lengths) = mesh::kerr(self.n, self.spacing, 1.0, 0.5, 2.5);
        let kerr_ne = kerr_complex.n_edges();
        let kerr_fields = Fields::new(kerr_lengths, vec![0.0; kerr_ne]);
        let kerr_gauge = symmetry::gauge_generator(&kerr_complex, 0);
        let kerr_j = noether_current(&kerr_complex, &kerr_fields, &kerr_gauge.delta_lengths, &kerr_gauge.delta_phases, &params);
        results.push(kerr_j.abs());
        let kerr_trans = symmetry::translation_generator(&kerr_complex, &kerr_fields, 0, self.n);
        let kerr_tj = noether_current(&kerr_complex, &kerr_fields, &kerr_trans.delta_lengths, &kerr_trans.delta_phases, &params);
        results.push(kerr_tj.abs());

        results
    }

    pub fn time(&self) -> f64 { 0.0 }
}

/// Action landscape: Einstein-Maxwell action vs uniform scale factor, with gradient descent.
#[wasm_bindgen]
pub struct WasmReggeActionSim {
    complex: SimplicialComplex,
    base_lengths: Vec<f64>,
    n_edges: usize,
    // Landscape data
    scales: Vec<f64>,
    actions: Vec<f64>,
    // GD state
    current_scale: f64,
    gd_path: Vec<f64>,
    iteration: usize,
}

#[wasm_bindgen]
impl WasmReggeActionSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmReggeActionSim {
        let (complex, base_lengths) = mesh::flat_hypercubic(3, 1.0);
        let n_edges = complex.n_edges();
        let params = ActionParams::default();

        // Compute action landscape over scale factors
        let n_pts = 64;
        let mut scales = Vec::with_capacity(n_pts);
        let mut actions = Vec::with_capacity(n_pts);

        for i in 0..n_pts {
            let s = 0.3 + 2.7 * (i as f64 / (n_pts - 1) as f64);
            scales.push(s);
            let scaled: Vec<f64> = base_lengths.iter().map(|l| l * s).collect();
            let fields = Fields::new(scaled, vec![0.0; n_edges]);
            let a = einstein_maxwell_action(&complex, &fields, &params);
            actions.push(a);
        }

        WasmReggeActionSim {
            complex, base_lengths, n_edges,
            scales, actions,
            current_scale: 2.5, gd_path: vec![2.5], iteration: 0,
        }
    }

    pub fn step_n(&mut self, n: usize) {
        let lr = 0.001;
        let params = ActionParams::default();
        let eps = 1e-5;

        for _ in 0..n {
            let compute_action = |s: f64| -> f64 {
                let scaled: Vec<f64> = self.base_lengths.iter().map(|l| l * s).collect();
                let fields = Fields::new(scaled, vec![0.0; self.n_edges]);
                einstein_maxwell_action(&self.complex, &fields, &params)
            };

            let grad = (compute_action(self.current_scale + eps) - compute_action(self.current_scale - eps)) / (2.0 * eps);
            self.current_scale -= lr * grad;
            self.current_scale = self.current_scale.clamp(0.3, 3.0);
            self.gd_path.push(self.current_scale);
            self.iteration += 1;
        }
    }

    pub fn scales(&self) -> Vec<f64> { self.scales.clone() }
    pub fn actions(&self) -> Vec<f64> { self.actions.clone() }
    pub fn gd_path(&self) -> Vec<f64> { self.gd_path.clone() }
    pub fn current_scale(&self) -> f64 { self.current_scale }
    pub fn current_action(&self) -> f64 {
        let scaled: Vec<f64> = self.base_lengths.iter().map(|l| l * self.current_scale).collect();
        let fields = Fields::new(scaled, vec![0.0; self.n_edges]);
        einstein_maxwell_action(&self.complex, &fields, &ActionParams::default())
    }
    pub fn time(&self) -> f64 { self.iteration as f64 }
}

// ================================================================
// phyz-compile: Physics Kernel Compiler (inline, demonstrates concept)
// ================================================================

/// Kernel IR: build a simple physics kernel and return its DAG structure.
#[wasm_bindgen]
pub struct WasmCompileIrSim {
    // IR node descriptions: [type, name, depth, parent_idx]
    node_types: Vec<String>,
    node_names: Vec<String>,
    node_depths: Vec<usize>,
    node_parents: Vec<i32>,
}

#[wasm_bindgen]
impl WasmCompileIrSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmCompileIrSim {
        // Build a sample physics kernel IR: Laplacian + time stepping
        let node_types = vec![
            "program".into(), "field".into(), "field".into(), "stencil".into(),
            "binop".into(), "binop".into(), "store".into(),
        ];
        let node_names = vec![
            "heat_step".into(), "u (input)".into(), "u_next (output)".into(),
            "laplacian(u)".into(), "dt * lap".into(), "u + dt*lap".into(),
            "store u_next".into(),
        ];
        let node_depths = vec![0, 1, 1, 1, 2, 2, 1];
        let node_parents = vec![-1, 0, 0, 0, 3, 4, 0];

        WasmCompileIrSim { node_types, node_names, node_depths, node_parents }
    }

    pub fn num_nodes(&self) -> usize { self.node_types.len() }
    pub fn node_types(&self) -> Vec<String> { self.node_types.clone() }
    pub fn node_names(&self) -> Vec<String> { self.node_names.clone() }
    pub fn node_depths(&self) -> Vec<usize> { self.node_depths.clone() }
    pub fn node_parents(&self) -> Vec<i32> { self.node_parents.clone() }
    pub fn time(&self) -> f64 { 0.0 }
}

/// WGSL output: display generated WGSL source for different physics ops.
#[wasm_bindgen]
pub struct WasmCompileWgslSim {
    sources: Vec<String>,
    labels: Vec<String>,
    current: usize,
}

#[wasm_bindgen]
impl WasmCompileWgslSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmCompileWgslSim {
        let labels = vec!["heat_diffusion".into(), "wave_equation".into(), "advection".into()];
        let sources = vec![
            // Heat diffusion kernel
            "@group(0) @binding(0) var<storage,read> u: array<f32>;\n\
             @group(0) @binding(1) var<storage,read_write> u_next: array<f32>;\n\
             @compute @workgroup_size(8,8,1)\n\
             fn heat_step(@builtin(global_invocation_id) gid: vec3u) {\n\
             \x20 let i = gid.x + gid.y * N;\n\
             \x20 let lap = u[i-1] + u[i+1] + u[i-N] + u[i+N] - 4.0*u[i];\n\
             \x20 u_next[i] = u[i] + dt * alpha * lap;\n\
             }".into(),
            // Wave equation kernel
            "@group(0) @binding(0) var<storage,read> u: array<f32>;\n\
             @group(0) @binding(1) var<storage,read> u_prev: array<f32>;\n\
             @group(0) @binding(2) var<storage,read_write> u_next: array<f32>;\n\
             @compute @workgroup_size(8,8,1)\n\
             fn wave_step(@builtin(global_invocation_id) gid: vec3u) {\n\
             \x20 let i = gid.x + gid.y * N;\n\
             \x20 let lap = u[i-1] + u[i+1] + u[i-N] + u[i+N] - 4.0*u[i];\n\
             \x20 u_next[i] = 2.0*u[i] - u_prev[i] + c2*dt2*lap;\n\
             }".into(),
            // Advection kernel
            "@group(0) @binding(0) var<storage,read> u: array<f32>;\n\
             @group(0) @binding(1) var<storage,read> vx: array<f32>;\n\
             @group(0) @binding(2) var<storage,read_write> u_next: array<f32>;\n\
             @compute @workgroup_size(8,8,1)\n\
             fn advect(@builtin(global_invocation_id) gid: vec3u) {\n\
             \x20 let i = gid.x + gid.y * N;\n\
             \x20 let upwind = select(u[i]-u[i-1], u[i+1]-u[i], vx[i]<0.0);\n\
             \x20 u_next[i] = u[i] - dt * vx[i] * upwind / dx;\n\
             }".into(),
        ];
        WasmCompileWgslSim { sources, labels, current: 0 }
    }

    pub fn num_kernels(&self) -> usize { self.sources.len() }
    pub fn current_label(&self) -> String { self.labels[self.current].clone() }
    pub fn current_source(&self) -> String { self.sources[self.current].clone() }
    pub fn all_labels(&self) -> Vec<String> { self.labels.clone() }
    /// Cycle to next kernel
    pub fn next(&mut self) { self.current = (self.current + 1) % self.sources.len(); }
    pub fn source_lines(&self) -> usize { self.sources[self.current].lines().count() }
    pub fn time(&self) -> f64 { self.current as f64 }
}

/// Fusion viz: two kernels side-by-side, fuse them, animate the merge.
#[wasm_bindgen]
pub struct WasmCompileFusionSim {
    // Kernel A nodes
    a_names: Vec<String>,
    a_count: usize,
    // Kernel B nodes
    b_names: Vec<String>,
    b_count: usize,
    // Fused kernel nodes
    fused_names: Vec<String>,
    fused_count: usize,
    // Animation progress (0.0 = separate, 1.0 = fully fused)
    progress: f64,
    time: f64,
}

#[wasm_bindgen]
impl WasmCompileFusionSim {
    #[allow(clippy::new_without_default)]
    pub fn new() -> WasmCompileFusionSim {
        let a_names = vec!["load u".into(), "laplacian".into(), "mul dt".into(), "store tmp".into()];
        let b_names = vec!["load tmp".into(), "load v".into(), "add".into(), "store v_next".into()];
        let fused_names = vec![
            "load u".into(), "laplacian".into(), "mul dt".into(),
            "load v".into(), "add".into(), "store v_next".into(),
        ];

        WasmCompileFusionSim {
            a_count: a_names.len(), b_count: b_names.len(), fused_count: fused_names.len(),
            a_names, b_names, fused_names,
            progress: 0.0, time: 0.0,
        }
    }

    pub fn step_n(&mut self, _n: usize) {
        self.progress = (self.progress + 0.008).min(1.0);
        self.time += 0.016;
        // Auto-reset after fully fused
        if self.progress >= 1.0 {
            self.progress = 1.0;
            // Hold for a bit then reset
            if self.time > 10.0 {
                self.progress = 0.0;
                self.time = 0.0;
            }
        }
    }

    pub fn a_names(&self) -> Vec<String> { self.a_names.clone() }
    pub fn b_names(&self) -> Vec<String> { self.b_names.clone() }
    pub fn fused_names(&self) -> Vec<String> { self.fused_names.clone() }
    pub fn a_count(&self) -> usize { self.a_count }
    pub fn b_count(&self) -> usize { self.b_count }
    pub fn fused_count(&self) -> usize { self.fused_count }
    pub fn progress(&self) -> f64 { self.progress }
    pub fn time(&self) -> f64 { self.time }
}
