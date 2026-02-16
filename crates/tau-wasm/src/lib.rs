use wasm_bindgen::prelude::*;

use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::{Model, ModelBuilder, State};
use tau_rigid::{aba, forward_kinematics, total_energy};

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
        let spacing = bob_radius * 2.0;

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

        // Multiple passes so the impulse wave propagates through the chain
        for _ in 0..self.n {
            for i in 0..(self.n - 1) {
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

                if dist < 2.0 * r && dist > 1e-10 {
                    let nx = dx / dist;

                    // Project angular velocities to bob velocities along contact normal
                    let vxi = self.state.v[i] * l * self.state.q[i].cos();
                    let vxj = self.state.v[j] * l * self.state.q[j].cos();
                    let rel_vn = (vxj - vxi) * nx;

                    if rel_vn < 0.0 {
                        // Near-elastic collision: swap angular velocities
                        let e = 0.999;
                        let vi = self.state.v[i];
                        let vj = self.state.v[j];
                        self.state.v[i] = vj * e;
                        self.state.v[j] = vi * e;

                        // Separate bobs by adjusting angles
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
// Particle Physics: DEM / MPM Concept (Phase 5: tau-particle)
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
// World Generation + Training (Phase 7: tau-world)
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
// Electromagnetic Field Simulation (Phase 8: tau-em)
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
// Molecular Dynamics (Phase 9: tau-md)
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
// Lattice Gauge Theory (Phase 10: tau-qft)
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
