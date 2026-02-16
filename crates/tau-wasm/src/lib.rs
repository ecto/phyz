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
