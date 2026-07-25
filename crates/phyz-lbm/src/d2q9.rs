//! D2Q9 Lattice Boltzmann solver for 2D incompressible flow.
//!
//! Nine velocity directions on 2D square lattice:
//! ```text
//!   6   2   5
//!    \  |  /
//!   3 - 0 - 1
//!    /  |  \
//!   7   4   8
//! ```
//!
//! The solver supports BGK, TRT and MRT collision (see [`CollisionModel`]),
//! optional Smagorinsky LES ([`Turbulence`]), Guo body forcing, and declarative
//! domain boundaries ([`Boundaries`]) that [`LatticeBoltzmann2D::step`] applies
//! automatically.
//!
//! ```
//! use phyz_lbm::{analytic, boundary, LatticeBoltzmann2D};
//!
//! // Force-driven plane Poiseuille flow.
//! let (ny, nu, u_peak) = (17usize, 0.05, 0.05);
//! let g = analytic::poiseuille_force_for_peak(ny as f64, u_peak, nu);
//! let mut lbm = LatticeBoltzmann2D::new(4, ny, nu)
//!     .with_boundaries(boundary::channel_2d())
//!     .with_force([g, 0.0]);
//! lbm.initialize_uniform(1.0, [0.0, 0.0]);
//! lbm.run(20_000);
//!
//! let centre = lbm.velocity(0, ny / 2)[0];
//! assert!((centre - u_peak).abs() / u_peak < 1e-3);
//! ```

use crate::boundary::{Boundaries, Boundary, Side};
use crate::collision::{CollisionModel, Turbulence, smagorinsky_tau, trt_omega_minus};

/// D2Q9 discrete velocities: [vx, vy]
pub const E: [[i32; 2]; 9] = [
    [0, 0],   // 0: rest
    [1, 0],   // 1: east
    [0, 1],   // 2: north
    [-1, 0],  // 3: west
    [0, -1],  // 4: south
    [1, 1],   // 5: northeast
    [-1, 1],  // 6: northwest
    [-1, -1], // 7: southwest
    [1, -1],  // 8: southeast
];

/// D2Q9 weights
pub const W: [f64; 9] = [
    4.0 / 9.0, // 0: rest
    1.0 / 9.0, // 1-4: cardinal
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0, // 5-8: diagonal
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Opposite direction indices for bounce-back
pub const OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Index of the lattice direction equal to `(ex, ey)`.
#[inline]
fn dir_index(ex: i32, ey: i32) -> usize {
    let mut i = 0;
    while i < 9 {
        if E[i][0] == ex && E[i][1] == ey {
            return i;
        }
        i += 1;
    }
    panic!("({ex}, {ey}) is not a D2Q9 direction")
}

/// Direction obtained by flipping component `axis` — specular reflection.
#[inline]
fn mirror_index(i: usize, axis: usize) -> usize {
    let mut e = E[i];
    e[axis] = -e[axis];
    dir_index(e[0], e[1])
}

/// Equilibrium distribution `f_i^eq = w_i ρ [1 + 3(e_i·u) + 9/2(e_i·u)² - 3/2 u²]`.
#[inline]
pub fn equilibrium(i: usize, rho: f64, u: [f64; 2]) -> f64 {
    let ex = E[i][0] as f64;
    let ey = E[i][1] as f64;
    let eu = ex * u[0] + ey * u[1];
    let uu = u[0] * u[0] + u[1] * u[1];
    W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
}

/// Guo's forcing source term, before the `(1 - ω/2)` prefactor:
/// `S_i = w_i [ (e_i - u)/c_s² + (e_i·u) e_i / c_s⁴ ] · F`.
#[inline]
fn guo_source(i: usize, u: [f64; 2], force: [f64; 2]) -> f64 {
    let ex = E[i][0] as f64;
    let ey = E[i][1] as f64;
    let eu = ex * u[0] + ey * u[1];
    let ef = ex * force[0] + ey * force[1];
    W[i] * (3.0 * ((ex - u[0]) * force[0] + (ey - u[1]) * force[1]) + 9.0 * eu * ef)
}

/// d'Humières moment basis for D2Q9, rows: ρ, e, ε, jx, qx, jy, qy, pxx, pxy.
const M: [[f64; 9]; 9] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
    [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0],
];

/// Row norms `Σ_i M[k][i]²`. The rows are orthogonal, so `M⁻¹ = Mᵀ / norms`.
const M_NORM: [f64; 9] = [9.0, 36.0, 36.0, 6.0, 12.0, 6.0, 12.0, 4.0, 4.0];

/// D2Q9 Lattice Boltzmann solver for 2D flow.
pub struct LatticeBoltzmann2D {
    /// Grid width
    pub nx: usize,
    /// Grid height
    pub ny: usize,
    /// Kinematic viscosity
    pub nu: f64,
    /// Relaxation time τ = 3ν + 0.5
    pub tau: f64,
    /// Distribution functions f_i at each grid point
    /// Shape: [nx, ny, 9]
    pub f: Vec<f64>,
    /// Temporary buffer holding the post-collision, pre-streaming state.
    /// Boundary conditions read from it, so it must survive `stream`.
    f_temp: Vec<f64>,
    /// Collision operator. Defaults to TRT with the bounce-back magic number.
    pub collision: CollisionModel,
    /// Sub-grid turbulence closure. Defaults to none.
    pub turbulence: Turbulence,
    /// Uniform body force per unit volume, applied with Guo's scheme.
    pub force: [f64; 2],
    /// Domain boundary conditions, applied automatically by [`Self::step`].
    pub boundaries: Boundaries<2>,
}

/// Which macroscopic quantity a Zou–He face prescribes.
#[derive(Clone, Copy)]
enum ZouHe {
    Velocity([f64; 2]),
    Pressure(f64),
}

impl LatticeBoltzmann2D {
    /// Create new 2D LBM solver.
    ///
    /// Defaults: TRT collision with `Λ = 3/16`, no turbulence model, no body
    /// force, fully periodic boundaries.
    ///
    /// # Arguments
    /// * `nx` - Grid width
    /// * `ny` - Grid height
    /// * `nu` - Kinematic viscosity
    pub fn new(nx: usize, ny: usize, nu: f64) -> Self {
        let tau = 3.0 * nu + 0.5;
        let size = nx * ny * 9;
        Self {
            nx,
            ny,
            nu,
            tau,
            f: vec![0.0; size],
            f_temp: vec![0.0; size],
            collision: CollisionModel::default(),
            turbulence: Turbulence::default(),
            force: [0.0, 0.0],
            boundaries: Boundaries::periodic(),
        }
    }

    /// Select the collision operator. Chainable.
    #[must_use]
    pub fn with_collision(mut self, collision: CollisionModel) -> Self {
        self.collision = collision;
        self
    }

    /// Enable a sub-grid turbulence closure. Chainable.
    #[must_use]
    pub fn with_turbulence(mut self, turbulence: Turbulence) -> Self {
        self.turbulence = turbulence;
        self
    }

    /// Set a uniform body force (gravity, pressure gradient). Chainable.
    #[must_use]
    pub fn with_force(mut self, force: [f64; 2]) -> Self {
        self.force = force;
        self
    }

    /// Declare the domain boundaries. Chainable.
    #[must_use]
    pub fn with_boundaries(mut self, boundaries: Boundaries<2>) -> Self {
        self.boundaries = boundaries;
        self
    }

    /// Change the viscosity, keeping `tau` consistent.
    pub fn set_viscosity(&mut self, nu: f64) {
        self.nu = nu;
        self.tau = 3.0 * nu + 0.5;
    }

    /// Initialize with uniform density and velocity.
    pub fn initialize_uniform(&mut self, rho: f64, u: [f64; 2]) {
        self.initialize_with(|_, _| (rho, u));
    }

    /// Initialize from a field function returning `(ρ, u)` at each node.
    pub fn initialize_with<F>(&mut self, field: F)
    where
        F: Fn(usize, usize) -> (f64, [f64; 2]),
    {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let (rho, u) = field(x, y);
                for i in 0..9 {
                    self.set_f(x, y, i, equilibrium(i, rho, u));
                }
            }
        }
    }

    /// Get distribution function at (x, y, i).
    #[inline]
    fn get_f(&self, x: usize, y: usize, i: usize) -> f64 {
        self.f[x + y * self.nx + i * self.nx * self.ny]
    }

    /// Set distribution function at (x, y, i).
    #[inline]
    fn set_f(&mut self, x: usize, y: usize, i: usize, val: f64) {
        self.f[x + y * self.nx + i * self.nx * self.ny] = val;
    }

    /// Get temporary distribution function at (x, y, i).
    #[inline]
    fn get_f_temp(&self, x: usize, y: usize, i: usize) -> f64 {
        self.f_temp[x + y * self.nx + i * self.nx * self.ny]
    }

    /// Set temporary distribution function at (x, y, i).
    #[inline]
    fn set_f_temp(&mut self, x: usize, y: usize, i: usize, val: f64) {
        self.f_temp[x + y * self.nx + i * self.nx * self.ny] = val;
    }

    /// Compute macroscopic density at (x, y).
    pub fn density(&self, x: usize, y: usize) -> f64 {
        (0..9).map(|i| self.get_f(x, y, i)).sum()
    }

    /// Compute macroscopic velocity at (x, y).
    ///
    /// Includes the `F/2` correction required by Guo forcing, so this is the
    /// physical fluid velocity whether or not a body force is active.
    pub fn velocity(&self, x: usize, y: usize) -> [f64; 2] {
        let mut rho = 0.0;
        let mut u = [0.0, 0.0];
        for (i, e) in E.iter().enumerate() {
            let f = self.get_f(x, y, i);
            rho += f;
            u[0] += f * e[0] as f64;
            u[1] += f * e[1] as f64;
        }
        if rho < 1e-12 {
            return [0.0, 0.0];
        }
        [
            (u[0] + 0.5 * self.force[0]) / rho,
            (u[1] + 0.5 * self.force[1]) / rho,
        ]
    }

    /// Maximum velocity magnitude in domain.
    pub fn max_velocity(&self) -> f64 {
        let mut umax: f64 = 0.0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let u = self.velocity(x, y);
                umax = umax.max((u[0] * u[0] + u[1] * u[1]).sqrt());
            }
        }
        umax
    }

    /// Impose a velocity at a single node, preserving the non-equilibrium part.
    ///
    /// This is Guo's non-equilibrium extrapolation: the equilibrium part is
    /// replaced with `f^eq(ρ, u_target)` while `f^neq` is carried over from the
    /// nearest interior neighbour (or from the node itself if it is already
    /// interior). Overwriting *all* populations with their equilibrium values —
    /// as this function used to do — discards `f^neq` entirely and so discards
    /// the viscous stress at the boundary, injecting an O(1) error every step.
    ///
    /// Prefer declaring [`Boundary::Velocity`] on a face: whole-face Zou–He is
    /// exact for the D2Q9 moment system, where this node-local version is only
    /// first-order in the distance to the donor node.
    pub fn set_velocity_bc(&mut self, x: usize, y: usize, u: [f64; 2]) {
        // Pick a donor node: step inward if we are on a domain edge.
        let dx = if x == 0 {
            1
        } else if x + 1 == self.nx {
            -1
        } else {
            0
        };
        let dy = if y == 0 {
            1
        } else if y + 1 == self.ny {
            -1
        } else {
            0
        };
        let sx = (x as i32 + dx) as usize;
        let sy = (y as i32 + dy) as usize;

        let rho = self.density(sx, sy);
        let u_donor = self.velocity(sx, sy);
        for i in 0..9 {
            let f_neq = self.get_f(sx, sy, i) - equilibrium(i, rho, u_donor);
            self.set_f(x, y, i, equilibrium(i, rho, u) + f_neq);
        }
    }

    /// Set no-slip boundary condition (full-way bounce-back) at (x, y).
    ///
    /// Node-local helper for obstacles inside the domain. Domain faces are
    /// better declared as [`Boundary::NoSlip`], which uses halfway bounce-back
    /// and is second-order accurate.
    pub fn set_no_slip_bc(&mut self, x: usize, y: usize) {
        let f_new: [f64; 9] = std::array::from_fn(|i| self.get_f(x, y, OPP[i]));
        for (i, &val) in f_new.iter().enumerate() {
            self.set_f(x, y, i, val);
        }
    }

    /// Effective relaxation time at a node, including any eddy viscosity.
    #[inline]
    fn effective_tau(&self, fi: &[f64; 9], feq: &[f64; 9], rho: f64) -> f64 {
        match self.turbulence {
            Turbulence::None => self.tau,
            Turbulence::Smagorinsky { cs } => {
                let (mut qxx, mut qxy, mut qyy) = (0.0, 0.0, 0.0);
                for i in 0..9 {
                    let n = fi[i] - feq[i];
                    let ex = E[i][0] as f64;
                    let ey = E[i][1] as f64;
                    qxx += ex * ex * n;
                    qxy += ex * ey * n;
                    qyy += ey * ey * n;
                }
                let q = (2.0 * (qxx * qxx + 2.0 * qxy * qxy + qyy * qyy)).sqrt();
                smagorinsky_tau(self.tau, rho, q, cs)
            }
        }
    }

    /// Collision step. Writes the post-collision state into `f_temp`.
    fn collide(&mut self) {
        let force = self.force;
        let forced = force[0] != 0.0 || force[1] != 0.0;

        for y in 0..self.ny {
            for x in 0..self.nx {
                let mut fi = [0.0f64; 9];
                let mut rho = 0.0;
                let mut mx = 0.0;
                let mut my = 0.0;
                for i in 0..9 {
                    let v = self.get_f(x, y, i);
                    fi[i] = v;
                    rho += v;
                    mx += v * E[i][0] as f64;
                    my += v * E[i][1] as f64;
                }
                let inv_rho = if rho.abs() < 1e-12 { 0.0 } else { 1.0 / rho };
                // Guo: the momentum that enters f^eq carries half the force.
                let u = [
                    (mx + 0.5 * force[0]) * inv_rho,
                    (my + 0.5 * force[1]) * inv_rho,
                ];

                let feq: [f64; 9] = std::array::from_fn(|i| equilibrium(i, rho, u));
                let src: [f64; 9] = if forced {
                    std::array::from_fn(|i| guo_source(i, u, force))
                } else {
                    [0.0; 9]
                };
                let tau = self.effective_tau(&fi, &feq, rho);

                let post = match self.collision {
                    CollisionModel::Bgk => collide_bgk(&fi, &feq, &src, tau),
                    CollisionModel::Trt { magic } => collide_trt(&fi, &feq, &src, tau, magic),
                    CollisionModel::Mrt => collide_mrt(&fi, &feq, &src, tau),
                };

                for i in 0..9 {
                    self.set_f_temp(x, y, i, post[i]);
                }
            }
        }
    }

    /// Streaming step.
    ///
    /// Always wraps. On a non-periodic face the wrapped-in populations are
    /// exactly the ones [`Self::apply_boundaries`] overwrites, so the wrap is
    /// never observable there.
    fn stream(&mut self) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                for (i, e) in E.iter().enumerate() {
                    let xp = (x as i32 + e[0]).rem_euclid(self.nx as i32) as usize;
                    let yp = (y as i32 + e[1]).rem_euclid(self.ny as i32) as usize;
                    let f = self.get_f_temp(x, y, i);
                    self.set_f(xp, yp, i, f);
                }
            }
        }
    }

    /// Apply every declared domain boundary.
    ///
    /// Faces are visited x before y. Where two non-periodic faces meet, the
    /// corner node is therefore resolved by the y face — walls win over
    /// inlets/outlets in the usual channel setup, which is the standard choice.
    pub fn apply_boundaries(&mut self) {
        if self.boundaries.all_periodic() {
            return;
        }
        for axis in 0..2 {
            for side in [Side::Min, Side::Max] {
                let bc = self.boundaries.get(axis, side);
                if bc.is_periodic() {
                    continue;
                }
                // Inward normal and in-plane tangent.
                let sign = if side == Side::Min { 1 } else { -1 };
                let n = if axis == 0 { [sign, 0] } else { [0, sign] };
                let t = if axis == 0 { [0, 1] } else { [1, 0] };
                let (fixed, len) = if axis == 0 {
                    (if side == Side::Min { 0 } else { self.nx - 1 }, self.ny)
                } else {
                    (if side == Side::Min { 0 } else { self.ny - 1 }, self.nx)
                };

                for k in 0..len {
                    let (x, y) = if axis == 0 { (fixed, k) } else { (k, fixed) };
                    match bc {
                        Boundary::Periodic => {}
                        Boundary::NoSlip => self.bounce_back(x, y, n, [0.0, 0.0]),
                        Boundary::MovingWall(uw) => self.bounce_back(x, y, n, uw),
                        Boundary::Symmetry => self.specular(x, y, n, axis),
                        Boundary::Velocity(uw) => self.zou_he(x, y, n, t, ZouHe::Velocity(uw)),
                        Boundary::Pressure(rho) => self.zou_he(x, y, n, t, ZouHe::Pressure(rho)),
                    }
                }
            }
        }
    }

    /// Halfway bounce-back, optionally with wall motion.
    ///
    /// `f_i(t+1) = f̃_ī(t) + 2 w_i ρ_w (e_i · u_w) / c_s²` for the unknown
    /// directions, where `f̃` is the post-collision state. The wall plane lies
    /// half a lattice spacing outside the node.
    fn bounce_back(&mut self, x: usize, y: usize, n: [i32; 2], uw: [f64; 2]) {
        let rho_w: f64 = (0..9).map(|i| self.get_f_temp(x, y, i)).sum();
        let moving = uw[0] != 0.0 || uw[1] != 0.0;
        for i in 0..9 {
            if E[i][0] * n[0] + E[i][1] * n[1] <= 0 {
                continue;
            }
            let mut v = self.get_f_temp(x, y, OPP[i]);
            if moving {
                let eu = E[i][0] as f64 * uw[0] + E[i][1] as f64 * uw[1];
                v += 6.0 * W[i] * rho_w * eu;
            }
            self.set_f(x, y, i, v);
        }
    }

    /// Specular reflection — a free-slip / symmetry plane.
    fn specular(&mut self, x: usize, y: usize, n: [i32; 2], axis: usize) {
        for i in 0..9 {
            if E[i][0] * n[0] + E[i][1] * n[1] <= 0 {
                continue;
            }
            let v = self.get_f_temp(x, y, mirror_index(i, axis));
            self.set_f(x, y, i, v);
        }
    }

    /// Zou–He velocity / pressure boundary.
    ///
    /// Reconstructs the three unknown populations from the known ones by
    /// enforcing the density and momentum constraints plus the "bounce-back of
    /// the non-equilibrium normal part" closure. Unlike an equilibrium
    /// overwrite this keeps `f^neq`, so the viscous stress at the boundary is
    /// preserved and the condition is second-order accurate.
    fn zou_he(&mut self, x: usize, y: usize, n: [i32; 2], t: [i32; 2], kind: ZouHe) {
        let i_n = dir_index(n[0], n[1]);
        let i_t = dir_index(t[0], t[1]);
        let i_tm = dir_index(-t[0], -t[1]);
        let i_np = dir_index(n[0] + t[0], n[1] + t[1]);
        let i_nm = dir_index(n[0] - t[0], n[1] - t[1]);

        let f_t = self.get_f(x, y, i_t);
        let f_tm = self.get_f(x, y, i_tm);
        let f_rest = self.get_f(x, y, 0);
        let f_on = self.get_f(x, y, OPP[i_n]);
        let f_onp = self.get_f(x, y, OPP[i_np]);
        let f_onm = self.get_f(x, y, OPP[i_nm]);

        let sum_parallel = f_rest + f_t + f_tm;
        let sum_outgoing = f_on + f_onp + f_onm;

        let (rho, un, ut) = match kind {
            ZouHe::Velocity(uw) => {
                let un = uw[0] * n[0] as f64 + uw[1] * n[1] as f64;
                let ut = uw[0] * t[0] as f64 + uw[1] * t[1] as f64;
                ((sum_parallel + 2.0 * sum_outgoing) / (1.0 - un), un, ut)
            }
            ZouHe::Pressure(rho_w) => {
                let un = 1.0 - (sum_parallel + 2.0 * sum_outgoing) / rho_w;
                (rho_w, un, 0.0)
            }
        };

        // Transverse momentum carried by the in-plane populations.
        let nt = 0.5 * (f_t - f_tm);

        self.set_f(x, y, i_n, f_on + (2.0 / 3.0) * rho * un);
        self.set_f(x, y, i_np, f_onp + rho * un / 6.0 + 0.5 * rho * ut - nt);
        self.set_f(x, y, i_nm, f_onm + rho * un / 6.0 - 0.5 * rho * ut + nt);
    }

    /// One full LBM step: collision, streaming, boundary conditions.
    pub fn step(&mut self) {
        self.collide();
        self.stream();
        self.apply_boundaries();
    }

    /// Alias for [`Self::step`], kept for backwards compatibility.
    ///
    /// The name is now a misnomer — boundaries are applied too.
    pub fn collide_and_stream(&mut self) {
        self.step();
    }

    /// Advance `steps` steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Advance until the velocity field stops changing, or `max_steps` elapse.
    ///
    /// Returns the number of steps taken and the final residual
    /// `max|Δu| / max|u|`. Useful for driving a case to steady state without
    /// guessing a step count.
    pub fn run_to_steady_state(
        &mut self,
        tol: f64,
        max_steps: usize,
        check_every: usize,
    ) -> (usize, f64) {
        let check_every = check_every.max(1);
        let mut prev = self.velocity_field();
        let mut residual = f64::INFINITY;
        for s in 1..=max_steps {
            self.step();
            if s % check_every == 0 {
                let now = self.velocity_field();
                let mut du: f64 = 0.0;
                let mut umax: f64 = 0.0;
                for (a, b) in now.iter().zip(&prev) {
                    du = du.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs());
                    umax = umax.max(a[0].abs()).max(a[1].abs());
                }
                residual = if umax > 0.0 { du / umax } else { du };
                if residual < tol {
                    return (s, residual);
                }
                prev = now;
            }
        }
        (max_steps, residual)
    }

    /// Snapshot of the velocity at every node, row-major in `x`.
    pub fn velocity_field(&self) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(self.nx * self.ny);
        for y in 0..self.ny {
            for x in 0..self.nx {
                out.push(self.velocity(x, y));
            }
        }
        out
    }

    /// Total mass in the domain.
    pub fn total_mass(&self) -> f64 {
        (0..self.ny)
            .flat_map(|y| (0..self.nx).map(move |x| (x, y)))
            .map(|(x, y)| self.density(x, y))
            .sum()
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let rho = self.density(x, y);
                let u = self.velocity(x, y);
                ke += 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
            }
        }
        ke
    }
}

/// Single-relaxation-time collision with Guo forcing.
#[inline]
fn collide_bgk(fi: &[f64; 9], feq: &[f64; 9], src: &[f64; 9], tau: f64) -> [f64; 9] {
    let omega = 1.0 / tau;
    let pref = 1.0 - 0.5 * omega;
    std::array::from_fn(|i| fi[i] - omega * (fi[i] - feq[i]) + pref * src[i])
}

/// Two-relaxation-time collision with Guo forcing.
///
/// Populations are split into symmetric and antisymmetric parts about each
/// opposite pair; the symmetric rate sets the viscosity while the
/// antisymmetric rate is fixed by the magic parameter. The source term is split
/// the same way and each half gets its own `(1 - ω/2)` prefactor, which is what
/// keeps the forcing second-order accurate under TRT.
#[inline]
fn collide_trt(fi: &[f64; 9], feq: &[f64; 9], src: &[f64; 9], tau: f64, magic: f64) -> [f64; 9] {
    let omega_p = 1.0 / tau;
    let omega_m = trt_omega_minus(omega_p, magic);
    let pref_p = 1.0 - 0.5 * omega_p;
    let pref_m = 1.0 - 0.5 * omega_m;

    std::array::from_fn(|i| {
        let o = OPP[i];
        let neq_p = 0.5 * ((fi[i] - feq[i]) + (fi[o] - feq[o]));
        let neq_m = 0.5 * ((fi[i] - feq[i]) - (fi[o] - feq[o]));
        let src_p = 0.5 * (src[i] + src[o]);
        let src_m = 0.5 * (src[i] - src[o]);
        fi[i] - omega_p * neq_p - omega_m * neq_m + pref_p * src_p + pref_m * src_m
    })
}

/// Full multiple-relaxation-time collision in moment space, with Guo forcing.
///
/// `m* = m - S(m - m^eq) + (I - S/2) M S_guo`, then transformed back with
/// `M⁻¹ = Mᵀ / norms` (the basis is orthogonal). The ghost-mode rates follow
/// Lallemand & Luo; `s_q` is set to `8(2 - s_ν)/(8 - s_ν)`, the choice that
/// reproduces `Λ = 3/16`, so MRT inherits TRT's viscosity-independent wall.
#[inline]
fn collide_mrt(fi: &[f64; 9], feq: &[f64; 9], src: &[f64; 9], tau: f64) -> [f64; 9] {
    let s_nu = 1.0 / tau;
    let s_q = 8.0 * (2.0 - s_nu) / (8.0 - s_nu);
    let rates = [0.0, 1.64, 1.54, 0.0, s_q, 0.0, s_q, s_nu, s_nu];

    let mut post_m = [0.0f64; 9];
    for k in 0..9 {
        let (mut m, mut m_eq, mut m_src) = (0.0, 0.0, 0.0);
        for i in 0..9 {
            m += M[k][i] * fi[i];
            m_eq += M[k][i] * feq[i];
            m_src += M[k][i] * src[i];
        }
        post_m[k] = m - rates[k] * (m - m_eq) + (1.0 - 0.5 * rates[k]) * m_src;
    }

    std::array::from_fn(|i| {
        let mut v = 0.0;
        for k in 0..9 {
            v += M[k][i] * post_m[k] / M_NORM[k];
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic;
    use crate::boundary;
    use crate::collision::MAGIC_BOUNCE_BACK;

    #[test]
    fn test_equilibrium_rest() {
        let feq = equilibrium(0, 1.0, [0.0, 0.0]);
        assert!((feq - W[0]).abs() < 1e-12);
    }

    #[test]
    fn equilibrium_recovers_its_own_moments() {
        let (rho, u) = (1.03, [0.05, -0.02]);
        let feq: [f64; 9] = std::array::from_fn(|i| equilibrium(i, rho, u));
        let m0: f64 = feq.iter().sum();
        let mx: f64 = (0..9).map(|i| feq[i] * E[i][0] as f64).sum();
        let my: f64 = (0..9).map(|i| feq[i] * E[i][1] as f64).sum();
        assert!((m0 - rho).abs() < 1e-14);
        assert!((mx - rho * u[0]).abs() < 1e-14);
        assert!((my - rho * u[1]).abs() < 1e-14);
    }

    #[test]
    fn mrt_basis_is_orthogonal_with_the_stated_norms() {
        for a in 0..9 {
            for b in 0..9 {
                let dot: f64 = (0..9).map(|i| M[a][i] * M[b][i]).sum();
                let want = if a == b { M_NORM[a] } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "M[{a}]·M[{b}] = {dot}");
            }
        }
    }

    #[test]
    fn trt_with_bgk_magic_reproduces_bgk() {
        let (rho, u) = (1.01, [0.03, 0.01]);
        let feq: [f64; 9] = std::array::from_fn(|i| equilibrium(i, rho, u));
        let fi: [f64; 9] = std::array::from_fn(|i| feq[i] * (1.0 + 0.01 * i as f64));
        let src: [f64; 9] = std::array::from_fn(|i| guo_source(i, u, [1e-5, 0.0]));
        let tau = 0.9;
        let bgk = collide_bgk(&fi, &feq, &src, tau);
        let trt = collide_trt(&fi, &feq, &src, tau, (tau - 0.5) * (tau - 0.5));
        for i in 0..9 {
            assert!((bgk[i] - trt[i]).abs() < 1e-14, "dir {i}");
        }
    }

    #[test]
    fn every_collision_operator_conserves_mass_and_momentum() {
        // Perturb first, then build f^eq from the *perturbed* moments — that is
        // the invariant collision actually has to respect.
        let base: [f64; 9] = std::array::from_fn(|i| equilibrium(i, 1.0, [0.04, -0.03]));
        let fi: [f64; 9] = std::array::from_fn(|i| base[i] * (1.0 + 0.02 * (i as f64 - 4.0)));
        let rho: f64 = fi.iter().sum();
        let u = [
            (0..9).map(|i| fi[i] * E[i][0] as f64).sum::<f64>() / rho,
            (0..9).map(|i| fi[i] * E[i][1] as f64).sum::<f64>() / rho,
        ];
        let feq: [f64; 9] = std::array::from_fn(|i| equilibrium(i, rho, u));
        let src = [0.0f64; 9];
        let m0: f64 = fi.iter().sum();
        let px: f64 = (0..9).map(|i| fi[i] * E[i][0] as f64).sum();

        for post in [
            collide_bgk(&fi, &feq, &src, 0.8),
            collide_trt(&fi, &feq, &src, 0.8, MAGIC_BOUNCE_BACK),
            collide_mrt(&fi, &feq, &src, 0.8),
        ] {
            let m1: f64 = post.iter().sum();
            let px1: f64 = (0..9).map(|i| post[i] * E[i][0] as f64).sum();
            assert!((m1 - m0).abs() < 1e-12, "mass drift {m1} vs {m0}");
            assert!((px1 - px).abs() < 1e-12, "momentum drift {px1} vs {px}");
        }
    }

    #[test]
    fn mrt_round_trips_an_equilibrium_state() {
        // At equilibrium with no force, collision is the identity.
        let feq: [f64; 9] = std::array::from_fn(|i| equilibrium(i, 1.0, [0.02, 0.01]));
        let post = collide_mrt(&feq, &feq, &[0.0; 9], 0.75);
        for i in 0..9 {
            assert!((post[i] - feq[i]).abs() < 1e-13, "dir {i}");
        }
    }

    #[test]
    fn test_density_conservation() {
        let mut lbm = LatticeBoltzmann2D::new(32, 32, 0.1);
        lbm.initialize_uniform(1.0, [0.01, 0.0]);
        let rho0 = lbm.total_mass();
        lbm.run(100);
        let rho1 = lbm.total_mass();
        assert!(
            (rho1 - rho0).abs() / rho0 < 1e-10,
            "Density not conserved: {rho0} vs {rho1}"
        );
    }

    #[test]
    fn walls_conserve_mass() {
        let mut lbm =
            LatticeBoltzmann2D::new(24, 24, 0.08).with_boundaries(boundary::cavity_2d([0.05, 0.0]));
        lbm.initialize_uniform(1.0, [0.0, 0.0]);
        let m0 = lbm.total_mass();
        lbm.run(300);
        let m1 = lbm.total_mass();
        assert!((m1 - m0).abs() / m0 < 1e-9, "mass drift {m0} -> {m1}");
    }

    #[test]
    fn test_viscous_dissipation() {
        let mut lbm = LatticeBoltzmann2D::new(32, 32, 0.1);
        lbm.initialize_with(|_, y| {
            let ux = if y < 16 { 0.1 } else { -0.1 };
            (1.0, [ux, 0.0])
        });
        let ke0 = lbm.kinetic_energy();
        lbm.run(1000);
        let ke1 = lbm.kinetic_energy();
        assert!(ke1 < ke0 * 0.9, "KE should decrease: {ke0} -> {ke1}");
    }

    #[test]
    fn no_slip_walls_bring_the_fluid_to_rest() {
        // Uniform flow between stationary walls must decay to nothing.
        let mut lbm = LatticeBoltzmann2D::new(8, 16, 0.1).with_boundaries(boundary::channel_2d());
        lbm.initialize_uniform(1.0, [0.05, 0.0]);
        lbm.run(4000);
        assert!(lbm.max_velocity() < 1e-4, "u_max = {}", lbm.max_velocity());
    }

    #[test]
    fn symmetry_walls_do_not_drag_the_fluid() {
        // Free-slip walls exert no shear, so uniform flow must persist.
        let bc = boundary::Boundaries::periodic().set_axis(1, Boundary::Symmetry);
        let mut lbm = LatticeBoltzmann2D::new(8, 16, 0.05).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.05, 0.0]);
        lbm.run(2000);
        for y in 0..lbm.ny {
            let u = lbm.velocity(3, y);
            assert!((u[0] - 0.05).abs() < 1e-6, "y={y}: {:?}", u);
            assert!(u[1].abs() < 1e-8, "y={y}: {:?}", u);
        }
    }

    #[test]
    fn a_body_force_accelerates_unbounded_fluid_at_the_newtonian_rate() {
        // Periodic box, no walls: du/dt = F/ρ exactly.
        let g = 1e-5;
        let mut lbm = LatticeBoltzmann2D::new(8, 8, 0.05).with_force([g, 0.0]);
        lbm.initialize_uniform(1.0, [0.0, 0.0]);
        let steps = 500;
        lbm.run(steps);
        let u = lbm.velocity(4, 4)[0];
        // Guo's velocity is defined at the half step, hence the +1/2.
        let expect = g * (steps as f64 + 0.5);
        assert!(
            (u - expect).abs() / expect < 1e-6,
            "u = {u}, expected {expect}"
        );
    }

    #[test]
    fn zou_he_velocity_inlet_imposes_the_requested_velocity() {
        let bc = boundary::inlet_outlet_channel_2d([0.02, 0.0], 1.0);
        let mut lbm = LatticeBoltzmann2D::new(24, 12, 0.06).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.02, 0.0]);
        lbm.run(500);
        // Interior of the inlet face (skip the two wall nodes).
        for y in 1..lbm.ny - 1 {
            let u = lbm.velocity(0, y);
            assert!((u[0] - 0.02).abs() < 1e-9, "y={y}: ux={}", u[0]);
        }
    }

    #[test]
    fn pressure_outlet_holds_its_density() {
        let bc = boundary::inlet_outlet_channel_2d([0.02, 0.0], 1.0);
        let mut lbm = LatticeBoltzmann2D::new(24, 12, 0.06).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.02, 0.0]);
        lbm.run(500);
        for y in 1..lbm.ny - 1 {
            let rho = lbm.density(lbm.nx - 1, y);
            assert!((rho - 1.0).abs() < 1e-9, "y={y}: rho={rho}");
        }
    }

    #[test]
    fn set_velocity_bc_preserves_the_non_equilibrium_part() {
        let mut lbm = LatticeBoltzmann2D::new(8, 8, 0.05);
        lbm.initialize_with(|x, _| (1.0, [0.01 * x as f64, 0.0]));
        lbm.run(5); // build up a real f^neq
        let donor = (1usize, 4usize);
        let rho = lbm.density(donor.0, donor.1);
        let u_donor = lbm.velocity(donor.0, donor.1);
        let neq_before: Vec<f64> = (0..9)
            .map(|i| lbm.get_f(donor.0, donor.1, i) - equilibrium(i, rho, u_donor))
            .collect();
        assert!(
            neq_before.iter().any(|v| v.abs() > 1e-9),
            "test is vacuous without a non-equilibrium part"
        );

        lbm.set_velocity_bc(0, 4, [0.5, 0.0]);
        let rho_bc = lbm.density(0, 4);
        let u_bc = lbm.velocity(0, 4);
        for i in 0..9 {
            let neq_after = lbm.get_f(0, 4, i) - equilibrium(i, rho_bc, u_bc);
            assert!(
                (neq_after - neq_before[i]).abs() < 1e-6,
                "dir {i}: f^neq {} -> {neq_after}",
                neq_before[i]
            );
        }
    }

    #[test]
    fn smagorinsky_is_inactive_on_a_uniform_field() {
        use crate::collision::Turbulence;
        let mut a = LatticeBoltzmann2D::new(16, 16, 0.05);
        let mut b = LatticeBoltzmann2D::new(16, 16, 0.05)
            .with_turbulence(Turbulence::Smagorinsky { cs: 0.16 });
        a.initialize_uniform(1.0, [0.03, 0.0]);
        b.initialize_uniform(1.0, [0.03, 0.0]);
        a.run(50);
        b.run(50);
        for (fa, fb) in a.f.iter().zip(&b.f) {
            assert!((fa - fb).abs() < 1e-13);
        }
    }

    #[test]
    fn smagorinsky_damps_a_sheared_field_more_than_dns() {
        use crate::collision::Turbulence;
        let init = |lbm: &mut LatticeBoltzmann2D| {
            lbm.initialize_with(|_, y| {
                let ux = if y < 16 { 0.1 } else { -0.1 };
                (1.0, [ux, 0.0])
            });
        };
        let mut dns = LatticeBoltzmann2D::new(32, 32, 0.02);
        let mut les = LatticeBoltzmann2D::new(32, 32, 0.02)
            .with_turbulence(Turbulence::Smagorinsky { cs: 0.16 });
        init(&mut dns);
        init(&mut les);
        dns.run(200);
        les.run(200);
        assert!(
            les.kinetic_energy() < dns.kinetic_energy(),
            "LES should add dissipation: {} vs {}",
            les.kinetic_energy(),
            dns.kinetic_energy()
        );
    }

    #[test]
    fn poiseuille_is_exact_under_trt() {
        // The headline result: with Λ = 3/16 the bounce-back wall sits exactly
        // halfway between nodes, so the discrete profile is the analytic one.
        let (ny, nu, u_peak) = (17usize, 0.05, 0.02);
        let h = ny as f64;
        let g = analytic::poiseuille_force_for_peak(h, u_peak, nu);
        let mut lbm = LatticeBoltzmann2D::new(4, ny, nu)
            .with_boundaries(boundary::channel_2d())
            .with_force([g, 0.0]);
        lbm.initialize_uniform(1.0, [0.0, 0.0]);
        lbm.run(40_000);

        for y in 0..ny {
            let want = analytic::poiseuille_force_driven(y as f64, -0.5, h - 0.5, g, nu);
            let got = lbm.velocity(0, y)[0];
            assert!(
                (got - want).abs() / u_peak < 1e-6,
                "y={y}: got {got}, want {want}"
            );
        }
    }
}
