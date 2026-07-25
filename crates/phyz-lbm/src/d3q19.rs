//! D3Q19 Lattice Boltzmann solver for 3D incompressible flow.
//!
//! Nineteen velocity directions on 3D cubic lattice:
//! - 1 rest (0)
//! - 6 face-centered (±x, ±y, ±z)
//! - 12 edge-centered (±x±y, ±y±z, ±z±x)
//!
//! Feature parity with [`crate::d2q9`] apart from the collision operator:
//! BGK and TRT are available, but [`CollisionModel::Mrt`] falls back to TRT
//! with [`crate::collision::MAGIC_BOUNCE_BACK`]. The 19×19 moment transform
//! costs an order of magnitude more per node than TRT and buys nothing for wall
//! accuracy, which is what the magic parameter already fixes; it is worth
//! adding only if independent bulk-viscosity control is needed.
//!
//! Velocity and pressure faces use Guo's non-equilibrium extrapolation rather
//! than Zou–He. In 3D a Zou–He face has five unknowns and needs an
//! under-determined transverse-momentum closure; the extrapolation scheme is
//! second-order, unambiguous, and equally preserves `f^neq`.

use crate::boundary::{Boundaries, Boundary, Side};
use crate::collision::{
    CollisionModel, MAGIC_BOUNCE_BACK, Turbulence, smagorinsky_tau, trt_omega_minus,
};

/// D3Q19 discrete velocities: [vx, vy, vz]
pub const E: [[i32; 3]; 19] = [
    [0, 0, 0], // 0: rest
    [1, 0, 0], // 1-6: face
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0], // 7-18: edge
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
];

/// D3Q19 weights
pub const W: [f64; 19] = [
    1.0 / 3.0,  // 0: rest
    1.0 / 18.0, // 1-6: face
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0, // 7-18: edge
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Opposite direction indices for bounce-back
pub const OPP: [usize; 19] = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
];

/// Index of the lattice direction equal to `(ex, ey, ez)`.
#[inline]
fn dir_index(e: [i32; 3]) -> usize {
    let mut i = 0;
    while i < 19 {
        if E[i][0] == e[0] && E[i][1] == e[1] && E[i][2] == e[2] {
            return i;
        }
        i += 1;
    }
    panic!("({}, {}, {}) is not a D3Q19 direction", e[0], e[1], e[2])
}

/// Direction obtained by flipping component `axis` — specular reflection.
#[inline]
fn mirror_index(i: usize, axis: usize) -> usize {
    let mut e = E[i];
    e[axis] = -e[axis];
    dir_index(e)
}

/// Equilibrium distribution `f_i^eq`.
#[inline]
pub fn equilibrium(i: usize, rho: f64, u: [f64; 3]) -> f64 {
    let e = E[i];
    let eu = e[0] as f64 * u[0] + e[1] as f64 * u[1] + e[2] as f64 * u[2];
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
}

/// Guo's forcing source term, before the `(1 - ω/2)` prefactor.
#[inline]
fn guo_source(i: usize, u: [f64; 3], force: [f64; 3]) -> f64 {
    let e = E[i];
    let eu = e[0] as f64 * u[0] + e[1] as f64 * u[1] + e[2] as f64 * u[2];
    let ef = e[0] as f64 * force[0] + e[1] as f64 * force[1] + e[2] as f64 * force[2];
    let du: f64 = (0..3).map(|d| (E[i][d] as f64 - u[d]) * force[d]).sum();
    W[i] * (3.0 * du + 9.0 * eu * ef)
}

/// D3Q19 Lattice Boltzmann solver for 3D flow.
pub struct LatticeBoltzmann3D {
    /// Grid size in x
    pub nx: usize,
    /// Grid size in y
    pub ny: usize,
    /// Grid size in z
    pub nz: usize,
    /// Kinematic viscosity
    pub nu: f64,
    /// Relaxation time τ = 3ν + 0.5
    pub tau: f64,
    /// Distribution functions f_i at each grid point
    /// Shape: [nx, ny, nz, 19]
    pub f: Vec<f64>,
    /// Post-collision, pre-streaming state. Boundary conditions read from it.
    f_temp: Vec<f64>,
    /// Collision operator. Defaults to TRT with the bounce-back magic number.
    pub collision: CollisionModel,
    /// Sub-grid turbulence closure. Defaults to none.
    pub turbulence: Turbulence,
    /// Uniform body force per unit volume, applied with Guo's scheme.
    pub force: [f64; 3],
    /// Domain boundary conditions, applied automatically by [`Self::step`].
    pub boundaries: Boundaries<3>,
}

impl LatticeBoltzmann3D {
    /// Create new 3D LBM solver.
    ///
    /// Defaults: TRT collision with `Λ = 3/16`, no turbulence model, no body
    /// force, fully periodic boundaries.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `nu` - Kinematic viscosity
    pub fn new(nx: usize, ny: usize, nz: usize, nu: f64) -> Self {
        let tau = 3.0 * nu + 0.5;
        let size = nx * ny * nz * 19;
        Self {
            nx,
            ny,
            nz,
            nu,
            tau,
            f: vec![0.0; size],
            f_temp: vec![0.0; size],
            collision: CollisionModel::default(),
            turbulence: Turbulence::default(),
            force: [0.0; 3],
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
    pub fn with_force(mut self, force: [f64; 3]) -> Self {
        self.force = force;
        self
    }

    /// Declare the domain boundaries. Chainable.
    #[must_use]
    pub fn with_boundaries(mut self, boundaries: Boundaries<3>) -> Self {
        self.boundaries = boundaries;
        self
    }

    /// Change the viscosity, keeping `tau` consistent.
    pub fn set_viscosity(&mut self, nu: f64) {
        self.nu = nu;
        self.tau = 3.0 * nu + 0.5;
    }

    /// Initialize with uniform density and velocity.
    pub fn initialize_uniform(&mut self, rho: f64, u: [f64; 3]) {
        self.initialize_with(|_, _, _| (rho, u));
    }

    /// Initialize from a field function returning `(ρ, u)` at each node.
    pub fn initialize_with<F>(&mut self, field: F)
    where
        F: Fn(usize, usize, usize) -> (f64, [f64; 3]),
    {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let (rho, u) = field(x, y, z);
                    for i in 0..19 {
                        self.set_f(x, y, z, i, equilibrium(i, rho, u));
                    }
                }
            }
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize, i: usize) -> usize {
        x + y * self.nx + z * self.nx * self.ny + i * self.nx * self.ny * self.nz
    }

    /// Get distribution function at (x, y, z, i).
    #[inline]
    fn get_f(&self, x: usize, y: usize, z: usize, i: usize) -> f64 {
        self.f[self.index(x, y, z, i)]
    }

    /// Set distribution function at (x, y, z, i).
    #[inline]
    fn set_f(&mut self, x: usize, y: usize, z: usize, i: usize, val: f64) {
        let idx = self.index(x, y, z, i);
        self.f[idx] = val;
    }

    /// Get temporary distribution function at (x, y, z, i).
    #[inline]
    fn get_f_temp(&self, x: usize, y: usize, z: usize, i: usize) -> f64 {
        self.f_temp[self.index(x, y, z, i)]
    }

    /// Set temporary distribution function at (x, y, z, i).
    #[inline]
    fn set_f_temp(&mut self, x: usize, y: usize, z: usize, i: usize, val: f64) {
        let idx = self.index(x, y, z, i);
        self.f_temp[idx] = val;
    }

    /// Compute macroscopic density at (x, y, z).
    pub fn density(&self, x: usize, y: usize, z: usize) -> f64 {
        (0..19).map(|i| self.get_f(x, y, z, i)).sum()
    }

    /// Compute macroscopic velocity at (x, y, z).
    ///
    /// Includes the `F/2` correction required by Guo forcing.
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> [f64; 3] {
        let mut rho = 0.0;
        let mut u = [0.0; 3];
        for (i, e) in E.iter().enumerate() {
            let f = self.get_f(x, y, z, i);
            rho += f;
            for d in 0..3 {
                u[d] += f * e[d] as f64;
            }
        }
        if rho < 1e-12 {
            return [0.0; 3];
        }
        std::array::from_fn(|d| (u[d] + 0.5 * self.force[d]) / rho)
    }

    /// Maximum velocity magnitude in domain.
    pub fn max_velocity(&self) -> f64 {
        let mut umax: f64 = 0.0;
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let u = self.velocity(x, y, z);
                    umax = umax.max((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt());
                }
            }
        }
        umax
    }

    /// Impose a velocity at a single node, preserving the non-equilibrium part.
    ///
    /// Guo's non-equilibrium extrapolation. Replaces the old behaviour of
    /// overwriting every population with its equilibrium value, which discarded
    /// `f^neq` — and with it the viscous stress — at the boundary.
    pub fn set_velocity_bc(&mut self, x: usize, y: usize, z: usize, u: [f64; 3]) {
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
        let dz = if z == 0 {
            1
        } else if z + 1 == self.nz {
            -1
        } else {
            0
        };
        let s = (
            (x as i32 + dx) as usize,
            (y as i32 + dy) as usize,
            (z as i32 + dz) as usize,
        );
        self.extrapolate_from(x, y, z, s, None, Some(u));
    }

    /// Set no-slip boundary condition (full-way bounce-back) at (x, y, z).
    ///
    /// Node-local helper for interior obstacles. Domain faces are better
    /// declared as [`Boundary::NoSlip`], which uses halfway bounce-back.
    pub fn set_no_slip_bc(&mut self, x: usize, y: usize, z: usize) {
        let f_new: [f64; 19] = std::array::from_fn(|i| self.get_f(x, y, z, OPP[i]));
        for (i, &val) in f_new.iter().enumerate() {
            self.set_f(x, y, z, i, val);
        }
    }

    /// Effective relaxation time at a node, including any eddy viscosity.
    #[inline]
    fn effective_tau(&self, fi: &[f64; 19], feq: &[f64; 19], rho: f64) -> f64 {
        match self.turbulence {
            Turbulence::None => self.tau,
            Turbulence::Smagorinsky { cs } => {
                let mut q = [[0.0f64; 3]; 3];
                for i in 0..19 {
                    let n = fi[i] - feq[i];
                    for a in 0..3 {
                        for b in 0..3 {
                            q[a][b] += E[i][a] as f64 * E[i][b] as f64 * n;
                        }
                    }
                }
                let qq: f64 = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .map(|(a, b)| q[a][b] * q[a][b])
                    .sum();
                smagorinsky_tau(self.tau, rho, (2.0 * qq).sqrt(), cs)
            }
        }
    }

    /// Collision step. Writes the post-collision state into `f_temp`.
    fn collide(&mut self) {
        let force = self.force;
        let forced = force.iter().any(|&c| c != 0.0);
        // MRT is not implemented for D3Q19; TRT with Λ = 3/16 is equivalent
        // for wall accuracy and far cheaper. See the module docs.
        let model = match self.collision {
            CollisionModel::Mrt => CollisionModel::Trt {
                magic: MAGIC_BOUNCE_BACK,
            },
            other => other,
        };

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let mut fi = [0.0f64; 19];
                    let mut rho = 0.0;
                    let mut mom = [0.0f64; 3];
                    for i in 0..19 {
                        let v = self.get_f(x, y, z, i);
                        fi[i] = v;
                        rho += v;
                        for d in 0..3 {
                            mom[d] += v * E[i][d] as f64;
                        }
                    }
                    let inv_rho = if rho.abs() < 1e-12 { 0.0 } else { 1.0 / rho };
                    let u: [f64; 3] = std::array::from_fn(|d| (mom[d] + 0.5 * force[d]) * inv_rho);

                    let feq: [f64; 19] = std::array::from_fn(|i| equilibrium(i, rho, u));
                    let src: [f64; 19] = if forced {
                        std::array::from_fn(|i| guo_source(i, u, force))
                    } else {
                        [0.0; 19]
                    };
                    let tau = self.effective_tau(&fi, &feq, rho);

                    let post: [f64; 19] = match model {
                        CollisionModel::Bgk => {
                            let omega = 1.0 / tau;
                            let pref = 1.0 - 0.5 * omega;
                            std::array::from_fn(|i| {
                                fi[i] - omega * (fi[i] - feq[i]) + pref * src[i]
                            })
                        }
                        CollisionModel::Trt { magic } => {
                            let omega_p = 1.0 / tau;
                            let omega_m = trt_omega_minus(omega_p, magic);
                            let pref_p = 1.0 - 0.5 * omega_p;
                            let pref_m = 1.0 - 0.5 * omega_m;
                            std::array::from_fn(|i| {
                                let o = OPP[i];
                                let np = 0.5 * ((fi[i] - feq[i]) + (fi[o] - feq[o]));
                                let nm = 0.5 * ((fi[i] - feq[i]) - (fi[o] - feq[o]));
                                let sp = 0.5 * (src[i] + src[o]);
                                let sm = 0.5 * (src[i] - src[o]);
                                fi[i] - omega_p * np - omega_m * nm + pref_p * sp + pref_m * sm
                            })
                        }
                        CollisionModel::Mrt => unreachable!("mapped to TRT above"),
                    };

                    for i in 0..19 {
                        self.set_f_temp(x, y, z, i, post[i]);
                    }
                }
            }
        }
    }

    /// Streaming step. Wraps; non-periodic faces are fixed up afterwards.
    fn stream(&mut self) {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    for (i, e) in E.iter().enumerate() {
                        let xp = (x as i32 + e[0]).rem_euclid(self.nx as i32) as usize;
                        let yp = (y as i32 + e[1]).rem_euclid(self.ny as i32) as usize;
                        let zp = (z as i32 + e[2]).rem_euclid(self.nz as i32) as usize;
                        let f = self.get_f_temp(x, y, z, i);
                        self.set_f(xp, yp, zp, i, f);
                    }
                }
            }
        }
    }

    /// Apply every declared domain boundary. Faces are visited x, then y, then z.
    pub fn apply_boundaries(&mut self) {
        if self.boundaries.all_periodic() {
            return;
        }
        let dims = [self.nx, self.ny, self.nz];
        for axis in 0..3 {
            for side in [Side::Min, Side::Max] {
                let bc = self.boundaries.get(axis, side);
                if bc.is_periodic() {
                    continue;
                }
                let sign = if side == Side::Min { 1 } else { -1 };
                let mut n = [0i32; 3];
                n[axis] = sign;
                let fixed = if side == Side::Min { 0 } else { dims[axis] - 1 };
                let (a1, a2) = match axis {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };

                for p in 0..dims[a1] {
                    for q in 0..dims[a2] {
                        let mut c = [0usize; 3];
                        c[axis] = fixed;
                        c[a1] = p;
                        c[a2] = q;
                        let (x, y, zc) = (c[0], c[1], c[2]);
                        match bc {
                            Boundary::Periodic => {}
                            Boundary::NoSlip => self.bounce_back(x, y, zc, n, [0.0; 3]),
                            Boundary::MovingWall(uw) => self.bounce_back(x, y, zc, n, uw),
                            Boundary::Symmetry => self.specular(x, y, zc, n, axis),
                            Boundary::Velocity(uw) => {
                                let s = Self::inward(c, axis, sign);
                                self.extrapolate_from(x, y, zc, s, None, Some(uw));
                            }
                            Boundary::Pressure(rho) => {
                                let s = Self::inward(c, axis, sign);
                                self.extrapolate_from(x, y, zc, s, Some(rho), None);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The interior neighbour one step along the inward normal.
    #[inline]
    fn inward(c: [usize; 3], axis: usize, sign: i32) -> (usize, usize, usize) {
        let mut s = c;
        s[axis] = (c[axis] as i32 + sign) as usize;
        (s[0], s[1], s[2])
    }

    /// Halfway bounce-back, optionally with wall motion.
    fn bounce_back(&mut self, x: usize, y: usize, z: usize, n: [i32; 3], uw: [f64; 3]) {
        let rho_w: f64 = (0..19).map(|i| self.get_f_temp(x, y, z, i)).sum();
        let moving = uw.iter().any(|&c| c != 0.0);
        for i in 0..19 {
            let dot: i32 = (0..3).map(|d| E[i][d] * n[d]).sum();
            if dot <= 0 {
                continue;
            }
            let mut v = self.get_f_temp(x, y, z, OPP[i]);
            if moving {
                let eu: f64 = (0..3).map(|d| E[i][d] as f64 * uw[d]).sum();
                v += 6.0 * W[i] * rho_w * eu;
            }
            self.set_f(x, y, z, i, v);
        }
    }

    /// Specular reflection — a free-slip / symmetry plane.
    fn specular(&mut self, x: usize, y: usize, z: usize, n: [i32; 3], axis: usize) {
        for i in 0..19 {
            let dot: i32 = (0..3).map(|d| E[i][d] * n[d]).sum();
            if dot <= 0 {
                continue;
            }
            let v = self.get_f_temp(x, y, z, mirror_index(i, axis));
            self.set_f(x, y, z, i, v);
        }
    }

    /// Guo's non-equilibrium extrapolation.
    ///
    /// `f_i = f_i^eq(ρ, u) + [f_i(nb) - f_i^eq(ρ_nb, u_nb)]`. Whichever of `ρ`
    /// and `u` is not prescribed is taken from the interior neighbour `src`.
    /// The bracketed term is `f^neq`, carried over verbatim — that is the whole
    /// point of the scheme.
    fn extrapolate_from(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        src: (usize, usize, usize),
        rho_target: Option<f64>,
        u_target: Option<[f64; 3]>,
    ) {
        let (sx, sy, sz) = src;
        let rho_nb = self.density(sx, sy, sz);
        let u_nb = self.velocity(sx, sy, sz);
        let rho = rho_target.unwrap_or(rho_nb);
        let u = u_target.unwrap_or(u_nb);
        for i in 0..19 {
            let f_neq = self.get_f(sx, sy, sz, i) - equilibrium(i, rho_nb, u_nb);
            self.set_f(x, y, z, i, equilibrium(i, rho, u) + f_neq);
        }
    }

    /// One full LBM step: collision, streaming, boundary conditions.
    pub fn step(&mut self) {
        self.collide();
        self.stream();
        self.apply_boundaries();
    }

    /// Alias for [`Self::step`], kept for backwards compatibility.
    pub fn collide_and_stream(&mut self) {
        self.step();
    }

    /// Advance `steps` steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Total mass in the domain.
    pub fn total_mass(&self) -> f64 {
        let mut m = 0.0;
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    m += self.density(x, y, z);
                }
            }
        }
        m
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let rho = self.density(x, y, z);
                    let u = self.velocity(x, y, z);
                    ke += 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                }
            }
        }
        ke
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic;
    use crate::boundary::Boundaries;

    #[test]
    fn test_equilibrium_rest() {
        let feq = equilibrium(0, 1.0, [0.0, 0.0, 0.0]);
        assert!((feq - W[0]).abs() < 1e-12);
    }

    #[test]
    fn weights_and_velocities_satisfy_the_lattice_isotropy_conditions() {
        let w: f64 = W.iter().sum();
        assert!((w - 1.0).abs() < 1e-14);
        for d in 0..3 {
            let first: f64 = (0..19).map(|i| W[i] * E[i][d] as f64).sum();
            assert!(first.abs() < 1e-14);
        }
        for a in 0..3 {
            for b in 0..3 {
                let second: f64 = (0..19)
                    .map(|i| W[i] * E[i][a] as f64 * E[i][b] as f64)
                    .sum();
                let want = if a == b { 1.0 / 3.0 } else { 0.0 };
                assert!((second - want).abs() < 1e-14, "({a},{b}) = {second}");
            }
        }
    }

    #[test]
    fn opposites_are_involutive_and_actually_opposite() {
        for i in 0..19 {
            assert_eq!(OPP[OPP[i]], i);
            for d in 0..3 {
                assert_eq!(E[OPP[i]][d], -E[i][d]);
            }
        }
    }

    #[test]
    fn test_density_conservation() {
        let mut lbm = LatticeBoltzmann3D::new(16, 16, 16, 0.1);
        lbm.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
        let rho0 = lbm.total_mass();
        lbm.run(100);
        let rho1 = lbm.total_mass();
        assert!((rho1 - rho0).abs() / rho0 < 1e-10, "Density not conserved");
    }

    #[test]
    fn no_slip_walls_conserve_mass_and_stop_the_flow() {
        let bc = Boundaries::periodic().set_axis(1, Boundary::NoSlip);
        let mut lbm = LatticeBoltzmann3D::new(6, 12, 6, 0.1).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.05, 0.0, 0.0]);
        let m0 = lbm.total_mass();
        lbm.run(2500);
        assert!((lbm.total_mass() - m0).abs() / m0 < 1e-9);
        assert!(lbm.max_velocity() < 1e-4, "u_max = {}", lbm.max_velocity());
    }

    #[test]
    fn body_force_accelerates_unbounded_fluid_at_the_newtonian_rate() {
        let g = 1e-5;
        let mut lbm = LatticeBoltzmann3D::new(6, 6, 6, 0.05).with_force([g, 0.0, 0.0]);
        lbm.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
        lbm.run(200);
        let u = lbm.velocity(3, 3, 3)[0];
        let expect = g * 200.5; // Guo's velocity lives at the half step
        assert!((u - expect).abs() / expect < 1e-6, "u = {u}, want {expect}");
    }

    #[test]
    fn poiseuille_slab_matches_the_analytic_parabola() {
        // Walls along y, periodic in x and z: the 3D problem reduces to the
        // same parabola, and TRT reproduces it to round-off.
        let (ny, nu, u_peak) = (13usize, 0.06, 0.02);
        let h = ny as f64;
        let g = analytic::poiseuille_force_for_peak(h, u_peak, nu);
        let bc = Boundaries::periodic().set_axis(1, Boundary::NoSlip);
        let mut lbm = LatticeBoltzmann3D::new(3, ny, 3, nu)
            .with_boundaries(bc)
            .with_force([g, 0.0, 0.0]);
        lbm.initialize_uniform(1.0, [0.0; 3]);
        lbm.run(12_000);

        for y in 0..ny {
            let want = analytic::poiseuille_force_driven(y as f64, -0.5, h - 0.5, g, nu);
            let got = lbm.velocity(1, y, 1)[0];
            assert!(
                (got - want).abs() / u_peak < 1e-5,
                "y={y}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn symmetry_planes_do_not_drag_the_fluid() {
        let bc = Boundaries::periodic().set_axis(2, Boundary::Symmetry);
        let mut lbm = LatticeBoltzmann3D::new(6, 6, 10, 0.05).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.05, 0.0, 0.0]);
        lbm.run(500);
        for z in 0..lbm.nz {
            let u = lbm.velocity(2, 2, z);
            assert!((u[0] - 0.05).abs() < 1e-6, "z={z}: {u:?}");
            assert!(u[2].abs() < 1e-8, "z={z}: {u:?}");
        }
    }

    #[test]
    fn velocity_face_imposes_its_velocity_and_keeps_the_non_equilibrium_part() {
        let bc = Boundaries::periodic()
            .set(0, Side::Min, Boundary::Velocity([0.02, 0.0, 0.0]))
            .set(0, Side::Max, Boundary::Pressure(1.0));
        let mut lbm = LatticeBoltzmann3D::new(12, 6, 6, 0.06).with_boundaries(bc);
        lbm.initialize_uniform(1.0, [0.02, 0.0, 0.0]);
        lbm.run(200);
        for y in 0..lbm.ny {
            let u = lbm.velocity(0, y, 3);
            assert!((u[0] - 0.02).abs() < 1e-9, "y={y}: {u:?}");
            let rho = lbm.density(lbm.nx - 1, y, 3);
            assert!((rho - 1.0).abs() < 1e-9, "outlet rho = {rho}");
        }
    }

    #[test]
    fn mrt_falls_back_to_trt_rather_than_silently_misbehaving() {
        let make = |model| {
            let mut lbm = LatticeBoltzmann3D::new(8, 8, 8, 0.05).with_collision(model);
            lbm.initialize_with(|x, _, _| (1.0, [0.02 * (x as f64 / 8.0).sin(), 0.0, 0.0]));
            lbm.run(20);
            lbm.f
        };
        let mrt = make(CollisionModel::Mrt);
        let trt = make(CollisionModel::Trt {
            magic: MAGIC_BOUNCE_BACK,
        });
        assert!(mrt.iter().zip(&trt).all(|(a, b)| (a - b).abs() < 1e-14));
    }

    #[test]
    fn smagorinsky_adds_dissipation_in_3d() {
        use crate::collision::Turbulence;
        let init = |lbm: &mut LatticeBoltzmann3D| {
            lbm.initialize_with(|_, y, _| (1.0, [if y < 8 { 0.1 } else { -0.1 }, 0.0, 0.0]));
        };
        let mut dns = LatticeBoltzmann3D::new(16, 16, 4, 0.02);
        let mut les = LatticeBoltzmann3D::new(16, 16, 4, 0.02)
            .with_turbulence(Turbulence::Smagorinsky { cs: 0.16 });
        init(&mut dns);
        init(&mut les);
        dns.run(100);
        les.run(100);
        assert!(les.kinetic_energy() < dns.kinetic_energy());
    }
}
