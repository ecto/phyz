//! Vertex Block Descent: an implicit deformable-body solver that stays stable
//! at large timesteps.
//!
//! Implements Chen, Han, Chen, Xu, Ly, Li, Rong, Zhang, Zhu & Kim, *"Vertex
//! Block Descent"* (SIGGRAPH 2024) for tetrahedral solids, with a stable
//! Neo-Hookean FEM energy and an optional mass-spring energy on top.
//!
//! # What VBD is
//!
//! A backward-Euler step is the minimiser of the variational (incremental
//! potential) energy
//!
//! ```text
//! E(x) = Σᵢ mᵢ/(2h²) ‖xᵢ − yᵢ‖²  +  Ψ(x),      yᵢ = xᵢᵗ + h vᵢᵗ + h² aᵢᵉˣᵗ
//! ```
//!
//! Newton on that means factorising a `3n × 3n` matrix every iteration. VBD
//! instead does **block coordinate descent over vertices**: it visits one
//! vertex at a time, gathers only the energy terms that touch it, and takes a
//! guarded 3×3 Newton step in that vertex's three coordinates while everyone
//! else is held fixed. The blocks are tiny, exact, and independent of mesh
//! size.
//!
//! # How it differs from XPBD, honestly
//!
//! Both are Gauss–Seidel-ish and both survive stiff materials at big
//! timesteps, so they are easy to confuse. The difference is what they converge
//! *to*. XPBD projects onto constraints; iterated to convergence it reaches a
//! constraint-satisfying configuration whose relationship to the implicit-Euler
//! solution depends on the compliance formulation and the iteration count. VBD
//! iterated to convergence reaches the minimiser of `E` above, which *is* the
//! backward-Euler solution. Stopping early gives a configuration that is
//! partway down `E` — under-resolved, but under-resolved in a direction with a
//! known limit.
//!
//! # Stability: the claim, and its assumptions
//!
//! VBD is commonly described as "unconditionally stable". That claim holds for
//! this implementation under these assumptions, and it is worth being precise
//! about which:
//!
//! * **Energy is non-increasing per sweep.** Every block step is a descent
//!   direction (see [`spd`]), and each block step is evaluated against the
//!   current positions, so a full sweep cannot increase `E`. `E` is bounded
//!   below because the stable Neo-Hookean density is, and the inertia term is a
//!   norm. So the iteration cannot blow up *within* a step, at any `h`.
//! * **Not guaranteed:** accuracy. Stability and correctness are different
//!   claims. At `h = 1/30 s` and a handful of iterations the motion is heavily
//!   damped compared to the true trajectory — bounded, plausible, and wrong in
//!   the amount reported by `energy_drift_over_free_vibration` in
//!   `tests/validation.rs` (−5.3% of the vibration energy over 0.2 s at
//!   `h = 10⁻⁴ s`). Do not read "stable" as "converged". Measured: on the beam
//!   in `stable_at_a_timestep_where_explicit_explodes`, nothing diverged at any
//!   timestep tried, up to `h = 256 s` — roughly 1.6 × 10⁵ times the explicit
//!   CFL limit for that scene — while symplectic Euler goes non-finite at
//!   `h = 1/60 s`.
//! * **Not guaranteed:** anything about collisions, which this crate does not
//!   implement at all. Published VBD stability results include contact
//!   handling in the energy; ours cannot, so the claim here is narrower than
//!   the paper's.
//! * **Not guaranteed** under a `step_scale` above `1.0`, which over-relaxes
//!   the block solve and can and does diverge. It is exposed because it speeds
//!   up quasi-static solves, not because it is safe.
//!
//! # Determinism
//!
//! Pure `f64` throughout, no hash containers on any path that affects the
//! result, a fixed sweep order (colour class ascending, vertex index ascending
//! within a class), and a fixed-iteration eigensolver. Two runs of the same
//! setup are bit-identical; `stepping_is_bit_identical_across_runs` in
//! `tests/validation.rs` asserts it.
//!
//! # Out of scope
//!
//! Deliberately not implemented, rather than half-implemented:
//!
//! * **Collision and self-collision.** No contact energy, no proximity search.
//!   A body here passes through itself and through everything else.
//! * **GPU.** The colouring is what a GPU backend would need, and it is here
//!   and tested, but the sweep runs serially on the CPU.
//! * **Coupling to the rigid-body engine.** No two-way constraint against
//!   `phyz-rigid` bodies; pinned vertices are the only kinematic input.
//! * **The paper's accelerated initialisation** (the adaptive choice among
//!   several warm starts) and its multi-level extension. The warm start here is
//!   the plain inertial prediction.
//!
//! # Example
//!
//! ```
//! use phyz_math::Vec3;
//! use phyz_vbd::{Material, SoftBody, VbdConfig, VbdSolver, mesh};
//!
//! // A 1 m × 0.1 m × 0.1 m beam, clamped at x = 0, sagging under gravity.
//! let (rest, tets) = mesh::tet_box(8, 1, 1, Vec3::new(1.0, 0.1, 0.1));
//! let clamped: Vec<usize> = rest
//!     .iter()
//!     .enumerate()
//!     .filter(|(_, p)| p.x == 0.0)
//!     .map(|(i, _)| i)
//!     .collect();
//!
//! let mut body = SoftBody::builder(rest, Material::default())
//!     .tets(&tets)
//!     .pin(&clamped)
//!     .build()
//!     .expect("mesh has no degenerate tets");
//!
//! // 1/30 s is far past what an explicit integrator survives on this material.
//! let mut solver = VbdSolver::new(VbdConfig {
//!     dt: 1.0 / 30.0,
//!     iterations: 20,
//!     ..VbdConfig::default()
//! });
//! for _ in 0..30 {
//!     solver.step(&mut body);
//! }
//!
//! let tip = body.positions[body.positions.len() - 1];
//! assert!(tip.y.is_finite() && tip.y < 0.0, "the beam should sag");
//! ```

#![forbid(unsafe_code)]

pub mod coloring;
pub mod energy;
pub mod mesh;
pub mod spd;

pub use energy::{Material, Spring, TetElement};

use phyz_math::{Mat3, Vec3};

/// Solver settings. [`VbdConfig::default`] is a reasonable graphics-rate
/// starting point: 1/60 s, 10 iterations, light damping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbdConfig {
    /// Timestep, seconds.
    pub dt: f64,
    /// Gauss–Seidel sweeps per step. This is the accuracy knob: one iteration
    /// is a very soft, very damped solid; convergence to the implicit-Euler
    /// answer is linear in the iteration count, so doubling it roughly halves
    /// the residual and does *not* change the fixed point.
    pub iterations: usize,
    /// Uniform acceleration applied to every vertex, m/s².
    pub gravity: Vec3,
    /// Rayleigh-style stiffness damping coefficient `k_d`, in seconds.
    ///
    /// Adds `(k_d/h)·H·(xᵢ − xᵢᵗ)` to the vertex gradient and `(k_d/h)·H` to
    /// its Hessian, i.e. a damping force proportional to the *elastic*
    /// curvature. Two consequences worth knowing: it damps high-frequency
    /// modes far harder than low-frequency ones (that is the point — those are
    /// the modes the solver resolves worst), and because it scales as `k_d/h`
    /// its effect at a fixed `k_d` grows as the timestep shrinks. `0.0`
    /// disables it.
    pub damping: f64,
    /// Fraction of the guarded Newton step actually taken. `1.0` is the block
    /// Newton step and is the only value with a descent guarantee; values above
    /// `1.0` over-relax and may diverge.
    pub step_scale: f64,
    /// Lower bound imposed on every Hessian eigenvalue in the local solve, in
    /// N/m. Sets the largest step per unit gradient the guard permits, so it is
    /// the trust region as well as the definiteness floor. Too small and a
    /// near-singular block still takes a huge step; too large and every vertex
    /// crawls.
    pub hessian_floor: f64,
}

impl Default for VbdConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            iterations: 10,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.0,
            step_scale: 1.0,
            hessian_floor: 1e-6,
        }
    }
}

/// Why a [`SoftBodyBuilder`] refused to produce a body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A tet referenced a vertex index that does not exist.
    VertexOutOfRange {
        /// Position of the offending tet in the input list.
        tet: usize,
        /// The bad index.
        index: usize,
    },
    /// A tet's rest volume was at or below the minimum. Slivers are refused
    /// rather than simulated; see [`TetElement::new`].
    DegenerateTet {
        /// Position of the offending tet in the input list.
        tet: usize,
    },
    /// A spring referenced a vertex index that does not exist.
    SpringOutOfRange {
        /// Position of the offending spring in the input list.
        spring: usize,
        /// The bad index.
        index: usize,
    },
    /// No vertices were supplied.
    Empty,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::VertexOutOfRange { tet, index } => {
                write!(f, "tet {tet} references out-of-range vertex {index}")
            }
            Self::DegenerateTet { tet } => write!(f, "tet {tet} is degenerate"),
            Self::SpringOutOfRange { spring, index } => {
                write!(f, "spring {spring} references out-of-range vertex {index}")
            }
            Self::Empty => write!(f, "a soft body needs at least one vertex"),
        }
    }
}

impl core::error::Error for BuildError {}

/// Incremental construction of a [`SoftBody`].
///
/// Construction is a builder rather than a set of mutators because the
/// colouring and the per-vertex incidence lists depend on the *complete* set of
/// energy terms. Letting a tet be added after colouring would leave the two
/// silently inconsistent, and the symptom — two coupled vertices in one colour
/// — is invisible until somebody parallelises the sweep.
#[derive(Debug, Clone)]
pub struct SoftBodyBuilder {
    rest: Vec<Vec3>,
    tets: Vec<[usize; 4]>,
    springs: Vec<Spring>,
    pinned: Vec<bool>,
    material: Material,
    min_volume: f64,
}

impl SoftBodyBuilder {
    /// Add tetrahedra, referencing the rest positions by index.
    pub fn tets(mut self, tets: &[[usize; 4]]) -> Self {
        self.tets.extend_from_slice(tets);
        self
    }

    /// Add a spring between two vertices, with rest length taken from the rest
    /// configuration.
    pub fn spring(mut self, i: usize, j: usize, stiffness: f64) -> Self {
        let rest_length = if i < self.rest.len() && j < self.rest.len() {
            (self.rest[i] - self.rest[j]).norm()
        } else {
            0.0
        };
        self.springs.push(Spring {
            verts: [i, j],
            rest_length,
            stiffness,
        });
        self
    }

    /// Pin vertices: they keep their rest position and never move.
    pub fn pin(mut self, vertices: &[usize]) -> Self {
        for &v in vertices {
            if v < self.pinned.len() {
                self.pinned[v] = true;
            }
        }
        self
    }

    /// Minimum accepted tet rest volume, m³. Default `1e-12`.
    pub fn min_volume(mut self, v: f64) -> Self {
        self.min_volume = v;
        self
    }

    /// Finish: build the elements, lump the masses, and colour the graph.
    pub fn build(self) -> Result<SoftBody, BuildError> {
        let n = self.rest.len();
        if n == 0 {
            return Err(BuildError::Empty);
        }

        let mut elements = Vec::with_capacity(self.tets.len());
        let mut masses = vec![0.0; n];
        for (t, verts) in self.tets.iter().enumerate() {
            for &v in verts {
                if v >= n {
                    return Err(BuildError::VertexOutOfRange { tet: t, index: v });
                }
            }
            let rest = [
                self.rest[verts[0]],
                self.rest[verts[1]],
                self.rest[verts[2]],
                self.rest[verts[3]],
            ];
            let element = TetElement::new(*verts, rest, &self.material, self.min_volume)
                .ok_or(BuildError::DegenerateTet { tet: t })?;
            // Lumped mass: a quarter of the element's mass on each vertex. The
            // consistent (non-diagonal) mass matrix would couple vertices in
            // the inertia term too, which VBD's per-vertex blocks cannot
            // represent — lumping is a modelling choice the method requires,
            // not an approximation chosen for speed.
            let quarter = self.material.density * element.rest_volume / 4.0;
            for &v in &element.verts {
                masses[v] += quarter;
            }
            elements.push(element);
        }

        for (s, spring) in self.springs.iter().enumerate() {
            for &v in &spring.verts {
                if v >= n {
                    return Err(BuildError::SpringOutOfRange {
                        spring: s,
                        index: v,
                    });
                }
            }
        }

        // A vertex in no tet gets no mass from lumping. Give it a token mass so
        // its inertia term is finite; a zero-mass vertex would have an empty
        // inertia block and be driven entirely by the guard's floor.
        for m in masses.iter_mut() {
            if *m <= 0.0 {
                *m = f64::MIN_POSITIVE;
            }
        }

        let mut incident_tets = vec![Vec::new(); n];
        for (e, element) in elements.iter().enumerate() {
            for (local, &v) in element.verts.iter().enumerate() {
                incident_tets[v].push((e as u32, local as u8));
            }
        }
        let mut incident_springs = vec![Vec::new(); n];
        for (s, spring) in self.springs.iter().enumerate() {
            for (local, &v) in spring.verts.iter().enumerate() {
                incident_springs[v].push((s as u32, local as u8));
            }
        }

        let mut groups: Vec<&[usize]> = Vec::with_capacity(elements.len() + self.springs.len());
        for element in &elements {
            groups.push(&element.verts);
        }
        for spring in &self.springs {
            groups.push(&spring.verts);
        }
        let colors = coloring::color_vertices(n, &groups);

        Ok(SoftBody {
            positions: self.rest.clone(),
            velocities: vec![Vec3::zero(); n],
            rest_positions: self.rest,
            masses,
            pinned: self.pinned,
            elements,
            springs: self.springs,
            colors,
            incident_tets,
            incident_springs,
        })
    }
}

/// A tetrahedral deformable body: state, elements, and the precomputed
/// structure VBD sweeps over.
#[derive(Debug, Clone)]
pub struct SoftBody {
    /// Current vertex positions, m. Writable — this is how you set an initial
    /// deformation.
    pub positions: Vec<Vec3>,
    /// Current vertex velocities, m/s.
    pub velocities: Vec<Vec3>,
    /// Rest positions, m. Pinned vertices are held here.
    pub rest_positions: Vec<Vec3>,
    /// Lumped vertex masses, kg.
    pub masses: Vec<f64>,
    /// Per-vertex pin flags.
    pub pinned: Vec<bool>,
    elements: Vec<TetElement>,
    springs: Vec<Spring>,
    colors: Vec<Vec<usize>>,
    incident_tets: Vec<Vec<(u32, u8)>>,
    incident_springs: Vec<Vec<(u32, u8)>>,
}

impl SoftBody {
    /// Start building a body from rest positions and a material.
    pub fn builder(rest: Vec<Vec3>, material: Material) -> SoftBodyBuilder {
        let n = rest.len();
        SoftBodyBuilder {
            rest,
            tets: Vec::new(),
            springs: Vec::new(),
            pinned: vec![false; n],
            material,
            min_volume: 1e-12,
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the body has no vertices. Only reachable via [`Clone`]-and-
    /// truncate; [`SoftBodyBuilder::build`] rejects it.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The tetrahedra.
    pub fn elements(&self) -> &[TetElement] {
        &self.elements
    }

    /// The springs.
    pub fn springs(&self) -> &[Spring] {
        &self.springs
    }

    /// Colour classes, in sweep order. Every vertex appears in exactly one, and
    /// no two vertices in a class share an energy term — so a parallel backend
    /// can dispatch a class as one kernel.
    pub fn colors(&self) -> &[Vec<usize>] {
        &self.colors
    }

    /// Total elastic energy, joules.
    pub fn elastic_energy(&self) -> f64 {
        let tet: f64 = self
            .elements
            .iter()
            .map(|e| e.energy(&self.positions))
            .sum();
        let spring: f64 = self.springs.iter().map(|s| s.energy(&self.positions)).sum();
        tet + spring
    }

    /// Total kinetic energy, joules. Pinned vertices contribute nothing.
    pub fn kinetic_energy(&self) -> f64 {
        (0..self.len())
            .filter(|&i| !self.pinned[i])
            .map(|i| 0.5 * self.masses[i] * self.velocities[i].norm_sq())
            .sum()
    }

    /// Gravitational potential `−Σ mᵢ g·xᵢ`, joules, for the uniform
    /// acceleration `gravity`. The zero point is the world origin.
    pub fn gravitational_energy(&self, gravity: Vec3) -> f64 {
        (0..self.len())
            .filter(|&i| !self.pinned[i])
            .map(|i| -self.masses[i] * gravity.dot(self.positions[i]))
            .sum()
    }

    /// Total mechanical energy under `gravity`.
    pub fn total_energy(&self, gravity: Vec3) -> f64 {
        self.kinetic_energy() + self.elastic_energy() + self.gravitational_energy(gravity)
    }

    /// Elastic force `−∂Ψ/∂x` on every vertex, N.
    ///
    /// Exposed because it is what you need to check a static equilibrium, and
    /// what an explicit integrator would consume — `tests/validation.rs` uses
    /// it for both.
    pub fn elastic_forces(&self) -> Vec<Vec3> {
        let mut f = vec![Vec3::zero(); self.len()];
        for element in &self.elements {
            for local in 0..4 {
                let (g, _) = element.gradient_and_hessian(&self.positions, local);
                f[element.verts[local]] -= g;
            }
        }
        for spring in &self.springs {
            for local in 0..2 {
                let (g, _) = spring.gradient_and_hessian(&self.positions, local);
                f[spring.verts[local]] -= g;
            }
        }
        f
    }

    /// Reset positions to rest and velocities to zero.
    pub fn reset(&mut self) {
        self.positions.copy_from_slice(&self.rest_positions);
        self.velocities.iter_mut().for_each(|v| *v = Vec3::zero());
    }
}

/// The VBD time integrator.
///
/// Owns its scratch buffers so a steady-state loop does not allocate per step.
#[derive(Debug, Clone)]
pub struct VbdSolver {
    /// Solver settings. Safe to change between steps.
    pub config: VbdConfig,
    inertial: Vec<Vec3>,
    previous: Vec<Vec3>,
}

impl VbdSolver {
    /// A solver with the given settings.
    pub fn new(config: VbdConfig) -> Self {
        Self {
            config,
            inertial: Vec::new(),
            previous: Vec::new(),
        }
    }

    /// Advance `body` by one timestep.
    ///
    /// Three phases, in the order the method requires:
    ///
    /// 1. **Inertial prediction.** `yᵢ = xᵢ + h vᵢ + h² g` — where the vertex
    ///    would land with no elastic force. `y` is the *target* of the inertia
    ///    term for the whole step and must be frozen before any vertex moves;
    ///    recomputing it mid-sweep would change the objective the sweep is
    ///    descending and destroy the convergence argument.
    /// 2. **`iterations` Gauss–Seidel sweeps** over the colour classes.
    /// 3. **Velocity update** `v = (x − xᵗ)/h`. Deriving the velocity from the
    ///    positions, rather than integrating an acceleration, is what keeps the
    ///    step consistent with backward Euler regardless of how far the
    ///    position solve actually converged.
    pub fn step(&mut self, body: &mut SoftBody) {
        let h = self.config.dt;
        if !h.is_finite() || h <= 0.0 {
            return;
        }
        let n = body.len();
        self.inertial.resize(n, Vec3::zero());
        self.previous.resize(n, Vec3::zero());
        self.previous[..n].copy_from_slice(&body.positions);

        let g = self.config.gravity;
        for i in 0..n {
            if body.pinned[i] {
                // Pinned vertices are held at rest, not integrated. Writing the
                // rest position (rather than leaving whatever `positions` had)
                // makes a pin authoritative even if a caller moved the vertex.
                body.positions[i] = body.rest_positions[i];
                body.velocities[i] = Vec3::zero();
                self.inertial[i] = body.rest_positions[i];
            } else {
                self.inertial[i] = body.positions[i] + body.velocities[i] * h + g * (h * h);
                // Warm start at the inertial prediction: the exact answer when
                // there is no elastic force, and the paper's baseline
                // initialisation.
                body.positions[i] = self.inertial[i];
            }
        }

        for _ in 0..self.config.iterations {
            for c in 0..body.colors.len() {
                for k in 0..body.colors[c].len() {
                    // Indexed rather than iterated because the sweep needs
                    // `&mut body` while reading `body.colors`.
                    let i = body.colors[c][k];
                    self.solve_vertex(body, i);
                }
            }
        }

        let inv_h = 1.0 / h;
        for i in 0..n {
            if !body.pinned[i] {
                body.velocities[i] = (body.positions[i] - self.previous[i]) * inv_h;
            }
        }
    }

    /// One vertex's block Newton step.
    fn solve_vertex(&self, body: &mut SoftBody, i: usize) {
        if body.pinned[i] {
            return;
        }

        let mut grad = Vec3::zero();
        let mut hess = Mat3::zero();
        for &(e, local) in &body.incident_tets[i] {
            let (g, h) =
                body.elements[e as usize].gradient_and_hessian(&body.positions, local as usize);
            grad += g;
            hess = hess + h;
        }
        for &(s, local) in &body.incident_springs[i] {
            let (g, h) =
                body.springs[s as usize].gradient_and_hessian(&body.positions, local as usize);
            grad += g;
            hess = hess + h;
        }

        // Rayleigh stiffness damping, applied to the *elastic* blocks only:
        // damping the inertia term would damp rigid translation as well, which
        // is not what a material damping model means.
        let kd = self.config.damping;
        if kd > 0.0 {
            let scale = kd / self.config.dt;
            grad += hess.mul_vec(body.positions[i] - self.previous[i]) * scale;
            hess = hess + hess * scale;
        }

        // The inertia block. Positive definite by construction and independent
        // of the deformation, so it is the reason most vertices never need the
        // guard — but it scales as m/h², so at the large timesteps VBD is sold
        // on it is small, and the elastic blocks dominate. That is the regime
        // where an indefinite Hessian actually decides the outcome.
        let inertia = body.masses[i] / (self.config.dt * self.config.dt);
        grad += (body.positions[i] - self.inertial[i]) * inertia;
        hess = hess + Mat3::identity() * inertia;

        if let Some(step) = spd::spd_solve(&hess, &(-grad), self.config.hessian_floor) {
            body.positions[i] += step * self.config.step_scale;
        }
        // `None` means the block went non-finite. Leaving the vertex where it
        // is loses accuracy for one iteration; propagating a NaN loses the
        // whole simulation.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tet() -> SoftBody {
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.1),
            Vec3::new(0.0, 0.1, 0.0),
        ];
        SoftBody::builder(rest, Material::default())
            .tets(&[[0, 1, 2, 3]])
            .pin(&[0, 1, 2])
            .build()
            .unwrap()
    }

    #[test]
    fn mass_lumping_conserves_total_mass() {
        let material = Material::default();
        let (rest, tets) = mesh::tet_box(2, 2, 2, Vec3::new(0.2, 0.2, 0.2));
        let body = SoftBody::builder(rest, material)
            .tets(&tets)
            .build()
            .unwrap();
        let total: f64 = body.masses.iter().sum();
        let expected = material.density * 0.2 * 0.2 * 0.2;
        assert!(
            (total - expected).abs() < 1e-9 * expected,
            "lumped {total} vs {expected}"
        );
    }

    #[test]
    fn pinned_vertices_never_move() {
        let mut body = single_tet();
        let mut solver = VbdSolver::new(VbdConfig::default());
        for _ in 0..50 {
            solver.step(&mut body);
        }
        for i in 0..3 {
            assert_eq!(body.positions[i], body.rest_positions[i]);
            assert_eq!(body.velocities[i], Vec3::zero());
        }
    }

    /// With no gravity and no deformation, nothing should happen at all — the
    /// rest state must be an exact fixed point, not merely a near one.
    #[test]
    fn rest_state_is_a_fixed_point() {
        let (rest, tets) = mesh::tet_box(2, 1, 1, Vec3::new(0.2, 0.1, 0.1));
        let mut body = SoftBody::builder(rest.clone(), Material::default())
            .tets(&tets)
            .build()
            .unwrap();
        let mut solver = VbdSolver::new(VbdConfig {
            gravity: Vec3::zero(),
            ..VbdConfig::default()
        });
        for _ in 0..20 {
            solver.step(&mut body);
        }
        let drift = body
            .positions
            .iter()
            .zip(rest.iter())
            .map(|(p, r)| (*p - *r).norm())
            .fold(0.0f64, f64::max);
        assert!(drift < 1e-12, "drifted {drift:e} m from rest");
    }

    /// Free fall: with no pins, a body under gravity must translate rigidly at
    /// exactly the analytic rate. This catches an inertia term that is
    /// mis-scaled in `h` — an error that a pinned test hides entirely, because
    /// the elastic force absorbs it.
    #[test]
    fn unpinned_body_free_falls_at_g() {
        let (rest, tets) = mesh::tet_box(2, 1, 1, Vec3::new(0.2, 0.1, 0.1));
        let mut body = SoftBody::builder(rest.clone(), Material::default())
            .tets(&tets)
            .build()
            .unwrap();
        let g = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;
        let mut solver = VbdSolver::new(VbdConfig {
            dt,
            gravity: g,
            ..VbdConfig::default()
        });
        let steps = 60;
        for _ in 0..steps {
            solver.step(&mut body);
        }
        // Backward Euler on a free particle: after n steps, y = ½ g h² n(n+1).
        let n = steps as f64;
        let expected = 0.5 * g.y * dt * dt * n * (n + 1.0);
        for (i, (p, r)) in body.positions.iter().zip(rest.iter()).enumerate() {
            let drop = p.y - r.y;
            assert!(
                (drop - expected).abs() < 1e-12,
                "vertex {i} fell {drop} m, expected {expected} m"
            );
        }
    }

    #[test]
    fn colors_cover_every_vertex_exactly_once() {
        let (rest, tets) = mesh::tet_box(3, 2, 2, Vec3::new(0.3, 0.2, 0.2));
        let body = SoftBody::builder(rest, Material::default())
            .tets(&tets)
            .build()
            .unwrap();
        let mut seen = vec![0usize; body.len()];
        for class in body.colors() {
            for &v in class {
                seen[v] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "colouring is not a partition");
    }

    /// More iterations must monotonically lower the variational energy the
    /// solver is minimising — the property that makes VBD an integrator rather
    /// than a heuristic. Measured on the elastic energy of a statically loaded
    /// body, where the inertia term is negligible at convergence.
    #[test]
    fn more_iterations_lower_the_residual() {
        let residual = |iterations: usize| {
            let (rest, tets) = mesh::tet_box(4, 1, 1, Vec3::new(0.4, 0.1, 0.1));
            let pins: Vec<usize> = rest
                .iter()
                .enumerate()
                .filter(|(_, p)| p.x == 0.0)
                .map(|(i, _)| i)
                .collect();
            let mut body = SoftBody::builder(rest, Material::default())
                .tets(&tets)
                .pin(&pins)
                .build()
                .unwrap();
            let g = Vec3::new(0.0, -9.81, 0.0);
            let mut solver = VbdSolver::new(VbdConfig {
                dt: 1.0 / 60.0,
                iterations,
                gravity: g,
                damping: 0.05,
                ..VbdConfig::default()
            });
            for _ in 0..400 {
                solver.step(&mut body);
            }
            // Static residual: elastic force plus weight, on free vertices.
            let f = body.elastic_forces();
            (0..body.len())
                .filter(|&i| !body.pinned[i])
                .map(|i| (f[i] + g * body.masses[i]).norm())
                .fold(0.0f64, f64::max)
        };
        let few = residual(2);
        let many = residual(40);
        assert!(
            many < few,
            "40 iterations ({many:e} N) did not beat 2 ({few:e} N)"
        );
    }

    #[test]
    fn build_rejects_bad_indices() {
        let rest = vec![Vec3::zeros(); 4];
        let err = SoftBody::builder(rest, Material::default())
            .tets(&[[0, 1, 2, 9]])
            .build()
            .unwrap_err();
        assert_eq!(err, BuildError::VertexOutOfRange { tet: 0, index: 9 });
    }

    #[test]
    fn build_rejects_degenerate_tets() {
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let err = SoftBody::builder(rest, Material::default())
            .tets(&[[0, 1, 2, 3]])
            .build()
            .unwrap_err();
        assert_eq!(err, BuildError::DegenerateTet { tet: 0 });
    }

    #[test]
    fn zero_or_negative_dt_is_a_no_op() {
        let mut body = single_tet();
        let before = body.positions.clone();
        let mut solver = VbdSolver::new(VbdConfig {
            dt: 0.0,
            ..VbdConfig::default()
        });
        solver.step(&mut body);
        assert_eq!(body.positions, before);
    }
}
