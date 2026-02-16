# tau — Multi-Physics Differentiable Simulation Engine

## Project Overview

tau is a pure-Rust, GPU-acceleratable physics simulation engine designed for differentiable multi-physics simulation at scale. It extends rigid body dynamics with particles, electromagnetism, quantum mechanics, gravity, and emergent physics—all differentiable for machine learning and inverse problems.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ tau / High-level API + World management                     │
├─────────────────────────────────────────────────────────────┤
│ Solvers:                                                    │
│  • tau-rigid (ABA/CRBA/RNEA)  → forward/inverse dynamics   │
│  • tau-collision (GJK/EPA)    → contact detection           │
│  • tau-contact (soft contacts) → contact solving             │
│  • tau-gpu (wgpu compute)     → parallel batching           │
│  • tau-particle (MPM)         → material point method       │
│  • tau-em (Yee FDTD)          → electromagnetism           │
│  • tau-md (Velocity Verlet)   → molecular dynamics         │
│  • tau-qft (HMC/SU(3))        → lattice gauge theory       │
│  • tau-gravity (N-body+GR)    → gravitational fields       │
│  • tau-lbm (D3Q19)            → lattice Boltzmann         │
│  • tau-compile (Naga IR)      → JIT physics kernels       │
│  • tau-prob (ensembles)       → uncertainty propagation    │
├─────────────────────────────────────────────────────────────┤
│ Cross-cutting:                                              │
│  • tau-diff (automatic differentiation via finite-diff +   │
│             analytical chain-rule through solvers)         │
│  • tau-guardian (conservation + adaptive time-stepping)    │
│  • tau-emerge (equation-free + Lattice Boltzmann)         │
│  • tau-real2sim (inverse problems via differentiable opt)  │
│  • tau-format (universal .tau JSON multi-domain spec)      │
├─────────────────────────────────────────────────────────────┤
│ Math Foundation:                                            │
│  • tau-math (spatial algebra, 6D vectors, Plücker txforms)│
│  • tau-model (Model, State, Body, Joint abstractions)      │
└─────────────────────────────────────────────────────────────┘
```

### Core Types

**Phase 1 built:**
- `Model`: static kinematic tree, masses, joint types, DOF offsets
- `State`: mutable (q, v, ctrl, time, cached body transforms)
- `Body`: mass, inertia, parent, joint index
- `Joint`: single revolute type (angle about Z), parent_to_joint transform, axis, damping, limits
- `SpatialVec`: [angular; linear] 6D vectors following Featherstone
- `SpatialTransform`: Plücker transform (R, r) for coordinate changes
- `SpatialInertia`: mass, center-of-mass, inertia matrix
- `Solver` trait: `step(&Model, &mut State)` + `step_with_jacobians()`
- `Simulator`: pluggable solver with semi-implicit Euler and RK4

**Dynamics algorithms (Phase 1):**
- Forward kinematics: propagate `body_xform` from root to leaves
- ABA (Articulated Body Algorithm): O(n) forward dynamics
- RNEA (Recursive Newton-Euler): O(n) inverse dynamics
- CRBA (Composite Rigid Body): O(n²) mass matrix computation

**Differentiation (Phase 1):**
- Finite-difference Jacobians: ∂(q', v')/∂(q, v, ctrl) via 2·(nq+nv+nv) perturbations
- Analytical Jacobians for semi-implicit Euler via chain rule through ABA

### Key Design Decisions

| Decision | Rationale | Examples |
|----------|-----------|----------|
| **Featherstone spatial algebra** | O(n) dynamics, natural for articulated bodies, generalizes to constraints | ABA, RNEA, CRBA all O(n) |
| **Plücker coordinates** | Compact cross-product rules for velocity/force propagation | `SpatialVec::cross_motion/cross_force` |
| **Semi-implicit Euler + RK4** | Energy-conserving, stable for stiff joints; RK4 for high-order accuracy | pendulum example: <1% drift over 20s |
| **Finite-diff + analytical gradients** | Finite-diff accurate, analytical fast; mix as needed | `finite_diff_jacobians` for validation, analytical for optimization loops |
| **Joint type enum** | Extensible to prismatic, spherical, free, fixed | Phase 2: expand JointType, joint_transform(), motion_subspace() |
| **GPU-first batching** | Parallel environments in nworld dimension; matrix kernels over loops | Phase 4: wgpu compute shaders, batch ABA |
| **Modular solver trait** | Plug in different integrators, custom solvers | Phase 5: MPM solver; Phase 8: FDTD solver |
| **Differentiable by default** | All algorithms track gradients; optimization-friendly | Jacobian computation in Phase 1; inverse problems in Phase 17 |

### Existing Examples & Tests

- `examples/pendulum.rs`: validates period, demonstrates energy conservation and gradients
- `examples/double_pendulum.rs`: two-link chain, tests CRBA + RK4 energy conservation
- `tau-diff` tests: Jacobian finite-diff vs. analytical agreement
- `tau-rigid` tests: forward kinematics, ABA correctness, energy computation

### Papers & References (All Phases)

- **Featherstone (2008):** "Rigid Body Dynamics Algorithms" — foundational ABA/CRBA/RNEA
- **Muller & Gross (2004):** "Interaction of Deformable Objects" — point-in-tetrahedron for MPM P2G/G2P
- **Gast et al. (2015):** "A Material Point Method for Snow Simulation" — material models
- **Yee (1966):** "Numerical Solution of Initial Boundary Value Problems" — FDTD Yee grid
- **Boyd et al. (2008):** "Boyd et al. on ADMM" — contact solving with splittng methods
- **Monaghan (1992):** "Smoothed particle hydrodynamics" — SPH as alternative to MPM
- **Creutz & Mikromandraki (1989):** "Wilson Loops and the String Tension" — lattice gauge theory
- **Arndt et al. (2018):** "exaStencils: A Framework for Multigrid Codes" — stencil scheduling
- **Succi (2001):** "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond" — LBM theory
- **Thawani et al. (2022):** "Deep Learning for Computational Fluid Dynamics" — ML/surrogate integration

---

## Phase 2: Multi-joint + MJCF

### Objective
Extend the single-revolute joint system to support all standard joint types and load complex models from MuJoCo MJCF XML files. Implement quaternion integration and analytical derivatives for all joint types.

### Specification

#### Joint Type Expansion
Extend `JointType` enum in `tau-model/src/joint.rs`:
```rust
pub enum JointType {
    Revolute,      // 1 DOF, rotation about axis
    Prismatic,     // 1 DOF, translation along axis
    Spherical,     // 3 DOF, 3D rotation (use quaternions q=[w;x;y;z])
    Free,          // 6 DOF, 3D translation + 3D rotation
    Fixed,         // 0 DOF, rigid attachment
    Hinge,         // Alias for revolute (MuJoCo compat)
    Slide,         // Alias for prismatic (MuJoCo compat)
    Ball,          // Alias for spherical (MuJoCo compat)
}
```

Implement `joint_transform(q: &[f64]) -> SpatialTransform` for each type:
- **Revolute:** Rodrigues' formula, angle about joint axis
- **Prismatic:** translate along joint axis by distance q
- **Spherical:** quaternion q=[w;x;y;z] exponential map; Q = exp(ω dt) where ω is angular velocity
- **Free:** 3-element position q[0:3], 4-element quaternion q[3:7]
- **Fixed:** identity (no DOF, returns static transform)

Update `Joint` to track DOF count per type:
```rust
impl Joint {
    pub fn ndof(&self) -> usize {
        match self.joint_type {
            JointType::Revolute | JointType::Prismatic => 1,
            JointType::Spherical => 3,
            JointType::Free => 6,
            JointType::Fixed => 0,
            // ... aliases
        }
    }
}
```

#### Motion Subspace (Analytical Jacobians)
Each joint type has a motion subspace matrix S (6×ndof) such that V_successor = V_parent + S · v_joint.

Implement `motion_subspace(&self) -> DMat` for each joint:
- **Revolute/Prismatic:** single column [ω; v] or [0; v]
- **Spherical:** 3-column matrix for angular velocity in local frame (will be rotated by R(q))
- **Free:** 6×6 identity-like structure
- **Fixed:** 6×0 (zero matrix)

Key insight: spherical joint motion subspace must rotate with the joint's current orientation. Update `aba()` in `tau-rigid/src/aba.rs` to handle this dynamically.

#### Quaternion Integration for Spherical/Free Joints
Semi-implicit Euler for quaternions is tricky (must stay normalized). Implement quaternion exponential map:

```rust
// tau-math quaternion utilities
pub fn quat_exp(w: &Vec3) -> Quat {
    // w is angular velocity, returns q(dt) = exp(w dt / 2)
    let theta = w.norm();
    if theta < 1e-10 {
        Quat { w: 1.0, v: w * 0.5 }
    } else {
        let half_theta = theta * 0.5;
        Quat {
            w: half_theta.cos(),
            v: w * (half_theta.sin() / theta),
        }
    }
}

pub fn quat_mul(p: &Quat, q: &Quat) -> Quat { /* ... */ }

pub fn quat_to_matrix(q: &Quat) -> Mat3 { /* ... */ }
pub fn matrix_to_quat(R: &Mat3) -> Quat { /* ... */ }
```

Integrate spherical joint:
```
dq/dt = 0.5 * q * [0; ω]  (quaternion kinematics)
In semi-implicit Euler: q_{n+1} = q_n * exp(ω_n dt / 2)
```

#### tau-mjcf: MJCF XML Parser
Create new crate `crates/tau-mjcf/` with MJCF parser. Support:

**XML parsing:**
- Dependency: `quick_xml` crate for XML parsing
- Parse `<mujoco>`, `<worldbody>`, `<body>`, `<joint>`, `<inertial>`, `<geom>` elements
- Support class definitions and template inheritance (MuJoCo defaults system)

**Model building:**
```rust
pub struct MjcfLoader {
    defaults: HashMap<String, ElementDefaults>,
    // ...
}

impl MjcfLoader {
    pub fn from_file(path: &str) -> Result<MjcfLoader, Error> { /* ... */ }
    pub fn build_model(&self) -> Model { /* ... */ }
}
```

**Features to implement:**
- Parse body hierarchy, joint definitions, inertial properties
- Apply MuJoCo defaults (e.g., `<default class="...">` overrides)
- Convert `<inertial>` mass/inertia to `SpatialInertia`
- Handle joint types: hinge (revolute), slide (prismatic), ball (spherical), free
- Parse body poses (pos, quat) and joint frames
- Gravity and timestep configuration
- Geometry collision detection setup (tag as collidable)

**Example: Load Ant from MJCF**
```rust
let loader = MjcfLoader::from_file("models/ant.xml")?;
let model = loader.build_model();
// Should have 8 legs × 2 joints + 1 root = 17 bodies, various joint types
```

#### Analytical Derivatives for New Joint Types
Extend `tau-diff/src/lib.rs` to compute analytical step Jacobians for spherical and free joints:

- **Revolute/Prismatic:** already implemented (scalar angle/distance)
- **Spherical:** quaternion perturbation requires Euler angle parameterization or tangent space perturbation
  - Use exponential map: perturb as q(ε) = exp(ε u_i) * q where u_i are basis vectors
  - Or use Euler angles as intermediate representation
- **Free:** combine position and quaternion derivatives

Fallback to finite differences if analytical becomes complex; Phase 2 goal is **correct** derivatives, not necessarily hand-coded for all types.

#### New Examples & Tests
- `examples/multi_joint.rs`: simple 2-link arm with revolute + prismatic joints
- `examples/ant.rs`: load MuJoCo Ant model (8 legs, 16 joints), run short simulation
- `tests/mjcf_parser.rs`: parse several MJCF files, verify body count, joint types
- Unit tests in `tau-mjcf`: XML parsing, defaults merging, inertia computation

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 3: Collision + Contacts

### Objective
Implement spatial collision detection (GJK/EPA, sweep-and-prune) and soft contact resolution. Support ground plane, ball drop, and cartpole examples.

### Specification

#### tau-collision: GJK/EPA Convex Collision
Create `crates/tau-collision/` for collision detection.

**Geometry types:**
```rust
pub enum Geometry {
    Sphere { radius: f64 },
    Capsule { radius: f64, length: f64 },
    Box { half_extents: Vec3 },
    Cylinder { radius: f64, height: f64 },
    Mesh { vertices: Vec<Vec3>, faces: Vec<[usize; 3]> }, // convex hull assumed
    Plane { normal: Vec3 }, // half-space
}

pub struct Collision {
    // Body index pairs involved
    pub body_i: usize,
    pub body_j: usize,
    // Contact info
    pub contact_point: Vec3,   // world frame
    pub contact_normal: Vec3,  // direction from i to j
    pub penetration_depth: f64,
}
```

**GJK (Gilbert-Johnson-Keerthi) algorithm:**
- Compute minimum distance between two convex shapes
- Output: closest point pair, separation distance, or (if negative) penetration
- Support function for each geometry type

**EPA (Expanding Polytope Algorithm):**
- Refine GJK's separating polytope to find contact normal and depth
- Used when shapes overlap (distance < 0)

**Broad-phase: Sweep-and-Prune**
- Maintain axis-aligned bounding boxes (AABBs) for each body
- Sort bodies along each axis; find overlapping pairs in O(n log n) time
- Cull pairs with no AABB overlap before running GJK/EPA

#### tau-contact: Soft Contact Solver
Create `crates/tau-contact/` for contact dynamics.

**Contact model:**
MuJoCo-style soft contacts using Lagrangian penalty method:
- Contact force: F = k · max(0, penetration_depth)^p · contact_normal
- Friction: Coulomb friction with viscous damping
- Bounce: coefficient of restitution e ∈ [0, 1]

Configurable parameters:
```rust
pub struct ContactMaterial {
    pub stiffness: f64,        // k (N/m)
    pub damping: f64,          // c (N·s/m)
    pub friction: f64,         // μ (dimensionless)
    pub bounce: f64,           // e (dimensionless)
    pub soft_cfm: f64,         // constraint force mixing
    pub soft_erp: f64,         // error reduction parameter
}
```

**Contact Jacobians:**
- For each contact, compute ∂F_contact/∂(q, v) via implicit function theorem
- Contact constraint: distance(body_i, body_j) ≥ 0
- Lagrange multiplier λ satisfies complementarity: λ ≥ 0, distance ≥ 0, λ·distance = 0

**Integration with ABA:**
- Modify ABA to include contact forces in external force vector
- Iterative contact resolution: solve ABA → find new contacts → update forces → repeat

#### Contacts from Distance
Implement "Contacts from Distance" (CFD) method:
- Query distance between all body pairs (broad-phase + GJK/EPA)
- For each contact with positive penetration, generate reaction force
- No separate constraint solver; forces act directly in ABA

```rust
pub fn find_contacts(model: &Model, state: &State) -> Vec<Collision> {
    let mut contacts = Vec::new();
    for (i, j) in broad_phase_pairs(&state) {
        if let Some(contact) = gjk_epa(&model.geoms[i], &model.geoms[j],
                                        &state.body_xform[i], &state.body_xform[j]) {
            if contact.penetration_depth > 0.0 {
                contacts.push(contact);
            }
        }
    }
    contacts
}

pub fn contact_forces(contacts: &[Collision], materials: &[ContactMaterial]) -> Vec<SpatialVec> {
    // Return contact wrenches (6D forces) for each body
}
```

#### Examples
- **Ball drop:** single sphere falls onto ground plane; measure bounce height vs. gravity time
- **Cartpole with collision:** cartpole on ground plane; pole collides with ground if it falls
- **Contact stacking:** stack 5 boxes on ground; stability test

#### Tests
- Unit tests: GJK distance, EPA penetration, broad-phase pairing
- Integration: forward simulation with contacts, energy conservation check
- Regression: ball bounce, box stack height comparison to reference

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 4: GPU Parallel Simulation

### Objective
Accelerate batch simulation on GPU using wgpu compute shaders. Target 1000+ parallel environments.

### Specification

#### tau-gpu: WGPU Compute Backend
Create `crates/tau-gpu/` for GPU-accelerated simulation.

**Architecture:**
- Double-buffered state: state(t), state(t+dt) on GPU memory
- Batch dimension: simulate nworld independent environments in parallel
- State layout: per-world state contiguous in memory (SoA/AoS trade-offs)

**Compute shader strategy:**
- Naga IR compilation: write kernels in WGSL, target SPIR-V/MSL/HLSL
- Kernel fusion: combine ABA forward-backward passes into single shader
- Matrix operations: use standard matrix kernel patterns (mm, transpose, axpy)

**Data layout:**
```rust
pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub q_buffer: wgpu::Buffer,      // (nworld, nq)
    pub v_buffer: wgpu::Buffer,      // (nworld, nv)
    pub qdd_buffer: wgpu::Buffer,    // (nworld, nv) [scratch]
    pub mass_matrix: wgpu::Buffer,   // (nworld, nv, nv)
}

pub struct GpuSimulator {
    pub device: wgpu::Device,
    pub nworld: usize,
    pub state: GpuState,
    pub aba_pipeline: wgpu::ComputePipeline,
    pub integrate_pipeline: wgpu::ComputePipeline,
}
```

#### Batched ABA Kernel
Implement ABA as compute shader:
1. **Forward pass (root → leaves):** compute velocities, bias forces in parallel for independent bodies
2. **Backward pass (leaves → root):** accumulate forces, compute accelerations
3. **Quaternion update:** for each free/spherical joint, integrate q with quaternion exponential

Each work group handles one body; nworld × nbodies total threads.

#### Quaternion Kernels
Matrix-based quaternion operations (WGSL):
```wgsl
// Quaternion multiplication (optimized as matrix × vector)
fn quat_mul(p: vec4f, q: vec4f) -> vec4f { /* ... */ }

// Exponential map: exp(ω dt/2) as normalized vector
fn quat_exp(omega: vec3f, dt: f32) -> vec4f { /* ... */ }

// Quaternion to matrix (matrix batch multiply)
fn quat_to_mat3(q: vec4f) -> mat3x3f { /* ... */ }
```

#### Batch Integration: Semi-implicit Euler
GPU kernel for integration step:
```wgsl
@compute @workgroup_size(256)
fn integrate_semi_implicit(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let world_idx = gid.x;
    let dof_idx = gid.y;

    // Load q, v, qdd for this DOF
    let q = q_buffer[world_idx][dof_idx];
    let v = v_buffer[world_idx][dof_idx];
    let qdd = qdd_buffer[world_idx][dof_idx];

    // Semi-implicit Euler
    let v_new = v + dt * qdd;
    let q_new = q + dt * v_new;

    // Store back
    v_buffer[world_idx][dof_idx] = v_new;
    q_buffer[world_idx][dof_idx] = q_new;
}
```

#### Synchronization & Memory Model
- Read-compute-write barriers between ABA passes
- Use `storageBarrier()` to sync within work group
- Texture-to-buffer copies for CPU readback (reduced bandwidth)

#### Example: 1000 Parallel Pendula
```rust
let ngpu = 1000;
let gpu_sim = GpuSimulator::new(&device, &model, ngpu)?;

let mut states = vec![model.default_state(); ngpu];
// Initialize with random q, v
for s in &mut states {
    s.q[0] = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
    s.v[0] = rand::random::<f64>() * 1.0;
}

gpu_sim.load_states(&states)?;

// Simulate 1000 pendula for 10 seconds
for _ in 0..10000 {
    gpu_sim.step()?;
}

let final_states = gpu_sim.readback_states()?;
```

Benchmark target: 10000 steps of 1000 pendula < 1 second.

#### Tests & Validation
- Compare GPU ABA output vs. CPU for single environment
- Compare batch simulation (nworld=1000) energy conservation to single-environment RK4
- Performance: measure throughput (steps/second for nworld environments)
- Gradient validation: if possible, compute Jacobians on GPU and compare to CPU finite-diff

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 5: Particle Physics (MPM)

### Objective
Implement Material Point Method (MPM) for particles (elastic, plastic, granular, fluid). Couple with rigid bodies.

### Specification

#### tau-particle: MPM Solver
Create `crates/tau-particle/` for material point method.

**Particle state:**
```rust
pub struct Particle {
    pub x: Vec3,           // position (world frame)
    pub v: Vec3,           // velocity
    pub mass: f64,         // particle mass
    pub volume: f64,       // reference volume
    pub F: Mat3,           // deformation gradient
    pub C: Mat3,           // affine velocity field
    pub J: f64,            // determinant of F (volume change)
    pub material: Material,
}

pub enum Material {
    Elastic { E: f64, nu: f64 },           // Young's modulus, Poisson ratio
    Plastic { E: f64, yield: f64 },        // elastic with yield stress
    Granular { phi: f64 },                 // friction angle
    Fluid { viscosity: f64, eos: Eos },    // viscosity + equation of state
}
```

**Grid structure:**
- Eulerian grid (background mesh) for P2G/G2P transfers
- Grid spacing h; nparticles per cell ~ 4-8
- Compute grid bounds from particle positions

**P2G (Particle to Grid):**
Transfer particle mass, momentum, and deformation to grid nodes.

For each particle:
1. Find nearby grid nodes (within support radius, typically 2-3 grid cells)
2. Compute weight w_ip = N(x_p - x_i) (cubic spline kernel)
3. m_i += w_ip * m_p
4. momentum_i += w_ip * m_p * v_p
5. F_i += ∂w_ip/∂x (stress term)

**Constitutive models (stress computation):**

- **Elastic (Neo-Hookean):**
  - ψ(F) = μ/2 (|F|²_F - 3 - 2 ln det F) + λ/2 (ln det F)²
  - P = ∂ψ/∂F; stress on grid from divergence of P

- **Plastic (J-plasticity):**
  - Decompose F = F_e * F_p; track F_p as history
  - Yield criterion: von Mises on deviatoric stress
  - Update: clamp plastic stretch if violated

- **Granular (Drucker-Prager):**
  - Effective friction μ_eff = tan(φ) where φ is friction angle
  - Yield: ||dev σ|| > √3 μ_eff ||σ||_tr

- **Fluid (APIC with EOS):**
  - Pressure from density: p = γ (ρ/ρ₀)^7  (water-like EOS)
  - Or ideal gas: p = ρ_0 c_s² (J - 1) where c_s is sound speed

**Grid forces:**
- Gravity: f_i = m_i * g
- External forces (e.g., from rigid body coupling)
- Boundary conditions: zero velocity at domain edges

**Grid integration:**
```
m_i * a_i = f_i
v_i' = v_i + (f_i / m_i) * dt
```

**G2P (Grid to Particle):**
Update particle state from grid:
```
v_p' = Σ_i w_ip * v_i'
v_p_pic = v_p'
v_p_flip = v_p + (v_p' - Σ_i w_ip * v_i)  // FLIP correction
v_p = (1-α) v_pic + α v_flip               // PIC-FLIP blending
x_p += v_p * dt
F_p = (I + dt * ∇v) * F_p                 // update deformation gradient
```

**Implicit integration (optional, Phase 5+):**
- Use Lax-Friedrichs or semi-implicit scheme for stability in fluids
- Phase 5 goal: explicit time-stepping for simplicity

#### Rigid-Particle Coupling
Integrate with tau-rigid via modified contact forces:

1. **Collision detection:** find particles inside rigid bodies or overlapping surfaces
2. **Penalty coupling:** apply reaction forces to particles from bodies, and equal-opposite to bodies
3. **Momentum transfer:** particles hitting body → impulse, body deformation → particle scatter

Alternative: use MPM grid as a deformable surface, compute contact normals from distance function.

#### Examples
- **Granular column collapse:** 10k particles, tall column collapses; measure runout distance
- **Water filling container:** fluid MPM, rigid walls; visual test
- **Rigid body in granular:** sphere falls into sand pile; should create crater

#### Tests
- Particle mass conservation: Σ m_p before/after step
- Momentum conservation: Σ p_p = Σ m_p * v_p (no external forces)
- Energy decay: kinetic energy decreases over time (viscous dissipation)
- Material model: verify stress-strain curve for elastic material

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 6: Python + vcad Integration

### Objective
Create PyO3 bindings for tau, integrate with vcad-kernel-physics, enable JAX/NumPy interoperability.

### Specification

#### tau-py: PyO3 Bindings
Create `crates/tau-py/` with Python interface.

**Cargo.toml setup:**
```toml
[package]
name = "tau-py"
...

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
tau = { path = "../tau" }
tau-rigid = { path = "../tau-rigid" }
...

[lib]
name = "tau"
crate-type = ["cdylib"]
```

**Core Python classes:**
```python
import tau

# Build model (Python constructor wrapping ModelBuilder)
model = tau.Model.from_builder() \
    .add_revolute_body("link1", -1, ..., ...) \
    .add_revolute_body("link2", 0, ..., ...) \
    .build()

# Create state
state = model.default_state()
state.q[0] = 0.5
state.v[0] = 0.1

# Step simulation
sim = tau.Simulator()
sim.step(model, state)

# Access state
print(state.q, state.v, state.time)

# Jacobians
jac = sim.step_with_jacobians(model, state)
dq_dq = jac.dqnext_dq  # NumPy array
```

**NumPy/JAX interop:**
- State arrays (q, v) as NumPy arrays; auto-convert to/from Rust DVec
- Jacobian matrices as NumPy or JAX DeviceArray
- Support `__array_interface__` for zero-copy access

**Model loading from MJCF:**
```python
loader = tau.MjcfLoader("ant.xml")
model = loader.build_model()
```

#### vcad-kernel-physics Integration
Replace Rapier with tau in vcad-kernel-physics:

1. **Geometry conversion:** BRep faces → collision geometry (mesh or AABB)
2. **Constraint system:** vcad constraints → tau joints
3. **Simulation loop:** vcad step → tau.Simulator.step()
4. **Rendering bridge:** tau State → vcad body poses

#### Gym Interface Compatibility
Implement `gym.Env` wrapper:
```python
class TauEnv(gym.Env):
    def __init__(self, model_path: str):
        self.model = tau.MjcfLoader(model_path).build_model()
        self.state = self.model.default_state()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,)
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.model.nv,)
        )

    def reset(self):
        self.state = self.model.default_state()
        return self._get_obs()

    def step(self, action):
        self.state.ctrl[:] = action
        sim = tau.Simulator()
        sim.step(self.model, self.state)
        reward = self._compute_reward()
        done = self._is_done()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.state.q, self.state.v])
```

#### Example: Ant in JAX
```python
import jax
import jax.numpy as jnp
import tau

model = tau.MjcfLoader("ant.xml").build_model()

@jax.jit
def step_fn(state, action):
    state.ctrl[:] = action
    tau.Simulator().step(model, state)
    return state

# Vectorized over batch of initial conditions
@jax.vmap
def batch_step(state):
    return step_fn(state, jnp.zeros(model.nv))
```

#### Tests
- Load MJCF and verify model structure matches expected (body count, joint types)
- Python class instantiation, state manipulation
- NumPy array conversion round-trip
- Gym environment step/reset compatibility
- JAX JIT compilation of simulation loop

### Verify
```bash
cd crates/tau-py && maturin develop && python -c "import tau; print(tau.__version__)"
cd /Users/cam/Developer/tau && cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 7: World Generation + Training

### Objective
Add procedural world generation, sensor models, tendons, and data export for ML training.

### Specification

#### World Generation API
Create `tau-world` module with procedural generation:

```rust
pub struct WorldGenerator {
    seed: u64,
    rng: StdRng,
}

impl WorldGenerator {
    pub fn new(seed: u64) -> Self { /* ... */ }

    pub fn random_chain(
        &mut self,
        nlinkages: usize,
        mass_range: [f64; 2],
        length_range: [f64; 2],
    ) -> Model { /* ... */ }

    pub fn random_quadruped(&mut self) -> Model { /* ... */ }

    pub fn platform_with_obstacles(
        &mut self,
        width: f64,
        nobs: usize,
        obs_height_range: [f64; 2],
    ) -> Model { /* ... */ }
}
```

#### Sensor Models
Extend State with optional sensor outputs:

```rust
pub enum Sensor {
    JointState { joint_idx: usize },           // q[i], v[i]
    BodyAccel { body_idx: usize },             // linear acceleration
    BodyAngularVel { body_idx: usize },        // ω in body frame
    ForceTorque { body_idx: usize },           // reaction wrench
    Rangefinder { body_idx: usize, max_dist: f64 }, // distance to nearest obstacle
    Imu { body_idx: usize },                   // acceleration + angular velocity
    FrameCapture { body_idx: usize },          // body transform snapshot
}

pub struct SensorOutput {
    pub sensor_id: usize,
    pub timestamp: f64,
    pub data: Vec<f64>,  // flattened output
}

pub struct World {
    pub model: Model,
    pub state: State,
    pub sensors: Vec<Sensor>,
    pub sensor_history: Vec<Vec<SensorOutput>>,
}

impl World {
    pub fn step(&mut self) {
        sim.step(&self.model, &mut self.state);
        // Record sensor outputs
        for sensor in &self.sensors {
            self.sensor_history.push(self.read_sensor(sensor));
        }
    }

    fn read_sensor(&self, sensor: &Sensor) -> Vec<SensorOutput> { /* ... */ }
}
```

#### Tendon Model
Simple 1D actuator along body chain:

```rust
pub struct Tendon {
    pub path: Vec<usize>,      // body indices
    pub stiffness: f64,        // spring constant
    pub rest_length: f64,
    pub max_force: f64,        // saturation
}

impl Tendon {
    pub fn current_length(&self, state: &State) -> f64 {
        // Sum line segments along path
    }

    pub fn compute_forces(&self, state: &State) -> Vec<(usize, SpatialVec)> {
        // Return applied forces on each body
    }
}
```

Integrate tendon forces into ABA as external forces.

#### Data Export for Training
```rust
pub struct TrajectoryRecorder {
    states: Vec<(Vec<f64>, Vec<f64>, f64)>,  // (q, v, t)
    actions: Vec<Vec<f64>>,                   // ctrl inputs
    sensors: Vec<Vec<SensorOutput>>,
}

impl TrajectoryRecorder {
    pub fn to_numpy_dict(&self) -> HashMap<String, ndarray::Array2<f64>> {
        // q: (nsteps, nq)
        // v: (nsteps, nv)
        // ctrl: (nsteps, nv)
        // time: (nsteps,)
    }

    pub fn to_hdf5(&self, path: &str) -> Result<(), Error> { /* ... */ }

    pub fn to_json(&self, path: &str) -> Result<(), Error> { /* ... */ }
}
```

#### WASM Build
Update Cargo.toml for wasm32-unknown-unknown target:
```toml
[target.wasm32-unknown-unknown]
# Build with --features="wasm"

[features]
wasm = ["web-sys", "wasm-bindgen"]
```

Create `crates/tau-wasm/` wrapper for browser simulation:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmSimulator {
    model: Model,
    state: State,
    sim: Simulator,
}

#[wasm_bindgen]
impl WasmSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(model_json: &str) -> WasmSimulator { /* ... */ }

    #[wasm_bindgen]
    pub fn step(&mut self) {
        self.sim.step(&self.model, &mut self.state);
    }

    #[wasm_bindgen]
    pub fn get_state_json(&self) -> String {
        serde_json::to_string(&(self.state.q, self.state.v)).unwrap()
    }
}
```

#### Examples
- Procedurally generate and train 10 random chains with reinforcement learning (dummy RL policy)
- Export sensor data from multi-environment batch to JSON
- WASM demo: interactive 3D visualization with WebGL

#### Tests
- Tendon force computation correctness
- Sensor output validity (bounds checks)
- Trajectory export format compatibility

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 8: Electromagnetic Field Simulation (FDTD)

### Objective
Implement Yee grid FDTD solver for Maxwell's equations on GPU.

### Specification

#### tau-em: FDTD Solver
Create `crates/tau-em/` for electromagnetic simulation.

**Yee grid structure:**
- E-fields at cell edges; H-fields at cell faces
- Staggered grid avoids checkerboard instability
- Each cell (i, j, k); E at (i±0.5, j, k) etc.; H offset by half

**Maxwell's equations (in vacuum):**
```
∇ × E = -∂B/∂t = -μ₀ ∂H/∂t
∇ × H = ∂D/∂t + J = ε₀ ∂E/∂t + σ E
```

**Discretized update (Courant-stable with CFL = 1/√3):**
```
H_new[i] = H_old[i] - (dt/μ₀) ∇ × E
E_new[i] = (1 - σ dt/ε₀) E_old[i] + (dt/ε₀) ∇ × H
```

**Grid structure:**
```rust
pub struct YeeGrid {
    pub nx: usize, ny: usize, nz: usize,
    pub dx: f64,                            // grid spacing
    pub dt: f64,
    pub eps_r: Array3D<f64>,                // relative permittivity
    pub mu_r: Array3D<f64>,                 // relative permeability
    pub sigma: Array3D<f64>,                // conductivity
    pub ex, ey, ez: Array3D<f64>,           // E-fields
    pub hx, hy, hz: Array3D<f64>,           // H-fields
}

pub struct FdtdSolver {
    grid: YeeGrid,
    sources: Vec<Source>,
    boundary: BoundaryCondition,
}

pub enum Source {
    PointDipole { pos: Vec3, freq: f64, amplitude: f64 },
    PlaneWave { dir: Vec3, freq: f64, amplitude: f64 },
    CurrentLoop { pos: Vec3, radius: f64, freq: f64 },
}

pub enum BoundaryCondition {
    Pml { order: usize, sigma_max: f64 },   // PML absorbing boundary
    Periodic,
    PerfectConductor,
}
```

**PML (Perfectly Matched Layer):**
- Additional layer of "lossy" material at domain boundaries
- Absorbs outgoing waves without reflection
- Implement via auxiliary differential equations (ADE) formulation
- Order 1-4 typical; higher = more expensive but less reflection

**GPU implementation:**
- Parallel grid update: each work item updates one field component
- Boundary handling: PML coefficients precomputed
- Source injection: add source current at specified cell

**Example waveguide:**
```rust
let solver = FdtdSolver::new(YeeGrid {
    nx: 64, ny: 64, nz: 256,
    dx: 0.1e-6,  // 100 nm
    dt: dx / (3e8 * sqrt(3.0)),
    // ...
});

// TM mode waveguide (perfect conductors at x=0, y=0)
for step in 0..10000 {
    solver.step();
    if step % 100 == 0 {
        let energy = solver.total_energy();
        println!("Step {}: Energy = {}", step, energy);
    }
}
```

**Observables:**
- Total energy: (ε₀ |E|² + |B|²/μ₀) integrated over grid
- Field at probe points: time-series E(r, t)
- Power flux: Poynting vector S = E × H integrated over surface
- Resonance frequencies: FFT of probe signal

#### Differentiable FDTD via Adjoint Method
Forward solve: ∂E/∂t = f(E, H, ρ) where ρ are material parameters.

Adjoint solve: ∂λ/∂t = -∂f/∂E^T λ (backward in time).

Compute gradient: ∂L/∂ρ = ∫ (∂f/∂ρ)^T λ dt.

Implement for loss = |E(probe) - E_target|²; optimize material layout to focus energy.

**Example: waveguide inverter**
- Design material structure to split incident wave equally to 4 output ports
- Loss = minimize variance of port powers
- Optimize via adjoint gradient descent

#### Tests
- Energy conservation: total electromagnetic energy constant with PML boundary
- Plane wave reflection from perfect conductor: should see standing wave
- Antenna radiation pattern: far-field reconstruction from near-field

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 9: Molecular Dynamics

### Objective
Implement molecular dynamics with interatomic potentials (LJ, Coulomb, bonded) and GPU acceleration.

### Specification

#### tau-md: Velocity Verlet Integrator
Create `crates/tau-md/` for MD simulation.

**Velocity Verlet algorithm:**
```
x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt²
a(t+dt) = F(x(t+dt)) / m
v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
```

**Particle representation:**
```rust
pub struct Particle {
    pub x: Vec3,
    pub v: Vec3,
    pub f: Vec3,        // force accumulator
    pub mass: f64,
    pub atom_type: u32, // for force field lookup
}

pub struct MdSystem {
    pub particles: Vec<Particle>,
    pub forces: Vec<(usize, usize, Vec3)>, // (i, j, f_ij) for bonds
}
```

**Force field models:**

- **Lennard-Jones:** V(r) = 4ε [(σ/r)¹² - (σ/r)⁶]
  - Parameters: ε (depth), σ (distance)
  - Common: argon ε=0.01 eV, σ=3.4 Å

- **Coulomb:** V(r) = k q_i q_j / r
  - Charge-based interactions
  - Long-range; typically use Ewald or cutoff

- **Bonded (harmonic):** V(r) = 0.5 k_bond (r - r_0)²
  - Connect specific pairs with spring constant k_bond
  - Rest length r_0

**Neighbor list:**
- Build every N steps (typically 10-20)
- Cutoff r_cut with small buffer (skin)
- Pairs with |r_ij| > r_cut excluded

**Periodic boundary conditions:**
```rust
pub fn minimum_image(&self, dr: Vec3) -> Vec3 {
    let mut r = dr;
    for d in 0..3 {
        if r[d] > 0.5 * self.box_size[d] {
            r[d] -= self.box_size[d];
        } else if r[d] < -0.5 * self.box_size[d] {
            r[d] += self.box_size[d];
        }
    }
    r
}
```

**Thermostat (Langevin dynamics):**
```
f_total = F_intermolecular - γ v + random_force
```
- Damping coefficient γ
- Random force ~ √(2 γ k_B T m) for target temperature T

**Example: Argon fluid**
```rust
let mut system = MdSystem::new(864); // 6×6×24 fcc lattice
system.initialize_velocities(temperature, mass_argon);

for step in 0..1_000_000 {
    system.compute_forces();      // LJ + pair forces
    system.velocity_verlet_step(dt);

    if step % 1000 == 0 {
        let ke = system.kinetic_energy();
        let pe = system.potential_energy();
        let t = 2.0 * ke / (3.0 * system.num_particles() * K_B);
        println!("Step {}: T={:.2}K, E={}", step, t, ke + pe);
    }
}
```

#### GPU Acceleration
Implement force computation on GPU (similar to tau-gpu strategy):
- Neighbor list: broad-phase on GPU
- Pairwise LJ forces: each thread pair → force kernel
- Force accumulation: atomic operations or histogram approach
- Velocity Verlet: matrix operations (batch integration)

#### ML/MM Surrogates
Optionally replace expensive force fields with neural network:
```rust
pub trait ForceField {
    fn compute_force(&self, r: Vec3, atom_i: u32, atom_j: u32) -> Vec3;
}

pub struct NeuralNetworkForceField {
    // Trained model (e.g., via PyTorch/TensorFlow)
    pub model: Arc<dyn Fn(Vec3) -> Vec3>,
}

impl ForceField for NeuralNetworkForceField {
    fn compute_force(&self, r: Vec3, _: u32, _: u32) -> Vec3 {
        (self.model)(r)
    }
}
```

#### Tests
- LJ pair potential: verify force = -dV/dr numerically
- Energy conservation: total energy drifts < 0.1% over 1000 steps
- Radial distribution function (RDF): compare to reference
- Diffusion coefficient: measure via mean-square-displacement

#### Example: Argon radial distribution
Run long equilibration; compute RDF g(r) = ρ(r) / ρ_bulk; compare to literature values.

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 10: Lattice Gauge Theory (QFT)

### Objective
Implement lattice gauge theory (QCD-like) with HMC sampling. Start U(1), progress to SU(2), then SU(3).

### Specification

#### tau-qft: Lattice Gauge Theory Solver
Create `crates/tau-qft/` for quantum field theory on the lattice.

**Lattice QCD basics:**
- 4D spacetime lattice (Nt × Nx × Ny × Nz)
- Link variables U_μ(n) ∈ SU(N), unitary (N×N) matrices
- Wilson action: S = β/N Σ (1 - Re Tr U_plaq) where U_plaq = U_1 U_2 U_1† U_2†
- Partition function: Z = ∫ D[U] exp(-S(U))

**Phase 1: U(1) lattice**
- U(1) = complex phase e^(iθ)
- Action: S = β Σ (1 - cos(θ_μ(n) + θ_ν(n+μ) - θ_μ(n+ν) - θ_ν(n)))
- Simpler than SU(2); tests algorithm structure

```rust
pub struct Lattice<N: Dim> {
    nt: usize, nx: usize, ny: usize, nz: usize,
    links: Array4D<Group<N>>,  // U_μ(n)
    beta: f64,                  // coupling constant
}

pub trait Group: Clone + Debug {
    fn identity() -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn inv(&self) -> Self;
    fn action_density(&self, neighbors: &[Self]) -> f64;
}

pub struct U1 { angle: f64 }
impl Group for U1 { /* ... */ }

pub struct SU2 { q: [f64; 4] } // Quaternion representation
impl Group for SU2 { /* ... */ }

pub struct SU3 { m: [[Complex; 3]; 3] }
impl Group for SU3 { /* ... */ }
```

**Hybrid Monte Carlo (HMC):**
- Introduce conjugate momenta p ~ N(0, 1) for each link
- Hamiltonian: H = 0.5 Σ |p|² + S(U)
- Leapfrog integrator:
  ```
  p ← p - 0.5 dt ∇_U S(U)
  U ← U * exp(i dt p)  [exponential map]
  p ← p - 0.5 dt ∇_U S(U)
  ```
- Metropolis accept/reject: exp(-ΔH)

**Force computation (∇_U S):**
For each link U_μ(n), compute derivative of Wilson action:
```
dS/dU_μ(n) ∝ Σ (plaquettes touching U_μ(n))
```

For U(1): dS/dθ = β sin(...)
For SU(N): dS/dU = matrix derivative; use representation in su(N) Lie algebra

**Observables:**
- Plaquette expectation ⟨Tr U_plaq⟩
- Wilson loop (for confinement): exp(-σ · Area)
- Polyakov loop (deconfinement): Tr Π_n U_0(n, t)
- Chiral condensate (in fermionic extensions)

**Checkerboard parallelization (for GPU):**
- Color lattice sites like checkerboard (alternating black/white)
- Update all black sites in parallel; then all white sites
- No data hazards within color

#### Example: Phase transition in U(1)
```rust
let mut lattice = Lattice::<U1>::new(4, 4, 4, 4);
let betas = [0.5, 1.0, 2.0, 5.0];

for &beta in &betas {
    lattice.beta = beta;

    // Thermalize
    for step in 0..1000 {
        lattice.hmc_step(n_md_steps, dt);
    }

    // Measure
    let plaq = lattice.measure_plaquette();
    println!("β={}, ⟨Tr U_plaq⟩ = {:.4}", beta, plaq);
}

// Near phase transition: plaq jumps from disordered to ordered
```

#### SU(2) and SU(3) Extensions
- Replace U1 with SU2/SU3 in Group trait
- Quaternion or 3×3 matrix representation
- Exponential map: exp(i θ_a T_a) using generators
- Force computation: su(N) Lie algebra derivatives

#### Tests
- Action density: should decrease with HMC steps (on average)
- Autocorrelation: measure integrated autocorr. time τ_int
- Phase transition detection: monitor order parameter
- Energy conservation: Hamiltonian drift in leapfrog

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 11: Multi-Scale Coupling

### Objective
Couple different solvers (QM→MM, EM→rigid, particle→rigid) via handshake regions and subcycling.

### Specification

#### Handshake Regions
Define overlapping spatial regions where two solvers interact:
```rust
pub struct Coupling {
    pub solver_a: SolverType,
    pub solver_b: SolverType,
    pub overlap_region: BoundingBox,
    pub force_transfer: ForceTransfer,
}

pub enum ForceTransfer {
    Direct { damping: f64 },            // F_a = -k (v_a - v_b)
    Flux { rate: f64 },                 // momentum transfer rate
    Barrier { potential: fn(Vec3) -> f64 }, // effective potential
}
```

#### Multi-Solver World
```rust
pub struct MultiSolverWorld {
    pub rigid: RigidBodySolver,
    pub em: FdtdSolver,
    pub particles: MdSystem,
    pub couplings: Vec<Coupling>,
}

impl MultiSolverWorld {
    pub fn step(&mut self) {
        // Step each solver with subcycling
        for _ in 0..dt_fine / dt_coarse {
            self.em.step();      // fine timescale (EM has fastest modes)
        }
        self.rigid.step();       // medium timescale
        for _ in 0..dt_coarse / dt_particle {
            self.particles.step();  // coarse timescale
        }

        // Exchange forces
        for coupling in &self.couplings {
            let f_a = self.compute_coupling_force(&coupling);
            self.apply_force_a(f_a);
            self.apply_force_b(-f_a);  // Newton's 3rd law
        }
    }
}
```

#### QM → MM Boundary (Future)
Sketch for QM/MM coupling:
- QM region: solve Schrödinger equation or DFT for electron density ρ(r)
- MM region: classical mechanics
- Boundary: QM atoms interact with MM atoms via Coulomb + Lennard-Jones
- Force on QM atom: -∇ E_QM - ∇ E_MM-QM

#### Electromagnetic → Rigid Body Coupling
Lorentz force on charged bodies:
```
F = q (E + v × B)
τ = μ × B
```
where μ is magnetic dipole moment.

```rust
pub fn lorentz_force(
    charge: f64,
    position: Vec3,
    velocity: Vec3,
    e_field: &Vec3,
    b_field: &Vec3,
) -> Vec3 {
    charge * (e_field + velocity.cross(b_field))
}

pub fn magnetic_torque(
    dipole_moment: Vec3,
    b_field: &Vec3,
) -> Vec3 {
    dipole_moment.cross(b_field)
}
```

#### Subcycling / IMEX Integration
**r-RESPA (Reference System Propagator Algorithm):**
- Partition forces: F_fast (short-range) + F_slow (long-range)
- Inner loop: dt_fast substeps with F_fast + F_coupling
- Outer loop: one step with F_slow

**Adaptive time-stepping:**
- Monitor local error estimate
- Reduce dt if error > tolerance; increase if < 0.1 × tolerance
- Implemented in Phase 13 (Guardian)

#### Examples
- **Lorentz-forced pendulum:** oscillating EM field drives rotating conductor
- **Ion in magnetic trap:** charged particle confined by B-field, coupled to rigid trap structure
- **Particle-rigid coupling:** soft sphere lands on rigid surface, deforms via MPM

#### Tests
- Energy conservation across coupling (with dissipation accounted)
- Momentum conservation (with external forces removed)
- Stability: no spurious oscillations at interfaces

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 12: Gravity

### Objective
Implement gravity from constant g to full general relativity via 5 layers: Poisson solver, N-body, post-Newtonian, Regge calculus, numerical GR.

### Specification

#### tau-gravity: Layered Gravity Solver
Create `crates/tau-gravity/` with gravity hierarchy.

**Layer 1: Constant g**
```rust
pub struct ConstantGravity { pub g: Vec3 }
impl GravitySolver for ConstantGravity { /* simple */ }
```

**Layer 2: Poisson Solver (Local fields)**
Solve Poisson equation on GPU grid for density-dependent gravity:
```
∇² Φ = 4π G ρ
g = -∇ Φ
```
Using FFT on periodic domain or multigrid for non-periodic.

**Layer 3: N-body (Far-field + small systems)**
Naive: O(N²) pairwise gravitational forces. Optimize with:
- Barnes-Hut tree: group distant particles, O(N log N)
- Fast Multipole Method (FMM): hierarchical expansions, O(N)
- Integration: Particular crate (if available) or custom tree

```rust
pub struct BarnesHutTree {
    root: Box<OctreeNode>,
}

impl BarnesHutTree {
    pub fn compute_forces(&self, particles: &mut [Particle], theta: f64) {
        // θ = angular tolerance; larger θ = faster, less accurate
        for p in particles {
            self.traverse_force(self.root, p, theta);
        }
    }
}
```

**Layer 4: Post-Newtonian Corrections (1PN + 2.5PN)**

1PN (first post-Newtonian) relativistic corrections:
```
a = a_Newton + a_1PN + O(c⁻⁴)

a_1PN = -G²m₁m₂/(c²r³) [vector corrections to Kepler orbit]
```
Effect: perihelion precession (Mercury: 43"/century).

2.5PN gravitational radiation:
```
a_gw ∝ -c⁻⁵ (r̈ × r̈)
```
Radiative energy loss causes binary orbit decay.

Example: Mercury precession
```
Δω = 43.1 arcsec/century (GR prediction)
     43.11 ± 0.45 arcsec/century (observed)
```

**Layer 5: Numerical GR (BSSN formulation)**
Einstein equations in 3+1 ADM formalism:
```
∂_t g_ij = 2 K_ij
∂_t K_ij = -∇_i ∇_j α + α (R_ij + K_i^k K_kj - K K_ij) + 4π (S_ij - 1/3 γ_ij S)
∂_t α = -α K
```
where:
- g_ij: spatial metric
- K_ij: extrinsic curvature
- α: lapse function (time coordinate)
- R_ij: Ricci tensor

BSSN reformulation improves stability. Implement on GPU grid.

**Example: Solar System**
```rust
let mut gravity = PostNewtonianSolver::new(max_order=2.5);

let sun = Particle { x: Vec3::zeros(), m: M_sun, v: Vec3::zeros() };
let mercury = Particle {
    x: Vec3::new(57.9e9, 0.0, 0.0),  // 57.9M km
    m: M_mercury,
    v: Vec3::new(0.0, 47.4e3, 0.0),   // 47.4 km/s
};

let mut system = vec![sun, mercury];

for step in 0..1_000_000 {
    gravity.compute_forces(&mut system);
    system.velocity_verlet_step(dt);

    if step % 10000 == 0 {
        let perihelion = compute_perihelion(&system);
        let precession = (perihelion - perihelion_0) * 180 / π;
        println!("Perihelion precession: {:.2} arcsec", precession);
    }
}

// Expected: 43.1 arcsec/century
```

**Regge Calculus (Simplicial gravity):**
Alternative to differential geometry using simplicial complexes:
- Vertices, edges (labeled by edge length)
- Deficit angles encode curvature
- Equations of motion: vary edge lengths to minimize action

Advanced topic; Phase 12+ stretch goal.

#### Tests
- 1PN: Mercury perihelion precession = 43.1 ± 0.5 arcsec/century
- 2.5PN: binary neutron star orbit decay timescale
- Energy conservation: total energy (kinetic + potential + gravitational radiation) conserved

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 13: Self-Correcting Simulation

### Objective
Monitor conservation laws, adapt time-stepping, and gracefully degrade when accuracy drops.

### Specification

#### tau-guardian: Adaptive Time-Stepping + Conservation Monitoring
Create `crates/tau-guardian/` for simulation health management.

**Conservation monitors:**
```rust
pub struct ConservationMonitor {
    pub energy_error: f64,
    pub momentum_error: Vec3,
    pub angular_momentum_error: Vec3,
}

impl ConservationMonitor {
    pub fn check(&self, state: &State, model: &Model) -> ConservationMonitor {
        let energy = total_energy(model, state);
        let momentum = total_momentum(state);
        let ang_momentum = total_angular_momentum(state, model);

        ConservationMonitor {
            energy_error: (energy - self.baseline_energy).abs() / self.baseline_energy,
            momentum_error: momentum - self.baseline_momentum,
            angular_momentum_error: ang_momentum - self.baseline_ang_momentum,
        }
    }
}
```

**Shadow Hamiltonian (modified Hamiltonian structure-preservation):**
Symplectic integrators conserve a "shadow" Hamiltonian H_shadow = H + O(dt²).
Monitor |H_shadow - H| to detect instability.

**Adaptive time-stepping (PI controller):**
```
dt_new = dt_old * (tolerance / error)^(0.4) * (tolerance / error_prev)^(0.2)
```
Increase dt if error dropping; decrease if rising.

**Embedded RK method (for error estimation):**
Use two RK methods of different orders (e.g., RK4 and RK5):
```
error ≈ |y_rk4 - y_rk5| / dt
```

**Auto-switching:**
```
if error > loose_tolerance:
    switch to RK4 (more robust)
elif error < tight_tolerance:
    switch to RK2 (faster)
```

#### Graceful Degradation
```rust
pub enum SolverQuality {
    Excellent,  // error < 0.01 tolerance
    Good,       // 0.01 < error < 0.1 tolerance
    Marginal,   // 0.1 < error < tolerance
    Poor,       // error > tolerance, dt reduced
    Critical,   // multiple conservation law violations
}

pub fn degrade_strategy(quality: SolverQuality) {
    match quality {
        Excellent => { /* proceed normally */ },
        Good => { /* no action */ },
        Marginal => { /* reduce dt by 50%, reduce integration order */ },
        Poor => { /* reduce dt by 90%, switch to symplectic RK2 */ },
        Critical => { /* pause simulation, alert user */ },
    }
}
```

#### r-RESPA / Multi-Rate Integration
Reference System Propagator Algorithm:
```rust
pub fn rrespa_step(slow_forces: &dyn Fn() -> Vec3, fast_forces: &dyn Fn() -> Vec3) {
    let n_inner = (dt_outer / dt_inner).ceil() as usize;

    // Outer half-step slow force
    v += 0.5 * slow_forces() * dt_outer;

    for _ in 0..n_inner {
        // Inner full-step fast force
        v += fast_forces() * dt_inner;
        x += v * dt_inner;
    }

    // Outer half-step slow force
    v += 0.5 * slow_forces() * dt_outer;
}
```

#### Examples
- Run pendulum with decreasing tolerance; watch dt adapt
- Double pendulum: monitor energy error over 100 seconds
- Chaotic system (e.g.,3-body): detect when solution becomes untrustworthy

#### Tests
- Energy error stays below tolerance
- No spurious conservation law violations
- dt adaptation monotonic (not oscillating wildly)
- Switching between methods smooth (no jumps)

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 14: Emergent Physics

### Objective
Implement Lattice Boltzmann method (LBM) and equation-free simulation for emergent dynamics.

### Specification

#### tau-lbm: Lattice Boltzmann Method
Create `crates/tau-lbm/` for LBM solver.

**Lattice Boltzmann basics:**
- Distribution function f_i(x, t) at each lattice site and discrete velocity e_i
- Nine velocities in 2D (D2Q9), 27 in 3D (D3Q27):
  ```
  e_0 = [0, 0]        (rest)
  e_1...e_4 = [±1, 0], [0, ±1]
  e_5...e_8 = [±1, ±1]  (diagonals)
  ```

- Each f_i carries momentum/energy/stress information

**BGK collision operator (Bhatnagar-Gross-Krook):**
```
f_i(t+1) = f_i(t) - (1/τ) [f_i(t) - f_i^eq(t)]
```
where f_i^eq is equilibrium distribution and τ is relaxation time (related to viscosity).

**Equilibrium distribution (D3Q27):**
```
f_i^eq = w_i ρ [1 + 3 (e_i · u) + 9/2 (e_i · u)² - 3/2 (u · u)]
```
where u is macroscopic velocity and w_i are weights (tabulated).

**Macroscopic variables (moments):**
```
ρ = Σ_i f_i
u = (1/ρ) Σ_i f_i e_i
P = c_s² ρ I + (1/ρ) Σ_i f_i (e_i - u)(e_i - u)^T
```
where c_s = 1/√3 is lattice sound speed.

**Example: lid-driven cavity (Navier-Stokes verification)**
```rust
let mut lbm = LatticeBoltzmann::new(64, 64);  // 2D grid
lbm.initialize_uniform(density=1.0, u=0.0);

// Boundary conditions
for x in 0..64 {
    lbm.set_velocity_bc(x, 63, Vec2::new(0.1, 0.0));  // lid at u=0.1
    lbm.set_no_slip_bc(x, 0);                         // bottom wall
}

for step in 0..10000 {
    lbm.collide_and_stream();

    if step % 1000 == 0 {
        let u_max = lbm.max_velocity();
        println!("Step {}: max velocity = {:.4}", step, u_max);
    }
}

// Compare velocity profile at x=0.5 to analytical Navier-Stokes solution
```

#### Equation-Free / Coarse Projective Integration
For systems with multiple timescales, extract low-dimensional dynamics:

1. **Lift:** fine-scale microscopic state → coarse macro state (e.g., density)
2. **Simulate:** fine solver for short burst (micro time)
3. **Restrict:** micro state → macro state
4. **Extrapolate:** project macro dynamics (equation-free time step)

```rust
pub struct EquationFreeWrapper<F: FineSolver, C: CoarseProjector> {
    fine: F,
    coarse: C,
}

impl EquationFreeWrapper {
    pub fn step(&mut self, macro_state: &mut Vec<f64>) {
        let micro = self.coarse.lift(macro_state);

        // Inner microscopic simulation
        for _ in 0..t_burst / dt_micro {
            self.fine.step(&micro);
        }

        let macro_new = self.coarse.restrict(&micro);
        // Extrapolate
        let dmacro = &macro_new - macro_state;
        *macro_state = &macro_new + dmacro;  // simple forward extrapolation
    }
}
```

#### Emergence Detection
Measure whether fine-scale dynamics converge to low-dimensional manifold:

```rust
pub fn effective_information(micro_states: &[Vec<f64>], coarse_dim: usize) -> f64 {
    // PCA: project to coarse_dim principal components
    // Measure: mutual information I(full; coarse)
    // High I → strong emergence
}
```

#### Example: Lattice Boltzmann → Navier-Stokes
Show that LBM at long wavelengths/times recovers:
```
∂u/∂t + (u·∇)u = -∇p + ν ∇²u
∇·u = 0
```
Verification: compare cavity flow from LBM to FEM Navier-Stokes solver.

#### Tests
- LBM energy conservation: total KE decreases smoothly (viscous dissipation)
- Navier-Stokes compatibility: lid-driven cavity matches reference
- Equation-free convergence: coarse dynamics correct for RD or LBM source

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 15: Physics Compiler

### Objective
Create a domain-specific language + JIT compiler for physics kernels.

### Specification

#### tau-compile: Physics IR + Kernel Fusion
Create `crates/tau-compile/` for compilation pipeline.

**Physics DSL (Rust macros or custom syntax):**
```rust
physics_kernel! {
    kernel heat_diffusion(T: &mut Field3D, dt: f64) {
        let dx = 1.0;
        let kappa = 1.0;

        stencil T_new {
            center: T[i, j, k],
            laplacian: (T[i+1,j,k] + T[i-1,j,k] - 2*T[i,j,k]) / (dx*dx),
            // ... y, z
        }

        for (i, j, k) in T.iter_mut() {
            T_new[i,j,k] = T[i,j,k] + kappa * laplacian[i,j,k] * dt;
        }
    }
}
```

**Physics IR (expression graph):**
```rust
pub enum PhysicsOp {
    Load(FieldRef),
    Store(FieldRef, Box<PhysicsOp>),
    StencilOp { name: String, args: Vec<PhysicsOp> },
    BinOp { op: BinOp, lhs: Box<PhysicsOp>, rhs: Box<PhysicsOp> },
    Constant(f64),
}

pub struct PhysicsProgram {
    ops: Vec<PhysicsOp>,
    fields: HashMap<String, FieldMeta>,
}
```

**Compilation to Naga IR:**
- Convert physics IR → wgpu/Naga intermediate representation
- Target backends: WGSL, SPIR-V, Metal Shading Language

**Kernel fusion:**
- Fuse adjacent stencils to reduce memory bandwidth
- Example: diffusion + advection into single kernel
- Analysis: detect when two ops can be fused (same domain, compatible schedules)

**Auto-differentiation pass:**
- Augment kernel with forward-mode AD
- Track tangent values alongside primal
- Example: if primal computes T_new from T, tangent computes dT_new from dT

**Scheduling hints:**
```rust
physics_kernel! {
    kernel stencil_op(u: &Field3D, v: &Field3D) {
        schedule {
            tile_size: (16, 16, 16),      // work group shape
            loop_unroll: true,
            prefetch: true,
            fuse_next: "stencil_op2",     // fuse with next kernel
        }
        ...
    }
}
```

#### Example: Heat Equation Compiler
```rust
let heat_kernel = physics_kernel! {
    fn heat_pde(T: &mut Field3D, dt: f64) {
        // ∂T/∂t = κ ∇²T
        for (i, j, k) in T.iter_mut() {
            let lap = (T[i+1,j,k] + T[i-1,j,k] + T[i,j+1,k] + T[i,j-1,k]
                      + T[i,j,k+1] + T[i,j,k-1] - 6*T[i,j,k]);
            T[i,j,k] += 1.0 * lap * dt;
        }
    }
};

// Compile to GPU kernel
let gpu_kernel = heat_kernel.compile_to_wgsl()?;

// Use in simulation
for step in 0..1000 {
    gpu_kernel.dispatch(&device, &mut T_buffer, dt)?;
}
```

#### Verification
- Compiled kernel output matches interpreter output
- GPU dispatch produces same results as CPU loop
- Performance: measure speedup from fusion (expect 1.5-3x for heat + advection)

#### Tests
- Simple stencil: nearest-neighbor averaging
- Fusion correctness: fused kernel ≡ unfused chain
- Derivative computation: analytical vs. AD gradients agree

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 16: Probabilistic Simulation

### Objective
Support uncertain parameters, ensemble inference, and randomized algorithms.

### Specification

#### tau-prob: Probabilistic State & Solvers
Create `crates/tau-prob/` for uncertainty quantification.

**Distribution-wrapped state:**
```rust
pub struct Distribution<T> {
    pub samples: Vec<T>,  // particle ensemble
    pub weights: Vec<f64>, // importance weights
}

pub struct ProbabilisticState {
    pub q: Distribution<DVec>,  // nsamples × nq matrix
    pub v: Distribution<DVec>,
    pub parameters: Distribution<Vec<f64>>, // uncertain physical params
}

impl ProbabilisticState {
    pub fn mean(&self) -> (DVec, DVec) {
        let q_mean = self.q.weighted_mean();
        let v_mean = self.v.weighted_mean();
        (q_mean, v_mean)
    }

    pub fn covariance(&self) -> (DMat, DMat) {
        // Σ w_i (x_i - μ)(x_i - μ)^T
    }
}
```

**Ensemble simulation (batch with sampling):**
```rust
pub fn ensemble_step(
    model: &Model,
    state: &mut ProbabilisticState,
    sim: &Simulator,
) {
    // Step each particle in ensemble
    for i in 0..state.q.samples.len() {
        let mut s = State {
            q: state.q.samples[i].clone(),
            v: state.v.samples[i].clone(),
            ..
        };

        // Apply sampled parameters to model
        let model_i = model.with_params(&state.parameters.samples[i]);
        sim.step(&model_i, &mut s);

        state.q.samples[i] = s.q;
        state.v.samples[i] = s.v;
    }

    // Resampling (e.g., if weights become unbalanced)
    state.resample(threshold=0.5);
}
```

**Randomized smoothing for contacts:**
When contact detection is uncertain (numerical noise near boundary), perturb contact normal:
```
n_perturbed ~ N(n, σ² I)
```
Average contact forces over samples → smooth, differentiable contact solver.

**SVGD (Stein Variational Gradient Descent):**
For parameter inference, optimize particle locations to match observations:
```
x_i^{t+1} = x_i^t + ε [∇_x log p(x) + (1/N) Σ_j k(x_i, x_j) ∇_x log p(x_j)]
```
Kernel k encourages diversity; gradient term maximizes posterior.

#### Stochastic Computation Graph
Track uncertainty through algorithm:
```
state_ensemble --ABA--> accelerations_ensemble --integrate--> new_state_ensemble
                                                                       |
                                                              track variance
```

#### Observables with Uncertainty
```rust
pub fn trajectory_uncertainty(
    trajectories: &[Vec<State>],
) -> Vec<(Vec3, Vec3)> {
    // For each timestep, compute mean ± std of body position
    vec![(mean_x, std_x), ...]
}
```

#### Example: Ensemble Cartpole with Parameter Uncertainty
```rust
let mut model = make_cartpole();

// Sample parameters: length, mass, friction
let nsamples = 100;
let mut state = ProbabilisticState::uniform_samples(model, nsamples);
state.parameters.sample_from(
    |rng| {
        length: Normal::new(1.0, 0.1).sample(rng),
        mass: Normal::new(1.0, 0.05).sample(rng),
        friction: Uniform::new(0.0, 0.5).sample(rng),
    }
);

for step in 0..1000 {
    ensemble_step(&model, &mut state, &sim);

    if step % 100 == 0 {
        let (q_mean, q_std) = state.q.mean_and_std();
        println!("Step {}: q = {:.3} ± {:.3}", step, q_mean[0], q_std[0]);
    }
}
```

#### Tests
- Ensemble mean trajectory ≈ deterministic trajectory (at central parameters)
- Ensemble standard deviation grows over time (chaos)
- Resampling maintains moment preservation
- SVGD convergence: particles spread to cover posterior

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Phase 17: Reality Ingestion + Universal Format

### Objective
Inverse problems (real2sim) and universal multi-domain format (.tau JSON schema).

### Specification

#### tau-real2sim: Differentiable Parameter Estimation
Create `crates/tau-real2sim/` for trajectory matching and system identification.

**Trajectory matching (motion capture):**
```rust
pub struct TrajectoryMatcher {
    pub reference: Trajectory,        // motion capture data
    pub loss_weights: LossWeights,
}

impl TrajectoryMatcher {
    pub fn compute_loss(
        &self,
        model: &Model,
        params: &PhysicsParams,
    ) -> f64 {
        let mut state = model.default_state();
        let mut loss = 0.0;

        for (t, obs) in self.reference.iter() {
            // Run simulation
            let sim = Simulator::new();
            sim.step(model, &mut state);

            // Compare to observation
            let pred = state.body_xform[obs.body_idx].translation;
            loss += (pred - obs.position).norm_squared();
        }
        loss
    }

    pub fn gradient_descent(&self, model: &mut Model, params: &mut PhysicsParams) {
        for _ in 0..100 {
            let grad = numerical_gradient(|p| self.compute_loss(model, p), params);
            params -= 0.01 * grad;  // gradient step
        }
    }
}
```

**Observation adapters (from different sensors):**
```rust
pub trait ObservationAdapter {
    fn extract(&self, state: &State, model: &Model) -> Vec<f64>;
}

pub struct JointAngleObserver { joint_idx: usize }
impl ObservationAdapter for JointAngleObserver {
    fn extract(&self, state: &State, _: &Model) -> Vec<f64> {
        vec![state.q[self.joint_idx]]
    }
}

pub struct EndEffectorPoseObserver { body_idx: usize }
impl ObservationAdapter for EndEffectorPoseObserver {
    fn extract(&self, state: &State, _: &Model) -> Vec<f64> {
        let xf = state.body_xform[self.body_idx];
        [xf.translation.x, xf.translation.y, xf.translation.z].to_vec()
    }
}
```

**Iterative refinement (parameter sweep + gradient refinement):**
```
1. Grid search over coarse parameter space
2. Gradient descent from best grid point
3. Uncertainty estimate: Hessian @ optimum
4. Refine observables if residual too large
```

#### tau-format: Universal Physics Format
Create `.tau` JSON schema for multi-domain physics specification.

**Schema structure:**
```json
{
  "version": "1.0",
  "name": "ant-robot",
  "description": "MuJoCo Ant model",

  "world": {
    "gravity": [0.0, 0.0, -9.81],
    "dt": 0.001,
    "default_contact_material": { "stiffness": 10000, "damping": 100 }
  },

  "domains": {
    "rigid_body": {
      "type": "rigid-body-dynamics",
      "bodies": [
        { "name": "torso", "mass": 1.0, "inertia": [...] },
        { "name": "leg1", "mass": 0.5, "inertia": [...] }
      ],
      "joints": [
        { "type": "free", "parent": "world", "child": "torso" },
        { "type": "revolute", "parent": "torso", "child": "leg1", "axis": [1,0,0] }
      ]
    },
    "collision": {
      "type": "convex-collision",
      "geometries": [
        { "body": "torso", "shape": "box", "half_extents": [0.1, 0.1, 0.3] },
        { "body": "leg1", "shape": "capsule", "radius": 0.05, "length": 0.4 }
      ]
    },
    "particles": {
      "type": "mpm",
      "material": "sand",
      "particles": 10000,
      "grid_resolution": [32, 32, 32]
    }
  },

  "couplings": [
    {
      "name": "collision-contact",
      "source": "collision",
      "target": "rigid_body",
      "force_transfer": "direct",
      "damping": 0.1
    }
  ],

  "parameters": {
    "masses": { "type": "distribution", "base": [1.0, 0.5], "uncertainty": 0.1 },
    "friction": { "type": "scalar", "value": 0.5 }
  },

  "importers": [
    { "format": "mjcf", "source": "ant.xml" },
    { "format": "urdf", "source": "robot.urdf" },
    { "format": "usd", "source": "scene.usda" }
  ]
}
```

**Multi-domain coupling rules:**
```json
{
  "couplings": [
    {
      "solver_a": "rigid_body",
      "solver_b": "em",
      "overlap_region": { "type": "aabb", "min": [-1,-1,-1], "max": [1,1,1] },
      "force_transfer": "lorentz"
    },
    {
      "solver_a": "rigid_body",
      "solver_b": "particles",
      "interface": "contact",
      "penalty_stiffness": 1000
    }
  ]
}
```

**Format parsers & loaders:**
```rust
pub fn load_tau_model(path: &str) -> Result<MultiDomainWorld, Error> {
    let json = std::fs::read_to_string(path)?;
    let spec: TauSpec = serde_json::from_str(&json)?;

    // Build each domain
    let rigid = build_rigid_body(&spec.domains["rigid_body"])?;
    let collision = build_collision(&spec.domains["collision"])?;

    // Set up couplings
    let couplings = build_couplings(&spec.couplings)?;

    Ok(MultiDomainWorld { rigid, collision, couplings })
}

// MJCF importer
pub fn from_mjcf(path: &str) -> Result<TauSpec, Error> {
    let loader = MjcfLoader::from_file(path)?;
    let model = loader.build_model();
    // Convert to TauSpec
}

// URDF importer
pub fn from_urdf(path: &str) -> Result<TauSpec, Error> { /* ... */ }

// USD importer (with physics)
pub fn from_usd(path: &str) -> Result<TauSpec, Error> { /* ... */ }

// SDF (Gazebo)
pub fn from_sdf(path: &str) -> Result<TauSpec, Error> { /* ... */ }
```

#### Example: real2sim pendulum
```rust
// 1. Load real motion capture data
let mocap = load_mocap_file("pendulum_motion.c3d")?;

// 2. Define model and parameters to optimize
let mut model = build_pendulum_model();
let mut params = PhysicsParams {
    length: 1.0,
    mass: 1.0,
    friction: 0.01,
    ..
};

// 3. Matcher
let matcher = TrajectoryMatcher {
    reference: mocap,
    loss_weights: LossWeights {
        position: 1.0,
        velocity: 0.1,
        energy: 0.01,
    },
};

// 4. Optimize
matcher.gradient_descent(&mut model, &mut params);

// 5. Export to .tau format
let world_spec = TauSpec {
    domains: {
        "rigid_body": build_rigid_from_model(&model),
    },
    parameters: {
        "length": params.length,
        "mass": params.mass,
        "friction": params.friction,
    },
};
export_tau(&world_spec, "pendulum_identified.tau")?;
```

#### Tests
- Load MJCF, URDF, USD, SDF files; verify model structure
- .tau schema validation (JSON schema)
- Round-trip: build model → export .tau → load .tau → same model
- real2sim: trajectory matching convergence

### Verify
```bash
cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all --check
```

---

## Implementation Notes

### Cross-Phase Dependencies
- Phases 2-3 extend Phase 1 (multi-joint, collision)
- Phase 4 (GPU) builds on Phases 2-3 with parallelization
- Phases 5-10 are largely independent (can be done in parallel)
- Phase 11+ couples earlier phases
- Phase 13 (guardian) improves all earlier phases
- Phases 15-17 integrate and optimize earlier work

### Testing Strategy
- Unit tests for each new crate
- Integration tests combining phases
- Benchmarks for performance-critical paths (ABA, GPU kernels, GJK)
- Validation against reference implementations (MuJoCo, PhysX, etc.)

### Documentation
- Each crate: high-level docs in `lib.rs`
- Examples: well-commented, pedagogical
- Papers: cite key references for each solver
- API stability: keep public interfaces stable after Phase 6

### Performance Targets (Aspirational)
- Single pendulum: < 1 μs per step (CPU)
- 1000 parallel pendula: < 1 ms per step (GPU)
- 100M particle MPM: < 10 ms per step (GPU)
- 1000-body rigid body scene: < 100 ms per step (CPU)
