//! Trajectory adjoint through the **convex contact solve** — the same contact
//! model the forward simulator runs.
//!
//! # Why this module exists
//!
//! [`crate::rollout`]'s adjoint differentiates a self-contained per-vertex
//! ground-penalty contact model. The forward simulation path
//! (`phyz::Simulator::step_with_contacts`) does not use that model: it detects
//! ground contacts with a margin, assembles the convex problem
//! (`phyz_contact::assemble`), and solves it with the active-set Newton in
//! `phyz_contact::convex` under MuJoCo-style solref/solimp stabilization. A
//! gradient of the penalty model is a gradient of *different physics* than the
//! engine integrates — the third instance of the solver/gradient-disagreement
//! bug class in this codebase, and the largest.
//!
//! This module closes it. [`convex_adjoint_gradient`] runs **exactly** the
//! forward step `Simulator::step_with_contacts` runs — same detection, same
//! assembly, same warm-started solve — caches each step's [`phyz_contact::ContactProblem`]
//! and [`ContactSolution`], and backpropagates through the implicit-function-
//! theorem sensitivities of the *actual solve the forward pass performed*
//! ([`phyz_contact::gradient::FixedPointSensitivity`]), at the regimes the
//! solver converged to.
//!
//! # How the backward pass works
//!
//! One contacted step is
//!
//! ```text
//! qdd     = ABA(q, v, u; π)                        (no contact forces)
//! v_free  = v + dt·qdd
//! f*      = argmin ½ fᵀ(A(q,π) + R(q,depth))f + fᵀ b(q,v_free)   s.t. cone
//! v'      = v_free + M⁻¹(q,π) Jᵀ(q) f*
//! q'      = Φ(q, v')                               (integrate_configuration)
//! ```
//!
//! Differentiating: `dv' = dv_free + d(M⁻¹Jᵀ f*)|_{f fixed} + M⁻¹Jᵀ df*`,
//! and by the IFT at the converged fixed point with the active set held
//! fixed, `df*` follows from the exact map linearization
//! ([`FixedPointSensitivity::apply`]): the differential of the stationarity
//! residual `(A + R)f* + b − e_n·bias` (impulses held fixed) on the sticking
//! rows and sliding normal rows, plus the sliding tangential channel
//! (slip-direction rotation) through each sliding contact's own tangential
//! block. Separating contacts respond to nothing, by construction.
//!
//! The smooth factors (`ABA`, kinematics, assembly, `Φ`) are differentiated by
//! **central finite differences per lane** — the same discipline
//! [`crate::semi_implicit_step_jacobians`] documents for its ABA block. The
//! contact solve itself is *never* finite-differenced: its derivative is the
//! exact IFT sensitivity of the converged solution, which is what makes the
//! gradient the derivative of the physics rather than of the solver
//! iterations. Within a lane the contact set is **frozen** to the identities
//! the nominal step detected (each contact is re-evaluated from its material
//! anchor on the body), so a lane never re-runs detection and cannot flip the
//! active set.
//!
//! # Where the gradient is exact, one-sided, or undefined
//!
//! - **Interior of a regime** (a contact firmly separated beyond the margin,
//!   firmly loaded and sticking, or firmly sliding): the gradient is the exact
//!   derivative of the regularized contact model, up to central-difference
//!   error (`~1e-9` relative) in the smooth blocks.
//! - **Contact activation**: the margin band (see
//!   `ContactMaterial::impedance_at`) ramps the impedance smoothly to zero at
//!   the band's outer edge, so a contact entering or leaving the *detected*
//!   set does so at zero force and the gradient is continuous across the
//!   crossing. This is precisely what the margin was added for. The exception
//!   is a manifold truncated at `MAX_MANIFOLD_POINTS`: a point excluded by the
//!   cap can still carry impedance, and the gradient is one-sided there.
//! - **Stick/slide and load/unload transitions**: the true derivative does not
//!   exist at the exact transition; the IFT holds the regime the solver
//!   converged to, so the reported gradient is the one-sided derivative of
//!   that branch. This matches `phyz_contact::gradient`'s documented contract.
//! - **Impact timing**: at a genuine contact-making event the gradient of the
//!   regularized model is biased relative to ideal rigid contact (Suh et al.,
//!   ICML 2022); see `phyz_contact::gradient`'s module docs.
//!
//! # Failure policy
//!
//! A step whose contact solve did not converge is not a fixed point and has
//! no IFT sensitivity; [`FixedPointSensitivity::at`] refuses it by design.
//! This adjoint **degrades loudly**: it returns
//! [`ConvexAdjointError::Unconverged`] (or
//! [`ConvexAdjointError::SingularKkt`] for a degenerate active set) rather
//! than substituting anything. A silent wrong gradient is the failure mode
//! this module exists to eliminate.
//!
//! # Scope
//!
//! Ground **and body-body** contacts. A body-body contact is frozen as a
//! feature *pair* ([`Anchor::Pair`]): both surface points in their own body's
//! frame, the normal in the frame of the body owning the reference face, and
//! the contact point riding the *other* body's vertex — so a lane sees the
//! contact rotate and translate with both bodies. That is what carries the
//! tangential-friction-between-two-moving-bodies channel — a foot on a board's
//! grip tape, a hand on a lever — which is the case a ground-only adjoint
//! cannot express at all.
//!
//! The per-vertex `∂J/∂(mesh vertex)` channel of [`crate::rollout`] is not
//! reproduced here; that adjoint remains the (clearly documented) penalty-model
//! path for the vcad surface-gradient seam.

use phyz_collision::Collision;
use phyz_contact::gradient::FixedPointSensitivity;
use phyz_contact::{
    ContactAssembly, ContactCache, ContactMaterial, ContactSolution, ContactSolverConfig, assemble,
    find_contacts, find_ground_contacts_model_with_drop, regularization_diag, solve_contacts_warm,
};
use phyz_math::{DVec, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration};

use crate::rollout::FinalStateObjective;
pub use crate::rollout::N_INERTIA_PARAMS;
use crate::rollout::inertia_params;

/// Central-difference step for the smooth (non-solve) blocks.
///
/// Smaller than the usual `1e-6` because the margin band's impedance
/// smoothstep has curvature on the `margin ≈ 1e-3 m` length scale: at
/// `h = 1e-6` the truncation error of a lane through the band is a few parts
/// in `1e5` of the step Jacobian, while at `1e-8` it is below `1e-8`. The
/// pieces being differenced are smooth closed-form evaluations (no iterative
/// solve inside), so the round-off floor at this step size is `~1e-8`
/// relative — the truncation/round-off crossover sits near here.
const FD_EPS: f64 = 1e-8;

/// A differentiable rollout through the forward simulator's contact path.
///
/// The forward dynamics are exactly `phyz::Simulator::step_with_contacts`:
/// ground contact detection with margin, convex assembly, warm-started
/// active-set Newton solve, semi-implicit Euler integration.
pub struct ConvexContactRollout<'a> {
    /// Topology, gravity, timestep, damping, geometry and **nominal** inertias.
    pub model: &'a Model,
    /// World z of the ground plane (m).
    pub ground_height: f64,
    /// Contact material for every body (the same single-material convention
    /// `Simulator::step_with_contacts` uses).
    pub material: ContactMaterial,
    /// Solver configuration. Use [`ContactSolverConfig::simulation`] to
    /// differentiate the physics the simulator ships, or
    /// [`ContactSolverConfig::gradients`] for smoother sensitivities.
    pub config: ContactSolverConfig,
    /// Initial joint positions (`Model` layout, length `nq`).
    pub q0: DVec,
    /// Initial joint velocities (length `nv`).
    pub v0: DVec,
    /// Number of steps.
    pub steps: usize,
    /// Open-loop control at step `t` (length `nv`; must not read the state).
    pub ctrl: &'a dyn Fn(usize) -> DVec,
}

/// Everything one unified backward pass produces.
pub struct ConvexAdjointGradients {
    /// The objective at the nominal rollout.
    pub objective: f64,
    /// `dJ/dq0` (length `nq`).
    pub d_q0: DVec,
    /// `dJ/dv0` (length `nv`).
    pub d_v0: DVec,
    /// `dJ/du_t` per step (each length `nv`).
    pub d_ctrl: Vec<DVec>,
    /// `dJ/dπ` per body, canonical packing `[m, cx, cy, cz, Ixx, Iyy, Izz,
    /// Ixy, Ixz, Iyz]`.
    pub d_inertia: Vec<[f64; N_INERTIA_PARAMS]>,
}

/// Why an adjoint pass refused to produce a gradient.
///
/// Every variant is a *loud* refusal in place of a silently wrong number.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvexAdjointError {
    /// A step's contact solve did not converge, so its solution is not a KKT
    /// point and the IFT does not apply.
    Unconverged {
        /// The offending step index.
        step: usize,
        /// Iterations the solve used before giving up.
        iterations: usize,
        /// The residual it stalled at.
        residual: f64,
    },
    /// The KKT matrix at a step's converged active set is singular — the
    /// derivative genuinely does not exist there.
    SingularKkt {
        /// The offending step index.
        step: usize,
    },
    /// Formerly: the trajectory produced a body-body contact, which this
    /// adjoint did not cover.
    ///
    /// Body-body contacts are differentiated now — see [`Anchor::Pair`] — and
    /// nothing constructs this variant any more. It is retained for one release
    /// so that a downstream `match` on [`ConvexAdjointError`] does not break,
    /// and because a caller that special-cased this refusal (falling back to a
    /// sampling optimizer, say) should get a compile-time nudge to delete that
    /// branch rather than silently keeping dead code that will never fire.
    #[deprecated(note = "body-body contacts are supported; this variant is never \
                constructed and will be removed in the next minor release")]
    BodyBodyContact {
        /// The step at which the body-body contact appeared.
        step: usize,
    },
    /// A detected contact could not be pinned to a material anchor on its
    /// body (unsupported geometry variant).
    UnsupportedGeometry {
        /// The offending step index.
        step: usize,
        /// The body carrying the unsupported geometry.
        body: usize,
    },
}

impl core::fmt::Display for ConvexAdjointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unconverged {
                step,
                iterations,
                residual,
            } => write!(
                f,
                "contact solve at step {step} did not converge \
                 ({iterations} iterations, residual {residual:.3e}); \
                 refusing to differentiate a non-KKT point"
            ),
            Self::SingularKkt { step } => write!(
                f,
                "KKT matrix at step {step} is singular at the converged \
                 active set; the derivative does not exist there"
            ),
            #[allow(deprecated)]
            Self::BodyBodyContact { step } => write!(
                f,
                "body-body contact at step {step}: this refusal is retired — \
                 body-body contacts are differentiated"
            ),
            Self::UnsupportedGeometry { step, body } => write!(
                f,
                "could not anchor a ground contact on body {body} at step \
                 {step}: unsupported geometry for the frozen-contact lanes"
            ),
        }
    }
}

impl std::error::Error for ConvexAdjointError {}

// ---------------------------------------------------------------------------
// Frozen contact identities
// ---------------------------------------------------------------------------

/// A contact pinned to the feature pair that produced it, so a perturbed
/// configuration re-evaluates *the same* contact smoothly instead of re-running
/// detection.
///
/// Freezing the feature pair is what makes the FD lanes legitimate. Narrow-phase
/// feature selection (which face is the reference face, which clipped vertices
/// survive, which box corner is deepest) is combinatorial: a lane that re-ran
/// detection could return a manifold with a different *number* of points, and
/// differencing two different contact sets is not a derivative of anything. The
/// design doc sanctions exactly this — §4.4, "feature selection is treated as a
/// discrete decision held fixed for the step" — and it is what MuJoCo and Dojo
/// do. The price is stated there and again in this module's header: the gradient
/// does not see "the manifold would have had a different point if the box had
/// rotated slightly more".
#[derive(Debug, Clone, Copy)]
enum Anchor {
    /// A contact against the world `z = ground_height` plane.
    ///
    /// The support point decomposes as
    /// `p_world(q) = pos(q) + Rᵀ(q)·material_point + world_offset`:
    /// `material_point` is a body-frame point (a box corner, a mesh vertex, a
    /// capsule hemisphere centre, a sphere centre), and `world_offset` is the
    /// constant world-axis part (`−r·ẑ` for a sphere or capsule dropping onto a
    /// z-plane; zero otherwise). This covers every geometry
    /// `find_ground_contacts` emits.
    Ground {
        body: usize,
        material_point: Vec3,
        world_offset: Vec3,
    },
    /// A contact between two moving bodies.
    ///
    /// Both surface points are frozen in their own body's frame, and the normal
    /// is frozen in the frame of whichever body *owns the reference face*
    /// (`normal_frame`), so that rotating that body rotates the contact normal
    /// with it. That `dn/dq` channel is the one carrying "the board tilted, so
    /// the friction direction rotated" — the whole reason a foot-on-grip-tape
    /// gradient is worth having, and the channel a world-frozen normal drops
    /// entirely.
    ///
    /// Freezing the normal in *a* body frame rather than in the world is what
    /// generalizes [`Anchor::Ground`] rather than merely extending it: for a
    /// ground contact the owner is the world, whose frame does not move, so the
    /// world-fixed `+ẑ` of the ground case is this same rule evaluated on a
    /// static owner.
    Pair {
        body_i: usize,
        body_j: usize,
        /// Surface point on `body_i`, in `body_i`'s frame.
        point_i: Vec3,
        /// Surface point on `body_j`, in `body_j`'s frame.
        point_j: Vec3,
        /// The body whose frame `normal_local` is expressed in — see
        /// [`Anchor::reference_body`].
        normal_frame: usize,
        /// Contact normal in `normal_frame`'s body frame. In world terms it is
        /// the direction `body_i` must move to separate, matching
        /// `Collision::contact_normal`.
        normal_local: Vec3,
    },
}

impl Anchor {
    /// Recover the anchor of a detected collision.
    ///
    /// `drop` is a ground contact's world-axis drop as reported by
    /// `find_ground_contacts_model_with_drop`: the radius by which a sphere or
    /// capsule support point hangs below its centre along world `−ẑ`, zero for
    /// material-point contacts (box corners, cylinder rims, mesh vertices).
    /// Detection reports it per contact because a multi-shape body no longer
    /// determines the producing shape by itself. It is unread for a body-body
    /// contact, whose two surface points are recovered from the manifold
    /// geometry instead.
    ///
    /// Detection reports the *midsurface* point in both cases. For the ground
    /// the support point itself is at `z = ground_height − depth`; for a pair
    /// the two surface points straddle the midsurface by `depth/2` along the
    /// normal, `body_i`'s on the far side since `+normal` is the direction `i`
    /// must move to separate.
    fn of(c: &Collision, drop: f64, state: &State, ground_height: f64) -> Self {
        // `xform.rot` is world→body, so body coordinates of a world point are
        // `R (p − pos)`, and `Rᵀ` carries a body direction back to world.
        if c.body_j == usize::MAX {
            let world_offset = Vec3::new(0.0, 0.0, -drop);
            let xform = &state.body_xform[c.body_i];
            let support = Vec3::new(
                c.contact_point.x,
                c.contact_point.y,
                ground_height - c.penetration_depth,
            );
            let material_point = xform.rot * (support - xform.pos - world_offset);
            return Self::Ground {
                body: c.body_i,
                material_point,
                world_offset,
            };
        }

        let n = c.contact_normal;
        let half = 0.5 * c.penetration_depth;
        // `i` must move along `+n` to separate, so its surface point is the one
        // on the `−n` side of the midsurface.
        let surface_i = c.contact_point - n * half;
        let surface_j = c.contact_point + n * half;
        let xi = &state.body_xform[c.body_i];
        let xj = &state.body_xform[c.body_j];
        let normal_frame = Self::reference_body(c.body_i, c.body_j, n, state);
        let owner = &state.body_xform[normal_frame];
        Self::Pair {
            body_i: c.body_i,
            body_j: c.body_j,
            point_i: xi.rot * (surface_i - xi.pos),
            point_j: xj.rot * (surface_j - xj.pos),
            normal_frame,
            normal_local: owner.rot * n,
        }
    }

    /// Which of the two bodies owns the face the contact normal came from.
    ///
    /// The narrow phase does not report this: `Manifold::normal` arrives from
    /// GJK/EPA as a world direction, a function of *both* poses, with no record
    /// of which shape's feature generated it. But for the face contacts that
    /// dominate a resting or riding manifold, the normal *is* a face normal of
    /// exactly one of the two boxes — which means it is exactly a coordinate
    /// axis of that body's frame, and merely some oblique direction in the
    /// other's. So the owner is recoverable after the fact: express the normal
    /// in each body frame and take the one that lands closest to a principal
    /// axis.
    ///
    /// This matters, and picking the wrong body is not a small error. Freezing
    /// the normal to the rider instead of to the plank it stands on makes
    /// `dJ/d(plank roll)` come out as zero when it is genuinely nonzero — the
    /// tilt no longer steers the friction — which is a *silently* missing
    /// gradient channel, the failure mode this module exists to prevent. It was
    /// caught by `body_body_adjoint.rs`'s slide case (adjoint `5e-12` against
    /// an FD of `-9.7e-3`) and is why that test sweeps every DOF of both bodies
    /// rather than only the ones the scene obviously moves.
    ///
    /// For curved pairs (sphere/capsule) the normal follows from the centres
    /// and belongs to neither face; the alignment score is then near-arbitrary
    /// and so is the choice. That is an approximation of the same order as
    /// freezing the feature pair at all (§4.4), not an additional one: either
    /// body's frame transports a centre-determined normal about equally well.
    fn reference_body(body_i: usize, body_j: usize, n_world: Vec3, state: &State) -> usize {
        // How nearly a direction is a coordinate axis of a frame: 1 exactly on
        // an axis, 1/sqrt(3) at the worst-case body diagonal.
        let axis_alignment = |body: usize| -> f64 {
            let local = state.body_xform[body].rot * n_world;
            local.x.abs().max(local.y.abs()).max(local.z.abs())
        };
        if axis_alignment(body_j) > axis_alignment(body_i) {
            body_j
        } else {
            body_i
        }
    }

    /// Re-evaluate the collision this anchor stands for at (the FK of) a
    /// perturbed state — same identity, smoothly moved geometry.
    fn collision(&self, state: &State, ground_height: f64) -> Collision {
        match *self {
            Self::Ground {
                body,
                material_point,
                world_offset,
            } => {
                let xform = &state.body_xform[body];
                let support = xform.pos + xform.rot.transpose() * material_point + world_offset;
                let depth = ground_height - support.z;
                Collision {
                    body_i: body,
                    body_j: usize::MAX,
                    contact_point: Vec3::new(support.x, support.y, ground_height - depth * 0.5),
                    contact_normal: Vec3::z(),
                    penetration_depth: depth,
                }
            }
            Self::Pair {
                body_i,
                body_j,
                point_i,
                point_j,
                normal_frame,
                normal_local,
            } => {
                let xi = &state.body_xform[body_i];
                let xj = &state.body_xform[body_j];
                let pi = xi.pos + xi.rot.transpose() * point_i;
                let pj = xj.pos + xj.rot.transpose() * point_j;
                // The frozen body-frame normal stays unit under a rotation, so
                // no renormalization is needed and none is done: a `normalize`
                // here would divide by a quantity that is identically one, and
                // its derivative would be a spurious zero-magnitude channel.
                let n = state.body_xform[normal_frame].rot.transpose() * normal_local;
                // Positive = overlapping, consistent with the ground branch:
                // the surfaces have swapped sides along `n` by this much.
                let depth = (pj - pi).dot(n);
                // The contact point rides the body that owns the *vertex*, not
                // the midpoint of the two anchors.
                //
                // Averaging looks natural and is wrong, in a way that is only
                // visible in a body-body pair. A face-vertex contact is located
                // by the vertex; the face body contributes the plane the vertex
                // is measured against, and sliding the face body *within its own
                // plane* must not move the contact at all. `(pi + pj)/2` instead
                // moves it by half of any such slide, which shows up as a
                // spurious moment arm: the measured symptom was
                // `dJ/d(plank x) = +9.7e-3` on a block sliding along a plank,
                // where translation invariance puts the true value at `-8.2e-5`.
                // The ground branch never had the bug because it already pins
                // the point to `body_i` and takes only `z` from the plane —
                // this is that same rule with "the plane" generalized to
                // whichever body owns the normal.
                let vertex = if normal_frame == body_i { pj } else { pi };
                let sign = if normal_frame == body_i { -1.0 } else { 1.0 };
                Collision {
                    body_i,
                    body_j,
                    contact_point: vertex + n * (sign * 0.5 * depth),
                    contact_normal: n,
                    penetration_depth: depth,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Everything the backward pass needs about one step, cached rather than
/// recomputed: re-solving during the backward pass would re-run the
/// warm-started solver on a different warm-start history, which is
/// path-dependent — the cache is what guarantees the adjoint differentiates
/// the solves the forward pass actually performed.
struct StepRecord {
    q: DVec,
    v: DVec,
    u: DVec,
    v_next: DVec,
    anchors: Vec<Anchor>,
    contact: Option<(ContactAssembly, ContactSolution)>,
}

/// The assembled problem and its solution for a contacted step (`None` when
/// the step had no contacts).
type StepSolve = Option<(ContactAssembly, ContactSolution)>;

/// One step of the forward pass, mirroring `Simulator::step_with_contacts`
/// operation for operation (FK → detect → free velocity → assemble →
/// warm-started solve → integrate → FK).
fn forward_step(
    model: &Model,
    state: &mut State,
    ground_height: f64,
    material: &ContactMaterial,
    config: &ContactSolverConfig,
    cache: &mut ContactCache,
) -> (Vec<(Collision, f64)>, StepSolve) {
    let dt = model.dt;

    let (xforms, _velocities) = forward_kinematics(model, state);
    state.body_xform = xforms;

    // Same detection as `Simulator::step_with_contacts` — the full collision
    // set — plus each ground contact's world-axis drop, which the anchor
    // recovery below needs. Body-body contacts carry a drop of zero; they are
    // rejected as unsupported before it is ever read.
    let mut contacts: Vec<(Collision, f64)> =
        find_ground_contacts_model_with_drop(model, state, ground_height, material.margin);
    let body_contacts = find_contacts(model, state, material.margin);
    contacts.extend(body_contacts.into_iter().map(|c| (c, 0.0)));

    let qdd = aba(model, state);
    let free_qd = &state.v + &(&qdd * dt);

    let bare: Vec<Collision> = contacts.iter().map(|(c, _)| c.clone()).collect();
    let record = if contacts.is_empty() {
        state.v = free_qd;
        None
    } else {
        let materials = vec![material.clone(); model.bodies.len().max(1)];
        let asm = assemble(model, state, &bare, &materials, &free_qd, dt, config);
        let seed = cache.warm_start(state, &bare);
        let solution = solve_contacts_warm(&asm.problem, config, &seed);
        cache.store(state, &bare, &solution.impulses);
        state.v = &free_qd + &asm.velocity_delta(&solution.impulses);
        Some((asm, solution))
    };

    let v_clone = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v_clone.as_slice(), dt);
    state.time += dt;

    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;

    (contacts, record)
}

/// Run the nominal rollout, recording every step.
fn forward_rollout(
    rollout: &ConvexContactRollout,
) -> Result<(Vec<StepRecord>, State), ConvexAdjointError> {
    let model = rollout.model;
    let mut state = model.default_state();
    state.q = rollout.q0.clone();
    state.v = rollout.v0.clone();
    let mut cache = ContactCache::default();
    let mut records = Vec::with_capacity(rollout.steps);

    for t in 0..rollout.steps {
        let q = state.q.clone();
        let v = state.v.clone();
        let u = (rollout.ctrl)(t);
        state.ctrl = u.clone();

        // Anchors are read off the *pre-step* FK, the same transforms the
        // detection inside `forward_step` will use.
        let (xforms, _) = forward_kinematics(model, &state);
        state.body_xform = xforms;

        let (contacts, contact) = forward_step(
            model,
            &mut state,
            rollout.ground_height,
            &rollout.material,
            &rollout.config,
            &mut cache,
        );

        let mut anchors = Vec::with_capacity(contacts.len());
        // Recover anchors against the pre-step transforms; `forward_step`
        // already advanced `state`, so rebuild them from (q, v).
        let mut pre = model.default_state();
        pre.q = q.clone();
        pre.v = v.clone();
        let (pre_xf, _) = forward_kinematics(model, &pre);
        pre.body_xform = pre_xf;
        for (c, drop) in &contacts {
            anchors.push(Anchor::of(c, *drop, &pre, rollout.ground_height));
        }

        records.push(StepRecord {
            q,
            v,
            u,
            v_next: state.v.clone(),
            anchors,
            contact,
        });
    }
    Ok((records, state))
}

/// Run the nominal rollout and return the objective only — the FD oracle for
/// gates, and a cheap primal probe. The dynamics are identical to
/// [`convex_adjoint_gradient`]'s forward pass (and to
/// `Simulator::step_with_contacts`).
pub fn convex_rollout_objective(
    rollout: &ConvexContactRollout,
    objective: &FinalStateObjective,
) -> f64 {
    let model = rollout.model;
    let mut state = model.default_state();
    state.q = rollout.q0.clone();
    state.v = rollout.v0.clone();
    let mut cache = ContactCache::default();
    for t in 0..rollout.steps {
        state.ctrl = (rollout.ctrl)(t);
        forward_step(
            model,
            &mut state,
            rollout.ground_height,
            &rollout.material,
            &rollout.config,
            &mut cache,
        );
    }
    (objective.value)(state.q.as_slice(), state.v.as_slice())
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

/// The smooth pieces of one step at a (possibly perturbed) input, with the
/// contact set frozen to the nominal anchors and the impulses frozen to `f*`.
///
/// `residual` is the KKT residual `(A + R)f* + b − e_n·bias` — the object
/// whose differential the IFT contracts. `gf` is `M⁻¹ Jᵀ f*`, the explicit
/// (impulse-held-fixed) contact term of `v'`.
struct Pieces {
    v_free: DVec,
    residual: Vec<f64>,
    /// Per contact, `M_t(θ)·t_rel` for sliding contacts (`A`'s own tangential
    /// 2x2 block applied to `t* − f_t`; the regularizer floor is constant and
    /// drops out of the difference). Zero for non-sliding contacts.
    mt_rel: Vec<[f64; 2]>,
    gf: DVec,
}

// Ten arguments: this is the frozen step function itself — every input is a
// genuine independent variable of the map being differentiated.
#[allow(clippy::too_many_arguments)]
// Stride arithmetic into the flat, row-major Delassus (base = 3*c).
#[allow(clippy::needless_range_loop)]
fn eval_pieces(
    model: &Model,
    ground_height: f64,
    material: &ContactMaterial,
    config: &ContactSolverConfig,
    anchors: &[Anchor],
    f_star: &[Vec3],
    t_rel: &[[f64; 2]],
    q: &DVec,
    v: &DVec,
    u: &DVec,
) -> Pieces {
    let dt = model.dt;
    let mut state = model.default_state();
    state.q = q.clone();
    state.v = v.clone();
    state.ctrl = u.clone();
    let (xforms, _) = forward_kinematics(model, &state);
    state.body_xform = xforms;

    let qdd = aba(model, &state);
    let v_free = &state.v + &(&qdd * dt);

    if anchors.is_empty() {
        return Pieces {
            v_free,
            residual: Vec::new(),
            mt_rel: Vec::new(),
            gf: DVec::zeros(model.nv),
        };
    }

    let contacts: Vec<Collision> = anchors
        .iter()
        .map(|a| a.collision(&state, ground_height))
        .collect();
    let materials = vec![material.clone(); model.bodies.len().max(1)];
    let asm = assemble(model, &state, &contacts, &materials, &v_free, dt, config);
    let p = &asm.problem;
    let n = p.n;
    let dim = 3 * n;
    let mut flat = vec![0.0; dim];
    for (c, f) in f_star.iter().enumerate() {
        flat[3 * c] = f.x;
        flat[3 * c + 1] = f.y;
        flat[3 * c + 2] = f.z;
    }
    let mut residual = vec![0.0; dim];
    for c in 0..n {
        let reg = regularization_diag(p, c, config);
        let base = 3 * c;
        for r in 0..3 {
            let mut acc = p.free_velocity[base + r];
            for (col, fc) in flat.iter().enumerate() {
                acc += p.delassus[(base + r) * dim + col] * fc;
            }
            acc += reg[r] * flat[base + r];
            if r == 0 {
                acc -= p.rows[c].bias;
            }
            residual[base + r] = acc;
        }
    }
    let mut mt_rel = vec![[0.0f64; 2]; n];
    for (c, tr) in t_rel.iter().enumerate() {
        let base = 3 * c;
        for i in 0..2 {
            mt_rel[c][i] = p.delassus[(base + 1 + i) * dim + base + 1] * tr[0]
                + p.delassus[(base + 1 + i) * dim + base + 2] * tr[1];
        }
    }
    let gf = asm.velocity_delta(f_star);

    Pieces {
        v_free,
        residual,
        mt_rel,
        gf,
    }
}

/// `q' = Φ(q, v')` — the shared configuration update.
fn phi(model: &Model, q: &DVec, v_next: &DVec) -> DVec {
    let mut out = q.clone();
    integrate_configuration(model, out.as_mut_slice(), v_next.as_slice(), model.dt);
    out
}

/// A model with body `b`'s inertia rebuilt from a perturbed 10-vector.
fn perturbed_model(model: &Model, body: usize, params: &[f64; N_INERTIA_PARAMS]) -> Model {
    use phyz_math::{Mat3, SpatialInertia};
    let mut m = model.clone();
    let p = params;
    m.bodies[body].inertia = SpatialInertia::new(
        p[0],
        Vec3::new(p[1], p[2], p[3]),
        Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
    );
    m
}

/// One nominal rollout forward, one adjoint pass backward: `J`, `dJ/dq0`,
/// `dJ/dv0`, `dJ/du_t`, and `dJ/dπ` per body — all through the convex contact
/// model the forward simulator integrates.
///
/// Errors instead of returning a wrong gradient; see [`ConvexAdjointError`].
pub fn convex_adjoint_gradient(
    rollout: &ConvexContactRollout,
    objective: &FinalStateObjective,
) -> Result<ConvexAdjointGradients, ConvexAdjointError> {
    let model = rollout.model;
    let nq = model.nq;
    let nv = model.nv;
    let nb = model.bodies.len();
    let (records, final_state) = forward_rollout(rollout)?;

    let j0 = (objective.value)(final_state.q.as_slice(), final_state.v.as_slice());
    let (gq, gv) = (objective.gradient)(final_state.q.as_slice(), final_state.v.as_slice());
    assert_eq!(gq.len(), nq, "objective ∂g/∂q length");
    assert_eq!(gv.len(), nv, "objective ∂g/∂v length");
    let mut lam_q = DVec::from_slice(&gq);
    let mut lam_v = DVec::from_slice(&gv);

    let mut d_ctrl = vec![DVec::zeros(nv); rollout.steps];
    let mut d_inertia = vec![[0.0f64; N_INERTIA_PARAMS]; nb];
    let nominal_params: Vec<[f64; N_INERTIA_PARAMS]> = model
        .bodies
        .iter()
        .map(|b| inertia_params(&b.inertia))
        .collect();

    for (t, rec) in records.iter().enumerate().rev() {
        // IFT sensitivity of this step's *recorded* solve. Cached, never
        // re-solved: warm starting makes a re-solve path-dependent.
        let fps: Option<FixedPointSensitivity> = match &rec.contact {
            None => None,
            Some((asm, sol)) => {
                if !sol.converged {
                    return Err(ConvexAdjointError::Unconverged {
                        step: t,
                        iterations: sol.iterations,
                        residual: sol.residual,
                    });
                }
                Some(
                    FixedPointSensitivity::at(&asm.problem, sol, &rollout.config)
                        .ok_or(ConvexAdjointError::SingularKkt { step: t })?,
                )
            }
        };
        let t_rel: Vec<[f64; 2]> = match (&rec.contact, &fps) {
            (Some((asm, _)), Some(s)) => (0..asm.problem.n)
                .map(|c| s.slide_tangent(c).map(|st| st.t_rel).unwrap_or([0.0; 2]))
                .collect(),
            _ => Vec::new(),
        };
        let f_star: Vec<Vec3> = rec
            .contact
            .as_ref()
            .map(|(_, s)| s.impulses.clone())
            .unwrap_or_default();

        // Directional derivative of v' from a directional derivative of the
        // smooth pieces, closing the contact channel with the IFT.
        let dv_next = |dp: &Pieces| -> DVec {
            match (&rec.contact, &fps) {
                (Some((asm, _)), Some(s)) => {
                    // df* from the exact map linearization; dv' = dv_free +
                    // d(M⁻¹Jᵀ f*)|_f + M⁻¹Jᵀ df*.
                    let df = s.apply(&dp.residual, &dp.mt_rel);
                    &(&dp.v_free + &dp.gf) + &asm.velocity_delta(&df)
                }
                _ => dp.v_free.clone(),
            }
        };

        let eval = |m: &Model, q: &DVec, v: &DVec, u: &DVec| -> Pieces {
            eval_pieces(
                m,
                rollout.ground_height,
                &rollout.material,
                &rollout.config,
                &rec.anchors,
                &f_star,
                &t_rel,
                q,
                v,
                u,
            )
        };

        // Central difference of the pieces along one input lane.
        let lane = |mp: &Model,
                    mm: &Model,
                    qp: &DVec,
                    qm: &DVec,
                    vp: &DVec,
                    vm: &DVec,
                    up: &DVec,
                    um: &DVec,
                    h: f64|
         -> Pieces {
            let a = eval(mp, qp, vp, up);
            let b = eval(mm, qm, vm, um);
            let inv = 1.0 / (2.0 * h);
            Pieces {
                v_free: &(&a.v_free - &b.v_free) * inv,
                residual: a
                    .residual
                    .iter()
                    .zip(&b.residual)
                    .map(|(x, y)| (x - y) * inv)
                    .collect(),
                mt_rel: a
                    .mt_rel
                    .iter()
                    .zip(&b.mt_rel)
                    .map(|(x, y)| [(x[0] - y[0]) * inv, (x[1] - y[1]) * inv])
                    .collect(),
                gf: &(&a.gf - &b.gf) * inv,
            }
        };

        // dq' along a v'-direction, by central difference through Φ.
        let dphi_dvnext = |dvn: &DVec| -> DVec {
            let norm = dvn.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if norm == 0.0 {
                return DVec::zeros(nq);
            }
            let s = FD_EPS / norm;
            let vp = &rec.v_next + &(dvn * s);
            let vm = &rec.v_next - &(dvn * s);
            &(&phi(model, &rec.q, &vp) - &phi(model, &rec.q, &vm)) * (1.0 / (2.0 * s))
        };

        // Contract a lane's (dq', dv') against the incoming adjoint.
        let contract = |dqn: &DVec, dvn: &DVec| -> f64 {
            let mut acc = 0.0;
            for i in 0..nq {
                acc += lam_q[i] * dqn[i];
            }
            for j in 0..nv {
                acc += lam_v[j] * dvn[j];
            }
            acc
        };

        let mut new_lam_q = DVec::zeros(nq);
        let mut new_lam_v = DVec::zeros(nv);

        // --- q lanes ---
        for i in 0..nq {
            let h = FD_EPS * rec.q[i].abs().max(1.0);
            let mut qp = rec.q.clone();
            let mut qm = rec.q.clone();
            qp[i] += h;
            qm[i] -= h;
            let dp = lane(model, model, &qp, &qm, &rec.v, &rec.v, &rec.u, &rec.u, h);
            let dvn = dv_next(&dp);
            // Direct Φ_q block plus the v'-mediated part.
            let dqn_direct = &(&phi(model, &qp, &rec.v_next) - &phi(model, &qm, &rec.v_next))
                * (1.0 / (2.0 * h));
            let dqn = &dqn_direct + &dphi_dvnext(&dvn);
            new_lam_q[i] = contract(&dqn, &dvn);
        }

        // --- v lanes ---
        for j in 0..nv {
            let h = FD_EPS * rec.v[j].abs().max(1.0);
            let mut vp = rec.v.clone();
            let mut vm = rec.v.clone();
            vp[j] += h;
            vm[j] -= h;
            let dp = lane(model, model, &rec.q, &rec.q, &vp, &vm, &rec.u, &rec.u, h);
            let dvn = dv_next(&dp);
            let dqn = dphi_dvnext(&dvn);
            new_lam_v[j] = contract(&dqn, &dvn);
        }

        // --- control lanes ---
        for j in 0..nv {
            let h = FD_EPS * rec.u[j].abs().max(1.0);
            let mut up = rec.u.clone();
            let mut um = rec.u.clone();
            up[j] += h;
            um[j] -= h;
            let dp = lane(model, model, &rec.q, &rec.q, &rec.v, &rec.v, &up, &um, h);
            let dvn = dv_next(&dp);
            let dqn = dphi_dvnext(&dvn);
            d_ctrl[t][j] = contract(&dqn, &dvn);
        }

        // --- inertia-parameter lanes ---
        for b in 0..nb {
            for k in 0..N_INERTIA_PARAMS {
                let h = FD_EPS * nominal_params[b][k].abs().max(1.0);
                let mut pp = nominal_params[b];
                let mut pm = nominal_params[b];
                pp[k] += h;
                pm[k] -= h;
                let mp = perturbed_model(model, b, &pp);
                let mm = perturbed_model(model, b, &pm);
                let dp = lane(&mp, &mm, &rec.q, &rec.q, &rec.v, &rec.v, &rec.u, &rec.u, h);
                let dvn = dv_next(&dp);
                let dqn = dphi_dvnext(&dvn);
                d_inertia[b][k] += contract(&dqn, &dvn);
            }
        }

        lam_q = new_lam_q;
        lam_v = new_lam_v;
    }

    Ok(ConvexAdjointGradients {
        objective: j0,
        d_q0: lam_q,
        d_v0: lam_v,
        d_ctrl,
        d_inertia,
    })
}
