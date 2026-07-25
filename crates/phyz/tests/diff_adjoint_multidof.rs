//! Gates for the multi-DOF (spherical / free) extension of the trajectory
//! adjoint (`phyz::diff`).
//!
//! Same discipline as `diff_adjoint.rs`: every gradient claim is checked
//! against a closed form or a central finite difference of the *same*
//! deterministic primal rollout, with the FD step chosen per scalar class so
//! truncation `O(h²)` and roundoff `O(ε·|J|/h)` both sit well below the gate.
//!
//! Positions here use the diff rollout's own coordinate layout
//! ([`phyz::diff::DofLayout`]): quaternions for the rotational sub-blocks of
//! spherical and free joints, so `nq > nv`. Build `q0` from
//! `DofLayout::neutral_q` and index it through `DofLayout::q_offsets`.

use phyz::diff::{
    AdjointRollout, CollisionMesh, ContactSetup, DofLayout, FinalStateObjective, GroundContact,
    adjoint_rollout_gradient, rollout_objective,
};
use phyz::math::{DVec, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz::model::{Joint, JointType, Model, ModelBuilder};

/// Build a spatial inertia from the canonical 10-vector packing
/// `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` (symmetric off-diagonals).
fn si_from(p: [f64; 10]) -> SpatialInertia {
    SpatialInertia::new(
        p[0],
        Vec3::new(p[1], p[2], p[3]),
        Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
    )
}

/// The objective `J = q_T[i]`. The index is captured, so the closures are
/// leaked to `'static` — one pair of tiny allocations per gate, which keeps
/// the objective a one-liner at every call site.
fn q_objective(i: usize) -> FinalStateObjective<'static> {
    FinalStateObjective {
        value: Box::leak(Box::new(move |q: &[f64], _v: &[f64]| q[i])),
        gradient: Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[i] = 1.0;
            (gq, vec![0.0; v.len()])
        })),
    }
}

/// The objective `J = v_T[i]`.
fn v_objective(i: usize) -> FinalStateObjective<'static> {
    FinalStateObjective {
        value: Box::leak(Box::new(move |_q: &[f64], v: &[f64]| v[i])),
        gradient: Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
            let mut gv = vec![0.0; v.len()];
            gv[i] = 1.0;
            (vec![0.0; q.len()], gv)
        })),
    }
}

/// Central-difference sweep of `dJ/dπ` for every body/scalar of a model built
/// by `build`, gated against the adjoint's `d_inertia`. Returns the number of
/// live channels so a caller can assert the gate is not passing on zeros.
#[allow(clippy::too_many_arguments)]
fn gate_inertia_vs_fd(
    build: &dyn Fn(&[[f64; 10]]) -> Model,
    nominal: &[[f64; 10]],
    q0: &dyn Fn(&Model) -> Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    ctrl: &dyn Fn(usize) -> DVec,
    objective: &FinalStateObjective,
    h_for: &dyn Fn(usize) -> f64,
    gate: f64,
    dead: f64,
) -> usize {
    let j_at = |ps: &[[f64; 10]]| -> f64 {
        let m = build(ps);
        let ro = AdjointRollout {
            q0: q0(&m),
            v0: v0.clone(),
            steps,
            ctrl,
            model: &m,
            contact: None,
        };
        rollout_objective(&ro, objective)
    };

    let model = build(nominal);
    let rollout = AdjointRollout {
        q0: q0(&model),
        v0: v0.clone(),
        steps,
        ctrl,
        model: &model,
        contact: None,
    };
    let g = adjoint_rollout_gradient(&rollout, objective);

    let mut live = 0;
    for b in 0..nominal.len() {
        for k in 0..10 {
            let h = h_for(k);
            let perturb = |sign: f64| -> f64 {
                let mut ps = nominal.to_vec();
                ps[b][k] += sign * h;
                j_at(&ps)
            };
            let fd = (perturb(1.0) - perturb(-1.0)) / (2.0 * h);
            let adj = g.d_inertia[b][k];
            if fd.abs() > dead {
                live += 1;
                let rel = (adj - fd).abs() / fd.abs();
                assert!(
                    rel <= gate,
                    "dJ/dπ[body {b}][{k}]: adjoint {adj} vs fd {fd} (rel {rel:.3e})"
                );
            } else {
                assert!(
                    adj.abs() <= dead,
                    "dJ/dπ[body {b}][{k}]: fd dead ({fd}) but adjoint {adj}"
                );
            }
        }
    }
    live
}

// ---------------------------------------------------------------------------
// Gate 1 — the layout itself.
// ---------------------------------------------------------------------------

/// A free joint contributes 7 position and 6 velocity coordinates, a spherical
/// joint 4 and 3: `nq > nv`, which is precisely what the old `nq == nv`
/// assertion rejected.
#[test]
fn layout_splits_q_and_v_for_multi_dof_joints() {
    let model = ModelBuilder::new()
        .add_free_body("base", -1, SpatialTransform::identity(), si_from(BASE_PI))
        .add_spherical_body("ball", 0, SpatialTransform::identity(), si_from(LINK_PI))
        .add_revolute_body("link", 1, SpatialTransform::identity(), si_from(LINK_PI))
        .build();

    let layout = DofLayout::of(&model);
    assert_eq!((layout.nq, layout.nv), (7 + 4 + 1, 6 + 3 + 1));
    assert_eq!(layout.q_offsets, vec![0, 7, 11]);
    assert_eq!(layout.v_offsets, vec![0, 6, 9]);
    // The model's own packing is the exponential-coordinate one — it must not
    // be used to index a diff-rollout q.
    assert_eq!(model.nq, model.nv);
    assert_ne!(layout.nq, model.nq);

    // neutral_q puts identity quaternions in both rotational sub-blocks.
    let q = layout.neutral_q(&model);
    assert_eq!(q.len(), layout.nq);
    assert_eq!(q[3], 1.0, "free joint quaternion w");
    assert_eq!(q[7], 1.0, "spherical joint quaternion w");
    assert_eq!(q.iter().filter(|&&x| x != 0.0).count(), 2);
}

const BASE_PI: [f64; 10] = [10.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
const LINK_PI: [f64; 10] = [1.0, 0.1, 0.0, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0];

// ---------------------------------------------------------------------------
// Gate 2 — free-joint closed form (exact, 1e-12).
// ---------------------------------------------------------------------------

/// A single free body pushed along +Z by a constant body-frame force, COM on
/// the origin and inertia diagonal. Nothing rotates, so the whole rollout has
/// `a_z = F/m − g` exactly and semi-implicit Euler gives
/// `v_T[5] = steps·dt·(F/m − g)`. Hence `dJ/dm = −steps·dt·F/m²` in closed
/// form, and every other inertia scalar is structurally dead.
///
/// This is the gate on the `ndof×ndof` solve itself: `D` is the full 6×6
/// spatial inertia here, not a scalar, so a wrong factorisation or a wrong
/// motion-subspace ordering cannot pass.
#[test]
fn free_body_thrust_closed_form() {
    const MASS: f64 = 2.5;
    const FORCE: f64 = 40.0;
    const G: f64 = 9.81;
    const DT: f64 = 1.0 / 480.0;
    const STEPS: usize = 96;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(DT)
        .add_free_body(
            "pod",
            -1,
            SpatialTransform::identity(),
            si_from([MASS, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0]),
        )
        .build();

    let layout = DofLayout::of(&model);
    // Free-joint v is [ω; v_lin]: index 5 is linear z.
    let ctrl = |_t: usize| DVec::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, FORCE]);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: layout.neutral_q(&model),
        v0: vec![0.0; layout.nv],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let obj = v_objective(5);
    let g = adjoint_rollout_gradient(&rollout, &obj);

    let t_total = STEPS as f64 * DT;
    let j_expected = t_total * (FORCE / MASS - G);
    assert!(
        (g.objective - j_expected).abs() / j_expected.abs() <= 1e-12,
        "J = {} vs closed form {j_expected}",
        g.objective
    );

    let d_mass_expected = -t_total * FORCE / (MASS * MASS);
    let rel = (g.d_inertia[0][0] - d_mass_expected).abs() / d_mass_expected.abs();
    assert!(
        rel <= 1e-12,
        "dJ/dm = {} vs closed form {d_mass_expected} (rel {rel:.3e})",
        g.d_inertia[0][0]
    );
    for (k, &gk) in g.d_inertia[0].iter().enumerate().skip(1) {
        assert!(
            gk.abs() <= 1e-10 * d_mass_expected.abs(),
            "π[{k}] should not influence a non-rotating pod, got {gk}"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 3 — free-joint qdd agrees with the concrete ABA.
// ---------------------------------------------------------------------------

/// The generic multi-DOF solve is an independent implementation of
/// `phyz_rigid::aba`'s matrix path. At the identity quaternion the two
/// coordinate layouts coincide (`exp(0) = 1`), so both must return the same
/// `qdd` for the same `(q, v, ctrl)` — a cross-check that the generic solve is
/// *right*, not merely self-consistent with its own finite differences.
#[test]
fn free_base_qdd_matches_concrete_aba() {
    const DT: f64 = 1e-3;
    let joint = Joint {
        joint_type: JointType::Revolute,
        parent_to_joint: SpatialTransform::from_translation(Vec3::new(0.25, 0.0, 0.0)),
        axis: Vec3::new(0.0, 1.0, 0.0),
        damping: 0.03,
        ..Default::default()
    };
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body("base", -1, SpatialTransform::identity(), si_from(BASE_PI))
        .add_body("leg", 0, joint, si_from(LINK_PI))
        .build();

    let layout = DofLayout::of(&model);
    let v: Vec<f64> = vec![0.11, -0.07, 0.19, 0.3, -0.2, 0.05, 0.4];
    let u: Vec<f64> = vec![0.5, -0.3, 0.2, 1.0, -2.0, 3.0, 0.07];

    // Concrete rollout: exponential coordinates, all zero (identity rotation).
    let mut state = model.default_state();
    state.q[6] = 0.37; // the revolute angle
    for (i, &vi) in v.iter().enumerate() {
        state.v[i] = vi;
    }
    for (i, &ui) in u.iter().enumerate() {
        state.ctrl[i] = ui;
    }
    let qdd_concrete = phyz::rigid::aba(&model, &state);

    // Diff rollout: quaternion coordinates at identity, same revolute angle.
    let mut q0 = layout.neutral_q(&model);
    q0[layout.q_offsets[1]] = 0.37;
    let ctrl = |_t: usize| DVec::from_slice(&u);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0,
        v0: v.clone(),
        steps: 1,
        ctrl: &ctrl,
    };
    // One step of the diff rollout: v' = v + dt·qdd, so qdd = (v' − v)/dt.
    // Read v' out through an objective per component.
    for j in 0..layout.nv {
        let obj = v_objective(j);
        let v_next = rollout_objective(&rollout, &obj);
        let qdd = (v_next - v[j]) / DT;
        assert!(
            (qdd - qdd_concrete[j]).abs() <= 1e-9 * (1.0 + qdd_concrete[j].abs()),
            "qdd[{j}]: generic {qdd} vs phyz_rigid::aba {}",
            qdd_concrete[j]
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 4 — free base + revolute leg, all 20 π scalars vs central FD (1e-6).
// ---------------------------------------------------------------------------

/// A free-floating base with an offset, lopsided leg, started tilted and
/// spinning so the quaternion sub-block, the `Eᵀ·v_lin` translation update and
/// the articulated coupling across the free joint are all live. This is the
/// case the old `assert_single_dof` panicked on.
///
/// FD steps: mass 1e-5, COM 1e-6, inertia 1e-6 — with `J = q_T[0]` (base x,
/// O(1e-2) m) and gradients O(1e-3), truncation lands ~1e-9 and roundoff
/// ~1e-8 relative, two decades under the gate.
#[test]
fn free_base_chain_inertia_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 200;

    let base: [f64; 10] = [
        4.0, 0.02, -0.01, 0.03, 0.09, 0.11, 0.13, 0.006, 0.004, 0.002,
    ];
    let leg: [f64; 10] = [
        0.8, 0.18, 0.05, -0.04, 0.012, 0.015, 0.018, 0.003, 0.001, 0.002,
    ];

    let build = |ps: &[[f64; 10]]| -> Model {
        let joint = Joint {
            joint_type: JointType::Revolute,
            parent_to_joint: SpatialTransform::from_translation(Vec3::new(0.3, 0.05, 0.0)),
            axis: Vec3::new(0.0, 1.0, 0.0),
            damping: 0.02,
            ..Default::default()
        };
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .dt(DT)
            .add_free_body("base", -1, SpatialTransform::identity(), si_from(ps[0]))
            .add_body("leg", 0, joint, si_from(ps[1]))
            .build()
    };

    // Tilted base: a normalised quaternion well away from identity.
    let q0 = |m: &Model| -> Vec<f64> {
        let layout = DofLayout::of(m);
        let mut q = layout.neutral_q(m);
        let n = (0.9f64 * 0.9 + 0.2 * 0.2 + 0.3 * 0.3 + 0.1 * 0.1).sqrt();
        q[0] = 0.1;
        q[1] = -0.2;
        q[2] = 0.4;
        q[3] = 0.9 / n;
        q[4] = 0.2 / n;
        q[5] = 0.3 / n;
        q[6] = 0.1 / n;
        q[7] = -0.35; // revolute angle
        q
    };
    let ctrl = |_t: usize| DVec::from_slice(&[0.3, -0.2, 0.15, 1.0, -0.5, 2.0, 0.04]);
    let v0 = vec![0.25, -0.4, 0.3, 0.2, 0.1, -0.15, 0.35];
    let obj = q_objective(0); // base x

    let live = gate_inertia_vs_fd(
        &build,
        &[base, leg],
        &q0,
        v0,
        STEPS,
        &ctrl,
        &obj,
        &|k| match k {
            0 => 1e-5,
            1..=3 => 1e-6,
            _ => 1e-6,
        },
        1e-6,
        1e-8,
    );
    assert!(
        live >= 16,
        "only {live}/20 live channels — free base not exercising the packing"
    );
}

// ---------------------------------------------------------------------------
// Gate 5 — spherical joint, all 10 π scalars vs central FD (1e-6).
// ---------------------------------------------------------------------------

/// A 3-DOF spherical pendulum: `ndof = 3`, a purely angular motion subspace,
/// and a quaternion configuration whose only rate is `exp(−dt·ω) ⊗ p`. Started
/// off-axis with a spin so the gyroscopic bias `v ×* I v` is live too — the
/// term a spherical joint has and a revolute chain never does in isolation.
#[test]
fn spherical_pendulum_inertia_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 200;

    let bob: [f64; 10] = [
        1.2, 0.3, 0.08, -0.05, 0.02, 0.035, 0.05, 0.004, 0.003, 0.002,
    ];

    let build = |ps: &[[f64; 10]]| -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .dt(DT)
            .add_spherical_body("bob", -1, SpatialTransform::identity(), si_from(ps[0]))
            .build()
    };

    let q0 = |m: &Model| -> Vec<f64> {
        let layout = DofLayout::of(m);
        let mut q = layout.neutral_q(m);
        let n = (0.95f64 * 0.95 + 0.2 * 0.2 + 0.15 * 0.15 + 0.1 * 0.1).sqrt();
        q[0] = 0.95 / n;
        q[1] = 0.2 / n;
        q[2] = 0.15 / n;
        q[3] = 0.1 / n;
        q
    };
    let ctrl = |_t: usize| DVec::from_slice(&[0.02, -0.01, 0.015]);
    // Non-aligned spin: makes the ω × Iω bias non-zero for a non-spherical I.
    let v0 = vec![0.6, -0.35, 0.25];
    // J = the quaternion's x component: a scalar function of the whole
    // Lie-group trajectory, so Φ_q ≠ I is exercised, not bypassed.
    let obj = q_objective(1);

    let live = gate_inertia_vs_fd(
        &build,
        &[bob],
        &q0,
        v0,
        STEPS,
        &ctrl,
        &obj,
        &|k| match k {
            0 => 1e-5,
            1..=3 => 1e-6,
            _ => 1e-7,
        },
        1e-6,
        1e-8,
    );
    assert!(
        live >= 7,
        "only {live}/10 live channels — spherical gate under-constrained"
    );
}

// ---------------------------------------------------------------------------
// Gate 6 — quaternions stay on the manifold.
// ---------------------------------------------------------------------------

/// The configuration update normalises, so a long rollout must not drift off
/// the unit sphere — otherwise `joint_transform`'s rotation would slowly stop
/// being a rotation and every gradient above would be measuring the wrong
/// forward model.
#[test]
fn free_joint_quaternion_stays_unit() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 5000;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "tumbler",
            -1,
            SpatialTransform::identity(),
            si_from(BASE_PI),
        )
        .build();
    let layout = DofLayout::of(&model);

    let ctrl = |_t: usize| DVec::from_slice(&[0.0; 6]);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: layout.neutral_q(&model),
        v0: vec![2.0, -1.5, 3.0, 0.5, 0.2, 1.0],
        steps: STEPS,
        ctrl: &ctrl,
    };

    let norm_sq = FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[3] * q[3] + q[4] * q[4] + q[5] * q[5] + q[6] * q[6],
        gradient: &|q: &[f64], v: &[f64]| (vec![0.0; q.len()], vec![0.0; v.len()]),
    };
    let n2 = rollout_objective(&rollout, &norm_sq);
    assert!(
        (n2 - 1.0).abs() < 1e-12,
        "quaternion norm² drifted to {n2} after {STEPS} steps"
    );
}

// ---------------------------------------------------------------------------
// Gate 7 — models/ant.xml: the regression case, sampled channels vs FD.
// ---------------------------------------------------------------------------

/// `models/ant.xml` has a free-floating torso and eight hinges — the model the
/// adjoint used to panic on outright. Gated on a sample of channels rather
/// than all 90: the FD oracle costs two full rollouts per channel, and the
/// sample already covers the free-base body, a mid-chain hip and a leaf ankle.
#[test]
fn ant_free_base_inertia_gradient_matches_fd() {
    const STEPS: usize = 200;
    const GATE: f64 = 1e-5;

    let load = || -> Model {
        phyz_mjcf::MjcfLoader::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/ant.xml"
        ))
        .expect("models/ant.xml should load")
        .build_model()
    };

    let model = load();
    let layout = DofLayout::of(&model);
    assert_eq!(layout.nv, 6 + 8, "ant: free torso + 8 hinges");
    assert_eq!(layout.nq, 7 + 8, "ant: quaternion torso + 8 hinges");
    assert_ne!(layout.nq, model.nq, "ant is exactly the nq ≠ nv case");

    let mut q0 = layout.neutral_q(&model);
    // Perturb the hinges off the singular all-zero pose.
    for j in 1..model.joints.len() {
        let qi = layout.q_offsets[j];
        q0[qi] = 0.15 * (j as f64 % 3.0 - 1.0) + 0.1;
    }
    let v0: Vec<f64> = (0..layout.nv)
        .map(|i| 0.05 * (i as f64 % 5.0 - 2.0))
        .collect();
    let ctrl = |_t: usize| DVec::zeros(14);
    let obj = q_objective(2); // torso z

    let params_of = |m: &Model| -> Vec<[f64; 10]> {
        m.bodies
            .iter()
            .map(|b| phyz::diff::inertia_params(&b.inertia))
            .collect()
    };
    let nominal = params_of(&model);

    let j_at = |ps: &[[f64; 10]]| -> f64 {
        let mut m = load();
        for (b, p) in ps.iter().enumerate() {
            m.bodies[b].inertia = si_from(*p);
        }
        let ro = AdjointRollout {
            model: &m,
            contact: None,
            q0: q0.clone(),
            v0: v0.clone(),
            steps: STEPS,
            ctrl: &ctrl,
        };
        rollout_objective(&ro, &obj)
    };

    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: q0.clone(),
        v0: v0.clone(),
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    // Torso mass/COM/Izz, a hip mass and COM, an ankle mass and Iyy.
    let samples: [(usize, usize); 7] = [(0, 0), (0, 1), (0, 6), (1, 0), (1, 1), (2, 0), (2, 5)];
    let mut live = 0;
    for (b, k) in samples {
        // Relative FD step, and a loose one at 1e-3·|π|. The ant's scalars
        // span three decades (torso mass 10 vs ankle inertia 0.01) so a fixed
        // absolute step is wrong at one end or the other; and the objective is
        // only weakly sensitive here (`J ≈ −0.83` with gradients ~1e-5, i.e.
        // relative sensitivity ~1e-5), which puts the FD roundoff floor
        // `ε·|J|/(h·|J'|)` at ~1e-4 for h = 1e-7·|π| and ~1e-7 for 1e-3·|π|.
        // Measured agreement across h ∈ [1e-3, 1e-7]·|π| bottoms out at
        // 1e-3·|π| (rel 1e-9…1e-6) and degrades monotonically below it —
        // roundoff, not a modelling error.
        let h = 1e-3 * nominal[b][k].abs().max(1e-2);
        let perturb = |sign: f64| -> f64 {
            let mut ps = nominal.clone();
            ps[b][k] += sign * h;
            j_at(&ps)
        };
        let fd = (perturb(1.0) - perturb(-1.0)) / (2.0 * h);
        let adj = g.d_inertia[b][k];
        if fd.abs() > 1e-8 {
            live += 1;
            let rel = (adj - fd).abs() / fd.abs();
            assert!(
                rel <= GATE,
                "ant dJ/dπ[body {b}][{k}]: adjoint {adj} vs fd {fd} (rel {rel:.3e})"
            );
        } else {
            assert!(
                adj.abs() <= 1e-7,
                "ant dJ/dπ[body {b}][{k}]: fd dead ({fd}) but adjoint {adj}"
            );
        }
    }
    assert!(live >= 4, "only {live} live ant channels — gate too weak");
}

// ---------------------------------------------------------------------------
// Gate 8 — vertex adjoint on a free-floating body (1e-4).
// ---------------------------------------------------------------------------

/// A free box released exactly touching the plane at rest, settling into the
/// penalty spring. This routes the contact channel through the *free* joint:
/// the wrench cotangent χ is now priced across a 6×6 articulated solve, and
/// the vertex Jacobian is evaluated at kinematics that came out of the
/// quaternion `joint_transform`.
///
/// Two closed-form anchors alongside the FD gate, as in `diff_adjoint.rs`'s
/// prismatic version: symmetric load sharing gives `∂z_T/∂z_v = −1/4` on each
/// of the four bottom vertices, and the top face plus both tangential axes are
/// structurally dead (no rotation develops, no friction in the model).
#[test]
fn free_box_on_plane_vertex_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 800;
    const GATE: f64 = 1e-4;
    const HALF: f64 = 0.05;
    const MASS: f64 = 0.5;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "crate",
            -1,
            SpatialTransform::identity(),
            si_from([MASS, 0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0]),
        )
        .build();
    let layout = DofLayout::of(&model);

    let ground = GroundContact {
        height: -HALF, // bottom face exactly touching at z = 0
        stiffness: 2e3,
        damping: 20.0,
    };
    let box_vertices = |half: f64| -> Vec<Vec3> {
        let mut v = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    v.push(Vec3::new(sx * half, sy * half, sz * half));
                }
            }
        }
        v
    };

    let ctrl = |_t: usize| DVec::zeros(layout.nv);
    let obj = q_objective(2); // free-joint translation z

    let j_with_mesh = |vertices: Vec<Vec3>| -> f64 {
        let meshes = [CollisionMesh { body: 0, vertices }];
        let ro = AdjointRollout {
            model: &model,
            contact: Some(ContactSetup {
                ground,
                meshes: &meshes,
            }),
            q0: layout.neutral_q(&model),
            v0: vec![0.0; layout.nv],
            steps: STEPS,
            ctrl: &ctrl,
        };
        rollout_objective(&ro, &obj)
    };

    let meshes = [CollisionMesh {
        body: 0,
        vertices: box_vertices(HALF),
    }];
    let rollout = AdjointRollout {
        model: &model,
        contact: Some(ContactSetup {
            ground,
            meshes: &meshes,
        }),
        q0: layout.neutral_q(&model),
        v0: vec![0.0; layout.nv],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    // Settled at the equilibrium penetration d = mg/(4k).
    let d_eq = MASS * 9.81 / (4.0 * ground.stiffness);
    assert!(
        (g.objective + d_eq).abs() < 1e-6,
        "box should settle at z = −mg/4k = {}, got {}",
        -d_eq,
        g.objective
    );

    // h = 1e-7 m: truncation and roundoff both ~1e-9 relative on the live
    // channel (|∂J/∂z| = 0.25).
    const H: f64 = 1e-7;
    let samples: [(usize, usize); 4] = [(0, 2), (6, 2), (0, 0), (1, 2)];
    for (vi, axis) in samples {
        let mut vp = box_vertices(HALF);
        let mut vm = box_vertices(HALF);
        match axis {
            0 => {
                vp[vi].x += H;
                vm[vi].x -= H;
            }
            1 => {
                vp[vi].y += H;
                vm[vi].y -= H;
            }
            _ => {
                vp[vi].z += H;
                vm[vi].z -= H;
            }
        }
        let fd = (j_with_mesh(vp) - j_with_mesh(vm)) / (2.0 * H);
        let adj = match axis {
            0 => g.d_vertices[0][vi].x,
            1 => g.d_vertices[0][vi].y,
            _ => g.d_vertices[0][vi].z,
        };
        if fd.abs() > 1e-6 {
            let rel = (adj - fd).abs() / fd.abs();
            assert!(
                rel <= GATE,
                "∂J/∂x[v{vi}][{axis}]: adjoint {adj} vs fd {fd} (rel {rel:.3e})"
            );
        } else {
            assert!(
                adj.abs() <= 1e-6,
                "∂J/∂x[v{vi}][{axis}]: fd dead ({fd}) but adjoint {adj}"
            );
        }
    }

    // Vertices are ordered (sx, sy, sz) with sz fastest: even = bottom.
    for vi in 0..8 {
        let gz = g.d_vertices[0][vi].z;
        if vi % 2 == 0 {
            let rel = (gz + 0.25).abs() / 0.25;
            assert!(
                rel <= 1e-3,
                "bottom vertex {vi}: ∂J/∂z = {gz}, expected −0.25 (rel {rel:.3e})"
            );
        } else {
            assert!(gz.abs() <= 1e-9, "top vertex {vi} should be dead, got {gz}");
        }
        assert!(
            g.d_vertices[0][vi].x.abs() <= 1e-9 && g.d_vertices[0][vi].y.abs() <= 1e-9,
            "tangential channels must be dead without rotation/friction (v{vi})"
        );
    }
}
