//! Gates for the trajectory adjoint (`phyz::diff`).
//!
//! Every gradient claim is checked against an exact closed form or a central
//! finite difference of the *same* deterministic primal rollout. Step sizes
//! are chosen per scalar class so the central-difference truncation `O(h²)`
//! and roundoff `O(ε·|J|/h)` both sit well below the gate (stated at each
//! gate).

use phyz::diff::{
    AdjointRollout, CollisionMesh, ContactSetup, FinalStateObjective, GroundContact,
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

/// The objective `J = q_T[0]` (final position of the first DOF).
fn q0_objective<'a>() -> FinalStateObjective<'a> {
    FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[0],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[0] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    }
}

/// The objective `J = v_T[0]` (final velocity of the first DOF).
fn v0_objective<'a>() -> FinalStateObjective<'a> {
    FinalStateObjective {
        value: &|_q: &[f64], v: &[f64]| v[0],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gv = vec![0.0; v.len()];
            gv[0] = 1.0;
            (vec![0.0; q.len()], gv)
        },
    }
}

// ---------------------------------------------------------------------------
// Gate 1 — flywheel closed form (exact, 1e-12).
// ---------------------------------------------------------------------------

/// Torque-driven flywheel: revolute about Z, COM on the axis, gravity along
/// −Z (no moment). Semi-implicit Euler gives `v_T = steps·dt·τ/I_zz`
/// **exactly**, so `dJ/dI_zz = −steps·dt·τ/I_zz²` is an exact discrete closed
/// form and the adjoint (which is exact arithmetic, not FD) must hit it to
/// machine precision. All nine other inertia scalars have exactly zero
/// influence in this configuration.
#[test]
fn flywheel_inertia_gradient_matches_closed_form() {
    const TAU: f64 = 0.02;
    const IZZ: f64 = 0.05;
    const DT: f64 = 1.0 / 480.0;
    const STEPS: usize = 96; // 0.2 s

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_revolute_body(
            "disc",
            -1,
            SpatialTransform::identity(),
            si_from([1.4, 0.0, 0.0, 0.0, 0.03, 0.03, IZZ, 0.0, 0.0, 0.0]),
        )
        .build();

    let ctrl = |_t: usize| DVec::from_slice(&[TAU]);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: vec![0.0],
        v0: vec![0.0],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let obj = v0_objective();

    let g = adjoint_rollout_gradient(&rollout, &obj);

    let t_total = STEPS as f64 * DT;
    let j_expected = TAU * t_total / IZZ;
    assert!(
        (g.objective - j_expected).abs() / j_expected <= 1e-12,
        "J = {} vs closed form {}",
        g.objective,
        j_expected
    );

    let d_izz_expected = -TAU * t_total / (IZZ * IZZ);
    let rel = (g.d_inertia[0][6] - d_izz_expected).abs() / d_izz_expected.abs();
    assert!(
        rel <= 1e-12,
        "dJ/dIzz = {} vs closed form {} (rel {rel:.3e})",
        g.d_inertia[0][6],
        d_izz_expected
    );

    // Every other channel is structurally dead here (COM on the axis,
    // gravity along the axis): the adjoint must return exact zeros-ish.
    for (k, &gk) in g.d_inertia[0].iter().enumerate() {
        if k == 6 {
            continue;
        }
        assert!(
            gk.abs() <= 1e-10 * d_izz_expected.abs(),
            "π[{k}] should not influence the flywheel, got {gk}"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 2 — gravity pendulum, all 10 scalars vs central FD (1e-6).
// ---------------------------------------------------------------------------

/// One revolute joint about Z with gravity along −X and a deliberately
/// lopsided body (offset COM in all three axes, dense inertia): the mass,
/// COM, and inertia channels are all live. Each of the 10 `dJ/dπ` scalars is
/// gated against a central difference of the same primal rollout.
///
/// FD steps per class: mass 1e-5 (m ≈ 1.3), COM 1e-6 (|c| ≈ 0.4), inertia
/// 1e-7 (I ≈ 0.03). With J ≈ O(1) rad these put truncation ~1e-10 and
/// roundoff ~1e-9 relative — two-plus decades under the 1e-6 gate.
#[test]
fn pendulum_inertia_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 300;
    const GATE: f64 = 1e-6;

    let pi0: [f64; 10] = [1.3, 0.4, 0.1, 0.05, 0.02, 0.03, 0.04, 0.005, 0.002, 0.001];

    let build = |p: [f64; 10]| -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(-9.81, 0.0, 0.0))
            .dt(DT)
            .add_revolute_body("bob", -1, SpatialTransform::identity(), si_from(p))
            .build()
    };

    let ctrl = |_t: usize| DVec::from_slice(&[0.01]);
    let obj = q0_objective();

    let j_at = |p: [f64; 10]| -> f64 {
        let m = build(p);
        let ro = AdjointRollout {
            model: &m,
            contact: None,
            q0: vec![0.3],
            v0: vec![0.2],
            steps: STEPS,
            ctrl: &ctrl,
        };
        rollout_objective(&ro, &obj)
    };

    let model = build(pi0);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: vec![0.3],
        v0: vec![0.2],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    let h_for = |k: usize| match k {
        0 => 1e-5,     // mass
        1..=3 => 1e-6, // COM
        _ => 1e-7,     // inertia
    };

    let mut live = 0;
    for k in 0..10 {
        let h = h_for(k);
        let mut pp = pi0;
        pp[k] += h;
        let mut pm = pi0;
        pm[k] -= h;
        let fd = (j_at(pp) - j_at(pm)) / (2.0 * h);
        let adj = g.d_inertia[0][k];
        if fd.abs() > 1e-8 {
            live += 1;
            let rel = (adj - fd).abs() / fd.abs();
            assert!(
                rel <= GATE,
                "dJ/dπ[{k}]: adjoint {adj} vs fd {fd} (rel {rel:.3e})"
            );
        } else {
            assert!(
                adj.abs() <= 1e-8,
                "dJ/dπ[{k}]: fd is dead ({fd}) but adjoint says {adj}"
            );
        }
    }
    // Mass (0), COM x/y (1, 2), and I_zz (6) must all be live in this
    // configuration — the gate is not vacuously passing on zeros.
    assert!(
        live >= 4,
        "only {live} live channels — gate under-constrained"
    );
}

// ---------------------------------------------------------------------------
// Gate 3 — two-link chain with mixed axes: articulated propagation (1e-6).
// ---------------------------------------------------------------------------

/// Two links with different joint axes (Z then X, with a mount offset) so the
/// motion is genuinely 3D and the ABA backward pass propagates articulated
/// inertia across the joint. All 20 π-scalars gated against central FD —
/// this is the gate the single-body pendulum cannot provide.
#[test]
fn two_link_chain_inertia_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 250;
    const GATE: f64 = 1e-6;

    let p1: [f64; 10] = [
        1.1, 0.25, 0.05, -0.03, 0.02, 0.025, 0.03, 0.004, 0.001, 0.002,
    ];
    let p2: [f64; 10] = [
        0.7, 0.15, -0.04, 0.06, 0.01, 0.012, 0.014, 0.002, 0.001, 0.0015,
    ];

    let build = |pa: [f64; 10], pb: [f64; 10]| -> Model {
        let joint2 = Joint {
            joint_type: JointType::Revolute,
            parent_to_joint: SpatialTransform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
            axis: Vec3::new(1.0, 0.0, 0.0),
            damping: 0.02,
            ..Joint::revolute(SpatialTransform::identity())
        };
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .dt(DT)
            .add_revolute_body("link1", -1, SpatialTransform::identity(), si_from(pa))
            .add_body("link2", 0, joint2, si_from(pb))
            .build()
    };

    let ctrl = |_t: usize| DVec::from_slice(&[0.02, -0.01]);
    let obj = q0_objective();
    let q0 = vec![0.2, -0.3];
    let v0 = vec![0.1, 0.25];

    let j_at = |pa: [f64; 10], pb: [f64; 10]| -> f64 {
        let m = build(pa, pb);
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

    let model = build(p1, p2);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: q0.clone(),
        v0: v0.clone(),
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    // Inertia step 1e-6 (not 1e-7): the chain objective sits near O(1) with
    // gradients ~1e-3, so a 1e-7 step leaves the FD oracle's roundoff at the
    // 1e-6 gate itself; 1e-6 keeps truncation ~1e-8 and roundoff ~1e-7 rel.
    let h_for = |k: usize| match k {
        0 => 1e-5,
        1..=3 => 1e-6,
        _ => 1e-6,
    };

    let mut live = 0;
    for b in 0..2 {
        for k in 0..10 {
            let h = h_for(k);
            let perturb = |sign: f64| -> f64 {
                let mut pa = p1;
                let mut pb = p2;
                if b == 0 {
                    pa[k] += sign * h;
                } else {
                    pb[k] += sign * h;
                }
                j_at(pa, pb)
            };
            let fd = (perturb(1.0) - perturb(-1.0)) / (2.0 * h);
            let adj = g.d_inertia[b][k];
            if fd.abs() > 1e-8 {
                live += 1;
                let rel = (adj - fd).abs() / fd.abs();
                assert!(
                    rel <= GATE,
                    "dJ/dπ[body {b}][{k}]: adjoint {adj} vs fd {fd} (rel {rel:.3e})"
                );
            } else {
                assert!(
                    adj.abs() <= 1e-8,
                    "dJ/dπ[body {b}][{k}]: fd dead ({fd}) but adjoint {adj}"
                );
            }
        }
    }
    // A 3D two-link chain should light up the large majority of channels.
    assert!(
        live >= 14,
        "only {live}/20 live channels — chain not exercising the packing"
    );
}

// ---------------------------------------------------------------------------
// Gate 4 — determinism (bit-identical).
// ---------------------------------------------------------------------------

#[test]
fn adjoint_is_deterministic() {
    const DT: f64 = 1e-3;
    let model = ModelBuilder::new()
        .gravity(Vec3::new(-9.81, 0.0, 0.0))
        .dt(DT)
        .add_revolute_body(
            "bob",
            -1,
            SpatialTransform::identity(),
            si_from([1.3, 0.4, 0.1, 0.05, 0.02, 0.03, 0.04, 0.005, 0.002, 0.001]),
        )
        .build();
    let ctrl = |_t: usize| DVec::from_slice(&[0.01]);
    let obj = q0_objective();
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: vec![0.3],
        v0: vec![0.2],
        steps: 200,
        ctrl: &ctrl,
    };

    let a = adjoint_rollout_gradient(&rollout, &obj);
    let b = adjoint_rollout_gradient(&rollout, &obj);
    assert_eq!(a.objective, b.objective, "objective must be bit-identical");
    for k in 0..10 {
        assert_eq!(
            a.d_inertia[0][k], b.d_inertia[0][k],
            "dJ/dπ[{k}] must be bit-identical"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 5 — contact vertex adjoint: box settling on the plane (1e-4).
// ---------------------------------------------------------------------------

/// A box on a vertical prismatic joint, released at rest exactly touching
/// the ground, settling into the penalty spring. Releasing *in touch with
/// zero velocity* keeps the whole trajectory inside one smooth branch of the
/// contact law (no impact-time discontinuities), so a central FD on a vertex
/// coordinate is a clean oracle.
///
/// Two closed-form anchors on top of the FD gate: at equilibrium the four
/// bottom vertices share the load symmetrically, so `∂q_T/∂z_v = −1/4`
/// per bottom vertex; and vertices that never touch (top face) or move
/// tangentially (x, y — no rotation, no friction) contribute exactly zero.
#[test]
fn box_on_plane_vertex_gradient_matches_fd() {
    const DT: f64 = 1e-3;
    const STEPS: usize = 800;
    const GATE: f64 = 1e-4;
    const HALF: f64 = 0.05;
    const MASS: f64 = 0.5;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_prismatic_body(
            "box",
            -1,
            SpatialTransform::identity(),
            Vec3::new(0.0, 0.0, 1.0),
            si_from([MASS, 0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0]),
        )
        .build();

    let ground = GroundContact {
        height: -HALF, // bottom face exactly touching at q = 0
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

    let ctrl = |_t: usize| DVec::from_slice(&[0.0]);
    let obj = q0_objective();

    let j_with_mesh = |vertices: Vec<Vec3>| -> f64 {
        let meshes = [CollisionMesh { body: 0, vertices }];
        let ro = AdjointRollout {
            model: &model,
            contact: Some(ContactSetup {
                ground,
                meshes: &meshes,
            }),
            q0: vec![0.0],
            v0: vec![0.0],
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
        q0: vec![0.0],
        v0: vec![0.0],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    // Settled? equilibrium penetration d = mg/(4k) below the touch height.
    let d_eq = MASS * 9.81 / (4.0 * ground.stiffness);
    assert!(
        (g.objective + d_eq).abs() < 1e-6,
        "box should settle at q = −mg/4k = {}, got {}",
        -d_eq,
        g.objective
    );

    // FD gate on sampled vertex coordinates: two bottom-z, one bottom-x, one
    // top-z. h = 1e-7 m keeps truncation and roundoff ~1e-9 relative on the
    // live channel (|∂J/∂z| = 0.25).
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

    // Closed-form anchors. Vertices are ordered (sx, sy, sz) with sz fastest:
    // even indices are bottom (sz = −1), odd are top.
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

// ---------------------------------------------------------------------------
// Gate 6 — contact vertex adjoint under rotation: tilting paddle (1e-4).
// ---------------------------------------------------------------------------

/// A box offset along +X on a revolute joint about Y, resting tilted on the
/// ground: the vertex→wrench chain now includes a live rotation (E ≠ I), a
/// torque arm, and rotational vertex velocities — the frame-convention gate
/// the translating box cannot provide. Started near its contact equilibrium
/// (found by settling once) so the FD oracle stays on one smooth branch.
#[test]
fn tilting_paddle_vertex_gradient_matches_fd() {
    const DT: f64 = 5e-4;
    const STEPS: usize = 1200;
    const GATE: f64 = 1e-4;

    let joint = Joint {
        joint_type: JointType::Revolute,
        parent_to_joint: SpatialTransform::identity(),
        axis: Vec3::new(0.0, 1.0, 0.0),
        damping: 0.05,
        ..Joint::revolute(SpatialTransform::identity())
    };
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_body(
            "paddle",
            -1,
            joint,
            si_from([0.5, 0.3, 0.0, 0.0, 5e-4, 4e-3, 4e-3, 0.0, 0.0, 0.0]),
        )
        .build();

    let ground = GroundContact {
        height: -0.035,
        stiffness: 2e3,
        damping: 20.0,
    };
    // Box centered at (0.3, 0, 0), half extents (0.05, 0.04, 0.03).
    let paddle_vertices = || -> Vec<Vec3> {
        let mut v = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    v.push(Vec3::new(0.3 + sx * 0.05, sy * 0.04, sz * 0.03));
                }
            }
        }
        v
    };

    let ctrl = |_t: usize| DVec::from_slice(&[0.0]);
    let obj = q0_objective();

    let j_with_mesh = |vertices: Vec<Vec3>, q0: f64| -> f64 {
        let meshes = [CollisionMesh { body: 0, vertices }];
        let ro = AdjointRollout {
            model: &model,
            contact: Some(ContactSetup {
                ground,
                meshes: &meshes,
            }),
            q0: vec![q0],
            v0: vec![0.0],
            steps: STEPS,
            ctrl: &ctrl,
        };
        rollout_objective(&ro, &obj)
    };

    // Settle once from a shallow tilt to find the equilibrium; start the
    // gated rollout there (smooth branch for the FD oracle).
    let q_eq = j_with_mesh(paddle_vertices(), 0.012);

    let meshes = [CollisionMesh {
        body: 0,
        vertices: paddle_vertices(),
    }];
    let rollout = AdjointRollout {
        model: &model,
        contact: Some(ContactSetup {
            ground,
            meshes: &meshes,
        }),
        q0: vec![q_eq],
        v0: vec![0.0],
        steps: STEPS,
        ctrl: &ctrl,
    };
    let g = adjoint_rollout_gradient(&rollout, &obj);

    // FD on sampled coordinates of the two contacting vertices (far bottom
    // edge, indices with sx = +1, sz = −1) plus a never-touching top vertex.
    const H: f64 = 1e-7;
    let samples: [(usize, usize); 5] = [(4, 2), (6, 2), (4, 0), (6, 1), (1, 2)];
    let mut live = 0;
    for (vi, axis) in samples {
        let mut vp = paddle_vertices();
        let mut vm = paddle_vertices();
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
        let fd = (j_with_mesh(vp, q_eq) - j_with_mesh(vm, q_eq)) / (2.0 * H);
        let adj = match axis {
            0 => g.d_vertices[0][vi].x,
            1 => g.d_vertices[0][vi].y,
            _ => g.d_vertices[0][vi].z,
        };
        if fd.abs() > 1e-6 {
            live += 1;
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
    // Both z-channels and the x-channel of a contacting vertex must be live
    // under rotation (world height depends on body x through the tilt).
    assert!(
        live >= 3,
        "only {live} live samples — rotation chain not exercised"
    );
}
