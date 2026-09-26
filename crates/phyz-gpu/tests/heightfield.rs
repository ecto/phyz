//! Heightfield terrain in the GPU contact shader.
//!
//! Same ladder as the CPU suite (`phyz/tests/heightfield_contact.rs`):
//! 1. A flat 1×1 field must be *the same ground* as the plane path — the
//!    shader routes both through one code path, so this catches the
//!    degenerate case regressing.
//! 2. A uniform 5° slope must reproduce the tilted-gravity workaround it
//!    exists to replace.
//! 3. A bowl must steer a ball to its bottom — curvature, not just tilt.
//! 4. And the GPU penalty path must agree with the CPU impulse path on a
//!    bumpy field, to the loose f32/penalty-vs-impulse tolerance the README
//!    precision policy allows.

use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::BodyContactGains;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Heightfield, Model, ModelBuilder};

const TIME_CONST: f64 = 0.02;
/// The same contact response as a natural frequency: `omega = 1/tc`.
/// Mass-proportional gains `k = m*omega^2` put every body at the same
/// frequency, so resting penetration is `g/omega^2` regardless of mass.
const OMEGA: f64 = 1.0 / TIME_CONST;

fn free_body_model(geometry: Geometry, start: Vec3, inertia: f64) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::from_translation(start),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * inertia),
        )
        .build();
    model.bodies[0].geometry = Some(geometry);
    model
}

fn gains(model: &Model) -> Vec<BodyContactGains> {
    BodyContactGains::uniform_frequency(model, OMEGA, 1.0)
}

/// World position of the free body: spawn translation plus the linear q
/// slots (parent frame is the world for a root free joint).
fn world_pos(start: Vec3, q: &[f64]) -> Vec3 {
    start + Vec3::new(q[3], q[4], q[5])
}

#[test]
fn flat_heightfield_matches_flat_plane() {
    let start = Vec3::new(0.0, 0.0, 1.0);
    let model = free_body_model(
        Geometry::Box {
            half_extents: Vec3::new(0.1, 0.1, 0.1),
        },
        start,
        0.01,
    );

    let Ok(mut sim_plane) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut sim_hf = GpuBatchSimulator::new(model.clone(), 1).unwrap();

    let g = gains(&model);
    sim_plane
        .enable_ground_contact_per_body(0.0, 0.8, &g)
        .unwrap();
    sim_hf
        .enable_contact_terrain(0.0, 0.8, &g, &[], Some(&Heightfield::flat(0.0)))
        .unwrap();

    let st = model.default_state();
    sim_plane.load_states(std::slice::from_ref(&st));
    sim_hf.load_states(&[st]);

    for _ in 0..2000 {
        sim_plane.step();
        sim_hf.step();
    }

    let qp = sim_plane.readback_states()[0].q.clone();
    let qh = sim_hf.readback_states()[0].q.clone();
    for k in 0..qp.len() {
        assert!(
            (qp[k] - qh[k]).abs() < 1e-6,
            "q[{k}] diverged: plane {} vs flat heightfield {}",
            qp[k],
            qh[k]
        );
    }
    // And the box actually rests at the plane result, mg/k penetration
    // included — same prediction as tests/contact_drop.rs.
    let expected = -0.9 - 9.81 * TIME_CONST * TIME_CONST;
    assert!(
        (qp[5] - expected).abs() < 1e-3,
        "resting height {} != predicted {expected}",
        qp[5]
    );
}

/// Frictionless ball on a uniform 5° field slides at g·sinθ — the same
/// speed the tilted-gravity workaround produces on flat ground. Also
/// exercises the terrain-swap hook: the field is uploaded through
/// `set_heightfield` over an initially flat grid of the same size.
#[test]
fn uniform_slope_matches_tilted_gravity() {
    let theta: f64 = 5.0_f64.to_radians();
    let g = 9.81;
    let start = Vec3::new(0.0, 0.0, 0.12);
    let geom = Geometry::Sphere { radius: 0.1 };

    let model_hf = free_body_model(geom.clone(), start, 0.01);
    let Ok(mut sim_hf) = GpuBatchSimulator::new(model_hf, 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // Enable contact on a flat grid, then swap in the slope — the per-
    // training-iteration randomization path, not just the constructor path.
    let n = 65;
    let flat = Heightfield::new(Vec3::new(-32.0, -32.0, 0.0), 1.0, n, n);
    sim_hf
        .enable_contact_terrain(0.0, 0.0, &gains(&sim_hf.model.clone()), &[], Some(&flat))
        .unwrap();
    let mut slope = flat.clone();
    for iy in 0..n {
        for ix in 0..n {
            let x = -32.0 + ix as f64;
            slope.heights[iy * n + ix] = (-x * theta.tan()) as f32;
        }
    }
    sim_hf.set_heightfield(&slope).unwrap();

    // The workaround: flat plane, gravity tilted 5° about y.
    let mut model_tilt = ModelBuilder::new()
        .gravity(Vec3::new(g * theta.sin(), 0.0, -g * theta.cos()))
        .dt(0.001)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::from_translation(start),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build();
    model_tilt.bodies[0].geometry = Some(geom);
    let mut sim_tilt = GpuBatchSimulator::new(model_tilt, 1).unwrap();
    let g_tilt = gains(&sim_tilt.model.clone());
    sim_tilt
        .enable_ground_contact_per_body(0.0, 0.0, &g_tilt)
        .unwrap();

    let st_hf = sim_hf.model.default_state();
    sim_hf.load_states(&[st_hf]);
    let st_tilt = sim_tilt.model.default_state();
    sim_tilt.load_states(&[st_tilt]);

    for _ in 0..1000 {
        sim_hf.step();
        sim_tilt.step();
    }

    let s_hf = &sim_hf.readback_states()[0];
    let s_tilt = &sim_tilt.readback_states()[0];
    let speed_hf = Vec3::new(s_hf.v[3], s_hf.v[4], s_hf.v[5]).norm();
    let speed_tilt = Vec3::new(s_tilt.v[3], s_tilt.v[4], s_tilt.v[5]).norm();
    let expected = g * theta.sin() * 1.0;

    assert!(
        (speed_hf - speed_tilt).abs() < 0.02 * expected,
        "slope {speed_hf:.4} m/s vs tilted gravity {speed_tilt:.4} m/s"
    );
    assert!(
        (speed_hf - expected).abs() < 0.05 * expected,
        "slide speed {speed_hf:.4} != g·sinθ·t = {expected:.4}"
    );

    // Riding the surface, not through it: penalty rest depth is g·cosθ·tc².
    let p = world_pos(start, s_hf.q.as_slice());
    let gap = p.z - slope.height(p.x, p.y);
    assert!(
        (gap - 0.1).abs() < 0.01,
        "ball is {gap:.4} above the slope, expected ~0.1"
    );
}

/// A high-inertia ball (slides, doesn't roll — see the CPU twin for why)
/// released on the wall of a paraboloid bowl ends up parked near the centre.
#[test]
fn ball_settles_at_the_bottom_of_a_bowl() {
    let n = 41;
    let mut hf = Heightfield::new(Vec3::new(-2.0, -2.0, 0.0), 0.1, n, n);
    for iy in 0..n {
        for ix in 0..n {
            let x = -2.0 + 0.1 * ix as f64;
            let y = -2.0 + 0.1 * iy as f64;
            hf.heights[iy * n + ix] = (0.5 * (x * x + y * y)) as f32;
        }
    }

    let start_r = 1.0;
    let start = Vec3::new(start_r, 0.0, hf.height(start_r, 0.0) + 0.1);
    let model = free_body_model(Geometry::Sphere { radius: 0.1 }, start, 10.0);
    let Ok(mut sim) = GpuBatchSimulator::new(model, 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_contact_terrain(0.0, 0.4, &gains(&sim.model.clone()), &[], Some(&hf))
        .unwrap();
    let st = sim.model.default_state();
    sim.load_states(&[st]);

    for _ in 0..4000 {
        sim.step();
    }

    let s = &sim.readback_states()[0];
    let p = world_pos(start, s.q.as_slice());
    let r = (p.x * p.x + p.y * p.y).sqrt();
    assert!(p.z.is_finite() && r.is_finite(), "bowl produced NaN");
    assert!(
        r < 0.25,
        "ball did not reach the bottom: r = {r:.3} (started at {start_r})"
    );
    let gap = p.z - hf.height(p.x, p.y);
    assert!(
        (gap - 0.1).abs() < 0.03,
        "ball rests {gap:.3} above the surface, expected ~0.1"
    );
}

/// GPU penalty vs CPU impulse on a bumpy field. The two contact models are
/// different by construction (penalty sinks mg/k ≈ 4 mm; the convex solve
/// doesn't), so the comparison is where the body ends up, at the loose
/// tolerance the README's f32 precision policy sets for cross-engine runs —
/// not force-level agreement.
#[test]
fn gpu_matches_cpu_on_bumpy_field() {
    // Deterministic gentle bumps: ±3 cm on a 10 cm grid.
    let n = 41;
    let mut hf = Heightfield::new(Vec3::new(-2.0, -2.0, 0.0), 0.1, n, n);
    for iy in 0..n {
        for ix in 0..n {
            let x = -2.0 + 0.1 * ix as f64;
            let y = -2.0 + 0.1 * iy as f64;
            hf.heights[iy * n + ix] = (0.03 * (3.0 * x).sin() * (2.0 * y).cos()) as f32;
        }
    }

    let start = Vec3::new(0.15, 0.1, 0.4);
    let model = free_body_model(
        Geometry::Box {
            half_extents: Vec3::new(0.1, 0.1, 0.1),
        },
        start,
        0.01,
    );

    let Ok(mut gpu) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    gpu.enable_contact_terrain(0.0, 0.8, &gains(&model), &[], Some(&hf))
        .unwrap();
    let st = model.default_state();
    gpu.load_states(std::slice::from_ref(&st));

    let cpu_sim = phyz::sim::Simulator::new();
    let material = phyz::contact::ContactMaterial {
        friction: 0.8,
        ..phyz::contact::ContactMaterial::default()
    };
    let mut cpu_state = st;

    let steps = 2000;
    for _ in 0..steps {
        gpu.step();
        cpu_sim.step_with_contacts_heightfield(&model, &mut cpu_state, &hf, &material);
    }

    let gq = gpu.readback_states()[0].q.clone();
    let gpu_pos = world_pos(start, gq.as_slice());
    let cpu_pos = cpu_state.body_xform[0].pos;

    assert!(
        gpu_pos.z.is_finite() && cpu_pos.z.is_finite(),
        "a path produced NaN: gpu {gpu_pos:?}, cpu {cpu_pos:?}"
    );
    let diff = gpu_pos - cpu_pos;
    assert!(
        diff.norm() < 0.05,
        "engines disagree on the resting pose: gpu {gpu_pos:?} vs cpu {cpu_pos:?} \
         (|Δ| = {:.4})",
        diff.norm()
    );
    // Both settled on (not through) the terrain.
    for (name, p) in [("gpu", gpu_pos), ("cpu", cpu_pos)] {
        let gap = p.z - hf.height(p.x, p.y);
        assert!(
            gap > 0.05 && gap < 0.15,
            "{name} box floats/sinks: {gap:.3} above surface"
        );
    }
}
