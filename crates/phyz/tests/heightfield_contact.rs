//! Heightfield ground contact on the CPU impulse path.
//!
//! Three questions, in rising order of terrain:
//! 1. Is a flat heightfield *the same ground* as the flat plane it replaces?
//! 2. Does a uniform slope reproduce the tilted-gravity workaround it was
//!    built to retire (ipse fakes ramps by tilting the model's gravity)?
//! 3. Does curvature actually steer a body — does a ball let loose on the
//!    wall of a bowl end up at the bottom?

use phyz::contact::ContactMaterial;
use phyz::model::{Geometry, Heightfield, ModelBuilder};
use phyz::sim::Simulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};

fn free_body_model(geometry: Geometry, start: Vec3, mass: f64) -> phyz::model::Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::from_translation(start),
            SpatialInertia::new(mass, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build();
    model.bodies[0].geometry = Some(geometry);
    model
}

/// A flat 1×1 heightfield is not "approximately" the plane — the stepped
/// trajectories must be bit-for-bit identical, because the degenerate case
/// is the bridge every existing caller walks across.
#[test]
fn flat_heightfield_matches_flat_plane_exactly() {
    let geom = Geometry::Box {
        half_extents: Vec3::new(0.1, 0.1, 0.1),
    };
    let model = free_body_model(geom, Vec3::new(0.0, 0.0, 0.5), 1.0);
    let material = ContactMaterial::default();
    let hf = Heightfield::flat(0.0);

    let sim_plane = Simulator::new();
    let sim_hf = Simulator::new();
    let mut s_plane = model.default_state();
    let mut s_hf = model.default_state();

    for _ in 0..1500 {
        sim_plane.step_with_contacts(&model, &mut s_plane, 0.0, &material);
        sim_hf.step_with_contacts_heightfield(&model, &mut s_hf, &hf, &material);
    }

    for k in 0..model.nq {
        assert_eq!(
            s_plane.q[k], s_hf.q[k],
            "q[{k}] diverged: plane {} vs heightfield {}",
            s_plane.q[k], s_hf.q[k]
        );
    }
    for k in 0..model.nv {
        assert_eq!(s_plane.v[k], s_hf.v[k], "v[{k}] diverged");
    }
    // And the box actually landed rather than both paths falling forever.
    assert!(
        (s_plane.q[5] - (-0.4)).abs() < 5e-3,
        "box did not settle on the plane: q_z = {}",
        s_plane.q[5]
    );
}

/// A frictionless ball on a uniform 5° heightfield slides with the same
/// acceleration as on flat ground with gravity tilted 5° — the workaround
/// this feature replaces. `a = g·sinθ` either way; the two simulations must
/// agree on the speed picked up, and the analytic value must be right.
#[test]
fn uniform_slope_matches_tilted_gravity() {
    let theta: f64 = 5.0_f64.to_radians();
    let g = 9.81;
    let geom = Geometry::Sphere { radius: 0.1 };
    let material = ContactMaterial {
        friction: 0.0,
        ..ContactMaterial::default()
    };

    // Terrain: slope rising along -x at tan(θ), so the ball slides toward +x.
    // 64 m wide, centred, so the run never leaves the grid.
    let mut hf = Heightfield::new(Vec3::new(-32.0, -32.0, 0.0), 1.0, 65, 65);
    for iy in 0..65 {
        for ix in 0..65 {
            let x = -32.0 + ix as f64;
            hf.heights[iy * 65 + ix] = (-x * theta.tan()) as f32;
        }
    }

    // Ball starts just above the surface at x = 0 (surface height 0 there).
    let model_hf = free_body_model(geom.clone(), Vec3::new(0.0, 0.0, 0.12), 1.0);
    let sim_hf = Simulator::new();
    let mut s_hf = model_hf.default_state();

    // The workaround: flat ground, gravity rotated 5° about y.
    let mut model_tilt = ModelBuilder::new()
        .gravity(Vec3::new(g * theta.sin(), 0.0, -g * theta.cos()))
        .dt(0.001)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, 0.12)),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build();
    model_tilt.bodies[0].geometry = Some(geom);
    let sim_tilt = Simulator::new();
    let mut s_tilt = model_tilt.default_state();

    let steps = 1000; // 1 s
    for _ in 0..steps {
        sim_hf.step_with_contacts_heightfield(&model_hf, &mut s_hf, &hf, &material);
        sim_tilt.step_with_contacts(&model_tilt, &mut s_tilt, 0.0, &material);
    }

    let speed_hf = Vec3::new(s_hf.v[3], s_hf.v[4], s_hf.v[5]).norm();
    let speed_tilt = Vec3::new(s_tilt.v[3], s_tilt.v[4], s_tilt.v[5]).norm();
    let expected = g * theta.sin() * 1.0; // v = a·t after 1 s of sliding

    assert!(
        (speed_hf - speed_tilt).abs() < 0.02 * expected.max(1e-9),
        "slope {speed_hf:.4} m/s vs tilted gravity {speed_tilt:.4} m/s"
    );
    assert!(
        (speed_hf - expected).abs() < 0.05 * expected,
        "slide speed {speed_hf:.4} != g·sinθ·t = {expected:.4}"
    );
    // Still riding the surface, not tunnelled through or bounced away.
    // (`q`'s linear slots are offsets from the spawn transform; body_xform
    // carries the world pose.)
    let pos = s_hf.body_xform[0].pos;
    let z = pos.z;
    let surface = hf.height(pos.x, pos.y);
    assert!(
        (z - surface - 0.1).abs() < 0.02,
        "ball is {z:.3} over surface {surface:.3}, expected ~0.1 above"
    );
}

/// A ball released on the wall of a bowl ends up at the bottom. Curvature is
/// the whole point of a heightfield — the normal must steer the ball inward,
/// and contact dissipation must eventually park it near the centre.
#[test]
fn ball_settles_at_the_bottom_of_a_bowl() {
    // Paraboloid bowl h = 0.5·r², 4 m wide, 10 cm cells.
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
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::from_translation(start),
            // A deliberately huge moment of inertia: the ball slides instead
            // of rolling, so friction can actually dissipate its energy. A
            // freely *rolling* ball in a frictionless-at-the-contact-patch
            // sense orbits a paraboloid forever — with unit inertia this
            // test watched the ball reach the bottom, convert slide to spin,
            // and climb back out to a permanent r ≈ 0.4 orbit.
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 10.0),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.1 });
    let material = ContactMaterial {
        friction: 0.4,
        ..ContactMaterial::default()
    };
    let sim = Simulator::new();
    let mut state = model.default_state();

    for _ in 0..4000 {
        sim.step_with_contacts_heightfield(&model, &mut state, &hf, &material);
    }

    let pos = state.body_xform[0].pos;
    let (x, y, z) = (pos.x, pos.y, pos.z);
    let r = (x * x + y * y).sqrt();
    assert!(z.is_finite() && r.is_finite(), "bowl produced NaN");
    assert!(
        r < 0.25,
        "ball did not reach the bottom: r = {r:.3} (started at {start_r})"
    );
    // Resting on the surface, not through it.
    let gap = z - hf.height(x, y);
    assert!(
        (gap - 0.1).abs() < 0.03,
        "ball rests {gap:.3} above the surface, expected ~0.1"
    );
}
