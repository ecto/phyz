//! Box contacts are manifolds, not points.
//!
//! The regression that forced this: a hip-yawed foot — a box rotated about
//! Z — contacted the deck through its single deepest corner and rocked on
//! it, and every loose-stance GPU rollout died in about a second. A box
//! resting on a plane needs its face's corners each pushing back, or it has
//! no roll/pitch stiffness at all.

use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::BodyContactGains;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Joint, ModelBuilder};

/// Contact natural frequency, rad/s. `omega = 1/tc` for MuJoCo's solref
/// time constant, so 50 is its 0.02 s default: every body then rests at
/// `g/omega^2` of penetration regardless of mass, which is the property
/// that lets one setting serve a torso and a toe at once.
const OMEGA: f64 = 50.0;

/// A free box, yawed 30° about Z, dropped onto the ground: it must settle
/// FLAT and stay put — not rock, not spin, not walk away.
#[test]
fn a_yawed_box_rests_flat_without_rocking() {
    let half = Vec3::new(0.1, 0.06, 0.02);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "foot",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(
                0.6,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1e-3, 2e-3, 3e-3)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });

    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_per_body(
        0.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
    )
    .unwrap();

    let mut s = model.default_state();
    s.q[2] = 0.5236; // 30° yaw — exp-coords angular block leads
    s.q[5] = half.z + 0.05; // small drop
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..2000 {
        sim.step();
    }
    let out = &sim.readback_states()[0];

    // Settled at rest height, still yawed, and NOT tilted: with single-point
    // contact this box rocks forever on one corner and roll/pitch walk away.
    let z = out.q[5];
    assert!(
        (z - half.z).abs() < 0.01,
        "box settled at {z:.4}, expected ~{:.4}",
        half.z
    );
    let (wx, wy) = (out.q[0], out.q[1]);
    assert!(
        wx.abs() < 0.02 && wy.abs() < 0.02,
        "box is tilted: roll/pitch exp-coords ({wx:.4}, {wy:.4}) — rocking on a corner"
    );
    let speed: f64 = (0..6).map(|i| out.v[i] * out.v[i]).sum::<f64>().sqrt();
    assert!(speed < 0.05, "box still moving at {speed:.3} after 2 s");
}
