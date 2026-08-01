//! Does the GPU ground-contact pipeline actually stop a falling body?
//!
//! The narrowest possible question, asked of the simplest possible model: a
//! single free box dropped from 1 m onto the plane at z = 0. If this fails,
//! nothing built on top of GPU contact means anything.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, ModelBuilder};

/// Contact response time constant (MuJoCo's solref default).
const TIME_CONST: f64 = 0.02;

#[test]
fn gpu_contact_stops_a_dropped_box() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.1, 0.1, 0.1),
    });

    let Ok(mut sim) = GpuBatchSimulator::new(model, 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact(0.0, TIME_CONST, 1.0, 0.8)
        .unwrap();
    let st = sim.model.default_state();
    sim.load_states(&[st]);

    for i in 0..2000 {
        sim.step();
        if i % 500 == 499 {
            let s = sim.readback_states();
            eprintln!("step {i}: z = {:.4}", s[0].q.as_slice()[2]);
        }
    }

    let s = sim.readback_states();
    let z = s[0].q.as_slice()[2];
    assert!(z.is_finite(), "contact produced NaN");

    // The box spawns at world z = 1.0 and has a 0.1 m half-extent, so it
    // rests with its centre at world z = 0.1, i.e. q = -0.9. A penalty
    // contact sinks in by exactly the depth where the spring carries the
    // weight. With mass-derived stiffness k = m/tc^2, that depth is
    // mg/k = g * tc^2 — independent of mass, which is the property that
    // makes one setting work for a torso and a toe at once.
    let expected = -0.9 - 9.81 * TIME_CONST * TIME_CONST;
    assert!(
        (z - expected).abs() < 1e-3,
        "resting height {z:.5} != predicted {expected:.5} (mg/k penetration)"
    );

    // And it is actually at rest, not still moving through.
    let vz = s[0].v.as_slice()[5];
    assert!(vz.abs() < 1e-2, "box has not settled: vz = {vz}");
}
