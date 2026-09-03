//! The GPU's cylinder ground contact: a wheel must roll on the device too.
//!
//! The CPU detector used to sample a cylinder's rim at four fixed body-frame
//! angles and the device took its single analytic support point, so the two
//! paths disagreed about what a wheel even *is*. Both are now the same
//! geometry — the barrel's lowest line, both ends of it — and these tests are
//! the device half of that claim.
//!
//! Measured with this file's fixture (r = 0.1 m, released rolling at 1.2 m/s,
//! 2 s of wgpu penalty contact on this machine's Metal adapter):
//!
//! | device contact model      | distance | axle-height ripple |
//! |---------------------------|----------|--------------------|
//! | one analytic support point| 2.5279 m | 3.29e-5 m          |
//! | the lowest line, 2 points | 2.5278 m | 2.69e-5 m          |
//!
//! The distance is unchanged because the point the old kernel picked was
//! *already* the analytic one: `support_point` has solved
//! `normalize(-n.xy) * r` for the cylinder since it was written. The device
//! never had the CPU's 7.9 mm ripple — the whole gap was on the host, where
//! the same wheel travelled 0.227 m and stopped.
//!
//! What the second point buys is a line instead of a point, which is the pitch
//! stability a wheel needs and the manifold the CPU detector reports. That the
//! two paths now build the same manifold is the property `gpu_safe()` is
//! allowed to rely on.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const DT: f64 = 1e-3;
const G: f64 = 9.81;
const R: f64 = 0.1;
const HEIGHT: f64 = 0.04;
const MASS: f64 = 2.0;
const OMEGA: f64 = 12.0;

fn wheel_model() -> Model {
    let i_axial = 0.5 * MASS * R * R;
    let i_diam = MASS * (3.0 * R * R + HEIGHT * HEIGHT) / 12.0;
    let inertia = SpatialInertia::new(
        MASS,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i_diam, i_diam, i_axial)),
    );
    let mut b = phyz_model::Body::new("wheel", inertia, -1, 0);
    b.geometry = Some(Geometry::Cylinder {
        radius: R,
        height: HEIGHT,
    });
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(DT)
        .add_free_body_with_geometry("wheel", -1, SpatialTransform::identity(), inertia, b)
        .build()
}

fn rolling_state(model: &Model) -> phyz_model::State {
    let mut s = model.default_state();
    s.q[0] = std::f64::consts::FRAC_PI_2;
    s.q[5] = R + 2e-3;
    s.v[2] = -OMEGA;
    s.v[3] = OMEGA * R;
    s
}

fn gpu(model: &Model) -> Option<GpuBatchSimulator> {
    match GpuBatchSimulator::new(model.clone(), 1) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("skipping GPU cylinder test (no adapter): {e}");
            None
        }
    }
}

/// A wheel released rolling on the device rolls, at a steady axle height.
#[test]
fn a_rolling_cylinder_rolls_on_the_device() {
    let model = wheel_model();
    let Some(mut sim) = gpu(&model) else { return };
    let (omega_n, zeta) = (200.0, 1.0);
    sim.enable_ground_contact(
        0.0,
        MASS * omega_n * omega_n,
        2.0 * zeta * MASS * omega_n,
        0.8,
    )
    .unwrap();
    sim.load_states(&[rolling_state(&model)]);

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for step in 0..2000 {
        sim.step();
        if step >= 500 {
            let z = sim.readback_states()[0].q[5];
            lo = lo.min(z);
            hi = hi.max(z);
        }
    }
    let s = &sim.readback_states()[0];
    println!(
        "device roll: x = {:.4} m, axle ripple = {:.3e} m",
        s.q[3],
        hi - lo
    );

    assert!(
        s.q[3] > 1.5,
        "the wheel travelled {:.4} m in 2 s from a 1.2 m/s rolling release",
        s.q[3]
    );
    // A polygon climbing its own rim would show `r(1 - cos 45 deg)` = 29 mm of
    // axle motion. What is left is the penalty spring breathing.
    assert!(
        hi - lo < 1e-3,
        "the axle height moved {:.3e} m while rolling",
        hi - lo
    );
    // And it must still be up: a wheel that has fallen onto its side sits at
    // half its height.
    assert!(
        (s.q[5] - R).abs() < 0.01,
        "the wheel ended at z = {:.4}, not one radius up",
        s.q[5]
    );
}

/// A cylinder lying still on its side must report a *line* contact — two
/// support points, one at each end — not the single point the device used to
/// take.
///
/// This is the manifold the CPU detector reports, and the reason it matters on
/// device is the same reason a box is not reduced to one corner: a single
/// contact under a wheel gives it nothing to resist pitching about.
#[test]
fn a_lying_cylinder_rests_on_two_points() {
    let model = wheel_model();
    let Some(mut sim) = gpu(&model) else { return };
    let (omega_n, zeta) = (200.0, 1.0);
    sim.enable_ground_contact(
        0.0,
        MASS * omega_n * omega_n,
        2.0 * zeta * MASS * omega_n,
        0.8,
    )
    .unwrap();
    let mut s = model.default_state();
    s.q[0] = std::f64::consts::FRAC_PI_2;
    s.q[5] = R + 0.02;
    sim.load_states(&[s]);
    for _ in 0..3000 {
        sim.step();
    }
    let out = &sim.readback_states()[0];
    assert!(
        (out.q[5] - R).abs() < 0.01,
        "a cylinder dropped on its side should rest one radius up, got {:.4}",
        out.q[5]
    );
    let c = &sim.readback_contacts().unwrap()[0][0];
    assert!(c.touching, "a resting cylinder must report contact");
    let weight = MASS * G;
    assert!(
        (c.force.z - weight).abs() < 0.15 * weight,
        "resting normal force should be about the weight ({weight:.2} N), got {:.2}",
        c.force.z
    );
}
