//! GPU ground contact: geometry coverage, stability, and contact readback.
//!
//! Issues #53 and #54: a mesh-collider robot fell straight through the GPU
//! ground plane (unsupported geometry silently packed as "no collision"),
//! a single global stiffness cannot serve mixed-mass models, and contact
//! state could not be observed from outside at all. These tests pin the
//! fixes: mesh AABB collision, per-body gains, the collidable-body count,
//! and `readback_contacts`.

use phyz_gpu::{BodyContactGains, GpuBatchSimulator};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const DT: f64 = 1e-3;
const G: f64 = 9.81;

fn ball_inertia(mass: f64, radius: f64) -> SpatialInertia {
    let i = 0.4 * mass * radius * radius;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

fn with_geometry(name: &str, inertia: SpatialInertia, geometry: Geometry) -> phyz_model::Body {
    let mut b = phyz_model::Body::new(name, inertia, -1, 0);
    b.geometry = Some(geometry);
    b
}

/// One free-floating body with the given collision geometry.
fn free_body_model(mass: f64, radius: f64, geometry: Geometry) -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(DT)
        .add_free_body_with_geometry(
            "body",
            -1,
            SpatialTransform::identity(),
            ball_inertia(mass, radius),
            with_geometry("body", ball_inertia(mass, radius), geometry),
        )
        .build()
}

fn gpu_sim(model: &Model, nworld: usize) -> Option<GpuBatchSimulator> {
    match GpuBatchSimulator::new(model.clone(), nworld) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("skipping GPU test (no adapter): {e}");
            None
        }
    }
}

/// A sphere dropped onto the ground must come to rest on it, and the contact
/// readback must report the resting normal force (= weight) at the resting
/// point.
#[test]
fn sphere_rests_on_ground_and_reports_contact() {
    let mass = 1.0;
    let radius = 0.1;
    let model = free_body_model(mass, radius, Geometry::Sphere { radius });
    let Some(mut sim) = gpu_sim(&model, 1) else {
        return;
    };

    // omega = 200 rad/s, zeta = 1: well inside the stability bound
    // (omega * dt = 0.2) and critically damped.
    let (omega, zeta) = (200.0, 1.0);
    let k = mass * omega * omega;
    let d = 2.0 * zeta * mass * omega;
    let collidable = sim.enable_ground_contact(0.0, k, d, 0.5).unwrap();
    assert_eq!(collidable, 1);

    let mut s = model.default_state();
    s.q[5] = 0.5; // free joint q = [wx, wy, wz, x, y, z]
    sim.load_states(&[s]);

    // Airborne: no contact reported.
    sim.step();
    let contacts = sim.readback_contacts().unwrap();
    assert!(!contacts[0][0].touching, "airborne body reports contact");
    assert_eq!(contacts[0][0].force.z, 0.0);

    for _ in 0..2000 {
        sim.step();
    }

    let states = sim.readback_states();
    let z = states[0].q[5];
    assert!(
        (z - radius).abs() < 0.02,
        "sphere should rest with centre one radius above ground, got z = {z}"
    );

    let contacts = sim.readback_contacts().unwrap();
    let c = &contacts[0][0];
    assert!(c.touching, "resting sphere must report touching");
    assert!(
        c.penetration > 0.0 && c.penetration < 0.01,
        "resting penetration should be small and positive, got {}",
        c.penetration
    );
    let weight = mass * G;
    assert!(
        (c.force.z - weight).abs() < 0.15 * weight,
        "resting normal force should be about the weight ({weight:.2} N), got {:.2}",
        c.force.z
    );
    assert!(
        c.point.z.abs() < 0.01,
        "contact point should sit on the ground plane, got z = {}",
        c.point.z
    );
}

/// The vcad regression from issue #54: a convex-hull (`Geometry::Mesh`)
/// collider used to pack as "no geometry", so the body fell straight through
/// a ground plane it had explicitly been given.
#[test]
fn mesh_body_stands_on_ground() {
    let mass = 1.0;
    let half = 0.1;
    // A cube-ish hull, as vcad's convex-hull colliders are.
    let vertices = vec![
        Vec3::new(-half, -half, -half),
        Vec3::new(half, -half, -half),
        Vec3::new(-half, half, -half),
        Vec3::new(half, half, -half),
        Vec3::new(-half, -half, half),
        Vec3::new(half, -half, half),
        Vec3::new(-half, half, half),
        Vec3::new(half, half, half),
    ];
    let faces = vec![[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7]];
    let model = free_body_model(mass, half, Geometry::Mesh { vertices, faces });
    let Some(mut sim) = gpu_sim(&model, 1) else {
        return;
    };

    let collidable = sim
        .enable_ground_contact(0.0, mass * 200.0 * 200.0, 2.0 * mass * 200.0, 0.5)
        .unwrap();
    assert_eq!(collidable, 1, "mesh geometry must be collidable");

    let mut s = model.default_state();
    s.q[5] = 0.5;
    sim.load_states(&[s]);
    for _ in 0..2000 {
        sim.step();
    }

    let states = sim.readback_states();
    let z = states[0].q[5];
    assert!(
        (z - half).abs() < 0.02,
        "mesh body should rest with its AABB bottom on the ground, got z = {z}"
    );

    let contacts = sim.readback_contacts().unwrap();
    assert!(
        contacts[0][0].touching,
        "resting mesh body must report touching"
    );
}

/// Issue #53's measured case: one global stiffness cannot serve a 5-kg body
/// and a 1-gram body at dt = 1e-3, but per-body gains at a shared contact
/// frequency can.
#[test]
fn mixed_masses_stable_with_per_body_gains() {
    let heavy_r = 0.1;
    let light_r = 0.01;
    let heavy_m = 5.0;
    let light_m = 0.001;
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(DT)
        .add_free_body_with_geometry(
            "heavy",
            -1,
            SpatialTransform::identity(),
            ball_inertia(heavy_m, heavy_r),
            with_geometry(
                "heavy",
                ball_inertia(heavy_m, heavy_r),
                Geometry::Sphere { radius: heavy_r },
            ),
        )
        .add_free_body_with_geometry(
            "light",
            -1,
            SpatialTransform::identity(),
            ball_inertia(light_m, light_r),
            with_geometry(
                "light",
                ball_inertia(light_m, light_r),
                Geometry::Sphere { radius: light_r },
            ),
        )
        .build();
    let Some(mut sim) = gpu_sim(&model, 1) else {
        return;
    };

    // The empty global window, reported instead of NaN.
    let unstable = phyz_gpu::GroundContactParams {
        ground_height: 0.0,
        stiffness: 1.354e4,
        damping: 10.0,
        friction: 0.5,
    };
    assert!(unstable.check_stability(&model).is_err());

    // Per-body gains at one shared frequency: both bodies stable.
    let gains = BodyContactGains::uniform_frequency(&model, 200.0, 1.0);
    let collidable = sim
        .enable_ground_contact_per_body(0.0, 0.5, &gains)
        .unwrap();
    assert_eq!(collidable, 2);

    let mut s = model.default_state();
    s.q[5] = 0.3; // heavy z
    s.q[9] = 0.5; // light x: offset so they don't overlap (no body-body contact anyway)
    s.q[11] = 0.3; // light z
    sim.load_states(&[s]);
    for _ in 0..3000 {
        sim.step();
    }

    let states = sim.readback_states();
    for j in 0..model.nq {
        assert!(
            states[0].q[j].is_finite(),
            "q[{j}] went non-finite with per-body gains"
        );
    }
    let heavy_z = states[0].q[5];
    let light_z = states[0].q[11];
    assert!(
        (heavy_z - heavy_r).abs() < 0.02,
        "heavy body should rest on the ground, got z = {heavy_z}"
    );
    assert!(
        (light_z - light_r).abs() < 0.01,
        "light body should rest on the ground, got z = {light_z}"
    );
}

/// Issue #54: enabling contact on a model with no collidable geometry is an
/// error, not a silent no-op.
#[test]
fn no_collidable_geometry_is_an_error() {
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(DT)
        .add_free_body(
            "bare",
            -1,
            SpatialTransform::identity(),
            ball_inertia(1.0, 0.1),
        )
        .build();
    let Some(mut sim) = gpu_sim(&model, 1) else {
        return;
    };

    let err = sim.enable_ground_contact(0.0, 1e4, 50.0, 0.5).unwrap_err();
    assert!(
        err.contains("no body has GPU-collidable geometry"),
        "unexpected error: {err}"
    );
    assert!(err.contains("bare"), "error should name the body: {err}");
}

/// Contact readback without an enabled contact pass is an error, not zeros.
#[test]
fn contact_readback_requires_contact_enabled() {
    let model = free_body_model(1.0, 0.1, Geometry::Sphere { radius: 0.1 });
    let Some(sim) = gpu_sim(&model, 1) else {
        return;
    };
    let err = sim.readback_contacts().unwrap_err();
    assert!(err.contains("not enabled"), "unexpected error: {err}");
}
