//! Default (explicit dry friction) physics is bit-for-bit what it was before
//! `FrictionLossIntegration` existed: the golden values below were captured
//! at a7c34a0 (the commit before the opt-in) with this exact model and state.

use phyz_contact::{ContactMaterial, ContactSolverConfig, assemble, find_ground_contacts_model};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

fn model() -> Model {
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(2e-3)
        .add_free_body(
            "foot",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.02, 0.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(1e-3, 3e-3, 4e-3)),
            ),
        )
        .add_revolute_body(
            "link",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, 0.02)),
            SpatialInertia::new(
                0.1,
                Vec3::new(0.03, 0.0, 0.04),
                Mat3::from_diagonal(&Vec3::new(1e-4, 1e-4, 1e-5)),
            ),
        )
        .build();
    m.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.1, 0.05, 0.02),
    });
    let j = &mut m.joints[1];
    j.axis = Vec3::new(0.0, 1.0, 0.0);
    j.damping = 0.05;
    j.armature = 1e-3;
    j.friction_loss = 0.05;
    m
}

fn digest(xs: &[f64]) -> u64 {
    // FNV-1a over the raw bits.
    let mut h: u64 = 0xcbf29ce484222325;
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn digests() -> (u64, u64) {
    let m = model();
    let mut s = m.default_state();
    s.q = DVec::from_slice(&[0.01, -0.02, 0.003, 0.001, 0.0, 0.0195, 0.2]);
    s.v = DVec::from_slice(&[0.1, -0.05, 0.02, 0.03, 0.0, -0.01, 3e-4]);
    let (xf, _) = phyz_rigid::forward_kinematics(&m, &s);
    s.body_xform = xf;
    let qdd = phyz_rigid::aba(&m, &s);
    let mat = ContactMaterial::default();
    let contacts = find_ground_contacts_model(&m, &s, 0.0, mat.margin);
    assert!(!contacts.is_empty());
    let free = &s.v + &(&qdd * m.dt);
    let asm = assemble(
        &m,
        &s,
        &contacts,
        &vec![mat; m.bodies.len()],
        &free,
        m.dt,
        &ContactSolverConfig::simulation(),
    );
    (digest(qdd.as_slice()), digest(asm.inv_mass.as_slice()))
}

#[test]
fn explicit_friction_default_is_bit_identical_to_pre_opt_in() {
    let (a, b) = digests();
    eprintln!("GOLDEN aba={a:#x} inv_mass={b:#x}");
    assert_eq!((a, b), (GOLD_ABA, GOLD_INV_MASS));
}

// Captured at a7c34a0 on aarch64-apple-darwin (release). f64 IEEE ops are
// deterministic per target; a mismatch elsewhere means the default path moved.
const GOLD_ABA: u64 = 0x9ffc5ef96e341111;
const GOLD_INV_MASS: u64 = 0xa6afb23bc4680670;
