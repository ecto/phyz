//! contact-speed's exactness gate (docs/contact-speed.md).
//!
//! Every commit on the contact-speed lane is a pure-speed change to collision
//! detection or contact assembly. The claim each one makes is that the
//! trajectory does not move by a single bit, and this test is where that claim
//! is checked: each scene is stepped through `Simulator::step_with_contacts`
//! and every state along the way is folded into one FNV-1a hash of the raw
//! `q`/`v` bits. The goldens were recorded on the lane's base (`contact-audit`,
//! 03fc682) before any change landed.
//!
//! The scenes are chosen to reach every detection path the K1 does, and a few
//! it does not:
//!
//! - `pile`: seven free bodies (boxes, cylinders, a capsule, a sphere) dropped
//!   into each other and the ground. It includes a box-cylinder pair whose
//!   AABBs overlap for the whole run while the shapes stay ~1 cm apart — the
//!   K1's hand-against-hip case — and box-box / box-cylinder / sphere-capsule
//!   pairs that do come into contact (GJK, EPA, the manifold).
//! - `k1_stance`, `k1_step`: the K1 as ipse builds it (URDF + vendor pads,
//!   20 collision shapes), under scripted PD. Skipped when the vendor assets
//!   are absent; `K1_DIR` points at them.
//!
//! **If this fails, the bits moved.** On this lane that is a bug in the
//! change, not a reason to update the constant. A change that legitimately
//! moves bits (a reordered floating-point sum) does not belong behind this
//! gate; it gets its own bounded-difference test and says so in the doc.

#[path = "support/k1.rs"]
mod k1;

use phyz::Simulator;
use phyz::phyz_contact::ContactMaterial;
use phyz::phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
use phyz::phyz_model::{Geometry, Model, ModelBuilder, State};

fn fnv(h: &mut u64, state: &State) {
    for x in state.q.iter().chain(state.v.iter()) {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }
}

fn rollout(model: &Model, mut state: State, steps: usize, mut ctrl: impl FnMut(&mut State)) -> u64 {
    let sim = Simulator::new();
    let mat = ContactMaterial::default();
    let mut h: u64 = 0xcbf29ce484222325;
    for _ in 0..steps {
        ctrl(&mut state);
        sim.step_with_contacts(model, &mut state, 0.0, &mat);
        fnv(&mut h, &state);
    }
    h
}

/// `(geometry, [wx, wy, wz, x, y, z])` for each free body of the pile.
fn pile() -> (Model, State) {
    use std::f64::consts::FRAC_PI_2;
    let q = std::f64::consts::FRAC_PI_4;
    let bodies: Vec<(Geometry, [f64; 6])> = vec![
        // A 45-deg box, and an upright cylinder 1 cm off its face along the
        // diagonal: overlapping AABBs, never in contact.
        (
            Geometry::Box {
                half_extents: Vec3::new(0.1, 0.1, 0.1),
            },
            [0.0, 0.0, q, 0.0, 0.0, 0.1],
        ),
        (
            Geometry::Cylinder {
                radius: 0.05,
                height: 0.2,
            },
            [0.0, 0.0, 0.0, 0.16 * q.cos(), 0.16 * q.sin(), 0.1],
        ),
        // A small tilted box dropped onto the big one.
        (
            Geometry::Box {
                half_extents: Vec3::new(0.05, 0.05, 0.05),
            },
            [0.05, 0.02, 0.3, 0.0, 0.0, 0.26],
        ),
        // A lying capsule with a sphere dropped on it.
        (
            Geometry::Capsule {
                radius: 0.04,
                length: 0.2,
            },
            [FRAC_PI_2, 0.0, 0.0, -0.3, 0.0, 0.25],
        ),
        (
            Geometry::Sphere { radius: 0.06 },
            [0.0, 0.0, 0.0, -0.3, 0.02, 0.45],
        ),
        // A lying cylinder with a flat box dropped across it.
        (
            Geometry::Cylinder {
                radius: 0.05,
                height: 0.1,
            },
            [FRAC_PI_2, 0.0, 0.0, 0.0, -0.35, 0.3],
        ),
        (
            Geometry::Box {
                half_extents: Vec3::new(0.08, 0.03, 0.02),
            },
            [0.0, 0.0, 0.2, 0.02, -0.35, 0.45],
        ),
    ];
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3);
    for _ in &bodies {
        b = b.add_free_body(
            "p",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(0.5, 0.08),
        );
    }
    let mut model = b.build();
    let mut state = model.default_state();
    for (k, (g, q)) in bodies.into_iter().enumerate() {
        model.bodies[k].geometry = Some(g);
        for (i, x) in q.iter().enumerate() {
            state.q[6 * k + i] = *x;
        }
    }
    (model, state)
}

fn k1_rollout(script: &str, steps: usize) -> Option<u64> {
    let model = k1::urdf_k1(1e-3)?;
    let map = k1::k1_map(&model);
    let state = k1::k1_state(&model, &map);
    Some(rollout(&model, state, steps, |s| {
        k1::k1_ctrl(&map, script, s)
    }))
}

/// Recorded on 03fc682 (the lane's base). See the module docs before
/// touching these.
const PILE: u64 = 0x0c15_d37d_d099_a96b;
const K1_STANCE: u64 = 0xc27d_9d0a_ee11_6de9;
const K1_STEP: u64 = 0x0367_bbb4_2182_9512;

#[test]
fn contact_speed_pile_is_bit_exact() {
    let (model, state) = pile();
    let got = rollout(&model, state, 1500, |_| {});
    assert_eq!(got, PILE, "pile moved: got {got:#018x}");
}

#[test]
fn contact_speed_k1_is_bit_exact() {
    let (Some(stance), Some(step)) = (k1_rollout("stance", 400), k1_rollout("step", 1500)) else {
        eprintln!(
            "contact_speed_k1_is_bit_exact: K1 assets not found under {} — skipped",
            k1::k1_dir().display()
        );
        return;
    };
    assert_eq!(
        (stance, step),
        (K1_STANCE, K1_STEP),
        "K1 moved: got stance {stance:#018x}, step {step:#018x}"
    );
}
