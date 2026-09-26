//! A free body's linear velocity is stored in the body frame. When the body
//! spins, that vector turns; it must not grow. Explicit Euler on the
//! `−ω × v` term grows it by `(|ω| dt)² / 2` per step, which for a wheel is
//! a runaway. These pin the exact turn.

use phyz::Simulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3, quat_exp};
use phyz_model::{Geometry, ModelBuilder};

const R: f64 = 0.027;
const M: f64 = 0.06;

fn wheel(gravity: bool) -> phyz_model::Model {
    let i = 0.4 * M * R * R;
    let g = if gravity { -GRAVITY } else { 0.0 };
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, g))
        .dt(1e-3)
        .add_free_body(
            "wheel",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(M, Vec3::zeros(), Mat3::identity() * i),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: R });
    model
}

/// World-frame linear velocity of the free body at joint 0.
fn world_speed(s: &phyz_model::State) -> Vec3 {
    let rot = quat_exp(&Vec3::new(s.q[0], s.q[1], s.q[2]));
    rot.rotate(Vec3::new(s.v[3], s.v[4], s.v[5]))
}

#[test]
fn a_spinning_body_in_free_space_keeps_its_velocity() {
    let model = wheel(false);
    let mut s = model.default_state();
    // 74 rad/s about y, 2 m/s along x: a skateboard wheel at walking pace.
    s.v[1] = 2.0 / R;
    s.v[3] = 2.0;
    let sim = Simulator::new();
    for _ in 0..1000 {
        sim.step(&model, &mut s);
    }
    let v = world_speed(&s);
    assert!(
        (v - Vec3::new(2.0, 0.0, 0.0)).norm() < 1e-9,
        "after 1 s of spinning at 74 rad/s the world velocity is {v:?}, not (2, 0, 0): \
         the body-frame Coriolis turn is being integrated to first order"
    );
    assert!((s.q[3] - 2.0).abs() < 1e-9, "travelled {} m, not 2", s.q[3]);
}

#[test]
fn a_wheel_rolling_on_the_plane_keeps_its_speed() {
    let model = wheel(true);
    let mut s = model.default_state();
    s.q[5] = R;
    // Rolling: the contact point `v + ω × (−R ẑ)` is at rest, so ω_y = +v/R.
    s.v[1] = 2.0 / R;
    s.v[3] = 2.0;
    let material = phyz_contact::ContactMaterial {
        friction: 0.8,
        restitution: 0.0,
        ..Default::default()
    };
    let sim = Simulator::new();
    for _ in 0..1000 {
        sim.step_with_contacts(&model, &mut s, 0.0, &material);
    }
    let v = world_speed(&s);
    assert!(
        (v.norm() - 2.0).abs() < 0.02,
        "a wheel rolling at 2 m/s is doing {:.3} m/s a second later ({v:?})",
        v.norm()
    );
    assert!(
        v.z.abs() < 0.01 && (s.q[5] - R).abs() < 1e-3,
        "the wheel left the floor: {:?}, z = {}",
        v,
        s.q[5]
    );
}
