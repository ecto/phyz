//! A body collides as the shape it *is*, not as its first collision box.
//!
//! The contact pass used to pack `Body::collisions[0]` and drop the rest. For
//! a single-shape link that is invisible; for a convex-decomposed one it
//! deletes most of the body. Measured on ipse's skateboard rig (ecto/phyz#82):
//! `skate_deck_tail` carries 18 boxes, and box 0 sat 22.2 mm in the air while
//! the set's true lowest point was 0.9 mm *into* the ground — so the tail-tip
//! ground contact the whole pre-tip scenario rested on did not exist on
//! device, while the CPU referee had it at exactly that depth.
//!
//! These tests fail loudly on that: the second box is the load-bearing one,
//! and a body whose only support is its second box floats or falls through
//! without the fix.

use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::{BodyContactGains, BodyPlane};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, Heightfield, Joint, Model, ModelBuilder};

const OMEGA: f64 = 1.0 / 0.02;

fn gains(model: &Model) -> Vec<BodyContactGains> {
    BodyContactGains::uniform_frequency(model, OMEGA, 1.0)
}

fn inst(pos: Vec3, rot: Mat3, half: Vec3) -> GeomInstance {
    GeomInstance {
        name: None,
        origin: SpatialTransform::new(rot, pos),
        geometry: Geometry::Box { half_extents: half },
    }
}

/// A free body at `start` carrying an explicit collision set.
fn free_body(start: Vec3, collisions: Vec<GeomInstance>) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::from_translation(start),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.02),
        )
        .build();
    model.bodies[0].collisions = collisions;
    model
}

fn world_pos(start: Vec3, q: &[f64]) -> Vec3 {
    start + Vec3::new(q[3], q[4], q[5])
}

/// Roll a model on both engines over flat ground and return the resting
/// positions, or `None` when there is no adapter.
fn settle(
    model: &Model,
    start: Vec3,
    steps: usize,
    hf: Option<&Heightfield>,
) -> Option<(Vec3, Vec3)> {
    let Ok(mut gpu) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    gpu.enable_contact_terrain(0.0, 0.8, &gains(model), &[], hf)
        .unwrap();
    let st = model.default_state();
    gpu.load_states(std::slice::from_ref(&st));

    let cpu_sim = phyz::sim::Simulator::new();
    let material = phyz::contact::ContactMaterial {
        friction: 0.8,
        ..phyz::contact::ContactMaterial::default()
    };
    let mut cpu_state = st;

    for _ in 0..steps {
        gpu.step();
        match hf {
            Some(h) => cpu_sim.step_with_contacts_heightfield(model, &mut cpu_state, h, &material),
            None => cpu_sim.step_with_contacts(model, &mut cpu_state, 0.0, &material),
        };
    }

    let gq = gpu.readback_states()[0].q.clone();
    Some((world_pos(start, gq.as_slice()), cpu_state.body_xform[0].pos))
}

/// The lowest box is the *second* one, so a device that reads `collisions[0]`
/// has nothing under the body and it falls straight through the floor.
///
/// This is the ipse tail case in miniature: box 0 is up in the air, box 1 is
/// the one on the ground.
#[test]
fn the_second_collision_box_is_the_one_that_lands() {
    let half = Vec3::new(0.1, 0.1, 0.05);
    let model = free_body(
        Vec3::new(0.0, 0.0, 0.6),
        vec![
            // Box 0: high on the body, never touches anything.
            inst(Vec3::new(0.0, 0.0, 0.4), Mat3::identity(), half),
            // Box 1: the body's actual foot.
            inst(Vec3::new(0.0, 0.0, -0.2), Mat3::identity(), half),
        ],
    );
    let Some((gpu, cpu)) = settle(&model, Vec3::new(0.0, 0.0, 0.6), 2000, None) else {
        return;
    };

    // The body origin must come to rest 0.25 m up: 0.2 down to box 1's centre
    // plus its 0.05 half-height. Reading box 0 instead would put it at
    // −0.35 m — through the floor.
    assert!(
        (cpu.z - 0.25).abs() < 0.02,
        "the CPU referee is not resting on box 1: z = {:.4}",
        cpu.z
    );
    assert!(
        (gpu.z - cpu.z).abs() < 0.02,
        "device rests at z = {:.4} where the CPU rests at {:.4}: \
         the second collision box is invisible to the contact pass",
        gpu.z,
        cpu.z
    );
}

/// A multi-box body on a tilted surface: rotation is where a dropped instance
/// stops being a height error and becomes a torque error.
///
/// A 5 deg uniform slope, and an L of two boxes whose *lower* one is offset in
/// x — so the body's support polygon, and therefore where it stops sliding
/// and how it settles, depends on an instance the old pass never packed.
#[test]
fn a_multi_box_body_rests_on_a_tilted_plane() {
    // Uniform 5 deg slope, as a heightfield: the CPU has an exact referee for
    // it (`find_heightfield_contacts_model`) and the GPU reads the same field.
    let n = 81;
    let cell = 0.1;
    let slope = (5.0f64).to_radians().tan();
    let mut hf = Heightfield::new(Vec3::new(-4.0, -4.0, 0.0), cell, n, n);
    for iy in 0..n {
        for ix in 0..n {
            let x = -4.0 + cell * ix as f64;
            hf.heights[iy * n + ix] = (slope * x) as f32;
        }
    }

    let half = Vec3::new(0.08, 0.08, 0.04);
    let model = free_body(
        Vec3::new(0.0, 0.0, 0.5),
        vec![
            // Upper arm of the L, offset in +x.
            inst(Vec3::new(0.16, 0.0, 0.08), Mat3::identity(), half),
            // Lower arm — the one that actually meets the slope.
            inst(Vec3::new(0.0, 0.0, 0.0), Mat3::identity(), half),
        ],
    );

    let Some((gpu, cpu)) = settle(&model, Vec3::new(0.0, 0.0, 0.5), 3000, Some(&hf)) else {
        return;
    };
    assert!(
        gpu.z.is_finite() && cpu.z.is_finite(),
        "a path produced NaN: gpu {gpu:?} cpu {cpu:?}"
    );
    let d = (gpu - cpu).norm();
    assert!(
        d < 0.06,
        "engines disagree on the resting pose on a slope: gpu {gpu:?} vs cpu {cpu:?} (|Δ| = {d:.4})"
    );
    // Both are ON the slope, not through it or floating above it.
    for (name, p) in [("gpu", gpu), ("cpu", cpu)] {
        let gap = p.z - hf.height(p.x, p.y);
        assert!(
            gap > 0.0 && gap < 0.12,
            "{name} body floats or sinks on the slope: {gap:.3} m above the surface"
        );
    }
}

// ── Tilted body-attached faces (the kicktail) ──

/// Deck body + a rider box, so a face can be hung off the deck at an angle.
fn deck_and_rider(deck_half: Vec3, rider_half: Vec3) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "deck",
            -1,
            Joint::fixed(SpatialTransform::identity()),
            SpatialInertia::new(2.0, Vec3::zeros(), Mat3::identity() * 0.05),
        )
        .add_body(
            "rider",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.02),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: deck_half,
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: rider_half,
    });
    model
}

/// A face at 15 deg holds a rider that a flat face at the same offset misses.
///
/// This is the second half of #82: the pre-tip stance puts the loaded foot
/// beyond the flex hinge, on a kicktail rising 15 deg out of the deck plane.
/// One untilted plane left that foot 6 to 25 mm above the only surface the
/// device knew about, and the rider fell. A face with a `tilt` catches it.
#[test]
fn a_tilted_face_holds_what_a_flat_face_drops() {
    let deck_half = Vec3::new(0.4, 0.1, 0.01);
    let rider_half = Vec3::new(0.05, 0.05, 0.05);
    let model = deck_and_rider(deck_half, rider_half);

    let kick = (15.0f64).to_radians();
    // `tilt` is body -> face, so the face's own +Z leans back over the deck by
    // `kick`; the ramp rises toward +x.
    // Row 2 of a body -> face rotation IS the face normal in body
    // coordinates, so `(-sin, 0, cos)` leans the normal back toward -x and the
    // ramp therefore rises toward +x.
    let tilt = Mat3::new(
        kick.cos(),
        0.0,
        kick.sin(),
        0.0,
        1.0,
        0.0,
        -kick.sin(),
        0.0,
        kick.cos(),
    );

    // The rider sits out at x = 0.3 on the ramp. Face origin at the deck's
    // top surface, pivoting about the body origin.
    let x = 0.3;
    let ramp_top = deck_half.z + x * kick.tan();

    let tilted = BodyPlane {
        body: 0,
        offset: deck_half.z,
        max_depth: 0.05,
        half_x: 0.5,
        half_y: deck_half.y,
        exclude: vec![],
        tilt,
        center: Vec3::zeros(),
    };
    let flat = BodyPlane::flat(0, deck_half.z, 0.05, 0.5, deck_half.y);

    let mut rested = Vec::new();
    for (label, plane) in [("tilted", &tilted), ("flat", &flat)] {
        let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        // Ground far below, so the only thing that can hold the rider is the
        // face.
        sim.enable_ground_contact_with_plane(
            -10.0,
            0.8,
            &gains(&model),
            std::slice::from_ref(plane),
        )
        .expect("contact");

        let mut st = model.default_state();
        // Rider free joint: q = [exp-coords(3), pos(3)], just above the ramp.
        let ro = model.q_offsets[model.bodies[1].joint_idx];
        st.q.as_mut_slice()[ro + 3] = x;
        st.q.as_mut_slice()[ro + 5] = ramp_top + rider_half.z + 0.002;
        sim.load_states(std::slice::from_ref(&st));
        for _ in 0..800 {
            sim.step();
        }
        let z = sim.readback_states()[0].q.as_slice()
            [model.q_offsets[model.bodies[1].joint_idx] + 5];
        assert!(z.is_finite(), "{label} face produced NaN");
        rested.push((label, z));
    }

    let tilted_z = rested[0].1;
    let flat_z = rested[1].1;
    assert!(
        (tilted_z - (ramp_top + rider_half.z)).abs() < 0.02,
        "the tilted face did not hold the rider: rested at z = {tilted_z:.4}, \
         ramp surface + half-height is {:.4}",
        ramp_top + rider_half.z
    );
    assert!(
        flat_z < tilted_z - 0.02,
        "the flat face was expected to MISS this rider (it is {:.1} mm above \
         the flat top) but held it at z = {flat_z:.4}",
        (x * kick.tan()) * 1000.0
    );
}
