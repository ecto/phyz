//! The GPU contact solve must stay the CPU contact solve.
//!
//! `phyz-contact` states the contact model once, in Rust. The GPU runs it in
//! WGSL, and WGSL cannot share Rust code — so "one model instantiated twice"
//! is not something the compiler can enforce here. This file is what enforces
//! it instead: the shader is checked against the Rust solver running under the
//! *same documented restriction*, so a change to one that is not mirrored in
//! the other fails here rather than silently producing a second contact model.
//!
//! That is the failure this whole effort exists to prevent. phyz has twice
//! shipped two implementations of one idea that drifted apart, and both times
//! the symptom was a confident wrong number rather than a crash.

use phyz::Simulator;
use phyz_contact::material::ContactMaterial;
use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::{BodyContactGains, GroundContactParams};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

const DT: f64 = 0.001;

/// The two crates carry the same constants by duplication, because `phyz-gpu`
/// does not depend on `phyz-contact`. Duplication is fine; *silent* duplication
/// is not.
#[test]
fn the_duplicated_constants_still_agree() {
    assert_eq!(
        phyz_gpu::DEFAULT_CONTACT_SWEEPS,
        phyz_contact::GPU_SWEEPS,
        "the GPU's sweep budget and the CPU reference's must match, or \
         `ContactSolverConfig::gpu_equivalent()` stops being a reference for it"
    );

    // `solref_erp_from` is a hand copy of `SolRef::error_reduction`.
    for dt in [1e-4, 1e-3, 2e-3, 1e-2] {
        let solref = phyz_contact::SolRef::default();
        let gpu = GroundContactParams::solref_erp_from(solref.timeconst, solref.dampratio, dt);
        let cpu = solref.error_reduction(dt);
        assert!(
            (gpu - cpu).abs() < 1e-15,
            "solref error reduction diverged at dt={dt}: gpu {gpu} vs cpu {cpu}"
        );
    }
}

fn free_box(pos: Vec3, half: Vec3, mass: f64) -> Model {
    let c = mass / 3.0;
    let inertia = Mat3::from_diagonal(&Vec3::new(
        c * (half.y * half.y + half.z * half.z),
        c * (half.x * half.x + half.z * half.z),
        c * (half.x * half.x + half.y * half.y),
    ));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::from_translation(pos),
            SpatialInertia::new(mass, Vec3::zeros(), inertia),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    model
}

fn body_pos(state: &State) -> Vec3 {
    let q = state.q.as_slice();
    Vec3::new(q[3], q[4], q[5])
}

/// Roll a scenario on both engines and return the largest position gap over
/// any *finite* frame, plus whether either engine went non-finite.
///
/// The finiteness check is not defensive noise. `f64::max` returns the other
/// operand when one side is NaN, so a peak-gap reduction skips non-finite
/// frames entirely and reports a blown-up run as an excellent one — the exact
/// trap ipse's GPU probes documented.
fn divergence(model: &Model, init: impl Fn(&mut State), steps: usize, friction: f64) -> f64 {
    let mut cpu_state = model.default_state();
    init(&mut cpu_state);
    let gpu_state = cpu_state.clone();

    let material = ContactMaterial {
        friction,
        ..ContactMaterial::default()
    };
    let sim = Simulator::new();
    let mut cpu = Vec::with_capacity(steps);
    for _ in 0..steps {
        sim.step_with_contacts(model, &mut cpu_state, 0.0, &material);
        cpu.push(body_pos(&cpu_state));
    }

    let Ok(mut gpu) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return 0.0;
    };
    let gains = BodyContactGains::uniform_frequency(&gpu.model, 50.0, 1.0);
    gpu.enable_contact_impulse(0.0, friction, &gains, &[], None)
        .expect("impulse contact");
    gpu.load_states(&[gpu_state]);

    let mut peak = 0.0f64;
    for (k, cp) in cpu.iter().enumerate() {
        gpu.step();
        let g = body_pos(&gpu.readback_states()[0]);
        let d = (g - cp).norm();
        assert!(
            d.is_finite(),
            "engine went non-finite at step {k}: cpu {cp:?} gpu {g:?}"
        );
        if d > peak {
            peak = d;
        }
    }
    peak
}

/// A box dropped flat: the narrowest statement of normal-direction agreement.
///
/// Penalty contact could never pass this. It rests `mg/k` deep by
/// construction, which at these gains is about 4 mm — forty times the tolerance
/// here. Passing it is the evidence that the GPU is solving the constraint
/// rather than approximating it with a spring.
#[test]
fn a_dropped_box_lands_where_the_cpu_puts_it() {
    let model = free_box(Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.1, 0.1, 0.1), 1.0);
    let peak = divergence(&model, |_| {}, 1500, 0.8);
    assert!(
        peak < 5e-3,
        "GPU impulse contact diverged from the CPU solve by {peak:.6} m"
    );
}

/// Sliding, where the friction model is what decides the answer.
///
/// The old GPU friction was `mu*f_n` regularized by slip SPEED, which creeps
/// below the regularization threshold instead of sticking; the CPU has real
/// stiction as the interior of a second-order cone. A creeping contact fails
/// this by walking away from the CPU's resting place.
#[test]
fn a_sliding_box_stops_where_the_cpu_stops_it() {
    let model = free_box(Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.1, 0.1, 0.1), 1.0);
    let peak = divergence(
        &model,
        |s| {
            s.v.as_mut_slice()[3] = 2.0;
        },
        1500,
        0.8,
    );
    assert!(peak < 5e-2, "sliding diverged by {peak:.6} m");
}

/// A light body under gains sized for a heavier one is the configuration that
/// produced the wheel blowup: an explicit tangential damper of slope
/// `mu*f_n/SLIP_EPS` against a 0.1 kg wheel reversed its spin every step and
/// reached NaN in twenty. An impulse solve cannot do that — a dissipative
/// impulse brings relative velocity to zero and no further — so this test is
/// really asserting that property, and `divergence` asserts finiteness at
/// every step rather than only at the end.
#[test]
fn a_light_body_does_not_explode() {
    let model = free_box(Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.1, 0.1, 0.1), 0.1);
    let peak = divergence(
        &model,
        |s| {
            s.v.as_mut_slice()[3] = 2.0;
            s.v.as_mut_slice()[0] = 3.0;
        },
        1500,
        0.8,
    );
    assert!(peak.is_finite(), "light body went non-finite");
    assert!(peak < 1e-1, "light body diverged by {peak:.6} m");
}

/// A body at rest must rest ON the surface, not a `margin` above it.
///
/// Contacts are detected within `ContactMaterial::margin` (1 mm by default)
/// and the impedance tapers to zero across that band, so a separated contact
/// inside it should carry no load. The CPU gets that from the impedance
/// regularizer `R = (1-d)/d * A_nn` on the normal row: as `d -> 0` the row
/// goes infinitely soft. The GPU used the impedance in the bias only, so its
/// normal row stayed rigid for a separated contact and stopped the body at
/// the OUTER edge of the margin — measured 0.98 mm high on a plank resting
/// flat, and it did not shrink with more sweeps, because it was not a
/// convergence error. A millimetre of standoff is a different stance, which
/// is what the K1 pre-tip regime is made of.
#[test]
fn a_resting_box_does_not_float_on_the_margin() {
    let half = Vec3::new(0.1, 0.1, 0.1);
    let model = free_box(Vec3::new(0.0, 0.0, half.z + 0.002), half, 1.0);
    let material = ContactMaterial::default();
    let sim = Simulator::new();
    let mut cpu_state = model.default_state();
    for _ in 0..2000 {
        sim.step_with_contacts(&model, &mut cpu_state, 0.0, &material);
    }
    let Ok(mut gpu) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let gains = BodyContactGains::uniform_frequency(&gpu.model, 50.0, 1.0);
    gpu.enable_contact_impulse(0.0, material.friction, &gains, &[], None)
        .expect("impulse contact");
    gpu.load_states(&[model.default_state()]);
    for _ in 0..2000 {
        gpu.step();
    }
    let gz = body_pos(&gpu.readback_states()[0]).z;
    let cz = body_pos(&cpu_state).z;
    assert!(
        (gz - cz).abs() < 3e-4,
        "GPU rests {:.6} m from the CPU's resting height (gpu {gz:.6}, cpu {cz:.6}); \
         a gap near the 1 mm contact margin means the normal row is rigid where \
         the impedance says it should be soft",
        gz - cz
    );
}
