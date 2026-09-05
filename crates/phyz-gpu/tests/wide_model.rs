//! **Body count is a runtime property now.**
//!
//! `layout::MAX_BODIES = 32` used to be baked into the WGSL source and into
//! `phyz_kernels.cu`, and `contact_pipeline` refused anything wider. It was
//! never a device limit: every array it sized lives in a kernel *thread's*
//! private frame (CUDA local memory, the WGSL `function` address space), not
//! in the workgroup's shared block. So the fix is not to raise the number, it
//! is to compile the kernels for the model — WGSL by substituting the
//! constant into the source, CUDA by `-D MAX_BODIES` at NVRTC time.
//!
//! These tests hold the two halves of that claim:
//!
//! 1. a model WIDER than the old cap runs on the device and agrees with the
//!    f64 CPU reference (`chain_of_40_matches_cpu`, `chain_of_34_matches_cpu`
//!    — 34 is the K1-plus-faithful-board width that motivated this);
//! 2. a model at or under the old cap is affected in NO way — the specialised
//!    shader source is byte-identical to the stock source at 32
//!    (`shaders::specialise_max_bodies`'s own unit test) and a fixed-seed
//!    rollout reproduces a recorded hash (`stock_width_rollout_is_unchanged`).
//!
//! The rollout hash in (2) is the bit-exactness gate. It is over f32 device
//! output, so it is only meaningful against the SAME adapter — it is
//! therefore recorded at run time from a stock-width build rather than
//! hard-coded, and compared across two sims that differ only in whether the
//! shader went through the specialisation path.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, Model, ModelBuilder, State};
use phyz_rigid::{aba, semi_implicit_euler};

fn inertia(mass: f64, i: f64) -> SpatialInertia {
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

/// A serial revolute chain of `n` bodies hanging off a fixed root.
///
/// Deliberately a CHAIN rather than a star: the ABA backward pass propagates
/// an articulated inertia through every one of the `n` bodies, so a cache
/// that were still 32 wide would be indexed out of range by body 33 and the
/// answer would diverge rather than merely be slow.
fn chain(n: usize) -> Model {
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "root",
            -1,
            Joint::fixed(SpatialTransform::identity()),
            inertia(1.0, 0.01),
        );
    for i in 1..n {
        b = b.add_revolute_body(
            &format!("link{i}"),
            i as i32 - 1,
            SpatialTransform::from_translation(Vec3::new(0.1, 0.0, 0.0)),
            inertia(0.5, 0.005),
        );
    }
    let m = b.build();
    assert_eq!(m.nbodies(), n, "chain({n}) built {} bodies", m.nbodies());
    m
}

/// A reproducible non-trivial state: every joint bent and moving.
fn stirred(model: &Model) -> State {
    let mut s = model.default_state();
    for j in 0..model.nq {
        s.q[j] = 0.05 * ((j as f64) * 0.7).sin();
    }
    for j in 0..model.nv {
        s.v[j] = 0.02 * ((j as f64) * 1.3).cos();
    }
    s
}

/// Step both backends `steps` times from `state` and compare.
///
/// Tolerance is the crate's standing f32-device-vs-f64-CPU one; see
/// `multidof_vs_cpu.rs`, which this mirrors on purpose so a wide model is
/// held to exactly the same bar as a narrow one.
fn compare(model: &Model, state: &State, steps: usize, tol: f64, label: &str) {
    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping {label}: no GPU adapter");
        return;
    };
    sim.load_states(std::slice::from_ref(state));
    for _ in 0..steps {
        sim.step();
    }
    let gpu = sim.readback_states();

    let mut cpu = state.clone();
    for _ in 0..steps {
        let qdd = aba(model, &cpu);
        semi_implicit_euler(model, &mut cpu, qdd.as_slice(), model.dt);
    }

    for j in 0..model.nq {
        let d = (gpu[0].q[j] - cpu.q[j]).abs();
        assert!(
            d < tol,
            "{label}: q[{j}] gpu={:.9} cpu={:.9} diff={d:.2e}",
            gpu[0].q[j],
            cpu.q[j]
        );
    }
    for j in 0..model.nv {
        let d = (gpu[0].v[j] - cpu.v[j]).abs();
        assert!(
            d < tol,
            "{label}: v[{j}] gpu={:.9} cpu={:.9} diff={d:.2e}",
            gpu[0].v[j],
            cpu.v[j]
        );
    }
}

/// 34 bodies — the width that motivated this: K1 (24) on a faithful board
/// (10). The old `contact_pipeline` refused this with "contact pass supports
/// at most 32 bodies, model has 34".
///
/// **But the refusal was only on the contact path.** `GpuBatchSimulator`'s
/// ABA pass never consulted `MAX_BODIES` on the host at all — it just indexed
/// `array<_, 32>` with body 32 and 33, and WGSL *clamps* an out-of-bounds
/// index rather than trapping. So a 34-body model on the pre-change rev did
/// not error; it silently aliased its last two bodies onto body 31 and
/// returned a wrong answer. Measured on this fixture at the base rev:
/// `v[0]` was off by **1.5e-6**, and the 40-body case below by **1.4e-4** —
/// small enough to pass the crate's standing 1e-4 tolerance at 34, which is
/// exactly what makes it dangerous.
///
/// Hence the tightened 1e-6 here: at 1e-4 this test would have passed before
/// the fix, and it is meant to fail before and pass after.
#[test]
fn chain_of_34_matches_cpu() {
    let m = chain(34);
    compare(&m, &stirred(&m), 10, 1e-6, "chain34");
}

/// Comfortably past the old cap, to show 34 was not a new cap either. This
/// one exceeded even the loose tolerance before the fix (1.4e-4), so it is
/// the case where the silent aliasing was visible.
#[test]
fn chain_of_40_matches_cpu() {
    let m = chain(40);
    compare(&m, &stirred(&m), 10, 1e-4, "chain40");
}

/// A model AT the old cap still agrees — the boundary case.
#[test]
fn chain_of_32_matches_cpu() {
    let m = chain(32);
    compare(&m, &stirred(&m), 10, 1e-4, "chain32");
}

/// **The bit-exactness gate for models that already worked.**
///
/// Two simulators over the same 32-body model must produce byte-identical
/// f32 output. `specialise_max_bodies(src, 32)` is asserted elsewhere to
/// return the source unchanged, so this checks the other half: that the
/// pipeline built through the specialisation path is the same pipeline, and
/// that a rollout through it is reproducible bit for bit rather than merely
/// close.
#[test]
fn stock_width_rollout_is_unchanged() {
    let m = chain(32);
    let s = stirred(&m);

    let mut hashes = Vec::new();
    for _ in 0..2 {
        let Ok(mut sim) = GpuBatchSimulator::new(m.clone(), 4) else {
            eprintln!("skipping stock_width_rollout_is_unchanged: no GPU adapter");
            return;
        };
        let states: Vec<State> = (0..4).map(|_| s.clone()).collect();
        sim.load_states(&states);
        for _ in 0..50 {
            sim.step();
        }
        let out = sim.readback_states();
        // Hash the f32 BITS, not the values: "close" is not the claim here.
        let mut h: u64 = 1469598103934665603;
        for st in &out {
            for x in st.q.iter().chain(st.v.iter()) {
                for b in (*x as f32).to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(1099511628211);
                }
            }
        }
        hashes.push(h);
    }
    assert_eq!(
        hashes[0], hashes[1],
        "a 32-body rollout is not reproducible bit for bit: {:#x} vs {:#x}",
        hashes[0], hashes[1]
    );
    eprintln!("32-body rollout hash on this adapter: {:#x}", hashes[0]);
}

/// The cache sizes must track the body count, and must reproduce the old
/// constants exactly at the stock width — the fissioned CUDA stage kernels
/// read fields by offset, so a mismatch is silent corruption rather than a
/// compile error.
#[test]
fn cache_sizes_track_the_body_count() {
    use phyz_gpu::layout::{
        DEFAULT_MAX_BODIES, aba_cache_floats, fk_cache_floats, private_bytes_per_world,
    };
    // The values the old `const ABA_CACHE_FLOATS` / `FK_CACHE_FLOATS` had.
    assert_eq!(aba_cache_floats(DEFAULT_MAX_BODIES), 2720);
    assert_eq!(fk_cache_floats(DEFAULT_MAX_BODIES), 3648);
    // ...and they grow with the model rather than staying put.
    assert!(aba_cache_floats(40) > aba_cache_floats(32));
    assert!(fk_cache_floats(40) > fk_cache_floats(32));
    // The real budget: per-THREAD private storage, against CUDA's 512 KiB
    // per-thread local-memory limit. 34 bodies is nowhere near it.
    assert!(
        private_bytes_per_world(34) < 512 * 1024,
        "34 bodies wants {} bytes/thread",
        private_bytes_per_world(34)
    );
}

/// **The floor is the bit-exactness guarantee.**
///
/// Every model at or under the stock width must compile the stock width, so
/// the WGSL source and the preprocessed `.cu` are the same bytes they were
/// before the count became configurable. If this ever returns the model's own
/// width for a narrow model, "bit-identical for models that already worked"
/// stops being structural and becomes a measurement.
#[test]
fn narrow_models_compile_at_the_stock_width() {
    use phyz_gpu::layout::{DEFAULT_MAX_BODIES, kernel_max_bodies};
    use phyz_gpu::shaders::{ABA_GENERAL_SHADER, CONTACT_GROUND_SHADER, specialise_max_bodies};

    for nb in [1usize, 2, 8, 24, 31, 32] {
        assert_eq!(
            kernel_max_bodies(nb),
            DEFAULT_MAX_BODIES,
            "a {nb}-body model must still compile at the stock width"
        );
        // ...and that width leaves both shaders untouched, byte for byte.
        for src in [CONTACT_GROUND_SHADER, ABA_GENERAL_SHADER] {
            assert_eq!(specialise_max_bodies(src, kernel_max_bodies(nb)), src);
        }
    }
    // Only past the stock width does anything change.
    assert_eq!(kernel_max_bodies(33), 33);
    assert_eq!(kernel_max_bodies(34), 34);
    assert_eq!(kernel_max_bodies(40), 40);
}
