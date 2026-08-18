//! The CUDA C kernels against the f64 CPU reference — and against the WGSL
//! kernels where a wgpu adapter exists.
//!
//! Every test is written once, generic over the kernel backend, and run:
//!
//! * on [`HostBatchSimulator`] (feature `cuda-host`): the CUDA C source
//!   compiled as host C++, so the port is checked on any machine;
//! * on [`CudaBatchSimulator`] (feature `cuda`): the real device, skipped
//!   with a message when there is no CUDA driver.
//!
//! The CPU is the reference for both, exactly as it is for the wgpu path.
//! Tolerances follow the existing wgpu-vs-CPU tests: f32 kernels against f64
//! dynamics, so 1e-3-ish after a step and looser along a chaotic chain.
//! Kernel-vs-kernel (CUDA C vs WGSL) is held far tighter — both are f32
//! doing the same arithmetic, so anything past ~1e-4 is a port bug.

#![cfg(any(feature = "cuda", feature = "cuda-host"))]

use phyz_gpu::cuda::{BatchSim, KernelBackend};
use phyz_gpu::{BodyContactGains, GpuBatchSimulator, PdDof};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Joint, Model, ModelBuilder, State};
use phyz_rigid::{aba, semi_implicit_euler};

// ── Models (shared with the wgpu tests' shapes) ────────────────────────────

fn rod(m: f64, half_len: f64) -> SpatialInertia {
    let len = 2.0 * half_len;
    let i = m * len * len / 12.0;
    SpatialInertia::new(
        m,
        Vec3::new(0.0, -half_len, 0.0),
        Mat3::from_diagonal(&Vec3::new(i, 0.0, i)),
    )
}

fn iso(mass: f64, i: f64) -> SpatialInertia {
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

fn double_pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body("upper", -1, SpatialTransform::identity(), rod(1.0, 0.5))
        .add_revolute_body(
            "lower",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            rod(1.0, 0.5),
        )
        .build()
}

fn arm_6dof() -> Model {
    let length = 0.3;
    let inertia = SpatialInertia::new(
        0.5,
        Vec3::new(0.0, 0.0, -length / 2.0),
        Mat3::from_diagonal(&Vec3::new(0.00375, 0.00375, 0.001)),
    );
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_revolute_body("j1", -1, SpatialTransform::identity(), inertia);
    for k in 1i32..6 {
        b = b.add_revolute_body(
            &format!("j{}", k + 1),
            k - 1,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
            inertia,
        );
    }
    b.build()
}

fn free_base_with_limb() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "torso",
            -1,
            Joint::free(SpatialTransform::identity()),
            iso(5.0, 0.3),
        )
        .add_revolute_body(
            "limb",
            0,
            SpatialTransform::from_translation(Vec3::new(0.2, 0.0, 0.0)),
            iso(1.0, 0.05),
        )
        .build()
}

fn ball_joint_model() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "ball",
            -1,
            Joint::spherical(SpatialTransform::identity()),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, 0.0, -0.3),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.02)),
            ),
        )
        .build()
}

fn prismatic_model() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_prismatic_body(
            "slider",
            -1,
            SpatialTransform::identity(),
            Vec3::new(0.0, 0.0, 1.0),
            iso(1.0, 0.1),
        )
        .build()
}

fn free_sphere(mass: f64, radius: f64) -> Model {
    let mut body = phyz_model::Body::new("body", iso(mass, 0.4 * mass * radius * radius), -1, 0);
    body.geometry = Some(Geometry::Sphere { radius });
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_free_body_with_geometry(
            "body",
            -1,
            SpatialTransform::identity(),
            iso(mass, 0.4 * mass * radius * radius),
            body,
        )
        .build()
}

// ── Reference ─────────────────────────────────────────────────────────────

fn cpu_step(model: &Model, s: &mut State) {
    let qdd = aba(model, s);
    semi_implicit_euler(model, s, qdd.as_slice(), model.dt);
}

fn assert_state_close(label: &str, got: &State, want: &State, tol: f64) {
    for j in 0..want.q.len() {
        let d = (got.q[j] - want.q[j]).abs();
        assert!(
            d < tol,
            "{label}: q[{j}] got={:.9} want={:.9} diff={d:.2e}",
            got.q[j],
            want.q[j]
        );
    }
    for j in 0..want.v.len() {
        let d = (got.v[j] - want.v[j]).abs();
        assert!(
            d < tol,
            "{label}: v[{j}] got={:.9} want={:.9} diff={d:.2e}",
            got.v[j],
            want.v[j]
        );
    }
}

// ── The suite, generic over the backend ───────────────────────────────────

fn one_step<B: KernelBackend>(
    mk: impl Fn(Model, usize) -> BatchSim<B>,
    model: &Model,
    s: &State,
    tol: f64,
    label: &str,
) {
    let mut sim = mk(model.clone(), 1);
    sim.load_states(std::slice::from_ref(s));
    sim.step();
    let out = sim.readback_states();
    let mut cpu = s.clone();
    cpu_step(model, &mut cpu);
    assert_state_close(label, &out[0], &cpu, tol);
}

fn suite_single_steps<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let m = double_pendulum();
    let mut s = m.default_state();
    s.q[0] = 0.3;
    s.q[1] = 0.2;
    s.v[0] = 0.1;
    s.v[1] = -0.1;
    one_step(&mk, &m, &s, 5e-3, "double pendulum");

    let m = arm_6dof();
    let mut s = m.default_state();
    for (j, q) in [0.1, -0.2, 0.3, -0.1, 0.2, -0.3].into_iter().enumerate() {
        s.q[j] = q;
    }
    one_step(&mk, &m, &s, 1e-2, "6-dof arm");

    let m = prismatic_model();
    let mut s = m.default_state();
    s.q[0] = 0.2;
    s.v[0] = -0.5;
    one_step(&mk, &m, &s, 1e-4, "prismatic");

    let m = ball_joint_model();
    let mut s = m.default_state();
    s.q[0] = 0.3;
    s.q[1] = -0.2;
    s.q[2] = 0.1;
    s.v[0] = 0.5;
    s.v[1] = -0.4;
    s.v[2] = 0.3;
    one_step(&mk, &m, &s, 1e-3, "ball joint");

    let m = free_base_with_limb();
    let mut s = m.default_state();
    s.q[0] = 0.2;
    s.q[1] = -0.1;
    s.q[2] = 0.3;
    s.q[3] = 0.1;
    s.q[4] = 0.2;
    s.q[5] = 0.5;
    s.q[6] = 0.4;
    s.v[0] = 0.3;
    s.v[3] = -0.2;
    s.v[6] = 1.0;
    one_step(&mk, &m, &s, 1e-3, "free base with limb");
}

fn suite_trajectory<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let model = double_pendulum();
    let mut sim = mk(model.clone(), 3);
    let mut cpu: Vec<State> = (0..3)
        .map(|i| {
            let mut s = model.default_state();
            s.q[0] = 0.4 + 0.2 * i as f64;
            s.q[1] = -0.3;
            s
        })
        .collect();
    sim.load_states(&cpu);
    for _ in 0..200 {
        sim.step();
        for s in cpu.iter_mut() {
            cpu_step(&model, s);
        }
    }
    let out = sim.readback_states();
    for (w, (g, c)) in out.iter().zip(&cpu).enumerate() {
        assert_state_close(&format!("trajectory world {w}"), g, c, 5e-3);
    }
    // Worlds started apart must stay apart: no cross-world aliasing.
    assert!((out[0].q[0] - out[1].q[0]).abs() > 1e-3);
}

fn suite_pd<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let inertia = SpatialInertia::new(1.0, Vec3::new(0.0, 0.0, -0.5), Mat3::identity() * 0.1);
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.002)
        .add_revolute_body("link", -1, SpatialTransform::identity(), inertia)
        .build();
    let (kp, kd, max_force, target) = (30.0, 2.0, 25.0, 0.8);

    let mut sim = mk(model.clone(), 4);
    sim.enable_pd_control(&[PdDof {
        q_index: 0,
        v_index: 0,
        kp,
        kd,
        max_force,
    }])
    .unwrap();
    sim.set_position_targets(&vec![vec![target]; 4]).unwrap();
    let mut init = model.default_state();
    init.q[0] = 0.3;
    sim.load_states(&vec![init.clone(); 4]);

    let mut cpu = init.clone();
    for _ in 0..100 {
        let tau = (kp * (target - cpu.q[0]) - kd * cpu.v[0]).clamp(-max_force, max_force);
        cpu.ctrl[0] = tau;
        cpu_step(&model, &mut cpu);
    }
    for _ in 0..100 {
        sim.step();
    }
    let out = sim.readback_states();
    for (w, s) in out.iter().enumerate() {
        assert!(
            (s.q[0] - cpu.q[0]).abs() < 1e-3,
            "pd world {w}: kernel q {} vs cpu q {}",
            s.q[0],
            cpu.q[0]
        );
    }
    assert!(
        (out[0].q[0] - 0.3).abs() > 0.05,
        "servo did not move the joint"
    );
}

fn suite_contact<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let (mass, radius) = (1.0, 0.1);
    let model = free_sphere(mass, radius);
    let mut sim = mk(model.clone(), 2);
    let (omega, zeta) = (200.0, 1.0);
    let k = mass * omega * omega;
    let d = 2.0 * zeta * mass * omega;
    assert_eq!(sim.enable_ground_contact(0.0, k, d, 0.5).unwrap(), 1);

    let mut s = model.default_state();
    s.q[5] = 0.5;
    sim.load_states(&[s.clone(), s]);
    sim.step();
    let c = sim.readback_contacts().unwrap();
    assert!(!c[0][0].touching, "airborne body reports contact");

    for _ in 0..2000 {
        sim.step();
    }
    let out = sim.readback_states();
    let z = out[0].q[5];
    assert!(
        (z - radius).abs() < 0.02,
        "sphere should rest one radius above ground, got z = {z}"
    );
    let c = sim.readback_contacts().unwrap();
    let c = &c[1][0];
    assert!(c.touching);
    let weight = mass * GRAVITY;
    assert!(
        (c.force.z - weight).abs() < 0.15 * weight,
        "resting normal force ~ weight ({weight:.2}), got {:.2}",
        c.force.z
    );
    assert!(c.point.z.abs() < 1e-6);

    // Per-body gains path, and the no-geometry error path.
    let mut sim2 = mk(model.clone(), 1);
    let gains = BodyContactGains::uniform_frequency(&model, omega, zeta);
    assert_eq!(
        sim2.enable_ground_contact_per_body(0.0, 0.5, &gains)
            .unwrap(),
        1
    );
    let bare = double_pendulum();
    let mut sim3 = mk(bare, 1);
    assert!(sim3.enable_ground_contact(0.0, 1.0, 1.0, 0.5).is_err());
    assert!(sim3.readback_contacts().is_err());
}

fn suite_ant<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let Ok(loader) = phyz_mjcf::MjcfLoader::from_file(path) else {
        eprintln!("skipping ant: {path} not found");
        return;
    };
    let model = loader.build_model();
    let mut sim = mk(model.clone(), 8);
    let mut s = model.default_state();
    s.q[5] = 0.75;
    sim.load_states(&vec![s.clone(); 8]);
    let mut cpu = s;
    for _ in 0..100 {
        sim.step();
        cpu_step(&model, &mut cpu);
    }
    let out = sim.readback_states();
    assert!(
        out.iter()
            .all(|s| s.q.as_slice().iter().all(|x| x.is_finite())),
        "ant diverged"
    );
    assert!(
        out[0].q[5] < 0.75,
        "ant torso should fall; z = {}",
        out[0].q[5]
    );
    assert_state_close("ant 100 steps", &out[0], &cpu, 5e-3);
}

/// CUDA C against WGSL: same f32 arithmetic on both sides, so the two must
/// agree far more tightly than either agrees with the f64 CPU.
fn suite_vs_wgpu<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let model = free_base_with_limb();
    let Ok(wg) = GpuBatchSimulator::new(model.clone(), 4) else {
        eprintln!("skipping kernel-vs-kernel: no wgpu adapter");
        return;
    };
    let mut cu = mk(model.clone(), 4);
    let states: Vec<State> = (0..4)
        .map(|i| {
            let mut s = model.default_state();
            s.q[6] = 0.3 + 0.1 * i as f64;
            s.q[5] = 1.0;
            s.v[0] = 0.5;
            s.v[4] = 0.2 * i as f64;
            s
        })
        .collect();
    wg.load_states(&states);
    cu.load_states(&states);
    let ctrl = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]; 4];
    wg.set_controls(&ctrl);
    cu.set_controls(&ctrl);
    for _ in 0..300 {
        wg.step();
        cu.step();
    }
    let a = wg.readback_states();
    let b = cu.readback_states();
    for (w, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_state_close(&format!("cuda-c vs wgsl world {w}"), y, x, 2e-4);
    }

    // Contact readback, kernel vs kernel.
    let (mass, radius) = (1.0, 0.1);
    let model = free_sphere(mass, radius);
    let Ok(mut wg) = GpuBatchSimulator::new(model.clone(), 1) else {
        return;
    };
    let mut cu = mk(model.clone(), 1);
    let (k, d) = (mass * 200.0 * 200.0, 2.0 * mass * 200.0);
    wg.enable_ground_contact(0.0, k, d, 0.5).unwrap();
    cu.enable_ground_contact(0.0, k, d, 0.5).unwrap();
    let mut s = model.default_state();
    s.q[5] = 0.3;
    wg.load_states(std::slice::from_ref(&s));
    cu.load_states(std::slice::from_ref(&s));
    for _ in 0..1500 {
        wg.step();
        cu.step();
    }
    let a = &wg.readback_contacts().unwrap()[0][0];
    let b = &cu.readback_contacts().unwrap()[0][0];
    assert_eq!(a.touching, b.touching);
    assert!(
        (a.penetration - b.penetration).abs() < 1e-5,
        "{a:?} vs {b:?}"
    );
    assert!((a.force.z - b.force.z).abs() < 1e-2, "{a:?} vs {b:?}");
    assert_state_close(
        "contact rollout cuda-c vs wgsl",
        &cu.readback_states()[0],
        &wg.readback_states()[0],
        1e-4,
    );
}

fn run_all<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B> + Copy) {
    suite_single_steps(mk);
    suite_trajectory(mk);
    suite_pd(mk);
    suite_contact(mk);
    suite_ant(mk);
    suite_vs_wgpu(mk);
}

// ── Host harness ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda-host")]
mod host {
    use super::*;
    use phyz_gpu::HostBatchSimulator;

    fn mk(m: Model, n: usize) -> HostBatchSimulator {
        HostBatchSimulator::new(m, n).unwrap()
    }

    #[test]
    fn single_steps_match_cpu() {
        suite_single_steps(mk);
    }
    #[test]
    fn trajectory_matches_cpu() {
        suite_trajectory(mk);
    }
    #[test]
    fn pd_servo_matches_cpu() {
        suite_pd(mk);
    }
    #[test]
    fn ground_contact_behaves() {
        suite_contact(mk);
    }
    #[test]
    fn ant_matches_cpu() {
        suite_ant(mk);
    }
    #[test]
    fn matches_wgsl_kernels() {
        suite_vs_wgpu(mk);
    }
    #[test]
    fn rejects_oversized_models() {
        let mut b = ModelBuilder::new();
        for i in 0i32..40 {
            b = b.add_revolute_body(
                &format!("b{i}"),
                i - 1,
                SpatialTransform::identity(),
                iso(1.0, 0.1),
            );
        }
        assert!(HostBatchSimulator::new(b.build(), 1).is_err());
    }
}

// ── Real device ───────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use phyz_gpu::CudaBatchSimulator;

    #[test]
    fn everything_matches_cpu_on_cuda() {
        match CudaBatchSimulator::new(double_pendulum(), 1) {
            Ok(sim) => eprintln!("running on {}", sim.backend().device_name()),
            Err(e) => {
                eprintln!("skipping CUDA tests: {e}");
                return;
            }
        }
        run_all(|m, n| CudaBatchSimulator::new(m, n).unwrap());
    }
}
