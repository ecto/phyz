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

use phyz_gpu::contact_pipeline::BodyPlane;
use phyz_gpu::cuda::{BatchSim, KernelBackend};
use phyz_gpu::policy_pipeline::{
    KernelRng, ObsOp, PolicySpec, XF_STRIDE, observe_reference, policy_reference,
};
use phyz_gpu::{BodyContactGains, GpuBatchSimulator, PdDof};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{
    GeomInstance, Geometry, Heightfield, Joint, JointType, Model, ModelBuilder, State,
};
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

/// A wgpu adapter worth comparing kernels against: a real GPU (or Metal /
/// Vulkan on a real device). A software rasterizer (llvmpipe over EGL, which
/// is what a headless CUDA pod hands wgpu when it cannot open /dev/dri) is
/// not one — it runs the WGSL, but its LLVM float pipeline is not the CPU's
/// and not the GPU's, and it is the CPU that is the reference here.
fn wgpu_hardware_adapter() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) else {
        return false;
    };
    let info = adapter.get_info();
    let software = matches!(info.device_type, wgpu::DeviceType::Cpu)
        || info.name.to_lowercase().contains("llvmpipe")
        || info.name.to_lowercase().contains("swiftshader");
    if software {
        eprintln!(
            "wgpu adapter is software ({}); kernel-vs-kernel skipped",
            info.name
        );
    }
    !software
}

/// CUDA C against WGSL: same f32 arithmetic on both sides, so the two must
/// agree far more tightly than either agrees with the f64 CPU.
fn suite_vs_wgpu<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    if !wgpu_hardware_adapter() {
        return;
    }
    let model = free_base_with_limb();
    let Ok(mut wg) = GpuBatchSimulator::new(model.clone(), 4) else {
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

// ── Unified contact, kernel vs kernel ─────────────────────────────────────
//
// The wgpu tests (body_plane_contact, heightfield, contact_impulse_parity,
// joint_spring_vs_cpu, box_manifold) hold the WGSL to the CPU referee. Here
// the CUDA C is held to the WGSL on the same scenarios, tightly: both are
// f32 doing the same arithmetic, so anything past ~1e-4 is a port bug.

fn box_inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let (lx, ly, lz) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            mass / 12.0 * (ly * ly + lz * lz),
            mass / 12.0 * (lx * lx + lz * lz),
            mass / 12.0 * (lx * lx + ly * ly),
        )),
    )
}

fn free_box(half: Vec3, mass: f64) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "box",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(mass, half),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    model
}

/// Two free boxes: a deck (with an off-centre collision instance) and a
/// rider, as in the wgpu body-plane tests.
fn deck_and_rider() -> (Model, BodyPlane) {
    let deck_half = Vec3::new(0.4, 0.2, 0.05);
    let rider_half = Vec3::new(0.1, 0.1, 0.1);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "deck",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(4.0, deck_half),
        )
        .add_body(
            "rider",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(1.0, rider_half),
        )
        .build();
    model.bodies[0].collisions = vec![GeomInstance::centered(Geometry::Box {
        half_extents: deck_half,
    })];
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: deck_half,
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: rider_half,
    });
    let plane = BodyPlane {
        body: 0,
        offset: deck_half.z,
        max_depth: 0.05,
        half_x: deck_half.x,
        half_y: deck_half.y,
        exclude: vec![],
    };
    (model, plane)
}

fn sprung_pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "arm",
            -1,
            Joint {
                joint_type: JointType::Revolute,
                parent_to_joint: SpatialTransform::identity(),
                axis: Vec3::new(0.0, 1.0, 0.0),
                stiffness: 3.0,
                spring_ref: 0.4,
                damping: 0.4,
                armature: 0.02,
                name: "hinge".into(),
                ..Joint::default()
            },
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, 0.0, -0.3),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.02)),
            ),
        )
        .build()
}

fn bumpy_field() -> Heightfield {
    let n = 41;
    let mut hf = Heightfield::new(Vec3::new(-2.0, -2.0, 0.0), 0.1, n, n);
    for iy in 0..n {
        for ix in 0..n {
            let x = -2.0 + 0.1 * ix as f64;
            let y = -2.0 + 0.1 * iy as f64;
            hf.heights[iy * n + ix] = (0.03 * (3.0 * x).sin() * (2.0 * y).cos()) as f32;
        }
    }
    hf
}

/// Run `steps` on both kernels from `states` after `setup` configures each,
/// and compare states (and contact readback when enabled) tightly.
#[allow(clippy::too_many_arguments)]
fn kernel_vs_kernel<B: KernelBackend>(
    label: &str,
    model: &Model,
    states: &[State],
    steps: usize,
    (tol_q, tol_v): (f64, f64),
    mk: &impl Fn(Model, usize) -> BatchSim<B>,
    setup_wg: impl FnOnce(&mut GpuBatchSimulator),
    setup_cu: impl FnOnce(&mut BatchSim<B>),
) -> bool {
    if !wgpu_hardware_adapter() {
        return false;
    }
    let Ok(mut wg) = GpuBatchSimulator::new(model.clone(), states.len()) else {
        eprintln!("skipping {label}: no wgpu adapter");
        return false;
    };
    let mut cu = mk(model.clone(), states.len());
    setup_wg(&mut wg);
    setup_cu(&mut cu);
    wg.load_states(states);
    cu.load_states(states);
    for _ in 0..steps {
        wg.step();
        cu.step();
    }
    let a = wg.readback_states();
    let b = cu.readback_states();
    for (w, (x, y)) in a.iter().zip(&b).enumerate() {
        for k in 0..model.nq {
            let d = (x.q[k] - y.q[k]).abs();
            assert!(
                d < tol_q,
                "{label}: cuda-c vs wgsl world {w}: q[{k}] got={} want={} diff={d:.2e}",
                y.q[k],
                x.q[k]
            );
        }
        for k in 0..model.nv {
            let d = (x.v[k] - y.v[k]).abs();
            assert!(
                d < tol_v,
                "{label}: cuda-c vs wgsl world {w}: v[{k}] got={} want={} diff={d:.2e}",
                y.v[k],
                x.v[k]
            );
        }
    }
    if let (Ok(ca), Ok(cb)) = (wg.readback_contacts(), cu.readback_contacts()) {
        for (w, (ra, rb)) in ca.iter().zip(&cb).enumerate() {
            for (bi, (p, q)) in ra.iter().zip(rb).enumerate() {
                assert_eq!(
                    p.touching, q.touching,
                    "{label}: touching, world {w} body {bi}"
                );
                assert!(
                    (p.penetration - q.penetration).abs() < 1e-4
                        && (p.force - q.force).norm() < 1e-2 * (1.0 + p.force.norm()),
                    "{label}: contact readback world {w} body {bi}: {p:?} vs {q:?}"
                );
            }
        }
    }
    true
}

fn tumbling_boxes(model: &Model) -> Vec<State> {
    (0..3)
        .map(|i| {
            let mut s = model.default_state();
            // Tilted, off the ground, moving sideways and spinning.
            s.q[0] = 0.4;
            s.q[1] = -0.3 + 0.1 * i as f64;
            s.q[5] = 0.35;
            s.v[3] = 1.5;
            s.v[4] = -0.5 * i as f64;
            s.v[1] = 2.0;
            s
        })
        .collect()
}

fn suite_unified_contact<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    // 1. Penalty mode with per-body gains, Coulomb friction and the box
    //    manifold: a tumbling box thrown onto the ground.
    let model = free_box(Vec3::new(0.1, 0.15, 0.08), 2.0);
    let g = BodyContactGains::uniform_frequency(&model, 60.0, 1.0);
    let states = tumbling_boxes(&model);
    if !kernel_vs_kernel(
        "penalty tumbling box",
        &model,
        &states,
        1500,
        (5e-4, 5e-4),
        &mk,
        |wg| {
            wg.enable_ground_contact_per_body(0.0, 0.7, &g).unwrap();
        },
        |cu| {
            cu.enable_ground_contact_per_body(0.0, 0.7, &g).unwrap();
        },
    ) {
        return;
    }

    // 2. The same throw, velocity-level impulse solve with sweeps.
    kernel_vs_kernel(
        "impulse tumbling box",
        &model,
        &states,
        1500,
        (5e-4, 5e-4),
        &mk,
        |wg| {
            wg.enable_contact_impulse(0.0, 0.7, &g, None, None).unwrap();
        },
        |cu| {
            cu.enable_contact_impulse(0.0, 0.7, &g, None, None).unwrap();
        },
    );

    // 3. Heightfield terrain, penalty and impulse, plus a mid-run terrain swap.
    let hf = bumpy_field();
    let mut hf2 = hf.clone();
    for v in &mut hf2.heights {
        *v = -*v;
    }
    let states: Vec<State> = (0..2)
        .map(|i| {
            let mut s = model.default_state();
            s.q[3] = 0.15 + 0.3 * i as f64;
            s.q[4] = 0.1;
            s.q[5] = 0.4;
            s.v[3] = 0.8;
            s
        })
        .collect();
    kernel_vs_kernel(
        "penalty heightfield",
        &model,
        &states,
        1200,
        (5e-4, 5e-4),
        &mk,
        |wg| {
            wg.enable_contact_terrain(0.0, 0.8, &g, None, Some(&hf))
                .unwrap();
            wg.set_heightfield(&hf2).unwrap();
        },
        |cu| {
            cu.enable_contact_terrain(0.0, 0.8, &g, None, Some(&hf))
                .unwrap();
            cu.set_heightfield(&hf2).unwrap();
        },
    );
    kernel_vs_kernel(
        "impulse heightfield",
        &model,
        &states,
        1200,
        (5e-4, 5e-4),
        &mk,
        |wg| {
            wg.enable_contact_impulse(0.0, 0.8, &g, None, Some(&hf))
                .unwrap();
        },
        |cu| {
            cu.enable_contact_impulse(0.0, 0.8, &g, None, Some(&hf))
                .unwrap();
        },
    );

    // 4. Body-attached finite face: a rider dropped onto a deck, partly over
    //    the edge and pushed sideways, deck resting on the ground.
    let (model, plane) = deck_and_rider();
    let g = BodyContactGains::uniform_frequency(&model, 50.0, 1.0);
    let states: Vec<State> = (0..2)
        .map(|i| {
            let mut s = model.default_state();
            s.q[5] = 0.05; // deck resting on the ground
            s.q[6 + 3] = 0.3 + 0.1 * i as f64; // rider over the nose
            s.q[6 + 4] = 0.05;
            s.q[6 + 5] = 0.25;
            s.v[6 + 3] = 0.6;
            s
        })
        .collect();
    kernel_vs_kernel(
        "penalty body plane",
        &model,
        &states,
        1500,
        (5e-4, 5e-4),
        &mk,
        |wg| {
            wg.enable_ground_contact_with_plane(0.0, 0.8, &g, Some(&plane))
                .unwrap();
        },
        |cu| {
            cu.enable_ground_contact_with_plane(0.0, 0.8, &g, Some(&plane))
                .unwrap();
        },
    );
    // Once the rider is at rest on the deck (~0.9 s in), a resting contact's
    // impulse update sits on a branch edge and the two f32 compilers flip it
    // on different steps: `v` shows one-step transients of a few 1e-3 that
    // the next step removes, while `q` never leaves rounding level (~1e-6,
    // probed). So `q` is held tight and `v` is allowed the transient.
    kernel_vs_kernel(
        "impulse body plane",
        &model,
        &states,
        1500,
        (5e-4, 1e-2),
        &mk,
        |wg| {
            wg.enable_contact_impulse(0.0, 0.8, &g, Some(&plane), None)
                .unwrap();
        },
        |cu| {
            cu.enable_contact_impulse(0.0, 0.8, &g, Some(&plane), None)
                .unwrap();
        },
    );

    // 5. Passive joint spring and armature in the ABA pass, against the CPU
    //    directly (no contact involved).
    let model = sprung_pendulum();
    let mut s = model.default_state();
    s.q[0] = 1.2;
    let mut cu = mk(model.clone(), 1);
    cu.load_states(std::slice::from_ref(&s));
    let mut cpu = s.clone();
    for _ in 0..2000 {
        cu.step();
        cpu_step(&model, &mut cpu);
    }
    assert_state_close(
        "sprung pendulum vs cpu",
        &cu.readback_states()[0],
        &cpu,
        2e-3,
    );
}

// ── The device-resident control loop: FK readout, observation, policy ─────

/// Four worlds of the free base + limb, each in its own pose and motion, run
/// through the on-device loop for three control steps with physics between
/// them, against the CPU reference in `policy_pipeline`: the same FK, the
/// same op table, the same MLP, and the same random stream (xorshift64 +
/// Box–Muller from `world_seed`), so observations, noise, actions,
/// log-probs and the PD targets they write are all checked to f32
/// precision. Then the state history: two device snapshots against two
/// direct readbacks.
fn suite_policy<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let model = free_base_with_limb();
    let nworld = 4;
    let limb_q = model.q_offsets[model.bodies[1].joint_idx];
    let limb_v = model.v_offsets[model.bodies[1].joint_idx];

    let mut sim = mk(model.clone(), nworld);
    sim.enable_pd_control(&[PdDof {
        q_index: limb_q,
        v_index: limb_v,
        kp: 20.0,
        kd: 1.0,
        max_force: 10.0,
    }])
    .unwrap();

    let spec = PolicySpec {
        obs: vec![
            ObsOp::BodyPitch(0),
            ObsOp::BodyRoll(0),
            ObsOp::V(0),
            ObsOp::V(1),
            ObsOp::V(2),
            ObsOp::QMinus(limb_q, 0.1),
            ObsOp::V(limb_v),
            ObsOp::Const(0.5),
            ObsOp::BodyYawError(0, 0.3),
            ObsOp::BodyPosZ(1),
        ],
        hidden: 16,
        act_slots: vec![0],
        act_clamp: 0.4,
        act_clamp_slots: None,
        noise_rho: 0.3,
        input_noise: vec![0.01, 0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
        history_steps: 3,
    };
    sim.enable_policy(spec.clone()).unwrap();

    // Weights from a fixed stream — anything nontrivial and reproducible.
    let mut wr = KernelRng(0xC0FFEE);
    let weights: Vec<f64> = (0..spec.n_weights()).map(|_| 0.3 * wr.normal()).collect();
    let std = vec![0.2];
    let base: Vec<Vec<f64>> = (0..nworld).map(|w| vec![0.05 * w as f64]).collect();
    sim.set_policy_weights(&weights).unwrap();
    sim.set_policy_std(&std).unwrap();
    sim.set_policy_base_targets(&base).unwrap();
    let seed = 99u64;
    sim.seed_policy(seed).unwrap();

    // Distinct poses and motions per world.
    let mut states: Vec<State> = (0..nworld)
        .map(|w| {
            let mut s = model.default_state();
            let f = w as f64;
            s.q[0] = 0.2 * f;
            s.q[1] = -0.1 * f;
            s.q[2] = 0.3 * f;
            s.q[5] = 1.0 + 0.1 * f;
            s.q[limb_q] = 0.4 - 0.2 * f;
            s.v[0] = 0.5 * f;
            s.v[1] = -0.2;
            s.v[3] = 0.1 * f;
            s.v[limb_v] = 0.7;
            s
        })
        .collect();
    sim.load_states(&states);
    sim.enable_state_history(2).unwrap();

    // Host mirror of every world's stream and AR(1) state.
    let mut rngs: Vec<KernelRng> = (0..nworld).map(|w| KernelRng::for_world(seed, w)).collect();
    let mut zs = vec![vec![0.0f64; spec.n_out()]; nworld];
    let mut want_targets: Vec<Vec<f64>> = vec![vec![0.0; 1]; nworld];

    let mut want_obs: Vec<Vec<f64>> = Vec::new();
    let mut want_act: Vec<Vec<f64>> = Vec::new();
    let mut want_logp: Vec<f64> = Vec::new();

    for step in 0..3 {
        if step == 0 {
            sim.record_state(0).unwrap();
        }
        // The device: FK -> obs -> policy -> targets, from its own q/v.
        sim.run_policy(step).unwrap();
        // The reference: same q/v (the device's, so physics precision does
        // not leak into this comparison), CPU FK, CPU policy.
        let dev_states = sim.readback_states();
        for w in 0..nworld {
            let mut s = dev_states[w].clone();
            let mut obs = observe_reference(&model, &mut s, &spec.obs);
            let (act, logp) = policy_reference(
                &spec,
                &weights,
                &std,
                &mut rngs[w],
                &mut obs,
                &mut zs[w],
                &base[w],
                &mut want_targets[w],
            );
            want_obs.push(obs);
            want_act.push(act);
            want_logp.push(logp);
        }
        // Targets are what the PD pass will read: check them now.
        let got_t = sim.readback_targets().unwrap();
        for w in 0..nworld {
            let d = (got_t[w] as f64 - want_targets[w][0]).abs();
            assert!(
                d < 2e-4,
                "step {step} world {w}: target got {} want {} diff {d:.2e}",
                got_t[w],
                want_targets[w][0]
            );
        }
        // Physics between control steps, PD driving toward those targets.
        for _ in 0..5 {
            sim.step();
        }
        if step == 0 {
            sim.record_state(1).unwrap();
            states = sim.readback_states();
        }
    }

    // History rows against the reference rows.
    let (obs_h, out_h) = sim.readback_policy_history(0..3).unwrap();
    let n_in = spec.n_in();
    let n_out = spec.n_out();
    for (row, want) in want_obs.iter().enumerate() {
        for i in 0..n_in {
            let got = obs_h[row * n_in + i] as f64;
            let d = (got - want[i]).abs();
            assert!(
                d < 1e-4,
                "obs row {row} feature {i} ({:?}): got {got} want {} diff {d:.2e}",
                spec.obs[i],
                want[i]
            );
        }
    }
    for (row, want) in want_act.iter().enumerate() {
        for k in 0..n_out {
            let got = out_h[row * (n_out + 1) + k] as f64;
            let d = (got - want[k]).abs();
            assert!(
                d < 2e-4,
                "act row {row} k {k}: got {got} want {} diff {d:.2e}",
                want[k]
            );
        }
        let got_lp = out_h[row * (n_out + 1) + n_out] as f64;
        let d = (got_lp - want_logp[row]).abs();
        assert!(
            d < 1e-3,
            "logp row {row}: got {got_lp} want {} diff {d:.2e}",
            want_logp[row]
        );
    }
    // The observations moved between steps (the physics ran) and the noise
    // was recorded (an entry with noise differs from the noiseless op).
    assert!(
        (obs_h[0] - obs_h[nworld * n_in]).abs() > 0.0
            || (obs_h[5] - obs_h[nworld * n_in + 5]).abs() > 0.0
    );

    // FK readout against CPU FK on the same state.
    sim.compute_kinematics().unwrap();
    let kin = sim.readback_kinematics().unwrap();
    let cur = sim.readback_states();
    for w in 0..nworld {
        let mut s = cur[w].clone();
        let (xf, _) = phyz_rigid::forward_kinematics(&model, &s);
        s.body_xform = xf;
        for b in 0..model.nbodies() {
            let row = &kin[(w * model.nbodies() + b) * XF_STRIDE..][..XF_STRIDE];
            let x = &s.body_xform[b];
            for r in 0..3 {
                for c in 0..3 {
                    let d = (row[r * 3 + c] as f64 - x.rot[(r, c)]).abs();
                    assert!(d < 1e-5, "fk world {w} body {b} rot[{r}][{c}] diff {d:.2e}");
                }
            }
            for (k, want) in [x.pos.x, x.pos.y, x.pos.z].into_iter().enumerate() {
                let d = (row[9 + k] as f64 - want).abs();
                assert!(d < 1e-5, "fk world {w} body {b} pos[{k}] diff {d:.2e}");
            }
        }
    }

    // State history: slot 0 was the loaded state, slot 1 the state after
    // the first five steps — the same numbers a direct readback gave then.
    let (qh, vh) = sim.readback_state_history(0..2).unwrap();
    let (nq, nv) = (model.nq, model.nv);
    let slot0 =
        phyz_gpu::layout::unpack_states(&model, nworld, &qh[..nworld * nq], &vh[..nworld * nv]);
    let slot1 =
        phyz_gpu::layout::unpack_states(&model, nworld, &qh[nworld * nq..], &vh[nworld * nv..]);
    for w in 0..nworld {
        // Slot 0 vs the states we loaded (f32 round trip only).
        let mut loaded = model.default_state();
        let f = w as f64;
        loaded.q[0] = 0.2 * f;
        loaded.q[1] = -0.1 * f;
        loaded.q[2] = 0.3 * f;
        loaded.q[5] = 1.0 + 0.1 * f;
        loaded.q[limb_q] = 0.4 - 0.2 * f;
        loaded.v[0] = 0.5 * f;
        loaded.v[1] = -0.2;
        loaded.v[3] = 0.1 * f;
        loaded.v[limb_v] = 0.7;
        assert_state_close(
            &format!("history slot 0 world {w}"),
            &slot0[w],
            &loaded,
            1e-6,
        );
        assert_state_close(
            &format!("history slot 1 world {w}"),
            &slot1[w],
            &states[w],
            1e-6,
        );
    }

    // The shape guards.
    assert!(
        sim.run_policy(3).is_err(),
        "step past history_steps must be refused"
    );
    assert!(
        sim.record_state(2).is_err(),
        "slot past history must be refused"
    );
    let bad = PolicySpec {
        act_slots: vec![7],
        ..spec.clone()
    };
    assert!(
        sim.enable_policy(bad).is_err(),
        "action slot past the PD table must be refused"
    );
}

/// Every PD servo tracks its uploaded base target, not just the ones the
/// policy actions. Four servos, two of them actioned; on a *fresh* sim
/// (targets still zero from allocation) one policy pass must leave the
/// other two at their non-zero base — issue #76, where they stayed at
/// zero and drove a partially-actuated robot to a different pose.
fn suite_policy_base_targets<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let model = arm_6dof();
    let nworld = 3;
    let pd: Vec<PdDof> = (0..4)
        .map(|k| PdDof {
            q_index: k,
            v_index: k,
            kp: 20.0,
            kd: 1.0,
            max_force: 10.0,
        })
        .collect();
    let spec = PolicySpec {
        obs: vec![ObsOp::Const(1.0), ObsOp::QMinus(0, 0.0), ObsOp::V(0)],
        hidden: 8,
        act_slots: vec![0, 2],
        act_clamp: 0.5,
        act_clamp_slots: None,
        noise_rho: 0.0,
        input_noise: vec![0.0, 0.0, 0.0],
        history_steps: 1,
    };
    let base: Vec<Vec<f64>> = (0..nworld)
        .map(|w| (0..4).map(|j| 0.1 + 0.01 * (4 * w + j) as f64).collect())
        .collect();
    let mut sim = mk(model.clone(), nworld);
    sim.enable_pd_control(&pd).unwrap();
    sim.enable_policy(spec.clone()).unwrap();
    // Zero weights and zero exploration: the action is exactly zero, so
    // every slot — actioned or not — must read back its base.
    sim.set_policy_weights(&vec![0.0; spec.n_weights()])
        .unwrap();
    sim.set_policy_std(&vec![1e-9; spec.n_out()]).unwrap();
    sim.set_policy_base_targets(&base).unwrap();
    sim.seed_policy(11).unwrap();
    sim.load_states(&vec![model.default_state(); nworld]);
    sim.run_policy(0).unwrap();
    let got = sim.readback_targets().unwrap();
    for w in 0..nworld {
        for slot in 0..4 {
            let d = (got[w * 4 + slot] as f64 - base[w][slot]).abs();
            assert!(
                d < 1e-6,
                "world {w} slot {slot}: got {} want base {} diff {d:.2e}",
                got[w * 4 + slot],
                base[w][slot]
            );
        }
    }
}

/// The per-action-slot clamp. Four PD servos on a 6-DOF arm, two of them
/// actioned with clamps a factor 35 apart: with an action large enough to
/// saturate both, each slot must sit at `base + ±clamp_k`, not at one
/// shared limit. Then: filling every slot with the scalar reproduces the
/// scalar path bit for bit.
fn suite_policy_clamp_slots<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let model = arm_6dof();
    let nworld = 3;
    let pd: Vec<PdDof> = (0..4)
        .map(|k| PdDof {
            q_index: k,
            v_index: k,
            kp: 20.0,
            kd: 1.0,
            max_force: 10.0,
        })
        .collect();
    let spec = PolicySpec {
        obs: vec![ObsOp::Const(1.0), ObsOp::QMinus(0, 0.0), ObsOp::V(0)],
        hidden: 8,
        act_slots: vec![0, 2],
        act_clamp: 0.5,
        act_clamp_slots: Some(vec![0.02, 0.7]),
        noise_rho: 0.0,
        input_noise: vec![0.0, 0.0, 0.0],
        history_steps: 1,
    };
    // A non-zero base row for every servo, distinct per world.
    let base: Vec<Vec<f64>> = (0..nworld)
        .map(|w| (0..4).map(|j| 0.1 + 0.01 * (4 * w + j) as f64).collect())
        .collect();
    // Weights big enough that both actions saturate their clamp.
    let mut wr = KernelRng(0x5EED);
    let weights: Vec<f64> = (0..spec.n_weights()).map(|_| 4.0 * wr.normal()).collect();
    let std = vec![0.1, 0.1];

    let run = |spec: &PolicySpec| -> (Vec<f32>, Vec<f64>) {
        let mut sim = mk(model.clone(), nworld);
        sim.enable_pd_control(&pd).unwrap();
        sim.enable_policy(spec.clone()).unwrap();
        sim.set_policy_weights(&weights).unwrap();
        sim.set_policy_std(&std).unwrap();
        sim.set_policy_base_targets(&base).unwrap();
        sim.seed_policy(4242).unwrap();
        sim.load_states(&vec![model.default_state(); nworld]);
        sim.run_policy(0).unwrap();
        let (_, out) = sim.readback_policy_history(0..1).unwrap();
        let acts = (0..nworld * spec.n_out())
            .map(|i| out[i / spec.n_out() * (spec.n_out() + 1) + i % spec.n_out()] as f64)
            .collect();
        (sim.readback_targets().unwrap(), acts)
    };

    let (got, acts) = run(&spec);
    let limits = spec.act_clamp_slots.clone().unwrap();
    for w in 0..nworld {
        // The actioned slots: base + the action clamped by *that* slot.
        for (k, &slot) in spec.act_slots.iter().enumerate() {
            let a = acts[w * spec.n_out() + k];
            let want = base[w][slot] + a.clamp(-limits[k], limits[k]);
            let d = (got[w * 4 + slot] as f64 - want).abs();
            assert!(
                d < 2e-6,
                "world {w} action {k} slot {slot}: got {} want {want} (action {a}, clamp {}) diff {d:.2e}",
                got[w * 4 + slot],
                limits[k]
            );
            assert!(
                a.abs() > limits[k],
                "world {w} action {k} = {a} did not reach the clamp {} — the test proves nothing",
                limits[k]
            );
        }
    }
    // The two clamps really are different in the result.
    assert!(
        (got[0] as f64 - base[0][0]).abs() < 0.5 * (got[2] as f64 - base[0][2]).abs(),
        "both slots moved by the same amount — the per-slot clamp did nothing"
    );

    // Same value in every slot == the scalar path, bit for bit.
    let scalar = PolicySpec {
        act_clamp: 0.37,
        act_clamp_slots: None,
        ..spec.clone()
    };
    let broadcast = PolicySpec {
        act_clamp: 0.37,
        act_clamp_slots: Some(vec![0.37; spec.n_out()]),
        ..spec.clone()
    };
    let (a, _) = run(&scalar);
    let (b, _) = run(&broadcast);
    assert_eq!(a, b, "broadcast per-slot clamp is not the scalar path");

    // The shape guard.
    let bad = PolicySpec {
        act_clamp_slots: Some(vec![0.1]),
        ..spec.clone()
    };
    let mut sim = mk(model.clone(), nworld);
    sim.enable_pd_control(&pd).unwrap();
    assert!(
        sim.enable_policy(bad).is_err(),
        "act_clamp_slots of the wrong length must be refused"
    );
}

// ── Launch graphs ─────────────────────────────────────────────────────────

/// A replayed step span against the same launches issued by hand.
///
/// A CUDA Graph records the kernels, their arguments and their buffer
/// addresses, and replaying it re-runs exactly those. So the bar here is not
/// a tolerance — it is bit-identity, and anything short of it means the
/// capture picked up something that should not have been in it, or a stale
/// recording survived a change that should have discarded it.
///
/// On a backend without graph support this still checks the contract from
/// the other side: `step_many(n)` must be `n` calls to `step()`.
fn suite_graph_replay<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B>) {
    let Some(model) = ant_or_skip() else { return };
    let nworld = 16;
    let steps = 60;
    let chunk = 20;
    let gains = BodyContactGains::uniform_frequency(&model, 100.0, 1.0);
    let pd: Vec<PdDof> = (6..model.nv)
        .map(|i| PdDof {
            q_index: i,
            v_index: i,
            kp: 40.0,
            kd: 2.0,
            max_force: 60.0,
        })
        .collect();

    let mut start = model.default_state();
    start.q[5] = 0.75;
    let states = vec![start; nworld];

    let mut sim = mk(model.clone(), nworld);
    sim.enable_contact_impulse(0.0, 0.8, &gains, None, None)
        .unwrap();
    sim.enable_pd_control(&pd).unwrap();
    sim.set_position_targets(&vec![vec![0.2; pd.len()]; nworld])
        .unwrap();
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);

    // The reference: graphs off, one launch at a time.
    sim.set_graphs_enabled(false);
    assert!(!sim.graphs_enabled(), "graphs must stay off once disabled");
    sim.load_states(&states);
    for _ in 0..steps {
        sim.step();
    }
    let want = sim.readback_states();

    // Same simulator, graphs back on, from the same initial states.
    sim.set_graphs_enabled(true);
    sim.load_states(&states);
    for _ in 0..steps {
        sim.step();
    }
    assert_bit_identical("step() replayed", &model, &want, &sim.readback_states());

    // And as one graph over a whole control period.
    sim.load_states(&states);
    for _ in 0..(steps / chunk) {
        sim.step_many(chunk).unwrap();
    }
    assert_bit_identical(
        "step_many() replayed",
        &model,
        &want,
        &sim.readback_states(),
    );

    // Capture invalidation: a different sweep count is a different launch
    // sequence, so the cached recording must not be replayed for it.
    sim.contact_sweeps = 4;
    sim.load_states(&states);
    for _ in 0..(steps / chunk) {
        sim.step_many(chunk).unwrap();
    }
    let few_sweeps = sim.readback_states();
    sim.set_graphs_enabled(false);
    sim.load_states(&states);
    for _ in 0..steps {
        sim.step();
    }
    assert_bit_identical(
        "sweep count changed under a cached graph",
        &model,
        &sim.readback_states(),
        &few_sweeps,
    );
    // Guard the guard: if 4 and 16 sweeps land in the same place, the check
    // above would pass on a stale replay too.
    assert!(
        (0..model.nv).any(|j| want[0].v[j] != few_sweeps[0].v[j]),
        "4 sweeps and 16 sweeps produced identical states — the sweep count \
         is not reaching the solve, so this test proves nothing"
    );
}

fn ant_or_skip() -> Option<Model> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    match phyz_mjcf::MjcfLoader::from_file(path) {
        Ok(l) => Some(l.build_model()),
        Err(_) => {
            eprintln!("skipping graph replay: {path} not found");
            None
        }
    }
}

fn assert_bit_identical(what: &str, model: &Model, want: &[State], got: &[State]) {
    assert_eq!(want.len(), got.len(), "{what}: world count");
    for (w, (a, b)) in want.iter().zip(got).enumerate() {
        for j in 0..model.nq {
            assert_eq!(
                a.q[j].to_bits(),
                b.q[j].to_bits(),
                "{what}: world {w} q[{j}] {} vs {}",
                a.q[j],
                b.q[j]
            );
        }
        for j in 0..model.nv {
            assert_eq!(
                a.v[j].to_bits(),
                b.v[j].to_bits(),
                "{what}: world {w} v[{j}] {} vs {}",
                a.v[j],
                b.v[j]
            );
        }
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn run_all<B: KernelBackend>(mk: impl Fn(Model, usize) -> BatchSim<B> + Copy) {
    suite_single_steps(mk);
    suite_trajectory(mk);
    suite_pd(mk);
    suite_contact(mk);
    suite_ant(mk);
    suite_vs_wgpu(mk);
    suite_unified_contact(mk);
    suite_policy(mk);
    suite_policy_base_targets(mk);
    suite_policy_clamp_slots(mk);
    suite_graph_replay(mk);
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
    fn unified_contact_matches_wgsl_kernels() {
        suite_unified_contact(mk);
    }
    #[test]
    fn device_policy_loop_matches_cpu() {
        suite_policy(mk);
    }
    #[test]
    fn policy_writes_every_base_target() {
        suite_policy_base_targets(mk);
    }
    #[test]
    fn policy_clamps_per_action_slot() {
        suite_policy_clamp_slots(mk);
    }
    #[test]
    fn step_many_matches_repeated_step() {
        suite_graph_replay(mk);
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
