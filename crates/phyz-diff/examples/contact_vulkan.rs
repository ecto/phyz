//! The K1 contact step (traced kinematics, rolled PGS solve, traced finish)
//! batched on the GPU through tang-gpu: WGSL for the traced kernels, GLSL
//! 450 for the solve, both to SPIR-V for Vulkan (an Adreno, a 3090) or
//! through naga elsewhere. Checks every environment against the f64 split
//! pipeline, then times steps.
//!
//! `cargo run --release -p phyz-diff --features vulkan --example contact_vulkan [ENVS] [SWEEPS]`

use std::time::Instant;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_split::ContactKernels;
use phyz_diff::contact_step::ContactPoint;
use phyz_math::{SpatialTransformExt as _, Vec3};
use phyz_model::{Actuator, Model};
use tang_expr::codegen::Dialect;
use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

fn main() {
    let mut args = std::env::args().skip(1);
    let envs: usize = args.next().map_or(16384, |s| s.parse().unwrap());
    let sweeps: usize = args.next().map_or(5, |s| s.parse().unwrap());

    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let split = ContactKernels::build(&model, &points, &mat, &config);
    let lay = split.layout();
    let (nq, nv) = (model.nq, model.nv);
    let (nx, nfin, dim) = (
        split.kinematics_inputs(),
        split.finish_inputs(),
        3 * points.len(),
    );
    let (asm_fn, fin_fn) = split.compile();

    // A standing rollout in the f64 split pipeline: its states (in contact)
    // and warm starts, cycled across the environments.
    let mut s = model.default_state();
    let (xf, _) = phyz_rigid::forward_kinematics(&model, &s);
    let low = points
        .iter()
        .map(|p| xf[p.body].body_to_world_point(p.local).z)
        .fold(f64::INFINITY, f64::min);
    s.q[5] += 0.01 - low;
    let mut x: Vec<f64> =
        s.q.as_slice()
            .iter()
            .chain(s.v.as_slice())
            .chain(s.ctrl.as_slice())
            .copied()
            .collect();
    let mut warm = vec![0.0; dim];
    let mut rec = vec![0.0; lay.len];
    let mut out = vec![0.0; nq + nv];
    let (mut xs, mut warms) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        xs.push(x.clone());
        warms.push(warm.clone());
        asm_fn(&x, &mut rec);
        let (f, v1) = split.solve(&rec, &warm, sweeps);
        let fin: Vec<f64> = x[..nq + nv].iter().chain(&v1).copied().collect();
        fin_fn(&fin, &mut out);
        x[..nq + nv].copy_from_slice(&out);
        warm = f;
    }
    let pick = |e: usize| 100 + e % 200;
    let x_in: Vec<f32> = (0..envs)
        .flat_map(|e| xs[pick(e)].iter().map(|&v| v as f32))
        .collect();
    let w_in: Vec<f32> = (0..envs)
        .flat_map(|e| warms[pick(e)].iter().map(|&v| v as f32))
        .collect();

    let device = GpuDevice::new_sync().expect("gpu");
    eprintln!("device: {}", device.name);
    let mut cache = KernelCache::new();
    let (kin, fin) = split.traced_sources(Dialect::Wgsl);
    let kin = kin.replace(" k_kin(", " main(");
    let fin = fin.replace(" k_finish(", " main(");
    let solve = split.solve_source(Dialect::Wgsl);

    let bx = GpuBuffer::from_slice(&device, &x_in);
    let bw = GpuBuffer::from_slice(&device, &w_in);
    let brec = GpuBuffer::uninit(&device, envs * lay.len);
    let bscr = GpuBuffer::uninit(&device, envs * lay.scratch);
    let bf = GpuBuffer::uninit(&device, envs * dim);
    let bfin = GpuBuffer::uninit(&device, envs * nfin);
    let bout = GpuBuffer::uninit(&device, envs * (nq + nv));
    let n = envs as u32;
    let mut step = |cache: &mut KernelCache| {
        cache.dispatch_buffers(&device, &kin, &[&bx, &brec], &[1], &[n, 0, 0, 0], n, 256);
        cache.dispatch_buffers(
            &device,
            &solve,
            &[&brec, &bw, &bx, &bscr, &bf, &bfin],
            &[3, 4, 5],
            &[n, sweeps as u32, 0, 0],
            n,
            64,
        );
        cache.dispatch_buffers(&device, &fin, &[&bfin, &bout], &[1], &[n, 0, 0, 0], n, 256);
    };
    let t = Instant::now();
    step(&mut cache);
    cache.flush(&device);
    let got_out = bout.to_vec_sync(&device);
    let got_f = bf.to_vec_sync(&device);
    eprintln!("first step (compiles) {:.2}s", t.elapsed().as_secs_f64());

    // every environment against the f64 split pipeline
    let mut worst: f64 = 0.0;
    for e in 0..envs.min(2000) {
        let i = pick(e);
        asm_fn(&xs[i], &mut rec);
        let (f, v1) = split.solve(&rec, &warms[i], sweeps);
        let fin: Vec<f64> = xs[i][..nq + nv].iter().chain(&v1).copied().collect();
        fin_fn(&fin, &mut out);
        let scale = out.iter().chain(&f).fold(1.0f64, |m, v| m.max(v.abs()));
        let got = got_out[e * (nq + nv)..(e + 1) * (nq + nv)]
            .iter()
            .chain(&got_f[e * dim..(e + 1) * dim]);
        for (a, b) in got.zip(out.iter().chain(&f)) {
            let d = (*a as f64 - b).abs() / scale;
            worst = if d.is_nan() || d > worst { d } else { worst };
        }
    }
    println!(
        "f32 GPU vs f64 split pipeline: worst relative error {worst:.2e} (nx {nx}, rec {}, {} points)",
        lay.len,
        points.len()
    );

    let reps = 50;
    let t = Instant::now();
    for _ in 0..reps {
        step(&mut cache);
    }
    cache.flush(&device);
    let _ = bout.to_vec_sync(&device);
    let dt = t.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{envs} envs x {sweeps} sweeps: {:.3} ms per step, {:.2} M env-steps/s",
        dt * 1e3,
        envs as f64 / dt / 1e6
    );
}

/// The K1 at dt = 1 ms with position servos (D as implicit joint damping) and
/// the vendor MJCF foot pads as eight sole corners. Mirrors
/// `tests/contact_parity.rs`.
fn k1() -> (Model, Vec<ContactPoint>) {
    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.001),
        gravity: None,
    };
    let mut model = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts)
        .expect("K1 URDF")
        .model;
    model.actuators.clear();
    for j in 0..model.joints.len() {
        if model.joints[j].ndof() == 1 {
            let name = model.joints[j].name.clone();
            let leg = ["Hip", "Knee", "Ankle"].iter().any(|k| name.contains(k));
            let (kp, kd) = if leg { (80.0, 2.0) } else { (20.0, 0.5) };
            model.joints[j].damping = kd;
            model
                .actuators
                .push(Actuator::position(&name, &name, j, 1.0, kp, 0.0));
        }
    }
    let mut points = Vec::new();
    for foot in ["left_foot_link", "right_foot_link"] {
        let body = model.bodies.iter().position(|b| b.name == foot).unwrap();
        for (x, y) in [
            (0.116, 0.035),
            (0.116, -0.035),
            (-0.064, 0.035),
            (-0.064, -0.035),
        ] {
            points.push(ContactPoint {
                body,
                local: Vec3::new(x, y, -0.038),
            });
        }
    }
    (model, points)
}
