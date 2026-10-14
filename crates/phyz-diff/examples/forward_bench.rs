//! Emit the K1's traced ABA and full contact-free step as C (f32 and f64),
//! with random states and phyz reference outputs, and time phyz on the same states.
//!
//! ```text
//! cargo run -p phyz-diff --release --example forward_bench -- OUT_DIR [N]
//! ```
//!
//! `bench-c/bench.c` then compiles against the emitted kernels and reports
//! latency at batch 1, batched throughput, and error against the reference.

use std::io::Write;
use std::time::Instant;

use phyz_diff::forward::{ForwardDynamics, Step};
use tang_expr::codegen::Dialect;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("usage: forward_bench OUT_DIR [N]"));
    let n: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(1000);
    std::fs::create_dir_all(&out).unwrap();

    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.002),
        gravity: None,
    };
    let model = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts)
        .expect("K1 URDF")
        .model;

    let mut seed = 0x5eed_u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut states = Vec::with_capacity(n);
    for _ in 0..n {
        let mut s = model.default_state();
        for k in 0..model.nq {
            s.q[k] = rnd();
        }
        for k in 0..model.nv {
            s.v[k] = 2.0 * rnd();
        }
        for k in 0..s.ctrl.len() {
            s.ctrl[k] = rnd();
        }
        states.push(s);
    }

    let mut meta = format!("n {n}\nnq {} nv {}\n", model.nq, model.nv);

    // ABA alone: (q, v, qfrc) -> qdd.
    let t = Instant::now();
    let fd = ForwardDynamics::build(&model);
    let trace_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut inputs = Vec::new();
    let mut reference = Vec::new();
    for s in &states {
        let qfrc = phyz_rigid::actuation::generalized_forces(&model, s);
        inputs.extend(fd.inputs(s, qfrc.as_slice()));
        reference.extend_from_slice(phyz_rigid::aba(&model, s).as_slice());
    }
    emit(
        &out,
        "aba",
        &fd.kernel(Dialect::C).source,
        &fd.kernel_c_f64().source,
        &fd.kernel(Dialect::Msl).source,
        &inputs,
        &reference,
    );
    let us = time_us(&states, |s| {
        std::hint::black_box(phyz_rigid::aba(&model, s));
    });
    meta += &format!(
        "aba: n_inputs {} n_out {} live_nodes {} trace_ms {trace_ms:.1} phyz_us {us}\n",
        fd.n_inputs(),
        model.nv,
        fd.live_nodes()
    );

    // Full contact-free step: (q, v, ctrl) -> (q', v').
    let t = Instant::now();
    let step = Step::build(&model);
    let trace_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut inputs = Vec::new();
    let mut reference = Vec::new();
    for s in &states {
        inputs.extend(step.inputs(s));
        let mut s1 = s.clone();
        phyz_step(&model, &mut s1);
        reference.extend_from_slice(s1.q.as_slice());
        reference.extend_from_slice(s1.v.as_slice());
    }
    emit(
        &out,
        "step",
        &step.kernel(Dialect::C).source,
        &step.kernel_c_f64().source,
        &step.kernel(Dialect::Msl).source,
        &inputs,
        &reference,
    );
    let us = time_us(&states, |s| {
        let mut s1 = s.clone();
        phyz_step(&model, &mut s1);
        std::hint::black_box(s1);
    });
    meta += &format!(
        "step: n_inputs {} n_out {} live_nodes {} trace_ms {trace_ms:.1} phyz_us {us}\n",
        step.n_inputs(),
        model.nq + model.nv,
        step.live_nodes()
    );

    std::fs::write(out.join("meta.txt"), &meta).unwrap();
    print!("{meta}");
}

fn phyz_step(model: &phyz_model::Model, s: &mut phyz_model::State) {
    let qdd = phyz_rigid::aba(model, s);
    phyz_rigid::semi_implicit_euler(model, s, qdd.as_slice(), model.dt);
}

/// Write `{name}.metal`, `{name}_f32.c`, `{name}_f64.c`, `{name}_inputs.bin`, `{name}_reference.bin`.
fn emit(
    out: &std::path::Path,
    name: &str,
    c32: &str,
    c64: &str,
    msl: &str,
    inputs: &[f64],
    reference: &[f64],
) {
    std::fs::write(out.join(format!("{name}.metal")), msl).unwrap();
    std::fs::write(out.join(format!("{name}_f32.c")), c32).unwrap();
    std::fs::write(out.join(format!("{name}_f64.c")), c64).unwrap();
    write_f64(&out.join(format!("{name}_inputs.bin")), inputs);
    write_f64(&out.join(format!("{name}_reference.bin")), reference);
}

/// Batch-1 wall time of `f` over `states` (5 passes): "median M p99 P".
fn time_us(states: &[phyz_model::State], mut f: impl FnMut(&phyz_model::State)) -> String {
    let mut ns: Vec<f64> = Vec::with_capacity(states.len() * 5);
    for _ in 0..5 {
        for s in states {
            let t = Instant::now();
            f(std::hint::black_box(s));
            ns.push(t.elapsed().as_nanos() as f64);
        }
    }
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| ns[((ns.len() - 1) as f64 * p) as usize] / 1e3;
    format!("median {:.2} p99 {:.2}", pct(0.5), pct(0.99))
}

fn write_f64(path: &std::path::Path, xs: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in xs {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}
