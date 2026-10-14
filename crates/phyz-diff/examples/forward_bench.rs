//! Emit the K1's traced ABA as C (f32 and f64), plus random states and
//! `phyz_rigid::aba` reference outputs, and time `phyz_rigid::aba` itself.
//!
//! ```text
//! cargo run -p phyz-diff --release --example forward_bench -- OUT_DIR [N]
//! ```
//!
//! `bench-c/bench.c` then compiles against the emitted kernels and reports
//! latency at batch 1, batched throughput, and error against the reference.

use std::io::Write;
use std::time::Instant;

use phyz_diff::forward::ForwardDynamics;
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

    let t = Instant::now();
    let fd = ForwardDynamics::build(&model);
    let trace_ms = t.elapsed().as_secs_f64() * 1e3;
    // tang-expr emits `Neg(Lit(-x))` as `--x`, which C parses as a decrement.
    let src = fd.kernel(Dialect::C).source.replace("(--", "(- -");
    std::fs::write(out.join("k1_f32.c"), &src).unwrap();
    std::fs::write(out.join("k1_f64.c"), to_f64(&src)).unwrap();

    // Random states and reference outputs.
    let mut seed = 0x5eed_u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut inputs = Vec::with_capacity(n * fd.n_inputs());
    let mut reference = Vec::with_capacity(n * model.nv);
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
        let qfrc = phyz_rigid::actuation::generalized_forces(&model, &s);
        inputs.extend(fd.inputs(&s, qfrc.as_slice()));
        reference.extend_from_slice(phyz_rigid::aba(&model, &s).as_slice());
        states.push(s);
    }
    write_f64(&out.join("inputs.bin"), &inputs);
    write_f64(&out.join("reference.bin"), &reference);

    // phyz_rigid::aba per call, batch 1 (includes its own qfrc).
    let mut ns: Vec<f64> = Vec::with_capacity(n * 5);
    for _ in 0..5 {
        for s in &states {
            let t = Instant::now();
            std::hint::black_box(phyz_rigid::aba(&model, std::hint::black_box(s)));
            ns.push(t.elapsed().as_nanos() as f64);
        }
    }
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| ns[((ns.len() - 1) as f64 * p) as usize] / 1e3;

    let meta = format!(
        "n {n}\nn_inputs {}\nnv {}\nlive_nodes {}\ntrace_ms {trace_ms:.1}\nphyz_aba_us median {:.2} p99 {:.2} p999 {:.2}\n",
        fd.n_inputs(),
        model.nv,
        fd.live_nodes(),
        pct(0.5),
        pct(0.99),
        pct(0.999),
    );
    std::fs::write(out.join("meta.txt"), &meta).unwrap();
    print!("{meta}");
}

/// The C dialect emits f32; for f64 swap the type and drop literal `f` suffixes.
fn to_f64(src: &str) -> String {
    let src = src.replace("float", "double");
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    for (i, &c) in b.iter().enumerate() {
        let prev_digit = i > 0 && (b[i - 1].is_ascii_digit() || b[i - 1] == b'.');
        let next_word = b
            .get(i + 1)
            .is_some_and(|n| n.is_ascii_alphanumeric() || *n == b'_');
        if c == b'f' && prev_digit && !next_word && !is_ident_tail(b, i) {
            continue;
        }
        out.push(c as char);
    }
    out
}

/// True if `b[i]` ends an identifier like `t12f`, not a numeric literal.
fn is_ident_tail(b: &[u8], i: usize) -> bool {
    let mut j = i;
    while j > 0 && (b[j - 1].is_ascii_digit() || b[j - 1] == b'.') {
        j -= 1;
    }
    j > 0 && (b[j - 1].is_ascii_alphabetic() || b[j - 1] == b'_')
}

fn write_f64(path: &std::path::Path, xs: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in xs {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}
