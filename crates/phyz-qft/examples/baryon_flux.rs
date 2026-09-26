//! The gluon flux tube of a static baryon: correlate three-quark Wilson loops
//! with the action density and write the vacuum-suppression field.
//!
//! ```sh
//! cargo run -p phyz-qft --release --example baryon_flux -- baryon.qcdf
//! cargo run -p phyz-qft --release --example baryon_flux -- baryon.qcdf --configs=40 --r=6
//! ```
//!
//! The file is rewritten after every configuration so a viewer can watch the
//! signal converge. View with `kosm-qcd baryon.qcdf`.

use phyz_qft::su3::{FieldFile, FluxAccumulator, Su3Lattice};
use std::time::Instant;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::args()
        .find_map(|a| {
            a.strip_prefix(&format!("--{name}="))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(default)
}

fn main() -> std::io::Result<()> {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "baryon.qcdf".into());
    let l: usize = arg("size", 16);
    let nt: usize = arg("nt", l);
    let beta: f64 = arg("beta", 6.0);
    let therm: usize = arg("therm", 60);
    let configs: usize = arg("configs", 20);
    let sep: usize = arg("sep", 10);
    let smear: usize = arg("smear", 10);
    let t_len: usize = arg("T", 4);
    let r: i32 = arg("r", 5);
    let seed: u64 = arg("seed", 3);

    // Three quarks at 120° around the junction, radius ≈ r (exact for r = 5).
    let q = |ang: f64| {
        let a = ang.to_radians();
        [
            (r as f64 * a.cos()).round() as i32,
            (r as f64 * a.sin()).round() as i32,
            0,
        ]
    };
    let quarks = [q(0.0), q(126.87), q(-126.87)];
    eprintln!(
        "{nt}×{l}³ β = {beta}; quarks {quarks:?} (a ≈ {:.3} fm), T = {t_len}, {smear} stout steps",
        lattice_spacing_fm(beta)
    );

    let dims = [nt, l, l, l];
    let t0 = Instant::now();
    let mut lat = Su3Lattice::hot(dims, beta, seed);
    for _ in 0..therm {
        lat.update(4);
    }
    eprintln!(
        "thermalized ⟨P⟩ = {:.5} ({:.0?})",
        lat.average_plaquette(),
        t0.elapsed()
    );

    let mut acc = FluxAccumulator::new([l, l, l], quarks, t_len, 2);
    for c in 0..configs {
        for _ in 0..sep {
            lat.update(4);
        }
        let mut smooth = lat.clone();
        smooth.stout_smear(0.1, smear);
        let fs = smooth.field_strength();
        let action = fs.action_density();
        let [ex, ey, ez] = fs.electric_sq();
        let electric: Vec<f64> = (0..action.len()).map(|i| ex[i] + ey[i] + ez[i]).collect();
        acc.add(&smooth, &[&action, &electric]);
        eprintln!(
            "config {:>3}/{configs}  ⟨W₃Q⟩ = {:.4}  ({:.0?})",
            c + 1,
            acc.mean_loop(),
            t0.elapsed()
        );
        write(&out, &acc, l, beta)?;
    }
    eprintln!("wrote {out}");
    Ok(())
}

/// Recentre the offset grid so the junction sits mid-box, and store
/// `C(r)` (≈ 1 far away, < 1 inside the flux tube) for each density.
fn write(path: &str, acc: &FluxAccumulator, l: usize, beta: f64) -> std::io::Result<()> {
    let h = l / 2;
    let shift = |c: Vec<f64>| -> Vec<f64> {
        let mut out = vec![0.0; c.len()];
        for z in 0..l {
            for y in 0..l {
                for x in 0..l {
                    let src = ((x + h) % l) + l * (((y + h) % l) + l * ((z + h) % l));
                    out[x + l * (y + l * z)] = c[src];
                }
            }
        }
        out
    };
    let action = shift(acc.correlation(0));
    let electric = shift(acc.correlation(1));
    FieldFile {
        dims: [1, l, l, l],
        a_fm: lattice_spacing_fm(beta),
        fields: vec![("action", &action), ("electric", &electric)],
        quarks: acc
            .quarks
            .iter()
            .map(|q| q.map(|c| (c + h as i32) as f64))
            .collect(),
    }
    .write(path)
}

/// Wilson-action lattice spacing from the Necco-Sommer r₀ fit, r₀ = 0.5 fm.
fn lattice_spacing_fm(beta: f64) -> f64 {
    let x = beta - 6.0;
    0.5 * (-1.6804 - 1.7331 * x + 0.7849 * x * x - 0.4428 * x * x * x).exp()
}
