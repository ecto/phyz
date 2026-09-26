//! The QCD vacuum "lava lamp": thermalize an SU(3) lattice, smooth away the UV
//! noise, and write the action and topological charge densities for viewing.
//!
//! ```sh
//! cargo run -p phyz-qft --release --example qcd_vacuum -- vacuum.qcdf
//! cargo run -p phyz-qft --release --example qcd_vacuum -- vacuum.qcdf --size=12 --smear=30
//! ```
//!
//! View with `cargo run -p kosm-qcd --release -- vacuum.qcdf` in kosm.

use phyz_qft::su3::{FieldFile, Su3Lattice};
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
        .unwrap_or_else(|| "vacuum.qcdf".into());
    let l: usize = arg("size", 16);
    let nt: usize = arg("nt", l);
    let beta: f64 = arg("beta", 6.0);
    let therm: usize = arg("therm", 100);
    let smear: usize = arg("smear", 40);
    let rho: f64 = arg("rho", 0.1);
    let seed: u64 = arg("seed", 1);

    let dims = [nt, l, l, l];
    eprintln!(
        "{nt}×{l}³ at β = {beta}, {therm} updates (1 HB + 4 OR), {smear} stout steps ρ = {rho}"
    );
    let t0 = Instant::now();
    let mut lat = Su3Lattice::hot(dims, beta, seed);
    for i in 0..therm {
        lat.update(4);
        if i % 10 == 9 {
            eprintln!(
                "  update {:>4}  ⟨P⟩ = {:.5}  ({:.0?})",
                i + 1,
                lat.average_plaquette(),
                t0.elapsed()
            );
        }
    }

    lat.stout_smear(rho, smear);
    let fs = lat.field_strength();
    let action = fs.action_density();
    let topo = fs.topological_charge_density();
    let q: f64 = topo.iter().sum();
    eprintln!("smoothed ⟨P⟩ = {:.5}, Q = {q:+.3}", lat.average_plaquette());

    FieldFile {
        dims,
        a_fm: lattice_spacing_fm(beta),
        fields: vec![("action", &action), ("topo", &topo)],
        quarks: vec![],
    }
    .write(&out)?;
    eprintln!("wrote {out} in {:.1?}", t0.elapsed());
    Ok(())
}

/// Wilson-action lattice spacing from the Necco-Sommer r₀ fit, r₀ = 0.5 fm
/// (valid 5.7 ≤ β ≤ 6.92).
fn lattice_spacing_fm(beta: f64) -> f64 {
    let x = beta - 6.0;
    let ln_a_over_r0 = -1.6804 - 1.7331 * x + 0.7849 * x * x - 0.4428 * x * x * x;
    0.5 * ln_a_over_r0.exp()
}
