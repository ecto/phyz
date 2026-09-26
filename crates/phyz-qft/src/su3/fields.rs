//! Smoothing and gluon field observables: stout smearing, the clover field
//! strength, and per-site action and topological charge densities.
//!
//! All densities are in lattice units (multiply by `a⁻⁴` for physical units).
//! The field strength is anti-Hermitian, `F_μν = TA(Q_μν) / 4`, where `Q_μν` is
//! the sum of the four plaquette leaves around a site and `TA` takes the
//! traceless anti-Hermitian part.

use super::lattice::Su3Lattice;
use super::mat::Su3;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Plane pairs `(μ, ν)` with μ < ν, in the order [`FieldStrength::f`] stores them.
pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Clover field strength at every site: `f[site][plane]` for the six planes
/// in [`PLANES`].
pub struct FieldStrength {
    pub f: Vec<[Su3; 6]>,
}

fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

impl Su3Lattice {
    /// One stout smearing step (Morningstar-Peardon) on every link at once:
    /// `U' = exp(−ρ · TA(U A)) U`, with `A` the staple sum. Each step moves the
    /// field a flow time of about `ρ` toward the classical minimum, stripping
    /// UV noise while keeping topology.
    pub fn stout_step(&mut self, rho: f64) {
        let n = self.n_sites();
        let new: [Vec<Su3>; 4] =
            std::array::from_fn(|mu| par_map(n, |site| self.stout_link(site, mu, rho)));
        self.links = new;
    }

    /// The stout-smeared value of one link, from the current (unsmeared)
    /// links. Lets callers spread a smearing step over several frames.
    pub fn stout_link(&self, site: usize, mu: usize, rho: f64) -> Su3 {
        let u = self.links[mu][site];
        let x = (u * self.staple(site, mu)).traceless_antihermitian();
        Su3::exp_antihermitian(&x.scale(-rho)) * u
    }

    /// `n` stout steps of size `rho`.
    pub fn stout_smear(&mut self, rho: f64, n: usize) {
        for _ in 0..n {
            self.stout_step(rho);
        }
    }

    /// Sum of the four plaquette leaves in the (μ, ν) plane touching `site`.
    pub fn clover_leaves(&self, site: usize, mu: usize, nu: usize) -> Su3 {
        let u = |d: usize, s: usize| self.links[d][s];
        let (f, b) = (|s, d| self.fwd(s, d), |s, d| self.bwd(s, d));

        let n_mu = f(site, mu);
        let n_nu = f(site, nu);
        let n_mmu = b(site, mu);
        let n_mnu = b(site, nu);
        let n_mmu_nu = b(n_nu, mu);
        let n_mmu_mnu = b(n_mmu, nu);
        let n_mu_mnu = f(n_mnu, mu);

        // U_μ(n) U_ν(n+μ) U_μ†(n+ν) U_ν†(n)
        let l1 = (u(mu, site) * u(nu, n_mu)).mul_dag(&(u(nu, site) * u(mu, n_nu)));
        // U_ν(n) U_μ†(n+ν−μ) U_ν†(n−μ) U_μ(n−μ)
        let l2 = u(nu, site).mul_dag(&u(mu, n_mmu_nu)).mul_dag(&u(nu, n_mmu)) * u(mu, n_mmu);
        // U_μ†(n−μ) U_ν†(n−μ−ν) U_μ(n−μ−ν) U_ν(n−ν)
        let l3 = (u(nu, n_mmu_mnu) * u(mu, n_mmu)).dag_mul(&(u(mu, n_mmu_mnu) * u(nu, n_mnu)));
        // U_ν†(n−ν) U_μ(n−ν) U_ν(n−ν+μ) U_μ†(n)
        let l4 = u(nu, n_mnu).dag_mul(&u(mu, n_mnu)) * u(nu, n_mu_mnu).mul_dag(&u(mu, site));

        l1 + l2 + l3 + l4
    }

    /// Clover-averaged field strength on every site.
    pub fn field_strength(&self) -> FieldStrength {
        FieldStrength {
            f: par_map(self.n_sites(), |site| self.field_strength_at(site)),
        }
    }

    /// Clover field strength at one site, in [`PLANES`] order.
    pub fn field_strength_at(&self, site: usize) -> [Su3; 6] {
        PLANES.map(|(mu, nu)| {
            self.clover_leaves(site, mu, nu)
                .traceless_antihermitian()
                .scale(0.25)
        })
    }
}

impl FieldStrength {
    /// Action density `E(x) = −Σ_{μ<ν} Tr F_μν²` (= ¼ G^a_μν G^a_μν), ≥ 0.
    pub fn action_density(&self) -> Vec<f64> {
        self.f
            .iter()
            .map(|fs| fs.iter().map(|f| -(*f * *f).re_tr()).sum())
            .collect()
    }

    /// Topological charge density
    /// `q(x) = (1/32π²) ε_μνρσ Tr F_μν F_ρσ`, which sums to an integer `Q` on
    /// smooth fields.
    pub fn topological_charge_density(&self) -> Vec<f64> {
        // ε expands to 8 (F01F23 − F02F13 + F03F12); plane indices from PLANES.
        let norm = 8.0 / (32.0 * std::f64::consts::PI * std::f64::consts::PI);
        self.f
            .iter()
            .map(|f| {
                let t = |a: usize, b: usize| (f[a] * f[b]).re_tr();
                norm * (t(0, 5) - t(1, 4) + t(2, 3))
            })
            .collect()
    }

    /// Chromo-electric energy density per spatial axis, `−Tr F_0i²` for
    /// i = x, y, z. Their sum is the electric half of the action density.
    pub fn electric_sq(&self) -> [Vec<f64>; 3] {
        std::array::from_fn(|i| self.f.iter().map(|f| -(f[i] * f[i]).re_tr()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_fields_vanish() {
        let lat = Su3Lattice::cold([4, 4, 4, 4], 6.0, 0);
        let fs = lat.field_strength();
        assert!(fs.action_density().iter().all(|&e| e.abs() < 1e-24));
        assert!(
            fs.topological_charge_density()
                .iter()
                .all(|&q| q.abs() < 1e-24)
        );
    }

    #[test]
    fn clover_leaves_are_one_plaquette_each() {
        // Each leaf is a closed plaquette loop, so its trace equals that of the
        // corresponding forward plaquette (cyclicity), leaf by leaf.
        let lat = Su3Lattice::hot([4, 4, 4, 4], 6.0, 1);
        let (site, mu, nu) = (53, 1, 3);
        let q = lat.clover_leaves(site, mu, nu);
        let want = lat.plaquette(site, mu, nu).re_tr()
            + lat.plaquette(lat.bwd(site, mu), mu, nu).re_tr()
            + lat
                .plaquette(lat.bwd(lat.bwd(site, mu), nu), mu, nu)
                .re_tr()
            + lat.plaquette(lat.bwd(site, nu), mu, nu).re_tr();
        assert!((q.re_tr() - want).abs() < 1e-10);
    }

    #[test]
    fn stout_smearing_lowers_action() {
        let mut lat = Su3Lattice::hot([4, 4, 4, 4], 6.0, 2);
        for _ in 0..10 {
            lat.update(2);
        }
        let mut last = lat.action();
        for _ in 0..5 {
            lat.stout_step(0.1);
            let s = lat.action();
            assert!(s < last, "{s} ≥ {last}");
            last = s;
        }
    }

    /// Smoothed configurations carry near-integer topological charge.
    /// `cargo test -p phyz-qft --release -- --ignored topological_charge`
    #[test]
    #[ignore = "about a minute in release"]
    fn topological_charge_is_near_integer() {
        let mut lat = Su3Lattice::hot([12, 12, 12, 12], 6.0, 7);
        for _ in 0..60 {
            lat.update(4);
        }
        let mut qs = Vec::new();
        for _ in 0..8 {
            for _ in 0..20 {
                lat.update(4);
            }
            let mut smooth = lat.clone();
            smooth.stout_smear(0.1, 60);
            let q: f64 = smooth
                .field_strength()
                .topological_charge_density()
                .iter()
                .sum();
            eprintln!("Q = {q:+.3}");
            qs.push(q);
        }
        // Clover Q on a 0.1 fm lattice lands within ~0.15 of an integer after
        // heavy smoothing; demand it of most configurations.
        let near = qs.iter().filter(|q| (*q - q.round()).abs() < 0.2).count();
        assert!(near >= 6, "{qs:?}");
        // A (1.2 fm)⁴ box has ⟨Q²⟩ ≈ 1–2, so a correctly normalized density
        // must show nonzero sectors too, not just Q = 0.
        assert!(qs.iter().any(|q| q.round() != 0.0), "{qs:?}");
    }
}
