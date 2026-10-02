//! SU(3) pure-gauge lattice with Cabibbo-Marinari heatbath and overrelaxation.
//!
//! Wilson action `S = β Σ_P (1 − Re Tr U_P / 3)`, β = 6/g². Updates sweep one
//! direction and one checkerboard parity at a time: every link in such a set
//! has a staple built only from links outside the set, so the whole set updates
//! in parallel. Each link draws from its own RNG stream keyed by
//! `(seed, sweep, μ, site)`, so results are bit-identical with or without the
//! `parallel` feature.

use super::mat::{C64, Quat, SUBGROUPS, Su3, left_mul_sub, sub_block};
use super::rng::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct Su3Lattice {
    /// Extents `[nt, nx, ny, nz]`; direction 0 is Euclidean time.
    pub dims: [usize; 4],
    pub beta: f64,
    /// `links[μ][site]`, site = t + nt (x + nx (y + ny z)).
    pub links: [Vec<Su3>; 4],
    fwd: Vec<[usize; 4]>,
    bwd: Vec<[usize; 4]>,
    /// Sites of each checkerboard parity, `(t + x + y + z) % 2`.
    parity_sites: [Vec<usize>; 2],
    seed: u64,
    /// Counts update passes so each gets fresh random streams.
    pass: u64,
}

impl Su3Lattice {
    /// Cold start: every link is the identity.
    pub fn cold(dims: [usize; 4], beta: f64, seed: u64) -> Self {
        assert!(
            dims.iter().all(|&d| d >= 2 && d % 2 == 0),
            "checkerboard updates need even extents ≥ 2, got {dims:?}"
        );
        let n = dims.iter().product::<usize>();
        let mut fwd = vec![[0; 4]; n];
        let mut bwd = vec![[0; 4]; n];
        for site in 0..n {
            let c = coords(dims, site);
            for mu in 0..4 {
                let mut f = c;
                f[mu] = (c[mu] + 1) % dims[mu];
                fwd[site][mu] = index(dims, f);
                let mut b = c;
                b[mu] = (c[mu] + dims[mu] - 1) % dims[mu];
                bwd[site][mu] = index(dims, b);
            }
        }
        let parity_sites = std::array::from_fn(|p| {
            (0..n)
                .filter(|&s| coords(dims, s).iter().sum::<usize>() % 2 == p)
                .collect()
        });
        Self {
            dims,
            beta,
            links: std::array::from_fn(|_| vec![Su3::IDENTITY; n]),
            fwd,
            bwd,
            parity_sites,
            seed,
            pass: 0,
        }
    }

    /// Hot start: links drawn uniformly (Haar) from SU(3).
    pub fn hot(dims: [usize; 4], beta: f64, seed: u64) -> Self {
        let mut lat = Self::cold(dims, beta, seed);
        let n = lat.n_sites();
        for mu in 0..4 {
            for site in 0..n {
                let mut rng = Rng::stream(seed ^ 0x9e37_79b9, u64::MAX, mu, site);
                lat.links[mu][site] = random_su3(&mut rng);
            }
        }
        lat
    }

    #[inline]
    pub fn n_sites(&self) -> usize {
        self.fwd.len()
    }

    #[inline]
    pub fn coords(&self, site: usize) -> [usize; 4] {
        coords(self.dims, site)
    }

    #[inline]
    pub fn index(&self, c: [usize; 4]) -> usize {
        index(self.dims, c)
    }

    #[inline]
    pub fn fwd(&self, site: usize, mu: usize) -> usize {
        self.fwd[site][mu]
    }

    #[inline]
    pub fn bwd(&self, site: usize, mu: usize) -> usize {
        self.bwd[site][mu]
    }

    #[inline]
    pub fn link(&self, mu: usize, site: usize) -> &Su3 {
        &self.links[mu][site]
    }

    /// Plaquette `U_μ(n) U_ν(n+μ) U_μ†(n+ν) U_ν†(n)`.
    pub fn plaquette(&self, site: usize, mu: usize, nu: usize) -> Su3 {
        let a = self.links[mu][site] * self.links[nu][self.fwd(site, mu)];
        let b = self.links[nu][site] * self.links[mu][self.fwd(site, nu)];
        a.mul_dag(&b)
    }

    /// Sum of the six staples around `U_μ(n)`, oriented so that the local
    /// action is `−(β/3) Re Tr(U_μ(n) A)`.
    pub fn staple(&self, site: usize, mu: usize) -> Su3 {
        let n_mu = self.fwd(site, mu);
        let mut a = Su3::ZERO;
        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            // Upper: U_ν(n+μ) U_μ†(n+ν) U_ν†(n)
            let n_nu = self.fwd(site, nu);
            a += self.links[nu][n_mu]
                .mul_dag(&self.links[mu][n_nu])
                .mul_dag(&self.links[nu][site]);

            // Lower: U_ν†(n+μ−ν) U_μ†(n−ν) U_ν(n−ν)
            let n_mnu = self.bwd(site, nu);
            let n_mu_mnu = self.bwd(n_mu, nu);
            a += self.links[nu][n_mu_mnu].dag_mul(&self.links[mu][n_mnu].dagger())
                * self.links[nu][n_mnu];
        }
        a
    }

    /// `⟨Re Tr U_P⟩ / 3` averaged over all sites and the six planes.
    pub fn average_plaquette(&self) -> f64 {
        let per_site = |site: usize| {
            let mut s = 0.0;
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    s += self.plaquette(site, mu, nu).re_tr();
                }
            }
            s
        };
        let n = self.n_sites();
        #[cfg(feature = "parallel")]
        let sum: f64 = (0..n).into_par_iter().map(per_site).sum();
        #[cfg(not(feature = "parallel"))]
        let sum: f64 = (0..n).map(per_site).sum();
        sum / (n as f64 * 6.0 * 3.0)
    }

    /// Wilson action `β Σ_P (1 − Re Tr U_P / 3)`.
    pub fn action(&self) -> f64 {
        self.beta * 6.0 * self.n_sites() as f64 * (1.0 - self.average_plaquette())
    }

    /// Spatially averaged Polyakov loop `⟨Tr Π_t U_0(t, x)⟩ / 3`.
    pub fn polyakov_loop(&self) -> C64 {
        let [nt, nx, ny, nz] = self.dims;
        let mut sum = C64::ZERO;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut site = self.index([0, x, y, z]);
                    let mut p = Su3::IDENTITY;
                    for _ in 0..nt {
                        p = p * self.links[0][site];
                        site = self.fwd(site, 0);
                    }
                    sum += p.trace();
                }
            }
        }
        sum.scale(1.0 / (3.0 * (nx * ny * nz) as f64))
    }

    /// One heatbath sweep over every link.
    pub fn heatbath_sweep(&mut self) {
        self.sweep(Update::Heatbath);
    }

    /// One microcanonical overrelaxation sweep (action-preserving).
    pub fn overrelax_sweep(&mut self) {
        self.sweep(Update::Overrelax);
    }

    /// The usual compound update: one heatbath then `n_or` overrelaxation sweeps.
    pub fn update(&mut self, n_or: usize) {
        self.heatbath_sweep();
        for _ in 0..n_or {
            self.overrelax_sweep();
        }
    }

    fn sweep(&mut self, kind: Update) {
        for mu in 0..4 {
            for parity in 0..2 {
                self.half_sweep(mu, parity, kind);
            }
        }
    }

    /// Update only the direction-`mu` links of one checkerboard `parity`: one
    /// eighth of a sweep. Callers that must stay responsive (a browser frame)
    /// can interleave these; eight in the order μ = 0..4, parity = 0..2 equal
    /// one [`Self::heatbath_sweep`] or [`Self::overrelax_sweep`].
    pub fn half_sweep_heatbath(&mut self, mu: usize, parity: usize) {
        self.half_sweep(mu, parity, Update::Heatbath);
    }

    /// Overrelaxation counterpart of [`Self::half_sweep_heatbath`].
    pub fn half_sweep_overrelax(&mut self, mu: usize, parity: usize) {
        self.half_sweep(mu, parity, Update::Overrelax);
    }

    fn half_sweep(&mut self, mu: usize, parity: usize, kind: Update) {
        let pass = self.pass;
        self.pass += 1;
        let this = &*self;
        let sites = &this.parity_sites[parity];
        let new_link = |&site: &usize| {
            let a = this.staple(site, mu);
            let mut rng = Rng::stream(this.seed, pass, mu, site);
            (
                site,
                update_link(this.links[mu][site], &a, this.beta, kind, &mut rng),
            )
        };
        #[cfg(feature = "parallel")]
        let updated: Vec<(usize, Su3)> = sites.par_iter().map(new_link).collect();
        #[cfg(not(feature = "parallel"))]
        let updated: Vec<(usize, Su3)> = sites.iter().map(new_link).collect();
        for (site, u) in updated {
            self.links[mu][site] = u;
        }
    }
}

#[derive(Clone, Copy)]
enum Update {
    Heatbath,
    Overrelax,
}

/// Cabibbo-Marinari: update `u` through its three SU(2) subgroups in turn.
fn update_link(mut u: Su3, a: &Su3, beta: f64, kind: Update, rng: &mut Rng) -> Su3 {
    let mut w = u * *a;
    for sg in SUBGROUPS {
        let v = Quat::project(sub_block(&w, sg));
        let k = v.norm();
        let x = if k < 1e-12 {
            random_su2(rng)
        } else {
            let v_hat = Quat(v.0.map(|c| c / k));
            match kind {
                // Sample X ∝ exp((β/3) Re Tr(X w)) = exp((2βk/3) (X v̂)_0).
                Update::Heatbath => kp_su2(2.0 * beta * k / 3.0, rng).mul(&v_hat.conj()),
                // X = v̂†² reflects through v̂, keeping Re Tr(X w) fixed.
                Update::Overrelax => v_hat.conj().mul(&v_hat.conj()),
            }
        };
        let xm = x.to_mat();
        left_mul_sub(&mut u, sg, &xm);
        left_mul_sub(&mut w, sg, &xm);
    }
    u.reunitarize();
    u
}

/// Kennedy-Pendleton: sample SU(2) `y` with density ∝ exp(α y₀) on S³.
fn kp_su2(alpha: f64, rng: &mut Rng) -> Quat {
    let y0 = loop {
        let r1 = 1.0 - rng.uniform();
        let r2 = rng.uniform();
        let r3 = 1.0 - rng.uniform();
        let c = (2.0 * std::f64::consts::PI * r2).cos();
        let lambda2 = -(r1.ln() + c * c * r3.ln()) / (2.0 * alpha);
        let r4 = rng.uniform();
        if r4 * r4 <= 1.0 - lambda2 {
            break 1.0 - 2.0 * lambda2;
        }
    };
    let [y1, y2, y3] = unit_vec3(rng).map(|c| c * (1.0 - y0 * y0).max(0.0).sqrt());
    Quat([y0, y1, y2, y3])
}

fn unit_vec3(rng: &mut Rng) -> [f64; 3] {
    let cos_t = 2.0 * rng.uniform() - 1.0;
    let sin_t = (1.0 - cos_t * cos_t).sqrt();
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    [sin_t * phi.cos(), sin_t * phi.sin(), cos_t]
}

fn random_su2(rng: &mut Rng) -> Quat {
    let q = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
    let n = q.iter().map(|a| a * a).sum::<f64>().sqrt();
    Quat(q.map(|a| a / n))
}

/// Haar-random SU(3): Gram-Schmidt a Gaussian complex matrix, then fix the
/// determinant phase through the cross-product third row.
pub(crate) fn random_su3(rng: &mut Rng) -> Su3 {
    let mut m = Su3::ZERO;
    for row in &mut m.m {
        for e in row {
            *e = C64::new(rng.normal(), rng.normal());
        }
    }
    m.reunitarize();
    m
}

#[inline]
fn coords(dims: [usize; 4], site: usize) -> [usize; 4] {
    let [nt, nx, ny, _] = dims;
    [
        site % nt,
        (site / nt) % nx,
        (site / (nt * nx)) % ny,
        site / (nt * nx * ny),
    ]
}

#[inline]
fn index(dims: [usize; 4], [t, x, y, z]: [usize; 4]) -> usize {
    let [nt, nx, ny, _] = dims;
    t + nt * (x + nx * (y + ny * z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staple_matches_plaquette_action() {
        // Re Tr(U A) must equal the sum of Re Tr over the six plaquettes
        // containing U, whatever orientation they come in.
        let lat = Su3Lattice::hot([4, 4, 4, 4], 6.0, 1);
        let (site, mu) = (37, 2);
        let lhs = (lat.links[mu][site] * lat.staple(site, mu)).re_tr();
        let mut rhs = 0.0;
        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            rhs += lat.plaquette(site, mu, nu).re_tr();
            rhs += lat.plaquette(lat.bwd(site, nu), mu, nu).re_tr();
        }
        assert!((lhs - rhs).abs() < 1e-10, "{lhs} vs {rhs}");
    }

    #[test]
    fn overrelaxation_preserves_action() {
        let mut lat = Su3Lattice::hot([4, 4, 4, 4], 6.0, 2);
        lat.heatbath_sweep();
        let s0 = lat.action();
        lat.overrelax_sweep();
        let s1 = lat.action();
        assert!((s0 - s1).abs() / s0 < 1e-10, "{s0} → {s1}");
    }

    #[test]
    fn links_stay_in_su3() {
        let mut lat = Su3Lattice::hot([4, 4, 4, 4], 5.7, 3);
        lat.update(2);
        for mu in 0..4 {
            for u in &lat.links[mu] {
                assert!(u.mul_dag(u).dist_sqr(&Su3::IDENTITY) < 1e-20);
                assert!((u.det() - C64::ONE).norm_sqr() < 1e-20);
            }
        }
    }

    #[test]
    fn cold_start_is_ordered() {
        let lat = Su3Lattice::cold([4, 4, 4, 4], 6.0, 0);
        assert!((lat.average_plaquette() - 1.0).abs() < 1e-14);
        assert!((lat.polyakov_loop().re - 1.0).abs() < 1e-14);
    }

    #[test]
    fn strong_coupling_plaquette() {
        // Leading strong-coupling result ⟨P⟩ = β/18 + O(β²); at β = 0.5 the
        // correction is ~1 %.
        let mut lat = Su3Lattice::hot([4, 4, 4, 4], 0.5, 4);
        for _ in 0..20 {
            lat.update(0);
        }
        let mut sum = 0.0;
        for _ in 0..40 {
            lat.update(0);
            sum += lat.average_plaquette();
        }
        let p = sum / 40.0;
        assert!((p - 0.5 / 18.0).abs() < 0.004, "⟨P⟩ = {p}");
    }

    #[test]
    fn deterministic_per_seed() {
        let mut a = Su3Lattice::hot([4, 4, 4, 4], 6.0, 9);
        let mut b = Su3Lattice::hot([4, 4, 4, 4], 6.0, 9);
        a.update(1);
        b.update(1);
        assert_eq!(a.average_plaquette(), b.average_plaquette());
    }

    /// Reference value ⟨P⟩ = 0.5937 at β = 6.0 (Wilson action, large volume).
    /// `cargo test -p phyz-qft --release -- --ignored plaquette_beta6`
    #[test]
    #[ignore = "a few seconds in release"]
    fn plaquette_beta6() {
        let mut lat = Su3Lattice::hot([8, 8, 8, 8], 6.0, 5);
        for _ in 0..100 {
            lat.update(4);
        }
        let mut sum = 0.0;
        let n = 100;
        for _ in 0..n {
            lat.update(4);
            sum += lat.average_plaquette();
        }
        let p = sum / n as f64;
        eprintln!("β=6.0 8⁴ ⟨P⟩ = {p:.5}");
        assert!((p - 0.5937).abs() < 0.002, "⟨P⟩ = {p}");
    }
}
