//! Static three-quark (baryon) Wilson loops and the gluon flux they bind.
//!
//! Three quarks sit at fixed spatial offsets from a junction point `x₀`. Each
//! quark line runs from `x₀` along a lattice staircase to its quark at time
//! `t₀`, up the time direction for `T` steps, and back to `x₀` at `t₀ + T`:
//!
//! `M_i = P_i(t₀) · L_i(t₀→t₀+T) · P_i(t₀+T)†`
//!
//! The colour-singlet contraction is `W₃Q = ε_abc ε_a'b'c' M₁_aa' M₂_bb' M₃_cc' / 6`.
//! The flux picture is then the correlation of `W₃Q` with any local density
//! `S(x)` measured on the middle time slice:
//!
//! `C(r) = ⟨W₃Q · S(x₀ + r)⟩ / (⟨W₃Q⟩ ⟨S⟩)`
//!
//! `C < 1` where the quarks expel the vacuum fluctuations, which is the flux
//! tube. Averaging over every `(t₀, x₀)` of a configuration gives the
//! statistics that make this feasible on a laptop.

use super::lattice::Su3Lattice;
use super::mat::{C64, Su3};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Lattice step `(axis, ±1)` in a spatial staircase.
type Step = (usize, bool);

/// Staircase of unit steps from the origin to spatial offset `d` (x, y, z)
/// that stays as close as possible to the straight line.
pub fn staircase(d: [i32; 3]) -> Vec<Step> {
    let n: i32 = d.iter().map(|c| c.abs()).sum();
    let mut pos = [0i32; 3];
    let mut steps = Vec::with_capacity(n as usize);
    for _ in 0..n {
        // take the step that keeps the walker closest to the line 0→d
        let best = (0..3)
            .filter(|&a| pos[a] != d[a])
            .min_by(|&a, &b| {
                let dist = |axis: usize| {
                    let mut p = pos;
                    p[axis] += d[axis].signum();
                    line_dist(p, d)
                };
                dist(a).total_cmp(&dist(b))
            })
            .unwrap();
        pos[best] += d[best].signum();
        steps.push((best + 1, d[best] > 0));
    }
    steps
}

fn line_dist(p: [i32; 3], d: [i32; 3]) -> f64 {
    let (p, d) = (p.map(f64::from), d.map(f64::from));
    let dd = d.iter().map(|c| c * c).sum::<f64>();
    let t = p.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>() / dd;
    p.iter()
        .zip(&d)
        .map(|(a, b)| (a - t * b).powi(2))
        .sum::<f64>()
}

impl Su3Lattice {
    /// Ordered link product along `steps` starting at `site`; returns the
    /// product and the end site. Direction indices are lattice axes (1–3 spatial).
    pub fn path(&self, mut site: usize, steps: &[Step]) -> (Su3, usize) {
        let mut u = Su3::IDENTITY;
        for &(mu, fwd) in steps {
            if fwd {
                u = u * *self.link(mu, site);
                site = self.fwd(site, mu);
            } else {
                site = self.bwd(site, mu);
                u = u.mul_dag(self.link(mu, site));
            }
        }
        (u, site)
    }

    /// Straight time-like line of `t_len` links starting at `site`.
    pub fn time_line(&self, mut site: usize, t_len: usize) -> (Su3, usize) {
        let mut u = Su3::IDENTITY;
        for _ in 0..t_len {
            u = u * *self.link(0, site);
            site = self.fwd(site, 0);
        }
        (u, site)
    }

    /// `Re W₃Q` for the junction at `site` (its time coordinate is `t₀`).
    pub fn baryon_loop(&self, site: usize, paths: &[Vec<Step>; 3], t_len: usize) -> f64 {
        let top = self.time_line(site, t_len).1;
        let m: [Su3; 3] = std::array::from_fn(|i| {
            let (lower, q) = self.path(site, &paths[i]);
            let (line, _) = self.time_line(q, t_len);
            let (upper, _) = self.path(top, &paths[i]);
            (lower * line).mul_dag(&upper)
        });
        epsilon_contract(&m).re / 6.0
    }
}

/// `ε_abc ε_a'b'c' A_aa' B_bb' C_cc'`.
fn epsilon_contract([a, b, c]: &[Su3; 3]) -> C64 {
    const PERMS: [([usize; 3], f64); 6] = [
        ([0, 1, 2], 1.0),
        ([1, 2, 0], 1.0),
        ([2, 0, 1], 1.0),
        ([0, 2, 1], -1.0),
        ([2, 1, 0], -1.0),
        ([1, 0, 2], -1.0),
    ];
    let mut sum = C64::ZERO;
    for (p, sp) in PERMS {
        for (q, sq) in PERMS {
            let t = a.m[p[0]][q[0]] * b.m[p[1]][q[1]] * c.m[p[2]][q[2]];
            sum += t.scale(sp * sq);
        }
    }
    sum
}

/// Running sums for the baryon–density correlation, averaged over
/// configurations and over every junction position in each.
pub struct FluxAccumulator {
    /// Spatial extents `[nx, ny, nz]` of the relative-offset grid.
    pub n: [usize; 3],
    pub quarks: [[i32; 3]; 3],
    pub t_len: usize,
    paths: [Vec<Step>; 3],
    /// `Σ W · S_k(x₀ + r)` per density `k`, per offset `r`.
    ws: Vec<Vec<f64>>,
    /// `Σ S_k` over all sites measured.
    s: Vec<f64>,
    w: f64,
    samples: f64,
}

impl FluxAccumulator {
    /// `quarks` are spatial offsets from the junction; `n_densities` is how many
    /// density fields each call to [`Self::add`] correlates.
    pub fn new(n: [usize; 3], quarks: [[i32; 3]; 3], t_len: usize, n_densities: usize) -> Self {
        Self {
            n,
            quarks,
            t_len,
            paths: quarks.map(staircase),
            ws: vec![vec![0.0; n.iter().product()]; n_densities],
            s: vec![0.0; n_densities],
            w: 0.0,
            samples: 0.0,
        }
    }

    /// Accumulate one configuration. `densities[k][site]` are per-site fields
    /// (e.g. action density) measured on `lat`.
    pub fn add(&mut self, lat: &Su3Lattice, densities: &[&[f64]]) {
        let [nt, nx, ny, nz] = lat.dims;
        assert_eq!([nx, ny, nz], self.n, "offset grid must match the lattice");
        let t_mid = self.t_len / 2;
        let paths = &self.paths;
        let t_len = self.t_len;

        let per_t0 = |t0: usize| {
            let mut ws = vec![vec![0.0; nx * ny * nz]; densities.len()];
            let mut w_sum = 0.0;
            for z0 in 0..nz {
                for y0 in 0..ny {
                    for x0 in 0..nx {
                        let w = lat.baryon_loop(lat.index([t0, x0, y0, z0]), paths, t_len);
                        w_sum += w;
                        let tm = (t0 + t_mid) % nt;
                        for (k, dens) in densities.iter().enumerate() {
                            let out = &mut ws[k];
                            for rz in 0..nz {
                                let z = (z0 + rz) % nz;
                                for ry in 0..ny {
                                    let y = (y0 + ry) % ny;
                                    let row = rx_row(rz, ry, nx, ny);
                                    for rx in 0..nx {
                                        let x = (x0 + rx) % nx;
                                        out[row + rx] += w * dens[lat.index([tm, x, y, z])];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (ws, w_sum)
        };

        #[cfg(feature = "parallel")]
        let parts: Vec<_> = (0..nt).into_par_iter().map(per_t0).collect();
        #[cfg(not(feature = "parallel"))]
        let parts: Vec<_> = (0..nt).map(per_t0).collect();

        for (ws, w) in parts {
            for (acc, part) in self.ws.iter_mut().zip(ws) {
                for (a, p) in acc.iter_mut().zip(part) {
                    *a += p;
                }
            }
            self.w += w;
        }
        for (k, dens) in densities.iter().enumerate() {
            self.s[k] += dens.iter().sum::<f64>();
        }
        self.samples += (nt * nx * ny * nz) as f64;
    }

    /// `⟨W₃Q⟩` so far; should be clearly positive for a usable signal.
    pub fn mean_loop(&self) -> f64 {
        self.w / self.samples
    }

    /// `C_k(r) = ⟨W S_k(x₀+r)⟩ / (⟨W⟩⟨S_k⟩)` on the offset grid, indexed
    /// `rx + nx (ry + ny rz)`; offsets wrap periodically.
    pub fn correlation(&self, k: usize) -> Vec<f64> {
        let w = self.w / self.samples;
        let s = self.s[k] / self.samples;
        self.ws[k]
            .iter()
            .map(|ws| ws / self.samples / (w * s))
            .collect()
    }
}

#[inline]
fn rx_row(rz: usize, ry: usize, nx: usize, ny: usize) -> usize {
    nx * (ry + ny * rz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staircase_reaches_target() {
        for d in [[4, 0, 0], [-3, 4, 0], [-3, -4, 0], [2, -1, 3]] {
            let steps = staircase(d);
            let mut p = [0i32; 3];
            for (axis, fwd) in &steps {
                p[axis - 1] += if *fwd { 1 } else { -1 };
            }
            assert_eq!(p, d);
            assert_eq!(steps.len() as i32, d.iter().map(|c| c.abs()).sum::<i32>());
        }
    }

    #[test]
    fn cold_baryon_loop_is_one() {
        let lat = Su3Lattice::cold([4, 6, 6, 4], 6.0, 0);
        let paths = [[2, 0, 0], [-1, 2, 0], [-1, -2, 0]].map(staircase);
        assert!((lat.baryon_loop(0, &paths, 2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn baryon_loop_is_gauge_invariant() {
        // Random gauge transform g(x): U_μ(x) → g(x) U_μ(x) g(x+μ)†.
        let mut lat = Su3Lattice::hot([4, 6, 6, 4], 6.0, 3);
        let paths = [[2, 0, 0], [-1, 2, 0], [-1, -2, 1]].map(staircase);
        let before = lat.baryon_loop(17, &paths, 3);
        let g: Vec<Su3> = Su3Lattice::hot([4, 6, 6, 4], 6.0, 99).links[0].clone();
        for mu in 0..4 {
            for s in 0..lat.n_sites() {
                let f = lat.fwd(s, mu);
                lat.links[mu][s] = (g[s] * lat.links[mu][s]).mul_dag(&g[f]);
            }
        }
        let after = lat.baryon_loop(17, &paths, 3);
        assert!((before - after).abs() < 1e-10, "{before} vs {after}");
    }

    #[test]
    fn epsilon_of_identity_is_six() {
        let id = Su3::IDENTITY;
        assert!((epsilon_contract(&[id, id, id]).re - 6.0).abs() < 1e-12);
    }
}
