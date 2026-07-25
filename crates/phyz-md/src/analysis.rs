//! Structural analysis of MD configurations.
//!
//! Currently the radial distribution function, which is the standard way to
//! check that a simulated fluid actually has the right structure: `g(r)` for a
//! Lennard-Jones fluid at a given reduced density and temperature is a
//! published, reproducible curve, so comparing against it validates the
//! potential, the neighbor list, the periodic boundaries, and the integrator
//! all at once.

use crate::field::cell::{Lattice, min_image, vec3};

/// A radial distribution function histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Rdf {
    /// Bin centers in Å.
    pub r: Vec<f64>,
    /// `g(r)` at each bin center.
    pub g: Vec<f64>,
    /// Number of configurations accumulated.
    pub samples: usize,
    bins: usize,
    r_max: f64,
    counts: Vec<f64>,
    n_atoms: usize,
    volume: f64,
}

impl Rdf {
    /// A histogram with `bins` bins out to `r_max` Å.
    pub fn new(bins: usize, r_max: f64) -> Self {
        let bins = bins.max(1);
        let dr = r_max / bins as f64;
        Self {
            r: (0..bins).map(|i| (i as f64 + 0.5) * dr).collect(),
            g: vec![0.0; bins],
            samples: 0,
            bins,
            r_max,
            counts: vec![0.0; bins],
            n_atoms: 0,
            volume: 0.0,
        }
    }

    /// Accumulate one configuration.
    ///
    /// `r_max` must not exceed half the smallest box width, or the
    /// minimum-image convention counts some separations twice.
    pub fn accumulate(&mut self, positions: &[[f64; 3]], cell: &Lattice) {
        let n = positions.len();
        let dr = self.r_max / self.bins as f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = min_image(vec3::sub(positions[i], positions[j]), Some(cell));
                let r = vec3::norm(d);
                if r < self.r_max {
                    let b = (r / dr) as usize;
                    if b < self.bins {
                        // Each pair contributes to both atoms' shells.
                        self.counts[b] += 2.0;
                    }
                }
            }
        }
        self.samples += 1;
        self.n_atoms = n;
        self.volume = cell.volume().abs();
    }

    /// Normalize the histogram into `g(r)`.
    ///
    /// Each bin is divided by the number of pairs an ideal gas of the same
    /// density would put in that spherical shell.
    pub fn finish(&mut self) -> &[f64] {
        if self.samples == 0 || self.n_atoms == 0 || self.volume <= 0.0 {
            return &self.g;
        }
        let dr = self.r_max / self.bins as f64;
        let rho = self.n_atoms as f64 / self.volume;
        for b in 0..self.bins {
            let r_lo = b as f64 * dr;
            let r_hi = r_lo + dr;
            let shell = 4.0 / 3.0 * std::f64::consts::PI * (r_hi.powi(3) - r_lo.powi(3));
            let ideal = rho * shell * self.n_atoms as f64 * self.samples as f64;
            self.g[b] = if ideal > 0.0 {
                self.counts[b] / ideal
            } else {
                0.0
            };
        }
        &self.g
    }

    /// The position and height of the first peak in `g(r)`.
    ///
    /// Call after [`Self::finish`].
    pub fn first_peak(&self) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        for (b, &g) in self.g.iter().enumerate() {
            // Ignore the empty core below the first rise.
            if g < 0.5 {
                if best.is_some() {
                    break;
                }
                continue;
            }
            match best {
                Some((_, gb)) if g <= gb => break,
                _ => best = Some((self.r[b], g)),
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An ideal gas has g(r) = 1 everywhere.
    #[test]
    fn uniform_points_give_g_of_one() {
        let l = 30.0;
        let cell = Lattice::cubic(l);
        let n = 4000;
        // A plain LCG, not a low-discrepancy sequence: quasi-random points are
        // deliberately *not* Poisson-distributed, and their pair correlation
        // would show structure that a real ideal gas does not have.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| [next() * l, next() * l, next() * l])
            .collect();
        let mut rdf = Rdf::new(30, 12.0);
        rdf.accumulate(&positions, &cell);
        let g = rdf.finish();
        // Skip the innermost bins, where the shell volume is tiny and the
        // sampling noise is correspondingly large.
        for (b, &gv) in g.iter().enumerate().skip(8) {
            assert!(
                (gv - 1.0).abs() < 0.2,
                "bin {b} (r = {:.2}) has g = {gv:.3}",
                rdf.r[b]
            );
        }
    }

    #[test]
    fn a_crystal_shows_a_sharp_first_shell() {
        // Simple cubic lattice, spacing 3 Å: the first shell is at exactly 3 Å
        // with 6 neighbors.
        let spacing = 3.0;
        let n_side = 8;
        let mut positions = Vec::new();
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    positions.push([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
                }
            }
        }
        let cell = Lattice::cubic(n_side as f64 * spacing);
        let mut rdf = Rdf::new(120, 10.0);
        rdf.accumulate(&positions, &cell);
        rdf.finish();
        let (r_peak, g_peak) = rdf.first_peak().expect("expected a first peak");
        assert!((r_peak - spacing).abs() < 0.15, "first peak at {r_peak}");
        assert!(g_peak > 5.0, "crystalline peak should be sharp: {g_peak}");
    }
}
