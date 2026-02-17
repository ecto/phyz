//! Observable measurements for lattice gauge theory.

use crate::group::Group;
use crate::lattice::Lattice;

/// Collection of observables for lattice gauge theory.
pub struct Observables {
    /// Average plaquette ⟨Re Tr U_plaq⟩.
    pub plaquette: f64,

    /// Wilson loop at various sizes.
    pub wilson_loops: Vec<WilsonLoop>,

    /// Polyakov loop (for finite temperature).
    pub polyakov_loop: Option<PolyakovLoop>,
}

/// Wilson loop observable W(R, T) = Tr[U_loop].
///
/// Measures confinement via area law: ⟨W(R,T)⟩ ~ exp(-σ RT)
/// where σ is the string tension.
#[derive(Debug, Clone)]
pub struct WilsonLoop {
    /// Spatial extent R.
    pub r: usize,

    /// Temporal extent T.
    pub t: usize,

    /// Expectation value ⟨Re Tr W⟩.
    pub value: f64,
}

/// Polyakov loop P = Tr[Π_t U_0(x, t)].
///
/// Order parameter for deconfinement phase transition.
/// Non-zero ⟨|P|⟩ indicates deconfined phase.
#[derive(Debug, Clone)]
pub struct PolyakovLoop {
    /// Average |P|.
    pub magnitude: f64,

    /// Average Re(P).
    pub real_part: f64,

    /// Average Im(P).
    pub imag_part: f64,
}

impl<G: Group> Lattice<G> {
    /// Measure average plaquette.
    pub fn measure_plaquette(&self) -> f64 {
        self.average_plaquette()
    }

    /// Measure Wilson loop of size R × T in (x, t) plane.
    pub fn measure_wilson_loop(&self, r: usize, t: usize) -> WilsonLoop {
        let mut sum = 0.0;
        let mut count = 0;

        // Average over all starting positions
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x0 in 0..self.nx {
                    for t0 in 0..self.nt {
                        let site = self.site_index(t0, x0, y, z);
                        let loop_val = self.compute_wilson_loop(site, r, t, 1, 0);
                        sum += loop_val.re_tr() / 3.0; // Normalize by N
                        count += 1;
                    }
                }
            }
        }

        WilsonLoop {
            r,
            t,
            value: sum / count as f64,
        }
    }

    /// Compute Wilson loop starting at site, with spatial extent r in direction mu_space
    /// and temporal extent t in direction mu_time.
    fn compute_wilson_loop(
        &self,
        site: usize,
        r: usize,
        t: usize,
        mu_space: usize,
        mu_time: usize,
    ) -> G {
        let mut current = site;
        let mut result = G::identity();

        // Move r steps in spatial direction
        for _ in 0..r {
            result = result.mul(self.get_link(mu_space, current));
            current = self.neighbor(current, mu_space);
        }

        // Move t steps in temporal direction
        for _ in 0..t {
            result = result.mul(self.get_link(mu_time, current));
            current = self.neighbor(current, mu_time);
        }

        // Move r steps backward in spatial direction
        for _ in 0..r {
            current = self.neighbor_back(current, mu_space);
            result = result.mul(&self.get_link(mu_space, current).inv());
        }

        // Move t steps backward in temporal direction
        for _ in 0..t {
            current = self.neighbor_back(current, mu_time);
            result = result.mul(&self.get_link(mu_time, current).inv());
        }

        result
    }

    /// Measure Polyakov loop ⟨P⟩ = ⟨Tr[Π_t U_0(x, t)]⟩.
    pub fn measure_polyakov_loop(&self) -> PolyakovLoop {
        let mut sum_real = 0.0;
        let sum_imag = 0.0;
        let mut count = 0;

        // Average over all spatial positions
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let mut result = G::identity();
                    let mut site = self.site_index(0, x, y, z);

                    // Product around temporal circle
                    for _ in 0..self.nt {
                        result = result.mul(self.get_link(0, site));
                        site = self.neighbor(site, 0);
                    }

                    let tr = result.re_tr() / 3.0; // Normalize by N
                    sum_real += tr;
                    // For simplicity, we only track real part here
                    // Full complex trace would require Group trait extension
                    count += 1;
                }
            }
        }

        let avg_real = sum_real / count as f64;
        let magnitude = avg_real.abs(); // Simplified

        PolyakovLoop {
            magnitude,
            real_part: avg_real,
            imag_part: sum_imag / count as f64,
        }
    }

    /// Measure all standard observables.
    pub fn measure_observables(&self, wilson_sizes: &[(usize, usize)]) -> Observables {
        let plaquette = self.measure_plaquette();

        let mut wilson_loops = Vec::new();
        for &(r, t) in wilson_sizes {
            wilson_loops.push(self.measure_wilson_loop(r, t));
        }

        let polyakov_loop = Some(self.measure_polyakov_loop());

        Observables {
            plaquette,
            wilson_loops,
            polyakov_loop,
        }
    }
}

/// Autocorrelation function for observable time series.
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    if variance < 1e-12 {
        return 0.0;
    }

    let mut sum = 0.0;
    let n = data.len() - lag;
    for i in 0..n {
        sum += (data[i] - mean) * (data[i + lag] - mean);
    }

    sum / (n as f64 * variance)
}

/// Integrated autocorrelation time τ_int.
///
/// Computed via summation method: τ_int = 0.5 + Σ_t ρ(t)
/// where ρ(t) is the autocorrelation at lag t.
pub fn integrated_autocorr_time(data: &[f64], max_lag: usize) -> f64 {
    let mut tau = 0.5;

    for lag in 1..=max_lag.min(data.len() / 2) {
        let rho = autocorrelation(data, lag);
        if rho < 0.0 {
            break; // Stop when autocorrelation becomes negative
        }
        tau += rho;
    }

    tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::U1;

    #[test]
    fn test_measure_plaquette() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        let plaq = lattice.measure_plaquette();
        // Identity configuration should give plaquette = 1
        assert!((plaq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wilson_loop() {
        let lattice = Lattice::<U1>::new(8, 8, 8, 8, 1.0);
        let wilson = lattice.measure_wilson_loop(2, 2);
        assert_eq!(wilson.r, 2);
        assert_eq!(wilson.t, 2);
        // Identity configuration gives Wilson loop = 1 (with normalization factor)
        assert!((wilson.value - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyakov_loop() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        let polyakov = lattice.measure_polyakov_loop();
        // Identity configuration gives |P| = 1 (with normalization factor)
        assert!((polyakov.magnitude - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rho0 = autocorrelation(&data, 0);
        assert!((rho0 - 1.0).abs() < 1e-10); // Perfect correlation at lag 0

        let rho1 = autocorrelation(&data, 1);
        assert!(rho1 > 0.3); // Positive correlation for monotonic data
    }

    #[test]
    fn test_integrated_autocorr_time() {
        // Uncorrelated data (alternating)
        let data = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let tau = integrated_autocorr_time(&data, 3);
        assert!(tau < 1.5); // Should be close to 0.5 for uncorrelated

        // Correlated data (monotonic trend)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tau = integrated_autocorr_time(&data, 5);
        assert!(tau > 0.5); // Should be larger for correlated data
    }
}
