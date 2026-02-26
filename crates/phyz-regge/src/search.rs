//! SVD-based symmetry search.
//!
//! Discovers approximate symmetries of the Einstein-Maxwell action by
//! sampling gradients at many field configurations and finding directions
//! in field space that are (approximately) orthogonal to all gradients.
//!
//! # Algorithm
//!
//! 1. Sample K random field configs around a background (perturb lengths
//!    multiplicatively ×(1 ± ε), phases additively ± ε)
//! 2. Compute gradient g_k = ∇S(φ_k) for each
//! 3. Build matrix M ∈ R^{K × 2n_edges}, rows = g_k
//! 4. Gram-Schmidt the known generators, project them out of each row
//! 5. SVD of projected M → right singular vectors with smallest σ are
//!    candidate approximate symmetries
//! 6. Score each candidate: σ (violation), overlap with known generators

use crate::action::{ActionParams, Fields, einstein_maxwell_grad};
use crate::complex::SimplicialComplex;
use crate::symmetry::{Generator, orthonormalize, project_out_span};

use phyz_math::DMat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for the symmetry search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of random field configurations to sample.
    pub n_samples: usize,
    /// Scale of random perturbations around the background.
    pub perturbation_scale: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            perturbation_scale: 1e-3,
            seed: 42,
        }
    }
}

/// A candidate approximate symmetry found by the search.
#[derive(Debug, Clone)]
pub struct CandidateSymmetry {
    /// Generator direction in packed [δl..., δθ...] form.
    pub generator: Vec<f64>,
    /// Singular value — measures how badly this direction fails to be a symmetry.
    /// Zero means exact symmetry.
    pub violation: f64,
    /// Overlap (dot product) with each known generator: (name, cosine).
    pub overlaps: Vec<(String, f64)>,
}

/// Results from a symmetry search.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Candidate symmetries, ordered by increasing violation (σ).
    pub candidates: Vec<CandidateSymmetry>,
    /// Number of gradient samples used.
    pub n_samples: usize,
    /// Names of known generators that were projected out (or used for overlap).
    pub known_generators: Vec<String>,
}

impl SearchResults {
    /// Human-readable report.
    pub fn report(&self) -> String {
        let mut s = format!(
            "Symmetry search: {} samples, {} known generators\n",
            self.n_samples,
            self.known_generators.len()
        );
        s.push_str(&format!("Found {} candidates:\n", self.candidates.len()));
        for (i, c) in self.candidates.iter().enumerate() {
            s.push_str(&format!("  [{i}] violation={:.2e}", c.violation));
            for (name, overlap) in &c.overlaps {
                if overlap.abs() > 0.1 {
                    s.push_str(&format!(" {name}={overlap:.3}"));
                }
            }
            s.push('\n');
        }
        s
    }

    /// Filter to candidates that are both low-violation and novel
    /// (low overlap with all known generators).
    pub fn novel_candidates(&self, threshold: f64) -> Vec<&CandidateSymmetry> {
        self.candidates
            .iter()
            .filter(|c| c.violation < threshold && c.overlaps.iter().all(|(_, o)| o.abs() < 0.5))
            .collect()
    }
}

/// Search for approximate symmetries via SVD of the gradient matrix.
///
/// Known generators are projected out before the SVD so that the search
/// focuses on novel directions. The `known_generators` are also used to
/// compute overlap scores for each candidate.
pub fn search_symmetries(
    complex: &SimplicialComplex,
    background: &Fields,
    known_generators: &[Generator],
    params: &ActionParams,
    config: &SearchConfig,
) -> SearchResults {
    let n_edges = complex.n_edges();
    let dim = 2 * n_edges;
    let k = config.n_samples;

    // Orthonormalize known generators for projection.
    let ortho_known = orthonormalize(known_generators);

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Build gradient matrix M: each row is a gradient at a perturbed config.
    let mut m_data = vec![0.0; k * dim];

    for row in 0..k {
        // Perturb lengths multiplicatively, phases additively.
        let eps = config.perturbation_scale;
        let lengths: Vec<f64> = background
            .lengths
            .iter()
            .map(|&l| l * (1.0 + eps * (2.0 * rng.r#gen::<f64>() - 1.0)))
            .collect();
        let phases: Vec<f64> = background
            .phases
            .iter()
            .map(|&p| p + eps * (2.0 * rng.r#gen::<f64>() - 1.0))
            .collect();

        let perturbed = Fields::new(lengths, phases);
        let mut grad = einstein_maxwell_grad(complex, &perturbed, params);

        // Project out known generators.
        project_out_span(&mut grad, &ortho_known);

        // Store in row-major order.
        for (j, &val) in grad.iter().enumerate() {
            m_data[row * dim + j] = val;
        }
    }

    // SVD via tang-la.
    let m = DMat::from_fn(k, dim, |i, j| m_data[i * dim + j]);
    let svd = m.svd(false, true);

    let vt = &svd.vt;
    let singular_values = &svd.s;

    // Sort by increasing singular value.
    let n_sv = singular_values.len();
    let mut sv_indices: Vec<(f64, usize)> = singular_values
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    sv_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Normalize known generators for overlap computation.
    let known_normalized: Vec<(String, Vec<f64>)> = known_generators
        .iter()
        .map(|g| {
            let p = g.normalized().pack();
            (g.name.clone(), p)
        })
        .collect();

    // Build candidate list.
    let n_candidates = n_sv.min(dim);
    let mut candidates = Vec::with_capacity(n_candidates);

    for &(sigma, idx) in sv_indices.iter().take(n_candidates) {
        let gen_vec: Vec<f64> = (0..dim).map(|j| vt[(idx, j)]).collect();

        let overlaps: Vec<(String, f64)> = known_normalized
            .iter()
            .map(|(name, kp)| {
                let dot: f64 = gen_vec.iter().zip(kp.iter()).map(|(a, b)| a * b).sum();
                (name.clone(), dot)
            })
            .collect();

        candidates.push(CandidateSymmetry {
            generator: gen_vec,
            violation: sigma,
            overlaps,
        });
    }

    let known_names = known_generators.iter().map(|g| g.name.clone()).collect();

    SearchResults {
        candidates,
        n_samples: k,
        known_generators: known_names,
    }
}

/// Generic symmetry search taking a gradient closure and perturbation closure.
///
/// This enables searching for symmetries of any action (e.g., Einstein-Yang-Mills
/// with SU(2)) without requiring the `Fields` type.
///
/// - `dim`: total number of DOF in the packed vector
/// - `perturb_and_grad`: given `&mut StdRng`, returns a gradient vector at a
///   randomly perturbed configuration
pub fn search_symmetries_generic(
    dim: usize,
    known_generators: &[Generator],
    config: &SearchConfig,
    perturb_and_grad: impl Fn(&mut StdRng) -> Vec<f64>,
) -> SearchResults {
    let k = config.n_samples;

    let ortho_known = orthonormalize(known_generators);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut m_data = vec![0.0; k * dim];

    for row in 0..k {
        let mut grad = perturb_and_grad(&mut rng);
        project_out_span(&mut grad, &ortho_known);

        for (j, &val) in grad.iter().enumerate() {
            m_data[row * dim + j] = val;
        }
    }

    let m = DMat::from_fn(k, dim, |i, j| m_data[i * dim + j]);
    let svd = m.svd(false, true);

    let vt = &svd.vt;
    let singular_values = &svd.s;

    let n_sv = singular_values.len();
    let mut sv_indices: Vec<(f64, usize)> = singular_values
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    sv_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let known_normalized: Vec<(String, Vec<f64>)> = known_generators
        .iter()
        .map(|g| {
            let p = g.normalized().pack();
            (g.name.clone(), p)
        })
        .collect();

    let n_candidates = n_sv.min(dim);
    let mut candidates = Vec::with_capacity(n_candidates);

    for &(sigma, idx) in sv_indices.iter().take(n_candidates) {
        let gen_vec: Vec<f64> = (0..dim).map(|j| vt[(idx, j)]).collect();

        let overlaps: Vec<(String, f64)> = known_normalized
            .iter()
            .map(|(name, kp)| {
                let dot: f64 = gen_vec.iter().zip(kp.iter()).map(|(a, b)| a * b).sum();
                (name.clone(), dot)
            })
            .collect();

        candidates.push(CandidateSymmetry {
            generator: gen_vec,
            violation: sigma,
            overlaps,
        });
    }

    let known_names = known_generators.iter().map(|g| g.name.clone()).collect();

    SearchResults {
        candidates,
        n_samples: k,
        known_generators: known_names,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;
    use crate::symmetry::all_gauge_generators;

    #[test]
    fn test_rediscover_gauge_symmetry() {
        // Verify that gauge generators lie in the null space of the gradient
        // matrix built by the search. For each gauge generator G, every
        // sampled gradient g_k should satisfy g_k · G = 0 (exact gauge
        // invariance). We test this directly without requiring n_samples > dim.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let dim = 2 * complex.n_edges();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.2)
            .collect();

        let background = Fields::new(lengths, phases);
        let params = ActionParams::default();

        // Sample gradients at perturbed configs (same as search internals).
        let eps = 1e-3;
        let k = 50;
        let mut sample_rng = StdRng::seed_from_u64(99);
        let mut gradients = Vec::with_capacity(k);

        for _ in 0..k {
            let perturbed_lengths: Vec<f64> = background
                .lengths
                .iter()
                .map(|&l| l * (1.0 + eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0)))
                .collect();
            let perturbed_phases: Vec<f64> = background
                .phases
                .iter()
                .map(|&p| p + eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0))
                .collect();

            let perturbed = Fields::new(perturbed_lengths, perturbed_phases);
            gradients.push(einstein_maxwell_grad(&complex, &perturbed, &params));
        }

        // Every gauge generator should be orthogonal to every gradient.
        let gauge_gens = all_gauge_generators(&complex);
        for gauge_g in &gauge_gens {
            let gp = gauge_g.pack();
            for (row, grad) in gradients.iter().enumerate() {
                let dot: f64 = grad.iter().zip(gp.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-10,
                    "gradient[{row}] · {} = {dot} (expected 0)",
                    gauge_g.name
                );
            }
        }

        // Also verify search_symmetries runs and produces results.
        let config = SearchConfig {
            n_samples: k,
            perturbation_scale: eps,
            seed: 99,
        };
        let results = search_symmetries(&complex, &background, &[], &params, &config);
        assert_eq!(results.candidates.len(), dim.min(k));
    }

    #[test]
    fn test_gauge_projected_out() {
        // When gauge generators are provided as known, candidates should
        // have near-zero overlap with them.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(77);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.2)
            .collect();

        let background = Fields::new(lengths, phases);
        let params = ActionParams::default();
        let gauge_gens = all_gauge_generators(&complex);

        let config = SearchConfig {
            n_samples: 100,
            perturbation_scale: 1e-3,
            seed: 55,
        };

        let results = search_symmetries(&complex, &background, &gauge_gens, &params, &config);

        // Candidates should NOT have high gauge overlap (gauge was projected out).
        let gauge_normalized: Vec<Vec<f64>> =
            gauge_gens.iter().map(|g| g.normalized().pack()).collect();

        for candidate in results.candidates.iter().take(20) {
            let gauge_overlap_sq: f64 = gauge_normalized
                .iter()
                .map(|gp| {
                    let dot: f64 = candidate
                        .generator
                        .iter()
                        .zip(gp.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    dot * dot
                })
                .sum();
            assert!(
                gauge_overlap_sq < 0.5,
                "candidate has high gauge overlap after projection: {gauge_overlap_sq:.4}"
            );
        }
    }

    #[test]
    fn test_deterministic_search() {
        // Same seed should produce identical results.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let phases = vec![0.1; complex.n_edges()];
        let background = Fields::new(lengths, phases);
        let params = ActionParams::default();
        let config = SearchConfig {
            n_samples: 50,
            perturbation_scale: 1e-3,
            seed: 1234,
        };

        let r1 = search_symmetries(&complex, &background, &[], &params, &config);
        let r2 = search_symmetries(&complex, &background, &[], &params, &config);

        assert_eq!(r1.candidates.len(), r2.candidates.len());
        for (c1, c2) in r1.candidates.iter().zip(r2.candidates.iter()) {
            assert!(
                (c1.violation - c2.violation).abs() < 1e-15,
                "non-deterministic: {} vs {}",
                c1.violation,
                c2.violation
            );
            for (v1, v2) in c1.generator.iter().zip(c2.generator.iter()) {
                assert!(
                    (v1 - v2).abs() < 1e-15,
                    "non-deterministic generator components"
                );
            }
        }
    }

    #[test]
    fn test_su2_gauge_null_space() {
        // SU(2) gauge generators should lie in the null space of the
        // Einstein-Yang-Mills gradient matrix. For SU(2), gauge generators
        // are field-dependent (involve the adjoint representation).
        use crate::su2::Su2;
        use crate::yang_mills::{einstein_yang_mills_grad, su2_gauge_generator};

        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let n_edges = complex.n_edges();

        let mut rng = StdRng::seed_from_u64(42);
        let elements: Vec<Su2> = (0..n_edges)
            .map(|_| {
                Su2::exp(&[
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                ])
            })
            .collect();

        let alpha = 1.0;
        let eps = 1e-3;
        let k = 30;
        let mut sample_rng = StdRng::seed_from_u64(99);

        for _ in 0..k {
            let perturbed_lengths: Vec<f64> = lengths
                .iter()
                .map(|&l| l * (1.0 + eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0)))
                .collect();
            let perturbed_elements: Vec<Su2> = elements
                .iter()
                .map(|u| {
                    let delta = [
                        eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0),
                        eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0),
                        eps * (2.0 * sample_rng.r#gen::<f64>() - 1.0),
                    ];
                    Su2::exp(&delta).mul(u)
                })
                .collect();

            let grad =
                einstein_yang_mills_grad(&complex, &perturbed_lengths, &perturbed_elements, alpha);

            // Compute field-dependent gauge generators at this config.
            for v in 0..complex.n_vertices {
                for dir in 0..3 {
                    let gauge_dof = su2_gauge_generator(&complex, &perturbed_elements, v, dir);
                    // Full generator: zero for lengths, gauge_dof for field.
                    let dot: f64 = grad[n_edges..]
                        .iter()
                        .zip(gauge_dof.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    assert!(
                        dot.abs() < 1e-8,
                        "gradient · su2_gauge_v{v}_d{dir} = {dot} (expected 0)"
                    );
                }
            }
        }
    }

    #[test]
    fn test_search_report() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let background = Fields::new(lengths, phases);
        let params = ActionParams::default();
        let config = SearchConfig {
            n_samples: 30,
            perturbation_scale: 1e-3,
            seed: 42,
        };

        let results = search_symmetries(&complex, &background, &[], &params, &config);
        let report = results.report();
        assert!(report.contains("Symmetry search:"));
        assert!(report.contains("30 samples"));
    }
}
