//! Stein Variational Gradient Descent (SVGD) for parameter inference.
//!
//! SVGD optimizes particle locations to match a target posterior distribution
//! while maintaining diversity through a repulsive kernel term.

use crate::Distribution;

/// Compute RBF kernel: k(x, y) = exp(-||x - y||² / (2h²))
fn rbf_kernel(x: &[f64], y: &[f64], bandwidth: f64) -> f64 {
    let mut sq_dist = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        sq_dist += diff * diff;
    }
    (-sq_dist / (2.0 * bandwidth * bandwidth)).exp()
}

/// Compute gradient of RBF kernel w.r.t. first argument: ∇_x k(x, y)
fn rbf_kernel_grad(x: &[f64], y: &[f64], bandwidth: f64) -> Vec<f64> {
    let k = rbf_kernel(x, y, bandwidth);
    let h2 = bandwidth * bandwidth;
    x.iter()
        .zip(y)
        .map(|(xi, yi)| -(xi - yi) * k / h2)
        .collect()
}

/// Median heuristic for bandwidth selection.
///
/// Sets bandwidth to median pairwise distance between particles.
fn median_bandwidth(particles: &[Vec<f64>]) -> f64 {
    let n = particles.len();
    if n <= 1 {
        return 1.0;
    }

    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq_dist = 0.0;
            for (pi, pj) in particles[i].iter().zip(&particles[j]) {
                let diff = pi - pj;
                sq_dist += diff * diff;
            }
            distances.push(sq_dist.sqrt());
        }
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances[distances.len() / 2]
}

/// Perform one SVGD update step.
///
/// Updates particle positions using:
/// x_i^{t+1} = x_i^t + ε * φ(x_i)
///
/// where φ(x_i) = (1/N) Σ_j [k(x_i, x_j) ∇_x log p(x_j) + ∇_x k(x_i, x_j)]
///
/// # Arguments
/// * `particles` - Current particle positions
/// * `log_prob_grads` - Gradient of log posterior at each particle: ∇_x log p(x_i)
/// * `step_size` - Learning rate ε
/// * `bandwidth` - Kernel bandwidth (use None for median heuristic)
pub fn svgd_step(
    particles: &mut Distribution<Vec<f64>>,
    log_prob_grads: &[Vec<f64>],
    step_size: f64,
    bandwidth: Option<f64>,
) {
    let n = particles.samples.len();
    if n == 0 {
        return;
    }

    let d = particles.samples[0].len();
    assert_eq!(log_prob_grads.len(), n);

    // Compute bandwidth
    let h = bandwidth.unwrap_or_else(|| median_bandwidth(&particles.samples));

    // Compute SVGD directions
    let mut directions = vec![vec![0.0; d]; n];

    for (i, direction) in directions.iter_mut().enumerate().take(n) {
        for (j, log_prob_grad) in log_prob_grads.iter().enumerate().take(n) {
            let k_ij = rbf_kernel(&particles.samples[i], &particles.samples[j], h);
            let grad_k = rbf_kernel_grad(&particles.samples[i], &particles.samples[j], h);

            // φ(x_i) += k(x_i, x_j) * ∇ log p(x_j) + ∇_x k(x_i, x_j)
            for (dim, dir_val) in direction.iter_mut().enumerate().take(d) {
                *dir_val += k_ij * log_prob_grad[dim] + grad_k[dim];
            }
        }

        // Average over particles
        for dir_val in direction.iter_mut().take(d) {
            *dir_val /= n as f64;
        }
    }

    // Update particles
    for (particle, direction) in particles.samples.iter_mut().zip(&directions) {
        for (p_val, &d_val) in particle.iter_mut().zip(direction) {
            *p_val += step_size * d_val;
        }
    }
}

/// Simplified SVGD for parameter inference given observations.
///
/// Assumes Gaussian likelihood: log p(θ | data) ∝ -||simulate(θ) - data||²/(2σ²)
///
/// # Arguments
/// * `particles` - Parameter particles to optimize
/// * `simulator` - Function that simulates forward model: θ → predicted observations
/// * `observations` - Observed data
/// * `noise_std` - Observation noise standard deviation
/// * `step_size` - SVGD learning rate
/// * `prior_grad` - Optional gradient of log prior: ∇ log p(θ)
pub fn svgd_inference_step<F, P>(
    particles: &mut Distribution<Vec<f64>>,
    simulator: F,
    observations: &[f64],
    noise_std: f64,
    step_size: f64,
    prior_grad: Option<P>,
) where
    F: Fn(&[f64]) -> Vec<f64>,
    P: Fn(&[f64]) -> Vec<f64>,
{
    let n = particles.samples.len();
    let mut log_prob_grads = Vec::with_capacity(n);

    let sigma2 = noise_std * noise_std;

    for particle in &particles.samples {
        // Simulate forward model
        let pred = simulator(particle);

        // Compute gradient of log likelihood via finite differences
        let eps = 1e-6;
        let mut grad = vec![0.0; particle.len()];

        for dim in 0..particle.len() {
            let mut particle_plus = particle.clone();
            particle_plus[dim] += eps;
            let pred_plus = simulator(&particle_plus);

            // ∇ log p(y | θ) ≈ (∇ predictions)^T * (observations - predictions) / σ²
            let mut grad_dim = 0.0;
            for j in 0..pred.len() {
                let residual = observations[j] - pred[j];
                let d_pred = (pred_plus[j] - pred[j]) / eps;
                grad_dim += d_pred * residual / sigma2;
            }
            grad[dim] = grad_dim;
        }

        // Add prior gradient if provided
        if let Some(ref prior_fn) = prior_grad {
            let prior_g = prior_fn(particle);
            for dim in 0..grad.len() {
                grad[dim] += prior_g[dim];
            }
        }

        log_prob_grads.push(grad);
    }

    svgd_step(particles, &log_prob_grads, step_size, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel() {
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];
        let k = rbf_kernel(&x, &y, 1.0);

        // k(x, y) = exp(-1/2) ≈ 0.606
        assert!((k - (-0.5_f64).exp()).abs() < 1e-6);

        // k(x, x) = 1
        let k_self = rbf_kernel(&x, &x, 1.0);
        assert!((k_self - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_grad() {
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 0.0];
        let grad = rbf_kernel_grad(&x, &y, 1.0);

        // Should be negative (repulsive) in x direction
        assert!(grad[0] < 0.0);
        assert!(grad[1].abs() < 1e-10);
    }

    #[test]
    fn test_median_bandwidth() {
        let particles = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let h = median_bandwidth(&particles);
        // Distances: 1, 1, sqrt(2), 1, sqrt(2), 1 → median = 1
        assert!((h - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_svgd_step_basic() {
        // Simple test: particles should move toward target
        let mut particles =
            Distribution::uniform(vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]]);

        // Gradient pointing toward zero
        let log_prob_grads = vec![vec![2.0], vec![1.0], vec![-1.0], vec![-2.0]];

        svgd_step(&mut particles, &log_prob_grads, 0.1, Some(1.0));

        // Particles should move toward zero on average
        let mean_after: f64 = particles.samples.iter().map(|p| p[0]).sum::<f64>() / 4.0;

        // Mean should still be near 0 but particles should have moved
        assert!(mean_after.abs() < 1.0);
    }

    #[test]
    fn test_svgd_inference_simple() {
        // Test that SVGD inference runs without panicking
        let mut particles = Distribution::uniform(vec![vec![1.0], vec![2.0], vec![3.0]]);

        // True model: a = 2.0, observations at x = 1.0, 2.0
        let observations = vec![2.0, 4.0];

        let simulator = |params: &[f64]| -> Vec<f64> {
            let a = params[0];
            vec![a * 1.0, a * 2.0]
        };

        // Run a few SVGD steps - just verify it doesn't panic
        for _ in 0..5 {
            svgd_inference_step(
                &mut particles,
                &simulator,
                &observations,
                0.1,
                0.01,
                None::<fn(&[f64]) -> Vec<f64>>,
            );
        }

        // Verify particles are still finite
        for particle in &particles.samples {
            assert!(particle[0].is_finite());
        }
    }
}
