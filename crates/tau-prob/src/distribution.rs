//! Distribution representation for uncertain variables.

use rand::Rng;
use tau_math::{DMat, DVec};

/// Distribution represented as a weighted particle ensemble.
#[derive(Debug, Clone)]
pub struct Distribution<T> {
    /// Particle samples.
    pub samples: Vec<T>,
    /// Importance weights (should sum to 1.0).
    pub weights: Vec<f64>,
}

impl<T: Clone> Distribution<T> {
    /// Create a uniform distribution from samples.
    pub fn uniform(samples: Vec<T>) -> Self {
        let n = samples.len();
        let w = 1.0 / n as f64;
        Self {
            samples,
            weights: vec![w; n],
        }
    }

    /// Create a distribution with custom weights.
    pub fn weighted(samples: Vec<T>, weights: Vec<f64>) -> Self {
        assert_eq!(samples.len(), weights.len());
        Self { samples, weights }
    }

    /// Normalize weights to sum to 1.
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Effective sample size: 1 / Σ w_i²
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    /// Resample particles if effective sample size drops below threshold.
    ///
    /// Uses systematic resampling for low variance.
    pub fn resample_if_needed(&mut self, threshold_ratio: f64)
    where
        T: Clone,
    {
        let n = self.samples.len() as f64;
        let eff = self.effective_sample_size();

        if eff < threshold_ratio * n {
            self.systematic_resample();
        }
    }

    /// Systematic resampling (low variance resampling).
    fn systematic_resample(&mut self)
    where
        T: Clone,
    {
        let n = self.samples.len();
        let mut new_samples = Vec::with_capacity(n);

        // Cumulative weight distribution
        let mut cumsum = vec![0.0; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + self.weights[i];
        }

        let mut rng = rand::thread_rng();
        let u0: f64 = rng.r#gen();
        let step = 1.0 / n as f64;

        for i in 0..n {
            let u = (u0 + i as f64 * step) % 1.0;
            // Binary search for the sample index
            let idx = cumsum
                .iter()
                .position(|&c| c >= u)
                .unwrap_or(n)
                .saturating_sub(1);
            new_samples.push(self.samples[idx].clone());
        }

        self.samples = new_samples;
        self.weights = vec![1.0 / n as f64; n];
    }
}

impl Distribution<DVec> {
    /// Compute weighted mean of vector samples.
    pub fn weighted_mean(&self) -> DVec {
        let n = self.samples[0].len();
        let mut mean = DVec::zeros(n);
        for (sample, weight) in self.samples.iter().zip(&self.weights) {
            mean += sample * *weight;
        }
        mean
    }

    /// Compute weighted covariance matrix.
    pub fn weighted_covariance(&self) -> DMat {
        let mean = self.weighted_mean();
        let n = mean.len();
        let mut cov = DMat::zeros(n, n);

        for (sample, weight) in self.samples.iter().zip(&self.weights) {
            let diff = sample - &mean;
            cov += diff.clone() * diff.transpose() * *weight;
        }
        cov
    }

    /// Compute mean and standard deviation.
    pub fn mean_and_std(&self) -> (DVec, DVec) {
        let mean = self.weighted_mean();
        let cov = self.weighted_covariance();
        let std = DVec::from_iterator(
            cov.nrows(),
            cov.diagonal().iter().map(|&v| v.max(0.0).sqrt()),
        );
        (mean, std)
    }
}

impl Distribution<Vec<f64>> {
    /// Compute weighted mean of parameter vectors.
    pub fn weighted_mean(&self) -> Vec<f64> {
        let n = self.samples[0].len();
        let mut mean = vec![0.0; n];
        for (sample, weight) in self.samples.iter().zip(&self.weights) {
            for i in 0..n {
                mean[i] += sample[i] * weight;
            }
        }
        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let samples = vec![
            DVec::from_vec(vec![1.0, 2.0]),
            DVec::from_vec(vec![3.0, 4.0]),
            DVec::from_vec(vec![5.0, 6.0]),
        ];
        let dist = Distribution::uniform(samples);

        assert_eq!(dist.weights.len(), 3);
        assert!((dist.weights[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean() {
        let samples = vec![
            DVec::from_vec(vec![1.0, 0.0]),
            DVec::from_vec(vec![0.0, 1.0]),
        ];
        let dist = Distribution::weighted(samples, vec![0.75, 0.25]);

        let mean = dist.weighted_mean();
        assert!((mean[0] - 0.75).abs() < 1e-10);
        assert!((mean[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sample_size() {
        let samples = vec![DVec::zeros(2); 4];
        let mut dist = Distribution::uniform(samples);

        let ess = dist.effective_sample_size();
        assert!((ess - 4.0).abs() < 1e-10);

        // Skewed weights
        dist.weights = vec![0.97, 0.01, 0.01, 0.01];
        let ess2 = dist.effective_sample_size();
        assert!(ess2 < 2.0); // Much lower effective sample size
    }

    #[test]
    fn test_normalize_weights() {
        let samples = vec![DVec::zeros(2); 3];
        let mut dist = Distribution::weighted(samples, vec![2.0, 3.0, 5.0]);

        dist.normalize_weights();
        let sum: f64 = dist.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((dist.weights[0] - 0.2).abs() < 1e-10);
    }
}
