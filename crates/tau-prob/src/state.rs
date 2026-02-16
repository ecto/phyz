//! Probabilistic state with uncertain positions, velocities, and parameters.

use crate::Distribution;
use rand_distr::{Distribution as RandDist, Normal};
use tau_math::DVec;
use tau_model::Model;

/// Probabilistic state with distribution over q, v, and physical parameters.
#[derive(Debug, Clone)]
pub struct ProbabilisticState {
    /// Distribution over generalized positions.
    pub q: Distribution<DVec>,
    /// Distribution over generalized velocities.
    pub v: Distribution<DVec>,
    /// Distribution over uncertain physical parameters.
    pub parameters: Distribution<Vec<f64>>,
    /// Simulation time (shared across ensemble).
    pub time: f64,
}

impl ProbabilisticState {
    /// Create a probabilistic state with uniform samples around a central state.
    pub fn uniform_samples(model: &Model, nsamples: usize) -> Self {
        let q_samples = vec![DVec::zeros(model.nq); nsamples];
        let v_samples = vec![DVec::zeros(model.nv); nsamples];
        let param_samples = vec![vec![]; nsamples];

        Self {
            q: Distribution::uniform(q_samples),
            v: Distribution::uniform(v_samples),
            parameters: Distribution::uniform(param_samples),
            time: 0.0,
        }
    }

    /// Create samples by perturbing a central state.
    pub fn perturbed_samples(
        model: &Model,
        q0: &DVec,
        v0: &DVec,
        nsamples: usize,
        q_std: f64,
        v_std: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut q_samples = Vec::with_capacity(nsamples);
        let mut v_samples = Vec::with_capacity(nsamples);

        for _ in 0..nsamples {
            let mut q = q0.clone();
            let mut v = v0.clone();

            for i in 0..model.nq {
                let noise = Normal::new(0.0, q_std).unwrap().sample(&mut rng);
                q[i] += noise;
            }

            for i in 0..model.nv {
                let noise = Normal::new(0.0, v_std).unwrap().sample(&mut rng);
                v[i] += noise;
            }

            q_samples.push(q);
            v_samples.push(v);
        }

        let param_samples = vec![vec![]; nsamples];

        Self {
            q: Distribution::uniform(q_samples),
            v: Distribution::uniform(v_samples),
            parameters: Distribution::uniform(param_samples),
            time: 0.0,
        }
    }

    /// Sample parameters for each particle using a custom sampling function.
    ///
    /// Example:
    /// ```ignore
    /// state.sample_parameters(|rng| {
    ///     vec![
    ///         Normal::new(1.0, 0.1).unwrap().sample(rng),
    ///         Uniform::new(0.0, 0.5).sample(rng),
    ///     ]
    /// });
    /// ```
    pub fn sample_parameters<F>(&mut self, mut sampler: F)
    where
        F: FnMut(&mut rand::rngs::ThreadRng) -> Vec<f64>,
    {
        let mut rng = rand::thread_rng();
        let nsamples = self.q.samples.len();
        let mut param_samples = Vec::with_capacity(nsamples);

        for _ in 0..nsamples {
            param_samples.push(sampler(&mut rng));
        }

        self.parameters = Distribution::uniform(param_samples);
    }

    /// Compute mean state.
    pub fn mean(&self) -> (DVec, DVec) {
        let q_mean = self.q.weighted_mean();
        let v_mean = self.v.weighted_mean();
        (q_mean, v_mean)
    }

    /// Compute mean and standard deviation for positions and velocities.
    pub fn mean_and_std(&self) -> ((DVec, DVec), (DVec, DVec)) {
        let (q_mean, q_std) = self.q.mean_and_std();
        let (v_mean, v_std) = self.v.mean_and_std();
        ((q_mean, v_mean), (q_std, v_std))
    }

    /// Resample particles if effective sample size drops below threshold.
    pub fn resample(&mut self, threshold_ratio: f64) {
        self.q.resample_if_needed(threshold_ratio);
        self.v.resample_if_needed(threshold_ratio);
        self.parameters.resample_if_needed(threshold_ratio);
    }

    /// Number of samples in ensemble.
    pub fn nsamples(&self) -> usize {
        self.q.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_model::ModelBuilder;

    fn make_pendulum() -> Model {
        use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};

        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::new(0.0, -0.5, 0.0),
                    Mat3::from_diagonal(&Vec3::new(1.0 / 12.0, 0.0, 1.0 / 12.0)),
                ),
            )
            .build()
    }

    #[test]
    fn test_uniform_samples() {
        let model = make_pendulum();
        let state = ProbabilisticState::uniform_samples(&model, 100);

        assert_eq!(state.nsamples(), 100);
        assert_eq!(state.q.samples.len(), 100);
        assert_eq!(state.v.samples.len(), 100);
    }

    #[test]
    fn test_perturbed_samples() {
        let model = make_pendulum();
        let q0 = DVec::from_vec(vec![0.1]);
        let v0 = DVec::from_vec(vec![0.0]);

        let state = ProbabilisticState::perturbed_samples(&model, &q0, &v0, 100, 0.01, 0.01);

        assert_eq!(state.nsamples(), 100);

        let (q_mean, _) = state.mean();
        // Mean should be close to q0
        assert!((q_mean[0] - 0.1).abs() < 0.02);
    }

    #[test]
    fn test_sample_parameters() {
        use rand_distr::Uniform;
        let model = make_pendulum();
        let mut state = ProbabilisticState::uniform_samples(&model, 50);

        state.sample_parameters(|rng| {
            vec![
                Normal::new(1.0, 0.1).unwrap().sample(rng),
                Uniform::new(0.0, 1.0).sample(rng),
            ]
        });

        assert_eq!(state.parameters.samples.len(), 50);
        assert_eq!(state.parameters.samples[0].len(), 2);
    }
}
