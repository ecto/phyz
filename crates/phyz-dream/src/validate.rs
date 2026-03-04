use crate::data::SampleConfig;
use crate::model::DreamModel;
use phyz_model::Model;
use tang_train::Rng;

/// Validation results comparing surrogate to exact ABA.
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub max_error: f64,
    pub mean_error: f64,
    pub rmse: f64,
    pub max_relative_error: f64,
    pub n_samples: usize,
}

/// Validate a trained surrogate against exact ABA on fresh random samples.
pub fn validate(
    dream: &mut DreamModel,
    model: &Model,
    n_samples: usize,
    seed: u64,
) -> ValidationResult {
    let config = SampleConfig::default();
    let mut rng = Rng::new(seed);

    let mut max_error = 0.0f64;
    let mut sum_error = 0.0f64;
    let mut sum_sq_error = 0.0f64;
    let mut max_relative_error = 0.0f64;

    for _ in 0..n_samples {
        let state = crate::data::sample_state(model, &mut rng, &config);
        let exact = phyz_rigid::aba(model, &state);
        let approx = dream.predict(model, &state);

        let mut sample_error = 0.0f64;
        for i in 0..model.nv {
            let err = (exact[i] - approx[i]).abs();
            sample_error = sample_error.max(err);

            let rel = if exact[i].abs() > 1e-8 {
                err / exact[i].abs()
            } else {
                0.0
            };
            max_relative_error = max_relative_error.max(rel);
        }

        max_error = max_error.max(sample_error);
        sum_error += sample_error;
        sum_sq_error += sample_error * sample_error;
    }

    ValidationResult {
        max_error,
        mean_error: sum_error / n_samples as f64,
        rmse: (sum_sq_error / n_samples as f64).sqrt(),
        max_relative_error,
        n_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SampleConfig;
    use crate::train::{TrainConfig, train};
    use crate::tests::make_double_pendulum;

    #[test]
    fn validate_trained_model() {
        let model = make_double_pendulum();
        let config = TrainConfig {
            n_samples: 10_000,
            hidden_dim: 128,
            n_hidden: 2,
            batch_size: 256,
            epochs: 100,
            lr: 1e-3,
            weight_decay: 1e-4,
            seed: 99,
            sample_config: SampleConfig::default(),
        };

        let (mut dream, _losses) = train(&model, &config);
        let result = validate(&mut dream, &model, 200, 777);

        assert!(
            result.mean_error < 5.0,
            "mean_error too high: {}",
            result.mean_error
        );
    }
}
