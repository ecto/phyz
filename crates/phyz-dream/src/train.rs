use crate::data::{SampleConfig, generate_dataset};
use crate::model::{DreamModel, NormStats};
use phyz_model::Model;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};
use tang_train::{DataLoader, Dataset, ModuleAdam, Trainer, TensorDataset, mse_loss, mse_loss_grad};

/// Training configuration.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub n_samples: usize,
    pub hidden_dim: usize,
    pub n_hidden: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
    pub sample_config: SampleConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_samples: 100_000,
            hidden_dim: 128,
            n_hidden: 2,
            batch_size: 256,
            epochs: 100,
            lr: 1e-3,
            weight_decay: 1e-4,
            seed: 42,
            sample_config: SampleConfig::default(),
        }
    }
}

/// Build a normalized dataset from raw data, returning (normalized_dataset, input_norm, output_norm).
fn normalize_dataset(
    raw: &TensorDataset<f32>,
) -> (TensorDataset<f32>, NormStats, NormStats) {
    let input_norm = NormStats::from_dataset(raw, true);
    let output_norm = NormStats::from_dataset(raw, false);

    let n = raw.len();
    let (sample_in, sample_out) = raw.get(0);
    let in_dim = sample_in.shape().dims()[0];
    let out_dim = sample_out.shape().dims()[0];

    let mut norm_inputs = Vec::with_capacity(n * in_dim);
    let mut norm_outputs = Vec::with_capacity(n * out_dim);

    for i in 0..n {
        let (inp, tgt) = raw.get(i);
        norm_inputs.extend(input_norm.normalize(inp.data()));
        norm_outputs.extend(output_norm.normalize(tgt.data()));
    }

    let inputs = Tensor::new(norm_inputs, Shape::new(vec![n, in_dim]));
    let targets = Tensor::new(norm_outputs, Shape::new(vec![n, out_dim]));

    (TensorDataset::new(inputs, targets), input_norm, output_norm)
}

/// Train a surrogate dynamics model on ABA ground truth.
///
/// Returns the trained model and per-epoch loss history.
pub fn train(model: &Model, config: &TrainConfig) -> (DreamModel, Vec<f64>) {
    // Generate raw dataset
    let raw_ds = generate_dataset(model, config.n_samples, &config.sample_config, config.seed);

    // Compute normalization and build normalized dataset
    let (norm_ds, input_norm, output_norm) = normalize_dataset(&raw_ds);

    // Build model
    let mut dream = DreamModel::new(
        model.nq,
        model.nv,
        config.hidden_dim,
        config.n_hidden,
        config.seed + 1000,
    );
    dream.set_norm_stats(input_norm, output_norm);

    // Train
    let mut loader = DataLoader::new(&norm_ds, config.batch_size);
    let optimizer = ModuleAdam::with_weight_decay(config.lr, config.weight_decay);

    let losses = Trainer::new(
        &mut dream,
        optimizer,
        |pred: &Tensor<f32>, target: &Tensor<f32>| {
            (mse_loss(pred, target).to_f64(), mse_loss_grad(pred, target))
        },
    )
    .epochs(config.epochs)
    .fit(&mut loader);

    (dream, losses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_double_pendulum;

    #[test]
    fn train_loss_decreases() {
        let model = make_double_pendulum();
        let config = TrainConfig {
            n_samples: 2000,
            hidden_dim: 64,
            n_hidden: 2,
            batch_size: 128,
            epochs: 50,
            lr: 1e-3,
            weight_decay: 0.0,
            seed: 42,
            sample_config: SampleConfig::default(),
        };

        let (_dream, losses) = train(&model, &config);

        assert!(!losses.is_empty());
        // First loss should be larger than last
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }
}
