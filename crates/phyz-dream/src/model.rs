use phyz_math::DVec;
use phyz_model::{Model, State};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tang_tensor::{Shape, Tensor};
use tang_train::{Dataset, Linear, Module, SiLU, Sequential, TensorDataset};

/// Per-feature normalization statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl NormStats {
    /// Compute normalization stats from a dataset. Std floored at 1e-6.
    pub fn from_dataset(ds: &TensorDataset<f32>, is_input: bool) -> Self {
        let n = ds.len();
        let (sample, _tgt) = ds.get(0);
        let (dim, get_vec): (usize, Box<dyn Fn(usize) -> Tensor<f32>>) = if is_input {
            let d = sample.shape().dims()[0];
            (d, Box::new(|i| ds.get(i).0))
        } else {
            let (_, t) = ds.get(0);
            let d = t.shape().dims()[0];
            (d, Box::new(|i| ds.get(i).1))
        };

        let mut mean = vec![0.0f64; dim];
        for i in 0..n {
            let t = get_vec(i);
            for j in 0..dim {
                mean[j] += t.data()[j] as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        let mut var = vec![0.0f64; dim];
        for i in 0..n {
            let t = get_vec(i);
            for j in 0..dim {
                let d = t.data()[j] as f64 - mean[j];
                var[j] += d * d;
            }
        }

        let std: Vec<f32> = var
            .iter()
            .zip(&mean)
            .map(|(v, _)| ((*v / n as f64).sqrt() as f32).max(1e-6))
            .collect();
        let mean: Vec<f32> = mean.iter().map(|&m| m as f32).collect();

        Self { mean, std }
    }

    pub fn normalize(&self, data: &[f32]) -> Vec<f32> {
        data.iter()
            .zip(&self.mean)
            .zip(&self.std)
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect()
    }

    pub fn denormalize(&self, data: &[f32]) -> Vec<f32> {
        data.iter()
            .zip(&self.mean)
            .zip(&self.std)
            .map(|((&x, &m), &s)| x * s + m)
            .collect()
    }
}

/// Metadata for a trained dream model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DreamMeta {
    pub nq: usize,
    pub nv: usize,
    pub hidden_dim: usize,
    pub n_hidden: usize,
    pub input_norm: NormStats,
    pub output_norm: NormStats,
}

/// Learned surrogate dynamics model.
pub struct DreamModel {
    pub(crate) net: Sequential<f32>,
    pub(crate) meta: DreamMeta,
}

impl DreamModel {
    /// Create a new untrained dream model.
    pub fn new(nq: usize, nv: usize, hidden_dim: usize, n_hidden: usize, seed: u64) -> Self {
        let in_dim = nq + nv + nv; // q + v + ctrl
        let mut layers: Vec<Box<dyn Module<f32>>> = Vec::new();
        let mut rng_seed = seed;

        // Input layer
        layers.push(Box::new(Linear::new(in_dim, hidden_dim, rng_seed)));
        layers.push(Box::new(SiLU::new()));
        rng_seed += 1;

        // Hidden layers
        for _ in 0..n_hidden.saturating_sub(1) {
            layers.push(Box::new(Linear::new(hidden_dim, hidden_dim, rng_seed)));
            layers.push(Box::new(SiLU::new()));
            rng_seed += 1;
        }

        // Output layer
        layers.push(Box::new(Linear::new(hidden_dim, nv, rng_seed)));

        let net = Sequential::new(layers);
        let meta = DreamMeta {
            nq,
            nv,
            hidden_dim,
            n_hidden,
            input_norm: NormStats {
                mean: vec![0.0; in_dim],
                std: vec![1.0; in_dim],
            },
            output_norm: NormStats {
                mean: vec![0.0; nv],
                std: vec![1.0; nv],
            },
        };

        Self { net, meta }
    }

    /// Attach normalization stats (typically computed from training data).
    pub fn set_norm_stats(&mut self, input_norm: NormStats, output_norm: NormStats) {
        self.meta.input_norm = input_norm;
        self.meta.output_norm = output_norm;
    }

    /// Predict joint accelerations from a state. Drop-in replacement for `aba(model, state)`.
    pub fn predict(&mut self, model: &Model, state: &State) -> DVec {
        // Flatten state to f32 input
        let mut raw = Vec::with_capacity(model.nq + model.nv + model.nv);
        for i in 0..model.nq {
            raw.push(state.q[i] as f32);
        }
        for i in 0..model.nv {
            raw.push(state.v[i] as f32);
        }
        for i in 0..model.nv {
            raw.push(state.ctrl[i] as f32);
        }

        let normalized = self.meta.input_norm.normalize(&raw);
        let input = Tensor::new(normalized, Shape::new(vec![1, raw.len()]));
        let output = self.net.forward(&input);
        let denorm = self.meta.output_norm.denormalize(output.data());

        let mut qdd = DVec::zeros(model.nv);
        for i in 0..model.nv {
            qdd[i] = denorm[i] as f64;
        }
        qdd
    }

    /// Raw batch inference (caller handles normalization).
    pub fn predict_batch(&mut self, inputs: &Tensor<f32>) -> Tensor<f32> {
        self.net.forward(inputs)
    }

    /// Save model weights and metadata to a directory.
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        // Save weights
        let state = self.net.state_dict();
        let tensors: HashMap<String, Tensor<f32>> = state.into_iter().collect();
        tang_safetensors::save_f32(&tensors, &dir.join("weights.safetensors"))?;

        // Save metadata
        let meta_json = serde_json::to_string_pretty(&self.meta)?;
        std::fs::write(dir.join("meta.json"), meta_json)?;

        Ok(())
    }

    /// Load a trained model from a directory.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let meta_str = std::fs::read_to_string(dir.join("meta.json"))?;
        let meta: DreamMeta = serde_json::from_str(&meta_str)?;

        let mut model = Self::new(meta.nq, meta.nv, meta.hidden_dim, meta.n_hidden, 0);
        model.meta = meta;

        let tensors = tang_safetensors::load_f32(&dir.join("weights.safetensors"))?;
        let state: Vec<(String, Tensor<f32>)> = tensors.into_iter().collect();
        model.net.load_state_dict(&state);

        Ok(model)
    }

    pub fn meta(&self) -> &DreamMeta {
        &self.meta
    }
}

impl Module<f32> for DreamModel {
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.net.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        self.net.backward(grad_output)
    }

    fn parameters(&self) -> Vec<&tang_train::Parameter<f32>> {
        self.net.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut tang_train::Parameter<f32>> {
        self.net.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &tang_train::Parameter<f32>)> {
        self.net.named_parameters()
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut tang_train::Parameter<f32>)> {
        self.net.named_parameters_mut()
    }

    fn state_dict(&self) -> Vec<(String, Tensor<f32>)> {
        self.net.state_dict()
    }

    fn load_state_dict(&mut self, state: &[(String, Tensor<f32>)]) {
        self.net.load_state_dict(state);
    }

    fn zero_grad(&mut self) {
        self.net.zero_grad();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape() {
        let mut model = DreamModel::new(2, 2, 64, 2, 42);
        let input = Tensor::new(vec![0.0f32; 6], Shape::new(vec![1, 6]));
        let output = model.net.forward(&input);
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn save_load_roundtrip() {
        let model = DreamModel::new(2, 2, 32, 1, 42);
        let dir = std::env::temp_dir().join("phyz_dream_test_save_load");
        let _ = std::fs::remove_dir_all(&dir);
        model.save(&dir).unwrap();

        let loaded = DreamModel::load(&dir).unwrap();
        assert_eq!(loaded.meta().nq, 2);
        assert_eq!(loaded.meta().nv, 2);
        assert_eq!(loaded.meta().hidden_dim, 32);

        // Weights should match
        let s1 = model.net.state_dict();
        let s2 = loaded.net.state_dict();
        assert_eq!(s1.len(), s2.len());
        for ((n1, t1), (n2, t2)) in s1.iter().zip(s2.iter()) {
            assert_eq!(n1, n2);
            assert_eq!(t1.data(), t2.data());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn predict_returns_correct_dim() {
        let phyz_model = crate::tests::make_double_pendulum();
        let mut dream = DreamModel::new(2, 2, 32, 1, 42);
        let state = phyz_model.default_state();
        let qdd = dream.predict(&phyz_model, &state);
        assert_eq!(qdd.len(), 2);
    }
}
