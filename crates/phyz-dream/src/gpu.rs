//! GPU-accelerated batched inference for DreamModel via tang-compute.

use tang_compute::device::ComputeDevice;
use tang_compute::modules::Linear;
use tang_compute::tensor::ComputeTensor;
use tang_expr::node::ExprId;

use crate::model::DreamModel;
use tang::Scalar;
use tang_train::Module;

/// GPU-resident MLP for batched surrogate dynamics inference.
///
/// Holds Linear layers on device. Forward pass applies:
/// Linear → SiLU → ... → Linear (no activation on last layer).
/// Normalization is handled on CPU before/after GPU forward.
pub struct GpuDreamModel<D: ComputeDevice> {
    layers: Vec<Linear<D::Buffer>>,
    input_mean: Vec<f32>,
    input_inv_std: Vec<f32>,
    output_mean: Vec<f32>,
    output_std: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    n_layers: usize,
}

impl<D: ComputeDevice> GpuDreamModel<D> {
    /// Upload a trained CPU DreamModel to the GPU device.
    pub fn from_cpu(dev: &D, cpu: &DreamModel) -> Self {
        let named = cpu.net.named_parameters();
        let find = |name: &str| -> Vec<f32> {
            named
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing parameter: {name}"))
                .1
                .data
                .data()
                .to_vec()
        };

        let meta = cpu.meta();
        let in_dim = meta.nq + meta.nv + meta.nv;
        let out_dim = meta.nv;

        let mut layers = Vec::new();
        let mut dims = vec![in_dim];
        for _ in 0..meta.n_hidden {
            dims.push(meta.hidden_dim);
        }
        dims.push(out_dim);

        // Sequential indices: Linear at 0, 2, 4, ... (SiLU at 1, 3, ...)
        for (i, window) in dims.windows(2).enumerate() {
            let seq_idx = i * 2;
            let w = find(&format!("{seq_idx}.weight"));
            let b = find(&format!("{seq_idx}.bias"));
            let lin = Linear::new(dev, &w, &b, window[0], window[1]);
            layers.push(lin);
        }

        let n_layers = layers.len();

        let input_inv_std: Vec<f32> = meta.input_norm.std.iter().map(|s| 1.0 / s).collect();

        Self {
            layers,
            input_mean: meta.input_norm.mean.clone(),
            input_inv_std,
            output_mean: meta.output_norm.mean.clone(),
            output_std: meta.output_norm.std.clone(),
            in_dim,
            out_dim,
            n_layers,
        }
    }

    /// Batched forward: raw inputs [batch, in_dim] → denormalized outputs [batch, out_dim].
    ///
    /// Takes **unnormalized** flat f32 data (q ++ v ++ ctrl per sample).
    /// Returns denormalized accelerations.
    pub fn predict_batch(
        &self,
        dev: &D,
        raw_inputs: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        assert_eq!(raw_inputs.len(), batch * self.in_dim);

        // CPU normalize
        let mut normalized = Vec::with_capacity(raw_inputs.len());
        for row in raw_inputs.chunks(self.in_dim) {
            for (j, &x) in row.iter().enumerate() {
                normalized.push((x - self.input_mean[j]) * self.input_inv_std[j]);
            }
        }

        // Upload + GPU forward
        let input_gpu = ComputeTensor::from_data(dev, &normalized, &[batch, self.in_dim]);
        let output_gpu = self.forward_gpu(dev, &input_gpu);

        // Download + CPU denormalize
        let raw_out = output_gpu.to_vec();
        let mut denormalized = Vec::with_capacity(raw_out.len());
        for row in raw_out.chunks(self.out_dim) {
            for (j, &x) in row.iter().enumerate() {
                denormalized.push(x * self.output_std[j] + self.output_mean[j]);
            }
        }

        denormalized
    }

    /// GPU-only forward on pre-normalized data. No upload/download.
    pub fn forward_gpu(
        &self,
        dev: &D,
        input: &ComputeTensor<D::Buffer>,
    ) -> ComputeTensor<D::Buffer> {
        let mut x = self.layers[0].forward_2d(dev, input);

        for i in 1..self.n_layers {
            x = silu(dev, &x);
            x = self.layers[i].forward_2d(dev, &x);
        }

        x
    }
}

/// SiLU activation: x * sigmoid(x), fused into a single kernel.
fn silu<D: ComputeDevice>(
    dev: &D,
    x: &ComputeTensor<D::Buffer>,
) -> ComputeTensor<D::Buffer> {
    let buf = dev.elementwise(
        &[&x.buffer],
        x.numel(),
        &|ids: &[ExprId]| {
            let one = ExprId::from_f64(1.0);
            let neg = -ids[0];
            let exp_neg = Scalar::exp(neg);
            let sigmoid = one / (one + exp_neg);
            ids[0] * sigmoid
        },
    );
    ComputeTensor::from_buffer(buf, x.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_compute::CpuDevice;

    #[test]
    fn gpu_forward_shape() {
        let dev = CpuDevice::new();
        let cpu_model = DreamModel::new(2, 2, 64, 2, 42);
        let gpu_model = GpuDreamModel::from_cpu(&dev, &cpu_model);

        let raw = vec![0.0f32; 6 * 4];
        let out = gpu_model.predict_batch(&dev, &raw, 4);
        assert_eq!(out.len(), 4 * 2);
    }

    #[test]
    fn gpu_cpu_match() {
        let dev = CpuDevice::new();

        let phyz_model = crate::tests::make_double_pendulum();
        let config = crate::train::TrainConfig {
            n_samples: 1000,
            hidden_dim: 32,
            n_hidden: 1,
            batch_size: 64,
            epochs: 10,
            lr: 1e-3,
            weight_decay: 0.0,
            seed: 42,
            sample_config: crate::data::SampleConfig::default(),
        };
        let (mut cpu_dream, _) = crate::train::train(&phyz_model, &config);

        let gpu_dream = GpuDreamModel::from_cpu(&dev, &cpu_dream);

        // Generate test states
        let mut rng = tang_train::Rng::new(999);
        let sample_cfg = crate::data::SampleConfig::default();
        let n_test = 16;
        let _in_dim = crate::data::input_dim(&phyz_model);

        let mut raw_inputs = Vec::new();
        let mut states = Vec::new();
        for _ in 0..n_test {
            let s = crate::data::sample_state(&phyz_model, &mut rng, &sample_cfg);
            for i in 0..phyz_model.nq {
                raw_inputs.push(s.q[i] as f32);
            }
            for i in 0..phyz_model.nv {
                raw_inputs.push(s.v[i] as f32);
            }
            for i in 0..phyz_model.nv {
                raw_inputs.push(s.ctrl[i] as f32);
            }
            states.push(s);
        }

        // GPU batched
        let gpu_results = gpu_dream.predict_batch(&dev, &raw_inputs, n_test);

        // CPU per-sample
        let mut max_diff = 0.0f32;
        for (i, state) in states.iter().enumerate() {
            let cpu_qdd = cpu_dream.predict(&phyz_model, state);
            for j in 0..phyz_model.nv {
                let gpu_val = gpu_results[i * phyz_model.nv + j];
                let cpu_val = cpu_qdd[j] as f32;
                let diff = (gpu_val - cpu_val).abs();
                max_diff = max_diff.max(diff);
            }
        }

        assert!(
            max_diff < 0.1,
            "GPU and CPU should match, max_diff={max_diff}"
        );
    }

    #[test]
    fn gpu_batch_sizes() {
        let dev = CpuDevice::new();
        let cpu_model = DreamModel::new(2, 2, 32, 1, 42);
        let gpu_model = GpuDreamModel::from_cpu(&dev, &cpu_model);

        for batch in [1, 8, 64, 256] {
            let raw = vec![0.0f32; 6 * batch];
            let out = gpu_model.predict_batch(&dev, &raw, batch);
            assert_eq!(out.len(), batch * 2, "batch={batch}");
        }
    }
}
