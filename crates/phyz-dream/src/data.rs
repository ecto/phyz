use phyz_model::Model;
use tang_tensor::{Shape, Tensor};
use tang_train::{Rng, TensorDataset};

/// Configuration for random state sampling.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    pub q_range: [f64; 2],
    pub v_range: [f64; 2],
    pub ctrl_range: [f64; 2],
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            q_range: [-std::f64::consts::PI, std::f64::consts::PI],
            v_range: [-5.0, 5.0],
            ctrl_range: [-10.0, 10.0],
        }
    }
}

/// Returns the input dimensionality for a model: nq + nv + nv (q, v, ctrl).
pub fn input_dim(model: &Model) -> usize {
    model.nq + model.nv + model.nv
}

/// Sample a random state respecting joint limits when available.
pub fn sample_state(model: &Model, rng: &mut Rng, config: &SampleConfig) -> phyz_model::State {
    let mut state = model.default_state();

    // Sample q per joint, respecting limits
    for (i, joint) in model.joints.iter().enumerate() {
        let offset = model.q_offsets[i];
        let ndof = joint.ndof();
        for d in 0..ndof {
            let (lo, hi) = if let Some(limits) = joint.limits {
                (limits[0], limits[1])
            } else {
                (config.q_range[0], config.q_range[1])
            };
            state.q[offset + d] = lo + rng.next_f64() * (hi - lo);
        }
    }

    // Sample v
    for i in 0..model.nv {
        state.v[i] = config.v_range[0] + rng.next_f64() * (config.v_range[1] - config.v_range[0]);
    }

    // Sample ctrl respecting actuator ctrl_range
    for (i, act) in model.actuators.iter().enumerate() {
        let (lo, hi) = if let Some(cr) = act.ctrl_range {
            (cr[0], cr[1])
        } else {
            (config.ctrl_range[0], config.ctrl_range[1])
        };
        state.ctrl[i] = lo + rng.next_f64() * (hi - lo);
    }
    // Fill remaining ctrl dimensions (no actuator) with config range
    for i in model.actuators.len()..model.nv {
        state.ctrl[i] =
            config.ctrl_range[0] + rng.next_f64() * (config.ctrl_range[1] - config.ctrl_range[0]);
    }

    state
}

/// Generate a dataset of (q,v,ctrl) -> qdd pairs using ABA ground truth.
pub fn generate_dataset(
    model: &Model,
    n_samples: usize,
    config: &SampleConfig,
    seed: u64,
) -> TensorDataset<f32> {
    let in_dim = input_dim(model);
    let out_dim = model.nv;
    let mut rng = Rng::new(seed);

    let mut input_data = Vec::with_capacity(n_samples * in_dim);
    let mut output_data = Vec::with_capacity(n_samples * out_dim);

    for _ in 0..n_samples {
        let state = sample_state(model, &mut rng, config);
        let qdd = phyz_rigid::aba(model, &state);

        // Flatten input: q ++ v ++ ctrl
        for i in 0..model.nq {
            input_data.push(state.q[i] as f32);
        }
        for i in 0..model.nv {
            input_data.push(state.v[i] as f32);
        }
        for i in 0..model.nv {
            input_data.push(state.ctrl[i] as f32);
        }

        // Output: qdd
        for i in 0..out_dim {
            output_data.push(qdd[i] as f32);
        }
    }

    let inputs = Tensor::new(input_data, Shape::new(vec![n_samples, in_dim]));
    let targets = Tensor::new(output_data, Shape::new(vec![n_samples, out_dim]));
    TensorDataset::new(inputs, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_double_pendulum;
    use tang_train::Dataset;

    #[test]
    fn dataset_shapes() {
        let model = make_double_pendulum();
        let ds = generate_dataset(&model, 100, &SampleConfig::default(), 42);
        let (inp, tgt): (tang_tensor::Tensor<f32>, tang_tensor::Tensor<f32>) = ds.get(0);
        assert_eq!(inp.shape().dims(), &[input_dim(&model)]);
        assert_eq!(tgt.shape().dims(), &[model.nv]);
    }

    #[test]
    fn deterministic_seed() {
        let model = make_double_pendulum();
        let cfg = SampleConfig::default();
        let ds1 = generate_dataset(&model, 10, &cfg, 123);
        let ds2 = generate_dataset(&model, 10, &cfg, 123);
        for i in 0..10 {
            let (a, _): (tang_tensor::Tensor<f32>, _) = ds1.get(i);
            let (b, _): (tang_tensor::Tensor<f32>, _) = ds2.get(i);
            assert_eq!(a.data(), b.data());
        }
    }

    #[test]
    fn input_dim_correct() {
        let model = make_double_pendulum();
        // double pendulum: nq=2, nv=2 -> input_dim = 2+2+2 = 6
        assert_eq!(input_dim(&model), 6);
    }
}
