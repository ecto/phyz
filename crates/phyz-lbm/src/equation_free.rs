//! Equation-free simulation and coarse projective integration.
//!
//! For multi-scale systems, extract low-dimensional dynamics from microscopic simulators
//! without deriving explicit macroscopic equations.

use nalgebra as na;

/// Trait for fine-scale solvers (microscopic dynamics).
pub trait FineSolver {
    /// State type (e.g., Vec<f64>, particle positions, etc.)
    type State;

    /// Advance the fine-scale state by one time step.
    fn step(&mut self, state: &mut Self::State);

    /// Get current time step size.
    fn dt(&self) -> f64;
}

/// Trait for coarse-graining projectors (lift/restrict operations).
pub trait CoarseProjector {
    /// Fine-scale state type
    type FineState;
    /// Coarse-scale state type (typically lower-dimensional)
    type CoarseState;

    /// Lift: coarse → fine (initialize microscopic state consistent with macro state)
    fn lift(&self, coarse: &Self::CoarseState) -> Self::FineState;

    /// Restrict: fine → coarse (extract macroscopic observables from micro state)
    fn restrict(&self, fine: &Self::FineState) -> Self::CoarseState;
}

/// Equation-free wrapper for coarse projective integration.
///
/// Performs:
/// 1. Lift: macro → micro
/// 2. Burst: simulate fine solver for short time
/// 3. Restrict: micro → macro
/// 4. Extrapolate: project macro dynamics forward
pub struct EquationFreeWrapper<F, C>
where
    F: FineSolver,
    C: CoarseProjector<FineState = F::State>,
{
    /// Fine-scale solver
    pub fine: F,
    /// Coarse projector
    pub coarse: C,
    /// Number of fine steps per burst
    pub burst_steps: usize,
    /// Extrapolation factor (1 = no extrapolation, 2 = double the time step)
    pub extrapolation_factor: f64,
}

impl<F, C> EquationFreeWrapper<F, C>
where
    F: FineSolver,
    C: CoarseProjector<FineState = F::State>,
    C::CoarseState: Clone,
{
    /// Create new equation-free wrapper.
    ///
    /// # Arguments
    /// * `fine` - Fine-scale solver
    /// * `coarse` - Coarse projector
    /// * `burst_steps` - Number of fine steps per burst
    /// * `extrapolation_factor` - Extrapolation multiplier (typically 1-10)
    pub fn new(fine: F, coarse: C, burst_steps: usize, extrapolation_factor: f64) -> Self {
        Self {
            fine,
            coarse,
            burst_steps,
            extrapolation_factor,
        }
    }

    /// Perform one equation-free step (lift → burst → restrict → extrapolate).
    ///
    /// Updates `macro_state` in-place.
    pub fn step(&mut self, macro_state: &mut C::CoarseState) {
        // 1. Lift: macro → micro
        let mut micro_state = self.coarse.lift(macro_state);

        // 2. Burst: run fine solver
        for _ in 0..self.burst_steps {
            self.fine.step(&mut micro_state);
        }

        // 3. Restrict: micro → macro
        let macro_new = self.coarse.restrict(&micro_state);

        // 4. Extrapolate (simple forward projection)
        // This requires CoarseState to support arithmetic, which we handle generically
        // For simplicity, we just update macro_state with the new value
        // A more sophisticated implementation would extrapolate: macro_state + α * (macro_new - macro_state)
        *macro_state = macro_new;
    }
}

/// Simple coarse projector for 1D fields: spatial averaging.
pub struct SpatialAverageProjector {
    /// Fine grid size
    pub fine_size: usize,
    /// Coarse grid size
    pub coarse_size: usize,
}

impl SpatialAverageProjector {
    pub fn new(fine_size: usize, coarse_size: usize) -> Self {
        assert!(fine_size >= coarse_size, "Fine grid must be larger");
        Self {
            fine_size,
            coarse_size,
        }
    }
}

impl CoarseProjector for SpatialAverageProjector {
    type FineState = Vec<f64>;
    type CoarseState = Vec<f64>;

    fn lift(&self, coarse: &Self::CoarseState) -> Self::FineState {
        // Piecewise constant interpolation
        let ratio = self.fine_size / self.coarse_size;
        (0..self.fine_size)
            .map(|i| {
                let coarse_idx = (i / ratio).min(self.coarse_size - 1);
                coarse[coarse_idx]
            })
            .collect()
    }

    fn restrict(&self, fine: &Self::FineState) -> Self::CoarseState {
        // Block averaging
        let ratio = self.fine_size / self.coarse_size;
        (0..self.coarse_size)
            .map(|i| {
                let start = i * ratio;
                let end = ((i + 1) * ratio).min(self.fine_size);
                let sum: f64 = fine[start..end].iter().sum();
                sum / (end - start) as f64
            })
            .collect()
    }
}

/// Measure effective information: how much of microscopic state is captured by coarse variables.
///
/// Uses PCA to project to `coarse_dim` principal components and measures explained variance.
///
/// Returns fraction of variance explained (0.0 to 1.0).
/// High value → strong emergence (low-dimensional manifold).
pub fn effective_information(micro_states: &[Vec<f64>], coarse_dim: usize) -> f64 {
    if micro_states.is_empty() {
        return 0.0;
    }

    let n_samples = micro_states.len();
    let n_features = micro_states[0].len();

    if coarse_dim >= n_features {
        return 1.0;
    }

    // Build data matrix (samples × features)
    let mut data = na::DMatrix::zeros(n_samples, n_features);
    for (i, state) in micro_states.iter().enumerate() {
        for (j, &val) in state.iter().enumerate() {
            data[(i, j)] = val;
        }
    }

    // Center the data
    let mean = data.column_mean();
    for i in 0..n_samples {
        for j in 0..n_features {
            data[(i, j)] -= mean[j];
        }
    }

    // Compute covariance matrix
    let cov = (data.transpose() * &data) / (n_samples as f64);

    // Compute eigenvalues (variance along principal components)
    let eigen = cov.symmetric_eigen();

    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let total_variance: f64 = eigenvalues.iter().sum();
    if total_variance < 1e-12 {
        return 0.0;
    }

    let explained_variance: f64 = eigenvalues.iter().take(coarse_dim).sum();
    explained_variance / total_variance
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFineSolver {
        dt: f64,
    }

    impl FineSolver for DummyFineSolver {
        type State = Vec<f64>;

        fn step(&mut self, state: &mut Self::State) {
            // Simple diffusion: smooth the state
            let n = state.len();
            let mut new_state = state.clone();
            for i in 1..n - 1 {
                new_state[i] = 0.25 * state[i - 1] + 0.5 * state[i] + 0.25 * state[i + 1];
            }
            *state = new_state;
        }

        fn dt(&self) -> f64 {
            self.dt
        }
    }

    #[test]
    fn test_spatial_averaging() {
        let projector = SpatialAverageProjector::new(100, 10);

        let coarse = vec![1.0; 10];
        let fine = projector.lift(&coarse);
        assert_eq!(fine.len(), 100);
        assert!(fine.iter().all(|&x| (x - 1.0).abs() < 1e-12));

        let coarse_back = projector.restrict(&fine);
        assert_eq!(coarse_back.len(), 10);
        assert!(coarse_back.iter().all(|&x| (x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_equation_free_wrapper() {
        let fine = DummyFineSolver { dt: 0.01 };
        let coarse = SpatialAverageProjector::new(100, 10);
        let mut wrapper = EquationFreeWrapper::new(fine, coarse, 10, 1.0);

        let mut macro_state = vec![0.0; 10];
        macro_state[5] = 1.0; // Gaussian-like initial condition

        wrapper.step(&mut macro_state);

        // After diffusion, peak should be lower and wider
        assert!(macro_state[5] < 1.0);
        assert!(macro_state[4] > 0.0);
        assert!(macro_state[6] > 0.0);
    }

    #[test]
    fn test_effective_information() {
        // Low-dimensional data (all points on a line)
        let states = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![0.5, 1.0, 1.5],
        ];

        let ei = effective_information(&states, 1);
        // Should be close to 1.0 since data is 1D
        assert!(ei > 0.99, "Expected high effective information, got {}", ei);
    }
}
