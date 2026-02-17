//! Optimization algorithms for parameter estimation.

use crate::trajectory_matcher::{PhysicsParams, TrajectoryMatcher};
use phyz_model::Model;

/// Configuration for optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Convergence threshold (stop if loss change < threshold).
    pub convergence_threshold: f64,
    /// Epsilon for numerical gradient computation.
    pub gradient_epsilon: f64,
    /// Print progress every N iterations.
    pub print_every: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            gradient_epsilon: 1e-6,
            print_every: 10,
        }
    }
}

/// Trait for optimization algorithms.
pub trait Optimizer {
    /// Optimize physics parameters to minimize trajectory matching loss.
    fn optimize(
        &self,
        model: &Model,
        matcher: &TrajectoryMatcher,
        params: &mut PhysicsParams,
    ) -> OptimizationResult;
}

/// Result of optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final loss value.
    pub final_loss: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether optimization converged.
    pub converged: bool,
    /// Loss history.
    pub loss_history: Vec<f64>,
}

/// Gradient descent optimizer.
pub struct GradientDescentOptimizer {
    /// Configuration.
    pub config: OptimizerConfig,
}

impl GradientDescentOptimizer {
    /// Create a new gradient descent optimizer with default config.
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Create optimizer with custom configuration.
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self { config }
    }
}

impl Default for GradientDescentOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for GradientDescentOptimizer {
    fn optimize(
        &self,
        model: &Model,
        matcher: &TrajectoryMatcher,
        params: &mut PhysicsParams,
    ) -> OptimizationResult {
        let mut loss_history = Vec::new();
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.config.max_iterations {
            // Compute current loss
            let loss = matcher.compute_loss(model, params);
            loss_history.push(loss);

            // Check convergence
            if (prev_loss - loss).abs() < self.config.convergence_threshold {
                return OptimizationResult {
                    final_loss: loss,
                    iterations: iter + 1,
                    converged: true,
                    loss_history,
                };
            }

            // Print progress
            if iter % self.config.print_every == 0 {
                println!("Iteration {}: loss = {:.6e}", iter, loss);
            }

            // Compute gradient
            let grad = matcher.numerical_gradient(model, params, self.config.gradient_epsilon);

            // Gradient descent update
            let mut param_vec = params.to_vec();
            for (i, g) in grad.iter().enumerate() {
                param_vec[i] -= self.config.learning_rate * g;
            }
            params.from_vec(&param_vec);

            prev_loss = loss;
        }

        let final_loss = matcher.compute_loss(model, params);
        OptimizationResult {
            final_loss,
            iterations: self.config.max_iterations,
            converged: false,
            loss_history,
        }
    }
}

/// Adam optimizer (adaptive moment estimation).
pub struct AdamOptimizer {
    /// Configuration.
    pub config: OptimizerConfig,
    /// Beta1 parameter for first moment.
    pub beta1: f64,
    /// Beta2 parameter for second moment.
    pub beta2: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer.
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Create Adam optimizer with custom configuration.
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            config,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for AdamOptimizer {
    fn optimize(
        &self,
        model: &Model,
        matcher: &TrajectoryMatcher,
        params: &mut PhysicsParams,
    ) -> OptimizationResult {
        let mut loss_history = Vec::new();
        let mut prev_loss = f64::INFINITY;

        let n_params = params.to_vec().len();
        let mut m = vec![0.0; n_params]; // First moment
        let mut v = vec![0.0; n_params]; // Second moment

        for iter in 0..self.config.max_iterations {
            // Compute current loss
            let loss = matcher.compute_loss(model, params);
            loss_history.push(loss);

            // Check convergence
            if (prev_loss - loss).abs() < self.config.convergence_threshold {
                return OptimizationResult {
                    final_loss: loss,
                    iterations: iter + 1,
                    converged: true,
                    loss_history,
                };
            }

            // Print progress
            if iter % self.config.print_every == 0 {
                println!("Iteration {}: loss = {:.6e}", iter, loss);
            }

            // Compute gradient
            let grad = matcher.numerical_gradient(model, params, self.config.gradient_epsilon);

            // Update biased first and second moment estimates
            for i in 0..n_params {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            }

            // Compute bias-corrected moment estimates
            let t = (iter + 1) as f64;
            let m_hat: Vec<f64> = m
                .iter()
                .map(|&mi| mi / (1.0 - self.beta1.powf(t)))
                .collect();
            let v_hat: Vec<f64> = v
                .iter()
                .map(|&vi| vi / (1.0 - self.beta2.powf(t)))
                .collect();

            // Update parameters
            let mut param_vec = params.to_vec();
            for i in 0..n_params {
                param_vec[i] -=
                    self.config.learning_rate * m_hat[i] / (v_hat[i].sqrt() + self.epsilon);
            }
            params.from_vec(&param_vec);

            prev_loss = loss;
        }

        let final_loss = matcher.compute_loss(model, params);
        OptimizationResult {
            final_loss,
            iterations: self.config.max_iterations,
            converged: false,
            loss_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory_matcher::{Trajectory, TrajectoryMatcher, TrajectoryObservation};
    use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, -0.5, 0.0)),
            )
            .build()
    }

    #[test]
    fn test_gradient_descent_optimizer() {
        let model = make_pendulum();

        let mut traj = Trajectory::new();
        traj.add_observation(TrajectoryObservation {
            time: 0.0,
            body_idx: 0,
            position: Some([0.0, -0.5, 0.0]),
            orientation: None,
            joint_angles: Some(vec![0.0]),
        });

        let matcher = TrajectoryMatcher::new(traj, 0.001);

        let mut params = PhysicsParams::new();
        params.set_model_param("test", 1.0);

        let config = OptimizerConfig {
            max_iterations: 5,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            gradient_epsilon: 1e-6,
            print_every: 1,
        };

        let optimizer = GradientDescentOptimizer::with_config(config);
        let result = optimizer.optimize(&model, &matcher, &mut params);

        assert!(result.iterations <= 5);
        assert_eq!(result.loss_history.len(), result.iterations);
    }

    #[test]
    fn test_adam_optimizer() {
        let model = make_pendulum();

        let mut traj = Trajectory::new();
        traj.add_observation(TrajectoryObservation {
            time: 0.0,
            body_idx: 0,
            position: Some([0.0, -0.5, 0.0]),
            orientation: None,
            joint_angles: Some(vec![0.0]),
        });

        let matcher = TrajectoryMatcher::new(traj, 0.001);

        let mut params = PhysicsParams::new();
        params.set_model_param("test", 1.0);

        let config = OptimizerConfig {
            max_iterations: 5,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            gradient_epsilon: 1e-6,
            print_every: 1,
        };

        let optimizer = AdamOptimizer::with_config(config);
        let result = optimizer.optimize(&model, &matcher, &mut params);

        assert!(result.iterations <= 5);
        assert_eq!(result.loss_history.len(), result.iterations);
    }
}
