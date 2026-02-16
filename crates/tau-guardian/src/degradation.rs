//! Solver quality assessment and graceful degradation strategies.
//!
//! Monitors simulation health and adapts solver parameters when accuracy drops.

use crate::conservation::ConservationMonitor;

/// Solver quality levels based on conservation law violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverQuality {
    /// Error < 0.01 * tolerance — simulation is excellent
    Excellent,
    /// 0.01 * tolerance < error < 0.1 * tolerance — good quality
    Good,
    /// 0.1 * tolerance < error < tolerance — marginal but acceptable
    Marginal,
    /// tolerance < error < 10 * tolerance — poor, needs intervention
    Poor,
    /// error > 10 * tolerance — critical, multiple violations
    Critical,
}

impl SolverQuality {
    /// Assess quality based on conservation error relative to tolerance.
    pub fn assess(error: f64, tolerance: f64) -> Self {
        let ratio = error / tolerance;

        if ratio < 0.01 {
            Self::Excellent
        } else if ratio < 0.1 {
            Self::Good
        } else if ratio < 1.0 {
            Self::Marginal
        } else if ratio < 10.0 {
            Self::Poor
        } else {
            Self::Critical
        }
    }

    /// Assess quality from conservation monitor.
    pub fn assess_from_monitor(monitor: &ConservationMonitor, tolerance: f64) -> Self {
        let error = monitor.max_relative_error();
        Self::assess(error, tolerance)
    }

    /// Check if quality is acceptable (Excellent, Good, or Marginal).
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::Excellent | Self::Good | Self::Marginal)
    }

    /// Check if intervention is needed (Poor or Critical).
    pub fn needs_intervention(&self) -> bool {
        matches!(self, Self::Poor | Self::Critical)
    }
}

/// Strategy for responding to degraded solver quality.
#[derive(Debug, Clone)]
pub struct DegradationStrategy {
    /// Current quality level
    pub quality: SolverQuality,
    /// Suggested dt reduction factor
    pub dt_factor: f64,
    /// Suggested solver change
    pub solver_suggestion: SolverSuggestion,
    /// Whether to pause simulation
    pub should_pause: bool,
}

/// Solver type suggestions based on quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverSuggestion {
    /// Keep current solver
    KeepCurrent,
    /// Switch to more robust RK4
    SwitchToRk4,
    /// Switch to faster RK2 (if accuracy sufficient)
    SwitchToRk2,
    /// Switch to symplectic integrator
    SwitchToSymplectic,
}

impl DegradationStrategy {
    /// Create degradation strategy for given quality.
    pub fn for_quality(quality: SolverQuality) -> Self {
        match quality {
            SolverQuality::Excellent => Self {
                quality,
                dt_factor: 1.5, // Can increase dt
                solver_suggestion: SolverSuggestion::SwitchToRk2,
                should_pause: false,
            },
            SolverQuality::Good => Self {
                quality,
                dt_factor: 1.0, // Keep dt
                solver_suggestion: SolverSuggestion::KeepCurrent,
                should_pause: false,
            },
            SolverQuality::Marginal => Self {
                quality,
                dt_factor: 0.5, // Reduce dt by 50%
                solver_suggestion: SolverSuggestion::KeepCurrent,
                should_pause: false,
            },
            SolverQuality::Poor => Self {
                quality,
                dt_factor: 0.1, // Reduce dt by 90%
                solver_suggestion: SolverSuggestion::SwitchToRk4,
                should_pause: false,
            },
            SolverQuality::Critical => Self {
                quality,
                dt_factor: 0.05, // Aggressive reduction
                solver_suggestion: SolverSuggestion::SwitchToSymplectic,
                should_pause: true,
            },
        }
    }

    /// Apply degradation strategy to model timestep.
    pub fn apply_dt_adjustment(&self, current_dt: f64) -> f64 {
        current_dt * self.dt_factor
    }
}

/// Auto-switching controller that changes solvers based on quality.
#[derive(Debug, Clone)]
pub struct AutoSwitchController {
    /// Tolerance for quality assessment
    pub tolerance: f64,
    /// Number of consecutive poor-quality steps before switching
    pub switch_threshold: usize,
    /// Current count of poor-quality steps
    poor_count: usize,
    /// Current count of excellent-quality steps
    excellent_count: usize,
}

impl AutoSwitchController {
    /// Create a new auto-switch controller.
    pub fn new(tolerance: f64, switch_threshold: usize) -> Self {
        Self {
            tolerance,
            switch_threshold,
            poor_count: 0,
            excellent_count: 0,
        }
    }

    /// Update controller with new quality and return strategy.
    pub fn update(&mut self, quality: SolverQuality) -> DegradationStrategy {
        match quality {
            SolverQuality::Poor | SolverQuality::Critical => {
                self.poor_count += 1;
                self.excellent_count = 0;
            }
            SolverQuality::Excellent => {
                self.excellent_count += 1;
                self.poor_count = 0;
            }
            _ => {
                // Good or Marginal: reset both counters
                self.poor_count = 0;
                self.excellent_count = 0;
            }
        }

        // Determine strategy
        let mut strategy = DegradationStrategy::for_quality(quality);

        // Override solver suggestion based on consecutive patterns
        if self.poor_count >= self.switch_threshold {
            strategy.solver_suggestion = SolverSuggestion::SwitchToRk4;
        } else if self.excellent_count >= self.switch_threshold {
            strategy.solver_suggestion = SolverSuggestion::SwitchToRk2;
        }

        strategy
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.poor_count = 0;
        self.excellent_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessment() {
        assert_eq!(SolverQuality::assess(0.005, 1.0), SolverQuality::Excellent);
        assert_eq!(SolverQuality::assess(0.05, 1.0), SolverQuality::Good);
        assert_eq!(SolverQuality::assess(0.5, 1.0), SolverQuality::Marginal);
        assert_eq!(SolverQuality::assess(5.0, 1.0), SolverQuality::Poor);
        assert_eq!(SolverQuality::assess(50.0, 1.0), SolverQuality::Critical);
    }

    #[test]
    fn test_degradation_strategy() {
        let excellent = DegradationStrategy::for_quality(SolverQuality::Excellent);
        assert!(excellent.dt_factor > 1.0);
        assert!(!excellent.should_pause);

        let critical = DegradationStrategy::for_quality(SolverQuality::Critical);
        assert!(critical.dt_factor < 0.1);
        assert!(critical.should_pause);
    }

    #[test]
    fn test_auto_switch_controller() {
        let mut controller = AutoSwitchController::new(1e-4, 3);

        // Three poor steps should trigger switch to RK4
        for _ in 0..2 {
            let strategy = controller.update(SolverQuality::Poor);
            assert_eq!(strategy.solver_suggestion, SolverSuggestion::SwitchToRk4);
        }

        let strategy = controller.update(SolverQuality::Poor);
        assert_eq!(strategy.solver_suggestion, SolverSuggestion::SwitchToRk4);

        // Reset with good quality
        controller.update(SolverQuality::Good);

        // Three excellent steps should trigger switch to RK2
        for _ in 0..3 {
            controller.update(SolverQuality::Excellent);
        }

        let strategy = controller.update(SolverQuality::Excellent);
        assert_eq!(strategy.solver_suggestion, SolverSuggestion::SwitchToRk2);
    }
}
