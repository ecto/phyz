//! Subcycling and multi-timescale integration.
//!
//! Implements r-RESPA (Reference System Propagator Algorithm) for
//! partitioning forces into fast and slow components.

/// Timescale classification for different physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeScale {
    /// Ultra-fast (e.g., electronic motion, high-frequency EM).
    UltraFast,
    /// Fast (e.g., bond vibrations, EM field updates).
    Fast,
    /// Medium (e.g., angle bending, soft collisions).
    Medium,
    /// Slow (e.g., rigid body motion, torsional rotation).
    Slow,
    /// Very slow (e.g., long-range forces, thermal diffusion).
    VerySlow,
}

impl TimeScale {
    /// Get typical timestep size for this scale (relative units).
    pub fn typical_dt_ratio(&self) -> usize {
        match self {
            TimeScale::UltraFast => 1,
            TimeScale::Fast => 10,
            TimeScale::Medium => 100,
            TimeScale::Slow => 1000,
            TimeScale::VerySlow => 10000,
        }
    }
}

/// Subcycling schedule for multi-timescale integration.
#[derive(Clone, Debug)]
pub struct SubcyclingSchedule {
    /// Base timestep (smallest unit).
    pub dt_base: f64,
    /// Subcycling ratios for each level.
    /// e.g., [1, 10, 100] means:
    /// - Level 0: dt_base (every step)
    /// - Level 1: 10 * dt_base (every 10 steps)
    /// - Level 2: 100 * dt_base (every 100 steps)
    pub ratios: Vec<usize>,
}

impl SubcyclingSchedule {
    /// Create a new subcycling schedule.
    pub fn new(dt_base: f64, ratios: Vec<usize>) -> Self {
        Self { dt_base, ratios }
    }

    /// Create a schedule from timescales.
    pub fn from_timescales(dt_base: f64, scales: &[TimeScale]) -> Self {
        let ratios = scales.iter().map(|s| s.typical_dt_ratio()).collect();
        Self { dt_base, ratios }
    }

    /// Get timestep for a given level.
    pub fn dt_for_level(&self, level: usize) -> f64 {
        if level < self.ratios.len() {
            self.dt_base * self.ratios[level] as f64
        } else {
            self.dt_base
        }
    }

    /// Check if a level should step at given global step count.
    pub fn should_step(&self, level: usize, global_step: usize) -> bool {
        if level >= self.ratios.len() {
            return true;
        }
        global_step.is_multiple_of(self.ratios[level])
    }

    /// Get number of substeps for a level relative to base.
    pub fn num_substeps(&self, level: usize) -> usize {
        if level < self.ratios.len() {
            self.ratios[level]
        } else {
            1
        }
    }
}

/// r-RESPA force partitioning.
///
/// Partitions forces into fast (short-range) and slow (long-range) components
/// for efficient multi-timescale integration.
pub struct RespaPartition {
    /// Cutoff distance for fast forces.
    pub r_short: f64,
    /// Cutoff distance for slow forces.
    pub r_long: f64,
}

impl RespaPartition {
    /// Create a new RESPA partition.
    pub fn new(r_short: f64, r_long: f64) -> Self {
        assert!(r_short < r_long, "r_short must be less than r_long");
        Self { r_short, r_long }
    }

    /// Classify a distance as fast, slow, or both.
    pub fn classify(&self, r: f64) -> ForceType {
        if r <= self.r_short {
            ForceType::Fast
        } else if r <= self.r_long {
            ForceType::Slow
        } else {
            ForceType::None
        }
    }
}

/// Force type classification for RESPA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceType {
    /// Fast-varying force (updated every substep).
    Fast,
    /// Slow-varying force (updated every outer step).
    Slow,
    /// No force (beyond cutoff).
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timescale_ratios() {
        assert_eq!(TimeScale::UltraFast.typical_dt_ratio(), 1);
        assert_eq!(TimeScale::Fast.typical_dt_ratio(), 10);
        assert_eq!(TimeScale::Medium.typical_dt_ratio(), 100);
        assert_eq!(TimeScale::Slow.typical_dt_ratio(), 1000);
    }

    #[test]
    fn test_subcycling_schedule() {
        let schedule = SubcyclingSchedule::new(0.001, vec![1, 10, 100]);

        assert_eq!(schedule.dt_for_level(0), 0.001);
        assert_eq!(schedule.dt_for_level(1), 0.01);
        assert_eq!(schedule.dt_for_level(2), 0.1);

        // Level 0 steps every time
        assert!(schedule.should_step(0, 0));
        assert!(schedule.should_step(0, 1));
        assert!(schedule.should_step(0, 10));

        // Level 1 steps every 10
        assert!(schedule.should_step(1, 0));
        assert!(!schedule.should_step(1, 1));
        assert!(schedule.should_step(1, 10));

        // Level 2 steps every 100
        assert!(schedule.should_step(2, 0));
        assert!(!schedule.should_step(2, 10));
        assert!(schedule.should_step(2, 100));
    }

    #[test]
    fn test_from_timescales() {
        let scales = vec![TimeScale::Fast, TimeScale::Medium, TimeScale::Slow];
        let schedule = SubcyclingSchedule::from_timescales(1e-6, &scales);

        assert_eq!(schedule.ratios, vec![10, 100, 1000]);
    }

    #[test]
    fn test_respa_partition() {
        let partition = RespaPartition::new(2.5, 10.0);

        assert_eq!(partition.classify(2.0), ForceType::Fast);
        assert_eq!(partition.classify(5.0), ForceType::Slow);
        assert_eq!(partition.classify(15.0), ForceType::None);
    }

    #[test]
    #[should_panic(expected = "r_short must be less than r_long")]
    fn test_respa_partition_invalid() {
        RespaPartition::new(10.0, 5.0);
    }

    #[test]
    fn test_num_substeps() {
        let schedule = SubcyclingSchedule::new(0.001, vec![1, 10, 100]);

        assert_eq!(schedule.num_substeps(0), 1);
        assert_eq!(schedule.num_substeps(1), 10);
        assert_eq!(schedule.num_substeps(2), 100);
    }
}
