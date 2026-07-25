//! Observation / action spaces.
//!
//! Deliberately minimal: RL practitioners only ever need a bounded box for
//! continuous control, and a box maps 1:1 onto `gymnasium.spaces.Box`, so the
//! Python binding is a field-for-field copy with no translation layer.

/// A continuous, axis-aligned box space. Bounds are per-element and always
/// materialised (no scalar broadcast) so the Python side can hand numpy a
/// contiguous array without allocating.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxSpace {
    /// Per-environment shape, excluding the leading `num_envs` axis.
    pub shape: Vec<usize>,
    /// Lower bounds, flattened, `len() == shape.product()`.
    pub low: Vec<f32>,
    /// Upper bounds, flattened, `len() == shape.product()`.
    pub high: Vec<f32>,
}

impl BoxSpace {
    /// A box with the same bounds on every element.
    pub fn uniform(dim: usize, low: f32, high: f32) -> Self {
        Self {
            shape: vec![dim],
            low: vec![low; dim],
            high: vec![high; dim],
        }
    }

    /// An effectively unbounded box, the usual choice for observations.
    pub fn unbounded(dim: usize) -> Self {
        Self::uniform(dim, f32::NEG_INFINITY, f32::INFINITY)
    }

    /// Bounds given explicitly per element.
    pub fn new(low: Vec<f32>, high: Vec<f32>) -> Self {
        assert_eq!(low.len(), high.len(), "low/high length mismatch");
        Self {
            shape: vec![low.len()],
            low,
            high,
        }
    }

    /// Number of scalars per environment.
    pub fn dim(&self) -> usize {
        self.shape.iter().product()
    }

    /// Clamp a flat batch (`num_envs * dim`) into the box, in place.
    pub fn clamp_batch(&self, data: &mut [f32]) {
        let d = self.dim();
        if d == 0 {
            return;
        }
        for (i, x) in data.iter_mut().enumerate() {
            let k = i % d;
            *x = x.clamp(self.low[k], self.high[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_batch_wraps_per_env() {
        let space = BoxSpace::new(vec![-1.0, 0.0], vec![1.0, 2.0]);
        let mut data = vec![-5.0, 5.0, 0.5, -1.0];
        space.clamp_batch(&mut data);
        assert_eq!(data, vec![-1.0, 2.0, 0.5, 0.0]);
    }
}
