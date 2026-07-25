//! Deterministic, backend-independent RNG.
//!
//! Every random draw in an episode is a pure function of
//! `(base_seed, env_index, episode_index, draw_index)`. Nothing depends on
//! wall-clock time, thread scheduling, or how many environments happen to be
//! resetting in the same step. That is what makes rollouts bit-reproducible
//! across `num_envs` values and across backends — see
//! `docs/design/batched-envs.md`, "Determinism".

/// SplitMix64 — the standard cheap 64-bit mixer.
#[inline]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A counter-based RNG stream. Stateless given its key: reproducible by
/// construction, and safe to reconstruct on any backend.
#[derive(Debug, Clone, Copy)]
pub struct Stream {
    key: u64,
    counter: u64,
}

impl Stream {
    /// Derive the stream for one environment's one episode.
    pub const fn new(base_seed: u64, env_index: u64, episode_index: u64) -> Self {
        let key = splitmix64(base_seed ^ splitmix64(env_index.wrapping_mul(0x2545_F491_4F6C_DD1D)))
            ^ splitmix64(episode_index.wrapping_add(0x1234_5678_9ABC_DEF0));
        Self { key, counter: 0 }
    }

    /// Next raw 64 bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let v = splitmix64(self.key ^ splitmix64(self.counter));
        self.counter = self.counter.wrapping_add(1);
        v
    }

    /// Uniform in `[0, 1)`. Uses 53 bits, so the mapping is exact in f64 and
    /// identical on every platform (no transcendental functions involved).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    /// Uniform in `[-half_range, half_range)`.
    #[inline]
    pub fn uniform_sym(&mut self, half_range: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * half_range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_reproducible() {
        let mut a = Stream::new(42, 3, 7);
        let mut b = Stream::new(42, 3, 7);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn streams_are_independent_across_envs_and_episodes() {
        let a = Stream::new(42, 3, 7).next_u64();
        let b = Stream::new(42, 4, 7).next_u64();
        let c = Stream::new(42, 3, 8).next_u64();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn uniform_is_in_range() {
        let mut s = Stream::new(1, 0, 0);
        for _ in 0..10_000 {
            let x = s.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
