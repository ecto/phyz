//! Counter-keyed RNG streams: one independent stream per `(seed, pass, μ, site)`
//! so parallel updates are reproducible regardless of thread scheduling.

/// xoshiro256++ seeded through splitmix64.
pub(crate) struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    pub fn stream(seed: u64, pass: u64, mu: usize, site: usize) -> Self {
        let mut x = seed;
        let mut key = splitmix(&mut x);
        key ^= pass.wrapping_mul(0xd1b5_4a32_d192_ed03);
        key ^= (mu as u64).wrapping_mul(0xabc9_8388_fb8f_ac03);
        key ^= (site as u64).wrapping_mul(0x8cb9_2ba7_2f3d_8dd7);
        Self {
            s: std::array::from_fn(|_| splitmix(&mut key)),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
