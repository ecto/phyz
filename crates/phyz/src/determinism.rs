//! Tools for checking that a rollout reproduces, and for telling chaos from
//! bugs when it does not.
//!
//! Contact-rich rigid-body simulation is chaotic. A box tipping on a corner
//! and a foot at the edge of stiction both sit on a knife edge where a
//! last-ulp difference in one number decides which way the next microsecond
//! goes, and from there the trajectories separate exponentially. That is
//! physics, and no amount of engineering removes it.
//!
//! What engineering *can* remove is the second source of difference: the
//! engine itself computing different numbers from the same inputs, because a
//! container iterated in memory order, a solver swept its rows in whatever
//! order the broad phase produced them, or a platform's `sin` disagreed with
//! another's in the last bit. When both sources are present you cannot tell
//! them apart, which means you cannot bisect a regression, cannot reproduce a
//! published number, and cannot trust a parameter search. That is the failure
//! this module exists to make impossible to miss.
//!
//! It gives you three things:
//!
//! 1. [`RolloutHasher`] — a stable fingerprint of a trajectory. Stable across
//!    machines, Rust versions and optimization levels, because it is defined
//!    on the IEEE bit patterns and uses a fixed hash (FNV-1a), not
//!    [`std::hash::DefaultHasher`], whose output is explicitly not stable.
//! 2. [`ulp_offset`] and [`ulps_between`] — perturb an input by exactly *n*
//!    representable steps, and measure how far apart two numbers are in the
//!    same units.
//! 3. [`divergence`] — run two rollouts that differ by a known perturbation
//!    and report how their separation grows, so you can read off the Lyapunov
//!    time of *your* scene and know how long a number from it means anything.
//!
//! # The rule of thumb this is for
//!
//! A difference that appears at step 0 is a bug. A difference that starts at
//! `~1e-16` and grows smoothly by a fixed factor per unit time is chaos. A
//! difference that is zero for 400 steps and then jumps to `1e-3` is a
//! discrete event — a contact set change, an active-set flip — being reached
//! at a different step; that is *usually* chaos too, but it is also exactly
//! what an order-dependence bug looks like, which is why the ordering in the
//! broad phase and the contact solve is now fixed by identity rather than by
//! position. See `docs/determinism.md`.

use phyz_model::State;

/// FNV-1a over the IEEE-754 bit patterns of a trajectory.
///
/// Deliberately not [`std::hash::DefaultHasher`]: that is SipHash with a
/// per-process random key on some configurations, and its algorithm is
/// documented as unstable across Rust releases. A golden hash checked into the
/// repository has to outlive both.
///
/// The hash is over **bits**, so `-0.0` and `0.0` hash differently and a `NaN`
/// hashes to whatever its payload says. Both are correct for the purpose: this
/// is a bitwise-equality check, not a numerical one. If you want a tolerance
/// comparison, use [`state_distance`].
///
/// ```
/// use phyz::determinism::RolloutHasher;
/// let mut h = RolloutHasher::new();
/// h.absorb_f64(1.0);
/// h.absorb_f64(2.0);
/// let a = h.finish();
///
/// let mut h = RolloutHasher::new();
/// h.absorb_f64(1.0);
/// h.absorb_f64(2.0);
/// assert_eq!(a, h.finish());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RolloutHasher {
    h: u64,
}

impl RolloutHasher {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// A fresh hasher.
    pub fn new() -> Self {
        Self {
            h: Self::OFFSET_BASIS,
        }
    }

    /// Absorb one `f64` by its bit pattern.
    #[inline]
    pub fn absorb_f64(&mut self, x: f64) {
        self.absorb_u64(x.to_bits());
    }

    /// Absorb a raw 64-bit word — useful for lengths, indices and flags, so
    /// that a rollout of 100 steps cannot collide with one of 200.
    #[inline]
    pub fn absorb_u64(&mut self, w: u64) {
        for b in w.to_le_bytes() {
            self.h ^= b as u64;
            self.h = self.h.wrapping_mul(Self::PRIME);
        }
    }

    /// Absorb a slice of `f64`, length first.
    pub fn absorb_slice(&mut self, xs: &[f64]) {
        self.absorb_u64(xs.len() as u64);
        for &x in xs {
            self.absorb_f64(x);
        }
    }

    /// Absorb the *dynamical* content of a state: `q`, `v` and `time`.
    ///
    /// `body_xform` is deliberately left out. It is a pure function of `q`
    /// recomputed by forward kinematics at the end of every step, so hashing it
    /// would add no information while making the fingerprint depend on whether
    /// the caller happened to leave it stale.
    pub fn absorb_state(&mut self, state: &State) {
        self.absorb_slice(state.q.as_slice());
        self.absorb_slice(state.v.as_slice());
        self.absorb_f64(state.time);
    }

    /// The fingerprint so far.
    pub fn finish(&self) -> u64 {
        self.h
    }

    /// The fingerprint so far, formatted the way the golden tests and
    /// `phyz-bench` print it.
    pub fn hex(&self) -> String {
        format!("{:016x}", self.h)
    }
}

impl Default for RolloutHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Fingerprint a whole rollout: absorb the state after every step.
///
/// `step` advances `state` by one timestep. The initial state is absorbed
/// first, so a zero-step rollout still fingerprints its input.
///
/// ```no_run
/// use phyz::{Simulator, determinism::hash_rollout};
/// # let (model, mut state) = unimplemented!();
/// let sim = Simulator::new();
/// let h = hash_rollout(&mut state, 1_000, |s| sim.step(&model, s));
/// println!("{h:016x}");
/// ```
pub fn hash_rollout<F>(state: &mut State, steps: usize, mut step: F) -> u64
where
    F: FnMut(&mut State),
{
    let mut h = RolloutHasher::new();
    h.absorb_state(state);
    for _ in 0..steps {
        step(state);
        h.absorb_state(state);
    }
    h.finish()
}

/// `x` moved `n` representable steps toward `+inf` (or `-inf` for negative
/// `n`).
///
/// This is the honest way to say "perturb by one ulp": adding `f64::EPSILON`
/// is a different size at every magnitude and is a no-op above `2/EPSILON`.
///
/// Zero steps toward the sign of `n` from `±0.0` lands on the smallest
/// subnormal of that sign, which is the correct neighbour. `NaN` and
/// infinities are returned unchanged — there is no next value to move to.
///
/// ```
/// use phyz::determinism::ulp_offset;
/// assert_eq!(ulp_offset(1.0, 1), 1.0 + f64::EPSILON);
/// assert_eq!(ulp_offset(ulp_offset(1.0, 1), -1), 1.0);
/// assert_eq!(ulp_offset(2.5, 0), 2.5);
/// ```
pub fn ulp_offset(x: f64, n: i64) -> f64 {
    if n == 0 || !x.is_finite() {
        return x;
    }
    // Map to a sign-magnitude-ordered integer line, step on it, map back. This
    // is the standard total order on floats: negatives descend from the top of
    // the unsigned range, so flipping them makes the whole line monotone.
    let key = f64_to_ordered(x);
    ordered_to_f64(key + n as i128)
}

/// How many representable steps separate `a` and `b`.
///
/// Saturates at [`i64::MAX`] rather than wrapping, and returns [`i64::MAX`] if
/// either value is not finite. Sign follows `b - a`.
///
/// ```
/// use phyz::determinism::{ulp_offset, ulps_between};
/// assert_eq!(ulps_between(1.0, ulp_offset(1.0, 7)), 7);
/// assert_eq!(ulps_between(3.0, 3.0), 0);
/// ```
pub fn ulps_between(a: f64, b: f64) -> i64 {
    if !a.is_finite() || !b.is_finite() {
        return i64::MAX;
    }
    let d = f64_to_ordered(b) - f64_to_ordered(a);
    d.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Map an `f64` onto a monotone integer line (the standard total order).
fn f64_to_ordered(x: f64) -> i128 {
    let bits = x.to_bits() as i64;
    // Negative floats have the sign bit set and count *upward* in magnitude as
    // the payload grows, so they need reversing to become monotone.
    let ordered = if bits < 0 { i64::MIN - bits } else { bits };
    ordered as i128
}

/// Inverse of [`f64_to_ordered`], clamped to the finite range.
fn ordered_to_f64(k: i128) -> f64 {
    let lo = f64_to_ordered(f64::MIN);
    let hi = f64_to_ordered(f64::MAX);
    let k = k.clamp(lo, hi) as i64;
    let bits = if k < 0 { i64::MIN - k } else { k };
    f64::from_bits(bits as u64)
}

/// Euclidean distance between two states over the stacked `(q, v)` vector.
///
/// Mixes units — radians with metres, and rates with positions — which is
/// exactly right here: the question is not "how far apart physically" but
/// "how many digits do these two rollouts still share". Returns
/// [`f64::INFINITY`] if either state has gone non-finite, and panics if the
/// two states have different dimensions, which is a model mismatch rather than
/// a divergence.
pub fn state_distance(a: &State, b: &State) -> f64 {
    assert_eq!(a.q.len(), b.q.len(), "states have different nq");
    assert_eq!(a.v.len(), b.v.len(), "states have different nv");
    let mut sum = 0.0;
    for (x, y) in
        a.q.as_slice()
            .iter()
            .zip(b.q.as_slice())
            .chain(a.v.as_slice().iter().zip(b.v.as_slice()))
    {
        let d = x - y;
        if !d.is_finite() {
            return f64::INFINITY;
        }
        sum += d * d;
    }
    sum.sqrt()
}

/// One sample from a [`divergence`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivergenceSample {
    /// Step index the sample was taken at.
    pub step: usize,
    /// Simulated time at that step.
    pub time: f64,
    /// `state_distance` between the perturbed and reference rollouts.
    pub distance: f64,
}

/// How two rollouts that started a known distance apart separate over time.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    /// The perturbation applied, in representable steps of the touched value.
    pub ulps: i64,
    /// The separation at step 0 — normally the size of one ulp of the
    /// perturbed coordinate, and the number every later sample is relative to.
    pub initial: f64,
    /// Separation over time, sampled every `stride` steps.
    pub samples: Vec<DivergenceSample>,
    /// Fitted growth rate `lambda` in `d(t) ~ d(0) e^{lambda t}`, per second of
    /// simulated time, over the exponential part of the curve.
    ///
    /// `None` when there is no exponential part to fit: the rollout never
    /// separated (an integrable scene, or one where the perturbation was
    /// rounded away), or it went non-finite before enough samples accumulated.
    pub lyapunov: Option<f64>,
    /// Simulated time at which the separation first exceeded `threshold`.
    /// `None` if it never did.
    pub time_to_threshold: Option<f64>,
    /// The threshold `time_to_threshold` refers to.
    pub threshold: f64,
}

impl Divergence {
    /// `ln(2) / lambda`: the simulated time in which the separation doubles.
    ///
    /// The practical reading of the Lyapunov rate — "this scene loses one bit
    /// of agreement every `doubling_time` seconds", so a `f64` rollout
    /// starting from identical inputs holds about `52 * doubling_time` seconds
    /// of meaning before the last digit of the answer is noise.
    pub fn doubling_time(&self) -> Option<f64> {
        self.lyapunov
            .filter(|l| *l > 0.0)
            .map(|l| std::f64::consts::LN_2 / l)
    }

    /// The final separation reached.
    pub fn final_distance(&self) -> f64 {
        self.samples.last().map_or(0.0, |s| s.distance)
    }
}

/// Run the same rollout twice — once from `state`, once from a copy perturbed
/// by `ulps` representable steps in coordinate `q_index` — and report how the
/// two separate.
///
/// This is the measurement that tells you what a number from your scene is
/// worth. It costs two rollouts.
///
/// `step` must be a *pure* stepping function: given the same state it must
/// produce the same next state, with no dependence on which of the two
/// rollouts it is being called for. If you are stepping through
/// [`crate::Simulator`], either give each rollout its own simulator or build
/// one with [`crate::Simulator::with_warm_start`]`(false)` — a shared
/// simulator's contact cache would otherwise leak one rollout into the other
/// and the number you get back would be measuring that instead.
///
/// # Reading the result
///
/// - `lyapunov <= 0` or `None`, `final_distance` still at `~1e-16`: the scene
///   is not chaotic over this horizon. Bitwise reproducibility is achievable
///   and any difference you see is a bug.
/// - `lyapunov > 0`: the scene is chaotic. `doubling_time` tells you how fast
///   agreement is lost; past `52 * doubling_time` seconds the trajectories are
///   independent and comparing them point by point is meaningless.
/// - `initial == 0.0`: the perturbation did nothing — `q_index` is not a live
///   coordinate, or the value was overwritten before it was read.
pub fn divergence<F>(
    model_nq: usize,
    state: &State,
    q_index: usize,
    ulps: i64,
    steps: usize,
    stride: usize,
    mut step: F,
) -> Divergence
where
    F: FnMut(usize, &mut State),
{
    assert!(q_index < model_nq, "q_index {q_index} out of range");
    assert!(stride > 0, "stride must be positive");

    let mut a = state.clone();
    let mut b = state.clone();
    b.q[q_index] = ulp_offset(b.q[q_index], ulps);

    let initial = state_distance(&a, &b);
    let threshold = 1e-3;
    let mut samples = Vec::with_capacity(steps / stride + 1);
    let mut time_to_threshold = None;

    for k in 0..steps {
        // Index the two rollouts distinctly so a caller holding two simulators
        // can dispatch on it; a genuinely pure stepper ignores it.
        step(0, &mut a);
        step(1, &mut b);
        if (k + 1) % stride == 0 || k + 1 == steps {
            let d = state_distance(&a, &b);
            if time_to_threshold.is_none() && d > threshold {
                time_to_threshold = Some(a.time);
            }
            samples.push(DivergenceSample {
                step: k + 1,
                time: a.time,
                distance: d,
            });
            if !d.is_finite() {
                break;
            }
        }
    }

    Divergence {
        ulps,
        initial,
        lyapunov: fit_lyapunov(initial, &samples),
        samples,
        time_to_threshold,
        threshold,
    }
}

/// Least-squares slope of `ln(distance)` against `time`, over the window where
/// the curve is actually exponential.
///
/// Two windows are excluded on purpose. At the start the separation is pinned
/// near the perturbation size and the log is dominated by rounding; at the end
/// it saturates at the scale of the scene, where the growth is bounded by
/// geometry rather than by the dynamics. Fitting through either flattens the
/// estimate toward zero and would report a chaotic scene as a tame one.
fn fit_lyapunov(initial: f64, samples: &[DivergenceSample]) -> Option<f64> {
    if initial <= 0.0 {
        return None;
    }
    // The exponential band: grown by at least a factor of 8 out of the noise,
    // and not yet past a tenth of a metre/radian, where saturation sets in.
    let band: Vec<&DivergenceSample> = samples
        .iter()
        .filter(|s| {
            s.distance.is_finite()
                && s.distance > 8.0 * initial
                && s.distance < 1e-1
                && s.time > 0.0
        })
        .collect();
    if band.len() < 4 {
        return None;
    }
    let n = band.len() as f64;
    let (mut st, mut sy, mut stt, mut sty) = (0.0, 0.0, 0.0, 0.0);
    for s in &band {
        let t = s.time;
        let y = (s.distance / initial).ln();
        st += t;
        sy += y;
        stt += t * t;
        sty += t * y;
    }
    let denom = n * stt - st * st;
    if denom.abs() < f64::MIN_POSITIVE {
        return None;
    }
    Some((n * sty - st * sy) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_offset_walks_the_representable_line() {
        assert_eq!(ulp_offset(1.0, 1), 1.0 + f64::EPSILON);
        assert_eq!(ulp_offset(1.0, -1), 1.0 - f64::EPSILON / 2.0);
        // Round trip at several magnitudes, including across zero, where the
        // sign-magnitude representation is not monotone and a naive
        // `from_bits(to_bits + 1)` gets it wrong.
        for x in [-1e300, -1.0, -1e-300, 0.0, 1e-300, 1.0, 1e300] {
            for n in [-5, -1, 1, 5] {
                assert_eq!(ulp_offset(ulp_offset(x, n), -n), x, "x={x} n={n}");
                assert_eq!(ulps_between(x, ulp_offset(x, n)), n, "x={x} n={n}");
            }
        }
        // Stepping off zero lands on the smallest subnormal, not on zero.
        assert_eq!(ulp_offset(0.0, 1), f64::from_bits(1));
        assert!(ulp_offset(0.0, -1) < 0.0);
        // Non-finite inputs have no neighbour.
        assert!(ulp_offset(f64::NAN, 3).is_nan());
        assert_eq!(ulp_offset(f64::INFINITY, -1), f64::INFINITY);
    }

    #[test]
    fn hasher_is_order_and_content_sensitive_and_reproducible() {
        let of = |xs: &[f64]| {
            let mut h = RolloutHasher::new();
            h.absorb_slice(xs);
            h.finish()
        };
        assert_eq!(of(&[1.0, 2.0, 3.0]), of(&[1.0, 2.0, 3.0]));
        assert_ne!(of(&[1.0, 2.0, 3.0]), of(&[1.0, 3.0, 2.0]));
        // One ulp must move the hash — that is the whole point of hashing bits
        // rather than rounded values.
        assert_ne!(of(&[1.0]), of(&[ulp_offset(1.0, 1)]));
        // Length is part of the fingerprint, so a prefix cannot collide.
        assert_ne!(of(&[1.0, 2.0]), of(&[1.0, 2.0, 0.0]));
        // Signed zero is a bit-level difference and is reported as one.
        assert_ne!(of(&[0.0]), of(&[-0.0]));
    }

    /// The Lyapunov fit must report growth where there is growth and stay
    /// quiet where there is none. Exercised on synthetic curves so the test
    /// has no dependence on any scene's physics.
    #[test]
    fn lyapunov_fit_recovers_a_known_rate() {
        let d0 = 1e-16;
        let lambda = 3.0;
        let samples: Vec<DivergenceSample> = (1..=400)
            .map(|k| {
                let time = k as f64 * 0.01;
                DivergenceSample {
                    step: k,
                    time,
                    distance: d0 * (lambda * time).exp(),
                }
            })
            .collect();
        let got = fit_lyapunov(d0, &samples).expect("should fit");
        assert!((got - lambda).abs() < 1e-9, "got {got}");

        // A flat curve is not chaos, and must not be reported as a small
        // positive exponent.
        let flat: Vec<DivergenceSample> = (1..=400)
            .map(|k| DivergenceSample {
                step: k,
                time: k as f64 * 0.01,
                distance: d0,
            })
            .collect();
        assert_eq!(fit_lyapunov(d0, &flat), None);
    }
}
