//! Tape-based reverse-mode scalar: one backward pass, every input's gradient.
//!
//! [`Rev`] implements [`tang::Scalar`], so any computation written generically
//! over `T: Scalar` — the model-layout mirrors in `crate::model_generic`, the
//! rollout step in `crate::rollout::step` — can be evaluated once on `Rev` and
//! then differentiated with respect to *all* of its seeded inputs in a single
//! [`backward`] sweep. That is the reverse-mode complement to
//! [`crate::multidual::MultiDual`]: multidual amortises the primal across `N`
//! forward tangents, `Rev` amortises it across every input at once, which is
//! the right shape when the output is one scalar (an adjoint contraction) and
//! the inputs are hundreds of lanes.
//!
//! # Semantics
//!
//! Branches (`min`/`max`/`abs`/`select`/comparisons/pivoting) follow the
//! **primal**, exactly as [`tang::Dual`]'s do: the recorded tape is the
//! derivative of the branch the `f64` computation takes, which is the
//! one-sided/frozen-branch semantics every adjoint in this workspace already
//! promises. A `Rev` computation's primal is bit-identical to the plain `f64`
//! computation because every operation computes its value with the same
//! expression and only *additionally* records weights.
//!
//! The transcendentals the engine routes through [`phyz_math::fp`]
//! (`sin`, `cos`, `sin_cos`, `atan2`, `tanh`) are routed through `fp` here
//! too, so a `Rev` evaluation of a mirror of engine code reproduces the
//! engine's bits, not merely its values. The remainder use `std`, matching
//! [`tang::Scalar`]'s `f64` impl.
//!
//! # Tape discipline
//!
//! The tape is **thread-local**. A differentiation pass is:
//!
//! ```ignore
//! let _guard = tape_scope();            // clears the tape
//! let x = Rev::var(3.0);
//! let y = x * x;
//! let g = backward(y);
//! assert_eq!(g.of(x), 6.0);
//! ```
//!
//! `Rev` values must not escape the scope that created them: an index into a
//! cleared tape is meaningless. Nothing enforces this at compile time (the
//! type is `Copy` by `Scalar`'s demand); [`backward`] and [`Gradients::of`]
//! bounds-check instead of trusting.
//!
//! `Rev` is `Send + Sync` because `Scalar` requires it, but a value is only
//! meaningful on the thread whose tape recorded it — moving one across
//! threads gives wrong (bounds-checked, not unsafe) answers, the same
//! contract a raw index into any thread-local arena has.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::cell::RefCell;
use tang::Scalar;

/// Sentinel parent index: "constant, no tape node".
const NONE: u32 = u32::MAX;

/// One recorded operation: up to two parents with local partial derivatives.
#[derive(Clone, Copy)]
struct Node {
    w: [f64; 2],
    p: [u32; 2],
}

thread_local! {
    static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}

/// A value participating in reverse-mode differentiation.
///
/// Constants (from `from_f64`, `ZERO`, …) carry no tape node at all, so
/// lifting a whole model's worth of parameters costs nothing until an actual
/// variable touches them.
#[derive(Clone, Copy)]
pub struct Rev {
    /// The primal value.
    pub val: f64,
    idx: u32,
}

impl Rev {
    /// A constant: participates in arithmetic, contributes no gradient.
    #[inline]
    pub fn constant(val: f64) -> Self {
        Rev { val, idx: NONE }
    }

    /// A differentiation variable: seeds a leaf on the tape.
    pub fn var(val: f64) -> Self {
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            t.push(Node {
                w: [0.0; 2],
                p: [NONE; 2],
            });
            (t.len() - 1) as u32
        });
        Rev { val, idx }
    }

    #[inline]
    fn push1(val: f64, p: u32, w: f64) -> Rev {
        if p == NONE {
            return Rev::constant(val);
        }
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            t.push(Node {
                w: [w, 0.0],
                p: [p, NONE],
            });
            (t.len() - 1) as u32
        });
        Rev { val, idx }
    }

    #[inline]
    fn push2(val: f64, p0: u32, w0: f64, p1: u32, w1: f64) -> Rev {
        if p0 == NONE && p1 == NONE {
            return Rev::constant(val);
        }
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            t.push(Node {
                w: [w0, w1],
                p: [p0, p1],
            });
            (t.len() - 1) as u32
        });
        Rev { val, idx }
    }
}

/// Clears the thread's tape and returns a guard; the tape is cleared again
/// when the guard drops, so a scope's `Rev` values cannot leak usable indices
/// into the next scope.
pub fn tape_scope() -> TapeScope {
    TAPE.with(|t| t.borrow_mut().clear());
    TapeScope { _priv: () }
}

/// Guard returned by [`tape_scope`].
pub struct TapeScope {
    _priv: (),
}

impl Drop for TapeScope {
    fn drop(&mut self) {
        TAPE.with(|t| t.borrow_mut().clear());
    }
}

/// The number of nodes currently on this thread's tape (diagnostics).
pub fn tape_len() -> usize {
    TAPE.with(|t| t.borrow().len())
}

/// Adjoints of every tape node with respect to one output.
pub struct Gradients {
    adj: Vec<f64>,
}

impl Gradients {
    /// `d(output)/d(x)`; zero for constants and for values recorded on a
    /// different (or since-cleared) tape.
    #[inline]
    pub fn of(&self, x: Rev) -> f64 {
        if x.idx == NONE {
            return 0.0;
        }
        self.adj.get(x.idx as usize).copied().unwrap_or(0.0)
    }
}

/// One reverse sweep: the gradient of `output` with respect to every variable
/// seeded since the enclosing [`tape_scope`].
pub fn backward(output: Rev) -> Gradients {
    TAPE.with(|t| {
        let t = t.borrow();
        let mut adj = vec![0.0f64; t.len()];
        if output.idx != NONE {
            if let Some(a) = adj.get_mut(output.idx as usize) {
                *a = 1.0;
            }
            for i in (0..=(output.idx as usize).min(t.len().saturating_sub(1))).rev() {
                let a = adj[i];
                if a == 0.0 {
                    continue;
                }
                let n = t[i];
                for k in 0..2 {
                    if n.p[k] != NONE {
                        adj[n.p[k] as usize] += n.w[k] * a;
                    }
                }
            }
        }
        Gradients { adj }
    })
}

// ---------------------------------------------------------------------------
// Operator impls
// ---------------------------------------------------------------------------

impl Add for Rev {
    type Output = Rev;
    #[inline]
    fn add(self, rhs: Rev) -> Rev {
        Rev::push2(self.val + rhs.val, self.idx, 1.0, rhs.idx, 1.0)
    }
}
impl Sub for Rev {
    type Output = Rev;
    #[inline]
    fn sub(self, rhs: Rev) -> Rev {
        Rev::push2(self.val - rhs.val, self.idx, 1.0, rhs.idx, -1.0)
    }
}
impl Mul for Rev {
    type Output = Rev;
    #[inline]
    fn mul(self, rhs: Rev) -> Rev {
        Rev::push2(self.val * rhs.val, self.idx, rhs.val, rhs.idx, self.val)
    }
}
impl Div for Rev {
    type Output = Rev;
    #[inline]
    fn div(self, rhs: Rev) -> Rev {
        let inv = 1.0 / rhs.val;
        Rev::push2(
            self.val * inv,
            self.idx,
            inv,
            rhs.idx,
            -self.val * inv * inv,
        )
    }
}
impl Neg for Rev {
    type Output = Rev;
    #[inline]
    fn neg(self) -> Rev {
        Rev::push1(-self.val, self.idx, -1.0)
    }
}
impl AddAssign for Rev {
    #[inline]
    fn add_assign(&mut self, rhs: Rev) {
        *self = *self + rhs;
    }
}
impl SubAssign for Rev {
    #[inline]
    fn sub_assign(&mut self, rhs: Rev) {
        *self = *self - rhs;
    }
}
impl MulAssign for Rev {
    #[inline]
    fn mul_assign(&mut self, rhs: Rev) {
        *self = *self * rhs;
    }
}
impl DivAssign for Rev {
    #[inline]
    fn div_assign(&mut self, rhs: Rev) {
        *self = *self / rhs;
    }
}

impl PartialEq for Rev {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}
impl PartialOrd for Rev {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}
impl Default for Rev {
    fn default() -> Self {
        Rev::constant(0.0)
    }
}
impl fmt::Debug for Rev {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rev({})", self.val)
    }
}
impl fmt::Display for Rev {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl Scalar for Rev {
    const ZERO: Self = Rev {
        val: 0.0,
        idx: NONE,
    };
    const ONE: Self = Rev {
        val: 1.0,
        idx: NONE,
    };
    const TWO: Self = Rev {
        val: 2.0,
        idx: NONE,
    };
    const HALF: Self = Rev {
        val: 0.5,
        idx: NONE,
    };
    const PI: Self = Rev {
        val: core::f64::consts::PI,
        idx: NONE,
    };
    const TAU: Self = Rev {
        val: core::f64::consts::TAU,
        idx: NONE,
    };
    const FRAC_PI_2: Self = Rev {
        val: core::f64::consts::FRAC_PI_2,
        idx: NONE,
    };
    const EPSILON: Self = Rev {
        val: f64::EPSILON,
        idx: NONE,
    };
    const INFINITY: Self = Rev {
        val: f64::INFINITY,
        idx: NONE,
    };
    const NEG_INFINITY: Self = Rev {
        val: f64::NEG_INFINITY,
        idx: NONE,
    };

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Rev::push1(s, self.idx, 0.5 / s)
    }
    #[inline]
    fn abs(self) -> Self {
        // Matches `Dual::abs`: derivative `signum(x)` (zero at exactly 0).
        Rev::push1(self.val.abs(), self.idx, self.val.signum())
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = phyz_math::fp::sin_cos(self.val);
        Rev::push1(s, self.idx, c)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = phyz_math::fp::sin_cos(self.val);
        Rev::push1(c, self.idx, -s)
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        Rev::push1(t, self.idx, 1.0 + t * t)
    }
    #[inline]
    fn asin(self) -> Self {
        Rev::push1(
            self.val.asin(),
            self.idx,
            1.0 / (1.0 - self.val * self.val).sqrt(),
        )
    }
    #[inline]
    fn acos(self) -> Self {
        Rev::push1(
            self.val.acos(),
            self.idx,
            -1.0 / (1.0 - self.val * self.val).sqrt(),
        )
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let (y, x) = (self.val, other.val);
        let d = x * x + y * y;
        Rev::push2(
            phyz_math::fp::atan2(y, x),
            self.idx,
            x / d,
            other.idx,
            -y / d,
        )
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = phyz_math::fp::sin_cos(self.val);
        (Rev::push1(s, self.idx, c), Rev::push1(c, self.idx, -s))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        // Same branch rule as `Dual::min`: `<` picks self, ties pick other.
        if self.val < other.val { self } else { other }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.val > other.val { self } else { other }
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline]
    fn recip(self) -> Self {
        let inv = self.val.recip();
        Rev::push1(inv, self.idx, -inv * inv)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        Rev::push1(
            self.val.powi(n),
            self.idx,
            f64::from(n) * self.val.powi(n - 1),
        )
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let flipped = self.val.signum() != sign.val.signum();
        Rev::push1(
            self.val.copysign(sign.val),
            self.idx,
            if flipped { -1.0 } else { 1.0 },
        )
    }
    #[inline]
    fn signum(self) -> Self {
        Rev::constant(self.val.signum())
    }
    #[inline]
    fn floor(self) -> Self {
        Rev::constant(self.val.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Rev::constant(self.val.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Rev::constant(self.val.round())
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        Rev::push1(e, self.idx, e)
    }
    #[inline]
    fn ln(self) -> Self {
        Rev::push1(self.val.ln(), self.idx, 1.0 / self.val)
    }
    #[inline]
    fn powf(self, p: Self) -> Self {
        let val = self.val.powf(p.val);
        if p.idx == NONE {
            // Constant exponent: `d = p·x^(p−1)`, which stays finite at
            // `x = 0, p ≥ 1` where the general rule's `ln(x)` term would
            // manufacture a NaN out of `0·(−∞)`. The impedance sigmoid
            // (`x.powf(power)` with `x ∈ [0, 1]`) sits exactly on that corner.
            return Rev::push1(val, self.idx, p.val * self.val.powf(p.val - 1.0));
        }
        Rev::push2(
            val,
            self.idx,
            p.val * self.val.powf(p.val - 1.0),
            p.idx,
            val * self.val.ln(),
        )
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = phyz_math::fp::tanh(self.val);
        Rev::push1(t, self.idx, 1.0 - t * t)
    }
    #[inline]
    fn sinh(self) -> Self {
        Rev::push1(self.val.sinh(), self.idx, self.val.cosh())
    }
    #[inline]
    fn cosh(self) -> Self {
        Rev::push1(self.val.cosh(), self.idx, self.val.sinh())
    }
    #[inline]
    fn acosh(self) -> Self {
        Rev::push1(
            self.val.acosh(),
            self.idx,
            1.0 / (self.val * self.val - 1.0).sqrt(),
        )
    }
    #[inline]
    fn asinh(self) -> Self {
        Rev::push1(
            self.val.asinh(),
            self.idx,
            1.0 / (self.val * self.val + 1.0).sqrt(),
        )
    }
    #[inline]
    fn atanh(self) -> Self {
        Rev::push1(
            self.val.atanh(),
            self.idx,
            1.0 / (1.0 - self.val * self.val),
        )
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        Rev::constant(v)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.val
    }
    #[inline]
    fn from_i32(v: i32) -> Self {
        Rev::constant(f64::from(v))
    }
    #[inline]
    fn select(cond: Self, a: Self, b: Self) -> Self {
        // Straight-through, like `Dual::select`: the condition contributes no
        // gradient, the chosen branch keeps its own.
        if cond.val > 0.0 { a } else { b }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reverse-mode gradients must match forward-mode duals on the full
    /// arithmetic surface — the same pairing `multidual` pins against `Dual`.
    #[test]
    fn matches_dual_on_composite_expression() {
        use tang::Dual;
        let f = |x: [f64; 3], seed: usize| -> (f64, f64) {
            let v: Vec<Dual<f64>> = (0..3)
                .map(|i| {
                    if i == seed {
                        Dual::var(x[i])
                    } else {
                        Dual::constant(x[i])
                    }
                })
                .collect();
            let r = (v[0] * v[1]).sin() + (v[2].abs() + Dual::constant(1.0)).ln() * v[0].sqrt()
                - v[1].atan2(v[2])
                + (v[0] - v[2]).max(v[1] * Dual::constant(0.3)).tanh()
                + v[2].powf(Dual::constant(2.0));
            (r.real, r.dual)
        };

        let x = [1.7, -0.4, 0.9];
        let _g = tape_scope();
        let v: Vec<Rev> = x.iter().map(|&xi| Rev::var(xi)).collect();
        let r = (v[0] * v[1]).sin() + (v[2].abs() + Rev::constant(1.0)).ln() * v[0].sqrt()
            - v[1].atan2(v[2])
            + (v[0] - v[2]).max(v[1] * Rev::constant(0.3)).tanh()
            + v[2].powf(Rev::constant(2.0));
        let g = backward(r);

        for (seed, vs) in v.iter().enumerate() {
            let (val, dual) = f(x, seed);
            // fp routing can differ from libm by an ulp; allow a few.
            assert!(
                (r.val - val).abs() <= 1e-14 * val.abs().max(1.0),
                "primal {} vs {}",
                r.val,
                val
            );
            assert!(
                (g.of(*vs) - dual).abs() <= 1e-12 * dual.abs().max(1.0),
                "grad[{seed}] {} vs dual {}",
                g.of(*vs),
                dual
            );
        }
    }

    /// `x.powf(2.0)` at `x = 0` must not manufacture a NaN — the impedance
    /// sigmoid evaluates exactly there whenever a contact sits at zero
    /// violation.
    #[test]
    fn powf_constant_exponent_is_finite_at_zero_base() {
        let _g = tape_scope();
        let x = Rev::var(0.0);
        let y = x.powf(Rev::constant(2.0));
        let g = backward(y);
        assert_eq!(y.val, 0.0);
        assert_eq!(g.of(x), 0.0);
    }

    /// Fan-out accumulates: `y = x·x + x` has gradient `2x + 1`.
    #[test]
    fn fan_out_accumulates() {
        let _g = tape_scope();
        let x = Rev::var(3.0);
        let y = x * x + x;
        let g = backward(y);
        assert_eq!(g.of(x), 7.0);
    }

    /// Constants stay off the tape entirely.
    #[test]
    fn constants_cost_no_nodes() {
        let _g = tape_scope();
        let a = Rev::constant(2.0);
        let b = Rev::constant(3.0);
        let _c = a * b + a.sin();
        assert_eq!(tape_len(), 0);
    }
}
