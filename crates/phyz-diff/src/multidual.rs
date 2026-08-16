//! Vector-mode forward AD: one pass through the dynamics, `N` derivatives out.
//!
//! [`tang::Dual`] carries one derivative alongside the value, so extracting an
//! `n`-column Jacobian means running the function `n` times. Every one of those
//! runs recomputes the same primal: the same kinematic tree walk, the same
//! `sin`/`cos` per joint, the same memory traffic — all to produce one column.
//!
//! [`MultiDual<N>`] carries `N` derivatives instead. One pass computes the
//! primal once and propagates `N` tangent directions through it, so the
//! primal's cost is amortised across the whole chunk rather than paid `N`
//! times. That is the difference between the adjoint's `nq + nv` dual-ABA
//! passes per timestep and `⌈(nq + nv) / N⌉` of them.
//!
//! # Lane-for-lane identical to `Dual`
//!
//! Every derivative rule here is the same expression as [`tang::Dual`]'s, in
//! the same association order, applied per lane. Lane `k` of a `MultiDual`
//! computation is therefore **bit-identical** to a scalar `Dual` computation
//! seeded in direction `k` — not merely close. That is what makes this a safe
//! substitution inside a rollout that promises bitwise reproducibility, and it
//! is asserted directly by `tests::lanes_match_scalar_dual_bitwise` over the
//! full arithmetic and transcendental surface. (Not an intra-doc link: a
//! `#[cfg(test)]` item does not exist in a docs build, so linking to one is an
//! error under `-D warnings`.)
//!
//! Widening is therefore never a numerical change, only a speed one. If a
//! gradient moves when `N` changes, that is a bug in this file.
//!
//! # Choosing `N`
//!
//! `N` is a const parameter, so each width is a separate monomorphisation with
//! its tangent array in registers. Wider is not automatically better: lanes
//! beyond the column count are wasted arithmetic, so a 1-DOF model wants
//! `N = 1` and pays for anything larger. [`for_lanes`] does that dispatch —
//! smallest supported width that covers the request.
//!
//! ```
//! use phyz_diff::multidual::MultiDual;
//! use tang::Scalar;
//!
//! // f(x, y) = x² · y, seeded in both directions at once.
//! let x = MultiDual::<2>::var(3.0, 0);
//! let y = MultiDual::<2>::var(5.0, 1);
//! let f = x * x * y;
//!
//! assert_eq!(f.real, 45.0);
//! assert_eq!(f.dual[0], 30.0); // ∂f/∂x = 2xy
//! assert_eq!(f.dual[1], 9.0);  // ∂f/∂y = x²
//! ```

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use tang::Scalar;

/// A real value carrying `N` simultaneous derivative lanes.
///
/// See the [module docs](self) for why the lanes are bit-identical to `N`
/// separate [`tang::Dual`] evaluations.
#[derive(Clone, Copy, Debug)]
pub struct MultiDual<const N: usize> {
    /// The value.
    pub real: f64,
    /// The derivative in each seeded direction.
    pub dual: [f64; N],
}

impl<const N: usize> MultiDual<N> {
    /// A constant: value with every derivative zero.
    #[inline]
    pub fn constant(real: f64) -> Self {
        Self {
            real,
            dual: [0.0; N],
        }
    }

    /// A variable seeded in lane `k`: `∂self/∂(lane k) = 1`, all others zero.
    ///
    /// Panics if `k >= N`, which is a programming error rather than a runtime
    /// condition — a lane index out of range means the chunking is wrong.
    #[inline]
    pub fn var(real: f64, k: usize) -> Self {
        assert!(k < N, "lane {k} out of range for MultiDual<{N}>");
        let mut dual = [0.0; N];
        dual[k] = 1.0;
        Self { real, dual }
    }

    /// Construct with explicit derivatives.
    #[inline]
    pub fn new(real: f64, dual: [f64; N]) -> Self {
        Self { real, dual }
    }

    /// Apply `f` to every lane — the shape every unary rule takes.
    #[inline]
    fn map(&self, f: impl Fn(f64) -> f64) -> [f64; N] {
        core::array::from_fn(|k| f(self.dual[k]))
    }

    /// Combine lanes pairwise with `other` — the shape every binary rule takes.
    #[inline]
    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> [f64; N] {
        core::array::from_fn(|k| f(self.dual[k], other.dual[k]))
    }
}

impl<const N: usize> Default for MultiDual<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> fmt::Display for MultiDual<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.real)
    }
}

// Comparison is on the value alone, matching `tang::Dual`: an ordering that
// consulted the tangents would make `min`/`max`/`select` disagree with the
// scalar path and silently change which branch a rollout takes.
impl<const N: usize> PartialEq for MultiDual<N> {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real
    }
}

impl<const N: usize> PartialOrd for MultiDual<N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl<const N: usize> Add for MultiDual<N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            real: self.real + o.real,
            dual: self.zip(&o, |a, b| a + b),
        }
    }
}

impl<const N: usize> Sub for MultiDual<N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            real: self.real - o.real,
            dual: self.zip(&o, |a, b| a - b),
        }
    }
}

impl<const N: usize> Mul for MultiDual<N> {
    type Output = Self;
    #[inline]
    // The `+` inside `Mul` is the product rule, not a transposed operator:
    // d(xy) = x·dy + y·dx. Clippy's heuristic cannot tell the two apart.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, o: Self) -> Self {
        // Associated exactly as `Dual`'s is:
        // `self.dual * o.real + self.real * o.dual`.
        Self {
            real: self.real * o.real,
            dual: self.zip(&o, |a, b| a * o.real + self.real * b),
        }
    }
}

impl<const N: usize> Div for MultiDual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.real;
        Self {
            real: self.real * inv,
            dual: self.zip(&o, |a, b| (a * o.real - self.real * b) * (inv * inv)),
        }
    }
}

impl<const N: usize> Neg for MultiDual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: self.map(|a| -a),
        }
    }
}

impl<const N: usize> AddAssign for MultiDual<N> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<const N: usize> SubAssign for MultiDual<N> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<const N: usize> MulAssign for MultiDual<N> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<const N: usize> DivAssign for MultiDual<N> {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl<const N: usize> Scalar for MultiDual<N> {
    const ZERO: Self = Self {
        real: 0.0,
        dual: [0.0; N],
    };
    const ONE: Self = Self {
        real: 1.0,
        dual: [0.0; N],
    };
    const TWO: Self = Self {
        real: 2.0,
        dual: [0.0; N],
    };
    const HALF: Self = Self {
        real: 0.5,
        dual: [0.0; N],
    };
    const PI: Self = Self {
        real: core::f64::consts::PI,
        dual: [0.0; N],
    };
    const TAU: Self = Self {
        real: core::f64::consts::TAU,
        dual: [0.0; N],
    };
    const FRAC_PI_2: Self = Self {
        real: core::f64::consts::FRAC_PI_2,
        dual: [0.0; N],
    };
    const EPSILON: Self = Self {
        real: f64::EPSILON,
        dual: [0.0; N],
    };
    const INFINITY: Self = Self {
        real: f64::INFINITY,
        dual: [0.0; N],
    };
    const NEG_INFINITY: Self = Self {
        real: f64::NEG_INFINITY,
        dual: [0.0; N],
    };

    #[inline]
    fn sqrt(self) -> Self {
        let r = self.real.sqrt();
        Self {
            real: r,
            dual: self.map(|a| a / (2.0 * r)),
        }
    }

    #[inline]
    fn abs(self) -> Self {
        let s = self.real.signum();
        Self {
            real: self.real.abs(),
            dual: self.map(|a| a * s),
        }
    }

    #[inline]
    fn sin(self) -> Self {
        let c = self.real.cos();
        Self {
            real: self.real.sin(),
            dual: self.map(|a| a * c),
        }
    }

    #[inline]
    fn cos(self) -> Self {
        let s = self.real.sin();
        Self {
            real: self.real.cos(),
            dual: self.map(|a| -a * s),
        }
    }

    #[inline]
    fn tan(self) -> Self {
        let c = self.real.cos();
        Self {
            real: self.real.tan(),
            dual: self.map(|a| a / (c * c)),
        }
    }

    #[inline]
    fn asin(self) -> Self {
        let d = (1.0 - self.real * self.real).sqrt();
        Self {
            real: self.real.asin(),
            dual: self.map(|a| a / d),
        }
    }

    #[inline]
    fn acos(self) -> Self {
        let d = (1.0 - self.real * self.real).sqrt();
        Self {
            real: self.real.acos(),
            dual: self.map(|a| -a / d),
        }
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.real * self.real + other.real * other.real;
        Self {
            real: self.real.atan2(other.real),
            dual: self.zip(&other, |a, b| (a * other.real - self.real * b) / denom),
        }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real.sin_cos();
        (
            Self {
                real: s,
                dual: self.map(|a| a * c),
            },
            Self {
                real: c,
                dual: self.map(|a| -a * s),
            },
        )
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.real < other.real { self } else { other }
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.real > other.real { self } else { other }
    }

    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[inline]
    fn recip(self) -> Self {
        let inv = self.real.recip();
        Self {
            real: inv,
            dual: self.map(|a| -a * inv * inv),
        }
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let r = self.real.powi(n);
        let d = n as f64 * self.real.powi(n - 1);
        Self {
            real: r,
            dual: self.map(|a| a * d),
        }
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let flipped = self.real.signum() != sign.real.signum();
        Self {
            real: self.real.copysign(sign.real),
            dual: self.map(|a| if flipped { -a } else { a }),
        }
    }

    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.real.signum())
    }

    #[inline]
    fn floor(self) -> Self {
        Self::constant(self.real.floor())
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::constant(self.real.ceil())
    }

    #[inline]
    fn round(self) -> Self {
        Self::constant(self.real.round())
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.real.exp();
        Self {
            real: e,
            dual: self.map(|a| a * e),
        }
    }

    #[inline]
    fn ln(self) -> Self {
        Self {
            real: self.real.ln(),
            dual: self.map(|a| a / self.real),
        }
    }

    #[inline]
    fn powf(self, p: Self) -> Self {
        let val = self.real.powf(p.real);
        let ln_x = self.real.ln();
        Self {
            real: val,
            dual: self.zip(&p, |a, b| val * (b * ln_x + p.real * a / self.real)),
        }
    }

    #[inline]
    fn tanh(self) -> Self {
        let t = self.real.tanh();
        Self {
            real: t,
            dual: self.map(|a| a * (1.0 - t * t)),
        }
    }

    #[inline]
    fn sinh(self) -> Self {
        let c = self.real.cosh();
        Self {
            real: self.real.sinh(),
            dual: self.map(|a| a * c),
        }
    }

    #[inline]
    fn cosh(self) -> Self {
        let s = self.real.sinh();
        Self {
            real: self.real.cosh(),
            dual: self.map(|a| a * s),
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        let d = (self.real * self.real - 1.0).sqrt();
        Self {
            real: self.real.acosh(),
            dual: self.map(|a| a / d),
        }
    }

    #[inline]
    fn asinh(self) -> Self {
        let d = (self.real * self.real + 1.0).sqrt();
        Self {
            real: self.real.asinh(),
            dual: self.map(|a| a / d),
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        let d = 1.0 - self.real * self.real;
        Self {
            real: self.real.atanh(),
            dual: self.map(|a| a / d),
        }
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self.real
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        Self::constant(v as f64)
    }

    #[inline]
    fn select(cond: Self, a: Self, b: Self) -> Self {
        if cond.real > 0.0 { a } else { b }
    }
}

/// The widths [`for_lanes`] can dispatch to, narrowest first.
///
/// Powers of two up to 16. Beyond that the tangent array stops living in
/// registers and the amortisation reverses; a model wanting more columns runs
/// several 16-wide chunks instead.
pub const SUPPORTED_WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];

/// The width a chunk of `lanes` columns should use: the narrowest supported
/// width that covers it, capped at 16.
///
/// Lanes beyond the column count are wasted arithmetic, so this never rounds
/// up further than it must — a 1-DOF model gets `N = 1` and pays exactly what
/// the scalar `Dual` path pays.
pub fn width_for(lanes: usize) -> usize {
    *SUPPORTED_WIDTHS
        .iter()
        .find(|&&w| w >= lanes)
        .unwrap_or(SUPPORTED_WIDTHS.last().expect("non-empty"))
}

/// Call `f` monomorphised at the width [`width_for`] picks for `lanes`.
///
/// The const parameter has to be known at compile time, so the runtime column
/// count is turned into one of a fixed set of instantiations here rather than
/// at every call site.
///
/// ```
/// use phyz_diff::multidual::{MultiDual, for_lanes};
/// use tang::Scalar;
///
/// // Differentiate f(x) = Σ xᵢ² in however many directions were asked for.
/// let xs = [1.0, 2.0, 3.0];
/// let grad = for_lanes(xs.len(), Grad(&xs));
/// assert_eq!(grad, vec![2.0, 4.0, 6.0]);
///
/// struct Grad<'a>(&'a [f64]);
/// impl phyz_diff::multidual::LaneOp for Grad<'_> {
///     type Out = Vec<f64>;
///     fn call<const N: usize>(self) -> Vec<f64> {
///         let seeded: Vec<MultiDual<N>> = self
///             .0
///             .iter()
///             .enumerate()
///             .map(|(i, &x)| MultiDual::var(x, i))
///             .collect();
///         let f = seeded
///             .iter()
///             .fold(MultiDual::<N>::constant(0.0), |acc, &x| acc + x * x);
///         (0..self.0.len()).map(|i| f.dual[i]).collect()
///     }
/// }
/// ```
pub fn for_lanes<Op: LaneOp>(lanes: usize, op: Op) -> Op::Out {
    match width_for(lanes) {
        1 => op.call::<1>(),
        2 => op.call::<2>(),
        4 => op.call::<4>(),
        8 => op.call::<8>(),
        _ => op.call::<16>(),
    }
}

/// A computation that is generic over the lane width, for [`for_lanes`].
///
/// This is a trait rather than a closure because a closure cannot be generic
/// over a const parameter.
pub trait LaneOp {
    /// What the computation produces.
    type Out;
    /// Run it at width `N`.
    fn call<const N: usize>(self) -> Self::Out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Dual;

    /// Exercise a chain long enough to compound any rule that disagrees:
    /// every arithmetic operator and every transcendental the dynamics can
    /// reach.
    fn workload<S: Scalar>(x: S, y: S) -> S {
        let a = x * y + x / y - y;
        let b = a.sin() * a.cos() + a.tan();
        let c = (b * b + S::ONE).sqrt().ln();
        let d = c.exp() * x.powi(3) + y.powf(S::from_f64(1.7));
        let e = d.recip() + d.abs() - d.tanh() + d.sinh() * d.cosh();
        let f = e.atan2(x) + x.max(y) - x.min(y) + e.signum() * e;
        f + x.asin() + y.acos() + (x * S::TWO).asinh() + (y + S::TWO).acosh() + x.atanh()
    }

    /// The property this whole module rests on: lane `k` of a `MultiDual` run
    /// equals a scalar `Dual` run seeded in direction `k`, **bit for bit**.
    ///
    /// Not "to 1e-15" — exactly. A rollout that promises bitwise
    /// reproducibility cannot have its gradients shift because a chunk width
    /// changed, so approximate agreement here would not be good enough.
    #[test]
    fn lanes_match_scalar_dual_bitwise() {
        let (x0, y0) = (0.37, 0.61);

        let dx = workload(Dual::var(x0), Dual::constant(y0));
        let dy = workload(Dual::constant(x0), Dual::var(y0));

        let m = workload(MultiDual::<2>::var(x0, 0), MultiDual::<2>::var(y0, 1));

        assert_eq!(m.real.to_bits(), dx.real.to_bits(), "value");
        assert_eq!(m.dual[0].to_bits(), dx.dual.to_bits(), "lane 0 vs ∂/∂x");
        assert_eq!(m.dual[1].to_bits(), dy.dual.to_bits(), "lane 1 vs ∂/∂y");
    }

    /// Widening must not change a lane's value: the same seed at width 1, 4
    /// and 16 has to produce identical bits, or `for_lanes` would make results
    /// depend on the model's DOF count.
    #[test]
    fn width_does_not_change_the_answer() {
        let (x0, y0) = (0.37, 0.61);
        let w1 = workload(MultiDual::<1>::var(x0, 0), MultiDual::<1>::constant(y0));
        let w4 = workload(MultiDual::<4>::var(x0, 0), MultiDual::<4>::constant(y0));
        let w16 = workload(MultiDual::<16>::var(x0, 0), MultiDual::<16>::constant(y0));

        assert_eq!(w1.dual[0].to_bits(), w4.dual[0].to_bits());
        assert_eq!(w1.dual[0].to_bits(), w16.dual[0].to_bits());
    }

    /// Unseeded lanes must stay exactly zero — a lane that picks up numerical
    /// dust would silently pollute the padding columns of the last chunk.
    #[test]
    fn unseeded_lanes_stay_zero() {
        let m = workload(MultiDual::<8>::var(0.37, 0), MultiDual::<8>::constant(0.61));
        for (k, d) in m.dual.iter().enumerate().skip(1) {
            assert_eq!(*d, 0.0, "lane {k} picked up a nonzero derivative");
        }
    }

    #[test]
    fn width_dispatch_picks_the_narrowest_cover() {
        assert_eq!(width_for(0), 1);
        assert_eq!(width_for(1), 1);
        assert_eq!(width_for(2), 2);
        assert_eq!(width_for(3), 4);
        assert_eq!(width_for(8), 8);
        assert_eq!(width_for(9), 16);
        // Beyond the widest supported width, chunking is the caller's job.
        assert_eq!(width_for(17), 16);
        assert_eq!(width_for(1000), 16);
    }

    #[test]
    fn for_lanes_reaches_the_expected_width() {
        struct Width;
        impl LaneOp for Width {
            type Out = usize;
            fn call<const N: usize>(self) -> usize {
                N
            }
        }
        assert_eq!(for_lanes(1, Width), 1);
        assert_eq!(for_lanes(5, Width), 8);
        assert_eq!(for_lanes(40, Width), 16);
    }

    /// A derivative that is actually right, checked against a closed form
    /// rather than only against the scalar path — so a rule that is wrong in
    /// both places cannot hide.
    #[test]
    fn derivatives_are_correct_not_merely_consistent() {
        // f(x, y) = x³·sin(y); ∂f/∂x = 3x²·sin(y), ∂f/∂y = x³·cos(y).
        let (x, y) = (1.3_f64, 0.7_f64);
        let f = MultiDual::<2>::var(x, 0).powi(3) * MultiDual::<2>::var(y, 1).sin();

        assert!((f.real - x.powi(3) * y.sin()).abs() < 1e-14);
        assert!((f.dual[0] - 3.0 * x * x * y.sin()).abs() < 1e-14);
        assert!((f.dual[1] - x.powi(3) * y.cos()).abs() < 1e-14);
    }
}
