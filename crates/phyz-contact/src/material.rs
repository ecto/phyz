//! Contact material properties.

// `pow` comes from `phyz_math::fp` rather than `f64::powf`: the impedance
// sigmoid is evaluated once per contact per step, and the platform libms
// disagree with each other in the last ulp. See `docs/determinism.md`.
use phyz_math::fp;

/// MuJoCo-style `solref`: the reference response of a violated constraint.
///
/// The pair `(timeconst, dampratio)` describes the mass-normalized damped
/// spring the solver *pretends* the constraint is, rather than a raw
/// stiffness/damping pair. Parameterizing by a time constant is what makes the
/// behaviour independent of the contacting bodies' masses: a 20 ms recovery is
/// 20 ms whether the box weighs a gram or a tonne.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolRef {
    /// Time constant of the constraint's reference response, in seconds.
    ///
    /// MuJoCo's default is `0.02`, and its guidance — that it should be at
    /// least twice the timestep — applies here for the same reason: the
    /// discrete correction below reduces to a per-step gain of
    /// `dt / (2*timeconst*dampratio^2 + dt)`, which saturates at 1 (a full,
    /// popping correction in one step) as `timeconst -> 0`.
    pub timeconst: f64,
    /// Damping ratio. `1` is critically damped, which is what a contact
    /// wants: penetration decays without overshooting into a bounce.
    pub dampratio: f64,
}

impl Default for SolRef {
    fn default() -> Self {
        // MuJoCo's defaults.
        Self {
            timeconst: 0.02,
            dampratio: 1.0,
        }
    }
}

impl SolRef {
    /// Fraction of the current penetration to remove in a step of `dt`.
    ///
    /// Derived from the reference spring rather than picked: a critically
    /// damped spring with time constant `tau` and damping ratio `zeta` has
    /// mass-normalized `k = 1/(tau^2 zeta^2)` and `b = 2/(tau zeta^2)`. One
    /// implicit step of that spring removes
    ///
    /// ```text
    ///   dt*k / (b + dt*k)  =  dt / (2*tau*zeta^2 + dt)
    /// ```
    ///
    /// of the violation. Taking the implicit (rather than the explicit
    /// `dt^2 k`) gain is what keeps the correction bounded by 1 for every
    /// `dt`, so a large step can never over-correct and launch the body.
    pub fn error_reduction(&self, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }
        let tau = self.timeconst.max(0.0);
        let zeta = self.dampratio.max(1e-6);
        let denom = 2.0 * tau * zeta * zeta + dt;
        if denom <= 0.0 { 1.0 } else { dt / denom }
    }
}

/// MuJoCo-style `solimp`: constraint impedance as a function of the violation.
///
/// Impedance `d` in `(0, 1)` blends between a fully soft constraint (`d -> 0`,
/// no force at all) and a fully rigid one (`d -> 1`). Making it depend on
/// penetration depth is the point: a contact that has just been made is soft,
/// so making and breaking contact stays smooth (and differentiable), while a
/// deeply penetrating contact stiffens up and pushes hard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolImp {
    /// Impedance at zero violation.
    pub dmin: f64,
    /// Impedance at and beyond `width` of violation.
    pub dmax: f64,
    /// Violation (in metres, for contacts) over which `dmin -> dmax`.
    pub width: f64,
    /// Where in `[0, 1]` the sigmoid's inflection sits.
    pub midpoint: f64,
    /// Sigmoid power. `1` is linear; larger is flatter at both ends.
    pub power: f64,
}

impl Default for SolImp {
    fn default() -> Self {
        // MuJoCo's defaults: (dmin, dmax, width, midpoint, power).
        Self {
            dmin: 0.9,
            dmax: 0.95,
            width: 0.001,
            midpoint: 0.5,
            power: 2.0,
        }
    }
}

impl SolImp {
    /// Impedance at a constraint violation of `r` metres.
    ///
    /// The interpolant is MuJoCo's polynomial sigmoid: `a x^p` below the
    /// midpoint and `1 - b (1-x)^p` above it, with `a` and `b` chosen so the
    /// halves meet at the midpoint. It is `C^1` in `r` everywhere except the
    /// clamp at `x = 1`.
    pub fn impedance(&self, r: f64) -> f64 {
        let dmin = self.dmin.clamp(1e-4, 1.0 - 1e-9);
        let dmax = self.dmax.clamp(1e-4, 1.0 - 1e-9);
        if self.width <= 0.0 {
            return dmax;
        }
        let x = (r.abs() / self.width).clamp(0.0, 1.0);
        let mid = self.midpoint.clamp(1e-6, 1.0 - 1e-6);
        let p = self.power.max(1.0);
        let y = if x <= mid {
            fp::pow(x, p) / fp::pow(mid, p - 1.0)
        } else {
            1.0 - fp::pow(1.0 - x, p) / fp::pow(1.0 - mid, p - 1.0)
        };
        dmin + y * (dmax - dmin)
    }

    /// `d(impedance)/dr` at a **non-negative** violation `r`.
    ///
    /// Mirrors [`Self::impedance`] branch for branch, including its clamps: past
    /// `width` the impedance is pinned at `dmax`, so the derivative is zero, and
    /// a non-positive `width` makes it constant everywhere.
    ///
    /// Defined for `r >= 0` only. `impedance` takes `r.abs()`, so the function
    /// has a kink at the origin whenever `power == 1`; for `power > 1` (the
    /// default) both one-sided derivatives are zero there and this returns the
    /// value that is correct from either side.
    pub fn impedance_derivative(&self, r: f64) -> f64 {
        let dmin = self.dmin.clamp(1e-4, 1.0 - 1e-9);
        let dmax = self.dmax.clamp(1e-4, 1.0 - 1e-9);
        if self.width <= 0.0 || r < 0.0 {
            return 0.0;
        }
        let x = r / self.width;
        if x >= 1.0 {
            // Clamped at dmax: the sigmoid has stopped moving.
            return 0.0;
        }
        let mid = self.midpoint.clamp(1e-6, 1.0 - 1e-6);
        let p = self.power.max(1.0);
        let dy_dx = if x <= mid {
            p * fp::pow(x, p - 1.0) / fp::pow(mid, p - 1.0)
        } else {
            p * fp::pow(1.0 - x, p - 1.0) / fp::pow(1.0 - mid, p - 1.0)
        };
        dy_dx * (dmax - dmin) / self.width
    }
}

/// Material properties for contact interactions.
#[derive(Debug, Clone)]
pub struct ContactMaterial {
    /// Contact stiffness (N/m).
    pub stiffness: f64,
    /// Contact damping (N·s/m).
    pub damping: f64,
    /// Coefficient of friction (dimensionless).
    pub friction: f64,
    /// Coefficient of restitution (0 = inelastic, 1 = elastic).
    ///
    /// Was `bounce`, a field no solver ever read — `ContactMaterial::bouncy()`
    /// did nothing. It is now honoured by the convex solve, entering as a
    /// target normal velocity rather than a post-solve velocity reset. See
    /// `docs/design/differentiable-contact.md` §4.3.
    pub restitution: f64,
    /// Constraint force mixing (for numerical stability).
    pub soft_cfm: f64,
    /// Error reduction parameter (for constraint drift correction).
    pub soft_erp: f64,
    /// Reference response of the contact constraint (MuJoCo `solref`).
    ///
    /// This is what gives the convex solve a penetration-recovery bias.
    /// Without it `ContactRow::depth` was carried around and never read, so
    /// penetration was never paid back and a stack crept forever.
    pub solref: SolRef,
    /// Depth-dependent constraint impedance (MuJoCo `solimp`).
    ///
    /// Supersedes a single scalar `ContactSolverConfig::regularization` as the
    /// softness knob: the regularizer is now `R = (1-d)/d * diag(A)` per row,
    /// with the config value acting as a floor.
    pub solimp: SolImp,
    /// Contact margin, in metres: how far apart surfaces may be and still be
    /// included in the solve.
    ///
    /// This used to be inert ("the narrow phase owns detection"). It is not:
    /// with *soft* contact a candidate at exactly zero depth still carries the
    /// full `solimp.dmin` share of the load, so a detection predicate that cuts
    /// off at zero gap cuts off a contact that was carrying real force. Measured
    /// on a humanoid stance, the least-loaded foot corner was carrying 22.3 N —
    /// 11% of body weight — on the step before it left the contact set, and went
    /// to 0 N the moment it rose a nanometre above the plane.
    ///
    /// The margin is the band above the surface over which a contact is still
    /// generated, with a *negative* `penetration_depth`, and its impedance
    /// tapers to zero — see [`ContactMaterial::impedance_at`]. That turns the
    /// step into a ramp.
    ///
    /// The default is [`SolImp::default`]'s `width`, i.e. the same 1 mm scale
    /// over which solimp already varies the impedance on the penetrating side.
    /// Picking the same scale makes impedance one continuous, `C^1` function of
    /// signed gap across the whole band `[-margin, +width]` rather than two
    /// curves glued at zero.
    ///
    /// Combines with `max` (MuJoCo's rule): the larger margin is the one asking
    /// for earlier detection, and `min` would silently defeat it.
    pub margin: f64,
}

impl Default for ContactMaterial {
    fn default() -> Self {
        Self {
            stiffness: 10000.0,
            damping: 100.0,
            friction: 0.5,
            restitution: 0.0,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
            solref: SolRef::default(),
            solimp: SolImp::default(),
            // Same scale as `SolImp::default().width`; see the field docs.
            margin: 0.001,
        }
    }
}

impl ContactMaterial {
    /// Create a new contact material with custom parameters.
    pub fn new(stiffness: f64, damping: f64, friction: f64, restitution: f64) -> Self {
        Self {
            stiffness,
            damping,
            friction,
            restitution,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
            ..Default::default()
        }
    }

    /// Constraint impedance for a contact at *signed* penetration `depth`,
    /// where negative means the surfaces are separated by `-depth` metres.
    ///
    /// On the penetrating side (`depth >= 0`) this is exactly
    /// [`SolImp::impedance`] and nothing has changed. Inside the margin band
    /// (`-margin < depth < 0`) it is `dmin` scaled by a smoothstep that reaches
    /// zero at `depth = -margin`.
    ///
    /// **Why impedance, and why this makes the force continuous.** The
    /// regularizer is `R = (1-d)/d * A_nn` (see
    /// [`crate::convex::regularization_diag`]), so a single contact resolving a
    /// free approach velocity `v` carries a normal impulse of about
    /// `v / (A_nn + R) = d * v / A_nn` — *linear in `d`*. Taking `d` smoothly to
    /// zero therefore takes the normal force smoothly to zero, and the contact
    /// leaves the set carrying nothing instead of carrying 11% of body weight.
    ///
    /// It also cannot pull: `d -> 0` means "infinitely soft", not "attractive".
    /// The normal impulse stays in the friction cone's `f_n >= 0` half-line, and
    /// `ContactRow::from_material` clamps the stabilization bias at
    /// `depth.max(0.0)`, so a separated contact has no bias to push *or* pull
    /// with. Friction rides on `mu * f_n` and vanishes with it.
    ///
    /// The join at `depth = 0` is `C^1`: smoothstep has zero slope at its top
    /// end, and the solimp sigmoid has zero slope at `x = 0` for `power > 1`.
    pub fn impedance_at(&self, depth: f64) -> f64 {
        if depth >= 0.0 {
            return self.solimp.impedance(depth);
        }
        let gap = -depth;
        if self.margin <= 0.0 || gap >= self.margin {
            return 0.0;
        }
        let s = 1.0 - gap / self.margin;
        let ramp = s * s * (3.0 - 2.0 * s);
        ramp * self.solimp.impedance(0.0)
    }

    /// `d(impedance)/d(depth)`, the derivative of [`Self::impedance_at`].
    ///
    /// This is not a nicety. Inside the margin band the stabilization bias is
    /// identically zero, so impedance is the *only* channel by which depth
    /// reaches the impulses — and it is a strong one: the measured force ramps
    /// from 17.66 N to 0 across a 1 mm band, roughly `1.8e4 N/m`. A gradient
    /// that dropped this term would report exactly zero sensitivity across the
    /// whole region where contact activation is differentiable, which is
    /// precisely the region a contact-timing gradient lives in.
    ///
    /// On the penetrating side it is the solimp sigmoid's own slope, worth up
    /// to `(dmax-dmin)/dmin` of the primary term. That used to be dropped as a
    /// "second-order path"; it is included now, so there is one expression
    /// rather than two regimes with different honesty.
    ///
    /// Both branches are exact, and they agree at `depth = 0` (both zero for
    /// `power > 1`), which is what makes [`Self::impedance_at`] `C^1` there.
    pub fn dimpedance_ddepth(&self, depth: f64) -> f64 {
        if depth >= 0.0 {
            return self.solimp.impedance_derivative(depth);
        }
        let gap = -depth;
        if self.margin <= 0.0 || gap >= self.margin {
            return 0.0;
        }
        // d/d(depth) of `smoothstep(s) * dmin` with `s = 1 + depth/margin`.
        let s = 1.0 - gap / self.margin;
        6.0 * s * (1.0 - s) / self.margin * self.solimp.impedance(0.0)
    }

    /// Create a bouncy material (high restitution).
    pub fn bouncy() -> Self {
        Self {
            stiffness: 10000.0,
            damping: 50.0,
            friction: 0.3,
            restitution: 0.8,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
            ..Default::default()
        }
    }

    /// Create a soft material (low stiffness).
    pub fn soft() -> Self {
        Self {
            stiffness: 1000.0,
            damping: 200.0,
            friction: 0.7,
            restitution: 0.1,
            soft_cfm: 0.001,
            soft_erp: 0.2,
            ..Default::default()
        }
    }

    /// Create a rigid material (high stiffness).
    pub fn rigid() -> Self {
        Self {
            stiffness: 50000.0,
            damping: 100.0,
            friction: 0.5,
            restitution: 0.0,
            soft_cfm: 0.00001,
            soft_erp: 0.2,
            ..Default::default()
        }
    }

    /// Combine the two materials of a contacting pair into the one material
    /// the contact is actually solved with.
    ///
    /// A contact belongs to *two* surfaces, so using one side's material — as
    /// `assemble` did, always taking `body_i`'s — makes the physics depend on
    /// which body the narrow phase happened to list first. Rubber on ice slid
    /// or gripped depending on collision-pair ordering.
    ///
    /// The rules, and why each one:
    ///
    /// - **Friction: `max`.** This is MuJoCo's rule (`mj_contactParam` takes
    ///   the elementwise maximum of the two geoms' friction when neither has
    ///   priority), and it is the conservative choice for a simulator: a
    ///   sticky surface stays sticky against a slippery one, so a foot does
    ///   not silently lose grip because the floor was authored with a low
    ///   default. Note this is `max`, not `min` — the mixing rule matters
    ///   more than any single material's number, so it follows MuJoCo rather
    ///   than the intuition that the slipperier surface should win.
    /// - **Margin: `max`.** Also MuJoCo's. The larger margin is the one that
    ///   wants contacts generated earlier; taking the min would silently
    ///   defeat a surface that asked for early detection.
    /// - **Restitution: `max`.** Same argument: a bouncy ball stays bouncy on
    ///   an inelastic floor.
    /// - **Everything stiffness-like — `stiffness`, `damping`, `soft_cfm`,
    ///   `soft_erp`, and every `solref`/`solimp` scalar: geometric mean.**
    ///   These span orders of magnitude (`soft_cfm` from `1e-5` to `1e-3`,
    ///   stiffness from `1e3` to `5e4`), and the arithmetic mean of two such
    ///   numbers is just the larger one — soft rubber against a rigid floor
    ///   would come out rigid. The geometric mean is the mean that respects a
    ///   log-scaled quantity, and it also makes the combination invariant to
    ///   the units the quantity is expressed in. MuJoCo uses a `solmix`-
    ///   weighted arithmetic average here; with no per-material mix weight to
    ///   honour, the geometric mean is the better-behaved default.
    ///
    /// The rule is commutative in both arguments, which is the property that
    /// makes the result independent of narrow-phase ordering.
    pub fn combine(a: &Self, b: &Self) -> Self {
        // Geometric mean, guarding the sign/zero cases: `sqrt` of a negative
        // is NaN, and a zero on either side must stay zero.
        fn gmean(x: f64, y: f64) -> f64 {
            if x <= 0.0 || y <= 0.0 {
                0.5 * (x + y)
            } else {
                (x * y).sqrt()
            }
        }

        Self {
            stiffness: gmean(a.stiffness, b.stiffness),
            damping: gmean(a.damping, b.damping),
            friction: a.friction.max(b.friction),
            restitution: a.restitution.max(b.restitution),
            soft_cfm: gmean(a.soft_cfm, b.soft_cfm),
            soft_erp: gmean(a.soft_erp, b.soft_erp),
            solref: SolRef {
                timeconst: gmean(a.solref.timeconst, b.solref.timeconst),
                dampratio: gmean(a.solref.dampratio, b.solref.dampratio),
            },
            solimp: SolImp {
                dmin: gmean(a.solimp.dmin, b.solimp.dmin),
                dmax: gmean(a.solimp.dmax, b.solimp.dmax),
                width: gmean(a.solimp.width, b.solimp.width),
                midpoint: gmean(a.solimp.midpoint, b.solimp.midpoint),
                power: gmean(a.solimp.power, b.solimp.power),
            },
            margin: a.margin.max(b.margin),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impedance_spans_dmin_to_dmax() {
        let s = SolImp::default();
        assert!((s.impedance(0.0) - s.dmin).abs() < 1e-12);
        assert!((s.impedance(s.width) - s.dmax).abs() < 1e-12);
        assert!((s.impedance(10.0 * s.width) - s.dmax).abs() < 1e-12);
        // Monotone in between, which is what makes deeper penetration stiffer.
        let mut prev = s.impedance(0.0);
        for k in 1..=50 {
            let d = s.impedance(s.width * k as f64 / 50.0);
            assert!(
                d >= prev - 1e-15,
                "impedance must be monotone: {prev} -> {d}"
            );
            prev = d;
        }
    }

    #[test]
    fn impedance_sigmoid_halves_meet_at_the_midpoint() {
        let s = SolImp::default();
        let d = s.impedance(s.width * s.midpoint);
        let want = s.dmin + s.midpoint * (s.dmax - s.dmin);
        assert!((d - want).abs() < 1e-12, "{d} vs {want}");
    }

    #[test]
    fn error_reduction_is_bounded_and_monotone_in_dt() {
        let r = SolRef::default();
        for dt in [1e-4, 1e-3, 2e-3, 1e-2, 1.0, 100.0] {
            let e = r.error_reduction(dt);
            assert!((0.0..=1.0).contains(&e), "dt={dt} gave {e}");
        }
        assert!(r.error_reduction(2e-3) < r.error_reduction(1e-2));
        // A zero time constant means "fix it this step".
        let instant = SolRef {
            timeconst: 0.0,
            dampratio: 1.0,
        };
        assert!((instant.error_reduction(2e-3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impedance_at_ramps_to_zero_across_the_margin() {
        let m = ContactMaterial::default();
        assert!(m.margin > 0.0);

        // Penetrating side is untouched.
        for r in [0.0, 1e-5, 1e-4, 1e-3, 1e-2] {
            assert_eq!(m.impedance_at(r), m.solimp.impedance(r));
        }

        // Monotone down to exactly zero at the band edge, and zero past it.
        let mut prev = m.impedance_at(0.0);
        for k in 1..=100 {
            let d = m.impedance_at(-m.margin * k as f64 / 100.0);
            assert!(d <= prev + 1e-15, "must be monotone: {prev} -> {d}");
            assert!(d >= 0.0, "impedance must never go negative: {d}");
            prev = d;
        }
        assert_eq!(m.impedance_at(-m.margin), 0.0);
        assert_eq!(m.impedance_at(-2.0 * m.margin), 0.0);

        // A zero margin is the old hard cutoff, exactly.
        let hard = ContactMaterial {
            margin: 0.0,
            ..Default::default()
        };
        assert_eq!(hard.impedance_at(-1e-12), 0.0);
        assert_eq!(hard.impedance_at(0.0), hard.solimp.dmin);
    }

    /// The join at zero depth is `C^1`, which is the property that makes the
    /// contact-activation event differentiable rather than merely continuous.
    /// Both one-sided derivatives are zero there: smoothstep is flat at its top
    /// end, and the solimp sigmoid is flat at `x = 0` for `power > 1`.
    #[test]
    fn impedance_at_is_c1_across_zero_depth() {
        let m = ContactMaterial::default();
        let h = 1e-9;
        let below = (m.impedance_at(0.0) - m.impedance_at(-h)) / h;
        let above = (m.impedance_at(h) - m.impedance_at(0.0)) / h;
        // Compared against the ramp's own characteristic slope — smoothstep
        // peaks at `1.5 * dmin / margin` — since an absolute tolerance would
        // just be measuring the choice of `margin`.
        let scale = 1.5 * m.solimp.dmin / m.margin;
        assert!(
            below.abs() < 1e-4 * scale,
            "left slope at 0 is {below}, vs ramp slope scale {scale}"
        );
        assert!(
            above.abs() < 1e-4 * scale,
            "right slope at 0 is {above}, vs ramp slope scale {scale}"
        );
        // And they agree with each other, which is what "C^1" means here.
        assert!((below - above).abs() < 1e-4 * scale);

        // Continuous everywhere in the band: no sample-to-sample cliff.
        let n = 20_000;
        let mut prev = m.impedance_at(-m.margin);
        for k in 1..=n {
            let d = m.impedance_at(-m.margin + 2.0 * m.margin * k as f64 / n as f64);
            assert!(
                (d - prev).abs() < 1e-3,
                "impedance jumped by {} at step {k}",
                (d - prev).abs()
            );
            prev = d;
        }
    }

    /// `dimpedance_ddepth` against a central difference of `impedance_at`,
    /// across both branches and the join between them.
    #[test]
    fn impedance_derivative_matches_finite_difference() {
        let m = ContactMaterial::default();
        let h = 1e-9;
        let mut worst = 0.0f64;
        // Inside the margin band and through the solimp sigmoid, but not at
        // the two clamp kinks (-margin and +width) where the derivative is
        // genuinely one-sided, and not at the join itself: a central difference
        // at zero straddles both branches, so it measures the curvature there
        // rather than the slope. The join is covered by
        // `impedance_at_is_c1_across_zero_depth`, which is the assertion that
        // actually belongs there — both one-sided slopes are zero.
        for k in (-95i32..=95).filter(|k| *k != 0) {
            let depth = if k < 0 {
                m.margin * k as f64 / 100.0
            } else {
                m.solimp.width * k as f64 / 100.0
            };
            let fd = (m.impedance_at(depth + h) - m.impedance_at(depth - h)) / (2.0 * h);
            let got = m.dimpedance_ddepth(depth);
            let rel = (got - fd).abs() / fd.abs().max(1.0);
            worst = worst.max(rel);
            assert!(rel < 1e-5, "depth {depth}: analytic {got} vs FD {fd}");
        }
        assert!(worst < 1e-5, "worst relative error {worst}");

        // Past the clamps the impedance is pinned and the derivative is zero.
        assert_eq!(m.dimpedance_ddepth(-2.0 * m.margin), 0.0);
        assert_eq!(m.dimpedance_ddepth(2.0 * m.solimp.width), 0.0);
    }

    #[test]
    fn margin_combines_by_max() {
        let a = ContactMaterial {
            margin: 5e-3,
            ..Default::default()
        };
        let b = ContactMaterial {
            margin: 1e-4,
            ..Default::default()
        };
        assert_eq!(ContactMaterial::combine(&a, &b).margin, 5e-3);
        assert_eq!(ContactMaterial::combine(&b, &a).margin, 5e-3);
    }

    #[test]
    fn combination_is_commutative_and_picks_the_documented_extremes() {
        let a = ContactMaterial::soft();
        let b = ContactMaterial::rigid();
        let ab = ContactMaterial::combine(&a, &b);
        let ba = ContactMaterial::combine(&b, &a);
        assert_eq!(ab.friction, ba.friction);
        assert_eq!(ab.stiffness, ba.stiffness);
        assert_eq!(ab.soft_cfm, ba.soft_cfm);

        assert_eq!(ab.friction, a.friction.max(b.friction));
        // Geometric, not arithmetic: soft rubber on a rigid floor must not
        // come out rigid.
        assert!((ab.stiffness - (a.stiffness * b.stiffness).sqrt()).abs() < 1e-9);
        assert!(ab.stiffness < 0.5 * (a.stiffness + b.stiffness));
    }

    #[test]
    fn combining_a_material_with_itself_is_the_identity() {
        let m = ContactMaterial::bouncy();
        let c = ContactMaterial::combine(&m, &m);
        assert!((c.stiffness - m.stiffness).abs() < 1e-9);
        assert!((c.friction - m.friction).abs() < 1e-12);
        assert!((c.restitution - m.restitution).abs() < 1e-12);
        assert!((c.solref.timeconst - m.solref.timeconst).abs() < 1e-12);
        assert!((c.solimp.width - m.solimp.width).abs() < 1e-12);
    }
}
