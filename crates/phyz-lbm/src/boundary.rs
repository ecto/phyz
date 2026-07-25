//! Declarative domain boundary conditions.
//!
//! Boundaries are declared **once** on the solver and applied automatically by
//! every `step()`. Previously the streaming step wrapped unconditionally with
//! `rem_euclid`, so a domain was periodic unless the caller remembered to patch
//! wall nodes by hand after every single step — a silent correctness trap.
//!
//! Streaming still wraps; that is deliberate and harmless. The populations that
//! wrap into a face are exactly the ones a boundary condition has to overwrite
//! (those with `e_i · n > 0` for inward normal `n`), so the boundary pass
//! replaces every wrapped value and nothing else.
//!
//! ```
//! use phyz_lbm::{Boundaries, Boundary, LatticeBoltzmann2D, Side};
//!
//! // Plane channel: periodic along x, no-slip walls top and bottom.
//! let bc = Boundaries::periodic()
//!     .set(1, Side::Min, Boundary::NoSlip)
//!     .set(1, Side::Max, Boundary::NoSlip);
//! let mut lbm = LatticeBoltzmann2D::new(8, 32, 0.05).with_boundaries(bc);
//! lbm.initialize_uniform(1.0, [0.0, 0.0]);
//! lbm.step();
//! ```

/// Which end of an axis a face sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The `index == 0` face.
    Min,
    /// The `index == n - 1` face.
    Max,
}

impl Side {
    /// Index into the per-axis face pair.
    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Side::Min => 0,
            Side::Max => 1,
        }
    }
}

/// Condition applied to one face of the domain.
///
/// `D` is the spatial dimension (2 for D2Q9, 3 for D3Q19).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Boundary<const D: usize> {
    /// Wrap around to the opposite face. This is the default.
    #[default]
    Periodic,
    /// Stationary no-slip wall (halfway bounce-back). The wall plane sits half
    /// a lattice spacing outside the last fluid node.
    NoSlip,
    /// No-slip wall translating in its own plane, e.g. a cavity lid.
    /// Halfway bounce-back with the standard momentum correction.
    MovingWall([f64; D]),
    /// Prescribed velocity (inlet). Zou–He in 2D, Guo non-equilibrium
    /// extrapolation in 3D — both preserve the non-equilibrium part.
    Velocity([f64; D]),
    /// Prescribed density/pressure (outlet), zero tangential velocity.
    Pressure(f64),
    /// Free-slip / mirror plane (specular reflection).
    Symmetry,
}

impl<const D: usize> Boundary<D> {
    /// Whether this face needs no post-streaming treatment.
    #[inline]
    pub fn is_periodic(&self) -> bool {
        matches!(self, Boundary::Periodic)
    }
}

/// The full set of domain faces: `faces[axis][side]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundaries<const D: usize> {
    /// Indexed by `[axis][Side::idx()]`.
    pub faces: [[Boundary<D>; 2]; D],
}

impl<const D: usize> Default for Boundaries<D> {
    fn default() -> Self {
        Self::periodic()
    }
}

impl<const D: usize> Boundaries<D> {
    /// All faces periodic.
    pub fn periodic() -> Self {
        Self {
            faces: [[Boundary::Periodic; 2]; D],
        }
    }

    /// All faces no-slip walls (a closed box).
    pub fn closed_box() -> Self {
        Self {
            faces: [[Boundary::NoSlip; 2]; D],
        }
    }

    /// Set one face. Chainable.
    #[must_use]
    pub fn set(mut self, axis: usize, side: Side, boundary: Boundary<D>) -> Self {
        self.faces[axis][side.idx()] = boundary;
        self
    }

    /// Both faces of one axis.
    #[must_use]
    pub fn set_axis(mut self, axis: usize, boundary: Boundary<D>) -> Self {
        self.faces[axis] = [boundary; 2];
        self
    }

    /// Look up one face.
    #[inline]
    pub fn get(&self, axis: usize, side: Side) -> Boundary<D> {
        self.faces[axis][side.idx()]
    }

    /// True if every face is periodic (lets the solver skip the boundary pass).
    pub fn all_periodic(&self) -> bool {
        self.faces
            .iter()
            .all(|pair| pair.iter().all(Boundary::is_periodic))
    }
}

/// A lid-driven cavity: all walls no-slip, the `y = ny-1` lid moving at `u`.
pub fn cavity_2d(lid_velocity: [f64; 2]) -> Boundaries<2> {
    Boundaries::closed_box().set(1, Side::Max, Boundary::MovingWall(lid_velocity))
}

/// A plane channel: periodic along `x`, no-slip walls along `y`.
pub fn channel_2d() -> Boundaries<2> {
    Boundaries::periodic().set_axis(1, Boundary::NoSlip)
}

/// A driven channel: velocity inlet at `x = 0`, pressure outlet at `x = nx-1`,
/// no-slip walls along `y`.
pub fn inlet_outlet_channel_2d(inlet: [f64; 2], outlet_density: f64) -> Boundaries<2> {
    Boundaries::periodic()
        .set_axis(1, Boundary::NoSlip)
        .set(0, Side::Min, Boundary::Velocity(inlet))
        .set(0, Side::Max, Boundary::Pressure(outlet_density))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_periodic() {
        let b: Boundaries<2> = Boundaries::default();
        assert!(b.all_periodic());
        let b3: Boundaries<3> = Boundaries::default();
        assert!(b3.all_periodic());
    }

    #[test]
    fn setters_target_the_right_face() {
        let b = channel_2d();
        assert!(b.get(0, Side::Min).is_periodic());
        assert!(b.get(0, Side::Max).is_periodic());
        assert_eq!(b.get(1, Side::Min), Boundary::NoSlip);
        assert_eq!(b.get(1, Side::Max), Boundary::NoSlip);
        assert!(!b.all_periodic());
    }

    #[test]
    fn cavity_lid_is_on_top() {
        let b = cavity_2d([0.1, 0.0]);
        assert_eq!(b.get(1, Side::Max), Boundary::MovingWall([0.1, 0.0]));
        assert_eq!(b.get(1, Side::Min), Boundary::NoSlip);
        assert_eq!(b.get(0, Side::Min), Boundary::NoSlip);
    }
}
