//! Mechanisms shared by the loop tests.
//!
//! Both are modelled the way the crate expects: a spanning **tree** that ABA
//! can integrate, plus a loop-closure constraint that re-imposes the cut
//! joint. Every one of them is genuinely unrepresentable in `phyz-rigid`
//! alone.
//!
//! All motion is in the XY plane, all revolute axes are +Z (the joint default),
//! gravity is -Y.

#![allow(dead_code)]

use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};

/// Uniform rod of length `len` lying along its body frame's +X axis, hinged at
/// the body origin. Inertia is about the COM, which is what `SpatialInertia`
/// expects (see `SpatialInertia::rod`, whose COM is the origin).
pub fn rod_x(mass: f64, len: f64) -> SpatialInertia {
    let i = mass * len * len / 12.0;
    SpatialInertia::new(
        mass,
        Vec3::new(0.5 * len, 0.0, 0.0),
        Mat3::from_diagonal(&Vec3::new(0.0, i, i)),
    )
}

/// Four-bar linkage geometry. Ground pivots at the origin and at `(d, 0)`.
pub struct FourBar {
    pub crank: f64,
    pub coupler: f64,
    pub rocker: f64,
    pub ground: f64,
}

/// The crank-rocker used throughout the tests, exactly closable at
/// `q = [pi/2, atan2(1,2) - pi/2, pi/2]`: crank tip at `(0, 1)`, rocker tip at
/// `(2, 2)`, and `|(2,2) - (0,1)| = sqrt(5)` is the coupler length.
pub const FOUR_BAR: FourBar = FourBar {
    crank: 1.0,
    coupler: 2.23606797749979, // sqrt(5), spelled out so the model is a constant
    rocker: 2.0,
    ground: 2.0,
};

/// Spanning tree of the four-bar: crank -> coupler as one chain, rocker as a
/// second chain grounded at `(d, 0)`. The cut joint is the coupler-rocker pin.
pub fn four_bar_model(dt: f64) -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -9.81, 0.0))
        .dt(dt)
        .add_revolute_body(
            "crank",
            -1,
            SpatialTransform::identity(),
            rod_x(1.0, FOUR_BAR.crank),
        )
        .add_revolute_body(
            "coupler",
            0,
            SpatialTransform::new(Mat3::identity(), Vec3::new(FOUR_BAR.crank, 0.0, 0.0)),
            rod_x(1.0, FOUR_BAR.coupler),
        )
        .add_revolute_body(
            "rocker",
            -1,
            SpatialTransform::new(Mat3::identity(), Vec3::new(FOUR_BAR.ground, 0.0, 0.0)),
            rod_x(1.0, FOUR_BAR.rocker),
        )
        .build()
}

/// The assembled configuration of [`four_bar_model`], exact to double
/// precision by construction.
pub fn four_bar_state(model: &Model) -> State {
    let mut s = model.default_state();
    let theta1 = std::f64::consts::FRAC_PI_2;
    s.q[0] = theta1;
    s.q[1] = (1.0_f64).atan2(2.0) - theta1; // coupler angle, relative to crank
    s.q[2] = std::f64::consts::FRAC_PI_2;
    s
}

/// The loop closure of [`four_bar_model`]: coupler tip pinned to rocker tip.
pub fn four_bar_closure() -> phyz_loop::LoopConstraintSet {
    let mut set = phyz_loop::LoopConstraintSet::new();
    set.push(phyz_loop::LoopConstraint::point(
        "coupler-rocker pin",
        phyz_loop::Anchor::body(1, Vec3::new(FOUR_BAR.coupler, 0.0, 0.0)),
        phyz_loop::Anchor::body(2, Vec3::new(FOUR_BAR.rocker, 0.0, 0.0)),
    ));
    set
}

/// Slider-crank geometry: crank radius `r` at the origin, connecting rod `l`,
/// slider translating along +X through the origin.
pub const CRANK_R: f64 = 1.0;
pub const ROD_L: f64 = 3.0;

/// Spanning tree of the slider-crank: crank -> rod as one chain, slider as a
/// prismatic body grounded at the origin. The cut joint is the rod-slider pin.
pub fn slider_crank_model(dt: f64) -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -9.81, 0.0))
        .dt(dt)
        .add_revolute_body(
            "crank",
            -1,
            SpatialTransform::identity(),
            rod_x(1.0, CRANK_R),
        )
        .add_revolute_body(
            "rod",
            0,
            SpatialTransform::new(Mat3::identity(), Vec3::new(CRANK_R, 0.0, 0.0)),
            rod_x(1.0, ROD_L),
        )
        .add_prismatic_body(
            "slider",
            -1,
            SpatialTransform::identity(),
            Vec3::new(1.0, 0.0, 0.0),
            SpatialInertia::new(
                2.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1e-3, 1e-3, 1e-3)),
            ),
        )
        .build()
}

/// Closed-form slider displacement for crank angle `theta`:
/// `x = r cos(theta) + sqrt(l^2 - r^2 sin^2(theta))`.
pub fn slider_x(theta: f64) -> f64 {
    CRANK_R * theta.cos() + (ROD_L * ROD_L - CRANK_R * CRANK_R * theta.sin() * theta.sin()).sqrt()
}

/// Closed-form slider velocity: `dx/dtheta * thetadot`.
pub fn slider_xdot(theta: f64, thetadot: f64) -> f64 {
    let s = theta.sin();
    let root = (ROD_L * ROD_L - CRANK_R * CRANK_R * s * s).sqrt();
    let dxdtheta = -CRANK_R * s - CRANK_R * CRANK_R * s * theta.cos() / root;
    dxdtheta * thetadot
}

/// The slider-crank assembled at crank angle `theta`, at rest.
pub fn slider_crank_state(model: &Model, theta: f64) -> State {
    let mut s = model.default_state();
    let tip = Vec3::new(CRANK_R * theta.cos(), CRANK_R * theta.sin(), 0.0);
    let x = slider_x(theta);
    // Absolute rod angle points from the crank tip to the slider pin.
    let rod_abs = (0.0 - tip.y).atan2(x - tip.x);
    s.q[0] = theta;
    s.q[1] = rod_abs - theta;
    s.q[2] = x;
    s
}

/// The loop closure of [`slider_crank_model`]: rod tip pinned to the slider.
pub fn slider_crank_closure() -> phyz_loop::LoopConstraintSet {
    let mut set = phyz_loop::LoopConstraintSet::new();
    set.push(phyz_loop::LoopConstraint::point(
        "rod-slider pin",
        phyz_loop::Anchor::body(1, Vec3::new(ROD_L, 0.0, 0.0)),
        phyz_loop::Anchor::body(2, Vec3::zeros()),
    ));
    set
}
