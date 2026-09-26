//! A cylinder's ground contact is its lowest LINE.
//!
//! The ground fast path used to sample each end cap's rim at four *body-frame*
//! angles. Those angles rotate with the wheel and never include the barrel
//! point that actually touches, so a rolling cylinder's effective radius
//! ripples by `r(1 − cos 45°) = 0.293 r` once per quarter turn — 7.9 mm on the
//! 27 mm skateboard wheel that filed the gap (ipse #233,
//! `GAP_CYLINDER_GROUND`). The wheel was, in the report's words, a square.
//!
//! The analytic replacement takes the lowest generator line `c ± hâ + r·û`
//! with `û = −normalize(ẑ − (ẑ·â)â)`, both ends carrying their own depth.
//! These tests pin down what that buys and what it must not break:
//!
//! - the ripple, measured against the old sampler reimplemented here (§1),
//! - both ends of the line, each with its own depth, under tilt (§2),
//! - agreement with a brute-force surface reference over 200 orientations (§3),
//! - the support polygon a cylinder standing on its cap still needs (§4),
//! - the per-contact world offset the convex adjoint anchors on (§5).

use phyz_contact::{find_ground_contacts_model, find_ground_contacts_model_with_offset};
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

/// A 27 mm skateboard wheel: the shape the gap was measured on.
const R: f64 = 0.027;
const H: f64 = 0.034;

fn free_body_model(geom: Geometry) -> Model {
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "b",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 1e-4),
        )
        .build();
    m.bodies[0].geometry = Some(geom);
    m
}

/// Place the body at `pos`, physically rotated by `rot` (world→body already).
fn posed_state(model: &Model, pos: Vec3, world_to_body: Mat3) -> State {
    let mut state = model.default_state();
    state.body_xform[0] = SpatialTransform {
        rot: world_to_body,
        pos,
    };
    state
}

/// World→body for a body physically rotated by `angle` about `axis`.
fn w2b(axis: Vec3, angle: f64) -> Mat3 {
    Quat::from_axis_angle(axis, -angle).to_matrix()
}

/// The deepest ground contact, or `None` if nothing touches.
fn deepest(model: &Model, state: &State, margin: f64) -> Option<f64> {
    find_ground_contacts_model(model, state, 0.0, margin)
        .into_iter()
        .map(|c| c.penetration_depth)
        .fold(None, |acc: Option<f64>, d| {
            Some(acc.map_or(d, |a| a.max(d)))
        })
}

// -------------------------------------------------------------------------
// §1 The ripple
// -------------------------------------------------------------------------

/// The sampler this change replaces, reimplemented so the improvement is
/// measured rather than asserted: rim points of both caps at four fixed
/// body-frame angles.
fn legacy_deepest(xform: &SpatialTransform, radius: f64, height: f64) -> f64 {
    let b2w = |v: Vec3| xform.rot.transpose() * v;
    let hz = b2w(Vec3::new(0.0, 0.0, height * 0.5));
    let ex = b2w(Vec3::x()) * radius;
    let ey = b2w(Vec3::y()) * radius;
    let mut lowest = f64::INFINITY;
    for k in 0..4 {
        let t = k as f64 * std::f64::consts::FRAC_PI_2;
        let r = ex * t.cos() + ey * t.sin();
        for cap in [hz, -hz] {
            lowest = lowest.min((xform.pos + cap + r).z);
        }
    }
    -lowest
}

/// Spin a wheel about its own axis at a fixed axle height and watch the depth.
///
/// A wheel is a wheel: the deepest point is `r` below the axle for every spin
/// angle, so the depth must not move at all. The old sampler's does, by
/// `0.293 r`, which is the whole bug — that band is a 7.9 mm bump the wheel
/// climbs four times per revolution.
#[test]
fn rolling_ripple_collapses_from_millimetres_to_rounding() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    // Axis along world y (a wheel on an axle), axle at exactly one radius.
    let lay = w2b(Vec3::x(), std::f64::consts::FRAC_PI_2);
    let axle_z = R;

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lo_old, mut hi_old) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..720 {
        let spin = i as f64 * std::f64::consts::TAU / 720.0;
        // Spin about the axle the wheel already lies along. (`lay` first in
        // the world→body product is the spin applied in the laid frame; the
        // other order tumbles the axle instead, which is a different — and
        // silently ripple-free — motion.)
        let rot = lay.mul_mat(&w2b(Vec3::y(), spin));
        let state = posed_state(&model, Vec3::new(0.0, 0.0, axle_z), rot);
        let d = deepest(&model, &state, 1e-3).expect("a wheel at z = r touches");
        lo = lo.min(d);
        hi = hi.max(d);
        let old = legacy_deepest(&state.body_xform[0], R, H);
        lo_old = lo_old.min(old);
        hi_old = hi_old.max(old);
    }

    let ripple = hi - lo;
    let ripple_old = hi_old - lo_old;
    println!("ripple: analytic {ripple:.3e} m, four-rim-sample {ripple_old:.3e} m");

    // The old sampler's ripple is r(1 - cos 45°) = 7.9 mm, and this test is
    // the record of that number rather than a claim about it.
    assert!(
        (ripple_old - R * (1.0 - std::f64::consts::FRAC_PI_4.cos())).abs() < 1e-9,
        "legacy ripple {ripple_old:.6} m is not r(1 - cos 45°)"
    );
    assert!(
        ripple_old > 7.8e-3,
        "legacy ripple {ripple_old:.6} m under 7.8 mm — the fixture stopped \
         reproducing the reported bug"
    );
    // Analytic: the support point is exactly r below the axle at every angle,
    // so all that is left is the rounding of the rotation itself.
    assert!(
        ripple < 1e-15,
        "analytic ripple {ripple:.3e} m: the contact still moves with spin"
    );
}

// -------------------------------------------------------------------------
// §2 The line, and both of its ends
// -------------------------------------------------------------------------

/// A level cylinder lying on its side touches along a segment: two contacts,
/// at the two cap planes, at equal depth.
#[test]
fn level_cylinder_contacts_both_ends_of_its_lowest_line() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let state = posed_state(
        &model,
        Vec3::new(0.0, 0.0, R - 1e-3),
        w2b(Vec3::x(), std::f64::consts::FRAC_PI_2),
    );
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert_eq!(contacts.len(), 2, "a lying cylinder rests on a line");
    for c in &contacts {
        assert!(
            (c.penetration_depth - 1e-3).abs() < 1e-12,
            "depth {:.6e} m, expected 1e-3",
            c.penetration_depth
        );
    }
    // The axis lies along world y after the roll, so the two ends straddle y.
    let ys: Vec<f64> = contacts.iter().map(|c| c.contact_point.y).collect();
    let span = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - ys.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (span - H).abs() < 1e-12,
        "the contact segment spans {span:.6} m, expected the height {H:.6} m"
    );
}

/// Tilt the axis and the two ends stop agreeing: the low end is deeper by
/// `h·sin(tilt)`, and only it is in contact once the tilt exceeds the margin.
#[test]
fn tilting_the_axis_separates_the_two_end_depths() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let tilt = 0.05_f64;
    // Lay the axis along y, then pitch it about x by `tilt`.
    let rot = w2b(Vec3::x(), std::f64::consts::FRAC_PI_2 + tilt);
    let state = posed_state(&model, Vec3::new(0.0, 0.0, R), rot);
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 1.0);
    // With a metre of margin every candidate survives; the two line ends are
    // the two deepest.
    let mut depths: Vec<f64> = contacts.iter().map(|c| c.penetration_depth).collect();
    depths.sort_by(|a, b| b.total_cmp(a));
    let gap = depths[0] - depths[1];
    let expect = H * tilt.sin();
    assert!(
        (gap - expect).abs() < 1e-12,
        "end-to-end depth difference {gap:.6e} m, expected h·sin(tilt) = {expect:.6e} m"
    );
    // And the deeper end is the true lowest point: r·cos(tilt) below the axis
    // and h/2·sin(tilt) below the centre.
    let expect_deep = R * tilt.cos() + 0.5 * H * tilt.sin() - R;
    assert!(
        (depths[0] - expect_deep).abs() < 1e-12,
        "deepest {:.6e} m, expected {expect_deep:.6e} m",
        depths[0]
    );
}

// -------------------------------------------------------------------------
// §3 Against a brute-force reference
// -------------------------------------------------------------------------

/// The lowest point of the cylinder's surface, found by dense sampling.
///
/// Slow and dumb on purpose: barrel and both cap discs, 2000 azimuths, so it
/// converges to the true minimum from above and cannot inherit a bug from the
/// detector it is checking.
fn brute_force_lowest(xform: &SpatialTransform, radius: f64, height: f64) -> f64 {
    let b2w_p = |v: Vec3| xform.pos + xform.rot.transpose() * v;
    let mut lowest = f64::INFINITY;
    let n = 2000;
    for i in 0..n {
        let t = i as f64 * std::f64::consts::TAU / n as f64;
        let (c, s) = (t.cos(), t.sin());
        for &z in &[-0.5 * height, 0.5 * height] {
            // Cap disc: every radius, not only the rim.
            for j in 0..=8 {
                let rr = radius * j as f64 / 8.0;
                lowest = lowest.min(b2w_p(Vec3::new(rr * c, rr * s, z)).z);
            }
        }
        // Barrel: the rim circle swept along the axis is covered by the caps'
        // rims plus the interior samples below.
        for j in 0..=8 {
            let z = -0.5 * height + height * j as f64 / 8.0;
            lowest = lowest.min(b2w_p(Vec3::new(radius * c, radius * s, z)).z);
        }
    }
    lowest
}

/// Over 200 orientations spanning level, tilted and upright, the analytic
/// deepest contact must agree with the brute-force surface minimum.
///
/// The reference samples the surface, so it can only ever report a point at or
/// *above* the true minimum: the analytic answer is allowed to be deeper by
/// the sampling resolution and must never be shallower.
#[test]
fn analytic_depth_matches_a_brute_force_surface_scan() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    // 2000 azimuths over a 27 mm rim leaves 2πr/2000 spacing, and the chord
    // sag over half of that is what the reference can miss by.
    let tol = R * (1.0 - (std::f64::consts::PI / 2000.0).cos()) + 1e-12;

    let mut worst: f64 = 0.0;
    for i in 0..200 {
        // A spread of axes and spins that is deterministic and not aligned to
        // the sampling of either implementation.
        let a = i as f64 * 0.31_f64;
        let b = i as f64 * 0.97_f64;
        let rot = w2b(Vec3::new(0.3, 0.9, 0.2).normalize(), b).mul_mat(&w2b(Vec3::x(), a));
        let xform = SpatialTransform {
            rot,
            pos: Vec3::new(0.0, 0.0, 0.01),
        };
        let state = posed_state(&model, xform.pos, xform.rot);
        let reference = -brute_force_lowest(&xform, R, H);
        let Some(d) = deepest(&model, &state, 1.0) else {
            panic!("no contact at orientation {i}");
        };
        let err = d - reference;
        assert!(
            err > -tol,
            "orientation {i}: analytic depth {d:.9} m is {:.2e} m SHALLOWER than \
             the brute-force surface minimum {reference:.9} m — the detector is \
             missing the point that touches",
            -err
        );
        assert!(
            err < tol,
            "orientation {i}: analytic depth {d:.9} m is {err:.2e} m deeper than \
             the surface, i.e. off the shape"
        );
        worst = worst.max(err.abs());
    }
    println!("worst |analytic - brute force| over 200 orientations: {worst:.3e} m (tol {tol:.3e})");
}

// -------------------------------------------------------------------------
// §4 Standing on a cap
// -------------------------------------------------------------------------

/// Upright, the whole rim touches and the manifold must be a polygon: one
/// point cannot hold a cylinder up, and the old sampler's four rim points were
/// the right answer for exactly this pose. The new path reproduces them.
#[test]
fn upright_cylinder_keeps_its_rim_support_polygon() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let state = posed_state(
        &model,
        Vec3::new(0.0, 0.0, 0.5 * H - 1e-4),
        Mat3::identity(),
    );
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert_eq!(contacts.len(), 4, "an upright cylinder rests on its rim");
    for c in &contacts {
        assert!(
            (c.penetration_depth - 1e-4).abs() < 1e-12,
            "rim depth {:.3e}",
            c.penetration_depth
        );
        let r = (c.contact_point.x.powi(2) + c.contact_point.y.powi(2)).sqrt();
        assert!(
            (r - R).abs() < 1e-12,
            "rim point at radius {r:.6}, expected {R}"
        );
    }
    // The four span the rim rather than bunching: opposite pairs.
    let cx: f64 = contacts.iter().map(|c| c.contact_point.x).sum();
    let cy: f64 = contacts.iter().map(|c| c.contact_point.y).sum();
    assert!(
        cx.abs() < 1e-12 && cy.abs() < 1e-12,
        "the rim manifold is off-centre ({cx:.3e}, {cy:.3e}) — not a support polygon"
    );
}

/// The switch between "lowest line" and "cap rim" must be a rounding event,
/// not a step: sweep the axis through vertical and watch the depth.
#[test]
fn the_upright_degeneracy_is_continuous_in_depth() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let mut prev: Option<(f64, f64)> = None;
    for i in -60..=60 {
        // Down to 1e-12 rad from vertical, well past the 1e-9 branch point.
        let tilt = 1e-12 * (10.0_f64).powf(i as f64 / 10.0);
        let state = posed_state(
            &model,
            Vec3::new(0.0, 0.0, 0.5 * H - 1e-4),
            w2b(Vec3::x(), tilt),
        );
        let d = deepest(&model, &state, 0.0).expect("upright cylinder touches");
        if let Some((pt, pd)) = prev {
            // The lowest point is `c_z − (h/2)cos θ − r sin θ`, so the depth
            // grows at `r` near vertical and at most `r + h/2` anywhere.
            // Plus a few ulps of the depth itself: the branch point is a
            // rounding event, and rounding is what this bound has to allow.
            let bound = (R + 0.5 * H) * (tilt - pt).abs() + 1e-14;
            assert!(
                (d - pd).abs() <= bound,
                "depth jumped {:.3e} m between tilts {pt:.3e} and {tilt:.3e} rad \
                 (bound {bound:.3e})",
                (d - pd).abs()
            );
        }
        prev = Some((tilt, d));
    }
}

// -------------------------------------------------------------------------
// §5 The anchor the adjoint reads
// -------------------------------------------------------------------------

/// Detection reports each contact's world offset, and for a rolling wheel it
/// is exactly `−r·ẑ` — the same shape of answer a sphere gives, which is why
/// the existing `Anchor::Ground` machinery differentiates a wheel without a
/// new case.
#[test]
fn a_rolling_wheels_offset_is_the_spheres_offset() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let lay = w2b(Vec3::x(), std::f64::consts::FRAC_PI_2);
    for i in 0..16 {
        let spin = i as f64 * std::f64::consts::TAU / 16.0;
        let rot = lay.mul_mat(&w2b(Vec3::y(), spin));
        let state = posed_state(&model, Vec3::new(0.0, 0.0, R - 1e-4), rot);
        for (c, off) in find_ground_contacts_model_with_offset(&model, &state, 0.0, 0.0) {
            assert!(
                (off - Vec3::new(0.0, 0.0, -R)).norm() < 1e-12,
                "spin {spin:.3}: offset {off:?}, expected (0, 0, -{R})"
            );
            assert!(c.penetration_depth > 0.0);
        }
    }
}

/// Tilted, the offset picks up the horizontal component the scalar `drop` API
/// cannot carry — and it still points from the cap centre to a point on the
/// rim, i.e. it has length exactly `r`.
#[test]
fn the_tilted_offset_is_a_rim_vector_of_length_r() {
    let model = free_body_model(Geometry::Cylinder {
        radius: R,
        height: H,
    });
    let tilt = 0.4_f64;
    let rot = w2b(Vec3::x(), std::f64::consts::FRAC_PI_2 + tilt);
    let state = posed_state(&model, Vec3::new(0.0, 0.0, R), rot);
    let hits = find_ground_contacts_model_with_offset(&model, &state, 0.0, 0.0);
    assert!(!hits.is_empty());
    for (_, off) in &hits {
        assert!(
            (off.norm() - R).abs() < 1e-12,
            "offset length {:.6}, expected the radius {R}",
            off.norm()
        );
        // û is the steepest downhill direction perpendicular to the axis, so
        // its vertical component is −cos(tilt) for an axis `tilt` off level.
        assert!(
            (off.z + R * tilt.cos()).abs() < 1e-12,
            "offset z {:.6}, expected {:.6}",
            off.z,
            -R * tilt.cos()
        );
    }
}
