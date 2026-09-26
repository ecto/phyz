//! Body-body contacts inside the margin band.
//!
//! The ground path has had a margin since the day a K1 foot corner was
//! measured carrying 22.3 N — 11 % of body weight — on the step before it
//! vanished from the contact set. The body-body path had none, for a
//! structural reason: its manifold came from EPA, which is only defined where
//! the shapes overlap, so a separated pair had a distance but no direction and
//! no manifold could be built for it.
//!
//! GJK was already computing that direction. The closest point `v` of the
//! Minkowski difference to the origin is `a* − b*` for the witness pair, so
//! `−v̂` is the separating normal from `a` toward `b` — the same sense EPA
//! reports. It was simply discarded. With it surfaced, one face-clipping path
//! serves both branches.
//!
//! These tests pin the band: what enters it, with what sign, in what shape,
//! and that the transition through zero is continuous rather than a jump.

use phyz_collision::{Geometry, contact_manifold, contact_manifold_within};
use phyz_contact::find_contacts;
use phyz_math::{Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry as ModelGeometry, ModelBuilder, State};

const HALF: f64 = 0.1;
const MARGIN: f64 = 0.01;

fn unit_box() -> Geometry {
    Geometry::Box {
        half_extents: Vec3::new(HALF, HALF, HALF),
    }
}

/// Two boxes separated along `x` by `gap`, both axis-aligned.
fn pair(gap: f64) -> Option<phyz_collision::Manifold> {
    contact_manifold_within(
        &unit_box(),
        &unit_box(),
        &Vec3::zeros(),
        &Mat3::identity(),
        &Vec3::new(2.0 * HALF + gap, 0.0, 0.0),
        &Mat3::identity(),
        MARGIN,
    )
}

// -------------------------------------------------------------------------
// The band itself
// -------------------------------------------------------------------------

/// A pair inside the band reports contacts with a negative depth equal to
/// minus the gap; a pair outside it reports nothing.
#[test]
fn a_separated_pair_inside_the_band_reports_a_negative_depth() {
    let gap = 0.004;
    let m = pair(gap).expect("a pair 4 mm apart is inside a 10 mm band");
    assert!(!m.points.is_empty());
    for p in &m.points {
        assert!(
            (p.depth + gap).abs() < 1e-9,
            "depth {:.6} for a {gap} m gap; it should be −gap",
            p.depth
        );
    }

    // Outside the band: nothing.
    assert!(
        pair(MARGIN + 1e-6).is_none(),
        "a pair beyond the margin still reported a contact"
    );
    // Exactly at the band edge is already excluded — the impedance has
    // tapered to zero by then, so the exclusion is a no-op rather than a step.
    assert!(pair(MARGIN).is_none());
}

/// Zero margin is the old behaviour, exactly: nothing separated is reported.
#[test]
fn zero_margin_reproduces_the_hard_cutoff() {
    for gap in [1e-9, 1e-4, 0.004] {
        assert!(
            contact_manifold_within(
                &unit_box(),
                &unit_box(),
                &Vec3::zeros(),
                &Mat3::identity(),
                &Vec3::new(2.0 * HALF + gap, 0.0, 0.0),
                &Mat3::identity(),
                0.0,
            )
            .is_none(),
            "gap {gap} produced a contact at zero margin"
        );
    }
}

/// A separated pair gets the *same shape* of manifold as a touching one — four
/// clipped face points, not a single witness point. If the manifold changed
/// shape at the crossing, the margin would remove one discontinuity and
/// introduce another.
#[test]
fn the_manifold_does_not_change_shape_across_the_crossing() {
    let separated = pair(0.004).expect("inside the band");
    let overlapping = contact_manifold(
        &unit_box(),
        &unit_box(),
        &Vec3::zeros(),
        &Mat3::identity(),
        &Vec3::new(2.0 * HALF - 0.004, 0.0, 0.0),
        &Mat3::identity(),
    )
    .expect("overlapping");

    assert_eq!(
        separated.points.len(),
        overlapping.points.len(),
        "a face pair should clip to the same number of points either side of \
         contact ({} separated vs {} overlapping)",
        separated.points.len(),
        overlapping.points.len()
    );
    assert_eq!(separated.points.len(), 4, "a box face pair is a quad");
}

/// Depth is continuous through zero: sampling across the crossing must not
/// jump. This is the property the whole exercise buys.
#[test]
fn depth_is_continuous_through_contact() {
    let sample = |gap: f64| -> Option<f64> {
        let m = contact_manifold_within(
            &unit_box(),
            &unit_box(),
            &Vec3::zeros(),
            &Mat3::identity(),
            &Vec3::new(2.0 * HALF + gap, 0.0, 0.0),
            &Mat3::identity(),
            MARGIN,
        )?;
        Some(
            m.points
                .iter()
                .map(|p| p.depth)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    };

    // Walk from a 8 mm gap to 4 mm of overlap.
    let mut prev: Option<(f64, f64)> = None;
    for i in 0..=60 {
        let gap = 0.008 - 0.0002 * i as f64;
        let Some(d) = sample(gap) else { continue };
        if gap > 0.0 {
            assert!(
                (d + gap).abs() < 1e-6,
                "at gap {gap:.4} the depth was {d:.6}, not −gap"
            );
        } else {
            assert!(
                d >= 0.0,
                "at overlap {:.4} the depth was {d:.6}, which is not a \
                 penetration",
                -gap
            );
        }
        // Only the separated side has `depth == −gap` by construction. Past
        // the crossing the depth comes from EPA, and that it still agrees for
        // axis-aligned boxes is a property of this geometry, not of the API —
        // so the equality above is asserted on the separated branch only.
        if let Some((pg, pd)) = prev {
            let step = (d - pd).abs();
            assert!(
                step < 1e-3,
                "depth jumped {step:.2e} between gap {pg:.4} and {gap:.4}"
            );
        }
        prev = Some((gap, d));
    }
    assert!(prev.is_some(), "the sweep never produced a manifold");
}

// -------------------------------------------------------------------------
// The normal, which is the part that was missing
// -------------------------------------------------------------------------

/// The separated normal points from `a` toward `b`, the same sense EPA
/// reports for an overlapping pair. Getting this backwards would drive
/// separated pairs together — the failure mode a body-body contact already
/// had once.
#[test]
fn the_separated_normal_points_from_a_to_b() {
    let m = pair(0.004).expect("inside the band");
    assert!(
        m.normal.x > 0.99,
        "b is at +x of a, so the normal should be +x; got {:?}",
        m.normal
    );

    // And it agrees with the overlapping branch, which comes from EPA.
    let overlapping = contact_manifold(
        &unit_box(),
        &unit_box(),
        &Vec3::zeros(),
        &Mat3::identity(),
        &Vec3::new(2.0 * HALF - 0.004, 0.0, 0.0),
        &Mat3::identity(),
    )
    .expect("overlapping");
    assert!(
        m.normal.dot(overlapping.normal) > 0.99,
        "separated normal {:?} disagrees with the EPA normal {:?}",
        m.normal,
        overlapping.normal
    );
}

/// The normal follows the geometry, not the axis it was first tested on: a
/// pair separated along `z`, and one rotated 45°, must both report the true
/// closest direction.
#[test]
fn the_separated_normal_follows_the_geometry() {
    let along_z = contact_manifold_within(
        &unit_box(),
        &unit_box(),
        &Vec3::zeros(),
        &Mat3::identity(),
        &Vec3::new(0.0, 0.0, 2.0 * HALF + 0.004),
        &Mat3::identity(),
        MARGIN,
    )
    .expect("inside the band");
    assert!(along_z.normal.z > 0.99, "got {:?}", along_z.normal);

    // A sphere offset diagonally: the normal is the centre line.
    let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
    let sphere = Geometry::Sphere { radius: 0.05 };
    let m = contact_manifold_within(
        &sphere,
        &sphere,
        &Vec3::zeros(),
        &Mat3::identity(),
        &(dir * (0.1 + 0.004)),
        &Mat3::identity(),
        MARGIN,
    )
    .expect("spheres 4 mm apart");
    assert!(
        m.normal.dot(dir) > 0.999,
        "normal {:?} is not along the centre line {dir:?}",
        m.normal
    );
    assert!((m.points[0].depth + 0.004).abs() < 1e-9);
}

/// Rotated boxes: the band still works when the closest feature is an edge
/// rather than a face, and the normal still points a → b.
#[test]
fn a_rotated_pair_still_reports_the_band() {
    let rot = Quat::from_axis_angle(Vec3::z(), std::f64::consts::FRAC_PI_4).to_matrix();
    // A 45°-yawed box reaches sqrt(2)*HALF along x from its centre.
    let reach = std::f64::consts::SQRT_2 * HALF;
    let gap = 0.003;
    let m = contact_manifold_within(
        &unit_box(),
        &unit_box(),
        &Vec3::zeros(),
        &Mat3::identity(),
        &Vec3::new(HALF + reach + gap, 0.0, 0.0),
        &rot,
        MARGIN,
    )
    .expect("edge-to-face pair inside the band");
    assert!(m.normal.x > 0.99, "normal {:?}", m.normal);
    let deepest = m
        .points
        .iter()
        .map(|p| p.depth)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (deepest + gap).abs() < 1e-6,
        "deepest depth {deepest:.6} for a {gap} gap"
    );
}

/// A negative or non-finite margin is treated as zero rather than as
/// "everything is in contact".
#[test]
fn a_nonsense_margin_is_clamped() {
    for m in [-1.0, f64::NAN, f64::NEG_INFINITY] {
        assert!(
            contact_manifold_within(
                &unit_box(),
                &unit_box(),
                &Vec3::zeros(),
                &Mat3::identity(),
                &Vec3::new(2.0 * HALF + 0.004, 0.0, 0.0),
                &Mat3::identity(),
                m,
            )
            .is_none(),
            "margin {m} was not clamped"
        );
    }
}

// -------------------------------------------------------------------------
// The broad phase has to agree with the margin
// -------------------------------------------------------------------------

/// The margin must survive the broad phase.
///
/// `find_contacts` culls pairs with `sweep_and_prune` before the narrow phase
/// runs, so a margin applied only in the narrow phase is capped by whatever
/// the broad phase already threw away. An AABB gap never exceeds the true
/// surface gap, so two boxes 4 mm apart have AABBs 4 mm apart and are culled
/// outright — the band would be unreachable for exactly the pairs it exists
/// to catch.
///
/// This is a full-pipeline test on purpose: `contact_manifold_within` was
/// correct the whole time, and every unit test above passed, while
/// `find_contacts` reported band contacts on only 4 of 158 body-body contacts
/// through a humanoid fall.
#[test]
fn the_margin_survives_the_broad_phase() {
    let inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia)
        .add_free_body("b", -1, SpatialTransform::identity(), inertia)
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(ModelGeometry::Box {
            half_extents: Vec3::new(HALF, HALF, HALF),
        });
    }

    let place = |state: &mut State, gap: f64| {
        state.body_xform[0] = SpatialTransform::new(Mat3::identity(), Vec3::zeros());
        state.body_xform[1] =
            SpatialTransform::new(Mat3::identity(), Vec3::new(2.0 * HALF + gap, 0.0, 0.0));
    };

    let mut state = model.default_state();
    place(&mut state, 0.004);

    let with = find_contacts(&model, &state, MARGIN);
    assert!(
        !with.is_empty(),
        "a pair 4 mm apart produced no contact at a 10 mm margin — the broad \
         phase culled it before the narrow phase could apply the band"
    );
    for c in &with {
        assert!(
            (c.penetration_depth + 0.004).abs() < 1e-9,
            "depth {} for a 4 mm gap",
            c.penetration_depth
        );
    }

    // Still nothing outside the band, and nothing at zero margin.
    assert!(find_contacts(&model, &state, 0.0).is_empty());
    place(&mut state, MARGIN * 2.0);
    assert!(find_contacts(&model, &state, MARGIN).is_empty());
}
