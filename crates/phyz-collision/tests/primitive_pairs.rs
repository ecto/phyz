//! Narrow-phase correctness across the full matrix of supported primitives.
//!
//! Every pair is checked in three configurations — overlapping, exactly
//! touching, and separated — against *analytic* separations and normals, not
//! merely for a non-empty result. A contact with the wrong normal is worse
//! than no contact at all, so the normal and depth are asserted, not the
//! contact count.

use phyz_collision::{Geometry, contact_manifold, gjk_distance_rot};
use phyz_math::{Mat3, Vec3};

const HALF: f64 = 0.2;

/// Every primitive, sized so its extent along the world +z axis is `HALF`.
/// Placing two of them `HALF + HALF + gap` apart therefore leaves exactly
/// `gap` of separation along z.
fn primitives() -> Vec<(&'static str, Geometry)> {
    vec![
        (
            "box",
            Geometry::Box {
                half_extents: Vec3::new(HALF, HALF, HALF),
            },
        ),
        ("sphere", Geometry::Sphere { radius: HALF }),
        (
            "capsule",
            Geometry::Capsule {
                radius: HALF * 0.5,
                length: HALF,
            },
        ),
        (
            "cylinder",
            Geometry::Cylinder {
                radius: HALF,
                height: 2.0 * HALF,
            },
        ),
    ]
}

/// Signed distance must match the analytic gap for every ordered pair, in all
/// three regimes. This is the assertion that used to fail: GJK returned
/// `dir.norm()` — the magnitude of an unnormalized triple cross product — as
/// the separation, so anything whose terminating simplex was larger than a
/// single point (i.e. everything but sphere/sphere) reported a distance with
/// no geometric meaning (box/cylinder at a true 0.1 m gap reported 0.3265).
#[test]
fn signed_distance_matches_analytic_gap_for_all_pairs() {
    let r = Mat3::identity();
    for (na, ga) in primitives() {
        for (nb, gb) in primitives() {
            for gap in [-0.1, 0.0, 0.1] {
                let pa = Vec3::zeros();
                let pb = Vec3::new(0.0, 0.0, 2.0 * HALF + gap);
                let d = gjk_distance_rot(&ga, &gb, &pa, &pb, &r, &r);
                assert!(
                    (d - gap).abs() < 1e-6,
                    "{na}/{nb} at gap {gap}: expected signed distance {gap}, got {d}",
                );
            }
        }
    }
}

/// Overlapping pairs must produce a manifold whose normal is the true
/// separating direction (+z here) and whose depth is the true overlap.
#[test]
fn overlapping_pairs_have_analytic_normal_and_depth() {
    let r = Mat3::identity();
    let overlap = 0.1;
    for (na, ga) in primitives() {
        for (nb, gb) in primitives() {
            let pa = Vec3::zeros();
            let pb = Vec3::new(0.0, 0.0, 2.0 * HALF - overlap);
            let m = contact_manifold(&ga, &gb, &pa, &r, &pb, &r).unwrap_or_else(|| {
                panic!("{na}/{nb} overlapping by {overlap} produced no contact")
            });
            // 5e-3 rather than machine precision: for two curved surfaces EPA
            // refines a polytope over a sphere, and its normal converges only
            // to the resolution of that polytope. The depth, which is a
            // face-plane distance, is exact to 1e-6.
            assert!(
                (m.normal - Vec3::z()).norm() < 5e-3,
                "{na}/{nb}: normal should be +z, got {:?}",
                m.normal,
            );
            assert!(!m.points.is_empty());
            for p in &m.points {
                assert!(
                    (p.depth - overlap).abs() < 1e-6,
                    "{na}/{nb}: depth should be {overlap}, got {}",
                    p.depth,
                );
            }
        }
    }
}

/// Separated pairs must produce no manifold at all.
#[test]
fn separated_pairs_have_no_manifold() {
    let r = Mat3::identity();
    for (na, ga) in primitives() {
        for (nb, gb) in primitives() {
            let pa = Vec3::zeros();
            let pb = Vec3::new(0.0, 0.0, 2.0 * HALF + 0.1);
            assert!(
                contact_manifold(&ga, &gb, &pa, &r, &pb, &r).is_none(),
                "{na}/{nb} separated by 0.1 reported a contact",
            );
        }
    }
}

/// A half-space plane against each primitive. Previously the plane's support
/// function described a two-point set on the normal axis rather than a
/// half-space, so *every* plane query — overlapping, touching or separated —
/// reported penetration, with normals pointing sideways.
#[test]
fn plane_against_each_primitive() {
    let r = Mat3::identity();
    let plane = Geometry::Plane { normal: Vec3::z() };
    for (nb, gb) in primitives() {
        for gap in [-0.1, 0.0, 0.1] {
            let pa = Vec3::zeros();
            let pb = Vec3::new(0.0, 0.0, HALF + gap);
            let d = gjk_distance_rot(&plane, &gb, &pa, &pb, &r, &r);
            assert!(
                (d - gap).abs() < 1e-6,
                "plane/{nb} at gap {gap}: expected {gap}, got {d}",
            );
        }

        // Overlapping: normal +z, depth equal to the overlap.
        let pb = Vec3::new(0.0, 0.0, HALF - 0.1);
        let m = contact_manifold(&plane, &gb, &Vec3::zeros(), &r, &pb, &r)
            .unwrap_or_else(|| panic!("plane/{nb} overlapping produced no contact"));
        assert!(
            (m.normal - Vec3::z()).norm() < 1e-4,
            "plane/{nb}: normal should be +z, got {:?}",
            m.normal,
        );
        for p in &m.points {
            assert!(
                (p.depth - 0.1).abs() < 1e-6,
                "plane/{nb}: depth should be 0.1, got {}",
                p.depth,
            );
        }

        // Separated: nothing.
        let pb = Vec3::new(0.0, 0.0, HALF + 0.1);
        assert!(
            contact_manifold(&plane, &gb, &Vec3::zeros(), &r, &pb, &r).is_none(),
            "plane/{nb} separated by 0.1 reported a contact",
        );
    }
}

/// A box resting on a plane must give a four-point face manifold, which is
/// what keeps a stack from tipping.
#[test]
fn box_on_plane_is_a_four_point_face() {
    let r = Mat3::identity();
    let plane = Geometry::Plane { normal: Vec3::z() };
    let b = Geometry::Box {
        half_extents: Vec3::new(0.5, 0.5, 0.5),
    };
    let m = contact_manifold(
        &plane,
        &b,
        &Vec3::zeros(),
        &r,
        &Vec3::new(0.0, 0.0, 0.49),
        &r,
    )
    .expect("box overlapping the ground plane");
    assert_eq!(m.points.len(), 4, "expected a 4-point face manifold");
    assert!((m.normal - Vec3::z()).norm() < 1e-6);
    for p in &m.points {
        assert!(
            (p.depth - 0.01).abs() < 1e-9,
            "each corner overlaps by 0.01, got {}",
            p.depth,
        );
    }
}

/// The plane normal lives in the body frame and must be rotated into the
/// world. Under a 90° rotation about +x the body-frame +z normal becomes the
/// world +y normal, and a sphere offset along +y overlaps the half-space.
#[test]
fn plane_normal_is_rotated_into_the_world() {
    // R_x(90°): rot * (0,0,1) = (0,1,0).
    let rot = Mat3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    let plane = Geometry::Plane { normal: Vec3::z() };
    let world_n = rot * Vec3::z();
    let s = Geometry::Sphere { radius: 0.2 };
    // Sphere centre 0.1 m along the world normal from the plane: it sticks out
    // by 0.1 and overlaps by 0.1.
    let pb = world_n * 0.1;
    let d = gjk_distance_rot(&plane, &s, &Vec3::zeros(), &pb, &rot, &Mat3::identity());
    assert!(
        (d + 0.1).abs() < 1e-6,
        "rotated plane vs sphere: expected -0.1, got {d}",
    );

    let m = contact_manifold(&plane, &s, &Vec3::zeros(), &rot, &pb, &Mat3::identity())
        .expect("rotated plane overlaps sphere");
    assert!(
        (m.normal - world_n).norm() < 1e-4,
        "normal should be the rotated plane normal {world_n:?}, got {:?}",
        m.normal,
    );
}

/// Total containment: a small sphere entirely inside a large box. The minimum
/// translation that separates them is `box_half + sphere_radius` along the
/// nearest face normal — GJK's origin-enclosure logic must not mistake the
/// deeply-enclosed origin for separation.
#[test]
fn sphere_fully_inside_box_reports_containment_depth() {
    let r = Mat3::identity();
    let b = Geometry::Box {
        half_extents: Vec3::new(1.0, 1.0, 0.3),
    };
    let s = Geometry::Sphere { radius: 0.1 };
    // Concentric: the nearest exit is through a ±z face, 0.3 + 0.1 away.
    let d = gjk_distance_rot(&b, &s, &Vec3::zeros(), &Vec3::zeros(), &r, &r);
    assert!(
        (d + 0.4).abs() < 1e-5,
        "expected containment depth -0.4, got {d}",
    );

    let m = contact_manifold(&b, &s, &Vec3::zeros(), &r, &Vec3::zeros(), &r)
        .expect("containment is a contact");
    assert!(
        m.normal.z.abs() > 0.999,
        "escape direction should be ±z, got {:?}",
        m.normal,
    );
    assert!((m.points[0].depth - 0.4).abs() < 1e-5);
}

/// The exact scenario from `phyz/tests/contact_stability.rs`: a 0.1 m sphere
/// whose centre sits 0.05 m inside a (0.5, 0.5, 0.1) box, on the +z side.
///
/// Analytic answer: the box's top face is at z = 0.1 and the sphere's lowest
/// point is at z = -0.05, so the +z escape distance is 0.15 — smaller than the
/// -z escape (0.25) or either lateral escape (0.6), so the minimum-translation
/// normal is exactly +z with depth 0.15.
#[test]
fn contact_stability_box_sphere_case() {
    let r = Mat3::identity();
    let b = Geometry::Box {
        half_extents: Vec3::new(0.5, 0.5, 0.1),
    };
    let s = Geometry::Sphere { radius: 0.1 };
    let pos_b = Vec3::new(0.0, 0.0, 0.05);

    let d = gjk_distance_rot(&b, &s, &Vec3::zeros(), &pos_b, &r, &r);
    assert!(d < 0.0, "overlapping box/sphere must report penetration");
    assert!((d + 0.15).abs() < 1e-5, "expected depth 0.15, got {}", -d);

    let m = contact_manifold(&b, &s, &Vec3::zeros(), &r, &pos_b, &r)
        .expect("overlapping box/sphere must produce a manifold");
    assert!(
        (m.normal - Vec3::z()).norm() < 1e-4,
        "normal should be +z (push the sphere out through the top face), got {:?}",
        m.normal,
    );
    assert_eq!(m.points.len(), 1, "a sphere touches at one point");
    assert!((m.points[0].depth - 0.15).abs() < 1e-5);
}
