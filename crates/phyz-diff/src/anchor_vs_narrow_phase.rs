//! Is the frozen anchor's geometry derivative the narrow phase's own?
//!
//! [`super::Anchor`] exists because feature selection is combinatorial and a
//! lane that re-ran detection could difference two different contact sets
//! (design doc §4.4). [`super`]'s docs call the price of that an
//! approximation — but it is only an approximation if the anchor's *smooth*
//! motion differs from what the narrow phase would report for the same feature
//! pair. This file measures that difference directly, on the geometry rather
//! than through a trajectory: central differences of the contact point, normal
//! and depth, once through [`super::Anchor::collision`] and once through
//! `find_contacts` itself.
//!
//! The answer has two halves, and they point in opposite directions.
//!
//! # Face-face: the anchor is not an approximation at all
//!
//! `clip_faces` locates a face-vertex contact by a vertex of the incident body
//! and measures it against the plane of a face of the reference body;
//! [`super::Anchor::Pair`] freezes exactly those two objects — the vertex in
//! the incident body's frame, the plane's normal in the reference body's frame
//! — and re-evaluates the same `depth = (p_j − p_i)·n`. The anchor model **is**
//! the narrow phase's own face-vertex formula with the feature choice held
//! fixed. Measured below: all three terms agree to `1e-10`, which is the
//! difference's own round-off floor, not a leading-order match.
//!
//! That was invisible before `fix/clip-faces-manifolds`, because until then
//! 80 % of body-body pairs never reached the face-clip path at all: they fell
//! through to `single_point`, whose witness is a box corner chosen by the sign
//! of a cancelling component, and a narrow phase like that has no derivative to
//! agree with. The residual `1.2e-3` in ipse's 12-step jump window was that
//! fallback rather than this freezing — `probe/gradient-stack`'s
//! `solver_adjoint_probe` drops to `8e-6` once the manifolds are real. Nothing
//! needed to be carried out of the narrow phase to close it.
//!
//! # Edge-face: there is no derivative to carry
//!
//! Roll the block so it rests on the plank along one bottom **edge** and the
//! picture changes completely. The normal is still exactly `−ẑ`, and the
//! contact point and depth still track the anchor to `1e-11` — but the
//! narrow phase's *normal* has a crease there, and its two one-sided slopes
//! are different functions:
//!
//! | lane | anchor | detection, forward | detection, backward |
//! | --- | --- | --- | --- |
//! | plank pitch | `(-1.000, 0, 0)` | `(-1.000, 0, 0)` | `(-0.882, 0.294, 0)` |
//! | block roll | `(0, 0, 0)` | `(-0.115, -0.288, 0)` | `(0, 0, 0)` |
//!
//! An edge lying on a face is a tie: the contact normal is a face normal of
//! the plank, and rolling the block cannot move it — until the roll crosses
//! zero, at which point the *other* edge takes over and the reference feature
//! changes. The support direction sits on a crease of the Minkowski
//! difference, and a central difference across it reports the average of two
//! branches, which is a derivative of nothing.
//!
//! This is what a "tight per-contact finite difference of the real narrow
//! phase" would have installed: `(-0.0576, -0.144, 0)` on the block-roll lane,
//! a number belonging to neither side. The frozen anchor does something
//! strictly better — it reproduces **one of the two branches exactly**, which
//! is what a one-sided derivative is, and matches this module's stated contract
//! for stick/slide and load/unload transitions. `anchor_is_a_one_sided_branch`
//! below pins that, and it is the reason no narrow-phase geometry derivative
//! was carried into `eval_pieces`: on the face manifolds that make up a resting
//! or riding stance there is nothing to carry, and on an edge contact there is
//! nothing to carry it *from*.

use super::*;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform};
use phyz_model::{Geometry, ModelBuilder};

/// Step for every central and one-sided difference here. On a pose whose
/// geometry lives at the `1e-2 m` scale the truncation of a smooth
/// face-vertex formula is `~1e-12` relative and the round-off floor `~1e-10`.
const H: f64 = 1.0e-6;

/// The floor those differences reach, with two decades of headroom: a term the
/// anchor modelled with the wrong feature — the wrong vertex, the wrong plane,
/// the wrong body's frame — would be `O(1)` against this, not `O(1e-10)`.
const FD_FLOOR: f64 = 1.0e-8;

const TERM: [&str; 3] = ["contact point", "normal", "depth"];

fn inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let c = mass / 3.0;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            c * (h.y * h.y + h.z * h.z),
            c * (h.x * h.x + h.z * h.z),
            c * (h.x * h.x + h.y * h.y),
        )),
    )
}

/// A plank and a block, both free boxes — the same reduction
/// `tests/body_body_adjoint.rs` uses for the foot-on-grip-tape case.
fn stack() -> (Model, Vec3, Vec3) {
    let plank = Vec3::new(0.30, 0.12, 0.02);
    let block = Vec3::new(0.04, 0.04, 0.04);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "plank",
            -1,
            SpatialTransform::identity(),
            inertia(2.0, plank),
        )
        .add_free_body(
            "block",
            -1,
            SpatialTransform::identity(),
            inertia(1.0, block),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: plank,
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: block,
    });
    (model, plank, block)
}

/// Block flat on the plank, yawed against it by `0.11 rad` so the clip is
/// oblique — every manifold point is an edge crossing rather than a corner
/// resting on a corner. Roll and pitch stay small enough that the plank's top
/// plane is within `margin` across the block's whole 80 mm footprint, so all
/// four clipped points survive detection.
///
/// This is the resting-manifold case: a foot on grip tape, a wheel on a deck,
/// the 15-contact skate stance ipse's jump window is made of.
fn face_face_q(plank: Vec3, block: Vec3) -> DVec {
    // Free-joint layout is `[rot(3), pos(3)]` per body.
    let mut q = DVec::zeros(12);
    q[0] = 4.0e-3;
    q[1] = 2.0e-3;
    q[2] = 0.11;
    q[5] = plank.z;
    q[9] = 0.02;
    q[11] = 2.0 * plank.z + block.z - 2.0e-4;
    q
}

/// Block rolled `0.35 rad` and lowered onto the flat plank, so it touches
/// along one bottom edge: the separating normal is a face normal of the
/// *plank* while the block's own support face is tilted 20 degrees away from
/// it, and the manifold is a two-point segment rather than a quadrilateral.
fn edge_face_q(plank: Vec3, block: Vec3) -> DVec {
    let roll: f64 = 0.35;
    let mut q = DVec::zeros(12);
    q[5] = plank.z;
    q[6] = roll;
    q[9] = 0.02;
    // Height of the lowest rolled edge above the block's centre.
    q[11] = 2.0 * plank.z + block.y * roll.sin().abs() + block.z * roll.cos().abs() - 2.0e-4;
    q
}

/// FK a configuration into a state with body transforms filled in.
fn at(model: &Model, q: &DVec) -> State {
    let mut s = model.default_state();
    s.q = q.clone();
    let (xf, _) = forward_kinematics(model, &s);
    s.body_xform = xf;
    s
}

/// Line up a perturbed contact set against the nominal one, returning
/// `perturbed[match_of[c]]` for each nominal `c`.
///
/// Matching by array index does not work, and finding that out is a third of
/// what this file measured. `clip_faces` emits the clipped polygon in whatever
/// order the clip walk produced, and that order flips under arbitrarily small
/// pose changes: on the edge-face case a `1e-6 rad` pitch of the block
/// exchanges the two segment endpoints in the output array while moving each of
/// them by `5e-8 m`. Differenced by index that reads as a contact point moving
/// `8e-2 m` per `1e-6 rad` — a derivative of `4e4`, an artefact of the
/// instrument and not a property of the geometry. It is worth knowing that the
/// order is not stable; it is not worth mistaking for physics.
///
/// So each nominal contact is matched to the nearest perturbed contact of the
/// same body pair. `None` if that is not a bijection or if any match is farther
/// than `tol` — either means the feature set genuinely changed.
fn match_contacts(nominal: &[Collision], perturbed: &[Collision], tol: f64) -> Option<Vec<usize>> {
    if nominal.len() != perturbed.len() {
        return None;
    }
    let mut taken = vec![false; perturbed.len()];
    let mut out = Vec::with_capacity(nominal.len());
    for c in nominal {
        let (best, dist) = perturbed
            .iter()
            .enumerate()
            .filter(|(k, p)| !taken[*k] && p.body_i == c.body_i && p.body_j == c.body_j)
            .map(|(k, p)| (k, (p.contact_point - c.contact_point).norm()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
        if dist > tol {
            return None;
        }
        taken[best] = true;
        out.push(best);
    }
    Some(out)
}

/// One lane's worth of both models, matched contact by contact: the anchor's
/// contact set at `q ± h e_j` and the narrow phase's, already reordered onto
/// the nominal indexing.
struct Lane {
    anchor_p: Vec<Collision>,
    anchor_m: Vec<Collision>,
    detect_p: Vec<Collision>,
    detect_m: Vec<Collision>,
}

/// Evaluate both models along lane `j`, or panic if the narrow phase changed
/// its feature set — the witness-tie discontinuity, where there is no
/// derivative for the anchor to be judged against, and the one outcome that
/// would redirect this whole line of work.
fn lane(
    model: &Model,
    q: &DVec,
    margin: f64,
    anchors: &[Anchor],
    nominal: &[Collision],
    j: usize,
) -> Lane {
    let ground = 0.0;
    let mut qp = q.clone();
    let mut qm = q.clone();
    qp[j] += H;
    qm[j] -= H;
    let (sp, sm) = (at(model, &qp), at(model, &qm));

    let dp = find_contacts(model, &sp, margin);
    let dm = find_contacts(model, &sm, margin);
    // A contact may move by far more than the `H`-scale motion that produced
    // it and still be the same contact; what it may not do is move a body
    // length, which is what a witness landing on a different feature looks
    // like.
    let tol = 1.0e-3;
    let (mp, mm) = match (
        match_contacts(nominal, &dp, tol),
        match_contacts(nominal, &dm, tol),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => panic!(
            "lane {j} changed the contact set under a {H:e} perturbation \
             ({} nominal, {} at +h, {} at -h) — the narrow phase has no \
             derivative there and the anchor cannot be judged against it",
            nominal.len(),
            dp.len(),
            dm.len()
        ),
    };
    Lane {
        anchor_p: anchors.iter().map(|a| a.collision(&sp, ground)).collect(),
        anchor_m: anchors.iter().map(|a| a.collision(&sm, ground)).collect(),
        detect_p: mp.iter().map(|&k| dp[k].clone()).collect(),
        detect_m: mm.iter().map(|&k| dm[k].clone()).collect(),
    }
}

/// The three compared quantities of one contact, each divided by its own
/// scale so the numbers are comparable: the contact point and the depth are
/// lengths on the bodies' `1e-1 m` scale, the normal is a unit vector.
fn terms(c: &Collision) -> [Vec3; 3] {
    [
        c.contact_point / 0.1,
        c.contact_normal,
        Vec3::new(c.penetration_depth / 0.1, 0.0, 0.0),
    ]
}

/// Set up a scene: model, nominal contacts, and their anchors.
fn scene(q: &DVec, model: &Model, margin: f64, want: usize) -> (Vec<Collision>, Vec<Anchor>) {
    let nominal = at(model, q);
    let contacts = find_contacts(model, &nominal, margin);
    assert_eq!(
        contacts.len(),
        want,
        "the scene did not produce the manifold this case is about"
    );
    let anchors = contacts
        .iter()
        // Body-body contacts: the ground-support kind is unread on this
        // branch, and `Material` is what detection reports for them.
        .map(|c| Anchor::of(c, GroundSupport::Material, &nominal, 0.0))
        .collect();
    (contacts, anchors)
}

/// **The closure.** On a face manifold the anchor's geometry derivative *is*
/// the narrow phase's, in all three terms, to the finite difference's own
/// floor.
///
/// This is the whole of what the frozen anchor was suspected of costing on
/// ipse's jump window, measured on the geometry where the suspicion could be
/// settled instead of inferred from a trajectory.
#[test]
fn face_manifold_derivative_matches_detection() {
    let (model, plank, block) = stack();
    let q = face_face_q(plank, block);
    let margin = 1.0e-3;
    // A quadrilateral: the block's four bottom corners clipped to the plank.
    let (nominal, anchors) = scene(&q, &model, margin, 4);

    let mut worst = [0.0_f64; 3];
    for j in 0..model.nv {
        let l = lane(&model, &q, margin, &anchors, &nominal, j);
        for c in 0..nominal.len() {
            let a = terms(&l.anchor_p[c]);
            let b = terms(&l.anchor_m[c]);
            let dpc = terms(&l.detect_p[c]);
            let dmc = terms(&l.detect_m[c]);
            for t in 0..3 {
                let anchor_d = (a[t] - b[t]) / (2.0 * H);
                let detect_d = (dpc[t] - dmc[t]) / (2.0 * H);
                worst[t] = worst[t].max((anchor_d - detect_d).norm());
            }
        }
    }

    for t in 0..3 {
        println!(
            "  face-face: d({})/dq anchor vs detection {:.3e}",
            TERM[t], worst[t]
        );
    }
    for t in 0..3 {
        assert!(
            worst[t] < FD_FLOOR,
            "face-face: d({})/dq disagrees with the narrow phase by {:.3e} — the \
             frozen anchor is not re-evaluating the feature detection chose",
            TERM[t],
            worst[t]
        );
    }
}

/// **The limit, stated exactly.** On an edge-face contact the narrow phase's
/// normal has a crease, and the anchor reproduces one of its two one-sided
/// branches exactly.
///
/// Two claims, and the second is only interesting because of the first:
///
/// 1. A crease is really there — some lane's forward and backward slopes of
///    the *detected* normal differ by an `O(1)` amount, so no two-sided
///    derivative exists and a central difference of the narrow phase is the
///    average of two different functions.
/// 2. On **every** lane the anchor's slope equals the forward slope or the
///    backward slope to the difference's own floor. That is what a one-sided
///    derivative is, and it is what this module promises everywhere else it
///    meets a non-smooth transition.
///
/// Together they say the thing worth saying about carrying `d(normal)/d(pose)`
/// out of the narrow phase here: there is no derivative there to carry, and
/// finite-differencing one would install a number belonging to neither branch.
#[test]
fn anchor_is_a_one_sided_branch_at_an_edge_contact() {
    let (model, plank, block) = stack();
    let q = edge_face_q(plank, block);
    let margin = 1.0e-3;
    // A segment: the block touches the plank along one edge, two points.
    let (nominal, anchors) = scene(&q, &model, margin, 2);
    let nom_terms: Vec<[Vec3; 3]> = nominal.iter().map(terms).collect();

    // A one-sided difference carries `O(h)` truncation where the central one
    // carries `O(h^2)`, so this comparison cannot reach `FD_FLOOR` and must not
    // be asked to. `1e-4` is two decades above the `~1e-6` truncation these
    // slopes actually show and three decades below the `O(0.1)` crease being
    // separated from — it distinguishes "on a branch" from "between branches"
    // with room to spare, which is all it is for.
    const BRANCH_FLOOR: f64 = 1.0e-4;

    let mut crease = 0.0_f64;
    let mut worst_branch = [0.0_f64; 3];
    for j in 0..model.nv {
        let l = lane(&model, &q, margin, &anchors, &nominal, j);
        for (c, nom) in nom_terms.iter().enumerate() {
            let a = terms(&l.anchor_p[c]);
            let b = terms(&l.anchor_m[c]);
            let dpc = terms(&l.detect_p[c]);
            let dmc = terms(&l.detect_m[c]);
            for t in 0..3 {
                // The anchor is smooth, so its central difference is its
                // derivative outright — no branch to choose.
                let anchor_d = (a[t] - b[t]) / (2.0 * H);
                // The narrow phase's are one-sided about the nominal pose.
                let d_fwd = (dpc[t] - nom[t]) / H;
                let d_bwd = (nom[t] - dmc[t]) / H;
                if t == 1 {
                    crease = crease.max((d_fwd - d_bwd).norm());
                }
                let miss = (anchor_d - d_fwd).norm().min((anchor_d - d_bwd).norm());
                worst_branch[t] = worst_branch[t].max(miss);
            }
        }
    }

    println!("  edge-face: detected normal, forward vs backward slope {crease:.3e}");
    for t in 0..3 {
        println!(
            "  edge-face: d({})/dq anchor vs nearest detected branch {:.3e}",
            TERM[t], worst_branch[t]
        );
    }

    assert!(
        crease > 0.1,
        "this case exists to exhibit a crease in the detected normal and there \
         is none: forward and backward slopes differ by only {crease:.3e}, so \
         either the scene stopped being edge-on or the narrow phase became \
         smooth here"
    );
    for t in 0..3 {
        assert!(
            worst_branch[t] < BRANCH_FLOOR,
            "edge-face: d({})/dq is {:.3e} from *either* one-sided branch of the \
             narrow phase — the anchor is not a one-sided derivative of it, which \
             is a stronger failure than the crease itself",
            TERM[t],
            worst_branch[t]
        );
    }
}
