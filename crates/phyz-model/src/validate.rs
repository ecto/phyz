//! Model sanity checks.
//!
//! These catch the class of authoring mistake that produces a *plausible*
//! simulation rather than an obviously broken one. The motivating case: a joint
//! whose `range` excludes zero. `Model::default_state()` starts every joint at
//! `q = 0`, so such a model begins already violating its own limit, the limit
//! spring fires at full strength on step one, and the robot catapults. Nothing
//! about that failure points at the model file.

use crate::{JointType, Model};

/// A problem found in a model.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// Which joint or body it concerns.
    pub subject: String,
    /// What is wrong.
    pub message: String,
    /// Whether the model is unusable (`true`) or merely suspect (`false`).
    pub fatal: bool,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.fatal { "error" } else { "warning" };
        write!(f, "{kind}: {}: {}", self.subject, self.message)
    }
}

impl Model {
    /// Check the model for authoring mistakes.
    ///
    /// Returns every issue found, most serious first. An empty result means the
    /// model is self-consistent as far as these checks can tell — not that it
    /// is physically sensible.
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        let rest = self.default_state();

        for (ji, joint) in self.joints.iter().enumerate() {
            let name = self
                .bodies
                .iter()
                .find(|b| b.joint_idx == ji)
                .map(|b| b.name.clone())
                .unwrap_or_else(|| format!("joint_{ji}"));

            if let Some([lo, hi]) = joint.limits {
                if lo > hi {
                    issues.push(Issue {
                        subject: name.clone(),
                        message: format!("limit range is inverted: [{lo}, {hi}]"),
                        fatal: true,
                    });
                } else if joint.ndof() == 1 {
                    let q0 = rest.q[self.q_offsets[ji]];
                    if q0 < lo || q0 > hi {
                        issues.push(Issue {
                            subject: name.clone(),
                            message: format!(
                                "rest pose q = {q0} is outside the limit range [{lo}, {hi}]; \
                                 the model starts in violation and will be flung by the limit \
                                 force on the first step"
                            ),
                            fatal: true,
                        });
                    }
                }
            }

            if joint.damping < 0.0 {
                issues.push(Issue {
                    subject: name.clone(),
                    message: format!("negative damping ({}) injects energy", joint.damping),
                    fatal: true,
                });
            }

            if matches!(
                joint.joint_type,
                JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide
            ) {
                let a = joint.axis;
                let n = (a.x * a.x + a.y * a.y + a.z * a.z).sqrt();
                if n < 1e-9 {
                    issues.push(Issue {
                        subject: name.clone(),
                        message: "joint axis is zero-length".to_string(),
                        fatal: true,
                    });
                }
            }
        }

        for body in &self.bodies {
            if body.inertia.mass <= 0.0 {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: format!("non-positive mass ({})", body.inertia.mass),
                    fatal: true,
                });
            }
            for k in 0..3 {
                if body.inertia.inertia[(k, k)] < 0.0 {
                    issues.push(Issue {
                        subject: body.name.clone(),
                        message: "negative inertia on the diagonal".to_string(),
                        fatal: true,
                    });
                    break;
                }
            }
        }

        issues.extend(self.material_issues());

        if self.dt <= 0.0 {
            issues.push(Issue {
                subject: "model".to_string(),
                message: format!("non-positive timestep ({})", self.dt),
                fatal: true,
            });
        }

        issues.sort_by_key(|i| !i.fatal);
        issues
    }

    /// Contact-material authoring mistakes.
    ///
    /// Split out from [`Model::validate`] (which calls it) so a caller that
    /// only just attached materials can check them without re-walking the
    /// whole model.
    ///
    /// The physically-impossible values are fatal. The rest are warnings,
    /// because each has a legitimate use and only the author knows which case
    /// they are in — a validator that cannot tell a mistake from a choice
    /// should say so rather than pick.
    fn material_issues(&self) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Would-be grippiest body, used for the reach warning below.
        let max_friction = self
            .bodies
            .iter()
            .filter_map(|b| b.material.as_ref())
            .map(|m| m.friction)
            .fold(f64::NEG_INFINITY, f64::max);

        for body in &self.bodies {
            let Some(m) = &body.material else { continue };

            if m.friction < 0.0 {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: format!(
                        "negative friction ({}); the Coulomb cone would invert and \
                         friction would drive motion instead of resisting it",
                        m.friction
                    ),
                    fatal: true,
                });
            }
            if !(0.0..=1.0).contains(&m.restitution) {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: format!(
                        "restitution {} is outside [0, 1]; above 1 an impact returns more \
                         energy than it received",
                        m.restitution
                    ),
                    fatal: true,
                });
            }
            if m.margin < 0.0 {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: format!("negative contact margin ({})", m.margin),
                    fatal: true,
                });
            }

            // Dead material: set, but on a body the contact pipeline never
            // looks at. Silent today — the value is simply never read — and
            // the usual cause is naming the wrong link of a chain, e.g. an
            // ankle rather than the foot that carries the collision shape.
            if body.geometry.is_none() && body.collisions.is_empty() {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: "has a contact material but no collision geometry, so the \
                              material is never read; did you mean a child body?"
                        .to_string(),
                    fatal: false,
                });
            }

            // The `max`-friction reach warning. A body's material applies to
            // EVERY contact it takes part in, and friction combines by `max`,
            // so the grippiest body in the model grips whatever it touches —
            // including the ground. This is the skateboard deck: grip tape put
            // on the deck instead of the shoes also grabbed the road and
            // braked the board.
            //
            // Only the single grippiest body is flagged, and only when it is
            // grippy enough to be a deliberate choice rather than a rounding
            // of the default. Flagging every above-default body would fire on
            // every correctly-authored robot foot.
            let broad = body.collisions.len() > 1;
            if m.friction >= 1.0 && m.friction >= max_friction && broad {
                issues.push(Issue {
                    subject: body.name.clone(),
                    message: format!(
                        "friction {} is the highest in the model and this body carries {} \
                         collision shapes; because friction combines by max, all of them \
                         grip — including against the ground. If only one surface should \
                         be grippy, put the material on the body that touches it",
                        m.friction,
                        body.collisions.len()
                    ),
                    fatal: false,
                });
            }
        }

        issues
    }

    /// Issues that make the model unusable.
    pub fn fatal_issues(&self) -> Vec<Issue> {
        self.validate().into_iter().filter(|i| i.fatal).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelBuilder;
    use phyz_math::{SpatialInertia, SpatialTransform, Vec3};

    fn one_hinge() -> Model {
        ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build()
    }

    #[test]
    fn a_clean_model_has_no_issues() {
        assert!(one_hinge().validate().is_empty());
    }

    /// The bug this module exists for.
    #[test]
    fn rest_pose_outside_the_limit_range_is_fatal() {
        let mut m = one_hinge();
        m.joints[0].limits = Some([0.5, 1.0]); // q = 0 is outside
        let issues = m.fatal_issues();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(issues[0].message.contains("outside the limit range"));
    }

    #[test]
    fn rest_pose_on_the_boundary_is_accepted() {
        let mut m = one_hinge();
        m.joints[0].limits = Some([0.0, 1.0]);
        assert!(m.validate().is_empty());
    }

    #[test]
    fn inverted_range_is_fatal() {
        let mut m = one_hinge();
        m.joints[0].limits = Some([1.0, -1.0]);
        assert!(m.fatal_issues()[0].message.contains("inverted"));
    }

    #[test]
    fn zero_mass_is_fatal() {
        let mut m = one_hinge();
        m.bodies[0].inertia.mass = 0.0;
        assert!(m.fatal_issues()[0].message.contains("non-positive mass"));
    }

    #[test]
    fn negative_damping_is_fatal() {
        let mut m = one_hinge();
        m.joints[0].damping = -1.0;
        assert!(m.fatal_issues()[0].message.contains("negative damping"));
    }

    // -----------------------------------------------------------------------
    // Contact materials
    // -----------------------------------------------------------------------

    fn with_geometry(mut m: Model) -> Model {
        for b in &mut m.bodies {
            b.geometry = Some(crate::Geometry::Sphere { radius: 0.1 });
            b.collisions.push(crate::GeomInstance::new(
                crate::Geometry::Sphere { radius: 0.1 },
                SpatialTransform::identity(),
            ));
        }
        m
    }

    fn subjects_mentioning(issues: &[Issue], needle: &str) -> Vec<String> {
        issues
            .iter()
            .filter(|i| i.message.contains(needle))
            .map(|i| i.subject.clone())
            .collect()
    }

    #[test]
    fn a_model_with_no_materials_reports_nothing_new() {
        let m = with_geometry(one_hinge());
        assert!(m.bodies.iter().all(|b| b.material.is_none()));
        assert!(m.material_issues().is_empty());
    }

    #[test]
    fn physically_impossible_material_values_are_fatal() {
        let mut m = with_geometry(one_hinge());
        m.bodies[0].material = Some(crate::ContactMaterial {
            friction: -0.5,
            restitution: 1.4,
            margin: -1e-3,
            ..Default::default()
        });
        let issues = m.material_issues();
        assert_eq!(
            issues.iter().filter(|i| i.fatal).count(),
            3,
            "negative friction, restitution > 1 and negative margin are each fatal: {issues:?}"
        );
        assert!(!subjects_mentioning(&issues, "negative friction").is_empty());
        assert!(!subjects_mentioning(&issues, "outside [0, 1]").is_empty());
    }

    /// The silent one: a material on a body the contact pipeline never looks
    /// at is simply never read, and the usual cause is naming the wrong link.
    #[test]
    fn a_material_on_a_geometryless_body_is_flagged() {
        let mut m = one_hinge();
        m.bodies[0].material = Some(crate::ContactMaterial::default());
        let issues = m.material_issues();
        let dead = subjects_mentioning(&issues, "never read");
        assert_eq!(dead.len(), 1, "{issues:?}");
        assert!(issues.iter().all(|i| !i.fatal), "a warning, not fatal");

        // Give it a shape and the warning goes away.
        let fixed = with_geometry(m);
        assert!(subjects_mentioning(&fixed.material_issues(), "never read").is_empty());
    }

    /// The deck-vs-shoe warning: the grippiest body in the model, when it
    /// carries several collision shapes, grips through all of them.
    #[test]
    fn the_grippiest_multi_shape_body_gets_the_max_combine_warning() {
        let mut m = with_geometry(one_hinge());
        // A second shape on body 0 — a deck with a tail as well as a top.
        m.bodies[0].collisions.push(crate::GeomInstance::new(
            crate::Geometry::Sphere { radius: 0.1 },
            SpatialTransform::identity(),
        ));
        m.bodies[0].material = Some(crate::ContactMaterial {
            friction: 1.5,
            ..Default::default()
        });
        let warned = subjects_mentioning(&m.material_issues(), "combines by max");
        assert_eq!(warned, vec![m.bodies[0].name.clone()], "{warned:?}");

        // The same friction on a body with a single shape is the ordinary,
        // correct case — one surface, one material — and must stay quiet.
        let mut single = with_geometry(one_hinge());
        single.bodies[0].material = Some(crate::ContactMaterial {
            friction: 1.5,
            ..Default::default()
        });
        assert!(
            subjects_mentioning(&single.material_issues(), "combines by max").is_empty(),
            "a single-shape grippy body is the normal case and must not warn"
        );

        // Nor must a below-1.0 friction, which is not a deliberate grip.
        let mut mild = with_geometry(one_hinge());
        mild.bodies[0].collisions.push(crate::GeomInstance::new(
            crate::Geometry::Sphere { radius: 0.1 },
            SpatialTransform::identity(),
        ));
        mild.bodies[0].material = Some(crate::ContactMaterial {
            friction: 0.7,
            ..Default::default()
        });
        assert!(subjects_mentioning(&mild.material_issues(), "combines by max").is_empty());
    }
}
