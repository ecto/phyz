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
}
