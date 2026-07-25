//! Errors produced while importing URDF.

use thiserror::Error;

/// Errors that can occur while converting a URDF into a phyz `Model`.
#[derive(Debug, Error)]
pub enum UrdfError {
    /// The file could not be read or the XML could not be parsed.
    #[error("failed to parse URDF: {0}")]
    Parse(String),

    /// The robot has no links at all.
    #[error("URDF `{robot}` contains no links")]
    NoLinks { robot: String },

    /// A joint refers to a link that is not declared.
    #[error("joint `{joint}` references unknown link `{link}`")]
    UnknownLink { joint: String, link: String },

    /// A link is named as the child of more than one joint.
    #[error("link `{link}` is the child of multiple joints (`{first}` and `{second}`)")]
    DuplicateChild {
        link: String,
        first: String,
        second: String,
    },

    /// Two links share a name.
    #[error("duplicate link name `{0}`")]
    DuplicateLink(String),

    /// The link graph is not a tree rooted at a single base link.
    #[error("URDF is not a single kinematic tree: found {0} root links ({1})")]
    MultipleRoots(usize, String),

    /// The link graph contains a cycle, so no valid tree ordering exists.
    #[error("URDF link graph contains a cycle; {0} links are unreachable from the root")]
    Cycle(usize),

    /// A joint type that has no phyz equivalent.
    #[error("joint `{joint}` has unsupported type `{joint_type}`")]
    UnsupportedJointType { joint: String, joint_type: String },

    /// A joint axis was given as the zero vector.
    #[error("joint `{joint}` has a degenerate (zero-length) axis")]
    DegenerateAxis { joint: String },
}

/// Result alias for URDF import.
pub type Result<T> = std::result::Result<T, UrdfError>;
