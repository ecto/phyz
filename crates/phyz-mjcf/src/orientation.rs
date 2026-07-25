//! MJCF orientation attributes: `quat`, `euler`, `axisangle`, `xyaxes`, `zaxis`.
//!
//! MuJoCo lets any frame-bearing element state its orientation five different
//! ways, and reference models use all of them — `euler` in particular is
//! pervasive. Reading only `quat` silently leaves every other form at identity,
//! which puts parts of the model in the wrong orientation with no error.

use crate::attrs::Attrs;
use crate::{MjcfError, Result};
use phyz_math::{Mat3, Quat, Vec3};

/// Compiler settings that change how orientation attributes are interpreted.
#[derive(Debug, Clone)]
pub struct AngleConfig {
    /// `compiler/angle="degree"`.
    pub degrees: bool,
    /// `compiler/eulerseq`, e.g. `"xyz"`. Lower case denotes rotation about the
    /// moving (intrinsic) frame, upper case about the fixed (extrinsic) frame.
    pub eulerseq: String,
}

impl Default for AngleConfig {
    fn default() -> Self {
        Self {
            degrees: false,
            eulerseq: "xyz".to_string(),
        }
    }
}

impl AngleConfig {
    /// Convert an angle attribute value to radians.
    pub fn to_radians(&self, angle: f64) -> f64 {
        if self.degrees {
            angle.to_radians()
        } else {
            angle
        }
    }
}

/// Parse whichever orientation attribute is present, in MuJoCo's precedence
/// order, returning the element's rotation matrix.
///
/// Returns `None` when the element specifies no orientation at all, leaving the
/// caller's default (usually identity) in place.
pub fn parse_orientation(a: &Attrs, element: &str, cfg: &AngleConfig) -> Result<Option<Mat3>> {
    if let Some(v) = a.floats("quat") {
        let q = expect_n(element, "quat", &v, 4)?;
        let quat = Quat::new(q[0], q[1], q[2], q[3]);
        if quat.norm() < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "quat",
                a.get("quat").unwrap_or_default(),
                "quaternion has zero norm",
            ));
        }
        return Ok(Some(quat.normalize().to_matrix()));
    }

    if let Some(v) = a.floats("axisangle") {
        let aa = expect_n(element, "axisangle", &v, 4)?;
        let axis = Vec3::new(aa[0], aa[1], aa[2]);
        let norm = axis.norm();
        if norm < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "axisangle",
                a.get("axisangle").unwrap_or_default(),
                "rotation axis is the zero vector",
            ));
        }
        return Ok(Some(
            Quat::from_axis_angle(axis / norm, cfg.to_radians(aa[3])).to_matrix(),
        ));
    }

    if let Some(v) = a.floats("euler") {
        let e = expect_n(element, "euler", &v, 3)?;
        return Ok(Some(euler_to_rot(&e, cfg)?));
    }

    if let Some(v) = a.floats("xyaxes") {
        let xy = expect_n(element, "xyaxes", &v, 6)?;
        return Ok(Some(xyaxes_to_rot(element, &xy, a)?));
    }

    if let Some(v) = a.floats("zaxis") {
        let z = expect_n(element, "zaxis", &v, 3)?;
        let z = Vec3::new(z[0], z[1], z[2]);
        let norm = z.norm();
        if norm < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "zaxis",
                a.get("zaxis").unwrap_or_default(),
                "z axis is the zero vector",
            ));
        }
        return Ok(Some(rotation_z_to(z / norm)));
    }

    Ok(None)
}

/// Reject an orientation attribute with the wrong number of components, rather
/// than silently falling back to identity.
fn expect_n(element: &str, key: &str, v: &[f64], n: usize) -> Result<Vec<f64>> {
    if v.len() != n {
        let raw = v
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        return Err(MjcfError::invalid_attr(
            element,
            key,
            &raw,
            format!("expected {n} numbers, found {}", v.len()),
        ));
    }
    Ok(v.to_vec())
}

/// Compose an Euler triple according to `eulerseq`.
fn euler_to_rot(angles: &[f64], cfg: &AngleConfig) -> Result<Mat3> {
    let seq: Vec<char> = cfg.eulerseq.chars().collect();
    if seq.len() != 3 {
        return Err(MjcfError::invalid_attr(
            "compiler",
            "eulerseq",
            &cfg.eulerseq,
            "expected exactly 3 characters from x/y/z/X/Y/Z",
        ));
    }

    let mut q = Quat::identity();
    for (i, c) in seq.iter().enumerate() {
        let axis = match c.to_ascii_lowercase() {
            'x' => Vec3::new(1.0, 0.0, 0.0),
            'y' => Vec3::new(0.0, 1.0, 0.0),
            'z' => Vec3::new(0.0, 0.0, 1.0),
            _ => {
                return Err(MjcfError::invalid_attr(
                    "compiler",
                    "eulerseq",
                    &cfg.eulerseq,
                    format!("'{c}' is not one of x/y/z/X/Y/Z"),
                ));
            }
        };
        let step = Quat::from_axis_angle(axis, cfg.to_radians(angles[i]));
        // Lower case rotates about the moving (intrinsic) frame, upper case
        // about the fixed (extrinsic) frame.
        q = if c.is_ascii_lowercase() {
            q.mul(&step)
        } else {
            step.mul(&q)
        };
    }
    Ok(q.normalize().to_matrix())
}

/// Build a rotation from an x axis and a y-axis hint (y is orthogonalised).
fn xyaxes_to_rot(element: &str, xy: &[f64], a: &Attrs) -> Result<Mat3> {
    let raw = a.get("xyaxes").unwrap_or_default();
    let x = Vec3::new(xy[0], xy[1], xy[2]);
    let y_hint = Vec3::new(xy[3], xy[4], xy[5]);
    if x.norm() < 1e-12 {
        return Err(MjcfError::invalid_attr(
            element,
            "xyaxes",
            raw,
            "x axis is the zero vector",
        ));
    }
    let x = x / x.norm();
    // Gram-Schmidt the y hint against x.
    let y = y_hint - x * x.dot(y_hint);
    if y.norm() < 1e-12 {
        return Err(MjcfError::invalid_attr(
            element,
            "xyaxes",
            raw,
            "y axis is parallel to the x axis",
        ));
    }
    let y = y / y.norm();
    Ok(Mat3::from_cols(x, y, x.cross(y)))
}

/// Minimal rotation taking +Z onto `dir`, which must be unit length.
pub fn rotation_z_to(dir: Vec3) -> Mat3 {
    let world_z = Vec3::new(0.0, 0.0, 1.0);
    let dot = world_z.dot(dir);
    if dot > 1.0 - 1e-12 {
        return Mat3::identity();
    }
    if dot < -1.0 + 1e-12 {
        // Antiparallel: any axis perpendicular to Z works.
        return Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::PI).to_matrix();
    }
    let axis = world_z.cross(dir);
    let angle = dot.clamp(-1.0, 1.0).acos();
    Quat::from_axis_angle(axis / axis.norm(), angle).to_matrix()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::AttrMap;
    use quick_xml::events::BytesStart;

    fn attrs(pairs: &[(&str, &str)]) -> Attrs {
        let mut e = BytesStart::new("body");
        for (k, v) in pairs {
            e.push_attribute((*k, *v));
        }
        Attrs::read(&e, AttrMap::new()).unwrap()
    }

    fn assert_close(a: Vec3, b: Vec3) {
        assert!((a - b).norm() < 1e-9, "expected {b:?}, got {a:?}");
    }

    #[test]
    fn absent_orientation_is_none() {
        let r = parse_orientation(&attrs(&[("pos", "1 2 3")]), "body", &AngleConfig::default());
        assert!(r.unwrap().is_none());
    }

    #[test]
    fn euler_degrees_rotates_about_z() {
        let cfg = AngleConfig {
            degrees: true,
            eulerseq: "xyz".into(),
        };
        let r = parse_orientation(&attrs(&[("euler", "0 0 90")]), "body", &cfg)
            .unwrap()
            .unwrap();
        assert_close(r * Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn euler_case_selects_intrinsic_or_extrinsic() {
        let intrinsic = parse_orientation(
            &attrs(&[("euler", "90 90 0")]),
            "body",
            &AngleConfig {
                degrees: true,
                eulerseq: "xyz".into(),
            },
        )
        .unwrap()
        .unwrap();
        let extrinsic = parse_orientation(
            &attrs(&[("euler", "90 90 0")]),
            "body",
            &AngleConfig {
                degrees: true,
                eulerseq: "XYZ".into(),
            },
        )
        .unwrap()
        .unwrap();
        let z = Vec3::new(0.0, 0.0, 1.0);
        // Intrinsic xyz composes as Rx(90) * Ry(90): z -> x.
        assert_close(intrinsic * z, Vec3::new(1.0, 0.0, 0.0));
        // Extrinsic XYZ composes the other way, Ry(90) * Rx(90): z -> -y.
        assert_close(extrinsic * z, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn axisangle_matches_equivalent_quat() {
        let cfg = AngleConfig::default();
        let aa = parse_orientation(
            &attrs(&[("axisangle", "0 0 1 1.5707963267948966")]),
            "body",
            &cfg,
        )
        .unwrap()
        .unwrap();
        let q = parse_orientation(
            &attrs(&[("quat", "0.7071067811865476 0 0 0.7071067811865476")]),
            "body",
            &cfg,
        )
        .unwrap()
        .unwrap();
        assert_close(aa * Vec3::new(1.0, 0.0, 0.0), q * Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn zaxis_maps_z_onto_target() {
        let r = parse_orientation(
            &attrs(&[("zaxis", "1 0 0")]),
            "body",
            &AngleConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_close(r * Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn xyaxes_orthogonalises_the_y_hint() {
        let r = parse_orientation(
            &attrs(&[("xyaxes", "1 0 0 1 1 0")]),
            "body",
            &AngleConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_close(r * Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_close(r * Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn degenerate_orientations_error() {
        let cfg = AngleConfig::default();
        assert!(parse_orientation(&attrs(&[("quat", "0 0 0 0")]), "body", &cfg).is_err());
        assert!(parse_orientation(&attrs(&[("zaxis", "0 0 0")]), "body", &cfg).is_err());
        assert!(parse_orientation(&attrs(&[("axisangle", "0 0 0 1")]), "body", &cfg).is_err());
        assert!(parse_orientation(&attrs(&[("euler", "1 2")]), "body", &cfg).is_err());
    }
}
