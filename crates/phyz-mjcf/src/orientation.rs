//! MJCF orientation attributes: `quat`, `euler`, `axisangle`, `xyaxes`, `zaxis`.

use crate::attrs::Attrs;
use crate::{MjcfError, Result};
use phyz_math::{Mat3, Quat, Vec3};

/// Compiler settings that change how orientation attributes are interpreted.
#[derive(Debug, Clone)]
pub struct AngleConfig {
    /// `compiler/angle="degree"`.
    pub degrees: bool,
    /// `compiler/eulerseq`, e.g. `"xyz"`. Lower case = intrinsic (moving axes),
    /// upper case = extrinsic (fixed axes).
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

/// Parse whichever orientation attribute is present, in MuJoCo's precedence order.
///
/// Returns `None` when the element specifies no orientation at all (identity).
pub fn parse_orientation(attrs: &Attrs, cfg: &AngleConfig) -> Result<Option<Quat>> {
    let element = attrs.element();

    if let Some(q) = attrs.fixed::<4>("quat")? {
        let quat = Quat::new(q[0], q[1], q[2], q[3]);
        if quat.norm() < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "quat",
                attrs.get("quat").unwrap_or_default(),
                "quaternion has zero norm",
            ));
        }
        return Ok(Some(quat.normalize()));
    }

    if let Some(aa) = attrs.fixed::<4>("axisangle")? {
        let axis = Vec3::new(aa[0], aa[1], aa[2]);
        let norm = axis.norm();
        if norm < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "axisangle",
                attrs.get("axisangle").unwrap_or_default(),
                "rotation axis is the zero vector",
            ));
        }
        let angle = cfg.to_radians(aa[3]);
        return Ok(Some(Quat::from_axis_angle(axis / norm, angle)));
    }

    if let Some(e) = attrs.fixed::<3>("euler")? {
        return Ok(Some(euler_to_quat(element, &e, cfg)?));
    }

    if let Some(xy) = attrs.fixed::<6>("xyaxes")? {
        return Ok(Some(xyaxes_to_quat(element, &xy, attrs)?));
    }

    if let Some(z) = attrs.fixed::<3>("zaxis")? {
        let z = Vec3::new(z[0], z[1], z[2]);
        let norm = z.norm();
        if norm < 1e-12 {
            return Err(MjcfError::invalid_attr(
                element,
                "zaxis",
                attrs.get("zaxis").unwrap_or_default(),
                "z axis is the zero vector",
            ));
        }
        return Ok(Some(quat_from_z(z / norm)));
    }

    Ok(None)
}

/// Compose an Euler triple according to `eulerseq`.
fn euler_to_quat(element: &str, angles: &[f64; 3], cfg: &AngleConfig) -> Result<Quat> {
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
        // Lower case rotates about the moving (intrinsic) frame, upper case about
        // the fixed (extrinsic) frame.
        q = if c.is_ascii_lowercase() {
            q.mul(&step)
        } else {
            step.mul(&q)
        };
    }
    let _ = element;
    Ok(q.normalize())
}

/// Build a rotation from an x axis and a y-axis hint (y is orthogonalised).
fn xyaxes_to_quat(element: &str, xy: &[f64; 6], attrs: &Attrs) -> Result<Quat> {
    let raw = attrs.get("xyaxes").unwrap_or_default();
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
    let z = x.cross(y);
    Ok(Quat::from_matrix(&mat_from_cols(x, y, z)).normalize())
}

/// Minimal rotation taking +Z onto `z` (which must be unit length).
fn quat_from_z(z: Vec3) -> Quat {
    let world_z = Vec3::new(0.0, 0.0, 1.0);
    let dot = world_z.dot(z);
    if dot > 1.0 - 1e-12 {
        return Quat::identity();
    }
    if dot < -1.0 + 1e-12 {
        // Antiparallel: any axis perpendicular to Z works.
        return Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::PI);
    }
    let axis = world_z.cross(z);
    let angle = dot.clamp(-1.0, 1.0).acos();
    Quat::from_axis_angle(axis / axis.norm(), angle)
}

/// Rotation whose columns are the given basis vectors.
pub fn mat_from_cols(x: Vec3, y: Vec3, z: Vec3) -> Mat3 {
    Mat3::from_cols(x, y, z)
}

/// Frame for a `fromto`-specified geom: midpoint, orientation taking +Z along the
/// segment, and the segment half-length.
pub struct FromTo {
    pub center: Vec3,
    pub quat: Quat,
    pub half_length: f64,
}

/// Interpret a `fromto="x1 y1 z1 x2 y2 z2"` attribute.
pub fn parse_fromto(element: &str, ft: &[f64; 6]) -> Result<FromTo> {
    let a = Vec3::new(ft[0], ft[1], ft[2]);
    let b = Vec3::new(ft[3], ft[4], ft[5]);
    let delta = b - a;
    let len = delta.norm();
    if len < 1e-12 {
        return Err(MjcfError::invalid_attr(
            element,
            "fromto",
            &format!(
                "{} {} {} {} {} {}",
                ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]
            ),
            "endpoints coincide, giving a zero-length geom",
        ));
    }
    Ok(FromTo {
        center: (a + b) * 0.5,
        quat: quat_from_z(delta / len),
        half_length: len * 0.5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn attrs(pairs: &[(&str, &str)]) -> Attrs {
        Attrs::from_map(
            "body",
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>(),
        )
    }

    fn assert_vec_close(a: Vec3, b: Vec3) {
        assert!((a - b).norm() < 1e-9, "expected {b:?}, got {a:?}");
    }

    #[test]
    fn no_orientation_attribute_yields_none() {
        assert!(
            parse_orientation(&attrs(&[("pos", "1 2 3")]), &AngleConfig::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn euler_degrees_rotates_about_z() {
        let cfg = AngleConfig {
            degrees: true,
            eulerseq: "xyz".to_string(),
        };
        let q = parse_orientation(&attrs(&[("euler", "0 0 90")]), &cfg)
            .unwrap()
            .unwrap();
        assert_vec_close(q.rotate(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn euler_case_selects_intrinsic_or_extrinsic() {
        // 90 deg about x then 90 deg about y. Intrinsic (lower) and extrinsic
        // (upper) disagree, which is the whole point of the case convention.
        let intrinsic = parse_orientation(
            &attrs(&[("euler", "90 90 0")]),
            &AngleConfig {
                degrees: true,
                eulerseq: "xyz".into(),
            },
        )
        .unwrap()
        .unwrap();
        let extrinsic = parse_orientation(
            &attrs(&[("euler", "90 90 0")]),
            &AngleConfig {
                degrees: true,
                eulerseq: "XYZ".into(),
            },
        )
        .unwrap()
        .unwrap();
        let v = Vec3::new(0.0, 0.0, 1.0);
        assert!((intrinsic.rotate(v) - extrinsic.rotate(v)).norm() > 1e-6);
        // Intrinsic xyz composes as Rx(90) * Ry(90): z -> x -> x.
        assert_vec_close(intrinsic.rotate(v), Vec3::new(1.0, 0.0, 0.0));
        // Extrinsic XYZ composes the other way, Ry(90) * Rx(90): z -> -y -> -y.
        assert_vec_close(extrinsic.rotate(v), Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn axisangle_matches_quat() {
        let cfg = AngleConfig::default();
        let aa = parse_orientation(&attrs(&[("axisangle", "0 0 1 1.5707963267948966")]), &cfg)
            .unwrap()
            .unwrap();
        assert_vec_close(
            aa.rotate(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn zaxis_maps_z_onto_target() {
        let cfg = AngleConfig::default();
        let q = parse_orientation(&attrs(&[("zaxis", "1 0 0")]), &cfg)
            .unwrap()
            .unwrap();
        assert_vec_close(q.rotate(Vec3::new(0.0, 0.0, 1.0)), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn xyaxes_orthogonalises_y() {
        let cfg = AngleConfig::default();
        // y hint is not perpendicular to x; it should be projected.
        let q = parse_orientation(&attrs(&[("xyaxes", "1 0 0 1 1 0")]), &cfg)
            .unwrap()
            .unwrap();
        assert_vec_close(q.rotate(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(1.0, 0.0, 0.0));
        assert_vec_close(q.rotate(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn fromto_gives_midpoint_and_half_length() {
        let ft = parse_fromto("geom", &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
        assert_vec_close(ft.center, Vec3::new(0.0, 0.0, 1.0));
        assert!((ft.half_length - 1.0).abs() < 1e-12);
        assert_vec_close(
            ft.quat.rotate(Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(0.0, 0.0, 1.0),
        );
    }

    #[test]
    fn fromto_along_x_rotates_z_onto_x() {
        let ft = parse_fromto("geom", &[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_vec_close(ft.center, Vec3::zeros());
        assert!((ft.half_length - 1.0).abs() < 1e-12);
        assert_vec_close(
            ft.quat.rotate(Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn degenerate_fromto_errors() {
        assert!(parse_fromto("geom", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn zero_quat_errors() {
        let err = parse_orientation(&attrs(&[("quat", "0 0 0 0")]), &AngleConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero norm"), "{err}");
    }
}
