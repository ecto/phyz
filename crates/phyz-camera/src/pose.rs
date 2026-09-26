//! Camera placement in the world, and the matrices that follow from it.
//!
//! # Optical frame convention (OpenCV)
//!
//! The camera looks down its own **+Z**, **+X** is right across the image and
//! **+Y** is *down* the image. Nothing in this crate ever uses a +X-forward
//! "ROS body" camera frame: mount rotations belong in the extrinsics. See
//! [`phyz_world::CameraIntrinsics`] for the full statement of the convention.

use phyz_math::{Mat3, SpatialTransform, Vec3};
use phyz_world::CameraIntrinsics;

/// Where a camera's optical frame sits in the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    /// Rotation taking *optical* coordinates into world coordinates.
    pub world_from_optical: Mat3,
    /// Optical-frame origin in world coordinates, metres.
    pub position: Vec3,
}

impl CameraPose {
    /// Identity pose: at the world origin, looking down world `+Z`.
    pub fn identity() -> Self {
        Self {
            world_from_optical: Mat3::identity(),
            position: Vec3::zeros(),
        }
    }

    /// Compose a body's world transform with camera extrinsics.
    ///
    /// `body_xform` is a world→body Plücker transform straight out of
    /// [`phyz_world::SensorContext::xforms`]: `pos` is the body origin in world
    /// coordinates and `rot` maps world coordinates into the body frame.
    /// `extrinsics` follows the same convention one level down (body → optical),
    /// matching `phyz_model::GeomInstance::origin` and
    /// `phyz_world::Sensor::Camera::origin`.
    pub fn from_body(body_xform: &SpatialTransform, extrinsics: &SpatialTransform) -> Self {
        let world_from_body = body_xform.rot.transpose();
        Self {
            world_from_optical: world_from_body.mul_mat(&extrinsics.rot.transpose()),
            position: body_xform.pos + world_from_body.mul_vec(extrinsics.pos),
        }
    }

    /// A camera at `eye` whose optical axis points at `target`.
    ///
    /// `up` is a world-space hint for which way is up *in the image*; the image
    /// `+Y` axis ends up pointing away from it, since image `+Y` is down. The
    /// hint is orthogonalised against the view direction, and a hint parallel to
    /// it falls back to world `+Z` (then world `+X`) so the result is always a
    /// valid rotation.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - eye)
            .try_normalize()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let hint = [up, Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)]
            .into_iter()
            .find_map(|h| forward.cross(h).try_normalize().map(|_| h))
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        // right = forward × up_hint gives a right-handed (right, down, forward)
        // basis, which is exactly OpenCV's (X, Y, Z).
        let right = forward.cross(hint).normalize();
        let down = forward.cross(right);
        Self {
            world_from_optical: Mat3::from_cols(right, down, forward),
            position: eye,
        }
    }

    /// Express a world point in the optical frame.
    pub fn to_optical(&self, p_world: Vec3) -> Vec3 {
        self.world_from_optical
            .transpose()
            .mul_vec(p_world - self.position)
    }

    /// Express an optical-frame point in world coordinates.
    pub fn to_world(&self, p_optical: Vec3) -> Vec3 {
        self.position + self.world_from_optical.mul_vec(p_optical)
    }

    /// Project a world point straight to pixel coordinates, or `None` if it is
    /// outside the depth range. See [`CameraIntrinsics::project`].
    pub fn project(&self, intrinsics: &CameraIntrinsics, p_world: Vec3) -> Option<(f64, f64)> {
        intrinsics.project(self.to_optical(p_world))
    }

    /// World→optical matrix, column-major, ready for WGSL's `mat4x4<f32>`.
    pub fn view_matrix(&self) -> [f32; 16] {
        let r = self.world_from_optical.transpose();
        let t = r.mul_vec(self.position);
        let mut m = [0.0f32; 16];
        for c in 0..3 {
            for row in 0..3 {
                m[c * 4 + row] = r.get(row, c) as f32;
            }
        }
        m[12] = -t.x as f32;
        m[13] = -t.y as f32;
        m[14] = -t.z as f32;
        m[15] = 1.0;
        m
    }
}

/// Optical→clip projection matrix, column-major for WGSL.
///
/// Maps the OpenCV optical frame onto wgpu's clip space: NDC `x` right and `y`
/// **up** in `[-1, 1]` (hence the sign flip, since image `+Y` is down), and NDC
/// `z` in `[0, 1]` with 0 at the near plane. An off-centre principal point comes
/// out as the usual clip-space skew in the third column.
pub fn projection_matrix(intrinsics: &CameraIntrinsics) -> [f32; 16] {
    let (w, h) = (intrinsics.width as f64, intrinsics.height as f64);
    let (near, far) = (intrinsics.near, intrinsics.far);
    let mut m = [0.0f32; 16];
    m[0] = (2.0 * intrinsics.fx / w) as f32;
    m[5] = (-2.0 * intrinsics.fy / h) as f32;
    m[8] = (2.0 * intrinsics.cx / w - 1.0) as f32;
    m[9] = (1.0 - 2.0 * intrinsics.cy / h) as f32;
    m[10] = (far / (far - near)) as f32;
    m[11] = 1.0;
    m[14] = (-far * near / (far - near)) as f32;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intr() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 48.0,
            width: 128,
            height: 96,
            near: 0.05,
            far: 20.0,
        }
    }

    /// Apply a column-major 4x4 to a homogeneous point.
    fn apply(m: &[f32; 16], p: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        for row in 0..4 {
            for c in 0..4 {
                out[row] += m[c * 4 + row] as f64 * p[c];
            }
        }
        out
    }

    #[test]
    fn projection_matrix_agrees_with_the_pinhole_formula() {
        let k = intr();
        let m = projection_matrix(&k);
        let p = Vec3::new(0.3, -0.2, 2.0);
        let clip = apply(&m, [p.x, p.y, p.z, 1.0]);
        assert!((clip[3] - p.z).abs() < 1e-9, "w must be optical z");

        let (u_ndc, v_ndc) = (clip[0] / clip[3], clip[1] / clip[3]);
        let (u, v) = k.project(p).unwrap();
        // NDC → pixel, with y flipped because image +Y is down. The matrix is
        // f32, so the tolerance is a float32 epsilon on a ~100 px coordinate,
        // not a claim about the algebra.
        assert!((0.5 * (u_ndc + 1.0) * k.width as f64 - u).abs() < 1e-4);
        assert!((0.5 * (1.0 - v_ndc) * k.height as f64 - v).abs() < 1e-4);
    }

    #[test]
    fn projection_matrix_maps_near_and_far_to_zero_and_one() {
        let k = intr();
        let m = projection_matrix(&k);
        for (z, want) in [(k.near, 0.0), (k.far, 1.0)] {
            let clip = apply(&m, [0.0, 0.0, z, 1.0]);
            assert!((clip[2] / clip[3] - want).abs() < 1e-6, "z={z}");
        }
    }

    #[test]
    fn view_matrix_matches_to_optical() {
        let pose = CameraPose::look_at(
            Vec3::new(1.0, -2.0, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let m = pose.view_matrix();
        let p = Vec3::new(0.2, 0.3, 1.1);
        let got = apply(&m, [p.x, p.y, p.z, 1.0]);
        let want = pose.to_optical(p);
        assert!((got[0] - want.x).abs() < 1e-5);
        assert!((got[1] - want.y).abs() < 1e-5);
        assert!((got[2] - want.z).abs() < 1e-5);
        assert!((got[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn look_at_puts_the_target_at_the_principal_point() {
        let k = intr();
        let target = Vec3::new(3.0, 1.0, 0.4);
        let eye = Vec3::new(-1.0, 0.5, 1.2);
        let pose = CameraPose::look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
        let (u, v) = pose.project(&k, target).unwrap();
        assert!((u - k.cx).abs() < 1e-9, "u={u}");
        assert!((v - k.cy).abs() < 1e-9, "v={v}");
        // And the depth is the straight-line distance, because the target is on
        // the optical axis.
        let d = pose.to_optical(target).z;
        assert!((d - (target - eye).norm()).abs() < 1e-9);
    }

    #[test]
    fn look_at_puts_the_up_hint_above_the_centre() {
        let k = intr();
        let pose = CameraPose::look_at(
            Vec3::zeros(),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        // A point 1 m higher than the target must land above the centre row,
        // i.e. at a smaller v, since image +Y points down.
        let (_, v) = pose.project(&k, Vec3::new(0.0, 5.0, 1.0)).unwrap();
        assert!(v < k.cy, "v={v} should be above cy={}", k.cy);
    }

    #[test]
    fn from_body_composes_extrinsics_the_same_way_geoms_do() {
        // Body at (1,0,0), yawed 90°, with the camera 0.1 m up its own +Z.
        let world_from_body = Mat3::rotation_z(std::f64::consts::FRAC_PI_2);
        let body_xform =
            SpatialTransform::new(world_from_body.transpose(), Vec3::new(1.0, 0.0, 0.0));
        let extr = SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, 0.1));
        let pose = CameraPose::from_body(&body_xform, &extr);
        assert!((pose.position - Vec3::new(1.0, 0.0, 0.1)).norm() < 1e-12);
        // Extrinsics are pure translation, so the optical axes are the body
        // axes: optical +X follows the body's yaw onto world +Y, and optical +Z
        // is unaffected by a yaw.
        let right = pose.world_from_optical.mul_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (right - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-12,
            "{right:?}"
        );
        let fwd = pose.world_from_optical.mul_vec(Vec3::new(0.0, 0.0, 1.0));
        assert!((fwd - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12, "{fwd:?}");
    }

    #[test]
    fn unproject_inverts_project() {
        let k = intr();
        let p = Vec3::new(-0.4, 0.7, 3.0);
        let (u, v) = k.project(p).unwrap();
        let back = k.unproject(u, v, p.z);
        assert!((back - p).norm() < 1e-12);
    }
}
