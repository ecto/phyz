//! Second-order (Coulomb) friction cone.
//!
//! A contact impulse `f = (f_n, f_u, f_w)` — one normal and two tangential
//! components in the contact frame — is physical exactly when it lies in
//!
//! ```text
//! K_mu = { f : ||(f_u, f_w)|| <= mu * f_n,  f_n >= 0 }
//! ```
//!
//! This is a genuine second-order cone, not the pyramidal approximation. The
//! difference is observable: a pyramid gives up to `sqrt(2)` more friction
//! along its diagonal than along its axes, so a block sliding at 45 degrees
//! decelerates differently than one sliding along x. See
//! `docs/design/differentiable-contact.md` §4.1.
//!
//! **Stiction is the cone interior.** When the solver can hold the contact
//! stuck with an impulse strictly inside the cone, it does, and the tangential
//! velocity goes to zero. Sliding is the cone boundary, where
//! `||f_t|| = mu * f_n` anti-parallel to slip. Nothing here thresholds on
//! sliding speed — the old model's `min(mu*Fn, c*|v_t|)` made friction vanish
//! as `|v_t| -> 0` regardless of normal load, so nothing could ever stick.

use phyz_math::Vec3;

/// A contact impulse in the contact frame: `x` normal, `(y, z)` tangential.
pub type ConeVec = Vec3;

/// Euclidean projection of `f` onto the friction cone `K_mu`.
///
/// The three cases are the standard SOC projection:
/// - already inside the cone: unchanged;
/// - inside the *polar* cone: projects to the origin (the contact separates);
/// - otherwise: onto the cone boundary.
pub fn project_cone(f: ConeVec, mu: f64) -> ConeVec {
    let fn_ = f.x;
    let ft = Vec3::new(0.0, f.y, f.z);
    let ft_norm = (f.y * f.y + f.z * f.z).sqrt();

    // Frictionless: just clamp the normal component.
    if mu <= 0.0 {
        return Vec3::new(fn_.max(0.0), 0.0, 0.0);
    }

    if ft_norm <= mu * fn_ {
        // Inside the cone.
        return f;
    }
    if ft_norm <= -fn_ / mu {
        // Inside the polar cone: the nearest cone point is the apex.
        return Vec3::zeros();
    }

    // Boundary projection.
    let scale = (mu * ft_norm + fn_) / (mu * mu + 1.0);
    let t_scale = if ft_norm > 0.0 {
        mu * scale / ft_norm
    } else {
        0.0
    };
    Vec3::new(scale, ft.y * t_scale, ft.z * t_scale)
}

/// True when `f` is inside `K_mu` up to `tol`.
pub fn in_cone(f: ConeVec, mu: f64, tol: f64) -> bool {
    let ft = (f.y * f.y + f.z * f.z).sqrt();
    f.x >= -tol && ft <= mu * f.x + tol
}

/// True when `f` is strictly inside `K_mu` — the stiction condition.
pub fn in_cone_interior(f: ConeVec, mu: f64, tol: f64) -> bool {
    let ft = (f.y * f.y + f.z * f.z).sqrt();
    f.x > tol && ft < mu * f.x - tol
}

/// Build an orthonormal contact frame from a unit normal.
///
/// Returns `(normal, tangent_u, tangent_w)`, right-handed. The tangent choice
/// is arbitrary but must be *stable* across steps for friction anchors to mean
/// anything, so it is derived deterministically from the normal alone.
pub fn contact_frame(normal: &Vec3) -> (Vec3, Vec3, Vec3) {
    let n = normal.normalize();
    // Pick the axis least aligned with n so the cross product is well
    // conditioned.
    let a = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::x()
    } else if n.y.abs() <= n.z.abs() {
        Vec3::y()
    } else {
        Vec3::z()
    };
    let u = n.cross(a).normalize();
    let w = n.cross(u);
    (n, u, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_point_is_unchanged() {
        let f = Vec3::new(1.0, 0.1, 0.1);
        let p = project_cone(f, 0.5);
        assert!((p - f).norm() < 1e-12);
    }

    #[test]
    fn polar_cone_projects_to_apex() {
        // Deep in the polar cone: large negative normal, small tangent.
        let f = Vec3::new(-1.0, 0.01, 0.0);
        let p = project_cone(f, 0.5);
        assert!(p.norm() < 1e-12, "expected apex, got {p:?}");
    }

    #[test]
    fn exterior_point_lands_on_the_boundary() {
        let mu = 0.5;
        let f = Vec3::new(1.0, 3.0, 0.0);
        let p = project_cone(f, mu);
        let ft = (p.y * p.y + p.z * p.z).sqrt();
        assert!(
            (ft - mu * p.x).abs() < 1e-12,
            "not on boundary: ft={ft}, mu*fn={}",
            mu * p.x
        );
        // Projection is idempotent.
        let p2 = project_cone(p, mu);
        assert!((p2 - p).norm() < 1e-12);
    }

    /// The projection must be the *nearest* cone point: no feasible point may
    /// be closer than the one returned.
    #[test]
    fn projection_is_nearest_point() {
        let mu = 0.7;
        let f = Vec3::new(0.3, 1.4, -0.9);
        let p = project_cone(f, mu);
        let d = (p - f).norm();
        // Sample the cone and check nothing beats it.
        for i in 0..60 {
            for j in 0..60 {
                let fn_ = i as f64 * 0.05;
                let theta = j as f64 * std::f64::consts::TAU / 60.0;
                let r = mu * fn_;
                let q = Vec3::new(fn_, r * theta.cos(), r * theta.sin());
                assert!(
                    (q - f).norm() >= d - 1e-9,
                    "found closer feasible point {q:?}"
                );
            }
        }
    }

    /// Frictionless contact keeps only a non-negative normal impulse.
    #[test]
    fn zero_friction_clamps_normal_only() {
        let p = project_cone(Vec3::new(2.0, 5.0, -5.0), 0.0);
        assert!((p - Vec3::new(2.0, 0.0, 0.0)).norm() < 1e-12);
        let p = project_cone(Vec3::new(-2.0, 5.0, -5.0), 0.0);
        assert!(p.norm() < 1e-12);
    }

    /// The cone is isotropic: rotating the tangential part rotates the
    /// projection by the same angle. This is what a pyramidal cone fails.
    #[test]
    fn cone_is_isotropic_in_the_tangent_plane() {
        let mu = 0.6;
        let magnitude = 2.0;
        let reference = project_cone(Vec3::new(0.5, magnitude, 0.0), mu);
        let ref_ft = (reference.y.powi(2) + reference.z.powi(2)).sqrt();
        for k in 1..16 {
            let theta = k as f64 * std::f64::consts::TAU / 16.0;
            let f = Vec3::new(0.5, magnitude * theta.cos(), magnitude * theta.sin());
            let p = project_cone(f, mu);
            let ft = (p.y.powi(2) + p.z.powi(2)).sqrt();
            assert!(
                (ft - ref_ft).abs() < 1e-12,
                "anisotropy at {theta}: {ft} vs {ref_ft}"
            );
            assert!((p.x - reference.x).abs() < 1e-12);
        }
    }

    #[test]
    fn contact_frame_is_orthonormal() {
        for n in [
            Vec3::z(),
            Vec3::x(),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-0.3, 0.7, 0.2),
        ] {
            let (n, u, w) = contact_frame(&n);
            assert!((n.norm() - 1.0).abs() < 1e-12);
            assert!((u.norm() - 1.0).abs() < 1e-12);
            assert!((w.norm() - 1.0).abs() < 1e-12);
            assert!(n.dot(u).abs() < 1e-12);
            assert!(n.dot(w).abs() < 1e-12);
            assert!(u.dot(w).abs() < 1e-12);
            // Right-handed.
            assert!((u.cross(w) - n).norm() < 1e-12);
        }
    }
}
