//! Elastic energies, and the per-vertex gradients and Hessian blocks VBD needs.
//!
//! VBD never assembles a global stiffness matrix. Every energy term here has to
//! answer one question, for one vertex at a time: *given the current positions,
//! what is `∂Ψ/∂xᵢ` and `∂²Ψ/∂xᵢ²`?* — a 3-vector and a 3×3 block, with the
//! coupling to the term's other vertices deliberately dropped. Dropping it is
//! what makes the method block-*coordinate* descent rather than Newton; the
//! Gauss–Seidel sweep recovers the coupling across iterations.

use phyz_math::{Mat3, Vec3};

/// An isotropic hyperelastic material, in the parameters people actually
/// measure.
///
/// `density` is used only to lump element mass onto vertices; it plays no part
/// in the energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Young's modulus `E`, in Pa.
    pub youngs_modulus: f64,
    /// Poisson's ratio `ν`. Must be in `(-1, 0.5)`; values very close to `0.5`
    /// are nearly incompressible and converge slowly here (see the crate docs).
    pub poisson_ratio: f64,
    /// Mass density, kg/m³.
    pub density: f64,
}

impl Default for Material {
    /// A soft rubbery solid: `E = 1 MPa`, `ν = 0.3`, `ρ = 1000 kg/m³`.
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0e6,
            poisson_ratio: 0.3,
            density: 1000.0,
        }
    }
}

impl Material {
    /// The Lamé parameters `(μ, λ)` of linear elasticity.
    pub fn lame(&self) -> (f64, f64) {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (mu, lambda)
    }
}

/// One tetrahedron's worth of stable Neo-Hookean energy, with everything that
/// depends only on the rest shape precomputed.
///
/// The rest shape enters through the constant "shape gradients" `bᵢ`, defined
/// by `∂F/∂xᵢ,ₐ = eₐ ⊗ bᵢ`. Because `F = Dₛ Dₘ⁻¹` is *linear* in the vertex
/// positions, those are genuinely constant — which is the whole reason a tet's
/// per-vertex gradient collapses to `V · P bᵢ` and its Hessian block to a
/// double contraction of `∂²Ψ/∂F²` with `bᵢ` twice.
#[derive(Debug, Clone, PartialEq)]
pub struct TetElement {
    /// Vertex indices, in positive-orientation order (see [`TetElement::new`]).
    pub verts: [usize; 4],
    /// Rest volume, m³. Always positive.
    pub rest_volume: f64,
    /// `μ` of the stable Neo-Hookean energy (not the Lamé `μ`; see below).
    pub mu: f64,
    /// `λ` of the stable Neo-Hookean energy.
    pub lambda: f64,
    dm_inv: Mat3,
    shape_grads: [Vec3; 4],
}

impl TetElement {
    /// Build an element from four rest positions.
    ///
    /// Returns `None` for a degenerate tet (rest volume below `min_volume`);
    /// a sliver has an ill-conditioned `Dₘ⁻¹` and would inject enormous
    /// spurious stiffness rather than fail loudly, so refusing it is safer than
    /// simulating it.
    ///
    /// A negatively oriented tet is not an error — it is what you get from half
    /// the reasonable ways to enumerate a hex decomposition. It is fixed by
    /// swapping the last two vertices, which is deterministic and leaves the
    /// element's geometry alone.
    pub fn new(
        verts: [usize; 4],
        rest: [Vec3; 4],
        material: &Material,
        min_volume: f64,
    ) -> Option<Self> {
        let mut verts = verts;
        let mut rest = rest;
        let mut dm = shape_matrix(&rest);
        if dm.determinant() < 0.0 {
            verts.swap(2, 3);
            rest.swap(2, 3);
            dm = shape_matrix(&rest);
        }
        let rest_volume = dm.determinant() / 6.0;
        if !rest_volume.is_finite() || rest_volume <= min_volume {
            return None;
        }
        let dm_inv = dm.try_inverse()?;

        // bᵢ for i = 1..3 is row (i−1) of Dₘ⁻¹; b₀ is minus their sum, because
        // x₀ appears in all three columns of Dₛ with a minus sign.
        let b1 = dm_inv.row(0);
        let b2 = dm_inv.row(1);
        let b3 = dm_inv.row(2);
        let b0 = -(b1 + b2 + b3);

        // Stable Neo-Hookean (Smith et al. 2018) is written in its own μ and λ,
        // which are *not* the Lamé parameters: matching the small-strain limit
        // of linear elasticity needs μ_snh = 4/3 μ and λ_snh = λ + 5/6 μ.
        // Feeding the Lamé values straight in gives a material about 25% too
        // soft in shear, which is exactly the kind of error a beam-deflection
        // test catches and an eyeball test does not.
        let (mu_lame, lambda_lame) = material.lame();
        let mu = 4.0 / 3.0 * mu_lame;
        let lambda = lambda_lame + 5.0 / 6.0 * mu_lame;

        Some(Self {
            verts,
            rest_volume,
            mu,
            lambda,
            dm_inv,
            shape_grads: [b0, b1, b2, b3],
        })
    }

    /// The deformation gradient `F = Dₛ Dₘ⁻¹` at the given world positions.
    pub fn deformation_gradient(&self, x: &[Vec3]) -> Mat3 {
        let p = [
            x[self.verts[0]],
            x[self.verts[1]],
            x[self.verts[2]],
            x[self.verts[3]],
        ];
        shape_matrix(&p).mul_mat(&self.dm_inv)
    }

    /// Elastic energy stored in this element, in joules.
    pub fn energy(&self, x: &[Vec3]) -> f64 {
        self.rest_volume * snh_energy(&self.deformation_gradient(x), self.mu, self.lambda)
    }

    /// `∂Ψ/∂xᵢ` and `∂²Ψ/∂xᵢ²` for local vertex `local` (0..4).
    ///
    /// The Hessian block is obtained from three directional derivatives of the
    /// first Piola–Kirchhoff stress rather than by forming the 9×9
    /// `∂²Ψ/∂F²` — same numbers, a third of the algebra, and no 81-entry table
    /// to get an index wrong in.
    pub fn gradient_and_hessian(&self, x: &[Vec3], local: usize) -> (Vec3, Mat3) {
        let f = self.deformation_gradient(x);
        let b = self.shape_grads[local];
        let v = self.rest_volume;

        let p = snh_stress(&f, self.mu, self.lambda);
        let grad = p.mul_vec(b) * v;

        let mut cols = [Vec3::zero(); 3];
        for (c, col) in cols.iter_mut().enumerate() {
            // dF = e_c ⊗ b, i.e. row c is b and the other rows are zero.
            let e = basis(c);
            let df = Mat3::from_cols(e * b.x, e * b.y, e * b.z);
            let dp = snh_stress_differential(&f, &df, self.mu, self.lambda);
            *col = dp.mul_vec(b) * v;
        }
        let hess = Mat3::from_cols(cols[0], cols[1], cols[2]);
        // Analytically symmetric; symmetrise so round-off cannot make the
        // eigensolver see a non-symmetric matrix and return nonsense.
        (grad, symmetrize(&hess))
    }
}

/// A linear spring between two vertices, as an *additional* energy term.
///
/// Springs are here because they are cheap, and because a compressed spring is
/// the smallest example of an energy with a genuinely indefinite per-vertex
/// Hessian — see the crate-level notes on the descent-direction guard. They are
/// not a substitute for the FEM energy: a mass-spring lattice has no
/// well-defined Poisson ratio and its effective stiffness depends on the mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    /// The two endpoints.
    pub verts: [usize; 2],
    /// Rest length, m.
    pub rest_length: f64,
    /// Stiffness, N/m.
    pub stiffness: f64,
}

impl Spring {
    /// Energy `½k(‖xᵢ−xⱼ‖ − L)²`.
    pub fn energy(&self, x: &[Vec3]) -> f64 {
        let d = x[self.verts[0]] - x[self.verts[1]];
        let l = d.norm();
        0.5 * self.stiffness * (l - self.rest_length).powi(2)
    }

    /// `∂Ψ/∂xᵢ` and `∂²Ψ/∂xᵢ²` for local endpoint `local` (0 or 1).
    ///
    /// Returns zeros at coincident endpoints: the energy is not differentiable
    /// there, and any finite value we invented would be a direction chosen at
    /// random.
    pub fn gradient_and_hessian(&self, x: &[Vec3], local: usize) -> (Vec3, Mat3) {
        let other = 1 - local;
        let d = x[self.verts[local]] - x[self.verts[other]];
        let l = d.norm();
        if l <= f64::EPSILON {
            return (Vec3::zero(), Mat3::zero());
        }
        let n = d / l;
        let k = self.stiffness;
        let nnt = outer(n, n);
        let grad = n * (k * (l - self.rest_length));
        // k·nnᵀ + k(1 − L/l)(I − nnᵀ). The second term is *negative* whenever
        // the spring is compressed (l < L), and with enough compression the
        // block stops being positive definite. That is not a bug in the
        // formula — it is real buckling curvature — and it is precisely the
        // case the descent guard exists for.
        let hess = nnt * k + (Mat3::identity() - nnt) * (k * (1.0 - self.rest_length / l));
        (grad, symmetrize(&hess))
    }
}

/// `Dₛ = [p₁−p₀, p₂−p₀, p₃−p₀]` as columns.
fn shape_matrix(p: &[Vec3; 4]) -> Mat3 {
    Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0])
}

fn basis(i: usize) -> Vec3 {
    match i {
        0 => Vec3::new(1.0, 0.0, 0.0),
        1 => Vec3::new(0.0, 1.0, 0.0),
        _ => Vec3::new(0.0, 0.0, 1.0),
    }
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn symmetrize(m: &Mat3) -> Mat3 {
    (*m + m.transpose()) * 0.5
}

/// `∂J/∂F`: columns are `f₁×f₂`, `f₂×f₀`, `f₀×f₁`.
fn cofactor(f: &Mat3) -> Mat3 {
    Mat3::from_cols(f.c1.cross(f.c2), f.c2.cross(f.c0), f.c0.cross(f.c1))
}

fn frobenius_dot(a: &Mat3, b: &Mat3) -> f64 {
    a.c0.dot(b.c0) + a.c1.dot(b.c1) + a.c2.dot(b.c2)
}

/// `α = 1 + μ/λ − μ/(4λ)`, the rest-stability constant of stable Neo-Hookean.
fn snh_alpha(mu: f64, lambda: f64) -> f64 {
    1.0 + mu / lambda - mu / (4.0 * lambda)
}

/// Stable Neo-Hookean energy density (Smith, Goldade, Kim & Kim 2018):
/// `Ψ = μ/2 (I_C − 3) + λ/2 (J − α)² − μ/2 log(1 + I_C)`.
///
/// The `log(1 + I_C)` — as opposed to the textbook `log J` — is the "stable"
/// part: it is finite and smooth for *every* `F`, including inverted elements
/// with `J ≤ 0`. A classical Neo-Hookean returns `NaN` the first time an
/// element inverts, and at the timesteps VBD is meant for, elements invert.
fn snh_energy(f: &Mat3, mu: f64, lambda: f64) -> f64 {
    let ic = f.norm_sq();
    let j = f.determinant();
    let alpha = snh_alpha(mu, lambda);
    0.5 * mu * (ic - 3.0) + 0.5 * lambda * (j - alpha).powi(2) - 0.5 * mu * (1.0 + ic).ln()
}

/// First Piola–Kirchhoff stress `P = ∂Ψ/∂F` of [`snh_energy`].
fn snh_stress(f: &Mat3, mu: f64, lambda: f64) -> Mat3 {
    let ic = f.norm_sq();
    let j = f.determinant();
    let alpha = snh_alpha(mu, lambda);
    *f * (mu - mu / (1.0 + ic)) + cofactor(f) * (lambda * (j - alpha))
}

/// `dP = ∂²Ψ/∂F² : dF`, the directional derivative of [`snh_stress`].
fn snh_stress_differential(f: &Mat3, df: &Mat3, mu: f64, lambda: f64) -> Mat3 {
    let ic = f.norm_sq();
    let j = f.determinant();
    let alpha = snh_alpha(mu, lambda);
    let cof = cofactor(f);
    let dic = 2.0 * frobenius_dot(f, df);
    let dj = frobenius_dot(&cof, df);
    let dcof = Mat3::from_cols(
        df.c1.cross(f.c2) + f.c1.cross(df.c2),
        df.c2.cross(f.c0) + f.c2.cross(df.c0),
        df.c0.cross(f.c1) + f.c0.cross(df.c1),
    );

    *df * (mu - mu / (1.0 + ic))
        + *f * (mu * dic / (1.0 + ic).powi(2))
        + cof * (lambda * dj)
        + dcof * (lambda * (j - alpha))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tet() -> ([Vec3; 4], TetElement) {
        let rest = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let m = Material {
            youngs_modulus: 1.0e5,
            poisson_ratio: 0.3,
            density: 1000.0,
        };
        let e = TetElement::new([0, 1, 2, 3], rest, &m, 1e-12).unwrap();
        (rest, e)
    }

    /// A tet at rest is at rest: `F = I`, and the stress must vanish exactly,
    /// which is the point of `α`.
    #[test]
    fn rest_state_is_stress_free() {
        let (rest, e) = unit_tet();
        let f = e.deformation_gradient(&rest);
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((f.get(r, c) - want).abs() < 1e-14);
            }
        }
        let p = snh_stress(&f, e.mu, e.lambda);
        assert!(p.norm_sq().sqrt() < 1e-9, "residual stress {p:?}");
        for local in 0..4 {
            let (g, _) = e.gradient_and_hessian(&rest, local);
            assert!(g.norm() < 1e-9, "residual force on vertex {local}: {g:?}");
        }
    }

    #[test]
    fn rest_volume_matches_geometry() {
        let (_, e) = unit_tet();
        assert!((e.rest_volume - 1.0 / 6.0).abs() < 1e-15);
    }

    /// Inverted elements must still produce finite numbers. This is the single
    /// property that separates stable Neo-Hookean from the textbook one.
    #[test]
    fn inverted_element_stays_finite() {
        let (_, e) = unit_tet();
        let x = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -0.5), // pulled through the opposite face
        ];
        let f = e.deformation_gradient(&x);
        assert!(
            f.determinant() < 0.0,
            "test did not actually invert the tet"
        );
        assert!(e.energy(&x).is_finite());
        for local in 0..4 {
            let (g, h) = e.gradient_and_hessian(&x, local);
            assert!(g.norm().is_finite() && h.norm_sq().is_finite());
        }
    }

    /// The analytic gradient against central differences of the energy. A wrong
    /// sign or a transposed shape gradient dies here and nowhere else.
    #[test]
    fn tet_gradient_matches_finite_differences() {
        let (_, e) = unit_tet();
        let x0 = vec![
            Vec3::new(0.02, -0.01, 0.0),
            Vec3::new(1.1, 0.05, -0.03),
            Vec3::new(-0.04, 0.9, 0.07),
            Vec3::new(0.06, -0.02, 1.15),
        ];
        let h = 1e-6;
        let mut worst = 0.0f64;
        for local in 0..4 {
            let (g, _) = e.gradient_and_hessian(&x0, local);
            for a in 0..3 {
                let mut xp = x0.clone();
                let mut xm = x0.clone();
                bump(&mut xp[local], a, h);
                bump(&mut xm[local], a, -h);
                let fd = (e.energy(&xp) - e.energy(&xm)) / (2.0 * h);
                worst = worst.max((component(g, a) - fd).abs() / (fd.abs() + 1.0));
            }
        }
        assert!(worst < 1e-6, "worst relative gradient error {worst:e}");
    }

    /// The analytic Hessian block against central differences of the gradient.
    #[test]
    fn tet_hessian_matches_finite_differences() {
        let (_, e) = unit_tet();
        let x0 = vec![
            Vec3::new(0.02, -0.01, 0.0),
            Vec3::new(1.1, 0.05, -0.03),
            Vec3::new(-0.04, 0.9, 0.07),
            Vec3::new(0.06, -0.02, 1.15),
        ];
        let h = 1e-6;
        let mut worst = 0.0f64;
        for local in 0..4 {
            let (_, hess) = e.gradient_and_hessian(&x0, local);
            for c in 0..3 {
                let mut xp = x0.clone();
                let mut xm = x0.clone();
                bump(&mut xp[local], c, h);
                bump(&mut xm[local], c, -h);
                let gp = e.gradient_and_hessian(&xp, local).0;
                let gm = e.gradient_and_hessian(&xm, local).0;
                for r in 0..3 {
                    let fd = (component(gp, r) - component(gm, r)) / (2.0 * h);
                    worst = worst.max((hess.get(r, c) - fd).abs() / (fd.abs() + 1.0));
                }
            }
        }
        assert!(worst < 1e-5, "worst relative Hessian error {worst:e}");
    }

    #[test]
    fn spring_gradient_and_hessian_match_finite_differences() {
        let s = Spring {
            verts: [0, 1],
            rest_length: 1.0,
            stiffness: 250.0,
        };
        let x0 = vec![Vec3::new(0.1, 0.2, -0.05), Vec3::new(1.3, -0.1, 0.4)];
        let h = 1e-6;
        for local in 0..2 {
            let (g, hess) = s.gradient_and_hessian(&x0, local);
            for c in 0..3 {
                let mut xp = x0.clone();
                let mut xm = x0.clone();
                bump(&mut xp[local], c, h);
                bump(&mut xm[local], c, -h);
                let fd_e = (s.energy(&xp) - s.energy(&xm)) / (2.0 * h);
                assert!((component(g, c) - fd_e).abs() < 1e-5, "grad {c}");
                let gp = s.gradient_and_hessian(&xp, local).0;
                let gm = s.gradient_and_hessian(&xm, local).0;
                for r in 0..3 {
                    let fd = (component(gp, r) - component(gm, r)) / (2.0 * h);
                    assert!((hess.get(r, c) - fd).abs() < 1e-4, "hess {r},{c}");
                }
            }
        }
    }

    /// A sufficiently compressed spring has an indefinite per-vertex Hessian.
    /// Documenting it as a test so nobody "fixes" the sign later.
    #[test]
    fn compressed_spring_hessian_is_indefinite() {
        let s = Spring {
            verts: [0, 1],
            rest_length: 1.0,
            stiffness: 100.0,
        };
        // Half the rest length, and offset off-axis so the transverse block is
        // visible: transverse curvature is k(1 − L/l) = −100.
        let x = vec![Vec3::zeros(), Vec3::new(0.5, 0.0, 0.0)];
        let (_, hess) = s.gradient_and_hessian(&x, 0);
        assert!(
            hess.get(1, 1) < 0.0,
            "transverse curvature {}",
            hess.get(1, 1)
        );
    }

    #[test]
    fn degenerate_tet_is_rejected() {
        let flat = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        assert!(TetElement::new([0, 1, 2, 3], flat, &Material::default(), 1e-12).is_none());
    }

    /// Inverted input is repaired, not rejected, and the repair is a swap of
    /// the last two vertices.
    #[test]
    fn negative_orientation_is_repaired() {
        let rest = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let e = TetElement::new([0, 1, 2, 3], rest, &Material::default(), 1e-12).unwrap();
        assert!(e.rest_volume > 0.0);
        assert_eq!(e.verts, [0, 1, 3, 2]);
    }

    fn bump(v: &mut Vec3, axis: usize, h: f64) {
        match axis {
            0 => v.x += h,
            1 => v.y += h,
            _ => v.z += h,
        }
    }

    fn component(v: Vec3, axis: usize) -> f64 {
        match axis {
            0 => v.x,
            1 => v.y,
            _ => v.z,
        }
    }
}
