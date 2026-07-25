//! Lattice cells and minimum-image displacements.

/// A (possibly triclinic) lattice cell with per-axis periodicity flags.
///
/// `a`, `b`, `c` are the lattice vectors in Å.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    /// First lattice vector.
    pub a: [f64; 3],
    /// Second lattice vector.
    pub b: [f64; 3],
    /// Third lattice vector.
    pub c: [f64; 3],
    /// Which axes are periodic.
    pub periodic: [bool; 3],
}

impl Lattice {
    /// A fully periodic orthorhombic cell with the given edge lengths (Å).
    pub fn orthorhombic(lx: f64, ly: f64, lz: f64) -> Self {
        Self {
            a: [lx, 0.0, 0.0],
            b: [0.0, ly, 0.0],
            c: [0.0, 0.0, lz],
            periodic: [true; 3],
        }
    }

    /// A fully periodic cubic cell of edge `l` (Å).
    pub fn cubic(l: f64) -> Self {
        Self::orthorhombic(l, l, l)
    }

    /// True when the cell is diagonal (no off-diagonal lattice components).
    pub fn is_orthorhombic(&self) -> bool {
        self.a[1].abs()
            + self.a[2].abs()
            + self.b[0].abs()
            + self.b[2].abs()
            + self.c[0].abs()
            + self.c[1].abs()
            < 1e-9
    }

    /// The cell matrix `H` with the lattice vectors as *columns*, so that a
    /// Cartesian displacement is `d = H s` for fractional `s`.
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.a[0], self.b[0], self.c[0]],
            [self.a[1], self.b[1], self.c[1]],
            [self.a[2], self.b[2], self.c[2]],
        ]
    }

    /// Signed cell volume `det H` in Å³.
    pub fn volume(&self) -> f64 {
        let h = self.matrix();
        h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
    }

    /// `H⁻¹`, or `None` for a degenerate cell. Rows of `H⁻¹` are the reciprocal
    /// lattice vectors divided by 2π.
    pub fn inverse(&self) -> Option<[[f64; 3]; 3]> {
        let h = self.matrix();
        let det = self.volume();
        let scale = h
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        if det.abs() <= 1e-12 * scale * scale * scale {
            return None;
        }
        let id = 1.0 / det;
        Some([
            [
                (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * id,
                (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * id,
                (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * id,
            ],
            [
                (h[1][2] * h[2][0] - h[1][0] * h[2][2]) * id,
                (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * id,
                (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * id,
            ],
            [
                (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * id,
                (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * id,
                (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * id,
            ],
        ])
    }

    /// Fractional coordinates `s = H⁻¹ r` (raw, not wrapped).
    pub fn to_fractional(&self, r: [f64; 3]) -> [f64; 3] {
        let Some(hinv) = self.inverse() else {
            return r;
        };
        mat_vec(&hinv, r)
    }

    /// Cartesian coordinates `r = H s`.
    pub fn to_cartesian(&self, s: [f64; 3]) -> [f64; 3] {
        mat_vec(&self.matrix(), s)
    }

    /// Wrap a position into the primary cell along the periodic axes.
    ///
    /// Uses `rem_euclid` on the fractional coordinate, so a position an
    /// arbitrary number of box lengths outside the cell lands in `[0, 1)` in
    /// one shot — a single-crossing correction silently loses atoms whenever a
    /// step is large (a hot start, a bad timestep, a minimizer jump).
    pub fn wrap(&self, r: [f64; 3]) -> [f64; 3] {
        if self.inverse().is_none() {
            return r;
        }
        let mut s = self.to_fractional(r);
        for (k, sk) in s.iter_mut().enumerate() {
            if self.periodic[k] {
                *sk = sk.rem_euclid(1.0);
                // rem_euclid can return exactly 1.0 for tiny negative inputs.
                if *sk >= 1.0 {
                    *sk = 0.0;
                }
            }
        }
        self.to_cartesian(s)
    }

    /// Perpendicular widths: the distance between opposite faces along each
    /// lattice direction, `V / |b × c|` and cyclic. The minimum-image
    /// convention is only valid for cutoffs below half the smallest width.
    pub fn perp_widths(&self) -> [f64; 3] {
        let v = self.volume().abs();
        let areas = [
            vec3::norm(cross(self.b, self.c)),
            vec3::norm(cross(self.c, self.a)),
            vec3::norm(cross(self.a, self.b)),
        ];
        let mut out = [f64::INFINITY; 3];
        for (k, w) in out.iter_mut().enumerate() {
            if areas[k] > 1e-12 {
                *w = v / areas[k];
            }
        }
        out
    }

    /// Reciprocal lattice vectors including the `2π` factor, as rows: the
    /// wavevector for integer triple `m` is `k = m₀ B₀ + m₁ B₁ + m₂ B₂`.
    pub fn reciprocal(&self) -> Option<[[f64; 3]; 3]> {
        // Rows of H⁻¹ are the reciprocal vectors over 2π, since H⁻¹ H = I
        // means row_k · col_l = δ_kl.
        let hinv = self.inverse()?;
        let tau = std::f64::consts::TAU;
        Some([
            vec3::scale(hinv[0], tau),
            vec3::scale(hinv[1], tau),
            vec3::scale(hinv[2], tau),
        ])
    }
}

#[inline]
fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Minimum-image displacement `ri - rj`, wrapped into the cell if one is
/// present. Diagonal (orthorhombic) cells take a fast per-axis path (exact
/// minimum image); general cells wrap in fractional coordinates (`s = H⁻¹d`,
/// round the periodic components, map back). For a non-orthorhombic cell,
/// fractional rounding recovers the exact minimum image whenever that image
/// is shorter than half the cell's minimum slab width — so with an
/// interaction cutoff below that bound (the usual MD condition, and true of
/// every strained cell in an elastic-constant sweep) all images that matter
/// are exact; displacements near the Wigner–Seitz boundary may wrap to a
/// near-minimal image instead, the standard trade-off. Returns the raw
/// displacement for degenerate (non-invertible) cells.
#[inline]
pub fn min_image(d: [f64; 3], cell: Option<&Lattice>) -> [f64; 3] {
    let Some(c) = cell else { return d };
    let lx = c.a[0];
    let ly = c.b[1];
    let lz = c.c[2];
    let off_diag =
        c.a[1].abs() + c.a[2].abs() + c.b[0].abs() + c.b[2].abs() + c.c[0].abs() + c.c[1].abs();
    if off_diag > 1e-9 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        // Skewed, mirrored (negative-length), or degenerate: the general
        // path handles any invertible cell and falls back to the raw
        // displacement otherwise.
        return min_image_general(d, c);
    }
    let mut out = d;
    let dims = [lx, ly, lz];
    for (k, &l) in dims.iter().enumerate() {
        if c.periodic[k] {
            out[k] -= l * (out[k] / l).round();
        }
    }
    out
}

/// General-cell minimum image via fractional-coordinate rounding. `H` has the
/// lattice vectors as columns; `s = H⁻¹ d`, each periodic component of `s` is
/// rounded to the nearest lattice translation, and the result maps back to
/// Cartesian.
fn min_image_general(d: [f64; 3], c: &Lattice) -> [f64; 3] {
    // H = [a b c] as columns.
    let h = [
        [c.a[0], c.b[0], c.c[0]],
        [c.a[1], c.b[1], c.c[1]],
        [c.a[2], c.b[2], c.c[2]],
    ];
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    // Degeneracy is judged relative to the cell's own scale (|det| ~ scale³
    // for a well-conditioned cell): an absolute epsilon would let a large,
    // nearly-flat cell through and blow up `1/det` into garbage wraps.
    let scale = h
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    if det.abs() <= 1e-12 * scale * scale * scale {
        return d;
    }
    let inv_det = 1.0 / det;
    // Rows of H⁻¹ via the adjugate.
    let hinv = [
        [
            (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * inv_det,
            (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * inv_det,
            (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inv_det,
        ],
        [
            (h[1][2] * h[2][0] - h[1][0] * h[2][2]) * inv_det,
            (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * inv_det,
            (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * inv_det,
        ],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * inv_det,
            (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * inv_det,
            (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * inv_det,
        ],
    ];
    let mut s = [0.0; 3];
    for k in 0..3 {
        s[k] = hinv[k][0] * d[0] + hinv[k][1] * d[1] + hinv[k][2] * d[2];
    }
    for (k, sk) in s.iter_mut().enumerate() {
        if c.periodic[k] {
            *sk -= sk.round();
        }
    }
    [
        h[0][0] * s[0] + h[0][1] * s[1] + h[0][2] * s[2],
        h[1][0] * s[0] + h[1][1] * s[1] + h[1][2] * s[2],
        h[2][0] * s[0] + h[2][1] * s[1] + h[2][2] * s[2],
    ]
}

/// Small `[f64; 3]` helpers shared by the `field` numerics.
pub(crate) mod vec3 {
    #[inline]
    pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    #[inline]
    pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    #[inline]
    pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    #[inline]
    pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    #[inline]
    pub fn norm2(a: [f64; 3]) -> f64 {
        dot(a, a)
    }
    #[inline]
    pub fn norm(a: [f64; 3]) -> f64 {
        norm2(a).sqrt()
    }
    #[inline]
    pub fn add_assign(a: &mut [f64; 3], b: [f64; 3]) {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthorhombic_wraps_per_axis() {
        let cell = Lattice {
            a: [10.0, 0.0, 0.0],
            b: [0.0, 10.0, 0.0],
            c: [0.0, 0.0, 10.0],
            periodic: [true, true, true],
        };
        let d = min_image([9.6, 0.2, -9.9], Some(&cell));
        assert!((d[0] - (-0.4)).abs() < 1e-12);
        assert!((d[1] - 0.2).abs() < 1e-12);
        assert!((d[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn sheared_cell_wraps_via_fractional_rounding() {
        let cell = Lattice {
            a: [10.0, 0.0, 0.0],
            b: [2.0, 10.0, 0.0],
            c: [0.0, 0.0, 10.0],
            periodic: [true, true, true],
        };
        let d = min_image([9.6, 0.2, 0.0], Some(&cell));
        assert!((d[0] - (-0.4)).abs() < 1e-12);
        assert!((d[1] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn no_cell_returns_raw_displacement() {
        let d = min_image([9.6, 0.2, 0.0], None);
        assert_eq!(d, [9.6, 0.2, 0.0]);
    }
}
