//! Terrain as a regular grid of heights.
//!
//! A [`Heightfield`] is the ground the contact solvers test against when a
//! flat plane is not enough: bumps, cracks, ramps and bowls, sampled
//! bilinearly between grid nodes. A flat plane is the degenerate 1×1 field,
//! so callers that used to pass `ground_height` can pass
//! [`Heightfield::flat`] and get bit-for-bit the same surface.

use phyz_math::Vec3;

/// A regular grid of terrain heights, sampled bilinearly.
///
/// Node `(ix, iy)` sits at world `(origin.x + ix·cell, origin.y + iy·cell)`
/// with surface height `origin.z + heights[iy·nx + ix]`. Between nodes the
/// surface is the bilinear patch through the four surrounding nodes; outside
/// the grid the query is clamped to the border, so the terrain extends
/// flatly (per-edge) to infinity rather than falling away to zero. That
/// clamp is also what makes a 1×1 field an exact infinite plane.
///
/// Heights are stored as `f32` deliberately: the GPU contact shader consumes
/// this buffer verbatim, and keeping one representation means the CPU and
/// GPU paths sample *identical* terrain instead of terrain that agrees to
/// rounding. Construction is a couple of allocations, cheap enough to draw a
/// fresh field per training iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct Heightfield {
    /// World position of node `(0, 0)`; `origin.z` is added to every height.
    pub origin: Vec3,
    /// Grid spacing in metres, same in x and y. Must be positive.
    pub cell: f64,
    /// Number of nodes along x. At least 1.
    pub nx: usize,
    /// Number of nodes along y. At least 1.
    pub ny: usize,
    /// Row-major heights, `ny` rows of `nx`: `heights[iy * nx + ix]`.
    pub heights: Vec<f32>,
}

impl Heightfield {
    /// A heightfield of `nx × ny` nodes spaced `cell` apart, all at height
    /// zero above `origin`.
    ///
    /// # Panics
    /// If `nx` or `ny` is zero or `cell` is not a positive finite number.
    pub fn new(origin: Vec3, cell: f64, nx: usize, ny: usize) -> Self {
        assert!(nx >= 1 && ny >= 1, "heightfield needs at least one node");
        assert!(
            cell.is_finite() && cell > 0.0,
            "heightfield cell must be positive, got {cell}"
        );
        Self {
            origin,
            cell,
            nx,
            ny,
            heights: vec![0.0; nx * ny],
        }
    }

    /// The infinite flat plane at `z = height`: a single node whose border
    /// clamp covers the whole world.
    pub fn flat(height: f64) -> Self {
        Self::new(Vec3::new(0.0, 0.0, height), 1.0, 1, 1)
    }

    /// The grid index and intra-cell fraction for one world coordinate,
    /// clamped to the grid. `n` is the node count along that axis.
    fn locate(&self, w: f64, o: f64, n: usize) -> (usize, f64) {
        if n < 2 {
            return (0, 0.0);
        }
        let u = ((w - o) / self.cell).clamp(0.0, (n - 1) as f64);
        // A query exactly on the far border indexes the last cell at t = 1
        // rather than one past it.
        let i = (u as usize).min(n - 2);
        (i, u - i as f64)
    }

    /// Height of node `(ix, iy)` including the origin offset.
    fn node(&self, ix: usize, iy: usize) -> f64 {
        self.origin.z + self.heights[iy * self.nx + ix] as f64
    }

    /// Surface height at world `(x, y)`, bilinear between nodes, clamped to
    /// the border outside the grid.
    pub fn height(&self, x: f64, y: f64) -> f64 {
        let (ix, tx) = self.locate(x, self.origin.x, self.nx);
        let (iy, ty) = self.locate(y, self.origin.y, self.ny);
        if self.nx < 2 && self.ny < 2 {
            return self.node(0, 0);
        }
        let (ix1, iy1) = ((ix + 1).min(self.nx - 1), (iy + 1).min(self.ny - 1));
        let h00 = self.node(ix, iy);
        let h10 = self.node(ix1, iy);
        let h01 = self.node(ix, iy1);
        let h11 = self.node(ix1, iy1);
        (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) + (h01 * (1.0 - tx) + h11 * tx) * ty
    }

    /// Unit surface normal at world `(x, y)`.
    ///
    /// The analytic normal of the bilinear patch: `(-∂h/∂x, -∂h/∂y, 1)`
    /// normalized. Outside the grid the clamp makes the surface flat, so the
    /// normal degrades to `+ẑ` there, and a 1×1 field reports `+ẑ`
    /// everywhere — exactly the flat-plane contact normal.
    pub fn normal(&self, x: f64, y: f64) -> Vec3 {
        let (ix, tx) = self.locate(x, self.origin.x, self.nx);
        let (iy, ty) = self.locate(y, self.origin.y, self.ny);
        // Beyond the border the clamped surface is flat, so the slope there
        // is zero — not the border cell's slope, which would report a normal
        // the actual (clamped) surface no longer has.
        let inside = |w: f64, o: f64, n: usize| w >= o && w <= o + (n - 1) as f64 * self.cell;
        let (mut dhdx, mut dhdy) = (0.0, 0.0);
        if self.nx >= 2 && inside(x, self.origin.x, self.nx) {
            let iy1 = (iy + 1).min(self.ny - 1);
            let d0 = self.node(ix + 1, iy) - self.node(ix, iy);
            let d1 = self.node(ix + 1, iy1) - self.node(ix, iy1);
            dhdx = (d0 * (1.0 - ty) + d1 * ty) / self.cell;
        }
        if self.ny >= 2 && inside(y, self.origin.y, self.ny) {
            let ix1 = (ix + 1).min(self.nx - 1);
            let d0 = self.node(ix, iy + 1) - self.node(ix, iy);
            let d1 = self.node(ix1, iy + 1) - self.node(ix1, iy);
            dhdy = (d0 * (1.0 - tx) + d1 * tx) / self.cell;
        }
        Vec3::new(-dhdx, -dhdy, 1.0).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_field_is_the_plane_everywhere() {
        let hf = Heightfield::flat(0.25);
        for &(x, y) in &[(0.0, 0.0), (-1e3, 42.0), (7.5, -7.5)] {
            assert_eq!(hf.height(x, y), 0.25);
            assert_eq!(hf.normal(x, y), Vec3::z());
        }
    }

    #[test]
    fn bilinear_interpolates_between_nodes() {
        let mut hf = Heightfield::new(Vec3::zeros(), 1.0, 2, 2);
        hf.heights = vec![0.0, 1.0, 2.0, 3.0]; // h(x,y) = x + 2y on [0,1]²
        assert!((hf.height(0.5, 0.5) - 1.5).abs() < 1e-12);
        assert!((hf.height(0.25, 0.75) - 1.75).abs() < 1e-12);
        // Clamped beyond the border: flat continuation of the edge value.
        assert!((hf.height(5.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((hf.height(-5.0, -5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn normal_matches_the_slope() {
        // Uniform 45° ramp along x: h = x.
        let mut hf = Heightfield::new(Vec3::zeros(), 1.0, 3, 2);
        hf.heights = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let n = hf.normal(1.0, 0.5);
        let expect = Vec3::new(-1.0, 0.0, 1.0).normalize();
        assert!((n - expect).norm() < 1e-12, "normal {n:?}");
        // Outside the grid the clamp flattens the surface.
        assert_eq!(hf.normal(100.0, 0.0), Vec3::z());
    }

    #[test]
    fn origin_offsets_the_whole_surface() {
        let hf = Heightfield::new(Vec3::new(10.0, -3.0, 2.0), 0.5, 1, 1);
        assert_eq!(hf.height(0.0, 0.0), 2.0);
    }
}
