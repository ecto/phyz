//! What comes back from a render.
//!
//! The buffers are deliberately wrapped in enums rather than being bare `Vec`s.
//! The MVP path always reads back to the CPU, but a batched, GPU-resident path
//! (render N environments, hand the textures straight to a learner without ever
//! touching host memory) is the whole point of rasterizing in the first place,
//! and it can be added as another variant without changing a single signature.

use phyz_world::CameraIntrinsics;

/// RGBA8 colour pixels, row-major from the top-left, 4 bytes per pixel.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ColorBuffer {
    /// Pixels already copied to host memory.
    Cpu(Vec<u8>),
}

/// Linear depth in metres along the optical axis, row-major from the top-left,
/// one `f32` per pixel. `0.0` means "no return".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DepthBuffer {
    /// Pixels already copied to host memory.
    Cpu(Vec<f32>),
}

/// One RGBD observation.
///
/// See [`CameraIntrinsics`] for the optical-frame and depth conventions; the
/// short version is OpenCV optical frame (+Z forward, +Y down), `v = 0` is the
/// top image row, and depth is metres of optical-axis `Z`, not ray length.
#[derive(Debug, Clone)]
pub struct CameraFrame {
    /// Intrinsics the frame was rendered with.
    pub intrinsics: CameraIntrinsics,
    /// Simulation time of the render, in seconds.
    pub timestamp: f64,
    /// Shaded colour.
    pub color: ColorBuffer,
    /// Linear depth.
    pub depth: DepthBuffer,
}

impl CameraFrame {
    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.intrinsics.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.intrinsics.height
    }

    /// Colour pixels if they live on the CPU, `None` if they are still on the
    /// GPU.
    pub fn color_cpu(&self) -> Option<&[u8]> {
        match &self.color {
            ColorBuffer::Cpu(v) => Some(v),
        }
    }

    /// Depth pixels if they live on the CPU, `None` if they are still on the
    /// GPU.
    pub fn depth_cpu(&self) -> Option<&[f32]> {
        match &self.depth {
            DepthBuffer::Cpu(v) => Some(v),
        }
    }

    /// Depth at a pixel, in metres. `None` when the pixel is out of bounds, the
    /// depth is not on the CPU, or nothing was hit.
    pub fn depth_at(&self, u: u32, v: u32) -> Option<f32> {
        if u >= self.width() || v >= self.height() {
            return None;
        }
        let d = self.depth_cpu()?[(v * self.width() + u) as usize];
        (d > 0.0).then_some(d)
    }

    /// RGBA at a pixel. `None` when out of bounds or not on the CPU.
    pub fn color_at(&self, u: u32, v: u32) -> Option<[u8; 4]> {
        if u >= self.width() || v >= self.height() {
            return None;
        }
        let px = self.color_cpu()?;
        let i = ((v * self.width() + u) as usize) * 4;
        Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
    }

    /// Depth at the principal point, i.e. straight down the optical axis.
    pub fn depth_at_principal_point(&self) -> Option<f32> {
        self.depth_at(
            self.intrinsics.cx.floor().max(0.0) as u32,
            self.intrinsics.cy.floor().max(0.0) as u32,
        )
    }

    /// Fraction of pixels that returned a depth, in `[0, 1]`.
    pub fn depth_coverage(&self) -> f64 {
        let Some(d) = self.depth_cpu() else {
            return 0.0;
        };
        if d.is_empty() {
            return 0.0;
        }
        d.iter().filter(|&&x| x > 0.0).count() as f64 / d.len() as f64
    }

    /// Back-project the depth image into an optical-frame point cloud, skipping
    /// pixels with no return. Pixel centres are sampled at half-integers.
    pub fn point_cloud(&self) -> Vec<phyz_math::Vec3> {
        let Some(d) = self.depth_cpu() else {
            return Vec::new();
        };
        let (w, h) = (self.width(), self.height());
        let mut out = Vec::new();
        for v in 0..h {
            for u in 0..w {
                let z = d[(v * w + u) as usize];
                if z > 0.0 {
                    out.push(
                        self.intrinsics
                            .unproject(u as f64 + 0.5, v as f64 + 0.5, z as f64),
                    );
                }
            }
        }
        out
    }
}
