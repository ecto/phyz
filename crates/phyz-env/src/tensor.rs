//! Tensor interop.
//!
//! # What is actually zero-copy today
//!
//! * **Rust → Rust, CPU.** [`BatchView`] borrows the environment's internal
//!   buffer directly. No copy, no allocation.
//! * **Rust → Python.** The Python binding wraps the same pointer in a numpy
//!   array via `PyArray::borrow_from_array`, so `env.step()` hands the policy a
//!   view of simulator memory (design-only in this pass).
//! * **Rust → `tang::Tensor`.** `Batch::into_tang` *moves* the `Vec<f32>`
//!   into the tensor, which is copy-free but consumes the buffer.
//!   `BatchView::to_tang` copies, because `tang_tensor::Tensor` owns its
//!   `Vec<S>` and has no borrowed constructor
//!   (`tang/crates/tang-tensor/src/tensor.rs:16`).
//! * **GPU → GPU.** Not yet possible. `phyz-gpu` is on wgpu 23,
//!   `tang-gpu` is on wgpu 24, and `tang_gpu::GpuDevice` owns its
//!   `wgpu::Device` by value with no constructor from an existing one. The
//!   design doc lists the three upstream changes that unblock it.
//!
//! Everything here is deliberately explicit about which of those it is. A
//! "zero-copy" claim that quietly memcpys 4096 × 376 floats per step is worse
//! than no claim.

/// A borrowed, row-major `[num_envs, dim]` view of a batch buffer.
#[derive(Debug, Clone, Copy)]
pub struct BatchView<'a> {
    /// The flattened data, `num_envs * dim` long.
    pub data: &'a [f32],
    /// Number of environments (leading axis).
    pub num_envs: usize,
    /// Per-environment width.
    pub dim: usize,
}

impl<'a> BatchView<'a> {
    /// Wrap a flat buffer. Panics if the length does not match the shape.
    pub fn new(data: &'a [f32], num_envs: usize, dim: usize) -> Self {
        assert_eq!(
            data.len(),
            num_envs * dim,
            "buffer of {} does not match {num_envs}×{dim}",
            data.len()
        );
        Self {
            data,
            num_envs,
            dim,
        }
    }

    /// Row `i`, i.e. environment `i`'s slice.
    pub fn env(&self, i: usize) -> &'a [f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// The shape, as tang and numpy would express it.
    pub fn shape(&self) -> [usize; 2] {
        [self.num_envs, self.dim]
    }

    /// Copy into an owned `tang` tensor.
    ///
    /// This **copies**. Use `Batch::into_tang` when you can give up the
    /// buffer, or wait for a borrowed-tensor constructor upstream.
    #[cfg(feature = "tang")]
    pub fn to_tang(&self) -> tang_tensor::Tensor<f32> {
        tang_tensor::Tensor::new(
            self.data.to_vec(),
            tang_tensor::Shape::new(vec![self.num_envs, self.dim]),
        )
    }
}

/// An owned batch buffer that can be moved into a tensor without copying.
#[derive(Debug, Clone)]
pub struct Batch {
    /// The flattened data.
    pub data: Vec<f32>,
    /// Number of environments.
    pub num_envs: usize,
    /// Per-environment width.
    pub dim: usize,
}

impl Batch {
    /// Take ownership of a flat buffer.
    pub fn new(data: Vec<f32>, num_envs: usize, dim: usize) -> Self {
        assert_eq!(data.len(), num_envs * dim);
        Self {
            data,
            num_envs,
            dim,
        }
    }

    /// Borrow as a [`BatchView`].
    pub fn view(&self) -> BatchView<'_> {
        BatchView::new(&self.data, self.num_envs, self.dim)
    }

    /// Move the buffer into a `tang` tensor. No copy: the `Vec` allocation is
    /// transferred.
    #[cfg(feature = "tang")]
    pub fn into_tang(self) -> tang_tensor::Tensor<f32> {
        tang_tensor::Tensor::new(
            self.data,
            tang_tensor::Shape::new(vec![self.num_envs, self.dim]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_rows() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = BatchView::new(&data, 3, 2);
        assert_eq!(v.env(1), &[3.0, 4.0]);
        assert_eq!(v.shape(), [3, 2]);
    }

    #[test]
    #[should_panic]
    fn view_rejects_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let _ = BatchView::new(&data, 2, 2);
    }
}
