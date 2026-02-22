//! GPU-accelerated Lanczos eigensolver.
//!
//! Same algorithm as [`crate::lanczos::lanczos`] but with GPU-accelerated
//! SpMV, dot products, and reorthogonalization. The Hamiltonian is
//! pre-built as a CSR sparse matrix and uploaded to GPU once.

use crate::csr::build_csr;
use crate::diag::Spectrum;
use crate::hilbert::U1HilbertSpace;
use nalgebra::{DMatrix, DVector};
use phyz_gpu::sparse::{request_device, GpuPrecision, GpuSparseMatrix, GpuVecOps};
use phyz_regge::SimplicialComplex;
use std::sync::Arc;

/// Chunked q_bank that splits Lanczos vectors across multiple GPU buffers
/// to stay within `max_storage_buffer_binding_size` / `max_buffer_size`.
struct QBank {
    chunks: Vec<wgpu::Buffer>,
    vecs_per_chunk: u32,
    vec_bytes: u64,
}

impl QBank {
    fn new(ops: &GpuVecOps, max_vecs: u32) -> Self {
        let elem_size = ops.precision.elem_size() as u64;
        let vec_bytes = ops.dim as u64 * elem_size;

        let limits = ops.device.limits();
        let limit = (limits.max_buffer_size).min(limits.max_storage_buffer_binding_size as u64);

        let raw_vpc = (limit / vec_bytes) as u32;

        // Round down so that vpc * elem_size is a multiple of
        // min_storage_buffer_offset_alignment (needed for scalar_result_buf offsets).
        let align = limits.min_storage_buffer_offset_alignment;
        let elems_per_align = (align as u64 / elem_size).max(1) as u32;
        let vpc = (raw_vpc / elems_per_align) * elems_per_align;
        assert!(
            vpc > 0,
            "single vector ({vec_bytes} bytes) exceeds GPU buffer limit ({limit} bytes)"
        );

        // If everything fits in one chunk, just use max_vecs directly.
        if vpc >= max_vecs {
            let size = max_vecs as u64 * vec_bytes;
            let buf = ops.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q_bank_chunk_0"),
                size: size.max(elem_size),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            return Self {
                chunks: vec![buf],
                vecs_per_chunk: max_vecs,
                vec_bytes,
            };
        }

        let n_chunks = max_vecs.div_ceil(vpc);
        let mut chunks = Vec::with_capacity(n_chunks as usize);
        for i in 0..n_chunks {
            let n = if i < n_chunks - 1 {
                vpc
            } else {
                max_vecs - vpc * (n_chunks - 1)
            };
            let size = n as u64 * vec_bytes;
            let buf = ops.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("q_bank_chunk_{i}")),
                size: size.max(elem_size),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            chunks.push(buf);
        }

        Self {
            chunks,
            vecs_per_chunk: vpc,
            vec_bytes,
        }
    }

    fn chunk_and_local(&self, index: u32) -> (usize, u32) {
        let chunk = (index / self.vecs_per_chunk) as usize;
        let local = index % self.vecs_per_chunk;
        (chunk, local)
    }

    fn upload(&self, ops: &GpuVecOps, index: u32, data: &[f64]) {
        let (chunk, local) = self.chunk_and_local(index);
        let offset = local as u64 * self.vec_bytes;
        match ops.precision {
            GpuPrecision::F64 => {
                ops.queue
                    .write_buffer(&self.chunks[chunk], offset, bytemuck::cast_slice(data));
            }
            GpuPrecision::F32 => {
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                ops.queue.write_buffer(
                    &self.chunks[chunk],
                    offset,
                    bytemuck::cast_slice(&f32_data),
                );
            }
        }
    }

    fn download(&self, ops: &GpuVecOps, index: u32) -> Vec<f64> {
        let (chunk, local) = self.chunk_and_local(index);
        let offset = local as u64 * self.vec_bytes;

        let staging = ops.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qbank_download_staging"),
            size: self.vec_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ops.encoder();
        encoder.copy_buffer_to_buffer(&self.chunks[chunk], offset, &staging, 0, self.vec_bytes);
        ops.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        ops.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("channel closed")
            .expect("buffer map failed");

        let data = slice.get_mapped_range();
        let result = match ops.precision {
            GpuPrecision::F64 => bytemuck::cast_slice::<u8, f64>(&data).to_vec(),
            GpuPrecision::F32 => {
                let f32s: &[f32] = bytemuck::cast_slice(&data);
                f32s.iter().map(|&v| v as f64).collect()
            }
        };
        drop(data);
        staging.unmap();
        result
    }

    fn copy_to_buf(&self, encoder: &mut wgpu::CommandEncoder, index: u32, buf: &wgpu::Buffer) {
        let (chunk, local) = self.chunk_and_local(index);
        let offset = local as u64 * self.vec_bytes;
        encoder.copy_buffer_to_buffer(&self.chunks[chunk], offset, buf, 0, self.vec_bytes);
    }

    fn copy_from_buf(&self, encoder: &mut wgpu::CommandEncoder, buf: &wgpu::Buffer, index: u32) {
        let (chunk, local) = self.chunk_and_local(index);
        let offset = local as u64 * self.vec_bytes;
        encoder.copy_buffer_to_buffer(buf, 0, &self.chunks[chunk], offset, self.vec_bytes);
    }

    fn run_multi_dot(&self, ops: &GpuVecOps, w_buf: &wgpu::Buffer, n_vecs: u32) {
        if n_vecs == 0 {
            return;
        }
        let mut offset = 0u32;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let start = i as u32 * self.vecs_per_chunk;
            if start >= n_vecs {
                break;
            }
            let end = ((i as u32 + 1) * self.vecs_per_chunk).min(n_vecs);
            let count = end - start;
            ops.run_multi_dot_range(chunk, w_buf, count, offset);
            offset += count;
        }
    }

    fn run_batch_subtract(&self, ops: &GpuVecOps, w_buf: &wgpu::Buffer, n_vecs: u32) {
        if n_vecs == 0 {
            return;
        }
        let mut offset = 0u32;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let start = i as u32 * self.vecs_per_chunk;
            if start >= n_vecs {
                break;
            }
            let end = ((i as u32 + 1) * self.vecs_per_chunk).min(n_vecs);
            let count = end - start;
            ops.run_batch_subtract_range(chunk, w_buf, count, offset);
            offset += count;
        }
    }
}

/// Run GPU-accelerated Lanczos diagonalization.
///
/// Same interface as [`crate::lanczos::lanczos_diagonalize`] but uses GPU
/// for the heavy linear algebra (SpMV, dot products, reorthogonalization).
///
/// Returns `Err` if no GPU adapter is available.
pub fn gpu_lanczos_diagonalize(
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
    metric_weights: Option<&[f64]>,
    n_eigenvalues: usize,
    max_iter: Option<usize>,
) -> Result<Spectrum, String> {
    let dim = hilbert.dim();
    let max_iter = max_iter
        .unwrap_or_else(|| (20 * n_eigenvalues).max(100))
        .min(dim);
    let m = max_iter.min(dim);
    let k = n_eigenvalues.min(m);

    // Build CSR on CPU
    eprintln!("  GPU Lanczos: building CSR (dim={dim})...");
    let csr = build_csr(hilbert, complex, g_squared, metric_weights);
    eprintln!(
        "  GPU Lanczos: CSR built (nnz={}, density={:.4}%)",
        csr.nnz(),
        100.0 * csr.nnz() as f64 / (dim as f64 * dim as f64)
    );

    // Initialize GPU
    let (device, queue, precision) = request_device()?;
    let tol = match precision {
        GpuPrecision::F64 => 1e-10,
        GpuPrecision::F32 => 1e-5,
    };
    let double_reorth = precision == GpuPrecision::F32;

    eprintln!(
        "  GPU Lanczos: precision={:?}, tol={tol:.0e}, max_iter={m}",
        precision
    );

    gpu_lanczos_inner(
        device,
        queue,
        precision,
        &csr,
        dim,
        k,
        m,
        tol,
        double_reorth,
    )
}

/// Inner GPU Lanczos loop. Separated for testability.
fn gpu_lanczos_inner(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    precision: GpuPrecision,
    csr: &crate::csr::CsrMatrix,
    dim: usize,
    k: usize,
    m: usize,
    tol: f64,
    double_reorth: bool,
) -> Result<Spectrum, String> {
    let dim32 = dim as u32;
    let ops = GpuVecOps::new(device.clone(), queue.clone(), precision, dim32);

    // Upload CSR to GPU
    let gpu_matrix = GpuSparseMatrix::upload(
        &device,
        &queue,
        &csr.row_ptr,
        &csr.col_indices,
        &csr.values,
        precision,
    );

    // Allocate GPU buffers
    let q_bank = QBank::new(&ops, (m + 1) as u32);
    let w_buf = ops.create_vec_buffer("w");

    // Tridiagonal elements (CPU)
    let mut alpha: Vec<f64> = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);

    // Initial vector: deterministic seed (same as CPU lanczos)
    let mut q0 = vec![0.0f64; dim];
    for i in 0..dim {
        q0[i] = ((i as f64 + 1.0) * 0.618033988749895).fract() - 0.5;
    }
    let norm: f64 = q0.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut q0 {
        *x /= norm;
    }

    // Upload q0 to q_bank[0]
    q_bank.upload(&ops, 0, &q0);

    let mut prev_eigenvalues = vec![f64::MAX; k];

    for j in 0..m {
        // --- Round trip 1: SpMV + dot -> alpha ---

        // Encode: w = H * q_j
        let mut encoder = ops.encoder();
        // Copy q_j from (possibly chunked) q_bank into a flat buffer for SpMV.
        let qj_buf = ops.create_vec_buffer("qj");
        q_bank.copy_to_buf(&mut encoder, j as u32, &qj_buf);

        // w = H * qj
        ops.encode_spmv(&mut encoder, &gpu_matrix, &qj_buf, &w_buf);

        // alpha = dot(qj, w)
        ops.encode_dot(&mut encoder, &qj_buf, &w_buf);
        let a = ops.submit_and_read_scalar(encoder);
        alpha.push(a);

        // --- Encode: three-term recurrence + reorthogonalization ---
        let mut encoder = ops.encoder();

        // w -= alpha * q_j
        ops.encode_axpy(&mut encoder, -a, &qj_buf, &w_buf);

        // w -= beta[j-1] * q_{j-1}
        if j > 0 {
            let qjm1_buf = ops.create_vec_buffer("qjm1");
            q_bank.copy_to_buf(&mut encoder, (j - 1) as u32, &qjm1_buf);
            // Need to submit the copy before reading qjm1_buf
            ops.submit(encoder);

            let mut encoder = ops.encoder();
            ops.encode_axpy(&mut encoder, -beta[j - 1], &qjm1_buf, &w_buf);
            ops.submit(encoder);
        } else {
            ops.submit(encoder);
        }

        // Full reorthogonalization via multi_dot + batch_subtract
        let n_vecs = (j + 1) as u32;
        q_bank.run_multi_dot(&ops, &w_buf, n_vecs);
        q_bank.run_batch_subtract(&ops, &w_buf, n_vecs);

        // Double reorthogonalization for f32
        if double_reorth {
            q_bank.run_multi_dot(&ops, &w_buf, n_vecs);
            q_bank.run_batch_subtract(&ops, &w_buf, n_vecs);
        }

        // --- Round trip 2: norm -> beta ---
        let mut encoder = ops.encoder();
        ops.encode_dot(&mut encoder, &w_buf, &w_buf);
        let b_sq = ops.submit_and_read_scalar(encoder);
        let b = b_sq.sqrt();

        // Check convergence periodically
        if (j + 1) % 10 == 0 || j == m - 1 || b < 1e-14 {
            let spec = diagonalize_tridiagonal(&alpha, &beta, k);
            let max_change = spec
                .iter()
                .zip(prev_eigenvalues.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if max_change < tol {
                eprintln!(
                    "  GPU Lanczos converged at iteration {} (change={:.2e})",
                    j + 1,
                    max_change
                );
                return Ok(recover_eigenvectors_gpu(
                    &alpha, &beta, &ops, &q_bank, dim, k,
                ));
            }
            prev_eigenvalues = spec;
        }

        if b < 1e-14 {
            eprintln!(
                "  GPU Lanczos: invariant subspace found at iteration {}",
                j + 1
            );
            return Ok(recover_eigenvectors_gpu(
                &alpha, &beta, &ops, &q_bank, dim, k,
            ));
        }

        beta.push(b);

        // q_{j+1} = w / beta
        let mut encoder = ops.encoder();
        ops.encode_scale(&mut encoder, 1.0 / b, &w_buf);
        q_bank.copy_from_buf(&mut encoder, &w_buf, (j + 1) as u32);
        ops.submit(encoder);
    }

    eprintln!("  GPU Lanczos: max iterations ({m}) reached");
    Ok(recover_eigenvectors_gpu(
        &alpha, &beta, &ops, &q_bank, dim, k,
    ))
}

/// Diagonalize the tridiagonal matrix to get eigenvalues only.
fn diagonalize_tridiagonal(alpha: &[f64], beta: &[f64], k: usize) -> Vec<f64> {
    let m = alpha.len();
    let mut t = DMatrix::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
        if i > 0 {
            t[(i, i - 1)] = beta[i - 1];
            t[(i - 1, i)] = beta[i - 1];
        }
    }
    let eig = t.symmetric_eigen();
    let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.truncate(k);
    vals
}

/// Recover eigenvectors by downloading Lanczos vectors from GPU.
fn recover_eigenvectors_gpu(
    alpha: &[f64],
    beta: &[f64],
    ops: &GpuVecOps,
    q_bank: &QBank,
    dim: usize,
    k: usize,
) -> Spectrum {
    let m = alpha.len();
    let mut t = DMatrix::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
        if i > 0 {
            t[(i, i - 1)] = beta[i - 1];
            t[(i - 1, i)] = beta[i - 1];
        }
    }
    let eig = t.symmetric_eigen();

    let mut indexed: Vec<(usize, f64)> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = k.min(indexed.len());
    let n_q = m; // number of Lanczos vectors we have

    // Download all Lanczos vectors from GPU
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_q);
    for j in 0..n_q {
        q_vecs.push(q_bank.download(ops, j as u32));
    }

    let mut energies = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);

    for &(idx, eval) in indexed.iter().take(n) {
        energies.push(eval);

        // v_k = sum_j s_{j,k} * q_j (computed in f64 on CPU)
        let mut v = DVector::zeros(dim);
        for j in 0..n_q {
            let coeff = eig.eigenvectors[(j, idx)];
            for i in 0..dim {
                v[i] += coeff * q_vecs[j][i];
            }
        }
        let norm = v.norm();
        if norm > 1e-15 {
            v /= norm;
        }
        states.push(v);
    }

    Spectrum { energies, states }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag;
    use crate::hamiltonian::{build_hamiltonian, KSParams};
    use crate::lanczos::lanczos_diagonalize;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    fn has_gpu() -> bool {
        request_device().is_ok()
    }

    #[test]
    fn test_gpu_lanczos_vs_dense() {
        if !has_gpu() {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        // Dense reference
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let dense = diag::diagonalize(&h, Some(5));

        // GPU Lanczos
        let gpu = gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None).unwrap();

        let (_, _, precision) = request_device().unwrap();
        let tol = match precision {
            GpuPrecision::F64 => 1e-8,
            GpuPrecision::F32 => 1e-3,
        };

        let e0_diff = (dense.ground_energy() - gpu.ground_energy()).abs();
        assert!(
            e0_diff < tol,
            "E0 mismatch: dense={}, gpu={}, diff={e0_diff}",
            dense.ground_energy(),
            gpu.ground_energy()
        );

        let gap_diff = (dense.gap() - gpu.gap()).abs();
        assert!(
            gap_diff < tol,
            "Gap mismatch: dense={}, gpu={}, diff={gap_diff}",
            dense.gap(),
            gpu.gap()
        );
    }

    #[test]
    fn test_gpu_lanczos_vs_cpu_lanczos() {
        if !has_gpu() {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let cpu = lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None);
        let gpu = gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None).unwrap();

        let (_, _, precision) = request_device().unwrap();
        let tol = match precision {
            GpuPrecision::F64 => 1e-8,
            GpuPrecision::F32 => 1e-2, // f32: degenerate levels may reorder
        };

        // Compare distinct energy levels (degeneracies may reorder in f32)
        let cpu_levels = distinct_levels(&cpu.energies, 1e-4);
        let gpu_levels = distinct_levels(&gpu.energies, 1e-4);
        let n_check = 3.min(cpu_levels.len()).min(gpu_levels.len());
        for i in 0..n_check {
            let diff = (cpu_levels[i] - gpu_levels[i]).abs();
            assert!(
                diff < tol,
                "Level {i} mismatch: cpu={}, gpu={}, diff={diff}",
                cpu_levels[i],
                gpu_levels[i]
            );
        }
    }

    fn distinct_levels(energies: &[f64], gap_tol: f64) -> Vec<f64> {
        let mut levels = vec![energies[0]];
        for &e in &energies[1..] {
            if (e - levels.last().unwrap()).abs() > gap_tol {
                levels.push(e);
            }
        }
        levels
    }

    #[test]
    fn test_gpu_lanczos_strong_coupling() {
        if !has_gpu() {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let (_, _, precision) = request_device().unwrap();

        // f32 at g²=1e6 has condition number ~1e12, way beyond f32 precision.
        // Use moderate coupling for f32.
        let (g_sq, e0_tol) = match precision {
            GpuPrecision::F64 => (1e6, 1e-2),
            GpuPrecision::F32 => (1e2, 0.5), // moderate coupling, relaxed tolerance
        };

        let gpu = gpu_lanczos_diagonalize(&hs, &complex, g_sq, None, 3, None).unwrap();
        assert!(
            gpu.ground_energy().abs() < e0_tol,
            "Strong coupling E0 = {} (g²={g_sq})",
            gpu.ground_energy()
        );
    }

    #[test]
    #[ignore] // slow: dim=3135
    fn test_gpu_lanczos_two_pentachorons() {
        if !has_gpu() {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        let complex =
            SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
        let hs = U1HilbertSpace::new(&complex, 1);

        let cpu = lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None);
        let gpu = gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None).unwrap();

        let (_, _, precision) = request_device().unwrap();
        let tol = match precision {
            GpuPrecision::F64 => 1e-6,
            GpuPrecision::F32 => 1e-2,
        };

        let e0_diff = (cpu.ground_energy() - gpu.ground_energy()).abs();
        assert!(
            e0_diff < tol,
            "2-pent E0 mismatch: cpu={}, gpu={}, diff={e0_diff}",
            cpu.ground_energy(),
            gpu.ground_energy()
        );
    }
}
