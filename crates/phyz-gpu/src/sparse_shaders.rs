//! WGSL compute shaders for sparse linear algebra (SpMV, dot, axpy, etc.).
//!
//! Each operation has f64 and f32 variants. The f64 variants use
//! `enable f64;` WGSL extension (supported on Metal/Apple Silicon, most Vulkan).
//!
//! All per-element shaders use grid-stride loops to support dim > 65535 * 256.
//! Phase2 reduction shaders use serial accumulation to handle > 256 partials.

/// SpMV: y = A * x, grid-stride over rows.
pub const SPMV_F64: &str = r#"
enable f64;

@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> x: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> dim: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var row = gid.x; row < dim; row += n_threads) {
        let start = row_ptr[row];
        let end = row_ptr[row + 1u];

        var sum: f64 = 0.0;
        for (var k = start; k < end; k++) {
            let col = col_idx[k];
            let a = bitcast<f64>(vals[k]);
            let b = bitcast<f64>(x[col]);
            sum += a * b;
        }

        y[row] = bitcast<vec2<u32>>(sum);
    }
}
"#;

pub const SPMV_F32: &str = r#"
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> dim: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var row = gid.x; row < dim; row += n_threads) {
        let start = row_ptr[row];
        let end = row_ptr[row + 1u];

        var sum: f32 = 0.0;
        for (var k = start; k < end; k++) {
            let col = col_idx[k];
            sum += vals[k] * x[col];
        }

        y[row] = sum;
    }
}
"#;

/// AXPY: y += alpha * x
pub const AXPY_F64: &str = r#"
enable f64;

struct Params { dim: u32, _pad: u32, alpha: vec2<u32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let alpha = bitcast<f64>(params.alpha);
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        let xi = bitcast<f64>(x[i]);
        let yi = bitcast<f64>(y[i]);
        y[i] = bitcast<vec2<u32>>(yi + alpha * xi);
    }
}
"#;

pub const AXPY_F32: &str = r#"
struct Params { dim: u32, alpha: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        y[i] += params.alpha * x[i];
    }
}
"#;

/// SCALE: x *= alpha
pub const SCALE_F64: &str = r#"
enable f64;

struct Params { dim: u32, _pad: u32, alpha: vec2<u32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let alpha = bitcast<f64>(params.alpha);
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        let xi = bitcast<f64>(x[i]);
        x[i] = bitcast<vec2<u32>>(alpha * xi);
    }
}
"#;

pub const SCALE_F32: &str = r#"
struct Params { dim: u32, alpha: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        x[i] *= params.alpha;
    }
}
"#;

/// DOT phase 1: partial sums via shared memory tree reduction.
/// Each workgroup accumulates via grid-stride, then reduces 256 elements.
pub const DOT_PHASE1_F64: &str = r#"
enable f64;

@group(0) @binding(0) var<storage, read> a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> dim: u32;

var<workgroup> wg_shmem: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    var val: f64 = 0.0;
    for (var i = gid.x; i < dim; i += n_threads) {
        val += bitcast<f64>(a[i]) * bitcast<f64>(b[i]);
    }
    wg_shmem[lid.x] = bitcast<vec2<u32>>(val);
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            let l = bitcast<f64>(wg_shmem[lid.x]);
            let r = bitcast<f64>(wg_shmem[lid.x + stride]);
            wg_shmem[lid.x] = bitcast<vec2<u32>>(l + r);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        partials[wid.x] = wg_shmem[0];
    }
}
"#;

pub const DOT_PHASE1_F32: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@group(0) @binding(3) var<uniform> dim: u32;

var<workgroup> wg_shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    var val: f32 = 0.0;
    for (var i = gid.x; i < dim; i += n_threads) {
        val += a[i] * b[i];
    }
    wg_shmem[lid.x] = val;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            wg_shmem[lid.x] += wg_shmem[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        partials[wid.x] = wg_shmem[0];
    }
}
"#;

/// DOT phase 2: reduce partial sums to a single scalar.
/// Single workgroup launch. Serial accumulation handles > 256 partials.
pub const DOT_PHASE2_F64: &str = r#"
enable f64;

@group(0) @binding(0) var<storage, read> partials: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> result: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> n_partials: u32;

var<workgroup> wg_shmem: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    var val: f64 = 0.0;
    for (var k = lid.x; k < n_partials; k += 256u) {
        val += bitcast<f64>(partials[k]);
    }
    wg_shmem[lid.x] = bitcast<vec2<u32>>(val);
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            let l = bitcast<f64>(wg_shmem[lid.x]);
            let r = bitcast<f64>(wg_shmem[lid.x + stride]);
            wg_shmem[lid.x] = bitcast<vec2<u32>>(l + r);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[0] = wg_shmem[0];
    }
}
"#;

pub const DOT_PHASE2_F32: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> n_partials: u32;

var<workgroup> wg_shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    var val: f32 = 0.0;
    for (var k = lid.x; k < n_partials; k += 256u) {
        val += partials[k];
    }
    wg_shmem[lid.x] = val;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            wg_shmem[lid.x] += wg_shmem[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[0] = wg_shmem[0];
    }
}
"#;

/// Multi-dot: compute dot(q_bank[k], w) for k=0..n_vecs in one dispatch.
/// 2D dispatch: X = element workgroups, Y = vector index.
/// Grid-stride over elements within each workgroup.
pub const MULTI_DOT_PHASE1_F64: &str = r#"
enable f64;

struct Params { dim: u32, n_vecs: u32, stride: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_bank: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> w: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec2<u32>>;

var<workgroup> wg_shmem: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let vec_idx = wid.y;
    let n_threads_x = nwg.x * 256u;

    var val: f64 = 0.0;
    if (vec_idx < params.n_vecs) {
        for (var elem = gid.x; elem < params.dim; elem += n_threads_x) {
            let q_offset = vec_idx * params.stride + elem;
            val += bitcast<f64>(q_bank[q_offset]) * bitcast<f64>(w[elem]);
        }
    }
    wg_shmem[lid.x] = bitcast<vec2<u32>>(val);
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            let l = bitcast<f64>(wg_shmem[lid.x]);
            let r = bitcast<f64>(wg_shmem[lid.x + s]);
            wg_shmem[lid.x] = bitcast<vec2<u32>>(l + r);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        partials[wid.x + vec_idx * nwg.x] = wg_shmem[0];
    }
}
"#;

pub const MULTI_DOT_PHASE1_F32: &str = r#"
struct Params { dim: u32, n_vecs: u32, stride: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_bank: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> wg_shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let vec_idx = wid.y;
    let n_threads_x = nwg.x * 256u;

    var val: f32 = 0.0;
    if (vec_idx < params.n_vecs) {
        for (var elem = gid.x; elem < params.dim; elem += n_threads_x) {
            let q_offset = vec_idx * params.stride + elem;
            val += q_bank[q_offset] * w[elem];
        }
    }
    wg_shmem[lid.x] = val;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            wg_shmem[lid.x] += wg_shmem[lid.x + s];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        partials[wid.x + vec_idx * nwg.x] = wg_shmem[0];
    }
}
"#;

/// Multi-dot phase 2: reduce partials per vector.
/// One workgroup per vector. Serial accumulation handles > 256 partials.
pub const MULTI_DOT_PHASE2_F64: &str = r#"
enable f64;

struct Params { n_partials_per_vec: u32, n_vecs: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partials: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> overlaps: array<vec2<u32>>;

var<workgroup> wg_shmem: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let vec_idx = wid.x;
    if (vec_idx >= params.n_vecs) { return; }

    var val: f64 = 0.0;
    let base = vec_idx * params.n_partials_per_vec;
    for (var k = lid.x; k < params.n_partials_per_vec; k += 256u) {
        val += bitcast<f64>(partials[base + k]);
    }
    wg_shmem[lid.x] = bitcast<vec2<u32>>(val);
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            let l = bitcast<f64>(wg_shmem[lid.x]);
            let r = bitcast<f64>(wg_shmem[lid.x + s]);
            wg_shmem[lid.x] = bitcast<vec2<u32>>(l + r);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        overlaps[vec_idx] = wg_shmem[0];
    }
}
"#;

pub const MULTI_DOT_PHASE2_F32: &str = r#"
struct Params { n_partials_per_vec: u32, n_vecs: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> overlaps: array<f32>;

var<workgroup> wg_shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let vec_idx = wid.x;
    if (vec_idx >= params.n_vecs) { return; }

    var val: f32 = 0.0;
    let base = vec_idx * params.n_partials_per_vec;
    for (var k = lid.x; k < params.n_partials_per_vec; k += 256u) {
        val += partials[base + k];
    }
    wg_shmem[lid.x] = val;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            wg_shmem[lid.x] += wg_shmem[lid.x + s];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        overlaps[vec_idx] = wg_shmem[0];
    }
}
"#;

/// Batch subtract: w -= sum_k overlaps[k] * q_bank[k*stride..k*stride+dim]
pub const BATCH_SUBTRACT_F64: &str = r#"
enable f64;

struct Params { dim: u32, n_vecs: u32, stride: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_bank: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> overlaps: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> w: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        var acc: f64 = 0.0;
        for (var k = 0u; k < params.n_vecs; k++) {
            let overlap = bitcast<f64>(overlaps[k]);
            let q_val = bitcast<f64>(q_bank[k * params.stride + i]);
            acc += overlap * q_val;
        }

        let wi = bitcast<f64>(w[i]);
        w[i] = bitcast<vec2<u32>>(wi - acc);
    }
}
"#;

pub const BATCH_SUBTRACT_F32: &str = r#"
struct Params { dim: u32, n_vecs: u32, stride: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_bank: array<f32>;
@group(0) @binding(2) var<storage, read> overlaps: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_threads = nwg.x * 256u;
    for (var i = gid.x; i < params.dim; i += n_threads) {
        var acc: f32 = 0.0;
        for (var k = 0u; k < params.n_vecs; k++) {
            acc += overlaps[k] * q_bank[k * params.stride + i];
        }

        w[i] -= acc;
    }
}
"#;
