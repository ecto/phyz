//! WGSL kernels for a batched MLP (forward, input and weight gradients),
//! shared by the GPU trainers. Layout: one environment per row, weights
//! `[out][in]` then biases, layer after layer (the CPU `Mlp` layout).

pub const WG: u32 = 256;

/// `(weights, bias)` offsets of each layer in the parameter vector (the CPU
/// `Mlp` layout: weights `[out][in]` then biases, layer after layer).
pub fn layer_offsets(layer: &[(usize, usize); 3]) -> [(usize, usize); 3] {
    let mut o = 0;
    std::array::from_fn(|l| {
        let (a, n) = layer[l];
        let r = (o, o + a * n);
        o += a * n + n;
        r
    })
}

/// Storage bindings (read-only unless in `rw`) then the uniform params.
pub fn header(names: &[&str], rw: &[usize]) -> String {
    let mut s = String::from("struct P { count: u32, a: u32, b: u32, c: u32 }\n");
    for (i, n) in names.iter().enumerate() {
        let mode = if rw.contains(&i) {
            "read_write"
        } else {
            "read"
        };
        s += &format!("@group(0) @binding({i}) var<storage, {mode}> {n}: array<f32>;\n");
    }
    s += &format!("@group(0) @binding({}) var<uniform> p: P;\n", names.len());
    s += &format!(
        "@compute @workgroup_size({WG})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let k = gid.x;\n    if (k >= p.count) {{ return; }}\n"
    );
    s
}

/// `G[W[j, i]] += Σ_e gz[e, j] x[e, i]`, then the biases: one workgroup
/// per weight, its lanes striding over the environments, then a tree sum in
/// workgroup memory. Launch `len * WG` threads (a multiple of `WG`, so no
/// lane leaves before the barriers).
pub fn bwd_w_kernel((a, n): (usize, usize), (w, b): (usize, usize), envs: usize) -> String {
    let an = a * n;
    header(&["gz", "x", "g"], &[2])
        .replace(
            "@compute",
            &format!("var<workgroup> part: array<f32, {WG}>;\n@compute"),
        )
        .replace(
            "gid: vec3<u32>) {",
            "gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {",
        )
        + &format!(
            "    let wk = k / {WG}u; let lane = lid.x;\n\
         var s = 0.0;\n\
         if (wk < {an}u) {{\n\
             let j = wk / {a}u; let i = wk % {a}u;\n\
             for (var e = lane; e < {envs}u; e += {WG}u) {{ s += gz[e * {n}u + j] * x[e * {a}u + i]; }}\n\
         }} else {{\n\
             let j = wk - {an}u;\n\
             for (var e = lane; e < {envs}u; e += {WG}u) {{ s += gz[e * {n}u + j]; }}\n\
         }}\n\
         part[lane] = s;\n\
         workgroupBarrier();\n\
         for (var h = {}u; h > 0u; h = h / 2u) {{\n\
             if (lane < h) {{ part[lane] += part[lane + h]; }}\n\
             workgroupBarrier();\n\
         }}\n\
         if (lane == 0u) {{\n\
             if (wk < {an}u) {{ g[{w}u + wk] += part[0]; }} else {{ g[{b}u + wk - {an}u] += part[0]; }}\n\
         }}\n}}\n",
            WG / 2
        )
}

/// `gx[e, i] = Σ_j gz[e, j] W[j, i]`, times `1 - h²` when the input `h` is a
/// tanh activation (the previous layer's pre-activation gradient).
pub fn bwd_x_kernel((a, n): (usize, usize), (w, _): (usize, usize), tanh_in: bool) -> String {
    let scale = if tanh_in {
        "(1.0 - h[k] * h[k])"
    } else {
        "1.0"
    };
    header(&["w", "gz", "h", "gx"], &[3])
        + &format!(
            "    let e = k / {a}u; let i = k % {a}u;\n\
             var s = 0.0;\n\
             for (var j = 0u; j < {n}u; j++) {{ s += gz[e * {n}u + j] * w[{w}u + j * {a}u + i]; }}\n\
             gx[k] = s * {scale};\n}}\n"
        )
}

/// `store = act` with `z = b[j] + Σ_i W[j, i] x[e, i]` for `k = e * n + j`
/// (`store` and `act` are WGSL using `e`, `j`, `k`, `z`).
pub fn dense_kernel(
    (a, n): (usize, usize),
    (w, b): (usize, usize),
    store: &str,
    act: &str,
) -> String {
    header(&["w", "x", "y"], &[2])
        + &format!(
            "    let e = k / {n}u; let j = k % {n}u;\n\
             var z = w[{b}u + j];\n\
             for (var i = 0u; i < {a}u; i++) {{ z += w[{w}u + j * {a}u + i] * x[e * {a}u + i]; }}\n\
             {store} = {act};\n}}\n"
        )
}
