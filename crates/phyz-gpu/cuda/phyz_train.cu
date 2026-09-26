// phyz PPO update kernels, CUDA C edition.
//
// The companion of `phyz_kernels.cu`. That file runs the *collection* half of
// a reinforcement-learning iteration on device — PD, contact, ABA, integrate,
// FK, observation, policy. This file runs the *update* half: the minibatch
// loop of a clipped-surrogate PPO step over a three-layer tanh MLP actor and
// a three-layer tanh MLP critic, with Adam.
//
// It compiles the same two ways, from one text:
//
//   * On device, through NVRTC at runtime. `__CUDACC_RTC__` is defined and
//     the `phyz_train_*` `__global__` entry points are real kernels.
//   * On the host, as plain C++ (build.rs, feature `cuda-host`), with the
//     thread bodies walked serially by the `phyz_train_host_*` loops. This is
//     how the port is pinned against the f64 CPU reference on machines with
//     no NVIDIA GPU at all.
//
// # Precision
//
// Activations, gradients and the forward weights are `float`. The Adam state
// — parameters, first and second moment — is `double`, exactly as tang's
// `ModuleAdam` keeps `m`/`v` in `f64` regardless of the parameter scalar.
// Every `train_adam` step writes both the `double` master parameter and the
// `float` mirror the forward pass reads. The master weights are therefore
// never rounded through `float` between steps, and a long run does not drift
// the way a pure-f32 optimizer would; only the gradient carries f32 error.
//
// # Reductions
//
// Every accumulation over the minibatch — weight gradients, bias gradients,
// the loss and KL scalars — is a *sequential* loop inside one thread, in row
// order, with a `double` accumulator. That costs some parallelism at these
// sizes (a weight gradient is one thread per weight, thousands of threads,
// each walking the batch) and buys two things worth more: bit-identical
// results between the device and the host walk of the same source, and an
// accumulation error that does not depend on how the hardware happened to
// schedule the block.

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define PHYZ_ON_DEVICE 1
#define PHYZ_DEV __device__ __forceinline__
#else
#define PHYZ_ON_DEVICE 0
#include <math.h>
#define PHYZ_DEV static inline
#endif

typedef unsigned int u32;

// Widest hidden layer a thread may hold in registers/local memory. The actor
// and critic of a locomotion policy sit at 32-128; this is checked host-side
// at pipeline construction, not here.
#define TRAIN_MAX_H 256
// Widest action dimension a per-row thread may hold.
#define TRAIN_MAX_OUT 64

PHYZ_DEV float fclampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ═══════════════════════════════════════════════════════════════════════════
// Parameter layout
// ═══════════════════════════════════════════════════════════════════════════
//
// One flat array per network, in tang's `Mlp::to_flat` order, so a host that
// speaks tang can upload its weights with a memcpy and read them back the
// same way:
//
//   W1 [n_h * n_in]   row-major, W1[j * n_in + i]
//   b1 [n_h]
//   W2 [n_h * n_h]
//   b2 [n_h]
//   W3 [n_out * n_h]
//   b3 [n_out]
//
// The actor's flat array carries one more block after `b3`:
//
//   log_std [n_out]
//
// which is a leaf parameter of the actor, not of the MLP. Adam is elementwise
// and tang shares one step counter across the whole `params` slice it is
// handed, so treating the concatenation as a single array is not an
// approximation of tang's behaviour — it is the same arithmetic.

PHYZ_DEV u32 off_b1(u32 n_in, u32 n_h) { return n_h * n_in; }
PHYZ_DEV u32 off_W2(u32 n_in, u32 n_h) { return n_h * n_in + n_h; }
PHYZ_DEV u32 off_b2(u32 n_in, u32 n_h) { return n_h * n_in + n_h + n_h * n_h; }
PHYZ_DEV u32 off_W3(u32 n_in, u32 n_h) { return n_h * n_in + n_h + n_h * n_h + n_h; }
PHYZ_DEV u32 off_b3(u32 n_in, u32 n_h, u32 n_out) {
    return n_h * n_in + n_h + n_h * n_h + n_h + n_out * n_h;
}
// The actor's `log_std` block follows `b3`; only the host addresses it, by
// passing an already-offset pointer, so there is no device-side accessor.

// ═══════════════════════════════════════════════════════════════════════════
// Forward
// ═══════════════════════════════════════════════════════════════════════════

// One thread per minibatch row. Reads row `idx[row]` of `x` (the whole
// iteration's packed batch stays resident; the minibatch is an index list,
// never a copy), and writes the two hidden activations and the output.
PHYZ_DEV void fwd_thread(u32 row, u32 b, u32 n_in, u32 n_h, u32 n_out,
                         const float* w, const float* x, const u32* idx,
                         float* h1, float* h2, float* out) {
    if (row >= b) return;
    const float* W1 = w;
    const float* b1 = w + off_b1(n_in, n_h);
    const float* W2 = w + off_W2(n_in, n_h);
    const float* b2 = w + off_b2(n_in, n_h);
    const float* W3 = w + off_W3(n_in, n_h);
    const float* b3 = w + off_b3(n_in, n_h, n_out);
    const float* xr = x + (unsigned long long)idx[row] * n_in;

    float a1[TRAIN_MAX_H];
    for (u32 j = 0; j < n_h; j++) {
        float s = b1[j];
        for (u32 i = 0; i < n_in; i++) s += W1[j * n_in + i] * xr[i];
        a1[j] = tanhf(s);
        h1[row * n_h + j] = a1[j];
    }
    float a2[TRAIN_MAX_H];
    for (u32 j = 0; j < n_h; j++) {
        float s = b2[j];
        for (u32 i = 0; i < n_h; i++) s += W2[j * n_h + i] * a1[i];
        a2[j] = tanhf(s);
        h2[row * n_h + j] = a2[j];
    }
    for (u32 k = 0; k < n_out; k++) {
        float s = b3[k];
        for (u32 i = 0; i < n_h; i++) s += W3[k * n_h + i] * a2[i];
        out[row * n_out + k] = s;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PPO policy gradient
// ═══════════════════════════════════════════════════════════════════════════
//
// One thread per row. `mean` is the actor's output for the row; `act`,
// `logp_old` and `adv` are the row's stored sample. Writes:
//
//   dmean   [b, n_out]  dL/dmean, the actor MLP's output gradient
//   dlogstd [b, n_out]  per-row dL/dlogstd, reduced by `grad_b` afterwards
//   stats   [b, 4]      col 0 policy loss, col 1 (logp_old - logp),
//                       col 2 Schulman k3, col 3 left for the critic
//
// This is the CPU reference term for term, including the detail that the
// gradient flows only through whichever surrogate branch is the active
// (minimum) one, and that the entropy bonus is a constant -1 per dimension
// added to dlogstd — folded in here rather than in a second pass so the
// entropy coefficient does not need its own launch.
PHYZ_DEV void ppo_grad_thread(u32 row, u32 b, u32 n_out, float clip, float entropy_coef,
                              const float* mean, const float* logstd,
                              const float* act, const float* logp_old, const float* adv,
                              const u32* idx,
                              float* dmean, float* dlogstd, float* stats) {
    if (row >= b) return;
    u32 s = idx[row];
    const float LN_2PI = 1.8378770664093453f;

    float sd[TRAIN_MAX_OUT];
    for (u32 d = 0; d < n_out; d++) sd[d] = expf(logstd[d]);

    float logp = 0.0f;
    for (u32 d = 0; d < n_out; d++) {
        float m = mean[row * n_out + d];
        float z = (act[(unsigned long long)s * n_out + d] - m) / sd[d];
        logp += -0.5f * z * z - logf(sd[d]) - 0.5f * LN_2PI;
    }
    float a = adv[s];
    float log_ratio = logp - logp_old[s];
    float ratio = expf(log_ratio);
    float clipped = fclampf(ratio, 1.0f - clip, 1.0f + clip);
    float unclipped_obj = ratio * a;
    float clipped_obj = clipped * a;
    float ploss = -(unclipped_obj < clipped_obj ? unclipped_obj : clipped_obj);

    stats[row * 4 + 0] = ploss;
    stats[row * 4 + 1] = logp_old[s] - logp;
    stats[row * 4 + 2] = ratio - 1.0f - log_ratio;

    int active = unclipped_obj <= clipped_obj + 1e-12f;
    for (u32 d = 0; d < n_out; d++) {
        if (!active) {
            dmean[row * n_out + d] = 0.0f;
            dlogstd[row * n_out + d] = -entropy_coef / (float)b;
            continue;
        }
        float coef = -a * ratio / (float)b;
        float m = mean[row * n_out + d];
        float z = (act[(unsigned long long)s * n_out + d] - m) / sd[d];
        // d logp / d mean = z / std; d logp / d logstd = z^2 - 1.
        dmean[row * n_out + d] = coef * (z / sd[d]);
        dlogstd[row * n_out + d] = coef * (z * z - 1.0f) - entropy_coef / (float)b;
    }
}

// Critic: squared error to the GAE return, with the *gradient* clipped to a
// Huber delta past which one absurd target cannot drag the value head. The
// reported loss stays the plain squared error, so the logged number is
// comparable across runs with and without the clip.
PHYZ_DEV void value_grad_thread(u32 row, u32 b, float vdelta,
                                const float* values, const float* ret, const u32* idx,
                                float* dv, float* stats) {
    if (row >= b) return;
    u32 s = idx[row];
    float err = values[row] - ret[s];
    stats[row * 4 + 3] = err * err;
    float d = fclampf(err, -vdelta, vdelta);
    dv[row] = 2.0f * d / (float)b;
}

// ═══════════════════════════════════════════════════════════════════════════
// Backward
// ═══════════════════════════════════════════════════════════════════════════

// Backprop one gradient through a weight matrix and a tanh: given `dout`
// [b, n_out] and `W` [n_out, n_h] and the *post-activation* `hact` [b, n_h],
// write `dh[row, j] = (sum_k dout[row,k] * W[k,j]) * (1 - hact[row,j]^2)`.
// One thread per (row, j).
PHYZ_DEV void bwd_hidden_thread(u32 t, u32 b, u32 n_out, u32 n_h,
                                const float* dout, const float* W, const float* hact,
                                float* dh) {
    if (t >= b * n_h) return;
    u32 row = t / n_h;
    u32 j = t - row * n_h;
    float s = 0.0f;
    for (u32 k = 0; k < n_out; k++) s += dout[row * n_out + k] * W[k * n_h + j];
    float a = hact[row * n_h + j];
    dh[row * n_h + j] = s * (1.0f - a * a);
}

// Weight gradient of a linear layer whose input is a *row-indexed* batch
// buffer: gW[o, i] = sum_row dout[row, o] * x[idx[row], i]. One thread per
// weight, walking the batch sequentially in a double accumulator.
PHYZ_DEV void grad_w_idx_thread(u32 t, u32 b, u32 n_out, u32 n_in,
                                const float* dout, const float* x, const u32* idx,
                                float* gw, u32 gw_off) {
    if (t >= n_out * n_in) return;
    u32 o = t / n_in;
    u32 i = t - o * n_in;
    double acc = 0.0;
    for (u32 r = 0; r < b; r++)
        acc += (double)dout[r * n_out + o] * (double)x[(unsigned long long)idx[r] * n_in + i];
    gw[gw_off + o * n_in + i] = (float)acc;
}

// Same, for a layer whose input is a dense per-row activation buffer.
PHYZ_DEV void grad_w_thread(u32 t, u32 b, u32 n_out, u32 n_in,
                            const float* dout, const float* x, float* gw, u32 gw_off) {
    if (t >= n_out * n_in) return;
    u32 o = t / n_in;
    u32 i = t - o * n_in;
    double acc = 0.0;
    for (u32 r = 0; r < b; r++)
        acc += (double)dout[r * n_out + o] * (double)x[r * n_in + i];
    gw[gw_off + o * n_in + i] = (float)acc;
}

// Bias gradient: gb[o] = sum_row dout[row, o]. Doubles as the reducer for the
// actor's per-row `dlogstd`.
PHYZ_DEV void grad_b_thread(u32 t, u32 b, u32 n_out,
                            const float* dout, float* gb, u32 gb_off) {
    if (t >= n_out) return;
    double acc = 0.0;
    for (u32 r = 0; r < b; r++) acc += (double)dout[r * n_out + t];
    gb[gb_off + t] = (float)acc;
}

// ═══════════════════════════════════════════════════════════════════════════
// Adam
// ═══════════════════════════════════════════════════════════════════════════
//
// tang's `ModuleAdam::step`, elementwise, with the bias-correction factors
// computed host-side from the shared step counter. `p` is the f64 master
// parameter, `w` the f32 mirror the forward pass reads.
PHYZ_DEV void adam_thread(u32 t, u32 n, double lr, double beta1, double beta2,
                          double eps, double weight_decay, double bc1, double bc2,
                          const float* g, double* m, double* v, double* p, float* w) {
    if (t >= n) return;
    double x = p[t];
    if (weight_decay > 0.0) x = x * (1.0 - lr * weight_decay);
    double gr = (double)g[t];
    double mt = beta1 * m[t] + (1.0 - beta1) * gr;
    double vt = beta2 * v[t] + (1.0 - beta2) * gr * gr;
    m[t] = mt;
    v[t] = vt;
    double m_hat = mt / bc1;
    double v_hat = vt / bc2;
    x -= lr * m_hat / (sqrt(v_hat) + eps);
    p[t] = x;
    w[t] = (float)x;
}

// Sum the four statistic columns over the minibatch, one thread per column,
// sequentially in a double. `out` is four floats the host reads back — the
// only readback in the inner loop, and it is there because the KL brake has
// to be a host-side decision.
PHYZ_DEV void reduce_stats_thread(u32 t, u32 b, const float* stats, float* out) {
    if (t >= 4) return;
    double acc = 0.0;
    for (u32 r = 0; r < b; r++) acc += (double)stats[r * 4 + t];
    out[t] = (float)acc;
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

#if PHYZ_ON_DEVICE

extern "C" __global__ void phyz_train_fwd(u32 b, u32 n_in, u32 n_h, u32 n_out,
                                          const float* w, const float* x, const u32* idx,
                                          float* h1, float* h2, float* out) {
    fwd_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_in, n_h, n_out, w, x, idx, h1, h2, out);
}

extern "C" __global__ void phyz_train_ppo_grad(u32 b, u32 n_out, float clip, float entropy_coef,
                                               const float* mean, const float* logstd,
                                               const float* act, const float* logp_old,
                                               const float* adv, const u32* idx,
                                               float* dmean, float* dlogstd, float* stats) {
    ppo_grad_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_out, clip, entropy_coef,
                    mean, logstd, act, logp_old, adv, idx, dmean, dlogstd, stats);
}

extern "C" __global__ void phyz_train_value_grad(u32 b, float vdelta, const float* values,
                                                 const float* ret, const u32* idx,
                                                 float* dv, float* stats) {
    value_grad_thread(blockIdx.x * blockDim.x + threadIdx.x, b, vdelta, values, ret, idx, dv, stats);
}

extern "C" __global__ void phyz_train_bwd_hidden(u32 b, u32 n_out, u32 n_h, const float* dout,
                                                 const float* W, const float* hact, float* dh) {
    bwd_hidden_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_out, n_h, dout, W, hact, dh);
}

extern "C" __global__ void phyz_train_grad_w_idx(u32 b, u32 n_out, u32 n_in, const float* dout,
                                                 const float* x, const u32* idx,
                                                 float* gw, u32 gw_off) {
    grad_w_idx_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_out, n_in, dout, x, idx, gw, gw_off);
}

extern "C" __global__ void phyz_train_grad_w(u32 b, u32 n_out, u32 n_in, const float* dout,
                                             const float* x, float* gw, u32 gw_off) {
    grad_w_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_out, n_in, dout, x, gw, gw_off);
}

extern "C" __global__ void phyz_train_grad_b(u32 b, u32 n_out, const float* dout,
                                             float* gb, u32 gb_off) {
    grad_b_thread(blockIdx.x * blockDim.x + threadIdx.x, b, n_out, dout, gb, gb_off);
}

extern "C" __global__ void phyz_train_adam(u32 n, double lr, double beta1, double beta2,
                                           double eps, double weight_decay, double bc1, double bc2,
                                           const float* g, double* m, double* v,
                                           double* p, float* w) {
    adam_thread(blockIdx.x * blockDim.x + threadIdx.x, n, lr, beta1, beta2, eps, weight_decay,
                bc1, bc2, g, m, v, p, w);
}

extern "C" __global__ void phyz_train_reduce_stats(u32 b, const float* stats, float* out) {
    reduce_stats_thread(blockIdx.x * blockDim.x + threadIdx.x, b, stats, out);
}

#else  // host: walk the grid serially

extern "C" void phyz_train_host_fwd(u32 n_threads, u32 b, u32 n_in, u32 n_h, u32 n_out,
                                    const float* w, const float* x, const u32* idx,
                                    float* h1, float* h2, float* out) {
    for (u32 t = 0; t < n_threads; t++)
        fwd_thread(t, b, n_in, n_h, n_out, w, x, idx, h1, h2, out);
}

extern "C" void phyz_train_host_ppo_grad(u32 n_threads, u32 b, u32 n_out, float clip,
                                         float entropy_coef, const float* mean,
                                         const float* logstd, const float* act,
                                         const float* logp_old, const float* adv, const u32* idx,
                                         float* dmean, float* dlogstd, float* stats) {
    for (u32 t = 0; t < n_threads; t++)
        ppo_grad_thread(t, b, n_out, clip, entropy_coef, mean, logstd, act, logp_old, adv, idx,
                        dmean, dlogstd, stats);
}

extern "C" void phyz_train_host_value_grad(u32 n_threads, u32 b, float vdelta,
                                           const float* values, const float* ret, const u32* idx,
                                           float* dv, float* stats) {
    for (u32 t = 0; t < n_threads; t++)
        value_grad_thread(t, b, vdelta, values, ret, idx, dv, stats);
}

extern "C" void phyz_train_host_bwd_hidden(u32 n_threads, u32 b, u32 n_out, u32 n_h,
                                           const float* dout, const float* W, const float* hact,
                                           float* dh) {
    for (u32 t = 0; t < n_threads; t++)
        bwd_hidden_thread(t, b, n_out, n_h, dout, W, hact, dh);
}

extern "C" void phyz_train_host_grad_w_idx(u32 n_threads, u32 b, u32 n_out, u32 n_in,
                                           const float* dout, const float* x, const u32* idx,
                                           float* gw, u32 gw_off) {
    for (u32 t = 0; t < n_threads; t++)
        grad_w_idx_thread(t, b, n_out, n_in, dout, x, idx, gw, gw_off);
}

extern "C" void phyz_train_host_grad_w(u32 n_threads, u32 b, u32 n_out, u32 n_in,
                                       const float* dout, const float* x, float* gw, u32 gw_off) {
    for (u32 t = 0; t < n_threads; t++)
        grad_w_thread(t, b, n_out, n_in, dout, x, gw, gw_off);
}

extern "C" void phyz_train_host_grad_b(u32 n_threads, u32 b, u32 n_out, const float* dout,
                                       float* gb, u32 gb_off) {
    for (u32 t = 0; t < n_threads; t++) grad_b_thread(t, b, n_out, dout, gb, gb_off);
}

extern "C" void phyz_train_host_adam(u32 n_threads, u32 n, double lr, double beta1, double beta2,
                                     double eps, double weight_decay, double bc1, double bc2,
                                     const float* g, double* m, double* v, double* p, float* w) {
    for (u32 t = 0; t < n_threads; t++)
        adam_thread(t, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2, g, m, v, p, w);
}

extern "C" void phyz_train_host_reduce_stats(u32 n_threads, u32 b, const float* stats, float* out) {
    for (u32 t = 0; t < n_threads; t++) reduce_stats_thread(t, b, stats, out);
}

#endif
