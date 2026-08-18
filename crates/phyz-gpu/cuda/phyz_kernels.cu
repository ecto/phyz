// phyz batch-simulation kernels, CUDA C edition.
//
// This file is the CUDA counterpart of the WGSL in `src/shaders.rs` and
// `src/pd_pipeline.rs`: the same four passes (PD servo, ground contact,
// general ABA, joint-aware semi-implicit Euler) over the same flat f32
// buffers, laid out exactly as `src/layout.rs` documents. It is a hand
// port, function for function, and the parity tests against the f64 CPU
// reference are what keep the two kernel sources honest — there is no
// automatic translation between them.
//
// It compiles two ways, from one text:
//
//   * On device, through NVRTC at runtime (`cudarc::nvrtc::compile_ptx`).
//     `__CUDACC_RTC__` is defined, the `PHYZ_GLOBAL` wrappers become real
//     `__global__` entry points, and each thread handles one grid index.
//
//   * On the host, as plain C++ (build.rs, behind the `cuda-host` feature).
//     The same thread bodies are called from `phyz_host_*` loops that walk
//     the grid serially. This is how the port is verified on a machine
//     with no NVIDIA GPU at all: the *exact* source that NVRTC will
//     compile runs against phyz's CPU dynamics.
//
// Keep it in the common subset: no CUDA vector types, no <math.h> beyond
// what NVRTC provides as builtins, no C++ features NVRTC rejects.

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define PHYZ_ON_DEVICE 1
#define PHYZ_DEV __device__ __forceinline__
#else
#define PHYZ_ON_DEVICE 0
#include <math.h>
#include <string.h>
#define PHYZ_DEV static inline
#endif

typedef unsigned int u32;
typedef int i32;

// Mirrors layout.rs.
#define MAX_BODIES 32u
#define BODY_STRIDE 32u
#define GEOM_STRIDE 16u
#define CS_STRIDE 8u
#define PD_DOF_STRIDE 8u

// ── Scalar helpers ─────────────────────────────────────────────────────────

PHYZ_DEV i32 f_as_i(float f) {
#if PHYZ_ON_DEVICE
    return __float_as_int(f);
#else
    i32 i;
    memcpy(&i, &f, 4);
    return i;
#endif
}

PHYZ_DEV u32 f_as_u(float f) {
#if PHYZ_ON_DEVICE
    return __float_as_uint(f);
#else
    u32 u;
    memcpy(&u, &f, 4);
    return u;
#endif
}

PHYZ_DEV float fmax_(float a, float b) { return a > b ? a : b; }
PHYZ_DEV float fmin_(float a, float b) { return a < b ? a : b; }
PHYZ_DEV float fclamp(float x, float lo, float hi) { return fmin_(fmax_(x, lo), hi); }
PHYZ_DEV float fabs_(float a) { return a < 0.0f ? -a : a; }

// ── Small value types ──────────────────────────────────────────────────────

struct v3 { float x, y, z; };
struct v4 { float x, y, z, w; };  // quaternion as (w, x, y, z) is stored x=w, y=x, z=y, w=z — see qw()/qx() below
struct sv6 { float a[6]; };
struct r9 { float r[9]; };
struct m66 { float m[36]; };
struct xf { r9 rot; v3 pos; };

PHYZ_DEV v3 v3_(float x, float y, float z) { v3 r; r.x = x; r.y = y; r.z = z; return r; }
PHYZ_DEV v3 v3_add(v3 a, v3 b) { return v3_(a.x + b.x, a.y + b.y, a.z + b.z); }
PHYZ_DEV v3 v3_sub(v3 a, v3 b) { return v3_(a.x - b.x, a.y - b.y, a.z - b.z); }
PHYZ_DEV v3 v3_scale(v3 a, float s) { return v3_(a.x * s, a.y * s, a.z * s); }
PHYZ_DEV float v3_dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PHYZ_DEV float v3_len(v3 a) { return sqrtf(v3_dot(a, a)); }
PHYZ_DEV v3 cross3(v3 a, v3 b) {
    return v3_(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Quaternion (w, x, y, z) packed into v4 as (x=w, y=x, z=y, w=z), matching
// the WGSL vec4 convention so the maths reads the same in both files.
PHYZ_DEV v4 v4_(float w, float x, float y, float z) { v4 r; r.x = w; r.y = x; r.z = y; r.w = z; return r; }

PHYZ_DEV sv6 sv_zero() { sv6 r; for (u32 i = 0; i < 6; i++) r.a[i] = 0.0f; return r; }
PHYZ_DEV sv6 sv_(float a0, float a1, float a2, float a3, float a4, float a5) {
    sv6 r; r.a[0] = a0; r.a[1] = a1; r.a[2] = a2; r.a[3] = a3; r.a[4] = a4; r.a[5] = a5; return r;
}
PHYZ_DEV float sv_dot(sv6 a, sv6 b) {
    return a.a[0]*b.a[0] + a.a[1]*b.a[1] + a.a[2]*b.a[2] + a.a[3]*b.a[3] + a.a[4]*b.a[4] + a.a[5]*b.a[5];
}
PHYZ_DEV sv6 sv_add(sv6 a, sv6 b) { sv6 r; for (u32 i = 0; i < 6; i++) r.a[i] = a.a[i] + b.a[i]; return r; }
PHYZ_DEV sv6 sv_sub(sv6 a, sv6 b) { sv6 r; for (u32 i = 0; i < 6; i++) r.a[i] = a.a[i] - b.a[i]; return r; }
PHYZ_DEV sv6 sv_scale(sv6 a, float s) { sv6 r; for (u32 i = 0; i < 6; i++) r.a[i] = a.a[i] * s; return r; }
PHYZ_DEV v3 sv_ang(sv6 s) { return v3_(s.a[0], s.a[1], s.a[2]); }
PHYZ_DEV v3 sv_lin(sv6 s) { return v3_(s.a[3], s.a[4], s.a[5]); }
PHYZ_DEV sv6 sv_from(v3 ang, v3 lin) { return sv_(ang.x, ang.y, ang.z, lin.x, lin.y, lin.z); }

// Spatial motion cross product: v_m x w
PHYZ_DEV sv6 sv_cross_motion(sv6 v, sv6 w) {
    v3 va = sv_ang(v), vl = sv_lin(v), wa = sv_ang(w), wl = sv_lin(w);
    v3 ra = cross3(va, wa);
    v3 rl = v3_add(cross3(va, wl), cross3(vl, wa));
    return sv_from(ra, rl);
}

// Spatial force cross product: v_m x* f
PHYZ_DEV sv6 sv_cross_force(sv6 v, sv6 f) {
    v3 va = sv_ang(v), vl = sv_lin(v), fa = sv_ang(f), fl = sv_lin(f);
    v3 ra = v3_add(cross3(va, fa), cross3(vl, fl));
    v3 rl = cross3(va, fl);
    return sv_from(ra, rl);
}

// ── 6x6 matrices (column-major, 36 floats) ────────────────────────────────

PHYZ_DEV m66 m6_zero() { m66 r; for (u32 i = 0; i < 36; i++) r.m[i] = 0.0f; return r; }

PHYZ_DEV sv6 m6_mul_vec(const m66* m, sv6 v) {
    sv6 r;
    for (u32 i = 0; i < 6; i++) {
        float s = 0.0f;
        for (u32 j = 0; j < 6; j++) s += m->m[j * 6 + i] * v.a[j];
        r.a[i] = s;
    }
    return r;
}

PHYZ_DEV m66 m6_add(const m66* a, const m66* b) { m66 r; for (u32 i = 0; i < 36; i++) r.m[i] = a->m[i] + b->m[i]; return r; }
PHYZ_DEV m66 m6_sub(const m66* a, const m66* b) { m66 r; for (u32 i = 0; i < 36; i++) r.m[i] = a->m[i] - b->m[i]; return r; }

// Outer product a * b^T (column-major)
PHYZ_DEV m66 m6_outer(sv6 a, sv6 b) {
    m66 r;
    for (u32 c = 0; c < 6; c++)
        for (u32 row = 0; row < 6; row++)
            r.m[c * 6 + row] = a.a[row] * b.a[c];
    return r;
}

// X^T * A * X (6x6)
PHYZ_DEV m66 m6_XtAX(const m66* xt, const m66* a, const m66* x) {
    m66 tmp;
    for (u32 c = 0; c < 6; c++)
        for (u32 r = 0; r < 6; r++) {
            float s = 0.0f;
            for (u32 k = 0; k < 6; k++) s += a->m[k * 6 + r] * x->m[c * 6 + k];
            tmp.m[c * 6 + r] = s;
        }
    m66 result;
    for (u32 c = 0; c < 6; c++)
        for (u32 r = 0; r < 6; r++) {
            float s = 0.0f;
            for (u32 k = 0; k < 6; k++) s += xt->m[k * 6 + r] * tmp.m[c * 6 + k];
            result.m[c * 6 + r] = s;
        }
    return result;
}

PHYZ_DEV m66 transpose6(const m66* m) {
    m66 t;
    for (u32 r = 0; r < 6; r++)
        for (u32 c = 0; c < 6; c++)
            t.m[c * 6 + r] = m->m[r * 6 + c];
    return t;
}

// X = [R, 0; -R*skew(p), R]
PHYZ_DEV m66 build_motion_transform(r9 rot, v3 pos) {
    m66 m = m6_zero();
    for (u32 r = 0; r < 3; r++)
        for (u32 c = 0; c < 3; c++) {
            m.m[c * 6 + r] = rot.r[r * 3 + c];
            m.m[(c + 3) * 6 + (r + 3)] = rot.r[r * 3 + c];
        }
    float px = pos.x, py = pos.y, pz = pos.z;
    // skew(p), row-major: [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
    float skp[9];
    skp[0] = 0.0f; skp[1] = -pz;  skp[2] = py;
    skp[3] = pz;   skp[4] = 0.0f; skp[5] = -px;
    skp[6] = -py;  skp[7] = px;   skp[8] = 0.0f;
    for (u32 r = 0; r < 3; r++)
        for (u32 c = 0; c < 3; c++) {
            float s = 0.0f;
            for (u32 k = 0; k < 3; k++) s += rot.r[r * 3 + k] * skp[k * 3 + c];
            m.m[c * 6 + (r + 3)] = -s;
        }
    return m;
}

// ── Rotations (row-major 3x3) ─────────────────────────────────────────────

PHYZ_DEV r9 identity_rot() {
    r9 r;
    r.r[0] = 1.0f; r.r[1] = 0.0f; r.r[2] = 0.0f;
    r.r[3] = 0.0f; r.r[4] = 1.0f; r.r[5] = 0.0f;
    r.r[6] = 0.0f; r.r[7] = 0.0f; r.r[8] = 1.0f;
    return r;
}

PHYZ_DEV v3 rot_mul(r9 r, v3 v) {
    return v3_(r.r[0]*v.x + r.r[1]*v.y + r.r[2]*v.z,
               r.r[3]*v.x + r.r[4]*v.y + r.r[5]*v.z,
               r.r[6]*v.x + r.r[7]*v.y + r.r[8]*v.z);
}

PHYZ_DEV v3 rot_tmul(r9 r, v3 v) {
    return v3_(r.r[0]*v.x + r.r[3]*v.y + r.r[6]*v.z,
               r.r[1]*v.x + r.r[4]*v.y + r.r[7]*v.z,
               r.r[2]*v.x + r.r[5]*v.y + r.r[8]*v.z);
}

// A * B
PHYZ_DEV r9 compose_rot(r9 a, r9 b) {
    r9 r;
    for (u32 row = 0; row < 3; row++)
        for (u32 col = 0; col < 3; col++) {
            float s = 0.0f;
            for (u32 k = 0; k < 3; k++) s += a.r[row * 3 + k] * b.r[k * 3 + col];
            r.r[row * 3 + col] = s;
        }
    return r;
}

PHYZ_DEV r9 transpose_rot(r9 a) {
    r9 t;
    t.r[0] = a.r[0]; t.r[1] = a.r[3]; t.r[2] = a.r[6];
    t.r[3] = a.r[1]; t.r[4] = a.r[4]; t.r[5] = a.r[7];
    t.r[6] = a.r[2]; t.r[7] = a.r[5]; t.r[8] = a.r[8];
    return t;
}

// Revolute joint rotation: Rodrigues with -angle (matches CPU joint_transform_slice)
PHYZ_DEV r9 revolute_rot(v3 axis, float angle) {
    float neg_a = -angle;
    float s = sinf(neg_a);
    float c = cosf(neg_a);
    float t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    r9 rot;
    rot.r[0] = t*x*x + c;     rot.r[1] = t*x*y - s*z;   rot.r[2] = t*x*z + s*y;
    rot.r[3] = t*x*y + s*z;   rot.r[4] = t*y*y + c;     rot.r[5] = t*y*z - s*x;
    rot.r[6] = t*x*z - s*y;   rot.r[7] = t*y*z + s*x;   rot.r[8] = t*z*z + c;
    return rot;
}

// Quaternion exponential of a rotation vector, (w, x, y, z). Matches tang's
// Quat::exp including the small-angle branch.
PHYZ_DEV v4 quat_exp(v3 omega) {
    float angle = v3_len(omega);
    if (angle < 1e-6f) return v4_(1.0f, omega.x * 0.5f, omega.y * 0.5f, omega.z * 0.5f);
    float half = angle * 0.5f;
    float s = sinf(half) / angle;
    return v4_(cosf(half), omega.x * s, omega.y * s, omega.z * s);
}

// Quaternion (w, x, y, z) to row-major rotation matrix.
PHYZ_DEV r9 quat_to_rot(v4 qt) {
    float w = qt.x, x = qt.y, y = qt.z, z = qt.w;
    r9 r;
    r.r[0] = 1.0f - 2.0f*(y*y + z*z); r.r[1] = 2.0f*(x*y - w*z);        r.r[2] = 2.0f*(x*z + w*y);
    r.r[3] = 2.0f*(x*y + w*z);        r.r[4] = 1.0f - 2.0f*(x*x + z*z); r.r[5] = 2.0f*(y*z - w*x);
    r.r[6] = 2.0f*(x*z - w*y);        r.r[7] = 2.0f*(y*z + w*x);        r.r[8] = 1.0f - 2.0f*(x*x + y*y);
    return r;
}

// Hamilton product, both (w, x, y, z).
PHYZ_DEV v4 qmul(v4 a, v4 b) {
    return v4_(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
               a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
               a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
               a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}

PHYZ_DEV v4 qnormalize(v4 q) {
    float n = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    return v4_(q.x / n, q.y / n, q.z / n, q.w / n);
}

// Matches tang's Quat::log, including the small-angle branch.
PHYZ_DEV v3 qlog(v4 qt) {
    v3 vv = v3_(qt.y, qt.z, qt.w);
    float n = v3_len(vv);
    if (n < 1e-6f) return v3_scale(vv, 2.0f);
    float angle = 2.0f * atan2f(n, qt.x);
    return v3_scale(vv, angle / n);
}

PHYZ_DEV v3 qrotate(v4 qt, v3 p) {
    v3 u = v3_(qt.y, qt.z, qt.w);
    v3 t = v3_scale(cross3(u, p), 2.0f);
    return v3_add(v3_add(p, v3_scale(t, qt.x)), cross3(u, t));
}

// self.compose(other): rot = self.rot * other.rot, pos = other.pos + other.rot^T * self.pos
PHYZ_DEV xf compose_transform(r9 self_rot, v3 self_pos, r9 other_rot, v3 other_pos) {
    xf out;
    out.rot = compose_rot(self_rot, other_rot);
    out.pos = v3_add(other_pos, rot_tmul(other_rot, self_pos));
    return out;
}

// X * v = [R*w, R*(v - p×w)]
PHYZ_DEV sv6 apply_motion(r9 rot, v3 pos, sv6 sv) {
    v3 w = sv_ang(sv);
    v3 vel = sv_lin(sv);
    v3 shifted = v3_sub(vel, cross3(pos, w));
    return sv_from(rot_mul(rot, w), rot_mul(rot, shifted));
}

// X^{-T} * f: force = R^T f, tau = R^T tau + p × (R^T f)
PHYZ_DEV sv6 inv_apply_force(r9 rot, v3 pos, sv6 fv) {
    v3 tau = sv_ang(fv);
    v3 force = sv_lin(fv);
    v3 rt_f = rot_tmul(rot, force);
    v3 rt_tau = rot_tmul(rot, tau);
    return sv_from(v3_add(rt_tau, cross3(pos, rt_f)), rt_f);
}

// I_spatial = [[I + m*cx*cx^T, m*cx], [m*cx^T, m*E]]
PHYZ_DEV m66 rigid_inertia_to_m6(float mass, v3 com, const float* inertia) {
    m66 M = m6_zero();
    float* m = M.m;
    float cx = com.x, cy = com.y, cz = com.z;
    float ixx = inertia[0], iyy = inertia[1], izz = inertia[2];
    float ixy = inertia[3], ixz = inertia[4], iyz = inertia[5];

    m[0*6+0] = ixx + mass * (cy*cy + cz*cz);
    m[1*6+0] = ixy - mass * cx * cy;
    m[2*6+0] = ixz - mass * cx * cz;
    m[0*6+1] = ixy - mass * cx * cy;
    m[1*6+1] = iyy + mass * (cx*cx + cz*cz);
    m[2*6+1] = iyz - mass * cy * cz;
    m[0*6+2] = ixz - mass * cx * cz;
    m[1*6+2] = iyz - mass * cy * cz;
    m[2*6+2] = izz + mass * (cx*cx + cy*cy);

    m[3*6+0] = 0.0f;      m[4*6+0] = -mass*cz;  m[5*6+0] = mass*cy;
    m[3*6+1] = mass*cz;   m[4*6+1] = 0.0f;      m[5*6+1] = -mass*cx;
    m[3*6+2] = -mass*cy;  m[4*6+2] = mass*cx;   m[5*6+2] = 0.0f;

    m[0*6+3] = 0.0f;      m[1*6+3] = mass*cz;   m[2*6+3] = -mass*cy;
    m[0*6+4] = -mass*cz;  m[1*6+4] = 0.0f;      m[2*6+4] = mass*cx;
    m[0*6+5] = mass*cy;   m[1*6+5] = -mass*cx;  m[2*6+5] = 0.0f;

    m[3*6+3] = mass; m[4*6+4] = mass; m[5*6+5] = mass;
    return M;
}

// ── Joints ─────────────────────────────────────────────────────────────────
// 0=revolute, 1=prismatic, 2=fixed, 3=ball (3 DOF), 4=free (6 DOF)

PHYZ_DEV u32 joint_ndof(u32 jtype) {
    if (jtype == 2u) return 0u;
    if (jtype == 3u) return 3u;
    if (jtype == 4u) return 6u;
    return 1u;
}

// Column k of the motion subspace S (6 x ndof). Mirrors
// phyz_model::Joint::motion_subspace_matrix, free joint angular-then-linear.
PHYZ_DEV sv6 subspace_col(u32 jtype, v3 axis, u32 k) {
    if (jtype == 0u) return sv_(axis.x, axis.y, axis.z, 0.0f, 0.0f, 0.0f);
    if (jtype == 1u) return sv_(0.0f, 0.0f, 0.0f, axis.x, axis.y, axis.z);
    if (jtype == 3u || jtype == 4u) { sv6 s = sv_zero(); s.a[k] = 1.0f; return s; }
    return sv_zero();
}

// In-place inverse of the leading n x n block (row-major, stride n).
// Gauss-Jordan with partial pivoting; false if singular.
PHYZ_DEV bool invert_small(float* m, u32 n) {
    float inv[36];
    for (u32 r = 0; r < n; r++)
        for (u32 c = 0; c < n; c++)
            inv[r * n + c] = (r == c) ? 1.0f : 0.0f;

    for (u32 col = 0; col < n; col++) {
        u32 piv = col;
        float best = fabs_(m[col * n + col]);
        for (u32 r = col + 1; r < n; r++) {
            float a = fabs_(m[r * n + col]);
            if (a > best) { best = a; piv = r; }
        }
        if (best < 1e-20f) return false;

        if (piv != col) {
            for (u32 c = 0; c < n; c++) {
                float t1 = m[col * n + c]; m[col * n + c] = m[piv * n + c]; m[piv * n + c] = t1;
                float t2 = inv[col * n + c]; inv[col * n + c] = inv[piv * n + c]; inv[piv * n + c] = t2;
            }
        }

        float d = m[col * n + col];
        for (u32 c = 0; c < n; c++) { m[col * n + c] /= d; inv[col * n + c] /= d; }

        for (u32 r = 0; r < n; r++) {
            if (r == col) continue;
            float f = m[r * n + col];
            if (f == 0.0f) continue;
            for (u32 c = 0; c < n; c++) {
                m[r * n + c] -= f * m[col * n + c];
                inv[r * n + c] -= f * inv[col * n + c];
            }
        }
    }
    for (u32 i = 0; i < n * n; i++) m[i] = inv[i];
    return true;
}

// Body table access
PHYZ_DEV float bf(const float* bodies, u32 bi, u32 off) { return bodies[bi * BODY_STRIDE + off]; }
PHYZ_DEV i32 body_parent(const float* bodies, u32 bi) { return f_as_i(bodies[bi * BODY_STRIDE + 0u]); }
PHYZ_DEV u32 body_jtype(const float* bodies, u32 bi) { return f_as_u(bodies[bi * BODY_STRIDE + 1u]); }
PHYZ_DEV u32 body_qoff(const float* bodies, u32 bi) { return f_as_u(bodies[bi * BODY_STRIDE + 2u]); }
PHYZ_DEV u32 body_voff(const float* bodies, u32 bi) { return f_as_u(bodies[bi * BODY_STRIDE + 3u]); }
PHYZ_DEV r9 body_ptj_rot(const float* bodies, u32 bi) { r9 r; for (u32 k = 0; k < 9; k++) r.r[k] = bf(bodies, bi, 14u + k); return r; }
PHYZ_DEV v3 body_ptj_pos(const float* bodies, u32 bi) { return v3_(bf(bodies, bi, 23u), bf(bodies, bi, 24u), bf(bodies, bi, 25u)); }
PHYZ_DEV v3 body_axis(const float* bodies, u32 bi) { return v3_(bf(bodies, bi, 26u), bf(bodies, bi, 27u), bf(bodies, bi, 28u)); }

// Joint transform (rot, pos) for body `bi` from this world's q slice.
// Ball/free coordinate maps are the INVERSE of the integrated rotation
// (exp(-w)), matching revolute_rot's negated angle and the CPU
// joint_transform_slice.
PHYZ_DEV xf joint_transform(const float* bodies, u32 bi, const float* q, u32 q_base) {
    u32 jtype = body_jtype(bodies, bi);
    u32 q_off = body_qoff(bodies, bi);
    v3 axis = body_axis(bodies, bi);
    xf j;
    j.pos = v3_(0.0f, 0.0f, 0.0f);
    if (jtype == 0u) {
        j.rot = revolute_rot(axis, q[q_base + q_off]);
    } else if (jtype == 1u) {
        j.rot = identity_rot();
        j.pos = v3_scale(axis, q[q_base + q_off]);
    } else if (jtype == 3u) {
        v3 w = v3_(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
        j.rot = quat_to_rot(quat_exp(v3_scale(w, -1.0f)));
    } else if (jtype == 4u) {
        v3 w = v3_(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
        j.rot = quat_to_rot(quat_exp(v3_scale(w, -1.0f)));
        j.pos = v3_(q[q_base + q_off + 3u], q[q_base + q_off + 4u], q[q_base + q_off + 5u]);
    } else {
        j.rot = identity_rot();
    }
    return j;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass -1: PD position servo. One thread per (world, servoed DOF).
// tau = clamp(kp*(target - q) - kd*v, ±max_force) → ctrl
// ═══════════════════════════════════════════════════════════════════════════

PHYZ_DEV void pd_thread(u32 idx, u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                        const float* dofs, const float* q, const float* v,
                        const float* targets, float* ctrl) {
    u32 total = nworld * n_dofs;
    if (idx >= total) return;
    u32 world = idx / n_dofs;
    u32 d = idx % n_dofs;
    u32 base = d * PD_DOF_STRIDE;
    u32 q_index = (u32)dofs[base];
    u32 v_index = (u32)dofs[base + 1u];
    float kp = dofs[base + 2u];
    float kd = dofs[base + 3u];
    float max_force = dofs[base + 4u];

    float qj = q[world * nq + q_index];
    float vj = v[world * nv + v_index];
    float tgt = targets[world * n_dofs + d];
    float tau = kp * (tgt - qj) - kd * vj;
    ctrl[world * nv + v_index] = fclamp(tau, -max_force, max_force);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 0: FK + ground-plane penalty contact. One thread per world.
// Writes ext_forces (spatial force per body, body frame) and contact_state.
// ═══════════════════════════════════════════════════════════════════════════

PHYZ_DEV void contact_thread(u32 world_idx, u32 nworld, u32 nbodies, u32 nv,
                             float ground_height, float dt,
                             const float* bodies, const float* geometry,
                             const float* q, const float* v,
                             float* ext_forces, float* contact_state) {
    (void)v;
    if (world_idx >= nworld) return;

    u32 nb = nbodies;
    u32 q_base = world_idx * nv;

    // Clear external forces for this env
    u32 ef_env_base = world_idx * nb * 6u;
    for (u32 i = 0; i < nb; i++)
        for (u32 k = 0; k < 6u; k++)
            ext_forces[ef_env_base + i * 6u + k] = 0.0f;

    // FK: body-to-world rotation and body origin in world for each body
    r9 w_rot[MAX_BODIES];
    v3 w_pos[MAX_BODIES];

    for (u32 i = 0; i < nb; i++) {
        i32 parent = body_parent(bodies, i);
        r9 ptj_rot = body_ptj_rot(bodies, i);
        v3 ptj_pos = body_ptj_pos(bodies, i);
        xf j = joint_transform(bodies, i, q, q_base);

        // x_tree = j.compose(ptj)
        r9 tree_rot = compose_rot(j.rot, ptj_rot);
        v3 tree_pos = v3_add(ptj_pos, rot_tmul(ptj_rot, j.pos));

        // tang's SpatialTransform: `rot` is world→body, `pos` is the body
        // origin IN WORLD coordinates. compose gives
        //   pos_i = pos_parent + rot_parentᵀ · tree_pos.
        r9 tree_rt = transpose_rot(tree_rot);
        if (parent < 0) {
            w_rot[i] = tree_rt;
            w_pos[i] = tree_pos;
        } else {
            u32 pi = (u32)parent;
            w_rot[i] = compose_rot(w_rot[pi], tree_rt);
            w_pos[i] = v3_add(w_pos[pi], rot_mul(w_rot[pi], tree_pos));
        }
    }

    for (u32 i = 0; i < nb; i++) {
        u32 gbase = i * GEOM_STRIDE;
        u32 gtype = (u32)geometry[gbase];
        u32 cs_base = (world_idx * nb + i) * CS_STRIDE;
        if (gtype == 0u) {
            for (u32 k = 0; k < CS_STRIDE; k++) contact_state[cs_base + k] = 0.0f;
            continue;
        }

        v3 pos = w_pos[i];
        float min_z = pos.z;
        float cx = pos.x;
        float cy = pos.y;
        if (gtype == 1u) {
            float radius = geometry[gbase + 1u];
            min_z = pos.z - radius;
        } else if (gtype == 2u) {
            float hz = geometry[gbase + 3u];
            min_z = pos.z - hz;
        } else if (gtype == 3u) {
            float radius = geometry[gbase + 1u];
            float length = geometry[gbase + 2u];
            min_z = pos.z - length * 0.5f - radius;
        } else if (gtype == 4u) {
            float height = geometry[gbase + 2u];
            min_z = pos.z - height * 0.5f;
        } else if (gtype == 5u) {
            v3 mn = v3_(geometry[gbase + 1u], geometry[gbase + 2u], geometry[gbase + 3u]);
            v3 mx = v3_(geometry[gbase + 4u], geometry[gbase + 5u], geometry[gbase + 6u]);
            min_z = 3.4e38f;
            for (u32 c = 0; c < 8u; c++) {
                v3 corner = mn;
                if ((c & 1u) != 0u) corner.x = mx.x;
                if ((c & 2u) != 0u) corner.y = mx.y;
                if ((c & 4u) != 0u) corner.z = mx.z;
                v3 wc = v3_add(pos, rot_mul(w_rot[i], corner));
                if (wc.z < min_z) { min_z = wc.z; cx = wc.x; cy = wc.y; }
            }
        }

        float penetration = ground_height - min_z;
        float prev_pen = contact_state[cs_base + 1u];
        if (penetration <= 0.0f) {
            for (u32 k = 0; k < CS_STRIDE; k++) contact_state[cs_base + k] = 0.0f;
            continue;
        }

        // Kelvin-Voigt: f = k*pen + d*pen_rate, clamped to push only.
        // The plus sign is correct — see CONTACT_GROUND_SHADER in shaders.rs.
        float k_body = geometry[gbase + 8u];
        float d_body = geometry[gbase + 9u];
        float pen_rate = (penetration - prev_pen) / dt;
        float f_z = fmax_(k_body * penetration + d_body * pen_rate, 0.0f);

        // World force [0,0,f_z] into the body frame (w_rot is body→world).
        v3 fw = v3_(0.0f, 0.0f, f_z);
        v3 fb = rot_tmul(w_rot[i], fw);

        u32 ef_base = ef_env_base + i * 6u;
        ext_forces[ef_base + 3u] += fb.x;
        ext_forces[ef_base + 4u] += fb.y;
        ext_forces[ef_base + 5u] += fb.z;

        contact_state[cs_base]      = 1.0f;
        contact_state[cs_base + 1u] = penetration;
        contact_state[cs_base + 2u] = cx;
        contact_state[cs_base + 3u] = cy;
        contact_state[cs_base + 4u] = ground_height;
        contact_state[cs_base + 5u] = 0.0f;
        contact_state[cs_base + 6u] = 0.0f;
        contact_state[cs_base + 7u] = f_z;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 1: general ABA. One thread per world, serial tree traversal within.
// Bodies must be topologically sorted (parent < child).
// ═══════════════════════════════════════════════════════════════════════════

// U = I_A S (6 x n, column k at u_mat[k*6..]), D = Sᵀ U + dt*damping·I
// (row-major, stride ndof), u = τ − damping·v − Sᵀ p_A.
PHYZ_DEV void joint_udu(const float* bodies, u32 i, u32 jtype, u32 ndof, v3 axis,
                        float damping_val, float dt,
                        const m66* ia, sv6 pa,
                        const float* ctrl, const float* v, u32 v_slot,
                        float* u_mat, float* d_mat, float* u_vec) {
    (void)bodies; (void)i;
    for (u32 k = 0; k < ndof; k++) {
        sv6 s_k = subspace_col(jtype, axis, k);
        sv6 uk = m6_mul_vec(ia, s_k);
        for (u32 r = 0; r < 6u; r++) u_mat[k * 6u + r] = uk.a[r];
        u_vec[k] = ctrl[v_slot + k] - damping_val * v[v_slot + k] - sv_dot(s_k, pa);
    }
    for (u32 r = 0; r < ndof; r++) {
        sv6 s_r = subspace_col(jtype, axis, r);
        for (u32 c = 0; c < ndof; c++) {
            float acc_d = 0.0f;
            for (u32 t = 0; t < 6u; t++) acc_d += s_r.a[t] * u_mat[c * 6u + t];
            d_mat[r * ndof + c] = acc_d;
        }
        // Implicit joint damping — must match phyz_rigid::aba exactly.
        d_mat[r * ndof + r] += dt * damping_val;
    }
}

PHYZ_DEV void aba_thread(u32 world_idx, u32 nworld, u32 nv, float dt, u32 nbodies,
                         float gx, float gy, float gz,
                         const float* bodies, const float* q, const float* v,
                         const float* ctrl, float* qdd, const float* ext_forces) {
    if (world_idx >= nworld) return;

    u32 nb = nbodies;
    u32 q_base = world_idx * nv;  // nq == nv: ball/free use exponential coordinates
    u32 v_base = world_idx * nv;

    sv6 vel[MAX_BODIES];
    sv6 c_bias[MAX_BODIES];
    sv6 p_a[MAX_BODIES];
    m66 i_a[MAX_BODIES];
    sv6 acc[MAX_BODIES];
    r9 x_rot[MAX_BODIES];
    v3 x_pos[MAX_BODIES];

    // Gravity as base acceleration: a0 = [0; -g]
    sv6 a0 = sv_(0.0f, 0.0f, 0.0f, -gx, -gy, -gz);

    // ── Pass 1: forward — velocities and bias forces ──
    for (u32 i = 0; i < nb; i++) {
        i32 parent = body_parent(bodies, i);
        u32 jtype = body_jtype(bodies, i);
        u32 v_off = body_voff(bodies, i);

        float mass = bf(bodies, i, 4u);
        v3 com = v3_(bf(bodies, i, 5u), bf(bodies, i, 6u), bf(bodies, i, 7u));
        float inertia[6];
        for (u32 k = 0; k < 6u; k++) inertia[k] = bf(bodies, i, 8u + k);
        r9 ptj_rot = body_ptj_rot(bodies, i);
        v3 ptj_pos = body_ptj_pos(bodies, i);
        v3 axis = body_axis(bodies, i);

        xf j = joint_transform(bodies, i, q, q_base);
        xf composed = compose_transform(j.rot, j.pos, ptj_rot, ptj_pos);
        x_rot[i] = composed.rot;
        x_pos[i] = composed.pos;

        u32 ndof = joint_ndof(jtype);
        sv6 v_joint = sv_zero();
        for (u32 k = 0; k < ndof; k++)
            v_joint = sv_add(v_joint, sv_scale(subspace_col(jtype, axis, k), v[v_base + v_off + k]));

        if (parent < 0) {
            vel[i] = v_joint;
            c_bias[i] = sv_zero();
        } else {
            u32 pi = (u32)parent;
            sv6 v_parent = apply_motion(x_rot[i], x_pos[i], vel[pi]);
            vel[i] = sv_add(v_parent, v_joint);
            c_bias[i] = sv_cross_motion(vel[i], v_joint);
        }

        i_a[i] = rigid_inertia_to_m6(mass, com, inertia);

        sv6 iv = m6_mul_vec(&i_a[i], vel[i]);
        p_a[i] = sv_cross_force(vel[i], iv);

        u32 ef_base = (world_idx * nb + i) * 6u;
        sv6 ef;
        for (u32 k = 0; k < 6u; k++) ef.a[k] = ext_forces[ef_base + k];
        p_a[i] = sv_sub(p_a[i], ef);
    }

    // ── Pass 2: backward — articulated inertias and forces ──
    for (u32 ii = 0; ii < nb; ii++) {
        u32 i = nb - 1u - ii;
        i32 parent = body_parent(bodies, i);
        u32 jtype = body_jtype(bodies, i);
        u32 v_off = body_voff(bodies, i);
        v3 axis = body_axis(bodies, i);
        float damping_val = bf(bodies, i, 29u);

        if (jtype == 2u) {
            if (parent >= 0) {
                u32 pi = (u32)parent;
                m66 x_mot = build_motion_transform(x_rot[i], x_pos[i]);
                m66 x_mot_t = transpose6(&x_mot);
                m66 ia_parent = m6_XtAX(&x_mot_t, &i_a[i], &x_mot);
                i_a[pi] = m6_add(&i_a[pi], &ia_parent);
                sv6 p_parent = inv_apply_force(x_rot[i], x_pos[i], p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_parent);
            }
            continue;
        }

        u32 ndof = joint_ndof(jtype);
        float u_mat[36];
        float d_mat[36];
        float u_vec[6];
        joint_udu(bodies, i, jtype, ndof, axis, damping_val, dt, &i_a[i], p_a[i],
                  ctrl, v, v_base + v_off, u_mat, d_mat, u_vec);

        // Singular articulated inertia: treat the joint as fixed.
        if (!invert_small(d_mat, ndof)) {
            if (parent >= 0) {
                u32 pi = (u32)parent;
                m66 x_mot_s = build_motion_transform(x_rot[i], x_pos[i]);
                m66 x_mot_st = transpose6(&x_mot_s);
                m66 ia_par_s = m6_XtAX(&x_mot_st, &i_a[i], &x_mot_s);
                i_a[pi] = m6_add(&i_a[pi], &ia_par_s);
                sv6 p_par_s = inv_apply_force(x_rot[i], x_pos[i], p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_par_s);
            }
            continue;
        }

        if (parent >= 0) {
            u32 pi = (u32)parent;

            // W = U D⁻¹ (6 x n)
            float w_mat[36];
            for (u32 c = 0; c < ndof; c++)
                for (u32 r = 0; r < 6u; r++) {
                    float s = 0.0f;
                    for (u32 jj = 0; jj < ndof; jj++) s += u_mat[jj * 6u + r] * d_mat[jj * ndof + c];
                    w_mat[c * 6u + r] = s;
                }

            // I_a^A = I_A − W Uᵀ
            m66 ia_new = i_a[i];
            for (u32 c = 0; c < ndof; c++) {
                sv6 wc, uc;
                for (u32 r = 0; r < 6u; r++) { wc.a[r] = w_mat[c * 6u + r]; uc.a[r] = u_mat[c * 6u + r]; }
                m66 outer = m6_outer(wc, uc);
                ia_new = m6_sub(&ia_new, &outer);
            }

            // p_a^A = p_A + I_a^A c + W u
            sv6 ia_c = m6_mul_vec(&ia_new, c_bias[i]);
            sv6 wu = sv_zero();
            for (u32 c = 0; c < ndof; c++)
                for (u32 r = 0; r < 6u; r++) wu.a[r] += w_mat[c * 6u + r] * u_vec[c];
            sv6 p_new = sv_add(sv_add(p_a[i], ia_c), wu);

            m66 x_mot = build_motion_transform(x_rot[i], x_pos[i]);
            m66 x_mot_t = transpose6(&x_mot);
            m66 ia_parent = m6_XtAX(&x_mot_t, &ia_new, &x_mot);
            i_a[pi] = m6_add(&i_a[pi], &ia_parent);

            sv6 p_parent = inv_apply_force(x_rot[i], x_pos[i], p_new);
            p_a[pi] = sv_add(p_a[pi], p_parent);
        }
    }

    // ── Pass 3: forward — accelerations ──
    for (u32 i = 0; i < nb; i++) {
        i32 parent = body_parent(bodies, i);
        u32 jtype = body_jtype(bodies, i);
        u32 v_off = body_voff(bodies, i);
        v3 axis = body_axis(bodies, i);
        float damping_val = bf(bodies, i, 29u);

        sv6 a_parent;
        if (parent < 0) a_parent = apply_motion(x_rot[i], x_pos[i], a0);
        else            a_parent = apply_motion(x_rot[i], x_pos[i], acc[(u32)parent]);

        sv6 a_c = sv_add(a_parent, c_bias[i]);
        u32 ndof = joint_ndof(jtype);
        if (ndof == 0u) { acc[i] = a_c; continue; }

        // Rebuild U, D, u rather than carrying them from pass 2 (private
        // storage pressure — same trade the WGSL makes).
        float u_mat[36];
        float d_mat[36];
        float u_vec[6];
        joint_udu(bodies, i, jtype, ndof, axis, damping_val, dt, &i_a[i], p_a[i],
                  ctrl, v, v_base + v_off, u_mat, d_mat, u_vec);

        if (!invert_small(d_mat, ndof)) {
            acc[i] = a_c;
            for (u32 k = 0; k < ndof; k++) qdd[v_base + v_off + k] = 0.0f;
            continue;
        }

        // qdd = D⁻¹ (u − Uᵀ a_c)
        float rhs[6];
        for (u32 k = 0; k < ndof; k++) {
            float uta = 0.0f;
            for (u32 t = 0; t < 6u; t++) uta += u_mat[k * 6u + t] * a_c.a[t];
            rhs[k] = u_vec[k] - uta;
        }

        sv6 a_new = a_c;
        for (u32 k = 0; k < ndof; k++) {
            float qdd_k = 0.0f;
            for (u32 jj = 0; jj < ndof; jj++) qdd_k += d_mat[k * ndof + jj] * rhs[jj];
            qdd[v_base + v_off + k] = qdd_k;
            a_new = sv_add(a_new, sv_scale(subspace_col(jtype, axis, k), qdd_k));
        }
        acc[i] = a_new;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 2: joint-aware semi-implicit Euler. One thread per (world, joint).
// Must match phyz_rigid::semi_implicit_euler.
// ═══════════════════════════════════════════════════════════════════════════

PHYZ_DEV void integrate_thread(u32 idx, u32 nworld, u32 nv, float dt, u32 nbodies,
                               float* q, float* v, const float* qdd, const float* bodies) {
    u32 nb = nbodies;
    if (idx >= nworld * nb) return;

    u32 world_idx = idx / nb;
    u32 body_idx = idx % nb;

    u32 jtype = body_jtype(bodies, body_idx);
    u32 ndof = joint_ndof(jtype);
    if (ndof == 0u) return;

    u32 q_off = world_idx * nv + body_qoff(bodies, body_idx);
    u32 v_off = world_idx * nv + body_voff(bodies, body_idx);

    // Velocity first (semi-implicit).
    for (u32 k = 0; k < ndof; k++) v[v_off + k] = v[v_off + k] + dt * qdd[v_off + k];

    if (jtype == 0u || jtype == 1u) {
        q[q_off] = q[q_off] + dt * v[v_off];
        return;
    }

    if (jtype == 3u) {
        v3 omega = v3_(v[v_off], v[v_off + 1u], v[v_off + 2u]);
        v4 cur = quat_exp(v3_(q[q_off], q[q_off + 1u], q[q_off + 2u]));
        v4 nxt = qnormalize(qmul(cur, quat_exp(v3_scale(omega, dt))));
        v3 lg = qlog(nxt);
        q[q_off] = lg.x; q[q_off + 1u] = lg.y; q[q_off + 2u] = lg.z;
        return;
    }

    // Free: v = [angular(3), linear(3)], q = [exp-coords(3), pos(3)].
    v3 omega = v3_(v[v_off], v[v_off + 1u], v[v_off + 2u]);
    v3 lin = v3_(v[v_off + 3u], v[v_off + 4u], v[v_off + 5u]);
    v4 cur = quat_exp(v3_(q[q_off], q[q_off + 1u], q[q_off + 2u]));

    v3 world_lin = qrotate(cur, lin);
    q[q_off + 3u] = q[q_off + 3u] + dt * world_lin.x;
    q[q_off + 4u] = q[q_off + 4u] + dt * world_lin.y;
    q[q_off + 5u] = q[q_off + 5u] + dt * world_lin.z;

    v4 nxt = qnormalize(qmul(cur, quat_exp(v3_scale(omega, dt))));
    v3 lg = qlog(nxt);
    q[q_off] = lg.x; q[q_off + 1u] = lg.y; q[q_off + 2u] = lg.z;
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

#if PHYZ_ON_DEVICE

extern "C" __global__ void phyz_pd(u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                                   const float* dofs, const float* q, const float* v,
                                   const float* targets, float* ctrl) {
    pd_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nq, nv, n_dofs, dofs, q, v, targets, ctrl);
}

extern "C" __global__ void phyz_contact(u32 nworld, u32 nbodies, u32 nv, float ground_height, float dt,
                                        const float* bodies, const float* geometry,
                                        const float* q, const float* v,
                                        float* ext_forces, float* contact_state) {
    contact_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nbodies, nv, ground_height, dt,
                   bodies, geometry, q, v, ext_forces, contact_state);
}

extern "C" __global__ void phyz_aba(u32 nworld, u32 nv, float dt, u32 nbodies,
                                    float gx, float gy, float gz,
                                    const float* bodies, const float* q, const float* v,
                                    const float* ctrl, float* qdd, const float* ext_forces) {
    aba_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nv, dt, nbodies, gx, gy, gz,
               bodies, q, v, ctrl, qdd, ext_forces);
}

extern "C" __global__ void phyz_integrate(u32 nworld, u32 nv, float dt, u32 nbodies,
                                          float* q, float* v, const float* qdd, const float* bodies) {
    integrate_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nv, dt, nbodies, q, v, qdd, bodies);
}

#else  // host: walk the grid serially

extern "C" void phyz_host_pd(u32 n_threads, u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                             const float* dofs, const float* q, const float* v,
                             const float* targets, float* ctrl) {
    for (u32 t = 0; t < n_threads; t++)
        pd_thread(t, nworld, nq, nv, n_dofs, dofs, q, v, targets, ctrl);
}

extern "C" void phyz_host_contact(u32 n_threads, u32 nworld, u32 nbodies, u32 nv,
                                  float ground_height, float dt,
                                  const float* bodies, const float* geometry,
                                  const float* q, const float* v,
                                  float* ext_forces, float* contact_state) {
    for (u32 t = 0; t < n_threads; t++)
        contact_thread(t, nworld, nbodies, nv, ground_height, dt, bodies, geometry, q, v,
                       ext_forces, contact_state);
}

extern "C" void phyz_host_aba(u32 n_threads, u32 nworld, u32 nv, float dt, u32 nbodies,
                              float gx, float gy, float gz,
                              const float* bodies, const float* q, const float* v,
                              const float* ctrl, float* qdd, const float* ext_forces) {
    for (u32 t = 0; t < n_threads; t++)
        aba_thread(t, nworld, nv, dt, nbodies, gx, gy, gz, bodies, q, v, ctrl, qdd, ext_forces);
}

extern "C" void phyz_host_integrate(u32 n_threads, u32 nworld, u32 nv, float dt, u32 nbodies,
                                    float* q, float* v, const float* qdd, const float* bodies) {
    for (u32 t = 0; t < n_threads; t++)
        integrate_thread(t, nworld, nv, dt, nbodies, q, v, qdd, bodies);
}

#endif
