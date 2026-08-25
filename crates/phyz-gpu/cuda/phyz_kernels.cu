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
/// Same as `PHYZ_DEV`, for a struct member function where `static` is illegal.
#define PHYZ_M __device__ __forceinline__
#else
#define PHYZ_ON_DEVICE 0
#include <math.h>
#include <string.h>
#define PHYZ_DEV static inline
#define PHYZ_M inline
#endif

typedef unsigned int u32;
typedef int i32;

// Mirrors layout.rs.
#define MAX_BODIES 32u
/// Velocity-DOF bound for the per-step `U`/`D⁻¹` cache. A model wider than
/// this still runs — `aba_thread_c` falls back to refactorising, exactly as
/// it did before the cache existed.
#define MAX_NV 64u
#define BODY_STRIDE 36u
// The geometry table is indexed by COLLISION INSTANCE, not by body: a body's
// slice is [bf(bodies,i,33), +bf(bodies,i,34)). Mirrors layout.rs / shaders.rs.
#define GEOM_STRIDE 24u
// Plane records share the geometry buffer and its stride; see layout.rs.
#define PLANE_STRIDE 24u
#define CS_STRIDE 184u
// Body-attached-face readback + per-point detail. [0..8] is the GROUND result
// and cannot hold a face contact too (a wheel touches both in one step), so
// the face gets blocks of its own. Mirrors shaders.rs / layout.rs.
#define CS_PLANE_RB_OFF 80u
#define CS_PLANE_DETAIL_OFF 88u
#define PLANE_DETAIL_STRIDE 6u
// Face manifold slots per body, separate from MAX_PTS and larger: the face
// branch builds a CLIPPED manifold against a SET of faces. Mirrors shaders.rs.
#define MAX_PLANE_PTS 16u
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

PHYZ_DEV float u_as_f(u32 u) {
#if PHYZ_ON_DEVICE
    return __uint_as_float(u);
#else
    float f;
    memcpy(&f, &u, 4);
    return f;
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
// Pass 0: FK + contact. One thread per world.
//
// A hand port of CONTACT_GROUND_SHADER in shaders.rs — read that file for
// the physics and the measurements behind every bound; the comments here
// are only the ones a reader of THIS file needs. Ground/terrain contact,
// the body-attached face (a deck), penalty forces or the velocity-level
// impulse solve, Coulomb friction, box manifolds. Writes ext_forces (spatial
// force per body, body frame) and contact_state (readback + warm-start
// impulses).
//
// Parameters arrive as one CP_STRIDE-float buffer laid out exactly like the
// WGSL `ContactParams` uniform (u32 fields bitcast), packed once on the host
// by `contact_pipeline::ContactParams` for both backends.
// ═══════════════════════════════════════════════════════════════════════════

#define CP_STRIDE 32u
#define CS_IMPULSE_OFF 8u
#define CS_PLANE_OFF 32u
#define MAX_PTS 8u
#define SLIP_EPS 1e-3f

// ContactParams, decoded once per thread.
typedef struct {
    u32 nworld, nbodies, nv;
    float ground_height, dt, friction;
    u32 nplanes, plane_base;
    u32 hf_nx, hf_ny;
    float hf_ox, hf_oy, hf_oz, hf_cell;
    u32 solve_mode;
    float restitution, restitution_threshold, solref_erp, margin;
    float solimp_dmin, solimp_dmax, solimp_width, solimp_mid, solimp_power;
} cparams_t;

PHYZ_DEV cparams_t decode_cparams(const float* p) {
    cparams_t c;
    c.nworld = f_as_u(p[0]); c.nbodies = f_as_u(p[1]); c.nv = f_as_u(p[2]);
    c.ground_height = p[3]; c.dt = p[4]; c.friction = p[5];
    c.nplanes = f_as_u(p[6]); c.plane_base = f_as_u(p[7]);
    // p[8..11] reserved (were the single plane's offset / depth / half-extents)
    c.hf_nx = f_as_u(p[11]); c.hf_ny = f_as_u(p[12]);
    c.hf_ox = p[13]; c.hf_oy = p[14]; c.hf_oz = p[15]; c.hf_cell = p[16];
    c.solve_mode = f_as_u(p[17]);
    // p[18] reserved (was a sweep index)
    c.restitution = p[19]; c.restitution_threshold = p[20];
    c.solref_erp = p[21]; c.margin = p[22];
    c.solimp_dmin = p[23]; c.solimp_dmax = p[24]; c.solimp_width = p[25];
    c.solimp_mid = p[26]; c.solimp_power = p[27];
    return c;
}

PHYZ_DEV float fsign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
PHYZ_DEV v3 v3_neg(v3 a) { return v3_(-a.x, -a.y, -a.z); }
PHYZ_DEV v3 v3_normalize(v3 a) { float l = v3_len(a); return l > 0.0f ? v3_scale(a, 1.0f / l) : a; }

// mu*f_n*min(1, vt/SLIP_EPS): Coulomb regularized by slip speed.
PHYZ_DEV float coulomb(float mu_fn, float vt) { return mu_fn * fmin_(1.0f, vt / SLIP_EPS); }

// ── Body-attached-face readback ──
//
// One solved face contact into the touching body's readback block: the
// aggregate (touching, deepest penetration, deepest point, TOTAL force) plus a
// per-point entry keyed by the warm-start rank, carrying the plane index so
// the host can group points by face. Called by both the impulse and the
// penalty branch with the world force on the touching body and its normal
// component. Mirrors `plane_readback` in shaders.rs.
PHYZ_DEV void plane_readback(float *contact_state, u32 world_idx, u32 nb, u32 i,
                             u32 pl, u32 rank, float penetration, v3 point_w,
                             v3 n_w, v3 f_w, float f_n) {
    u32 rb = (world_idx * nb + i) * CS_STRIDE + CS_PLANE_RB_OFF;
    contact_state[rb] = 1.0f;
    if (penetration > contact_state[rb + 1u]) {
        contact_state[rb + 1u] = penetration;
        contact_state[rb + 2u] = point_w.x;
        contact_state[rb + 3u] = point_w.y;
        contact_state[rb + 4u] = point_w.z;
    }
    contact_state[rb + 5u] += f_w.x;
    contact_state[rb + 6u] += f_w.y;
    contact_state[rb + 7u] += f_w.z;

    u32 d = (world_idx * nb + i) * CS_STRIDE + CS_PLANE_DETAIL_OFF + rank * PLANE_DETAIL_STRIDE;
    // Biased by one: a zeroed slot must read as "empty", not as "plane 0".
    contact_state[d] = (float)pl + 1.0f;
    contact_state[d + 1u] = penetration;
    contact_state[d + 2u] = f_n;
    contact_state[d + 3u] = n_w.x;
    contact_state[d + 4u] = n_w.y;
    contact_state[d + 5u] = n_w.z;
}

// ── The incident face of a box against a contact normal ──
//
// The four corners, in BODY coordinates, of the box face most opposed to
// `n_body`. Sampling a box's eight CORNERS against a face plane is only
// correct while the face is at least as big as the box; on a narrow strip the
// corners are all outside it and their "penetration" is measured against the
// strip's INFINITE extension, which is a fiction. Mirrors shaders.rs.
PHYZ_DEV void box_incident_face(const float* geometry, u32 g, v3 n_body, v3* out) {
    u32 gbase = g * GEOM_STRIDE;
    v3 h = v3_(geometry[gbase + 1u], geometry[gbase + 2u], geometry[gbase + 3u]);
    v3 o_p = v3_(geometry[gbase + 10u], geometry[gbase + 11u], geometry[gbase + 12u]);
    r9 o_r; for (u32 k = 0; k < 9u; k++) o_r.r[k] = geometry[gbase + 13u + k];
    v3 n = rot_mul(o_r, n_body);
    float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    u32 axis = 0u;
    if (ay >= ax && ay >= az) axis = 1u;
    else if (az >= ax && az >= ay) axis = 2u;
    // -sign so the face points INTO the surface; sign(0) would collapse the
    // face to the box centre, so the tie goes to -1.
    float sgn = -1.0f;
    if (axis == 0u && n.x < 0.0f) sgn = 1.0f;
    if (axis == 1u && n.y < 0.0f) sgn = 1.0f;
    if (axis == 2u && n.z < 0.0f) sgn = 1.0f;
    for (u32 c = 0; c < 4u; c++) {
        float s0 = (c & 1u) ? 1.0f : -1.0f;
        float s1 = (c & 2u) ? 1.0f : -1.0f;
        v3 v;
        if (axis == 0u) v = v3_(sgn * h.x, s0 * h.y, s1 * h.z);
        else if (axis == 1u) v = v3_(s0 * h.x, sgn * h.y, s1 * h.z);
        else v = v3_(s0 * h.x, s1 * h.y, sgn * h.z);
        out[c] = v3_add(o_p, rot_tmul(o_r, v));
    }
}

// I^-1 w for the body's rotational inertia about its COM (cofactor inverse).
PHYZ_DEV v3 inertia_solve(const float* bodies, u32 bidx, v3 w) {
    float xx = bf(bodies, bidx, 8u), yy = bf(bodies, bidx, 9u), zz = bf(bodies, bidx, 10u);
    float xy = bf(bodies, bidx, 11u), xz = bf(bodies, bidx, 12u), yz = bf(bodies, bidx, 13u);
    float c00 = yy * zz - yz * yz;
    float c01 = xz * yz - xy * zz;
    float c02 = xy * yz - xz * yy;
    float det = xx * c00 + xy * c01 + xz * c02;
    if (fabs_(det) < 1e-20f) return v3_(0.0f, 0.0f, 0.0f);
    float c11 = xx * zz - xz * xz;
    float c12 = xy * xz - xx * yz;
    float c22 = xx * yy - xy * xy;
    float inv = 1.0f / det;
    return v3_(inv * (c00 * w.x + c01 * w.y + c02 * w.z),
               inv * (c01 * w.x + c11 * w.y + c12 * w.z),
               inv * (c02 * w.x + c12 * w.y + c22 * w.z));
}

// Effective mass body `bidx` presents at body-frame offset `r` (from the
// origin) along unit body-frame direction `u`. Massless = immovable.
PHYZ_DEV float contact_eff_mass(const float* bodies, u32 bidx, v3 r, v3 u) {
    float m = bf(bodies, bidx, 4u);
    if (m <= 0.0f) return 1e30f;
    v3 rc = v3_sub(r, v3_(bf(bodies, bidx, 5u), bf(bodies, bidx, 6u), bf(bodies, bidx, 7u)));
    v3 a = cross3(rc, u);
    float ang = fmax_(v3_dot(a, inertia_solve(bodies, bidx, a)), 0.0f);
    return 1.0f / (1.0f / m + ang);
}

// Explicit-damper bound: m_eff/dt. Explicit-spring bound: 4 m_eff/dt^2.
PHYZ_DEV float max_damping(float m_eff, float dt) { return m_eff / fmax_(dt, 1e-9f); }
PHYZ_DEV float max_stiffness(float m_eff, float dt) {
    float d = fmax_(dt, 1e-9f);
    return 4.0f * m_eff / (d * d);
}

// Orthonormal tangents around n; must match phyz_contact::cone::contact_frame.
PHYZ_DEV void contact_tangents(v3 n, v3* u, v3* w) {
    v3 a = v3_(1.0f, 0.0f, 0.0f);
    if (fabs_(n.x) > 0.9f) a = v3_(0.0f, 1.0f, 0.0f);
    *u = v3_normalize(cross3(n, a));
    *w = cross3(n, *u);
}

// SolImp impedance, branch for branch with phyz_contact::material::SolImp.
PHYZ_DEV float solimp_impedance(const cparams_t* c, float r) {
    float dmin = fclamp(c->solimp_dmin, 1e-4f, 1.0f - 1e-9f);
    float dmax = fclamp(c->solimp_dmax, 1e-4f, 1.0f - 1e-9f);
    if (c->solimp_width <= 0.0f) return dmax;
    float x = fclamp(fabs_(r) / c->solimp_width, 0.0f, 1.0f);
    float mid = fclamp(c->solimp_mid, 1e-6f, 1.0f - 1e-6f);
    float pw = fmax_(c->solimp_power, 1.0f);
    float y;
    if (x <= mid) y = powf(x, pw) / powf(mid, pw - 1.0f);
    else          y = 1.0f - powf(1.0f - x, pw) / powf(1.0f - mid, pw - 1.0f);
    return dmin + y * (dmax - dmin);
}

// ContactMaterial::impedance_at: solimp when penetrating, smoothstep to zero
// across the margin when separated.
PHYZ_DEV float impedance_at(const cparams_t* c, float depth) {
    if (depth >= 0.0f) return solimp_impedance(c, depth);
    float gap = -depth;
    if (c->margin <= 0.0f || gap >= c->margin) return 0.0f;
    float sc = 1.0f - gap / c->margin;
    return sc * sc * (3.0f - 2.0f * sc) * solimp_impedance(c, 0.0f);
}

// Smoothstep restitution ramp between v_rest and 2 v_rest.
PHYZ_DEV float effective_restitution(const cparams_t* c, float e, float approach) {
    float vr = c->restitution_threshold;
    if (vr <= 0.0f) return e;
    float sp = fabs_(approach);
    if (sp <= vr) return 0.0f;
    if (sp >= 2.0f * vr) return e;
    float t = (sp - vr) / vr;
    return e * t * t * (3.0f - 2.0f * t);
}

// ── Heightfield terrain (mirrors phyz_model::Heightfield) ──

PHYZ_DEV float hf_node(const cparams_t* c, const float* hf, u32 ix, u32 iy) {
    return c->hf_oz + hf[iy * c->hf_nx + ix];
}

// Cell index and intra-cell fraction along one axis, clamped to the grid.
PHYZ_DEV void hf_locate(const cparams_t* c, float w, float o, u32 n, u32* i, float* t) {
    if (n < 2u) { *i = 0u; *t = 0.0f; return; }
    float u = fclamp((w - o) / c->hf_cell, 0.0f, (float)(n - 1u));
    u32 ii = (u32)u;
    if (ii > n - 2u) ii = n - 2u;
    *i = ii;
    *t = u - (float)ii;
}

// Terrain sample at world (x, y): unit normal out, height returned. The flat
// plane at ground_height when no heightfield is loaded.
PHYZ_DEV float terrain(const cparams_t* c, const float* hf, float px, float py, v3* n_out) {
    u32 nx = c->hf_nx, ny = c->hf_ny;
    if (nx == 0u) { *n_out = v3_(0.0f, 0.0f, 1.0f); return c->ground_height; }
    u32 ix, iy; float tx, ty;
    hf_locate(c, px, c->hf_ox, nx, &ix, &tx);
    hf_locate(c, py, c->hf_oy, ny, &iy, &ty);
    u32 ix1 = ix + 1u < nx - 1u ? ix + 1u : nx - 1u;
    u32 iy1 = iy + 1u < ny - 1u ? iy + 1u : ny - 1u;
    float h00 = hf_node(c, hf, ix, iy);
    float h10 = hf_node(c, hf, ix1, iy);
    float h01 = hf_node(c, hf, ix, iy1);
    float h11 = hf_node(c, hf, ix1, iy1);
    float h = (h00 * (1.0f - tx) + h10 * tx) * (1.0f - ty)
            + (h01 * (1.0f - tx) + h11 * tx) * ty;
    float dhdx = 0.0f, dhdy = 0.0f;
    float span_x = (float)(nx - 1u) * c->hf_cell;
    float span_y = (float)(ny - 1u) * c->hf_cell;
    if (nx >= 2u && px >= c->hf_ox && px <= c->hf_ox + span_x)
        dhdx = ((h10 - h00) * (1.0f - ty) + (h11 - h01) * ty) / c->hf_cell;
    if (ny >= 2u && py >= c->hf_oy && py <= c->hf_oy + span_y)
        dhdy = ((h01 - h00) * (1.0f - tx) + (h11 - h10) * tx) / c->hf_cell;
    *n_out = v3_normalize(v3_(-dhdx, -dhdy, 1.0f));
    return h;
}

// Origin of collision instance i: pos at [10..13], rot
// (body -> shape, row-major) at [13..22].
PHYZ_DEV v3 geom_origin_pos(const float* geometry, u32 i) {
    u32 g = i * GEOM_STRIDE;
    return v3_(geometry[g + 10u], geometry[g + 11u], geometry[g + 12u]);
}
PHYZ_DEV r9 geom_origin_rot(const float* geometry, u32 i) {
    u32 g = i * GEOM_STRIDE;
    r9 r; for (u32 k = 0; k < 9u; k++) r.r[k] = geometry[g + 13u + k];
    return r;
}

// Corner c (0..8) of collision instance i's box, body frame, origin applied.
PHYZ_DEV v3 box_corner(const float* geometry, u32 i, u32 c) {
    u32 g = i * GEOM_STRIDE;
    v3 h = v3_(geometry[g + 1u], geometry[g + 2u], geometry[g + 3u]);
    float sx = (c & 1u) != 0u ? 1.0f : -1.0f;
    float sy = (c & 2u) != 0u ? 1.0f : -1.0f;
    float sz = (c & 4u) != 0u ? 1.0f : -1.0f;
    v3 corner = v3_(h.x * sx, h.y * sy, h.z * sz);
    return v3_add(geom_origin_pos(geometry, i), rot_tmul(geom_origin_rot(geometry, i), corner));
}

// Support point of collision instance i against unit body-frame direction n_body,
// body frame, instance origin applied.
PHYZ_DEV v3 support_point(const float* geometry, u32 i, v3 n_body) {
    u32 g = i * GEOM_STRIDE;
    u32 gtype = (u32)geometry[g];
    v3 o_p = geom_origin_pos(geometry, i);
    r9 o_r = geom_origin_rot(geometry, i);
    v3 n = rot_mul(o_r, n_body);
    v3 support = v3_(0.0f, 0.0f, 0.0f);
    if (gtype == 1u) {
        support = v3_scale(n, -geometry[g + 1u]);
    } else if (gtype == 2u) {
        v3 h = v3_(geometry[g + 1u], geometry[g + 2u], geometry[g + 3u]);
        support = v3_(-h.x * fsign(n.x), -h.y * fsign(n.y), -h.z * fsign(n.z));
    } else if (gtype == 3u) {
        float radius = geometry[g + 1u];
        float half_len = geometry[g + 2u] * 0.5f;
        support = v3_sub(v3_(0.0f, 0.0f, -half_len * fsign(n.z)), v3_scale(n, radius));
    } else if (gtype == 4u) {
        float radius = geometry[g + 1u];
        float half_h = geometry[g + 2u] * 0.5f;
        v3 radial = v3_(-n.x, -n.y, 0.0f);
        float rl = v3_len(radial);
        v3 rim = v3_(0.0f, 0.0f, 0.0f);
        if (rl > 1e-6f) rim = v3_scale(radial, radius / rl);
        support = v3_add(rim, v3_(0.0f, 0.0f, -half_h * fsign(n.z)));
    } else if (gtype == 5u) {
        // Mesh via its body-frame AABB, resolved through sign() so a flat
        // face ties to its centre (see shaders.rs for the measurement).
        v3 mn = v3_(geometry[g + 1u], geometry[g + 2u], geometry[g + 3u]);
        v3 mx = v3_(geometry[g + 4u], geometry[g + 5u], geometry[g + 6u]);
        v3 mc = v3_scale(v3_add(mn, mx), 0.5f);
        v3 mh = v3_scale(v3_sub(mx, mn), 0.5f);
        support = v3_sub(mc, v3_(mh.x * fsign(n.x), mh.y * fsign(n.y), mh.z * fsign(n.z)));
    }
    return v3_add(o_p, rot_tmul(o_r, support));
}

// Contact point cpt of collision instance i (box: corner; else: support against n_body).
PHYZ_DEV v3 contact_pt(const float* geometry, u32 i, u32 gtype, u32 cpt, v3 n_body) {
    if (gtype == 2u) return box_corner(geometry, i, cpt);
    return support_point(geometry, i, n_body);
}

PHYZ_DEV void add_ext(float* ext_forces, u32 ef_base, v3 torque, v3 force) {
    ext_forces[ef_base + 0u] += torque.x;
    ext_forces[ef_base + 1u] += torque.y;
    ext_forces[ef_base + 2u] += torque.z;
    ext_forces[ef_base + 3u] += force.x;
    ext_forces[ef_base + 4u] += force.y;
    ext_forces[ef_base + 5u] += force.z;
}

// ── Forward kinematics, shared by the contact pass and the FK readout ──────
// Fills, per body: body-to-world rotation `w_rot` (so `rot_tmul(w_rot, x)`
// takes a world vector into the body frame), body origin in world `w_pos`,
// and the body-frame spatial velocity (`w_omega`, `w_lin`). With `use_free`
// the velocity is the FREE velocity v + dt*qdd (impulse-mode contact);
// otherwise `v` as it stands. Verbatim the loop the contact pass always ran.
// The `q`-only half of the contact pass's FK chain. A contact sweep changes
// velocities alone, so the body-to-world rotations and origins — and the
// joint transforms they are built from — are the same for every sweep of a
// step. Same argument as `aba_cache_t`, same bit-identity.
// The cache is reached through METHODS, not members, so the same code can
// carry it in two places: on the thread's local stack (`fk_cache_t`, what the
// fused kernel uses) or in a global structure-of-arrays buffer
// (`fk_cache_g`, what the fissioned stage kernels use, world index fastest so
// the loads coalesce across worlds). Same values, same order, same bits.
//
// Field offsets, in floats per world, for the SoA layout.
#define FKC_W_ROT    0u
#define FKC_W_POS    (FKC_W_ROT    + MAX_BODIES * 9u)
#define FKC_TREE_ROT (FKC_W_POS    + MAX_BODIES * 3u)
#define FKC_TREE_POS (FKC_TREE_ROT + MAX_BODIES * 9u)
#define FKC_N_SEL    (FKC_TREE_POS + MAX_BODIES * 3u)
#define FKC_SEL_PEN  (FKC_N_SEL    + MAX_BODIES)
#define FKC_SEL_G    (FKC_SEL_PEN  + MAX_BODIES * MAX_PTS)
#define FKC_SEL_C    (FKC_SEL_G    + MAX_BODIES * MAX_PTS)
#define FKC_PL_N     (FKC_SEL_C    + MAX_BODIES * MAX_PTS)
#define FKC_PL_PEN   (FKC_PL_N     + MAX_BODIES)
#define FKC_PL_U     (FKC_PL_PEN   + MAX_BODIES * MAX_PLANE_PTS)
#define FKC_PL_V     (FKC_PL_U     + MAX_BODIES * MAX_PLANE_PTS)
#define FKC_PL_GPL   (FKC_PL_V     + MAX_BODIES * MAX_PLANE_PTS)
/// Floats per world. Mirrored by `layout::FK_CACHE_FLOATS`.
#define FK_CACHE_FLOATS (FKC_PL_GPL + MAX_BODIES * MAX_PLANE_PTS)

struct fk_cache_t {
    r9 w_rot_[MAX_BODIES];
    v3 w_pos_[MAX_BODIES];
    r9 tree_rot_[MAX_BODIES];
    v3 tree_pos_[MAX_BODIES];
    u32 n_sel_[MAX_BODIES];
    float sel_pen_[MAX_BODIES * MAX_PTS];
    u32 sel_g_[MAX_BODIES * MAX_PTS];
    u32 sel_c_[MAX_BODIES * MAX_PTS];
    u32 pl_n_[MAX_BODIES];
    float pl_pen_[MAX_BODIES * MAX_PLANE_PTS];
    float pl_u_[MAX_BODIES * MAX_PLANE_PTS];
    float pl_v_[MAX_BODIES * MAX_PLANE_PTS];
    u32 pl_gpl_[MAX_BODIES * MAX_PLANE_PTS];

    PHYZ_M r9 w_rot(u32 i) const { return w_rot_[i]; }
    PHYZ_M void set_w_rot(u32 i, r9 x) { w_rot_[i] = x; }
    PHYZ_M v3 w_pos(u32 i) const { return w_pos_[i]; }
    PHYZ_M void set_w_pos(u32 i, v3 x) { w_pos_[i] = x; }
    PHYZ_M r9 tree_rot(u32 i) const { return tree_rot_[i]; }
    PHYZ_M void set_tree_rot(u32 i, r9 x) { tree_rot_[i] = x; }
    PHYZ_M v3 tree_pos(u32 i) const { return tree_pos_[i]; }
    PHYZ_M void set_tree_pos(u32 i, v3 x) { tree_pos_[i] = x; }
    PHYZ_M u32 n_sel(u32 i) const { return n_sel_[i]; }
    PHYZ_M void set_n_sel(u32 i, u32 x) { n_sel_[i] = x; }
    PHYZ_M float sel_pen(u32 k) const { return sel_pen_[k]; }
    PHYZ_M void set_sel_pen(u32 k, float x) { sel_pen_[k] = x; }
    PHYZ_M u32 sel_g(u32 k) const { return sel_g_[k]; }
    PHYZ_M void set_sel_g(u32 k, u32 x) { sel_g_[k] = x; }
    PHYZ_M u32 sel_c(u32 k) const { return sel_c_[k]; }
    PHYZ_M void set_sel_c(u32 k, u32 x) { sel_c_[k] = x; }
    PHYZ_M u32 pl_n(u32 i) const { return pl_n_[i]; }
    PHYZ_M void set_pl_n(u32 i, u32 x) { pl_n_[i] = x; }
    PHYZ_M float pl_pen(u32 k) const { return pl_pen_[k]; }
    PHYZ_M void set_pl_pen(u32 k, float x) { pl_pen_[k] = x; }
    PHYZ_M float pl_u(u32 k) const { return pl_u_[k]; }
    PHYZ_M void set_pl_u(u32 k, float x) { pl_u_[k] = x; }
    PHYZ_M float pl_v(u32 k) const { return pl_v_[k]; }
    PHYZ_M void set_pl_v(u32 k, float x) { pl_v_[k] = x; }
    PHYZ_M u32 pl_gpl(u32 k) const { return pl_gpl_[k]; }
    PHYZ_M void set_pl_gpl(u32 k, u32 x) { pl_gpl_[k] = x; }
};

/// The same cache in a global SoA buffer: element `e` of world `w` lives at
/// `p[e * nworld + w]`, so a warp's 32 worlds read 32 adjacent floats.
struct fk_cache_g {
    float* p;
    u32 w;
    u32 nworld;

    PHYZ_M float ldf(u32 o) const { return p[o * nworld + w]; }
    PHYZ_M void stf(u32 o, float x) { p[o * nworld + w] = x; }
    PHYZ_M u32 ldu(u32 o) const { return f_as_u(ldf(o)); }
    PHYZ_M void stu(u32 o, u32 x) { stf(o, u_as_f(x)); }

    PHYZ_M r9 w_rot(u32 i) const { r9 r; for (u32 k = 0; k < 9u; k++) r.r[k] = ldf(FKC_W_ROT + i * 9u + k); return r; }
    PHYZ_M void set_w_rot(u32 i, r9 x) { for (u32 k = 0; k < 9u; k++) stf(FKC_W_ROT + i * 9u + k, x.r[k]); }
    PHYZ_M v3 w_pos(u32 i) const { u32 o = FKC_W_POS + i * 3u; return v3_(ldf(o), ldf(o + 1u), ldf(o + 2u)); }
    PHYZ_M void set_w_pos(u32 i, v3 x) { u32 o = FKC_W_POS + i * 3u; stf(o, x.x); stf(o + 1u, x.y); stf(o + 2u, x.z); }
    PHYZ_M r9 tree_rot(u32 i) const { r9 r; for (u32 k = 0; k < 9u; k++) r.r[k] = ldf(FKC_TREE_ROT + i * 9u + k); return r; }
    PHYZ_M void set_tree_rot(u32 i, r9 x) { for (u32 k = 0; k < 9u; k++) stf(FKC_TREE_ROT + i * 9u + k, x.r[k]); }
    PHYZ_M v3 tree_pos(u32 i) const { u32 o = FKC_TREE_POS + i * 3u; return v3_(ldf(o), ldf(o + 1u), ldf(o + 2u)); }
    PHYZ_M void set_tree_pos(u32 i, v3 x) { u32 o = FKC_TREE_POS + i * 3u; stf(o, x.x); stf(o + 1u, x.y); stf(o + 2u, x.z); }
    PHYZ_M u32 n_sel(u32 i) const { return ldu(FKC_N_SEL + i); }
    PHYZ_M void set_n_sel(u32 i, u32 x) { stu(FKC_N_SEL + i, x); }
    PHYZ_M float sel_pen(u32 k) const { return ldf(FKC_SEL_PEN + k); }
    PHYZ_M void set_sel_pen(u32 k, float x) { stf(FKC_SEL_PEN + k, x); }
    PHYZ_M u32 sel_g(u32 k) const { return ldu(FKC_SEL_G + k); }
    PHYZ_M void set_sel_g(u32 k, u32 x) { stu(FKC_SEL_G + k, x); }
    PHYZ_M u32 sel_c(u32 k) const { return ldu(FKC_SEL_C + k); }
    PHYZ_M void set_sel_c(u32 k, u32 x) { stu(FKC_SEL_C + k, x); }
    PHYZ_M u32 pl_n(u32 i) const { return ldu(FKC_PL_N + i); }
    PHYZ_M void set_pl_n(u32 i, u32 x) { stu(FKC_PL_N + i, x); }
    PHYZ_M float pl_pen(u32 k) const { return ldf(FKC_PL_PEN + k); }
    PHYZ_M void set_pl_pen(u32 k, float x) { stf(FKC_PL_PEN + k, x); }
    PHYZ_M float pl_u(u32 k) const { return ldf(FKC_PL_U + k); }
    PHYZ_M void set_pl_u(u32 k, float x) { stf(FKC_PL_U + k, x); }
    PHYZ_M float pl_v(u32 k) const { return ldf(FKC_PL_V + k); }
    PHYZ_M void set_pl_v(u32 k, float x) { stf(FKC_PL_V + k, x); }
    PHYZ_M u32 pl_gpl(u32 k) const { return ldu(FKC_PL_GPL + k); }
    PHYZ_M void set_pl_gpl(u32 k, u32 x) { stu(FKC_PL_GPL + k, x); }
};

#define PHYZ_FK_PLAIN 0u
#define PHYZ_FK_BUILD 1u
#define PHYZ_FK_REUSE 2u

template <class FC>
PHYZ_DEV void fk_world_c(const float* bodies, u32 nb, u32 nv,
                         const float* q, u32 q_base, const float* v, u32 v_base,
                         const float* qdd, float dt, bool use_free,
                         r9* w_rot, v3* w_pos, v3* w_omega, v3* w_lin,
                         FC* fc, u32 fk_mode) {
    bool fk_reuse = fk_mode == PHYZ_FK_REUSE;
    for (u32 i = 0; i < nb; i++) {
        if (fk_reuse) {
            // Cached: only the velocities are rebuilt.
            i32 parent_r = body_parent(bodies, i);
            u32 jtype_r = body_jtype(bodies, i);
            u32 v_off_r = body_voff(bodies, i);
            v3 axis_r = body_axis(bodies, i);
            float vv_r[6];
            for (u32 k = 0; k < 6u; k++) {
                u32 idx = v_base + v_off_r + k;
                if (v_off_r + k < nv) vv_r[k] = use_free ? v[idx] + dt * qdd[idx] : v[idx];
                else vv_r[k] = 0.0f;
            }
            v3 jo = v3_(0.0f, 0.0f, 0.0f);
            v3 jl = v3_(0.0f, 0.0f, 0.0f);
            if (jtype_r == 0u) {
                jo = v3_scale(axis_r, vv_r[0]);
            } else if (jtype_r == 1u) {
                jl = v3_scale(axis_r, vv_r[0]);
            } else if (jtype_r == 3u) {
                jo = v3_(vv_r[0], vv_r[1], vv_r[2]);
            } else if (jtype_r == 4u) {
                jo = v3_(vv_r[0], vv_r[1], vv_r[2]);
                jl = v3_(vv_r[3], vv_r[4], vv_r[5]);
            }
            w_rot[i] = fc->w_rot(i);
            w_pos[i] = fc->w_pos(i);
            if (parent_r < 0) {
                w_omega[i] = jo;
                w_lin[i] = jl;
            } else {
                u32 pi = (u32)parent_r;
                r9 tr = fc->tree_rot(i);
                v3 tp = fc->tree_pos(i);
                v3 pw = rot_mul(tr, w_omega[pi]);
                v3 pv = v3_sub(rot_mul(tr, w_lin[pi]),
                               rot_mul(tr, cross3(tp, w_omega[pi])));
                w_omega[i] = v3_add(pw, jo);
                w_lin[i] = v3_add(pv, jl);
            }
            continue;
        }
        i32 parent = body_parent(bodies, i);
        u32 jtype = body_jtype(bodies, i);
        u32 v_off = body_voff(bodies, i);
        v3 axis = body_axis(bodies, i);
        r9 ptj_rot = body_ptj_rot(bodies, i);
        v3 ptj_pos = body_ptj_pos(bodies, i);
        xf j = joint_transform(bodies, i, q, q_base);

        float vv[6];
        for (u32 k = 0; k < 6u; k++) {
            u32 idx = v_base + v_off + k;
            // Only the first ndof entries are read below; guard the slice end.
            if (v_off + k < nv) vv[k] = use_free ? v[idx] + dt * qdd[idx] : v[idx];
            else vv[k] = 0.0f;
        }
        v3 j_omega = v3_(0.0f, 0.0f, 0.0f);
        v3 j_lin = v3_(0.0f, 0.0f, 0.0f);
        if (jtype == 0u) {
            j_omega = v3_scale(axis, vv[0]);
        } else if (jtype == 1u) {
            j_lin = v3_scale(axis, vv[0]);
        } else if (jtype == 3u) {
            j_omega = v3_(vv[0], vv[1], vv[2]);
        } else if (jtype == 4u) {
            j_omega = v3_(vv[0], vv[1], vv[2]);
            j_lin = v3_(vv[3], vv[4], vv[5]);
        }

        // x_tree = j.compose(ptj)
        r9 tree_rot = compose_rot(j.rot, ptj_rot);
        v3 tree_pos = v3_add(ptj_pos, rot_tmul(ptj_rot, j.pos));
        r9 tree_rt = transpose_rot(tree_rot);
        if (fk_mode == PHYZ_FK_BUILD) {
            fc->set_tree_rot(i, tree_rot);
            fc->set_tree_pos(i, tree_pos);
        }
        if (parent < 0) {
            w_rot[i] = tree_rt;
            w_pos[i] = tree_pos;
            w_omega[i] = j_omega;
            w_lin[i] = j_lin;
        } else {
            u32 pi = (u32)parent;
            w_rot[i] = compose_rot(w_rot[pi], tree_rt);
            w_pos[i] = v3_add(w_pos[pi], rot_mul(w_rot[pi], tree_pos));
            // apply_motion in the body frame: w_c = R w_p, v_c = R v_p - R (p x w_p)
            v3 pw = rot_mul(tree_rot, w_omega[pi]);
            v3 pv = v3_sub(rot_mul(tree_rot, w_lin[pi]),
                           rot_mul(tree_rot, cross3(tree_pos, w_omega[pi])));
            w_omega[i] = v3_add(pw, j_omega);
            w_lin[i] = v3_add(pv, j_lin);
        }
        if (fk_mode == PHYZ_FK_BUILD) {
            fc->set_w_rot(i, w_rot[i]);
            fc->set_w_pos(i, w_pos[i]);
        }
    }
}

/// The FK chain with no cache — the standalone contact pass's call.
PHYZ_DEV void fk_world(const float* bodies, u32 nb, u32 nv,
                       const float* q, u32 q_base, const float* v, u32 v_base,
                       const float* qdd, float dt, bool use_free,
                       r9* w_rot, v3* w_pos, v3* w_omega, v3* w_lin) {
    fk_world_c(bodies, nb, nv, q, q_base, v, v_base, qdd, dt, use_free,
               w_rot, w_pos, w_omega, w_lin, (fk_cache_t*)0, PHYZ_FK_PLAIN);
}

template <class FC>
PHYZ_DEV void contact_thread_c(u32 world_idx, const float* cparams,
                               const float* bodies, const float* geometry,
                               const float* q, const float* v,
                               float* ext_forces, float* contact_state,
                               const float* hf_heights, const float* qdd,
                               FC* fc, u32 fk_mode) {
    cparams_t cp = decode_cparams(cparams);
    if (world_idx >= cp.nworld) return;

    u32 nb = cp.nbodies;
    u32 q_base = world_idx * cp.nv;
    u32 v_base = world_idx * cp.nv;
    u32 solve_mode = cp.solve_mode;
    float dt = cp.dt;

    // Clear external forces for this env
    u32 ef_env_base = world_idx * nb * 6u;
    for (u32 i = 0; i < nb; i++)
        for (u32 k = 0; k < 6u; k++)
            ext_forces[ef_env_base + i * 6u + k] = 0.0f;

    // FK: body-to-world rotation, body origin in world, and body-frame
    // spatial velocity (angular, linear). Penalty mode uses the current
    // velocity; impulse mode the FREE velocity v + dt*qdd — see the IMPULSE
    // MODE block in shaders.rs.
    r9 w_rot[MAX_BODIES];
    v3 w_pos[MAX_BODIES];
    v3 w_omega[MAX_BODIES];
    v3 w_lin[MAX_BODIES];
    fk_world_c(bodies, nb, cp.nv, q, q_base, v, v_base, qdd, dt, solve_mode != 0u,
               w_rot, w_pos, w_omega, w_lin, fc, fk_mode);

    // ── Ground / terrain contact, over every collision instance ──
    //
    // A body's shapes compete for ONE manifold, exactly as
    // `phyz_contact::solver::find_ground_contacts_model` does on the CPU: pool
    // every candidate point of every instance, rank by depth, keep the deepest
    // MAX_PTS (the CPU keeps 4). Slot identity is (body, depth rank), not
    // (body, corner) — see MAX_CONTACT_PTS in layout.rs. Mirrors shaders.rs.
    for (u32 i = 0; i < nb; i++) {
        u32 gbegin = (u32)bf(bodies, i, 33u);
        u32 gcount = (u32)bf(bodies, i, 34u);
        u32 cs_base = (world_idx * nb + i) * CS_STRIDE;
        if (gcount == 0u) {
            for (u32 k = 0; k < CS_STRIDE; k++) contact_state[cs_base + k] = 0.0f;
            continue;
        }

        // World +Z in this body's frame: support selection uses world "down"
        // even on a heightfield (the small-slope assumption).
        v3 z_body = rot_tmul(w_rot[i], v3_(0.0f, 0.0f, 1.0f));

        float sel_pen[MAX_PTS];
        u32 sel_g[MAX_PTS];
        u32 sel_c[MAX_PTS];
        u32 n_sel = 0u;
        if (fk_mode == PHYZ_FK_REUSE) {
            n_sel = fc->n_sel(i);
            for (u32 k = 0; k < n_sel; k++) {
                sel_pen[k] = fc->sel_pen(i * MAX_PTS + k);
                sel_g[k] = fc->sel_g(i * MAX_PTS + k);
                sel_c[k] = fc->sel_c(i * MAX_PTS + k);
            }
        }
        for (u32 g = gbegin; fk_mode != PHYZ_FK_REUSE && g < gbegin + gcount; g++) {
            u32 gt = (u32)geometry[g * GEOM_STRIDE];
            if (gt == 0u) continue;
            u32 np = gt == 2u ? 8u : 1u;
            for (u32 c = 0; c < np; c++) {
                v3 sp = contact_pt(geometry, g, gt, c, z_body);
                v3 spw = v3_add(w_pos[i], rot_mul(w_rot[i], sp));
                v3 tn; float th = terrain(&cp, hf_heights, spw.x, spw.y, &tn);
                float pen = tn.z * (th - spw.z);
                bool hit = solve_mode == 1u ? (pen > -cp.margin) : (pen > 0.0f);
                if (!hit) continue;
                if (n_sel < MAX_PTS) n_sel++;
                else if (pen <= sel_pen[MAX_PTS - 1u]) continue;
                u32 k = n_sel - 1u;
                while (k > 0u && sel_pen[k - 1u] < pen) {
                    sel_pen[k] = sel_pen[k - 1u];
                    sel_g[k] = sel_g[k - 1u];
                    sel_c[k] = sel_c[k - 1u];
                    k--;
                }
                sel_pen[k] = pen;
                sel_g[k] = g;
                sel_c[k] = c;
            }
        }

        if (fk_mode == PHYZ_FK_BUILD) {
            fc->set_n_sel(i, n_sel);
            for (u32 k = 0; k < n_sel; k++) {
                fc->set_sel_pen(i * MAX_PTS + k, sel_pen[k]);
                fc->set_sel_g(i * MAX_PTS + k, sel_g[k]);
                fc->set_sel_c(i * MAX_PTS + k, sel_c[k]);
            }
        }

        // Every selected point is in contact, so the manifold size IS the
        // within-body load-sharing divisor (see shaders.rs for the 8x
        // overshoot this prevents).
        u32 n_active = n_sel > 0u ? n_sel : 1u;

        v3 f_w_total = v3_(0.0f, 0.0f, 0.0f);
        float deepest = 0.0f;
        v3 deepest_w = v3_(0.0f, 0.0f, 0.0f);
        bool any_touch = false;

        // Slots beyond the manifold carry no impulse: clearing them stops a
        // warm start re-applying force at a contact that has ended.
        for (u32 k = n_sel; k < MAX_PTS; k++) {
            u32 dead = cs_base + CS_IMPULSE_OFF + k * 3u;
            contact_state[dead] = 0.0f;
            contact_state[dead + 1u] = 0.0f;
            contact_state[dead + 2u] = 0.0f;
        }

        for (u32 cpt = 0; cpt < n_sel; cpt++) {
            u32 g = sel_g[cpt];
            u32 gbase = g * GEOM_STRIDE;
            u32 gtype = (u32)geometry[gbase];
            float pt_scale = gtype == 2u ? 0.25f : 1.0f;

            v3 support = contact_pt(geometry, g, gtype, sel_c[cpt], z_body);
            v3 sup_w = v3_add(w_pos[i], rot_mul(w_rot[i], support));

            v3 n_w; float terr_h = terrain(&cp, hf_heights, sup_w.x, sup_w.y, &n_w);
            float penetration = sel_pen[cpt];
            v3 n_body = rot_tmul(w_rot[i], n_w);

            // Velocity of the contact point, body frame.
            v3 v_point = v3_add(w_lin[i], cross3(w_omega[i], support));
            float v_normal = v3_dot(v_point, n_body);

            if (solve_mode == 1u) {
                // ── Impulse mode: one staged Coulomb update for this slot ──
                u32 slot = cs_base + CS_IMPULSE_OFF + cpt * 3u;
                v3 f_c = v3_(contact_state[slot], contact_state[slot + 1u], contact_state[slot + 2u]);
                v3 t_u, t_w; contact_tangents(n_body, &t_u, &t_w);

                float b_n = v3_dot(v_point, n_body);
                float b_u = v3_dot(v_point, t_u);
                float b_w = v3_dot(v_point, t_w);

                float m_n = contact_eff_mass(bodies, i, support, n_body) / (float)n_active;
                float a_nn = 1.0f / fmax_(m_n, 1e-9f);

                float d_imp = impedance_at(&cp, penetration);
                float bias = d_imp * cp.solref_erp * fmax_(penetration, 0.0f) / fmax_(dt, 1e-9f);
                float e = effective_restitution(&cp, cp.restitution, fmin_(b_n, 0.0f));
                float b_n_eff = b_n * (1.0f + e);
                float r_n = b_n_eff - a_nn * f_c.x;
                float fn_new = fmax_((bias - r_n) / a_nn, 0.0f);

                float a_uu = (float)n_active / fmax_(contact_eff_mass(bodies, i, support, t_u), 1e-9f);
                float a_ww = (float)n_active / fmax_(contact_eff_mass(bodies, i, support, t_w), 1e-9f);
                float r_u = b_u - a_uu * f_c.y;
                float r_w = b_w - a_ww * f_c.z;
                float tu = -r_u / a_uu;
                float tw = -r_w / a_ww;
                float limit = cp.friction * fn_new;
                float tn = sqrtf(tu * tu + tw * tw);
                if (tn > limit) {
                    float sc = tn > 0.0f ? limit / tn : 0.0f;
                    tu *= sc; tw *= sc;
                }

                contact_state[slot] = fn_new;
                contact_state[slot + 1u] = tu;
                contact_state[slot + 2u] = tw;

                v3 f_body = v3_scale(v3_add(v3_add(v3_scale(n_body, fn_new), v3_scale(t_u, tu)), v3_scale(t_w, tw)),
                                     1.0f / fmax_(dt, 1e-9f));
                add_ext(ext_forces, ef_env_base + i * 6u, cross3(support, f_body), f_body);

                f_w_total = v3_add(f_w_total, rot_mul(w_rot[i], f_body));
                any_touch = true;
                if (penetration > deepest) { deepest = penetration; deepest_w = sup_w; }
                continue;
            }

            // Penalty: Kelvin-Voigt f = k*pen - d*v_n, both gains bounded by
            // the effective mass the point presents (see shaders.rs).
            float m_n = contact_eff_mass(bodies, i, support, n_body);
            float k_body = fmin_(pt_scale * geometry[gbase + 8u], max_stiffness(m_n, dt));
            float d_body = pt_scale * geometry[gbase + 9u];
            d_body = fmin_(d_body, max_damping(m_n, dt));
            float f_n = fmax_(k_body * penetration - d_body * v_normal, 0.0f);

            v3 v_tan = v3_sub(v_point, v3_scale(n_body, v_normal));
            float vt = v3_len(v_tan);
            v3 f_body = v3_scale(n_body, f_n);
            if (vt > 1e-6f) {
                v3 t_dir = v3_scale(v_tan, 1.0f / vt);
                float f_t = fmin_(coulomb(cp.friction * f_n, vt),
                                  max_damping(contact_eff_mass(bodies, i, support, t_dir), dt) * vt);
                f_body = v3_sub(f_body, v3_scale(t_dir, f_t));
            }

            add_ext(ext_forces, ef_env_base + i * 6u, cross3(support, f_body), f_body);

            f_w_total = v3_add(f_w_total, rot_mul(w_rot[i], f_body));
            any_touch = true;
            if (penetration > deepest) {
                deepest = penetration;
                deepest_w = v3_(sup_w.x, sup_w.y, terr_h);
            }
        }

        if (!any_touch) {
            for (u32 k = 0; k < 8u; k++) contact_state[cs_base + k] = 0.0f;
            continue;
        }

        contact_state[cs_base]      = 1.0f;
        contact_state[cs_base + 1u] = deepest;
        contact_state[cs_base + 2u] = deepest_w.x;
        contact_state[cs_base + 3u] = deepest_w.y;
        contact_state[cs_base + 4u] = deepest_w.z;
        contact_state[cs_base + 5u] = f_w_total.x;
        contact_state[cs_base + 6u] = f_w_total.y;
        contact_state[cs_base + 7u] = f_w_total.z;
    }

    // ── Body-attached contact faces (a deck top, a kicktail, a concave strip) ──
    //
    // A compound top surface is a SET of faces: the kicktail rises 15 deg off
    // the deck and lives on a separate flex-hinged body, so one untilted plane
    // could not express it (ecto/phyz#82) — and, since ecto/phyz#85 measured
    // the cost of the flat approximation, one face per box of a convex-
    // decomposed concave deck, so a foot rests on the RAILS where the real
    // surface is rather than on a centreline plane below them.
    //
    // Body-major, and it had to become body-major: this pass used to run
    // face-major and hand each body its warm-start ranks in FACE order, which
    // with a strip set would fill all eight slots from whichever strips came
    // first — the shallowest ones — and drop the deepest, load-bearing
    // contacts. The pool is per BODY and ranked deepest-first across every
    // face it touches, exactly as the ground branch pools every collision
    // instance against one terrain, and `n_pts_active` is the size of that
    // pooled manifold for the same reason. Mirrors shaders.rs.
    if (cp.nplanes > 0u) {
    for (u32 i = 0; i < nb; i++) {
        u32 cs_i = (world_idx * nb + i) * CS_STRIDE;
        // Readback block, cleared before it is accumulated into.
        for (u32 k = 0; k < 8u + MAX_PLANE_PTS * PLANE_DETAIL_STRIDE; k++) {
            contact_state[cs_i + CS_PLANE_RB_OFF + k] = 0.0f;
        }

        u32 gbegin = (u32)bf(bodies, i, 33u);
        u32 gcount = (u32)bf(bodies, i, 34u);

        // ── Pool this body's candidates across every face, deepest first ──
        //
        // A candidate is its position in the FACE's own 2-D frame plus its
        // depth: (u, v, penetration). Mirrors shaders.rs.
        float sel_pen[MAX_PLANE_PTS];
        float sel_u[MAX_PLANE_PTS];
        float sel_v[MAX_PLANE_PTS];
        u32 sel_g[MAX_PLANE_PTS];
        u32 sel_pl[MAX_PLANE_PTS];
        u32 n_sel = 0u;

        // A reusing sweep already has this body's clipped manifold; the
        // search below is `q`-only and cannot have moved. See `fk_cache_t`.
        if (fk_mode == PHYZ_FK_REUSE) {
            n_sel = fc->pl_n(i);
            for (u32 k = 0; k < n_sel; k++) {
                u32 s = i * MAX_PLANE_PTS + k;
                sel_pen[k] = fc->pl_pen(s);
                sel_u[k] = fc->pl_u(s);
                sel_v[k] = fc->pl_v(s);
                u32 gpl = fc->pl_gpl(s);
                sel_g[k] = gpl & 0xffffu;
                sel_pl[k] = gpl >> 16u;
            }
        }

        if (gcount > 0u && fk_mode != PHYZ_FK_REUSE) {
        for (u32 pl = 0; pl < cp.nplanes; pl++) {
            u32 pbase = cp.plane_base + pl * PLANE_STRIDE;
            u32 excl = f_as_u(geometry[pbase + 16u]);
            if ((excl & (1u << i)) != 0u) continue;
            u32 pb = f_as_u(geometry[pbase]);
            float face_half_x = geometry[pbase + 1u];
            float face_half_y = geometry[pbase + 2u];
            float max_depth = geometry[pbase + 3u];
            v3 face_o = v3_(geometry[pbase + 4u], geometry[pbase + 5u], geometry[pbase + 6u]);
            r9 face_r; for (u32 k = 0; k < 9u; k++) face_r.r[k] = geometry[pbase + 7u + k];
            v3 n_w = rot_mul(w_rot[pb], rot_tmul(face_r, v3_(0.0f, 0.0f, 1.0f)));
            v3 p0_w = v3_add(w_pos[pb], rot_mul(w_rot[pb], face_o));
            v3 n_body = rot_tmul(w_rot[i], n_w);

            for (u32 g = gbegin; g < gbegin + gcount; g++) {
                u32 gt = (u32)geometry[g * GEOM_STRIDE];
                if (gt == 0u) continue;

                if (gt != 2u) {
                    // Non-box shapes contact through their single support
                    // point; there is no face to clip.
                    v3 sp = support_point(geometry, g, n_body);
                    v3 spw = v3_add(w_pos[i], rot_mul(w_rot[i], sp));
                    v3 rf = rot_mul(face_r, v3_sub(rot_tmul(w_rot[pb], v3_sub(spw, w_pos[pb])), face_o));
                    float pen = -v3_dot(v3_sub(spw, p0_w), n_w);
                    if (pen <= 0.0f || pen > max_depth) continue;
                    if (fabsf(rf.x) > face_half_x || fabsf(rf.y) > face_half_y) continue;
                    {
                        bool keep = true;
                        if (n_sel < MAX_PLANE_PTS) n_sel++;
                        else if (pen <= sel_pen[MAX_PLANE_PTS - 1u]) keep = false;
                        if (keep) {
                            u32 q = n_sel - 1u;
                            while (q > 0u && sel_pen[q - 1u] < pen) {
                                sel_pen[q] = sel_pen[q - 1u];
                                sel_u[q] = sel_u[q - 1u];
                                sel_v[q] = sel_v[q - 1u];
                                sel_g[q] = sel_g[q - 1u];
                                sel_pl[q] = sel_pl[q - 1u];
                                q--;
                            }
                            sel_pen[q] = pen;
                            sel_u[q] = rf.x;
                            sel_v[q] = rf.y;
                            sel_g[q] = g;
                            sel_pl[q] = pl;
                        }
                    }
                    continue;
                }

                // ── A box meets a face: the CLIPPED manifold ──
                // Sutherland-Hodgman, the incident quad against the
                // rectangle's four half-planes, in the face's 2-D frame.
                v3 inc[4];
                box_incident_face(geometry, g, n_body, inc);
                float qu[4], qv[4], qp[4];
                for (u32 c = 0; c < 4u; c++) {
                    v3 spw = v3_add(w_pos[i], rot_mul(w_rot[i], inc[c]));
                    v3 rf = rot_mul(face_r, v3_sub(rot_tmul(w_rot[pb], v3_sub(spw, w_pos[pb])), face_o));
                    qu[c] = rf.x; qv[c] = rf.y;
                    qp[c] = -v3_dot(v3_sub(spw, p0_w), n_w);
                }

                // Depth over the overlap is AFFINE in the in-face
                // coordinates, so a clipped vertex's depth is exact.
                float d1u = qu[1] - qu[0], d1v = qv[1] - qv[0];
                float d2u = qu[2] - qu[0], d2v = qv[2] - qv[0];
                float det = d1u * d2v - d1v * d2u;
                if (fabsf(det) < 1e-12f) continue;
                float ga = ((qp[1] - qp[0]) * d2v - (qp[2] - qp[0]) * d1v) / det;
                float gb = ((qp[2] - qp[0]) * d1u - (qp[1] - qp[0]) * d2u) / det;

                // The 2-bit corner index is not a winding; 0,1,3,2 is the
                // quad's perimeter.
                float pu[8], pv[8];
                pu[0] = qu[0]; pv[0] = qv[0];
                pu[1] = qu[1]; pv[1] = qv[1];
                pu[2] = qu[3]; pv[2] = qv[3];
                pu[3] = qu[2]; pv[3] = qv[2];
                u32 pn = 4u;

                for (u32 e = 0; e < 4u && pn > 0u; e++) {
                    u32 axis = e >> 1u;
                    bool upper = (e & 1u) != 0u;
                    float lim = axis == 1u ? (upper ? face_half_y : -face_half_y)
                                           : (upper ? face_half_x : -face_half_x);
                    float ou[8], ov[8];
                    u32 on = 0u;
                    for (u32 k = 0; k < pn; k++) {
                        float cu = pu[k], cv = pv[k];
                        u32 kp = (k + pn - 1u) % pn;
                        float ru = pu[kp], rv = pv[kp];
                        float ca = axis == 1u ? cv : cu;
                        float pa = axis == 1u ? rv : ru;
                        bool cin = upper ? (ca <= lim) : (ca >= lim);
                        bool pin = upper ? (pa <= lim) : (pa >= lim);
                        if (cin != pin) {
                            float d = ca - pa;
                            float t = fabsf(d) > 1e-20f ? (lim - pa) / d : 0.0f;
                            if (on < 8u) { ou[on] = ru + (cu - ru) * t; ov[on] = rv + (cv - rv) * t; on++; }
                        }
                        if (cin) {
                            if (on < 8u) { ou[on] = cu; ov[on] = cv; on++; }
                        }
                    }
                    for (u32 k = 0; k < on; k++) { pu[k] = ou[k]; pv[k] = ov[k]; }
                    pn = on;
                }

                for (u32 k = 0; k < pn; k++) {
                    float cu = pu[k], cv = pv[k];
                    float pen = qp[0] + ga * (cu - qu[0]) + gb * (cv - qv[0]);
                    if (pen <= 0.0f || pen > max_depth) continue;
                    {
                        bool keep = true;
                        if (n_sel < MAX_PLANE_PTS) n_sel++;
                        else if (pen <= sel_pen[MAX_PLANE_PTS - 1u]) keep = false;
                        if (keep) {
                            u32 q = n_sel - 1u;
                            while (q > 0u && sel_pen[q - 1u] < pen) {
                                sel_pen[q] = sel_pen[q - 1u];
                                sel_u[q] = sel_u[q - 1u];
                                sel_v[q] = sel_v[q - 1u];
                                sel_g[q] = sel_g[q - 1u];
                                sel_pl[q] = sel_pl[q - 1u];
                                q--;
                            }
                            sel_pen[q] = pen;
                            sel_u[q] = cu;
                            sel_v[q] = cv;
                            sel_g[q] = g;
                            sel_pl[q] = pl;
                        }
                    }
                }
            }
        }
        }

        if (fk_mode == PHYZ_FK_BUILD) {
            fc->set_pl_n(i, n_sel);
            for (u32 k = 0; k < n_sel; k++) {
                u32 s = i * MAX_PLANE_PTS + k;
                fc->set_pl_pen(s, sel_pen[k]);
                fc->set_pl_u(s, sel_u[k]);
                fc->set_pl_v(s, sel_v[k]);
                fc->set_pl_gpl(s, (sel_g[k] & 0xffffu) | (sel_pl[k] << 16u));
            }
        }

        // Slots beyond the manifold carry a stale impulse; drop them for the
        // same reason the ground branch does.
        if (solve_mode == 1u) {
            for (u32 k = n_sel; k < MAX_PLANE_PTS; k++) {
                u32 dead = cs_i + CS_PLANE_OFF + k * 3u;
                contact_state[dead] = 0.0f;
                contact_state[dead + 1u] = 0.0f;
                contact_state[dead + 2u] = 0.0f;
            }
        }
        if (n_sel == 0u) continue;

        u32 n_pts_active = n_sel;

        for (u32 cpt = 0; cpt < n_sel; cpt++) {
            u32 slot_rank = cpt;
            u32 pl = sel_pl[cpt];
            u32 pbase = cp.plane_base + pl * PLANE_STRIDE;
            u32 pb = f_as_u(geometry[pbase]);
            v3 face_o = v3_(geometry[pbase + 4u], geometry[pbase + 5u], geometry[pbase + 6u]);
            r9 face_r; for (u32 k = 0; k < 9u; k++) face_r.r[k] = geometry[pbase + 7u + k];
            v3 n_w = rot_mul(w_rot[pb], rot_tmul(face_r, v3_(0.0f, 0.0f, 1.0f)));

            u32 g = sel_g[cpt];
            u32 gbase = g * GEOM_STRIDE;
            u32 gtype = (u32)geometry[gbase];
            float pt_scale = gtype == 2u ? 0.25f : 1.0f;

            float penetration = sel_pen[cpt];
            // The contact point is already in the face's 2-D frame, so
            // placing it needs no clamp and no second footprint pass.
            v3 r_p = v3_add(face_o, rot_tmul(face_r, v3_(sel_u[cpt], sel_v[cpt], -penetration)));
            v3 sup_w = v3_add(w_pos[pb], rot_mul(w_rot[pb], r_p));
            v3 support_c = rot_tmul(w_rot[i], v3_sub(sup_w, w_pos[i]));

            v3 v_i_w = rot_mul(w_rot[i], v3_add(w_lin[i], cross3(w_omega[i], support_c)));
            v3 v_p_w = rot_mul(w_rot[pb], v3_add(w_lin[pb], cross3(w_omega[pb], r_p)));
            v3 v_rel = v3_sub(v_i_w, v_p_w);
            float v_normal = v3_dot(v_rel, n_w);

            // Series effective mass of the pair along the normal.
            v3 n_i = rot_tmul(w_rot[i], n_w);
            v3 n_p = rot_tmul(w_rot[pb], n_w);
            float m_n = 1.0f / (1.0f / contact_eff_mass(bodies, i, support_c, n_i)
                              + 1.0f / contact_eff_mass(bodies, pb, r_p, n_p));

            if (solve_mode == 1u) {
                // ── Impulse mode on the face: both bodies move ──
                u32 pslot = cs_i + CS_PLANE_OFF + slot_rank * 3u;
                v3 pf = v3_(contact_state[pslot], contact_state[pslot + 1u], contact_state[pslot + 2u]);
                v3 pu, pw; contact_tangents(n_w, &pu, &pw);
                float bn = v_normal;
                float bu = v3_dot(v_rel, pu);
                float bw = v3_dot(v_rel, pw);

                float ann = (float)n_pts_active / fmax_(m_n, 1e-9f);
                float d_imp = impedance_at(&cp, penetration);
                float bias = d_imp * cp.solref_erp * fmax_(penetration, 0.0f) / fmax_(dt, 1e-9f);
                float ee = effective_restitution(&cp, cp.restitution, fmin_(bn, 0.0f));
                float nf = fmax_((bias - (bn * (1.0f + ee) - ann * pf.x)) / ann, 0.0f);

                v3 u_i = rot_tmul(w_rot[i], pu);
                v3 u_p = rot_tmul(w_rot[pb], pu);
                v3 w_i = rot_tmul(w_rot[i], pw);
                v3 w_p = rot_tmul(w_rot[pb], pw);
                float m_u = 1.0f / (1.0f / contact_eff_mass(bodies, i, support_c, u_i)
                                  + 1.0f / contact_eff_mass(bodies, pb, r_p, u_p));
                float m_w = 1.0f / (1.0f / contact_eff_mass(bodies, i, support_c, w_i)
                                  + 1.0f / contact_eff_mass(bodies, pb, r_p, w_p));
                float auu = (float)n_pts_active / fmax_(m_u, 1e-9f);
                float aww = (float)n_pts_active / fmax_(m_w, 1e-9f);
                float ptu = -(bu - auu * pf.y) / auu;
                float ptw = -(bw - aww * pf.z) / aww;
                float plim = cp.friction * nf;
                float ptn = sqrtf(ptu * ptu + ptw * ptw);
                if (ptn > plim) {
                    float psc = ptn > 0.0f ? plim / ptn : 0.0f;
                    ptu *= psc; ptw *= psc;
                }

                contact_state[pslot] = nf;
                contact_state[pslot + 1u] = ptu;
                contact_state[pslot + 2u] = ptw;

                v3 fw2 = v3_scale(v3_add(v3_add(v3_scale(n_w, nf), v3_scale(pu, ptu)), v3_scale(pw, ptw)),
                                  1.0f / fmax_(dt, 1e-9f));
                v3 fi2 = rot_tmul(w_rot[i], fw2);
                add_ext(ext_forces, ef_env_base + i * 6u, cross3(support_c, fi2), fi2);
                v3 fp2 = rot_tmul(w_rot[pb], v3_neg(fw2));
                add_ext(ext_forces, ef_env_base + pb * 6u, cross3(r_p, fp2), fp2);

                // Readback. The normal force is the converged normal
                // impulse over dt, the same quantity the penalty branch
                // reports.
                plane_readback(contact_state, world_idx, nb, i, pl, slot_rank,
                               penetration, sup_w, n_w, fw2, nf / fmax_(dt, 1e-9f));
                continue;
            }

            float k_body = fmin_(pt_scale * geometry[gbase + 8u], max_stiffness(m_n, dt));
            float d_body = pt_scale * geometry[gbase + 9u];
            d_body = fmin_(d_body, max_damping(m_n, dt));
            float f_n = fmax_(k_body * penetration - d_body * v_normal, 0.0f);

            v3 v_tan = v3_sub(v_rel, v3_scale(n_w, v_normal));
            float vt = v3_len(v_tan);
            v3 f_w = v3_scale(n_w, f_n);
            if (vt > 1e-6f) {
                v3 t_dir = v3_scale(v_tan, 1.0f / vt);
                v3 t_i = rot_tmul(w_rot[i], t_dir);
                v3 t_p = rot_tmul(w_rot[pb], t_dir);
                float m_t = 1.0f / (1.0f / contact_eff_mass(bodies, i, support_c, t_i)
                                  + 1.0f / contact_eff_mass(bodies, pb, r_p, t_p));
                float f_t = fmin_(coulomb(cp.friction * f_n, vt), max_damping(m_t, dt) * vt);
                f_w = v3_sub(f_w, v3_scale(t_dir, f_t));
            }

            // Action on the touching body, reaction on the face's body,
            // at one shared point.
            v3 f_i = rot_tmul(w_rot[i], f_w);
            add_ext(ext_forces, ef_env_base + i * 6u, cross3(support_c, f_i), f_i);
            v3 f_p = rot_tmul(w_rot[pb], v3_neg(f_w));
            add_ext(ext_forces, ef_env_base + pb * 6u, cross3(r_p, f_p), f_p);

            plane_readback(contact_state, world_idx, nb, i, pl, slot_rank,
                           penetration, sup_w, n_w, f_w, f_n);
        }
    }
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
                        const float* q, u32 q_base,
                        float* u_mat, float* d_mat, float* u_vec) {
    for (u32 k = 0; k < ndof; k++) {
        sv6 s_k = subspace_col(jtype, axis, k);
        sv6 uk = m6_mul_vec(ia, s_k);
        for (u32 r = 0; r < 6u; r++) u_mat[k * 6u + r] = uk.a[r];
        u_vec[k] = ctrl[v_slot + k] - damping_val * v[v_slot + k] - sv_dot(s_k, pa);
    }
    // Passive joint spring, single-DOF joints only — the exact clause the
    // CPU's passive_force applies (joint.rs): f += -k * (q - q_ref).
    // Explicit like the CPU's, so no D-matrix term.
    float stiffness_val = bf(bodies, i, 30u);
    if (ndof == 1u && stiffness_val != 0.0f) {
        float spring_ref = bf(bodies, i, 31u);
        u_vec[0] += -stiffness_val * (q[q_base + body_qoff(bodies, i)] - spring_ref);
    }
    for (u32 r = 0; r < ndof; r++) {
        sv6 s_r = subspace_col(jtype, axis, r);
        for (u32 c = 0; c < ndof; c++) {
            float acc_d = 0.0f;
            for (u32 t = 0; t < 6u; t++) acc_d += s_r.a[t] * u_mat[c * 6u + t];
            d_mat[r * ndof + c] = acc_d;
        }
        // Implicit joint damping — must match phyz_rigid::aba exactly — and
        // armature (rotor inertia) on the diagonal, like the CPU's crba/aba.
        d_mat[r * ndof + r] += dt * damping_val + bf(bodies, i, 32u);
    }
}

// The part of an ABA solve that depends only on `q`, `v` and `dt` — and so
// is the SAME for every contact sweep within one step, because a sweep
// changes `ext_forces` and nothing else. Carrying it across the sweeps of a
// step is what `PHYZ_ABA_REUSE` below buys; the values are bit-identical to
// recomputing them, since the inputs are bit-identical.
// Reached through METHODS, not members, for the same reason as `fk_cache_t`:
// the fused kernel keeps it on the local stack, the fissioned stage kernels
// keep it in a global SoA buffer with the world index fastest-varying.
//
// Field offsets, in floats per world.
#define ABC_X_ROT  0u
#define ABC_X_POS  (ABC_X_ROT  + MAX_BODIES * 9u)
#define ABC_I_A    (ABC_X_POS  + MAX_BODIES * 3u)
#define ABC_C_BIAS (ABC_I_A    + MAX_BODIES * 36u)
#define ABC_P_BIAS (ABC_C_BIAS + MAX_BODIES * 6u)
#define ABC_U_PACK (ABC_P_BIAS + MAX_BODIES * 6u)
#define ABC_DINV   (ABC_U_PACK + MAX_NV * 6u)
#define ABC_UDU_OK (ABC_DINV   + MAX_NV * 6u)
/// Floats per world. Mirrored by `layout::ABA_CACHE_FLOATS`.
#define ABA_CACHE_FLOATS (ABC_UDU_OK + MAX_BODIES)

struct aba_cache_t {
    r9 x_rot_[MAX_BODIES];
    v3 x_pos_[MAX_BODIES];
    /// Articulated inertia AFTER pass 2's child propagation, before this
    /// body's own `-W Uᵀ` — exactly what pass 2 and pass 3 read.
    m66 i_a_[MAX_BODIES];
    sv6 c_bias_[MAX_BODIES];
    /// `v × (I v)`: the bias force before `ext_forces` is subtracted.
    sv6 p_bias_[MAX_BODIES];
    /// `U = I_a S`, packed by v-slot; `D⁻¹` likewise with a fixed row stride
    /// of 6. Both are functions of `i_a` alone (see `joint_udu`: the only
    /// sweep-varying term is `u_vec`, which reads `p_a`), and `i_a[i]` is
    /// final when pass 2 reaches body `i` — so every reusing sweep reads
    /// exactly the matrix pass 2 factorised.
    float u_pack_[MAX_NV * 6u];
    float dinv_pack_[MAX_NV * 6u];
    /// Per body: did `invert_small` succeed? 0 selects the singular
    /// "treat the joint as fixed" branch, which has to be carried too.
    unsigned char udu_ok_[MAX_BODIES];

    PHYZ_M r9 x_rot(u32 i) const { return x_rot_[i]; }
    PHYZ_M void set_x_rot(u32 i, r9 x) { x_rot_[i] = x; }
    PHYZ_M v3 x_pos(u32 i) const { return x_pos_[i]; }
    PHYZ_M void set_x_pos(u32 i, v3 x) { x_pos_[i] = x; }
    PHYZ_M m66 i_a(u32 i) const { return i_a_[i]; }
    PHYZ_M void set_i_a(u32 i, const m66& x) { i_a_[i] = x; }
    PHYZ_M sv6 c_bias(u32 i) const { return c_bias_[i]; }
    PHYZ_M void set_c_bias(u32 i, sv6 x) { c_bias_[i] = x; }
    PHYZ_M sv6 p_bias(u32 i) const { return p_bias_[i]; }
    PHYZ_M void set_p_bias(u32 i, sv6 x) { p_bias_[i] = x; }
    PHYZ_M float u_pack(u32 k) const { return u_pack_[k]; }
    PHYZ_M void set_u_pack(u32 k, float x) { u_pack_[k] = x; }
    PHYZ_M float dinv_pack(u32 k) const { return dinv_pack_[k]; }
    PHYZ_M void set_dinv_pack(u32 k, float x) { dinv_pack_[k] = x; }
    PHYZ_M bool udu_ok(u32 i) const { return udu_ok_[i] != 0u; }
    PHYZ_M void set_udu_ok(u32 i, bool x) { udu_ok_[i] = x ? 1u : 0u; }
};

/// The same cache in a global SoA buffer; see `fk_cache_g`.
struct aba_cache_g {
    float* p;
    u32 w;
    u32 nworld;

    PHYZ_M float ldf(u32 o) const { return p[o * nworld + w]; }
    PHYZ_M void stf(u32 o, float x) { p[o * nworld + w] = x; }

    PHYZ_M r9 x_rot(u32 i) const { r9 r; for (u32 k = 0; k < 9u; k++) r.r[k] = ldf(ABC_X_ROT + i * 9u + k); return r; }
    PHYZ_M void set_x_rot(u32 i, r9 x) { for (u32 k = 0; k < 9u; k++) stf(ABC_X_ROT + i * 9u + k, x.r[k]); }
    PHYZ_M v3 x_pos(u32 i) const { u32 o = ABC_X_POS + i * 3u; return v3_(ldf(o), ldf(o + 1u), ldf(o + 2u)); }
    PHYZ_M void set_x_pos(u32 i, v3 x) { u32 o = ABC_X_POS + i * 3u; stf(o, x.x); stf(o + 1u, x.y); stf(o + 2u, x.z); }
    PHYZ_M m66 i_a(u32 i) const { m66 r; for (u32 k = 0; k < 36u; k++) r.m[k] = ldf(ABC_I_A + i * 36u + k); return r; }
    PHYZ_M void set_i_a(u32 i, const m66& x) { for (u32 k = 0; k < 36u; k++) stf(ABC_I_A + i * 36u + k, x.m[k]); }
    PHYZ_M sv6 c_bias(u32 i) const { sv6 r; for (u32 k = 0; k < 6u; k++) r.a[k] = ldf(ABC_C_BIAS + i * 6u + k); return r; }
    PHYZ_M void set_c_bias(u32 i, sv6 x) { for (u32 k = 0; k < 6u; k++) stf(ABC_C_BIAS + i * 6u + k, x.a[k]); }
    PHYZ_M sv6 p_bias(u32 i) const { sv6 r; for (u32 k = 0; k < 6u; k++) r.a[k] = ldf(ABC_P_BIAS + i * 6u + k); return r; }
    PHYZ_M void set_p_bias(u32 i, sv6 x) { for (u32 k = 0; k < 6u; k++) stf(ABC_P_BIAS + i * 6u + k, x.a[k]); }
    PHYZ_M float u_pack(u32 k) const { return ldf(ABC_U_PACK + k); }
    PHYZ_M void set_u_pack(u32 k, float x) { stf(ABC_U_PACK + k, x); }
    PHYZ_M float dinv_pack(u32 k) const { return ldf(ABC_DINV + k); }
    PHYZ_M void set_dinv_pack(u32 k, float x) { stf(ABC_DINV + k, x); }
    PHYZ_M bool udu_ok(u32 i) const { return f_as_u(ldf(ABC_UDU_OK + i)) != 0u; }
    PHYZ_M void set_udu_ok(u32 i, bool x) { stf(ABC_UDU_OK + i, u_as_f(x ? 1u : 0u)); }
};

#define PHYZ_ABA_BUILD 0u
#define PHYZ_ABA_REUSE 1u

/// The sweep-varying half of `joint_udu`: `u = tau − c v − Sᵀ p_a`, plus the
/// explicit joint spring. Same expressions in the same order.
PHYZ_DEV void joint_uvec(const float* bodies, u32 i, u32 jtype, u32 ndof, v3 axis,
                         float damping_val, sv6 pa,
                         const float* ctrl, const float* v, u32 v_slot,
                         const float* q, u32 q_base, float* u_vec) {
    for (u32 k = 0; k < ndof; k++) {
        sv6 s_k = subspace_col(jtype, axis, k);
        u_vec[k] = ctrl[v_slot + k] - damping_val * v[v_slot + k] - sv_dot(s_k, pa);
    }
    float stiffness_val = bf(bodies, i, 30u);
    if (ndof == 1u && stiffness_val != 0.0f) {
        float spring_ref = bf(bodies, i, 31u);
        u_vec[0] += -stiffness_val * (q[q_base + body_qoff(bodies, i)] - spring_ref);
    }
}

template <class KC>
PHYZ_DEV void aba_thread_c(u32 world_idx, u32 nworld, u32 nv, float dt, u32 nbodies,
                           float gx, float gy, float gz,
                           const float* bodies, const float* q, const float* v,
                           const float* ctrl, float* qdd, const float* ext_forces,
                           KC* kc, u32 mode) {
    if (world_idx >= nworld) return;

    u32 nb = nbodies;
    u32 q_base = world_idx * nv;  // nq == nv: ball/free use exponential coordinates
    u32 v_base = world_idx * nv;

    bool reuse = mode == PHYZ_ABA_REUSE;
    // The U/D cache is a fixed-width local array; a model wider than MAX_NV
    // simply refactorises, as before.
    bool udu_fits = nv <= MAX_NV;
    // Pass 2 reads the cache on a reusing sweep and fills it on a building
    // one; pass 3 reads it either way.
    bool udu_cached = reuse && udu_fits;
    bool udu_store = !reuse && udu_fits;
    bool udu_ready = udu_fits;
    sv6 vel[MAX_BODIES];
    sv6 p_a[MAX_BODIES];
    sv6 acc[MAX_BODIES];

    // Gravity as base acceleration: a0 = [0; -g]
    sv6 a0 = sv_(0.0f, 0.0f, 0.0f, -gx, -gy, -gz);

    // ── Pass 1: forward — velocities and bias forces ──
    //
    // Everything here but the `ext_forces` read is a function of `q`, `v` and
    // the body table, so a reusing sweep skips straight to that read.
    for (u32 i = 0; reuse && i < nb; i++) {
        u32 ef_base = (world_idx * nb + i) * 6u;
        sv6 ef;
        for (u32 kk = 0; kk < 6u; kk++) ef.a[kk] = ext_forces[ef_base + kk];
        p_a[i] = sv_sub(kc->p_bias(i), ef);
    }
    for (u32 i = 0; !reuse && i < nb; i++) {
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
        kc->set_x_rot(i, composed.rot);
        kc->set_x_pos(i, composed.pos);

        u32 ndof = joint_ndof(jtype);
        sv6 v_joint = sv_zero();
        for (u32 k = 0; k < ndof; k++)
            v_joint = sv_add(v_joint, sv_scale(subspace_col(jtype, axis, k), v[v_base + v_off + k]));

        if (parent < 0) {
            vel[i] = v_joint;
            kc->set_c_bias(i, sv_zero());
        } else {
            u32 pi = (u32)parent;
            sv6 v_parent = apply_motion(composed.rot, composed.pos, vel[pi]);
            vel[i] = sv_add(v_parent, v_joint);
            kc->set_c_bias(i, sv_cross_motion(vel[i], v_joint));
        }

        m66 ia_i = rigid_inertia_to_m6(mass, com, inertia);
        kc->set_i_a(i, ia_i);

        sv6 iv = m6_mul_vec(&ia_i, vel[i]);
        p_a[i] = sv_cross_force(vel[i], iv);
        kc->set_p_bias(i, p_a[i]);

        u32 ef_base = (world_idx * nb + i) * 6u;
        sv6 ef;
        for (u32 kk = 0; kk < 6u; kk++) ef.a[kk] = ext_forces[ef_base + kk];
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
        r9 xr = kc->x_rot(i);
        v3 xp = kc->x_pos(i);

        if (jtype == 2u) {
            if (parent >= 0) {
                u32 pi = (u32)parent;
                // The inertia half of the propagation is `q`-only: a reusing
                // sweep already has the articulated `i_a` and must NOT add
                // the child in a second time.
                if (!reuse) {
                    m66 x_mot = build_motion_transform(xr, xp);
                    m66 x_mot_t = transpose6(&x_mot);
                    m66 ia_i = kc->i_a(i);
                    m66 ia_parent = m6_XtAX(&x_mot_t, &ia_i, &x_mot);
                    m66 ia_p = kc->i_a(pi);
                    kc->set_i_a(pi, m6_add(&ia_p, &ia_parent));
                }
                sv6 p_parent = inv_apply_force(xr, xp, p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_parent);
            }
            continue;
        }

        u32 ndof = joint_ndof(jtype);
        float u_mat[36];
        float d_mat[36];
        float u_vec[6];
        bool d_ok;
        if (udu_cached) {
            // `i_a[i]` is what it was when this sweep's BUILD pass factorised
            // it, so `U` and `D⁻¹` are too. Only `u_vec` moved.
            joint_uvec(bodies, i, jtype, ndof, axis, damping_val, p_a[i],
                       ctrl, v, v_base + v_off, q, q_base, u_vec);
            for (u32 k = 0; k < ndof; k++)
                for (u32 r = 0; r < 6u; r++)
                    u_mat[k * 6u + r] = kc->u_pack((v_off + k) * 6u + r);
            for (u32 r = 0; r < ndof; r++)
                for (u32 c = 0; c < ndof; c++)
                    d_mat[r * ndof + c] = kc->dinv_pack((v_off + r) * 6u + c);
            d_ok = kc->udu_ok(i);
        } else {
            m66 ia_i = kc->i_a(i);
            joint_udu(bodies, i, jtype, ndof, axis, damping_val, dt, &ia_i, p_a[i],
                      ctrl, v, v_base + v_off, q, q_base, u_mat, d_mat, u_vec);
            d_ok = invert_small(d_mat, ndof);
            if (udu_store) {
                for (u32 k = 0; k < ndof; k++)
                    for (u32 r = 0; r < 6u; r++)
                        kc->set_u_pack((v_off + k) * 6u + r, u_mat[k * 6u + r]);
                for (u32 r = 0; r < ndof; r++)
                    for (u32 c = 0; c < ndof; c++)
                        kc->set_dinv_pack((v_off + r) * 6u + c, d_mat[r * ndof + c]);
                kc->set_udu_ok(i, d_ok);
            }
        }

        // Singular articulated inertia: treat the joint as fixed.
        if (!d_ok) {
            if (parent >= 0) {
                u32 pi = (u32)parent;
                if (!reuse) {
                    m66 x_mot_s = build_motion_transform(xr, xp);
                    m66 x_mot_st = transpose6(&x_mot_s);
                    m66 ia_i2 = kc->i_a(i);
                    m66 ia_par_s = m6_XtAX(&x_mot_st, &ia_i2, &x_mot_s);
                    m66 ia_p = kc->i_a(pi);
                    kc->set_i_a(pi, m6_add(&ia_p, &ia_par_s));
                }
                sv6 p_par_s = inv_apply_force(xr, xp, p_a[i]);
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
            m66 ia_new = kc->i_a(i);
            for (u32 c = 0; c < ndof; c++) {
                sv6 wc, uc;
                for (u32 r = 0; r < 6u; r++) { wc.a[r] = w_mat[c * 6u + r]; uc.a[r] = u_mat[c * 6u + r]; }
                m66 outer = m6_outer(wc, uc);
                ia_new = m6_sub(&ia_new, &outer);
            }

            // p_a^A = p_A + I_a^A c + W u
            sv6 ia_c = m6_mul_vec(&ia_new, kc->c_bias(i));
            sv6 wu = sv_zero();
            for (u32 c = 0; c < ndof; c++)
                for (u32 r = 0; r < 6u; r++) wu.a[r] += w_mat[c * 6u + r] * u_vec[c];
            sv6 p_new = sv_add(sv_add(p_a[i], ia_c), wu);

            if (!reuse) {
                m66 x_mot = build_motion_transform(xr, xp);
                m66 x_mot_t = transpose6(&x_mot);
                m66 ia_parent = m6_XtAX(&x_mot_t, &ia_new, &x_mot);
                m66 ia_p = kc->i_a(pi);
                kc->set_i_a(pi, m6_add(&ia_p, &ia_parent));
            }

            sv6 p_parent = inv_apply_force(xr, xp, p_new);
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

        r9 xr = kc->x_rot(i);
        v3 xp = kc->x_pos(i);
        sv6 a_parent;
        if (parent < 0) a_parent = apply_motion(xr, xp, a0);
        else            a_parent = apply_motion(xr, xp, acc[(u32)parent]);

        sv6 a_c = sv_add(a_parent, kc->c_bias(i));
        u32 ndof = joint_ndof(jtype);
        if (ndof == 0u) { acc[i] = a_c; continue; }

        // U and D⁻¹ come from pass 2 whenever the cache is wide enough for
        // this model; only `u_vec` has to be rebuilt. (The WGSL still makes
        // the other trade — rebuild everything — for private-storage room.)
        float u_mat[36];
        float d_mat[36];
        float u_vec[6];
        bool d_ok;
        if (udu_ready) {
            joint_uvec(bodies, i, jtype, ndof, axis, damping_val, p_a[i],
                       ctrl, v, v_base + v_off, q, q_base, u_vec);
            for (u32 k = 0; k < ndof; k++)
                for (u32 r = 0; r < 6u; r++)
                    u_mat[k * 6u + r] = kc->u_pack((v_off + k) * 6u + r);
            for (u32 r = 0; r < ndof; r++)
                for (u32 c = 0; c < ndof; c++)
                    d_mat[r * ndof + c] = kc->dinv_pack((v_off + r) * 6u + c);
            d_ok = kc->udu_ok(i);
        } else {
            m66 ia_i = kc->i_a(i);
            joint_udu(bodies, i, jtype, ndof, axis, damping_val, dt, &ia_i, p_a[i],
                      ctrl, v, v_base + v_off, q, q_base, u_mat, d_mat, u_vec);
            d_ok = invert_small(d_mat, ndof);
        }

        if (!d_ok) {
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


/// One ABA solve with a fresh factorisation — the standalone pass.
PHYZ_DEV void aba_thread(u32 world_idx, u32 nworld, u32 nv, float dt, u32 nbodies,
                         float gx, float gy, float gz,
                         const float* bodies, const float* q, const float* v,
                         const float* ctrl, float* qdd, const float* ext_forces) {
    aba_cache_t kc;
    aba_thread_c(world_idx, nworld, nv, dt, nbodies, gx, gy, gz,
                 bodies, q, v, ctrl, qdd, ext_forces, &kc, PHYZ_ABA_BUILD);
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
// A whole impulse-mode step (or several), in one thread per world.
//
// The unfused sequence is PD, a leading ABA, `sweeps` x [contact, ABA], and
// integrate — 2 + 2*sweeps launches, every one of them one thread per world
// with no cross-thread coupling whatsoever. The kernel boundaries therefore
// buy nothing but a barrier nobody needs, and they cost: each ABA launch
// rebuilds the articulated-body factorisation from scratch, and within a
// step that factorisation cannot have changed (`q`, `v`, `dt` are fixed
// until `integrate`; a sweep moves `ext_forces` alone). Fusing lets one
// `aba_cache_t` live across the sweeps, so sweeps 1..n skip the rigid
// inertias, the joint transforms and — the expensive part — the two 6x6
// congruence products per body that propagate inertia up the tree.
//
// The arithmetic that remains is unchanged and runs in the same order, so
// the result is bit-identical to the unfused sequence. `nsteps` steps are
// fused as well, which is legal exactly while no host input lands between
// them (one control period).
// ═══════════════════════════════════════════════════════════════════════════
PHYZ_DEV void step_impulse_thread(u32 world_idx, u32 nworld, u32 nq, u32 nv,
                                  u32 n_dofs, u32 has_pd, float dt, u32 nbodies,
                                  float gx, float gy, float gz,
                                  u32 sweeps, u32 nsteps,
                                  const float* pd_dofs, const float* targets,
                                  const float* cparams, const float* bodies,
                                  const float* geometry, const float* hf_heights,
                                  float* q, float* v, float* ctrl, float* qdd,
                                  float* ext_forces, float* contact_state) {
    if (world_idx >= nworld) return;
    aba_cache_t kc;
    fk_cache_t fc;
    for (u32 s = 0; s < nsteps; s++) {
        if (has_pd != 0u)
            for (u32 d = 0; d < n_dofs; d++)
                pd_thread(world_idx * n_dofs + d, nworld, nq, nv, n_dofs,
                          pd_dofs, q, v, targets, ctrl);

        aba_thread_c(world_idx, nworld, nv, dt, nbodies, gx, gy, gz,
                     bodies, q, v, ctrl, qdd, ext_forces, &kc, PHYZ_ABA_BUILD);
        for (u32 w = 0; w < sweeps; w++) {
            contact_thread_c(world_idx, cparams, bodies, geometry, q, v,
                             ext_forces, contact_state, hf_heights, qdd, &fc,
                             w == 0u ? PHYZ_FK_BUILD : PHYZ_FK_REUSE);
            aba_thread_c(world_idx, nworld, nv, dt, nbodies, gx, gy, gz,
                         bodies, q, v, ctrl, qdd, ext_forces, &kc, PHYZ_ABA_REUSE);
        }

        for (u32 b = 0; b < nbodies; b++)
            integrate_thread(world_idx * nbodies + b, nworld, nv, dt, nbodies,
                             q, v, qdd, bodies);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FK readout. One thread per world. Writes, per body, XF_STRIDE floats:
//   [0..9]   rotation, WORLD -> BODY, row-major (the CPU `State::body_xform`
//            convention: `rot * (p_world - pos)` is the point in the body frame)
//   [9..12]  body origin in world
//   [12..15] angular velocity, body frame
//   [15..18] linear velocity, body frame
// The same chain the contact pass computes, made readable by other passes
// (the observation pass below) and by the host (`readback_kinematics`).
// ═══════════════════════════════════════════════════════════════════════════

#define XF_STRIDE 18u

PHYZ_DEV void fk_thread(u32 world_idx, u32 nworld, u32 nv, u32 nbodies,
                        const float* bodies, const float* q, const float* v, float* xforms) {
    if (world_idx >= nworld) return;
    u32 nb = nbodies;
    r9 w_rot[MAX_BODIES];
    v3 w_pos[MAX_BODIES];
    v3 w_omega[MAX_BODIES];
    v3 w_lin[MAX_BODIES];
    fk_world(bodies, nb, nv, q, world_idx * nv, v, world_idx * nv, q, 0.0f, false,
             w_rot, w_pos, w_omega, w_lin);
    for (u32 i = 0; i < nb; i++) {
        u32 base = (world_idx * nb + i) * XF_STRIDE;
        r9 wb = transpose_rot(w_rot[i]);
        for (u32 k = 0; k < 9u; k++) xforms[base + k] = wb.r[k];
        xforms[base + 9u] = w_pos[i].x;  xforms[base + 10u] = w_pos[i].y; xforms[base + 11u] = w_pos[i].z;
        xforms[base + 12u] = w_omega[i].x; xforms[base + 13u] = w_omega[i].y; xforms[base + 14u] = w_omega[i].z;
        xforms[base + 15u] = w_lin[i].x; xforms[base + 16u] = w_lin[i].y; xforms[base + 17u] = w_lin[i].z;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Observation pass. One thread per world; one row of n_in features from a
// small op table (OBS_OP_STRIDE floats per feature: kind, a, b, c). Mirrors
// `policy_pipeline::ObsOp` — keep the kinds in step with it.
//   0 Const(c)
//   1 QMinus(a, c)        q[a] - c
//   2 V(a)                v[a]
//   3 BodyPitch(a)        atan2(-r02, r22) of body a's world->body rotation
//   4 BodyRoll(a)         atan2( r12, r22)
//   5 BodyYawError(a, c)  wrap(c - atan2(r01, r00))
//   6 BodyPosZ(a)         body origin z, world
//   7 BodyPos(a, b)       body origin, world, on axis b (0 x, 1 y, 2 z)
//   8 ComOverSupport      com - mean(support points), axis c, aux window
//                         (a, b) = (offset, n_support) into `aux`
//   9 WorldSlot(a)        wconst[world*n_wc + a]
//  10 QMinusWorld(a, b)   q[a] - wconst[world*n_wc + b]
// Writes obs[obs_off + world*n_in + k].
//
// Kinds 9 and 10 read the *per-world* constant row `wconst` (n_wc floats
// per world) instead of the op table's shared `c`: one batch can then carry
// a different command, or a different reference posture, in every world.
// The row is written between control steps and its address never moves, so
// a captured graph stays valid. A slot at or past n_wc reads zero, matching
// `ObsOp::eval_with`.
//
// Kind 8 is the one op that reduces: over every body with mass (the
// `com` table, [mass, cx, cy, cz] per body, uploaded once with the spec)
// and over a caller-supplied set of body-frame support points. Its
// payload does not fit the four-float table row, so it lives in `aux`:
//   [heading_flag, heading_body, (body, ox, oy, oz) * n_support]
// With heading_flag the horizontal pair is rotated into heading_body's
// yaw before the axis is selected. Mirrors `ObsOp::ComOverSupport`.
// ═══════════════════════════════════════════════════════════════════════════

#define OBS_OP_STRIDE 4u

PHYZ_DEV float wrap_pi(float a) {
    const float PI_F = 3.14159265358979323846f;
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

// A body-frame point in world coordinates, from an FK readout row whose
// rotation is world->body (so its transpose lifts the offset).
PHYZ_DEV void obs_point_world(const float* xf, float ox, float oy, float oz,
                              double* px, double* py, double* pz) {
    *px = (double)xf[9u]  + (double)(xf[0u] * ox + xf[3u] * oy + xf[6u] * oz);
    *py = (double)xf[10u] + (double)(xf[1u] * ox + xf[4u] * oy + xf[7u] * oz);
    *pz = (double)xf[11u] + (double)(xf[2u] * ox + xf[5u] * oy + xf[8u] * oz);
}

PHYZ_DEV void obs_thread(u32 world_idx, u32 nworld, u32 nq, u32 nv, u32 nbodies,
                         u32 n_in, u32 obs_off, u32 n_wc,
                         const float* ops, const float* aux, const float* com,
                         const float* wconst,
                         const float* q, const float* v,
                         const float* xforms, float* obs) {
    if (world_idx >= nworld) return;
    u32 qb = world_idx * nq;
    u32 vb = world_idx * nv;
    u32 wcb = world_idx * n_wc;
    for (u32 k = 0; k < n_in; k++) {
        u32 kind = (u32)ops[k * OBS_OP_STRIDE];
        u32 a = (u32)ops[k * OBS_OP_STRIDE + 1u];
        u32 b = (u32)ops[k * OBS_OP_STRIDE + 2u];
        float c = ops[k * OBS_OP_STRIDE + 3u];
        float val = 0.0f;
        if (kind == 0u) {
            val = c;
        } else if (kind == 1u) {
            val = q[qb + a] - c;
        } else if (kind == 2u) {
            val = v[vb + a];
        } else if (kind >= 3u && kind <= 6u) {
            u32 xb = (world_idx * nbodies + a) * XF_STRIDE;
            float r00 = xforms[xb + 0u], r01 = xforms[xb + 1u], r02 = xforms[xb + 2u];
            float r12 = xforms[xb + 5u], r22 = xforms[xb + 8u];
            if (kind == 3u) val = atan2f(-r02, r22);
            else if (kind == 4u) val = atan2f(r12, r22);
            else if (kind == 5u) val = wrap_pi(c - atan2f(r01, r00));
            else val = xforms[xb + 11u];
        } else if (kind == 9u) {
            val = (a < n_wc) ? wconst[wcb + a] : 0.0f;
        } else if (kind == 10u) {
            val = q[qb + a] - ((b < n_wc) ? wconst[wcb + b] : 0.0f);
        } else if (kind == 7u) {
            u32 xb = (world_idx * nbodies + a) * XF_STRIDE;
            val = xforms[xb + 9u + (b > 2u ? 2u : b)];
        } else if (kind == 8u) {
            // a = aux offset, b = number of support points, c = axis.
            u32 n_sup = b;
            if (n_sup > 0u) {
                double cx = 0.0, cy = 0.0, cz = 0.0, mtot = 0.0;
                for (u32 i = 0; i < nbodies; i++) {
                    float m = com[i * 4u];
                    if (m <= 0.0f) continue;
                    const float* xf = &xforms[(world_idx * nbodies + i) * XF_STRIDE];
                    double px, py, pz;
                    obs_point_world(xf, com[i * 4u + 1u], com[i * 4u + 2u], com[i * 4u + 3u],
                                    &px, &py, &pz);
                    cx += (double)m * px; cy += (double)m * py; cz += (double)m * pz;
                    mtot += (double)m;
                }
                if (mtot > 0.0) { cx /= mtot; cy /= mtot; cz /= mtot; }
                else { cx = 0.0; cy = 0.0; cz = 0.0; }

                double tx = 0.0, ty = 0.0, tz = 0.0;
                for (u32 i = 0; i < n_sup; i++) {
                    const float* sp = &aux[a + 2u + i * 4u];
                    const float* xf = &xforms[(world_idx * nbodies + (u32)sp[0]) * XF_STRIDE];
                    double px, py, pz;
                    obs_point_world(xf, sp[1], sp[2], sp[3], &px, &py, &pz);
                    tx += px; ty += py; tz += pz;
                }
                double inv = 1.0 / (double)n_sup;
                double ex = cx - tx * inv, ey = cy - ty * inv, ez = cz - tz * inv;

                u32 axis = (u32)c;
                if (aux[a] != 0.0f) {
                    const float* hx = &xforms[(world_idx * nbodies + (u32)aux[a + 1u]) * XF_STRIDE];
                    double yaw = atan2((double)hx[1u], (double)hx[0u]);
                    double sy = sin(yaw), cyw = cos(yaw);
                    if (axis == 0u) val = (float)(cyw * ex + sy * ey);
                    else if (axis == 1u) val = (float)(-sy * ex + cyw * ey);
                    else val = (float)ez;
                } else {
                    if (axis == 0u) val = (float)ex;
                    else if (axis == 1u) val = (float)ey;
                    else val = (float)ez;
                }
            }
        }
        obs[obs_off + world_idx * n_in + k] = val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Policy pass. One thread per world: a two-hidden-layer tanh MLP over that
// world's observation row, a diagonal-Gaussian sample around its mean, the
// sample's log-probability, and the clamped action written into the PD
// target row on top of a per-world base target. Mirrors
// `policy_pipeline::PolicySpec`; the CPU reference is `policy_reference`.
//
// weights, row-major, in this order (the same flat layout an MLP with
// layers [n_in->n_h], [n_h->n_h], [n_h->n_out] exports weight-then-bias):
//   W1[n_h*n_in] b1[n_h] W2[n_h*n_h] b2[n_h] W3[n_out*n_h] b3[n_out]
//
// Randomness: one xorshift64 stream per world (two u32 words in `rng`),
// standard normals by Box-Muller from two draws — the same recipe as
// `XorShift::normal` on the host, so a test can replay it exactly. Draw
// order per call: input noise for every entry whose scale is non-zero (in
// index order), then one normal per action.
// ═══════════════════════════════════════════════════════════════════════════

#define POLICY_MAX_IN 128u
#define POLICY_MAX_H 256u
#define POLICY_MAX_OUT 32u

struct rng64 { u32 lo, hi; };

PHYZ_DEV u32 rng_next_hi53(rng64* r, u32* lo_out) {
    // xorshift64: x ^= x<<13; x ^= x>>7; x ^= x<<17. Done on the u64 built
    // from the two words. Returns the top 21 bits and the low 32 of x>>11
    // via the out params, so the caller can form (x >> 11) as a double.
    unsigned long long x = ((unsigned long long)r->hi << 32) | (unsigned long long)r->lo;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->lo = (u32)(x & 0xffffffffull);
    r->hi = (u32)(x >> 32);
    unsigned long long top = x >> 11;
    *lo_out = (u32)(top & 0xffffffffull);
    return (u32)(top >> 32);
}

PHYZ_DEV double rng_uniform(rng64* r) {
    u32 lo;
    u32 hi = rng_next_hi53(r, &lo);
    // (x >> 11) / 2^53, exactly as the host does it.
    double top = (double)hi * 4294967296.0 + (double)lo;
    return top / 9007199254740992.0;
}

PHYZ_DEV double rng_normal(rng64* r) {
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

PHYZ_DEV void policy_thread(u32 world_idx, u32 nworld, u32 n_in, u32 n_h, u32 n_out, u32 n_dofs,
                            float act_clamp, u32 has_clamp_slots, float rho, u32 obs_off, u32 out_off,
                            const float* weights, const float* stdv, const float* in_noise,
                            float* obs, float* rng, float* z,
                            const float* act_slots, const float* act_clamp_slots,
                            const float* base_targets,
                            float* targets, float* out) {
    if (world_idx >= nworld) return;
    if (n_in > POLICY_MAX_IN || n_h > POLICY_MAX_H || n_out > POLICY_MAX_OUT) return;

    rng64 r;
    r.lo = f_as_u(rng[world_idx * 2u]);
    r.hi = f_as_u(rng[world_idx * 2u + 1u]);

    float x[POLICY_MAX_IN];
    float* row = obs + obs_off + world_idx * n_in;
    for (u32 i = 0; i < n_in; i++) {
        float xi = row[i];
        if (in_noise[i] != 0.0f) {
            xi += in_noise[i] * (float)rng_normal(&r);
            row[i] = xi;
        }
        x[i] = xi;
    }

    const float* W1 = weights;
    const float* b1 = W1 + n_h * n_in;
    const float* W2 = b1 + n_h;
    const float* b2 = W2 + n_h * n_h;
    const float* W3 = b2 + n_h;
    const float* b3 = W3 + n_out * n_h;

    float h1[POLICY_MAX_H];
    for (u32 j = 0; j < n_h; j++) {
        float s = b1[j];
        for (u32 i = 0; i < n_in; i++) s += W1[j * n_in + i] * x[i];
        h1[j] = tanhf(s);
    }
    float h2[POLICY_MAX_H];
    for (u32 j = 0; j < n_h; j++) {
        float s = b2[j];
        for (u32 i = 0; i < n_h; i++) s += W2[j * n_h + i] * h1[i];
        h2[j] = tanhf(s);
    }

    // Every PD servo tracks its base target, not just the action slots:
    // without this the non-action servos keep whatever the target buffer
    // held at allocation (zero on a fresh sim) — a different robot.
    for (u32 j = 0; j < n_dofs; j++)
        targets[world_idx * n_dofs + j] = base_targets[world_idx * n_dofs + j];

    float keep = sqrtf(fmax_(0.0f, 1.0f - rho * rho));
    const float LN_2PI = 1.8378770664093453f;
    float logp = 0.0f;
    for (u32 k = 0; k < n_out; k++) {
        float m = b3[k];
        for (u32 i = 0; i < n_h; i++) m += W3[k * n_h + i] * h2[i];
        float zk = rho * z[world_idx * n_out + k] + keep * (float)rng_normal(&r);
        z[world_idx * n_out + k] = zk;
        float sd = stdv[k];
        float a = m + sd * zk;
        float zi = (a - m) / sd;
        logp += -0.5f * zi * zi - logf(sd) - 0.5f * LN_2PI;
        out[out_off + world_idx * (n_out + 1u) + k] = a;
        u32 slot = (u32)act_slots[k];
        float lim = has_clamp_slots ? act_clamp_slots[k] : act_clamp;
        float applied = fclamp(a, -lim, lim);
        targets[world_idx * n_dofs + slot] = base_targets[world_idx * n_dofs + slot] + applied;
    }
    out[out_off + world_idx * (n_out + 1u) + n_out] = logp;

    rng[world_idx * 2u] = u_as_f(r.lo);
    rng[world_idx * 2u + 1u] = u_as_f(r.hi);
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// The FISSIONED stage passes.
//
// Same sequence as the unfused one — PD, ABA, `sweeps` x [contact, ABA],
// integrate — but with the two per-step caches carried between the launches
// in GLOBAL structure-of-arrays buffers instead of on a thread's local stack.
// That is the whole trade: the fused kernel keeps `aba_cache_t` and
// `fk_cache_t` in its frame (19.3 KB, 255 registers, ~2% occupancy on a
// 4090), while each stage kernel here carries only its own temporaries and
// reads the cache through coalesced loads — element `e` of world `w` at
// `cache[e * nworld + w]`, so a warp's 32 worlds hit 32 adjacent floats.
//
// The arithmetic is the fused kernel's, unchanged and in the same order: the
// cache round-trips through memory as the same f32/u32 bits it would have
// held in the frame. `tests/fission_step.rs` asserts that on the bits.
// ═══════════════════════════════════════════════════════════════════════════

// `MODE` is a template constant, not a kernel argument, on purpose: with it
// folded the reusing entry — the one that runs `sweeps` times a step against
// the building entry's once — drops the rigid inertias, the joint transforms
// and the congruence products from its register budget entirely.
template <u32 MODE>
PHYZ_DEV void aba_g_thread(u32 world_idx, u32 nworld, u32 nv, float dt, u32 nbodies,
                           float gx, float gy, float gz,
                           const float* bodies, const float* q, const float* v,
                           const float* ctrl, float* qdd, const float* ext_forces,
                           float* aba_cache) {
    if (world_idx >= nworld) return;
    aba_cache_g kc;
    kc.p = aba_cache;
    kc.w = world_idx;
    kc.nworld = nworld;
    aba_thread_c(world_idx, nworld, nv, dt, nbodies, gx, gy, gz,
                 bodies, q, v, ctrl, qdd, ext_forces, &kc, MODE);
}

template <u32 FK_MODE>
PHYZ_DEV void contact_g_thread(u32 world_idx, u32 nworld, const float* cparams,
                               const float* bodies, const float* geometry,
                               const float* q, const float* v,
                               float* ext_forces, float* contact_state,
                               const float* hf_heights, const float* qdd,
                               float* fk_cache) {
    if (world_idx >= nworld) return;
    fk_cache_g fc;
    fc.p = fk_cache;
    fc.w = world_idx;
    fc.nworld = nworld;
    contact_thread_c(world_idx, cparams, bodies, geometry, q, v, ext_forces,
                     contact_state, hf_heights, qdd, &fc, FK_MODE);
}

/// The contact pass with a fresh FK chain — the standalone pass.
PHYZ_DEV void contact_thread(u32 world_idx, const float* cparams,
                             const float* bodies, const float* geometry,
                             const float* q, const float* v,
                             float* ext_forces, float* contact_state,
                             const float* hf_heights, const float* qdd) {
    contact_thread_c(world_idx, cparams, bodies, geometry, q, v, ext_forces,
                     contact_state, hf_heights, qdd, (fk_cache_t*)0, PHYZ_FK_PLAIN);
}

#if PHYZ_ON_DEVICE

extern "C" __global__ void phyz_pd(u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                                   const float* dofs, const float* q, const float* v,
                                   const float* targets, float* ctrl) {
    pd_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nq, nv, n_dofs, dofs, q, v, targets, ctrl);
}

extern "C" __global__ void phyz_contact(const float* cparams,
                                        const float* bodies, const float* geometry,
                                        const float* q, const float* v,
                                        float* ext_forces, float* contact_state,
                                        const float* hf_heights, const float* qdd) {
    contact_thread(blockIdx.x * blockDim.x + threadIdx.x, cparams,
                   bodies, geometry, q, v, ext_forces, contact_state, hf_heights, qdd);
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

extern "C" __global__ void phyz_step_impulse(u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                                            u32 has_pd, float dt, u32 nbodies,
                                            float gx, float gy, float gz,
                                            u32 sweeps, u32 nsteps,
                                            const float* pd_dofs, const float* targets,
                                            const float* cparams, const float* bodies,
                                            const float* geometry, const float* hf_heights,
                                            float* q, float* v, float* ctrl, float* qdd,
                                            float* ext_forces, float* contact_state) {
    step_impulse_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nq, nv, n_dofs,
                        has_pd, dt, nbodies, gx, gy, gz, sweeps, nsteps,
                        pd_dofs, targets, cparams, bodies, geometry, hf_heights,
                        q, v, ctrl, qdd, ext_forces, contact_state);
}

#define PHYZ_ABA_ENTRY(NAME, MODE)                                              \
    extern "C" __global__ void NAME(u32 nworld, u32 nv, float dt, u32 nbodies,  \
                                    float gx, float gy, float gz,               \
                                    const float* bodies, const float* q,        \
                                    const float* v, const float* ctrl,          \
                                    float* qdd, const float* ext_forces,        \
                                    float* aba_cache) {                         \
        aba_g_thread<MODE>(blockIdx.x * blockDim.x + threadIdx.x, nworld, nv,   \
                           dt, nbodies, gx, gy, gz, bodies, q, v, ctrl, qdd,    \
                           ext_forces, aba_cache);                              \
    }

PHYZ_ABA_ENTRY(phyz_aba_c_build, PHYZ_ABA_BUILD)
PHYZ_ABA_ENTRY(phyz_aba_c_reuse, PHYZ_ABA_REUSE)

#define PHYZ_CONTACT_ENTRY(NAME, MODE)                                          \
    extern "C" __global__ void NAME(u32 nworld, const float* cparams,           \
                                    const float* bodies, const float* geometry, \
                                    const float* q, const float* v,             \
                                    float* ext_forces, float* contact_state,    \
                                    const float* hf_heights, const float* qdd,  \
                                    float* fk_cache) {                          \
        contact_g_thread<MODE>(blockIdx.x * blockDim.x + threadIdx.x, nworld,   \
                               cparams, bodies, geometry, q, v, ext_forces,     \
                               contact_state, hf_heights, qdd, fk_cache);       \
    }

PHYZ_CONTACT_ENTRY(phyz_contact_c_build, PHYZ_FK_BUILD)
PHYZ_CONTACT_ENTRY(phyz_contact_c_reuse, PHYZ_FK_REUSE)

extern "C" __global__ void phyz_fk(u32 nworld, u32 nv, u32 nbodies,
                                   const float* bodies, const float* q, const float* v, float* xforms) {
    fk_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nv, nbodies, bodies, q, v, xforms);
}

extern "C" __global__ void phyz_obs(u32 nworld, u32 nq, u32 nv, u32 nbodies, u32 n_in, u32 obs_off,
                                    u32 n_wc,
                                    const float* ops, const float* aux, const float* com,
                                    const float* wconst,
                                    const float* q, const float* v,
                                    const float* xforms, float* obs) {
    obs_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, nq, nv, nbodies, n_in, obs_off,
               n_wc, ops, aux, com, wconst, q, v, xforms, obs);
}

extern "C" __global__ void phyz_policy(u32 nworld, u32 n_in, u32 n_h, u32 n_out, u32 n_dofs,
                                       float act_clamp, u32 has_clamp_slots, float rho,
                                       u32 obs_off, u32 out_off,
                                       const float* weights, const float* stdv, const float* in_noise,
                                       float* obs, float* rng, float* z,
                                       const float* act_slots, const float* act_clamp_slots,
                                       const float* base_targets,
                                       float* targets, float* out) {
    policy_thread(blockIdx.x * blockDim.x + threadIdx.x, nworld, n_in, n_h, n_out, n_dofs,
                  act_clamp, has_clamp_slots, rho, obs_off, out_off, weights, stdv, in_noise,
                  obs, rng, z, act_slots, act_clamp_slots, base_targets, targets, out);
}

#else  // host: walk the grid serially

extern "C" void phyz_host_pd(u32 n_threads, u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                             const float* dofs, const float* q, const float* v,
                             const float* targets, float* ctrl) {
    for (u32 t = 0; t < n_threads; t++)
        pd_thread(t, nworld, nq, nv, n_dofs, dofs, q, v, targets, ctrl);
}

extern "C" void phyz_host_contact(u32 n_threads, const float* cparams,
                                  const float* bodies, const float* geometry,
                                  const float* q, const float* v,
                                  float* ext_forces, float* contact_state,
                                  const float* hf_heights, const float* qdd) {
    for (u32 t = 0; t < n_threads; t++)
        contact_thread(t, cparams, bodies, geometry, q, v, ext_forces, contact_state, hf_heights, qdd);
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

extern "C" void phyz_host_step_impulse(u32 n_threads, u32 nworld, u32 nq, u32 nv, u32 n_dofs,
                                      u32 has_pd, float dt, u32 nbodies,
                                      float gx, float gy, float gz,
                                      u32 sweeps, u32 nsteps,
                                      const float* pd_dofs, const float* targets,
                                      const float* cparams, const float* bodies,
                                      const float* geometry, const float* hf_heights,
                                      float* q, float* v, float* ctrl, float* qdd,
                                      float* ext_forces, float* contact_state) {
    for (u32 t = 0; t < n_threads; t++)
        step_impulse_thread(t, nworld, nq, nv, n_dofs, has_pd, dt, nbodies, gx, gy, gz,
                            sweeps, nsteps, pd_dofs, targets, cparams, bodies, geometry,
                            hf_heights, q, v, ctrl, qdd, ext_forces, contact_state);
}

extern "C" void phyz_host_aba_c(u32 n_threads, u32 nworld, u32 nv, float dt, u32 nbodies,
                                float gx, float gy, float gz,
                                const float* bodies, const float* q, const float* v,
                                const float* ctrl, float* qdd, const float* ext_forces,
                                float* aba_cache, u32 mode) {
    for (u32 t = 0; t < n_threads; t++) {
        if (mode == PHYZ_ABA_REUSE)
            aba_g_thread<PHYZ_ABA_REUSE>(t, nworld, nv, dt, nbodies, gx, gy, gz,
                                         bodies, q, v, ctrl, qdd, ext_forces, aba_cache);
        else
            aba_g_thread<PHYZ_ABA_BUILD>(t, nworld, nv, dt, nbodies, gx, gy, gz,
                                         bodies, q, v, ctrl, qdd, ext_forces, aba_cache);
    }
}

extern "C" void phyz_host_contact_c(u32 n_threads, u32 nworld, const float* cparams,
                                    const float* bodies, const float* geometry,
                                    const float* q, const float* v,
                                    float* ext_forces, float* contact_state,
                                    const float* hf_heights, const float* qdd,
                                    float* fk_cache, u32 fk_mode) {
    for (u32 t = 0; t < n_threads; t++) {
        if (fk_mode == PHYZ_FK_REUSE)
            contact_g_thread<PHYZ_FK_REUSE>(t, nworld, cparams, bodies, geometry, q, v,
                                            ext_forces, contact_state, hf_heights, qdd, fk_cache);
        else
            contact_g_thread<PHYZ_FK_BUILD>(t, nworld, cparams, bodies, geometry, q, v,
                                            ext_forces, contact_state, hf_heights, qdd, fk_cache);
    }
}

extern "C" void phyz_host_fk(u32 n_threads, u32 nworld, u32 nv, u32 nbodies,
                             const float* bodies, const float* q, const float* v, float* xforms) {
    for (u32 t = 0; t < n_threads; t++)
        fk_thread(t, nworld, nv, nbodies, bodies, q, v, xforms);
}

extern "C" void phyz_host_obs(u32 n_threads, u32 nworld, u32 nq, u32 nv, u32 nbodies, u32 n_in, u32 obs_off,
                              u32 n_wc,
                              const float* ops, const float* aux, const float* com,
                              const float* wconst,
                              const float* q, const float* v,
                              const float* xforms, float* obs) {
    for (u32 t = 0; t < n_threads; t++)
        obs_thread(t, nworld, nq, nv, nbodies, n_in, obs_off, n_wc, ops, aux, com, wconst,
                   q, v, xforms, obs);
}

extern "C" void phyz_host_policy(u32 n_threads, u32 nworld, u32 n_in, u32 n_h, u32 n_out, u32 n_dofs,
                                 float act_clamp, u32 has_clamp_slots, float rho,
                                 u32 obs_off, u32 out_off,
                                 const float* weights, const float* stdv, const float* in_noise,
                                 float* obs, float* rng, float* z,
                                 const float* act_slots, const float* act_clamp_slots,
                                 const float* base_targets,
                                 float* targets, float* out) {
    for (u32 t = 0; t < n_threads; t++)
        policy_thread(t, nworld, n_in, n_h, n_out, n_dofs, act_clamp, has_clamp_slots, rho,
                      obs_off, out_off, weights, stdv, in_noise, obs, rng, z,
                      act_slots, act_clamp_slots, base_targets, targets, out);
}

#endif
