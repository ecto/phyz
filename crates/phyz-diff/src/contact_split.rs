//! The contact step as three kernels, with the dense work in loops.
//!
//! [`crate::contact_step::ContactStep`] unrolls everything into one
//! straight-line graph. At the K1's eight sole corners and five sweeps that is
//! ~125k nodes and 5 MB of C, bound by instruction fetch rather than
//! arithmetic, and four fifths of it is dense linear algebra (24 triangular
//! solves for `M⁻¹ Jᵀ`, the 24 x 24 Delassus product) that gains nothing from
//! unrolling. Here the step is split where the work changes shape:
//!
//! 1. **kinematics** (traced, [`ContactKernels::kinematics`]): `(q, v, ctrl)`
//!    → everything with tree structure: ABA, CRBA and Featherstone's LTL
//!    factor `M = Lᵀ L` (no fill-in in tree order), `Y = L⁻ᵀ Jᵀ` (nonzero
//!    only on each point's path to the root), `A = Y Yᵀ`, `b = J qd_free`,
//!    and per-point bias, impedance coefficient and activity, laid out as
//!    [`RecordLayout`]. Constant folding drops the structural zeros, so `A`
//!    costs ~12 terms per same-foot pair and ~6 per cross-foot pair;
//! 2. **solve** (loops, [`ContactKernels::solve`] in f64 and
//!    [`ContactKernels::solve_source`] as C or MSL): staged-Coulomb PGS with
//!    the sweep count a runtime argument, then
//!    `v1 = qd_free + M⁻¹ Jᵀ f = qd_free + L⁻¹ Σ_r f_r Y_r`;
//! 3. **finish** (traced): `(q, v, v1)` → `(q', v')`.
//!
//! The solve is the unrolled trace's PGS operation for operation; `A` is
//! formed as `Y Yᵀ` instead of `J (M⁻¹ Jᵀ)`, equal up to round-off.

use phyz_contact::{ContactMaterial, ContactSolverConfig, convex::ContactCoupling};
use phyz_model::Model;
use tang_expr::codegen::{ComputeKernel, Dialect};
use tang_expr::compile::CompiledMany;
use tang_expr::{ExprGraph, ExprId, trace};

use crate::contact_step::{
    ContactPoint, DISC_ITERS, Kinematics, dof_parents, traced_kinematics, traced_ltl,
};
use crate::forward::{fold_constants, traced_finish_step};

/// Offsets of each block in one environment's kinematics record.
#[derive(Clone, Copy, Debug)]
pub struct RecordLayout {
    /// `qd_free`, `nv`.
    pub free_qd: usize,
    /// LTL factor `L` of `M = Lᵀ L`, lower triangle packed by rows
    /// (`i (i + 1) / 2 + j`), `nv (nv + 1) / 2`.
    pub l: usize,
    /// `1 / L_ii`, `nv`.
    pub inv_diag: usize,
    /// `A = J M⁻¹ Jᵀ`, row-major `dim x dim`.
    pub a: usize,
    /// `b = J qd_free`, `dim`.
    pub b: usize,
    /// `Y = L⁻ᵀ Jᵀ`, `dim` rows of `nv`.
    pub y: usize,
    /// Stabilization bias, one per point.
    pub bias: usize,
    /// Impedance coefficient `(1 - d) / d`, one per point.
    pub dcoef: usize,
    /// Activity mask, one per point.
    pub active: usize,
    /// Record length.
    pub len: usize,
    /// Per-environment scratch the C/MSL solve needs (`nv`, for `v1`).
    pub scratch: usize,
}

/// The split contact step for one model and point set.
pub struct ContactKernels {
    /// `(q, v, ctrl)` → kinematics record.
    pub kinematics: ExprGraph,
    /// Kinematics outputs, in [`RecordLayout`] order.
    pub kinematics_out: Vec<ExprId>,
    /// `(q, v, v1)` → `(q', v')`.
    pub finish: ExprGraph,
    /// Finish outputs.
    pub finish_out: Vec<ExprId>,
    /// Position DOFs.
    pub nq: usize,
    /// Velocity DOFs.
    pub nv: usize,
    /// Controls.
    pub nu: usize,
    /// Contact points.
    pub n_points: usize,
    mu: f64,
    reg: f64,
    relax: f64,
}

impl ContactKernels {
    /// Trace the kinematics and finish kernels.
    pub fn build(
        model: &Model,
        points: &[ContactPoint],
        material: &ContactMaterial,
        config: &ContactSolverConfig,
    ) -> Self {
        assert!(
            material.restitution <= 0.0,
            "contact kernels: restitution must be 0"
        );
        assert!(
            !config.mujoco_compat,
            "contact kernels: mujoco_compat is not traced"
        );
        assert!(
            config.coupling == ContactCoupling::Full,
            "contact kernels: only full coupling is traced"
        );
        let (nq, nv) = (model.nq, model.nv);
        let nu = model.default_state().ctrl.len();
        let var = |i: usize| ExprId::var(i as u16);
        let lambda = dof_parents(model);
        let (kinematics, kinematics_out) = trace(|| {
            let q: Vec<ExprId> = (0..nq).map(var).collect();
            let v: Vec<ExprId> = (nq..nq + nv).map(var).collect();
            let ctrl: Vec<ExprId> = (nq + nv..nq + nv + nu).map(var).collect();
            let Kinematics {
                free_qd,
                mass,
                jac,
                bias,
                dcoef,
                active,
                ..
            } = traced_kinematics(model, points, material, config, &q, &v, &ctrl);
            let (l, inv) = traced_ltl(&mass, &lambda);
            let dot = |a: &[ExprId], b: &[ExprId]| {
                let mut s = ExprId::ZERO;
                for (x, y) in a.iter().zip(b) {
                    s += *x * *y;
                }
                s
            };
            // Y_r = L⁻ᵀ J_rᵀ: Lᵀ is upper triangular with L[m][k] ≠ 0 only
            // for m a descendant of k, so solve from the leaves up.
            let y: Vec<Vec<ExprId>> = jac
                .iter()
                .map(|j| {
                    let mut y = vec![ExprId::ZERO; nv];
                    for k in (0..nv).rev() {
                        let mut s = j[k];
                        for m in k + 1..nv {
                            s -= l[m][k] * y[m];
                        }
                        y[k] = s * inv[k];
                    }
                    y
                })
                .collect();
            let dim = y.len();
            let mut a = vec![ExprId::ZERO; dim * dim];
            for i in 0..dim {
                for k in i..dim {
                    let e = dot(&y[i], &y[k]);
                    a[i * dim + k] = e;
                    a[k * dim + i] = e;
                }
            }
            let b: Vec<ExprId> = jac.iter().map(|j| dot(j, &free_qd)).collect();
            let mut out = free_qd;
            for (i, row) in l.iter().enumerate() {
                out.extend_from_slice(&row[..=i]);
            }
            out.extend(inv);
            out.extend(a);
            out.extend(b);
            out.extend(y.into_iter().flatten());
            out.extend(bias);
            out.extend(dcoef);
            out.extend(active);
            out
        });
        let (kinematics, kinematics_out) = fold_constants(&kinematics, &kinematics_out);
        let (finish, finish_out) = trace(|| {
            let q: Vec<ExprId> = (0..nq).map(var).collect();
            let v: Vec<ExprId> = (nq..nq + nv).map(var).collect();
            let v1: Vec<ExprId> = (nq + nv..nq + 2 * nv).map(var).collect();
            let (q1, v1) = traced_finish_step(model, &q, &v, v1, model.dt);
            q1.into_iter().chain(v1).collect::<Vec<_>>()
        });
        let (finish, finish_out) = fold_constants(&finish, &finish_out);
        Self {
            kinematics,
            kinematics_out,
            finish,
            finish_out,
            nq,
            nv,
            nu,
            n_points: points.len(),
            mu: material.friction,
            reg: config.regularization,
            relax: config.relaxation,
        }
    }

    /// Where each block sits in a kinematics record.
    pub fn layout(&self) -> RecordLayout {
        let (nv, np) = (self.nv, self.n_points);
        let dim = 3 * np;
        let l = nv;
        let inv_diag = l + nv * (nv + 1) / 2;
        let a = inv_diag + nv;
        let b = a + dim * dim;
        let y = b + dim;
        let bias = y + dim * nv;
        let dcoef = bias + np;
        let active = dcoef + np;
        RecordLayout {
            free_qd: 0,
            l,
            inv_diag,
            a,
            b,
            y,
            bias,
            dcoef,
            active,
            len: active + np,
            scratch: nv,
        }
    }

    /// Inputs to the kinematics kernel: `nq + nv + nu`.
    pub fn kinematics_inputs(&self) -> usize {
        self.nq + self.nv + self.nu
    }

    /// Inputs to the finish kernel: `nq + 2 nv`.
    pub fn finish_inputs(&self) -> usize {
        self.nq + 2 * self.nv
    }

    /// Live nodes in the two traced kernels.
    pub fn live_nodes(&self) -> (usize, usize) {
        (
            self.kinematics.live_set(&self.kinematics_out).len(),
            self.finish.live_set(&self.finish_out).len(),
        )
    }

    /// f64 closures for the two traced kernels.
    pub fn compile(&self) -> (CompiledMany, CompiledMany) {
        (
            self.kinematics.compile_many(&self.kinematics_out),
            self.finish.compile_many(&self.finish_out),
        )
    }

    /// Emitted kinematics and finish kernels, renamed `k_kin` / `k_finish`.
    pub fn traced_sources(&self, dialect: Dialect) -> (String, String) {
        let rename = |k: ComputeKernel, name: &str| k.source.replace(" k0(", &format!(" {name}("));
        (
            rename(
                self.kinematics
                    .to_kernel(&self.kinematics_out, self.kinematics_inputs(), dialect),
                "k_kin",
            ),
            rename(
                self.finish
                    .to_kernel(&self.finish_out, self.finish_inputs(), dialect),
                "k_finish",
            ),
        )
    }

    /// The solve in f64, from one kinematics record and a warm start (`3n`):
    /// returns `(f, v1)`. The reference for [`Self::solve_source`].
    pub fn solve(&self, rec: &[f64], warm: &[f64], sweeps: usize) -> (Vec<f64>, Vec<f64>) {
        let lay = self.layout();
        let (nv, np) = (self.nv, self.n_points);
        let dim = 3 * np;
        let l = |i: usize, j: usize| rec[lay.l + i * (i + 1) / 2 + j];
        let inv = |i: usize| rec[lay.inv_diag + i];
        let a = &rec[lay.a..lay.a + dim * dim];
        let b = &rec[lay.b..lay.b + dim];
        let y = |r: usize, k: usize| rec[lay.y + r * nv + k];
        let act = |c: usize| rec[lay.active + c];
        let sel = |c: f64, x: f64, y: f64| if c > 0.0 { x } else { y };
        let emax = |x: f64, y: f64| sel(x - y, x, y);

        let mut f: Vec<f64> = (0..dim).map(|i| warm[i] * act(i / 3)).collect();
        for _ in 0..sweeps {
            for c in 0..np {
                let base = 3 * c;
                let mut r = [0.0; 3];
                for (row, rr) in r.iter_mut().enumerate() {
                    let mut acc = b[base + row];
                    for k in 0..np {
                        if k != c {
                            for col in 0..3 {
                                acc += a[(base + row) * dim + 3 * k + col] * f[3 * k + col];
                            }
                        }
                    }
                    *rr = acc;
                }
                let r_n = emax(rec[lay.dcoef + c] * a[base * dim + base], self.reg);
                let m = |i: usize, j: usize| {
                    let e = a[(base + i) * dim + base + j];
                    match (i == j, i) {
                        (true, 0) => e + r_n,
                        (true, _) => e + self.reg,
                        _ => e,
                    }
                };
                let unclamped =
                    (rec[lay.bias + c] - r[0] - m(0, 1) * f[base + 1] - m(0, 2) * f[base + 2])
                        / m(0, 0);
                let f_n = emax(unclamped, 0.0);
                let r_u = r[1] + m(1, 0) * f_n;
                let r_w = r[2] + m(2, 0) * f_n;
                let det = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
                let t_u = -(m(2, 2) * r_u - m(1, 2) * r_w) / det;
                let t_w = -(m(1, 1) * r_w - m(2, 1) * r_u) / det;
                let limit = self.mu * f_n;
                let t_norm = (t_u * t_u + t_w * t_w).sqrt();
                // The disc step only where its result is used: sliding, and a
                // positive limit. Same values as the trace's selects.
                let slide = t_norm - limit;
                let (g_u, g_w) = if slide > 0.0 {
                    if limit > 0.0 {
                        let d =
                            disc_step([[m(1, 1), m(1, 2)], [m(2, 1), m(2, 2)]], [r_u, r_w], limit);
                        (d[0], d[1])
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    (t_u, t_w)
                };
                let target = [f_n, g_u, g_w];
                for k in 0..3 {
                    f[base + k] = (f[base + k] + (target[k] - f[base + k]) * self.relax) * act(c);
                }
            }
        }

        // v1 = qd_free + L⁻¹ Σ_r f_r Y_r.
        let mut x = vec![0.0; nv];
        for k in 0..nv {
            let mut g: f64 = (0..dim).map(|r| y(r, k) * f[r]).sum();
            for (m, xm) in x.iter().enumerate().take(k) {
                g -= l(k, m) * xm;
            }
            x[k] = g * inv(k);
        }
        let w = x;
        let v1 = (0..nv).map(|k| rec[lay.free_qd + k] + w[k]).collect();
        (f, v1)
    }

    /// The solve as a C function or an MSL kernel, in f32:
    ///
    /// ```text
    /// k_solve(rec, warm, x, scratch, f_out, fin, count, sweeps)
    /// ```
    ///
    /// `rec` holds [`RecordLayout`] records, `warm` / `f_out` hold `3n`
    /// impulses, `x` is the kinematics kernel's input (to copy `q, v`),
    /// `scratch` is `layout().scratch` floats, and `fin` receives the finish
    /// kernel's input `[q, v, v1]`, all per environment. C loops over
    /// environments with each one's data contiguous (array of structs); MSL
    /// runs one environment per thread with every buffer slot-major
    /// (`slot * count + env`, structure of arrays) so neighbouring threads
    /// read neighbouring words. Pair it with [`Self::traced_sources_soa`].
    /// `Dialect::Wgsl` gives GLSL 450 for Vulkan instead (WGSL has no
    /// preprocessor), array of structs to pair with
    /// `traced_sources(Dialect::Wgsl)`.
    pub fn solve_source(&self, dialect: Dialect) -> String {
        let lay = self.layout();
        let (nq, nv, np) = (self.nq, self.nv, self.n_points);
        let (prelude, sig, open, close) = match dialect {
            Dialect::C => (
                "#include <math.h>\n#define IDX unsigned long\n\
                 #define REC(i) rec_[(IDX)e * NREC + (i)]\n\
                 #define SCR(i) scratch_[(IDX)e * NSCR + (i)]\n\
                 #define WARM(i) warm[(IDX)e * DIM + (i)]\n\
                 #define FOUT(i) f_out[(IDX)e * DIM + (i)]\n\
                 #define XIN(i) x[(IDX)e * NX + (i)]\n\
                 #define FIN(i) fin[(IDX)e * NFIN + (i)]\n\
                 #define SWEEPS ((int)sweeps)\n",
                "void k_solve(const float* rec_, const float* warm, const float* x, float* scratch_, float* f_out, float* fin, int count, int sweeps)",
                "    for (int e = 0; e < count; e++) {\n",
                "    }\n",
            ),
            Dialect::Msl => (
                "#include <metal_stdlib>\nusing namespace metal;\n#define IDX ulong\n\
                 #define REC(i) rec_[(IDX)(i) * count + e]\n\
                 #define SCR(i) scratch_[(IDX)(i) * count + e]\n\
                 #define WARM(i) warm[(IDX)(i) * count + e]\n\
                 #define FOUT(i) f_out[(IDX)(i) * count + e]\n\
                 #define XIN(i) x[(IDX)(i) * count + e]\n\
                 #define FIN(i) fin[(IDX)(i) * count + e]\n\
                 #define SWEEPS ((int)sweeps)\n",
                "kernel void k_solve(device const float* rec_ [[buffer(0)]], device const float* warm [[buffer(1)]], device const float* x [[buffer(2)]], device float* scratch_ [[buffer(3)]], device float* f_out [[buffer(4)]], device float* fin [[buffer(5)]], constant uint& count [[buffer(6)]], constant uint& sweeps [[buffer(7)]], uint e [[thread_position_in_grid]])",
                "    if (e >= count) { return; }\n    {\n",
                "    }\n",
            ),
            // GLSL 450 compute for Vulkan (tang-gpu's SPIR-V path), array
            // of structs like the WGSL traced kernels: bindings rec, warm,
            // x, scratch, f_out, fin, then a uniform (count, sweeps)
            Dialect::Wgsl => (
                "#version 450\n\
                 layout(local_size_x = 64) in;\n\
                 layout(std430, set = 0, binding = 0) readonly buffer B0 { float rec_[]; };\n\
                 layout(std430, set = 0, binding = 1) readonly buffer B1 { float warm[]; };\n\
                 layout(std430, set = 0, binding = 2) readonly buffer B2 { float x[]; };\n\
                 layout(std430, set = 0, binding = 3) buffer B3 { float scratch_[]; };\n\
                 layout(std430, set = 0, binding = 4) buffer B4 { float f_out[]; };\n\
                 layout(std430, set = 0, binding = 5) buffer B5 { float fin[]; };\n\
                 layout(std140, set = 0, binding = 6) uniform Params { uint count; uint sweeps; uint p2; uint p3; } params;\n\
                 #define REC(i) rec_[e * NREC + uint(i)]\n\
                 #define SCR(i) scratch_[e * NSCR + uint(i)]\n\
                 #define WARM(i) warm[e * DIM + uint(i)]\n\
                 #define FOUT(i) f_out[e * DIM + uint(i)]\n\
                 #define XIN(i) x[e * NX + uint(i)]\n\
                 #define FIN(i) fin[e * NFIN + uint(i)]\n\
                 #define SWEEPS int(params.sweeps)\n",
                "void main()",
                "    uint e = gl_GlobalInvocationID.x;\n    if (e >= params.count) { return; }\n    {\n",
                "    }\n",
            ),
            d => panic!("solve_source: no {d:?} template"),
        };
        format!(
            r#"{prelude}// Rolled contact solve (phyz-diff contact_split).
#define NP {np}
#define DIM {dim}
#define NV {nv}
#define NQ {nq}
#define NREC {nrec}
#define NSCR {nscr}
#define NX {nx}
#define NFIN {nfin}
#define OFF_L {ol}
#define OFF_INV {oinv}
#define OFF_A {oa}
#define OFF_B {ob}
#define OFF_Y {oy}
#define OFF_BIAS {obias}
#define OFF_DC {odc}
#define OFF_ACT {oact}
#define MU {mu:e}f
#define REG {reg:e}f
#define RELAX {relax:e}f
#define DISC_ITERS {disc}
#define SEL(c, a, b) ((c) > 0.0f ? (a) : (b))
#define EMAX(a, b) SEL((a) - (b), (a), (b))
#define LIDX(i, j) ((i) * ((i) + 1) / 2 + (j))
#define LL(i, j) REC(OFF_L + LIDX(i, j))
#define YY(r, k) REC(OFF_Y + (r) * NV + (k))
#define AA(i, j) REC(OFF_A + (i) * DIM + (j))

{sig} {{
{open}        float f[DIM];
        for (int i = 0; i < DIM; i++) f[i] = WARM(i) * REC(OFF_ACT + i / 3);
        for (int s = 0; s < SWEEPS; s++) {{
            for (int c = 0; c < NP; c++) {{
                int base = 3 * c;
                float r[3];
                for (int row = 0; row < 3; row++) {{
                    float acc = REC(OFF_B + base + row);
                    for (int k = 0; k < NP; k++) {{
                        if (k == c) continue;
                        acc += AA(base + row, 3 * k) * f[3 * k] + AA(base + row, 3 * k + 1) * f[3 * k + 1] + AA(base + row, 3 * k + 2) * f[3 * k + 2];
                    }}
                    r[row] = acc;
                }}
                float r_n = EMAX(REC(OFF_DC + c) * AA(base, base), REG);
                float m00 = AA(base, base) + r_n, m01 = AA(base, base + 1), m02 = AA(base, base + 2);
                float m10 = AA(base + 1, base), m11 = AA(base + 1, base + 1) + REG, m12 = AA(base + 1, base + 2);
                float m20 = AA(base + 2, base), m21 = AA(base + 2, base + 1), m22 = AA(base + 2, base + 2) + REG;
                float unclamped = (REC(OFF_BIAS + c) - r[0] - m01 * f[base + 1] - m02 * f[base + 2]) / m00;
                float f_n = EMAX(unclamped, 0.0f);
                float r_u = r[1] + m10 * f_n, r_w = r[2] + m20 * f_n;
                float det = m11 * m22 - m12 * m21;
                float t_u = -(m22 * r_u - m12 * r_w) / det;
                float t_w = -(m11 * r_w - m21 * r_u) / det;
                float limit = MU * f_n;
                float t_norm = sqrt(t_u * t_u + t_w * t_w);
                float slide = t_norm - limit;
                float g_u = t_u, g_w = t_w;
                if (slide > 0.0f) {{
                    // Sliding: the disc step, only taken where its result is used.
                    g_u = 0.0f;
                    g_w = 0.0f;
                    if (limit > 0.0f) {{
                        float k = 0.0f;
                        float tu = (m22 * -r_u - m12 * -r_w) / det, tw = (m11 * -r_w - m21 * -r_u) / det;
                        for (int it = 0; it < DISC_ITERS; it++) {{
                            float a_ = m11 + k, d_ = m22 + k, dk = a_ * d_ - m12 * m21;
                            float nrm = sqrt(tu * tu + tw * tw);
                            float su = (d_ * tu - m12 * tw) / dk, sw = (a_ * tw - m21 * tu) / dk;
                            float dn = -(tu * su + tw * sw) / nrm;
                            float phi = 1.0f / nrm - 1.0f / limit;
                            float dphi = -dn / (nrm * nrm);
                            k = EMAX(k - phi / dphi, 0.0f);
                            a_ = m11 + k; d_ = m22 + k; dk = a_ * d_ - m12 * m21;
                            tu = (d_ * -r_u - m12 * -r_w) / dk;
                            tw = (a_ * -r_w - m21 * -r_u) / dk;
                        }}
                        float sc = limit / sqrt(tu * tu + tw * tw);
                        g_u = tu * sc;
                        g_w = tw * sc;
                    }}
                }}
                float act = REC(OFF_ACT + c);
                f[base] = (f[base] + (f_n - f[base]) * RELAX) * act;
                f[base + 1] = (f[base + 1] + (g_u - f[base + 1]) * RELAX) * act;
                f[base + 2] = (f[base + 2] + (g_w - f[base + 2]) * RELAX) * act;
            }}
        }}
        for (int i = 0; i < DIM; i++) FOUT(i) = f[i];
        // v1 = qd_free + L⁻¹ Σ_r f_r Y_r, solved into scratch. (A private
        // `float z[NV]` here gave wrong results on an M4 Max; cause not
        // isolated.)
        for (int k = 0; k < NV; k++) {{
            float g = 0.0f;
            for (int r = 0; r < DIM; r++) g += YY(r, k) * f[r];
            for (int m = 0; m < k; m++) g -= LL(k, m) * SCR(m);
            SCR(k) = g * REC(OFF_INV + k);
        }}
        for (int j = 0; j < NQ + NV; j++) FIN(j) = XIN(j);
        for (int j = 0; j < NV; j++) FIN(NQ + NV + j) = REC(j) + SCR(j);
{close}}}
"#,
            dim = 3 * np,
            nrec = lay.len,
            nscr = lay.scratch,
            nx = self.kinematics_inputs(),
            nfin = self.finish_inputs(),
            ol = lay.l,
            oinv = lay.inv_diag,
            oa = lay.a,
            ob = lay.b,
            oy = lay.y,
            obias = lay.bias,
            odc = lay.dcoef,
            oact = lay.active,
            mu = self.mu,
            reg = self.reg,
            relax = self.relax,
            disc = DISC_ITERS,
        )
    }

    /// [`Self::traced_sources`] for MSL with every buffer slot-major
    /// (`slot * count + gid`), to pair with the MSL [`Self::solve_source`].
    pub fn traced_sources_soa(&self) -> (String, String) {
        let (kin, fin) = self.traced_sources(Dialect::Msl);
        (to_soa(&kin), to_soa(&fin))
    }
}

/// Rewrite tang-expr MSL indexing `buf[gid * N + k]` to `buf[k * count + gid]`.
fn to_soa(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("[gid * ") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + "[gid * ".len()..];
        let close = tail.find(']').expect("unterminated index");
        let inner = &tail[..close]; // "N + k"
        let k = inner.split(" + ").nth(1).expect("index without offset");
        out.push_str(&format!("[{k} * count + gid]"));
        rest = &tail[close + 1..];
    }
    out.push_str(rest);
    out
}

/// `contact_step::disc_block_step` in f64.
fn disc_step(m: [[f64; 2]; 2], r: [f64; 2], limit: f64) -> [f64; 2] {
    let solve = |k: f64, rhs: [f64; 2]| {
        let (a, b, c, d) = (m[0][0] + k, m[0][1], m[1][0], m[1][1] + k);
        let det = a * d - b * c;
        [
            (d * rhs[0] - b * rhs[1]) / det,
            (a * rhs[1] - c * rhs[0]) / det,
        ]
    };
    let mut k = 0.0f64;
    let mut t = solve(k, [-r[0], -r[1]]);
    for _ in 0..DISC_ITERS {
        let nrm = (t[0] * t[0] + t[1] * t[1]).sqrt();
        let st = solve(k, t);
        let dn = -(t[0] * st[0] + t[1] * st[1]) / nrm;
        let phi = 1.0 / nrm - 1.0 / limit;
        let dphi = -dn / (nrm * nrm);
        let next = k - phi / dphi;
        k = if next > 0.0 { next } else { 0.0 };
        t = solve(k, [-r[0], -r[1]]);
    }
    let s = limit / (t[0] * t[0] + t[1] * t[1]).sqrt();
    [t[0] * s, t[1] * s]
}
