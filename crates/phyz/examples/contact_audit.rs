//! contact-audit: the analytic scenario suite (docs/contact-audit.md).
//!
//! Six scenarios with closed-form answers, stepped exactly as
//! `Simulator::step_with_contacts` steps them (same detection, assembly, warm
//! start, solve and integration) but hand-rolled so the per-contact impulses
//! and the solver's iteration count are visible. The file is rev-portable: the
//! two calls between REV-SHIM markers are the only thing that differs between
//! 21a33f91 (no free-joint turn pair) and 68fc142.
//!
//! `cargo run --release -p phyz --example contact_audit` prints one JSON row
//! per (scenario, dt).

use phyz::phyz_contact::{
    ContactCache, ContactMaterial, ContactSolverConfig, assemble, find_contacts,
    find_ground_contacts_model, solve_contacts_warm,
};
use phyz::phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz::phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz::phyz_rigid::{aba, forward_kinematics, integrate_configuration};
use std::time::Instant;

const MU: f64 = 0.5;
const WORLD: usize = usize::MAX;

struct Stats {
    iters: Vec<usize>,
    nonconv: usize,
    resid_max: f64,
    solve_ns: u128,
    step_ns: u128,
    steps: usize,
}
impl Stats {
    fn new() -> Self {
        Stats { iters: vec![], nonconv: 0, resid_max: 0.0, solve_ns: 0, step_ns: 0, steps: 0 }
    }
    fn json(&self) -> String {
        let n = self.iters.len().max(1) as f64;
        format!(
            "\"iters_mean\":{:.3},\"iters_max\":{},\"nonconverged\":{},\"resid_max\":{:e},\"step_us\":{:.3},\"solve_share\":{:.3}",
            self.iters.iter().sum::<usize>() as f64 / n,
            self.iters.iter().max().copied().unwrap_or(0),
            self.nonconv,
            self.resid_max,
            self.step_ns as f64 / self.steps.max(1) as f64 / 1e3,
            self.solve_ns as f64 / self.step_ns.max(1) as f64
        )
    }
}

/// One contact after a step: world point, normal force, tangential force, pair.
#[derive(Clone)]
struct Seen {
    p: Vec3,
    fn_: f64,
    ft: f64,
    bi: usize,
    bj: usize,
}

struct World {
    model: Model,
    state: State,
    mat: ContactMaterial,
    cfg: ContactSolverConfig,
    cache: ContactCache,
    last: Vec<Seen>,
    stats: Stats,
}

impl World {
    fn new(model: Model, q: &[f64], mat: ContactMaterial) -> Self {
        let mut state = model.default_state();
        state.q = DVec::from_slice(q);
        World { model, state, mat, cfg: ContactSolverConfig::simulation(), cache: ContactCache::default(), last: vec![], stats: Stats::new() }
    }

    fn step(&mut self, ctrl: &[(usize, f64)]) {
        let t0 = Instant::now();
        let model = &self.model;
        let state = &mut self.state;
        let dt = model.dt;
        for i in 0..state.ctrl.len() {
            state.ctrl[i] = 0.0;
        }
        for &(i, f) in ctrl {
            state.ctrl[i] = f;
        }
        let v_before = state.v.clone();
        let (x, _) = forward_kinematics(model, state);
        state.body_xform = x;
        let mut contacts = find_ground_contacts_model(model, state, 0.0, self.mat.margin);
        contacts.extend(find_contacts(model, state, self.mat.margin));
        let mut qdd = aba(model, state);
        rev_strip(model, state.v.as_slice(), qdd.as_mut_slice());
        let free_qd = &state.v + &(&qdd * dt);
        self.last.clear();
        if contacts.is_empty() {
            state.v = free_qd;
        } else {
            let materials = model.contact_materials(&self.mat);
            let asm = assemble(model, state, &contacts, &materials, &free_qd, dt, &self.cfg);
            let seed = self.cache.warm_start(state, &contacts);
            let ts = Instant::now();
            let sol = solve_contacts_warm(&asm.problem, &self.cfg, &seed);
            self.stats.solve_ns += ts.elapsed().as_nanos();
            self.cache.store(state, &contacts, &sol.impulses);
            self.stats.iters.push(sol.iterations);
            if !sol.converged {
                self.stats.nonconv += 1;
            }
            self.stats.resid_max = self.stats.resid_max.max(sol.residual);
            for (c, imp) in contacts.iter().zip(&sol.impulses) {
                self.last.push(Seen {
                    p: c.contact_point,
                    fn_: imp.x / dt,
                    ft: (imp.y * imp.y + imp.z * imp.z).sqrt() / dt,
                    bi: c.body_i,
                    bj: c.body_j,
                });
            }
            state.v = &free_qd + &asm.velocity_delta(&sol.impulses);
        }
        rev_turn(model, v_before.as_slice(), state.v.as_mut_slice(), dt);
        let v = state.v.clone();
        integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
        state.time += dt;
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;
    }
}

// REV-SHIM-BEGIN (68fc142: the exact free-joint turn pair Simulator uses)
fn rev_strip(model: &Model, v: &[f64], qdd: &mut [f64]) {
    phyz::phyz_rigid::strip_free_joint_coriolis(model, v, qdd);
}
fn rev_turn(model: &Model, v_before: &[f64], v: &mut [f64], dt: f64) {
    phyz::phyz_rigid::rotate_free_joint_velocities(model, v_before, v, dt);
}
// REV-SHIM-END

fn builder(dt: f64, g: Vec3) -> ModelBuilder {
    ModelBuilder::new().gravity(g).dt(dt)
}

fn boxes(dt: f64, g: Vec3, specs: &[(Vec3, f64)]) -> Model {
    let mut b = builder(dt, g);
    for (h, m) in specs {
        let i = Vec3::new(
            m / 3.0 * (h.y * h.y + h.z * h.z),
            m / 3.0 * (h.x * h.x + h.z * h.z),
            m / 3.0 * (h.x * h.x + h.y * h.y),
        );
        b = b.add_free_body("box", -1, SpatialTransform::identity(), SpatialInertia::new(*m, Vec3::zeros(), Mat3::from_diagonal(&i)));
    }
    let mut model = b.build();
    for (k, (h, _)) in specs.iter().enumerate() {
        model.bodies[k].geometry = Some(Geometry::Box { half_extents: *h });
    }
    model
}

fn down() -> Vec3 {
    Vec3::new(0.0, 0.0, -GRAVITY)
}

fn row(name: &str, dt: f64, body: String, w: &World) {
    println!("{{\"scenario\":\"{name}\",\"dt\":{dt},\"engine\":\"phyz {}\",{body},{}}}", env!("CARGO_PKG_VERSION"), w.stats.json());
}

fn rest(dt: f64) {
    let a = 0.1;
    let mut w = World::new(boxes(dt, down(), &[(Vec3::new(a, a, a), 1.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a], ContactMaterial::default());
    for _ in 0..(10.0 / dt).round() as usize {
        w.step(&[]);
    }
    let fns: Vec<f64> = w.last.iter().map(|s| s.fn_).collect();
    let tot: f64 = fns.iter().sum();
    let cerr = fns.iter().map(|f| (f - GRAVITY / 4.0).abs()).fold(0.0, f64::max);
    let ft = w.last.iter().map(|s| s.ft).fold(0.0, f64::max);
    let q = &w.state.q;
    row("a_rest", dt, format!(
        "\"n_contacts\":{},\"total_fn\":{tot},\"mg\":{},\"corner_fn_err_max\":{cerr:e},\"tangential_max\":{ft:e},\"drift_xy\":{:e},\"sink\":{:e},\"rot\":{:e}",
        fns.len(), GRAVITY, (q[3] * q[3] + q[4] * q[4]).sqrt(), a - q[5], (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()), &w);
}

fn incline(dt: f64, deg: f64) {
    let a = 0.1;
    let th = deg.to_radians();
    let g = Vec3::new(GRAVITY * th.sin(), 0.0, -GRAVITY * th.cos());
    let mut w = World::new(boxes(dt, g, &[(Vec3::new(a, a, a), 1.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a], ContactMaterial::default());
    let n = (2.0 / dt).round() as usize;
    let (mut ts, mut xs) = (vec![], vec![]);
    for _ in 0..n {
        w.step(&[]);
        ts.push(w.state.time);
        xs.push(w.state.q[3]);
    }
    let acc = 2.0 * quad_coeff(&ts[n / 2..], &xs[n / 2..]);
    let ana = (GRAVITY * (th.sin() - MU * th.cos())).max(0.0);
    row(&format!("b_incline_{deg}"), dt, format!("\"accel\":{acc},\"analytic\":{ana},\"err\":{:e},\"disp\":{:e}", acc - ana, xs[n - 1]), &w);
}

/// Least-squares leading coefficient of a quadratic fit.
fn quad_coeff(t: &[f64], x: &[f64]) -> f64 {
    let n = t.len() as f64;
    let tm = t.iter().sum::<f64>() / n;
    let mut s = [[0.0f64; 3]; 3];
    let mut r = [0.0f64; 3];
    for (&ti, &xi) in t.iter().zip(x) {
        let u = ti - tm;
        let b = [1.0, u, u * u];
        for i in 0..3 {
            r[i] += b[i] * xi;
            for j in 0..3 {
                s[i][j] += b[i] * b[j];
            }
        }
    }
    // Cramer on the 3x3.
    let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut m2 = s;
    for i in 0..3 {
        m2[i][2] = r[i];
    }
    det(m2) / det(s)
}

fn drop_sphere(dt: f64, e: f64) {
    let r = 0.05;
    let i = 0.4 * r * r;
    let mut model = builder(dt, down())
        .add_free_body("ball", -1, SpatialTransform::identity(), SpatialInertia::new(1.0, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))))
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: r });
    let mat = ContactMaterial { restitution: e, ..ContactMaterial::default() };
    let h0 = 0.5;
    let mut w = World::new(model, &[0.0, 0.0, 0.0, 0.0, 0.0, r + h0], mat);
    let (mut maxpen, mut settle, mut bounces, mut vprev) = (0.0f64, None, 0, 0.0);
    let (mut first_apex, mut rising) = (0.0f64, false);
    for _ in 0..(3.0 / dt).round() as usize {
        w.step(&[]);
        let pen = r - w.state.q[5];
        maxpen = maxpen.max(pen);
        let vz = w.state.v[5];
        if vprev < -1e-3 && vz > 1e-3 {
            bounces += 1;
            if bounces == 1 {
                rising = true;
            }
        }
        if rising {
            if vz > 0.0 {
                first_apex = first_apex.max(-pen);
            } else {
                rising = false;
            }
        }
        vprev = vz;
        if settle.is_none() && w.state.time > 0.4 && vz.abs() < 1e-3 && pen.abs() < 5e-3 {
            settle = Some(w.state.time);
        }
    }
    let ratio = (first_apex / h0).sqrt();
    row(&format!("c_sphere_drop_e{e}"), dt, format!(
        "\"max_pen\":{maxpen:e},\"rest_pen\":{:e},\"settle_t\":{},\"rebounds\":{bounces},\"e_eff\":{ratio},\"e_nominal\":{e}",
        r - w.state.q[5], settle.map_or("null".into(), |t: f64| format!("{t}"))), &w);
}

fn drop_box(dt: f64) {
    let a = 0.1;
    let mut w = World::new(boxes(dt, down(), &[(Vec3::new(a, a, a), 1.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a + 0.3], ContactMaterial::default());
    let (mut maxpen, mut settle) = (0.0f64, None);
    for _ in 0..(3.0 / dt).round() as usize {
        w.step(&[]);
        maxpen = maxpen.max(a - w.state.q[5]);
        if settle.is_none() && w.state.time > 0.2 && w.state.v.norm() < 1e-3 {
            settle = Some(w.state.time);
        }
    }
    let q = &w.state.q;
    let tilt = (q[0] * q[0] + q[1] * q[1]).sqrt().to_degrees();
    let cerr = w.last.iter().map(|s| (s.fn_ - GRAVITY / 4.0).abs()).fold(0.0, f64::max);
    row("c_box_drop", dt, format!(
        "\"max_pen\":{maxpen:e},\"rest_pen\":{:e},\"settle_t\":{},\"tilt_deg\":{tilt:e},\"n_contacts\":{},\"corner_fn_err_max\":{cerr:e}",
        a - q[5], settle.map_or("null".into(), |t: f64| format!("{t}")), w.last.len()), &w);
}

fn push(dt: f64, f: f64) {
    let a = 0.1;
    let mut w = World::new(boxes(dt, down(), &[(Vec3::new(a, a, a), 1.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a], ContactMaterial::default());
    for _ in 0..(2.0 / dt).round() as usize {
        w.step(&[(3, f)]);
    }
    let fr: Vec<f64> = w.last.iter().filter(|s| s.p.x > w.state.q[3]).map(|s| s.fn_).collect();
    let bk: Vec<f64> = w.last.iter().filter(|s| s.p.x < w.state.q[3]).map(|s| s.fn_).collect();
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    row("d_push_loadshift", dt, format!(
        "\"front_each\":{},\"back_each\":{},\"front_ana\":{},\"back_ana\":{},\"slide\":{:e},\"n_contacts\":{}",
        mean(&fr), mean(&bk), (GRAVITY + f) / 4.0, (GRAVITY - f) / 4.0, w.state.q[3], w.last.len()), &w);
}

fn tip(dt: f64, deg: f64) {
    let (hx, hz) = (0.05, 0.1);
    let th = deg.to_radians();
    let cx = -hx * th.cos() + hz * th.sin();
    let cz = hx * th.sin() + hz * th.cos();
    let mut w = World::new(boxes(dt, down(), &[(Vec3::new(hx, 0.05, hz), 1.0)]), &[0.0, th, 0.0, cx, 0.0, cz + 1e-5], ContactMaterial::default());
    for _ in 0..(2.0 / dt).round() as usize {
        w.step(&[]);
    }
    let q = &w.state.q;
    let fin = (q[0] * q[0] + q[1] * q[1]).sqrt().to_degrees();
    row(&format!("d_tip_{deg}"), dt, format!("\"final_tilt\":{fin},\"toppled\":{},\"analytic_critical\":{}", fin > 45.0, (hx / hz).atan().to_degrees()), &w);
}

fn torsion(dt: f64, tau: f64) {
    let a = 0.1;
    let cap = MU * GRAVITY * a * 2f64.sqrt();
    let izz = (2.0 * a) * (2.0 * a) / 6.0;
    let mut w = World::new(boxes(dt, down(), &[(Vec3::new(a, a, a), 1.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a], ContactMaterial::default());
    for _ in 0..(1.0 / dt).round() as usize {
        w.step(&[(2, tau)]);
    }
    row(&format!("e_torsion_{tau}"), dt, format!("\"wz\":{},\"cap4corner\":{cap},\"wz_analytic\":{}", w.state.v[2], ((tau - cap) / izz).max(0.0)), &w);
}

fn stack(dt: f64) {
    let a = 0.1;
    let h = Vec3::new(a, a, a);
    let mut w = World::new(boxes(dt, down(), &[(h, 1.0), (h, 2.0)]), &[0.0, 0.0, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0 * a], ContactMaterial::default());
    for _ in 0..(10.0 / dt).round() as usize {
        w.step(&[]);
    }
    let g: Vec<&Seen> = w.last.iter().filter(|s| s.bj == WORLD || s.bi == WORLD).collect();
    let it: Vec<&Seen> = w.last.iter().filter(|s| s.bj != WORLD && s.bi != WORLD).collect();
    let q = &w.state.q;
    row("f_stack", dt, format!(
        "\"ground_total\":{},\"ground_ana\":{},\"inter_total\":{},\"inter_ana\":{},\"n_ground\":{},\"n_inter\":{},\"top_drift_xy\":{:e},\"top_sink\":{:e}",
        g.iter().map(|s| s.fn_).sum::<f64>(), 3.0 * GRAVITY, it.iter().map(|s| s.fn_).sum::<f64>(), 2.0 * GRAVITY, g.len(), it.len(),
        (q[9] * q[9] + q[10] * q[10]).sqrt(), 3.0 * a - q[11]), &w);
}

fn main() {
    let only: Option<String> = std::env::args().nth(1);
    for dt in [0.00025, 0.0005, 0.001] {
        let run = |n: &str| only.as_deref().is_none_or(|o| n.starts_with(o));
        if run("a") { rest(dt); }
        if run("b") { for d in [20.0, 26.0, 27.5, 35.0] { incline(dt, d); } }
        if run("c") { drop_sphere(dt, 0.0); drop_sphere(dt, 0.5); drop_box(dt); }
        if run("d") { push(dt, 2.0); tip(dt, 25.0); tip(dt, 28.0); }
        if run("e") { torsion(dt, 0.5); torsion(dt, 0.9); }
        if run("f") { stack(dt); }
    }
}
