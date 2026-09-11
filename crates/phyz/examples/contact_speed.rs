//! contact-speed: where the K1 step's time goes (docs/contact-speed.md).
//!
//! The step is `Simulator::step_with_contacts`'s, hand-rolled (the audit's
//! harness shape) so each stage can be timed and its heap allocations counted:
//! FK, ground detection, body-body detection, ABA, assembly, warm start,
//! solve, velocity update, integration. A second, untimed pass gives the
//! headline µs/step (thread CPU time, min and median over runs); a third ("anatomy") freezes sampled states and times the
//! insides of detection and assembly by re-running their public pieces.
//!
//! Every row carries `state_hash` (FNV-1a over the final `q` and `v` bits):
//! a pure-speed change must leave it unchanged.
//!
//! Scenes: `k1u_*` is the K1 as ipse's `StandingRig` builds it (URDF
//! collision primitives, the vendor MJCF's foot pads and armature, STL-fitted
//! boxes for mesh-only links: 20 shapes); `k1m_*` is the vendor MJCF as
//! phyz-mjcf loads it (MuJoCo's own 20 geoms, the 6 mesh links as `Mesh`). Plus two analytic box
//! scenes. `K1_DIR` overrides the asset directory.
//!
//! `cargo run --release -p phyz --example contact_speed [scene-filter]`

use phyz::phyz_collision::{self as pc, AABB, sweep_and_prune};
use phyz::phyz_contact::{
    ContactCache, ContactMaterial, ContactSolverConfig, assemble, find_contacts,
    find_ground_contacts_model, solve_contacts_warm,
};
use phyz::phyz_math::{DMat, GRAVITY, Mat3, SpatialInertia, SpatialTransform, SpatialTransformExt, Vec3};
use phyz::phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz::phyz_rigid::{
    aba, crba, forward_kinematics, integrate_configuration, relative_point_jacobian,
    rotate_free_joint_velocities, strip_free_joint_coriolis,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// ───────────────────────── counting allocator ─────────────────────────

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(n as u64, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.load(Relaxed)
}

// ───────────────────────── the K1 ─────────────────────────

#[path = "../tests/support/k1.rs"]
mod k1;
use k1::*;

// ───────────────────────── the clock ─────────────────────────

// Thread CPU time, not wall time: the bench machine is shared, and a
// descheduled thread must not bill its wait to the step.
#[cfg(target_os = "macos")]
fn cpu_ns() -> u64 {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
    }
    // CLOCK_THREAD_CPUTIME_ID
    unsafe { clock_gettime_nsec_np(16) }
}
#[cfg(not(target_os = "macos"))]
fn cpu_ns() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

// ───────────────────────── the box scenes ─────────────────────────

fn boxes(dt: f64, specs: &[(Vec3, f64)]) -> Model {
    let mut b = ModelBuilder::new().gravity(Vec3::new(0.0, 0.0, -GRAVITY)).dt(dt);
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

// ───────────────────────── the step ─────────────────────────

const STAGES: [&str; 9] =
    ["fk", "detect_ground", "detect_body", "aba", "assemble", "warm_start", "solve", "vel_update", "integrate"];

#[derive(Default, Clone)]
struct Prof {
    ns: [u128; 9],
    al: [u64; 9],
    steps: u64,
    contacts_ground: u64,
    contacts_body: u64,
    solves: u64,
    iters: u64,
    nonconv: u64,
}

struct Scene {
    name: String,
    model: Model,
    state: State,
    k1: Option<(K1Map, String)>,
    secs: f64,
}

fn apply_ctrl(sc: &Scene, state: &mut State) {
    for i in 0..state.ctrl.len() {
        state.ctrl[i] = 0.0;
    }
    if let Some((map, script)) = &sc.k1 {
        k1_ctrl(map, script, state);
    }
}

/// One step, `Simulator::step_with_contacts` exactly (FK first, as the audit
/// harness does; `Simulator` additionally re-runs FK at the end).
fn step(model: &Model, state: &mut State, cache: &mut ContactCache, mat: &ContactMaterial, cfg: &ContactSolverConfig, prof: Option<&mut Prof>) -> usize {
    let mut p = prof;
    let mut t = cpu_ns();
    let mut a = allocs();
    let mut mark = |p: &mut Option<&mut Prof>, i: usize| {
        if let Some(p) = p.as_deref_mut() {
            let now = cpu_ns();
            let an = allocs();
            p.ns[i] += (now - t) as u128;
            p.al[i] += an - a;
            t = cpu_ns();
            a = allocs();
        }
    };
    let dt = model.dt;
    let v_before = state.v.clone();
    let (x, _) = forward_kinematics(model, state);
    state.body_xform = x;
    mark(&mut p, 0);
    let mut contacts = find_ground_contacts_model(model, state, 0.0, mat.margin);
    let ng = contacts.len();
    mark(&mut p, 1);
    contacts.extend(find_contacts(model, state, mat.margin));
    mark(&mut p, 2);
    let mut qdd = aba(model, state);
    strip_free_joint_coriolis(model, state.v.as_slice(), qdd.as_mut_slice());
    let free_qd = &state.v + &(&qdd * dt);
    mark(&mut p, 3);
    let mut iters = 0;
    let mut conv = true;
    if contacts.is_empty() {
        state.v = free_qd;
    } else {
        let materials = model.contact_materials(mat);
        let asm = assemble(model, state, &contacts, &materials, &free_qd, dt, cfg);
        mark(&mut p, 4);
        let seed = cache.warm_start(state, &contacts);
        mark(&mut p, 5);
        let sol = solve_contacts_warm(&asm.problem, cfg, &seed);
        iters = sol.iterations;
        conv = sol.converged;
        mark(&mut p, 6);
        cache.store(state, &contacts, &sol.impulses);
        state.v = &free_qd + &asm.velocity_delta(&sol.impulses);
        mark(&mut p, 7);
    }
    rotate_free_joint_velocities(model, v_before.as_slice(), state.v.as_mut_slice(), dt);
    let v = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
    state.time += dt;
    mark(&mut p, 8);
    if let Some(p) = p {
        p.steps += 1;
        p.contacts_ground += ng as u64;
        p.contacts_body += (contacts.len() - ng) as u64;
        if !contacts.is_empty() {
            p.solves += 1;
            p.iters += iters as u64;
            p.nonconv += u64::from(!conv);
        }
    }
    contacts.len()
}

fn fnv(state: &State) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in state.q.iter().chain(state.v.iter()) {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ───────────────────────── anatomy ─────────────────────────

/// A replica of `phyz_contact::solver::placed_shapes` + `convert_geometry`
/// (private there), for counting and timing the detection pieces.
struct Placed {
    body: usize,
    geom: pc::Geometry,
    pos: Vec3,
    rot: Mat3,
}

fn conv(g: &Geometry) -> pc::Geometry {
    match g {
        Geometry::Sphere { radius } => pc::Geometry::Sphere { radius: *radius },
        Geometry::Capsule { radius, length } => pc::Geometry::Capsule { radius: *radius, length: *length },
        Geometry::Box { half_extents } => pc::Geometry::Box { half_extents: *half_extents },
        Geometry::Cylinder { radius, height } => pc::Geometry::Cylinder { radius: *radius, height: *height },
        Geometry::Mesh { vertices, faces } => pc::Geometry::Mesh { vertices: vertices.clone(), faces: faces.clone() },
        Geometry::Plane { normal } => pc::Geometry::Plane { normal: *normal },
    }
}

fn kind(g: &pc::Geometry) -> &'static str {
    match g {
        pc::Geometry::Sphere { .. } => "sphere",
        pc::Geometry::Capsule { .. } => "capsule",
        pc::Geometry::Box { .. } => "box",
        pc::Geometry::Cylinder { .. } => "cylinder",
        pc::Geometry::Mesh { .. } => "mesh",
        pc::Geometry::Plane { .. } => "plane",
        #[allow(unreachable_patterns)]
        _ => "other",
    }
}

fn placed(model: &Model, state: &State) -> Vec<Placed> {
    let mut out = Vec::new();
    for (i, body) in model.bodies.iter().enumerate() {
        let x = &state.body_xform[i];
        let mut push = |g: &Geometry, o: &SpatialTransform| {
            let sx = SpatialTransform::new(o.rot * x.rot, x.body_to_world_point(o.pos));
            out.push(Placed { body: i, geom: conv(g), pos: sx.pos, rot: sx.rot.transpose() });
        };
        if body.collisions.is_empty() {
            if let Some(g) = &body.geometry {
                push(g, &SpatialTransform::identity());
            }
        } else {
            for inst in &body.collisions {
                push(&inst.geometry, &inst.origin);
            }
        }
    }
    out
}

#[derive(Default)]
struct Anatomy {
    samples: u64,
    shapes: u64,
    bp_pairs: u64,
    np_calls: u64,
    np_hits: u64,
    ns: HashMap<&'static str, f64>,
    /// per shape pair: (tests, hits)
    pairs: HashMap<(usize, usize), (u64, u64)>,
    pair_kind: HashMap<(usize, usize), String>,
    np_kind_ns: HashMap<String, (f64, u64)>,
    jac_nnz_cols: u64,
    nv: usize,
    contacts: u64,
}

fn time_reps<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = cpu_ns();
    for _ in 0..reps {
        f();
    }
    (cpu_ns() - t) as f64 / reps as f64
}

fn invert_symmetric(m: &DMat) -> DMat {
    let n = m.nrows();
    let mut a = vec![0.0; n * n];
    let mut inv = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            a[r * n + c] = m[(r, c)];
        }
        inv[r * n + r] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].abs() < 1e-14 {
            continue;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= f * a[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    DMat::from_fn(n, n, |r, c| inv[r * n + c])
}

fn anatomy(model: &Model, state: &State, mat: &ContactMaterial, an: &mut Anatomy, reps: usize) {
    let margin = mat.margin;
    let add = |an: &mut Anatomy, k: &'static str, v: f64| *an.ns.entry(k).or_default() += v;
    an.samples += 1;
    an.nv = model.nv;

    // detection pieces
    let sh = placed(model, state);
    an.shapes += sh.len() as u64;
    add(an, "det.placed_shapes", time_reps(reps, || {
        std::hint::black_box(placed(model, state));
    }));
    let aabbs: Vec<AABB> = sh.iter().map(|s| AABB::from_geometry(&s.geom, &s.pos, &s.rot).expanded(0.5 * margin)).collect();
    add(an, "det.aabbs", time_reps(reps, || {
        std::hint::black_box(sh.iter().map(|s| AABB::from_geometry(&s.geom, &s.pos, &s.rot).expanded(0.5 * margin)).collect::<Vec<_>>());
    }));
    let pairs = sweep_and_prune(&aabbs);
    an.bp_pairs += pairs.len() as u64;
    add(an, "det.sweep_and_prune", time_reps(reps, || {
        std::hint::black_box(sweep_and_prune(&aabbs));
    }));
    add(an, "det.weld_groups", time_reps(reps, || {
        std::hint::black_box(model.weld_groups());
    }));
    let welds = model.weld_groups();
    let passed: Vec<(usize, usize)> = pairs.iter().copied().filter(|&(i, j)| model.may_collide(sh[i].body, sh[j].body, &welds)).collect();
    add(an, "det.may_collide", time_reps(reps, || {
        std::hint::black_box(pairs.iter().filter(|&&(i, j)| model.may_collide(sh[i].body, sh[j].body, &welds)).count());
    }));
    let mut np_total = 0.0;
    for &(i, j) in &passed {
        let (a, b) = (&sh[i], &sh[j]);
        let m = pc::contact_manifold_within(&a.geom, &b.geom, &a.pos, &a.rot, &b.pos, &b.rot, margin);
        let ns = time_reps(reps, || {
            std::hint::black_box(pc::contact_manifold_within(&a.geom, &b.geom, &a.pos, &a.rot, &b.pos, &b.rot, margin));
        });
        np_total += ns;
        let hit = m.is_some_and(|m| !m.points.is_empty());
        an.np_calls += 1;
        an.np_hits += u64::from(hit);
        let e = an.pairs.entry((i, j)).or_default();
        e.0 += 1;
        e.1 += u64::from(hit);
        let mut ks = [kind(&a.geom), kind(&b.geom)];
        ks.sort();
        let key = format!("{}-{}{}", ks[0], ks[1], if hit { "+hit" } else { "" });
        an.pair_kind.entry((i, j)).or_insert_with(|| {
            format!("{}:{}/{}:{}", model.bodies[a.body].name, kind(&a.geom), model.bodies[b.body].name, kind(&b.geom))
        });
        let e = an.np_kind_ns.entry(key).or_default();
        e.0 += ns;
        e.1 += 1;
    }
    add(an, "det.narrowphase", np_total);
    add(an, "det.find_contacts_total", time_reps(reps, || {
        std::hint::black_box(find_contacts(model, state, margin));
    }));
    add(an, "det.ground_total", time_reps(reps, || {
        std::hint::black_box(find_ground_contacts_model(model, state, 0.0, margin));
    }));

    // assembly pieces
    let mut contacts = find_ground_contacts_model(model, state, 0.0, margin);
    contacts.extend(find_contacts(model, state, margin));
    an.contacts += contacts.len() as u64;
    if contacts.is_empty() {
        return;
    }
    let free_qd = state.v.clone();
    let cfg = ContactSolverConfig::simulation();
    let materials = model.contact_materials(mat);
    add(an, "asm.total", time_reps(reps, || {
        std::hint::black_box(assemble(model, state, &contacts, &materials, &free_qd, model.dt, &cfg));
    }));
    add(an, "asm.fk", time_reps(reps, || {
        std::hint::black_box(forward_kinematics(model, state));
    }));
    add(an, "asm.crba", time_reps(reps, || {
        std::hint::black_box(crba(model, state));
    }));
    let mass = crba(model, state);
    add(an, "asm.invert", time_reps(reps, || {
        std::hint::black_box(invert_symmetric(&mass));
    }));
    let inv = invert_symmetric(&mass);
    let (xf, _) = forward_kinematics(model, state);
    let jac = |c: &pc::Collision| relative_point_jacobian(model, &xf, c.body_i, c.attachment_j(), c.contact_point);
    add(an, "asm.jacobians", time_reps(reps, || {
        for c in &contacts {
            std::hint::black_box(jac(c));
        }
    }));
    let js: Vec<DMat> = contacts.iter().map(jac).collect();
    for j in &js {
        an.jac_nnz_cols += (0..model.nv).filter(|&c| (0..3).any(|r| j[(r, c)] != 0.0)).count() as u64;
    }
    let nv = model.nv;
    add(an, "asm.minv_jt", time_reps(reps, || {
        for jc in &js {
            let mut m = DMat::zeros(nv, 3);
            for r in 0..nv {
                for k in 0..3 {
                    let mut acc = 0.0;
                    for col in 0..nv {
                        acc += inv[(r, col)] * jc[(k, col)];
                    }
                    m[(r, k)] = acc;
                }
            }
            std::hint::black_box(m);
        }
    }));
    let n = js.len();
    add(an, "asm.delassus", time_reps(reps, || {
        let dim = 3 * n;
        let mut d = vec![0.0; dim * dim];
        for a in 0..n {
            for b in 0..n {
                for r in 0..3 {
                    for k in 0..3 {
                        let mut acc = 0.0;
                        for col in 0..nv {
                            acc += js[a][(r, col)] * js[b][(k, col)];
                        }
                        d[(3 * a + r) * dim + 3 * b + k] = acc;
                    }
                }
            }
        }
        std::hint::black_box(d);
    }));
}

// ───────────────────────── main ─────────────────────────

fn scenes(dt: f64) -> Vec<Scene> {
    let mut v = Vec::new();
    for (tag, build) in [("k1u", urdf_k1 as fn(f64) -> Option<Model>), ("k1m", mjcf_k1)] {
        let Some(model) = build(dt) else {
            eprintln!("contact_speed: K1 assets not found under {} — skipping {tag}", k1_dir().display());
            continue;
        };
        for (script, secs) in [("stance", 2.0), ("single", 1.0), ("step", 3.0)] {
            let map = k1_map(&model);
            let state = k1_state(&model, &map);
            v.push(Scene { name: format!("{tag}_{script}"), model: model.clone(), state, k1: Some((map, script.into())), secs });
        }
    }
    let a = 0.1;
    let m = boxes(dt, &[(Vec3::new(a, a, a), 1.0)]);
    let mut s = m.default_state();
    s.q[5] = a;
    v.push(Scene { name: "box_rest".into(), model: m, state: s, k1: None, secs: 2.0 });
    let m = boxes(dt, &[(Vec3::new(a, a, a), 1.0), (Vec3::new(a, a, a), 1.0)]);
    let mut s = m.default_state();
    s.q[5] = a;
    s.q[11] = 3.0 * a;
    v.push(Scene { name: "box_stack".into(), model: m, state: s, k1: None, secs: 2.0 });
    v
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let dt = 1e-3;
    let reps: usize = std::env::var("CS_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let mat = ContactMaterial::default();
    let cfg = ContactSolverConfig::simulation();
    for sc in scenes(dt) {
        if !sc.name.contains(&filter) {
            continue;
        }
        let n = (sc.secs / dt).round() as usize;
        let model = &sc.model;

        // 1. clean µs/step: `reps` identical runs, median of means.
        let mut runs = Vec::new();
        let mut hash = 0;
        let mut trunk_z_end = f64::NAN;
        for _ in 0..reps {
            let mut state = sc.state.clone();
            let mut cache = ContactCache::default();
            let t = cpu_ns();
            for _ in 0..n {
                apply_ctrl(&sc, &mut state);
                step(model, &mut state, &mut cache, &mat, &cfg, None);
            }
            runs.push((cpu_ns() - t) as f64 / n as f64 / 1e3);
            let h = fnv(&state);
            assert!(hash == 0 || hash == h, "non-deterministic run");
            hash = h;
            let (x, _) = forward_kinematics(model, &state);
            trunk_z_end = x[model.body_index("Trunk").unwrap_or(0)].pos.z;
        }
        runs.sort_by(f64::total_cmp);
        let us = runs[runs.len() / 2];

        // 2. instrumented pass: per-stage time + allocations; anatomy samples.
        let mut prof = Prof::default();
        let mut an = Anatomy::default();
        let mut state = sc.state.clone();
        let mut cache = ContactCache::default();
        let a0 = allocs();
        let every = (n / 20).max(1);
        for i in 0..n {
            apply_ctrl(&sc, &mut state);
            if i % every == every / 2 {
                let mut s = state.clone();
                let (x, _) = forward_kinematics(model, &s);
                s.body_xform = x;
                anatomy(model, &s, &mat, &mut an, 50);
            }
            step(model, &mut state, &mut cache, &mat, &cfg, Some(&mut prof));
        }
        let _ = a0;
        let tot: u128 = prof.ns.iter().sum();
        let st = prof.steps.max(1) as f64;
        let stages = STAGES
            .iter()
            .enumerate()
            .map(|(i, s)| format!("\"{s}\":[{:.3},{:.3},{:.2}]", prof.ns[i] as f64 / st / 1e3, prof.ns[i] as f64 / tot.max(1) as f64, prof.al[i] as f64 / st))
            .collect::<Vec<_>>()
            .join(",");
        let sm = an.samples.max(1) as f64;
        let mut ns: Vec<_> = an.ns.iter().collect();
        ns.sort_by(|a, b| a.0.cmp(b.0));
        let anat = ns.iter().map(|(k, v)| format!("\"{k}\":{:.3}", *v / sm / 1e3)).collect::<Vec<_>>().join(",");
        let mut kinds: Vec<_> = an.np_kind_ns.iter().collect();
        kinds.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let kinds = kinds.iter().map(|(k, (t, c))| format!("\"{k}\":[{:.3},{:.2}]", t / sm / 1e3, *c as f64 / sm)).collect::<Vec<_>>().join(",");
        let never: Vec<String> = an.pairs.iter().filter(|(_, (_, h))| *h == 0).map(|(k, _)| format!("\"{}\"", an.pair_kind[k])).collect();
        let ever: Vec<String> = an.pairs.iter().filter(|(_, (_, h))| *h > 0).map(|(k, (t, h))| format!("\"{} {h}/{t}\"", an.pair_kind[k])).collect();
        println!(
            "{{\"scene\":\"{}\",\"dt\":{dt},\"steps\":{n},\"nv\":{},\"nbodies\":{},\"us_per_step\":{us:.3},\"us_min\":{:.3},\"us_runs\":[{}],\"state_hash\":\"{hash:016x}\",\"trunk_z_end\":{trunk_z_end:.4},\
\"contacts_ground\":{:.2},\"contacts_body\":{:.2},\"iters_mean\":{:.2},\"nonconverged\":{},\"allocs_per_step\":{:.1},\
\"stages_us_share_allocs\":{{{stages}}},\"anatomy_us\":{{{anat}}},\"shapes\":{:.1},\"bp_pairs\":{:.2},\"np_calls\":{:.2},\"np_hits\":{:.2},\"jac_nnz_cols_mean\":{:.1},\"np_by_kind_us_calls\":{{{kinds}}},\"np_pairs_never_hit\":[{}],\"np_pairs_hit\":[{}]}}",
            sc.name, model.nv, model.bodies.len(), runs[0],
            runs.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(","),
            prof.contacts_ground as f64 / st, prof.contacts_body as f64 / st, prof.iters as f64 / prof.solves.max(1) as f64, prof.nonconv,
            prof.al.iter().sum::<u64>() as f64 / st,
            an.shapes as f64 / sm, an.bp_pairs as f64 / sm, an.np_calls as f64 / sm, an.np_hits as f64 / sm,
            an.jac_nnz_cols as f64 / an.contacts.max(1) as f64,
            never.join(","), ever.join(","),
        );
    }
}
