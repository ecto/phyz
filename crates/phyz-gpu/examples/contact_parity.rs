//! CPU convex contact vs GPU penalty contact, on one identical initial state.
//!
//! phyz has three contact models today: `phyz-contact`'s convex impulse solve
//! (the CPU path, and the one the design doc specifies), `phyz-gpu`'s penalty
//! forces, and `phyz-diff/src/rollout/step.rs`'s per-vertex penalty with no
//! friction. This example measures the first two against each other, because
//! the practical consequence of their disagreement is that a policy trained on
//! the GPU does not work on the CPU — and the CPU is the path that matches the
//! deployed controller.
//!
//! # Reading the output
//!
//! Divergence is reported as the position and velocity gap between the two
//! engines at matched timesteps, from a shared initial state. Both engines
//! integrate the same model at the same `dt`, so at step 0 the gap is exactly
//! zero and everything after it is contact.
//!
//! # The NaN trap
//!
//! `f64::max` and `f64::min` return the *other* operand when one side is NaN
//! (IEEE 754 `maxNum` semantics, which Rust follows). So a reduction like
//! `gaps.iter().fold(0.0, f64::max)` silently skips every non-finite frame and
//! reports the largest *finite* gap — which, in a run that went NaN at step 20
//! of 2000, is whatever tiny number step 19 held. ipse hit exactly this: a
//! skate-wheel blowup reported "peak board pitch" from frame 0 and looked like
//! a clean result. Every reduction here goes through [`Gap::accumulate`], which
//! checks `is_finite` first and latches a divergence step, so a blown-up run
//! reports as blown up rather than as excellent.

use phyz::Simulator;
use phyz_contact::material::ContactMaterial;
use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::BodyContactGains;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

/// Contact response time constant, and the natural frequency it names.
/// This is MuJoCo's `solref` default, and the value the GPU tests use.
const TIME_CONST: f64 = 0.02;
const OMEGA: f64 = 1.0 / TIME_CONST;
const DT: f64 = 0.001;

/// A divergence accumulator that cannot be fooled by a non-finite frame.
#[derive(Debug, Default)]
struct Gap {
    /// Largest gap seen across all *finite* frames.
    peak: f64,
    /// Step at which either engine first produced a non-finite value, if any.
    /// `Some` here invalidates `peak` as a summary — the run did not complete.
    diverged_at: Option<usize>,
}

impl Gap {
    /// Fold one frame in. Non-finite input latches `diverged_at` instead of
    /// being silently dropped, which is the whole point of this type.
    fn accumulate(&mut self, step: usize, value: f64) {
        if !value.is_finite() {
            self.diverged_at.get_or_insert(step);
            return;
        }
        if value > self.peak {
            self.peak = value;
        }
    }

    /// A summary that says "blew up" when the run blew up.
    fn report(&self) -> String {
        match self.diverged_at {
            Some(k) => format!("NON-FINITE at step {k}"),
            None => format!("{:.3e}", self.peak),
        }
    }
}

/// Build a single free box, positioned and moving as given.
///
/// A box rather than a sphere because a box has a face, and a face is what
/// makes the two engines' contact-point choices differ: the CPU generates a
/// manifold of corner contacts, the GPU generates the single support point in
/// the ground normal direction.
fn free_box(pos: Vec3, half: Vec3, mass: f64) -> Model {
    let c = mass / 3.0;
    let inertia = Mat3::from_diagonal(&Vec3::new(
        c * (half.y * half.y + half.z * half.z),
        c * (half.x * half.x + half.z * half.z),
        c * (half.x * half.x + half.y * half.y),
    ));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::from_translation(pos),
            SpatialInertia::new(mass, Vec3::zeros(), inertia),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    model
}

/// The world position of body 0's origin, read out of a state's `q`.
///
/// A free joint's `q` is `[rot(3), pos(3)]` in phyz's layout, so the
/// translation is at indices 3..6.
fn body_pos(state: &State) -> Vec3 {
    let q = state.q.as_slice();
    Vec3::new(q[3], q[4], q[5])
}

/// Linear velocity of body 0, from `v`. A free joint's `v` is
/// `[angular(3), linear(3)]`.
fn body_linvel(state: &State) -> Vec3 {
    let v = state.v.as_slice();
    Vec3::new(v[3], v[4], v[5])
}

/// Roll a CPU trace under one contact-solver configuration.
fn cpu_trace(
    model: &Model,
    mut state: State,
    steps: usize,
    material: &ContactMaterial,
    config: phyz_contact::ContactSolverConfig,
) -> Vec<(Vec3, Vec3)> {
    let sim = Simulator::new().with_contact_config(config);
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        sim.step_with_contacts(model, &mut state, 0.0, material);
        trace.push((body_pos(&state), body_linvel(&state)));
    }
    trace
}

/// Mean position gap over the last quarter of the run.
///
/// Peak gap is dominated by the impact transient — the two engines resolve a
/// collision on slightly different steps, which produces a large
/// instantaneous difference that says little about whether they agree on the
/// physics. The settled gap is what a policy actually experiences, because a
/// policy spends its time in the sustained regime, not in the first
/// millisecond of touchdown. Reported alongside the peak, never instead of
/// it: a run that blows up late has a small settled gap and a huge peak.
fn settled_gap(a: &[(Vec3, Vec3)], b: &[(Vec3, Vec3)]) -> String {
    let start = a.len() * 3 / 4;
    let mut sum = 0.0;
    let mut n = 0usize;
    for ((ap, _), (bp, _)) in a[start..].iter().zip(&b[start..]) {
        let d = (*bp - *ap).norm();
        if !d.is_finite() {
            return "NON-FINITE".to_string();
        }
        sum += d;
        n += 1;
    }
    format!("{:.3e}", sum / n.max(1) as f64)
}

/// Peak position/velocity gap between two traces, NaN-safe.
fn trace_gap(a: &[(Vec3, Vec3)], b: &[(Vec3, Vec3)]) -> (Gap, Gap) {
    let (mut p, mut v) = (Gap::default(), Gap::default());
    for (k, ((ap, av), (bp, bv))) in a.iter().zip(b).enumerate() {
        p.accumulate(k, (*bp - *ap).norm());
        v.accumulate(k, (*bv - *av).norm());
    }
    (p, v)
}

/// Roll one scenario on both engines and report how far apart they get.
fn compare(name: &str, model: Model, init: impl Fn(&mut State), steps: usize, friction: f64) {
    // ── shared initial state ──
    let mut cpu_state = model.default_state();
    init(&mut cpu_state);
    let gpu_state = cpu_state.clone();

    let material = ContactMaterial {
        friction,
        ..ContactMaterial::default()
    };

    // ── CPU, full Delassus: the reference physics ──
    let cpu_trace_full = cpu_trace(
        &model,
        cpu_state.clone(),
        steps,
        &material,
        phyz_contact::ContactSolverConfig::simulation(),
    );

    // ── CPU, block-diagonal: the same model under the GPU's restriction.
    // The gap between this and the line above is the DELIBERATE APPROXIMATION,
    // measured in f64 with no GPU and no shader in the picture.
    let cpu_trace_bd = cpu_trace(
        &model,
        cpu_state.clone(),
        steps,
        &material,
        phyz_contact::ContactSolverConfig::gpu_equivalent(),
    );
    let (approx_bd, _) = trace_gap(&cpu_trace_full, &cpu_trace_bd);

    // ── The chaos floor ──
    //
    // Roll the CPU against ITSELF from a state perturbed by 1 nm. Contact is
    // non-smooth, so a tumbling or sliding body is genuinely chaotic: ipse's
    // ollie work found a 1e-15 parameter change moving a score by hundreds of
    // points. Any engine-vs-engine gap at or below this number is not a
    // measurement of disagreement — it is the same trajectory, diverged by
    // arithmetic. Reporting parity without it invites chasing a "bug" that is
    // really a Lyapunov exponent.
    // Perturb whichever channel the scenario actually excites: position for a
    // straight drop, but SPIN for a tumble, where the outcome is which face
    // the box settles on. A control that jitters an insensitive coordinate
    // reports a reassuring zero and proves nothing.
    let mut jittered = cpu_state.clone();
    jittered.q.as_mut_slice()[3] += 1e-9;
    jittered.v.as_mut_slice()[0] *= 1.0 + 1e-9;
    jittered.v.as_mut_slice()[1] *= 1.0 + 1e-9;
    let cpu_trace_jit = cpu_trace(
        &model,
        jittered,
        steps,
        &material,
        phyz_contact::ContactSolverConfig::simulation(),
    );
    let chaos = settled_gap(&cpu_trace_full, &cpu_trace_jit);

    // ── CPU, per-body: within-body blocks kept, cross-chain blocks dropped.
    // This is the operator the GPU can actually afford. The gap between this
    // and `Full` is what the GPU pays for being unable to run an articulated
    // solve per contact row.
    let mut per_body = phyz_contact::ContactSolverConfig::gpu_equivalent();
    per_body.coupling = phyz_contact::ContactCoupling::PerBody;
    let cpu_trace_pb = cpu_trace(&model, cpu_state.clone(), steps, &material, per_body);
    let (approx_pb, _) = trace_gap(&cpu_trace_full, &cpu_trace_pb);

    let cpu_trace = cpu_trace_full;

    // ── GPU: penalty forces ──
    let Ok(mut gpu) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("{name}: skipping, no GPU adapter");
        return;
    };
    let gains = BodyContactGains::uniform_frequency(&gpu.model, OMEGA, 1.0);
    if std::env::var("PENALTY").is_ok() {
        gpu.enable_ground_contact_per_body(0.0, friction, &gains)
            .expect("gpu contact");
    } else {
        gpu.enable_contact_impulse(0.0, friction, &gains, &[], None)
            .expect("gpu contact");
        if let Ok(n) = std::env::var("SWEEPS") {
            gpu.contact_sweeps = n.parse().expect("SWEEPS");
        }
    }
    gpu.load_states(&[gpu_state]);

    let mut pos_gap = Gap::default();
    let mut vel_gap = Gap::default();
    let mut gpu_tr = Vec::with_capacity(steps);
    for (k, (cpu_p, cpu_v)) in cpu_trace.iter().enumerate() {
        gpu.step();
        let g = gpu.readback_states().remove(0);
        let (gp, gv) = (body_pos(&g), body_linvel(&g));
        pos_gap.accumulate(k, (gp - cpu_p).norm());
        vel_gap.accumulate(k, (gv - cpu_v).norm());
        gpu_tr.push((gp, gv));
    }
    let settled = settled_gap(&cpu_trace, &gpu_tr);
    // GPU against the CPU running the GPU's OWN restriction. If this is much
    // smaller than the gap to `Full`, what is left is the documented
    // approximation rather than an implementation defect — that separation is
    // the entire reason `gpu_equivalent()` exists.
    let vs_ref = settled_gap(&cpu_trace_pb, &gpu_tr);
    if std::env::var("TRACE")
        .map(|v| name.contains(&v))
        .unwrap_or(false)
    {
        for k in (0..steps).step_by(steps / 40) {
            let (cp, _) = cpu_trace[k];
            let (gp, _) = gpu_tr[k];
            let (jp, _) = cpu_trace_jit[k];
            println!(
                "  t={:.3}  gpu-cpu {:.3e}   jitter-cpu {:.3e}   cpu z {:+.4}",
                k as f64 * DT,
                (gp - cp).norm(),
                (jp - cp).norm(),
                cp.z
            );
        }
    }

    // Final resting height on each engine, which is the single number that
    // most directly shows the penalty-vs-impulse difference: a penalty
    // contact sinks to mg/k, an impulse solve does not sink at all.
    let cpu_final = cpu_trace.last().map(|(p, _)| p.z).unwrap_or(f64::NAN);
    let gpu_final = body_pos(&gpu.readback_states()[0]).z;

    println!(
        "{name:<22} chaos {:>10}  approx bd {:>10} pb {:>10}   gpu peak {:>10} settled {:>10} vs-ref {:>10}   z cpu {cpu_final:+.5} gpu {gpu_final:+.5}",
        chaos,
        approx_bd.report(),
        approx_pb.report(),
        pos_gap.report(),
        settled,
        vs_ref,
    );
    let _ = vel_gap.report();
}

fn main() {
    println!("CPU convex impulse contact  vs  GPU penalty contact");
    println!("dt = {DT}, contact tc = {TIME_CONST}, {} steps\n", 2000);

    let half = Vec3::new(0.1, 0.1, 0.1);

    // 1. Straight drop. Normal direction only — no friction, no rotation.
    //    The cleanest possible statement of the penalty-vs-impulse gap.
    compare(
        "drop, flat",
        free_box(Vec3::new(0.0, 0.0, 0.5), half, 1.0),
        |_| {},
        2000,
        0.8,
    );

    // 2. Drop with sideways velocity. Now friction decides where it stops,
    //    and the two models disagree about what friction is: the CPU has a
    //    real cone with stiction in its interior, the GPU has Coulomb
    //    regularized by slip speed, which creeps below SLIP_EPS.
    compare(
        "drop + slide (mu 0.8)",
        free_box(Vec3::new(0.0, 0.0, 0.5), half, 1.0),
        |s| {
            s.v.as_mut_slice()[3] = 2.0;
        },
        2000,
        0.8,
    );

    // 3. Same, but nearly frictionless — isolates how much of case 2's gap
    //    is the friction model as opposed to the normal model.
    compare(
        "drop + slide (mu 0.05)",
        free_box(Vec3::new(0.0, 0.0, 0.5), half, 1.0),
        |s| {
            s.v.as_mut_slice()[3] = 2.0;
        },
        2000,
        0.05,
    );

    // 4. Tumbling drop. The box lands on a corner, so the contact manifold
    //    and the torque arm both matter — this is the regime a foot is in
    //    during any real step, and where a single support point is least
    //    like a face manifold.
    compare(
        "drop + tumble",
        free_box(Vec3::new(0.0, 0.0, 0.5), half, 1.0),
        |s| {
            s.v.as_mut_slice()[0] = 3.0;
            s.v.as_mut_slice()[1] = 1.5;
        },
        2000,
        0.8,
    );

    // 5. A light body under a heavy load is the configuration that produced
    //    the wheel blowup: gains sized for carried mass, applied to a body
    //    whose own mass is much smaller. Here it is just a light box, which
    //    checks the damper caps hold on their own.
    compare(
        "light box (0.1 kg)",
        free_box(Vec3::new(0.0, 0.0, 0.5), half, 0.1),
        |s| {
            s.v.as_mut_slice()[3] = 2.0;
        },
        2000,
        0.8,
    );
}
