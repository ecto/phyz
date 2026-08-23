//! Does the device's impulse solve HOLD what the CPU's holds?
//!
//! ```text
//! cargo run --release -p phyz-gpu --example manifold_rank_probe
//! ```
//!
//! The ipse pre-tip regime is a body resting on a face with the load near an
//! edge: the CPU keeps the stance, the device tips out of it. Two candidate
//! causes are checkable from phyz alone and are separated here:
//!
//! * **convergence** — the device runs a fixed `contact_sweeps` of matrix-free
//!   PGS with a per-body load-sharing preconditioner, so an under-converged
//!   manifold is softer than the CPU's Newton solve. Sweeping the count says
//!   whether the gap closes with iterations.
//! * **manifold width** — the device keeps the deepest `MAX_CONTACT_PTS` = 8
//!   candidates per body, the CPU `MAX_MANIFOLD_POINTS` = 4. On a coplanar
//!   box face those extra points change the `n_active` divisor, hence the
//!   step size of every point's update.
//!
//! The probe is a box dropped flat with its centre of mass offset toward one
//! edge (a heavy nose), which is the pre-tip stance in miniature: it either
//! settles or it rolls off the edge.
//!
//! # What it measured, including the part that did not work
//!
//! Convergence is NOT the gap: 4 -> 256 sweeps moves the resting pitch by
//! 0.01 deg and the height not at all.
//!
//! What it did find is a **standoff**: the device settles 0.98 mm HIGHER than
//! the CPU convex solve, and flat in the sweep count for the same reason —
//! not a convergence error. The cause is real. Contacts are detected within
//! `ContactMaterial::margin` (1 mm) and the impedance tapers to zero across
//! that band, so a separated contact inside it should carry no load. The CPU
//! gets that from `phyz_contact::convex::regularization_diag`: the normal row
//! carries `R = max((1-d)/d * A_nn, 1e-6)`, so as `d -> 0` the row goes
//! infinitely soft. The GPU uses the impedance in the BIAS only, leaving its
//! normal row rigid at every detected point — so it kills the approach
//! velocity at the outer edge of the margin and stops there.
//!
//! **Adding `R` to the GPU normal row fixes this fixture and makes the real
//! task worse, so it is not in the tree.** The mechanism is now known, and it
//! is arithmetic rather than "PGS cannot carry a soft row": `a_nn` in the
//! kernels is the ISOLATED-body diagonal, deliberately an OVER-estimate of the
//! true `A_nn`. That is safe while it is only a preconditioner — it
//! under-relaxes — but `R` is not a preconditioner, it enters the fixed point:
//!
//! ```text
//!     f / f_rigid = A_true / (A_true + (1-d)/d * a_nn)
//! ```
//!
//! With `a_nn >> A_true` for a foot backed by a whole robot, that row is far
//! softer than the CPU's, which builds `R` from the true `A_nn` out of an
//! assembled Delassus operator. The device cannot reproduce the CPU's `R`
//! without the true `A_nn`.
//!
//! A bounded variant WAS measured and is also not in the tree. Confining the
//! softening to points that are SEPARATED and merely inside the margin — where
//! the CPU carries essentially no load and the device carries a full rigid
//! stop — and applying it as a multiply (`fn_new = d * rigid_update`) rather
//! than a division by `d`, takes this fixture's standoff to **0.03 mm**, flat
//! from 4 sweeps to 16. It is better than the `R` form on both counts. It
//! still fails `unified_contact_matches_wgsl_kernels` on the impulse
//! heightfield case (cuda-c 1.9e-6 vs wgsl 7.2e-4 on `v[0]`), and that
//! disagreement was not run to ground, so it is not shippable. Anyone
//! resuming: the branch on `penetration < 0.0` is a discontinuity at
//! `penetration == 0`, and the two backends do not have to land on the same
//! side of it.
//!
//! **The standoff is not the ipse pre-tip gap.** Measured on that ruler
//! (`gpu_policy_parity models/skate-bc-init-pretip-s7.policy 8 7`,
//! `PARITY_MODE=pretip`, wgpu, deterministic actor, against a CPU f64 holding
//! all 8 episodes for the full 625-step horizon at return 546.47):
//!
//! ```text
//!   device, main                          229.21 +- 95.03   len 321.5   5/8 falls   ratio 0.51
//!   manifold cap 4 (SEL_PTS)              216.88 +- 97.65   len 310.5   5/8 falls   ratio 0.50
//!   separated-side softening              217.12 +- 96.43   len 310.5   5/8 falls   ratio 0.50
//!   both                                  214.71 +- 97.38   len 306.5   5/8 falls   ratio 0.49
//! ```
//!
//! Flat. Together with the sweep count (4 -> 256), the contact frequency
//! (60..250 rad/s), the carried mass (0.25x..4x) and the timestep
//! (1.0/0.5/0.25 ms), every knob inside the contact SOLVE is now a measured
//! null on that stance. What is left is the contact SET: the device meets the
//! deck top as two flat `BodyPlane` rectangles where the CPU collides the
//! foot's boxes against the deck's 27 and the kicktail's 18. With `R` on both impulse sites the
//! standoff falls 0.98 mm -> 0.09 mm here, and the ipse pre-tip parity run
//! (`gpu_policy_parity`, 8 episodes, seed 7, wgpu, deterministic actor)
//! regresses from `len ratio 0.51, 5/8 falls` to `0.09, 8/8 falls` against a
//! CPU that holds all 8 for the full horizon. A softer normal row inside 16
//! PGS sweeps with a per-body preconditioner is not the same object as a
//! softer row inside the CPU's fully coupled Newton solve, and the K1 stance
//! is the difference. Whoever picks this up: the standoff is real and worth
//! removing, but it has to come with the convergence to carry it — more
//! sweeps, tighter coupling, or the regularizer folded into the
//! preconditioner rather than added to a diagonal that is already an
//! under-estimate.
use phyz::Simulator;
use phyz_contact::material::ContactMaterial;
use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::BodyContactGains;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Joint, Model, ModelBuilder, State};

fn half() -> Vec3 { Vec3::new(0.20, 0.10, 0.02) }
const MASS: f64 = 5.0;
/// Centre of mass toward +x, i.e. over the edge the box wants to roll across.
const COM_X: f64 = 0.16;

fn model() -> Model {
    let (lx, ly, lz) = (2.0 * half().x, 2.0 * half().y, 2.0 * half().z);
    let inertia = SpatialInertia::new(
        MASS,
        Vec3::new(COM_X, 0.0, 0.0),
        Mat3::from_diagonal(&Vec3::new(
            MASS / 12.0 * (ly * ly + lz * lz),
            MASS / 12.0 * (lx * lx + lz * lz),
            MASS / 12.0 * (lx * lx + ly * ly),
        )),
    );
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.0005)
        .add_body(
            "plank",
            -1,
            Joint::free(SpatialTransform::identity()),
            inertia,
        )
        .build();
    m.bodies[0].geometry = Some(Geometry::Box { half_extents: half() });
    m
}

fn start(m: &Model) -> State {
    let mut s = m.default_state();
    s.q[5] = half().z + 0.002;
    s
}

/// Pitch (rotation about y) and height after `steps`.
fn readout(q: &[f64]) -> (f64, f64) {
    (q[1].to_degrees(), q[5])
}

fn main() {
    let m = model();
    let steps = 3000; // 1.5 s at dt 0.5 ms
    let s0 = start(&m);

    // CPU: the convex impulse solve, the referee.
    let cpu = Simulator::new();
    let mat = ContactMaterial {
        friction: 0.8,
        ..ContactMaterial::default()
    };
    let mut cpu_state = s0.clone();
    for _ in 0..steps {
        cpu.step_with_contacts(&m, &mut cpu_state, 0.0, &mat);
    }
    let (cpu_pitch, cpu_z) = readout(cpu_state.q.as_slice());
    println!("model: plank {:?} m, {MASS} kg, com x {COM_X} m, dt 0.5 ms, {steps} steps", half());
    println!("  cpu convex        pitch {cpu_pitch:>8.3} deg   z {cpu_z:>8.5} m");

    let gains = BodyContactGains::uniform_frequency(&m, 100.0, 1.0);
    for sweeps in [4usize, 16, 64, 256] {
        let mut gpu = GpuBatchSimulator::new(m.clone(), 1).unwrap();
        gpu.enable_contact_impulse(0.0, 0.8, &gains, &[], None).unwrap();
        gpu.contact_sweeps = sweeps;
        gpu.load_states(std::slice::from_ref(&s0));
        for _ in 0..steps {
            gpu.step();
        }
        let (p, z) = readout(gpu.readback_states()[0].q.as_slice());
        println!(
            "  gpu impulse s={sweeps:<4} pitch {p:>8.3} deg   z {z:>8.5} m   (d pitch {:>7.3}, d z {:>8.5})",
            p - cpu_pitch,
            z - cpu_z
        );
    }
    println!("  device manifold cap {} pts", phyz_gpu::layout::MAX_CONTACT_PTS);
}
