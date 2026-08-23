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
