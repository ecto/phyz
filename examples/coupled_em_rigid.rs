//! `phyz-coupling`: a charged rigid body driven by an electromagnetic field.
//!
//! This is the multi-physics story end to end. Two solvers run at very
//! different timescales and exchange forces through a shared handshake
//! region:
//!
//! * **Electromagnetic** — a static crossed E/B field standing in for an
//!   FDTD domain, stepped at the *fast* timescale.
//! * **Rigid body** — a charged bead on a prismatic joint, integrated by
//!   `phyz-rigid`'s ABA at the *slow* timescale.
//!
//! The pieces `phyz-coupling` contributes:
//!
//! * [`BoundingBox`] — the spatial overlap where the two solvers talk.
//! * [`SubcyclingSchedule`] — r-RESPA style ratios so the EM solver takes
//!   many substeps per rigid-body step, instead of dragging the whole
//!   simulation down to the EM timestep.
//! * [`lorentz_force`] / [`magnetic_torque`] — the actual force transfer.
//! * [`Coupling`] / [`ForceTransfer`] — the declarative description of the
//!   handshake, including the velocity-damping channel back from the field.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p phyz-examples --example coupled_em_rigid
//! ```

use phyz_coupling::{
    BoundingBox, Coupling, ForceTransfer, SolverType, SubcyclingSchedule, TimeScale, lorentz_force,
    magnetic_torque,
};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};
use phyz_rigid::aba_with_external_forces;

/// Charge on the bead (C) — large, so the motion is visible in a few steps.
const CHARGE: f64 = 1.0e-3;
/// Bead mass (kg).
const MASS: f64 = 0.05;
/// Magnetic dipole moment of the bead (A·m²).
fn dipole() -> Vec3 {
    Vec3::new(0.0, 0.0, 2.0e-3)
}

/// Rigid-body timestep (the *slow* scale).
const DT_RIGID: f64 = 1.0e-3;
const RIGID_STEPS: usize = 200;

/// Static electric field (V/m).
fn e_field(_p: &Vec3) -> Vec3 {
    Vec3::new(20.0, 0.0, 0.0)
}

/// Static magnetic field (T).
fn b_field(_p: &Vec3) -> Vec3 {
    Vec3::new(0.0, 0.0, 0.5)
}

/// A bead free to slide along X, carrying charge and a magnetic dipole.
fn charged_bead() -> Model {
    let inertia = SpatialInertia::new(
        MASS,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(1.0e-5, 1.0e-5, 1.0e-5)),
    );
    ModelBuilder::new()
        // Gravity off: we want to see the Lorentz force alone.
        .gravity(Vec3::zeros())
        .dt(DT_RIGID)
        .add_prismatic_body(
            "bead",
            -1,
            SpatialTransform::identity(),
            Vec3::new(1.0, 0.0, 0.0),
            inertia,
        )
        .build()
}

fn main() {
    println!("=== phyz-coupling: EM field ↔ rigid body ===\n");

    // -----------------------------------------------------------------
    // 1. Declare the handshake region and the transfer law.
    // -----------------------------------------------------------------
    let overlap = BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

    let coupling = Coupling {
        solver_a: SolverType::Electromagnetic,
        solver_b: SolverType::RigidBody,
        overlap_region: overlap.clone(),
        // Radiation reaction, crudely: the field drains momentum from the
        // bead in proportion to their relative velocity.
        force_transfer: ForceTransfer::Direct { damping: 2.0e-3 },
    };

    println!(
        "handshake  : {:?} ↔ {:?} over a {:.1} m³ box centred at {:?}",
        coupling.solver_a,
        coupling.solver_b,
        overlap.volume(),
        overlap.center(),
    );

    // -----------------------------------------------------------------
    // 2. Multi-timescale schedule: EM is fast, rigid body is slow.
    // -----------------------------------------------------------------
    let schedule = SubcyclingSchedule::from_timescales(
        DT_RIGID / TimeScale::Slow.typical_dt_ratio() as f64,
        &[TimeScale::Fast, TimeScale::Slow],
    );
    let em_substeps = schedule.num_substeps(0);
    let rigid_substeps = schedule.num_substeps(1);

    println!(
        "subcycling : EM dt={:.3e}s ({} substeps), rigid dt={:.3e}s ({} substeps)",
        schedule.dt_for_level(0),
        em_substeps,
        schedule.dt_for_level(1),
        rigid_substeps,
    );
    println!(
        "             → the rigid solver takes steps {}x larger than the EM \
         solver, instead of both running at the EM timestep\n",
        (schedule.dt_for_level(1) / schedule.dt_for_level(0)).round() as usize
    );

    // -----------------------------------------------------------------
    // 3. Run the coupled loop.
    // -----------------------------------------------------------------
    let model = charged_bead();
    let mut state = model.default_state();

    println!("  step      x (m)      v (m/s)    F_lorentz (N)   F_couple (N)");
    println!("  ────────────────────────────────────────────────────────────");

    let mut em_steps = 0usize;
    for step in 0..RIGID_STEPS {
        let pos = Vec3::new(state.q[0], 0.0, 0.0);
        let vel = Vec3::new(state.v[0], 0.0, 0.0);

        // -- fast solver: advance the EM field. A real FDTD solve would run
        //    here; the fields are static, so we only count the substeps the
        //    schedule asks for.
        for sub in 0..em_substeps {
            if schedule.should_step(0, step * em_substeps + sub) {
                em_steps += 1;
            }
        }

        // -- force transfer: EM → rigid.
        let e = e_field(&pos);
        let b = b_field(&pos);
        let mut force = if overlap.contains(&pos) {
            lorentz_force(CHARGE, pos, vel, &e, &b)
        } else {
            // Outside the handshake region the solvers are decoupled.
            Vec3::zeros()
        };

        // -- back-reaction through the declared transfer law. The "field
        //    side" of the pair is at rest at the origin.
        let coupling_force =
            coupling
                .force_transfer
                .compute_force(&pos, &vel, &Vec3::zeros(), &Vec3::zeros());
        force += coupling_force;

        let torque = magnetic_torque(dipole(), &b);

        // -- slow solver: one rigid-body step under the transferred wrench.
        //    The prismatic joint runs along X, so only that component does
        //    work; the torque is reported for completeness.
        let mut wrench = vec![phyz_math::SpatialVec::zero(); model.nbodies()];
        wrench[0] = phyz_math::SpatialVec::new(torque, force);

        let qdd = aba_with_external_forces(&model, &state, Some(&wrench));
        for k in 0..model.nv {
            state.v[k] += DT_RIGID * qdd[k];
        }
        for k in 0..model.nq {
            state.q[k] += DT_RIGID * state.v[k];
        }

        if step % 40 == 0 {
            println!(
                "  {step:>4}  {:>10.6}  {:>10.6}   {:>12.3e}   {:>12.3e}",
                state.q[0],
                state.v[0],
                force.x - coupling_force.x,
                coupling_force.x,
            );
        }
    }

    println!("\nEM substeps run  : {em_steps}");
    println!("rigid steps run  : {RIGID_STEPS}");
    println!("final position   : {:.6} m", state.q[0]);
    println!("final velocity   : {:.6} m/s", state.v[0]);

    // The bead is pushed along +X by qE and slowed by the coupling damping,
    // so it must end up displaced but not runaway.
    assert!(state.q[0] > 0.0, "qE should push the bead along +X");
    assert!(state.q[0].is_finite(), "coupled loop should stay stable");
}
