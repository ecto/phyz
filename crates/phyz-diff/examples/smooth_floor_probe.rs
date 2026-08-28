//! What is the contact-free accuracy floor of the convex adjoint, in the
//! current mode?
//!
//! Free fall has a closed-form gradient (`dq_z(T)/dv_z0 = steps·dt` for a free
//! joint at identity orientation), so the printed relative error is pure
//! instrument noise. Run twice:
//!
//! ```text
//! cargo run --release -p phyz-diff --example smooth_floor_probe
//! PHYZ_SMOOTH_ADJOINT=1 cargo run --release -p phyz-diff --example smooth_floor_probe
//! ```
//!
//! The lane machinery's floor is its central-difference step (~1e-7..1e-8
//! relative — the same noise `phyz-tang`'s chained-ballistic test documents);
//! the reverse-mode path's is rounding.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

fn main() {
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1e-3, 1e-3, 1e-3)),
            ),
        )
        .build();
    let steps = 32;
    let mut q0 = DVec::zeros(6);
    q0[5] = 10.0; // far above any ground
    let ctrl = |_t: usize| DVec::zeros(6);
    let obj = FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[5],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[5] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    };
    let rollout = ConvexContactRollout {
        model: &model,
        ground_height: 0.0,
        material: ContactMaterial::default(),
        config: ContactSolverConfig::gradients(),
        q0,
        v0: DVec::zeros(6),
        steps,
        ctrl: &ctrl,
    };
    let g = convex_adjoint_gradient(&rollout, &obj).expect("free flight cannot refuse");
    let want = steps as f64 * model.dt;
    let err = (g.d_v0[5] - want).abs() / want;
    println!(
        "mode: {}",
        if std::env::var("PHYZ_SMOOTH_ADJOINT").is_ok_and(|v| v == "1" || v == "true") {
            "reverse-mode smooth adjoint"
        } else {
            "finite-difference lanes"
        }
    );
    println!(
        "dq_z/dv_z0 over {steps} contact-free steps: {:.17e}",
        g.d_v0[5]
    );
    println!("exact:                                      {:.17e}", want);
    println!("relative error:                             {err:.3e}");
}
