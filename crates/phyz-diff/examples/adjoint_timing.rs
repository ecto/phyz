//! Timing harness for the trajectory adjoint: forward rollout vs one adjoint
//! backward pass vs central finite differences over every inertia parameter.
//!
//! Mirrors the `gradient` suite of `phyz-bench` for callers who only have this
//! crate. Run with:
//!
//! ```sh
//! cargo run --release -p phyz-diff --example adjoint_timing
//! ```
//!
//! The `max_rel_err` column cross-checks the adjoint against the same central
//! differences it is timed against, so a fast-but-wrong run cannot pass
//! unnoticed.

use phyz_diff::rollout::{
    AdjointRollout, FinalStateObjective, adjoint_rollout_gradient, rollout_objective,
};
use phyz_math::{DVec, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, JointType, Model, ModelBuilder};
use std::time::Instant;

fn si_from(p: [f64; 10]) -> SpatialInertia {
    SpatialInertia::new(
        p[0],
        Vec3::new(p[1], p[2], p[3]),
        Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
    )
}

fn q0_objective<'a>() -> FinalStateObjective<'a> {
    FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[0],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[0] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    }
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn bench(
    name: &str,
    build: &dyn Fn(&[[f64; 10]]) -> Model,
    p0: Vec<[f64; 10]>,
    q0: Vec<f64>,
    v0: Vec<f64>,
    ctrl: &dyn Fn(usize) -> DVec,
    steps: usize,
) {
    let obj = q0_objective();
    let model = build(&p0);
    fn mk<'a>(
        m: &'a Model,
        q0: Vec<f64>,
        v0: Vec<f64>,
        steps: usize,
        ctrl: &'a dyn Fn(usize) -> DVec,
    ) -> AdjointRollout<'a> {
        AdjointRollout {
            model: m,
            contact: None,
            q0,
            v0,
            steps,
            ctrl,
        }
    }

    let fwd = time(200, || {
        std::hint::black_box(rollout_objective(
            &mk(&model, q0.clone(), v0.clone(), steps, ctrl),
            &obj,
        ));
    });
    let adj = time(50, || {
        std::hint::black_box(adjoint_rollout_gradient(
            &mk(&model, q0.clone(), v0.clone(), steps, ctrl),
            &obj,
        ));
    });

    let nb = p0.len();
    let fd = time(5, || {
        for b in 0..nb {
            for k in 0..10 {
                for s in [1.0, -1.0] {
                    let mut pp = p0.clone();
                    pp[b][k] += s * 1e-6;
                    let m = build(&pp);
                    std::hint::black_box(rollout_objective(
                        &mk(&m, q0.clone(), v0.clone(), steps, ctrl),
                        &obj,
                    ));
                }
            }
        }
    });

    // Correctness cross-check against central FD.
    let g = adjoint_rollout_gradient(&mk(&model, q0.clone(), v0.clone(), steps, ctrl), &obj);
    let mut max_rel: f64 = 0.0;
    for b in 0..nb {
        for k in 0..10 {
            let h = if k == 0 {
                1e-5
            } else if k <= 3 {
                1e-6
            } else {
                1e-7
            };
            let (mut pp, mut pm) = (p0.clone(), p0.clone());
            pp[b][k] += h;
            pm[b][k] -= h;
            let d =
                (rollout_objective(&mk(&build(&pp), q0.clone(), v0.clone(), steps, ctrl), &obj)
                    - rollout_objective(
                        &mk(&build(&pm), q0.clone(), v0.clone(), steps, ctrl),
                        &obj,
                    ))
                    / (2.0 * h);
            if d.abs() > 1e-8 {
                max_rel = max_rel.max((g.d_inertia[b][k] - d).abs() / d.abs());
            }
        }
    }

    println!(
        "{name:20} params={:3}  fwd={:8.1}us  adjoint={:9.1}us ({:5.2}x)  centralFD={:9.1}us ({:5.2}x)  -> adjoint {:5.2}x faster than FD   max_rel_err={max_rel:.2e}",
        nb * 10,
        fwd * 1e6,
        adj * 1e6,
        adj / fwd,
        fd * 1e6,
        fd / fwd,
        fd / adj,
    );
}

fn main() {
    const DT: f64 = 1e-3;

    let pend = |p: &[[f64; 10]]| -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(-9.81, 0.0, 0.0))
            .dt(DT)
            .add_revolute_body("bob", -1, SpatialTransform::identity(), si_from(p[0]))
            .build()
    };
    bench(
        "pendulum",
        &pend,
        vec![[1.3, 0.4, 0.1, 0.05, 0.02, 0.03, 0.04, 0.005, 0.002, 0.001]],
        vec![0.35],
        vec![-0.2],
        &|_t| DVec::from_slice(&[0.01]),
        300,
    );

    let chain = |p: &[[f64; 10]]| -> Model {
        let joint2 = Joint {
            joint_type: JointType::Revolute,
            parent_to_joint: SpatialTransform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
            axis: Vec3::new(1.0, 0.0, 0.0),
            damping: 0.02,
            limits: None,
            ..Default::default()
        };
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .dt(DT)
            .add_revolute_body("link1", -1, SpatialTransform::identity(), si_from(p[0]))
            .add_body("link2", 0, joint2, si_from(p[1]))
            .build()
    };
    bench(
        "double pendulum",
        &chain,
        vec![
            [
                1.1, 0.25, 0.05, -0.03, 0.02, 0.025, 0.03, 0.004, 0.001, 0.002,
            ],
            [
                0.7, 0.15, -0.04, 0.06, 0.01, 0.012, 0.014, 0.002, 0.001, 0.0015,
            ],
        ],
        vec![0.2, -0.3],
        vec![0.1, 0.25],
        &|_t| DVec::from_slice(&[0.02, -0.01]),
        250,
    );
}
