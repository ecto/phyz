//! The four benchmark models, end to end through the vector-env API.
//!
//! This is the credibility gate: load MJCF → batch → reset → step → finite,
//! bounded observations and sane rewards, with determinism preserved.

use phyz_env::{Benchmark, VecEnv, make};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

const ALL: [Benchmark; 4] = [
    Benchmark::Ant,
    Benchmark::HalfCheetah,
    Benchmark::Humanoid,
    Benchmark::ShadowHand,
];

#[test]
fn every_benchmark_loads_and_steps() {
    for b in ALL {
        let mut env = make(b, MODELS, 8).unwrap_or_else(|e| panic!("{b:?}: {e}"));
        let nu = env.action_space().dim();
        let obs_dim = env.observation_space().dim();

        let batch = env.reset(Some(7));
        assert!(
            batch.obs.iter().all(|x| x.is_finite()),
            "{b:?}: reset produced non-finite observations"
        );
        assert_eq!(batch.obs.len(), 8 * obs_dim);

        // A deterministic non-trivial action pattern, so actuator paths are
        // actually exercised rather than left at zero.
        let actions: Vec<f32> = (0..8 * nu)
            .map(|i| ((i as f32) * 0.31).sin() * 0.6)
            .collect();

        for t in 0..50 {
            let batch = env.step(&actions);
            assert!(
                batch.obs.iter().all(|x| x.is_finite()),
                "{b:?}: non-finite observation at step {t}"
            );
            assert!(
                batch.obs.iter().all(|x| x.abs() <= 10.0 + 1e-6),
                "{b:?}: observations escaped the clip at step {t}"
            );
            assert!(
                batch.rewards.iter().all(|r| r.is_finite()),
                "{b:?}: non-finite reward at step {t}"
            );
        }
    }
}

#[test]
fn benchmark_action_spaces_match_the_models() {
    let cases = [
        (Benchmark::Ant, 8),
        (Benchmark::HalfCheetah, 6),
        (Benchmark::Humanoid, 17),
        (Benchmark::ShadowHand, 20),
    ];
    for (b, nu) in cases {
        let env = make(b, MODELS, 1).unwrap();
        assert_eq!(env.action_space().dim(), nu, "{b:?} action dim");
    }
}

/// Determinism must survive contact, servos and floating bases — not just the
/// pendulum the unit tests use.
#[test]
fn benchmarks_are_bit_reproducible() {
    for b in ALL {
        let trace = |seed: u64| {
            let mut env = make(b, MODELS, 4).unwrap();
            let nu = env.action_space().dim();
            env.reset(Some(seed));
            let mut out = Vec::new();
            for t in 0..30 {
                let a: Vec<f32> = (0..4 * nu)
                    .map(|i| (((i + t) as f32) * 0.19).cos() * 0.4)
                    .collect();
                let batch = env.step(&a);
                out.extend_from_slice(&batch.obs);
                out.extend_from_slice(&batch.rewards);
            }
            out
        };
        assert_eq!(trace(3), trace(3), "{b:?} is not reproducible");
        assert_ne!(trace(3), trace(4), "{b:?} ignores its seed");
    }
}

/// Locomotion benchmarks must terminate when the torso leaves the healthy
/// height band. If nothing ever terminates, the alive bonus is free and the
/// task is broken.
#[test]
fn locomotion_terminates_when_it_falls() {
    let mut env = make(Benchmark::Humanoid, MODELS, 4).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));

    let zeros = vec![0.0f32; 4 * nu];
    let mut saw_termination = false;
    for _ in 0..400 {
        let batch = env.step(&zeros);
        if batch.terminated.iter().any(|t| *t) {
            saw_termination = true;
            break;
        }
    }
    assert!(
        saw_termination,
        "an unactuated humanoid must fall out of the healthy band"
    );
}

/// Half-cheetah has no task-level termination condition — only truncation.
///
/// It *does* still terminate via the divergence guard, because the prototype
/// contact model cannot yet keep a long tumbling capsule stable; see
/// `half_cheetah_contact_is_known_to_be_unstable`. This test pins the task
/// definition, not the physics.
#[test]
fn half_cheetah_has_no_task_termination_condition() {
    let env = make(Benchmark::HalfCheetah, MODELS, 1).unwrap();
    let task = Benchmark::HalfCheetah.task(env.model());
    assert!(task.termination.healthy_z.is_none());
    assert!(
        task.termination.max_velocity.is_some(),
        "the divergence guard must stay on even when the task never terminates"
    );
}

/// **Known limitation, deliberately asserted so it cannot be forgotten.**
///
/// Half-cheetah's torso is a 1 m capsule. When it lands on one end, the contact
/// torque about the other end is large enough that the penalty contact model
/// overshoots and the environment diverges; the divergence guard then
/// terminates and resets it. The other three benchmarks are stable.
///
/// This test asserts the *current* behaviour. When the contact model is
/// replaced with a real solver (design doc, B5) this test should start failing
/// — at which point delete it and re-enable a stability assertion.
#[test]
fn half_cheetah_contact_is_known_to_be_unstable() {
    let n = 32;
    let mut env = make(Benchmark::HalfCheetah, MODELS, n).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(1));

    let mut terminations = 0usize;
    for t in 0..300 {
        let a: Vec<f32> = (0..n * nu)
            .map(|i| (((i + t) as f32) * 0.23).sin() * 0.5)
            .collect();
        let batch = env.step(&a);
        terminations += batch.terminated.iter().filter(|t| **t).count();
    }

    assert!(
        terminations > 0,
        "half-cheetah is expected to trip the divergence guard today; if it no \
         longer does, the contact model was fixed — delete this test"
    );

    // Whatever the physics does, the batch must stay usable: the guard exists
    // so one diverged environment never poisons the others.
    let batch = env.step(&vec![0.0; n * nu]);
    assert!(
        batch.obs.iter().all(|x| x.is_finite()),
        "the divergence guard must keep observations finite"
    );
}

/// The hand's position servos must actually track their command: driving a
/// finger to a target should move it further than commanding zero.
#[test]
fn position_servos_track_their_command() {
    let mut env = make(Benchmark::ShadowHand, MODELS, 2).unwrap();
    let nu = env.action_space().dim();

    let final_q = |cmd: f32| {
        let mut env2 = make(Benchmark::ShadowHand, MODELS, 2).unwrap();
        let mut c = vec![0.0f32; 2 * nu];
        for v in c.iter_mut() {
            *v = cmd;
        }
        env2.reset(Some(0));
        for _ in 0..100 {
            env2.step(&c);
        }
        env2.states().next().unwrap().q.as_slice().to_vec()
    };

    env.reset(Some(0));
    let held = final_q(0.0);
    let driven = final_q(0.8);

    let moved = held
        .iter()
        .zip(&driven)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        moved > 1e-3,
        "position servos did not move the hand (max delta {moved:.2e})"
    );
}
