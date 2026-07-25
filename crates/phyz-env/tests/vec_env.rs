//! Behavioural contract for a phyz vector env.
//!
//! These tests are the executable half of `docs/design/batched-envs.md`: the
//! GPU backend, when it lands, must pass this same file.

use phyz_env::{AutoresetMode, BatchEnv, EnvConfig, RewardTerm, TaskSpec, VecEnv};
use phyz_math::{SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};

fn pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.01)
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
        )
        .build()
}

fn config(model: &Model, n: usize) -> EnvConfig {
    let mut c = EnvConfig::new(model, n);
    c.task.reset_qpos_noise = 0.3;
    c.task.reset_qvel_noise = 0.1;
    c.task.max_episode_steps = 8;
    c.seed = 1234;
    c
}

fn rollout(n: usize, seed: u64, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let model = pendulum();
    let mut env = BatchEnv::new(model.clone(), config(&model, n)).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(seed));

    let mut obs_trace = Vec::new();
    let mut rew_trace = Vec::new();
    for t in 0..steps {
        // Deterministic pseudo-policy so the trace depends only on the sim.
        let actions: Vec<f32> = (0..n * nu)
            .map(|i| ((i + t) as f32 * 0.37).sin() * 0.5)
            .collect();
        let b = env.step(&actions);
        obs_trace.extend_from_slice(&b.obs);
        rew_trace.extend_from_slice(&b.rewards);
    }
    (obs_trace, rew_trace)
}

#[test]
fn same_seed_gives_bit_identical_rollouts() {
    let (a_obs, a_rew) = rollout(8, 99, 40);
    let (b_obs, b_rew) = rollout(8, 99, 40);
    assert_eq!(a_obs, b_obs, "observations diverged for identical seeds");
    assert_eq!(a_rew, b_rew, "rewards diverged for identical seeds");
}

#[test]
fn different_seed_gives_different_rollout() {
    let (a, _) = rollout(8, 1, 20);
    let (b, _) = rollout(8, 2, 20);
    assert_ne!(a, b);
}

/// Environment `k` must behave identically regardless of how many siblings it
/// has. This is what makes a result reproducible when someone re-runs it with a
/// different `num_envs` — the property Rapier and Brax both fail to promise.
#[test]
fn env_k_is_independent_of_batch_size() {
    let steps = 20;
    let (big, _) = rollout(8, 7, steps);
    let (small, _) = rollout(4, 7, steps);

    let model = pendulum();
    let obs_dim = BatchEnv::new(model.clone(), config(&model, 1))
        .unwrap()
        .observation_space()
        .dim();

    for t in 0..steps {
        for k in 0..4 {
            let b = &big[(t * 8 + k) * obs_dim..(t * 8 + k + 1) * obs_dim];
            let s = &small[(t * 4 + k) * obs_dim..(t * 4 + k + 1) * obs_dim];
            assert_eq!(b, s, "env {k} diverged at step {t} across batch sizes");
        }
    }
}

#[test]
fn truncation_fires_at_the_time_limit_and_is_not_termination() {
    let model = pendulum();
    let mut cfg = config(&model, 2);
    cfg.task.max_episode_steps = 5;
    cfg.task.termination.healthy_z = None;
    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));

    let zeros = vec![0.0f32; 2 * nu];
    for t in 1..=5 {
        let b = env.step(&zeros);
        if t < 5 {
            assert!(!b.truncated[0] && !b.terminated[0], "early done at {t}");
        } else {
            assert!(b.truncated[0], "expected truncation at step 5");
            assert!(!b.terminated[0], "truncation must not set terminated");
            assert!(b.episode_done[0]);
            assert_eq!(b.episode_length[0], 5);
        }
    }
}

/// Gymnasium >= 1.0 `NextStep` autoreset: the step following a done reports
/// reward 0, both flags false, and the *reset* observation.
#[test]
fn next_step_autoreset_matches_gymnasium() {
    let model = pendulum();
    let mut cfg = config(&model, 1);
    cfg.task.max_episode_steps = 3;
    cfg.autoreset = AutoresetMode::NextStep;
    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));
    let zeros = vec![0.0f32; nu];

    env.step(&zeros);
    env.step(&zeros);
    let b = env.step(&zeros);
    assert!(b.truncated[0], "episode should end on step 3");

    let b = env.step(&zeros);
    assert!(!b.truncated[0] && !b.terminated[0]);
    assert_eq!(b.rewards[0], 0.0, "the reset step carries no reward");
    assert!(!b.final_valid[0], "final_obs is a SameStep-only channel");
}

/// `SameStep` reports the terminal observation out-of-band and the reset
/// observation in `obs`, so a single step yields both.
#[test]
fn same_step_autoreset_reports_final_obs() {
    let model = pendulum();
    let mut cfg = config(&model, 1);
    cfg.task.max_episode_steps = 3;
    cfg.autoreset = AutoresetMode::SameStep;
    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));
    let zeros = vec![0.0f32; nu];

    env.step(&zeros);
    env.step(&zeros);
    let b = env.step(&zeros);
    assert!(b.truncated[0]);
    assert!(b.final_valid[0], "terminal obs must be preserved");
    assert_ne!(
        &b.final_obs[..b.obs.len()],
        &b.obs[..],
        "reset obs should differ from terminal obs"
    );
}

#[test]
fn actions_outside_the_space_are_clamped_not_rejected() {
    let model = pendulum();
    let mut env = BatchEnv::new(model.clone(), config(&model, 2)).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));
    let wild = vec![f32::MAX; 2 * nu];
    let b = env.step(&wild);
    assert!(b.obs.iter().all(|x| x.is_finite()));
}

#[test]
fn reward_terms_compose() {
    let model = pendulum();
    let mut cfg = config(&model, 1);
    cfg.task = TaskSpec {
        reward: vec![
            RewardTerm::Alive { weight: 1.0 },
            RewardTerm::CtrlCost { weight: 0.5 },
        ],
        max_episode_steps: 100,
        ..TaskSpec::default()
    };
    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();
    let nu = env.action_space().dim();
    env.reset(Some(0));
    let a = vec![1.0f32; nu];
    let b = env.step(&a);
    // 1.0 alive - 0.5 * (nu * 1.0)
    let expect = 1.0 - 0.5 * nu as f32;
    assert!((b.rewards[0] - expect).abs() < 1e-6, "got {}", b.rewards[0]);
}

#[test]
fn envs_with_different_initial_states_diverge() {
    let model = pendulum();
    let mut env = BatchEnv::new(model.clone(), config(&model, 4)).unwrap();
    let nu = env.action_space().dim();
    let dim = env.observation_space().dim();
    let b = env.reset(Some(5));
    let first = b.obs[0];
    assert!(
        (1..4).any(|i| (b.obs[i * dim] - first).abs() > 1e-9),
        "reset noise should decorrelate environments"
    );
    let _ = env.step(&vec![0.0; 4 * nu]);
}

/// The flagship model must at least load, batch, and step without NaN.
/// It is *not* yet claimed to be physically faithful — see the design doc's
/// benchmark-gap table.
#[test]
fn ant_mjcf_loads_and_steps() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let Ok(loader) = phyz_mjcf::MjcfLoader::from_file(path) else {
        eprintln!("skipping: {path} not found");
        return;
    };
    let model = loader.build_model();
    assert!(model.nbodies() >= 9, "ant should have torso + 8 leg bodies");

    let mut cfg = EnvConfig::new(&model, 16);
    cfg.task.max_episode_steps = 20;
    cfg.ground = Some(phyz_env::GroundContact::default());
    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();

    let nu = env.action_space().dim();
    env.reset(Some(0));
    for _ in 0..20 {
        let b = env.step(&vec![0.0; 16 * nu]);
        assert!(
            b.obs.iter().all(|x| x.is_finite()),
            "ant produced non-finite observations"
        );
    }
}
