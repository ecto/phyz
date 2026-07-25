//! Measure batched-env throughput on the CPU reference backend.
//!
//! ```text
//! cargo run --release -p phyz-env --example ant_throughput
//! ```
//!
//! The number this prints is the *floor*, not the headline: it is one thread,
//! f64, with ground contact and joint limits active. It exists so the GPU
//! backend has something honest to beat.
//!
//! Do not compare it to numbers taken before contact existed — a model with no
//! geoms does far less work per step and reports a much larger figure for the
//! same physics quality.

use phyz_env::{BatchEnv, EnvConfig, GroundContact, RewardTerm, VecEnv};
use std::time::Instant;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let model = match phyz_mjcf::MjcfLoader::from_file(path) {
        Ok(l) => l.build_model(),
        Err(e) => {
            eprintln!("could not load {path}: {e}");
            return;
        }
    };

    println!(
        "ant: {} bodies, nq={}, nv={}, {} actuators, dt={}",
        model.nbodies(),
        model.nq,
        model.nv,
        model.actuators.len(),
        model.dt
    );

    let num_envs = 1024;
    let frame_skip = 5;
    let steps = 200;

    let mut cfg = EnvConfig::new(&model, num_envs);
    cfg.frame_skip = frame_skip;
    cfg.ground = Some(GroundContact::default());
    cfg.task.max_episode_steps = 1000;
    cfg.task.reset_qpos_noise = 0.1;
    cfg.task.reward = vec![
        RewardTerm::Alive { weight: 1.0 },
        RewardTerm::CtrlCost { weight: 0.005 },
    ];
    println!(
        "contact: {}  joint limits: {}",
        cfg.ground.is_some(),
        model.joints.iter().filter(|j| j.limits.is_some()).count()
    );

    let mut env = BatchEnv::new(model.clone(), cfg).unwrap();
    let nu = env.action_space().dim();
    println!(
        "obs_dim={} action_dim={} num_envs={num_envs}",
        env.observation_space().dim(),
        nu
    );

    env.reset(Some(0));
    let actions = vec![0.0f32; num_envs * nu];

    // Warm up so the timing excludes first-touch page faults.
    for _ in 0..10 {
        env.step(&actions);
    }

    let t0 = Instant::now();
    let mut total_reward = 0.0f64;
    for _ in 0..steps {
        let b = env.step(&actions);
        total_reward += b.rewards.iter().map(|r| *r as f64).sum::<f64>();
    }
    let secs = t0.elapsed().as_secs_f64();

    let control_steps = (steps * num_envs) as f64;
    let physics_steps = control_steps * frame_skip as f64;
    println!("\n{steps} control steps × {num_envs} envs in {secs:.3}s");
    println!("  {:.0} env-steps/s", control_steps / secs);
    println!("  {:.0} physics-steps/s", physics_steps / secs);
    println!("  mean reward/step {:.4}", total_reward / control_steps);
}
