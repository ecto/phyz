//! The existence proof: trajectory optimisation as an ordinary tang training
//! loop.
//!
//! A box on a plane, 12 free control scalars (two 6-DOF pushes, at the start
//! of the window and a third of the way in), a target position, and a plain
//! `ModuleAdam` closing the loop on gradients that came out of the contact
//! solver. Nothing here knows it is doing physics: the controls are a
//! `tang_train::Parameter`, the loss is a dot product, the optimiser is the
//! same one that trains a network.
//!
//! Run with `PHYZ_SOLVER_ADJOINT=1` to close the contact channel through the
//! solver's executed sweeps rather than the implicit function theorem.
//!
//! ```text
//! cargo run --release -p phyz-tang --example trajopt_box
//! ```

use std::time::Instant;

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};
use phyz_tang::{PhysicsStep, PhysicsTape};
use tang_tensor::{Shape, Tensor};
use tang_train::{ModuleAdam, Optimizer, Parameter};

const HALF: f64 = 0.05;
const STEPS: usize = 40;
/// The two steps whose controls are free. 2 x 6 DOF = the 12 decision
/// variables; every other step coasts.
const PUSH_AT: [usize; 2] = [0, STEPS / 3];
const TARGET: [f64; 2] = [0.06, 0.03];

fn box_model() -> Model {
    let mass = 1.0;
    let ix = mass / 12.0 * (2.0 * HALF) * (2.0 * HALF) * 2.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(ix, ix, ix)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(HALF, HALF, HALF),
    });
    model
}

fn main() {
    let model = box_model();
    let op = PhysicsStep::new(&model);
    let nv = op.ctrl_dim();
    let nq = model.nq;

    let mut state0 = Tensor::<f64>::zeros(Shape::from_slice(&[op.state_dim()]));
    state0.data_mut()[5] = HALF; // resting on the plane

    // The 12 decision variables: two 6-vectors, flat.
    let mut u = Parameter::new(Tensor::<f64>::zeros(Shape::from_slice(&[
        PUSH_AT.len() * nv
    ])));
    let mut opt = ModuleAdam::new(60.0);
    let zero = Tensor::<f64>::zeros(Shape::from_slice(&[nv]));

    let t0 = Instant::now();
    let mut converged = None;

    for iter in 0..200 {
        // Forward: 40 chained PhysicsStep ops, controls read off the parameter.
        let mut tape = PhysicsTape::new(&op, state0.clone());
        for t in 0..STEPS {
            let c = match PUSH_AT.iter().position(|&p| p == t) {
                Some(k) => Tensor::new(
                    u.data.data()[k * nv..(k + 1) * nv].to_vec(),
                    Shape::from_slice(&[nv]),
                ),
                None => zero.clone(),
            };
            tape.step(&c).unwrap();
        }

        // Loss: squared distance of the final xy from the target.
        let s = tape.state().data();
        let (dx, dy) = (s[nq - 3] - TARGET[0], s[nq - 2] - TARGET[1]);
        let loss = dx * dx + dy * dy;

        if iter % 20 == 0 || loss < 1e-6 {
            println!(
                "iter {iter:3}  loss {loss:.6e}  xy ({:.5}, {:.5})",
                s[nq - 3],
                s[nq - 2]
            );
        }
        if loss < 1e-6 {
            converged = Some(iter);
            break;
        }

        // Seed the cotangent on the final state, pull it back through all 40.
        let mut w = Tensor::<f64>::zeros(Shape::from_slice(&[op.state_dim()]));
        w.data_mut()[nq - 3] = 2.0 * dx;
        w.data_mut()[nq - 2] = 2.0 * dy;
        let g = tape.backward(&w).expect("adjoint refused");

        // Gather the per-step control gradients back into the flat parameter.
        let mut flat = vec![0.0; PUSH_AT.len() * nv];
        for (k, &t) in PUSH_AT.iter().enumerate() {
            flat[k * nv..(k + 1) * nv].copy_from_slice(g.d_ctrl[t].data());
        }
        u.zero_grad();
        u.accumulate_grad(&Tensor::new(flat, u.data.shape().clone()));
        opt.step(&mut [&mut u]);
    }

    let wall = t0.elapsed();
    match converged {
        Some(i) => println!("\nconverged in {i} iterations, {:.2?} wall", wall),
        None => println!("\ndid not reach 1e-6 in 200 iterations, {:.2?} wall", wall),
    }
    println!("controls {:?}", u.data.data());
}
