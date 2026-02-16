//! Equation-free simulation demonstration.
//!
//! Shows how to extract coarse-scale dynamics from fine-scale simulation
//! without deriving explicit macroscopic equations.

use tau_lbm::equation_free::{
    CoarseProjector, EquationFreeWrapper, FineSolver, SpatialAverageProjector,
    effective_information,
};

/// Simple reaction-diffusion fine solver (discrete).
#[derive(Clone)]
struct ReactionDiffusionSolver {
    nx: usize,
    dt: f64,
    dx: f64,
    diffusion_coeff: f64,
    reaction_rate: f64,
}

impl ReactionDiffusionSolver {
    fn new(nx: usize, diffusion_coeff: f64, reaction_rate: f64) -> Self {
        Self {
            nx,
            dt: 0.001,
            dx: 1.0 / nx as f64,
            diffusion_coeff,
            reaction_rate,
        }
    }
}

impl FineSolver for ReactionDiffusionSolver {
    type State = Vec<f64>;

    fn step(&mut self, state: &mut Self::State) {
        let mut new_state = state.clone();
        let d_over_dx2 = self.diffusion_coeff / (self.dx * self.dx);

        for i in 1..self.nx - 1 {
            // Diffusion: ∂u/∂t = D ∇²u
            let laplacian = state[i - 1] - 2.0 * state[i] + state[i + 1];
            let diffusion = d_over_dx2 * laplacian;

            // Reaction: ∂u/∂t = k u (1 - u)  (logistic growth)
            let reaction = self.reaction_rate * state[i] * (1.0 - state[i]);

            new_state[i] = state[i] + self.dt * (diffusion + reaction);
        }

        // Boundary conditions: zero flux
        new_state[0] = new_state[1];
        new_state[self.nx - 1] = new_state[self.nx - 2];

        *state = new_state;
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

fn main() {
    println!("Equation-Free Simulation Demo");
    println!("=============================");
    println!();

    // Fine-scale setup
    let nx_fine = 1000;
    let nx_coarse = 50;

    let fine_solver = ReactionDiffusionSolver::new(nx_fine, 0.01, 0.1);
    let projector = SpatialAverageProjector::new(nx_fine, nx_coarse);

    println!("Fine grid: {} points", nx_fine);
    println!("Coarse grid: {} points", nx_coarse);
    println!();

    // Initial condition: localized Gaussian pulse
    let mut macro_state = vec![0.0; nx_coarse];
    for i in 0..nx_coarse {
        let x = i as f64 / nx_coarse as f64;
        macro_state[i] = 0.5 * (-((x - 0.5) * (x - 0.5)) / 0.01).exp();
    }

    // Create equation-free wrapper
    let mut ef_wrapper = EquationFreeWrapper::new(fine_solver, projector, 100, 1.0);

    println!("Running equation-free simulation...");
    println!("Step    Max(u)    Center(u)");
    println!("----    ------    ---------");

    let n_steps = 100;
    for step in 0..=n_steps {
        if step % 10 == 0 {
            let max_u = macro_state.iter().cloned().fold(0.0, f64::max);
            let center_u = macro_state[nx_coarse / 2];
            println!("{:4}    {:.4}     {:.4}", step, max_u, center_u);
        }

        ef_wrapper.step(&mut macro_state);
    }

    println!();
    println!("✓ Equation-free simulation complete");

    // Measure effective information
    println!("\nMeasuring emergence...");

    // Collect trajectory of fine states
    let mut fine_solver2 = ReactionDiffusionSolver::new(nx_fine, 0.01, 0.1);
    let projector2 = SpatialAverageProjector::new(nx_fine, nx_coarse);

    let mut fine_state = projector2.lift(&vec![0.5; nx_coarse]);
    let mut trajectory = Vec::new();

    for _ in 0..50 {
        trajectory.push(fine_state.clone());
        fine_solver2.step(&mut fine_state);
    }

    // Test different coarse dimensions
    println!("\nEffective Information (variance explained):");
    println!("Coarse dim    Explained variance");
    println!("----------    ------------------");
    for dim in [1, 2, 5, 10, 20, 50] {
        if dim <= nx_coarse {
            let ei = effective_information(&trajectory, dim);
            println!("{:10}    {:.4}", dim, ei);
        }
    }

    println!("\n✓ Emergence analysis complete");
}
