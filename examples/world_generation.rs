//! Example: Procedural world generation.
//!
//! Demonstrates using WorldGenerator to create random articulated systems.

use tau_world::WorldGenerator;

fn main() {
    println!("=== Tau World Generation Example ===\n");

    let mut generator = WorldGenerator::new(42);

    // Generate a random chain
    println!("1. Random chain (5 links):");
    let chain = generator.random_chain(5, [0.5, 2.0], [0.3, 1.0]);
    println!("   Bodies: {}", chain.nbodies());
    println!("   DOFs: {} position, {} velocity", chain.nq, chain.nv);
    for (i, body) in chain.bodies.iter().enumerate() {
        println!(
            "   Body {}: mass={:.2}, name={}",
            i, body.inertia.mass, body.name
        );
    }

    // Generate a quadruped
    println!("\n2. Random quadruped:");
    let quad = generator.random_quadruped();
    println!("   Bodies: {}", quad.nbodies());
    println!("   DOFs: {} position, {} velocity", quad.nq, quad.nv);
    println!("   Structure: 1 torso (free) + 4 legs × 2 segments");

    // Generate platform with obstacles
    println!("\n3. Platform with obstacles:");
    let platform = generator.platform_with_obstacles(10.0, 8, [0.5, 2.0]);
    println!("   Bodies: {}", platform.nbodies());
    println!("   DOFs: {} (all fixed)", platform.nv);
    println!("   Platform width: 10m, obstacles: 8");

    // Generate random tree
    println!("\n4. Random tree structure:");
    let tree = generator.random_tree(3, 2);
    println!("   Bodies: {}", tree.nbodies());
    println!("   DOFs: {} position, {} velocity", tree.nq, tree.nv);
    println!("   Depth: 3, branching factor: 2");

    // Multiple chains with different seeds
    println!("\n5. Testing determinism:");
    let mut gen1 = WorldGenerator::new(999);
    let mut gen2 = WorldGenerator::new(999);
    let m1 = gen1.random_chain(3, [0.5, 1.5], [0.4, 0.8]);
    let m2 = gen2.random_chain(3, [0.5, 1.5], [0.4, 0.8]);
    println!(
        "   Same seed produces identical models: {}",
        m1.bodies[0].inertia.mass == m2.bodies[0].inertia.mass
    );

    println!("\n✓ World generation complete!");
}
