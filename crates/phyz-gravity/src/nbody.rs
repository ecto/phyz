//! N-body gravity solver with Barnes-Hut tree (Layer 3).
//!
//! Implements:
//! - Naive O(N²) pairwise forces
//! - Barnes-Hut octree for O(N log N) approximation
//!
//! # Barnes-Hut Algorithm
//!
//! 1. Build octree with center-of-mass for each node
//! 2. For each particle, traverse tree:
//!    - If node far enough away (θ test), use COM approximation
//!    - Otherwise, recurse to children
//! 3. θ = s/d (cell size / distance); larger θ = faster, less accurate

use crate::{G, GravityParticle, GravitySolver};
use phyz_math::Vec3;

/// N-body gravity solver.
#[derive(Debug, Clone)]
pub struct NBodySolver {
    /// Use Barnes-Hut tree approximation.
    pub use_tree: bool,
    /// Barnes-Hut opening angle parameter.
    pub theta: f64,
    /// Softening length to prevent singularities (m).
    pub softening: f64,
}

impl NBodySolver {
    /// Create a new N-body solver.
    pub fn new() -> Self {
        Self {
            use_tree: false,
            theta: 0.5,
            softening: 1e-3,
        }
    }

    /// Create with Barnes-Hut tree.
    pub fn with_tree(theta: f64, softening: f64) -> Self {
        Self {
            use_tree: true,
            theta,
            softening,
        }
    }

    /// Compute pairwise gravitational force (naive O(N²)).
    pub fn compute_pairwise_forces(&self, particles: &mut [GravityParticle]) {
        let n = particles.len();

        // Reset forces
        for p in particles.iter_mut() {
            p.reset_force();
        }

        // Pairwise forces
        for i in 0..n {
            for j in i + 1..n {
                let r = particles[j].x - particles[i].x;
                let r2 = r.norm_squared() + self.softening * self.softening;
                let r_mag = r2.sqrt();

                // F = G * m1 * m2 / r² * r̂
                let f_mag = G * particles[i].m * particles[j].m / r2;
                let f = r / r_mag * f_mag;

                // Newton's third law
                particles[i].add_force(f);
                particles[j].add_force(-f);
            }
        }
    }
}

impl Default for NBodySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl GravitySolver for NBodySolver {
    fn compute_forces(&mut self, particles: &mut [GravityParticle]) {
        if self.use_tree {
            // Build Barnes-Hut tree
            let tree = BarnesHutTree::build(particles, self.softening);
            tree.compute_forces(particles, self.theta);
        } else {
            self.compute_pairwise_forces(particles);
        }
    }

    fn potential_energy(&self, particles: &[GravityParticle]) -> f64 {
        let n = particles.len();
        let mut u = 0.0;

        for i in 0..n {
            for j in i + 1..n {
                let r = (particles[j].x - particles[i].x).norm();
                let r_soft = (r * r + self.softening * self.softening).sqrt();
                u -= G * particles[i].m * particles[j].m / r_soft;
            }
        }

        u
    }
}

/// Barnes-Hut octree node.
#[derive(Debug, Clone)]
pub struct OctreeNode {
    /// Center of mass.
    pub com: Vec3,
    /// Total mass.
    pub mass: f64,
    /// Bounding box center.
    pub center: Vec3,
    /// Half-width of box.
    pub half_size: f64,
    /// Children (8 octants), None if leaf.
    pub children: Option<Box<[OctreeNode; 8]>>,
    /// Particle indices (if leaf).
    pub particles: Vec<usize>,
}

/// Maximum octree subdivision depth. Prevents infinite recursion when two or
/// more particles are coincident (or closer than f64 can separate at this
/// scale); such particles end up sharing a multi-particle leaf.
const MAX_DEPTH: u32 = 32;

/// Octant index of `x` relative to a box centered at `center`.
fn octant_of(center: Vec3, x: Vec3) -> usize {
    let mut idx = 0;
    if x.x >= center.x {
        idx |= 1;
    }
    if x.y >= center.y {
        idx |= 2;
    }
    if x.z >= center.z {
        idx |= 4;
    }
    idx
}

impl OctreeNode {
    /// Create a new empty node.
    fn new(center: Vec3, half_size: f64) -> Self {
        Self {
            com: Vec3::zeros(),
            mass: 0.0,
            center,
            half_size,
            children: None,
            particles: Vec::new(),
        }
    }

    /// Check if node is a leaf.
    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Get octant index for a position.
    fn octant(&self, x: Vec3) -> usize {
        octant_of(self.center, x)
    }

    /// Get child center for octant.
    fn child_center(&self, octant: usize) -> Vec3 {
        let offset = self.half_size / 2.0;
        Vec3::new(
            self.center.x + if octant & 1 != 0 { offset } else { -offset },
            self.center.y + if octant & 2 != 0 { offset } else { -offset },
            self.center.z + if octant & 4 != 0 { offset } else { -offset },
        )
    }

    /// Insert a particle into the tree.
    ///
    /// `all` is the full particle slice so that a leaf being subdivided can look
    /// up the position of the occupant it has to push down into a child.
    /// `depth` guards against infinite subdivision of coincident particles: at
    /// [`MAX_DEPTH`] the node stays a multi-particle leaf.
    fn insert(&mut self, particle_idx: usize, all: &[GravityParticle], depth: u32) {
        let particle_pos = all[particle_idx].x;
        let particle_mass = all[particle_idx].m;

        // Update center of mass
        let total_mass = self.mass + particle_mass;
        if total_mass > 0.0 {
            self.com = (self.com * self.mass + particle_pos * particle_mass) / total_mass;
        }
        self.mass = total_mass;

        if self.is_leaf() {
            if self.particles.is_empty() || depth >= MAX_DEPTH {
                // Empty leaf, or depth cap reached: keep as a (possibly
                // multi-particle) leaf. Coincident particles land here.
                self.particles.push(particle_idx);
                return;
            }

            // Occupied leaf: subdivide and push the existing occupants down
            // alongside the new particle.
            let existing = std::mem::take(&mut self.particles);

            let half = self.half_size / 2.0;
            self.children = Some(Box::new([
                OctreeNode::new(self.child_center(0), half),
                OctreeNode::new(self.child_center(1), half),
                OctreeNode::new(self.child_center(2), half),
                OctreeNode::new(self.child_center(3), half),
                OctreeNode::new(self.child_center(4), half),
                OctreeNode::new(self.child_center(5), half),
                OctreeNode::new(self.child_center(6), half),
                OctreeNode::new(self.child_center(7), half),
            ]));

            let children = self.children.as_mut().expect("just set");
            for idx in existing {
                let octant = octant_of(self.center, all[idx].x);
                children[octant].insert(idx, all, depth + 1);
            }
            let octant = octant_of(self.center, particle_pos);
            children[octant].insert(particle_idx, all, depth + 1);
        } else {
            // Internal node: recurse
            let octant = self.octant(particle_pos);
            if let Some(ref mut children) = self.children {
                children[octant].insert(particle_idx, all, depth + 1);
            }
        }
    }

    /// Compute gravitational acceleration from this node on a particle.
    fn acceleration(&self, x: Vec3, softening: f64) -> Vec3 {
        let r = self.com - x;
        let r2 = r.norm_squared() + softening * softening;
        let r_mag = r2.sqrt();

        // a = G * M / r² * r̂
        G * self.mass / r2 * (r / r_mag)
    }

    /// Recursively compute the force on particle `i`.
    ///
    /// Leaves are summed directly (excluding `i` itself, so a particle never
    /// attracts itself through its own leaf's center of mass). Internal nodes
    /// use the COM approximation once the opening-angle criterion is met.
    fn compute_force_on(
        &self,
        i: usize,
        all: &[GravityParticle],
        theta: f64,
        softening: f64,
    ) -> Vec3 {
        if self.mass == 0.0 {
            return Vec3::zeros();
        }

        let particle = &all[i];

        if self.is_leaf() {
            // Direct summation over the leaf's occupants, skipping self.
            let mut force = Vec3::zeros();
            for &j in &self.particles {
                if j == i {
                    continue;
                }
                let r = all[j].x - particle.x;
                let r2 = r.norm_squared() + softening * softening;
                let r_mag = r2.sqrt();
                force += r / r_mag * (G * particle.m * all[j].m / r2);
            }
            return force;
        }

        let r = (self.com - particle.x).norm();

        // Barnes-Hut criterion: s/d < θ
        let s = 2.0 * self.half_size;
        if r > 0.0 && (s / r) < theta {
            // Use COM approximation
            self.acceleration(particle.x, softening) * particle.m
        } else {
            // Recurse to children
            let mut force = Vec3::zeros();
            if let Some(ref children) = self.children {
                for child in children.iter() {
                    force += child.compute_force_on(i, all, theta, softening);
                }
            }
            force
        }
    }
}

/// Barnes-Hut tree for O(N log N) gravity.
#[derive(Debug, Clone)]
pub struct BarnesHutTree {
    /// Root node.
    pub root: OctreeNode,
    /// Softening length.
    pub softening: f64,
}

impl BarnesHutTree {
    /// Build tree from particles.
    pub fn build(particles: &[GravityParticle], softening: f64) -> Self {
        // Compute bounding box
        let mut min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for p in particles {
            min.x = min.x.min(p.x.x);
            min.y = min.y.min(p.x.y);
            min.z = min.z.min(p.x.z);
            max.x = max.x.max(p.x.x);
            max.y = max.y.max(p.x.y);
            max.z = max.z.max(p.x.z);
        }

        if particles.is_empty() {
            return Self {
                root: OctreeNode::new(Vec3::zeros(), 1.0),
                softening,
            };
        }

        let center = (min + max) / 2.0;
        // Half-width large enough to contain the whole bounding box, with 10% padding.
        let half_size = ((max - min).norm() / 2.0) * 1.1;
        let half_size = if half_size > 0.0 { half_size } else { 1.0 };

        let mut root = OctreeNode::new(center, half_size);

        // Insert all particles
        for i in 0..particles.len() {
            root.insert(i, particles, 0);
        }

        Self { root, softening }
    }

    /// Compute forces on all particles using tree.
    pub fn compute_forces(&self, particles: &mut [GravityParticle], theta: f64) {
        let forces: Vec<Vec3> = (0..particles.len())
            .map(|i| self.root.compute_force_on(i, particles, theta, self.softening))
            .collect();

        for (p, f) in particles.iter_mut().zip(forces) {
            p.reset_force();
            p.add_force(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbody_two_particle() {
        let mut solver = NBodySolver::new();
        let mut particles = vec![
            GravityParticle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1e10),
            GravityParticle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        ];

        solver.compute_forces(&mut particles);

        // Force should be along x-axis
        assert!(particles[0].f.y.abs() < 1e-20);
        assert!(particles[0].f.z.abs() < 1e-20);

        // Newton's third law
        assert!((particles[0].f.x + particles[1].f.x).abs() < 1e-20);
    }

    #[test]
    fn test_barnes_hut_tree() {
        let particles = vec![
            GravityParticle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1e10),
            GravityParticle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1e10),
            GravityParticle::new(Vec3::new(0.0, 1.0, 0.0), Vec3::zeros(), 1e10),
        ];

        let tree = BarnesHutTree::build(&particles, 1e-3);

        assert_eq!(tree.root.mass, 3e10);
        assert!(tree.root.half_size > 0.0);

        // Every particle must be reachable in exactly one leaf.
        let mut found = vec![0usize; 3];
        collect_leaf_indices(&tree.root, &mut found);
        assert_eq!(found, vec![1, 1, 1], "each particle stored exactly once");
    }

    fn collect_leaf_indices(node: &OctreeNode, counts: &mut [usize]) {
        if let Some(ref children) = node.children {
            for c in children.iter() {
                collect_leaf_indices(c, counts);
            }
        } else {
            for &i in &node.particles {
                counts[i] += 1;
            }
        }
    }

    /// Deterministic LCG so the test is reproducible without a `rand` dep.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 53 bits -> [0, 1)
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        /// Uniform in [-1, 1).
        fn next_sym(&mut self) -> f64 {
            self.next_f64() * 2.0 - 1.0
        }
    }

    fn random_cloud(n: usize, seed: u64) -> Vec<GravityParticle> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let x = Vec3::new(rng.next_sym(), rng.next_sym(), rng.next_sym());
                let m = 1e9 * (0.5 + rng.next_f64());
                GravityParticle::new(x, Vec3::zeros(), m)
            })
            .collect()
    }

    /// Mean relative error of Barnes-Hut forces vs. the O(N²) reference.
    fn bh_error(theta: f64, particles: &[GravityParticle], softening: f64) -> f64 {
        let mut reference = particles.to_vec();
        NBodySolver {
            use_tree: false,
            theta,
            softening,
        }
        .compute_pairwise_forces(&mut reference);

        let mut approx = particles.to_vec();
        BarnesHutTree::build(&approx, softening).compute_forces(&mut approx, theta);

        let mut total = 0.0;
        for (a, r) in approx.iter().zip(reference.iter()) {
            let denom = r.f.norm();
            assert!(denom > 0.0);
            total += (a.f - r.f).norm() / denom;
        }
        total / particles.len() as f64
    }

    #[test]
    fn test_barnes_hut_matches_direct_summation() {
        let particles = random_cloud(400, 0x5eed);
        let softening = 1e-2;

        let e_tight = bh_error(0.1, &particles, softening);
        let e_mid = bh_error(0.5, &particles, softening);
        let e_loose = bh_error(1.0, &particles, softening);

        // Small theta => nearly exact (theta -> 0 degenerates to direct summation).
        assert!(e_tight < 1e-3, "theta=0.1 mean rel err {e_tight}");
        // Default theta => a few percent.
        assert!(e_mid < 3e-2, "theta=0.5 mean rel err {e_mid}");
        // Loose theta => still a usable approximation, just cruder.
        assert!(e_loose < 2e-1, "theta=1.0 mean rel err {e_loose}");

        // Accuracy must improve monotonically as the opening angle tightens.
        assert!(
            e_tight < e_mid && e_mid < e_loose,
            "error should shrink with theta: {e_tight} < {e_mid} < {e_loose}"
        );
    }

    #[test]
    fn test_direct_summation_conserves_momentum() {
        let particles = random_cloud(200, 0xc0ffee);
        let mut direct = particles.clone();
        NBodySolver::new().compute_pairwise_forces(&mut direct);

        // Newton's third law: forces are antisymmetric, so dP/dt = sum(F) = 0.
        let net: Vec3 = direct.iter().fold(Vec3::zeros(), |acc, p| acc + p.f);
        let scale: f64 = direct.iter().map(|p| p.f.norm()).sum();
        assert!(
            net.norm() / scale < 1e-12,
            "net force {net:?} (scale {scale})"
        );
    }

    #[test]
    fn test_barnes_hut_no_self_interaction() {
        // A single particle feels no force at all.
        let mut particles = vec![GravityParticle::new(Vec3::zeros(), Vec3::zeros(), 1e12)];
        let tree = BarnesHutTree::build(&particles, 1e-3);
        tree.compute_forces(&mut particles, 0.5);
        assert_eq!(particles[0].f.norm(), 0.0);
    }

    #[test]
    fn test_barnes_hut_coincident_particles_terminate() {
        // Degenerate cloud: many exactly-coincident particles must not recurse forever.
        let mut particles: Vec<_> = (0..16)
            .map(|_| GravityParticle::new(Vec3::new(1.0, 2.0, 3.0), Vec3::zeros(), 1e9))
            .collect();
        particles.push(GravityParticle::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::zeros(),
            1e9,
        ));

        let tree = BarnesHutTree::build(&particles, 1e-3);
        assert!((tree.root.mass - 17e9).abs() < 1.0);

        tree.compute_forces(&mut particles, 0.5);
        assert!(particles.iter().all(|p| p.f.norm().is_finite()));
    }

    #[test]
    fn test_potential_energy() {
        let solver = NBodySolver::new();
        let particles = vec![
            GravityParticle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1e10),
            GravityParticle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        ];

        let u = solver.potential_energy(&particles);

        // U = -G*m1*m2/r ≈ -6.67e-11 * 1e10 * 1e10 / 1.0 = -6.67e9
        assert!(u < 0.0);
        assert!((u + 6.67e9).abs() < 1e8);
    }
}
