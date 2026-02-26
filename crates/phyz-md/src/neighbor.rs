//! Neighbor list for efficient force computation.

use crate::Particle;
use phyz_math::Vec3;

/// Neighbor list for efficient pairwise force computation.
#[derive(Clone, Debug)]
pub struct NeighborList {
    /// List of neighbor pairs (i, j) where i < j.
    pub pairs: Vec<(usize, usize)>,
    /// Cutoff radius.
    pub r_cut: f64,
    /// Skin distance (buffer for rebuilding).
    pub skin: f64,
    /// Last rebuild positions.
    last_positions: Vec<Vec3>,
}

impl NeighborList {
    /// Create a new neighbor list.
    pub fn new(r_cut: f64, skin: f64) -> Self {
        Self {
            pairs: Vec::new(),
            r_cut,
            skin,
            last_positions: Vec::new(),
        }
    }

    /// Build neighbor list from particles.
    pub fn build(&mut self, particles: &[Particle], box_size: Option<Vec3>) {
        self.pairs.clear();
        self.last_positions.clear();

        let r_search = self.r_cut + self.skin;
        let r_search_sq = r_search * r_search;

        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let mut dr = particles[j].x - particles[i].x;

                // Apply minimum image convention if periodic
                if let Some(box_size) = box_size {
                    dr = minimum_image(dr, box_size);
                }

                if dr.norm_squared() < r_search_sq {
                    self.pairs.push((i, j));
                }
            }

            self.last_positions.push(particles[i].x);
        }
    }

    /// Check if rebuild is needed based on particle displacement.
    pub fn needs_rebuild(&self, particles: &[Particle], box_size: Option<Vec3>) -> bool {
        if self.last_positions.len() != particles.len() {
            return true;
        }

        let max_disp_sq = (0.5 * self.skin).powi(2);

        for (i, last_x) in self.last_positions.iter().enumerate() {
            let mut dr = particles[i].x - last_x;

            // Apply minimum image if periodic
            if let Some(box_size) = box_size {
                dr = minimum_image(dr, box_size);
            }

            if dr.norm_squared() > max_disp_sq {
                return true;
            }
        }

        false
    }
}

/// Apply minimum image convention for periodic boundaries.
pub fn minimum_image(mut dr: Vec3, box_size: Vec3) -> Vec3 {
    fn wrap(val: f64, size: f64) -> f64 {
        if val > 0.5 * size { val - size }
        else if val < -0.5 * size { val + size }
        else { val }
    }
    dr.x = wrap(dr.x, box_size.x);
    dr.y = wrap(dr.y, box_size.y);
    dr.z = wrap(dr.z, box_size.z);
    dr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimum_image() {
        let box_size = Vec3::new(10.0, 10.0, 10.0);
        let dr = Vec3::new(6.0, 3.0, -7.0);

        let dr_min = minimum_image(dr, box_size);

        // 6.0 > 5.0, so wrap: 6.0 - 10.0 = -4.0
        // 3.0 is fine
        // -7.0 < -5.0, so wrap: -7.0 + 10.0 = 3.0
        assert_eq!(dr_min, Vec3::new(-4.0, 3.0, 3.0));
    }

    #[test]
    fn test_neighbor_list_build() {
        let particles = vec![
            Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0),
            Particle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0),
            Particle::new(Vec3::new(5.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0),
        ];

        let mut nlist = NeighborList::new(2.0, 0.5);
        nlist.build(&particles, None);

        // Pairs (0,1) should be in list (r=1.0 < 2.5)
        // Pair (0,2) should NOT be in list (r=5.0 > 2.5)
        // Pair (1,2) should NOT be in list (r=4.0 > 2.5)
        assert_eq!(nlist.pairs.len(), 1);
        assert!(nlist.pairs.contains(&(0, 1)));
    }

    #[test]
    fn test_neighbor_list_rebuild() {
        let mut particles = vec![
            Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0),
            Particle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0),
        ];

        let mut nlist = NeighborList::new(2.0, 0.5);
        nlist.build(&particles, None);

        // Should not need rebuild initially
        assert!(!nlist.needs_rebuild(&particles, None));

        // Move particle slightly (< 0.5 * skin)
        particles[0].x += Vec3::new(0.1, 0.0, 0.0);
        assert!(!nlist.needs_rebuild(&particles, None));

        // Move particle significantly (> 0.5 * skin)
        particles[0].x += Vec3::new(0.3, 0.0, 0.0);
        assert!(nlist.needs_rebuild(&particles, None));
    }
}
