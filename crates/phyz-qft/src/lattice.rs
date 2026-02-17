//! 4D lattice structure for gauge theory.

use crate::group::Group;

/// 4D lattice for gauge theory simulation.
///
/// Links U_μ(n) connect site n to n+μ in direction μ ∈ {0,1,2,3}.
/// Total of 4 * nt * nx * ny * nz links.
pub struct Lattice<G: Group> {
    /// Lattice dimensions [nt, nx, ny, nz].
    pub nt: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Inverse coupling constant β = 2N/g² for SU(N).
    pub beta: f64,

    /// Link variables U_μ(n).
    /// Storage: links[direction][site_index] where direction ∈ {0,1,2,3}
    /// and site_index is linearized from (t, x, y, z).
    links: Vec<Vec<G>>,

    /// Number of lattice sites.
    n_sites: usize,
}

impl<G: Group> Lattice<G> {
    /// Create new lattice with all links set to identity.
    pub fn new(nt: usize, nx: usize, ny: usize, nz: usize, beta: f64) -> Self {
        let n_sites = nt * nx * ny * nz;
        let mut links = Vec::with_capacity(4);
        for _ in 0..4 {
            links.push(vec![G::identity(); n_sites]);
        }

        Self {
            nt,
            nx,
            ny,
            nz,
            beta,
            links,
            n_sites,
        }
    }

    /// Initialize with random configuration (hot start).
    pub fn randomize(&mut self) {
        for mu in 0..4 {
            for site in 0..self.n_sites {
                self.links[mu][site] = G::random();
            }
        }
    }

    /// Get link U_μ(n).
    #[inline]
    pub fn get_link(&self, mu: usize, site: usize) -> &G {
        &self.links[mu][site]
    }

    /// Get mutable link U_μ(n).
    #[inline]
    pub fn get_link_mut(&mut self, mu: usize, site: usize) -> &mut G {
        &mut self.links[mu][site]
    }

    /// Set link U_μ(n).
    #[inline]
    pub fn set_link(&mut self, mu: usize, site: usize, value: G) {
        self.links[mu][site] = value;
    }

    /// Convert (t, x, y, z) coordinates to linear site index.
    #[inline]
    pub fn site_index(&self, t: usize, x: usize, y: usize, z: usize) -> usize {
        t + self.nt * (x + self.nx * (y + self.ny * z))
    }

    /// Convert linear site index to (t, x, y, z) coordinates.
    #[inline]
    pub fn site_coords(&self, site: usize) -> (usize, usize, usize, usize) {
        let t = site % self.nt;
        let x = (site / self.nt) % self.nx;
        let y = (site / (self.nt * self.nx)) % self.ny;
        let z = site / (self.nt * self.nx * self.ny);
        (t, x, y, z)
    }

    /// Get neighboring site in direction mu (with periodic BC).
    #[inline]
    pub fn neighbor(&self, site: usize, mu: usize) -> usize {
        let (t, x, y, z) = self.site_coords(site);
        let (t, x, y, z) = match mu {
            0 => ((t + 1) % self.nt, x, y, z),
            1 => (t, (x + 1) % self.nx, y, z),
            2 => (t, x, (y + 1) % self.ny, z),
            3 => (t, x, y, (z + 1) % self.nz),
            _ => panic!("Invalid direction: {}", mu),
        };
        self.site_index(t, x, y, z)
    }

    /// Get neighboring site in backward direction -mu (with periodic BC).
    #[inline]
    pub fn neighbor_back(&self, site: usize, mu: usize) -> usize {
        let (t, x, y, z) = self.site_coords(site);
        let (t, x, y, z) = match mu {
            0 => ((t + self.nt - 1) % self.nt, x, y, z),
            1 => (t, (x + self.nx - 1) % self.nx, y, z),
            2 => (t, x, (y + self.ny - 1) % self.ny, z),
            3 => (t, x, y, (z + self.nz - 1) % self.nz),
            _ => panic!("Invalid direction: {}", mu),
        };
        self.site_index(t, x, y, z)
    }

    /// Compute plaquette U_μν(n) = U_μ(n) U_ν(n+μ) U_μ†(n+ν) U_ν†(n).
    pub fn plaquette(&self, site: usize, mu: usize, nu: usize) -> G {
        let n_mu = self.neighbor(site, mu);
        let n_nu = self.neighbor(site, nu);

        // U_μ(n)
        let u1 = self.get_link(mu, site);
        // U_ν(n+μ)
        let u2 = self.get_link(nu, n_mu);
        // U_μ†(n+ν)
        let u3 = self.get_link(mu, n_nu).inv();
        // U_ν†(n)
        let u4 = self.get_link(nu, site).inv();

        u1.mul(u2).mul(&u3).mul(&u4)
    }

    /// Compute staple sum for link U_μ(n).
    ///
    /// Staple sum = Σ_ν≠μ [upper_staple(ν) + lower_staple(ν)]
    /// where upper_staple(ν) = U_ν(n+μ) U_μ†(n+ν) U_ν†(n)
    /// and lower_staple(ν) = U_ν†(n+μ-ν) U_μ†(n-ν) U_ν(n-ν)
    pub fn staple_sum(&self, site: usize, mu: usize) -> G {
        let mut staple = G::identity();
        let first = true;

        for nu in 0..4 {
            if nu == mu {
                continue;
            }

            // Upper staple: U_ν(n+μ) U_μ†(n+ν) U_ν†(n)
            let n_mu = self.neighbor(site, mu);
            let n_nu = self.neighbor(site, nu);

            let upper = self
                .get_link(nu, n_mu)
                .mul(&self.get_link(mu, n_nu).inv())
                .mul(&self.get_link(nu, site).inv());

            if first {
                staple = upper;
            } else {
                staple = staple.mul(&upper);
            }

            // Lower staple: U_ν†(n+μ-ν) U_μ†(n-ν) U_ν(n-ν)
            let n_back_nu = self.neighbor_back(site, nu);
            let n_mu_back_nu = self.neighbor_back(n_mu, nu);

            let lower = self
                .get_link(nu, n_mu_back_nu)
                .inv()
                .mul(&self.get_link(mu, n_back_nu).inv())
                .mul(self.get_link(nu, n_back_nu));

            staple = staple.mul(&lower);
        }

        staple
    }

    /// Compute total Wilson action S = β/N Σ (1 - Re Tr U_plaq).
    pub fn action(&self) -> f64 {
        let mut s = 0.0;
        for site in 0..self.n_sites {
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let plaq = self.plaquette(site, mu, nu);
                    // For U(1): Re Tr = cos(θ), already in [-1,1]
                    // For SU(N): would need normalization by N
                    s += 1.0 - plaq.re_tr();
                }
            }
        }
        self.beta * s
    }

    /// Compute average plaquette ⟨Re Tr U_plaq⟩.
    pub fn average_plaquette(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for site in 0..self.n_sites {
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let plaq = self.plaquette(site, mu, nu);
                    sum += plaq.re_tr();
                    count += 1;
                }
            }
        }

        sum / count as f64
    }

    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of links (4 per site).
    pub fn n_links(&self) -> usize {
        4 * self.n_sites
    }

    /// Clone links (for HMC checkpointing).
    pub(crate) fn clone_links(&self) -> Vec<Vec<G>> {
        self.links.clone()
    }

    /// Restore links from checkpoint (for HMC rejection).
    pub(crate) fn restore_links(&mut self, links: Vec<Vec<G>>) {
        self.links = links;
    }

    /// Set link without bounds checking (for HMC update).
    #[inline]
    pub(crate) fn set_link_unchecked(&mut self, mu: usize, site: usize, value: G) {
        self.links[mu][site] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::U1;

    #[test]
    fn test_lattice_creation() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        assert_eq!(lattice.n_sites(), 256);
        assert_eq!(lattice.n_links(), 1024);
    }

    #[test]
    fn test_site_indexing() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        let site = lattice.site_index(1, 2, 3, 0);
        let (t, x, y, z) = lattice.site_coords(site);
        assert_eq!((t, x, y, z), (1, 2, 3, 0));
    }

    #[test]
    fn test_neighbors() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        let site = lattice.site_index(0, 0, 0, 0);

        // Forward neighbor
        let n0 = lattice.neighbor(site, 0);
        assert_eq!(lattice.site_coords(n0), (1, 0, 0, 0));

        // Backward neighbor (wraps around)
        let n0_back = lattice.neighbor_back(site, 0);
        assert_eq!(lattice.site_coords(n0_back), (3, 0, 0, 0));
    }

    #[test]
    fn test_identity_action() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        // All links are identity, so plaquettes are identity
        // Re Tr(I) = 1 for U(1), so action = β * Σ(1-1) = 0
        let action = lattice.action();
        assert!(action.abs() < 1e-10);
    }

    #[test]
    fn test_average_plaquette_identity() {
        let lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        let plaq = lattice.average_plaquette();
        // For U(1) with identity links, Re Tr U = cos(0) = 1
        assert!((plaq - 1.0).abs() < 1e-10);
    }
}
