//! Cell-list-backed Verlet neighbor lists.
//!
//! The pair list holds every `(i, j)` with `i < j` inside `cutoff + skin`, so
//! it stays valid until some atom has moved more than `skin/2` — at which point
//! the two atoms of any pair may have closed `skin` between them and a pair
//! that was outside the search radius could have crossed the cutoff.
//!
//! Construction bins atoms into a lattice of cells at least `cutoff + skin`
//! wide and scans a half stencil (13 forward neighbor cells plus the home
//! cell), which is O(N) at fixed density instead of the O(N²) all-pairs double
//! loop. The `checks` counter records how many candidate distances were
//! evaluated, which makes the scaling directly assertable in a test rather than
//! inferred from wall-clock noise.

use std::collections::HashSet;

use super::cell::{Lattice, min_image, vec3};

/// The 13 forward half-stencil cell offsets plus the home cell `(0,0,0)`.
///
/// A half stencil visits each neighboring cell pair once. Combined with the
/// `i < j` rule inside the home cell, every pair is generated exactly once.
const HALF_STENCIL: [[i32; 3]; 14] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
];

/// A Verlet neighbor list built from cell lists.
#[derive(Clone, Debug, Default)]
pub struct NeighborList {
    /// Interaction cutoff in Å.
    pub cutoff: f64,
    /// Verlet skin in Å: pairs are listed out to `cutoff + skin`.
    pub skin: f64,
    /// Pairs `(i, j)` with `i < j` within `cutoff + skin` at the last build.
    pairs: Vec<(usize, usize)>,
    /// Positions at the last build, used by the rebuild heuristic.
    reference: Vec<[f64; 3]>,
    /// Candidate distances evaluated during the last build.
    checks: usize,
    /// Whether the last build used the O(N²) fallback.
    used_fallback: bool,
    /// Excluded pairs (`i < j`), skipped when listing.
    exclusions: HashSet<(usize, usize)>,
    /// Number of builds performed over this list's lifetime.
    builds: usize,
}

impl NeighborList {
    /// A list with the given interaction cutoff and Verlet skin (both Å).
    pub fn new(cutoff: f64, skin: f64) -> Self {
        Self {
            cutoff,
            skin: skin.max(0.0),
            ..Default::default()
        }
    }

    /// The search radius `cutoff + skin`.
    #[inline]
    pub fn search_radius(&self) -> f64 {
        self.cutoff + self.skin
    }

    /// The current pair list.
    #[inline]
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Candidate distances evaluated in the most recent build.
    ///
    /// For a cell-list build this grows linearly with atom count at fixed
    /// density; for the O(N²) fallback it is `N(N-1)/2`.
    #[inline]
    pub fn checks(&self) -> usize {
        self.checks
    }

    /// Whether the last build fell back to the all-pairs loop.
    #[inline]
    pub fn used_fallback(&self) -> bool {
        self.used_fallback
    }

    /// How many times this list has been rebuilt.
    #[inline]
    pub fn builds(&self) -> usize {
        self.builds
    }

    /// Exclude a pair from the list (bonded 1-2/1-3 neighbors, typically).
    ///
    /// Exclusions must be registered before the list is built.
    pub fn exclude(&mut self, i: usize, j: usize) {
        if i != j {
            self.exclusions.insert(ordered(i, j));
        }
    }

    /// Replace the exclusion set.
    pub fn set_exclusions<I: IntoIterator<Item = (usize, usize)>>(&mut self, pairs: I) {
        self.exclusions = pairs
            .into_iter()
            .filter(|(i, j)| i != j)
            .map(|(i, j)| ordered(i, j))
            .collect();
    }

    /// The excluded pairs, each stored as `(min, max)`.
    pub fn exclusions(&self) -> &HashSet<(usize, usize)> {
        &self.exclusions
    }

    /// Whether the pair list must be rebuilt for the given positions.
    ///
    /// True once any atom has drifted more than `skin/2` from its position at
    /// the last build, or when the atom count changed.
    pub fn needs_rebuild(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> bool {
        if self.reference.len() != positions.len() {
            return true;
        }
        if self.skin <= 0.0 {
            return true;
        }
        let limit = 0.5 * self.skin;
        let limit2 = limit * limit;
        positions.iter().zip(&self.reference).any(|(p, r)| {
            let d = min_image(vec3::sub(*p, *r), cell);
            vec3::norm2(d) > limit2
        })
    }

    /// Rebuild the pair list if the displacement heuristic calls for it.
    /// Returns whether a rebuild happened.
    pub fn maybe_build(&mut self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> bool {
        if self.needs_rebuild(positions, cell) {
            self.build(positions, cell);
            true
        } else {
            false
        }
    }

    /// Build the pair list from scratch.
    pub fn build(&mut self, positions: &[[f64; 3]], cell: Option<&Lattice>) {
        self.pairs.clear();
        self.checks = 0;
        self.builds += 1;
        self.reference.clear();
        self.reference.extend_from_slice(positions);

        let n = positions.len();
        if n < 2 {
            self.used_fallback = false;
            return;
        }
        let r_search = self.search_radius();
        let r2 = r_search * r_search;

        match Grid::build(positions, cell, r_search) {
            Some(grid) => {
                self.used_fallback = false;
                self.build_binned(positions, cell, &grid, r2);
            }
            None => {
                self.used_fallback = true;
                self.build_all_pairs(positions, cell, r2);
            }
        }
    }

    fn build_all_pairs(&mut self, positions: &[[f64; 3]], cell: Option<&Lattice>, r2: f64) {
        let n = positions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                self.consider(i, j, positions, cell, r2);
            }
        }
    }

    fn build_binned(
        &mut self,
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
        grid: &Grid,
        r2: f64,
    ) {
        let [nx, ny, nz] = grid.dims;
        for cx in 0..nx {
            for cy in 0..ny {
                for cz in 0..nz {
                    let home = grid.index(cx, cy, cz);
                    for (s, off) in HALF_STENCIL.iter().enumerate() {
                        let Some(other) =
                            grid.offset_index([cx as i32, cy as i32, cz as i32], *off)
                        else {
                            continue;
                        };
                        if s == 0 {
                            // Home cell: internal pairs, i < j.
                            let atoms = grid.atoms_in(home);
                            for (a, &i) in atoms.iter().enumerate() {
                                for &j in &atoms[a + 1..] {
                                    self.consider(i, j, positions, cell, r2);
                                }
                            }
                        } else {
                            for &i in grid.atoms_in(home) {
                                for &j in grid.atoms_in(other) {
                                    self.consider(i, j, positions, cell, r2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn consider(
        &mut self,
        i: usize,
        j: usize,
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
        r2: f64,
    ) {
        let (lo, hi) = (i.min(j), i.max(j));
        if self.exclusions.contains(&(lo, hi)) {
            return;
        }
        self.checks += 1;
        let d = min_image(vec3::sub(positions[lo], positions[hi]), cell);
        if vec3::norm2(d) <= r2 {
            self.pairs.push((lo, hi));
        }
    }
}

#[inline]
fn ordered(i: usize, j: usize) -> (usize, usize) {
    (i.min(j), i.max(j))
}

/// A uniform bin decomposition of the simulation domain.
///
/// Periodic systems bin in fractional coordinates (which handles triclinic
/// cells without special cases); aperiodic systems bin in Cartesian space over
/// the atoms' bounding box.
struct Grid {
    dims: [usize; 3],
    periodic: [bool; 3],
    /// CSR-style bucket storage: `starts[c]..starts[c+1]` indexes `items`.
    starts: Vec<usize>,
    items: Vec<usize>,
}

impl Grid {
    /// Build a grid whose cells are at least `r_search` wide in every
    /// direction, or `None` when binning cannot help (too few cells per axis,
    /// or a degenerate cell) and the caller should fall back to all-pairs.
    fn build(positions: &[[f64; 3]], cell: Option<&Lattice>, r_search: f64) -> Option<Self> {
        if r_search <= 0.0 {
            return None;
        }
        let n = positions.len();
        let (coords, dims, periodic) = match cell {
            Some(c) => {
                let hinv = c.inverse()?;
                let widths = c.perp_widths();
                let mut dims = [0usize; 3];
                for k in 0..3 {
                    let d = (widths[k] / r_search).floor() as i64;
                    // A half stencil with fewer than three cells along a
                    // periodic axis would pair an atom with its own image.
                    if c.periodic[k] && d < 3 {
                        return None;
                    }
                    dims[k] = d.max(1) as usize;
                }
                let mut coords = Vec::with_capacity(n);
                for p in positions {
                    let s = [
                        hinv[0][0] * p[0] + hinv[0][1] * p[1] + hinv[0][2] * p[2],
                        hinv[1][0] * p[0] + hinv[1][1] * p[1] + hinv[1][2] * p[2],
                        hinv[2][0] * p[0] + hinv[2][1] * p[1] + hinv[2][2] * p[2],
                    ];
                    let mut idx = [0usize; 3];
                    for k in 0..3 {
                        let f = s[k].rem_euclid(1.0);
                        idx[k] = ((f * dims[k] as f64) as usize).min(dims[k] - 1);
                    }
                    coords.push(idx);
                }
                (coords, dims, c.periodic)
            }
            None => {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for p in positions {
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
                let mut dims = [0usize; 3];
                for k in 0..3 {
                    let extent = (hi[k] - lo[k]).max(0.0);
                    dims[k] = ((extent / r_search).floor() as i64).max(1) as usize;
                }
                let mut coords = Vec::with_capacity(n);
                for p in positions {
                    let mut idx = [0usize; 3];
                    for k in 0..3 {
                        let extent = (hi[k] - lo[k]).max(1e-30);
                        let f = ((p[k] - lo[k]) / extent).clamp(0.0, 1.0);
                        idx[k] = ((f * dims[k] as f64) as usize).min(dims[k] - 1);
                    }
                    coords.push(idx);
                }
                (coords, dims, [false; 3])
            }
        };

        let ncells = dims[0] * dims[1] * dims[2];
        // With only a handful of cells the stencil scan degenerates to
        // all-pairs anyway, and the bookkeeping only costs time.
        if ncells < 8 {
            return None;
        }

        // Counting sort into CSR buckets.
        let mut counts = vec![0usize; ncells + 1];
        let flat: Vec<usize> = coords
            .iter()
            .map(|c| (c[0] * dims[1] + c[1]) * dims[2] + c[2])
            .collect();
        for &c in &flat {
            counts[c + 1] += 1;
        }
        for i in 0..ncells {
            counts[i + 1] += counts[i];
        }
        let starts = counts.clone();
        let mut cursor = counts;
        let mut items = vec![0usize; n];
        for (i, &c) in flat.iter().enumerate() {
            items[cursor[c]] = i;
            cursor[c] += 1;
        }

        Some(Self {
            dims,
            periodic,
            starts,
            items,
        })
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (x * self.dims[1] + y) * self.dims[2] + z
    }

    #[inline]
    fn atoms_in(&self, cell: usize) -> &[usize] {
        &self.items[self.starts[cell]..self.starts[cell + 1]]
    }

    /// The cell reached by stepping `off` from `base`, or `None` when the step
    /// leaves an aperiodic boundary.
    #[inline]
    fn offset_index(&self, base: [i32; 3], off: [i32; 3]) -> Option<usize> {
        let mut idx = [0usize; 3];
        for k in 0..3 {
            let n = self.dims[k] as i32;
            let mut v = base[k] + off[k];
            if self.periodic[k] {
                v = v.rem_euclid(n);
            } else if v < 0 || v >= n {
                return None;
            }
            idx[k] = v as usize;
        }
        Some(self.index(idx[0], idx[1], idx[2]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference pair list: the honest O(N²) double loop.
    fn brute(positions: &[[f64; 3]], cell: Option<&Lattice>, r: f64) -> Vec<(usize, usize)> {
        let r2 = r * r;
        let mut out = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let d = min_image(vec3::sub(positions[i], positions[j]), cell);
                if vec3::norm2(d) <= r2 {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn lattice_positions(n_side: usize, spacing: f64) -> (Vec<[f64; 3]>, Lattice) {
        let mut pos = Vec::new();
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    // Offset each site slightly so the configuration is not
                    // perfectly symmetric (ties on the cutoff are brittle).
                    let jitter = ((i * 7 + j * 13 + k * 29) % 11) as f64 * 0.01;
                    pos.push([
                        i as f64 * spacing + jitter,
                        j as f64 * spacing + jitter * 0.5,
                        k as f64 * spacing + jitter * 0.25,
                    ]);
                }
            }
        }
        let l = n_side as f64 * spacing;
        (pos, Lattice::cubic(l))
    }

    #[test]
    fn cell_list_reproduces_the_brute_force_pair_list_periodic() {
        let (pos, cell) = lattice_positions(6, 2.5);
        let mut nl = NeighborList::new(3.0, 0.5);
        nl.build(&pos, Some(&cell));
        assert!(!nl.used_fallback(), "expected the binned path");

        let mut got = nl.pairs().to_vec();
        got.sort_unstable();
        let mut want = brute(&pos, Some(&cell), nl.search_radius());
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn cell_list_reproduces_the_brute_force_pair_list_aperiodic() {
        let (pos, _) = lattice_positions(6, 2.5);
        let mut nl = NeighborList::new(3.0, 0.5);
        nl.build(&pos, None);
        assert!(!nl.used_fallback());

        let mut got = nl.pairs().to_vec();
        got.sort_unstable();
        let mut want = brute(&pos, None, nl.search_radius());
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn triclinic_cell_reproduces_the_brute_force_pair_list() {
        let (mut pos, _) = lattice_positions(6, 2.5);
        let l = 15.0;
        let cell = Lattice {
            a: [l, 0.0, 0.0],
            b: [3.0, l, 0.0],
            c: [1.0, 2.0, l],
            periodic: [true; 3],
        };
        for p in &mut pos {
            *p = cell.wrap(*p);
        }
        let mut nl = NeighborList::new(3.0, 0.4);
        nl.build(&pos, Some(&cell));

        let mut got = nl.pairs().to_vec();
        got.sort_unstable();
        let mut want = brute(&pos, Some(&cell), nl.search_radius());
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn small_cells_fall_back_to_all_pairs_rather_than_double_counting() {
        // A box only ~2 cells wide at this cutoff: the half stencil would
        // alias an atom with its own image, so the fallback must engage.
        let pos = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let cell = Lattice::cubic(7.0);
        let mut nl = NeighborList::new(3.0, 0.5);
        nl.build(&pos, Some(&cell));
        assert!(nl.used_fallback());
        let mut got = nl.pairs().to_vec();
        got.sort_unstable();
        let mut want = brute(&pos, Some(&cell), nl.search_radius());
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn exclusions_are_omitted_from_the_pair_list() {
        let (pos, cell) = lattice_positions(6, 2.5);
        let mut nl = NeighborList::new(3.0, 0.5);
        nl.build(&pos, Some(&cell));
        let baseline = nl.pairs().len();
        let victim = nl.pairs()[0];

        let mut nl2 = NeighborList::new(3.0, 0.5);
        nl2.exclude(victim.0, victim.1);
        nl2.build(&pos, Some(&cell));
        assert_eq!(nl2.pairs().len(), baseline - 1);
        assert!(!nl2.pairs().contains(&victim));
    }

    #[test]
    fn rebuild_triggers_only_past_half_the_skin() {
        let (mut pos, cell) = lattice_positions(6, 2.5);
        let mut nl = NeighborList::new(3.0, 1.0);
        nl.build(&pos, Some(&cell));
        assert!(!nl.needs_rebuild(&pos, Some(&cell)));

        pos[0][0] += 0.4; // < skin/2
        assert!(!nl.needs_rebuild(&pos, Some(&cell)));

        pos[0][0] += 0.3; // now > skin/2
        assert!(nl.needs_rebuild(&pos, Some(&cell)));
        assert!(nl.maybe_build(&pos, Some(&cell)));
        assert!(!nl.maybe_build(&pos, Some(&cell)));
    }

    /// The headline scaling claim: candidate-distance count per atom must stay
    /// bounded as the system grows at fixed density.
    ///
    /// Counting checks rather than timing keeps this deterministic — an O(N²)
    /// build would show `checks/N` growing linearly with N (it does: the
    /// fallback path is asserted against below).
    #[test]
    fn neighbor_build_is_linear_in_atom_count() {
        let mut per_atom = Vec::new();
        let mut counts = Vec::new();
        for &n_side in &[6usize, 8, 10, 12] {
            let (pos, cell) = lattice_positions(n_side, 2.5);
            let mut nl = NeighborList::new(3.0, 0.5);
            nl.build(&pos, Some(&cell));
            assert!(!nl.used_fallback());
            per_atom.push(nl.checks() as f64 / pos.len() as f64);
            counts.push(pos.len());
        }
        // 6³ = 216 atoms up to 12³ = 1728 atoms, an 8× growth. Under O(N)
        // scaling checks/atom stays bounded by the number of atoms within one
        // stencil of cells; under O(N²) it is (N-1)/2, i.e. 108 → 864 here.
        for (&n, &c) in counts.iter().zip(&per_atom) {
            assert!(c < 100.0, "checks/atom = {c} at N = {n} — not O(N)");
        }
        // Cell widths differ slightly between box sizes (the grid takes
        // floor(width / r_search) cells), so allow modest variation but
        // nothing like the 8× an all-pairs build would show.
        let (first, last) = (per_atom[0], per_atom[per_atom.len() - 1]);
        assert!(
            last / first < 1.5,
            "checks per atom grew with N: {per_atom:?}"
        );
    }
}
