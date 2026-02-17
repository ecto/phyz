# phyz-regge: Research Journal

## 2026-02-16 — Phase 2: Boosts, new backgrounds, Richardson extrapolation

### Context

Phase 1 found 19 known symmetries on RN with a clean gap to the O(h²) noise
floor at ~2.6e-4 (n=2). No novel symmetries. Three questions remained: do
Lorentz boosts help, do other backgrounds reveal different structure, and can
Richardson extrapolation separate real symmetries from discretization artifacts?

### Changes

| File | What |
|------|------|
| `src/symmetry.rs` | Relaxed `rotation_generator` axis asserts from 1..=3 to 0..=3. Added `boost_generator()`, `all_boost_generators()` (3 generators: tx, ty, tz). +40 lines, +2 tests. |
| `src/mesh.rs` | Extracted `metric_line_element()` for full 4×4 metric tensors and `deform_by_metric()` shared helper. Refactored `reissner_nordstrom()` to use it. Added `de_sitter()` (conformally flat cosmological coords) and `kerr()` (slow-rotation approximation with frame dragging). +130 lines, +4 tests. |
| `src/richardson.rs` | **New module.** `richardson_extrapolation()` runs search at multiple resolutions, fits σ_k(h) = a + b·h^p, reports extrapolated violations and convergence orders. Auto-scales sample count. |
| `src/lib.rs` | Registered `pub mod richardson`. |
| `examples/rn_symmetry_search.rs` | Wired boost generators (now 27 known). |
| `examples/de_sitter_symmetry_search.rs` | **New.** Env vars `DS_N`, `DS_L`, etc. |
| `examples/kerr_symmetry_search.rs` | **New.** Env vars `KERR_N`, `KERR_SPIN`, etc. |
| `examples/richardson_search.rs` | **New.** Configurable `RICH_NS=2,3`, `RICH_BG=rn/ds/kerr`. |

### Results

#### RN with boosts (n=2, M=0.1, 27 known)

| Violation range | Count | Interpretation |
|----------------|-------|---------------|
| 0 - 1e-14 | 22 | Known symmetries (15 gauge + 4 translation + 3 boost/conformal) |
| 1e-14 - 5e-4 | 0 | Clean gap |
| 5e-4 - 1e-2 | ~120 | O(h²) discretization noise |

Boosts are exact on flat space (δl=0) but broken on RN (f≠g), as expected.
Boost violations on RN exceed rotation violations, confirming time-space
anisotropy is larger than the lattice discretization error.

#### de Sitter (n=2, L=10)

| Violation range | Count | Interpretation |
|----------------|-------|---------------|
| 0 | 120 | All gauge pairs — conformally flat means all vertices equivalent |
| ~1.5e-12 | 5 | Boosts + translations + conformal (near-exact due to f=g) |
| 1e-2 - 1e2 | ~115 | Discretization noise (stronger curvature → higher floor) |

Key result: **boosts and rotations have comparable violation** since the
de Sitter metric is isotropic (f=g). This validates the boost generators.

#### Kerr (n=2, M=0.1, a=0.3)

25 exact symmetries, noise floor ~6e-4. Very similar to RN — the frame-dragging
off-diagonal terms are small at this a/M ratio. Would need larger spin or finer
mesh to see clear breaking of non-z rotations.

#### Richardson extrapolation (RN, n=2→3, 71s release)

| Spectral indices | Convergence | Extrapolated | Interpretation |
|-----------------|-------------|-------------|----------------|
| k=0-1 | h^60 (vanish at n=3) | ~0 (negative) | Pure discretization artifacts |
| k=2-78 | h^1 | ~5e-4 to ~1e-4 (positive) | Real violations, slowly converging |
| k=79+ | h^2+ | Negative | Discretization artifacts |

**No candidates with extrapolated violation < 1e-8.** The Richardson method
successfully separates artifact modes from genuine symmetry-breaking, but
confirms no hidden symmetries on RN at this resolution.

### Crate stats

- ~3,400 lines across 9 modules
- 43 tests, all passing
- RN search: ~590ms (n=2, release), Richardson n=2+3: ~71s (release)

### Open questions

- The h^1 convergence of the middle spectrum (k=2-78) is slower than the
  expected h^2 from Regge calculus. Is this from the midpoint metric
  approximation in `deform_by_metric`, or a fundamental limit?
- Kerr with larger spin (a~0.9M) — does z-rotation clearly separate from
  the broken xy/xz rotations?
- n=4 Richardson is feasible (~30min release) — would a third point improve
  the convergence order estimates?
- de Sitter has much higher violation ceiling than RN (~1e2 vs ~3e1) due to
  the conformal factor. Is this an artifact of the coordinate choice?

---

## 2026-02-16 — SO(3) rotation generators and search calibration

### Context

The symmetry search runs on a discretized Reissner-Nordstrom black hole
background, looking for novel symmetries of the Einstein-Maxwell action.
Prior to this session, the search found no novel candidates — best violations
were ~0.18 (n=3) and ~1.6e-4 (n=2). Three compounding issues:

1. **SO(3) rotations missing from known generators** — the RN background is
   spherically symmetric, so 3 rotation generators are exact symmetries, but
   the search didn't know about them and wasted capacity rediscovering them.
2. **Samples << DOF** — 200 samples with 2430 DOF (n=3) meant the SVD was
   massively underdetermined; most "candidates" were rank-deficient artifacts.
3. **No feedback** when sample count was too low for the DOF.

### Changes

| File | What |
|------|------|
| `src/symmetry.rs` | Added `rotation_generator()` — 90-degree lattice permutation in spatial (xy, xz, yz) planes. Same approach as translation generators. Added `all_rotation_generators()` convenience fn. +144 lines. |
| `examples/rn_symmetry_search.rs` | Wired rotation generators into known set (now 24: 16 gauge + 4 translation + 3 rotation + 1 conformal). Changed defaults to n=2 / 500 samples. Added stderr warning when samples < DOF - n_known. |

### Results

With n=2 (240 DOF), 500 samples, 24 known generators:

| Violation range | Count | Interpretation |
|----------------|-------|---------------|
| 0 - 1e-14 | 19 | Known symmetries (gauge + translation + rotation) |
| 1e-14 - 2.6e-4 | 0 | **Clean gap — no novel symmetries** |
| 2.6e-4 - 1e-2 | ~120 | O(h^2) discretization noise |
| > 1e-2 | ~100 | Non-symmetry directions |

The clean gap is the key result. The search is now correctly configured.
No novel continuous symmetries at this lattice resolution.

### Crate stats

- 2,943 lines across 8 modules
- 37 tests, all passing
- Search runs in ~21s (n=2, 500 samples, debug build)

### Open questions

- Can Richardson extrapolation (n=2,3,4) push the noise floor low enough to
  reveal symmetries that are currently hidden?
- Are Lorentz boost generators worth adding? (broken on RN, useful on flat)
- What's the practical limit on mesh size before runtime becomes prohibitive?
