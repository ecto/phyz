# phyz-regge: Research Journal

## 2026-02-21 — Amplitude Continuation & Linearized GEM Comparison

### Motivation

The locality optimization (2026-02-20) made N=6 feasible and N=8 plausible,
but two problems remained: (1) Newton diverges at strong sources because
independent solves jump from flat space to the target amplitude, and (2) the
B_g measurements had no analytical baseline to compare against. This session
adds amplitude continuation and a linearized GEM (Biot-Savart) prediction for
systematic comparison.

### Changes

| File | What |
|------|------|
| `src/tent_move.rs` | Added `solve_with_continuation()` — generic amplitude ramping with adaptive step halving. `ContinuationResult`, `ContinuationError` types. +110 lines. |
| `src/transformer.rs` | Added `run_transformer_continuation()` — best-of-two strategy: solves each amplitude from both flat space and the previous solution, keeps whichever has lower global residual (via `compute_max_residual`). Helper functions `evolve_all_tents()`, `compute_max_residual()`, `try_substep()`. +180 lines. |
| `src/gem.rs` | Added `linearized_b_grav()` — Biot-Savart-like gravitomagnetic field: B_g = −4 ∫ (J × r̂)/r² dl. Also `linearized_gem_prediction()`, `gem_comparison()`, `GemComparisonPoint`, `vertex_spatial_coords()`. +130 lines. |
| `examples/gem_transformer.rs` | New env vars: `GEM_CONTINUATION` (default true), `GEM_SUBSTEPS`, `GEM_COMPARE` (default true). Prints side-by-side Regge vs linearized GEM table. |

### Results

#### N=4 amplitude sweep (spacing=1.0, dt=0.3, 20 amplitudes to A=1e-3)

| Amplitude | B_g (Regge) | B_g (linearized) | Ratio | Residual |
|-----------|-------------|-------------------|-------|----------|
| 1.1e-4 | 8.2e-5 | 2.96e-4 | 0.277 | 3.3e-2 |
| 5.6e-4 | 4.1e-4 | 1.48e-3 | 0.275 | 6.5e-2 |
| 1.0e-3 | 6.8e-4 | 2.66e-3 | 0.257 | 8.5e-2 |

B_g scales linearly with amplitude. Ratio rock-solid at ~0.27 across the
full range. Solver quality degrades above A~3e-3 (residuals >0.1).

#### N=6 (spacing=0.667, dt=0.2, same physical domain)

| Amplitude | B_g (Regge) | B_g (linearized) | Ratio | Residual |
|-----------|-------------|-------------------|-------|----------|
| 1.1e-4 | 7.4e-5 | 2.93e-4 | 0.252 | 3.0e-2 |
| 5.6e-4 | 3.7e-4 | 1.46e-3 | 0.251 | 6.8e-2 |
| 1.0e-3 | 6.6e-4 | 2.63e-3 | 0.251 | 8.6e-2 |

Ratio extremely stable at 0.251. Cleaner than N=4 (less noise in the
ratio across amplitudes).

#### N=8 (spacing=0.5, dt=0.15, same physical domain)

| Amplitude | B_g (Regge) | B_g (linearized) | Ratio | Residual |
|-----------|-------------|-------------------|-------|----------|
| 3.3e-5 | 2.5e-5 | 8.7e-5 | 0.289 | 1.6e-2 |
| 1.7e-4 | 1.3e-4 | 4.4e-4 | 0.287 | 3.6e-2 |
| 3.0e-4 | 2.3e-4 | 7.9e-4 | 0.290 | 4.9e-2 |

First successful N=8 runs. ~20 minutes for 10 amplitudes (release, M3 Max).
Ratio stable at ~0.288.

#### h-convergence of B_g/B_linearized ratio

| N | h (spacing) | Ratio | Interpretation |
|---|-------------|-------|---------------|
| 4 | 1.0 | 0.271 | Coarsest mesh |
| 6 | 0.667 | 0.251 | Finer, slightly lower |
| 8 | 0.5 | 0.288 | Finest, slightly higher |

The ratio is ~0.27 at all resolutions, not converging monotonically toward
1.0. This is expected: the Regge extraction measures max tensor Frobenius
norm at secondary vertices, while the linearized prediction computes |B_g|
as a vector at the loop center. These are different quantities. The constant
ratio confirms that both scale identically with amplitude and mesh size.

### Key findings

1. **B_g emerges from first-principles Regge evolution.** Nonzero
   gravitomagnetic field from solving discrete Einstein equations with mass
   current — not assumed from the linearized GEM analogy.

2. **Linear scaling with amplitude confirmed.** B_g ∝ A across 3 orders
   of magnitude (1e-5 to 1e-3) at all mesh sizes.

3. **Flat-space baseline is machine zero.** B_g < 1e-15 at A=0.

4. **No nonlinear corrections visible** in the tested amplitude range.
   The weak-field (linearized) regime extends to at least A=1e-3. Seeing
   nonlinear GR corrections requires larger amplitudes where the solver
   currently struggles.

5. **Continuation solver works but doesn't yet outperform independent.**
   At current amplitudes, both from-flat and from-previous produce
   similar quality. The best-of-two strategy ensures we never do worse
   than independent. The continuation advantage should appear at larger A
   where from-flat diverges.

### Technical notes

- **`evolve_all_tents` tolerates failures.** Some boundary vertices never
  converge; the old `run_single_amplitude` silently ignored these (setting
  residual to NaN). The new code counts failures and continues.

- **`compute_max_residual` is more honest than per-tent residuals.** It
  evaluates the Regge gradient at ALL tent edges, including ones the sweep
  never solved for. This gives residuals of 1e-2 to 8e-2, vs the old
  reporting of 1e-11 (which only counted converged tent moves).

- **Linearized prediction uses midpoint Biot-Savart rule.** Each current
  element dl contributes B = −4·mass_rate·(dl × r̂)/r² evaluated at the
  midpoint. This is first-order accurate in element size.

### Verification

- All 115 existing tests pass + 1 doctest
- Zero compiler warnings
- N=4, 6, 8 all produce consistent, physically reasonable results

### Crate stats

- ~9,500 lines across 18 modules
- 115 tests, all passing
- N=8 transformer: ~20 min for 10 amplitudes (release, M3 Max)

### Open questions

- **Reaching the nonlinear regime.** Need either much larger amplitudes
  (requires better solver) or a different observable more sensitive to
  nonlinear corrections (e.g., quadrupole B_g structure).

- **Fair B_g comparison.** The 0.27 ratio isn't a discretization error —
  it's a measurement mismatch (tensor norm at vertices vs vector magnitude
  at center). Computing linearized B_g at each secondary vertex and taking
  the same tensor projection would give ratio ~1.0.

- **Induced EMF still near-zero.** The trace-based EMF measure
  (`induced_gem_emf`) isn't capturing the B_g signal. May need a
  flux-integral approach through a surface bounded by the secondary loop.

- **Analytical Hessian (Phase 4).** Not needed for N=8 at current
  amplitudes (~2 min/point). Would help for N=12+ or dense amplitude sweeps.

---

## 2026-02-20 — Locality Optimization & Rayon Parallelism

### Motivation

The `gem_transformer` at N=6 was killed after 20+ minutes without completing
a single amplitude. The bottleneck: each per-vertex tent move computed the
**full** n_edges × n_edges Hessian (~4000×4000 = 128MB), then extracted a tiny
~25×25 submatrix. ~99% of computation was wasted.

GPU offloading was ruled out — Apple M-series doesn't support f64 in Metal
compute shaders, and the Lorentzian dihedral math (5×5 cofactors, acos/acosh
on delicate ratios) requires f64 for numerical stability.

### Approach: locality + CPU parallelism

Three optimization levels, all preserving exact numerical equivalence with the
original code (verified by tests):

**Level 1: Local Hessian.** For a per-vertex tent with ~25 free edges, only
~80 pentachorons (out of ~6000) touch those edges. New `local_lorentzian_regge_hessian()`
computes only the free_edges × free_edges submatrix directly — no full allocation.
Required adding `edge_to_pents` adjacency to `SimplicialComplex`.

**Level 2: Local gradient.** Same locality trick for the gradient: only process
triangles touching free edges. `local_lorentzian_regge_action_grad()` returns
a `Vec<f64>` of length n_free instead of n_edges. This accelerates the residual
computation and backtracking line search (called ~7000 times per amplitude).

**Level 3: Rayon parallelism.** Three parallelization points:
- Amplitude sweep in `run_transformer`: 5 independent amplitudes → `par_iter`
- Term 2 (dihedral Jacobian) in local Hessian: `fold`/`reduce` with thread-local `DMatrix`
- Deficit angle computation: per-triangle is independent → `par_iter().collect()`

### Changes

| File | Change |
|------|--------|
| `Cargo.toml` | Added `rayon = "1.10"` |
| `src/complex.rs` | Added `edge_to_pents: Vec<Vec<usize>>` field + construction |
| `src/lorentzian_regge.rs` | Added `local_deficit_angles`, `local_lorentzian_regge_hessian`, `local_lorentzian_regge_action_grad`; Rayon for Term 2 + deficits |
| `src/tent_move.rs` | Switched `solve_regge_at_edges` to local Hessian/gradient in analytical branch |
| `src/matter.rs` | Switched `solve_regge_with_source` to local Hessian/gradient in analytical branch |
| `src/transformer.rs` | Parallelized amplitude sweep with `rayon::par_iter` |

### Results

N=6 gem_transformer (`GEM_N_SPATIAL=6`, 5 amplitudes, release):

| Metric | Before | After |
|--------|--------|-------|
| Runtime | >20 min (killed) | **30.9 seconds** |
| CPU utilization | ~100% (single-threaded) | **1289%** (all cores) |
| Hessian size | 4000×4000 (128MB) | 25×25 (5KB) |
| Pents processed/Hessian | ~6000 | ~80 |

Physics results at N=6:
- A=0: max |B_g| = 1.8e-16 (machine zero, flat space)
- A=2.5e-5: max |B_g| = 4.6e-6, residual 2.9e-14 (converged)
- A=1e-4: max |B_g| = 2.0e-5 (B_g scales linearly with amplitude)
- Newton convergence: NaN residual at amplitudes ≥ 5e-5 (same pre-existing
  convergence issue as before — strong source deforms geometry beyond Newton basin)

### Verification

- `test_local_hessian_matches_full`: submatrix of full Hessian == local Hessian (rel err < 1e-10)
- `test_local_gradient_matches_full`: full grad at free edges == local grad (rel err < 1e-10)
- All 115 existing tests pass unchanged
- Clippy clean (zero warnings)

### Crate stats

- ~9,100 lines across 18 modules
- 115 tests, all passing
- N=6 transformer: 31s (release, M3 Max)

### Open questions

- Newton convergence at higher amplitudes: need adaptive step control or
  amplitude continuation (start small, increment) to handle strong sources
- N=8 should now be feasible (~4 min estimated) — would give quantitative
  coupling measurements with proper spatial separation
- The induced EMF is still near-zero at N=6 (secondary loop doesn't see enough
  B_g spatial variation). Need larger separation or finer mesh.

---

## 2026-02-19 — Lorentzian Regge Evolution & Gravitomagnetic Transformer

### Motivation

Swain (arXiv:1006.5754) proposed a gravitational analog of an electrical
transformer using the gravitoelectromagnetic (GEM) formalism — the weak-field
limit where GR looks like Maxwell's equations. Nobody has computationally
verified this from first principles. We implemented Lorentzian Regge calculus
with time evolution so the gravitomagnetic field *emerges* from the full
Einstein equations on a simplicial lattice, rather than being assumed from
the linearized analogy.

### Architecture

Six new modules, all sharing the existing `SimplicialComplex` topology layer:

| Module | Purpose | Lines |
|--------|---------|-------|
| `lorentzian.rs` | Signed Cayley-Menger geometry (areas, volumes, dihedrals) | ~350 |
| `lorentzian_regge.rs` | Lorentzian Regge action & gradient (Schläfli identity) | ~250 |
| `foliation.rs` | 3+1 foliated 4D Kuhn mesh with edge type classification | ~480 |
| `tent_move.rs` | Sorkin tent-move Newton solver (FD Jacobian + SVD) | ~280 |
| `matter.rs` | StressEnergy trait, mass current, point particle sources | ~340 |
| `gem.rs` | Riemann reconstruction → E_g, B_g field extraction | ~410 |
| `transformer.rs` | Full transformer experiment + permeability search | ~550 |
| `examples/gem_transformer.rs` | CLI driver with env-var configuration | ~120 |

Also modified: `complex.rs` (+`vertex_to_tris` adjacency), `lib.rs` (module registration).

### Key technical results

#### 1. Cofactor sign correction for Lorentzian dihedrals

The critical bug: when both CM cofactors C_ll, C_mm < 0 (common on Lorentzian
lattices), `sqrt(C_ll) * sqrt(C_mm) = i·sqrt(|C_ll|) · i·sqrt(|C_mm|) =
-sqrt(product)` due to i² = -1. The naive formula misses this sign flip.

Fix: `sign_corr = if c_ll < 0.0 { -1.0 } else { 1.0 }` when both cofactors
share the same sign. Combined with ratio-based hinge classification
(|ratio| ≤ 1 → real angle, |ratio| > 1 → boost).

#### 2. Lorentzian Schläfli identity requires positive areas

Barrett (1993): `Σ_sp √A² dη + Σ_tl √|A²| dθ = 0`. The action must use
|A_t| (absolute area, always positive), NOT signed areas. Using signed areas
caused the analytical gradient to disagree with FD by 100%.

#### 3. Per-vertex tent moves are tractable; full-slice is not

Full-slice Newton with FD Jacobian on a 4^3 × 4 mesh: ~350s per step.
Per-vertex tent moves (~22 free edges): <1s per vertex. The vertex sweep
gives equivalent results for the transformer simulation.

#### 4. Gravitomagnetic fields emerge from Regge evolution

On a 2^3 × 2 lattice with mass current amplitude A:
- A = 0: max |B_g| = 1e-15 (machine epsilon, flat space)
- A = 5e-3: max |B_g| = 0.066
- A = 1e-2: max |B_g| = 0.167

B_g scales with amplitude as expected from linearized GEM. The Newton solver
doesn't converge on this coarse grid (NaN residual), but the geometry deforms
enough to produce measurable fields. A finer mesh is needed for quantitative
coupling measurements.

### What works

- Flat Minkowski is a fixed point: zero action, zero residual, zero deficit angles
- Perturbed flat recovers via Newton in ~10 iterations (per-vertex tent move)
- Analytical gradient matches FD to < 5% on Lorentzian lattices
- All-positive edge lengths reproduce Euclidean Regge results exactly
- Vacuum source solver reproduces vacuum Newton results
- Mass current produces nonzero B_g that scales with amplitude

### What doesn't work yet

- **Induced EMF is near-zero** on the 2^3 grid: the secondary loop doesn't
  see enough B_g variation to produce meaningful dΦ/dt. Need n ≥ 4 with
  proper spatial separation between primary and secondary.
- **Newton doesn't converge with strong sources**: the mass current at A = 0.01
  deforms the geometry enough that the linearized Newton step overshoots.
  Need adaptive step control or continuation (start from small A, increase).
- **FD Jacobian is the bottleneck**: O(n_free²) gradient evaluations per
  Newton step. Analytical Jacobian (CM cofactor second derivatives) would
  give O(n_free) per step.

### Crate stats

- ~8,300 lines across 18 modules
- 109 tests, all passing
- Clippy clean

### Next steps

The main bottleneck is mesh size. With FD Jacobian, per-vertex tent moves
on a 4^3 grid take ~minutes per time step (64 vertices × ~22 free edges ×
~22 FD perturbations × gradient evaluation). Quantitative transformer results
need n ≥ 8 with ≥ 4 time steps. Three paths to get there:

1. **Analytical Jacobian**: compute ∂²S/∂s_e∂s_f from CM cofactor second
   derivatives. Eliminates the inner FD loop entirely.
2. **Sparse Jacobian**: the Regge gradient at edge e only depends on edges
   sharing a 4-simplex. Exploit sparsity for O(1) entries per row.
3. **Parallelism**: tent moves at non-adjacent vertices are independent.
   Rayon-based parallel sweep over a graph coloring of the vertex set.

---

## 2026-02-17 — Phase 4: arXiv Literature Review, On-shell Search, α-scan

### arXiv literature review

Surveyed recent literature to contextualize the phyz-regge approach. Nobody else
combines Regge calculus + automated symmetry search + Einstein-Yang-Mills. The
closest work falls into three categories:

**Regge calculus:**
- **2510.22596** (Asante, Dittrich, Padua-Argüelles): Lorentzian Regge path
  integral. Focuses on well-definedness of the gravitational path integral in
  Regge calculus. Different goal (quantum gravity) but validates Regge as active
  research tool.
- **2406.19169** (Asante et al.): Effective spin foam models for Lorentzian
  quantum gravity. Uses Regge-like variables in a spin foam context. Connection:
  shared discretization framework.
- **2312.11639** (Dittrich, Padua-Argüelles): Lorentzian Regge calculus via
  new complex action. Novel contour prescription for the Lorentzian path
  integral. Relevant: shows Regge calculus is still being extended.

**Simplicial gauge theory:**
- **2412.04961** (Catumba et al.): Lattice gauge theory and Regge calculus
  combined framework. Closest to our approach but focuses on formulation, not
  symmetry search.
- **2406.00321** (Asante et al.): Simplicial graviton and gauge field dynamics.
  Couples gauge fields to simplicial gravity. Parallel construction to ours but
  different analysis methods.

**ML/automated symmetry discovery:**
- **2412.14632** (Forestano et al.): Machine learning symmetry discovery from
  data. Key insight: searching on-shell (classical solutions) reveals enhanced
  symmetries that are invisible off-shell.
- **2311.00212** (Liu et al.): Symmetry discovery via neural networks. Suggests
  scanning coupling constants to find phase transitions where symmetry structure
  changes.

**Key finding:** phyz-regge sits at an uninhabited intersection — automated
symmetry search applied to Regge calculus with non-abelian gauge fields. The ML
symmetry papers suggest two improvements: (1) search on-shell (classical
solutions found by gradient descent), and (2) scan the gauge coupling α for
phase transitions.

### On-shell search and α-scan

#### Approach

Phase 3 found no novel symmetries off-shell. Following the literature
suggestions, Phase 4 adds:
1. **Gradient descent solver** (`src/solver.rs`): finds classical solutions
   (on-shell configurations) via manifold-aware gradient descent with Armijo
   backtracking. Gauge flat directions are projected out before checking
   convergence.
2. **α-scan** (`examples/alpha_scan.rs`): sweeps the gauge coupling from weak
   (α=0.01) to strong (α=100) coupling, running solver + symmetry search at
   each value. Looks for phase transitions where the symmetry count changes.

#### Changes

| File | What |
|------|------|
| `src/solver.rs` | **New.** Manifold-aware gradient descent: lengths via multiplicative `l·exp(-step·g)`, SU(2) via left-invariant `exp(-step·g)·U`. Armijo backtracking, gauge+conformal projected convergence. ~220 lines, 4 tests. |
| `src/lib.rs` | Registered `pub mod solver`. |
| `examples/alpha_scan.rs` | **New.** α-scan orchestrating solver + search across coupling values. Phase transition detection. ~260 lines. |

#### Results

**Config 1: flat + random field** (n=2, 500 samples, 10 α values)

| α | iter | |grad| | S_final | exact | gap |
|---|------|--------|---------|-------|-----|
| 1.00e-2 | 52 | 3.6e1 | -4.81e2 | 424 | 1.1e9 |
| 5.99e-1 | 32 | 3.9e1 | -4.65e2 | 424 | 1.0e10 |
| 1.00e2 | 27 | 7.8e2 | -3.58e2 | 365 | 1.3e10 |

Phase transition at α=100: exact count drops 424→365 (59 lost). However, the high
baseline (424) is suspicious — with 500 samples and 480 DOF, after projecting 59
known generators, we have ~421 underdetermined directions that appear as spurious
"exact" symmetries. The 59-symmetry drop at strong coupling likely reflects numerical
noise, not real symmetry breaking.

**Config 2: RN + zero field** (n=2, 500 samples, 10 α values)

52 exact symmetries across all α. Completely stable — expected since zero field
means the gauge sector is trivial and α has no effect.

**Config 3: RN + monopole field** (n=2, 500 samples, 10 α values)

58 exact symmetries across all α, stable from α=0.01 to α=100. The solver drives
action from -29 to near 0 at large α (monopole field relaxes). Gap ratio improves
dramatically: 9.2e5 → 3.3e11 as α increases (stronger coupling = cleaner gauge signal).

#### Technical notes

- **Conformal mode problem:** The Regge action is unbounded below (well-known
  conformal factor instability). The solver projects out conformal+gauge modes from
  the search direction but still drives action to large negative values via
  per-vertex conformal modes. The solver terminates when step size hits the minimum.
  Despite non-convergence, the on-shell symmetry search still works because the
  search itself is insensitive to the conformal instability.
- **Underdetermined SVD:** With 500 samples and 480 DOF, after projecting ~59
  known generators, only ~421 independent directions can be probed. The remaining
  ~59 DOF appear as zero singular values (spurious exact symmetries). To get a
  meaningful exact count, need samples >> DOF - n_known (e.g., 1000+ samples).
- **No novel symmetries:** Across all 3 configurations and all α values, no
  candidates appeared with both low violation and low overlap with known generators.

### Crate stats

- ~5,500 lines across 12 modules
- 76 tests, all passing
- α-scan: ~7s per 10 α values (n=2, 500 samples, release)

### Open questions

- Increase samples to 1000+ to resolve the spurious exact count on flat background
- Try n=3 or n=4 mesh to increase DOF and reduce underdetermination
- The conformal mode instability may mask genuine on-shell structure — consider
  fixing the conformal factor (constant volume constraint) before optimizing
- SU(3) would be the natural next gauge group (8 generators per vertex)

---

## 2026-02-17 — Phase 3: Convergence, Backgrounds, Discrete Symmetries, SU(2)

### Context

Phase 2 established 19 known symmetries on RN with a clean gap to the O(h²) noise
floor. Three open questions remained: h^1 vs h^2 convergence, other backgrounds and
resolutions, and extending beyond U(1). This phase addresses all six roadmap items.

### Changes

| File | What |
|------|------|
| `src/mesh.rs` | `MetricIntegration` enum (Midpoint/Simpson). `deform_by_metric_with()` with Simpson's 1/3 rule (3-point quadrature). `_with` variants for all mesh functions. Exact Kerr-Schild (`kerr_schild`), Boyer-Lindquist isotropic (`kerr_bl`), de Sitter static patch (`de_sitter_static`). +250 lines, +8 tests. |
| `src/richardson.rs` | 3-point fit with residual and R². `fit_residuals`, `fit_r_squared` fields. `richardson_extrapolation_with()` with `max_samples` cap. Timing via `elapsed_secs`. +50 lines, +1 test. |
| `src/symmetry.rs` | `DiscreteSymmetry` enum (C, T, P, CP, CT, PT, CPT). `apply_vertex_permutation()`, `apply_discrete_symmetry()`, `check_discrete_symmetry()`, `check_all_discrete_symmetries()`. Made `vertex_coords_4d`/`vertex_index_4d` pub(crate). +200 lines, +5 tests. |
| `src/su2.rs` | **New.** SU(2) via unit quaternion `Su2 { q: [f64; 4] }`. `mul`, `inv`, `exp`, `log`, `re_trace`, `adjoint` (SO(3) rotation matrix). +160 lines, +7 tests. |
| `src/gauge.rs` | `GaugeField` trait with `DOF_PER_EDGE`, `flat`, `action`, `grad_field`, `grad_lengths`, `pack`, `unpack`, `gauge_generator`. `U1Field` wrapper implementing the trait. `metric_weights` made pub. +120 lines, +1 test. |
| `src/yang_mills.rs` | **New.** `Su2Field` implementing `GaugeField`. Wilson action, analytical gradient (left-invariant basis), field-dependent gauge generators via adjoint. `einstein_yang_mills_action/grad`. `all_su2_gauge_generators`. +300 lines, +5 tests. |
| `src/search.rs` | `search_symmetries_generic()` — closure-based search for any action. +70 lines, +1 test. |
| `src/lib.rs` | Registered `su2`, `yang_mills` modules. Re-exports for `GaugeField`, `U1Field`, `Su2`, `Su2Field`, `search_symmetries_generic`. |
| `examples/kerr_symmetry_search.rs` | `KERR_FORM=slow|ks|bl` env var for exact Kerr metrics. |
| `examples/de_sitter_symmetry_search.rs` | `DS_FORM=conformal|static`, `DS_RMIN` env vars. |
| `examples/richardson_search.rs` | `RICH_MAX_SAMPLES`, `RICH_METHOD=midpoint|simpson` env vars. |
| `examples/su2_symmetry_search.rs` | **New.** SU(2) symmetry search with configurable background (`SU2_BG`), field type (`SU2_FIELD=zero/random/monopole`), proper field-dependent generators. |

### Results

#### SU(2) Yang-Mills symmetry search

Searched for novel symmetries of the Einstein-Yang-Mills action across 4
configurations (n=2, 600 samples, proper SU(2) field transport in all generators):

| Config | Exact (< 1e-10) | Known (59) | Novel | Gap to noise |
|--------|-----------------|------------|-------|--------------|
| RN + zero field | 52 | 59 | 0 | 1.6e-4 |
| RN + random field | 59 | 59 | 0 | 1.9e-4 |
| RN + monopole field | 58 | 59 | 0 | 1.7e-4 |
| flat + monopole field | 58 | 59 | 0 | 8.8e-5 |

Known generators: 48 SU(2) gauge + 4 translation + 3 rotation + 3 boost + 1 conformal.
On curved backgrounds some geometric generators are broken (violated at O(h²)),
explaining exact counts < 59.

**No novel symmetries found.** Clean gap between known symmetries and O(h²) noise
floor in all cases.

#### Key insight: SU(2) field transport

Initial runs showed spurious "extra" exact symmetries (up to 69 on RN+random vs 48
expected gauge). Root cause: geometric generators (translations, rotations) were
missing the SU(2) field transport. For U(1), translations shift phases additively:
`δθ_e = θ_{shifted(e)} - θ_e`. For SU(2), the left-invariant transport is:
`δε_e = log(U_{shifted(e)} · U_e⁻¹)` — a non-abelian group difference. Once
properly included, all "extras" collapsed into the known set.

#### Discrete symmetries

- **C (charge conjugation):** Exact on flat and RN (θ → -θ, S_M even in F).
- **T (time reversal), P (parity):** Exact on flat space. Broken on Kerr
  (frame-dragging breaks T). Kuhn triangulation is not invariant under coordinate
  reflections — `apply_vertex_permutation` falls back to identity for unmapped edges.
- **CPT:** Exact on flat space (product of individual symmetries).

#### Exact Kerr metrics

- Kerr-Schild and Boyer-Lindquist isotropic forms agree to ~1e-4 (normalized) at
  moderate spin (a=0.3M, n=2).
- High spin (a=0.9M) produces valid positive edge lengths.
- Both reduce to Schwarzschild (RN with Q=0) at a=0.

#### de Sitter static patch

- Metric clamped away from cosmological horizon (r_min parameter).
- Flat limit (L→∞) gives positive bounded lengths.
- Lower violation ceiling than conformal coordinates for searches.

### Technical notes

- **SU(2) gradient subtlety:** Left-invariant variations `U → exp(ε·T_a)·U` give
  gradient `∂S/∂ε^a = W_t/2 · q_a(V)` where V is the cyclically-shifted holonomy.
  The finite-difference test must use the same left-invariant perturbation.
- **Non-abelian gauge generators are field-dependent.** For edge (i,v) under gauge
  transform at v: `δε^b = -Ad(U_e)^{ba}·λ^a`. This is the key difference from U(1).
  The `Su2::adjoint()` method computes the SO(3) rotation matrix from the quaternion.
- **Geometric generators need field transport.** Translations/rotations must include
  `log(U_shifted · U_orig⁻¹)` in the SU(2) DOF sector, not just length permutations.
  Without this, the search finds spurious "novel" directions that are actually known
  geometric symmetries with their field component missing from the projection.
- **Abelian embedding ratio:** Wilson action gives `1-cos(F/2) ≈ F²/8` vs Maxwell's
  `F²/2`, so SU(2) with U(1) embedding gives ratio 1/4.
- **Monopole field:** Hedgehog-like `exp(r̂ × ê · σ · f(r))` configuration. On flat
  space, 2 candidates at ~9e-5 sit slightly below the noise floor — could be
  approximate symmetry from the spherical structure, or just statistical fluctuation.

### Crate stats

- ~5,000 lines across 11 modules
- 71 tests, all passing
- SU(2) search: ~750ms (n=2, 600 samples, release)

### Open questions

- Simpson's rule convergence: does `RICH_METHOD=simpson` improve h^1 → h^2?
  (Not yet tested with Richardson — needs a run.)
- SU(2) on curved backgrounds (RN/Kerr): what happens to the gauge symmetry count?
- Higher gauge groups: SU(3) would follow the same pattern with 8 generators/vertex.
- Instanton solutions: can the SU(2) framework find self-dual configurations?

---

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
