# phyz-regge: Research Journal

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
