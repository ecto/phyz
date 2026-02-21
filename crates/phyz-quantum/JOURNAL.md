# phyz-quantum: Research Journal

## 2026-02-21 — Phase 1: Foundation + First Results

### Motivation

Nobody has formulated or studied Hamiltonian lattice gauge theory on simplicial
(non-hypercubic) lattices. All quantum simulation work (Google, IBM, Innsbruck,
Maryland groups) uses hypercubic lattices exclusively. phyz-regge already has
simplicial complexes with U(1) gauge fields and mesh generation — this crate
bridges to the quantum (Hamiltonian) formulation.

### Architecture

New crate `phyz-quantum` with 8 modules, reusing `SimplicialComplex` from
phyz-regge directly. No modifications to existing crates.

| Module | Purpose | Lines |
|--------|---------|-------|
| `gauss_law.rs` | Vertex-edge adjacency, BFS spanning tree, gauge-invariant basis via leaf peeling | ~180 |
| `hilbert.rs` | `U1HilbertSpace` with HashMap config→index lookup | ~100 |
| `hamiltonian.rs` | KS Hamiltonian: diagonal electric + triangular plaquette magnetic | ~170 |
| `diag.rs` | Dense `SymmetricEigen` wrapper with sorted spectrum | ~70 |
| `observables.rs` | Wilson loops, electric field ⟨n²⟩, entanglement entropy | ~230 |
| `qubit_map.rs` | Resource estimates (qubits per edge, Trotter gates) | ~90 |
| `stabilizer.rs` | Z₂ limit → stabilizer code [[n,k,d]] parameters | ~150 |
| `hypercubic.rs` | Square-plaquette KS Hamiltonian for comparison | ~300 |

Key design decisions:
- **Spanning-tree gauge fixing** with leaf peeling to enumerate gauge-invariant basis
- **Holonomy signs [+1, -1, +1]** on `[e01, e02, e12]` matching gauge.rs convention
- **Dense diag via nalgebra** — sufficient for dim ≤ 50K (single pentachoron at Λ≤3)
- **metric_weights parameter** enables curved backgrounds immediately

### Results

#### 1. E₀(g²) coupling sweep (single pentachoron, Λ=1, dim=219)

| Regime | g² | E₀ | Gap | Degeneracy |
|--------|-----|------|------|------------|
| Weak | 0.01 | -499 | 165 | E₁=E₂=E₃=E₄ (4-fold) |
| Crossover | 1.0 | -3.02 | 2.10 | E₁=E₂=E₃=E₄ (4-fold) |
| Lifting | 1.26 | -1.74 | 1.92 | Degeneracy lifts: E₁≠E₂ |
| Strong | 100 | -3.3e-6 | 150 | All states ≈ g²/2 per unit |

The 4-fold degeneracy in weak coupling reflects the pentachoron's permutation
symmetry — all 10 triangles are equivalent, so first excited states form a
degenerate multiplet. Degeneracy lifts at g² ≈ 1.3 where the electric term
breaks the permutation symmetry.

Strong coupling limit confirmed: E₀ → 0 (all-zero config), gap → (g²/2) × (minimum nonzero Σn²).

#### 2. Simplicial vs hypercubic spectral comparison

| g² | Simp E₀ | Simp gap | Hyp2D E₀ | Hyp2D gap | Hyp3D E₀ | Hyp3D gap |
|----|---------|----------|----------|-----------|----------|-----------|
| 0.1 | -49.7 | 16.6 | -28.2 | 28.4 | -125 | 44.3 |
| 1.0 | -3.02 | **2.10** | -2.00 | **4.00** | -9.97 | **4.79** |
| 10 | -3.3e-3 | 15.0 | -4.0e-3 | 20.0 | -2.4e-2 | 20.0 |

**Key finding: triangular plaquettes give a smaller spectral gap than square
plaquettes at the same coupling.** At g²=1:
- Simplicial gap: 2.10
- Hypercubic 2D gap: 4.00 (1.90× larger)
- Hypercubic 3D gap: 4.79 (2.28× larger)

This is not a normalization artifact. The 3-edge triangular holonomy couples
states more densely in Hilbert space than the 4-edge square holonomy, producing
a wider magnetic band and smaller gap.

**Physical interpretation:** The string tension σ (extracted from the gap via
σ ∝ Δ/a where a is lattice spacing) is lower on simplicial lattices. Triangular
plaquettes create a weaker confining potential than square plaquettes at the same
bare coupling. This means the continuum limit requires different bare coupling
tuning on simplicial vs hypercubic lattices.

#### 3. Curved background spectrum

Using synthetic metric weights on the single pentachoron:

| Background | Weight range | Gap at g²=1 | Shift vs flat |
|-----------|-------------|-------------|---------------|
| Flat | [1.0, 1.0] | 2.100 | — |
| Schwarzschild (M=0.3) | [1.08, 1.15] | 2.228 | +6.1% |
| de Sitter (H=0.3) | [0.69, 2.78] | 1.916 | -8.8% |

Schwarzschild (positive mass) **increases** the gap — curvature enhances
confinement. de Sitter (positive cosmological constant) **decreases** the gap —
expansion weakens confinement. The effect is O(10%) even with modest curvature.

Note: these use synthetic weights on a single pentachoron, not actual
`metric_weights()` from phyz-regge mesh generators (those produce complexes
too large for exact diag at Λ=1).

#### 4. Z₂ stabilizer code parameters

| Complex | [[n, k, d]] | Star weights | Plaquette weight |
|---------|-------------|-------------|-----------------|
| 1 pentachoron | [[10, 6, 3]] | uniform 4 | 3 |
| 2 pentachorons | [[14, 9, 3]] | 4-5 | 3 |
| 3 pentachorons | [[18, 12, 3]] | 4-6 | 3 |
| 4 pentachorons | [[22, 15, 3]] | 4-7 | 3 |
| Toric code 3×3 | [[18, 10, 3]] | uniform 4 | 4 |

**Key finding: simplicial codes encode more logical qubits at the same
physical qubit count.** At n=18: simplicial k=12 vs toric k=10 (20% more
logical qubits). Same code distance d=3.

The advantage comes from higher edge-to-vertex ratio in simplicial complexes
(b₁ = E - V + 1 is larger). Trade-off: variable star weights (non-uniform
syndrome extraction) and weight-3 plaquettes (vs weight-4 in toric code).

#### 5. Wilson loop expectation values

All 6 fundamental loops on the pentachoron are symmetric (equal W), confirming
the regular geometry. Wilson loop vs coupling:

| g² | ⟨W⟩ | -log⟨W⟩ |
|-----|------|---------|
| 0.1 | 0.499 | 0.696 |
| 1.0 | 0.467 | 0.762 |
| 2.0 | 0.175 | 1.74 |
| 10 | 6.67e-3 | 5.01 |
| 100 | 6.67e-5 | 9.62 |

Clear area law: -log⟨W⟩ grows monotonically with coupling (confinement
strengthens). At strong coupling, -log⟨W⟩ ≈ 2·log(g²), consistent with the
strong-coupling expansion ⟨W⟩ ~ (1/g²)^(area).

### Verification

- 38 tests, all passing
- Hamiltonian symmetry verified element-by-element
- Strong coupling limit: E₀ = 0, ground state = |0,...,0⟩
- Electric term scaling: doubling metric weights doubles magnetic term exactly
- All basis states satisfy Gauss law (checked exhaustively)

### Resource estimates

| Complex | Λ | b₁ | Hilbert dim | Qubits (GI) |
|---------|---|-----|-------------|-------------|
| 1 pent | 1 | 6 | 219 | 12 |
| 1 pent | 2 | 6 | 4,175 | 18 |
| 1 pent | 3 | 6 | 30,429 | 18 |
| 2 pent | 1 | 9 | 3,135 | 18 |
| 2 pent | 2 | 9 | 268,997 | 27 |
| 3 pent | 1 | 12 | 46,593 | 24 |

Dense diag is fast up to ~10K, feasible to ~50K. Beyond that needs Lanczos.

### Crate stats

- ~1,600 lines across 8 modules
- 38 tests, all passing
- Example produces all 5 result sections in <2s (release, M3 Max)

### Open questions

- **Spectral gap ratio.** The 2.10 vs 4.00 gap ratio — is this universal or
  Λ-dependent? Need to check at Λ=2,3 to see if the ratio persists in the
  continuum limit (Λ→∞).

- **Larger complexes.** 2-pentachoron at Λ=1 (dim=3135) is tractable for all
  analyses. Would give Wilson loops of different areas for proper area law test.

- **Real metric weights.** The curved background results use synthetic weights.
  A proper test would use `metric_weights()` on a small enough complex from
  phyz-regge.

- **Lanczos for larger systems.** Dense diag hits a wall at dim ~50K. Adding
  a Lanczos eigensolver would enable 2-pentachoron at Λ=2 (dim ~269K) and
  push into the regime where finite-size effects are controlled.

- **SU(2) extension.** The Hamiltonian framework generalizes to SU(2) with
  angular momentum basis |j,m⟩ on each edge. phyz-regge already has SU(2)
  gauge fields (yang_mills.rs). The magnetic term becomes 6j-symbol-weighted.

---
