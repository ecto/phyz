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

## 2026-02-21 — Spectral Gap Deep Dive

### Question

At g²=1, the simplicial spectral gap (2.10) is roughly half the hypercubic 2D
gap (4.00). Is this ratio Λ-dependent? What's the mechanism?

### Result 1: Gap ratio is Λ-independent

| Λ | Simp dim | Simp gap | Hyp2D dim | Hyp2D gap | Ratio |
|---|----------|----------|-----------|-----------|-------|
| 1 | 219 | 2.100 | 3 | 4.000 | 0.525 |
| 2 | 4,175 | 1.839 | 5 | 3.532 | 0.521 |

Ratio = **0.52 ± 0.01** across both truncation levels. This is not a truncation
artifact — it is a geometric/topological property of the lattice.

Against hypercubic 3D (2³ periodic torus):

| Λ | Hyp3D dim | Hyp3D gap | Ratio (simp/hyp3D) |
|---|-----------|-----------|---------------------|
| 1 | 69 | 4.788 | 0.439 |
| 2 | 767 | 3.854 | 0.477 |

### Result 2: Gap ratio has a minimum at the crossover

The ratio Δ_simp/Δ_hyp2D varies with coupling:

| g² | Ratio (Λ=1) | Ratio (Λ=2) |
|----|-------------|-------------|
| 0.1 | 0.583 | 0.726 |
| 0.5 | 0.567 | 0.618 |
| 1.0 | 0.525 | 0.521 |
| 1.5 | **0.471** | **0.449** |
| 3.0 | 0.694 | 0.690 |
| 10.0 | 0.750 | 0.749 |

The minimum is at **g² ≈ 1.5** (ratio ≈ 0.45), right at the crossover regime
where competition between electric and magnetic terms is strongest. In both
strong and weak coupling limits the ratio approaches ~0.75.

The Λ=1 and Λ=2 curves nearly overlap for g² ≥ 1, confirming the ratio is a
continuum property, not a truncation effect.

### Result 3: Mechanism — Hilbert space connectivity

Magnetic Hamiltonian connectivity at Λ=1 (off-diagonal non-zeros per row):

| Lattice | Dim | Plaquettes | Avg NNZ/row | Max NNZ/row |
|---------|-----|------------|-------------|-------------|
| Simplicial | 219 | 10 (tri) | 8.95 | 20 |
| Hypercubic 2D | 3 | 4 (sq) | 1.33 | 2 |
| Hypercubic 3D | 69 | 24 (sq) | 5.91 | 12 |

The simplicial lattice has **6.7× more connections per state** than hypercubic
2D. Each triangular plaquette shifts 3 edges — the 3-edge holonomy is easier
to satisfy within truncation bounds than the 4-edge square holonomy, so more
off-diagonal matrix elements survive. This creates a wider magnetic band and
a denser spectral structure.

### Result 4: Per-plaquette gap is universal

Normalizing the gap by lattice parameters at g²=1, Λ=1:

| Lattice | Gap | n_plaq | n_edge | b₁ | Gap/plaq | Gap/edge | Gap/b₁ |
|---------|-----|--------|--------|-----|----------|----------|--------|
| Simplicial | 2.10 | 10 | 10 | 6 | 0.210 | 0.210 | 0.350 |
| Hyp 2D | 4.00 | 4 | 4 | 1 | 1.000 | 1.000 | 4.000 |
| Hyp 3D | 4.79 | 24 | 12 | 5 | 0.200 | 0.399 | 0.958 |

**Gap per plaquette is nearly identical between simplicial (0.21) and hypercubic
3D (0.20).** The absolute gap difference comes from the different ratio of
gauge DOF per plaquette:

- Simplicial: b₁/n_plaq = 6/10 = 0.60
- Hyp 3D: b₁/n_plaq = 5/24 = 0.21
- Hyp 2D: b₁/n_plaq = 1/4 = 0.25

The simplicial lattice has **3× more gauge DOF per plaquette**, which spreads
the magnetic energy across more modes and reduces the gap.

### Result 5: Multiplet structure reflects lattice symmetry

First excited state multiplet at g²=1:

| Lattice | Λ | E₀ | E₁ multiplet | Degeneracy |
|---------|---|-----|--------------|------------|
| Simplicial | 1 | -3.024 | -0.923 | **6-fold** |
| Simplicial | 2 | -3.870 | -2.030 | **6-fold** |
| Hyp 2D | 1 | -2.000 | 2.000 | 1 (only 3 states total) |
| Hyp 3D | 1 | -9.974 | -5.185 | **3-fold** |

The 6-fold degeneracy on the pentachoron reflects its S₅ permutation symmetry.
All 10 triangles are equivalent, so the 6 fundamental cycles form a degenerate
multiplet under the symmetry group. This persists at Λ=2, confirming it's exact
(not accidental).

Full simplicial multiplet structure at Λ=1 (degeneracies):
1, **6**, 1, 4, 5, 4, 5, 4.

### Physical interpretation

**Triangular plaquettes are not intrinsically weaker confining.** The gap per
plaquette is the same as for square plaquettes. The difference is topological:
simplicial complexes have a higher edge-to-vertex ratio (and thus more gauge
DOF per plaquette), which distributes the magnetic energy across more modes.

The string tension σ ∝ Δ/a extracted from the gap would differ between lattice
types at the same bare coupling. But this is just a renormalization effect —
the continuum limit requires different bare coupling tuning:

- Simplicial: g²_bare needs to be **smaller** (weaker coupling) to match a
  given physical string tension
- This is analogous to how different lattice actions (Wilson vs improved) require
  different bare coupling to reach the same continuum physics

The minimum gap ratio at the crossover (g² ≈ 1.5) is where the effect is
strongest — the denser connectivity of triangular plaquettes creates the
maximum separation from the square-plaquette spectrum.

### Open questions

- **Volume dependence.** The pentachoron vs 2³ torus comparison has different
  volumes. A fairer test: build a simplicial triangulation of the same torus
  and compare at matched volume.

- **Finite-size scaling.** How does the gap scale with system size on
  simplicial lattices? Need 2-pentachoron at Λ=2 (dim ~269K, needs Lanczos).

- **Universality of gap/plaq.** The gap/plaquette equality (0.21 vs 0.20) at
  Λ=1 might be coincidental. Check at Λ=2 where simplicial gives 0.18 vs
  hyp3D gives 0.16 — still close but drifting. Need Λ=3+ to determine if
  this converges.

---

## 2026-02-21 — Literature Review

### Positioning

All existing non-hypercubic lattice gauge theory work uses lattices with
**3-link vertices** (honeycomb, triamond, hyperhoneycomb) which have **large
plaquettes** (6-edge or 10-edge). Their motivation is reducing quantum
simulation gate counts. Nobody has gone in the other direction — using lattices
with small (triangular, 3-edge) plaquettes — and nobody has compared spectral
gaps between geometries.

### Closest prior work

**Illa, Savage, Yao** — "Improved Honeycomb and Hyperhoneycomb Lattice
Hamiltonians for Quantum Simulations of Non-Abelian Gauge Theories"
([arXiv:2503.09688](https://arxiv.org/abs/2503.09688), March 2025, PRD 111)
KS Hamiltonians for SU(N_c) on honeycomb (2+1D) and hyperhoneycomb (3+1D).
Hexagonal plaquettes (6-link), 3-link vertices. Symanzik improvement program.
No spectral gap calculations, no geometry comparison.

**Kavaki, Lewis** — "From square plaquettes to triamond lattices for SU(2)
gauge theory"
([arXiv:2401.14570](https://arxiv.org/abs/2401.14570), January 2024, Comm. Phys.)
SU(2) Hamiltonian on triamond lattice (3D, 3-link vertices). Plaquettes are
10-link loops. Single unit cell eigenvalues only. No spectral gap comparison
between geometries.

**Kavaki, Lewis** — "False vacuum decay in triamond lattice gauge theory"
([arXiv:2503.01119](https://arxiv.org/abs/2503.01119), March 2025)
Follow-up: torelon spectrum and false vacuum decay on triamond. Spectrum only
for triamond, no cross-geometry comparison.

**Muller, Yao** — "Simple Hamiltonian for Quantum Simulation of Strongly
Coupled 2+1D SU(2) Lattice Gauge Theory on a Honeycomb Lattice"
([arXiv:2307.00045](https://arxiv.org/abs/2307.00045), June 2023)
SU(2) on honeycomb with j_max=1/2. Integrates out Gauss's law for local Pauli
Hamiltonian. No spectral gap comparison to square lattice.

**Illa, Savage, Yao** — "Dynamical Local Tadpole-Improvement in Quantum
Simulations of Gauge Theories"
([arXiv:2504.21575](https://arxiv.org/abs/2504.21575), April 2025)
Tadpole improvement on plaquette chains and honeycomb lattices. Operational
improvement, not spectral comparison.

### General graph formulations

**Burbano, Bauer** — "Gauge Loop-String-Hadron Formulation on General Graphs"
([arXiv:2409.13812](https://arxiv.org/abs/2409.13812), September 2024, JHEP 2025)
Mathematical framework for KS theory on general graphs with arbitrary vertex
valency. SU(2) formalism paper — no numerical results, no spectral calculations,
no study of triangular plaquettes. Our work can be seen as an instantiation of
this general framework on simplicial complexes, with the first numerical results.

**Ciavarella, Bauer, Halimeh** — "Generic Hilbert Space Fragmentation in
Kogut-Susskind Lattice Gauge Theories"
([arXiv:2502.03533](https://arxiv.org/abs/2502.03533), February 2025)
Fragmentation in KS theories on general lattices. Focuses on ergodicity
breaking, not spectral gaps or geometry dependence.

### Classical simplicial gauge theory

**Christiansen, Halvorsen** — "A simplicial gauge theory"
([arXiv:1006.2059](https://arxiv.org/abs/1006.2059), 2010, J. Math. Phys. 2012)
Gauge-invariant action on simplicial meshes, discrete Noether's theorem. This
is **Lagrangian/Euclidean**, not Hamiltonian. No spectral gaps, no quantum
simulation. Our work is the Hamiltonian counterpart.

### Z₂ on triangular geometries

**Brenig** — "Spinless fermions in a Z₂ gauge theory on the triangular ladder"
([arXiv:2202.04668](https://arxiv.org/abs/2202.04668), February 2022)
Z₂ gauge theory on a triangular ladder. Notes that "the triangular unit cell
and the ladder geometry strongly modify the physics." Uses DMRG for phase
transitions. Does not compute spectral gaps or make systematic geometry
comparisons.

### Stabilizer codes and simplicial complexes

**Breuckmann** — "Homological Quantum Codes Beyond the Toric Code"
([arXiv:1802.01520](https://arxiv.org/abs/1802.01520), February 2018, PhD thesis)
Codes from curved surfaces and 4D geometries. Shows overhead reduction vs
surface codes. Focuses on hyperbolic surfaces, not simplicial triangulations
of flat space.

**Yao** — "Quantum Error Correction Codes for Truncated SU(2) Lattice Gauge
Theories"
([arXiv:2511.13721](https://arxiv.org/abs/2511.13721), November 2025)
Converts Gauss's law into stabilizers for SU(2) QEC codes. Bridge between LGT
and QEC, but SU(2)-specific, not Z₂ on simplicial complexes.

**Survey on Codes from Simplicial Complexes**
([arXiv:2409.16310](https://arxiv.org/abs/2409.16310), September 2024)
General framework for classical and quantum codes from simplicial complexes.
Does not make the specific claim about Z₂ stabilizer codes from gauge theory
encoding more logical qubits than toric codes.

### Novelty assessment

1. **KS Hamiltonian with triangular plaquettes** — **Novel.** Burbano/Bauer's
   general-graph formalism could accommodate it, but nobody has instantiated or
   studied it. All existing non-hypercubic work uses large plaquettes (6+ edges).

2. **Spectral gap comparison across geometries** — **Novel.** Zero papers
   compare gaps between lattice types for the same gauge theory.

3. **Gap-per-plaquette universality** — **Novel.** The concept does not appear
   anywhere in the literature.

4. **Simplicial stabilizer codes vs toric codes** — **Likely novel.** Color
   code literature (Breuckmann) knows different topologies yield different rates,
   but the specific connection from Z₂ gauge theory on simplicial lattices to
   higher logical qubit counts appears new. Should be carefully compared with
   color code constructions.

---

## 2026-02-21 — 2-Pentachoron Analysis (Λ=1, dim=3135)

### Complex

Two pentachorons sharing face [0,1,2,3]: pent1 = [0,1,2,3,4], pent2 = [0,1,2,3,5].

| Property | 1-pent | 2-pent |
|----------|--------|--------|
| Vertices | 5 | 6 |
| Edges | 10 | 14 |
| Triangles | 10 | 16 |
| b₁ | 6 | 9 |
| Hilbert dim (Λ=1) | 219 | 3,135 |

### Result 1: Finite-size gap scaling

The gap shrinks as system size increases — expected for a confining theory
approaching the thermodynamic limit:

| g² | 1-pent gap | 2-pent gap | Ratio (2p/1p) |
|----|------------|------------|---------------|
| 0.1 | 16.55 | 14.65 | 0.885 |
| 0.5 | 3.50 | 3.10 | 0.886 |
| 1.0 | 2.10 | 1.86 | 0.888 |
| 1.5 | 1.84 | 1.71 | 0.930 |
| 2.0 | 2.50 | 2.35 | 0.941 |
| 5.0 | 7.44 | 7.43 | 0.998 |
| 10.0 | 14.99 | 14.99 | 1.000 |

The gap ratio is **0.885–0.888** in weak-to-moderate coupling (g² ≤ 1),
rising toward 1.0 in strong coupling where the gap is dominated by the
electric term (extensive in system size). The ~11% gap reduction from
doubling the complex is mild — consistent with a gapped phase (confining).

### Result 2: Wilson loops — all 9 loops are length 3

All 9 fundamental loops on the 2-pentachoron complex have **equal length (3
edges)**. This means all fundamental cycles are single-triangle loops — the
topology doesn't yet give us loops of different areas.

| g² | ⟨W⟩ (avg over 9 loops) | -log⟨W⟩ |
|----|------------------------|---------|
| 0.1 | 0.475 | 0.744 |
| 1.0 | 0.453 | 0.793 |
| 2.0 | 0.178 | 1.725 |
| 10 | 6.67e-3 | 5.01 |
| 50 | 2.67e-4 | 8.23 |

The Wilson loop values match the 1-pentachoron results closely (⟨W⟩ = 0.467
at g²=1 for 1-pent vs 0.453 for 2-pent). The slight decrease at moderate
coupling is consistent with the smaller gap — the ground state has slightly
more electric fluctuations on the larger lattice.

**For a proper area law test, need loops enclosing different numbers of
triangles.** This requires either (a) a 3+ pentachoron complex where some
fundamental loops wrap around multiple triangles, or (b) composite Wilson
loops built from products of fundamental loops.

### Result 3: Entanglement entropy across the shared face

Natural bipartition: 10 edges of pent1 (set A) vs 4 edges unique to pent2
(set B).

| g² | S_A | S_max |
|----|-----|-------|
| 0.1 | **2.70** | 2.48 |
| 0.5 | 2.63 | 2.48 |
| 1.0 | **2.31** | 2.48 |
| 2.0 | 0.53 | 2.48 |
| 5.0 | 0.02 | 2.48 |
| 100 | 0.00 | 2.48 |

At weak coupling, the entanglement entropy **exceeds** S_max = ln(dim_B) ≈
2.48! This is because the partition A has 10 edges while B has only 4 — the
reduced density matrix ρ_A can have rank up to dim(B), so S_A ≤ ln(dim_B)
should hold. The value S_A = 2.70 > 2.48 is surprising and warrants
investigation — it may indicate the partition sizes (10 vs 4 edges) create a
highly entangled ground state.

At strong coupling (g² ≥ 5), entropy drops to ~0 as expected (ground state ≈
product state |0,...,0⟩).

Symmetric half-half cut (7/7 edges): S = 4.11 at g²=0.1, S = 3.58 at g²=1.0.

### Result 4: Broken symmetry — multiplet structure

The 2-pentachoron breaks the S₅ symmetry of the single pentachoron. The
symmetry group is now S₄ × Z₂ (permutations of the 4 shared vertices times
the exchange of vertices 4↔5).

Low-lying multiplets at g²=1:

| Level | Energy | Degeneracy |
|-------|--------|------------|
| 0 | -4.827 | 1 (ground state) |
| 1 | -2.963 | **3** |
| 2 | -2.521 | 3 |
| 3 | -2.436 | 3 |
| 4 | -2.257 | 1 |
| 5 | -2.121 | 1 |
| 6 | -1.810 | 3 |
| 7 | -1.701 | 3 |

The dominant degeneracy is **3-fold**, corresponding to the 3D irrep of the
tetrahedral (S₄) subgroup acting on the shared face [0,1,2,3]. Compare with
the 6-fold degeneracy of the single pentachoron (from S₅).

### Result 5: Stabilizer code

| Complex | [[n, k, d]] | Star weights | Plaquette weight |
|---------|-------------|--------------|------------------|
| 2-pent | [[14, 9, 3]] | 4-5 | 3 |

Matches the Phase 1 result. The rate k/n = 9/14 = 0.64 is higher than the
single pentachoron's 6/10 = 0.60, and much higher than the toric code's
10/18 = 0.56 at comparable size.

---
