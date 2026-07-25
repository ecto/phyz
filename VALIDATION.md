# phyz solver validation

Every entry compares a phyz solver against a closed-form solution or published reference data, and reports a quantitative error — not a pass/fail bit. Tolerances are declared before the measurement is taken and are never relaxed to make a benchmark pass.

**40 passed · 2 failed · 4 reported (diagnostic)**

## Failures

| Benchmark | Metric | Measured | Expected | Error | Tolerance |
|---|---|---:|---:|---:|---:|
| `gravity.pn.mercury_precession` | precession of the eccentricity vector (arcsec/century), 400 orbits at 4000 steps/orbit | 1.433255e1 | 4.299726e1 | 6.667e-1 (rel) | 2.000e-2 |
| `gravity.pn.mercury_convergence` | \|Δϖ_measured − Δϖ_GR\| / Δϖ_GR at 4000 steps/orbit (60 orbits) | 6.663155e-1 | 0.000000e0 | 6.663e-1 (abs) | 2.000e-2 |

## Rigid-body dynamics — Featherstone ABA (`phyz-rigid`)

### Rigid-rod pendulum period at θ₀ = 5° — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_period.5deg`
- **reference**: Exact solution of θ̈ = −ω₀² sin θ: T = (4/ω₀)K(sin(θ₀/2)), ω₀ = √(3g/2L) for a uniform rod hinged at one end
- **metric**: period (s), L = 1 m, RK4 with Δt = 1e-5
- **measured**: 1.639006406e0
- **expected**: 1.639006406e0
- **error**: 2.8550e-12 (rel) — tolerance 1.0000e-6
- _note_: small-angle T₀ = 1.638226 s would be off by 0.05% here — the elliptic correction is the thing being resolved

### Rigid-rod pendulum period at θ₀ = 30° — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_period.30deg`
- **reference**: Exact solution of θ̈ = −ω₀² sin θ: T = (4/ω₀)K(sin(θ₀/2)), ω₀ = √(3g/2L) for a uniform rod hinged at one end
- **metric**: period (s), L = 1 m, RK4 with Δt = 1e-5
- **measured**: 1.666745878e0
- **expected**: 1.666745878e0
- **error**: 2.9399e-12 (rel) — tolerance 1.0000e-6
- _note_: small-angle T₀ = 1.638226 s would be off by 1.71% here — the elliptic correction is the thing being resolved

### Rigid-rod pendulum period at θ₀ = 90° — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_period.90deg`
- **reference**: Exact solution of θ̈ = −ω₀² sin θ: T = (4/ω₀)K(sin(θ₀/2)), ω₀ = √(3g/2L) for a uniform rod hinged at one end
- **metric**: period (s), L = 1 m, RK4 with Δt = 1e-5
- **measured**: 1.933665045e0
- **expected**: 1.933665045e0
- **error**: 3.6331e-12 (rel) — tolerance 1.0000e-6
- _note_: small-angle T₀ = 1.638226 s would be off by 15.28% here — the elliptic correction is the thing being resolved

### Rigid-rod pendulum period at θ₀ = 150° — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_period.150deg`
- **reference**: Exact solution of θ̈ = −ω₀² sin θ: T = (4/ω₀)K(sin(θ₀/2)), ω₀ = √(3g/2L) for a uniform rod hinged at one end
- **metric**: period (s), L = 1 m, RK4 with Δt = 1e-5
- **measured**: 2.886888544e0
- **expected**: 2.886888544e0
- **error**: 5.4653e-12 (rel) — tolerance 1.0000e-6
- _note_: small-angle T₀ = 1.638226 s would be off by 43.25% here — the elliptic correction is the thing being resolved

### Pendulum period error converges at fourth order in Δt — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_period_order`
- **reference**: Classical RK4 is fourth-order accurate; the period error inherits that order
- **metric**: relative period error at θ₀ = 90°, Δt = 1e−3 s
- **measured**: 7.628220382e-13
- **expected**: 0.000000000e0
- **error**: 7.6282e-13 (abs) — tolerance 1.0000e-8
- **convergence in Δt**: measured order p = 4.009 (expected 4.0 ± 0.6) — OK

  | Δt | error | ratio |
  |---:|---:|---:|
  | 8.000000e-3 | 3.210297e-9 | NaN |
  | 4.000000e-3 | 2.093311e-10 | 15.336 |
  | 2.000000e-3 | 1.333245e-11 | 15.701 |
  | 1.000000e-3 | 7.628220e-13 | 17.478 |
- _note_: A fourth-order fit here confirms the ABA acceleration is a smooth, correct function of (q, v) — a wrong-but-consistent force law would still converge, but to the wrong period, which the amplitude sweep above catches.

### Total mechanical energy is conserved along the pendulum swing — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.pendulum_energy`
- **reference**: Autonomous conservative system: T + V is an exact integral of motion
- **metric**: peak |ΔE| / (swing energy) over 20 s at Δt = 1e−4 s
- **measured**: 1.623230314e-14
- **expected**: 0.000000000e0
- **error**: 1.6232e-14 (abs) — tolerance 1.0000e-8
- _note_: Tests `phyz_rigid::kinetic_energy` (via CRBA) and `phyz_rigid::potential_energy` jointly: an inconsistency between the mass matrix and the centre-of-mass height shows up here even when ABA itself is correct.

### Fast symmetric top: mean precession rate — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.top_precession`
- **reference**: Goldstein, *Classical Mechanics* 3e §5.7 — Ω = mgl/(I₃ω₃) in the fast-top limit
- **metric**: ⟨ψ̇⟩ (rad/s), ω₃ = 200 rad/s, θ₀ = 30°, averaged over 10 s
- **measured**: 2.458201617e-1
- **expected**: 2.451662500e-1
- **error**: 2.6672e-3 (rel) — tolerance 1.0000e-2
- _note_: Released from rest in θ and ψ, so the top executes cuspidal nutation; the mean of ψ̇ over many nutation cycles is the fast-top rate. Modelled as a z–x–z gimbal of three revolute joints, which exercises a multi-body chain with a non-trivial inertia tensor.

### Top precession approaches the fast-top limit as ω₃⁻² — PASS

- **crate**: `phyz-rigid`
- **id**: `rigid.top_precession_order`
- **reference**: The next correction to Ω = mgl/(I₃ω₃) is O(mglI₁/(I₃²ω₃²)) (Goldstein §5.7)
- **metric**: relative deviation from mgl/(I₃ω₃) at ω₃ = 400 rad/s
- **measured**: 6.764505243e-4
- **expected**: 0.000000000e0
- **error**: 6.7645e-4 (abs) — tolerance 5.0000e-3
- **convergence in 1/ω₃**: measured order p = 2.025 (expected 2.0 ± 0.4) — OK

  | 1/ω₃ | error | ratio |
  |---:|---:|---:|
  | 2.000000e-2 | 4.518930e-2 | NaN |
  | 1.000000e-2 | 1.122850e-2 | 4.025 |
  | 5.000000e-3 | 2.690907e-3 | 4.173 |
  | 2.500000e-3 | 6.764505e-4 | 3.978 |
- _note_: This is a physics convergence test rather than a numerical one: the *analytic* formula is the approximation, and the simulator must reproduce the rate at which it becomes exact.

## Gravity — Newtonian and post-Newtonian (`phyz-gravity`)

### Kepler two-body: specific orbital energy is conserved — PASS

- **crate**: `phyz-gravity`
- **id**: `gravity.kepler.energy`
- **reference**: Closed form: E = −μ/2a is an exact integral of the Newtonian two-body problem
- **metric**: peak |ΔE|/|E| over 20 orbits (e = 0.2, 2000 steps/orbit)
- **measured**: 2.573116254e-6
- **expected**: 0.000000000e0
- **error**: 2.5731e-6 (abs) — tolerance 9.8696e-5
- _note_: Velocity Verlet is symplectic, so the energy error must stay bounded rather than growing secularly.
- _note_: Tolerance 10(2π/N)² = 9.87e-5 is the theoretical O((ωΔt)²) bound at N = 2000 steps/orbit, not a fitted number; the convergence entries below verify the Δt² scaling that justifies it.

### Kepler two-body: specific angular momentum is conserved — PASS

- **crate**: `phyz-gravity`
- **id**: `gravity.kepler.angular_momentum`
- **reference**: Closed form: h = r × v is exactly conserved for any central force
- **metric**: peak |Δh|/|h| over 20 orbits
- **measured**: 1.259656687e-14
- **expected**: 0.000000000e0
- **error**: 1.2597e-14 (abs) — tolerance 1.0000e-9

### Kepler two-body: Laplace–Runge–Lenz vector is conserved — PASS

- **crate**: `phyz-gravity`
- **id**: `gravity.kepler.lrl`
- **reference**: Closed form: e = (v×h)/μ − r̂ is exactly conserved only for a 1/r² force (Bertrand/Runge–Lenz symmetry)
- **metric**: peak |Δe|/|e| over 20 orbits
- **measured**: 3.655732329e-4
- **expected**: 0.000000000e0
- **error**: 3.6557e-4 (abs) — tolerance 1.9739e-3
- _note_: The LRL vector is the sharpest of the three: it detects any spurious non-1/r² component in the force law, which energy and angular momentum do not.
- _note_: Unlike the energy error, the apsidal error of a symplectic integrator accumulates linearly in the number of orbits, so the pre-registered bound is N_orbits · 10(2π/N_steps)² = 1.97e-3.

### Kepler two-body: energy error scales as Δt² (velocity Verlet) — PASS

- **crate**: `phyz-gravity`
- **id**: `gravity.kepler.energy_order`
- **reference**: Velocity Verlet is a second-order symplectic integrator; the shadow-Hamiltonian energy error is O(Δt²) and bounded
- **metric**: peak |ΔE|/|E| over 5 orbits at 4000 steps/orbit
- **measured**: 6.432918642e-7
- **expected**: 0.000000000e0
- **error**: 6.4329e-7 (abs) — tolerance 2.4674e-5
- **convergence in Δt/T**: measured order p = 2.000 (expected 2.0 ± 0.2) — OK

  | Δt/T | error | ratio |
  |---:|---:|---:|
  | 2.000000e-3 | 4.115354e-5 | NaN |
  | 1.000000e-3 | 1.029165e-5 | 3.999 |
  | 5.000000e-4 | 2.573116e-6 | 4.000 |
  | 2.500000e-4 | 6.432919e-7 | 4.000 |

### Kepler two-body: LRL error scales as Δt² — PASS

- **crate**: `phyz-gravity`
- **id**: `gravity.kepler.lrl_order`
- **reference**: Second-order integrator on an exactly-conserved vector
- **metric**: peak |Δe|/|e| over 5 orbits at 4000 steps/orbit
- **measured**: 2.520140399e-5
- **expected**: 0.000000000e0
- **error**: 2.5201e-5 (abs) — tolerance 1.2337e-4
- **convergence in Δt/T**: measured order p = 2.000 (expected 2.0 ± 0.2) — OK

  | Δt/T | error | ratio |
  |---:|---:|---:|
  | 2.000000e-3 | 1.612451e-3 | NaN |
  | 1.000000e-3 | 4.031963e-4 | 3.999 |
  | 5.000000e-4 | 1.008043e-4 | 4.000 |
  | 2.500000e-4 | 2.520140e-5 | 4.000 |

### Mercury perihelion precession from integrated 1PN equations of motion — FAIL

- **crate**: `phyz-gravity`
- **id**: `gravity.pn.mercury_precession`
- **reference**: Einstein (1915); Will, *Living Rev. Rel.* 17 (2014) — Δϖ = 6πGM/(c²a(1−e²)) = 42.98″/century for Mercury
- **metric**: precession of the eccentricity vector (arcsec/century), 400 orbits at 4000 steps/orbit
- **measured**: 1.433254917e1
- **expected**: 4.299725816e1
- **error**: 6.6666e-1 (rel) — tolerance 2.0000e-2
- _note_: Newtonian control run (same integrator, same Δt, 1PN term switched off) drifts -381.9085″/century; that baseline is subtracted from the 1PN run so the residual is the physical effect and not integrator error.
- _note_: Closed-form check of the textbook formula alone: 6πGM/(c²a(1−e²)) = 42.997″/century — this is what the pre-existing `test_mercury_precession` (crates/phyz-gravity/src/pn.rs:321-334) asserted, without ever calling the solver.
- _note_: The docstring at pn.rs:78-86 states a_1PN = Gm_j/r² [(4G(m_i+m_j)/r − v_i²)n + 4(v_i·v_j)n − (v_i·n)v_j]/c², which is not the standard EIH 1PN acceleration. The EIH form (Will 1993 eq. 6.80; Blanchet 2014 eq. 203) is a_1PN = −(Gm_j/r²c²){ n̂[4Gm_i/r + 5Gm_j/r − v_i² − 2v_j² + 4v_i·v_j + (3/2)(n̂·v_j)²] + (v_i − v_j)(4n̂·v_i − 3n̂·v_j) }, with n̂ pointing from j to i. The code at pn.rs:89-114 differs from EIH in three places: the mass coefficient is 4(m_i+m_j) rather than 4m_i + 5m_j; the velocity term multiplies v_j rather than (v_i − v_j); and the overall sign is positive in the code's n = (x_j − x_i)/r convention where EIH requires negative. Any one of those changes the precession.

### Integrated Mercury precession converges to the GR value as Δt → 0 — FAIL

- **crate**: `phyz-gravity`
- **id**: `gravity.pn.mercury_convergence`
- **reference**: Δϖ = 6πGM/(c²a(1−e²)); a correct 1PN force law makes the residual a pure integrator error that vanishes as Δt²
- **metric**: |Δϖ_measured − Δϖ_GR| / Δϖ_GR at 4000 steps/orbit (60 orbits)
- **measured**: 6.663155290e-1
- **expected**: 0.000000000e0
- **error**: 6.6632e-1 (abs) — tolerance 2.0000e-2
- **convergence in Δt/T**: measured order p = 0.000 (expected 2.0 ± 0.5) — MISMATCH

  | Δt/T | error | ratio |
  |---:|---:|---:|
  | 1.000000e-3 | 6.663941e-1 | NaN |
  | 5.000000e-4 | 6.663313e-1 | 1.000 |
  | 2.500000e-4 | 6.663155e-1 | 1.000 |
- _note_: If the residual does not shrink under refinement, the discrepancy is in the force law, not the integrator.

## Electromagnetics — FDTD on a Yee grid (`phyz-em`)

### Numerical dispersion, 1-D PEC cavity mode m=1 — PASS

- **crate**: `phyz-em`
- **id**: `em.yee_dispersion.m1`
- **reference**: Yee (1966); Taflove & Hagness §4.3, eq. 4.14 — sin²(ωΔt/2)/(cΔt)² = Σ_a sin²(k_aΔx/2)/Δx²
- **metric**: ω of the m=1 standing mode (rad/s)
- **measured**: 1.471467353e10
- **expected**: 1.471467353e10
- **error**: 2.6877e-12 (rel) — tolerance 1.0000e-9
- _note_: For one spatial eigenmode the leapfrog is exactly E^{n+1} = 2cos(ωΔt)E^n − E^{n−1}, so a correct implementation matches the analytic Yee root to round-off. This directly tests the update coefficients Δt/(μΔx) and Δt/(εΔx).

### Numerical dispersion, 1-D PEC cavity mode m=3 (coarse, kΔx large) — PASS

- **crate**: `phyz-em`
- **id**: `em.yee_dispersion.m3`
- **reference**: Yee (1966); Taflove & Hagness §4.3, eq. 4.14
- **metric**: ω of the m=3 standing mode (rad/s)
- **measured**: 8.800370794e10
- **expected**: 8.800370794e10
- **error**: 1.0473e-13 (rel) — tolerance 1.0000e-9

### Phase-velocity error vanishes as Δx² under refinement — PASS

- **crate**: `phyz-em`
- **id**: `em.dispersion_convergence`
- **reference**: Second-order accuracy of the Yee scheme; ω/ck − 1 = −(kΔx)²(1 − S²)/24 + O(Δx⁴)
- **metric**: |ω_num − ck| / ck at the finest grid (Δx = L/128)
- **measured**: 2.300803253e-5
- **expected**: 0.000000000e0
- **error**: 2.3008e-5 (abs) — tolerance 3.4508e-5
- **convergence in Δx/L**: measured order p = 2.000 (expected 2.0 ± 0.1) — OK

  | Δx/L | error | ratio |
  |---:|---:|---:|
  | 6.250000e-2 | 1.472338e-3 | NaN |
  | 3.125000e-2 | 3.681179e-4 | 4.000 |
  | 1.562500e-2 | 9.203155e-5 | 4.000 |
  | 7.812500e-3 | 2.300803e-5 | 4.000 |
- _note_: Tolerance is 1.5 × (error at Δx = L/16) / 64 = 3.451e-5, derived from the measured coarse grid rather than chosen after the fact.

### TM₁₁₀ square-cavity resonance vs the discrete Yee root — PASS

- **crate**: `phyz-em`
- **id**: `em.cavity_tm110.discrete`
- **reference**: Yee dispersion relation with k = (π/L, π/L, 0)
- **metric**: ω of the TM₁₁₀ mode (rad/s)
- **measured**: 3.329143762e10
- **expected**: 3.329143762e10
- **error**: 5.3282e-13 (rel) — tolerance 1.0000e-9
- _note_: Exercises the x- and y-curl updates and the PEC boundary on four walls.

### TM₁₁₀ square-cavity resonance vs closed-form f = (c/2)√((m/Lx)²+(n/Ly)²) — PASS

- **crate**: `phyz-em`
- **id**: `em.cavity_tm110.physical`
- **reference**: Pozar, *Microwave Engineering* 4e, §6.3 — rectangular cavity resonant frequency
- **metric**: resonant frequency (Hz), L = 40 mm, 41×41 cells
- **measured**: 5.298496859e9
- **expected**: 5.299632000e9
- **error**: 2.1419e-4 (rel) — tolerance 2.0000e-3
- _note_: Residual error is the Yee grid-dispersion error at kΔx = π/40.

### Cavity resonance error vanishes as Δx² under refinement — PASS

- **crate**: `phyz-em`
- **id**: `em.cavity_convergence`
- **reference**: Second-order accuracy of the Yee scheme
- **metric**: |f_num − f_exact| / f_exact at 81×81
- **measured**: 5.354654448e-5
- **expected**: 0.000000000e0
- **error**: 5.3547e-5 (abs) — tolerance 8.0368e-5
- **convergence in Δx/L**: measured order p = 2.000 (expected 2.0 ± 0.1) — OK

  | Δx/L | error | ratio |
  |---:|---:|---:|
  | 1.000000e-1 | 3.429041e-3 | NaN |
  | 5.000000e-2 | 8.568683e-4 | 4.002 |
  | 2.500000e-2 | 2.141924e-4 | 4.000 |
  | 1.250000e-2 | 5.354654e-5 | 4.000 |

### Reflection coefficient of the CPML absorbing boundary — PASS

- **crate**: `phyz-em`
- **id**: `em.cpml_reflection`
- **reference**: Roden & Gedney (2000); Taflove & Hagness ch. 7 — a correctly implemented 10-cell CPML reaches R < −60 dB for a normally incident broadband pulse
- **metric**: worst broadband reflection R (dB), 10-cell CPML
- **measured**: -9.031680093e1
- **expected**: -6.000000000e1
- **error**: 0.0000e0 (abs) — tolerance 0.0000e0
- _note_: One-sided criterion: any value at or below −60 dB passes. Measured by differencing the probe against the same excitation in a 4000-cell domain, so what remains is exactly what the boundary sent back.

### CPML drains a radiating pulse out of the grid — PASS

- **crate**: `phyz-em`
- **id**: `em.cpml_drains_the_grid`
- **reference**: An absorbing boundary removes outgoing energy; after several transits an open domain should retain a negligible fraction of the peak field energy, and orders of magnitude less than an impedance-mismatched lossy layer
- **metric**: residual field energy / peak, after 4000 steps on a 300-cell grid
- **measured**: 2.571522080e-6
- **expected**: 1.618300643e-4
- **error**: 0.0000e0 (abs) — tolerance 0.0000e0
- _note_: The cheap `LossyAbsorber` retains 1.618e-1 of the peak on the same problem — 62932× more energy left ringing in the domain. Criterion: CPML must leave at most 1/1000 of that, i.e. 1.618e-4.

### Residual energy left by the cheap graded-conductivity absorber — REPORT

- **crate**: `phyz-em`
- **id**: `em.lossy_absorber_residual`
- **reference**: `BoundaryCondition::LossyAbsorber` adds electric loss σ without the matching magnetic loss σ* = σμ/ε, so it is impedance-mismatched at every frequency and is documented as not being a PML
- **metric**: residual field energy / peak, after 4000 steps on a 300-cell grid
- **measured**: 1.618300643e-1
- **expected**: 0.000000000e0
- **error**: 1.6183e-1 (abs) — tolerance 0.0000e0
- _note_: Reported, not failed: this boundary is offered as the cheap option and the crate documents its limits. The measurement quantifies the gap so a caller can decide; use `BoundaryCondition::Cpml` when reflections matter.

## Fluids — lattice Boltzmann D2Q9 (`phyz-lbm`)

### Plane Poiseuille flow: velocity profile vs u(y) = F y(H−y)/(2ρν) — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.poiseuille.profile`
- **reference**: Closed form for steady laminar channel flow (Batchelor §4.2), via `phyz_lbm::analytic::poiseuille_force_driven`
- **metric**: relative L2 profile error, 21-node channel, ν = 0.05, default collision
- **measured**: 2.407028014e-12
- **expected**: 0.000000000e0
- **error**: 2.4070e-12 (abs) — tolerance 1.0000e-3
- _note_: The body force is set from the analytic relation, so this tests the viscosity the collision operator actually realises together with the wall treatment.

### Poiseuille error is independent of viscosity (BGK) — REPORT

- **crate**: `phyz-lbm`
- **id**: `lbm.poiseuille.viscosity_independence.bgk`
- **reference**: The analytic profile is exact at every ν once the force is rescaled to hold u_peak fixed, so a correct wall treatment gives a ν-independent error (Ginzburg & d'Humières 2003 on the TRT magic parameter)
- **metric**: spread of the relative L2 error across ν ∈ {0.02, 0.05, 0.2, 1}
- **measured**: 1.432024236e-1
- **expected**: 0.000000000e0
- **error**: 1.4320e-1 (abs) — tolerance 1.0000e-3
- _note_: errors by ν: ν=0.02 → 3.045e-3, ν=0.05 → 2.732e-3, ν=0.2 → 2.857e-3, ν=1 → 1.459e-1
- _note_: Reported, not failed: `CollisionModel::Bgk` is not the crate default. Plain BGK bounce-back places the no-slip plane at a τ-dependent position, so its error moves with viscosity even though the physical problem does not — which is precisely why `CollisionModel::default()` is TRT with Λ = 3/16. The TRT and MRT rows above are the pass/fail claims.

### Poiseuille error is independent of viscosity (TRT) — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.poiseuille.viscosity_independence.trt`
- **reference**: The analytic profile is exact at every ν once the force is rescaled to hold u_peak fixed, so a correct wall treatment gives a ν-independent error (Ginzburg & d'Humières 2003 on the TRT magic parameter)
- **metric**: spread of the relative L2 error across ν ∈ {0.02, 0.05, 0.2, 1}
- **measured**: 5.568178844e-12
- **expected**: 0.000000000e0
- **error**: 5.5682e-12 (abs) — tolerance 1.0000e-3
- _note_: errors by ν: ν=0.02 → 5.588e-12, ν=0.05 → 2.407e-12, ν=0.2 → 1.274e-13, ν=1 → 1.964e-14

### Poiseuille error is independent of viscosity (MRT) — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.poiseuille.viscosity_independence.mrt`
- **reference**: The analytic profile is exact at every ν once the force is rescaled to hold u_peak fixed, so a correct wall treatment gives a ν-independent error (Ginzburg & d'Humières 2003 on the TRT magic parameter)
- **metric**: spread of the relative L2 error across ν ∈ {0.02, 0.05, 0.2, 1}
- **measured**: 1.452941022e-5
- **expected**: 0.000000000e0
- **error**: 1.4529e-5 (abs) — tolerance 1.0000e-3
- _note_: errors by ν: ν=0.02 → 8.382e-7, ν=0.05 → 2.651e-7, ν=0.2 → 3.095e-7, ν=1 → 1.479e-5

### Taylor–Green vortex: kinetic-energy decay rate vs E(t) = E₀ exp(−2ν(k_x²+k_y²)t) — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.taylor_green.decay`
- **reference**: Taylor & Green (1937); exact unsteady Navier–Stokes solution on a periodic domain
- **metric**: effective viscosity from the decay rate, input ν = 0.02, 48² lattice
- **measured**: 2.001876160e-2
- **expected**: 2.000000000e-2
- **error**: 9.3808e-4 (rel) — tolerance 2.0000e-2
- _note_: Measures the viscosity the collide/stream pair actually delivers, which is what ties τ = 3ν + 1/2 to physical dissipation.

### Taylor–Green vortex: velocity field keeps its analytic shape — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.taylor_green.field`
- **reference**: Taylor & Green (1937), via `phyz_lbm::analytic::taylor_green_velocity`
- **metric**: relative L2 error of the velocity field after 1000 steps, 48² lattice
- **measured**: 2.016110519e-3
- **expected**: 0.000000000e0
- **error**: 2.0161e-3 (abs) — tolerance 2.0000e-2
- _note_: Energy alone can be right while the field is wrong; this pins the shape.

### Taylor–Green field error vanishes as Δx² under refinement — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.taylor_green.convergence`
- **reference**: Chapman–Enskog: LBM recovers Navier–Stokes to second order in Δx
- **metric**: relative L2 field error at 64² under diffusive scaling
- **measured**: 1.201508899e-3
- **expected**: 0.000000000e0
- **error**: 1.2015e-3 (abs) — tolerance 1.8416e-3
- **convergence in Δx/L**: measured order p = 2.016 (expected 2.0 ± 0.4) — OK

  | Δx/L | error | ratio |
  |---:|---:|---:|
  | 6.250000e-2 | 1.964346e-2 | NaN |
  | 3.125000e-2 | 4.833092e-3 | 4.064 |
  | 1.562500e-2 | 1.201509e-3 | 4.023 |
- _note_: Tolerance is 1.5 × (error at 16²) / 16 = 1.842e-3, i.e. what two halvings at second order must deliver from the measured coarse grid. The order fit is the substantive claim; this bound only pins the constant.

### Lid-driven cavity Re = 100: u on the vertical centreline — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.cavity_re100.u`
- **reference**: Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table I
- **metric**: worst |Δu| / u_lid over the 17 tabulated stations, 65² lattice
- **measured**: 5.033107212e-3
- **expected**: 0.000000000e0
- **error**: 5.0331e-3 (abs) — tolerance 4.0000e-2
- _note_: 134000 steps, steady-state residual 1.00e-8
- _note_: Ghia's own data is a 129² multigrid solution; a few percent of the lid speed is the discretisation difference at 65², not solver error. The tolerance is set to that gap, and the vortex-position check below is the shape test that a loose profile tolerance cannot provide.

### Lid-driven cavity Re = 100: v on the horizontal centreline — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.cavity_re100.v`
- **reference**: Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table II
- **metric**: worst |Δv| / u_lid over the 17 tabulated stations, 65² lattice
- **measured**: 3.352899070e-3
- **expected**: 0.000000000e0
- **error**: 3.3529e-3 (abs) — tolerance 4.0000e-2

### Lid-driven cavity Re = 100: primary vortex position — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.cavity_re100.vortex_position`
- **reference**: Ghia et al. (1982) Table I — u changes sign at y/L ≈ 0.734 at Re = 100
- **metric**: y/L of the centreline zero crossing
- **measured**: 7.306487536e-1
- **expected**: 7.340000000e-1
- **error**: 3.3512e-3 (abs) — tolerance 3.0000e-2
- _note_: A profile tolerance loose enough to absorb the 65²-vs-129² grid difference cannot detect a misplaced vortex; this can.

### A closed cavity conserves mass exactly — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.cavity.mass_conservation`
- **reference**: Bounce-back and moving-wall boundaries are mass-conserving by construction
- **metric**: |Δm|/m₀ after running to steady state
- **measured**: 5.291964398e-12
- **expected**: 0.000000000e0
- **error**: 5.2920e-12 (abs) — tolerance 1.0000e-9
- _note_: Guards the whole boundary framework composing correctly on one domain.

### Guo forcing injects momentum only along the applied force — PASS

- **crate**: `phyz-lbm`
- **id**: `lbm.forcing.transverse_isotropy`
- **reference**: A uniform force on a periodic domain produces uniform acceleration; any transverse velocity is lattice anisotropy in the source term
- **metric**: max |u_y| after 500 steps of a pure +x body force
- **measured**: 0.000000000e0
- **expected**: 0.000000000e0
- **error**: 0.0000e0 (abs) — tolerance 1.0000e-15
- _note_: Catches a mis-signed or mis-weighted direction in the forcing source term.

## Molecular dynamics — Lennard-Jones (`phyz-md`)

### Velocity Verlet: bounded energy error scales as Δt² — PASS

- **crate**: `phyz-md`
- **id**: `md.verlet_energy_order`
- **reference**: Hairer, Lubich & Wanner, *Geometric Numerical Integration* — a symplectic second-order integrator has an O(Δt²) *bounded* energy error, with no secular term
- **metric**: peak |ΔE|/|E| over 40 reduced time units, LJ dimer, Δt = 0.001 τ
- **measured**: 8.624402237e-5
- **expected**: 0.000000000e0
- **error**: 8.6244e-5 (abs) — tolerance 1.0000e-4
- **convergence in Δt/τ**: measured order p = 2.004 (expected 2.0 ± 0.3) — OK

  | Δt/τ | error | ratio |
  |---:|---:|---:|
  | 8.000000e-3 | 5.565542e-3 | NaN |
  | 4.000000e-3 | 1.382597e-3 | 4.025 |
  | 2.000000e-3 | 3.451110e-4 | 4.006 |
  | 1.000000e-3 | 8.624402e-5 | 4.002 |
- _note_: The dimer never approaches the 2.5σ cutoff, so this isolates the integrator from force-truncation and neighbour-list artefacts.

### Velocity Verlet: no secular energy drift on the LJ dimer — PASS

- **crate**: `phyz-md`
- **id**: `md.verlet_secular_drift`
- **reference**: Symplectic integrators conserve a shadow Hamiltonian exactly, so d⟨E⟩/dt = 0
- **metric**: |d(ΔE/E)/dt| per reduced time unit, Δt = 0.001 τ
- **measured**: 2.040172758e-9
- **expected**: 0.000000000e0
- **error**: 2.0402e-9 (abs) — tolerance 1.0000e-6
- _note_: secular slopes across Δt: Δt=0.0080τ → 1.104e-7, Δt=0.0040τ → 3.097e-8, Δt=0.0020τ → 8.042e-9, Δt=0.0010τ → 2.040e-9

### First integration step is a valid velocity-Verlet step — PASS

- **crate**: `phyz-md`
- **id**: `md.startup_consistency`
- **reference**: A correct velocity-Verlet start-up evaluates a(0) before the first drift, so the local truncation error of step 1 is O(Δt³) like every other step
- **metric**: |ΔE|/|E| across the first step alone, Δt = 0.001 τ, v(0)·a(0) ≠ 0
- **measured**: 2.150589824e-8
- **expected**: 0.000000000e0
- **error**: 2.1506e-8 (abs) — tolerance 1.0000e-6
- **convergence in Δt/τ**: measured order p = 3.011 (expected 3.0 ± 0.3) — OK

  | Δt/τ | error | ratio |
  |---:|---:|---:|
  | 8.000000e-3 | 1.127236e-5 | NaN |
  | 4.000000e-3 | 1.391246e-6 | 8.102 |
  | 2.000000e-3 | 1.726829e-7 | 8.057 |
  | 1.000000e-3 | 2.150590e-8 | 8.030 |
- _note_: `MdSystem::step` primes the force accumulator when `self.step == 0`, so it holds F(x(0)) before the first drift. An earlier revision did not, and this benchmark measured order 0.993 against it.

### LJ fluid g(r): excluded volume inside the repulsive core — PASS

- **crate**: `phyz-md`
- **id**: `md.rdf.core_exclusion`
- **reference**: Verlet, *Phys. Rev.* 165 (1968) 201 — g(r) ≈ 0 for r* < 0.85 at ρ* = 0.8442
- **metric**: max g(r) for r* < 0.8
- **measured**: 0.000000000e0
- **expected**: 0.000000000e0
- **error**: 0.0000e0 (abs) — tolerance 2.0000e-2
- _note_: production average T* = 0.7150 (target 0.722); ε = 0.0103 eV, σ = 3.4 Å

### LJ fluid g(r): first-peak position — PASS

- **crate**: `phyz-md`
- **id**: `md.rdf.first_peak_position`
- **reference**: Verlet (1968); Hansen & McDonald, *Theory of Simple Liquids* Fig. 4.2 — first peak at r* ≈ 1.09 for ρ* = 0.8442, T* = 0.722
- **metric**: r* of the first maximum of g(r)
- **measured**: 1.081659947e0
- **expected**: 1.090000000e0
- **error**: 8.3401e-3 (abs) — tolerance 4.0000e-2

### LJ fluid g(r): first-peak height — PASS

- **crate**: `phyz-md`
- **id**: `md.rdf.first_peak_height`
- **reference**: Verlet (1968) — g(r_max) ≈ 3.0 for ρ* = 0.8442, T* = 0.722
- **metric**: g(r) at the first maximum
- **measured**: 3.012272342e0
- **expected**: 3.000000000e0
- **error**: 4.0908e-3 (rel) — tolerance 1.2000e-1

### LJ fluid g(r): first-minimum position — PASS

- **crate**: `phyz-md`
- **id**: `md.rdf.first_minimum`
- **reference**: Verlet (1968) — first minimum at r* ≈ 1.55
- **metric**: r* of the first minimum of g(r)
- **measured**: 1.551946881e0
- **expected**: 1.550000000e0
- **error**: 1.9469e-3 (abs) — tolerance 6.0000e-2

### LJ fluid g(r): depth of the first minimum — PASS

- **crate**: `phyz-md`
- **id**: `md.rdf.first_minimum_depth`
- **reference**: Verlet (1968) — g ≈ 0.60 at the first minimum
- **metric**: g(r) at the first minimum
- **measured**: 5.695322944e-1
- **expected**: 6.000000000e-1
- **error**: 5.0780e-2 (rel) — tolerance 2.0000e-1
- _note_: measured minimum at r* = 1.552

### LJ fluid excess energy at ρ* = 0.8442, T* = 0.722 — REPORT

- **crate**: `phyz-md`
- **id**: `md.thermo.energy`
- **reference**: Verlet, *Phys. Rev.* 159 (1967) 98, Table II gives U*/N ≈ −5.7 for the *truncated* potential; Johnson, Zollweg & Gubbins, *Mol. Phys.* 78 (1993) 591 give ≈ −5.4 truncated-and-shifted at 2.5σ, which is what this crate implements
- **metric**: ⟨U⟩/N in reduced units
- **measured**: -5.205049962e0
- **expected**: -5.400000000e0
- **error**: 3.6102e-2 (rel) — tolerance 5.0000e-2
- _note_: Reported: the truncation convention shifts the reference by ~5%, more than the statistical error here, so a tight pass/fail claim would be about the convention rather than the solver.

### LJ fluid virial pressure at ρ* = 0.8442, T* = 0.722 — REPORT

- **crate**: `phyz-md`
- **id**: `md.thermo.pressure`
- **reference**: Verlet (1967) Table II — P* ≈ 0.3 at this state point, with the same truncation-convention caveat as the energy
- **metric**: ⟨P⟩ in reduced units
- **measured**: 8.262776527e-1
- **expected**: 3.000000000e-1
- **error**: 5.2628e-1 (abs) — tolerance 3.5000e-1

