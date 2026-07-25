# phyz-validate

Closed-form physics benchmarks for every phyz solver.

`phyz-regge` and `phyz-quantum` already validate analytic gradients against
finite differences across ~120 tests. This crate holds the classical solvers to
the same standard: each benchmark integrates a solver on a problem with a known
analytic solution or published reference data, reports a **quantitative error
metric**, and — wherever a discretization parameter exists — demonstrates the
**expected convergence order** under refinement.

```bash
cargo run --release -p phyz-validate            # console + target/validation/*.{md,json}
cargo run --release -p phyz-validate -- out/    # choose the output directory
cargo test --release -p phyz-validate -- --ignored   # regression guard
```

The binary exits non-zero when any benchmark fails.

## Ground rules

- **Tolerances are declared before the measurement.** They are derived from
  theory (e.g. `10(2πΔt/T)²` for the bounded energy error of a second-order
  symplectic integrator) or from the spread of the published reference, never
  fitted to the number the solver happens to produce.
- **A `FAIL` is a finding about the solver**, not about the benchmark. Failures
  are recorded in `tests/validation.rs::KNOWN_FAILURES` with a one-line
  diagnosis, and each entry's `notes` cite the file and lines responsible.
- **Convergence order beats a single tolerance.** Halving `Δt` or `Δx` must
  reduce the error by the predicted factor. A wrong force law shows up as an
  error that refuses to shrink under refinement, which a single-tolerance
  assertion cannot distinguish from an under-resolved run.

## What is covered

| Suite | Benchmarks |
|---|---|
| `rigid` | pendulum period vs the elliptic-integral solution at 5–150°, RK4 fourth-order convergence, total-energy conservation, fast-top precession `Ω = mgl/(I₃ω₃)` and its `ω₃⁻²` approach |
| `gravity` | Kepler energy / angular momentum / Laplace–Runge–Lenz conservation with `Δt²` scaling; Mercury perihelion precession from *integrated* 1PN equations of motion |
| `em` | numerical dispersion vs the analytic Yee relation, `Δx²` phase-velocity convergence, TM₁₁₀ cavity resonance, measured absorbing-boundary reflection coefficient |
| `lbm` | Poiseuille profile vs `u(y) = Fy(H−y)/2ρν` and its viscosity-independence across BGK/TRT/MRT, Taylor–Green decay rate and field shape, lid-driven cavity and primary-vortex position vs Ghia, Ghia & Shin (1982), mass conservation, forcing isotropy |
| `md` | velocity-Verlet bounded energy error and secular drift, start-up consistency, Lennard-Jones `g(r)` and `⟨U⟩/N` at ρ\* = 0.8442, T\* = 0.722 |

The LBM closed-form solutions and the Ghia reference tables are taken from
[`phyz_lbm::analytic`](../phyz-lbm/src/analytic.rs) rather than restated here,
so this suite and `phyz-lbm`'s own `tests/validation.rs` are driven by one set
of formulas.

## Adding a benchmark

Return a [`Validation`](src/report.rs) from a suite function. The error and
status are computed from `(measured, expected, error_kind, tolerance)`; attach a
`Convergence::fit` when a refinement parameter exists, and use `.note()` to
record anything a reader of the published report would need — including
implementation defects the benchmark exposes.
