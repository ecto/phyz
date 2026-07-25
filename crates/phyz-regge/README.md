# phyz-regge

4D Regge calculus with a U(1) gauge field.

Discrete general relativity: spacetime as a simplicial complex, curvature as
deficit angles around triangles, and an Einstein-Maxwell action over edge
lengths and gauge links.

## Modules

* `complex`, `mesh` — simplicial complexes, incidence, 4-simplices
* `geometry` — simplex volumes, dihedral angles, deficit angles
* `action` — the Regge action `S[l, θ] = S_R[l] + α·S_M[l, θ]`
* `gauge`, `gem` — U(1) links, plaquettes, gravito-electromagnetism
* `regge`, `lorentzian_regge`, `lorentzian`, `foliation` — Euclidean and
  Lorentzian evolution
* `solver`, `search`, `richardson` — solving and continuum extrapolation
* `matter`, `symmetry` — matter coupling and symmetry diagnostics

This crate is research-grade: the API is expected to move.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
