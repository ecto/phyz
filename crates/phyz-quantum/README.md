# phyz-quantum

Hamiltonian lattice gauge theory on simplicial complexes.

Exact-diagonalization-scale quantum simulation of gauge theories.

## Modules

* `hilbert`, `qubit_map` — Hilbert space construction and qubit encodings
* `hamiltonian`, `gauss_law` — the Kogut-Susskind Hamiltonian and its
  physical-sector constraint
* `lanczos`, `diag` — sparse and dense spectrum
* `su2_quantum`, `hypercubic`, `triangulated_torus` — concrete lattices
* `observables`, `traced`, `stabilizer` — measurement, reduced density
  matrices, stabilizer formalism
* `ryu_takayanagi`, `jacobson` — holographic entanglement diagnostics

This crate is research-grade: the API is expected to move.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
