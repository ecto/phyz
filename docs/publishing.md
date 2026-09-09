# Publishing phyz

The crates.io idiom: **public crates depend on each other by version**, git
dependencies exist only for crates that are not published yet, and a developer
who wants to build against a local sibling checkout does it in an untracked
`.cargo/config.toml` — never in a committed manifest. phyz already follows the
last part; see `.cargo/config.toml.example`.

## One version for the whole workspace: 0.4.0

crates.io today has `phyz` 0.3.0 and `phyz-{model,math,rigid,contact,diff,world,
collision}` at 0.1.0 — the sub-crates were published before they shared
`[workspace.package] version`. They do share it now, so the whole workspace moves
to **0.4.0** together and **the sub-crates jump 0.1.0 → 0.4.0 in one step**. That
is a large-looking jump with no large-looking break behind it; it is only the
version numbers converging. Downstream pins them as `{ path, version = "0.4" }`.

`phyz-camera` is a first publish and starts at 0.4.0 like everything else.

The bump is in this branch's manifests because that is how phyz cuts a release:
`.github/workflows/release.yml` fires on a `v*.*.*` tag and refuses a tag whose
number does not match `[workspace.package] version`. The version has to land
before the tag.

## Publish order

Automated — `git tag v0.4.0 && git push origin v0.4.0` runs it. The workflow's
order was widened to cover everything kosm consumes:

```
phyz-math -> phyz-model -> phyz-collision -> phyz-rigid -> phyz-contact
          -> phyz-diff  -> phyz-world     -> phyz-camera -> phyz
```

## The tang dependency

`tang`, `tang-la` and `tang-expr` are plain registry deps. `tang-mesh`,
`tang-tensor`, `tang-train`, `tang-safetensors` and `tang-compute` are not on
crates.io yet, so they stay git-with-rev — but each now also carries the version
tang plans to publish (`0.1.0`), so `cargo publish` of a dependent resolves the
moment tang releases. Once tang 0.2.1 / 0.1.0 are on the index, drop the
`git`/`rev` keys and the `[patch.crates-io]` block at the bottom of the root
manifest, and raise the `tang` requirement to `0.2.1`.

Until then `phyz-gpu`, `phyz-env` and `phyz-dream` cannot actually be uploaded —
their tang deps are not on the index. Nothing downstream needs them.

## Not published

| crate | why |
| --- | --- |
| `phyz-bench` | benchmark harness and its recorded results, not a public API |
| `phyz-examples` | a target-only shell that compile-checks `examples/`; ships no library |
| `phyz-home` | the phyz.dev site (Trunk/wasm bundle), not a library crate |
| `phyz-wasm` | a cdylib for the phyz.dev demos, not a library crate |
| `phyz-validate` | validation suites against reference solutions, not a public API |
| `phyz-quantum` | depends on unpublished `tang-mesh` via the optional `mesh` feature |
| `phyz-py` | a pyo3 extension module shipped through PyPI, not crates.io |

## Position in the stack

tang → **phyz** → kosm-render → vcad → kosm.
