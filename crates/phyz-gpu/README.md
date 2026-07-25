# phyz-gpu

Batched rigid-body simulation on the GPU, via wgpu.

Runs thousands of independent environments in parallel with one compute
dispatch per timestep — the shape reinforcement learning and sampling-based
control need.

| Type | Purpose |
| --- | --- |
| `GpuBatchSimulator` | N independent worlds sharing one `Model` |
| `GpuSimulator` | single-world GPU stepping |
| `GpuState` | GPU-side state buffers, upload/readback |
| `ContactPipeline` | optional ground-contact penalty pass before ABA |

## Example

```rust,no_run
use phyz_gpu::GpuBatchSimulator;

# fn demo(model: phyz_model::Model, states: Vec<phyz_model::State>) -> Result<(), String> {
let nworld = 1024;
let mut sim = GpuBatchSimulator::new(model.clone(), nworld)?;
sim.enable_ground_contact(0.0, 1.0e4, 5.0e1, 0.6)?;

sim.load_states(&states);
sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);

for _ in 0..500 {
    sim.step();
}

let final_states = sim.readback_states();
# Ok(())
# }
```

Use `with_device_queue` to share an existing `wgpu::Device`/`Queue` with the
rest of your application instead of creating a private one.

## Precision

The compute kernels are **f32**. Agreement with the f64 CPU path is typically
~1e-7 over a few hundred steps; long chaotic rollouts will diverge. Run
`cargo run --release -p phyz-examples --example gpu_batch` for the side-by-side
comparison.

## Requirements

A working wgpu adapter (Metal, Vulkan or DX12). `GpuBatchSimulator::new`
returns `Err` rather than panicking when none is available, so callers can fall
back to the CPU path.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
