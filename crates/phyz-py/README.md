# phyz-py: Python Bindings for phyz Physics Engine

Python bindings for the phyz multi-physics differentiable simulation engine.

## Installation

```bash
cd crates/phyz-py
maturin develop
```

## Usage

```python
import phyz
import numpy as np

# Load model from MJCF
loader = phyz.MjcfLoader("ant.xml")
model = loader.build_model()

# Create state
state = model.default_state()

# Run simulation
sim = phyz.Simulator()
for _ in range(1000):
    sim.step(model, state)

# Access state
print("Position:", state.q)
print("Velocity:", state.v)

# Compute Jacobians
jac = sim.step_with_jacobians(model, state)
print("dq_next/dq:", jac.dqnext_dq)
```

## Features

- NumPy/JAX interoperability
- Automatic differentiation via Jacobians
- MJCF model loading
- Gymnasium environment compatibility
