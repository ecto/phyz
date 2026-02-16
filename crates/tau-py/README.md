# tau-py: Python Bindings for tau Physics Engine

Python bindings for the tau multi-physics differentiable simulation engine.

## Installation

```bash
cd crates/tau-py
maturin develop
```

## Usage

```python
import tau
import numpy as np

# Load model from MJCF
loader = tau.MjcfLoader("ant.xml")
model = loader.build_model()

# Create state
state = model.default_state()

# Run simulation
sim = tau.Simulator()
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
