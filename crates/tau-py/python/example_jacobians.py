"""Example of computing Jacobians with tau."""

import tau
import numpy as np

# Build simple model
builder = tau.ModelBuilder()
builder = builder.dt(0.01)
builder = builder.add_revolute_body(
    name="link1",
    parent=-1,
    pos=[0.0, 0.0, 0.0],
    mass=1.0,
    inertia=[0.1, 0.1, 0.01],
)
model = builder.build()

# Create state
state = model.default_state()
state.q = np.array([0.1])
state.v = np.array([0.2])

# Compute Jacobians
sim = tau.Simulator()
jac = sim.step_with_jacobians(model, state)

print("Step Jacobians:")
print(f"  dq_next/dq:\n{np.array(jac.dqnext_dq)}")
print(f"  dq_next/dv:\n{np.array(jac.dqnext_dv)}")
print(f"  dv_next/dq:\n{np.array(jac.dvnext_dq)}")
print(f"  dv_next/dv:\n{np.array(jac.dvnext_dv)}")
print(f"  dv_next/dctrl:\n{np.array(jac.dvnext_dctrl)}")

print(f"\nState after step: q={np.array(state.q)}, v={np.array(state.v)}")
