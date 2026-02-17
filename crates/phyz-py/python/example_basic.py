"""Basic example of using tau Python bindings."""

import tau
import numpy as np

# Build a simple pendulum model
builder = tau.ModelBuilder()
builder = builder.dt(0.01)
builder = builder.add_revolute_body(
    name="pendulum",
    parent=-1,  # attached to world
    pos=[0.0, 0.0, 1.0],  # joint position
    mass=1.0,
    inertia=[0.1, 0.1, 0.01],  # Ixx, Iyy, Izz
)
model = builder.build()

print(f"Model: {model}")
print(f"  nq={model.nq}, nv={model.nv}, nbodies={model.nbodies}")
print(f"  dt={model.dt}, gravity={np.array(model.gravity)}")

# Create state
state = model.default_state()
print(f"\nState: {state}")

# Set initial condition
state.q = np.array([np.pi / 4])  # 45 degrees
state.v = np.array([0.0])

# Run simulation
sim = tau.Simulator()
print("\nSimulating 1000 steps...")
for i in range(1000):
    sim.step(model, state)
    if i % 100 == 0:
        print(f"  t={state.time:.3f}, q={np.array(state.q)[0]:.3f}, v={np.array(state.v)[0]:.3f}")

print("\nDone!")
