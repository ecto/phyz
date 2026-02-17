"""Example of JAX integration with tau."""

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available, install with: pip install jax[cpu]")
    exit(1)

import tau
import numpy as np


def create_model():
    """Create a simple pendulum model."""
    builder = tau.ModelBuilder()
    builder = builder.dt(0.01)
    builder = builder.add_revolute_body(
        name="pendulum",
        parent=-1,
        pos=[0.0, 0.0, 1.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    return builder.build()


def step_fn(q, v, ctrl, model, sim):
    """Single simulation step (for JAX wrapping)."""
    state = model.default_state()
    state.q = q
    state.v = v
    state.ctrl = ctrl
    sim.step(model, state)
    return np.array(state.q), np.array(state.v)


if __name__ == "__main__":
    model = create_model()
    sim = tau.Simulator()

    # Initial conditions
    q0 = np.array([np.pi / 4])
    v0 = np.array([0.0])
    ctrl = np.array([0.0])

    print("Running single step...")
    q1, v1 = step_fn(q0, v0, ctrl, model, sim)
    print(f"  q: {q0} -> {q1}")
    print(f"  v: {v0} -> {v1}")

    # Note: Full JAX JIT compilation would require pure JAX reimplementation
    # of the physics. This example shows NumPy array compatibility.
    print("\nNote: For full JAX autodiff, consider reimplementing step_fn in pure JAX")
    print("using the Jacobians provided by tau:")

    state = model.default_state()
    state.q = q0
    state.v = v0
    jac = sim.step_with_jacobians(model, state)

    print(f"\nJacobians from tau:")
    print(f"  dq_next/dq shape: {np.array(jac.dqnext_dq).shape}")
    print(f"  dv_next/dv shape: {np.array(jac.dvnext_dv).shape}")
