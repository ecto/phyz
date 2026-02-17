"""Basic tests for tau Python bindings."""

import pytest
import numpy as np
import tau


def test_version():
    """Test that version string is available."""
    assert hasattr(tau, '__version__')
    assert isinstance(tau.__version__, str)


def test_model_builder():
    """Test ModelBuilder API."""
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

    assert model.nq == 1
    assert model.nv == 1
    assert model.nbodies == 1
    assert model.dt == 0.01


def test_state_creation():
    """Test State creation and access."""
    builder = tau.ModelBuilder()
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 0.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    # Check dimensions
    assert len(state.q) == 1
    assert len(state.v) == 1
    assert len(state.ctrl) == 1

    # Check initial values
    assert state.time == 0.0
    assert np.allclose(state.q, 0.0)
    assert np.allclose(state.v, 0.0)


def test_state_modification():
    """Test State array modification."""
    builder = tau.ModelBuilder()
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 0.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    # Set values
    state.q = np.array([0.5])
    state.v = np.array([1.0])
    state.ctrl = np.array([0.1])
    state.time = 1.0

    # Check values
    assert np.allclose(state.q, 0.5)
    assert np.allclose(state.v, 1.0)
    assert np.allclose(state.ctrl, 0.1)
    assert state.time == 1.0


def test_simulator_step():
    """Test Simulator.step()."""
    builder = tau.ModelBuilder()
    builder = builder.dt(0.01)
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 1.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    # Set initial condition
    state.q = np.array([0.1])
    state.v = np.array([1.0])  # Give it some velocity

    # Step
    sim = tau.Simulator()
    q_before = np.array(state.q).copy()
    sim.step(model, state)

    # State should change
    assert state.time == model.dt
    assert not np.allclose(state.q, q_before)


def test_simulator_rk4():
    """Test RK4 solver."""
    builder = tau.ModelBuilder()
    builder = builder.dt(0.01)
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 1.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    state.q = np.array([0.1])
    state.v = np.array([0.0])

    sim = tau.Simulator.rk4()
    sim.step(model, state)

    assert state.time == model.dt


def test_jacobians():
    """Test step_with_jacobians."""
    builder = tau.ModelBuilder()
    builder = builder.dt(0.01)
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 1.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    state.q = np.array([0.1])
    state.v = np.array([0.2])

    sim = tau.Simulator()
    jac = sim.step_with_jacobians(model, state)

    # Check Jacobian shapes
    assert np.array(jac.dqnext_dq).shape == (1, 1)
    assert np.array(jac.dqnext_dv).shape == (1, 1)
    assert np.array(jac.dvnext_dq).shape == (1, 1)
    assert np.array(jac.dvnext_dv).shape == (1, 1)
    assert np.array(jac.dvnext_dctrl).shape == (1, 1)

    # Jacobians should be non-zero
    assert not np.allclose(jac.dqnext_dv, 0.0)


def test_simulate():
    """Test Simulator.simulate() for multiple steps."""
    builder = tau.ModelBuilder()
    builder = builder.dt(0.01)
    builder = builder.add_revolute_body(
        name="link1",
        parent=-1,
        pos=[0.0, 0.0, 1.0],
        mass=1.0,
        inertia=[0.1, 0.1, 0.01],
    )
    model = builder.build()
    state = model.default_state()

    state.q = np.array([0.1])

    sim = tau.Simulator()
    sim.simulate(model, state, 100)

    assert state.time == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
