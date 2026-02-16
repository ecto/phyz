//! Python bindings for Simulator.

use pyo3::prelude::*;
use tau::Simulator;

use crate::model::PyModel;
use crate::state::PyState;
use crate::jacobians::PyStepJacobians;

/// Python wrapper for tau::Simulator
#[pyclass(name = "Simulator", unsendable)]
pub struct PySimulator {
    inner: Simulator,
}

#[pymethods]
impl PySimulator {
    /// Create a new simulator with semi-implicit Euler solver
    #[new]
    fn new() -> Self {
        Self {
            inner: Simulator::new(),
        }
    }

    /// Create a simulator with RK4 solver
    #[staticmethod]
    fn rk4() -> Self {
        Self {
            inner: Simulator::rk4(),
        }
    }

    /// Advance simulation by one timestep
    fn step(&self, model: &PyModel, state: &mut PyState) {
        self.inner.step(&model.inner, &mut state.inner);
    }

    /// Advance simulation by one timestep and return Jacobians
    fn step_with_jacobians(&self, model: &PyModel, state: &mut PyState) -> PyStepJacobians {
        let jac = self.inner.step_with_jacobians(&model.inner, &mut state.inner);
        PyStepJacobians { inner: jac }
    }

    /// Run simulation for n steps
    fn simulate(&self, model: &PyModel, state: &mut PyState, n: usize) {
        self.inner.simulate(&model.inner, &mut state.inner, n);
    }

    fn __repr__(&self) -> String {
        "Simulator()".to_string()
    }
}
