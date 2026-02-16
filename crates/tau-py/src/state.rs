//! Python bindings for State.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tau_model::State;

/// Python wrapper for tau::State
#[pyclass(name = "State")]
pub struct PyState {
    pub(crate) inner: State,
}

#[pymethods]
impl PyState {
    /// Generalized positions
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.q.as_slice())
    }

    #[setter]
    fn set_q(&mut self, q: PyReadonlyArray1<f64>) -> PyResult<()> {
        let q_slice = q.as_slice()?;
        if q_slice.len() != self.inner.q.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q dimension mismatch",
            ));
        }
        self.inner.q.copy_from_slice(q_slice);
        Ok(())
    }

    /// Generalized velocities
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.v.as_slice())
    }

    #[setter]
    fn set_v(&mut self, v: PyReadonlyArray1<f64>) -> PyResult<()> {
        let v_slice = v.as_slice()?;
        if v_slice.len() != self.inner.v.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "v dimension mismatch",
            ));
        }
        self.inner.v.copy_from_slice(v_slice);
        Ok(())
    }

    /// Control inputs
    #[getter]
    fn ctrl<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.ctrl.as_slice())
    }

    #[setter]
    fn set_ctrl(&mut self, ctrl: PyReadonlyArray1<f64>) -> PyResult<()> {
        let ctrl_slice = ctrl.as_slice()?;
        if ctrl_slice.len() != self.inner.ctrl.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ctrl dimension mismatch",
            ));
        }
        self.inner.ctrl.copy_from_slice(ctrl_slice);
        Ok(())
    }

    /// Simulation time
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }

    #[setter]
    fn set_time(&mut self, time: f64) {
        self.inner.time = time;
    }

    fn __repr__(&self) -> String {
        format!(
            "State(nq={}, nv={}, time={:.3})",
            self.inner.q.len(),
            self.inner.v.len(),
            self.inner.time
        )
    }

    /// Clone the state
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
