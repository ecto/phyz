//! Python bindings for StepJacobians.

use numpy::PyArray2;
use pyo3::prelude::*;
use phyz_diff::StepJacobians;

/// Python wrapper for phyz::StepJacobians
#[pyclass(name = "StepJacobians")]
pub struct PyStepJacobians {
    pub(crate) inner: StepJacobians,
}

#[pymethods]
impl PyStepJacobians {
    /// ∂q_{t+1}/∂q_t
    #[getter]
    fn dqnext_dq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nq = self.inner.dqnext_dq.nrows();
        let data: Vec<f64> = self.inner.dqnext_dq.as_slice().to_vec();
        PyArray2::from_vec2(
            py,
            &(0..nq)
                .map(|i| data[i * nq..(i + 1) * nq].to_vec())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to create array")
    }

    /// ∂q_{t+1}/∂v_t
    #[getter]
    fn dqnext_dv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nq = self.inner.dqnext_dv.nrows();
        let nv = self.inner.dqnext_dv.ncols();
        let data: Vec<f64> = self.inner.dqnext_dv.as_slice().to_vec();
        PyArray2::from_vec2(
            py,
            &(0..nq)
                .map(|i| data[i * nv..(i + 1) * nv].to_vec())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to create array")
    }

    /// ∂q_{t+1}/∂ctrl_t (Note: phyz uses dvnext_dctrl only, this returns zeros)
    #[getter]
    fn dqnext_dctrl<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nq = self.inner.dqnext_dq.nrows();
        let nv = self.inner.dvnext_dctrl.ncols();
        // For semi-implicit Euler, dq_next/dctrl = 0 (control affects v directly)
        PyArray2::zeros(py, [nq, nv], false)
    }

    /// ∂v_{t+1}/∂q_t
    #[getter]
    fn dvnext_dq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nv = self.inner.dvnext_dq.nrows();
        let nq = self.inner.dvnext_dq.ncols();
        let data: Vec<f64> = self.inner.dvnext_dq.as_slice().to_vec();
        PyArray2::from_vec2(
            py,
            &(0..nv)
                .map(|i| data[i * nq..(i + 1) * nq].to_vec())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to create array")
    }

    /// ∂v_{t+1}/∂v_t
    #[getter]
    fn dvnext_dv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nv = self.inner.dvnext_dv.nrows();
        let data: Vec<f64> = self.inner.dvnext_dv.as_slice().to_vec();
        PyArray2::from_vec2(
            py,
            &(0..nv)
                .map(|i| data[i * nv..(i + 1) * nv].to_vec())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to create array")
    }

    /// ∂v_{t+1}/∂ctrl_t
    #[getter]
    fn dvnext_dctrl<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nv = self.inner.dvnext_dctrl.nrows();
        let data: Vec<f64> = self.inner.dvnext_dctrl.as_slice().to_vec();
        PyArray2::from_vec2(
            py,
            &(0..nv)
                .map(|i| data[i * nv..(i + 1) * nv].to_vec())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to create array")
    }

    fn __repr__(&self) -> String {
        format!(
            "StepJacobians(nq={}, nv={})",
            self.inner.dqnext_dq.nrows(),
            self.inner.dvnext_dv.nrows()
        )
    }
}
