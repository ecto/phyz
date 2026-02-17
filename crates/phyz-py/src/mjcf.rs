//! Python bindings for MJCF loader.

use pyo3::prelude::*;
use phyz_mjcf::MjcfLoader;

use crate::model::PyModel;

/// Python wrapper for phyz::MjcfLoader
#[pyclass(name = "MjcfLoader")]
pub struct PyMjcfLoader {
    inner: MjcfLoader,
}

#[pymethods]
impl PyMjcfLoader {
    /// Load an MJCF XML file
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let loader = MjcfLoader::from_file(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load MJCF: {}", e)))?;
        Ok(Self { inner: loader })
    }

    /// Build the phyz Model from the loaded MJCF
    fn build_model(&self) -> PyResult<PyModel> {
        let model = self.inner.build_model();
        Ok(PyModel { inner: model })
    }

    fn __repr__(&self) -> String {
        "MjcfLoader()".to_string()
    }
}
