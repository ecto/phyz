//! Python bindings for Model.

use numpy::PyArray1;
use pyo3::prelude::*;
use tau_model::{Model, ModelBuilder};
use tau_math::{SpatialInertia, SpatialTransform, Vec3};

use crate::state::PyState;

/// Python wrapper for tau::Model
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Model,
}

#[pymethods]
impl PyModel {
    /// Create a default state for this model
    fn default_state(&self) -> PyResult<PyState> {
        Ok(PyState {
            inner: self.inner.default_state(),
        })
    }

    /// Number of position DOFs
    #[getter]
    fn nq(&self) -> usize {
        self.inner.nq
    }

    /// Number of velocity DOFs
    #[getter]
    fn nv(&self) -> usize {
        self.inner.nv
    }

    /// Number of bodies
    #[getter]
    fn nbodies(&self) -> usize {
        self.inner.nbodies()
    }

    /// Integration timestep
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt
    }

    /// Gravity vector [x, y, z]
    #[getter]
    fn gravity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &[
            self.inner.gravity.x,
            self.inner.gravity.y,
            self.inner.gravity.z,
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(nq={}, nv={}, nbodies={}, dt={})",
            self.inner.nq, self.inner.nv, self.inner.nbodies(), self.inner.dt
        )
    }
}

/// Python wrapper for ModelBuilder
#[pyclass(name = "ModelBuilder")]
pub(crate) struct PyModelBuilder {
    inner: ModelBuilder,
}

#[pymethods]
impl PyModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: ModelBuilder::new(),
        }
    }

    /// Set the gravity vector [x, y, z]
    fn gravity(mut slf: PyRefMut<'_, Self>, g: [f64; 3]) -> PyRefMut<'_, Self> {
        let builder = std::mem::replace(&mut slf.inner, ModelBuilder::new());
        slf.inner = builder.gravity(Vec3::new(g[0], g[1], g[2]));
        slf
    }

    /// Set the timestep
    fn dt(mut slf: PyRefMut<'_, Self>, dt: f64) -> PyRefMut<'_, Self> {
        let builder = std::mem::replace(&mut slf.inner, ModelBuilder::new());
        slf.inner = builder.dt(dt);
        slf
    }

    /// Add a revolute body
    ///
    /// Args:
    ///     name: Body name
    ///     parent: Parent body index (-1 for world)
    ///     pos: Position [x, y, z] of joint in parent frame
    ///     mass: Body mass
    ///     inertia: Inertia matrix diagonal [Ixx, Iyy, Izz]
    fn add_revolute_body(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        parent: i32,
        pos: [f64; 3],
        mass: f64,
        inertia: [f64; 3],
    ) -> PyRefMut<'_, Self> {
        let xform = SpatialTransform::translation(Vec3::new(pos[0], pos[1], pos[2]));
        let si = SpatialInertia::new(
            mass,
            Vec3::zeros(),
            nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(
                inertia[0], inertia[1], inertia[2],
            )),
        );
        let builder = std::mem::replace(&mut slf.inner, ModelBuilder::new());
        slf.inner = builder.add_revolute_body(&name, parent, xform, si);
        slf
    }

    /// Build the model
    fn build(mut slf: PyRefMut<'_, Self>) -> PyModel {
        let builder = std::mem::replace(&mut slf.inner, ModelBuilder::new());
        PyModel {
            inner: builder.build(),
        }
    }

    fn __repr__(&self) -> String {
        "ModelBuilder()".to_string()
    }
}
