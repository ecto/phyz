//! Python bindings for phyz physics engine.

use pyo3::prelude::*;

mod model;
mod simulator;
mod state;
mod mjcf;
mod jacobians;

use model::{PyModel, PyModelBuilder};
use simulator::PySimulator;
use state::PyState;
use mjcf::PyMjcfLoader;
use jacobians::PyStepJacobians;

/// phyz - Multi-physics differentiable simulation engine
#[pymodule]
fn phyz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyModelBuilder>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PySimulator>()?;
    m.add_class::<PyMjcfLoader>()?;
    m.add_class::<PyStepJacobians>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
