//! Contact material properties.
//!
//! The types themselves live in [`phyz_model::material`], because a
//! [`phyz_model::Body`] carries its own [`ContactMaterial`] and the model crate
//! cannot depend on this one. They are re-exported here so
//! `phyz_contact::ContactMaterial` — the path every caller uses — keeps
//! working.

pub use phyz_model::material::{ContactMaterial, SolImp, SolRef};
