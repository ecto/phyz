pub mod data;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod model;
pub mod train;
pub mod validate;

pub use data::{SampleConfig, generate_dataset, input_dim};
pub use model::{DreamMeta, DreamModel, NormStats};
pub use train::{TrainConfig, train};
pub use validate::{ValidationResult, validate};

#[cfg(feature = "gpu")]
pub use gpu::GpuDreamModel;

#[cfg(test)]
pub(crate) mod tests {
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    pub fn make_double_pendulum() -> phyz_model::Model {
        let length = 1.0;
        let mass = 1.0;
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0)),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build()
    }
}
