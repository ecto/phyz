//! Sensor models for extracting observations from simulation state.
//!
//! Every sensor here reads real simulation quantities. Sensors that need the
//! world around them (rangefinders, contact sensors) go through a
//! [`SensorContext`], which carries the collision scene along with a single
//! shared kinematics/dynamics pass so that reading N sensors does not run
//! forward kinematics N times.

use crate::scene::{PlacedShape, Scene, ShapeOwner, placed_shapes};
use phyz_collision::{Ray, gjk_distance_rot, ray_intersect};
use phyz_math::{Mat3, SpatialTransform, SpatialVec, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{BodyKinematics, aba, body_wrenches, forward_kinematics_acc};

/// Sensor types for extracting observations from simulation.
#[derive(Debug, Clone)]
pub enum Sensor {
    /// Joint position and velocity sensor. Output: `q` then `v`, `2 × ndof`.
    JointState { joint_idx: usize },
    /// Classical linear acceleration of the body origin, in world coordinates.
    ///
    /// This is a *kinematic* acceleration: a body in free fall reads `g`
    /// downward, not zero. For what an accelerometer would report, use
    /// [`Sensor::Imu`].
    BodyAccel { body_idx: usize },
    /// Body angular velocity in the body frame.
    BodyAngularVel { body_idx: usize },
    /// Wrench transmitted through the body's joint, in the body frame.
    /// Output: `[τx, τy, τz, fx, fy, fz]`.
    ForceTorque { body_idx: usize },
    /// Rangefinder: distance along a ray to the nearest geometry.
    ///
    /// `origin` and `direction` are given in the sensor body's frame, so the
    /// ray follows the body as it moves. Output: `[distance]`, clamped to
    /// `max_dist` when nothing is in range.
    Rangefinder {
        body_idx: usize,
        /// Ray origin in body coordinates.
        origin: Vec3,
        /// Ray direction in body coordinates (need not be normalized).
        direction: Vec3,
        /// Maximum range; also the reading when nothing is hit.
        max_dist: f64,
    },
    /// Contact sensor: whether this body touches anything else, and how deeply.
    ///
    /// Output: `[count, penetration_depth, nx, ny, nz]`, where the normal
    /// points away from the deepest contacting shape. Depth is positive when
    /// overlapping and negative inside the proximity margin; it is `NaN` if the
    /// shapes overlap but EPA could not resolve a depth, so an unknown value
    /// can never be mistaken for a real reading.
    Contact {
        body_idx: usize,
        /// Separation below which a pair counts as touching. Zero means "only
        /// actual overlap"; a small positive value gives a proximity band.
        margin: f64,
    },
    /// IMU: specific force (what an accelerometer reads, i.e. `a − g`) followed
    /// by angular velocity, both in the body frame.
    Imu { body_idx: usize },
    /// Snapshot of body transform: `[x, y, z, qw, qx, qy, qz]` in world frame.
    FrameCapture { body_idx: usize },
}

/// Output from a sensor reading.
#[derive(Debug, Clone)]
pub struct SensorOutput {
    /// Sensor identifier (index in sensor array).
    pub sensor_id: usize,
    /// Simulation time when reading was taken.
    pub timestamp: f64,
    /// Flattened sensor data.
    pub data: Vec<f64>,
}

/// Everything a batch of sensors needs, computed once.
///
/// Building this runs forward kinematics, forward dynamics (ABA), and the
/// inverse-dynamics wrench pass, so construct one per timestep and share it
/// across all sensors rather than one per sensor.
pub struct SensorContext<'a> {
    /// The model being observed.
    pub model: &'a Model,
    /// The state being observed.
    pub state: &'a State,
    /// Static geometry the sensors can see.
    pub scene: &'a Scene,
    kinematics: BodyKinematics,
    wrenches: Vec<SpatialVec>,
}

impl<'a> SensorContext<'a> {
    /// Build a context for the given state and scene.
    pub fn new(model: &'a Model, state: &'a State, scene: &'a Scene) -> Self {
        // Accelerations and reaction forces both need the true qdd, which is
        // whatever the current controls and constraints produce right now.
        let qdd = aba(model, state);
        Self {
            model,
            state,
            scene,
            kinematics: forward_kinematics_acc(model, state, &qdd),
            wrenches: body_wrenches(model, state, &qdd),
        }
    }

    /// World→body transforms for this state.
    pub fn xforms(&self) -> &[SpatialTransform] {
        &self.kinematics.xforms
    }

    /// Body-frame spatial velocities.
    pub fn velocities(&self) -> &[SpatialVec] {
        &self.kinematics.velocities
    }

    /// Rotation taking body-`i` coordinates into the world frame.
    fn body_to_world(&self, i: usize) -> Mat3 {
        self.kinematics.xforms[i].rot.transpose()
    }

    /// Every collision shape in the world except those on `exclude`.
    fn shapes(&self, exclude: Option<usize>) -> Vec<PlacedShape> {
        placed_shapes(self.model, &self.kinematics.xforms, self.scene, exclude)
    }
}

/// A rangefinder hit, for callers that want more than the distance.
#[derive(Debug, Clone, Copy)]
pub struct RangeHit {
    /// Distance from the ray origin.
    pub distance: f64,
    /// What was hit.
    pub owner: ShapeOwner,
    /// World-space contact point.
    pub point: Vec3,
    /// World-space surface normal.
    pub normal: Vec3,
}

impl Sensor {
    /// Read this sensor.
    ///
    /// Sensors that do not need the scene ignore it; see [`Sensor::read_state`]
    /// for a convenience wrapper.
    pub fn read(&self, ctx: &SensorContext, sensor_id: usize) -> SensorOutput {
        let model = ctx.model;
        let state = ctx.state;

        let data = match self {
            Sensor::JointState { joint_idx } => {
                let q_off = model.q_offsets[*joint_idx];
                let v_off = model.v_offsets[*joint_idx];
                let ndof = model.joints[*joint_idx].ndof();

                let mut out = Vec::with_capacity(ndof * 2);
                out.extend((0..ndof).map(|i| state.q[q_off + i]));
                out.extend((0..ndof).map(|i| state.v[v_off + i]));
                out
            }

            Sensor::BodyAccel { body_idx } => {
                let i = check_body(model, *body_idx);
                // Classical acceleration is computed in the body frame, then
                // rotated out to world.
                let a_body = ctx.kinematics.classical_linear_accel(i);
                let a = ctx.body_to_world(i).mul_vec(a_body);
                vec![a.x, a.y, a.z]
            }

            Sensor::BodyAngularVel { body_idx } => {
                let i = check_body(model, *body_idx);
                let w = ctx.kinematics.velocities[i].angular;
                vec![w.x, w.y, w.z]
            }

            Sensor::ForceTorque { body_idx } => {
                let i = check_body(model, *body_idx);
                let f = ctx.wrenches[i];
                vec![
                    f.angular.x,
                    f.angular.y,
                    f.angular.z,
                    f.linear.x,
                    f.linear.y,
                    f.linear.z,
                ]
            }

            Sensor::Rangefinder {
                body_idx,
                origin,
                direction,
                max_dist,
            } => {
                let hit = self.cast(ctx, *body_idx, origin, direction, *max_dist);
                vec![hit.map_or(*max_dist, |h| h.distance)]
            }

            Sensor::Contact { body_idx, margin } => {
                let i = check_body(model, *body_idx);
                contact_reading(ctx, i, *margin)
            }

            Sensor::Imu { body_idx } => {
                let i = check_body(model, *body_idx);
                // An accelerometer measures specific force: proper acceleration
                // minus gravity, in the sensor's own frame. At rest this reads
                // +9.81 m/s² "up", which is the classic sign trap.
                let world_to_body = ctx.kinematics.xforms[i].rot;
                let a_body = ctx.kinematics.classical_linear_accel(i);
                let g_body = world_to_body.mul_vec(model.gravity);
                let f = a_body - g_body;
                let w = ctx.kinematics.velocities[i].angular;
                vec![f.x, f.y, f.z, w.x, w.y, w.z]
            }

            Sensor::FrameCapture { body_idx } => {
                let i = check_body(model, *body_idx);
                let xf = &ctx.kinematics.xforms[i];
                // `xf.rot` maps world→body; the body's orientation in the world
                // frame is its transpose.
                let (w, x, y, z) = mat3_to_quat(&xf.rot.transpose());
                vec![xf.pos.x, xf.pos.y, xf.pos.z, w, x, y, z]
            }
        };

        SensorOutput {
            sensor_id,
            timestamp: state.time,
            data,
        }
    }

    /// Read a sensor without any static scene geometry.
    ///
    /// Rangefinders and contact sensors still see the model's own bodies, but
    /// nothing else. Building the context is not free, so prefer
    /// [`SensorContext`] plus [`Sensor::read`] when reading several sensors.
    pub fn read_state(&self, model: &Model, state: &State, sensor_id: usize) -> SensorOutput {
        let scene = Scene::empty();
        self.read(&SensorContext::new(model, state, &scene), sensor_id)
    }

    /// Full rangefinder result, including what was hit and where.
    ///
    /// Returns `None` if nothing is within range. Panics if this is not a
    /// [`Sensor::Rangefinder`].
    pub fn cast_ray(&self, ctx: &SensorContext) -> Option<RangeHit> {
        let Sensor::Rangefinder {
            body_idx,
            origin,
            direction,
            max_dist,
        } = self
        else {
            panic!("cast_ray() is only valid on Sensor::Rangefinder, not {self:?}");
        };
        self.cast(ctx, *body_idx, origin, direction, *max_dist)
    }

    fn cast(
        &self,
        ctx: &SensorContext,
        body_idx: usize,
        origin: &Vec3,
        direction: &Vec3,
        max_dist: f64,
    ) -> Option<RangeHit> {
        let i = check_body(ctx.model, body_idx);

        // Lift the body-frame ray into world coordinates.
        let rot = ctx.body_to_world(i);
        let world_origin = ctx.kinematics.xforms[i].pos + rot.mul_vec(*origin);
        let world_dir = rot.mul_vec(*direction);

        let Some(ray) = Ray::new(world_origin, world_dir) else {
            panic!(
                "Sensor::Rangefinder on body {body_idx} has a zero-length direction; \
                 a rangefinder with no direction has no meaningful reading"
            );
        };

        let mut best: Option<RangeHit> = None;
        // The sensor's own body is excluded so it does not range-find itself.
        for shape in ctx.shapes(Some(i)) {
            let Some(hit) = ray_intersect(&shape.geometry, &shape.pos, &shape.rot, &ray) else {
                continue;
            };
            if hit.distance > max_dist {
                continue;
            }
            if best.is_none_or(|b| hit.distance < b.distance) {
                best = Some(RangeHit {
                    distance: hit.distance,
                    owner: shape.owner,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        best
    }

    /// Get expected output dimension for this sensor.
    ///
    /// For [`Sensor::JointState`] this is per-DOF: the actual reading has
    /// `2 × ndof` entries.
    pub fn output_dim(&self) -> usize {
        match self {
            Sensor::JointState { .. } => 2, // q + v (multiplied by ndof at runtime)
            Sensor::BodyAccel { .. } => 3,
            Sensor::BodyAngularVel { .. } => 3,
            Sensor::ForceTorque { .. } => 6,
            Sensor::Rangefinder { .. } => 1,
            Sensor::Contact { .. } => 5,      // count + depth + normal
            Sensor::Imu { .. } => 6,          // accel (3) + gyro (3)
            Sensor::FrameCapture { .. } => 7, // pos (3) + quat (4)
        }
    }
}

/// Validate a body index up front rather than quietly reading zeros.
///
/// A sensor pointed at a body that does not exist is a configuration bug, and
/// an RL policy trained on the resulting zeros would be silently wrong.
fn check_body(model: &Model, body_idx: usize) -> usize {
    assert!(
        body_idx < model.nbodies(),
        "sensor references body {body_idx}, but the model only has {} bodies",
        model.nbodies()
    );
    body_idx
}

/// `[count, depth, nx, ny, nz]` for the shapes touching body `i`.
fn contact_reading(ctx: &SensorContext, i: usize, margin: f64) -> Vec<f64> {
    let own: Vec<PlacedShape> = ctx
        .shapes(None)
        .into_iter()
        .filter(|s| s.owner == ShapeOwner::Body(i))
        .collect();
    if own.is_empty() {
        // No geometry means nothing can be touched; that is a real answer, not
        // a placeholder.
        return vec![0.0, 0.0, 0.0, 0.0, 0.0];
    }

    let mut count = 0.0;
    let mut deepest: f64 = 0.0;
    let mut normal = Vec3::zeros();

    for other in ctx.shapes(Some(i)) {
        // Skip the body's own parent/child? No: adjacent links genuinely do
        // touch, and filtering that is the contact pipeline's job, not the
        // sensor's.
        for mine in &own {
            let sep = gjk_distance_rot(
                &mine.geometry,
                &other.geometry,
                &mine.pos,
                &other.pos,
                &mine.rot,
                &other.rot,
            );
            if sep > margin {
                continue;
            }
            count += 1.0;

            let axis = (other.pos - mine.pos).normalize();
            let (depth, dir) = if sep > 0.0 {
                // Inside the proximity margin but not actually overlapping:
                // the separation itself is the (negative) depth.
                (-sep, axis)
            } else {
                // Overlapping. EPA gives the exact depth and direction; if the
                // polytope is degenerate it reports NaN rather than a plausible
                // zero, so a caller cannot mistake "unknown" for "just touching".
                match phyz_collision::epa_penetration_rot(
                    &mine.geometry,
                    &other.geometry,
                    &mine.pos,
                    &other.pos,
                    &mine.rot,
                    &other.rot,
                ) {
                    Some((d, n)) => (d, n),
                    None => (f64::NAN, axis),
                }
            };
            // `NaN >= x` is false, so an unresolved depth would silently lose
            // to a resolved one; propagate it explicitly instead.
            if depth.is_nan() || (!deepest.is_nan() && depth >= deepest) {
                deepest = depth;
                normal = dir;
            }
        }
    }

    vec![count, deepest, normal.x, normal.y, normal.z]
}

/// Helper function to convert rotation matrix to quaternion.
/// Returns (w, x, y, z).
fn mat3_to_quat(mat: &phyz_math::Mat3) -> (f64, f64, f64, f64) {
    // Shepperd's method for numerical stability
    let trace = mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let y = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let z = (mat[(1, 0)] - mat[(0, 1)]) / s;
        (w, x, y, z)
    } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
        let s = (1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let x = 0.25 * s;
        let y = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let z = (mat[(0, 2)] + mat[(2, 0)]) / s;
        (w, x, y, z)
    } else if mat[(1, 1)] > mat[(2, 2)] {
        let s = (1.0 + mat[(1, 1)] - mat[(0, 0)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let x = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let y = 0.25 * s;
        let z = (mat[(1, 2)] + mat[(2, 1)]) / s;
        (w, x, y, z)
    } else {
        let s = (1.0 + mat[(2, 2)] - mat[(0, 0)] - mat[(1, 1)]).sqrt() * 2.0;
        let w = (mat[(1, 0)] - mat[(0, 1)]) / s;
        let x = (mat[(0, 2)] + mat[(2, 0)]) / s;
        let y = (mat[(1, 2)] + mat[(2, 1)]) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Obstacle;
    use phyz_collision::Geometry as CGeom;
    use phyz_math::{SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    fn one_link() -> Model {
        ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build()
    }

    #[test]
    fn test_joint_state_sensor() {
        let model = one_link();
        let mut state = model.default_state();
        state.q[0] = 0.5;
        state.v[0] = 1.0;

        let output = Sensor::JointState { joint_idx: 0 }.read_state(&model, &state, 0);

        assert_eq!(output.data.len(), 2);
        assert_eq!(output.data[0], 0.5);
        assert_eq!(output.data[1], 1.0);
    }

    #[test]
    fn test_body_angular_vel_sensor() {
        let model = one_link();
        let mut state = model.default_state();
        state.v[0] = 2.0;

        let output = Sensor::BodyAngularVel { body_idx: 0 }.read_state(&model, &state, 0);

        assert_eq!(output.data.len(), 3);
        // The joint spins about Z at 2 rad/s.
        assert!((output.data[2] - 2.0).abs() < 1e-10);
        assert!(output.data[0].abs() < 1e-10 && output.data[1].abs() < 1e-10);
    }

    #[test]
    fn test_imu_sensor_reads_gravity_at_rest() {
        // A body held at rest by a locked joint measures +g upward, the
        // textbook accelerometer reading — not zero.
        let model = ModelBuilder::new()
            .add_fixed_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::zeros()),
            )
            .build();
        let state = model.default_state();

        let output = Sensor::Imu { body_idx: 0 }.read_state(&model, &state, 0);
        assert_eq!(output.data.len(), 6);
        assert!(
            (output.data[2] - 9.81).abs() < 1e-6,
            "specific force z = {}",
            output.data[2]
        );
        assert!(output.data[3..].iter().all(|w| w.abs() < 1e-12));
    }

    #[test]
    fn test_frame_capture_sensor() {
        let model = one_link();
        let state = model.default_state();
        let output = Sensor::FrameCapture { body_idx: 0 }.read_state(&model, &state, 0);

        assert_eq!(output.data.len(), 7); // 3 pos + 4 quat
        assert!((output.data[3] - 1.0).abs() < 1e-12, "identity quaternion");
    }

    #[test]
    fn test_frame_capture_reports_world_orientation() {
        // Rotating the joint by +90° about Z must give a quaternion with a
        // positive Z component. Reporting the world→body transform instead
        // would flip the sign.
        let model = one_link();
        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_2;

        let output = Sensor::FrameCapture { body_idx: 0 }.read_state(&model, &state, 0);
        let (w, z) = (output.data[3], output.data[6]);
        let half = std::f64::consts::FRAC_PI_4;
        assert!((w - half.cos()).abs() < 1e-9, "qw = {w}");
        assert!((z - half.sin()).abs() < 1e-9, "qz = {z}");
    }

    #[test]
    fn rangefinder_measures_known_sphere_distance() {
        // Sensor body at the origin, sphere of radius 1 centred 5 m along +X:
        // the surface is exactly 4 m away.
        let model = one_link();
        let state = model.default_state();
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(5.0, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);

        let sensor = Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_dist: 10.0,
        };
        let out = sensor.read(&ctx, 0);
        assert_eq!(out.data.len(), 1);
        assert!((out.data[0] - 4.0).abs() < 1e-9, "got {}", out.data[0]);

        let hit = sensor.cast_ray(&ctx).unwrap();
        assert_eq!(hit.owner, ShapeOwner::Obstacle(0));
        assert!((hit.point.x - 4.0).abs() < 1e-9);
    }

    #[test]
    fn rangefinder_returns_max_dist_when_nothing_in_range() {
        let model = one_link();
        let state = model.default_state();
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(50.0, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);

        let sensor = Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_dist: 10.0,
        };
        assert_eq!(sensor.read(&ctx, 0).data[0], 10.0);
        assert!(sensor.cast_ray(&ctx).is_none());
    }

    #[test]
    fn rangefinder_ray_follows_the_body() {
        // The ray is specified in body coordinates, so rotating the body by 90°
        // about Z must swing a +X ray onto the +Y target.
        let model = one_link();
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(0.0, 5.0, 0.0),
        ));
        let sensor = Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_dist: 10.0,
        };

        let state = model.default_state();
        let ctx = SensorContext::new(&model, &state, &scene);
        assert_eq!(sensor.read(&ctx, 0).data[0], 10.0, "ray points away");

        let mut turned = model.default_state();
        turned.q[0] = std::f64::consts::FRAC_PI_2;
        let ctx = SensorContext::new(&model, &turned, &scene);
        assert!(
            (sensor.read(&ctx, 0).data[0] - 4.0).abs() < 1e-9,
            "got {}",
            sensor.read(&ctx, 0).data[0]
        );
    }

    #[test]
    fn rangefinder_sees_the_ground_plane() {
        let model = one_link();
        let state = model.default_state();
        let scene = Scene::empty().with_ground(-2.0);
        let ctx = SensorContext::new(&model, &state, &scene);

        let out = Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::new(0.0, 0.0, -1.0),
            max_dist: 10.0,
        }
        .read(&ctx, 0);
        assert!((out.data[0] - 2.0).abs() < 1e-9, "got {}", out.data[0]);
    }

    #[test]
    fn rangefinder_ignores_its_own_body() {
        // A body wrapped in its own collision sphere must not report 0.
        let mut model = one_link();
        model.bodies[0].geometry = Some(phyz_model::Geometry::Sphere { radius: 0.2 });
        let state = model.default_state();
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(5.0, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);

        let out = Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_dist: 10.0,
        }
        .read(&ctx, 0);
        assert!((out.data[0] - 4.0).abs() < 1e-9, "got {}", out.data[0]);
    }

    #[test]
    #[should_panic(expected = "zero-length direction")]
    fn rangefinder_with_no_direction_fails_loudly() {
        let model = one_link();
        let state = model.default_state();
        let scene = Scene::empty();
        let ctx = SensorContext::new(&model, &state, &scene);
        Sensor::Rangefinder {
            body_idx: 0,
            origin: Vec3::zeros(),
            direction: Vec3::zeros(),
            max_dist: 1.0,
        }
        .read(&ctx, 0);
    }

    #[test]
    #[should_panic(expected = "only has 1 bodies")]
    fn out_of_range_body_index_fails_loudly() {
        let model = one_link();
        let state = model.default_state();
        Sensor::BodyAngularVel { body_idx: 7 }.read_state(&model, &state, 0);
    }

    #[test]
    fn contact_sensor_detects_overlap() {
        let mut model = one_link();
        model.bodies[0].geometry = Some(phyz_model::Geometry::Sphere { radius: 1.0 });
        let state = model.default_state();

        // Overlapping sphere: centres 1.5 apart, radii 1.0 + 1.0 → depth 0.5.
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(1.5, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);
        let out = Sensor::Contact {
            body_idx: 0,
            margin: 0.0,
        }
        .read(&ctx, 0);

        assert_eq!(out.data.len(), 5);
        assert_eq!(out.data[0], 1.0, "one contact");
        assert!((out.data[1] - 0.5).abs() < 0.1, "depth {}", out.data[1]);
    }

    #[test]
    fn contact_sensor_reports_no_contact_when_apart() {
        let mut model = one_link();
        model.bodies[0].geometry = Some(phyz_model::Geometry::Sphere { radius: 1.0 });
        let state = model.default_state();

        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(5.0, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);
        let out = Sensor::Contact {
            body_idx: 0,
            margin: 0.0,
        }
        .read(&ctx, 0);
        assert_eq!(out.data[0], 0.0);
    }

    #[test]
    fn contact_sensor_margin_creates_a_proximity_band() {
        let mut model = one_link();
        model.bodies[0].geometry = Some(phyz_model::Geometry::Sphere { radius: 1.0 });
        let state = model.default_state();

        // Gap of 0.5 between surfaces.
        let scene = Scene::empty().with_obstacle(Obstacle::new(
            "ball",
            CGeom::Sphere { radius: 1.0 },
            Vec3::new(2.5, 0.0, 0.0),
        ));
        let ctx = SensorContext::new(&model, &state, &scene);

        let tight = Sensor::Contact {
            body_idx: 0,
            margin: 0.1,
        }
        .read(&ctx, 0);
        assert_eq!(tight.data[0], 0.0);

        let loose = Sensor::Contact {
            body_idx: 0,
            margin: 1.0,
        }
        .read(&ctx, 0);
        assert_eq!(loose.data[0], 1.0);
    }

    #[test]
    fn body_accel_reports_free_fall() {
        // A single free body under gravity accelerates downward at g.
        let model = ModelBuilder::new()
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();
        let state = model.default_state();

        let out = Sensor::BodyAccel { body_idx: 0 }.read_state(&model, &state, 0);
        assert_eq!(out.data.len(), 3);
        assert!(
            (out.data[2] + 9.81).abs() < 1e-6,
            "free fall should read -g, got {}",
            out.data[2]
        );

        // And the same body's IMU reads zero specific force in free fall.
        let imu = Sensor::Imu { body_idx: 0 }.read_state(&model, &state, 0);
        assert!(
            imu.data[..3].iter().all(|a| a.abs() < 1e-6),
            "free-fall IMU should read ~0, got {:?}",
            &imu.data[..3]
        );
    }

    #[test]
    fn force_torque_holds_up_a_static_load() {
        // A 3 kg mass rigidly welded to the world: the joint must transmit
        // 3 × 9.81 N upward.
        let mass = 3.0;
        let model = ModelBuilder::new()
            .add_fixed_body(
                "post",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(mass, Vec3::zeros()),
            )
            .build();
        let state = model.default_state();

        let out = Sensor::ForceTorque { body_idx: 0 }.read_state(&model, &state, 0);
        assert_eq!(out.data.len(), 6);
        let fz = out.data[5];
        assert!(
            (fz - mass * 9.81).abs() < 1e-6,
            "expected {} N, got {fz}",
            mass * 9.81
        );
    }

    #[test]
    fn force_torque_reports_the_moment_of_an_offset_mass() {
        // Same mass, now hung 2 m along +X: the joint sees the same vertical
        // force plus a moment of m·g·d about Y.
        let (mass, arm) = (3.0, 2.0);
        let model = ModelBuilder::new()
            .add_fixed_body(
                "arm",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(mass, Vec3::new(arm, 0.0, 0.0)),
            )
            .build();
        let state = model.default_state();

        let out = Sensor::ForceTorque { body_idx: 0 }.read_state(&model, &state, 0);
        assert!(
            (out.data[5] - mass * 9.81).abs() < 1e-6,
            "fz {}",
            out.data[5]
        );
        assert!(
            (out.data[1].abs() - mass * 9.81 * arm).abs() < 1e-6,
            "expected |τy| = {}, got {}",
            mass * 9.81 * arm,
            out.data[1]
        );
    }

    #[test]
    fn output_dims_match_actual_readings() {
        let model = one_link();
        let state = model.default_state();
        let scene = Scene::empty();
        let ctx = SensorContext::new(&model, &state, &scene);

        for sensor in [
            Sensor::BodyAccel { body_idx: 0 },
            Sensor::BodyAngularVel { body_idx: 0 },
            Sensor::ForceTorque { body_idx: 0 },
            Sensor::Rangefinder {
                body_idx: 0,
                origin: Vec3::zeros(),
                direction: Vec3::new(1.0, 0.0, 0.0),
                max_dist: 1.0,
            },
            Sensor::Contact {
                body_idx: 0,
                margin: 0.0,
            },
            Sensor::Imu { body_idx: 0 },
            Sensor::FrameCapture { body_idx: 0 },
        ] {
            let out = sensor.read(&ctx, 0);
            assert_eq!(
                out.data.len(),
                sensor.output_dim(),
                "output_dim mismatch for {sensor:?}"
            );
        }
    }
}
