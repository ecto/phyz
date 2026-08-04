//! End-to-end: import a URDF humanoid stub, mount a camera on its head, and
//! render what the robot sees.
//!
//! This is the wiring test — model → `SensorContext::xforms()` → instance
//! buffer → image — and it also pins down the two things a head-mounted camera
//! gets wrong most often: the mount rotation from a ROS-style `+X` forward link
//! to the OpenCV optical frame, and the camera seeing its own head.
//!
//! Skips cleanly when no wgpu adapter is available.

use phyz_camera::{
    CameraError, CameraIntrinsics, CameraPose, RenderScene, RgbdCamera, SceneOptions, body_pose,
    mesh, sensor_pose,
};
use phyz_math::{Mat3, SpatialTransform, Vec3};
use phyz_urdf::{BaseKind, UrdfOptions};
use phyz_world::{Scene, Sensor, SensorContext};

/// A cut-down humanoid in the shape of a Booster K1: floating torso, a neck
/// joint, and a head link the camera bolts to. Written inline so the test has
/// no asset dependency.
const HUMANOID_URDF: &str = r#"
<robot name="k1_stub">
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.24 0.16 0.40"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.24 0.16 0.40"/></geometry>
    </collision>
  </link>

  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.09"/></geometry>
    </visual>
  </link>

  <joint name="neck_pitch" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.30" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.7" upper="0.7" effort="10" velocity="5"/>
  </joint>

  <link name="upper_arm">
    <inertial>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.24"/></geometry>
    </visual>
  </link>

  <joint name="shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="upper_arm"/>
    <origin xyz="0 -0.14 0.16" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="20" velocity="5"/>
  </joint>
</robot>
"#;

/// The mount rotation from a ROS-style link frame (+X forward, +Y left, +Z up)
/// to an OpenCV optical frame (+Z forward, +X right, +Y down).
///
/// As a *coordinate* transform body → optical, its rows are the optical axes
/// expressed in body coordinates: optical X = body −Y, optical Y = body −Z,
/// optical Z = body +X.
fn ros_to_optical() -> Mat3 {
    Mat3::new(
        0.0, -1.0, 0.0, // optical X
        0.0, 0.0, -1.0, // optical Y
        1.0, 0.0, 0.0, // optical Z
    )
}

fn camera_or_skip(k: CameraIntrinsics) -> Option<RgbdCamera> {
    match RgbdCamera::new(k) {
        Ok(c) => Some(c),
        Err(CameraError::NoAdapter) => {
            eprintln!("skipping: no wgpu adapter");
            None
        }
        Err(e) => panic!("camera setup failed: {e}"),
    }
}

#[test]
fn ros_to_optical_is_a_rotation_that_points_the_camera_forward() {
    // Pure CPU, so this one runs even without a GPU. A body→optical coordinate
    // transform R must be orthonormal, and its transpose must carry optical +Z
    // onto body +X — "the camera looks where the head looks".
    let r = ros_to_optical();
    let rt = r.transpose();
    assert!((r.mul_mat(&rt) - Mat3::identity()).norm_sq() < 1e-24);
    assert!(
        (r.determinant() - 1.0).abs() < 1e-12,
        "must not be a reflection"
    );

    let forward = rt.mul_vec(Vec3::new(0.0, 0.0, 1.0));
    assert!(
        (forward - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12,
        "{forward:?}"
    );
    // Optical +Y (image down) must be body −Z, i.e. actually down.
    let down = rt.mul_vec(Vec3::new(0.0, 1.0, 0.0));
    assert!(
        (down - Vec3::new(0.0, 0.0, -1.0)).norm() < 1e-12,
        "{down:?}"
    );
}

#[test]
fn head_mounted_camera_renders_the_robot_and_the_ground() {
    let robot = phyz_urdf::load_str(
        HUMANOID_URDF,
        &UrdfOptions {
            base: BaseKind::Floating,
            dt: Some(0.002),
            ..Default::default()
        },
    )
    .expect("URDF should import");

    let head = robot.body_index("head").expect("head link");
    let model = robot.model;
    let state = model.default_state();

    // A 0.05 m forward offset on the head link, rotated into the optical frame.
    let extrinsics = SpatialTransform::new(ros_to_optical(), Vec3::new(0.05, 0.0, 0.02));
    let intrinsics = CameraIntrinsics::from_vfov(96, 96, 1.2, 0.02, 20.0);
    let sensor = Sensor::Camera {
        body_idx: head,
        origin: extrinsics,
        intrinsics,
    };

    // A camera reads as an empty scalar observation: images must never be
    // smuggled through `SensorOutput.data`.
    let scene = Scene::empty().with_ground(0.0);
    let ctx = SensorContext::free_flight(&model, &state, &scene);
    let out = sensor.read(&ctx, 0);
    assert!(out.data.is_empty(), "camera must not emit scalar samples");
    assert_eq!(sensor.output_dim(), 0);

    let Some(mut cam) = camera_or_skip(intrinsics) else {
        return;
    };

    // Exclude the head so the camera is not staring at the inside of its own
    // skull, then render from the pose the sensor descriptor implies.
    let opts = SceneOptions::new().exclude_body(head);
    let render_scene = RenderScene::from_context(&ctx, &opts);
    assert!(
        render_scene.triangle_count() > 0,
        "the robot should tessellate to something"
    );

    let frame = cam
        .render_sensor(&ctx, &sensor, &render_scene, 0)
        .expect("render should succeed");

    assert_eq!(frame.width(), 96);
    assert_eq!(frame.height(), 96);
    assert_eq!(frame.depth_cpu().unwrap().len(), 96 * 96);
    assert_eq!(frame.color_cpu().unwrap().len(), 96 * 96 * 4);

    // The camera looks along the torso's +X with a ground plane below, so a
    // decent chunk of the image must be ground. Not all of it: the horizon is
    // in view.
    let coverage = frame.depth_coverage();
    assert!(
        coverage > 0.2 && coverage < 1.0,
        "expected the ground to fill part of the frame, coverage = {coverage}"
    );

    // Every returned depth is finite, positive, and inside the frustum.
    for &d in frame.depth_cpu().unwrap() {
        assert!(d.is_finite(), "non-finite depth {d}");
        assert!(
            d == 0.0 || (0.02..=20.0).contains(&d),
            "out-of-range depth {d}"
        );
    }

    // The pose the sensor implies is the pose you get by hand from the same
    // transforms.
    let by_hand = body_pose(ctx.xforms(), head, &extrinsics).unwrap();
    let by_sensor = sensor_pose(&ctx, &sensor, 0).unwrap();
    assert_eq!(by_hand, by_sensor);
    // The head sits ~0.30 m above the torso origin, which starts at the world
    // origin, plus the 0.02 m mount offset.
    assert!(
        (by_sensor.position.z - 0.32).abs() < 1e-9,
        "camera at z = {}",
        by_sensor.position.z
    );
}

#[test]
fn a_non_camera_sensor_is_rejected() {
    let robot = phyz_urdf::load_str(HUMANOID_URDF, &UrdfOptions::default()).unwrap();
    let model = robot.model;
    let state = model.default_state();
    let scene = Scene::empty();
    let ctx = SensorContext::free_flight(&model, &state, &scene);

    let err = sensor_pose(&ctx, &Sensor::Imu { body_idx: 0 }, 3).unwrap_err();
    assert!(matches!(err, CameraError::NotACamera { sensor_id: 3 }));
}

#[test]
fn a_camera_on_a_missing_body_errors_instead_of_panicking() {
    let err = body_pose(&[], 7, &SpatialTransform::identity()).unwrap_err();
    assert!(matches!(
        err,
        CameraError::UnknownBody {
            body_idx: 7,
            nbodies: 0
        }
    ));
}

#[test]
fn an_stl_mesh_renders_at_the_distance_it_was_placed() {
    // STL is the one mesh format supported so far, and it has to survive the
    // whole path: parse → tessellate → instance → depth. A single facet
    // 1.5 m ahead, big enough to cover the principal point.
    let stl = b"solid quad\n\
facet normal 0 0 0\nouter loop\n\
vertex -1 -1 0\nvertex 1 -1 0\nvertex 1 1 0\nendloop\nendfacet\n\
facet normal 0 0 0\nouter loop\n\
vertex -1 -1 0\nvertex 1 1 0\nvertex -1 1 0\nendloop\nendfacet\n\
endsolid quad\n";
    let tri = mesh::parse_stl(stl).expect("STL should parse");
    assert_eq!(tri.triangle_count(), 2);

    let k = CameraIntrinsics::from_vfov(64, 64, 1.0, 0.05, 10.0);
    let Some(mut cam) = camera_or_skip(k) else {
        return;
    };

    let mut scene = RenderScene::new();
    let m = scene.add_mesh(tri);
    scene.add_instance(phyz_camera::Instance {
        mesh: m,
        world_from_local: Mat3::identity(),
        position: Vec3::new(0.0, 0.0, 1.5),
        albedo: [0.9, 0.4, 0.2],
        body: None,
    });

    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();
    let d = frame
        .depth_at_principal_point()
        .expect("the facet covers the axis");
    assert!((d - 1.5).abs() < 1e-4, "STL facet read {d} m, expected 1.5");
}

#[test]
fn unsupported_mesh_formats_say_so() {
    let err = mesh::load_mesh("head.dae").unwrap_err();
    assert!(matches!(err, CameraError::UnsupportedMeshFormat { .. }));
}
