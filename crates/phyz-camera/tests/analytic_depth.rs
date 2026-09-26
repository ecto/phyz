//! Does the depth image agree with closed-form geometry?
//!
//! These are the tests that matter. A camera that renders a plausible-looking
//! picture but is off by a factor, a sign, or half a pixel is worse than no
//! camera at all, because everything downstream trains on the lie. So every
//! assertion here is against a number computed by hand, not against a stored
//! image.
//!
//! Every test degrades to a skip when no wgpu adapter exists, so a headless CI
//! box without a GPU reports "ok" rather than hanging or panicking.

use phyz_camera::{
    CameraError, CameraFrame, CameraIntrinsics, CameraPose, Instance, RenderScene, RgbdCamera,
    Tessellation, mesh,
};
use phyz_math::{Mat3, Vec3};

/// 128×96, 90°-ish horizontal FOV, principal point deliberately *off* centre so
/// a swapped or centred-by-accident cx/cy cannot pass.
fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics {
        fx: 80.0,
        fy: 90.0,
        cx: 60.0,
        cy: 52.0,
        width: 128,
        height: 96,
        near: 0.05,
        far: 50.0,
    }
}

/// Build a camera, or `None` when this machine has no GPU.
fn camera(k: CameraIntrinsics) -> Option<RgbdCamera> {
    match RgbdCamera::new(k) {
        Ok(c) => Some(c),
        Err(CameraError::NoAdapter) => {
            eprintln!("skipping: no wgpu adapter");
            None
        }
        Err(e) => panic!("camera setup failed: {e}"),
    }
}

/// A scene holding one instance of a mesh at a world placement.
fn one(mesh: mesh::TriMesh, rot: Mat3, position: Vec3) -> RenderScene {
    let mut s = RenderScene::new();
    let m = s.add_mesh(mesh);
    s.add_instance(Instance {
        mesh: m,
        world_from_local: rot,
        position,
        albedo: [0.8, 0.8, 0.8],
        body: None,
    });
    s
}

/// Depth at the pixel *containing* the continuous coordinate `(u, v)`.
fn depth_at_uv(frame: &CameraFrame, u: f64, v: f64) -> Option<f32> {
    frame.depth_at(u.floor() as u32, v.floor() as u32)
}

#[test]
fn a_wall_at_two_metres_reads_two_metres_everywhere() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // A 40 m square wall in the plane z = 2 (world), camera at the origin
    // looking down world +Z with the OpenCV frame aligned to world axes.
    // Every visible pixel must read exactly 2.0 — depth is optical-axis Z, not
    // ray length, so the corners are 2.0 as well. If this test ever reads
    // sqrt(2² + x² + y²) at the corners, the depth convention has drifted.
    let wall = mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0);
    let scene = one(wall, Mat3::identity(), Vec3::new(0.0, 0.0, 2.0));
    let pose = CameraPose::identity();

    let frame = cam.render(&scene, &pose).unwrap();
    assert_eq!(frame.depth_cpu().unwrap().len(), k.pixel_count());
    assert!(
        (frame.depth_coverage() - 1.0).abs() < 1e-12,
        "wall should fill the image, coverage = {}",
        frame.depth_coverage()
    );

    for v in 0..k.height {
        for u in 0..k.width {
            let d = frame.depth_at(u, v).expect("wall covers every pixel");
            assert!(
                (d - 2.0).abs() < 1e-4,
                "pixel ({u},{v}) reads {d}, expected 2.0"
            );
        }
    }
}

#[test]
fn a_tilted_wall_matches_the_ray_plane_intersection() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Same wall, rotated 30° about the camera's X axis so depth varies down the
    // image. For a plane through p0 with unit normal n, the optical-axis depth
    // along the ray through pixel (u, v) is
    //     Z = (n · p0) / (n · d),   d = ((u-cx)/fx, (v-cy)/fy, 1)
    // which is what every pixel is checked against.
    let angle = 30f64.to_radians();
    let rot = Mat3::rotation_x(angle);
    let p0 = Vec3::new(0.0, 0.0, 3.0);
    let n = rot.mul_vec(Vec3::new(0.0, 0.0, -1.0));

    let wall = mesh::plane(Vec3::new(0.0, 0.0, -1.0), 30.0);
    let scene = one(wall, rot, p0);
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();

    let mut checked = 0;
    for v in 0..k.height {
        for u in 0..k.width {
            let Some(d) = frame.depth_at(u, v) else {
                continue;
            };
            let dir = Vec3::new(
                (u as f64 + 0.5 - k.cx) / k.fx,
                (v as f64 + 0.5 - k.cy) / k.fy,
                1.0,
            );
            let expected = n.dot(p0) / n.dot(dir);
            assert!(
                (d as f64 - expected).abs() < 2e-3,
                "pixel ({u},{v}): got {d}, expected {expected}"
            );
            checked += 1;
        }
    }
    assert!(checked > k.pixel_count() / 2, "only {checked} pixels hit");
}

#[test]
fn a_sphere_reads_its_near_pole_on_axis_and_its_silhouette_where_predicted() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Sphere of radius 0.5 centred 3 m down the optical axis. The nearest point
    // is at Z = 3 - 0.5 = 2.5, and it lies exactly on the axis, so the pixel at
    // the principal point must read 2.5 (up to tessellation error).
    let (radius, dist) = (0.5, 3.0);
    let tess = Tessellation {
        segments: 128,
        rings: 64,
        ..Default::default()
    };
    let scene = one(
        mesh::sphere(radius, &tess),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, dist),
    );
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();

    let d = depth_at_uv(&frame, k.cx, k.cy).expect("sphere covers the principal point");
    assert!(
        (d as f64 - (dist - radius)).abs() < 2e-3,
        "on-axis depth {d}, expected {}",
        dist - radius
    );

    // The silhouette is a circle of angular radius asin(r/dist); its image
    // radius is fx·tan(asin(r/dist)) horizontally. One pixel inside must return,
    // and a couple of pixels outside must not.
    let half_angle = (radius / dist).asin();
    let r_px = k.fx * half_angle.tan();
    assert!(
        depth_at_uv(&frame, k.cx + r_px - 1.5, k.cy).is_some(),
        "just inside the silhouette should return"
    );
    assert!(
        depth_at_uv(&frame, k.cx + r_px + 2.5, k.cy).is_none(),
        "just outside the silhouette should not return"
    );
}

#[test]
fn a_known_point_lands_on_the_predicted_pixel() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // A small sphere, off-axis, seen from a non-trivial camera pose. The
    // predicted pixel comes from the intrinsics; the observed pixel comes from
    // the centroid of the returned depths. A sphere is used rather than a box
    // because its silhouette is a circle centred on the projected centre, so the
    // centroid is a fair estimator of where the point landed.
    let target = Vec3::new(0.7, -0.4, 1.1);
    let eye = Vec3::new(-2.0, -1.5, 2.2);
    let radius = 0.12;
    let pose = CameraPose::look_at(eye, Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.0, 0.0, 1.0));

    let tess = Tessellation {
        segments: 96,
        rings: 48,
        ..Default::default()
    };
    let scene = one(mesh::sphere(radius, &tess), Mat3::identity(), target);
    let frame = cam.render(&scene, &pose).unwrap();

    let (u_hat, v_hat) = pose.project(&k, target).expect("target must be in range");
    let expected_depth = pose.to_optical(target).z;

    let (mut su, mut sv, mut n) = (0.0, 0.0, 0.0);
    for v in 0..k.height {
        for u in 0..k.width {
            if frame.depth_at(u, v).is_some() {
                su += u as f64 + 0.5;
                sv += v as f64 + 0.5;
                n += 1.0;
            }
        }
    }
    assert!(n > 8.0, "sphere should cover several pixels, got {n}");
    let (u_obs, v_obs) = (su / n, sv / n);
    assert!(
        (u_obs - u_hat).abs() < 1.0 && (v_obs - v_hat).abs() < 1.0,
        "centroid ({u_obs:.2},{v_obs:.2}) vs predicted ({u_hat:.2},{v_hat:.2})"
    );

    // The nearest visible point of a sphere is one radius in front of its
    // centre, along the line of sight — and since the centre is only a few
    // degrees off axis, that is a radius of optical-axis depth to good accuracy.
    let dmin = frame
        .depth_cpu()
        .unwrap()
        .iter()
        .copied()
        .filter(|&d| d > 0.0)
        .fold(f32::INFINITY, f32::min) as f64;
    assert!(
        (expected_depth - radius - dmin).abs() < 5e-3,
        "nearest depth {dmin}, expected {}",
        expected_depth - radius
    );
}

#[test]
fn depth_respects_the_near_and_far_planes() {
    let mut k = intrinsics();
    k.near = 1.0;
    k.far = 5.0;
    let Some(mut cam) = camera(k) else { return };

    let wall = |z: f64| {
        one(
            mesh::plane(Vec3::new(0.0, 0.0, -1.0), 40.0),
            Mat3::identity(),
            Vec3::new(0.0, 0.0, z),
        )
    };

    // Inside the range: full coverage.
    let inside = cam.render(&wall(3.0), &CameraPose::identity()).unwrap();
    assert!((inside.depth_coverage() - 1.0).abs() < 1e-12);

    // Closer than near, and further than far: nothing comes back.
    for z in [0.5, 8.0] {
        let f = cam.render(&wall(z), &CameraPose::identity()).unwrap();
        assert_eq!(
            f.depth_coverage(),
            0.0,
            "a wall at {z} m must be clipped by near={} far={}",
            k.near,
            k.far
        );
    }
}

#[test]
fn the_nearer_surface_wins() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Two walls, near one drawn *second* so a missing depth test would leave the
    // far one visible.
    let mut scene = RenderScene::new();
    let far = scene.add_mesh(mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0));
    let near = scene.add_mesh(mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0));
    for (m, z) in [(far, 4.0), (near, 1.5)] {
        scene.add_instance(Instance {
            mesh: m,
            world_from_local: Mat3::identity(),
            position: Vec3::new(0.0, 0.0, z),
            albedo: [0.8, 0.8, 0.8],
            body: None,
        });
    }
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();
    let d = frame.depth_at_principal_point().unwrap();
    assert!((d - 1.5).abs() < 1e-4, "expected the near wall, got {d}");
}

#[test]
fn the_point_cloud_round_trips_through_the_intrinsics() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Back-projecting a wall at z = 2.5 must give points that all sit on that
    // plane, which checks unproject against whatever the rasterizer actually
    // produced rather than against itself.
    let scene = one(
        mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, 2.5),
    );
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();
    let cloud = frame.point_cloud();
    assert_eq!(cloud.len(), k.pixel_count());
    for p in &cloud {
        assert!((p.z - 2.5).abs() < 1e-4, "point off the plane: {p:?}");
    }
    // Corners of the image should be well away from the axis, i.e. the cloud
    // really spans the frustum rather than collapsing to a point.
    let spread = cloud
        .iter()
        .map(|p| p.x.abs().max(p.y.abs()))
        .fold(0.0f64, f64::max);
    assert!(spread > 1.0, "cloud spread only {spread} m");
}

#[test]
fn colour_comes_back_shaded_and_opaque() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    let scene = one(
        mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();

    let rgba = frame.color_cpu().unwrap();
    assert_eq!(rgba.len(), k.pixel_count() * 4);
    let px = frame.color_at(k.width / 2, k.height / 2).unwrap();
    assert_eq!(px[3], 255, "alpha must be opaque");
    assert!(px[0] > 20, "lit surface should not be black: {px:?}");
}

/// A wall of two painted triangles filling the view at `z`, red on the left
/// edge and blue on the right, so the interpolation across the face is
/// visible in the rendered pixels.
fn painted_wall(z: f64, half: f64) -> mesh::TriMesh {
    let mut m = mesh::TriMesh::empty();
    let red = [1.0, 0.0, 0.0];
    let blue = [0.0, 0.0, 1.0];
    let (a, b) = (Vec3::new(-half, -half, z), Vec3::new(half, -half, z));
    let (c, d) = (Vec3::new(half, half, z), Vec3::new(-half, half, z));
    m.push_triangle_painted(a, b, c, [red, blue, blue]);
    m.push_triangle_painted(a, c, d, [red, blue, red]);
    m
}

#[test]
fn an_unpainted_mesh_renders_exactly_as_its_instance_albedo() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Vertex colour defaults to white and multiplies, so a mesh built by any
    // geometric builder must be untouched by the feature's existence. The
    // green channel is 0.8 albedo like every other test; red and blue equal
    // it, which is only true if nothing tinted the surface.
    let scene = one(
        mesh::plane(Vec3::new(0.0, 0.0, -1.0), 20.0),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();
    let px = frame.color_at(k.width / 2, k.height / 2).unwrap();
    assert!(px[0] > 20, "lit surface should not be black: {px:?}");
    assert!(
        px[0] == px[1] && px[1] == px[2],
        "a grey instance albedo must stay grey when nothing painted it: {px:?}"
    );
}

#[test]
fn vertex_colour_paints_the_surface_and_interpolates_across_it() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    let scene = one(
        painted_wall(0.0, 4.0),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let frame = cam.render(&scene, &CameraPose::identity()).unwrap();

    // The wall spans the view, so sample well inside either edge. Which pixel
    // column is "left" follows the projection the depth tests already pin.
    let left = frame.color_at(k.width / 8, k.height / 2).unwrap();
    let right = frame.color_at(k.width - k.width / 8, k.height / 2).unwrap();

    assert!(
        left[0] > left[2] && right[2] > right[0],
        "red edge should stay red and blue edge blue: left {left:?} right {right:?}"
    );
    let mid = frame.color_at(k.width / 2, k.height / 2).unwrap();
    assert!(
        mid[0] > 10 && mid[2] > 10,
        "the middle should be a blend of both corners, not one of them: {mid:?}"
    );
    assert_eq!(mid[3], 255, "alpha must stay opaque");
}

#[test]
fn repainting_a_mesh_needs_the_camera_told_and_then_shows() {
    let k = intrinsics();
    let Some(mut cam) = camera(k) else { return };

    // Same geometry, same instance count, different vertex colours: the only
    // thing that changed is data the cheap path does not look at.
    let scene = one(
        painted_wall(0.0, 4.0),
        Mat3::identity(),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let red_edge = cam
        .render(&scene, &CameraPose::identity())
        .unwrap()
        .color_at(k.width / 8, k.height / 2)
        .unwrap();
    assert!(
        red_edge[0] > red_edge[2],
        "left edge starts red: {red_edge:?}"
    );

    let mut repainted = mesh::TriMesh::empty();
    let green = [0.0, 1.0, 0.0];
    let (a, b) = (Vec3::new(-4.0, -4.0, 0.0), Vec3::new(4.0, -4.0, 0.0));
    let (c, d) = (Vec3::new(4.0, 4.0, 0.0), Vec3::new(-4.0, 4.0, 0.0));
    repainted.push_triangle_painted(a, b, c, [green; 3]);
    repainted.push_triangle_painted(a, c, d, [green; 3]);
    let scene2 = one(repainted, Mat3::identity(), Vec3::new(0.0, 0.0, 2.0));

    let stale = cam
        .render(&scene2, &CameraPose::identity())
        .unwrap()
        .color_at(k.width / 8, k.height / 2)
        .unwrap();
    assert!(
        stale[0] > stale[1],
        "documents the cheap path: without invalidate_scene the old vertices \
         are still drawn, so this is still red: {stale:?}"
    );

    cam.invalidate_scene();
    let fresh = cam
        .render(&scene2, &CameraPose::identity())
        .unwrap()
        .color_at(k.width / 8, k.height / 2)
        .unwrap();
    assert!(
        fresh[1] > fresh[0] && fresh[1] > fresh[2],
        "after invalidate_scene the repaint must show green: {fresh:?}"
    );
}

#[test]
fn invalid_intrinsics_are_rejected_not_rendered() {
    let mut k = intrinsics();
    k.far = k.near; // degenerate frustum
    match RgbdCamera::new(k) {
        Err(CameraError::InvalidIntrinsics { .. }) => {}
        Err(CameraError::NoAdapter) => eprintln!("skipping: no wgpu adapter"),
        other => panic!(
            "expected InvalidIntrinsics, got {other:?}",
            other = other.err()
        ),
    }
}
