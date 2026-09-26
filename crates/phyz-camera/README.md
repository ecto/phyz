# phyz-camera

Headless RGBD camera sensing for the [phyz](https://phyz.dev) physics engine.

A robot-mounted pinhole camera that produces an RGB image plus a **linear depth
image in metres** every step, cheap enough to sit inside a training loop as a
world-model observation source. It is a wgpu **rasterizer**, not a ray tracer:
at the 64–128 px resolutions a world model actually consumes, rasterizing a few
thousand triangles costs microseconds.

## Conventions

- **Optical frame is OpenCV**: the camera looks down its own `+Z`, `+X` is right
  across the image, `+Y` is **down** the image. Not ROS/REP-103 `+X`-forward — a
  mount rotation lives in the extrinsics, not in the optical frame.
- **Pixels** are row-major from the top-left; `v = 0` is the top row; pixel
  centres are at half-integers. `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.
- **Depth is linear metres of optical-axis `Z`** — not ray length, not a
  normalised depth-buffer value. A wall 2 m ahead reads `2.0` in every pixel
  that sees it, corners included. `0.0` means "no return".
- **Extrinsics** use phyz's Plücker convention: `origin.pos` is the optical
  origin in body coordinates, `origin.rot` maps body coordinates into the
  optical frame.

## Usage

```rust,no_run
use phyz_camera::{CameraPose, RenderScene, RgbdCamera, SceneOptions};
use phyz_math::Vec3;
use phyz_world::{CameraIntrinsics, Scene, SensorContext};

# fn demo(model: &phyz_model::Model, state: &phyz_model::State)
#     -> Result<(), Box<dyn std::error::Error>> {
let intrinsics = CameraIntrinsics::from_vfov(128, 128, 1.0, 0.05, 10.0);
let mut camera = RgbdCamera::new(intrinsics)?;

let scene = Scene::empty().with_ground(0.0);
let ctx = SensorContext::new(model, state, &scene);
let render_scene = RenderScene::from_context(&ctx, &SceneOptions::new());

let pose = CameraPose::look_at(
    Vec3::new(0.0, -2.0, 1.0),
    Vec3::zeros(),
    Vec3::new(0.0, 0.0, 1.0),
);
let frame = camera.render(&render_scene, &pose)?;
let depth = frame.depth_cpu().unwrap();
let rgba = frame.color_cpu().unwrap();
# Ok(())
# }
```

Body poses come from `SensorContext::xforms()` — the same single per-step
forward-kinematics pass every other phyz sensor reads — so an image can never
be a step out of sync with the proprioception recorded next to it.

`RgbdCamera::with_device_queue` takes an existing `Arc<wgpu::Device>` /
`Arc<wgpu::Queue>`, so a camera can share the GPU context that
`phyz_gpu::GpuBatchSimulator` is already using. This crate therefore pins the
same wgpu version phyz-gpu does.

`RgbdCamera::new` returns `CameraError::NoAdapter` when no GPU is available, so
headless CI can skip rather than hang.

## Not yet supported

Mesh formats other than STL; textures, shadows and global illumination; batched
multi-environment rendering and GPU-resident output (the buffer enums in
`phyz_camera::frame` exist so that can be added without an API break).

## License

MIT
