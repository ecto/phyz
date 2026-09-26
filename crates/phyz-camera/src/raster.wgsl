// Flat/Lambert rasterization for the phyz RGBD camera.
//
// Two colour attachments come out of one pass:
//   location 0: Rgba8Unorm shaded colour
//   location 1: R32Float  linear depth in metres, i.e. the optical-frame Z of
//               the surface. Cleared to 0.0, which is the "no return" value.
//
// `z_optical` is passed as a varying rather than reconstructed from the depth
// buffer: the rasterizer interpolates varyings perspective-correctly, and a
// coordinate of the world position is a linear function of world position, so
// the interpolated value is exact — not an approximation of the plane fit.

struct Uniforms {
    view: mat4x4<f32>,        // world -> optical
    proj: mat4x4<f32>,        // optical -> clip
    light_dir: vec4<f32>,     // world-space direction *towards* the light, xyz
    shading: vec4<f32>,       // x: ambient, y: diffuse, zw: unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Instance: rows of the local -> world affine transform, translation in .w.
    @location(2) row0: vec4<f32>,
    @location(3) row1: vec4<f32>,
    @location(4) row2: vec4<f32>,
    @location(5) albedo: vec4<f32>,
    // Per-vertex tint, white unless the mesh was painted. Location 6 keeps it
    // clear of the instance rows above.
    @location(6) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) albedo: vec3<f32>,
    @location(2) z_optical: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = vec3<f32>(
        dot(in.row0.xyz, in.position) + in.row0.w,
        dot(in.row1.xyz, in.position) + in.row1.w,
        dot(in.row2.xyz, in.position) + in.row2.w,
    );
    // The instance rotation is orthonormal, so it doubles as its own normal
    // matrix; per-axis mesh scaling is baked into the vertex data instead.
    let n = vec3<f32>(
        dot(in.row0.xyz, in.normal),
        dot(in.row1.xyz, in.normal),
        dot(in.row2.xyz, in.normal),
    );

    let optical = u.view * vec4<f32>(world, 1.0);

    var out: VsOut;
    out.clip = u.proj * optical;
    out.normal_world = n;
    // Instance albedo stays a tint over the vertex colour, so an unpainted
    // mesh (white) renders exactly as it did before vertex colour existed,
    // and a painted one can still be dimmed or recoloured per instance.
    out.albedo = in.albedo.xyz * in.color;
    out.z_optical = optical.z;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> FsOut {
    var n = normalize(in.normal_world);
    // Back faces are lit as if their normal were flipped, so a camera inside a
    // shell or looking at a one-sided ground quad still gets sane shading
    // instead of a black silhouette.
    if (!front_facing) {
        n = -n;
    }
    let lambert = max(dot(n, normalize(u.light_dir.xyz)), 0.0);
    let shade = u.shading.x + u.shading.y * lambert;

    var out: FsOut;
    out.color = vec4<f32>(clamp(in.albedo * shade, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    out.depth = in.z_optical;
    return out;
}
