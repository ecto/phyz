// Instanced point rendering for parameter space visualization.
// Each point is a billboard quad that renders as a glowing sphere.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    // Per-vertex (quad corners)
    @location(0) quad_pos: vec2<f32>,
    // Per-instance
    @location(1) position: vec3<f32>,  // world position (log_g2, s_ee, a_cut)
    @location(2) color: vec4<f32>,     // RGBA with alpha for age-based glow
    @location(3) size: f32,            // point size
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Billboard: offset quad in screen space
    let world_pos = vec4<f32>(in.position, 1.0);
    let clip = camera.view_proj * world_pos;
    let aspect = 1.0; // adjust if needed
    let offset = in.quad_pos * in.size * 0.02;
    out.clip_pos = clip + vec4<f32>(offset.x, offset.y * aspect, 0.0, 0.0);
    out.uv = in.quad_pos;
    out.color = in.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Radial falloff for glow effect
    let dist = length(in.uv);
    if dist > 1.0 {
        discard;
    }

    // Soft glow: exponential falloff
    let glow = exp(-dist * dist * 3.0);
    let core = smoothstep(0.4, 0.0, dist);

    let brightness = glow * 0.6 + core * 0.4;
    return vec4<f32>(in.color.rgb * brightness, in.color.a * brightness);
}
