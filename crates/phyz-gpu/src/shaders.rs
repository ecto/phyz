//! WGSL compute shaders for GPU-accelerated physics.

/// WGSL shader for semi-implicit Euler integration.
///
/// Each work item processes one DOF across all worlds in parallel.
pub const INTEGRATE_SHADER: &str = r#"
struct SimParams {
    nworld: u32,
    nv: u32,
    dt: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> qdd: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_dofs = params.nworld * params.nv;

    if (idx >= total_dofs) {
        return;
    }

    let dt = params.dt;

    // Semi-implicit Euler: v' = v + dt * qdd, q' = q + dt * v'
    let v_old = v[idx];
    let qdd_val = qdd[idx];
    let v_new = v_old + dt * qdd_val;
    let q_old = q[idx];
    let q_new = q_old + dt * v_new;

    v[idx] = v_new;
    q[idx] = q_new;
}
"#;

/// WGSL shader for simplified ABA (single revolute joint systems).
///
/// This is a simplified version that handles pendulum-like systems
/// with single revolute joints. For multi-body systems, we'd need
/// a more complex shader with tree traversal.
pub const ABA_SIMPLE_SHADER: &str = r#"
struct SimParams {
    nworld: u32,
    nv: u32,
    dt: f32,
    _padding: u32,
}

struct BodyParams {
    mass: f32,
    inertia: f32,
    com_y: f32,
    damping: f32,
    gravity_y: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<uniform> body: BodyParams;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> ctrl: array<f32>;
@group(0) @binding(5) var<storage, read_write> qdd: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world_idx = gid.x;

    if (world_idx >= params.nworld) {
        return;
    }

    // For single revolute joint: qdd = (tau - damping*v - m*g*L*sin(q)) / I
    let idx = world_idx;
    let q_val = q[idx];
    let v_val = v[idx];
    let tau = ctrl[idx];

    // Gravity torque: m * g * L * sin(q)
    // Note: gravity_y is magnitude (positive), com_y is typically negative
    // ABA uses base acceleration trick, so we need positive sign here
    let gravity_torque = body.mass * body.gravity_y * body.com_y * sin(q_val);

    // Total torque: applied torque + gravity torque - damping torque
    let total_torque = tau + gravity_torque - body.damping * v_val;

    // Total inertia for pendulum: I = m*L²/3 (parallel axis theorem)
    // For simplicity, we pass the computed inertia from CPU
    let total_inertia = body.inertia;

    // qdd = torque / inertia
    qdd[idx] = total_torque / total_inertia;
}
"#;
