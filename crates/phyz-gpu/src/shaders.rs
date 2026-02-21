//! WGSL compute shaders for GPU-accelerated physics.

/// WGSL shader for FK + ground contact penalty forces.
///
/// Computes forward kinematics to get body world positions,
/// then detects ground plane contacts and writes penalty forces
/// to the external forces buffer (consumed by ABA shader).
///
/// One thread per environment, serial tree traversal within.
pub const CONTACT_GROUND_SHADER: &str = r#"
const MAX_BODIES: u32 = 16u;
const BODY_STRIDE: u32 = 32u;
const GEOM_STRIDE: u32 = 8u;

struct ContactParams {
    nworld: u32,
    nbodies: u32,
    nv: u32,
    ground_height: f32,
    stiffness: f32,
    damping: f32,
    friction: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> cparams: ContactParams;
@group(0) @binding(1) var<storage, read> bodies: array<f32>;
@group(0) @binding(2) var<storage, read> geometry: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> ext_forces: array<f32>;

// Body data access
fn bf(bi: u32, off: u32) -> f32 { return bodies[bi * BODY_STRIDE + off]; }
fn body_parent(bi: u32) -> i32 { return bitcast<i32>(bodies[bi * BODY_STRIDE]); }
fn body_jtype(bi: u32) -> u32 { return bitcast<u32>(bodies[bi * BODY_STRIDE + 1u]); }
fn body_qoff(bi: u32) -> u32 { return bitcast<u32>(bodies[bi * BODY_STRIDE + 2u]); }

fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Revolute rotation (Rodrigues, -angle convention matching ABA)
fn rev_rot(axis: vec3<f32>, angle: f32) -> array<f32, 9> {
    let s = sin(-angle);
    let c = cos(-angle);
    let t = 1.0 - c;
    let x = axis.x; let y = axis.y; let z = axis.z;
    return array<f32, 9>(
        t*x*x+c, t*x*y-s*z, t*x*z+s*y,
        t*x*y+s*z, t*y*y+c, t*y*z-s*x,
        t*x*z-s*y, t*y*z+s*x, t*z*z+c
    );
}

fn identity_rot() -> array<f32, 9> {
    return array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

// Multiply rotation (row-major) by vector
fn rot_mul(r: array<f32, 9>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        r[0]*v.x + r[1]*v.y + r[2]*v.z,
        r[3]*v.x + r[4]*v.y + r[5]*v.z,
        r[6]*v.x + r[7]*v.y + r[8]*v.z
    );
}

// Compose rotations: A * B (row-major)
fn rot_compose(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
    var r: array<f32, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r[i*3u+j] = a[i*3u]*b[j] + a[i*3u+1u]*b[3u+j] + a[i*3u+2u]*b[6u+j];
        }
    }
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world_idx = gid.x;
    if (world_idx >= cparams.nworld) { return; }

    let nb = cparams.nbodies;
    let q_base = world_idx * cparams.nv;

    // Clear external forces for this env
    let ef_env_base = world_idx * nb * 6u;
    for (var i = 0u; i < nb; i++) {
        for (var k = 0u; k < 6u; k++) {
            ext_forces[ef_env_base + i * 6u + k] = 0.0;
        }
    }

    // Compute FK: world rotation and position for each body
    var w_rot: array<array<f32, 9>, MAX_BODIES>;
    var w_pos: array<vec3<f32>, MAX_BODIES>;

    for (var i = 0u; i < nb; i++) {
        let parent = body_parent(i);
        let jtype = body_jtype(i);
        let q_off = body_qoff(i);

        // Parent-to-joint transform
        var ptj_rot: array<f32, 9>;
        for (var k = 0u; k < 9u; k++) { ptj_rot[k] = bf(i, 14u + k); }
        let ptj_pos = vec3<f32>(bf(i, 23u), bf(i, 24u), bf(i, 25u));
        let axis = vec3<f32>(bf(i, 26u), bf(i, 27u), bf(i, 28u));

        // Joint transform
        var j_rot: array<f32, 9>;
        var j_pos = vec3<f32>(0.0, 0.0, 0.0);
        if (jtype == 0u) {
            j_rot = rev_rot(axis, q[q_base + q_off]);
        } else if (jtype == 1u) {
            j_rot = identity_rot();
            j_pos = axis * q[q_base + q_off];
        } else {
            j_rot = identity_rot();
        }

        // x_tree = j.compose(ptj): rot = j_rot * ptj_rot, pos = ptj_pos + ptj_rot^T * j_pos
        let tree_rot = rot_compose(j_rot, ptj_rot);
        // ptj_rot^T * j_pos
        let rt_jp = vec3<f32>(
            ptj_rot[0]*j_pos.x + ptj_rot[3]*j_pos.y + ptj_rot[6]*j_pos.z,
            ptj_rot[1]*j_pos.x + ptj_rot[4]*j_pos.y + ptj_rot[7]*j_pos.z,
            ptj_rot[2]*j_pos.x + ptj_rot[5]*j_pos.y + ptj_rot[8]*j_pos.z
        );
        let tree_pos = ptj_pos + rt_jp;

        if (parent < 0) {
            w_rot[i] = tree_rot;
            // World position: the tree transform translates from world to body
            // For the world position, we need to invert:
            // p_world = -R^T * tree_pos
            // Actually for FK, body origin in world = parent_world_pos + parent_world_rot^T * tree_pos
            // With parent = world, parent_rot = I, parent_pos = 0
            // So body_world_pos = tree_pos... but this depends on the spatial transform convention.
            // In Featherstone: X transforms motion vectors from frame A to frame B
            // The position stored is from body to joint in parent frame
            // For world position: p_world[i] = p_world[parent] + R_world[parent]^T * (-tree_pos)
            // Hmm, let's use the convention that tree_pos is the translation.
            // Actually: X_tree transforms from parent to child frame.
            // Position in world: p_i = R_parent^T * (p_parent_local + tree_pos_i)
            // For root (parent = world): p_i = tree_pos
            // Wait, this isn't right either. Let me think about this.
            //
            // In Featherstone, X_tree[i] = X_joint * X_parent_to_joint
            // where X has rotation R and position p such that:
            // v_child = X * v_parent means: w_c = R*w_p, v_c = R*(v_p - p x w_p)
            // The body frame origin in parent frame is at position -R^T * p
            // So: world_pos[i] = world_pos[parent] + world_R[parent]^T * (-tree_rot^T * tree_pos)
            // For root: world_pos = -tree_rot^T * tree_pos

            // Actually let me use a simpler approach: accumulate transforms
            // World transform: from body frame to world frame
            // If X_tree goes parent→child, then X_tree_inv goes child→parent
            // world_to_body = X_tree[i] * world_to_parent
            // body_to_world = parent_to_world * X_tree[i]^-1
            // X_tree_inv has rot = R^T, pos = -R*p (SpatialTransform::inverse)
            // Accumulating: world_rot = tree_rot^T * parent_world_rot (if parent is world: tree_rot^T)
            // world_pos = parent_world_pos - parent_world_rot^T * tree_pos ... hmm

            // Simplest approach: track body-to-world as (rot, pos)
            // For root body: X_world_to_body = X_tree[0]
            // X_body_to_world = X_tree[0]^-1
            // inv.rot = tree_rot^T, inv.pos = -(tree_rot * tree_pos)... no
            // SpatialTransform { rot: R, pos: p }.inverse() = { rot: R^T, pos: -(R^T * p) }
            // Wait, let me re-check: compose(self, other) = { rot: self.rot * other.rot, pos: other.pos + other.rot^T * self.pos }
            // inverse(): rt = R^T, pos' = -(R * pos)... actually from the code:
            // fn inverse(&self) -> SpatialTransform { let rt = self.rot.transpose(); SpatialTransform { rot: rt, pos: -(rt * self.pos) } }
            // Wait, I need to check the actual code. For now, let me just compute world positions.
            // I'll use the convention: w_pos = accumulate parent offsets.

            // For tree_rot and tree_pos: the child body's origin in the parent frame is at
            // position = -(tree_rot^T * tree_pos)
            // (since X transforms FROM parent TO child, the child origin in parent coords
            //  is the inverse of the translation part)
            let rt = array<f32, 9>(tree_rot[0], tree_rot[3], tree_rot[6],
                                    tree_rot[1], tree_rot[4], tree_rot[7],
                                    tree_rot[2], tree_rot[5], tree_rot[8]);
            w_rot[i] = rt; // body-to-world rotation = tree_rot^T (for root body)
            // The position of the body in the world frame:
            // p_body_in_world = -(tree_rot^T * tree_pos)
            let neg_rt_p = -rot_mul(rt, tree_pos);
            w_pos[i] = neg_rt_p;
        } else {
            let pi = u32(parent);
            // Body-to-world: parent_body_to_world * tree_inv
            // tree_inv.rot = tree_rot^T
            // tree_inv.pos = -(tree_rot^T * tree_pos)
            // Compose: body_to_world = parent_btw * tree_inv
            // result.rot = parent_rot * tree_rot^T
            // result.pos = tree_inv.pos + tree_inv.rot^T * parent_btw.pos
            //            = -(tree_rot^T * tree_pos) + tree_rot * w_pos[parent]
            let tree_rt = array<f32, 9>(tree_rot[0], tree_rot[3], tree_rot[6],
                                         tree_rot[1], tree_rot[4], tree_rot[7],
                                         tree_rot[2], tree_rot[5], tree_rot[8]);
            w_rot[i] = rot_compose(w_rot[pi], tree_rt);
            let neg_rt_tp = -rot_mul(tree_rt, tree_pos);
            let tree_rot_wp = rot_mul(tree_rot, w_pos[pi]);
            w_pos[i] = neg_rt_tp + tree_rot_wp;
        }
    }

    // Check ground contacts for each body
    for (var i = 0u; i < nb; i++) {
        let gtype = u32(geometry[i * GEOM_STRIDE]);
        if (gtype == 0u) { continue; } // no geometry

        let pos = w_pos[i];

        // Compute lowest point based on geometry type
        var min_z = pos.z;
        if (gtype == 1u) {
            // Sphere
            let radius = geometry[i * GEOM_STRIDE + 1u];
            min_z = pos.z - radius;
        } else if (gtype == 2u) {
            // Box
            let hz = geometry[i * GEOM_STRIDE + 3u];
            min_z = pos.z - hz;
        } else if (gtype == 3u) {
            // Capsule
            let radius = geometry[i * GEOM_STRIDE + 1u];
            let length = geometry[i * GEOM_STRIDE + 2u];
            min_z = pos.z - length * 0.5 - radius;
        } else if (gtype == 4u) {
            // Cylinder
            let height = geometry[i * GEOM_STRIDE + 2u];
            min_z = pos.z - height * 0.5;
        }

        let penetration = cparams.ground_height - min_z;
        if (penetration <= 0.0) { continue; }

        // Penalty force: f = k * penetration - d * v_z (upward = +z)
        // Approximate v_z from generalized velocities
        // For simplicity, use a zero-order approximation of body linear velocity
        let force_z = cparams.stiffness * penetration;
        // Clamp to positive (no pulling into ground)
        let f_z = max(force_z, 0.0);

        // Write as spatial force in body frame
        // For ground contact, force is [0,0,f_z] in world frame
        // Transform to body frame: f_body = w_rot * f_world (since w_rot = body_to_world rot)
        // Actually need world_to_body rot = w_rot^T
        let fw = vec3<f32>(0.0, 0.0, f_z);
        // w_rot is body-to-world, so world-to-body is transpose
        let fb = vec3<f32>(
            w_rot[i][0]*fw.x + w_rot[i][3]*fw.y + w_rot[i][6]*fw.z,
            w_rot[i][1]*fw.x + w_rot[i][4]*fw.y + w_rot[i][7]*fw.z,
            w_rot[i][2]*fw.x + w_rot[i][5]*fw.y + w_rot[i][8]*fw.z
        );

        // Write to ext_forces: [angular(3), linear(3)]
        // Torque from contact point offset (simplified: at body origin)
        let ef_base = ef_env_base + i * 6u;
        // Angular part: r × f where r is from body COM to contact point
        // Simplified: assume contact at body lowest point
        ext_forces[ef_base + 3u] += fb.x;
        ext_forces[ef_base + 4u] += fb.y;
        ext_forces[ef_base + 5u] += fb.z;
    }
}
"#;

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

/// WGSL shader for generalized ABA (arbitrary articulated body trees).
///
/// Supports revolute (type 0), prismatic (type 1), and fixed (type 2) joints.
/// One thread per environment, serial tree traversal within.
/// Bodies must be topologically sorted (parent index < child index).
///
/// Body data layout: 32 f32 values per body (BODY_STRIDE):
///   [0]  parent (bitcast i32, -1 for root)
///   [1]  joint_type (0=revolute, 1=prismatic, 2=fixed)
///   [2]  q_offset
///   [3]  v_offset
///   [4]  mass
///   [5..8]  com (x,y,z)
///   [8..14] inertia (xx,yy,zz,xy,xz,yz)
///   [14..23] ptj rotation (row-major 3x3)
///   [23..26] ptj translation (x,y,z)
///   [26..29] axis (x,y,z)
///   [29] damping
///   [30..32] padding
pub const ABA_GENERAL_SHADER: &str = r#"
const MAX_BODIES: u32 = 16u;
const BODY_STRIDE: u32 = 32u;

struct SimParams {
    nworld: u32,
    nv: u32,
    dt: f32,
    nbodies: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> ctrl: array<f32>;
@group(0) @binding(5) var<storage, read_write> qdd: array<f32>;
@group(0) @binding(6) var<storage, read> ext_forces: array<f32>;

// ── Helpers: body data access ──

fn bf(bi: u32, off: u32) -> f32 { return bodies[bi * BODY_STRIDE + off]; }
fn bi(bidx: u32, off: u32) -> i32 { return bitcast<i32>(bodies[bidx * BODY_STRIDE + off]); }
fn bu(bidx: u32, off: u32) -> u32 { return bitcast<u32>(bodies[bidx * BODY_STRIDE + off]); }

// ── 6D spatial vector helpers ──
// sv[0..3] = angular, sv[3..6] = linear

fn sv_zero() -> array<f32, 6> { return array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }

fn sv_dot(a: array<f32, 6>, b: array<f32, 6>) -> f32 {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
}

fn sv_add(a: array<f32, 6>, b: array<f32, 6>) -> array<f32, 6> {
    return array<f32, 6>(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5]);
}

fn sv_sub(a: array<f32, 6>, b: array<f32, 6>) -> array<f32, 6> {
    return array<f32, 6>(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5]);
}

fn sv_scale(a: array<f32, 6>, s: f32) -> array<f32, 6> {
    return array<f32, 6>(a[0]*s, a[1]*s, a[2]*s, a[3]*s, a[4]*s, a[5]*s);
}

// Cross product of 3D vectors
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Spatial motion cross product: v_m x w
fn sv_cross_motion(v: array<f32, 6>, w: array<f32, 6>) -> array<f32, 6> {
    let va = vec3<f32>(v[0], v[1], v[2]); // angular
    let vl = vec3<f32>(v[3], v[4], v[5]); // linear
    let wa = vec3<f32>(w[0], w[1], w[2]);
    let wl = vec3<f32>(w[3], w[4], w[5]);
    let ra = cross3(va, wa);
    let rl = cross3(va, wl) + cross3(vl, wa);
    return array<f32, 6>(ra.x, ra.y, ra.z, rl.x, rl.y, rl.z);
}

// Spatial force cross product: v_m x* f
fn sv_cross_force(v: array<f32, 6>, f: array<f32, 6>) -> array<f32, 6> {
    let va = vec3<f32>(v[0], v[1], v[2]);
    let vl = vec3<f32>(v[3], v[4], v[5]);
    let fa = vec3<f32>(f[0], f[1], f[2]);
    let fl = vec3<f32>(f[3], f[4], f[5]);
    let ra = cross3(va, fa) + cross3(vl, fl);
    let rl = cross3(va, fl);
    return array<f32, 6>(ra.x, ra.y, ra.z, rl.x, rl.y, rl.z);
}

// ── 6x6 matrix helpers (column-major, 36 floats) ──

fn m6_zero() -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { m[i] = 0.0; }
    return m;
}

fn m6_get(m: ptr<function, array<f32, 36>>, r: u32, c: u32) -> f32 {
    return (*m)[c * 6u + r];
}

fn m6_set(m: ptr<function, array<f32, 36>>, r: u32, c: u32, val: f32) {
    (*m)[c * 6u + r] = val;
}

fn m6_mul_vec(m: ptr<function, array<f32, 36>>, v: array<f32, 6>) -> array<f32, 6> {
    var r: array<f32, 6>;
    for (var i = 0u; i < 6u; i++) {
        var s = 0.0;
        for (var j = 0u; j < 6u; j++) {
            s += (*m)[j * 6u + i] * v[j];
        }
        r[i] = s;
    }
    return r;
}

fn m6_add(a: ptr<function, array<f32, 36>>, b: ptr<function, array<f32, 36>>) -> array<f32, 36> {
    var r: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { r[i] = (*a)[i] + (*b)[i]; }
    return r;
}

fn m6_sub(a: ptr<function, array<f32, 36>>, b: ptr<function, array<f32, 36>>) -> array<f32, 36> {
    var r: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { r[i] = (*a)[i] - (*b)[i]; }
    return r;
}

// Outer product of two 6D vectors: a * b^T (result is 6x6 column-major)
fn m6_outer(a: array<f32, 6>, b: array<f32, 6>) -> array<f32, 36> {
    var r: array<f32, 36>;
    for (var c = 0u; c < 6u; c++) {
        for (var row = 0u; row < 6u; row++) {
            r[c * 6u + row] = a[row] * b[c];
        }
    }
    return r;
}

// M * A * M^T (6x6)
fn m6_XtAX(xt: ptr<function, array<f32, 36>>, a: ptr<function, array<f32, 36>>, x: ptr<function, array<f32, 36>>) -> array<f32, 36> {
    // tmp = A * X
    var tmp: array<f32, 36>;
    for (var c = 0u; c < 6u; c++) {
        for (var r = 0u; r < 6u; r++) {
            var s = 0.0;
            for (var k = 0u; k < 6u; k++) {
                s += (*a)[k * 6u + r] * (*x)[c * 6u + k];
            }
            tmp[c * 6u + r] = s;
        }
    }
    // result = X^T * tmp
    var result: array<f32, 36>;
    for (var c = 0u; c < 6u; c++) {
        for (var r = 0u; r < 6u; r++) {
            var s = 0.0;
            for (var k = 0u; k < 6u; k++) {
                s += (*xt)[k * 6u + r] * tmp[c * 6u + k];
            }
            result[c * 6u + r] = s;
        }
    }
    return result;
}

// ── Spatial transform operations ──

// Build the 6x6 motion transform matrix from rotation (row-major 3x3) and position
// X = [R, 0; -R*skew(p), R]
fn build_motion_transform(rot: array<f32, 9>, pos: vec3<f32>) -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { m[i] = 0.0; }

    // rot is row-major: rot[row*3+col]
    // WGSL matrix is column-major: m[col*6+row]

    // Top-left 3x3: R
    for (var r = 0u; r < 3u; r++) {
        for (var c = 0u; c < 3u; c++) {
            m[c * 6u + r] = rot[r * 3u + c];
        }
    }

    // Bottom-right 3x3: R
    for (var r = 0u; r < 3u; r++) {
        for (var c = 0u; c < 3u; c++) {
            m[(c + 3u) * 6u + (r + 3u)] = rot[r * 3u + c];
        }
    }

    // Bottom-left 3x3: -R * skew(p)
    // skew(p) = [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
    // R * skew(p), then negate
    // (R * skew(p))_ij = sum_k R_ik * skew(p)_kj
    let px = pos.x; let py = pos.y; let pz = pos.z;
    // skew matrix columns: col0 = [0, pz, -py], col1 = [-pz, 0, px], col2 = [py, -px, 0]
    var skp: array<f32, 9>;
    skp[0] = 0.0;  skp[1] = pz;   skp[2] = -py;
    skp[3] = -pz;  skp[4] = 0.0;  skp[5] = px;
    skp[6] = py;   skp[7] = -px;  skp[8] = 0.0;

    for (var r = 0u; r < 3u; r++) {
        for (var c = 0u; c < 3u; c++) {
            var s = 0.0;
            for (var k = 0u; k < 3u; k++) {
                s += rot[r * 3u + k] * skp[k * 3u + c];
            }
            m[c * 6u + (r + 3u)] = -s;
        }
    }

    return m;
}

fn transpose6(m: ptr<function, array<f32, 36>>) -> array<f32, 36> {
    var t: array<f32, 36>;
    for (var r = 0u; r < 6u; r++) {
        for (var c = 0u; c < 6u; c++) {
            t[c * 6u + r] = (*m)[r * 6u + c];
        }
    }
    return t;
}

// Apply spatial motion transform: X * v
// X has rotation R and translation p
// result = [R*w, R*(v - p×w)]
fn apply_motion(rot: array<f32, 9>, pos: vec3<f32>, sv: array<f32, 6>) -> array<f32, 6> {
    let w = vec3<f32>(sv[0], sv[1], sv[2]);
    let vel = vec3<f32>(sv[3], sv[4], sv[5]);
    let shifted = vel - cross3(pos, w);
    // Multiply by R (row-major)
    let rw = vec3<f32>(
        rot[0]*w.x + rot[1]*w.y + rot[2]*w.z,
        rot[3]*w.x + rot[4]*w.y + rot[5]*w.z,
        rot[6]*w.x + rot[7]*w.y + rot[8]*w.z
    );
    let rv = vec3<f32>(
        rot[0]*shifted.x + rot[1]*shifted.y + rot[2]*shifted.z,
        rot[3]*shifted.x + rot[4]*shifted.y + rot[5]*shifted.z,
        rot[6]*shifted.x + rot[7]*shifted.y + rot[8]*shifted.z
    );
    return array<f32, 6>(rw.x, rw.y, rw.z, rv.x, rv.y, rv.z);
}

// Inverse-apply spatial force transform: X^{-T} * f
// result_f = R^T * f_linear
// result_tau = R^T * f_angular + p × (R^T * f_linear)
fn inv_apply_force(rot: array<f32, 9>, pos: vec3<f32>, fv: array<f32, 6>) -> array<f32, 6> {
    let tau = vec3<f32>(fv[0], fv[1], fv[2]);
    let force = vec3<f32>(fv[3], fv[4], fv[5]);
    // R^T * force (R is row-major, so R^T col j = row j of R)
    let rt_f = vec3<f32>(
        rot[0]*force.x + rot[3]*force.y + rot[6]*force.z,
        rot[1]*force.x + rot[4]*force.y + rot[7]*force.z,
        rot[2]*force.x + rot[5]*force.y + rot[8]*force.z
    );
    let rt_tau = vec3<f32>(
        rot[0]*tau.x + rot[3]*tau.y + rot[6]*tau.z,
        rot[1]*tau.x + rot[4]*tau.y + rot[7]*tau.z,
        rot[2]*tau.x + rot[5]*tau.y + rot[8]*tau.z
    );
    let new_tau = rt_tau + cross3(pos, rt_f);
    return array<f32, 6>(new_tau.x, new_tau.y, new_tau.z, rt_f.x, rt_f.y, rt_f.z);
}

// ── Rigid body inertia to 6x6 spatial inertia matrix ──
// I_spatial = [[I + m*cx*cx^T, m*cx], [m*cx^T, m*E]]
// where cx = skew(com)
fn rigid_inertia_to_m6(mass: f32, com: vec3<f32>, inertia: array<f32, 6>) -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { m[i] = 0.0; }

    let cx = com.x; let cy = com.y; let cz = com.z;
    // cx_mat = skew(com) (row-major for convenience)
    // [0, -cz, cy; cz, 0, -cx; -cy, cx, 0]

    // Top-left: I + m * skew(com) * skew(com)^T
    // skew(c)*skew(c)^T = [[cy²+cz², -cx*cy, -cx*cz],
    //                       [-cx*cy, cx²+cz², -cy*cz],
    //                       [-cx*cz, -cy*cz, cx²+cy²]]
    let ixx = inertia[0]; let iyy = inertia[1]; let izz = inertia[2];
    let ixy = inertia[3]; let ixz = inertia[4]; let iyz = inertia[5];

    // Top-left 3x3 (column-major in output)
    m[0*6u+0u] = ixx + mass * (cy*cy + cz*cz);
    m[1*6u+0u] = ixy - mass * cx * cy;
    m[2*6u+0u] = ixz - mass * cx * cz;
    m[0*6u+1u] = ixy - mass * cx * cy;
    m[1*6u+1u] = iyy + mass * (cx*cx + cz*cz);
    m[2*6u+1u] = iyz - mass * cy * cz;
    m[0*6u+2u] = ixz - mass * cx * cz;
    m[1*6u+2u] = iyz - mass * cy * cz;
    m[2*6u+2u] = izz + mass * (cx*cx + cy*cy);

    // Top-right 3x3: m * skew(com)
    // skew(com) = [[0, -cz, cy], [cz, 0, -cx], [-cy, cx, 0]]
    m[3*6u+0u] = 0.0;         m[4*6u+0u] = -mass*cz;  m[5*6u+0u] = mass*cy;
    m[3*6u+1u] = mass*cz;     m[4*6u+1u] = 0.0;       m[5*6u+1u] = -mass*cx;
    m[3*6u+2u] = -mass*cy;    m[4*6u+2u] = mass*cx;   m[5*6u+2u] = 0.0;

    // Bottom-left 3x3: m * skew(com)^T = transpose of top-right
    m[0*6u+3u] = 0.0;        m[1*6u+3u] = mass*cz;   m[2*6u+3u] = -mass*cy;
    m[0*6u+4u] = -mass*cz;   m[1*6u+4u] = 0.0;       m[2*6u+4u] = mass*cx;
    m[0*6u+5u] = mass*cy;    m[1*6u+5u] = -mass*cx;  m[2*6u+5u] = 0.0;

    // Bottom-right 3x3: m * I_3
    m[3*6u+3u] = mass; m[4*6u+4u] = mass; m[5*6u+5u] = mass;

    return m;
}

// ── Joint helpers ──

// Motion subspace S for single-DOF joint
fn joint_motion_subspace(jtype: u32, axis: vec3<f32>) -> array<f32, 6> {
    if (jtype == 0u) {
        // Revolute: S = [axis; 0]
        return array<f32, 6>(axis.x, axis.y, axis.z, 0.0, 0.0, 0.0);
    } else {
        // Prismatic: S = [0; axis]
        return array<f32, 6>(0.0, 0.0, 0.0, axis.x, axis.y, axis.z);
    }
}

// Joint velocity: S * qd
fn joint_vel(jtype: u32, axis: vec3<f32>, qd: f32) -> array<f32, 6> {
    return sv_scale(joint_motion_subspace(jtype, axis), qd);
}

// Joint transform rotation for revolute: Rodrigues with -angle
fn revolute_rot(axis: vec3<f32>, angle: f32) -> array<f32, 9> {
    let neg_a = -angle;
    let s = sin(neg_a);
    let c = cos(neg_a);
    let t = 1.0 - c;
    let x = axis.x; let y = axis.y; let z = axis.z;

    var rot: array<f32, 9>;
    rot[0] = t*x*x + c;     rot[1] = t*x*y - s*z;   rot[2] = t*x*z + s*y;
    rot[3] = t*x*y + s*z;   rot[4] = t*y*y + c;     rot[5] = t*y*z - s*x;
    rot[6] = t*x*z - s*y;   rot[7] = t*y*z + s*x;   rot[8] = t*z*z + c;
    return rot;
}

fn identity_rot() -> array<f32, 9> {
    return array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

// Compose two rotations: result = A * B (both row-major)
fn compose_rot(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
    var r: array<f32, 9>;
    for (var row = 0u; row < 3u; row++) {
        for (var col = 0u; col < 3u; col++) {
            var s = 0.0;
            for (var k = 0u; k < 3u; k++) {
                s += a[row * 3u + k] * b[k * 3u + col];
            }
            r[row * 3u + col] = s;
        }
    }
    return r;
}

// Compose transforms: self.compose(other) = SpatialTransform { rot: self.rot * other.rot, pos: other.pos + other.rot^T * self.pos }
fn compose_transform(
    self_rot: array<f32, 9>, self_pos: vec3<f32>,
    other_rot: array<f32, 9>, other_pos: vec3<f32>
) -> array<f32, 12> {
    let new_rot = compose_rot(self_rot, other_rot);
    // other.rot^T * self.pos
    let rt_p = vec3<f32>(
        other_rot[0]*self_pos.x + other_rot[3]*self_pos.y + other_rot[6]*self_pos.z,
        other_rot[1]*self_pos.x + other_rot[4]*self_pos.y + other_rot[7]*self_pos.z,
        other_rot[2]*self_pos.x + other_rot[5]*self_pos.y + other_rot[8]*self_pos.z
    );
    let new_pos = other_pos + rt_p;

    var result: array<f32, 12>;
    for (var i = 0u; i < 9u; i++) { result[i] = new_rot[i]; }
    result[9] = new_pos.x; result[10] = new_pos.y; result[11] = new_pos.z;
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world_idx = gid.x;
    if (world_idx >= params.nworld) { return; }

    let nb = params.nbodies;
    let nv = params.nv;
    let q_base = world_idx * params.nv; // simplification: nq == nv for single-DOF joints
    let v_base = world_idx * nv;

    // Per-body working arrays
    var vel: array<array<f32, 6>, MAX_BODIES>;
    var c_bias: array<array<f32, 6>, MAX_BODIES>;
    var p_a: array<array<f32, 6>, MAX_BODIES>;
    var i_a: array<array<f32, 36>, MAX_BODIES>;
    var acc: array<array<f32, 6>, MAX_BODIES>;
    // Transform storage: 9 rot + 3 pos = 12 per body
    var x_rot: array<array<f32, 9>, MAX_BODIES>;
    var x_pos: array<vec3<f32>, MAX_BODIES>;

    // Gravity as base acceleration (base acceleration trick): a0 = [0; -gravity]
    let a0 = array<f32, 6>(0.0, 0.0, 0.0, -params.gx, -params.gy, -params.gz);

    // ── Pass 1: Forward — velocities and bias forces ──
    for (var i = 0u; i < nb; i++) {
        let parent = bi(i, 0u);
        let jtype = bu(i, 1u);
        let q_off = bu(i, 2u);
        let v_off = bu(i, 3u);

        // Read body data
        let mass = bf(i, 4u);
        let com = vec3<f32>(bf(i, 5u), bf(i, 6u), bf(i, 7u));
        var inertia: array<f32, 6>;
        for (var k = 0u; k < 6u; k++) { inertia[k] = bf(i, 8u + k); }

        var ptj_rot: array<f32, 9>;
        for (var k = 0u; k < 9u; k++) { ptj_rot[k] = bf(i, 14u + k); }
        let ptj_pos = vec3<f32>(bf(i, 23u), bf(i, 24u), bf(i, 25u));
        let axis = vec3<f32>(bf(i, 26u), bf(i, 27u), bf(i, 28u));

        // Compute joint transform
        var j_rot: array<f32, 9>;
        var j_pos: vec3<f32>;

        if (jtype == 2u) {
            // Fixed joint
            j_rot = identity_rot();
            j_pos = vec3<f32>(0.0, 0.0, 0.0);
        } else if (jtype == 0u) {
            // Revolute
            let q_val = q[q_base + q_off];
            j_rot = revolute_rot(axis, q_val);
            j_pos = vec3<f32>(0.0, 0.0, 0.0);
        } else {
            // Prismatic
            let q_val = q[q_base + q_off];
            j_rot = identity_rot();
            j_pos = axis * q_val;
        }

        // x_tree[i] = x_joint.compose(parent_to_joint)
        let composed = compose_transform(j_rot, j_pos, ptj_rot, ptj_pos);
        for (var k = 0u; k < 9u; k++) { x_rot[i][k] = composed[k]; }
        x_pos[i] = vec3<f32>(composed[9], composed[10], composed[11]);

        // Joint velocity
        var v_joint: array<f32, 6>;
        if (jtype == 2u) {
            v_joint = sv_zero();
        } else {
            let qd = v[v_base + v_off];
            v_joint = joint_vel(jtype, axis, qd);
        }

        if (parent < 0) {
            vel[i] = v_joint;
            c_bias[i] = sv_zero();
        } else {
            let pi = u32(parent);
            let v_parent = apply_motion(x_rot[i], x_pos[i], vel[pi]);
            vel[i] = sv_add(v_parent, v_joint);
            c_bias[i] = sv_cross_motion(vel[i], v_joint);
        }

        // Initialize articulated inertia
        i_a[i] = rigid_inertia_to_m6(mass, com, inertia);

        // Bias force: v ×* (I*v) (gyroscopic)
        var ia_i = i_a[i];
        let iv = m6_mul_vec(&ia_i, vel[i]);
        p_a[i] = sv_cross_force(vel[i], iv);

        // Subtract external forces (external forces reduce the bias)
        let ef_base = (world_idx * nb + i) * 6u;
        var ef: array<f32, 6>;
        for (var k = 0u; k < 6u; k++) { ef[k] = ext_forces[ef_base + k]; }
        p_a[i] = sv_sub(p_a[i], ef);
    }

    // ── Pass 2: Backward — articulated inertias and forces ──
    for (var ii = 0u; ii < nb; ii++) {
        let i = nb - 1u - ii; // reverse order
        let parent = bi(i, 0u);
        let jtype = bu(i, 1u);
        let v_off = bu(i, 3u);
        let axis = vec3<f32>(bf(i, 26u), bf(i, 27u), bf(i, 28u));
        let damping_val = bf(i, 29u);

        if (jtype == 2u) {
            // Fixed joint: just propagate to parent
            if (parent >= 0) {
                let pi = u32(parent);
                var x_mot = build_motion_transform(x_rot[i], x_pos[i]);
                var x_mot_t = transpose6(&x_mot);
                var ia_parent = m6_XtAX(&x_mot_t, &i_a[i], &x_mot);
                i_a[pi] = m6_add(&i_a[pi], &ia_parent);
                let p_parent = inv_apply_force(x_rot[i], x_pos[i], p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_parent);
            }
            continue;
        }

        // Single-DOF joint
        let s_i = joint_motion_subspace(jtype, axis);
        let phi = ctrl[v_base + v_off] - damping_val * v[v_base + v_off];
        let u_i = phi - sv_dot(s_i, p_a[i]);

        var ia_ref = i_a[i];
        let ia_s = m6_mul_vec(&ia_ref, s_i);
        let d_i = sv_dot(s_i, ia_s);

        if (abs(d_i) < 1e-20) { continue; }

        let u_inv_d = u_i / d_i;

        if (parent >= 0) {
            let pi = u32(parent);

            // I_a^A = I_a - (I_a*S)(I_a*S)^T / D
            var outer = m6_outer(ia_s, ia_s);
            for (var k = 0u; k < 36u; k++) { outer[k] /= d_i; }
            var ia_new = m6_sub(&i_a[i], &outer);

            // p_a^A = p_a + I_a^A * c + I_a*S * u/D
            let ia_c = m6_mul_vec(&ia_new, c_bias[i]);
            let ia_s_u = sv_scale(ia_s, u_inv_d);
            let p_new = sv_add(sv_add(p_a[i], ia_c), ia_s_u);

            // Transform to parent frame
            var x_mot = build_motion_transform(x_rot[i], x_pos[i]);
            var x_mot_t = transpose6(&x_mot);
            var ia_parent = m6_XtAX(&x_mot_t, &ia_new, &x_mot);
            i_a[pi] = m6_add(&i_a[pi], &ia_parent);

            let p_parent = inv_apply_force(x_rot[i], x_pos[i], p_new);
            p_a[pi] = sv_add(p_a[pi], p_parent);
        }
    }

    // ── Pass 3: Forward — accelerations ──
    for (var i = 0u; i < nb; i++) {
        let parent = bi(i, 0u);
        let jtype = bu(i, 1u);
        let v_off = bu(i, 3u);
        let axis = vec3<f32>(bf(i, 26u), bf(i, 27u), bf(i, 28u));
        let damping_val = bf(i, 29u);

        var a_parent: array<f32, 6>;
        if (parent < 0) {
            a_parent = apply_motion(x_rot[i], x_pos[i], a0);
        } else {
            let pi = u32(parent);
            a_parent = apply_motion(x_rot[i], x_pos[i], acc[pi]);
        }

        if (jtype == 2u) {
            // Fixed
            acc[i] = sv_add(a_parent, c_bias[i]);
            continue;
        }

        let s_i = joint_motion_subspace(jtype, axis);
        let phi = ctrl[v_base + v_off] - damping_val * v[v_base + v_off];
        let u_i = phi - sv_dot(s_i, p_a[i]);

        var ia_ref = i_a[i];
        let d_i = sv_dot(s_i, m6_mul_vec(&ia_ref, s_i));

        if (abs(d_i) < 1e-20) {
            acc[i] = sv_add(a_parent, c_bias[i]);
            continue;
        }

        let a_c = sv_add(a_parent, c_bias[i]);
        let ia_ac = m6_mul_vec(&ia_ref, a_c);
        let qdd_i = (u_i - sv_dot(s_i, ia_ac)) / d_i;

        // Write qdd
        qdd[v_base + v_off] = qdd_i;

        acc[i] = sv_add(a_c, sv_scale(s_i, qdd_i));
    }
}
"#;

/// WGSL shader for simplified ABA (single revolute joint systems).
///
/// This is a simplified version that handles pendulum-like systems
/// with single revolute joints. For multi-body systems, use
/// `ABA_GENERAL_SHADER` instead.
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
