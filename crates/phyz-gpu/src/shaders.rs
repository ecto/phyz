//! WGSL compute shaders for GPU-accelerated physics.

/// WGSL shader for FK + ground contact penalty forces.
///
/// Computes forward kinematics to get body world positions,
/// then detects ground plane contacts and writes penalty forces
/// to the external forces buffer (consumed by ABA shader).
///
/// One thread per environment, serial tree traversal within.
pub const CONTACT_GROUND_SHADER: &str = r#"
const MAX_BODIES: u32 = 32u;
const BODY_STRIDE: u32 = 36u;
const GEOM_STRIDE: u32 = 24u;
// Per-body contact slot: 8 floats of readback state, then MAX_PTS impulse
// vec3s. The impulses live in this buffer rather than their own because the
// WebGPU baseline allows 8 storage buffers per stage and the pass already
// binds 8; a ninth binding validates away on a conforming device.
const CS_STRIDE: u32 = 56u;
const CS_IMPULSE_OFF: u32 = 8u;
// The body-attached plane gets its own impulse slots: a body can rest on the
// deck and the ground in the same step (a wheel does exactly that), so the two
// contacts must not share a warm-start slot or they overwrite each other.
const CS_PLANE_OFF: u32 = 32u;

struct ContactParams {
    nworld: u32,
    nbodies: u32,
    nv: u32,
    ground_height: f32,
    dt: f32,
    friction: f32,
    // Body-attached contact plane; plane_body < 0 disables it.
    plane_body: i32,
    plane_offset: f32,
    plane_max_depth: f32,
    plane_half_x: f32,
    plane_half_y: f32,
    // Heightfield terrain. hf_nx == 0 means "no heightfield": the ground is
    // the flat plane at ground_height, exactly as before the feature.
    hf_nx: u32,
    hf_ny: u32,
    hf_ox: f32,
    hf_oy: f32,
    hf_oz: f32,
    hf_cell: f32,
    // ── Impulse solve ──
    // 0 = legacy penalty forces. 1 = velocity-level convex impulse solve,
    // the same problem `phyz_contact::solve_contacts` states. See the
    // IMPULSE MODE block below.
    solve_mode: u32,
    // Which sweep of the step this dispatch is. Sweep 0 seeds the impulses;
    // later sweeps refine them against a fresh `qdd`.
    sweep: u32,
    // Restitution, and the approach speed below which it ramps to zero.
    restitution: f32,
    restitution_threshold: f32,
    // solref/solimp, matching `phyz_contact::material`.
    solref_erp: f32,
    margin: f32,
    solimp_dmin: f32,
    solimp_dmax: f32,
    solimp_width: f32,
    solimp_mid: f32,
    solimp_power: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> cparams: ContactParams;
@group(0) @binding(1) var<storage, read> bodies: array<f32>;
@group(0) @binding(2) var<storage, read> geometry: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> ext_forces: array<f32>;
// Per-body contact state: [touching, penetration, point xyz, force xyz].
// Persists across steps so the previous penetration can supply a damping
// rate; every collidable body's slot is rewritten each pass.
@group(0) @binding(6) var<storage, read_write> contact_state: array<f32>;
// Heightfield nodes, row-major (iy * nx + ix). Bound even when there is no
// terrain — the binding count is fixed — with hf_nx == 0 routing around it.
@group(0) @binding(7) var<storage, read> hf_heights: array<f32>;
// Free acceleration from the ABA pass of this sweep. In impulse mode the
// contact solve reads it to form the free velocity `v + dt*qdd`; the whole
// scheme is built on this being the response to the impulses the PREVIOUS
// sweep wrote, which is what makes the iteration matrix-free.
@group(0) @binding(8) var<storage, read> qdd: array<f32>;
// Contact impulses share `contact_state`, at CS_IMPULSE_OFF: one vec3 per
// (body, contact point) slot, in that contact's own frame
// [normal, tangent u, tangent w].
//
// Slot identity is (body, corner index), which is stable across sweeps AND
// across steps for as long as the geometry is, so the previous step's
// impulses warm-start this one — matching what `Simulator` does on the CPU
// with its `ContactCache`.

// Generalized velocity this pass should do kinematics with.
//
// Penalty mode wants the CURRENT velocity. Impulse mode wants the FREE
// velocity `v + dt*qdd` — the velocity the body would end the step with given
// everything already applied, including the contact impulses the previous
// sweep wrote. That substitution is the entire reason this pass can evaluate
// the Delassus residual without ever forming the Delassus operator.
fn vel(idx: u32) -> f32 {
    if (cparams.solve_mode == 0u) { return v[idx]; }
    return v[idx] + cparams.dt * qdd[idx];
}

// Body data access
fn bf(bi: u32, off: u32) -> f32 { return bodies[bi * BODY_STRIDE + off]; }
fn body_parent(bi: u32) -> i32 { return bitcast<i32>(bodies[bi * BODY_STRIDE]); }
fn body_jtype(bi: u32) -> u32 { return bitcast<u32>(bodies[bi * BODY_STRIDE + 1u]); }
fn body_qoff(bi: u32) -> u32 { return bitcast<u32>(bodies[bi * BODY_STRIDE + 2u]); }
fn body_voff(bi: u32) -> u32 { return bitcast<u32>(bodies[bi * BODY_STRIDE + 3u]); }

fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Coulomb friction, regularized by slip SPEED rather than by the normal
// damping coefficient.
//
// Capping friction at `d * vt` (with `d` the contact's normal damping) is
// the tempting form, and it is wrong: at the slip speeds a standing robot
// actually produces — millimetres per second — that cap sits orders of
// magnitude below `mu * f_n`, so a foot cannot hold a static sideways load
// and creeps. Measured on the K1's skate stance: hip-roll spread pushes the
// feet apart, and the left foot slid 16 cm across the deck in 0.2 s and off
// the rail, while the CPU impulse solver held it indefinitely. That is not
// a penalty-vs-impulse difference; it is a missing static friction.
//
// `mu * f_n * min(1, vt / SLIP_EPS)` is the standard regularization: full
// Coulomb above a 1 mm/s slip, linear (and therefore stable) below it.
const SLIP_EPS: f32 = 1e-3;
// Contact slots per body: a box contacts through all eight corners, every
// other shape through one. Sizing the impulse buffer by this keeps slot
// identity (body, corner) stable across sweeps and across steps.
const MAX_PTS: u32 = 8u;

fn coulomb(mu_fn: f32, vt: f32) -> f32 {
    return mu_fn * min(1.0, vt / SLIP_EPS);
}

// ── Impulse-like bounds on the dissipative contact terms ──
//
// A penalty contact's SPRING is bounded by omega*dt; its DAMPERS are bounded
// by something else entirely, and that bound is what a skateboard exposed.
//
// Both dissipative terms here — the normal damper `d*v_n`, and the friction
// regularization `mu*f_n*vt/SLIP_EPS`, which is a tangential damper of slope
// `mu*f_n/SLIP_EPS` — are integrated explicitly. An explicit damper is stable
// only while `d*dt <= m_eff`, where `m_eff` is the mass the contact point
// actually presents along the force. Above that the damper overshoots: it
// does not merely stop the relative motion, it REVERSES it, larger every
// step.
//
// Measured, on the pop that this exists for: a skate wheel has a 0.1 kg mass
// and a 2.9e-5 kg m^2 spin inertia at a 27 mm radius, while its gains were
// sized for the ~5.75 kg it carries. `mu*f_n/SLIP_EPS` came to 5.7e4 N s/m,
// or 41 N m s/rad about the axle against 2.9e-5 — an explicit decay rate of
// 1.4e6 /s at dt = 1 ms, so the wheel's spin was multiplied by about -1400
// per step. Four wheels reached 19000 rad/s in twenty steps and the state was
// NaN by 0.02 s. The pop never happened; `f64::max` skipped the NaN frames
// and reported frame 0's numbers as the peak.
//
// The fix is the one an impulse solver gets for free: a dissipative impulse
// may bring the relative velocity to zero and no further. Capping each damper
// at `m_eff/dt` is exactly that cap, and it is unconditionally stable for any
// gain. Where the old gains were already stable the cap does not bind, so the
// quiet regime is untouched.
//
// `m_eff` is the FREE-BODY effective mass at the contact point — articulation
// is ignored, because the contact pass has no articulated inertia to hand.
// Attaching a body to a chain generally raises the inertia it presents, so
// this under-estimates, which errs toward damping too little. Too little
// damping is a softer contact; too much is divergence.

// I^-1 * w for the body's rotational inertia about its COM, body frame.
// Stored as (xx, yy, zz, xy, xz, yz) at [8..14]; inverted by cofactors.
fn inertia_solve(bidx: u32, w: vec3<f32>) -> vec3<f32> {
    let xx = bf(bidx, 8u); let yy = bf(bidx, 9u); let zz = bf(bidx, 10u);
    let xy = bf(bidx, 11u); let xz = bf(bidx, 12u); let yz = bf(bidx, 13u);
    let c00 = yy * zz - yz * yz;
    let c01 = xz * yz - xy * zz;
    let c02 = xy * yz - xz * yy;
    let det = xx * c00 + xy * c01 + xz * c02;
    // A massless or degenerate body carries no rotational inertia; report an
    // unbounded angular response rather than dividing by zero.
    if (abs(det) < 1e-20) { return vec3<f32>(0.0, 0.0, 0.0); }
    let c11 = xx * zz - xz * xz;
    let c12 = xy * xz - xx * yz;
    let c22 = xx * yy - xy * xy;
    let inv = 1.0 / det;
    return vec3<f32>(
        inv * (c00 * w.x + c01 * w.y + c02 * w.z),
        inv * (c01 * w.x + c11 * w.y + c12 * w.z),
        inv * (c02 * w.x + c12 * w.y + c22 * w.z),
    );
}

// Effective mass this body presents at contact offset `r` (from the body
// ORIGIN, body frame) along unit direction `u` (body frame):
//
//     1/m_eff = 1/m + (r_c x u)^T I^-1 (r_c x u),   r_c = r - com
//
// the standard contact-point effective mass. A massless body is treated as
// immovable, which is what the world body is.
fn contact_eff_mass(bidx: u32, r: vec3<f32>, u: vec3<f32>) -> f32 {
    let m = bf(bidx, 4u);
    if (m <= 0.0) { return 1e30; }
    let rc = r - vec3<f32>(bf(bidx, 5u), bf(bidx, 6u), bf(bidx, 7u));
    let a = cross3(rc, u);
    let ang = max(dot(a, inertia_solve(bidx, a)), 0.0);
    return 1.0 / (1.0 / m + ang);
}

// The largest damping coefficient an explicit step may apply against `m_eff`
// without reversing the velocity it is meant to remove.
fn max_damping(m_eff: f32) -> f32 {
    return m_eff / max(cparams.dt, 1e-9);
}

// The largest spring an explicit step may carry against `m_eff`: semi-implicit
// Euler is stable while `w*dt < 2`, i.e. `k*dt^2/m_eff < 4`.
//
// The damping bound alone leaves this one standing, and it is the next wall a
// pop hits. Measured: a skate wheel presents 0.1 kg to its ground contact while
// its gains are sized for 5.75 kg, so `k = 5.75*w^2` crosses `4*m/dt^2` at
// w = 264 — and the sweep degrades between 250 and 275, exactly there. Bounding
// it lets the FOOT contact, which presents far more mass and so has a much
// higher wall of its own, go on stiffening past the point where the wheel has
// stopped. A stiffer contact stores less energy and returns less of it, which
// is what moves a penalty pop toward the impulse solver's.
fn max_stiffness(m_eff: f32) -> f32 {
    let dt = max(cparams.dt, 1e-9);
    return 4.0 * m_eff / (dt * dt);
}



// ── IMPULSE MODE ──
//
// The same convex problem `phyz_contact::convex` states:
//
//     minimize_f  1/2 f^T (A + R) f + f^T b     s.t.  f_c in K_mu(c)
//
// solved by the same staged Coulomb update — normal impulse first, then the
// tangential impulse clamped into the friction disc that normal admits, so
// stiction is the interior of a genuine second-order cone rather than a
// viscous damper that vanishes as the slip does.
//
// # Why this is matrix-free, and why that matters
//
// A GPU cannot afford to ASSEMBLE the Delassus operator `A = J M^-1 J^T`:
// each of the `3n` rows needs its own articulated-body solve. But projected
// Gauss-Seidel never needs `A` as a matrix — it needs two things: the
// residual `A f + b`, and a diagonal to divide by.
//
// The residual comes for free from the pass structure. The host runs
// [contact, ABA] in a loop, so by the time this shader runs again the ABA
// pass has already propagated the impulses this shader last wrote through
// the FULL articulated chain. Reading `v + dt*qdd` at a contact point IS
// evaluating `(A f + b)_c` — exactly, with the true `M^-1`, including every
// cross-contact and cross-chain coupling term. Nothing is dropped.
//
// The diagonal is the only approximation, and it is a PRECONDITIONER, not
// physics: it sets the step size, and the fixed point of the iteration is
// determined entirely by the residual. So using the cheap isolated-body
// effective mass here does not bias the answer the sweeps converge to; it
// only affects how fast they get there. That is the whole reason this design
// beats assembling an approximate `A`, which would move the fixed point and
// therefore the physics.
//
// The step-size direction is the safe one by construction. An isolated body
// always presents LESS mass than the same body backed by its chain (a foot
// alone is lighter than foot-plus-robot), so `a_nn = 1/m_isolated` is an
// OVER-estimate of the true diagonal, and dividing by it under-relaxes.
// Under-relaxation converges slowly; over-relaxation diverges. The cheap
// number errs toward the stable side every time.
//
// What remains approximate, and is documented as such: the sweep budget is
// finite and has no early exit, so the iterate is not a converged KKT point.
// `ContactSolverConfig::gpu_equivalent()` reproduces that budget exactly on
// the CPU so the two can be compared without confounding it with a bug.

// An orthonormal contact frame around `n`. Must match
// `phyz_contact::cone::contact_frame`, or the two engines' tangential
// impulses live in different bases and comparing them is meaningless.
fn contact_tangents(n: vec3<f32>) -> mat2x3<f32> {
    var a = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(n.x) > 0.9) { a = vec3<f32>(0.0, 1.0, 0.0); }
    let u = normalize(cross(n, a));
    let w = cross(n, u);
    return mat2x3<f32>(u, w);
}

// SolImp impedance, mirroring `phyz_contact::material::SolImp::impedance`
// branch for branch. Gets its own function so the two can be diffed by eye.
fn solimp_impedance(r: f32) -> f32 {
    let dmin = clamp(cparams.solimp_dmin, 1e-4, 1.0 - 1e-9);
    let dmax = clamp(cparams.solimp_dmax, 1e-4, 1.0 - 1e-9);
    if (cparams.solimp_width <= 0.0) { return dmax; }
    let x = clamp(abs(r) / cparams.solimp_width, 0.0, 1.0);
    let mid = clamp(cparams.solimp_mid, 1e-6, 1.0 - 1e-6);
    let pw = max(cparams.solimp_power, 1.0);
    var y: f32;
    if (x <= mid) { y = pow(x, pw) / pow(mid, pw - 1.0); }
    else { y = 1.0 - pow(1.0 - x, pw) / pow(1.0 - mid, pw - 1.0); }
    return dmin + y * (dmax - dmin);
}

// `ContactMaterial::impedance_at`: solimp on the penetrating side, and a
// smoothstep ramp to zero across the margin on the separated side.
//
// The margin is not a detection tolerance, it is part of the model. A contact
// that is detected but not yet penetrating still carries force, tapering to
// zero over the band, which is what keeps a lightly-loaded support point from
// being cut off while it is still holding something up. The GPU had no margin
// at all, so its contact set switched discontinuously where the CPU's faded —
// measured as a 0.24 m divergence on a tumbling box, which more sweeps made
// WORSE because the solver was converging accurately to a different problem.
fn impedance_at(depth: f32) -> f32 {
    if (depth >= 0.0) { return solimp_impedance(depth); }
    let gap = -depth;
    if (cparams.margin <= 0.0 || gap >= cparams.margin) { return 0.0; }
    let sc = 1.0 - gap / cparams.margin;
    return sc * sc * (3.0 - 2.0 * sc) * solimp_impedance(0.0);
}

// Effective restitution after the smooth low-speed ramp. Mirrors
// `ContactProblem::effective_restitution`: smoothstep between v_rest and
// 2*v_rest, so it stays C^1 in the approach speed rather than switching.
fn effective_restitution(e: f32, approach: f32) -> f32 {
    let vr = cparams.restitution_threshold;
    if (vr <= 0.0) { return e; }
    let sp = abs(approach);
    if (sp <= vr) { return 0.0; }
    if (sp >= 2.0 * vr) { return e; }
    let t = (sp - vr) / vr;
    return e * t * t * (3.0 - 2.0 * t);
}

// ── Heightfield terrain ──
//
// Mirrors phyz_model::Heightfield exactly: node (ix, iy) at
// (hf_ox + ix*cell, hf_oy + iy*cell), height hf_oz + hf_heights[iy*nx + ix],
// bilinear between nodes, border-clamped outside the grid — with a ZERO
// slope beyond the border, matching the clamped (flat) surface there rather
// than the border cell's slope, which describes a surface that no longer
// exists. The CPU detector samples the same f32 node buffer, so both
// engines stand on identical terrain rather than terrain that agrees to
// rounding.

fn hf_node(ix: u32, iy: u32) -> f32 {
    return cparams.hf_oz + hf_heights[iy * cparams.hf_nx + ix];
}

// Cell index and intra-cell fraction along one axis, clamped to the grid.
fn hf_locate(w: f32, o: f32, n: u32) -> vec2<f32> {
    if (n < 2u) { return vec2<f32>(0.0, 0.0); }
    let u = clamp((w - o) / cparams.hf_cell, 0.0, f32(n - 1u));
    // A query exactly on the far border indexes the last cell at t = 1
    // rather than one past it.
    let i = min(u32(u), n - 2u);
    return vec2<f32>(f32(i), u - f32(i));
}

// Terrain sample at world (x, y): xyz = unit surface normal, w = height.
// With no heightfield loaded this is the flat plane at ground_height.
fn terrain(p: vec2<f32>) -> vec4<f32> {
    let nx = cparams.hf_nx;
    let ny = cparams.hf_ny;
    if (nx == 0u) {
        return vec4<f32>(0.0, 0.0, 1.0, cparams.ground_height);
    }
    let lx = hf_locate(p.x, cparams.hf_ox, nx);
    let ly = hf_locate(p.y, cparams.hf_oy, ny);
    let ix = u32(lx.x); let tx = lx.y;
    let iy = u32(ly.x); let ty = ly.y;
    let ix1 = min(ix + 1u, nx - 1u);
    let iy1 = min(iy + 1u, ny - 1u);

    let h00 = hf_node(ix, iy);
    let h10 = hf_node(ix1, iy);
    let h01 = hf_node(ix, iy1);
    let h11 = hf_node(ix1, iy1);
    let h = (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty)
          + (h01 * (1.0 - tx) + h11 * tx) * ty;

    // Analytic bilinear-patch gradient, zeroed outside the grid.
    var dhdx = 0.0;
    var dhdy = 0.0;
    let span_x = f32(nx - 1u) * cparams.hf_cell;
    let span_y = f32(ny - 1u) * cparams.hf_cell;
    if (nx >= 2u && p.x >= cparams.hf_ox && p.x <= cparams.hf_ox + span_x) {
        dhdx = ((h10 - h00) * (1.0 - ty) + (h11 - h01) * ty) / cparams.hf_cell;
    }
    if (ny >= 2u && p.y >= cparams.hf_oy && p.y <= cparams.hf_oy + span_y) {
        dhdy = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / cparams.hf_cell;
    }
    let n = normalize(vec3<f32>(-dhdx, -dhdy, 1.0));
    return vec4<f32>(n, h);
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

// Quaternion exponential (w, x, y, z), matching tang's Quat::exp.
fn cquat_exp(omega: vec3<f32>) -> vec4<f32> {
    let angle = length(omega);
    if (angle < 1e-6) {
        return vec4<f32>(1.0, omega.x * 0.5, omega.y * 0.5, omega.z * 0.5);
    }
    let half = angle * 0.5;
    let s = sin(half) / angle;
    return vec4<f32>(cos(half), omega.x * s, omega.y * s, omega.z * s);
}

// Quaternion to row-major rotation matrix.
fn cq_to_rot(qt: vec4<f32>) -> array<f32, 9> {
    let w = qt.x; let x = qt.y; let y = qt.z; let z = qt.w;
    return array<f32, 9>(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),
        2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)
    );
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

// R^T * v
fn rot_t_mul(r: array<f32, 9>, vv: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        r[0]*vv.x + r[3]*vv.y + r[6]*vv.z,
        r[1]*vv.x + r[4]*vv.y + r[7]*vv.z,
        r[2]*vv.x + r[5]*vv.y + r[8]*vv.z
    );
}

// Corner `c` (0..8) of body i's box, in BODY coordinates with the collision
// instance's origin applied. Boxes contact through all penetrating corners —
// a single support point lets an angled foot rock on one corner, which is
// exactly what felled the loose-stance rollouts.
fn box_corner(i: u32, c: u32) -> vec3<f32> {
    let gbase = i * GEOM_STRIDE;
    let h = vec3<f32>(geometry[gbase + 1u], geometry[gbase + 2u], geometry[gbase + 3u]);
    let sx = select(-1.0, 1.0, (c & 1u) != 0u);
    let sy = select(-1.0, 1.0, (c & 2u) != 0u);
    let sz = select(-1.0, 1.0, (c & 4u) != 0u);
    let corner = vec3<f32>(h.x * sx, h.y * sy, h.z * sz);
    let o_p = vec3<f32>(geometry[gbase + 10u], geometry[gbase + 11u], geometry[gbase + 12u]);
    var o_r: array<f32, 9>;
    for (var k = 0u; k < 9u; k++) { o_r[k] = geometry[gbase + 13u + k]; }
    return o_p + rot_t_mul(o_r, corner);
}

// Support point of body i's shape in the direction of -n_body (n_body a unit
// vector in the body's own frame): the point of the shape that reaches
// furthest against the contact normal, returned in BODY coordinates with the
// collision instance's own origin (offset + rotation) applied.
//
// The offset is not a detail: a K1 foot pad sits 2.6 cm forward of its
// ankle, and ignoring it costs the robot its whole sagittal support margin.
// The rotation is not either — an axis-aligned lowest point is only the true
// support point while the body is upright, which is exactly the case a
// test fixture starts in and a walking robot never stays in.
fn support_point(i: u32, n_body: vec3<f32>) -> vec3<f32> {
    let gbase = i * GEOM_STRIDE;
    let gtype = u32(geometry[gbase]);
    // Instance origin: pos at [10..13], rot (body -> shape, row-major) at [13..22].
    let o_p = vec3<f32>(geometry[gbase + 10u], geometry[gbase + 11u], geometry[gbase + 12u]);
    var o_r: array<f32, 9>;
    for (var k = 0u; k < 9u; k++) { o_r[k] = geometry[gbase + 13u + k]; }
    let n = rot_mul(o_r, n_body);

    var support = vec3<f32>(0.0, 0.0, 0.0);
    if (gtype == 1u) {
        support = -n * geometry[gbase + 1u];
    } else if (gtype == 2u) {
        let h = vec3<f32>(geometry[gbase + 1u], geometry[gbase + 2u], geometry[gbase + 3u]);
        support = vec3<f32>(-h.x * sign(n.x), -h.y * sign(n.y), -h.z * sign(n.z));
    } else if (gtype == 3u) {
        let radius = geometry[gbase + 1u];
        let half_len = geometry[gbase + 2u] * 0.5;
        support = vec3<f32>(0.0, 0.0, -half_len * sign(n.z)) - n * radius;
    } else if (gtype == 4u) {
        let radius = geometry[gbase + 1u];
        let half_h = geometry[gbase + 2u] * 0.5;
        let radial = vec3<f32>(-n.x, -n.y, 0.0);
        let rl = length(radial);
        var rim = vec3<f32>(0.0, 0.0, 0.0);
        if (rl > 1e-6) { rim = radial / rl * radius; }
        support = rim + vec3<f32>(0.0, 0.0, -half_h * sign(n.z));
    } else if (gtype == 5u) {
        // Mesh, via its body-frame AABB — asymmetric (min/max, not
        // half-extents), so an off-centre hull keeps its true offset.
        //
        // Resolved through sign(), like the box above, and that matters:
        // an AABB's support is DEGENERATE whenever a normal component is
        // zero (flat on the ground, the whole bottom face ties). sign(0)
        // is 0, which picks the face centre; a `select` on `n > 0` would
        // break the tie toward a corner instead, and the r x f torque
        // about that corner tips a body that should rest flat — measured
        // as a mesh cube launching to z = 1.27 m from a 0.5 m drop.
        let mn = vec3<f32>(geometry[gbase + 1u], geometry[gbase + 2u], geometry[gbase + 3u]);
        let mx = vec3<f32>(geometry[gbase + 4u], geometry[gbase + 5u], geometry[gbase + 6u]);
        let mc = (mn + mx) * 0.5;
        let mh = (mx - mn) * 0.5;
        support = mc - vec3<f32>(mh.x * sign(n.x), mh.y * sign(n.y), mh.z * sign(n.z));
    }
    return o_p + rot_t_mul(o_r, support);
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world_idx = gid.x;
    if (world_idx >= cparams.nworld) { return; }

    let nb = cparams.nbodies;
    let q_base = world_idx * cparams.nv;
    let v_base = world_idx * cparams.nv;

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
    // Body-frame spatial velocities (angular, linear), same recursion as the
    // CPU: v_i = X_tree_i * v_parent + S_i * qd_i. Contact damping and
    // friction both need the velocity of the contact POINT, which a
    // finite-differenced penetration cannot supply once a shape reports more
    // than one contact point — and friction cannot be had from it at all.
    var w_omega: array<vec3<f32>, MAX_BODIES>;
    var w_lin: array<vec3<f32>, MAX_BODIES>;

    for (var i = 0u; i < nb; i++) {
        let parent = body_parent(i);
        let jtype = body_jtype(i);
        let q_off = body_qoff(i);
        let v_off = body_voff(i);

        // Parent-to-joint transform
        var ptj_rot: array<f32, 9>;
        for (var k = 0u; k < 9u; k++) { ptj_rot[k] = bf(i, 14u + k); }
        let ptj_pos = vec3<f32>(bf(i, 23u), bf(i, 24u), bf(i, 25u));
        let axis = vec3<f32>(bf(i, 26u), bf(i, 27u), bf(i, 28u));

        // Joint transform. Ball and free joints must be handled here too: a
        // floating-base model whose root is treated as fixed puts every body at
        // the origin, so no foot ever reaches the ground and contact silently
        // does nothing.
        var j_rot: array<f32, 9>;
        var j_pos = vec3<f32>(0.0, 0.0, 0.0);
        var j_omega = vec3<f32>(0.0, 0.0, 0.0);
        var j_lin = vec3<f32>(0.0, 0.0, 0.0);
        if (jtype == 0u) {
            j_rot = rev_rot(axis, q[q_base + q_off]);
            j_omega = axis * vel(v_base + v_off);
        } else if (jtype == 1u) {
            j_rot = identity_rot();
            j_pos = axis * q[q_base + q_off];
            j_lin = axis * vel(v_base + v_off);
        } else if (jtype == 3u) {
            // Coordinate map is the INVERSE of the integrated rotation
            // (exp(-w)), matching the negated angle in rev_rot and the CPU
            // joint_transform_slice.
            let w = vec3<f32>(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
            j_rot = cq_to_rot(cquat_exp(-w));
            j_omega = vec3<f32>(vel(v_base + v_off), vel(v_base + v_off + 1u), vel(v_base + v_off + 2u));
        } else if (jtype == 4u) {
            // Free: q = [exp-coords(3), pos(3)] — angular first, matching v.
            let w = vec3<f32>(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
            j_rot = cq_to_rot(cquat_exp(-w));
            j_pos = vec3<f32>(q[q_base + q_off + 3u], q[q_base + q_off + 4u], q[q_base + q_off + 5u]);
            j_omega = vec3<f32>(vel(v_base + v_off), vel(v_base + v_off + 1u), vel(v_base + v_off + 2u));
            j_lin = vec3<f32>(vel(v_base + v_off + 3u), vel(v_base + v_off + 4u), vel(v_base + v_off + 5u));
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

        // tang's SpatialTransform convention (see phyz_math::SpatialTransformExt):
        // in x_world_to_body = x_tree.compose(x_world_to_parent), `rot` is the
        // world→body rotation and `pos` is the BODY ORIGIN IN WORLD coordinates
        // — pos is NOT a Plücker translation needing inversion. compose gives
        //   pos_i = pos_parent + rot_parentᵀ · tree_pos,
        // i.e. tree_pos is the child origin in the parent frame, carried out to
        // world by the parent's body→world rotation. The previous version here
        // treated tree_pos as needing `-rotᵀ·pos`, which put a free body at
        // height +z below the ground plane at -z: contact fired while airborne
        // and never fired on the floor.
        let tree_rt = array<f32, 9>(tree_rot[0], tree_rot[3], tree_rot[6],
                                     tree_rot[1], tree_rot[4], tree_rot[7],
                                     tree_rot[2], tree_rot[5], tree_rot[8]);
        if (parent < 0) {
            w_rot[i] = tree_rt; // body-to-world rotation = tree_rotᵀ
            w_pos[i] = tree_pos;
            w_omega[i] = j_omega;
            w_lin[i] = j_lin;
        } else {
            let pi = u32(parent);
            w_rot[i] = rot_compose(w_rot[pi], tree_rt);
            w_pos[i] = w_pos[pi] + rot_mul(w_rot[pi], tree_pos);
            // apply_motion, in the BODY frame: w_c = R*w_p,
            // v_c = R*v_p - R*(p x w_p). `tree_rot` is the world→body
            // (parent→child) rotation this needs — the untransposed twin of
            // the `tree_rt` the pose recursion above uses.
            let pw = rot_mul(tree_rot, w_omega[pi]);
            let pv = rot_mul(tree_rot, w_lin[pi])
                   - rot_mul(tree_rot, cross3(tree_pos, w_omega[pi]));
            w_omega[i] = pw + j_omega;
            w_lin[i] = pv + j_lin;
        }
    }

    // Check ground contacts for each body
    for (var i = 0u; i < nb; i++) {
        let gbase = i * GEOM_STRIDE;
        let gtype = u32(geometry[gbase]);
        // Slots are indexed by BODY index, matching readback_contacts, so a
        // body with no geometry still owns its slot — clear it here rather
        // than relying on the buffer never having been written, so the
        // "not touching" invariant holds without an allocation-order argument.
        let cs_base = (world_idx * nb + i) * CS_STRIDE;
        if (gtype == 0u) {
            for (var k = 0u; k < CS_STRIDE; k++) { contact_state[cs_base + k] = 0.0; }
            continue;
        }

        // World +Z expressed in this body's frame. Support selection, the
        // contact normal and every force below live in body coordinates,
        // matching what ext_forces wants (r x f and f, body frame).
        // Support-point SELECTION uses world "down" even on a heightfield —
        // the small-slope assumption the CPU detector documents: for the
        // shallow terrain a walking or skating robot trains on, down and the
        // local surface normal pick the same feature.
        let z_body = rot_t_mul(w_rot[i], vec3<f32>(0.0, 0.0, 1.0));

        // Boxes contact through EVERY penetrating corner; other shapes
        // through the single support point they actually touch at. A box
        // reduced to one point can rock on that corner with nothing to
        // resist it, which is what felled the loose-stance rollouts — a
        // resting box needs a real support polygon. Per-point stiffness is
        // a quarter of the body's, so a flat-resting face carries the same
        // total spring as a single-point shape does.
        var n_pts = 1u;
        var pt_scale = 1.0;
        if (gtype == 2u) { n_pts = 8u; pt_scale = 0.25; }

        // Accumulated for the readback slot: total force, and the deepest
        // point, which is the one worth reporting as THE contact.
        var f_w_total = vec3<f32>(0.0, 0.0, 0.0);
        var deepest = 0.0;
        var deepest_w = vec3<f32>(0.0, 0.0, 0.0);
        var any_touch = false;

        // How many of this body's points are actually in contact.
        //
        // This is the within-body load sharing, and in impulse mode it is not
        // optional. Every point of a rigid body's manifold pushes the SAME
        // mass, but each one only sees the body's total effective mass when
        // it sizes its own impulse. Left uncorrected, k coplanar corners each
        // apply the full correction and the body gets k times the impulse it
        // needed: measured on a box landing on its face, the eight corners
        // overshot by 8x and the state was 5e6 metres away within two seconds.
        //
        // Dividing the effective mass by the active count is the diagonal of
        // the within-body coupling block — the same term that, on the CPU,
        // is the difference between `ContactCoupling::BlockDiagonal` (up to
        // 78 mm of error) and `PerBody` (0.1 mm). It is the cheap half of the
        // Delassus operator, and this is where the GPU pays for it.
        var n_active = 0u;
        if (cparams.solve_mode == 1u) {
            for (var cq = 0u; cq < n_pts; cq++) {
                var sq: vec3<f32>;
                if (gtype == 2u) { sq = box_corner(i, cq); }
                else { sq = support_point(i, z_body); }
                let sq_w = w_pos[i] + rot_mul(w_rot[i], sq);
                let tq = terrain(sq_w.xy);
                if (tq.z * (tq.w - sq_w.z) > -cparams.margin) { n_active++; }
            }
            n_active = max(n_active, 1u);
        }

        for (var cpt = 0u; cpt < n_pts; cpt++) {
            var support: vec3<f32>;
            if (gtype == 2u) { support = box_corner(i, cpt); }
            else { support = support_point(i, z_body); }
            let sup_w = w_pos[i] + rot_mul(w_rot[i], support);

            // Terrain under this contact point (the flat plane when no
            // heightfield is loaded). Penetration is measured along the local
            // surface normal, and every force below acts along it — on a flat
            // field this reduces exactly to the old ground test.
            let terr = terrain(sup_w.xy);
            let n_w = terr.xyz;
            let penetration = n_w.z * (terr.w - sup_w.z);
            let in_contact = select(penetration > 0.0, penetration > -cparams.margin, cparams.solve_mode == 1u);
            if (!in_contact) {
                // Separated: drop any impulse this slot carried. Without this
                // the cross-step warm start would keep re-applying force at a
                // contact that has already left the ground.
                if (cparams.solve_mode == 1u) {
                    let dead = cs_base + CS_IMPULSE_OFF + cpt * 3u;
                    contact_state[dead] = 0.0;
                    contact_state[dead + 1u] = 0.0;
                    contact_state[dead + 2u] = 0.0;
                }
                continue;
            }
            let n_body = rot_t_mul(w_rot[i], n_w);

            // Velocity of the contact POINT, body frame. This replaces the
            // finite-differenced penetration rate the single-point model
            // used: that needed one history slot per contact and there is
            // only one slot per body, and it cannot produce a tangential
            // velocity at all, so friction was simply absent.
            let v_point = w_lin[i] + cross3(w_omega[i], support);
            let v_normal = dot(v_point, n_body);

            // ── Impulse mode: one staged Coulomb update for this slot ──
            if (cparams.solve_mode == 1u) {
                let slot = cs_base + CS_IMPULSE_OFF + cpt * 3u;
                // Sweep 0 of the very first step has nothing to warm-start
                // from; the buffer is zeroed at construction, so the seed is
                // simply whatever the previous step converged to.
                var f_c = vec3<f32>(contact_state[slot], contact_state[slot + 1u], contact_state[slot + 2u]);

                let tang = contact_tangents(n_body);
                let t_u = tang[0];
                let t_w = tang[1];

                // Free contact-space velocity. `v_point` is already built
                // from `v + dt*qdd`, so this IS (A f + b)_c with the true
                // articulated M^-1 — see the IMPULSE MODE block.
                let b_n = dot(v_point, n_body);
                let b_u = dot(v_point, t_u);
                let b_w = dot(v_point, t_w);

                // Diagonal preconditioner: the isolated-body effective mass.
                // An over-estimate of A_nn, hence under-relaxing, hence safe.
                let m_n = contact_eff_mass(i, support, n_body) / f32(n_active);
                let a_nn = 1.0 / max(m_n, 1e-9);

                // solref/solimp position stabilization, as
                // `ContactRow::from_material`: drive the post-step normal
                // velocity to a SEPARATING `bias` proportional to depth
                // rather than to zero, or accumulated penetration is frozen
                // in and a stack creeps down forever.
                let d_imp = impedance_at(penetration);
                let bias = d_imp * cparams.solref_erp * max(penetration, 0.0) / max(cparams.dt, 1e-9);

                // Restitution enters as a target normal velocity, folded into
                // b, rather than a post-solve velocity reset — the same
                // choice `point_mass_problem` makes, and what keeps it from
                // fighting the solver.
                let e = effective_restitution(cparams.restitution, min(b_n, 0.0));
                let b_n_eff = b_n * (1.0 + e);

                // The residual excludes this contact's own contribution,
                // which `v_point` already contains, so add it back: the
                // update is coordinate descent on the contact's own block.
                let r_n = b_n_eff - a_nn * f_c.x;
                var fn_new = max((bias - r_n) / a_nn, 0.0);

                // Tangential, at the normal just chosen, then clamped into
                // the disc of radius mu*f_n. Isotropic, so a contact sliding
                // at any heading loses speed identically — the property a
                // pyramidal cone gives up.
                // Tangential diagonals get their OWN effective masses. Reusing
                // the normal's is tempting and wrong: `contact_eff_mass`
                // depends on the lever arm `r x u`, which for a corner contact
                // points somewhere else entirely along a tangent than along
                // the normal. Measured on a tumbling box, sharing the normal's
                // mass left a 0.23 m slide error that more sweeps only
                // sharpened, because the solve was converging accurately to
                // the wrong tangential step size.
                let a_uu = f32(n_active) / max(contact_eff_mass(i, support, t_u), 1e-9);
                let a_ww = f32(n_active) / max(contact_eff_mass(i, support, t_w), 1e-9);
                let r_u = b_u - a_uu * f_c.y;
                let r_w = b_w - a_ww * f_c.z;
                var tu = -r_u / a_uu;
                var tw = -r_w / a_ww;
                // Clamp into the friction disc of radius mu*f_n, isotropically
                // — a contact sliding at any heading loses speed identically,
                // the property a pyramidal cone gives up.
                //
                // Re-aiming the clamped impulse along the slip direction was
                // tried and is WRONG here: `b_u`/`b_w` already contain this
                // contact's own current impulse, so using them as the slip
                // axis moves the fixed point rather than correcting it, and it
                // cost 4x on the high-friction slides while helping nothing.
                let limit = cparams.friction * fn_new;
                let tn = sqrt(tu * tu + tw * tw);
                if (tn > limit) {
                    let sc = select(0.0, limit / tn, tn > 0.0);
                    tu = tu * sc;
                    tw = tw * sc;
                }

                contact_state[slot] = fn_new;
                contact_state[slot + 1u] = tu;
                contact_state[slot + 2u] = tw;

                // Emit as a FORCE (impulse/dt) so the ABA pass, which speaks
                // in forces, propagates it through the chain unchanged.
                let f_body = (n_body * fn_new + t_u * tu + t_w * tw) / max(cparams.dt, 1e-9);
                let torque_i = cross3(support, f_body);
                let ef_b = ef_env_base + i * 6u;
                ext_forces[ef_b + 0u] += torque_i.x;
                ext_forces[ef_b + 1u] += torque_i.y;
                ext_forces[ef_b + 2u] += torque_i.z;
                ext_forces[ef_b + 3u] += f_body.x;
                ext_forces[ef_b + 4u] += f_body.y;
                ext_forces[ef_b + 5u] += f_body.z;

                f_w_total = f_w_total + rot_mul(w_rot[i], f_body);
                any_touch = true;
                if (penetration > deepest) { deepest = penetration; deepest_w = sup_w; }
                continue;
            }

            // Penalty normal force with per-body gains, Kelvin-Voigt:
            // f = k*pen - d*v_n. v_normal is the contact point's velocity
            // along the OUTWARD normal, so it is negative while the body is
            // still moving into the ground — hence the minus sign, and hence
            // a damper that always opposes the approach. max(_, 0) stops it
            // pulling the body back down as it separates, which is the real
            // hazard in any penalty contact.
            let m_n = contact_eff_mass(i, support, n_body);
            let k_body = min(pt_scale * geometry[gbase + 8u], max_stiffness(m_n));
            var d_body = pt_scale * geometry[gbase + 9u];
            // The damper may remove the approach velocity, never reverse it.
            d_body = min(d_body, max_damping(m_n));
            let f_n = max(k_body * penetration - d_body * v_normal, 0.0);

            // Coulomb friction opposing the tangential velocity.
            let v_tan = v_point - n_body * v_normal;
            let vt = length(v_tan);
            var f_body = n_body * f_n;
            if (vt > 1e-6) {
                let t_dir = v_tan / vt;
                // min(Coulomb limit, impulse that just stops the slip): the
                // physical cap and the non-reversal cap, which is the same
                // projection an impulse solver applies.
                let f_t = min(
                    coulomb(cparams.friction * f_n, vt),
                    max_damping(contact_eff_mass(i, support, t_dir)) * vt,
                );
                f_body = f_body - t_dir * f_t;
            }

            // Spatial force in the body frame: [angular = r x f, linear = f].
            // The torque is not optional decoration — without it contact acts
            // through the body origin, so no shape can resist tipping and a
            // resting box has no support polygon at all.
            let torque = cross3(support, f_body);
            let ef_base = ef_env_base + i * 6u;
            ext_forces[ef_base + 0u] += torque.x;
            ext_forces[ef_base + 1u] += torque.y;
            ext_forces[ef_base + 2u] += torque.z;
            ext_forces[ef_base + 3u] += f_body.x;
            ext_forces[ef_base + 4u] += f_body.y;
            ext_forces[ef_base + 5u] += f_body.z;

            f_w_total = f_w_total + rot_mul(w_rot[i], f_body);
            any_touch = true;
            if (penetration > deepest) {
                deepest = penetration;
                // On the surface, not at the shape's lowest vertex — that
                // vertex is below the terrain, and the depth is already
                // reported in its own slot.
                deepest_w = vec3<f32>(sup_w.x, sup_w.y, terr.w);
            }
        }

        if (!any_touch) {
            for (var k = 0u; k < CS_STRIDE; k++) { contact_state[cs_base + k] = 0.0; }
            continue;
        }

        // Contact state for readback (world frame): the deepest point, on
        // the ground surface, and the body's TOTAL contact force, so a box
        // resting on four corners reports the load it actually carries
        // rather than one corner's share.
        contact_state[cs_base]      = 1.0;
        contact_state[cs_base + 1u] = deepest;
        contact_state[cs_base + 2u] = deepest_w.x;
        contact_state[cs_base + 3u] = deepest_w.y;
        contact_state[cs_base + 4u] = deepest_w.z;
        contact_state[cs_base + 5u] = f_w_total.x;
        contact_state[cs_base + 6u] = f_w_total.y;
        contact_state[cs_base + 7u] = f_w_total.z;
    }

    // ── Body-attached contact plane (the deck's top face) ──
    //
    // Same penalty model as the ground, two differences: the plane moves
    // with its body, and the reaction lands on that body — a deck that felt
    // no rider could never be kicked out from under one.
    //
    // Deliberately NOT general body-body contact. The plane is the body's
    // local +Z face at `offset`, infinite in extent; a foot only ever meets
    // a deck's top face, and an infinite moving plane captures that at a
    // fraction of a broad phase's cost.
    //
    // Contacts here are not written to contact_state: those slots are
    // per-body and already hold the ground result, and a foot on a deck
    // would otherwise overwrite the ground reading with a different surface.
    if (cparams.plane_body >= 0) {
        let pb = u32(cparams.plane_body);
        // Plane frame in world. w_rot is body→world, so the body's local +Z
        // in world coordinates is rot_mul.
        let n_w = rot_mul(w_rot[pb], vec3<f32>(0.0, 0.0, 1.0));
        let p0_w = w_pos[pb] + rot_mul(w_rot[pb], vec3<f32>(0.0, 0.0, cparams.plane_offset));

        for (var i = 0u; i < nb; i++) {
            if (i == pb) { continue; }
            let gbase = i * GEOM_STRIDE;
            let gtype = u32(geometry[gbase]);
            if (gtype == 0u) { continue; }
            if (geometry[gbase + 7u] != 0.0) { continue; }

            let n_body = rot_t_mul(w_rot[i], n_w);
            var n_pts = 1u;
            var pt_scale = 1.0;
            if (gtype == 2u) { n_pts = 8u; pt_scale = 0.25; }

            // ── How much of this shape is actually over the face ──
            //
            // The face is a rectangle, not an infinite plane, and a shape may
            // hang over its edge. Dropping the overhanging points outright is
            // the obvious reading and it is wrong: a deck narrower than a foot
            // still carries that foot, along its edge. Measured on the K1, the
            // foot is 22 cm across a 19.7 cm deck — feet point ACROSS the board,
            // so toes and heel overhang, as a real skater's do — and deleting
            // those corners deleted the whole roll support. Standing fell from
            // 3.00 s to 0.8 s in every stance.
            //
            // So: intersect the shape's footprint with the face, in the face's
            // own 2-D coordinates. No overlap means nothing is over the face and
            // the shape genuinely falls. Otherwise every contact point is
            // clamped into the overlap, which puts the force on the deck's edge
            // where the deck actually has material — support is kept, and the
            // phantom support out past the edge, which is what an infinite plane
            // invents, is gone.
            //
            // The footprint is taken as an AABB in face coordinates: exact while
            // the shape is aligned with the face, and a slight over-estimate
            // under yaw.
            var lo = vec2<f32>(1e30, 1e30);
            var hi = vec2<f32>(-1e30, -1e30);
            for (var cpt = 0u; cpt < n_pts; cpt++) {
                var sp: vec3<f32>;
                if (gtype == 2u) { sp = box_corner(i, cpt); }
                else { sp = support_point(i, n_body); }
                let rp = rot_t_mul(w_rot[pb], w_pos[i] + rot_mul(w_rot[i], sp) - w_pos[pb]);
                lo = min(lo, rp.xy);
                hi = max(hi, rp.xy);
            }
            let face_lo = vec2<f32>(-cparams.plane_half_x, -cparams.plane_half_y);
            let face_hi = vec2<f32>(cparams.plane_half_x, cparams.plane_half_y);
            let ov_lo = max(lo, face_lo);
            let ov_hi = min(hi, face_hi);
            if (ov_lo.x > ov_hi.x || ov_lo.y > ov_hi.y) { continue; }

            // Active-point count on the face, for the same within-body load
            // sharing the ground branch needs.
            var n_pts_active = 1u;
            if (cparams.solve_mode == 1u) {
                var cnt = 0u;
                for (var cq = 0u; cq < n_pts; cq++) {
                    var sq: vec3<f32>;
                    if (gtype == 2u) { sq = box_corner(i, cq); }
                    else { sq = support_point(i, n_body); }
                    let d = -dot(w_pos[i] + rot_mul(w_rot[i], sq) - p0_w, n_w);
                    if (d > 0.0 && d <= cparams.plane_max_depth) { cnt++; }
                }
                n_pts_active = max(cnt, 1u);
            }

            for (var cpt = 0u; cpt < n_pts; cpt++) {
                var support: vec3<f32>;
                if (gtype == 2u) { support = box_corner(i, cpt); }
                else { support = support_point(i, n_body); }
                let sup_w0 = w_pos[i] + rot_mul(w_rot[i], support);
                // Penetration is measured at the shape's own point: that is
                // where its material is, and the face is planar, so sliding the
                // application point along the face does not change the depth.
                let penetration = -dot(sup_w0 - p0_w, n_w);
                // The upper bound guards a body approaching from BELOW an
                // infinite plane against being captured and catapulted through.
                if (penetration <= 0.0 || penetration > cparams.plane_max_depth) {
                    if (cparams.solve_mode == 1u) {
                        let dead = (world_idx * nb + i) * CS_STRIDE + CS_PLANE_OFF + cpt * 3u;
                        contact_state[dead] = 0.0;
                        contact_state[dead + 1u] = 0.0;
                        contact_state[dead + 2u] = 0.0;
                    }
                    continue;
                }

                // The contact point in the plane body's own frame, pulled into
                // the overlap: the force lands where the face has material.
                var r_p = rot_t_mul(w_rot[pb], sup_w0 - w_pos[pb]);
                let in_face = clamp(r_p.xy, ov_lo, ov_hi);
                r_p = vec3<f32>(in_face.x, in_face.y, r_p.z);
                let sup_w = w_pos[pb] + rot_mul(w_rot[pb], r_p);
                // The same world point, as a lever arm on the touching body, so
                // action and reaction act at one point rather than two.
                let support_c = rot_t_mul(w_rot[i], sup_w - w_pos[i]);

                // Relative velocity of the two material points at the contact,
                // world frame. Body-frame spatial velocities rotate out with
                // rot_mul, matching the FK convention above.
                let v_i_w = rot_mul(w_rot[i], w_lin[i] + cross3(w_omega[i], support_c));
                let v_p_w = rot_mul(w_rot[pb], w_lin[pb] + cross3(w_omega[pb], r_p));
                let v_rel = v_i_w - v_p_w;
                let v_normal = dot(v_rel, n_w);

                // Both bodies move, so the pair's effective mass is the series
                // combination of what each presents at the contact point.
                let n_i = rot_t_mul(w_rot[i], n_w);
                let n_p = rot_t_mul(w_rot[pb], n_w);
                let m_n = 1.0 / (1.0 / contact_eff_mass(i, support_c, n_i)
                    + 1.0 / contact_eff_mass(pb, r_p, n_p));

                // ── Impulse mode on the body-attached face ──
                //
                // Same staged Coulomb update as the ground branch; the only
                // difference is that both bodies move, so every diagonal is
                // the SERIES combination `m_n` already computed above, and the
                // impulse is applied equal-and-opposite at one shared point.
                if (cparams.solve_mode == 1u) {
                    let pslot = (world_idx * nb + i) * CS_STRIDE + CS_PLANE_OFF + cpt * 3u;
                    var pf = vec3<f32>(contact_state[pslot], contact_state[pslot + 1u], contact_state[pslot + 2u]);

                    let ptang = contact_tangents(n_w);
                    let pu = ptang[0];
                    let pw = ptang[1];
                    let bn = v_normal;
                    let bu = dot(v_rel, pu);
                    let bw = dot(v_rel, pw);

                    let ann = f32(n_pts_active) / max(m_n, 1e-9);
                    let d_imp = impedance_at(penetration);
                    let bias = d_imp * cparams.solref_erp * max(penetration, 0.0) / max(cparams.dt, 1e-9);
                    let ee = effective_restitution(cparams.restitution, min(bn, 0.0));
                    var nf = max((bias - (bn * (1.0 + ee) - ann * pf.x)) / ann, 0.0);

                    let u_i = rot_t_mul(w_rot[i], pu);
                    let u_p = rot_t_mul(w_rot[pb], pu);
                    let w_i = rot_t_mul(w_rot[i], pw);
                    let w_p = rot_t_mul(w_rot[pb], pw);
                    let m_u = 1.0 / (1.0 / contact_eff_mass(i, support_c, u_i)
                        + 1.0 / contact_eff_mass(pb, r_p, u_p));
                    let m_w = 1.0 / (1.0 / contact_eff_mass(i, support_c, w_i)
                        + 1.0 / contact_eff_mass(pb, r_p, w_p));
                    let auu = f32(n_pts_active) / max(m_u, 1e-9);
                    let aww = f32(n_pts_active) / max(m_w, 1e-9);
                    var ptu = -(bu - auu * pf.y) / auu;
                    var ptw = -(bw - aww * pf.z) / aww;
                    let plim = cparams.friction * nf;
                    let ptn = sqrt(ptu * ptu + ptw * ptw);
                    if (ptn > plim) {
                        let psc = select(0.0, plim / ptn, ptn > 0.0);
                        ptu = ptu * psc;
                        ptw = ptw * psc;
                    }

                    contact_state[pslot] = nf;
                    contact_state[pslot + 1u] = ptu;
                    contact_state[pslot + 2u] = ptw;

                    let fw2 = (n_w * nf + pu * ptu + pw * ptw) / max(cparams.dt, 1e-9);
                    let fi2 = rot_t_mul(w_rot[i], fw2);
                    let ti2 = cross3(support_c, fi2);
                    let e_i = ef_env_base + i * 6u;
                    ext_forces[e_i + 0u] += ti2.x;
                    ext_forces[e_i + 1u] += ti2.y;
                    ext_forces[e_i + 2u] += ti2.z;
                    ext_forces[e_i + 3u] += fi2.x;
                    ext_forces[e_i + 4u] += fi2.y;
                    ext_forces[e_i + 5u] += fi2.z;

                    let fp2 = rot_t_mul(w_rot[pb], -fw2);
                    let tp2 = cross3(r_p, fp2);
                    let e_p = ef_env_base + pb * 6u;
                    ext_forces[e_p + 0u] += tp2.x;
                    ext_forces[e_p + 1u] += tp2.y;
                    ext_forces[e_p + 2u] += tp2.z;
                    ext_forces[e_p + 3u] += fp2.x;
                    ext_forces[e_p + 4u] += fp2.y;
                    ext_forces[e_p + 5u] += fp2.z;
                    continue;
                }

                let k_body = min(pt_scale * geometry[gbase + 8u], max_stiffness(m_n));
                var d_body = pt_scale * geometry[gbase + 9u];
                d_body = min(d_body, max_damping(m_n));
                let f_n = max(k_body * penetration - d_body * v_normal, 0.0);

                let v_tan = v_rel - n_w * v_normal;
                let vt = length(v_tan);
                var f_w = n_w * f_n;
                if (vt > 1e-6) {
                    let t_dir = v_tan / vt;
                    let t_i = rot_t_mul(w_rot[i], t_dir);
                    let t_p = rot_t_mul(w_rot[pb], t_dir);
                    let m_t = 1.0 / (1.0 / contact_eff_mass(i, support_c, t_i)
                        + 1.0 / contact_eff_mass(pb, r_p, t_p));
                    let f_t = min(
                        coulomb(cparams.friction * f_n, vt),
                        max_damping(m_t) * vt,
                    );
                    f_w = f_w - t_dir * f_t;
                }

                // Action on the touching body, in its own frame.
                let f_i = rot_t_mul(w_rot[i], f_w);
                let torque_i = cross3(support_c, f_i);
                let ef_i = ef_env_base + i * 6u;
                ext_forces[ef_i + 0u] += torque_i.x;
                ext_forces[ef_i + 1u] += torque_i.y;
                ext_forces[ef_i + 2u] += torque_i.z;
                ext_forces[ef_i + 3u] += f_i.x;
                ext_forces[ef_i + 4u] += f_i.y;
                ext_forces[ef_i + 5u] += f_i.z;

                // Equal and opposite on the plane's body, at the same point.
                let f_p = rot_t_mul(w_rot[pb], -f_w);
                let torque_p = cross3(r_p, f_p);
                let ef_p = ef_env_base + pb * 6u;
                ext_forces[ef_p + 0u] += torque_p.x;
                ext_forces[ef_p + 1u] += torque_p.y;
                ext_forces[ef_p + 2u] += torque_p.z;
                ext_forces[ef_p + 3u] += f_p.x;
                ext_forces[ef_p + 4u] += f_p.y;
                ext_forces[ef_p + 5u] += f_p.z;
            }
        }
    }
}
"#;

/// Flat semi-implicit Euler for single-DOF-only models.
///
/// Correct only when every joint is revolute or prismatic, where `q` and `v`
/// share a parameterisation. Used by [`crate::GpuSimulator`], which is
/// pendulum-only by construction. General models must use
/// [`INTEGRATE_SHADER`], which is joint-aware.
pub const INTEGRATE_SIMPLE_SHADER: &str = r#"
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
    if (idx >= params.nworld * params.nv) { return; }

    let v_new = v[idx] + params.dt * qdd[idx];
    v[idx] = v_new;
    q[idx] = q[idx] + params.dt * v_new;
}
"#;

/// Joint-aware semi-implicit Euler.
///
/// Must match `phyz_rigid::semi_implicit_euler` exactly. A flat `q += dt * v`
/// is wrong for ball and free joints because `q` and `v` use different
/// parameterisations — a free joint's `q` is `[exp-coords(3), pos(3)]`, which
/// matches `v`'s `[angular(3), linear(3)]` slot for slot, but the rotational
/// slots are exponential coordinates (needing a Lie-group step) and the linear
/// velocity is body-frame (needing a rotation into the parent frame).
///
/// One thread per (environment, joint) pair, so joints in the same environment
/// touch disjoint `q`/`v` ranges and no synchronisation is needed.
pub const INTEGRATE_SHADER: &str = r#"
const BODY_STRIDE: u32 = 36u;

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
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> qdd: array<f32>;
@group(0) @binding(4) var<storage, read> bodies: array<f32>;

fn bf_i(bidx: u32, off: u32) -> f32 { return bodies[bidx * BODY_STRIDE + off]; }
fn bu_i(bidx: u32, off: u32) -> u32 { return bitcast<u32>(bodies[bidx * BODY_STRIDE + off]); }

fn ndof_of(jtype: u32) -> u32 {
    if (jtype == 2u) { return 0u; }
    if (jtype == 3u) { return 3u; }
    if (jtype == 4u) { return 6u; }
    return 1u;
}

fn qexp(omega: vec3<f32>) -> vec4<f32> {
    let angle = length(omega);
    if (angle < 1e-6) {
        return vec4<f32>(1.0, omega.x * 0.5, omega.y * 0.5, omega.z * 0.5);
    }
    let half = angle * 0.5;
    let s = sin(half) / angle;
    return vec4<f32>(cos(half), omega.x * s, omega.y * s, omega.z * s);
}

// Hamilton product, both as (w, x, y, z).
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

// Matches tang's Quat::log, including the small-angle branch.
fn qlog(qt: vec4<f32>) -> vec3<f32> {
    let vv = vec3<f32>(qt.y, qt.z, qt.w);
    let n = length(vv);
    if (n < 1e-6) { return vv * 2.0; }
    let angle = 2.0 * atan2(n, qt.x);
    return vv * (angle / n);
}

fn qrotate(qt: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let u = vec3<f32>(qt.y, qt.z, qt.w);
    let t = cross(u, p) * 2.0;
    return p + t * qt.x + cross(u, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nb = params.nbodies;
    if (idx >= params.nworld * nb) { return; }

    let world_idx = idx / nb;
    let body_idx = idx % nb;

    let jtype = bu_i(body_idx, 1u);
    let ndof = ndof_of(jtype);
    if (ndof == 0u) { return; }

    let q_off = world_idx * params.nv + bu_i(body_idx, 2u);
    let v_off = world_idx * params.nv + bu_i(body_idx, 3u);
    let dt = params.dt;

    // Velocity first (semi-implicit).
    for (var k = 0u; k < ndof; k++) {
        v[v_off + k] = v[v_off + k] + dt * qdd[v_off + k];
    }

    if (jtype == 0u || jtype == 1u) {
        q[q_off] = q[q_off] + dt * v[v_off];
        return;
    }

    if (jtype == 3u) {
        // Ball: compose the rotation, then re-log to exponential coordinates.
        let omega = vec3<f32>(v[v_off], v[v_off + 1u], v[v_off + 2u]);
        let cur = qexp(vec3<f32>(q[q_off], q[q_off + 1u], q[q_off + 2u]));
        let nxt = normalize(qmul(cur, qexp(omega * dt)));
        let lg = qlog(nxt);
        q[q_off] = lg.x; q[q_off + 1u] = lg.y; q[q_off + 2u] = lg.z;
        return;
    }

    // Free: v = [angular(3), linear(3)], q = [exp-coords(3), pos(3)].
    let omega = vec3<f32>(v[v_off], v[v_off + 1u], v[v_off + 2u]);
    let lin = vec3<f32>(v[v_off + 3u], v[v_off + 4u], v[v_off + 5u]);
    let cur = qexp(vec3<f32>(q[q_off], q[q_off + 1u], q[q_off + 2u]));

    let world_lin = qrotate(cur, lin);
    q[q_off + 3u] = q[q_off + 3u] + dt * world_lin.x;
    q[q_off + 4u] = q[q_off + 4u] + dt * world_lin.y;
    q[q_off + 5u] = q[q_off + 5u] + dt * world_lin.z;

    let nxt = normalize(qmul(cur, qexp(omega * dt)));
    let lg = qlog(nxt);
    q[q_off] = lg.x; q[q_off + 1u] = lg.y; q[q_off + 2u] = lg.z;
}
"#;

/// WGSL shader for generalized ABA (arbitrary articulated body trees).
///
/// Supports revolute (0), prismatic (1), fixed (2), ball (3, 3 DOF) and free
/// (4, 6 DOF) joints. One thread per environment, serial tree traversal within.
/// Bodies must be topologically sorted (parent index < child index).
///
/// Body data layout: 36 f32 values per body (BODY_STRIDE):
///   `[0]`  parent (bitcast i32, -1 for root)
///   `[1]`  joint_type (0=revolute, 1=prismatic, 2=fixed, 3=ball, 4=free)
///   `[2]`  q_offset
///   `[3]`  v_offset
///   `[4]`  mass
///   [5..8]  com (x,y,z)
///   [8..14] inertia (xx,yy,zz,xy,xz,yz)
///   [14..23] ptj rotation (row-major 3x3)
///   [23..26] ptj translation (x,y,z)
///   [26..29] axis (x,y,z)
///   `[29]` damping
///   `[30]` passive spring stiffness, `[31]` spring reference angle
///   `[32]` armature (rotor inertia)
///   [33..36] padding
pub const ABA_GENERAL_SHADER: &str = r#"
const MAX_BODIES: u32 = 32u;
const BODY_STRIDE: u32 = 36u;

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

// ── Multi-DOF joint support ──
//
// Joint types: 0=revolute, 1=prismatic, 2=fixed, 3=ball (3 DOF), 4=free (6 DOF).
// Every benchmark model (ant, half-cheetah, humanoid, hand-on-a-floating-base)
// has a floating root, so welding multi-DOF joints — as this kernel used to —
// silently produced a different robot. See docs/design/batched-envs.md, B4.

fn joint_ndof(jtype: u32) -> u32 {
    if (jtype == 2u) { return 0u; }
    if (jtype == 3u) { return 3u; }
    if (jtype == 4u) { return 6u; }
    return 1u;
}

// Column `k` of the motion subspace matrix S (6 x ndof).
//
// Mirrors phyz_model::Joint::motion_subspace_matrix and
// phyz_rigid::kinematics::joint_velocity exactly, including the free joint's
// angular-then-linear velocity ordering.
fn subspace_col(jtype: u32, axis: vec3<f32>, k: u32) -> array<f32, 6> {
    if (jtype == 0u) {
        return array<f32, 6>(axis.x, axis.y, axis.z, 0.0, 0.0, 0.0);
    }
    if (jtype == 1u) {
        return array<f32, 6>(0.0, 0.0, 0.0, axis.x, axis.y, axis.z);
    }
    if (jtype == 3u) {
        // Ball: angular DOF only.
        var s = sv_zero();
        s[k] = 1.0;
        return s;
    }
    if (jtype == 4u) {
        // Free: v = [angular(3), linear(3)], so S is the 6x6 identity.
        var s = sv_zero();
        s[k] = 1.0;
        return s;
    }
    return sv_zero();
}

// Quaternion exponential of a rotation vector, as (w, x, y, z).
// Matches tang's Quat::exp including the small-angle branch.
fn quat_exp(omega: vec3<f32>) -> vec4<f32> {
    let angle = length(omega);
    if (angle < 1e-6) {
        return vec4<f32>(1.0, omega.x * 0.5, omega.y * 0.5, omega.z * 0.5);
    }
    let half = angle * 0.5;
    let s = sin(half) / angle;
    return vec4<f32>(cos(half), omega.x * s, omega.y * s, omega.z * s);
}

// Quaternion (w, x, y, z) to row-major rotation matrix.
fn quat_to_rot(qt: vec4<f32>) -> array<f32, 9> {
    let w = qt.x; let x = qt.y; let y = qt.z; let z = qt.w;
    var r: array<f32, 9>;
    r[0] = 1.0 - 2.0*(y*y + z*z); r[1] = 2.0*(x*y - w*z);       r[2] = 2.0*(x*z + w*y);
    r[3] = 2.0*(x*y + w*z);       r[4] = 1.0 - 2.0*(x*x + z*z); r[5] = 2.0*(y*z - w*x);
    r[6] = 2.0*(x*z - w*y);       r[7] = 2.0*(y*z + w*x);       r[8] = 1.0 - 2.0*(x*x + y*y);
    return r;
}

// ── Small dense solve for the ndof x ndof articulated inertia block ──
//
// ndof <= 6, so Gauss-Jordan with partial pivoting is both simple and fast
// enough; the cost is dwarfed by the 6x6 spatial products around it.

const MAX_JDOF: u32 = 6u;

fn mat_get(m: ptr<function, array<f32, 36>>, n: u32, r: u32, c: u32) -> f32 {
    return (*m)[r * n + c];
}

// In-place inverse of the leading n x n block (row-major, stride n).
// Returns false if the block is singular, in which case the caller falls back
// to zero acceleration rather than emitting NaNs into the state buffer.
fn invert_small(m: ptr<function, array<f32, 36>>, n: u32) -> bool {
    var inv: array<f32, 36>;
    for (var r = 0u; r < n; r++) {
        for (var c = 0u; c < n; c++) {
            inv[r * n + c] = select(0.0, 1.0, r == c);
        }
    }

    for (var col = 0u; col < n; col++) {
        // Partial pivot.
        var piv = col;
        var best = abs((*m)[col * n + col]);
        for (var r = col + 1u; r < n; r++) {
            let a = abs((*m)[r * n + col]);
            if (a > best) { best = a; piv = r; }
        }
        if (best < 1e-20) { return false; }

        if (piv != col) {
            for (var c = 0u; c < n; c++) {
                let t1 = (*m)[col * n + c];
                (*m)[col * n + c] = (*m)[piv * n + c];
                (*m)[piv * n + c] = t1;
                let t2 = inv[col * n + c];
                inv[col * n + c] = inv[piv * n + c];
                inv[piv * n + c] = t2;
            }
        }

        let d = (*m)[col * n + col];
        for (var c = 0u; c < n; c++) {
            (*m)[col * n + c] /= d;
            inv[col * n + c] /= d;
        }

        for (var r = 0u; r < n; r++) {
            if (r == col) { continue; }
            let f = (*m)[r * n + col];
            if (f == 0.0) { continue; }
            for (var c = 0u; c < n; c++) {
                (*m)[r * n + c] -= f * (*m)[col * n + c];
                inv[r * n + c] -= f * inv[col * n + c];
            }
        }
    }

    for (var i = 0u; i < n * n; i++) { (*m)[i] = inv[i]; }
    return true;
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
        } else if (jtype == 3u) {
            // Ball: q = exponential coordinates (3). Coordinate map is the
            // INVERSE rotation (exp(-w)), matching revolute_rot's negated
            // angle and the CPU joint_transform_slice.
            let w = vec3<f32>(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
            j_rot = quat_to_rot(quat_exp(-w));
            j_pos = vec3<f32>(0.0, 0.0, 0.0);
        } else if (jtype == 4u) {
            // Free: q = [exponential coordinates(3), pos(3)] — angular first,
            // matching v's [angular; linear].
            let w = vec3<f32>(q[q_base + q_off], q[q_base + q_off + 1u], q[q_base + q_off + 2u]);
            j_rot = quat_to_rot(quat_exp(-w));
            j_pos = vec3<f32>(q[q_base + q_off + 3u], q[q_base + q_off + 4u], q[q_base + q_off + 5u]);
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

        // Joint velocity: S * qd, summed over the joint's DOFs.
        let ndof = joint_ndof(jtype);
        var v_joint = sv_zero();
        for (var k = 0u; k < ndof; k++) {
            v_joint = sv_add(v_joint, sv_scale(subspace_col(jtype, axis, k), v[v_base + v_off + k]));
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
        let stiffness_val = bf(i, 30u);
        let spring_ref = bf(i, 31u);

        if (jtype == 2u) {
            // Fixed joint: just propagate to parent
            if (parent >= 0) {
                let pi = u32(parent);
                var x_mot = build_motion_transform(x_rot[i], x_pos[i]);
                var x_mot_t = transpose6(&x_mot);
                // Local copies, not pointers into the array: naga's SPIR-V
                // backend never caches `&arr[dynamic_index]` passed to a
                // function (gfx-rs/wgpu#7315) and panics at write time. The
                // Metal backend accepted it, which is why this only surfaced
                // on the first Vulkan machine.
                var ia_self = i_a[i];
                var ia_parent = m6_XtAX(&x_mot_t, &ia_self, &x_mot);
                var ia_pi = i_a[pi];
                i_a[pi] = m6_add(&ia_pi, &ia_parent);
                let p_parent = inv_apply_force(x_rot[i], x_pos[i], p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_parent);
            }
            continue;
        }

        // Joint with 1..6 DOF. U = I_A S (6 x n), D = Sᵀ U (n x n),
        // u = τ − Sᵀ p_A (n).
        let ndof = joint_ndof(jtype);
        var u_mat: array<f32, 36>;   // U, column k at [k*6 .. k*6+6]
        var d_mat: array<f32, 36>;   // D, row-major with stride ndof
        var u_vec: array<f32, 6>;

        var ia_ref = i_a[i];
        for (var k = 0u; k < ndof; k++) {
            let s_k = subspace_col(jtype, axis, k);
            let uk = m6_mul_vec(&ia_ref, s_k);
            for (var r = 0u; r < 6u; r++) { u_mat[k * 6u + r] = uk[r]; }
            u_vec[k] = ctrl[v_base + v_off + k]
                - damping_val * v[v_base + v_off + k]
                - sv_dot(s_k, p_a[i]);
        }
        // Passive joint spring, single-DOF joints only — the exact clause
        // CPU passive_force applies (joint.rs): f += -k * (q - q_ref).
        // Explicit like the CPU's, so no D-matrix term.
        if (ndof == 1u && stiffness_val != 0.0) {
            let q_off_s = bu(i, 2u);
            u_vec[0] += -stiffness_val * (q[q_base + q_off_s] - spring_ref);
        }
        for (var r = 0u; r < ndof; r++) {
            let s_r = subspace_col(jtype, axis, r);
            for (var c = 0u; c < ndof; c++) {
                var acc_d = 0.0;
                for (var t = 0u; t < 6u; t++) { acc_d += s_r[t] * u_mat[c * 6u + t]; }
                d_mat[r * ndof + c] = acc_d;
            }
            // Implicit joint damping — must match phyz_rigid::aba exactly, or
            // the two backends diverge on any damped model.
            // Armature (rotor inertia) joins it on the diagonal: on the K1
            // it exceeds the ankle's link inertia ~100x, and without it the
            // PD gains scaled by the CPU's armature-bearing mass matrix
            // blow the model over in 0.2 s — measured on the skate rig.
            d_mat[r * ndof + r] += params.dt * damping_val + bf(i, 32u);
        }

        // A singular articulated inertia means the joint carries no effective
        // mass; treat it as fixed rather than dividing by ~0.
        if (!invert_small(&d_mat, ndof)) {
            if (parent >= 0) {
                let pi = u32(parent);
                var x_mot_s = build_motion_transform(x_rot[i], x_pos[i]);
                var x_mot_st = transpose6(&x_mot_s);
                var ia_self_s = i_a[i];
                var ia_par_s = m6_XtAX(&x_mot_st, &ia_self_s, &x_mot_s);
                var ia_pi_s = i_a[pi];
                i_a[pi] = m6_add(&ia_pi_s, &ia_par_s);
                let p_par_s = inv_apply_force(x_rot[i], x_pos[i], p_a[i]);
                p_a[pi] = sv_add(p_a[pi], p_par_s);
            }
            continue;
        }

        if (parent >= 0) {
            let pi = u32(parent);

            // W = U D⁻¹  (6 x n)
            var w_mat: array<f32, 36>;
            for (var c = 0u; c < ndof; c++) {
                for (var r = 0u; r < 6u; r++) {
                    var s = 0.0;
                    for (var j = 0u; j < ndof; j++) {
                        s += u_mat[j * 6u + r] * d_mat[j * ndof + c];
                    }
                    w_mat[c * 6u + r] = s;
                }
            }

            // I_a^A = I_A − W Uᵀ
            var ia_new = i_a[i];
            for (var c = 0u; c < ndof; c++) {
                var wc: array<f32, 6>;
                var uc: array<f32, 6>;
                for (var r = 0u; r < 6u; r++) {
                    wc[r] = w_mat[c * 6u + r];
                    uc[r] = u_mat[c * 6u + r];
                }
                var outer = m6_outer(wc, uc);
                ia_new = m6_sub(&ia_new, &outer);
            }

            // p_a^A = p_A + I_a^A c + W u
            let ia_c = m6_mul_vec(&ia_new, c_bias[i]);
            var wu = sv_zero();
            for (var c = 0u; c < ndof; c++) {
                for (var r = 0u; r < 6u; r++) {
                    wu[r] += w_mat[c * 6u + r] * u_vec[c];
                }
            }
            let p_new = sv_add(sv_add(p_a[i], ia_c), wu);

            var x_mot = build_motion_transform(x_rot[i], x_pos[i]);
            var x_mot_t = transpose6(&x_mot);
            var ia_parent = m6_XtAX(&x_mot_t, &ia_new, &x_mot);
            var ia_pi_w = i_a[pi];
            i_a[pi] = m6_add(&ia_pi_w, &ia_parent);

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
        let stiffness_val = bf(i, 30u);
        let spring_ref = bf(i, 31u);

        var a_parent: array<f32, 6>;
        if (parent < 0) {
            a_parent = apply_motion(x_rot[i], x_pos[i], a0);
        } else {
            let pi = u32(parent);
            a_parent = apply_motion(x_rot[i], x_pos[i], acc[pi]);
        }

        let a_c = sv_add(a_parent, c_bias[i]);
        let ndof = joint_ndof(jtype);
        if (ndof == 0u) {
            acc[i] = a_c;
            continue;
        }

        // Rebuild U, D and u rather than carrying them from pass 2: at
        // MAX_BODIES bodies the extra 72 floats per body would spill private
        // storage to device memory and cost more than the recompute.
        var u_mat: array<f32, 36>;
        var d_mat: array<f32, 36>;
        var u_vec: array<f32, 6>;

        var ia_ref = i_a[i];
        for (var k = 0u; k < ndof; k++) {
            let s_k = subspace_col(jtype, axis, k);
            let uk = m6_mul_vec(&ia_ref, s_k);
            for (var r = 0u; r < 6u; r++) { u_mat[k * 6u + r] = uk[r]; }
            u_vec[k] = ctrl[v_base + v_off + k]
                - damping_val * v[v_base + v_off + k]
                - sv_dot(s_k, p_a[i]);
        }
        // Passive joint spring, single-DOF joints only — the exact clause
        // CPU passive_force applies (joint.rs): f += -k * (q - q_ref).
        // Explicit like the CPU's, so no D-matrix term.
        if (ndof == 1u && stiffness_val != 0.0) {
            let q_off_s = bu(i, 2u);
            u_vec[0] += -stiffness_val * (q[q_base + q_off_s] - spring_ref);
        }
        for (var r = 0u; r < ndof; r++) {
            let s_r = subspace_col(jtype, axis, r);
            for (var c = 0u; c < ndof; c++) {
                var acc_d = 0.0;
                for (var t = 0u; t < 6u; t++) { acc_d += s_r[t] * u_mat[c * 6u + t]; }
                d_mat[r * ndof + c] = acc_d;
            }
            // Implicit joint damping — must match phyz_rigid::aba exactly, or
            // the two backends diverge on any damped model.
            // Armature (rotor inertia) joins it on the diagonal: on the K1
            // it exceeds the ankle's link inertia ~100x, and without it the
            // PD gains scaled by the CPU's armature-bearing mass matrix
            // blow the model over in 0.2 s — measured on the skate rig.
            d_mat[r * ndof + r] += params.dt * damping_val + bf(i, 32u);
        }

        if (!invert_small(&d_mat, ndof)) {
            acc[i] = a_c;
            for (var k = 0u; k < ndof; k++) { qdd[v_base + v_off + k] = 0.0; }
            continue;
        }

        // qdd = D⁻¹ (u − Uᵀ a_c)
        var rhs: array<f32, 6>;
        for (var k = 0u; k < ndof; k++) {
            var uta = 0.0;
            for (var t = 0u; t < 6u; t++) { uta += u_mat[k * 6u + t] * a_c[t]; }
            rhs[k] = u_vec[k] - uta;
        }

        var a_new = a_c;
        for (var k = 0u; k < ndof; k++) {
            var qdd_k = 0.0;
            for (var j = 0u; j < ndof; j++) { qdd_k += d_mat[k * ndof + j] * rhs[j]; }
            qdd[v_base + v_off + k] = qdd_k;
            a_new = sv_add(a_new, sv_scale(subspace_col(jtype, axis, k), qdd_k));
        }
        acc[i] = a_new;
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
