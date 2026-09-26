"""MuJoCo mirror of crates/phyz/examples/contact_speed.rs's K1 scenes.

Vendor K1_22dof.xml (its own 20 collision geoms: the URDF primitives, the
foot/shank/trunk boxes, and 6 links as convex mesh hulls), floor overridden to phyz's
material (mu 0.5, condim 3, MuJoCo-default solref/solimp = phyz's defaults),
elliptic cone, Newton, tol 1e-10, Euler, dt 1 ms: the audit's mj_k1.py setup.
PD is native (gainprm/biasprm position servos clamped by forcerange), same
gains/limits/pose/targets as the phyz example. Targets are refreshed every
CHUNK steps from Python and the C loop runs the chunk, so the timing is
mj_step's own (thread CPU time: the bench machine is shared); the scripted targets are therefore piecewise constant at
CHUNK ms (phyz evaluates them every step -- a trajectory difference, not a
cost one).

usage: <ipse>/.venv-mj/bin/python mj_scenes.py   (prints one JSON row per scene)
"""
import json, math, os, re, time
import mujoco, numpy as np

XML = os.environ.get("K1_DIR", "/Users/cam/Developer/booster_assets/robots/K1") + "/K1_22dof.xml"
CHUNK = int(os.environ.get("CS_CHUNK", "10"))
REPS = int(os.environ.get("CS_REPS", "5"))
JOINTS = ["AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
          "Left_Elbow_Yaw", "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
          "Right_Elbow_Yaw", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
          "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
          "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]
KP = [1.6076564945148037, 4.96794603104, 14.219637448679467, 12.858821292846736, 0.5937363343523681,
      1.7562086058847597, 14.216133999885427, 12.854414564672165, 0.594081067660144, 1.75529654190604,
      604.180261289572, 500.3593451001352, 54.445200651623594, 278.2593675021229, 92.548975055296,
      90.9807259388928, 604.1759041201899, 500.3530313430554, 54.4445516726996, 278.25672367342963,
      92.548975055296, 90.9807259388928]
KD = [k * (0.05 if j >= 10 else 0.1) for j, k in enumerate(KP)]
EFFORT = [6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 30, 35, 20, 40, 20, 20, 30, 35, 20, 40, 20, 20]
Q0 = [0, 0, 0, -1.4, 0.25, 0, 0, 1.4, 0.25, 0, -0.05, 0, 0, 0.1, -0.05, 0, -0.05, 0, 0, 0.1, -0.05, 0]
TRUNK_Z = 0.5532540584455157


def target(scene, t, k):
    sm = lambda x: (lambda x: x * x * (3 - 2 * x))(min(max(x, 0.0), 1.0))

    def lift(leg, s):
        return {0: -0.3 * s, 3: 0.6 * s, 4: -0.3 * s}.get(k - (10 + 6 * leg), 0.0)
    if scene == "single":
        return Q0[k] + lift(0, sm((t - 0.3) / 0.2))
    if scene == "step":
        if t < 0.3:
            return Q0[k]
        ph = ((t - 0.3) / 0.8) % 1.0
        leg, x = (0, ph / 0.5) if ph < 0.5 else (1, (ph - 0.5) / 0.5)
        return Q0[k] + lift(leg, math.sin(math.pi * x))
    return Q0[k]


def load():
    src = open(XML).read()
    src = src.replace('<option timestep="0.001" tolerance="1e-6" impratio="10" solver="Newton"/>',
                      '<option timestep="0.001" tolerance="1e-10" iterations="100" solver="Newton" cone="elliptic" integrator="Euler"/>')
    src = src.replace('<geom solimp="0.90 0.95 0.001" solref="0.001 1"/>', '<geom solimp="0.9 0.95 0.001" solref="0.02 1"/>')
    src = re.sub(r'<geom name="ground"[^>]*/>', '<geom name="ground" type="plane" pos="0 0 0" size="0 0 1" condim="3" friction="0.5 0.005 0.0001"/>', src)
    cwd = os.getcwd(); os.chdir(os.path.dirname(XML))
    m = mujoco.MjModel.from_xml_string(src)
    os.chdir(cwd)
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_friction[g] = [0.5, 0.005, 0.0001]; m.geom_condim[g] = 3
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS]
    for j, a in enumerate(act):
        m.actuator_gaintype[a] = 0; m.actuator_biastype[a] = 1
        m.actuator_gainprm[a, 0] = KP[j]; m.actuator_biasprm[a, :3] = [0, -KP[j], -KD[j]]
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1; m.actuator_forcerange[a] = [-EFFORT[j], EFFORT[j]]
    return m, act


def main():
    m, act = load()
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
    ncoll = int(sum(1 for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]))
    for scene, secs in (("stance", 2.0), ("single", 1.0), ("step", 3.0)):
        n = int(round(secs / m.opt.timestep))
        runs = []
        for _ in range(REPS):
            d = mujoco.MjData(m)
            d.qpos[2] = TRUNK_Z; d.qpos[3] = 1.0
            for j, a in enumerate(qadr):
                d.qpos[a] = Q0[j]
            mujoco.mj_forward(m, d)
            ncon = iters = 0
            w = 0.0
            for s0 in range(0, n, CHUNK):
                t = d.time
                for j, a in enumerate(act):
                    d.ctrl[a] = target(scene, t, j)
                k = min(CHUNK, n - s0)
                t0 = time.thread_time_ns(); mujoco.mj_step(m, d, k); w += (time.thread_time_ns() - t0) * 1e-9
                ncon += d.ncon; iters += int(d.solver_niter[0])
            runs.append(w / n * 1e6)
        runs.sort()
        chunks = math.ceil(n / CHUNK)
        print(json.dumps(dict(scene="mj_" + scene, engine="mujoco " + mujoco.__version__, nv=int(m.nv), collision_geoms=ncoll,
                              steps=n, us_per_step=runs[len(runs) // 2], us_min=runs[0], us_runs=[round(r, 2) for r in runs],
                              ncon_mean=ncon / chunks, iters_mean=iters / chunks, trunk_z_end=float(d.qpos[2]))))


main()
