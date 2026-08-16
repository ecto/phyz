#!/usr/bin/env python3
"""Cross-engine agreement: does phyz's articulated dynamics match MuJoCo's?

Speed comparisons say how fast an engine is. They say nothing about whether it
is computing the right thing. This harness asks the other question — given the
same model, the same initial state and the same integrator, do phyz and MuJoCo
trace the same trajectory, and for how long?

    python3 mujoco_agreement.py --json agreement.json --markdown agreement.md

Requires `mujoco` (`pip install mujoco`). Deliberately not wired into CI or
`make bench`, for the same reason as `mujoco_bench.py`: a Python dependency
would put the Rust benchmark suite out of reach for most people.

What is compared
----------------

**Articulated-body forward dynamics on a kinematic tree under gravity, and
nothing else.** Both engines are stripped down to that before the comparison:

* joint damping, springs and dry friction are zeroed,
* joint limits are removed,
* contact, equality and limit constraints are disabled,
* the integrator is semi-implicit Euler on both sides.

Armature is deliberately *kept*: it is a constant added to the mass-matrix
diagonal, implemented identically by both engines, so it belongs to the
dynamics rather than to the modelling differences. On a model with very light
distal links it is also what keeps the mass matrix well conditioned — zeroing
it makes both engines integrate an ill-conditioned system at 1 ms and measures
the timestep rather than the agreement.

Every one of those is modelled differently by the two engines — MuJoCo
integrates damping implicitly and treats limits as constraints; phyz applies
damping explicitly and limits as a soft penalty. Leaving any of them on would
produce a divergence number that measures the difference between two
approximations of a *constraint*, not between two computations of the
*dynamics*. The Rust side does the same stripping and reports what it changed;
this script asserts the two lists are compatible.

Free-floating bases are out of scope for now: MuJoCo's `qpos` packs a free
joint as position-then-quaternion, phyz's differentiable layout differs from
its `Model` layout, and aligning them is a separate piece of work. Models with
a free joint are skipped **and reported as skipped**, never silently dropped.

Reading the output
------------------

`max_abs_dq` is the largest absolute joint-coordinate difference over the whole
horizon. `divergence_time` is the first time the two trajectories separate by
more than `--tol`; `null` means they never did.

On a chaotic model the two will separate eventually no matter how correct both
are — that is the physics, not a bug, and it is why `divergence_time` is
reported next to the horizon rather than as a pass/fail. Compare it against
phyz's own 1-ulp divergence figure (`phyz-bench --suite divergence`): if two
engines separate on the same timescale that phyz separates from *itself* under
a one-ulp perturbation, the disagreement is chaos amplifying round-off, not a
difference in the dynamics.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import mujoco
except ImportError:  # pragma: no cover - environment-dependent
    print(
        "mujoco is not installed. `pip install mujoco`, or see this directory's "
        "README. This harness is optional and not part of `make bench`.",
        file=sys.stderr,
    )
    raise SystemExit(1)

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

DT = 1.0e-3
STEPS = 5000
TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Synthetic models. The repository's own MJCF files are mostly free-floating
# (ant, half cheetah, humanoid), so the fixed-base coverage comes from here:
# chains and trees that exercise hinge and slide joints, multiple axes,
# off-diagonal inertia, and branching topology.
# ---------------------------------------------------------------------------


def chain_xml(n: int, joint_type: str = "hinge", axis: str = "0 1 0") -> str:
    """An `n`-link serial chain hanging along -z."""
    body = ""
    for i in range(n):
        pos = "0 0 0" if i == 0 else "0 0 -1"
        body += f"""
      <body name="link{i}" pos="{pos}">
        <inertial pos="0 0 -0.5" mass="{1.0 + 0.1 * i}"
                  fullinertia="0.11 0.13 0.09 0.01 0.02 0.015"/>
        <joint name="j{i}" type="{joint_type}" axis="{axis}"/>"""
    body += "\n      </body>" * n
    return f"""<mujoco model="chain{n}">
  <option gravity="0 0 -9.81" timestep="{DT}" integrator="Euler"/>
  <worldbody>{body}
  </worldbody>
</mujoco>"""


def tree_xml() -> str:
    """A branching tree: two children off one root, different joint axes."""
    return f"""<mujoco model="tree">
  <option gravity="0 0 -9.81" timestep="{DT}" integrator="Euler"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <inertial pos="0 0 -0.5" mass="2.0" fullinertia="0.2 0.25 0.15 0.01 0.0 0.02"/>
      <joint name="root_j" type="hinge" axis="0 1 0"/>
      <body name="left" pos="-0.3 0 -1">
        <inertial pos="0 0 -0.4" mass="1.0" diaginertia="0.08 0.08 0.02"/>
        <joint name="left_j" type="hinge" axis="1 0 0"/>
      </body>
      <body name="right" pos="0.3 0 -1">
        <inertial pos="0 0 -0.4" mass="0.7" fullinertia="0.06 0.07 0.02 0.005 0.001 0.0"/>
        <joint name="right_j" type="hinge" axis="0 1 0"/>
        <body name="tip" pos="0 0 -0.8">
          <inertial pos="0 0 -0.2" mass="0.3" diaginertia="0.01 0.01 0.005"/>
          <joint name="tip_j" type="slide" axis="0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def synthetic_models() -> dict[str, str]:
    return {
        "chain_1": chain_xml(1),
        "chain_2": chain_xml(2),
        "chain_4": chain_xml(4),
        "chain_8": chain_xml(8),
        "chain_x_axis": chain_xml(3, axis="1 0 0"),
        "slide_chain": chain_xml(3, joint_type="slide", axis="1 0 0"),
        "tree": tree_xml(),
    }


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def neutralise_mujoco(model: mujoco.MjModel) -> list[str]:
    """Strip everything the two engines model differently. Returns what changed.

    Kept deliberately parallel to `neutralise()` in `phyz_traj.rs` — if the two
    ever drift apart, the comparison silently stops being like-for-like, so the
    two lists are cross-checked by the caller.
    """
    changed: list[str] = []

    def note(s: str) -> None:
        if s not in changed:
            changed.append(s)

    if np.any(model.dof_damping != 0):
        model.dof_damping[:] = 0
        note("joint damping")
    if np.any(model.jnt_stiffness != 0) or np.any(model.qpos_spring != 0):
        model.jnt_stiffness[:] = 0
        model.qpos_spring[:] = 0
        note("joint springs")
    if np.any(model.dof_frictionloss != 0):
        model.dof_frictionloss[:] = 0
        note("dry friction")
    if np.any(model.jnt_limited != 0):
        model.jnt_limited[:] = 0
        note("joint limits")

    # Contact, equality and limit constraints off; semi-implicit Euler on.
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_EQUALITY
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
        | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        | mujoco.mjtDisableBit.mjDSBL_SPRING
        | mujoco.mjtDisableBit.mjDSBL_DAMPER
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    )
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    note("contact/equality/limit constraints")
    return changed


def mujoco_trajectory(path: Path, q0: np.ndarray, steps: int, dt: float):
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = dt
    neutralised = neutralise_mujoco(model)

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    free = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i]
        in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL)
    ]
    if free:
        return None, names, neutralised, "model has a free or ball joint"

    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0

    traj = np.empty((steps + 1, model.nq))
    traj[0] = data.qpos
    for k in range(steps):
        mujoco.mj_step(model, data)
        traj[k + 1] = data.qpos
    return traj, names, neutralised, None


def phyz_trajectory(binary: Path, path: Path, q0: np.ndarray, steps: int, dt: float):
    cmd = [
        str(binary),
        "--model",
        str(path),
        "--steps",
        str(steps),
        "--dt",
        repr(dt),
        "--q0",
        ",".join(repr(float(x)) for x in q0),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"phyz-traj failed: {out.stderr.strip()}")
    payload = json.loads(out.stdout)
    # serde_json writes a non-finite f64 as `null`, so a blown-up rollout
    # arrives as None rather than nan. Convert explicitly — silently coercing
    # would turn "this diverged" into a type error three functions later.
    q = np.array(
        [[np.nan if x is None else float(x) for x in row] for row in payload["q"]],
        dtype=float,
    )
    return q, payload


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(name: str, path: Path, binary: Path, steps: int, dt: float, tol: float):
    model = mujoco.MjModel.from_xml_path(str(path))
    nq = model.nq

    # An initial configuration that is off-axis everywhere, so no joint sits at
    # a symmetry point where the gravitational torque vanishes and both engines
    # trivially agree by doing nothing. Bounded and small: a monotonically
    # growing seed folds a many-jointed model (a hand, say) through itself into
    # a configuration whose dynamics are stiff enough to be unstable at 1 ms in
    # *both* engines, which measures the timestep rather than the agreement.
    q0 = np.array([0.15 * np.sin(i + 1) for i in range(nq)])

    mj_traj, mj_names, mj_neutralised, skip = mujoco_trajectory(path, q0, steps, dt)
    if skip is not None:
        return {"model": name, "skipped": skip}

    phyz_traj, payload = phyz_trajectory(binary, path, q0, steps, dt)

    if phyz_traj.shape != mj_traj.shape:
        return {
            "model": name,
            "skipped": f"shape mismatch: phyz {phyz_traj.shape} vs mujoco {mj_traj.shape}",
        }

    # Joint order must line up or the comparison is nonsense. Both engines walk
    # the body tree in declaration order, so this should hold; assert it rather
    # than assume it.
    #
    # phyz materialises a zero-DOF `Fixed` joint for every body MJCF attaches
    # rigidly to its parent. Those carry no configuration coordinate and have
    # no MuJoCo counterpart, so they are dropped before comparing order — by
    # DOF count, not by whether the name happens to be empty.
    phyz_names = [
        n
        for n, ndof in zip(payload["joint_names"], payload["joint_ndof"])
        if ndof > 0
    ]
    if [n for n in phyz_names if n] and phyz_names != mj_names:
        return {
            "model": name,
            "skipped": f"joint order differs: phyz {phyz_names} vs mujoco {mj_names}",
        }

    # The stripping has to have been equivalent on both sides.
    phyz_neutralised = set(payload["neutralised"])
    mj_only = {"contact/equality/limit constraints"}
    if phyz_neutralised - set(mj_neutralised) - mj_only:
        return {
            "model": name,
            "skipped": (
                f"neutralisation differs: phyz stripped {sorted(phyz_neutralised)}, "
                f"mujoco stripped {sorted(mj_neutralised)}"
            ),
        }

    # A rollout that went non-finite has no agreement to report. Say which
    # engine did it and when, rather than propagating nan into the metrics or
    # crashing three lines later.
    for label, traj in (("phyz", phyz_traj), ("mujoco", mj_traj)):
        bad = np.nonzero(~np.isfinite(traj).all(axis=1))[0]
        if bad.size:
            return {
                "model": name,
                "skipped": (
                    f"{label} went non-finite at t = {bad[0] * dt:.4f} s "
                    f"(step {int(bad[0])}); unstable at this timestep, so there "
                    f"is no trajectory to compare"
                ),
            }

    diff = np.abs(phyz_traj - mj_traj)
    per_step = diff.max(axis=1)
    exceeded = np.nonzero(per_step > tol)[0]
    divergence_time = float(exceeded[0] * dt) if exceeded.size else None

    return {
        "model": name,
        "nq": int(nq),
        "steps": steps,
        "dt": dt,
        "horizon_s": steps * dt,
        "tol": tol,
        "max_abs_dq": float(per_step.max()),
        "max_abs_dq_at_1s": float(per_step[: min(len(per_step), int(1.0 / dt))].max()),
        "final_abs_dq": float(per_step[-1]),
        "divergence_time": divergence_time,
        "neutralised": sorted(phyz_neutralised | set(mj_neutralised)),
        "joint_names": mj_names,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "target" / "release" / "phyz-traj",
        help="phyz-traj binary (cargo build --release -p phyz-bench)",
    )
    args = ap.parse_args()

    if not args.binary.exists():
        print(
            f"{args.binary} not found — build it with:\n"
            f"  cargo build --release -p phyz-bench --bin phyz-traj",
            file=sys.stderr,
        )
        return 1

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, xml in synthetic_models().items():
            path = tmp / f"{name}.xml"
            path.write_text(xml)
            results.append(compare(name, path, args.binary, args.steps, args.dt, args.tol))

        # The repository's own models, so the harness covers files a user would
        # actually load rather than only ones written for it.
        for path in sorted(MODELS_DIR.glob("*.xml")):
            results.append(
                compare(path.stem, path, args.binary, args.steps, args.dt, args.tol)
            )

    report = {
        "schema_version": 1,
        "harness": "mujoco_agreement",
        "mujoco_version": mujoco.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "settings": {"steps": args.steps, "dt": args.dt, "tol": args.tol},
        "results": results,
    }

    lines = [
        "### phyz vs MuJoCo — trajectory agreement",
        "",
        f"MuJoCo {mujoco.__version__}, {args.steps} steps at dt = {args.dt} s "
        f"({args.steps * args.dt:g} s), tolerance {args.tol:g}.",
        "",
        "Gravity-driven articulated dynamics only: damping, springs, dry "
        "friction and joint limits are zeroed on both sides, contact and "
        "constraints disabled, armature kept, semi-implicit Euler on both.",
        "",
        "| model | nq | max abs Δq | Δq at 1 s | first divergence | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"| `{r['model']}` | — | — | — | — | skipped: {r['skipped']} |")
            continue
        div = "never" if r["divergence_time"] is None else f"{r['divergence_time']:.3f} s"
        verdict = "agrees" if r["divergence_time"] is None else "diverges"
        lines.append(
            f"| `{r['model']}` | {r['nq']} | {r['max_abs_dq']:.3e} | "
            f"{r['max_abs_dq_at_1s']:.3e} | {div} | {verdict} |"
        )
    markdown = "\n".join(lines)

    print(markdown)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}", file=sys.stderr)
    if args.markdown:
        args.markdown.write_text(markdown + "\n")
        print(f"wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
