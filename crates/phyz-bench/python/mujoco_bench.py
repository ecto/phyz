#!/usr/bin/env python3
"""Optional MuJoCo / MJX comparison harness for phyz.

Deliberately separate from the Rust build and from CI: MuJoCo needs a Python
environment, and MJX needs JAX and (to be interesting) a GPU. Making that a
hard dependency of `make bench` would make the benchmark suite unrunnable for
most people, which is worse than not having the comparison.

Emits the same JSON shape as the Rust harness (schema_version 1) so results
merge into one document.

Scenes match `crates/phyz-bench/src/scenes.rs`:

  pendulum         1 revolute link, 1 m, 1 kg, hinge about +Y, no contact
  double_pendulum  2 such links
  ant              models/ant.xml, loaded by MuJoCo directly
  box_stack_8      8 free 10 cm / 1 kg boxes on a ground plane

Settings match `crates/phyz-bench/src/settings.rs`: dt = 1 ms (articulated) or
2 ms (contact), gravity -9.81 z, 4 solver iterations, friction 0.5,
restitution 0.

Read the caveats printed into the output. MuJoCo's soft-constraint contact
model is not phyz's penalty model and is not Rapier's impulse solver; the
contact row compares three different algorithms doing the same job.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ANT_XML = REPO_ROOT / "models" / "ant.xml"

DT_ARTICULATED = 1.0e-3
DT_CONTACT = 2.0e-3
SOLVER_ITERATIONS = 4
FRICTION = 0.5
RESTITUTION = 0.0
GRAVITY = 9.81

STEPS_PER_REP = 20_000
WARMUP_REPS = 2
REPS = 7

# --------------------------------------------------------------------------
# Scene definitions as MJCF strings, matching the phyz models term for term.
# --------------------------------------------------------------------------

PENDULUM_XML = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="{dt}" iterations="{iters}"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 -0.5" mass="1" diaginertia="0.0833333333 0.0833333333 0"/>
    </body>
  </worldbody>
</mujoco>
"""

DOUBLE_PENDULUM_XML = """
<mujoco model="double_pendulum">
  <option gravity="0 0 -9.81" timestep="{dt}" iterations="{iters}"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 -0.5" mass="1" diaginertia="0.0833333333 0.0833333333 0"/>
      <body name="link2" pos="0 0 -1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 -0.5" mass="1" diaginertia="0.0833333333 0.0833333333 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def box_stack_xml(n: int, dt: float) -> str:
    half, gap, mass = 0.05, 0.01, 1.0
    inertia = mass * (2 * half) ** 2 / 6.0
    bodies = []
    for i in range(n):
        z = half + i * (2 * half + gap)
        bodies.append(
            f"""
    <body name="box{i}" pos="0 0 {z}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia} {inertia} {inertia}"/>
      <geom type="box" size="{half} {half} {half}" friction="{FRICTION} 0.005 0.0001"/>
    </body>"""
        )
    return f"""
<mujoco model="box_stack_{n}">
  <option gravity="0 0 -{GRAVITY}" timestep="{dt}" iterations="{SOLVER_ITERATIONS}"/>
  <default>
    <geom solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.1" friction="{FRICTION} 0.005 0.0001"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""


SCENES = {
    "pendulum": dict(
        dt=DT_ARTICULATED,
        xml=lambda: PENDULUM_XML.format(dt=DT_ARTICULATED, iters=SOLVER_ITERATIONS),
        qpos0=[1.0],
        description="1 revolute link, 1 m, 1 kg, no contact",
        contact=False,
    ),
    "double_pendulum": dict(
        dt=DT_ARTICULATED,
        xml=lambda: DOUBLE_PENDULUM_XML.format(
            dt=DT_ARTICULATED, iters=SOLVER_ITERATIONS
        ),
        qpos0=[1.0, 0.5],
        description="2 revolute links, 1 m / 1 kg each, no contact",
        contact=False,
    ),
    "ant": dict(
        dt=DT_ARTICULATED,
        xml=None,  # loaded from models/ant.xml
        qpos0=None,
        description="free-floating torso + 8 hinges (14 DOF), no contact",
        contact=False,
    ),
    "box_stack_8": dict(
        dt=DT_CONTACT,
        xml=lambda: box_stack_xml(8, DT_CONTACT),
        qpos0=None,
        description="8 free 10 cm boxes (1 kg) stacked on a ground plane, contact enabled",
        contact=True,
    ),
}


def timing_block(samples: list[float], work_per_rep: int) -> dict:
    """Match the Rust harness's `Timing` shape exactly."""
    per_unit = sorted(s / work_per_rep for s in samples)
    median = per_unit[len(per_unit) // 2]
    lo, hi = per_unit[0], per_unit[-1]
    return {
        "reps": len(per_unit),
        "work_per_rep": work_per_rep,
        "median_sec_per_unit": median,
        "min_sec_per_unit": lo,
        "max_sec_per_unit": hi,
        "spread": (hi - lo) / median if median else 0.0,
        "throughput_per_sec": 1.0 / median if median else float("inf"),
    }


def settings_block(dt: float, contact: bool, caveats: list[str]) -> dict:
    return {
        "dt": dt,
        "gravity": [0.0, 0.0, -GRAVITY],
        "solver_iterations": SOLVER_ITERATIONS,
        "contact": (
            {
                "stiffness": 1.0e5,
                "damping": 1.0e3,
                "friction": FRICTION,
                "restitution": RESTITUTION,
            }
            if contact
            else None
        ),
        "caveats": caveats,
    }


MUJOCO_CAVEATS = [
    "MuJoCo contact is a soft-constraint model parameterised by solref/solimp. "
    "Only the friction coefficient is directly comparable to phyz's penalty "
    "model; solref/solimp were left at MuJoCo's defaults rather than tuned to "
    "imitate phyz, because tuning them to match would be choosing the answer.",
    "MuJoCo's `iterations` is set to 4 to match the solver-iteration budget "
    "asked of every engine, but its solver is a different algorithm (PGS/CG on "
    "a convex formulation) and the iteration counts are not equivalent units of "
    "work.",
    "MuJoCo runs f64, matching phyz. MJX runs f32 on GPU by default.",
]


def bench_mujoco(scene_name: str) -> dict:
    import mujoco  # imported late so --help works without it

    spec = SCENES[scene_name]
    if spec["xml"] is None:
        model = mujoco.MjModel.from_xml_path(str(ANT_XML))
        model.opt.timestep = spec["dt"]
        model.opt.iterations = SOLVER_ITERATIONS
    else:
        model = mujoco.MjModel.from_xml_string(spec["xml"]())
    data = mujoco.MjData(model)

    def reset():
        mujoco.mj_resetData(model, data)
        if spec["qpos0"] is not None:
            for i, q in enumerate(spec["qpos0"]):
                data.qpos[i] = q
        elif scene_name == "ant":
            data.qpos[2] = 0.75
            for i in range(7, model.nq):
                data.qpos[i] = 0.1
        mujoco.mj_forward(model, data)

    def run():
        reset()
        for _ in range(STEPS_PER_REP):
            mujoco.mj_step(model, data)

    for _ in range(WARMUP_REPS):
        run()
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        run()
        samples.append(time.perf_counter() - t0)

    timing = timing_block(samples, STEPS_PER_REP)
    return {
        "engine": f"mujoco {mujoco.__version__}",
        "scene": scene_name,
        "description": spec["description"],
        "dof": int(model.nv),
        "batch": 1,
        "settings": settings_block(spec["dt"], spec["contact"], MUJOCO_CAVEATS),
        "timing": timing,
        "metrics": [
            {
                "name": "realtime_factor",
                "value": timing["throughput_per_sec"] * spec["dt"],
                "unit": "x",
            },
            {
                "name": "stable",
                "value": 1.0 if all(abs(q) < 1e6 for q in data.qpos) else 0.0,
                "unit": "bool",
            },
        ],
        "notes": [
            "Measured the same way as the Rust harness: fixed warmup, "
            f"{REPS} timed repetitions of {STEPS_PER_REP} steps, median reported.",
            "Timed through MuJoCo's Python bindings. mj_step is a single C call "
            "per step, so per-step interpreter overhead is one FFI crossing — "
            "small relative to the ant and box-stack scenes, NOT small relative "
            "to the single pendulum. Treat the pendulum row as an upper bound on "
            "MuJoCo's cost, not a measurement of its C core.",
        ],
    }


def bench_mjx(scene_name: str, batch_sizes: list[int]) -> list[dict]:
    import jax
    import mujoco
    from mujoco import mjx

    spec = SCENES[scene_name]
    if spec["xml"] is None:
        model = mujoco.MjModel.from_xml_path(str(ANT_XML))
        model.opt.timestep = spec["dt"]
    else:
        model = mujoco.MjModel.from_xml_string(spec["xml"]())

    mjx_model = mjx.put_model(model)
    records = []

    for batch in batch_sizes:
        data = mjx.make_data(mjx_model)
        batched = jax.vmap(lambda _: data)(jax.numpy.arange(batch))

        @jax.jit
        def rollout(d):
            def one(carry, _):
                return mjx.step(mjx_model, carry), None

            out, _ = jax.lax.scan(one, d, None, length=STEPS_PER_REP_MJX)
            return out

        step_fn = jax.vmap(rollout)
        # Compile before timing: JIT time is real but is not per-step cost.
        jax.block_until_ready(step_fn(batched))

        samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(step_fn(batched))
            samples.append(time.perf_counter() - t0)

        work = STEPS_PER_REP_MJX * batch
        timing = timing_block(samples, work)
        records.append(
            {
                "engine": f"mjx (jax {jax.__version__})",
                "scene": scene_name,
                "description": f"{spec['description']} — batch {batch}",
                "dof": int(model.nv),
                "batch": batch,
                "settings": settings_block(spec["dt"], spec["contact"], MUJOCO_CAVEATS),
                "timing": timing,
                "metrics": [],
                "notes": [
                    f"Backend: {jax.default_backend()}.",
                    "JIT compilation is excluded from the timing (compiled once "
                    "before the timed repetitions). Compilation is a real cost at "
                    "small batch sizes and is not represented here.",
                    "MJX defaults to f32; the phyz CPU path is f64. Not a "
                    "like-for-like precision comparison.",
                ],
            }
        )
    return records


STEPS_PER_REP_MJX = 1000


def run_cmd(*args) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def collect_meta(extra_libs: list[dict]) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu": run_cmd("sysctl", "-n", "machdep.cpu.brand_string")
            or platform.processor()
            or "unknown",
            "cores": len(__import__("os").sched_getaffinity(0))
            if hasattr(__import__("os"), "sched_getaffinity")
            else __import__("os").cpu_count() or 0,
            "memory_gib": None,
            "gpu": None,
        },
        "os": {
            "family": platform.system().lower(),
            "release": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
        },
        "toolchain": {
            "rustc": "n/a (Python harness)",
            "profile": f"python {platform.python_version()}",
            "rustflags": None,
        },
        "libraries": extra_libs,
        "git_commit": run_cmd("git", "rev-parse", "HEAD"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="mujoco-results.json", help="output path")
    ap.add_argument(
        "--scenes",
        nargs="*",
        default=list(SCENES),
        choices=list(SCENES),
        help="scenes to run",
    )
    ap.add_argument("--mjx", action="store_true", help="also run the MJX batch sweep")
    ap.add_argument(
        "--mjx-batches",
        nargs="*",
        type=int,
        default=[1, 128, 1024, 4096],
        help="MJX batch sizes",
    )
    args = ap.parse_args()

    try:
        import mujoco
    except ImportError:
        print(
            "mujoco is not installed. See crates/phyz-bench/python/README.md",
            file=sys.stderr,
        )
        return 1

    libs = [{"name": "mujoco", "version": mujoco.__version__, "note": "f64 C core via Python bindings"}]

    suites = []
    results = [bench_mujoco(s) for s in args.scenes]
    suites.append(
        {
            "name": "MuJoCo comparison",
            "description": "MuJoCo on scenes matching the phyz benchmark suite, measured "
            "with the same protocol. Run separately from the Rust harness.",
            "skipped": None,
            "results": results,
        }
    )

    if args.mjx:
        try:
            import jax

            libs.append({"name": "jax", "version": jax.__version__, "note": jax.default_backend()})
            mjx_results = bench_mjx("ant", args.mjx_batches)
            suites.append(
                {
                    "name": "MJX batched comparison",
                    "description": "MJX batched rollouts, the closest external analogue to "
                    "the phyz-gpu batch sweep.",
                    "skipped": None,
                    "results": mjx_results,
                }
            )
        except ImportError as e:
            suites.append(
                {
                    "name": "MJX batched comparison",
                    "description": "MJX batched rollouts.",
                    "skipped": f"jax/mjx not available: {e}",
                    "results": [],
                }
            )

    report = {"schema_version": 1, "meta": collect_meta(libs), "suites": suites}
    Path(args.json).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.json}", file=sys.stderr)

    for suite in suites:
        print(f"\n## {suite['name']}")
        if suite["skipped"]:
            print(f"skipped: {suite['skipped']}")
            continue
        for r in suite["results"]:
            t = r["timing"]
            print(
                f"  {r['engine']:24s} {r['scene']:18s} batch={r['batch']:<6d} "
                f"{t['throughput_per_sec']:>12,.0f} steps/s  "
                f"({t['median_sec_per_unit'] * 1e6:.3f} µs/step)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
