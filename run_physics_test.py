"""Test V12 derived dynamics: plate settling + xArm position holding.

Headless quick check:
  python run_physics_test.py --seconds 3
Native interactive MuJoCo Viewer (toggle Run to begin physics):
  python run_physics_test.py --viewer
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "xarm_description" / "xarm5_lab_physics.xml"
HOME = [0.0, -0.60, -1.20, 0.80, 0.0]


def start_state(model, data):
    data.qpos[:] = model.qpos0
    for i, angle in enumerate(HOME, start=1):
        name = f"joint{i}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint missing: {name}")
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            angle = max(float(lo), min(float(hi), angle))
        data.qpos[model.jnt_qposadr[jid]] = angle
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position")
        if aid >= 0:
            data.ctrl[aid] = angle
    mujoco.mj_forward(model, data)


def plate_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reaction_plate")
    return float(data.xpos[bid, 2]) if bid >= 0 else math.nan


def list_plate_contacts(model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reaction_plate_box")
    out = set()
    for k in range(data.ncon):
        contact = data.contact[k]
        if gid not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == gid else contact.geom1
        out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(other)))
    return sorted(str(x) for x in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--viewer", action="store_true", help="Run interactive MuJoCo viewer instead of headless test")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    path = args.scene.resolve()
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run prepare_physics_scene.py first")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    start_state(model, data)
    print(f"Physics scene: {path.name}; timestep={model.opt.timestep}s, actuators={model.nu}")
    print(f"Initial plate centre z: {plate_height(model, data):.4f}m")
    if args.viewer:
        from mujoco import viewer as mj_viewer
        print("MuJoCo Viewer: press Run to step physics; Control panel moves arm servo targets")
        mj_viewer.launch(model, data)
        return
    steps = max(1, int(args.seconds / model.opt.timestep))
    interval = max(1, int(0.5 / model.opt.timestep))
    for i in range(steps):
        mujoco.mj_step(model, data)
        if (i + 1) % interval == 0 or i == steps - 1:
            print(f"t={data.time:.2f}s plate_z={plate_height(model, data):.4f}m "
                  f"plate_contacts={list_plate_contacts(model, data)}")
    print("A resting plate should stop falling and have contact with a support surface.")
    print("If it falls through or jumps, inspect geometries, initial intersections and contact parameters.")


if __name__ == "__main__":
    main()
