"""Replay ordered state captures in the lab's contact-physics simulation.

Run: python replay_trajectory.py
Check without a window: python replay_trajectory.py --headless
Sparse samples are joined with smooth joint-space segments, not original timing.
"""
from __future__ import annotations
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import robot_kinematics as kin
from physics_runtime import PhysicsRuntime, load_physics_model

ROOT = Path(__file__).resolve().parent


def load_records(path=None):
    if path is None:
        files = list((ROOT / 'outputs' / 'state_captures').glob('*.jsonl'))
        if not files:
            raise ValueError('No state captures found')
        path = max(files, key=lambda file: file.stat().st_mtime_ns)
    path = Path(path).resolve()
    records = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            q = np.array([record['joints'][name]['position_rad'] for name in kin.ARM])
            fingers = np.array([record['joints'][side+'_finger_joint']['target_m']
                                for side in ('left', 'right')])
            if not np.isfinite(q).all() or not np.isfinite(fingers).all():
                raise ValueError('Nonfinite joint values')
            records.append((record, q, fingers))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{path}:{line_number}: {exc}') from exc
    if not records:
        raise ValueError(f'Empty capture: {path}')
    return path, records


def replay(records, segment_seconds=3., hold_seconds=1., headless=False):
    model = load_physics_model(kin.SCENE)
    data = mujoco.MjData(model)
    info = kin.arm_info(model)
    for record, q, fingers in records:
        for angle, (jid, _, _) in zip(q, info):
            if model.jnt_limited[jid] and not model.jnt_range[jid, 0] <= angle <= model.jnt_range[jid, 1]:
                raise ValueError(f"Joint outside limits in {record['name']}")
        if not (-.04 <= fingers[0] <= 0 and 0 <= fingers[1] <= .04):
            raise ValueError('Gripper target outside limits')
    kin.set_angles(model, data, info, records[0][1])
    # The capture records the primary plate, but not individual stored plates.
    first = records[0][0]
    address = model.jnt_qposadr[model.joint('reaction_plate_free').id]
    if 'plate_xyz_world_m' in first and 'plate_quat_wxyz_world' in first:
        pose = np.array(first['plate_xyz_world_m'] + first['plate_quat_wxyz_world'], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError('Invalid captured plate pose')
        pose[3:] /= np.linalg.norm(pose[3:])
        data.qpos[address:address+7] = pose
    runtime = PhysicsRuntime(model, data, info)
    controller = runtime.controller
    for side, target in zip(('left', 'right'), records[0][2]):
        data.qpos[model.jnt_qposadr[model.joint(side+'_finger_joint').id]] = target
    controller.target[:] = records[0][2]
    data.ctrl[controller.act] = controller.target
    mujoco.mj_forward(model, data)
    stack = [body for body in controller.plate_bodies if body != controller.plate]
    start_heights = {body: float(data.xpos[body, 2]) for body in stack}
    viewer_context = nullcontext(None) if headless else mujoco.viewer.launch_passive(model, data)
    results = []
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [-.25, .48, .17]
            viewer.cam.distance = 1.45
            viewer.cam.azimuth = 115
            viewer.cam.elevation = -28
        print('Close the simulation window to exit. Physics replay, no object attachment.', flush=True)
        previous_q = records[0][1].copy()
        previous_fingers = records[0][2].copy()
        for index, (record, target_q, target_fingers) in enumerate(records):
            duration = segment_seconds if index else 0.
            frames = max(1, int(np.ceil((duration+hold_seconds)/.02)))
            print(f"[{index+1}/{len(records)}] {record['name']} captured contact={record.get('bilateral_gripper_contact')}", flush=True)
            for frame in range(frames):
                if viewer is not None and not viewer.is_running():
                    return results
                started = time.perf_counter()
                t = min(1., (frame+1)*.02/duration) if duration else 1.
                smooth = t*t*(3-2*t)
                runtime.set_arm_target(previous_q + smooth*(target_q-previous_q))
                controller.target[:] = previous_fingers + smooth*(target_fingers-previous_fingers)
                data.ctrl[controller.act] = controller.target
                runtime.advance(.02)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0., .02-(time.perf_counter()-started)))
            actual_q = kin.angles(data, info)
            error = float(np.max(np.abs(np.degrees(actual_q-target_q))))
            lift = max((float(data.xpos[b, 2])-start_heights[b] for b in stack), default=0.)
            result = dict(name=record['name'], joint_error_deg=error,
                          held=bool(controller.held), stack_lift_m=lift, blocked=runtime.blocked)
            results.append(result)
            print(json.dumps(result), flush=True)
            previous_q, previous_fingers = target_q.copy(), target_fingers.copy()
        if np.any(data.warning.number):
            raise RuntimeError('MuJoCo reported a physics warning')
        if viewer is not None:
            print('Replay complete. Close the simulation window to exit.', flush=True)
            while viewer.is_running():
                runtime.advance(.02)
                viewer.sync()
                time.sleep(.02)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='JSONL capture; defaults to newest')
    parser.add_argument('--segment-seconds', type=float, default=3.)
    parser.add_argument('--hold-seconds', type=float, default=1.)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.segment_seconds) or args.segment_seconds <= 0 or not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error('Segment seconds must be positive; hold seconds must be nonnegative')
    path, records = load_records(args.source)
    print(f'Source: {path}; {len(records)} captured states', flush=True)
    replay(records, args.segment_seconds, args.hold_seconds, args.headless)


if __name__ == '__main__':
    main()
