"""Capture RGB and depth from the three V15 MuJoCo cameras.

python capture_three_cameras.py
python capture_three_cameras.py --camera wrist_gripper --width 640 --height 480

Dynamic wrist camera rendering uses the CURRENT JOINT CONFIGURATION of this script.
To capture robot motion, integrate this function in the control loop and pass mjData.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from setup_tcp_cameras import OUTPUT, CAMERA_NAMES, set_arm_pose

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=Path, default=OUTPUT)
    ap.add_argument('--camera', choices=('all', *CAMERA_NAMES), default='all')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--output', type=Path, default=ROOT / 'camera_output_v15')
    ap.add_argument('--joints', type=float, nargs=5, help='Robot joint pose in radians for the snapshot')
    args = ap.parse_args()
    if args.width <= 0 or args.height <= 0:
        ap.error('Width and height must be positive')
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    set_arm_pose(model, data)
    if args.joints:
        for i, q in enumerate(args.joints, 1):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
            if jid < 0:
                raise RuntimeError(f'Missing joint{i}')
            data.qpos[int(model.jnt_qposadr[jid])] = q
        mujoco.mj_forward(model, data)

    args.output.mkdir(parents=True, exist_ok=True)
    names = CAMERA_NAMES if args.camera == 'all' else [args.camera]
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for name in names:
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=name)
            rgb = renderer.render().copy()
            Image.fromarray(rgb).save(args.output / f'{name}_rgb.png')
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=name)
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            renderer.disable_depth_rendering()
            np.save(args.output / f'{name}_depth_m.npy', depth)
            print(f'Captured {name}: RGB + depth(m) in {args.output}')
    print('The camera positions are exact SIM ground-truth, not real-world calibration.')


if __name__ == '__main__':
    main()
