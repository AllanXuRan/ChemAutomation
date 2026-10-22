"""Ordered snapshots of actual simulation state for user-authored workflows."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import robot_kinematics as kin


def capture_state(runtime, tcp_id, name, sequence):
    """Call under the viewer lock for a consistent snapshot."""
    model, data = runtime.model, runtime.data
    p, rotation = kin.tcp_pose(data, tcp_id)
    joints = {}
    for index, (_, qadr, vadr) in enumerate(runtime.info):
        joints[kin.ARM[index]] = {
            'position_rad': float(data.qpos[qadr]),
            'position_deg': float(np.degrees(data.qpos[qadr])),
            'velocity_rad_s': float(data.qvel[vadr]),
            'target_rad': float(runtime.arm_target[index]),
            'target_deg': float(np.degrees(runtime.arm_target[index])),
        }
    for side in ('left', 'right'):
        jid = model.joint(f'{side}_finger_joint').id
        qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        actuator = model.actuator(f'{side}_finger_position').id
        joints[f'{side}_finger_joint'] = {
            'position_m': float(data.qpos[qadr]),
            'velocity_m_s': float(data.qvel[vadr]),
            'target_m': float(data.ctrl[actuator]),
        }
    flange = model.body('link_eef').id
    normal = data.xmat[flange].reshape(3, 3)[:, 2]
    plate = runtime.controller.plate
    return {
        'version': 1,
        'sequence': sequence,
        'name': name.strip() or f'step_{sequence:03d}',
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'simulation_time_s': float(data.time),
        'coordinate_frame': 'world (robot base)',
        'tcp_xyz_world_m': dict(zip('xyz', p.tolist())),
        'tcp_R_world': rotation.tolist(),
        'flange_tilt_deg': float(np.degrees(np.arccos(np.clip(-normal[2], -1, 1)))),
        'joints': joints,
        'bilateral_gripper_contact': bool(runtime.controller.held),
        'collision_blocked': runtime.blocked,
        'plate_xyz_world_m': data.xpos[plate].tolist(),
        'plate_quat_wxyz_world': data.xquat[plate].tolist(),
        'all_plates': {model.body(body).name: {
            'xyz_world_m': data.xpos[body].tolist(),
            'quat_wxyz_world': data.xquat[body].tolist(),
        } for body in sorted(runtime.controller.plate_bodies)},
    }


class StateRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.count = 0

    def append(self, runtime, tcp_id, name):
        record = capture_state(runtime, tcp_id, name, self.count+1)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(line)
        self.count += 1
        return record
