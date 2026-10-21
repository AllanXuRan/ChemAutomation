"""Add UFACTORY G2's nominal TCP and three cameras to an existing physics MJCF.

Keep V12 geometry untouched. The output is a SEPARATE XML, not an overwrite.

From xArm official bio_gripper.urdf.xacro:
  - left_finger_joint anchor in gripper-base: [0.059, 0, 0.027]
  - link_tcp origin in gripper-base:      [0.135, 0, 0.055]

Supports MJCF exporters that fuse the fixed bio_gripper_base_link body.
Requires MuJoCo 3.x, numpy. No ROS 2.

  python setup_tcp_cameras.py
  python setup_tcp_cameras.py --reset-camera-config
  python setup_tcp_cameras.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'xarm_description' / 'xarm5_lab_physics.xml'
OUTPUT = ROOT / 'xarm_description' / 'xarm5_lab_fk_camera.xml'
CONFIG = ROOT / 'camera_setup.json'
CALIBRATION = ROOT / 'camera_calibration_v15.json'

G2_FINGER_ANCHOR_LOCAL = np.array([0.059, 0.0, 0.027], dtype=float)
G2_TCP_LOCAL = np.array([0.135, 0.0, 0.055], dtype=float)
CAMERA_NAMES = ('global_overhead', 'wrist_gripper', 'plate_overhead')
OLD_CAMERAS = {'ot2_rgbd', 'reader_rgbd', 'waiting_rgbd'}
CAMERA_VISUAL_PARTS = ('housing', 'lens', 'indicator')
DEFAULT_ARM_Q = [0.0, -0.60, -1.20, 0.80, 0.0]


def fmt(values):
    return ' '.join(f'{float(x):.10g}' for x in values)


def find_id(model, kind, name):
    obj_id = mujoco.mj_name2id(model, kind, name)
    if obj_id < 0:
        raise RuntimeError(f'{name!r} not found; expected a G2 gripper in the physics MJCF')
    return int(obj_id)


def set_arm_pose(model, data):
    data.qpos[:] = model.qpos0
    for i, angle in enumerate(DEFAULT_ARM_Q, 1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
        if jid < 0:
            raise RuntimeError(f'Missing joint{i}')
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            angle = float(np.clip(angle, lo, hi))
        data.qpos[int(model.jnt_qposadr[jid])] = angle
    mujoco.mj_forward(model, data)


def gripper_frame_in_parent(model, data):
    """Return parent body and gripper base frame (position and R) in parent.

    left_finger_joint is a prismatic joint with zero rpy relative to G2 base.
    Its anchor and axis orientation reconstruct the G2 base even when fused.
    """
    jid = find_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_finger_joint')
    child = int(model.jnt_bodyid[jid])
    parent = int(model.body_parentid[child])
    if parent <= 0:
        raise RuntimeError('Left finger parent is world; cannot attach camera/TCP correctly')
    # G2 left finger is translated, without rotating, from the gripper base.
    rot_gripper_world = data.xmat[child].reshape(3, 3).copy()
    p_gripper_world = np.array(data.xanchor[jid], float) - rot_gripper_world @ G2_FINGER_ANCHOR_LOCAL
    rot_parent_world = data.xmat[parent].reshape(3, 3).copy()
    p_parent_world = data.xpos[parent].copy()
    rot_gripper_parent = rot_parent_world.T @ rot_gripper_world
    p_gripper_parent = rot_parent_world.T @ (p_gripper_world - p_parent_world)
    return parent, p_gripper_parent, rot_gripper_parent


def name_of_body(model, body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def look_rotation(position, look_at, up_hint=(0, 1, 0)):
    """Local camera +X right, +Y up, -Z forward (expressed in parent frame)."""
    position = np.asarray(position, float)
    look_at = np.asarray(look_at, float)
    forward = look_at - position
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError('Camera position and look-at point cannot coincide')
    forward /= norm
    up = np.asarray(up_hint, float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, [1., 0., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return np.column_stack((right, up, -forward))


def quat_of(rot):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).ravel())
    return q


def site_point(model, data, name):
    sid = find_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


def default_config(model, data):
    ot2 = site_point(model, data, 'ot2_pickup')
    reader = site_point(model, data, 'versamax_load')
    centre = (ot2 + reader) / 2
    # Positions here are in WORLD frame for fixed cameras and G2 BASE frame for wrist.
    return {
        'cameras': {
            'global_overhead': {
                'frame': 'world',
                'position_xyz_m': np.round(centre + [0, -0.04, 1.45], 5).tolist(),
                'look_at_xyz_m': np.round(centre, 5).tolist(),
                'fovy_deg': 70.0
            },
            'wrist_gripper': {
                'frame': 'gripper',
                'position_xyz_m': [-0.035, -0.12, 0.14],
                'look_at_xyz_m': G2_TCP_LOCAL.tolist(),
                'fovy_deg': 65.0
            },
            'plate_overhead': {
                'frame': 'world',
                'position_xyz_m': np.round(ot2 + [0, 0, 0.45], 5).tolist(),
                'look_at_xyz_m': np.round(ot2, 5).tolist(),
                'fovy_deg': 48.0
            },
        },
        'notes': 'world=robot base, gripper=UFACTORY G2 base frame; metres; virtual pinhole cameras'
    }


def validate_config(config):
    cams = config['cameras']
    for name in CAMERA_NAMES:
        if name not in cams:
            raise ValueError(f'camera_setup.json missing {name}')
        cf = cams[name]
        expect = 'gripper' if name == 'wrist_gripper' else 'world'
        if cf.get('frame') != expect:
            raise ValueError(f'{name}.frame must be {expect!r}')
        for key in ('position_xyz_m', 'look_at_xyz_m'):
            p = np.asarray(cf[key], dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError(f'{name}.{key} must be three finite values')
        if np.linalg.norm(np.asarray(cf['look_at_xyz_m']) - np.asarray(cf['position_xyz_m'])) < 1e-8:
            raise ValueError(f'{name} position and target are the same')
        fov = float(cf['fovy_deg'])
        if not np.isfinite(fov) or not 2 < fov < 170:
            raise ValueError(f'{name}.fovy_deg must be between 2 and 170')


def get_camera_pose_in_parent(name, cf, gp, gR):
    pos = np.asarray(cf['position_xyz_m'], float)
    target = np.asarray(cf['look_at_xyz_m'], float)
    if name == 'wrist_gripper':
        p = gp + gR @ pos
        t = gp + gR @ target
        up = gR @ np.array([0, 0, 1.], float)
        return p, look_rotation(p, t, up)
    return pos, look_rotation(pos, target, [0, 1, 0])


def find_body_xml(root, name):
    return next((node for node in root.iter('body') if node.get('name') == name), None)


def add_camera_visuals(parent, name, p, rot):
    """Non-colliding, zero-mass camera housing behind the optical centre.

    Both fixed cameras and the wrist camera share the SAME parent frame as
    their virtual pinhole camera; therefore both move together automatically.
    MuJoCo optical axis is camera local -Z.
    """
    p = np.asarray(p, dtype=float)
    R = np.asarray(rot, dtype=float)
    q = quat_of(R)
    # Keep the body entirely BEHIND the optical origin (+Z) to avoid
    # accidentally rendering the camera case in its own RGB image.
    parts = (
        ('housing', (0.0, 0.0, 0.041), 'box', '0.034 0.027 0.027', '0.17 0.22 0.30 1'),
        ('lens', (0.0, 0.0, 0.010), 'cylinder', '0.013 0.006', '0.04 0.08 0.12 1'),
        ('indicator', (0.022, 0.019, 0.012), 'sphere', '0.004', '0.09 0.85 0.26 1'),
    )
    for part, offset, shape, size, color in parts:
        centre = p + R @ np.asarray(offset, float)
        ET.SubElement(parent, 'geom', name=f'cam_{name}_{part}',
                      type=shape, pos=fmt(centre), quat=fmt(q), size=size,
                      rgba=color, group='1', contype='0', conaffinity='0',
                      mass='0')


def remove_named_cameras(root):
    remove_names = OLD_CAMERAS | set(CAMERA_NAMES)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'camera' and child.get('name') in remove_names:
                parent.remove(child)
            if child.tag == 'geom' and any(child.get('name') == f'cam_{n}_{part}'
                                          for n in CAMERA_NAMES for part in CAMERA_VISUAL_PARTS):
                parent.remove(child)


def build(source=SOURCE, output=OUTPUT, config_path=CONFIG, reset_config=False):
    source, output, config_path = Path(source), Path(output), Path(config_path)
    if not source.is_file():
        raise FileNotFoundError(f'Missing physics scene: {source}\nRun prepare_physics_scene.py first')
    # Read MuJoCo compiled kinematics to handle bodies fused in URDF conversion.
    model = mujoco.MjModel.from_xml_path(str(source.resolve()))
    data = mujoco.MjData(model)
    set_arm_pose(model, data)
    parent_id, p_g, R_g = gripper_frame_in_parent(model, data)
    parent_name = name_of_body(model, parent_id)
    if not parent_name:
        raise RuntimeError('Could not find parent body name for the gripper finger joint')

    if reset_config or not config_path.exists():
        configuration = default_config(model, data)
        config_path.write_text(json.dumps(configuration, indent=2, ensure_ascii=False), encoding='utf-8')
        print('[CONFIG] Created', config_path)
    else:
        configuration = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(configuration)

    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Physics scene has no worldbody')
    parent_xml = find_body_xml(root, parent_name)
    if parent_xml is None:
        raise RuntimeError(f'Compiled body {parent_name!r} is absent from XML (export mismatch)')
    for old in list(parent_xml.findall('site')):
        if old.get('name') == 'g2_tcp':
            parent_xml.remove(old)
    tcp_parent = p_g + R_g @ G2_TCP_LOCAL
    ET.SubElement(parent_xml, 'site', name='g2_tcp', type='sphere', pos=fmt(tcp_parent),
                  size='0.008', rgba='0.05 0.95 0.28 1')
    remove_named_cameras(root)
    # A visible (but non-colliding) IK target marker; updated at runtime.
    initial_goal = site_point(model, data, 'ot2_pickup')
    for old in list(world.findall('site')):
        if old.get('name') == 'ik_target':
            world.remove(old)
    ET.SubElement(world, 'site', name='ik_target', type='sphere',
                  pos=fmt(initial_goal), size='0.014', rgba='1 0.15 0.05 0.9',
                  group='1')

    for name in CAMERA_NAMES:
        cf = configuration['cameras'][name]
        pos, rot = get_camera_pose_in_parent(name, cf, p_g, R_g)
        container = parent_xml if name == 'wrist_gripper' else world
        ET.SubElement(container, 'camera', name=name, mode='fixed',
                      pos=fmt(pos), xyaxes=fmt([*rot[:, 0], *rot[:, 1]]),
                      fovy=f'{float(cf["fovy_deg"]):.10g}')
        #add_camera_visuals(container, name, pos, rot)

    # Keep original xArm, OT-2, VersaMax, collision geoms & dynamics unchanged.
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    try:
        checked = mujoco.MjModel.from_xml_path(str(output.resolve()))
    except Exception:
        print(f'Output XML saved at {output}; MuJoCo compilation failed:')
        raise
    print(f'[OK] Output: {output}')
    print(f'[OK] G2 TCP site attached to MuJoCo parent body: {parent_name}')
    print(f'     Local position in that body: {np.round(tcp_parent, 5).tolist()} m')
    print(f'     Official G2-base TCP: {G2_TCP_LOCAL.tolist()} m')
    print('     Camera housings hidden; virtual cameras enabled')
    print('     Red sphere: IK target; green sphere: G2 TCP')
    print('     Cameras: ' + ', '.join(CAMERA_NAMES))
    for name in CAMERA_NAMES:
        cid = find_id(checked, mujoco.mjtObj.mjOBJ_CAMERA, name)
        bname = name_of_body(checked, int(checked.cam_bodyid[cid]))
        print(f'     {name}: attached to {bname!r}')
    return output


def report(scene=OUTPUT, width=640, height=480, angles=None, export=CALIBRATION):
    scene = Path(scene)
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    set_arm_pose(model, data)
    if angles is not None:
        for i, v in enumerate(angles, 1):
            jid = find_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
            data.qpos[int(model.jnt_qposadr[jid])] = float(v)
        mujoco.mj_forward(model, data)
    sid = find_id(model, mujoco.mjtObj.mjOBJ_SITE, 'g2_tcp')
    out = {
        'convention': 'T_world_camera_cv with OpenCV x-right y-down z-forward; world at robot base',
        'image_size': [width, height],
        'joint_angles_rad': [float(data.qpos[model.jnt_qposadr[find_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]]) for i in range(1, 6)],
        'g2_tcp_xyz_world_m': data.site_xpos[sid].tolist(),
        'cameras': {},
        'warning': 'Wrist camera world extrinsics vary with joints. Values are only for the listed joint pose.'
    }
    for name in CAMERA_NAMES:
        cid = find_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        pos = np.array(data.cam_xpos[cid], float)
        R_mj = np.asarray(data.cam_xmat[cid], float).reshape(3, 3)
        R_cv = R_mj @ np.diag([1., -1., -1.])
        T = np.eye(4)
        T[:3, :3] = R_cv
        T[:3, 3] = pos
        fovy = float(model.cam_fovy[cid])
        f = height / (2 * math.tan(math.radians(fovy) / 2))
        K = [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]
        out['cameras'][name] = {
            'K': K, 'T_world_camera_cv': T.tolist(), 'position_world_m': pos.tolist(),
            'fovy_deg': fovy, 'parent_body': name_of_body(model, int(model.cam_bodyid[cid]))
        }
        print(f'{name}: world={np.round(pos, 4).tolist()}  fovy={fovy:.1f} deg')
    if export:
        Path(export).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[OK] Calibration snapshot: {export}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--scene', type=Path, default=OUTPUT)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--reset-camera-config', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--joints', type=float, nargs=5)
    args = ap.parse_args()
    if args.report:
        report(scene=args.scene, angles=args.joints)
    else:
        build(args.source, args.scene, args.config, args.reset_camera_config)
        report(scene=args.scene)


if __name__ == '__main__':
    main()
