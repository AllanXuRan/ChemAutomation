"""xArm5 FK and position-only IK at the official Bio Gripper G2 TCP site.

  python robot_kinematics.py list
  python robot_kinematics.py fk
  python robot_kinematics.py ik --target ot2_pickup --offset 0 0 0.10 --viewer
  python robot_kinematics.py ik --xyz -0.3 0.3 0.25 --viewer

Only joint positions are computed; no trajectory, collisions or physical grasping.
The G2 TCP site is installed using setup_tcp_cameras.py first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / 'xarm_description' / 'xarm5_lab_fk_camera.xml'
Q0 = np.array([0, -.60, -1.20, .80, 0], dtype=float)
ARM = [f'joint{i}' for i in range(1, 6)]


def arm_info(model):
    result = []
    for name in ARM:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f'Expected hinge joint: {name}')
        result.append((int(j), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))
    return result


def clip_angles(model, info, q):
    q = np.asarray(q, float).copy()
    for i, (jid, _, _) in enumerate(info):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[i] = np.clip(q[i], lo + 1e-7, hi - 1e-7)
    return q


def set_angles(model, data, info, q):
    for (_, ix, _), v in zip(info, q):
        data.qpos[ix] = float(v)
    mujoco.mj_forward(model, data)


def angles(data, info):
    return np.array([data.qpos[ix] for _, ix, _ in info])


def site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f'Unknown site {name}. Run setup_tcp_cameras.py, then robot_kinematics.py list')
    return int(sid)


def tcp_pose(data, sid):
    return np.array(data.site_xpos[sid], float), np.asarray(data.site_xmat[sid], float).reshape(3, 3).copy()


def jac_ik(model, data, info, tcp_sid, desired, seed, tol=0.003, max_iters=500, damping=.06):
    q = clip_angles(model, info, seed)
    set_angles(model, data, info, q)
    cols = [dof for _, _, dof in info]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    best_q = q.copy()
    best_error = np.inf
    for iteration in range(max_iters):
        p, _ = tcp_pose(data, tcp_sid)
        e = np.asarray(desired) - p
        err = np.linalg.norm(e)
        if err < best_error:
            best_error, best_q = float(err), q.copy()
        if err < tol:
            return q.copy(), float(err), iteration + 1
        jacp.fill(0)
        jacr.fill(0)
        mujoco.mj_jacSite(model, data, jacp, jacr, tcp_sid)
        J = jacp[:, cols]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), e)
        dq = np.clip(dq, -.10, .10)
        if np.linalg.norm(dq) < 1e-10:
            break
        improved = False
        for scale in (1., .5, .25, .125, .0625):
            trial = clip_angles(model, info, q + scale*dq)
            set_angles(model, data, info, trial)
            p_trial, _ = tcp_pose(data, tcp_sid)
            if np.linalg.norm(desired - p_trial) < err - 1e-9:
                q = trial
                improved = True
                break
        if not improved:
            break
    set_angles(model, data, info, best_q)
    return best_q, float(best_error), max_iters


def solve_ik(model, data, info, tcp_sid, target, tol):
    current = angles(data, info)
    seeds = [current, Q0, np.zeros(5), [0.7, -0.5, -1.0, 0.7, 0],
             [-0.7, -0.5, -1.0, 0.7, 0], [0.0, 0.3, -1.3, 0.5, 0.0]]
    best = (None, np.inf, 0)
    for seed in seeds:
        result = jac_ik(model, data, info, tcp_sid, target, seed, tol)
        if result[1] < best[1]:
            best = result
        if result[1] <= tol:
            break
    set_angles(model, data, info, best[0])
    return best


def target_position(model, data, args):
    if args.xyz is not None:
        return np.asarray(args.xyz, float)
    sid = site_id(model, args.target)
    return np.asarray(data.site_xpos[sid], float)


def show(model, data, camera='global_overhead'):
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as v:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid >= 0:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = cid
        else:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.distance = 2.0
        print('Viewer: press Esc to switch to free orbit camera; green dot is g2_tcp.')
        while v.is_running():
            v.sync()
            time.sleep(.03)


def main():
    ap = argparse.ArgumentParser(description='MuJoCo FK / IK using Bio Gripper G2 TCP')
    ap.add_argument('command', choices=('list', 'fk', 'ik'))
    ap.add_argument('--scene', type=Path, default=SCENE)
    ap.add_argument('--tcp-site', default='g2_tcp')
    ap.add_argument('--target', default='ot2_pickup')
    ap.add_argument('--xyz', type=float, nargs=3)
    ap.add_argument('--offset', type=float, nargs=3, default=[0., 0., 0.])
    ap.add_argument('--joints', type=float, nargs=5)
    ap.add_argument('--tolerance', type=float, default=.003)
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--camera', default='global_overhead')
    ap.add_argument('--save', type=Path)
    args = ap.parse_args()
    if not args.scene.is_file():
        ap.error(f'Missing {args.scene}; run setup_tcp_cameras.py')
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    info = arm_info(model)
    tcp_sid = site_id(model, args.tcp_site)
    q_init = clip_angles(model, info, args.joints if args.joints is not None else Q0)
    data.qpos[:] = model.qpos0
    set_angles(model, data, info, q_init)
    if args.command == 'list':
        for name, (jid, _, _) in zip(ARM, info):
            print(name, 'joint range:', model.jnt_range[jid].tolist())
        for i in range(model.nsite):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if n and (n == 'g2_tcp' or 'ot2' in n or 'versamax' in n):
                print('site:', n)
        for i in range(model.ncam):
            print('camera:', mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i))
        return
    target = None
    error = None
    if args.command == 'ik':
        target = target_position(model, data, args) + np.asarray(args.offset)
        q, error, iterations = solve_ik(model, data, info, tcp_sid, target, args.tolerance)
        print('Target XYZ:', np.round(target, 5).tolist())
        print('IK success:', bool(error <= args.tolerance), 'error(mm):', round(error*1000, 2), 'iterations:', iterations)
        if error > args.tolerance:
            print('NOT CONVERGED: check target reachability and orientation/collision separately.')
    tcp_xyz, tcp_rot = tcp_pose(data, tcp_sid)
    print('G2 TCP world XYZ:', np.round(tcp_xyz, 5).tolist())
    print('G2 TCP world R:\n', np.round(tcp_rot, 4))
    print('Arm joints rad:', np.round(angles(data, info), 5).tolist())
    print('Arm joints deg:', np.round(np.degrees(angles(data, info)), 3).tolist())
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({'mode': args.command, 'tcp_site': args.tcp_site,
                                         'tcp_xyz_world_m': tcp_xyz.tolist(),
                                         'target_xyz_world_m': target.tolist() if target is not None else None,
                                         'ik_error_m': error,
                                         'joint_angles_rad': angles(data, info).tolist()}, indent=2), encoding='utf-8')
    if args.viewer:
        # Keep physics actuators aligned if the model is unpaused later.
        for n, q in zip(ARM, angles(data, info)):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + '_position')
            if aid >= 0:
                data.ctrl[aid] = float(q)
        mujoco.mj_forward(model, data)
        show(model, data, args.camera)


if __name__ == '__main__':
    main()
