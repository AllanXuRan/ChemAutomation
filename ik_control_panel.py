"""Live FK / IK panel for xArm5 + Bio Gripper G2 in MuJoCo.

Uses V16 xarm_description/xarm5_lab_fk_camera.xml created by setup_tcp_cameras.py.
No ROS 2 and no changes to your OT-2 / VersaMax dimensions.

This is KINEMATIC visualization (mj_forward without mj_step): NOT a
collision-checked trajectory planner or a force-controlled physical grasp.

Run: python ik_control_panel.py
"""
from __future__ import annotations

import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import mujoco
import mujoco.viewer
import numpy as np

import robot_kinematics as kin

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / 'xarm_description' / 'xarm5_lab_fk_camera.xml'
OUTPUT_POSE = ROOT / 'last_ik_pose.json'
CAMERAS = ('Free camera', 'global_overhead', 'wrist_gripper', 'plate_overhead')
DEFAULT_TARGETS = ('ot2_pickup', 'versamax_load', 'ot2_waiting', 'ot2_stack_top')


def actuator_hold(model, data, joint_positions):
    """Align existing position-actuator targets with the kinematic pose."""
    for name, q in zip(kin.ARM, joint_positions):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                name + '_position')
        if aid >= 0:
            data.ctrl[aid] = float(q)


def main():
    if not SCENE.is_file():
        raise SystemExit(f'No {SCENE}\nRun: python setup_tcp_cameras.py')

    model = mujoco.MjModel.from_xml_path(str(SCENE.resolve()))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    info = kin.arm_info(model)
    tcp_id = kin.site_id(model, 'g2_tcp')
    goal_id = kin.site_id(model, 'ik_target')

    data.qpos[:] = model.qpos0
    q0 = kin.clip_angles(model, info, kin.Q0)
    kin.set_angles(model, data, info, q0)
    actuator_hold(model, data, q0)
    mujoco.mj_forward(model, data)

    targets = []
    for name in DEFAULT_TARGETS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
            targets.append(name)
    if not targets:
        print('WARNING: no named lab targets, use Manual XYZ')

    root = tk.Tk()
    root.title('xArm5 / G2 TCP - Live FK and IK')
    root.geometry('735x870')
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill='both', expand=True)

    ttk.Label(frame, text='G2 夹爪 TCP 控制台 — FK / IK',
              font=('Segoe UI', 14, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='世界坐标原点为 xArm5 底座；距离单位米；关节滑块单位为度。',
              wraplength=675).pack(anchor='w', pady=(4, 12))

    target_box = ttk.LabelFrame(frame, text='逆运动学 IK：TCP 去哪里？', padding=9)
    target_box.pack(fill='x', pady=3)
    tk_target = tk.StringVar(value=targets[0] if targets else 'Manual XYZ')
    ttk.Label(target_box, text='选择已知目标点').grid(row=0, column=0, sticky='w', pady=3)
    target_select = ttk.Combobox(target_box, textvariable=tk_target,
                                 values=['Manual XYZ', *targets], state='readonly', width=23)
    target_select.grid(row=0, column=1, sticky='w', padx=6)

    goal_fields = [tk.StringVar(value='0.00000') for _ in range(3)]
    for j, axis in enumerate('XYZ'):
        ttk.Label(target_box, text=f'{axis} (m)').grid(row=1, column=j*2, sticky='w', padx=4, pady=8)
        ttk.Entry(target_box, textvariable=goal_fields[j], width=13).grid(
            row=1, column=j*2+1, sticky='w', padx=3)
    offset_field = tk.StringVar(value='0.10')
    ttk.Label(target_box, text='目标上方偏移 Z (m)').grid(row=2, column=0, columnspan=2, sticky='w', pady=5)
    ttk.Entry(target_box, textvariable=offset_field, width=12).grid(row=2, column=2, sticky='w')
    ttk.Label(target_box, text='（设为 0 = 到目标点）').grid(row=2, column=3, columnspan=3, sticky='w')

    settings = ttk.Frame(target_box)
    settings.grid(row=3, column=0, columnspan=6, sticky='w', pady=6)
    allow_closest = tk.BooleanVar(value=False)
    ttk.Checkbutton(settings, text='未收敛时允许移动到最近解（仅供检查）',
                    variable=allow_closest).pack(side='left')
    tolerance = 0.003

    ik_buttons = ttk.Frame(target_box)
    ik_buttons.grid(row=4, column=0, columnspan=6, sticky='w', pady=6)

    joints_box = ttk.LabelFrame(frame, text='正运动学 FK：直接拖动五个关节（度）', padding=9)
    joints_box.pack(fill='x', pady=10)
    joint_scales = []
    joint_numbers = []
    suppress_slider = [False]
    moving = {'active': False, 'start': 0.0, 'from': None, 'to': None,
              'duration': 1.8}

    info_text = tk.StringVar(value='准备就绪。选择目标并点击 IK 按钮。')
    pose_text = tk.StringVar(value='')
    status_label = ttk.Label(frame, textvariable=info_text, wraplength=670,
                             foreground='#12578a')
    status_label.pack(fill='x', pady=(4, 2))
    ttk.Label(frame, textvariable=pose_text, font=('Consolas', 10),
              wraplength=670).pack(fill='x', pady=(0, 8))

    camera_box = ttk.LabelFrame(frame, text='MuJoCo 观察视角', padding=9)
    camera_box.pack(fill='x', pady=(3, 8))
    cam_sel = tk.StringVar(value='Free camera')
    ttk.Combobox(camera_box, textvariable=cam_sel, values=CAMERAS,
                 state='readonly', width=26).pack(side='left', padx=6)

    def read_current():
        with viewer.lock():
            return kin.angles(data, info).copy()

    def show_fk():
        # Call within UI thread; viewer.lock is held for consistent readings.
        with viewer.lock():
            p, _ = kin.tcp_pose(data, tcp_id)
            q = kin.angles(data, info)
        pose_text.set(f'G2 TCP XYZ = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m  |  '
                      f'q (deg) = [{", ".join(f"{v:.1f}" for v in np.degrees(q))}]')

    def update_slider_labels(q):
        suppress_slider[0] = True
        try:
            for i, val in enumerate(q):
                joint_scales[i].set(math.degrees(float(val)))
                joint_numbers[i].set(f'{math.degrees(float(val)):+.2f}°')
        finally:
            suppress_slider[0] = False

    def apply_q(q, announce=False):
        q = kin.clip_angles(model, info, q)
        with viewer.lock():
            kin.set_angles(model, data, info, q)
            actuator_hold(model, data, q)
        update_slider_labels(q)
        if announce:
            info_text.set('已更新关节姿态（运动学预览，不模拟实际速度和力）')
        show_fk()
        return q

    def move_manual(*_):
        if suppress_slider[0]:
            return
        moving['active'] = False
        values = np.radians([s.get() for s in joint_scales])
        apply_q(values)

    for i, (jid, _, _) in enumerate(info):
        lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-math.pi, math.pi)
        line = ttk.Frame(joints_box)
        line.pack(fill='x', pady=2)
        ttk.Label(line, text=f'joint{i+1}', width=9).pack(side='left')
        number = tk.StringVar()
        slider = tk.Scale(line, from_=math.degrees(float(lo)), to=math.degrees(float(hi)),
                          resolution=0.2, showvalue=False, orient='horizontal',
                          length=415, command=move_manual)
        slider.pack(side='left', fill='x', expand=True)
        ttk.Label(line, textvariable=number, width=12).pack(side='left')
        joint_scales.append(slider)
        joint_numbers.append(number)

    def set_goal_for(name):
        if name == 'Manual XYZ':
            return
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            return
        with viewer.lock():
            v = data.site_xpos[sid].copy()
        for field, val in zip(goal_fields, v):
            field.set(f'{val:.5f}')
        set_goal_marker()

    def goal_xyz():
        v = np.array([float(f.get().strip()) for f in goal_fields], dtype=float)
        v[2] += float(offset_field.get())
        if not np.isfinite(v).all():
            raise ValueError('XYZ / offset must be finite numbers')
        return v

    def set_goal_marker():
        try:
            xyz = goal_xyz()
            with viewer.lock():
                model.site_pos[goal_id] = xyz
                mujoco.mj_forward(model, data)
        except (ValueError, TypeError):
            return

    def choose_target(_=None):
        set_goal_for(tk_target.get())

    target_select.bind('<<ComboboxSelected>>', choose_target)

    def ik_go(animated=False):
        try:
            xyz = goal_xyz()
            q_orig = read_current()
            # Solve in separate mjData; never touch the Viewer while iterating.
            tmp = mujoco.MjData(model)
            tmp.qpos[:] = data.qpos.copy()
            kin.set_angles(model, tmp, info, q_orig)
            q_new, err, n_it = kin.solve_ik(model, tmp, info, tcp_id, xyz, tolerance)
            msg = f'IK error: {err*1000:.2f} mm; attempts: {n_it}; target: {np.round(xyz, 3)}'
            info_text.set(msg)
            set_goal_marker()
            if err > tolerance and not allow_closest.get():
                messagebox.showwarning('IK 未收敛', msg + '\n将保持原姿态。可改目标点或勾选最近解预览。')
                return
            if animated:
                moving.update({'active': True, 'start': time.perf_counter(),
                               'from': q_orig.copy(), 'to': q_new.copy(),
                               'duration': 1.8})
            else:
                moving['active'] = False
                apply_q(q_new)
        except Exception as exc:
            messagebox.showerror('IK failed', str(exc))

    ttk.Button(ik_buttons, text='IK 立即到达（预览）', command=lambda: ik_go(False)).pack(side='left', padx=3)
    ttk.Button(ik_buttons, text='IK 平滑运动（预览）', command=lambda: ik_go(True)).pack(side='left', padx=3)
    ttk.Button(ik_buttons, text='显示目标位置', command=set_goal_marker).pack(side='left', padx=3)

    bottom = ttk.Frame(joints_box)
    bottom.pack(fill='x', pady=(10, 2))
    ttk.Button(bottom, text='回到初始姿态', command=lambda: (moving.update(active=False), apply_q(q0, True))).pack(side='left', padx=4)
    ttk.Button(bottom, text='刷新 FK', command=show_fk).pack(side='left', padx=4)

    def save_pose():
        with viewer.lock():
            p, R = kin.tcp_pose(data, tcp_id)
            q = kin.angles(data, info)
        record = {'tcp_xyz_world_m': p.tolist(), 'tcp_R_world': R.tolist(),
                  'joint_angles_rad': q.tolist(), 'joint_angles_deg': np.degrees(q).tolist(),
                  'goal_xyz_world_m': goal_xyz()}
        OUTPUT_POSE.write_text(json.dumps(record, indent=2), encoding='utf-8')
        info_text.set(f'关节与 TCP 位姿已保存：{OUTPUT_POSE.name}')

    ttk.Button(bottom, text='保存当前关节姿态', command=save_pose).pack(side='left', padx=4)

    def camera_switch():
        choice = cam_sel.get()
        with viewer.lock():
            if choice == 'Free camera':
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = [0, 0.2, 0.0]
                viewer.cam.distance = 2.2
                viewer.cam.azimuth = 115
                viewer.cam.elevation = -30
            else:
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, choice)
                if cid >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cid
                else:
                    info_text.set(f'相机未找到：{choice}')

    ttk.Button(camera_box, text='切换视角', command=camera_switch).pack(side='left', padx=6)

    ttk.Label(frame, text='注意：IK 仅约束 XYZ；xArm5 不支持任意 6D 位姿。此版本不做轨迹碰撞检测或实际抓取。',
              wraplength=680, foreground='#795c32').pack(anchor='w', pady=(5, 0))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        update_slider_labels(q0)
        if targets:
            set_goal_for(targets[0])
        camera_switch()
        show_fk()

        def tick():
            if not viewer.is_running():
                root.destroy()
                return
            if moving['active']:
                t = (time.perf_counter() - moving['start']) / moving['duration']
                frac = min(1., max(0., t))
                u = frac*frac*(3 - 2*frac)  # smoothstep
                q = (1-u)*moving['from'] + u*moving['to']
                apply_q(q)
                if frac >= 1.:
                    moving['active'] = False
            viewer.sync()
            root.after(40, tick)

        root.protocol('WM_DELETE_WINDOW', root.destroy)
        tick()
        root.mainloop()


if __name__ == '__main__':
    main()
