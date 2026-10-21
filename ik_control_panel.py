"""Live FK / IK panel for xArm5 + Bio Gripper G2 in MuJoCo.

Uses V16 xarm_description/xarm5_lab_fk_camera.xml created by setup_tcp_cameras.py.
No ROS 2 and no changes to your OT-2 / VersaMax dimensions.

Continuous contact dynamics with actuator targets, frictional grasping, and a
horizontal flange constraint. Motions are not automatically collision planned.

Run: python ik_control_panel.py
"""
from __future__ import annotations

import json
import math
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

import mujoco
import mujoco.viewer
import numpy as np

import robot_kinematics as kin
from grasp_workflow import WorkflowPanel
from keyboard_ee import CartesianJog, KEY_DIRECTIONS
from physics_runtime import PhysicsRuntime, load_physics_model
from state_capture import StateRecorder

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

    model = load_physics_model(SCENE)
    data = mujoco.MjData(model)
    workflow = None
    info = kin.arm_info(model)
    tcp_id = kin.site_id(model, 'g2_tcp')
    goal_id = kin.site_id(model, 'ik_target')

    data.qpos[:] = model.qpos0
    q0 = kin.level_flange(model, info, kin.Q0)
    kin.set_angles(model, data, info, q0)
    actuator_hold(model, data, q0)
    mujoco.mj_forward(model, data)
    physics = PhysicsRuntime(model, data, info)
    jog_data = mujoco.MjData(model)

    targets = []
    for name in DEFAULT_TARGETS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
            targets.append(name)
    if not targets:
        print('WARNING: no named lab targets, use Manual XYZ')

    root = tk.Tk()
    root.title('xArm5 / G2 TCP - Contact Physics')
    root.geometry('760x1020')
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

    keyboard_box = ttk.LabelFrame(frame, text='键盘移动 EE（机器人底座 / 世界坐标）', padding=8)
    keyboard_box.pack(fill='x', pady=5)
    keyboard_speed = tk.StringVar(value='0.03')
    ttk.Label(keyboard_box, text='速度 (m/s)').pack(side='left')
    ttk.Entry(keyboard_box, textvariable=keyboard_speed, width=7).pack(side='left', padx=5)
    keyboard_pad = tk.Label(keyboard_box, takefocus=True, relief='groove', padx=10, pady=8,
                            text='点击这里启用：A/D −X/+X · W/S +Y/−Y · Ctrl/Space −Z/+Z\n按住 Q 渐闭 · E 渐开；松键停止；Esc 退出', bg='#eeeeee')
    keyboard_pad.pack(side='left', fill='x', expand=True)
    jog = CartesianJog(model, info, tcp_id, level=True)
    pressed_keys = set()
    keyboard_clock = [time.perf_counter()]

    def keyboard_stop(_=None):
        pressed_keys.clear()
        keyboard_pad.configure(bg='#eeeeee')

    def keyboard_enable(_=None):
        pause_manual()
        keyboard_clock[0] = time.perf_counter()
        keyboard_pad.focus_set()
        keyboard_pad.configure(bg='#d9efda')

    def keyboard_press(event):
        key = event.keysym.lower()
        if key == 'escape':
            keyboard_stop()
            root.focus_set()
            return 'break'
        if key in ('q', 'e'):
            if key not in pressed_keys:
                moving['active'] = False
                workflow.stop()
                pressed_keys.add(key)
            return 'break'
        if key in KEY_DIRECTIONS:
            moving['active'] = False
            workflow.stop()
            pressed_keys.add(key)
            return 'break'

    def keyboard_release(event):
        pressed_keys.discard(event.keysym.lower())
        if event.keysym.lower() in KEY_DIRECTIONS or event.keysym.lower() in ('q', 'e'):
            return 'break'

    keyboard_pad.bind('<Button-1>', keyboard_enable)
    keyboard_pad.bind('<KeyPress>', keyboard_press)
    keyboard_pad.bind('<KeyRelease>', keyboard_release)
    keyboard_pad.bind('<FocusOut>', keyboard_stop)

    joints_box = ttk.LabelFrame(frame, text='正运动学 FK：直接拖动五个关节（度）', padding=9)
    joints_box.pack(fill='x', pady=10)
    joint_values = []
    joint_numbers = []
    moving = {'active': False, 'start': 0.0, 'from': None, 'to': None,
              'duration': 1.8}

    info_text = tk.StringVar(value='准备就绪。选择目标并点击 IK 按钮。')
    pose_text = tk.StringVar(value='')
    status_label = ttk.Label(frame, textvariable=info_text, wraplength=670,
                             foreground='#12578a')
    status_label.pack(fill='x', pady=(4, 2))
    ttk.Label(frame, textvariable=pose_text, font=('Consolas', 10),
              wraplength=670).pack(fill='x', pady=(0, 8))
    physics_text = tk.StringVar(value='物理仿真运行中')
    ttk.Label(frame, textvariable=physics_text, wraplength=670).pack(fill='x')

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
        # Updating the variable avoids Scale.set's deferred command callback.
        for value, number, angle in zip(joint_values, joint_numbers, q):
            degrees = math.degrees(float(angle))
            value.set(degrees)
            number.set(f'{degrees:+.2f}°')

    def apply_q(q, announce=False):
        q = kin.level_flange(model, info, q)
        with viewer.lock():
            physics.set_arm_target(q)
        update_slider_labels(q)
        if announce:
            info_text.set('已设置关节目标，机械臂由驱动器逐渐移动；碰撞可能阻挡运动')
        show_fk()
        return q

    def move_manual(*_):
        pause_manual()
        values = np.radians([v.get() for v in joint_values])
        try:
            apply_q(values)
        except ValueError as exc:
            update_slider_labels(read_current())
            info_text.set(str(exc))

    def cancel_motion():
        keyboard_stop()
        moving['active'] = False

    def pause_manual():
        cancel_motion()
        if workflow is not None:
            workflow.stop()

    def jog_gripper(direction, dt):
        with viewer.lock():
            held = workflow.controller.jog_fingers(direction, dt)
        info_text.set('检测到双侧夹持接触；请缓慢抬起确认。' if held else
                      '夹爪目标逐渐闭合；靠接触与摩擦抓取。' if direction > 0 else
                      '夹爪目标逐渐张开；反应盘受重力与接触作用。')

    for i, (jid, _, _) in enumerate(info):
        lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-math.pi, math.pi)
        line = ttk.Frame(joints_box)
        line.pack(fill='x', pady=2)
        ttk.Label(line, text=f'joint{i+1}', width=9).pack(side='left')
        number = tk.StringVar()
        value = tk.DoubleVar(value=math.degrees(float(q0[i])))
        slider = tk.Scale(line, from_=math.degrees(float(lo)), to=math.degrees(float(hi)),
                          resolution=0.2, showvalue=False, orient='horizontal',
                          length=415, variable=value, command=move_manual)
        slider.pack(side='left', fill='x', expand=True)
        if i == 3:
            slider.configure(state='disabled')
        ttk.Label(line, textvariable=number, width=12).pack(side='left')
        joint_values.append(value)
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

    def ik_go(animated=False, align_plate=False):
        try:
            pause_manual()
            xyz = goal_xyz()
            q_orig = read_current()
            # Solve in separate mjData; never touch the Viewer while iterating.
            tmp = mujoco.MjData(model)
            with viewer.lock():
                tmp.qpos[:] = data.qpos
                plate_rot = data.xmat[physics.controller.plate].reshape(3, 3).copy()
            kin.set_angles(model, tmp, info, q_orig)
            yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) if align_plate else None
            q_new, err, n_it = kin.solve_ik(model, tmp, info, tcp_id, xyz, tolerance,
                                          level=True, flange_yaw=yaw)
            msg = f'IK error: {err*1000:.2f} mm; iterations: {n_it}; target: {np.round(xyz, 3)}'
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

    ttk.Button(ik_buttons, text='IK 设置驱动目标', command=lambda: ik_go(False)).pack(side='left', padx=3)
    ttk.Button(ik_buttons, text='IK 平滑移动', command=lambda: ik_go(True)).pack(side='left', padx=3)
    ttk.Button(ik_buttons, text='IK 对齐反应盘', command=lambda: ik_go(True, True)).pack(side='left', padx=3)
    ttk.Button(ik_buttons, text='显示目标位置', command=set_goal_marker).pack(side='left', padx=3)

    bottom = ttk.Frame(joints_box)
    bottom.pack(fill='x', pady=(10, 2))
    def return_initial():
        pause_manual()
        apply_q(q0, True)

    ttk.Button(bottom, text='回到初始姿态', command=return_initial).pack(side='left', padx=4)
    ttk.Button(bottom, text='刷新 FK', command=show_fk).pack(side='left', padx=4)

    def save_pose():
        with viewer.lock():
            p, R = kin.tcp_pose(data, tcp_id)
            q = kin.angles(data, info)
        record = {'tcp_xyz_world_m': p.tolist(), 'tcp_R_world': R.tolist(),
                  'joint_angles_rad': q.tolist(), 'joint_angles_deg': np.degrees(q).tolist(),
                  'goal_xyz_world_m': goal_xyz().tolist()}
        OUTPUT_POSE.write_text(json.dumps(record, indent=2), encoding='utf-8')
        info_text.set(f'关节与 TCP 位姿已保存：{OUTPUT_POSE.name}')

    ttk.Button(bottom, text='保存当前关节姿态', command=save_pose).pack(side='left', padx=4)

    recorder = StateRecorder(ROOT / 'outputs' / 'state_captures' /
                             f'states_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl')
    capture_box = ttk.LabelFrame(joints_box, text='自由流程：按顺序采集实际状态', padding=6)
    capture_box.pack(fill='x', pady=(8, 0))
    capture_name = tk.StringVar(value='step_001')
    capture_status = tk.StringVar(value='未采集；可输入步骤名称，记录 XYZ、关节角和夹爪开度。')
    ttk.Entry(capture_box, textvariable=capture_name, width=23).pack(side='left', padx=4)

    def collect_state():
        try:
            with viewer.lock():
                record = recorder.append(physics, tcp_id, capture_name.get())
        except (OSError, ValueError) as exc:
            info_text.set(f'采集失败：{exc}')
            return
        capture_status.set(f'已采集 {recorder.count} 个状态；最新：{record["name"]}')
        info_text.set(f'状态已追加保存：{recorder.path}')
        capture_name.set(f'step_{recorder.count+1:03d}')

    ttk.Button(capture_box, text='采集当前状态', command=collect_state).pack(side='left', padx=4)
    ttk.Label(capture_box, textvariable=capture_status, wraplength=270).pack(side='left', padx=4)

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

    ttk.Label(frame, text='接触物理已启用：重力、摩擦与夹爪驱动；法兰水平约束开启，joint4 自动补偿。双侧接触仅表示夹持候选，请缓慢抬起确认；没有自动避障。',
              wraplength=680, foreground='#795c32').pack(anchor='w', pady=(5, 0))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        def prepare_workflow_goal(name, offset):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
                raise ValueError(f'Unknown target: {name}')
            tk_target.set(name)
            offset_field.set(str(offset))
            set_goal_for(name)

        workflow = WorkflowPanel(root, model, data, info, tcp_id, viewer,
                                    apply_q, prepare_workflow_goal,
                                    cancel_motion, controller=physics.controller)
        ttk.Button(camera_box, text='抓取流程示教', command=workflow.open).pack(side='left', padx=6)
        ttk.Button(camera_box, text='重置仿真', command=workflow.reset).pack(side='left', padx=6)
        update_slider_labels(q0)
        if targets:
            set_goal_for(targets[0])
        camera_switch()
        show_fk()

        def tick():
            if not viewer.is_running():
                root.destroy()
                return
            now = time.perf_counter()
            dt = now - keyboard_clock[0]
            keyboard_clock[0] = now
            if root.focus_get() == keyboard_pad:
                grip_direction = int('q' in pressed_keys) - int('e' in pressed_keys)
                if grip_direction:
                    jog_gripper(grip_direction, dt)
            if pressed_keys.intersection(KEY_DIRECTIONS) and root.focus_get() == keyboard_pad:
                try:
                    with viewer.lock():
                        jog_data.qpos[:] = data.qpos
                        kin.set_angles(model, jog_data, info, physics.arm_target)
                        q, target, error = jog.solve(jog_data, pressed_keys, float(keyboard_speed.get()), dt)
                    if q is not None:
                        apply_q(q)
                        tk_target.set('Manual XYZ')
                        offset_field.set('0')
                        for field, value in zip(goal_fields, target):
                            field.set(f'{value:.5f}')
                        set_goal_marker()
                        info_text.set('键盘移动 EE；松开停止。法兰保持水平，允许绕竖直轴转动。')
                    elif target is not None:
                        info_text.set('该方向不可达或关节变化过大，已保持当前位置；请换方向或调整关节姿态。')
                except (ValueError, TypeError) as exc:
                    keyboard_stop()
                    info_text.set(str(exc))
            if moving['active']:
                t = (time.perf_counter() - moving['start']) / moving['duration']
                frac = min(1., max(0., t))
                u = frac*frac*(3 - 2*frac)  # smoothstep
                q = (1-u)*moving['from'] + u*moving['to']
                apply_q(q)
                if frac >= 1.:
                    moving['active'] = False
            workflow.tick()
            with viewer.lock():
                physics.advance(dt)
                held = physics.controller.held
                q_actual = kin.angles(data, info)
                tracking_error = np.degrees(np.max(np.abs(q_actual-physics.arm_target)))
                flange = model.body('link_eef').id
                normal = data.xmat[flange].reshape(3, 3)[:, 2]
                tilt = np.degrees(np.arccos(np.clip(-normal[2], -1, 1)))
                sim_time = data.time
            physics_text.set(f'物理时间 {sim_time:.2f}s | 双侧夹持接触：{"有" if held else "无"} | '
                             f'法兰倾角 {tilt:.3f}° | 关节跟踪误差 {tracking_error:.2f}°'
                             + (f'\n碰撞阻挡，已停止推进：{physics.blocked}；请反方向退回。' if physics.blocked else ''))
            show_fk()
            viewer.sync()
            root.after(40, tick)

        root.protocol('WM_DELETE_WINDOW', root.destroy)
        tick()
        root.mainloop()


if __name__ == '__main__':
    main()
