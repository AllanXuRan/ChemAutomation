"""Live position/rotation editor for V15's three cameras.

  python camera_editor.py

World camera XYZ use xArm base coordinates. Wrist camera XYZ use G2 BASE frame.
Select camera -> edit 6 coordinates + FOV -> Preview -> Save and rebuild.
Supports a kinematic demo oscillation to check wrist camera attachment.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

import setup_tcp_cameras as base


def run():
    import tkinter as tk
    from tkinter import ttk, messagebox
    from mujoco import viewer as mj_viewer

    scene = base.OUTPUT
    if not scene.is_file() or not base.CONFIG.is_file():
        raise SystemExit('First run: python setup_tcp_cameras.py')
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    base.set_arm_pose(model, data)
    _, gripper_p, gripper_R = base.gripper_frame_in_parent(model, data)
    cfg = json.loads(base.CONFIG.read_text(encoding='utf-8'))
    base.validate_config(cfg)
    cameras = cfg['cameras']

    root = tk.Tk()
    root.title('V16 - Three physical cameras | World + G2 wrist')
    root.geometry('740x540')
    outer = ttk.Frame(root, padding=15)
    outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='三个实体相机：全局俯视 / G2 随动 / 反应板俯视',
              font=('Segoe UI', 12, 'bold')).pack(anchor='w')
    ttk.Label(outer, text='世界相机用机器人底座 XYZ；随动相机用 G2 夹爪局部 XYZ（单位：m）').pack(anchor='w', pady=6)
    header = ttk.Frame(outer)
    header.pack(fill='x', pady=7)
    ttk.Label(header, text='Camera:').pack(side='left')
    active = tk.StringVar(value=base.CAMERA_NAMES[0])
    box = ttk.Combobox(header, textvariable=active, values=base.CAMERA_NAMES, state='readonly', width=28)
    box.pack(side='left', padx=10)
    mode_text = tk.StringVar()
    ttk.Label(header, textvariable=mode_text, foreground='#135F83').pack(side='left')

    keys = ['px', 'py', 'pz', 'tx', 'ty', 'tz', 'fovy']
    labels = ['Camera X (m)', 'Camera Y (m)', 'Camera Z (m)',
              'Look-at X (m)', 'Look-at Y (m)', 'Look-at Z (m)', 'Vertical FOV (deg)']
    fields = {name: tk.StringVar() for name in keys}
    for key, label in zip(keys, labels):
        line = ttk.Frame(outer)
        line.pack(fill='x', pady=5)
        ttk.Label(line, text=label, width=23).pack(side='left')
        ttk.Entry(line, width=19, textvariable=fields[key]).pack(side='left')

    status = tk.StringVar(value='Select camera and press Preview. Save persists values to JSON and rebuilds scene.')
    ttk.Label(outer, textvariable=status, foreground='#155C88', wraplength=690).pack(anchor='w', pady=10)
    buttons = ttk.Frame(outer)
    buttons.pack(fill='x', pady=10)
    paused_motion = [False]

    def populate(name):
        conf = cameras[name]
        p, t = conf['position_xyz_m'], conf['look_at_xyz_m']
        vals = [*p, *t, conf['fovy_deg']]
        for key, value in zip(keys, vals):
            fields[key].set(f'{float(value):.6f}')
        mode_text.set('G2 LOCAL FRAME' if name == 'wrist_gripper' else 'WORLD FRAME')

    def read_form():
        vals = [float(fields[key].get()) for key in keys]
        name = active.get()
        return {'frame': 'gripper' if name == 'wrist_gripper' else 'world',
                'position_xyz_m': vals[:3], 'look_at_xyz_m': vals[3:6],
                'fovy_deg': vals[6]}

    def apply_to_live(name, conf):
        base.validate_config({'cameras': {**cameras, name: conf}})
        cid = base.find_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        pos, rot = base.get_camera_pose_in_parent(name, conf, gripper_p, gripper_R)
        model.cam_pos[cid, :] = pos
        model.cam_quat[cid, :] = base.quat_of(rot)
        model.cam_fovy[cid] = float(conf['fovy_deg'])
        # Move visible camera hardware together with its optical camera.
        parts = {
            'housing': (0.0, 0.0, 0.041),
            'lens': (0.0, 0.0, 0.010),
            'indicator': (0.022, 0.019, 0.012),
        }
        for suffix, offset in parts.items():
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f'cam_{name}_{suffix}')
            if gid >= 0:
                model.geom_pos[gid] = pos + rot @ np.asarray(offset, float)
                model.geom_quat[gid] = base.quat_of(rot)
        mujoco.mj_forward(model, data)
        return cid

    with mj_viewer.launch_passive(model, data) as viewer:
        previously_active = [active.get()]
        populate(active.get())

        def apply_form():
            name = active.get()
            conf = read_form()
            with viewer.lock():
                cid = apply_to_live(name, conf)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cid
            cameras[name] = conf
            viewer.sync()
            status.set(f'Preview: {name}, frame={conf["frame"]}, xyz={conf["position_xyz_m"]}')

        def preview():
            try:
                apply_form()
            except Exception as e:
                messagebox.showerror('Preview failed', str(e))

        def change_camera(event=None):
            old = previously_active[0]
            # Selection changed; current inputs still belong to OLD camera.
            try:
                vals = [float(fields[key].get()) for key in keys]
                cameras[old] = {'frame': 'gripper' if old == 'wrist_gripper' else 'world',
                                'position_xyz_m': vals[:3], 'look_at_xyz_m': vals[3:6],
                                'fovy_deg': vals[6]}
            except ValueError:
                pass
            previously_active[0] = active.get()
            populate(active.get())
            preview()

        box.bind('<<ComboboxSelected>>', change_camera)

        def save():
            try:
                apply_form()
                base.validate_config(cfg)
                base.CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
                # Rebuild from unmodified physics XML so no camera/site duplication.
                base.build(base.SOURCE, base.OUTPUT, base.CONFIG)
                base.report(base.OUTPUT)
                messagebox.showinfo('Saved', f'Config: {base.CONFIG}\nScene: {base.OUTPUT}\nCalibration snapshot: {base.CALIBRATION}')
            except Exception as e:
                messagebox.showerror('Save failed', str(e))

        def motion_toggle():
            paused_motion[0] = not paused_motion[0]
            status.set('Joint2 demonstration enabled. Wrist camera moves rigidly with G2.' if paused_motion[0]
                       else 'Joint2 demonstration paused.')

        ttk.Button(buttons, text='Preview current camera', command=preview).pack(side='left', padx=4)
        ttk.Button(buttons, text='Save + rebuild MJCF', command=save).pack(side='left', padx=4)
        ttk.Button(buttons, text='Toggle Joint2 motion demo', command=motion_toggle).pack(side='left', padx=4)
        ttk.Label(outer, text='相机外壳、镜头和光轴会跟随相机参数一起移动。Esc 切回自由视角。',
                  wraplength=690).pack(anchor='w')

        arm2 = base.find_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')
        qadr = int(model.jnt_qposadr[arm2])
        q_reference = float(data.qpos[qadr])
        t0 = time.perf_counter()

        def tick():
            if not viewer.is_running():
                root.destroy()
                return
            if paused_motion[0]:
                q = q_reference + 0.18 * math.sin(0.7 * (time.perf_counter() - t0))
                if model.jnt_limited[arm2]:
                    q = np.clip(q, *model.jnt_range[arm2])
                with viewer.lock():
                    data.qpos[qadr] = float(q)
                    mujoco.mj_forward(model, data)
            viewer.sync()
            root.after(70, tick)

        def close_window():
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', close_window)
        preview()
        tick()
        root.mainloop()


if __name__ == '__main__':
    run()
