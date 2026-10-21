"""User-taught pick/place using contact physics."""
from __future__ import annotations

import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import mujoco
import numpy as np
import robot_kinematics as kin

CONFIG = Path(__file__).resolve().parent / 'grasp_workflow.json'
DESTINATIONS = ('versamax_load', 'ot2_waiting', 'ot2_stack_top')
WORKFLOW_ACTIONS = ('move', 'grasp', 'move', 'move', 'release', 'move')




def validate_steps(model, info, steps):
    if not isinstance(steps, list) or not steps:
        raise ValueError('流程不能为空')
    held = False
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get('name'), str):
            raise ValueError('每个步骤必须有名称和动作参数')
        q = np.asarray(step['joints_rad'], float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError(f"{step['name']} 尚未示教或关节数值无效")
        if not np.allclose(q, kin.clip_angles(model, info, q), atol=1e-6):
            raise ValueError(f"{step['name']} 超出关节范围")
        duration = step['duration_s']
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError('步骤时间必须为数字')
        if not np.isfinite(duration) or duration < 0.2:
            raise ValueError('步骤时间必须至少 0.2 秒')
        action = step['action']
        if action == 'grasp':
            if held:
                raise ValueError('不能重复抓取')
            held = True
        elif action == 'release':
            if not held:
                raise ValueError('必须先抓取再放开')
            held = False
        elif action != 'move':
            raise ValueError(f'未知动作 {action}')
    if held:
        raise ValueError('流程必须包含放开步骤')


def validate_record(model, info, record):
    if record.get('version') != 1:
        raise ValueError('不支持的流程版本')
    validate_steps(model, info, record['steps'])
    if tuple(s['action'] for s in record['steps']) != WORKFLOW_ACTIONS:
        raise ValueError('需要六步取放板流程')
    clearance = float(record['clearance_m'])
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError('上方高度必须为正数')
    kin.site_id(model, record['destination'])


class WorkflowPanel:
    def __init__(self, parent, model, data, info, tcp, viewer, apply_q, prepare_goal, cancel_motion, controller):
        self.model, self.data, self.info = model, data, info
        self.viewer, self.apply_q = viewer, apply_q
        self.prepare_goal, self.cancel_motion = prepare_goal, cancel_motion
        self.controller = controller
        self.window = None
        self.running = False
        self.index = 0
        self.segment = None
        self.action_started = None
        self.continuous = False
        self.status = tk.StringVar(parent, value='选择步骤 → 设置参考目标 → 主面板调整 IK / FK → 记录姿态')
        self.destination = tk.StringVar(parent, value='versamax_load')
        self.duration = tk.StringVar(parent, value='2.0')
        self.clearance = tk.StringVar(parent, value='0.10')
        self.steps = [dict(name=name, action=action, joints_rad=None, duration_s=2.0)
                      for name, action in [('取板上方', 'move'), ('抓取位置', 'grasp'),
                                           ('抬起', 'move'), ('放板上方', 'move'),
                                           ('放板位置', 'release'), ('撤离', 'move')]]

    def open(self):
        if self.window is not None and self.window.winfo_exists():
            self.window.lift()
            return
        w = self.window = tk.Toplevel()
        w.title('抓取流程示教（接触物理）')
        w.geometry('690x520')
        f = ttk.Frame(w, padding=12)
        f.pack(fill='both', expand=True)
        ttk.Label(f, text='按顺序示教六个步骤；抓取 / 放开动作在该步到达后执行。').pack(anchor='w')
        line = ttk.Frame(f)
        line.pack(fill='x', pady=8)
        ttk.Label(line, text='放板参考位置').pack(side='left')
        ttk.Combobox(line, textvariable=self.destination, values=DESTINATIONS, width=20).pack(side='left')
        ttk.Label(line, text='上方高度(m)').pack(side='left', padx=5)
        ttk.Entry(line, textvariable=self.clearance, width=8).pack(side='left')
        self.listbox = tk.Listbox(f, height=7, exportselection=False)
        self.listbox.pack(fill='x')
        self.listbox.bind('<<ListboxSelect>>', self.selected)
        line = ttk.Frame(f)
        line.pack(fill='x', pady=8)
        ttk.Label(line, text='本步时间(s)').pack(side='left')
        ttk.Entry(line, textvariable=self.duration, width=8).pack(side='left')
        for text, fn in [('设置参考目标', self.reference), ('记录当前关节姿态', self.capture)]:
            ttk.Button(line, text=text, command=lambda fn=fn: self.guard(fn)).pack(side='left', padx=4)
        for commands in [[('单步执行', lambda: self.start(False)), ('连续执行', lambda: self.start(True)), ('暂停', self.stop), ('重置', self.reset)],
                         [('保存流程', self.save), ('加载流程', self.load)]]:
            row = ttk.Frame(f)
            row.pack(fill='x', pady=4)
            for text, fn in commands:
                ttk.Button(row, text=text, command=lambda fn=fn: self.guard(fn)).pack(side='left', padx=3)
        ttk.Label(f, textvariable=self.status, wraplength=650).pack(fill='x', pady=10)
        ttk.Label(f, text=('参考目标仅用于定位：先在主面板对齐反应盘，再记录实际到达姿态。\n抓取依靠接触与摩擦，失败会停止；释放后受重力下落。没有自动避障。'), wraplength=650).pack(fill='x')
        w.protocol('WM_DELETE_WINDOW', lambda: (self.stop(), w.destroy()))
        self.refresh()

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.stop()
            self.status.set(str(exc))
            messagebox.showerror('抓取流程', str(exc), parent=self.window)

    def refresh(self):
        selected = self.listbox.curselection()
        self.listbox.delete(0, 'end')
        for i, s in enumerate(self.steps):
            self.listbox.insert('end', f"{i+1}. {s['name']}  [{'已示教' if s['joints_rad'] is not None else '待记录'}]  {s['duration_s']:.1f}s")
        self.listbox.selection_set(selected[0] if selected else 0)

    def selection(self):
        selected = self.listbox.curselection()
        return int(selected[0]) if selected else 0

    def selected(self, _=None):
        self.duration.set(str(self.steps[self.selection()]['duration_s']))

    def reference(self):
        self.stop()
        self.cancel_motion()
        i = self.selection()
        z = float(self.clearance.get())
        if not np.isfinite(z) or z <= 0:
            raise ValueError('上方高度必须为正数')
        self.prepare_goal('ot2_pickup' if i < 3 else self.destination.get(), z if i in (0, 2, 3, 5) else 0)
        self.status.set('参考目标已写入主面板。执行 IK 并调整关节朝向，然后记录姿态。')

    def capture(self):
        self.stop()
        self.cancel_motion()
        duration = float(self.duration.get())
        if not np.isfinite(duration) or duration < .2:
            raise ValueError('时间必须至少 0.2 秒')
        with self.viewer.lock():
            q = kin.angles(self.data, self.info).tolist()
        q = kin.level_flange(self.model, self.info, q).tolist()
        self.steps[self.selection()].update(joints_rad=q, duration_s=duration)
        self.refresh()
        self.status.set('已记录；执行顺序由列表决定，单步从上次完成的位置继续。')

    def start(self, continuous):
        if self.running:
            return
        validate_steps(self.model, self.info, self.steps)
        for step in self.steps:
            q = np.asarray(step['joints_rad'], float)
            if not np.allclose(q, kin.level_flange(self.model, self.info, q), atol=1e-6):
                raise ValueError(f"{step['name']} 的法兰不水平，请在水平锁定面板中重新示教")
        if self.index >= len(self.steps):
            raise ValueError('流程已完成；请重置后再次运行')
        self.cancel_motion()
        self.continuous = continuous
        self.running = True
        self.segment = None

    def stop(self):
        self.running = False
        self.segment = None
        self.action_started = None

    def reset(self):
        self.stop()
        self.cancel_motion()
        with self.viewer.lock():
            self.controller.reset()
            q = kin.angles(self.data, self.info).copy()
        self.apply_q(q)
        self.index = 0
        self.status.set('已还原机械臂和微孔板，下一步为取板上方。')

    def tick(self):
        if not self.running:
            return
        self.guard(self.advance)

    def advance(self):
        s = self.steps[self.index]
        if self.segment is None:
            with self.viewer.lock():
                q = kin.angles(self.data, self.info).copy()
            self.segment = (time.perf_counter(), q)
            self.action_started = None
        start, q = self.segment
        t = min(1., (time.perf_counter() - start) / s['duration_s'])
        u = t*t*(3-2*t)
        self.apply_q((1-u)*q + u*np.asarray(s['joints_rad']))
        self.status.set(f"执行 {self.index+1}/{len(self.steps)}：{s['name']} ({t:.0%})")
        if t < 1:
            return
        with self.viewer.lock():
            error = np.max(np.abs(kin.angles(self.data, self.info)-s['joints_rad']))
        if error > .015:
            if time.perf_counter()-start > s['duration_s']+6:
                raise ValueError('关节尚未到位，可能受碰撞阻挡；流程已停止')
            self.status.set('等待机械臂实际到位…')
            return
        if s['action'] in ('grasp', 'release'):
            if self.action_started is None:
                with self.viewer.lock():
                    if s['action'] == 'grasp':
                        self.controller.attach()
                    else:
                        self.controller.release()
                self.action_started = time.perf_counter()
            elapsed = time.perf_counter()-self.action_started
            self.status.set('等待夹爪接触确认…' if s['action'] == 'grasp' else '等待夹爪张开…')
            if elapsed < 2.5:
                return
            with self.viewer.lock():
                held = self.controller.held
            if s['action'] == 'grasp' and not held:
                raise ValueError('未检测到双侧有效夹持；请检查夹爪朝向和抓取高度')
            if s['action'] == 'release' and held:
                raise ValueError('反应盘仍接触双侧夹指，请检查释放空间')
        self.index += 1
        self.segment = None
        if not self.continuous or self.index == len(self.steps):
            self.stop()
            self.status.set(('流程完成，反应盘继续受重力与接触作用。') if self.index == len(self.steps) else '本步完成，可继续单步或连续执行。')

    def save(self):
        record = dict(version=1, destination=self.destination.get(),
                      clearance_m=float(self.clearance.get()), steps=self.steps)
        validate_record(self.model, self.info, record)
        CONFIG.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        self.status.set(f'已保存到 {CONFIG.name}')

    def load(self):
        record = json.loads(CONFIG.read_text(encoding='utf-8'))
        validate_record(self.model, self.info, record)
        self.reset()
        self.steps = record['steps']
        self.destination.set(record['destination'])
        self.clearance.set(str(record['clearance_m']))
        self.refresh()
        self.status.set('已加载流程；运行前检查场景位置是否与示教时一致。')
