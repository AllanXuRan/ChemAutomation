"""Optional 2D OT-2/VersaMax layout editor (standard-library Tkinter only).

Mouse: drag an instrument to relocate; shift+mouse wheel rotates the selected
instrument. The right side lets you set exact X/Y/Yaw and OT-2 deck slots.
Save -> writes lab_layout.json. Rebuild -> regenerates the MuJoCo scene.
"""
from __future__ import annotations
import json
import math
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from build_lab_scene import (facing_yaw_deg, canonicalize_facing, rotate_xy,
    apply_geometry_defaults, validate_geometry, versamax_local_regions, opening_side,
    ot2_slot_xy, table_geometry)

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'lab_layout.json'
SCALE=290
CW,CH=820,600
CX,CY=410,345


def xy_screen(xy):
    return (CX+float(xy[0])*SCALE,CY-float(xy[1])*SCALE)


def rotate(xy,deg):
    return rotate_xy(xy, deg)


def corners(cen,w,d,deg):
    out=[]
    for x,y in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]:
        xx,yy=rotate((x,y),deg)
        out.extend(xy_screen((cen[0]+xx,cen[1]+yy)))
    return out


class LayoutEditor:
    def __init__(self,root):
        self.root=root
        self.data=json.loads(CONFIG.read_text(encoding='utf-8'))
        apply_geometry_defaults(self.data)
        canonicalize_facing(self.data)
        self.sel=tk.StringVar(value='ot2')
        self.drag=None
        self.detail=[]
        root.title('Robot-centred OT-2 / VersaMax layout editor')
        root.geometry('1150x850')
        wrapper=ttk.Frame(root,padding=10)
        wrapper.pack(fill='both',expand=True)
        self.cv=tk.Canvas(wrapper,width=CW,height=CH,bg='#eef2f7',highlightthickness=1,
                          highlightbackground='#94a3b8')
        self.cv.grid(row=0,column=0,sticky='nsew')
        # Scrollable control pane: additional calibration controls never hide Save/Rebuild.
        rightwrap=ttk.Frame(wrapper)
        rightwrap.grid(row=0,column=1,sticky='nsew',padx=5)
        self.right_canvas=tk.Canvas(rightwrap,width=300,height=760,highlightthickness=0)
        self.right_canvas.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(rightwrap,orient='vertical',command=self.right_canvas.yview)
        scrollbar.pack(side='right',fill='y')
        self.right_canvas.configure(yscrollcommand=scrollbar.set)
        right=ttk.Frame(self.right_canvas,padding=(12,2,2,2))
        self.right_window=self.right_canvas.create_window((0,0),window=right,anchor='nw')
        right.bind('<Configure>',lambda e:self.right_canvas.configure(scrollregion=self.right_canvas.bbox('all')))
        self.right_canvas.bind('<Configure>',lambda e:self.right_canvas.itemconfigure(self.right_window,width=e.width))
        # On Windows: wheel scroll when mouse is over the panel or its children.
        def scroll_right(e):
            over=self.root.winfo_containing(*self.root.winfo_pointerxy())
            node=over
            while node is not None:
                if node==self.right_canvas:
                    self.right_canvas.yview_scroll(-int(e.delta/120) if e.delta else 0,'units')
                    break
                node=getattr(node,'master',None)
        self.root.bind_all('<MouseWheel>',scroll_right,add='+')
        wrapper.rowconfigure(0,weight=1);wrapper.columnconfigure(0,weight=1)
        ttk.Label(right,text='选择要移动的设备',font=('Segoe UI',12,'bold')).pack(anchor='w',pady=(0,8))
        for key,title in [('ot2','OT-2'),('versamax','VersaMax')]:
            ttk.Radiobutton(right,text=title,variable=self.sel,value=key,command=self.reset_fields).pack(anchor='w')
        ttk.Separator(right).pack(fill='x',pady=9)
        self.fields={}
        self.yaw_entry=None
        for key,txt in [('x','X (m)'),('y','Y (m)'),('yaw','旋转角度 (deg)')]:
            frame=ttk.Frame(right)
            frame.pack(fill='x',pady=3)
            ttk.Label(frame,text=txt,width=15).pack(side='left')
            v=tk.StringVar()
            ent=ttk.Entry(frame,textvariable=v,width=11)
            ent.pack(side='left')
            ent.bind('<Return>',lambda e:self.apply_fields())
            self.fields[key]=v
            if key == 'yaw':
                self.yaw_entry = ent
        self.auto_face = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text='开口始终指向机器人 (自动角度)',
                        variable=self.auto_face, command=self.set_orientation_mode).pack(anchor='w', pady=(5,2))
        ttk.Button(right,text='应用 XYZ / 旋转',command=self.apply_fields).pack(fill='x',pady=4)
        ttk.Button(right,text='两台设备开口朝向机器人',command=self.face_robot).pack(fill='x',pady=4)
        ttk.Label(right,text='温控模块旋转 (度)',font=('Segoe UI',10,'bold')).pack(anchor='w',pady=(7,3))
        self.temp_angles={}
        for key,title in [
            ('temperature_yaw_deg','整个温控模块'),
            ('temperature_base_yaw_deg','下方底座'),
            ('temperature_block_yaw_deg','上方铝板')
        ]:
            fr=ttk.Frame(right);fr.pack(fill='x',pady=2)
            ttk.Label(fr,text=title,width=16).pack(side='left')
            v=tk.StringVar(value=str(self.data['ot2'].get(key,90.0 if key=='temperature_base_yaw_deg' else 0.0)))
            ent=ttk.Entry(fr,width=9,textvariable=v);ent.pack(side='left')
            ent.bind('<Return>',lambda e:self.apply_temperature_angles())
            self.temp_angles[key]=v
        ttk.Button(right,text='应用温控板角度',command=self.apply_temperature_angles).pack(fill='x',pady=4)
        ttk.Label(right,text='温控模块自由位置 (OT-2局部坐标，m)',
                  font=('Segoe UI',10,'bold')).pack(anchor='w',pady=(8,3))
        self.temperature_offset = {}
        for i,axis in enumerate(('X','Y','Z')):
            fr=ttk.Frame(right);fr.pack(fill='x',pady=2)
            ttk.Label(fr,text=f'温控模块 {axis} 位置',width=16).pack(side='left')
            var=tk.StringVar(value=str(self.data['ot2']['temperature_local_xyz_m'][i]))
            ent=ttk.Entry(fr,width=9,textvariable=var);ent.pack(side='left')
            ent.bind('<Return>',lambda e:self.apply_temperature_offset())
            self.temperature_offset[axis]=var
        ttk.Button(right,text='应用模块自由 XYZ',
                   command=self.apply_temperature_offset).pack(fill='x',pady=4)
        ttk.Label(right,text='等待区/堆叠区位置（仅用编号定位）',font=('Segoe UI',10,'bold')).pack(anchor='w',pady=(8,3))
        self.slots={}
        for key,text,values in [
            ('waiting_slot','返回等待槽位',tuple(range(1,12))),
            ('stack_slot','堆叠槽位',tuple(range(1,12)))]:
            fr=ttk.Frame(right);fr.pack(fill='x',pady=3)
            ttk.Label(fr,text=text,width=16).pack(side='left')
            v=tk.StringVar(value=str(self.data['ot2'][key]))
            box=ttk.Combobox(fr,width=7,textvariable=v,values=values,state='readonly')
            box.pack(side='left')
            box.bind('<<ComboboxSelected>>',lambda e:self.apply_slots())
            self.slots[key]=v
        fr=ttk.Frame(right);fr.pack(fill='x',pady=4)
        ttk.Label(fr,text='板堆叠数量',width=16).pack(side='left')
        self.stack_var=tk.StringVar(value=str(self.data['ot2']['stack_count']))
        ent=ttk.Entry(fr,width=10,textvariable=self.stack_var);ent.pack(side='left')
        ent.bind('<Return>',lambda e:self.apply_slots())
        ttk.Button(right,text='应用槽位 / 板数量',command=self.apply_slots).pack(fill='x',pady=6)
        ttk.Button(right,text='编辑仪器 / 铝板 / 托盘 / 反应板尺寸',
                   command=self.open_geometry_editor).pack(fill='x',pady=(6,5))
        ttk.Separator(right).pack(fill='x',pady=8)
        ttk.Button(right,text='保存 lab_layout.json',command=self.save).pack(fill='x',pady=4)
        ttk.Button(right,text='保存并重新生成 MuJoCo XML',command=self.rebuild).pack(fill='x',pady=4)
        ttk.Label(right,text='鼠标拖动设备：移动\nShift + 滚轮：切换手动旋转 ±5°\nCtrl + 拖温控模块：独立调整模块XY\n自动朝向打开时角度框锁定\n红点：xArm5 世界坐标原点\n设备位置均相对于机器人底座',foreground='#64748b',justify='left').pack(anchor='w',pady=12)
        self.msg=tk.StringVar(value='初始位置；可以手动拖动设备。')
        ttk.Label(root,textvariable=self.msg).pack(anchor='w',padx=14,pady=5)
        self.cv.bind('<Button-1>',self.mouse_down)
        self.cv.bind('<B1-Motion>',self.mouse_drag)
        self.cv.bind('<ButtonRelease-1>',self.mouse_up)
        self.cv.bind('<MouseWheel>',self.mouse_wheel)
        self.reset_fields()

    def get_center(self,which):
        return list(map(float,self.data[which]['position_xy']))

    def reset_fields(self):
        c=self.data[self.sel.get()]
        self.fields['x'].set(f"{c['position_xy'][0]:.3f}")
        self.fields['y'].set(f"{c['position_xy'][1]:.3f}")
        self.fields['yaw'].set(f"{float(c['yaw_deg']):.3f}")
        auto = c.get('orientation_mode', 'face_robot') == 'face_robot'
        self.auto_face.set(auto)
        if self.yaw_entry is not None:
            self.yaw_entry.configure(state='disabled' if auto else 'normal')
        self.draw()

    def set_orientation_mode(self):
        dev = self.data[self.sel.get()]
        dev['orientation_mode'] = 'face_robot' if self.auto_face.get() else 'manual'
        if self.auto_face.get():
            dev['yaw_deg'] = round(facing_yaw_deg(self.sel.get(), dev['position_xy'], opening_side(self.data, self.sel.get())), 6)
        self.reset_fields()

    def apply_fields(self):
        try:
            key = self.sel.get()
            c = self.data[key]
            xy = [float(self.fields['x'].get()), float(self.fields['y'].get())]
            if not all(math.isfinite(v) for v in xy):
                raise ValueError('X/Y must be finite')
            c['position_xy'] = xy
            if c.get('orientation_mode', 'face_robot') == 'face_robot':
                c['yaw_deg'] = round(facing_yaw_deg(key, xy, opening_side(self.data, key)), 6)
            else:
                yaw = float(self.fields['yaw'].get())
                if not math.isfinite(yaw):
                    raise ValueError('Angle must be finite')
                c['yaw_deg'] = yaw
            self.reset_fields()
            return True
        except ValueError as exc:
            messagebox.showerror('Invalid numbers',str(exc))
            return False

    def apply_slots(self):
        try:
            o=self.data['ot2']
            slots={k:int(v.get()) for k,v in self.slots.items()}
            count=int(self.stack_var.get())
            if not all(1 <= slot <= 11 for slot in slots.values()):
                raise ValueError('槽位编号必须为 1 到 11')
            if count < 0:
                raise ValueError('板堆叠数量不能为负数')
            o.update(slots, stack_count=count)
            self.draw()
            return True
        except ValueError as e:
            messagebox.showerror('无效配置',str(e))
            return False

    def apply_temperature_angles(self):
        try:
            angles={key:float(v.get()) for key,v in self.temp_angles.items()}
            if not all(math.isfinite(val) for val in angles.values()):
                raise ValueError('温控板角度必须为有限数字')
            self.data['ot2'].update(angles)
            self.draw()
            return True
        except ValueError as exc:
            messagebox.showerror('温控板角度错误',str(exc))
            return False

    def apply_temperature_offset(self):
        try:
            vals=[float(self.temperature_offset[ax].get()) for ax in ('X','Y','Z')]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError('温控模块 XYZ 偏移必须是有限数字')
            self.data['ot2']['temperature_local_xyz_m']=vals
            self.draw()
            self.msg.set('温控模块独立 XYZ 已更新，保存并重新生成 XML 后生效。')
            return True
        except ValueError as exc:
            messagebox.showerror('温控模块位置错误',str(exc))
            return False

    def face_robot(self):
        if not self.apply_fields():
            return
        for which in ('ot2', 'versamax'):
            self.data[which]['orientation_mode'] = 'face_robot'
        canonicalize_facing(self.data)
        self.reset_fields()
        self.msg.set('两台设备开口现在会持续指向机器人；拖动也会自动更新角度。')

    def mouse_down(self,e):
        # Ctrl + left-drag the temperature module, independently of OT-2.
        # This changes its local XY in the OT-2 base coordinate frame.
        if e.state & 0x0004:
            o=self.data['ot2']
            ox,oy=rotate(o['temperature_local_xyz_m'][:2],float(o['yaw_deg']))
            px,py=xy_screen((o['position_xy'][0]+ox,o['position_xy'][1]+oy))
            if abs(e.x-px)<55 and abs(e.y-py)<55:
                self.drag=('temperature',e.x,e.y,list(o['temperature_local_xyz_m'][:2]))
                return
        hit=[]
        for which in ('ot2','versamax'):
            px,py=xy_screen(self.get_center(which))
            if abs(e.x-px)<115 and abs(e.y-py)<90:
                hit.append(((e.x-px)**2+(e.y-py)**2,which))
        if hit:
            which=min(hit)[1]
            self.sel.set(which);self.reset_fields()
            self.drag=(which,e.x,e.y,self.get_center(which))

    def mouse_drag(self,e):
        if self.drag is None:return
        which,px,py,origin=self.drag
        if which=='temperature':
            yaw=float(self.data['ot2']['yaw_deg'])
            dx,dy=rotate(((e.x-px)/SCALE,-(e.y-py)/SCALE), -yaw)
            xyz=self.data['ot2']['temperature_local_xyz_m']
            xyz[0],xyz[1]=round(origin[0]+dx,4),round(origin[1]+dy,4)
            for i,axis in enumerate(('X','Y')):
                self.temperature_offset[axis].set(str(xyz[i]))
            self.draw()
            return
        self.data[which]['position_xy']=[round(origin[0]+(e.x-px)/SCALE,4),
                                           round(origin[1]-(e.y-py)/SCALE,4)]
        if self.data[which].get('orientation_mode', 'face_robot') == 'face_robot':
            self.data[which]['yaw_deg'] = round(facing_yaw_deg(which, self.data[which]['position_xy'], opening_side(self.data, which)), 6)
        self.reset_fields()

    def mouse_up(self,e):
        self.drag=None

    def mouse_wheel(self,e):
        # Rotate selected device by +/-5 degrees while holding Shift.
        if not (e.state & 0x0001):return
        k=self.sel.get()
        self.data[k]['orientation_mode']='manual'
        self.data[k]['yaw_deg']+=5 if e.delta>0 else -5
        self.reset_fields()

    def draw(self):
        cv=self.cv
        cv.delete('all')
        global SCALE, CX, CY
        table_center,table_size=table_geometry(self.data)
        SCALE=min(290, (CW-95)/table_size[0], (CH-95)/table_size[1])
        CX=CW/2-table_center[0]*SCALE
        CY=CH/2+table_center[1]*SCALE
        for t in range(-1000,1001,100):
            x=CX+t*SCALE/1000
            cv.create_line(x,0,x,CH,fill='#e4e8ef')
            y=CY+t*SCALE/1000
            cv.create_line(0,y,CW,y,fill='#e4e8ef')
        cv.create_polygon(corners(table_center,*table_size,0),
                          fill='#cbd5e1',outline='#8695a6',width=2)
        for which,fill,outline in [('ot2','#b2dbd3','#147d72'),('versamax','#e8e4d9','#727c91')]:
            c=self.data[which]
            dims=c['body_size_m']
            w,d=float(dims[0]),float(dims[1])
            xy=self.get_center(which)
            deg=float(c['yaw_deg'])
            cv.create_polygon(corners(xy,w,d,deg),fill=fill,outline=outline,
                              width=4 if self.sel.get()==which else 2)
            x,y=xy_screen(xy)
            # Labels positioned outside the object to avoid obscuring Deck slots.
            label_y=y-(max(w,d)/2)*SCALE-20
            cv.create_text(x,label_y,text=which.upper(),font=('Segoe UI',11,'bold'),fill='#0f172a')
            cv.create_text(x,label_y+16,text=f'{xy[0]:+.2f}, {xy[1]:+.2f} | {deg:.1f}°',
                           fill='#334155',font=('Segoe UI',8))
            local_open=(0,-d*.36)
            ax,ay=rotate(local_open,deg)
            xx,yy=xy_screen((xy[0]+ax,xy[1]+ay))
            cv.create_line(x,y,xx,yy,fill='#d45f20',width=3,arrow='last')

        # V11: show ALL 11 slots in the editor, regardless of whether the
        # optional filled turquoise decorations are disabled in MuJoCo.
        # The borders are only a location guide on the base surface.
        o=self.data['ot2']
        sw,sd=map(float,o['slot_tile_size_m'])
        for i in range(1,12):
            local=ot2_slot_xy(i)
            ox,oy=rotate(local,float(o['yaw_deg']))
            world_xy=(o['position_xy'][0]+ox,o['position_xy'][1]+oy)
            x,y=xy_screen(world_xy)
            color=('#15803d' if i==o['waiting_slot'] else
                   '#2563eb' if i==o['stack_slot'] else '#475569')
            cv.create_polygon(corners(world_xy,sw,sd,float(o['yaw_deg'])),
                              fill='',outline=color,width=2)
            cv.create_text(x,y,text=str(i),fill=color,font=('Segoe UI',9,'bold'))
        local_xyz=o['temperature_local_xyz_m']
        rox,roy=rotate(local_xyz[:2],float(o['yaw_deg']))
        centre=(o['position_xy'][0]+rox,o['position_xy'][1]+roy)
        cv.create_text(*xy_screen(centre), text='Temp', fill='#6b21a8',
                       font=('Segoe UI',9,'bold'), anchor='s')
        base_deg=float(o['yaw_deg'])+float(o['temperature_yaw_deg'])+float(o.get('temperature_base_yaw_deg',90))
        block_deg=float(o['yaw_deg'])+float(o['temperature_yaw_deg'])+float(o.get('temperature_block_yaw_deg',0))
        bw,bd,_=o['temperature_base_size_m']
        aw,ad,_=o['temperature_block_size_m']
        cv.create_polygon(corners(centre,bw,bd,base_deg),fill='#94a3b8',outline='#475569',width=1)
        cv.create_polygon(corners(centre,aw,ad,block_deg),fill='#f9e2a3',outline='#d97706',width=2)

        # Reader notch (RED) and moving tray (YELLOW), same positions/sizes as 3D.
        v=self.data['versamax']
        r=versamax_local_regions(self.data)
        vx,vy=self.get_center('versamax')
        v_yaw=float(v['yaw_deg'])
        rx,ry=rotate(r['opening_center_xy'],v_yaw)
        tx,ty=rotate(r['tray_center_xy'],v_yaw)
        recess_center=(vx+rx,vy+ry)
        tray_center=(vx+tx,vy+ty)
        # Open southwest notch: dark floor; ONLY north/east red edge lines.
        cv.create_polygon(corners(recess_center,*r['opening_size_xy'],v_yaw),
                          fill='#303b4b',outline='')
        for wall in ('north_wall_local', 'east_wall_local'):
            p1, p2 = r[wall]
            aa = rotate(p1,v_yaw); bb = rotate(p2,v_yaw)
            aa_screen = xy_screen((vx+aa[0], vy+aa[1]))
            bb_screen = xy_screen((vx+bb[0], vy+bb[1]))
            cv.create_line(*aa_screen,*bb_screen,fill='#dc2626',width=4)
        cv.create_polygon(corners(tray_center,*r['tray_size_xy'],v_yaw),
                          fill='#ffe6a3',outline='#eab308',width=2)
        cv.create_text(*xy_screen(tray_center),text='Plate Tray',
                       font=('Segoe UI',8),fill='#7c2d12')
        cx,cy=xy_screen((0,0))
        cv.create_oval(cx-21,cy-21,cx+21,cy+21,fill='#e9a8a7',outline='#b91c1c',width=2)
        cv.create_text(cx,cy,text='xArm5',font=('Segoe UI',8,'bold'))
        cv.create_line(cx,cy,cx+70,cy,fill='#ef4444',width=2,arrow='last')
        cv.create_text(cx+80,cy,text='+X',anchor='w',fill='#b91c1c')
        cv.create_line(cx,cy,cx,cy-65,fill='#2563eb',width=2,arrow='last')
        cv.create_text(cx,cy-80,text='+Y',fill='#1d4ed8')
        cv.create_text(18,18,text='顶视布局：机器人底座 = (0,0,0)',anchor='nw',fill='#1e293b',font=('Segoe UI',11,'bold'))

    def open_geometry_editor(self):
        """Change real 3D dimensions, not just 2D decorative rectangle sizes."""
        dlg=tk.Toplevel(self.root)
        dlg.title('编辑实际尺寸 - 全部单位为米 (m)')
        dlg.geometry('980x800')
        dlg.transient(self.root)
        frame=ttk.Frame(dlg,padding=14)
        frame.pack(fill='both',expand=True)
        entries={}
        def add_values(parent,label,section,key,unit='m'):
            ttk.Label(parent,text=label,width=21).pack(anchor='w',pady=(6,0))
            val=self.data[section][key]
            is_array=isinstance(val,list)
            sv=tk.StringVar(value=', '.join(str(x) for x in val) if is_array else str(val))
            ttk.Entry(parent,textvariable=sv,width=38).pack(anchor='w',pady=(1,2))
            entries[(section,key)]=(sv,is_array)
        left=ttk.LabelFrame(frame,text='OT-2 / 反应板（单位 m）',padding=10)
        right=ttk.LabelFrame(frame,text='VersaMax / 红框 / 黄色托盘（单位 m）',padding=10)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,10))
        right.grid(row=0,column=1,sticky='nsew')
        frame.columnconfigure(0,weight=1)
        frame.columnconfigure(1,weight=1)
        for label,section,key in [
            ('OT-2 外形 宽,深,高','ot2','body_size_m'),
            ('桌面最小 长 X,宽 Y','table','size_xy'),
            ('自动适配边距 (m)','table','auto_fit_margin_m'),
            ('OT-2 实体底板厚度 (默认0.068m)','ot2','base_thickness_m'),
            ('Deck 厚度 (m)','ot2','deck_thickness_m'),
            ('槽位平面 X,Y (m)','ot2','slot_tile_size_m'),
            ('Deck 额外竖直偏移 (m)','ot2','deck_vertical_offset_m'),
            ('温控底座 X尺寸,Y尺寸,Z尺寸','ot2','temperature_base_size_m'),
            ('温控铝块 X尺寸,Y尺寸,Z尺寸','ot2','temperature_block_size_m'),
            ('反应板原始 X,Y,Z (m)','plate','size_m'),
            ('板堆叠垂直间距','ot2','stack_plate_pitch_m'),
        ]:add_values(left,label,section,key)
        for label,section,key in [
            ('VersaMax 机身 X尺寸,Y尺寸,Z尺寸','versamax','body_size_m'),
            ('红色凹槽 X尺寸,Y尺寸,Z尺寸','versamax','recess_size_m'),
            ('黄色托盘 X尺寸,Y尺寸,Z厚度','versamax','tray_size_m'),
            ('托盘向内收回距离（0=完全伸出）','versamax','tray_retraction_m'),
            ('托盘上表面高度','versamax','tray_surface_height_m'),
        ]: add_values(right,label,section,key)
        slot_settings=ttk.Frame(left);slot_settings.pack(fill='x',pady=(8,3))
        ttk.Label(slot_settings,text='槽位尺寸来源').pack(side='left')
        slot_mode=tk.StringVar(value=self.data['ot2']['slot_size_mode'])
        ttk.Combobox(slot_settings, textvariable=slot_mode, state='readonly', width=23,
                     values=('plate_plus_1mm','reference_128x86','custom')).pack(side='left',padx=6)
        ttk.Label(left, text='plate_plus_1mm: 反应板平放长宽各+1mm（总尺寸）\n'
                        'reference_128x86: OT-2 开源参考模型的槽位宽深\n'
                        'custom: 直接使用上方输入的槽位 X,Y',
                  foreground='#64748b', justify='left').pack(anchor='w')
        settings_fr=ttk.Frame(left);settings_fr.pack(fill='x',pady=8)
        auto_table=tk.BooleanVar(value=bool(self.data['table'].get('auto_fit',True)))
        ttk.Checkbutton(settings_fr,text='桌面自动覆盖两台设备（保留最小尺寸）',variable=auto_table).pack(anchor='w')
        show_deck=tk.BooleanVar(value=bool(self.data['ot2'].get('show_deck_slab',False)))
        ttk.Checkbutton(settings_fr,text='显示额外 Deck 平板（默认关）',variable=show_deck).pack(anchor='w')
        show_tiles=tk.BooleanVar(value=bool(self.data['ot2'].get('show_slot_tiles',False)))
        ttk.Checkbutton(settings_fr,text='显示青色实心槽位（可选；默认关）',variable=show_tiles).pack(anchor='w')
        show_outlines=tk.BooleanVar(value=bool(self.data['ot2'].get('show_slot_outlines',True)))
        ttk.Checkbutton(settings_fr,text='在 OT-2 底板上显示 11 个槽位轮廓（推荐开启）',variable=show_outlines).pack(anchor='w')
        axes_fr=ttk.Frame(left);axes_fr.pack(fill='x',pady=9)
        ttk.Label(axes_fr,text='反应板平放时的板面').pack(side='left')
        axes_var=tk.StringVar(value=self.data['plate'].get('flat_surface_axes','XY'))
        ttk.Combobox(axes_fr,values=('XZ','XY'),state='readonly',width=7,
                     textvariable=axes_var).pack(side='left',padx=7)
        ttk.Label(left,text='XZ: 原始 Y=厚度，Z=深度；XY: 原始 Z=厚度',
                  foreground='#64748b').pack(anchor='w')
        ttk.Label(frame,text='独立设置各个部件 X/Y/Z 尺寸，不限制板和托盘之间的大小关系。\n'
                   '红色凹槽西、南侧开放；收回距离=0 时两者南端齐平，可自行调整位置。',
                   foreground='#64748b',justify='left').grid(row=1,column=0,columnspan=2,sticky='w',pady=12)
        def apply():
            candidate=json.loads(json.dumps(self.data))
            try:
                for (sec,key),(var,is_array) in entries.items():
                    raw=var.get().strip()
                    value=[float(x.strip()) for x in raw.split(',')] if is_array else float(raw)
                    if is_array and len(value)!=len(candidate[sec][key]):
                        raise ValueError(f'{sec}.{key} 需要 {len(candidate[sec][key])} 个数字')
                    candidate[sec][key]=value
                candidate['plate']['flat_surface_axes']=axes_var.get()
                candidate['ot2']['slot_size_mode']=slot_mode.get()
                # 'custom' uses the explicit size input; other modes are
                # derived from plate/or from the reference in apply_defaults.
                apply_geometry_defaults(candidate)
                candidate['table']['auto_fit']=bool(auto_table.get())
                candidate['ot2']['show_deck_slab']=bool(show_deck.get())
                candidate['ot2']['show_slot_tiles']=bool(show_tiles.get())
                candidate['ot2']['show_slot_outlines']=bool(show_outlines.get())
                validate_geometry(candidate)
                canonicalize_facing(candidate)
            except (ValueError,KeyError) as exc:
                messagebox.showerror('尺寸无效',str(exc),parent=dlg)
                return
            self.data=candidate
            self.reset_fields()
            self.msg.set('尺寸已更新（尚未保存）；请点 保存并重新生成 XML。')
            dlg.destroy()
        ttk.Button(frame,text='应用尺寸并返回',command=apply).grid(row=2,column=0,columnspan=2,pady=6)

    def save(self):
        if not self.apply_fields():
            return False
        if not self.apply_slots() or not self.apply_temperature_angles():
            return False
        if not self.apply_temperature_offset():
            return False
        apply_geometry_defaults(self.data)
        canonicalize_facing(self.data)
        try:
            validate_geometry(self.data)
        except ValueError as exc:
            messagebox.showerror('几何尺寸校验失败',str(exc))
            return False
        # Validate config in memory before writing to disk.
        try:
            slots = (self.data['ot2']['waiting_slot'],
                     self.data['ot2']['stack_slot'])
            if not all(1 <= int(slot) <= 11 for slot in slots):
                raise ValueError('OT-2 槽位编号应在 1 到 11 之间')
            if int(self.data['ot2']['stack_count']) < 0:
                raise ValueError('板堆叠数量不能为负数')
        except (ValueError, KeyError) as exc:
            messagebox.showerror('不能保存无效布局',str(exc))
            return False
        CONFIG.write_text(json.dumps(self.data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        self.msg.set('已保存: '+str(CONFIG))
        return True

    def rebuild(self):
        if not self.save():
            return
        try:
            for script, args in [('build_lab_scene.py', ['--build-only']),
                                 ('prepare_physics_scene.py', []),
                                 ('setup_tcp_cameras.py', [])]:
                result=subprocess.run([sys.executable,str(ROOT/script),*args],
                                      cwd=str(ROOT),capture_output=True,text=True,timeout=150)
                if result.returncode!=0:
                    messagebox.showerror('生成失败',script+'\n'+(result.stderr or result.stdout)[-2400:])
                    return
        except Exception as e:
            messagebox.showerror('生成失败',str(e));return
        self.msg.set('布局、物理及 TCP / 相机场景已更新；重启 ik_control_panel.py 查看。')
        messagebox.showinfo('生成成功','全部场景已更新。请关闭旧窗口并重新运行 ik_control_panel.py。')


if __name__=='__main__':
    if not CONFIG.exists():
        raise SystemExit('Missing lab_layout.json beside layout_editor.py')
    root=tk.Tk()
    app=LayoutEditor(root)
    root.mainloop()
