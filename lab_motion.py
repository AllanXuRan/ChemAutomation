"""Cooperative MuJoCo motion primitives extracted from the verified four-plate demo.

Generators set a command and yield; the existing BT mainloop advances physics.
No action owns a second simulation loop or reloads the scene.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
import robot_kinematics as kin
from physics_runtime import PhysicsRuntime, load_physics_model
ROOT=Path(__file__).resolve().parent
HOME=np.asarray(json.loads((ROOT/'ik_records'/'initial_pose.json').read_text())['joint_angles_rad'])
DT=.02

class LabMotion:
    def __init__(self):
        self.config=json.loads((ROOT/'lab_layout.json').read_text(encoding='utf-8'))
        self.model=load_physics_model(kin.SCENE)
        self.data=mujoco.MjData(self.model)
        self.info=kin.arm_info(self.model)
        self.tcp=kin.site_id(self.model,'g2_tcp')
        kin.set_angles(self.model,self.data,self.info,HOME)
        self.runtime=PhysicsRuntime(self.model,self.data,self.info)
        self.controller=self.runtime.controller
        self.sources=sorted(self.controller.plate_bodies,key=lambda b:self.data.xpos[b,2],reverse=True)
        if len(self.sources)!=4:raise ValueError('The verified workflow requires four input plates')
        self.carried=self.sources[0]
        self.processed=[]
        self.cycle=0
        self.records=[]
        self.grasp_yaw=0.
        self.initialized=False
        self.pick_context={}
        self.place_context={}
        (ROOT/'outputs').mkdir(exist_ok=True)

    def advance(self,dt=DT):
        self.runtime.advance(dt)
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError('Invalid physics state or MuJoCo warning')

    def stop(self):
        # Preserve the current physical pose and gripper force; never teleport.
        held=self.holding(self.carried)
        self.runtime.reset_targets()
        self.controller.target[:]=[0.,0.] if held else self.data.qpos[self.controller.finger_q]
        self.data.ctrl[self.controller.act]=self.controller.target

    def wait(self,seconds):
        for _ in range(max(1,int(np.ceil(seconds/DT)))):yield None

    def initialize(self):
        if self.initialized:return
        yield from self.move('initial: settle four-plate source',HOME,[-.04,.04],seconds=2.)
        self.sources=sorted(self.controller.plate_bodies,key=lambda b:self.data.xpos[b,2],reverse=True)
        self.source_positions={b:self.data.xpos[b].copy() for b in self.sources}
        for b in self.sources:
            if not np.allclose(self.data.xpos[b,:2],[.2,.2],atol=.002,rtol=0):
                raise RuntimeError('Input plate is outside the configured source stack')
        self.initialized=True

    def select_plate(self,plate_id):
        if not self.initialized:raise RuntimeError('Mission has not been initialized')
        remaining=[b for b in self.sources if b not in self.processed]
        if not remaining:raise RuntimeError('Source stack is empty')
        self.carried=max(remaining,key=lambda b:self.data.xpos[b,2])
        self.cycle=plate_id
        self.pick_context.clear();self.place_context.clear()
        self.grasp_yaw=0.

    def prepare_pick(self,station,refresh=False):
        if station in self.pick_context and not refresh:return self.pick_context[station]
        yaw=HOME[0]-HOME[4] if station=='reader' else 0.
        self.grasp_yaw=yaw
        centre=self.data.xpos[self.carried].copy()
        bias=max(.005,float(self.config['table']['top_z'])+.014-centre[2]) if station=='source' else 0.
        contact=centre+np.array([.008*np.cos(yaw),.008*np.sin(yaw),bias])
        approach=contact+([0,0,.075] if station=='source' else [0,-.045,.075])
        context=dict(contact=contact,approach=approach,z0=float(centre[2]),yaw=yaw)
        self.pick_context[station]=context
        return context

    def pick_high(self):
        context=self.prepare_pick('source')
        high=context['contact'].copy();high[2]=.32
        yield from self.move('source: high vertical staging',self.solve(high),[-.04,.04],seconds=10.)

    def pick_approach(self,station):
        context=self.prepare_pick(station)
        yield from self.move(station+': approach',self.solve(context['approach'],context['yaw']),[-.04,.04],seconds=10.)

    def pick_descend(self,station):
        context=self.prepare_pick(station)
        yield from self.move(station+': descend',self.solve(context['contact'],context['yaw']),seconds=7.)

    def close_gripper(self,station):
        yield from self.move(station+': close',self.runtime.arm_target.copy(),[0.,0.],seconds=3.,held=True)

    def lift(self,station):
        context=self.prepare_pick(station)
        self.grasp_yaw=context['yaw']
        target=self.data.site_xpos[self.tcp].copy()+([0,0,.075] if station=='source' else [0,-.045,.075])
        yield from self.move(station+': lift',self.solve(target),seconds=7.,held=True)

    def verify_grasp(self,station):
        context=self.pick_context[station]
        return self.holding(self.carried) and self.data.xpos[self.carried,2]>=context['z0']+.03

    def target(self,station):
        site={'aluminium':'ot2_pickup','reader':'versamax_load','storage':'ot2_stack_top'}[station]
        target=self.data.site_xpos[kin.site_id(self.model,site)].copy()
        if station=='aluminium':target[1]-=.006
        if station=='storage':
            target[1]-=.014
            target[2]+=len(self.processed)*2*self.model.geom_size[self.model.geom('reaction_plate_box').id,2]
        return target

    def prepare_place(self,station,refresh=False):
        if station in self.place_context and not refresh:return self.place_context[station]
        target=self.target(station)
        offset=self.data.xpos[self.carried]-self.data.site_xpos[self.tcp]
        landing=target-offset
        approach=landing+([0,-.045,.055] if station=='aluminium' else [0,-.035,.075])
        if station=='storage':
            hover=target.copy();hover[2]=max(.18,float(target[2])+.06)
            approach=hover-offset
        context=dict(target=target,landing=landing,approach=approach)
        self.place_context[station]=context
        return context

    def place_approach(self,station):
        self.grasp_yaw=HOME[0]-HOME[4] if station=='reader' else 0.
        context=self.prepare_place(station,refresh=True)
        yield from self.move(station+': approach',self.solve(context['approach']),seconds=10.,held=True)

    def place_descend(self,station):
        context=self.prepare_place(station)
        yield from self.move(station+': descend',self.solve(context['landing']+[0,0,.002]),seconds=8.,held=True)

    def release(self,station):
        yield from self.move(station+': release',self.runtime.arm_target.copy(),[-.04,.04],seconds=3.)

    def retract(self,station):
        context=self.place_context[station]
        yield from self.move(station+': retract',self.solve(context['approach']),seconds=7.)

    def verify_placement(self,station):
        error=self.data.xpos[self.carried]-self.place_context[station]['target']
        return not self.holding(self.carried) and np.linalg.norm(error[:2])<=.025 and abs(error[2])<=.01

    def transit(self,label,xyz):
        yield from self.move(label,self.solve(np.asarray(xyz)),seconds=12.,held=True)

    def storage_hover(self):
        current=self.runtime.arm_target[0]-self.runtime.arm_target[4]
        hover=self.target('storage');hover[2]=max(.18,float(hover[2])+.06)
        self.storage_hover_point=hover
        offset=self.data.xpos[self.carried]-self.data.site_xpos[self.tcp]
        yield from self.move('storage: above slot2 before rotation',self.solve(hover-offset,current),seconds=14.,held=True)

    def storage_align(self):
        current=self.runtime.arm_target[0]-self.runtime.arm_target[4]
        hover=self.storage_hover_point
        count=max(1,int(np.ceil(abs(current)/.20)))
        previous=current
        for index,angle in enumerate(np.linspace(current,0.,count+1)[1:]):
            offset=self.data.xpos[self.carried]-self.data.site_xpos[self.tcp]
            delta=angle-previous;c,s=np.cos(delta),np.sin(delta)
            rotated=offset.copy();rotated[:2]=[c*offset[0]-s*offset[1],s*offset[0]+c*offset[1]]
            yield from self.move(f'storage: rotate above slot2 {index+1}/{count}',self.solve(hover-rotated,angle),seconds=4.,held=True)
            if np.linalg.norm(self.data.xpos[self.carried,:2]-hover[:2])>.008 or abs(self.data.xpos[self.carried,2]-hover[2])>.006:
                raise RuntimeError('Plate left slot2 hover region during alignment')
            previous=angle
        self.grasp_yaw=0.
        self.prepare_place('storage',refresh=True)

    def verify_storage_aligned(self):
        q=kin.angles(self.data,self.info)
        return self.holding(self.carried) and abs(q[0]-q[4])<.02 and np.linalg.norm(self.data.xpos[self.carried]-self.storage_hover_point)<.01

    def commit_cycle(self):
        if self.carried in self.processed:raise RuntimeError('Duplicate plate commit')
        positions={b:self.data.xpos[b].copy() for b in self.processed+[self.carried]}
        yield from self.wait(2.)
        for b,position in positions.items():
            if np.linalg.norm(self.data.xpos[b]-position)>.003:raise RuntimeError('Storage stack is unstable')
        for b in self.sources:
            if b not in positions and np.linalg.norm(self.data.xpos[b,:2]-self.source_positions[b][:2])>.01:
                raise RuntimeError('Unprocessed source plate displaced')
        if np.max(np.abs(kin.angles(self.data,self.info)-HOME))>.025:raise RuntimeError('Final HOME not reached')
        self.processed.append(self.carried)

    def holding(self, body):
        fingers = set()
        force = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            bodies = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if body not in bodies:
                continue
            other = bodies[1] if bodies[0] == body else bodies[0]
            if other in self.controller.finger_bodies:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                if force[0] > 0.02:
                    fingers.add(other)
        return len(fingers) == 2

    def plan(self, start, goal):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        transporting = self.holding(self.carried)
        tcp_rotation = self.data.site_xmat[self.tcp].reshape(3, 3)
        relative_position = tcp_rotation.T @ (self.data.xpos[self.carried] - self.data.site_xpos[self.tcp])
        relative_rotation = tcp_rotation.T @ self.data.xmat[self.carried].reshape(3, 3)
        plate_address = self.model.jnt_qposadr[self.model.body_jntadr[self.carried]]
        robot_geoms = np.array([g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in self.runtime.robot_bodies], dtype=int)

        def safe(q):
            margins = self.model.geom_margin.copy()
            self.model.geom_margin[robot_geoms] = np.maximum(margins[robot_geoms], 0.003)
            try:
                kin.set_angles(self.model, scratch, self.info, q)
                if transporting:
                    rotation = scratch.site_xmat[self.tcp].reshape(3, 3)
                    scratch.qpos[plate_address:plate_address + 3] = scratch.site_xpos[self.tcp] + rotation @ relative_position
                    quat = np.empty(4)
                    mujoco.mju_mat2Quat(quat, (rotation @ relative_rotation).ravel())
                    scratch.qpos[plate_address + 3:plate_address + 7] = quat
                    mujoco.mj_forward(self.model, scratch)
                for contact in scratch.contact:
                    g1, g2 = (int(contact.geom1), int(contact.geom2))
                    b1, b2 = (int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2]))
                    if b1 not in self.runtime.robot_bodies and b2 not in self.runtime.robot_bodies:
                        continue
                    if b1 in self.controller.plate_bodies and b2 in self.controller.finger_bodies or (b2 in self.controller.plate_bodies and b1 in self.controller.finger_bodies):
                        continue
                    other = g2 if b1 in self.runtime.robot_bodies else g1
                    name = self.model.geom(other).name or ''
                    clearance = 0.002 if name.startswith(('versamax_', 'ot2_frame_post_', 'ot2_side_rail_')) else -0.0002
                    if contact.dist < clearance:
                        return False
                return True
            finally:
                self.model.geom_margin[:] = margins

        def edge(a, b):
            count = max(2, int(np.max(np.abs(b - a)) / 0.012) + 1)
            for index, t in enumerate(np.linspace(0, 1, count)):
                if not safe(a + (b - a) * t):
                    return False
                if index % 16 == 15:
                    yield None
            return True
        if not safe(goal):
            print('Goal contacts:', [(self.model.geom(a).name, self.model.geom(b).name, depth) for (a, b), depth in self.runtime.forbidden_contacts(scratch).items()], flush=True)
            raise RuntimeError('Destination configuration collides')
        if (yield from edge(start, goal)):
            return [goal]
        rng = np.random.default_rng(42)
        nodes = [start.copy()]
        parents = [-1]
        low = np.minimum(start, goal) - [0.7, 0.7, 0.7, 0.7, 0.7]
        high = np.maximum(start, goal) + [0.7, 0.7, 0.7, 0.7, 0.7]
        for attempt in range(6000):
            if attempt % 2 == 0:
                yield None
            sample = goal.copy() if rng.random() < 0.2 else rng.uniform(low, high)
            sample[3] = -sample[1] - sample[2]
            if transporting:
                sample[4] = sample[0] - (start[0] - start[4])
            if not np.allclose(sample, kin.clip_angles(self.model, self.info, sample)):
                continue
            distances = [np.linalg.norm(n - sample) for n in nodes]
            nearest = int(np.argmin(distances))
            delta = sample - nodes[nearest]
            trial = nodes[nearest] + delta * min(1.0, 0.2 / max(np.linalg.norm(delta), 1e-12))
            if not (yield from edge(nodes[nearest], trial)):
                continue
            nodes.append(trial)
            parents.append(nearest)
            if not (yield from edge(trial, goal)):
                continue
            path = [goal]
            index = len(nodes) - 1
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            path.reverse()
            result = []
            index = 0
            while index < len(path) - 1:
                end = len(path) - 1
                while end > index + 1 and (not (yield from edge(path[index], path[end]))):
                    end -= 1
                result.append(path[end])
                index = end
            return result
        raise RuntimeError('No collision-free joint path found')

    def move(self, name, q, grip=None, seconds=5.0, held=False, planned=False):
        if not planned:
            route = (yield from self.plan(self.runtime.arm_target.copy(), q))
            for index, waypoint in enumerate(route):
                final = index == len(route) - 1
                yield from self.move(name + (f' via {index + 1}/{len(route)}' if len(route) > 1 else ''), waypoint, grip if final else None, max(seconds / len(route), float(np.max(np.abs(waypoint - self.runtime.arm_target))) / (0.09 if held else 0.2)), held, True)
            return
        print(name, flush=True)
        start = self.runtime.arm_target.copy()
        fingers = self.controller.target.copy()
        target = fingers if grip is None else np.asarray(grip)
        monitor_grasp = held and self.holding(self.carried)
        lost_time = 0.0
        for frame in range(int((seconds + 1.0) / 0.02)):
            t = min(1.0, (frame + 1) * 0.02 / seconds)
            u = t * t * t * (10 - 15 * t + 6 * t * t)
            self.runtime.set_arm_target(start + u * (q - start))
            self.controller.target[:] = fingers + u * (target - fingers)
            self.data.ctrl[self.controller.act] = self.controller.target
            yield None
            if monitor_grasp:
                lost_time = 0.0 if self.holding(self.carried) else lost_time + 0.02
                if lost_time > 0.2:
                    raise RuntimeError(name + ': sustained loss of contact')
            if self.runtime.blocked:
                raise RuntimeError(name + ': collision ' + self.runtime.blocked)
        error = float(np.max(np.abs(kin.angles(self.data, self.info) - q)))
        if error > 0.025:
            raise RuntimeError(name + ': joint tracking error ' + str(error))
        if held and (not self.holding(self.carried)):
            print('Failed plate position:', self.data.xpos[self.carried], 'TCP:', self.data.site_xpos[self.tcp], flush=True)
            raise RuntimeError(name + ': grasp lost')
        result = dict(stage=name, bt_action=getattr(self, "action_name", ""), cycle=self.cycle, plate=self.model.body(self.carried).name, held=self.holding(self.carried), plate_xyz=self.data.xpos[self.carried].tolist(), joint_angles_rad=q.tolist(), joint_angles_deg=np.degrees(q).tolist(), actual_joint_angles_rad=kin.angles(self.data, self.info).tolist(), tcp_xyz_world_m=self.data.site_xpos[self.tcp].tolist(), tcp_R_world=self.data.site_xmat[self.tcp].reshape(3, 3).tolist(), plate_quat_wxyz=self.data.xquat[self.carried].tolist(), gripper_target_m=target.tolist(), gripper_actual_m=self.data.qpos[self.controller.finger_q].tolist(), duration_s=seconds)
        self.records.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / 'outputs' / 'bt_workflow_progress.json').write_text(json.dumps(self.records, indent=2), encoding='utf-8')

    def solve(self, xyz, yaw=None):
        if yaw is None:
            yaw = self.grasp_yaw
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        q, error, _ = kin.solve_ik(self.model, scratch, self.info, self.tcp, xyz, 5e-05, level=True, flange_yaw=yaw)
        if error > 0.001:
            raise RuntimeError('Unreachable waypoint: ' + str(xyz) + ' error=' + str(error))
        return q

    def home(self, label):
        if self.holding(self.carried):
            current_yaw = self.runtime.arm_target[0] - self.runtime.arm_target[4]
            prehome = HOME.copy()
            prehome[4] = prehome[0] - current_yaw
            yield from self.move(label + ' high approach', prehome, seconds=20.0, held=True)
            yield from self.move(label + ' final wrist alignment', HOME, seconds=24.0, held=True)
        else:
            yield from self.move(label, HOME, seconds=18.0)
        if np.max(np.abs(kin.angles(self.data, self.info) - HOME)) > 0.025:
            raise RuntimeError('HOME was not reached')

    def set_yaw(self, yaw, label):
        current = self.runtime.arm_target[0] - self.runtime.arm_target[4]
        if abs(current - yaw) < 0.005:
            self.grasp_yaw = yaw
            return
        clear = np.array([-0.05, 0.3, 0.34])
        yield from self.move(label + ' clear', self.solve(clear, current), seconds=12.0, held=True)
        count = max(1, int(np.ceil(abs(yaw - current) / 0.26)))
        for index, angle in enumerate(np.linspace(current, yaw, count + 1)[1:]):
            yield from self.move(label + f' turn {index + 1}/{count}', self.solve(clear, angle), seconds=4.0, held=True)
        self.grasp_yaw = yaw
