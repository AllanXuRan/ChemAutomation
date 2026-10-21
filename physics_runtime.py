"""Contact dynamics for the interactive lab; no plate attachment or pose writes."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import robot_kinematics as kin


def finger_sections(model, side):
    """Clip source triangles along X before constructing small convex pieces."""
    mesh = model.mesh(f'{side}_finger_g2_mujoco').id
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[mesh])
    vertices = model.mesh_vert[start:start+count] @ rotation.reshape(3, 3).T + model.mesh_pos[mesh]
    start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    faces = model.mesh_face[start:start+count]
    pieces = []
    for low, high in [(-.005, .015), (.015, .0355), (.0355, .0675), (.0675, .112)]:
        points = []
        for face in faces:
            polygon = list(vertices[face])
            for boundary, sign in [(low, 1), (high, -1)]:
                clipped = []
                for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                    inside_a = sign*(a[0]-boundary) >= 0
                    inside_b = sign*(b[0]-boundary) >= 0
                    if inside_a:
                        clipped.append(a)
                    if inside_a != inside_b:
                        clipped.append(a+(b-a)*(boundary-a[0])/(b[0]-a[0]))
                polygon = clipped
            points.extend(polygon)
        pieces.append(np.unique(np.round(points, 8), axis=0))
    return pieces


def load_physics_model(scene):
    """Build in memory, preserving the generated scene and instrument dimensions."""
    scene = Path(scene).resolve()
    root = ET.parse(scene).getroot()
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            asset.set('file', str((scene.parent / asset.get('file')).resolve()))
    source_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    option = root.find('option')
    option.set('timestep', '0.002')
    option.set('integrator', 'implicitfast')
    option.set('gravity', '0 0 -9.81')
    option.set('iterations', '100')
    option.set('cone', 'elliptic')
    actuator = root.find('actuator')
    for act in actuator:
        if act.get('joint') in kin.ARM:
            act.set('kp', '1200')
            act.set('kv', '70')
    # Enable actual supports/housings, leaving labels, outlines and wells visual.
    support_names = {'ot2_waiting_pad', 'ot2_stack_pad', 'temperature_contact_plate',
                     'versamax_recess_floor', 'versamax_recess_north_face',
                     'versamax_recess_east_face', 'versamax_console',
                     'versamax_tray_insert', 'versamax_tray_guide_0', 'versamax_tray_guide_1'}
    for geom in root.iter('geom'):
        name = geom.get('name', '')
        if name in support_names or name.startswith(('ot2_deck_slot_', 'ot2_stack_plate_')):
            geom.set('contype', '1')
            geom.set('conaffinity', '1')
    # Replace the whole-finger convex hull by several clipped mesh pieces. The hull fills the
    # concave gripping region, but a tip-only proxy leaves the shank intangible.
    for side in ('left', 'right'):
        body = root.find(f".//body[@name='bio_gripper_{side}_finger']")
        for geom in body.findall('geom'):
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
        joint = body.find('joint')
        joint.set('damping', '1')
        joint.set('actuatorfrcrange', '-4 4')
        for part, vertices in zip(('root', 'knuckle', 'shank', 'pad'), finger_sections(source_model, side)):
            mesh_name = f'{side}_finger_{part}_hull'
            ET.SubElement(root.find('asset'), 'mesh', name=mesh_name,
                          vertex=' '.join(f'{v:.8g}' for v in vertices.ravel()))
            name = f'{side}_grip_pad' if part == 'pad' else f'{side}_finger_{part}_collision'
            ET.SubElement(body, 'geom', name=name, type='mesh', mesh=mesh_name, mass='0',
                          rgba='0.2 0.2 0.2 0', group='3', condim='4',
                          friction='1.2 0.01 0.001', solref='0.004 1', solimp='0.95 0.99 0.001')
        ET.SubElement(actuator, 'position', name=f'{side}_finger_position',
                      joint=f'{side}_finger_joint', kp='1500', kv='4',
                      ctrllimited='true', ctrlrange=joint.get('range'),
                      forcelimited='true', forcerange='-4 4')
    # Enforce the flange relation in the dynamics solver as well as IK.
    tendon = root.find('tendon')
    if tendon is None:
        tendon = ET.SubElement(root, 'tendon')
    fixed = ET.SubElement(tendon, 'fixed', name='flange_level_sum')
    for name in ('joint2', 'joint3', 'joint4'):
        ET.SubElement(fixed, 'joint', joint=name, coef='1')
    ET.SubElement(root.find('equality'), 'tendon', name='flange_horizontal',
                  tendon1='flange_level_sum', polycoef='0 0 0 0 0',
                  solref='-1000000 -2000', solimp='0.9999 0.99999 0.0001')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class PhysicsRuntime:
    def __init__(self, model, data, info):
        self.model, self.data, self.info = model, data, info
        self.arm_act = [model.actuator(n + '_position').id for n in kin.ARM]
        self.arm_target = kin.level_flange(model, info, kin.angles(data, info))
        self.command = self.arm_target.copy()
        self.remainder = 0.
        self.blocked = ''
        self.collision_data = mujoco.MjData(model)
        mount = model.body('lab_robot_mount').id
        self.robot_bodies = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != mount:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == mount:
                self.robot_bodies.add(body)
        self.controller = PhysicalGraspController(self)
        self.reset_targets()

    def reset_targets(self):
        self.arm_target = kin.level_flange(self.model, self.info, kin.angles(self.data, self.info))
        self.command = self.arm_target.copy()
        self.data.ctrl[self.arm_act] = self.command
        self.remainder = 0.
        self.blocked = ''

    def set_arm_target(self, q):
        self.arm_target = kin.level_flange(self.model, self.info, q)
        return self.arm_target.copy()

    def forbidden_contacts(self, data):
        depths = {}
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if b1 not in self.robot_bodies and b2 not in self.robot_bodies:
                continue
            if ((b1 == self.controller.plate and b2 in self.controller.finger_bodies) or
                    (b2 == self.controller.plate and b1 in self.controller.finger_bodies)):
                continue
            pair = tuple(sorted((g1, g2)))
            depths[pair] = max(depths.get(pair, 0.), -float(contact.dist))
        return depths

    def allow_command(self, candidate):
        scratch = self.collision_data
        scratch.qpos[:] = self.data.qpos
        kin.set_angles(self.model, scratch, self.info, self.command)
        previous = self.forbidden_contacts(scratch)
        kin.set_angles(self.model, scratch, self.info, candidate)
        for pair, depth in self.forbidden_contacts(scratch).items():
            if depth > .0003 and depth > previous.get(pair, 0.) + 1e-7:
                names = [self.model.geom(g).name or self.model.body(int(self.model.geom_bodyid[g])).name
                         for g in pair]
                self.blocked = ' / '.join(names)
                return False
        self.blocked = ''
        return True

    def advance(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid simulation elapsed time')
        self.remainder += min(elapsed, .1)
        h = self.model.opt.timestep
        steps = int((self.remainder + 1e-12) / h)
        for _ in range(steps):
            delta = self.arm_target - self.command
            # One interpolation fraction preserves joint2+joint3+joint4=0.
            candidate = self.command + delta * min(1., .5*h / max(np.max(np.abs(delta)), 1e-12))
            if np.max(np.abs(delta)) > 1e-10:
                if self.allow_command(candidate):
                    self.command = candidate
                else:
                    # Cancel the remaining push; a new target may retreat.
                    self.arm_target = self.command.copy()
            self.data.ctrl[self.arm_act] = self.command
            mujoco.mj_step(self.model, self.data)
        self.remainder -= steps*h
        mujoco.mj_forward(self.model, self.data)


class PhysicalGraspController:
    def __init__(self, runtime):
        self.runtime = runtime
        self.model, self.data = runtime.model, runtime.data
        self.plate = self.model.body('reaction_plate').id
        self.finger_bodies = [self.model.body(f'bio_gripper_{s}_finger').id for s in ('left', 'right')]
        self.act = [self.model.actuator(f'{s}_finger_position').id for s in ('left', 'right')]
        self.finger_q = [self.model.jnt_qposadr[self.model.joint(f'{s}_finger_joint').id]
                         for s in ('left', 'right')]
        self.opened = np.array([-.04, .04])
        self.target = self.opened.copy()
        self.initial = self.data.qpos.copy()
        self.data.qpos[self.finger_q] = self.opened
        self.data.ctrl[self.act] = self.target
        mujoco.mj_forward(self.model, self.data)

    @property
    def held(self):
        """Both finger pads currently exert positive normal contact force."""
        touching = set()
        force = np.zeros(6)
        for n in range(self.data.ncon):
            contact = self.data.contact[n]
            bodies = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if self.plate not in bodies:
                continue
            other = bodies[1] if bodies[0] == self.plate else bodies[0]
            if other in self.finger_bodies:
                mujoco.mj_contactForce(self.model, self.data, n, force)
                if force[0] > .02:
                    touching.add(other)
        return len(touching) == 2

    def jog_fingers(self, direction, dt, speed=.02):
        if direction not in (-1, 0, 1) or not np.isfinite(dt) or dt < 0:
            raise ValueError('Invalid gripper direction or time interval')
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('Gripper speed must be positive')
        if direction:
            goal = np.zeros(2) if direction > 0 else self.opened
            self.target += np.clip(goal-self.target, -speed*min(dt, .1), speed*min(dt, .1))
            self.data.ctrl[self.act] = self.target
        return self.held

    def attach(self):
        # Request closing; actual success is checked after physics settles.
        self.target[:] = 0
        self.data.ctrl[self.act] = self.target

    def release(self):
        self.target[:] = self.opened
        self.data.ctrl[self.act] = self.target

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial
        self.data.qpos[self.finger_q] = self.opened
        self.target[:] = self.opened
        self.data.ctrl[self.act] = self.target
        self.runtime.reset_targets()
        mujoco.mj_forward(self.model, self.data)
