"""The simulated workcell the plate workflow drives: Allan's model and his
PhysicsRuntime with our plates and pads, plate poses, level-flange IK, arm and gripper commands,
and one tick of physics.

Skills, checks and the backend reach the workcell only through this class. A class with the same
methods over the real xArm and G2 can replace it later; the MuJoCo-only parts (placing plates by
hand, the scratch collision check used when planning) are marked as simulation helpers."""
from dataclasses import dataclass

import mujoco
import numpy as np

from plate_bt import scene

PLATE_HALF = np.array([0.064, 0.042, 0.0075])   # Allan's plate box
PAD_INNER_CLOSED = 0.035                        # each pad's inner face from the TCP at finger 0 (measured on the hull)
PAD_AHEAD = 0.0135                              # the pads' centre lies this far ahead of the TCP, along the fingers
FINGER_MAX = 0.04                               # Allan's finger joint range
FINGER_SPEED = 0.02                             # m/s per finger, as Allan's jog_fingers
GUARD_DEPTH = 0.0003                            # Allan's collision guard: contacts deeper than this block a move


class PlanError(RuntimeError):
    """A motion that cannot be planned: IK misses, or a waypoint collides."""


@dataclass
class Pose:
    """A plate's or the TCP's position (m) and yaw (rad) in Allan's world frame."""
    xyz: np.ndarray
    yaw: float

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, float)

    def moved(self, dxyz=(0, 0, 0), dyaw=0.0):
        return Pose(self.xyz + np.asarray(dxyz, float), self.yaw + dyaw)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SimWorld:
    def __init__(self, cfg, n_plates, n_tips=0, cameras=None):
        self.cfg = cfg
        self.model, self.data, self.info, self.runtime, self.kin = scene.build(cfg, n_plates, n_tips, cameras)
        m = self.model
        self.dt = 1.0 / cfg["tick_hz"]
        self.tcp = self.kin.site_id(m, "g2_tcp")
        self.grip = self.runtime.controller
        self.scratch = mujoco.MjData(m)
        self.plates = [m.body(scene.plate_name(k)).id for k in range(1, n_plates + 1)]
        self.plate_geom = [m.geom("reaction_plate_box" if k == 1 else f"reaction_plate_box_{k}").id
                           for k in range(1, n_plates + 1)]
        self.plate_qadr = [int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.plates]
        self.plate_vadr = [int(m.jnt_dofadr[m.body_jntadr[b]]) for b in self.plates]
        self.tips = [m.body(scene.tip_name(i)).id for i in range(n_tips)]
        self.finger_bodies = set(self.grip.finger_bodies)
        self.finger_act = list(self.grip.act)
        self.active = None
        self.devices = []         # other driven parts of the cell (the reader's drawer), updated each tick
        for joint, armature in cfg.get("joint_armature", {}).items():
            m.dof_armature[m.jnt_dofadr[m.joint(joint).id]] = armature
        for k in range(n_plates):
            self.set_plate_contact(k, cfg["plate_contact"]["friction"])
        self.set_grip_force(cfg["grasp"]["force_n"])

    # ---- time and physics ------------------------------------------------------------------
    @property
    def time(self):
        return float(self.data.time)

    def step(self):
        """One tick: each device sets its controls, Allan's runtime moves the arm command toward
        its target (guarded against collisions) and steps the physics."""
        for device in self.devices:
            device.update()
        self.runtime.advance(self.dt)

    @property
    def blocked(self):
        return self.runtime.blocked

    # ---- plates ----------------------------------------------------------------------------
    def plate_pose(self, k):
        """Pose of plate k (0-based index), and its tilt from level in degrees."""
        b = self.plates[k]
        R = self.data.xmat[b].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        return Pose(self.data.xpos[b].copy(), yaw), tilt

    def put_plate(self, k, pose):
        """Simulation helper: a person sets plate k down at `pose` (at rest)."""
        self.put_body(self.plates[k], pose)

    def put_body(self, b, pose):
        """Simulation helper: free body b at `pose` (level, at rest)."""
        a, v = int(self.model.jnt_qposadr[self.model.body_jntadr[b]]), int(self.model.jnt_dofadr[self.model.body_jntadr[b]])
        self.data.qpos[a:a + 3] = pose.xyz
        self.data.qpos[a + 3:a + 7] = yaw_quat(pose.yaw)
        self.data.qvel[v:v + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def overlaps(self, k=None, body=None):
        """Contacts deeper than the guard depth between plate k (or any body) and anything: an
        empty list means it was put down without overlapping anything."""
        b = self.plates[k] if body is None else body
        out = []
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if c.dist > -GUARD_DEPTH or b not in (b1, b2):
                continue
            other = c.geom2 if b1 == b else c.geom1
            out.append((self.model.geom(other).name, -float(c.dist)))
        return out

    def set_plate_contact(self, k, mu):
        """Plate k's contacts take the pads' parameters (Allan's physics_runtime.py: condim 4,
        the pads' solref) with friction mu. MuJoCo would otherwise average the pads' stiff solref with
        the plate's default 0.02, and the held plate creeps. The plate gets priority, so its
        parameters are used whole; mu below the pads' 1.2 models a wet plate. The config's
        `plate_contact.solref`, if given, replaces the pads' solref."""
        g, pad = self.plate_geom[k], self.model.geom("left_grip_pad").id
        m = self.model
        m.geom_priority[g] = 1
        m.geom_condim[g] = m.geom_condim[pad]
        m.geom_friction[g] = [mu, *m.geom_friction[pad][1:]]
        m.geom_solref[g] = self.cfg["plate_contact"].get("solref", m.geom_solref[pad])
        m.geom_solimp[g] = m.geom_solimp[pad]

    def set_active(self, k):
        """The plate the gripper may touch: Allan's grasp controller and collision guard exempt
        finger contacts with this plate only. His controller reads `plate_bodies` (a set, since his
        four-plate workflow) where it read `plate` before; both are set."""
        self.active = k
        self.grip.plate = self.plates[k]
        self.grip.plate_bodies = {self.plates[k]}

    def held(self):
        """Both pads press on the active plate (Allan's PhysicalGraspController.held)."""
        return self.active is not None and self.grip.held

    def pad_contacts(self):
        """(normal force, tangential force, friction coefficient) of each finger contact with the
        active plate."""
        out, f = [], np.zeros(6)
        if self.active is None:
            return out
        pb = self.plates[self.active]
        for n in range(self.data.ncon):
            c = self.data.contact[n]
            bodies = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if pb in bodies and bodies & self.finger_bodies:
                mujoco.mj_contactForce(self.model, self.data, n, f)
                out.append((float(f[0]), float(np.hypot(f[1], f[2])), float(c.friction[0])))
        return out

    # ---- arm -------------------------------------------------------------------------------
    def tcp_pose(self):
        R = self.data.site_xmat[self.tcp].reshape(3, 3)
        return Pose(self.data.site_xpos[self.tcp].copy(), float(np.arctan2(R[1, 0], R[0, 0])))

    def joints(self):
        return self.kin.angles(self.data, self.info)

    @property
    def command(self):
        return self.runtime.command.copy()

    def set_arm_target(self, q):
        self.runtime.set_arm_target(q)

    def hold_arm(self):
        """Stop where the command is now."""
        self.runtime.set_arm_target(self.runtime.command)

    def arrived(self, rad=0.002, rad_s=0.02):
        q = self.joints()
        v = np.array([self.data.qvel[dof] for _, _, dof in self.info])
        return np.max(np.abs(q - self.runtime.arm_target)) < rad and np.max(np.abs(v)) < rad_s

    def ik(self, xyz, point, seed):
        """Joint angles that put the TCP at xyz with a level flange and the fingers pointing at
        `point` (rad, Allan's flange_yaw), starting from `seed`."""
        s = self.scratch
        s.qpos[:] = self.data.qpos
        seed = np.asarray(seed, float)
        best, why = None, "no seed admits a level flange"
        # the given seed first; if it misses, Allan's own seeds (robot_kinematics.solve_ik), as he
        # wrote them and with joint1 turned toward the target, keeping the solution nearest the seed
        towards = np.arctan2(xyz[1], xyz[0])
        allan = [self.kin.Q0, np.zeros(5), [0.7, -0.5, -1.0, 0.7, 0], [-0.7, -0.5, -1.0, 0.7, 0], [0.0, 0.3, -1.3, 0.5, 0.0]]
        turned = [np.r_[towards, np.asarray(s, float)[1:]] for s in allan]
        for k, start in enumerate([seed, *allan, *turned]):
            start = np.asarray(start, float).copy()
            try:
                q, err, _ = self.kin.jac_ik(self.model, s, self.info, self.tcp, np.asarray(xyz, float), start,
                                             1e-5, level=True, flange_yaw=point)
            except ValueError as e:
                why = str(e)
                continue
            if err <= 2e-4:
                if k == 0:
                    return q
                if best is None or np.max(np.abs(q - seed)) < np.max(np.abs(best - seed)):
                    best = q
            else:
                why = f"misses by {err * 1000:.1f} mm"
        if best is None:
            raise PlanError(f"IK at {np.round(xyz, 3).tolist()} pointing {np.degrees(point):.0f} deg: {why}")
        return best

    # ---- gripper ---------------------------------------------------------------------------
    def finger(self):
        """The fingers' commanded opening per side (m); 0 is closed."""
        return float(self.grip.target[1])

    def set_finger(self, q):
        q = float(np.clip(q, 0.0, FINGER_MAX))
        self.grip.target[:] = [-q, q]
        self.data.ctrl[self.finger_act] = self.grip.target

    def set_grip_force(self, newton):
        for a in self.finger_act:
            self.model.actuator_forcerange[a] = [-newton, newton]

    def grip_force(self):
        return float(self.model.actuator_forcerange[self.finger_act[0], 1])

    @staticmethod
    def opening_for(grasp, clearance):
        """Finger opening that leaves `clearance` between each pad and a plate held by its
        long or short edges."""
        half = PLATE_HALF[1] if grasp == "long" else PLATE_HALF[0]
        return min(FINGER_MAX, half - PAD_INNER_CLOSED + clearance)

    # ---- planning helper (simulation) ------------------------------------------------------
    def collisions(self, q, finger, carry=None, ignore=None):
        """Simulation helper for planning: contacts deeper than the guard depth between the robot
        (arm at q, fingers at `finger`), or a carried plate, and anything else. `carry` is
        (plate index, plate pose relative to the TCP as a 4x4 matrix); plate `ignore` is left out."""
        s, m = self.scratch, self.model
        s.qpos[:] = self.data.qpos
        a = self.grip.finger_q
        s.qpos[a] = [-finger, finger]
        self.kin.set_angles(m, s, self.info, q)
        if carry is not None:
            k, rel = carry
            T = np.eye(4)
            T[:3, :3] = s.site_xmat[self.tcp].reshape(3, 3)
            T[:3, 3] = s.site_xpos[self.tcp]
            P = T @ rel
            qa = self.plate_qadr[k]
            s.qpos[qa:qa + 3] = P[:3, 3]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, P[:3, :3].ravel())
            s.qpos[qa + 3:qa + 7] = quat
            mujoco.mj_forward(m, s)
        robot = self.runtime.robot_bodies
        carried = self.plates[carry[0]] if carry is not None else None
        skip = self.plates[ignore] if ignore is not None else None
        hits = set()
        for c in s.contact[:s.ncon]:
            if c.dist > -GUARD_DEPTH:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if skip in (b1, b2) or b1 in self.tips or b2 in self.tips:   # a tip rides on its plate; the guard still watches it
                continue
            if carried is not None and carried in (b1, b2):
                if {b1, b2} & self.finger_bodies:         # the grasp itself
                    continue
            elif (b1 in robot) == (b2 in robot):          # robot-robot, or two things the robot does not move
                continue
            hits.add(f"{m.geom(c.geom1).name or m.body(b1).name}/{m.geom(c.geom2).name or m.body(b2).name}")
        return sorted(hits)

    def plate_in_tcp(self, k):
        """Plate k's pose relative to the TCP (4x4)."""
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.tcp]
        P = np.eye(4)
        P[:3, :3] = self.data.xmat[self.plates[k]].reshape(3, 3)
        P[:3, 3] = self.data.xpos[self.plates[k]]
        return np.linalg.inv(T) @ P
