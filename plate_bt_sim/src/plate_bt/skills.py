"""Arm and gripper motions of the plate workflow, as generators.

Each `next()` on a skill issues one tick of commands; the caller steps the physics between
calls. A skill's return value (its StopIteration value) is True when it finished and False when
it stopped short: Allan's collision guard blocked the arm, or the arm did not arrive in time.
Planning happens before the first command: a path that IK cannot follow, or whose waypoints
collide, raises PlanError and nothing moves.

Moves are straight TCP lines cut into IK waypoints, followed on a minimum-jerk profile (as in
scripts/sim/run_task.py, where a full-speed start at each waypoint shook the plate loose)."""
import numpy as np

from plate_bt.world import PAD_AHEAD, PlanError, Pose, wrap

SEGMENT_M = 0.01             # IK waypoints this far apart along a line...
SEGMENT_RAD = np.radians(3)  # ...and this far apart in pointing
JOINT_STEP = 0.02           # rad between checked configurations of a joint-space move
ARRIVE_TIMEOUT_S = 4.0
BRAKE_S = 0.4


class Skills:
    def __init__(self, world, cfg):
        self.w = world
        self.cfg = cfg
        self.speed_scale = 1.0
        self.carry_rad_s = None  # a slower carry for a weak grasp (the tray's), until release
        self.carry = None        # (plate index, plate pose in the TCP frame) while a plate is held
        self.last_dq = np.zeros(5)

    # ---- planning ----------------------------------------------------------------------------
    def waypoints(self, path, seed=None):
        """Joint waypoints along straight lines through `path`, a list of (xyz, point), starting
        from the current TCP pose. The pointing turns the short way round."""
        w = self.w
        seed = w.command if seed is None else seed
        start = w.tcp_pose()
        prev_xyz, prev_point = start.xyz, start.yaw
        qs = []
        for xyz, point in path:
            xyz = np.asarray(xyz, float)
            point = prev_point + wrap(point - prev_point)
            n = max(1, int(np.ceil(max(np.linalg.norm(xyz - prev_xyz) / SEGMENT_M,
                                       abs(point - prev_point) / SEGMENT_RAD))))
            for i in range(1, n + 1):
                u = i / n
                seed = w.ik(prev_xyz + (xyz - prev_xyz) * u, prev_point + (point - prev_point) * u, seed)
                qs.append(seed)
            prev_xyz, prev_point = xyz, point
        return qs

    def check(self, qs, finger, carry=True):
        """PlanError if any waypoint puts the robot, or the plate it carries (unless carry is
        False), into something."""
        ignore = self.carry[0] if self.carry is not None and not carry else None
        for i, q in enumerate(qs):
            hits = self.w.collisions(q, finger, self.carry if carry else None, ignore)
            if hits:
                tcp = np.round(self.w.scratch.site_xpos[self.w.tcp], 3).tolist()
                raise PlanError(f"waypoint {i + 1}/{len(qs)} at TCP {tcp} collides: {', '.join(hits[:3])}")

    # ---- execution ---------------------------------------------------------------------------
    def rad_s(self):
        s = self.cfg["speed"]
        carry = self.carry_rad_s or s["carry_rad_s"]
        return (carry if self.carry else s["empty_rad_s"]) * self.speed_scale

    def follow(self, qs, stop=None):
        """Move through joint waypoints on a minimum-jerk time profile, then wait to arrive. If
        `stop()` turns true on the way, hold where the arm is and finish (a guarded move)."""
        w = self.w
        path = np.array([w.command] + [np.asarray(q, float) for q in qs])
        seg = np.max(np.abs(np.diff(path, axis=0)), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        length = cum[-1]
        if length > 1e-9:
            duration = max(1.875 * length / self.rad_s(), 3 * w.dt)   # min-jerk peak speed: 1.875 x mean
            t0 = w.time
            prev = path[0]
            while True:
                tau = min((w.time - t0 + w.dt) / duration, 1.0)
                s = (10 - 15 * tau + 6 * tau ** 2) * tau ** 3 * length
                k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
                u = 0.0 if seg[k] < 1e-12 else (s - cum[k]) / seg[k]
                target = path[k] + (path[k + 1] - path[k]) * u
                self.last_dq, prev = target - prev, target
                w.set_arm_target(target)
                yield
                if w.blocked:
                    return False
                if stop is not None and stop():
                    self.last_dq = np.zeros(5)
                    w.hold_arm()
                    return True
                if tau >= 1.0:
                    break
        self.last_dq = np.zeros(5)
        t0 = w.time
        while w.time - t0 < ARRIVE_TIMEOUT_S:
            if w.arrived():
                return True
            yield
            if w.blocked:
                return False
        return False

    def go(self, path, stop=None):
        """Plan the path, check it, follow it. A guarded move (`stop` given) is meant to end in
        contact, so only the robot is checked, not the plate it sets down."""
        qs = self.waypoints(path)
        self.check(qs, self.w.finger(), carry=stop is None)
        return (yield from self.follow(qs, stop))

    def brake(self):
        """Bring a move that was interrupted to a smooth stop: keep going in its last direction
        with the speed falling to zero over BRAKE_S."""
        w = self.w
        n = max(1, int(round(BRAKE_S / w.dt)))
        dq, target = self.last_dq.copy(), w.command
        for i in range(n):
            target = target + dq * (1 - (i + 1) / n)
            w.set_arm_target(target)
            yield
        self.last_dq = np.zeros(5)
        w.hold_arm()

    def wait(self, seconds):
        t0 = self.w.time
        while self.w.time - t0 < seconds - 1e-9:
            yield
        return True

    def fingers(self, q):
        """Open or close the fingers to opening q at Allan's finger speed, then settle."""
        w = self.w
        step = 0.02 * w.dt
        while abs(w.finger() - q) > 1e-9:
            f = w.finger()
            w.set_finger(f + np.clip(q - f, -step, step))
            yield
        yield from self.wait(self.cfg["grasp"]["settle_s"])
        return True

    def close(self):
        """Close on the plate: the finger target goes to zero and the force limit holds it."""
        g = self.cfg["grasp"]
        w = self.w
        step = 0.02 * w.dt
        t0 = w.time
        while w.time - t0 < g["close_s"]:
            w.set_finger(max(0.0, w.finger() - step))
            yield
        yield from self.wait(g["settle_s"])
        return w.held()

    # ---- places ------------------------------------------------------------------------------
    @staticmethod
    def tcp_for(plate, point, h=0.0, beyond=0.0):
        """The TCP position that grasps a plate at `plate` with the fingers pointing at `point`:
        the pads' centre on the plate's centre, or `beyond` m past it along the fingers, raised by h."""
        back = PAD_AHEAD - beyond
        return plate.xyz + [-back * np.cos(point), -back * np.sin(point), h]

    def transit(self, area, target_xyz, point, here_area=None, leave_z=None):
        """From where the TCP is to `target_xyz` with the fingers pointing at `point`. Inside the
        OT-2 (`here_area` has a front_y): up to `leave_z` (the area's approach height) and straight
        out through the area's front; elsewhere: up to transit height. Then a joint-space move to
        the target area's front at the target's height (or above the target at transit height),
        and straight in or down. `area` and `here_area` are config entries. The joint-space leg keeps the flange level (Allan's
        joint2 + joint3 + joint4 = 0 is linear) and stays clear of the singularity a straight
        line meets where it passes over the robot's base."""
        w, z = self.w, self.cfg["transit_z"]
        here = w.tcp_pose()
        leave = (here_area or {}).get("front_y")
        if here_area is None:      # between areas (a move was interrupted): straight to the joint-space leg
            out = []
        elif leave is not None:    # inside the OT-2: up to the approach height, then out at that height
            up = max(here.xyz[2], leave_z if leave_z is not None else here.xyz[2])
            out = [((here.xyz[0], here.xyz[1], up), here.yaw), ((here.xyz[0], leave, up), here.yaw)]
        else:
            out = [((here.xyz[0], here.xyz[1], max(z, here.xyz[2])), here.yaw)]
        if "front_y" in area:      # in at the target's height (above the heater, transit height is out of reach)
            entry = (target_xyz[0], area["front_y"], target_xyz[2])
        else:
            entry = (target_xyz[0], target_xyz[1], max(z, target_xyz[2]))
        into = [(tuple(target_xyz), point)]
        if out and not (yield from self.go(out)):
            return False
        point = here.yaw + wrap(point - here.yaw)
        q = self.w.ik(entry, point, self.w.command)
        if not (yield from self.joint_move(q, vias=self.hubs([here.yaw + wrap(point - here.yaw) / 2, point, here.yaw]))):
            return False
        return (yield from self.go(into))

    def hubs(self, points):
        """Lazily, joints at each of the config's hub positions with the fingers pointing at each
        of `points`: the via points tried, in order, when a joint-space move's straight line
        would collide."""
        for xyz in self.cfg["hubs"]:
            for point in points:
                try:
                    yield self.w.ik(xyz, point, self.w.command)
                except PlanError:
                    pass

    def joint_move(self, q, vias=()):
        """A straight line in joint space to q, checked for collisions every JOINT_STEP rad; if
        it collides, two lines through the first of `vias` that clears."""
        def line(a, b):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / JOINT_STEP)))
            return [a + (b - a) * (i / n) for i in range(1, n + 1)]
        start = self.w.command
        error = None
        for via in [None, *vias]:
            qs = line(start, q) if via is None else line(start, via) + line(via, q)
            try:
                self.check(qs, self.w.finger())
                return (yield from self.follow(qs))
            except PlanError as e:
                error = error or e
        raise error

    def line(self, xyz, point=None, stop=None):
        """A straight move to xyz, keeping the pointing unless one is given; `stop` as in follow."""
        point = self.w.tcp_pose().yaw if point is None else point
        return (yield from self.go([(tuple(xyz), point)], stop))
