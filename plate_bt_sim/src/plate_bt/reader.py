"""The plate reader (Allan's VersaMax) as the plate workflow uses it: a drawer that opens to take
a plate, closes for the measurement, and opens again to give the plate back.

`SimReader` drives the drawer the scene builds (scene._drawer): each tick it moves the drawer's
actuator target along a minimum-jerk profile, so a plate on the tray rides in and out without
sliding. On the real cell, a class with the same methods calls the reader's own interface."""
from plate_bt.scene import DRAWER_ACTUATOR, DRAWER_JOINT

SETTLED_M = 0.0005           # the drawer is at a position when within this of it...
SETTLED_M_S = 0.002          # ...and slower than this


class SimReader:
    def __init__(self, world, cfg):
        self.w, self.cfg = world, cfg["reader"]
        m = world.model
        self.act = m.actuator(DRAWER_ACTUATOR).id
        j = m.joint(DRAWER_JOINT).id
        self.qadr, self.vadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.open_at, self.closed_at = 0.0, self.cfg["travel_m"]
        self.profile = None          # (start time, from, to) of the drawer's current move
        self.target = self.closed_at
        self.put(self.closed_at)
        world.devices.append(self)

    # ---- the reader's interface --------------------------------------------------------------
    def open(self):
        self._go(self.open_at)

    def close(self):
        self._go(self.closed_at)

    def is_open(self):
        return self.target != self.closed_at and self._settled()

    def is_closed(self):
        return self.target == self.closed_at and self._settled()

    def position(self):
        """How far the drawer is drawn in from open (m)."""
        return float(self.w.data.qpos[self.qadr])

    # ---- simulation ----------------------------------------------------------------------------
    def put(self, q):
        """Simulation helper: the drawer at q, at rest, with no move under way."""
        self.w.data.qpos[self.qadr] = q
        self.w.data.qvel[self.vadr] = 0
        self.w.data.ctrl[self.act] = q
        self.target, self.profile = q, None

    def update(self):
        """One tick: the actuator target along the current move's profile (SimWorld.step)."""
        if self.profile is None:
            return
        t0, a, b = self.profile
        tau = min((self.w.time - t0 + self.w.dt) / self.cfg["move_s"], 1.0)
        self.w.data.ctrl[self.act] = a + (b - a) * (10 - 15 * tau + 6 * tau ** 2) * tau ** 3
        if tau >= 1.0:
            self.profile = None

    def _go(self, q):
        self.profile = (self.w.time, float(self.w.data.ctrl[self.act]), q)
        self.target = q

    def _settled(self):
        return (self.profile is None and abs(self.position() - self.target) < SETTLED_M
                and abs(float(self.w.data.qvel[self.vadr])) < SETTLED_M_S)

    def moving(self):
        return not self._settled()

    def wait_settled(self):
        """Generator: tick until the drawer has finished its move."""
        while not self._settled():
            yield
        return True
