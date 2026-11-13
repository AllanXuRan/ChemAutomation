"""What the robot sees, as placeholders.

`Perception` stands in for the cameras: it returns the simulation's true poses of plates, the
reader's tray pocket and the top of each pile. `ScriptedDetector` stands in for the
formulation anomaly detector: it reports the label a scenario gives a plate. On the real robot
both are replaced by models that look at images; the calls stay the same."""
import numpy as np

from plate_bt import scene
from plate_bt.world import PLATE_HALF, Pose

PILE_RADIUS = 0.03           # a plate whose centre is this close to a pile's axis is on the pile


class Perception:
    def __init__(self, world, cfg):
        self.w, self.cfg = world, cfg
        m = world.model
        self.insert = m.geom("versamax_tray_insert").id

    def plate(self, k):
        """(pose, tilt in degrees) of plate k."""
        return self.w.plate_pose(k)

    def tray_pocket(self):
        """Where a plate sits in the tray: centred on the insert, on its top face, long axis
        along the tray (the area's plate_yaw)."""
        d = self.w.data
        g = self.insert
        R = d.geom_xmat[g].reshape(3, 3)
        top = d.geom_xpos[g][2] + self.w.model.geom_size[g][2]
        yaw = np.arctan2(R[1, 0], R[0, 0]) + np.radians(self.cfg["areas"]["tray"]["plate_yaw"])
        return Pose((d.geom_xpos[g][0], d.geom_xpos[g][1], top + PLATE_HALF[2]), float(yaw))

    def pile_top(self, area, exclude=None):
        """(pose of the plate on top of the area's pile, its index), leaving out plate `exclude`.
        An empty pile gives the pose a plate under the first one would have, and None."""
        a = self.cfg["areas"][area]
        pad_top = scene.TABLE_TOP + self.cfg["pad_size"][2]
        axis = np.asarray(a["xy"], float)
        best, best_k = Pose((*axis, pad_top - PLATE_HALF[2]), np.radians(a["plate_yaw"])), None
        for k in range(len(self.w.plates)):
            if k == exclude:
                continue
            p, _ = self.w.plate_pose(k)
            if np.hypot(*(p.xyz[:2] - axis)) < PILE_RADIUS and p.xyz[2] > best.xyz[2] + 0.005:
                best, best_k = p, k
        return best, best_k


class ScriptedDetector:
    """The formulation anomaly detector's placeholder: `labels` maps a plate number (1-based, as
    Allan's blackboard counts) to 'bubble' or 'tip'; every other plate is 'ok'."""

    def __init__(self, labels):
        self.labels = {int(k): v for k, v in (labels or {}).items()}

    def inspect(self, plate_id, image=None):
        """(label, score) for the plate; a real detector would look at `image`."""
        label = self.labels.get(int(plate_id), "ok")
        return label, (0.95 if label != "ok" else 0.02)
