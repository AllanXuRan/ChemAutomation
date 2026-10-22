"""Incremental world-frame TCP jogging, using only the current IK branch."""
import mujoco
import numpy as np
import robot_kinematics as kin

KEY_DIRECTIONS = {
    'a': (-1, 0, 0), 'd': (1, 0, 0),
    'w': (0, 1, 0), 's': (0, -1, 0),
    'control_l': (0, 0, -1), 'control_r': (0, 0, -1),
    'space': (0, 0, 1),
}


ROTATION_KEYS = {'q': -1, 'e': 1}

def direction(keys):
    vectors = {KEY_DIRECTIONS[k] for k in keys if k in KEY_DIRECTIONS}
    result = np.sum(list(vectors), axis=0) if vectors else np.zeros(3)
    length = np.linalg.norm(result)
    return result / max(1., length)


class CartesianJog:
    def __init__(self, model, info, tcp, level=False):
        self.model, self.info, self.tcp = model, info, tcp
        self.level = level
        self.scratch = mujoco.MjData(model)

    def solve(self, data, keys, speed, dt):
        if not np.isfinite(speed) or not 0 < speed <= .2:
            raise ValueError('键盘速度需在 0～0.2 m/s 之间')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('无效时间间隔')
        delta = direction(keys) * speed * min(dt, .1)
        rotation = sum(ROTATION_KEYS.get(key, 0) for key in keys)
        if np.linalg.norm(delta) == 0 and rotation == 0:
            return None, None, None
        self.scratch.qpos[:] = data.qpos
        mujoco.mj_forward(self.model, self.scratch)
        q = kin.angles(self.scratch, self.info)
        if rotation:
            q[4] += rotation * np.radians(10.) * min(dt, .1)
            q = kin.level_flange(self.model, self.info, q) if self.level else kin.clip_angles(self.model, self.info, q)
            kin.set_angles(self.model, self.scratch, self.info, q)
        target = self.scratch.site_xpos[self.tcp].copy() + delta
        if np.linalg.norm(delta) == 0:
            return q, target, 0.

        result, error, _ = kin.jac_ik(
            self.model, self.scratch, self.info, self.tcp, target, q,
            tol=.00002, max_iters=30, damping=.01, level=self.level)
        # Do not switch IK branches or accumulate unreachable target drift.
        if error > max(.00005, np.linalg.norm(delta)*.15) or np.max(np.abs(result-q)) > .08:
            return None, target, error
        return result, target, error
