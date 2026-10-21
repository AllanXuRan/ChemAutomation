import unittest

import mujoco
import numpy as np

import robot_kinematics as kin
from keyboard_ee import CartesianJog


class FlangeLevelTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(kin.SCENE))
        self.data = mujoco.MjData(self.model)
        self.info = kin.arm_info(self.model)
        self.tcp = kin.site_id(self.model, 'g2_tcp')
        self.flange = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link_eef')

    def assert_level(self):
        normal = self.data.xmat[self.flange].reshape(3, 3)[:, 2]
        # MJCF body quaternions are rounded, so allow 0.001 degrees.
        np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-5)

    def test_level_through_interpolation_and_yaw(self):
        a = kin.level_flange(self.model, self.info, kin.Q0)
        b = kin.level_flange(self.model, self.info, [.8, .3, -1.4, 0, 1.2])
        for fraction in np.linspace(0, 1, 31):
            kin.set_angles(self.model, self.data, self.info, a*(1-fraction) + b*fraction)
            self.assert_level()
        with self.assertRaises(ValueError):
            kin.level_flange(self.model, self.info, [0, 2, .1, 0, 0])

    def test_pickup_ik_and_keyboard_path(self):
        kin.set_angles(self.model, self.data, self.info,
                       kin.level_flange(self.model, self.info, kin.Q0))
        target = self.data.site_xpos[kin.site_id(self.model, 'ot2_pickup')].copy()
        q, error, _ = kin.solve_ik(self.model, self.data, self.info,
                                  self.tcp, target, .003, level=True)
        self.assertLess(error, .003)
        self.assert_level()
        jog = CartesianJog(self.model, self.info, self.tcp, level=True)
        for _ in range(30):
            q, _, _ = jog.solve(self.data, {'space'}, .01, .04)
            self.assertIsNotNone(q)
            kin.set_angles(self.model, self.data, self.info, q)
            self.assert_level()


if __name__ == '__main__':
    unittest.main()
