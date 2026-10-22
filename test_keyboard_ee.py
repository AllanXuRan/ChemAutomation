import unittest
import mujoco
import numpy as np
import robot_kinematics as kin
from keyboard_ee import CartesianJog, direction


class KeyboardJogTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(kin.SCENE))
        self.data = mujoco.MjData(self.model)
        self.info = kin.arm_info(self.model)
        self.tcp = kin.site_id(self.model, 'g2_tcp')
        kin.set_angles(self.model, self.data, self.info, kin.Q0)
        self.jog = CartesianJog(self.model, self.info, self.tcp)

    def test_mapping_and_diagonal_speed(self):
        expected = {'a': [-1, 0, 0], 'd': [1, 0, 0], 'w': [0, 1, 0],
                    's': [0, -1, 0], 'control_l': [0, 0, -1], 'space': [0, 0, 1]}
        for key, vector in expected.items():
            np.testing.assert_allclose(direction({key}), vector)
        np.testing.assert_allclose(direction({'a', 'd'}), np.zeros(3))
        self.assertAlmostEqual(np.linalg.norm(direction({'w', 'd', 'space'})), 1.)
        np.testing.assert_allclose(direction({'control_l', 'control_r'}), [0, 0, -1])

    def test_submillimeter_commands_and_no_live_mutation(self):
        for key in ('a', 'd', 'w', 's', 'control_l', 'space'):
            with self.subTest(key=key):
                before = self.data.qpos.copy()
                p = self.data.site_xpos[self.tcp].copy()
                q, target, error = self.jog.solve(self.data, {key}, .01, .04)
                self.assertIsNotNone(q)
                np.testing.assert_array_equal(self.data.qpos, before)
                kin.set_angles(self.model, self.data, self.info, q)
                np.testing.assert_allclose(self.data.site_xpos[self.tcp], p + direction({key})*.0004, atol=.00005)
                self.assertLess(error, .00005)

    def test_flange_rotation_keys(self):
        q0 = kin.level_flange(self.model, self.info, kin.Q0)
        kin.set_angles(self.model, self.data, self.info, q0)
        jog = CartesianJog(self.model, self.info, self.tcp, level=True)
        before = self.data.qpos.copy()
        for key, sign in (('q', -1), ('e', 1)):
            q, target, error = jog.solve(self.data, {key}, .03, .04)
            self.assertAlmostEqual(q[4]-q0[4], sign*np.radians(10)*.04)
            np.testing.assert_allclose(q[:4], q0[:4])
            np.testing.assert_array_equal(self.data.qpos, before)
            kin.set_angles(self.model, self.data, self.info, q)
            np.testing.assert_allclose(self.data.site_xpos[self.tcp], target)
            kin.set_angles(self.model, self.data, self.info, q0)
        self.assertEqual(jog.solve(self.data, {'q', 'e'}, .03, .04), (None, None, None))
        q, _, _ = jog.solve(self.data, {'e'}, .03, 10.)
        self.assertAlmostEqual(q[4]-q0[4], np.radians(1))
        upper = q0.copy()
        upper[4] = self.model.jnt_range[self.info[4][0], 1]-1e-7
        kin.set_angles(self.model, self.data, self.info, upper)
        q, _, _ = jog.solve(self.data, {'e'}, .03, .04)
        self.assertLessEqual(q[4], self.model.jnt_range[self.info[4][0], 1])

    def test_long_frame_bounded_and_idle(self):
        q, target, error = self.jog.solve(self.data, {'d'}, .03, 10.)
        self.assertIsNotNone(q)
        self.assertAlmostEqual(np.linalg.norm(target-self.data.site_xpos[self.tcp]), .003)
        self.assertEqual(self.jog.solve(self.data, set(), .03, .04), (None, None, None))
        with self.assertRaises(ValueError):
            self.jog.solve(self.data, {'d'}, float('nan'), .04)


if __name__ == '__main__':
    unittest.main()
