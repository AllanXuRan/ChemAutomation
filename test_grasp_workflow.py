import unittest
import mujoco
import robot_kinematics as kin
from grasp_workflow import validate_steps


class WorkflowValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(kin.SCENE))
        self.info = kin.arm_info(self.model)

    def test_validation(self):
        steps = [dict(name=a, action=a, joints_rad=kin.Q0.tolist(), duration_s=1.)
                 for a in ('move', 'grasp', 'move', 'release')]
        validate_steps(self.model, self.info, steps)
        steps[1]['joints_rad'] = None
        with self.assertRaises(ValueError):
            validate_steps(self.model, self.info, steps)
        steps[1]['joints_rad'] = kin.Q0.tolist()
        steps[-1]['action'] = 'move'
        with self.assertRaises(ValueError):
            validate_steps(self.model, self.info, steps)
        steps[-1]['action'] = 'release'
        steps[0]['duration_s'] = float('nan')
        with self.assertRaises(ValueError):
            validate_steps(self.model, self.info, steps)


if __name__ == '__main__':
    unittest.main()
