import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

import robot_kinematics as kin
from physics_runtime import PhysicsRuntime, load_physics_model
from state_capture import StateRecorder


class StateCaptureTests(unittest.TestCase):
    def test_records_actual_state_and_appends_without_mutation(self):
        model = load_physics_model(kin.SCENE)
        data = mujoco.MjData(model)
        info = kin.arm_info(model)
        tcp = kin.site_id(model, 'g2_tcp')
        kin.set_angles(model, data, info, kin.level_flange(model, info, kin.Q0))
        runtime = PhysicsRuntime(model, data, info)
        target = runtime.arm_target.copy()
        target[0] += .2
        runtime.set_arm_target(target)
        before = data.qpos.copy()
        with tempfile.TemporaryDirectory(dir=kin.ROOT) as directory:
            recorder = StateRecorder(Path(directory) / 'states.jsonl')
            first = recorder.append(runtime, tcp, '取板上方')
            np.testing.assert_array_equal(data.qpos, before)
            self.assertAlmostEqual(first['joints']['joint1']['position_rad'], before[0])
            self.assertAlmostEqual(first['joints']['joint1']['target_rad'], target[0])
            np.testing.assert_allclose(list(first['tcp_xyz_world_m'].values()), data.site_xpos[tcp])
            self.assertEqual(len(first['joints']), 7)
            self.assertEqual(len(first['all_plates']),4)
            for body in runtime.controller.plate_bodies:
                np.testing.assert_allclose(first['all_plates'][model.body(body).name]['xyz_world_m'],data.xpos[body])
            runtime.advance(.04)
            second = recorder.append(runtime, tcp, '')
            records = [json.loads(line) for line in recorder.path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records, [first, second])
            self.assertEqual([r['sequence'] for r in records], [1, 2])
            self.assertEqual(records[0]['name'], '取板上方')
            self.assertEqual(records[1]['name'], 'step_002')
            self.assertGreater(second['simulation_time_s'], first['simulation_time_s'])


if __name__ == '__main__':
    unittest.main()
