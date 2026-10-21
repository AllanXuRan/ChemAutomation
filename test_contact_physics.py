import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

import robot_kinematics as kin
from physics_runtime import PhysicsRuntime, load_physics_model
from grasp_workflow import WorkflowPanel


class ContactPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.model = load_physics_model(kin.SCENE)
        self.data = mujoco.MjData(self.model)
        self.info = kin.arm_info(self.model)
        self.tcp = kin.site_id(self.model, 'g2_tcp')
        kin.set_angles(self.model, self.data, self.info,
                       kin.level_flange(self.model, self.info, kin.Q0))
        self.runtime = PhysicsRuntime(self.model, self.data, self.info)
        self.controller = self.runtime.controller

    def run_for(self, seconds):
        for _ in range(round(seconds / .02)):
            self.runtime.advance(.02)
            self.assertTrue(np.isfinite(self.data.qpos).all())
            normal = self.data.xmat[self.model.body('link_eef').id].reshape(3, 3)[:, 2]
            self.assertLess(np.degrees(np.arccos(np.clip(-normal[2], -1, 1))), .1)

    def aligned_pose(self, xyz):
        """Find a level pose with gripper pads parallel to the plate's long edges."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        q, error, _ = kin.solve_ik(self.model, scratch, self.info, self.tcp,
                                   xyz, .00001, level=True, flange_yaw=0.)
        self.assertLess(error, .00001)
        return q

    def test_plate_settles_on_support(self):
        initial_z = self.data.xpos[self.controller.plate, 2]
        self.run_for(2)
        z = self.data.xpos[self.controller.plate, 2]
        self.assertGreater(initial_z-z, .003)
        self.assertLess(initial_z-z, .01)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertAlmostEqual(self.data.time, 2)
        self.assertFalse(np.any(self.data.warning.number))

    def test_finger_shank_collision_blocks_push_and_allows_retreat(self):
        initial = self.runtime.arm_target.copy()
        target = initial.copy()
        target[0] += .25
        kin.set_angles(self.model, self.data, self.info, target)
        shank = self.model.geom('left_finger_shank_collision').id
        obstacle = self.data.geom_xpos[shank].copy()
        source = ET.parse(kin.SCENE).getroot()
        ET.SubElement(source.find('worldbody'), 'geom', name='test_obstacle', type='sphere',
                      size='.012', pos=' '.join(str(v) for v in obstacle))
        with tempfile.TemporaryDirectory(dir=kin.ROOT) as directory:
            scene = Path(directory) / 'obstacle.xml'
            ET.ElementTree(source).write(scene, encoding='utf-8')
            model = load_physics_model(scene)
        data = mujoco.MjData(model)
        info = kin.arm_info(model)
        kin.set_angles(model, data, info, initial)
        runtime = PhysicsRuntime(model, data, info)
        runtime.set_arm_target(target)
        peak = 0.
        for _ in range(200):
            runtime.advance(.02)
            depths = runtime.forbidden_contacts(data)
            peak = max(peak, max(depths.values(), default=0.))
        self.assertIn('test_obstacle', runtime.blocked)
        self.assertLess(peak, .002)
        self.assertLess(abs(kin.angles(data, info)[0]-initial[0]), .24)
        runtime.set_arm_target(initial)
        for _ in range(200):
            runtime.advance(.02)
        self.assertEqual(runtime.blocked, '')
        self.assertLess(np.max(np.abs(kin.angles(data, info)-initial)), .005)
        self.assertFalse(np.any(data.warning.number))

    def test_visible_supports_are_collidable_but_labels_are_not(self):
        for name in ('ot2_waiting_pad', 'ot2_stack_pad', 'ot2_stack_plate_1',
                     'versamax_tray_insert', 'versamax_recess_floor'):
            self.assertEqual(self.model.geom_contype[self.model.geom(name).id], 1)
        label = self.model.geom('ot2_slot_outline_1_y_-1').id
        self.assertEqual(self.model.geom_contype[label], 0)

    def test_targets_do_not_teleport_and_empty_close_does_not_grasp(self):
        before = self.data.qpos.copy()
        target = self.runtime.arm_target.copy()
        target[0] += .1
        self.runtime.set_arm_target(target)
        self.controller.jog_fingers(1, .04)
        np.testing.assert_array_equal(self.data.qpos, before)
        np.testing.assert_allclose(self.controller.target, [-.0392, .0392])
        held_target = self.controller.target.copy()
        self.controller.jog_fingers(0, .04)
        np.testing.assert_array_equal(self.controller.target, held_target)
        self.controller.attach()
        self.run_for(2)
        self.assertFalse(self.controller.held)
        self.assertGreater(self.data.time, 1.9)
        self.controller.reset()
        self.assertEqual(self.data.time, 0.)
        np.testing.assert_allclose(self.data.qpos[self.controller.finger_q], [-.04, .04])
        self.run_for(.1)
        self.assertFalse(np.any(self.data.warning.number))

    def test_contact_lift_hold_and_release(self):
        pickup = self.data.site_xpos[kin.site_id(self.model, 'ot2_pickup')].copy()
        # Initialize above the plate, avoiding an unrelated transit through equipment.
        above = pickup + [0, 0, .06]
        kin.set_angles(self.model, self.data, self.info, self.aligned_pose(above))
        self.runtime.reset_targets()
        self.run_for(1)
        self.runtime.set_arm_target(self.aligned_pose(pickup))
        self.run_for(3)
        for _ in range(100):
            self.controller.jog_fingers(1, .02)
            self.run_for(.02)
        self.assertTrue(self.controller.held)
        self.runtime.set_arm_target(self.aligned_pose(above))
        self.run_for(4)
        self.assertGreater(self.data.xpos[self.controller.plate, 2], pickup[2]+.04)
        self.assertTrue(self.controller.held)
        for _ in range(100):
            self.controller.jog_fingers(-1, .02)
            self.run_for(.02)
        self.assertFalse(self.controller.held)
        self.assertLess(self.data.xpos[self.controller.plate, 2], pickup[2]+.01)
        self.assertFalse(np.any(self.data.warning.number))

    def test_physical_workflow_waits_for_contact_then_completes(self):
        pickup = self.data.site_xpos[kin.site_id(self.model, 'ot2_pickup')].copy()
        above = pickup + [0, 0, .06]
        poses = [self.aligned_pose(p) for p in
                 [above, pickup, above, above+[.01, 0, 0],
                  pickup+[.01, 0, 0], above+[.01, 0, 0]]]
        kin.set_angles(self.model, self.data, self.info, poses[0])
        self.runtime.reset_targets()
        self.run_for(1)
        viewer = SimpleNamespace(lock=nullcontext)
        with patch('grasp_workflow.tk.StringVar'):
            panel = WorkflowPanel(None, self.model, self.data, self.info, self.tcp,
                                  viewer, self.runtime.set_arm_target, lambda *a: None,
                                  lambda: None, controller=self.controller)
        for step, q in zip(panel.steps, poses):
            step.update(joints_rad=q.tolist(), duration_s=2.)
        panel.start(True)
        now = [0.]
        with patch('grasp_workflow.time.perf_counter', side_effect=lambda: now[0]):
            for _ in range(1600):
                if not panel.running:
                    break
                panel.advance()
                self.run_for(.02)
                now[0] += .02
        self.assertFalse(panel.running)
        self.assertEqual(panel.index, 6)
        self.assertFalse(self.controller.held)
        self.assertLess(self.data.xpos[self.controller.plate, 2], pickup[2]+.01)


if __name__ == '__main__':
    unittest.main()
