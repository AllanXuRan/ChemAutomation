"""Headless regression checks, including rebuilding the complete scene in isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import build_lab_scene as lab
import prepare_physics_scene as physics
import setup_tcp_cameras as cameras
import robot_kinematics as kin
import layout_editor
from grasp_workflow import validate_record, WORKFLOW_ACTIONS


class ProjectChecks(unittest.TestCase):
    def test_layout_mouse_select(self):
        centers = {'ot2': (.2, .2), 'versamax': (1., 1.)}
        editor = SimpleNamespace(get_center=centers.__getitem__, sel=Mock(), reset_fields=Mock())
        x, y = layout_editor.xy_screen(centers['ot2'])
        layout_editor.LayoutEditor.mouse_down(editor, SimpleNamespace(x=x, y=y, state=0))
        editor.sel.set.assert_called_once_with('ot2')
        self.assertEqual(editor.drag[0], 'ot2')

    def test_invalid_slots_abort_save(self):
        editor = SimpleNamespace(apply_fields=lambda: True, apply_slots=lambda: False,
                                 apply_temperature_angles=Mock())
        self.assertFalse(layout_editor.LayoutEditor.save(editor))
        editor.apply_temperature_angles.assert_not_called()

    def test_full_scene_build(self):
        config = json.loads(lab.CONFIG.read_text(encoding='utf-8'))
        lab.apply_geometry_defaults(config)
        lab.canonicalize_facing(config)
        lab.validate_geometry(config)
        with tempfile.TemporaryDirectory(prefix='scene_check_', dir=lab.ROOT) as folder:
            folder = Path(folder)
            scene, dynamic, final = [folder / name for name in ('scene.xml', 'physics.xml', 'final.xml')]
            with contextlib.redirect_stdout(io.StringIO()):
                lab.build(config, lab.URDF, scene, config['plate']['start_at'])
                physics.build(scene, physics.DEFAULT_CONFIG, dynamic)
                cameras.build(dynamic, final, cameras.CONFIG)
                report = cameras.report(final, export=None)
            model = mujoco.MjModel.from_xml_path(str(final))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for name, xyz in lab.targets(config).items():
                np.testing.assert_allclose(data.site_xpos[kin.site_id(model, name)], xyz, atol=1e-7)
            self.assertEqual(len(report['cameras']), 3)
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'reaction_plate_free'), 0)

    def test_invalid_layout_does_not_persist_migration(self):
        config = json.loads(lab.CONFIG.read_text(encoding='utf-8'))
        config['ot2'].pop('show_slot_outlines', None)
        config['ot2']['position_xy'] = [float('nan'), 0.5]
        with tempfile.TemporaryDirectory(prefix='layout_check_', dir=lab.ROOT) as folder:
            path = Path(folder) / 'layout.json'
            original = json.dumps(config)
            path.write_text(original, encoding='utf-8')
            with self.assertRaises(ValueError):
                lab.load_config(path)
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_final_ik_iteration_is_retained(self):
        model = mujoco.MjModel.from_xml_path(str(kin.SCENE))
        data = mujoco.MjData(model)
        info = kin.arm_info(model)
        tcp = kin.site_id(model, 'g2_tcp')
        kin.set_angles(model, data, info, kin.Q0)
        target = data.site_xpos[tcp].copy() + [.001, 0, 0]
        q, error, iterations = kin.jac_ik(model, data, info, tcp, target, kin.Q0, tol=1e-8, max_iters=1)
        self.assertLess(error, .001)
        self.assertEqual(iterations, 1)
        np.testing.assert_allclose(kin.angles(data, info), q)
        self.assertAlmostEqual(np.linalg.norm(data.site_xpos[tcp]-target), error)
        with self.assertRaises(ValueError):
            kin.jac_ik(model, data, info, tcp, [float('nan'), 0, 0], kin.Q0)

    def test_workflow_record_checked_before_loading(self):
        model = mujoco.MjModel.from_xml_path(str(kin.SCENE))
        info = kin.arm_info(model)
        record = dict(version=1, destination='versamax_load', clearance_m=.1,
                      steps=[dict(name=action, action=action, joints_rad=kin.Q0.tolist(), duration_s=1.)
                             for action in WORKFLOW_ACTIONS])
        validate_record(model, info, record)
        for bad in (float('nan'), -.1):
            record['clearance_m'] = bad
            with self.assertRaises(ValueError):
                validate_record(model, info, record)
        record['clearance_m'] = .1
        record['destination'] = 'nonexistent_site'
        with self.assertRaises(ValueError):
            validate_record(model, info, record)


if __name__ == '__main__':
    unittest.main()
