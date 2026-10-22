import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import py_trees
import robot_kinematics as kin
from lab_motion import LabMotion,HOME,DT
from chem_lab_bt.lab_behavior_tree import Blackboard,Settings,AsyncAction,Status,build_tree,main
from chem_lab_bt.mujoco_backend import MuJoCoBackend

class CountingBackend:
    def __init__(self):self.starts=[];self.cancels=[];self.polls=0
    def start_action(self,name,plate_id):self.starts.append((name,plate_id))
    def poll_action(self,name):
        self.polls+=1
        return Status.RUNNING if self.polls==1 else Status.SUCCESS
    def cancel_action(self,name):self.cancels.append(name)

class AsyncNodeTests(unittest.TestCase):
    def test_start_once_and_success_cache(self):
        state=Blackboard();backend=CountingBackend();node=AsyncAction('Move',state,backend)
        node.tick_once();self.assertEqual(node.status,Status.RUNNING)
        node.tick_once();self.assertEqual(node.status,Status.SUCCESS)
        node.stop(Status.INVALID);node.tick_once()
        self.assertEqual(backend.starts,[('Move',1)])
        self.assertEqual(backend.polls,2)

    def test_invalidation_cancels_running_action(self):
        state=Blackboard();backend=CountingBackend();node=AsyncAction('Move',state,backend)
        node.tick_once();node.stop(Status.INVALID)
        self.assertEqual(backend.cancels,['Move'])
        self.assertNotIn('Move',state.completed)

    def test_original_mock_fault_scenarios(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(['--plates','4','--scenario','chem_lab_bt/scenarios.json','--compact'])
        self.assertTrue(result['success'])
        self.assertEqual(result['plates_finished'],4)
        self.assertEqual(result['measured'],3)
        self.assertEqual(result['rejected'],1)
        self.assertEqual(result['preventive_interventions'],1)
        self.assertEqual(result['recoveries_completed']['alignment'],1)
        self.assertEqual(result['recoveries_completed']['placement'],1)

class MuJoCoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.state=Blackboard(compact_log=True)
        self.backend=MuJoCoBackend(self.state,Settings(plates=1),headless=True)
        self.backend.event=lambda event,action,**extra:self.backend.events.append(dict(event=event,action=action,**extra))
        self.motion=self.backend.motion

    def tearDown(self):self.backend.close()

    def test_poll_does_not_advance_physics_and_cancel_does_not_teleport(self):
        before=self.motion.data.qpos.copy();margins=self.motion.model.geom_margin.copy()
        self.backend.start_action('SourceHigh',1)
        self.assertEqual(self.motion.data.time,0.)
        self.assertEqual(self.backend.poll_action('SourceHigh'),Status.RUNNING)
        self.assertEqual(self.motion.data.time,0.)
        np.testing.assert_array_equal(self.motion.data.qpos,before)
        self.backend.cancel_action('SourceHigh')
        np.testing.assert_array_equal(self.motion.data.qpos,before)
        np.testing.assert_array_equal(self.motion.model.geom_margin,margins)
        self.assertIsNone(self.backend.task)
        self.backend.advance();self.assertAlmostEqual(self.motion.data.time,DT)

    def test_duplicate_start_is_rejected(self):
        self.backend.start_action('SourceHigh',1)
        with self.assertRaises(RuntimeError):self.backend.start_action('SourceHigh',1)

    def test_priority_action_cancels_previous_task(self):
        self.backend.start_action('SourceHigh',1)
        self.backend.poll_action('SourceHigh')
        self.backend.start_action('StabiliseGrasp',1)
        self.assertEqual(self.backend.active_name,'StabiliseGrasp')
        self.assertIn({'event':'cancel','action':'SourceHigh'},self.backend.events)
        self.assertEqual(self.backend.poll_action('StabiliseGrasp'),Status.FAILURE)
        self.assertIn('lost plate',self.state.last_error)

    def test_unverified_grasp_does_not_succeed(self):
        self.motion.initialized=True;self.backend.select_plate();self.motion.prepare_pick('source')
        self.backend.start_action('VerifySourceGrasp',1)
        self.assertEqual(self.backend.poll_action('VerifySourceGrasp'),Status.FAILURE)
        self.assertFalse(self.state.carrying_plate)

    def test_unknown_physical_action_fails_safely(self):
        node=AsyncAction('UnimplementedInstrumentAction',self.state,self.backend)
        node.tick_once()
        self.assertEqual(node.status,Status.FAILURE)
        self.assertTrue(self.state.emergency_latched)
        self.assertNotIn(node.name,self.state.completed)

    def test_reader_start_is_idempotent_across_wait_cancellation(self):
        with patch.object(self.motion,'verify_placement',return_value=True):
            self.backend.start_measurement()
            original=self.backend.measurements[self.motion.carried]
            self.backend.advance();self.backend.start_measurement()
        self.assertEqual(self.backend.measurements[self.motion.carried],original)
        self.backend.start_action('WaitForReadingComplete',1)
        self.assertEqual(self.backend.poll_action('WaitForReadingComplete'),Status.RUNNING)
        self.backend.cancel_action('WaitForReadingComplete')
        self.backend.start_action('WaitForReadingComplete',1)
        for _ in range(120):
            self.backend.advance()
            if self.backend.poll_action('WaitForReadingComplete')==Status.SUCCESS:break
        else:self.fail('Simulated reader never completed')
        self.assertTrue(self.state.reader_complete)
        self.assertEqual(self.backend.measurement_starts[self.motion.carried],1)
        self.assertEqual(self.backend.reader_mode,'simulated_timer')

    def test_physical_tree_has_implemented_unique_atomic_actions(self):
        tree=build_tree(self.state,self.backend,Settings(plates=1))
        names=[]
        def visit(node):
            if isinstance(node,AsyncAction):names.append(node.name)
            for child in getattr(node,'children',[]):visit(child)
        visit(tree.root.children[-1])
        self.assertEqual(len(names),len(set(names)))
        self.assertFalse(set(names)-set(self.backend.actions))
        for earlier,later in [('SourceClose','SourceLift'),('HomeBeforeStorage','ReturnToStorageSlot'),
                              ('ReturnToStorageSlot','AlignWithStorageSlot'),('AlignWithStorageSlot','StorageDescend'),
                              ('StorageDescend','PlacePlateInSlot'),('HomeAfterStorage','CommitStoredPlate')]:
            self.assertLess(names.index(earlier),names.index(later))
        self.assertGreater(len(names),40)

    def test_stop_retains_a_real_physical_grasp(self):
        from test_contact_physics import ContactPhysicsTests
        case=ContactPhysicsTests();case.setUp();case.place_test_plate_at_pickup()
        p=case.data.site_xpos[kin.site_id(case.model,'ot2_pickup')].copy();above=p+[0,0,.06]
        kin.set_angles(case.model,case.data,case.info,case.aligned_pose(above));case.runtime.reset_targets();case.run_for(1)
        case.runtime.set_arm_target(case.aligned_pose(p));case.run_for(3)
        for _ in range(100):case.controller.jog_fingers(1,.02);case.run_for(.02)
        case.runtime.set_arm_target(case.aligned_pose(above));case.run_for(4)
        self.assertTrue(case.controller.held)
        m=self.motion
        for key in ('model','data','info','tcp','runtime','controller'):setattr(m,key,getattr(case,key))
        m.carried=case.controller.plate
        before=m.data.qpos.copy();z=float(m.data.xpos[m.carried,2])
        m.stop()
        np.testing.assert_array_equal(m.data.qpos,before)
        np.testing.assert_allclose(m.controller.target,[0,0])
        case.run_for(.5)
        self.assertTrue(m.holding(m.carried))
        self.assertLess(abs(float(m.data.xpos[m.carried,2])-z),.002)

if __name__=='__main__':unittest.main()
