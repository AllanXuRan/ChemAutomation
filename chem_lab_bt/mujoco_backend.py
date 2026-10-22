"""Action adapter for one persistent MuJoCo runtime and the existing BT loop."""
from __future__ import annotations
import json
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from lab_motion import LabMotion, HOME, DT, ROOT
from trajectory_recording import export_notebook
from chem_lab_bt.lab_behavior_tree import RobotBackend, Status, announce

class ActionFailure(RuntimeError):
    pass

class MuJoCoBackend(RobotBackend):
    physical_flow=True
    reader_mode='simulated_timer'

    def __init__(self,state,settings,*,headless=False,speed=4.,motion=None):
        self.state,self.settings=state,settings
        self.motion=motion if motion is not None else LabMotion()
        self.viewer=None if headless else mujoco.viewer.launch_passive(self.motion.model,self.motion.data)
        if self.viewer is not None:
            self.viewer.cam.lookat[:]=[0.,.38,.20]
            self.viewer.cam.distance=1.8;self.viewer.cam.azimuth=115;self.viewer.cam.elevation=-28
        self.speed=speed
        self.active_name=None;self.task=None
        self.expected_carrying=False;self.grasp_reference=None;self.contact_lost=0.
        self.events=[];self.last_failed_action=None
        self.measurements={};self.measurement_starts={}
        m=self.motion
        self.actions={
            'InitializeMission':m.initialize,
            'SelectSourcePlate':lambda:self.once(self.select_plate),
            'SourceHigh':m.pick_high,
            'SourceApproach':lambda:m.pick_approach('source'),
            'SourceDescend':lambda:m.pick_descend('source'),
            'SourceClose':lambda:m.close_gripper('source'),
            'SourceLift':lambda:m.lift('source'),
            'VerifySourceGrasp':lambda:self.check(m.verify_grasp('source'),'Source grasp was not verified'),
            'SourceHighTransit':lambda:m.transit('source: clear high transit',[.05,.30,.35]),
            'AluminiumClearTransit':lambda:m.transit('aluminium: clear transit',[-.30,.46,.27]),
            'AluminiumApproachForPlace':lambda:m.place_approach('aluminium'),
            'AluminiumDescendForPlace':lambda:m.place_descend('aluminium'),
            'AluminiumRelease':lambda:m.release('aluminium'),
            'AluminiumRetractAfterPlace':lambda:m.retract('aluminium'),
            'VerifyAluminiumPlaced':lambda:self.check(m.verify_placement('aluminium'),'Aluminium placement was not verified'),
            'HomeAfterAluminium':lambda:m.home('after aluminium: HOME empty'),
            'AluminiumApproachForPick':lambda:m.pick_approach('aluminium'),
            'AluminiumDescendForPick':lambda:m.pick_descend('aluminium'),
            'AluminiumClose':lambda:m.close_gripper('aluminium'),
            'AluminiumLift':lambda:m.lift('aluminium'),
            'VerifyAluminiumGrasp':lambda:self.check(m.verify_grasp('aluminium'),'Aluminium regrasp was not verified'),
            'AluminiumExitTransit':lambda:m.transit('aluminium: exit into clear space',[-.24,.35,.30]),
            'HomeBeforeReader':lambda:m.home('before reader: HOME with plate'),
            'ReaderApproachForPlace':lambda:m.place_approach('reader'),
            'AlignWithReader':lambda:self.check(m.holding(m.carried),'No plate for reader alignment'),
            'VerifyReaderAligned':lambda:self.check(self.reader_aligned(),'Reader approach was not aligned'),
            'ReaderDescendForPlace':lambda:m.place_descend('reader'),
            'LoadPlateIntoReader':lambda:m.release('reader'),
            'ReaderRetractAfterPlace':lambda:m.retract('reader'),
            'VerifyReaderLoaded':lambda:self.check(m.verify_placement('reader'),'Reader placement was not verified'),
            'HomeAfterReader':lambda:m.home('after reader: HOME empty'),
            'StartReaderMeasurement':lambda:self.once(self.start_measurement),
            'WaitForReadingComplete':self.wait_measurement,
            'ReaderApproachForPick':lambda:m.pick_approach('reader'),
            'ReaderDescendForPick':lambda:m.pick_descend('reader'),
            'ReaderClose':lambda:m.close_gripper('reader'),
            'ReaderLift':lambda:m.lift('reader'),
            'VerifyReaderGrasp':lambda:self.check(m.verify_grasp('reader'),'Reader regrasp was not verified'),
            'HomeBeforeStorage':lambda:m.home('before storage: HOME with plate'),
            'ReturnToStorageSlot':m.storage_hover,
            'AlignWithStorageSlot':m.storage_align,
            'VerifyStorageAligned':lambda:self.check(m.verify_storage_aligned(),'Slot2 alignment was not verified'),
            'StorageDescend':lambda:m.place_descend('storage'),
            'PlacePlateInSlot':lambda:m.release('storage'),
            'StorageRetract':lambda:m.retract('storage'),
            'VerifyReturned':lambda:self.check(m.verify_placement('storage'),'Storage placement was not verified'),
            'HomeAfterStorage':lambda:m.home('cycle complete: HOME empty'),
            'CommitStoredPlate':m.commit_cycle,
            'StabiliseGrasp':self.stabilise,
            'RepositionForGrasp':self.recover_grasp,
            'RetractAndRealign':self.recover_alignment,
            'RepositionForPlacement':self.unsupported_placement_recovery,
        }

    @staticmethod
    def once(fn):
        fn()
        if False:yield None

    @staticmethod
    def check(condition,message):
        if not condition:raise ActionFailure(message)
        if False:yield None

    def event(self,event,action,**extra):
        item=dict(event=event,action=action,plate_id=self.state.plate_id,tick=self.state.tick_number,
                  simulation_time_s=float(self.motion.data.time),**extra)
        self.events.append(item)
        (ROOT/'outputs'/'bt_action_events.json').write_text(json.dumps(self.events,indent=2),encoding='utf-8')

    def select_plate(self):
        self.motion.select_plate(self.state.plate_id)
        self.state.selected_body=self.motion.model.body(self.motion.carried).name
        self.state.plate_location='source'
        self.state.reader_complete=False
        self.expected_carrying=False;self.grasp_reference=None;self.contact_lost=0.

    def relative_plate(self):
        m=self.motion
        return m.data.site_xmat[m.tcp].reshape(3,3).T@(m.data.xpos[m.carried]-m.data.site_xpos[m.tcp])

    def refresh(self):
        m,s=self.motion,self.state
        held=m.holding(m.carried)
        s.carrying_plate=held
        s.storage_count=len(m.processed)
        source_xy=np.asarray(m.config['plate']['source_stack']['position_xy_m'])
        s.source_remaining=sum(b not in m.processed and np.linalg.norm(m.data.xpos[b,:2]-source_xy)<.02 and m.data.xpos[b,2]<float(m.config['table']['top_z'])+.08 for b in m.sources)
        if self.expected_carrying:
            self.contact_lost=0. if held else self.contact_lost+DT
            if self.contact_lost>.2:
                s.emergency_latched=True;s.emergency_reason='Selected plate lost bilateral contact'
            drift=0. if self.grasp_reference is None else np.linalg.norm(self.relative_plate()-self.grasp_reference)
            s.slip_risk=float(min(1.,max(.05,drift/.025)))
        else:
            self.contact_lost=0.;s.slip_risk=.05

    def advance(self):
        if self.viewer is not None and not self.viewer.is_running():raise KeyboardInterrupt
        started=time.perf_counter()
        try:
            self.motion.advance(DT)
            self.refresh()
        except (RuntimeError,ValueError) as exc:
            self.state.emergency_latched=True;self.state.emergency_reason=str(exc)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(max(0.,DT/self.speed-(time.perf_counter()-started)))

    def start_action(self,action,plate_id):
        if self.task is not None:
            if action in {'StabiliseGrasp','RepositionForGrasp','RetractAndRealign','RepositionForPlacement'} and action!=self.active_name:
                self.cancel_action(self.active_name)
            else:raise RuntimeError(f'Action already active: {self.active_name}')
        if self.state.emergency_latched:raise RuntimeError('Emergency is latched')
        if plate_id!=self.state.plate_id:raise ValueError('Plate identity mismatch')
        if action not in self.actions:
            self.actions[action]=lambda:self.check(False,f'No validated physical implementation for {action}')
        if action in {'AluminiumRelease','LoadPlateIntoReader','PlacePlateInSlot'}:
            self.expected_carrying=False;self.grasp_reference=None
        self.active_name=action;self.task=self.actions[action]()
        self.state.active_action=action
        self.motion.action_name=action
        self.event('start',action);announce(self.state,f'START   {action}')

    def poll_action(self,action):
        if action!=self.active_name or self.task is None:raise RuntimeError('Poll without matching active action')
        try:
            next(self.task)
            return Status.RUNNING
        except StopIteration:
            self.task=None;self.active_name=None;self.state.active_action=''
            if action in {'SourceClose','AluminiumClose','ReaderClose','VerifySourceGrasp','VerifyAluminiumGrasp','VerifyReaderGrasp'}:
                self.expected_carrying=True;self.grasp_reference=self.relative_plate();self.contact_lost=0.
            locations={'VerifyAluminiumPlaced':'aluminium','VerifyReaderLoaded':'reader','VerifyReturned':'storage'}
            if action in locations:self.state.plate_location=locations[action]
            self.refresh()
            self.event('success',action);announce(self.state,f'DONE    {action}')
            return Status.SUCCESS
        except (RuntimeError,ValueError,KeyError) as exc:
            self.task.close();self.task=None;self.active_name=None;self.state.active_action=''
            self.last_failed_action=action;self.state.last_error=str(exc)
            self.motion.stop()
            if not isinstance(exc,ActionFailure):
                self.state.emergency_latched=True;self.state.emergency_reason=f'{action}: {exc}'
            self.event('failure',action,error=str(exc))
            return Status.FAILURE

    def cancel_action(self,action):
        if self.task is None or action!=self.active_name:return
        self.task.close();self.task=None;self.active_name=None;self.state.active_action=''
        self.motion.stop()
        self.expected_carrying=self.motion.holding(self.motion.carried)
        self.event('cancel',action);announce(self.state,f'CANCEL  {action}')

    def emergency_stop(self):
        if self.active_name is not None:self.cancel_action(self.active_name)
        self.motion.stop()
        self.state.stop_sent=True

    def reader_aligned(self):
        m=self.motion;context=m.place_context['reader']
        return m.holding(m.carried) and np.linalg.norm(m.data.site_xpos[m.tcp]-context['approach'])<.01

    def start_measurement(self):
        if not self.motion.verify_placement('reader'):raise ActionFailure('Reader is not loaded')
        body=self.motion.carried
        if body not in self.measurements:
            self.measurements[body]=float(self.motion.data.time)
            self.measurement_starts[body]=1
        self.state.reader_complete=False

    def wait_measurement(self):
        start=self.measurements[self.motion.carried]
        while self.motion.data.time-start<2.:yield None
        self.state.reader_complete=True

    def stabilise(self):
        if not self.motion.holding(self.motion.carried):raise ActionFailure('Cannot stabilise a lost plate')
        self.motion.stop()
        self.motion.controller.target[:]=[0.,0.]
        self.motion.data.ctrl[self.motion.controller.act]=self.motion.controller.target
        samples=[]
        for _ in range(75):
            samples.append(self.relative_plate())
            yield None
        if not self.motion.holding(self.motion.carried) or np.max(np.ptp(np.asarray(samples[-25:]),axis=0))>.001:
            raise ActionFailure('Grasp did not stabilise')
        self.grasp_reference=self.relative_plate();self.refresh()

    def recovery_station(self):
        action=self.last_failed_action or ''
        return 'source' if 'Source' in action else 'aluminium' if 'Aluminium' in action else 'reader'

    def recovery_cache_keys(self,kind):
        if kind=='grasp':
            prefix={'source':'Source','aluminium':'Aluminium','reader':'Reader'}[self.recovery_station()]
            return {prefix+'Approach',prefix+'Descend',prefix+'Close',prefix+'Lift','Verify'+prefix+'Grasp',
                    prefix+'ApproachForPick',prefix+'DescendForPick'}
        if kind=='alignment':
            if 'Storage' in (self.last_failed_action or ''):
                return {'ReturnToStorageSlot','AlignWithStorageSlot','VerifyStorageAligned','StorageDescend','PlacePlateInSlot','StorageRetract','VerifyReturned'}
            return {'AlignWithReader','VerifyReaderAligned','ReaderDescendForPlace','LoadPlateIntoReader','VerifyReaderLoaded'}
        return set()

    def recover_grasp(self):
        m=self.motion;station=self.recovery_station()
        if m.holding(m.carried):raise ActionFailure('Recovery requires inspection of a partially held plate')
        context=m.prepare_pick(station)
        if abs(m.data.xpos[m.carried,2]-context['z0'])>.015:raise ActionFailure('Plate has moved off its support')
        yield from m.move(station+': recovery open',m.runtime.arm_target.copy(),[-.04,.04],seconds=2.)
        context=m.prepare_pick(station,refresh=True)
        yield from m.move(station+': recovery retract',m.solve(context['approach'],context['yaw']),seconds=6.)

    def recover_alignment(self):
        m=self.motion
        if not m.holding(m.carried):raise ActionFailure('No verified grasp for alignment recovery')
        if 'Storage' in (self.last_failed_action or ''):
            current=m.runtime.arm_target[0]-m.runtime.arm_target[4]
            offset=m.data.xpos[m.carried]-m.data.site_xpos[m.tcp]
            yield from m.move('storage: recovery hover',m.solve(m.storage_hover_point-offset,current),seconds=6.,held=True)
            return
        context=m.place_context.get('reader')
        if context is None:raise ActionFailure('No reader approach to recover')
        yield from m.move('reader: recovery retract',m.solve(context['approach']),seconds=6.,held=True)

    def unsupported_placement_recovery(self):
        raise ActionFailure('Placement recovery needs a validated regrasp; stopping for review')
        yield None

    def finalize(self,finished):
        m=self.motion
        status=dict(backend='mujoco',plates_finished=finished,requested_plates=self.settings.plates,
                    halted=self.state.emergency_latched,last_error=self.state.last_error,
                    current_stage=self.state.current_stage,reader_mode=self.reader_mode)
        (ROOT/'outputs'/'bt_workflow_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        if finished!=self.settings.plates or self.state.emergency_latched:return
        if len(m.processed)!=finished:raise RuntimeError('BT completion and physical plate count differ')
        if any(self.measurement_starts.get(b)!=1 for b in m.processed):
            raise RuntimeError('Reader start count does not match completed plates')
        target=m.data.site_xpos[kin_site(m,'ot2_stack_top')].copy();target[1]-=.014
        thickness=2*m.model.geom_size[m.model.geom('reaction_plate_box').id,2]
        for index,body in enumerate(m.processed):
            expected=target+[0,0,index*thickness]
            error=m.data.xpos[body]-expected
            if np.linalg.norm(error[:2])>.025 or abs(error[2])>.01:raise RuntimeError('Final stack verification failed')
        payload=dict(status,completed_plates=finished,coordinate_frame='world (robot base)',distance_unit='m',
                     final_plates={m.model.body(b).name:m.data.xpos[b].tolist() for b in m.processed},
                     reader_start_counts={m.model.body(b).name:n for b,n in self.measurement_starts.items()},
                     stages=m.records,action_events=self.events)
        (ROOT/'outputs'/'bt_workflow_verified.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        export_notebook(payload,ROOT/'trajectory_notes'/'04_behavior_tree_workflow.ipynb')
        print(f'BT PHYSICS VERIFIED: {finished} plates; {len(m.records)} waypoints.',flush=True)

    def keep_open(self):
        if self.viewer is not None:
            print('BT completed; close the simulation window to exit.',flush=True)
            while self.viewer.is_running():
                self.motion.advance(DT);self.viewer.sync();time.sleep(DT/self.speed)

    def close(self):
        if self.active_name is not None:self.cancel_action(self.active_name)
        if self.viewer is not None:self.viewer.close()

def kin_site(motion,name):
    return motion.model.site(name).id
