"""Editable chemistry-lab Behavior Tree skeleton (MOCK mode only).

Robot task: formulation inspection -> grasp -> transfer -> align/measure -> storage.
Faults: bubble/tip -> quarantine (MOCK); grasp/alignment/placement -> bounded retry;
        slip risk -> preventive action during carrying; drop -> latched safety stop.
Based on the user's 2026-10-21 revised workflow. Does not move a real robot.

Install: python -m pip install 'py_trees>=2.2,<3'
Run:     python lab_behavior_tree.py --plates 3 --scenario scenarios.json --compact
         python lab_behavior_tree.py --inject align --trace --tick-seconds 0

This program does not connect to MuJoCo, xArm5, G2 or lab instruments yet.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import py_trees

Status = py_trees.common.Status
BASE = Path(__file__).resolve().parent


@dataclass
class Settings:
    """Editable defaults. bt_config.json overrides these at startup."""
    plates: int = 2
    mock_action_ticks: int = 2
    tick_seconds: float = 0.1
    max_ticks: int = 500
    slip_threshold: float = 0.7
    max_grasp_retries: int = 2
    max_alignment_retries: int = 2
    max_placement_retries: int = 2
    max_preventive_interventions_per_plate: int = 3


@dataclass
class Blackboard:
    """Single shared mission state; can later be replaced by py_trees.blackboard."""
    plate_id: int = 1
    tick_number: int = 0
    completed: set[str] = field(default_factory=set)
    current_stage: str = "idle"
    plate_finished: bool = False
    plate_outcome: str = ""             # measured / rejected
    carrying_plate: bool = False
    fault: Optional[str] = None         # bubble / tip / grasp / alignment
    slip_risk: float = 0.05
    emergency_latched: bool = False
    emergency_reason: str = ""
    stop_sent: bool = False
    retries: dict[str, int] = field(default_factory=lambda: {"grasp": 0, "alignment": 0, "placement": 0})
    recovery_actions_completed: dict[str, int] = field(default_factory=lambda: {"grasp": 0, "alignment": 0, "placement": 0})
    preventive_interventions: int = 0
    preventive_this_plate: int = 0
    rejected_plates: int = 0
    measured_plates: int = 0
    human_interventions_required: int = 0
    compact_log: bool = False

    def next_plate(self) -> None:
        if self.emergency_latched:
            raise RuntimeError("Cannot advance while an emergency is latched")
        self.plate_id += 1
        self.completed.clear()
        self.current_stage = "idle"
        self.plate_finished = False
        self.plate_outcome = ""
        self.carrying_plate = False
        self.fault = None
        self.slip_risk = 0.05
        self.preventive_this_plate = 0
        self.retries = {"grasp": 0, "alignment": 0, "placement": 0}


def announce(state: Blackboard, message: str) -> None:
    """--compact suppresses low-level command spam while keeping meaningful events."""
    if state.compact_log:
        if message.startswith(('START ', 'CANCEL ')):
            return
        if message.startswith('DONE '):
            name = message.split()[-1]
            milestones = {
                'InspectFormulation', 'VerifyGrasp', 'TransferToReader',
                'AlignWithReader', 'VerifyReaderAligned', 'WaitForReadingComplete',
                'VerifyReturned', 'MoveRejectedPlateToBin', 'StabiliseGrasp',
                'RetractAndRealign', 'RepositionForPlacement', 'PlacePlateInSlot',
            }
            if name not in milestones:
                return
    print(f"[Plate {state.plate_id} | tick {state.tick_number:03}] {message}")


# EDIT HERE if you add a new fault or a new stage to the tree.
# The default action is the step at which the mock sensor/error is injected.
EVENT_ACTIONS = {
    'bubble': 'InspectFormulation',
    'tip': 'InspectFormulation',
    'grasp': 'GraspPlate',
    'align': 'AlignWithReader',
    'placement': 'PlacePlateInSlot',
    'slip': 'TransferToReader',
    'drop': 'TransferToReader',
}
# On these stages the sample tree's carrying_plate flag is True.
CARRYING_ACTIONS = {'TransferToReader', 'ReturnToStorageSlot'}
ALIGNMENT_ACTIONS = {'AlignWithReader', 'VerifyReaderAligned', 'LoadPlateIntoReader'}
GRASP_ACTIONS = {'GraspPlate', 'VerifyGrasp'}
PLACEMENT_ACTIONS = {'PlacePlateInSlot', 'VerifyReturned'}


@dataclass
class ScenarioEvent:
    kind: str
    action: str
    risk: float = 0.93
    remaining: int = 1
    triggered: int = 0


def load_scenarios(path: Path) -> dict[int, list[ScenarioEvent]]:
    """Parse human-editable per-plate fault injections; validate early."""
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find scenario JSON: {path}')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) != {'plates'} or not isinstance(raw['plates'], dict):
        raise ValueError('Scenario JSON must contain one object: {"plates": {"1": [...], ...}}')
    result = {}
    for key, entries in raw['plates'].items():
        try:
            plate_id = int(key)
        except (ValueError, TypeError):
            raise ValueError(f'Invalid plate ID: {key!r}') from None
        if plate_id < 1 or str(plate_id) != str(key):
            raise ValueError(f'Plate key must be a positive integer string: {key!r}')
        if not isinstance(entries, list):
            raise ValueError(f'Plate {plate_id}: expected a list of events')
        parsed = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or 'type' not in entry:
                raise ValueError(f'Plate {plate_id} event {i+1}: expected object with type')
            unknown = set(entry) - {'type', 'at', 'risk', 'times'}
            if unknown:
                raise ValueError(f'Plate {plate_id} event {i+1}: unsupported keys {sorted(unknown)}')
            kind = entry['type']
            if kind not in EVENT_ACTIONS:
                raise ValueError(f'Plate {plate_id}: unknown type {kind!r}; choose {list(EVENT_ACTIONS)}')
            action = entry.get('at', EVENT_ACTIONS[kind])
            if not isinstance(action, str) or not action:
                raise ValueError(f'Plate {plate_id}: at must be an action name')
            if kind in {'bubble', 'tip'} and action != EVENT_ACTIONS[kind]:
                raise ValueError(f'Plate {plate_id}: {kind!r} must occur at {EVENT_ACTIONS[kind]}')
            if kind == 'align' and action not in ALIGNMENT_ACTIONS:
                raise ValueError(f'Plate {plate_id}: align at must be one of {sorted(ALIGNMENT_ACTIONS)}')
            if kind == 'grasp' and action not in GRASP_ACTIONS:
                raise ValueError(f'Plate {plate_id}: grasp at must be one of {sorted(GRASP_ACTIONS)}')
            if kind == 'placement' and action not in PLACEMENT_ACTIONS:
                raise ValueError(f'Plate {plate_id}: placement at must be one of {sorted(PLACEMENT_ACTIONS)}')
            if kind in {'slip', 'drop'} and action not in CARRYING_ACTIONS:
                raise ValueError(f'Plate {plate_id}: {kind!r} at must be one of {sorted(CARRYING_ACTIONS)}')
            risk = entry.get('risk', 0.93)
            if type(risk) not in (int, float) or not 0 <= risk <= 1:
                raise ValueError(f'Plate {plate_id}: risk must be between 0 and 1')
            times = entry.get('times', 1)
            if type(times) is not int or times < 1:
                raise ValueError(f'Plate {plate_id}: times must be a positive integer')
            if kind in {'bubble', 'tip', 'drop'} and times != 1:
                raise ValueError(f'Plate {plate_id}: {kind!r} can only occur once per plate')
            parsed.append(ScenarioEvent(kind=kind, action=action, risk=float(risk), remaining=times))
        # Once bubble/tip rejects a plate it never reaches the later stages.
        if sum(ev.kind in {'bubble', 'tip'} for ev in parsed) > 1:
            raise ValueError(f'Plate {plate_id}: bubble and tip cannot both be simulated on one rejected plate')
        result[plate_id] = parsed
    return result


class RobotBackend:
    """THE future integration boundary for MuJoCo / real xArm.

    start_action(): issue a command exactly once (non-blocking).
    poll_action(): return RUNNING/SUCCESS/FAILURE, without restarting it.
    cancel_action(): stop an interrupted command safely.
    emergency_stop(): request stop; true hardware safety must be independent.
    """

    def start_action(self, action: str, plate_id: int) -> None:
        raise NotImplementedError

    def poll_action(self, action: str) -> Status:
        raise NotImplementedError

    def cancel_action(self, action: str) -> None:
        raise NotImplementedError

    def emergency_stop(self) -> None:
        raise NotImplementedError


class MockBackend(RobotBackend):
    """Deterministic robot with independently scheduled events for each plate.

    Each event is injected on the first poll of its chosen action. The 'times'
    setting can inject the same fault on successive attempts at that stage.
    """

    def __init__(self, state: Blackboard, settings: Settings,
                 scenarios: Optional[dict[int, list[ScenarioEvent]]] = None):
        self.state, self.settings = state, settings
        self.scenarios = scenarios if scenarios is not None else {}
        self.active: dict[str, int] = {}

    def start_action(self, action: str, plate_id: int) -> None:
        if action in self.active:
            raise RuntimeError(f"Duplicate start for {action}")
        self.active[action] = max(1, self.settings.mock_action_ticks)
        announce(self.state, f"START   {action}")

    def poll_action(self, action: str) -> Status:
        if action not in self.active:
            raise RuntimeError(f"Poll called without start: {action}")
        s = self.state
        # Each plate has its own event queue. This prevents faults intended for
        # plate 3 from affecting plate 1, even if the action names are identical.
        for event in self.scenarios.get(s.plate_id, []):
            if event.action != action or event.remaining < 1:
                continue
            event.remaining -= 1
            event.triggered += 1
            if event.kind in {'bubble', 'tip', 'grasp', 'align', 'placement'}:
                s.fault = 'alignment' if event.kind == 'align' else event.kind
                self.active.pop(action)
                announce(s, f"FAULT   {s.fault} at {action}")
                return Status.FAILURE
            if event.kind == 'slip':
                s.slip_risk = event.risk
                announce(s, f"RISK    predicted slip={s.slip_risk:.2f} (not dropped)")
                # Risk is read by the high-priority mitigation branch next tick.
                return Status.RUNNING
            if event.kind == 'drop':
                s.emergency_latched = True
                s.emergency_reason = f"plate dropped during {action}"
                self.active.pop(action)
                announce(s, "CRITICAL plate dropped: protective stop requested")
                self.emergency_stop()
                return Status.FAILURE

        self.active[action] -= 1
        if self.active[action] > 0:
            return Status.RUNNING
        self.active.pop(action)
        announce(s, f"DONE    {action}")
        return Status.SUCCESS

    def cancel_action(self, action: str) -> None:
        if action in self.active:
            self.active.pop(action)
            announce(self.state, f"CANCEL  {action}")

    def emergency_stop(self) -> None:
        for name in tuple(self.active):
            self.cancel_action(name)
        if not self.state.stop_sent:
            self.state.stop_sent = True
            announce(self.state, "STOP    emergency latched; human authorisation required")


class Condition(py_trees.behaviour.Behaviour):
    """Check a predicate. True -> SUCCESS; False -> FAILURE."""

    def __init__(self, name: str, predicate: Callable[[], bool]):
        super().__init__(name)
        self.predicate = predicate

    def update(self) -> Status:
        return Status.SUCCESS if self.predicate() else Status.FAILURE


class AsyncAction(py_trees.behaviour.Behaviour):
    """Tick-friendly action: start only once; poll until complete.

    cache=True: completed tasks are not executed twice after BT preemption.
    cache=False: mitigation / recovery actions can be repeated on a later fault.
    """

    def __init__(self, name: str, state: Blackboard, backend: RobotBackend,
                 *, cache: bool = True, on_success: Optional[Callable[[], None]] = None,
                 fault_on_failure: Optional[str] = None):
        super().__init__(name)
        self.state, self.backend = state, backend
        self.cache, self.on_success = cache, on_success
        self.fault_on_failure = fault_on_failure
        self.started = False

    def update(self) -> Status:
        if self.cache and self.name in self.state.completed:
            return Status.SUCCESS
        if not self.started:
            self.state.current_stage = self.name
            self.backend.start_action(self.name, self.state.plate_id)
            self.started = True
        result = self.backend.poll_action(self.name)
        if result == Status.SUCCESS:
            if self.on_success is not None:
                self.on_success()
            if self.cache:
                self.state.completed.add(self.name)
        elif result == Status.FAILURE and not self.state.emergency_latched:
            if self.state.fault is None and self.fault_on_failure:
                self.state.fault = self.fault_on_failure
            elif self.state.fault is None:
                self.state.emergency_latched = True
                self.state.emergency_reason = f"unhandled failure of {self.name}"
        return result

    def terminate(self, new_status: Status) -> None:
        if new_status == Status.INVALID and self.started:
            self.backend.cancel_action(self.name)
        self.started = False


class PreventSlip(AsyncAction):
    def __init__(self, state: Blackboard, backend: RobotBackend, settings: Settings):
        self.settings = settings
        def on_success():
            # MOCK ONLY: future real backend must confirm reduced slip risk.
            state.slip_risk = 0.05
            state.preventive_interventions += 1
            state.preventive_this_plate += 1
            announce(state, "PREVENT slip risk reduced; resume interrupted task")
        super().__init__("StabiliseGrasp", state, backend, cache=False, on_success=on_success)

    def update(self) -> Status:
        if self.state.preventive_this_plate >= self.settings.max_preventive_interventions_per_plate:
            self.state.emergency_latched = True
            self.state.emergency_reason = "slip risk persists after preventive interventions"
            announce(self.state, "CRITICAL preventive intervention budget exhausted")
            return Status.RUNNING  # Hold priority; next tick latches emergency branch.
        return super().update()


class EmergencyStop(py_trees.behaviour.Behaviour):
    def __init__(self, state: Blackboard, backend: RobotBackend):
        super().__init__("LatchedEmergencyStop")
        self.state, self.backend = state, backend

    def update(self) -> Status:
        self.backend.emergency_stop()
        # RUNNING holds the priority branch; automatic restart is forbidden.
        return Status.RUNNING


class RetryRecovery(py_trees.behaviour.Behaviour):
    """Recover once; clear fault so the failed main action can be re-run."""

    def __init__(self, name: str, state: Blackboard, backend: RobotBackend,
                 *, kind: str, max_retries: int):
        super().__init__(name)
        self.state, self.backend = state, backend
        self.kind, self.max_retries = kind, max_retries
        self.started = False

    def update(self) -> Status:
        if not self.started:
            if self.state.retries[self.kind] >= self.max_retries:
                self.state.emergency_latched = True
                self.state.emergency_reason = f"{self.kind} recovery retries exhausted"
                return Status.RUNNING  # Do not fall through to normal BT on this tick.
            self.state.retries[self.kind] += 1
            announce(self.state, f"RECOVER {self.kind}, attempt {self.state.retries[self.kind]}/{self.max_retries}")
            self.backend.start_action(self.name, self.state.plate_id)
            self.started = True
        result = self.backend.poll_action(self.name)
        if result == Status.SUCCESS:
            self.state.fault = None
            self.state.recovery_actions_completed[self.kind] += 1
            # A failed insertion may have already passed the alignment check.
            # Recovery must re-verify alignment rather than skipping it via cache.
            if self.kind == 'alignment':
                self.state.completed.difference_update({
                    'AlignWithReader', 'VerifyReaderAligned',
                    'LoadPlateIntoReader', 'VerifyReaderLoaded',
                })
            elif self.kind == 'grasp':
                self.state.completed.difference_update({'MoveToOT2Pickup', 'GraspPlate', 'VerifyGrasp'})
            elif self.kind == 'placement':
                # RepositionForPlacement is a high-level skill: if the plate was
                # already released, the future real adapter MUST safely regrasp.
                self.state.carrying_plate = True
                self.state.completed.difference_update({'PlacePlateInSlot', 'VerifyReturned'})
            announce(self.state, f"RESUME  {self.kind} recovery completed")
        elif result == Status.FAILURE:
            self.state.emergency_latched = True
            self.state.emergency_reason = f"{self.kind} recovery action failed"
            return Status.RUNNING  # Hold recovery priority until emergency takes over.
        return result

    def terminate(self, new_status: Status) -> None:
        if new_status == Status.INVALID and self.started:
            self.backend.cancel_action(self.name)
        self.started = False


class FinishPlate(py_trees.behaviour.Behaviour):
    def __init__(self, state: Blackboard, *, outcome: str):
        super().__init__(f"FinishPlate_{outcome}")
        self.state, self.outcome = state, outcome

    def update(self) -> Status:
        if not self.state.plate_finished:
            self.state.plate_finished = True
            self.state.plate_outcome = self.outcome
            if self.outcome == "measured":
                self.state.measured_plates += 1
            else:
                self.state.rejected_plates += 1
            announce(self.state, f"PLATE   {self.outcome.upper()} — next plate may start")
        return Status.SUCCESS


def seq(name: str, *nodes: py_trees.behaviour.Behaviour) -> py_trees.composites.Sequence:
    return py_trees.composites.Sequence(name=name, memory=True, children=list(nodes))


def build_tree(state: Blackboard, backend: RobotBackend, settings: Settings):
    """EDIT HERE: the tree follows the updated workflow diagram.

    Root is a reactive (memory=False) priority selector:
       1. Emergency / dropped plate
       2. Failure-specific reactive recovery
       3. Preventive mitigation for predicted slip while carrying
       4. Normal plate execution

    Normal workflow is cached by stage (completed set), so preemption does not
    repeat irreversible formulation or measurement steps.
    """
    def a(name: str, **kw):
        return AsyncAction(name, state, backend, **kw)

    def c(name: str, predicate: Callable[[], bool]):
        return Condition(name, predicate)

    # ===== 1. Safety, highest priority: no auto-resume after actual drop =====
    safety = seq('01_CRITICAL_SAFETY',
                 c('CriticalFault?', lambda: state.emergency_latched),
                 EmergencyStop(state, backend))

    # ===== 2. Failures that already happened (reactive recovery) =====
    # Do not turn chemical defect detection into automatic chemical repair.
    # The bin action is a MOCK placeholder for a validated quarantine procedure.
    bubble_recovery = seq('BubbleRecovery',
        c('Bubbles?', lambda: state.fault == 'bubble'),
        a('CheckQuarantineSafe', cache=False),
        a('MoveRejectedPlateToBin', cache=False),
        FinishPlate(state, outcome='rejected'))

    tip_recovery = seq('TipFailureRecovery',
        c('TipFailure?', lambda: state.fault == 'tip'),
        a('CheckQuarantineSafe', cache=False),
        a('MoveRejectedPlateToBin', cache=False),
        FinishPlate(state, outcome='rejected'))

    grasp_recovery = seq('GraspFailureRecovery',
        c('GraspFailure?', lambda: state.fault == 'grasp'),
        RetryRecovery('RepositionForGrasp', state, backend,
                      kind='grasp', max_retries=settings.max_grasp_retries))

    alignment_recovery = seq('AlignmentFailureRecovery',
        c('AlignmentFailure?', lambda: state.fault == 'alignment'),
        RetryRecovery('RetractAndRealign', state, backend,
                      kind='alignment', max_retries=settings.max_alignment_retries))

    placement_recovery = seq('PlacementFailureRecovery',
        c('PlacementFailure?', lambda: state.fault == 'placement'),
        RetryRecovery('RepositionForPlacement', state, backend,
                      kind='placement', max_retries=settings.max_placement_retries))

    recovery_by_type = py_trees.composites.Selector(
        name='RecoveryByFailureType', memory=False,
        children=[bubble_recovery, tip_recovery, grasp_recovery,
                  alignment_recovery, placement_recovery])
    recovery = seq('02_REACTIVE_RECOVERY',
        c('RecoverableFault?', lambda: state.fault is not None),
        recovery_by_type)

    # ===== 3. Predicted risk, not yet a failure =====
    # The monitoring/vision/touch module will update slip_risk asynchronously.
    # The BT controls intervention selection, NOT camera frame acquisition rate.
    preventive = seq('03_PREDICTIVE_MITIGATION',
        c('SlipRiskHighWhileCarrying?', lambda:
          state.carrying_plate and state.fault is None and
          state.slip_risk >= settings.slip_threshold),
        PreventSlip(state, backend, settings))

    # ===== 4. NORMAL FLOW (your updated figure, expanded at reader) =====
    # The diagram says Alignment -> Return to Slot; we explicitly keep the
    # necessary reader handoff/measurement/retrieval steps between them.
    main = seq('04_NORMAL_ONE_PLATE',
        # Plate prepared by external formulation station; inspect outcome.
        a('WaitForFormulationComplete'),
        a('InspectFormulation'),            # inject bubble or tip here

        # Grasp with the xArm5/G2. On grasp FAIL, root handles reposition.
        a('MoveToOT2Pickup'),
        a('GraspPlate', fault_on_failure='grasp'),
        a('VerifyGrasp', fault_on_failure='grasp',
          on_success=lambda: setattr(state, 'carrying_plate', True)),

        # Transfer with continuous risk updates; BT can interrupt it.
        a('TransferToReader'),

        # Alignment/retry, then read the plate (implicit in flowchart).
        a('AlignWithReader', fault_on_failure='alignment'),
        a('VerifyReaderAligned', fault_on_failure='alignment'),
        a('LoadPlateIntoReader', fault_on_failure='alignment',
          on_success=lambda: setattr(state, 'carrying_plate', False)),
        a('VerifyReaderLoaded'),
        a('StartReaderMeasurement'),
        a('WaitForReadingComplete'),
        a('RetrievePlate', on_success=lambda: setattr(state, 'carrying_plate', True)),

        # Final return-to-slot stage, with its own placement-retry branch.
        a('ReturnToStorageSlot'),
        a('PlacePlateInSlot', fault_on_failure='placement',
          on_success=lambda: setattr(state, 'carrying_plate', False)),
        a('VerifyReturned', fault_on_failure='placement'),
        FinishPlate(state, outcome='measured'),
    )

    root = py_trees.composites.Selector(
        name='ROOT_PRIORITY_SELECTOR', memory=False,
        children=[safety, recovery, preventive, main])
    return py_trees.trees.BehaviourTree(root)


def load_settings(path: Path) -> Settings:
    values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown bt_config.json settings: {', '.join(sorted(unknown))}")
    obj = Settings(**values)
    if obj.plates < 1 or obj.mock_action_ticks < 1 or obj.max_ticks < 1:
        raise ValueError("plates, mock_action_ticks, max_ticks must be positive")
    if obj.tick_seconds < 0 or not 0 <= obj.slip_threshold <= 1:
        raise ValueError("tick_seconds must be >=0; slip_threshold in [0,1]")
    if min(obj.max_grasp_retries, obj.max_alignment_retries, obj.max_placement_retries) < 0:
        raise ValueError("max retries must be >=0")
    if obj.max_preventive_interventions_per_plate < 0:
        raise ValueError("max_preventive_interventions_per_plate must be >=0")
    return obj


def main():
    parser = argparse.ArgumentParser(description="Chemical lab BT skeleton — mock only")
    parser.add_argument("--config", type=Path, default=BASE / "bt_config.json")
    parser.add_argument("--plates", type=int, help="Override number of plates")
    parser.add_argument("--tick-seconds", type=float, help="Override mock tick duration")
    parser.add_argument("--inject", choices=("none", "bubble", "tip", "grasp", "align", "placement", "slip", "drop"), default="none",
                        help="Legacy mode: inject one fault into plate 1 only")
    parser.add_argument("--scenario", type=Path, metavar="FILE",
                        help="Per-plate scenario JSON; allows different faults for each plate")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--compact", action="store_true", help="Show key milestones and fault/recovery events only")
    output_group.add_argument("--trace", action="store_true", help="Print root/stage after every tick")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.plates is not None:
        settings.plates = args.plates
    if args.tick_seconds is not None:
        settings.tick_seconds = args.tick_seconds
    if settings.plates < 1 or settings.tick_seconds < 0:
        parser.error("--plates must be >=1 and --tick-seconds >=0")

    if args.scenario is not None and args.inject != 'none':
        parser.error('Choose either --scenario or --inject; do not use both')
    if args.scenario is not None:
        scenarios = load_scenarios(args.scenario)
    elif args.inject != 'none':
        kind = args.inject
        scenarios = {1: [ScenarioEvent(kind=kind, action=EVENT_ACTIONS[kind])]}
    else:
        scenarios = {}
    if any(plate_id > settings.plates for plate_id in scenarios):
        print('NOTE: events configured for plates beyond the --plates limit will not run.')

    state = Blackboard(compact_log=args.compact)
    backend = MockBackend(state, settings, scenarios=scenarios)
    tree = build_tree(state, backend, settings)
    print("=== xArm5 chemical laboratory BT (MOCK ONLY; NO ROBOT COMMANDS) ===")
    print(f"Plan: {settings.plates} plates; scenario=" +
          (str(args.scenario) if args.scenario else f"legacy --inject {args.inject}"))
    for plate_id in sorted(scenarios):
        desc = ', '.join(f'{ev.kind}@{ev.action}' + (f'({ev.risk:.2f})' if ev.kind == 'slip' else '') +
                         (f'x{ev.remaining}' if ev.remaining > 1 else '')
                         for ev in scenarios[plate_id]) or 'none'
        print(f'  Plate {plate_id}: {desc}')

    finished = 0
    for tick_no in range(1, settings.max_ticks + 1):
        state.tick_number = tick_no
        # For real hardware, refresh sensor state outside of tree.tick() here,
        # or have a separate thread updating a thread-safe shared state.
        tree.tick()
        if args.trace:
            print(f"  TICK {tick_no:03d}: root={tree.root.status.name}; "
                  f"stage={state.current_stage}; completed={len(state.completed)}; "
                  f"fault={state.fault}; risk={state.slip_risk:.2f}")

        if state.emergency_latched:
            if not state.stop_sent:
                # Ensures latched-stop handler ticks before leaving the runner.
                tree.tick()
            state.human_interventions_required += 1
            print(f"\nHALTED: {state.emergency_reason}; manual review required (MOCK)")
            break

        if state.plate_finished:
            print(f'  => Plate {state.plate_id}: {state.plate_outcome.upper()}')
            pending = [(ev.kind, ev.remaining) for ev in scenarios.get(state.plate_id, []) if ev.remaining]
            if pending:
                print(f'  NOTE: untriggered scenario events on Plate {state.plate_id}: {pending}')
            finished += 1
            if finished >= settings.plates:
                print("\nBATCH FINISHED (MOCK)")
                break
            # Invalidate composite memory; cached stages reset for next plate.
            tree.root.stop(Status.INVALID)
            state.next_plate()
            announce(state, "NEXT PLATE")

        time.sleep(settings.tick_seconds)
    else:
        print("\nTIMEOUT: max_ticks exceeded; inspect RUNNING stage / faults")

    print("SUMMARY:", {
        "plates_finished": finished,
        "measured": state.measured_plates,
        "rejected": state.rejected_plates,
        "preventive_interventions": state.preventive_interventions,
        "recoveries_completed": state.recovery_actions_completed,
        "human_interventions_required": state.human_interventions_required,
    })


if __name__ == "__main__":
    main()
