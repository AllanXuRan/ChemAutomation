# Behavior-tree driven MuJoCo workflow

From robot_sim:

```powershell
python full_lab_workflow.py
python full_lab_workflow.py --headless --compact
python full_lab_workflow.py --plates 1 --headless --compact
python chem_lab_bt/lab_behavior_tree.py --backend mujoco --speed 8
python chem_lab_bt/lab_behavior_tree.py --backend mock --scenario chem_lab_bt/scenarios.json --compact
```

The original mainloop in lab_behavior_tree.py still ticks the tree, handles emergency/timeout, invalidates composite memory, and advances plate IDs. In MuJoCo mode it calls backend.advance() once per tick before tree.tick(). There is only one model/data/runtime for the whole batch.

- workflow_nodes.py: the task order, divided into SourceToAluminium, AluminiumToReader and ReaderToSlot2 subtrees. Each approach, descent, close, lift, verification, HOME move, alignment, release and retraction is a distinct AsyncAction.
- mujoco_backend.py: implements start_action, poll_action, cancel_action and emergency_stop; owns physical sensor feedback, selected plate identity, simulated reader state and action events.
- ../lab_motion.py: reusable motion generators, IK, cooperative collision planning, smooth trajectories, contact checks, source-height compensation and increasing storage height. No mission loop and no hidden simulation loop.
- lab_behavior_tree.py: existing priority selector, normal/recovery/prevention/safety branches and mainloop. Mock behavior remains available.
- ../full_lab_workflow.py: compatibility entry point; it now runs this BT instead of a monolithic trajectory loop.

`start_action` creates one generator. `poll_action` advances it to its next yield and returns RUNNING/SUCCESS/FAILURE. The mainloop advances physics separately by 0.02 simulated seconds per tick; motion actions and path planning yield control. Success caching prevents repeated releases/reader starts after tree interruption. Each plate has unique source/aluminium/reader grasp node names.

Cancellation closes the generator and holds the current arm pose. An actual held plate retains closing force; cancellation does not reset the scene, teleport joints/plates, or open the gripper. Higher-priority stabilization can preempt an active move. Resume re-plans the interrupted atomic move from the current physical state.

The latest storage order is explicit in ReaderToSlot2:

```
ReaderClose -> ReaderLift -> VerifyReaderGrasp -> HomeBeforeStorage
-> ReturnToStorageSlot (move above slot2 while retaining HOME yaw)
-> AlignWithStorageSlot (rotate above slot2 at >=0.18 m world Z)
-> VerifyStorageAligned -> StorageDescend -> PlacePlateInSlot
-> StorageRetract -> VerifyReturned -> HomeAfterStorage -> CommitStoredPlate
```

The blackboard carries selected_body, plate_location, source_remaining, storage_count, carrying_plate, slip_risk, reader_complete, active_action and last_error. Physical grasp/placement verification drives state; it is not inferred from elapsed motion time alone.

Reader measurement is currently a clearly labelled two-second simulated timer. StartReaderMeasurement is idempotent per plate; WaitForReadingComplete resumes the existing request. Replace those two implementations when integrating an actual reader. No chemical inspection or quarantine result is fabricated in MuJoCo mode.

Grasp recovery can open and retreat for another verified attempt if the plate remains supported. Alignment recovery retracts while holding. Unsupported placement/quarantine recovery fails and latches the existing stop branch; collision, persistent contact loss, invalid physics and tracking errors also stop. Mock fault injection remains restricted to mock mode.

Successful simulation runs write:

- outputs/bt_workflow_verified.json: physical stage data, BT action events, final plate poses and one reader start per plate.
- trajectory_notes/04_behavior_tree_workflow.ipynb: readable verified positions and joint data.
- outputs/bt_tree.txt: expanded tree structure.
- outputs/bt_action_events.json: action start/success/cancel/failure events with plate ID, BT tick and simulation time.
- outputs/bt_workflow_status.json: latest completion/halt status. Partial progress is kept separately and is not a success report.

Use --trace for per-tick tree status, --max-ticks to override the bound, --no-keep-open to close the viewer on completion. The four-plate input configuration and fixed HOME remain unchanged. Previous monolithic code is archived in outputs/before_bt_integration.
