# Four-plate lab workflow

Run from robot_sim:

```powershell
python full_lab_workflow.py
python full_lab_workflow.py --headless
python full_lab_workflow.py --speed 8
```

The default run processes four plates once and stops at HOME when the source is empty.
`--plates 1` is a diagnostic single-cycle run with the other source plates left in place.

1. Four separate physical plates start at world XY (0.20, 0.20) on the table. Slot2 starts empty.
2. Pick the top source plate, lift, transfer to the temperature aluminium block, release, and return to HOME.
3. Pick the same plate from aluminium, return through the exact fixed HOME joint pose, place it on the reader tray, and return to HOME.
4. Pick the same plate from the reader and return through HOME. Keep the reader/HOME orientation while transferring directly above slot2. At a plate-centre height of at least 0.18 m, rotate to align with the slot, compensating the TCP offset to keep the plate over the slot. Descend, release, retract vertically, and return to HOME.
5. Repeat for the remaining source plates. Storage target height increases by one plate thickness each cycle.

The scene configuration is lab_layout.json. The fixed HOME pose is ik_records/initial_pose.json.
The aluminium target is 6 mm toward the robot from its centre; the slot2 target is 14 mm toward the robot from the slot centre. Both offsets leave the plate centre over the actual support and avoid workspace extension limits.

Motion uses joint-space collision checking and smooth segments. The held plate orientation is preserved while routing, with wrist alignment performed at high clearance. Source pickup height is adjusted to keep the fingers above the table, including the bottom plate.
There are no weld attachments or plate pose writes after initialization. Grip success is checked for the selected plate, not any plate. Sustained contact loss, collision blocking, joint tracking error, incorrect placement, disturbed source plates, unstable storage, or physics warnings stop the task.
Reader placement is simulated handling only; this script does not perform a real instrument measurement.

Successful BT runs write:

- outputs/bt_workflow_verified.json: stage records, action events and final plate positions.
- outputs/bt_action_events.json: per-plate action starts, successes, cancellations and failures.
- outputs/bt_tree.txt: expanded behavior-tree structure.
- outputs/bt_workflow_status.json: latest completion or halt status.
- trajectory_notes/04_behavior_tree_workflow.ipynb: verified positions, joint angles and plotting cells.

outputs/bt_workflow_progress.json records completed waypoints during an unfinished run; it is not a success record. Earlier standalone demonstrations and their verified notes (01 through 03) remain available. Runtime outputs and local archives under outputs are ignored by Git.

## Behavior-tree integration

The same full_lab_workflow.py command now runs the existing behavior-tree mainloop with MuJoCoBackend. Task order is in chem_lab_bt/workflow_nodes.py; atomic motion is in lab_motion.py. See chem_lab_bt/README.md. New verified BT notes are saved as trajectory_notes/04_behavior_tree_workflow.ipynb; previous notes are preserved.
