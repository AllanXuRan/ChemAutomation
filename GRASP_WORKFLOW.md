# Interactive IK and physical grasping

Run `python ik_control_panel.py` from the repository root. The scene uses dynamic plates, gravity, friction and force-limited arm/gripper actuators. Grasping relies on bilateral finger contact; no weld attachment is used.

## Controls

Focus the keyboard movement panel before holding keys:

- A / D: world -X / +X.
- W / S: world +Y / -Y.
- Ctrl / Space: down / up.
- Q / E: flange rotation (10 degrees per second).
- 1 / 3: gradually close / open the gripper.
- Escape or loss of focus stops keyboard jogging.

Approach above the selected plate, align the fingers with its long sides, descend slowly, close with 1, and check bilateral contact before lifting. Lift a small distance first to confirm the grasp. Descend to a support before releasing with 3. A contact indicator alone does not establish that a plate can be transported stably.

## Saved poses and notes

The initial pose comes from `ik_records/initial_pose.json` and remains separate from newly collected poses. Every panel launch creates a unique `ik_records/ik_session_*.ipynb`; pressing the pose-save button appends to that notebook and updates `last_ik_pose.json`. Restarting creates a new session notebook.

State captures under `outputs/state_captures` also include all plate poses. Runtime outputs are local and ignored by Git. Saved session notebooks and verified `trajectory_notes` are versioned.

## Layout and automatic operation

Run `python layout_editor.py` to adjust layout positions, including the upper aluminium block. The current four source plates start at table XY (0.20, 0.20); slot2 is initially empty. Restart the simulation after rebuilding the scene.

For the complete automated workflow, use `python full_lab_workflow.py`. See [FULL_WORKFLOW.md](FULL_WORKFLOW.md) and [chem_lab_bt/README.md](chem_lab_bt/README.md). Do not use earlier taught trajectories without checking them against the current layout.

The contact/dynamics values are simulation settings, not hardware calibration. See `physics_runtime.py` for the active gripper forces and collision checks.
