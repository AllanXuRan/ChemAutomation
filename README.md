# ChemAutomation robot simulation

MuJoCo simulation of an xArm5 with a G2 gripper, a four-plate source stack, an aluminium reaction block, a reader tray and slot2 storage. The behavior tree controls the complete plate-handling workflow.

## Setup and run

Use Python 3.11 or later with Tkinter available. Run commands from this repository root. Keep the complete `xarm_description/meshes` directory.

```powershell
python -m pip install -r requirements.txt
python full_lab_workflow.py
```

The default run handles four plates and finishes at the fixed HOME pose. For a single-plate diagnostic or a headless batch:

```powershell
python full_lab_workflow.py --plates 1 --headless --compact
python full_lab_workflow.py --headless --compact
```

Reader measurement currently uses a two-second simulated timer. It is not connected to a real instrument.

## Project map

| Location | Purpose |
|---|---|
| `full_lab_workflow.py` | Main entry point; runs the existing BT mainloop |
| `chem_lab_bt/workflow_nodes.py` | Three task subtrees and 48 atomic actions per plate |
| `chem_lab_bt/mujoco_backend.py` | Physical action execution, feedback, cancellation and stop |
| `lab_motion.py` | IK, collision planning and cooperative motion primitives |
| `ik_control_panel.py` | Interactive IK, gripper control and session recording |
| `layout_editor.py` / `lab_layout.json` | Layout editing and saved configuration |
| `ik_records/initial_pose.json` | Fixed initial joint pose |
| `ik_records/` | Saved IK sessions and taught poses |
| `trajectory_notes/` | Verified trajectory notebooks and position notes |
| `replay_trajectory.py` | Replay of collected trajectories |
| `demo_pick_place.py` / `demo_aluminium_to_reader.py` | Earlier standalone demonstrations |
| `Tactile/test_digit_mujoco.py` | Optional DIGIT camera smoke test; requires OpenCV and hardware |

Detailed instructions: [complete workflow](FULL_WORKFLOW.md), [behavior tree](chem_lab_bt/README.md), [interactive controls](GRASP_WORKFLOW.md).

## Interactive operation

```powershell
python ik_control_panel.py
python layout_editor.py
```

With the keyboard movement panel focused: A/D moves X, W/S moves Y, Ctrl/Space moves Z, Q/E rotates the flange, and 1/3 closes/opens the gripper. Each IK launch creates a new notebook in `ik_records`; the save button appends to that session. The fixed initial pose is stored separately.

## Validation and records

```powershell
python -m unittest test_bt_integration test_contact_physics test_keyboard_ee test_flange_level test_state_capture test_project_checks
```

The current snapshot passed 33 tests and a complete four-plate physical run: 192 successful atomic actions, 237 waypoints, no failed actions or manual intervention. Each plate started its simulated reader measurement once. The reader-to-storage order is HOME, above slot2, rotate to align, descend, release, retract, HOME.

The verified record is [04_behavior_tree_workflow.ipynb](trajectory_notes/04_behavior_tree_workflow.ipynb). Successful new runs update that notebook and write `outputs/bt_workflow_verified.json`, `outputs/bt_action_events.json`, `outputs/bt_tree.txt` and `outputs/bt_workflow_status.json`. Runtime outputs, caches and logs are ignored by Git; saved teaching sessions and verified notes are versioned.
