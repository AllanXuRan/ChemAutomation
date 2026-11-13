# plate_bt: Allan's behaviour tree driving his MuJoCo workcell

This package runs Xu's overall workflow (`claude-notes/2026-10-22_overall-workflow.md`) with
Allan's tree, `ChemAutomation/chem_lab_bt/lab_behavior_tree.py`, unchanged. The tree drives his
simulated xArm5 and G2 through his `PhysicsRuntime`. Nothing in `ChemAutomation/` is edited:
his tree, his loader, his IK and his runtime are imported. Results are on
`claude-notes/plate-bt.html`; the plan is `plans/plate-bt.md`.

## Running it

In the `fyp-mujoco` environment (MuJoCo 3.x, `py_trees` 2.x), on the GPU machine:

```bash
MUJOCO_GL=egl python scripts/bt/check_layout.py                                      # reach of every area, the drawer
MUJOCO_GL=egl python scripts/bt/run_plates.py --scenario configs/plate_bt_run.yaml   # writes data/plate_bt_<ts>/
MUJOCO_GL=egl python scripts/bt/render_run.py data/plate_bt_<ts>                     # clips and stills
MUJOCO_GL=egl python scripts/bt/run_plates.py --scenario configs/plate_bt_continuous.yaml
MUJOCO_GL=egl python scripts/bt/render_grid.py data/plate_bt_<ts>                    # one video, four cameras
MUJOCO_GL=egl python tests/test_plate_bt.py
```

A run writes `states.npz` (one record per 1/30 s tick: qpos, slip risk, grip force, the active
action and the fault), `trace.json` (every action, check, measurement and scenario event, and
each plate's outcome) and `run.log` (your tree's own messages).

## How it fits your tree

| Your code | What plate_bt does with it |
|---|---|
| `build_tree` | Called as it is; `tree.py` inserts five actions at the head of `04_NORMAL_ONE_PLATE`: `FetchNewPlate`, `GraspNewPlate`, `VerifyNewPlateGrasp`, `PlaceOnHeater`, `VerifyOnHeater`. They use your fault kinds (`grasp`, `placement`), so your recovery branches handle them |
| `RobotBackend` | Implemented by `backend.MujocoBackend`. `ACTIONS` at the top of `backend.py` maps each of your action names to the method that runs it |
| `Blackboard`, `Settings`, `bt_config.json` | Used as they are: retry limits, the slip threshold 0.7, the preventive budget |
| Your runner (`main()`) | `runner.run` follows it: tick, finish a plate, reset the tree, next plate, stop on a latched emergency. Before each tick, `backend.sense()` writes the slip risk and checks for a drop, where your comment leaves room for sensor updates |
| `load_physics_model`, `PhysicsRuntime`, `jac_ik`, `PhysicalGraspController` | Used for every motion, so your collision guard checks every command; `controller.plate` is switched to the plate being handled |

## Modules

| File | Role |
|---|---|
| `scene.py` | Your scene plus more plates (copies of `reaction_plate`), three table pads (new plate, storage pile, bin pile), your tray as a drawer, and a parking shelf, built in memory; your stack in slot 2 is left out |
| `world.py` | `SimWorld`: plate poses, IK with a level flange, arm and finger commands, one tick of physics. **Replace this class for the real robot** |
| `reader.py` | `SimReader`: opens and closes the reader's drawer. **Replace this class with the reader's own interface** |
| `skills.py` | Motions as generators, one tick of commands per `next()`: straight lines, joint-space moves with via points, guarded set-downs, braking |
| `checks.py` | Grasp pre-check, test lift, slip risk, alignment, placement: each returns pass or fail with its numbers |
| `perception.py` | Placeholders: poses from the simulation (plates, the tray pocket, the top of each pile), and `ScriptedDetector` for bubbles and fallen tips |
| `backend.py` | The actions, the recoveries, `sense()` |
| `events.py` | Scenarios: the physical failure each plate meets |
| `runner.py` | The tick loop and the run's records |

## Changing things

- **Areas, grasps, speeds, tolerances:** `configs/plate_bt.yaml`. Your sites (`ot2_pickup`,
  `versamax_load`) give the heater and the tray. The new-plate area, the storage pile and the bin
  pile are table positions there; after moving one, run `scripts/bt/check_layout.py`.
- **The reader's cycle:** the arm sets the plate on the open tray, goes home, checks it is outside
  `reader.keep_out`, and only then does the drawer close; after the measurement the drawer opens
  (the arm still clear) and the arm fetches the plate. The drawer's travel, speed and slot are
  under `reader:` in the config.
- **How a run starts:** a scenario's `start` puts plates on the piles before the first plate and
  can stack the new plates on the new-plate area (`configs/plate_bt_continuous.yaml`;
  `events.py` lists the keys).
- **A new failure:** add a key to `events.py`; make the physical change in `on_plate_start` or
  `fetch_new_plate` (`backend.py`); have the action that should notice it return False after a
  check from `checks.py`. Your tree then runs the recovery for that action's fault kind.
- **A new action in your tree:** add its name to `ACTIONS` with a method that returns a
  generator (`yield` once per tick; `return False` to fail).
- **Real robot:** implement `SimWorld`'s methods over the xArm SDK and the G2 (poses, `ik`,
  `set_arm_target`, `set_finger`, `set_grip_force`, `held`, `pad_contacts`). Replace
  `SimReader` with calls to the reader (`open`, `close`, `is_open`, `is_closed`), `Perception`
  with cameras and `ScriptedDetector` with the anomaly detector. `put_plate`,
  `overlaps`, `set_plate_contact` and `collisions` are simulation helpers (scenarios, planning
  checks) with no real-robot counterpart.

## What differs from your physics and layout

Each item is in `configs/plate_bt.yaml` with its reason:

- `impratio` 10: under the default, the plate creeps out of a held grasp.
- Armature 0.05 kg·m² on `joint5`: without it, turning joint5 shakes the fingers loose.
- Plate contacts take the pads' parameters, so the held plate does not creep.
- Your stack in OT-2 slot 2 is left out: used plates go on a storage pile and rejected plates on a
  bin pile, both on the table (Kevin, 2026-10-22). Every grasp then holds the plate's long edges.
- The new-plate area is on the table: on your waiting pad in slot 3 a long-edge grasp collides
  with the temperature module (the heater needs the long edges).
- Your tray is a drawer on a slide, with a slot cut through the housing behind the recess; your
  model has it fixed.
- The tray grasp holds the plate's top 6 mm, at 4 N and half speed: your tray guides are inside
  the pads' reach.
