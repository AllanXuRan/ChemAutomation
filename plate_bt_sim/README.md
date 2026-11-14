# plate_bt_sim: your behaviour tree on the plate workflow, with physical failures

Kevin's FYP simulation, built on this repository. It runs Xu's overall workflow (a new plate to
the heater, formulation inspection, the reader, then storage; a plate with bubbles or a fallen
tip goes to a bin) with your tree, `chem_lab_bt/lab_behavior_tree.py`, unchanged. The tree drives
your simulated xArm5 and G2 through your `PhysicsRuntime`. Nothing outside this folder is edited:
your tree, loader, IK and runtime are imported from the repository root.

What it adds to your cell: a new-plate area, a storage pile and a bin pile on the table; the
reader's tray as a drawer; the failures Kevin and Xu named, caused physically where they can be
(a new plate left skewed, a skewed plate on the storage pile, a pipette tip left on the plate,
bubbles), and checks that notice them, so your recovery branches run on physics rather than
injected faults. `StabiliseGrasp` and `RetractAndRealign` are implemented for your tree, but no
scenario causes a slipping plate or a misaligned tray.

This is a second MuJoCo backend next to your `chem_lab_bt/mujoco_backend.py`. It does not set
`physical_flow`, so `build_tree` builds `04_NORMAL_ONE_PLATE` and this package puts its five
new-plate actions in front. To try another tree, build it in `src/plate_bt/tree.py` (or call your
own builder there) and give any new action names an entry in `ACTIONS` in `src/plate_bt/backend.py`.

## Running it

MuJoCo 3.16, py_trees 2.6, numpy 2.4 (your `requirements.txt`), plus PyYAML and Pillow. From this
folder, `plate_bt_sim/`:

```bash
export MUJOCO_GL=egl            # headless rendering; leave it out on a desktop
python tests/test_plate_bt.py                                                # checks, the backend's contract, one plate end to end
python scripts/bt/check_layout.py                                            # reach of every area, the drawer
python scripts/bt/run_plates.py --scenario configs/plate_bt_run.yaml         # four plates (clean, skewed new plate, bubbles, skewed storage pile); writes data/plate_bt_<ts>/
python scripts/bt/render_run.py data/plate_bt_<ts>                           # clips and stills
python scripts/bt/run_plates.py --scenario configs/plate_bt_continuous.yaml  # five plates in one run
python scripts/bt/render_grid.py data/plate_bt_<ts>                          # one video, four cameras
```

A run writes `states.npz` (one record per 1/30 s tick: qpos, slip risk, grip force, the active
action and the fault), `trace.json` (every action, check, measurement and scenario event, and
each plate's outcome) and `run.log` (your tree's own messages).

## Tested

On 13 Nov 2026, against your `main` at f449b59 (the four-plate workflow) with
`fix/relative-mesh-paths` merged, MuJoCo 3.16.0, py_trees 2.6.0, numpy 2.4.6:

- `tests/test_plate_bt.py`: all 8 pass, including one clean plate end to end.
- `configs/plate_bt_continuous.yaml`: as intended. Plates 1, 2 and 4 are measured and end on the
  storage pile; plate 3 (bubbles) and plate 5 (a pipette tip on it) end on the bin pile; plate 2,
  skewed in the stack, is grasped again after `RepositionForGrasp`. 1024 s simulated in 403 s; no
  contact deeper than your 0.3 mm guard.
- `configs/plate_bt_run.yaml` without the wet plate (five plates): as intended. The clean plate,
  the skewed new plate (`RepositionForGrasp`), the drawer stopping short (`RetractAndRealign`) and
  the skewed plate on the storage pile (`RepositionForPlacement`) are measured; the bubbles plate
  goes to the bin. 1151 s simulated in 272 s; no contact deeper than 0.3 mm.

**Open: the wet plate.** With the wet plate (friction 0.15) the full six-plate run stops at plate 4:
after `StabiliseGrasp` the slip risk stays above 0.7, and your tree latches its stop when the
budget of three interventions is spent. With your new `noslip_iterations` 10 the pads' contact
forces carry tangential components that cancel each other and grow with the grip, and the slip risk
(the sum of each contact's friction over the friction available) counts them, so a firmer grip no
longer lowers it. Measured on the plate, the pads' net friction is about its weight (0.39 N against
16 N of squeeze at 8 N). Summing friction as vectors fixes the reading at 2 N but not at the
heater's set-down after the grip is raised; switching the no-slip solver off lets the plate slide
in the grasp on the way to storage. Not fixed yet.

## How it fits your tree

| Your code | What plate_bt does with it |
|---|---|
| `build_tree` | Called as it is; `tree.py` inserts five actions at the head of `04_NORMAL_ONE_PLATE`: `FetchNewPlate`, `GraspNewPlate`, `VerifyNewPlateGrasp`, `PlaceOnHeater`, `VerifyOnHeater`. They use your fault kinds (`grasp`, `placement`), so your recovery branches handle them |
| `RobotBackend` | Implemented by `backend.MujocoBackend`. `ACTIONS` at the top of `backend.py` maps each of your action names to the method that runs it |
| `Blackboard`, `Settings`, `bt_config.json` | Used as they are: retry limits, the slip threshold 0.7, the preventive budget |
| Your runner (`main()`) | `runner.run` follows it: tick, finish a plate, reset the tree, next plate, stop on a latched emergency. Before each tick, `backend.sense()` writes the slip risk and checks for a drop, where your comment leaves room for sensor updates |
| `load_physics_model`, `PhysicsRuntime`, `jac_ik`, `PhysicalGraspController` | Used for every motion, so your collision guard checks every command; `controller.plate` and `controller.plate_bodies` are switched to the plate being handled |

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
- Your stack in OT-2 slot 2 and your source stack on the table are left out: new plates come from
  the new-plate area, used plates go on a storage pile and rejected plates on a bin pile, all on
  the table (Kevin, 2026-10-22). Every grasp then holds the plate's long edges.
- The heater is held as your `lab_motion.py` holds it, since the temperature block moved 3 cm
  further out: the TCP 8 mm past the plate's centre toward the robot, and the plate set 8 mm short
  of `ot2_pickup` in y (you use 6 mm; 8 mm keeps the 35 mm approach in reach). At the site itself
  the approach is out of reach with a level flange.
- Your grasp controller's `plate_bodies` is set to the plate being handled only, so the collision
  guard and `held` see that plate and no other.
- The grip is gentle, 2 N; `StabiliseGrasp` raises it to your limit, 8 N.
- The new-plate area is on the table: on your waiting pad in slot 3 a long-edge grasp collides
  with the temperature module (the heater needs the long edges).
- Your tray is a drawer on a slide, with a slot cut through the housing behind the recess; your
  model has it fixed.
- The tray grasp holds the plate's top 6 mm, at 4 N and half speed: your tray guides are inside
  the pads' reach.
