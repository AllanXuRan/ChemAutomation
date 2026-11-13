"""plate_bt (src/plate_bt/): the checks on hand-made numbers, the backend's
contract with Allan's tree, and one plate end to end in his MuJoCo workcell. Needs MuJoCo and
py_trees:

    MUJOCO_GL=egl python tests/test_plate_bt.py      (or pytest tests/test_plate_bt.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from plate_bt import checks, runner, tree  # noqa: E402
from plate_bt.backend import MujocoBackend  # noqa: E402
from plate_bt.events import Scenario  # noqa: E402
from plate_bt.world import Pose, SimWorld  # noqa: E402

CFG = yaml.safe_load((REPO / "configs" / "plate_bt.yaml").read_text())
TOL = CFG["tolerance"]


def test_pose_error_treats_a_half_turn_as_the_same_footprint():
    a, b = Pose((0, 0, 0), 0.0), Pose((0.003, 0.004, 0), np.pi + np.radians(2))
    dxy, dyaw = checks.pose_error(a, b)
    assert abs(dxy - 0.005) < 1e-9 and abs(dyaw - 2.0) < 1e-6


def test_grasp_precheck_and_placement():
    taught = Pose((0.32, -0.18, 0.0005), 0.0)
    assert checks.grasp_precheck(taught.moved((0.001, 0, 0)), taught, TOL).ok
    assert not checks.grasp_precheck(taught.moved((0.004, -0.003, 0), np.radians(9)), taught, TOL).ok
    assert checks.placed(taught, 0.2, taught, TOL).ok
    assert not checks.placed(taught.moved(dyaw=np.radians(8)), 0.2, taught, TOL).ok     # skewed on the pile
    assert not checks.placed(taught, 3.0, taught, TOL).ok                                # tilted


def test_clear_of_the_reader():
    keep_out = CFG["reader"]["keep_out"]
    assert checks.clear_of([CFG["reader"]["clear_xyz"]], keep_out).ok
    assert not checks.clear_of([(0.0, 0.18, 0.4), (0.07, 0.45, 0.2)], keep_out).ok     # one point over the tray


def test_slip_risk_is_the_share_of_friction_used():
    weight = 0.0396 * 9.81
    dry = [(2.0, weight / 2, 1.2)] * 2            # (normal, tangential, mu) per pad
    wet = [(2.0, weight / 2, 0.15)] * 2
    assert checks.slip_risk(dry) < 0.1 < 0.6 < checks.slip_risk(wet) < 0.7
    assert checks.slip_risk([]) == 0.0


def test_scenario_start_counts_the_bodies():
    sc = Scenario([{}, {"anomaly": "tip"}, {"storage_plate_dyaw": 8}], {"storage": 3, "bin": 1, "supply": "stack"})
    assert sc.stacked and sc.bodies == 3 + 3 + 1 + 1 and sc.tips == 1
    try:
        Scenario([{}], {"supply": "conveyor"})
        raise AssertionError("an unknown supply must raise")
    except ValueError:
        pass


def backend():
    lbt = tree.allan_bt()
    settings = lbt.load_settings(REPO.parent / "chem_lab_bt" / "bt_config.json")
    state = lbt.Blackboard()
    world = SimWorld(CFG, 1)
    b = MujocoBackend(state, world, CFG, Scenario([{}]), settings)
    b.start()
    return lbt, state, world, b


def test_backend_contract():
    """Allan's RobotBackend: start once, poll until done, cancel stops the move."""
    lbt, state, world, b = backend()
    b.start_action("WaitForFormulationComplete", 1)
    try:
        b.start_action("WaitForFormulationComplete", 1)
        raise AssertionError("a second start must raise")
    except RuntimeError:
        pass
    assert b.poll_action("WaitForFormulationComplete") == lbt.Status.RUNNING
    try:
        b.poll_action("InspectFormulation")
        raise AssertionError("a poll without start must raise")
    except RuntimeError:
        pass
    b.cancel_action("WaitForFormulationComplete")
    assert "WaitForFormulationComplete" not in b.active and b.braking is not None
    b.start_action("InspectFormulation", 1)        # an action started now waits for the arm to stop
    assert b.poll_action("InspectFormulation") == lbt.Status.RUNNING


def test_tree_has_the_new_plate_steps_first():
    lbt, state, world, b = backend()
    bt = tree.build(state, b, lbt.load_settings(REPO.parent / "chem_lab_bt" / "bt_config.json"))
    main = tree.find(bt.root, "04_NORMAL_ONE_PLATE")
    assert [c.name for c in main.children[:6]] == ["FetchNewPlate", "GraspNewPlate", "VerifyNewPlateGrasp",
                                                  "PlaceOnHeater", "VerifyOnHeater", "WaitForFormulationComplete"]


def test_one_clean_plate_end_to_end():
    """New-plate area, heater, reader (in and out of the drawer), storage pile: measured, the
    drawer closed again, and no contact deeper than Allan's guard outside the grasp."""
    with tempfile.TemporaryDirectory() as tmp:
        summary, world = runner.run(CFG, Scenario([{"name": "clean"}]), Path(tmp), log=lambda s: None)
    assert [o["outcome"] for o in summary["outcomes"]] == ["measured"]
    assert summary["piles"] == {"storage": [1], "bin": []}
    assert not summary["collisions"], summary["collisions"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
