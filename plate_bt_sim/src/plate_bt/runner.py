"""The tick loop: Allan's runner logic over the simulated workcell.

Each tick: sense (slip risk, drop), tick Allan's tree, let the backend finish braking, step the
physics by one tick. When a plate finishes, the tree is reset and the next plate starts, as in
Allan's `main()`; a latched emergency stop ends the run. Writes, into `out`:

    states.npz   one record per tick: time, qpos, slip risk and slip speed, grip force, plate number,
                 the active action and the fault (as indices into `names` and `faults`)
    trace.json   the backend's events (actions, checks, measurements, scenario events), the
                 outcome of each plate, and the collision audit
"""
import json
import time

import numpy as np
import py_trees

from plate_bt import tree as T
from plate_bt.backend import MujocoBackend
from plate_bt.world import GUARD_DEPTH, SimWorld
from mujoco_bridge.load import CHEM

Status = py_trees.common.Status
AUDIT_EVERY = 3              # ticks between collision audits


def audit(world, k):
    """Contacts deeper than Allan's guard depth that are not the grasp: the robot touching
    anything, or plates pressing into something. Returns (pair, depth in mm) for each."""
    m, d = world.model, world.data
    robot = world.runtime.robot_bodies
    plates = set(world.plates)
    out = []
    for c in d.contact[:d.ncon]:
        if c.dist > -GUARD_DEPTH:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if k is not None and {b1, b2} & {world.plates[k]} and {b1, b2} & world.finger_bodies:
            continue
        if not ({b1, b2} & robot or {b1, b2} & plates):
            continue
        out.append((f"{m.geom(c.geom1).name or m.body(b1).name}/{m.geom(c.geom2).name or m.body(b2).name}",
                    round(-float(c.dist) * 1000, 2)))
    return out


def run(cfg, scenario, out, max_sim_s=1500.0, log=print):
    out.mkdir(parents=True, exist_ok=True)
    lbt = T.allan_bt()
    settings = lbt.load_settings(CHEM / "chem_lab_bt" / "bt_config.json")
    settings.plates = len(scenario)
    world = SimWorld(cfg, scenario.bodies, scenario.tips)
    state = lbt.Blackboard(compact_log=True)
    backend = MujocoBackend(state, world, cfg, scenario, settings)
    bt = T.build(state, backend, settings)
    backend.start()
    names, faults = ["idle"], ["none"]
    rec = {k: [] for k in ("time", "qpos", "slip_risk", "slip_mm_s", "grip_n", "plate", "action", "fault", "carrying")}
    outcomes, collisions = [], {}
    t_wall = time.time()
    backend.on_plate_start(state.plate_id)
    finished = 0
    for tick in range(int(max_sim_s / world.dt)):
        state.tick_number = tick
        backend.sense()
        bt.tick()
        backend.service()
        world.step()
        stage = backend.current
        if stage not in names:
            names.append(stage)
        fault = state.fault or "none"
        if fault not in faults:
            faults.append(fault)
        for key, v in (("time", world.time), ("qpos", world.data.qpos.copy()), ("slip_risk", state.slip_risk),
                       ("slip_mm_s", backend.slip_mm_s),
                       ("grip_n", world.grip_force()), ("plate", state.plate_id), ("action", names.index(stage)),
                       ("fault", faults.index(fault)), ("carrying", state.carrying_plate)):
            rec[key].append(v)
        if tick % AUDIT_EVERY == 0:
            for pair, depth in audit(world, backend.k if world.held() else None):
                c = collisions.setdefault(pair, dict(ticks=0, max_mm=0.0, first_t=round(world.time, 3)))
                c["ticks"] += 1
                c["max_mm"] = max(c["max_mm"], depth)
        if state.emergency_latched:
            if not state.stop_sent:
                bt.tick()
            for _ in range(round(1.0 / world.dt)):      # the arm brakes to rest
                backend.service()
                world.step()
            outcomes.append(dict(plate=state.plate_id, outcome="halted", reason=state.emergency_reason,
                                 t=round(world.time, 3)))
            log(f"HALTED: {state.emergency_reason}")
            break
        if state.plate_finished:
            outcomes.append(dict(plate=state.plate_id, outcome=state.plate_outcome, t=round(world.time, 3),
                                 retries=dict(state.retries), preventive=state.preventive_this_plate))
            log(f"plate {state.plate_id}: {state.plate_outcome} at t={world.time:.1f} s")
            finished += 1
            if finished >= settings.plates:
                break
            bt.root.stop(Status.INVALID)
            state.next_plate()
            backend.on_plate_start(state.plate_id)
    else:
        outcomes.append(dict(plate=state.plate_id, outcome="timeout", t=round(world.time, 3)))
    np.savez_compressed(out / "states.npz", **{k: np.array(v) for k, v in rec.items()},
                        names=np.array(names), faults=np.array(faults))
    summary = dict(plates=len(scenario), sim_s=round(world.time, 2), wall_s=round(time.time() - t_wall, 1),
                   ticks=len(rec["time"]), tick_hz=cfg["tick_hz"], outcomes=outcomes,
                   measured=state.measured_plates, rejected=state.rejected_plates,
                   preventive_interventions=state.preventive_interventions,
                   recoveries=state.recovery_actions_completed, collisions=collisions,
                   scenario=scenario.plates, start=scenario.start, piles={a: [b + 1 for b in ks] for a, ks in backend.piles.items()})
    (out / "trace.json").write_text(json.dumps(dict(summary=summary, events=backend.trace), indent=1))
    return summary, world
