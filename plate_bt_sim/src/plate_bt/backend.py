"""MujocoBackend: Allan's RobotBackend over the simulated workcell.

Allan's tree (`chem_lab_bt/lab_behavior_tree.py`) calls `start_action` once per action, then
`poll_action` every tick until it returns SUCCESS or FAILURE, and `cancel_action` when a
higher-priority branch takes over. ACTIONS below maps each action name to a method that returns a
generator; each poll runs it for one tick of commands (skills.py). A method returns False to fail
the action; it sets `state.fault` itself only for the formulation anomaly, since Allan's tree
gives every other checked action its fault kind (`fault_on_failure`).

`sense()` runs before every tick of the tree, as Allan's runner leaves room for: it writes the
slip risk of a carried plate to the blackboard and latches the emergency stop if the plate drops.

Recoveries follow Xu's diagram: a grasp, alignment or placement is first tried at its taught pose
and, after a failed check, at the pose perception measures.

Used plates go on the storage pile and rejected plates on the bin pile, both on the table. The
reader's drawer is closed while the reader measures, and moves only while the arm is clear of
the reader (Kevin, 2026-10-22)."""
import numpy as np
import py_trees

from plate_bt import checks
from plate_bt.perception import Perception, ScriptedDetector
from plate_bt.reader import SimReader
from plate_bt.scene import TABLE_TOP, TIP_RADIUS
from plate_bt.skills import Skills
from plate_bt.world import PLATE_HALF, PlanError, Pose

Status = py_trees.common.Status
PLATE_H = 2 * PLATE_HALF[2]
HOVER_M = 0.012            # aligning over the tray: the plate's underside this far above its seat
NEAR_M = 0.05              # a target this close (horizontally) is reached by a straight line, not a transit

# The steps tree.py puts before Allan's one-plate sequence, and the actions whose cached success
# a recovery must clear so the tree runs them again (Allan's RetryRecovery clears his own).
NEW_PLATE_GRASP = ("FetchNewPlate", "GraspNewPlate", "VerifyNewPlateGrasp")
HEATER_PLACE = ("PlaceOnHeater", "VerifyOnHeater")

ACTIONS = {
    # before Allan's sequence (tree.py)
    "FetchNewPlate": "fetch_new_plate",
    "GraspNewPlate": "grasp_new_plate",
    "VerifyNewPlateGrasp": "test_lift",
    "PlaceOnHeater": "place_on_heater",
    "VerifyOnHeater": "verify_on_heater",
    # Allan's one-plate sequence
    "WaitForFormulationComplete": "wait_for_formulation",
    "InspectFormulation": "inspect_formulation",
    "MoveToOT2Pickup": "move_to_heater",
    "GraspPlate": "grasp_on_heater",
    "VerifyGrasp": "test_lift",
    "TransferToReader": "transfer_to_reader",
    "AlignWithReader": "align_with_reader",
    "VerifyReaderAligned": "verify_reader_aligned",
    "LoadPlateIntoReader": "load_plate_into_reader",
    "VerifyReaderLoaded": "verify_reader_loaded",
    "StartReaderMeasurement": "start_reading",
    "WaitForReadingComplete": "wait_for_reading",
    "RetrievePlate": "retrieve_plate",
    "ReturnToStorageSlot": "return_to_storage",
    "PlacePlateInSlot": "place_in_storage",
    "VerifyReturned": "verify_returned",
    # Allan's recoveries
    "CheckQuarantineSafe": "check_quarantine_safe",
    "MoveRejectedPlateToBin": "move_rejected_plate_to_bin",
    "RepositionForGrasp": "reposition_for_grasp",
    "RetractAndRealign": "retract_and_realign",
    "RepositionForPlacement": "reposition_for_placement",
    "StabiliseGrasp": "stabilise_grasp",
}


class MujocoBackend:
    """Implements Allan's RobotBackend contract (start_action, poll_action, cancel_action,
    emergency_stop) for SimWorld. `trace` collects what happened, for the run's records."""

    def __init__(self, state, world, cfg, scenario, settings):
        self.state, self.w, self.cfg, self.scenario, self.settings = state, world, cfg, scenario, settings
        self.sk = Skills(world, cfg)
        self.per = Perception(world, cfg)
        self.det = ScriptedDetector(scenario.anomalies)
        self.reader = SimReader(world, cfg)
        self.tol = cfg["tolerance"]
        self.active, self.braking, self.polled = {}, None, False
        self.current = "idle"         # the action started last (Allan's recoveries leave current_stage as it was)
        self.trace = []
        self.k = None                 # index of the current plate's body
        self.next_body = 0            # next parked plate a person may put down
        self.supply = []              # plate bodies stacked on the new-plate area, plate 1 first
        self.tips_used = 0
        self.piles = {"storage": [], "bin": []}    # plate bodies put on each pile by the robot
        self.taught = self._taught()
        self.target, self.grasp_area, self.place_area, self.below = dict(self.taught), None, None, None
        self.holding = None           # which edges the pads hold: long or short
        self.on_tray = None           # the plate's pose on the tray when the drawer closed
        self.at = "home"              # the area the TCP is in
        self.stabilised = False       # StabiliseGrasp ran for this plate: full force, slower moves
        self.risk, self.slip_mm_s, self.prev_rel, self.carry_seen = 0.0, 0.0, None, None

    # ---- Allan's RobotBackend contract -------------------------------------------------------
    def start_action(self, action, plate_id):
        if action in self.active:
            raise RuntimeError(f"Duplicate start for {action}")
        if action not in ACTIONS:
            raise NotImplementedError(f"MujocoBackend has no action {action}")
        self.active[action] = getattr(self, ACTIONS[action])()
        self.current = action
        self.note("start", action)

    def poll_action(self, action):
        self.polled = True
        if self.braking is not None:          # an interrupted move comes to rest first
            try:
                next(self.braking)
                return Status.RUNNING
            except StopIteration:
                self.braking = None
        gen = self.active.get(action)
        if gen is None:
            raise RuntimeError(f"Poll called without start: {action}")
        try:
            next(gen)
            return Status.RUNNING
        except StopIteration as stop:
            ok = stop.value is not False
        except PlanError as e:
            # a motion that cannot be planned is not a grasp, alignment or placement failure the
            # tree could recover from: latch the stop, so Allan's AsyncAction sets no fault
            ok = False
            self.state.emergency_latched = True
            self.state.emergency_reason = f"cannot plan {action}: {e}"
            self.note("plan error", action, error=str(e))
        del self.active[action]
        self.note("done" if ok else "failed", action, tcp=np.round(self.w.tcp_pose().xyz, 4).tolist(),
                  **({"blocked": self.w.blocked} if not ok and self.w.blocked else {}))
        return Status.SUCCESS if ok else Status.FAILURE

    def cancel_action(self, action):
        gen = self.active.pop(action, None)
        if gen is not None:
            gen.close()
            self.braking = self.sk.brake()
            self.at = None            # the arm stops between areas
            self.note("cancel", action)

    def emergency_stop(self):
        for action in tuple(self.active):
            self.cancel_action(action)
        if self.braking is None:
            self.braking = self.sk.brake()
        self.note("emergency stop", None, reason=self.state.emergency_reason)

    # ---- around the tree's tick ----------------------------------------------------------------
    def sense(self):
        """Before each tick: the slip risk of a carried plate, and the drop check."""
        s, w = self.state, self.w
        if not s.carrying_plate or self.k is None or self.sk.carry is None:
            s.slip_risk, self.risk, self.slip_mm_s, self.prev_rel = 0.0, 0.0, 0.0, None
            return
        # friction use while the pads alone hold the plate (a support under it takes load off the
        # pads without any slip), smoothed over slip.filter_s; and how fast it slides, for the record
        rel_now = w.plate_in_tcp(self.k)[:3, 3]
        if self.sk.carry is not self.carry_seen:      # a new grasp: start the filters afresh
            self.carry_seen, self.prev_rel, self.slip_mm_s, self.risk = self.sk.carry, None, 0.0, 0.0
        a = min(1.0, w.dt / self.cfg["slip"]["filter_s"])
        if self._airborne():
            self.risk += a * (checks.slip_risk(w.pad_contacts()) - self.risk)
            if self.prev_rel is not None:
                self.slip_mm_s += a * (np.linalg.norm(rel_now - self.prev_rel) / w.dt * 1000 - self.slip_mm_s)
        else:
            self.risk, self.slip_mm_s = 0.0, 0.0
        self.prev_rel = rel_now
        s.slip_risk = self.risk
        k, rel = self.sk.carry
        T = np.eye(4)
        T[:3, :3] = w.data.site_xmat[w.tcp].reshape(3, 3)
        T[:3, 3] = w.data.site_xpos[w.tcp]
        expected = (T @ rel)[:3, 3]
        if expected[2] - w.data.xpos[w.plates[k]][2] > self.tol["drop_m"] and not s.emergency_latched:
            s.emergency_latched = True
            s.emergency_reason = "plate dropped while carried"
            self.note("drop", None)

    def service(self):
        """After each tick: keep braking when no action ran this tick."""
        if not self.polled and self.braking is not None:
            try:
                next(self.braking)
            except StopIteration:
                self.braking = None
        self.polled = False

    def on_plate_start(self, plate_id):
        """Reset per-plate settings, and make the scenario's changes for this plate that do not
        wait for a step: where the reader's drawer stops when it opens, a plate a person puts on
        the storage pile."""
        self.target = dict(self.taught)
        self.sk.speed_scale = 1.0
        self.stabilised = False
        self.w.set_grip_force(self.cfg["grasp"]["force_n"])
        entry = self.scenario.plate(plate_id)
        short = float(entry.get("drawer_short", 0.0))
        self.reader.open_error = short
        if short:
            self.note("event", None, what=f"the reader's drawer will stop {short * 1000:.0f} mm short of open")
        if self.scenario.stacked:     # the plate is the top of the supply stack
            below = len(self.scenario) - plate_id
            self.target["new_plate"] = self.taught["new_plate"].moved((0, 0, PLATE_H * below))
        if "storage_plate_dyaw" in entry:
            k = self._take_parked()
            top, _ = self.per.pile_top("storage")
            self.w.put_plate(k, Pose(top.xyz + [0, 0, PLATE_H], top.yaw + np.radians(entry["storage_plate_dyaw"])))
            self._no_overlap(k)
            self.note("event", None, what=f"a person puts a plate on the storage pile, turned {entry['storage_plate_dyaw']} deg")

    def start(self):
        """Put the arm at home, fingers open, and let everything settle for a second."""
        w, home = self.w, self.cfg["home"]
        q = w.ik(home["xyz"], np.radians(home["point"]), w.joints())
        w.kin.set_angles(w.model, w.data, w.info, q)
        w.runtime.reset_targets()
        self._set_up()
        for _ in range(round(1.0 / w.dt)):
            w.step()

    def _set_up(self):
        """The cell before the first plate (the scenario's `start`): plates already on the piles,
        and the new plates stacked on the new-plate area, the first plate on top."""
        sc = self.scenario
        for area in ("storage", "bin"):
            n = sc.start.get(area, 0)
            for level in range(n):
                k = self._take_parked()
                self.w.put_plate(k, self.taught[area].moved((0, 0, PLATE_H * level)))
                self._no_overlap(k)
            if n:
                self.note("event", None, what=f"{n} plate{'s' if n > 1 else ''} already on the {area} pile")
        if sc.stacked:
            self.supply = [self._take_parked() for _ in sc.plates]
            for i in reversed(range(len(sc))):
                entry, k = sc.plates[i], self.supply[i]
                off = entry.get("new_plate", {})
                level = len(sc) - 1 - i
                if "friction" in entry:
                    self.w.set_plate_contact(k, entry["friction"])
                self.w.put_plate(k, self.taught["new_plate"].moved((off.get("dx", 0), off.get("dy", 0), PLATE_H * level),
                                                                   np.radians(off.get("dyaw", 0))))
                self._no_overlap(k)
            self.note("event", None, what=f"{len(sc)} new plates stacked on the new-plate area")

    def note(self, kind, action, **data):
        self.trace.append(dict(t=round(self.w.time, 3), tick=self.state.tick_number, plate=self.state.plate_id,
                               kind=kind, action=action, **data))

    # ---- helpers -------------------------------------------------------------------------------
    def _taught(self):
        """Each area's taught plate pose: Allan's site (moved by the area's `offset`, m), or a pad's
        centre on the table."""
        w, out = self.w, {}
        for name, a in self.cfg["areas"].items():
            if "site" in a:
                xyz = w.data.site_xpos[w.kin.site_id(w.model, a["site"])].copy() + np.asarray(a.get("offset", (0, 0, 0)), float)
            else:
                xyz = np.array([*a["xy"], TABLE_TOP + self.cfg["pad_size"][2] + PLATE_HALF[2]])
            out[name] = Pose(xyz, np.radians(a["plate_yaw"]))
        return out

    def _point(self, area, plate_yaw):
        """Pointing that grasps a plate at plate_yaw the way the area's config says."""
        a = self.cfg["areas"][area]
        return plate_yaw + np.radians(a["point"] - a["plate_yaw"])

    def _airborne(self):
        """No support under the held plate: it touches nothing but the pads below its middle (a
        pipette tip lying on it does not count)."""
        g = self.w.plate_geom[self.k]
        z = self.w.data.xpos[self.w.plates[self.k]][2]
        for c in self.w.data.contact[:self.w.data.ncon]:
            if g in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == g else c.geom1
                if int(self.w.model.geom_bodyid[other]) not in self.w.finger_bodies and c.pos[2] < z:
                    return False
        return True

    def _take_parked(self):
        if self.next_body >= len(self.w.plates):
            raise RuntimeError("no parked plate left: the scenario needs more plates")
        k, self.next_body = self.next_body, self.next_body + 1
        return k

    def _no_overlap(self, k):
        hits = self.w.overlaps(k)
        if hits:
            raise RuntimeError(f"plate {k + 1} put down overlapping {hits}")

    def _check(self, check):
        self.note("check", self.current, check=check.name, ok=check.ok, **check.values)
        return check.ok

    def _tcp_for_plate(self, plate):
        """TCP position and pointing that put the carried plate at `plate`, from where the plate
        sits in the grasp now (perception of the held plate: it may have slid since the lift)."""
        rel = self.w.plate_in_tcp(self.k)
        self.sk.carry = (self.k, rel)
        P = np.eye(4)
        P[:3, :3] = [[np.cos(plate.yaw), -np.sin(plate.yaw), 0], [np.sin(plate.yaw), np.cos(plate.yaw), 0], [0, 0, 1]]
        P[:3, 3] = plate.xyz
        T = P @ np.linalg.inv(rel)
        return T[:3, 3], float(np.arctan2(T[1, 0], T[0, 0]))

    def _move(self, area, xyz, point):
        """Straight there if near, else a transit through the areas' fronts, leaving the area the
        TCP is in at that area's approach height."""
        if np.hypot(*(self.w.tcp_pose().xyz[:2] - np.asarray(xyz)[:2])) < NEAR_M:
            ok = yield from self.sk.line(xyz, point)
        else:
            leave_z = self.taught[self.at].xyz[2] + self._above(self.at) if self.at in self.taught else None
            here = None if self.at is None else self.cfg["areas"].get(self.at, {})
            ok = yield from self.sk.transit(self.cfg["areas"].get(area, {}), xyz, point, here, leave_z)
        self.at = area if ok else self.at
        return ok

    def _grasp_offsets(self, area):
        """(height, pads beyond the plate's centre) of the area's grasp."""
        a = self.cfg["areas"][area]
        return a.get("grasp_h", 0.0), a.get("pads_beyond", 0.0)

    def _approach(self, area, plate):
        """Empty gripper to above a plate, pads opened for the area's grasp."""
        self.w.set_finger(self.w.opening_for(self.cfg["areas"][area]["grasp"], self.cfg["grasp"]["clearance"]))
        point = self._point(area, plate.yaw)
        above = self.sk.tcp_for(plate, point, *self._grasp_offsets(area)) + [0, 0, self._above(area)]
        return (yield from self._move(area, above, point))

    def _grasp(self, area):
        """Pre-check, down, close. The pre-check compares where the plate is with the pose the
        grasp is planned for (taught, or measured after a recovery)."""
        self.grasp_area = area
        a = self.cfg["areas"][area]
        self.holding = a["grasp"]
        planned = self.target[area]
        seen, _ = self.per.plate(self.k)
        if not self._check(checks.grasp_precheck(seen, planned, self.tol)):
            return False
        point = self._point(area, planned.yaw)
        if not (yield from self.sk.line(self.sk.tcp_for(planned, point, *self._grasp_offsets(area)), point)):
            return False
        if "force_n" in a and not self.stabilised:
            self.w.set_grip_force(a["force_n"])
        self.sk.carry_rad_s = a.get("carry_rad_s")
        yield from self.sk.close()
        self.sk.carry = (self.k, self.w.plate_in_tcp(self.k))
        return True

    def test_lift(self):
        """Lift a little, wait, and check the plate came along level."""
        w = self.w
        z_plate, z_tcp = w.plate_pose(self.k)[0].xyz[2], w.tcp_pose().xyz[2]
        if not (yield from self.sk.line(w.tcp_pose().xyz + [0, 0, self.cfg["grasp"]["test_lift"]])):
            return False
        yield from self.sk.wait(self.cfg["grasp"]["settle_s"])
        pose, tilt = w.plate_pose(self.k)
        ok = self._check(checks.lift_check(w.held(), pose.xyz[2] - z_plate, w.tcp_pose().xyz[2] - z_tcp, tilt, self.tol))
        self.sk.carry = (self.k, w.plate_in_tcp(self.k)) if ok else None
        return ok

    def _place(self, area, seat):
        """Carry the plate above its seat, lower it until it almost touches, open the pads."""
        g = self.cfg["grasp"]
        xyz, point = self._tcp_for_plate(seat.moved((0, 0, self._above(area))))
        if not (yield from self._move(area, xyz, point)):
            return False
        return (yield from self._lower_and_release(seat, self._release_above(area)))

    def _lower_and_release(self, seat, release_above):
        """Lower the plate to `release_above` over its seat, stopping early if it touches down
        (a guarded move: the plate may hang lower in the grasp than when it was lifted), then
        open the pads."""
        xyz, point = self._tcp_for_plate(seat.moved((0, 0, release_above)))
        scale, self.sk.speed_scale = self.sk.speed_scale, self.sk.speed_scale * self.cfg["speed"]["set_down_factor"]
        ok = yield from self.sk.line(xyz, point, stop=lambda: not self._airborne())
        self.sk.speed_scale = scale
        held, _ = self.per.plate(self.k)
        ok = ok and (yield from self._release())
        let_go, _ = self.per.plate(self.k)
        self.note("set down", self.current, seat=np.round(seat.xyz, 4).tolist(),
                  held_off_mm=round(checks.pose_error(held, seat)[0] * 1000, 2),
                  released_off_mm=round(checks.pose_error(let_go, seat)[0] * 1000, 2))
        return ok

    def _release_above(self, area):
        return self.cfg["areas"][area].get("release_above", self.cfg["grasp"]["release_above"])

    def _release(self):
        q = self.w.opening_for(self.holding, self.cfg["grasp"]["clearance"])
        yield from self.sk.fingers(q)
        self.sk.carry, self.sk.carry_rad_s = None, None
        if not self.stabilised:
            self.w.set_grip_force(self.cfg["grasp"]["force_n"])
        return True

    def _above(self, area):
        """Approach and retreat height over the area's seat."""
        return self.cfg["areas"][area].get("above", self.cfg["grasp"]["above"])

    def _retreat(self, area):
        return (yield from self.sk.line(self.w.tcp_pose().xyz + [0, 0, self._above(area)]))

    def _home(self):
        home = self.cfg["home"]
        return (yield from self._move("home", home["xyz"], np.radians(home["point"])))

    def _pile_seat(self, area):
        """Where the next plate goes on a pile: the taught (or corrected) position and yaw, on the
        measured top of the pile."""
        top, _ = self.per.pile_top(area, exclude=self.k)
        self.below = top
        t = self.target[area]
        return Pose((t.xyz[0], t.xyz[1], top.xyz[2] + PLATE_H), t.yaw)

    def _verify_pile(self, area):
        """In line with the plate below (or the pad), then up and home."""
        pose, tilt = self.per.plate(self.k)
        seat = Pose(self.below.xyz + [0, 0, PLATE_H], self.below.yaw)
        if not self._check(checks.placed(pose, tilt, seat, self.tol, f"{area} placement")):
            return False
        self.piles[area].append(self.k)
        if not (yield from self._retreat(area)):
            return False
        return (yield from self._home())

    def _clear_of_reader(self):
        """Whether the TCP and the flange are outside the reader's keep-out box."""
        w = self.w
        flange = w.data.xpos[w.model.body("link_eef").id]
        return self._check(checks.clear_of([w.tcp_pose().xyz, flange], self.cfg["reader"]["keep_out"],
                                           "arm clear of the reader"))

    # ---- actions: the new plate ----------------------------------------------------------------
    def fetch_new_plate(self):
        """The next plate: the top of the supply stack, or one a person puts on the new-plate area
        (off its taught pose if the scenario says so); the arm goes above where it is taught to be."""
        entry = self.scenario.plate(self.state.plate_id)
        if self.scenario.stacked:
            self.k = self.supply[self.state.plate_id - 1]
            self.w.set_active(self.k)
            self.note("event", None, what="the robot takes the top plate of the supply stack", plate_body=self.k + 1)
            return (yield from self._approach("new_plate", self.target["new_plate"]))
        self.k = self._take_parked()
        self.w.set_active(self.k)
        off = entry.get("new_plate", {})
        pose = self.taught["new_plate"].moved((off.get("dx", 0), off.get("dy", 0), 0), np.radians(off.get("dyaw", 0)))
        if "friction" in entry:
            self.w.set_plate_contact(self.k, entry["friction"])
        self.w.put_plate(self.k, pose)
        self._no_overlap(self.k)
        self.note("event", None, what="a person puts a new plate on the new-plate area", plate_body=self.k + 1,
                  **({"offset": off} if off else {}), **({"friction": entry["friction"]} if "friction" in entry else {}))
        return (yield from self._approach("new_plate", self.target["new_plate"]))

    def grasp_new_plate(self):
        return (yield from self._grasp("new_plate"))

    def place_on_heater(self):
        self.place_area = "heater"
        return (yield from self._place("heater", self.target["heater"]))

    def verify_on_heater(self):
        pose, tilt = self.per.plate(self.k)
        if not self._check(checks.placed(pose, tilt, self.taught["heater"], self.tol, "heater placement")):
            return False
        if not (yield from self._retreat("heater")):
            return False
        return (yield from self._home())

    # ---- actions: Allan's one-plate sequence ---------------------------------------------------
    def wait_for_formulation(self):
        """The OT-2's dispense (not simulated); for a plate whose scenario says so, the OT-2 leaves
        a pipette tip lying on it."""
        yield from self.sk.wait(self.cfg["timing"]["formulation_s"])
        if self.scenario.plate(self.state.plate_id).get("anomaly") == "tip":
            plate, _ = self.per.plate(self.k)
            c, s = np.cos(plate.yaw), np.sin(plate.yaw)
            dx, dy = 0.012, 0.008                      # off the plate's centre, in its frame
            b = self.w.tips[self.tips_used]
            self.tips_used += 1
            self.w.put_body(b, Pose(plate.xyz + [c * dx - s * dy, s * dx + c * dy, PLATE_HALF[2] + TIP_RADIUS],
                                    plate.yaw + np.radians(25)))
            hits = self.w.overlaps(body=b)
            if hits:
                raise RuntimeError(f"pipette tip put down overlapping {hits}")
            self.note("event", None, what="the OT-2 leaves a pipette tip lying on the plate")
        return True

    def inspect_formulation(self):
        label, score = self.det.inspect(self.state.plate_id)
        self.note("detector", "InspectFormulation", label=label, score=score)
        if label != "ok":
            self.state.fault = label
            return False
        return (yield from self.sk.wait(0))

    def move_to_heater(self):
        return (yield from self._approach("heater", self.target["heater"]))

    def grasp_on_heater(self):
        return (yield from self._grasp("heater"))

    def transfer_to_reader(self):
        """The reader opens its drawer (the arm, at the heater, is clear of it) while the plate is
        carried above the tray."""
        if not self.reader.is_open():
            if not self._clear_of_reader():
                return False
            self.reader.open()
            self.note("event", None, what="the reader opens its drawer")
        xyz, point = self._tcp_for_plate(self.target["tray"].moved((0, 0, self._above("tray"))))
        if not (yield from self._move("tray", xyz, point)):
            return False
        return (yield from self.reader.wait_settled())

    def align_with_reader(self):
        xyz, point = self._tcp_for_plate(self.target["tray"].moved((0, 0, HOVER_M)))
        return (yield from self._move("tray", xyz, point))

    def verify_reader_aligned(self):
        pose, _ = self.per.plate(self.k)
        return self._check(checks.aligned(pose, self.per.tray_pocket(), self.tol)) and (yield from self.sk.wait(0))

    def load_plate_into_reader(self):
        self.place_area = "tray"
        return (yield from self._lower_and_release(self.target["tray"], self._release_above("tray")))

    def verify_reader_loaded(self):
        pose, tilt = self.per.plate(self.k)
        if not self._check(checks.placed(pose, tilt, self.per.tray_pocket(), self.tol, "reader loaded")):
            return False
        return (yield from self._retreat("tray"))

    def start_reading(self):
        """The arm moves clear of the reader; then the reader draws the plate in and measures."""
        if not (yield from self._home()):
            return False
        if not self._clear_of_reader():
            return False
        self.on_tray, _ = self.per.plate(self.k)
        self.reader.close()
        self.note("event", None, what="the reader closes its drawer")
        yield from self.reader.wait_settled()
        self.note("event", None, what="the reader measures (not simulated)")
        return True

    def wait_for_reading(self):
        """The measurement's time, then the reader opens its drawer; the arm is still clear."""
        yield from self.sk.wait(self.cfg["timing"]["reading_s"])
        if not self._clear_of_reader():
            return False
        self.reader.open()
        self.note("event", None, what="the reader opens its drawer")
        return (yield from self.reader.wait_settled())

    def retrieve_plate(self):
        pose, _ = self.per.plate(self.k)
        self.target["tray"] = pose
        if self.on_tray is not None:          # where the drawer brought the plate back, against where it took it
            dxy, dyaw = checks.pose_error(pose, self.on_tray)
            self.note("reader", "RetrievePlate", moved_on_tray_mm=round(dxy * 1000, 3), turned_deg=round(dyaw, 3),
                      drawer_open_mm=round(self.reader.position() * 1000, 2))
        if not (yield from self._approach("tray", pose)):
            return False
        if not (yield from self._grasp("tray")):
            return False
        return (yield from self.test_lift())

    def return_to_storage(self):
        """Carry the plate above the storage pile; once the arm has left the reader, the reader
        closes its drawer."""
        xyz, point = self._tcp_for_plate(self._pile_seat("storage").moved((0, 0, self._above("storage"))))
        if not (yield from self._move("storage", xyz, point)):
            return False
        if not self._clear_of_reader():
            return False
        self.reader.close()
        self.note("event", None, what="the reader closes its drawer")
        return True

    def place_in_storage(self):
        self.place_area = "storage"
        return (yield from self._place("storage", self._pile_seat("storage")))

    def verify_returned(self):
        return (yield from self._verify_pile("storage"))

    # ---- actions: Allan's recoveries -----------------------------------------------------------
    def check_quarantine_safe(self):
        pose, tilt = self.per.plate(self.k)
        ok = self._check(checks.placed(pose, tilt, self.taught["heater"], self.tol, "plate on heater, gripper empty"))
        return ok and not self.w.held() and (yield from self.sk.wait(0))

    def move_rejected_plate_to_bin(self):
        """A plate the detector flagged: from the heater onto the bin pile."""
        self.place_area = "bin"
        for step in (lambda: self._approach("heater", self.target["heater"]), lambda: self._grasp("heater"),
                     self.test_lift, lambda: self._place("bin", self._pile_seat("bin")),
                     lambda: self._verify_pile("bin")):
            if not (yield from step()):
                return False
        return True

    def reposition_for_grasp(self):
        """Open, rise above the plate, take its measured pose as the grasp pose."""
        area = self.grasp_area
        self.sk.carry = None
        yield from self.sk.fingers(self.w.opening_for(self.cfg["areas"][area]["grasp"], self.cfg["grasp"]["clearance"]))
        seen, _ = self.per.plate(self.k)
        self.target[area] = seen
        self.note("measured", "RepositionForGrasp", area=area, xyz=np.round(seen.xyz, 4).tolist(),
                  yaw_deg=round(float(np.degrees(seen.yaw)), 2))
        if not (yield from self._approach(area, seen)):
            return False
        if area == "new_plate":
            self.state.completed.difference_update(NEW_PLATE_GRASP[1:])
        return True

    def retract_and_realign(self):
        """Lift clear of the tray, measure the pocket, aim the next alignment at it."""
        if not (yield from self.sk.line(self.w.tcp_pose().xyz + [0, 0, 0.02])):
            return False
        pocket = self.per.tray_pocket()
        self.target["tray"] = pocket
        self.note("measured", "RetractAndRealign", xyz=np.round(pocket.xyz, 4).tolist(),
                  yaw_deg=round(float(np.degrees(pocket.yaw)), 2))
        return True

    def reposition_for_placement(self):
        """Grasp the plate where it was let go, as the area grasps a plate (a plate brought from
        the tray was held by its top: the pads go down to its middle), lift it, and aim the next
        placement in line with what it should rest on (the plate below on a pile, or the heater's
        seat)."""
        area = self.place_area
        seen, _ = self.per.plate(self.k)
        point = self._point(area, seen.yaw)
        if not (yield from self.sk.line(self.sk.tcp_for(seen, point, *self._grasp_offsets(area)), point)):
            return False
        yield from self.sk.close()
        self.sk.carry = (self.k, self.w.plate_in_tcp(self.k))
        if not (yield from self.test_lift()):
            return False
        if area in self.piles:
            below, _ = self.per.pile_top(area, exclude=self.k)
            self.target[area] = Pose(below.xyz, below.yaw)
            self.note("measured", "RepositionForPlacement", area=area, yaw_deg=round(float(np.degrees(below.yaw)), 2))
        if area == "heater":
            self.state.completed.difference_update(HEATER_PLACE)
        return True

    def stabilise_grasp(self):
        """The arm has braked to a stop; raise the grip force to Allan's limit, let the contact
        settle, and run the rest of this plate's moves slower."""
        g = self.cfg["grasp"]
        yield from self.sk.wait(0.2)
        self.w.set_grip_force(g["force_max_n"])
        self.note("event", "StabiliseGrasp", what=f"grip force {g['force_n']} -> {g['force_max_n']} N")
        yield from self.sk.wait(0.5)
        self.sk.speed_scale = self.cfg["speed"]["stabilised_factor"]
        self.stabilised = True
        return True
