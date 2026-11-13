"""Check the plate workflow's layout before a run: that each area's grasp
is reachable, with a level flange and nothing in the way, at the grasp and at its approach height,
for every plate a pile may hold; and that the reader's drawer carries a plate in and out.

    MUJOCO_GL=egl python scripts/bt/check_layout.py [--levels 6] [--try bin 0.45 0.0 0]

`--try AREA X Y YAW` checks AREA at another place and turn (pointing turns with it) instead of
the config's. Prints one line per area and level: ok, or what fails."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from plate_bt.backend import MujocoBackend  # noqa: E402
from plate_bt.events import Scenario  # noqa: E402
from plate_bt.runner import audit  # noqa: E402
from plate_bt.world import PLATE_HALF, PlanError, Pose, SimWorld  # noqa: E402

PLATE_H = 2 * PLATE_HALF[2]


def reach(b, area, levels):
    """For each level of the area's pile (one level if it is not a pile): IK and collisions with
    the pads closed on a plate at the seat, and with the plate lifted to the approach height."""
    w, a = b.w, b.cfg["areas"][area]
    out = []
    for level in range(levels if a.get("pile") else 1):
        for k in range(len(w.plates)):            # plates below this level, on the pile
            w.put_plate(k, b.taught[area].moved((0, 0, PLATE_H * k)) if k < level else Pose((3 + 0.2 * k, 3, 0), 0))
        seat = b.taught[area].moved((0, 0, PLATE_H * level + 0.0005))
        k = level
        point = b._point(area, seat.yaw)
        tcp = b.sk.tcp_for(seat, point, *b._grasp_offsets(area))
        result = "ok"
        for name, dz in (("grasp", 0.0), ("approach", b._above(area))):
            try:
                q = w.ik(tcp + [0, 0, dz], point, w.joints())
            except PlanError as e:
                result = f"{name}: {e}"
                break
            T = np.eye(4)
            w.kin.set_angles(w.model, w.scratch, w.info, q)
            T[:3, :3] = w.scratch.site_xmat[w.tcp].reshape(3, 3)
            T[:3, 3] = w.scratch.site_xpos[w.tcp]
            P = np.eye(4)
            P[:3, :3] = [[np.cos(seat.yaw), -np.sin(seat.yaw), 0], [np.sin(seat.yaw), np.cos(seat.yaw), 0], [0, 0, 1]]
            P[:3, 3] = seat.xyz + [0, 0, dz]
            hits = w.collisions(q, w.opening_for(a["grasp"], 0.0), carry=(k, np.linalg.inv(T) @ P))
            if hits:
                result = f"{name}: {', '.join(hits[:3])}"
                break
        out.append((area, level + 1, result))
    return out


def drawer(b):
    """A plate on the open tray, the drawer closed and opened again: how far the drawer and the
    plate went in, how far the plate moved on the tray, and the contacts deeper than the guard."""
    w, r = b.w, b.reader
    r.put(r.open_at)
    pocket = b.per.tray_pocket()
    w.put_plate(0, pocket.moved((0, 0, 0.0005)))
    for _ in range(round(1.0 / w.dt)):
        w.step()
    start, _ = b.per.plate(0)
    deepest, inside = [], None
    for go in (r.close, r.open):
        go()
        while r.moving():
            w.step()
            deepest += audit(w, None)
        if inside is None:
            inside = (r.position(), b.per.plate(0)[0].xyz[1] - start.xyz[1])
    end, tilt = b.per.plate(0)
    return inside, np.linalg.norm(end.xyz - start.xyz) * 1000, np.degrees(abs(end.yaw - start.yaw)), tilt, deepest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "plate_bt.yaml")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--try", dest="tries", nargs=4, action="append", default=[], metavar=("AREA", "X", "Y", "YAW"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for area, x, y, yaw in args.tries:
        a = cfg["areas"][area]
        a["point"] += float(yaw) - a["plate_yaw"]
        a.update(xy=[float(x), float(y)], plate_yaw=float(yaw))
    world = SimWorld(cfg, args.levels)
    lbt_state = type("State", (), {"tick_number": 0, "plate_id": 1})()
    b = MujocoBackend(lbt_state, world, cfg, Scenario([{}]), None)
    b.start()
    b.reader.put(b.reader.open_at)
    for area in cfg["areas"]:
        for name, level, result in reach(b, area, args.levels):
            print(f"{name:10s} plate {level}: {result}")
    (drawn, plate_in), moved, turned, tilt, contacts = drawer(b)
    print(f"drawer: closed {drawn * 1000:.1f} mm in, plate {plate_in * 1000:.1f} mm in; back open, the plate "
          f"moved {moved:.2f} mm, turned {turned:.2f} deg, tilt {tilt:.2f} deg; "
          f"contacts deeper than 0.3 mm: {sorted(set(p for p, _ in contacts)) or 'none'}")


if __name__ == "__main__":
    main()
