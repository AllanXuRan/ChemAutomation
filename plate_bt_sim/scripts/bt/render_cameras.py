"""Stills of a plate-workflow run from the lab's three cameras, in MuJoCo.

    MUJOCO_GL=egl python scripts/bt/render_cameras.py data/plate_bt_<ts>_continuous [--out DIR]

Replays the run (scripts/bt/render_run.py's Replay) with the cameras of configs/lab_cameras.yaml
added to the model, and writes, at each camera's own resolution:

    <moment>_<camera>.png   each camera at each moment of MOMENTS
    overview_<moment>.png   a free view of the cell with each camera's view drawn as a pyramid
    records.json            the tick and time of each moment, and the cameras' world poses

into <run>/render/lab_cameras/ (or --out)."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_run import Replay, camera, font  # noqa: E402

from mujoco_bridge import cameras as C  # noqa: E402

# (name, plate, action, seconds after the action starts)
MOMENTS = (
    ("heater_wait", 1, "WaitForFormulationComplete", 1.0),     # plate 1 on the heater, the arm out of the OT-2
    ("heater_grasp", 1, "GraspPlate", 1.5),                    # the pads closing on it
    ("tray_align", 1, "AlignWithReader", 0.5),                 # carried over the reader's open tray
    ("tray_loaded", 1, "VerifyReaderLoaded", 0.5),             # let go on the tray
    ("reading", 1, "WaitForReadingComplete", 1.0),             # the drawer closed, the arm clear
    ("tip_on_plate", 5, "InspectFormulation", 1.0),            # plate 5 with the fallen pipette tip on it
)
OVERVIEW = dict(lookat=(-0.15, 0.45, 0.25), distance=1.9, azimuth=125, elevation=-28)
# lit lines come out brighter, so each colour's channels stay near 0 or 1 to keep its hue
COLOURS = {"wrist": (1.0, 0.9, 0.0, 1.0), "reader": (0.1, 0.75, 1.0, 1.0), "ot2": (0.9, 0.2, 0.75, 1.0)}
LABELS = {"wrist": "wrist camera", "reader": "reader camera", "ot2": "OT-2 camera"}
OVERVIEW_SIZE = (1600, 1000)


def tick_of(st, plate, action, after_s):
    names = [str(n) for n in st["names"]]
    if action not in names:
        return None
    hit = np.flatnonzero((st["plate"] == plate) & (st["action"] == names.index(action)))
    if not len(hit):
        return None
    t = st["time"][hit[0]] + after_s
    return min(int(np.searchsorted(st["time"], t)), len(st["time"]) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cams = C.load()
    rp = Replay(args.run, cams)
    m, d, st = rp.world.model, rp.world.data, rp.st
    C.make_room(m, cams, OVERVIEW_SIZE)
    rs = C.Renderers(m)
    out = args.out or args.run / "render" / "lab_cameras"
    out.mkdir(parents=True, exist_ok=True)
    f = font(22)
    record = dict(run=str(args.run), config=json.loads(json.dumps(cams)), moments=[], images=[])
    for name, plate, action, after in MOMENTS:
        i = tick_of(st, plate, action, after)
        if i is None:
            print(f"skip {name}: plate {plate} never runs {action}")
            continue
        rp.pose(i)
        record["moments"].append(dict(name=name, plate=plate, action=action, tick=i, t=round(float(st["time"][i]), 2),
                                      cameras={n: dict(pos=d.cam_xpos[C.camera_id(m, n)].round(4).tolist(),
                                                       mat=d.cam_xmat[C.camera_id(m, n)].round(5).tolist())
                                               for n in cams}))
        for cname, cam in cams.items():
            file = f"{name}_{cname}.png"
            Image.fromarray(C.image(rs, d, cam, scene_option=rp.opt)).save(out / file)
            record["images"].append(dict(file=file, moment=name, camera=cname))
        r = rs.get(*OVERVIEW_SIZE)
        r.update_scene(d, camera=camera(OVERVIEW), scene_option=rp.opt)
        C.draw_frustums(r.scene, m, d, cams, COLOURS)
        C.draw_axes(r.scene)
        img = Image.fromarray(r.render())
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle((0, 0, 470, 74 + 34 * len(cams)), fill=(15, 20, 30, 190))
        draw.text((14, 8), f"t = {st['time'][i]:.1f} s  plate {plate}  {action}", font=f, fill="white")
        for n, cname in enumerate(cams):
            y = 44 + 34 * n
            draw.rectangle((14, y, 40, y + 22), fill=tuple(int(255 * v) for v in COLOURS[cname][:3]))
            draw.text((50, y - 1), LABELS[cname], font=f, fill="white")
        draw.text((14, 44 + 34 * len(cams)), "axes at the robot base: x red, y green, z blue", font=f, fill="white")
        img.save(out / f"overview_{name}.png")
        print(f"{name}: tick {i}, t {st['time'][i]:.1f} s", flush=True)
    (out / "records.json").write_text(json.dumps(record, indent=1))
    rs.close()
    rp.r.close()
    print(out)


if __name__ == "__main__":
    main()
