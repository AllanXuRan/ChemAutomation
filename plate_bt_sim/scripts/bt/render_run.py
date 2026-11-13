"""Render a plate-workflow run with MuJoCo's own renderer.

    MUJOCO_GL=egl python scripts/bt/render_run.py data/plate_bt_<ts>_run [--speed 4]

Replays states.npz in the run's own model (config.yaml, scenario.yaml), and writes into
<run>/render/:

    plate_<k>.mp4   one clip per plate, `speed` times real time: a free camera over the cell, an
                    inset top view, and what the tree is doing (branch, action, fault, slip
                    risk, grip force), with each check that fails, each detector flag and each
                    measurement shown as it happens
    stills/         a close view of each of those moments, of each plate's outcome, and of the
                    first plate in the closed reader with the arm clear
    stills.json     what each still shows
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from plate_bt.events import Scenario  # noqa: E402
from plate_bt.world import SimWorld  # noqa: E402

W, H = 1280, 720
MAIN = dict(lookat=(0.0, 0.27, 0.08), distance=1.7, azimuth=100, elevation=-40)
TOP = dict(lookat=(0.0, 0.25, 0.0), distance=1.6, azimuth=90, elevation=-70)
CLOSE = {   # a close camera per area, for the stills
    "heater": dict(lookat=(-0.19, 0.6, 0.17), distance=0.65, azimuth=60, elevation=-35),
    "tray": dict(lookat=(0.07, 0.45, 0.11), distance=0.55, azimuth=180, elevation=-62),     # from the east, over the arm
    "reader": dict(lookat=(0.03, 0.36, 0.18), distance=1.15, azimuth=135, elevation=-28),   # the arm clear of the reader
    "storage": dict(lookat=(0.30, 0.25, 0.05), distance=0.55, azimuth=150, elevation=-40),
    "bin": dict(lookat=(0.45, 0.0, 0.02), distance=0.55, azimuth=150, elevation=-45),
    "new_plate": dict(lookat=(0.32, -0.18, 0.02), distance=0.55, azimuth=150, elevation=-45),
}
RECOVERY = {"CheckQuarantineSafe", "MoveRejectedPlateToBin", "RepositionForGrasp", "RetractAndRealign",
            "RepositionForPlacement"}
BRANCH_COLOUR = {"01 safety": (220, 50, 47), "02 recovery": (230, 140, 30), "03 preventive": (200, 170, 0),
                 "04 normal": (60, 160, 90)}
EVENT_S = 5.0                # an event stays on screen this long (simulation time)


def branch(action):
    if action == "StabiliseGrasp":
        return "03 preventive"
    if action in RECOVERY:
        return "02 recovery"
    if action == "LatchedEmergencyStop":
        return "01 safety"
    return "04 normal"


def area_of(action, plate_entry):
    """Which area a moment is about, for its close still; None for one between areas."""
    a = action or ""
    if a == "StabiliseGrasp":
        return None
    if "NewPlate" in a or a == "RepositionForGrasp" and "new_plate" in plate_entry:
        return "new_plate"
    if a == "StartReaderMeasurement":
        return "reader"
    if "Reader" in a or a in ("RetractAndRealign", "AlignWithReader", "RetrievePlate"):
        return "tray"
    if a in ("PlacePlateInSlot", "VerifyReturned", "ReturnToStorageSlot") or \
            a == "RepositionForPlacement" and "storage_plate_dyaw" in plate_entry:
        return "storage"
    if a == "MoveRejectedPlateToBin":
        return "bin"
    return "heater"


def describe(e):
    """One line for an event of the trace, or None if it is not worth showing."""
    k = e["kind"]
    if k == "check" and not e["ok"]:
        vals = ", ".join(f"{a} {e[a]}" for a in ("dxy_mm", "dyaw_deg", "tilt_deg", "lag_mm") if a in e)
        return f"CHECK FAILED  {e['check']}: {vals}"
    if k == "detector" and e["label"] != "ok":
        return f"DETECTOR  {e['label']} (score {e['score']})"
    if k == "measured":
        return f"MEASURED  {e['action']}: yaw {e.get('yaw_deg')} deg"
    if k == "event":
        return f"EVENT  {e['what']}"
    if k in ("drop", "emergency stop", "plan error"):
        return f"{k.upper()}  {e.get('reason') or e.get('error') or ''}"
    return None


def camera(spec):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = spec["lookat"]
    c.distance, c.azimuth, c.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
    return c


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        if Path(f).is_file():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


class Replay:
    def __init__(self, run, cameras=None):
        """`cameras`: the lab's cameras (mujoco_bridge.cameras.load()) to add to the model."""
        self.run = run
        self.cfg = yaml.safe_load((run / "config.yaml").read_text())
        self.scenario = Scenario.load(run / "scenario.yaml")
        self.world = SimWorld(self.cfg, self.scenario.bodies, self.scenario.tips, cameras)
        m = self.world.model
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        if cameras:
            from mujoco_bridge import cameras as C
            C.make_room(m, cameras, (W, H))
        self.r = mujoco.Renderer(m, H, W)
        self.st = np.load(run / "states.npz")
        self.trace = json.loads((run / "trace.json").read_text())
        self.f_big, self.f_small = font(24), font(19)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0            # Allan's target and TCP markers

    def pose(self, i):
        """The workcell at tick i (the reader's drawer is in qpos)."""
        w, st = self.world, self.st
        w.data.qpos[:] = st["qpos"][i]
        mujoco.mj_forward(w.model, w.data)

    def image(self, spec, size=None):
        self.r.update_scene(self.world.data, camera=camera(spec), scene_option=self.opt)
        img = Image.fromarray(self.r.render())
        return img.resize(size, Image.LANCZOS) if size else img

    def overlay(self, img, i, speed, events):
        st = self.st
        plate = int(st["plate"][i])
        action = str(st["names"][st["action"][i]])
        fault = str(st["faults"][st["fault"][i]])
        b = branch(action)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, 520, 210), fill=(15, 20, 30, 190))
        name = self.scenario.plate(plate).get("name", "")
        d.text((14, 10), f"Plate {plate}/{len(self.scenario)}: {name}", font=self.f_big, fill="white")
        d.text((14, 44), f"t = {st['time'][i]:7.1f} s    video x{speed}", font=self.f_small, fill=(200, 200, 200))
        d.rectangle((14, 76, 30, 92), fill=BRANCH_COLOUR[b])
        d.text((38, 72), f"branch {b}", font=self.f_small, fill="white")
        d.text((14, 100), f"action  {action}", font=self.f_small, fill="white")
        d.text((14, 126), f"fault   {fault}", font=self.f_small,
               fill=(255, 110, 100) if fault != "none" else (200, 200, 200))
        risk = float(st["slip_risk"][i])
        d.text((14, 152), f"slip risk {risk:4.2f}", font=self.f_small, fill="white")
        x0, x1 = 170, 500
        d.rectangle((x0, 158, x1, 172), outline=(200, 200, 200))
        d.rectangle((x0, 158, x0 + (x1 - x0) * min(risk, 1.0), 172),
                    fill=(220, 60, 50) if risk >= 0.7 else (90, 170, 230))
        tx = x0 + (x1 - x0) * 0.7
        d.line((tx, 154, tx, 176), fill=(255, 220, 0), width=2)
        d.text((14, 180), f"grip force {st['grip_n'][i]:.1f} N", font=self.f_small, fill=(200, 200, 200))
        now = st["time"][i]
        shown = [(t, s) for t, s in events if t <= now < t + EVENT_S][-3:]
        for n, (t, s) in enumerate(shown):
            y = H - 44 * (len(shown) - n)
            d.rectangle((0, y, W, y + 40), fill=(15, 20, 30, 200))
            d.text((14, y + 8), s, font=self.f_small, fill=(255, 210, 120))
        return img

    def frame(self, i, speed, events):
        self.pose(i)
        img = self.image(MAIN)
        inset = self.image(TOP, (384, 216))
        img.paste(inset, (W - 384 - 10, 10))
        ImageDraw.Draw(img).rectangle((W - 394, 10, W - 10, 226), outline="white", width=2)
        return self.overlay(img, i, speed, events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--speed", type=int, default=4, help="ticks per video frame; the video runs this much faster")
    ap.add_argument("--plates", type=int, nargs="*", help="only these plates")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()
    rp = Replay(args.run)
    out = args.run / "render"
    (out / "stills").mkdir(parents=True, exist_ok=True)
    st, tr = rp.st, rp.trace
    events = [(e["t"], s) for e in tr["events"] if (s := describe(e))]
    outcomes = {o["plate"]: o for o in tr["summary"]["outcomes"]}
    events += [(o["t"] - 0.1, f"PLATE {p} {o['outcome'].upper()}") for p, o in outcomes.items()]
    events.sort()
    plates = [] if args.stills_only else args.plates or sorted({int(p) for p in st["plate"]})
    for p in plates:
        ticks = np.flatnonzero(st["plate"] == p)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-crf", "26", "-preset", "medium", "-movflags", "+faststart", str(out / f"plate_{p}.mp4")],
                              stdin=subprocess.PIPE)
        for i in ticks[::args.speed]:
            ff.stdin.write(np.asarray(rp.frame(i, args.speed, events)).tobytes())
        ff.stdin.close()
        ff.wait()
        print(f"plate {p}: {len(ticks[::args.speed])} frames", flush=True)
    # stills: each failed check, flag, measurement and outcome, from the area's close camera
    stills = []
    moments = [e for e in tr["events"] if describe(e) and (e["kind"] != "event" or e["action"] == "StabiliseGrasp")]
    moments += [dict(t=o["t"] - 0.2, plate=p, kind="outcome", action="VerifyReturned"
                     if o["outcome"] == "measured" else "MoveRejectedPlateToBin") for p, o in outcomes.items()]
    moments += [dict(e, kind="reading", t=e["t"] - 0.6, action="StartReaderMeasurement") for e in tr["events"]
                if e["kind"] == "event" and "measures" in e["what"]][:1]
    for n, e in enumerate(sorted(moments, key=lambda e: e["t"])):
        i = min(int(np.searchsorted(st["time"], e["t"] + 0.5)), len(st["time"]) - 1)
        while st["plate"][i] != e["plate"]:       # an outcome is the plate's last moment
            i -= 1
        area = area_of(e.get("action"), rp.scenario.plate(int(e["plate"])))
        rp.pose(i)
        img = rp.overlay(rp.image(CLOSE[area] if area else MAIN), i, 1, events)
        name = f"{n:02d}_plate{e['plate']}_{e['kind'].replace(' ', '-')}.jpg"
        img.save(out / "stills" / name, quality=88)
        stills.append(dict(file=name, t=e["t"], plate=e["plate"], kind=e["kind"], area=area,
                           text=describe(e) if e["kind"] not in ("outcome", "reading") else f"plate {e['plate']} {e['kind']}"))
    (out / "stills.json").write_text(json.dumps(stills, indent=1))
    rp.r.close()
    print(out)


if __name__ == "__main__":
    main()
