"""Render a whole plate-workflow run as one video from four cameras.

    MUJOCO_GL=egl python scripts/bt/render_grid.py data/plate_bt_<ts>_continuous [--speed 4] [--lab]
    MUJOCO_GL=egl python scripts/bt/render_grid.py data/plate_bt_<ts>_continuous --preview 100 160

Writes <run>/render/grid.mp4: 1920 x 1080, a 2 x 2 grid of 960 x 540 views (behind the robot,
Allan's wrist camera, inside the OT-2, over the reader's tray), `speed` times real time. --lab
swaps the last three for the lab's cameras (configs/lab_cameras.yaml): the
wrist camera, the OT-2's built-in camera and the camera above the reader, each at its own aspect
ratio, and writes grid_lab.mp4. The first
view carries what the tree is doing and how many plates each pile holds; checks that fail, detector
flags, measurements and scenario events run along the bottom as they happen. `--preview T ...`
writes single frames at those simulation times instead (render/grid_<T>.jpg)."""
import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_run import BRANCH_COLOUR, CLOSE, EVENT_S, MAIN, Replay, branch, camera, describe, font  # noqa: E402

from mujoco_bridge import cameras as C  # noqa: E402
from plate_bt.scene import TABLE_TOP  # noqa: E402
from plate_bt.world import PLATE_HALF  # noqa: E402

W, H = 1920, 1080
TW, TH = W // 2, H // 2
INSIDE = dict(lookat=(-0.19, 0.58, 0.18), distance=0.42, azimuth=-90, elevation=-30)   # from the OT-2's back, facing out
VIEWS = (("behind the robot", MAIN), ("wrist camera", "wrist_gripper"),
         ("inside the OT-2", INSIDE), ("over the reader's tray", CLOSE["tray"]))
LAB_VIEWS = (("behind the robot", MAIN), ("wrist camera", "lab_wrist"), ("OT-2 camera", "lab_ot2"),
             ("reader camera", "lab_reader"))
PILES = (("supply", "new_plate"), ("storage", "storage"), ("bin", "bin"))


class Grid:
    def __init__(self, run, lab=False):
        self.cams = C.load() if lab else None
        self.views = LAB_VIEWS if lab else VIEWS
        self.rp = Replay(run, self.cams)
        w = self.rp.world
        self.r = mujoco.Renderer(w.model, TH, TW)
        self.lab_r = C.Renderers(w.model)
        self.f_big, self.f, self.f_tag = font(21), font(16), font(15)
        cfg = self.rp.cfg
        self.axes = {name: np.array(cfg["areas"][area]["xy"]) for name, area in PILES}
        self.pad_top = TABLE_TOP + cfg["pad_size"][2]

    def view(self, spec):
        if self.cams and isinstance(spec, str) and spec.startswith(C.PREFIX):
            # a lab camera at its own aspect ratio, centred in the tile
            cam = self.cams[spec[len(C.PREFIX):]]
            w, h = cam["resolution"]
            size = (TW, round(TW * h / w)) if w / h >= TW / TH else (round(TH * w / h), TH)
            tile = Image.new("RGB", (TW, TH))
            tile.paste(Image.fromarray(C.image(self.lab_r, self.rp.world.data, cam, size, self.rp.opt)),
                       ((TW - size[0]) // 2, (TH - size[1]) // 2))
            return tile
        cam = spec if isinstance(spec, str) else camera(spec)
        self.r.update_scene(self.rp.world.data, camera=cam, scene_option=self.rp.opt)
        return Image.fromarray(self.r.render())

    def piles(self):
        """Plates resting on each pile: centre within 3 cm of the pile's axis, at a plate's height
        on the pile (a plate carried over it is not counted)."""
        w = self.rp.world
        out = {}
        for name, axis in self.axes.items():
            n = 0
            for b in w.plates:
                x, y, z = w.data.xpos[b]
                level = (z - self.pad_top - PLATE_HALF[2]) / (2 * PLATE_HALF[2])
                if np.hypot(x - axis[0], y - axis[1]) < 0.03 and abs(level - round(level)) < 0.15 and level > -0.5:
                    n += 1
            out[name] = n
        return out

    def status(self, img, i, speed):
        st, sc = self.rp.st, self.rp.scenario
        plate = int(st["plate"][i])
        action = str(st["names"][st["action"][i]])
        fault = str(st["faults"][st["fault"][i]])
        b = branch(action)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, 430, 186), fill=(15, 20, 30, 195))
        d.text((12, 8), f"Plate {plate}/{len(sc)}: {sc.plate(plate).get('name', '')}", font=self.f_big, fill="white")
        d.text((12, 38), f"t = {st['time'][i]:7.1f} s    video x{speed}", font=self.f, fill=(200, 200, 200))
        d.rectangle((12, 64, 26, 78), fill=BRANCH_COLOUR[b])
        d.text((34, 61), f"branch {b}", font=self.f, fill="white")
        d.text((12, 84), f"action  {action}", font=self.f, fill="white")
        d.text((12, 106), f"fault   {fault}", font=self.f, fill=(255, 110, 100) if fault != "none" else (200, 200, 200))
        risk = float(st["slip_risk"][i])
        d.text((12, 128), f"slip risk {risk:4.2f}   grip {st['grip_n'][i]:.1f} N", font=self.f, fill="white")
        x0, x1 = 250, 418
        d.rectangle((x0, 132, x1, 144), outline=(200, 200, 200))
        d.rectangle((x0, 132, x0 + (x1 - x0) * min(risk, 1.0), 144), fill=(220, 60, 50) if risk >= 0.7 else (90, 170, 230))
        d.line((x0 + (x1 - x0) * 0.7, 128, x0 + (x1 - x0) * 0.7, 148), fill=(255, 220, 0), width=2)
        p = self.piles()
        d.text((12, 152), f"plates: supply {p['supply']}   storage pile {p['storage']}   bin pile {p['bin']}",
               font=self.f, fill=(200, 200, 200))

    def frame(self, i, speed, events):
        self.rp.pose(i)
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img, "RGBA")
        for n, (label, spec) in enumerate(self.views):
            x, y = (n % 2) * TW, (n // 2) * TH
            img.paste(self.view(spec), (x, y))
            tw = d.textlength(label, font=self.f_tag)
            d.rectangle((x + TW - tw - 22, y + 8, x + TW - 8, y + 32), fill=(15, 20, 30, 190))
            d.text((x + TW - tw - 15, y + 11), label, font=self.f_tag, fill="white")
        d.line((TW, 0, TW, H), fill=(240, 240, 240), width=2)
        d.line((0, TH, W, TH), fill=(240, 240, 240), width=2)
        self.status(img, i, speed)
        now = self.rp.st["time"][i]
        shown = [(t, s) for t, s in events if t <= now < t + EVENT_S][-3:]
        for n, (t, s) in enumerate(shown):
            y = H - 40 * (len(shown) - n)
            d.rectangle((0, y, W, y + 36), fill=(15, 20, 30, 205))
            d.text((14, y + 8), s, font=self.f, fill=(255, 210, 120))
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--speed", type=int, default=4, help="ticks per video frame; the video runs this much faster")
    ap.add_argument("--preview", type=float, nargs="*", help="only write frames at these simulation times")
    ap.add_argument("--lab", action="store_true", help="the lab's three cameras instead of the earlier views")
    args = ap.parse_args()
    g = Grid(args.run, args.lab)
    stem = "grid_lab" if args.lab else "grid"
    st, tr = g.rp.st, g.rp.trace
    events = [(e["t"], s) for e in tr["events"] if (s := describe(e))]
    events += [(o["t"] - 0.1, f"PLATE {o['plate']} {o['outcome'].upper()}") for o in tr["summary"]["outcomes"]]
    events.sort()
    out = args.run / "render"
    out.mkdir(exist_ok=True)
    if args.preview:
        for t in args.preview:
            i = min(int(np.searchsorted(st["time"], t)), len(st["time"]) - 1)
            g.frame(i, args.speed, events).save(out / f"{stem}_{t:g}.jpg", quality=88)
        print(out)
        return
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                           "-preset", "medium", "-movflags", "+faststart", str(out / f"{stem}.mp4")], stdin=subprocess.PIPE)
    ticks = np.arange(0, len(st["time"]), args.speed)
    for n, i in enumerate(ticks):
        ff.stdin.write(np.asarray(g.frame(i, args.speed, events)).tobytes())
        if n % 1000 == 0:
            print(f"frame {n}/{len(ticks)}", flush=True)
    ff.stdin.close()
    ff.wait()
    g.r.close()
    g.lab_r.close()
    g.rp.r.close()
    print(out / f"{stem}.mp4")


if __name__ == "__main__":
    main()
