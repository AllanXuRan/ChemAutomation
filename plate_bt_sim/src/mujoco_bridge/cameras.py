"""The lab's three cameras (configs/lab_cameras.yaml), for MuJoCo and Blender.

Each camera is fixed to a body of Allan's model (or the world) and given in that body's frame by
its position, the point it looks at and an up direction. Both renderers use the same convention:
a camera looks down its local -z with +y up, so one rotation serves both.

MuJoCo: `edit(cams)` returns an edit for `mujoco_bridge.load` (and `plate_bt.scene.build`) that
adds the cameras, and a non-colliding housing behind each, to the scene's XML before it compiles.
Blender: `local_pose(cam)` gives the 4 x 4 pose in the camera's body frame; the caller multiplies
it by the body's world pose and the transform into our deck frame. This module needs only numpy
(and PyYAML to read the file), so it imports inside Blender too."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "lab_cameras.yaml"
PREFIX = "lab_"                      # MuJoCo camera names: lab_wrist, lab_reader, lab_ot2


def load(path=CONFIG):
    """{name: camera record} from the config, each with its housing settings under "housing": the
    file's `housing`, with the camera's own `housing` entries (e.g. its size) over it."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    base = cfg.get("housing")
    return {n: dict(c, name=n, housing=dict(base or {}, **c["housing"]) if "housing" in c else base)
            for n, c in cfg["cameras"].items()}


def rotation(pos, look_at, up):
    """Columns: the camera's x (image right), y (image up) and z (backwards) in its body frame."""
    back = np.asarray(pos, float) - np.asarray(look_at, float)
    z = back / np.linalg.norm(back)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError(f"camera up {up} is along its line of sight")
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def local_pose(cam):
    """4 x 4 pose of the camera in its body's frame."""
    M = np.eye(4)
    M[:3, :3] = rotation(cam["pos"], cam["look_at"], cam["up"])
    M[:3, 3] = cam["pos"]
    return M


def _fmt(v):
    return " ".join(f"{float(x):.8g}" for x in v)


def _quat(R):
    """MuJoCo quaternion (w, x, y, z) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def add(root, cams):
    """Add each camera, and its housing, to the MJCF tree `root`, inside its body."""
    for c in cams.values():
        parent = root.find("worldbody") if c["body"] == "world" else \
            next((b for b in root.iter("body") if b.get("name") == c["body"]), None)
        if parent is None:
            raise KeyError(f"camera {c['name']}: no body {c['body']!r} in the scene")
        R = rotation(c["pos"], c["look_at"], c["up"])
        ET.SubElement(parent, "camera", name=PREFIX + c["name"], mode="fixed", pos=_fmt(c["pos"]),
                      xyaxes=_fmt([*R[:, 0], *R[:, 1]]), fovy=f"{c['fovy']:g}")
        h = c.get("housing")
        if h:
            half = np.asarray(h["size"], float) / 2
            centre = np.asarray(c["pos"], float) + R[:, 2] * (half[2] + 0.002)   # behind the lens
            ET.SubElement(parent, "geom", name=f"{PREFIX}{c['name']}_housing", type="box", pos=_fmt(centre),
                          quat=_fmt(_quat(R)), size=_fmt(half), rgba=_fmt(h["rgba"]), contype="0",
                          conaffinity="0", density="0", group="1")


def edit(cams, then=None):
    """An edit for mujoco_bridge.load: `then(root)` first (another edit, or None), then the cameras."""
    def apply(root):
        if then is not None:
            then(root)
        add(root, cams)
    return apply


def camera_id(model, name):
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, PREFIX + name)


# ---------------------------------------------------------------- MuJoCo rendering

class Renderers:
    """One mujoco.Renderer per image size, created on first use (a Renderer has a fixed size)."""

    def __init__(self, model):
        self.model, self.r = model, {}

    def get(self, w, h):
        import mujoco
        g = self.model.vis.global_
        if w > g.offwidth or h > g.offheight:
            raise ValueError(f"{w} x {h} is larger than the model's offscreen buffer; raise it before the first render")
        if (w, h) not in self.r:
            self.r[(w, h)] = mujoco.Renderer(self.model, h, w)
        return self.r[(w, h)]

    def close(self):
        for r in self.r.values():
            r.close()


def make_room(model, cams, at_least=(1280, 720)):
    """Raise the model's offscreen buffer to fit every camera's resolution."""
    g = model.vis.global_
    g.offwidth = max(g.offwidth, at_least[0], *(c["resolution"][0] for c in cams.values()))
    g.offheight = max(g.offheight, at_least[1], *(c["resolution"][1] for c in cams.values()))


def image(renderers, data, cam, size=None, scene_option=None):
    """The view of lab camera `cam` (a record from load()) as an (h, w, 3) array, at its own
    resolution or at `size` (w, h)."""
    w, h = size or cam["resolution"]
    r = renderers.get(w, h)
    r.update_scene(data, camera=PREFIX + cam["name"], scene_option=scene_option)
    return r.render()


def _line(scn, a, b, rgba, width):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def draw_frustums(scn, model, data, cams, colours, depth=0.25, width=3.0):
    """Each lab camera's view as a pyramid of lines `depth` metres deep, added to the mjvScene of
    a renderer after update_scene (so another camera shows where these look)."""
    for name, cam in cams.items():
        c = camera_id(model, name)
        p, R = data.cam_xpos[c], data.cam_xmat[c].reshape(3, 3)
        ty = np.tan(np.radians(cam["fovy"]) / 2)
        tx = ty * cam["resolution"][0] / cam["resolution"][1]
        corners = [p + depth * (-R[:, 2] + sx * tx * R[:, 0] + sy * ty * R[:, 1])
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for i, q in enumerate(corners):
            _line(scn, p, q, colours[name], width)
            _line(scn, q, corners[(i + 1) % 4], colours[name], width)
        _line(scn, corners[2], corners[3], colours[name], width * 2)       # the image's top edge


def draw_axes(scn, origin=(0, 0, 0), length=0.2, width=4.0):
    """Allan's world axes at `origin`: x red, y green, z blue."""
    o = np.asarray(origin, float)
    for axis, rgba in enumerate(((1, 0.1, 0.1, 1), (0.1, 0.8, 0.1, 1), (0.15, 0.3, 1, 1))):
        _line(scn, o, o + length * np.eye(3)[axis], rgba, width)
