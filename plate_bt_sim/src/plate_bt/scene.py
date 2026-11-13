"""Allan's workcell with what the plate workflow adds, built in memory:
more plates (copies of his `reaction_plate`), the new-plate area and the two piles (storage and
bin) on the table, the reader's tray as a drawer, and a parking shelf far from the cell where
plates wait until a person puts them on the new-plate area. Allan's stack in OT-2 slot 2 is left
out: storage is a pile on the table (Kevin, 2026-10-22). His files are not written."""
import copy
import math
import re
import xml.etree.ElementTree as ET

from mujoco_bridge import cameras as C
from mujoco_bridge import load as L

TABLE_TOP = -0.01            # Allan's lab_layout.json, table.top_z
PARK_XY = (3.0, 3.0)         # the parking shelf, outside every camera
PARK_PITCH = 0.2
TIP_RADIUS, TIP_LENGTH = 0.0035, 0.045        # a 200 uL pipette tip, as a capsule (its taper is not modelled)
PADS = {"new_plate": "0.85 0.55 0.15 1", "storage": "0.3 0.6 0.85 1", "bin": "0.75 0.3 0.3 1"}
OT2_STACK = ("ot2_stack_pad", "ot2_stack_plate_1", "ot2_stack_plate_2", "ot2_stack_plate_3")
DRAWER_GEOMS = ("versamax_sliding_tray", "versamax_tray_insert", "versamax_tray_guide_0", "versamax_tray_guide_1")
DRAWER_JOINT = "versamax_drawer_slide"
DRAWER_ACTUATOR = "versamax_drawer"


def tip_name(i):
    return f"pipette_tip_{i}"


def park_tip_xyz(i):
    return (PARK_XY[0] + PARK_PITCH * (i + 0.5), PARK_XY[1] + 0.07, TABLE_TOP + TIP_RADIUS)


def plate_name(k):
    """Plate k (1-based); plate 1 is Allan's own reaction_plate."""
    return "reaction_plate" if k == 1 else f"reaction_plate_{k}"


def park_xyz(k):
    return (PARK_XY[0] + PARK_PITCH * k, PARK_XY[1], TABLE_TOP + 0.0075)


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _vec(s):
    return [float(x) for x in s.split()]


def _edit(cfg, n_plates, n_tips):
    def edit(root):
        world = root.find("worldbody")
        plate = world.find("body[@name='reaction_plate']")
        plate.set("pos", _fmt(park_xyz(1)))
        for k in range(2, n_plates + 1):
            twin = copy.deepcopy(plate)
            for el in twin.iter():
                if el.get("name"):
                    el.set("name", f"{el.get('name')}_{k}")
            twin.set("name", plate_name(k))
            twin.set("pos", _fmt(park_xyz(k)))
            world.append(twin)
        _remove_ot2_stack(world)
        _drawer(root, cfg["reader"])
        sx, sy, sz = (s / 2 for s in cfg["pad_size"])
        for area, rgba in PADS.items():
            a = cfg["areas"][area]
            pad = _geom(f"{area}_pad", (*a["xy"], TABLE_TOP + sz), (sx, sy, sz), rgba)
            pad.set("euler", f"0 0 {math.radians(a['plate_yaw']):.6g}")      # Allan's compiler reads radians
            world.append(pad)
        for i in range(n_tips):
            tip = ET.SubElement(world, "body", name=tip_name(i), pos=_fmt(park_tip_xyz(i)))
            ET.SubElement(tip, "freejoint", name=f"{tip_name(i)}_free")
            ET.SubElement(tip, "geom", name=f"{tip_name(i)}_geom", type="capsule", size=_fmt((TIP_RADIUS,)),
                          fromto=_fmt((-TIP_LENGTH / 2, 0, 0, TIP_LENGTH / 2, 0, 0)), density="400",
                          rgba="0.92 0.93 0.96 1", contype="1", conaffinity="1",
                          # a capsule rolls where a real tip's taper and the plate's wells would hold it:
                          # rolling friction, and priority over the plate so these contact parameters apply
                          priority="2", condim="6", friction="1.0 0.01 0.005", solref="0.004 1")
        world.append(_geom("parking_shelf", (PARK_XY[0] + PARK_PITCH * (n_plates + 1) / 2, PARK_XY[1], TABLE_TOP - 0.01),
                           (PARK_PITCH * (n_plates + 2) / 2, 0.1, 0.01), "0.5 0.5 0.5 1"))
    return edit


def _remove_ot2_stack(world):
    """Allan's stacks: the one in OT-2 slot 2 (fixed geoms) and, since his four-plate workflow, his
    source stack on the table (free bodies ot2_stack_plate_N_body, with reaction_plate at its
    bottom). The new-plate area and the two piles take their place."""
    ot2 = world.find("body[@name='ot2_enclosure']")
    for el in list(ot2):
        if el.get("name") in OT2_STACK or el.get("name") == "ot2_stack_top_visual":
            ot2.remove(el)
    for body in list(world.findall("body")):
        if re.fullmatch(r"ot2_stack_plate_\d+_body", body.get("name", "")):
            world.remove(body)


def _drawer(root, rc):
    """The reader's tray, insert and guides move into a body on a slide joint (+y draws the tray
    in), driven by a position actuator; a slot through the housing behind the recess takes the
    drawer and its plate. Positions are in the reader's frame, from Allan's geoms."""
    reader = root.find("worldbody/body[@name='versamax_reader']")
    geoms = {g.get("name"): g for g in reader.findall("geom")}
    tray = geoms["versamax_sliding_tray"]
    (cx, cy, cz), (hx, hy, hz) = _vec(tray.get("pos")), _vec(tray.get("size"))
    travel, gap = rc["travel_m"], rc["slot_clearance_m"]
    drawer = ET.SubElement(reader, "body", name="versamax_drawer", pos=_fmt((cx, cy, 0)))
    ET.SubElement(drawer, "joint", name=DRAWER_JOINT, type="slide", axis="0 1 0", limited="true",
                  range=_fmt((-0.01, travel + 0.005)), damping=str(rc["damping"]))
    for name in DRAWER_GEOMS:
        g = geoms[name]
        reader.remove(g)
        x, y, z = _vec(g.get("pos"))
        g.set("pos", _fmt((x - cx, y - cy, z)))
        drawer.append(g)
    # the slot: from the recess's back face through the north shell to behind the closed drawer
    shell = geoms["versamax_north_shell"]
    (sx, sy, sz), (shx, shy, shz) = _vec(shell.get("pos")), _vec(shell.get("size"))
    x0, x1 = sx - shx, sx + shx                        # shell
    y0, y1 = sy - shy, sy + shy
    z0, z1 = sz - shz, sz + shz
    back = cy + hy + travel + gap                      # behind the closed drawer
    w0, w1 = cx - hx - gap, cx + hx + gap              # the slot's sides
    top = cz + hz + rc["slot_height_m"]                # above the plate on the tray
    reader.remove(shell)
    for name, lo, hi in (("versamax_north_shell", (x0, back, z0), (x1, y1, z1)),
                         ("versamax_north_shell_west", (x0, y0, z0), (w0, back, z1)),
                         ("versamax_north_shell_east", (w1, y0, z0), (x1, back, z1)),
                         ("versamax_north_shell_top", (w0, y0, top), (w1, back, z1))):
        reader.append(_box(name, lo, hi, material="versamax_white"))
    # Allan's dark face on the recess's back wall, with the slot's opening left out; the slot is lined dark
    face = geoms["versamax_recess_north_face"]
    (fx, fy, fz), (fhx, fhy, fhz) = _vec(face.get("pos")), _vec(face.get("size"))
    reader.remove(face)
    fy0, fy1 = fy - fhy, fy + fhy
    for name, lo, hi in (("versamax_recess_north_face_west", (fx - fhx, fy0, fz - fhz), (w0, fy1, fz + fhz)),
                         ("versamax_recess_north_face_east", (w1, fy0, fz - fhz), (fx + fhx, fy1, fz + fhz)),
                         ("versamax_recess_north_face_top", (w0, fy0, top), (w1, fy1, fz + fhz)),
                         ("versamax_slot_back", (w0, back, z0), (w1, back + 0.0006, top)),
                         ("versamax_slot_roof", (w0, y0, top - 0.0006), (w1, back, top)),
                         ("versamax_slot_west", (w0, y0, z0), (w0 + 0.0006, back, top)),
                         ("versamax_slot_east", (w1 - 0.0006, y0, z0), (w1, back, top))):
        reader.append(_box(name, lo, hi, material="versamax_black", contype="0", conaffinity="0"))
    actuator = root.find("actuator")
    ET.SubElement(actuator, "position", name=DRAWER_ACTUATOR, joint=DRAWER_JOINT, kp=str(rc["kp"]),
                  ctrlrange=_fmt((-0.01, travel + 0.005)), ctrllimited="true")


def _box(name, lo, hi, **attrs):
    pos = [(a + b) / 2 for a, b in zip(lo, hi)]
    half = [(b - a) / 2 for a, b in zip(lo, hi)]
    return ET.Element("geom", name=name, type="box", pos=_fmt(pos), size=_fmt(half), **attrs)


def _geom(name, pos, half, rgba):
    return ET.Element("geom", name=name, type="box", pos=_fmt(pos), size=_fmt(half), rgba=rgba,
                      contype="1", conaffinity="1", friction="0.8 0.005 0.0001")


def build(cfg, n_plates, n_tips=0, cameras=None):
    """(model, data, info, runtime, kin) for Allan's workcell with n_plates plates and n_tips
    pipette tips, all parked. `cameras` (mujoco_bridge.cameras.load()) adds the lab's cameras and
    their housings, which neither collide nor weigh anything; runs leave it out."""
    edit = _edit(cfg, n_plates, n_tips)
    if cameras:
        edit = C.edit(cameras, then=edit)
    return L.load(cfg.get("physics"), edit=edit)
