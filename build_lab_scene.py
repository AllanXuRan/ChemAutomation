"""Build a configurable OT-2 -> xArm5/G2 -> VersaMax MuJoCo workcell.

Requires an existing xarm_description/xarm5_bio.urdf, converted earlier.
No ROS2/Gazebo installation required. Geometry is a schematic placeholder:
positions and dimensions must be calibrated against the real lab before use.

Robot BASE frame IS the MuJoCo WORLD frame: (0,0,0), +X right, +Y back, +Z up.
Tabletop is slightly BELOW the origin. All dimensions in metres.

Edit lab_layout.json, then:
    python build_lab_scene.py --build-only
    python interactive_lab_viewer.py

For an alternate plate location (visualization only):
    python build_lab_scene.py --plate-at reader --build-only
"""

from __future__ import annotations
import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
URDF = ROOT / "xarm_description" / "xarm5_bio.urdf"
SCENE = ROOT / "xarm_description" / "xarm5_lab_scene.xml"
CONFIG = ROOT / "lab_layout.json"
DEFAULT_JOINT_ANGLES = (0.0, -0.60, -1.20, 0.80, 0.0)

# OT-2 geometry is only a schematic approximation; don't use slot XY as calibration!
OT2_SIZE = (0.63, 0.57, 0.66)
OT2_DECK_Z = 0.068
VERSAMAX_SIZE = (0.58, 0.38, 0.22)
TEMPERATURE_SUPPORTED_SLOTS = {1, 3, 4, 6, 7, 9, 10}

# Published OT-2 deck holder coordinates from PyLabRobot (metres):
# https://github.com/PyLabRobot/pylabrobot/blob/main/pylabrobot/resources/opentrons/deck.py
# These are reference positions, NOT a replacement for the Opentrons hardware DXF.
OT2_SLOT_PITCH_X_M = 0.1325
OT2_SLOT_PITCH_Y_M = 0.0905
OT2_SLOT_REFERENCE_SIZE_M = (0.128, 0.086)
SLOT_PLATE_CLEARANCE_TOTAL_M = 0.001  # +1 mm of total width and total depth
LEGACY_PLACEHOLDER_SLOT_SIZE_M = (0.116, 0.078)


def apply_geometry_defaults(config):
    """Migrate V5 configs without losing instrument placement or deck slots."""
    o, v, plate = config['ot2'], config['versamax'], config['plate']
    o.setdefault('body_size_m', [0.63, 0.57, 0.66])
    o.setdefault('temperature_slot', 6)  # legacy metadata only; free XYZ is authoritative
    # V9: seat the Deck directly ON the lower OT-2 base, not floating 4 cm above it.
    # Keep old deck_height_m untouched for backwards compatibility, but it is
    # only used if the user chooses deck_mount_mode='custom'.
    o.setdefault('deck_height_m', 0.105)
    # Bottom OT-2 supporting plinth is 6.8 cm above its local bottom.
    # Keep a user's explicit value; V8 had no base_thickness_m, so it gets 0.068.
    o.setdefault('base_thickness_m', 0.068)
    o.setdefault('deck_thickness_m', 0.002)
    o.setdefault('deck_mount_mode', 'on_base')
    # Deck and turquoise slots used to look suspended. Hide both by default;
    # equipment and plates rest directly on the plinth top.
    o.setdefault('show_deck_slab', False)
    o.setdefault('show_slot_tiles', False)
    # V11: all 11 deck-slot OUTLINES are visible on the real lower support
    # surface, regardless of the deprecated raised/filled tile option.
    o.setdefault('show_slot_outlines', True)
    # V12: preserve custom dimensions, but automatically replace the old
    # 116x78 mm placeholder with plate footprint +1 mm on each TOTAL axis.
    # OT-2 publicly specifies ANSI/SLAS-compatible slots, while the exact
    # cutout geometry is in its official DXF. This 1 mm addition is a
    # user-requested SIMULATION clearance, not a factory tolerance.
    previous_slot = o.get('slot_tile_size_m')
    if 'slot_size_mode' not in o:
        is_legacy = previous_slot is None or (
            len(previous_slot) == 2 and
            all(abs(float(a) - b) < 1e-8 for a, b in
                zip(previous_slot, LEGACY_PLACEHOLDER_SLOT_SIZE_M))
        )
        o['slot_size_mode'] = 'plate_plus_1mm' if is_legacy else 'custom'
    o.setdefault('slot_tile_size_m', list(LEGACY_PLACEHOLDER_SLOT_SIZE_M))
    # Arbitrary signed lift relative to the lower base; 0 = direct contact.
    o.setdefault('deck_vertical_offset_m', 0.0)
    # Fine placement relative to the selected slot in the OT-2 local XYZ frame.
    o.setdefault('temperature_offset_xyz_m', [0.0, 0.0, 0.0])
    # Migrate legacy slot-relative coordinates ONCE to independent OT-2 local XY.
    # Third component is a vertical offset relative to top of lower plinth.
    if 'temperature_local_xyz_m' not in o:
        old_slot = int(o.get('temperature_slot', 6))
        n = old_slot - 1
        row, col = divmod(n, 3)
        xx = (col-1)*OT2_SLOT_PITCH_X_M
        yy = (row-1.5)*OT2_SLOT_PITCH_Y_M
        ox, oy, oz = map(float, o['temperature_offset_xyz_m'])
        o['temperature_local_xyz_m'] = [xx + ox, yy + oy, oz]
    o.setdefault('temperature_base_size_m', [0.090, 0.194, 0.084])
    o.setdefault('temperature_block_size_m', [0.128, 0.084, float(o.get('temperature_block_height_m', 0.042))])
    v.setdefault('body_size_m', [0.58, 0.38, 0.22])
    # V7 geometry: southwest-corner notch, OPEN on WEST and SOUTH.
    # recess_size_m = [X width, Y depth, vertical height cut down from top].
    # Old V6 JSON configs are migrated without changing instrument XY/Yaw.
    v.setdefault('body_size_m', [0.58, 0.38, 0.22])
    if 'recess_size_m' not in v:
        rw = float(v.get('recess_width_m', 0.200))
        rd = float(v.get('recess_depth_m', 0.130))
        rh = float(v['body_size_m'][2]) - float(v.get('recess_floor_height_m', 0.075))
        v['recess_size_m'] = [rw, rd, rh]
        # Preserve the user's existing tray dimensions EXACTLY during migration.
        # Do not force a tray to fit within the recess or change its height.
        if 'tray_size_m' not in v:
            v['tray_size_m'] = [0.160, 0.115, 0.012]
    v.setdefault('tray_size_m', [0.160, 0.115, 0.012])
    # 0: tray front edge flush with SOUTH edge; signed values also allowed.
    v.setdefault('tray_retraction_m', 0.0)
    v.setdefault('tray_surface_height_m', 0.100)
    # Delete V6 knobs whose meaning is no longer valid. Do not re-use them.
    for old_key in ('opening_side', 'recess_width_m', 'recess_depth_m',
                    'recess_offset_m', 'recess_floor_height_m', 'tray_extension_m'):
        v.pop(old_key, None)
    if 'size_m' not in plate:
        plate['size_m'] = [2*float(x) for x in plate.get('half_size', [0.06388,0.04274,0.0075])]
    # Keep only ONE authoritative key for dimensions. V5 duplicates are migrated away.
    plate.pop('half_size', None)
    # Keep the user's measured values untouched. 0.128,0.015,0.085 can mean
    # 128 mm wide, 15 mm thick, 85 mm deep, measured in local X,Y,Z axes.
    # For a HORIZONTAL plate, local XZ becomes horizontal XY (90-degree roll).
    if 'flat_surface_axes' not in plate:
        a, b, c = map(float, plate['size_m'])
        plate['flat_surface_axes'] = 'XZ' if b < c else 'XY'
    # Always update the visual slot size after determining how this user's
    # plate is oriented flat. Does not mutate the user's plate dimensions.
    slot_mode = o['slot_size_mode']
    if slot_mode == 'plate_plus_1mm':
        flat = plate_flat_size_m(plate)
        o['slot_tile_size_m'] = [round(flat[0] + SLOT_PLATE_CLEARANCE_TOTAL_M, 6),
                                 round(flat[1] + SLOT_PLATE_CLEARANCE_TOTAL_M, 6)]
    elif slot_mode == 'reference_128x86':
        o['slot_tile_size_m'] = list(OT2_SLOT_REFERENCE_SIZE_M)
    elif slot_mode != 'custom':
        raise ValueError(f'Unknown OT-2 slot_size_mode: {slot_mode}')
    o.pop('temperature_block_height_m', None)
    table = config['table']
    table.setdefault('auto_fit', True)
    table.setdefault('auto_fit_margin_m', 0.20)
    # table.size_xy remains the user-chosen MINIMUM size in auto-fit mode.
    return config


def opening_side(config, which):
    """Device facing: OT2 front and VersaMax SOUTH exposed edge (-local Y)."""
    return 'front'


def versamax_local_regions(config):
    """Authoritative southwest opening and freely sized tray geometry.

    Both 2D layout-editor and 3D MJCF builder call this SAME function.
    Local reader: +X right/east, +Y back/north.
    Notch shares its WEST and SOUTH edges with the instrument footprint.
    """
    v = config['versamax']
    w, d, h = map(float, v['body_size_m'])
    rw, rd, rh = map(float, v['recess_size_m'])
    tw, td, th = map(float, v['tray_size_m'])
    inset = float(v['tray_retraction_m'])
    xmin, ymin = -w / 2, -d / 2
    xmax, ymax = xmin + rw, ymin + rd
    tray_center = (xmin + rw / 2, ymin + inset + td / 2)
    return {
        'opening_center_xy': ((xmin + xmax) / 2, (ymin + ymax) / 2),
        'opening_size_xy': (rw, rd),
        'opening_south_y': ymin,
        'opening_west_x': xmin,
        'opening_north_y': ymax,
        'opening_east_x': xmax,
        'recess_floor_z': h - rh,
        'tray_center_xy': tray_center,
        'tray_size_xy': (tw, td),
        'tray_south_y': ymin + inset,
        'opening_edge_xy': (xmin + rw / 2, ymin),
        'opening_dir_xy': (0.0, -1.0),
        'north_wall_local': ((xmin, ymax), (xmax, ymax)),
        'east_wall_local': ((xmax, ymin), (xmax, ymax)),
    }


def validate_geometry(config):
    """Validate only numeric types that MuJoCo actually requires.

    All geometry dimensions remain INDEPENDENT. No checks that a plate fits
    the tray, that the tray fits the recess, that the recess fits the case,
    that a stack pitch exceeds plate height, or that the plate is above a tray.
    Overlap/overhang is allowed for layout and grasp-pose experiments.
    """
    o, v, plate = config['ot2'], config['versamax'], config['plate']

    def positive_xyz(label, value):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f'{label}: enter exactly 3 dimensions [X, Y, Z] (m)')
        try:
            sizes = [float(x) for x in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label}: all dimensions must be numbers') from exc
        if not all(math.isfinite(x) and x > 0 for x in sizes):
            raise ValueError(f'{label}: each geometry size must be positive and finite')
        return sizes

    for key in ('temperature_base_size_m', 'temperature_block_size_m', 'body_size_m'):
        positive_xyz(f'ot2.{key}', o[key])
    for key in ('body_size_m', 'recess_size_m', 'tray_size_m'):
        positive_xyz(f'versamax.{key}', v[key])
    positive_xyz('plate.size_m', plate['size_m'])
    tile = o.get('slot_tile_size_m', [0.129, 0.086])
    if not isinstance(tile,(list,tuple)) or len(tile)!=2 or not all(math.isfinite(float(x)) and float(x)>0 for x in tile):
        raise ValueError('ot2.slot_tile_size_m must contain 2 positive values')
    if plate.get('flat_surface_axes') not in ('XY', 'XZ'):
        raise ValueError('plate.flat_surface_axes must be XY or XZ')
    if o.get('deck_mount_mode') not in ('on_base', 'custom'):
        raise ValueError('ot2.deck_mount_mode must be on_base or custom')
    positive_xyz('ot2.temperature_base_size_m', o['temperature_base_size_m'])
    local = o['temperature_local_xyz_m']
    if not isinstance(local,(list,tuple)) or len(local)!=3 or not all(math.isfinite(float(z)) for z in local):
        raise ValueError('ot2.temperature_local_xyz_m needs finite X/Y/Z offsets')
    for key in ('base_thickness_m', 'deck_thickness_m'):
        if not math.isfinite(float(o[key])) or float(o[key]) <= 0:
            raise ValueError(f'ot2.{key} must be positive')
    if not math.isfinite(float(o['deck_vertical_offset_m'])):
        raise ValueError('ot2.deck_vertical_offset_m must be finite')
    table = config['table']
    size = table['size_xy']
    if len(size)!=2 or not all(math.isfinite(float(x)) and float(x)>0 for x in size):
        raise ValueError('table.size_xy must contain two positive finite dimensions')
    if not math.isfinite(float(table['auto_fit_margin_m'])) or float(table['auto_fit_margin_m'])<0:
        raise ValueError('table.auto_fit_margin_m must be non-negative')

    # These are COORDINATE OFFSETS, not physical dimensions. They can be
    # negative, zero or positive. The user chooses the desired placement.
    for section, key in (
        (o, 'stack_plate_pitch_m'),
        (v, 'tray_retraction_m'),
        (v, 'tray_surface_height_m'),
    ):
        try:
            value = float(section[key])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{key} must be a numeric value') from exc
        if not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    return True


def fmt(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def el(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def box(parent, name, xyz, half_size, material, **attrs):
    return el(parent, "geom", name=name, type="box", pos=fmt(xyz),
              size=fmt(half_size), material=material, **attrs)


def facing_yaw_deg(which, position_xy, side='front'):
    """Rotate chosen local opening direction to the robot world origin."""
    x,y = map(float, position_xy)
    if math.hypot(x,y)<1e-9:
        raise ValueError('Instrument centre cannot be at robot origin')
    direction=(0.,-1.) if which=='ot2' or side=='front' else (1.,0.)
    target=math.atan2(-y,-x)
    opening=math.atan2(direction[1],direction[0])
    return math.degrees((target-opening+math.pi)%(2*math.pi)-math.pi)


def yaw_quat(yaw_deg):
    """MJCF quaternion w x y z; independent of compiler angle units."""
    a = math.radians(float(yaw_deg)) / 2.0
    return fmt((math.cos(a), 0.0, 0.0, math.sin(a)))


def canonicalize_facing(config):
    """Make any auto-facing yaw agree with its X/Y; return whether changed."""
    changed = False
    for name in ("ot2", "versamax"):
        dev = config[name]
        mode = dev.get("orientation_mode", "face_robot")
        if mode not in ("face_robot", "manual"):
            raise ValueError(f"{name}.orientation_mode must be face_robot or manual")
        if mode == "face_robot":
            new_angle = round(facing_yaw_deg(name, dev["position_xy"], opening_side(config, name)), 6)
            if abs(float(dev.get("yaw_deg", 0.0)) - new_angle) > 1e-7:
                changed = True
            dev["yaw_deg"] = new_angle
        else:
            dev["yaw_deg"] = float(dev["yaw_deg"])
        dev["orientation_mode"] = mode
    return changed


def load_config(path=CONFIG):
    if not path.is_file():
        raise FileNotFoundError(f"Missing layout config: {path}\nPlace lab_layout.json beside this script.")
    c = json.loads(path.read_text(encoding="utf-8"))
    before = json.dumps(c, sort_keys=True)
    apply_geometry_defaults(c)
    canonicalize_facing(c)
    if json.dumps(c, sort_keys=True) != before:
        # JSON is the single source of truth, including computed auto yaw.
        path.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated auto-facing yaw(s) in {path}")
    for field in ("table", "ot2", "versamax", "plate"):
        if field not in c:
            raise ValueError(f"Missing config section '{field}'")
    o = c["ot2"]
    slots = (o['waiting_slot'], o['stack_slot'])
    if any(type(s) is not int or not 1 <= s <= 11 for s in slots):
        raise ValueError('OT-2 waiting/stack slots must be integers 1..11')
    # Keep the 1..11 indexing because coordinates depend on slot numbers,
    # but do not prevent overlapping placements or unofficial module slots.
    if int(o['stack_count']) < 0:
        raise ValueError('stack_count must be a non-negative integer')
    validate_geometry(c)
    for key in ("temperature_yaw_deg", "temperature_base_yaw_deg", "temperature_block_yaw_deg"):
        val = float(o.get(key, 0.0))
        if not math.isfinite(val):
            raise ValueError(f"Invalid OT-2 angle: {key}={val}")
    if c["plate"]["start_at"] not in ("pickup", "reader", "waiting"):
        raise ValueError("plate.start_at must be pickup / reader / waiting")
    if len(c["table"]["center_xy"]) != 2 or len(c["versamax"]["position_xy"]) != 2:
        raise ValueError("XY positions must have exactly two entries")
    return c


def rotate_xy(xy, yaw_deg):
    rad = math.radians(float(yaw_deg))
    x, y = map(float, xy)
    return (x*math.cos(rad)-y*math.sin(rad), x*math.sin(rad)+y*math.cos(rad))


def transform_xy(xy, position_xy, yaw_deg):
    rx, ry = rotate_xy(xy, yaw_deg)
    return (float(position_xy[0])+rx, float(position_xy[1])+ry)


def ot2_slot_xy(slot):
    """Approximate OT-2 grid. Front row 1,2,3; back row 10,11."""
    if not 1 <= int(slot) <= 11:
        raise ValueError("OT-2 has numbered slots 1..11 (12 is fixed trash)")
    n = int(slot)-1
    row, col = divmod(n, 3)
    return ((col-1)*OT2_SLOT_PITCH_X_M, (row-1.5)*OT2_SLOT_PITCH_Y_M)


def ot2_deck_surface_z(ot2):
    """Height above OT-2's local bottom where the deck TOP is located."""
    if ot2.get('deck_mount_mode', 'on_base') == 'on_base':
        return (float(ot2['base_thickness_m'])
                + (float(ot2['deck_thickness_m']) if ot2.get('show_deck_slab', False) else 0.0)
                + float(ot2.get('deck_vertical_offset_m', 0.0)))
    return float(ot2['deck_height_m'])


def plate_flat_size_m(plate):
    """Size XYZ when a plate is resting flat, without changing measured data.

    XY: local XY is the wide flat face; source XYZ -> world XYZ.
    XZ: local XZ is the wide flat face; rotate 90 deg about X in the scene,
        thus source [X, thickness Y, depth Z] -> flat world [X, Z, Y].
    The plate body uses the flat equivalent box so well markers face upward.
    """
    a,b,c = map(float, plate['size_m'])
    return [a,c,b] if plate.get('flat_surface_axes', 'XY') == 'XZ' else [a,b,c]


def temperature_local_xyz(o):
    """Free-placement module position in OT-2 local coordinates.

    XY is independent of any slot. Z=0 means its bottom sits directly on
    the upper surface of the OT-2 support plinth / (optional) thin deck.
    """
    x, y, z = map(float, o['temperature_local_xyz_m'])
    return (x, y, ot2_deck_surface_z(o) + z)


def table_geometry(config):
    """(centre_xy, full_size_xy) of the table, optionally enclosing both instruments.

    Rectangle bounds are computed from each rotated device's actual footprint,
    plus the robot's base. Existing table.size_xy remains a minimum size.
    """
    t = config['table']
    c = tuple(map(float, t['center_xy']))
    size = tuple(map(float, t['size_xy']))
    if not t.get('auto_fit', True):
        return c, size
    xmin = ymin = -0.15
    xmax = ymax = 0.15
    for name in ('ot2','versamax'):
        obj = config[name]
        x,y = map(float, obj['position_xy'])
        w,d = map(float, obj['body_size_m'][:2])
        a = math.radians(float(obj['yaw_deg']))
        hx = abs(w*math.cos(a))/2 + abs(d*math.sin(a))/2
        hy = abs(w*math.sin(a))/2 + abs(d*math.cos(a))/2
        xmin, xmax = min(xmin,x-hx), max(xmax,x+hx)
        ymin, ymax = min(ymin,y-hy), max(ymax,y+hy)
    margin = float(t['auto_fit_margin_m'])
    # Enclose objects *and* at least requested table size.
    w = max(size[0], xmax-xmin+2*margin)
    d = max(size[1], ymax-ymin+2*margin)
    # Center the larger table around the instrument bounding rectangle.
    return ((xmin+xmax)/2, (ymin+ymax)/2), (w,d)


def targets(c):
    table_z=float(c['table']['top_z'])
    o,v,p=c['ot2'],c['versamax'],c['plate']
    ph=plate_flat_size_m(p)[2]/2
    local_height=ot2_deck_surface_z(o)
    mx,my,mz=temperature_local_xyz(o)
    tx=transform_xy((mx,my),o['position_xy'],o['yaw_deg'])
    wx=transform_xy(ot2_slot_xy(o['waiting_slot']),o['position_xy'],o['yaw_deg'])
    sx=transform_xy(ot2_slot_xy(o['stack_slot']),o['position_xy'],o['yaw_deg'])
    v_local=versamax_local_regions(c)['tray_center_xy']
    vx=transform_xy(v_local,v['position_xy'],v['yaw_deg'])
    n=int(o['stack_count'])
    pitch=float(o['stack_plate_pitch_m'])
    return {
      'ot2_pickup': (*tx,table_z+mz+float(o['temperature_base_size_m'][2])+float(o['temperature_block_size_m'][2])+ph),
      'ot2_waiting': (*wx,table_z+local_height+0.003+ph),
      'ot2_stack_top': (*sx,table_z+local_height+0.003+ph+(n-1)*pitch if n else table_z+local_height+0.003+ph),
      'versamax_load': (*vx,table_z+float(v['tray_surface_height_m'])+ph),
    }


def camera_xyaxes(position, target):
    fwd = np.asarray(target, float)-np.asarray(position, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.,0.,1.])
    if np.linalg.norm(right) < 1e-7:
        right = np.array([1.,0.,0.])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fmt([*right, *up])


def resolve_mesh_paths(root, urdf):
    """Maintain compatibility with the existing exported robot MJCF meshes."""
    asset = root.find("asset")
    if asset is None:
        return
    compiler = root.find("compiler")
    for node in asset:
        if node.tag not in ("mesh", "texture", "hfield", "skin"):
            continue
        raw = node.get("file")
        if not raw:
            continue
        source = Path(raw)
        if source.is_file():
            node.set("file", source.resolve().as_posix())
            continue
        candidates = [urdf.parent]
        if compiler is not None:
            for setting in ("meshdir" if node.tag == "mesh" else "texturedir", "assetdir"):
                val = compiler.get(setting)
                if val:
                    candidates.insert(0, urdf.parent / val)
        found = next((folder/source for folder in candidates if (folder/source).is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Mesh/texture not found: {raw}; keep the complete xarm_description/meshes directory")
        node.set("file", found.resolve().as_posix())
    if compiler is not None:
        for attr in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(attr, None)
        compiler.set("strippath", "false")


def materials(root):
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        world = root.find("worldbody")
        if world is None:
            root.append(asset)
        else:
            root.insert(list(root).index(world), asset)
    palette = {
        "robot_white": ("0.90 0.92 0.95 1", .32, .58),
        "robot_dark": ("0.25 0.29 0.34 1", .4, .65),
        "gripper": ("0.16 0.18 0.21 1", .45, .6),
        "metal": ("0.57 0.62 0.68 1", .7, .68),
        "table": ("0.66 0.71 0.77 1", .3, .4),
        "table_edge": ("0.22 0.26 0.31 1", .4, .55),
        "floor": ("0.35 0.40 0.46 1", .12, .2),
        "ot2_frame": ("0.51 0.59 0.63 1", .4, .55),
        "ot2_deck": ("0.66 0.73 0.77 1", .35, .5),
        "ot2_slot": ("0.26 0.44 0.46 1", .22, .3),
        "ot2_slot_outline": ("0.17 0.26 0.32 1", .18, .35),
        "ot2_slot_label": ("0.10 0.19 0.26 1", .12, .22),
        "temperature": ("0.69 0.72 0.77 1", .65, .78),
        "temperature_top": ("0.25 0.32 0.39 1", .42, .6),
        "versamax_white": ("0.81 0.84 0.83 1", .32, .45),
        "versamax_teal": ("0.27 0.56 0.57 1", .37, .46),
        "versamax_black": ("0.17 0.20 0.24 1", .3, .48),
        "plate": ("0.96 0.97 0.99 1", .32, .42),
        "well": ("0.22 0.37 0.52 1", .2, .32),
        "waiting": ("0.15 0.57 0.40 1", .35, .4),
        "stack": ("0.22 0.45 0.77 1", .35, .42),
    }
    existing = {e.get("name") for e in asset.findall("material")}
    for name, (rgba, specular, shine) in palette.items():
        if name not in existing:
            el(asset, "material", name=name, rgba=rgba,
               specular=str(specular), shininess=str(shine))
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Missing robot model worldbody")
    for body in world.iter("body"):
        bname = body.get("name", "").lower()
        for geom in body.findall("geom"):
            if not (geom.get("mesh") or geom.get("type") == "mesh"):
                continue
            name = (bname + " " + geom.get("name", "") + " " + geom.get("mesh", "")).lower()
            if any(s in name for s in ("finger", "gripper", "knuckle")):
                mat = "gripper"
            elif any(s in name for s in ("link2", "link4", "link_base", "base_link")):
                mat = "robot_dark"
            else:
                mat = "robot_white"
            geom.set("material", mat)
            geom.attrib.pop("rgba", None)


def robot_at_world_origin(root):
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("No worldbody")
    mount = ET.Element("body", name="lab_robot_mount", pos="0 0 0")
    for node in list(world):
        if node.tag in ("body", "geom", "site", "frame"):
            world.remove(node)
            mount.append(node)
    world.insert(0, mount)
    el(world, "site", name="robot_base_origin", pos="0 0 0", size="0.015",
       rgba="1 0.25 0.25 0.9", group="4")
    return world


def add_table(world, c):
    t = c["table"]
    (tx,ty),(w,d) = table_geometry(c)
    z, h = float(t['top_z']), float(t['height'])
    thickness = 0.07
    # Table supports robot whose base origin is 10mm above its top.
    box(world, "table_top", (tx,ty,z-thickness/2), (w/2,d/2,thickness/2), "table")
    floor_z = z-h
    el(world, "geom", name="floor", type="plane", pos=fmt((0,0,floor_z)),
       size="3.5 3.5 0.1", material="floor", contype="1", conaffinity="1")
    leg_height = h-thickness
    for ix,x in enumerate((tx-w/2+.065,tx+w/2-.065)):
        for iy,y in enumerate((ty-d/2+.065,ty+d/2-.065)):
            box(world, f"table_leg_{ix}_{iy}", (x,y,floor_z+leg_height/2),
                (.035,.035,leg_height/2), "table_edge")
    # A shallow metal robot mounting pad underneath base 0,0,0
    box(world, "robot_mount_pad", (0,0,z+0.004), (.104,.104,.004), "metal")


def add_96_well_markers(body, prefix, half_size):
    sx,sy,sz=map(float,half_size)
    pitch_x=2*sx/13.
    pitch_y=2*sy/9.
    radius=min(0.003,pitch_x*.34,pitch_y*.34)
    for row in range(8):
        for col in range(12):
            x=(col-5.5)*pitch_x
            y=(row-3.5)*pitch_y
            el(body,'geom',name=f'{prefix}_well_{row+1}_{col+1}',type='cylinder',
               pos=fmt((x,y,sz+0.00035)),size=fmt((radius,0.0003)),
               material='well',contype='0',conaffinity='0',group='1')


def add_surface_slot_guides(body, ot2, support_top_z):
    """Draw eleven numbered OT-2 deck footprints on the solid base surface.

    These are FLAT visual guides, not raised pedestals or collision objects.
    Their lower faces sit exactly on the top of the supporting base (6.8 cm
    by default), so removing an optional Deck slab never leaves them floating.
    """
    sw, sd = map(float, ot2['slot_tile_size_m'])  # full X,Y footprint
    edge = min(0.0020, sw / 18, sd / 18)
    half_t = 0.00016
    z = support_top_z + half_t
    # Seven-segment numerals directly on the support, visible in MuJoCo
    # without changing the viewer's optional label display settings.
    segments = {
        '0': 'ab cdef'.replace(' ', ''),
        '1': 'bc',
        '2': 'abged',
        '3': 'abgcd',
        '4': 'fgbc',
        '5': 'afgcd',
        '6': 'afgecd',
        '7': 'abc',
        '8': 'abcdefg',
        '9': 'abcdfg',
    }
    for slot in range(1, 12):
        x, y = ot2_slot_xy(slot)
        # Four narrow strips on top of the supporting base.
        for end in (-1, 1):
            box(body, f'ot2_slot_outline_{slot}_y_{end}',
                (x, y + end * (sd / 2 - edge / 2), z),
                (sw / 2, edge / 2, half_t), 'ot2_slot_outline',
                contype='0', conaffinity='0')
            box(body, f'ot2_slot_outline_{slot}_x_{end}',
                (x + end * (sw / 2 - edge / 2), y, z),
                (edge / 2, sd / 2, half_t), 'ot2_slot_outline',
                contype='0', conaffinity='0')
        # Numerals near the north-west corner, leaving the centre free for plates.
        digits = str(slot)
        char_w = 0.008
        char_h = 0.014
        gap = 0.003
        total = len(digits) * char_w + (len(digits) - 1) * gap
        left = x - sw / 2 + 0.009
        top = y + sd / 2 - 0.007
        for i, digit in enumerate(digits):
            xc = left + i * (char_w + gap) + char_w / 2
            yc = top - char_h / 2
            defs = {
                'a': (0, +char_h / 2, char_w / 2, 0.00055),
                'g': (0, 0, char_w / 2, 0.00055),
                'd': (0, -char_h / 2, char_w / 2, 0.00055),
                'f': (-char_w / 2, +char_h / 4, 0.00055, char_h / 4),
                'b': (+char_w / 2, +char_h / 4, 0.00055, char_h / 4),
                'e': (-char_w / 2, -char_h / 4, 0.00055, char_h / 4),
                'c': (+char_w / 2, -char_h / 4, 0.00055, char_h / 4),
            }
            for segment in segments[digit]:
                dx, dy, hx, hy = defs[segment]
                box(body, f'ot2_slot_number_{slot}_digit_{i}_{segment}',
                    (xc + dx, yc + dy, support_top_z + 0.00038),
                    (hx, hy, 0.00012), 'ot2_slot_label',
                    contype='0', conaffinity='0')


def add_ot2(world,c):
    o,t,p=c['ot2'],c['table'],c['plate']
    table_z=float(t['top_z'])
    w,d,h=map(float,o['body_size_m'])
    deck_z=ot2_deck_surface_z(o)
    base_t=float(o['base_thickness_m'])
    deck_t=float(o['deck_thickness_m'])
    hx,hy=w/2,d/2
    body=el(world,'body',name='ot2_enclosure',
            pos=fmt((*o['position_xy'],table_z)),quat=yaw_quat(o['yaw_deg']))
    box(body,'ot2_base',(0,0,base_t/2),(hx,hy,base_t/2),'ot2_frame')
    if o.get('show_deck_slab', False):
        box(body,'ot2_deck',(0,0,deck_z-deck_t/2),(.265,.245,deck_t/2),'ot2_deck')
    beam=.012
    for i,x in enumerate((-hx+beam,hx-beam)):
        for j,y in enumerate((-hy+beam,hy-beam)):
            box(body,f'ot2_frame_post_{i}_{j}',(x,y,h/2),(beam,beam,h/2),'ot2_frame')
    for j,y in enumerate((-hy+beam,hy-beam)):
        box(body,f'ot2_top_x_{j}',(0,y,h-beam),(hx,beam,beam),'ot2_frame')
    for i,x in enumerate((-hx+beam,hx-beam)):
        box(body,f'ot2_top_y_{i}',(x,0,h-beam),(beam,hy,beam),'ot2_frame')
    for i,x in enumerate((-hx+beam,hx-beam)):
        box(body,f'ot2_side_rail_{i}',(x,0,.16),(beam,hy,beam),'ot2_frame')
    box(body,'ot2_back_rail',(0,hy-beam,.16),(hx,beam,beam),'ot2_frame')
    if o.get('show_slot_tiles', False):
        slot_w,slot_d=map(float,o['slot_tile_size_m'])
        for slot in range(1,12):
            x,y=ot2_slot_xy(slot)
            box(body,f'ot2_deck_slot_{slot}',(x,y,deck_z+.00055),
                (slot_w/2,slot_d/2,.00055), 'ot2_slot',
                contype='0',conaffinity='0')
    if o.get('show_slot_outlines', True):
        # Flat on the actual supporting surface, not a second floating platform.
        top = (deck_z if o.get('show_deck_slab', False) else base_t)
        if o.get('show_slot_tiles', False):
            top = deck_z + 0.0011
        add_surface_slot_guides(body, o, top)
    # Two independent editable rectangular solids: base housing + upper aluminium block.
    mx,my,mz=temperature_local_xyz(o)
    mod=el(body,'body',name='ot2_temperature_module',
           pos=fmt((mx,my,mz)),quat=yaw_quat(o['temperature_yaw_deg']))
    bw,bd,bh=map(float,o['temperature_base_size_m'])
    aw,ad,ah=map(float,o['temperature_block_size_m'])
    base_angle=float(o.get('temperature_base_yaw_deg',90))
    block_angle=float(o.get('temperature_block_yaw_deg',0))
    box(mod,'temperature_module_body',(0,0,bh/2),(bw/2,bd/2,bh/2),'temperature',
        quat=yaw_quat(base_angle))
    box(mod,'temperature_aluminium_block',(0,0,bh+ah/2),(aw/2,ad/2,ah/2),
        'temperature_top',quat=yaw_quat(block_angle))
    box(mod,'temperature_contact_plate',(0,0,bh+ah-.0003),
        (max(.002,aw/2-.002),max(.002,ad/2-.002),.0003),'temperature',
        contype='0',conaffinity='0',quat=yaw_quat(block_angle))
    wx,wy=ot2_slot_xy(o['waiting_slot'])
    phalf=[float(x)/2 for x in plate_flat_size_m(p)]
    # Both visual landing pads share the same Deck-slot footprint; otherwise
    # the coloured pads look inconsistent with the eleven numbered slots.
    # Plates themselves remain at their independently measured dimensions.
    slot_half = [float(x)/2 for x in o['slot_tile_size_m']]
    box(body,'ot2_waiting_pad',(wx,wy,deck_z+.0015),
        (slot_half[0],slot_half[1],.0015),'waiting',contype='0',conaffinity='0')
    sx,sy=ot2_slot_xy(o['stack_slot'])
    box(body,'ot2_stack_pad',(sx,sy,deck_z+.0015),
        (slot_half[0],slot_half[1],.0015),'stack',contype='0',conaffinity='0')
    pitch=float(o['stack_plate_pitch_m'])
    for i in range(int(o['stack_count'])):
        z=deck_z+.003+phalf[2]+i*pitch
        box(body,f'ot2_stack_plate_{i+1}',(sx,sy,z),phalf,'plate',contype='0',conaffinity='0')
        if i==int(o['stack_count'])-1:
            top=el(body,'body',name='ot2_stack_top_visual',pos=fmt((sx,sy,z)))
            add_96_well_markers(top,'stack',phalf)
    return body


def add_versamax(world, c):
    """VersaMax with a TRUE SW-corner cutout (no west/south high walls).

    The dimensions of notch and tray are intentionally independent.
    With retraction=0, their southern edges are aligned; the user may select
    any positive tray size, even when it overlaps its surroundings.
    This is a STATIC layout preview, not a validated physical mechanism.
    """
    v = c['versamax']
    table_z = float(c['table']['top_z'])
    w, d, h = map(float, v['body_size_m'])
    rw, rd, rh = map(float, v['recess_size_m'])
    hx, hy = w/2, d/2
    floor = h - rh
    reg = versamax_local_regions(c)
    body = el(world, 'body', name='versamax_reader',
              pos=fmt((*v['position_xy'], table_z)), quat=yaw_quat(v['yaw_deg']))

    # 1. Housing is assembled from POSITIVE-SIZE boxes only.
    # When a requested recess is larger/deeper than the case, skip shell
    # segments that have no physical volume instead of rejecting the input.
    # The red recess dimensions still follow the JSON exactly.
    floor_in_housing = max(0.0, floor)
    if floor_in_housing > 1e-8:
        box(body, 'versamax_lower_housing', (0, 0, floor_in_housing / 2),
            (hx, hy, floor_in_housing / 2), 'versamax_white')

    # 2. L-shaped upper housing: only NORTH and EAST material is present.
    shell_height = h - floor_in_housing
    zc = floor_in_housing + shell_height / 2
    zh = shell_height / 2
    north_span = max(0.0, d-rd)
    east_span = max(0.0, w-rw)
    southern_occupied = min(d, rd)
    if shell_height > 1e-8 and north_span > 1e-8:
        box(body, 'versamax_north_shell', (0, hy-north_span/2, zc),
            (hx, north_span/2, zh), 'versamax_white')
    if shell_height > 1e-8 and east_span > 1e-8 and southern_occupied > 1e-8:
        box(body, 'versamax_east_shell', (hx-east_span/2, -hy+southern_occupied/2, zc),
            (east_span/2, southern_occupied/2, zh), 'versamax_white')

    # 3. Dark inset floor within recess (purely visual, no 'fake wall').
    ox,oy = reg['opening_center_xy']
    box(body,'versamax_recess_floor',(ox,oy,floor+.0003),
        (rw/2,rd/2,.0003),'versamax_black',contype='0',conaffinity='0')
    # Two subtle dark inner faces show the north/east vertical inside edges.
    # They are embedded at existing shell surfaces, no west/south barriers.
    if shell_height > 1e-8 and north_span > 1e-8:
        box(body,'versamax_recess_north_face',
            (ox,reg['opening_north_y']+.0003,zc),
            (rw/2,.0003,zh),'versamax_black',contype='0',conaffinity='0')
    if shell_height > 1e-8 and east_span > 1e-8 and southern_occupied > 1e-8:
        box(body,'versamax_recess_east_face',
            (reg['opening_east_x']+.0003,oy,zc),
            (.0003,rd/2,zh),'versamax_black',contype='0',conaffinity='0')

    # 4. The instrument control panel is on the remaining SOLID top shell.
    # Put it at +X,+Y, not above the notch.
    console_x = max(-hx+.12, min(hx-.12, hx-.14))
    console_y = hy-.080
    box(body,'versamax_console',(console_x,console_y,h+.001),
        (.10,.042,.002),'versamax_teal',contype='0',conaffinity='0')
    box(body,'versamax_screen',(console_x,console_y,h+.004),
        (.066,.016,.001),'versamax_black',contype='0',conaffinity='0')

    # 5. Tray fits ENTIRELY inside southwest cavity, even when fully extended.
    tcx,tcy = reg['tray_center_xy']
    tw,td,th = map(float,v['tray_size_m'])
    surf = float(v['tray_surface_height_m'])
    box(body,'versamax_sliding_tray',(tcx,tcy,surf-th/2),
        (tw/2,td/2,th/2),'versamax_black')
    # Optional decorative insert. For a very narrow tray don't generate a
    # negative-size inset geom (MuJoCo rightly refuses negative geometry).
    if tw > .012 and td > .012:
        box(body,'versamax_tray_insert',(tcx,tcy,surf-.00025),
            (tw/2-.006,td/2-.006,.00025),'temperature',
            contype='0',conaffinity='0')
    # Thin guide rails are VISUAL details, not a bound on plate dimensions.
    if tw > .006:
        for i,sgn in enumerate((-1,1)):
            box(body,f'versamax_tray_guide_{i}',
                (tcx+sgn*(tw/2-.003),tcy,surf+.004),
                (.003,td/2,.004),'versamax_black',contype='0',conaffinity='0')
    return body


def add_reaction_plate(world, c, place):
    p=c["plate"]
    t=targets(c)
    point=t[{"pickup":"ot2_pickup","reader":"versamax_load","waiting":"ot2_waiting"}[place]]
    if place == "reader":
        yaw=float(c["versamax"]["yaw_deg"])
    elif place == "pickup":
        # Reaction plate follows the temperature aluminium block, NOT its base.
        o=c["ot2"]
        yaw=(float(o["yaw_deg"])+float(o["temperature_yaw_deg"])
             +float(o.get("temperature_block_yaw_deg",0.0)))
    else:
        yaw=float(c["ot2"]["yaw_deg"])
    body=el(world,"body",name="reaction_plate",pos=fmt(point),
            quat=yaw_quat(yaw))
    if not p.get("preview_static_plate", True):
        el(body,"freejoint",name="reaction_plate_free")
    phalf=[float(x)/2 for x in plate_flat_size_m(p)]
    el(body,"geom",name="reaction_plate_box",type="box",size=fmt(phalf),
       mass="0.055",material="plate",friction="0.8 0.006 0.0001",
       contype="1",conaffinity="1")
    add_96_well_markers(body,"reaction",phalf)


def add_target_sites(world, c):
    points=targets(c)
    colors={"ot2_pickup":"1 0.7 0.1 0.9",
            "ot2_waiting":"0.0 0.95 0.5 0.9",
            "ot2_stack_top":"0.2 0.5 1 0.9",
            "versamax_load":"0.9 0.3 0.8 0.9"}
    for name,xyz in points.items():
        el(world,"site",name=name,type="sphere",pos=fmt(xyz),
           size="0.008",rgba=colors[name],group="4")


def add_lights_and_cameras(world,c):
    # Fixed cameras reference the robot-origin convention.
    el(world,"light",name="key_light",pos="0 -0.8 2.0",dir="0 0.3 -1",
       directional="true",diffuse="0.91 0.92 1",specular="0.22 0.22 0.22",
       castshadow="true")
    el(world,"light",name="fill_light",pos="0.9 0.8 1.65",dir="-0.3 -0.4 -1",
       directional="true",diffuse="0.43 0.48 0.55",specular="0.1 0.1 0.1",
       castshadow="false")
    # Dynamically frame the expanded tabletop rather than clipping remote gear.
    (cx,cy),(tw,td)=table_geometry(c)
    span=max(tw,td,1.25)
    target=(cx,cy,0.13)
    configs={
        "overview": ((cx+span*.72,cy-span*.92,max(1.30,span*.65)),target,49),
        "top": ((cx,cy,max(2.00,span*1.45)),(cx,cy,0.04),46),
        "side": ((cx+span*.94,cy,max(1.15,span*.56)),target,48),
    }
    for name,(pos,target,fovy) in configs.items():
        el(world,"camera",name="lab_"+name,pos=fmt(pos),
           xyaxes=camera_xyaxes(pos,target),fovy=str(fovy))


def add_visual(root):
    visual=root.find("visual")
    if visual is None:
        visual=ET.Element("visual")
        world=root.find("worldbody")
        if world is not None:
            root.insert(list(root).index(world), visual)
        else:
            root.append(visual)
    if visual.find("headlight") is None:
        el(visual,"headlight",ambient="0.38 0.38 0.38",
           diffuse="0.55 0.55 0.55",specular="0.13 0.13 0.13")
    if visual.find("quality") is None:
        el(visual,"quality",shadowsize="2048")
    if visual.find("global") is None:
        el(visual,"global",offwidth="1280",offheight="800")


def add_robot_position_actuators(root, robot_model):
    import mujoco
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.Element("actuator")
        root.append(actuator)
    existing = {a.get("name") for a in actuator}
    for i in range(1,6):
        jname=f"joint{i}"
        jid=mujoco.mj_name2id(robot_model,mujoco.mjtObj.mjOBJ_JOINT,jname)
        if jid<0:
            raise RuntimeError(f"Missing joint in URDF: {jname}")
        low,high=robot_model.jnt_range[jid] if robot_model.jnt_limited[jid] else (-6.28,6.28)
        if f"{jname}_position" not in existing:
            el(actuator,"position",name=f"{jname}_position",joint=jname,
               kp="85",kv="12",ctrllimited="true",ctrlrange=fmt((low,high)))


def report_layout(c, out, plate_at):
    o,v=c["ot2"],c["versamax"]
    t=targets(c)
    dist={name:round(math.hypot(x,y),3) for name,(x,y,z) in t.items()}
    report={
        "units":"metres",
        "coordinate_frame":"xArm5 BASE = WORLD origin; +X right, +Y back, +Z up",
        "robot_base_world":[0,0,0],
        'table_top_z':c['table']['top_z'],
        'table_auto_fit':c['table']['auto_fit'],
        'table_actual_center_xy':table_geometry(c)[0],
        'table_actual_size_xy':table_geometry(c)[1],
        "ot2_temperature_slot":o["temperature_slot"],
        "ot2_waiting_slot":o["waiting_slot"],
        "ot2_stack_slot":o["stack_slot"],
        "ot2_stack_count":o["stack_count"],
        "ot2_xy":o["position_xy"],
        "ot2_orientation_mode":o["orientation_mode"],
        "ot2_yaw_deg":o["yaw_deg"],
        "ot2_temperature_yaw_deg":o["temperature_yaw_deg"],
        "ot2_temperature_base_yaw_deg":o.get("temperature_base_yaw_deg",90.0),
        "ot2_temperature_block_yaw_deg":o.get("temperature_block_yaw_deg",0.0),
        "versamax_xy":v["position_xy"],
        "versamax_orientation_mode":v["orientation_mode"],
        "versamax_yaw_deg":v["yaw_deg"],
        "versamax_recess_open_edges":["west","south"],
        "versamax_recess_size_m":v["recess_size_m"],
        "versamax_tray_retraction_m":v["tray_retraction_m"],
        "versamax_cutout_center_local_m":versamax_local_regions(c)["opening_center_xy"],
        "versamax_tray_center_local_m":versamax_local_regions(c)["tray_center_xy"],
        "versamax_tray_size_m":v["tray_size_m"],
        "aluminium_block_size_m":o["temperature_block_size_m"],
        "ot2_deck_surface_z_m":ot2_deck_surface_z(o),
        "ot2_slot_tile_size_m":o['slot_tile_size_m'],
        "ot2_slot_size_mode":o['slot_size_mode'],
        "ot2_slot_center_pitch_xy_m": [OT2_SLOT_PITCH_X_M, OT2_SLOT_PITCH_Y_M],
        "ot2_slot_size_note": "plate_plus_1mm is modelling clearance, not certified manufacturer tolerance",
        "temperature_local_xyz_m":o["temperature_local_xyz_m"],
        "ot2_plinth_height_m":o["base_thickness_m"],
        "ot2_show_deck_slab":o["show_deck_slab"],
        "ot2_show_slot_tiles":o["show_slot_tiles"],
        "plate_measurement_axes":c['plate']['flat_surface_axes'],
        "plate_flat_size_m":plate_flat_size_m(c['plate']),
        "reaction_plate_size_m":c["plate"]["size_m"],
        "reaction_plate_start":plate_at,
        "targets_robot_base_frame":t,
        "horizontal_distance_from_robot_base":dist,
        "WARNING":"Schematic scene: validate actual geometry, plate gripper coupling, OT-2 doors, reachability, collisions and instrument protocols before hardware operation"
    }
    path=out.with_name("lab_layout_targets.json")
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Targets in robot-base (WORLD) coordinates:\n{json.dumps(t,indent=2)}")
    print(f"Saved target report: {path}")
    for name, dist in dist.items():
        if dist > 0.70:
            print(f"WARNING: {name} XY distance {dist:.3f} m exceeds nominal 0.70 m arm reach")


def build(c, urdf, output, plate_at):
    import mujoco
    if not urdf.is_file():
        raise FileNotFoundError(f"Missing robot URDF: {urdf}")
    print("[1/4] Importing xArm5/G2 original URDF ...")
    robot_model=mujoco.MjModel.from_xml_path(str(urdf.resolve()))
    print(f" robot joints={robot_model.njnt} geoms={robot_model.ngeom}")
    print("[2/4] Exporting MJCF and decorating robot ...")
    robot_xml=output.with_name("xarm5_robot_exported.xml")
    mujoco.mj_saveLastXML(str(robot_xml.resolve()),robot_model)
    tree=ET.parse(robot_xml)
    root=tree.getroot()
    root.set("model","xArm5 Bio G2 - OT2 and VersaMax | base-origin frame")
    resolve_mesh_paths(root,urdf)
    materials(root)
    world=robot_at_world_origin(root)
    print("[3/4] Adding table, slotted OT-2, thermal block, plate stack, VersaMax...")
    add_table(world,c)
    add_ot2(world,c)
    add_versamax(world,c)
    add_reaction_plate(world,c,plate_at)
    add_target_sites(world,c)
    add_lights_and_cameras(world,c)
    add_visual(root)
    add_robot_position_actuators(root,robot_model)
    ET.indent(tree,space="  ")
    tree.write(output,encoding="utf-8",xml_declaration=True)
    print("[4/4] Checking compiled MuJoCo scene...")
    model=mujoco.MjModel.from_xml_path(str(output.resolve()))
    print(f"OK: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.ncam} cameras")
    report_layout(c,output,plate_at)
    return model


def run_preview(model):
    import mujoco
    import mujoco.viewer
    data=mujoco.MjData(model)
    data.qpos[:]=model.qpos0
    for i,angle in enumerate(DEFAULT_JOINT_ANGLES,1):
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"joint{i}")
        if j<0:
            raise RuntimeError(f"Missing joint{i}")
        lo,hi=model.jnt_range[j]
        if model.jnt_limited[j] and not (lo<=angle<=hi):
            raise ValueError(f"joint{i} {angle} outside {lo,hi}")
        data.qpos[model.jnt_qposadr[j]]=angle
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,f"joint{i}_position")
        if a>=0:
            data.ctrl[a]=angle
    mujoco.mj_forward(model,data)
    print("Viewer: open CONTROL for position actuator sliders; JOINT values may be grey.")
    mujoco.viewer.launch(model,data)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=CONFIG,help="layout JSON path")
    p.add_argument("--urdf",type=Path,default=URDF)
    p.add_argument("--build-only",action="store_true")
    p.add_argument("--print-layout",action="store_true",help="Print target points without requiring MuJoCo")
    p.add_argument("--plate-at",choices=("pickup","reader","waiting"),default=None)
    a=p.parse_args()
    c=load_config(a.config)
    plate_at=a.plate_at or c["plate"]["start_at"]
    if a.print_layout:
        print(json.dumps({"robot_base_world":[0,0,0],"target_points_m":targets(c)},indent=2))
        return
    model=build(c,a.urdf.resolve(),a.urdf.parent / "xarm5_lab_scene.xml",plate_at)
    if not a.build_only:
        run_preview(model)


if __name__=="__main__":
    main()
