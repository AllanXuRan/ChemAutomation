"""Verify generated MuJoCo XML rotations match lab_layout.json.

Run after build: python verify_layout.py
Needs Python stdlib only (no MuJoCo import). Both scene poses and JSON are
checked; optional compiled-MuJoCo verification if mujoco is installed.
"""
from __future__ import annotations
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from build_lab_scene import (
    facing_yaw_deg, transform_xy, ot2_slot_xy,
    apply_geometry_defaults, validate_geometry, opening_side,
    versamax_local_regions, targets, temperature_local_xyz,
    ot2_deck_surface_z, plate_flat_size_m, table_geometry)

ROOT = Path(__file__).resolve().parent
CONF = ROOT / 'lab_layout.json'
SCENE = ROOT / 'xarm_description' / 'xarm5_lab_scene.xml'


def floats(s):
    return [float(x) for x in s.split()]


def angle_delta(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def quat_to_yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def check(label, actual, expected, tol):
    delta = actual-expected
    if abs(delta)>tol:
        raise AssertionError(f'{label}: actual {actual:.6f}, JSON expected {expected:.6f}, error {delta:.6f}')
    print(f'  PASS {label}: {actual:.4f} (error {delta:+.2e})')


def validate():
    config = json.loads(CONF.read_text(encoding='utf-8'))
    apply_geometry_defaults(config)
    validate_geometry(config)
    root = ET.parse(SCENE).getroot()
    print('Reading JSON:', CONF)
    print('Reading MJCF:', SCENE)
    world = root.find('worldbody')
    if world is None:
        raise AssertionError('Scene is missing worldbody')
    for key, body_name in [('ot2','ot2_enclosure'),('versamax','versamax_reader')]:
        c = config[key]
        body = world.find(f'body[@name="{body_name}"]')
        if body is None:
            raise AssertionError(f'Cannot find {body_name} in XML')
        if body.get('quat') is None:
            raise AssertionError(f'{body_name} still uses euler angles! Rebuild with V5.')
        xyz = floats(body.get('pos'))
        check(key+' X (m)', xyz[0], float(c['position_xy'][0]), 1e-6)
        check(key+' Y (m)', xyz[1], float(c['position_xy'][1]), 1e-6)
        yaw = quat_to_yaw(floats(body.get('quat')))
        dyaw = angle_delta(yaw, float(c['yaw_deg']))
        check(key+' yaw delta (deg)',dyaw,0.0,0.0005)
        if c.get('orientation_mode', 'face_robot') == 'face_robot':
            target = facing_yaw_deg(key, c['position_xy'], opening_side(config, key))
            check(key+' opening faces robot (deg)',angle_delta(yaw,target),0.0,0.0005)
        print(f'  {key}: XML={yaw:.5f} deg, JSON={c["yaw_deg"]} deg, mode={c.get("orientation_mode") or "face_robot"}')
    temp = world.find('body[@name="ot2_enclosure"]/body[@name="ot2_temperature_module"]')
    if temp is None:
        raise AssertionError('Missing temperature module in XML')
    temp_xy = floats(temp.get('pos'))
    expected_xyz = temperature_local_xyz(config['ot2'])
    check('Temp local X (m)',temp_xy[0],expected_xyz[0],1e-6)
    check('Temp local Y (m)',temp_xy[1],expected_xyz[1],1e-6)
    check('Temp local Z (m)',temp_xy[2],expected_xyz[2],1e-6)
    ot2body = world.find('body[@name="ot2_enclosure"]')
    basegeom = ot2body.find('geom[@name="ot2_base"]')
    base_pos,base_half=floats(basegeom.get('pos')),floats(basegeom.get('size'))
    z_base_top=base_pos[2]+base_half[2]
    check('OT-2 lower base top z (m)',z_base_top,float(config['ot2']['base_thickness_m']),1e-6)
    deckgeom=ot2body.find('geom[@name="ot2_deck"]')
    show_deck=bool(config['ot2'].get('show_deck_slab',False))
    if show_deck:
        if deckgeom is None: raise AssertionError('Requested Deck slab missing')
        deck_pos,deck_half=floats(deckgeom.get('pos')),floats(deckgeom.get('size'))
        z_deck_bottom=deck_pos[2]-deck_half[2]
        if config['ot2'].get('deck_mount_mode','on_base')=='on_base':
            check('Deck bottom minus base top',z_deck_bottom-z_base_top,
                  float(config['ot2']['deck_vertical_offset_m']),1e-6)
        check('Deck upper surface (m)',deck_pos[2]+deck_half[2],
              ot2_deck_surface_z(config['ot2']),1e-6)
    else:
        if deckgeom is not None: raise AssertionError('Deck slab should be disabled')
        if config['ot2'].get('deck_mount_mode','on_base')=='on_base':
            check('Operating platform = base top (m)',ot2_deck_surface_z(config['ot2']),
                  float(config['ot2']['base_thickness_m']) + float(config['ot2']['deck_vertical_offset_m']),1e-6)
    tile=ot2body.find('geom[@name="ot2_deck_slot_1"]')
    show_tiles=bool(config['ot2'].get('show_slot_tiles',False))
    if show_tiles:
        if tile is None: raise AssertionError('Expected visible Deck tile')
        half=floats(tile.get('size'))
        for ax,val in enumerate(config['ot2']['slot_tile_size_m']):
            check(f'Deck tile axis {ax}',half[ax]*2,float(val),1e-6)
    elif tile is not None:
        raise AssertionError('Physical/visual Deck slot tiles should be disabled')

    # V12: verify every visible outline uses the same JSON dimensions and
    # the slot centres are consistent with the selected centre pitch.
    if config['ot2'].get('slot_size_mode') == 'plate_plus_1mm':
        flat = plate_flat_size_m(config['plate'])
        for axis in (0, 1):
            check(f'Slot plate+1mm size axis {axis}',
                  float(config['ot2']['slot_tile_size_m'][axis]),
                  float(flat[axis]) + 0.001, 1e-8)
    # All 11 position guides lie flat against the OT-2 base/deck surface.
    if config['ot2'].get('show_slot_outlines', True):
        actual_base = float(config['ot2']['base_thickness_m'])
        if config['ot2'].get('show_deck_slab', False):
            actual_base = ot2_deck_surface_z(config['ot2'])
        if config['ot2'].get('show_slot_tiles', False):
            actual_base = ot2_deck_surface_z(config['ot2']) + 0.0011
        for n in range(1,12):
            outlines=ot2body.findall(f'geom[@name="ot2_slot_outline_{n}_y_1"]')
            if len(outlines)!=1:
                raise AssertionError(f'Slot {n} outline missing')
            g=outlines[0]
            # Border is horizontal at +/- (depth/2 - edge/2), and
            # has half-extent exactly half the full slot width.
            half=floats(g.get('size'))
            check(f'Slot {n} visual X size', half[0]*2,
                  float(config['ot2']['slot_tile_size_m'][0]), 1e-6)
            z=floats(g.get('pos'))[2]
            ht=floats(g.get('size'))[2]
            check(f'OT-2 slot {n} contact base Z',z-ht,actual_base,1e-6)
        print('  PASS all 11 slot borders lie on their actual supporting surface')

    # Physical table top should match the actual auto-fit or manual dimensions.
    actual_center,actual_size=table_geometry(config)
    table_geom=world.find('geom[@name="table_top"]')
    if table_geom is None: raise AssertionError('Missing table geometry')
    for ax,(got,exp) in enumerate(zip(floats(table_geom.get('size'))[:2], actual_size)):
        check(f'Actual table full axis {ax}',got*2,exp,1e-6)
    for ax,(got,exp) in enumerate(zip(floats(table_geom.get('pos'))[:2], actual_center)):
        check(f'Actual table XY {ax}',got,exp,1e-6)
    tmod_yaw = quat_to_yaw(floats(temp.get('quat')))
    check('Temp module yaw delta (deg)',angle_delta(tmod_yaw,config['ot2']['temperature_yaw_deg']),0,0.0005)
    for key, geomname in [('temperature_base_yaw_deg','temperature_module_body'),('temperature_block_yaw_deg','temperature_aluminium_block')]:
        geom=temp.find(f'geom[@name="{geomname}"]')
        if geom is None or geom.get('quat') is None:
            raise AssertionError(f'{geomname} is missing quaternion rotation')
        yaw=quat_to_yaw(floats(geom.get('quat')))
        check(geomname+' yaw delta (deg)',angle_delta(yaw,config['ot2'][key]),0,0.0005)
    # Exact dimensions: user-editable JSON -> generated MJCF geom half-sizes.
    def assert_geom_size(scope, geom_name, full_m):
        gg=scope.find(f'geom[@name="{geom_name}"]')
        if gg is None:
            raise AssertionError('Missing geometry '+geom_name)
        if gg.get('size') is None:
            raise AssertionError('Missing size attribute: '+geom_name)
        actual=floats(gg.get('size'))
        if len(actual)!=len(full_m):
            raise AssertionError('Unexpected shape dimensions: '+geom_name)
        for axis,(got,expected_full) in enumerate(zip(actual,full_m)):
            check(f'{geom_name} half-size axis {axis} (m)',got,expected_full/2,1e-6)

    assert_geom_size(temp,'temperature_module_body',config['ot2']['temperature_base_size_m'])
    assert_geom_size(temp,'temperature_aluminium_block',config['ot2']['temperature_block_size_m'])
    platebody=world.find('body[@name="reaction_plate"]')
    if platebody is None:
        raise AssertionError('Missing reaction_plate')
    assert_geom_size(platebody,'reaction_plate_box',plate_flat_size_m(config['plate']))
    print(f'  PASS plate flat face={config["plate"]["flat_surface_axes"]}, measured XYZ={config["plate"]["size_m"]}, world-flat XYZ={plate_flat_size_m(config["plate"])}')
    # Pickup site must follow the temperature module's XY and Z offsets.
    pickup=world.find('site[@name="ot2_pickup"]')
    for axis,(got,want) in enumerate(zip(floats(pickup.get('pos')), targets(config)['ot2_pickup'])):
        check(f'ot2_pickup target axis {axis}',got,want,1e-6)
    reader=world.find('body[@name="versamax_reader"]')
    if reader is None:
        raise AssertionError('Missing VersaMax body')
    region=versamax_local_regions(config)
    tray=reader.find('geom[@name="versamax_sliding_tray"]')
    if tray is None:
        raise AssertionError('Sliding tray missing')
    dims=(*region['tray_size_xy'],config['versamax']['tray_size_m'][2])
    assert_geom_size(reader,'versamax_sliding_tray',dims)
    tt=floats(tray.get('pos'))
    for i,v in enumerate(region['tray_center_xy']):
        check(f'Tray local XY axis {i}',tt[i],v,1e-6)
    # Recess is a real removed top volume, not a black-painted patch.
    if reader.find('geom[@name="versamax_white_housing"]') is not None:
        raise AssertionError('Old uncut VersaMax solid housing still present')
    recess=reader.find('geom[@name="versamax_recess_floor"]')
    if recess is None:
        raise AssertionError('Missing open-top VersaMax recess')
    rc=floats(recess.get('pos'))
    for i,v in enumerate(region['opening_center_xy']):
        check(f'Recess XY axis {i}',rc[i],v,1e-6)
    # Open SW notch must be a real absence of upper housing, not a 4-wall tub.
    for forbidden in ('versamax_front_left_shell', 'versamax_front_right_shell',
                      'versamax_recess_inner_wall_0', 'versamax_recess_inner_wall_1',
                      'versamax_recess_south_wall', 'versamax_recess_west_wall',
                      'versamax_tray_front_lip'):
        if reader.find(f'geom[@name="{forbidden}"]') is not None:
            raise AssertionError(f'Forbidden old wall/part still present: {forbidden}')
    # Upper housing segments only exist when enough physical volume remains.
    # If the user intentionally makes a recess wider than the body, that
    # segment is omitted rather than rejecting the whole scene.
    bx0, by0, bh0 = map(float, config['versamax']['body_size_m'])
    rw0, rd0, rh0 = map(float, config['versamax']['recess_size_m'])
    shell_exists = rh0 > 0
    for required, should_exist in (
        ('versamax_north_shell', shell_exists and rd0 < by0),
        ('versamax_east_shell', shell_exists and rw0 < bx0),
        ('versamax_recess_north_face', shell_exists and rd0 < by0),
        ('versamax_recess_east_face', shell_exists and rw0 < bx0),
    ):
        if should_exist and reader.find(f'geom[@name="{required}"]') is None:
            raise AssertionError('Missing physically possible housing segment: '+required)
    bx,by,bh=map(float,config['versamax']['body_size_m'])
    rw,rd,rh=map(float,config['versamax']['recess_size_m'])
    floor_geom=reader.find('geom[@name="versamax_recess_floor"]')
    floor_pos=floats(floor_geom.get('pos'))
    check('Recess floor Z (m)',floor_pos[2],bh-rh+.0003,1e-6)
    check('Recess width (m)',floats(floor_geom.get('size'))[0]*2,rw,1e-6)
    check('Recess depth (m)',floats(floor_geom.get('size'))[1]*2,rd,1e-6)
    inset=float(config['versamax']['tray_retraction_m'])
    check('Tray south edge vs notch south edge (m)',
          tt[1]-float(config['versamax']['tray_size_m'][1])/2,
          -by/2+inset,1e-6)
    if abs(inset)<1e-9:
        print('  PASS fully extended tray exactly flush with SOUTH notch edge')
    # The user may make the tray larger than the recess, or move it above or
    # below the floor. Such overlaps are permitted during layout editing.
    print('  PASS tray has requested independent X/Y/Z dimensions (no fit constraints)')
    # Target matches tray centre, rotated by same yaw as the instrument.
    plate_targets=targets(config)
    load_site=world.find('site[@name="versamax_load"]')
    if load_site is None:
        raise AssertionError('Missing versamax_load site')
    for i,(v1,v2) in enumerate(zip(floats(load_site.get('pos')),plate_targets['versamax_load'])):
        check(f'versamax_load world XYZ axis {i}',v1,v2,1e-6)
    print('  PASS cutout/tray/plate/temperature dimensions match JSON')
    # All static scene objects should be attached to the parent device so their
    # rotations are inherited. A fixed plate prevents a misleading free-fall.
    plate = world.find('body[@name="reaction_plate"]')
    if plate is not None and config['plate'].get('preview_static_plate', True):
        if plate.find('freejoint') is not None:
            raise AssertionError('Static layout preview has unexpected freejoint plate')
    report = SCENE.with_name('lab_layout_targets.json')
    if report.exists():
        j=json.loads(report.read_text(encoding='utf-8'))
        for device in ('ot2','versamax'):
            check(f'Report {device} yaw diff',angle_delta(float(j[f'{device}_yaw_deg']), float(config[device]['yaw_deg'])),0,0.0005)
        print('  PASS targets JSON yaw matches layout JSON')
    else:
        print('  WARNING: targets report absent (was scene built?)')
    # If MuJoCo is available, read compiled body transforms, as an extra check.
    try:
        import mujoco
    except ImportError:
        print('  INFO: MuJoCo not installed here, XML quaternion checks passed.')
    else:
        model=mujoco.MjModel.from_xml_path(str(SCENE))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for key, name in [('ot2','ot2_enclosure'),('versamax','versamax_reader')]:
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid<0:
                raise AssertionError(name+' missing in compiled scene')
            angle=quat_to_yaw(data.xquat[bid])
            check('COMPILED '+key+' yaw diff',angle_delta(angle,config[key]['yaw_deg']),0,0.005)
            check('COMPILED '+key+' X',float(data.xpos[bid][0]),config[key]['position_xy'][0],1e-4)
            check('COMPILED '+key+' Y',float(data.xpos[bid][1]),config[key]['position_xy'][1],1e-4)
        print('  PASS compiled MuJoCo poses are consistent')
    print('\nALL CHECKS PASSED: JSON, XML and opening directions agree.')


if __name__ == '__main__':
    validate()
