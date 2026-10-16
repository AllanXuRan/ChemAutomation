"""Check OT-2 slot sizes and centres without opening MuJoCo.

Run from the same folder as lab_layout.json:
    python check_slot_layout.py
"""
from build_lab_scene import (
    load_config, plate_flat_size_m, ot2_slot_xy,
    OT2_SLOT_PITCH_X_M, OT2_SLOT_PITCH_Y_M,
)


def main():
    c = load_config()
    p = plate_flat_size_m(c['plate'])
    s = c['ot2']['slot_tile_size_m']
    print('OT-2 slot size mode:', c['ot2']['slot_size_mode'])
    print(f'Plate footprint: {p[0]*1000:.2f} x {p[1]*1000:.2f} mm')
    print(f'Slot footprint:  {s[0]*1000:.2f} x {s[1]*1000:.2f} mm')
    print(f'Slot minus plate: {(s[0]-p[0])*1000:.2f} mm X, {(s[1]-p[1])*1000:.2f} mm Y')
    print(f'Centre pitch: X={OT2_SLOT_PITCH_X_M*1000:g} mm, '
          f'Y={OT2_SLOT_PITCH_Y_M*1000:g} mm')
    print('All eleven slot centres (metres, OT-2 local frame):')
    for slot in range(1,12):
        x,y=ot2_slot_xy(slot)
        print(f'  Slot {slot:2}:  x={x:+.4f}, y={y:+.4f}')
    if any(s[i]<p[i] for i in range(2)):
        print('NOTE: The chosen visual slot is smaller than the plate in at least one axis.')
    print('Geometry dimensions are visualization references; verify with actual Deck CAD for hardware.')


if __name__ == '__main__':
    main()
