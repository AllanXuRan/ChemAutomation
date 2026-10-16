"""Review and optionally archive obsolete root-level demo scripts safely.

Usage:
    python cleanup_project.py
    python cleanup_project.py --archive
    python cleanup_project.py --delete-legacy

Only two named legacy scripts are touched. Your important JSON, MJCF, URDF,
meshes, and all current generators are NEVER deleted or moved.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY = {
    'capture_lab_cameras.py': 'old three camera names; replaced by capture_three_cameras.py',
    'interactive_lab_viewer.py': 'old static Viewer; replaced for FK/IK by ik_control_panel.py',
}
OPTIONAL_DIAGNOSTICS = {
    'run_physics_test.py': 'only needed for physics regression tests',
    'check_slot_layout.py': 'only needed after slot layout edits',
    'verify_layout.py': 'only needed after geometry or layout edits',
}
CRITICAL = (
    'build_lab_scene.py', 'layout_editor.py', 'lab_layout.json',
    'prepare_physics_scene.py', 'physics_settings.json',
    'setup_tcp_cameras.py', 'camera_setup.json',
    'robot_kinematics.py', 'ik_control_panel.py',
    'camera_editor.py', 'capture_three_cameras.py',
)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--archive', action='store_true', help='Move old scripts into archive folder')
    mode.add_argument('--delete-legacy', action='store_true', help='Permanently delete two known legacy scripts after confirmation')
    ap.add_argument('--include-diagnostics', action='store_true',
                    help='With --archive, archive diagnostic scripts as well')
    args = ap.parse_args()

    print('\nCURRENT PROJECT FILES (KEEP):')
    for filename in CRITICAL:
        print(' ', 'OK' if (ROOT / filename).is_file() else 'MISSING', filename)
    print('  KEEP xarm_description/ (all robot assets + generated XML)')
    print('  KEEP camera_calibration_v15.json (generated snapshot, may be regenerated)')
    print('  KEEP last_ik_pose.json (if saved)')
    print('\nOLD FILES (CAN ARCHIVE/REMOVE):')
    for filename, description in LEGACY.items():
        if (ROOT / filename).exists():
            print(f'  {filename}  --  {description}')
    print('\nOPTIONAL DIAGNOSTICS (RECOMMENDED KEEP):')
    for filename, description in OPTIONAL_DIAGNOSTICS.items():
        if (ROOT / filename).exists():
            print(f'  {filename}  --  {description}')
    print('\nDO NOT delete any of the active .json files or any robot XML/mesh folders.')

    files = [name for name in LEGACY if (ROOT / name).is_file()]
    if args.archive and args.include_diagnostics:
        files.extend([name for name in OPTIONAL_DIAGNOSTICS if (ROOT / name).is_file()])
    if not args.archive and not args.delete_legacy:
        print('\nNo changes made. Use --archive to clean safely.')
        return
    if not files:
        print('Nothing to move/delete.')
        return
    if args.delete_legacy:
        print('\nDELETE PERMANENTLY:', ', '.join(files))
        if input('Type DELETE to confirm: ').strip() != 'DELETE':
            print('Cancelled. Nothing deleted.')
            return
        for filename in files:
            (ROOT / filename).unlink()
            print('DELETED', filename)
    else:
        dest = ROOT / 'archive' / ('legacy_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        dest.mkdir(parents=True, exist_ok=True)
        for filename in files:
            shutil.move(str(ROOT / filename), str(dest / filename))
            print('ARCHIVED', filename, '->', dest)


if __name__ == '__main__':
    main()
