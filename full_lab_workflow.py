"""Compatibility entry point: the existing behavior-tree mainloop owns the mission.
Run python full_lab_workflow.py --headless or --speed 8.
"""
import sys
from chem_lab_bt.lab_behavior_tree import main
from trajectory_recording import export_notebook

def run(headless=False,playback_speed=4.,plate_limit=4):
    args=['--backend','mujoco','--plates',str(plate_limit),'--speed',str(playback_speed)]
    if headless:args.append('--headless')
    result=main(args)
    if not result['success']:raise RuntimeError('Behavior-tree mission did not complete')
    return result['records']

if __name__=='__main__':
    result=main(['--backend','mujoco',*sys.argv[1:]])
    if not result['success']:raise SystemExit(1)
