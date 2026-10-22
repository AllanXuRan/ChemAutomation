"""Standalone aluminium-to-reader physical trajectory, returning through HOME.
Run python demo_aluminium_to_reader.py; use --headless to verify without a window.
"""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import robot_kinematics as kin
from physics_runtime import PhysicsRuntime, load_physics_model
ROOT=Path(__file__).resolve().parent
ALUMINIUM_START=[-0.21989629307768632, 0.6264389270180409, 0.16944554069909196]
ALUMINIUM_QUAT=[0.999999136366252, -3.6403435134225694e-06, 3.452937817059085e-07, -0.0013142501202288595]
PICK_ABOVE=[2.109634726332112, 0.8606461156245122, -2.1064139821820307, 1.2457678665575185, 2.0949961023928285]
PICK_CONTACT=[2.0773319337615637, 1.3003208763218734, -2.7326254979906373, 1.4323046216687638, 2.06269330982228]
HOME=np.array([1.5428710587630783, -1.0855947947390834, -1.0681415022228964, 2.1537362969619824, -0.03490658503987258])
PICK_POSES=np.array([[2.322652345189196, 1.2471184092033718, -2.6434371104329317, 1.396312263510416, 2.29330455224699], [2.320160944208125, 1.3364970279338222, -2.5924255670235254, 1.255928994237401, 2.3055331601350093], [2.3201609518269346, 1.3364702340030787, -2.5923969015367763, 1.25592691029341, 2.305522327887651], [2.32016095964124, 1.336467833571176, -2.592395572194718, 1.2559280058587858, 2.3055200992248643], [2.320160955654623, 1.336466742361526, -2.592394936286949, 1.2559284698624058, 2.305522912888908], [2.320160953691931, 1.3364663681106836, -2.5923947186388303, 1.2559286282267088, 2.3055236487419375], [2.327143702115575, 1.190550623505391, -2.678454705901613, 1.4879072043794095, 2.3265382133789685], [2.0760628294903234, 1.2156585852244288, -2.697769420401413, 1.4821047701464416, 2.0658427056261854], [2.0553138280691923, 1.2377097733763764, -2.627370835937667, 1.3896560405011307, 2.0663678405709427], [2.054659976002644, 1.2366099001292916, -2.627401258511631, 1.3907950550561692, 2.077433956393529]])

def export_notebook(payload,path):
    """Export readable positions and embedded trajectory data without executing cells."""
    def markdown(text,identifier):
        return dict(cell_type='markdown',id=identifier,metadata={},source=text.splitlines(keepends=True))
    def code(text,identifier):
        return dict(cell_type='code',id=identifier,metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[])
    cells=[markdown('# Verified trajectory: aluminium to HOME to reader\n\n'
                    'World coordinates relative to robot base, distances in metres. '
                    'Initial pose -> aluminium -> grasp -> lift -> HOME -> reader -> place -> release -> retract.\n'
                    'The HOME stage uses the fixed initial joint pose. Waypoints include collision-avoidance points.\n', 'intro')]
    table='| Stage | TCP X | TCP Y | TCP Z | Plate X | Plate Y | Plate Z | Held |\n'
    table+='|---|---:|---:|---:|---:|---:|---:|---|\n'
    for record in payload['stages']:
        values=record['tcp_xyz_world_m']+record['plate_xyz']
        table+='| '+record['stage']+' | '+' | '.join(f'{v:.6f}' for v in values)+f" | {record['held']} |\n"
    cells.append(markdown(table,'positions'))
    for index,record in enumerate(payload['stages']):
        cells.append(markdown('## '+record['stage']+'\n\n```json\n'+json.dumps(record,indent=2)+'\n```\n',f'waypoint-{index}'))
    cells.append(code('trajectory = '+repr(payload)+'\nprint("Stages:", len(trajectory["stages"]))\n','embedded-data'))
    cells.append(code('import matplotlib.pyplot as plt\n'
                      'points = [s["tcp_xyz_world_m"] for s in trajectory["stages"]]\n'
                      'fig = plt.figure(figsize=(8, 6))\nax = fig.add_subplot(111, projection="3d")\n'
                      'ax.plot(*zip(*points), marker="o")\n'
                      'for i, p in enumerate(points): ax.text(*p, str(i+1))\n'
                      'ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)", title="TCP waypoints")\n'
                      'plt.show()\n','plot'))
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                      'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Trajectory notebook: {path}',flush=True)

def run(headless=False, playback_speed=4.):
    model=load_physics_model(kin.SCENE)
    data=mujoco.MjData(model)
    info=kin.arm_info(model)
    tcp=kin.site_id(model,'g2_tcp')
    kin.set_angles(model,data,info,HOME)
    runtime=PhysicsRuntime(model,data,info)
    controller=runtime.controller
    carried=model.body('ot2_stack_plate_3_body').id
    address=model.jnt_qposadr[model.joint('ot2_stack_plate_3_free').id]
    data.qpos[address:address+7]=ALUMINIUM_START+ALUMINIUM_QUAT
    mujoco.mj_forward(model,data)
    controller.initial[:]=data.qpos
    records=[]
    context=nullcontext(None) if headless else mujoco.viewer.launch_passive(model,data)
    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:]=[-.2,.45,.22]
            viewer.cam.distance=1.6
            viewer.cam.azimuth=115
            viewer.cam.elevation=-28
        def step():
            if viewer is not None and not viewer.is_running():
                raise KeyboardInterrupt
            clock=time.perf_counter()
            runtime.advance(.02)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.,.02/playback_speed-(time.perf_counter()-clock)))
        def plan(start,goal):
            scratch=mujoco.MjData(model)
            scratch.qpos[:]=data.qpos
            transporting=controller.held
            tcp_rotation=data.site_xmat[tcp].reshape(3,3)
            relative_position=tcp_rotation.T@(data.xpos[carried]-data.site_xpos[tcp])
            relative_rotation=tcp_rotation.T@data.xmat[carried].reshape(3,3)
            plate_address=model.jnt_qposadr[model.body_jntadr[carried]]
            def safe(q):
                kin.set_angles(model,scratch,info,q)
                if transporting:
                    rotation=scratch.site_xmat[tcp].reshape(3,3)
                    scratch.qpos[plate_address:plate_address+3]=scratch.site_xpos[tcp]+rotation@relative_position
                    quat=np.empty(4)
                    mujoco.mju_mat2Quat(quat,(rotation@relative_rotation).ravel())
                    scratch.qpos[plate_address+3:plate_address+7]=quat
                    mujoco.mj_forward(model,scratch)
                return not any(depth>.0002 for depth in runtime.forbidden_contacts(scratch).values())
            def edge(a,b):
                count=max(2,int(np.max(np.abs(b-a))/.025)+1)
                return all(safe(a+(b-a)*t) for t in np.linspace(0,1,count))
            if not safe(goal):
                print('Goal contacts:',[(model.geom(a).name,model.geom(b).name,depth) for (a,b),depth in runtime.forbidden_contacts(scratch).items()],flush=True)
                raise RuntimeError('Destination configuration collides')
            if edge(start,goal):return [goal]
            rng=np.random.default_rng(42)
            nodes=[start.copy()];parents=[-1]
            low=np.minimum(start,goal)-[.7,.7,.7,.7,.7]
            high=np.maximum(start,goal)+[.7,.7,.7,.7,.7]
            for attempt in range(6000):
                sample=goal.copy() if rng.random()<.2 else rng.uniform(low,high)
                sample[3]=-sample[1]-sample[2]
                if controller.held:sample[4]=sample[0]-(start[0]-start[4])
                if not np.allclose(sample,kin.clip_angles(model,info,sample)):continue
                distances=[np.linalg.norm(n-sample) for n in nodes]
                nearest=int(np.argmin(distances));delta=sample-nodes[nearest]
                trial=nodes[nearest]+delta*min(1.,.2/max(np.linalg.norm(delta),1e-12))
                if not edge(nodes[nearest],trial):continue
                nodes.append(trial);parents.append(nearest)
                if not edge(trial,goal):continue
                path=[goal];index=len(nodes)-1
                while index>=0:path.append(nodes[index]);index=parents[index]
                path.reverse()
                result=[];index=0
                while index<len(path)-1:
                    end=len(path)-1
                    while end>index+1 and not edge(path[index],path[end]):end-=1
                    result.append(path[end]);index=end
                return result
            raise RuntimeError('No collision-free joint path found')

        def move(name,q,grip=None,seconds=5.,held=False,planned=False):
            if not planned:
                route=plan(runtime.arm_target.copy(),q)
                for index,waypoint in enumerate(route):
                    final=index==len(route)-1
                    move(name+(f' via {index+1}/{len(route)}' if len(route)>1 else ''),
                         waypoint,grip if final else None,
                         max(seconds/len(route),float(np.max(np.abs(waypoint-runtime.arm_target)))/(.09 if held else .2)),
                         held,True)
                return

            print(name,flush=True)
            start=runtime.arm_target.copy()
            fingers=controller.target.copy()
            target=fingers if grip is None else np.asarray(grip)
            for frame in range(int((seconds+1.)/.02)):
                t=min(1.,(frame+1)*.02/seconds)
                u=t*t*t*(10-15*t+6*t*t)
                runtime.set_arm_target(start+u*(q-start))
                controller.target[:]=fingers+u*(target-fingers)
                data.ctrl[controller.act]=controller.target
                step()
                if runtime.blocked:
                    raise RuntimeError(name+': collision '+runtime.blocked)
            error=float(np.max(np.abs(kin.angles(data,info)-q)))
            if error>.025:
                raise RuntimeError(name+': joint tracking error '+str(error))
            if held and not controller.held:
                print('Failed plate position:',data.xpos[carried], 'TCP:',data.site_xpos[tcp],flush=True)
                raise RuntimeError(name+': grasp lost')
            result=dict(stage=name,held=bool(controller.held),plate_xyz=data.xpos[carried].tolist(),
                        joint_angles_rad=q.tolist(),joint_angles_deg=np.degrees(q).tolist(),
                        actual_joint_angles_rad=kin.angles(data,info).tolist(),
                        tcp_xyz_world_m=data.site_xpos[tcp].tolist(),
                        tcp_R_world=data.site_xmat[tcp].reshape(3,3).tolist(),
                        plate_quat_wxyz=data.xquat[carried].tolist(),
                        gripper_target_m=target.tolist(),
                        gripper_actual_m=data.qpos[controller.finger_q].tolist(),duration_s=seconds)
            records.append(result)
            print(json.dumps(result),flush=True)
        def solve(xyz,yaw=None):
            if yaw is None:yaw=grasp_yaw
            scratch=mujoco.MjData(model)
            scratch.qpos[:]=data.qpos
            q,error,_=kin.solve_ik(model,scratch,info,tcp,xyz,.00005,level=True,flange_yaw=yaw)
            if error>.001:
                raise RuntimeError('Unreachable waypoint: '+str(xyz)+' error='+str(error))
            return q
        move('01 initial',HOME,[-.04,.04],seconds=1.)
        move('02 approach plate on aluminium',np.array(PICK_ABOVE),seconds=10.)
        move('03 descend to aluminium plate',np.array(PICK_CONTACT),seconds=8.)
        move('04 grasp aluminium plate',runtime.arm_target.copy(),[0.,0.],seconds=3.,held=True)
        grasp_yaw=runtime.arm_target[0]-runtime.arm_target[4]
        offset=data.xpos[carried]-data.site_xpos[tcp]
        retreat=data.site_xpos[tcp].copy()+[0,-.045,.055]
        move('05 lift clear of aluminium',solve(retreat),seconds=8.,held=True)
        move('06 clear transit',solve(np.array([-.24,.35,.30])),seconds=12.,held=True)
        move('07 return to initial pose with plate',HOME,seconds=30.,held=True)
        grasp_yaw=HOME[0]-HOME[4]
        target=data.site_xpos[kin.site_id(model,'versamax_load')].copy()
        offset=data.xpos[carried]-data.site_xpos[tcp]
        landing=target-offset
        above=landing+[0,-.045,.10]
        move('08 approach reader',solve(above),seconds=12.,held=True)
        move('09 lower above reader tray guides',solve(landing+[0,0,.012]),seconds=10.,held=True)
        move('10 release on reader',runtime.arm_target.copy(),[-.04,.04],seconds=3.)
        move('11 retract from reader',solve(above),seconds=6.)
        delta=data.xpos[carried]-target
        if np.linalg.norm(delta[:2])>.025 or abs(delta[2])>.01:
            raise RuntimeError('Reader placement outside target: '+str(delta))
        report=ROOT/'outputs'/'reader_verified_trajectory.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        payload={'destination':'versamax_reader_tray','coordinate_frame':'world (robot base)',
                 'distance_unit':'m','grasp_flange_yaw_rad':grasp_yaw,
                 'placement_error_m':delta.tolist(),'stages':records}
        report.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        export_notebook(payload,ROOT/'trajectory_notes'/'02_aluminium_home_reader.ipynb')
        note=ROOT/'trajectory_notes'/'02_aluminium_home_reader.md'
        rows=['# Verified trajectory: aluminium -> HOME -> reader', '',
              '| Stage | TCP X | TCP Y | TCP Z | Held |', '|---|---:|---:|---:|---|']
        for record in records:
            rows.append('| '+record['stage']+' | '+' | '.join(f'{v:.6f}' for v in record['tcp_xyz_world_m'])+' | '+str(record['held'])+' |')
        rows+=['', 'Placement error (m): '+str(delta.tolist()), 'See the notebook for joint angles, plate poses and gripper states.']
        note.write_text('\n'.join(rows)+'\n',encoding='utf-8')
        print('PASS: aluminium -> grasp -> HOME -> reader -> release.',flush=True)
        print(f'Validated joint trajectory: {report}',flush=True)
        if viewer is not None:
            print('Complete; close window to exit.',flush=True)
            while viewer.is_running():step()
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--speed',type=float,default=4.,help='Viewer playback speed; physics timestep is unchanged')
    args=parser.parse_args()
    if not np.isfinite(args.speed) or args.speed<=0:parser.error('Speed must be positive')
    try:run(args.headless,args.speed)
    except KeyboardInterrupt:pass
