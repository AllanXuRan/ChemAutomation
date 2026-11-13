"""Load Allan's workcell through his own loader, physics_runtime.load_physics_model."""
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHEM = REPO.parent   # this folder sits inside Allan's ChemAutomation
WINDOWS_MESHES = re.compile(r'file="D:/Chemautomation/robot_sim/xarm_description/')


def allan_modules():
    """Import Allan's modules from the submodule: (robot_kinematics, physics_runtime)."""
    if str(CHEM) not in sys.path:
        sys.path.insert(0, str(CHEM))
    import physics_runtime
    import robot_kinematics
    return robot_kinematics, physics_runtime


def load(options=None, edit=None):
    """(model, data, info, runtime, kin) for Allan's scene, with his contact physics and his
    starting pose (Q0 with a level flange), as his tests set it up. `options` overrides fields
    of the compiled model's `opt` (e.g. {"noslip_iterations": 10}); every override is a change
    to Allan's physics, so callers record it. `edit(root)` may add to the scene's XML tree before
    his loader compiles it (plate_bt adds plates and pads); it runs on a copy, never his file."""
    import xml.etree.ElementTree as ET

    import mujoco
    kin, pr = allan_modules()
    scene = Path(kin.SCENE)
    text = scene.read_text()
    if WINDOWS_MESHES.search(text):
        # the scene files at b8ca623 name their meshes by Allan's Windows paths; his branch
        # fix/relative-mesh-paths makes them relative (merged on this branch); this covers older commits
        text = WINDOWS_MESHES.sub(f'file="{CHEM.as_posix()}/xarm_description/', text)
    if edit is not None or text != scene.read_text():
        root = ET.fromstring(text)
        for asset in root.iter():
            if asset.get("file"):     # absolute, so the copy compiles from any folder
                asset.set("file", str((scene.parent / asset.get("file")).resolve()))
        if edit is not None:
            edit(root)
        tmp = Path(tempfile.mkdtemp()) / scene.name
        tmp.write_text(ET.tostring(root, encoding="unicode"))
        scene = tmp
    model = pr.load_physics_model(scene)
    for key, value in (options or {}).items():
        setattr(model.opt, key, type(getattr(model.opt, key))(value))
    data = mujoco.MjData(model)
    info = kin.arm_info(model)
    kin.set_angles(model, data, info, kin.level_flange(model, info, kin.Q0))
    runtime = pr.PhysicsRuntime(model, data, info)
    return model, data, info, runtime, kin
