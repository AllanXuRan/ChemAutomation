"""Allan's behaviour tree with the new-plate steps in front.

Allan's `build_tree` is used as it is. His one-plate sequence starts at
WaitForFormulationComplete; Kevin's workflow first takes a new plate from the new-plate area and
puts it on the heater, so five actions go at the head of that sequence. Their failures use his
fault kinds, so his recovery branches handle them: a failed grasp of the new plate runs
RepositionForGrasp, a failed placement on the heater runs RepositionForPlacement."""
import sys

from mujoco_bridge.load import CHEM


def allan_bt():
    """Allan's lab_behavior_tree module, imported from the submodule."""
    path = str(CHEM / "chem_lab_bt")
    if path not in sys.path:
        sys.path.insert(0, path)
    import lab_behavior_tree
    return lab_behavior_tree


def find(node, name):
    if node.name == name:
        return node
    for child in getattr(node, "children", []):
        hit = find(child, name)
        if hit is not None:
            return hit
    return None


def build(state, backend, settings):
    lbt = allan_bt()
    tree = lbt.build_tree(state, backend, settings)
    main = find(tree.root, "04_NORMAL_ONE_PLATE")

    def a(name, **kw):
        return lbt.AsyncAction(name, state, backend, **kw)

    def carrying(flag):
        return lambda: setattr(state, "carrying_plate", flag)

    new_plate = [
        a("FetchNewPlate"),
        a("GraspNewPlate", fault_on_failure="grasp"),
        a("VerifyNewPlateGrasp", fault_on_failure="grasp", on_success=carrying(True)),
        a("PlaceOnHeater", fault_on_failure="placement", on_success=carrying(False)),
        a("VerifyOnHeater", fault_on_failure="placement"),
    ]
    for i, node in enumerate(new_plate):
        main.insert_child(node, i)
    return tree
