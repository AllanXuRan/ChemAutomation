"""Shared readable trajectory notebook export."""
import json

def export_notebook(payload, path):
    """Export readable positions and embedded trajectory data without executing cells."""

    def markdown(text, identifier):
        return dict(cell_type='markdown', id=identifier, metadata={}, source=text.splitlines(keepends=True))

    def code(text, identifier):
        return dict(cell_type='code', id=identifier, metadata={}, source=text.splitlines(keepends=True), execution_count=None, outputs=[])
    cells = [markdown('# Verified four-plate lab workflow\n\nWorld coordinates relative to robot base, distances in metres. Source stack -> aluminium -> HOME -> reader -> HOME -> slot2 -> HOME; repeat for four plates.\nThe HOME stage uses the fixed initial joint pose. Waypoints include collision-avoidance points.\n', 'intro')]
    table = '| Stage | TCP X | TCP Y | TCP Z | Plate X | Plate Y | Plate Z | Held |\n'
    table += '|---|---:|---:|---:|---:|---:|---:|---|\n'
    for record in payload['stages']:
        values = record['tcp_xyz_world_m'] + record['plate_xyz']
        table += '| ' + record['stage'] + ' | ' + ' | '.join((f'{v:.6f}' for v in values)) + f" | {record['held']} |\n"
    cells.append(markdown(table, 'positions'))
    for index, record in enumerate(payload['stages']):
        cells.append(markdown('## ' + record['stage'] + '\n\n```json\n' + json.dumps(record, indent=2) + '\n```\n', f'waypoint-{index}'))
    cells.append(code('trajectory = ' + repr(payload) + '\nprint("Stages:", len(trajectory["stages"]))\n', 'embedded-data'))
    cells.append(code('import matplotlib.pyplot as plt\npoints = [s["tcp_xyz_world_m"] for s in trajectory["stages"]]\nfig = plt.figure(figsize=(8, 6))\nax = fig.add_subplot(111, projection="3d")\nax.plot(*zip(*points), marker="o")\nfor i, p in enumerate(points): ax.text(*p, str(i+1))\nax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)", title="TCP waypoints")\nplt.show()\n', 'plot'))
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Trajectory notebook: {path}', flush=True)
