"""Run the plate workflow in Allan's MuJoCo workcell with no viewer.

    MUJOCO_GL=egl python scripts/bt/run_plates.py [--scenario configs/plate_bt_run.yaml] [--plates N]

Writes data/plate_bt_<timestamp>/: states.npz and trace.json (src/plate_bt/runner.py), and
run.log (Allan's tree's own messages and ours)."""
import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from plate_bt import runner  # noqa: E402
from plate_bt.events import Scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "plate_bt.yaml")
    ap.add_argument("--scenario", type=Path, default=REPO / "configs" / "plate_bt_run.yaml")
    ap.add_argument("--plates", type=int, help="run only the first N plates of the scenario")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-sim-s", type=float, default=3000.0, help="stop the run after this much simulated time")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    scenario = Scenario.load(args.scenario)
    if args.plates:
        scenario = Scenario(scenario.plates[:args.plates], scenario.start)
    out = args.out or REPO / "data" / f"plate_bt_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary, _ = runner.run(cfg, scenario, out, args.max_sim_s, log=lambda s: print(s, file=sys.stderr))
    (out / "run.log").write_text(buf.getvalue())
    (out / "config.yaml").write_text(args.config.read_text())
    (out / "scenario.yaml").write_text(yaml.safe_dump(scenario.to_dict(), sort_keys=False))
    print(json.dumps({k: summary[k] for k in ("sim_s", "wall_s", "outcomes", "collisions")}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
