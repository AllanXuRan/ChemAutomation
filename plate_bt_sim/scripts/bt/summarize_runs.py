"""A summary of plate-workflow runs, as one small JSON: for each run and plate, the outcome, the actions over time (merged into
segments, with the tree branch each belongs to), every check with its numbers, the scenario
events, the slip risk at 10 Hz with its full-rate maxima, and the collision audit.

    python scripts/bt/summarize_runs.py data/plate_bt_<ts>_run_a data/plate_bt_<ts>_run_b --out runs.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

RECOVERY = {"CheckQuarantineSafe", "MoveRejectedPlateToBin", "RepositionForGrasp", "RetractAndRealign",
            "RepositionForPlacement"}


def branch(action):
    return "preventive" if action == "StabiliseGrasp" else "recovery" if action in RECOVERY else "normal"


def summarize(run):
    st = np.load(run / "states.npz")
    tr = json.loads((run / "trace.json").read_text())
    s = tr["summary"]
    t, plate, names = st["time"], st["plate"], st["names"]
    action = names[st["action"]]
    out = dict(run=run.name, sim_s=s["sim_s"], wall_s=s["wall_s"], ticks=s["ticks"], collisions=s["collisions"],
               plates=[])
    for p, entry in enumerate(s["scenario"], start=1):
        idx = np.flatnonzero(plate == p)
        if not len(idx):
            continue
        segs, start = [], idx[0]
        for a, b in zip(idx, idx[1:].tolist() + [None]):
            if b is None or action[b] != action[a]:
                if action[a] not in ("idle",) and not action[a].startswith("FinishPlate"):
                    segs.append(dict(action=str(action[a]), branch=branch(str(action[a])),
                                     t0=round(float(t[start]), 2), t1=round(float(t[a]), 2)))
                start = b
        ev = [e for e in tr["events"] if e["plate"] == p and e["kind"] in
              ("check", "detector", "measured", "event", "cancel", "set down", "drop", "plan error", "reader")]
        keep = idx[::3]
        oc = next((o for o in s["outcomes"] if o["plate"] == p), {})
        risk = st["slip_risk"][idx]
        prevent = [g for g in segs if g["branch"] == "preventive"]
        after = risk[t[idx] > prevent[-1]["t1"]] if prevent else risk[:0]
        out["plates"].append(dict(plate=p, name=entry.get("name", ""), scenario=entry, outcome=oc,
                                  t0=round(float(t[idx[0]]), 2), t1=round(float(t[idx[-1]]), 2),
                                  segments=segs, events=ev,
                                  risk_max=round(float(risk.max()), 3),
                                  risk_max_after_preventive=round(float(after.max()), 3) if len(after) else None,
                                  risk=[[round(float(t[i]), 1), round(float(st["slip_risk"][i]), 3)] for i in keep],
                                  grip=[[round(float(t[i]), 1), round(float(st["grip_n"][i]), 2)] for i in keep]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps([summarize(r) for r in args.runs]))
    print(args.out, args.out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
