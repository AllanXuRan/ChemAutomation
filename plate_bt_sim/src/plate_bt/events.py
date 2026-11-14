"""The scenario: what happens to each plate of a run. Every failure is a
physical change to the workcell, made where nothing overlaps, except the formulation anomaly,
which the detector placeholder reports (for a fallen tip, a tip is also left lying on the plate).

`start` (optional) sets the cell up before the first plate:

    storage: n, bin: n                     plates already on the storage and bin piles
    supply: stack | person                 the new plates start stacked on the new-plate area (the
                                           first plate on top), or a person puts each one down
                                           when it is needed (the default)

`plates`: one entry per plate, in order:

    name: <label for the trace and the page>
    new_plate: {dx: m, dy: m, dyaw: deg}   the new plate is off its taught pose (on the area, or on
                                           the plate below it in the supply stack)
    anomaly: bubble | tip                  the detector flags the formulation
    storage_plate_dyaw: deg               before this plate is stored, a person puts a plate on the
                                           storage pile, turned by this much
"""
from pathlib import Path

import yaml

KEYS = {"name", "new_plate", "anomaly", "storage_plate_dyaw"}
START_KEYS = {"storage", "bin", "supply"}


class Scenario:
    def __init__(self, plates, start=None):
        self.start = dict(start or {})
        if set(self.start) - START_KEYS or self.start.get("supply", "person") not in ("stack", "person"):
            raise ValueError(f"start: keys {sorted(START_KEYS)}, supply stack or person")
        for i, p in enumerate(plates):
            unknown = set(p) - KEYS
            if unknown:
                raise ValueError(f"plate {i + 1}: unknown keys {sorted(unknown)}")
            if p.get("anomaly") not in (None, "bubble", "tip"):
                raise ValueError(f"plate {i + 1}: anomaly must be bubble or tip")
        self.plates = [dict(p) for p in plates]

    @classmethod
    def load(cls, path):
        d = yaml.safe_load(Path(path).read_text())
        return cls(d["plates"], d.get("start"))

    def to_dict(self):
        return dict(start=self.start, plates=self.plates) if self.start else dict(plates=self.plates)

    @property
    def stacked(self):
        """Whether the new plates start stacked on the new-plate area."""
        return self.start.get("supply") == "stack"

    def __len__(self):
        return len(self.plates)

    def plate(self, plate_id):
        """The entry for plate number plate_id (1-based, as Allan's blackboard counts)."""
        return self.plates[plate_id - 1] if 1 <= plate_id <= len(self.plates) else {}

    @property
    def bodies(self):
        """Plates the simulation needs: one per workflow plate, those on the piles at the start,
        and those people add."""
        return (len(self.plates) + self.start.get("storage", 0) + self.start.get("bin", 0)
                + sum("storage_plate_dyaw" in p for p in self.plates))

    @property
    def tips(self):
        """Pipette tips the simulation needs: one per plate with a fallen tip."""
        return sum(p.get("anomaly") == "tip" for p in self.plates)

    @property
    def anomalies(self):
        return {i + 1: p["anomaly"] for i, p in enumerate(self.plates) if p.get("anomaly")}
