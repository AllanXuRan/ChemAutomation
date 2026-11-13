"""The checks behind each failure in Xu's diagram. Each takes numbers the
robot can measure (poses from Perception, contact forces) and returns a Check: whether it passed
and the values it was decided on, which go into the run's trace."""
from dataclasses import dataclass, field

import numpy as np

from plate_bt.world import wrap


@dataclass
class Check:
    name: str
    ok: bool
    values: dict = field(default_factory=dict)


def pose_error(a, b):
    """(horizontal distance in m, yaw difference in degrees) between two plate poses. A plate
    turned by 180 degrees fits the same footprint, so yaw is compared modulo 180."""
    dxy = float(np.hypot(*(a.xyz[:2] - b.xyz[:2])))
    dyaw = float(np.degrees(abs(wrap(2 * (a.yaw - b.yaw)) / 2)))
    return dxy, dyaw


def grasp_precheck(seen, planned, tol):
    """Cannot grasp: the plate is not where the grasp was planned."""
    dxy, dyaw = pose_error(seen, planned)
    return Check("grasp pre-check", dxy <= tol["grasp_xy_m"] and dyaw <= tol["grasp_yaw_deg"],
                 dict(dxy_mm=round(dxy * 1000, 2), dyaw_deg=round(dyaw, 2)))


def lift_check(held, dz_plate, dz_tcp, tilt, tol):
    """Unstable grasp: after the test lift both pads press, the plate rose with the TCP, and it
    hangs level."""
    lag = abs(dz_plate - dz_tcp)
    return Check("test lift", held and lag <= tol["lift_follow_m"] and tilt <= tol["held_tilt_deg"],
                 dict(held=bool(held), lag_mm=round(lag * 1000, 2), tilt_deg=round(tilt, 2)))


def slip_risk(contacts):
    """Slipping risk: the share of the available friction the pads use to hold the plate,
    sum |f_t| / sum mu f_n over the pad contacts ((normal, tangential, mu) each). At 1 the plate
    slides; a dry plate at a 2 N grip sits near 0.1, a wet one (mu 0.15) near 0.65 before it
    accelerates. The placeholder for a tactile sensor; 0 without contact."""
    cap = sum(mu * fn for fn, ft, mu in contacts if fn > 0)
    return float(min(1.5, sum(ft for fn, ft, mu in contacts if fn > 0) / cap)) if cap > 1e-6 else 0.0


def aligned(plate, pocket, tol):
    """Misaligned: the held plate is not over the tray pocket."""
    dxy, dyaw = pose_error(plate, pocket)
    return Check("alignment", dxy <= tol["align_xy_m"] and dyaw <= tol["align_yaw_deg"],
                 dict(dxy_mm=round(dxy * 1000, 2), dyaw_deg=round(dyaw, 2)))


def placed(plate, tilt, seat, tol, name="placement"):
    """Cannot place: after release the plate does not rest on its seat, level and in line with it."""
    dxy, dyaw = pose_error(plate, seat)
    dz = float(plate.xyz[2] - seat.xyz[2])
    ok = dxy <= tol["place_xy_m"] and dyaw <= tol["place_yaw_deg"] and tilt <= tol["place_tilt_deg"] and abs(dz) < 0.003
    return Check(name, ok, dict(dxy_mm=round(dxy * 1000, 2), dyaw_deg=round(dyaw, 2),
                                tilt_deg=round(tilt, 2), dz_mm=round(dz * 1000, 2)))


def clear_of(points, keep_out, name="clear"):
    """Whether every point lies outside a box given as x and y ranges; the values are each point's
    distance outside it (negative: inside), in mm."""
    (x0, x1), (y0, y1) = keep_out
    outside = [max(x0 - p[0], p[0] - x1, y0 - p[1], p[1] - y1) for p in points]
    return Check(name, bool(min(outside) > 0), dict(outside_mm=[round(float(d) * 1000, 1) for d in outside]))
