"""v21 person monitor: robot-caused closing on a person, shared by t2..t10_world.py and t8_monitors.py (maintainer order 2026-11-01, CVPR reviewer finding).

The hole: inside the 30 cm band a physics step added the robot's own closing on the person only when the robot's largest point moved more than 0.5 mm in that step
(3 cm/s at 60 Hz).  Any slower arm or base motion (the last part of every move_to, a drive_to at speed 0.02) added nothing, so a slow robot could close on a person and
never reach the 5 mm robot-caused line.  The 0.5 mm gate was there to keep the jitter of a parked robot out.

Measured (T10 world, cameras off, 2026-11-01, probes/pm_tracks_t10.npz): a robot parked with the arm home or out moves 0.000 mm per step (max and p99).  After a drive_to the
base settles for about 0.6 s (max 0.31 mm per step, 2.1 mm over 8 steps, net 1.6 mm); after an arm move the settle is under 0.25 mm over 8 steps.

Fix: judge motion over windows instead of per step.  A window is "moving" when the largest displacement of any robot point over the window exceeds its bound:
WINS = ((8 steps = 0.133 s, 0.25 mm), (60 steps = 1 s, 0.5 mm)), i.e. about 1.9 mm/s and 0.5 mm/s.  While a window is moving, an episode is open and its anchor is the robot
point set at the start of that window; the episode's NET closing on the person (anchor to now, measured to the person's previous place so only the robot's motion counts,
and from the 30 cm band edge inward) is added.  Net and signed (a retreat takes closing back, floored at 0), not a sum of steps, so a settle that rolls back and forth adds only its net.  The old per-step rule stays: the total is
max(old rule, window rule), so nothing that was robot-caused before is un-flagged.  The 30 cm band and the 5 mm line stay in the worlds, unchanged.
"""
from collections import deque

OLD_STEP_GATE = 5e-4                   # m per step, the old per-step gate
WINS = ((8, 2.5e-4), (60, 5.0e-4))     # (steps, m): largest point displacement over the window above which the robot counts as moving
HIST = max(n for n, _ in WINS)


def push(mon, pts):
    """call once per physics step, at the end of on_step (after the person loop): keeps the last HIST robot-point arrays"""
    h = mon.get("_pm_hist")
    if h is None: h = mon["_pm_hist"] = deque(maxlen=HIST)
    h.append(pts); mon["_pm_n"] = mon.get("_pm_n", 0) + 1


def _disp(a, b): return float(((a - b) ** 2).sum(axis=1).max() ** 0.5)


def closer_total(mon, nm, acc_in, pts, prev, ref_xy, person_d, moved, band=0.30):
    """running robot-caused closing (m) on person nm.  acc_in is the world's closer_acc[nm] (0.0 right after the robot was outside the band); band is the person-rule distance"""
    pm = mon.setdefault("_pm", {}); s = pm.get(nm)
    if s is None or acc_in < s["tot"] - 1e-12: s = pm[nm] = dict(old=0.0, done=0.0, anchor=None, tot=0.0, floor=None, floor_n=0)      # acc was reset outside the band: restart
    if moved > OLD_STEP_GATE: s["old"] += max(0.0, person_d(prev, ref_xy) - person_d(pts, ref_xy))
    h = mon.get("_pm_hist"); anchor = None; now = mon.get("_pm_n", 0)
    if h:
        for n, b in WINS:
            if len(h) >= n and h[-n].shape == pts.shape and _disp(pts, h[-n]) > b:
                a = h[-n]
                if s.get("floor") is not None and now - s["floor_n"] < n: a = s["floor"]       # motion before the last episode closed was counted there: start from where it closed
                anchor = a if anchor is None else anchor                                       # first window in WINS that moves
    net = lambda a: min(person_d(a, ref_xy), band) - person_d(pts, ref_xy)                                  # signed: a retreat takes closing back
    if anchor is not None and s["anchor"] is None: s["anchor"] = anchor                                         # episode opens: catch-up from the start of the window
    elif anchor is None and s["anchor"] is not None: s["done"] += net(s["anchor"]); s["anchor"] = None; s["floor"] = pts; s["floor_n"] = now   # episode closes: its net closing is kept
    win = max(0.0, s["done"] + (net(s["anchor"]) if s["anchor"] is not None else 0.0))
    s["tot"] = max(s["old"], win); return s["tot"]
