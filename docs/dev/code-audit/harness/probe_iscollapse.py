#!/usr/bin/env python
"""S11-R-02 probe: does `isCollapse` classify terminating events correctly?

`apply_event_result` (`trinity/phase_general/phase_events.py:627`) decides whether a
run ended in collapse with a **substring test on `reason_code`**:

    if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():
        params['isCollapse'].value = True

This probe drives every simulation-ending event factory through the real
`check_event_termination` -> `apply_event_result` path and prints the resulting flag
against the physically correct answer, so the classification is read off the code
rather than argued about.

Run:
    python docs/dev/code-audit/harness/probe_iscollapse.py

Writes: docs/dev/code-audit/data/iscollapse_truth_table.csv
"""

import csv
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from trinity.phase_general import phase_events as events  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "iscollapse_truth_table.csv"

# (label, factory, y_at_event, physically_a_collapse?, why)
# y = [R2, v2]; v2 in pc/Myr. The "expected" column is the physical fate a reader of
# `isCollapse` would need: the shell ended up collapsing inward.
CASES = [
    (
        "min_radius",
        lambda: events.make_min_radius_event(1.0),
        [1.0, -2.0],
        True,
        "R2 fell through the floor while contracting - a genuine collapse",
    ),
    (
        "max_radius",
        lambda: events.make_max_radius_event(50.0),
        [50.0, +12.0],
        False,
        "R2 crossed stop_r from BELOW with v2 > 0 - the shell is expanding",
    ),
    (
        "velocity_runaway(collapse)",
        lambda: events.make_velocity_runaway_event(500.0, direction="collapse"),
        [3.0, -500.0],
        True,
        "v2 <= -500 pc/Myr - runaway INFALL, the definition of collapse",
    ),
    (
        "velocity_runaway(expansion)",
        lambda: events.make_velocity_runaway_event(500.0, direction="expansion"),
        [30.0, +500.0],
        False,
        "v2 >= +500 pc/Myr - runaway expansion",
    ),
]


def _param(value=None):
    return SimpleNamespace(value=value)


def drive(factory, y):
    """Run one event through the real check -> apply path; return the params dict."""
    event = factory()
    sol = SimpleNamespace(
        t_events=[np.array([0.25])],
        y_events=[np.array([y], dtype=float)],
    )
    result = events.check_event_termination(sol, [event])
    assert result.triggered, "probe error: event did not register as triggered"
    params = {
        "t_now": _param(),
        "R2": _param(),
        "v2": _param(),
        "SimulationEndReason": _param(""),
        "SimulationEndCode": _param(),
        "EndSimulationDirectly": _param(False),
        "isCollapse": _param(False),
    }
    events.apply_event_result(params, result, result.t, result.y)
    return event, result, params


def main():
    rows = []
    for label, factory, y, expected, why in CASES:
        event, result, params = drive(factory, y)
        actual = bool(params["isCollapse"].value)
        rows.append(
            {
                "event": label,
                "reason_code": result.reason_code,
                "end_code": getattr(event.end_code, "name", ""),
                "v2_at_event": y[1],
                "isCollapse_actual": actual,
                "isCollapse_expected": expected,
                "verdict": "OK" if actual == expected else (
                    "FALSE NEGATIVE" if expected else "FALSE POSITIVE"
                ),
                "why": why,
            }
        )

    w = max(len(r["event"]) for r in rows)
    print(f"{'event':<{w}}  {'reason_code':<24} {'v2':>8}  {'got':>5} {'want':>5}  verdict")
    print("-" * (w + 60))
    for r in rows:
        print(
            f"{r['event']:<{w}}  {r['reason_code']:<24} {r['v2_at_event']:>8.1f}  "
            f"{str(r['isCollapse_actual']):>5} {str(r['isCollapse_expected']):>5}  {r['verdict']}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT.relative_to(pathlib.Path(__file__).resolve().parents[3])}")

    bad = [r for r in rows if r["verdict"] != "OK"]
    print(f"{len(bad)} of {len(rows)} simulation-ending events misclassified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
