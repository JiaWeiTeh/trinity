#!/usr/bin/env python
"""Phase-6 probes TBL-01 (cooling-table age clamp) and W-3 (swallowed errors).

Both were left open by the first Phase-6 pass because each was framed as needing a
run that never materialised: TBL-01 wanted `t > 10` Myr, W-3 wanted a run that
actually emits "Bubble properties calculation failed". Neither *mechanism* needs a
run — only the *frequency* does. This settles the mechanisms and states the
frequency limit explicitly.

    python docs/dev/code-audit/harness/phase6_tbl01_w3.py

Writes: docs/dev/code-audit/data/phase6_tbl01_w3.csv
"""

import csv
import logging
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from trinity.cooling.non_CIE.read_cloudy import get_fileage, get_filename  # noqa: E402
from trinity.phase1b_energy_implicit import get_betadelta  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "phase6_tbl01_w3.csv"
COOLING = ROOT / "lib" / "default" / "opiate"

DEFAULT_STOP_T_MYR = 15.0  # registry.py:378


def tbl01():
    """Does get_filename silently clamp past the table's last age?"""
    ages = sorted({get_fileage(f) for f in os.listdir(COOLING) if f.endswith(".dat")})
    amax = max(ages)
    rows = []
    print("== TBL-01: non-CIE cooling-table age clamp ==")
    print(f"age grid [yr]: {[f'{a:.2e}' for a in ages]}")
    print(f"grid max     : {amax:.3e} yr = {amax/1e6:g} Myr")
    print(f"default stop_t: {DEFAULT_STOP_T_MYR:g} Myr  (registry.py:378)")
    print()
    for age in (5.0e6, 9.9e6, 1.0e7, 1.01e7, 1.5e7, 5.0e7):
        fn = get_filename(age, 1.0, True, str(COOLING))
        served = fn if isinstance(fn, str) else f"interp{[f[-16:-4] for f in fn]}"
        clamped = isinstance(fn, str) and age > amax
        print(f"  age={age:9.3e} yr -> {'CLAMPED  ' if clamped else '         '}{served}")
        rows.append(
            {
                "probe": "TBL-01",
                "case": f"age={age:.3e}yr",
                "result": served,
                "silent": "yes" if clamped else "-",
                "note": "returns the last-grid file, no warning, no exception" if clamped else "",
            }
        )
    frac = (DEFAULT_STOP_T_MYR - amax / 1e6) / DEFAULT_STOP_T_MYR
    print()
    print(f"  => on a default-length run the clamp covers t = {amax/1e6:g}..{DEFAULT_STOP_T_MYR:g} Myr")
    print(f"     = the last {100*frac:.0f} % of the run, with the cooling frozen at the {amax/1e6:g} Myr table.")
    rows.append(
        {
            "probe": "TBL-01",
            "case": "default-run exposure",
            "result": f"{amax/1e6:g}-{DEFAULT_STOP_T_MYR:g} Myr",
            "silent": "yes",
            "note": f"last {100*frac:.0f}% of a stop_t=15 Myr run uses the {amax/1e6:g} Myr cooling table",
        }
    )
    return rows


class _Boom(Exception):
    """Stand-in for whatever get_bubbleproperties_pure can raise."""


def w3():
    """Is an arbitrary exception swallowed into a constant residual plateau?"""
    print()
    print("== W-3: what survives a get_bubbleproperties_pure failure ==")
    rows = []

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    get_betadelta.logger.addHandler(handler)
    original = get_betadelta.get_bubbleproperties_pure

    def _raise(*_a, **_kw):
        raise _Boom("simulated cooling-table bounds error")

    try:
        get_betadelta.get_bubbleproperties_pure = _raise
        # params is only touched after the try block, so a bare dict suffices:
        # the failure path returns before any params access.
        out = get_betadelta.get_residual_pure(0.8, -0.1, {}, dMdt_guess=None)
    finally:
        get_betadelta.get_bubbleproperties_pure = original
        get_betadelta.logger.removeHandler(handler)

    levels = [logging.getLevelName(r.levelno) for r in records]
    print(f"  returned            : {out}")
    print(f"  exception propagated: no")
    print(f"  log records emitted : {levels}")
    for r in records:
        print(f"    {logging.getLevelName(r.levelno)}: {r.getMessage()}")
    print()
    print("  => the ONLY trace is a WARNING line in trinity.log. The residual becomes a")
    print("     constant (100.0, 100.0) plateau; nothing reaches dictionary.jsonl,")
    print("     metadata.json, SimulationEndCode, or the process exit code.")

    rows.append(
        {
            "probe": "W-3",
            "case": "get_bubbleproperties_pure raises",
            "result": str(out),
            "silent": "warning-only",
            "note": (
                f"bare `except Exception` at get_betadelta.py:437-439; returns the constant "
                f"(100.0, 100.0, None) plateau; log levels emitted: {levels}; "
                f"no exception propagates and no run-level channel records it"
            ),
        }
    )
    return rows


def main():
    rows = tbl01() + w3()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["probe", "case", "result", "silent", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
