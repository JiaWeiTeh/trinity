"""Fit the self-similar expansion exponents from a run's dictionary.jsonl.

Static review cannot tell you a dropped term is missing; a wrong slope can.
For a uniform ambient density and a slowly varying mechanical luminosity:

    energy-driven (Weaver)   R ~ t^(3/5)    v ~ t^(-2/5)
    momentum-driven          R ~ t^(1/2)    v ~ t^(-1/2)

`default.param` ships `dens_profile densPL` with `densPL_alpha 0`, i.e. uniform,
so those exponents are the correct expectation for a default run.

    python docs/dev/code-audit/harness/phase6_asymptotics.py outputs/<run>/dictionary.jsonl

Reports; does not assert. L_mech is not constant (the SPS tables evolve) and the
first segments of each phase are a transient, so the fit uses the **late half** of
each phase and prints the residual scatter next to the slope. A slope quoted
without its scatter is not evidence.
"""

import json
import math
import sys

EXPECTED = {
    "energy": {"R2": 3 / 5, "v2": -2 / 5},
    "implicit": {"R2": 3 / 5, "v2": -2 / 5},  # still energy-driven, cooling included
    "momentum": {"R2": 1 / 2, "v2": -1 / 2},
}


def loglog_slope(xs, ys):
    """Least-squares slope of ln y vs ln x, plus rms residual in dex."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 4:
        return None, None, len(pts)
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None, None, n
    slope = sxy / sxx
    inter = my - slope * mx
    rms = math.sqrt(sum((p[1] - (slope * p[0] + inter)) ** 2 for p in pts) / n) / math.log(10)
    return slope, rms, n


def main(path):
    with open(path) as fh:
        rows = [json.loads(line) for line in fh]
    by_phase = {}
    for r in rows:
        by_phase.setdefault(r.get("current_phase", "?"), []).append(r)

    print(f"# Asymptotic exponents — {path}\n")
    print(f"{len(rows)} snapshots, phases: {[f'{k}:{len(v)}' for k, v in by_phase.items()]}\n")
    print("| phase | qty | measured | expected | delta | rms [dex] | n (late half) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for phase, rs in by_phase.items():
        rs = sorted(rs, key=lambda r: r.get("t_now", 0))
        late = rs[len(rs) // 2:]  # drop the transient
        ts = [r.get("t_now") for r in late]
        for qty in ("R2", "v2"):
            ys = [r.get(qty) for r in late]
            if any(t is None or y is None for t, y in zip(ts, ys)):
                continue
            # v2 is negative during collapse; fit |v2| and note it
            vals = [abs(y) for y in ys] if qty == "v2" else ys
            slope, rms, n = loglog_slope(ts, vals)
            exp = EXPECTED.get(phase, {}).get(qty)
            if slope is None:
                print(f"| {phase} | {qty} | — | {exp} | — | — | {n} |")
                continue
            d = f"{slope - exp:+.3f}" if exp is not None else "—"
            e = f"{exp:+.3f}" if exp is not None else "—"
            print(f"| {phase} | {qty} | {slope:+.3f} | {e} | {d} | {rms:.4f} | {n} |")
    print("\nNotes: `v2` is fitted as |v2| so a collapse phase still yields a slope.")
    print("Expected exponents assume uniform density and slowly varying L_mech;")
    print("a large rms means the phase is not on the attractor and the slope is not")
    print("a meaningful test. Compare slopes across runs, not against the ideal alone.")


if __name__ == "__main__":
    main(sys.argv[1])
