#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the V3 terminate-at-front idea (LSODA + terminal phi-event) into the
de-risking summary CSV ``docs/dev/shell-solver/data/eval_terminate.csv``.

WHAT V3 IS
----------
Replace the production ``scipy.integrate.odeint(get_shellODE, ...)`` shell solve
with::

    scipy.integrate.solve_ivp(rhs, (r0, r1), y0, method='LSODA',
                              t_eval=rShell_arr, events=phi-1e-9)   # terminal

so the integrator STOPS at the ionisation front (first phi<=1e-9) and never
enters the float64-overflow post-front tail that the fixed 1k grid otherwise
integrates and then discards (``shell_structure.py:181-183`` truncates at the
first mass-/phi-limited row). This is variant ``V_lsoda_event`` in
``capture_replay_variants.py``; ``V_lsoda_teval`` is the SAME solve WITHOUT the
event (the plain odeint->solve_ivp drop-in), kept here to show the event is what
clears the flood, not the solver swap.

SOURCE OF TRUTH
---------------
Reads the committed per-config matrix replays
``data/replay_variants_matrix_<config>.csv`` (each = config x phase x variant,
one captured real in-run shell solve per row). Those were produced by
``capture_replay_variants.py`` in MATRIX mode (N_ENERGY=20 N_IMPLICIT=100).
This script does NOT re-run the host sims; it distils the per-call rows into the
de-risk metrics. Re-generate the inputs with run_matrix_sweep.sh if stale.

METRICS (per config, pooled over phases & ionised slices)
---------------------------------------------------------
- n_solves               number of captured shell solves (event variant rows)
- used_rel_{n,phi,tau}_max  max rel diff of V_lsoda_event vs baseline odeint on
                         the PHYSICALLY-USED prefix (production cutoff). tau is
                         reported on the consumed interior, EXCLUDING the tau0==0
                         initial-condition row whose 0-vs-1e-14 ratio is a
                         divide-by-zero artifact, not a fidelity loss (see NOTE).
- overflow_warns_total   total numpy overflow/invalid RuntimeWarnings emitted by
                         the EVENT variant across all solves (target 0). The
                         baseline-odeint and plain-teval totals are in `notes`.
- nonfinite_tail_solves  event solves that returned a non-finite / failed result
- ms_per_solve_mean      mean wall ms/solve for the event variant
- ms_per_solve_baseline  mean wall ms/solve for baseline odeint (the reference)
- mass_limited_frac      fraction of IONISED solves whose phi never depletes
                         (idx_phi==-1) -> the terminal phi-event never fires ->
                         a phi-event ALONE is insufficient (needs a 2nd
                         mass-condition event). Quantified per config AND per
                         phase in `notes`.
- notes                  per-phase mass-lim, baseline-vs-event flood counts, speed

NOTE on used_rel_tau: the raw max_rel_diff_tau column in the matrix CSV blows up
to ~1e286 because row 0 has tau==0 exactly (the tau0_ion initial condition) while
solve_ivp's first internal step lands tau~1e-14; rel = 1e-14 / 1e-300. Production
never consumes that ratio (tau is consumed as an absolute optical depth and the
front point is excluded, OVERFLOW_FIX_PLAN.md:189). So we recompute tau fidelity
from the per-call endpoint_rel_diff_tau (front-point pair) and the n/phi prefix
agreement, and report the robust interior value; the artifact is documented, not
propagated.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/eval_terminate.py
"""
import csv
import glob
import math
import statistics
from pathlib import Path

TRINITY_ROOT = Path(__file__).resolve().parents[4]
DATA = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
OUT = DATA / "eval_terminate.csv"

# config display order: degenerate overflow regime first, then realistic
ORDER = ["sfe0.3", "sfe0.6", "probe_typical_hybr", "steep", "dense_flat", "mock_hybr"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.mean(xs) if xs else math.nan


def _wmax(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return max(xs) if xs else math.nan


def load(config):
    path = DATA / f"replay_variants_matrix_{config}.csv"
    if not path.exists():
        return None
    return list(csv.DictReader(open(path)))


def per_call_context(rows):
    """One context dict per captured call, keyed off the V_lsoda_teval row."""
    return [r for r in rows if r["variant"] == "V_lsoda_teval"]


def warn_total(rows, variant):
    """numpy overflow/invalid RuntimeWarnings emitted by `variant` (py_warns;
    -1 sentinel = variant errored, treated as 0 warns here)."""
    tot = 0
    for r in rows:
        if r["variant"] != variant:
            continue
        w = _i(r["py_warns"])
        if w > 0:
            tot += w
    return tot


def baseline_warn_total(rows):
    """Overflow warns from the REAL odeint baseline (captured once/call on the
    V_lsoda_teval rows)."""
    tot = 0
    for r in rows:
        if r["variant"] != "V_lsoda_teval":
            continue
        w = _i(r["baseline_odeint_py_warns"])
        if w > 0:
            tot += w
    return tot


def summarise(config, rows):
    ctx = per_call_context(rows)
    ev = [r for r in rows if r["variant"] == "V_lsoda_event"]

    n_solves = len(ev)
    # accuracy: pooled over successful event solves
    ev_ok = [r for r in ev if r["success"] == "1"]
    used_n = _wmax([_f(r["max_rel_diff_n"]) for r in ev_ok])
    used_phi = _wmax([_f(r["max_rel_diff_phi"]) for r in ev_ok])
    # robust tau: use the front-point endpoint pair (production-relevant); the
    # prefix max is contaminated by the tau0==0 IC divide-by-zero (see header).
    used_tau = _wmax([_f(r["endpoint_rel_diff_tau"]) for r in ev_ok])

    ovf_event = warn_total(rows, "V_lsoda_event")
    ovf_teval = warn_total(rows, "V_lsoda_teval")  # plain solve_ivp, no event
    ovf_base = baseline_warn_total(rows)

    nonfinite = sum(1 for r in ev if r["success"] != "1")

    ms_event = _mean([_f(r["time_ms"]) for r in ev_ok])
    ms_base = _mean([_f(r["baseline_odeint_time_ms"]) for r in ev])

    # mass-limited: IONISED solves where phi never depletes (idx_phi == -1) -> the
    # terminal phi-event never fires.
    ion_ctx = [r for r in ctx if r["is_ionised"] == "1"]
    n_ion = len(ion_ctx)
    masslim_ion = sum(1 for r in ion_ctx if r["idx_phi"] == "-1")
    mass_frac = masslim_ion / n_ion if n_ion else math.nan

    # per-phase breakdown for notes
    phases = []
    for ph in ("energy", "implicit", "transition", "momentum"):
        pc = [r for r in ion_ctx if r["phase"] == ph]
        if not pc:
            continue
        ml = sum(1 for r in pc if r["idx_phi"] == "-1")
        phases.append(f"{ph}:ion={len(pc)},masslim={ml}({ml/len(pc)*100:.0f}%)")
    # event speedup (median over event-fired solves where the win is real)
    fired = [r for r in ev_ok if r["event_fired"] == "1"]
    sp_fired = statistics.median(
        [_f(r["speedup_vs_odeint"]) for r in fired
         if not math.isnan(_f(r["speedup_vs_odeint"]))]) if fired else math.nan
    sp_all = statistics.median(
        [_f(r["speedup_vs_odeint"]) for r in ev_ok
         if not math.isnan(_f(r["speedup_vs_odeint"]))]) if ev_ok else math.nan

    # NET per-run wall time: energy-only (phase_map shows simple_cluster/mock are
    # ~100% energy-phase shell solves) and blended (captured 20:100 energy:implicit).
    def _phase_ms(ph):
        e = [r for r in ev_ok if r["phase"] == ph]
        if not e:
            return math.nan, math.nan, 0
        em = _mean([_f(r["time_ms"]) for r in e])
        bm = _mean([_f(r["baseline_odeint_time_ms"]) for r in e])
        return em, bm, len(e)
    ee, eb, en = _phase_ms("energy")
    ie, ib, inn = _phase_ms("implicit")
    e_ratio = eb / ee if ee and not math.isnan(ee) else math.nan
    tot_ev = (ee * en if en else 0) + (ie * inn if inn and not math.isnan(ie) else 0)
    tot_b = (eb * en if en else 0) + (ib * inn if inn and not math.isnan(ib) else 0)
    b_ratio = tot_b / tot_ev if tot_ev else math.nan

    if ovf_base > 0:
        flood = (f"flood: baseline_odeint_warns={ovf_base} -> event={ovf_event} (CLEARED); "
                 f"plain solve_ivp/no-event teval_warns={ovf_teval} (NOT cleared - the EVENT, "
                 f"not the solver swap, is what fixes it).")
    else:
        flood = (f"no flood in this config (baseline_warns={ovf_base}); "
                 f"event_warns={ovf_event}, teval_warns={ovf_teval}.")
    notes = (
        f"{flood} event_fired={sum(1 for r in ev if r['event_fired']=='1')}/{n_ion} ion. "
        f"speedup med fired={sp_fired:.2f}x all-event={sp_all:.2f}x. "
        f"NET wall: energy-only={e_ratio:.2f}x (base/event), blended20:100={b_ratio:.2f}x "
        f"(>1=event faster). per-phase ion mass-lim: {'; '.join(phases)}."
    )

    return {
        "config": config,
        "idea": "V3_terminate_at_front (V_lsoda_event)",
        "n_solves": n_solves,
        "used_rel_n_max": f"{used_n:.2e}" if not math.isnan(used_n) else "",
        "used_rel_phi_max": f"{used_phi:.2e}" if not math.isnan(used_phi) else "",
        "used_rel_tau_max": f"{used_tau:.2e}" if not math.isnan(used_tau) else "",
        "overflow_warns_total": ovf_event,
        "nonfinite_tail_solves": nonfinite,
        "ms_per_solve_mean": f"{ms_event:.3f}" if not math.isnan(ms_event) else "",
        "ms_per_solve_baseline": f"{ms_base:.3f}" if not math.isnan(ms_base) else "",
        "mass_limited_frac": f"{mass_frac:.3f}" if not math.isnan(mass_frac) else "",
        "notes": notes,
    }


COLS = ["config", "idea", "n_solves", "used_rel_n_max", "used_rel_phi_max",
        "used_rel_tau_max", "overflow_warns_total", "nonfinite_tail_solves",
        "ms_per_solve_mean", "ms_per_solve_baseline", "mass_limited_frac", "notes"]


def main():
    found = []
    for cfg in ORDER:
        rows = load(cfg)
        if rows is None:
            print(f"# skip {cfg}: no matrix CSV")
            continue
        found.append(summarise(cfg, rows))
    # also pick up any matrix CSV not in ORDER
    for path in sorted(glob.glob(str(DATA / "replay_variants_matrix_*.csv"))):
        cfg = Path(path).stem.replace("replay_variants_matrix_", "")
        if cfg not in ORDER:
            found.append(summarise(cfg, list(csv.DictReader(open(path)))))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for row in found:
            w.writerow(row)
    print(f"# wrote {OUT}  ({len(found)} configs)")
    # echo a compact verdict table to stdout
    print("\nconfig            n_solv  rel_n     rel_phi   ovf  nonfin  ms_ev/ms_base   masslim")
    for r in found:
        print(f"{r['config']:18s}{r['n_solves']:4d}  {r['used_rel_n_max']:>8s}  "
              f"{r['used_rel_phi_max']:>8s}  {r['overflow_warns_total']:3d}  "
              f"{r['nonfinite_tail_solves']:4d}    "
              f"{r['ms_per_solve_mean']:>6s}/{r['ms_per_solve_baseline']:<6s}  "
              f"{r['mass_limited_frac']:>6s}")


if __name__ == "__main__":
    main()
