#!/usr/bin/env python3
"""Reduce the f_kappa(n_H) sweep to one summary row per run -> summary.csv (run ON HPC).

The 819-combo sweep is many GB of dictionary.jsonl that must stay on the cluster. The calibration needs
only a handful of scalars per run (theta_blowout, theta_max, cooling_fired, the axes). This walks the
sweep once and writes one small summary.csv (a few hundred KB for ~800 runs) -- that tiny table is what
you rsync to your laptop and plot; the jsonl never crosses the wire.

Modeled on paper/II-survey/reduce_survey.py (same pattern): STDLIB-ONLY (orjson/tqdm optional with
fallbacks; NO numpy/scipy/trinity import), so it runs on a bare login node and never trips the solver's
numpy pin. Axes are recovered from the run-folder name (authoritative for the sweep), rCloud from
metadata.json (reliable there, in pc).

The single calibration metric is theta_blowout = bubble_LTotal/Lmech_total at first R2>rCloud (the
developed in-cloud cooling fraction), plus theta_max and cooling_fired -- computed STREAMING here to
exactly match the proven array-based harvest() in make_kappa_blowout_calibration.py (validated by
--selftest against a committed run).

Usage:
    python docs/dev/transition/pdv-trigger/data/reduce_fkappa_sweep.py outputs/sweep_fkappa_nH
    python .../reduce_fkappa_sweep.py outputs/sweep_fkappa_nH -o summary.csv --workers 8
    python .../reduce_fkappa_sweep.py --selftest   # validate theta_blowout vs a committed cal run
Then:
    python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py outputs/sweep_fkappa_nH/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

try:
    from orjson import loads as _loads
except Exception:          # noqa: BLE001 - stays runnable on a bare node
    from json import loads as _loads

try:
    from tqdm import tqdm
except Exception:          # noqa: BLE001
    def tqdm(it, **_kw):
        return it

_TRIGGER = 0.95            # cooling_balance: theta -> 0.95 fires the energy->momentum transition

# run-name axes, e.g. 1e7_sfe030_n3e3_PL0_coolingBoostKappa12p0
_RE_MCLOUD = re.compile(r"^([0-9.]+e[0-9]+)")
_RE_SFE = re.compile(r"_sfe(\d+)")
_RE_NCORE = re.compile(r"_n([0-9][0-9.]*(?:e[0-9]+)?)(?=_|$)")
_RE_FK = re.compile(r"coolingBoostKappa(\d+)p(\d+)")


def parse_axes(name: str) -> dict:
    out = {k: None for k in ("mCloud", "sfe", "nCore", "cooling_boost_kappa")}
    m = _RE_MCLOUD.match(name)
    if m:
        out["mCloud"] = float(m.group(1))
    m = _RE_SFE.search(name)
    if m:
        out["sfe"] = int(m.group(1)) / 100.0
    m = _RE_NCORE.search(name)
    if m:
        out["nCore"] = float(m.group(1))
    m = _RE_FK.search(name)
    if m:
        out["cooling_boost_kappa"] = float(f"{m.group(1)}.{m.group(2)}")
    return out


def read_rcloud(run_dir: Path):
    p = run_dir / "metadata.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
    except Exception:
        return None
    # rCloud may sit at top level or nested; walk shallowly for it.
    def find(o):
        if isinstance(o, dict):
            if "rCloud" in o and isinstance(o["rCloud"], (int, float)):
                return o["rCloud"]
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        return None
    return find(meta)


def reduce_run(jsonl_path: Path) -> dict:
    """One streaming pass -> the calibration scalars. Matches make_kappa_blowout_calibration.harvest():
    theta = bubble_LTotal/Lmech_total over IMPLICIT rows; theta_blowout at first R2>rCloud (else
    theta_max); cooling_fired = reached momentum/transition AND (no blowout OR theta_max>=0.95)."""
    run_dir = jsonl_path.parent
    row = {"run_name": run_dir.name}
    row.update(parse_axes(run_dir.name))
    rCloud = read_rcloud(run_dir)
    row["rCloud"] = rCloud

    def g(d, k):
        v = d.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    n_impl = 0
    theta_max = None
    theta_blowout = None      # theta at first R2>rCloud (within implicit)
    blowout_t = None
    reached_momentum = False
    last_phase = None
    t_final = None
    with open(jsonl_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = _loads(line)
            except ValueError:
                continue
            phase = d.get("current_phase")
            last_phase = phase
            t_final = g(d, "t_now")
            if phase in ("transition", "momentum"):
                reached_momentum = True
            if phase != "implicit":
                continue
            n_impl += 1
            Lcool, Lmech = g(d, "bubble_LTotal"), g(d, "Lmech_total")
            if Lcool is None or Lmech is None or Lmech == 0:
                continue
            theta = Lcool / Lmech
            if theta_max is None or theta > theta_max:
                theta_max = theta
            R2 = g(d, "R2")
            if (theta_blowout is None and blowout_t is None
                    and rCloud and R2 is not None and R2 > rCloud):
                theta_blowout = theta
                blowout_t = g(d, "t_now")

    if n_impl == 0:
        raise ValueError("no implicit-phase rows")
    if theta_blowout is None:      # never crossed rCloud -> cooling fired first (use the peak)
        theta_blowout = theta_max
    cooling_fired = bool(reached_momentum and (blowout_t is None or
                                               (theta_max is not None and theta_max >= _TRIGGER)))
    row.update({
        "n_impl": n_impl, "t_final": t_final, "phase_final": last_phase,
        "theta_blowout": theta_blowout, "theta_max": theta_max,
        "blowout_t": blowout_t, "reached_momentum": reached_momentum,
        "cooling_fired": cooling_fired,
    })
    return row


_COLUMNS = ["run_name", "mCloud", "sfe", "nCore", "cooling_boost_kappa", "rCloud",
            "n_impl", "t_final", "phase_final", "theta_blowout", "theta_max",
            "blowout_t", "reached_momentum", "cooling_fired"]


def _reduce_safe(path):
    try:
        return reduce_run(path), None
    except Exception as exc:  # noqa: BLE001 - one bad run shouldn't kill the sweep
        return None, f"{path.parent.name}: {exc}"


def _selftest():
    """Validate streaming theta_blowout against the proven array harvester on a committed cal run."""
    here = Path(__file__).resolve().parent
    repo = here.parents[4]
    run = repo / "outputs" / "kcal" / "cal_compact__k1" / "dictionary.jsonl"
    if not run.exists():
        print(f"selftest SKIP: {run} not present (run the cal grid first)")
        return
    sys.path.insert(0, str(here))
    from make_kappa_blowout_calibration import harvest
    ref = harvest(str(run.parent))["theta_blowout"]
    got = reduce_run(run)["theta_blowout"]
    assert abs(got - ref) < 1e-9, f"theta_blowout mismatch: streaming {got} vs harvest {ref}"
    print(f"selftest OK: streaming theta_blowout={got:.6f} == harvest()={ref:.6f} (cal_compact__k1)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", nargs="?", help="directory of run subdirectories")
    ap.add_argument("-o", "--output", default=None, help="output CSV (default <sweep_dir>/summary.csv)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--selftest", action="store_true", help="validate theta vs harvest() on a committed run")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not args.sweep_dir:
        ap.error("sweep_dir is required (or pass --selftest)")

    sweep = Path(args.sweep_dir)
    out_path = Path(args.output) if args.output else sweep / "summary.csv"
    print(f"scanning {sweep} ...", file=sys.stderr, flush=True)
    sims = sorted(sweep.glob("*/dictionary.jsonl")) or sorted(sweep.rglob("dictionary.jsonl"))
    if not sims:
        print(f"No dictionary.jsonl under {sweep}", file=sys.stderr)
        return 1

    print(f"reducing {len(sims)} runs (workers={args.workers}) ...", file=sys.stderr, flush=True)
    rows, failed = [], 0
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for row, err in tqdm(pool.imap_unordered(_reduce_safe, sims), total=len(sims)):
                (rows.append(row) if row else (_warn(err), None)) and None
                failed += 0 if row else 1
    else:
        for path in tqdm(sims, total=len(sims)):
            row, err = _reduce_safe(path)
            if row:
                rows.append(row)
            else:
                failed += 1
                _warn(err)

    if not rows:
        print("No runs could be reduced.", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(_COLUMNS))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS + extra, restval="")
        w.writeheader()
        w.writerows(rows)
    kb = out_path.stat().st_size / 1024
    print(f"Wrote {len(rows)} runs -> {out_path} ({kb:.0f} KB)"
          + (f"  [{failed} skipped]" if failed else ""))
    fired = sum(1 for r in rows if r.get("cooling_fired"))
    print(f"  cooling_fired: {fired}/{len(rows)}")
    return 0


def _warn(msg):
    if msg:
        print(f"WARN: skipped {msg}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
