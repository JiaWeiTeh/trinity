#!/usr/bin/env python3
"""Is the C3c crossover wind-sensitive, or just the phase handover moving?

PLAN.md §3c stage 3. `t_cross` alone is a misleading discriminator: the C3c
crossover is structurally confined to the transition/momentum phases (the
confinement ratio at transition entry is < 1 on every config measured), so
`t_entry` is a hard FLOOR on `t_cross` and a cloud that never reaches transition
cannot cross at any wind strength. Reporting such a run as "wind pushed the
crossover out" is a false positive — it voided the first stage-3 ladder.

This tool separates the two effects by reporting, per run:

  t_entry      first snapshot in the transition phase (the handover)
  t_cross      first snapshot where P_C3a > P_conf   (same definition the
               offline screen uses, incl. the ramped F_ram/(4 pi R2^2) conf)
  lag          t_cross - t_entry, the part NOT explained by the handover
  lag/t_entry  the lag as a fraction of the handover time
  %tr_dur      the lag as a fraction of the transition phase's own duration
  ratio@entry  P_C3a/P_conf at transition entry — how close to crossing the run
               already is when it arrives
  ratio@cross  P_C3a/P_conf at the first row past the crossing. Lands at
               1.03-1.47 rather than 1.0 because snapshots are segment-spaced;
               a lag under ~2% of t_entry is AT THE RESOLUTION LIMIT and must
               not be reported as a resolved ordering.

Runs that never enter transition are reported as VOID, not as "never crossed".

Usage (from the repo root):
    python docs/dev/phii-identity/harness/lag_vs_handover.py \
        --out docs/dev/phii-identity/data/b5s3_ladder_lag.csv \
        outputs/phii/b1__<sha>/<config> [...]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import c3_offline_screen as S  # noqa: E402
from _stamp import stamp  # noqa: E402

COLS = [
    "run",
    "status",
    "t_entry",
    "t_cross",
    "lag",
    "lag_over_t_entry",
    "pct_of_transition_duration",
    "ratio_at_entry",
    "ratio_at_cross",
]


def params_for(run_dir):
    """Same param load the screen uses, incl. the metadata.json overlay for rCloud."""
    p = S.read_param(str(sorted(run_dir.glob("*.param"))[0]))
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            meta = {}
        for k, v in meta.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in p:
                p[k].value = v
    return p


def ratio(rec, params):
    n = S.candidates_for_row(rec, params).get("C3a_cavity")
    Pb, R2, F_ram = rec.get("Pb"), rec.get("R2"), rec.get("F_ram")
    if not n or n <= 0 or not Pb or Pb <= 0:
        return None
    conf = (F_ram / (4.0 * math.pi * R2**2)) if (F_ram and R2 and F_ram > 0) else Pb
    return S.n_to_P(n, params) / conf if conf > 0 else None


def analyse(run_dir):
    if not (run_dir / "dictionary.jsonl").exists():
        return {"run": run_dir.name, "status": "no output"}
    _, reg, _, err = S.analyse(run_dir)
    if err:
        return {"run": run_dir.name, "status": err}
    params = params_for(run_dir)
    recs = [json.loads(ln) for ln in (run_dir / "dictionary.jsonl").open() if ln.strip()]
    tr = [r for r in recs if r.get("current_phase") == "transition" and r.get("t_now") is not None]
    if not tr:
        # Not "never crossed" — the run never reached a phase where crossing is possible.
        return {"run": run_dir.name, "status": "VOID (never enters transition)"}
    t_cross = next((float(r["t_cross"]) for r in reg if r["t_cross"] != "never"), None)
    ts = [r["t_now"] for r in tr]
    t_entry, dur = min(ts), max(ts) - min(ts)
    row = {
        "run": run_dir.name,
        "status": "ok" if t_cross is not None else "reaches transition, never crosses",
        "t_entry": f"{t_entry:.6g}",
        "ratio_at_entry": (
            f"{ratio(min(tr, key=lambda r: r['t_now']), params):.4f}"
            if ratio(min(tr, key=lambda r: r["t_now"]), params)
            else "NA"
        ),
    }
    if t_cross is None:
        return row
    after = [r for r in recs if (r.get("t_now") or -1) >= t_cross]
    r_cross = ratio(min(after, key=lambda r: r["t_now"]), params) if after else None
    row.update(
        {
            "t_cross": f"{t_cross:.6g}",
            "lag": f"{t_cross - t_entry:+.6g}",
            "lag_over_t_entry": f"{(t_cross - t_entry) / t_entry:+.4f}",
            "pct_of_transition_duration": (
                f"{100 * (t_cross - t_entry) / dur:.1f}" if dur > 0 else "NA"
            ),
            "ratio_at_cross": f"{r_cross:.4f}" if r_cross else "NA",
        }
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [analyse(d) for d in args.runs]
    w = max(len(r["run"]) for r in rows)
    print(
        f"{'run':<{w}} {'status':<34}{'t_entry':>10}{'t_cross':>10}{'lag':>11}"
        f"{'lag/t_en':>10}{'%tr_dur':>9}{'r@entry':>9}{'r@cross':>9}"
    )
    for r in rows:
        print(
            f"{r['run']:<{w}} {r.get('status',''):<34}{r.get('t_entry','-'):>10}"
            f"{r.get('t_cross','-'):>10}{r.get('lag','-'):>11}{r.get('lag_over_t_entry','-'):>10}"
            f"{r.get('pct_of_transition_duration','-'):>9}{r.get('ratio_at_entry','-'):>9}"
            f"{r.get('ratio_at_cross','-'):>9}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# t_cross vs the energy->transition handover (PLAN.md §3c stage 3).\n")
        fh.write("# lag = t_cross - t_entry is the part NOT explained by the handover moving.\n")
        fh.write("# VOID = the run never reaches transition, so it cannot cross at any wind\n")
        fh.write("# strength; that is not evidence about winds.\n")
        fh.write("# ratio_at_cross > 1 is snapshot granularity: a lag under ~2% of t_entry is\n")
        fh.write("# at the resolution limit and is NOT a resolved ordering.\n")
        wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
