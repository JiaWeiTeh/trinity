#!/usr/bin/env python3
"""Tabulate h4_eval.csv into a markdown results table + a no-op diff check
(control configs: V0 vs PDVCAP trajectories must be byte/track-identical because
the cap never activates where PdV<Lmech). Pure reads of committed CSVs — no sims.

    python docs/dev/transition/pt4/h4_analyze.py            # print markdown
    python docs/dev/transition/pt4/h4_analyze.py --noop     # control no-op diff
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "h4_eval.csv"
TRAJ = HERE / "traj"


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _fmt(x, nd=3):
    try:
        v = float(x)
        if v != v:
            return "nan"
        if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.2e}"
        return f"{v:.{nd}g}"
    except (TypeError, ValueError):
        return str(x) if x not in (None, "") else "-"


def results_table():
    rows = _rows(EVAL)

    # sort: config, then variant (V0 first), then t_window
    def key(r):
        return (r["config"], r["variant"] != "V0", float(r.get("t_window") or 0))

    rows.sort(key=key)
    cols = [
        ("config", "config"),
        ("variant", "variant"),
        ("t_window", "t_win"),
        ("end_code", "code"),
        ("reached_phase", "phase"),
        ("final_t", "final_t"),
        ("final_R2", "R2"),
        ("final_v2", "v2"),
        ("final_Eb", "final_Eb"),
        ("min_Eb_seen", "min_Eb"),
        ("cap_activated", "cap"),
        ("max_pdv_ratio_in_window", "PdV/L_in"),
        ("max_pdv_ratio_after", "PdV/L_after"),
        ("survived_past_window", "survived"),
        ("self_sustained", "selfsust"),
        ("runtime_s", "rt_s"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    print(head)
    print(sep)
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if k in (
                "config",
                "variant",
                "reached_phase",
                "cap_activated",
                "survived_past_window",
                "self_sustained",
                "end_code",
            ):
                cells.append(str(v) if v not in (None, "") else "-")
            else:
                cells.append(_fmt(v))
        print("| " + " | ".join(cells) + " |")


def noop_check(configs):
    """For each control config, diff every matched (t_now,Eb,R2) row between the
    V0 and PDVCAP trajectory CSVs. Cap must never fire -> bit/track-identical."""
    print("| config | V0 code | PDVCAP code | cap_act | max|ΔR2| | max rel|ΔEb| | no-op? |")
    print("|---|---|---|---|---|---|---|")
    erows = {(_r["config"], _r["variant"]): _r for _r in _rows(EVAL)}
    for cfg in configs:
        v0 = TRAJ / f"h4_traj_{cfg}_V0.csv"
        # PDVCAP control tag is {cfg}_PDVCAP_tw1e-2
        cap = TRAJ / f"h4_traj_{cfg}_PDVCAP_tw1e-2.csv"
        if not (v0.exists() and cap.exists()):
            print(f"| {cfg} | (missing traj) | | | | | |")
            continue
        a = {float(r["t_now"]): r for r in _rows(v0) if r["t_now"]}
        b = {float(r["t_now"]): r for r in _rows(cap) if r["t_now"]}
        common = sorted(set(a) & set(b))
        dR2 = dEb = 0.0
        for t in common:
            ra, rb = a[t], b[t]
            try:
                dR2 = max(dR2, abs(float(ra["R2"]) - float(rb["R2"])))
                eb0 = float(ra["Eb"])
                if eb0 != 0:
                    dEb = max(dEb, abs(float(ra["Eb"]) - float(rb["Eb"])) / abs(eb0))
            except (TypeError, ValueError, KeyError):
                pass
        # eval-row tags: V0 row config==cfg variant==V0; PDVCAP row config==cfg variant==PDVCAP
        er0 = erows.get((cfg, "V0"), {})
        erc = erows.get((cfg, "PDVCAP"), {})
        cap_act = erc.get("cap_activated", "?")
        noop = (
            "**bit-identical**"
            if (dR2 == 0 and dEb == 0)
            else ("track-identical (fp)" if dEb < 1e-6 else "DIFFERS")
        )
        print(
            f"| {cfg} | {er0.get('end_code','?')} | {erc.get('end_code','?')} "
            f"| {cap_act} | {_fmt(dR2)} | {_fmt(dEb)} | {noop} |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noop", action="store_true", help="control no-op diff table")
    ap.add_argument("--configs", nargs="*", default=["small_1e6", "simple_cluster", "pl2_steep"])
    args = ap.parse_args()
    if not EVAL.exists():
        sys.exit(f"missing {EVAL}")
    if args.noop:
        noop_check(args.configs)
    else:
        results_table()


if __name__ == "__main__":
    main()
