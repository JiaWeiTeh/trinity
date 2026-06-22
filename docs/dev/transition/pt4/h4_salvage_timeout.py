#!/usr/bin/env python3
"""On a TIMEOUT (grind / long cap that runs past the wall clock), the worker was
killed before h4_run_variant.py could write its CSV row / trajectory. Salvage
both from the partial dictionary.jsonl trinity streamed to disk, and append an
eval row marked end_code=timeout so the grind is recorded (not silently dropped).
The in-memory cap telemetry (n1a/n1b) is lost with the killed process, so those
are marked 'killed'; the PdV/Lmech trajectory + survived/self-sustained flags are
recomputed from the partial jsonl (same definitions as the driver).

Usage: python h4_salvage_timeout.py --tag T --config C --variant V \
           --t_window TW --kappa K --out DIR --csv row.csv --traj traj.csv \
           --timeout_s N
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

FOURPI = 4.0 * math.pi


def _pdv_ratio(r):
    R2, Pb, v2, Lm = r.get("R2"), r.get("Pb"), r.get("v2"), r.get("Lmech_total")
    try:
        if None in (R2, Pb, v2, Lm) or Lm <= 0:
            return None
        val = FOURPI * R2 * R2 * Pb * v2 / Lm
        return val if val == val else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--t_window", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--timeout_s", type=float, default=None)
    args = ap.parse_args()

    rows = []
    hits = list(Path(args.out).rglob("dictionary.jsonl"))
    if hits:
        jsonl = str(max(hits, key=lambda p: p.stat().st_mtime))
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda d: d.get("t_now", 0.0))

    order = {"energy": 1, "implicit": 2, "transition": 3, "momentum": 4}
    drank = max([order.get(r.get("current_phase"), 0) for r in rows], default=0)
    reached = {0: "", 1: "1a", 2: "1b", 3: "1c", 4: "momentum"}[drank]

    last = rows[-1] if rows else {}
    E0 = rows[0].get("Eb") if rows else None
    Ebs = [r.get("Eb") for r in rows if isinstance(r.get("Eb"), (int, float))]
    min_Eb_seen = min(Ebs) if Ebs else None
    tw = args.t_window
    post = [
        r
        for r in rows
        if isinstance(r.get("t_now"), (int, float))
        and r["t_now"] >= tw
        and isinstance(r.get("Eb"), (int, float))
    ]
    self_sustained = None
    final_Eb_growth = None
    if len(post) >= 2:
        final_Eb_growth = post[-1]["Eb"] - post[-2]["Eb"]
        self_sustained = bool(post[-1]["Eb"] > post[-2]["Eb"] and post[-1]["Eb"] > 0)
    survived_past_window = bool(post and post[-1]["Eb"] > 0)
    # PdV/Lmech telemetry recomputed from the jsonl (driver telemetry was killed)
    ratios = [(_pdv_ratio(r), r.get("t_now")) for r in rows]
    rin = [v for v, t in ratios if v is not None and t is not None and t < tw]
    raf = [v for v, t in ratios if v is not None and t is not None and t >= tw]

    row = {
        "config": args.config,
        "variant": args.variant,
        "t_window": tw,
        "kappa": args.kappa,
        "crashed": False,
        "crash_phase": "",
        "crash_excpt": "",
        "end_reason": f"TIMEOUT after {args.timeout_s}s (grind; partial jsonl salvaged)",
        "end_code": "timeout",
        "reached_phase": reached,
        "final_t": last.get("t_now"),
        "final_R2": last.get("R2"),
        "final_v2": last.get("v2"),
        "final_Eb": last.get("Eb"),
        "E0": E0,
        "min_Eb_seen": min_Eb_seen,
        "cap_activated": "killed(unknown)",
        "n1a": "",
        "n1b": "",
        "max_pdv_ratio": round(max(rin + raf), 4) if (rin or raf) else 0.0,
        "max_pdv_ratio_in_window": round(max(rin), 4) if rin else 0.0,
        "max_pdv_ratio_after": round(max(raf), 4) if raf else 0.0,
        "survived_past_window": survived_past_window,
        "self_sustained": self_sustained,
        "final_Eb_growth": final_Eb_growth,
        "n_rows": len(rows),
        "runtime_s": args.timeout_s,
    }
    write_header = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    if args.traj and rows:
        cols = [
            "t_now", "phase", "R2", "v2", "Eb", "Pb", "R1", "T0",
            "Lmech_total", "bubble_LTotal", "pdv_over_lmech", "rCloud",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
        with open(args.traj, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "t_now": r.get("t_now"), "phase": r.get("current_phase"),
                        "R2": r.get("R2"), "v2": r.get("v2"), "Eb": r.get("Eb"),
                        "Pb": r.get("Pb"), "R1": r.get("R1"), "T0": r.get("T0"),
                        "Lmech_total": r.get("Lmech_total"),
                        "bubble_LTotal": r.get("bubble_LTotal"),
                        "pdv_over_lmech": _pdv_ratio(r), "rCloud": r.get("rCloud"),
                    }
                )
    print(
        f"[salvage {args.variant} {args.config} tw={tw}] timeout rows={len(rows)} "
        f"reached={reached} final_t={row['final_t']} final_Eb={row['final_Eb']} "
        f"survived={survived_past_window} selfsust={self_sustained}"
    )


if __name__ == "__main__":
    main()
