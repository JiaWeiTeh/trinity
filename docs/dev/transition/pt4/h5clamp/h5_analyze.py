#!/usr/bin/env python3
"""H5 clamp-width sweep — assemble the full W0->W1->W2->W3->hybr sweep table and
print the per-config trend verdict. Pure reads; no sims re-run.

Sources (all committed):
  - W0  (default legacy box): cleanroom/data/c0_<cfg>_legacy.csv     (committed)
  - W1/W2/W3 (widened legacy box): data/rows/<cfg>_<W>.csv           (this experiment)
  - hybr (true-unbounded root reference): cleanroom/data/c0_<cfg>_h0.csv (committed)

W0 and hybr crossing/pin metrics are computed here with the SAME logic the sim
runner uses (cool_beta/cool_delta on/off the box edge, ratio < 0.05), so the table
is internally consistent. The box edges used for pinning:
  W0   beta [0,1]  delta [-1,0]      hybr  unbounded (pin N/A -> "")

Writes h5_sweep.csv (the deliverable) and prints the per-config trend table +
falsification verdict.

    python h5_analyze.py
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom", "data"))
CAP = os.path.join(HERE, "data")  # capture CSVs: h5_capture_<cfg>.csv
OUT = os.path.join(HERE, "h5_sweep.csv")

TRIGGER = 0.05
PIN_EPS = 0.02
CONFIGS = ["small_dense_highsfe", "simple_cluster", "midrange_pl0",
           "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]
# (beta_min, beta_max, delta_min, delta_max) per width; mirror h5_variants.WIDTHS.
BOX = {
    "W0": (0.0, 1.0, -1.0, 0.0),
    "W1": (-1.0, 2.0, -2.0, 1.0),
    "W2": (-4.0, 4.0, -4.0, 4.0),
    "W3": (-20.0, 20.0, -20.0, 20.0),
}
COLS = ["config", "box_width", "beta_min", "beta_max", "delta_min", "delta_max",
        "crosses", "cross_t", "ratio_min", "beta_at_cross", "boundary_pin_frac",
        "reached_phase", "n_rows", "crashed", "crash_excpt", "runtime_s"]


def _load_c0(path):
    """Return [(t, ratio, beta, delta, phase), ...] from a cleanroom c0 CSV."""
    out = []
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            t = float(r["t_now"]); Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (t > 0 and Lg > 0 and Lg == Lg and Ll == Ll):
            continue
        try:
            b = float(r.get("cool_beta", "nan"))
        except (TypeError, ValueError):
            b = float("nan")
        try:
            d = float(r.get("cool_delta", "nan"))
        except (TypeError, ValueError):
            d = float("nan")
        out.append((t, (Lg - Ll) / Lg, b, d, r.get("phase", "")))
    return out


def _reached_phase(seq):
    order = {"energy": 1, "implicit": 2, "transition": 3, "momentum": 4}
    drank, deepest = 0, ""
    for _, _, _, _, ph in seq:
        rk = order.get(ph, 0)
        if rk > drank:
            drank, deepest = rk, {1: "1a", 2: "1b", 3: "1c", 4: "momentum"}[rk]
    return deepest


def _metrics_from_c0(seq, box):
    """Crossing/pin metrics for a c0 sequence under the given box (None box => hybr)."""
    cross_i = next((i for i, (t, ra, b, d, ph) in enumerate(seq) if ra < TRIGGER), None)
    crosses = cross_i is not None
    cross_t = seq[cross_i][0] if crosses else None
    beta_x = seq[cross_i][2] if crosses else None
    ratio_min = min((ra for _, ra, _, _, _ in seq), default=None)
    pin_frac = None
    if box is not None:
        bmin, bmax, dmin, dmax = box

        def on_b(b, d):
            hit = False
            if b == b:
                hit = hit or abs(b - bmin) <= PIN_EPS or abs(b - bmax) <= PIN_EPS
            if d == d:
                hit = hit or abs(d - dmin) <= PIN_EPS or abs(d - dmax) <= PIN_EPS
            return hit
        pre = seq[:cross_i + 1] if crosses else seq
        pin_frac = (sum(on_b(b, d) for _, _, b, d, _ in pre) / len(pre)) if pre else None
    return crosses, cross_t, ratio_min, (beta_x if beta_x == beta_x else None), pin_frac


def _fmt(x, g="%.6g"):
    return "" if x is None or (isinstance(x, float) and x != x) else (g % x)


def _load_capture(cfg):
    """Read the per-segment box counterfactual -> {width: [(t, ratio, beta, delta)]}.
    Sources (both legacy-betadelta re-solved per epoch under each box, recording the
    counterfactual (beta, delta, ratio)):
      - h5_replay_<cfg>.csv   : replayed on the COMMITTED legacy trajectory
                                (cleanroom c0_*_legacy.csv), covers the CROSSING region.
      - h5_capture_<cfg>.csv  : per-segment during a live W0 sim (early segments only).
    Replay rows take precedence at shared epochs (they reach the crossing). Missing
    files -> whatever is present (possibly {})."""
    out: dict[str, dict] = {}  # width -> {t: (t, ratio, beta, delta)}
    for fname, prec in ((f"h5_capture_{cfg}.csv", 0), (f"h5_replay_{cfg}.csv", 1)):
        p = os.path.join(CAP, fname)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            w = r.get("width")
            if w not in ("W0", "W1", "W2", "W3"):
                continue
            try:
                t = float(r["t_now"]); ra = float(r["ratio"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                b = float(r.get("beta", "nan"))
            except (TypeError, ValueError):
                b = float("nan")
            try:
                d = float(r.get("delta", "nan"))
            except (TypeError, ValueError):
                d = float("nan")
            # round t to merge replay/capture epochs; replay (prec=1) overwrites capture
            key = round(t, 7)
            slot = out.setdefault(w, {})
            if key not in slot or prec >= slot[key][4]:
                slot[key] = (t, ra, b, d, prec)
    # collapse to sorted lists of (t, ratio, beta, delta)
    return {w: [(t, ra, b, d) for (t, ra, b, d, _) in sorted(v.values())]
            for w, v in out.items()}


def _metrics_from_capture(seq, box):
    """Same crossing/pin metrics as _metrics_from_c0 but on capture tuples
    (t, ratio, beta, delta)."""
    cross_i = next((i for i, (t, ra, b, d) in enumerate(seq) if ra < TRIGGER), None)
    crosses = cross_i is not None
    cross_t = seq[cross_i][0] if crosses else None
    beta_x = seq[cross_i][2] if crosses else None
    ratio_min = min((ra for _, ra, _, _ in seq), default=None)
    bmin, bmax, dmin, dmax = box

    def on_b(b, d):
        hit = False
        if b == b:
            hit = hit or abs(b - bmin) <= PIN_EPS or abs(b - bmax) <= PIN_EPS
        if d == d:
            hit = hit or abs(d - dmin) <= PIN_EPS or abs(d - dmax) <= PIN_EPS
        return hit
    pre = seq[:cross_i + 1] if crosses else seq
    pin = (sum(on_b(b, d) for _, _, b, d in pre) / len(pre)) if pre else None
    return crosses, cross_t, ratio_min, (beta_x if beta_x == beta_x else None), pin


def build_rows():
    rows = []
    for cfg in CONFIGS:
        # --- W0 (committed legacy) ---
        seq = _load_c0(os.path.join(CLEAN, f"c0_{cfg}_legacy.csv"))
        cr, ct, rm, bx, pin = _metrics_from_c0(seq, BOX["W0"])
        bmin, bmax, dmin, dmax = BOX["W0"]
        rows.append({"config": cfg, "box_width": "W0", "beta_min": bmin, "beta_max": bmax,
                     "delta_min": dmin, "delta_max": dmax, "crosses": cr,
                     "cross_t": _fmt(ct), "ratio_min": _fmt(rm), "beta_at_cross": _fmt(bx, "%.4f"),
                     "boundary_pin_frac": _fmt(pin, "%.4f"), "reached_phase": _reached_phase(seq),
                     "n_rows": len(seq), "crashed": "", "crash_excpt": "committed c0_legacy",
                     "runtime_s": ""})
        # --- W1/W2/W3 (capture-replay: legacy betadelta re-solved per segment on
        #     the W0 trajectory under each widened box; h5_capture.py) ---
        cap = _load_capture(cfg)
        for w in ("W1", "W2", "W3"):
            bmin, bmax, dmin, dmax = BOX[w]
            seqw = cap.get(w)
            if seqw:
                cr, ct, rm, bx, pin = _metrics_from_capture(seqw, BOX[w])
                rows.append({"config": cfg, "box_width": w, "beta_min": bmin, "beta_max": bmax,
                             "delta_min": dmin, "delta_max": dmax, "crosses": cr,
                             "cross_t": _fmt(ct), "ratio_min": _fmt(rm),
                             "beta_at_cross": _fmt(bx, "%.4f"), "boundary_pin_frac": _fmt(pin, "%.4f"),
                             "reached_phase": "1b(capture)", "n_rows": len(seqw),
                             "crashed": "", "crash_excpt": "capture-replay", "runtime_s": ""})
            else:
                rows.append({"config": cfg, "box_width": w, "beta_min": bmin, "beta_max": bmax,
                             "delta_min": dmin, "delta_max": dmax, "crosses": "", "cross_t": "",
                             "ratio_min": "", "beta_at_cross": "", "boundary_pin_frac": "",
                             "reached_phase": "", "n_rows": 0, "crashed": "MISSING",
                             "crash_excpt": "no capture data", "runtime_s": ""})
        # --- hybr (committed h0; true-unbounded-root reference, NOT the W->inf limit) ---
        seq = _load_c0(os.path.join(CLEAN, f"c0_{cfg}_h0.csv"))
        cr, ct, rm, bx, _ = _metrics_from_c0(seq, None)
        rows.append({"config": cfg, "box_width": "hybr", "beta_min": "", "beta_max": "",
                     "delta_min": "", "delta_max": "", "crosses": cr, "cross_t": _fmt(ct),
                     "ratio_min": _fmt(rm), "beta_at_cross": _fmt(bx, "%.4f"),
                     "boundary_pin_frac": "", "reached_phase": _reached_phase(seq),
                     "n_rows": len(seq), "crashed": "", "crash_excpt": "committed c0_h0 (hybr)",
                     "runtime_s": ""})
    return rows


def main():
    rows = build_rows()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"# wrote {len(rows)} rows -> {OUT}\n")

    # per-config trend: how does cross_t / ratio_min move W0 -> W3 (legacy box sweep)?
    by_cfg: dict[str, dict] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {})[r["box_width"]] = r

    def g(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    hdr = f"{'config':22}" + "".join(f"{w:>16}" for w in ("W0", "W1", "W2", "W3", "hybr"))
    print("=== cross_t (None = no 0.05 crossing) ===")
    print(hdr)
    for cfg in CONFIGS:
        cells = by_cfg[cfg]
        line = f"{cfg:22}"
        for w in ("W0", "W1", "W2", "W3", "hybr"):
            r = cells.get(w, {})
            crosses = str(r.get("crosses", "")).lower() == "true"
            ct = g(r, "cross_t")
            line += f"{(f'{ct:.4g}' if crosses and ct is not None else 'None'):>16}"
        print(line)
    print("\n=== ratio_min ===")
    print(hdr)
    for cfg in CONFIGS:
        cells = by_cfg[cfg]
        line = f"{cfg:22}"
        for w in ("W0", "W1", "W2", "W3", "hybr"):
            rm = g(cells.get(w, {}), "ratio_min")
            line += f"{(f'{rm:.4f}' if rm is not None else '—'):>16}"
        print(line)
    print("\n=== boundary_pin_frac (legacy boxes only) ===")
    print(f"{'config':22}" + "".join(f"{w:>10}" for w in ("W0", "W1", "W2", "W3")))
    for cfg in CONFIGS:
        cells = by_cfg[cfg]
        line = f"{cfg:22}"
        for w in ("W0", "W1", "W2", "W3"):
            pf = g(cells.get(w, {}), "boundary_pin_frac")
            line += f"{(f'{pf:.2f}' if pf is not None else '—'):>10}"
        print(line)

    # verdict heuristic per config: does widening move/vanish the crossing?
    print("\n=== per-config verdict ===")
    for cfg in CONFIGS:
        cells = by_cfg[cfg]
        w0 = cells.get("W0", {}); w3 = cells.get("W3", {})
        c0c = str(w0.get("crosses", "")).lower() == "true"
        c3c = str(w3.get("crosses", "")).lower() == "true"
        t0, t3 = g(w0, "cross_t"), g(w3, "cross_t")
        # coverage: does the widened-box (capture/replay) data extend to the W0
        # crossing time? If not, "no crossing" is TRUNCATED, not a real vanish.
        cap = _load_capture(cfg)
        w3seq = cap.get("W3") or cap.get("W2") or cap.get("W1")
        w3_tmax = max((t for t, _, _, _ in w3seq), default=None) if w3seq else None
        covers = (t0 is not None and w3_tmax is not None and w3_tmax >= t0)
        if not c0c:
            verdict = "W0 never crosses (control)"
        elif not c3c and not covers:
            verdict = (f"INCONCLUSIVE — widened-box data truncated at t<={w3_tmax} < W0 "
                       f"cross_t={t0} (did not reach the crossing; see h5_pinning_summary.csv "
                       f"for the decisive beta-on-edge test)")
        elif c0c and not c3c and covers:
            verdict = "crossing VANISHES as box widens => box-CAUSED (supports H5)"
        elif c0c and c3c and t0 is not None and t3 is not None and t3 > 1.5 * t0:
            verdict = f"crossing MOVES LATER ({t0:.3g}->{t3:.3g}) => box-influenced (supports H5)"
        elif c0c and c3c:
            verdict = f"crossing UNCHANGED ({t0:.3g}->{t3:.3g}) => box NOT the cause (refutes H5)"
        else:
            verdict = "indeterminate"
        print(f"  {cfg:22}: {verdict}")

    # consistency: the capture also re-solves W0 on the same trajectory; its W0
    # crossing must match the committed c0_*_legacy.csv W0 crossing (proves the
    # capture wrapper reproduces the legacy solver at the default box).
    print("\n=== consistency: capture-W0 vs committed-legacy-W0 (cross_t) ===")
    for cfg in CONFIGS:
        cap = _load_capture(cfg)
        seqw = cap.get("W0")
        if not seqw:
            continue
        crc, ctc, rmc, _, _ = _metrics_from_capture(seqw, BOX["W0"])
        seq = _load_c0(os.path.join(CLEAN, f"c0_{cfg}_legacy.csv"))
        cr0, ct0, rm0, _, _ = _metrics_from_c0(seq, BOX["W0"])
        ok = (crc == cr0) and (ct0 is None or ctc is None
                               or abs((ctc or 0) - (ct0 or 0)) <= 0.02 * (ct0 or 1))
        print(f"  {cfg:22}: capture cross_t={ctc} ratio_min={rmc:.4f}  | "
              f"committed cross_t={ct0} ratio_min={rm0:.4f}  [{'MATCH' if ok else 'DIFF'}]")


if __name__ == "__main__":
    main()
