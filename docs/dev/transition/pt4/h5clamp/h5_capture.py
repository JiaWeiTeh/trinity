#!/usr/bin/env python3
"""H5 box-width CAPTURE-REPLAY (the feasible causal test).

WHY THIS instead of 4 separate full sims per config: a full LEGACY sim with a
WIDENED (β,δ) box is prohibitively slow — the L-BFGS-B fallback scans a large
(often infeasible) region per cooling segment (xi-out-of-bounds / MonotonicError
warnings), and small_dense's first implicit segment alone runs many minutes. So
instead we run ONE legacy sim per config (the DEFAULT box W0 — the fastest path
the code already takes) and, AT EACH implicit segment, additionally re-solve
betadelta with each WIDENED box on the SAME params/epoch, recording the
counterfactual (β,δ,Lgain,Lloss,ratio).

This isolates the BOX's causal effect on the cooling ratio at each real epoch:
  - production trigger reads Lloss = betadelta_result.bubble_properties.bubble_LTotal
    (+leak) and Lgain = Lmech_total; ratio = (Lgain-Lloss)/Lgain
    (run_energy_implicit_phase.py:855-857, 1142-1173). We record exactly that
    quantity at the box-perturbed (β,δ).
  - If widening the box collapses Lloss (ratio recovers above 0.05) at the epoch
    where W0 crosses, the crossing is CAUSED by the box (supports H5 for that
    config). If Lloss/ratio is unchanged by widening, the box was not binding
    there → the W0 crossing is genuine (refutes H5 for that config).

CAVEAT (documented in H5_FINDINGS.md): this is a PER-SEGMENT counterfactual — the
trajectory (R2,v2,Eb,Pb,T0) is held on the W0-legacy path; only the box's effect
on the betadelta solve at each epoch is varied. It does NOT let the trajectory
itself diverge under a wider box (a full sim would). It is, however, a CLEANER
isolation of the box→Lloss→ratio mechanism, and W3 (wide-box legacy) is still NOT
hybr (different root-finder). Use c0_<cfg>_h0.csv as the hybr reference.

Production untouched: we monkeypatch the get_betadelta box constants (via
h5_variants) only inside a wrapper around solve_betadelta_pure, restoring W0 after
each segment so the REAL trajectory is the unmodified legacy one.

    OMP_NUM_THREADS=1 timeout 2400 python h5_capture.py \
        --param ../../cleanroom/configs/small_dense_highsfe.param \
        --stop_t 0.1 --out data/h5_capture_small_dense_highsfe.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLEANROOM = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom"))
sys.path.insert(0, CLEANROOM)

TRIGGER = 0.05
WIDTHS = ["W0", "W1", "W2", "W3"]
COLS = ["config", "t_now", "width", "beta", "delta", "Lgain", "Lloss",
        "ratio", "converged", "err"]

# rows captured across all segments (also written INCREMENTALLY to the out CSV so a
# time-boxed/killed run still yields every completed segment).
CAPTURE: list[dict] = []


def _install_capture(cfg, out_path):
    """Wrap solve_betadelta_pure so each REAL (W0) segment also re-solves under the
    widened boxes on the same params, recording the counterfactual ratio. The REAL
    return value is always the W0 (default-box) solve, so the trajectory is the
    unmodified legacy one. Each segment's rows are appended+flushed to out_path
    immediately, so partial progress survives a kill/timeout."""
    import trinity.phase1b_energy_implicit.get_betadelta as gbd
    import h5_variants

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fh = open(out_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    writer.writeheader()
    fh.flush()

    orig = gbd.solve_betadelta_pure

    def _emit(row):
        CAPTURE.append(row)
        writer.writerow(row)
        fh.flush()

    def wrapped(beta_guess, delta_guess, params, method="grid"):
        # ---- REAL solve: default box (W0) — this is what the sim integrates ----
        h5_variants.apply("W0")
        real = orig(beta_guess, delta_guess, params, method)
        t_now = params["t_now"].value
        phase = params["current_phase"].value if "current_phase" in params else ""
        if phase == "implicit":
            # Seed the wide-box solves with the W0 RESULT (not the cold energy-phase
            # guess): the W0 (beta,delta) is the closest available point to the
            # wide-box optimum, so when the box does NOT bind the wide solve hits the
            # "already converged" early-exit (get_betadelta.py:638) immediately
            # (fast). When the box DOES bind, the solver moves off the W0 point — and
            # that movement (plus its cost) is itself the signal we are capturing.
            seed_b, seed_d = real.beta, real.delta
            for w in WIDTHS:
                if w == "W0":
                    res = real
                else:
                    h5_variants.apply(w)
                    try:
                        res = orig(seed_b, seed_d, params, method)
                    except BaseException as e:  # noqa: BLE001
                        _emit({"config": cfg, "t_now": t_now, "width": w,
                               "beta": "", "delta": "", "Lgain": "",
                               "Lloss": "", "ratio": "", "converged": "",
                               "err": f"{type(e).__name__}"[:40]})
                        continue
                lg, ll = res.L_gain, res.L_loss
                ratio = ((lg - ll) / lg) if (lg and lg > 0 and ll is not None and ll == ll) else None
                _emit({
                    "config": cfg, "t_now": t_now, "width": w,
                    "beta": res.beta, "delta": res.delta,
                    "Lgain": lg, "Lloss": ll,
                    "ratio": ("" if ratio is None else ratio),
                    "converged": res.converged, "err": ""})
            # restore the default box for the rest of THIS segment's physics
            h5_variants.apply("W0")
        return real

    gbd.solve_betadelta_pure = wrapped
    # the implicit phase imported the symbol by name; rebind there too.
    import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rimp
    rimp.solve_betadelta_pure = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--stop_t", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    cfg = os.path.splitext(os.path.basename(args.param))[0]
    # incremental writer installed here; each segment's rows are flushed to args.out
    # so a time-boxed/killed run still yields every completed segment.
    _install_capture(cfg, args.out)

    import c0_consistency as c0
    run_dir = args.run_dir or f"/tmp/h5cap/{cfg}"
    t0 = time.time()
    crashed = ""
    try:
        c0.run_config(args.param, args.stop_t, refine=1.0, solver="legacy", run_dir=run_dir)
    except SystemExit as e:
        crashed = f"SystemExit:{e}"
    except BaseException as e:  # noqa: BLE001
        crashed = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
    runtime = time.time() - t0

    # quick per-width crossing summary from the capture
    by_w = {}
    for r in CAPTURE:
        by_w.setdefault(r["width"], []).append(r)
    print(f"[{cfg}] captured {len(CAPTURE)} rows ({len(by_w.get('W0', []))} implicit segments) "
          f"in {runtime:.0f}s crashed={crashed!r}")
    for w in WIDTHS:
        seq = sorted([r for r in by_w.get(w, []) if r["ratio"] != ""],
                     key=lambda r: r["t_now"])
        if not seq:
            print(f"  {w}: no data"); continue
        cross = next((r for r in seq if float(r["ratio"]) < TRIGGER), None)
        rmin = min(float(r["ratio"]) for r in seq)
        ct = f"{cross['t_now']:.5g}" if cross else "None"
        print(f"  {w}: n={len(seq)} cross_t={ct} ratio_min={rmin:.4f} "
              f"beta@last={seq[-1]['beta']}")


if __name__ == "__main__":
    main()
