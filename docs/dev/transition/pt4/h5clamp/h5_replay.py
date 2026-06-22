#!/usr/bin/env python3
"""H5 box-width TARGETED REPLAY (the decisive, FAST causal test).

The full wide-box legacy sim and even the per-segment capture are dominated by the
ODE integration through hundreds of stiff early implicit segments (measured: the
first implicit segment alone runs minutes). But the causal H5 question only needs
the betadelta solve AT the committed-legacy epochs — especially the CROSSING epoch.

So this replays the COMMITTED legacy trajectory (cleanroom/data/c0_<cfg>_legacy.csv,
which already contains the crossing): trinity is set up ONCE (cloud props + SPS +
CIE cooling — fast, no ODE), then for each committed implicit epoch we reconstruct
the segment's params exactly as the implicit loop does
(run_energy_implicit_phase.py:740-789):
  - set state t_now/R2/v2/Eb/T0 from the committed row,
  - cool_alpha = t_now/R2*v2,
  - refresh SPS feedback via get_current_sps_feedback + updateDict
    (Lmech_total/v_mech_total/pdot_total/pdotdot_total — deterministic in t),
  - rebuild the non-CIE cooling structure (read_cloudy.get_coolingStructure),
  - set Pb/c_sound and bubble_Leak (Cf=1 -> leak 0) from the committed Pb,
  - current_phase='implicit',
then solve betadelta (LEGACY) under each box width, seeded with the committed
(beta,delta), and record (beta, delta, Lgain, Lloss, ratio). NO ODE integration —
one betadelta solve per (epoch, width), seconds each.

Lloss = betadelta_result.bubble_properties.bubble_LTotal (+leak); Lgain =
Lmech_total; ratio = (Lgain-Lloss)/Lgain — the exact production trigger quantity
(run_energy_implicit_phase.py:855-857, 1142-1173).

Consistency gate: at W0 the replayed (beta, ratio) must reproduce the committed
row's (cool_beta, (Lgain-Lloss)/Lgain) — proves the reconstruction is faithful.

Caveats (documented in H5_FINDINGS.md): (1) per-segment counterfactual — the
trajectory is held on the committed W0-legacy path; only the box's effect on the
betadelta solve at each epoch is varied. (2) W3 (wide-box legacy) != hybr (a
different root-finder). c0_<cfg>_h0.csv is the hybr reference.

    OMP_NUM_THREADS=1 python h5_replay.py \
        --param ../../cleanroom/configs/small_dense_highsfe.param \
        --legacy ../../cleanroom/data/c0_small_dense_highsfe_legacy.csv \
        --out data/h5_replay_small_dense_highsfe.csv
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TRIGGER = 0.05
WIDTHS = ["W0", "W1", "W2", "W3"]
COLS = ["config", "t_now", "width", "beta", "delta", "Lgain", "Lloss",
        "ratio", "converged", "residual", "committed_beta", "committed_ratio", "err"]


def _f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def _setup_params(param_path):
    """Run trinity setup steps 1-3 (cloud props, SPS, CIE cooling) — fast, no ODE."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    import numpy as np
    import scipy.interpolate
    from trinity._input import read_param
    from trinity.phase0_init import get_InitCloudProp
    from trinity.sps import read_sps

    params = read_param.read_param(param_path)
    params["betadelta_solver"].value = "legacy"
    get_InitCloudProp.get_InitCloudProp(params)
    f_mass = params["mCluster"] / params["sps_refmass"]
    sps_data = read_sps.read_sps(f_mass, params)
    params["sps_data"].value = sps_data
    params["sps_f"].value = read_sps.get_interpolation(sps_data)
    logT, logLambda = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    params["cStruc_cooling_CIE_logLambda"].value = logLambda
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logLambda, kind="linear")
    params["current_phase"].value = "implicit"
    return params


def _prime_epoch(params, row):
    """Reconstruct the implicit-loop segment setup for a committed row
    (run_energy_implicit_phase.py:740-789)."""
    from trinity.sps.update_feedback import get_current_sps_feedback
    from trinity._input.dictionary import updateDict
    from trinity.cooling.non_CIE import read_cloudy

    t = _f(row, "t_now"); R2 = _f(row, "R2"); v2 = _f(row, "v2")
    Eb = _f(row, "Eb"); T0 = _f(row, "T0")
    params["t_now"].value = t
    params["R2"].value = R2
    params["v2"].value = v2
    params["Eb"].value = Eb
    params["T0"].value = T0
    params["cool_alpha"].value = t / R2 * v2
    # feedback (deterministic in t) -> Lmech_total/v_mech_total/pdot_total/pdotdot_total
    feedback = get_current_sps_feedback(t, params)
    updateDict(params, feedback)
    # non-CIE cooling structure at this state
    cooling_nonCIE, heating_nonCIE, netcooling_interpolation = read_cloudy.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = netcooling_interpolation
    # get_residual_pure recomputes Pb internally from (R2,Eb,Lmech_total,v_mech_total)
    # and does NOT read params['Pb'] or params['c_sound']; c_sound only feeds the
    # leak term, which is 0 for the sealed (Cf=1) committed legacy runs.
    if "bubble_Leak" in params:
        params["bubble_Leak"].value = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--legacy", required=True, help="committed c0_<cfg>_legacy.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="solve every Nth implicit epoch (1=all)")
    ap.add_argument("--t-min", type=float, default=0.0,
                    help="only replay committed epochs with t_now >= this [Myr] "
                         "(skip the stiff early epochs; focus on the crossing region)")
    args = ap.parse_args()

    cfg = os.path.splitext(os.path.basename(args.param))[0]
    params = _setup_params(args.param)

    import h5_variants
    import trinity.phase1b_energy_implicit.get_betadelta as gbd

    # committed implicit-phase rows with a finite ratio
    rows = []
    for r in csv.DictReader(open(args.legacy)):
        if r.get("phase") != "implicit":
            continue
        t = _f(r, "t_now"); lg = _f(r, "bubble_Lgain"); ll = _f(r, "bubble_Lloss")
        if t and t > 0 and lg and lg > 0 and ll is not None and t >= args.t_min:
            r["_ratio"] = (lg - ll) / lg
            rows.append(r)
    rows = rows[::max(1, args.stride)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fh = open(args.out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()

    out_rows = []
    for r in rows:
        t = _f(r, "t_now")
        cb = _f(r, "cool_beta"); cr = r.get("_ratio")
        try:
            _prime_epoch(params, r)
        except BaseException as e:  # noqa: BLE001
            w.writerow({"config": cfg, "t_now": t, "width": "PRIME_ERR",
                        "err": f"{type(e).__name__}: {e}"[:80]}); fh.flush()
            continue
        seed_b = _f(r, "cool_beta"); seed_d = _f(r, "cool_delta")
        if seed_b is None or seed_b != seed_b:
            seed_b = 0.5
        if seed_d is None or seed_d != seed_d:
            seed_d = -0.5
        for width in WIDTHS:
            h5_variants.apply(width)
            try:
                res = gbd.solve_betadelta_pure(seed_b, seed_d, params)
                lg, ll = res.L_gain, res.L_loss
                ratio = ((lg - ll) / lg) if (lg and lg > 0 and ll is not None and ll == ll) else None
                row = {"config": cfg, "t_now": t, "width": width,
                       "beta": res.beta, "delta": res.delta, "Lgain": lg, "Lloss": ll,
                       "ratio": ("" if ratio is None else ratio),
                       "converged": res.converged, "residual": res.total_residual,
                       "committed_beta": cb, "committed_ratio": cr, "err": ""}
            except BaseException as e:  # noqa: BLE001
                row = {"config": cfg, "t_now": t, "width": width, "err": f"{type(e).__name__}"[:40],
                       "committed_beta": cb, "committed_ratio": cr}
            w.writerow(row); fh.flush()
            out_rows.append(row)
        h5_variants.apply("W0")
    fh.close()

    # summary per width
    by_w = {}
    for r in out_rows:
        if r.get("ratio") not in ("", None):
            by_w.setdefault(r["width"], []).append(r)
    print(f"[{cfg}] replayed {len(rows)} committed implicit epochs x {len(WIDTHS)} boxes")
    for width in WIDTHS:
        seq = sorted(by_w.get(width, []), key=lambda r: r["t_now"])
        if not seq:
            print(f"  {width}: no data"); continue
        cross = next((r for r in seq if float(r["ratio"]) < TRIGGER), None)
        rmin = min(float(r["ratio"]) for r in seq)
        ct = f"{cross['t_now']:.5g}" if cross else "None"
        print(f"  {width}: n={len(seq)} cross_t={ct} ratio_min={rmin:.4f}")
    # consistency: W0 replay vs committed
    w0 = sorted(by_w.get("W0", []), key=lambda r: r["t_now"])
    if w0:
        import statistics
        difs = [abs(float(r["ratio"]) - float(r["committed_ratio"]))
                for r in w0 if r.get("committed_ratio") not in ("", None)]
        if difs:
            print(f"  W0-vs-committed ratio |Δ|: median={statistics.median(difs):.3g} "
                  f"max={max(difs):.3g} (small => faithful reconstruction)")


if __name__ == "__main__":
    main()
