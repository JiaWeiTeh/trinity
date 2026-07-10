#!/usr/bin/env python3
"""Phase-1 of the f_A workflow (SOURCE_TERM_DESIGN.md §3 Phase 1): offline completeness.

Two deliverables in one builder (both reuse make_fA_source_boost's monkeypatch `_solve`, so the
production code is untouched and f_A=1 is 1-ULP-equivalent to production):

  (A) COVERAGE EXTENSION -- run the f_A theta-response on the configs the §2 six-config screen
      did NOT cover: the 2 new standard configs small_1e6 (route-a control) and normal_n1e3
      (native-fire), from committed Phase-1 trajectories `data/traj_<cfg>.csv`, PLUS the 2
      captured FM1 fixtures (stiff 5e9 ~ fail_repro control -- NOTE its sfe=0.01 != fail_repro's
      0.1 -- and a mild cluster) via the make_fm1_rootcheck loader. These are the band's boundary
      definers; the point is they meet f_A offline before HPC (Phase 4).

  (B) CONDENSATION-EDGE MAP -- per config x epoch, raise f_A until the solved bubble_dMdt <= 0
      (the McKee-Cowie evaporation->condensation reversal) or the solve fails. Grid {16,24,32,48,64}
      then bisect the last-positive / first-nonpositive bracket. REGISTERED PREDICTION
      (SOURCE_TERM_DESIGN §3 Phase 1 / FINDINGS §15 P4): the edge sits near local theta ~ 1
      (the reversal IS cooling balance); an edge at theta << 1 would falsify the "gradual
      approach" reading and must be written up, not tuned around.

Row selection (audit G10): FA._sample_rows gives ~N_ROWS even rows; we additionally force-include
the first ROWS_EARLY accepted rows (the dense race window t<0.06 Myr) so the edge map is not blind
to the decisive early epoch.

COVERAGE here: the 6 cleanroom configs (cleanroom c0 CSVs) + normal_n1e3 (traj CSV) + the stiff-5e9
& mild-cluster FM1 fixtures. small_1e6 is included IFF `data/traj_small_1e6.csv` exists (it is on
the §8d diffuse cliff and may not be offline-screenable in-container -- see FINDINGS §15a).

CONTAMINATION: replayed frozen states -- structural verdicts only (dial / dMdt sign / stability /
edge location). NO fire threshold is quotable from this builder (same grade as §2).

REPRODUCE (from repo root; reads committed trajectories/fixtures, minutes):
    python docs/dev/transition/pdv-trigger/data/make_fA_edge_map.py
    # env: N_ROWS (default 6), FA_COV (coverage grid, default 1,2,4,8,16),
    #      FA_EDGE (edge grid, default 16,24,32,48,64), CONFIGS (comma list to subset)
Deliverables:
    docs/dev/transition/pdv-trigger/data/fA_edge_map.csv        (edge per config x epoch)
    docs/dev/transition/pdv-trigger/data/fA_coverage9.csv       (theta-response, new configs+fixtures)
    docs/dev/transition/pdv-trigger/fA_edge_map.png
"""
import csv
import importlib.util
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import make_fA_source_boost as FA   # noqa: E402  reuse _solve / _sample_rows / constants
import make_da_replay as DR         # noqa: E402  build_params / replay_row
import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402

_spec = importlib.util.spec_from_file_location("_fm1", os.path.join(_HERE, "make_fm1_rootcheck.py"))
_fm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm1)

N_ROWS = int(os.environ.get("N_ROWS", "6"))
ROWS_EARLY = 3
FA_COV = [float(x) for x in os.environ.get("FA_COV", "1,2,4,8,16").split(",")]
FA_EDGE = [float(x) for x in os.environ.get("FA_EDGE", "16,24,32,48,64,96,128").split(",")]
FIRE = FA.FIRE

# nCore [cm^-3] per config (the 6 cleanroom come from make_da_screen; the 2 new are fixed here).
_NCORE = {"small_1e6": 1e2, "normal_n1e3": 1e3}


def _ncore(cfg):
    return _NCORE.get(cfg, DR.DS.NCORE.get(cfg, float("nan")))


def _traj_csv(cfg):
    """Cleanroom c0 CSV for the canonical 6; else the Phase-1 traj_<cfg>.csv fixture."""
    cr = f"docs/dev/transition/cleanroom/data/c0_{cfg}_h0.csv"
    return cr if os.path.exists(cr) else os.path.join(_HERE, f"traj_{cfg}.csv")


def _theta(L_eff, Lmech):
    return (L_eff / Lmech) if (np.isfinite(L_eff) and Lmech) else float("nan")


def _find_edge(params, Lmech):
    """Raise f_A on FA_EDGE until dMdt<=0 or solve fails; bisect the bracket.
    Returns (fA_edge, theta_at_edge, dMdt_at_edge, reason). theta_at_edge = theta at the last
    f_A with dMdt>0 (the approach to the reversal -- the quantity the theta~1 prediction is about)."""
    last_pos = (1.0, None)     # (fA, theta) at the last positive-dMdt solve
    for fa in FA_EDGE:
        L, dM, *_rest, ok = FA._solve(params, fa)
        if not ok:
            return _bisect_edge(params, Lmech, last_pos[0], fa, last_pos[1], "solve_fail")
        if dM <= 0:
            return _bisect_edge(params, Lmech, last_pos[0], fa, last_pos[1], "dMdt<=0")
        last_pos = (fa, _theta(L, Lmech))
    return (float("nan"), last_pos[1], float("nan"), "no_edge_in_range")


def _bisect_edge(params, Lmech, lo, hi, theta_lo, reason, iters=6):
    """Bisect [lo, hi] for the f_A where dMdt crosses 0 / the solve fails."""
    t_lo = theta_lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        L, dM, *_rest, ok = FA._solve(params, mid)
        if ok and dM > 0:
            lo, t_lo = mid, _theta(L, Lmech)
        else:
            hi = mid
    L, dM, *_rest, ok = FA._solve(params, lo)
    return (hi, t_lo, (dM if ok else float("nan")), reason)


def _rows_for(cfg):
    """Sampled rows (DataFrame) for a config, or None. Force-includes the first ROWS_EARLY."""
    import pandas as pd
    path = _traj_csv(cfg)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#")
    d = df[(df["Pb"] > 0) & np.isfinite(df["bubble_Lloss"])].reset_index(drop=True)
    if len(d) < 2:
        return None
    idx = set(np.linspace(0, len(d) - 1, N_ROWS).round().astype(int).tolist())
    idx |= set(range(min(ROWS_EARLY, len(d))))
    return d.iloc[sorted(idx)].reset_index(drop=True)


def _run_config(cfg, edge_rows, cov_rows, series):
    """Replay each sampled row: coverage theta-response (FA_COV) + edge sweep. Appends in place."""
    params = DR.build_params(cfg)
    sample = _rows_for(cfg)
    if sample is None:
        print(f"[{cfg}] no usable trajectory ({_traj_csv(cfg)}) -- skip"); return
    n = _ncore(cfg)
    new_cfg = cfg in _NCORE
    ser = {"t": [], "theta": {fa: [] for fa in FA_COV}}
    for _, row in sample.iterrows():
        DR.replay_row(params, row)
        Lmech = float(row["Lmech_total"]); t = float(row["t_now"])
        # coverage theta-response (only persisted for the NEW configs; the 6 are already §2)
        if new_cfg:
            ser["t"].append(t)
            for fa in FA_COV:
                L, dM, *_r, ok = FA._solve(params, fa)
                th = _theta(L, Lmech) if ok else float("nan")
                ser["theta"][fa].append(th)
                cov_rows.append(dict(config=cfg, nCore=n, t_now=t, fA=fa,
                                     theta=th, dMdt=dM, solver_ok=ok))
        # edge sweep
        fA_edge, th_edge, dM_edge, reason = _find_edge(params, Lmech)
        edge_rows.append(dict(config=cfg, nCore=n, t_now=t, R2=float(row["R2"]),
                              fA_edge=fA_edge, theta_at_edge=th_edge,
                              dMdt_at_edge=dM_edge, reason=reason))
        print(f"[{cfg}] t={t:.4f} edge fA={fA_edge:.1f} theta@edge={th_edge if th_edge is None else round(th_edge,2)} ({reason})")
    if new_cfg:
        series[f"{cfg} (n={n:.0e})"] = ser


def _run_fixture(label, fixture_name, edge_rows):
    """FM1 single-state fixture: G1 identity + edge sweep only (no trajectory, no Lmech-row)."""
    fixture, params = _fm1._load(fixture_name)
    Lmech = float(params["Lmech_total"].value) if "Lmech_total" in params else float("nan")
    if "bubble_dMdt" in params:
        params["bubble_dMdt"].value = float("nan")
    bp0 = BL.get_bubbleproperties_pure(params)
    L1, _, *_r, ok1 = FA._solve(params, 1.0)
    g1 = abs(L1 - float(bp0.bubble_LTotal)) / abs(float(bp0.bubble_LTotal)) if bp0.bubble_LTotal else float("nan")
    fA_edge, th_edge, dM_edge, reason = _find_edge(params, Lmech)
    edge_rows.append(dict(config=f"fixture:{label}", nCore=float("nan"),
                          t_now=float(params["t_now"].value), R2=float(params["R2"].value),
                          fA_edge=fA_edge, theta_at_edge=th_edge, dMdt_at_edge=dM_edge,
                          reason=f"{reason};G1={g1:.0e}"))
    print(f"[fixture:{label}] G1={g1:.1e} edge fA={fA_edge:.1f} "
          f"theta@edge={th_edge if th_edge is None else round(th_edge,3)} ({reason})")


def main():
    edge_rows, cov_rows, series = [], [], {}
    configs = (os.environ["CONFIGS"].split(",") if os.environ.get("CONFIGS")
               else list(DR.DS.CLEANROOM) + ["normal_n1e3", "small_1e6"])
    for cfg in configs:
        try:
            _run_config(cfg, edge_rows, cov_rows, series)
        except Exception as e:
            print(f"[{cfg}] FAILED: {type(e).__name__}: {e}")
    for label, fx, _note in _fm1._STATES:
        try:
            _run_fixture(label, fx, edge_rows)
        except Exception as e:
            print(f"[fixture {label}] FAILED: {type(e).__name__}: {e}")

    with open(os.path.join(_HERE, "fA_edge_map.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "t_now", "R2", "fA_edge",
                                           "theta_at_edge", "dMdt_at_edge", "reason"])
        w.writeheader(); w.writerows(edge_rows)
        fh.write("# condensation edge: fA_edge = first f_A with solved dMdt<=0 (or solve fail); "
                 "theta_at_edge = theta at the last dMdt>0 f_A (approach to the reversal).\n")
        fh.write("# REGISTERED PREDICTION: theta_at_edge ~ 1 (McKee-Cowie). Replayed states -- "
                 "no fire threshold quotable.\n")
    with open(os.path.join(_HERE, "fA_coverage9.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "t_now", "fA", "theta",
                                           "dMdt", "solver_ok"])
        w.writeheader(); w.writerows(cov_rows)
        fh.write("# f_A theta-response for the configs NOT in the §2 six-config screen "
                 "(small_1e6, normal_n1e3). theta = (L1+fA*(L2+L3))/Lmech.\n")
    print(f"\nwrote fA_edge_map.csv ({len(edge_rows)} rows), fA_coverage9.csv ({len(cov_rows)} rows)")

    # ---- verdict on the edge prediction --------------------------------------------------------
    th_at = [r["theta_at_edge"] for r in edge_rows
             if r["theta_at_edge"] is not None and np.isfinite(r["theta_at_edge"])
             and r["reason"].startswith(("dMdt<=0", "solve_fail"))]
    n_edge = sum(1 for r in edge_rows if not np.isnan(r["fA_edge"]))
    print("\nVERDICT:")
    print(f"  edges found (dMdt<=0 or fail within f_A<=64): {n_edge}/{len(edge_rows)} states")
    if th_at:
        arr = np.array(th_at)
        print(f"  theta_at_edge over found edges: median={np.median(arr):.2f} "
              f"[{arr.min():.2f}, {arr.max():.2f}] (n={len(arr)})")
        near1 = np.mean((arr > 0.7)) * 100
        print(f"  PREDICTION theta_at_edge~1: {near1:.0f}% of edges have theta>0.7 "
              f"({'SUPPORTS' if near1 >= 60 else 'TENSION WITH'} the McKee-Cowie reading)")
    else:
        print("  no dMdt<=0 edges within f_A<=64 -- the condensation edge is BEYOND the grid on "
              "these states (consistent with §2 P4 'no cliff in range'; the reversal is gradual).")

    # ---- figure ---------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # (L) edge map: fA_edge vs config, colored by theta_at_edge
    cfgs = sorted({r["config"] for r in edge_rows}, key=lambda c: _ncore(c) if not np.isnan(_ncore(c)) else 1e9)
    for i, c in enumerate(cfgs):
        rs = [r for r in edge_rows if r["config"] == c]
        y = [r["fA_edge"] if np.isfinite(r["fA_edge"]) else FA_EDGE[-1] * 1.1 for r in rs]
        th = [r["theta_at_edge"] if (r["theta_at_edge"] is not None and np.isfinite(r["theta_at_edge"])) else np.nan for r in rs]
        sc = axL.scatter([i] * len(y), y, c=th, cmap="plasma", vmin=0, vmax=1.2, s=40,
                         edgecolor="0.3", linewidth=0.4)
    axL.axhline(FA_EDGE[-1], ls=":", color="0.5", lw=1)
    axL.text(0.02, FA_EDGE[-1] * 1.02, "grid top (64); above = no edge found", fontsize=7,
             transform=axL.get_yaxis_transform())
    axL.set_xticks(range(len(cfgs)))
    axL.set_xticklabels([c.replace("fixture:", "fx:") for c in cfgs], rotation=30, ha="right", fontsize=6.5)
    axL.set_ylabel(r"$f_A$ at the condensation edge (dMdt$\leq$0)")
    axL.set_title("Condensation-edge map (color = θ at the edge)\nprediction: edges appear near θ≈1",
                  fontsize=10, fontweight="bold")
    cb = fig.colorbar(sc, ax=axL); cb.set_label(r"$\theta_{\rm at\ edge}$", fontsize=8)
    axL.grid(True, axis="y", alpha=0.2)
    # (R) new-config coverage: theta_max vs fA
    if series:
        cmap = plt.get_cmap("viridis")
        labels = sorted(series, key=lambda L: float(L.split("n=")[1].rstrip(")")))
        axR.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
        axR.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
        for i, label in enumerate(labels):
            ser = series[label]; col = cmap(i / max(1, len(labels) - 1))
            tmax = [np.nanmax(ser["theta"][fa]) if np.isfinite(ser["theta"][fa]).any() else np.nan
                    for fa in FA_COV]
            axR.semilogx(FA_COV, tmax, "o-", color=col, lw=1.8, ms=5, label=label)
        axR.set_xlabel(r"$f_A$"); axR.set_ylabel(r"$\theta_{\max}$ over sampled rows")
        axR.set_title("New-config coverage (the §2 six-config screen missed these)",
                      fontsize=10, fontweight="bold")
        axR.legend(fontsize=7.5); axR.grid(True, which="both", alpha=0.2)
    else:
        axR.text(0.5, 0.5, "no new-config coverage this run", ha="center", va="center")
    fig.suptitle("f_A Phase-1: condensation-edge map + missing-config coverage "
                 "(replayed states, no production edit)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fA_edge_map.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
