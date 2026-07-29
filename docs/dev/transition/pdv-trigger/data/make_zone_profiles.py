#!/usr/bin/env python3
"""L1/L2/L3 zone profiles — T(r), n(r) and where the radiation actually comes from.

The companion to `zone_resolution.csv` (which counts grid points per zone): this one records the
PHYSICAL profiles the bubble-structure solver produces, for the dense / mid / diffuse L21b benches
at the f_A=1 baseline.

RESULT (FINDINGS §22): it FALSIFIED the report's long-standing §1 anatomy claim. L2+L3 really is a
razor-thin, dense skin on the contact discontinuity and its emissivity per unit volume really is
orders of magnitude higher than L1's — but it is ~1e5x thinner, so the interior's volume wins and
**L1 emits ~70% of L_cool** (measured 60-77% across all 14 committed __none arms; L2 15-34%,
L3 1-25%). Cross-checked two ways: the cumulative dL/dr reconstruction here, and the solver's own
bubble_L2Conduction/bubble_L3Intermediate against bubble_LTotal in runs/data/bench_state_traj/.
f_A therefore has a lever on ~a quarter of the cooling at dose 1 (its share grows with dose).

WHAT IS CAPTURED. `bubble_luminosity` integrates each zone twice: once for the luminosity
(`trapezoid(integrand, x=r)`) and once for the volume-weighted mean temperature
(`trapezoid(r**2 * T, x=r)`). Monkeypatching `_trapezoid` therefore yields, per zone, both the
emission integrand dL/dr and — by dividing the second call's y by x**2 — the temperature profile
T(r) on that zone's own r-grid. Density is then EXACT, not inferred: the solver itself sets
n = Pb / ((mu_convert/mu_ion) * k_B * T) (`bubble_luminosity.py:673`), evaluated at the same Pb
captured from the same evaluation; cgs via n_cgs = n_au / cvt.ndens_cgs2au (the convention at
`:785`). Sanity anchors from the live debug line: at the 3e4 K front n ~ 1.8e5 cm^-3, in the
3.1e7 K interior n ~ 1.8e2 cm^-3, and n*T is constant to <1% across all three zones (near-uniform
Pb, the reason the emissivity n^2*Lambda(T) peaks at the cold end).

Same harness as make_zone_resolution.py: run committed __none bench params, capture the 3rd
(settled) energy-phase evaluation, early-exit. Nothing is left running; cwd is a temp dir.

    python docs/dev/transition/pdv-trigger/data/make_zone_profiles.py
Deliverables: data/zone_profiles.csv (committed; the durable record) + zone_profiles.png
"""

import contextlib
import csv
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
REPO = HERE.parents[3]
PARAMS = PDV / "runs" / "params" / "bench5"

# (label, n_bar_H, committed __none param) — the dense/mid/diffuse span of the L21b suite.
CONFIGS = [
    ("dense", 228000.0, PARAMS / "bench5_m5e5_r2p5__none.param"),
    ("mid", 5520.0, PARAMS / "bench3_m1e5_r5__none.param"),
    ("diffuse", 43.1, PARAMS / "bench1_m5e4_r20__none.param"),
]
CAPTURE_EVAL = 3  # eval 1 is the thin-front transient
ZONES = ["L1", "L2", "L3"]
# Downsample per zone for the committed CSV (L1 is ~57k raw points). Endpoints always kept.
KEEP = {"L1": 400, "L2": 300, "L3": 200}


def _thin(arrs, n_keep):
    """Log-spaced-in-index subsample that always keeps both endpoints."""
    n = len(arrs[0])
    if n <= n_keep:
        idx = np.arange(n)
    else:
        idx = np.unique(np.round(np.linspace(0, n - 1, n_keep)).astype(int))
    return [a[idx] for a in arrs]


def _probe(label, n_bar, param_path):
    import trinity.bubble_structure.bubble_luminosity as bl
    from trinity._functions import unit_conversions as cvt
    from trinity._input import read_param
    from trinity import main

    real_trap, real_bl = bl._trapezoid, bl._bubble_luminosity
    calls = []

    def trap(y, x=None, **kw):
        if x is not None and hasattr(x, "__len__") and len(x) > 10:
            calls.append((np.abs(np.asarray(y, dtype=float)), np.asarray(x, dtype=float)))
        return real_trap(y, x=x, **kw) if x is not None else real_trap(y, **kw)

    state = {"n": 0, "rows": None}

    def wrap(*a, **k):
        calls.clear()
        out = real_bl(*a, **k)
        state["n"] += 1
        # Per zone the calls come in pairs: (luminosity integrand, r^2*T). Six calls => L1,L2,L3.
        if state["n"] == CAPTURE_EVAL and len(calls) >= 6:
            p = a[0]
            Pb = float(p["Pb"].value)
            R2 = float(p["R2"].value)
            fac = float(p["mu_convert"].value) / float(p["mu_ion"].value) * float(p["k_B"].value)
            rows = []
            cum_offset = 0.0
            per_zone = {}
            for zone, (dLdr, r), (y_t, _) in zip(ZONES, calls[0::2], calls[1::2]):
                with np.errstate(all="ignore"):
                    T = y_t / r**2
                good = np.isfinite(T) & (T > 0) & np.isfinite(dLdr)
                per_zone[zone] = (r[good], T[good], dLdr[good])
            # Cumulative emitted fraction, integrated from R2 INWARD: L3 (shallowest) -> L2 -> L1.
            for zone in ["L3", "L2", "L1"]:
                r, T, dLdr = per_zone[zone]
                depth = R2 - r
                o = np.argsort(depth)
                r, T, dLdr, depth = r[o], T[o], dLdr[o], depth[o]
                cum = np.concatenate(
                    [[0.0], np.cumsum(0.5 * (dLdr[1:] + dLdr[:-1]) * np.diff(depth))]
                )
                per_zone[zone] = (r, T, dLdr, depth, cum + cum_offset)
                cum_offset += float(cum[-1])
            total = cum_offset if cum_offset > 0 else 1.0
            for zone in ZONES:
                r, T, dLdr, depth, cum = per_zone[zone]
                r, T, dLdr, depth, cum = _thin([r, T, dLdr, depth, cum], KEEP[zone])
                n_cgs = Pb / (fac * T) / cvt.ndens_cgs2au
                for rr, tt, nn, dd, ll, cc in zip(r, T, n_cgs, depth, dLdr, cum):
                    rows.append(
                        {
                            "config": label,
                            "n_bar_H": f"{n_bar:g}",
                            "zone": zone,
                            "R2_pc": f"{R2:.6e}",
                            "Pb_au": f"{Pb:.6e}",
                            "r_pc": f"{rr:.9e}",
                            "depth_R2_minus_r_pc": f"{dd:.6e}",
                            "T_K": f"{tt:.6e}",
                            "n_cgs": f"{nn:.6e}",
                            "dLdr_abs_au": f"{ll:.6e}",
                            "cum_L_frac_from_R2": f"{cc / total:.6f}",
                        }
                    )
            state["rows"] = rows
            raise SystemExit(0)
        return out

    bl._trapezoid, bl._bubble_luminosity = trap, wrap
    params = read_param.read_param(str(param_path))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.start_expansion(params)
    except SystemExit:
        pass
    finally:
        bl._trapezoid, bl._bubble_luminosity = real_trap, real_bl
    return state["rows"]


def collect():
    sys.path.insert(0, str(REPO))
    logging.disable(logging.CRITICAL)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)  # bench path2output is relative + git-ignored; keep the repo clean
        try:
            for label, n_bar, p in CONFIGS:
                got = _probe(label, n_bar, p)
                if got:
                    rows += got
                    print(f"  captured {label:8s} ({len(got)} rows)")
        finally:
            os.chdir(cwd)
    logging.disable(logging.NOTSET)
    if not rows:
        sys.exit("no profiles captured (do the bench params exist?)")
    out = HERE / "zone_profiles.csv"
    with out.open("w", newline="") as fh:
        fh.write(
            "# L1/L2/L3 physical profiles from the bubble-structure solver: T(r), n(r) and the "
            "emission integrand, dense/mid/diffuse L21b benches, f_A=1 baseline, 3rd settled "
            "energy-phase evaluation. Captured by monkeypatching bubble_luminosity._trapezoid "
            "(same harness as make_zone_resolution.py). n is EXACT, not inferred: "
            "n = Pb/((mu_convert/mu_ion)*k_B*T) as the solver sets it (bubble_luminosity.py:673), "
            "cgs via n_au/cvt.ndens_cgs2au. depth = R2 - r (0 at the contact discontinuity, "
            "increasing inward). cum_L_frac_from_R2 = fraction of the TOTAL cooling luminosity "
            "emitted between R2 and this depth, integrated inward across L3->L2->L1. Zones are "
            "downsampled for the committed record (L1 400 / L2 300 / L3 200 pts, endpoints kept). "
            "Regenerate: python docs/dev/transition/pdv-trigger/data/make_zone_profiles.py\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    return rows


def figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(HERE))
    from _trinity_style import use_trinity_style

    use_trinity_style()

    ZC = {"L1": "#c1121f", "L2": "#1f6feb", "L3": "#2a9d3f"}
    ZLBL = {
        "L1": r"$L_1$ hot interior (CIE)",
        "L2": r"$L_2$ conduction front",
        "L3": r"$L_3$ sliver to $10^4$ K",
    }
    labels = [(c, n) for c, n, _ in CONFIGS]
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 9.2), sharex="col")
    for j, (cfg, n_bar) in enumerate(labels):
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        R2 = float(sub[0]["R2_pc"])
        for zone in ZONES:
            z = [r for r in sub if r["zone"] == zone]
            d = np.array([float(r["depth_R2_minus_r_pc"]) for r in z])
            T = np.array([float(r["T_K"]) for r in z])
            n = np.array([float(r["n_cgs"]) for r in z])
            c = np.array([float(r["cum_L_frac_from_R2"]) for r in z])
            d = np.clip(d, 1e-12, None)  # the R2 endpoint sits at depth 0 on a log axis
            for ax, y in ((axes[0, j], T), (axes[1, j], n), (axes[2, j], c)):
                ax.plot(
                    d,
                    y,
                    "-",
                    color=ZC[zone],
                    lw=1.9,
                    label=ZLBL[zone] if (j == 0 and ax is axes[0, j]) else None,
                )
        for i in range(3):
            axes[i, j].set_xscale("log")
            axes[i, j].grid(alpha=0.2, which="both")
        axes[0, j].set_yscale("log")
        axes[1, j].set_yscale("log")
        axes[0, j].set_title(
            f"{cfg}   " r"$\bar{n}_H=$" f"{n_bar:g}" r" cm$^{-3}$" "\n" r"$R_2=$" f"{R2:.3f} pc",
            fontsize=10.5,
        )
        axes[2, j].set_xlabel(r"depth below the contact discontinuity  $R_2-r$  [pc]")
        axes[2, j].set_ylim(-0.03, 1.03)
        axes[0, j].axhline(10**5.5, color="0.35", ls="--", lw=1.0)
    axes[0, 0].set_ylabel(r"$T$  [K]")
    axes[1, 0].set_ylabel(r"$n$  [cm$^{-3}$]")
    axes[2, 0].set_ylabel("cumulative fraction of\n" r"$L_{\rm cool}$ emitted, from $R_2$ inward")
    axes[0, 0].text(
        1.4e-9,
        10**5.65,
        r"$10^{5.5}$ K  (CIE switch $\equiv$ $L_1$/$L_2$ line)",
        fontsize=8,
        color="0.35",
    )
    axes[0, 0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        r"L1/L2/L3 anatomy — $T$ and $n$ vs depth below $R_2$, and where $L_{\rm cool}$ "
        r"is actually emitted ($f_A=1$ baseline)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = PDV / "zone_profiles.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    figure(collect())
