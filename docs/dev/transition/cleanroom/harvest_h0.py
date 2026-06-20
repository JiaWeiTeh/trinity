#!/usr/bin/env python3
"""H0 trigger harvest + G0 divergence (PLAN.md S3, S5, S6).

For each completed full run, find where each candidate transition family WOULD fire,
vs the independent Eb-peak oracle (the PdV-inclusive net-energy zero crossing). The
value is the DIVERGENCE: which family fires at/just after the Eb-peak without
resetting across the WR/SN surge, and which is right for the steep blowout config.

Families (PLAN.md S5):
  F0  instantaneous rate-ratio (CURRENT): (Lgain-Lloss)/Lgain < eps   (eps=0.05)
  F1  cumulative energy:  integral(Lloss)/integral(Lgain) > 1-eta      (eta in 0.2..0.4)
  F2  timescale:          t_cool/t_dyn < k,  t_cool=Eb/Lloss, t_dyn=R2/v2  (k in 1..3)
  F3  force/continuity:   4*pi*R2^2*Pb / (F_rad+F_ram) < 1   (PROVISIONAL force defn)
  F4  blowout (geometric): R2 > rCloud
Oracle: Eb-peak = first t with (Lgain - Lloss - 4*pi*R2^2*v2*Pb) <= 0  [and argmax(Eb)].

    python harvest_h0.py docs/dev/transition/cleanroom/data/c0_*_st6.csv
"""
from __future__ import annotations

import csv, glob, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOURPI = 4.0 * math.pi

# rCloud (pc) per config -- run-const, excluded from snapshots, so taken from the
# validated config table (the original sweep; cross-checked in PLAN.md S4). F4 = R2 > rCloud.
RCLOUD = {"large_diffuse_lowsfe": 88.0, "simple_cluster": 1.69, "small_dense_highsfe": 0.33,
          "midrange_pl0": 8.53, "pl2_steep": 21.4, "be_sphere": 15.5}


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        d = {}
        for k, v in r.items():
            if k == "phase":
                d[k] = v
            else:
                try: d[k] = float(v)
                except (ValueError, TypeError): d[k] = None
        rows.append(d)
    rows.sort(key=lambda d: d.get("t_now") or 0.0)
    return rows


def _first(rows, cond):
    """first t_now where cond(row) true, else None."""
    for r in rows:
        try:
            if cond(r):
                return r["t_now"]
        except (TypeError, ZeroDivisionError):
            continue
    return None


def harvest(path):
    rows = load(path)
    impl = [r for r in rows if r["phase"] == "implicit"]
    name = Path(path).stem.replace("c0_", "").replace("_h0", "").replace("_st6", "")
    rcloud = RCLOUD.get(name)

    def Lg(r):  # Lgain = mechanical luminosity (matches production trigger)
        return r.get("Lmech_total")

    def Ll(r):  # Lloss = bubble radiative cooling (0 outside implicit)
        v = r.get("bubble_Lloss")
        return v if (v is not None and v == v) else 0.0

    # --- Eb-peak oracle (two ways) ---
    pdv = lambda r: FOURPI * r["R2"]**2 * r["v2"] * r["Pb"]
    eb_zero = _first(impl, lambda r: (Lg(r) - Ll(r) - pdv(r)) <= 0)
    eb_argmax = max((r for r in rows if isinstance(r.get("Eb"), float)),
                    key=lambda r: r["Eb"], default=None)
    eb_argmax = eb_argmax["t_now"] if eb_argmax else None

    # --- F0 current ---
    f0 = _first(impl, lambda r: Lg(r) > 0 and (Lg(r) - Ll(r)) / Lg(r) < 0.05)

    # --- F1 cumulative (left-rectangle integral of Lgain, Lloss over the whole run) ---
    f1 = {}
    for eta in (0.20, 0.25, 0.30, 0.40):
        Ig = Il = 0.0; hit = None
        for i, r in enumerate(rows[:-1]):
            dt = rows[i + 1]["t_now"] - r["t_now"]
            if dt <= 0 or Lg(r) is None:
                continue
            Ig += Lg(r) * dt; Il += Ll(r) * dt
            if Ig > 0 and Il / Ig > (1 - eta):
                hit = r["t_now"]; break
        f1[eta] = hit

    # --- F2 timescale ---
    f2 = {k: _first(impl, lambda r, k=k: Ll(r) > 0 and r["v2"] > 0
                    and (r["Eb"] / Ll(r)) / (r["R2"] / r["v2"]) < k) for k in (1, 2, 3)}

    # --- F3 force/continuity. STRUCTURAL FINDING: Pb == P_HII to machine precision
    # (bubble-shell pressure continuity by construction; P_ram=0, F_ISM=0), so a
    # "Pb vs P_HII" criterion is DEGENERATE. The only genuinely-competing outward
    # driver is radiation: F_ram is the bubble-pressure (thermal) force, F_rad the
    # radiation force. F3 = thermal force drops below radiation force. (physics call.)
    f3 = _first(impl, lambda r: (r.get("F_rad") or 0) > 0 and (r.get("F_ram") or 0) > 0
                and r["F_ram"] < r["F_rad"])

    # --- F4 blowout ---
    f4 = _first(rows, lambda r: rcloud and r["R2"] > rcloud) if rcloud else None

    return dict(name=name, t_end=rows[-1]["t_now"], eb_zero=eb_zero, eb_argmax=eb_argmax,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4, rCloud=rcloud)


def _fmt(t):
    return f"{t:.2f}" if isinstance(t, float) else "never"


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_st6.csv")))
    res = [harvest(p) for p in paths]
    print(f"{'config':22s} {'Eb*(0)':>7s} {'Eb*(max)':>8s} {'F0':>6s} "
          f"{'F1.25':>6s} {'F2.k1':>6s} {'F3':>6s} {'F4blow':>7s} {'rCloud':>7s}")
    for a in res:
        rc = f"{a['rCloud']:.1f}" if a['rCloud'] else "?"
        print(f"{a['name']:22s} {_fmt(a['eb_zero']):>7s} {_fmt(a['eb_argmax']):>8s} "
              f"{_fmt(a['f0']):>6s} {_fmt(a['f1'][0.25]):>6s} {_fmt(a['f2'][1]):>6s} "
              f"{_fmt(a['f3']):>6s} {_fmt(a['f4']):>7s} {rc:>7s}")
    print("\n# F1 cumulative firing vs eta (t_fire):")
    for a in res:
        print(f"  {a['name']:22s} " + "  ".join(f"eta{e}:{_fmt(a['f1'][e])}" for e in (0.20,0.25,0.30,0.40)))
    print("\nLegend: Eb*(0)=net-energy zero-crossing oracle; Eb*(max)=argmax(Eb). "
          "A good trigger fires at/just after Eb*. F0=current; never=stalls.")


if __name__ == "__main__":
    main()
