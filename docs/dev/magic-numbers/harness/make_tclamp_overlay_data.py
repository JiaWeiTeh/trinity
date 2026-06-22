#!/usr/bin/env python3
"""Tabulate get_dudt(T) under the OLD (1e4) vs NEW (table-edge) low-T floor, once,
into a committed CSV so the overlay figure is a pure read (no physics re-run).

The only difference between the old and new code is the clamp; the downstream
branch logic is identical. So both curves come from the *working-tree* get_dudt:
  * new(T) = get_dudt(T)                  -- the new file-tied floor (3162 K edge)
  * old(T) = get_dudt(max(T, 1e4))        -- reproduces the old hard 1e4 floor
    (for T>=1e4 the new clamp never fires, so get_dudt(T) IS the old behaviour;
     for T<1e4, max(T,1e4)=1e4 reproduces the old `if T<1e4: T=1e4` exactly).

Run from repo root:
    python docs/dev/magic-numbers/harness/make_tclamp_overlay_data.py
    -> docs/dev/magic-numbers/data/tclamp_dudt_overlay.csv
"""
import csv
import os
import sys

import numpy as np
import scipy.interpolate

sys.path.insert(0, os.getcwd())

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity.cooling.net_coolingcurve as ncc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "tclamp_dudt_overlay.csv")


def setup():
    p = read_param("param/simple_cluster.param")
    p["t_now"].value = 0.1
    logT, logL = np.loadtxt(p["path_cooling_CIE"].value, unpack=True)
    p["cStruc_cooling_CIE_logT"].value = logT
    p["cStruc_cooling_CIE_logLambda"].value = logL
    p["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(logT, logL, kind="linear")
    c, h, nc = non_CIE.get_coolingStructure(p)
    p["cStruc_cooling_nonCIE"].value = c
    p["cStruc_heating_nonCIE"].value = h
    p["cStruc_net_nonCIE_interpolation"].value = nc
    _, nonCIE_Tmin = ncc._noncie_cutoffs(c)
    return p, nonCIE_Tmin


def main():
    params, nonCIE_Tmin = setup()
    nd_au = 1e2 * cvt.ndens_cgs2au   # representative cgs ndens 1e2 cm^-3
    ph_au = 1e10 * cvt.phi_cgs2au    # representative cgs phi  1e10 /cm^2/s

    # T spans the raise zone (<3162), the over-floored decade [3162,1e4), and into
    # the real operating range; non-CIE branch only (all <= 10**5.5) keeps it 1:1.
    Tgrid = np.logspace(3.0, 4.7, 160)
    rows = []
    for T in Tgrid:
        new = ncc.get_dudt(0.1, nd_au, T, ph_au, params)        # new file-tied floor
        old = ncc.get_dudt(0.1, nd_au, max(T, 1e4), ph_au, params)  # reproduces old 1e4 floor
        rows.append((T, np.log10(T), old, new))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "log10T", "dudt_old_1e4floor", "dudt_new_tableedge"])
        w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} rows)  nonCIE_Tmin={nonCIE_Tmin} (edge {10**nonCIE_Tmin:.1f} K)")


if __name__ == "__main__":
    main()
