#!/usr/bin/env python3
"""
Compute rCloud and the cloud-edge density drop (nEdge -> nISM) for each
cleanroom config, straight from the production pipeline.

Uses the real read_param + get_InitCloudProp path so all unit conversions
match production exactly. Then samples the production density profile
get_density_profile() just inside and just outside rCloud to measure the
n drop at the cloud edge (which feeds Lloss ~ n^2 in the swept shell / ISM).

Read-only use of production functions; no sims, no production edits.

Run: python docs/dev/transition/pt4/h2_rcloud_compute.py
Writes: docs/dev/transition/pt4/h2_rcloud_edge.csv
"""
import csv
import os

import numpy as np

from trinity._input.read_param import read_param
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.cloud_properties.density_profile import get_density_profile
import trinity._functions.unit_conversions as cvt

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.normpath(os.path.join(HERE, "..", "cleanroom", "configs"))
OUT = os.path.join(HERE, "h2_rcloud_edge.csv")

CONFIGS = ["simple_cluster", "pl2_steep", "small_dense_highsfe",
           "midrange_pl0", "large_diffuse_lowsfe", "be_sphere"]

rows = []
hdr = (f"{'config':24s} {'prof':6s} {'rCloud[pc]':>11s} {'nCore':>10s} "
       f"{'nEdge':>10s} {'nISM':>7s} {'n(0.99Rc)':>11s} {'n(1.01Rc)':>11s} "
       f"{'n_in/n_out':>11s} {'nEdge/nISM':>11s}")
print(hdr)
print("-" * len(hdr))
for name in CONFIGS:
    params = read_param(os.path.join(CFG, name + ".param"))
    props = get_InitCloudProp(params)
    rCloud = props.rCloud
    nISM_cgs = params["nISM"].value * cvt.ndens_au2cgs
    nCore_cgs = params["nCore"].value * cvt.ndens_au2cgs
    nEdge_cgs = props.nEdge * cvt.ndens_au2cgs

    # Sample the PRODUCTION density profile just inside / just outside rCloud.
    # The tanh bridge has width SMOOTH_FRAC*rCloud = 1% rCloud, so sample a few
    # percent out to clear it and read the true ambient floor.
    n_just_in_cgs = get_density_profile(0.97 * rCloud, params) * cvt.ndens_au2cgs
    n_just_out_cgs = get_density_profile(1.10 * rCloud, params) * cvt.ndens_au2cgs

    drop_in_out = n_just_in_cgs / n_just_out_cgs
    drop_edge_ism = nEdge_cgs / nISM_cgs

    prof = params["dens_profile"].value.replace("dens", "")
    print(f"{name:24s} {prof:6s} {rCloud:11.4g} {nCore_cgs:10.3g} "
          f"{nEdge_cgs:10.3g} {nISM_cgs:7.3g} {n_just_in_cgs:11.4g} "
          f"{n_just_out_cgs:11.4g} {drop_in_out:11.4g} {drop_edge_ism:11.4g}")
    rows.append(dict(config=name, profile=params["dens_profile"].value,
                     rCloud_pc=rCloud, nCore_cgs=nCore_cgs, nEdge_cgs=nEdge_cgs,
                     nISM_cgs=nISM_cgs, n_0p97Rc_cgs=n_just_in_cgs,
                     n_1p10Rc_cgs=n_just_out_cgs, drop_in_over_out=drop_in_out,
                     nEdge_over_nISM=drop_edge_ism))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nWrote {OUT}")
