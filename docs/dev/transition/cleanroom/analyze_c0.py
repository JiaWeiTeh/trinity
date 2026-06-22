#!/usr/bin/env python3
"""Cross-config C0 analyzer: summarize one or more c0_*.csv per config AND over
time, so no single run/phase is taken at face value (PLAN.md S2/S0.1).

Reports per config: implicit-phase substrate residuals (res_beta GENUINE trajectory,
res_T0_struct solver T-residual on converged segs), the negative-beta fraction, and
the f_ret physical anchor by phase vs the literature 0.01-0.1 band.

    python analyze_c0.py docs/dev/transition/cleanroom/data/c0_*.csv
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import Counter

LIT = (0.01, 0.10)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stats(xs):
    xs = [x for x in xs if isinstance(x, float) and x == x]
    if not xs:
        return None
    s = sorted(xs)
    return dict(n=len(s), med=st.median(s), p90=s[min(len(s) - 1, int(.9 * len(s)))],
                mx=max(s), mn=min(s))


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: _f(v) if k not in ("phase",) else v for k, v in r.items()})
    return rows


def analyze(path):
    rows = load(path)
    name = path.split("/")[-1].replace("c0_", "").replace(".csv", "")
    impl = [r for r in rows if r.get("phase") == "implicit"]
    conv = [r for r in impl if r.get("betadelta_converged")] or impl
    ph = dict(Counter(r.get("phase") for r in rows))
    t_end = max((r["t_now"] for r in rows if isinstance(r.get("t_now"), float)), default=None)
    rb = _stats([r.get("res_beta") for r in impl])
    rT = _stats([r.get("res_T0_struct") for r in conv])
    betas = [r.get("cool_beta") for r in impl if isinstance(r.get("cool_beta"), float)]
    fneg = (sum(1 for b in betas if b < 0) / len(betas)) if betas else None
    # f_ret end value per phase
    fret = {}
    for p in ("energy", "implicit", "transition", "momentum"):
        seg = [r for r in rows if r.get("phase") == p and isinstance(r.get("f_ret"), float)]
        if seg:
            fret[p] = (seg[0]["f_ret"], seg[-1]["f_ret"], min(r["f_ret"] for r in seg))
    return dict(name=name, ph=ph, t_end=t_end, n_impl=len(impl),
                n_conv=len([r for r in impl if r.get("betadelta_converged")]),
                rb=rb, rT=rT, fneg=fneg, fret=fret)


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: analyze_c0.py c0_*.csv")
    res = [analyze(p) for p in paths]
    print(f"{'config':22s} {'t_end':>8s} {'impl':>5s} {'conv':>5s} "
          f"{'res_beta med/p90/max':>24s} {'T0resid med/max':>16s} {'b<0':>5s}")
    for a in sorted(res, key=lambda x: x["name"]):
        rb = a["rb"] or {}
        rT = a["rT"] or {}
        rb_s = (f"{rb.get('med',0):.1%}/{rb.get('p90',0):.1%}/{rb.get('mx',0):.1%}"
                if rb else "-")
        rT_s = f"{rT.get('med',0):.2%}/{rT.get('mx',0):.2%}" if rT else "-"
        fneg = f"{a['fneg']:.0%}" if a["fneg"] is not None else "-"
        te = f"{a['t_end']:.3f}" if a["t_end"] else "-"
        print(f"{a['name']:22s} {te:>8s} {a['n_impl']:5d} {a['n_conv']:5d} "
              f"{rb_s:>24s} {rT_s:>16s} {fneg:>5s}")
    print(f"\nf_ret by phase (first->last, min)  [literature band {LIT}]")
    for a in sorted(res, key=lambda x: x["name"]):
        parts = []
        for p in ("energy", "implicit", "transition", "momentum"):
            if p in a["fret"]:
                f0, f1, fm = a["fret"][p]
                parts.append(f"{p[:4]}:{f0:.2g}->{f1:.2g}")
        print(f"  {a['name']:22s} " + "  ".join(parts))
    # aggregate flags
    rbeds = [a["rb"]["med"] for a in res if a["rb"]]
    print(f"\nAGGREGATE: res_beta median spans {min(rbeds):.1%}-{max(rbeds):.1%} across "
          f"{len(rbeds)} configs (per-config variation => do NOT trust one run).")
    negc = [a["name"] for a in res if a["fneg"]]
    print(f"  configs with ANY negative beta: {negc or 'none (in the t-range run)'}")


if __name__ == "__main__":
    main()
