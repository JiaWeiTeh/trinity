#!/usr/bin/env python3
"""H5 Part A (offline): is legacy (beta,delta) soft-locked on the clamp box boundary
during the cooling-ratio decline to 0.05? Pure read of committed c0 data.
Box: beta in [0,1], delta in [-1,0] (get_betadelta.py:41-44). Boundary if within EPS.
"""
import csv, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom", "data"))
EPS = 0.02  # within 2% of a bound counts as "on the boundary"
CFG = ["small_dense_highsfe","simple_cluster","midrange_pl0","pl2_steep","be_sphere","large_diffuse_lowsfe"]

def load(path):
    out=[]
    if not os.path.exists(path): return out
    for r in csv.DictReader(open(path)):
        try:
            t=float(r["t_now"]); b=float(r["cool_beta"]); d=float(r["cool_delta"])
            Lg=float(r["bubble_Lgain"]); Ll=float(r["bubble_Lloss"])
        except (ValueError,TypeError,KeyError): continue
        if t>0 and Lg>0 and b==b and d==d:
            out.append((t,b,d,(Lg-Ll)/Lg))
    return out

def on_boundary(b,d):
    return (b<=EPS or b>=1-EPS or d<=-1+EPS or d>=-EPS)

print(f"{'config':22}{'crosses?':9}{'cross_t':>9}{'beta@cross':>11}{'pin_frac_all':>13}{'pin_frac_preX':>14}{'hybr_beta_max':>14}")
for name in CFG:
    leg=load(f"{DATA}/c0_{name}_legacy.csv"); hyb=load(f"{DATA}/c0_{name}_h0.csv")
    if not leg: 
        print(f"{name:22}(no legacy data)"); continue
    cross_i=next((i for i,(t,b,d,r) in enumerate(leg) if r<0.05), None)
    crosses = cross_i is not None
    cross_t = f"{leg[cross_i][0]:.4g}" if crosses else "—"
    beta_x  = f"{leg[cross_i][1]:.3f}" if crosses else "—"
    pin_all = sum(on_boundary(b,d) for _,b,d,_ in leg)/len(leg)
    pre = leg[:cross_i+1] if crosses else leg
    pin_pre = sum(on_boundary(b,d) for _,b,d,_ in pre)/len(pre)
    hbmax = max((b for _,b,_,_ in hyb), default=float("nan")) if hyb else float("nan")
    print(f"{name:22}{str(crosses):9}{cross_t:>9}{beta_x:>11}{pin_all:>13.2f}{pin_pre:>14.2f}{hbmax:>14.2f}")
