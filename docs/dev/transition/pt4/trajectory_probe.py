#!/usr/bin/env python3
"""Finer probe: print the Lloss(t), Lgain(t), ratio(t) trajectory at sampled
times for hybr vs legacy on selected configs, to characterise the surge-then-
collapse shape and the matched-time divergence. Reads committed CSVs only.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "cleanroom", "data")


def load_traj(cfg, tag):
    path = os.path.join(DATA, f"c0_{cfg}_{tag}.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                t = float(r["t_now"]); lg = float(r["bubble_Lgain"])
                ll = float(r["bubble_Lloss"]); R2 = float(r["R2"])
                rC = float(r["rCloud"]) if r.get("rCloud") else float("nan")
                T0 = float(r["T0"]) if r.get("T0") else float("nan")
            except (ValueError, TypeError, KeyError):
                continue
            if lg != lg or ll != ll or lg <= 0:
                continue
            out.append((t, lg, ll, (lg - ll) / lg, R2, rC, T0))
    return out


def sample(traj, n=14):
    """Pick ~n rows log-ish spaced by index."""
    if len(traj) <= n:
        return traj
    step = len(traj) / n
    idx = sorted(set(int(i * step) for i in range(n)) | {len(traj) - 1})
    return [traj[i] for i in idx]


def main():
    for cfg in ["simple_cluster", "pl2_steep", "small_dense_highsfe", "large_diffuse_lowsfe"]:
        print(f"\n{'='*92}\n{cfg}\n{'='*92}")
        for tag in ["h0", "legacy"]:
            traj = load_traj(cfg, tag)
            if not traj:
                print(f"  [{tag}] no data"); continue
            print(f"  [{tag}]  t        Lgain        Lloss        ratio     R2/rCloud    T0")
            for (t, lg, ll, rat, R2, rC, T0) in sample(traj):
                frac = R2 / rC if rC == rC and rC > 0 else float("nan")
                print(f"        {t:9.4g} {lg:11.4e} {ll:11.4e} {rat:9.4f} "
                      f"{frac:9.3f}   {T0:9.3e}")
            # peak Lloss
            pk = max(traj, key=lambda x: x[2])
            print(f"        Lloss PEAK={pk[2]:.4e} @ t={pk[0]:.4g} (R2/rCloud={pk[4]/pk[5] if pk[5]>0 else float('nan'):.3f}); "
                  f"final Lloss={traj[-1][2]:.4e} @ t={traj[-1][0]:.4g}")


if __name__ == "__main__":
    main()
