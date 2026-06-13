#!/usr/bin/env python3
"""Phase 2.3 four-arm diagnostics: aggregate stats + plots from arms_*.jsonl.

Reads the per-arm/per-segment jsonl emitted by arms.py and produces, for each
config:
  - a markdown stats table (stdout + arms_stats.md)
  - arms_summary.png : convergence and evaluation-count bars, G2 gates drawn
  - arms_rootmap.png : (beta, delta) maps -- production (arm A) vs hybr (arm D)
                       converged roots against the legacy box

Pure re-read of the jsonl; reruns are cheap and side-effect-free. Numbers are
only as good as the jsonl on disk -- regenerate after any new arms run.

Usage: python scratch/phase2/analyze_arms.py
"""
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = ["arms_mock4e3", "arms_simple1e5"]
ARMS = ["A", "B", "C", "D"]
ARM_LABEL = {
    "A": "A control",
    "B": "B metric",
    "C": "C cap+bounds",
    "D": "D hybr",
}
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # legacy hard bounds
G2_CONV, G2_EVALS = 80.0, 15.0  # gate thresholds


def load(cfg):
    by = defaultdict(list)
    for line in open(HERE / f"{cfg}.jsonl"):
        if line.strip():
            r = json.loads(line)
            by[r["arm"]].append(r)
    return by


def stats(rs):
    ok = [r for r in rs if "error" not in r]
    errs = [r for r in rs if "error" in r]
    evs = [r["n_evals"] for r in ok]
    n = len(rs)
    d = dict(
        segs=n,
        conv_f=sum(1 for r in ok if r.get("conv_f")),
        conv_g=sum(1 for r in ok if r.get("conv_g")),
        short=sum(1 for r in rs if r.get("short_circuit")),
        med_ev=st.median(evs) if evs else float("nan"),
        max_ev=max(evs) if evs else 0,
        aborts=len(errs),
        oob=sum(1 for r in ok if r.get("left_old_box")),
    )
    # abort categories
    cats = Counter()
    for r in errs:
        msg = r["error"]
        if "structure failure" in msg:
            cats["structure"] += 1
        elif "invalid dMdt" in msg:
            cats["neg_dMdt"] += 1
        elif "Timeout" in msg:
            cats["timeout"] += 1
        else:
            cats["other"] += 1
    return d, dict(cats)


def md_table(cfg, by):
    lines = [
        f"### {cfg}",
        "",
        "| arm | segs | conv f | conv g | short% | med ev | max ev "
        "| aborts (kind) | out-of-box |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        d, cats = stats(by[a])
        n = d["segs"]
        cf = f"{d['conv_f']}/{n} ({100 * d['conv_f'] / n:.0f}%)"
        cg = f"{d['conv_g']}/{n} ({100 * d['conv_g'] / n:.0f}%)"
        sh = f"{100 * d['short'] / n:.0f}%"
        ab = f"{d['aborts']}" + (f" ({', '.join(f'{k}:{v}' for k, v in cats.items())})" if cats else "")
        oob = "-" if a in ("A", "B") else str(d["oob"])
        lines.append(
            f"| {ARM_LABEL[a]} | {n} | {cf} | {cg} | {sh} | "
            f"{d['med_ev']:.0f} | {d['max_ev']} | {ab} | {oob} |"
        )
    lines.append("")
    return "\n".join(lines)


def plot_summary(data, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(ARMS))
    w = 0.38
    for k, cfg in enumerate(CONFIGS):
        by = data[cfg]
        conv = [100 * stats(by[a])[0]["conv_g"] / stats(by[a])[0]["segs"] for a in ARMS]
        ev = [stats(by[a])[0]["med_ev"] for a in ARMS]
        off = (k - 0.5) * w
        ax1.bar([i + off for i in x], conv, w, label=cfg)
        ax2.bar([i + off for i in x], ev, w, label=cfg)
    ax1.axhline(G2_CONV, ls="--", c="k", lw=1, label=f"G2 gate {G2_CONV:.0f}%")
    ax1.set_ylabel("convergence under g (%)")
    ax1.set_title("Convergence per arm")
    ax2.axhline(G2_EVALS, ls="--", c="k", lw=1, label=f"G2 gate {G2_EVALS:.0f}")
    ax2.set_ylabel("median evaluations / segment")
    ax2.set_title("Cost per arm (lower better)")
    for ax in (ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rootmap(data, path):
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 5.2))
    for ax, cfg in zip(axes, CONFIGS):
        by = data[cfg]
        A = [r for r in by["A"] if "error" not in r]
        D = [r for r in by["D"] if "error" not in r and r.get("conv_g")]
        ax.add_patch(
            Rectangle(
                (BOX_B[0], BOX_D[0]),
                BOX_B[1] - BOX_B[0],
                BOX_D[1] - BOX_D[0],
                fill=False,
                ec="crimson",
                lw=1.6,
                label="legacy box",
            )
        )
        ax.plot(
            [r["beta"] for r in A],
            [r["delta"] for r in A],
            "-o",
            c="0.55",
            ms=3,
            lw=0.8,
            label="A production (clamped)",
        )
        seg = [r["segment"] for r in D]
        sc = ax.scatter(
            [r["beta"] for r in D],
            [r["delta"] for r in D],
            c=seg,
            cmap="viridis",
            s=28,
            zorder=3,
            label="D hybr roots",
        )
        fig.colorbar(sc, ax=ax, label="segment")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\delta$")
        ax.set_title(cfg)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    data = {cfg: load(cfg) for cfg in CONFIGS}
    tables = ["# Phase 2.3 four-arm statistics", ""]
    for cfg in CONFIGS:
        t = md_table(cfg, data[cfg])
        print(t)
        tables.append(t)
    (HERE / "arms_stats.md").write_text("\n".join(tables))
    plot_summary(data, HERE / "arms_summary.png")
    plot_rootmap(data, HERE / "arms_rootmap.png")
    print("wrote arms_stats.md, arms_summary.png, arms_rootmap.png")


if __name__ == "__main__":
    main()
