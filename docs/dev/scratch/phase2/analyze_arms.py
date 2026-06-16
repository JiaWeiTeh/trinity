#!/usr/bin/env python3
"""Phase 2.3 four-arm diagnostics: aggregate stats + plots from arms_*.jsonl.

Reads the per-arm/per-segment jsonl emitted by arms.py and produces, for each
config:
  - a markdown stats table (stdout + arms_stats.md)
  - arms_summary.png : convergence and evaluation-count bars, G2 gates drawn
  - arms_rootmap.png : (beta, delta) maps -- production (arm A) vs hybr (arm D)
                       converged roots against the legacy box
  - arms_residual.png: residual g at each segment's accepted point, per arm vs
                       time -- where each arm wins or loses along the phase
  - arms_pareto.png  : convergence vs cost per arm with the G2 pass region --
                       the promotion decision in one view

Pure re-read of the jsonl; reruns are cheap and side-effect-free. Numbers are
only as good as the jsonl on disk -- regenerate after any new arms run.

Usage: python docs/dev/scratch/phase2/analyze_arms.py
"""

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

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
THRESH = 1e-4  # residual threshold for f and g (caption only)
# outcome stack colours: converged / not-converged / no-root handoff (gate).
# the "abort" is the acceptance gate firing (no physical root -> hand off to
# transition), NOT a solver failure -- so it is amber, not red, and the
# converged-among-solvable tick below shows the handoff-adjusted convergence.
C_CONV, C_NC, C_AB = "#2ca02c", "0.72", "#E69F00"
ARM_COLOR = {"A": "0.5", "B": "#1f77b4", "C": "#ff7f0e", "D": "#2ca02c"}
CFG_MARK = {"arms_mock4e3": "o", "arms_simple1e5": "s"}

# apply trinity's house style (paper/_lib/trinity.mplstyle). It sets
# text.usetex=True for paper figures, but scratch diagnostics must render without
# a LaTeX install, so force usetex off -- mathtext draws the $\beta$/$\delta$
# labels fine.
_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


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
        ab = f"{d['aborts']}" + (
            f" ({', '.join(f'{k}:{v}' for k, v in cats.items())})" if cats else ""
        )
        oob = "-" if a in ("A", "B") else str(d["oob"])
        lines.append(
            f"| {ARM_LABEL[a]} | {n} | {cf} | {cg} | {sh} | "
            f"{d['med_ev']:.0f} | {d['max_ev']} | {ab} | {oob} |"
        )
    lines.append("")
    return "\n".join(lines)


def _bucket(by, a):
    """(n, converged_g, not_converged, aborted, converged_f, med_ev) for one arm."""
    d = stats(by[a])[0]
    n, cv, ab = d["segs"], d["conv_g"], d["aborts"]
    return n, cv, n - cv - ab, ab, d["conv_f"], d["med_ev"]


def plot_summary(data, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    x = list(range(len(ARMS)))
    w = 0.38
    for k, cfg in enumerate(CONFIGS):
        by = data[cfg]
        xs = [i + (k - 0.5) * w for i in x]
        pcv, pnc, pab, pcf, ev, ns, cas, abn = [], [], [], [], [], [], [], []
        for a in ARMS:
            n, cv, nc, ab, cf, med = _bucket(by, a)
            pcv.append(100 * cv / n)
            pnc.append(100 * nc / n)
            pab.append(100 * ab / n)
            pcf.append(100 * cf / n)
            ev.append(med)
            ns.append((cv, n))
            cas.append(100 * cv / (n - ab) if (n - ab) > 0 else 0.0)  # conv among solvable
            abn.append(ab)
        f0 = k == 0
        # stacked outcomes: converged / not-converged / aborted (= 100% of segments)
        ax1.bar(xs, pcv, w, color=C_CONV, label="converged (g)" if f0 else None)
        ax1.bar(xs, pnc, w, bottom=pcv, color=C_NC, label="not converged" if f0 else None)
        bot = [c + m for c, m in zip(pcv, pnc)]
        ax1.bar(
            xs,
            pab,
            w,
            bottom=bot,
            color=C_AB,
            hatch="//",
            label="no-root handoff (gate)" if f0 else None,
        )
        for j, (xi, cf) in enumerate(zip(xs, pcf)):  # "also converged under f" tick
            ax1.plot(
                [xi - w / 2, xi + w / 2],
                [cf, cf],
                c="k",
                lw=1.3,
                label="also converged (f)" if (f0 and j == 0) else None,
            )
        # converged-among-solvable tick: aborts are no-root handoffs, not failures,
        # so this is the gate-relevant convergence (= 100% for D on both configs)
        for j, (xi, c_as, ab) in enumerate(zip(xs, cas, abn)):
            if ab > 0:
                ax1.plot(
                    [xi - w / 2, xi + w / 2],
                    [c_as, c_as],
                    c="#0072B2",
                    lw=2.0,
                    zorder=6,
                    label="converged / solvable" if (f0 and j == ARMS.index("D")) else None,
                )
        for xi, (cv, n) in zip(xs, ns):  # converged count atop each bar
            ax1.text(xi, 101, f"{cv}/{n}", ha="center", va="bottom", fontsize=6.5)
        ax2.bar(xs, ev, w, label=cfg.replace("arms_", ""))
        for a, xi, e in zip(ARMS, xs, ev):
            ax2.text(xi, e * 1.05, f"{e:.0f}", ha="center", va="bottom", fontsize=7)
            if a == "C" and e > 100:  # arm C hit its 240 s rescanning budget
                ax2.annotate(
                    "≈ full 240 s\nC budget/seg",
                    xy=(xi, e),
                    xytext=(xi, e * 2.2),
                    fontsize=6.5,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7),
                )
    ax1.axhline(G2_CONV, ls="--", c="k", lw=1, label=f"G2 gate {G2_CONV:.0f}%")
    ax1.set_ylim(0, 134)  # modest headroom above the 100% bars for the legend
    ax1.set_ylabel("% of sampled segments")
    ax1.set_title("Convergence per arm (stacked outcomes)")
    # the eval gate is specified for C/D only; A/B are the production-cost baseline
    ax2.plot(
        [1.5, 3.5],
        [G2_EVALS, G2_EVALS],
        ls="--",
        c="k",
        lw=1,
        label=f"G2 gate {G2_EVALS:.0f} (C/D only)",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("median evaluations / segment (log)")
    ax2.set_title("Cost per arm (lower better)")
    for ax in (ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], rotation=15)
        ax.grid(axis="y", alpha=0.3)
    # panel 1 bars fill 0-100%, so its legend sits in the headroom band above them
    ax1.legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.95)
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.suptitle(
        "Phase 2.3 four-arm shadow experiment — metric g, threshold "
        f"{THRESH:g}; bars: left = mock4e3, right = simple1e5  "
        "(amber = no-root handoff, a correct gate outcome, not a failure)",
        fontsize=9.5,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rootmap(data, path):
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        by = data[cfg]
        A = {r["segment"]: r for r in by["A"] if "error" not in r and "beta" in r}
        D = [r for r in by["D"] if "error" not in r and r.get("conv_g")]
        n_abort = sum(1 for r in by["D"] if "error" in r)
        allseg = list(A) + [r["segment"] for r in D]
        vmin, vmax = min(allseg), max(allseg)
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
        # clamp-error connectors: production point -> true root, same segment
        for j, r in enumerate(D):
            a = A.get(r["segment"])
            if a is not None:
                ax.plot(
                    [a["beta"], r["beta"]],
                    [a["delta"], r["delta"]],
                    c="0.6",
                    lw=0.6,
                    alpha=0.5,
                    zorder=1,
                    label="clamp error (A→D)" if j == 0 else None,
                )
        ax.scatter(
            [r["beta"] for r in A.values()],
            [r["delta"] for r in A.values()],
            c=list(A),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="s",
            s=18,
            edgecolor="0.3",
            lw=0.3,
            zorder=2,
            label="A production (clamped)",
        )
        sc = ax.scatter(
            [r["beta"] for r in D],
            [r["delta"] for r in D],
            c=[r["segment"] for r in D],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=34,
            edgecolor="k",
            lw=0.3,
            zorder=3,
            label="D hybr roots",
        )
        fig.colorbar(sc, ax=ax, label="segment")
        if n_abort:
            ax.text(
                0.97,
                0.03,
                f"{n_abort} segments aborted\n(no root located)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round", fc="mistyrose", ec="0.6", alpha=0.9),
            )
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\delta$")
        ax.set_title(cfg.replace("arms_", ""))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Converged roots vs clamped production (metric g): production "
        "rides the box edge; the true roots leave it",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residual_trace(data, path):
    """Residual g at each segment's accepted point, per arm, along the phase.

    Shows *where* each arm wins or loses, which the aggregate bars cannot: on
    the mock A/B/C ride above the threshold the whole phase while D sits on the
    floor (converged g floored at 1e-6 for display); D's aborts are marked.
    """
    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(9, 9.5), sharey=True)
    for k, (ax, cfg) in enumerate(zip(axes, CONFIGS)):
        by = data[cfg]
        for a in ARMS:
            ok = sorted((r for r in by[a] if "error" not in r), key=lambda r: r["t_now"])
            if not ok:
                continue
            ax.plot(
                [r["t_now"] for r in ok],
                [max(r["g"], 1e-6) for r in ok],
                "-o",
                ms=3,
                lw=2.0 if a == "D" else 1.0,
                color=ARM_COLOR[a],
                zorder=4 if a == "D" else 2,
                label=ARM_LABEL[a] if k == 0 else None,
            )
        for j, r in enumerate(r for r in by["D"] if "error" in r):  # D aborts
            ax.axvline(
                r["t_now"],
                color=C_AB,
                ls=":",
                lw=1,
                alpha=0.5,
                label="D abort (no root)" if (k == 0 and j == 0) else None,
            )
        ax.axhline(
            THRESH, color="k", ls="--", lw=1.2, label=f"threshold {THRESH:g}" if k == 0 else None
        )
        ax.set_yscale("log")
        ax.set_ylim(4e-7, 1e2)
        ax.set_xlabel(r"implicit-phase time $t$ [Myr]")
        ax.set_ylabel(r"residual $g$ at accepted $(\beta,\delta)$")
        ax.set_title(cfg.replace("arms_", ""))
        ax.grid(alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(
        r"Per-segment residual by arm — only D (hybr) drives $g$ below threshold; "
        r"A/B/C ride above it",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pareto(data, path):
    """Convergence vs cost per arm: the G2 promotion decision in one view.

    x = median evals/segment (log, lower better), y = convergence under g.
    The shaded corner is the G2 pass region (conv >= 80% AND <= 15 evals);
    arms are colored, configs are marker shapes. Only D reaches the corner.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.add_patch(
        Rectangle((8, G2_CONV), G2_EVALS - 8, 100 - G2_CONV, fc="#d5efd5", ec="none", zorder=0)
    )
    ax.axvline(G2_EVALS, color="k", ls="--", lw=1)
    ax.axhline(G2_CONV, color="k", ls="--", lw=1)
    for cfg in CONFIGS:
        by = data[cfg]
        for a in ARMS:
            d = stats(by[a])[0]
            ev, conv = d["med_ev"], 100 * d["conv_g"] / d["segs"]
            ax.scatter(
                ev,
                conv,
                marker=CFG_MARK[cfg],
                s=110,
                color=ARM_COLOR[a],
                edgecolor="k",
                lw=0.6,
                zorder=4,
            )
            ax.annotate(a, (ev, conv), textcoords="offset points", xytext=(7, 4), fontsize=9)
            ab = d["aborts"]
            if ab > 0:  # aborts are no-root handoffs -> show handoff-adjusted conv
                cas = 100 * d["conv_g"] / (d["segs"] - ab)
                ax.plot([ev, ev], [conv, cas], color=ARM_COLOR[a], ls=":", lw=1.0, zorder=3)
                ax.scatter(
                    ev,
                    cas,
                    marker=CFG_MARK[cfg],
                    s=110,
                    facecolor="none",
                    edgecolor=ARM_COLOR[a],
                    lw=1.6,
                    zorder=4,
                )
    ax.set_xscale("log")
    ax.set_xlim(8, 220)
    ax.set_ylim(-6, 108)
    ax.set_xlabel("median evaluations / segment  (log — lower better)")
    ax.set_ylabel(r"convergence under $g$  (%)")
    ax.set_title("Convergence vs cost per arm — upper-left is better")
    arm_h = [
        Line2D([], [], marker="o", ls="", color=ARM_COLOR[a], mec="k", label=ARM_LABEL[a])
        for a in ARMS
    ]
    cfg_h = [
        Line2D(
            [], [], marker=CFG_MARK[c], ls="", color="0.5", mec="k", label=c.replace("arms_", "")
        )
        for c in CONFIGS
    ]
    solv_h = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec="0.3",
            mew=1.5,
            label="○ conv / solvable\n(aborts = handoffs)",
        )
    ]
    ax.legend(
        handles=arm_h + cfg_h + solv_h + [Patch(fc="#d5efd5", ec="k", label="G2 pass region")],
        fontsize=8,
        loc="center right",
        ncol=1,
    )
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
    plot_residual_trace(data, HERE / "arms_residual.png")
    plot_pareto(data, HERE / "arms_pareto.png")
    print(
        "wrote arms_stats.md, arms_summary.png, arms_rootmap.png, "
        "arms_residual.png, arms_pareto.png"
    )


if __name__ == "__main__":
    main()
