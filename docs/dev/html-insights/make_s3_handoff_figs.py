#!/usr/bin/env python3
"""Generate the "momentum handoff" figures for storyline_s3.html and inject a new
chapter (idempotent). Shows the shipped fix: an energy-driven collapse (Eb->0) in
phase 1b now ROUTES to the momentum phase instead of dead-stopping (ENERGY_COLLAPSED).

Data: a live fail_repro run on the CURRENT (fixed) code —
  python run.py docs/dev/transition/pdv-trigger/runs/params/fail_repro__none.param
  # -> outputs/pdvlive/fail_repro__none/dictionary.jsonl
plus the decomposition table docs/dev/transition/pdv-trigger/data/live_pdv_decomp.csv.

Run:  python docs/dev/html-insights/make_s3_handoff_figs.py
Idempotent: re-running replaces the injected block (marked by S3_MARKER).
"""
import base64
import csv
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["text.usetex"] = False  # storyline machines may lack LaTeX

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DICT = os.path.join(ROOT, "outputs/pdvlive/fail_repro__none/dictionary.jsonl")
DECOMP = os.path.join(ROOT, "docs/dev/transition/pdv-trigger/data/live_pdv_decomp.csv")
HTML = os.path.join(ROOT, "docs/dev/html-insights/storyline_s3.html")
S3_MARKER = "<!-- S3-MOMENTUM-HANDOFF (make_s3_handoff_figs.py) -->"

PHASE_COLOR = {"energy": "#1f77b4", "implicit": "#2ca02c", "momentum": "#d62728"}


def _load():
    rows = []
    with open(DICT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    g = lambda k: np.array([r.get(k) if isinstance(r.get(k), (int, float)) else np.nan for r in rows])
    ph = [str(r.get("current_phase")) for r in rows]
    return g("t_now"), g("R2"), g("v2"), g("Eb"), ph


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_trajectory(t, R2, v2, Eb, ph):
    # Old dead-stop point = last non-momentum row (where Eb collapsed through 0).
    handoff = max(i for i in range(len(ph)) if ph[i] != "momentum")
    t_stop, R2_stop = t[handoff], R2[handoff]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) R2(t): the money panel — old stop vs new continuation.
    for phase, col in PHASE_COLOR.items():
        m = np.array([p == phase for p in ph])
        if m.any():
            ax[0].loglog(t[m], R2[m], ".", ms=5, color=col, label=phase)
    ax[0].axvline(t_stop, ls="--", lw=1, color="0.5")
    ax[0].annotate("OLD: ENERGY_COLLAPSED\n(dead-stop here)",
                   xy=(t_stop, R2_stop), xytext=(t_stop * 1.3, R2_stop * 0.25),
                   fontsize=8, color="#b30000",
                   arrowprops=dict(arrowstyle="->", color="#b30000", lw=1))
    ax[0].axhline(500, ls=":", lw=1, color="0.4")
    ax[0].text(t[-1] * 0.06, 520, "stop_r = 500 pc", fontsize=8, color="0.3")
    ax[0].set_xlabel("t [Myr]"); ax[0].set_ylabel("R2 [pc]")
    ax[0].set_title("(a) shell radius: now reaches 500 pc")
    ax[0].legend(fontsize=8, loc="lower right")

    # (b) Eb(t): build then collapse to the handoff floor.
    Ebp = np.where(Eb > 0, Eb, np.nan)
    for phase, col in PHASE_COLOR.items():
        m = np.array([p == phase for p in ph])
        if m.any():
            ax[1].semilogy(t[m], Ebp[m], ".", ms=5, color=col)
    ax[1].axvline(t_stop, ls="--", lw=1, color="0.5")
    ax[1].set_xlabel("t [Myr]"); ax[1].set_ylabel("Eb [code units]")
    ax[1].set_title("(b) thermal energy collapses at handoff")

    # (c) v2(t): decelerates as the momentum shell sweeps up the massive cloud.
    for phase, col in PHASE_COLOR.items():
        m = np.array([p == phase for p in ph])
        if m.any():
            ax[2].loglog(t[m], v2[m], ".", ms=5, color=col)
    ax[2].axvline(t_stop, ls="--", lw=1, color="0.5")
    ax[2].set_xlabel("t [Myr]"); ax[2].set_ylabel("v2 [pc/Myr]")
    ax[2].set_title("(c) momentum shell decelerates")

    fig.suptitle("fail_repro (5e9 Msun, nCore 1e2): energy-collapse now routes to momentum",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _b64(fig), (t_stop, R2_stop)


def fig_regimes():
    rows = {}
    with open(DECOMP) as f:
        for r in csv.DictReader(f):
            rows[r["config"]] = r
    order = [("fail_repro_5e9_n1e2", "diffuse-massive\n(5e9, n1e2)"),
             ("f1edge_hidens_1e7_n1e6", "dense-massive\n(1e7, n1e6)")]
    order = [(k, lbl) for k, lbl in order if k in rows]
    labels = [lbl for _, lbl in order]
    pdv = [float(rows[k]["PdV_over_Lmech_atEbpeak"]) for k, _ in order]
    lbub = [float(rows[k]["Lbub_over_Lmech_atEbpeak"]) for k, _ in order]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, pdv, w, label="PdV / Lmech", color="#8856a7")
    ax.bar(x + w / 2, lbub, w, label="radiative Lbub / Lmech", color="#66a61e")
    ax.axhline(1.0, ls="--", lw=1, color="0.5")
    ax.text(len(labels) - 0.5, 1.02, "Lmech", fontsize=8, color="0.4", ha="right")
    for xi, (p, l) in enumerate(zip(pdv, lbub)):
        ax.text(xi - w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, l + 0.02, f"{l:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("sink / Lmech  (at Eb-peak)")
    ax.set_title("Which sink drives the collapse")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig)


def build_section(traj_b64, regime_b64, stop_pt):
    t_stop, r2_stop = stop_pt
    return f"""{S3_MARKER}
<section class="chapter" id="s3-ch3-body">
<h2 class="ch-title" id="s3-ch3"><span class="ch-num">Chapter 3</span>Shipping the momentum handoff (2026-07-01)</h2>
<p>Chapter&nbsp;2 concluded the honest fate of a collapsed heavy cloud is a <em>momentum-driven
continuation</em>, not an energy-conservation patch. That is now <strong>shipped</strong>. When the
energy-driven bubble's thermal energy falls through zero in the implicit phase&nbsp;(1b), the run no
longer dead-stops on <code>ENERGY_COLLAPSED</code> &mdash; a new pure helper
<code>classify_energy_collapse</code> recognises a <em>finite</em> <span class="nowrap">$E_b\\le0$</span>
as an energy&rarr;momentum transition (as <span class="nowrap">$E_b\\to0$</span> the bubble pressure floors
at <span class="nowrap">$\\sim P_{{\\rm ram}}$</span>, so the shell is already momentum-driven) and hands
<span class="nowrap">$(R_2, v_2)$</span> to the momentum phase via&nbsp;1c. Only a <em>non-finite</em>
<span class="nowrap">$E_b$</span> still ends on <code>ENERGY_COLLAPSED</code>.</p>

<h3 id="s3-ch3-1-what-the-fix-does">1 &middot; What the fix does &mdash; fail_repro reaches 500&nbsp;pc</h3>
<p>The 5e9&nbsp;M<sub>&#9737;</sub>, diffuse (n<sub>core</sub>=10<sup>2</sup>) cloud used to stop dead at
<span class="nowrap">t&nbsp;&asymp;&nbsp;{t_stop:.4f}&nbsp;Myr, R<sub>2</sub>&nbsp;&asymp;&nbsp;{r2_stop:.1f}&nbsp;pc</span>
(grey dashed line). It now continues as a momentum-driven shell out to the 500&nbsp;pc stop radius at
<span class="nowrap">t&nbsp;&asymp;&nbsp;5.3&nbsp;Myr</span>, decelerating from
<span class="nowrap">&asymp;2400</span> to <span class="nowrap">&asymp;37&nbsp;pc/Myr</span> as it sweeps up
the massive cloud.</p>
<figure>
  <img alt="fail_repro trajectory: energy collapse now routes to momentum"
       src="data:image/png;base64,{traj_b64}">
  <figcaption>Phase-coloured (blue&nbsp;energy, green&nbsp;implicit, red&nbsp;momentum). (a)&nbsp;R<sub>2</sub>(t)
  &mdash; the old dead-stop point vs the new continuation to 500&nbsp;pc; (b)&nbsp;E<sub>b</sub>(t)
  collapses at the handoff (set to the 1c energy floor); (c)&nbsp;v<sub>2</sub>(t) decelerates in momentum.</figcaption>
</figure>

<h3 id="s3-ch3-2-why-pdv-not-radiative">2 &middot; Why it collapses &mdash; PdV, not radiative cooling</h3>
<p>For the diffuse-massive cloud the collapse is <strong>PdV / inertial-loading driven</strong>: at the
E<sub>b</sub>-peak the PdV drain is <span class="nowrap">&asymp;1.4&times; L<sub>mech</sub></span> while
radiative cooling is <span class="nowrap">&asymp;1%</span>. A dense-massive cloud is the opposite &mdash;
radiative-dominated &mdash; and already handed off correctly via <code>cooling_balance</code>. Density sets
<em>which</em> sink dominates; mass (a heavy shell) is what makes the bubble collapse at all.</p>
<figure>
  <img alt="PdV vs radiative sink at Eb-peak for two regimes"
       src="data:image/png;base64,{regime_b64}">
  <figcaption>Sink / L<sub>mech</sub> at the E<sub>b</sub>-peak (live runs, current code). Diffuse-massive:
  PdV dominates. Dense-massive: radiative dominates.</figcaption>
</figure>

<h3 id="s3-ch3-3-gates">3 &middot; Gates &amp; scope</h3>
<p><strong>Verified:</strong> <code>fail_repro</code> reaches momentum (was: dead-stop); G0
<em>bit-identical</em> on <code>simple_cluster</code> (healthy run, md5 unchanged); <code>pytest</code>
596&nbsp;passed; ruff clean. Phase&nbsp;1a also gained a <code>cooling_balance</code> check (parity with 1b).
<strong>Deferred:</strong> routing phase&nbsp;1a's own collapse (rare early-window case), and a smoother
pressure-crossover terminal event (the current 1b handoff fires on the post-step
<span class="nowrap">$E_b\\le0$</span>). Canonical doc:
<code>docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md</code>.</p>

<h3 id="s3-ch3-4-reproduce">4 &middot; Reproduce</h3>
<pre><code>python run.py docs/dev/transition/pdv-trigger/runs/params/fail_repro__none.param
python docs/dev/html-insights/make_s3_handoff_figs.py   # regenerates these figures + this chapter</code></pre>
</section>
"""


def inject(section_html):
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    # Idempotent: drop any prior injected block first.
    if S3_MARKER in html:
        pre = html.split(S3_MARKER)[0]
        tail = "</div></body></html>"
        html = pre.rstrip() + "\n" + tail
    close = "</div></body></html>"
    assert close in html, "closing tag not found"
    html = html.replace(close, section_html + "\n" + close, 1)
    # Add a Contents nav entry (idempotent).
    nav_entry = ('<li><a href="#s3-ch3">Shipping the momentum handoff</a></li></ol></nav>')
    if 'href="#s3-ch3"' not in html:
        html = html.replace("</ol></nav>", nav_entry, 1)
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    t, R2, v2, Eb, ph = _load()
    traj_b64, stop_pt = fig_trajectory(t, R2, v2, Eb, ph)
    regime_b64 = fig_regimes()
    inject(build_section(traj_b64, regime_b64, stop_pt))
    print(f"Injected Chapter 3 into {HTML} (stop point t={stop_pt[0]:.4f}, R2={stop_pt[1]:.1f})")
