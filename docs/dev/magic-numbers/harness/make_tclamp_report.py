#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a self-contained HTML report on the net_coolingcurve T-floor fix
(audit finding #1): measure-first, the dead-code verdict, and the bit-identical
file-tied floor that shipped.

Embeds figs/tclamp_*.png as base64 (single downloadable .html, works offline) and
renders LaTeX via MathJax (CDN). Figures come from make_tclamp_figures.py, which
is a pure read of the committed data/ artifacts.

    python docs/dev/magic-numbers/harness/make_tclamp_report.py
    -> docs/dev/magic-numbers/tclamp_report.html
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
OUT = os.path.join(HERE, "..", "tclamp_report.html")


def img(name, alt):
    p = os.path.join(FIGS, name)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'<figcaption>{alt}</figcaption></figure>')


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRINITY — the cooling-table T-floor story (audit finding #1)</title>
<script>
MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root{--ink:#1a1a1a;--mut:#555;--acc:#1b5e20;--box:#e8f5e9;--line:#ddd;--code:#f4f4f4;}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
       max-width:960px;margin:0 auto;padding:2.2rem 1.4rem 5rem;line-height:1.6;font-size:16px;}
  h1{font-size:1.95rem;line-height:1.25;margin:.2rem 0 .2rem;}
  h2{font-size:1.4rem;margin:2.6rem 0 .6rem;border-bottom:2px solid var(--acc);padding-bottom:.25rem;color:var(--acc);}
  h3{font-size:1.13rem;margin:1.7rem 0 .4rem;}
  .sub{color:var(--mut);font-size:1.05rem;margin:.1rem 0 1.2rem;}
  .tldr{background:var(--box);border:1px solid var(--acc);border-radius:8px;padding:1rem 1.2rem;margin:1.4rem 0;}
  code,kbd{background:var(--code);padding:.1em .35em;border-radius:4px;font-size:.9em;}
  pre{background:#f7f7f7;border:1px solid var(--line);padding:.8rem 1rem;border-radius:6px;overflow-x:auto;font-size:.84em;line-height:1.45;}
  .add{color:#1b5e20;} .del{color:#b71c1c;}
  table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem;}
  th,td{border:1px solid var(--line);padding:.45rem .6rem;text-align:left;vertical-align:top;}
  th{background:#f0f4f0;}
  figure{margin:1.6rem 0;text-align:center;}
  img{max-width:100%;border:1px solid var(--line);border-radius:6px;}
  figcaption{color:var(--mut);font-size:.85rem;margin-top:.4rem;font-style:italic;text-align:left;}
  .lesson{border-left:4px solid var(--acc);background:#fafdfa;padding:.6rem 1rem;margin:1rem 0;}
  .warn{border-left:4px solid #b71c1c;background:#fffafa;padding:.6rem 1rem;margin:1rem 0;}
  .meta{color:var(--mut);font-size:.85rem;}
  ul li{margin:.2rem 0;}
  .ok{color:var(--acc);font-weight:600;} .no{color:#b71c1c;font-weight:600;}
</style></head><body>

<h1>TRINITY — the cooling-table temperature-floor story</h1>
<p class="sub">Audit finding&nbsp;#1: a hard-coded <code>if T &lt; 1e4: T = 1e4</code> floor in the
bubble cooling hot path. The full arc — measure first, find it is dead code on a false premise, then
ship a file-tied floor that is provably bit-identical on every reachable state.</p>
<p class="meta">Generated from committed data &amp; figures
(<code>docs/dev/magic-numbers/{data,figs}/</code>) · branch
<code>bugfix/high-feedback-problems-more-detailed</code> · TRINITY (<code>trinity-sf</code>).
Reproduce: <code>python docs/dev/magic-numbers/harness/make_tclamp_report.py</code>.</p>

<div class="tldr">
<b>TL;DR.</b> <code>get_dudt</code> (the per-RHS cooling rate in the bubble-structure ODE) floored any
temperature below $10^4$&nbsp;K up to $10^4$&nbsp;K, with a comment admitting the constant was a guess and
that the table "only [goes] to 3.99". Both halves were wrong: the non-CIE table actually reaches
<b>3162&nbsp;K</b> ($\log_{10}T=3.5$), so $10^4$ over-floored the whole valid decade $[3162,10^4)$&nbsp;K —
and <b>across 9.46&nbsp;million</b> <code>get_dudt</code> calls over four regimes (baseline + two feedback
edges + the stiffest LSODA flood) the branch <b>fired exactly 0 times</b> (min&nbsp;T ever&nbsp;=&nbsp;30&nbsp;000&nbsp;K).
It was <b>dead code</b>. The fix ties the floor to the file —
$\log_{10}T &lt; T_{\min}\Rightarrow T\leftarrow 10^{T_{\min}}$ — which is <b>provably inert</b> on every
reachable state (per-call <b>576/576 bit-identical</b> for $T\ge10^4$; a full-run <b>byte-identical</b>
<code>dictionary.jsonl</code>) yet strictly more correct, and more robust, below the table edge.
</div>

<h2>0. Background — what the floor guards</h2>
<p>For each trial in the bubble-structure solve, TRINITY evaluates a net cooling rate
$\dot u(n,T,\phi)$ by switching between two tabulated regimes on temperature:</p>
$$\dot u(n,T,\phi)=\begin{cases}
-\,\Lambda_{\rm nonCIE}(n,T,\phi) & T_{\min}\le \log_{10}T\le T_{\rm cut}\quad(\text{non-CIE table})\\[2pt]
\text{interp}(\,\cdot\,) & T_{\rm cut}<\log_{10}T<T_{\rm CIE}\\[2pt]
-\,\chi_e\,n^2\,\Lambda_{\rm CIE}(T) & \log_{10}T\ge T_{\rm CIE}\quad(\text{CIE table})\\[2pt]
\textbf{raise} & \log_{10}T<T_{\min}\quad(\text{below the table})
\end{cases}$$
<p>The non-CIE grid spans $T_{\min}=3.5$ to $T_{\rm cut}=5.5$ in $\log_{10}T$ (i.e. <b>3162&nbsp;K to
316&nbsp;000&nbsp;K</b>). If the integrator ever hands <code>get_dudt</code> a $T$ below the table, the last
branch raises and the run dies — so a floor exists to keep $T$ in range. The original floor was:</p>
<pre><span class="del"># just a gate for limit
# TODO: this in the future has to depend on the file. It should
# be set such that it follows the minumu temperature of the cooling file.
# ... the temperature seem to run at some very low value (~1e3.91) and the
# lowest available value of the cooling file ... is only until 3.99. ...
if T &lt; 1e4:
    T = 1e4</span></pre>
<p>This is the classic "unjustified constant" smell — a magic number admitted to be a guess, on a
hot path, with the author unsure why it was needed. The audit's rule: <b>measure before fixing.</b></p>

<h2>1. The problem — an over-floor on a false premise</h2>
<p>Two facts collide. <b>(a)</b> the table's true minimum is $3162$&nbsp;K ($\log_{10}T=3.5$), not the
"3.99" the comment claims — so flooring to $10^4$&nbsp;K throws away a full valid decade of cooling physics,
replacing any $T\in[3162,10^4)$&nbsp;K with a rate evaluated $2.5\times$ hotter. <b>(b)</b> the comment's
worry that "the temperature seem[s] to run at $\sim10^{3.91}$" is the justification for the whole guard —
but if that never actually happens, the guard is inert. The geography:</p>
__FIG_SCHEMATIC__
<p>So there are two empirical questions a measurement must answer before any fix: <i>does the bubble ODE
ever drive $T$ below $10^4$ (does the clamp fire)?</i> and <i>does it ever go below the true table edge
$3162$&nbsp;K (is the raise ever in play)?</i> Everything downstream depends on the answers.</p>

<h2>2. The ideas — a method matrix (and a hypothesis to retract)</h2>
<p>The comment proposes one hypothesis (cold gas at $10^{3.91}$); the fix space has three options. Rather
than pick by argument, the plan was to <b>instrument the real function</b> and let the counters decide,
then gate whichever fix the data justified.</p>
<table>
<tr><th>idea / option</th><th>what it is</th><th>why considered</th><th>verdict from the data</th></tr>
<tr><td><b>H — the comment's hypothesis</b></td><td>$T$ runs cold (~$10^{3.91}$) and the floor saves a
real sub-table excursion</td><td>it is the stated reason the guard exists</td>
<td class="no">retracted — min&nbsp;T&nbsp;ever&nbsp;=&nbsp;30&nbsp;000&nbsp;K; never observed in any regime</td></tr>
<tr><td><b>1 — file-tied floor</b> (the TODO)</td><td>$\log_{10}T&lt;T_{\min}\Rightarrow T\leftarrow
10^{T_{\min}}$ (clamp to the real table edge)</td><td>removes the magic number; correct table coverage;
still prevents the raise</td><td class="ok">SHIPPED — provably inert on all reachable $T$, more correct
below</td></tr>
<tr><td>2 — leave + document</td><td>fix only the wrong comment; change nothing executable</td>
<td>smallest possible change</td><td>rejected — leaves the magic number and the over-floor in place</td></tr>
<tr><td>3 — remove entirely</td><td>delete the floor</td><td>it is dead code, so deletion is tempting</td>
<td>rejected — drops the raise-guard insurance for any future table/regime that <i>could</i> overshoot</td></tr>
</table>
<p>Option&nbsp;3 deserves a beat of its own, because "it's dead code, just delete it" is the seductive wrong
answer: the floor is dead <i>on the configs we run today</i>, but it is also the only thing standing between a
future sub-table $T$ and a hard crash. Tying it to the file keeps the insurance while making it correct.
The new floor is even <b>strictly more robust</b> than the old one — the old $10^4$ constant would itself
have raised on any cooling table whose minimum exceeds $10^4$&nbsp;K; $10^{T_{\min}}$ cannot.</p>

<h2>3. Working the chosen idea across regimes — the measurement</h2>
<p>A non-invasive instrument wrapped <code>get_dudt</code> and accumulated, per call, a $\log_{10}T$
histogram and counters for $T&lt;10^4$, $T&lt;3162$, and the accepted-profile minima. It was run on the four
must-test regimes — <code>simple_cluster</code> (the energy-driven baseline, a full run), the two
<code>f1edge</code> clouds that span feedback strength&nbsp;&times;&nbsp;density, and a stiff
high-density LSODA-flood — the stiffest case in scope, the one most likely to drive $T$ low. The temperature
the RHS actually sees, across all four:</p>
__FIG_HIST__
<p>Every regime's distribution hits a <b>hard wall at $\log_{10}T=4.45$</b> and stops. Nothing — not one of
9.46&nbsp;million evaluations — lands below it. The old floor ($\log_{10}T=4.0$) and the table edge
($3.5$) sit in a region the integrator simply never visits. Reading the minimum per config makes the margin
explicit:</p>
__FIG_MINT__
<p>Each regime bottoms out at exactly the $3\times10^4$&nbsp;K outer boundary — $3\times$ above the old
floor, $9.5\times$ above the raise boundary. (The handful of "accepted&nbsp;&lt;&nbsp;$3\times10^4$" samples
are all at $29\,999.99\dots$&nbsp;K — floating-point dust on the boundary, not a real excursion.) The
hypothesis $H$ is dead: the clamp is <b>dead code in every tested regime</b>.</p>

<h2>4. The measurement, in one number</h2>
<p>The headline is a ratio of two very different magnitudes:</p>
__FIG_FIRES__
<p><b>9,459,458 calls&nbsp;&rarr;&nbsp;0 clamp fires</b>, and <b>0</b> calls ever reached the table minimum
(so the guarded <code>raise</code> was never in play either). Because the $T&lt;10^4$ branch is never taken,
the question "which floor value is correct?" is <i>moot for results</i>: flooring to $10^4$, to the table
edge, or not at all are <b>bit-identical by construction</b> on every regime measured — the differing code
path is unreachable. That is what makes the fix a zero-risk correctness cleanup rather than a
physics-changing edit.</p>

<h2>5. The solution — and why it is correct</h2>
<p>The change is two hunks in <code>get_dudt</code>: delete the magic-number floor and its wrong comment,
and add a floor tied to the table minimum, placed <i>after</i> the cutoffs are computed so $T_{\min}$ is in
scope:</p>
<pre><span class="del">-    # just a gate for limit
-    # TODO: ... follow the minumu temperature of the cooling file ...
-    if T &lt; 1e4:
-        T = 1e4</span>
     nonCIE_Tcutoff, nonCIE_Tmin = _noncie_cutoffs(cooling_nonCIE)
     CIE_Tcutoff = _cie_tcutoff(logT_CIE)
<span class="add">+    # clamp a sub-table T up to the cooling file's minimum tabulated T so it
+    # degrades to the table edge via the non-CIE branch instead of the raise.
+    if np.log10(T) &lt; nonCIE_Tmin:
+        T = 10**nonCIE_Tmin</span></pre>
<p>The $10^{x}\!\to\!\log_{10}$ round-trip is exact for the bundled grid min ($\log_{10}(10^{3.5})=3.5$), so
the clamped value lands on the non-CIE branch's lower edge — no raise. What changes, and what does not:</p>
__FIG_OVERLAY__
<p>For every $T\ge10^4$ — the entire range any run reaches — old and new are <b>identical</b> (neither
clamps; the curves lie on top of each other). They differ only in the never-visited sub-$10^4$ decade,
where the new floor follows the real cooling rate down to the table edge instead of pinning the whole decade
to one $10^4$ value. The fix is therefore bit-identical where it runs and more correct where it doesn't.</p>

<h2>6. The validation journey, and the takeaway</h2>
<p>A change to a solver hot loop does not get to claim "inert" from a per-call check alone — that is
necessary but <b>not sufficient</b> for an iterative path, because a tiny per-call difference can compound
over a full integration. The gate ladder used here:</p>
__FIG_LADDER__
<ul>
<li><b>Measure first</b> — 9.46M calls, 4 regimes, the instrument above: clamp fires 0&times;, min&nbsp;T&nbsp;=&nbsp;30&nbsp;000&nbsp;K.</li>
<li><b>Per-call equivalence</b> vs <code>git show HEAD</code> over a $T$ grid spanning all branches:
<b>576/576 bit-identical</b> for $T\ge10^4$, the 144 sub-$10^4$ points diverge <i>by design</i>, and
<b>neither version raises</b>. Necessary, not sufficient.</li>
<li><b>Full-run byte-identity</b> — a capped <code>simple_cluster</code> run on the new code and on HEAD, in
<b>separate processes</b> (trinity leaks module-level globals in-process) at <b>matched simulation time</b>
(a fixed <code>stop_t</code> makes both truncate at the same&nbsp;$t$): identical
<code>dictionary.jsonl</code> sha256 across <b>169 snapshots</b>.</li>
<li><b>Suite + lint</b> — 574 tests pass (incl. three new <code>test_net_coolingcurve.py</code> cases pinning
the clamp-to-edge behaviour), ruff bug-class clean.</li>
</ul>
<div class="lesson"><b>Takeaway.</b> "Measure before fixing" turned a scary-looking magic number with a
worried TODO into a five-minute verdict: the guard was dead code on a false premise. The right move was
neither to trust the comment nor to delete the guard, but to make it <i>correct and file-tied</i> — and to
prove, with a full-run byte-identity in separate processes at matched&nbsp;$t$, that "provably inert" was
earned, not assumed.</p>

<p class="meta" style="margin-top:3rem">Sources — measurement &amp; gates:
<code>docs/dev/magic-numbers/TCLAMP_PLAN.md</code> (the three-banner plan + gate table) and
<code>docs/dev/magic-numbers/AUDIT.md</code> (finding&nbsp;#1). Data:
<code>docs/dev/magic-numbers/data/{simple_cluster,f1edge_lowdens,f1edge_hidens,conduction_stiff}_summary.json</code>
+ <code>tclamp_dudt_overlay.csv</code>. Harness:
<code>harness/{tclamp_instrument,verify_tclamp_equiv,make_tclamp_overlay_data,make_tclamp_figures}.py</code>,
gate fixture <code>harness/simple_cluster_capped.param</code>, tests
<code>test/test_net_coolingcurve.py</code>. Fix shipped in the cooling hot path
<code>trinity/cooling/net_coolingcurve.py</code> (commit <code>cc8ae76</code>).</p>

</body></html>
"""

HTML = HTML.replace("__FIG_SCHEMATIC__", img(
    "tclamp_schematic.png",
    "Figure 1. The cooling-table temperature axis. The non-CIE table covers [3162, 316000] K "
    "(log 3.5-5.5); below 3162 K get_dudt raises. The OLD floor (10^4 K, log 4.0) lifted the entire valid "
    "[3162, 10^4) K decade up to 10^4; the NEW floor sits at the true table edge (3162 K). The green band is "
    "where the bubble ODE actually lives - measured min T = 30000 K. Both of the comment's premises "
    "(T~10^3.91; 'table only to 3.99') are marked wrong."))
HTML = HTML.replace("__FIG_HIST__", img(
    "tclamp_temperature_histogram.png",
    "Figure 2. log10(T) histogram of every get_dudt RHS evaluation, all four regimes overlaid. Each "
    "distribution hits a hard wall at log T = 4.45 and stops - 0 of 9.46M calls fall below it. The old floor "
    "(4.0) and the table edge (3.5) lie in a region the integrator never visits."))
HTML = HTML.replace("__FIG_MINT__", img(
    "tclamp_minT_by_config.png",
    "Figure 3. The minimum T each regime ever passed to get_dudt. All four bottom out at the 3x10^4 K "
    "boundary - 3x above the old floor, 9.5x above the raise boundary. The red band (T < 10^4, where the "
    "clamp would fire) is never entered."))
HTML = HTML.replace("__FIG_FIRES__", img(
    "tclamp_calls_vs_fires.png",
    "Figure 4. Effort vs outcome: 9,459,458 get_dudt calls across four regimes, and the T<10^4 clamp fired "
    "0 times in every one. No call reached the table min either, so the guarded raise was never in play. The "
    "benign boundary hair (accepted solves at ~29999.99 K) is noted below."))
HTML = HTML.replace("__FIG_OVERLAY__", img(
    "tclamp_dudt_overlay.png",
    "Figure 5. get_dudt(T) under the old 10^4 floor (red) vs the new table-edge floor (green). (a) full "
    "view: identical for all T >= 10^4 (the curves coincide), divergent only below. (b) zoom: in the "
    "over-floored decade the old floor flatlines the whole [3162, 10^4) K range to one value while the new "
    "floor tracks the real rate. The grey/shaded region (T < 30000 K) is never reached by any run."))
HTML = HTML.replace("__FIG_LADDER__", img(
    "tclamp_validation_ladder.png",
    "Figure 6. The validation ladder: measure first (clamp fires 0x) -> per-call equivalence (576/576 "
    "bit-identical for T>=10^4; necessary but NOT sufficient) -> full-run byte-identity (identical sha256, "
    "separate processes, matched-t) -> suite + lint -> ship."))

assert "__FIG_" not in HTML, "unreplaced figure placeholder remains"
open(OUT, "w").write(HTML)
print("wrote", os.path.normpath(OUT), f"({os.path.getsize(OUT)//1024} KB)")
