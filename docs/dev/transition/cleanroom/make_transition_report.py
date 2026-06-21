#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single self-contained, light-mode, MathJax-rendered HTML report that
tells the implicit -> momentum transition-trigger investigation end to end:
identify the problem -> the ideas tested -> the chosen idea across configs ->
the measurements -> the solution -> the validation arc.

The 5 PNGs in figures/ are embedded as base64 so the file is standalone
(downloadable, opens offline; MathJax loads from CDN for the formulas). Every
figure is a pure read of a committed CSV in data/ (built by the plot_*.py
generators); every table number traces to those same CSVs.

Content blocks are plain strings with __FIG_*__ placeholders (LaTeX braces stay
literal); the placeholders are swapped for base64 <img> tags before writing.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/transition/cleanroom/plot_fret.py   docs/dev/transition/cleanroom/data/c0_*_st6.csv
    python docs/dev/transition/cleanroom/plot_f0path.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
    python docs/dev/transition/cleanroom/plot_g0.py     docs/dev/transition/cleanroom/data/c0_*_h0.csv
    python docs/dev/transition/cleanroom/plot_cert.py   docs/dev/transition/cleanroom/data/c0_*_st6.csv
    python docs/dev/transition/cleanroom/plot_beta.py   docs/dev/transition/cleanroom/data/c0_*_st6.csv
    python docs/dev/transition/cleanroom/make_transition_report.py   # -> transition_report.html
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = HERE / "transition_report.html"

FIGURES = {
    "__FIG_F0__": ("f0_pathology.png", "cooling ratio vs the 0.05 threshold, with mechanical luminosity"),
    "__FIG_BETA__": ("beta_repressurization.png", "beta(t) per config with negative-beta shaded over mechanical luminosity"),
    "__FIG_SURGE__": ("surge_coincidence.png", "per-config correlation of the cooling-ratio change with feedback, beta and delta"),
    "__FIG_PORTRAIT__": ("betadelta_portrait.png", "delta-beta phase portrait of all implicit rows coloured by time"),
    "__FIG_DIP__": ("dip_drivers.png", "cooling ratio, Lgain/Lloss, and wind/SN feedback vs time, three panels"),
    "__FIG_G0__": ("g0_divergence.png", "timeline per config showing where each candidate family would fire"),
    "__FIG_FRET__": ("fret_verdict.png", "retained-energy fraction vs time for all six configs against the observed band"),
    "__FIG_BLOW__": ("blowout_geometric.png", "blowout epoch vs cloud radius, one point per config"),
    "__FIG_MIX__": ("mixcool_rootfix.png", "retained energy and cooling ratio under a theta mixing-layer sink"),
    "__FIG_CERT__": ("cert_residuals.png", "res_beta and res_T0_struct vs time for all six configs"),
    "__FIG_DIPMECH__": ("dip_mechanism.png", "Lloss, emission-measure proxy and T0 through the early dip"),
    "__FIG_BEFOREAFTER__": ("before_after.png", "cooling trigger vs time, legacy (crosses) vs hybr (never crosses)"),
    "__FIG_LEGHYBR__": ("legacy_vs_hybr.png", "legacy vs hybr through the dip: ratio, Lloss, beta, three configs"),
    "__FIG_PDV__": ("pdv_trigger.png", "input partition (cooling/work/net) and F0 vs F0+PdV trigger ratios"),
    "__FIG_PDVMASS__": ("pdv_massspectrum.png", "Eb collapse for a 5e9 cluster and max PdV/Lmech per config"),
    "__FIG_LEGHYBREXTRA__": ("legacy_vs_hybr_extra.png", "legacy vs hybr: delta, beta+delta, Eb, Pb through the dip"),
    "__FIG_CAUSAL__": ("dip_causalorder.png", "causal ordering of the dip turnovers: v2, Lloss, ratio, Pb, T0"),
}


def img(name, alt):
    b64 = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


CSS = """
:root{--ink:#1f2733;--mut:#5b6675;--line:#e3e8ef;--bg:#ffffff;--panel:#f7f9fc;
--accent:#2c6fb3;--hyp:#7048e8;--find:#2f9e44;--warn:#e8842a;--bad:#d1495b;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:880px;margin:0 auto;padding:40px 22px 90px;}
h1{font-size:30px;line-height:1.22;margin:0 0 6px;letter-spacing:-.01em;}
h2{font-size:22px;margin:46px 0 12px;padding-top:10px;border-top:1px solid var(--line);}
h3{font-size:17px;margin:26px 0 8px;color:var(--accent);}
.sub{color:var(--mut);font-size:15px;margin:0 0 22px;}
p{margin:11px 0;} a{color:var(--accent);}
code{background:var(--panel);border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font:13.5px/1.4 "SFMono-Regular",Consolas,Menlo,monospace;}
.tldr{background:linear-gradient(180deg,#f3f8ff,#eef4fb);border:1px solid #cfe0f3;
border-radius:12px;padding:16px 20px;margin:18px 0 6px;}
.tldr b{color:var(--accent);}
.flow{display:flex;flex-direction:column;gap:0;margin:18px 0;}
.step{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:8px;padding:12px 16px;}
.step .q{font-weight:600;} .step .a{color:var(--mut);font-size:14.5px;margin-top:2px;}
.arrow{align-self:center;color:#9aa6b4;font-size:18px;line-height:1;margin:5px 0;}
.box{border:1px solid var(--line);border-radius:9px;padding:13px 17px;margin:14px 0;}
.box .lab{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.hyp{background:#f5f2ff;border-color:#e0d8fb;} .hyp .lab{color:var(--hyp);}
.find{background:#eef9f1;border-color:#cdeed6;} .find .lab{color:var(--find);}
.over{background:#fff6ec;border-color:#f6dcbd;} .over .lab{color:var(--warn);}
.warnbox{background:#fdeef0;border-color:#f3ccd3;} .warnbox .lab{color:var(--bad);}
figure{margin:20px 0;} figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block;}
figcaption{color:var(--mut);font-size:13.5px;margin-top:7px;text-align:center;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top;}
th{background:var(--panel);font-weight:600;} tbody tr:nth-child(even){background:#fbfcfe;}
.win{color:var(--find);font-weight:700;} .loss{color:var(--bad);font-weight:700;}
.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:20px;
padding:2px 11px;font-size:12.5px;font-weight:600;}
.muted{color:var(--mut);} .small{font-size:13px;}
hr{border:0;border-top:1px solid var(--line);margin:30px 0;}
footer{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px;}
"""

MATHJAX = """
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)'],['$','$']],
displayMath:[['$$','$$'],['\\\\[','\\\\]']]},svg:{fontCache:'global'}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<title>TRINITY transition trigger &mdash; why the hybr run stalls</title>'
    + MATHJAX + "<style>" + CSS + "</style></head><body><div class=\"wrap\">"
)

HERO = r"""
<span class="tag">TRINITY &middot; implicit&rarr;momentum transition trigger</span>
<h1>Why does the <code>hybr</code> run never leave the energy phase?</h1>
<p class="sub">A clean-room investigation &mdash; certify the substrate, then characterise. Identify the
mechanism, test five trigger ideas, run the chosen reading across six regimes, measure, and prototype
the root fix. Verified 2026-06-20; all artifacts committed under
<code>docs/dev/transition/cleanroom/</code>.</p>

<div class="tldr">
<p style="margin:0"><b>TL;DR.</b> Under the default solver <code>betadelta_solver=hybr</code>, every run stalls
in the implicit energy phase and never reaches momentum (<b>0/6</b> configs transition; legacy reaches it
5/6 &mdash; large_diffuse only near ~6 Myr). The hand-off fires on a single criterion, \((L_{\text{gain}}-L_{\text{loss}})/L_{\text{gain}} < 0.05\),
which never trips. We first <b>certified the substrate</b> (the hybr solver + the \(\beta,\delta,P_b,E_b\)
machinery the trigger reads) with an independent gate &mdash; it is <b>sound</b>, so hybr exposed real
behaviour, not a bug. Measuring the retained-energy fraction \(f_{\text{ret}}=E_b/\!\int\!L_{\text{mech}}\,dt\)
against 3D-sim / observed bands shows <b>unanimous under-cooling</b> (\(f_{\text{ret}}\) plateaus at
<b>0.25&ndash;0.40</b>, never the observed <b>0.01&ndash;0.1</b>). Harvesting every candidate trigger shows
<b>no cooling/force family ever fires</b> &mdash; the only physical end-of-energy-phase is <b>geometric
blowout</b> (\(R_2>r_{\text{cloud}}\)). <b>Verdict:</b> the stall is a <i>physics-completeness</i> signal,
not a mis-tuned threshold &mdash; the trigger tests for an event that does not occur. The root fix
(mixing-layer cooling, magnitude \(\theta\!\approx\!0.25\)) is the right direction but a naive bulk energy
sink stalls the solver, so it was <b>reverted</b>; production is unchanged (<code>pytest</code>
557&nbsp;passed).</p>
</div>
"""

FLOW_INTRO = r"""
<h2 id="flow">The chain of reasoning</h2>
<p>Each step answered one question and set up the next. This is the spine of the report; every link is
backed by a figure and a table below, and every number traces to a committed CSV in <code>data/</code>.</p>
<div class="flow">"""

STEPS = [
    ("1 &middot; Did the new <code>hybr</code> solver introduce a bug, or expose real behaviour?",
     r"Real behaviour. An independent substrate gate (C0) passes: the solver drives its temperature "
     r"residual to \(\le\!0.13\%\) and the \(\beta\!\leftrightarrow\!dP_b/dt\) residual is "
     r"finite-difference truncation (shrinks \(\propto\!\Delta t\)). hybr finds the true root."),
    ("2 &middot; If the substrate is sound, why does the trigger never fire?",
     r"The bubble retains too much energy to ever balance: \(f_{\text{ret}}\) plateaus at 0.25&ndash;0.40 "
     r"and never enters the observed 0.01&ndash;0.1 band. The modelled interior <b>under-cools</b>."),
    ("3 &middot; Is it just the metric form (the instantaneous rate-ratio)?",
     r"No. The cumulative-cooling (any \(\eta\)), cooling-timescale and force-balance families were harvested too &mdash; "
     r"the cumulative cooling never reaches \((1-\eta)\!\int\!L_{\text{gain}}\) either. Not a metric-form bug."),
    ("4 &middot; Then is there <i>any</i> cooling transition to trigger on?",
     r"No. \(E_b\) grows monotonically to t=6 in 5/6 configs &mdash; there is no \(E_b\)-peak to track. "
     r"The only family that fires at a physical epoch is <b>blowout</b> (\(R_2>r_{\text{cloud}}\)), "
     r"whose epoch is purely geometric (0.01&ndash;3.66 Myr \(\propto\) cloud size)."),
    ("5 &middot; Can we restore a cooling transition by adding the missing physics?",
     r"The magnitude is right &mdash; a mixing-layer sink \(\theta\!\approx\!0.25\) brings "
     r"\(f_{\text{ret}}\) into the observed band offline &mdash; but a naive bulk \(dE_b/dt\) sink drives "
     r"\(dM/dt<0\) and <b>stalls the solver</b>. Reverted: it must be integrated into the structure solve."),
]

FLOW_OUTRO = (
    "<p><b>Conclusion:</b> substrate sound &rarr; the bubble under-cools &rarr; no cooling family ever "
    "fires &rarr; the transition is <i>geometric, not thermal</i> &rarr; so the rate-ratio trigger isn't "
    "mis-tuned, it tests for an event that does not occur. Pragmatic fix = a blowout trigger; "
    "root fix = the cooling physics (mixing layer), integrated into the structure solve.</p>"
)

SEC_PROBLEM = r"""
<h2 id="problem">1 &middot; Identify the problem &mdash; the mechanism, not the symptom</h2>
<p>TRINITY walks a feedback bubble through four phases: energy &rarr; implicit energy-driven &rarr;
transition &rarr; momentum. The implicit&rarr;momentum hand-off is decided by a <i>single</i> criterion
(<code>run_energy_implicit_phase.py:1095</code>): switch once radiative cooling has nearly caught up with
the <i>instantaneous</i> mechanical power,
\[ \frac{L_{\text{gain}}-L_{\text{loss}}}{L_{\text{gain}}} < 0.05 . \]
Under the new default <code>hybr</code> solver this ratio plateaus at <b>0.5&ndash;0.85</b> and never approaches 0.05,
so all six configs sit in implicit until the 15&nbsp;Myr cap &mdash; <b>0/6</b> reach transition or momentum.</p>
<figure>__FIG_F0__<figcaption>The mechanism. <b>Top:</b> the cooling ratio (the current trigger) for all six
configs &mdash; it lives at \(0.5\!-\!0.85\), well over an order of magnitude above the 0.05 threshold (dashed), and
<i>jumps up</i> at the \(t\!\approx\!3\) Myr SN surge. <b>Bottom:</b> \(L_{\text{mech}}\) showing that surge.
Pure read of <code>data/c0_*_h0.csv</code> via <code>plot_f0path.py</code>.</figcaption></figure>
<p>The ratio's shape has <b>two distinct features with different causes</b> &mdash; we decomposed both into
\(L_{\text{gain}}\) and \(L_{\text{loss}}\) (verified, committed). <b>An early dip</b> (\(t<1\) Myr, before any
SN) in 5/6 configs: cooling \(L_{\text{loss}}\) <i>rises</i> \(\sim\!1.7\!-\!15\times\) while \(L_{\text{gain}}\) stays
flat (the bubble briefly becomes radiative), pulling the ratio down &mdash; the same first cooling episode the
<i>legacy</i> solver transitioned on. Then a <b>recovery</b>: \(L_{\text{loss}}\) <i>collapses</i> (the bubble
expands, \(n^2\Lambda\) falls) while \(L_{\text{gain}}\) is still flat &mdash; pure cooling, with supernovae
absent (\(L_{\text{mech,SN}}\!\approx\!0\) until \(\sim\!3\) Myr). Only the <i>later</i> plateau is SN-sustained.
So the early dip-then-surge is an \(L_{\text{loss}}\) story; the late surge is the feedback one.</p>
<figure>__FIG_DIP__<figcaption>Decomposing the dip-then-surge. <b>(1)</b> the cooling ratio dips early then
surges back up; <b>(2)</b> \(L_{\text{gain}}\) (solid) stays flat while \(L_{\text{loss}}\) (dashed) rises into
the dip and collapses out of it &mdash; the surge is \(L_{\text{loss}}\) collapsing, not \(L_{\text{gain}}\)
rising; <b>(3)</b> wind (solid) is steady and SN (dashed) only switch on at \(\sim\!3\) Myr, so the early dip is
pre-SN. Pure read of <code>data/c0_*_h0.csv</code> via <code>plot_dipdrivers.py</code>.</figcaption></figure>
<p>Two views of one effect. <b>(a) The metric is instantaneous:</b> every time a feedback source switches on
(Wolf&ndash;Rayet winds, then SNe) the \(L_{\text{gain}}\) denominator spikes, so the ratio jumps
<i>away</i> from 0.05 exactly when cooling might catch up &mdash; it tests an instantaneous numerator, not an
integrated budget. <b>(b) hybr finds the true root:</b> legacy clamped the cooling parameter \(\beta\!\in\![0,1]\)
so \(P_b\) could only decline and the ratio drifted to balance; hybr is unbounded with a physical
\(dM/dt>0\) gate and in places returns <b>negative \(\beta\)</b> (\(P_b\) <i>rising</i> &mdash; the bubble
re-pressurising under a feedback surge), keeping the interior net-heating.</p>
<div class="box over"><div class="lab">β alone vs β+δ &mdash; the right compression variable</div>A correction worth stating
plainly (per the archived β&ndash;δ study): the structural quantity that governs the interior velocity is
<b>not β alone, and not β+δ=0</b>. The bubble velocity ODE source term is \((\beta+\delta)/t = -t\,d\ln n/dt\),
so <b>β+δ</b> is the compression term and its interior-inflow trigger sits at <b>β+δ ≲ −0.4</b> (−0.5 in the
archive). β&lt;0 by itself is only <i>re-pressurisation</i> (\(P_b\) rising); δ&gt;0 (T rising)
partly cancels it. In our six, β dives to <b>−2.05</b> (re-pressurisation in all six, be_sphere only marginally),
but β+δ crosses −0.4 in
<b>only one</b> (large_diffuse, 10 rows) &mdash; everywhere else δ offsets β and the net compression stays above
the trigger. The archive's resulting inflow is in any case <b>&ldquo;real but cosmetic&rdquo;</b> (subsonic,
~10⁻⁶ of thermal), not an energy-budget term &mdash; so this corrects the <i>diagnostic framing</i>, not the
trigger verdict.</div>
<figure>__FIG_BETA__<figcaption>β (blue, the \(P_b\)-rate) vs β+δ (orange, the compression source) per config,
with the β+δ=−0.4 inflow trigger (dashed) and the sub-trigger region shaded red. β dives deep at the WR/SN
surge, but β+δ stays <i>above</i> −0.4 in 5/6 &mdash; only large_diffuse reaches it. Pure read of
<code>data/c0_*_h0.csv</code> via <code>plot_beta.py</code>.</figcaption></figure>
<p><b>So what actually triggers a surge &mdash; feedback, or the structure (\(\beta,\delta\))?</b> We measured
both, per config, making no assumptions (step-to-step correlation of the ratio change \(\Delta r\) with
\(\Delta L_{\text{mech}}\), \(\Delta\beta\), \(\Delta\delta\), over the implicit rows of each
<code>data/c0_*_h0.csv</code>). The answer is the same in <b>all six</b>: the surge co-moves with a feedback
power increase (\(\mathrm{corr}(\Delta r,\Delta L_{\text{mech}})>0\), +0.29 to +0.81) <i>and</i> with
\(\beta\) <i>dropping</i> toward re-pressurisation (\(\mathrm{corr}(\Delta r,\Delta\beta)<0\), down to
\(-0.89\)) &mdash; the two are co-driven by the same WR/SN episode (early \(\sim\!0.1\!-\!0.6\) Myr for
midrange / simple_cluster, the SN onset \(\sim\!3\) Myr for the other four). But it is <b>not</b> a fixed
\(\beta\) or \(\delta\) threshold: at the largest jump in each config \(\beta\) spans \(-0.2\) to \(+2.3\)
and \(\delta\) both signs &mdash; there is no &ldquo;\(\beta=-0.05\)&rdquo; trigger value. Negative-\(\beta\)
episodes do carry an <i>elevated</i> ratio (higher in \(\beta<0\) rows than \(\beta\ge0\) in 5/6 configs), so
re-pressurisation is a <i>co-symptom</i> of the surge, not its threshold.</p>
<figure>__FIG_SURGE__<figcaption>Per config, the step-to-step correlation of the cooling-ratio change
\(\Delta r\) with feedback (\(\Delta L_{\text{mech}}\), blue), \(\Delta\beta\) (orange), and
\(\Delta\delta\) (green). Blue \(>0\) and orange \(<0\) in <b>every</b> config: the surge is a feedback event
that simultaneously re-pressurises the bubble (\(\beta\) drops). Pure read of <code>data/c0_*_h0.csv</code>
via <code>plot_surge.py</code> (table in <code>data/surge_coincidence.csv</code>).</figcaption></figure>
<figure>__FIG_PORTRAIT__<figcaption>The same answer in 2-D: every implicit-phase row of all six runs in
\((\delta,\beta)\) space, coloured by time. The re-pressurisation rows (\(\beta<0\), shaded) sweep a wide
\(\delta>0\) band and light up at the \(\sim\!3\) Myr SN-epoch colours &mdash; that \(\delta>0\) (T rising) is
exactly what holds \(\beta+\delta\) above the inflow trigger: the dashed \(\beta+\delta=-0.4\) line (the
compression/inflow threshold, <b>not</b> \(\beta+\delta=0\)) is barely touched, and only by large_diffuse. So
re-pressurisation (\(\beta<0\)) is a late-time <i>feedback</i> event, while net compression
(\(\beta+\delta<-0.4\)) almost never happens. Pure read of <code>data/c0_*_st6.csv</code> via
<code>plot_phaseportrait.py</code>.</figcaption></figure>
<div class="box hyp"><div class="lab">the fork this sets up</div>Two honest readings, both kept open: either the
criterion needs replacing for the hybr regime (a <b>trigger</b> problem), or the bubble retains too much
energy to ever balance because the cooling physics is incomplete (a <b>physics</b> problem). The rest of the
report decides which &mdash; against external data, not against TRINITY's own assumptions.</div>
"""

SEC_IDEAS = r"""
<h2 id="ideas">2 &middot; Test the different ideas &mdash; the method matrix</h2>
<p>Before touching the trigger we drew a <b>trust boundary</b>: do not assume the substrate (the hybr solver
and the \(\beta,\delta,P_b,E_b\) machinery the ratio reads) is correct &mdash; certify it with a cheap,
independent gate first, and anchor the verdict to <i>external</i> physics (Lancaster+2021, El-Badry+2019,
Geen+2021, Pabst+2020), treating TRINITY / WARPFIELD / Weaver as hypotheses, not ground truth.</p>
<p>Five candidate trigger families were on the menu &mdash; <i>which fires when</i> was to be measured, not
inherited (a prior pass had leaked verdicts, so all prior firing-epochs/rankings were quarantined and
re-derived from scratch):</p>
<table><thead><tr><th>trigger</th><th>family</th><th>criterion</th><th>idea being tested</th></tr></thead><tbody>
<tr><td><b>rate-ratio</b> <span class="tag" style="background:#8d99ae">CURRENT</span></td><td>instantaneous rate-ratio</td><td>\((L_{\text{gain}}-L_{\text{loss}})/L_{\text{gain}} < \varepsilon\)</td><td>the production baseline</td></tr>
<tr><td>cumulative cooling</td><td>cumulative energy</td><td>\(\int\!L_{\text{loss}}\,dt / \int\!L_{\text{gain}}\,dt > 1-\eta\)</td><td>remove the instantaneous-reset artifact</td></tr>
<tr><td>cooling timescale</td><td>timescale</td><td>\(t_{\text{cool}}/t_{\text{dyn}} < k\)</td><td>classic Mac&nbsp;Low&ndash;McCray criterion</td></tr>
<tr><td>force balance</td><td>force / continuity</td><td>\(4\pi R^2 P_b /\,(\text{surviving forces}) < O(1)\)</td><td>thermal drive becomes subdominant</td></tr>
<tr><td><b>blowout</b></td><td>blowout (geometric)</td><td>\(R_2 > r_{\text{cloud}}\)</td><td>shell escapes the cloud</td></tr>
<tr><td>mixing-layer</td><td>mixing-flux balance</td><td>Lancaster+2021 fractal-interface cooling</td><td>the root-physics reading (no sharp 1D switch)</td></tr>
</tbody></table>
<p>Each family was harvested per implicit-phase segment across the full span and judged against an
independent oracle that depends on no candidate and no threshold: the <b>\(E_b\)-peak</b>
(PdV-inclusive net-energy zero crossing, \(L_{\text{gain}}-L_{\text{loss}}-4\pi R_2^2 v_2 P_b \le 0\)).</p>
<figure>__FIG_G0__<figcaption>The result, by shape alone: only the <b>blowout</b> trigger (+) fires at a physical
epoch. The rate-ratio, cumulative and force triggers <b>never</b> fire (annotated right); the cooling-timescale trigger (&#9660;) fires absurdly early (\(t\!\approx\!0\), an
artifact); the \(E_b\)-peak oracle (&#9733;) exists in only one config. Pure read of
<code>data/c0_*_h0.csv</code> via <code>plot_g0.py</code>.</figcaption></figure>
<div class="box over"><div class="lab">the path was not straight (kept in the story)</div>
A mid-run read at \(t\!\lesssim\!3\) Myr reported &ldquo;no negative \(\beta\) in any config&rdquo; &mdash;
<b>retracted</b> once the runs passed the SN epoch, where negative-\(\beta\) re-pressurisation appears in all
six (textbook &ldquo;don&rsquo;t trust one time-slice&rdquo;). And the appealing cumulative-cooling fix &mdash; &ldquo;the
problem is just the instantaneous numerator, integrate it away&rdquo; &mdash; was <b>falsified</b>: the
cumulative ratio never fires either.</div>
"""

SEC_CONFIGS = r"""
<h2 id="configs">3 &middot; Work the chosen reading across the configs</h2>
<p>The decisive measurement is the external-anchored one: the retained-energy fraction
\(f_{\text{ret}}(t)=E_b/\!\int\!L_{\text{mech}}\,dt\), run across all six regimes &mdash; spanning
<b>3 dex in cloud mass</b>, every density profile (flat, steep \(r^{-2}\), Bonnor&ndash;Ebert), and the full
sfe range. If \(f_{\text{ret}}\) curves into the observed 0.01&ndash;0.1 band, the trigger question is
well-posed; if it plateaus far above, the stall is an under-cooling <i>physics</i> gap.</p>
<figure>__FIG_FRET__<figcaption>The single most important figure. All six \(f_{\text{ret}}(t)\) curves plateau
at <b>0.25&ndash;0.40</b> &mdash; <b>below</b> the Weaver energy-conserving value 5/11≈0.45 (dashed) and <b>never
entering</b> the observed / 3D-sim band 0.01&ndash;0.1 (shaded). Unanimous across regimes. Pure read of
<code>data/c0_*_st6.csv</code> via <code>plot_fret.py</code>.</figcaption></figure>
<p>The verdict holds across <i>every</i> regime, not just the easy one &mdash; including the steep
\(r^{-2}\) crux, where \(L_{\text{loss}}\!\propto\!n^2\) collapses as the bubble expands into thin gas, so its
late-time cooling is weak &mdash; yet it still under-cools like the rest (it is mid-pack in retention, not the
extreme). The under-cooling is structural, not a one-config fluke.</p>
<table><thead><tr><th>config</th><th>mCloud / profile</th><th>\(f_{\text{ret}}\) end</th>
<th>\(f_{\text{ret}}\) min</th><th>in 0.01&ndash;0.1 band?</th></tr></thead><tbody>
<tr><td>large_diffuse_lowsfe</td><td>1e7 &middot; flat</td><td>0.248</td><td>0.248</td><td class="loss">no</td></tr>
<tr><td>be_sphere</td><td>1e6 &middot; Bonnor&ndash;Ebert</td><td>0.283</td><td>0.165</td><td class="loss">no</td></tr>
<tr><td>midrange_pl0</td><td>1e6 &middot; flat</td><td>0.330</td><td>0.169</td><td class="loss">no</td></tr>
<tr><td>pl2_steep <span class="tag" style="background:#e8842a">crux</span></td><td>1e6 &middot; \(r^{-2}\)</td><td>0.339</td><td>0.197</td><td class="loss">no</td></tr>
<tr><td>small_dense_highsfe</td><td>1e4 &middot; flat</td><td>0.383</td><td>0.160</td><td class="loss">no</td></tr>
<tr><td>simple_cluster</td><td>1e5 &middot; flat</td><td>0.397</td><td>0.150</td><td class="loss">no</td></tr>
</tbody></table>
<p class="small muted">Numbers read from the final / minimum <code>f_ret</code> rows of each
<code>data/c0_*_st6.csv</code>. Observed band: Lancaster+2021 (I&amp;II), Geen+2021, Pabst+2020 (refs in
<code>FINDINGS.md</code> &sect;5).</p>
"""

SEC_MEASURE = r"""
<h2 id="measure">4 &middot; The measurements &mdash; where each family would fire</h2>
<p>One row per config, harvested from <code>data/c0_*_h0.csv</code> (the G0 deliverable). The cooling and
force families are flatly <code>never</code>; the cooling-timescale trigger is an artifact at \(t\!\approx\!0\); the \(E_b\)-peak oracle
exists in only one config; and the blowout epoch tracks <b>cloud size</b>, not any cooling event &mdash;
spanning 0.01&nbsp;Myr (a 0.33-pc cloud) to 3.66&nbsp;Myr (an 88-pc cloud).</p>
<table><thead><tr><th>config</th><th>\(r_{\text{cloud}}\) [pc]</th>
<th>rate-ratio / cumulative / force</th><th>timescale (artifact)</th><th>\(E_b\)-peak</th>
<th><b>blowout</b> [Myr]</th></tr></thead><tbody>
<tr><td>small_dense_highsfe</td><td>0.33</td><td class="loss">never</td><td>0.00</td><td>&mdash;</td><td class="win">0.01</td></tr>
<tr><td>simple_cluster</td><td>1.69</td><td class="loss">never</td><td>0.00</td><td>&mdash;</td><td class="win">0.09</td></tr>
<tr><td>midrange_pl0</td><td>8.53</td><td class="loss">never</td><td>0.01</td><td>&mdash;</td><td class="win">0.39</td></tr>
<tr><td>be_sphere</td><td>15.50</td><td class="loss">never</td><td>0.01</td><td>&mdash;</td><td class="win">0.86</td></tr>
<tr><td>pl2_steep</td><td>21.40</td><td class="loss">never</td><td>0.00</td><td>&mdash;</td><td class="win">0.84</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>88.00</td><td class="loss">never</td><td>0.39</td><td>5.06</td><td class="win">3.66</td></tr>
</tbody></table>
<p class="small muted">Firing epochs from <code>harvest_h0.py</code> over <code>data/c0_*_h0.csv</code>.
A structural note from the same harvest: \(P_b \equiv P_{\text{HII}}\) to machine precision
(bubble&ndash;shell pressure continuity by construction, with \(P_{\text{ram}}=0,\,F_{\text{ISM}}=0\)), so any
pressure-balance force criterion is degenerate &mdash; another reason no force criterion bites.</p>
<figure>__FIG_BLOW__<figcaption>The one surviving transition, plotted: the blowout epoch against cloud
radius, one point per config. It hugs a slope-1 (\(t_{\text{blowout}}\!\propto\!r_{\text{cloud}}\)) guide over
\(2.5\) dex in size &mdash; the end of the energy phase is set by <i>geometry</i>, not by any cooling event
(the cooling and force triggers never fire). Pure read of <code>data/c0_*_h0.csv</code> via <code>plot_blowout.py</code>.</figcaption></figure>
<div class="box find"><div class="lab">measurement verdict (6/6, unanimous)</div>For these under-cooled bubbles
the energy&rarr;momentum transition <b>is not a cooling / energy / force event at all</b> &mdash; no scalar
threshold in the rate-ratio / cumulative / timescale family can express it. The only physical end-of-energy-phase is
<b>geometric blowout</b>, which is profile/size-dependent. The rate-ratio is not mis-<i>tuned</i>; it tests for
an event that does not occur in the hybr regime.</div>
"""

SEC_SOLUTION = r"""
<h2 id="solution">5 &middot; Solution &mdash; the diagnosis, and what (didn&rsquo;t) ship</h2>
<p>The deliverable here is a <b>diagnosis</b>, and a deliberate <i>non-change</i> to production. Retuning the
0.05 threshold is futile &mdash; there is no cooling-balance event in the hybr regime to tune toward.</p>
<div class="box find"><div class="lab">recommendation</div>
<b>Pragmatic interim</b> (if completable runs are needed now): a profile-aware <b>blowout</b> transition
(\(R_2>r_{\text{cloud}}\)) &mdash; the only candidate that fires physically (caveat: its epoch is geometric,
near-instant for compact clouds, so consider \(R_2>k\,r_{\text{cloud}}\) or a sustained criterion).
<b>Root fix:</b> integrate Lancaster/El-Badry mixing-layer cooling so a cooling transition exists at all.</div>
<p>We prototyped the root fix far enough to size it. Offline (<code>mixcool_whatif.py</code>) a mixing-layer
sink \(L_{\text{mix}}=\theta\,L_{\text{mech}}\) at the literature \(\theta\!\approx\!0.25\) brings
\(f_{\text{ret}}\) into the observed band in all six configs &mdash; the direction is right.</p>
<figure>__FIG_MIX__<figcaption>Root-fix sizing (static what-if on the committed CSVs). <b>Top:</b> the modified
retained energy \(f_{\text{ret}}-\theta\) enters the observed 0.01&ndash;0.1 band near the literature
\(\theta\!\approx\!0.25\) (dotted) for all six configs. <b>Bottom:</b> the modified minimum cooling ratio
crosses the 0.05 threshold &mdash; i.e. a cooling transition <i>would</i> now fire. The magnitude is right;
the red note records why the naive <i>dynamical</i> version is not. Pure read of <code>data/c0_*_h0.csv</code>
via <code>plot_mixcool.py</code>.</figcaption></figure>
<div class="box warnbox"><div class="lab">retracted &mdash; the naive implementation does not work</div>
Injecting that sink dynamically (subtracting \(\theta L_{\text{mech}}\) from \(dE_b/dt\) <i>after</i> the
\((\beta,\delta)\) structure solve) is <b>numerically non-viable</b>: it drives the conductive \(dM/dt<0\) (no
physical evaporation root), so hybr finds no root, the timestep-shrink guard spins, and the solver stalls. A
proper mixing-layer cooling must be solved <i>inside</i> the structure solve (so \(\beta,\delta\) are found
<i>with</i> it, keeping \(dM/dt>0\)) &mdash; a deeper change than a bulk energy sink. The prototype was
reverted.</div>
<div class="box find"><div class="lab">what shipped to production</div><b>Nothing.</b> The bulk-sink
injection and its <code>mixL_theta</code> param were reverted; <code>trinity/</code> is byte-identical to its
pre-prototype state and <code>pytest</code> is green (<b>557 passed</b>). The substrate stays frozen; the
investigation closed the trigger question against external physics without a production change.</div>
"""

SEC_ARC = r"""
<h2 id="arc">6 &middot; The rest of the arc &mdash; the validation journey</h2>
<p>The verdict only stands because the substrate was certified <i>first</i>, before any physics claim. The
certification gate (C0) is independent of the contested trigger:</p>
<figure>__FIG_CERT__<figcaption><b>Top:</b> \(\mathrm{res}_\beta\) (the genuine
\(\beta\!\leftrightarrow\!dP_b/dt\) trajectory residual) decays over each run &mdash; finite-difference
truncation, not a defect. <b>Bottom:</b> \(\mathrm{res}_{T_0}\) (the solver&rsquo;s own temperature residual)
stays tight. Pure read of <code>data/c0_*_st6.csv</code> via <code>plot_cert.py</code>.</figcaption></figure>
<table><thead><tr><th>config</th><th>\(\mathrm{res}_\beta\) median</th><th>\(\mathrm{res}_{T_0}\) median</th>
<th>neg-\(\beta\) rows</th><th>substrate</th></tr></thead><tbody>
<tr><td>large_diffuse_lowsfe</td><td>4.66%</td><td>0.13%</td><td>9.3%</td><td class="win">sound</td></tr>
<tr><td>simple_cluster</td><td>5.70%</td><td>0.00%</td><td>11.5%</td><td class="win">sound</td></tr>
<tr><td>small_dense_highsfe</td><td>3.84%</td><td>0.00%</td><td>12.0%</td><td class="win">sound</td></tr>
<tr><td>midrange_pl0</td><td>6.08%</td><td>0.00%</td><td>4.3%</td><td class="win">sound</td></tr>
<tr><td>pl2_steep</td><td>6.08%</td><td>0.00%</td><td>4.6%</td><td class="win">sound</td></tr>
<tr><td>be_sphere</td><td>5.42%</td><td>0.00%</td><td>1.1%</td><td class="win">sound</td></tr>
</tbody></table>
<p>The certification followed the project&rsquo;s equivalence discipline: a per-segment residual is necessary
but not sufficient, so <code>res_beta</code> was cleared with a <b>4&times; timestep-refinement</b> check
(median 6.65%&rarr;1.74%, scaling \(\propto\!\Delta t\) &mdash; the signature of truncation, not a bug),
across the <b>stiffest edge regimes</b> (the steep \(r^{-2}\) crux included), in <b>separate full runs</b> at
<b>matched simulation time</b>. The analytic adiabatic-Weaver null was infeasible (the solver cannot run with
\(L_{\text{loss}}\!\equiv\!0\)), so substrate trust rests on the internal residuals plus the energy-fraction
corroboration (the code reproduces Weaver 5/11, then stalls there &mdash; precisely the energy-conserving
limit the literature says under-cools).</p>
<div class="box find"><div class="lab">one-line takeaway</div>When a &ldquo;stall&rdquo; appears after a
solver upgrade, certify the substrate before tuning the threshold: here the upgrade was correct and the stall
was the model honestly reporting that <b>its bubble never cools enough to transition</b> &mdash; a physics gap,
not a trigger bug. Don&rsquo;t fix missing physics by tuning a scalar.</div>
"""

SEC_FOLLOWUP = r"""
<h2 id="followup">7 &middot; Follow-up &mdash; legacy-vs-hybr, the dip mechanism, and the WARPFIELD / θ critique</h2>

<h3>7.1&nbsp; Why the stall is new: BEFORE (legacy) vs AFTER (hybr)</h3>
<p>The early dip is the <i>first cooling episode</i>. Under the legacy clamped-\(\beta\) solver the ratio is driven
all the way down to 0.05 there and the run transitions to momentum; under hybr the <i>same</i> episode dips but
recovers (\(L_{\text{loss}}\) collapses) and never crosses. That is exactly why the stall appeared only after the
solver upgrade.</p>
<figure>__FIG_BEFOREAFTER__<figcaption>Cooling trigger \((L_{\text{mech}}-L_{\text{loss}})/L_{\text{mech}}\) over the
evolution. <b>Left (legacy):</b> the ratio dives through 0.05 at the first cooling episode &mdash; dots mark the
crossing where each run transitions (small_dense 0.024, pl2_steep 0.128, simple_cluster 0.178, midrange 0.822,
be_sphere 1.037 Myr &mdash; 5/6). <b>Right (hybr):</b> the same early dip, but it recovers and <b>never crosses</b>.
Real legacy runs (<code>--solver legacy</code>) vs the committed hybr runs. <span class="muted">large_diffuse is the
lone very-late crosser (~6 Myr, beyond the stop_t=2.5 window).</span></figcaption></figure>

<h3>7.2&nbsp; What the dip actually is: an emission-measure turnover, not a thermal trigger</h3>
<p>We tested the obvious hypothesis &mdash; that \(L_{\text{loss}}\) rises because gas enters the efficient-cooling
band (the \(\Lambda(T)\) peak, \(\sim\!10^5\!-\!10^6\) K) &mdash; and <b>the data refuted it.</b> \(T_0\) stays at
<b>at or above \(10^6\) K through the dip (typically 3&ndash;8\(\times10^6\); small_dense only grazes the \(10^6\) K
band edge)</b>, so \(\Lambda\) is effectively flat.
The dip is set purely by the <b>emission measure</b> \(n^2 V \propto (P_b/T_0)^2 R_2^3\): it <i>rises</i> because
volume growth outruns dilution, then <i>collapses</i> because \(R_2\) expansion dilutes \(n^2\) faster than \(V\)
grows (the log-slopes cross at the \(L_{\text{loss}}\) peak in all six configs; the EM proxy peaks within ~1.7&times;
in time of the real \(L_{\text{loss}}\) peak &mdash; 4/6 within 1.3&times;).</p>
<figure>__FIG_DIPMECH__<figcaption>Through the early dip (3 representative configs): \(L_{\text{loss}}\) (orange) and
the emission-measure proxy \(n^2V=(P_b/T_0)^2R_2^3\) (blue) rise-then-collapse together, while \(T_0\) (green, right
axis) sits far <i>above</i> the shaded \(10^5\!-\!10^6\) K cooling-peak band the entire time. Pure read of
<code>data/c0_*_h0.csv</code> via <code>plot_dipmechanism.py</code>.</figcaption></figure>
<div class="box find"><div class="lab">the mechanistic root of the under-cooling</div>The bubble retains too much
energy because <b>its interior stays too hot</b> (\(>\!10^6\) K, up in the weak-\(\Lambda\) bremsstrahlung tail) &mdash;
it never makes the \(\sim\!10^5\!-\!10^6\) K gas that radiates efficiently. So the fix is not &ldquo;add a loss
term&rdquo; but &ldquo;<b>create the cool, efficiently-radiating gas</b>&rdquo; &mdash; precisely what a turbulent
mixing layer does. The dip diagnosis and the mixing-layer root-fix are the same story from two directions.</div>
<p><b>What turns over first?</b> A natural worry: does something in the <i>velocity</i> (or \(\dot E_b\)) change
course first and <i>cause</i> \(L_{\text{loss}}\) to drop? We checked the turning-point order per config. The shell
velocity <i>does</i> lead: \(v_2\) decelerates to a minimum then re-accelerates, and that minimum leads (or
coincides with) the \(L_{\text{loss}}\)-peak / ratio-minimum in <b>4/5 early-dip configs</b> (small_dense 0.012&lt;0.015,
simple_cluster 0.084&lt;0.098, midrange 0.392&lt;0.432). But it is <b>not causal, and \(\dot E_b\) never flips</b>:
\(\dot E_b = L_{\text{gain}}-L_{\text{loss}}-4\pi R_2^2 v_2 P_b\) stays strongly positive through the whole dip (the
bubble keeps gaining energy). Underneath both \(v_2\) and \(L_{\text{loss}}\), <b>\(P_b\) and \(T_0\) fall
monotonically with no turning point</b> &mdash; neither causes the other; both track the interior
depressurisation/dilution. β,δ peak <i>after</i> \(L_{\text{loss}}\) &mdash; a lagging readout, not the trigger.</p>
<figure>__FIG_CAUSAL__<figcaption>Causal ordering through the dip (representative configs): normalised \(v_2\),
\(L_{\text{loss}}\), and ratio with their turning points marked, \(P_b\)/\(T_0\) on a log right axis. \(v_2\)-min
leads the \(L_{\text{loss}}\)-peak, but \(P_b\),\(T_0\) decline monotonically &mdash; the common driver. Pure read of
<code>data/c0_*_h0.csv</code> via <code>plot_dip_causalorder.py</code>.</figcaption></figure>
<div class="box over"><div class="lab">units &amp; formula audit &mdash; clean</div>Because a wrong unit or formula could
fake all of this, we audited the source against the registry units (<code>trinity/_input/registry.py</code>). No bug:
the velocity-ODE source \((\beta+\delta)/t\) is dimensionally \([1/\text{Myr}]\) (correctly scales a velocity,
<code>bubble_luminosity.py:411</code>); the work term \(4\pi R_2^2 v_2 P_b\) carries \(L_{\text{mech}}\) units
(\(M_\odot\,\text{pc}^2/\text{Myr}^3\)), so \(L_{\text{gain}}-L_{\text{loss}}-\dot W\) is unit-homogeneous;
<code>cool_beta_to_Ebdot_pure</code>, <code>bubble_E2P</code> (its cgs detour cancels to machine precision), and
<code>get_leak_luminosity</code> all check term-by-term. The dip is real physics, not a unit slip.</div>

<h3>7.3&nbsp; The WARPFIELD criterion is our current trigger in disguise</h3>
<p>A WARPFIELD-style switch \(\log_{10}L_{\text{mech}}-\log_{10}L_{\text{cool}}<0.05\) (i.e.
\(L_{\text{cool}}>0.89\,L_{\text{mech}}\), PdV excluded) is the <b>same family</b> as the current rate-ratio trigger.
On our hybr data it <b>does not fire</b>: the gap bottoms at <b>0.145&ndash;0.292 dex</b> (\(L_{\text{cool}}\) reaches
only 51&ndash;72% of \(L_{\text{mech}}\), never 89%). The looser 0.89 threshold gets <i>closer at the dip</i> but the
under-cooling wall still blocks it &mdash; the log-space reformulation doesn't escape the G0 verdict.</p>
<table><thead><tr><th>config</th><th>min WARPFIELD gap [dex]</th><th>max \(L_{\text{cool}}/L_{\text{mech}}\)</th><th>fires (&lt;0.05 dex)?</th></tr></thead><tbody>
<tr><td>small_dense_highsfe</td><td>0.145</td><td>0.717</td><td class="loss">no</td></tr>
<tr><td>simple_cluster</td><td>0.170</td><td>0.676</td><td class="loss">no</td></tr>
<tr><td>midrange_pl0</td><td>0.197</td><td>0.636</td><td class="loss">no</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>0.272</td><td>0.535</td><td class="loss">no</td></tr>
<tr><td>be_sphere</td><td>0.277</td><td>0.529</td><td class="loss">no</td></tr>
<tr><td>pl2_steep</td><td>0.292</td><td>0.511</td><td class="loss">no</td></tr>
</tbody></table>
<p>The one real difference is WARPFIELD's <b>leakage term</b> \(L_{\text{cool}}=L_b+L_{\text{leak}}\); our default
runs are sealed (\(\text{coverFraction}=1,\ L_{\text{leak}}=0\)). Leakage is an extra loss channel &mdash; whether a
<i>plausible</i> coverFraction closes the 0.15&ndash;0.3 dex gap and stays solver-healthy is <b>under test</b>
(Cf = 0.99 / 0.95 / 0.90). Note leakage <i>vents hot gas</i>; it does not <i>create</i> cool radiating gas, so it is a
different lever than mixing-layer cooling.</p>

<h3>7.4&nbsp; Open problems with the \(\theta\,L_{\text{mech}}\) mixing-layer term</h3>
<div class="box over"><div class="lab">(1) double-counting against L_cool</div>TRINITY's
\(L_{\text{cool}}=L_{\text{bubble}}+L_{\text{conduction}}+L_{\text{intermediate}}\) already contains a smooth 1-D
<i>interface</i> model. Lancaster's \(L_{\text{mix}}\) is the <i>same</i> interface done turbulently, so adding
\(\theta L_{\text{mech}}\) on top of the full \(L_{\text{cool}}\) <b>double-counts</b> the conductive/intermediate
interface. The correct move is to <b>replace</b> \(L_{\text{conduction}}+L_{\text{intermediate}}\) with the turbulent
term (or add only the excess) &mdash; not stack a flat fraction on the whole.</div>
<div class="box over"><div class="lab">(2) θ=const carries no state dependence</div>\(L_{\text{cool}}=\int n^2
\Lambda(T)\,dV\) responds to the actual bubble structure; \(\theta L_{\text{mech}}\) knows only the feedback.
\(\theta\!\approx\!\)const is an <i>emergent</i> 3-D-sim result in the efficient-cooling limit &mdash; it should depend
on density, metallicity, turbulent velocity, density contrast. A constant \(\theta\) shifts every config equally and
so <b>cannot reproduce the measured config-to-config spread</b> (\(f_{\text{ret}}\) 0.25&ndash;0.40). It is
magnitude-right (it <i>did</i> bring the bubble into the band) but <b>not predictive</b>; a faithful term ties
\(L_{\text{mix}}\) to interface area (\(\propto R_2^2\)), a mixing velocity, and the contact-discontinuity density,
reducing to \(\theta L_{\text{mech}}\) only in the efficient limit.</div>

<h3>7.5&nbsp; What actually changed legacy&rarr;hybr: the β-clamp, not the temperature</h3>
<p>Running the dip diagnostic on the <i>legacy</i> runs and overlaying them on hybr isolates the difference. At
<b>matched times the interior temperature \(T_0\) is comparable</b> in both (same order, \(\sim\!3\!-\!6\times10^6\) K
&mdash; within ~20% for small_dense and simple_cluster, ~2\(\times\) for pl2_steep) &mdash; the
temperature is <b>not</b> what changed. The difference is entirely in <b>β</b>: legacy clamps
\(\beta\!\in\![0,1]\), which forces a structure that keeps \(L_{\text{loss}}\) high so the ratio is driven
monotonically down to its crossing; hybr is unbounded and β swings to <b>+2…+4</b> right at the dip, the structure
\(L_{\text{loss}}\) <i>collapses</i>, and the ratio recovers.</p>
<figure>__FIG_LEGHYBR__<figcaption>Legacy (blue, solid) vs hybr (orange, dashed) through the dip, three early
crossers. <b>Left:</b> cooling ratio &mdash; legacy crosses 0.05 (marked) and transitions; hybr recovers.
<b>Middle:</b> \(L_{\text{loss}}\) &mdash; legacy keeps cooling, hybr's collapses out of the dip. <b>Right:</b>
β &mdash; legacy pinned in the shaded [0,1] clamp box, hybr escapes upward; the \(T_0\)-at-crossing annotation
shows the two temperatures are the same order (within ~2&times;). Pure read of <code>data/c0_*_legacy.csv</code>
+ <code>data/c0_*_h0.csv</code> via <code>plot_legacy_vs_hybr.py</code>.</figcaption></figure>
<p>The companion panels (δ, β+δ, \(E_b\), \(P_b\)) pin the discriminator more sharply: δ plunges negative in
<i>both</i> solvers (so δ alone isn't it), but <b>β+δ</b> is decisive &mdash; under legacy it dips <b>below 0</b>
(to ~−0.5 in pl2_steep, crossing the −0.4 line) while under hybr it stays <b>strictly positive</b> (min ~0.07&ndash;0.69).
Lifting the β-clamp lets β+δ re-sum positive, \(E_b\) stays supported (hybr retains ~2&times; more out of the dip),
and \(P_b\) takes a sharper but <i>recoverable</i> excursion. The cure direction made visible: keep β+δ positive
(don't clamp β) and \(E_b\) is sustained.</p>
<figure>__FIG_LEGHYBREXTRA__<figcaption>Companion to the above: legacy (blue, solid) vs hybr (orange, dashed) through
the dip for δ, β+δ (with the −0.4 line), \(E_b\), and \(P_b\), same three configs. The β+δ column is the
discriminator &mdash; legacy dips below 0, hybr stays positive. Pure read of <code>data/c0_*_legacy.csv</code> +
<code>data/c0_*_h0.csv</code> via <code>plot_legacy_vs_hybr_extra.py</code>.</figcaption></figure>
<div class="box find"><div class="lab">so the BEFORE/AFTER is a solver-root difference, not a cooling-physics one</div>
The C0 certification established hybr finds the <i>true</i> root; legacy, when the true root leaves \([0,1]\), is
pinned to the constraint <i>edge</i>. So legacy's &ldquo;transition&rdquo; was a <b>consequence of the β-clamp</b>
(a constrained edge-root that keeps \(L_{\text{loss}}\) high), not genuine extra cooling &mdash; the interior is
comparably hot in both (same order). This closes the loop with §7.1&ndash;7.2: the stall is what the <i>correct</i> root does, and
the old trigger fired only because the caged solver couldn't reach it.</div>

<h3>7.6&nbsp; Why not put the PdV work term into the trigger?</h3>
<p>The current trigger compares only \(L_{\text{gain}}\) vs \(L_{\text{loss}}\) and <b>excludes</b> the expansion-work
term \(\dot W = P_b\,dV/dt = 4\pi R_2^2 v\,P_b\) (the only difference between the trigger and the Eb-peak oracle). Why
keep it out? <b>Because PdV is the energy-DRIVING mechanism, not a loss.</b> The bubble energy budget is
\(\dot E_b = L_{\text{gain}} - L_{\text{loss}} - \dot W\): \(L_{\text{loss}}\) is irreversible <i>radiation leaving
the system</i>; \(\dot W\) is <i>reversible work the bubble does on the shell</i> &mdash; it is literally how an
energy-driven bubble drives. The energy&rarr;momentum transition is the onset of <i>catastrophic radiative cooling</i>
(\(L_{\text{loss}}\!\to\!L_{\text{gain}}\)), when the interior can no longer hold its over-pressure. Counting \(\dot W\)
as a &ldquo;loss&rdquo; conflates <i>doing work</i> with <i>dying</i>.</p>
<p>Mechanically, \((L_{\text{gain}}-L_{\text{loss}}-\dot W)/L_{\text{gain}}\) is just the normalised
\(\dot E_b/L_{\text{gain}}\) &mdash; so a PdV-inclusive trigger fires at the <b>\(E_b\)-peak</b> (\(\dot E_b\!=\!0\)),
which asks &ldquo;has \(E_b\) stopped growing?&rdquo; not &ldquo;is cooling winning?&rdquo; Those are different
events. And \(\dot W\) is <i>largest when the bubble is healthiest</i> (high \(R_2^2 v\), still hot, vigorously
driving), so the PdV-inclusive ratio is <i>lowest</i> exactly when the bubble is most energy-driven &mdash; firing
there is backwards.</p>
<figure>__FIG_PDV__<figcaption><b>Left:</b> where the input goes (simple_cluster) &mdash; \(L_{\text{gain}}\) splits
into radiated \(L_{\text{loss}}\) (~20%), expansion <b>work</b> \(\dot W\) (~45%), and a net \(\dot E_b\)
(~35%). Nearly half the input is work &mdash; more than is radiated. <b>Right:</b> F0 (solid, no PdV) vs F0+PdV (dashed
\(=\dot E_b/L_{\text{gain}}\)) for all six: subtracting \(\dot W\) pulls the ratio ~0.5&rarr;0.05&ndash;0.15
(<i>nearly</i> firing) but it still never crosses 0.05 in 5/6 &mdash; it is tracking the \(E_b\)-peak, which the
bubble never reaches. Pure read of <code>data/c0_*_h0.csv</code> via <code>plot_pdv.py</code>.</figcaption></figure>
<div class="box find"><div class="lab">verdict on PdV-in-the-trigger</div>It does <b>not</b> rescue the trigger and
is <b>physically the wrong term</b>. \(\dot W\!\approx\!0.43\!-\!0.46\,L_{\text{gain}}\) (measured), so adding it
pulls the ratio tantalisingly close to 0.05 &mdash; but it crosses only in large_diffuse, and only at <b>4.76 Myr</b>
(its \(E_b\)-peak), while the other five never peak (\(E_b\) grows monotonically to t=6). So PdV-inclusion swaps one
non-event (&ldquo;cooling never wins&rdquo;) for another (&ldquo;\(E_b\) never peaks&rdquo;) <i>and</i> mis-frames the
physics: it would declare the bubble momentum-driven at its energetic peak, mid-drive. So keep PdV out of the
<i>cooling-balance ratio</i> &mdash; but that is the cooling-driven view; the <b>regime-spanning</b> criterion (and
the large-cloud case) is the PdV-inclusive \(E_b\)-peak itself &mdash; see §7.7.</div>

<h3>7.7&nbsp; The other end of the spectrum: huge PdV, negative \(E_b\), and how it's handled</h3>
<p>The same \(\dot W\) term that is harmless work here becomes <i>catastrophic</i> for very massive clusters (the
prior <code>failed-large-clouds</code> investigation). A <code>5e9</code> \(M_\odot\) cloud at \(\text{sfe}\!\sim\!0.05\)
is a \(\sim\!5\times10^8\,M_\odot\) cluster (\(L_{\text{mech}}\!\sim\!500\times\) typical); it launches the shell at
<b>~2000&ndash;3700 km/s</b> (near free-expansion), so \(\dot W=4\pi R_2^2 v\,P_b\) <b>exceeds \(L_{\text{mech}}\)</b>
(\(\dot W/L_{\text{mech}}\!\to\!1.56\), with \(L_{\text{cool}}\!\sim\!1\%\)). \(E_b\) then <b>peaks and collapses
through zero into negative</b> &mdash; whereupon \(R_1\!\to\!R_2\) (shell volume\(\to\!0\)), the \(P_b\) divide blows
up, and the run crashes with \(E_b=\text{nan}\). It is the <i>opposite</i> failure to the stall, from the <i>same</i>
budget term.</p>
<figure>__FIG_PDVMASS__<figcaption><b>Left:</b> the \(5e9\) cluster's \(E_b\) rises, peaks at \(t\!\approx\!1.5\times
10^{-3}\) Myr (where \(\dot W/L_{\text{mech}}\!\to\!1\)), then plunges <b>negative</b> &rarr; the \(R_1\!\to\!R_2\)
NaN crash. <b>Right:</b> max \(\dot W/L_{\text{mech}}\) per config &mdash; <b>one control parameter sorts the
regimes:</b> our six typical clouds (and the healthy \(1e6\) control) sit below 1 (\(E_b\) grows, energy-driven &rarr;
the stall, because cooling never catches up either); only the \(5e9\) cluster exceeds 1 (\(E_b\) collapses, momentum
from birth &rarr; crash). My six are computed from <code>data/c0_*_h0.csv</code>; the two large-cloud values are the
<code>failed-large-clouds</code> PLAN's reliable post-IC-relaxation numbers.</figcaption></figure>
<div class="box find"><div class="lab">how it's dealt with &mdash; and the reconciliation with §7.6</div>
Two layers, both already scoped. <b>(1) Robustness (shipped in <code>failed-large-clouds</code>):</b> a geometry guard
(floor the shell volume so the \(P_b\) divide can't blow up, bit-identical when \(R_1\!\ll\!R_2\)) + a loud-fail that
detects non-finite or \(E_b\!\le\!0\) and <b>terminates cleanly</b> with an <code>ENERGY_COLLAPSED</code> reason instead
of crashing/NaN. <b>(2) Physics (&ldquo;family T&rdquo;, deferred to <i>this</i> workstream):</b> hand off to the
momentum phase at the <b>PdV-inclusive net-energy zero-crossing</b> \((L_{\text{gain}}-L_{\text{loss}}-\dot W)\le0\)
&mdash; the \(E_b\)-peak &mdash; <i>before</i> \(E_b\) goes negative. <b>This is the reconciliation:</b> PdV does not
belong in the cooling-balance <i>ratio</i> (§7.6), but the correct, <b>regime-spanning</b> transition is exactly the
PdV-inclusive \(E_b\)-peak: it fires immediately for the \(5e9\) cluster (PdV-driven, the principled fix for the crash)
and <i>never</i> for typical clouds (\(E_b\) keeps growing &mdash; which is precisely the stall, and exposes that the
missing physics there is cooling). One criterion, both ends.</div>
"""

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/cleanroom/</code> &mdash; reproducible
without re-running the (hours-long) hybr sims. Each figure is a pure read of a committed CSV.</p>
<table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>c0_consistency.py</code></td><td>substrate certification harness (C0: residuals + \(f_{\text{ret}}\))</td></tr>
<tr><td><code>harvest_h0.py</code></td><td>candidate-trigger firing-epoch harvest (the G0 deliverable)</td></tr>
<tr><td><code>mixcool_whatif.py</code></td><td>offline mixing-layer (\(\theta\)) calibration for the root fix</td></tr>
<tr><td><code>data/c0_*_st6.csv</code> &middot; <code>data/c0_*_h0.csv</code> &middot; <code>data/c0_*_legacy.csv</code> &middot; <code>data/surge_coincidence.csv</code></td><td>per-config hybr captures, legacy (BEFORE) captures, + the surge-coincidence table</td></tr>
<tr><td><code>plot_{fret,f0path,beta,surge,phaseportrait,dipdrivers,g0,blowout,mixcool,cert,dipmechanism,beforeafter,legacy_vs_hybr,legacy_vs_hybr_extra,pdv,pdv_massspectrum,dip_causalorder}.py</code> &rarr; <code>figures/*.png</code></td><td>the seventeen figures above (each a pure read of a CSV)</td></tr>
<tr><td><code>PLAN.md</code> &middot; <code>FINDINGS.md</code></td><td>the living plan / pre-registration &amp; the consolidated write-up</td></tr>
</tbody></table>
<p class="small muted">Rebuild this report:
<code>python docs/dev/transition/cleanroom/make_transition_report.py</code>
(figures regenerate from <code>data/</code> via the <code>plot_*.py</code> generators).</p>
<footer>TRINITY implicit&rarr;momentum transition investigation &middot; clean-room, certify-then-build &middot;
every number traces to a committed CSV in <code>docs/dev/transition/cleanroom/data/</code> &middot;
light-mode, MathJax-rendered.</footer>
"""


def main():
    parts = [HEAD, HERO, FLOW_INTRO]
    for i, (q, a) in enumerate(STEPS):
        parts.append(f'<div class="step"><div class="q">{q}</div><div class="a">{a}</div></div>')
        if i < len(STEPS) - 1:
            parts.append('<div class="arrow">&#9660;</div>')
    parts.append("</div>")
    parts.append(FLOW_OUTRO)
    parts += [SEC_PROBLEM, SEC_IDEAS, SEC_CONFIGS, SEC_MEASURE, SEC_SOLUTION, SEC_ARC,
              SEC_FOLLOWUP, SEC_REPRO]
    parts.append("</div></body></html>")
    html = "".join(parts)

    for token, (fname, alt) in FIGURES.items():
        html = html.replace(token, img(fname, alt))
    assert "__FIG_" not in html, "unreplaced figure placeholder remains"

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
