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
in the implicit energy phase and never reaches momentum (<b>0/6</b> configs transition; legacy reached it
6/6). The hand-off fires on a single criterion, \((L_{\text{gain}}-L_{\text{loss}})/L_{\text{gain}} < 0.05\),
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
Under the new default <code>hybr</code> solver this ratio plateaus near <b>0.5</b> and never approaches 0.05,
so all six configs sit in implicit until the 15&nbsp;Myr cap &mdash; <b>0/6</b> reach transition or momentum.</p>
<figure>__FIG_F0__<figcaption>The mechanism. <b>Top:</b> the cooling ratio (the current trigger) for all six
configs &mdash; it lives at \(\sim\!0.5\), an order of magnitude above the 0.05 threshold (dashed), and
<i>jumps up</i> at the \(t\!\approx\!3\) Myr SN surge. <b>Bottom:</b> \(L_{\text{mech}}\) showing that surge.
Pure read of <code>data/c0_*_h0.csv</code> via <code>plot_f0path.py</code>.</figcaption></figure>
<p>The ratio's shape has <b>two distinct features with different causes</b> &mdash; we decomposed both into
\(L_{\text{gain}}\) and \(L_{\text{loss}}\) (verified, committed). <b>An early dip</b> (\(t<1\) Myr, before any
SN) in 5/6 configs: cooling \(L_{\text{loss}}\) <i>rises</i> 2&ndash;20&times; while \(L_{\text{gain}}\) stays
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
<figure>__FIG_BETA__<figcaption>Why balance is never reached: \(\beta(t)\) per config (blue), with
\(\beta<0\) re-pressurisation shaded, over \(L_{\text{mech}}\) (grey). Negative-\(\beta\) segments appear in
<b>all six</b> configs at the SN epoch (1&ndash;12% of implicit rows). Pure read of <code>data/c0_*_st6.csv</code>
via <code>plot_beta.py</code>.</figcaption></figure>
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
\((\delta,\beta)\) space, coloured by time. The re-pressurisation band (\(\beta<0\), shaded) is not a tight
cluster at one \(\beta\), \(\delta\), or \(\beta+\delta\) value (the dotted \(\beta+\delta=0\) line is only a
reference) &mdash; it sweeps a wide \(\delta>0\) band and lights up at the \(\sim\!3\) Myr SN-epoch colours.
Re-pressurisation is a late-time <i>feedback</i> event, not a structure threshold. Pure read of
<code>data/c0_*_st6.csv</code> via <code>plot_phaseportrait.py</code>.</figcaption></figure>
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
at <b>0.25&ndash;0.40</b> &mdash; pinned near the Weaver energy-conserving value 5/11 (dashed) and <b>never
entering</b> the observed / 3D-sim band 0.01&ndash;0.1 (shaded). Unanimous across regimes. Pure read of
<code>data/c0_*_st6.csv</code> via <code>plot_fret.py</code>.</figcaption></figure>
<p>The verdict holds across <i>every</i> regime, not just the easy one &mdash; including the steep
\(r^{-2}\) crux, where \(L_{\text{loss}}\!\propto\!n^2\) collapses as the bubble expands into thin gas, giving
it the <i>highest</i> retention (least cooling) of the span. The under-cooling is structural, not a
one-config fluke.</p>
<table><thead><tr><th>config</th><th>mCloud / profile</th><th>\(f_{\text{ret}}\) end</th>
<th>\(f_{\text{ret}}\) min</th><th>in 0.01&ndash;0.1 band?</th></tr></thead><tbody>
<tr><td>large_diffuse_lowsfe</td><td>1e7 &middot; flat</td><td>0.248</td><td>0.248</td><td class="loss">no</td></tr>
<tr><td>be_sphere</td><td>1e6 &middot; Bonnor&ndash;Ebert</td><td>0.283</td><td>0.165</td><td class="loss">no</td></tr>
<tr><td>midrange_pl0</td><td>1e6 &middot; flat</td><td>0.330</td><td>0.169</td><td class="loss">no</td></tr>
<tr><td>pl2_steep <span class="tag" style="background:#e8842a">crux</span></td><td>1e6 &middot; \(r^{-2}\)</td><td>0.339</td><td>0.197</td><td class="loss">no</td></tr>
<tr><td>small_dense_highsfe</td><td>1e4 &middot; flat</td><td>0.383</td><td>0.150</td><td class="loss">no</td></tr>
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

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/cleanroom/</code> &mdash; reproducible
without re-running the (hours-long) hybr sims. Each figure is a pure read of a committed CSV.</p>
<table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>c0_consistency.py</code></td><td>substrate certification harness (C0: residuals + \(f_{\text{ret}}\))</td></tr>
<tr><td><code>harvest_h0.py</code></td><td>candidate-trigger firing-epoch harvest (the G0 deliverable)</td></tr>
<tr><td><code>mixcool_whatif.py</code></td><td>offline mixing-layer (\(\theta\)) calibration for the root fix</td></tr>
<tr><td><code>data/c0_*_st6.csv</code> &middot; <code>data/c0_*_h0.csv</code> &middot; <code>data/surge_coincidence.csv</code></td><td>per-config full-run captures + the surge-coincidence table (the evidence)</td></tr>
<tr><td><code>plot_{fret,f0path,beta,surge,phaseportrait,dipdrivers,g0,blowout,mixcool,cert}.py</code> &rarr; <code>figures/*.png</code></td><td>the ten figures above (each a pure read of a CSV)</td></tr>
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
    parts += [SEC_PROBLEM, SEC_IDEAS, SEC_CONFIGS, SEC_MEASURE, SEC_SOLUTION, SEC_ARC, SEC_REPRO]
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
