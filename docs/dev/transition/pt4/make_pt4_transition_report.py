#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single self-contained, light-mode, MathJax-rendered HTML report that
tells the *part-4* transition-trigger investigation as a flowing continuation of
the clean-room chapter ("the transition-trigger problem — geometric, not
thermal"). Three acts: H1 (is it an Lcool bug?), H2 (is breaking rCloud a fail?),
and R1 (the opt-in fix that shipped). Verdicts and every number are read straight
from the committed pt4 docs (H1_lcool_audit.md, H2_rcloud_audit.md,
r1shadow/R1_FINDINGS.md) and the figure source make_pt4_figures.py.

The 7 PNGs in figures/ are embedded as base64 so the file is standalone
(downloadable, opens offline; MathJax loads from CDN for the formulas). Each
figure is a pure read of a committed CSV (made by make_pt4_figures.py); every
table number traces to the committed pt4 audit docs.

Content blocks are plain strings with __FIG_*__ placeholders (LaTeX braces stay
literal); the placeholders are swapped for base64 <img> tags before writing. The
structure (standalone HTML, <body><div class="wrap">, base64 <img>, house-style
<style>, MathJax include) mirrors the sibling cleanroom report so
build_storylines.py ingests it as a chapter.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/transition/pt4/make_pt4_figures.py            # -> figures/*.png
    python docs/dev/transition/pt4/make_pt4_transition_report.py  # -> pt4_transition_report.html
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = HERE / "pt4_transition_report.html"

FIGURES = {
    "__FIG_H1_SURGE__": ("h1_lloss_surge_collapse.png",
                         "Lloss(t) for all six hybr configs, log-log, peak marked"),
    "__FIG_H1_CLAMP__": ("h1_beta_clamp_divergence.png",
                         "cooling ratio and cool_beta vs time, hybr vs legacy on simple_cluster"),
    "__FIG_H1_STATS__": ("h1_ratio_min_stats.png",
                         "bar chart of the cooling-ratio minimum per config, hybr vs legacy vs 0.05"),
    "__FIG_H2_RCLOUD__": ("h2_ratio_vs_rcloud.png",
                          "cooling ratio vs R2/rCloud, all configs, cloud-edge crossing marked"),
    "__FIG_H2_MATCHED__": ("h2_matched_r2.png",
                           "cooling ratio vs absolute R2, baseline vs bigcloud overlay"),
    "__FIG_H2_GRADIENT__": ("h2_dip_vs_density_gradient.png",
                            "where the cooling-ratio minimum sits vs the in-cloud density gradient"),
    "__FIG_R1__": ("r1_firing_preview.png",
                   "per config, where R1 would hand off vs the never-firing current trigger"),
    "__FIG_CLAMP_SOLVER__": ("clamp_vs_solver.png",
                             "per config: cooling ratio(t) and cool_beta(t), legacy vs hybr; legacy crosses 0.05 and transitions, hybr floors"),
    "__FIG_SOLVER_STATS__": ("solver_stats.png",
                             "legacy vs hybr bar stats: implicit segments, beta-delta convergence, cooling-ratio minimum, peak beta"),
    "__FIG_LEGHYBR_GRID__": ("legacy_vs_hybr_grid.png",
                             "legacy vs hybr across six configs for Eb, Lloss, Lmech, PdV, rShell (log-log)"),
    "__FIG_RUN_COST__": ("run_cost.png",
                         "current-version run cost: wall-clock runtime, implicit segments, per-segment cost per config"),
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
    '<title>TRINITY transition trigger pt4 &mdash; re-examining the verdict, and a fix</title>'
    + MATHJAX + "<style>" + CSS + "</style></head><body><div class=\"wrap\">"
)

HERO = r"""
<span class="tag">TRINITY &middot; transition trigger &middot; part 4</span>
<h1>Re-examining the verdict &mdash; is it a bug, a fake boundary, and what could actually fire?</h1>
<p class="sub">The previous chapter closed with a hard claim: under the default <code>hybr</code> solver the
implicit&rarr;momentum cooling-balance trigger \((L_{\text{gain}}-L_{\text{loss}})/L_{\text{gain}}<0.05\)
never fires, so the transition looked <i>geometric, not thermal</i>. Part 4 stress-tests that verdict against
three honest objections &mdash; and ships a fix. Verified 2026-06-22 on branch
<code>fix/transition-trigger-problem-pt4</code>; all artifacts committed under
<code>docs/dev/transition/pt4/</code>.</p>

<div class="tldr">
<p style="margin:0"><b>TL;DR.</b> Three challenges to last chapter's "geometric, not thermal" verdict, each
resolved against committed data, none overturning it. <b>H1 &mdash; is it an <code>Lcool</code> bug, does
cooling &ldquo;surge up&rdquo;?</b> No bug: the cooling integral is <b>byte-identical</b> across the recent
refactor chain. \(L_{\text{cool}}\) <i>does</i> surge \(\sim\!2\times\) early, <b>then collapses 4&ndash;9&times;</b>
(reconciling both intuitions); the legacy/hybr divergence is entirely the <b>β-clamp</b> (legacy clamps
\(\beta\!\in\![0,1]\) and crosses; hybr's free \(\beta\!\to\!+4\) under-cools). The ratio floors
<b>0.28&ndash;0.49</b>, never 0.05. <b>H2 &mdash; is &ldquo;breaking rCloud&rdquo; a failure, can we make rCloud
infinite?</b> No: crossing \(r_{\text{cloud}}\) is a <b>clean phase switch</b> (<code>is_simulation_ending=False</code>),
cooling is set by <b>local density, not rCloud proximity</b> (a matched-\(R_2\) experiment: identical ratio
0.4845 at \(R_2\!=\!0.894\) pc despite a 5.2&times; larger cloud), and the dip tracks the in-cloud density gradient
(steep <code>pl2_steep</code> bottoms at \(r_{\text{core}}\), not the edge). <b>R1 &mdash; the fix.</b> A
transition built on events that <i>do</i> occur &mdash; <b>blowout</b> (\(R_2>r_{\text{cloud}}\)) + <b>Eb-peak</b>
(\(\dot E_b\!\le\!0\)) &mdash; built shadow-first (byte-identical <code>dictionary.jsonl</code>), then exposed as
an <b>opt-in</b> <code>transition_trigger</code> keyword (default unchanged). Gates: byte-identical, unit
<b>14/14</b>, regression <b>588 passed</b>, drive end-to-end on <code>simple_cluster</code>.</p>
</div>
"""

FLOW_INTRO = r"""
<h2 id="flow">The chain of reasoning</h2>
<p>The prior chapter's verdict invited three good-faith objections. This chapter answers each in turn &mdash;
two confirm the diagnosis, the third ships a transition. Every link is backed by a figure and a number, and
every number traces to a committed pt4 audit doc (<code>H1_lcool_audit.md</code>,
<code>H2_rcloud_audit.md</code>, <code>r1shadow/R1_FINDINGS.md</code>).</p>
<div class="flow">"""

STEPS = [
    ("H1 &middot; Is the never-firing trigger a bug in <code>Lcool</code> &mdash; does cooling keep &ldquo;surging up&rdquo;?",
     r"No bug. The cooling integral is <b>byte-identical</b> across the recent refactor chain. \(L_{\text{cool}}\) "
     r"surges \(\sim\!2\times\) early <i>then collapses 4&ndash;9&times;</i>; the legacy/hybr split is the "
     r"<b>β-clamp</b>, not the integral. Ratio floors 0.28&ndash;0.49, never 0.05."),
    ("H2 &middot; Is &ldquo;breaking rCloud&rdquo; a failure &mdash; could an infinite rCloud let the trigger fire?",
     r"No. \(R_2>r_{\text{cloud}}\) is a <b>clean phase switch</b>, not a stop. Cooling is set by <b>local "
     r"density</b>, not proximity to the edge (matched-\(R_2\): identical ratio despite 5.2&times; rCloud), so a "
     r"bigger cloud cannot manufacture a cooling-balance event."),
    ("R1 &middot; If no cooling event fires, what event <i>can</i> end the energy phase &mdash; and ship safely?",
     r"A transition on events that <b>do</b> occur: <b>blowout</b> (\(R_2>r_{\text{cloud}}\)) + <b>Eb-peak</b> "
     r"(\(\dot E_b\!\le\!0\)). Built <b>shadow-first</b> (byte-identical), then exposed as an <b>opt-in</b> "
     r"<code>transition_trigger</code> keyword (default unchanged). Gates all green."),
]

FLOW_OUTRO = (
    "<p><b>Conclusion:</b> the &ldquo;geometric, not thermal&rdquo; verdict survives both stress tests "
    "&mdash; <code>Lcool</code> is correct (H1) and rCloud is a clean boundary whose removal cannot create a "
    "cooling event (H2) &mdash; and the chapter ends not with a diagnosis alone but with a <b>shipped, opt-in "
    "exit</b> (R1) built on the geometric/energetic events the model actually produces.</p>"
)

SEC_RECAP = r"""
<h2 id="recap">0 &middot; Where we left off &mdash; geometric, not thermal</h2>
<p>The clean-room chapter (part 3) certified the <code>hybr</code> substrate as sound, measured the
retained-energy fraction \(f_{\text{ret}}=E_b/\!\int\!L_{\text{mech}}\,dt\) plateauing at <b>0.25&ndash;0.40</b>
(never the observed 0.01&ndash;0.1 band), harvested every candidate trigger, and concluded that the
cooling-balance criterion <b>tests for an event that does not occur</b> &mdash; the only physical
end-of-energy-phase is <i>geometric blowout</i> (\(R_2>r_{\text{cloud}}\)). It read as a physics-completeness
signal, not a mis-tuned threshold.</p>
<p>That verdict is uncomfortable, and it should be challenged before it is trusted. Three objections deserve a
direct, data-backed answer:</p>
<div class="box hyp"><div class="lab">the three objections this chapter answers</div>
<b>H1.</b> Maybe it is simply a <i>bug</i> &mdash; a regression in <code>bubble_LTotal</code> (the
<code>Lcool</code> integral) that makes cooling &ldquo;keep surging up&rdquo;, so the ratio can never fall.
<b>H2.</b> Maybe &ldquo;breaking <code>rCloud</code>&rdquo; is an <i>artificial failure</i>: remove that
boundary, let the run continue self-consistently in an arbitrarily large cloud, and the cooling trigger would
fire after all. <b>R1.</b> If neither rescues the cooling trigger, then what event <i>can</i> end the energy
phase &mdash; and can it be shipped without disturbing production?</div>
<p>Each is read-only against the committed clean-room CSVs and production source; no production code was
changed for H1/H2, and the R1 change is opt-in and byte-identical by default. The six regime-spanning configs
are the same set as part 3 (\(\sim\!3\) dex in cloud mass, every density profile, the full sfe range), so a
verdict that holds across all six is regime-spanning, not a fluke.</p>
"""

SEC_H1 = r"""
<h2 id="h1">1 &middot; H1 &mdash; is it a <code>Lcool</code> bug? Does cooling &ldquo;surge up&rdquo;?</h2>
<p>The trigger reads \(L_{\text{loss}}=\) <code>bubble_LTotal</code>, the radiative cooling integral
\(L_{\text{loss}}=\int n^2\Lambda(T)\,dV\) (three zones &mdash; CIE bubble, conduction, intermediate &mdash;
each trapezoid-integrated; <code>bubble_luminosity.py</code>). If a recent refactor had made this surge upward
spuriously, the ratio could never fall, and the &ldquo;geometric&rdquo; verdict would be an artifact. We
audited it three ways: the git history, the direction of \(L_{\text{cool}}\) over time, and the legacy/hybr
divergence.</p>

<h3>1.1&nbsp; The cooling integral is byte-identical across the refactor chain</h3>
<p>Diffing the integrand, the \(L_*\) zone sums, the <code>np.abs(trapz)</code> calls, and the unit factors
(<code>Lambda_cgs2au</code>, <code>dudt_cgs2au</code>, <code>chi_e</code>) across the recent cooling-path
commits (<code>7f08e58</code>, <code>24c6914</code>, <code>4996060</code>, <code>60fb362</code>) shows
<b>all of them byte-identical</b>; <code>git blame</code> attributing lines to the regroup commit is the
<i>move</i>, not a content change. The only two genuine content changes in history are both benign and far too
small to matter: a deliberate \(+20\%\) <code>chi_e</code> factor on the CIE integrands (a physics fix
&mdash; CIE cooling is \(n_e n_H\Lambda\) &mdash; which would <i>help</i> the trigger fire, \(\sim\!15\times\)
too small to bridge \(0.3\!\to\!0.05\)) and a conduction dense-output sampling change worth \(\le\!0.18\%\) on
\(L_{\text{total}}\). No sign flip, no dropped factor, no au&harr;cgs slip. <b>The <code>Lcool</code>
computation is correct &mdash; no bug, no regression.</b></p>

<h3>1.2&nbsp; <code>Lcool</code> surges \(\sim\!2\times\) early, then collapses 4&ndash;9&times;</h3>
<p>The maintainer's intuition (&ldquo;<code>Lcool</code> keeps surging up&rdquo;) and the prior FINDINGS'
(&ldquo;<code>Lcool</code> collapses&rdquo;) are <b>both right, at different epochs.</b> In the first
\(\sim\!0.1\) Myr the emission measure rises (volume growth beats dilution while \(R_2\) is tiny), so
\(L_{\text{loss}}\) roughly doubles &mdash; the surge. Then \(R_2\) grows, \(n^2V\propto(P_b/T_0)^2R_2^3\)
dilutes, and \(L_{\text{loss}}\) <b>falls by 4&ndash;9&times;</b> while the interior stays at
\(T_0\!\approx\!3\!-\!8\times10^6\) K &mdash; too hot to enter the \(10^5\!-\!10^6\) K \(\Lambda\)-peak band, so
cooling stays weak. (simple_cluster: \(L_{\text{loss}}\) \(1.90\!\times\!10^8\to\) peak \(4.10\!\times\!10^8\)
@ \(t\!=\!0.098\to5.68\!\times\!10^7\) @ \(t\!=\!0.84\), a \(-7.2\times\) collapse.)</p>
<figure>__FIG_H1_SURGE__<figcaption>\(L_{\text{cool}}=L_{\text{loss}}\) for all six hybr configs (log-log); the
dot marks each curve's peak. Every config surges \(\sim\!2\times\) to an early peak, then collapses
4&ndash;9&times; as the bubble dilutes &mdash; an emission-measure turnover, not a runaway surge and not a bug.
Pure read of <code>../cleanroom/data/c0_*_h0.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>

<h3>1.3&nbsp; The real cause is the β-clamp, not the cooling integral</h3>
<p>Why did the legacy solver cross 0.05 while hybr never does, on the <i>same</i> cooling code? Both solvers
track identically to \(t\!\approx\!0.08\) Myr; then they split. Legacy clamps \(\beta\!\in\![0,1]\) (and
\(\delta\!\in\![-1,0]\)), so \(\beta\) is forced to \(\sim\!0.5\), the structure keeps \(L_{\text{loss}}\)
climbing, and the ratio is driven through 0.05 &rarr; transition. hybr is unbounded and \(\beta\) jumps to
\(+3.5\) (@ \(t\!=\!0.22\)), \(+4.2\) (@ \(t\!=\!0.46\)); the hotter/steeper profile makes \(L_{\text{loss}}\)
<b>collapse</b> and the ratio recovers to 0.92. Identical \(P_b\) and \(T_0\) pre-split &mdash; the <i>only</i>
difference is the \(\beta\) value the solver is <i>allowed</i> to reach. Legacy's &ldquo;crossing&rdquo; is a
<b>constrained edge-root artifact of the clamp</b>, consistent with part 3's certification that hybr finds the
true (unbounded) root.</p>
<figure>__FIG_H1_CLAMP__<figcaption>hybr (orange) vs legacy (blue) on simple_cluster. <b>Top:</b> the cooling
ratio &mdash; legacy is driven through the 0.05 threshold (dashed); hybr dips then recovers, never crossing.
<b>Bottom:</b> \(\beta=-t\,\partial_t\ln P_b\) &mdash; legacy is pinned in the shaded clamp band [0,1] while
hybr's free \(\beta\) escapes to \(+4\). The under-cooling is the unbounded root doing the right thing. Pure
read of <code>c0_simple_cluster_{h0,legacy}.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>
<figure>__FIG_H1_STATS__<figcaption>The cooling-ratio minimum per config: hybr (orange) floors at
<b>0.28&ndash;0.49</b> &mdash; <b>0/6</b> reach the 0.05 trigger (dashed) &mdash; while legacy (blue) dips to
\(\le\!0\) in <b>5/6</b>. The whole 0/6-vs-5/6 split is the β-clamp; the cooling integral is the same code in
both. Pure read of <code>H1_lcool_direction_summary.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>
<table><thead><tr><th>config</th><th>solver</th><th>ratio min</th><th>@ \(t\) [Myr]</th>
<th>ratio final</th><th>crosses 0.05?</th></tr></thead><tbody>
<tr><td>simple_cluster</td><td><b>hybr</b></td><td>0.324</td><td>0.098</td><td>0.764</td><td class="loss">no</td></tr>
<tr><td>large_diffuse_lowsfe</td><td><b>hybr</b></td><td>0.465</td><td>4.86</td><td>0.561</td><td class="loss">no</td></tr>
<tr><td>small_dense_highsfe</td><td><b>hybr</b></td><td>0.283</td><td>0.015</td><td>0.695</td><td class="loss">no</td></tr>
<tr><td>midrange_pl0</td><td><b>hybr</b></td><td>0.364</td><td>0.432</td><td>0.833</td><td class="loss">no</td></tr>
<tr><td>pl2_steep</td><td><b>hybr</b></td><td>0.489</td><td>0.037</td><td>0.831</td><td class="loss">no</td></tr>
<tr><td>be_sphere</td><td><b>hybr</b></td><td>0.471</td><td>0.556</td><td>0.829</td><td class="loss">no</td></tr>
<tr><td>simple_cluster</td><td>legacy</td><td>&minus;0.007</td><td>0.178</td><td>&mdash;</td><td class="win">yes @0.178</td></tr>
<tr><td>small_dense_highsfe</td><td>legacy</td><td>0.024</td><td>0.024</td><td>&mdash;</td><td class="win">yes @0.024</td></tr>
<tr><td>midrange_pl0</td><td>legacy</td><td>&minus;0.009</td><td>0.82</td><td>&mdash;</td><td class="win">yes @0.822</td></tr>
<tr><td>pl2_steep</td><td>legacy</td><td>&minus;0.001</td><td>0.128</td><td>&mdash;</td><td class="win">yes @0.128</td></tr>
<tr><td>be_sphere</td><td>legacy</td><td>&minus;0.020</td><td>1.04</td><td>&mdash;</td><td class="win">yes @1.04</td></tr>
</tbody></table>
<p class="small muted">From <code>H1_lcool_direction_summary.csv</code> (<code>analyze_lcool_direction.py</code>
+ <code>trajectory_probe.py</code> over <code>../cleanroom/data/c0_*_{h0,legacy}.csv</code>). large_diffuse
legacy never crosses (min 0.514, truncated).</p>
<div class="box find"><div class="lab">H1 verdict &mdash; not a bug</div>The <code>Lcool</code> integral is
byte-identical through every recent refactor and physics-identical to the original. \(L_{\text{cool}}\) surges
\(\sim\!2\times\) <i>then</i> collapses 4&ndash;9&times; (both intuitions reconciled). The 0/6-vs-5/6 split is
purely the <b>β-clamp</b>: legacy's caged root keeps \(L_{\text{loss}}\) high and crosses; hybr's free root
under-cools. The non-firing is a <b>physics result</b>, exactly as part 3 found &mdash; not a computational
artifact. Don't tune the 0.05 threshold and don't hunt a <code>Lcool</code> bug; there is neither.</div>
"""

SEC_H2 = r"""
<h2 id="h2">2 &middot; H2 &mdash; is &ldquo;breaking rCloud&rdquo; a fail? Can we make rCloud infinite?</h2>
<p>The second objection: the run &ldquo;breaks&rdquo; when the shell crosses \(r_{\text{cloud}}\) &mdash; maybe
that is an artificial wall, and if the cloud were arbitrarily large the run would continue self-consistently
and the cooling trigger would eventually fire. We checked what crossing rCloud actually does in code, whether
rCloud can be enlarged cleanly, and whether enlarging it lowers the cooling floor.</p>

<h3>2.1&nbsp; Crossing rCloud is a clean phase switch, not a failure</h3>
<p>rCloud is <b>derived, not free</b>: it is a <code>run_const</code>, <code>derived_init</code> param
computed from \((m_{\text{cloud}},n_{\text{core}},r_{\text{core}},\alpha)\) (or the Bonnor&ndash;Ebert
\(\Omega\)) at init &mdash; there is no <code>rCloud</code> input knob. Crucially, <b>\(R_2>r_{\text{cloud}}\)
is a clean phase-1a&rarr;1b hand-off</b>, not a stop: the <code>cloud_boundary</code> event carries
<code>is_simulation_ending = False</code>, and the only rCloud-keyed termination (<code>RCLOUD_BOUNDARY</code>,
code 3) lives in the <b>clean 0&ndash;9 exit band</b>, the same band as <code>STOPPING_TIME</code> &mdash; it
is explicitly <i>not</i> an error. The implicit phase carries <b>no rCloud event at all</b> and deliberately
integrates <i>past</i> it (out to 100&ndash;500&times; rCloud), to <code>stop_r</code> / <code>stop_t</code> /
collapse. Crossing rCloud produces <b>no NaN, no error, no crash</b>. The prior framing &ldquo;breaking rCloud
= fail&rdquo; is wrong as stated.</p>
<figure>__FIG_H2_RCLOUD__<figcaption>Cooling ratio vs \(R_2/r_{\text{cloud}}\) for all six configs. The ratio
bottoms out <b>at the cloud edge</b> (\(R_2/r_{\text{cloud}}\!\approx\!0.7\!-\!1.1\) for 5/6) at \(\sim\!0.3\!-\!0.5\),
then <b>recovers</b> as the shell runs into the dilute ISM &mdash; it moves <i>away</i> from 0.05 past the edge.
Crossing \(R_2=r_{\text{cloud}}\) (solid line) is a clean phase switch. Pure read of
<code>../cleanroom/data/c0_*_h0.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>

<h3>2.2&nbsp; Cooling is set by local density, not rCloud proximity (matched-\(R_2\))</h3>
<p>rCloud can't be set directly, but it can be enlarged <i>cleanly</i>: raise \(m_{\text{cloud}}\) (rCloud
grows as \(m_{\text{cloud}}^{1/3}\)) and lower <code>sfe</code> by the same factor so the cluster
(feedback) mass \(m_{\text{cluster}}=m_{\text{cloud}}\cdot\text{sfe}\) is unchanged &mdash; same \(n_{\text{core}}\),
same \(n(r)\) inside, same \(L_{\text{mech}}\), just a larger dense cloud. The committed experiment
(<code>sc_baseline.param</code> rCloud=1.69 pc vs <code>sc_bigcloud.param</code> rCloud=8.83 pc, a 5.2&times;
enlargement at fixed feedback) ran <b>both cleanly</b> (exit 0, no NaN), both under-cooling
(\(f_{\text{ret}}\) 0.46&rarr;0.22&ndash;0.26).</p>
<p>The decisive comparison: at the <b>same absolute shell radius \(R_2=0.894\) pc</b> &mdash; where the
baseline shell is at \(R_2/r_{\text{cloud}}=0.53\) (halfway to its edge) but the bigcloud shell is at
\(R_2/r_{\text{cloud}}=0.10\) (deep inside its larger cloud) &mdash; the cooling state is <b>identical</b>:
\(L_{\text{loss}}/L_{\text{gain}}=0.5155\), ratio \(0.4845\) in <i>both</i>. <b>The cooling floor is set by the
local density the shell sits in, not by proximity to rCloud.</b> So enlarging rCloud only changes <i>where</i>
the eventual density cliff sits, not the in-cloud cooling floor &mdash; a bigger cloud cannot manufacture a
cooling-balance event.</p>
<figure>__FIG_H2_MATCHED__<figcaption>The decisive test: cooling ratio vs <i>absolute</i> shell radius \(R_2\),
baseline (thick blue underlay) vs bigcloud (orange, 5.2&times; rCloud). The two curves <b>overlap exactly</b>
in-cloud; at the matched \(R_2=0.894\) pc (dotted) the ratio is \(0.4845\) in both, despite very different
\(R_2/r_{\text{cloud}}\). Local density sets the cooling, not the cloud size. Pure read of
<code>h2_sc_{baseline,bigcloud}.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>

<h3>2.3&nbsp; Where the dip sits tracks the in-cloud density gradient</h3>
<p>One loose end from part 3: why does <code>pl2_steep</code> bottom out deep inside the cloud
(\(R_2/r_{\text{cloud}}\!\approx\!0.06\)) while the flat clouds bottom at the edge (\(\approx\!1.0\))? Because
the cooling-ratio minimum sits at the <b>first steep density gradient the shell meets.</b> For flat
(\(\alpha\!=\!0\)) clouds the density holds constant to \(r_{\text{cloud}}\), so the only knee is the edge
&mdash; the dip lands there. For the \(\alpha\!=\!-2\) <code>pl2_steep</code> profile the density already falls
steeply <i>within</i> the cloud (from \(r_{\text{core}}\) outward), so the in-cloud dilution collapses
\(L_{\text{loss}}\) long before the edge. The dip location is a clean readout of the local profile, not of
rCloud.</p>
<figure>__FIG_H2_GRADIENT__<figcaption><b>Top:</b> the production ambient density profile \(n(r)\) per config;
the dot marks where the cooling ratio bottoms. It sits at the density &ldquo;knee&rdquo; &mdash; the cloud
<i>edge</i> for the four flat configs, but \(r_{\text{core}}\) (well inside) for steep <code>pl2_steep</code>.
<b>Bottom:</b> the correlation &mdash; the steeper the in-cloud decline \(n_{\text{core}}/n_{\text{edge}}\), the
deeper inside the dip sits. Pure read of <code>h2_crossing_summary.csv</code> + <code>h2_rcloud_edge.csv</code>
(production density profile) via <code>make_pt4_figures.py</code>.</figcaption></figure>
<table><thead><tr><th>config</th><th>rCloud [pc]</th><th>\(t_{\text{cross}}\) [Myr]</th>
<th>\(R_2^{\max}/r_{\text{cloud}}\)</th><th>ratio min</th>
<th>\(R_2/r_{\text{cloud}}\) @ ratio min</th></tr></thead><tbody>
<tr><td>small_dense_highsfe</td><td>0.326</td><td>0.0117</td><td>525&times;</td><td>0.283</td><td>1.11</td></tr>
<tr><td>simple_cluster</td><td>1.69</td><td>0.0902</td><td>147&times;</td><td>0.324</td><td>1.07</td></tr>
<tr><td>midrange_pl0</td><td>8.53</td><td>0.392</td><td>34.4&times;</td><td>0.364</td><td>1.06</td></tr>
<tr><td>pl2_steep <span class="tag" style="background:#e8842a">outlier</span></td><td>21.35</td><td>0.840</td><td>13.9&times;</td><td>0.489</td><td>0.064</td></tr>
<tr><td>be_sphere</td><td>15.5</td><td>0.856</td><td>15.2&times;</td><td>0.471</td><td>0.72</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>88.05</td><td>3.66</td><td>1.52&times;</td><td>0.465</td><td>1.22</td></tr>
</tbody></table>
<p class="small muted">From <code>h2_crossing_summary.csv</code> (<code>h2_analyze.py</code> over
<code>../cleanroom/data/c0_*_h0.csv</code>; rCloud injected from <code>h2_rcloud_edge.csv</code>). The in-cloud
ratio minimum is <b>0.28&ndash;0.49</b> everywhere &mdash; \(\sim\!6\!-\!10\times\) above 0.05.</p>
<div class="box find"><div class="lab">H2 verdict &mdash; clean boundary, refuted as a fix</div>Crossing
rCloud is a clean phase switch (<code>is_simulation_ending=False</code>), not a failure. Cooling is set by
<b>local density</b>, not rCloud proximity (matched-\(R_2\): identical ratio at \(R_2=0.894\) pc despite a
5.2&times; cloud), and the dip location tracks the in-cloud density gradient. So a bigger cloud holds the shell
in dense gas longer but pins \(L_{\text{loss}}/L_{\text{gain}}\) at its intrinsic in-cloud floor of
<b>0.28&ndash;0.49</b> &mdash; it never reaches 0.05. <b>H2 is vindicated as a diagnosis</b> (blowout and the
cooling stall are one geometric phenomenon) <b>but refuted as a remedy</b> (the under-cooling is intrinsic to
the energy-conserving interior; the lever is more cooling, not more cloud).</div>
"""

SEC_R1 = r"""
<h2 id="r1">3 &middot; R1 &mdash; the fix that shipped (opt-in)</h2>
<p>H1 and H2 both confirm: no cooling-balance event exists in the hybr regime to trigger on. So the fix is not
to tune the cooling trigger but to <b>add a transition built on events that <i>do</i> occur</b> &mdash; and to
ship it without disturbing the frozen substrate. R1 is that transition: two criteria in
<code>run_energy_implicit_phase.py</code>,</p>
<ul>
<li><b>blowout</b> &mdash; \(R_2 > r_{\text{cloud}}\), the geometric end-of-energy-phase part 3 identified;</li>
<li><b>Eb-peak</b> &mdash; \(\dot E_b = L_{\text{gain}}-L_{\text{loss}}-4\pi R_2^2 v_2 P_b \le 0\) (the
PdV-inclusive net-energy zero crossing), <i>already computed in production</i> as
<code>Edot_from_balance</code> at <code>get_betadelta.py:434</code> &mdash; the shadow only <i>reads</i> it.</li>
</ul>
<p>both reached through a pure, tested helper pair (<code>evaluate_r1_shadow()</code> for the criteria;
<code>parse_transition_triggers()</code> / <code>r1_transition_decision()</code> for the keyword&rarr;criterion
mapping).</p>

<h3>3.1&nbsp; Built shadow-first &mdash; inert, byte-identical</h3>
<p>R1 ships in two modes. <b>Shadow (always on, inert):</b> every implicit (1b) segment the criteria are
evaluated, the first firing logged, and a sideline <code>shadow_R1_1b.csv</code> written &mdash; it never sets
<code>termination_reason</code>, never breaks, never writes a physics param, so the main
<code>dictionary.jsonl</code> is <b>byte-identical</b>. <b>Drive (opt-in):</b> a new
<code>transition_trigger</code> param (a comma-separated set; default <code>cooling_balance</code> = the
unchanged production trigger). A non-default set (<code>blowout</code>, <code>ebpeak</code>, <code>r1</code>,
or e.g. <code>cooling_balance,blowout</code>) makes R1 end the energy phase, and <code>main.py</code> proceeds
to 1c&rarr;momentum on the same path as <code>cooling_balance</code>.</p>
<div class="box find"><div class="lab">gates &mdash; all passed</div>
<b>G1 byte-identical</b> (default): <code>dictionary.jsonl</code> sha256 <code>830b691a…</code> identical with
vs without the shadow, and with the keyword added (two independent re-gates). <b>G2 unit:</b>
<code>test/test_r1_shadow.py</code> <b>14/14</b> (criteria + parse/alias/validation + decision). <b>G3
regression:</b> <code>pytest -m "not stress"</code> <b>588 passed</b>. <b>Drive end-to-end:</b>
<code>transition_trigger=blowout</code> on <code>simple_cluster</code>: R1 fired &rarr; phase 1c (2.1 s) &rarr;
momentum, a clean hand-off.</div>

<h3>3.2&nbsp; Live shadow on all 8 configs &mdash; blowout fires for every in-cloud config</h3>
<p>The in-code shadow, run live on all eight configs (the six normal clouds + the two heavy 5e9 clouds),
confirms the offline preview to \(|\Delta t|=0\): <b>blowout fires for every in-cloud config</b> (0.012&ndash;3.66
Myr), and is R1's operative criterion there. <b>Eb-peak never fires in-cloud</b> for normal clouds
(\(\dot E_b\) stays positive &mdash; \(E_b\) is monotonic, exactly the part-3 result); it covers the
heavy-cloud end, where it is a <b>phase-1a event</b> (the 1b shadow is empty for the 5e9 clouds because
production's \(E_b\le0\) collapse stop precedes the 1b shadow site). The cooling ratio's minimum across all
rows is <b>0.283</b> &mdash; the current trigger never fires.</p>
<figure>__FIG_R1__<figcaption>R1 shadow preview, live-confirmed: per config, where the transition <i>would</i>
hand off &mdash; \(\star\) blowout (\(R_2>r_{\text{cloud}}\)) for the six normal clouds, \(\blacklozenge\)
Eb-peak for the two heavy 5e9 clouds &mdash; against the grey bar showing how long the <i>current</i> trigger
keeps the run energy-driven (it never fires). R1 gives a finite, defensible exit where the cooling-balance
trigger gives none. Pure read of <code>../cleanroom/data/c0_*_h0.csv</code> + <code>traj/h4_traj_*_V0.csv</code>
via <code>make_pt4_figures.py</code>.</figcaption></figure>
<table><thead><tr><th>config</th><th>blowout fires @ [Myr]</th><th>\(R_2/r_{\text{cloud}}\)</th>
<th>Eb-peak in-cloud?</th><th>first to fire</th></tr></thead><tbody>
<tr><td>small_dense_highsfe</td><td>0.0117</td><td>1.02</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>simple_cluster</td><td>0.0902</td><td>1.02</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>midrange_pl0</td><td>0.392</td><td>1.01</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>pl2_steep</td><td>0.840</td><td>1.05</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>be_sphere</td><td>0.856</td><td>1.01</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>3.66</td><td>1.00</td><td class="loss">no</td><td>blowout</td></tr>
<tr><td>fail_repro / fail_helix (5e9)</td><td>&mdash; (empty 1b)</td><td>&mdash;</td><td>Eb-peak is a <b>1a</b> event</td><td>&mdash;</td></tr>
</tbody></table>
<p class="small muted">From <code>r1shadow/r1_shadow_summary.csv</code> + <code>r1shadow/shadow_*.csv</code>
(8 configs). Offline blowout epoch matches the in-code <code>blowout_t</code> to \(|\Delta t|=0\).</p>
<div class="box over"><div class="lab">caveat &mdash; what is and isn't validated</div>R1 gives a defensible,
finite energy&rarr;momentum transition for every in-cloud config (where the cooling-balance trigger gives
none), and it is <b>opt-in</b> &mdash; default <code>cooling_balance</code>, byte-identical, committing nothing
until selected. <b>Validated:</b> the <i>drive</i> hands off cleanly on <code>simple_cluster</code>
(1b&rarr;1c&rarr;momentum). <b>Not yet validated:</b> the drive on the other configs, and especially the
heavy-cloud <b>Eb-peak hand-off into 1c</b> (a phase-1a event the 1b shadow can't cover) &mdash; the Path-2
continuity question that is the remaining make-or-break for using R1 as a <i>default</i>. That is future work.</div>
"""

SEC_CLAMP = r"""
<h2 id="clamp">4 &middot; Legacy vs hybr, in depth &mdash; the transition is a <em>solver</em> artifact</h2>
<p>H1 named the legacy&ndash;hybr divergence the &beta;-clamp. With the fix shipped, it is worth showing
&mdash; on the <b>current</b> version &mdash; exactly what the clamp does, and what it does <i>not</i>. Run
each config both ways (<code>betadelta_solver=legacy</code> vs <code>hybr</code>):</p>
<figure>__FIG_CLAMP_SOLVER__<figcaption>Per config, the <b>outcome</b> (cooling ratio, left axis) and the
<b>mechanism</b> (<code>cool_beta</code> &beta;, right axis; shaded = the legacy clamp box \([0,1]\)),
legacy (red) vs hybr (blue). Legacy&rsquo;s ratio dives through the 0.05 threshold and the run transitions
(5/6 configs); hybr floors at \(0.28\!-\!0.49\) and never does. <b>Circles</b> mark
\(R_2>r_{\text{cloud}}\) (blowout &mdash; the real geometric transition; legacy and hybr nearly coincident,
except where legacy has already exited). The reason for the ratio split is on the right axis:
legacy &beta; sits at the clamp edge while hybr&rsquo;s climbs to \(\sim\!+4\), out of the box. Pure read of
<code>c0_*_{legacy,h0}.csv</code> via <code>make_pt4_figures.py</code>.</figcaption></figure>
<p><b>Mechanism.</b> The clamped &beta; (peak \(\beta_{\max}\!\approx\!0.85\!-\!1.0\), pinned at the bound)
holds \(L_{\text{loss}}\) artificially high, so \((L_{\text{gain}}-L_{\text{loss}})/L_{\text{gain}}\) falls to
\(\le 0\); the unbounded root (\(\beta\!\to\!+3.4\!-\!4.6\)) collapses \(L_{\text{loss}}\) and the ratio
recovers. The clamped solver is also numerically <i>worse</i>: &beta;&ndash;&delta; convergence
\(0.13\!-\!0.47\) vs hybr&rsquo;s \(\sim\!0.65\), and it logs fewer implicit segments (it exits early).</p>
<figure>__FIG_SOLVER_STATS__<figcaption>Solver statistics, legacy vs hybr (current <code>c0</code> data):
implicit segments (legacy fewer &mdash; it exits early), &beta;&ndash;&delta; convergence fraction (legacy far
lower), cooling-ratio minimum (legacy \(\le\!0.05\) &rArr; transitions; hybr floors above), and peak &beta;
(legacy pinned at the clamp bound \(1\); hybr escapes to \(\sim\!4\)).</figcaption></figure>
<p><b>Where it&rsquo;s the same.</b> The mechanical input is solver-independent &mdash; \(L_{\text{mech}}\) is
identical legacy vs hybr; only the bubble&rsquo;s <i>response</i> differs. The grid confirms it across
\(E_b\), \(L_{\text{loss}}\), \(L_{\text{mech}}\), PdV \(=4\pi R_2^2 v_2 P_b\), and the shell radius \(R_2\):
\(L_{\text{mech}}\) overlaps; legacy&rsquo;s \(L_{\text{loss}}\) stays elevated; and legacy&rsquo;s traces
truncate at its early transition while hybr persists to the cap.</p>
<figure>__FIG_LEGHYBR_GRID__<figcaption>Legacy (red) vs hybr (blue), six configs (rows) &times; five quantities,
log&ndash;log. \(L_{\text{mech}}\) is the same (same feedback); the divergence is the bubble response. Circles
mark \(R_2>r_{\text{cloud}}\) (blowout). Pure read of <code>c0_*_{legacy,h0}.csv</code> via
<code>make_pt4_figures.py</code>.</figcaption></figure>
<p><b>The subtlety (H5).</b> It is <b>not the clamp width.</b> Widening the box \([0,1]\!\to\![-20,20]\) does
<i>not</i> remove the crossing &mdash; <code>small_dense</code> still crosses at the identical \(t=0.0242\) for
every box width (<code>h5clamp/H5_FINDINGS.md</code>). The legacy grid + L-BFGS-B simply lands at a low/edge
&beta; that holds \(L_{\text{loss}}\) high <i>wherever</i> the box is; the clamp edge merely coincides with
where the local solver converges. So the artifact is the <b>local solver</b>, not the bound &mdash; only the
unbounded <b>global</b> root (hybr) escapes it.</p>
<figure>__FIG_RUN_COST__<figcaption>Current-version run cost (hybr): wall-clock runtime, implicit segments, and
per-segment cost per config (<code>r1_shadow_summary.csv</code>). <code>large_diffuse</code> dominates the
wall-clock (it runs furthest before blowout); the \(5\times10^9\) clouds collapse in \(\sim\!1\) min. Caveat:
<code>stop_t</code> is per config (just past each blowout), so wall-clock is to-blowout, not matched-\(t\); the
per-segment cost factors that out.</figcaption></figure>
<p><b>Verdict.</b> Legacy&rsquo;s &ldquo;let cooling decide&rdquo; is not real cooling physics &mdash; it is
where the clamped local solver lands. The correct solver shows no cooling transition exists for these clouds;
the only physical end-of-energy-phase is geometric blowout / the \(E_b\)-peak (the R1 criteria).</p>
"""

SEC_CLOSE = r"""
<h2 id="close">Closing &mdash; the s1 arc, end to end</h2>
<p>This is where the storyline lands. The β&ndash;δ implicit solver was <b>repaired</b> (chapter 1); it was
made <b>fast</b> (chapter 2); part 3 then discovered it can <b>never leave the energy phase</b> &mdash; the
cooling-balance trigger tests for an event that does not occur, so the transition looked geometric, not
thermal. Part 4 stress-tested that verdict and it held: it is <b>not a <code>Lcool</code> bug</b> (H1 &mdash;
byte-identical integral; surge-then-collapse; the β-clamp is the whole legacy/hybr difference) and <b>not an
artificial rCloud wall</b> (H2 &mdash; a clean phase switch whose removal cannot create a cooling event,
because cooling is set by local density, not cloud size). With the diagnosis confirmed, the chapter ends by
<b>giving the energy phase an exit</b>: R1, a blowout + Eb-peak transition built on events the model actually
produces, shipped shadow-first and byte-identical, then exposed as an opt-in <code>transition_trigger</code>
keyword. The solver that could not leave the energy phase now has a diagnosed, defensible, opt-in door out
&mdash; with the heavy-cloud Eb-peak hand-off the one piece left to validate.</p>
"""

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/pt4/</code> &mdash; reproducible
without re-running the (hours-long) hybr sims. Each figure is a pure read of a committed CSV; each number
traces to one of the three pt4 audit docs.</p>
<table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>H1_lcool_audit.md</code></td><td>H1: read-only audit of <code>bubble_LTotal</code> (byte-identical history, surge-then-collapse, β-clamp)</td></tr>
<tr><td><code>analyze_lcool_direction.py</code> &middot; <code>trajectory_probe.py</code> &rarr; <code>H1_lcool_direction_summary.csv</code></td><td>per-config \(L_{\text{loss}}/L_{\text{gain}}\) direction, ratio min/final, 0.05-crossing, hybr vs legacy</td></tr>
<tr><td><code>H2_rcloud_audit.md</code></td><td>H2: rCloud is derived; crossing is a clean phase switch; cooling set by local density</td></tr>
<tr><td><code>h2_rcloud_compute.py</code> &rarr; <code>h2_rcloud_edge.csv</code> &middot; <code>h2_analyze.py</code> &rarr; <code>h2_crossing_summary.csv</code></td><td>rCloud + edge-density drop via the production pipeline; per-config crossing / ratio_min</td></tr>
<tr><td><code>sc_baseline.param</code> &middot; <code>sc_bigcloud.param</code> &rarr; <code>h2_sc_{baseline,bigcloud}.csv</code></td><td>the clean enlarged-rCloud experiment (5.2&times; cloud at fixed feedback); the matched-\(R_2\) test</td></tr>
<tr><td><code>r1shadow/R1_FINDINGS.md</code> &middot; <code>R1_SHADOW_PLAN.md</code></td><td>R1: shadow + opt-in keyword findings and plan</td></tr>
<tr><td><code>r1shadow/r1_shadow_summary.csv</code> &middot; <code>shadow_*.csv</code> (8) &middot; <code>GATE_RESULT.txt</code></td><td>live shadow firing epochs (all 8 configs) + the byte-identical gate</td></tr>
<tr><td><code>make_pt4_figures.py</code> &rarr; <code>figures/*.png</code></td><td>the seven figures above (each a pure read of a CSV)</td></tr>
<tr><td>Production (R1): <code>run_energy_implicit_phase.py</code> &middot; <code>registry.py</code>/<code>default.param</code> (<code>transition_trigger</code>) &middot; <code>test/test_r1_shadow.py</code></td><td>the shadow + opt-in drive (default-off, byte-identical)</td></tr>
</tbody></table>
<p class="small muted">Rebuild this report:
<code>python docs/dev/transition/pt4/make_pt4_transition_report.py</code>
(figures regenerate via <code>python docs/dev/transition/pt4/make_pt4_figures.py</code>).</p>
<footer>TRINITY implicit&rarr;momentum transition investigation, part 4 &middot; re-examine the verdict, then
ship an opt-in exit &middot; every number traces to a committed pt4 audit doc / CSV under
<code>docs/dev/transition/pt4/</code> &middot; light-mode, MathJax-rendered.</footer>
"""


def main():
    parts = [HEAD, HERO, FLOW_INTRO]
    for i, (q, a) in enumerate(STEPS):
        parts.append(f'<div class="step"><div class="q">{q}</div><div class="a">{a}</div></div>')
        if i < len(STEPS) - 1:
            parts.append('<div class="arrow">&#9660;</div>')
    parts.append("</div>")
    parts.append(FLOW_OUTRO)
    parts += [SEC_RECAP, SEC_H1, SEC_H2, SEC_R1, SEC_CLAMP, SEC_CLOSE, SEC_REPRO]
    parts.append("</div></body></html>")
    html = "".join(parts)

    for token, (fname, alt) in FIGURES.items():
        html = html.replace(token, img(fname, alt))
    assert "__FIG_" not in html, "unreplaced figure placeholder remains"

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    # This report is a new chapter of the S1 storyline book; the parent will
    # register it and rebuild via:
    #   python docs/dev/html-insights/build_storylines.py


if __name__ == "__main__":
    main()
