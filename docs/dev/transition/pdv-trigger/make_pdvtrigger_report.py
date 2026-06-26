#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pdvtrigger_report.py — build the self-contained HTML report for the
"PdV in the transition trigger" -> unresolved-interface-cooling-closure
workstream (docs/dev/transition/pdv-trigger/).

This is the sequel report to the s1 clean-room investigation: that one ended on
"the live question is the geometric / Eb-peak handoff and the missing
mixing-layer cooling." This report answers the missing-mixing-layer half —
boost the LOSS, not the trigger, without double-counting — and then carries it
through to this session's verdict: a cooling-MAGNITUDE knob, constant OR
Da-coupled, is NOT the transition trigger for normal clouds. theta_target(Da)
was TESTED (offline proxy + a gate-validated real-Da replay) and REFUTED; the
operative handoff is geometric blowout, and the cooling boost only corrects the
magnitude THROUGH that handoff.

Figures are embedded as base64 so the file is standalone (downloadable, opens
offline; MathJax loads from CDN for the formulas). They are NOT regenerated
here. The original storyline figures live in storyline_figs/ (built by
storyline_figs/make_storyline_figs.py); the 4 NEW figures
(theta_vs_density / fmix_vs_density / da_screen / da_replay) live at the
workstream ROOT (built by data/make_*.py). Every number in the prose traces to a
committed CSV in data/ or runs/data/ (verified 2026-06-24, extended 2026-06-25).

Self-contained / no usetex: the storyline merger
(docs/dev/html-insights/build_storylines.py) demands a single <h1>, a
<p class="sub"> subtitle, scoped <style>, and inline base64 figures. The book
template adds the three dev-doc banners, so this report carries the science only.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py   # -> pdvtrigger_report.html
    python docs/dev/html-insights/build_storylines.py                  # rebuild storyline_s5.html
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "storyline_figs"
OUT = HERE / "pdvtrigger_report.html"

# Figure tokens -> (path-relative-to-HERE, alt text). The first four live in
# storyline_figs/; the last four (this session's new figures) live at the
# workstream root. img() resolves either by joining onto HERE.
FIGURES = {
    "__FIG_FMIX__": (
        "storyline_figs/fig_fmix_convention.png",
        "f_mix needed to fire the handoff at blowout, with-PdV vs the consistent no-PdV trigger, per config",
    ),
    "__FIG_DOUBLE__": (
        "storyline_figs/fig_doublecount.png",
        "the single-count line vs the forbidden double-count region, with the max-closure and the MC draws",
    ),
    "__FIG_REGIME__": (
        "storyline_figs/fig_regime_split.png",
        "PdV/Lmech per config: normal clouds sub-critical near 0.45, the heavy 5e9 super-critical above 1",
    ),
    "__FIG_HEATMAP__": (
        "storyline_figs/fig_closure_heatmap.png",
        "frozen-trajectory fire-vs-blowout heatmap across the density grid: no single f_mix fires them all at blowout",
    ),
    "__FIG_FMIXDENS__": (
        "fmix_vs_density.png",
        "data-only scatter of the firing f_mix (1.36 to 3.81) vs ambient density: no horizontal constant line crosses all configs",
    ),
    "__FIG_THETADENS__": (
        "theta_vs_density.png",
        "TRINITY resolved L_cool/L_mech vs nCore with a schematic literature theta band (de-annotated, gap not quantified)",
    ),
    "__FIG_DASCREEN__": (
        "da_screen.png",
        "offline Da-shape proxy at blowout: non-monotonic in nCore, spread ~14x, fires dense clouds at birth -> NO-GO",
    ),
    "__FIG_DAREPLAY__": (
        "da_replay.png",
        "gate-validated real-Da replay at blowout: still non-monotonic, Da>>1 everywhere so theta_max*Da/(1+Da) saturates to a constant -> NO-GO",
    ),
    "__FIG_IDEAS__": (
        "ideas_comparison.png",
        "scoreboard of every transition-fix idea (constant f_mix, constant theta, theta_target(Da), live multiplier, kappa_eff Rung A probe, kappa_eff Rung B endgame) with verdict badges, plus three real-data evidence panels",
    ),
    "__FIG_KAPPA__": (
        "kappa_backreaction.png",
        "kappa_eff Rung A back-reaction on f1edge_hidens at matched t: f_kappa=2 raises L_cool x1.23-1.38 but dMdt x1.08-1.17 rides along, and a 2x kappa moves the loss ratio only +0.05-0.10 toward the 0.95 trigger",
    ),
}


def img(name, alt):
    b64 = base64.b64encode((HERE / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


# CSS lifted from the cleanroom report so the merged chapter matches s1's visual
# language (the merger scopes this under a per-chapter id, so class names are safe).
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
.box{border:1px solid var(--line);border-radius:9px;padding:13px 17px;margin:14px 0;}
.box .lab{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.hyp{background:#f5f2ff;border-color:#e0d8fb;} .hyp .lab{color:var(--hyp);}
.find{background:#eef9f1;border-color:#cdeed6;} .find .lab{color:var(--find);}
.over{background:#fff6ec;border-color:#f6dcbd;} .over .lab{color:var(--warn);}
.warnbox{background:#fdeef0;border-color:#f3ccd3;} .warnbox .lab{color:var(--bad);}
figure{margin:20px 0;} figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block;}
figcaption{color:var(--mut);font-size:13.5px;margin-top:7px;text-align:center;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;}
.tablewrap{overflow-x:auto;max-width:100%;}
.tablewrap table{margin:0;}
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
    "<title>TRINITY &mdash; interface cooling without double-counting</title>"
    + MATHJAX + "<style>" + CSS + "</style></head><body><div class=\"wrap\">"
)

HERO = r"""
<span class="tag">TRINITY &middot; PdV-in-the-trigger &rarr; unresolved interface cooling</span>
<h1>Boost the loss, not the trigger &mdash; and the trigger is blowout, not cooling</h1>
<p class="sub">The sequel to the clean-room transition study. That investigation ended on a single open
question &mdash; the geometric / \(E_b\)-peak handoff and the <i>missing mixing-layer cooling</i>. This
report answers the cooling half &mdash; what the maintainer&rsquo;s Paper-II note (&ldquo;adding unresolved
interface cooling to TRINITY without double-counting&rdquo;) buys, where it fires, and why a single constant
is not enough &mdash; and then carries it to this session&rsquo;s verdict: a cooling-<i>magnitude</i> knob,
constant <b>or</b> \(\mathrm{Da}\)-coupled, is <b>not</b> the transition trigger for normal clouds. Verified
2026-06-24, extended with live runs and the \(\theta_{\text{target}}(\mathrm{Da})\) test 2026-06-25; numbers
trace to committed CSVs under <code>docs/dev/transition/pdv-trigger/{data,runs/data}/</code>; the wiring
shipped opt-in and gated.</p>

<div class="tldr">
<p style="margin:0"><b>TL;DR.</b> s1 proved normal clouds never trip the radiative cooling trigger
\((L_{\text{mech}}-L_{\text{cool}})/L_{\text{mech}}<0.05\) &mdash; they hand off by <b>geometric blowout</b>, and
the live gap was the unresolved turbulent-mixing-layer cooling a 1D model can&rsquo;t see. The fix is to
<b>add the missing loss</b>, not to tune the trigger. The earlier results still stand:
<b>(1) don&rsquo;t double-count</b> &mdash; the <code>theta_target</code> closure
\(L_{\text{loss}}^{\text{eff}}=\max(L_{\text{cool}}+L_{\text{leak}},\;\theta\,L_{\text{mech}})\) is single-count
by construction (a \(5\times10^5\)-draw Monte-Carlo finds <b>0</b> double-count draws); <b>(2) the headline
boost was a trigger-convention slip</b> &mdash; consistently (PdV-out) the boost is
\(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\), not \(1.1\!-\!1.5\); <b>(3) no constant fires the grid</b> &mdash; the
firing \(f_{\text{mix}}\) spans \(\sim\!1.4\!\to\!3.8\) across density. <b>But the constant target is
degenerate, and the coupled one is refuted.</b> A flat \(\theta\!\approx\!0.95\) target is <i>bit-identical</i>
to the 0.95 trigger TRINITY already has, so it adds nothing; and
\(\theta_{\text{target}}(\mathrm{Da})\) &mdash; tested this session offline <i>and</i> by a gate-validated
real-Da replay &mdash; is <b>NO-GO</b> (real \(\mathrm{Da}\) is non-monotonic in density and \(\gg\!1\)
everywhere, so \(\theta_{\max}\mathrm{Da}/(1+\mathrm{Da})\) saturates back to a constant). <b>The pivot:</b> for
normal clouds the trigger is <b>geometric blowout</b> (TRINITY&rsquo;s default already does this); the cooling
boost corrects cooling <b>magnitude</b> through the handoff, it does not fire it. Live matched-\(t\) runs
confirm: no constant fires cooling across density. The heavy 5e9 cloud is super-critical and hands off via the
PdV / \(E_b\)-peak turnover. <b>Latest (2026-06-26):</b> the structural \(\kappa_{\text{eff}}\) endgame&rsquo;s
first rung is built and gated (&sect;11) &mdash; it confirms a faithful, state-coupled \(\kappa_{\text{eff}}\)
is required, not a scalar.</p>
</div>

<figure>__FIG_IDEAS__<figcaption><b>The whole storyline at a glance.</b> Every transition-fix idea tried,
left&rarr;right, with its verdict: the three scalar knobs (constant \(f_{\text{mix}}\), constant \(\theta\),
\(\theta_{\text{target}}(\mathrm{Da})\)) are <span style="color:#b3392f"><b>refuted</b></span>, the live
multiplier is <span style="color:#b3801f"><b>partial</b></span> (magnitude-only, mistimes by density), the
\(\kappa_{\text{eff}}\) Rung-A structural probe is <span style="color:#2a8aa8"><b>this work</b></span>
(gated/byte-identical-off), and the faithful Rung-B \(\kappa_{\text{eff}}\) is the
<span style="color:#3a8a3f"><b>endgame</b></span>. The three lower panels are the real-data evidence for the
key verdicts (\(f_{\text{mix}}\) spread, \(\mathrm{Da}\) saturation, and the Rung-A cooling-vs-evaporation
coupling). Built by <code>data/make_ideas_comparison.py</code> from the committed CSVs.</figcaption></figure>
"""

SEC_SETUP = r"""
<h2 id="setup">1 &middot; The setup &mdash; where s1 left off</h2>
<p>TRINITY integrates the bubble interior energy with
\[ \frac{dE_b}{dt} = L_{\text{mech}} - L_{\text{cool}} - \underbrace{4\pi R_2^2\,v_2\,P_b}_{\text{PdV}} - L_{\text{leak}} \]
(<code>get_betadelta.py:434</code>), so the PdV work term is <b>already</b> in the energy evolution. The default
energy&rarr;momentum handoff, though, watches only the <b>radiative</b> ratio
\[ \frac{L_{\text{mech}}-L_{\text{cool}}}{L_{\text{mech}}} < 0.05 \qquad\text{(cooling }\ge 95\%\text{)} \]
(<code>run_energy_implicit_phase.py:1200</code>) &mdash; <b>no PdV</b>. The clean-room investigation (s1) showed
this ratio never reaches \(0.05\) for normal clouds; they hand off by <b>geometric blowout</b> instead, and the
retained-energy fraction plateaus at \(0.25\!-\!0.40\), far above the observed \(0.01\!-\!0.1\) band. s1 named
the cause but did not fix it: a 1D model has no <b>turbulent mixing layers</b>, so it under-counts the
interface cooling that the literature says dominates real wind bubbles. This report is that fix &mdash; and the
test of whether the fix is a <i>trigger</i> or a <i>magnitude</i> correction.</p>
<div class="box hyp"><div class="lab">the move</div>The handoff isn&rsquo;t mis-tuned &mdash; it tests for an
event (radiative balance) that the under-cooled 1D bubble never reaches. So don&rsquo;t retune the threshold;
<b>add the missing cooling</b>. The whole report is about doing that <i>correctly</i>: which channel to boost, by
how much, and &mdash; the question this session settles &mdash; whether the added cooling <i>fires</i> the
transition (a trigger) or only gets \(E_b,P_b,R_2,v_2\) right <i>through</i> a transition that something else
drives (a magnitude correction).</p></div>
"""

SEC_DOUBLE = r"""
<h2 id="doublecount">2 &middot; Don&rsquo;t double-count</h2>
<p>A fractional loss \(\theta\equiv L_{\text{loss}}/L_{\text{mech}}\in[0,1]\) can <b>replace</b> the explicit
\(L_{\text{cool}}\) (the Lancaster-style \((1-\theta)\,L_{\text{mech}}\) reduced model), but it must <b>not be
added on top of</b> subtracting \(L_{\text{cool}}\). TRINITY already subtracts an explicit \(L_{\text{cool}}\);
stacking a \((1-\theta)\,L_{\text{mech}}\) input-rescale on top removes the <i>same</i> energy twice &mdash;
\(2\theta\,L_{\text{mech}}\) at consistency, driving the net energy negative for \(\theta>\tfrac12\). The fix is
to <b>add only the missing part</b>.</p>
<p>The shipped <code>theta_target</code> closure does exactly that:
\[ L_{\text{loss}}^{\text{eff}} = \max\!\big(L_{\text{cool}}+L_{\text{leak}},\;\theta_{\text{target}}\,L_{\text{mech}}\big). \]
It is single-count <b>by construction</b>: \(L_{\text{loss}}^{\text{eff}}/L_{\text{mech}}=\max(L_{\text{cool}}/L_{\text{mech}},\,\theta)\),
never the forbidden \(L_{\text{cool}}/L_{\text{mech}}+\theta\). It tops the resolved cooling up to the target where
the model under-counts, and switches <b>off</b> where the resolved cooling already exceeds the target. A
\(5\times10^5\)-draw Monte-Carlo over \((L_{\text{cool}}/L_{\text{mech}},\,\theta)\) confirms it: <b>0</b> draws
enter the double-count region (<code>data/doublecount_mc.csv</code>).</p>
<figure>__FIG_DOUBLE__<figcaption>The single-count diagram. The closure rides the \(\max(\cdot)\) envelope (the
single-count line) and never enters the shaded \(2\theta\) double-count region; the \(5\times10^5\) Monte-Carlo
draws (all on the single-count line, <code>frac_on_single_line=1.0</code>) land outside it everywhere. Read of
<code>data/doublecount_mc.csv</code> via <code>storyline_figs/make_storyline_figs.py</code>.</figcaption></figure>
<div class="box find"><div class="lab">consistency contract (note &sect;Code-level)</div>One helper,
<code>effective_Lloss</code>, returns the same \(L_{\text{loss}}^{\text{eff}}\) to <b>all three</b> sites &mdash; the
\(\beta\!-\!\delta\) residual, the \(E_b\) ODE, and the transition trigger. If the boosted loss appeared in the
trigger but not the ODE (or vice-versa) the run would be internally inconsistent; the single helper is what makes
&ldquo;boost the loss&rdquo; well-defined.</div>
"""

SEC_CONVENTION = r"""
<h2 id="convention">3 &middot; The trigger-convention fix &mdash; and the degeneracy it exposes</h2>
<p>The note&rsquo;s reported boost, \(f_{\text{mix}}\!\approx\!1.1\!-\!1.5\), was computed
with the <b>PdV term inside the screening trigger</b> &mdash; i.e. solving
\((L_{\text{mech}}-f\,L_{\text{cool}}-\text{PdV})/L_{\text{mech}}=0.05\). But that is <b>inconsistent with the
note&rsquo;s own recommended trigger</b>, which keeps PdV <i>out</i>. The physics reason is clean: <b>PdV is
reversible</b> &mdash; the work done on the shell is recoverable as shell momentum &mdash; whereas <b>cooling is
irreversible</b>. You fire the energy&rarr;momentum handoff on the irreversible channel, so PdV belongs in the
ODE only, never in the trigger.</p>
<p>Restore consistency &mdash; drop PdV from the screening ratio &mdash; and the required boost becomes
\[ f_{\text{mix}} = \frac{0.95}{L_{\text{cool}}/L_{\text{mech}}}\Bigg|_{\text{blowout}}, \]
i.e. lift the resolved \(L_{\text{cool}}/L_{\text{mech}}\) up to the threshold \(\theta\approx0.95\) at the blowout
epoch. That gives <b>\(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)</b> across the compact-to-mid clouds &mdash; larger
than the note&rsquo;s with-PdV column by exactly the \(\text{PdV}/L_{\text{mech}}\) offset it folded in.</p>
<figure>__FIG_FMIX__<figcaption>The convention fix, per config. The <b>with-PdV</b> screen (the note&rsquo;s
Table-2 column, \(1.1\!-\!1.5\)) understates the boost; the consistent <b>no-PdV</b> trigger gives
\(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\) (compact five) up to \(3.8\) for the diffuse \(10^7\,M_\odot\) cloud,
because that cloud cools least (\(L_{\text{cool}}/L_{\text{mech}}\approx0.25\) at blowout, so it needs the most
help). Both columns from <code>data/fmix_table.csv</code>.</figcaption></figure>
<div class="tablewrap"><table><thead><tr><th>config</th>
<th>\(L_{\text{cool}}/L_{\text{mech}}\) @blowout</th><th>\(\text{PdV}/L_{\text{mech}}\) @blowout</th>
<th>\(f_{\text{mix}}\) with-PdV (note)</th><th>\(f_{\text{mix}}\) no-PdV (consistent)</th></tr></thead><tbody>
<tr><td>small_dense_highsfe</td><td>0.697</td><td>0.182</td><td>1.10</td><td class="win">1.36</td></tr>
<tr><td>simple_cluster</td><td>0.667</td><td>0.206</td><td>1.12</td><td class="win">1.42</td></tr>
<tr><td>midrange_pl0</td><td>0.610</td><td>0.219</td><td>1.20</td><td class="win">1.56</td></tr>
<tr><td>be_sphere</td><td>0.511</td><td>0.308</td><td>1.26</td><td class="win">1.86</td></tr>
<tr><td>pl2_steep</td><td>0.342</td><td>0.441</td><td>1.49</td><td class="win">2.78</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>0.250</td><td>0.169</td><td>3.13</td><td class="win">3.81</td></tr>
</tbody></table></div>
<p class="small muted">Numbers read from <code>data/fmix_table.csv</code> (builder
<code>data/make_fmix_table.py</code>, from <code>pdv_combined_trigger.csv</code>). The consistent column is the
recommended one; the with-PdV column is kept only to show the size of the convention slip.</p>
<div class="box warnbox"><div class="lab">the degeneracy &mdash; a <i>constant</i> target adds nothing</div>
That &ldquo;consistent&rdquo; formula hides a trap. The \(0.95\) in
\(f_{\text{mix}}=0.95/(L_{\text{cool}}/L_{\text{mech}})\) <b>is the trigger threshold itself</b> &mdash; the
handoff fires when \(L_{\text{loss}}/L_{\text{mech}}\) reaches \(0.95\). So calibrating the boost to a <i>flat</i>
literature \(\theta\approx0.95\) is <b>bit-identical to the <code>fmix_no_pdv</code> column by construction</b>: it
just restates &ldquo;boost the resolved loss until it hits the threshold the trigger already has.&rdquo; The
mixing-layer literature (Lancaster) says \(\theta\) at GMC density is high and roughly <b>flat</b>
(\(\sim\!0.9\!-\!0.99\)) &mdash; which is exactly why a <i>constant</i> target, whether spelled \(f_{\text{mix}}\)
or \(\theta\), is <b>degenerate</b> with the \(0.95\) trigger TRINITY already runs. A constant adds no quantitative
content over &sect;3&rsquo;s table. The only non-degenerate upgrade is a target that <i>varies</i> along the
trajectory &mdash; tested, and refuted, in &sect;6.</div>
<figure>__FIG_FMIXDENS__<figcaption>The firing \(f_{\text{mix}}\) per config, data-only scatter (not a curve):
\(1.36\to3.81\) as density falls, no horizontal &ldquo;constant \(f_{\text{mix}}\)&rdquo; line crosses all six.
\(\text{pl2\_steep}\) and \(\text{simple\_cluster}\) share \(n_{\text{Core}}=10^5\) yet need \(2.78\) vs \(1.42\),
so there is no clean \(f_{\text{mix}}(n)\). Read of <code>data/fmix_table.csv</code> via
<code>data/make_fmix_spread_plot.py</code>.</figcaption></figure>
"""

SEC_REGIME = r"""
<h2 id="regime">4 &middot; The regime split &mdash; cooling boost vs PdV handoff</h2>
<p>The cooling boost is the right tool for <b>normal</b> clouds, but not for the heavy end. The control
parameter is \(\text{PdV}/L_{\text{mech}}\): it sits at \(\approx0.45\) for <i>every</i> normal cloud
(sub-critical &mdash; net energy keeps growing, \(E_b\) monotone, growth \(\sim\!1.5\!-\!2.4\times10^3\)), but at
\(\approx1.4\) for the heavy \(5\times10^9\,M_\odot\) <code>fail_repro</code> cloud (super-critical &mdash; PdV
<i>exceeds</i> \(L_{\text{mech}}\), \(E_b\) peaks and collapses, growth only \(1.014\times\)). A clean
sub-/super-critical split at \(\text{PdV}/L_{\text{mech}}=1\).</p>
<figure>__FIG_REGIME__<figcaption>\(\text{PdV}/L_{\text{mech}}\) per config. The six normal clouds cluster
sub-critical near \(0.45\) (median); the heavy 5e9 cloud is the lone super-critical point above unity. The whole
behavioural fork is which side of \(\text{PdV}/L_{\text{mech}}=1\) the cloud lands on. Read of
<code>data/pdv_regime_budget.csv</code>.</figcaption></figure>
<p>Consequently the two regimes need <b>different</b> handoffs. Normal clouds want the <b>cooling boost</b>
(consistent \(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)) to get their <i>magnitude</i> right near blowout. The heavy
cloud is the opposite &mdash; it cools so little (\(L_{\text{cool}}/L_{\text{mech}}\approx0.01\)) that it
<b>never</b> fires the cooling trigger even at \(f_{\text{mix}}=30\); it is PdV-dominated, so it hands off via the
<b>PdV / \(E_b\)-peak net-energy turnover</b> (\(L_{\text{mech}}-L_{\text{cool}}-\text{PdV}\le0\)) instead. Cooling
and PdV cover complementary halves of the mass range.</p>
<div class="box find"><div class="lab">the split, in one line</div>Normal clouds (sub-critical,
\(\text{PdV}/L_{\text{mech}}\approx0.45\)) &rarr; <b>cooling-magnitude boost</b>. Heavy clouds (super-critical,
\(\text{PdV}/L_{\text{mech}}>1\)) &rarr; <b>PdV / \(E_b\)-peak handoff</b>. The same diagnostic that says
&ldquo;PdV is negligible&rdquo; is false everywhere also says PdV is <i>decisive</i> only past the unity
crossing.</div>
"""

SEC_CLOSURE = r"""
<h2 id="closure">5 &middot; A constant knob is not enough (and a constant target is degenerate)</h2>
<p>If one constant \(f_{\text{mix}}\) fired every config at blowout, a single calibrated float would close the
problem. It does not. A <b>frozen-trajectory screen</b> (<code>data/closure_test.csv</code>) solves, on each
committed unboosted trajectory, for the \(f_{\text{mix}}\) that fires that config at its blowout epoch. The
answer spans <b>\(\sim\!1.4\to3.8\)</b> across the density grid: dense clouds already cool efficiently
(\(L_{\text{cool}}/L_{\text{mech}}\approx0.7\) at blowout, so they need only a small lift), diffuse clouds barely
cool (\(\approx0.25\), so they need a big one). No single constant fires them all at the right epoch.</p>
<figure>__FIG_HEATMAP__<figcaption>The frozen-trajectory fire-vs-blowout heatmap (config &times; boost value). At a
constant \(f_{\text{mix}}\!\approx\!2\), the compact/dense clouds fire right at blowout while the diffuse ones fire
well before it (offsets of order \(-1\) to \(-3.6\) Myr). The density ordering is monotone. Read of
<code>data/closure_test.csv</code>.</figcaption></figure>
<div class="box over"><div class="lab">everywhere: this is a FROZEN-TRAJECTORY SCREEN, not a forecast</div>
The screen freezes the <i>unboosted</i> trajectory and asks where a boost would fire on it. But boosting cooling
lowers \(P_b\!\to\!\) lowers \(\text{PdV}\,(\propto P_b)\!\to\!\) changes \(E_b(t),R_2(t),v_2(t)\) &mdash; it
<b>moves blowout itself</b>. So these fire-times <b>bound</b> the knob; they do not <b>forecast</b> it. The honest
test is a full boosted run at matched \(t\) (&sect;9, now DONE). Read every number in this section as a
screen.</div>
<p>The density-ordered spread looks like the argument for <i>coupling</i> the target to the bubble state. The
natural candidate is a <b>state-dependent \(\theta_{\text{target}}(\mathrm{Da})\)</b> &mdash; a Damk&ouml;hler
number \(\mathrm{Da}=\tau_{\text{turb}}/t_{\text{cool}}\) (Tan/Oh/Gronke) &mdash; with the form
\(\theta_{\text{target}}=\theta_{\max}\,\mathrm{Da}/(1+\mathrm{Da})\), which recovers the El-Badry (high-\(\mathrm{Da}\),
interface-dominated) and Weaver (low-\(\mathrm{Da}\), energy-driven) limits from one ratio. That hypothesis is the
one this session put on trial &mdash; <b>and refuted</b> (&sect;6). Note already that a <i>constant</i> target is
no help: by the &sect;3 degeneracy it is bit-identical to the \(0.95\) trigger TRINITY already has.</p>
"""

SEC_DA = r"""
<h2 id="datest">6 &middot; \(\theta_{\text{target}}(\mathrm{Da})\) tested &mdash; and refuted</h2>
<p>The density-ordered spread of &sect;5 is suggestive, but a hypothesis is not a result. This session tested the
coupled target \(\theta_{\text{target}}=\theta_{\max}\,\mathrm{Da}/(1+\mathrm{Da})\) in two steps, both reusing
trinity&rsquo;s own committed trajectories (no full re-runs). Both are <b>NO-GO</b>.</p>

<h3>Step A &mdash; offline proxy (<code>data/make_da_screen.py</code>)</h3>
<p>Under a fixed characteristic interface temperature, \(\mathrm{Da}\) collapses to the shape proxy
\(\mathrm{Da_{shape}}=(R_2/v_2)\cdot P_b\) (units absorbed by a swept normalization &mdash; a unit-independent
structural test). It fails two ways: \(\mathrm{Da_{shape}}\) at blowout is <b>non-monotonic in \(n_{\text{Core}}\)
and spans \(\sim\!14\times\)</b> (\(\text{pl2\_steep}\) at \(10^5\) gives \(4222\), <i>below</i>
\(\text{large\_diffuse}\) at \(10^2\) with \(4601\); \(\text{simple\_cluster}\) at \(10^5\) gives \(54690\)), and it
is large early (high \(P_b\) at small \(R_2\)), so any normalization that pushes the diffuse configs to
\(\theta\approx0.95\) fires the dense ones <b>at birth</b>. \(0/6\) valid sustained fires anywhere on the grid.</p>
<figure>__FIG_DASCREEN__<figcaption>The offline Da-shape screen at blowout. Non-monotonic in density, spread
\(\sim\!14\times\); a single crossing \(\theta_{\max}\mathrm{Da}/(1+\mathrm{Da})=0.95\) cannot coincide with
blowout across the grid. Read of <code>data/da_screen.csv</code> via <code>data/make_da_screen.py</code>.
</figcaption></figure>

<h3>Step A&prime; &mdash; the gate-validated real-Da replay (<code>data/make_da_replay.py</code>)</h3>
<p>The proxy collapses away the per-config \(T_{\text{int}}\) and \(\Lambda\) &mdash; the very quantities that
might separate the configs &mdash; so it does not by itself refute \(\theta_{\text{target}}(\mathrm{Da})\). The
decisive test recomputes the <b>real</b> \(\mathrm{Da}\) by <i>replaying trinity&rsquo;s own interface cooling</i>
on each committed trajectory row: re-invoke \(T_{\text{int}}(r),\Lambda(T_{\text{int}}),
n_{\text{int}}=P_b/(k_B T_{\text{int}})\Rightarrow t_{\text{cool,int}}\), then
\(\mathrm{Da}=(R_2/v_2)/t_{\text{cool,int}}\).</p>
<div class="box find"><div class="lab">the gate PASSES &mdash; the real Da is trustworthy</div>The replay
reproduces the logged <code>bubble_Lloss</code> to \(\le 3.9\times10^{-5}\) (tol \(10^{-3}\)) and the interface
zone \(L_3\) is <b>bit-identical</b> (reldiff \(0\)). So the verdict below is a <b>real refutation</b>, not a proxy
artifact.</div>
<p>Verdict: <b>also NO-GO</b> &mdash; \(0/6\) valid sustained fires under any single \((C,\theta_{\max})\). Three
decisive reasons:</p>
<ul>
<li><b>\(T_{\text{int}}\) is \(\sim\)constant across all configs (\(\sim\!21.4\!-\!22.6\) kK)</b> &mdash; the
radiative layer sits where \(\Lambda\) peaks, independent of cloud, so \(\mathrm{Da}\approx\text{proxy}\times\)const
and the proxy was a <i>good</i> approximation (its NO-GO carries over).</li>
<li><b>Real \(\mathrm{Da}\) at blowout is still NON-monotonic in \(n_{\text{Core}}\)</b> (\(\text{pl2\_steep}\)
\(10^5=4.7\times10^4\), <i>below</i> \(\text{large\_diffuse}\) \(10^2=5.6\times10^4\) and \(\text{midrange}\)
\(10^4=4.2\times10^5\); spread \(14\times\), \(4.7\times10^4\!-\!6.6\times10^5\)). No monotonic
\(\theta(\mathrm{Da})\) can order the configs by density.</li>
<li><b>\(\mathrm{Da}\gg1\) everywhere at blowout</b>, so \(\theta_{\max}\,\mathrm{Da}/(1+\mathrm{Da})\)
<b>saturates to \(\sim\!\theta_{\max}\) for every config</b> &mdash; it collapses to a <i>constant</i> target,
which is exactly the &sect;3 degeneracy that adds nothing.</li>
</ul>
<figure>__FIG_DAREPLAY__<figcaption>The gate-validated real-Da replay at blowout. Gate PASS (logged
<code>bubble_Lloss</code> reproduced to \(\le 3.9\times10^{-5}\); interface \(L_3\) bit-identical), so this is a
real refutation. Real \(\mathrm{Da}\) is non-monotonic in \(n_{\text{Core}}\) and \(\gg1\) everywhere &mdash;
\(\theta_{\max}\mathrm{Da}/(1+\mathrm{Da})\) saturates to a constant. Read of <code>data/da_replay.csv</code> via
<code>data/make_da_replay.py</code>.</figcaption></figure>
<div class="box warnbox"><div class="lab">result</div>Both a <b>constant</b> target (degenerate with the 0.95
trigger, &sect;3) and the <b>\(\mathrm{Da}\)-coupled</b> target (refuted here, gate-validated) fail as a
<i>trigger</i> mechanism. A cooling-magnitude knob is not what fires the energy&rarr;momentum transition for
normal clouds. The pivot &mdash; blowout is the trigger &mdash; is in &sect;10.</div>
"""

SEC_LIT = r"""
<h2 id="literature">7 &middot; Literature anchor &amp; reconciliation</h2>
<p>The physics content was checked against the mixing-layer / interface-cooling literature. The key
reconciliation this session: <b>the right GMC-density anchor is Lancaster&rsquo;s flat \(\theta\)-plateau, not an
El-Badry \(\sqrt{n}\) extrapolation.</b></p>
<ul>
<li><b>Lancaster+21a/b</b> &mdash; near-complete cooling of wind energy; at GMC density (\(n\sim10^2\!-\!10^6\)) the
right anchor is the <b>\(\theta\approx0.90\!-\!0.99\) plateau</b>, a derived, 3D-sim-validated result
(&ldquo;generic over \(>3\) dex in density&rdquo;) &mdash; <b>flat-and-high</b>. Fractal contact-discontinuity area,
\(D\sim2.5\!-\!2.7\); momentum enhancement \(\alpha_p\sim1.2\!-\!4\). This is the anchor for the schematic band
below.</li>
<li><b>El-Badry+19</b> &mdash; a <i>supernova-superbubble</i> paper (ambient \(n\sim0.1\!-\!10\)), so its
\(\theta(n)\) <b>must not be pushed to GMC densities</b>. Per the maintainer&rsquo;s updated note (PDF-checked
2026-06-25): \(\theta=(L_{\text{int}}/\dot E)/(11/5+L_{\text{int}}/\dot E)\) [Eq.&nbsp;37/38, <i>not</i> the old
&ldquo;Eq.&nbsp;35&rdquo;], \(\theta\) is <b>time-independent</b>, anchor \(\theta\approx0.61\) at
\(\lambda\,\delta v=1\), conductive evaporation suppressed \(3\!-\!30\times\) relative to the classic Weaver
solution.</li>
<li><b>Lancaster+25</b> &mdash; the \((1-\theta)\,L_{\text{mech}}\) reduced model (the input-rescale this report is
careful <i>not</i> to stack on top of the explicit \(L_{\text{cool}}\)).</li>
<li><b>Pittard+22</b> &mdash; wind-blown bubbles radiate up to \(\sim\!98\%\) of the injected energy.</li>
<li><b>Fielding+20</b> &mdash; turbulent radiative mixing layers; surface-area / cooling scaling exponent
\(d\approx0.5\).</li>
<li><b>Tan/Oh/Gronke+21</b> &mdash; the Damk&ouml;hler number \(\mathrm{Da}=\tau_{\text{turb}}/t_{\text{cool}}\) the
(now-refuted) \(\theta_{\text{target}}(\mathrm{Da})\) closure was built on.</li>
</ul>
<figure>__FIG_THETADENS__<figcaption>TRINITY&rsquo;s resolved \(L_{\text{cool}}/L_{\text{mech}}\) vs ambient
\(n_{\text{Core}}\) (real data, rising \(0.250\to0.697\)) against a <b>SCHEMATIC</b> literature band (de-annotated;
the gap is <b>not</b> quantified). The band is an arbitrary saturating stand-in, <i>not</i> digitized
\(\theta(n)\). Read of <code>data/fmix_table.csv</code> via <code>data/make_theta_density_plot.py</code>.
</figcaption></figure>
<div class="box over"><div class="lab">honesty &mdash; PDFs not opened this session</div>Our own session could
<b>not</b> open the El-Badry / Lancaster PDFs (web endpoints returned HTTP&nbsp;403). The El-Badry specifics above
are attributed to the maintainer&rsquo;s note&rsquo;s PDF check (2026-06-25); all bibcodes are verified, but the
exact equation/figure numbers stay <b>&ldquo;verify against the PDFs&rdquo;</b> before quoting in the paper. The
\(\theta\)-vs-\(n\) band in the figure is schematic (the gap is deliberately not quantified). Note finally that a
scalar boost <b>cannot</b> reproduce El-Badry&rsquo;s coupled cooling-up / evaporation-down behaviour &mdash; a
faithful \(\kappa_{\text{eff}}\) is a re-derivation, not a coefficient swap.</div>
"""

SEC_WIRING = r"""
<h2 id="wiring">8 &middot; The wiring &mdash; shipped, opt-in, gated</h2>
<p>An opt-in <code>cooling_boost_mode \(\in\) {none, multiplier, theta_target}</code> was wired into production,
following the note&rsquo;s &sect;Code-level rule: <b>one helper, three sites, default off &rArr; byte-identical</b>.
A single <code>effective_Lloss</code> returns the same \(L_{\text{loss}}^{\text{eff}}\) to the
\(\beta\!-\!\delta\) residual, the \(E_b\) ODE, and the <code>cooling_balance</code> trigger &mdash; the
consistency contract from &sect;2. The modes are <code>none</code> &rarr; \(L_{\text{cool}}+L_{\text{leak}}\)
(byte-identical); <code>multiplier</code> &rarr; \(L_{\text{leak}}+f_{\text{mix}}\,L_{\text{cool}}\);
<code>theta_target</code> &rarr; \(\max(L_{\text{cool}}+L_{\text{leak}},\,\theta\,L_{\text{mech}})\). An
unrecognised token falls back to the resolved loss, so a typo cannot perturb a run.</p>
<div class="box find"><div class="lab">the gate (real <code>simple_cluster</code> runs, separate processes)</div>
The three boost params mirror <code>transition_trigger</code>&rsquo;s snapshot-exclusion flags
(<code>exclude_from_snapshot=True, run_const=True</code>), so they drop out of <code>dictionary.jsonl</code> and a
default run is byte-identical. Confirmed: <b><code>none</code> is byte-identical to HEAD</b> through the
active-cooling region (snapshots 1&ndash;128; resolved cooling activates at snapshot 98, so the test bites past it
and passes). <b><code>multiplier</code> \(f=2\) diverges at snapshot 99</b> &mdash; the <i>first</i> active-cooling
step &mdash; proving the boost is genuinely live, not dead code. <b>20/20 tests pass</b>
(<code>test_cooling_boost.py</code> + <code>test_r1_shadow.py</code>); ruff F-rules clean. This is the wiring the
&sect;9 live runs exercise.</div>
"""

SEC_LIVE = r"""
<h2 id="live">9 &middot; The live matched-\(t\) edge runs &mdash; done</h2>
<p>Every firing-value above is a frozen-trajectory <i>screen</i>. This session ran the promised replacement:
<code>none</code> vs boosted, in <b>separate processes</b>, compared at <b>matched simulation time</b>, on the
edge configs &mdash; a feedback&times;density edge set (the edge configs vary SFE as well as density). Committed:
<code>runs/data/live_compare.csv</code> + per-arm harvest trajectories. <code>fired_cooling_boost</code> = handed
off via a <i>cooling</i> trigger (True) vs blew out / never transitioned (False).</p>
<div class="tablewrap"><table><thead><tr><th>config (boost)</th><th>\(n_{\text{Core}}\), sfe</th>
<th>\(t_{\text{trans}}\) none&rarr;boost</th><th>blowout (boost)</th><th>fired cooling?</th><th>reading</th>
</tr></thead><tbody>
<tr><td>f1edge_hidens (&times;2)</td><td>\(10^6\), 0.01</td><td>0.0314 &rarr; 0.0034</td><td>none (nan)</td>
<td class="win">True</td><td>dense fires cooling <b>at birth</b>, before any blowout (over-boost)</td></tr>
<tr><td>simple_cluster (&times;2)</td><td>\(10^5\), dflt</td><td>10.44 &rarr; 0.131</td><td>0.109</td>
<td class="loss">False</td><td>blows out (0.109) <i>before</i> it transitions (0.131); \(E_b\) shifts up to 47%</td></tr>
<tr><td>f1edge_lowdens (&times;2)</td><td>\(10^2\), 0.50</td><td>no transition (trunc.)</td><td>0.620</td>
<td class="loss">False</td><td>diffuse: doesn&rsquo;t fire by blowout; trims \(E_b\) 13%, blowout +9 kyr</td></tr>
<tr><td>f1edge_lowdens (&times;3)</td><td>\(10^2\), 0.50</td><td>no transition (trunc.)</td><td>0.639</td>
<td class="loss">False</td><td>doesn&rsquo;t fire even at &times;3; trims \(E_b\) 24%, blowout +28 kyr</td></tr>
<tr><td>fail_repro (&times;2)</td><td>heavy/path.</td><td>0.0034 &rarr; 0.0034</td><td>none (nan)</td>
<td class="loss">False</td><td>pathological heavy config; boost has no effect</td></tr>
</tbody></table></div>
<p>The pattern confirms <b>live</b> what the screens argued: <b>no constant fires cooling across density.</b> The
dense cloud over-fires (cooling at birth); the compact \(\text{simple\_cluster}\) blows out
<i>before</i> the boosted cooling crosses the threshold in a sustained way (and the live trajectory shift &mdash;
\(E_b\) up to \(-47\%\), \(v_2\) \(-44\%\), \(R_2\) \(-15\%\) &mdash; proves the frozen screen was insufficient,
as warned); the diffuse cloud does <b>not</b> fire by blowout even at \(\times3\), only trimming \(E_b\) by
\(13\%/24\%\); the heavy cloud is unaffected (cooling never rescues it).</p>
<div class="box over"><div class="lab">honest scope of the diffuse arm</div>All three
\(\text{f1edge\_lowdens}\) arms truncated at a 1200&nbsp;s wall-clock ceiling (<code>exit=124</code>) at
sim-time \(t\approx3.0\!-\!3.3\) Myr, after a blowout at \(\approx0.61\!-\!0.64\) Myr. So the clean statement is
&ldquo;not fired <b>by blowout</b>&rdquo; (a clean <i>No</i> for both \(\times2\) and \(\times3\)), <b>not</b>
&ldquo;never fires.&rdquo; The edge configs also confound density with SFE &mdash; read this as a dense-weak
&harr; diffuse-strong edge, not a one-variable density sweep.</div>
"""

SEC_BOTTOM = r"""
<h2 id="bottom">10 &middot; Bottom line &amp; the pivot</h2>
<p>A cooling-magnitude knob &mdash; constant <b>or</b> \(\mathrm{Da}\)-coupled &mdash; is <b>not</b> the transition
trigger for normal clouds. At blowout the resolved loss ratio is only \(0.25\!-\!0.70\) (well short of \(0.95\)),
the constant target is degenerate with the \(0.95\) trigger TRINITY already has (&sect;3), and the
\(\mathrm{Da}\)-coupled target neither orders by density nor discriminates (&sect;6, gate-validated). The live runs
(&sect;9) confirm no constant fires cooling across density. Convergent, data-backed conclusion &mdash; matching the
methods note&rsquo;s closing nuance:</p>
<ul>
<li><b>Drop \(\theta_{\text{target}}(\mathrm{Da})\) as a trigger mechanism</b> &mdash; refuted by a gate-validated
replay (&sect;6).</li>
<li><b>The trigger for normal clouds is geometric blowout (\(R_2=r_{\text{Cloud}}\))</b> &mdash; which
TRINITY&rsquo;s default already does. The &ldquo;runs never transition&rdquo; symptom is the cooling
<i>magnitude</i>, not the trigger.</li>
<li><b>Use the cooling boost to correct cooling MAGNITUDE</b> &mdash; a constant \(\theta\approx0.9\!-\!0.99\) (from
the literature plateau, via the existing <code>theta_target</code> mode) so \(E_b,P_b,R_2,v_2\), and evaporation
are right <i>through</i> the blowout handoff, <b>not</b> to fire it. The live runs confirm the magnitude correction
is well-behaved (it does not distort the trajectory pathologically).</li>
<li><b>Heavy clouds</b> (super-critical, \(\text{PdV}/L_{\text{mech}}>1\)) &rarr; the <b>PdV / \(E_b\)-peak
handoff</b>; cooling never fires for them.</li>
<li><b>Never double-count</b> &mdash; the boosted loss <i>replaces</i> the explicit \(L_{\text{cool}}\) via the
\(\max(\cdot)\) closure; it never stacks.</li>
</ul>
<div class="box hyp"><div class="lab">the long-term endgame</div>A constant \(\theta\) gets the magnitude
right but cannot couple cooling to evaporation (it can&rsquo;t reproduce El-Badry&rsquo;s \(3\!-\!30\times\)
evaporation suppression). The faithful form is a <b>\(\kappa_{\text{eff}}=\max(\kappa_{\text{Spitzer}},\,
\kappa_{\text{mix}})\)</b> re-derivation, \(\kappa_{\text{mix}}\sim\rho\,c_p\,D_{\text{turb}}\),
\(D_{\text{turb}}\sim\lambda\,\delta v\sim R_2 v_2\) &mdash; a re-derivation, not a coefficient swap. That is the
endgame; the constant-magnitude correction is the right step now. <b>Its first rung is now built and gated
&mdash; see &sect;11.</b></div>
"""

SEC_KAPPA = r"""
<h2 id="kappa">11 &middot; \(\kappa_{\text{eff}}\) Rung A &mdash; the endgame&rsquo;s first rung, built &amp; gated</h2>
<p>The endgame (&sect;10) is a faithful, state-coupled \(\kappa_{\text{eff}}\) <i>inside</i> the structure
solve. Before that multi-day re-derivation, this session built <b>Rung A</b>: a structural <b>probe</b> that
inflates the Spitzer prefactor \(C_{\text{thermal}}\!\to\!f_\kappa\,C_{\text{thermal}}\) at all three sites it
enters <code>bubble_luminosity.py</code> (the \(\dot M\) seed, the backward-ODE initial conditions, and the
temperature-ODE conduction term). Unlike the scalar boost on \(L_{\text{cool}}\), this raises cooling
<i>through</i> the structure, so the loss fraction \(\theta\) emerges as an <b>output</b>. It is gated by a new
<code>cooling_boost_kappa</code> param (default \(f_\kappa\!=\!1.0\)): <b>byte-identical when off</b> (the
\(f_\kappa\!=\!1\) run reproduces the <code>f1edge_hidens</code> <code>dictionary.jsonl</code> bit-for-bit), full
<code>pytest</code> 595 green, ruff F-rules clean. <b>Production is unchanged by default.</b></p>
<figure>__FIG_KAPPA__<figcaption>Rung-A back-reaction on the stiff dense edge <code>f1edge_hidens</code>,
\(f_\kappa\!=\!2\) vs the \(f_\kappa\!=\!1\) baseline, compared at <b>matched simulation time</b>. <b>Left
(absolute):</b> the cooling luminosity itself &mdash; both runs rise as the bubble develops, and the
\(f_\kappa\!=\!2\) (red) curve sits <b>above</b> the baseline (blue) at every \(t\): more conduction = more
cooling. <b>Middle (ratios):</b> the same comparison as \(f_\kappa\!=\!2\div f_\kappa\!=\!1\) &mdash; a value
<i>above 1.0 means the knob raised it</i>; the downward slope is the boost <i>shrinking</i> over time (from
\(\sim\!1.5\times\) to \(\sim\!1.23\times\)), not cooling falling. Cooling is raised \(1.2\!-\!1.5\times\)
&mdash; the intended effect &mdash; <b>but the evaporative mass flux \(\dot M\) rides along</b>
(\(1.08\!-\!1.17\times\)), exactly the El-Badry coupling a faithful \(\kappa_{\text{eff}}\) must instead
<i>suppress</i>; \(E_b\) is drained to \(0.90\!-\!0.96\times\). <b>Right:</b> even a \(2\times\) \(\kappa\)
moves the loss-ratio proxy only \(+0.05\!-\!0.10\), staying far below the \(0.95\) trigger. From
<code>data/make_kappa_backreaction.py</code> &rarr; <code>data/kappa_backreaction.csv</code>.</figcaption></figure>
<p><b>What Rung A settles.</b> (i) The plumbing takes a \(\kappa\) knob cleanly &mdash; cooling genuinely rises
through the structure, vindicating the structural approach over a scalar \(L_{\text{cool}}\) rescale. (ii) The
crux is real and <b>quantified</b>: a flat \(f_\kappa\) raises \(\dot M\) too (\(\approx\) half the fractional
rise of \(L_{\text{cool}}\)) &mdash; wrong sign vs El-Badry. (iii) Headroom is small and the back-reaction grows
with \(f_\kappa\), so <b>brute-forcing \(f_\kappa\) toward the trigger is non-viable</b>. Net: Rung A
<b>confirms Rung B is required, not optional</b> &mdash; only a state-coupled \(\kappa_{\text{eff}}\) that
decouples cooling-up from evaporation-down reaches the transition. Full scope &amp; the measured table:
<code>KAPPA_EFF_SCOPING.md</code> &sect;6a. <b>Rung B is now scoped on paper</b> (two independent verifications)
in <code>RUNGB_SCOPING.md</code>: the conductive flux at the front is <i>one quantity read twice</i>, so the
faithful \(\kappa_{\text{eff}}\) must <b>sever \(\dot M\) from the front balance</b> (entrainment-set,
\(>0\) by construction) &mdash; not swap \(\kappa\) &mdash; with a <b>numerical</b> mix-branch near-front IC
and an entrainment efficiency \(\alpha_{\text{mix}}\!\ll\!1\) as the real model. No production code touched.</p>
"""

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/pdv-trigger/</code> &mdash; reproducible
without re-running the (hours-long) sims. Each figure is a pure read of a committed CSV; the prose numbers trace to
the same files.</p>
<div class="tablewrap"><table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>data/fmix_table.csv</code> (+ <code>make_fmix_table.py</code>)</td><td>the note&rsquo;s Table 2, both trigger conventions; the &sect;3 headline \(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)</td></tr>
<tr><td><code>fmix_vs_density.png</code> (+ <code>data/make_fmix_spread_plot.py</code>)</td><td>&sect;3 data-only scatter of the firing \(f_{\text{mix}}\) (\(1.36\to3.81\)); states the constant-\(\theta\) degeneracy</td></tr>
<tr><td><code>theta_vs_density.png</code> (+ <code>data/make_theta_density_plot.py</code>)</td><td>&sect;7 TRINITY \(L_{\text{cool}}/L_{\text{mech}}\) vs \(n_{\text{Core}}\) with a SCHEMATIC, de-annotated literature band</td></tr>
<tr><td><code>data/doublecount_mc.csv</code> (+ <code>make_doublecount_mc.py</code>)</td><td>the \(5\times10^5\)-draw Monte-Carlo backing the &sect;2 single-count claim (0 draws in the \(2\theta\) region)</td></tr>
<tr><td><code>data/pdv_regime_budget.csv</code> (+ <code>make_pdv_regime_table.py</code>)</td><td>per-config \(\text{PdV}/L_{\text{mech}}\), \(E_b\) growth, sub-/super-critical regime (&sect;4)</td></tr>
<tr><td><code>data/closure_test.csv</code> (+ <code>make_closure_test.py</code>)</td><td>the 8-config staged frozen-trajectory screen behind the &sect;5 heatmap</td></tr>
<tr><td><code>da_screen.png</code> (+ <code>data/make_da_screen.py</code>, <code>data/da_screen.csv</code>)</td><td>&sect;6 Step A &mdash; offline Da-shape proxy at blowout (NO-GO: non-monotonic, fires dense at birth)</td></tr>
<tr><td><code>da_replay.png</code> (+ <code>data/make_da_replay.py</code>, <code>data/da_replay.csv</code>)</td><td>&sect;6 Step A&prime; &mdash; gate-validated real-Da replay (gate PASS; verdict NO-GO, \(\theta_{\text{target}}(\mathrm{Da})\) refuted)</td></tr>
<tr><td><code>runs/data/live_compare.csv</code> (+ per-arm <code>runs/data/harvest_*.csv</code>)</td><td>&sect;9 live matched-\(t\) edge runs (4 configs, separate processes): no constant fires cooling across density</td></tr>
<tr><td><code>data/kappa_backreaction.csv</code> (+ <code>make_kappa_backreaction.py</code>, <code>kappa_backreaction.png</code>)</td><td>&sect;11 \(\kappa_{\text{eff}}\) Rung-A back-reaction (\(f_\kappa{=}2\) vs \(1\), matched \(t\)): \(L_{\text{cool}}\!\uparrow\) but \(\dot M\!\uparrow\) rides along; the probe param is <code>runs/params/f1edge_hidens__kappa2.param</code></td></tr>
<tr><td><code>ideas_comparison.png</code> (+ <code>data/make_ideas_comparison.py</code>)</td><td>the at-a-glance scoreboard of all ideas + three real-data evidence panels (reads <code>fmix_table</code>, <code>da_replay</code>, <code>kappa_backreaction</code> CSVs)</td></tr>
<tr><td><code>KAPPA_EFF_SCOPING.md</code></td><td>&sect;11 endgame feasibility map + the &sect;6a Rung-A result table; the two-rung ladder and the \(\dot M>0\) crux</td></tr>
<tr><td><code>storyline_figs/*.png</code> (+ <code>make_storyline_figs.py</code>)</td><td>the four storyline figures (&sect;2 double-count, &sect;3 convention, &sect;4 regime, &sect;5 heatmap), each a pure read of the CSVs</td></tr>
<tr><td><code>PLAN.md</code> / <code>FINDINGS.md</code></td><td>the living plan (&ldquo;Outcome &amp; pivot&rdquo;) and the verified findings &mdash; the source of truth this report is a sibling of</td></tr>
</tbody></table></div>
<p class="small muted">Rebuild this report:
<code>python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py</code> (figures are committed PNGs, not
regenerated here). Then rebuild the storyline book:
<code>python docs/dev/html-insights/build_storylines.py</code>.</p>
<footer>TRINITY PdV-in-the-trigger &rarr; unresolved-interface-cooling closure &middot; boost the loss, not the
trigger &mdash; and the trigger is blowout &middot; every number traces to a committed CSV in
<code>docs/dev/transition/pdv-trigger/{data,runs/data}/</code> &middot; light-mode, MathJax-rendered.</footer>
"""


def main():
    parts = [
        HEAD, HERO,
        SEC_SETUP, SEC_DOUBLE, SEC_CONVENTION, SEC_REGIME, SEC_CLOSURE,
        SEC_DA, SEC_LIT, SEC_WIRING, SEC_LIVE, SEC_BOTTOM, SEC_KAPPA, SEC_REPRO,
    ]
    parts.append("</div></body></html>")
    html = "".join(parts)

    for token, (fname, alt) in FIGURES.items():
        html = html.replace(token, img(fname, alt))
    assert "__FIG_" not in html, "unreplaced figure placeholder remains"

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    # This report is the core chapter of the s5 storyline book; rebuild the
    # collection with: python docs/dev/html-insights/build_storylines.py


if __name__ == "__main__":
    main()
