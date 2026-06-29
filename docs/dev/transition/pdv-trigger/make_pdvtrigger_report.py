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
        "scoreboard of every transition-fix idea (constant f_mix, constant theta, theta_target(Da), live multiplier, kappa_eff Rung A = the cooling mechanism (this work), kappa_eff Rung B = optional fidelity bonus) with verdict badges, plus three real-data evidence panels",
    ),
    "__FIG_KAPPA__": (
        "kappa_backreaction.png",
        "kappa_eff Rung A back-reaction on f1edge_hidens at matched t: f_kappa=2 raises L_cool x1.23-1.38 but dMdt x1.08-1.17 rides along, and a 2x kappa moves the loss ratio only +0.05-0.10 toward the 0.95 trigger",
    ),
    "__FIG_EBPEAK__": (
        "ebpeak_trigger_test.png",
        "does PdV alone trigger the transition? PdV-inclusive ratio (Lloss+PdV)/Lgain vs t for compact, diffuse and mid: it peaks BELOW the 1.0 ebpeak threshold at f_kappa=1 (compact 0.91, diffuse 0.86, mid 0.90) then declines, so ebpeak never fires at f_kappa=1; under boost compact and mid climb and fire by f_kappa~4 while diffuse stays flat ~0.85",
    ),
    "__FIG_EB8__": (
        "ebpeak_8config_xcheck.png",
        "does the ebpeak finding hold across the 8 configs? frozen-trajectory screen peak PdV-inclusive ratio per config: 6 normal clouds peak 0.85-0.92 and never fire, only heavy-5e9 and the control fire; live full-run peaks (black diamonds) for simple_cluster and midrange_pl0 match the frozen bars to the digit (0.91, 0.90)",
    ),
    "__FIG_FKDEF__": (
        "fkappa_definition.png",
        "what is f_kappa? left: the Spitzer conductivity kappa_eff(T)=f_kappa*C_thermal*T^(5/2) that f_kappa multiplies, for f_kappa=1,2,4; right: the analytic seed scaling dMdt~f_kappa^(2/7) verified against the measured f_kappa=2 back-reaction (measured 1.2175 vs analytic 1.219 at the seed, <0.1%)",
    ),
    "__FIG_FKCAL__": (
        "kappa_blowout_calibration.png",
        "f_kappa calibration on full runs: developed theta=Lcool/Lmech at cloud dispersal vs f_kappa for compact/mid/diffuse, with the 0.95 cooling_balance trigger; compact crosses at f_kappa~4 (red ring=cooling fired), mid ~5, diffuse far higher (~60); right panel is the cumulative radiated fraction",
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
<h1>Boost the cooling <i>magnitude</i>, not the trigger &mdash; the default trigger <i>is</i> cooling-driven</h1>
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
everywhere, so \(\theta_{\max}\mathrm{Da}/(1+\mathrm{Da})\) saturates back to a constant). <b>The pivot:</b> the cooling
boost corrects cooling <b>magnitude</b>, it does not change the trigger. <span class="small muted">[framing
corrected 06-26: TRINITY&rsquo;s <i>default</i> trigger is the cooling-driven <code>cooling_balance</code>
(\(L_{\text{loss}}/L_{\text{gain}}\!>\!0.95\)); geometric blowout is <b>opt-in, default OFF</b> and is only the
<i>fallback symptom</i> when 1D cooling is too weak for <code>cooling_balance</code> to fire &mdash; so the job
of \(\kappa_{\text{eff}}\) is precisely to make that cooling-driven trigger fire (&sect;10&ndash;12).]</span> Live matched-\(t\) runs
confirm: no constant fires cooling across density. The heavy 5e9 cloud is super-critical and hands off via the
PdV / \(E_b\)-peak turnover. <b>Latest &mdash; the merge (2026-06-26):</b> the cooling-magnitude fix has a
mechanism that is <b>already built</b> &mdash; \(\kappa_{\text{eff}}\) (<code>cooling_boost_kappa</code>, Rung A)
raises the <i>emergent</i> cooling in-structure (&sect;11). The remaining work is <b>calibrating
\(f_\kappa(\text{properties})\)</b> to the target \(\theta(n_H)\) (El-Badry \(\lambda\delta v{=}\kappa_{\text{eff}}\)
+ Lancaster). The evaporation-decoupling re-derivation is an <b>optional fidelity bonus</b>, not the goal.</p>
</div>

<figure>__FIG_IDEAS__<figcaption><b>The whole storyline at a glance.</b> Every transition-fix idea tried,
left&rarr;right, with its verdict: the three scalar knobs (constant \(f_{\text{mix}}\), constant \(\theta\),
\(\theta_{\text{target}}(\mathrm{Da})\)) are <span style="color:#b3392f"><b>refuted</b></span>, the live
multiplier is <span style="color:#b3801f"><b>partial</b></span> (magnitude-only, mistimes by density),
\(\kappa_{\text{eff}}\) Rung A is the <span style="color:#2a8aa8"><b>cooling mechanism</b></span> (this work,
gated/byte-identical-off), and the faithful Rung-B evaporation-decoupling is an <span style="color:#3a8a3f">
<b>optional bonus</b></span>. The three lower panels are the real-data evidence for the key verdicts
(\(f_{\text{mix}}\) spread, \(\mathrm{Da}\) saturation, and the Rung-A cooling enhancement). Built by
<code>data/make_ideas_comparison.py</code> from the committed CSVs.</figcaption></figure>
"""

SEC_SETUP = r"""
<h2 id="setup">1 &middot; The setup &mdash; where s1 left off</h2>
<p>TRINITY integrates the bubble interior energy with
\[ \frac{dE_b}{dt} = L_{\text{mech}} - L_{\text{cool}} - \underbrace{4\pi R_2^2\,v_2\,P_b}_{\text{PdV}} - L_{\text{leak}} \]
(<code>get_betadelta.py:475</code>), so the PdV work term is <b>already</b> in the energy evolution. The default
energy&rarr;momentum handoff, though, watches only the <b>radiative</b> ratio
\[ \frac{L_{\text{mech}}-L_{\text{cool}}}{L_{\text{mech}}} < 0.05 \qquad\text{(cooling }\ge 95\%\text{)} \]
(<code>run_energy_implicit_phase.py:1206</code>) &mdash; <b>no PdV</b>. The clean-room investigation (s1) showed
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
<li><b>The default trigger is the cooling-driven <code>cooling_balance</code> (\(L_{\text{loss}}/L_{\text{gain}}\!>\!0.95\))</b>
&mdash; a <i>cooling</i>-driven transition, the same intent as the literature; geometric blowout
(\(R_2=r_{\text{Cloud}}\)) is <b>opt-in (default OFF)</b> and acts only as the <i>fallback symptom</i> when 1D
cooling is too weak for <code>cooling_balance</code> to fire. So the &ldquo;runs never transition&rdquo;
problem is the cooling <i>magnitude</i>, not the trigger family: the job of \(\kappa_{\text{eff}}\) is to make
the cooling-driven trigger fire (&sect;11). <span class="small muted">[framing corrected 06-26: see
<code>PLAN.md</code> ledger; <code>run_energy_implicit_phase.py:1206</code>, <code>default.param:282</code>.]</span></li>
<li><b>Correct cooling MAGNITUDE with \(\kappa_{\text{eff}}\), calibrated to a density-dependent target</b>
&mdash; <b>the merge:</b> \(\kappa_{\text{eff}}\) (<code>cooling_boost_kappa</code>, Rung A, <b>built</b>) is the
in-structure <i>mechanism</i> that raises the emergent cooling; the calibration <i>target</i> is \(\theta(n_H)\)
(El-Badry \(\lambda\delta v{=}\kappa_{\text{eff}}\) + Lancaster), the knob is \(f_\kappa(\text{properties})\). A
<i>constant</i> \(\theta\) via <code>theta_target</code> is the degenerate special case (\(\approx0.95\) = the
trigger); the real upgrade is the <b>density-dependent \(f_\kappa\) calibration</b>. So \(\theta,E_b,P_b,R_2,v_2\)
come out right <i>through</i> the blowout handoff because the cooling fraction emerges per cloud.</li>
<li><b>Heavy clouds</b> (super-critical, \(\text{PdV}/L_{\text{mech}}>1\)) &rarr; the <b>PdV / \(E_b\)-peak
handoff</b>; cooling never fires for them.</li>
<li><b>Never double-count</b> &mdash; the boosted loss <i>replaces</i> the explicit \(L_{\text{cool}}\) via the
\(\max(\cdot)\) closure; it never stacks.</li>
</ul>
<div class="box hyp"><div class="lab">the optional high-fidelity bonus</div>\(\kappa_{\text{eff}}\) (Rung A)
already delivers the <i>cooling</i> enhancement the goal needs. A separate, <b>optional</b> step would also make
evaporation <i>fall</i> while cooling rises (El-Badry&rsquo;s \(3\!-\!30\times\) suppression) &mdash; a faithful
\(\kappa_{\text{eff}}=\max(\kappa_{\text{Spitzer}},\,\kappa_{\text{mix}})\) re-derivation with
\(\kappa_{\text{mix}}\sim\rho\,c_p\,\lambda\delta v\). But two offline prototypes (FM1/FM1b, &sect;11) show the
1D \(\dot M\) is <b>front-anchored and resists it</b>, and the suppression is <b>not in the goal</b>. So it is a
fidelity bonus, not a blocker. <b>The main line is \(f_\kappa(\text{properties})\) calibration &mdash; see
&sect;11.</b></div>
"""

SEC_KAPPA = r"""
<h2 id="kappa">11 &middot; \(\kappa_{\text{eff}}\) Rung A &mdash; the cooling MECHANISM, built &amp; gated</h2>
<div class="box" style="border-left:4px solid #2a8aa8"><b>The merge (2026-06-26).</b> The goal is
<b>enhanced, density-dependent cooling matched to obs/3D</b> &mdash; and \(\kappa_{\text{eff}}\) is the
<b>mechanism that delivers it</b>, already built. The remaining work is <b>calibrating
\(f_\kappa(\text{properties})\)</b> so the emergent \(\theta\) tracks the target \(\theta(n_H)\) (El-Badry
\(\lambda\delta v{=}\kappa_{\text{eff}}\) + Lancaster). The faithful evaporation-<i>decoupling</i>
(&ldquo;Rung B&rdquo;) is an <b>optional high-fidelity bonus</b> &mdash; the 1D \(\dot M\) is front-anchored and
resists it (FM1/FM1b, <code>RUNGB_SCOPING.md</code>); that suppression is not in the goal.</div>
<p>This session built <b>Rung A</b> = a new gated param <code>cooling_boost_kappa</code> that inflates the
Spitzer prefactor \(C_{\text{thermal}}\!\to\!f_\kappa\,C_{\text{thermal}}\) at all three sites it enters
<code>bubble_luminosity.py</code> (the \(\dot M\) seed, the backward-ODE initial conditions, and the
temperature-ODE conduction term). Unlike a scalar boost on \(L_{\text{cool}}\), this raises cooling
<i>through</i> the structure, so the loss fraction \(\theta\) <b>emerges as an output</b> (El-Badry&rsquo;s own
approach). Default \(f_\kappa\!=\!1.0\) is <b>byte-identical</b> (the \(f_\kappa\!=\!1\) run reproduces the
<code>f1edge_hidens</code> <code>dictionary.jsonl</code> bit-for-bit), full <code>pytest</code> 595 green, ruff
clean. <b>Production is unchanged by default.</b></p>
<figure>__FIG_KAPPA__<figcaption>Rung-A back-reaction on the stiff dense edge <code>f1edge_hidens</code>,
\(f_\kappa\!=\!2\) vs the \(f_\kappa\!=\!1\) baseline, compared at <b>matched simulation time</b>. <b>Left
(absolute):</b> the cooling luminosity itself &mdash; both runs rise as the bubble develops, and the
\(f_\kappa\!=\!2\) (red) curve sits <b>above</b> the baseline (blue) at every \(t\): more conduction = more
cooling. <b>Middle (ratios):</b> the same comparison as \(f_\kappa\!=\!2\div f_\kappa\!=\!1\) &mdash; a value
<i>above 1.0 means the knob raised it</i>; the downward slope is the boost <i>shrinking</i> over time (from
\(\sim\!1.5\times\) to \(\sim\!1.23\times\)), not cooling falling. Cooling is raised \(1.2\!-\!1.5\times\)
&mdash; the intended effect (the goal) &mdash; while the evaporative mass flux \(\dot M\) rises only mildly
(\(1.08\!-\!1.17\times\), a <b>tolerated side effect</b>; an optional fidelity bonus would suppress it instead);
\(E_b\) is drained to \(0.90\!-\!0.96\times\). <b>Right:</b> at \(f_\kappa\!=\!2\) the loss-ratio proxy moves
\(+0.05\!-\!0.10\) &mdash; so reaching the obs/3D target needs a larger, <b>calibrated</b> \(f_\kappa\). From
<code>data/make_kappa_backreaction.py</code> &rarr; <code>data/kappa_backreaction.csv</code>.</figcaption></figure>
<p><b>What Rung A settles.</b> (i) \(\kappa_{\text{eff}}\) genuinely <b>raises the emergent cooling</b> through
the structure (\(\times1.23\!-\!1.38\)) &mdash; it <b>is the mechanism</b> the goal needs, vindicated over a
scalar \(L_{\text{cool}}\) rescale. (ii) It also nudges \(\dot M\) up (\(\approx\) half the fractional rise of
\(L_{\text{cool}}\)) &mdash; a <b>tolerated side effect</b> (\(\dot M\) stays positive/viable), <i>not</i> a
problem for the goal. (iii) <b>Remaining work = calibrate \(f_\kappa(\text{properties})\)</b> so the emergent
\(\theta\) tracks the target \(\theta(n_H)\) &mdash; reusing this knob, no new production code. The faithful
evaporation-<i>decoupling</i> (make \(\dot M\) <i>fall</i> while cooling rises) is an <b>optional bonus</b>:
two offline prototypes (<code>RUNGB_SCOPING.md</code> &mdash; <b>FM1</b>: imposing \(\dot M\) refuted;
<b>FM1b</b>: an interior loss term gives the El-Badry sign but negligible magnitude) show the 1D front-anchored
\(\dot M\) resists it. Full table: <code>KAPPA_EFF_SCOPING.md</code> &sect;6a. No production code touched.</p>
"""

SEC_EBPEAK = r"""
<h2 id="ebpeak">12 &middot; Does PdV <i>alone</i> trigger the transition? &mdash; the founding question, measured</h2>
<div class="box" style="border-left:4px solid #2a8aa8"><b>The question (2026-06-28).</b> The workstream is
named &ldquo;PdV-in-the-trigger.&rdquo; TRINITY&rsquo;s default <code>cooling_balance</code> trigger watches the
<i>radiative</i> ratio \(L_{\text{loss}}/L_{\text{gain}}\!\ge\!0.95\); the opt-in <code>ebpeak</code> trigger
watches the <b>PdV-inclusive</b> balance \(\dot E_b = L_{\text{gain}}-L_{\text{loss}}-\text{PdV}\le 0\), i.e.
\((L_{\text{loss}}+\text{PdV})/L_{\text{gain}}\!\ge\!1\) &mdash; the bubble&rsquo;s net energy stops growing, so
it rolls into momentum <i>naturally</i>. Since PdV is the dominant sink, does adding it tip the transition
<b>without</b> a cooling boost?</p></div>
<p>Tested on the actual code path: two runs with <code>transition_trigger=cooling_balance,ebpeak</code>
<b>active</b> at \(f_\kappa\!=\!1\) (<code>runs/params/cal_&#123;compact,diffuse&#125;__ebpeak.param</code>). Both
ran to <code>stop_t</code> and ended on <code>STOPPING_TIME</code> with shadow <code>ebpeak_t=None</code> &mdash;
<b>ebpeak never fired.</b> The PdV-inclusive ratio <b>peaks below the 1.0 threshold, then declines:</b> compact
peaks \(0.91\) at \(t\!\approx\!0.12\) (just past dispersal); diffuse peaks \(0.86\) at \(t\!\approx\!1.06\),
then falls as the bubble <b>re-accelerates in the low-density ISM</b> (the diffuse run reached \(t\!=\!1.5\),
\(R_2\!=\!191\) pc, \(v_2\!=\!168\) km/s, \(E_b\) still <i>growing</i> &mdash; the net energy never turns over).</p>
<figure>__FIG_EBPEAK__<figcaption>The PdV-inclusive ratio \((L_{\text{loss}}+\text{PdV})/L_{\text{gain}}\) (solid
blue, \(f_\kappa{=}1\), run to <code>stop_t</code>) sits far above radiative-only (dotted) &mdash; PdV is the
dominant sink &mdash; but <b>peaks below the green \(1.0\) <code>ebpeak</code> line and declines</b>, for
<b>both</b> configs. The \(f_\kappa{=}2,4\) curves expose the <b>cooling\(\leftrightarrow\)PdV trade-off</b>:
boosting cooling drains \(E_b\!\to\!\) lowers \(P_b\!\to\!\) lowers PdV, so for <b>diffuse</b> the PdV-inclusive
peak is nearly \(f_\kappa\)-<i>insensitive</i> (\(0.848\!\to\!0.849\!\to\!0.853\), flat) while the radiative
ratio nearly doubles (\(0.165\!\to\!0.297\)). From <code>data/make_ebpeak_trigger_test.py</code> &rarr;
<code>data/ebpeak_trigger_test.csv</code>.</figcaption></figure>
<p><b>What this settles.</b> (i) Including PdV (<code>ebpeak</code>) is a genuine <b>assist</b> &mdash; it lifts
the transition balance from radiative-only (\(0.66\) compact / \(0.17\) diffuse) up to \(\sim\!0.86\!-\!0.91\),
much closer to firing &mdash; but it <b>does not close the gap</b>; a cooling boost is still required. (ii) For
<b>diffuse</b> the trade-off makes the PdV path a near dead-end: you cannot push the PdV-inclusive peak to
\(1.0\) by boosting \(f_\kappa\); the only route to fire is the <i>radiative</i> <code>cooling_balance</code>
(\(f_\kappa\!\sim\!60\)). PdV helps the <b>compact</b> case (fires by \(f_\kappa\!\sim\!2\!-\!4\), where
<code>cooling_balance</code> also fires &mdash; <code>ebpeak</code> a hair earlier). (iii) So the
<b>complementary split stands but is downgraded:</b> PdV addresses transition <i>timing</i>,
\(\kappa_{\text{eff}}\) the cooling <i>magnitude</i> &mdash; PdV is an assist, <b>not a substitute</b> for the
calibrated boost. This <b>corrects</b> an earlier (06-26) optimistic extrapolation that diffuse would fire
\(\sim\!1.2\!-\!1.3\) Myr; the measured ratio is non-monotone and turns over below \(1.0\). Opt-in dev runs;
<b>no production code touched.</b></p>
<p><b>Does it hold across the 8 configs?</b> The four live full runs are density edges; the \(f_\kappa=1\)
conclusion <b>generalizes to the full 8-config universe</b> via the earlier frozen-trajectory screen. All
<b>6 normal</b> clouds peak at PdV-inclusive \(0.85\!-\!0.92\) and never fire; only the heavy \(5\times10^9\)
(super-critical, \(\mathrm{PdV}/L_{\rm mech}>1\)) and the \(10^6\,M_\odot\) control (a birth blip) fire. Two
of the live runs (<code>simple_cluster</code>, <code>midrange_pl0</code>) sit inside the frozen set and the
live peaks match the frozen bars <b>to the digit</b> (\(0.91\), \(0.90\)) &mdash; validating the screen for
the rest.</p>
<figure>__FIG_EB8__<figcaption>Peak PdV-inclusive ratio per config from the frozen screen (bars; blue = never
fires, green = fires), with the live full-run peaks overlaid as black diamonds for the two configs in both
sets. The six normal clouds cluster at \(0.85\!-\!0.92\), below the green \(1.0\) <code>ebpeak</code> line;
the live diamonds land on the bars. From <code>data/make_ebpeak_8config_xcheck.py</code>.</figcaption></figure>
"""

SEC_FKDEF = r"""
<h2 id="fkdef">13 &middot; What is \(f_\kappa\), precisely? &mdash; the conduction multiplier, with equations</h2>
<div class="box" style="border-left:4px solid #2a8aa8"><b>One line.</b> \(f_\kappa\) (the param
<code>cooling_boost_kappa</code>) is a dimensionless multiplier on TRINITY&rsquo;s <b>thermal-conduction
coefficient</b>. It does <i>not</i> multiply the cooling luminosity; it thickens the conduction layer so
more gas radiates, and the loss fraction \(\theta\) comes out as a result. Every equation below is read
straight from <code>bubble_luminosity.py</code> &mdash; line numbers given, no assumptions.</div>

<h3>The conductivity it scales</h3>
<p>The bubble interior is a Weaver conduction structure. TRINITY carries the classical
Spitzer&ndash;H&auml;rm thermal conductivity</p>
\[ \kappa_{\rm SP}(T) \;=\; C_{\rm th}\,T^{5/2}, \qquad
   C_{\rm th} = 6\times10^{-7}\ \mathrm{erg\,s^{-1}\,cm^{-1}\,K^{-7/2}} \quad(\text{the standard Spitzer value, }\ln\Lambda\!\sim\!30), \]
<p>(<code>C_thermal</code>, <code>registry.py:341</code>), with conductive heat flux
\(q = -\kappa_{\rm SP}\,\mathrm{d}T/\mathrm{d}r\). The knob is simply</p>
\[ \boxed{\;\kappa_{\rm eff}(T) \;=\; f_\kappa\,C_{\rm th}\,T^{5/2} \;=\; f_\kappa\,\kappa_{\rm SP}(T)\;},
   \qquad f_\kappa \equiv \texttt{cooling\_boost\_kappa}\ (\text{default }1,\ \ge 1;\ \texttt{registry.py:351}). \]
<p>Physically \(f_\kappa>1\) is <b>thermal transport above classical Spitzer</b> &mdash; the unresolved
sub-grid mixing / magnetically-modified conduction a 1D Spitzer model cannot see. It enters at exactly the
<b>three</b> places \(C_{\rm th}\) appears in <code>bubble_luminosity.py</code>:</p>

<p><b>(1) The evaporative mass-flux seed</b> (Weaver&#8197;+77 Eq.&nbsp;33; <code>bubble_luminosity.py:291&ndash;295</code>):</p>
\[ \dot M_{\rm init} = \frac{12}{75}\,(1.646)^{5/2}\,\frac{4\pi R_2^{3}}{t}\,\frac{\mu}{k_B}
   \left(\frac{t\,f_\kappa C_{\rm th}}{R_2^{2}}\right)^{\!2/7} P_b^{5/7}
   \;\;\Longrightarrow\;\; \boxed{\;\dot M \propto f_\kappa^{\,2/7}\;}. \]

<p><b>(2) The conduction-layer initial conditions</b> (Eq.&nbsp;44; <code>:370&ndash;381</code>): with
\(A \equiv \tfrac{25}{4}\,k_B/(\mu\,f_\kappa C_{\rm th}) \propto f_\kappa^{-1}\),</p>
\[ \Delta R_2 = \frac{T_{\rm init}^{5/2}}{A\,\dot M/(4\pi R_2^{2})} \;\propto\; f_\kappa,
   \qquad T=\Big(A\,\dot M\,\Delta R_2/4\pi R_2^2\Big)^{2/5}, \qquad \frac{\mathrm{d}T}{\mathrm{d}r}=-\frac{2}{5}\frac{T}{\Delta R_2}. \]
<p>i.e. the conduction/evaporation layer <b>thickens</b> with \(f_\kappa\). <span class="small muted">[Precisely:
\(\Delta R_2 \propto f_\kappa\) <i>at fixed</i> \(\dot M\) (this function takes \(\dot M\) as an argument);
folding in the seed \(\dot M \propto f_\kappa^{2/7}\) gives \(\Delta R_2 \propto f_\kappa^{5/7}\).]</span></p>

<p><b>(3) The temperature-curvature ODE</b> (Eq.&nbsp;42&ndash;43; <code>:406&ndash;409</code>):</p>
\[ \frac{\mathrm{d}^2T}{\mathrm{d}r^2}
   = \frac{P_b}{f_\kappa C_{\rm th}\,T^{5/2}}
     \!\left[\frac{\beta+\tfrac52\delta}{t} + \tfrac52\,(v-v_a)\frac{1}{T}\frac{\mathrm{d}T}{\mathrm{d}r}
            - \frac{1}{P_b}\frac{\mathrm{d}u}{\mathrm{d}t}\right]
     - \tfrac52\frac{1}{T}\!\left(\frac{\mathrm{d}T}{\mathrm{d}r}\right)^{2} - \frac{2}{r}\frac{\mathrm{d}T}{\mathrm{d}r}. \]
<p class="small muted">(\(v_a = \alpha\,r/t\) is the self-similar advection velocity, \(\alpha=\) <code>cool_alpha</code>;
\(\beta=\) <code>cool_beta</code>, \(\delta=\) <code>cool_delta</code> are the Weaver similarity exponents;
\(\mathrm{d}u/\mathrm{d}t\) is the local net cooling. The conduction term &mdash; the only one carrying
\(C_{\rm th}\) &mdash; scales as \(1/(f_\kappa C_{\rm th})\).)</p>

<h3>Why this is <i>not</i> a cooling multiplier (\(\theta\) emerges)</h3>
<p>The loss term is never multiplied by \(f_\kappa\). The local volumetric cooling
\(\mathrm{d}u/\mathrm{d}t = \texttt{net\_coolingcurve.get\_dudt}(t,n,T,\phi)\) is evaluated at the
<i>local</i> density \(n = P_b/[(\mu/\mu_{\rm ion})\,k_B T]\) and temperature inside the structure, and the
bubble cooling luminosity is the integral over that structure,</p>
\[ L_{\rm cool} = \int \frac{\mathrm{d}u}{\mathrm{d}t}\,\mathrm{d}V,
   \qquad \theta \equiv \frac{L_{\rm cool}}{L_{\rm mech}}. \]
<p>\(f_\kappa\) acts <b>only through the structure</b>: a larger \(\kappa_{\rm eff}\) makes the layer thicker
(\(\Delta R_2\propto f_\kappa\)), putting more gas in the \(10^5\!-\!10^6\,\)K band where the cooling
function \(\Lambda(T)\) peaks, so \(L_{\rm cool}\) (hence \(\theta\)) <b>emerges</b> higher &mdash; an
<i>output</i>, the way El-Badry obtains \(\theta\), not a post-hoc floor on \(L_{\rm cool}\).</p>
<figure>__FIG_FKDEF__<figcaption><b>Left:</b> what \(f_\kappa\) multiplies &mdash; the Spitzer conductivity
\(\kappa_{\rm eff}(T)=f_\kappa C_{\rm th}T^{5/2}\) for \(f_\kappa=1,2,4\). <b>Right:</b> the analytic seed
scaling \(\dot M\propto f_\kappa^{2/7}\) <b>verified against measurement</b> &mdash; the measured matched-\(t\)
ratio \(\dot M(f_\kappa{=}2)/\dot M(f_\kappa{=}1)\) equals \(1.2175\) at the seed vs analytic \(2^{2/7}=1.219\)
(\(\approx\!0.1\%\)); it softens as the run develops because \(P_b\) drains \(\sim3\%\) (the genuine
back-reaction). From <code>data/make_fkappa_definition.py</code> (Panel B reads
<code>data/kappa_backreaction.csv</code>).</figcaption></figure>

<h3>The side effect &mdash; why \(f_\kappa\) is a <i>probe</i>, not the final model</h3>
<p>Eq.&nbsp;(1) shows the same knob <b>raises the evaporative mass flux</b>, \(\dot M\propto f_\kappa^{2/7}\).
A faithful \(\kappa_{\rm eff}\) (El-Badry) would instead <b>suppress</b> evaporation while cooling rises; here
both rise together. So \(f_\kappa\) is a <b>structural probe</b> of the cooling-magnitude axis (the
<code>registry.py:351</code> note says exactly this), and the optional &ldquo;Rung B&rdquo; evaporation-
decoupling (&sect;11) is what would make \(\dot M\) fall instead.</p>

<h3>What \(f_\kappa\) buys &mdash; the calibration (does it answer the user&rsquo;s question?)</h3>
<p><b>Yes:</b> at \(f_\kappa=1\) some clouds never fire and sit well below \(\theta\sim0.9\), and they need a
<b>much larger</b> \(f_\kappa\). At \(f_\kappa=1\) the developed \(\theta\) (at cloud dispersal) is only</p>
\[ \theta(f_\kappa{=}1) \approx 0.17\ (\text{diffuse}),\quad 0.61\ (\text{mid}),\quad 0.67\ (\text{compact}), \]
<p>all below both the obs/3D \(\theta\!\sim\!0.9\) and the \(0.95\) <code>cooling_balance</code> trigger &mdash;
so the cooling-driven transition never fires and the cloud runs past dispersal under-cooled. Raising
\(f_\kappa\) raises \(\theta\); the <b>measured</b> full-run grid shows a steep density dependence:</p>
<div class="tablewrap"><table><thead><tr><th>config</th><th>\(n_{\rm core}\)</th>
<th>\(\theta(f_\kappa{=}1)\)</th><th>\(\theta(f_\kappa{=}4)\)</th><th>\(f_\kappa\) to fire (\(\theta\!\to\!0.95\))</th></tr></thead><tbody>
<tr><td>compact (simple_cluster)</td><td>\(10^5\)</td><td>0.667</td><td><b>1.024</b> (fired)</td><td>\(\approx 4\) <span class="small">(bracketed: fires at \(f_\kappa{=}4\))</span></td></tr>
<tr><td>mid (midrange_pl0)</td><td>\(10^4\)</td><td>0.610</td><td>0.814</td><td>\(\approx 5\!-\!6\) <span class="small">(extrapolated)</span></td></tr>
<tr><td>diffuse (f1edge_lowdens)</td><td>\(10^2\)</td><td>0.169</td><td>0.303</td><td>\(\approx 60\) <span class="small">(extrapolated)</span></td></tr>
</tbody></table></div>
<p class="small muted">\(\theta(f_\kappa{=}1)\) and \(\theta(f_\kappa{=}4)\) are <b>measured</b> full runs; only compact
reaches \(0.95\) within the measured \(f_\kappa\le4\) grid (it fires at \(f_\kappa{=}4\)), so the &ldquo;to fire&rdquo;
values for mid and diffuse are <b>extrapolations</b> beyond the grid, not measurements.</p>
<figure>__FIG_FKCAL__<figcaption><b>Left:</b> developed \(\theta=L_{\rm cool}/L_{\rm mech}\) (at cloud
dispersal) vs \(f_\kappa\) for the three measured configs, with the \(0.95\) <code>cooling_balance</code>
trigger. Compact crosses near \(f_\kappa\!\approx\!4\) (red ring = the cooling-driven transition fired &rarr;
momentum); mid is close behind; diffuse climbs far too slowly (needs \(f_\kappa\!\sim\!60\)). Dotted grey =
the old \(f_\kappa^{0.63}\) snapshot estimate, now known optimistic. <b>Right:</b> the cumulative radiated
fraction \(\int L_{\rm cool}\mathrm{d}t/\int L_{\rm mech}\mathrm{d}t\). From
<code>data/make_kappa_blowout_calibration.py</code>.</figcaption></figure>
<p class="small muted">Consistency note: the \(\theta\) here (the radiative loss fraction the
<code>cooling_balance</code> trigger watches) is a <i>different</i> ratio from the PdV-inclusive
\((L_{\rm loss}+P\mathrm{d}V)/L_{\rm gain}\) of &sect;12 (the <code>ebpeak</code> trigger). At \(f_\kappa=1\)
both sit below their thresholds for normal clouds; \(f_\kappa\) lifts the radiative one (this section), PdV
lifts the other (&sect;12). They are the two independent axes of the same &ldquo;1D cooling is too weak&rdquo;
problem.</p>
"""

SEC_TAXONOMY = r"""
<h2 id="taxonomy">14 &middot; Taxonomy &mdash; the approaches, disambiguated</h2>
<div class="box" style="border-left:4px solid #2a8aa8"><b>The point.</b> What looks like &ldquo;three ways to
boost cooling&rdquo; is really <b>two cooling-magnitude approaches on different sides of the solve, plus a
separate trigger axis</b>. The key disambiguation: <b>&ldquo;modify cooling like El-Badry with \(\kappa\)&rdquo;
and &ldquo;modify the conduction front \(k_f\)&rdquo; are the <i>same</i> knob</b> (<code>cooling_boost_kappa</code>)
&mdash; raising the conduction coefficient <i>is</i> the 1D stand-in for more radiating surface / mixing.</div>
<p>Every row is read from source &mdash; knob (<code>registry.py</code>), equation (file:line), no assumptions.</p>
<div class="tablewrap"><table><thead><tr>
<th>axis / approach</th><th>knob</th><th>what it changes (from source)</th>
<th>\(\theta\): imposed or <b>emergent</b>?</th><th>literature</th><th>status / verdict</th></tr></thead><tbody>
<tr><td colspan="6" style="background:#eef5f8"><b>A &middot; Outcome-side</b> &mdash; operate on \(L_{\rm loss}\)
<i>after</i> the structure solve; you impose the answer (<code>effective_Lloss</code>, <code>get_betadelta.py:334</code>)</td></tr>
<tr><td>scalar multiplier</td><td><code>cooling_boost_mode=multiplier</code>, \(f_{\rm mix}\)</td>
<td>\(L_{\rm loss} = L_{\rm leak} + f_{\rm mix}\,L_{\rm cool}\) &nbsp;(<code>:354</code>)</td>
<td>scaled (semi-imposed)</td><td>&mdash;</td>
<td>no single \(f_{\rm mix}\) fires across density (spans \(1.4\!-\!3.8\), &sect;3)</td></tr>
<tr><td><b>\(\theta\)-target floor</b> <span class="small">(&ldquo;sum like Lancaster \(\theta\)&rdquo;)</span></td>
<td><code>cooling_boost_mode=theta_target</code>, \(\theta\)</td>
<td>\(L_{\rm loss} = \max\!\big(L_{\rm cool}+L_{\rm leak},\ \theta\,L_{\rm mech}\big)\) &nbsp;(<code>:356</code>)</td>
<td><b>imposed</b> (top-down)</td><td><b>Lancaster</b> \(\theta\!\approx\!0.9\) (parameter-free loss fraction)</td>
<td>degenerate: constant \(\theta{=}0.95\) <i>is</i> the \(0.95\) trigger; \(\theta(\mathrm{Da})\) refuted (&sect;6)</td></tr>
<tr><td colspan="6" style="background:#eef5f8"><b>B &middot; Mechanism-side</b> &mdash; operate on the conduction
<i>inside</i> the structure solve; \(\theta\) comes out of the physics</td></tr>
<tr><td><b>\(\kappa_{\rm eff}\) conduction multiplier</b>
<span class="small">(&ldquo;El-Badry \(\kappa\)&rdquo; <b>=</b> &ldquo;modify \(k_f\) / conduction front&rdquo; &mdash; same knob)</span></td>
<td><code>cooling_boost_kappa</code>, \(f_\kappa\)</td>
<td>\(\kappa_{\rm eff}=f_\kappa\,C_{\rm th}\,T^{5/2}\) at 3 sites (<code>bubble_luminosity.py:291/370/406</code>)
&rarr; thicker conduction/evaporation front &rarr; more \(10^5\!-\!10^6\,\)K gas (more radiating <b>surface / mixing</b>)</td>
<td><b>emergent</b> (bottom-up)</td><td><b>El-Badry</b> mixing layer (\(\lambda\delta v \leftrightarrow \kappa_{\rm eff}\))</td>
<td>built/gated; measured \(f_\kappa\!\approx\!4\) (compact) &hellip; \(\sim\!60\) (diffuse) (&sect;13); side-effect: \(\dot M\!\uparrow\)</td></tr>
<tr><td colspan="6" style="background:#eef5f8"><b>C &middot; Trigger-side</b> &mdash; <i>when</i> to transition, not <i>how much</i> it cools</td></tr>
<tr><td>PdV-inclusive trigger</td><td><code>transition_trigger=ebpeak</code></td>
<td>fire when \(L_{\rm gain}-L_{\rm loss}-P\mathrm{d}V \le 0\) (<code>run_energy_implicit_phase.py:198,1206</code>)</td>
<td>n/a (timing only)</td><td>El-Badry/Lancaster &ldquo;cooling creeps up&rdquo;</td>
<td>does not fire alone at \(f_\kappa{=}1\); an assist, not a substitute (&sect;12)</td></tr>
</tbody></table></div>
<p><b>How to read it.</b> <b>A</b> imposes the result (a scaled cooling, or a floor on the loss fraction
&mdash; Lancaster&rsquo;s \(\theta\) lives here); <b>B</b> changes what <i>produces</i> the cooling, so \(\theta\)
emerges (El-Badry lives here, and it is the <i>same</i> knob as &ldquo;modify the conduction front&rdquo;);
<b>C</b> is a different axis entirely (the transition criterion). <b>A</b> and <b>B</b> must never be stacked
&mdash; the \(\max(\cdot)\) closure (&sect;2) keeps the loss single-count. The current direction is <b>B</b>
(\(\kappa_{\rm eff}\), &sect;11&ndash;13): the cooling-magnitude <i>mechanism</i>, calibrated to a
density-dependent target, with <b>C</b> (PdV) as an optional timing assist.</p>

<h3>\(f_\kappa\) vs \(f_{\rm mix}\) &mdash; the relation (they are <i>not</i> the same knob)</h3>
<p>Both raise the cooling, but at <b>different points in the calculation</b> &mdash; this is the row-A vs
row-B distinction made concrete:</p>
\[ \underbrace{L_{\rm loss} = L_{\rm leak} + f_{\rm mix}\,L_{\rm cool}}_{\textbf{A: } f_{\rm mix}\ \text{scales the OUTPUT } L_{\rm cool}}
   \qquad\text{vs}\qquad
   \underbrace{\kappa_{\rm eff} = f_\kappa\,C_{\rm th}\,T^{5/2}\ \Rightarrow\ L_{\rm cool}\ \text{emerges}}_{\textbf{B: } f_\kappa\ \text{scales the CONDUCTION COEFFICIENT}}. \]
<p><b>(i) The equivalence is exact by definition.</b> Since \(f_{\rm mix}\) is <i>defined</i> as the multiplier
on \(L_{\rm cool}\), the \(f_{\rm mix}\) that reproduces a given \(f_\kappa\) is, with no model assumed, just the
\(L_{\rm cool}\) boost ratio at matched conditions:</p>
\[ f_{\rm mix}^{\rm equiv}(f_\kappa) \;\equiv\; \frac{L_{\rm cool}(f_\kappa)}{L_{\rm cool}(f_\kappa{=}1)}\Big|_{\rm matched\ }t. \]
<p>The matched-\(t\) back-reaction (&sect;11) <b>measures</b> it: \(f_\kappa=2 \Rightarrow f_{\rm mix}^{\rm equiv}
\in[1.23,\,1.50]\) (developed \(\to\) seed). That much is unimpeachable.</p>
<p><b>(ii) The exponent is bracketed by one exact scaling, not derived.</b> \(L_{\rm cool}\) is <i>not</i> a clean
power of \(f_\kappa\): its effective exponent \(q\) (where \(f_{\rm mix}^{\rm equiv}=f_\kappa^{q}\)) runs from
\(q\!\approx\!0.58\) (seed) to \(q\!\approx\!0.30\) (developed), and the \(\theta_{\rm blowout}\) leverage (a
noisier proxy &mdash; measured at a <i>different</i> \(t\) per \(f_\kappa\)) scatters \(0.21\!-\!0.42\) across
configs (&sect;13). The one <b>exactly-derived</b> anchor is the evaporation rate
\(\dot M\propto f_\kappa^{2/7}\) (Weaver+77 Eq.&nbsp;33, verified at <code>bubble_luminosity.py:291</code>), and
the <i>developed</i> \(L_{\rm cool}\) boost lands on it: \(1.228\approx 2^{2/7}=1.219\). That is the
turbulent-mixing-layer relation \(L\sim\dot m\,\times\,\)enthalpy (El-Badry / Fielding / Tan-Oh-Gronke), with
\(\dot m\propto f_\kappa^{2/7}\). So the <b>low end</b> of the exponent (\(\sim\!2/7\)) is physically grounded;
the higher seed value is a transient. Net:</p>
\[ f_{\rm mix}^{\rm equiv} \;\approx\; f_\kappa^{\,q}, \quad q\sim 0.3\!-\!0.6\ (\text{bracketed below by } 2/7),
   \qquad\Rightarrow\qquad f_\kappa \;\approx\; (f_{\rm mix})^{1/q}\ \gg f_{\rm mix}. \]
<p>Because \(q<1\) you need a <b>much larger</b> \(f_\kappa\) than \(f_{\rm mix}\) (e.g. \(f_{\rm mix}\!=\!2\)
needs \(f_\kappa\!\sim\!6\!-\!10\)) &mdash; which is why the &sect;3 firing-\(f_{\rm mix}\) values (\(1.4\!-\!3.8\))
and the &sect;13 firing-\(f_\kappa\) values (\(\sim\!4\!-\!60\)) look so different: same physics, two
parametrisations. <b>But this is a heuristic magnitude map, not a closed-form law</b> &mdash; \(q\) is
epoch- and config-dependent, pinned only at the exact \(2/7\) evaporation floor.</p>
<div class="box" style="border-left:4px solid #b8860b"><b>But the magnitude is the lesser difference.</b>
\(f_{\rm mix}\) is a <b>pure rescale of one number</b> &mdash; it multiplies \(L_{\rm cool}\) and nothing else;
\(\dot M\), \(E_b\), \(P_b\), the shell dynamics are untouched (the bubble never &ldquo;knows&rdquo; the
cooling was boosted). \(f_\kappa\) is a <b>structural</b> change: raising the conduction self-consistently
also raises \(\dot M\propto f_\kappa^{2/7}\), drains \(E_b\), lowers \(P_b\), and \(\theta\) comes out as an
<i>output</i> (&sect;13). So even at <i>matched</i> \(L_{\rm cool}\) the two give <b>different trajectories</b>
&mdash; \(f_{\rm mix}\) is the crude floor (Lancaster-style outcome), \(f_\kappa\) the faithful El-Badry
mechanism. They are <b>not interchangeable</b>; the conversion above is only an \(L_{\rm cool}\)-magnitude
equivalence, not a dynamical one.</div>
"""

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/pdv-trigger/</code> &mdash; reproducible
without re-running the (hours-long) sims. Each figure is a pure read of a committed CSV; the prose numbers trace to
the same files.</p>
<div class="box" style="border-left:4px solid #2a8aa8"><b>Paper reproducibility manifest:</b>
<code>docs/dev/transition/pdv-trigger/REPRODUCE.md</code> &mdash; the single map from <b>every result in this
report</b> to <b>the exact <code>.param</code> + command + artifact</b>, tagged cheap (re-reads a committed CSV)
vs expensive (a full sim). Use it to re-run any piece for a paper and to prove the storyline is reproducible.</div>
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
<tr><td><code>fkappa_definition.png</code> (+ <code>data/make_fkappa_definition.py</code>)</td><td>&sect;13 the \(f_\kappa\) definition figure: the Spitzer conductivity it multiplies, and the analytic seed law \(\dot M\propto f_\kappa^{2/7}\) verified vs the measured back-reaction (1.2175 vs 1.219)</td></tr>
<tr><td><code>data/kappa_blowout_calibration.csv</code> (+ <code>make_kappa_blowout_calibration.py</code>, <code>kappa_blowout_calibration.png</code>)</td><td>&sect;13 the measured \(f_\kappa\) calibration: developed \(\theta\) vs \(f_\kappa\) for compact/mid/diffuse (full runs <code>cal_{compact,diffuse}__k*</code> + <code>cal_mid__ek*</code>); compact fires \(\theta\!\to\!0.95\) at \(f_\kappa\!\approx\!4\), diffuse needs \(\sim\!60\)</td></tr>
<tr><td><code>data/ebpeak_trigger_test.csv</code> (+ <code>make_ebpeak_trigger_test.py</code>, <code>ebpeak_trigger_test.png</code>)</td><td>&sect;12 the live ebpeak code-path test (compact/diffuse/mid, <code>cal_*__ebpeak</code> / <code>cal_mid__ek*</code>): ebpeak never fires at \(f_\kappa{=}1\) (peaks below 1.0 then declines)</td></tr>
<tr><td><code>data/ebpeak_8config_xcheck.csv</code> (+ <code>make_ebpeak_8config_xcheck.py</code>, <code>ebpeak_8config_xcheck.png</code>)</td><td>&sect;12 the 8-config frozen-screen cross-check + live overlay: 6 normal clouds peak 0.85&ndash;0.92, live matches frozen to the digit</td></tr>
<tr><td><code>data/_trinity_style.py</code></td><td>the shared TRINITY plot style (loads <code>paper/_lib/trinity.mplstyle</code>, LaTeX-free fallback) so every storyline figure is visually consistent</td></tr>
<tr><td><code>data/ebpeak_trigger_test.csv</code> (+ <code>make_ebpeak_trigger_test.py</code>, <code>ebpeak_trigger_test.png</code>)</td><td>&sect;12 does PdV alone trigger? Two <code>cooling_balance,ebpeak</code>-active runs (<code>runs/params/cal_&#123;compact,diffuse&#125;__ebpeak.param</code>): ebpeak never fires at \(f_\kappa{=}1\) (PdV-incl peaks \(0.91\)/\(0.86\) then declines); the trade-off keeps diffuse PdV-incl flat across \(f_\kappa\)</td></tr>
<tr><td><code>ideas_comparison.png</code> (+ <code>data/make_ideas_comparison.py</code>)</td><td>the at-a-glance scoreboard of all ideas + three real-data evidence panels (reads <code>fmix_table</code>, <code>da_replay</code>, <code>kappa_backreaction</code> CSVs)</td></tr>
<tr><td><code>KAPPA_EFF_SCOPING.md</code></td><td>&sect;11 the κ_eff cooling-mechanism feasibility map + the &sect;6a Rung-A result table; the mechanism vs the optional evaporation-decoupling bonus</td></tr>
<tr><td><code>storyline_figs/*.png</code> (+ <code>make_storyline_figs.py</code>)</td><td>the four storyline figures (&sect;2 double-count, &sect;3 convention, &sect;4 regime, &sect;5 heatmap), each a pure read of the CSVs</td></tr>
<tr><td><code>PLAN.md</code> / <code>FINDINGS.md</code></td><td>the living plan (&ldquo;Outcome &amp; pivot&rdquo;) and the verified findings &mdash; the source of truth this report is a sibling of</td></tr>
</tbody></table></div>
<p class="small muted">Rebuild this report:
<code>python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py</code> (figures are committed PNGs, not
regenerated here). Then rebuild the storyline book:
<code>python docs/dev/html-insights/build_storylines.py</code>.</p>
<footer>TRINITY PdV-in-the-trigger &rarr; unresolved-interface-cooling closure &middot; boost the cooling
<i>magnitude</i> (\(\kappa_{\text{eff}}\)), so the cooling-driven <code>cooling_balance</code> trigger fires;
including PdV (<code>ebpeak</code>) is an assist for timing, not a substitute &middot; every number traces to a
committed CSV in <code>docs/dev/transition/pdv-trigger/{data,runs/data}/</code> &middot; light-mode,
MathJax-rendered.</footer>
"""


def main():
    parts = [
        HEAD, HERO,
        SEC_SETUP, SEC_DOUBLE, SEC_CONVENTION, SEC_REGIME, SEC_CLOSURE,
        SEC_DA, SEC_LIT, SEC_WIRING, SEC_LIVE, SEC_BOTTOM, SEC_KAPPA, SEC_EBPEAK, SEC_FKDEF,
        SEC_TAXONOMY, SEC_REPRO,
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
