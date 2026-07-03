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
    "__FIG_FFORM__": (
        "fkappa_functional_form.png",
        "the composed f_kappa(n_H) form: left = measured theta(f_kappa) accelerates past 1 (fires before saturating); middle = TRINITY baseline theta0 rises with density while the El-Badry/Lancaster target is flat; right = the resulting f_kappa(n_H) ~ A n^-0.30 with the measured firing anchor",
    ),
    "__FIG_SWEEP__": (
        "fkappa_nH_sweep.png",
        "the 819-combo sweep de-conflation figure, faceted by sfe (three panels): f_kappa to fire theta=0.95 vs nCore, one line per cloud mass; the M_cl/sfe series do NOT collapse onto one curve (fan-out), the 1e7 line cliffs to f_kappa=1, and triangles mark cells that never fire by f_kappa=64",
    ),
    "__FIG_SCORE__": (
        "fkappa_sweep_analysis.png",
        "sweep prediction scorecard: left = measured fan-out with the all-data n^-0.60 fit (steeper than the pre-registered n^-0.30) and never-fire markers; right = pre-registered predicted vs measured f_kappa per cell, systematically off the 1:1 line because the predicted slope was too shallow",
    ),
    "__FIG_CLIFF__": (
        "fkappa_cliff_metric.png",
        "anatomy of the fan-out: baseline theta at f_kappa=1 vs density (left) cliffs at DIFFERENT nCore per cloud mass, but vs column N_H=nCore*rCloud (right) the cliffs roughly align at ~constant column, so the massive-cloud early firing is a swept-column catastrophic-cooling threshold",
    ),
    "__FIG_CAP__": (
        "fkappa_physical_cap.png",
        "the physical-cap reframing: left = f_kappa needed to fire vs column with physical cap lines (f_max=2/4/8); below a cap a cloud is momentum-driven, above it stays energy-driven, and 6 cells never fire under any cap; right = the momentum-vs-energy-driven split as a function of the assumed physical max enhancement f_max, crossing near f_max=8, with the 6 never-fire cells as the floor",
    ),
    "__FIG_DERIV__": (
        "fkappa_physical_derivation.png",
        "deriving the physical prescription: left = Spitzer conductivity (rises as T^5/2) vs El-Badry's temperature-independent kappa_mix at GMC density, showing kappa_mix dominates the cool mixing layer where Spitzer vanishes (so a scalar f_kappa multiplier cannot represent it); right = El-Badry's verified flat-high cooling target theta* (even at diffuse) vs TRINITY's measured rising 1D baseline theta0, the orange gap kappa_mix must supply is large at diffuse, arguing the diffuse never-fire is a 1D under-cooling artifact",
    ),
    "__FIG_T5ARMS__": (
        "theta5_arms.png",
        "the theta5 matrix: emergent theta_max vs f_mix for all 8 configs over 5 Myr, outcome-classed (fires-and-survives, fires-then-recollapses, route-a, PdV/handoff), with the 0.95 trigger and the Lancaster 0.90-0.99 band; f_mix=4 lifts the whole normal-GMC band across the trigger",
    ),
    "__FIG_T5LAW__": (
        "theta5_collapse_law.png",
        "the multiplier theta1-collapse law: smallest firing boost vs starting deficit 0.95/theta0 in log-log, five fired configs with grid brackets, the censored small_1e6 arrow, the fitted f_fire=1.4(0.95/theta0)^1.82 line and the much steeper kappa-knob law for contrast",
    ),
    "__FIG_T5METRIC__": (
        "theta5_metric_correction.png",
        "why blowout-theta was retired: per-config dumbbells from theta-at-blowout to theta_max over 5 Myr; the diffuse config is under-read 2.1x because its theta peaks at t~4.9 Myr, long after blowout",
    ),
    "__FIG_T5TARGET__": (
        "theta5_target_vs_emergent.png",
        "calibrate-don't-enforce: the El-Badry lambda-delta-v=3 target curve vs TRINITY's native theta0 (open) and f_mix=4 boosted theta_max (filled) per config; arrows show the boost lifting GMCs into the band; two configs at the same nCore=1e2 behave oppositely, so theta0 is not a function of density alone",
    ),
    "__FIG_T5KNOB__": (
        "theta5_knob_choice.png",
        "two-panel knob comparison with distinct x-axes: LEFT the structural f_kappa on one pre-fix sweep cell (crash windows at the condensation boundary, f=16 fire an artifact), RIGHT the post-solve f_mix multiplier (theta5) whose theta_max rises smoothly with no freezes",
    ),
    "__FIG_T5BMAP__": (
        "theta5b_fire_map.png",
        "fire map over theta5+theta5b: per config and f_mix the outcome class (fires in band / momentum without firing via Eb drain / stays energy-driven / NaN = solve never succeeded, L_loss stayed at its NaN default); the measured whole-band window 4 to 4.5 is shaded; the diffuse 8-Myr arms shown above the diffuse row",
    ),
    "__FIG_T5BLAW__": (
        "theta5b_law_check.png",
        "predicted vs measured f_fire for the six firing configs: the theta5-fit collapse law predicts the theta5b fine bracket out-of-sample with rms 0.064 dex",
    ),
    "__FIG_T5KMAP__": (
        "theta5k_fire_map.png",
        "theta5k fire map: outcome per config and f_kappa on the first rule-compliant kappa matrix, post no-root handoff -- 56/56 proper fates, zero freezes, five condensation handoffs on the old dead-window cells, and no single f_kappa fires the whole band",
    ),
    "__FIG_T5KRISE__": (
        "theta5k_theta_rise.png",
        "theta_max versus f_kappa per config: theta rises near-monotonically with f_kappa even where the fire set does not -- open markers are runs ended by the condensation handoff or drain before theta could cross 0.95",
    ),
    "__FIG_DMDTDIP__": (
        "dmdt_dip_traces.png",
        "the dMdt dip: per-segment mass-flux eigenvalue of the beta-delta solve for the dense controlled pair -- both arms dip below zero into the condensation branch the gate forbids; f_kappa=8 recovers through zero and fires while f_kappa=6 second-dives and is handed off",
    ),
    "__FIG_DMDTFLOW__": (
        "dmdt_tackle_flow.png",
        "flow diagram of how the dMdt dip was tackled: symptom (silent freeze) to diagnosis (negative eigenvalue refused by the gate) to physics identity (McKee-Cowie condensation reversal) to the three literature treatments, with the adopted path (energy-to-momentum regime switch) and the theta5k verdict",
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
    + MATHJAX
    + "<style>"
    + CSS
    + '</style></head><body><div class="wrap">'
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
everywhere, so \(\theta_{\max}\mathrm{Da}/(1+\mathrm{Da})\) saturates back to a constant). <span style="background:#eef9f1;border-radius:4px;padding:0 4px"><b>Addendum 2026-07-02 (&sect;16):</b> claim (3) was a <i>blowout-metric</i> artifact &mdash; under the \(\theta_{\max}\)-over-5-Myr rule a single \(f_{\rm mix}{=}4\) <b>does</b> fire the whole normal-GMC band.</span> <b>The pivot:</b> the cooling
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
+ Lancaster). The evaporation-decoupling re-derivation is an <b>optional fidelity bonus</b>, not the goal.
<b>Now done (2026-06-29, &sect;15):</b> El-Badry &sect;5.2 is <b>verified from the PDF</b>
(\(\kappa_{\text{mix}}{=}(\lambda\delta v)\rho k_B/\mu m_p\); \(\theta{=}\psi/(\tfrac{11}{5}{+}\psi)\), \(A_{\text{mix}}{\approx}3.5\)),
and the controlled <b>819-combo sweep</b> ran: the measured calibration is \(f_\kappa^{\text{fire}}\!\approx\!10^{3}n^{-0.60}\),
but \(f_\kappa\) is <b>multi-dimensional</b> (a \(\times2\!-\!32\) \(M_{\text{cl}}\)/sfe fan-out), the diffuse/high-sfe
corner <b>never fires</b> even at \(f_\kappa{=}64\) (needs the structural \(\kappa_{\text{mix}}\)), and the
&ldquo;1e7 breaks the power law&rdquo; cliff is a <b>constant-column catastrophic-cooling threshold</b>. <b>The
reframing (&sect;15.6):</b> don&rsquo;t force it &mdash; the <i>physical</i> \(f_\kappa\) <i>rises</i> with density
(opposite the fire-threshold), so a physically-bounded \(f_\kappa\) leaves the diffuse corner <b>energy-driven by
choice</b>, predicting a falsifiable <b>critical column</b> for the energy\(\to\)momentum split.</p>
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
"""

SEC_SWEEP = r"""
<h2 id="sweep">15 &middot; The \(f_\kappa(n_H)\) calibration &mdash; composed, verified, then swept</h2>
<p>&sect;11&ndash;13 left the merge with one job: <b>calibrate \(f_\kappa(\text{properties})\)</b> so the
<i>emergent</i> \(\theta=L_{\text{cool}}/L_{\text{mech}}\) reaches the obs/3D target, density-dependently. This
section does it three ways &mdash; a composed closed form, a line-by-line literature verification, and the
controlled 819-combo sweep that tests both &mdash; and ends on what the sweep actually taught us.</p>

<h3>15.1 &middot; A composed functional form (before the sweep)</h3>
<p>Rather than wait for the grid to fit \(f_\kappa(n_H)\) cold, we compose it from three separable, independently
checkable pieces: a <b>target</b> \(\theta^\star\) (the Lancaster 2021 plateau \(\approx0.9\), density-<i>independent</i>
over &gt;3 dex), a <b>baseline</b> \(\theta_0(n_H)\) (TRINITY&rsquo;s emergent loss fraction at \(f_\kappa{=}1\),
rising with density), and a <b>leverage</b> \(p\) (how \(\theta\) responds to \(f_\kappa\)). Inverting
\(\theta=\theta_0\,f_\kappa^{p}\) gives \(f_\kappa(n_H)=(\theta^\star/\theta_0(n_H))^{1/p}\).</p>
<div class="box over"><span class="lab">Course-correction (same day)</span> A first cut inverted the leverage in
<i>logit/odds</i> space and overshot the one measured firing anchor by \(\sim\!10\!-\!30\times\) (compact fires at
\(f_\kappa\!\approx\!3.4\), not \(\sim\!120\)). Cause: \(\theta(f_\kappa)\) <b>accelerates toward firing</b>
(convex, because the bubble transitions <i>before</i> \(\theta\) saturates), so a concave logit extrapolated from
the low-\(f_\kappa\) segment is wrong. The fix is the raw power-law exponent measured over the full range to firing
(\(p\!\approx\!0.31\)), which reproduces the anchor and matches the El-Badry back-reaction estimate
(\(f_{\text{mix}}{=}f_\kappa^{q},\ q{=}\ln1.3/\ln2\!\approx\!0.4\)). Only the amplitude changed; the slope did not.</div>
<figure>__FIG_FFORM__<figcaption><b>The composed form.</b> Left: the measured \(\theta(f_\kappa)\) accelerates past
1 (compact \(0.67\!\to\!0.74\!\to\!1.02\) at \(f_\kappa{=}1,2,4\)) &mdash; it fires before it saturates, which is why
the raw power-law leverage, not a logistic, is correct. Middle: the baseline \(\theta_0(n_H)\) rises while the
target is flat, so the density dependence of \(f_\kappa\) comes from the baseline. Right: the resulting
\(f_\kappa(n_H)\) with the measured firing anchor. From <code>data/make_fkappa_functional_form.py</code>.</figcaption></figure>

<h3>15.2 &middot; El-Badry 2019 &mdash; verified line-by-line from the PDF</h3>
<div class="box find"><span class="lab">Verified (maintainer-supplied PDF)</span> The El-Badry, Ostriker, Kim,
Quataert &amp; <b>Weisz</b> 2019 paper (MNRAS <b>490</b>, 1961; arXiv:1902.09547 &mdash; <i>not</i> ApJ 879, <i>not</i>
&ldquo;Weinberg&rdquo;) supplies the mechanism and the target, and its &sect;3.1/&sect;5.2 equations check out exactly:
<b>Eq 16</b> Spitzer \(\kappa{=}6{\times}10^{-7}T^{5/2}\); <b>Eq 19/20</b> saturation \(q_{\text{sat}}{=}\tfrac32\rho c_{s,\text{iso}}^3\)
(\(=5\phi\rho c^3,\ \phi{=}0.3\)); <b>Eq 21</b> the turbulent-mixing term \(\kappa_{\text{mix}}{=}(\lambda\delta v)\rho k_B/\mu m_p\)
&mdash; <i>temperature-independent</i>, \(\kappa{=}\max(\kappa_{\text{mix}},\kappa_{\text{Spitzer}})\); <b>Eq 35/37/38</b>
\(\theta{=}\psi/(\tfrac{11}{5}{+}\psi)\) with \(\psi{=}A_{\text{mix}}\sqrt{\lambda\delta v\,n_H}\), \(A_{\text{mix}}\!\approx\!1.7\)
analytic / <b>3.5</b> fit. The earlier in-container <code>[unverified]</code> hedge (a 403 access gap, not an error) is
<b>retracted</b>. Bonus: El-Badry themselves propose calibrating \(\lambda\delta v\) to 3D cooling rates &mdash;
exactly this workstream&rsquo;s strategy. Crucially the El-Badry target <i>saturates</i> to \(0.94\!-\!0.999\) across the
GMC range, so it <b>agrees with the flat Lancaster anchor to \(\sim\!15\%\) in \(f_\kappa\)</b> &mdash; the composed form
is robust to which verified target is used.</div>

<h3>15.3 &middot; The 819-combo sweep &mdash; predictions scored</h3>
<p>The controlled grid (7 \(n_{\text{core}}\) &times; 3 \(M_{\text{cl}}\) &times; 3 sfe = 63 cells &times; 13
\(f_\kappa\)) ran on Helix (\(\sim\)10 h). The composed form&rsquo;s predictions were <b>pre-registered</b>; the grid
graded them honestly.</p>
<div class="tablewrap"><table><thead><tr><th>prediction (pre-registered)</th><th>measured (63 cells)</th><th></th></tr></thead><tbody>
<tr><td>slope \(f_\kappa\propto n^{-0.30}\)</td><td>\(f_\kappa^{\text{fire}}\approx10^{3}\,n_{\text{core}}^{-0.60}\)</td><td class="loss">&times;2 too shallow</td></tr>
<tr><td>de-conflation: fan-out, not one curve</td><td>\(\times2\!-\!32\) spread across \(M_{\text{cl}}\)/sfe</td><td class="win">PASS</td></tr>
<tr><td>baseline \(\theta_0(n)\) logit slope 0.41/dex</td><td>1.13/dex</td><td class="loss">\(\sim\)3&times; steeper</td></tr>
<tr><td>leverage \(p\approx0.31\)</td><td>median 0.21</td><td class="muted">ballpark</td></tr>
<tr><td>diffuse unreachable \(\to\kappa_{\text{mix}}\)</td><td>6/63 low-\(n\)/high-sfe cells never fire at \(f_\kappa{=}64\)</td><td class="win">PASS</td></tr>
</tbody></table></div>
<p>The <b>qualitative physics held and is now measured</b>: \(f_\kappa\) falls steeply with density, is
<b>multi-dimensional</b> (the fan-out), and the diffuse/high-sfe corner is genuinely unreachable by a Spitzer boost
(those six cells need the El-Badry \(\kappa_{\text{mix}}\), Eq 21). The <b>slope was \(\times2\) too shallow</b> for a
nameable reason: the composed form is only as good as its baseline, and the original <b>6-anchor \(\theta_0(n)\) was
undersampled</b> (0.41/dex) &mdash; the clean 63-cell grid gives 1.13/dex. The logistic-vs-raw-power leverage debate
turned out to be second-order next to that.</p>
<figure>__FIG_SCORE__<figcaption><b>Scorecard.</b> Left: the measured fan-out with the all-data \(n^{-0.60}\) fit
(black) far steeper than the pre-registered \(n^{-0.30}\) (red), plus the never-fire markers. Right:
predicted-vs-measured \(f_\kappa\) per cell tilts systematically off the 1:1 line &mdash; the predicted slope was too
shallow. From <code>data/make_fkappa_sweep_analysis.py</code> (reads the committed <code>data/fkappa_nH_sweep.csv</code>).</figcaption></figure>
<figure>__FIG_SWEEP__<figcaption><b>The de-conflation figure</b> (faceted by sfe). One line per cloud mass; the series
do <b>not</b> collapse onto one \(n_{\text{core}}\) curve &mdash; \(f_\kappa\) depends on more than density. The 1e7
(yellow) line cliffs abruptly to \(f_\kappa{=}1\); triangles mark cells that never fire by \(f_\kappa{=}64\). From
<code>data/make_fkappa_nH_sweep.py</code> (reads <code>data/summary.csv</code>).</figcaption></figure>

<h3>15.4 &middot; Anatomy of the fan-out &mdash; the catastrophic-cooling cliff</h3>
<p>Why does the 1e7 cloud &ldquo;break the power law&rdquo;? Because its baseline \(\theta_0\) (at \(f_\kappa{=}1\),
no boost) <b>jumps past 0.95</b> &mdash; above that it fires with zero boost, so \(f_\kappa^{\text{fire}}\) collapses
to 1. That cliff sits at <i>lower density</i> for <i>more massive</i> clouds (\(n\!\approx\!3{\times}10^3\) for 1e7 vs
\(\approx\!2{\times}10^4\) for 1e5). The reason: at fixed density a 1e7 cloud is \(\sim\!4.6\times\) larger
(\(r_{\text{cloud}}\!\propto\!(M/n)^{1/3}\)), so it sweeps the same <b>column</b> \(N_H{=}n_{\text{core}}\,r_{\text{cloud}}\)
at lower density. Re-plotting vs column roughly <b>halves the cliff spread</b> (\(\times11\) in \(n_{\text{core}}\)
\(\to\!\times5.7\) in column; median cliff column \(\approx\!8{\times}10^{23}\,\text{cm}^{-2}\)). Physically it is
&ldquo;does catastrophic cooling beat cloud crossing&rdquo; &mdash; for big clouds, cooling wins at lower ambient density.</p>
<div class="box"><span class="lab">The fan-out is multi-dimensional (not pure column)</span> Across all 63 cells,
\(n_{\text{core}}\) is still the best <i>single</i> predictor of \(\theta_0\) (\(R^2{=}0.73\)); column is slightly
worse globally (\(0.71\)) even though it nails the cliff; a 2-variable
\(\theta_0\propto+0.11\ln n_{\text{core}}+0.06\ln r_{\text{cloud}}\) reaches \(R^2{=}0.75\) (coef ratio 2:1, so
<i>not</i> pure column). And \(f_\kappa^{\text{fire}}\) is <b>independent of cluster mass</b> \(M_\star{=}\text{sfe}\cdot M_{\text{cl}}\)
(\(R^2{=}0.002\)) &mdash; as it must be, since \(\theta\) is normalised by \(L_{\text{mech}}\propto M_\star\). So the
calibration needs \(f_\kappa(n_{\text{core}}, r_{\text{cloud}}[,\text{sfe}])\), or the structural \(\kappa_{\text{mix}}\)
for the corner that never fires. <span class="small muted">[The maintainer&rsquo;s PdV intuition is directionally
right &mdash; massive clouds fire with less boost &mdash; but the driver the data supports is the swept column / radiative
catastrophic cooling, not PdV directly; the firing metric is the radiative \(\theta\), and PdV is not isolable from the
reduced data.]</span></div>
<figure>__FIG_CLIFF__<figcaption><b>The cliff.</b> Baseline \(\theta\) at \(f_\kappa{=}1\) vs density (left) cliffs at
different \(n_{\text{core}}\) per cloud mass; vs column \(N_H\) (right) the cliffs roughly align at a near-constant
column &mdash; the massive-cloud early firing is a swept-column catastrophic-cooling threshold. From
<code>data/make_fkappa_cliff_metric.py</code>.</figcaption></figure>

<h3>15.5 &middot; The measurement metric &mdash; \(\theta\) at blowout</h3>
<p>What does &ldquo;triggering 0.95&rdquo; measure? \(\theta=L_{\text{cool}}/L_{\text{mech}}\) sampled <b>per timestep
during the energy phase</b> &mdash; not a fixed \(t\), not integrated to \(t_{\text{stop}}{=}10\). The reducer keeps
<code>theta_blowout</code> (\(\theta\) at the first \(R_2{&gt;}r_{\text{cloud}}\), i.e. the cloud edge) and
<code>theta_max</code> (the peak); &ldquo;fires&rdquo; \(=\) reached transition/momentum <b>and</b> (never blew out
<b>or</b> \(\theta_{\max}{\ge}0.95\)). <b>Why blowout:</b> the science question is whether the cluster transitions to
momentum-driven <i>while still inside the GMC</i>, and \(R_2{=}r_{\text{cloud}}\) is the natural end of that phase &mdash;
more principled than a fixed time, which would fold in post-escape ambient evolution. The runs split cleanly:
<b>403/819 cool before escaping</b> vs <b>416/819 reach blowout</b>.</p>
<div class="box find"><span class="lab">Is blowout a good metric? Yes &mdash; and it&rsquo;s robust</span> The
snapshot-vs-peak choice barely matters: \(\theta_{\max}-\theta_{\text{blowout}}\) has <b>median 0.004</b> (\(&gt;0.05\)
in only 5/63 cells), so the calibration is insensitive to it, and the cliff/fan-out is genuine physics, not a metric
artifact. <span class="small muted">One fixable imprecision: <code>theta_max</code> is taken over the whole implicit
phase, not capped at <code>blowout_t</code>, so a post-escape peak could falsely tag &ldquo;fired in-cloud&rdquo;
(\(\sim\)5 cells); capping it in the reducer is the clean fix. Alternatives answering different questions: \(\theta\) at
matched physical time (apples-to-apples leverage), or time-integrated \(\int L_{\text{cool}}\,dt/\int L_{\text{mech}}\,dt\)
(the total budget).]</span></div>
<h3>15.6 &middot; Don&rsquo;t force it &mdash; a physically-bounded \(f_\kappa\) and a critical column</h3>
<p>Searching \(f_\kappa\) up to 64 to make <i>every</i> cloud fire quietly assumes every cloud <b>must</b> become
momentum-driven. It shouldn&rsquo;t. A large constant \(f_\kappa\) is not &ldquo;enhanced conduction&rdquo; &mdash; it
multiplies the Spitzer \(T^{5/2}\) prefactor <i>everywhere</i>, over-conducting the hot interior; the physical
enhancement is El-Badry&rsquo;s temperature-independent \(\kappa_{\text{mix}}{=}(\lambda\delta v)\,n\,k_B\) (Eq 21),
which dominates only in the cool mixing layer.</p>
<div class="box find"><span class="lab">The sign flip &mdash; the crux</span> Because \(\kappa_{\text{mix}}\!\propto\!n\)
and \(\kappa_{\text{Spitzer}}\!\propto\!T^{5/2}\) (\(n\)-independent), the <b>physical</b> \(f_\kappa\) <b>rises</b>
with density (\(\propto n^{+1}\)), while the measured fire-threshold <b>falls</b> (\(\propto n^{-0.6}\)) &mdash;
<b>opposite signs</b>. So using the empirical \(-0.6\) as a prescription gives the <i>diffuse</i> clouds the
<i>most</i> boost &mdash; precisely the &ldquo;forcing&rdquo; that feels wrong. The physical (rising) prescription
gives diffuse the <i>least</i>: dense clouds transition, diffuse stay energy-driven and blow out. That is the honest
reading.</p></div>
<p><b>The experiment</b> (pure re-analysis of <code>summary.csv</code>, no new sims): cap the enhancement at a
physical \(f_{\max}\); a cloud is momentum-driven iff \(f_\kappa^{\text{fire}}\!\le\!f_{\max}\), else energy-driven.
A physically plausible \(f_{\max}\!\approx\!2\!-\!8\) predicts a <b>critical column</b>
\(N_{\text{crit}}\!\approx\!1\!-\!4{\times}10^{23}\,\text{cm}^{-2}\): above it momentum-driven, below it energy-driven;
<b>6/63 cells never fire under any cap</b> (genuinely energy-driven in 1D). This is a <b>falsifiable prediction</b>
to compare with Lancaster/PHANGS, not a knob tuned to force a transition.</p>
<figure>__FIG_CAP__<figcaption><b>The physical-cap reframing.</b> Left: \(f_\kappa\) to fire vs column with physical
cap lines &mdash; below a line the cloud is momentum-driven, above it stays energy-driven (the boundary is soft, as
column is only a partial predictor). Right: the momentum/energy split as a function of the assumed physical
\(f_{\max}\), with the 6 never-fire cells as the floor. From <code>data/make_fkappa_physical_cap.py</code>.</figcaption></figure>
<div class="box over"><span class="lab">The tension to keep honest</span> Lancaster&rsquo;s 3D finds catastrophic
cooling &ldquo;generic over &gt;3 dex in density&rdquo;, so a non-transitioning <i>1D</i> cloud is <b>either</b>
genuinely energy-driven <b>or</b> 1D-under-cooled (missing the \(\kappa_{\text{mix}}\) it cannot resolve). The
critical-column prediction is the dividing line; which side is right is settled against observations. Two routes
follow: <b>(a)</b> accept non-transition (a physical \(f_{\max}\)) &mdash; simple and honest about the 1D limit; or
<b>(b)</b> add the structural \(\kappa_{\text{mix}}\) (Rung B) if you trust the 3D result. Either way the deliverable
is a <i>physically-bounded</i> prescription, not \(f_\kappa\) cranked to 64. <span class="small">Testing it needs no
new grid &mdash; any \(f_\kappa(n){=}\text{clamp}(A n^{q},1,f_{\max})\) is read off the existing
<code>summary.csv</code>; a small 63-run generator sweep only confirms a chosen prescription as real runs.</span></div>
<h3>15.7 &middot; The physical prescription, derived &mdash; it is \(\kappa_{\text{mix}}(\lambda\delta v)\), not a power law</h3>
<p>Chasing the &ldquo;negative power isn&rsquo;t physical&rdquo; objection to its end settles the whole question.
There are <b>three</b> different \(f_\kappa(n)\), and only one is the mechanism: <b>(i) mechanism</b>
\(f_\kappa{=}\kappa_{\text{mix}}/\kappa_{\text{Spitzer}}{\propto}n\) (rises); <b>(ii) target</b>
\(\theta^\star(n;\lambda\delta v)\) (Eq 37/38, flat-high); <b>(iii) boost</b> to reach the target \(\propto n^{-0.6}\)
(falls &mdash; a boost factor, <i>not</i> a conductivity). The empirical \(-0.6\) is (iii); it is the wrong object to
call a prescription.</p>
<div class="box find"><span class="lab">Derived</span> Crossover \(\kappa_{\text{mix}}{=}\kappa_{\text{Spitzer}}\) at
\(n_{\text{crit}}{=}C_{\text{th}}T^{5/2}/((\lambda\delta v)k_B)\) \(=\) <b>0.25 cm\(^{-3}\)</b> (T=2&times;10\(^5\) K,
\(\lambda\delta v{=}1\)) &mdash; matching El-Badry&rsquo;s &ldquo;\(\kappa_{\text{mix}}\) dominates \(n\!\gtrsim\!0.2\)&rdquo;.
In the cool mixing layer (T\(\sim\)2&times;10\(^4\) K) \(\kappa_{\text{mix}}/\kappa_{\text{Spitzer}}\!\approx\!10^3\!-\!10^7\),
because Spitzer \(\propto T^{5/2}\) <b>vanishes</b> there &mdash; so a scalar \(f_\kappa\!\cdot\!\kappa_{\text{Spitzer}}\)
<b>cannot</b> represent it. The faithful object is the structural \(\kappa{=}\max(\kappa_{\text{mix}},\kappa_{\text{Spitzer}})\)
term (Rung B), with the mixing diffusivity \(\lambda\delta v\) the single physical parameter, saturation-capped.
<span class="small">(Don&rsquo;t import El-Badry&rsquo;s \(\lambda\delta v{\in}[1,10]\) pc&middot;km/s &mdash; doubly
off-regime for TRINITY (discrete-SN + ISM density); take \(\delta v\) from the code&rsquo;s \(v_{\rm rel}\) and
<b>calibrate \(\lambda\) so resolved \(\theta\) matches Lancaster 0.9&ndash;0.99</b>, the cadence-free magnitude
anchor &mdash; verified in <code>KMIX_DIFFUSIVITY.md</code>.)</span></div>
<figure>__FIG_DERIV__<figcaption><b>Why it is \(\kappa_{\text{mix}}\), not a scalar.</b> Left: Spitzer (\(\propto T^{5/2}\))
vanishes in the cool mixing layer where the T-independent \(\kappa_{\text{mix}}\) rules &mdash; no single multiplier can
bridge that. Right: El-Badry&rsquo;s verified target is flat-high <i>even at diffuse</i> (0.94 at \(n{=}10^2\)) while
TRINITY&rsquo;s 1D baseline is only 0.29; the orange gap is what \(\kappa_{\text{mix}}\) must supply. From
<code>data/make_fkappa_physical_derivation.py</code>.</figcaption></figure>
<div class="box over"><span class="lab">Course-correction on &sect;15.6</span> Because the verified target is flat-high
even at diffuse, the diffuse never-fire corner is most likely a <b>1D under-cooling artifact</b>, not a true
energy-driven fate. So the resolution tilts to <b>route (b): add \(\kappa_{\text{mix}}\)</b> (Rung B,
<b>re-promoted</b> from &ldquo;optional bonus&rdquo;), rather than route (a) &ldquo;accept non-transition&rdquo;.
&sect;15.6&rsquo;s point still stands &mdash; don&rsquo;t crank \(f_\kappa\) to 64 &mdash; but the physical answer is the
structural \(\kappa_{\text{mix}}\), and the &ldquo;derived number&rdquo; is \(\lambda\delta v\), not an \(f_{\max}\).
<span class="small">Next: wire the gated \(\kappa_{\text{mix}}\) mode (RUNGB_SCOPING &sect;8), default-off byte-identical;
this also reconciles that doc&rsquo;s \(\kappa_{\text{mix}}/\kappa_S\!\approx\!10^{24}\) absurdity &mdash; it came from
\(D_{\text{turb}}{=}R_2 v_2\); El-Badry&rsquo;s \(\lambda\delta v\) is the sane magnitude.]</span></div>
<p class="small muted">All numbers in this section trace to <code>data/summary.csv</code> (the reduced 819-run table)
and the builders named above; see <code>F_KAPPA_FUNCTIONAL_FORM.md</code> &sect;0&ndash;&sect;13 for the full
treatment.</p>
"""


SEC_THETA5 = r"""
<h2 id="theta5">16 &middot; The theta5 matrix &mdash; the rule-compliant calibration lands (2026-07-02)</h2>

<div class="box warnbox"><span class="lab">Supersession notice</span> Chapters 13&ndash;15 quote numbers measured
<b>at blowout</b> on short runs &mdash; \(f_\kappa\)-to-fire \(\approx4/5\!-\!6/60\) (compact/mid/diffuse), and the
&ldquo;no constant fires the grid&rdquo; verdict. The 2026-07-01 standing rules retired that metric
(\(\theta=\theta_{\max}\) over \(\ge\)5 Myr, from <code>dictionary.jsonl</code> accepted rows), and this chapter&rsquo;s
matrix re-measured everything under them. <b>The \(\approx60\) is dead</b> (a blowout artifact), and the
&ldquo;no constant&rdquo; verdict inverts. Everything quotable is graded in <code>CONTAMINATION.md</code>.</div>

<div class="tldr"><p style="margin:0"><b>TL;DR.</b> The 📏 protocol matrix &mdash; the canonical <b>8 configs
&times; \(f_{\rm mix}\in\{1,2,4,8\}\) &times; 5 Myr</b> on the production <code>multiplier</code> knob &mdash; ran on
Helix, <b>32/32 rule-compliant</b>. Results: <b>(1)</b> blowout under-read the diffuse baseline <b>2&times;</b>
(\(\theta_0{=}0.535\), peaking at \(t\!\approx\!4.9\) Myr); <b>(2)</b> the firing boost collapses on the starting
deficit, \(f_{\rm fire}\approx1.4\,(0.95/\theta_0)^{1.82}\); <b>(3)</b> a <b>single</b> \(f_{\rm mix}{=}4\) fires the
whole normal-GMC band &mdash; including the diffuse cloud &mdash; at \(\theta_{\max}\) 0.96&ndash;1.04;
<b>(4)</b> route-a survives, de-conflated: <code>small_1e6</code> (same \(n_{\rm core}\) as the firing diffuse
config) never fires through \(f{=}8\), and <code>fail_repro</code> rides the PR#715 handoff untouched;
<b>(5)</b> two new failure modes: <i>fire-then-recollapse</i> (dense cores at \(f\!\ge\!4\)) and <i>over-boost
\(E_b\)-drain</i> (momentum without firing at \(f{=}8\)). <b>\(f_{\rm mix}{=}4\) was ADOPTED on 2026-07-02</b> &mdash; the maintainer ruled that firing into the
momentum phase and then recollapsing is acceptable physics (an outcome, not a failure mode).</p></div>

<figure>__FIG_T5ARMS__<figcaption><b>The matrix.</b> Emergent \(\theta_{\max}\) (5 Myr) vs \(f_{\rm mix}\), all 8
configs, colored by outcome class; filled markers fired <code>cooling_balance</code>. At \(f_{\rm mix}{=}4\) every
normal-GMC config sits at or above the trigger; the control and the PdV-dominated heavy cloud do not &mdash; by
physics, not by tuning. From <code>data/make_theta5_figures.py</code> (REPRODUCE #29).</figcaption></figure>

<h3>16.1 &middot; The measured calibration</h3>
<div class="tablewrap"><table>
<tr><th>config</th><th>\(n_{\rm core}\)</th><th>\(\theta_0\) (5 Myr)</th><th>\(f_{\rm fire}\) bracket</th><th>law predicts</th><th>\(\theta_{\max}\) @ \(f{=}4\)</th><th>fate @ \(f{=}4\)</th></tr>
<tr><td>simple_cluster</td><td>1e5</td><td>0.676</td><td>(1,2]</td><td>2.6</td><td>1.002</td><td>fires; recollapses (fires+survives at \(f{=}2\))</td></tr>
<tr><td>pl2_steep</td><td>1e5</td><td>0.511</td><td>(2,4]</td><td>4.3</td><td>0.975</td><td>fires; recollapses</td></tr>
<tr><td>midrange_pl0</td><td>1e4</td><td>0.636</td><td>(2,4]</td><td>2.9</td><td>0.981</td><td>fires; recollapses</td></tr>
<tr><td>be_sphere</td><td>1e4</td><td>0.529</td><td>(2,4]</td><td>4.0</td><td>1.039</td><td>fires; recollapses</td></tr>
<tr><td>large_diffuse_lowsfe</td><td>1e2</td><td>0.535</td><td>(2,4]</td><td>3.9</td><td>0.957</td><td><b>fires; survives to 5 Myr</b></td></tr>
<tr><td>small_1e6 (control)</td><td>1e2</td><td>0.297</td><td>&gt;8</td><td>11.6</td><td>0.682</td><td>route-a (healthy)</td></tr>
<tr><td>fail_repro (5e9)</td><td>1e2</td><td>0.003</td><td>n/a</td><td>&mdash;</td><td>0.013</td><td>PR#715 handoff, boost-invariant</td></tr>
<tr><td>small_dense_highsfe</td><td>1e6</td><td>0.717</td><td>n/a</td><td>2.3</td><td>NaN</td><td>solve never succeeded &mdash; NaN default, domain-edge root (&sect;16.7); \(E_b\!\le\!0\) handoff</td></tr>
</table></div>
<p class="sub">Full margins: <code>runs/data/theta5_fmix_scorecard.csv</code>; per-run record:
<code>runs/data/theta5_summary.csv</code> (stamped, 32 rows).</p>

<figure>__FIG_T5LAW__<figcaption><b>The \(\theta_1\)-collapse law is knob-specific.</b> The multiplier&rsquo;s
leverage (\(\theta\propto f^{0.55}\)) is twice the kappa knob&rsquo;s (0.27) &mdash; no structural back-reaction
eats the boost. All cloud-property dependence flows through the one scalar \(\theta_0\) TRINITY already
computes.</figcaption></figure>

<figure>__FIG_T5METRIC__<figcaption><b>Why the standing rule mattered.</b> Blowout-\(\theta\) under-reads every
config and by 2.1&times; exactly where it decided the story (diffuse, peak at \(t\!\approx\!4.9\) Myr). The
&ldquo;diffuse needs \(f\!\approx\!60\)&rdquo; claim of ch. 13&ndash;15 dies here.</figcaption></figure>

<figure>__FIG_T5TARGET__<figcaption><b>Calibrate, don&rsquo;t enforce.</b> The \(f_{\rm mix}{=}4\) boost lifts the
emergent \(\theta\) into the El-Badry/Lancaster band for GMCs; two configs at the same \(n_{\rm core}{=}10^2\)
behave oppositely &mdash; the fire/route-a boundary is set by \(\theta_0\) (mass, SFE, structure), not by density
alone.</figcaption></figure>

<figure>__FIG_T5KNOB__<figcaption><b>Why the multiplier knob &mdash; two panels, two DIFFERENT knobs and
x-axes.</b> Left: the structural \(f_\kappa\) (in-ODE) on one pre-fix 819-sweep cell &mdash; the apparent
"windows" were later re-diagnosed as solver crashes at the condensation boundary (&sect;9b) and the \(f_\kappa{=}16\)
"fire" as an artifact (rule-compliant it CONDENSES, &sect;12); kept here as the historical evidence that
motivated the knob choice. Right: the post-solve \(f_{\rm mix}\) multiplier (theta5) &mdash; \(\theta_{\max}\)
rises smoothly, no freezes. <b>Correction (&sect;16.3):</b> the multiplier's fire <i>set</i> is also
non-monotonic &mdash; a fire-vs-\(E_b\)-drain race &mdash; but the failure mode is a healthy handoff, not a
breakdown; and the definitive like-for-like comparison is the rule-compliant theta5k matrix
(&sect;16.4).</figcaption></figure>

<h3>16.2 &middot; Referee robustness &mdash; why 4, and why a constant?</h3>
<p><b>Why 4 and not 2.5 / 3.4 / 4.7?</b> Honestly: 4 is a <i>grid point</i>, not a fitted optimum. Its defense is
that the law-predicted requirement for the hardest firing configs lands almost exactly on it (pl2_steep 4.3,
be_sphere 4.0, large_diffuse 3.9) &mdash; i.e. 4 is the <b>minimal single constant that fires the band</b>, and any
\(f\) in the window \([\sim\!4,\ \lesssim\!8)\) gives the same fire set (the conclusions are insensitive within the
window; 8 hits the over-boost ceiling). The planned <b>theta5b fine bracket</b>
(\(f\in\{2.5,3,3.5,4.5,5\}\) &times; 8 configs + a \(t_{\rm stop}{=}8\) Myr diffuse arm) turns that from a law-based
argument into a measured sensitivity table: it pins each config&rsquo;s \(f_{\rm fire}\) to &plusmn;0.5, measures the
workable window edges, and tests whether the diffuse cloud fires at \(f{=}2\) given more time (it grazed
\(\theta{=}0.9552\) at exactly \(t{=}5\)).</p>
<p><b>Why a constant and not \(f(\text{cloud properties})\)?</b> Four independent reasons, one per row of the
model-comparison table in <code>PLAN.md</code>: <b>(i) physical sign</b> &mdash; the real enhancement
\(\kappa_{\rm mix}/\kappa_{\rm Spitzer}\propto n\) <i>rises</i> with density, opposite to the falling
\(f(n)\) a chase-the-target prescription needs, so no physical \(f(n)\) exists; <b>(ii) empirical</b> &mdash; the
819-sweep refuted \(f(n_H)\)-only (32&times; spread at fixed \(n\)), and the full 3-axis lookup
(<code>'auto'</code>) is measured to interpolate into kappa dead windows; <b>(iii) sufficiency</b> &mdash; the
\(\theta_1\)-collapse law shows all property-dependence flows through \(\theta_0\), which the <i>solved bubble
already supplies</i> &mdash; a constant \(f\) then lets the physics pick the fire set (route-a) instead of forcing
it; <b>(iv) falsifiability</b> &mdash; one parameter makes one testable prediction
(\(\theta_0>0.95/f^{0.55}\Rightarrow\) fires); a per-cloud \(f\) has as many parameters as calibration cells and
predicts nothing. The referee-grade test: fit the law on a config subset, <b>predict</b> the held-out
\(f_{\rm fire}\) brackets from \(\theta_0\) alone, and show the constant-\(f\)+law model matches the measurements
as well as any multi-parameter \(f(\text{properties})\) fit.</p>

<div class="box find"><span class="lab">Resolved &mdash; the pin (2026-07-02)</span> (1) the
<b>fire-then-recollapse physics call is CLEARED</b>: the maintainer ruled that firing into the momentum phase
and then recollapsing is acceptable physics &mdash; so <b>\(f_{\rm mix}{=}4\) is the adopted working value</b>
(production default stays <code>none</code>; 4 is the documented recommended setting). Still open: (2) the
theta5b fine bracket + long diffuse arm &mdash; now a <i>referee sensitivity refinement</i> (window edges), not a
gate; (3) the dense-edge (\(n_{\rm core}{=}10^6\)) NaN-loss diagnosis &mdash; <b>RESOLVED 2026-07-03
(&sect;16.7)</b>: the NaN is the never-written \(L_{\rm loss}\) default from a solve that never succeeds
(domain-edge root, machine-flippable); and the \(f{=}8\)
\(E_b\)-drain (momentum <i>without</i> firing) remains the real over-boost pathology.</div>

<h3>16.3 &middot; theta5b &mdash; the fine bracket answers the referee (2026-07-02)</h3>
<div class="tldr"><p style="margin:0"><b>TL;DR.</b> The 43-arm referee matrix ran. <b>(1) The whole-band window
is measured: \(f_{\rm mix}\in[4,4.5]\)</b> &mdash; 3.5 misses pl2_steep, 5 drops midrange_pl0 (over-boost
\(E_b\)-drain). \(f_{\rm mix}{=}4\) is confirmed, now as a <i>measured</i> choice. <b>(2) The collapse law holds
out-of-sample at rms 0.064 dex</b> over six configs &mdash; one parameter (\(\theta_0\)) predicts every fine
firing threshold. <b>(3) A new systematic &mdash; the fire-vs-drain race:</b> below threshold, extra boost often
<i>prevents</i> firing (the boosted \(E_b\) drain hands off to momentum before \(\theta\) crosses) &mdash;
simple_cluster fires at \(f{=}2\), not at 2.5&ndash;3, again at 3.5+. Healthy runs, real gaps; this corrects
&sect;16's &ldquo;no dead windows&rdquo; phrasing (no <i>solver</i> freezes remains true). <b>(4) The diffuse
cloud at \(f{=}2\) fires at \(t\!\approx\!5.04\) Myr</b> &mdash; the theta5 graze was real; the \(f{=}1\) 8-Myr
arm shows the native peak (\(t\!\approx\!4.86\)) IS captured by the 5 Myr window. <b>(5) The dense edge fires at
every fine arm</b> (\(\theta\) 0.95&ndash;1.01) &mdash; theta5's NaNs were later traced (&sect;16.7) to a
never-succeeding solve at a machine-flippable domain-edge root, not a wall.
Artifacts: <code>data/theta5_fire_map.csv</code>, <code>data/theta5_law_check.csv</code>,
<code>theta5b_fire_map.png</code>, <code>theta5b_law_check.png</code> (REPRODUCE #30).</p></div>
<figure>__FIG_T5BMAP__<figcaption><b>The fire map.</b> Outcome per (config, \(f_{\rm mix}\)) over both matrices:
filled = fires in the Lancaster band; triangles = momentum <i>without</i> firing (drain won); squares = healthy
energy-driven; the shaded strip is the measured whole-band window.</figcaption></figure>
<figure>__FIG_T5BLAW__<figcaption><b>Out-of-sample law check.</b> The theta5-fit
\(f_{\rm fire}=1.4\,(0.95/\theta_0)^{1.82}\) vs the theta5b fine measurements: rms 0.064 dex &mdash; the
one-parameter constant-\(f\) model predicts, a per-cloud \(f(\text{properties})\) fit could only chase.
</figcaption></figure>

<h3>16.4 &middot; theta5k &mdash; the kappa knob retried honestly, and the freeze solved (2026-07-03)</h3>
<p>The maintainer challenged &sect;16.2's "kappa breaks non-monotonically": <i>is that physics, or a bug we
mis-read?</i> It was a bug with a physical cause. The autopsy of the 819-run sweep + a live local reproduction
(<code>KAPPA_FREEZE_MECHANISM.md</code>) showed the "dead windows" were the solver <b>crashing at the
evaporation&rarr;condensation boundary of the conduction front</b> (McKee &amp; Cowie 1977): approaching cooling
balance, the structure solve's mass-flux eigenvalue physically goes negative (the hybr solver literally converges
to \(\dot M = -85\,M_\odot/\mathrm{Myr}\) at \(f_\kappa{=}8\)), the dMdt&gt;0 acceptance gate refuses it, and the
runner froze holding its last state &mdash; 34/38 frozen sweep runs died at \(\theta \geq 0.8\), i.e. <i>on approach to
the trigger</i>. The fix (landed 2026-07-03): a persistent no-root streak now <b>hands off to the momentum phase</b>
&mdash; the standard semi-analytic treatment of catastrophic-cooling onset (Silich/Tenorio-Tagle lineage) &mdash;
and theta5k, the first rule-compliant kappa matrix (56 arms, 5 Myr, dictionary \(\theta_{\max}\)), validates it
at scale: <b>zero freezes; every arm ends in a proper fate</b>.</p>
<figure>__FIG_T5KMAP__<figcaption><b>The rule-compliant kappa verdict.</b> The five condensation handoffs (diamonds)
land exactly on the old "dead window" cells &mdash; simple_cluster 8/12/16 (\(\theta\) held at 0.533/0.587/0.624;
&sect;8e's famous 0.53), dense 6, pl2 16 &mdash; and the old "fires at \(f_\kappa{=}16\)" on simple_cluster is exposed
as a pre-fix artifact. The fire set is <i>still</i> non-monotonic, but now for an honest physical reason (the front
condenses, or the shell drains/dissolves, before global \(\theta\) crosses). The headline: <b>no single
\(f_\kappa\) fires the whole band</b> (best: \(f_\kappa{=}12\) at 5/6) &mdash; the multiplier's measured
[4,&nbsp;4.5] window (6/6) is now the production-knob argument on crash-free, like-for-like data.</figcaption></figure>
<figure>__FIG_T5KRISE__<figcaption><b>Reach vs race.</b> \(\theta_{\max}\) rises with \(f_\kappa\) essentially
monotonically everywhere it can be measured; where the fire set has holes, an open marker shows the run ended
(condensation handoff / drain) before \(\theta\) could cross &mdash; the knob's <i>reach</i> is fine, the
<i>race</i> is what it keeps losing. Fired-arm \(\theta_{\max}\) magnitudes above ~1.2 (dense: 1.99) are
structural-boost distortion &mdash; quote fire/no-fire from this matrix, not those values.</figcaption></figure>

<h3>16.5 &middot; The dMdt dip, start to finish &mdash; the story of one bug that turned out to be physics (2026-07-03)</h3>
<p>This subsection collects the whole arc in one place, because it is the cleanest single narrative in the
workstream: <i>a symptom that looked like solver chaos, a diagnosis that found a physical boundary, a fix taken
from the literature's own playbook, and a verdict that settled the knob choice.</i></p>
<p><b>Act 1 &mdash; the symptom.</b> Sweep runs at certain \(f_\kappa\) "froze": they ended mid&ndash;energy-phase at
\(t\approx0.4\) Myr with \(\theta\) stuck near 0.53 and exit code 0, and the fire set looked non-monotonic
("dead windows", &sect;16.2). The maintainer challenged the reading: <i>too much cooling should give a
monotonic-ish trend &mdash; is this a bug? a stale state?</i></p>
<p><b>Act 2 &mdash; the diagnosis.</b> The autopsy of the 819-run sweep showed 34/38 freezes died at
\(\theta\geq0.8\) &mdash; <i>on approach to the trigger</i> &mdash; and one froze at \(f_\kappa{=}1\) (unboosted:
the mode pre-exists the knob). A live instrumented run then caught the mechanism red-handed: the bubble-structure
solve's mass-flux eigenvalue <b>converged to a negative value</b> (\(\dot M_b=-85\,M_\odot\)/Myr) and the
dMdt&gt;0 acceptance gate refused it, leaving the runner to grind frozen state to its segment cap. The figure below
shows the decisive controlled pair: identical clouds, identical early timesteps, \(f_\kappa=6\) vs 8.</p>
<figure>__FIG_DMDTDIP__<figcaption><b>The dMdt dip.</b> Per-segment eigenvalue of the \(\beta\)&ndash;\(\delta\)
structure solve. Both arms accept an evaporative solve at birth (+320, off scale), then dip into the
<i>condensation branch</i> (shaded) that the model cannot represent &mdash; the closure \(T\propto\dot M^{2/5}\)
(Weaver Eq.&nbsp;44) simply has no \(\dot M&lt;0\) profile family. The arcs are smooth (solver exonerated &mdash;
consistent with the planar-analogue uniqueness of Tan, Oh &amp; Gronke 2021); the two fates differ only in whether
the front's budget <i>recovers to evaporation in time</i>: \(f_\kappa{=}8\) climbs back through zero at segment 28
and fires; \(f_\kappa{=}6\) nearly recovers (&minus;4.0), second-dives, and is handed off. The photo-finish is
physical; its exact \(f_\kappa\) edge is discretization-sensitive (one jump per trace correlates with segment-loop
events), so per-config \(f_{\kappa,\rm fire}\) values are razor-edge quantities.</figcaption></figure>
<p><b>Act 3 &mdash; the physics.</b> The reversal is textbook: when radiative losses in the conduction front exceed
the conductive heat supply, evaporation stops and the hot gas condenses onto the shell (McKee &amp; Cowie 1977).
Weaver's own Paper II already put the classical front budget at 60/40 (40% of the conductive flux radiated in the
interface) &mdash; the boundary was always nearby, and cooling balance, the very condition the transition trigger
waits for, <i>is</i> the reversal condition. A cooling-boosted bubble must pass through it.</p>
<p><b>Act 4 &mdash; the fix, from the literature's own playbook.</b> Codes that "just follow the sign" are
Lagrangian hydro (El-Badry+19; Vieser &amp; Hensler 07), where the interface flux is an emergent PDE outcome &mdash;
adopting that means a new profile family, a research project. Saturated conduction (Cowie &amp; McKee 77) caps the
runaway but does not remove the reversal. Semi-analytic models in TRINITY's own lineage (Weaver; Mac Low &amp;
McCray; Silich &amp; Tenorio-Tagle's catastrophic cooling) never integrate through the boundary &mdash; they
<b>switch regimes</b>. That is what shipped: a persistent no-physical-root streak now ends the energy phase as
<code>no_physical_root_handoff</code> (a <i>fate</i>, routed like the cooling-balance exit, never counted as a
\(\theta\) transition).</p>
<figure>__FIG_DMDTFLOW__<figcaption><b>The resolution at a glance.</b> Symptom &rarr; diagnosis &rarr; physics
identity &rarr; the three literature treatments &rarr; the adopted regime switch, validated by theta5k: 56/56
proper fates, zero freezes, the old "dead windows" reborn as labeled condensation fates &mdash; and still no
whole-band \(f_\kappa\) (best 5/6), which is what finally settles the production choice on the
\(f_{\rm mix}\) multiplier (window [4,&nbsp;4.5] fires 6/6, backed by the \(\theta_1\)-collapse law at
0.064 dex out-of-sample). The multiplier dodges the boundary <i>by construction</i>: it scales the cooling
luminosity after the structure solve and never touches the \(\dot M_b\) eigenvalue.</figcaption></figure>
<p><b>Epilogue &mdash; what we ended up going with.</b> Production: <code>cooling_boost_mode=multiplier</code>,
\(f_{\rm mix}=4\), cooling_balance trigger &mdash; every GMC-band cloud enters the momentum phase at
\(\theta\geq0.95\), with recollapse-after-transition accepted physics. The kappa knob survives as an honest
structural probe (its runs now end in proper fates), the condensation branch is documented as the model's domain
edge with a research-grade upgrade path, and the whole chain &mdash; autopsy CSV, trace CSV, freeze-watch
instrumentation, fix, and the 56-run validation &mdash; is committed and reproducible without re-running anything.</p>

<h3>16.6 &middot; theta5n &mdash; the maintainer's ninth config fires <i>natively</i>, and the law calls it (2026-07-03)</h3>
<p>The maintainer added a "normal cloud" to the standard band: \(M_{\rm cloud}=10^6\,M_\odot\),
\(n_{\rm core}=10^3\,{\rm cm^{-3}}\), \(\varepsilon_\star=0.01\) (so \(M_\star=10^4\,M_\odot\) &mdash;
the weakest driver in the band), flat profile. Fifteen arms ran (both knobs, standard rules). The headline:
<b>the unboosted arm fires on its own</b> &mdash; \(\theta\) crosses 0.95 at \(t\approx2.5\) Myr with
<code>cooling_boost_mode=none</code> (\(\theta_0=1.047\)). This is the route-a picture live: a normal
weak-feedback GMC transitions to momentum through the cooling-balance trigger with <i>no help</i>; the
\(f_{\rm mix}\) boost exists for the stronger-feedback/denser corners of the band, and \(f_{\rm mix}=4\)
does not break the natural transitioner (every multiplier arm 2&ndash;8 fires; boost simply moves the crossing
earlier, from 2.5 Myr at \(f{=}1\) to 0.3 Myr at \(f{=}8\); all arms then recollapse &mdash; accepted physics).
The \(\theta_1\)-collapse law scores its <b>seventh out-of-sample point at the opposite extreme</b>: predicted
\(f_{\rm fire}=1.4\,(0.95/1.047)^{1.82}=1.16\), measured 1.0 &mdash; residual 0.065 dex, exactly at the law's
rms (0.064 dex, unchanged with the new point folded in). The kappa knob shows one more race loss at the top
(\(f_\kappa{=}16\) DRAINs at \(\theta_{\max}=0.916\), fires 2&ndash;12) &mdash; the same non-monotonicity
story as &sect;16.4. Updated fire maps and law-check figure above include the ninth row.</p>

<h3>16.7 &middot; The dense-edge NaN cross, closed &mdash; a never-written default at a machine-flippable root (2026-07-03)</h3>
<p>The last open ticket. The fire map's NaN crosses (small_dense at \(f_{\rm mix}{=}4,8\)) turned out to be
bookkeeping honesty, not physics: <code>bubble_Lloss</code> defaults to NaN in the parameter registry, and in
those HPC arms the \(\beta\)&ndash;\(\delta\) solve <b>never succeeded once</b> (log: <i>structure solve
failed</i> at wandering \((\beta,\delta)\) like \(\beta{=}-0.04\)) &mdash; every snapshot faithfully wrote
the untouched default (\(\theta_{\rm first}={\rm nan}\) from row 1 is the tell). Mechanism: the boost enters
the energy-balance residual (\(L_{\rm loss}=L_{\rm leak}+f\,L_{\rm cool}\)), so raising \(f\) displaces the
root; on the band's most extreme config (\(n_{\rm core}{=}10^6\)) \(f\approx4\)&ndash;\(8\) lands it on the
edge of the integrable domain. DEBUG reproduction proved the edge is <b>machine-flippable</b>: the identical
params fire locally (mult4 recovers after one failed segment; mult8 after nine, \(L_{\rm loss}/L_{\rm mech}=1.01\))
while Helix never recovered &mdash; opposite \(\theta\) bookkeeping, same dynamical fate and collapse time
(the \(E_b\) ODE runs off the \(\beta\)-side \(\dot E\)). Rules of use (CONTAMINATION): an all-NaN arm is
never a physics outcome &mdash; quote the finite-\(\theta\) neighbors (dense's calibration evidence is the fine
arms 3.5/4.5/5, all fired); post&ndash;fix-#1 code turns any persistent solve-fail streak into the labeled
condensation handoff instead of a NaN grind. FINDINGS &sect;14.</p>
"""

SEC_SHIPPED = r"""
<h2 id="shipped">The shipped model &mdash; what \(f_{\rm mix}\) is, the equation it acts on, and how to run it</h2>
<p>Mirroring the \(f_\kappa\) definition section (&sect;13), here is the same treatment for the knob that
actually ships. The bubble-structure solve is <b>untouched</b>: at every implicit-phase segment the
\(\beta\)&ndash;\(\delta\) solver computes the Weaver-type structure and from it the bubble's intrinsic cooling
luminosity \(L_{\rm cool}\) (the \(n_e n_i\,\Lambda(T)\) integral over the bubble + conduction zone) and the
leakage term \(L_{\rm leak}\). The mixing boost is applied <i>after</i> the solve, as a scalar on the loss
budget (<code>get_betadelta.py::effective_Lloss</code>):</p>
<p style="text-align:center; font-size:1.15em">
\( L_{\rm loss} \;=\; L_{\rm leak} \;+\; f_{\rm mix}\, L_{\rm cool} \)
</p>
<p>and the energy phase ends &mdash; the run hands off to the momentum phase &mdash; when the cooling-balance
trigger fires (<code>run_energy_implicit_phase.py</code>):</p>
<p style="text-align:center; font-size:1.15em">
\( \dfrac{L_{\rm gain}-L_{\rm loss}}{L_{\rm gain}} \,&lt;\, 0.05
\;\;\Longleftrightarrow\;\;
\theta \equiv \dfrac{L_{\rm loss}}{L_{\rm gain}} \,\geq\, 0.95 \)
</p>
<p><b>What \(f_{\rm mix}\) stands for physically:</b> the 1D solve feeds the hot/cold interface with Spitzer
conduction only; real interfaces are turbulent fractal mixing layers with far more radiating surface
(Lancaster+21) &mdash; equivalently a turbulent diffusivity \(\lambda\,\delta v\) on top of Spitzer (El-Badry+19).
\(f_{\rm mix}\) is that unresolved enhancement as one constant. Because it multiplies \(L_{\rm cool}\)
<i>after</i> the structure solve, it never touches the \(\dot M_b\) eigenvalue &mdash; it is structurally immune
to the condensation-boundary problem of &sect;16.5 by construction.</p>
<p><b>The adopted value and its defense, all measured:</b> \(f_{\rm mix}=4\), the interior-bottom of the
measured whole-band window <b>[4, 4.5]</b> (theta5b; 4.5 works, 5 already drops midrange to an \(E_b\) drain;
2.5&ndash;3.5 miss part of the band). Per-cloud behavior collapses onto the one-parameter law
\( f_{\rm fire} = 1.4\,(0.95/\theta_0)^{1.82} \), validated out-of-sample at <b>0.064 dex rms over seven
configs</b> spanning \(\theta_0=0.51\) to \(1.05\) &mdash; including the ninth config that fires natively at
\(f{=}1\) (&sect;16.6). All cloud-property dependence flows through \(\theta_0\) (residuals show no
\(n\), \(M\), or SFE trend; the \(\theta_0\)-matched trio bounds \(|\partial\log f/\partial\log n|\lesssim0.02\)).
Momentum-then-recollapse is accepted physics (maintainer ruling 2026-07-02).</p>
<div class="box" style="border-left:4px solid #2a8aa8"><b>To run it, add to any <code>.param</code>:</b>
<pre>cooling_boost_mode     multiplier
cooling_boost_fmix     4</pre>
Everything else is already the default: <code>transition_trigger&nbsp;=&nbsp;cooling_balance</code> and
<code>phaseSwitch_LlossLgain&nbsp;=&nbsp;0.05</code> (i.e. the \(\theta\geq0.95\) trigger) ship as defaults, and
<code>cooling_boost_mode&nbsp;=&nbsp;none</code> is the byte-identical off state, so an unmodified run is
untouched. Worked examples: <code>runs/params/theta5b/*__mult4.param</code> (any config), or the ninth-config
set <code>runs/params/theta5n/</code>. The knob taxonomy stays: <code>cooling_boost_kappa</code> is a structural
probe (proper fates post fix #1, but no whole-band value exists and its fire edges are razor-thin);
<code>theta_target</code> and <code>'auto'</code> are demoted/opt-in.</div>
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
<tr><td><code>F_KAPPA_FUNCTIONAL_FORM.md</code> (&sect;0&ndash;&sect;10)</td><td>&sect;15 the full \(f_\kappa(n_H)\) treatment: the composed form, the El-Badry PDF verification, the sweep scorecard, the cliff/fan-out anatomy, and the metric assessment &mdash; the source doc this section condenses</td></tr>
<tr><td><code>fkappa_functional_form.png</code> (+ <code>data/make_fkappa_functional_form.py</code>)</td><td>&sect;15.1 the composed form (Lancaster target + measured baseline + raw-power leverage); reads <code>fmix_table</code>/<code>kappa_blowout_calibration</code>/<code>kappa_calibration_estimate</code> CSVs, no sims</td></tr>
<tr><td><code>data/summary.csv</code> (+ <code>data/reduce_fkappa_sweep.py</code>, on HPC)</td><td>&sect;15.3 the reduced 819-run table (one row per run: \(\theta_{\text{blowout}}\), \(\theta_{\max}\), <code>cooling_fired</code>, axes) &mdash; the multi-GB jsonl stays on the cluster</td></tr>
<tr><td><code>data/fkappa_nH_sweep.csv</code> + <code>fkappa_nH_sweep.png</code> (+ <code>data/make_fkappa_nH_sweep.py</code>)</td><td>&sect;15.3 the per-cell \(\theta{\sim}f_\kappa^{p}\) fit + the 3-panel faceted de-conflation figure (fan-out; 1e7 cliffs; never-fire triangles)</td></tr>
<tr><td><code>data/fkappa_sweep_scorecard.csv</code> + <code>fkappa_sweep_analysis.png</code> (+ <code>data/make_fkappa_sweep_analysis.py</code>)</td><td>&sect;15.3 the pre-registered prediction scorecard (slope \(n^{-0.60}\) vs predicted \(n^{-0.30}\); fan-out; never-fire; \(M_\star\)-independence) + predicted-vs-measured</td></tr>
<tr><td><code>data/fkappa_cliff_metric.csv</code> + <code>fkappa_cliff_metric.png</code> (+ <code>data/make_fkappa_cliff_metric.py</code>)</td><td>&sect;15.4&ndash;15.5 the catastrophic-cooling cliff (constant-column threshold), the multi-dimensional fan-out (\(R^2\) of \(n_{\text{core}}\)/column/2-var, \(M_\star\)-independence), and the metric sanity (\(\theta_{\max}{-}\theta_{\text{blowout}}\) median 0.004)</td></tr>
<tr><td><code>data/fkappa_physical_cap.csv</code> + <code>fkappa_physical_cap.png</code> (+ <code>data/make_fkappa_physical_cap.py</code>)</td><td>&sect;15.6 the physical-cap reframing: the sign flip (physical \(f_\kappa\!\propto\!n^{+1}\) vs fire-threshold \(n^{-0.6}\)), the momentum/energy split per \(f_{\max}\), and the falsifiable critical column \(N_{\text{crit}}\!\approx\!1\!-\!4{\times}10^{23}\)</td></tr>
<tr><td><code>data/fkappa_physical_derivation.csv</code> + <code>fkappa_physical_derivation.png</code> (+ <code>data/make_fkappa_physical_derivation.py</code>)</td><td>&sect;15.7 the derived physical prescription: the three \(f_\kappa(n)\) (mechanism/target/boost), the \(\kappa_{\text{mix}}{=}\kappa_{\text{Spitzer}}\) crossover \(n_{\text{crit}}{=}0.25\), why a scalar \(f_\kappa\) can't represent cool-layer mixing, and the \(\theta^\star\) gap arguing for \(\kappa_{\text{mix}}\) (Rung B)</td></tr>
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
        HEAD,
        HERO,
        SEC_SETUP,
        SEC_DOUBLE,
        SEC_CONVENTION,
        SEC_REGIME,
        SEC_CLOSURE,
        SEC_DA,
        SEC_LIT,
        SEC_WIRING,
        SEC_LIVE,
        SEC_BOTTOM,
        SEC_KAPPA,
        SEC_EBPEAK,
        SEC_FKDEF,
        SEC_TAXONOMY,
        SEC_SWEEP,
        SEC_THETA5,
        SEC_SHIPPED,
        SEC_REPRO,
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
