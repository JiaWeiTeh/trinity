#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pdvtrigger_report.py — build the self-contained HTML report for the
"PdV in the transition trigger" -> unresolved-interface-cooling-closure
workstream (docs/dev/transition/pdv-trigger/).

This is the sequel report to the s1 clean-room investigation: that one ended on
"the live question is the geometric / Eb-peak handoff and the missing
mixing-layer cooling." This report answers the missing-mixing-layer half —
boost the LOSS, not the trigger, without double-counting.

The 4 figures live in storyline_figs/ and are embedded as base64 so the file is
standalone (downloadable, opens offline; MathJax loads from CDN for the
formulas). The figures are NOT regenerated here — they are committed PNGs built
by storyline_figs/make_storyline_figs.py from the data/*.csv screens. Every
number in the prose traces to a committed CSV in data/ (verified 2026-06-24).

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

# 4 committed PNGs (do NOT regenerate). token -> (filename, alt text).
FIGURES = {
    "__FIG_FMIX__": (
        "fig_fmix_convention.png",
        "f_mix needed to fire the handoff at blowout, with-PdV vs the consistent no-PdV trigger, per config",
    ),
    "__FIG_DOUBLE__": (
        "fig_doublecount.png",
        "the single-count line vs the forbidden double-count region, with the max-closure and the MC draws",
    ),
    "__FIG_REGIME__": (
        "fig_regime_split.png",
        "PdV/Lmech per config: normal clouds sub-critical near 0.45, the heavy 5e9 super-critical above 1",
    ),
    "__FIG_HEATMAP__": (
        "fig_closure_heatmap.png",
        "frozen-trajectory fire-vs-blowout heatmap across the density grid: no single f_mix fires them all at blowout",
    ),
}


def img(name, alt):
    b64 = base64.b64encode((FIGS / name).read_bytes()).decode()
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
<h1>Interface cooling without double-counting &mdash; boost the loss, not the trigger</h1>
<p class="sub">The sequel to the clean-room transition study. That investigation ended on a single open
question &mdash; the geometric / \(E_b\)-peak handoff and the <i>missing mixing-layer cooling</i>. This
report answers the cooling half: what the maintainer&rsquo;s Paper-II note (&ldquo;adding unresolved interface
cooling to TRINITY without double-counting&rdquo;) actually buys, where it fires, and why a single constant
is not enough. Verified 2026-06-24; numbers trace to committed CSVs under
<code>docs/dev/transition/pdv-trigger/data/</code>; the wiring shipped opt-in and gated.</p>

<div class="tldr">
<p style="margin:0"><b>TL;DR.</b> s1 proved normal clouds never trip the radiative cooling trigger
\((L_{\text{mech}}-L_{\text{cool}})/L_{\text{mech}}<0.05\) &mdash; they hand off by <b>geometric blowout</b>, and
the live gap was the unresolved turbulent-mixing-layer cooling a 1D model can&rsquo;t see. The fix is to
<b>add the missing loss</b>, not to tune the trigger. Three results follow. <b>(1) Don&rsquo;t double-count:</b>
the missing cooling must <i>replace</i> the explicit \(L_{\text{cool}}\), never stack on top of it &mdash; the
<code>theta_target</code> closure \(L_{\text{loss}}^{\text{eff}}=\max(L_{\text{cool}}+L_{\text{leak}},\;\theta\,L_{\text{mech}})\)
is single-count by construction (a \(5\times10^5\)-draw Monte-Carlo finds <b>0</b> draws in the double-count
region). <b>(2) The headline boost was mis-stated by a trigger-convention slip:</b> the note&rsquo;s
\(f_{\text{mix}}\!\approx\!1.1\!-\!1.5\) put PdV <i>inside</i> the screening trigger &mdash; inconsistent with the
note&rsquo;s own PdV-<i>out</i> recommendation. Under the consistent (PdV-out) trigger the boost is
<b>\(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)</b>, which also matches the mixing-layer literature target.
<b>(3) A constant knob is not enough:</b> a frozen-trajectory screen shows the firing \(f_{\text{mix}}\) spans
\(\sim\!1.4\!\to\!3.8\) across the density grid &mdash; pointing to a state-dependent \(\theta_{\text{target}}(\mathrm{Da})\),
not a constant. Normal clouds want a cooling boost; the heavy 5e9 cloud is super-critical and hands off
via the PdV / \(E_b\)-peak turnover instead. The wiring is in, gated, byte-identical when off.</p>
</div>
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
interface cooling that the literature says dominates real wind bubbles. This report is that fix.</p>
<div class="box hyp"><div class="lab">the move</div>The handoff isn&rsquo;t mis-tuned &mdash; it tests for an
event (radiative balance) that the under-cooled 1D bubble never reaches. So don&rsquo;t retune the threshold;
<b>add the missing cooling</b> and let the existing trigger fire on its own. The whole report is about doing that
<i>correctly</i>: which channel to boost, by how much, and whether one constant can do it.</p></div>
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
<h2 id="convention">3 &middot; The key finding &mdash; the trigger-convention fix</h2>
<p>This is the headline. The note&rsquo;s reported boost, \(f_{\text{mix}}\!\approx\!1.1\!-\!1.5\), was computed
with the <b>PdV term inside the screening trigger</b> &mdash; i.e. solving
\((L_{\text{mech}}-f\,L_{\text{cool}}-\text{PdV})/L_{\text{mech}}=0.05\). But that is <b>inconsistent with the
note&rsquo;s own recommended trigger</b>, which keeps PdV <i>out</i>. The physics reason is clean: <b>PdV is
reversible</b> &mdash; the work done on the shell is recoverable as shell momentum &mdash; whereas <b>cooling is
irreversible</b>. You fire the energy&rarr;momentum handoff on the irreversible channel, so PdV belongs in the
ODE only, never in the trigger.</p>
<p>Restore consistency &mdash; drop PdV from the screening ratio &mdash; and the required boost becomes
\[ f_{\text{mix}} = \frac{0.95}{L_{\text{cool}}/L_{\text{mech}}}\Bigg|_{\text{blowout}}, \]
i.e. lift the resolved \(L_{\text{cool}}/L_{\text{mech}}\) up to the measured \(\theta\approx0.95\) at the blowout
epoch. That gives <b>\(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)</b> across the compact-to-mid clouds &mdash; larger
than the note&rsquo;s with-PdV column by exactly the \(\text{PdV}/L_{\text{mech}}\) offset it folded in. Crucially,
this consistent value <b>also matches the literature target</b>: lift a resolved \(L_{\text{cool}}/L_{\text{mech}}\approx0.25\!-\!0.7\)
up to the \(\theta\approx0.95\) that 3D simulations of fractal mixing layers measure.</p>
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
(consistent \(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)): it lifts their radiative ratio to fire the existing
trigger near blowout. The heavy cloud is the opposite &mdash; it cools so little
(\(L_{\text{cool}}/L_{\text{mech}}\approx0.01\)) that it <b>never</b> fires the cooling trigger even at
\(f_{\text{mix}}=30\); it is PdV-dominated, so it hands off via the <b>PdV / \(E_b\)-peak net-energy turnover</b>
(\(L_{\text{mech}}-L_{\text{cool}}-\text{PdV}\le0\)) instead. Cooling and PdV cover complementary halves of the
mass range.</p>
<div class="box find"><div class="lab">the split, in one line</div>Normal clouds (sub-critical,
\(\text{PdV}/L_{\text{mech}}\approx0.45\)) &rarr; <b>cooling boost</b>. Heavy clouds (super-critical,
\(\text{PdV}/L_{\text{mech}}>1\)) &rarr; <b>PdV / \(E_b\)-peak handoff</b>. The same diagnostic that says
&ldquo;PdV is negligible&rdquo; is false everywhere also says PdV is <i>decisive</i> only past the unity
crossing.</div>
"""

SEC_CLOSURE = r"""
<h2 id="closure">5 &middot; A constant knob is not enough</h2>
<p>If one constant \(f_{\text{mix}}\) fired every config at blowout, a single calibrated float would close the
problem. It does not. A <b>frozen-trajectory screen</b> (<code>data/closure_test.csv</code>) solves, on each
committed unboosted trajectory, for the \(f_{\text{mix}}\) that fires that config at its blowout epoch. The
answer spans <b>\(\sim\!1.4\to3.8\)</b> across the density grid: dense clouds already cool efficiently
(\(L_{\text{cool}}/L_{\text{mech}}\approx0.7\) at blowout, so they need only a small lift), diffuse clouds barely
cool (\(\approx0.25\), so they need a big one). No single constant fires them all at the right epoch.</p>
<figure>__FIG_HEATMAP__<figcaption>The frozen-trajectory fire-vs-blowout heatmap (config &times; boost value). At a
constant \(f_{\text{mix}}\!\approx\!2\), the compact/dense clouds fire right at blowout while the diffuse ones fire
well before it (offsets of order \(-1\) to \(-3.6\) Myr). The density ordering is monotone &mdash; exactly the
signature of a state-dependent boost. Read of <code>data/closure_test.csv</code>.</figcaption></figure>
<div class="box over"><div class="lab">everywhere: this is a FROZEN-TRAJECTORY SCREEN, not a forecast</div>
The screen freezes the <i>unboosted</i> trajectory and asks where a boost would fire on it. But boosting cooling
lowers \(P_b\!\to\!\) lowers \(\text{PdV}\,(\propto P_b)\!\to\!\) changes \(E_b(t),R_2(t),v_2(t)\) &mdash; it
<b>moves blowout itself</b>. So these fire-times <b>bound</b> the knob; they do not <b>forecast</b> it. The honest
test is a full boosted run at matched \(t\) (see &sect;7&ndash;8). Read every number in this section as a screen.</div>
<p>The density-ordered spread is the argument for coupling. The mixing-layer luminosity is not a constant: it
scales with the contact-discontinuity area (\(\propto R_2^2\)), the shear / turbulent velocity (\(\propto v_2\) or
the hot-gas sound speed), and the mixing-layer cooling function. The natural closure is a
<b>state-dependent \(\theta_{\text{target}}(\mathrm{Da})\)</b> &mdash; a Damk&ouml;hler number
\(\mathrm{Da}=\tau_{\text{turb}}/t_{\text{cool}}\) (Tan/Oh/Gronke) &mdash; or the fuller \(\kappa_{\text{eff}}\)
coupling \(\kappa_{\text{eff}}=\max(\kappa_{\text{Spitzer}},\,\kappa_{\text{mix}})\),
\(\kappa_{\text{mix}}\sim\rho\,c_p\,D_{\text{turb}}\), \(D_{\text{turb}}\sim\lambda\,\delta v\sim R_2 v_2\). A
constant is a calibration probe that points here; it is marked as such in code with a <code>ponytail:</code>
comment naming the ceiling.</p>
"""

SEC_LIT = r"""
<h2 id="literature">6 &middot; Literature anchor</h2>
<p>The physics content was checked against the mixing-layer / interface-cooling literature (web-verified this
session). All of it holds up:</p>
<ul>
<li><b>El-Badry+19</b> &mdash; interface cooling in wind bubbles; conductive evaporation suppressed \(3\!-\!30\times\)
relative to the classic Weaver solution; \(\theta/(1-\theta)\propto(\rho\,\lambda\,\delta v)^{1/2}\).</li>
<li><b>Lancaster+21b</b> &mdash; near-complete cooling of wind energy, \((1-\theta)\sim0.1\!-\!0.01\); fractal
contact-discontinuity area, fractal dimension \(D\sim2.5\!-\!2.7\); momentum enhancement \(\alpha_p\sim1.2\!-\!4\).</li>
<li><b>Lancaster+25</b> &mdash; the \((1-\theta)\,L_{\text{mech}}\) reduced model (the input-rescale this report is
careful <i>not</i> to stack on top of the explicit \(L_{\text{cool}}\)).</li>
<li><b>Pittard+22</b> &mdash; wind-blown bubbles radiate up to \(\sim\!98\%\) of the injected energy.</li>
<li><b>Fielding+20</b> &mdash; turbulent radiative mixing layers; surface-area / cooling scaling exponent
\(d\approx0.5\).</li>
<li><b>Tan/Oh/Gronke+21</b> &mdash; the Damk&ouml;hler number \(\mathrm{Da}=\tau_{\text{turb}}/t_{\text{cool}}\)
that the coupled \(\theta_{\text{target}}(\mathrm{Da})\) endgame is built on.</li>
</ul>
<div class="box over"><div class="lab">verify against the PDFs</div>The qualitative physics all checks out. The
exact equation / figure <b>numbers</b> &mdash; El-Badry Eq.&nbsp;35 / Fig.&nbsp;7, Lancaster+25 Eq.&nbsp;39 &mdash;
could not be confirmed this session (PDF hosts were blocked). Treat those specific references as
<b>&ldquo;verify against the PDFs&rdquo;</b> before quoting them in the paper.</div>
"""

SEC_WIRING = r"""
<h2 id="wiring">7 &middot; The wiring &mdash; shipped, opt-in, gated</h2>
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
(<code>test_cooling_boost.py</code> + <code>test_r1_shadow.py</code>); ruff F-rules clean.</div>
"""

SEC_BOTTOM = r"""
<h2 id="bottom">8 &middot; Bottom line &amp; what&rsquo;s next</h2>
<p>The transition trigger is not mis-tuned; the 1D bubble under-cools. The principled correction is to add the
missing interface cooling &mdash; once &mdash; and let the existing trigger fire on its own:</p>
<ul>
<li><b>Normal clouds</b> (sub-critical, \(\text{PdV}/L_{\text{mech}}\approx0.45\)) &rarr; a <b>cooling boost</b>,
consistent \(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\) (PdV-out trigger), ideally the coupled
\(\theta_{\text{target}}(\mathrm{Da})\) rather than a constant.</li>
<li><b>Heavy clouds</b> (super-critical, \(\text{PdV}/L_{\text{mech}}>1\)) &rarr; the <b>PdV / \(E_b\)-peak
handoff</b>; cooling never fires for them.</li>
<li><b>Never double-count</b> &mdash; the boosted loss <i>replaces</i> the explicit \(L_{\text{cool}}\) via the
\(\max(\cdot)\) closure; it never stacks.</li>
</ul>
<div class="box hyp"><div class="lab">the next rung</div>The wiring is in and safe, but every firing-value in
this report is a <b>frozen-trajectory screen</b> &mdash; it bounds the knob, it does not forecast it (boosting
cooling moves blowout itself). The next step is <b>matched-\(t\) edge-config live runs</b> (boosted vs unboosted,
separate processes: <code>simple_cluster</code> + <code>f1edge_{lowdens,hidens}</code> + a 5e9) that replace the
screen and settle constant-\(f_{\text{mix}}\) vs coupled \(\theta_{\text{target}}(\mathrm{Da})\).</div>
"""

SEC_REPRO = r"""
<h2 id="repro">Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/transition/pdv-trigger/</code> &mdash; reproducible
without re-running the (hours-long) sims. Each figure is a pure read of a committed CSV; the prose numbers trace to
the same files.</p>
<div class="tablewrap"><table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>data/fmix_table.csv</code> (+ <code>make_fmix_table.py</code>)</td><td>the note&rsquo;s Table 2, both trigger conventions; the &sect;3 headline \(f_{\text{mix}}\!\approx\!1.4\!-\!2.8\)</td></tr>
<tr><td><code>data/doublecount_mc.csv</code> (+ <code>make_doublecount_mc.py</code>)</td><td>the \(5\times10^5\)-draw Monte-Carlo backing the &sect;2 single-count claim (0 draws in the \(2\theta\) region)</td></tr>
<tr><td><code>data/pdv_regime_budget.csv</code> (+ <code>make_pdv_regime_table.py</code>)</td><td>per-config \(\text{PdV}/L_{\text{mech}}\), \(E_b\) growth, sub-/super-critical regime (&sect;4)</td></tr>
<tr><td><code>data/closure_test.csv</code> (+ <code>make_closure_test.py</code>)</td><td>the 8-config staged frozen-trajectory screen behind the &sect;5 heatmap</td></tr>
<tr><td><code>storyline_figs/*.png</code> (+ <code>make_storyline_figs.py</code>)</td><td>the four figures above, each a pure read of the CSVs</td></tr>
<tr><td><code>PLAN.md</code></td><td>the living plan / pre-registration, the re-entry ledger, and the full code map</td></tr>
</tbody></table></div>
<p class="small muted">Rebuild this report:
<code>python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py</code> (figures are committed PNGs, not
regenerated here). Then rebuild the storyline book:
<code>python docs/dev/html-insights/build_storylines.py</code>.</p>
<footer>TRINITY PdV-in-the-trigger &rarr; unresolved-interface-cooling closure &middot; boost the loss, not the
trigger &middot; every number traces to a committed CSV in
<code>docs/dev/transition/pdv-trigger/data/</code> &middot; light-mode, MathJax-rendered.</footer>
"""


def main():
    parts = [
        HEAD, HERO,
        SEC_SETUP, SEC_DOUBLE, SEC_CONVENTION, SEC_REGIME, SEC_CLOSURE,
        SEC_LIT, SEC_WIRING, SEC_BOTTOM, SEC_REPRO,
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
