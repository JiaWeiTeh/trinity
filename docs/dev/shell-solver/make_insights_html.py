#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single self-contained, light-mode, MathJax-rendered HTML "insights"
report that tells the shell-solver investigation end to end: background →
hypotheses → method → five diagnostics (plots + tables) → the shipped solution.

The 5 PNGs from make_plots.py are embedded as base64 so the file is standalone
(downloadable, opens offline; MathJax loads from CDN for the few formulas).

REPRODUCE
    cd /home/user/trinity
    python docs/dev/shell-solver/plots/make_plots.py        # (re)build the PNGs
    python docs/dev/shell-solver/make_insights_html.py      # -> insights.html
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
OUT = HERE / "insights.html"


def img(name, alt):
    b64 = base64.b64encode((PLOTS / name).read_bytes()).decode()
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
figure{margin:20px 0;} figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block;}
figcaption{color:var(--mut);font-size:13.5px;margin-top:7px;text-align:center;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top;}
th{background:var(--panel);font-weight:600;} tbody tr:nth-child(even){background:#fbfcfe;}
.win{color:var(--find);font-weight:700;} .loss{color:var(--bad);font-weight:700;}
.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:20px;
padding:2px 11px;font-size:12.5px;font-weight:600;}
.muted{color:var(--mut);} .small{font-size:13px;}
ol.toc{columns:2;font-size:14.5px;color:var(--accent);}
hr{border:0;border-top:1px solid var(--line);margin:30px 0;}
footer{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px;}
"""

MATHJAX = """
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)'],['$','$']],
displayMath:[['$$','$$'],['\\\\[','\\\\]']]},svg:{fontCache:'global'}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def main():
    P = []
    P.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRINITY shell-solver — insights</title>{MATHJAX}<style>{CSS}</style></head>
<body><div class="wrap">""")

    # ---- hero + TL;DR ----
    P.append("""
<span class="tag">TRINITY · shell-structure ODE integrator</span>
<h1>Should we migrate the shell ODE from <code>odeint</code> to <code>solve_ivp</code>?</h1>
<p class="sub">An empirical investigation — background, hypotheses, a replay harness, five
diagnostics, and the shipped fix. Verified 2026-06-18; all artifacts committed under
<code>docs/dev/shell-solver/</code>.</p>

<div class="tldr">
<p style="margin:0"><b>TL;DR.</b> The shell ODE is solved with <code>scipy.integrate.odeint</code>
(LSODA over a ~1000-point grid), which emits an "excess work" warning in one degenerate
configuration. We hypothesised a <code>solve_ivp</code> migration would be faster (a terminal
event could stop at the ionisation front and skip the wasted tail). Across <b>6 configurations ×
2 phases × 6 solver/stopping variants</b>, with <b>100 sampled solves per implicit phase</b>:
equivalence is settled (all LSODA variants match <code>odeint</code> to \\(\\sim\\!10^{-8}\\)),
but <b>no variant is faster over a realistic run</b> — the event's only win is
<b>energy-phase-only</b>. The warning itself is fixed <b>for free</b> by raising
<code>odeint</code>'s step ceiling (<code>mxstep=50000</code>; shipped, <code>pytest</code>
532&nbsp;passed). <b>Verdict: any migration is a robustness/cleanliness choice, not a speedup.</b></p>
</div>
""")

    # ---- the chain of reasoning (flow) ----
    P.append("""
<h2 id="flow">The chain of reasoning</h2>
<p>Each diagnostic answered one question and set up the next. This is the spine of the report;
every link is backed by a plot and a table below.</p>
<div class="flow">""")
    steps = [
        ("1 · Can the alternatives even reproduce <code>odeint</code>?",
         "Yes — every LSODA-family variant matches to \\(\\sim\\!10^{-8}\\); Radau/BDF drift to \\(\\sim\\!10^{-7}\\). Accuracy rules nothing out."),
        ("2 · Given equivalence, is anything <i>faster</i>?",
         "No — every variant's speed distribution sits mostly below 1.0×. Only the front-event pokes above, and only partly."),
        ("3 · Where does the event actually win?",
         "Only in the degenerate <b>energy</b> phase (4–5×). In the implicit phase it collapses to ~0.5× — a loss — everywhere."),
        ("4 · Why does the win vanish in implicit?",
         "The implicit phase is mostly <b>neutral / mass-limited</b> solves; the φ-front event has nothing to skip there."),
        ("5 · Then what fixes the warning we started with?",
         "It's localised to the degenerate energy phase. Raising <code>odeint</code>'s <code>mxstep</code> removes it, bit-identical — shipped."),
    ]
    for i, (q, a) in enumerate(steps):
        P.append(f'<div class="step"><div class="q">{q}</div><div class="a">{a}</div></div>')
        if i < len(steps) - 1:
            P.append('<div class="arrow">▼</div>')
    P.append('</div>')
    P.append('<p><b>Conclusion:</b> equivalence holds → no speedup exists → the one apparent win is '
             'energy-phase-only → because implicit is neutral/mass-limited → so migration is '
             'robustness-only, and the warning is fixed for free by <code>mxstep</code>.</p>')

    # ---- background ----
    P.append("""
<h2 id="bg">1 · Background &amp; motivation</h2>
<p>TRINITY integrates an expanding feedback bubble through its phases. Inside each step it solves a
stiff <b>shell-structure ODE</b> for density \\(n\\), ionising flux fraction \\(\\phi\\), and optical
depth \\(\\tau\\), via <code>scipy.integrate.odeint</code> (LSODA) over a fixed ~1000-point radius grid
(<code>trinity/shell_structure/shell_structure.py</code>, ionised and neutral branches).</p>
<p>In one <b>degenerate</b> configuration (<code>simple_cluster</code>, where code-unit values overflow
<code>float64</code> in the ionised tail), <code>odeint</code> exhausts its internal step ceiling and
prints <i>“Excess work done on this call …”</i>, silently truncating the integration. The bubble-structure
ODE was previously migrated <code>odeint → solve_ivp(dense_output)</code> for exactly such robustness
reasons, which raised the question for the shell ODE too.</p>
<p class="muted small">Two quantities recur below. Speedup of a variant vs the current default:
\\( S = t_{\\text{odeint}} / t_{\\text{variant}} \\) (so \\(S>1\\) is faster). Accuracy:
\\( \\mathrm{rel}_n = \\max_i \\lvert n_i^{\\text{var}} - n_i^{\\text{odeint}}\\rvert / \\lvert n_i^{\\text{odeint}}\\rvert \\)
over the physically-used prefix of the grid.</p>
""")

    # ---- hypotheses ----
    P.append("""
<h2 id="hyp">2 · Hypotheses</h2>
<div class="box hyp"><div class="lab">Hypothesis H1 — equivalence</div>
A modern <code>solve_ivp</code> integration can reproduce the legacy <code>odeint</code> shell solution
to round-off, so a switch is behaviour-preserving.</div>
<div class="box hyp"><div class="lab">Hypothesis H2 — speed (the appealing one)</div>
The physically-used part of the solve ends at the ionisation front (\\(\\phi \\to 0\\)); the grid past it is
discarded. A <code>solve_ivp</code> <b>terminal event</b> at \\(\\phi \\le 10^{-9}\\) would stop there and
skip the wasted tail — so the migration would also be <i>faster</i>.</div>
<div class="box hyp"><div class="lab">Hypothesis H3 — worth it</div>
Therefore the migration is justified on performance grounds (not only robustness).</div>
""")

    # ---- method ----
    P.append("""
<h2 id="method">3 · Method — a replay harness</h2>
<p>Rather than reason in the abstract, we measured. <code>capture_replay_variants.py</code> monkey-patches
<code>scipy.integrate.odeint</code> during a <i>real</i> simulation; on each captured shell solve it replays
six candidate variants plus the baseline on the identical inputs, recording accuracy, wall-time (median of
5&nbsp;reps), solver warnings, and event firing. The host run consumes the true <code>odeint</code> result, so
the simulation is unperturbed.</p>
<table><thead><tr><th>variant</th><th>solver</th><th>stopping rule</th><th>idea being tested</th></tr></thead>
<tbody>
<tr><td><b>odeint</b> <span class="tag" style="background:#8d99ae">CURRENT DEFAULT</span></td><td>LSODA</td><td>full fixed grid</td><td>the production baseline (1.00× ref)</td></tr>
<tr><td>LSODA + t_eval</td><td>LSODA</td><td>full grid via <code>solve_ivp</code></td><td>faithful drop-in</td></tr>
<tr><td>LSODA + front event</td><td>LSODA</td><td>terminal event at \\(\\phi\\le10^{-9}\\)</td><td><b>skip the overflow tail</b> (H2)</td></tr>
<tr><td>LSODA + dense_output</td><td>LSODA</td><td>continuous interpolant</td><td>the bubble-precedent style</td></tr>
<tr><td>Radau</td><td>Radau (implicit RK)</td><td>full grid</td><td>a different stiff solver</td></tr>
<tr><td>BDF</td><td>BDF</td><td>full grid</td><td>a different stiff solver</td></tr>
<tr><td>odeint mxstep=50k</td><td>LSODA (odeint)</td><td>raise step ceiling</td><td>the cheap warning fix</td></tr>
</tbody></table>
<p><b>Configurations (6):</b> two <b>degenerate</b> (<code>sfe0.3</code> = current default, <code>sfe0.6</code>)
and four <b>realistic</b> science regimes (<code>typical</code>, <code>steep</code>, <code>dense_flat</code>,
<code>mock_hybr</code>). <b>Phases:</b> energy and implicit.</p>
<div class="box over"><div class="lab">Diagnostic principle (a course-correction mid-study)</div>
What matters is the number of solves sampled <i>in</i> a phase, not the wall-time spent reaching it. An early
15-sample pass mis-read the implicit phase as fully ionised. We re-ran targeting <b>100 captured solves in the
implicit phase</b> per config — which overturned a conclusion (see Step&nbsp;3).</div>
""")

    # ---- the five diagnostics ----
    P.append('<h2 id="diag">4 · The five diagnostics</h2>')

    P.append(f"""
<h3>Step 1 — Equivalence gate</h3>
<p><b>Question:</b> can any variant reproduce <code>odeint</code>? <b>Finding:</b> all LSODA-family variants
agree to the \\(\\sim\\!10^{{-8}}\\) floor (the solver's own rtol); Radau and BDF — being different solvers —
drift to \\(\\sim\\!10^{{-7}}\\). <code>odeint(mxstep=50k)</code> is bit-identical.</p>
<figure>{img("1_accuracy_gate.png","accuracy gate")}<figcaption>Worst-case
\\(\\mathrm{{rel}}_n\\) per variant across all 12 cells.</figcaption></figure>
<div class="box find"><div class="lab">H1 confirmed</div>Equivalence is settled. Accuracy rules nothing
out — so the decision reduces to <b>speed</b> (and robustness).</div>
""")

    P.append(f"""
<h3>Step 2 — Speed</h3>
<p><b>Question:</b> is anything faster than the current default? <b>Finding:</b> every variant's
per-solve speed distribution sits mostly <i>below</i> 1.0× (slower). Only <code>LSODA + front event</code>
pokes above 1.0×, and only part of its spread; <code>odeint(mxstep=50k)</code> sits right at 1.0×.</p>
<figure>{img("2_speed_distribution.png","speed distribution")}<figcaption>Distribution of
\\(S = t_{{\\text{{odeint}}}}/t_{{\\text{{variant}}}}\\) per variant; dashed line = the current default.</figcaption></figure>
<div class="box find"><div class="lab">finding</div>The faithful drop-in is ~0.15–0.21× (5–7× <i>slower</i>);
Radau/BDF ~0.05× (~20× slower). The only speedup candidate is the front-event — so we drilled into it.</div>
""")

    P.append(f"""
<h3>Step 3 — The event win is energy-phase-only <span class="tag" style="background:#e8842a">overturned a conclusion</span></h3>
<p><b>Question:</b> where does the event win? <b>Finding:</b> only in the degenerate <b>energy</b> phase
(4.2–4.4×). In the <b>implicit</b> phase it collapses to ~0.5× — a net loss — in every configuration.</p>
<figure>{img("3_event_energy_only.png","event speedup by config and phase")}<figcaption>
LSODA+event median speedup, split by phase.</figcaption></figure>
<div class="box over"><div class="lab">why the 100-sample redo mattered</div>
At 15 implicit samples, <code>sfe0.3</code>'s implicit phase <i>looked</i> fully ionised and the event
scored <b>5.65×</b>. At 100 samples the implicit phase is revealed as <b>mixed (58 ionised / 42 neutral,
42% mass-limited)</b> and the event collapses to <b>0.53×</b>. The early conclusion ("event is a big win")
was an artefact of under-sampling.</div>
""")

    P.append(f"""
<h3>Step 4 — Why: phase composition</h3>
<p><b>Question:</b> why does the win vanish in implicit? <b>Finding:</b> energy solves are ~100% fully
ionised, so the event can skip the \\(\\phi\\)-overflow tail. The implicit phase is dominated by
<b>neutral</b> and <b>mass-limited</b> solves (the shell is swept up before \\(\\phi\\) depletes) — the
event has nothing to skip, and the <code>solve_ivp</code> call overhead dominates.</p>
<figure>{img("4_phase_composition.png","phase composition")}<figcaption>Ionised / neutral split and
mass-limited fraction, per config, per phase.</figcaption></figure>
<div class="box find"><div class="lab">H2 &amp; H3 falsified</div>There is no speed case for the migration:
the event's mechanism only applies to a minority of (energy-phase) solves.</div>
""")

    P.append(f"""
<h3>Step 5 — The fix</h3>
<p><b>Question:</b> then what fixes the original warning? <b>Finding:</b> the warning is localised to the
degenerate energy phase (100% of those solves). Raising <code>odeint</code>'s step ceiling
(<code>mxstep=50000</code>) lets the integration complete: <b>free (~1.0×) in the science configs</b> where
the ceiling was barely touched, and ~0.2× in the degenerate energy phase — because it now does the heavy
overflow work the warning was hiding. <b>Bit-identical (\\(\\mathrm{{rel}}_n=0\\)) throughout.</b></p>
<figure>{img("5_free_fix_mxstep.png","the mxstep fix")}<figcaption>Excess-work warning fraction (bars) vs
<code>odeint(mxstep=50k)</code> speed (◆); the warning lives only where the bars are tall.</figcaption></figure>
""")

    # ---- master tables ----
    P.append("""
<h2 id="tables">5 · Master results</h2>
<p class="small muted">Pooled over the implicit-heavy sample (20 energy + 100 implicit per config).
<code>ok</code> = successful / sampled solves; <code>speed</code> = median \\(S\\); <code>rel_n</code> = worst.</p>
<h3>By variant — solver &amp; stopping (the deciding axis)</h3>
<table><thead><tr><th>variant</th><th>solver · stopping</th>
<th>degenerate (sfe0.3/0.6)</th><th>realistic (typ/steep/dense/mock)</th></tr></thead><tbody>
<tr><td>LSODA + t_eval (drop-in)</td><td>LSODA · normal</td><td>240/240 · <span class="loss">0.16×</span> · 2.6e-9</td><td>480/480 · <span class="loss">0.21×</span> · 1.0e-8</td></tr>
<tr><td><b>LSODA + front event</b></td><td>LSODA · <b>smart stop</b></td><td>240/240 · 0.65× · 2.6e-9</td><td>480/480 · <span class="loss">0.29×</span> · 1.0e-8</td></tr>
<tr><td>LSODA + dense_output</td><td>LSODA · normal</td><td>240/240 · <span class="loss">0.15×</span> · 2.6e-9</td><td>480/480 · <span class="loss">0.21×</span> · 1.0e-8</td></tr>
<tr><td>Radau</td><td>Radau · normal</td><td>151/240 · <span class="loss">0.05×</span> · 3.9e-8</td><td>469/480 · <span class="loss">0.05×</span> · 1.7e-7</td></tr>
<tr><td>BDF</td><td>BDF · normal</td><td>151/240 · <span class="loss">0.05×</span> · 1.7e-7</td><td>469/480 · <span class="loss">0.05×</span> · 1.9e-7</td></tr>
<tr><td>odeint mxstep=50k</td><td>LSODA · mxstep=50k</td><td>240/240 · 0.98× · <span class="win">0</span></td><td>480/480 · <span class="win">1.00×</span> · <span class="win">0</span></td></tr>
</tbody></table>

<h3>Context — per config × phase (baseline <code>odeint ms</code> is the current default)</h3>
<table><thead><tr><th>config</th><th>phase</th><th>ion/neu</th><th>odeint ms</th><th>excess-work</th>
<th>mass-lim</th><th>t_eval</th><th>event</th><th>Radau/BDF ok</th></tr></thead><tbody>
<tr><td><b>sfe0.3</b> (default)</td><td>energy</td><td>20/0</td><td>9.54</td><td>100%</td><td>0%</td><td>0.10×</td><td class="win">4.18×</td><td>0/20</td></tr>
<tr><td><b>sfe0.3</b> (default)</td><td>implicit</td><td>58/42</td><td>0.92</td><td>15%</td><td>42%</td><td>0.18×</td><td class="loss">0.53×</td><td>85/100</td></tr>
<tr><td>sfe0.6</td><td>energy</td><td>20/0</td><td>12.62</td><td>100%</td><td>0%</td><td>0.10×</td><td class="win">4.39×</td><td>0/20</td></tr>
<tr><td>sfe0.6</td><td>implicit</td><td>71/29</td><td>1.21</td><td>34%</td><td>31%</td><td>0.16×</td><td class="loss">0.49×</td><td>66/100</td></tr>
<tr><td>steep</td><td>energy</td><td>20/0</td><td>1.45</td><td>20%</td><td>0%</td><td>0.25×</td><td>0.62×</td><td>16/20</td></tr>
<tr><td>steep</td><td>implicit</td><td>50/50</td><td>0.87</td><td>0%</td><td>50%</td><td>0.20×</td><td class="loss">0.34×</td><td>100/100</td></tr>
<tr><td>dense_flat</td><td>energy</td><td>20/0</td><td>1.42</td><td>20%</td><td>0%</td><td>0.25×</td><td>0.61×</td><td>16/20</td></tr>
<tr><td>dense_flat</td><td>implicit</td><td>50/50</td><td>0.87</td><td>0%</td><td>50%</td><td>0.20×</td><td class="loss">0.30×</td><td>100/100</td></tr>
<tr><td>typical</td><td>energy</td><td>20/0</td><td>1.22</td><td>15%</td><td>0%</td><td>0.24×</td><td>0.46×</td><td>17/20</td></tr>
<tr><td>typical</td><td>implicit</td><td>66/34</td><td>0.78</td><td>0%</td><td>34%</td><td>0.22×</td><td class="loss">0.35×</td><td>100/100</td></tr>
<tr><td>mock_hybr</td><td>energy</td><td>20/0</td><td>0.24</td><td>0%</td><td>95%</td><td>0.17×</td><td class="loss">0.14×</td><td>20/20</td></tr>
<tr><td>mock_hybr</td><td>implicit</td><td>79/21</td><td>0.48</td><td>0%</td><td>64%</td><td>0.17×</td><td class="loss">0.14×</td><td>100/100</td></tr>
</tbody></table>
<p class="small muted">Full 84-row detail (incl. the current-default row) in
<code>data/master_table.csv</code>.</p>
""")

    # ---- solution ----
    P.append("""
<h2 id="solution">6 · Solution &amp; what shipped</h2>
<div class="box find"><div class="lab">shipped</div>
<code>trinity/shell_structure/shell_structure.py</code> now passes <code>mxstep=_SHELL_ODE_MXSTEP</code>
(= 50000) on both <code>odeint</code> calls — a module constant, mirroring the bubble module's
<code>_BUBBLE_RTOL</code>. Bit-identical on the used prefix; <code>pytest</code>: <b>532 passed</b>.
This silences the warning without changing solver or API.</div>
<p><b>On the migration question (H3):</b> a <code>solve_ivp</code> switch would be a
<b>robustness/cleanliness</b> change — explicit <code>sol.success</code> flag, no consumption of an
uninitialised-memory tail — <i>not</i> a performance one. It is not written, and is optional.</p>
<p><b>Scope honestly left open:</b> a dedicated 45-minute <code>sfe0.3</code> run captured 100 implicit
solves but reached <b>zero transition-phase</b> solves — the transition/momentum phases sit beyond a
tractable capture budget and need the full-run "shadow" path. And ODE-level <code>rel_n</code> equivalence
is necessary but not sufficient: integration-level equivalence of the consumed scalars (<code>n_IF_Str</code>,
<code>F_rad</code> inputs) is still unverified.</p>
""")

    # ---- artifacts ----
    P.append("""
<h2 id="repro">7 · Artifacts &amp; reproducibility</h2>
<p class="small">Everything is committed under <code>docs/dev/shell-solver/</code> on branch
<code>bugfix/LSODA-shellODE</code> — reproducible without re-running the (hours-long) sims.</p>
<table><thead><tr><th>artifact</th><th>what</th></tr></thead><tbody>
<tr><td><code>harness/capture_replay_variants.py</code></td><td>the replay harness (per-phase matrix mode)</td></tr>
<tr><td><code>harness/run_matrix_sweep.sh</code></td><td>resumable, reset-safe sweep driver (100 implicit/config)</td></tr>
<tr><td><code>harness/aggregate_matrix.py</code></td><td>CSVs → <code>master_table.csv</code> + rendered tables</td></tr>
<tr><td><code>data/replay_variants_matrix_*.csv</code></td><td>per-config raw captures (the evidence)</td></tr>
<tr><td><code>plots/make_plots.py</code> + <code>plots/*.png</code></td><td>the five diagnostics above</td></tr>
<tr><td><code>MIGRATION_PLAN.md</code> §P0-matrix</td><td>the living plan &amp; verified conclusions</td></tr>
</tbody></table>
<p class="small muted">Rebuild this report:
<code>python docs/dev/shell-solver/plots/make_plots.py &amp;&amp; python docs/dev/shell-solver/make_insights_html.py</code></p>
<footer>TRINITY shell-solver investigation · generated from committed artifacts ·
numbers trace to <code>data/master_table.csv</code> · light-mode, MathJax-rendered.</footer>
""")

    P.append("</div></body></html>")
    OUT.write_text("".join(P), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
