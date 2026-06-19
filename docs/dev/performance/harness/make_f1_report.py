#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a self-contained HTML report on the bubble_luminosity performance arc.

Embeds the figs/*.png as base64 (so the single .html is downloadable / works
offline) and renders LaTeX via MathJax (CDN). The PRIOR-history sections (Eras
A/B/C) are filled from PREV_WORK_HTML; the F1 sections are written inline.

    python docs/dev/performance/harness/make_f1_report.py
    -> docs/dev/performance/F1_REPORT.html
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
OUT = os.path.join(HERE, "..", "F1_REPORT.html")


def img(name, alt):
    p = os.path.join(FIGS, name)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


PREV_WORK_HTML = r"""
<h2>1. Era A — the <code>odeint</code> crash, and the migration that made the 60k vestigial</h2>
<h3>1.1 Problem</h3>
<p><code>run.py</code> aborted <b>nondeterministically</b> — a bare <code>MonotonicError</code> (exit 1),
roughly <b>1 in 3</b> identical re-runs, same inputs, same numpy. The error surfaced ~100 lines and a module
away from the real fault. The root cause was not floating-point thread order (the original theory, later
<i>superseded</i>): a single-threaded, fixed-<code>PYTHONHASHSEED</code> repro reproduced the coin-flip. The
real mechanism:</p>
<blockquote>"when LSODA bails (istate ≠ 2), <code>scipy.integrate.odeint</code> returns <b>uninitialised memory</b>
for the un-integrated tail, and <i>those bytes</i> vary run-to-run … Consuming that garbage is what made the
whole bubble solve non-deterministic."</blockquote>
<p>So <code>fsolve</code> could converge on a ~0 <i>garbage</i> residual, and the temperature array could grow a
random tail that intermittently tripped the monotonicity guard.</p>

<h3>1.2 The origin of the 60&nbsp;000-point grid <span class="meta">(the crux of the whole arc)</span></h3>
<p>The ~60k grid was <b>not a free design choice</b> — it was <i>the output grid <code>odeint</code> was asked to
integrate on</i>. <code>odeint</code>'s API integrates/interpolates onto exactly the caller's array of points, so
<b>with <code>odeint</code> the output grid and the integration request were the same object</b>: ~60&nbsp;000
near-duplicate radii (<code>_create_radius_grid</code> packs 20&nbsp;000 points into a ~1.6&times;10⁻⁴ pc sliver,
$dr\approx5\times10^{-9}$ pc). Asking LSODA for thousands of near-duplicate output radii inside a handful of real
steps is exactly what stressed its dense-output interpolation and produced the spikes/crash:</p>
<blockquote>"Requesting thousands of near-duplicate output radii inside a handful of real steps is exactly what
stresses LSODA's dense-output interpolation … one unlucky interpolation returns a single off sample → the spike."</blockquote>

<h3>1.3 Idea &amp; fix — detect failure, and decouple accuracy from sampling</h3>
<p>The cure had two parts: (1) <b>never consume a failed solve</b> — check the solver's success flag and return a
deterministic penalty (<code>_SOLVER_FAIL_RESIDUAL = 1e3</code>) / raise <code>BubbleSolverError</code>; (2)
migrate the integration to <b><code>solve_ivp(dense_output=True)</code></b>. That migration is the pivot the whole
arc turns on — it <b>decouples integration accuracy from output sampling</b>:</p>
<blockquote>"the integrator chooses its own adaptive steps (accuracy set by rtol/atol) and the output grid is
sampled from the <i>continuous</i> solution. This decouples integration accuracy from output sampling — the
near-duplicate radii … are never requested of the integrator."</blockquote>
<p class="lesson"><b>The demotion.</b> Under <code>solve_ivp</code>, the integrator picks its own ~850–1000 adaptive
steps; the 60k is now merely the points you <i>sample the continuous solution at</i>. It went from a
<b>load-bearing integration grid</b> to a <b>vestigial output grid</b> — exactly what F1 later removes.</p>

<h3>1.4 What was tested &amp; the numbers</h3>
<ul>
<li>Determinism repro (#659): single-threaded, fixed seed → <b>794 / 786 / 792</b> <code>odeint</code> calls across
identical runs (an OK/crash coin-flip). Post-fix: 4 runs <b>byte-identical</b> (<code>dictionary.jsonl</code>
sha256 matched, run completes).</li>
<li>Migration output-diffs (98 snapshots × 67 fields, single-threaded): main solve change ≤ <b>1.4e-3</b>
(<code>dMdt</code>), velocity-residual solve converged <code>dMdt</code> shift ≤ <b>0.28%</b>, <b>0/67</b> fields
&gt;1%.</li>
<li>Wall time at the residual-solve migration: <b>222.7 s → 199.6 s (−10.4%)</b>.</li>
<li>"Smoking gun": ~854 internal LSODA steps for ~60&nbsp;000 output radii (≈70× dense-output interpolation).</li>
</ul>
<p class="meta">Shipped: <code>a245c29</code> (#659), <code>1eb7f4d</code>, <code>5f4f229</code>,
<code>76921f7</code> (PR #666). <b>Lesson:</b> detect &amp; fail loud — never consume a failed solve; and the
non-determinism was <i>memory garbage</i>, not thread FP order (only a single-threaded fixed-seed repro proved it).</p>

<h2>2. Era B — conduction-zone convergence (the decoupling pays off)</h2>
<p><b>Problem:</b> the conduction-zone luminosity used a fragile, under-resolved <b>~100-point <code>odeint</code>
re-solve</b> (~0.9% low). <b>Idea:</b> stop re-solving — <i>sample the already-computed dense solution</i> across
the conduction band, and pick the density $K$ from a convergence study, not a guess.</p>
<p><b>Tested:</b> <code>tools/bubble_conduction_convergence.py</code>, deterministic, $K\in\{500,2\text{k},10\text{k},
50\text{k},200\text{k}\}$ over 12 Phase-1a states (quickstart smoke). <b>Result:</b> convergence is
$\sim 1/K^2$; <b>$K=2000$ is within ~7&times;10⁻⁵</b> of the $K\!\to\!\infty$ value at the worst (thin-bubble)
state, costing <b>~1 ms/call</b>; the integral is <code>rtol</code>-independent (&lt;0.001% across rtol 1e-6→1e-10);
0 integration failures.</p>
<blockquote>"The converged integral is <b><code>rtol</code>-independent</b> … accuracy is set by the cheap output
sampling, not the integrator. … integrate once cheaply, refine the quadrature for free."</blockquote>
<p><b>Outcome (<code>5f4f229</code>):</b> deleted the re-solve + the well/under-resolved branch split, replaced by
sampling the dense solution at <code>_CONDUCTION_NPTS = 2000</code>; wall time neutral (<b>+0.7%</b>), and the
conduction bias tightened <i>toward</i> the converged value (a bounded physics correction). <b>Lesson:</b> the
same <code>solve_ivp</code> decoupling that fixed the crash bought accuracy <i>for free</i> — "efficiency without
brute force, but properly resolved."</p>

<h2>3. Era C — F2 "free wins" (bit-identical hot-path cleanups)</h2>
<p>A 2026-06-18 hot-path audit harvested a batch of <b>bit-identical</b> cleanups around the bubble path
(<code>4a13075</code>), gated by a <code>git show HEAD</code> value-diff harness + the full non-stress suite
(535 passed):</p>
<table>
<tr><th>item</th><th>change</th><th>effect</th></tr>
<tr><td><b>F2.3+F2.4</b></td><td><code>get_dudt</code>: cache cooling cutoffs (run-constants recomputed every
call) + move <code>Lambda_CIE</code> into its branch</td><td><b>+23.1%/call</b> — 163.7 → 125.9 µs; exactly
bit-identical (0 mismatches / 540 pts)</td></tr>
<tr><td><b>F2.2</b></td><td>gravity outputs (<code>grav_phi</code>, <code>grav_force_m</code>) computed then
discarded → disabled (<code>None</code> placeholders)</td><td>removes a <code>simpson</code> + a 60k-divide per
final structure solve; bit-identical</td></tr>
<tr><td><b>F2.1</b></td><td>default <code>log_level</code> DEBUG → INFO</td><td>no wall win (A/B was noise), but
<code>trinity.log</code> <b>~340× smaller</b> (2.7 MB → 8 KB)</td></tr>
<tr><td>F2.5</td><td><s>remove <code>pdotdot_total</code></s></td><td><b>DROPPED</b> — it <i>is</i> consumed by an
integrated RHS (phase-1b A12 coeff); not bit-identical</td></tr>
</table>
<p class="meta"><b>Lesson:</b> <b>measure, don't guess</b> — the audit retracted its own "logging is the biggest
free win" claim, and F2.5's "never consumed" was wrong (it fed the phase-1b trajectory). The strict bit-identity
bar kept "free" honest. (Follow-up <code>7f08e58</code> dropped a misleading <code>_legacy</code> suffix —
these were the sole production path all along.)</p>
"""

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRINITY — bubble_luminosity performance evolution (the 60k story + F1)</title>
<script>
MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root{--ink:#1a1a1a;--mut:#555;--acc:#1b5e20;--box:#e8f5e9;--line:#ddd;--code:#f4f4f4;}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
       max-width:920px;margin:0 auto;padding:2.2rem 1.4rem 5rem;line-height:1.6;font-size:16px;}
  h1{font-size:1.9rem;line-height:1.25;margin:.2rem 0 .2rem;}
  h2{font-size:1.4rem;margin:2.6rem 0 .6rem;border-bottom:2px solid var(--acc);padding-bottom:.25rem;color:var(--acc);}
  h3{font-size:1.13rem;margin:1.7rem 0 .4rem;}
  .sub{color:var(--mut);font-size:1.05rem;margin:.1rem 0 1.2rem;}
  .tldr{background:var(--box);border:1px solid var(--acc);border-radius:8px;padding:1rem 1.2rem;margin:1.4rem 0;}
  code,kbd{background:var(--code);padding:.1em .35em;border-radius:4px;font-size:.9em;}
  pre{background:var(--code);padding:.8rem 1rem;border-radius:6px;overflow-x:auto;font-size:.86em;}
  table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem;}
  th,td{border:1px solid var(--line);padding:.45rem .6rem;text-align:left;vertical-align:top;}
  th{background:#f0f4f0;}
  figure{margin:1.5rem 0;text-align:center;}
  img{max-width:100%;border:1px solid var(--line);border-radius:6px;}
  figcaption{color:var(--mut);font-size:.86rem;margin-top:.4rem;font-style:italic;}
  .lesson{border-left:4px solid var(--acc);background:#fafdfa;padding:.6rem 1rem;margin:1rem 0;}
  .warn{border-left:4px solid #b71c1c;background:#fffafa;padding:.6rem 1rem;margin:1rem 0;}
  .placeholder{color:#999;font-style:italic;padding:1rem;border:1px dashed #bbb;border-radius:6px;}
  .meta{color:var(--mut);font-size:.85rem;}
  ul li{margin:.2rem 0;}
  .ok{color:var(--acc);font-weight:600;}
</style></head><body>

<h1>TRINITY — the bubble-luminosity performance story</h1>
<p class="sub">From a load-bearing 60&nbsp;000-point <code>odeint</code> grid to its removal (F1). The full chain of how the bubble-structure hot path was made correct, then fast.</p>
<p class="meta">Generated from committed data &amp; figures · branch <code>fix/hotpath-resample</code> · TRINITY (<code>trinity-sf</code>)</p>

<div class="tldr">
<b>TL;DR.</b> <code>bubble_luminosity.py</code> solves the Weaver+77 bubble-structure ODE and, via an
<code>fsolve</code> over the mass-loss rate $\dot M$, is the dominant per-step cost of a run. Its hot path was
reworked in three eras: <b>(A)</b> a nondeterministic <code>odeint</code> crash was fixed by migrating to
<code>solve_ivp(dense_output)</code> — which quietly turned the original ~60k integration grid into a
<i>vestigial output grid</i>; <b>(B)</b> the conduction zone was converged with a $K{=}2000$ dense sample;
<b>(C)</b> "free win" cleanups (logging, dead gravity, cooling caches). <b>F1</b> finishes the job: it removes
the now-vestigial 60k resample from the dMdt residual — <b>~1.5&times; per call, ~2.3&times; full-run</b> on the
degenerate config, with full-run physical equivalence to <b>~6&times;10⁻⁶</b>.
</div>

<h2>0. Background — what the hot path computes</h2>
<p>For a trial mass-loss rate $\dot M$, TRINITY integrates the Weaver+77 bubble-structure ODE for the wind
velocity $v(r)$, temperature $T(r)$, and gradient $T'(r)$ inward across the shocked-wind bubble, from the outer
boundary $r_2'$ to the inner radius $R_1$ (a stiff system near small $T$):</p>
$$\frac{dv}{dr} = \frac{\beta+\delta}{t} + \frac{(v-v_{\rm term})\,T'}{T} - \frac{2v}{r},
\qquad v_{\rm term}=\frac{\alpha\,r}{t},$$
<p>with a coupled $T'' $ from the energy equation and a cooling source $\dot u(\,n,T,\phi\,)$. The physical
$\dot M$ is the root of a <b>velocity residual</b> — the integrated solution must arrive at $v\!\to\!0$ at the
inner radius:</p>
$$\mathcal{R}(\dot M)\;=\;\frac{v(R_1)-0}{v(r_2')+10^{-4}}\;\xrightarrow{\;\text{fsolve}\;}\;0 .$$
<p>So <code>_get_velocity_residuals</code> is evaluated <b>thousands of times per run</b> (every fsolve probe,
every step). Plus two rejection guards along the path — a minimum-temperature floor and monotonicity:</p>
$$\min_r T(r) \;\ge\; T_{\rm init}=3\times10^{4}\,\text{K},\qquad T(r)\ \text{monotonic}.$$
<p>The cost of that residual — and specifically how its solution is <i>sampled</i> — is the whole story below.</p>

<!--PREV_WORK-->

<h2>4. F1 — remove the vestigial 60k resample from the residual</h2>

<h3>4.1 Problem — resampling 60&nbsp;000 points to read 4 numbers</h3>
<p>After Era A, <code>_get_velocity_residuals</code> still did this every call: build a ~60k-point grid
(<code>_create_radius_grid</code> — three stitched <code>logspace(2e4)</code> chunks) and evaluate the dense
solution on all of it (<code>sol.sol(r_array)</code>), purely to read $v$ at the two endpoints and run the
$\min T$/monotonic guards. A microbenchmark put the resample at <b>~21 ms</b> vs the integration at
<b>~0.8 ms</b> — i.e. the residual spent <b>~96%</b> of its time resampling 60&nbsp;000 points to extract
4 numbers. Across thousands of fsolve calls/run, this is the dominant bubble-path cost.</p>

<h3>4.2 Idea — integrate once on a coarse grid</h3>
<p>Since <code>solve_ivp(dense_output)</code> already decoupled integration accuracy (set by
<code>rtol</code>/<code>atol</code>) from the output grid (Era A), the residual doesn't need 60k points — it
needs $v$ at the endpoints and a sufficiently-resolved $T(r)$ for the guards. So: integrate <b>once</b> with a
coarse <code>t_eval</code> of $N$ points and drop the dense resample entirely. The open question was
<i>which $N$</i>, and whether a coarse grid could miss a sharp $T$ feature and mis-fire a guard.</p>

<h3>4.3 Testing the options — a method matrix</h3>
<p>Six variants were captured side-by-side on every bubble call: the <code>baseline</code> (60k), four fixed
grids <code>M2000/M1000/M500/M200</code>, and <code>Mnodes</code> (adaptive nodes only, no <code>t_eval</code>).</p>
__FIG_VARIANT__
<p>Two findings settle the choice. <b>(a)</b> every coarse option beats the 60k baseline by ~1.5&times; and the
speed is <i>flat</i> across $N$ (981→934 ms from M2000→Mnodes). <b>(b)</b> accuracy is <b>npts-insensitive</b>:
<code>M2000=M1000=M500=M200</code> to the digit (worst $|\Delta\dot M/\dot M|\approx 3\times10^{-6}$), all ~1000&times;
inside the 0.3% gate. With both curves flat, the choice is pure robustness margin →
<b>$N=500$</b>: a <i>guaranteed</i> fixed resolution of the $\min T$/monotonic guards (vs <code>Mnodes</code>'
variable adaptive-node count, which is fine here but less predictable on an unseen config).</p>

<h3>4.4 The same idea across regimes — a 6-config sweep</h3>
<p>The per-call equivalence + speedup were measured across six configs spanning the regime space
(<code>mock_hybr</code>, <code>probe_typical_hybr</code>, <code>steep</code>, <code>dense_flat</code>,
<code>simple_cluster</code>, <code>sfe0.6</code>), in both the energy and implicit phases (20 + 100 captured
bubble-calls each):</p>
__FIG_PERCALL__
<p>The per-call speedup is <b>uniform ~1.5&times;</b> — because the 60k resample is a <i>fixed-size</i>
operation, the win is config-independent. Accuracy is <b>$\le 3.1\times10^{-6}$</b> everywhere.</p>

<h3>4.5 The validation journey — why per-call wasn't enough</h3>
<p>The per-call result was necessary but <b>not sufficient</b>, and proving that was the crux of the work:</p>
<div class="warn"><b>A false alarm.</b> A first full-run A/B (<code>ab_fullrun.py</code>) ran both variants
<i>in one Python process</i> and showed a dramatic "divergence" (a run stopping at $t{=}0.005$ vs $0.3$). It was
a <b>harness artifact</b> — trinity's module-level global state leaks between two <code>start_expansion</code>
calls in the same process. The fix: run each variant in a <i>separate</i> <code>run.py</code> process.</div>
<div class="lesson"><b>The domain constraint that forced the right test.</b> The 60k grid is front-loaded with
points at sharp edges. Could a coarse grid step over a sharp $\min T$ dip and mis-fire the rejection guard —
converging the fsolve to a subtly wrong $\dot M$ that compounds over a full run? A per-call diff cannot see
this. <b>Only a full-run equivalence test can</b> — and it must include the stiffest regimes.</div>
__FIG_WORKFLOW__
<p>So three edge cases were added — <code>simple_cluster</code> and two physically-extreme clouds:
low-density / high-mass / high-sfe and high-density / high-mass / low-sfe (the stiff, weak-feedback case) — and
each was run end-to-end on the original 60k code and the F1 code, in <b>separate processes</b>, compared at
<b>matched simulation time</b> (the runs truncate at different $t$ because F1 is faster, so a raw final-state
diff is meaningless).</p>
__FIG_OVERLAY__
<p>The trajectories overlay perfectly — including through <code>edge_hidens</code>'s sharp energy drop at the
stiff transition. Quantitatively, the matched-$t$ relative difference stays <b>~500&times; inside the 0.3%
gate</b> across the entire common range:</p>
__FIG_MATCHED__
<table>
<tr><th>config</th><th>regime</th><th>common range</th><th>worst rel-diff (R2/Eb/rShell)</th></tr>
<tr><td><code>simple_cluster</code></td><td>—</td><td>[0, 4.54] Myr (251 pts)</td><td>5.7&times;10⁻⁸</td></tr>
<tr><td><code>edge_lowdens</code></td><td>diffuse / hi-M / hi-sfe</td><td>[0, 3.734] full</td><td>6.5&times;10⁻⁹</td></tr>
<tr><td><code>edge_hidens</code></td><td>dense / hi-M / lo-sfe (stiff)</td><td>[0, 0.052] full</td><td><b>6.0&times;10⁻⁶</b></td></tr>
</table>

<h3>4.6 Solution &amp; result</h3>
<p>The change is two hunks: a constant <code>_RESIDUAL_NPTS = 500</code> and the
<code>_get_velocity_residuals</code> body (drop <code>_create_radius_grid</code> + the
<code>_solve_bubble_structure</code> dense resample; one <code>solve_ivp(t_eval=linspace(r2Prime,R1,500))</code>).
Validated at every level: per-call (3.1e-6), 538 unit tests, full-run equivalence (6e-6), and the opt-in stress
gates (betadelta golden-match + bubble-solver 0-crash). <b>Result: ~1.5&times; per call; ~2.3&times; full-run on
<code>simple_cluster</code></b> (it spends the most wall-time in bubble calls).</p>
<div class="lesson"><b>Why it works (and why the 60k was never needed for accuracy):</b> <code>solve_ivp</code>'s
adaptive stepping (rtol $10^{-6}$) already resolves the stiff solution; the 60k was <i>output</i>
over-resolution. 500 points converge the $\min T$/monotonic guards to the same $\dot M$. <br><br>
<b>The meta-lesson:</b> a per-call equivalence gate is <i>necessary but not sufficient</i> for a change to a
residual — only a full-run equivalence test, on the stiffest regimes, can clear it.</p>

<h2>5. The arc, in one line</h2>
<p>The 60&nbsp;000-point grid went from a <b>load-bearing <code>odeint</code> integration grid</b> (Era A, when
it was needed for accuracy) → a <b>vestigial output grid</b> (after the <code>solve_ivp(dense_output)</code>
migration decoupled accuracy from sampling) → <b>removed</b> from the residual (F1), once a full-run equivalence
study proved the coarse sample reproduces the physics. Correctness first, then speed — and the speed was hiding
in plain sight in a number that had outlived its original reason for being.</p>

<p class="meta" style="margin-top:3rem">Sources: <code>docs/dev/performance/{RESAMPLE_PLAN,HOTPATH_PLAN,P3_PRODUCTION_PATCH,F1_SUMMARY}.md</code>,
<code>docs/dev/archive/bubble/{integrator-robustness,conduction-convergence}.md</code>,
<code>docs/dev/performance/data/*.csv</code>. Figures: <code>make_f1_figures.py</code>. Reproduce this report:
<code>python docs/dev/performance/harness/make_f1_report.py</code>.</p>

</body></html>
"""

HTML = HTML.replace("<!--PREV_WORK-->", PREV_WORK_HTML)
HTML = HTML.replace("__FIG_VARIANT__", img("f1_variant_tradeoff.png",
                    "Figure F1-1. The six options: every coarse grid beats the 60k baseline ~1.5x (speed flat across npts) and accuracy is npts-insensitive — so M500 is the conservative robustness pick."))
HTML = HTML.replace("__FIG_PERCALL__", img("f1_percall.png",
                    "Figure F1-2. Per-call speedup (~1.5x, config-independent) and accuracy (~1e-6, ~1000x inside the 0.3% gate) across the six sweep configs."))
HTML = HTML.replace("__FIG_WORKFLOW__", img("f1_workflow.png",
                    "Figure F1-3. The validation ladder P0->P1->P2->P3->P5->G3, and the lesson that the per-call gate (P2) is necessary but not sufficient."))
HTML = HTML.replace("__FIG_OVERLAY__", img("f1_fullrun_overlay.png",
                    "Figure F1-4. Full-run equivalence: R2(t) and Eb(t), original-60k (solid) vs F1-coarse (dashed), overlaid for the three edge cases. F1 runs further only because it is faster."))
HTML = HTML.replace("__FIG_MATCHED__", img("f1_matched_reldiff.png",
                    "Figure F1-5. Matched-t R2 relative difference vs the 0.3% gate — ~500x inside it across the full common range, including the stiff edge_hidens case."))

open(OUT, "w").write(HTML)
print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")
