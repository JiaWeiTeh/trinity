"""Build a single self-contained HTML report telling the full failed-large-clouds story.

Reads the three committed PNGs (figures/) and the budget CSVs (data/), base64-embeds the
images so the file is portable/downloadable, and renders the math with MathJax (CDN).

Reproduce:
  python docs/dev/failed-large-clouds/figures/make_insights_html.py
  -> writes docs/dev/failed-large-clouds/insights.html
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "insights.html")


def img(name):
    with open(os.path.join(HERE, name), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


FIG1, FIG2, FIG3 = img("fig1_dEbdt_budget.png"), img("fig2_healthy_vs_failing.png"), img("fig3_bug_and_fix.png")
FIG4 = img("fig4_energy_driven_discriminator.png")

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRINITY — Failed Large Clouds: diagnosis &amp; fix</title>
<script>
MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root{--ink:#1c2330;--mut:#5b6678;--line:#e3e7ee;--accent:#d95f02;--green:#1b9e77;--purple:#7570b3;--bg:#fbfcfe;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:880px;margin:0 auto;padding:48px 28px 96px;}
  h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em}
  .sub{color:var(--mut);font-size:15px;margin:0 0 4px}
  h2{font-size:22px;margin:52px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
  h3{font-size:17px;margin:30px 0 8px}
  p{margin:12px 0}
  code{background:#eef1f6;padding:1.5px 5px;border-radius:4px;font-size:13.5px;
       font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}
  pre{background:#1c2330;color:#e6e9ef;padding:14px 16px;border-radius:9px;overflow:auto;font-size:13px;line-height:1.5}
  pre code{background:none;color:inherit;padding:0}
  table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}
  th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
  th{background:#f1f4f9;font-weight:600}
  tr:nth-child(even) td{background:#f7f9fc}
  figure{margin:26px 0;text-align:center}
  figure img{max-width:100%;border:1px solid var(--line);border-radius:9px;box-shadow:0 2px 12px rgba(20,30,50,.06)}
  figcaption{color:var(--mut);font-size:13.5px;margin-top:9px;text-align:left}
  .lead{font-size:17px;color:#33405a}
  .callout{border-left:4px solid var(--accent);background:#fff7f0;padding:12px 18px;border-radius:0 8px 8px 0;margin:18px 0}
  .callout.ok{border-color:var(--green);background:#f0faf6}
  .callout.warn{border-color:#c9a400;background:#fffbe9}
  .pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;
        background:#eef1f6;color:#445;margin-right:6px}
  .pill.ok{background:#d8f3e8;color:#0f6b48}.pill.bad{background:#fde0dc;color:#9b271b}
  .meta{color:var(--mut);font-size:13px;border-top:1px solid var(--line);margin-top:60px;padding-top:18px}
  .tag{font-family:"SF Mono",ui-monospace,monospace;font-size:12.5px;color:var(--accent)}
  ol.corr{counter-reset:c;list-style:none;padding-left:0}
  ol.corr li{position:relative;padding:10px 0 10px 44px;border-bottom:1px dashed var(--line)}
  ol.corr li:before{counter-increment:c;content:counter(c);position:absolute;left:0;top:10px;width:28px;height:28px;
       background:var(--accent);color:#fff;border-radius:50%;text-align:center;line-height:28px;font-weight:700;font-size:14px}
  .was{color:#9b271b}.now{color:#0f6b48;font-weight:600}
  .nowrap{white-space:nowrap}
</style>
</head>
<body>
<div class="wrap">

<h1>Failed large clouds — why the energy phase crashed, and the fix</h1>
<p class="sub">TRINITY feedback-bubble code · energy-driven phase · diagnosis &amp; minimal robustness fix</p>
<p class="sub"><span class="tag">docs/dev/failed-large-clouds/</span> · point-in-time analysis (2026-06-19)</p>

<p class="lead">Massive giant molecular clouds crashed TRINITY's energy phase with
<code>Eb=nan</code> / <code>R1 root finding failed</code>. This report walks the full arc: the symptom,
the numerical mechanism, the <em>physical</em> reason it only hits massive clouds, the ideas we tested,
the one we shipped across every affected config, and the honest list of claims we had to revise along the way.</p>

<div class="callout ok">
<span class="pill ok">RESOLVED</span> All three failing configs now terminate cleanly
(<code>ENERGY_COLLAPSED</code>, code&nbsp;51); healthy configs are a <strong>byte-identical no-op</strong>;
<code>pytest</code> 555&nbsp;passed. Fix = <strong>G</strong> (geometry guard) + <strong>F</strong>
(graceful collapse termination), ~93 lines across 4 files + a 6-case test.
</div>

<h2>1 · The problem</h2>
<p>For a giant-molecular-cloud mass <code>mCloud</code>, star-formation efficiency <code>sfe</code>, and
ambient density <code>nCore</code>, TRINITY integrates an expanding feedback bubble through its phases. A
band of <strong>massive, dense</strong> clouds — e.g. <code>mCloud=5e9</code>, <code>sfe=0.1</code>,
<code>nCore=1e2</code>, and the real Helix point (<code>sfe=0.05</code>, <code>PISM=0</code>) — did not just
give a wrong answer. They <strong>crashed</strong>: a <code>ValueError</code> traceback out of the
bubble-pressure solve, or silent <code>nan</code> propagating into the saved outputs.</p>
<p>A crash on a physically-plausible input is a correctness bug: a sweep over cloud masses hits these cells
and dies, or worse, records <code>nan</code> and continues.</p>

<h2>2 · The mechanism — how a crash falls out of the math</h2>
<p>The interior bubble pressure is set by the thermal energy <span class="nowrap">$E_b$</span> divided by the
hot-gas (shell) volume between the wind shock <span class="nowrap">$R_1$</span> and the outer shell
<span class="nowrap">$R_2$</span>:</p>
$$P_b = (\gamma-1)\,\frac{E_b}{V},\qquad V=\frac{4\pi}{3}\left(R_2^{3}-R_1^{3}\right).$$
<p>When a run drives <span class="nowrap">$E_b \to 0$</span>, the inner-shock solver
(<code>solve_R1</code>) pushes <span class="nowrap">$R_1 \to R_2$</span>, so the shell volume
collapses, <span class="nowrap">$V \to 0$</span>, and</p>
$$P_b=(\gamma-1)\frac{E_b}{V}\;\longrightarrow\;\frac{0}{0}\;\longrightarrow\;\text{NaN / }\tfrac{1}{0},$$
<p>which then poisons the next solve (a <span class="nowrap">$\sqrt{R_2}$</span> with
<span class="nowrap">$R_2&lt;0$</span> in <code>get_r1</code>, or a cooling-table out-of-bounds at
<span class="nowrap">$E_b\approx 0$</span>) and
raises. Figure&nbsp;3 shows the geometry of the collapse and where the old code died versus where the fix
now stops.</p>

<figure>
  <img src="__FIG3__" alt="Eb collapse and the fix">
  <figcaption><strong>Fig 3 — the crash and the fix.</strong> Top: bubble energy $E_b(t)$ peaks then falls
  through zero. Bottom: as $E_b\to0$ the inner shock $R_1$ converges onto the shell $R_2$, the shell volume
  $R_2^3-R_1^3\to0$, and $P_b=(\gamma-1)E_b/V\to1/0$. <span class="was">✗ OLD</span>: <code>ValueError</code>
  crash. <span class="now">✓ NEW</span>: detect $E_b\le0$ and stop cleanly with <code>ENERGY_COLLAPSED</code>.</figcaption>
</figure>

<h2>3 · The real question — <em>why</em> does $E_b\to0$ here?</h2>
<p>Stopping the divide-by-zero treats the symptom. The diagnosis needed the cause: what term in the energy
budget drains <span class="nowrap">$E_b$</span>? The energy-phase ODE
(<code>energy_phase_ODEs.py</code>) is</p>
$$\frac{dE_b}{dt}=\underbrace{L_{\rm mech}}_{\text{wind+SN in}}
-\underbrace{L_{\rm cool}}_{\text{radiative}}
-\underbrace{4\pi R_2^{2}\,P_b\,v_2}_{\text{PdV work on the shell}}
-\underbrace{L_{\rm leak}}_{\text{conduction}}.$$
<p>The first hypothesis was <em>catastrophic cooling</em>. Decomposing the budget from the run snapshots
falsified it: cooling is <strong>~1%</strong> of the mechanical input
(<span class="nowrap">$L_{\rm cool}/L_{\rm mech}\approx0.01$</span>), while the <strong>PdV expansion work</strong>
climbs from 0.52 to 1.56 of <span class="nowrap">$L_{\rm mech}$</span> and crosses unity exactly at the
<span class="nowrap">$E_b$</span> peak. <strong>The bubble is doing more expansion work than feedback puts
in</strong> — so its energy peaks and collapses.</p>

<figure>
  <img src="__FIG1__" alt="dEb/dt energy budget">
  <figcaption><strong>Fig 1 — the budget of $dE_b/dt$ for the failing run.</strong> Top: PdV work (orange)
  overtakes the mechanical input $L_{\rm mech}$ (green); radiative cooling (purple) is ~1% of input. Bottom:
  loss terms relative to $L_{\rm mech}$ — $\text{PdV}/L_{\rm mech}$ crosses 1 (net loss ⇒ $E_b$ collapses);
  $L_{\rm cool}/L_{\rm mech}\approx0.01$.</figcaption>
</figure>

<h2>4 · Why only massive clouds — healthy vs failing</h2>
<p>The discriminator is the <strong>shell velocity</strong>, since
<span class="nowrap">$\text{PdV}\propto v_2$</span>. A massive, concentrated cluster launches the shell at
~2000–3700&nbsp;km/s — near free-expansion (<span class="nowrap">$R\approx v t$</span>), out of the
self-similar Weaver equilibrium where PdV is a fixed sub-unity fraction of <span class="nowrap">$L_{\rm
mech}$</span>. A healthy cloud stays Weaver-like: <span class="nowrap">$\text{PdV}/L_{\rm mech}&lt;1$</span>,
the shell decelerates, and <span class="nowrap">$E_b$</span> grows.</p>

<figure>
  <img src="__FIG2__" alt="healthy vs failing">
  <figcaption><strong>Fig 2 — why the massive cloud dies and a healthy one does not.</strong> Both anchored
  at elapsed time since the energy phase began. <span style="color:#d95f02">Massive</span>:
  $\text{PdV}/L_{\rm mech}$ crosses 1, $v_2$ stays ~2000+ km/s, $E_b$ collapses after ~$10^{-3}$&nbsp;Myr.
  <span style="color:#1b9e77">Healthy</span>: $\text{PdV}/L_{\rm mech}\le0.95$ and declining, $v_2$
  decelerates, $E_b$ grows for ~0.3&nbsp;Myr. The dotted segment is the initial-condition relaxation, where
  the per-snapshot PdV proxy is uncertain near break-even.</figcaption>
</figure>

<h3>A verified side-result: the massive cloud's energy phase starts ~70× later</h3>
<p>With <span class="nowrap">$t_{\rm SF}=0$</span> (logged), the energy phase begins at
<span class="nowrap">$t_0=\Delta t_{\rm fs}$</span>, the free-streaming duration:</p>
$$t_0=\Delta t_{\rm fs}=\sqrt{\frac{3\dot M}{4\pi\,\rho_a\,v_0^{3}}}\;\propto\;\sqrt{M_{\rm cluster}}$$
<p>because <span class="nowrap">$\rho_a$</span> (same <code>nCore</code>) and the wind terminal velocity
<span class="nowrap">$v_0=2L_w/\dot p_w=3739\,$pc/Myr</span> (mass-independent) are shared, while
<span class="nowrap">$\dot M\propto M_{\rm cluster}$</span> (logged <span class="nowrap">$\dot M_{\rm
wind}$</span>: 1.451 vs <span class="nowrap">$2.901\times10^{-4}\,M_\odot$/yr</span> = ratio 5000). So the
<span class="nowrap">$t_0$</span> ratio is <span class="nowrap">$\sqrt{5000}=70.71$</span>, matching the
logged <span class="nowrap">$1.383\times10^{-3}/1.956\times10^{-5}=70.71$</span> exactly. The same scaling
explains the IC pressure: <span class="nowrap">$P_{b,0}\propto E_0/r_0^3\propto L_w^2\rho_a/\dot
p_w^2\propto\rho_a=n_{\rm core}\mu$</span> — set by ambient density and the wind, not by cluster mass.</p>

<h3>Were they ever genuinely energy-driven? (measured, 5 configs)</h3>
<p>This is the question that decides whether including PdV in a transition trigger is physically natural or a
band-aid. The cleanest signal is the <strong>reservoir growth</strong>
<span class="nowrap">$E_b^{\rm peak}/E_b^{\rm init}$</span> — a pure state variable, fully reliable. Healthy
controls build the hot-bubble thermal reservoir by <strong>$\ge\times$39,300–94,900</strong> (lower bounds —
still energy-driven at the $t\le1$ Myr window cap); all three failing
configs build it by <strong>$\times$1.014 (~1%)</strong>. The energy-driven reservoir essentially
<strong>never forms</strong>. The nuance: it is <em>not</em> "PdV&gt;1 from birth" — every config starts at the
same self-similar handoff <span class="nowrap">$\text{PdV}/L_{\rm mech}\approx0.5$</span>. The fork is the
<strong>direction</strong>: healthy clouds decelerate, so PdV/$L_{\rm mech}$ falls and the reservoir builds;
failing clouds never decelerate, so PdV/$L_{\rm mech}$ rises through 1 within ~7–10% of the phase and
<span class="nowrap">$E_b$</span> collapses.</p>

<figure>
  <img src="__FIG4__" alt="were they ever energy-driven">
  <figcaption><strong>Fig 4 — were the failing clouds ever energy-driven?</strong> Left: reservoir growth
  $E_b/E_{b,\rm init}$ — healthy (dashed) climb to $\sim$10$^4$, failing (solid) flatline at ~1 then collapse.
  Right: PdV/$L_{\rm mech}$ — failing rise through break-even, healthy fall and stay below. All start at the
  same ~0.5 handoff.</figcaption>
</figure>

<div class="callout">
<strong>Interpretation.</strong> The failing clouds are "stillborn" energy-driven bubbles: they inherit the
energy-driven initial condition but never establish the self-similar deceleration that <em>defines</em> the
phase. So a PdV-inclusive transition trigger would be detecting "this bubble failed to become energy-driven" —
a <strong>regime mismatch</strong>, not a healthy Weaver→momentum transition. That argues the deeper-correct fix
is to recognise these as free-expansion/momentum-dominated <em>earlier</em>, not to bolt a PdV term onto the
cooling-based transition test.</div>

<h2>5 · Ideas tested</h2>
<p>Two layers were kept distinct: cheap <em>numeric guards</em> (does just stopping the divide-by-zero
suffice?) and the <em>fix families</em> a production change would choose from.</p>

<h3>Numeric-guard variants (monkeypatched in the harness)</h3>
<table>
<tr><th>id</th><th>patch</th><th>hypothesis</th></tr>
<tr><td><code>V0</code></td><td>baseline (no patch)</td><td>crashes on the band (reference)</td></tr>
<tr><td><code>V1</code></td><td>clamp $R_1\le R_2(1-\varepsilon)$ in <code>solve_R1</code></td><td>kills the <code>inf</code></td></tr>
<tr><td><code>V2</code></td><td>floor shell volume $R_2^3-R_1^3\ge\varepsilon R_2^3$ in <code>bubble_E2P</code></td><td>same, via the divide site</td></tr>
<tr><td><code>V3</code></td><td>V1 + V2</td><td>combined guard</td></tr>
</table>

<h3>Fix families</h3>
<table>
<tr><th>id</th><th>family</th><th>what it changes</th><th>role</th></tr>
<tr><td><code>G</code></td><td>geometry guard</td><td>$R_1&lt;R_2$ / volume floor so the divide can't blow up</td><td>necessary safety net — <strong>not sufficient alone</strong></td></tr>
<tr><td><code>C</code></td><td>cancellation-free $P_b$</td><td>the <code>get_r1</code> identity removes the catastrophic cancellation at its source</td><td>optional conditioning</td></tr>
<tr><td><code>F</code></td><td>loud / graceful fail</td><td><code>isfinite</code> gate ⇒ clean termination with a reason</td><td>belt-and-suspenders for any NaN source</td></tr>
<tr><td><code>T</code></td><td>transition (deferred)</td><td>detect the PdV-inclusive net-energy zero-crossing ⇒ hand off to the momentum phase</td><td>the physically-complete end-state</td></tr>
</table>

<div class="callout warn">
<strong>Key empirical result — geometry guard alone is NOT enough.</strong> Smoke-testing <code>V3</code> on
<code>fail_repro</code>: with the geometry clamped, the ODE keeps integrating and $E_b$ crosses
<em>through zero into negative</em> (<span class="nowrap">$+7.4\times10^{8}\to-9.1\times10^{8}\to-1.0\times10^{12}$</span>),
giving negative $P_b$; the bubble solve then has no physical root, <code>fsolve</code> thrashes, and the run
never terminates (SIGTERM'd at 320&nbsp;s). So a guard must be paired with a graceful stop (<code>F</code>).</div>

<h2>6 · One idea across configs</h2>
<p>The shipped <code>G+F</code> fix was exercised across the whole affected band and the healthy controls.
The Helix point collapses <em>inside</em> phase&nbsp;1a (where per-segment solves raise <em>before</em> the
end-of-loop check), which forced a second round of coverage: <code>solve_R1</code> returns
<code>0.0</code> for non-physical <span class="nowrap">$R_2\le0$</span>, and the phase-1a bubble solve is
wrapped so any degenerate-collapse exception becomes a clean <code>ENERGY_COLLAPSED</code>.</p>
<table>
<tr><th>config</th><th>collapse phase</th><th>crashed</th><th>end state</th><th>final $R_2$ [pc]</th></tr>
<tr><td><code>fail_repro</code> (sfe0.1 / PISM1e4)</td><td>1b</td><td><span class="pill ok">False</span></td><td><code>ENERGY_COLLAPSED</code> (51)</td><td>9.73</td></tr>
<tr><td><code>fail_pism6</code> (sfe0.1 / PISM1e6)</td><td>1b</td><td><span class="pill ok">False</span></td><td><code>ENERGY_COLLAPSED</code> (51)</td><td>9.73</td></tr>
<tr><td><strong><code>fail_helix</code></strong> (real Helix: sfe0.05 / PISM0)</td><td>1a</td><td><span class="pill ok">False</span></td><td><code>ENERGY_COLLAPSED</code> (51)</td><td>7.03</td></tr>
<tr><td><code>small_1e5</code> / <code>small_1e6</code> (healthy)</td><td>—</td><td><span class="pill ok">False</span></td><td><strong>no-op (byte-identical to pre-fix)</strong></td><td>—</td></tr>
</table>

<h2>7 · Performance &amp; gates</h2>
<table>
<tr><th>gate</th><th>bar</th><th>result</th></tr>
<tr><td>Robustness</td><td><code>crashed=False</code> on the whole failing band</td><td><span class="pill ok">✅</span> all stop with code 51</td></tr>
<tr><td>No-op</td><td>healthy outputs identical to pre-fix</td><td><span class="pill ok">✅</span> byte-identical <code>dictionary.jsonl</code></td></tr>
<tr><td>Unit</td><td>guard fires only when degenerate; pinned</td><td><span class="pill ok">✅</span> 6/6, incl. a bit-identity pin</td></tr>
<tr><td>Regression</td><td>full suite green</td><td><span class="pill ok">✅</span> <code>pytest</code> 555 passed</td></tr>
</table>
<p>The volume floor is written so its branch is <strong>dead while the volume is positive</strong> — i.e.
provably bit-identical for every physical bubble — and that invariant is pinned by
<code>test_bubble_E2P_bit_identical_when_volume_positive</code>. There is no runtime cost on healthy runs.</p>

<h2>8 · The solution</h2>
<p><strong>Shipped: G + F.</strong> Family <code>T</code> (momentum handoff) is deferred to
<code>docs/dev/transition/</code> per maintainer decision; until then collapsed points are cleanly flagged
for sweeps to filter.</p>
<ol>
<li><strong>G — <code>bubble_E2P</code> volume floor</strong> (<code>get_bubbleParams.py:226-235</code>):
  <code>if (r2³−r1³) &lt;= 0: shell_volume = 1e-13·r2³</code>. Stops the divide-by-zero at its source.</li>
<li><strong>F — collapse ⇒ clean stop</strong> in both energy phases (<code>run_energy_phase.py</code>,
  <code>run_energy_implicit_phase.py</code>): when <code>not isfinite(Eb) or Eb &lt;= 0</code>, raise
  <code>EndSimulationDirectly</code> with the new <code>SimulationEndCode.ENERGY_COLLAPSED</code>
  (<code>(51,"energy_collapsed")</code>, the 50–59 "inspection-required" band). <code>main.py</code> then
  skips the later phases via the existing gate.</li>
<li><strong>Phase-1a coverage</strong>: <code>solve_R1</code> returns <code>0.0</code> for non-physical
  <span class="nowrap">$R_2\le0$</span>, and the phase-1a bubble solve is wrapped so a degenerate-collapse
  exception becomes <code>ENERGY_COLLAPSED</code> rather than a traceback.</li>
</ol>
<p>Total: ~93 lines across 4 files + <code>test/test_energy_collapse_guard.py</code> (6 cases). The
<code>cooling_balance</code> transition is untouched.</p>

<h2>9 · Corrections log</h2>
<p>Four assertions were made and then revised during the analysis — each settled by a specific
source or measurement, recorded here so the originals are not re-trusted.</p>
<ol class="corr">
<li><span class="was">"The failure is catastrophic cooling."</span> →
  <span class="now">PdV expansion work, not cooling.</span> Settled by the $dE_b/dt$ budget (Fig 1):
  $L_{\rm cool}/L_{\rm mech}\approx0.01$; $\text{PdV}/L_{\rm mech}$ runs 0.52→1.56, crossing 1 at the $E_b$ peak.</li>
<li><span class="was">"Snapshot 0's $P_b$ is a fixed seed/placeholder."</span> →
  <span class="now">The genuine Weaver initial condition,</span> computed
  $P_b=\texttt{bubble\_E2P}(E_0,r_0,R_1)$ (<code>run_energy_phase.py:97-100</code>).</li>
<li><span class="was">"$P_{b,0}$ is bit-identical across the two clouds."</span> →
  <span class="now">≈equal to ~6 sig figs</span> ($2.135768\times10^7$ vs $2.135766\times10^7$): $P_{b,0}\propto n_{\rm core}$
  and both share <code>nCore=1e2</code>, so it is <em>near</em>-equal, not bit-identical.</li>
<li><span class="was">"The healthy run starts later (a delay)."</span> →
  <span class="now">A plotting artifact.</span> <code>reliable_mask</code> had trimmed the $v_2$/$E_b$ state
  too; both runs' snap&nbsp;1 sits at the same elapsed $t-t_0\approx3\times10^{-5}$&nbsp;Myr. Fixed by plotting
  state at every snapshot and flagging only the PdV proxy.</li>
</ol>

<h2>10 · Reproduce</h2>
<pre><code># failing repro + healthy control (ephemeral runs)
python run.py docs/dev/failed-large-clouds/harness/params/fail_repro.param
python run.py docs/dev/failed-large-clouds/harness/params/small_1e6.param

# figures + this report (from the committed CSVs / PNGs, no re-run needed)
python docs/dev/failed-large-clouds/figures/make_energy_budget_figs.py
python docs/dev/failed-large-clouds/figures/make_insights_html.py

# tests
pytest test/test_energy_collapse_guard.py</code></pre>

<p class="meta">TRINITY (<code>trinity-sf</code>) · feedback-driven bubble-evolution code ·
report generated by <code>figures/make_insights_html.py</code> from committed artifacts under
<code>docs/dev/failed-large-clouds/</code>. Math via MathJax (needs a network connection to render).
This is a point-in-time analysis; line numbers may drift — re-verify against current source.</p>

</div>
</body>
</html>
"""

HTML = (HTML.replace("__FIG1__", FIG1).replace("__FIG2__", FIG2)
        .replace("__FIG3__", FIG3).replace("__FIG4__", FIG4))

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT}  ({len(HTML)/1024:.0f} KB)")
