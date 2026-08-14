#!/usr/bin/env python3
"""Build the self-contained decision dossier for the phase-1a fix.

Inlines the committed PNGs from ../figures as data URIs so the page needs no
network, and writes a single HTML file. The figures themselves come from
make_decision_figures.py, which reads only committed CSVs.

    python docs/dev/phase1a-init/harness/make_decision_figures.py
    python docs/dev/phase1a-init/harness/build_dossier.py [out.html]

Default output is <workstream>/decision_dossier.html.
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figures")
DEFAULT_OUT = os.path.join(HERE, "..", "decision_dossier.html")


def img(name):
    with open(os.path.join(FIGS, name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


HTML = """<title>Phase-1a early-time fix — decision dossier</title>
<style>
  :root {{
    --ground: #f6f7f7;
    --plate: #ffffff;
    --ink: #171c22;
    --ink-soft: #4a545f;
    --ink-faint: #78838f;
    --rule: #dfe3e4;
    --accent: #1f6b7c;        /* the converged / post-fix arm */
    --counter: #a84a30;       /* the artifact / pre-fix arm */
    --pass: #2f7048;
    --warn: #9a6714;
    --chip-bg: #eceff0;
    --measure: 68ch;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground: #10151a;
      --plate: #171d24;
      --ink: #e6ecef;
      --ink-soft: #a8b4bd;
      --ink-faint: #7d8993;
      --rule: #26303a;
      --accent: #5cb6c9;
      --counter: #dd8b6e;
      --pass: #64b98a;
      --warn: #d6a552;
      --chip-bg: #1e262e;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #10151a; --plate: #171d24; --ink: #e6ecef; --ink-soft: #a8b4bd;
    --ink-faint: #7d8993; --rule: #26303a; --accent: #5cb6c9; --counter: #dd8b6e;
    --pass: #64b98a; --warn: #d6a552; --chip-bg: #1e262e;
  }}
  :root[data-theme="light"] {{
    --ground: #f6f7f7; --plate: #ffffff; --ink: #171c22; --ink-soft: #4a545f;
    --ink-faint: #78838f; --rule: #dfe3e4; --accent: #1f6b7c; --counter: #a84a30;
    --pass: #2f7048; --warn: #9a6714; --chip-bg: #eceff0;
  }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 17px;
    line-height: 1.62;
    margin: 0;
    padding: 0 5vw 6rem;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; }}
  p, li {{ max-width: var(--measure); }}

  .eyebrow {{
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 11.5px; font-weight: 640; letter-spacing: .13em; text-transform: uppercase;
    color: var(--ink-faint); margin: 0 0 .45rem;
  }}
  h1 {{
    font-size: clamp(2rem, 4.4vw, 2.9rem); line-height: 1.12; margin: .2rem 0 .6rem;
    font-weight: 600; text-wrap: balance; letter-spacing: -.012em;
  }}
  h2 {{
    font-size: 1.42rem; margin: 3.4rem 0 .3rem; font-weight: 600;
    text-wrap: balance; letter-spacing: -.008em;
  }}
  h3 {{ font-size: 1.06rem; margin: 2rem 0 .2rem; font-weight: 640; }}
  .lede {{ font-size: 1.12rem; color: var(--ink-soft); max-width: 62ch; }}

  header.masthead {{ padding: 3.2rem 0 1.6rem; border-bottom: 2px solid var(--ink); }}
  .meta {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px; color: var(--ink-faint); display: flex; flex-wrap: wrap;
    gap: .4rem 1.6rem; margin-top: 1.2rem;
  }}

  /* the decision box — the reason this page exists */
  .decision {{
    background: var(--plate); border: 1px solid var(--rule);
    border-left: 4px solid var(--accent);
    padding: 1.5rem 1.7rem; margin: 2.4rem 0 1rem;
  }}
  .decision h2 {{ margin-top: 0; }}
  .options {{ display: grid; gap: .9rem; margin-top: 1.2rem; }}
  .option {{
    display: grid; grid-template-columns: auto 1fr; gap: .1rem .9rem;
    align-items: baseline; padding: .85rem 0; border-top: 1px solid var(--rule);
  }}
  .option .key {{
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: .1em; color: var(--accent);
    border: 1px solid var(--accent); border-radius: 2px; padding: .12rem .42rem;
    white-space: nowrap;
  }}
  .option p {{ margin: 0; max-width: 58ch; }}
  .option .conseq {{ grid-column: 2; color: var(--ink-soft); font-size: .93rem; margin-top: .25rem; }}

  figure {{ margin: 1.8rem 0 2.2rem; }}
  figure img {{
    display: block; width: 100%; height: auto; background: #fff;
    border: 1px solid var(--rule); border-radius: 3px;
  }}
  figcaption {{
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, sans-serif;
    font-size: 13.2px; line-height: 1.55; color: var(--ink-soft);
    margin-top: .7rem; max-width: 76ch;
  }}
  figcaption b {{ color: var(--ink); font-weight: 640; }}

  .scroll {{ overflow-x: auto; margin: 1.4rem 0 1.8rem; }}
  table {{
    border-collapse: collapse; font-family: ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
    font-size: 13.6px; width: 100%; min-width: 540px;
  }}
  th, td {{ text-align: left; padding: .52rem .8rem .52rem 0; border-bottom: 1px solid var(--rule); }}
  th {{
    font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 640; border-bottom: 1px solid var(--ink-faint);
  }}
  td.num {{ font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace; }}

  .chip {{
    display: inline-block; font-family: ui-sans-serif, -apple-system, sans-serif;
    font-size: 10.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    padding: .16rem .48rem; border-radius: 2px; background: var(--chip-bg); color: var(--ink-soft);
    white-space: nowrap;
  }}
  .chip.pass {{ color: var(--pass); box-shadow: inset 0 0 0 1px currentColor; background: transparent; }}
  .chip.fail {{ color: var(--counter); box-shadow: inset 0 0 0 1px currentColor; background: transparent; }}
  .chip.warn {{ color: var(--warn); box-shadow: inset 0 0 0 1px currentColor; background: transparent; }}

  code, .mono {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .87em; background: var(--chip-bg); padding: .08em .34em; border-radius: 2px;
  }}
  pre {{
    background: var(--plate); border: 1px solid var(--rule); border-radius: 3px;
    padding: 1rem 1.1rem; overflow-x: auto; font-size: 13px; line-height: 1.5;
  }}
  pre code {{ background: none; padding: 0; }}

  .callout {{
    border-left: 3px solid var(--counter); padding: .2rem 0 .2rem 1.1rem;
    margin: 1.6rem 0; color: var(--ink-soft);
  }}
  .callout strong {{ color: var(--ink); }}
  .callout.good {{ border-left-color: var(--pass); }}
  .callout.warn {{ border-left-color: var(--warn); }}

  hr.sec {{ border: 0; border-top: 1px solid var(--rule); margin: 3.4rem 0 0; }}
  footer {{ margin-top: 3.4rem; padding-top: 1.4rem; border-top: 2px solid var(--ink);
            font-size: 14px; color: var(--ink-soft); }}
  a {{ color: var(--accent); }}
  a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">TRINITY · docs/dev/phase1a-init · decided, ready to land</p>
  <h1>The phase-1a early-time fix, and what it costs</h1>
  <p class="lede">A hardcoded constant made every TRINITY run leave its first
  integration segment at exactly the same velocity, regardless of cloud mass,
  star-formation efficiency or density. The fix is implemented and gated. It
  also changes published-regime results in the first few thousand years — the
  part that needed a maintainer's sign-off, which it now has.</p>
  <div class="meta">
    <span>branch <b>hotfix/early-approximations</b></span>
    <span>fix: 0df441f + a944727</span>
    <span>evidence: data/gate_results.csv</span>
    <span>2026-08-05</span>
  </div>
</header>

<section class="decision">
  <p class="eyebrow">Decision taken — 2026-08-05</p>
  <h2>The bar was re-sited, and the shift accepted</h2>
  <p>This page was written to put one question to the maintainer: every gate
  asking <em>“is the new scheme internally correct?”</em> passed, but the
  pre-registered bar — <code>|ΔR₂| &lt; 1%</code> for t ≥ 3×10³ yr against stock
  — was missed on three of four configs, and it was recorded as failed rather
  than quietly moved. It has now been answered, with the numbers for both the
  old and the new bar in hand.</p>

  <p><b>Adopted bar:</b> <code>|ΔR₂| &lt; 5%</code> at 1 Myr — or at the end of
  the run if it terminates earlier — <b>and the stopping fate unchanged</b>. The
  fate clause is load-bearing: a loose radius threshold alone could pass a run
  that collapses when it should not, by comparing at its own truncated endpoint.
  All four configs pass; the worst is <code>f1edge_hidens</code> at +0.44%,
  11× inside. The three goldens pinned to the stock phase-1a exit state have
  been re-baselined against the new trajectory.</p>

  <p>Two things only became visible once both long configs were run to their
  <em>true</em> natural end — earlier fixed-arm GMC runs had been killed by an
  external SIGTERM at 8.2×10⁴ yr and misread as ending there:</p>

  <ul>
    <li><b>The trajectories converge, they do not merely stay in tolerance.</b>
    GMC control ΔR₂: −28.8% @100 yr → −0.95% @3×10³ → −0.28% @10⁴ → −0.037%
    @8×10⁴ → −0.002% @1 Myr → <b>−0.001% @2 Myr</b>, with Δv₂ +0.014% at 2 Myr.
    <code>simple_cluster</code> reaches −0.078% at 1 Myr. The disagreement is
    confined to the early transient — the part stock gets wrong.</li>
    <li><b>The fix is 16% faster</b> — 14m37s → 12m18s on
    <code>simple_cluster</code> to <code>stop_t=0.1</code>, each arm alone on
    the container. Almost all of it is in phase 1b, which the change does not
    touch: stock enters 1b at <code>v2_ODE/v2_alpha = 1.3167</code>, the fix at
    <code>1.0546</code>, and a 1a exit state already close to α-consistent is
    cheaper for 1b to continue from.</li>
  </ul>

  <p class="conseq">Recorded for the record: the GMC control passes the
  <em>original</em> bar too, at −0.949%, and is the only config that does — that
  bar was met by exactly the one scale <code>vd = -1e8</code> was tuned for.
  The threshold was adopted at 10% and tightened to 5% the same day, after the
  measurements and without re-running any of them: at 10% the bar sat ~23× above
  the worst config, loose enough to let a future regression through. The
  <em>form</em> of the bar — judged at 1 Myr / end of run, with the fate clause —
  was never in question.</p>
</section>

<hr class="sec">

<h2>The defect: an arithmetic result wearing a physical one’s clothes</h2>
<p>Phase 1a integrated in fixed 30-year segments, and for segment 0 it replaced
the shell’s computed acceleration with a hardcoded constant,
<code>vd = -1e8</code>. A constant right-hand side integrates <em>exactly</em>,
so the exit velocity had a closed form:</p>

<pre><code>v_exit = v0 − 1e8 × SEGMENT_DURATION = 3739.24 − 3000 = 739.24 pc/Myr = 722.82 km/s</code></pre>

<p>Because <code>v0 = 2L_w/ṗ_w</code> is mass-scale invariant, that is the same
number for <em>every</em> run on the bundled SB99 tables. The shell shed 80% of
its velocity in 30 years by arithmetic, not physics.</p>

<figure>
  <img src="{fingerprint}" alt="Segment-1 exit velocity plotted against cloud mass, flat at 722.82 km/s across four decades of mass.">
  <figcaption><b>Four decades of cloud mass, one velocity.</b> Each point is an
  independent run. The stock code leaves segment 1 at 722.82 km/s whether the
  cloud is 3×10³ or 3×10⁶ M<sub>☉</sub> — and the sub-GMC-scale probe (300 M<sub>☉</sub>,
  diamond) lands on the same value. Deleting the override alone does not fix it:
  the run then exits at 2429 km/s (grey square), a frozen-pressure snowplow.
  This invariance is the defect’s signature, and no solver-tolerance study can
  ever reveal it, because a constant RHS is integrated exactly.</figcaption>
</figure>

<div class="callout">
  <p><strong>Why tolerance sweeps missed it for years.</strong> Varying rtol and
  atol by 100× moves the answer by 2×10⁻¹². The error is perfectly reproducible,
  so every convergence check the code has ever run reported “converged”.</p>
</div>

<h2>The fix</h2>
<p>Segments now scale with the bubble’s own age, so the staleness of the frozen
driving terms is a constant <em>fraction</em> of the expansion time at every
object scale — instead of an absolute 30 years that is a small step for a giant
molecular cloud and the entire relaxation for a compact H II region.</p>

<pre><code>dt_segment = phase1a_segFrac × (t_now − tSF)      # default 0.1
                                                  # 0 restores the old fixed segment exactly</code></pre>

<p>The <code>vd = -1e8</code> override is deleted outright, along with the
<code>EarlyPhaseApproximation</code> flag that gated it — the flag was cleared
on only one of five phase-1a exit paths, so it could leak the constant into
phases 1b and 1c.</p>

<figure>
  <img src="{m43}" alt="compact-probe shell radius versus time, stock and fixed, against the observed radius and age.">
  <figcaption><b>The payoff, at sub-GMC scale.</b> Stock (rust) crosses the observed
  0.153 pc radius at 620 years — 21.8× too early — and the kink at 30 years is
  the segment-1 artifact itself. The fixed run (teal) reaches
  <b>0.196 pc at 5.09 km/s</b> at the observed age, against an observed
  0.153 pc and 5.0 km/s: the velocity is essentially exact and the radius is
  +28%, inside the wind-strength uncertainty of representing a single B star by
  an effective cluster.</figcaption>
</figure>

<h2>What passes</h2>
<div class="scroll">
<table>
  <thead><tr><th>Gate</th><th>What it proves</th><th>Result</th><th></th></tr></thead>
  <tbody>
    <tr><td>G1a</td><td>Schedule plumbing is inert when switched off</td>
        <td class="num">byte-identical, 124 rows</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>G1b</td><td>Override deletion changed exactly what it should</td>
        <td class="num">1×10⁻¹⁵ vs committed ablation</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>G2</td><td>Shipped code reproduces the converged reference</td>
        <td class="num">2.3×10⁻⁸ over 162+129 rows</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>G3</td><td>Trajectory obeys the Weaver similarity law</td>
        <td class="num">on-attractor from 1st decade</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>eps</td><td>Answer is converged, not a segment-count artifact</td>
        <td class="num">0.11% for 10× refinement</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>suite</td><td>Nothing else regressed</td>
        <td class="num">973 passed, ruff clean</td><td><span class="chip pass">pass</span></td></tr>
    <tr><td>G2 bar</td><td>Agreement with stock at t ≥ 3×10³ yr</td>
        <td class="num">−10.4% / +1.7% / −22.8%</td><td><span class="chip fail">missed</span></td></tr>
  </tbody>
</table>
</div>

<figure>
  <img src="{slope}" alt="Local logarithmic slope of radius versus time for stock and fixed runs, against the 3/5 law.">
  <figcaption><b>An independent observable, not another radius comparison.</b>
  Energy-driven expansion into a uniform medium obeys R ∝ t<sup>3/5</sup>. The
  fixed run sits on that slope from the first decade; stock plunges to 0.31
  immediately after segment 1 and needs ~10 kyr to climb back. Both settle
  together at 0.58 — the offset from exactly 0.600 is physical (the wind
  luminosity is not constant, and gravity and cooling contribute), and the fact
  that <b>both arms agree there</b> is what shows the fix moves the transient,
  not the asymptote.</figcaption>
</figure>

<figure>
  <img src="{eps}" alt="Radius at the observed age plotted against the segment fraction parameter, converging.">
  <figcaption><b>The result is a property of the physics, not of the step size.</b>
  Refining the segment fraction 10× (81 → 454 segments) moves the compact-probe radius at
  the observed age by 0.31%, then 0.11%. This is precisely the claim the old
  fixed-segment code could never make: its answer was a pure function of
  <code>SEGMENT_DURATION</code>.</figcaption>
</figure>

<hr class="sec">

<h2>What it costs — the part that needs your call</h2>
<p>This is the whole of the case against. On configs with published results, the
fixed and stock trajectories differ in the early phase, at matched simulation
time, in separate processes:</p>

<figure>
  <img src="{shift}" alt="Percent difference in radius between fixed and stock for three configurations, decaying with time.">
  <figcaption><b>The shift orders by core density — which is the mechanism, not a coincidence.</b>
  A denser cloud has a shorter dynamical time, so the absolute 30-year segment
  swallows a larger fraction of the relaxation and the artifact it injects is
  bigger. Note the signs differ, so this is not a uniform offset. Every curve
  decays monotonically into the 1% band: <b>lowdens by 10⁴ yr, simple_cluster by
  ~3.5×10⁴ yr</b>; hidens only in the last few percent before that cloud
  collapses.</figcaption>
</figure>

<div class="scroll">
<table>
  <thead><tr><th>Config</th><th>nCore</th><th class="num">3×10³ yr</th><th class="num">10⁴ yr</th><th class="num">3×10⁴ yr</th><th class="num">5×10⁴ yr</th><th>Fate</th></tr></thead>
  <tbody>
    <tr><td>f1edge_lowdens</td><td class="num">10²</td><td class="num">+1.70%</td><td class="num">+0.56%</td><td class="num">—</td><td class="num">—</td><td>unchanged</td></tr>
    <tr><td>simple_cluster</td><td class="num">10³</td><td class="num">−10.39%</td><td class="num">−3.35%</td><td class="num">−1.15%</td><td class="num">−0.66%</td><td>unchanged</td></tr>
    <tr><td>f1edge_hidens</td><td class="num">10⁶</td><td class="num">−22.75%</td><td class="num">−8.01%</td><td class="num">−1.96%</td><td class="num">—</td><td>same code, +28% later</td></tr>
  </tbody>
</table>
</div>

<div class="callout">
  <p><strong>Stopping fates do not change.</strong> Every config visits the same
  phases and ends on the same code. The one behavioural difference is on the
  stiffest edge, where the shell still collapses (<code>SHELL_COLLAPSED</code>)
  but 28% later, and from the transition phase rather than after entering
  momentum.</p>
</div>

<div class="callout good">
  <p><strong>The argument for accepting it.</strong> Stock is not a neutral
  reference in this window — it is the arithmetic artifact. Its slope is 0.31
  where theory says 0.6, and its segment-1 velocity is a constant independent of
  the cloud. The disagreement is largest exactly where stock is most wrong.</p>
</div>

<hr class="sec">

<h2>What it costs to run: nothing — it is cheaper</h2>
<p>“What it costs” was framed above as a question about trajectories. There is a
second reading, and it was worth measuring rather than assuming, because a
schedule that refines the early segments could plausibly have been slower. Each
arm was run <em>alone</em> on the container so the numbers are not contended,
and the times are each run’s own reported elapsed total, not an external timer.</p>

<figure>
  <img src="{cost}" alt="Left: stacked wall-clock bars for stock and fixed, split by phase 1a and 1b. Right: the ratio of ODE to alpha velocity at the phase handoff, for both arms.">
  <figcaption><b>16% faster, and the saving is not where the change is.</b>
  Phase 1a — the only phase this change touches — barely moves (2m26s → 2m18s,
  97 → 96 segments). Phase 1b, which is untouched, accounts for nearly all of
  it (12m11s → 10m00s). The right panel is the mechanism: phase 1b opens by
  iterating towards the α-consistent state, and the fix hands it a state already
  within 5% of it where stock hands over 32% away. A better phase-1a exit state
  is cheaper to continue from. Bars there are measured from 1.00 — a
  self-consistent handoff — so their length is the departure itself, not an
  axis-truncation effect.</figcaption>
</figure>

<hr class="sec">

<h2>A related constant, checked and cleared — with a warning</h2>
<p>The same audit that flags <code>vd = -1e8</code> also flags
<code>dt_switchon = 1e-3</code> Myr, which ramps the inner bubble radius over an
absolute time while phase 1a runs for 3×10⁻³ Myr — so it shapes the driving
pressure across the first third of the phase. It looked like the same defect
class, so I measured it.</p>

<figure>
  <img src="{e8b}" alt="Effect of removing the R1 ramp for two configurations, with the third annotated as stalled.">
  <figcaption><b>Small where it runs — and load-bearing where it doesn’t.</b>
  Removing the ramp shifts the radius by up to 5.8% early on, decaying to
  <b>0.006% at the compact probe's observed age</b>, well inside the noise floor. But on the
  stiffest config the run does not merely differ — it <b>stalls</b>: four
  snapshots in 90 minutes of wall clock, versus a complete run in minutes with
  the ramp in place. Removing the R1 suppression raises the bubble pressure, the
  structure ODE stiffens, and the solve stops converging.</figcaption>
</figure>

<div class="callout warn">
  <p><strong>The two constants are not the same class, and this is worth
  remembering.</strong> <code>vd = -1e8</code> papered over a
  <em>discretisation</em> error, and deleting it was measured safe.
  <code>dt_switchon</code> papers over genuine <em>stiffness</em>, and deleting
  it is fatal at high density. Had I judged from the source alone, both read as
  “uncalibrated magic number, remove it”. Note also that the two cheap configs
  would have given a confidently wrong answer here; only the stiffest edge
  revealed the constant is holding the solver up.</p>
</div>

<hr class="sec">

<h2>Reproducing any number on this page</h2>
<p>Every figure is generated from committed CSVs, with no simulation runs:</p>
<pre><code>python docs/dev/phase1a-init/harness/make_decision_figures.py   # figures
python docs/dev/phase1a-init/harness/build_dossier.py           # this page
python docs/dev/phase1a-init/harness/g3_slopes.py               # the slope table</code></pre>
<p>The full ledger — every gate, its pre-registered bar, the measured value and
the verdict — is <code>docs/dev/phase1a-init/data/gate_results.csv</code>. Each
trajectory CSV carries the exact config and command in its provenance header.
The reasoning and the open follow-ups are in <code>PLAN.md</code> §§4, 8–9.</p>

<footer>
  <p>The goldens are done: <b>three</b>, not the two this page originally said —
  <code>test_run_smoke</code> and <code>test_phase_boundary</code> in the default
  suite, plus the <code>-m stress</code>
  <code>test_betadelta_hybr_stress</code>. All three pinned the stock phase-1a
  exit state; re-baselined 2026-08-05, each site keeping the superseded values
  and the reason they move. Default suite: 987 passed, 0 failed.</p>
  <p>Nothing left deliberately undone. Magic-number audit finding #4
  (<code>vd = -1e8</code>) is marked fixed in
  <code>docs/dev/magic-numbers/AUDIT.md</code> — closed <em>in</em> this branch
  rather than after it lands, so the entry becomes true at exactly the moment
  the deletion merges. The same pass corrected that audit's finding #2
  (<code>dt_switchon</code>), whose recommendation — <em>“if inert,
  delete”</em> — E8b showed to be the wrong fix: the ramp is inert by that test
  and deleting it still stalls <code>f1edge_hidens</code> outright.</p>
  <p>Out of scope by decision, both recorded in <code>PLAN.md</code>: a
  scale-relative successor to the <code>dt_switchon</code> ramp (§8 E8b), and
  the multi-config scheme screen the repo still lacks (§9).</p>
</footer>

</div>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    html = HTML.format(
        fingerprint=img("decision_fingerprint.png"),
        m43=img("decision_m43.png"),
        shift=img("decision_shift.png"),
        slope=img("decision_slope.png"),
        eps=img("decision_eps.png"),
        e8b=img("decision_e8b.png"),
        cost=img("decision_cost.png"),
    )
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {os.path.normpath(out)}  ({len(html)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
