"""Build a single self-contained HTML report for the pt4 heavy-cloud experiments (H3, H4).

This is the part-4 chapter of storyline s3 ("Why large clouds failed (helix)"). It follows on
from "Failed large clouds — diagnosis & fix": for the heaviest clouds the bubble energy Eb is
drained through zero by PdV expansion work and now terminates cleanly as ENERGY_COLLAPSED. This
report asks whether those clouds can be kept alive, and what the two attempts (H3 Eb-floor, H4
PdV-cap) teach us about the regime.

Reads the committed PNGs (figures/), base64-embeds them so the file is portable/downloadable, and
renders the math with MathJax (CDN). Numbers are quoted verbatim from the committed pt4 dev docs
(H3_eb_floor_experiment.md, H4_pdvcap_experiment.md) — none invented.

Reproduce:
  python docs/dev/transition/pt4/make_pt4_heavy_report.py
  -> writes docs/dev/transition/pt4/pt4_heavy_report.html
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "pt4_heavy_report.html")


def img(name):
    with open(os.path.join(FIGDIR, name), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


FIG_H3 = img("h3_ebfloor_noop_and_grind.png")
FIG_H4_EB = img("h4_Eb_sweep_fail_repro.png")
FIG_H4_RATIO = img("h4_pdvratio_sweep_fail_repro.png")
FIG_H4_SUMMARY = img("h4_summary.png")
FIG_H4_CONTROL = img("h4_control_vs_collapse.png")

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRINITY — Heavy clouds, part 4: can we keep them alive?</title>
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
  .was{color:#9b271b}.now{color:#0f6b48;font-weight:600}
  .nowrap{white-space:nowrap}
  .banner{border-left:4px solid #c9a400;background:#fffbe9;padding:10px 16px;border-radius:0 8px 8px 0;
          margin:14px 0;font-size:13px;color:#4a4630}
</style>
</head>
<body>
<div class="wrap">

<h1>Heavy clouds, part 4 — can we keep them alive, and what do the attempts teach us?</h1>
<p class="sub">TRINITY feedback-bubble code · energy-driven phase · two diagnostic rescue experiments (H3, H4)</p>
<p class="sub"><span class="tag">docs/dev/transition/pt4/</span> · point-in-time analysis (2026-06-22) · branch <code>fix/transition-trigger-problem-pt4</code></p>

<div class="banner">
&#9888; <strong>These are point-in-time diagnostics, not a maintained spec.</strong> The source dev
docs (<code>H3_eb_floor_experiment.md</code>, <code>H4_pdvcap_experiment.md</code>) carry the full
three-paragraph staleness / living-plan / persist-diagnostics banners; treat every number and line
reference here as unverified and re-check against current source before relying on it. Both
experiments are <strong>monkeypatch-only — production code is untouched.</strong>
</div>

<p class="lead">The previous chapter diagnosed why the heaviest clouds (<code>mCloud=5e9</code>) crash
the energy phase: the bubble's thermal energy <span class="nowrap">$E_b$</span> is drained by
<strong>PdV expansion work</strong> faster than feedback supplies it, so it peaks, collapses through
zero, and now terminates cleanly as <code>ENERGY_COLLAPSED</code>. This part&nbsp;4 asks the natural
follow-up: <strong>can we keep these clouds alive?</strong> Two maintainer hypotheses were tested as
controlled, energy-injecting <em>diagnostics</em> — floor <span class="nowrap">$E_b$</span> (H3), or
cap the drain (H4). Neither can ship; both sharpen the diagnosis.</p>

<div class="callout">
<strong>The control variable.</strong> Everything below tracks one ratio,
$$\frac{\rm PdV}{L_{\rm mech}}=\frac{4\pi R_2^{2}\,P_b\,v_2}{L_{\rm mech}},$$
the expansion work the bubble does on its shell, relative to the mechanical (wind+SN) input. Above 1
the bubble loses energy net and <span class="nowrap">$E_b$</span> falls; below 1 it grows. The whole
heavy-cloud story is this ratio sitting above 1 for an extended early epoch.</div>

<h2>1 · H3 — floor $E_b>0$ so the bubble keeps expanding?</h2>
<p>The first idea is the most direct: if the run dies because <span class="nowrap">$E_b$</span>
collapses through zero, what if we simply <strong>floor <span class="nowrap">$E_b$</span> to a small
positive value</strong> so it never can? The <code>EBFLOOR</code> variant clamps
<span class="nowrap">$E_b=\max(E_b,\,10^{-3})$</span> at every consumer (so <span class="nowrap">$P_b>0$</span>,
<span class="nowrap">$R_1$</span> well-defined) and adds a <em>reflecting</em> floor on
<span class="nowrap">$dE_b/dt$</span> so the integrated state can only hold flat or grow at the
floor. This <strong>injects energy and violates conservation</strong> — it is a diagnostic to isolate
<em>whether "$E_b\to0$" is the sole failure mode</em>, not a fix candidate.</p>

<div class="callout warn">
<span class="pill bad">REFUTED</span> <strong>$E_b$-collapse is NOT the sole failure mode.</strong>
Forcing <span class="nowrap">$E_b>0$</span> does not make either heavy cloud behave like a healthy
one. The floor is a <strong>bit-identical no-op on 11 of 13</strong> healthy/stall configs
(<span class="nowrap">$E_b$</span> never collapses there, so the branch is never taken); the other two
differ only at the <span class="nowrap">$\sim$10$^{-8}$</span> level. On the
<span class="nowrap">$5\times10^{9}\,M_\odot$</span> clouds it rescues neither — it exposes a
<em>second</em> guard and a non-convergent grind instead.</div>

<h3>What the floor exposes on the two real-collapse configs</h3>
<p>Only the two <span class="nowrap">$5\times10^{9}\,M_\odot$</span> runs actually collapse at these
early times (<code>mass_5e8</code>/<code>mass_1e9</code> stay healthy with
<span class="nowrap">$E_b>0$</span> in the window) — and they fail by <strong>two different shipped
paths</strong> that the floor defeats neither:</p>
<table>
<tr><th>config</th><th>V0 (baseline)</th><th>EBFLOOR</th><th>what the floor exposes</th></tr>
<tr>
  <td><code>fail_repro</code><br>(<code>mCloud=5e9</code>, <code>PISM=1e4</code>, 1b)</td>
  <td><code>ENERGY_COLLAPSED</code> (51): one implicit step drives
      <span class="nowrap">$E_b:4.75\times10^{8}\to-9.14\times10^{8}$</span>, clean stop at
      <span class="nowrap">$t=0.00341$</span>, <span class="nowrap">$R_2=9.73$</span> pc.</td>
  <td><span class="pill bad">grind</span> energy phase <strong>bit-identical</strong> to V0 (52/52
      snapshots), but at the energy→implicit handoff the implicit <code>solve_ivp</code> hammers the
      RHS with <span class="nowrap">$\sim$1.2M drive- + 0.6M state-clamps</span> and <strong>never
      advances a single step</strong>; <span class="nowrap">$R_2$</span> freezes at 8.61 pc, killed by
      the 300 s timeout.</td>
  <td>removing the clean <span class="nowrap">$E_b\le0$</span> stop just exposes a non-convergent
      implicit grind — the energy-driven interior is genuinely degenerate as
      <span class="nowrap">$E_b\to0$</span>.</td>
</tr>
<tr>
  <td><strong><code>fail_helix</code></strong><br>(real Helix: <code>sfe=0.05</code>,
      <code>PISM=0</code>, 1a)</td>
  <td><code>ENERGY_COLLAPSED</code> (51) reached in phase 1a,
      <span class="nowrap">$t=0.00255$</span>, <span class="nowrap">$R_2=7.03$</span> pc.</td>
  <td><span class="pill bad">identical stop</span> <strong>byte-identical</strong> to V0
      (<span class="nowrap">$E_b=0.005085287954767937$</span> to the digit) — a <em>second,
      independent</em> guard fires.</td>
  <td>the floor clamps <span class="nowrap">$E_b$</span> in <code>bubble_E2P</code>/<code>solve_R1</code>
      but cannot stop the <strong>bubble-structure solve itself</strong> from degenerating as
      <span class="nowrap">$E_b\to0$</span> (cooling table out of bounds, <code>solve_R1</code> can't
      bracket) → <em>"bubble solve degenerate as Eb -&gt; 0"</em>.</td>
</tr>
</table>

<figure>
  <img src="__FIG_H3__" alt="EBFLOOR no-op and grind">
  <figcaption><strong>Fig 1 — H3: the floor is a no-op where it can be, and no rescue where it
  matters.</strong> Left: <code>fail_repro</code> $E_b(t)$ — V0 (blue) collapses through 0 to a clean
  <code>ENERGY_COLLAPSED</code> stop; <code>EBFLOOR</code> (orange dashed) avoids that stop but the
  implicit solver then grinds with $R_2$ stuck. Right: <code>simple_cluster</code> control — V0 and
  <code>EBFLOOR</code> overlay exactly (max&nbsp;rel|$\Delta E_b$|&nbsp;=&nbsp;0), a bit-identical
  no-op because $E_b$ never collapses there.</figcaption>
</figure>

<p>A side-result worth keeping: the cooling-balance transition trigger <strong>never fires under
EBFLOOR</strong> (the cooling-balance ratio <code>rmin</code> sits at 0.50–0.99 on every config, never
within <span class="nowrap">$100\times$</span> of the 0.05 threshold). Flooring
<span class="nowrap">$E_b$</span> adds <em>drive</em>, not <em>cooling</em> — it cannot manufacture a
cooling-balance event. The collapse regime is also narrow and <strong>mass-gated</strong>: only
<span class="nowrap">$5\times10^{9}\,M_\odot$</span> collapses early.</p>

<h2>2 · H4 — cap the PdV drain for an early window?</h2>
<p>If the cause is PdV super-criticality, the next idea throttles the cause directly: during
<span class="nowrap">$t<t_{\rm window}$</span> replace the drain with
<span class="nowrap">$\min({\rm PdV},\,\kappa\,L_{\rm mech})$</span> (<span class="nowrap">$\kappa=0.9$</span>),
leaving the shell-acceleration drive untouched so the bubble still expands, then lift the cap. The
question: does the bubble <strong>self-sustain</strong> after release, or re-collapse the moment
PdV/$L_{\rm mech}$ is unthrottled? Capping PdV <strong>under-drains the bubble / injects energy</strong>
while active — again a diagnostic, not a fix.</p>

<div class="callout warn">
<span class="pill">SURVIVABLE TRANSIENT, NON-PHYSICAL RESCUE</span> The heavy clouds are <strong>not
instantaneously "stillborn"</strong> — there exists a window after which the bubble self-sustains —
but the required window is longer than the natural PdV&gt;1 epoch the real physics imposes, so the
"rescue" is purchased entirely with injected, non-conserved energy.</div>

<h3>The drain stays super-critical for an extended epoch</h3>
<p>PdV/$L_{\rm mech}$ stays above 1 for roughly <span class="nowrap">$1.5$–$4.5\times10^{-3}$</span>
Myr from the phase start, and the outcome of a transient cap depends entirely on whether the window
outlasts that epoch. The maintainer's exact <span class="nowrap">$\sim$10$^{-3}$</span> Myr is too
short; the window splits the two clouds at <span class="nowrap">$3\times10^{-3}$</span>; and only a
<span class="nowrap">$10^{-2}$</span> Myr window keeps both alive:</p>

<table>
<tr><th>$t_{\rm window}$ [Myr]</th><th><code>fail_repro</code> (1b, sfe=0.1)</th><th><code>fail_helix</code> (1a, sfe=0.05)</th></tr>
<tr>
  <td><code>1e-3</code></td>
  <td><span class="pill bad">collapses</span> cap never even bites (PdV&lt;$L_{\rm mech}$ that early);
      <strong>byte-identical to V0</strong>, collapses at <span class="nowrap">$t=0.0034$</span>.</td>
  <td><span class="pill bad">collapses</span> cap lifts while PdV/$L\approx2.4$; same 1a stop ~$10^{-4}$
      Myr later (<span class="nowrap">$E_b\approx0.005$</span>).</td>
</tr>
<tr>
  <td><code>3e-3</code></td>
  <td><span class="pill bad">re-collapses</span> $E_b$ inflated to
      <span class="nowrap">$7.9\times10^{9}$</span>, dips hard at release, PdV/$L$ only crosses below 1
      at <span class="nowrap">$t\sim0.0046$</span> — too little, too late;
      <code>ENERGY_COLLAPSED</code> at <span class="nowrap">$t=0.00515$</span>.</td>
  <td><span class="pill ok">self-sustains</span> $E_b$ grows under the cap; at release it dips but
      PdV/$L$ falls through 1 (1.09→1.00→0.94→0.88) and $E_b$ <strong>recovers</strong> — a genuine
      survivable transient.</td>
</tr>
<tr>
  <td><code>1e-2</code></td>
  <td><span class="pill ok">self-sustains</span> but only after a full <span class="nowrap">$10^{-2}$</span>
      Myr (<span class="nowrap">$\sim$10$\times$</span> the proposed window) of injection has built
      <span class="nowrap">$E_b\approx2.2\times10^{10}$</span> and dropped PdV/$L$ to 0.74 by release.</td>
  <td><span class="pill ok">self-sustains</span></td>
</tr>
</table>

<figure>
  <img src="__FIG_H4_EB__" alt="H4 Eb sweep for fail_repro">
  <figcaption><strong>Fig 2 — H4: $E_b(t)$ for <code>fail_repro</code> across cap windows.</strong>
  V0 (black) collapses through 0; dotted vertical lines mark each cap release. $t_{\rm win}=10^{-3}$ is
  byte-identical to V0 (the cap never bites); $3\times10^{-3}$ inflates then re-collapses;
  $10^{-2}$ builds a large $E_b$ reservoir and self-sustains after release — a reservoir
  <em>manufactured</em> by an order of magnitude more injection than proposed.</figcaption>
</figure>

<figure>
  <img src="__FIG_H4_RATIO__" alt="H4 PdV/Lmech sweep for fail_repro">
  <figcaption><strong>Fig 3 — H4: PdV/$L_{\rm mech}(t)$ for <code>fail_repro</code>.</strong> The
  break-even line (PdV/$L_{\rm mech}=1$) is the whole story: the bubble self-sustains only if the cap
  is released <em>after</em> the ratio has fallen below 1. The off-scale negative spike at collapse is
  the $E_b\to0$ sign-flip artifact (clipped here), already shown in Fig&nbsp;2.</figcaption>
</figure>

<figure>
  <img src="__FIG_H4_SUMMARY__" alt="H4 survived / self-sustained summary">
  <figcaption><strong>Fig 4 — H4 outcomes vs $t_{\rm window}$.</strong> Survived-window (solid) and
  self-sustained (dashed) across the collapse configs. <code>fail_helix</code> flips to
  self-sustaining by $3\times10^{-3}$ Myr; <code>fail_repro</code> needs $10^{-2}$. ($t_{\rm win}=10^{-1}$
  rows read "no" only because the run truncates before the cap is released — a "cap never lifted"
  control, not a failure.)</figcaption>
</figure>

<h3>The control parameter cleanly separates the regimes</h3>
<p>The cap is a strict no-op on healthy/stall controls (<code>simple_cluster</code>,
<code>pl2_steep</code>, <code>small_1e6</code> all reach a clean <code>STOPPING_TIME</code> with
<span class="nowrap">max&nbsp;rel|$\Delta E_b$|&nbsp;$\le2.6\times10^{-5}$</span> vs V0) — a genuine
no-op to <span class="nowrap">$\ge4$</span> significant figures, the
<span class="nowrap">$\kappa=0.9$</span> graze being its only, negligible, honest cost. The reason is
simple: on those configs PdV/$L_{\rm mech}$ settles well below 1 once the self-similar phase is
established (after a brief IC-handoff spike on the first step — a snapshot-proxy artifact that does not
track the true $\dot E_b$, so the integrated cap stays a no-op), so the cap never meaningfully engages.
The same single ratio that leaves the controls untouched is what crosses 1 and stays there to kill the
<span class="nowrap">$5\times10^{9}$</span> clouds.</p>

<figure>
  <img src="__FIG_H4_CONTROL__" alt="control vs collapse PdV ratio">
  <figcaption><strong>Fig 5 — one control parameter sorts the regimes.</strong> PdV/$L_{\rm mech}(t)$
  for <code>simple_cluster</code> (control — below 1 after the brief IC-handoff spike, so the cap never
  bites; byte-identical) vs the two <span class="nowrap">$5\times10^{9}$</span> clouds (cross 1 and stay
  there → $E_b$ collapses). The break-even
  crossing is the discriminator between a healthy bubble and an over-drained one.</figcaption>
</figure>

<h2>3 · What it means — the heavy-cloud fix is momentum, not energy props</h2>
<p>Both experiments converge on the same conclusion: <strong>propping up the energy phase does not
revive these clouds.</strong> H3 shows that flooring <span class="nowrap">$E_b$</span> only swaps a
clean stop for a grind (<code>fail_repro</code>) or leaves a second degeneracy guard to fire
(<code>fail_helix</code>) — the energy-driven interior is genuinely degenerate as
<span class="nowrap">$E_b\to0$</span> along <em>more than one</em> axis. H4 shows that the only way to
keep the bubble alive is to manufacture an <span class="nowrap">$E_b$</span> reservoir the real
expansion work would have removed; the "rescue" is exactly the injected, non-conserved energy
<span class="nowrap">${\rm PdV}-\kappa L_{\rm mech}$</span> integrated over the window.</p>

<div class="callout ok">
<strong>The honest fate is a momentum-driven continuation</strong> (or added cooling/leakage physics),
<strong>not</strong> an energy-phase prop. The shipped <code>ENERGY_COLLAPSED</code> termination is
<em>diagnosing a real physics breakdown, not hiding a tractable continuation.</em> Whatever replaces it
must hand off to a momentum phase when the energy-driven interior over-drains — the regime input H4
quantifies (extended PdV super-criticality) is precisely what such a fix needs.</div>

<h3>The unifying picture: opposite extremes of one ratio</h3>
<p>This closes the loop with the rest of the storyline. The <strong>normal-cloud stall</strong>
(transition trigger never fires; the bubble stays energy-driven but never reaches the cooling-balance
event) and the <strong>$5\times10^{9}$ crash</strong> (PdV/$L_{\rm mech}$ crosses 1 and
<span class="nowrap">$E_b$</span> collapses) are <em>opposite extremes</em> of the same PdV/$L_{\rm mech}$
axis: stall clouds never over-drain (ratio stays comfortably below 1, the bubble lives but the trigger
sleeps); heavy clouds over-drain hard (ratio rises through 1, the bubble dies). A single physical
control parameter spans both pathologies — and points at the same handoff machinery as the cure.</p>

<p>The momentum handoff itself — the deferred family <code>T</code>, including the
<span class="nowrap">$E_b$</span>-peak detection that would mark where the energy-driven phase should
end and the momentum-driven continuation should begin — is the subject of the ongoing transition work
(<code>docs/dev/transition/</code>). H3 and H4 are the diagnostics that justify pursuing it rather than
patching the energy phase.</p>

<h2>4 · Reproduce</h2>
<pre><code># H3 Eb-floor matrix + analysis (monkeypatch-only; production untouched)
bash docs/dev/transition/pt4/h3_run_matrix.sh
python docs/dev/transition/pt4/h3_analyze.py

# H4 PdV-cap sweep + analysis
bash docs/dev/transition/pt4/h4_run_matrix.sh
python docs/dev/transition/pt4/h4_analyze.py

# figures + this report (from the committed CSVs / PNGs, no re-run needed)
python docs/dev/transition/pt4/make_pt4_figures.py   # h3_ebfloor_noop_and_grind, h4_control_vs_collapse
python docs/dev/transition/pt4/h4_figures.py         # h4_Eb_sweep_*, h4_pdvratio_sweep_*, h4_summary
python docs/dev/transition/pt4/make_pt4_heavy_report.py</code></pre>

<p class="meta">TRINITY (<code>trinity-sf</code>) · feedback-driven bubble-evolution code · part 4 of
storyline s3 ("Why large clouds failed"). Report generated by
<code>docs/dev/transition/pt4/make_pt4_heavy_report.py</code> from committed artifacts under
<code>docs/dev/transition/pt4/</code>; all numbers quoted verbatim from
<code>H3_eb_floor_experiment.md</code> / <code>H4_pdvcap_experiment.md</code>. Both experiments are
monkeypatch-only diagnostics — production code is untouched, and every "rescue" injects energy and
cannot ship. Math via MathJax (needs a network connection to render). Point-in-time analysis; line
numbers may drift — re-verify against current source.</p>

</div>
</body>
</html>
"""

HTML = (HTML.replace("__FIG_H3__", FIG_H3)
        .replace("__FIG_H4_EB__", FIG_H4_EB)
        .replace("__FIG_H4_RATIO__", FIG_H4_RATIO)
        .replace("__FIG_H4_SUMMARY__", FIG_H4_SUMMARY)
        .replace("__FIG_H4_CONTROL__", FIG_H4_CONTROL))

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT}  ({len(HTML)/1024:.0f} KB)")
