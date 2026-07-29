#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_pdvtrigger_report.py — build THE single consolidated HTML for the pdv-trigger workstream.

CONSOLIDATION (2026-07-28). This report replaces THREE documents with one:
  - the old pdvtrigger_report.html (2026-07-03 generator — the f_kappa/f_mix era; 25 days and
    five phases behind the record, audited section-by-section before merging),
  - ELBADRY_THETA_STORY.html (2026-07-01 — the El-Badry theory + imposition + reversal story),
  - phase6_brief.html (2026-07-28 — the corrected f_A-vs-f_mix head-to-head).
The originals are preserved under docs/dev/to-be-removed/ for the maintainer's review. Claims
were re-verified against FINDINGS.md (through §21), the source tree, and CONTAMINATION.md by
four independent audits (report-claims, story-claims, code-truth, ship-status) before merging;
superseded numbers are either struck in place or replaced, each with its citation.

Storyline contract (docs/dev/html-insights/build_storylines.py): single <h1>, <p class="sub">
subtitle, scoped <style>, inline base64 figures, no required scripts. Equations are matplotlib-
mathtext SVGs (committed as phase6_eq_*.svg, built by data/make_phase6_figures.py) embedded as
data URIs — offline, no CDN, survives the storyline merger's script stripping.

REPRODUCE
    python docs/dev/transition/pdv-trigger/data/make_phase6_figures.py   # eq SVGs + phase6 figs
    python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py     # -> pdvtrigger_report.html
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pdvtrigger_report.html"

# token -> (path relative to HERE, alt text)
FIGURES = {
    "__FIG_REGIME__": ("storyline_figs/fig_regime_split.png",
                       "PdV/Lmech per config: normal clouds sub-critical near 0.45, the 5e9 above 1"),
    "__FIG_DOUBLE__": ("storyline_figs/fig_doublecount.png",
                       "single-count line vs the forbidden double-count region, with the max-closure and MC draws"),
    "__FIG_ZONES__": ("zone_profiles.png",
                      "T and n vs depth below R2 for L1/L2/L3, three configs, and where L_cool is emitted"),
    "__FIG_FKDEF__": ("fkappa_definition.png",
                      "the Spitzer conductivity f_kappa multiplies; Mdot ~ f_kappa^(2/7) verified"),
    "__FIG_FABOOST__": ("fA_source_boost.png",
                        "the f_A source-term screen: the fourth knob corner"),
    "__FIG_EB1__": ("story_elbadry_f1_closedform.png",
                    "El-Badry theta(n) closed form crossing 0.95 at n~48; configs colored by fate"),
    "__FIG_EB6__": ("story_elbadry_f6_regime.png",
                    "live PdV-vs-radiative decomposition: fail_repro is 99% PdV at the Eb peak"),
    "__FIG_EB5__": ("story_elbadry_f5_reversal.png",
                    "default vs theta-imposed on merged code: imposing theta drives clouds to the velocity cap"),
    "__FIG_T5K__": ("theta5k_fire_map.png",
                    "first rule-compliant kappa matrix: no whole-band f_kappa exists"),
    "__FIG_DMDT__": ("dmdt_tackle_flow.png",
                     "the dMdt condensation-boundary diagnosis and the no-root handoff fix"),
    "__FIG_T5ARMS__": ("theta5_arms.png",
                       "theta_max vs f_mix, 8 configs, with the 0.95 trigger and the Lancaster band"),
    "__FIG_T5METRIC__": ("theta5_metric_correction.png",
                         "why blowout-theta was retired: the 2.1x diffuse under-read"),
    "__FIG_T5LAW__": ("theta5_collapse_law.png",
                      "the theta0-collapse law: smallest firing boost vs starting deficit"),
    "__FIG_T5BFIRE__": ("theta5b_fire_map.png",
                        "the fine bracket: measured f_mix window [4, 4.5]"),
    "__FIG_FAEDGE__": ("fA_edge_map.png",
                       "f_A condensation-edge map: no dMdt<=0 edge even at f_A=512"),
    "__FIG_BENCH5__": ("bench5_theta_tracks.png",
                       "bench5 theta(t) tracks against the L21b band"),
    "__FIG_SC0__": ("fa_state_screen.png",
                    "SC-0: three candidate f_A laws vs measured doses - all fail"),
    "__FIG_P61__": ("phase6_fig1_correction.png",
                    "raw vs corrected f_mix dose-response: the artifact and the fix"),
    "__FIG_P62__": ("phase6_fig2_headtohead.png",
                    "corrected dose-response, both knobs, three clean benches"),
    "__FIG_P63__": ("phase6_fig3_uniformity.png",
                    "band-entry dose vs density: the decision metric"),
    "__FIG_P64__": ("phase6_fig4_stale.png",
                    "Theta_cum split into frozen-no-root vs solved rows"),
    "__FIG_P65__": ("phase6_fig5_windcap.png",
                    "the 3-Myr wind-only cap moves 17/60 production arms"),
    "__FIG_P66__": ("phase6_fig6_slope.png",
                    "metric-2 slope passes [-1,0] but decays 3-8x slower than L21b"),
    "__FIG_P67__": ("phase6_fig7_mechanism.png",
                    "bench3 fm8: Theta_cum=4.635 is a frozen-solver artifact"),
}

EQS = {  # token -> phase6_eq_<key>.svg (alt = readable TeX)
    "__EQ_ENERGY__": ("energy", "dE_b/dt = L_mech - L_loss - P dV/dt; Theta_cum = int_W L_loss dt / int_W L_mech dt"),
    "__EQ_THETADEF__": ("theta_def", "theta = L_loss/L_mech; fire when (L_gain-L_loss)/L_gain < 0.05 <=> theta >~ 0.95"),
    "__EQ_COOLSPLIT__": ("cool_split", "L_cool = L_1 + L_2 + L_3"),
    "__EQ_LAYOUT__": ("layout", "R_1 [L_1 hot CIE] -> [L_2 conduction] -> [L_3 sliver] -> R_2 (CD <-> shell)"),
    "__EQ_FKAPPA__": ("fkappa", "kappa_eff = f_kappa C_th T^(5/2) (3 sites); Mdot_seed ~ f_kappa^(2/7) UP"),
    "__EQ_FMIX__": ("fmix", "structure solved FIRST, unboosted -> L_cool; then L_eff = L_leak + f_mix L_cool"),
    "__EQ_THETATARGET__": ("thetatarget", "L_loss = max(L_cool+L_leak, theta_t L_mech)"),
    "__EQ_FA__": ("fa", "du/dt -> f_A (du/dt)_rad for T<10^5.5 K; L_cool = L_1 + f_A(L_2+L_3)"),
    "__EQ_ELBADRY__": ("elbadry", "theta_EB(n) = A_mix sqrt(ldv n) / (11/5 + A_mix sqrt(ldv n)); A_mix=3.5, ldv=3, n_fire~48"),
    "__EQ_EQ47CH__": ("eq47_channels", "Eq 47 factorized: theta-channel (1-theta)^(37/35)/theta^(2/7) falls; C-channel (C/6e-7)^(2/7) rises. f_A moves theta, f_kappa moves C."),
    "__EQ_EQ47__": ("eq47", "mdot = mdot_0 (1-theta)^(37/35)/theta^(2/7) - evaporation falls as cooling rises"),
    "__EQ_NUMERATORS__": ("numerators", "Theta_raw = int(Lcool+Lleak)dt/int Lmech dt (superseded) vs Theta = int theta Lmech dt / int Lmech dt (corrected)"),
    "__EQ_STALE__": ("stale_split", "Theta_cum = solved part (physics) + frozen part (solver state)"),
    "__EQ_SLOPE__": ("slope", "1-Theta ~ t^-1/2 => dlog10(1-theta)/dlog10 t = -0.5, pass band [-1,0]"),
    "__EQ_DOUBLEBOOST__": ("doubleboost", "L_loss^fallback = f_mix(f_mix L_cool) = f_mix^2 L_cool"),
    "__EQ_FIRED__": ("fired", "fired = meta_fired OR (reached_momentum AND theta_max >= 0.95)"),
    "__EQ_EXTRAP__": ("extrap", "p = ln(Theta1/Theta0)/ln(d1/d0); d_band = d1 (0.90/Theta1)^(1/p)"),
    "__EQ_FK1__": ("fk_site1", "Mdot_seed = (12/75) xi^(5/2) 4pi R2^3/t (mu_ion/kB)(t f_kappa C_th/R2^2)^(2/7) Pb^(5/7) => Mdot ~ f_kappa^(2/7)"),
    "__EQ_FK2__": ("fk_site2", "dR2 = T_init^(5/2)/(C Mdot/(4pi R2^2)), C = (25/4) kB/(mu_ion f_kappa C_th) => dR2 ~ f_kappa"),
    "__EQ_FK3__": ("fk_site3", "d2T/dr2 = Pb/(f_kappa C_th T^(5/2))[(beta+5delta/2)/t + (5/2)(v-v_t)(1/T)dT/dr - udot/Pb] - (5/2T)(dT/dr)^2 - (2/r)dT/dr"),
    "__EQ_FA1__": ("fa_site1", "udot -> f_A udot if T < 10^5.5 K, inside the ODE RHS"),
    "__EQ_FA2__": ("fa_site2", "L2 -> f_A L2, L3 -> f_A L3; L1 and L_leak untouched"),
    "__EQ_INTEGRAND__": ("integrand", "L1 = int chi_e n^2 Lambda_CIE(T) 4pi r^2 dr; L2,3 = int udot_net(n,T,phi) 4pi r^2 dr"),
    "__EQ_NDENS__": ("ndens", "n(r) = Pb/((mu_conv/mu_ion) kB T(r)); n ~ 1/T at near-uniform Pb"),
    "__EQ_SC0__": ("sc0", "C1 El-Badry, C2 Lancaster Eq 11, C3 fitted f_A = 315 nbar^-0.335"),
}


def _b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode((HERE / path).read_bytes()).decode()


def img(token):
    path, alt = FIGURES[token]
    return f'<img src="{_b64(path, "image/png")}" alt="{alt}">'


def eq(token):
    key, alt = EQS[token]
    return (f'<div class="eq"><img src="{_b64(f"phase6_eq_{key}.svg", "image/svg+xml")}" '
            f'alt="{alt}"></div>')


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>The pdv-trigger workstream — one story: the trigger, the four knobs, the corrections, and what ships</title>
<style>
#pdv{font:15px/1.6 Georgia,serif;max-width:1000px;margin:2rem auto;padding:0 1.2rem;color:#1a1a1a;background:#fdfcf8}
#pdv h1{font-size:1.5rem;border-bottom:3px solid #8a1c1c;padding-bottom:.3rem}
#pdv h2{font-size:1.2rem;color:#8a1c1c;margin-top:2.2rem;border-bottom:1px solid #e0d9c8;padding-bottom:.2rem}
#pdv h3{font-size:1.02rem;color:#5a3030;margin-top:1.4rem}
#pdv .sub{color:#555;font-size:.95rem;margin-top:.2rem}
#pdv .eq{background:#f4f1e8;border-left:4px solid #8a1c1c;padding:.7rem 1rem;margin:.8rem 0;overflow-x:auto}
#pdv .note{background:#eef3f8;border-left:4px solid #2a5d8f;padding:.6rem 1rem;margin:.8rem 0;font-size:.92rem}
#pdv .warn{background:#fdf0ef;border-left:4px solid #c0392b;padding:.6rem 1rem;margin:.8rem 0;font-size:.92rem}
#pdv .good{background:#eef7ef;border-left:4px solid #1e7a3c;padding:.6rem 1rem;margin:.8rem 0;font-size:.92rem}
#pdv table{border-collapse:collapse;margin:1rem 0;font-size:.88rem;width:100%}
#pdv th,#pdv td{border:1px solid #c9c2b0;padding:.35rem .55rem;text-align:left;vertical-align:top}
#pdv th{background:#f4f1e8}
#pdv code{font-family:Menlo,Consolas,monospace;font-size:.85em;background:#f0ede4;padding:.05em .3em;border-radius:3px}
#pdv .small{font-size:.85rem;color:#555}
#pdv figure{margin:1.4rem 0;padding:0;text-align:center}
#pdv figure img{max-width:100%;height:auto;border:1px solid #ddd6c4;background:#fff;border-radius:3px}
#pdv figcaption{font-size:.85rem;color:#444;text-align:left;margin-top:.5rem;border-left:3px solid #d8cfb8;padding:.3rem .6rem}
#pdv del{color:#999}
#pdv .toc{background:#f7f5ee;border:1px solid #e0d9c8;padding:.6rem 1rem;font-size:.9rem}
#pdv .toc a{color:#8a1c1c;text-decoration:none}
#pdv sub,#pdv sup{font-size:.72em}
</style></head><body><div id="pdv">

<h1>The pdv-trigger workstream — one story: the trigger, the four knobs, the corrections, and what ships</h1>
<p class="sub">Consolidated 2026-07-28 on <code>feature/pdv-trigger-5</code> — replaces the 2026-07-03 report, the
El-Badry θ story (2026-07-01), and the Phase-6 brief (2026-07-28); originals preserved in
<code>docs/dev/to-be-removed/</code>. Every claim re-verified against <code>FINDINGS.md</code> (§1–§21),
<code>CONTAMINATION.md</code>, and the source tree by four independent audits; superseded numbers are struck
or replaced in place, each with its citation. Sources of truth: <code>FINDINGS.md</code>,
<code>SOURCE_TERM_DESIGN.md</code>, <code>FA_STATE_COUPLED.md</code>, <code>CONTAMINATION.md</code>,
<code>data/*.csv</code>.</p>

<div class="warn"><b>&#9888; How to read this document.</b> The workstream spans five knob generations and three
metric corrections; earlier prose survives here only where it is still true, and every retired number is struck
(<del>like this</del>) with the superseding citation. The current record ends at <code>FINDINGS §21</code>
(2026-07-28): the f<sub>mix</sub> Θ<sub>cum</sub> metric artifact is fixed (§18), "f_mix eliminated" is withdrawn,
the §16 double-boost is bounded out of the pdv benches (§19) but <b>measured live in a sibling campaign</b> (§21),
and the head-to-head is an open estimate the fm≤8 grid cannot settle.</div>

<div class="toc"><b>Contents</b> ·
<a href="#p1">1 The problem</a> ·
<a href="#p2">2 The four knobs + the Eq-47 factorization</a> ·
<a href="#p3">3 El-Badry theory: impose, break, calibrate, screen</a> ·
<a href="#p4">4 The f<sub>κ</sub> era</a> ·
<a href="#p5">5 The f<sub>mix</sub> era</a> ·
<a href="#p6">6 The f<sub>A</sub> era</a> ·
<a href="#p7">7 The 2026-07-27/28 corrections</a> ·
<a href="#p8">8 SC-0: the derived-law screen fails</a> ·
<a href="#p9">9 What needs to be shipped</a> ·
<a href="#p10">10 The consistency plan</a> ·
<a href="#p11">Artifacts &amp; reproduce</a></div>

<h2 id="p1">1 · The problem: an energy phase that never hands off</h2>
<p>TRINITY integrates an energy-driven wind bubble and must decide when that phase ends. The budget and the
window metric used throughout:</p>
__EQ_ENERGY__
<p>The default trigger is <code>cooling_balance</code> — fire when radiated losses balance the mechanical input
(<code>run_energy_implicit_phase.py:1296</code>; threshold <code>phaseSwitch_LlossLgain=0.05</code>,
<code>registry.py:382</code>):</p>
__EQ_THETADEF__
<p>The problem (FINDINGS §6a): TRINITY's resolved 1-D cooling peaks at θ ≈ 0.2–0.7 for normal GMCs, against the
3-D literature's Θ ≈ 0.90–0.99 (Lancaster 2021a,b) — a 1-D spherical interface under-counts the fractal interface
area, so the trigger never fires and clouds exit by geometric blowout instead. Two facts frame every fix:</p>
<ul>
<li><b>PdV is not a loss to add.</b> It already sits in the energy ODE as its own term; a trigger or an imposed θ
that counts it again double-drains the bubble (§2's Monte-Carlo closure and the §3 reversal both measure this).</li>
<li><b>The regime splits at PdV/L<sub>mech</sub> ≈ 1.</b> Normal clouds sit near 0.45 (cooling-correctable);
the 5×10⁹ M<sub>☉</sub> control is PdV-dominated and rides the PR#715 E<sub>b</sub>≤0 handoff regardless of any
cooling knob (FINDINGS §8b/§8c).</li>
</ul>
<figure>__FIG_REGIME__<figcaption><b>The regime split.</b> PdV/L<sub>mech</sub> per config: six normal clouds
cluster near 0.45; only the 5×10⁹ <code>fail_repro</code> exceeds unity. Cooling boosts are for the left group;
the right group's physics is the E<sub>b</sub>-peak handoff. (<code>pdv_regime_budget.csv</code>, FLAG: frozen
provenance — trust <code>live_pdv_decomp.csv</code> where they differ.)</figcaption></figure>
<figure>__FIG_DOUBLE__<figcaption><b>The single-count contract.</b> A (1−θ)L<sub>mech</sub> input rescale must never
stack on an explicit L<sub>cool</sub>: 5×10⁵ Monte-Carlo draws find zero double-count configurations on the shipped
<code>max()</code> closure (<code>doublecount_mc.csv</code>, CLEAN). The §16/§21 trigger-fallback bug is precisely a
violation of this contract on one code path.</figcaption></figure>

<h2 id="p2">2 · The four knobs — exactly which equations each acts on</h2>
<p>Everything the workstream tried is one of four registry knobs (<code>registry.py:384–388</code>; all default-inert,
all <code>exclude_from_snapshot</code>). The bubble's radiative loss splits by temperature zone,</p>
__EQ_COOLSPLIT__
<p>with L₁ the hot CIE interior (volume emission), and L₂+L₃ the thin conduction front draped on the contact
discontinuity at R₂:</p>
__EQ_LAYOUT__
<p>Pressure is ≈uniform across the subsonic interior, so the solver sets the density directly from the
temperature — which is why the front is dense and its emissivity n²Λ(T) per unit volume is highest:</p>
__EQ_NDENS__
<p>and what each zone actually integrates is:</p>
__EQ_INTEGRAND__

<div class="warn"><b>&#9888; CORRECTED 2026-07-28 (FINDINGS §22) — the front is NOT where most of the radiation
comes from.</b> Earlier versions of this report said L₂+L₃ is "where nearly all the radiation emerges" and that
L₁ is "hot and rarefied → weak cooling". <b>Measured, that is backwards.</b> Averaged over the accepted rows of
all 14 committed <code>__none</code> arms (<code>runs/data/bench_state_traj/</code>, from the solver's own
<code>bubble_L2Conduction</code>/<code>bubble_L3Intermediate</code> against <code>bubble_LTotal</code>):
<b>L₁ = 60–77%, L₂ = 15–34%, L₃ = 1–25%</b> (typically ≈70 / 26 / 2). The front's emissivity per unit volume
really is orders of magnitude higher — but it is ~10⁵× thinner, and the interior's sheer volume wins. So
f<sub>A</sub>, which scales only L₂+L₃, starts with a lever on about a quarter of the cooling (its share grows
with dose — by f<sub>A</sub>=16 the boosted L₂+L₃ dominates the total). This changes no published Θ<sub>cum</sub>
or fire number — those are measured from L<sub>loss</sub>, not from this decomposition — but it corrects the
physical story used to motivate the knob.</div>

<figure>__FIG_ZONES__<figcaption><b>The zone anatomy, measured.</b> <code>data/zone_profiles.csv</code>, built by
<code>data/make_zone_profiles.py</code> — captured live from the solver at the 3rd settled energy-phase
evaluation, f<sub>A</sub>=1, for the dense/mid/diffuse benches. <b>Top:</b> T climbs from 10⁴ K at the contact
discontinuity to ~3×10⁷ K in the interior; the dashed line is the 10<sup>5.5</sup> K CIE switch defining the
L₁/L₂ boundary. <b>Middle:</b> n mirrors it exactly (nT constant to &lt;1% across all three zones), ~5×10⁵ cm⁻³
at the front down to ~10² cm⁻³ inside. <b>Bottom:</b> cumulative fraction of L<sub>cool</sub> emitted going
inward from R₂ — it only reaches ≈0.3 by the end of L₂, so <b>~70% of the cooling comes from L₁</b>. The x-axis
is depth below R₂ on a log scale spanning ~10 decades: L₂+L₃ occupy the outermost ~10⁻⁷ pc, which is why they
are invisible on any linear-radius plot.</figcaption></figure>

<h3>2.1 f<sub>κ</sub> — <code>cooling_boost_kappa</code>: multiply the conduction coefficient (in-solve)</h3>
__EQ_FKAPPA__
<p>It enters at three places in <code>bubble_luminosity.py</code>, always through the same product
f<sub>κ</sub>·C_thermal. Written out in full:</p>
__EQ_FK1__
__EQ_FK2__
__EQ_FK3__
<p>(ξ is the dMdt factor, β/δ the cooling exponents, v<sub>t</sub> = α r/t the similarity velocity, u̇ the net
radiative source.) Site 1 is precisely why evaporation <b>rises</b> with f<sub>κ</sub>; site 3 is where the
conductivity divides the entire temperature-curvature term. It does <b>not</b> multiply L<sub>cool</sub> — it thickens the conduction layer so more gas sits where Λ(T) peaks,
and θ emerges as an output. Side effect that killed it as the final model: the evaporative flux
<b>rises</b> (Ṁ ∝ f<sub>κ</sub><sup>2/7</sup>, measured 1.2175 vs analytic 2<sup>2/7</sup> = 1.2190 at
f<sub>κ</sub>=2 — 0.12%). <b>That rise MATCHES El-Badry Eq 47, it does not contradict it</b> (§2.6). No validator; <code>'auto'</code> resolves a trilinear lookup on the 819-run
grid (<code>fkappa_auto.py</code>, ceiling 64), measured at f<sub>A</sub>=1 only.</p>
<figure>__FIG_FKDEF__<figcaption><b>What f<sub>κ</sub> multiplies.</b> Left: the Spitzer conductivity
κ = C<sub>th</sub>T<sup>5/2</sup>. Right: the Ṁ ∝ f<sub>κ</sub><sup>2/7</sup> analytic-vs-measured check
(0.1% agreement). From the 2026-07-03 report §13 — still current
(<code>kappa_backreaction.csv</code> CLEAN for the scaling check).</figcaption></figure>

<h3>2.2 f<sub>mix</sub> — <code>cooling_boost_mode='multiplier'</code>: multiply the resolved loss (post-solve)</h3>
__EQ_FMIX__
<p><code>get_betadelta.py:353–357</code>: the structure is solved first, unboosted, inside
<code>get_bubbleproperties_pure</code> (<code>get_betadelta.py:436</code>); the scalar applies after, at
<code>:473</code>. T(r), dMdt, and the zone split are <b>frozen</b> at the unboosted solution — the map
(β,δ)→structure never sees f<sub>mix</sub>; only the fixed-point (β,δ) root moves. L<sub>leak</sub> is never
scaled. It is the <i>frozen-structure limit</i> of f<sub>A</sub>. No validator — an unrecognised
<code>mode</code> token silently falls back to the resolved loss (<code>get_betadelta.py:357</code>).</p>

<h3>2.3 θ_target — <code>cooling_boost_mode='theta_target'</code>: impose a loss fraction (post-solve)</h3>
__EQ_THETATARGET__
<p><code>get_betadelta.py:355–356</code>. The <code>max</code> keeps it single-count. This is the production
vehicle the 2026-06-30 El-Badry imposition rode (§3). Setting θ<sub>t</sub> ≥ 0.95 fires the trigger
essentially by construction. The advertised θ<sub>max</sub>&lt;1 ceiling in its ParamSpec is
<b>not implemented anywhere</b> (code-truth audit, 2026-07-28) — documentation only.</p>

<h3>2.4 f<sub>A</sub> — <code>cooling_boost_fA</code>: multiply the interface source term (in-solve + integrals)</h3>
__EQ_FA__
<p>Its two edit sites in <code>bubble_luminosity.py</code>, written out:</p>
__EQ_FA1__
__EQ_FA2__
<p>The band edge is <code>_T_INTERFACE_BAND</code> at <code>:65</code>. Site 1's u̇ is the same u̇ that appears
inside f<sub>κ</sub>'s site-3 equation above — that is exactly why the structure responds. Because site 1 is inside both the
dMdt root-find and the profile integration, the structure <b>responds</b>: the interface gets cooler and denser,
and the evaporative flux <b>falls</b> — Eq 47's <i>θ-channel</i>, measured across theta5s (FINDINGS §15e).
This is <b>not</b> a sign advantage over f<sub>κ</sub>; the two knobs move different factors of the same
equation (§2.6).
Validated (<code>registry.py:117–148</code>: raises on f<sub>A</sub>≤0, warns on cross-knob combination);
default byte-identical (literal float-ops guard).</p>
<figure>__FIG_FABOOST__<figcaption><b>The fourth knob corner.</b> The f<sub>A</sub> offline screen (FINDINGS §15):
the source-term boost passes all four registered predictions — the corner of (in-solve × interface-targeted) the
other three knobs miss. </figcaption></figure>

<h3>2.6 The El-Badry Eq-47 factorization — why neither knob has a "wrong sign"
<span class="tag t-out">CORRECTED 2026-07-29</span></h3>
<p>The workstream long held that f<sub>κ</sub> raising evaporation was "the wrong El-Badry sign", and used
that to prefer f<sub>A</sub>. <b>Reading Eq 47 as printed, that is backwards.</b> The equation carries a
conduction factor whose normalization constant is <i>exactly</i> TRINITY's <code>C_thermal</code> default,
6×10⁻⁷ cgs (<code>registry.py:377</code>):</p>
__EQ_EQ47CH__
<p>So Eq 47 <b>rises</b> with conduction, ∂ln ṁ/∂ln C = +2/7 — and TRINITY's f<sub>κ</sub>, which multiplies
that very C, measures 1.2175 against the analytic 2<sup>2/7</sup> = 1.2190, i.e. it <b>reproduces El-Badry's
own scaling to 0.12%</b>. The falling behaviour lives in the <i>other</i> factor,
(1−θ)<sup>37/35</sup>/θ<sup>2/7</sup>, which decreases in θ. f<sub>A</sub> does not touch C; it raises the
radiative losses, i.e. θ. <b>Both knobs are consistent with the same equation — they move different variables
in it.</b> In El-Badry's parameterization these are independent: θ comes from Eq 38 (λδv and n̄), with no C.</p>
<div class="note"><b>The defensible concern, stated correctly.</b> A knob meant to represent
<i>turbulent-mixing-driven</i> cooling should act through <b>θ</b> — that is the channel by which efficient
cooling reduces the hot-gas mass (El-Badry §6.1: "<i>the mass of hot gas in the bubble interior is
reduced</i>" at large n<sub>H,0</sub> and λδv). Implementing mixing as a multiplier on the <i>Spitzer
coefficient</i> instead moves the C-channel, which raises ṁ. So f<sub>κ</sub> is the wrong <b>vehicle</b> for
mixing — not because its sign disagrees with Eq 47, but because it moves the wrong variable. That is a
narrower and survivable claim than "wrong sign", and it is the one the record should carry.</div>
<div class="warn"><b>What this does and does not change.</b> It does <b>not</b> rehabilitate f<sub>κ</sub>: its
two independent empirical failures stand — no whole-band f<sub>κ</sub> exists (best single value fires 5/6 vs
the multiplier's 6/6, FINDINGS §12) and κ<sub>mix</sub>/κ<sub>Spitzer</sub> ≈ 10³–10⁷ in the cool layer, so a
scalar on Spitzer C cannot represent mixing (§9b). It does remove one leg from the case for f<sub>A</sub> —
the leg that survived §18's withdrawal of the measurement-based case — and it points the same way as the
area argument in FINDINGS §22: an interface-area increase raises the conductive flux, so a faithful
area knob should make ṁ <b>rise</b>, per Eq 47's own C-exponent.</div>

<h3>2.5 The comparison — one table (code-truth audit, 2026-07-28)</h3>
<table>
<tr><th></th><th>f<sub>κ</sub></th><th>f<sub>mix</sub></th><th>θ_target</th><th>f<sub>A</sub></th></tr>
<tr><td>multiplies</td><td>Spitzer <code>C_thermal</code> (κ<sub>eff</sub>=f<sub>κ</sub>C<sub>th</sub>T<sup>5/2</sup>)</td>
<td>resolved L<sub>cool</sub>=<code>bubble_LTotal</code></td><td>replaces loss with θ<sub>t</sub>L<sub>mech</sub> via <code>max</code></td>
<td>in-band <code>dudt</code> + L₂,L₃</td></tr>
<tr><td>code sites</td><td><code>bubble_luminosity.py:304/398/441</code></td><td><code>get_betadelta.py:354</code></td>
<td><code>get_betadelta.py:356</code></td><td><code>bubble_luminosity.py:435–437/845–848</code></td></tr>
<tr><td>where in pipeline</td><td>inside dMdt fsolve + ODE</td><td>after the structure solve</td><td>after the structure solve</td>
<td>inside ODE + on the integrals</td></tr>
<tr><td>T(r)/dMdt respond?</td><td><b>yes</b> — dMdt <b>↑</b> ∝ f<sub>κ</sub><sup>2/7</sup> (Eq 47 <i>C</i>-channel)</td><td>no (frozen)</td><td>no (frozen)</td>
<td><b>yes</b> — dMdt <b>↓</b> (Eq 47 <i>θ</i>-channel)</td></tr>
<tr><td>temperature gate</td><td>none</td><td>none</td><td>none</td><td>T&lt;10<sup>5.5</sup> K in-ODE; unconditional on L₂+L₃</td></tr>
<tr><td>scales L<sub>leak</sub>?</td><td>indirect only</td><td>no</td><td>absorbed in <code>max</code></td><td>no (by design)</td></tr>
<tr><td><code>bubble_LTotal</code> in output</td><td>changed</td><td><b>unchanged (raw)</b></td><td><b>unchanged</b></td><td>changed</td></tr>
<tr><td>validator</td><td>none (resolver only)</td><td>none</td><td>none (ceiling unimplemented)</td><td>yes</td></tr>
<tr><td>tests</td><td><code>test_fkappa_auto.py</code> (lookup only)</td><td><code>test_cooling_boost.py</code> (pure fn only)</td>
<td><code>test_cooling_boost.py</code></td><td><code>test_fA_source_boost.py</code> (both sites)</td></tr>
</table>
<div class="warn"><b>Two wiring defects, verified in source (2026-07-28; the first is FINDINGS §16/§19/§21, the
second is NEW).</b>
(1) <b>The trigger-fallback double-boost.</b> On a no-physical-root segment the trigger seeds from
<code>params['bubble_Lloss']</code> — already the effective loss — and boosts it again
(<code>run_energy_implicit_phase.py:1244–1247</code>):
__EQ_DOUBLEBOOST__
Under <code>'theta_target'</code> the re-application is idempotent; under <code>'none'</code> inert; under
<code>'multiplier'</code> it is an f<sub>mix</sub>² error on the trigger only (the local value is never written
back, so Θ<sub>cum</sub> is untouched). §19 bounds it out of every pdv bench number; <b>§21 measures it live in
the rosette-cf campaign</b> (§9 below).
(2) <b>The phase-1a/1c energy-ODE asymmetry.</b> <code>energy_phase_ODEs.py:273</code> builds dE<sub>b</sub>/dt
from the <b>raw</b> <code>bubble_LTotal</code>, while the phase-1a trigger check
(<code>run_energy_phase.py:279</code>) uses the <b>effective</b> loss — so for f<sub>mix</sub>/θ_target the boost
reaches the trigger but not the energy budget in phases 1a/1c, contradicting <code>registry.py:384</code>'s
"feeds the β-δ residual, the energy ODE, AND the trigger consistently". (In phase 1b both flow through β, so 1b
is consistent.) f<sub>κ</sub>/f<sub>A</sub> are immune (they change <code>bubble_LTotal</code> itself). Registered
in the consistency plan (§10).</div>

<h2 id="p3">3 · El-Badry theory — imposed, broken, demoted to calibration, screened, falsified</h2>
<p>El-Badry+2019 (MNRAS 490, 1961 — <i>not</i> ApJ 879; citation corrected 2026-06-29) computes the equilibrium
cooling fraction of a turbulent mixing layer. The closed form (Eq 37/38) and its physics:</p>
__EQ_ELBADRY__
<p><b>Why √(λδv·n)</b> (the only prose derivation in the workstream, from the 2026-07-01 story, still current):
the mixing rate scales with λδv (mixing length × velocity dispersion — the one free knob, calibrated so the GMC
band lands on Lancaster's θ≈0.9–0.99); the layer's emission measure scales with n; the radiated luminosity, hence
θ, goes as the square root of their product. The 11/5 and A<sub>mix</sub> come from integrating Λ(T) through the
layer, which sits at T<sub>pk</sub>≈2×10⁴ K where Λ peaks. <b>The regime caveat that seeded everything after:</b>
El-Badry's θ is a <i>radiative</i> ratio — it says nothing about PdV/inertial losses.</p>
__EQ_EQ47__
<figure>__FIG_EB1__<figcaption><b>The closed form.</b> θ(n) crosses the 0.95 trigger at n<sub>fire</sub> ≈ 48 cm⁻³
(λδv=3; arithmetic re-verified 2026-07-28 — <code>LANCASTER_REFERENCE.md:143–145</code> is the loose one, flagged).
<del>Clouds denser than the crossing transition; diffuse clouds stay energy-driven</del> — the fate reading is
superseded (<code>PLAN.md</code>: route-a is not a clean density threshold; the firing diffuse config shares
n=10² with the never-firing <code>small_1e6</code>).</figcaption></figure>
<p><b>The imposition (2026-06-30)</b> rode <code>theta_target</code> (§2.3): θ(n) computed per step, forced into
<code>effective_Lloss</code>; the <code>max()</code> never selected the resolved term (resolved-wins 0/N), so
imposition ≡ assignment. <b>The reversal (2026-07-01, FINDINGS §8b):</b> on post-PR#715 code the default expands
the massive clouds, while imposing θ drives the same clouds to the −500 pc/Myr velocity cap — because at their
E<sub>b</sub>-peak the budget is 99% PdV / ~1% radiative, and an imposed radiative θ=0.99 <b>double-counts</b>
against the PdV term already in the ODE.</p>
<figure>__FIG_EB6__<figcaption><b>The measured double-count.</b> Live PdV-vs-radiative decomposition
(<code>live_pdv_decomp.csv</code>, CLEAN): <code>fail_repro</code> is 99% PdV at its E<sub>b</sub>-peak. An
imposed radiative θ there is a second withdrawal of energy the ODE already spent.</figcaption></figure>
<figure>__FIG_EB5__<figcaption><b>The reversal.</b> Default (green) vs θ-imposed (red) on merged code:
<code>fail_repro</code> and <code>pl2_steep</code> expand by default, collapse to the velocity cap when θ is
enforced (<code>newcode_default_vs_theta.csv</code>, CLEAN — "the canonical post-#715 massive-cloud fate
record").</figcaption></figure>
<div class="note"><b>The v₂&lt;0 question (2026-07-01, unsuperseded).</b> A recollapsing shell with E<sub>b</sub>&gt;0
is still an energy-driven bubble — the energy/momentum distinction is about E<sub>b</sub>, not the velocity sign;
"breathing" (recollapse then rebound) is physical. The solver hang seen under imposed θ is an E<sub>b</sub>→0
singularity, and the proposed fix is a pressure-crossover terminal event (<code>HIMASS_HANDOFF_PLAN §2</code>,
still ⏳). This retracted the author's own earlier v₂&lt;0→momentum suggestion. <i>Adjacent but distinct:</i> the
frozen-no-root rows of §7 are the same "held state" mechanism surfacing in the Θ<sub>cum</sub> metric.</div>
<p><b>The demotion (2026-07-01):</b> θ is an <i>output</i>; El-Badry/Lancaster became the calibration target and
θ_target a documented continuous opt-in override. <b>The final role (2026-07-25):</b> El-Badry re-entered as
SC-0's candidate C1 — a derived f<sub>A</sub> law — and was <b>falsified</b> (§8): θ<sub>EB</sub> saturates for
n ≳ 43, so the law is λδv-insensitive and cannot be tuned. The very saturation the closed-form plot celebrates is
what killed it as a law.</p>

<h2 id="p4">4 · The f<sub>κ</sub> era — the right thermodynamics, the wrong side effect</h2>
<p>Rung A multiplied conduction (§2.1) so θ would emerge — and it does: the matched-t back-reaction measured
L<sub>cool</sub> ×1.23–1.38 at f<sub>κ</sub>=2 with the loss ratio up +0.05–0.10
(<code>kappa_backreaction.csv</code> CLEAN). Three things ended the era:</p>
<ul>
<li><b>The calibration numbers of 2026-06-28/07-01 are retired.</b> <del>f<sub>κ</sub>-to-fire ≈ 4 (compact) /
5–6 (mid) / ~60 (diffuse); f<sub>κ</sub><sup>fire</sup> ≈ 10³n<sup>−0.60</sup>; 6/63 never fire</del> — all
blowout-metric / short-run contaminated (⛔ <code>CONTAMINATION.md</code> #3/#4). theta5 measured diffuse
f<sub>fire</sub> = <b>4, not 60</b>: the blowout metric under-read diffuse θ by ~2×.</li>
<li><b>The knob breaks structurally at high dose.</b> The "dead windows" were solver crashes at the
evaporation→condensation boundary (dMdt → −85 M<sub>☉</sub>/Myr at f<sub>κ</sub>=8); the McKee &amp; Cowie 1977
regime switch became the <code>no_physical_root_handoff</code> fate (FINDINGS §9b/§12).</li>
<li><b>No whole-band f<sub>κ</sub> exists.</b> The first rule-compliant kappa matrix (theta5k, 56 arms, zero
freezes): best single value fires 5/6 vs the multiplier's 6/6 — and it loses the fire-vs-drain race, not the
reach.</li>
</ul>
<figure>__FIG_T5K__<figcaption><b>theta5k.</b> 56/56 proper fates after the no-root handoff fix; five condensation
handoffs; no single f<sub>κ</sub> fires the band (<code>theta5k_summary.csv</code> CLEAN for FIRE/NO-FIRE;
fired-arm θ<sub>max</sub> ≳ 1.2 is structural distortion — quote fire/no-fire only).</figcaption></figure>
<figure>__FIG_DMDT__<figcaption><b>The dMdt condensation boundary, start to finish.</b> Symptom (dip) → diagnosis
(β–δ eigenvalue crossing) → physics identity (conduction-front evaporation reverses: McKee &amp; Cowie 1977) →
the adopted regime switch. The best-written diagnostic arc of the workstream (2026-07-03 report §16.5), kept
whole in FINDINGS §12–§14.</figcaption></figure>
<p><b>κ_mix (Rung B) stayed shelved</b>: in the cool layer κ<sub>mix</sub>/κ<sub>Spitzer</sub> ≈ 10³–10⁷, so no
scalar f<sub>κ</sub> can represent mixing (analytics CLEAN, <code>fkappa_physical_derivation.csv</code>) — but
the wired prototype saturates by λδv ≈ 0.01, so λδv is not a tunable knob
(<code>KMIX_SELFCONSISTENT.md</code>). The successor to both is f<sub>A</sub> (§6).</p>

<h2 id="p5">5 · The f<sub>mix</sub> era — the first rule-compliant calibration, and the adopted science setting</h2>
<p>The 2026-07-01 standing rules (θ only as θ<sub>max</sub> over ≥5 Myr from accepted rows; never blowout-θ)
retired every earlier number and produced theta5 — 8 configs × f<sub>mix</sub> {1,2,4,8} × 5 Myr, 32/32
compliant:</p>
<ul>
<li>θ₀ spans 0.297–0.717 (and 1.047 for the natively-firing ninth config, theta5n); the one-parameter collapse law
<b>f<sub>fire</sub> ≈ 1.4·(0.95/θ₀)<sup>1.82</sup></b> holds out-of-sample at rms 0.064 dex (theta5b, 7 configs);</li>
<li>a single <b>f<sub>mix</sub> = 4</b> fires the whole normal-GMC band (θ<sub>max</sub> 0.96–1.04); the measured
window is <b>[4, 4.5]</b> (theta5b: 3.5 misses <code>pl2_steep</code>, 5 drops <code>midrange</code> to the
fire-vs-drain race);</li>
<li>controls hold: <code>small_1e6</code> never fires through 8; <code>fail_repro</code> rides the PR#715 handoff
with or without boost;</li>
<li><b>f<sub>mix</sub> = 4 ADOPTED as the science setting (2026-07-02 maintainer ruling</b> — momentum-then-recollapse
is acceptable physics). Production default stays <code>none</code>; the setting rides in <code>.param</code> files.</li>
</ul>
<figure>__FIG_T5ARMS__<figcaption><b>theta5.</b> θ<sub>max</sub> vs f<sub>mix</sub>, all configs, with the 0.95
trigger and the Lancaster band (<code>theta5_summary.csv</code> CLEAN). Caption caveat carried from FINDINGS §19:
"fired" labels here (and in every fire map below) are <code>reached_momentum ∧ θ_max≥0.95</code> — 0 bench arms
recorded a live <code>cooling_balance</code> termination:
__EQ_FIRED__</figcaption></figure>
<figure>__FIG_T5METRIC__<figcaption><b>Why blowout-θ died.</b> The per-config dumbbells from θ-at-blowout to
θ<sub>max</sub>: the diffuse baseline was under-read ~2× (peak at t≈4.9 Myr, long after blowout). This one figure
retires the 2026-06-28→07-01 calibration numbers wherever they appear.</figcaption></figure>
<figure>__FIG_T5LAW__<figcaption><b>The collapse law.</b> Smallest firing boost vs starting deficit (0.95/θ₀) —
all property dependence flows through θ₀; the kappa knob's law (steeper, broken) shown for contrast.</figcaption></figure>
<figure>__FIG_T5BFIRE__<figcaption><b>theta5b.</b> The fine bracket: window [4, 4.5] measured; law validated
out-of-sample at 0.064 dex rms.</figcaption></figure>
<p><b>The shipped mechanics</b> (per <code>get_betadelta.py:353–357</code>; run recipe):
<code>cooling_boost_mode&nbsp;multiplier</code> + <code>cooling_boost_fmix&nbsp;4</code> in the <code>.param</code>.
Because f<sub>mix</sub> acts after the solve it is structurally immune to the condensation boundary that breaks
f<sub>κ</sub> — the same frozen-structure property that is its physical weakness in §6's comparison. It is the
science setting of the Paper II grid (<code>param/paperII_grid_sweep.param</code>, 10,560 runs) and the rosette-cf
survey (72 arms) — a fact the Phase-6 record under-weighted until §21.</p>

<h2 id="p6">6 · The f<sub>A</sub> era — the physically-correct knob, calibrated against Lancaster 21b</h2>
<p>f<sub>A</sub> (§2.4) is Lancaster's fractal-area factor as a source term: A<sub>eff</sub> = f<sub>A</sub>·4πR₂².
Wired 2026-07-06 with literal byte-identity at default (pre==postA==postB <code>dictionary.jsonl</code> sha256,
FINDINGS §15b/§15c). The measurements:</p>
<ul>
<li><b>theta5s (81/81, HPC-confirmed):</b> the collapse law transfers with exponent p = 3.330 (rms 0.055 dex),
confirming the registered p<sub>source</sub>≈3.3; dMdt suppression &lt;1 falling with dose matrix-wide — the
Eq-47 sign f<sub>mix</sub> cannot produce; fire thresholds 1→4→12→24→64 as density falls.</li>
<li><b>bench5 (60/60, HPC-confirmed):</b> five L21b Table-1 benchmark clouds × f<sub>A</sub> grid; the clean-blowout
benches enter the Θ band [0.90,0.99] at f<sub>A</sub> ≈ 13.9 (bench3) / 53.5 (bench2) / 74.8 (bench1) — steeply
density-dependent, fit f<sub>A</sub>(n̄) ≈ 315·n̄<sup>−0.335</sup>.</li>
<li><b>bench6:</b> the f<sub>A</sub> dose extension + the f<sub>mix</sub> head-to-head — the decision data whose
f<sub>mix</sub> half §7 corrects.</li>
</ul>
<figure>__FIG_FAEDGE__<figcaption><b>The safe-direction falsification.</b> The condensation-edge map: no dMdt≤0
edge exists for f<sub>A</sub> even at 512 (16× the physical range) — the source knob structurally cannot reach the
f<sub>κ</sub> condensation crash (FINDINGS §15a).</figcaption></figure>
<figure>__FIG_BENCH5__<figcaption><b>bench5.</b> θ(t) tracks against the L21b band; production arms censored at
fire, diagnostic (blowout) arms uncensored — the Θ<sub>cum</sub> calibration reads from the latter
(FINDINGS §15h).</figcaption></figure>

<h2 id="p7">7 · The 2026-07-27/28 corrections — the metric artifact, and what actually survives</h2>
<p>An external review (FINDINGS §17) found the bench6 f<sub>mix</sub> Θ<sub>cum</sub> was computed with a
numerator that omitted the boost; the fix (§18) and its fallout:</p>
__EQ_NUMERATORS__
<figure>__FIG_P61__<figcaption><b>The artifact and the fix.</b> Left: the superseded raw numerator that produced
§15j's "wrong-sign dose-response". Right: corrected — monotone rising on every clean bench. The superseded column
is kept as <code>theta_cum_raw_superseded</code> and reproduces the published §15j values exactly (auditable, not
overwritten). Gates: f<sub>A</sub> side bit-stable at 2.6×10⁻¹⁶ vs a ≤10⁻⁹ bar; all 60 bench5 rows
string-identical.</figcaption></figure>
<div class="warn"><b>Withdrawn (§18/§19):</b> "f_mix eliminated by measurement" and all three legs — the wrong
sign was the artifact; "never reaches the band" is a statement about the fm≤8 <i>grid</i> (which under-brackets
exactly as f<sub>A</sub>'s ≤16 grid once did); "fm8 false-fires" rested on the artifact plus a backwards
a-fortiori argument about the §16 bug.</div>
<figure>__FIG_P62__<figcaption><b>The corrected head-to-head.</b> f<sub>mix</sub> reaches a given Θ<sub>cum</sub>
at roughly an order of magnitude smaller dose — a statement about dose, not physical correctness.</figcaption></figure>
<figure>__FIG_P63__<figcaption><b>The decision metric.</b> Band-entry-dose uniformity: f<sub>A</sub> spreads 5.39×
(all measured); f<sub>mix</sub> 2.96× (bench3 measured ≈4; bench2/bench1 <b>extrapolated</b> ≈8.2/11.9 past the
fm≤8 grid). <b>The head-to-head inverts — as an estimate.</b> The extrapolation:
__EQ_EXTRAP__
and the response <i>saturates</i> (a global fm≤4 power law over-predicts the measured fm8 points), which biases
true entry doses <b>upward</b> — if bench1's true entry is ≈20 rather than 11.9, the spread returns to ≈5× and
the inversion vanishes. Settling it needs fm ∈ {12,16} on bench1/bench2 (~4 arms). Both knobs also carry a large
frozen-no-root share in Θ<sub>cum</sub> (below) — <i>worse on the f<sub>A</sub> side</i>.</figcaption></figure>
__EQ_STALE__
<figure>__FIG_P64__<figcaption><b>The shared caveat (§18, new).</b> A no-root β–δ segment leaves the bubble state
and <code>bubble_Lloss</code> frozen (<code>run_energy_implicit_phase.py:893/:929–930</code>) yet still logged:
Θ<sub>cum</sub> integrates held-over θ across real time. On the band-setting arms the frozen share is
<b>larger for f<sub>A</sub></b> (67/65/54%) than f<sub>mix</sub> (33/47/7%). Not an f<sub>mix</sub> artifact; not
fixed (it would move every published Θ<sub>cum</sub>); maintainer question Q3.</figcaption></figure>
<figure>__FIG_P67__<figcaption><b>What bench3-fm8's Θ<sub>cum</sub>=4.635 really is.</b> 71% frozen rows with θ
held at 3.45/2.23/3.02 across long spans — a solver-domain failure at high dose, not a measured cooling
fraction.</figcaption></figure>
<figure>__FIG_P65__<figcaption><b>Honesty closure (a):</b> the spec's 3-Myr wind-only window, implemented — moves
17/60 arms (all never-fired production arms) by 4.3–33.1%, superseding the 5–17% estimate. No diagnostic arm
crosses 3 Myr, so the decision metric is untouched. Quote <code>theta_cum_wind_only</code> for production
arms.</figcaption></figure>
__EQ_SLOPE__
<figure>__FIG_P66__<figcaption><b>Honesty closure (b):</b> Phase-5 metric 2's self-contained half — every
clean-blowout arm passes [−1,0], but at −0.06…−0.31 TRINITY's 1−θ decays 3–8× more slowly than L21b's t<sup>−1/2</sup>,
and the gap widens toward low density. A real fidelity gap the Θ<sub>cum</sub> band alone hides.</figcaption></figure>
<div class="note"><b>Also closed:</b> L<sub>leak</sub> ≡ 0 in all 120 bench trajectories (Phase-5 metric 6 is
vacuous; R&amp;P's 60–75% leakage is the mandatory paper caveat — maintainer question Q2) · the FIRE-label
semantics corrected in place everywhere (§19) · <code>parents[3]</code>→<code>[4]</code> and the 1%-vs-2% gate
docstring fixed; both param sets still regenerate byte-identically (§20).</div>

<h2 id="p8">8 · SC-0 — no derived f<sub>A</sub> law survives; the pre-registered stop</h2>
__EQ_SC0__
<p>Three candidates screened offline against measured doses (band entries + fire thresholds), discriminator =
spread of predicted/measured (calibration-invariant):</p>
<table>
<tr><th>candidate</th><th>spread</th><th>verdict</th></tr>
<tr><td>C1 El-Badry</td><td>3.3× (band) / 4.5× (fire) vs a 2× bar</td><td>λδv-insensitive (θ<sub>EB</sub> saturates for n≳43) — cannot be tuned</td></tr>
<tr><td>C2 Lancaster Eq 11</td><td>174–307×, 2–8 dex high</td><td><b>FALSIFIED</b> — ℓ<sub>cool</sub> ≈ 8×10⁻¹⁵ pc lies below every physical and numerical scale for every cascade index p&lt;1; Eq 11's operative ℓ is the truncation scale</td></tr>
<tr><td>C3 fitted 315·n̄<sup>−0.335</sup></td><td>56× off its fit points</td><td>a <b>local fit, not a law</b></td></tr>
</table>
<figure>__FIG_SC0__<figcaption><b>SC-0 (FINDINGS §15k), 14/14 arms, TERMINAL.</b> The pre-registered stop is in
force: SC-1…SC-5 are not started, no f<sub>A</sub> form ships, no production code is written. Unaffected by the §17/§18
corrections (its targets are f<sub>A</sub>-side doses only — verified).</figcaption></figure>

<h2 id="p9">9 · What needs to be shipped — the adjudicated list (2026-07-28)</h2>
<p class="small">Method: an independent ship-status audit (registry + decision record + campaign params) was
cross-examined against the code-truth audit and this session's own record; each disputed claim was re-verified in
source, and one was settled by a new measurement (§21). "Record" = what the documented rulings already say;
"judgment" = this consolidation's call, with its evidence.</p>
<table>
<tr><th>#</th><th>item</th><th>basis</th></tr>
<tr><td colspan="3"><b>SHIPPED / SHIP NOW (no maintainer input needed)</b></td></tr>
<tr><td>S1</td><td><b>The branch as-is.</b> Zero behavioral diff vs main (<code>git diff origin/main...HEAD -- trinity/</code>
= info strings + <code>default.param</code> regen + tests). The X1 metric fix, both regenerated CSVs (with
<code>theta_cum_raw_superseded</code> audit trail), <code>test_bench_theta_cum.py</code>, and FINDINGS §18–§21.</td>
<td>record + verified</td></tr>
<tr><td colspan="3"><b>KEEP OPT-IN (already ruled; caveats updated)</b></td></tr>
<tr><td>K1</td><td><b>f<sub>mix</sub> = 4</b> — not merely a "bench fallback": it is the <b>science setting of two live
campaigns</b> (Paper II grid, 10,560 runs; rosette-cf, 72 arms — both <code>multiplier</code>+<code>fmix 4</code>
under the default trigger; verified in their params). The R0→R2 "retirement ladder" should be <b>cancelled, not
halted</b>: its premise was withdrawn (§18) and its R2 precondition ("nothing in-repo relying on it") is already
false. Carry the §16/§21 bug caveat until fixed.</td><td>judgment; params verified</td></tr>
<tr><td>K2</td><td><b>f<sub>A</sub></b> — diagnostic/paper knob, default 1.0. f<sub>A</sub>(n̄) is a measurement of
record, never a law (SC-0). Its Θ<sub>cum</sub> calibration carries the frozen-row caveat (54–67% on band-setting
arms).</td><td>record (clause 2a)</td></tr>
<tr><td>K3</td><td><b>f<sub>κ</sub></b> incl. <code>'auto'</code> — keep, but its info string should say plainly:
structural probe, breaks at high dose (condensation boundary), raises Ṁ against the Eq-47 sign, <code>'auto'</code>
grid measured at f<sub>A</sub>=1 only. <b>θ_target</b> — keep as the documented override; its advertised 0..1
ceiling is unimplemented.</td><td>judgment + code-truth</td></tr>
<tr><td colspan="3"><b>DO NOT SHIP / CANCELLED</b></td></tr>
<tr><td>D1</td><td>Any default flip (clause 3, RULED 2026-07-22) · SC-1…SC-5 and any derived f<sub>A</sub> law
(§15k terminal) · the fm "eliminated" claim in any form (§18).</td><td>record</td></tr>
<tr><td colspan="3"><b>MAINTAINER DECISIONS — the exact questions</b></td></tr>
<tr><td>Q1</td><td>Clause 1's grounds: re-derive from the physical in-ODE asymmetry (recommended — it never depended
on the bench6 metric), or withdraw the "superseded/retirement" framing outright? Either answer also fixes the two
<b>misleading production info strings</b> (<code>registry.py:384–385</code>: "superseded for L21b calibration by
cooling_boost_fA … retained pending a state-coupled successor" — both clauses dead per §15k/§18).</td><td>record asks;
text proposed in PLAN</td></tr>
<tr><td>Q2</td><td>Is C<sub>f</sub>=1 (L<sub>leak</sub>≡0 in all 120 bench trajectories) expected, or is the leak
channel silently disabled? Phase-5 metric 6 is vacuous either way; the R&amp;P 60–75% caveat is mandatory for the
paper.</td><td>record (§20)</td></tr>
<tr><td>Q3</td><td>Is a Θ<sub>cum</sub> that is 54–67% frozen no-root rows publishable, or must the metric exclude
no-root rows / carry an uncertainty band? ("Exclude" moves every published Θ<sub>cum</sub>, f<sub>A</sub> side
included.)</td><td>record (§18)</td></tr>
<tr><td>Q4</td><td>Fund ~4 arms at fm ∈ {12,16} on bench1/bench2. The saturation argument (§7) means the current
2.96× spread is biased <i>low</i> — the inversion is unresolved in the direction that could restore §15j's verdict.
Measure, don't re-litigate.</td><td>record + judgment</td></tr>
<tr><td>Q5</td><td><b>The §16 fix is now load-bearing — measured, not hypothetical (FINDINGS §21).</b> All 36
rosette-cf fm4 arms fired through the stale fallback path, and <b>1/36
(<code>1e5_sfe001_n5e2…Cf1p0</code>) is double-boost-DEPENDENT</b>: θ_eff = 0.923 &lt; 0.95 at its fire row —
the fixed code would not have fired there. The Paper II grid runs the same configuration. Decision needed:
schedule the §16 fix (full rule-5 ladder, mode-<code>none</code>/f<sub>A</sub> byte-identity gate) under the
Paper II / rosette-cf workstream <b>before</b> the 72-dictionary maintainer reduction, and re-check the one
dependent arm after. Evidence: <code>data/rosette_fm4_doubleboost_check.csv</code> (exhaustive, from the
historical dictionaries at <code>5aa84723</code>).</td><td><b>new measurement</b></td></tr>
</table>

<h2 id="p10">10 · The consistency plan — what remains to make everything correct and consistent</h2>
<p class="small">The actionable mirror of this section lives in <code>PLAN.md</code> ("Consistency plan,
2026-07-28") — that copy is the living one.</p>
<ol>
<li><b>Registry info strings</b> (<code>registry.py:384–385</code> + <code>default.param</code> regen): replace the
dead "superseded … pending successor" clauses per Q1. Byte-neutral, R0-class; blocked only on Q1's one-word choice.</li>
<li><b>The §16 fix</b> under the campaign that needs it (Q5): consume stored effective <code>bubble_Lloss</code>
directly in the fallback; full ladder with byte-identity for <code>none</code>/f<sub>A</sub>; then re-run the
rosette dependent-arm check.</li>
<li><b>The 1a/1c dEb/dt asymmetry</b> (§2.5 defect 2): either route <code>get_ODE_Edot_pure</code> through
<code>effective_Lloss_from_params</code> (behavioral for fm/θ runs — needs the ladder) or correct
<code>registry.py:384</code>'s "consistently" claim. Decide jointly with the §16 fix.</li>
<li><b>Validators</b>: reject unknown <code>cooling_boost_mode</code> tokens (a typo now silently un-boosts a
10,560-run grid) and enforce θ_target's advertised 0..1. Load-time behavior change — gate and changelog.</li>
<li><b>fm {12,16} arms</b> (Q4): ~4 HPC arms convert the inversion from estimate to measurement;
<code>make_bench6_params.py</code> already documents the needed extension.</li>
<li><b>Frozen-row metric decision</b> (Q3), then regenerate both analysis CSVs under the chosen convention.</li>
<li><b>Doc debris</b>: <code>LANCASTER_REFERENCE.md:143–145</code> n<sub>fire</sub>/λδv pairing loosened;
FINDINGS §8 walkthrough pointer fixed (was <code>fig/elbadry_f*</code>); MANIFEST regenerated; the storyline book
(<code>build_storylines.py</code>) rebuilt against this consolidated report.</li>
<li><b>Still-open measurement debt</b> (unchanged): metric 2's Fig-17 dex half (re-digitization), metric 3 (α<sub>p</sub>;
needs a re-harvest with momentum columns), the dMdt reducer on Helix raw arms, V<sub>w</sub> [I]-grade.</li>
</ol>

<h2 id="p11">Artifacts &amp; reproduce</h2>
<p class="small">Everything in this report regenerates offline from committed artifacts:
<code>python data/make_bench5_analysis.py</code> · <code>python data/make_bench6_analysis.py</code> ·
<code>python data/make_bench_stale_segments.py</code> · <code>python data/make_rosette_fm4_doubleboost_check.py</code> ·
<code>python data/make_phase6_figures.py</code> (figures + equation SVGs) ·
<code>python make_pdvtrigger_report.py</code> (this file). Per-claim artifact provenance:
<code>REPRODUCE.md</code>; per-artifact quotability grades: <code>CONTAMINATION.md</code>. The corrected
Θ<sub>cum</sub> metric is pinned by <code>test/test_bench_theta_cum.py</code>; the f<sub>A</sub> wiring by
<code>test/test_fA_source_boost.py</code>. Equations render as committed SVGs (mathtext, no CDN, no scripts) —
the TeX source of each lives in its <code>alt</code> attribute.</p>
</div></body></html>
"""


def main():
    html = HTML
    for token in EQS:
        html = html.replace(token, eq(token))
    for token in FIGURES:
        html = html.replace(token, img(token))
    leftover = [t for t in list(EQS) + list(FIGURES) if t in html]
    if leftover:
        raise SystemExit(f"unreplaced tokens: {leftover}")
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
