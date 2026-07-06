# The source-term design — a physical, solver-robust, tunable in-ODE cooling boost (f_A), the generalized front IC, and the path off the multiplier

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-06):** design analysis + first offline prototype (`FINDINGS.md §15`,
`data/make_fA_source_boost.py`). Answers the maintainer's standing question: *"I want an f_κ that
increases the Spitzer C inside the ODE — more physical than the output-side multiplier because it
also moves conductive evaporation — but it breaks the solver. Find a solution; take inspiration
from papers."* The short answer: **boosting C was the right instinct pointed at the wrong term.**
The physical enhancement is turbulent-mixing *radiation* at the interface, not Spitzer *conduction*
— so the in-ODE knob should multiply the **cooling source term** in the interface band, not the
conductivity. That knob (f_A) back-reacts with the **correct El-Badry sign** (cooling up ⇒
evaporation *down*), never touches the stiff conduction operator or the Eq-44 IC family (so the
solver machinery that f_κ destabilized is untouched), and stays a **continuous dial** (κ_mix's
born-saturated failure cannot happen: the source term is linear, it does not change the diffusion
operator). §5 adds the piece the κ-side still lacks: a **generalized near-front IC** (one first
integral) that recovers Weaver Eq. 44, fixes the κ_mix boundary divergence, admits the
Cowie–McKee saturation cap, and — critically — **remains defined for dM/dt < 0**, opening the
condensation branch (fix #4 of `KAPPA_FREEZE_MECHANISM.md §7`) as an increment instead of a
rewrite.

---

## 0. The question, sharpened by what this workstream already measured

Three structural options and one production fallback have been measured (all references are to
sibling docs; re-verify against source per the banner):

| knob | where it acts | tunable? | physical? | solver-robust? | verdict |
|---|---|:--:|:--:|:--:|---|
| `cooling_boost_kappa` (f_κ, Rung A) | multiplies Spitzer C in the ODE + Eq-44 IC + Eq-33 seed (`bubble_luminosity.py:297/377/413`) | ✅ dial | ❌ wrong-sign coupling: dMdt ×1.08–1.17 *up* at f_κ=2 (`KAPPA_EFF_SCOPING.md §6a`) vs El-Badry's ÷3–30 *down*; over-conducts the hot interior (`F_KAPPA_FUNCTIONAL_FORM.md §11`) | ⚠️ walks into the condensation domain edge; crash-free only since fix #1; razor-edge f_fire | no whole-band f_κ (theta5k, `FINDINGS.md §12`) |
| κ_mix floor (Rung B) | max(κ_mix, κ_S) in the ODE RHS | ❌ born saturated (λδv is not a dial) | ✅ the faithful cool-layer mixing term | ❌ NaNs at early high-Pb epochs (hard-max); boundary IC diverges (`KMIX_SELFCONSISTENT.md §2/§2b`) | shelved |
| `cooling_boost_fmix` (multiplier) | scales L_cool *after* the solve (`get_betadelta.py:334-371`) | ✅ dial | ❌ no back-reaction: the structure never feels the loss; dMdt untouched | ✅ by construction | **production, f_mix=4** (`FINDINGS.md §10/§11`) |
| `theta_target` | imposes L_loss floor | — | ❌ double-counts PdV (`FINDINGS.md §8b/§8c`) | ✅ | demoted to opt-in |

The pattern across the first two rows is the tunable-vs-physical trade named in
`KMIX_SELFCONSISTENT.md §2a`: *"the thing we could tune isn't physical; the thing that's physical
we can't tune."* Both rows share one design assumption — that the enhancement must enter through
the **conductivity**. This doc questions exactly that assumption.

## 1. The physics: the enhancement is in the SOURCE, not the conductivity

Four independent literature anchors say turbulent interface enhancement is a *radiation*
(source-term) effect in a 1-D average, not a conduction effect:

1. **El-Badry et al. 2019 (MNRAS 490, 1961), result vi** (distilled in `ELBADRY_REFERENCE.md §6`):
   *"Mixing, not conduction, sets the cooling; Spitzer conduction mainly sets interior
   T/evaporation."* The two roles are separable — and TRINITY's f_κ knob conflated them, which is
   precisely why its dMdt moved the wrong way.
2. **Lancaster et al. 2021a/b (ApJ 914, 89/90)**: the interface is a *fractal turbulent mixing
   layer*; the enhanced dissipation is radiative loss in the mixed gas over an enlarged area
   A_turb ≫ 4πR². Projected onto a 1-D spherically-averaged profile, an area factor multiplies the
   **volumetric emissivity of the interface band** — n²Λ → f_A·n²Λ for T below the mixing band top
   — while the *mean radial* conductive transport stays Spitzer.
3. **Weaver et al. 1977 §V**: even the classical front radiates ~40% of the conductive flux
   (`KAPPA_FREEZE_MECHANISM.md §3`). The budget split (evaporate vs radiate) is the physical dial;
   f_A moves that split directly. f_κ instead scales the *supply* and lets the same 60/40-ish
   split ride — the wrong lever for a θ problem.
4. **El-Badry Eq. 47** (`ELBADRY_REFERENCE.md §5`): ṁ ∝ (1−θ)^{37/35}/θ^{2/7} — evaporation is a
   *decreasing* function of interface cooling. Any faithful in-ODE mechanism must reproduce that
   sign. A source-term boost does it automatically: radiated flux is flux that no longer drives
   evaporation. A conductivity boost cannot (it raises supply and evaporation together;
   measured, `KAPPA_EFF_SCOPING.md §6a`).

**The honest counter-argument, and its empirical resolution.** There IS a published precedent for
the maintainer's original instinct: Gupta, Nath, Sharma & Eichler 2018 (MNRAS 473, 1537,
arXiv:1705.10448) carry exactly TRINITY's knob, κ = 6e-7·f_T·T^{5/2}, through the Weaver solution
(T_c ∝ f_T^{−2/7} — verified against their text, 2026-07-06 lit sweep). And a fractal interface
wrinkles the *conductive contact area* too, so one can argue an area factor should multiply κ (the
f_κ picture) rather than the emissivity (the f_A picture) — the 1-D projection of a corrugated
interface is genuinely ambiguous on paper. The discriminator is empirical: the two projections
back-react on evaporation with **opposite signs** (f_κ: Ṁ ∝ f_κ^{+2/7} up, measured ×1.08–1.17;
f_A: radiated flux no longer drives evaporation, Ṁ down), and the 1-D-hydro ground truth
(El-Badry, ÷3–30) says **down**. So the source-side projection is the one that matches the
resolved physics; the κ-side projection is the one that measurably contradicts it.

**The scaling algebra that quantifies "the multiplier is less physical" (2026-07-06 lit sweep;
derivations [D] from verified ingredients — re-check exponents against the PDFs before
hard-coding):** in the classical self-similar bubble with κ = f_κ·C·T^{5/2}: T_c ∝ f_κ^{−2/7}
[verified, Gupta+18], n_c ∝ f_κ^{+2/7}, Ṁ_ev ∝ f_κ^{2/7} (the Eq-33 seed's exponent, confirmed),
and L_cool ∝ f_κ^{(4+2ν)/7} ≈ f_κ^{4/7–5/7} — *sublinear* (cutoff-dominated interface integral).
Two consequences: (i) the Rung-A measurement (L_cool ×1.38→1.23 at f_κ=2 vs self-similar 2^{5/7} =
1.64) is the same sublinearity plus (β,δ) back-reaction — consistent, not anomalous; (ii) an
output-side multiplier f_mix on L_cool *implies* f_κ = f_mix^{7/5} while silently dropping the
side-effects that implied f_κ owes: Ṁ ×f_mix^{2/5}, T_c ×f_mix^{−2/5}, n_c ×f_mix^{+2/5} (and the
(β,δ) root shift through all three). At the adopted f_mix=4 those dropped factors are ×1.74 on Ṁ
and ×0.57 on T_c — order-unity state errors in exactly the quantities El-Badry says conduction
physics controls (interior n, T move >1 dex; dynamics barely move). That is the precise,
citable content of the maintainer's "the multiplier is less physical" intuition — and note the
dropped Ṁ factor has the WRONG SIGN anyway (El-Badry wants Ṁ *suppressed*), so "fixing" the
multiplier by implementing f_κ would over-correct into the opposite error. f_A drops neither: the
state variables respond, with the right sign.

**The f_A knob, precisely:** inside `_get_bubble_ODE`, multiply the net radiative term
`dudt` by f_A **only where T < 10^5.5 K** (the non-CIE interface band — the same cut the L₂
conduction-zone integral uses), leaving the conduction prefactor Pb/(C·T^{5/2}), the κ′ term, the
Eq-44 IC, and the Eq-33 seed untouched. Consistently, the emergent loss becomes
L_eff = L₁ + f_A·(L₂+L₃). The production `multiplier` knob is then recognizable as **the
no-back-reaction limit of f_A that additionally (and unphysically) scales the CIE interior L₁** —
a tidy referee story: the shipped f_mix is the frozen-structure approximation of the physical f_A.

Physical prior on magnitude: f_A is an area/emissivity ratio — Lancaster's resolved interfaces
give area ratios of a few to ~10s; El-Badry's fitted A_mix=3.5 (vs 1.7 analytic) hides a similar
factor. So f_A ∈ [2, 16] is the defensible window to probe, same order as the adopted f_mix=4 and
the physically-plausible f_max ≈ 2–8 of `F_KAPPA_FUNCTIONAL_FORM.md §11`.

## 2. Why f_A cannot reproduce the two structural failure modes

- **No κ_mix-style saturation.** κ_mix failed as a dial because it changes the *diffusion
  operator*: once κ_mix ≫ κ_S in the cool layer the layer is isothermalized and further increases
  do nothing (`KMIX_SELFCONSISTENT.md §2a`). f_A multiplies a *source*, linear in the ODE RHS: the
  response can (and does, §3) self-limit *smoothly* through the profile, but there is no operator
  takeover, so the dial stays continuous.
- **No stiff-machinery perturbation.** The f_κ crash chain ran through the eigenvalue physics
  (dMdt driven to the condensation edge) *and* branch/warm-start fragility tuned to f_κ=1 noise
  (`KAPPA_FREEZE_MECHANISM.md §2`). f_A leaves dR2, the anchor gradient, and the hot-interior
  conduction untouched; its only route to the condensation edge is the honest one — genuinely
  radiating away the front's budget — which is (a) the physics we *want* represented and (b)
  already routed to the momentum handoff by fix #1 (`no_physical_root_handoff`).

## 3. The offline prototype — measured (2026-07-06)

`data/make_fA_source_boost.py` — the **queued-but-never-run second offline prototype** of the
`KAPPA_EFF_SCOPING.md §6.2` redirect (*"add mixing-layer L_mix only to the in-structure loss
integrand (~10⁵ K band, κ unchanged), and measure ΔL_cool vs ΔdMdt — the new make-or-break"*).
Same shadow-first machinery as the κ_mix harnesses: monkeypatch of `_get_bubble_ODE` (production
expression re-emitted verbatim; the boost branch unreachable at f_A=1 → **G1 bit-identity**),
`make_da_replay` state rebuild on the committed cleanroom trajectories, **no sims, no production
edit**. f_A ∈ {1,2,4,8,16} × ~10 rows/config × 6 cleanroom configs.

**MEASURED (2026-07-06, all four predictions PASS 6/6).** Gates: G1 identity 6/6 (three configs
bit-exact 0.0; the others ≤1.8e-16 = one ULP from the `L1+(L2+L3)` association in L_eff), G2
replay 6/6 (≤3.1e-7). Per config (θ_max over the ~10 sampled rows; dM = median dMdt(f_A)/dMdt(1)):

| config | n | θ_max @ f_A=1→2→4→8→16 | dM @ 2→16 | solves |
|---|---:|---|---|---|
| large_diffuse_lowsfe | 1e2 | 0.52→0.56→0.65→0.84→**1.21** | 0.98→**0.88** | 50/50 |
| be_sphere | 1e4 | 0.50→0.53→0.60→0.73→**0.99** | 0.98→**0.88** | 50/50 |
| midrange_pl0 | 1e4 | 0.50→0.52→0.57→0.68→0.89 | 0.98→**0.89** | 50/50 |
| pl2_steep | 1e5 | 0.50→0.52→0.56→0.66→0.85 | 0.98→**0.90** | 50/50 |
| simple_cluster | 1e5 | 0.61→0.64→0.71→0.86→**1.18** | 0.98→**0.89** | 50/50 |
| small_dense_highsfe | 1e6 | 0.55→0.58→0.65→0.80→**1.10** | 0.97→**0.85** | 50/50 |

- **P1 (dial): PASS 6/6.** θ_max rises smoothly and monotonically over the whole f_A ∈ [1,16]
  range — no κ_mix-style born-saturation, and no κ_mix-style dense ceiling (the n=1e6 config
  responds like the diffuse one; κ_mix plateaued it at θ≲0.5).
- **P2 (El-Badry sign): PASS 6/6.** The solved dMdt eigenvalue FALLS monotonically with f_A
  (×0.97–0.98 at f_A=2 → ×0.85–0.90 at f_A=16) in every config — the first TRINITY knob measured
  to move evaporation the direction El-Badry's resolved hydro demands (f_κ measured ×1.08–1.17 UP
  at f_κ=2). Magnitude is modest on these early-phase replay states vs El-Badry's late-time
  equilibrium ÷3–30 — the direction is the screen's verdict; the magnitude is L2's business.
- **P3 (stability): PASS 6/6.** 300/300 solves, including the early high-Pb epochs that NaN'd
  the hard-max κ_mix, at f_A values twice the f_κ≈8 crash point. The stiff operator is untouched
  and it shows.
- **P4 (domain edge): no condensation onsets in range** — dMdt stays positive everywhere probed;
  the McKee–Cowie edge is beyond f_A=16 on these states, i.e. the knob approaches it gradually
  (P2's slope) rather than cliff-jumping onto it like f_κ.

Two quantitative readings for calibration planning: (i) the response is **sub-linear**,
θ_max ∝ f_A^{~0.30} (16× boost ⇒ ~2.2× θ) — back-reaction thins the radiating layer (the classic
mixing-layer self-limiting), so the naive f_mix=4 multiplier *overstates* what a physical
interface enhancement of 4 delivers, and a calibrated f_A will sit **above** the matching f_mix
(structural screens suggest f_A ~ 8–16 lands where f_mix ~ 4 does — Lancaster-plausible area
ratios). (ii) On replayed states 4/6 configs cross θ=0.95 by f_A=16 and the other two sit at
0.85/0.89 — screen-grade evidence that a single whole-band f_A may exist where theta5k found no
whole-band f_κ (`FINDINGS.md §12`); the live theta5-protocol matrix (L2) decides.

**Honesty box (CONTAMINATION rules):** these are *structural-response screens* at logged (β, δ,
Eb, R2) states replayed from C0 baselines — not live coupled runs, not ≥5 Myr θ_max calibration
data. Nothing here is quotable as a fire threshold. The live measurement is step L2 of the ladder
(§6).

## 4. Where this leaves the κ-side (f_κ) — keep it, bounded, as the *conduction* knob

f_κ remains the right knob for genuinely conduction-side physics, with literature-anchored ranges
(2026-07-06 sweep): **suppression f_κ ∈ [~1e-3, 1]** — magnetically draped interfaces ≤1e-2
(Markevitch & Vikhlinin 2007, Phys. Rep. 443, 1 — a cold front is geometrically TRINITY's contact
discontinuity), tangled fields 1e-3–1e-2 (Chandran & Cowley 1998), ~0.2 in the optimistic
Narayan & Medvedev 2001 case; **enhancement only as a moderate area/corrugation factor f_κ ~ 1–30**
— beyond that the Lancaster regime decouples the loss from conduction entirely and the functional
form must change (§6 L4), not the constant. Note the community's C itself carries a factor ~2
spread (Spitzer lnΛ≈15 → C≈1.2e-6 vs the Mac Low & McCray 1988 6e-7 TRINITY uses): an implicit
f_κ ≈ 0.5–1 of pure convention.

Any f_κ > 1 must be bounded by the **Cowie & McKee 1977 saturation ceiling** q_sat ≈ 5φ_s ρc_s³
(φ_s ≈ 0.3, El-Badry Eq. 19/20), which the solver currently never enforces (audited: no cap
anywhere in the bubble solve; `KAPPA_FREEZE_MECHANISM.md §6`). Implementation note from the
front literature: use the **Dalton & Balbus 1993 smooth classical↔saturated interpolation**, e.g.
κ_eff = f_κ·κ_S/(1 + f_κ·κ_S|∇T|/q_sat), NOT a hard min() — a kinked flux limiter feeds spurious
Jacobians to the Powell-hybrid residual (the same lesson κ_mix's hard-max already taught,
`KMIX_SELFCONSISTENT.md §2b` limitation A). It needs the generalized IC below (the closed-form
Eq-44 IC assumes pure T^{5/2}).

## 5. The generalized near-front IC — one first integral, four payoffs

The near-front layer equation the Eq-44 IC encodes is steady advection–conduction (planar, x =
R2−r): d/dx(κ dT/dx) = c_p F_ṁ dT/dx, with F_ṁ = ṁ/4πR2², c_p = (5/2)k_B/μ. Integrating once from
the wall with wall temperature T_w and **wall radiative flux q_w** gives the closed first
integral:

```
κ(T)·dT/dx = q_w + c_p·F_ṁ·(T − T_w)                                   (★)
```

where q_w ≡ κ dT/dx|_wall is the conductive flux delivered into the wall, physically disposed of
by the wall's radiative sink (hence "wall radiative flux"). Honesty note: (★) lumps all radiative
loss *inside* the anchor layer (T_w → T_init) into q_w rather than distributing it — the exact
first integral would add ∫n²Λ dx. That is the same neglect Weaver's Eq. 44 already makes (his
layer is cooling-free); (★) upgrades it from "no layer cooling" to "layer cooling lumped at the
wall", and the distributed correction is exactly what the resolved ODE handles above the anchor.
The anchor offset and gradient for ANY κ(T) are then one quadrature:

```
dR2 = ∫_{T_w}^{T_init}  κ(T) / (q_w + c_p·F_ṁ·(T − T_w))  dT
dT/dr|_anchor = −(q_w + c_p·F_ṁ·(T_init − T_w)) / κ(T_init)
```

Payoffs, in increasing order of ambition (each recoverable/testable in isolation):

1. **Weaver recovered exactly.** q_w=0, T_w=0, κ=C·T^{5/2} ⇒ dR2 = (4/25)(μC/k_B)T_init^{5/2}/F_ṁ
   — algebraically identical to `_get_bubble_ODE_initial_conditions` (verified against
   `bubble_luminosity.py:380-388`). Gate: bit-level equality of the specialized branch.
2. **The κ_mix boundary divergence is fixed, not dodged.** `KMIX_SELFCONSISTENT.md §2` finding 4:
   scaling the Spitzer closure dR2 ∝ C by R(T_init) explodes past R1. Under (★) with
   κ_mix = a/T (a ∝ λδv·Pb), the quadrature ∫κ/(c_p F_ṁ(T−T_w))dT converges for T_w > 0 (the
   physical 1e4 K photoionized wall) — the "κ_mix-specific boundary re-derivation" the doc flagged
   as future work is literally this formula.
3. **Saturation-capped κ (rung #3)** slots in as κ_eff(T, ∇T) via the same quadrature (piecewise
   where the limiter binds; the saturated stretch is linear-in-x).
4. **The condensation branch (rung #4) becomes an increment.** (★) stays positive — hence the
   profile family stays defined — for F_ṁ < 0 whenever q_w > c_p|F_ṁ|(T_init−T_w): a condensing
   front is representable *as long as the wall radiates more than the advected enthalpy release*,
   which is exactly the McKee–Cowie condensation criterion. The Weaver-family hard edge that fix
   #1 currently routes around (`T ∝ (ṁ·dR2)^{2/5}` undefined for ṁ<0) is an artifact of setting
   q_w ≡ 0. With q_w > 0 the v(R1)=0 eigenvalue solve can walk dMdt *through* zero smoothly, and
   "condensation onset ≈ cooling balance" stops being a solver cliff at all. Limits: q_w-dominated
   ⇒ κdT/dx ≈ q_w ⇒ T ∝ x^{2/7} — **El-Badry's cooled-interior profile family (his Eq. 44,
   τ ∝ (1−ξ)^{2/7})**, recovered as the opposite closure of the same integral. So Weaver (2/5) and
   El-Badry (2/7) are the two poles of (★), and the physical front sits between them, indexed by
   q_w.
5. **Closure for q_w:** q_w is the sub-anchor interface radiation per unit area — in the f_A
   model it is computable from the resolved T_w→T_init intermediate zone (the L₃ integrand, per
   area): q_w ≈ f_A·L₃/(4πR2²) evaluated on the previous iterate (cheap fixed-point; the L₃
   region is already reconstructed every solve at `bubble_luminosity.py:756-794`). Start with
   q_w=0 (bit-identical), turn on with f_A.

## 6. Recommendation + the gated ladder (rule-5 depths)

**Do not move the paper off `multiplier` f_mix=4** — it is calibrated, rule-compliant, adopted
(2026-07-02 ruling). The path below is how the *physical* knob earns its way in as the successor
(and as the referee defense for f_mix itself):

- **L0 (done, this doc):** offline f_A prototype on replayed states; gates G1/G2; predictions
  P1–P4. Artifacts: `data/fA_source_boost{,_summary}.csv`, `fA_source_boost.png`, FINDINGS §15.
- **L1 (cheap, next):** extend the screen to the 2 captured stiff fixtures (5e9, mild cluster) +
  the θ-peak early epochs; probe f_A up to the condensation onset per config and record the
  (f_A, epoch) edge map — the honest domain boundary, expected to land near θ≈1 (McKee–Cowie).
- **L2 (the real gate):** wire `cooling_boost_mode='source'` + `cooling_boost_fA` as a gated
  registry param (default 1.0, **byte-identical off** — same standard as `cooling_boost_kappa`,
  sha-gated over a stiff-config run), then the 📏 theta5-protocol matrix (9 configs × f_A grid,
  ≥5 Myr, θ_max, separate processes, `OMP_NUM_THREADS=1`) on HPC — the like-for-like shootout
  against `FINDINGS.md §10–§13`: does a single f_A fire the whole normal-GMC band (multiplier
  [4,4.5] 6/6, kappa best 5/6)? Also read the emergent dMdt(t) suppression against El-Badry
  Eq. 47 — the first *fidelity* measurement, impossible for the multiplier by construction.
- **L3 (structural, separate workstream):** the (★) IC behind a flag: first as pure refactor at
  q_w=0 proving bit-identity (rule-5 "free win" standard: value-diff vs `git show HEAD` + A/A
  control for FP nondeterminism, matched t, separate processes), then q_w>0 + the condensation
  walk-through, retiring the fix-#1 handoff from "domain-edge escape" to "genuine transition
  criterion". The saturation cap rides this flag too.
- **L4 (the faithful endpoint, optional):** replace the constant f_A with a state-coupled
  interface loss — El-Badry's local closed form L_int = 4π√(αλδv)·R²Pb^{3/2}√Λ(T_pk)/(k_B T_pk)
  (`ELBADRY_REFERENCE.md §7` — needs only R, Pb, λδv, T_pk, all in-solve), or equivalently the
  Lancaster/Fielding channel Ė_int = (5/2)·P·v_in·A_eff with v_in ≈ c_s,cold(t_cool/t_sc)^{−1/4}
  and A_eff = 4πR²(L/λ)^{2−D}, D ≈ 2.5 (Fielding+2020; Lancaster+2021a/b, +2024 ApJ 970, and the
  2025 CEM papers arXiv:2505.22730/22733 — the closest published competitor model, validated
  ~25% against 3D RMHD). Then θ's density-dependence *emerges* (Pb^{3/2} at the interface) and
  λδv is again a physical constant, now on the source side where it is NOT born-saturated.

**Efficiency (the maintainer's other axis; keep separate from physics changes, own ladder):**

- The cost hotspot is the *nested* iteration: hybr over (β,δ) × per-trial dMdt fsolve × ODE
  integrations (the §8d diffuse cliff is this product). Candidate: **flatten to one 3-unknown
  root-find** F(β,δ,dMdt) = (energy residual, T residual, v(R1) residual) with a shared
  warm-start; removes the multiplicative iteration count. High-risk/high-reward — full rule-5
  ladder (per-call ⇒ full-run equivalence on `simple_cluster` + both `f1edge_*`, separate
  processes, matched t) before any default flip.
- The (★) formulation also enables integrating the structure in the **Kirchhoff variable**
  y = ∫κdT (linear conduction operator; for Spitzer y ∝ T^{7/2}, so the near-front layer becomes
  y ∝ x^{7/5} — regular — vs T ∝ x^{2/5} with divergent T″) — the natural fix for the LSODA
  micro-stepping across the ~1e-10 pc layer documented in
  `docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md`. This is standard practice in the
  nonlinear-conduction (Marshak / thermal-wave) literature — Zel'dovich & Raizer Ch. X;
  Hammer & Rosen 2003 — where such fronts are integrated by shooting from the *analytic*
  near-edge asymptote, exactly the Eq-44-anchor strategy TRINITY already uses. Same gate
  standard.

**Cheap analytic regression gates for any κ-side change (from the 2026-07-06 lit sweep):** the
classical self-similar scalings double as solver gates on an f_κ grid in the energy phase at
fixed t — T_c ∝ f_κ^{−2/7}, Ṁ ∝ f_κ^{2/7}, L_cool ∝ f_κ^{~5/7} — plus bit-identity at f_κ=1.
Deviations beyond the (β,δ) back-reaction envelope flag a broken IC/limiter before any expensive
full-run gate is spent. ([D]-grade exponents from verified ingredients; re-verify against the
Weaver appendix PDF before hard-coding — the appendix eq. numbers were not reachable in the
sweep.)

## 7. Reproduce

```bash
# the offline prototype (reads committed cleanroom trajectories; ~8 min, no sims):
python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py
# smoke variant:
FA_LIST=1,4 N_ROWS=2 CONFIGS=simple_cluster python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py
```

Artifacts: `data/fA_source_boost.csv` (per row × f_A), `data/fA_source_boost_summary.csv`
(per config × f_A), `fA_source_boost.png` (θ(t), θ_max(f_A) dial, dMdt(f_A)/dMdt(1) coupling).

## 7b. Citations for the eventual paper section (+ verification status, 2026-07-06 lit sweep)

Core: Spitzer 1962; Weaver et al. 1977 (ApJ 218, 377); Cowie & McKee 1977 (ApJ 211, 135); McKee &
Cowie 1977 (ApJ 215, 213); Dalton & Balbus 1993 (ApJ 404, 625); Mac Low & McCray 1988 (ApJ 324,
776 — the C=6e-7 source). f_κ precedent: Gupta et al. 2018 (MNRAS 473, 1537). Interface physics
in 1D: El-Badry et al. 2019 (MNRAS 490, 1961). Turbulent regime: Fielding et al. 2020 (ApJL 894,
L24); Tan, Oh & Gronke 2021 (MNRAS 502, 3179 — entrainment converges without resolving the Field
length: the transport coefficient, not the microstructure, sets global rates); Lancaster et al.
2021a/b (ApJ 914, 89/90), 2024 (ApJ 970), 2025 CEM (arXiv:2505.22730/22733). Suppression bounds:
Markevitch & Vikhlinin 2007; Chandran & Cowley 1998; Narayan & Medvedev 2001. Numerical-mixing
caution for any 3D-calibrated f: Gentry & Krumholz 2019 (MNRAS 483, 3647 — Eulerian mixing rates
are numerical-diffusion-dominated; treat as upper limits). Observational budget: Rosen et al.
2014 (MNRAS 442, 2701 — only 3–30% of wind energy accounted; mixing losses implicated).

⚠️ **Unverified-number flags carried from the sweep (full texts were unreachable; check PDFs
before hard-coding):** Weaver appendix equation numbers + exact near-edge coefficient; the CM77
saturated-reduction exponent (−5/6 vs −5/8 both circulate); El-Badry's exact κ₀ and turbulent
diffusivity magnitude; Lancaster 2021a's v_in equation numbers.

## 8. Sibling reconciliation (done with this commit — keep true)

- `INDEX.md` §2 table + §3 thread: this doc + FINDINGS §15 registered.
- `KAPPA_EFF_SCOPING.md §6.2`: the "second offline prototype (in-structure L_mix → ΔdMdt sign)"
  is now RUN — pointer here.
- `KMIX_SELFCONSISTENT.md §2` finding 4 / §3 route 3: the boundary re-derivation now has a
  concrete formula (§5 payoff 2) — pointer here. Route 0 (smooth-max + kprime) remains open and
  is *complementary* (κ-side), not superseded.
- `KAPPA_FREEZE_MECHANISM.md §7`: rung #3/#4 now have a concrete design (§5 payoffs 3/4).
- `FINDINGS.md §15`: the prototype result (ledger entry).
- `PLAN.md` ledger: dated entry pointing here.
