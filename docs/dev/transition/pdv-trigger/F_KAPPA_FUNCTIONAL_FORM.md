# f_κ(n_H) — a closed-form calibration target (composed, not fitted-cold)

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `REPRODUCE.md`, `runs/README.md`,
> `NOTE_PATCHES.md`, `KAPPA_EFF_SCOPING.md`, `RUNGB_SCOPING.md`, and any other notes in the same folder).
> They drift out of sync *with each other* as fast as they drift from the code. Any agent or person editing
> one MUST, as part of the visit, circle back through the siblings and reconcile: if a number, status, claim,
> or line reference here contradicts a sibling — or a sibling has gone stale — fix it (or flag it, dated) so
> no two docs in the workstream disagree. Never update one in isolation.

---

> ✅ **SWEEP RESULTS ARE IN (2026-06-29) — see §8.** The 819-combo grid ran on Helix. The composed form
> below (slope −0.30) was a **pre-registered prediction**; the **measured** central trend is steeper:
> **f_κ_fire ≈ 1.0×10³·n_core^(−0.60)** (θ\*=0.95). Scorecard: **fan-out confirmed** (f_κ is multi-dimensional,
> ×2–32 spread at fixed n; the de-conflation answer is "does NOT collapse to one n_H curve") ✅; **diffuse end
> needs κ_mix** (6/63 low-n high-sfe cells never fire even at f_κ=64) ✅; **slope was 2× too shallow** ❌
> (my 6-anchor baseline θ₀(n) gave 0.41/dex; the real grid gives 1.13/dex). Use §8's measured numbers for any
> magnitude; the §0–§6 composition below is the (partly-wrong) reasoning that the sweep tested. Artifacts:
> `data/fkappa_nH_sweep.csv`, `data/make_fkappa_sweep_analysis.py`, `fkappa_sweep_analysis.png`.

## 0. TL;DR — the form you can use now

```
   f_κ(n_H)  =  ( θ* / θ₀(n_H) )^(1/p)          [raw power; matches the measured firing anchor]

   with   θ*            = 0.90            target loss fraction  (Lancaster 2021 plateau, density-independent)
          logit θ₀(n_H) = −1.73 + 0.41·log₁₀(n_H)   TRINITY emergent baseline at f_κ=1 (fit, 6 anchors)
          p             ≈ 0.31            leverage of θ on f_κ, measured over the FULL range to firing
                                          (NOT the low-f_κ logit slope — see §3; range 0.21–0.42 by config)

   ≈ power law:   f_κ(n_H) ≈ 1.4×10² · n_H^(−0.30)         (θ*=0.90)
                  f_κ(n_H) ≈ 1.6×10² · n_H^(−0.30)         (θ*=0.95 = the shipped trigger)
```

| n_Core [cm⁻³] | baseline θ₀ (f_κ=1) | f_κ for θ\*=0.90 | f_κ for θ\*=0.95 | measured anchor |
|---:|---:|---:|---:|---:|
| 1e2 (diffuse) | 0.25 | **≈ 40** | ≈ 48 | >4 (unmeasured) |
| 1e4 (mid)     | 0.61 | **≈ 8**  | ≈ 9  | >4 (unmeasured) |
| 1e5 (compact) | 0.67 | **≈ 4**  | ≈ 5  | **≈ 3.4 (fires at f_κ=4)** |
| 1e6 (dense)   | 0.70 | **≈ 2.5**| ≈ 3  | — |

> ⚠️ The **slope (−0.30) is robust**; the **diffuse-end magnitude is uncertain by ~2–3×** (extrapolated from
> f_κ≤4 data where diffuse only reaches θ=0.30 — see §4). The diffuse value being in the **tens** is the
> result: a pure Spitzer-conduction boost of ~50× **cannot** physically reach the Lancaster plateau for diffuse
> clouds (§5, the saturation ceiling) — that regime needs El-Badry's κ_mix. The 819-combo HPC sweep pins `p(n_H)`.
>
> 🛠 **Correction (2026-06-29, same day):** the first cut of this doc inverted the leverage in **logit/odds
> space** (`q≈0.55`) and got f_κ ≈ 291 (diffuse) … 121 (compact) — **wrong by ~10–30×** at the one *measured*
> anchor (compact **fires at f_κ≈3.4**, not ~120). Cause: θ(f_κ) **accelerates toward firing** (convex:
> compact 0.667→0.739→**1.024**), because the bubble *transitions before θ saturates* — so a saturating
> (concave) logit extrapolated from the f_κ∈{1,2} segment overshoots. The fix is the **raw power-law exponent
> measured over the full range to firing** (p≈0.31), which reproduces the measured anchor and agrees with the
> independent El-Badry-back-reaction estimate (q≈0.33–0.45). Only the *amplitude* changed; the slope did not.

This is **not** a literature formula — there is **no published `f_κ ∝ n_H^p` law** (§3). It is *composed* from
three separable, independently-checkable pieces: a verified literature **target**, a measured TRINITY
**baseline**, and a measured **leverage**. Each can be replaced/refined without touching the others.

> ✅ **Target cross-check (2026-06-29):** El-Badry's `θ(n_H,λδv)` (Eq 37/38, **now verified from the PDF** —
> §2.1) saturates to 0.94–0.999 across the GMC range, so using it as the target gives f_κ ≈ 46/11/3.6
> (diffuse/mid/dense) — **within ~15% of the flat-Lancaster numbers above.** Both verified anchors agree; the
> form is robust to the target choice.

Artifact: `data/make_fkappa_functional_form.py` → `data/fkappa_functional_form.csv` + `fkappa_functional_form.png`.
Reproduce (no sims): `python docs/dev/transition/pdv-trigger/data/make_fkappa_functional_form.py`.

---

## 1. What f_κ is (verified in code, this branch)

`cooling_boost_kappa` (= f_κ), default `1.0`, multiplies the Spitzer–Härm conduction coefficient
`C_thermal = 6e-7` in **κ_eff(T) = f_κ · C_th · T^(5/2)**. Verified at three sites in
`trinity/bubble_structure/bubble_luminosity.py` (param read `registry.py:352`):

- `:291` dMdt seed (Weaver+77 Eq.33) ⇒ **dMdt ∝ f_κ^(2/7)** (measured 1.2175 vs 2^(2/7)=1.219 at f_κ=2);
- `:370` conduction-layer ICs (Eq.44) ⇒ layer thickness ΔR₂ ∝ f_κ at fixed dMdt;
- `:406` T-curvature ODE (Eq.42–43) ⇒ enters as 1/(f_κ·C_th·T^(5/2)).

It is **structural**, not a multiplier on `L_cool`: the loss fraction **θ = L_cool/L_mech emerges** as an
output (thicker front → more 10⁵–10⁶ K gas in the cooling-function peak). This is why a *functional form* for
f_κ is non-trivial — you can't read θ off f_κ algebraically, you invert a measured response curve.

**Physical reading (verified literature, §3):** classical Spitzer conduction is *suppressed* below 1 by tangled
magnetic fields (Narayan & Medvedev 2001 → f≈0.2; ISM-standard f~0.1). So **f_κ > 1 is not literal extra
Spitzer conduction** — it is a **proxy for turbulent-mixing-enhanced interface transport** (El-Badry/Lancaster),
folded into the one knob TRINITY exposes.

---

## 2. The composition (the three pieces)

### (1) TARGET θ\* — Lancaster plateau, density-independent  *(verified)*
Lancaster, Ostriker, Kim & Kim 2021 (Paper I ApJ 914, 89 = arXiv:**2104.07691**; Paper II 914, 90 =
**2104.07722**): turbulent mixing at a fractal interface (dimension 2.5–2.7) radiates the **vast majority** of
wind energy — retained fraction **1−Θ ~ 0.1–0.01, decreasing with time** — and (abstract, verbatim) this is
*"generic ... over more than three orders of magnitude in density."* So over the GMC range θ\* is **flat-and-high
≈ 0.9** (we report 0.95 too, = the shipped trigger). Momentum boost α_p ~ 1.2–4.

**El-Badry's θ(n_H, λδv) — now VERIFIED from the PDF (2026-06-29), and it AGREES with Lancaster.** El-Badry
et al. 2019 (MNRAS **490, 1961**; arXiv:1902.09547 — author **Weisz**, *not* Weinberg; *not* "ApJ 879"). The
specific algebra, confirmed against the paper's §3.1/§5.2 (PDF supplied by the maintainer; earlier `[unverified]`
hedge **retracted** — it was a 403 access gap, not an error, and the prior room's transcription was correct):
- **Eq 37**: `ψ ≡ L_int/Ė_th = A_mix·(λδv)^½·(n_H,0)^½`, with **A_mix ≈ 1.7 analytic (α=1, T_pk=2×10⁴ K), ≈ 3.5
  fit to their sims**; λδv in pc·km/s, n_H,0 the **ambient** density.
- **Eq 35/38**: `L_int/Ė_th = (11/5)·θ/(1−θ)` ⇒ `θ = ψ/(11/5 + ψ)`. Fiducial λδv=n_H=1 → ψ=3.5, θ=0.61 ✓.
- **Eq 21**: the mixing term is `κ_mix = (λδv)·ρ k_B/(μ m_p)` — a **temperature-INDEPENDENT** conductivity, with
  `κ = max(κ_mix, κ_Spitzer)`; κ_mix dominates where T ≲ 2×10⁵ K and n_H ≳ 0.2 cm⁻³. (This is the genuine
  "Rung-B" κ_mix; λδv is varied 1–10 pc·km/s.)
- **θ is independent of time** and depends on ambient ρ₀ but not Δt_SNe (their §5.2).

The one real caveat stands: n_H,0 is the **ambient** density and El-Badry's domain is **0.1–10 cm⁻³**, so GMC use
(1e2–1e6) is **extrapolated**. But the √n form **saturates**: θ_EB(λδv=1) = 0.94 (1e2) → 0.99 (1e4) → 0.999
(1e6), nearly λδv-independent there — i.e. **flat-and-high, matching Lancaster's plateau**. Using θ_EB as the
target gives f_κ ≈ **46 / 11 / 3.6** (diffuse/mid/dense), within ~15% of the Lancaster-θ\*=0.95 values (48/9/3).
So **both verified anchors give the same f_κ(n_H)** — the form is robust to the target choice. The density-shape
of the *target* is essentially flat where GMCs live; the density dependence of f_κ comes from θ₀(n_H) rising
(piece 2), exactly as El-Badry would predict (their θ flat, ours rising → the gap closes density-dependently).
**Bonus (verified, p6):** El-Badry *themselves* propose "use the cooling rates from 3D simulations as a
calibration point and adjust λδv to match their energetics" — i.e. they prescribe **this workstream's exact
strategy**.

### (2) BASELINE θ₀(n_H) — TRINITY's emergent loss fraction at f_κ=1  *(measured)*
Resolved L_cool/L_mech at blowout for 6 reference configs (`data/fmix_table.csv`), **rising** 0.25 (n=1e2) →
0.70 (n=1e6). Fit `logit(θ₀) = −1.73 + 0.41·log₁₀(n_H)`, RMS = 0.49 in logit. The RMS scatter is real (e.g.
`pl2_steep` and `simple_cluster` both sit at n=1e5 but θ₀=0.34 vs 0.67 — density-profile steepness + SFE matter),
which is exactly the **de-conflation** question the 819-sweep answers: is f_κ-to-target a function of n_H alone,
or also of mCloud/SFE/profile? **The density structure of f_κ comes from this rising baseline under a flat
target** — this is what dissolves the FINDINGS §2a worry that "flat target == the 0.95 trigger, no new content":
that equivalence only holds for a *linear* L_cool multiplier (`f_mix`), not for the *structural* f_κ knob whose
leverage is sub-linear and saturating (piece 3).

### (3) LEVERAGE p — how θ responds to f_κ  *(measured; this is the piece I first got wrong)*
Full-run grid `data/kappa_blowout_calibration.csv` (f_κ = 1,2,4 on compact/mid/diffuse). Two ways to read the
exponent, and the choice matters by ~10–30×:

- **The existing `kappa_calibration_estimate.csv` uses θ ∝ f_κ^0.63** — but that 0.63 was measured on *early*
  snapshots (θ≈0.01); at blowout the effective exponent is weaker, so that estimate self-labels "optimistic".
- **A logit/odds-space slope fit on the low points (f_κ∈{1,2})** looks attractive (bounded by θ→1) but is
  **wrong here** — it overshoots the measured firing anchor by ~10–30× (the 🛠 correction in §0). θ(f_κ) does
  **not** saturate: the bubble **fires** (transitions) before it does, so the curve is *convex* (compact
  0.667→0.739→**1.024** at f_κ=1,2,4), and a concave logit extrapolated from the bottom segment under-reads the
  acceleration.
- **What works: the raw power-law exponent fit over the FULL measured range, including the firing point**
  `ln θ = ln θ₀ + p·ln f_κ`. Measured **p = 0.31 (compact) / 0.21 (mid) / 0.42 (diffuse)**, median **0.31**.
  This reproduces the measured anchor (compact crosses θ=0.95 at **f_κ≈3.4**) and matches the independent
  El-Badry-back-reaction estimate (`f_mix = f_κ^q`, q = ln1.3/ln2 ≈ 0.4). It is **non-monotonic in n_H** (mid
  is the lowest), i.e. leverage depends on more than density — the de-conflation the 819-sweep resolves.

The functional form is then the raw-power inversion of θ = θ₀·f_κ^p:  **f_κ(n_H) = (θ\*/θ₀(n_H))^(1/p).**

---

## 3. The literature answer to "is there a functional form?" — **no off-the-shelf one**

The survey (10 sub-agents; most primary PDFs were 403-blocked in-container so those rows are from search snippets
+ citing papers with equation *numbers* flagged — **except El-Badry, whose PDF the maintainer supplied, so its
rows are now PDF-verified**) is unambiguous: **no paper writes a conduction/mixing enhancement factor as
`f ∝ n_H^p`.** The density-powers that *do* exist in the literature:

| relation | density power | source | status |
|---|---:|---|---|
| classical Spitzer κ = 6e-7·T^(5/2) | **n_H⁰** (none) | Spitzer 1962; Weaver+77; El-Badry Eq 16 | **verified (PDF + multi-source)** |
| El-Badry mixing κ_mix = (λδv)ρk_B/μm_p | **n_H¹** (T-independent) | El-Badry+2019 Eq 21 | **verified (PDF)** |
| El-Badry cooling efficiency θ(n_H,λδv) | √n_H, **saturates** | El-Badry+2019 Eq 37/38 (A_mix=3.5) | **verified (PDF)** |
| saturated heat flux q_sat = 5φ_s ρ c_s³ (φ_s≈0.3) | **n_H¹** | Cowie & McKee 1977; El-Badry ftn 4/Eq 19-20 | **verified (PDF)** |
| ⇒ effective κ in saturated limit | **n_H¹** | (from q_sat·ℓ_T/T) | derived |
| saturation parameter σ₀ = q_cl/q_sat | **n_H⁻¹** | Cowie & McKee 1977 / Balbus & McKee 1982 | def. verified; eq.# not |
| conduction-modified Weaver shell density ρ_sw | **n_H^(19/35)≈0.54** | Gupta, Nath & Sharma 2018 (MNRAS 473,1537) | verbatim snippet; eq.# not |
| turbulent diffusivity D (Greif/Klessen-Lin) | **ρ¹** (convention) | Greif+2009; Smagorinsky family → ρ⁰ | verified |
| Lancaster cooling efficiency Θ | **n_H⁰** (density-independent) | Lancaster+2021 | verbatim (abstract) |
| terminal SN momentum p_t | **n_H^(−0.06)** | Gentry+2017 | verbatim |
| metallicity Z on interface cooling | weak (Lyα-dominated) | El-Badry+2019 | qualitative, no exponent |

**Reading:** the only clean density power for *effective conduction* is the **saturated branch κ_sat ∝ n_H¹**
(rising) — which is the **ceiling**, not the target (§5). The *target* (Lancaster Θ) is **density-independent**.
So the n_H-dependence of f_κ is **not** inherited from any single literature scaling; it **emerges** from
inverting TRINITY's rising baseline against a flat target. Our composed result `f_κ ∝ n_H^(−0.30)` is therefore
a **TRINITY-specific calibration curve**, with the literature supplying the *target value* and the *ceiling*,
not the slope.

*(Adjacent mixing-layer scalings, for the κ_mix line of work: Tan/Oh/Gronke 2021/2023 — TML brightness ∝ M^0.5
subsonic, saturating ∝ M⁰ supersonic; strong-cooling inflow v_in ∝ u′^(3/4)(L/t_cool)^(1/4); Da ≡ τ_turb/t_cool.
Fielding+2020 fractal D=5/2. These feed a future temperature-independent κ_mix(properties), not f_κ·Spitzer.)*

---

## 4. Honest uncertainty — what the sweep is for

- **Magnitude at the diffuse end is extrapolated.** The leverage is measured only at f_κ∈{1,2,4}, and only the
  *compact* run actually reaches the target there (it fires at f_κ≈3.4); diffuse only reaches θ=0.30 at f_κ=4.
  So f_κ≈48 (θ\*=0.95, n=1e2) is a raw-power extrapolation, uncertain ~2–3× (the docs' older estimate gives
  ≈60). What is *not* in doubt: diffuse needs **tens×** the boost dense needs, and probably more than conduction
  can physically supply (§5).
- **p is treated as constant but varies with config.** Measured p = 0.21 (mid) … 0.42 (diffuse), non-monotonic
  in n_H — so leverage depends on more than density (SFE/profile). A p(n_H, mCloud, SFE) changes the amplitude
  (not much the slope). De-conflating it is the sweep's job (`data/reduce_fkappa_sweep.py` →
  `data/make_fkappa_nH_sweep.py`).
- **θ₀(n_H) has real scatter** (RMS 0.49 in logit) from profile steepness + SFE at fixed n_H (e.g. `pl2_steep`
  vs `simple_cluster`, both 1e5, θ₀=0.34 vs 0.67). If the scatter is structured (not noise), f_κ is a function
  of *more than* n_H — the central de-conflation result.

---

## 5. The physical bracket — why diffuse may be unreachable by f_κ at all

A real Spitzer-conduction boost **saturates**: the heat flux cannot exceed q_sat = 5φ_s ρ c_s³ (Cowie & McKee
1977 — adopted by El-Badry as q_sat = (3/2)ρc_s,iso³ with φ_s=0.3, Eq 19/20), so the effective conductivity
ceiling scales as **κ_sat ∝ n_H¹** — it **rises** with density. The *required* f_κ **falls** with density
(∝ n_H^(−0.30)). These run in **opposite directions**, so they cross:

- **Dense clouds:** required f_κ is small (~3) and the ceiling is high → reachable by conduction boost. ✓
- **Diffuse clouds:** required f_κ is large (~tens) but the ceiling is **low** → a pure f_κ·Spitzer boost is
  **unphysical** there. Reaching the plateau in diffuse gas needs the **temperature-INDEPENDENT turbulent-mixing
  conductivity κ_mix = (λδv)ρk_B/μm_p** (El-Badry Eq 21, now verified), implemented as `κ = max(κ_mix, κ_Spitzer)`
  — *not* a multiple of Spitzer. ✗ Note El-Badry's own κ_mix ∝ n_H¹ **rises** with density (it is a diffusivity
  ×ρ), the opposite sense to the *required-f_κ-vs-n_H* curve — because the two answer different questions
  (κ_mix matches a *conductivity*; f_κ(n_H) matches a *target θ*). The crossover where f_κ exceeds the ceiling is
  where TRINITY must switch from the f_κ knob to a κ_mix term.

This is consistent with the workstream's earlier Rung-A/Rung-B framing (`RUNGB_SCOPING.md`, `KAPPA_EFF_SCOPING.md`):
f_κ (Rung A) is the right *mechanism* and a usable calibration knob in the dense/compact regime; the diffuse end
is where a structural κ_mix (Rung B) is genuinely needed. The functional form makes the **boundary quantitative**:
it is roughly where `f_κ(n_H)` from §0 exceeds the local saturation ceiling — to be pinned once the sweep gives
the real q(n_H) and a front-temperature/scale-length estimate fixes the ceiling magnitude.

---

## 6. How to use / extend

- **Use now:** plug `f_κ(n_H) = (0.90 / θ₀(n_H))^(1/0.31)` with `θ₀(n_H) = logistic(−1.73 + 0.41·log₁₀ n_H)`
  (or the `≈140·n_H^(−0.30)` power law) as the provisional calibration target for the dense/compact regime;
  treat the diffuse end as a lower bound that likely needs κ_mix.
- **Refine after the sweep:** replace the single p with the measured p(n_H[, mCloud, SFE]); re-fit θ₀ on the
  819-grid baselines; re-emit this curve. The builder reads only committed CSVs, so swapping in
  `summary.csv`-derived anchors is a one-function edit (`_read_baselines`, `_measure_leverage`).
- **Do NOT ship it.** Per the workstream's hard constraint, θ/El-Badry/Lancaster/κ knobs are **dev-only paper
  diagnostics**; production stays byte-identical with the modes off (default trigger `cooling_balance` @ 0.95).

**Strategy note (degeneracy escape).** Because θ\* is ~flat where GMCs live (§2.1), calibrating "to θ\*(n_H)"
does **not** break the trigger degeneracy *through the target's density shape* — the density dependence comes
almost entirely from the rising **θ₀(n_H)**. The cleaner escape is to **calibrate f_κ once to the physical
θ\* (via κ_mix/λδv) and let the transition *time* emerge**: the falsifiable, degeneracy-free output is then the
**ordering** — which clouds transition before blowout (dense) vs blow out energy-driven first (diffuse) — which
is comparable to PHANGS. That θ\*≈0.95 coincides with the shipped trigger is then a *physical fact about GMC
densities*, not circular tuning. (Credit: external review, 2026-06-29.)

---

## 8. MEASURED — 819-combo sweep results & prediction scorecard (2026-06-29)

The controlled grid (7 nCore × 3 mCloud × 3 sfe = 63 cells × 13 f_κ) ran on Helix; reduced to
`data/fkappa_nH_sweep.csv` (per-cell θ(f_κ) fit + measured firing f_κ). Scored against the predictions
this doc pre-registered **before** the data existed (`data/make_fkappa_sweep_analysis.py` →
`data/fkappa_sweep_scorecard.csv`, `fkappa_sweep_analysis.png`):

| # | pre-registered (§0–§3) | **measured (63 cells)** | grade |
|---|---|---|---|
| P1 slope | f_κ ∝ n^(−0.30) | **f_κ_fire ≈ 1.0×10³·n_core^(−0.60)** | ❌ 2× too shallow |
| P2 de-conflation | fan-out, not one n_H curve | **×2–32 spread across mCloud/sfe at fixed n** | ✅ |
| P3 baseline θ₀(n) | logit slope **0.41**/dex | logit slope **1.13**/dex (`logit θ₀ ≈ −3.4 + 1.13·log₁₀n`) | ❌ ~3× steeper |
| P4 leverage p | 0.31 | median **0.21** (IQR 0.11–0.26) | ⚠ ballpark, point high |
| physical | diffuse unreachable by f_κ → κ_mix | **6/63 cells never fire at f_κ≤64** (all low-n, high-sfe) | ✅ |

**What this means.**
- **The qualitative physics held.** f_κ falls steeply with density; it is **multi-dimensional** (not f(n_H)
  alone — the fan-out); and the **diffuse, high-sfe corner is genuinely unreachable by a Spitzer boost** — the
  6 never-fire cells are exactly where §5's saturation argument said you must switch to the El-Badry κ_mix
  (Eq 21). Those are the load-bearing conclusions and they are now *measured*, not argued.
- **The slope was 2× too shallow, and §0–§3 names the cause:** the composed form is only as good as its
  baseline θ₀(n), and my **6-anchor θ₀(n) fit (0.41/dex) was badly undersampled** — the clean 63-cell grid
  gives **1.13/dex**. A steeper θ₀(n) under a flat target ⇒ a steeper f_κ(n). The logistic-vs-raw-power leverage
  debate (§3) turned out to be second-order next to this baseline error.
- **The corrected central form** (use this for magnitudes): **f_κ_fire ≈ 1.0×10³·n_core^(−0.60)** for θ\*=0.95
  → ≈ 65 (n=1e2) / 17 (1e3) / 4 (1e4) / 1 (1e5) — **but** with ×3–30 mCloud/sfe scatter, so quote it as a trend
  with a band, not a point. The fan-out is the real headline: **f_κ(n_H) alone is not a sufficient
  parametrization** — a usable calibration needs (n_core, mCloud, sfe), or a switch to the structural κ_mix.

**Next (post-sweep):** the de-conflation says calibrate on more than n_H. Two clean follow-ups — (a) regress
the measured f_κ_fire on (n_core, mCloud, sfe) to find the second axis (started in §9); (b) given the never-fire
corner, spec the gated El-Badry **κ_mix = (λδv)ρk_B/μm_p** mode (Eq 21, verified §7) for the diffuse end,
default-off byte-identical. Both are dev-only.

---

## 9. Anatomy of the fan-out — the catastrophic-cooling cliff (2026-06-29)

Inspecting the faceted figure (`fkappa_nH_sweep.png`, three panels by sfe), the **1e7 series visibly "breaks
the power law"**: it stays high then drops abruptly to f_κ=1. That cliff is the key to the fan-out, and it is
*physics*, not a plotting artifact. Builder: `data/make_fkappa_cliff_metric.py` →
`data/fkappa_cliff_metric.csv` + `fkappa_cliff_metric.png` (reads only `data/summary.csv`, no sims).

**The cliff.** For each cloud, the baseline θ@f_κ=1 (no boost) rises with density and then **jumps past 0.95**
— above that threshold the cloud fires the cooling transition with **zero boost**, so f_κ_fire collapses to 1.
The cliff sits at *lower density* for *more massive* clouds:

| cloud | θ@f_κ=1 crosses 0.95 at nCore |
|---|---|
| M=1e5 | ≈ 2×10⁴ |
| M=1e6 | ≈ 1×10⁴ |
| M=1e7 | ≈ 3×10³ |

**Why — and the partial collapse variable.** At fixed density a 1e7 cloud is ~4.6× larger (rCloud ∝
(M/n)^{1/3}), so it sweeps the same **column** `N_H = nCore·rCloud` at lower density. Re-plotting θ@f_κ=1 vs
column instead of density **roughly halves the cliff spread** (×11 in nCore → **×5.7** in column; median cliff
column ≈ **8×10²³ cm⁻²**, range ~2×10²³–10²⁴). So the cliff is approximately a **constant-column catastrophic-
cooling threshold**: the bubble cools to completion *before escaping the cloud* once it has swept enough column.
Physically this is "does catastrophic cooling beat cloud crossing" — for massive clouds (large rCloud) cooling
wins at lower ambient density. *(This is also why your earlier intuition that the 1e7 cloud "needs less boost"
is correct — but the driver the data supports is the swept column, not PdV directly; the firing metric here is
the radiative θ=L_cool/L_mech, and f_κ_fire is independent of cluster mass M★=sfe·mCloud, R²=0.002.)*

**It does NOT fully collapse — the fan-out is genuinely multi-dimensional.** Across all 63 cells, the *single*
best predictor of the baseline θ@f_κ=1 is **nCore** (R²=0.73); column is slightly worse globally (R²=0.71) even
though it nails the cliff onset; rCloud alone is poor (R²=0.33). A 2-variable fit
`θ ∝ +0.11·ln(nCore) + 0.06·ln(rCloud)` reaches R²=0.75 — a modest lift, with the nCore coefficient ~2× the
rCloud one (so it is *not* pure column). **Reading:** nCore is the primary axis; cloud size (via rCloud/column)
is a real but secondary axis whose effect is **concentrated at the cliff**, where it controls whether a cloud
fires with no boost at all. sfe shifts the curves too (compare the three panels). A clean calibration therefore
needs `f_κ(nCore, rCloud[, sfe])`, or the structural κ_mix for the corner that never fires.

## 10. The measurement metric — θ at blowout (is it a good choice?)

**What is measured.** θ = `bubble_LTotal`/`Lmech_total` (the radiative loss fraction L_cool/L_mech), sampled
**per timestep during the energy-driven (implicit) phase** — *not* at a fixed t, *not* integrated to stop_t.
Per run the reducer (`reduce_fkappa_sweep.py`) keeps two scalars: **`theta_blowout`** = θ at the first timestep
where **R2 > rCloud** (the bubble reaches the cloud edge — "blowout"; falls back to the peak if it cools before
escaping), and **`theta_max`** = the peak θ over the implicit phase. **"Fires"** = reached the transition/
momentum phase **AND** (never blew out **OR** `theta_max ≥ 0.95`). So `f_κ_fire_measured` is `theta_max`-based;
`f_κ_fire_fit` extrapolates `theta_blowout`.

**Why blowout.** The science question is *does the cluster transition to momentum-driven while still inside the
GMC?* Blowout (R2=rCloud) is the natural end of the in-cloud phase — past it the bubble is in the ambient
medium and the in-cloud feedback question is settled. Measuring to a fixed time or to stop_t would fold in
post-escape ambient evolution that is irrelevant to that question. The runs split cleanly into the two regimes
the metric is meant to separate: **403/819 cooled before escaping** (fire in-cloud) vs **416/819 reached
blowout** (energy-driven escape unless θ hit 0.95 first).

**Is it a good metric? Yes — and it's robust.** Empirically the snapshot-vs-peak distinction barely matters:
`theta_max − theta_blowout` has **median 0.004** (>0.05 in only **5/63** cells), so the calibration is
insensitive to that choice. The cliff/fan-out is genuine physics, not a metric artifact.

**One precision caveat (a fixable imprecision, not a fatal flaw).** `theta_max` is taken over the *whole*
implicit phase, **not capped at `blowout_t`**. So a cell that blew out at θ=0.6 and only later peaked at
θ=0.96 *in the ambient medium* would be tagged "fired" — but that firing is post-escape, not in-cloud. This
touches only the ~5 cells where `theta_max ≫ theta_blowout`. For a strict "fired **in-cloud**" criterion,
`theta_max` should be capped at `blowout_t` in the reducer (needs the per-run jsonl, cluster-side). Two
alternative metrics answer *different* questions and could be added if wanted: **θ at matched physical time**
(apples-to-apples leverage, removes the variable-epoch confound) and **time-integrated** ∫L_cool dt / ∫L_mech dt
to stop_t (the total energy budget, not the transition).

---

## 11. Don't force it — physically-bounded f_κ and a critical-column prediction (2026-06-29)

A reframing prompted by the never-fire corner and the f_κ=64 magnitudes. Searching f_κ up to 64 to make
*every* cloud fire quietly assumes every cloud **must** become momentum-driven. It shouldn't: if a cloud cannot
reach θ=0.95 with a *physically-bounded* enhancement, the honest answer is that it stays **energy-driven and
blows out** — not that it "needs more boost." Builder: `data/make_fkappa_physical_cap.py` →
`data/fkappa_physical_cap.csv` + `fkappa_physical_cap.png`.

**Why f_κ=64 is not "enhanced conduction."** A large *constant* f_κ multiplies the Spitzer T^(5/2) prefactor
**everywhere**, over-conducting in the hot interior where Spitzer genuinely rules. The physically-motivated
enhancement is El-Badry's **temperature-independent** mixing term (Eq 21, verified §7):
`κ_mix = (λδv)·n·k_B`, applied as `κ = max(κ_mix, κ_Spitzer)` — it dominates only in the cool mixing layer.

**The sign flip — the crux.** Because `κ_mix ∝ n` and `κ_Spitzer ∝ T^(5/2)` (n-independent), the *physical*
enhancement **rises** with density: `f_κ_physical ∝ n^(+1)` at fixed interface T. The measured *fire-threshold*
**falls**: `f_κ_fire ∝ n^(−0.6)`. **Opposite signs.** So using the empirical −0.6 as a prescription gives the
*diffuse* clouds the *most* boost — which is exactly the "forcing" we want to avoid. The physical (rising)
prescription gives diffuse the *least* → dense clouds transition, diffuse stay energy-driven. That is the honest
reading, and it matches the maintainer's instinct that forcing cooling to fire "doesn't feel right."

**The experiment (pure re-analysis of `summary.csv`, no new sims).** Cap the enhancement at a physical `f_max`.
A cloud is momentum-driven iff its measured `f_κ_fire ≤ f_max`, else energy-driven. The split:

| physical f_max | momentum | energy-driven | (soft) critical column N_crit |
|---:|---:|---:|---:|
| 2 | 18/63 | 45 | ≈ 3.7×10²³ |
| 4 | 24/63 | 39 | ≈ 1.7×10²³ |
| 8 | 31/63 | 32 | ≈ 1.2×10²³ |
| 64 | 57/63 | 6 | ≈ 1.7×10²² |

**6/63 cells never fire even at f_κ=64** — energy-driven under *any* cap (all low-n / high-sfe). A physically
plausible `f_max ≈ 2–8` predicts a **critical column N_crit ≈ 1–4×10²³ cm⁻²**: clouds above it are momentum-
driven, below it blow out energy-driven. This is a **falsifiable prediction** to compare against Lancaster/PHANGS,
*not* a knob tuned to force the transition. (The boundary is *soft* — column is only a partial predictor, §9 —
so N_crit has real scatter; nCore and cloud size both enter.)

**The open tension to keep honest.** Lancaster's 3D finds catastrophic cooling "generic over >3 dex in density"
— even diffuse clouds may cool via turbulent mixing a 1D model cannot resolve. So a non-transitioning 1D cloud is
**either** genuinely energy-driven **or** 1D-under-cooled (missing the El-Badry κ_mix). The critical-column
prediction is the *dividing line*; which side is physically right is settled against observations, not asserted.
The two routes are: **(a) accept non-transition** (physical f_max, this section) — simple, honest about the 1D
limit; **(b) add κ_mix** (Rung B) — if you trust Lancaster that diffuse clouds *should* cool. Either way the
deliverable is the same: a physically-bounded prescription, not f_κ cranked to 64.

**What sets f_max / the prescription (the next input).** `f_max` and the rising exponent are physics, not free
fit parameters: they follow from El-Badry's `λδv` (1–10 pc·km/s), the saturation ceiling (Cowie & McKee,
q_sat ∝ n), and magnetic suppression (f<1 for pure Spitzer). Pinning a physically-motivated `f_κ(n)` (normalised
by λδv, capped by saturation) is the remaining derivation — see §12 for how to test it.

## 12. How to test a physically-motivated f_κ(n) prescription — the sweep design

The good news: **most of this needs no new sims.** The 819-run `summary.csv` already holds θ(f_κ) for
f_κ ∈ [1, 64] at every cell, so *any* prescription `f_κ(n) = clamp(A·n^q, 1, f_max)` is testable by
**re-analysis** (interpolate θ at the prescribed f_κ; §11 did exactly this for the flat-cap case). Scan (A, q,
f_max) offline to find the prescription whose transition map matches obs — free.

A **new sweep is only needed** to (i) probe **f_κ < 1** (magnetic suppression at the diffuse end — the existing
grid starts at f_κ=1), or (ii) **verify a chosen prescription as real production runs** rather than interpolation.
For (ii) the cleanest design, no code change required: since `cooling_boost_kappa` is a scalar and `nCore` is
fixed per run, a small **generator** (extend `runs/make_params.py`) emits one run per cloud with
`cooling_boost_kappa = clamp(A·nCore^q, fmin, f_max)` for a handful of physically-motivated (A, q, f_max). That
is **63 runs per prescription** (vs the 819 of the free f_κ scan) — cheap. The eventual production form is a
gated `cooling_boost_kappa_mode = "density"` that computes f_κ(n) in-code (default-off byte-identical), but the
generator sweep validates the prescription first.

---

## 7. Provenance / caveats (read before citing a number)

- **El-Badry now VERIFIED (2026-06-29):** the maintainer supplied the El-Badry+2019 PDF (pp. 5–6, 13, 15). Its
  §3.1/§5.2 equations are confirmed line-by-line — Eq 16 (Spitzer C=6e-7·T^(5/2)), Eq 19/20 (saturation), Eq 21
  (κ_mix=(λδv)ρk_B/μm_p), Eq 35/37/38 (θ=ψ/(11/5+ψ), ψ=A_mix√(λδv·n_H), A_mix≈1.7 analytic / **3.5 fit**). The
  earlier in-container `[unverified]` hedge (a 403 access gap, *not* an error) is **retracted**: the prior room's
  transcription was correct. Branch note retained for the record: this branch lacks the prior room's commit
  `3e68143`/`elbadry_overlay.png`, but the equations are verified independently of that commit.
- **Other literature access:** the non-El-Badry rows of §3 still come from WebSearch snippets + citing papers
  (every other arXiv/ADS host 403s in-container); their **equation numbers remain unverified** and are flagged.
  Verbatim-confirmed: Lancaster 1−Θ~0.1–0.01 and ">3 dex in density"; Narayan & Medvedev "~5× below Spitzer".
- **Citation corrections rippled into the siblings:** El-Badry = MNRAS 490, 1961 (2019), author Weisz, arXiv
  1902.09547 (not ApJ 879 / not Weinberg); Lancaster Paper I = 2104.07691, Paper II = 2104.07722.

---
*Written 2026-06-29 on `feature/PdV-trigger-term-pt2`. Builders (no sims): `data/make_fkappa_functional_form.py`
(the composed pre-sweep form) and `data/make_fkappa_sweep_analysis.py` (the §8 scorecard, reads the committed
sweep result `data/fkappa_nH_sweep.csv`).*
