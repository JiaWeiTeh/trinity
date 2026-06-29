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

**Why flat, not a rising El-Badry √n curve.** El-Badry et al. 2019 (MNRAS **490, 1961**; arXiv:1902.09547 —
note: author **Weisz**, *not* Weinberg; *not* "ApJ 879") is a **supernova-superbubble** paper for ambient
n ~ 0.1–10 cm⁻³; its θ(n) must not be extrapolated to GMC densities, and its specific ψ/θ algebra
(ψ=A_mix√(λδv·n_H), θ=ψ/(11/5+ψ), A_mix≈3.5) **could not be verified** in-container (every arXiv/ADS/journal
host returns HTTP 403) — treat it as **[unverified]**. What *is* corroborated about El-Badry by multiple citing
papers: mixing is modelled as an **effective, temperature-INDEPENDENT diffusivity κ_mix = λδv** (their one free
mixing parameter), and θ depends on n_H **and** Λ(T_pk) **and** λδv — not n_H alone. The GMC-regime magnitude
therefore comes from **Lancaster** (flat ≈0.9), with El-Badry supplying the *mechanism/parametrization*.

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

The survey (10 sub-agents; **all primary PDFs 403-blocked in-container**, so equations are from search snippets
and citing papers — equation *numbers* unverified, flagged below) is unambiguous: **no paper writes a
conduction/mixing enhancement factor as `f ∝ n_H^p`.** The density-powers that *do* exist in the literature:

| relation | density power | source | status |
|---|---:|---|---|
| classical Spitzer κ = 6e-7·T^(5/2) | **n_H⁰** (none) | Spitzer 1962; Weaver+77; matches `C_thermal` | verified (multi-source) |
| saturated heat flux q_sat = 5φ_s ρ c_s³ (φ_s≈0.3) | **n_H¹** | Cowie & McKee 1977 | form verified; eq.# not |
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
1977), so the effective conductivity ceiling scales as **κ_sat ∝ n_H¹** — it **rises** with density. The
*required* f_κ **falls** with density (∝ n_H^(−0.30)). These run in **opposite directions**, so they cross:

- **Dense clouds:** required f_κ is small (~3) and the ceiling is high → reachable by conduction boost. ✓
- **Diffuse clouds:** required f_κ is large (~tens) but the ceiling is **low** → a pure f_κ·Spitzer boost is
  **unphysical** there. Reaching the Lancaster plateau in diffuse gas needs the **temperature-INDEPENDENT
  turbulent-mixing diffusivity κ_mix** (El-Badry's actual prescription), *not* more Spitzer. ✗

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

## 7. Provenance / caveats (read before citing a number)

- **Branch note (2026-06-29):** this branch (`feature/PdV-trigger-term-pt2`, from `feature/fervent-carson-ohpjm7`
  @ `3809f8e`) does **not** contain the previous room's commit `3e68143` ("El-Badry verification + overlay") — it
  is not in this history and not findable locally. So any "El-Badry §5.2 verified from the PDF" claim in an
  external handoff does **not** apply here; in this branch El-Badry's specific algebra is **[unverified]**
  (consistent with `FINDINGS.md` §2/§2a). There is no `elbadry_overlay.png` here.
- **Literature access:** every arXiv/ADS/MNRAS/ApJ/A&A host 403s through the container proxy. All equations in
  §2–§3 are from WebSearch snippets + citing papers; **equation numbers are unverified** and flagged. Verbatim-
  confirmed items: Lancaster 1−Θ~0.1–0.01 and "more than three orders of magnitude in density"; Narayan &
  Medvedev "factor of ~5 below Spitzer"; Spitzer/Weaver 6e-7·T^(5/2); q_sat = 5φ_s ρ c_s³.
- **Citation corrections rippled into the siblings:** El-Badry = MNRAS 490, 1961 (2019), author Weisz, arXiv
  1902.09547 (not ApJ 879 / not Weinberg); Lancaster Paper I = 2104.07691, Paper II = 2104.07722.

---
*Written 2026-06-29 on `feature/PdV-trigger-term-pt2`. Builder: `data/make_fkappa_functional_form.py`
(reads `fmix_table.csv`, `kappa_blowout_calibration.csv`, `kappa_calibration_estimate.csv`; no sims).*
