# El-Badry+2019 — full-paper reference (read THIS before re-reading the PDF)

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time reference, not a
> maintained spec. **Re-check each equation/number against the PDF if it is load-bearing.** The PDF is
> `arXiv:1902.09547v2` (MNRAS 490, 1961, 2019), 32 pages; this doc distills every equation and number this
> workstream needs so a future session does **not** re-read 32 pages (~the maintainer's token-saving ask).
>
> 🔄 **Living reference — correct on every visit.** If you re-derive or re-read and find a discrepancy, fix it
> here and date it. **Keep all banner paragraphs.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** Numbers here are transcribed from the PDF (2026-06-30);
> the offline θ calculator is `data/make_elbadry_theta.py` (no sims).
>
> 🔗 **Cross-check siblings:** `KMIX_SELFCONSISTENT.md` (the self-consistent solve this reframes),
> `KMIX_IMPLEMENTATION_SPEC.md`, `KMIX_PROTOTYPE.md`, `PLAN.md`, `FINDINGS.md`. Reconcile any number that disagrees.

---

## 0. The one-paragraph takeaway (why this paper matters to TRINITY)

El-Badry models SBs with conduction + cooling + a **turbulent-mixing diffusivity κ_mix**. The master result:
**all SB dynamics are set by a single dimensionless cooling efficiency `θ ≡ L_int/Ė_in`** (fraction of the SN
mechanical luminosity radiated in the shell/bubble interface), and **θ depends only on ambient density and the
mixing efficiency λδv** via a **calibrated closed form** (Eq 37/38). Crucially **`θ_ElBadry = θ_TRINITY`** — TRINITY's
`cooling_balance` trigger fires on θ = L_cool/L_mech, and `Ė_in = ESN/Δt_SNe = L_mech`, so they are the *same
number*. That makes El-Badry's θ(λδv, n) a **ready-made, 3D-informed prescription for exactly the quantity
TRINITY needs** — usable via TRINITY's existing `cooling_boost_mode='theta_target'` without porting κ_mix into
the Weaver structure ODE at all.

## 1. Classical Weaver recap (§2; the base TRINITY already implements)

- Momentum: `d/dt[(4π/3)R³ρ₀ dR/dt] = 4πR²P` (Eq 1). Energy: `d/dt[(4π/3)R³ P/(γ−1)] = Ė_in − 4πR² (dR/dt) P` (Eq 2).
- Solution: `R(t) = (125/154π)^{1/5} Ė_in^{1/5} ρ₀^{−1/5} t^{3/5}` (Eq 3); `P(t) ∝ Ė_in^{2/5} ρ₀^{3/5} t^{−4/5}` (Eq 4).
- Energy split: `E_kin = (15/77)Ė_in t`, `E_th = (5/11)Ė_in t` (Eq 6); retained `E_SB = (50/77)Ė_in t` ≈ 65%, lost 35% at the leading shock (Eq 7-8). `E_kin/E_SB = 3/10` (Eq 9).
- **Evaporation (no cooling), Cowie-McKee/Weaver:** mass flux `ṁ = −8πμm_p r²κ(dT/dr)/(5k_B T)` (Eq 11);
  `ṁ_nc = 16π μm_p R C T_int^{5/2}/(25 k_B)` (Eq 12). Interior temp `T_hot,nc = (205/84 · R²P/(Ct))^{2/7}` (Eq 13).
  **Conduction has NO effect on Weaver dynamics — only on interior density/temperature.** (This is why TRINITY's
  Spitzer conduction sits inside the *structure* solve, not the dynamics.)

## 2. Conduction (§3.1) — the κ TRINITY uses, and the two pieces TRINITY omits

- **Spitzer (Eq 16):** `κ_S = 1.70×10¹¹ T₇^{5/2} / (1 + 0.029 ln(T₇ n_{e,−2}^{−1/2}))`, T₇=T/10⁷K. Density enters
  only in the log ⇒ weak. **Approx `κ_S ≈ C·(T/K)^{5/2}`, C = 6×10⁻⁷ cgs** — *this is exactly TRINITY's
  `C_thermal`*. (El-Badry's sims use the full Eq 16; the approx is what we both quote.)
- **Parker (Eq 17, neutral gas):** `κ_P = 2.5×10⁵ T₄^{1/2}`, T₄=T/10⁴K. **Spitzer=Parker at T=6.6×10⁴ K**;
  below that, conduction is Parker. **But El-Badry: "the effects of including Parker are very small, since κ_P in
  the shell is orders of magnitude lower than κ_S."** ⇒ **Parker is negligible; our omitting it does NOT matter.**
  *(This corrects my prior-turn claim that Parker was a load-bearing missing piece — it is not. κ_P is tiny.)*
- **Saturation (Eq 18-20):** classical flux `q=−κ∇T` (Eq 18) overestimates when ∇T is steep; cap at
  `q_max ≈ (3/2)ρc_s,iso³`. Smooth interpolation `1/|q| = 1/|κ∇T| + 1/((3/2)ρc_s³)` (Eq 19), i.e.
  `κ_eff⁻¹ = κ⁻¹ + |∇T|/((3/2)ρc_s³)` (Eq 20). **Effect (App D): early-time flux >10× smaller with saturation;
  late-time modest (<2×). Reduces Mhot by ~15-20%, NEGLIGIBLE effect on θ (the cooling efficiency) and on E_SB.**
  ⇒ **saturation matters for early-time numerics + Mhot, NOT for θ.** *(This tempers my prior-turn claim that
  saturation was the key to θ — it mainly fixes Mhot/early-time stiffness, not the cooling efficiency.)*
- **κ ceiling (App C):** El-Badry caps κ at `κ_max = 1.8×10¹² (n_H/cm⁻³)` (Eq C3) purely for CFL tractability
  (high κ → tiny timestep, Eq C1-C2). Changes integrated properties <1%. **Lesson for us: even El-Badry had to
  bound κ; an unbounded κ_mix (our 10⁵–10⁸× blowup) is a known numerical hazard, not a physical signal.**

## 3. Mixing (§3.2) — the heart of it, and the answer to "did they use both?"

- **`κ_mix = (λδv)·ρk_B/(μm_p)` (Eq 21).** Temperature-INDEPENDENT (in ρ; note ρ=μm_p·n so κ_mix ∝ n; at fixed
  pressure n∝1/T ⇒ **κ_mix ∝ 1/T locally** — this is the kprime=−1/T point from `KMIX_SELFCONSISTENT.md` §2b).
- **YES, both at once (verbatim, p.6):** *"The actual conductivity, κ, in each simulation zone is set to the
  larger of κ_mix and thermal conductivity (κ_S or κ_P, with appropriate modifications for saturation)."* ⇒
  **`κ = max(κ_mix, κ_S/κ_P_saturated)` per zone.** Our `max(κ_mix, κ_Spitzer)` is the right *form*. f_κ (uniform
  scalar on Spitzer) is NOT El-Badry's method.
- **λδv estimate (Eq 22-23):** `λδv ∼ Δt_SNe v_rel²/(ρ_high/ρ_low) = 0.1 pc·km/s (v_rel/10)²(ρ_h/ρ_l/100)^{−1}(Δt_SNe/0.1Myr)`.
  Plausible range **0.1 ≲ λδv/(pc·km/s) ≲ 10**. **"The correct value of λδv … is presently quite uncertain"**;
  **path forward = calibrate λδv to 3D-sim cooling rates (future work they did NOT do).** λδv<1 is numerically
  unconverged in their grid (App A/B); they mostly use **λδv ∈ [1,10]**, fiducial **λδv=1**.
- **Crossover:** κ_mix dominates κ_S where **T < 2×10⁵ K and n_H > 0.2 cm⁻³** (matches our prototype).

## 4. The cooling efficiency θ (§5.2) — THE deliverable for TRINITY

- **Definition:** `θ ≡ L_int/Ė_in` = fraction of SN energy input radiated in the interface. Dynamics: substitute
  **`Ė_in → (1−θ)Ė_in`** everywhere in the Weaver solution (Eq 30-31): `R ∝ (1−θ)^{1/5}`, `p̂_rad ∝ (1−θ)^{4/5}`.
- **Derivation (thin isobaric mixing layer, cooling at a single T_pk):** Eq 32-36 give
  `L_int/Ė_th = (11/5)·θ/(1−θ)` (Eq 35) `≈ (2√7/5)(λδv)^{1/2}ρ₀^{1/2}[αΛ(T_pk)/(k_B²T_pk²)]^{1/2}` (Eq 36).
- **THE CLOSED FORM (Eq 37-38):**
  ```
  L_int/Ė_th  =  A_mix · (λδv / 1 pc·km/s)^{1/2} · (n_H,0 / 1 cm⁻³)^{1/2}        (Eq 37)
  θ           =  (L_int/Ė_th) / (11/5 + L_int/Ė_th)                              (Eq 38)
  A_mix = 1.7  (analytic, α=1, T_pk=2×10⁴ K)  ;  A_mix = 3.5  (fit to sims)
  ```
  So **`θ(λδv, n) = A_mix√(λδv·n) / (2.2 + A_mix√(λδv·n))`**, rising with both. **n_H,0 is the AMBIENT (pre-shock)
  hydrogen density.** T_pk≈2×10⁴ K (App E: volumetric cooling P²Λ/T² peaks at 2e4 even though Λ peaks at 1e5;
  **T_pk ~const across n_H,0 and λδv** — validates the ansatz).
- **Measured θ (Fig 7, t=10 Myr):** n_H,0=1, λδv=1 ⇒ **θ≈0.61**. Range over n_H,0∈[0.1,10], λδv∈[1,10]: θ≈0.4→0.9.
- **Breakdown:** the analytic model fails for **θ ≲ 0.05 (n_H,0 ≲ 0.001)** — cooling term no longer dominates Eq 42.
  **No stated upper-density limit; tested only to n_H,0=10.** Extrapolation above is unvalidated (see §6 caveats).
- **θ is time-INDEPENDENT** (L_int ∝ t): the layer-width time-dependence cancels n²ΛR². So a single equilibrium θ.

## 5. Interior structure & mass flux (§5.3) — secondary for the trigger, needed for Mhot/T

- Temp scale `T_s = (R²P/Ct)^{2/7} ∝ (1−θ)^{8/35}…` (Eq 40); profile `τ(ξ)=(77/20 · θ/(1−θ)(1−ξ))^{2/7}` (Eq 44,
  vs no-cooling (1−ξ)^{2/5}). Interior `T_sb = T₀ θ^{2/7}(1−θ)^{−2/35}…`, T₀≈7.7×10⁶ K (Eq 45).
- Mass flux `ṁ = ṁ₀ (1−θ)^{37/35}/θ^{2/7}…`, ṁ₀≈2900 M⊙/Myr (Eq 47); `M̂hot ≈ 250 M⊙ (1−θ)^{37/35}/θ^{2/7}…` (Eq 48).
- **Evaporation is SUPPRESSED by cooling: ṁ ~ 3-30× below Weaver no-cooling.** (The El-Badry headline; relevant to
  TRINITY's `dMdt`, not to the trigger θ.) Cooling adds `−L_int` to the energy balance (Eq 29).

## 6. Late-time only, and the honest caveats (§6, §7)

- **θ is a LATE-TIME (t ≳ 3 Myr) equilibrium.** At early times discrete-SN shocks dominate; fluxes >10× equilibrium;
  blast waves stay supersonic to the shell until t≈3 Myr (Fig 12); continuous-injection model invalid. **⇒ validates
  the maintainer's ≥5 Myr rule directly, and means our early-phase θ-peak (t~1 Myr, `KMIX_SELFCONSISTENT.md` §2b)
  was in the regime El-Badry's model does not cover.**
- **Mixing, not conduction, sets the cooling (result vi).** κ_S alone → interface thickness → 0 at high res → cooling → 0
  (App A); κ_mix (T-independent) sets a finite layer ⇒ converged finite cooling. **κ_mix is the essential piece; Spitzer
  conduction mainly sets interior T/evaporation.**
- **θ is the master parameter (result vi):** R, p, E, Mhot, T_sb all follow from θ via (1−θ). Get θ right ⇒ get the rest.
- **GMC-density extrapolation:** tested only n_H,0 ≤ 10. TRINITY's GMC/shell densities are far higher. The √n form
  extrapolated gives θ→0.9-0.999 at n≫10. **El-Badry HIMSELF (p.23): "at the densities of molecular clouds, the high
  values of θ would be consistent with an order of magnitude or more reduction in hot-gas energy."** And **Lancaster
  (3D, GMC) finds θ~0.9-0.99** — two independent supports that dense-cloud θ is high. **⇒ our self-consistent solve's
  "dense θ~0.35" is the OUTLIER and very likely a port artifact (kprime, hard-max, wrong epoch), not physics.**

## 7. The TRINITY mapping (how to actually use this)

| El-Badry | symbol | TRINITY |
|---|---|---|
| cooling efficiency | `θ = L_int/Ė_in` | **= θ trigger = L_cool/L_mech** (cooling_balance, fires ≥0.95) |
| energy input rate | `Ė_in = ESN/Δt_SNe` | `L_mech` (`Lmech_total`) |
| Spitzer coeff | `C = 6×10⁻⁷` | `C_thermal` (registry, identical) |
| ambient density | `n_H,0` (uniform) | **local pre-shock cloud density at the shell, `n_amb(R2)`** — NOT necessarily nCore; for a GMC profile it declines as R2 grows ⇒ θ(t) declines (early-high, matches the trajectory shape) |
| mixing knob | `λδv ∈ [0.1,10]` | the one free parameter — calibrate to Lancaster θ~0.9-0.99 |
| closed-form θ | Eq 37/38, A_mix=3.5 | feed as `cooling_boost_theta` target in `cooling_boost_mode='theta_target'` |

**Open mapping question (honest):** what exactly is `n_amb` for a stratified GMC — the local profile density at R2,
the swept-up shell density, or the interface density? El-Badry's ρ₀ is the *uniform ambient* the bubble expands into,
so the natural map is the **local cloud density at the shell radius**; this must be pinned before production.

*Transcribed from the PDF 2026-06-30 on `feature/PdV-trigger-term-pt2`. No production code touched.*
