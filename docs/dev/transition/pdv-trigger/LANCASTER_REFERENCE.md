# Lancaster reference (read THIS before re-reading any Lancaster PDF in this branch)

> 📌 **IMPRINT — canonical Lancaster reference for `feature/PdV-trigger-term-pt2`.** Whenever this branch needs
> a Lancaster fact (θ, αp, the mixing velocity, the momentum-driven picture, the ζ wind-vs-PIR split), **read
> this doc, do NOT re-open the PDFs** — that wastes tokens. If a fact you need is missing, add it here from the
> PDF and keep going. Sibling: `ELBADRY_REFERENCE.md` (same role for El-Badry+2019).
>
> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time distillation, not a
> maintained spec. **Re-check any load-bearing equation/number against the PDF.**
>
> 🔄 **Living reference — correct on every visit.** If you re-read and find drift, fix it here and date it.
> **Keep all banner paragraphs.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** Numbers transcribed from the PDF 2026-06-30.
>
> 🔗 **Cross-check siblings:** `ELBADRY_REFERENCE.md`, `KMIX_SELFCONSISTENT.md`, `PLAN.md`, `FINDINGS.md`.

---

## 0. ⚠️ The Lancaster paper landscape — get the year/paper right

There are **several** Lancaster papers; this workstream needs facts from **three different ones**, so don't
collapse them to one citation:

| short cite | ref | what it gives us |
|---|---|---|
| **Lancaster 2021, Paper I** | ApJ 914 (Theory; companion to below) | the **momentum-driven** theory: 3D interface mixing at a *fractal* bubble/shell interface → efficient cooling → bubble follows a momentum-driven "efficiently-cooled" (EC) solution, not energy-driven |
| **Lancaster 2021, Paper II** | **ApJ 914, 90** (2021), "Efficiently Cooled Stellar Wind Bubbles in Turbulent Clouds. II. Validation with Hydrodynamic Simulations" | **the θ-magnitude anchor — verified here in §7 (one of the supplied PDFs).** 3D sims: **Θ = 0.9–0.99**, αp~1.2–4, generic over nH≈40–2×10⁵ |
| **Lancaster 2024** | (Lancaster, Ostriker, Kim+ 2024) | the **αp ↔ ⟨vout⟩** boundary condition: matching interface energy fluxes sets the momentum enhancement |
| **Lancaster 2025** | **2505.22730v1** (May 2025), "Co-Evolution of Wind Bubbles & Photoionized Gas I" | the semi-analytic Co-Evolution Model (CEM); *uses* θ and αp from the above; adds the wind-vs-PIR ζ framework (§1–§6 below) |

**⚠️ Two corrections (2026-06-30, from reading the actual ApJ 914, 90 PDF):** (1) **ApJ 914, 90 is Paper II
(the *sims/validation* paper), not the theory paper** — I had it as "I: Theory" before; the title is literally
"II. Validation … with Hydrodynamic Simulations." The companion Paper I (Theory) is a separate article. (2) So
the θ~0.9–0.99 anchor is **Lancaster 2021 Paper II (ApJ 914, 90)** — §7 has the verified numbers. "2021" was
right; the 2025 PDF is a *different, newer* paper (the CEM, §1–§6).

## 1. The one-paragraph takeaway (what this 2025 paper gives TRINITY)

Lancaster 2025 builds a semi-analytic model for a wind-blown bubble (WBB) **co-evolving with the photoionized
region (PIR)** around a star cluster. For TRINITY's trigger work the load-bearing facts are: (a) Lancaster's
**`θ ≡ Ė_cool/Lw` is identical to El-Badry's and TRINITY's θ** (cooling fraction of the mechanical luminosity);
(b) the **(1−θ) energy-driven solution is identical to El-Badry's** (Eq 1–3 below); (c) Lancaster **separates the
cooling budget (θ) from the momentum budget (αp)** and couples them — *this is exactly the PdV-in-the-trigger
distinction*; (d) the momentum-driven limit (αp≈1, "all PdV-enhancement gone") is reached **when interface
cooling is efficient**, which is set by a **mixing velocity ⟨vout⟩** (their physical analogue of El-Badry's δv).

## 2. θ and the energy-driven solution (matches El-Badry exactly)

- **`θ ≡ Ė_cool/Lw`** (Table 1) = fraction of wind mechanical luminosity lost to cooling. **Same as El-Badry
  θ=L_int/Ė_in and TRINITY θ=L_cool/L_mech.** `Lw = ṀwVw²/2` is the wind mechanical luminosity (= L_mech).
- **Energy-driven-with-cooling solution (Eq 1–3), identical to El-Badry Eq 30–31:**
  ```
  R_ED,θ(t) = (1−θ)^{1/5} R_ED(t)        (Eq 1)
  P_ED,θ(t) = (1−θ)^{2/5} P_ED(t)        (Eq 2)
  p_ED,θ(t) = (1−θ)^{4/5} p_ED(t)        (Eq 3)
  ```
  "a constant fraction θ of the wind's energy is lost." Lancaster **cites El-Badry 2019** for the constant-θ
  result from a spherical constant-diffusivity mixing model. So El-Badry and Lancaster are the *same framework*.

## 3. αp — the momentum budget (THIS is where PdV lives)

- **`αp ≡ ṗr/ṗw`** = momentum enhancement factor: how much radial momentum the bubble carries **beyond** the
  direct wind momentum input. The excess is **PdV work** the hot interior does accelerating the shell.
  Energy-driven ⇒ αp ≫ 1 (lots of PdV enhancement); momentum-driven ⇒ αp ≈ 1 (no enhancement).
- **Momentum-driven solution:** `p_MD,α = αp·ṗw·t` (Eq 4), `R_MD,α = (αp)^{1/4} R_MD` (Eq 5).
- **θ ↔ αp are coupled** (§2.2.3): "more energy retention naturally leads to more momentum input." For the ED
  solution `αp,ED,θ ∝ (1−θ)^{4/5} t^{2/5}` (Eq 8).
- **αp is set by the mixing velocity (Lancaster 2024, Eq 10):**
  `αp = (3/4)·(Vw/4)/⟨vout⟩·(4πRw²/Aw)`, where **⟨vout⟩ = mean velocity of gas flowing into the interface
  mixing layer** (Eq 11). This is Lancaster's physical analogue of El-Badry's δv/λδv — the **mixing efficiency**
  that sets how much cools. **αp ≈ 1 (momentum-driven) is the expected outcome** when the dissipative scales are
  resolved (their §2.6), absent dynamically-important B-fields.

## 4. The momentum-driven conclusion + θ magnitude (the calibration anchor)

- Lancaster's central claim (2021b, carried here): 3D interface mixing makes cooling **so efficient that the
  bubble is effectively momentum-driven from soon after shell formation** — i.e. **θ is high (→0.9–0.99,
  Lancaster 2021c)** and the bubble retains little energy. This is **stronger cooling than El-Badry's 1D
  diffusivity** (El-Badry's bubbles stay energy-driven at θ~0.6–0.9 in his tested n≤10; the difference is the
  **3D fractal interface area**, which Lancaster's αp/⟨vout⟩ captures and El-Badry's spherical 1D does not).
- **For TRINITY's calibration:** the θ~0.9–0.99 plateau (Lancaster 2021c) is the GMC-regime magnitude anchor for
  λδv. It is **consistent with El-Badry's θ extrapolated to GMC density** (both →0.9–0.99 at n≫10), which is why
  λδv≈1 lands the GMC range in the Lancaster band (`make_elbadry_theta.py`).
- **Honest tension (route a vs b):** El-Badry's θ∝√n predicts **low** θ at low (diffuse) density → diffuse
  clouds stay energy-driven. Whether Lancaster's high-θ/momentum-driven picture extends to diffuse clouds (route
  b) or only holds at GMC density (route a) is **unresolved** and is exactly what decides "can some clouds
  genuinely not transition." This 2025 paper does not settle it (it assumes the momentum-driven limit).

## 5. The ζ wind-vs-PIR framework (this paper's new result — context, not directly TRINITY's θ)

- **`ζ ≡ Req/RSt`** = (radius where the WBB pressure-balances the PIR) / (Strömgren radius). **ζ>1 ⇒ winds
  dominate; ζ<1 ⇒ photoionized gas dominates.** Finding: for MW-like GMCs, individual massive OB stars, and
  dense low-Z early-universe regions, **0.1 ≲ ζ ≲ 1** — *both* winds and PIR matter; total FB momentum is within
  ~25% of summing them separately. Winds dominate only in the densest environments.
- **Relevance to TRINITY (which models the wind/SN bubble, not the PIR explicitly):** mostly *context* — it says
  the PIR is dynamically co-important in normal GMCs, a caveat for interpreting TRINITY's wind-only bubble. Not
  load-bearing for the θ trigger. (TRINITY does have a `P_HII` Strömgren term; full PIR co-evolution is out of
  scope here.) Strömgren radius `R_St = 10.1 pc (Q0/4e50)^{1/2}(nH/100)^{−2/3}` (Eq 12) if ever needed.

## 6. The PdV connection (why this paper matters for the branch) — see `PDV_TRIGGER_NOTE` in PLAN

Lancaster's **θ (cooling) vs αp (momentum/PdV)** split is the cleanest statement of the branch's core question:
the cooling efficiency θ is **PdV-exclusive by definition** (`Ė_cool/Lw`), while the energy→momentum transition
is fundamentally about the **momentum/PdV budget (αp→1)**. TRINITY mirrors this: `cooling_balance` (θ≥0.95) is
PdV-exclusive; `ebpeak` (`Edot_from_balance = Lmech − Lloss − 4πR2²v2·Pb ≤ 0`, the code says "PdV-inclusive") is
the PdV-inclusive criterion. For **massive clusters** (large Lw ⇒ large PdV), the PdV-inclusive criterion fires
earlier and is the more physical transition — consistent with Lancaster's momentum-driven (αp→1) picture.

## 7. Lancaster 2021 Paper II (ApJ 914, 90) — the sims paper, verified numbers (the θ anchor)

Read the actual PDF 2026-06-30. This is the **3D-hydro validation** of the efficiently-cooled (EC) theory.
The numbers TRINITY's calibration rests on:

- **Θ ≡ Ė_cool/Lw = 0.9–0.99** (the cooling fraction). They report the *retained* fraction
  **1−Θ ~ 0.1–0.01**, decreasing in time **∝ t^{−1/2}**, **for ALL models**. (Same θ as El-Badry/TRINITY.)
  Energy-retention formula (Eq 10, **corrected 2026-07-12 against the maintainer-supplied paper excerpt —
  the earlier transcription here (`(αR·ḃ/Vw)(1+fturb)/(2(1+αp))`) was WRONG**):
  `1−Θ = ( ½(1+f_turb)·(α_p/α_R) + S ) · (Ṙ_b/V_w)`, with `f_turb ≡ E_turb,sh/E_r,sh` (Eq 9) and
  `E_r,sh = (α_p/4α_R)·ṗ_w·R_b` (Eq 8). The qualitative reading survives: 1−Θ tracks Ṙ_b/V_w, and to the
  extent the parenthesized terms are constant, **1−Θ ∝ t^{−1/2}** (their stated expectation).
- **EC dynamics (momentum-driven):** `p_r = αp·ṗw·t` (Eq 3), `R_b ∝ (αp ṗw/ρ̄)^{1/4} t^{1/2}` (Eq 5) —
  shallower than energy-driven t^{3/5}. **αp ≈ Ξ within 6%** (momentum enhancement ≈ energy enhancement).
- **Measured αp ~ 1.2–4** (near the momentum-driven αp≈1, slightly enhanced). Fractal interface excess
  dimension d~0.4–0.7 (so D~2.4–2.7); turbulent velocity in the hot gas near the interface **vt ~ 200–400
  km/s** (this is Lancaster's physical mixing driver — the analogue of El-Badry's δv).
- **Density range tested: nH ≈ 40 to 2×10⁵ cm⁻³** (Mcloud = 5×10⁴, 10⁵, 5×10⁵ M☉; Rcloud = 2.5, 5, 10, 20 pc;
  uniform-ρ̄ turbulent boxes). "Generic over >3 orders of magnitude in density" and **density-insensitive** —
  but note the weak trend: 1−Θ ∝ ḃ/Vw ∝ ρ̄^{−1/4}, so **Θ rises weakly with density** (SAME sign as El-Badry).

### 7a. What this DECIDES for TRINITY (route a vs b, and the λδv value)

- **Lancaster's plateau is GMC-only (nH ≳ 40); it does NOT test diffuse ISM.** Their lowest density (~40 cm⁻³,
  the 5×10⁴ M☉ / 20 pc GMC) is still a dense cloud. So **Lancaster cannot adjudicate the diffuse end.**
- **Over the GMC range Lancaster and El-Badry AGREE** — both give Θ≈0.9–0.99 and both *rise* with n. (El-Badry
  at λδv=1: θ(40)=0.92, θ(2e5)=0.998 — right on the Lancaster band.) So the two 3D/1D anchors corroborate each
  other where they overlap.
- **Route-a is therefore the best-supported answer at the diffuse end:** below Lancaster's tested range
  (nH ≲ 40), only El-Badry's √n speaks, and it gives θ < 0.9 → **diffuse clouds genuinely stay energy-driven**
  (the user's "some clouds can't transition" — yes, the diffuse ones, *uncontradicted* by Lancaster). This is
  the falsifiable critical-density prediction.
- **λδv calibration (refined):** to make TRINITY's trigger fire (θ≥0.95) across Lancaster's *whole* momentum-
  driven GMC range (down to nH~40), need θ(λδv, 40) ≥ 0.95 ⟹ **λδv ≈ 3.5 pc·km/s** (n_fire ≈ 48). This is also
  **El-Badry's own calibration value (A_mix=3.5 was fit at λδv=3)** — doubly anchored. So adopt **λδv ≈ 3**
  (n_fire ≈ 50): GMC clouds (nH ≳ 50) transition; diffuse clouds (nH ≲ 50) stay energy-driven = fate. *(At
  λδv=1, n_fire=143 would wrongly exclude Lancaster's nH~40–140 momentum-driven clouds — so λδv=1 is a bit low;
  ~3 matches Lancaster better.)* **Caveat (2026-07-02): as a TRINITY fate prediction, the clean nH≷50 split is
  contradicted by `FINDINGS.md §10` point 4** — under the measured `multiplier` calibration, `small_1e6`
  (n=100) never fires through f=8 while `large_diffuse_lowsfe` (same n=100) fires at f=4; TRINITY's emergent
  transition boundary is **θ₀-based**, not a clean n-threshold. (The λδv/El-Badry closed-form threshold above
  stands as the literature-model statement.)

### 7b. Table 1 VERIFIED (2026-07-12, maintainer-supplied paper excerpts) — the Phase-5 bench5 anchor

> Provenance: the maintainer pasted image excerpts of the L21b paper (ApJ 914:90) into the working chat
> on 2026-07-12 — Table 1 (Parameters of Simulation Suite), the Eq 8–11 block, and Figure 17 with its
> caption. This section is the durable transcription (the imprint protocol): future sessions reference
> THIS, not the chat. Grade **[V]** for everything below except where marked.

**Table 1 — Parameters of Simulation Suite** (12 models = 3 masses × 4 radii; Δx and resolution omitted
— grid params, not physics inputs):

| M_cl (M⊙) | R_cl (pc) | n̄_H (cm⁻³) | v_t (km/s) |
|---:|---:|---:|---:|
| 5×10⁴ | 20 | 43.1 | 3.59 |
| 5×10⁴ | 10 | 345 | 5.08 |
| 5×10⁴ | 5 | 2760 | 7.18 |
| 5×10⁴ | 2.5 | 22,800 | 10.2 |
| 10⁵ | 20 | 86.3 | 5.08 |
| 10⁵ | 10 | 690 | 7.18 |
| 10⁵ | 5 | 5520 | 10.2 |
| 10⁵ | 2.5 | 44,200 | 14.4 |
| 5×10⁵ | 20 | 431 | 11.4 |
| 5×10⁵ | 10 | 3450 | 16.1 |
| 5×10⁵ | 5 | 27,600 | 22.7 |
| 5×10⁵ | 2.5 | 228,000 | 32.1 |

**Table Notes (verbatim meaning):** each model is run with **three values of the wind-source-particle
mass, M_*/M_cloud ≡ ε_* = 0.01, 0.1, and 1**. ⚠️ **This FALSIFIES the search-snippet assumption
"M_* = 5000 M⊙ fixed"** that the pre-verification SOURCE_TERM_DESIGN §3 Phase-5 mapping rested on:
ε_* is a fixed *ratio*, not a fixed mass (M_*=5000 arises only at 5e4/ε0.1 and 5e5/ε0.01 — presumably
where the snippet came from). Consequence: the spec's `sfe=0.05` mapping for the three 10⁵ M⊙ benches
matched **no published model**; corrected to ε_*=0.1 (see §3 Phase 5 in SOURCE_TERM_DESIGN).

**Consistency checks run 2026-07-12 (all pass; builder command in FINDINGS §15g):**
- n̄_H internal: n ∝ M/R³ holds across all 12 rows (e.g. 43.1×2=86.3, ×8=345, ×10=431).
- With **μ_H = 1.4** (mean mass per H nucleus), (M_cl, n̄_H) reproduces R_cl exactly in 10 of 12 rows
  ⇒ n̄_H is hydrogen-nucleus density, ρ = 1.4·m_H·n_H — **identical to TRINITY's `nCore` convention**
  (`registry.py` nCore: "Hydrogen nuclei number density… rho = nCore * mu_convert * m_H", μ_convert=1.4).
  **Exception (caught by the bench5 emit gate):** the 5×10⁴ and 5×10⁵ rows at R_cl=2.5 pc both imply
  R = 2.473 pc — their n̄_H carries the same ~3.3% internal offset ((2.5/2.473)³ = 1.033), presumably
  Table-1 rounding. The bench5 mapping pins the **published n̄_H** (the physically-important input;
  spec: nCore = n̄ exactly) and accepts the 1.1% radius slack.
- **v_t is the α_vir = 2 virial velocity**: α_vir = 5v_t²R/(3GM) = 1.996–2.016 for all 12 rows.
- t_ff(n̄_H): 6.63 / 1.66 / 0.59 / 0.21 / 0.09 Myr for the five bench configs — matches the spec's ≈values.

**Θ definition (maintainer gloss: "θ = L_cool/L_gain"):** Θ ≡ Ė_cool/ℒ_w — an *instantaneous rate
ratio* (Fig 17 caption + Eq 10 text), measured two ways (cooling in wind-polluted f_wind>10⁻⁴ gas;
energy-conservation residual). This aligns directly with TRINITY's instantaneous θ = L_loss/L_mech —
closer than the "cumulative Θ" reading the Phase-5 spec hedged on. (TRINITY's numerator includes
L_leak; L21b's is radiative — the Rogers & Pittard channel-split caveat stands.)

**Equations (verified from the excerpt):** E_r,sh = (α_p/4α_R)·ṗ_w·R_b (8); f_turb ≡ E_turb,sh/E_r,sh
(9); **1−Θ = (½(1+f_turb)·α_p/α_R + S)·(Ṙ_b/V_w)** (10), expecting 1−Θ ∝ t^{−1/2}; fractal area
A_b(R_b;ℓ) = 4πα_A·R_b²(R_b/ℓ)^d (11).

**Figure 17 (M_cl=10⁵ M⊙; rows R_cl=20/10 pc; columns ε_*=1/10/100%):** 1−Θ declines from ~0.1–0.15
at t≈0.01 Myr to ~0.01–0.04 at late t, tracking the theory line, until wind breakout of the box
(vertical lines), after which 1−Θ *rises*. Approximate anchor values — **⚠️ [V-plot-eyeball] grade,
read off a low-resolution image, ±0.2–0.3 dex; re-digitize before any quantitative fit**:

| panel (R_cl, ε_*) | 1−Θ @0.01 Myr | @0.1 Myr | @late (pre-breakout) | breakout ≈ |
|---|---|---|---|---|
| 20 pc, 1% | ~0.09 | ~0.06 | ~0.015 @ 1 Myr | ~1.1 Myr |
| 20 pc, 10% | ~0.15 | ~0.05 | ~0.025 @ 0.5 Myr | ~0.5 Myr |
| 20 pc, 100% | ~0.17 | ~0.05 | ~0.04 @ 0.2 Myr | ~0.2 Myr |
| 10 pc, 1% | ~0.06 | ~0.03 (dotted; solid dips <10⁻³ transiently ~0.08 Myr) | ~0.02 @ 0.7 Myr | ~0.85 Myr |
| 10 pc, 10% | ~0.10 | ~0.03 | ~0.015 @ 0.35 Myr | ~0.4 Myr |
| 10 pc, 100% | ~0.12 | ~0.03 | ~0.025 @ 0.12 Myr | ~0.15 Myr |

The bottom-middle panel (10 pc, ε_*=10%) is **bench-2's direct published track**. The top row
(20 pc, ε_*=10%) is NOT in the bench set (bench-1 is 5e4/20pc) — flagged as a cheap candidate bench-6
(direct track available) if the maintainer wants a sixth benchmark.

**Still UNVERIFIED ([I]-grade, search-snippet only): V_w** (3230/1759 km/s claim) — not in the supplied
excerpts (Table 1's v_t is the cloud *turbulent* velocity, NOT the wind velocity). Not needed to freeze
the bench5 .params (TRINITY supplies its own SB99 wind and the metrics normalize by TRINITY's L_mech);
needed only to overlay Eq-10 *theory* curves. Ask the maintainer for §2's wind parameters if that
overlay is wanted.

**TRINITY mapping (exact, decided 2026-07-12 — supersedes the naive mCloud=M_cl, sfe=ε_* draft):**
TRINITY's `.param` mCloud is **pre-SFE** (`read_param.py` rebinds: mCluster = sfe·mCloud_input, residual
gas = (1−sfe)·mCloud_input, and `get_InitCloudProp` derives rCloud from the *post-SF gas*). L21b instead
*adds* a star particle ε_*·M_cl to an M_cl gas cloud. Exact match: **mCloud_param = M_cl·(1+ε_*),
sfe = ε_*/(1+ε_*)** ⇒ post-SF gas = M_cl at nCore = n̄_H (so rCloud = R_cl exactly) and
mCluster = ε_*·M_cl exactly. (The naive mapping is only ~10%/3% off in gas mass/radius at ε_*=0.1, but
the exact one is free.)

**In-container run status (`FINDINGS.md §15h`, 2026-07-12) — COMPLETE:** the 60-arm bench5 matrix ran
in-container (HPC down) → **60/60, 59 compliant** (1 dense diag wall-killed, non-critical). **Fire map:**
bench5(n̄=2.28e5) fires UNMODIFIED (f_A≥1), bench4(n̄=4.42e4) at f_A≥4, bench3(n̄=5520) at f_A≥12, bench2/bench1
NOFIRE ≤16 — matches the registered θ_EB above (θ_EB falls with density). **Θ_cum/L21b-band calibration
(from the diagnostic arms, all complete):** the diffuse benches blow out cleanly (end R2≈rCloud), giving the
L21b breakout-window Θ_cum — bench3 enters the band [0.90,0.99] at **f_A≈16** (Θ_cum 0.965), bench2/bench1
do NOT reach it even at f_A=16 (max 0.54/0.40) → **f_A >16 / ≫16**. The dense benches censor at
shell-collapse (not the clean L21b window). **Result: no single global f_A reproduces L21b across density;
the required boost climbs steeply toward low density** (feeds Phase-6 ship decision). **HPC re-confirmation DONE 2026-07-19** (`FINDINGS §15j`): fidelity OK, and bench6 extends the
calibration — all clean benches reach the band (f_A entry 13.9/53.5/74.8); f_mix eliminated. The
Fig-17 tracks remain [V-plot-eyeball] grade — re-digitize before any quantitative fit.

*Transcribed from ApJ 914, 90 (Lancaster+2021 Paper II) and arXiv:2505.22730v1 (Lancaster+2025) on 2026-06-30,
`feature/PdV-trigger-term-pt2`. No production code touched.*
