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

| short cite | arXiv / ref | what it gives us |
|---|---|---|
| **Lancaster 2021b** | 2104.07691 (ApJ 914, 90), "Efficiently Cooled Stellar Wind Bubbles … I: Theory" | the **momentum-driven** theory: 3D interface mixing → efficient cooling → bubble follows momentum-driven (not energy-driven) |
| **Lancaster 2021c** | 2104.07722 (ApJ 914, 91), "… II: Validation with 3D sims" | the **θ ~ 0.9–0.99** cooling-efficiency result (the magnitude anchor this workstream uses) |
| **Lancaster 2024** | (Lancaster, Ostriker, Kim+ 2024) | the **αp ↔ ⟨vout⟩** boundary condition: matching interface energy fluxes sets the momentum enhancement |
| **Lancaster 2025** | **2505.22730v1** (May 2025), "Co-Evolution of Wind Bubbles & Photoionized Gas I" | **THIS PDF** — the semi-analytic Co-Evolution Model (CEM); *uses* θ and αp from the above; adds the wind-vs-PIR ζ framework |

**So "Lancaster 2021" was NOT wrong for the θ~0.9–0.99 claim** (that's 2021c). The 2025 PDF the maintainer
supplied is a *newer, different* paper (the CEM). Cite the θ magnitude as **Lancaster 2021c**; cite the
(1−θ)/αp framework + the wind+PIR co-evolution as **Lancaster 2025** (this PDF).

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

*Transcribed from arXiv:2505.22730v1 on 2026-06-30, `feature/PdV-trigger-term-pt2`. No production code touched.*
