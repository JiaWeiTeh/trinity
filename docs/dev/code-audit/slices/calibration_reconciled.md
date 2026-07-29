# Reconciliation — three-lens blind audit of `get_y0` (`trinity/phase0_init/get_InitPhaseParam.py`)

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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**Method note.** I read only `lensA.md`, `lensB.md`, `lensC.md`. I did not open the source file, any
stripped copy, or any other file in the repository, and I ran no code. Every quoted string below is
verbatim from the lens report it is attributed to. Notation was normalised before declaring a
mismatch (`0.4` ≡ `2/5`; `2·L/pdot` ≡ `2*L/pdot`; `2π` ≡ `2*pi`); the mismatches recorded here
survive that normalisation.

**Headline.** Seven S1 findings. Five are cases where the code alone is wrong and its own comment
already states the right formula (`v0`, and three of the four `T0` factors, plus a dropped term).
Two are the dangerous class where **code and comment agree with each other and both disagree with
the derived physics** (`E0`'s `5/7`, and the `2π` in the free-streaming duration) — invisible to any
reviewer who reads code and comments together.

---

### R-001 · `v0` is the reciprocal of the wind terminal velocity — the ratio `L/pdot` is inverted
- **quantity** — `v0`, the terminal wind velocity used for `dt_phase0`, `r0`, and returned as the initial shell velocity
- **pattern** — A≠B **and** A≠C (B = C)
- **Lens A says** — `134   v0 = 2.0 * pdot_W / Lmech_W`; "`v0 = 2 · pdot_W / Lmech_W`". And: "**I1** | `:130` vs `:134` | **Algebraically incompatible.** With `Mdot0 = pdot²/(2L)`, self-consistency requires `v0 = 2L/pdot`; the code writes `v0 = 2·pdot/L`, the reciprocal arrangement. Check: `Mdot0·v0 = pdot³/L² ≠ pdot`, and `½·Mdot0·v0² = pdot⁴/L³ ≠ L`. Only `v0 = 2L/pdot` satisfies both." Dimensionally: "**D1** | `:134` | `v0 = 2·pdot_W/Lmech_W` has dimension (Msun·pc·Myr⁻²)/(Msun·pc²·Myr⁻³) = **Myr/pc**, an inverse velocity, not pc/Myr."
- **Lens B says** — B-034: "'From: v = 2 * L / pdot (wind-only quantities)'. Coefficient `2`; numerator `L`, denominator `pdot`; both wind-only." B-033: "'Terminal velocity from winds [pc/Myr in AU units]'".
- **Lens C says** — C-001: "**⇒ v = 2 * L_w / pdot_w**" … "**Direction (the classic inversion error).** `v` is **proportional to L_w** and **inversely proportional to pdot_w**. … The wrong forms to look for in the implementation are `v = pdot_w/(2 L_w)`, `v = L_w/pdot_w` (missing the 2), and **`v = 2*pdot_w/L_w`**." **CONFIDENCE: HIGH.** Ranked #3 in C's "most likely to be wrong" list.
- **verdict** — **The code is wrong.** The comment (B) and the independent derivation (C) agree exactly, and C names this precise wrong form in advance as the classic inversion. C's confidence is HIGH and the result is elementary algebra from `L = ½Mdot v²`, `pdot = Mdot v`. Confidence: very high. Note this is the root of A's dimensional cascade D1→D5: `dt_phase0` becomes pc³/Myr², `t0` becomes a sum of two different dimensions, `r0` becomes pc²/Myr, `E0` becomes Msun·pc⁵/Myr⁵ — every returned value except `T0`'s prefactor is contaminated. The log label at `:180` ("`v0={v0:.6e} pc/Myr`") is therefore also false.
- **severity** — **S1** (results-wrong; all five returned values affected)
- **how to settle it** — Dimensional: `[Msun·pc·Myr⁻²]/[Msun·pc²·Myr⁻³] = Myr/pc`, an inverse velocity; only `2L/pdot` gives pc/Myr. Numerical: A's own worked case gives `v0 ≈ 1.9e-3` where the consistent arrangement gives `≈ 2.1e3 pc/Myr (~2000 km/s)`, which is exactly C's stated smell-test band ("`v` should land in the **1000–3000 km/s** band"). Assert `100 < v0*0.9778 < 1e4` km/s (C's E11).

---

### R-002 · `T0` time exponent has the wrong **sign**: code `+6/35`, doc and physics `−6/35`
- **quantity** — the `dt_phase0` factor in `T0`
- **pattern** — A≠B **and** A≠C (B = C)
- **Lens A says** — "`(dt_phase0)**(6.0/35.0)`" … "- Exponent on the duration: **+6/35** ≈ +0.1714285714 — **positive**, not negative."
- **Lens B says** — B-008: "`T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4`" … "`-6/35` on time"; **"Falsified if:** … a sign differs (e.g. `t^(+6/35)`)". B-044 repeats `t^(-6/35)`.
- **Lens C says** — C-011: "`T(x) = C_T * L36^{8/35} * n0^{2/35} * t6^{-6/35} * (1 - x)^{2/5}`" … "i.e. **p = 8/35, q = 2/35, r = -6/35, s = 2/5** (note the **negative** exponent on time…)". Derived by two independent legs plus three fetched secondary sources. **CONFIDENCE: HIGH** for the four exponents. Ranked #9.
- **verdict** — **The code is wrong.** The doc's own falsification clause literally names `t^(+6/35)` as the failure mode, and the code has it. C derives `α = -6/35` from the evaporation/conduction balance and cross-checks it against three independent quotations. Confidence: very high.
- **severity** — **S1**
- **how to settle it** — Physical argument: an adiabatically expanding, conductively evaporating bubble must **cool** as it ages at fixed `L`; a positive exponent makes it heat up forever. Numerically the sign flip costs a factor `dt^(12/35)`: for C's worked `dt = 8e-6 Myr` that is `≈ 0.018`, i.e. `T0` low by ~56×. Cross-check against C-015's independently confirmed companion relation `n ∝ t6^{-22/35}` — the pair `(T ∝ t^{-6/35}, n ∝ t^{-22/35})` must satisfy isobaricity `nT ∝ P ∝ t^{-4/5}`: `-6/35 - 22/35 = -28/35 = -4/5` ✓. With `+6/35` the identity fails.

---

### R-003 · `T0` luminosity exponent is `8/25`; doc and physics both say `8/35`
- **quantity** — the exponent on `(Lmech_W / WEAVER_L_REF)` in `T0`
- **pattern** — A≠B **and** A≠C (B = C)
- **Lens A says** — "`(Lmech_W / WEAVER_L_REF)**(8.0/25.0)`" … "- Exponent on the luminosity ratio: **+8/25** = +0.32." And "**I5** | `:173–175` | Exponent denominators are inconsistent within one formula: `8.0/25.0` alongside `2.0/35.0` and `6.0/35.0`." Also: "8/25 = 0.32 whereas 8/35 ≈ 0.2286."
- **Lens B says** — B-008: "exponents `8/35` on the scaled luminosity"; B-044 repeats "`(L/10^36)^(8/35)`". "**Falsified if:** any of the coefficient `1.51e6` or the exponents `8/35`, `2/35`, `-6/35`, `0.4` differ in the implemented expression".
- **Lens C says** — C-011: "**p = 8/35**", derived twice and confirmed by three fetched sources (`T_w = 1.6e6 n0^{2/35} (Ṁ_6 v_2000²)^{8/35} t6^{-6/35} K`; `T_b = 2.07e6 L36^{8/35} n0^{2/35} t6^{-6/35} K`). **CONFIDENCE: HIGH.**
- **verdict** — **The code is wrong** — a `25`-for-`35` digit slip in the denominator. Two lenses independently give `8/35`; the surviving `/35` denominators on the sibling factors (A's I5) are corroborating internal evidence. Confidence: very high.
- **severity** — **S1** (in isolation only `L36^{0.091}` — modest — but it compounds with R-004, where the exponent multiplies a `1e30`-sized unit error)
- **how to settle it** — The exponent family is fixed by the same similarity solution that gives `−6/35` and `2/35`; all three share the denominator 35 (they come from `T_c^{7/2} ∝ L^{4/5}ρ^{1/5}t^{-3/5}` ⇒ divide by 7/2 ⇒ `×2/7`, and `4/5 × 2/7 = 8/35`). Check `8/35 = (4/5)(2/7)` directly.

---

### R-004 · `T0` divides an **AU** luminosity by a bare `1e36` — the AU→erg/s conversion is missing
- **quantity** — the `L36` ratio fed to `T0`
- **pattern** — A≠B **and** A≠C (B = C)
- **Lens A says** — "**D6** | `:173` | `Lmech_W / 1e36` divides an **AU** luminosity by a bare `1e36`. If `1e36` is erg/s, the conversion `× cvt.L_au2cgs (≈6.0241e29)` is **absent**; `^(8/25)` makes the shortfall a factor ≈ **3.39e9** in `T0`." And: "**D8** … The `T0` product mixes **cgs** (density, converted), **AU** (luminosity, unconverted), **AU** (time, unconverted) inside one empirically-calibrated formula. Exactly one factor of three is converted; they cannot all match one calibration."
- **Lens B says** — B-009: "In the temperature formula, `L` is in erg/s and is normalised by `10^36 erg/s`; `n` is in cm^-3 and is normalised by `1 cm^-3`. **Falsified if:** the code feeds `L` and `n` into this expression in internal/AU units (Msun*pc^2/Myr^3 and Msun/pc^3 or pc^-3) without converting to erg/s and cm^-3 first." B-011: reference luminosity "with units **erg/s**".
- **Lens C says** — C-012: "**The units are the whole point.** … `L36` | wind mechanical luminosity **in units of 1e36 erg/s** — i.e. `L36 = L_w[erg/s] / 1e36`". "**The trap, quantified.** In astro units `1 M_sun pc²/Myr³ = 6.02606e29 erg/s`, so **`L36 = L_astro × 6.02606e-7`**." **CONFIDENCE: HIGH** for "C-012 required units (L36, n0 cm⁻³, t6 = Myr)". Ranked **#1** in C's list of things most likely to be wrong.
- **verdict** — **The code is wrong.** B's falsification clause describes exactly what A found; C independently states the required unit at HIGH confidence. Note A and C quantify *different* variants of the mistake (A: divides AU by `1e36`, shortfall `6.024e29` in the ratio; C: hypothesised feeding `L_astro` with no division at all, `26.4×` inflation). Normalised, both say the same missing operation: `× cvt.L_au2cgs ≈ 6.024e29`. As written the code's ratio is too small by `6.024e29`, so `T0` is too **cold** by `(6.024e29)^{0.32} ≈ 3.4e9` (or `≈6.4e6` once R-003 is fixed to `8/35`). Confidence: very high.
- **severity** — **S1**
- **how to settle it** — One numerical evaluation. Take C's worked case `L_w = 1e38 erg/s` (`L_astro ≈ 1.66e8`), `n = 1e3 cm⁻³`, `dt = 8e-6 Myr`: the code's expression returns `T0 ≈ 4e-4 K`, against a Weaver expectation of `~1e7 K`. Any `T0` not in `1e6–1e8 K` for a GMC-scale cluster confirms it. This is the cheapest single check in the whole audit, and it also exposes R-002 and R-003 in the same number.

---

### R-005 · The `(1 − xi)^{2/5}` radial factor is entirely absent from the code; `bubble_xi_Tb` is validated then never used
- **quantity** — the `(1 − bubble_xi_Tb)^{0.4}` factor of `T0`
- **pattern** — A≠B **and** A≠C (B = C; `0.4` and `2/5` are the same number and are **not** the disagreement — the disagreement is presence vs absence)
- **Lens A says** — "All three factors are multiplied; there is no additive term and no `(1 − x)`-style factor anywhere, in particular no appearance of `bubble_xi_Tb`." And "**I3** | `:78`, `:100–101` | `bubble_xi_Tb` is read and range-validated but **never used** in any computation, including `T0`. Its sole effect is the possibility of raising."
- **Lens B says** — B-008/B-044: "`… * t^(-6/35) * (1-xi)^0.4`" … "`0.4` on `(1-xi)`". "**Falsified if:** … the `(1-xi)` factor is absent/differently formed."
- **Lens C says** — C-011: "`T(x) = C_T * L36^{8/35} * n0^{2/35} * t6^{-6/35} * (1 - x)^{2/5}`, `x = r/R2 ∈ [0,1)`" … "**s = 2/5**"; derivation leg 1 fixes the sign: "`T` is **hottest at the centre and → 0 at the contact discontinuity**, so the factor must be `(1 - x)^{2/5}`". **CONFIDENCE: HIGH.** Ranked #4.
- **verdict** — **The code is wrong** — a factor dropped from the product. The docstring documents it, the `.param` schema supplies it, the function validates it, and then nothing multiplies by it. Effect: `T0` is too **hot** by `(1−xi)^{-0.4}` (e.g. ×2.5 at `xi = 0.9`), and `bubble_xi_Tb` is a dead knob — turning it in a `.param` file changes nothing. Confidence: very high.
- **severity** — **S1** (wrong `T0`) with an **S2** companion: a parameter the user can set has no effect, so any calibration done against it is meaningless
- **how to settle it** — Vary `bubble_xi_Tb` between two runs and diff `T0`; if it is bit-identical the factor is confirmed missing. (A pytest case asserting `T0(xi=0.9) < T0(xi=0.1)` is the smallest permanent check.)

---

### R-006 · **LOUD** — `E0` uses the energy fraction `5/7`; the derived Weaver thermal fraction is `5/11`
- **quantity** — `WEAVER_ENERGY_FRACTION`, the coefficient of `Lmech_W · dt_phase0` in `E0`
- **pattern** — **A = B ≠ C** (code and comment agree with each other and both disagree with the physics)
- **Lens A says** — "`WEAVER_ENERGY_FRACTION = 5.0 / 7.0`" … "Exact value 5/7 = 0.7142857142857143. Dimensionless." … "`:167` `E0 = (5/7) · Lmech_W · dt_phase0`."
- **Lens B says** — B-004: "'Energy fraction in bubble interior: E0 = (5/7) * Lw * dt'." B-005: "The `5/7` energy fraction is 'From Weaver+77, Eq. 20'." B-042: "'From Weaver+77, Eq. 20: E = (5/7) * L_w * t'."
- **Lens C says** — C-009: "**⇒ E_th = (5/11) * L * t = 0.454545... * L * t** with the rest split as **15/77 ≈ 0.1948** into shell kinetic energy and **27/77 ≈ 0.3506** radiated away by the shell. (5/11 = 35/77, so the three add to 77/77.)" C gives a **full independent derivation** (`E_th = (3/2)PV = (14π/25)ρA⁵t = 1750/3850 · Lt = (5/11)Lt`) **and** three fetched secondary quotations ("the gas in the shocked stellar wind region has **5/11** of the total energy, in purely thermal form"; "**15/77** of the total injected wind energy"; "**35/77 and 15/77**"). "**My derivation and the fetched values agree exactly.**" — "**CONFIDENCE: HIGH** for the value 5/11 (derived *and* fetched); **LOW** for its Weaver equation number."
- **verdict** — **Code and comment are both wrong; the misapplied-literature class.** C's LOW confidence attaches **only to the equation number "Eq. 20"**, not to the value — the value is HIGH, derived from scratch and independently corroborated. So the no-ruling-on-LOW-confidence rule does **not** shield this entry. `E0` is too large by `(5/7)/(5/11) = 11/7 = 1.571`. Confidence: high on the value; the *citation* "Weaver+77, Eq. 20" is separately unverified and should not be trusted as authority for `5/7`. Caveat worth recording: C notes an alternative defensible convention `E0 = L_w · dt` ("nothing radiated yet"), and `5/7 = 0.714` sits between `5/11` and `1` — so a deliberate-but-undocumented third choice cannot be excluded from the lens evidence alone. Nothing in A or B suggests that was intended: B quotes Weaver Eq. 20 for it.
- **severity** — **S1**
- **how to settle it** — Open Weaver et al. 1977 (ApJ 218, 377) Eq. 20 and read the printed fraction. Failing that, the derivation is self-checking: `E_th + E_kin + E_rad = Lt` requires `5/11 + 15/77 + 27/77 = 35/77 + 15/77 + 27/77 = 1` ✓, whereas `5/7 = 55/77` leaves only `22/77` for kinetic **and** radiated combined, contradicting the independently confirmed `E_kin = 15/77` unless the shell radiates only `7/77` — which conflicts with the standard thin-radiative-shell assumption the `5/7`'s own comment invokes ("assumes adiabatic index gamma = 5/3", B-006). Cross-check `E_kin = ½Mv²` against `R = 0.76287(Lt³/ρ)^{1/5}` numerically.

---

### R-007 · **LOUD (unresolved)** — free-streaming duration carries `2π`; both derivations C could construct give `4π`
- **quantity** — `dt_phase0 = sqrt(3 · Mdot0 / (2 · π · rhoa · v0³))`
- **pattern** — **A = B ≠ C**
- **Lens A says** — "`dt_phase0 = sqrt( 3 · Mdot0 / (2 · π · rhoa · v0³) )`. Coefficients transcribed: numerator `3.0`, denominator `2.0 * np.pi`, `v0` cubed." And, neutrally: "**I7** | `:151` + `:163` | The pair implies the exact identity `(4π/3)·rhoa·r0³ = 2·Mdot0·dt_phase0`: the swept sphere holds exactly **twice** the injected mass. The equal-mass form of the same identity would put `4π/3` where `:151` has `2π`. Stated as observed algebra, not as a verdict."
- **Lens B says** — B-038: "'From Rahner thesis Eq. 1.15:  dt = sqrt(3 * Mdot / (2 * pi * rho_a * v^3))'. Numerator coefficient `3`; denominator `2 * pi * rho_a * v^3`". B-017 gives the citation: "'Free-streaming phase duration: Rahner thesis Eq. 1.15 … pg 17'."
- **Lens C says** — C-003: "**⇒ dt = sqrt( 3 * Mdot / (4 * π * rho_a * v^3) )** … **a = 3, b = 4, c = 3** — all derived, none assumed." C-004 gives the competing convention: "**⇒ a = 1, b = 4, c = 3**", "Ratio: `dt(A) / dt(B) = sqrt(3)`". C-005 on the cited source: "**I could not open Rahner's PhD thesis at all**, so **Eq. 1.15 is unverified**" … "**I therefore cannot state what Rahner Eq. 1.15 says, and I will not guess an equation number or a coefficient for it.**" — "**CONFIDENCE: LOW — could not confirm.**" C-003 itself: "**CONFIDENCE: HIGH** for the algebra given the condition; **MEDIUM** that this is the condition the literature the code follows actually adopts."
- **verdict** — **Unresolved — needs an independent source.** C explicitly could not reach Rahner's thesis, so per the audit rules I may not rule against the code on C's authority. What I *can* record: (i) **neither** of the two standard conditions C derived produces `2π` — both give `4π`, differing only in the numerator (3 vs 1); (ii) A, reading only code, independently identified the code's implicit condition as "swept sphere holds exactly **twice** the injected mass", which is not a condition C recognises; (iii) the deviation is exactly a factor 2 in the denominator, i.e. `dt`, `r0`, and `E0` are all larger by `√2 = 1.414` than condition A would give. Every energy/momentum variant of the swept-mass condition C sketches collapses to the same `4π` (because `pdot = Mdot·v` and `L = ½Mdot v²`), which makes an independent `2π` derivation hard to construct. Suspicion is high, proof is absent.
- **severity** — **S1 if confirmed** (√2 systematic in `dt_phase0`, `r0`, `E0`); recorded as **unresolved**
- **how to settle it** — Read Rahner (2018) PhD thesis Eq. 1.15, p. 17 (`https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf`), or the equivalent equation in the WARPFIELD papers (Rahner et al. 2017, MNRAS), from a network where the host is reachable. If the thesis prints `4π`, the code mis-transcribed its own citation (an A=B≠source case, since B faithfully quotes what the code's comment says). If it prints `2π`, the citation chain is intact and C's condition is simply not the one adopted — then trace *which* condition gives `2π` before accepting it.

---

### R-008 · `nCore` guard excludes only negatives; C says zero is the failure mode that must be excluded
- **quantity** — the input guard `if nCore < 0`
- **pattern** — A≠C (plus an A≠B doc-drift on the error message)
- **Lens A says** — "**C2** `:97` — strict `<` against `0`. **`nCore == 0` passes the guard**, yet the message at `:98` says 'must be positive'. The guard as written enforces non-negativity, not positivity; the one value the message singles out as illegal (0) is exactly the value that slips through, and it is the value that breaks `:151`." Failure mode **F1**: "`rhoa = 0` at `:146` → `:151` divides by `0` … `dt_phase0 = inf` → `t0 = inf`, `r0 = inf`, `E0 = inf`, `T0 = inf`. … Either way the 'must be positive' message never appears."
- **Lens B says** — B-027 claims only that an "INPUT VALIDATION" section exists; **U-9**: "validation criteria are unstated … never say what is checked, what the valid ranges are, or what happens on failure". (B is silent on the boundary itself — see coverage gaps.)
- **Lens C says** — C-016 E4: "**reject `rho_a <= 0`.** A `rho_a < 0` check alone is insufficient — **zero is the failure, and it is the physically tempting value** ('no ambient medium'). Vacuum has no free-streaming transition; the wind never decelerates". **CONFIDENCE: HIGH.** Ranked #6: "Density guard written `rho_a < 0` instead of `rho_a <= 0` (**zero density is the realistic failure**)".
- **verdict** — **The code's guard boundary is wrong**, and C called this exact off-by-one-on-the-boundary in advance. The message text is separately wrong (it promises a positivity test the code does not perform) — an A≠B drift on top. Confidence: high.
- **severity** — **S2** (latent; requires `nCore = 0`, which produces `inf`/`nan` or an uncaught `ZeroDivisionError` rather than a plausible-looking wrong number)
- **how to settle it** — A pytest case: `nCore = 0` must raise `ValueError`, not return `inf`. Change `<` to `<=` on line 97.

---

### R-009 · `mu_convert` is never validated; C requires `mu_convert > 0`
- **quantity** — input validation of `mu_convert`
- **pattern** — A≠C (B silent)
- **Lens A says** — "`mu_convert` is read at `:76` and **never validated** (flag **F2**)." Failure mode **F2**: "`mu_convert <= 0` or NaN | **Not validated at all.** Negative → `rhoa < 0` → `np.sqrt` of a negative → `nan` + `RuntimeWarning`, and `nan` propagates silently into all five return values".
- **Lens B says** — B-024 describes what `mu_convert` *is* ("mass per H nucleus — for rho = n_H * mu_convert") but makes no validation claim; U-9 says validation criteria are unstated.
- **Lens C says** — C-016 E5: "`mu_convert = 0` | same as E4 via `rho_a = 0` | **guard `mu_convert > 0`**". **CONFIDENCE: HIGH.**
- **verdict** — **The code is missing a guard C says is required.** Worse than R-008 in one respect: a negative `mu_convert` yields `nan` in all five returned values with no exception and only a numpy `RuntimeWarning`. Confidence: high (though `mu_convert` may be schema-constrained upstream — outside all three lenses' read; see coverage gaps).
- **severity** — **S2**
- **how to settle it** — Check whether `trinity/_input/` schema validation already constrains `mu_convert > 0`. If not, add the guard at this trust boundary (C: "validate at the schema/parse boundary, not deep in the solver").

---

### R-010 · `bubble_xi_Tb` guard is inclusive at 1; C requires strict `< 1`
- **quantity** — the guard `if not (0 <= bubble_xi_Tb <= 1)`
- **pattern** — A≠C (B's inference agrees with C but is not a doc claim)
- **Lens A says** — "**C3** `:100` — chained comparison, **inclusive at both ends**: 0 and 1 are both accepted; only `x < 0` or `x > 1` raise."
- **Lens B says** — U-5: "The parameter list … requires `bubble_xi_Tb`, but the prose never states that `xi` = `bubble_xi_Tb`, never gives its range (**presumably 0 ≤ xi < 1 for `(1-xi)^0.4` to be real**), and never says what it physically represents." (An inference by B, explicitly flagged as under-specified in the prose.)
- **Lens C says** — C-016 E7: "`xi = 1` exactly | `(1-1)^{2/5} = 0` ⇒ **`T0 = 0 K`** — silently unphysical, not an exception. Downstream this can divide by `T0` or feed a cooling-table lookup with T=0 | require **`0 <= xi < 1`, strict on the upper end**". **CONFIDENCE: HIGH.** Ranked #7.
- **verdict** — **The code's upper bound is wrong** per C. But note the interaction with R-005: because the `(1−xi)` factor is *absent* from the code, `xi = 1` currently has **no** effect at all, so the defect is dormant. It becomes live the moment R-005 is fixed — which is exactly the trap: fixing the missing factor without also tightening this guard introduces a silent `T0 = 0 K`. Confidence: high.
- **severity** — **S2** (latent, and currently masked by R-005 — fix both together)
- **how to settle it** — Change `<= 1` to `< 1` in the same commit that restores the `(1 − xi)**(2/5)` factor; add a pytest case asserting `bubble_xi_Tb = 1.0` raises.

---

### R-011 · Bad SPS values are **clamped to `1e-100`**; C says they must be rejected, not clipped
- **quantity** — the three floors `MIN_LUMINOSITY`, `MIN_MOMENTUM`, `MIN_VELOCITY` applied to `Lmech_W`, `pdot_W`, `v0`
- **pattern** — **A = B ≠ C** (code and comment agree on the clamping *policy*; C rejects that policy)
- **Lens A says** — "**C4** `:115` — strict `<` against `1e-100`. Because the test is `<` and not `abs(...) <`, it also fires for **zero and every negative value**, and the replacement is `+1e-100`, i.e. a negative luminosity is silently converted to a positive one." **F7**: "the sign error is swallowed; only a 'very small' warning is emitted, whose text does not mention negativity". **F6**: "Two independent clamps compose into a silently unphysical, non-NaN result." **D9/I9**: "One magnitude `1e-100` used as a floor for three quantities of three different dimensions … dimensionally arbitrary sentinel."
- **Lens B says** — B-012: "The constants that follow are 'Minimum valid values to prevent division by zero' — i.e. floors applied to inputs before division." (B-013/B-014/B-015 name the three individually.)
- **Lens C says** — C-016, "**Non-guards (do not add):** clamping `rho_a` or `L_w` to a small positive epsilon converts a configuration error into a wrong-but-plausible trajectory. These are trust-boundary inputs (`.param` file / SPS table); **validate loudly**." E2: "reject `L_w <= 0`. A cluster with zero wind luminosity has no energy-driven phase at all; the right response is an error or an explicit 'no bubble' path, **not** a clipped `L_w`". E3: "reject; SPS interpolation off the end of a table can produce these, so this is not a hypothetical". **CONFIDENCE: HIGH.**
- **verdict** — **A design disagreement, and C is explicit that the code's choice is the wrong one.** The code does not merely floor a denominator: it converts negative luminosities into positive ones and composes two clamps into a finite, absurd, undetectable trajectory (A's F6). C anticipated exactly this ("wrong-but-plausible trajectory"). Confidence: high on the principle; the code's behaviour is not in dispute between A and B.
- **severity** — **S2** (latent — only reached with degenerate SPS output, which C notes is "not a hypothetical" at table edges)
- **how to settle it** — Replace the three clamps with `ValueError` raises on `<= 0` and keep the clamp only where a genuine denominator underflow is expected; add pytest cases feeding negative `Lmech_W`/`pdot_W` and asserting a raise. If clamping is retained for operational reasons, it must be surfaced in the returned state so the integrator can distinguish a clamped start (A: "none of the substitutions is recorded in the returned tuple or in any flag").

---

### R-012 · `MIN_MOMENTUM`'s stated rationale only makes sense under the *correct* `v = 2L/pdot` — corroborates R-001
- **quantity** — the documented purpose of the `pdot` floor
- **pattern** — A≠B
- **Lens A says** — `pdot_W` is floored at `:119`/`:121`, and in the code `pdot_W` appears in the **numerator** of `v0 = 2.0 * pdot_W / Lmech_W`; the load-bearing denominator guard for `v0` is the `Lmech_W` floor. ("Post-clamp invariant … This invariant is what keeps the divisions at `:130` and `:134` from raising `ZeroDivisionError`".)
- **Lens B says** — B-014: "One floor exists specifically to 'Prevent div by zero in velocity calculation' — i.e. **it guards the denominator of `v = 2 * L / pdot` (B-034), so it floors `pdot`**. **Falsified if:** the floor is applied to a different variable than `pdot`, or is not applied on the velocity path."
- **Lens C says** — C-016 E1: "`pdot_w = 0` | `v = 2L/pdot` → **ZeroDivisionError / inf** | reject `pdot_w <= 0`".
- **verdict** — Not a defect in itself; recorded because it is **independent evidence that R-001 is a code-side typo, not an intended alternative**. The floor is applied to `pdot` and named for the velocity division — which is only coherent if `pdot` is the divisor, i.e. if `v = 2L/pdot`, as B and C both state. The constant's existence, name, and placement were written against the correct formula; only the expression at `:134` deviates.
- **severity** — **S3** (docs/consistency), but load-bearing as evidence for the S1 R-001
- **how to settle it** — Already settled by inspection: the floor's stated purpose and the expression it guards are mutually inconsistent under the as-written `v0`. Fixing R-001 makes the comment true.

---

### R-013 · `mu_convert` documented as "(=1.4)" while the code consumes it as a mass in Msun
- **quantity** — `mu_convert` and `rhoa = nCore * mu_convert`
- **pattern** — A≠B (A = C)
- **Lens A says** — "`mu_convert` multiplies `nCore` at `:146` to make a mass density, so it is a mass per particle in **Msun** (consistent with the `'m_H'` entry of the `unit_map` in `unit_conversions.py`, which maps the `m_H` unit token to `m_H[g] × g2Msun`, i.e. Msun)." Dimension check: "`[pc⁻³] × [Msun] = **Msun/pc³**`. Dimensionally clean".
- **Lens B says** — B-036: "'nCore is hydrogen nuclei density n_H; **use mu_convert (=1.4)** for mass density'." B's own internal contradiction **C-1**: "A quantity that is *mass per H nucleus* cannot have the bare value 1.4 unless the mass unit is implicit; 1.4 is the conventional **dimensionless** mean-molecular-weight-per-hydrogen factor."
- **Lens C says** — Notation: "`rho_a = n_H * mu_convert` where `mu_convert` = **mass per hydrogen nucleus**"; in the worked example, "`mu_convert = 1.4 m_H`".
- **verdict** — **The comment is wrong, the code is right.** A and C agree the quantity is a mass per nucleus; C's worked example spells it `1.4 m_H`, and A confirms the repo's `unit_map` supplies `m_H` in Msun. The comment's parenthetical "(=1.4)" drops the `m_H` and would mislead anyone re-deriving `rhoa`. Confidence: high.
- **severity** — **S3** (misleading docs only)
- **how to settle it** — Print `mu_convert` at runtime; if it is ~7e-58 (i.e. `1.4 m_H` in Msun) rather than `1.4`, the comment is confirmed wrong. Reword to "mu_convert = 1.4 × m_H, in Msun".

---

### R-014 · Comment claims "handle both DescribedItem and raw value access"; the code only does `.value`
- **quantity** — parameter access pattern
- **pattern** — A≠B (C silent)
- **Lens A says** — "`params`, treated as a mapping whose values are objects exposing `.value`. … No key existence check, no `.value` attribute check, no type check: a missing key raises `KeyError` and a missing attribute raises `AttributeError`, both uncaught (flag **F8**)."
- **Lens B says** — B-023: "'Core properties - handle both DescribedItem and raw value access' — the code accepts either a wrapped `DescribedItem` or a bare value for the core properties. **Falsified if:** only one of the two access patterns is actually supported (e.g. a bare `params['nCore']` with no `.value` fallback, or vice versa)."
- **Lens C says** — (silent; outside C's scope)
- **verdict** — **The comment is wrong.** B's falsification condition is met exactly: only the `.value` path exists, and a raw value raises an uncaught `AttributeError`. Confidence: high.
- **severity** — **S3** (unless a caller actually passes raw values, in which case S2 — not determinable from these three lenses)
- **how to settle it** — Call `get_y0` with a plain-`dict` `params` whose values are bare floats; if it raises `AttributeError`, delete or fix the comment.

---

### R-015 · Docstring says the free-streaming phase is "integrated"; the code is closed-form
- **quantity** — the free-streaming phase solution method
- **pattern** — A≠B (C's reference implementation is also closed-form)
- **Lens A says** — "No `try`/`except`, no `else` clauses, no loops, no early returns — a single straight-line path with three raise-points and three clamps." `dt_phase0` is a single `np.sqrt` expression.
- **Lens B says** — B-016: "'Obtain[s] initial values for the energy-driven (Weaver) phase **by integrating** a brief free-streaming phase…'"; U-12: "the only free-streaming expression given … is a **closed-form square root**. The docstring does not say what, if anything, is integrated."
- **Lens C says** — C-017 gives the whole handover in closed form (`dt = sqrt(3*Mdot/(4*pi*rho_a*v**3))`), i.e. no integration is required.
- **verdict** — **Wording drift in the docstring.** Nothing is integrated; the analytic solution is used, which C confirms is the right approach. Confidence: high. Cosmetic, but it sends a reader looking for an ODE that does not exist.
- **severity** — **S3**
- **how to settle it** — Reword "by integrating" → "from the analytic free-streaming solution".

---

### R-016 · Comment writes `E = (5/7) * L_w * t` (absolute time) where the code correctly uses `dt` — the classic silent-error slot, *not* tripped here
- **quantity** — the time argument of `E0` (and of `T0`)
- **pattern** — A≠B (A = C — **the code is right**)
- **Lens A says** — "`:167` `E0 = (5/7) · Lmech_W · dt_phase0`" and "`(dt_phase0)**(6.0/35.0)`" — both use the elapsed free-streaming duration, not `t0`.
- **Lens B says** — B-042: "'From Weaver+77, Eq. 20: E = (5/7) * L_w * **t**'". B's internal contradiction **C-2**: "`get_InitPhaseParam.py:26` states `E0 = (5/7) * Lw * **dt**` (an interval…). `:166`, quoting the same Weaver+77 Eq. 20, states `E = (5/7) * L_w * **t**` (bare `t`, which elsewhere in this file denotes absolute time, e.g. `t0`, `tSF`). … This is a live ambiguity, not cosmetic: `t0 = tSF + dt`, so `dt` and `t0` differ by `tSF`." U-4 raises the same ambiguity for `T0`'s `t`.
- **Lens C says** — C-013: "Both `E_th = (5/11) L t` and `T ∝ t6^{-6/35}` are functions of the **time since the wind switched on**. At the handover that is **`dt`**, not `t0 = tSF + dt`. … For `E0` the error would be linear in `t0/dt` — potentially huge, since `dt ~ 1e-5 Myr`." **CONFIDENCE: HIGH.** Ranked **#2** in C's most-likely-wrong list.
- **verdict** — **The code is correct; only the comment is ambiguous.** This was C's #2 predicted defect and the code does not have it — worth recording as an explicit clearance so a future reader does not "fix" it. The two comment restatements (`dt` at L26, `t` at L166) contradict each other, and the `t` form would be the wrong one. Confidence: high.
- **severity** — **S3** (docs only) — but the comment invites a future regression, so it is worth correcting
- **how to settle it** — Nothing to settle in the code; change the comment at :166 (and the `t` in the `T0` comment) to `dt` and state "elapsed free-streaming duration, not absolute time".

---

### R-017 · "PHYSICAL CONSTANTS (with literature references)" banner spans three unreferenced numerical sentinels
- **quantity** — module constant block organisation
- **pattern** — A≠B
- **Lens A says** — "`MIN_LUMINOSITY = 1e-100`, `MIN_MOMENTUM = 1e-100`, `MIN_VELOCITY = 1e-100` … Three distinct names, one identical magnitude 1 × 10⁻¹⁰⁰, applied as floors to three quantities of three different dimensions … it is a pure underflow/zero-division sentinel."
- **Lens B says** — B-003: "A banner declares the block that follows to be 'PHYSICAL CONSTANTS (with literature references)'". B's **C-5**: "`:37-40` then define division-by-zero floors, which are numerical guards with no literature reference, inside/adjacent to that block."
- **Lens C says** — (silent on file organisation)
- **verdict** — **The banner over-claims.** Confirmed by A: the three floors are dimensionless sentinels with no physical provenance. Confidence: high. Trivial.
- **severity** — **S3**
- **how to settle it** — Split the banner, or move the sentinels below it.

---

### R-018 · The temperature comment normalises `n` by `1 cm^-3` in one restatement and not in the other
- **quantity** — the density factor of `T0`
- **pattern** — A≠B (partial; primarily an internal contradiction inside B, and **A = C** on the substance)
- **Lens A says** — "`(nCore * cvt.ndens_au2cgs)` — `cvt.ndens_au2cgs` converts pc⁻³ → cm⁻³, so this factor is supplied in **cgs number density (cm⁻³)**. The very presence of this one conversion is direct in-code evidence that the `1.51e6` coefficient was calibrated for **cgs-style** inputs." A explicitly declines to flag the dual treatment of `nCore` (raw AU at :146, cgs at :174) as an inconsistency.
- **Lens B says** — B-009 (L31): "`(n/1 cm^-3)^(2/35)`"; B-044 (L171): "density `n` **unscaled** in this restatement". B's **C-3**: "Numerically equivalent only if `n` is already in cm^-3, which the surrounding prose … does not establish for this module."
- **Lens C says** — C-012: "`n0` | **ambient** number density **in cm⁻³**"; C-014: "using `nCore` (hydrogen nuclei, cm⁻³) directly as Weaver's `n0` is **acceptable**; the induced error is ≤ 5%". **CONFIDENCE: HIGH** on the bound.
- **verdict** — **The code is right and the L31 comment is right; the L171 restatement is merely abbreviated.** Not a defect. Recorded because B raised it as a live contradiction that A and C jointly resolve: the density factor is the one of the three `T0` factors that *is* correctly converted. Confidence: high.
- **severity** — **S3** (documentation tidiness only)
- **how to settle it** — Already settled by A's transcription of `cvt.ndens_au2cgs`; make the L171 restatement carry the `cm^-3` normalisation for consistency.

---

## Minor doc-only observations (recorded, no entry warranted)

- `pdot_W`'s warning at `:120` prints a raw AU number with no conversion and no unit, while its sibling at `:116` converts to erg/s and labels it (A's **I8**). B is silent; C is silent.
- `E0` is logged with **no unit** at `:181` and as `erg` at `:191` — the only returned value logged twice in two unit states (A's **I10**). B-046 documents only the second.
- `Qi_tSF` and `Lbol_tSF` are computed solely to feed a log line and are evaluated even when INFO logging is off (A's **I11**). B-045 describes the summary as "one-time"; neither B nor C addresses the eager evaluation.
- The `Qi` log conversion (`np.log10(Qi_tSF * cvt.s2Myr)` labelled `[1/s]`) is **cleared**: A finds it consistent given an interpolator returning Myr⁻¹, and B-047 independently states the internal unit is `[1/Myr]`.

---

## Coverage gaps (one or two lenses silent — **not** discrepancies)

| # | topic | who is silent | what would close it |
|---|---|---|---|
| G-1 | Formula for `r0` | **B** — U-6: "no formula is given for how `r0` is obtained (unlike `t0`, `E0`, `T0`, all of which get formulas)" | A (`r0 = v0*dt_phase0`) and C (C-006 `r0 = v*dt`) agree; add the formula to the comment |
| G-2 | Whether the docstring documents `v0`, `E0`, `T0` as returns | **B** — U-1: the extracted Returns block is truncated after `r0`; A shows a 5-tuple is returned | Re-read the raw docstring for a documented-vs-returned mismatch |
| G-3 | NaN handling | **B and C** — A's C1/C3/F3 show `tSF=NaN` and `nCore=NaN` pass their guards while `bubble_xi_Tb=NaN` raises, and NaN defeats all three clamps (`NaN < x` is `False`), returning `(nan,)*5` silently | Decide a policy; C's guard table (C-016) covers zero/negative but not NaN |
| G-4 | Whether `fLmech_W`/`fpdot_W` really are **wind-only** | **A** can only see the dict keys; C-001 requires wind-only `L` and `pdot` ("only physical when using wind-only L and pdot", B-028) | Inspect where `sps_f` is constructed; a SNe-inclusive total would silently violate `v = 2L/pdot` even after R-001 is fixed |
| G-5 | Whether γ is configurable elsewhere | **A and C** — B-006 records "assumes adiabatic index gamma = 5/3" | Grep the `.param` schema for a `gamma`; if it exists and feeds this path, the hardcoded fraction is a second-order defect |
| G-6 | Whether `get_y0` is called once per run ("One-time feedback summary", B-045) | **A** (single-file read) and **C** | Check the caller; if it is called per-restart the "one-time" claim is false |
| G-7 | Whether `mu_convert`/`nCore` are already validated upstream in `trinity/_input/` | **A, B, C** all confined to this file | Determines whether R-008/R-009 are real holes or belt-and-braces |
| G-8 | Weaver+77 equation numbers (Eq. 20, Eq. 37) and Rahner Eq. 1.15 | **C could not fetch any of them** — "**Values I could NOT pin down**: Rahner PhD thesis Eq. 1.15 … Weaver+77 equation numbers … The exact numerical constant `C_T`" | Obtain the papers from an unrestricted network; settles R-006's citation and R-007 entirely |

---

## Summary table — one row per computed quantity

| quantity | Lens A (code) | Lens B (comment) | Lens C (physics) | verdict |
|---|---|---|---|---|
| `Mdot0` | `pdot_W**2 / (2*Lmech_W)` | `Mdot = pdot^2 / (2*L)` | `Mdot = pdot_w²/(2 L_w)` (HIGH) | **A=B=C — cleared** |
| `v0` | `2*pdot_W / Lmech_W` | `v = 2*L/pdot` | `v = 2 L_w/pdot_w` (HIGH) | **R-001 · code wrong (inverted) · S1** |
| `rhoa` | `nCore * mu_convert` → Msun/pc³ | `rho = n_H * mu_convert`, "[AU units: Msun/pc^3]" | `rho_a = n_H * mu_convert` (HIGH) | **A=B=C — cleared** (see R-013 for the "(=1.4)" drift) |
| `dt_phase0` prefactor `3` | `3.0` | `3` | `3` (condition A) or `1` (condition B) | **cleared conditional on the 4π question** |
| `dt_phase0` geometry factor | `2.0 * np.pi` | `2 * pi` (cited to Rahner Eq. 1.15) | `4π` in **both** derived conventions (algebra HIGH; Rahner **LOW — could not confirm**) | **R-007 · A=B≠C · unresolved, S1 if confirmed (√2)** |
| `dt_phase0` exponent on `v` | `v0**3` | `v^3` | `v^3` (HIGH) | **A=B=C — cleared** |
| `t0` | `tSF + dt_phase0` | `t0 = tSF + free-streaming duration [Myr]` | `t0 = tSF + dt` (HIGH) | **A=B=C — cleared** |
| `r0` | `v0 * dt_phase0` | "[pc]", no formula given | `r0 = v*dt` (HIGH) | **A=C — cleared** (B coverage gap G-1) |
| `E0` coefficient | `5/7` | `5/7`, "Weaver+77, Eq. 20" | **`5/11`** (derived **and** fetched, HIGH on the value; LOW only on the eq. number) | **R-006 · A=B≠C · code+comment both wrong · S1 (×1.571)** |
| `E0` operands | `Lmech_W * dt_phase0` | `Lw * dt` (L26) vs `L_w * t` (L166) | `L_w * dt` — the **age**, not `t0` (HIGH) | **code right; R-016 · comment ambiguous · S3** |
| `T0` prefactor | `1.51e6` | `1.51e6 K` | `C_T ≈ 1.5e6 K`, spread 1.5–2.5e6 (**MEDIUM**) | **A=B=C within C's stated tolerance — cleared** (C: "Do not spend effort on a 25% discrepancy in `C_T`") |
| `T0` luminosity exponent | **`8.0/25.0`** | `8/35` | `8/35` (HIGH) | **R-003 · code wrong · S1** |
| `T0` luminosity units | AU luminosity ÷ bare `1e36`; **no `cvt.L_au2cgs`** | `L` in erg/s ÷ `10^36 erg/s` | `L36 = L_w[erg/s]/1e36 = L_astro × 6.026e-7` (HIGH) | **R-004 · code wrong · S1 (C's #1 predicted defect)** |
| `T0` density exponent | `2.0/35.0` | `2/35` | `2/35` (HIGH) | **A=B=C — cleared** |
| `T0` density units | `nCore * cvt.ndens_au2cgs` → cm⁻³ | `n/1 cm^-3` (L31); unscaled (L171) | `n0` in cm⁻³; H-nuclei acceptable, ≤5% (HIGH) | **A=C — cleared**; R-018 · comment inconsistency · S3 |
| `T0` time exponent | **`+6.0/35.0`** | `-6/35` | `-6/35` (HIGH) | **R-002 · code wrong (sign) · S1** |
| `T0` time units / which time | `dt_phase0` raw, no conversion | `t`, units and definition never stated (U-4) | `t6` = **age in Myr**, no factor needed (HIGH); must be `dt`, not `t0` (HIGH) | **code right — cleared** (resolves A's conditional flag D7) |
| `T0` `(1−xi)` factor | **absent**; `bubble_xi_Tb` never used | `(1-xi)^0.4` | `(1-x)^{2/5}`, decreasing outward (HIGH) | **R-005 · code wrong (term dropped) · S1** |
| guard `tSF < 0` | `<`; 0, −0.0, NaN, +inf pass | "INPUT VALIDATION" exists; criteria unstated | — | **cleared** (NaN → G-3) |
| guard `nCore < 0` | `<`; **0 passes**, message says "must be positive" | criteria unstated (U-9) | reject `rho_a <= 0` — "zero is the failure" (HIGH) | **R-008 · code wrong boundary · S2** |
| guard `mu_convert` | **none** | none stated | `guard mu_convert > 0` (HIGH) | **R-009 · missing guard · S2** |
| guard `0 <= xi <= 1` | inclusive both ends | range never stated; B infers `0 ≤ xi < 1` | `0 <= xi < 1`, **strict** upper (HIGH) | **R-010 · code wrong boundary · S2 (masked by R-005)** |
| `Lmech_W`/`pdot_W`/`v0` floors | clamp to `+1e-100`, incl. negatives | "prevent division by zero" | **reject, do not clamp** (HIGH) | **R-011 · A=B≠C · S2** |
| `MIN_MOMENTUM` rationale | floors a **numerator** as written | "guards the denominator of `v = 2*L/pdot`" | `pdot_w = 0` breaks `v = 2L/pdot` | **R-012 · A≠B · S3 — corroborates R-001** |
| `params` access | `.value` only, uncaught `AttributeError` | "handle both DescribedItem and raw value access" | — | **R-014 · comment wrong · S3** |
| method description | closed-form, straight-line | "by integrating a brief free-streaming phase" | closed form (C-017) | **R-015 · comment wrong · S3** |
| constants banner | 3 dimensionless sentinels | "PHYSICAL CONSTANTS (with literature references)" | — | **R-017 · banner over-claims · S3** |
| return tuple | `(t0, r0, v0, E0, T0)` | `t0 [Myr]`, `r0 [pc]`, rest truncated in extract | `(t0, r0, v0, E0, T0)` per C-017 | **cleared on shape**; units contaminated by R-001 (G-2) |

---

## Reading order for a fixer

1. **R-001** first — it contaminates every returned value and is a one-token swap. Nothing else can be validated numerically until it is fixed.
2. **R-004, R-003, R-002, R-005** together — all four live in the same `T0` product; compounded, they put `T0` around `1e-4 K` where `1e7 K` is expected (using C's worked case), so one printed `T0` confirms all four at once.
3. **R-005 + R-010 in the same commit** — restoring `(1 − xi)^{2/5}` without tightening the guard to `xi < 1` creates a new silent `T0 = 0 K`.
4. **R-006** — needs the Weaver paper, but C's derivation is self-checking via `5/11 + 15/77 + 27/77 = 1`.
5. **R-007** — do not touch until Rahner Eq. 1.15 is read. This is the one place where the code may be right and C's derivation simply not the adopted condition.
6. **R-008/R-009/R-011** — guard hardening, independent of the physics fixes.
7. **R-012 through R-018** — documentation, after the code settles.
