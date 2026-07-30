# S3 phase0 init — reconciled

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

**Status (2026-07-30):** 🔀 reconciled slice report — merged from three blind lens reports
(`S3_phase0_lensA.md`, `S3_phase0_lensB.md`, `S3_phase0_lensC.md`). The reconciler read **no source
code**; every statement below traces to a lens report or to arithmetic performed on numbers the
lenses reported. Nothing here has been verified against the current tree.

---

## 0. Method note and the one place the reconciler did arithmetic

Three lenses examined `trinity/phase0_init/get_InitPhaseParam.py` and
`trinity/phase0_init/get_InitCloudProp.py` in mutual isolation: **A** read comment-stripped source
and transcribed prefactors as coded; **B** read comments/docstrings only and transcribed formulas as
documented; **C** saw redacted signatures plus `PHYSICS_SPEC.md` and derived each coefficient from
scratch, with literature access blocked.

The reconciler performed arithmetic in exactly four places, all on numbers the lenses supplied, all
flagged inline: (i) scaling A's numeric `T0` probe onto C's fiducial to establish that both computed
the *central* Weaver temperature; (ii) applying C's `(1−ξ)^{2/5}` factor to that result; (iii)
`R1/R2 = sqrt(11 v2/(3 v_w))` at `v2 = v_w`; (iv) the `v0` off-manifold ratio `1/(3/5) = 5/3`. These
are reconciliation inferences, not lens findings, and are marked as such.

---

## 1. Coverage table

| Area | A (coded) | B (documented) | C (derived) | Reconciled coverage |
|---|---|---|---|---|
| `get_y0` derived quantities (`Mdot0`, `v0`, `rhoa`, `dt`, `r0`, `E0`, `T0`) | full, line-exact | full, formula-exact | full, derived | **3/3 — strongest coverage in the slice** |
| Energy coefficient `E = (5/11) L t` | ✅ line 28/167 | ✅ lines 26/166 | ✅ derived exactly | **3/3** |
| Free-streaming `dt`/`r0` prefactors | ✅ line 151/163 | ✅ line 150 | ✅ derived (2 criteria) | **3/3** |
| `T0` exponents 8/35, 2/35, −6/35, 0.4 | ✅ lines 173–176 | ✅ lines 31/171 | ✅ derived | **3/3** |
| `WEAVER_TEMP_COEFFICIENT` 1.51e6 | ✅ line 32 | ✅ line 30 | ⚠️ **declined** (recalled only) | **2/3 — pending literature** |
| `WEAVER_L_REF` 1e36 + AU→cgs boundary | ✅ (conversion confirmed applied) | partial (units of `t` omitted) | ✅ requirement stated | **3/3** |
| Weaver radius law `ξ_E (L t³/ρ)^{1/5}` | **not in code** (A used it as its own probe) | **absent from prose** | ✅ derived 0.7628653 | **not coded — see §3 row 2** |
| `P_b = E_b/(2π(R2³−R1³))`, `R1 = sqrt(ṗ/(4πP_b))` | not computed in slice | not documented in slice | ✅ derived | **out of slice — parked** |
| Power-law mass integral `M(<r)` | ✅ hand-verified | ❌ **no formula anywhere** | ✅ derived, identical | **2/3 (A=C), documentation gap** |
| `rCloud` closed form | call only (helper is out of slice) | ❌ absent | ✅ derived C27/C28 | **1.5/3 — cannot adjudicate here** |
| `nEdge < nISM` correction block | full decision table | full intent + admissions | ✅ says validate instead | **3/3** |
| Radius grid construction | full, line-exact | partial (1.5·rCloud, "small radius") | assumed a *linear* grid (wrong) | **2/3 — C's premise refuted by A** |
| Bonnor–Ebert branch | control flow + `c_s/1e5` | docstring claims + admissions | ✅ back-solved `c_s`, `T` | **3/3, from three different angles** |
| `mCloud` vs `(1−sfe)·mCloud` normalisation | silent | silent | ✅ C-20 | **1/3 — single-lens** |
| Floors / clamps | ✅ exact numeric consequences | intent only | ✅ requirement stated | **3/3** |
| Ordering / shared-state mutation | ✅ line-exact | ✅ from docstring omissions | silent | **2/3** |
| Exception handling | ✅ "there are none" | n/a | n/a | **1/3, high confidence** |

---

## 2. The three-way coefficient table

`A` = as coded (Lens A) · `B` = as documented (Lens B) · `C` = as derived (Lens C).
Verdict key: **CLEAR** = A=B=C · **SIGNATURE** = A=B≠C (code and comment agree, physics disagrees) ·
**DOC-DRIFT** = A≠B, C=A · **CODE-DRIFT** = A≠B, C=B · **PENDING-LIT** = C declined to derive.

| # | Quantity | A (coded) | B (documented) | C (derived) | Verdict |
|---|---|---|---|---|---|
| 1 | **Energy coefficient `E0 = (a/b)·L_w·t`** | `5.0/11.0` (`WEAVER_ENERGY_FRACTION`, :28 → :167) | `5/11` at :26 **and** :166, cited Weaver+77 Eq. 20, "assumes γ=5/3" | `5/11` exactly, derived end-to-end from the momentum+energy pair; `= c_E/c_L = (14π/25)/(154π/125)` | **CLEAR (A=B=C)** — verified-correct coefficient. Caveat is *applicability*, not value: C derives `(5−w)/(11−w)` for a power law and `5/(9γ−4)` for general γ → see R-02. |
| 2 | **`R(t) = (c·L_w/ρ)^{1/5}·t^{3/5}` prefactor `ξ_E`** | **absent** — no radius similarity law is coded in this slice. `r0 = v0·dt_phase0` (free-streaming) only | **absent** — no radius law appears anywhere in the prose | `ξ_E = (250/308π)^{1/5} = 0.76286534`, `c = 250/(308π) = 0.25836841` | **NOT CODED.** A and C independently produced the *same* constant (A used `(250/308π)^{1/5}` in its own cross-check probe) → the derived value is **corroborated A+C**. C's note that `PHYSICS_SPEC.md` SPEC-050 quotes `0.762934` (5th-digit slip; `0.762934⁵ = 0.2584859 ≠ 0.2583684`) is a **spec defect outside this slice** — hand to whoever owns SPEC-050. |
| 3 | **Solid angle in `dt_phase0`/`R_fs`** | `4.0 * np.pi` in the denominator, numerator `3.0`, `v0**3`, whole quotient `np.sqrt` (:151) | `dt = sqrt(3*Mdot/(4*pi*rho_a*v^3))` (:150), cited Rahner thesis Eq. 1.15 | `R_fs = sqrt(3Ṁ/(4πρ v_w))` ⇒ `t_fs = sqrt(3Ṁ/(4πρ v_w³))`; **`4π` is full solid angle in both**; a `2π` here would be a `√2 = 1.414` error | **CLEAR (A=B=C)** — prefactor for prefactor. Residual: C rates only *medium* that criterion (A) "swept = ejected" is the intended one; criterion (B) "ρ_wind = ρ_ambient" gives `R_fs/√3`. A independently re-derived criterion (A) from the coded arithmetic and got an exact match, which raises confidence that (A) is intended. |
| 4 | **`2π` / `4π` elsewhere** | mass integral uses `4.0/3.0 · π` and `4π ρ [rCore³/3 + …]` (:261, :263) — A hand-verified against the analytic integral | mass integral **not documented at all** (B's S3-B-27) | identical `M(<r)` derived; separately warns `P_b = E_b/(2πR³)` is `(γ−1)(4π/3)` inverted, **not** a hemisphere | **A=C, B silent.** No spurious `2π` was found in coded S3 geometry. C's `2π` trap applies to `P_b`, which this slice does not compute → parked (R-26). |
| 5 | `Mdot0 = pdot²/(2L)` | `pdot_W**2 / (2.0 * Lmech_W)` (:130) | `Mdot = pdot^2/(2*L)`, derived in prose from `L = 0.5 Mdot v²`, `pdot = Mdot v` (:129) | `Ṁ_w = ṗ_w²/(2L_w)` (C19, definitional) | **CLEAR (A=B=C)** |
| 6 | `v0 = 2L/pdot` | `2.0 * Lmech_W / pdot_W` (:134) | `v = 2*L/pdot` (:133) | `v_w = 2L_w/ṗ_w` (C19) | **CLEAR (A=B=C)** |
| 7 | `T0` exponent on `L36` | `8.0/35.0` (:173) | `8/35` at :31 and :171 | `8/35` derived from conduction closure + similarity scalings | **CLEAR (A=B=C)** |
| 8 | `T0` exponent on `n0` | `2.0/35.0` (:174) | `2/35` at :31 and :171 | `2/35` derived | **CLEAR (A=B=C)** |
| 9 | `T0` exponent on `t` | `-6.0/35.0` (:175) | `−6/35` at :31 and :171 | `−6/35` derived (uniform); generalises to `−6/[7(5−w)]` | **CLEAR (A=B=C)** on value; regime caveat folded into R-02 |
| 10 | `T0` exponent on `(1−ξ)` | `0.4`, written as a decimal (:176) | `0.4` at :31 and :171 | `2/5`, derived from `T(r) = T_b(1−r/R2)^{2/5}`; `(1−0.98)^{2/5} = 0.2091279` | **CLEAR (A=B=C)**. Cosmetic: A notes `0.4` is the only exponent in the block not written as `a.0/b.0`; `14/35` would keep the family consistent. |
| 11 | **`WEAVER_TEMP_COEFFICIENT`** | `1.51e6` (:32) | `1.51e6 K`, cited Weaver+77 Eq. 37, at :30 and :171 | **DECLINED.** C: "recalled, unverified", offers `1.51e6` *or* `2.07e6`; its own closure derivation gives `1.78–1.82e6`; acceptance range `[1.4e6, 2.2e6]` K; self-rated **low** confidence for the number | **PENDING LITERATURE.** A=B, and the coded value sits inside C's own acceptance range — but C explicitly refused to assert a value, so **this is not an A=B≠C signature defect** and must not be reported as one. Needs a human with journal access. |
| 12 | `WEAVER_L_REF` | `1e36`, dividing `Lmech_W * cvt.L_au2cgs` (:35, :173) — the AU→cgs conversion **is** applied | `10^36 erg/s` at :31 and :171 | `1e36 erg/s` with an explicit AU→cgs conversion, **or** `1.6596e6` in AU — mixing them is the failure | **CLEAR (A=B=C)**, and A's transcription **refutes** both B-16 ("which density is fed to T0?") and C-12 ("a missed conversion makes T0 wrong by 7 orders"): A confirms `L_au2cgs` on the luminosity *and* `ndens_au2cgs` on the density. Recorded as a verified-correct unit boundary; dropped from the findings list. |
| 13 | `rCore_min = rCloud (nCore/nISM)^{1/α}` | coded exactly (:188) | written identically (:187), and B checked it follows from `nEdge = nISM` | not derived (C recommends validating instead of auto-correcting) | **A=B, C n/a.** A independently confirms it is the exact inversion. Value correct; *use* is the finding (R-09, R-13). |
| 14 | `nCore_min = nISM (rCloud/rCore)^{−α}` | coded exactly (:219) | written identically (:217) | not derived | **A=B, C n/a.** Same as row 13. |
| 15 | Mass-check thresholds | `1e-3` (:269, production) **and** `0.01` (:513, diagnostic) | `> 1%` (:513) only; BE branch claims "EXACT … guaranteed" (:304) | requires `< 1e-10` for `verify_mass_at_rCloud` (SPEC-061/062 T16) | **CONTESTED, three different numbers** — A=coded-two, B=documented-one-plus-an-exactness-claim, C=requires-a-third. → R-05. |
| 16 | Grid constants `1e-3`, `1e-10`, `1.5` | coded, hardcoded, config-independent (:437, :447, :443) | only `1.5*rCloud` documented; inner bound stated as "small radius" | requires resolution below `min(rCore, R_fs)`; **assumed a linear grid** and derived `Δr = 0.02 pc` | **A≠C on premise.** A shows the grid is `np.logspace`, so C's specific failure mode is wrong. The surviving defect is A's: one interval spanning `1e-10 → 1e-3` pc. → R-10, demoted from C's S2 to S3. |
| 17 | `0.5 * rCloud` repair | coded (:242) | "iteratively halve rCore" (:234) | not derived; C would reject the config instead | **A=B on the number, both diverge from C's expectation** that this should raise. → R-08. |

**Signature-defect count (A=B≠C): 0 confirmed.** Every A=B pair in this slice either matches C
(rows 1, 3, 5–10, 12), has no C counterpart to disagree with (rows 13, 14), or sits against a C that
explicitly declined to derive (row 11). The one place code+comment agree while the physics objects is
row 1's *scope* rather than its value — the `5/11` coefficient is right, and is applied where C
derives that a different coefficient belongs (R-02). That is a **regime** finding, not a coefficient
error, and is classified accordingly.

---

## 3. Divergence table

| Item | A says | B says | C says | Divergence class | Status | Disposition |
|---|---|---|---|---|---|---|
| `mu_convert` dimensions | Msun **per particle** (`m_H[g]·g2Msun`, `unit_conversions.py:375`); `rhoa = nCore*mu_convert` dimensionally correct | "mean molecular weight for mass (=1.4)" (`get_InitCloudProp:90`) vs "mass per H nucleus" (`get_InitPhaseParam:76`) — mutually incompatible | `ρ = mu_convert · m_H · n_H` with `mu_convert = 1.4` | **AB (doc-drift)**, C=A | corroborated | **Keep, S3.** Code correct, docstring wrong. B rated S2; demoted because A settles it. Aggravator: A-25 shows the module's own `__main__` already took the docstring literally. → R-06 |
| Time symbol in `E0` | code uses `dt_phase0`, **not** `t0` (:167) | `E0 = (5/11)Lw·**dt**` (:26) vs `E = (5/11)L_w·**t**` (:166) — same citation, two symbols | Weaver's `t` is time since bubble formation ⇒ `dt` | **AB (doc-drift, cosmetic)** | corroborated | **Demote to note.** B-14's S2 failure scenario (`E0` inflated by `(tSF+dt)/dt ≈ 1000×`) is **refuted by A**: the code uses `dt_phase0`. Only a symbol inconsistency remains. Dropped from the ranked list. |
| Which `L` enters `E0`/`T0` | `Lmech_W = sps_f['fLmech_W'](tSF)` — wind-only, used for **both** | WIND-ONLY mandate scoped only to the velocity; `E0` writes `L_w`, `T0` writes bare `L` | Weaver Eq. 20/37 both refer to the driving mechanical luminosity | **AB (doc gap)** | corroborated | **Drop.** A refutes the concern; record as verified-correct. |
| `[Msun pc²/Myr³]` annotation | line 188 is `Lbol_tSF` — a **luminosity** | annotation "is a power, not an energy; wrong if it annotates E0" | AU energy is `Msun pc² Myr⁻²` | **none** | refuted | **Drop.** B itself conditioned the finding on which variable it annotates; A shows it annotates a luminosity, so the annotation is right. |
| `bubble_xi_Tb` range | validated `0 <= xi <= 1` at :100 | "no valid range stated; NaN for xi > 1" | requires `xi < 1` | **AB (doc gap)** | contested → resolved | **Demote to S4.** A refutes the NaN path. Surviving sliver: `xi == 1` exactly ⇒ `0.0**0.4 = 0.0` ⇒ `T0 = 0 K`, unguarded. → R-25 |
| `T0` reported at ξ=0.98 vs central | code multiplies by `(1 − bubble_xi_Tb)**0.4` | records the factor, no interpretation | C-14: "if T0 is a ξ=0.98 quantity the seed **must** carry the 0.20913 factor" | **none** | refuted | **Drop C-14.** The code already carries it. This also materially changes C-13 — see next row. |
| `T0` vs shocked-wind ceiling | probe: `T0 = 6.7e7 K` at `nCore=1e3`, `v_w=2000 km/s` | silent | C-13: Weaver `T0` at `t_fs` is `1.29e8 K`, **2.3×** above the `5.53e7 K` ceiling | **AC (physics)** | **contested** | **Demote to S3 and qualify.** *Reconciler arithmetic:* scaling A's probe onto C's fiducial gives `6.7e7 × (1e5/1e3)^{2/35} × 10^{6/35} = 1.294e8` — an exact match, proving **both lenses computed the *central* value with `(1−ξ)^{0.4} ≈ 1`**. Applying the coded `(1−0.98)^{0.4} = 0.2091` factor gives `T0 ≈ 2.7e7 K`, **below** the `5.53e7 K` ceiling. So the violation C reports is real only for small `bubble_xi_Tb`. Neither A nor B states the shipped default, so this needs one lookup. → R-26 |
| Radius grid geometry | `np.logspace(log10(1e-3), log10(rCloud), 1000)` + 100 outside + isolated `1e-10` | "logspace from small radius to rCloud" | assumed **linear**, derived `Δr = 0.02 pc`, rCore inside first cell | **AC (premise)** | contested → partially refuted | **Demote C-18 S2→S3, merge into A-09.** The log grid resolves `rCore` fine; the real hole is the single `1e-10 → 1e-3` pc interval. → R-10 |
| `rCore` halving loop | `for _iter in range(50)`, bounded, no post-loop check | "unbounded loop, no iteration cap, no floor" | n/a | **AB (fact)** | contested → A wins | **Drop B-07's "unbounded" framing;** keep A-04's correct version (bounded, no convergence check). → R-08 |
| Floors log or not? | all three clamps `logger.warning` then continue | "prose does not say whether clamping is logged" | "must log at WARNING or raise … a physically-sized floor is silent" | **AC (partial)** | corroborated on substance | **Keep, S2, reframed.** A refutes "silent" (they warn) and satisfies C's magnitude bar (`1e-100` is a pure div-zero guard). The surviving defects are A's: execution continues with a fabricated state, and the `v0` clamp fires *after* `Mdot0` is formed, breaking `Mdot0 = pdot/v0` — which is exactly C's I10. → R-03 |
| BE `T_eff` semantics | `params['densBE_Teff'] = T_eff`; only arithmetic on the path is `c_s/1e5` | "T_eff is a recast of the support velocity dispersion, not a gas temperature" | back-solves `c_s ≈ 10–20 km/s` ⇒ `T ≈ 3e4–1e5 K`, vs SPEC-066 T15 expecting 10–30 K | **BC (physics)** | corroborated | **Keep, S3.** Two lenses reached the same conclusion from prose and from physics independently; A supplies the mechanism. → R-12 |
| BE stability | `is_stable` → log string only, never propagated | no stated range for `densBE_Omega` | `Ω = 14.1 > 14.04` is formally unstable | **ABC** | corroborated | **Keep, S3**, merged into R-12 |
| `α = −3` / `α` range | `(3.0+α)` divisor unguarded ⇒ `ZeroDivisionError`; `_validate_params` checks only that the key exists | correction block's `α ≠ 0` remark is the only guard | `−2 ≤ densPL_alpha ≤ 0` must be enforced (SPEC-003) | **AC (validation gap)** | corroborated | **Keep, promoted A's S4 → S3** on C's evidence that a range exists and is unenforced. → R-09 |
| `mCloud` vs `(1−sfe)·mCloud` | silent (sfe not read in this slice) | silent | C-20: 11% in `rCloud`, 30% in swept mass at `sfe = 0.3` | **single-lens** | single-lens | **Keep, S3, flagged out-of-slice-dependent.** Cannot be adjudicated from S3 alone. → R-17 |
| Weaver/Rahner equation numbers | n/a | transcribed verbatim; notes hygiene is *good* (one citation → one formula; no coefficient quoted two ways) | refuses to assert any equation number (no journal access) | **pending literature** | corroborated-as-unresolvable | **Keep, S4.** → R-19 |

---

## 4. The four specific questions in the brief

**(a) Free-streaming radius and its implied balance.** Fully reconciled, **A=B=C**. Code:
`dt_phase0 = sqrt(3·Mdot0/(4π·rhoa·v0³))`, `r0 = v0·dt_phase0`. A re-derived the balance from the
coded arithmetic alone and recovered "swept ambient mass = ejected wind mass" prefactor for
prefactor; C derived the same balance from scratch; B transcribed the closed form with a Rahner
Eq. 1.15 citation but — B's own observation — **never states the balance**. So the physics is right
and the documentation omits the one sentence that would let a reader check it. Residual risk (C,
medium confidence): criterion (B) "wind density = ambient density" gives `R_fs/√3`, and nothing in
code or prose names which criterion is intended.

**(b) Dimensional consistency of each initialised quantity.** A dimension-checked all six from the
arithmetic; C independently stated the expected AU units. They agree on every one: `t0` Myr, `r0` pc,
`v0` pc/Myr, `E0` Msun pc² Myr⁻², `T0` K, plus the intermediates `Mdot0` Msun/Myr, `rhoa` Msun/pc³.
**Two exceptions, both real:** the `dt_phase0**(−6/35)` factor in `T0` is dimensionally inhomogeneous
and correct only because the AU time unit happens to be Myr (R-07); and `densBE_sigma` is stored in
km/s while every other velocity in the codebase is pc/Myr, a 2.27% offset (R-11).

**(c) Is the initial state self-consistent with the ODE system it seeds? No — and this is the
headline.** Three independent routes converge:
- **A (coded):** `v0` is the wind terminal velocity `2L/pdot`, and `r0 = v0·dt_phase0` exactly.
- **C (derived):** the Weaver similarity solution that `E0 = (5/11)L·t` belongs to requires
  `v2 = (3/5)·R2/t`.
- **B (documented):** flagged the same collision from prose alone — free-streaming `r0`/`v0` combined
  with a Weaver-energy `E0` at one instant, "with no reconciliation stated".

*Reconciler arithmetic:* since `r0/dt_phase0 = v0` identically, the coded `v0` is `1/(3/5) = **5/3 =
1.667×** the on-manifold velocity at the seed's own radius and time. A's measurement that `r0` and the
Weaver radius agree to **13%** shows the *radius* is nearly on-manifold, which makes the **velocity**
the sole and unambiguous inconsistency — the two lenses' numbers do not conflict, they localise the
defect. C's consequence chain (`R1/R2 = sqrt(11 v2/(3 v_w))`) evaluated at the coded `v2 = v_w` gives
`R1/R2 = **1.915**`, worse than the `1.483` C assumed; if any downstream module forms
`V_b = (4π/3)(R2³ − R1³)` at step 0 it is negative. **That module is out of slice**, so the failure
mode is credible but unproven. → **R-01**

**(d) Is a uniform-medium coefficient applied to a power-law profile? Yes, unconditionally.** ABC
corroborated. A: `rhoa = nCore·mu_convert` is the **constant core** density, "self-consistent only
while `r0 ≤ rCore`; nothing in this function knows `rCore` or `rCloud`". B: same observation from the
comments, with no stated validity condition. C: derives that the correct energy fraction on a
power law is `(5−w)/(11−w)` — `1/3` at `α = −2` versus the coded `5/11`, a **1.36×** over-fill — that
`R_fs ∝ ρ₀^{−1/2}` so diffuse clouds push `r0` far outside `rCore`, and that at `α = −2` the
free-streaming criterion has **no unique root** at all without the flat core. C's rule is the right
one and is not implemented: `R_fs ≤ rCore` → uniform formulae are exactly right regardless of `α`;
`R_fs > rCore` → they are wrong. **Contested detail:** C uses `rCore = 0.01 pc` as the default and
concludes the published `paperII` grid exercises the bad branch; B cites the project convention
`rCore ≈ 1 pc`. The *structure* (an unchecked assumption) is corroborated; the *blast radius* is not.
→ **R-02**

---

## 5. What the audit verified as correct

A reconciled list shorter than the sum of its inputs is the goal, and several inputs were retired by
being **confirmed right**. Recording them, because a verified coefficient is a real audit result:

- `E0 = (5/11)·L_w·dt` — coefficient A=B=C, and the time argument is `dt_phase0`, not `t0` (kills B-14's 1000× scenario).
- `dt_phase0` free-streaming prefactor `3/(4π)` with `v0³` under a square root — A=B=C, prefactor for prefactor.
- All four `T0` exponents (`8/35`, `2/35`, `−6/35`, `0.4`) — A=B=C.
- `Mdot0 = pdot²/(2L)` and `v0 = 2L/pdot` — A=B=C, exact inverses of `L = ½Ṁv²`, `ṗ = Ṁv`.
- The `WEAVER_L_REF` unit boundary: `L_au2cgs` **is** applied before the `1e36` division, and `ndens_au2cgs` **is** applied before the `2/35` power (kills B-16 and C-12).
- The power-law mass integral `4πρ_c[rCore³/3 + (r^{3+α} − rCore^{3+α})/((3+α)rCore^α)]` — A hand-verified, C derived, identical.
- `rCore_min` and `nCore_min` are both exact inversions of `nEdge = nISM` for their respective unknowns.
- Wind-only `Lmech_W` is used for `Mdot0`, `v0`, `E0` **and** `T0` (kills B-19).
- All `params` writes precede the `get_density_profile`/`get_mass_profile` reads within each function.
- `bubble_xi_Tb` **is** validated to `[0,1]` (kills B-21's NaN path).
- The coded `(1−ξ)^{0.4}` factor means `T0` is already a ξ=0.98 quantity (kills C-14).
- `ξ_E = 0.7628653` — derived twice independently (A's probe, C's derivation) to all quoted digits; the `0.762934` in `PHYSICS_SPEC.md` SPEC-050 is the outlier.

---

## 6. Ranked findings

Severity floor for this slice is **S2** — no S1. Justification: no lens demonstrated a wrong
*converged* result or a crash on a shipped configuration. R-01's strongest consequence (negative
`V_b` at step 0) depends on out-of-slice code and was not observed; R-02 becomes S1 if someone
confirms a published grid point runs with `r0 > rCore`, which no lens checked.

```json
[
  {
    "id": "S3-R-01",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 163,
    "class": "state",
    "severity": "S2",
    "claim": "The returned initial state is not a point on the Weaver similarity solution that E0 assumes. r0 = v0*dt_phase0 with v0 = the wind terminal velocity 2L/pdot, so v0 = r0/dt exactly; the energy-driven similarity solution whose E0 = (5/11)*L*dt is used requires v2 = (3/5)*R2/t. The coded seed velocity is therefore 5/3 = 1.667x the on-manifold value at its own radius and time. Free-streaming kinematics (r0, v0) are combined with a Weaver energy budget (E0) at one instant with no reconciliation.",
    "evidence": "A (coded): get_InitPhaseParam.py:134 v0 = 2.0*Lmech_W/pdot_W; :163 r0 = v0*dt_phase0; :167 E0 = (5.0/11.0)*Lmech_W*dt_phase0. C (derived): eta = 3/5 from 5*eta-2 = 1, so v2 = (3/5)R2/t on the similarity track; R1/R2 = sqrt(11*v2/(3*v_w)) exactly. B (documented): flagged the same collision from comments alone (S3-B-17). Reconciler arithmetic on lens numbers: v0/(0.6*r0/dt) = 1/0.6 = 1.6667; C's R1/R2 relation at the coded v2 = v_w gives sqrt(11/3) = 1.915 (C assumed v2 = 0.6*v_w and got 1.483). A's independent probe measured r0 within 13% of the Weaver radius at the same t, which localises the inconsistency to the velocity, not the radius.",
    "expected": "Either exactly one of {r0, t0} is primary and the rest follow from the similarity law (v2_0 = 0.6*r0/dt, E0 = (5/11)L*dt), or the hand-off is documented as a deliberate approximation with the mismatch quantified. An explicit assertion R1_0 < R2_0 (equivalently v2_0 < (3/11)*v_w) at the end of get_y0 would catch the dangerous case.",
    "failure_scenario": "The energy-phase integrator starts off the similarity manifold and relaxes onto it as t^(-2/5), contaminating the earliest snapshots and any finite-difference estimate of alpha = v2*t/R2, beta and delta that is fed back into the bubble-structure solver. Worse case, unproven from this slice: if a downstream module forms V_b = (4pi/3)(R2^3 - R1^3) with R1 = sqrt(pdot/(4*pi*P_b)), then R1/R2 = 1.915 makes V_b negative and P_b negative at step 0, surfacing as an unrelated-looking first-step solver failure.",
    "repro": "Run param/simple_cluster.param; from the first record of dictionary.jsonl check v2*t_now/R2 (expect 0.6 on-manifold, ~1.0 as coded) and check R1 < R2.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S3-B-17", "S3-C-03", "S3-C-04", "S3-C-16", "S3-A-01(sec1.4)"]
  },
  {
    "id": "S3-R-02",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 146,
    "class": "regime",
    "severity": "S2",
    "claim": "Uniform-medium coefficients are applied unconditionally to a power-law cloud. rhoa is built from the constant core density nCore with no check that r0 <= rCore, and the 5/11 energy fraction, the sqrt(3*Mdot/(4*pi*rho*v^3)) free-streaming form and the -6/35 temperature exponent are all uniform-medium results. get_y0 never sees rCore or rCloud, so the condition that makes them valid cannot be tested where they are used.",
    "evidence": "A: get_InitPhaseParam.py:146 rhoa = nCore*mu_convert; :151 dt_phase0 uses rhoa; A states 'this is self-consistent only while r0 <= rCore. Nothing in this function knows rCore or rCloud' and lists only five params keys read (mu_convert, nCore, bubble_xi_Tb, tSF, sps_f). B: S3-B-18, same observation from the comments, no stated validity condition. C: derives E_b/(L_w t) = (5-w)/(11-w) for rho ~ r^-w, i.e. 1/3 at alpha = -2 versus the coded 5/11 (a 1.36x over-fill); derives R_fs^(2-w) = (3-w)Mdot/(4*pi*rho_ref*r_ref^w*v_w), whose exponent VANISHES at w = 2 so the criterion has no unique root; notes R_fs scales as rho^(-1/2) so diffuse clouds push r0 far outside rCore.",
    "expected": "C's rule, implemented as a check rather than an assumption: if R_fs <= rCore, the uniform formulae with rho = rho_core are exactly right regardless of alpha; if R_fs > rCore, the power-law coefficients apply. At minimum, get_y0 should receive rCore and assert r0 <= rCore, or log when it fails.",
    "failure_scenario": "For a diffuse cloud the free-streaming radius leaves the flat core, and then the ambient density, the energy fraction and the temperature exponent are all evaluated in a regime where none of them holds. At densPL_alpha = -2 the underlying criterion is not even well-posed. Every downstream number inherits the offset with no warning.",
    "repro": "Print r0 (get_y0) and rCore for param/simple_cluster.param and docs/dev/performance/f1edge_{lowdens,hidens}*.param and check r0 <= rCore in each. Then compare snapshot-0 Eb/(Lmech*t) for densPL_alpha = 0 and -2 at low nCore; both equalling 0.4545 confirms the uniform coefficient is unconditional.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S3-A-01(sec1.4 assumption 1)", "S3-B-18", "S3-C-02", "S3-C-22"]
  },
  {
    "id": "S3-R-03",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 115,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Three 1e-100 floors (MIN_LUMINOSITY, MIN_MOMENTUM, MIN_VELOCITY) replace computed values and let get_y0 return a fabricated initial state that passes every later check. Independently, the v0 floor fires AFTER Mdot0 has been formed, so once it bites the returned state no longer satisfies Mdot0 = pdot_W/v0 or L = 0.5*Mdot*v^2.",
    "evidence": "A: get_InitPhaseParam.py:115/119/136 warn-then-assign; :130 Mdot0 computed before the :136 v0 clamp. A evaluated the clamped paths numerically: both L and pdot clamped gives v0 = 2.0*1e-100/1e-100 = exactly 2.0 pc/Myr, dt = 2.08e-52 Myr, r0 = 4.15e-52 pc, E0 = 9.4e-153, T0 = 8.5e-10 K; the v0-clamped path gives dt = 2.34e+150 Myr, i.e. t0 = tSF + 2.3e150 Myr. All returned without error. C: I10 requires L_w = 0.5*Mdot*v^2 and pdot = Mdot*v to hold exactly after any flooring, and warns that independent floors break this silently. B: S3-B-20, prose does not say which quantity each floor clamps.",
    "expected": "Floor once and derive the rest (or floor v_w only and back out Mdot = pdot/v_w so pdot is exactly preserved). A no-wind SPS state at tSF is a configuration error and should raise or return a documented 'no phase 0' sentinel, not a number. Note A's transcription partly satisfies C: the clamps DO log at WARNING and 1e-100 is a pure divide-by-zero magnitude, not a physical floor -- the defect is continuing, plus the ordering.",
    "failure_scenario": "An SPS table with zero wind luminosity at tSF (or a tSF before wind onset) seeds the integration from r0 ~ 4e-52 pc, T0 ~ 1e-9 K or t0 ~ 1e150 Myr. In a sweep the warning scrolls past and the affected cells are indistinguishable in the output files.",
    "repro": "Call get_y0 with an sps_f whose fLmech_W and fpdot_W return 0.0 at tSF and inspect the returned tuple; separately assert Mdot0*v0 == pdot_W after a v0 clamp.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S3-A-08", "S3-B-20", "S3-C-08", "S3-C-09"]
  },
  {
    "id": "S3-R-04",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 230,
    "class": "state",
    "severity": "S2",
    "claim": "get_InitCloudProp overwrites params['nCore'] (with a strictly larger nCore_min) and params['rCore'] in its correction branches, while get_y0 reads params['nCore'] and params['mu_convert'] to build the ambient density. Which value phase 0 sees depends entirely on the call order of the two functions, which no docstring states and no code enforces. nCore is also absent from the docstring's exhaustive in-place-update list, and rCore is documented as '(user-specified)' in the same block that lists it as an in-place update.",
    "evidence": "A: get_InitCloudProp.py:228-230 nCore = nCore_min then params['nCore'].value = nCore; :206/:229/:248/:278 write rCore; get_InitPhaseParam.py:77 reads params['nCore'], :146 rhoa = nCore*mu_convert, :151 rhoa enters dt as rhoa^(-1/2), :174 nCore enters the T0 exponent. B: S3-B-02 (nCore missing from the Notes update list, though :215 raises it and :276 stores it back), S3-B-03 (rCore self-contradictory in one docstring block), S3-B-04 (no ordering contract).",
    "expected": "Either get_y0 takes the ambient density from the CloudProperties object rather than from a params key another function mutates, or the ordering is asserted and the docstring lists nCore as an in-place output.",
    "failure_scenario": "If get_y0 runs before the cloud correction, dt_phase0 and r0 are computed against a density the cloud no longer has; the mismatch scales as (nCore_corrected/nCore_user)^(1/2). Separately, a sweep over rCore silently collapses several inputs onto corrected values, so the recorded output rCore no longer matches the .param the run is labelled with.",
    "repro": "Instrument the call order of get_y0 and get_InitCloudProp in run.py for a config that triggers the nCore fix; log params['nCore'] at both call sites and diff input vs output rCore.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-11", "S3-B-02", "S3-B-03", "S3-B-04"]
  },
  {
    "id": "S3-R-05",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 269,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Every consistency failure in cloud initialisation warns and continues: the mass check (rel err > 1e-3) logs 'Continuing with current values', and the post-correction edge-density check (nEdge < nISM) logs 'Continuing anyway'. Nothing is raised and no flag is recorded in CloudProperties or params, so an unconverged cloud is indistinguishable from a converged one in the output. The same physical mass check carries three different tolerances across the slice: 1e-3 on the production path, 1e-2 in verify_mass_at_rCloud, and C's required 1e-10 for the branch whose docstring calls the result EXACT.",
    "evidence": "A: get_InitCloudProp.py:268-274 (1e-3, warn only), :252-256 (nEdge < nISM, warn only, no flag), :513 (0.01 in verify_mass_at_rCloud). B: S3-B-08 (the final safety check only warns, and both corrections target the boundary nEdge = nISM exactly so round-off can leave it marginally violated), S3-B-09 (BE branch claims 'EXACT results: M(rCloud) = mCloud guaranteed' at :304 while a 1% verifier with an interpolation fallback exists). C: S3-C-17 requires |M(<rCloud)/mCloud - 1| < 1e-10 per SPEC-061/062 T16.",
    "expected": "Raise on the mass check (mCloud is the primary user input), or at minimum one consistent threshold plus a flag on CloudProperties recording that a correction failed, so output metadata can mark the run. For the BE branch the bar should be machine precision, matching its own exactness claim.",
    "failure_scenario": "A batch sweep writes results for configurations whose cloud mass differs from the .param value or whose edge is less dense than the ambient medium; every mass-dependent quantity downstream (shell mass, gravity, column density) inherits the offset with no output-visible marker.",
    "repro": "Trigger the correction across a grid of alpha, nCore, rCore and assert nEdge >= nISM and mass_rel_err < 1e-10 on exit; count how often each warning fires.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S3-A-05", "S3-A-06", "S3-B-08", "S3-B-09", "S3-C-17"]
  },
  {
    "id": "S3-R-06",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 76,
    "class": "units",
    "severity": "S3",
    "claim": "mu_convert is documented two incompatible ways -- 'mean molecular weight for mass (=1.4)' (dimensionless) and 'mass per H nucleus - for rho = n_H * mu_convert' (dimensional) -- while the code uses the dimensional reading. The comment is the drifted artefact, not the code, and the drift has already produced a downstream error inside the module itself.",
    "evidence": "A resolves the ambiguity: unit_conversions.py:375 maps the unit string 'm_H' to m_H[g]*g2Msun, so mu_convert carries Msun per particle (1.4*m_H ~ 1.2e-57 Msun) and get_InitPhaseParam.py:146 rhoa = nCore*mu_convert is dimensionally correct. C independently expects rho = mu_convert*m_H*n_H with mu_convert = 1.4 -- the same physics. B: S3-B-01 records both prose statements verbatim. Aggravator: A-25 shows get_InitCloudProp.py:571-574's __main__ demo passes mu_convert = 1.4 and nCore = 1e3 as raw cgs-flavoured numbers, i.e. somebody already read the docstring literally.",
    "expected": "One convention stated once: mu_convert is a mass per hydrogen nucleus in Msun (= 1.4 m_H), so rho = n_H * mu_convert needs no extra m_H. The '(=1.4)' should be removed or written as '= 1.4 m_H'.",
    "failure_scenario": "A caller or a future edit takes the docstring at face value and multiplies by m_H again (or omits it), making rho_a wrong by ~1e57 or by 1.4. The 1.4 version is the dangerous one: r0 scales as rho^(-1/2), so it is a 16% radius error with nothing to trip on.",
    "repro": "Compare the numeric mu_convert value against the expression building rho_a in get_InitPhaseParam.py:146 and rhoCore in get_InitCloudProp.py:259; confirm both use the same factor and that neither multiplies by m_H again.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-B-01", "S3-A-25", "S3-C-07"]
  },
  {
    "id": "S3-R-07",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 175,
    "class": "units",
    "severity": "S3",
    "claim": "In the T0 expression the luminosity and density factors are explicitly normalised to named units (L_au2cgs then /1e36; ndens_au2cgs to cm^-3) but the time factor dt_phase0**(-6.0/35.0) is used bare. The line is dimensionally inhomogeneous and correct only because the AU time unit happens to be Myr, matching Weaver's t6 fitting variable. The docstring transcription gives units for L and n and none for t, so the assumption is unstated in both code and prose.",
    "evidence": "A: get_InitPhaseParam.py:173 carries L_au2cgs and the 1e36 divisor, :174 carries ndens_au2cgs, :175 carries neither; dt_phase0 is in Myr per the :151 dimension check. B: S3-B-15, the Eq. 37 transcription at :31 gives '(L/10^36 erg/s)' and '(n/1 cm^-3)' but bare 't'. C: S3-C-10/S3-C-12, t must be in Myr (t6) and the cgs boundary must be crossed explicitly.",
    "expected": "A WEAVER_T_REF = 1.0  # Myr divisor on line 175, mirroring WEAVER_L_REF, so the unit assumption is stated in the arithmetic; and '(t/1 Myr)^(-6/35)' in the docstring.",
    "failure_scenario": "If the AU time unit ever changes, or the formula is copied into a routine whose time is in yr or s, T0 silently scales by (unit ratio)^(-6/35) while the L and n factors stay correct. Feeding yr instead of Myr is a factor 11 in T0 -- wrong but plausible, with no dimensional tripwire.",
    "repro": "Compute T0 by hand from the logged L, n and dt_phase0 in Myr and compare with the returned T0.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-02", "S3-B-15", "S3-C-10"]
  },
  {
    "id": "S3-R-08",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 242,
    "class": "state",
    "severity": "S3",
    "claim": "In the nCore-correction repair loop the user's rCore is replaced by 0.5 * rCloud -- an arbitrary geometric value unrelated to the nEdge/nISM condition that triggered the repair -- and written back to params. The loop is capped at 50 iterations with no post-loop convergence check, so exhausting it leaves rCore >= rCloud (a core larger than its cloud) and the very next line stores that.",
    "evidence": "A: get_InitCloudProp.py:241-248 for _iter in range(50): rCore = 0.5*rCloud; rCloud, _ = compute_rCloud_powerlaw(...); if rCore < rCloud: break -- then params['rCore'].value = rCore at :248 and again at :278. rCore is set to half the PREVIOUS iterate's rCloud, so the stored value is 0.5x an intermediate that is not the final rCloud either. The only exit signal is the break; _iter is unused afterwards. B: S3-B-07 flagged the loop from the comment, but claimed it is unbounded -- A refutes that; the bounded-but-uncheckedconvergence version is the correct account.",
    "expected": "Solve the coupled (rCore, rCloud) fixed point properly, or raise and tell the user their (mCloud, nCore, rCore, alpha, nISM) combination is inconsistent. At minimum an else: clause on the for-loop that raises or warns when the repair did not converge.",
    "failure_scenario": "A sweep point silently runs with an rCore roughly half the cloud radius instead of the ~1 pc its .param specified; the whole density profile, and therefore every downstream number, belongs to a different cloud. Only a warning distinguishes it.",
    "repro": "Construct a densPL .param with alpha < 0 where nEdge < nISM and rCore_min >= rCloud (low mCloud, high nISM), call get_InitCloudProp, and compare params['rCore'] before and after.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-03", "S3-A-04", "S3-B-07"]
  },
  {
    "id": "S3-R-09",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 380,
    "class": "regime",
    "severity": "S3",
    "claim": "densPL_alpha's VALUE is never validated -- _validate_params only checks that the key exists. Three consequences: alpha = -3 divides by (3.0 + alpha) and raises a bare ZeroDivisionError in the mass check; alpha outside the physically allowed [-2, 0] is accepted; and the whole nEdge < nISM correction block's inequality directions (rCore_min as a LOWER bound, 'rCore_min < rCloud' as the feasibility test) hold only for alpha < 0. Separately, the correction is gated on 'alpha != 0', but for alpha = 0 the profile is homogeneous so nEdge = nCore and nCore < nISM makes the condition reachable with no correction and no warning.",
    "evidence": "A: get_InitCloudProp.py:380-409 validates presence only for densPL_alpha; :263-267 divides by (3.0+alpha); :188/:190 rCore_min and its test; :180 the 'nEdge < nISM and alpha != 0' gate. nEdge = nCore*(rCloud/rCore)^alpha is increasing in rCore iff alpha < 0. B: S3-B-06 (the 'only possible for alpha != 0' comment is false at alpha = 0), S3-B-28 ('increasing nCore shrinks rCloud, which only helps' holds for alpha < 0 only). C: S3-C-19 requires -2 <= densPL_alpha <= 0 enforced, per SPEC-003.",
    "expected": "Validate -2 <= densPL_alpha <= 0 with the offending value in the message (which makes the alpha = -3 crash unreachable and the alpha > 0 mislabelling moot), and either handle or reject alpha = 0 with nCore < nISM.",
    "failure_scenario": "A sweep walking alpha through -3 crashes with a bare ZeroDivisionError from a mass sanity check with no indication alpha is the culprit; a homogeneous config with nCore < nISM builds a cloud less dense than its own ambient medium and skips the correction entirely.",
    "repro": "Set densPL_alpha = -3 in a densPL .param and call get_InitCloudProp; separately run densPL_alpha = 0 with nCore below nISM and check whether any warning fires.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S3-A-07", "S3-A-21", "S3-B-06", "S3-B-28", "S3-C-19"]
  },
  {
    "id": "S3-R-10",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 447,
    "class": "numerical",
    "severity": "S3",
    "claim": "The radius grid contains a single interval spanning seven decades, from an isolated first point at 1e-10 pc straight to r_min = 1e-3 pc, with nothing in between. Any interpolation of n_arr or M_arr inside that gap is grossly wrong (for a constant-density core M ~ r^3, so linear interpolation at 1e-4 pc over-estimates enclosed mass by ~100x). Both 1e-10 and 1e-3 are hardcoded and independent of rCore, rCloud and r0. If rCloud < 1e-3 pc the 'inside' logspace is monotonically DECREASING and the later sort masks the inversion.",
    "evidence": "A: get_InitCloudProp.py:437 r_min = 1e-3; :440 logspace starts at r_min; :447 first array element is the literal 1e-10; :443 outer extent 1.5*rCloud; A verified that for rCloud = 5e-4 the r_inside array runs 1e-3 down to 5e-4 and np.all(np.diff > 0) is False. r0 ~ 5e-2 pc for a representative config but scales as pdot^(3/2)/(sqrt(rho)*L), so weak-feedback / high-density configs can push r0 below 1e-3 pc. C: S3-C-18 makes the same resolution demand -- but from the WRONG premise, assuming a LINEAR 1000-point grid with dr = 0.02 pc; A shows the grid is logarithmic, so C's specific scenario (rCore inside the first cell) does not occur and C-18 is demoted from S2 accordingly.",
    "expected": "Scale the innermost point off min(rCore, r0) rather than a fixed 1e-10 pc, or validate that the first radius the solver requests lies above r_min.",
    "failure_scenario": "For a weak-wind or very dense configuration whose free-streaming radius falls below 1e-3 pc, the initial swept-up mass read off M_arr is over-estimated by orders of magnitude, biasing the trajectory from step zero. For a very compact cloud the grid is inverted before the sort and essentially unresolved after it.",
    "repro": "np.interp(1e-4, r_arr, M_arr) vs the analytic (4/3)*pi*r^3*rho_core for a densPL cloud; and _create_radius_array(5e-4, 1e-4), inspecting np.diff of the pre-sort r_inside.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S3-A-09", "S3-A-20", "S3-C-18"]
  },
  {
    "id": "S3-R-11",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 349,
    "class": "units",
    "severity": "S3",
    "claim": "params['densBE_sigma'] is stored in km/s (be_result.c_s / 1.0e5) while every other velocity in the codebase is AU (pc/Myr). The two differ by only 2.27%, so a consumer assuming AU is wrong by an amount no sanity check would catch. The key is also absent from the docstring's in-place-update list and its name never appears in the prose at all.",
    "evidence": "A: get_InitCloudProp.py:349 params['densBE_sigma'].value = be_result.c_s / 1.0e5; unit_conversions.py:109 v_kms2au = 1.022712165045695. The only unit hint anywhere is an info string at :471. B: S3-B-11, the key is missing from the :90 Notes list and unnamed in prose; B notes the error is either ~2% (silently plausible) or 1e5x (obvious). C: S3-C-15 requires the AU convention throughout and warns that a km/s vs pc/Myr confusion is 'small enough to be invisible in a plot and large enough to break an equivalence gate'.",
    "expected": "Store in AU (be_result.c_s * cvt.v_cms2au) like every other velocity, and document the key with its unit in the Notes list.",
    "failure_scenario": "A densBE run's velocity dispersion is consumed 2.27% low (or a pressure derived from it 4.6% low) everywhere it is used, shifting the BE sphere's derived quantities by a few percent with no error and no warning.",
    "repro": "grep for densBE_sigma consumers and check whether any arithmetic combines it with a pc/Myr quantity without a v_kms2au factor.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-01", "S3-B-11", "S3-C-15"]
  },
  {
    "id": "S3-R-12",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 347,
    "class": "regime",
    "severity": "S3",
    "claim": "The Bonnor-Ebert branch is presented as a hydrostatic equilibrium whose T_eff is a gas temperature, but T_eff is a recast of a support velocity dispersion, and back-solving the BE relations for the shipped example gives c_s ~ 10-20 km/s, i.e. T ~ 3e4-1e5 K -- three orders of magnitude above a real GMC and far above SPEC-066's T15 expectation of 10-30 K. Compounding this, be_result.is_stable is consumed only to build a log string and never reaches CloudProperties or params, and densBE_Omega has no validated range even though BE spheres are unstable above a critical contrast.",
    "evidence": "B (from prose): S3-B-12, :347-348 says sigma = c_s is 'the transparent physical quantity behind the effective densBE_Teff'; :52 labels T_eff 'Effective temperature [K]'; the mu used in the sigma<->T conversion is never stated. S3-B-29, no stated range for densBE_Omega or xi_out. C (from physics, independently): S3-C-21 back-solves r_0 = c_s/sqrt(4*pi*G*rho_c) and M = 4*pi*rho_c*r_0^3*xi^2*psi' for param/cloud_example_BE.param (1e6 Msun, 1e4 cm^-3, Omega 14.1) and gets c_s = 10.6-19.8 km/s, T = 3.2e4-1.1e5 K, robust on dimensional grounds because c_s^2 ~ GM/R ~ 269 (km/s)^2; also notes Omega = 14.1 > 14.04 is formally unstable. A supplies the mechanism: S3-A-24, is_stable is used only in a logger.info string; CloudProperties carries T_eff and xi_out but not stability.",
    "expected": "Compute and log the implied c_s / T, with a note that the BE profile stands in for turbulent support rather than thermal hydrostatic equilibrium; validate densBE_Omega against the critical contrast; carry is_stable onto CloudProperties so output metadata records it.",
    "failure_scenario": "Any downstream physics that assumes the cloud is thermally supported, or that reads a cloud temperature, is inconsistent with the profile actually built. A sweep over densBE_Omega crossing the stability boundary produces output files with no way to tell which points started from a gravitationally unstable cloud.",
    "repro": "python run.py param/cloud_example_BE.param --dry-run; back out c_s from the reported r_0 and rCloud, and check numerically whether k_B*T_eff/(mu*m_H) = sigma^2 for mu = 1.4, 2.3, 0.6.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S3-B-12", "S3-B-29", "S3-C-21", "S3-A-24"]
  },
  {
    "id": "S3-R-13",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 188,
    "class": "numerical",
    "severity": "S3",
    "claim": "Both nEdge correction formulas degenerate as alpha -> 0: rCore_min = rCloud*(nCore/nISM)**(1.0/alpha) has 1/alpha in the exponent, and nCore_min = nISM*(rCloud/rCore)**(-alpha) collapses to nISM. The only guard is the exact test alpha != 0, which does nothing for small |alpha|.",
    "evidence": "A: get_InitCloudProp.py:188 and :219, with the :180 gate 'nEdge < nISM and alpha != 0'. Both are exact inversions of nEdge = nISM (A verified) -- the formulas are right, their conditioning is not. B: S3-B-05, the parenthetical 'only possible for alpha != 0' at :179 is a claim about when the condition arises, not a numerical guard.",
    "expected": "An explicit |alpha| > eps guard, or a branch for near-zero alpha, before either correction is evaluated.",
    "failure_scenario": "alpha = -1e-6 (a nearly-homogeneous cloud, reachable in a sweep) makes (nCore/nISM)**(1/alpha) underflow to 0 or overflow to inf, giving rCore_min = 0 or inf and a downstream divide-by-zero or a nonsensical cloud.",
    "repro": "Call the power-law branch with alpha = 1e-8 and nEdge < nISM and inspect rCore_min and nCore_min.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-B-05", "S3-A-21"]
  },
  {
    "id": "S3-R-14",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 485,
    "class": "deadcode",
    "severity": "S3",
    "claim": "verify_mass_at_rCloud and verify_key_radii_in_array -- the more complete checks, since they test the actual tabulated M_arr rather than a re-derived analytic mass -- are called only from the module's __main__ block, never from get_InitCloudProp. The production path uses a separate inline check with a different threshold. verify_key_radii_in_array also logs that the radii are 'exactly' in the array while testing with np.isclose (rtol 1e-5), and verify_mass_at_rCloud divides by mCloud with no zero guard.",
    "evidence": "A: S3-A-19, the only call sites are get_InitCloudProp.py:587/588/614/615/645/646, all inside if __name__ == '__main__'; the production check is the inline one at :268 with 1e-3 vs :513's 0.01. A: S3-A-10 and B: S3-B-10 both flag the exactness question -- A found that because rCloud is re-derived through 10**log10(rCloud) inside logspace and also appended verbatim, the grid can contain pairs of radii ~1 ulp apart (reproduced: rCloud = 19.87 yields both 19.869999999999997 and 19.87).",
    "expected": "Call the verifiers from get_InitCloudProp (they are the stronger checks), with one threshold, and drop the word 'exactly' from a message backed by np.isclose. A cannot see other modules, so they may be called elsewhere -- confirm before treating as dead.",
    "failure_scenario": "A genuine mass-profile error in the tabulated arrays passes unnoticed because the production path only checks a re-derived analytic mass; the array-based check that would catch it never runs.",
    "repro": "grep -rn 'verify_mass_at_rCloud\\|verify_key_radii_in_array' trinity/ test/",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-19", "S3-A-10", "S3-B-10"]
  },
  {
    "id": "S3-R-15",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 386,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "_validate_params enforces positivity for mCloud, nCore and rCore but not for nISM or mu_convert, although both are in the required list and both are used in division and in a square root. A negative mu_convert gives a negative rhoa and dt_phase0 = sqrt(negative) = nan, which propagates into t0, r0 and E0 with no error at all.",
    "evidence": "A: get_InitCloudProp.py:386-399 (positivity block covers three keys); :188 divides by nISM inside (nCore/nISM)**(1.0/alpha); :259 mu_convert forms rhoCore; get_InitPhaseParam.py:146 forms rhoa, which :151 square-roots. nISM = 0 is currently masked only because 'nEdge < nISM' is then false. C: S3-C-19 lists the fuller validation set the module should enforce.",
    "expected": "Add nISM > 0 and mu_convert > 0 to the same positivity block.",
    "failure_scenario": "A NaN initial state propagates silently through every returned quantity; nothing raises and nothing in the output distinguishes it until a downstream solver fails for an unrelated-looking reason.",
    "repro": "Call get_InitCloudProp with nISM = 0 and separately with a negative mu_convert; inspect the returned CloudProperties and the get_y0 tuple.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S3-A-22", "S3-C-19"]
  },
  {
    "id": "S3-R-16",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 342,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "_ensure_be_params_exist is a second source of BE defaults living outside the schema ('should normally be created by read_param.py, but this provides a safety fallback'), and it runs AFTER params['densBE_Teff'].value = T_eff has already been written -- yet densBE_Teff is neither in _validate_params' required list nor among the keys _ensure_be_params_exist creates. A params dict lacking it dies with a bare KeyError instead of the clear ValueError validation would have given. Separately, the three initial_cloud_* writes in get_InitCloudProp are each guarded by 'if key in params', so absent keys are silently skipped with no warning, leaving any stale prior value in place.",
    "evidence": "A: get_InitCloudProp.py:342 (write) precedes :345 (_ensure_be_params_exist); the validation densBE branch at :405-409 requires only densBE_Omega and gamma_adia; _ensure_be_params_exist at :467-472 covers densBE_f_m, densBE_xi_out, densBE_f_rho_rhoc, densBE_sigma -- not densBE_Teff. The module's own __main__ BE test at :622-637 has to supply densBE_Teff explicitly for exactly this reason. A: S3-A-18 for the guarded initial_cloud_* writes at :135-140. B: S3-B-26, the fallback duplicates schema defaults and can drift from read_param.py.",
    "expected": "One source of defaults (the schema in trinity/_input/), per the project convention. If the fallback stays, add densBE_Teff to it or move the call above line 339. Either guarantee the initial_cloud_* keys in the schema and drop the guards, or warn when they are absent.",
    "failure_scenario": "The fallback defaults drift from the schema and a BE run silently uses a different value for a key the schema also defines; or a params set missing initial_cloud_n_arr leaves whatever was there before, so downstream code reads the previous run's cloud.",
    "repro": "Run _init_bonnor_ebert_cloud with a params dict omitting 'densBE_Teff'; and compare every default _ensure_be_params_exist supplies against trinity/_input/ (default.param / schema).",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-16", "S3-A-18", "S3-B-26"]
  },
  {
    "id": "S3-R-17",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 89,
    "class": "state",
    "severity": "S3",
    "claim": "Which cloud mass normalises the density profile -- the total mCloud or the post-SFE (1-sfe)*mCloud -- is not resolved, and verify_mass_at_rCloud takes mCloud as an ARGUMENT, so two callers can choose differently.",
    "evidence": "C: S3-C-20, SPEC-005 records this as unresolved; at sfe = 0.3 (param/simple_cluster.param) the two readings differ by 0.7^(1/3) = 0.888 in rCloud (11%) and 30% in swept mass at fixed radius; paper/paper_densityProfile.py's _DEFAULTS uses mCloud*(1-0.01) with the comment 'post-SFE', which is evidence for the post-SFE reading but is a figure script, not the solver. A and B are both silent -- A's required-keys list for get_InitCloudProp does not include sfe, so the choice is made outside this slice.",
    "expected": "One documented convention used by the profile normalisation, the rCloud root, M_sh(r) and F_grav alike. If total mCloud normalises the profile but (1-sfe)*mCloud is the sweepable gas, the difference must be handled explicitly.",
    "failure_scenario": "An 11% rCloud error is a 30% swept-mass error at fixed radius, propagating directly into the gravity term and the shell equation of motion -- well above any integrator tolerance and invisible to a dimensionless test.",
    "repro": "Run param/simple_cluster.param (sfe 0.3) and check rCloud against (3*mCloud/(4*pi*rho_core))^(1/3) and against (3*0.7*mCloud/(4*pi*rho_core))^(1/3).",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S3-C-20"]
  },
  {
    "id": "S3-R-18",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 32,
    "class": "citation",
    "severity": "S4",
    "claim": "WEAVER_TEMP_COEFFICIENT = 1.51e6 K is documented and coded identically and cannot be adjudicated by this audit: Lens C explicitly DECLINED to derive it, offering only recalled literature values (1.51e6 or 2.07e6) and its own closure derivation of 1.78-1.82e6 at low confidence. The coded value falls inside C's own acceptance range [1.4e6, 2.2e6] K, so there is no evidence of error -- but no independent confirmation either.",
    "evidence": "A: get_InitPhaseParam.py:32 WEAVER_TEMP_COEFFICIENT = 1.51e6, used bare as the multiplicative prefactor at :172. B: 1.51e6 K at :30 and :171, cited to Weaver+77 Eq. 37 at both sites; B notes no coefficient in the slice is quoted two different ways. C: section 8 confidence ledger, 'Recalled, not derivable here, low confidence: the literature T_b prefactor (1.51e6 vs 2.07e6) ... I assert none.' C's separate trap 11 notes the widely-quoted (n_b, T_b) = (4.02e-3 cm^-3, 1.51e6 K) pair is isobarically inconsistent with the dynamical P_b/k_B by a factor 4.2, so at most one of the two literature prefactors can be right.",
    "expected": "A human with journal access confirms the Weaver+77 Eq. 37 prefactor and the mu / density convention it assumes. Until then this is marked PENDING LITERATURE and must NOT be reported as an A=B-disagrees-with-C signature defect.",
    "failure_scenario": "If the true prefactor is 2.07e6, T0 is 27% low from the first step; the -6/35 exponent means the error never grows loud enough to notice.",
    "repro": "Compute T0 in snapshot 0 and divide out (L36)^(8/35)*(n0)^(2/35)*(t6)^(-6/35)*(1-xi)^0.4 to recover the implied prefactor, then check it against the paper.",
    "confidence": "low",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "contested",
    "source_ids": ["S3-C-11", "S3-C-22(coeff table)", "S3-B-15"]
  },
  {
    "id": "S3-R-19",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 27,
    "class": "citation",
    "severity": "S4",
    "claim": "The slice cites Weaver+77 Eq. 20, Weaver+77 Eq. 37 and Rahner thesis Eq. 1.15 by equation NUMBER. None can be verified from this container, and the Rahner thesis renumbers independently of the 1977 paper, so a number correct in one is wrong in the other.",
    "evidence": "B: citations transcribed verbatim at get_InitPhaseParam.py:27, :30, :45, :166, :170. B's own hygiene audit is a POSITIVE result: no citation is attached to two different formulas, and every coefficient appears identically at both of its sites. C: S3-C-24 refuses to assert any equation number and recommends citing the relation instead. The formulas those numbers label are independently confirmed correct here (table rows 1, 3, 7-10).",
    "expected": "Cite by content -- 'Weaver et al. 1977, energy-driven self-similar solution' plus the formula -- rather than by an unverifiable equation number. This also disambiguates Weaver from Castor/McCray, which are different solutions with different radius coefficients.",
    "failure_scenario": "A wrong equation number is inherited by every downstream doc and cannot be checked without journal access; it also hides whether a coefficient came from Weaver or from a different solution.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S3-C-24", "S3-B(sec2)"]
  },
  {
    "id": "S3-R-20",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 453,
    "class": "numerical",
    "severity": "S4",
    "claim": "Because rCloud is re-derived through 10**log10(rCloud) inside logspace and also appended verbatim, the radius grid can contain pairs of radii separated by ~1 ulp, right at the cloud edge -- the most physically important radius in the array. np.sort at line 453 is additionally redundant, since np.unique already returns sorted output.",
    "evidence": "A: get_InitCloudProp.py:440, 443, 453; reproduced numerically -- rCloud = 19.87 gives both 19.869999999999997 and 19.87 (relative gap 1.79e-16), rCloud = 23.456789 gives a 1.52e-16 pair; rCloud = 100.0 or 5.0 produce no duplicate, so the defect is value-dependent.",
    "expected": "np.unique with a relative tolerance, or build the grid so rCloud is inserted exactly once.",
    "failure_scenario": "A consumer forming a finite difference across that pair divides by ~3.5e-15 pc; a consumer assuming a well-separated monotone grid sees a near-degenerate cell at the cloud edge.",
    "repro": "python -c \"import numpy as np; r=np.sort(np.unique(np.append(np.concatenate([[1e-10],np.logspace(-3,np.log10(19.87),1000),np.logspace(np.log10(19.87),np.log10(1.5*19.87),100)]),[1.0,19.87]))); d=np.diff(r)/r[:-1]; print(d.min())\"",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-A-10", "S3-B-10"]
  },
  {
    "id": "S3-R-21",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 546,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The module carries a ~120-line __main__ self-test block (Test 1: alpha=-2, Test 2: alpha=0, Test 3: BE, plus a MockParam double) that pytest never runs -- and it is unit-inconsistent with the module it demonstrates: it passes mu_convert = 1.4 and nCore = 1e3 as raw cgs-flavoured numbers rather than AU (mu_convert should be ~1.18e-57 Msun, nCore ~2.9e58 pc^-3) and prints the AU nEdge labelled 'cm^-3' without the ndens_au2cgs factor the production log applies.",
    "evidence": "A: S3-A-25, get_InitCloudProp.py:571-574 vs unit_conversions.py:375 and :88; :585 print(f'  nEdge = {props_PL.nEdge:.2e} cm^-3') vs the production :290 which applies cvt.ndens_au2cgs. B: S3-B-25, the block spans :545-661 with 'Check all tests passed' at :661. C: S3-C-26, MockParam at L558 is a test double in a production module's file. The three named cases (alpha=-2, alpha=0, BE) are exactly the coverage the pytest suite should hold.",
    "expected": "Move the three cases into test/test_*.py with AU values, per the project convention that checks live in the pytest suite. Flag, do not delete (CLAUDE.md rule 3 -- this is pre-existing).",
    "failure_scenario": "Someone reads the __main__ block as the worked example of how to call get_InitCloudProp and passes cgs values from real code. As written, 'All tests PASSED!' certifies only that the mass integral is self-consistent in whatever units it was handed.",
    "repro": "python -m trinity.phase0_init.get_InitCloudProp; grep test/ for equivalent cases.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S3-A-25", "S3-B-25", "S3-C-26"]
  },
  {
    "id": "S3-R-22",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 220,
    "class": "other",
    "severity": "S4",
    "claim": "Diagnostic messages are inconsistent or wrong. The nCore-fix warning always states the reason is 'cannot fix with rCore alone (rCore_min=... >= rCloud=...)', but that path is also reached from the branch where rCore_min < rCloud held and the fall-through was caused by rCore_try >= rCloud_try -- so the message asserts a condition the two printed numbers contradict. In get_InitPhaseParam the two adjacent clamp warnings disagree about units: Lmech_W is converted to cgs and labelled erg/s, while pdot_W and v0 are printed raw in AU with no unit and no format spec.",
    "evidence": "A: S3-A-12, get_InitCloudProp.py:190 -> :197 -> :210 all reach the shared warning at :220-226. A: S3-A-23, get_InitPhaseParam.py:116 f'Lmech_W={Lmech_W*cvt.L_au2cgs:.3e} erg/s' vs :120 f'pdot_W={pdot_W} is very small' and :137 f'v0={v0} is very small'.",
    "expected": "Two distinct messages naming which test failed; and pdot_W*cvt.pdot_au2cgs labelled 'dyne', v0*cvt.v_au2kms labelled 'km/s', matching the neighbouring line.",
    "failure_scenario": "A user debugging an unexpected nCore change is told rCore_min >= rCloud while the printed numbers show the opposite, and looks in the wrong place.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S3-A-12", "S3-A-23"]
  },
  {
    "id": "S3-R-23",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 24,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Pre-existing dead code, flagged not deleted (CLAUDE.md rule 3): unused 'import os' and 'from pathlib import Path'; logging re-imported inside __main__ although already imported at module level; params['rCore'].value = rCore in the BE path writes back exactly the value read 22 lines earlier (a no-op that reads as if rCore were a BE output); the ternary 'if mCloud > 0 else 0' whose else arm is unreachable because _validate_params already raises for mCloud <= 0; and 'nEdge = nCore*(rCloud/rCore)**alpha if alpha != 0 else nCore', whose two arms are numerically identical since x**0 == 1 and rCore is validated positive.",
    "evidence": "A: S3-A-17 (:24, :28, :550 -- zero occurrences of 'os.' or 'Path('), S3-A-15 (:318 read, :340 write-back, nothing assigns rCore between), S3-A-13 (:268 vs :394-395), S3-A-14 (:177 vs :398-399).",
    "expected": "Remove in a separate housekeeping change, not as a side effect of a physics fix.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S3-A-13", "S3-A-14", "S3-A-15", "S3-A-17"]
  },
  {
    "id": "S3-R-24",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 167,
    "class": "coefficient",
    "severity": "S4",
    "claim": "E0 = (5/11)*L_w(tSF)*dt_phase0 uses the INSTANTANEOUS mechanical luminosity, whereas the similarity solution it comes from assumes L_w has been constant since t = 0. The defensible generalisation is E0 = (5/11) * integral from 0 to dt of L_w dt'.",
    "evidence": "C: S3-C-23; E proportional to t follows from constant L_w. SB99 L_mech is roughly flat for t < 3 Myr and dt_phase0 ~ 1e-5 Myr, so the two agree to well under a percent for the bundled table. A confirms the code uses the instantaneous sps_f['fLmech_W'](tSF) at get_InitPhaseParam.py:111, :167.",
    "expected": "Either the integral form, or a comment recording that the instantaneous value is used because dt_phase0 is far inside the flat part of the SB99 wind curve.",
    "failure_scenario": "Negligible for the bundled table; a user-supplied SPS table with a steep leading edge would seed a systematically wrong E0 with no diagnostic.",
    "repro": "",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S3-C-23"]
  },
  {
    "id": "S3-R-25",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 176,
    "class": "numerical",
    "severity": "S4",
    "claim": "bubble_xi_Tb is validated to the CLOSED interval [0, 1], so xi == 1.0 exactly passes and gives 0.0**0.4 = 0.0, i.e. T0 = 0 K, with no guard. Separately, the radius array is truncated at 1.5*rCloud, an undocumented ceiling that any consumer interpolating past it silently extrapolates through.",
    "evidence": "A: get_InitPhaseParam.py:100 'not (0 <= bubble_xi_Tb <= 1)' raises, so the interval is closed; :176 (1.0 - bubble_xi_Tb)**0.4. This REFUTES B's S3-B-21 claim that the range is unvalidated and that xi > 1 gives NaN -- only the xi == 1 endpoint survives. A: get_InitCloudProp.py:443, outer extent 1.5*rCloud; B: S3-B-23, the ceiling and the inner cutoff are undocumented.",
    "expected": "Validate 0 <= bubble_xi_Tb < 1 (half-open), and document the 1.5*rCloud ceiling plus the inner cutoff in the _create_radius_array docstring.",
    "failure_scenario": "bubble_xi_Tb = 1.0 seeds a zero initial bubble temperature straight into the cooling tables. Once the shell exceeds 1.5*rCloud, a density or mass lookup extrapolates off the end of the array with no error.",
    "repro": "Call get_y0 with bubble_xi_Tb = 1.0 and inspect T0; check r_arr[-1]/rCloud for both profiles and find every consumer of initial_cloud_r_arr.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S3-B-21", "S3-B-23", "S3-A(sec1.5)"]
  },
  {
    "id": "S3-R-26",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 172,
    "class": "regime",
    "severity": "S4",
    "claim": "PARKED / NEEDS ONE LOOKUP. Lens C claims the seed T0 exceeds the shocked-wind ceiling T_max = 3*mu*m_H*v_w^2/(16*k_B) = 5.53e7 K by 2.3x. Reconciliation shows this holds only for small bubble_xi_Tb: both C's 1.29e8 K and A's 6.7e7 K probe are CENTRAL Weaver temperatures with the (1-xi)^0.4 factor at ~1, whereas the code applies that factor, and at the SPEC-040 convention xi = 0.98 it is 0.2091, giving T0 ~ 2.7e7 K -- comfortably BELOW the ceiling.",
    "evidence": "C: S3-C-13, T_max = 5.53e7 K at v_w = 2000 km/s, mu = 0.609; Weaver law at t_fs = 2.53e-6 Myr gives 1.29e8 K with prefactor 1.51e6. A: numeric probe gives T0 = 6.7e7 K at nCore = 1e3, dt = 2.53e-5 Myr, same v_w. RECONCILER ARITHMETIC on those two lens numbers: 6.7e7 * (1e5/1e3)^(2/35) * 10^(6/35) = 1.294e8, an exact match to C -- proving both used (1-xi)^0.4 ~ 1. Applying the coded (1-0.98)^0.4 = 0.2091 to C's value gives 2.70e7 K < 5.53e7 K. Neither A nor B states the shipped default for bubble_xi_Tb, so the conclusion rests on C's SPEC-040 reading of 0.98.",
    "expected": "Confirm the default bubble_xi_Tb in trinity/_input/default.param. If it is 0.98 the ceiling is not violated and C-13 should be closed; if it is near 0 the finding is live at S2 and an explicit T0 <= T_max assertion is warranted. Also parked here, for whichever slice owns the ODE and bubble structure: C's P_b = E_b/(2*pi*(R2^3 - R1^3)) with the 2*pi being (gamma-1)*(4*pi/3) inverted rather than a solid angle (S3-C-25), R1 = sqrt(pdot/(4*pi*P_b)) (S3-C-13 chain), and the Weaver radius coefficient xi_E = 0.7628653 vs PHYSICS_SPEC.md SPEC-050's 0.762934 -- none of which is computed in the S3 slice, so none can be adjudicated here.",
    "failure_scenario": "If the default xi is small, the bubble is seeded above the shocked-wind temperature and the cooling table is queried above its top edge (Gnat-Ferland stops at 1e8 K), producing an extrapolated Lambda in the very first transition-trigger evaluation.",
    "repro": "grep bubble_xi_Tb trinity/_input/default.param; then compare T0 in snapshot 0 against (3/16)*mu_ion*m_H*v_w^2/k_B with v_w = 2*Lmech_W/pdot_W from the same snapshot.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S3-C-13", "S3-C-14", "S3-C-05", "S3-C-25", "S3-A(sec1.4, sec1.5)"]
  }
]
```

---

## 7. Inputs retired (dropped or demoted, with reason)

| Lens finding | Reason retired |
|---|---|
| S3-B-14 (E0 uses `t0` not `dt`, ~1000× inflation) | **Refuted by A** — the code uses `dt_phase0` at :167. Only a docstring symbol inconsistency remains. |
| S3-B-19 (which L enters E0/T0) | **Refuted by A** — `Lmech_W` (wind-only) is used for all four expressions. |
| S3-B-22 (`Msun pc²/Myr³` annotates an energy) | **Refuted by A** — line 188 annotates `Lbol_tSF`, a luminosity. B itself conditioned the finding on this. |
| S3-B-16 (which density feeds T0) | **Refuted by A** — `nCore * ndens_au2cgs` gives cm⁻³, exactly as Eq. 37 requires. |
| S3-C-12 (missed AU→cgs on L, 7 orders in T0) | **Refuted by A** — `L_au2cgs` is applied before the `1e36` division. |
| S3-C-14 (seed must carry the 0.20913 ξ factor) | **Refuted by A** — the code already applies `(1 - bubble_xi_Tb)**0.4`. Folded into R-26 as the key to reconciling C-13. |
| S3-B-07 (unbounded halving loop) | **Refuted by A** — `range(50)`. The correct defect (bounded, no convergence check) is in R-08. |
| S3-B-21 (xi range unvalidated, NaN for xi>1) | **Refuted by A** — validated `[0,1]` at :100. Surviving `xi == 1` sliver demoted into R-25. |
| S3-C-18 (linear grid, dr = 0.02 pc, rCore in first cell) | **Premise refuted by A** — the grid is `np.logspace`. Demoted S2→S3 and merged into R-10 in its surviving form. |
| S3-C-09 (floors are silent, physically sized) | **Partly refuted by A** — they log at WARNING and `1e-100` is a pure div-zero magnitude. Surviving substance merged into R-03. |
| S3-C-05 (ξ_E must be 0.7628653) | **Not coded in this slice.** Retained only as the SPEC-050 note in §2 row 2 and parked in R-26. |
| S3-C-25, S3-C-13's R1 chain | **Out of slice** — neither `P_b` nor `R1` is computed in phase0_init. Parked in R-26 for the owning slice. |
| S3-C-01 (5/11 must be exactly 5/11) | **Verified correct** — A=B=C. Recorded in §5, not a finding. |
| S3-C-06 (R_fs prefactor), S3-C-19 partial, S3-C-15, S3-C-10, S3-C-17 (formulas) | **Verified correct** — each matched by A's transcription. Recorded in §5; only the unmet *validation* half of C-19 survives, in R-09 and R-15. |
| S3-A-19's uncertainty about other call sites | Retained at medium confidence inside R-14 with an explicit "confirm before treating as dead". |
