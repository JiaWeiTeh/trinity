# S1 units & helpers — reconciled

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

**Reconciler input:** `S1_units_helpers_lensA.md` (code behaviour, constants recomputed from the
code's own base scales), `S1_units_helpers_lensB.md` (comments/docstrings only),
`S1_units_helpers_lensC.md` (required values derived from constant *names*, values redacted).
No source was read. All three inputs are raw and unverified; everything below is a cross-lens
reconciliation, not an independent verification of the code.

---

## 0. Headline

**The constants are clean.** Lens A recomputed 22 conversion constants + 2 derived module-level
constants + 9 CGS primitives from the code; Lens C derived the required value of each from its name
alone. **30 of the 31 constants C covered agree between the two independent derivations.** There is
**no S1 finding in this slice** — no stored constant disagrees with the physically correct value,
and every one of C's relational invariants that tests arithmetic passes.

The single non-agreement (`tau_cgs2au`) is a **naming** defect, not an arithmetic one: the stored
number is exactly right for the dimension Lens B's docstring declares (surface mass density), and
wrong only for the two quantities the *name* `tau` suggests.

**C's largest stated uncertainty is resolved by A's data.** C could not know which M☉ convention the
code uses and tabulated both (legacy `1.98892e33` vs IAU-2015 nominal `1.9884099e33`, 2.6e-4 apart).
A's recomputation shows the code carries `1.988409870698051e33 g` — **IAU 2015 nominal**, i.e. C's
column B — and that *every* mass-bearing conversion is derivable from that one value to ≤1 ULP.
C's §4.4 "recover M☉ three different ways and demand agreement" test therefore **passes**, which
retires C's own S1-C-02 (mixed-convention constants). C's column B is the applicable column
throughout the table below.

Where the real risk in this slice sits: **the helpers, not the constants.** Three lenses
independently converge on the `find_nearest_*` family (silent out-of-range clamp, unbounded
single-step monotonicity exemption producing a wrong interior bracket) and on `_peak_prominences`
returning 0 for plateau-leading extrema. Those are the items to verify first.

---

## 1. Merged constants table

Columns: **stored** = the literal in the code as read by A · **A recomputed** = A's first-principles
value from the code's own pc/Myr/M☉ · **C required** = C's derivation from the name alone
(column B, IAU M☉ — see §0) · **verdict**.

`agree` = the two independent derivations match to floating-point precision.
`agree (rounding)` = they match to the digits stored; the stored literal is a truncation of the
full-precision value, not a different number.

### 1a. Conversion constants (`ConversionConstants`)

| # | constant | stored (A) | A recomputed | C required | verdict |
|---|---|---|---|---|---|
| 1 | `cm2pc` | 3.240779289444365e-19 | 3.240779289444365e-19 | 3.2407792894e-19 | **agree** |
| 2 | `km2pc` | 3.240779289444365e-14 | 3.240779289444365e-14 | *(not covered)* — derivable `1e5·cm2pc` ⇒ 3.2407792894e-14 | **agree** (C silent) |
| 3 | `s2Myr` | 3.168808781402895e-14 | 3.168808781402895e-14 | 3.1688087814e-14 | **agree** |
| 4 | `g2Msun` | 5.029144215870041e-34 | 5.029144215870041e-34 | 5.0291442159e-34 | **agree** (fixes M☉ = IAU nominal) |
| 5 | `ndens_cgs2au` | 2.937998946096347e+55 | 2.9379989460963475e+55 | 2.9379989461e+55 | **agree** |
| 6 | `phi_cgs2au` | 3.0047272630641653e+50 | 3.0047272630641657e+50 | 3.0047272631e+50 | **agree** |
| 7 | `E_cgs2au` | 5.260183968837699e-44 | 5.260183968837697e-44 | 5.2601839688e-44 | **agree** |
| 8 | `L_cgs2au` | 1.6599878161499254e-30 | 1.6599878161499253e-30 | 1.6599878161e-30 | **agree** |
| 9 | `pdot_cgs2au` | 1.623123174716277e-25 | 1.6231231747162772e-25 | 1.6231231747e-25 | **agree** |
| 10 | `pdotdot_cgs2au` | 5.122187189842638e-12 | 5.122187189842638e-12 | 5.1221871898e-12 | **agree** |
| 11 | `G_cgs2au` | 67400.3588611473 | 67400.35886114725 | 6.7400358861e+04 | **agree** |
| 12 | `v_kms2au` | 1.022712165045695 | 1.022712165045695 | 1.0227121650 | **agree** |
| 13 | `v_cms2au` | 1.022712165045695e-05 | 1.0227121650456949e-05 | 1.0227121650e-05 | **agree** |
| 14 | `F_cgs2au` | 1.623123174716277e-25 | 1.6231231747162772e-25 | 1.6231231747e-25 (`== pdot_cgs2au`) | **agree** + alias holds |
| 15 | `Pb_cgs2au` | 1545441495671.806 | 1545441495671.806 | 1.5454414957e+12 | **agree** |
| 16 | `k_B_cgs2au` | 5.260183968837699e-44 | 5.260183968837697e-44 | 5.2601839688e-44 (`== E_cgs2au`) | **agree** + alias holds |
| 17 | `c_therm_cgs2au` | 5.122187189842638e-12 | 5.122187189842638e-12 | 5.1221871898e-12 (`== pdotdot_cgs2au`) | **agree** + alias holds |
| 18 | `dudt_cgs2au` | 4.877042454381257e+25 | 4.877042454381258e+25 | 4.8770424544e+25 | **agree** |
| 19 | `Lambda_cgs2au` | 5.650062667161655e-86 | 5.650062667161653e-86 | 5.6500626672e-86 | **agree** |
| 20 | `tau_cgs2au` | 4788.452460043275 | 4788.452460043276 (`= g2Msun·pc2cm²`, g cm⁻² → M☉ pc⁻²) | 3.1688087814e-14 (`≡ s2Myr`, timescale) **or** 1.0 (optical depth) | **DISAGREE** — see R-06 |
| 21 | `gravPhi_cgs2au` | 1.045940172532453e-10 | 1.0459401725324526e-10 | 1.0459401725e-10 | **agree** |
| 22 | `grav_force_m_cgs2au` | 322743414.19646025 | 322743414.19646025 | 3.2274341420e+08 | **agree** |

### 1b. Derived module-level constants

| # | constant | stored (A) | A recomputed | C required | verdict |
|---|---|---|---|---|---|
| 23 | `Pb_au2_KcmInv` | computed at source as `Pb_au2cgs / K_B_CGS` | ⇒ 4686.67 K cm⁻³ per AU pressure | 4.6866675551e+03 | **agree** (see note) |
| 24 | `Mdot_au2Msunyr` | 1e-6 | 1e-6 exactly | 1.0e-06 exactly | **agree** |

> **Note on #23.** A's prose quotes the value as "4.6879e-4…", which is internally inconsistent with
> A's own statement that the constant is computed as `Pb_au2cgs / K_B_CGS` from constants A verified
> — that quotient is 4686.67, not 4.6879e-4. `4.6879e3` is C's *legacy-M☉* value. I treat A's digits
> as a transcription slip in the report and the computed value as 4686.67, because (a) it follows
> arithmetically from constants both lenses cleared, and (b) Lens B independently reports the source
> comment as "≈ 4.6867e+03", which matches the IAU-M☉ value to the digits quoted. **Cheap check
> worth doing:** print the constant and confirm 4686.67, not 4687.87 — a legacy-M☉ value here while
> `g2Msun` is IAU would be the exact mixed-convention defect C-02 was written to catch.

### 1c. CGS physical constants (`PhysicalConstantsCGS`)

| # | constant | stored (A) | A recomputed / first-principles | C required | verdict |
|---|---|---|---|---|---|
| 25 | `G` | 6.67430e-8 | CODATA-2018 6.67430e-8 | 6.67430e-8 | **agree** (exact match) |
| 26 | `k_B` | 1.380649e-16 | SI-exact | 1.380649e-16, SI-exact | **agree** (exact match) |
| 27 | `c` | 2.99792458e10 | SI-exact | 2.99792458e10, exact | **agree** (exact match) |
| 28 | `h` | 6.62607015e-27 | SI-exact | 6.62607015e-27, exact | **agree** (exact match) |
| 29 | `m_p` | 1.67262192e-24 | 1.67262192369e-24 (rel −2.2e-9) | 1.67262192369e-24 | **agree (rounding)** — 9 s.f. truncation |
| 30 | `m_e` | 9.1093837e-28 | 9.1093837015e-28 (rel −1.6e-10) | 9.1093837015e-28 | **agree (rounding)** — 8 s.f. truncation |
| 31 | `sigma_SB` | 5.670374e-5 | 5.670374419e-5 (rel −7.4e-8) | 5.670374419e-5, `= 2π⁵k_B⁴/(15h³c²)` | **agree (rounding)** — 7 s.f.; *fails* C's proposed 1e-9 assertion bar, see R-16 |
| 32 | `m_H` | 1.6735575e-24 | 1.6735328e-24 (rel **+1.47e-5**) | band `[1.6735e-24, 1.6738e-24]`; `\|m_H/(m_p+m_e)−1\| < 1.2e-4` | **agree-in-band**, provenance unresolved — see R-15 |
| 33 | `e` | 4.80320425e-10 | 4.803204713e-10 (rel −9.6e-8, pre-2019 value) | *(not covered)* | A-only; rounding/stale-convention class, see R-16 |

**Score: 30 agree · 1 disagree (`tau_cgs2au`, naming) · 2 not covered by C (`km2pc`, `e`, both
A-verified).** No constant in this slice is numerically wrong.

**Is `m_H` a precision difference or an error?** It is neither cleanly. A's +1.47e-5 offset is far
larger than a rounding of any named convention (A checked and excluded m_p+m_e, 1.008/N_A,
1.00794/N_A, and 1.00782503 u), but it sits *inside* C's own acceptance band and passes C's proposed
`m_H/(m_p+m_e)` test. Verdict: **not an error, but an unattributable literal** — a provenance defect
(R-15), not a coefficient defect. Same class for `e`. `m_p`, `m_e`, `sigma_SB` are unambiguously
**precision/rounding differences**, not errors.

---

## 2. Relational invariants (C §4) — pass/fail using A's values

### 2a. Reciprocal pairs (C §4.1)

| invariant | verdict | evidence |
|---|---|---|
| `X_cgs2au · X_au2cgs == 1` for all 21 stems, `\|product−1\| < 1e-15` | **PASS** | A: `InverseConversionConstants` stores each inverse as the float `1/x`; 16 of 21 round-trip to exactly 1.0, 5 (`g2Msun`, `phi`, `pdot`, `F`, `grav_force_m`) land at `1 − 1.11e-16` = 1 ULP, well inside C's bar. C's "hand-typed pairs land at 1 ± 1e-6" failure mode does **not** apply — the inverses are computed, not typed. |

### 2b. Aliases that must be bit-identical (C §4.2)

| invariant | verdict | evidence |
|---|---|---|
| `F_cgs2au == pdot_cgs2au` | **PASS** | A: byte-identical literal. Independently required by C-05 and flagged by B-09 as unverifiable from prose. |
| `F_au2cgs == pdot_au2cgs` | **PASS** | follows: `1/x` of identical values |
| `k_B_cgs2au == E_cgs2au` | **PASS** | A: identical literal (correct — K→K = 1) |
| `k_B_au2cgs == E_au2cgs` | **PASS** | as above |
| `c_therm_cgs2au == pdotdot_cgs2au` | **PASS** | A: identical literal. C called this "the least obvious and therefore most diagnostic" identity — it holds. |
| `c_therm_au2cgs == pdotdot_au2cgs` | **PASS** | as above |
| `tau_cgs2au == s2Myr` | **FAIL** | 4788.45 ≠ 3.1688e-14. Explained by R-06: `tau` holds a surface-density conversion. Not an arithmetic defect. |
| `tau_au2cgs == Myr2s` | **FAIL** | same cause |

### 2c. Composition identities (C §4.3)

| invariant | verdict | evidence |
|---|---|---|
| `ndens_cgs2au == pc2cm**3` | **PASS** | A recomputed exactly |
| `phi_cgs2au == pc2cm**2 · Myr2s` | **PASS** | A: `convert2au('cm**-2*s**-1')` returns `phi_cgs2au` **exactly** |
| `E_au2cgs == Msun2g·pc2cm²/Myr2s²` | **PASS** | ≤1 ULP |
| `L_cgs2au == E_cgs2au · Myr2s` | **PASS at 1e-12** | rel −2.1e-16; **not** bit-exact (A) — matters only if a test asserts exact equality |
| `pdot_au2cgs == Msun2g·pc2cm/Myr2s²` | **PASS** | ≤1 ULP |
| `pdotdot_cgs2au == pdot_cgs2au · Myr2s` | **PASS** | ≤1 ULP |
| `Pb_au2cgs == pdot_au2cgs / pc2cm**2` | **PASS** | ≤1 ULP |
| `Pb_au2_KcmInv == Pb_au2cgs / K_B_CGS` | **PASS by construction** | A: that is literally the source expression at `:287` |
| `dudt_cgs2au == Pb_cgs2au · Myr2s` | **PASS at 1e-12** | rel −1.8e-16; not bit-exact |
| `Lambda_cgs2au == L_cgs2au / pc2cm**3` | **PASS** | ≤1 ULP |
| **`Lambda_cgs2au · ndens_cgs2au² == dudt_cgs2au`** (cooling closure) | **PASS** | Reconciler arithmetic on A's stored values: `5.650062667161655e-86 × (2.937998946096347e55)² = 4.87704245e25` vs stored `4.877042454381257e25` — agreement to the ~9 s.f. my hand arithmetic resolves. **This retires C-07 (S1).** The volumetric rate carries exactly two density factors, as SPEC-082 requires. |
| `G_cgs2au == Msun2g·Myr2s²/pc2cm³` | **PASS** | A recomputed; cross-checked against textbook `G = 4.300917e-3 pc M☉⁻¹ (km/s)²` → `4.4985021e-3` pc³M☉⁻¹Myr⁻², matching C's column-B `4.498502e-3` to 7 s.f. **Two fully independent routes agree.** |
| `gravPhi_cgs2au == v_cms2au**2` | **PASS** | rel 3.7e-16 |
| `grav_force_m_cgs2au == v_cms2au · Myr2s` | **PASS** | ≤1 ULP |
| `v_kms2au == 1e5 · v_cms2au` | **PASS at 1e-12** | rel −2.2e-16; not bit-exact |
| `Mdot_au2Msunyr == 1e-6` (exactly; **not** `1/Myr2s`) | **PASS exactly** | A: `1e-6` literal, and the code's Myr is exactly 1e6 Julian years, so it is self-consistent |

### 2d. Self-consistency of the primitives (C §4.4)

| invariant | verdict | evidence |
|---|---|---|
| **Three independent M☉ recoveries agree** (`E_au2cgs·Myr2s²/pc2cm²` = `G_cgs2au·pc2cm³/Myr2s²` = `pdot_au2cgs·Myr2s²/pc2cm`) | **PASS** | A: "every one of the 18 compound constants is exactly derivable from these three base scales to ≤1 ULP", recomputed against the code's own M☉/pc/Myr. **This is the test C rated S1 (C-02); it is clean.** No mixed-convention constant. |
| `M_H_CGS ≈ M_P_CGS + M_E_CGS` to 1.5e-5 | **PASS (marginal)** | stored ratio = 1 + 1.47e-5, right at C's stated bar and well inside C-15's 1.2e-4 test |
| `SIGMA_SB == 2π⁵k_B⁴/(15h³c²)` to **1e-9** rel | **FAIL at 1e-9 · PASS at 1e-6** | rel −7.4e-8: the stored literal is a 7-s.f. truncation. A test written to C's tolerance would fail on correct code — use 1e-7. |
| module-level `*_CGS` == `CGS` dataclass fields | **PASS** for the 8 re-exported names | A + B (B claim 1.6: flat names are re-exports of the frozen dataclass fields). `e` has no flat re-export — coverage gap, not a mismatch. **A's own §2b prose is self-contradictory here** (one sentence says `c/sigma_SB/h` are *not* re-exported, the parenthetical says they are); only the `e` claim is unambiguous, so only that is carried forward. |
| `INV_CONV` field-by-field == `1/CONV` | **PASS** | A §2d |
| `pc2cm` exact IAU (`(648000/π)·1.495978707e13`) | **PASS — exact** | A and C both evaluate 3.0856775814913673e18; C's staleness traps (`3.086e18`, `3.08e18`) do not apply |
| `Myr2s` exact Julian (`1e6·365.25·86400`) | **PASS — exact** | 3.15576e13 both lenses; C's 365-day trap does not apply |

### 2e. Parser identities (C §5.1)

| invariant | verdict | evidence |
|---|---|---|
| **`a/b/c` flattens left-associatively** (`a·b⁻¹·c⁻¹`, not `a/(b/c)`) | **PASS** | A: `convert2au('cm**3/g/s**2')` = 67400.35886114728 = `G_cgs2au` (1 ULP). This is exactly C's distinguishing 3-token test; **C-17 (S1) is retired.** |
| **unknown token raises, never silently returns 1.0** | **PASS** | A: `yr`, `Msun/yr`, `dyne`, `eV`, `kms` all raise `UnitConversionError`; no `except → 1.0`, no `dict.get(tok, 1.0)`. **C-16 (S1, C's rank-1 trap) is retired.** |
| composability `convert2au('A/B') == convert2au('A')/convert2au('B')` | **PASS** on every pair A tested | `erg*cm**3/s` → `Lambda_cgs2au` exactly; `g*cm**-2` → `tau_cgs2au` exactly; `cm**-2*s**-1` → `phi_cgs2au` exactly |
| any pure power of `K` contributes exactly 1.0, incl. fractional | **PASS** | A: `K**-1` → 1.0; `s**(-3/2)` and `cm**(1/2)` parse via `Fraction`, so `K**(7/2)` will not raise |
| `convert2au('K cm-3')` returns `ndens_cgs2au` (no folded `k_B`) | **PASS on semantics, FAIL on that syntax** | K contributes 1.0 and cm⁻³ contributes `pc2cm³`, so `K*cm**-3` gives `ndens_cgs2au`; the parser never folds in `k_B`. But the *space-separated* form C wrote raises (whitespace is stripped first, fusing tokens) — see R-17. |

**Invariant score: 28 pass · 2 fail-by-naming (`tau` aliases) · 1 fail-at-C's-stated-tolerance
(`sigma_SB`, a rounding artefact).** Every invariant that tests arithmetic passes.

---

## 3. Coverage

| area | Lens A | Lens B | Lens C | reconciled status |
|---|---|---|---|---|
| 22 conversion constants | full (recomputed each) | 21 documented units/directions | 21 derived from names | **3-lens; cleared** |
| 2 derived module constants | full | both documented + one value quoted | both derived | **3-lens; cleared** |
| 9 CGS primitives | full | citation block only | 8 derived (no `e`) | **cleared, 1 provenance gap** |
| reciprocal / alias / composition invariants | verified by recomputation | flagged `F`≡`pdot` unverifiable from prose | authored the invariant set | **3-lens; all arithmetic invariants pass** |
| `convert2au` parser | executed, incl. failure modes | contract + undocumented edges | required behaviour + reference values | **3-lens; core correct, syntax subset narrow** |
| `find_nearest` | executed (ties, empty, 2-D, scalar) | undocumented-contract flag | required semantics | **3-lens; behaviour correct, contract undocumented** |
| `find_nearest_lower` / `_higher` | executed, both directions, boundaries | quotes the code's own admission of breach | predicted the breach blind | **3-lens; defect confirmed (R-02)** |
| `_is_monotonic_or_tolerable` | executed, all branches | 3 contradictory prose rules | required tolerance bands | **3-lens; R-01, R-09** |
| `get_soundspeed` | executed, value-checked | formula + unit gap | independent coefficient derivation | **3-lens; formula cleared (see below)** |
| `simplify` curvature / RMQ | executed + brute-forced | raised 2 worries | raised 2 worries | **cleared — both worries refuted** |
| `_peak_prominences` | executed vs scipy | predicted the sentinel bug blind | required prominence ≥ 0 | **3-lens; defect confirmed (R-04)** |
| `_simplify` dedup / budget / order | executed | contract contradictions | required invariants | **3-lens; R-03** |
| `_simplify_error` | executed | contract | required behaviour | **partial — R² direction not established by any lens** |
| `cluster.py` | executed | precedence + oversubscription flag | required precedence | **3-lens; R-11** |
| `logging_setup.py` | executed end-to-end | full contract | required behaviour | **3-lens; R-05** |
| `extract_example_snapshots.py` | executed | phase names + selection rule | `PHASES` must exclude `collapse` | **cleared — C's concern refuted by B** |

**Independently cleared, worth recording as positive results** (each verified by ≥2 lenses that never
saw each other's work):

1. **The whole constant set is internally consistent on one M☉** — C's S1-rated mixed-convention risk.
2. **The cooling closure `Λ·n² = du/dt`** — C's S1-rated normalisation risk.
3. **`convert2au` raises on unknown units** — C's rank-1 silent-failure trap.
4. **`convert2au` flattens `a/b/c` left-associatively** — C's S1-rated 1e53 parser slip.
5. **`get_soundspeed` returns pc/Myr with the right coefficient.** C derived the M☉-independent
   `(k_B/m_H)_AU = 8.6289090e-3 pc² Myr⁻² K⁻¹` from names alone and called it "the cleanest single
   number to check in this whole slice". A executed the function and got 151.3805 km/s at
   T = 1e6 K, γ = 5/3, μ = 0.6 m_H. Substituting into C's formula:
   `sqrt(5/3 × 8.6289090e-3 × 1e6 / 0.6) = 154.82 pc/Myr × 0.9777922 = 151.38 km/s` — **exact match
   to A's 5 quoted digits, from two derivations with no shared input.** B-04's worry (the docstring
   never states whether `k_B`/`μ` are cgs or AU) is answered: A confirms the implementation converts
   AU → cgs internally and back, and the result is dimensionally exact. B-04 reduces to a docstring gap.
6. **Menger curvature is mathematically correct** — B-17's two blind worries (unsigned cross? factor
   of 2?) both refuted by A's algebra.
7. **Sparse-table RMQ is correct**, brute-forced by A over n ∈ {1,2,3,5,8,17,33}, covering the
   power-of-two window lengths C-34 warned would hide a convention mix.
8. **`Mdot_au2Msunyr == 1e-6` exactly**, and self-consistent with the code's Julian Myr.
9. **Inverses are computed as `1/x`, not hand-typed** — C's "single cheapest test, highest-frequency
   failure mode" passes on all 21 pairs.
10. **`PHASES` contains no `collapse`** — C-45 refuted by B's phase-name inventory.

---

## 4. Divergences

| # | item | A says | B says | C says | class | resolution |
|---|---|---|---|---|---|---|
| D1 | `tau_cgs2au` | surface mass density, g cm⁻² → M☉ pc⁻² | docstring `:130` = **"Surface density, g/cm² → Msun/pc²"** | name demands `s2Myr` (timescale) or 1.0 (optical depth) | **A = B, C differs** | **Naming defect only.** Code and comment agree; the identifier is wrong. C's invariant "fails" because C could only see the name. → R-06, S3. Not arithmetic. |
| D2 | single-step monotonicity exemption | unbounded in depth (`if end-start == 1: continue` precedes the rtol test) | prose states it 3 incompatible ways: `:91` "shallow **and** localized"; `:102`/`:132` "**any depth**"; `:147` "**sub-percent**" | intent: depth-bounded, `MAX_SPIKE_LEN` 1–3 | **A ≠ B (partial), A ≠ C** | Code matches `:102`/`:132`; `:91` and `:147` are **stale comments (S3 doc-drift)**. The behaviour itself is the defect → R-01, S2. |
| D3 | `MONOTONIC_RTOL` | `1e-2` | `:147` says "sub-percent" (1e-2 is 1%, not sub-percent) | must be 1e-12…1e-6; "anything ≥ 1e-4 is too loose to be a noise filter" | **A ≠ C**, A ≠ B (mildly) | Genuine disagreement on a tuning constant, but it is a **judgement bar**, not a settled physical constant → R-09, S3, medium confidence. |
| D4 | `_DEDUP_TOL_DEFAULT` | `1e-6`, relative to per-axis range | `1e-6`, "unless the input has more than ~10⁶ uniformly-sampled points" | must be relative (**✔ code is**) and 1e-12…1e-9 | **A = B on value; A ≠ C on magnitude** | C's relative-vs-absolute S2 worry is **refuted**. C's magnitude bar is **vindicated by A's demonstration**: the collapse trip-wire is exactly `n ≳ 1/dedup_tol` → R-03. |
| D5 | ANSI escapes in the log file | verified present in the persisted `.log` | `:122` claims "**log files plain text**" | must not emit colour codes to files | **ABC — code contradicts its own doc and the requirement** | Code is the defect → R-05, S3. |
| D6 | `DedupWarningFilter` lifetime | suppresses for the process lifetime, no count | "State is per-process, so it **resets every run/task** — no cross-run leakage" | must not persist across runs in a sweep worker | **A ≠ B; C sides with the risk** | **Contested.** "Per-process" ≠ "per run/task" if a worker executes many parameter sets. Whether `setup_logging` is re-called per run is outside S1 → R-14, hand-off. |
| D7 | `find_nearest_higher` boundary comment | branch returns an index *below* `value` | comment is a **verbatim copy** of `find_nearest_lower`'s, saying "returned idx is actually higher than the value instead of the desired lower" | mirror-image postcondition required | **A ≠ B (the comment cannot describe this branch)** | B's blind prediction ("if the index arithmetic was copied too…") is **confirmed by A**. Doc-drift folded into R-02 and R-22. |
| D8 | `Msun/yr`, `yr` unit | `yr` is **not in the vocabulary**; `Msun/yr` raises | — | reference list includes `"Msun/yr" → 1.0e+06` | **A ≠ C** | Vocabulary gap: `Mdot_au2Msunyr` exists but no `.param` can express M☉/yr → R-17, S4. Fails **loudly**. |
| D9 | `K cm-3`, `erg/(s*cm**2)`, `km s**-1` | all raise (whitespace stripped; parens only around exponents) | edge cases undocumented (`''`, grouped denominators, repeated units) | expects these forms to parse | **A ≠ C on syntax, A ≠ B on documentation** | Accepted syntax is a narrow `*`/`**`-only subset. All rejections are **loud** → R-17, S4. |
| D10 | `sigma_SB` | 5.670374e-5, rel −7.4e-8 vs formula | cited "CODATA 2018 / IAU 2015" | must equal `2π⁵k_B⁴/(15h³c²)` to **1e-9** | **A ≠ C at C's tolerance only** | **Precision/rounding, not an error.** A test written at 1e-9 would fail on correct code → R-16. |
| D11 | `m_H` | 1.6735575e-24, matches **no** named convention (+1.47e-5 vs H-1) | collectively cited "CODATA 2018 / IAU 2015" — but m_H is not CODATA-tabulated | band `[1.6735e-24, 1.6738e-24]` — stored value **passes** | **A ≠ B (citation), A ≈ C (in band)** | Numerically fine; **provenance defect** → R-15, S4. All three lenses independently ask for per-constant provenance. |
| D12 | `e` (elementary charge) | pre-2019 value, rel −9.6e-8; not re-exported | included in the collective "CODATA 2018" attribution | not covered | **A ≠ B (citation)** | Rounding + stale-convention + unused → R-16, S4. |
| D13 | `ConversionConstants` scope | contains `km2pc` and `v_kms2au` | docstring `:59` says "**All** constants convert CGS → AU" | — | **A ≠ B** | km and km/s are not CGS. Stale docstring → R-24, S4. |
| D14 | 5%-prominence threshold | code is `prom ≥ 0.05·y_range` (`:587`) | `:298` says "exceeds"; `:652` says "≥" | — | **A resolves B's contradiction** | `:298` is the stale comment. Folded into "logged, not filed". |
| D15 | `_simplify` output order | returns **input order**, not x-sorted | this **is** the documented contract (`:298`, `:438`) | "x-order preserved" | **A = B; C differs** | B refutes A's "either document or sort" framing — it *is* documented. A-10 demoted to an external-consumer note. C's requirement is about the *input's* order being honoured, which it is. |
| D16 | `r_squared = 1.0` when `y` is constant | files it as a defect ("quality gate passes unconditionally") | — | **explicitly requires 1.0**, because NaN would make `NaN < warn_below_r2` False and *suppress* the warning | **A ≠ C on what is correct** | **C's reasoning is stronger and A's proposed fix (NaN) is the failure mode C names.** A-16's `r_squared` half **dropped**; its empty-input half retained. |
| D17 | `_simplify` sort of non-monotonic x | confirms the sort happens | `:438` claims the reordering is "unaffected"; `:298` accepts non-monotonic x | — | **B-internal contradiction, mechanism corroborated by A** | Consequence untested by any lens → R-10, S3. |
| — | *(spec, not code)* `v_au2kms` | — | — | SPEC-091's `0.977781` is wrong in the 6th digit; correct `0.9777922` | **C vs reference spec** | The code computes it as `1/v_kms2au` = 0.97779222, i.e. **the code is right and the spec is stale.** Recorded for the spec owner; no code finding. |

**No scope-creep divergences found.** Nothing in the slice is an unsanctioned addition that all three
lenses agree on.

---

## 5. Helper edge cases — reconciled (weighted heavily per brief)

| helper | edge case | A (executed) | B (documented) | C (required) | verdict |
|---|---|---|---|---|---|
| `find_nearest` | exact tie | first / lowest index (`argmin`) | undocumented | deterministic, and **consistent** with `_lower`/`_higher` so that `lower ≤ nearest ≤ higher` | **PASS** — `argmin` lowest-index is C's own stated acceptable convention |
| `find_nearest` | empty array | raises `ValueError` | undocumented | **must raise**, not return `array[0]`/`None` | **PASS** |
| `find_nearest` | out of range | returns endpoint (silent) | undocumented | correct for *this* function, but caller must range-check | **PASS with caveat** — the range check must live at the call site (S2/S4 slice) |
| `find_nearest` | 2-D input | returns a **flattened** index; A notes "the two callers then use it as a 1-D index" | undocumented | — | **A-only observation, no call site named.** Watch item, not filed. |
| `find_nearest` | NaN | untested | undocumented | `argmin` selects the NaN's index; a NaN cooling-table entry would be silently chosen | **NOT COVERED by any lens** — open gap |
| `find_nearest_lower` | **exact node** (`value == array[k]`) | **not** stepped ⇒ returns `k` | contract says "≤ value" | `≤` not `<` is **required**; strict `<` puts the interpolation weight at the wrong end | **PASS — corroborated clearance.** This is the off-by-one C rated most insidious ("invisible except exactly at nodes"); it is **not** present. |
| `find_nearest_lower` | `value < min(array)` | returns idx 0, whose value is **greater** than `value`; clamps silently | code's own comment: "the returned idx is actually **higher** than the value instead of the desired lower … **Not quite sure what to do with that for now** … this part of the code **shouldnt need to run anyway**" | **must raise or return a checkable sentinel**; "the classic silent-failure in this family" | **FAIL — 3-lens corroboration → R-02 (S2)** |
| `find_nearest_higher` | `value > max(array)` | returns last idx, value < `value`; clamps silently | boundary comment is a **verbatim copy** of the `lower` one and cannot describe this branch | mirror requirement | **FAIL — 3-lens → R-02 (S2)**, plus doc-drift R-22 |
| `find_nearest_lower` | duplicate values `[1,2,2,3]`, v = 2.5 | returns idx **1**, not the last idx with value ≤ v (idx 2) | undocumented | — | **A-only.** Value-equivalent; index-inequivalent **iff the caller indexes a parallel array** (SPS age→L, cooling T→Λ are exactly that shape). No lens named a call site → R-21, S4. |
| `find_nearest_higher` | array with one tolerated glitch | returns idx 9 (value 9.0) for `value = 50` on `[1,2,3,100,4,…,9]` — **wrong bracket in the interior**, no raise | `_is_monotonic_or_tolerable` documented as tolerating "any depth" single points | guard must reject real structure | **FAIL — worst helper behaviour found → R-01 (S2)** |
| both lookups | precondition symmetry | `_lower` uses strict `monotonic()` + `kindof_increasing()`; `_higher` uses lenient `_is_monotonic_or_tolerable` + endpoint direction. Same array: `_lower` raises, `_higher` answers. | `:78` documents the tolerant check for `_higher` **only**; `:161` documents the endpoint direction as deliberate | **predicted blind** from the signature layout (`_lower` at L30, before the guard block L94–143; `_higher` at L146, after it) | **FAIL — 3-lens → R-08 (S3).** C's blind prediction from line ordering alone is the single most striking cross-lens hit in this slice. |
| `_is_monotonic_or_tolerable` | relative vs absolute tolerance | **relative**: `\|L[start]−L[end]\| / max(\|L[start]\|,1e-300)` | "max relative drawdown" | **must** be relative (C's deepest unit-hygiene claim, S1) | **PASS — C-24 (S1) retired** |
| `_is_monotonic_or_tolerable` | `BOUNDARY_FRAC` | `0.01`, but excuses only the **low-index** end, never the high-index end | "leading fraction treated as a startup transient" | `0 < f < 0.5`; excuses "each end" (start-up **and** the ξ→1 singular end) | **PASS on the value** (C-26's dead-guard case does not apply). **A ≠ C on two-endedness** — the trailing exemption is missing; fails *safe* (stricter) → R-23, S4. |
| `_is_monotonic_or_tolerable` | `MAX_SPIKE_LEN` | `2` | "longest wrong-direction run" | 1–3 | **PASS** |
| `_is_monotonic_or_tolerable` | degenerate reduction to `monotonic()` | untested | — | required | **NOT COVERED** — open |
| `_is_monotonic_or_tolerable` | direction-agnostic (`guard(L) == guard(−L)`) | direction taken from `L[-1] >= L[0]` then diffs negated ⇒ agnostic by construction; not asserted | — | required | **PASS by construction, untested** |
| `monotonic` / `kindof_*` | plateaus, constant arrays | **non-strict** (≤/≥); constant arrays pass | `:67` "kind of, because includes equal values like `[1,2,3,3,4]`" | **must** be non-strict or converged plateaus raise | **PASS — 3-lens clearance (C-23 retired)** |
| `_peak_prominences` | plateau maximum, leading point | `[0,5,5,0]`, idx `[1,2]` → `[0, 5]`; scipy gives `[5,5]` | `:196` "+inf so the other side dominates" — B shows +inf makes the **empty** side dominate | flat-topped peak must have non-zero prominence | **FAIL — 3-lens → R-04 (S3)** |
| `_peak_prominences` | extremum at index 0 / n−1 | `[5,1,2,0,3]`, idx 0 (the **global max**) → prominence 0 | same sentinel bug | boundary peak's window extends to the array end | **FAIL**, but A supplies the mitigation: indices 0 and n−1 are separately mandatory, so the live exposure is the plateau case |
| `_peak_prominences` | interior non-plateau peaks | matches `scipy.signal.peak_prominences` exactly | — | ≥ 0 always | **PASS** |
| `_rmq` | empty range | manifests as the ±inf sentinel path (same root cause as above) | "shouldn't happen for real extrema" | must not silently return `st[0][lo]` | folded into R-04 |
| `_rmq` | power-of-two window lengths | brute-forced, 0 mismatches for n ∈ {1,2,3,5,8,17,33} | — | the convention-mix magnet (C-34, S2) | **PASS — C-34 retired** |
| `_simplify` | dense uniform input | 2e6 points → **1 point**, and the final endpoint `(1.0, 1.0)` is **lost** | endpoints are tier-1 mandatory | endpoints **must always** be retained | **FAIL — → R-03 (S2)** |
| `_simplify` | subset-selection (never resample) | index selection throughout; `_restore` maps to original positions | — | S1-rated: every output pair must be bit-identical to an input pair | **PASS on the resampling half** |
| `_simplify` | `nmin` below the floor | floored to 20 **after** the early return; `nmin=5` on 1000 pts → 26 | floor rationale `2 + 20 ≤ 20` is arithmetically false | `2 ≤ _COVERAGE_CHUNKS ≤ nmin` | **PASS on `_COVERAGE_CHUNKS = 20`**; rationale is wrong → R-19, S4 |
| `_simplify` | `dedup_tol = 0` "disables" | untested | rule still folds exact duplicates | — | **NOT COVERED** — open |
| `_simplify_error` | `np.interp` outside `x_s` | **clamps**, does not extrapolate | — | — | **PASS** (clamping is the safe behaviour for an error metric) |
| `_simplify_error` | `y_o` constant (`SS_tot = 0`) | returns `r_squared = 1.0`; A files this as a defect | — | **explicitly requires 1.0**; NaN would suppress the warning | **PASS — A's finding dropped, see D16** |
| `_simplify_error` | zero `y_orig` in max-rel-err | masked at `1e-30` | `:760` documents the `1e-30` skip | must guard the denominator | **PASS — 3-lens clearance** |
| `_simplify_error` | empty input | unguarded `ValueError` from `np.interp` | — | — | A-only, S4, logged not filed |
| `_simplify_error` | R² **direction** (simplified→original grid?) | not stated | not stated | S2-rated: reversing it reports R² = 1 always | **NOT COVERED by any lens** — highest-value open gap in `simplify` |
| `_x_uniform_coverage_idx` | bins uniform in **x** or in **index**? | not stated | name + docstring say x | S3-rated: index-binning is a no-op | **NOT COVERED** — open |
| `get_optimal_workers` | returns 0? | `max(1, ...)` present → 1 on a 2- or 4-core box | documented | must never return 0 (`Pool(0)` raises) | **PASS — C-42 retired** |

---

## 6. Demoted or dropped, with reasons

| lens item | action | why |
|---|---|---|
| **C-16** (S1, `convert2au` silently returns 1.0 for unknown units) | **dropped** | A executed it: every unknown token raises `UnitConversionError`. C's rank-1 trap is not present. |
| **C-17** (S1, alternating `a/b/c` parser) | **dropped** | A ran C's own distinguishing test (`cm**3/g/s**2` → `G_cgs2au`). Flattening is correct. |
| **C-02** (S1, mixed M☉ conventions) | **dropped** | A: all 18 compound constants derive from one M☉ to ≤1 ULP. |
| **C-07** (S1, cooling closure `Λ·n² = du/dt`) | **dropped** | Verified numerically from A's stored values in §2c. |
| **C-24** (S1, `MONOTONIC_RTOL` applied absolutely) | **dropped** | A shows the test is relative. Only the *magnitude* survives (→ R-09). |
| **C-29** (S1, `get_soundspeed` coefficient) | **dropped** | A's executed value matches C's independently derived `8.6289090e-3` to 5 s.f. |
| **C-05 / C-04 / C-06** (S2 alias identities) | **dropped** | All three aliases are byte-identical literals in the code. |
| **C-34** (S2, `_rmq` interval-convention mix) | **dropped** | A brute-forced all windows incl. power-of-two lengths: 0 mismatches. |
| **C-35** (S3, non-idempotent reducer) | **dropped** | Purely hypothetical; no lens found a non-min/max caller. |
| **C-42** (S3, `get_optimal_workers` returns 0) | **dropped** | The `max(1, ·)` guard is present. |
| **C-45** (S4, `PHASES` contains `collapse`) | **dropped** | B's phase inventory shows `energy/implicit/transition/momentum` only. |
| **C-19** (S3, `split_units` lexing `cm3`/`cm-3`) | **dropped** | C assumed a no-`**` syntax. The actual grammar is `name(**exponent)?` with `Fraction`; A verified `cm**3`, `cm**-3`, `s**(-3/2)`, `cm**(1/2)`. C's hazard does not exist. |
| **C-23** (S2, strict monotonic predicates) | **dropped** | A: predicates are non-strict; constant arrays pass. |
| **C-37** (S2, `_DEDUP_TOL` absolute) | **half dropped** | It *is* relative. The magnitude half survives, merged into R-03. |
| **B-17** (S3, Menger curvature sign / factor) | **dropped** | A's algebra: `2\|u×w\|/(\|u\|\|w\|\|u+w\|)` is exactly Menger; the cross is absolute-valued; the `+1` index map is right. |
| **B-16** (S3, `_peak_prominences` sentinel) | **kept and promoted in rank** | B predicted it blind from comments; A confirmed by execution against scipy; C required the invariant. → R-04. |
| **B-04** (S3, `get_soundspeed` unit system unstated) | **demoted to S4** | B asked a code lens to confirm the AU `k_B` is used; A confirms it is. Residual is a docstring gap only. |
| **B-21** (S4, "OR-on-Δ" vs "both") | **demoted, folded to prose** | A confirms the implementation is the De Morgan dual of the docstring rule — both comments are correct. |
| **A-16** (`r_squared = 1.0` for constant `y`) | **half dropped** | C argues, correctly, that 1.0 is the *right* answer and A's proposed NaN is the failure mode (`NaN < threshold` is False ⇒ warning suppressed). Empty-input half retained in prose. |
| **A-10** (output not x-sorted) | **demoted, folded to prose** | B shows input-order output **is** the documented contract at `:298`/`:438`. Only the external-consumer hazard remains. |
| **A-01** (`m_H` 1.47e-5 high) | **reclassified** | Passes C's own acceptance band and consistency test. Not a coefficient defect; re-filed as a **provenance/citation** item, R-15. |
| **A-02** (`tau_cgs2au` units) | **kept, reframed** | B's docstring shows code and comment agree it is a surface density. The defect is the **name**, not the value. → R-06. |

**Logged, not filed** (below the filing bar, or fully explained by another lens): `setup_logging`
clears pre-existing root handlers; `_simplify_error` raises an unguarded `np.interp ValueError` on
empty input; `extract_example_snapshots` `__doc__.strip()` breaks under `python -OO` and treats a
falsy-but-present `SimulationEndReason` as not-terminated; `__post_init__` unpacks a dead `value`
loop variable and checks only `> 0`; `:298`'s "exceeds 5 %" contradicts the code's `≥` (`:652` is
right); "output size is normally `nmin`" omits the shorter-than-`nmin` path; `L_cgs2au`,
`dudt_cgs2au` and `v_kms2au` are 1-ULP off their composition identities (harmless unless a test
asserts exact equality); A's §2b self-contradiction about which `*_CGS` names are re-exported.

---

## 7. Open — not resolved by any lens

1. **R² direction in `_simplify_error`** — C rates reversing it S2 (reports R² = 1 always); no lens
   established which grid is the reference. *Cheapest high-value check in the slice.*
2. **Does `_x_uniform_coverage_idx` bin in x or in index?** C: index-binning makes the whole
   coverage tier a no-op.
3. **`find_nearest` with NaN** — `argmin` would select the NaN's index.
4. **`_is_monotonic_or_tolerable(L, rtol=0, boundary_frac=0, max_spike_len=0) == monotonic(L)`.**
5. **`dedup_tol = 0`** — does it actually disable, or still fold exact duplicates?
6. **Is `find_nearest_higher` still on a production path?** B quotes a "RETAINED FALLBACK" comment
   saying the bubble-luminosity solver is moving to an event-based split that does not call it. This
   materially changes the priority of R-01. **Needs the call-site slice.**
7. **Does `DedupWarningFilter` state survive across runs inside a sweep worker?** Depends on whether
   `setup_logging` is re-invoked per run — outside S1.
8. **`_simplify` nesting guarantee** — no lens ran the subset test across budgets.
9. **Feature loss on genuinely non-monotonic x** — no lens ran B's proposed up-then-down test.
10. **Hand-off (not S1):** `μ_ion` / `μ_atom` defaults live in `.param`. A's working values (0.6,
    1.27 m_H) do not match C's composition-derived set (14/23 = 0.6087, 14/22 = 0.6364,
    14/11 = 1.2727, 14/6). A did not cite an in-slice source for those numbers, so this is **not** a
    finding here — but the μ convention (per particle vs per H nucleus) is C's rank-4 trap and
    belongs in the parameter/defaults slice.
11. **Hand-off (not S1):** whether any caller needs the isothermal γ = 1 sound speed (R-13).

---

## 8. The single most valuable action from this slice

Every clearance in §2 rests on constants that **no automated test protects**. A found the module's
own base-conversion self-test uses an *absolute* 1e-20 tolerance against constants of magnitude
1e-19 to 1e-34 — it passes no matter what the constants hold — and B found the "verified against
astropy" guarantee lives in a `__main__` block whose only accuracy check is skipped when astropy is
absent. C authored ~30 invariants; **every arithmetic one passes today**. Encoding C's §4 set as
pytest assertions is a few dozen lines, costs nothing at runtime, and would make this reconciliation
permanent instead of point-in-time. That is R-07, and it is the highest value-per-line item here.

---

```json
[
  {
    "id": "S1-R-01",
    "file": "trinity/_functions/operations.py",
    "line": 131,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "_is_monotonic_or_tolerable exempts a single-step monotonicity violation of ARBITRARY magnitude (`if end - start == 1: continue` precedes the MONOTONIC_RTOL drop test), and find_nearest_higher then returns a bracket that does not contain the query value, with no error.",
    "evidence": "A (executed): _is_monotonic_or_tolerable([1,2,3,1e9,4,...]) -> True (9 orders of magnitude); [1,2,3,0.03,4,...] -> True (99% plunge); by contrast a 3-step 0.5% wobble -> False. Consequence: find_nearest_higher([1.,2.,3.,100.,4.,5.,6.,7.,8.,9.], 50.0) returns idx 9 (value 9.0) although idx 3 holds 100 >= 50. B (comments): the rule is stated three incompatible ways -- :91 'both shallow (<= MONOTONIC_RTOL) AND localized', :102/:132 'any depth', :147 'sub-percent single-point spike'. The code matches :102/:132, so :91 and :147 are stale. C (required): MAX_SPIKE_LEN must be a small positive integer AND the tolerance must bound the excursion depth.",
    "expected": "Either the relative drop test applies to runs of length 1 as well, or the single-step exemption is bounded by MONOTONIC_RTOL exactly as the length-2 case is. Independently, find_nearest_higher must verify that the index it returns satisfies array[idx] >= value before returning it.",
    "failure_scenario": "A cooling or SPS table with one glitched sample passes the guard, and the +/-1 step logic returns a neighbouring index without checking that it brackets the query. The lookup silently interpolates off the wrong bracket -- a wrong physical value from a call that raised nothing. Mitigating context (B :79): a RETAINED FALLBACK comment says the bubble-luminosity solver is moving to an event-based split that does not call find_nearest_higher, so live exposure may be lower than it appears; this must be confirmed at the call sites.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import find_nearest_higher as f; a=[1.,2.,3.,100.,4.,5.,6.,7.,8.,9.]; print(f(a,50.), a[f(a,50.)])\"  # 9 9.0",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S1-A-03", "S1-B-03", "S1-C-26"]
  },
  {
    "id": "S1-R-02",
    "file": "trinity/_functions/operations.py",
    "line": 60,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "find_nearest_lower and find_nearest_higher clamp out-of-range queries to the nearest end index and return it with no signal, so the returned element violates the inequality the function name and docstring promise.",
    "evidence": "A (executed): find_nearest_lower(np.linspace(1,5,5), 0.1) -> idx 0, value 1.0 (1.0 > 0.1, no element <= 0.1 exists); find_nearest_higher(np.linspace(1,5,5), 99.) -> idx 4, value 5.0 (5.0 < 99). Same on decreasing input. Clamps at :60-63 and :179-182. B (comments): the code admits it at :56 -- 'when these happen, it means that the returned idx is actually higher than the value instead of the desired lower' followed by 'Not quite sure what to do with that for now' and 'this part of the code shouldnt need to run anyway'; :175 repeats the same text verbatim inside find_nearest_higher, where it cannot describe that branch. C (required, predicted blind): 'Returning array[0] -- which is higher than value -- violates the function's own name and postcondition, and turns an out-of-range query into a silent 1-sided extrapolation. This is the classic silent-failure in this family.'",
    "expected": "Raise, or return a sentinel the caller must check, when no element satisfies the requested inequality. If the clamp is deliberate, the docstring must state the postcondition it actually delivers, and every call site must range-check.",
    "failure_scenario": "A table lookup for a T, n, Phi or cluster age outside the tabulated grid is silently answered with the edge cell, so the run continues on extrapolated-as-constant physics instead of stopping. C's concrete case: the bubble temperature drifts below the CIE table's 1e4 K floor, L_cool is evaluated at the wrong temperature, and the (L_gain-L_loss)/L_gain energy-to-momentum transition trigger fires at the wrong time -- which is the code's headline prediction.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.operations import find_nearest_lower as f; a=np.linspace(1.,5.,5); print(f(a,0.1), a[f(a,0.1)])\"  # 0 1.0",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-A-04", "S1-B-01", "S1-B-02", "S1-C-21"]
  },
  {
    "id": "S1-R-03",
    "file": "trinity/_functions/simplify.py",
    "line": 466,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The dedup filter compares each point to its immediate predecessor rather than to the last kept point, so a curve made of many sub-tolerance steps collapses entirely -- including its final point, because the collapsed array then hits the `nmin >= x.size` early return that bypasses endpoint preservation.",
    "evidence": "A (executed): _simplify(np.linspace(0,1,2_000_000), np.linspace(0,1,2_000_000), nmin=100) returns arrays of size 1, x=[0.], y=[0.] -- the endpoint (1.0, 1.0) is gone. Mechanism: :473 keeps point i only if |x[i]-x[i-1]| > 1e-6*range_x OR |y[i]-y[i-1]| > 1e-6*range_y; at n=2e6 the uniform step is 5e-7 of range, so every interior point fails both, leaving x.size==1, which makes :496 return before mask[0]/mask[-1] at :631-632 are set. At n=200000 the same call correctly returns 100 points. Trip-wire is n ~ 1/dedup_tol = 1e6. C (required, independently): endpoints must ALWAYS be retained (S1-rated), and _DEDUP_TOL_DEFAULT must be 1e-12..1e-9 -- i.e. C's tolerance bar, derived from names alone, is exactly what would move the trip-wire out of reach. B: dedup_tol default 1e-6 is documented as safe 'unless the input has more than ~10^6 uniformly-sampled points'.",
    "expected": "Anchor the tolerance to the last RETAINED point rather than the immediate predecessor, and/or force-retain index x.size-1 on the early-return path. Consider tightening _DEDUP_TOL_DEFAULT toward C's 1e-9 band, which removes the trip-wire entirely.",
    "failure_scenario": "Any recorded quantity sampled densely enough (>~1e6 samples, or a locally stalled integrator segment) is written to output as a single point, losing the whole trajectory silently -- no warning fires because the R^2 check requires merged.size >= 2. Downstream audits then read a constant as physics.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _simplify; n=2_000_000; x=np.linspace(0,1,n); xs,ys=_simplify(x,x,nmin=100); print(xs.size, xs, ys)\"",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-A-05", "S1-C-32", "S1-C-37"]
  },
  {
    "id": "S1-R-04",
    "file": "trinity/_functions/simplify.py",
    "line": 196,
    "class": "sign",
    "severity": "S3",
    "claim": "_peak_prominences returns 0 for the leading point of a plateau extremum and for extrema at index 0 / n-1: the empty one-sided search interval leaves the shoulder at +inf, which propagates through np.maximum to y - inf = -inf and is then clipped to 0.",
    "evidence": "B (comments only, predicted blind): :196 sits inside the MAX-candidate block and says 'if a side is empty (shouldn't happen for real extrema) treat its shoulder as +inf so the other side dominates' -- but for a maximum the key col is the HIGHER of the two shoulders, so +inf makes the EMPTY side dominate; the :229 negative clamp then silently yields 0. A (executed, confirming B's prediction): _peak_prominences([0.,5.,5.,0.], [1,2]) -> [0., 5.] where scipy.signal.peak_prominences gives [5., 5.]; _peak_prominences([5,1,2,0,3], [0]) -> [0.] although index 0 is the global maximum. Interior non-plateau peaks agree with scipy exactly. Root cause includes the deliberate asymmetry in _prev_next_strict (prev uses >, next uses >=), which makes the right walk range empty for a plateau's leading point. C (required): all prominences >= 0, a flat-topped peak must have non-zero prominence, and 'with >= a flat top bounds itself and prominence collapses to 0, silently deleting flat-topped features'.",
    "expected": "Treat an empty side as 'no constraint' -- use the other side alone (equivalently seed the empty side with -inf for maxima and +inf for minima) -- matching scipy semantics.",
    "failure_scenario": "Points with spuriously-0 prominence fail the `proms >= 0.05*y_range` mandatory filter and are not promoted, so the leading edge of a flat-topped feature (a saturated Lmech between SN episodes, a plateau in shell velocity or bubble temperature) is dropped from the saved trajectory while its trailing edge is kept. A supplies the mitigation: indices 0 and n-1 are separately mandatory, so the live exposure is the plateau case, not the boundary case.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _peak_prominences as p; print(p(np.array([0.,5.,5.,0.]), np.array([1,2])))\"  # [0. 5.]",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S1-A-07", "S1-B-16", "S1-C-36"]
  },
  {
    "id": "S1-R-05",
    "file": "trinity/_functions/logging_setup.py",
    "line": 68,
    "class": "state",
    "severity": "S3",
    "claim": "ColoredFormatter.format mutates the shared LogRecord in place; because the console handler is registered before the file handler, ANSI escape sequences are written into the persisted .log file -- contradicting the module's own claim that log files are plain text.",
    "evidence": "A (executed end-to-end): :68 and :71 assign back to record.levelname / record.name; console addHandler at :270, file addHandler at :302. With a TTY-reporting stdout and use_colors=True the written line is b'2026-07-30 ... | \\x1b[33mWARNING \\x1b[0m | \\x1b[94mroot\\x1b[0m | shell temperature clamped'. B (comments): :122 states 'log files plain text'. C (required, independently): 'colour codes must be suppressed for the file handler and for non-TTY stdout, or the log files fill with \\x1b[ escapes'.",
    "expected": "Format a copy (build the coloured string locally, or copy.copy(record)) so downstream handlers see the unmodified record; the file handler must never receive colour codes.",
    "failure_scenario": "Archived run logs contain escape bytes, so grep/diff/parse of trinity_*.log across runs is corrupted and log comparison in a regression workflow reports false differences -- at exactly the scale where log-based diagnostics matter.",
    "repro": "PYTHONPATH=pkg python3 -c \"import sys,io,os,tempfile,logging; from trinity._functions.logging_setup import setup_logging;\\nclass T(io.StringIO):\\n  def isatty(self): return True\\nsys.stdout=T(); d=tempfile.mkdtemp(); l=setup_logging('INFO',True,True,d,'t.log',True); l.warning('x'); [h.flush() for h in logging.getLogger().handlers]; sys.stdout=sys.__stdout__; print(b'\\\\x1b[' in open(os.path.join(d,'t.log'),'rb').read())\"  # True",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S1-A-06", "S1-C-44"]
  },
  {
    "id": "S1-R-06",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 131,
    "class": "units",
    "severity": "S3",
    "claim": "`tau_cgs2au` holds the conversion for a SURFACE MASS DENSITY (g cm^-2 -> Msun pc^-2). The stored number is arithmetically correct for that dimension and the docstring agrees -- but the name `tau` denotes either an optical depth (dimensionless, factor 1.0) or a timescale (factor s2Myr). The identifier is the defect.",
    "evidence": "A (recomputed): 4788.452460043275 == g2Msun * pc2cm**2 == convert2au('g*cm**-2') bit-identically; recomputed 5.029144215870041e-34 * (3.0856775814913674e18)**2 = 4788.452460043276. B (comments): the docstring at :130 declares the row as 'Surface density | g/cm^2 -> Msun/pc^2' -- so code and comment AGREE. C (from the name alone): required s2Myr = 3.1688088e-14 under the timescale reading, or exactly 1.0 under the optical-depth reading; the invariants `tau_cgs2au == s2Myr` and `tau_au2cgs == Myr2s` therefore FAIL. C explicitly allowed for this: 'if the module holds 1.0 here, that is also correct under Reading 2 and the finding is a naming defect, not an arithmetic one.'",
    "expected": "Rename to match the quantity (e.g. sigma_cgs2au / surface_density_cgs2au). If a genuinely dimensionless optical depth is ever converted, it needs factor 1.0 -- i.e. no constant at all.",
    "failure_scenario": "Anyone applying tau_cgs2au to an actual optical depth scales a dimensionless number by 4788; f_abs = 1-exp(-tau) then saturates and direct radiation pressure is misreported. Conversely a reviewer checking 'tau should be 1' wrongly flags a correct constant. No lens identified a call site that uses it as an optical depth, so this is latent.",
    "repro": "python3 -c \"import trinity._functions.unit_conversions as c; print(c.convert2au('g*cm**-2') == c.CONV.tau_cgs2au)\"  # True",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S1-A-02", "S1-C-11"]
  },
  {
    "id": "S1-R-07",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 508,
    "class": "other",
    "severity": "S3",
    "claim": "The module advertises 'hardcoded for SPEED but verified against astropy for accuracy', yet nothing enforces that: the base-conversion self-test uses an ABSOLUTE 1e-20 tolerance against constants of magnitude 1e-19 to 1e-34 (so it passes regardless of their values), and the astropy comparison lives in a __main__ block that is additionally skipped when astropy is absent.",
    "evidence": "A: `passed = abs(result - expected) < 1e-20` compared against cm2pc = 3.24e-19 (tolerance is 3% of the value), s2Myr = 3.17e-14, g2Msun = 5.03e-34, km2pc = 3.24e-14 (tolerance is 3e5 times the value -- unconditionally true). The compound tests at :524 and the astropy tests at :578 do use relative tolerances. B: :3 and :59 claim 'derived from astropy.units and frozen', 'see verification test at bottom'; :481 heads the suite 'run with: python unit_conversions.py'; :560 is 'Test 6: Verify against astropy (if available)'. C authored ~30 relational invariants, every arithmetic one of which passes on the current constants.",
    "expected": "Relative comparisons throughout, and a pytest test that encodes C's invariant set: the 21 reciprocal pairs (|x*x_inv - 1| < 1e-15), the 6 alias identities (F==pdot, k_B==E, c_therm==pdotdot -- exact equality), the 16 composition identities incl. the cooling closure Lambda*ndens**2 == dudt, and the three-way Msun recovery. Note sigma_SB needs 1e-7, not C's proposed 1e-9 (see S1-R-16).",
    "failure_scenario": "A hand-edited constant drifts and CI stays green, because the only real comparison lives in a manually-invoked __main__ block whose base-conversion half cannot fail. This audit had to recompute 31 constants by hand precisely because no such gate exists.",
    "repro": "python3 -c \"print(abs(3.24e-14 - 0.0) < 1e-20)\"  # False -- but substitute any wrong g2Msun and the 1e-20 absolute test still passes",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-A-11", "S1-B-11", "S1-C-01", "S1-C-02", "S1-C-07"]
  },
  {
    "id": "S1-R-08",
    "file": "trinity/_functions/operations.py",
    "line": 163,
    "class": "divergence",
    "severity": "S3",
    "claim": "The two sibling lookups impose different monotonicity contracts and use different direction tests, so they can disagree about the same array: find_nearest_lower requires strict monotonic() and takes direction from kindof_increasing(); find_nearest_higher uses the lenient _is_monotonic_or_tolerable and takes direction from the endpoint comparison array[-1] >= array[0].",
    "evidence": "A (executed): on [1.,2.,3.,100.,4.,5.,6.,7.,8.,9.] find_nearest_lower raises MonotonicError while find_nearest_higher returns an answer. Guards at :40 vs :157; direction at :44 vs :163. B (comments): :78 documents the tolerant check for find_nearest_higher only; :161 documents the endpoint direction as deliberate ('robust to a tolerated local spike that would otherwise make the all-pairs kindof_increasing() return False'); find_nearest_lower has no such note. C (predicted blind from the signature layout): 'find_nearest_lower at L30 (before the monotonic-guard block at L94-L143) and find_nearest_higher at L146 (immediately after it) -- a layout that strongly suggests only the higher variant validates monotonicity.'",
    "expected": "One contract shared by both, or an explicit statement of why the asymmetry is intentional, in BOTH docstrings. If the tolerant guard is right for one, it is right for the other.",
    "failure_scenario": "Code that brackets a value by calling both helpers gets an exception from one and an index (possibly the wrong one, see S1-R-01) from the other on the same array. B notes the MonotonicError is caught by get_betadelta as a penalised, retried trial, so which helper a path happens to call determines whether a trial is penalised -- making the outcome depend on the code path rather than on the data.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import find_nearest_lower as lo, find_nearest_higher as hi; a=[1.,2.,3.,100.,4.,5.,6.,7.,8.,9.]; print(hi(a,4.2));\\ntry: lo(a,4.2)\\nexcept Exception as e: print(type(e).__name__)\"",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-A-15", "S1-B-05", "S1-C-28"]
  },
  {
    "id": "S1-R-09",
    "file": "trinity/_functions/operations.py",
    "line": 94,
    "class": "numerical",
    "severity": "S3",
    "claim": "MONOTONIC_RTOL = 1e-2 is four or more decades looser than a round-off filter needs. It is correctly applied as a RELATIVE tolerance (C's S1-rated units concern is refuted), but at 1% it admits real physical structure rather than numerical noise.",
    "evidence": "A: the test is |L[start]-L[end]| / max(|L[start]|, 1e-300) <= 1e-2, i.e. relative -- so the deepest units concern (an absolute threshold being unit-system-dependent) does NOT apply. C (derived from the guard's stated purpose): the tolerance must exceed accumulated round-off (~N*eps ~ 1e-13) and stay far below the fractional change a Weaver interior profile makes between adjacent samples (O(1e-2)), giving 1e-12..1e-6 with 1e-8 natural; 'anything >= 1e-4 is too loose to be a noise filter'. B: :147 describes the tolerated case as a 'sub-percent single-point spike', which 1e-2 (= 1%) is not.",
    "expected": "MONOTONIC_RTOL in 1e-12..1e-6 (C's band), or an explicit, dated justification in the source for why 1% is required by the numpy-version symptom CLAUDE.md documents. Because it is a tuning constant rather than a physical one, tightening it needs a full-run equivalence check per CLAUDE.md rule 5.",
    "failure_scenario": "A genuinely non-monotonic bubble density or temperature profile (a real inversion, not FP noise) is accepted, isobaricity silently breaks, and P_b becomes r-dependent inside a solver that assumes it is not. The guard reads as active while filtering nothing at the scale it was written for.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import MONOTONIC_RTOL; print(MONOTONIC_RTOL)\"  # 1e-2",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-C-25", "S1-C-24", "S1-A-03"]
  },
  {
    "id": "S1-R-10",
    "file": "trinity/_functions/simplify.py",
    "line": 438,
    "class": "state",
    "severity": "S3",
    "claim": "_simplify sorts its working copy by x, and the comment claims the reordering is harmless because 'the rest of the algorithm is sequence-based'. For the genuinely non-monotonic x the same function's contract explicitly accepts, sorting is a permutation that changes the point sequence -- so curvature triplets, sign changes, arc length and peak persistence are all computed on a different curve than the caller supplied.",
    "evidence": "B (comments): :298 'Input may be ascending, descending, or non-monotonic in x' vs :438 'the rest of the algorithm is sequence-based (curvature on triplets, sign changes, cumulative arc length, peak persistence) and is unaffected by the temporary reordering'. A confirms the mechanism: the working copy is sorted, :486 maps working indices through dedupe_idx, :488 through sort_order back to original positions, :489 sorts those positions. Neither A nor C tested a non-monotonic-x curve, so the CONSEQUENCE is unverified.",
    "expected": "Scope the 'unaffected by reordering' claim to ascending/descending input, or withdraw non-monotonic support, or select features on the unsorted sequence.",
    "failure_scenario": "A non-monotonic trajectory (a quantity plotted against a variable that reverses, or r(t) with a re-collapse) is thinned using features of a scrambled curve: real bends are missed and spurious ones introduced, while the output still looks plausible because ordering is restored on the way out.",
    "repro": "x = np.concatenate([np.linspace(0,1,500), np.linspace(1,0,500)]) with a sharp feature only on the return leg; check whether that feature survives _simplify.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-B-15"]
  },
  {
    "id": "S1-R-11",
    "file": "trinity/_functions/cluster.py",
    "line": 49,
    "class": "regime",
    "severity": "S3",
    "claim": "get_optimal_workers keys the 'use the full allocation' branch on SLURM_JOB_ID, while detect_allocated_cpus derives the count from SLURM_CPUS_PER_TASK -> SLURM_CPUS_ON_NODE -> sched_getaffinity -> os.cpu_count(). Inside a SLURM job that exports neither SLURM_CPUS_* variable, detection can fall through to os.cpu_count() (the whole node) AND the full-allocation branch is taken -- the exact oversubscription the module says it prevents.",
    "evidence": "B (comments): :3 gives the precedence list and the '64-core node, 4-core job -> ~31 workers' example; :49 keys the full-allocation branch on SLURM_JOB_ID. A (executed): env vars SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE are read first (rejecting non-digit and 0), then sched_getaffinity, then cpu_count(); get_optimal_workers returns max(1, cpu_count//2 - 1) off-SLURM; and 'the affinity/cpu_count detection path in detect_allocated_cpus is reachable ONLY under SLURM_JOB_ID'. C (required): SLURM -> cgroup quota -> sched_getaffinity -> cpu_count, always >= 1. The cgroup-quota rung is absent.",
    "expected": "Key both detection and the worker formula on the same signal, and add the cgroup rung (v2 /sys/fs/cgroup/cpu.max, v1 cpu.cfs_quota_us/cpu.cfs_period_us) before os.cpu_count(). The conservative halving should apply whenever the allocation is not positively known.",
    "failure_scenario": "A job submitted without --cpus-per-task on a 64-core node spawns 64 simulation subprocesses inside a 4-core allocation, thrashes, and is killed by the scheduler for exceeding its cgroup or time limit -- diagnosed as 'trinity is slow'. Mitigated only where cgroup/cpuset confinement makes sched_getaffinity correct.",
    "repro": "SLURM_JOB_ID=1 with SLURM_CPUS_PER_TASK and SLURM_CPUS_ON_NODE unset; call get_optimal_workers and compare to the granted core count.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-B-30", "S1-C-41"]
  },
  {
    "id": "S1-R-12",
    "file": "trinity/_functions/simplify.py",
    "line": 298,
    "class": "numerical",
    "severity": "S3",
    "claim": "The documented nesting guarantee -- 'the subset at any budget N is a superset of the subset at N-1' -- cannot hold, because two selection stages depend on nmin: the arc length is divided into nmin equal bins, and the coverage skeleton is capped at nmin-2 chunks; the nmin-dependent arc-length boundaries are then promoted ABOVE the bisection pool with |idx_dist| ~ nmin.",
    "evidence": "B (comments): :298 item 5 and :567 state the guarantee; :617, :694, :702, :712 establish the nmin-dependence and the promotion. A confirms the mechanism from the code: maxdist = total_arc / nmin (:619) and min(_COVERAGE_CHUNKS, max(0, nmin-2)) (:699). No lens ran the subset test across budgets, so the consequence is unverified.",
    "expected": "Either scope the nesting claim to the nmin-independent part of the selection (endpoints + prominent extrema), or make the arc-length and coverage stages budget-independent.",
    "failure_scenario": "The stated anti-flicker property is relied on -- comparing two runs written at different nmin, or regenerating a plot at a higher budget -- and points appear and disappear anyway, so a diff between two simplified outputs shows changes that are budget artefacts rather than physics.",
    "repro": "For a fixed curve, compute set(_simplify(x, y, nmin=N)) for N = 20..120 and assert set(N-1) is a subset of set(N).",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-B-18"]
  },
  {
    "id": "S1-R-13",
    "file": "trinity/_functions/operations.py",
    "line": 206,
    "class": "regime",
    "severity": "S3",
    "claim": "get_soundspeed applies params['gamma_adia'] unconditionally and selects mu on a hard `T > 1e4` test. The formula and the AU coefficient are correct, but a caller needing the isothermal (gamma=1) ionization-front sound speed would be served the adiabatic one -- a ~29% error -- and c_s is discontinuous at the mu switch.",
    "evidence": "A (executed): c_s = sqrt(gamma_adia * (k_B[au]*k_B_au2cgs) * T / (mu[au]*Msun2g)) * v_cms2au -- dimensionally exact, and A's value at T=1e6 K matches C's independently derived M_sun-independent coefficient (k_B/m_H)_AU = 8.6289090e-3 pc^2 Myr^-2 K^-1 to 5 s.f. So the UNITS are clean (C-29 retired). Remaining: A shows the mu switch at :206 is a strict `if T > 1e4`, so T = 1e4 exactly takes the atomic branch (matching B's documented 'T <= 1e4 -> mu_atom'), and c_s jumps ~45% across a single kelvin at the boundary; T must be a scalar (an array raises). C (required): sqrt(5/3) = 1.29, so one function serving both an isothermal and an adiabatic caller is a ~29% modelling error; the gamma used must be explicit per call site.",
    "expected": "Make gamma a parameter of the call (or document it per call site), and either smooth the ionisation transition or document the discontinuity. np.where for array support if any caller passes a profile.",
    "failure_scenario": "C: the 29% error lands in L_leak = (1-C_f) 4 pi R2^2 c_s (5/2) P_b; with the default C_f = 1.0 the term vanishes, so the bug is invisible in every fiducial run and appears only in the C_f < 1 runs a user turns on deliberately and trusts. Separately, a solver whose residual depends on c_s sees a jump discontinuity when the shell temperature crosses 1e4 K, so root-finders and adaptive steppers can stall or chatter there.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import get_soundspeed as g; import trinity._functions.unit_conversions as c; p={'mu_ion':0.6*c.CGS.m_H*c.g2Msun,'mu_atom':1.27*c.CGS.m_H*c.g2Msun,'gamma_adia':5/3.,'k_B':c.CGS.k_B*c.k_B_cgs2au}; print(g(1e4,p)*c.v_au2kms, g(1.00001e4,p)*c.v_au2kms)\"",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-C-30", "S1-A-12"]
  },
  {
    "id": "S1-R-14",
    "file": "trinity/_functions/logging_setup.py",
    "line": 106,
    "class": "state",
    "severity": "S3",
    "claim": "CONTESTED: whether DedupWarningFilter's seen-set leaks across parameter sets inside a sweep worker. A shows suppression lasts the lifetime of a filter instance; B quotes a docstring claiming 'State is per-process, so it resets every run/task -- no cross-run leakage'; C argues that in a --workers N sweep one process executes many parameter sets, so 'per-process' and 'per run/task' are not the same thing and run #2 onward would emit no warnings at all.",
    "evidence": "A (executed): filter() stores (levelno, getMessage()) in self._seen and returns False on any repeat; three identical WARNING records -> True, False, False. Attached to both console (:269) and file (:300) handlers. A also notes setup_logging calls root_logger.handlers.clear() at :243, which -- if setup_logging is re-invoked per run -- would install fresh filter instances and reset the state. Neither A nor B established whether the sweep runner calls setup_logging per parameter set, and that call site is outside slice S1.",
    "expected": "Determine whether setup_logging (and hence a fresh DedupWarningFilter) is invoked once per run inside a sweep worker. If not, reset the seen-set per run or instantiate the filter per run; and correct the docstring's 'resets every run/task' claim, which is true only under that condition.",
    "failure_scenario": "A sweep of 200 combinations emits warnings only for combination #1. 199 runs with suppressed unit/regime/convergence warnings (rCloud > rCloud_max, nEdge < nISM, super-critical Bonnor-Ebert) are treated as clean and the aggregate result is published. CLAUDE.md's own note that trinity leaks module-level global state in-process makes this concrete rather than hypothetical.",
    "repro": "pytest: run the logging path twice in one process (as a sweep worker would) and assert the second run's warnings are emitted.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "contested",
    "source_ids": ["S1-C-43", "S1-A-14", "S1-B-29"]
  },
  {
    "id": "S1-R-15",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 208,
    "class": "citation",
    "severity": "S4",
    "claim": "m_H = 1.6735575e-24 g is numerically acceptable (it passes the physics lens's own band and consistency test) but matches NO named convention, and the block that contains it is collectively cited to 'CODATA 2018 / IAU 2015' -- neither of which tabulates a hydrogen-atom mass.",
    "evidence": "A (recomputed): m_p + m_e - 13.6 eV/c^2 = 1.673532838e-24, so the stored value is +1.474e-05 relative. A excluded every candidate convention: it is not m_p+m_e (1.67353286e-24), not 1.008/N_A (1.673823e-24), not 1.00794/N_A (1.673724e-24), and not 1.00782503 u (1.673533e-24); stored minus the file's own m_p is 9.3558e-28 vs m_e = 9.1094e-28. C (required): the acceptance band is [1.6735e-24, 1.6738e-24] with |m_H/(m_p+m_e) - 1| < 1.2e-4 -- the stored value PASSES both. B: :194 attributes 'Values from CODATA 2018 / IAU 2015 resolutions' collectively to G, k_B, m_H, m_p, m_e, c, sigma_SB and e, while :3 separately says all constants are 'derived from astropy.units'.",
    "expected": "Per-constant provenance comments. State which convention m_H is meant to be (H-1 atom mass 1.6735328e-24, or standard atomic weight 1.6737237e-24) and set the literal to that value, or record where 1.6735575e-24 came from.",
    "failure_scenario": "Systematic rather than random: every mass-per-particle quantity scales by 1+1.5e-5 (mu_ion, mu_atom, hence sound speed by 7.4e-6, and every .param value declared in units of m_H via the unit_map at :375). Far below any physical uncertainty -- but it breaks a bit-identical regression gate and leaves the code a fixed 1.5e-5 off any published anchor, with no note saying why.",
    "repro": "python3 -c \"mp=1.67262192369e-24; me=9.1093837015e-28; Eb=13.605693123*1.602176634e-12/(2.99792458e10)**2; print((1.6735575e-24-(mp+me-Eb))/(mp+me-Eb))\"",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S1-A-01", "S1-B-10", "S1-C-15"]
  },
  {
    "id": "S1-R-16",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 220,
    "class": "coefficient",
    "severity": "S4",
    "claim": "sigma_SB, m_p, m_e are stored as truncations of the full-precision values (rounding, NOT errors), and the elementary charge e is the pre-2019 CODATA value while the block is cited as CODATA 2018. Consequence for testing: sigma_SB fails the physics lens's proposed 1e-9 assertion bar even though the code is correct.",
    "evidence": "A (recomputed): sigma_SB 5.670374e-5 vs 5.670374419e-5 (rel -7.4e-8, 7 s.f.); m_p 1.67262192e-24 vs 1.67262192369e-24 (rel -2.2e-9); m_e 9.1093837e-28 vs 9.1093837015e-28 (rel -1.6e-10); e 4.80320425e-10 vs the 2019-exact 1.602176634e-19 C x 2.99792458e9 = 4.803204713e-10 (rel -9.6e-8). A also notes e is NOT re-exported at the module level and is reachable only via CGS.e. C requires SIGMA_SB_CGS == 2*pi**5*k_B**4/(15*h**3*c**2) to 1e-9 relative, which the stored 7-s.f. literal cannot meet. C does not cover e.",
    "expected": "Store full-precision literals (sigma_SB = 5.670374419e-5, m_p = 1.67262192369e-24, m_e = 9.1093837015e-28, e = 4.803204713e-10) so the derived-formula identity holds tightly, OR keep the truncations and write the test at 1e-7 rather than C's 1e-9. Do not write the test at 1e-9 against the current literals -- it would fail on correct code. Note A's own report is self-contradictory about which *_CGS names are re-exported; only the claim that e has no flat re-export is unambiguous.",
    "failure_scenario": "No physical consequence at 1e-7. The real risk is a mis-set test tolerance that either fails on correct code (1e-9) or is loosened so far it stops catching a real drift.",
    "repro": "python3 -c \"import math; k=1.380649e-16;h=6.62607015e-27;c=2.99792458e10; s=2*math.pi**5*k**4/(15*h**3*c**2); print(s, (5.670374e-5-s)/s)\"",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-C-14", "S1-B-10"]
  },
  {
    "id": "S1-R-17",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 389,
    "class": "other",
    "severity": "S4",
    "claim": "convert2au accepts a narrow syntax subset: no parenthesised unit GROUPS (parentheses work only around exponents), no space-separated units (whitespace is stripped before tokenising, fusing tokens), and no 'yr' unit at all -- even though Mdot_au2Msunyr exists, so no .param can express Msun/yr.",
    "evidence": "A (executed): convert2au('erg/(s*cm**2)') -> UnitConversionError 'Cannot parse unit: (s'; convert2au('km s**-1') -> Unknown unit 'kms' (whitespace stripped at :360 fuses the tokens); convert2au('yr') and convert2au('Msun/yr') -> Unknown unit 'yr'; convert2au('cm**1/2') gives the misleading 'Cannot parse unit: 2' because '/' is split before exponents are parsed. C (expected, blind): its reference-value list includes 'Msun/yr' -> 1.0e+06 and 'K cm-3' -> ndens_cgs2au, and C's PISM boundary analysis assumes the K cm^-3 form is expressible. B: the docstring documents only two input contracts (None -> 1, invalid -> raise) and leaves '', whitespace-only, grouped denominators and repeated units unspecified.",
    "expected": "Either extend the vocabulary/grammar (a 'yr' entry, parenthesised groups, space tolerance) or document the accepted subset explicitly in the docstring, since .param unit strings are a trust boundary. Note the semantics are correct where the syntax is supported: 'K*cm**-3' does yield ndens_cgs2au and the parser never folds in k_B, satisfying C's PISM requirement.",
    "failure_scenario": "Low: every unsupported form raises loudly, so there is NO silent unit error here. The exposure is a .param written in natural astropy style being rejected at parse time, and the inability to express Msun/yr or a space-separated K cm-3 at all.",
    "repro": "python3 -c \"import trinity._functions.unit_conversions as u\\nfor s in ['erg/(s*cm**2)','km s**-1','Msun/yr','cm**1/2']:\\n  try: print(s, u.convert2au(s))\\n  except Exception as e: print(s,'->',e)\"",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-A-13", "S1-B-12", "S1-C-18", "S1-C-20"]
  },
  {
    "id": "S1-R-18",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 254,
    "class": "state",
    "severity": "S4",
    "claim": "Pb_au2_KcmInv and Mdot_au2Msunyr are declared module-level-only 'original definitions', so by the module's own description they sit OUTSIDE both the frozen=True protection and the __post_init__ positivity check that the module advertises as its safety mechanism.",
    "evidence": "B (comments): :254 'original definitions (derived constants that only exist at module level), not re-exports'; :59 frozen=True 'prevents accidental modification'; :140 'Verify that all constants are positive'. A confirms both are plain module-level assignments (:287, :289) and that InverseConversionConstants has no positivity check either; the check itself only enforces > 0, not correctness.",
    "expected": "Either put these two in a frozen container as well, or qualify the module docstring's 'we protect against accidental modification' to name the exceptions.",
    "failure_scenario": "Code assigns cvt.Pb_au2_KcmInv (a mutable module attribute) and every later pressure diagnostic in that process silently uses the altered value -- exactly the failure frozen=True was introduced to prevent. Higher stakes than it looks: C notes Pb_au2_KcmInv is also the natural inbound route for PISM (declared in K cm^-3), so a mutation would move the external pressure term in the EOM, not just a diagnostic.",
    "repro": "python3 -c \"import trinity._functions.unit_conversions as cvt; cvt.Pb_au2_KcmInv = 1.0; print(cvt.Pb_au2_KcmInv)\"  # no raise",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-B-13", "S1-C-08"]
  },
  {
    "id": "S1-R-19",
    "file": "trinity/_functions/simplify.py",
    "line": 502,
    "class": "other",
    "severity": "S4",
    "claim": "nmin is silently raised to 20 AFTER the `nmin >= x.size` early-return check, so a caller asking for fewer points gets more with no warning; and the stated rationale for the floor is arithmetically false.",
    "evidence": "A (executed): :496 returns everything when nmin >= x.size; :502 then does nmin = max(int(nmin), 20); _simplify(np.linspace(0,1,1000), np.sin(20*x), nmin=5) returned 26 points. B (comments): the floor is justified at :298 as 'matches the coverage-skeleton chunk count so endpoints + coverage always fit inside the budget' and at :498 'so the algorithm has enough budget for both endpoints and a meaningful coverage skeleton' -- but 2 endpoints + 20 chunks = 22 > 20. The property is actually delivered by the separate nmin-2 cap at :694. C: _COVERAGE_CHUNKS must satisfy 2 <= n <= nmin; at 20 it PASSES.",
    "expected": "Warn on the clamp (or document the floor of 20 as part of the contract), and fix the rationale text so someone changing _COVERAGE_CHUNKS or removing the nmin-2 cap is not misled.",
    "failure_scenario": "Someone raises _COVERAGE_CHUNKS while trusting the floor rationale and removes the nmin-2 cap as 'redundant', after which the mandatory set exceeds nmin at every small budget.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np,warnings; from trinity._functions.simplify import _simplify; x=np.linspace(0,1,1000); warnings.simplefilter('ignore'); print(_simplify(x,np.sin(20*x),nmin=5)[0].size)\"  # 26",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-A-17", "S1-B-22"]
  },
  {
    "id": "S1-R-20",
    "file": "trinity/_functions/simplify.py",
    "line": 625,
    "class": "numerical",
    "severity": "S4",
    "claim": "idx_dist is off by one relative to the arc-length grid: bins[j] is the bin of point j+1, so np.where(bins[:-1] != bins[1:]) records point j -- one index BEFORE each bin crossing -- and can never record the last two indices.",
    "evidence": "A: s_cum = np.cumsum(ds) has length n-1 with s_cum[j] = arc length from point 0 to point j+1 (:608); bins = (s_cum/maxdist).astype(int) (:624) therefore indexes points 1..n-1, but the returned index is the diff position j. Demonstration: _simplify([3,1,2,5,4],[9,1,4,25,16],nmin=3) returns 4 of 5 points (interior x=4 dropped) even though the effective nmin is 20, because idx_dist yields {0,1,2} instead of {1,2,3}. Single-lens: B documents the arc-length promotion tier but no off-by-one; C requires x-uniform (not arc-length) coverage and does not address this.",
    "expected": "np.where(bins[:-1] != bins[1:])[0] + 1 if the intent is 'the first point of each new arc-length bin'.",
    "failure_scenario": "Uniform arc-length coverage of the saved trajectory is systematically shifted one sample early and the tail of the curve is under-sampled. Cosmetic for a decimator, but the output points are not the ones nominally chosen.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _simplify; x=np.array([3.,1.,2.,5.,4.]); print(_simplify(x,x**2,nmin=3)[0])\"  # [3. 1. 2. 5.]",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-A-09"]
  },
  {
    "id": "S1-R-21",
    "file": "trinity/_functions/operations.py",
    "line": 55,
    "class": "numerical",
    "severity": "S4",
    "claim": "On an array with repeated values, find_nearest_lower returns the FIRST index holding the matching value rather than the last index whose value is <= the query. Value-equivalent, index-inequivalent.",
    "evidence": "A (executed): on [1,2,2,3] with v=2.5 it returns idx 1, not idx 2. Reported in A's prose, not filed by A as a finding. Not covered by B or C.",
    "expected": "If any caller uses the returned index into a PARALLEL array (SPS age -> L, cooling T -> Lambda are exactly that shape), the last-matching index is the correct choice; otherwise document that the first is returned.",
    "failure_scenario": "A plateau in the lookup axis (repeated ages or temperatures in a table) maps to the wrong row of the parallel value array. No lens identified a call site, so this is unquantified.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.operations import find_nearest_lower as f; print(f(np.array([1.,2.,2.,3.]), 2.5))\"  # 1",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-A-04"]
  },
  {
    "id": "S1-R-22",
    "file": "trinity/_functions/operations.py",
    "line": 175,
    "class": "other",
    "severity": "S4",
    "claim": "find_nearest_higher's boundary-condition comment is a verbatim copy of find_nearest_lower's, including 'the returned idx is actually higher than the value instead of the desired lower' and 'Not quite sure what to do with that for now' -- prose that cannot describe this function's branch, since for find_nearest_higher the failure is returning an index BELOW value.",
    "evidence": "B (comments): :175 duplicates :56 word for word. A confirms B's inference: the mirrored branch does return an index whose value is below the query at the top boundary (find_nearest_higher(linspace(1,5,5), 99.) -> idx 4, value 5.0). B's blind prediction -- 'the copy-paste suggests the mirrored branch was never re-derived; if the index arithmetic was copied too, find_nearest_higher may return a lower bracket at the array edges without raising' -- is confirmed.",
    "expected": "Describe the mirrored branch in its own terms. The behavioural fix is S1-R-02; this item is the documentation half.",
    "failure_scenario": "A maintainer reading the comment reasons about the wrong failure direction and 'fixes' the wrong boundary.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S1-B-02"]
  },
  {
    "id": "S1-R-23",
    "file": "trinity/_functions/operations.py",
    "line": 95,
    "class": "regime",
    "severity": "S4",
    "claim": "BOUNDARY_FRAC = 0.01 excuses monotonicity violations only at the LOW-index end of the array, never at the high-index end, and for any array of <= 100 elements the exemption covers only the very first step.",
    "evidence": "A: boundary_cut = max(1, ceil(0.01*n)), and the run classification tests `end <= boundary_cut` only. C (required): BOUNDARY_FRAC is 'the fraction of the array at EACH end where violations are excused', motivated by both integration start-up AND the xi -> 1 singular end of the Weaver profile where n ~ (1-x)^(-2/5) diverges; C's dead-guard case (f >= 0.5) does NOT apply at 0.01.",
    "expected": "If the trailing exemption is intended, test `start >= n-1-boundary_cut` as well. If it is deliberately one-sided, say so -- C's physical motivation names the trailing (singular) end explicitly.",
    "failure_scenario": "Fails SAFE: the guard is stricter than documented, so a legitimate excursion at the singular end of a Weaver profile raises MonotonicError instead of being excused. B notes such a raise is caught by get_betadelta as a penalised, retried trial, so the cost is wasted trials rather than wrong physics.",
    "repro": "",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S1-C-26", "S1-A-03"]
  },
  {
    "id": "S1-R-24",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 59,
    "class": "units",
    "severity": "S4",
    "claim": "Two naming/scoping hazards in a module whose declared purpose is unit hygiene: (a) ConversionConstants' docstring says 'All constants convert CGS -> Astronomy Units' but the class contains km->pc and km/s->pc/Myr entries, which are not CGS; (b) the internal Msun/pc/Myr system is abbreviated 'AU', colliding with the astronomical unit (1.495978707e13 cm), and that abbreviation propagates through the whole public API (convert2au, _cgs2au, _au2cgs, Pb_au2cgs, Mdot_au2Msunyr).",
    "evidence": "B: :59 vs :75 ('= cm2pc / 1e-5') and :108-:109 (km/s). A confirms km2pc and v_kms2au exist with those dimensions and are numerically correct. C independently flags the AU collision: 'note this collides with the astronomical unit, which is a real readability hazard but not a correctness one'.",
    "expected": "Reword the class docstring to 'CGS and common astronomical input units -> AU'. For the AU collision, at minimum call it out explicitly in the module docstring; both B and C rate it a readability rather than correctness hazard.",
    "failure_scenario": "A reader trusting 'All constants convert CGS' applies the km/s factor to a cm/s value and is wrong by 1e5; or a contributor reads 'convert to au' as 'convert to astronomical units' and applies or omits a 1.496e13 factor. Note the constants themselves are correct and v_kms2au == 1e5*v_cms2au holds to 1 ULP.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S1-B-08", "S1-B-14"]
  },
  {
    "id": "S1-R-25",
    "file": "trinity/_functions/simplify.py",
    "line": 25,
    "class": "other",
    "severity": "S4",
    "claim": "Two 'byte-identical' equivalence claims are asserted in comments with no cited artifact: 'Output is byte-identical to the straightforward numpy-indexed version' (:25) and 'The traversal is byte-identical to the queue version; verified against the BFS for n in {2, 3, 4, ..., 30 000}' (:656). Both accompany performance rewrites of hot paths, and CLAUDE.md rule 5 requires a committed harness/CSV for exactly this claim class.",
    "evidence": "B: :25 and :656, no test, file, or command named. A independently brute-forced the sparse-table RMQ (0 mismatches over all (lo,hi) for n in {1,2,3,5,8,17,33}) and verified the curvature algebra, but did NOT verify either byte-identical claim against a reference implementation -- so neither claim is corroborated or refuted.",
    "expected": "Per CLAUDE.md rule 5, a bit-identical claim needs a committed harness plus a value diff, and the reference implementation must still exist somewhere runnable. Add both to the pytest suite, or downgrade the comments to 'believed equivalent'.",
    "failure_scenario": "A later micro-optimisation breaks equivalence and nothing re-checks it, because the reference implementation the claim compares against is not in the tree.",
    "repro": "grep the pytest suite for a test that reimplements the naive prev/next-strictly-greater scan and the BFS bisection order and asserts equality.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S1-B-26"]
  },
  {
    "id": "S1-R-26",
    "file": "trinity/_functions/logging_setup.py",
    "line": 106,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "DedupWarningFilter suppresses every repeat of an identical WARNING+ message with no suppressed-count report, so a physics clamp that fires thousands of times is indistinguishable in the log from one that fired once.",
    "evidence": "A (executed): filter() stores (levelno, getMessage()) in self._seen and returns False on any repeat (:103-109); three identical WARNING records -> True, False, False. Attached to both console (:269) and file (:300) handlers. B: the docstring names the collapsed messages as physics diagnostics -- 'a super-critical Bonnor-Ebert sphere, nEdge < nISM, rCloud > rCloud_max' -- and states no suppressed-count mechanism. Distinct from S1-R-14, which is about cross-run leakage.",
    "expected": "Emit a final 'message X suppressed N times' summary at shutdown, or dedup with a periodic re-emit, so the frequency of a physics clamp is observable.",
    "failure_scenario": "A run in which rCloud > rCloud_max fires on every step reads as a single benign warning, hiding how much of the run is executing on clamped physics.",
    "repro": "PYTHONPATH=pkg python3 -c \"import logging; from trinity._functions.logging_setup import DedupWarningFilter as D; f=D(); r=logging.LogRecord('n',logging.WARNING,'p',1,'same',None,None); print(f.filter(r), f.filter(r), f.filter(r))\"  # True False False",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S1-A-14", "S1-B-29"]
  }
]
```
