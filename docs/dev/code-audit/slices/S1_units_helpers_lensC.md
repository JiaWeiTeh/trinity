# S1 units & helpers — Lens C (what it should be)

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

**Lens:** physics tier, blind. Inputs read: the redacted signature list for S1, and
`docs/dev/code-audit/reference/PHYSICS_SPEC.md` (§8 SPEC-090/091/092 primarily). **No `trinity/`
source, no comments, no docstrings, no other lens report.** Every number below is computed here
from CODATA/IAU primitives with a scratch arithmetic script; the derivation column is the
checkable object, not the digits.

---

## 0. Method and how to read this

The task is: given only a *name* like `Lambda_cgs2au` or `Pb_au2_KcmInv`, state the value the name
demands. I do this in three passes:

1. **Fix the primitives** (§2) — pc, Myr, M☉, k_B, G, m_H, … — from definitions, flagging where a
   choice of convention (which M☉, which year) moves the answer and by how much.
2. **Derive every conversion algebraically** from the primitives (§3). The algebra is exact; the
   digits inherit whatever M☉ the module picks. I tabulate both plausible M☉ so the reconciler can
   match either without me having to guess.
3. **State the identities** (§4). These are the strongest tests: they are true in *any* unit system
   and with *any* choice of M☉/year, so they cannot be "passed" by a compensating error, and they
   fail loudly if one member of a pair was edited without its partner.

A caution I want on the record before the table: **the digits are the weakest part of this
report.** The code may legitimately carry M☉ = 1.98892e33 (WARPFIELD/older-astropy heritage) or
1.98841e33 (IAU 2015 nominal), and the two differ by 2.6e-4 relative. A reconciler that flags a
4th-significant-figure mismatch against my table has probably found a convention difference, not a
bug. The identities in §4 are the part I will defend.

---

## 1. The natural unit system, and where cgs must survive

### 1.1 What the system must be

**SPEC-090** fixes it: internal dynamics in **`[M☉, pc, Myr]`** ("AU" in this codebase's sense;
note this collides with the *astronomical unit*, which is a real readability hazard but not a
correctness one). Temperature stays **K** in both systems — this is important and I use it below:
the conversion factor for any pure power of K is **exactly 1.0**, so a unit-string parser never
needs to know what `K^-3.5` means numerically, only that it contributes 1.

Why `[M☉, pc, Myr]` is the right choice here, physically: the problem's own scales are
`M_cl ~ 10⁴–10⁹ M☉`, `R2 ~ 0.1–500 pc`, `t ~ 0–15 Myr`, `v2 ~ 1–100 pc/Myr`. In these units every
state variable the integrator carries is O(10⁻²…10³). In cgs the same state spans `10¹⁸` (R),
`10¹³` (t), `10³⁷` (M), `10⁴⁹` (E) — and the shell EOM couples them multiplicatively. A stiff
LSODA-class solver with *relative* tolerances is insensitive to this, but the *absolute* tolerance
floor (`atol`) is not: an `atol` tuned in one system is meaningless in the other. Any absolute
tolerance anywhere in this code is therefore unit-system-dependent and is a latent bug; see the
`MONOTONIC_RTOL` / `_DEDUP_TOL_DEFAULT` items (§5.3, §5.5) where this becomes concrete.

A second, subtler reason: `G` in AU is **4.5e-3** — an O(1)-ish number. In cgs it is `6.67e-8`, and
in the shell EOM it multiplies `M_sh(M_cluster + M_sh/2)/R2²` where `M_sh ~ 10³⁹ g`. The AU system
keeps the gravity term's magnitude comparable to the pressure term's, which is exactly what you
want when they are differenced (near stall, SPEC-032, they nearly cancel).

### 1.2 What conventionally stays cgs, and why

Three families should be expected to remain cgs even inside an AU code, and each is a **mandatory
conversion boundary**:

| Family | Stays cgs because | Boundary where conversion is mandatory |
|---|---|---|
| **SPS / stellar-population drivers** (`Lbol`, `Lmech_W`, `Qi`, `pdot_W`, `Mdot`, `v_SN`) | Starburst99 and every SPS product is tabulated in `erg/s`, `s⁻¹`, `dyn`, `M☉/yr`, `cm/s`. The *table file* is cgs by construction. | At the SPS loader, once, immediately after read and **before** the `f_mass = M_cluster/sps_refmass` scaling (SPEC-073 says "multiplied by `f_mass` **after** unit conversion" — so the order is fixed and auditable). |
| **Cooling / atomic data** (`Λ(T)` in `erg cm³ s⁻¹`, `α_B` in `cm³ s⁻¹`, `C_thermal` in `erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²`, `σ_d` in `cm²`, `κ_IR` in `cm² g⁻¹`) | These are *atomic-physics* constants; the entire literature (Gnat & Ferland, Cloudy, Spitzer, Draine) tabulates them in cgs and no astro code re-tabulates them. Converting `Λ` to `M☉ pc⁵ Myr⁻³` (a 1e85 number) buys nothing and destroys the ability to eyeball it against a paper. | At the point where `Λ` meets `n²` to form a *rate*, and at the point where `C_thermal` meets `T^{7/2}/R` to form a *flux*. Either convert the whole product once (cleanest: compute `du/dt` in cgs, then apply `dudt_cgs2au`) or convert the constant once (`Lambda_cgs2au`). **Doing both is a 1e85 error; doing neither is a 1e-25 error.** Both fail loudly, which is the good case. |
| **Reported / diagnostic quantities** (`P/k_B` in `K cm⁻³`, `n` in `cm⁻³`, `v` in `km/s`, `L` in `erg/s`, `Ṁ` in `M☉/yr`) | Human-facing. A reader checks `P/k_B ~ 10⁷ K cm⁻³` against Weaver; nobody checks `2.1e3 M☉ pc⁻¹ Myr⁻²`. | At the output writer only. `Pb_au2_KcmInv`, `Mdot_au2Msunyr`, `v_au2kms` exist for exactly this and must appear **only** on the output path, never inside the RHS. |

The presence of `Lambda_cgs2au` and `c_therm_cgs2au` in the module tells me the code has chosen the
"convert the constant" route for cooling and conduction. That is a legitimate choice, but it means
the *table values themselves* must never be pre-multiplied a second time downstream.

### 1.3 The one boundary that has no conversion constant, and must

`PISM` is declared in **`K cm⁻³`** (SPEC-003, SPEC-092 item 4) — that is `P/k_B`, **not** a
pressure. The signature list contains `Pb_au2_KcmInv` (AU → K cm⁻³) but **no `KcmInv_2Pb_au`**.
So the inbound direction must be assembled as `P_ISM[AU] = PISM * ndens_cgs2au * k_B[AU]`, or
equivalently `PISM / Pb_au2_KcmInv`. Numerically, for the sweep's max `PISM = 10⁶ K cm⁻³`:

```
    P_ISM = 1e6 × 1.380649e-16 = 1.380649e-10 dyn cm⁻²
          = 1.380649e-10 × 1.5450e12 = 213.32  M☉ pc⁻¹ Myr⁻²
    (equivalently 1e6 / 4687.87 = 213.32 ✓ — the two routes must agree exactly)
```

That two-route agreement is a free, exact test.

---

## 2. Primitive constants (what each name must hold)

`PhysicalConstantsCGS` (L193) / `CGS` (L229) and the module-level `*_CGS` names (L236–243) must all
be **cgs**, and must be mutually identical — one object aliasing the other, not two hand-typed
copies.

| Name | Required value | Unit | Source / derivation | Digit I am unsure of |
|---|---|---|---|---|
| `K_B_CGS` | `1.380649e-16` | erg K⁻¹ | **Exact** by the 2019 SI redefinition (`1.380649e-23 J/K`). No uncertainty exists. | none — this is exact |
| `C_CGS` | `2.99792458e10` | cm s⁻¹ | **Exact** by definition of the metre. | none |
| `H_CGS` | `6.62607015e-27` | erg s | **Exact** by the 2019 SI (`6.62607015e-34 J s`). | none |
| `G_CGS` | `6.67430e-8` | cm³ g⁻¹ s⁻² | CODATA 2018 & 2022, `6.67430(15)e-11 m³ kg⁻¹ s⁻²`. Relative uncertainty 2.2e-5. | the 5th ("3") is CODATA-current; older codes carry `6.67408e-8` (CODATA 2014), `6.67259e-8` (CODATA 1986), or `6.674e-8`. All are acceptable to 1e-4; `6.67e-8` is not. |
| `M_P_CGS` | `1.67262192369e-24` | g | CODATA 2018 (`1.67262192369(51)e-27 kg`); CODATA 2022 `1.67262192595e-24`. | 10th digit only |
| `M_E_CGS` | `9.1093837015e-28` | g | CODATA 2018 (2022: `9.1093837139e-28`). | 10th digit only |
| `M_H_CGS` | `1.673533e-24` **or** `1.673724e-24` | g | **Two defensible values.** H-1 atom: `1.00782503207 u × 1.66053906660e-24 g/u = 1.6735328e-24`. Standard atomic weight of H (mass-averaged with D): `1.00794 u → 1.6737237e-24`. They differ by **1.1e-4 relative**. | the 5th digit — this is a *convention*, not a measurement |
| `SIGMA_SB_CGS` | `5.670374419e-5` | erg cm⁻² s⁻¹ K⁻⁴ | **Derived, exact:** `σ = 2π⁵k_B⁴/(15h³c²)`. I evaluated this from the exact `k_B`, `h`, `c` above and got `5.6703744192e-5` ✓. | none — must equal the formula |
| *(σ_T, if present)* | `6.6524587321e-25` | cm² | CODATA 2018 Thomson cross-section (2022: `6.6524587051e-25`). `= (8π/3)r_e²`, `r_e = e²/(m_e c²) = 2.8179403262e-13 cm`. In AU: `6.9868e-62 pc²`. | 9th digit |

**Not in `PhysicalConstantsCGS` but implied by the conversions:**

| Name | Required value | Unit | Derivation |
|---|---|---|---|
| pc | `3.0856775814913673e18` | cm | **Exact by IAU definition:** `1 pc = (648000/π) AU`, `AU = 1.495978707e13 cm` (exact, IAU 2012). I evaluated `648000/π = 206264.80624709636`, `× 1.495978707e13 = 3.0856775814913673e18`. Any code carrying `3.086e18` is 8e-5 low; `3.08e18` is 1.8e-3 low (→ 0.9% in `L` via the fifth-power sensitivity, SPEC-092 item 3). |
| Myr | `3.15576e13` | s | **Exact:** Julian year `= 365.25 × 86400 = 3.15576e7 s`, `× 1e6`. Tropical year gives `3.155693e13` (−2.1e-5, negligible); a 365-day year gives `3.1536e13` (**−6.9e-4**, not negligible); the shortcut `3.15e13` is −1.8e-3. |
| M☉ | `1.98892e33` *or* `1.9884099e33` | g | **Two live conventions.** IAU 2015 nominal fixes `GM☉ = 1.3271244e26 cm³ s⁻²` *exactly*; dividing by CODATA G gives `M☉ = 1.9884099e33 g` (this is astropy's `M_sun`). The older `1.98892e33` is widespread in the superbubble literature and in WARPFIELD-lineage code. **Ratio = 1.0002566** (2.6e-4). This propagates into *every* mass-bearing conversion below. |

---

## 3. Constants table — the primary deliverable

Notation: `pc ≡ 3.0856775815e18`, `Myr ≡ 3.15576e13`, `M ≡ M☉ in g`. Values quoted to 10
significant figures so the reconciler can distinguish a convention difference (digit 4–5) from a
real error. Column **A** uses `M = 1.98892e33`; column **B** uses `M = 1.9884099e33` (IAU nominal).
Where the two are identical the entry is mass-free — those rows are the ones I hold to full
precision.

### 3.1 Base length / time / mass

| Name | Value A (M=1.98892e33) | Value B (IAU M) | Unit | Derivation |
|---|---|---|---|---|
| `pc2cm` | `3.0856775815e+18` | same | cm per pc | `(648000/π) × 1.495978707e13`, exact IAU |
| `cm2pc` | `3.2407792894e-19` | same | pc per cm | `1/pc2cm` |
| `Myr2s` | `3.1557600000e+13` | same | s per Myr | `1e6 × 365.25 × 86400`, exact Julian |
| `s2Myr` | `3.1688087814e-14` | same | Myr per s | `1/Myr2s` |
| `Msun2g` | `1.9889200000e+33` | `1.9884098707e+33` | g per M☉ | convention (see §2) |
| `g2Msun` | `5.0278543129e-34` | `5.0291442159e-34` | M☉ per g | `1/Msun2g` |

### 3.2 Densities and fluxes

| Name | Value A | Value B | Unit | Derivation |
|---|---|---|---|---|
| `ndens_cgs2au` | `2.9379989461e+55` | same | (pc⁻³)/(cm⁻³) | `pc2cm³` — a number density *per cm³* becomes *per pc³* by multiplying by the number of cm³ in a pc³ |
| `ndens_au2cgs` | `3.4036771910e-56` | same | (cm⁻³)/(pc⁻³) | `cm2pc³ = 1/pc2cm³` |
| `phi_cgs2au` | `3.0047272631e+50` | same | (pc⁻²Myr⁻¹)/(cm⁻²s⁻¹) | photon flux `Φ = Q_i/(4πr²)` (SPEC-083): `pc2cm² × Myr2s = 9.5214063e36 × 3.15576e13` |
| `phi_au2cgs` | `3.3280890825e-51` | same | inverse | `1/phi_cgs2au` |
| *(mass density, if present)* | `6.7696416387e-23` | `6.7679053232e-23` | (g cm⁻³)/(M☉ pc⁻³) | `Msun2g/pc2cm³` — SPEC-091 quotes `6.7696e-23`, i.e. convention **A** |

### 3.3 Energy, luminosity, force, pressure

| Name | Value A | Value B | Unit | Derivation |
|---|---|---|---|---|
| `E_au2cgs` | `1.9015619174e+43` | `1.9010741942e+43` | erg per (M☉ pc² Myr⁻²) | `M × pc2cm² / Myr2s² = M × 9.5214063e36 / 9.9588e26` |
| `E_cgs2au` | `5.2588348075e-44` | `5.2601839688e-44` | inverse | `1/E_au2cgs` |
| `L_au2cgs` | `6.0256861023e+29` | `6.0241406007e+29` | (erg/s) per (M☉ pc² Myr⁻³) | `E_au2cgs / Myr2s` |
| `L_cgs2au` | `1.6595620532e-30` | `1.6599878161e-30` | inverse | `1/L_au2cgs` |
| `pdot_au2cgs` | `6.1625424796e+24` | `6.1609618763e+24` | dyn per (M☉ pc Myr⁻²) | `M × pc2cm / Myr2s²` |
| `pdot_cgs2au` | `1.6227068670e-25` | `1.6231231747e-25` | inverse | `1/pdot_au2cgs` |
| `F_au2cgs` | `6.1625424796e+24` | `6.1609618763e+24` | dyn per (M☉ pc Myr⁻²) | **must be identical to `pdot_au2cgs`** — force ≡ momentum rate |
| `F_cgs2au` | `1.6227068670e-25` | `1.6231231747e-25` | inverse | **must be identical to `pdot_cgs2au`** |
| `pdotdot_au2cgs` | `1.9527918725e+11` | `1.9522910096e+11` | (dyn/s) per (M☉ pc Myr⁻³) | `pdot_au2cgs / Myr2s` |
| `pdotdot_cgs2au` | `5.1208734227e-12` | `5.1221871898e-12` | inverse | `pdot_cgs2au × Myr2s` |
| `Pb_au2cgs` | `6.4723029256e-13` | `6.4706428733e-13` | (dyn cm⁻²) per (M☉ pc⁻¹ Myr⁻²) | `pdot_au2cgs / pc2cm² = M/(pc2cm × Myr2s²)` |
| `Pb_cgs2au` | `1.5450451122e+12` | `1.5454414957e+12` | inverse | `1/Pb_au2cgs` |
| `Pb_au2_KcmInv` | `4.6878699261e+03` | `4.6866675551e+03` | (K cm⁻³) per (M☉ pc⁻¹ Myr⁻²) | `Pb_au2cgs / K_B_CGS = 6.4723e-13 / 1.380649e-16` — converts a pressure to the `P/k_B` the parameter file and the literature quote |

### 3.4 Velocity, gravity, mass rate

| Name | Value A | Value B | Unit | Derivation |
|---|---|---|---|---|
| `v_kms2au` | `1.0227121650e+00` | same | (pc/Myr) per (km/s) | `1e5 × Myr2s / pc2cm = 3.15576e18/3.0856776e18` |
| `v_au2kms` | `9.7779222168e-01` | same | (km/s) per (pc/Myr) | `pc2cm/(1e5 × Myr2s)`. **SPEC-091's `0.977781` is rounded/slightly wrong in the 6th digit; the exact value is `0.9777922`.** |
| `v_cms2au` | `1.0227121650e-05` | same | (pc/Myr) per (cm/s) | `Myr2s/pc2cm` |
| `v_au2cms` | `9.7779222168e+04` | same | (cm/s) per (pc/Myr) | `pc2cm/Myr2s` |
| `G_cgs2au` | `6.7417650516e+04` | `6.7400358861e+04` | factor on `cm³g⁻¹s⁻²` → `pc³M☉⁻¹Myr⁻²` | `Msun2g × Myr2s² / pc2cm³` (length³ down, mass⁻¹ up, time⁻² up) |
| `G_au2cgs` | `1.4832910853e-05` | `1.4836716256e-05` | inverse | `1/G_cgs2au` |
| `gravPhi_cgs2au` | `1.0459401725e-10` | same | (pc²Myr⁻²) per (cm²s⁻²) | specific potential/energy: `(Myr2s/pc2cm)² = v_cms2au²` |
| `gravPhi_au2cgs` | `9.5607762878e+09` | same | inverse | `v_au2cms² = (9.7779222e4)²` |
| `grav_force_m_cgs2au` | `3.2274341420e+08` | same | (pc Myr⁻²) per (cm s⁻²) | acceleration (force per unit mass): `Myr2s²/pc2cm` |
| `grav_force_m_au2cgs` | `3.0984365784e-09` | same | inverse | `pc2cm/Myr2s²` |
| `Mdot_au2Msunyr` | `1.0000000000e-06` | same | (M☉/yr) per (M☉/Myr) | **Exactly 1e-6** by the definition of Myr. Mass cancels; time is 1e6 yr per Myr. |

### 3.5 Thermal / microphysics constants

| Name | Value A | Value B | Unit | Derivation |
|---|---|---|---|---|
| `k_B_cgs2au` | `5.2588348075e-44` | `5.2601839688e-44` | factor on `erg/K` → `M☉pc²Myr⁻²/K` | **identical to `E_cgs2au`** — K is invariant, so a per-K energy converts exactly like an energy |
| `k_B_au2cgs` | `1.9015619174e+43` | `1.9010741942e+43` | inverse | **identical to `E_au2cgs`** |
| `c_therm_cgs2au` | `5.1208734227e-12` | `5.1221871898e-12` | factor on `erg s⁻¹cm⁻¹K⁻⁷ᐟ²` → `M☉ pc Myr⁻³ K⁻⁷ᐟ²` | `E_cgs2au × Myr2s × pc2cm` = `L_cgs2au × pc2cm`. Note `erg s⁻¹ cm⁻¹ ≡ dyn s⁻¹`, so this **must equal `pdotdot_cgs2au`** |
| `c_therm_au2cgs` | `1.9527918725e+11` | `1.9522910096e+11` | inverse | **must equal `pdotdot_au2cgs`** |
| `dudt_cgs2au` | `4.8757915633e+25` | `4.8770424544e+25` | factor on `erg cm⁻³s⁻¹` → `M☉ pc⁻¹Myr⁻³` | `E_cgs2au × pc2cm³ × Myr2s` = `Pb_cgs2au × Myr2s` (since `erg cm⁻³ ≡ dyn cm⁻²`) |
| `dudt_au2cgs` | `2.0509490347e-26` | `2.0504229958e-26` | inverse | `1/dudt_cgs2au` |
| `Lambda_cgs2au` | `5.6486135076e-86` | `5.6500626672e-86` | factor on `erg cm³s⁻¹` → `M☉ pc⁵Myr⁻³` | `L_cgs2au / pc2cm³` = `Myr2s/(E_au2cgs × pc2cm³)`. Cooling *efficiency* (SPEC-081), i.e. the thing multiplied by a **density product**, not by `n` |
| `Lambda_au2cgs` | `1.7703459418e+85` | `1.7698918736e+85` | inverse | `L_au2cgs × pc2cm³` |
| `tau_cgs2au` | `3.1688087814e-14` | same | Myr per s | **Reading 1 (primary): `τ` is a timescale** (cooling / recombination / conduction time) ⇒ `tau_cgs2au ≡ s2Myr`. **Reading 2: `τ` is optical depth**, which is dimensionless ⇒ the only correct value is `1.0` and the constant should not exist at all. A constant pair was created, so Reading 1 is far more likely; but if the module holds `1.0` here, that is *also* correct under Reading 2 and the finding is a naming defect, not an arithmetic one. |
| `tau_au2cgs` | `3.1557600000e+13` | same | s per Myr | `≡ Myr2s` (Reading 1) |

### 3.6 Physical constants *expressed in AU* (what the code must obtain when it multiplies)

These are the values a correct module must *produce*, and are the sharpest end-to-end checks
because they can be cross-checked against numbers that appear in the literature.

| Quantity | Value A | Value B | Unit | Derivation / independent cross-check |
|---|---|---|---|---|
| `G` in AU | `4.499656e-3` | `4.498502e-3` | pc³ M☉⁻¹ Myr⁻² | `G_CGS × G_cgs2au`. **Cross-check:** the textbook `G = 4.300917e-3 pc M☉⁻¹ (km/s)²` (which is `GM☉_nominal/pc/1e10` — I verified `1.3271244e26/3.0856776e18/1e10 = 4.3009172e-3`), times `(km/s)²→(pc/Myr)² = 1.0227122² = 1.0459402`, gives `4.498502e-3` ✓ = column B. **SPEC-091 quotes `4.4985e-3` (column B) while quoting M☉ = 1.98892e33 (column A) — the spec's own table is internally inconsistent by 2.6e-4.** Whichever the code holds, it must be self-consistent with its own `Msun2g`. |
| `k_B` in AU | `7.2606050e-60` | `7.2624677e-60` | M☉ pc² Myr⁻² K⁻¹ | `1.380649e-16 × k_B_cgs2au`. SPEC-091's `7.261e-60` matches column A to its quoted precision. |
| `m_H` in AU | `8.414279e-58` | `8.416438e-58` | M☉ | `1.6735328e-24 × g2Msun` (H-1); with the standard-atomic-weight `m_H` it is `8.415240e-58` / `8.417400e-58` |
| `k_B/m_H` in AU | `8.6289090e-3` | `8.6289090e-3` | pc² Myr⁻² K⁻¹ | **M☉-independent** (both numerator and denominator scale as 1/M☉) ⇒ `= (k_B/m_H)_cgs × gravPhi_cgs2au = 8.249435e7 × 1.0459402e-10`. This is the single best sound-speed test: it is immune to the M☉ convention *and* to the pc/Myr rounding at the 1e-10 level. With `m_H = m_p` instead it is `8.6336083e-3` (+5.4e-4) — detectably different. |
| `C_thermal` in AU (for `6e-7` cgs) | `3.072524e-18` | `3.073312e-18` | M☉ pc Myr⁻³ K⁻⁷ᐟ² | `6e-7 × c_therm_cgs2au`; SPEC-043 fixes the cgs value |
| `σ_d` in AU (for `1.5e-21` cgs) | `4.86117e-58` | same | pc² | `1.5e-21 × cm2pc²`; SPEC-028 |
| `κ_IR` in AU (for `4 cm²/g`) | `8.35557e-4` | `8.35772e-4` | pc² M☉⁻¹ | `4 × cm2pc² × Msun2g`; SPEC-027 |
| `c` in AU | `3.0684523e5` | same | pc/Myr | `2.99792458e10 × v_cms2au` |

---

## 4. Invariant relations — the tests that survive any convention

These hold **regardless** of which M☉, which year, which `m_H`, or which unit system. They are the
strongest content in this report. Each is a one-line assertion.

### 4.1 Reciprocal pairs (25 of them)

For every stem `X` in
`{cm↔pc, s↔Myr, g↔Msun, ndens, phi, E, L, pdot, pdotdot, G, v_kms, v_cms, F, Pb, k_B, c_therm,
dudt, Lambda, tau, gravPhi, grav_force_m}`:

```
    X_cgs2au × X_au2cgs == 1        (to ≤ 4 ulp, i.e. |product − 1| < 1e-15)
```

This is exact-to-float **only if the inverse is computed as `1.0/x`**. If both members were typed
by hand from a calculator, the product lands at `1 ± 1e-6`. **Any pair whose product deviates by
more than ~1e-12 was hand-typed and one member has drifted.** This is the single cheapest test in
the whole S1 slice and it catches the highest-frequency failure mode (edit one, forget the other).

Same requirement for `cm2pc × pc2cm == 1`, `s2Myr × Myr2s == 1`, `g2Msun × Msun2g == 1`.

### 4.2 Aliases — quantities that must be *bit-identical*, not merely close

```
    F_cgs2au        == pdot_cgs2au          (force ≡ dp/dt)
    F_au2cgs        == pdot_au2cgs
    k_B_cgs2au      == E_cgs2au             (K is unit-invariant ⇒ energy/K converts as energy)
    k_B_au2cgs      == E_au2cgs
    c_therm_cgs2au  == pdotdot_cgs2au       (erg s⁻¹ cm⁻¹ ≡ dyn s⁻¹ — dimensionally the same)
    c_therm_au2cgs  == pdotdot_au2cgs
    tau_cgs2au      == s2Myr                (under Reading 1 of §3.5)
    tau_au2cgs      == Myr2s
```

The `c_therm ≡ pdotdot` identity is the least obvious and therefore the most diagnostic: if those
two differ, one of them was derived with a wrong power of `pc` or `Myr`.

### 4.3 Composition identities (each catches an exponent error)

```
    ndens_cgs2au        == pc2cm**3
    phi_cgs2au          == pc2cm**2 * Myr2s
    E_au2cgs            == Msun2g * pc2cm**2 / Myr2s**2
    L_cgs2au            == E_cgs2au * Myr2s
    pdot_au2cgs         == Msun2g * pc2cm / Myr2s**2
    pdotdot_cgs2au      == pdot_cgs2au * Myr2s
    Pb_au2cgs           == pdot_au2cgs / pc2cm**2
    Pb_au2_KcmInv       == Pb_au2cgs / K_B_CGS
    dudt_cgs2au         == Pb_cgs2au * Myr2s
    Lambda_cgs2au       == L_cgs2au / pc2cm**3
    Lambda_cgs2au * ndens_cgs2au**2 == dudt_cgs2au         ← the cooling closure
    G_cgs2au            == Msun2g * Myr2s**2 / pc2cm**3
    gravPhi_cgs2au      == v_cms2au**2
    grav_force_m_cgs2au == v_cms2au * Myr2s
    v_kms2au            == 1e5 * v_cms2au
    Mdot_au2Msunyr      == 1e-6                            ← exactly; NOT 1/Myr2s
```

The `Lambda × ndens² == dudt` closure is the one that guarantees the cooling path is
self-consistent: if `Λ` and `n` are each converted correctly but the *rate* constant was derived
independently, this identity is the only thing that catches the mismatch. Note it also encodes the
**normalisation choice** (SPEC-082): the identity as written assumes `du/dt = n_a n_b Λ` with *two*
density factors. If the code's `dudt_cgs2au` is consistent with `Lambda × ndens¹` or `ndens³`, the
cooling normalisation convention is wrong at the dimensional level, not merely at the factor-of-2
level.

### 4.4 Self-consistency of the primitives

```
    M_H_CGS  ≈ M_P_CGS + M_E_CGS      (to 1.5e-5 — I verified mp+me = 1.6735329e-24,
                                       and mp+me−13.6eV/c² = 1.6735328e-24, i.e. the
                                       electron binding energy is a 1.4e-8 effect)
    SIGMA_SB_CGS == 2π⁵ K_B_CGS⁴ / (15 H_CGS³ C_CGS²)      (exact; I evaluated 5.6703744192e-5)
    every derived conversion must use the SAME Msun2g the module exposes
    every derived conversion must use the SAME Myr2s the module exposes
    module-level *_CGS names == the corresponding fields of CGS / PhysicalConstantsCGS
    INV_CONV field-by-field == 1/CONV field-by-field
```

The "same M☉ throughout" requirement is checkable without knowing which M☉ was chosen:

```
    E_au2cgs * Myr2s**2 / pc2cm**2  ==  Msun2g        (recovers M☉ from the energy conversion)
    G_cgs2au * pc2cm**3 / Myr2s**2  ==  Msun2g        (recovers M☉ from the G conversion)
    pdot_au2cgs * Myr2s**2 / pc2cm  ==  Msun2g        (recovers M☉ from the force conversion)
```

All three must return the *same* number. A mismatch means one conversion was copied from a source
using a different M☉ — exactly the inconsistency I found in SPEC-091 itself (its `G` row assumes
`1.98841e33` while its mass row states `1.98892e33`).

---

## 5. Helper semantics — what each must do, including the edges

### 5.1 `convert2au(unit_string)` and its parser (L315, L389, L419, L431)

**Contract.** Return the multiplicative factor `f` such that `value_AU = value_in_unit_string × f`.
`convert2au("pc") == 1.0`, `convert2au("Msun") == 1.0`, `convert2au("Myr") == 1.0`,
`convert2au("K") == 1.0`.

**`split_by_slash(s)` — the single highest-risk function in this file.** For `"a/b/c"` the
mathematically correct reading in every scientific-unit convention is **left-associative
flattening**: `a·b⁻¹·c⁻¹`, i.e. *every* token after the first slash is inverted. The wrong reading —
alternating (`a/b/c → a·b⁻¹·c`, i.e. `a/(b/c)`) — is a natural implementation slip if the function
recurses and toggles `invert` at each level. The distinguishing test:

```
    convert2au("cm3/g/s2") must equal G_cgs2au = 6.7418e4
    an alternating parser returns cm2pc³ × Msun2g × Myr2s⁻² = 6.774e-49   (a 1e53 error)
```

That error is enormous and would fail loudly — the *dangerous* variant is a two-token case like
`"erg/s"` where there is nothing to alternate, so the bug hides until a three-token string appears.
`c_therm`'s `erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²` and `G`'s `cm³ g⁻¹ s⁻²` are exactly such strings.

**`split_units(s)`** must split a token into (base, exponent) handling: no exponent (`"g"` → 1),
positive (`"cm3"` → 3), explicit negative (`"cm-3"` → −3), explicit positive (`"cm+3"`), and
multi-character bases whose names contain digits or hyphens is not a case here but `Msun`/`M_sun`
is. It must **not** mis-lex `"s2"` as base `"s2"`, nor `"Msun"` as base `"M"` exponent `sun`.

**`accumulate(units, invert)`** must return `Π fᵢ^(±eᵢ)` with the sign set by `invert`. Correct
denominator rule: a token with cgs2au factor `f` and exponent `e` in the denominator contributes
`f^(−e)`. Verify with `"1/g"`: converting "per gram" to "per M☉" multiplies by `Msun2g`, i.e.
`g2Msun^(−1)` ✓.

**Edge cases the implementation must get right:**

| Input | Required behaviour | Why |
|---|---|---|
| `None` | return `1.0` | the `Optional[str]` in the signature says dimensionless parameters pass `None`; this is the documented no-op |
| `""` / whitespace-only | return `1.0` | same class; but it must be a *deliberate* branch, not a fall-through |
| unknown token, e.g. `"parsec"`, `"Mpc"`, `"solMass"`, a typo | **raise `UnitConversionError`** | This is the load-bearing one. Silently returning `1.0` for an unrecognised unit means a `.param` typo produces a run in the wrong units with **no diagnostic** — the worst failure mode in the whole slice, because the run completes and publishes numbers. `UnitConversionError` exists (L310), so the intent is clearly to raise; the audit must confirm there is no `except → 1.0` fallback and no `dict.get(tok, 1.0)`. |
| `"K"`, `"K-3.5"`, `"K7/2"` | factor exactly `1.0` for any exponent | temperature is the same unit in both systems; the parser need not even evaluate the exponent, but it must not *raise* on a fractional one, or `C_thermal`'s unit string becomes unparseable |
| case variants (`"myr"`, `"MYR"`, `"MSun"`) | either accept consistently or reject consistently | silently accepting `"myr"` but not `"Myr"` (or vice versa) creates a `.param` that works for one author and fails for another |
| composability | `convert2au("erg/s") == convert2au("erg") / convert2au("s")` | must hold for **every** composite string; this is a property test that needs no reference values at all and is the cheapest way to validate the whole parser |

**Reference values a correct parser must produce** (column A):

```
    "cm"        → 3.2407793e-19      "g"          → 5.0278543e-34
    "s"         → 3.1688088e-14      "erg"        → 5.2588348e-44
    "erg/s"     → 1.6595621e-30      "dyn"        → 1.6227069e-25
    "cm-3"      → 2.9379989e+55      "km/s"       → 1.0227122
    "cm3/g/s2"  → 6.7417651e+04      "erg/K"      → 5.2588348e-44
    "erg cm3/s" → 5.6486135e-86      "erg/cm3/s"  → 4.8757916e+25
    "cm2/g"     → 2.0888931e-04      "Msun/yr"    → 1.0e+06
    "K cm-3"    → 2.9379989e+55      "dyn/cm2"    → 1.5450451e+12
```

Note `"K cm-3"` → `2.938e55`, **not** a pressure. The `PISM` path must apply `k_B` separately
(§1.3). If `convert2au("K cm-3")` silently folded in `k_B`, the parser would be inconsistent with
its own composability property.

### 5.2 `find_nearest` / `find_nearest_lower` / `find_nearest_higher` (L19, L30, L146)

These are table-lookup helpers (SPS table by age, cooling table by T/n/Φ). Their edge behaviour
determines whether the code interpolates or silently extrapolates.

**`find_nearest(array, value)`** — `argmin |array − value|`.
- **Exact tie** (`value` equidistant from two entries): must be deterministic. `np.argmin`
  returns the **first** (lowest-index) minimum; on an ascending array that is the *smaller*
  neighbour. Either convention is defensible; what is not defensible is a tie-break that differs
  between `find_nearest` and `find_nearest_lower/higher`, because then `lower ≤ nearest ≤ higher`
  can fail.
- **Out of range**: nearest is the endpoint. This is correct *for this function* but is a silent
  clamp — the caller cannot distinguish "the query was inside the table" from "the query was
  10× past the last entry". Any use of this for the SPS age or the cooling temperature therefore
  needs an explicit range check at the call site, not here.
- **Empty array**: must raise, not return `array[0]` or `None`.
- **NaN**: `np.argmin` on an array containing NaN returns the NaN's index (NaN propagates through
  the comparison chain). A NaN in a cooling table would then be silently selected. Must be
  detected or the table must be validated at load.

**`find_nearest_lower(array, value)`** — the largest element **≤** `value`.
- The `≤` (not `<`) is required for bracketing: with strict `<`, an exact table node gives
  `lower = node−1`, `higher = node`, and the interpolation weight is 1.0 at the wrong end — an
  off-by-one that is invisible except exactly at nodes.
- **`value < min(array)`**: there is **no** lower element. The function must signal this (raise,
  or return a sentinel the caller checks). Returning `array[0]` — which is *higher* than `value` —
  violates the function's own name and postcondition, and turns an out-of-range query into a
  silent 1-sided extrapolation. **This is the classic silent-failure in this family.**
- **`value > max(array)`**: returns the last element; correct, but again the caller cannot tell.

**`find_nearest_higher(array, value)`** — the smallest element **≥** `value`; mirror-image
requirements, with the out-of-range failure at the top end.

**Symmetry requirement.** `find_nearest_lower` and `find_nearest_higher` must impose the *same*
preconditions. The line numbers show `find_nearest_lower` at L30 (before the monotonic-guard block
at L94–L143) and `find_nearest_higher` at L146 (immediately after it) — a layout that strongly
suggests only the *higher* variant validates monotonicity. If so, the two functions have different
robustness on the same input, which is a latent inconsistency: an array that raises
`MonotonicError` through one entry point sails through the other.

**Both `_lower` and `_higher` are only meaningful on a monotone array.** On an unsorted array,
"the largest element ≤ v" is still well-defined if implemented as a masked argmax, but an
implementation that assumes sortedness and uses `searchsorted` returns garbage. Which one is
correct depends on the caller; the safe contract is: validate monotone, then `searchsorted`.

### 5.3 The monotonicity guard (L68–L143, L186)

`kindof_increasing` / `kindof_decreasing` / `monotonic` must be the **non-strict** predicates:

```
    kindof_increasing(L) ⇔ ∀i: L[i] ≤ L[i+1]
    kindof_decreasing(L) ⇔ ∀i: L[i] ≥ L[i+1]
    monotonic(L)         ⇔ kindof_increasing(L) or kindof_decreasing(L)
    len(L) ≤ 1           ⇒ True   (vacuously)
    constant L           ⇒ True   (both predicates hold)
```

Using `<` instead of `≤` makes a converged plateau — which a bubble-structure integrator produces
routinely when a variable saturates — report non-monotonic, and then `MonotonicError` fires on
physically fine data.

**`_is_monotonic_or_tolerable(L, rtol, boundary_frac, max_spike_len)`** is the numerical-noise
tolerant version. `CLAUDE.md` states the motivating regime explicitly: numpy 2.1/2.2/2.4 emit
floating-point output this guard rejects while 2.0/2.3 pass. So the guard's job is to accept
violations of size ~round-off while rejecting real structure. Required properties:

1. **The tolerance must be relative.** `|L[i+1] − L[i]| ≤ rtol × max(|L[i]|, |L[i+1]|)` (or against
   the array's dynamic range). An **absolute** tolerance is a units bug: the same physical array in
   cgs vs AU differs by `1e43` (energy) or `1e12` (pressure), so an absolute threshold that is
   generous in one system is infinitely strict in the other. This is the deepest unit-hygiene claim
   in the helper set.
2. **`MONOTONIC_RTOL`** must sit strictly between machine noise and physical signal:
   `~1e-12 … 1e-6`. Lower bound: it must exceed `N_ops × eps ≈ 1e-13` for a few-hundred-step
   integration, or the guard rejects clean arithmetic. Upper bound: it must be far below the
   fractional change the profile makes between adjacent samples (a Weaver interior profile changes
   by O(1e-2) per sample), or real non-monotonicity is accepted. I would expect **1e-8 ± 2 decades**;
   anything ≥ 1e-4 is too loose to be a noise filter, anything ≤ 1e-14 too tight to help.
3. **`BOUNDARY_FRAC`** — the fraction of the array at each end where violations are excused
   (integration start-up and the `ξ → 1` singular end of the Weaver profile, SPEC-040, where
   `n ∝ (1−x)^{−2/5}` diverges). Must satisfy `0 < BOUNDARY_FRAC < 0.5`; **if it is ≥ 0.5 the two
   excused regions cover the whole array and the guard is dead code that always returns True.**
   Physically sensible: `0.01 … 0.1`.
4. **`MAX_SPIKE_LEN`** — the longest contiguous run of violating samples excused as a "spike".
   Must be a small positive integer (**1–3**). `0` disables spike tolerance entirely (then the
   parameter is dead); a value ≳ 5 lets a genuine local extremum through, which for a bubble
   pressure/temperature profile means a non-physical structure is accepted.
5. **Degenerate reduction:** `_is_monotonic_or_tolerable(L, rtol=0, boundary_frac=0,
   max_spike_len=0)` must equal `monotonic(L)`. If it does not, the tolerant version is not a
   relaxation of the strict one and the two can disagree about the *same* array for reasons
   unrelated to tolerance.
6. **Direction-agnostic:** must accept decreasing arrays with the same tolerance semantics.
   A guard that special-cases increasing and forgets decreasing passes `T(r)` and rejects `n(r)`
   (or vice versa) — and the Weaver profiles are one of each (SPEC-040).

### 5.4 `get_soundspeed(T, params)` (L189)

**Required value.**

```
    c_s = sqrt( γ k_B T / (μ m_H) )        [AU: pc/Myr]
        = sqrt( γ/μ × 8.6289090e-3 × T )   using (k_B/m_H)_AU = 8.6289090e-3 pc²Myr⁻²K⁻¹  (§3.6)
```

The coefficient `8.6289090e-3` is **M☉-convention-independent** and is the cleanest single number
to check in this whole slice.

**Anchors I computed:**

| regime | γ | μ | T | `c_s` [pc/Myr] | `c_s` [km/s] |
|---|---|---|---|---|---|
| ionized shell, isothermal | 1 | `14/22 = 0.63636` | 1e4 | 11.645 | **11.386** |
| fully-ionized, isothermal | 1 | `14/23 = 0.60870` | 1e4 | 11.906 | **11.642** |
| hot bubble, adiabatic | 5/3 | `14/23` | 1e6 | 153.7 | **150.3** |
| molecular cloud, isothermal | 1 | `14/6 = 2.3333` | 100 | 0.608 | **0.595** |

⚠️ **SPEC-055 is arithmetically self-inconsistent here and I am contradicting it.** It writes
`c_i = sqrt(k_B T_ion/(μ_ion,shell m_H))` with `μ = 14/22 = 0.636` but reports `1.166e6 cm/s`
(11.7 km/s). I get `1.13860e6 cm/s = 11.386 km/s` for `μ = 14/22`; the quoted `1.166e6` corresponds
to `μ = 14/23 = 0.6087` (I get `1.16419e6` for that). **The spec's stated μ and its stated number
disagree — and the discrepancy is precisely the per-particle-vs-per-shell-composition trap the same
document warns about in SPEC-092 item 1.** This is a spec defect, not necessarily a code defect,
but it means the audit must not use "11.7 km/s" as a reference value without re-deriving it.

**Edge cases:** `T ≤ 0` must raise rather than return NaN through `sqrt` (a NaN sound speed
propagates silently into the venting flux, SPEC-036, and into `R1`); `T` as an array must be
supported if any caller passes a profile; the `γ` used must be stated — **isothermal (`γ=1`) for
the D-type/ionization-front sound speed (SPEC-055) but adiabatic (`γ=5/3`) for the venting enthalpy
flux (SPEC-036)**, and a single function returning one of them for both callers is a modelling
error of ~29% in `c_s`.

**μ selection is the trap.** SPEC-092 item 2 lists four regimes with `n_tot/n_H` = 2.3 (bubble),
2.2 (ionized shell), 1.1 (atomic), 0.6 (molecular), and correspondingly `μ` = 14/23, 14/22, 14/11,
14/6. `get_soundspeed(T, params)` takes the whole `params` object, so the μ choice is made *inside*
— which means the caller cannot see which regime was assumed. That is a design smell: the function
should either take `mu` explicitly or select on a documented `T` threshold.

### 5.5 `simplify.py` — output downsampling (L24–L754)

This is not physics, but it decides what gets *written*, so an error here silently corrupts every
downstream figure and every audit test run on `dictionary.jsonl`.

**Overriding invariant: `_simplify` must be a *subset selection*, never a resampling.** Every
returned `(x, y)` pair must be one of the input pairs, bit-identical. If it interpolates onto a new
grid it fabricates data points that the physics never produced, and SPEC-007's force-closure test
(`M_sh dv2/dt = ΣF`) would then be evaluated on synthetic samples that satisfy no equation.

**`_prev_next_strict(y, greater)`** — for each `i`, the nearest index left/right with `y` **strictly**
greater (or strictly less). Standard monotonic stack, O(n). Requirements: sentinel `-1` / `n` when
none exists; **strictness matters for plateaus** — with `≥` a flat top makes every element of the
plateau its own bounding element and prominences collapse to 0, silently deleting flat-topped
features (e.g. a saturated `Lmech` between SN episodes).

**`_sparse_table(y, reducer)`** — shape `(⌊log₂n⌋+1, n)`, `st[0] = y`,
`st[k][i] = reducer(st[k−1][i], st[k−1][i+2^{k−1}])`. **The reducer must be idempotent**
(min/max) — a sparse table is only valid for overlapping-query-safe operations; passing `sum` or
`mean` gives silently wrong answers because the two halves of an O(1) query overlap. Must handle
`n = 1` (single row) and `n = 0`.

**`_rmq(st, lo, hi, reducer)`** — the off-by-one magnet. With a **closed** `[lo, hi]`,
`k = ⌊log₂(hi−lo+1)⌋` and the answer is `reducer(st[k][lo], st[k][hi−2^k+1])`. With a **half-open**
`[lo, hi)` it is `k = ⌊log₂(hi−lo)⌋`, `reducer(st[k][lo], st[k][hi−2^k])`. Mixing the two
conventions between `_peak_prominences`'s window construction and `_rmq`'s indexing gives an answer
that is right for most windows and wrong for windows of length exactly a power of two — a bug that
survives casual testing. **Empty range (`hi < lo`) must not silently return `st[0][lo]`**; it must
be excluded by the caller or raise.

**`_peak_prominences(y, idx)`** — prominence of peak `i` = `y[i] − max(min(y[l..i]), min(y[i..r]))`
where `l = prev_greater(i)+1`, `r = next_greater(i)−1`. Must be **≥ 0** for every peak (a negative
prominence means the windows or the reducer sense are inverted). Peaks at the array boundary take
the window out to the array end. Must return `0` (not NaN) for a peak equal to the global maximum
with no higher element on one side — the base is then that side's minimum.

**`_COVERAGE_CHUNKS` (L243)** — number of equal-width bins in **x** used to guarantee domain
coverage. Must be `≥ 2` and `≤ nmin` (default 100); if it exceeded `nmin` the coverage requirement
alone would blow the point budget. Plausible: **8–64**. The essential property is that the bins are
uniform in **x** (time/radius), not in **index** — index-uniform bins do nothing, because the
sample density in index is already whatever the solver chose.

**`_x_uniform_coverage_idx(x, pool_idx, n_chunks)`** — must return a sorted, unique subset of
`pool_idx`, at least one index per non-empty bin, and **must include the first and last samples**.
Dropping the endpoints changes the reported `t_end`, `R2_end`, `v2_end` — the exact numbers the
termination block (SPEC-105) and the phase-timeline figure (SPEC-017, which interpolates the
`v2 = 0` crossing) depend on. Empty `pool_idx` must return empty, not raise.

**`_DEDUP_TOL_DEFAULT` (L287)** — tolerance for treating consecutive samples as duplicates. **Must
be relative** (same argument as §5.3 item 1) and tiny: `1e-12 … 1e-9`. If absolute, it is
unit-system-dependent, and worse, it is *scale*-dependent within one run: `1e-9` absolute is
negligible for `R2 ~ 100 pc` but deletes every distinct sample for `Eb ~ 1e-8` in some regime.

**`_simplify(x, y, nmin=100, grad_inc=1.0, warn_below_r2=0.9, dedup_tol=...)`** must:
- return arrays of equal length; `len ≤ len(x)`;
- return the input **unchanged** when `len(x) ≤ nmin` (no simplification budget);
- preserve x-order and x-monotonicity;
- always retain index `0` and index `-1`;
- raise on `len(x) != len(y)` rather than truncating to the shorter;
- handle NaN/Inf without crashing *and* without silently deleting the flagged samples — a NaN in
  `Eb` is diagnostic information (SPEC-105's NaN inventory) and must survive to the output;
- **warn, not fail**, below `warn_below_r2`.

**R² definition (this is a real trap).** `R² = 1 − SS_res/SS_tot` where `y_pred` is the *simplified*
curve linearly interpolated back onto the **original** `x` grid, and `SS_tot = Σ(y − ȳ)²` on the
original. Computing it the other way round — interpolating the original onto the simplified grid —
is trivially exact at every retained node and reports `R² = 1` always. Separately: **constant `y`
gives `SS_tot = 0`**, so `R²` is `0/0`. A physically common case (`Lmech_SN = 0` before the first
supernova; `pdot_SN = 0`; `Qi` flat). The correct behaviour is to return `1.0` (perfect
reconstruction of a constant) or skip the check — **not** `NaN`, which then fails
`NaN < warn_below_r2` as `False` and silently suppresses the warning for exactly the arrays where a
bug would be most visible.

**`_restore(working_idx)`** must be the exact inverse of the dedup index mapping: for every
`i` in the working array, `_restore` must land on the original index whose `(x, y)` generated it.
Round-trip test: `_restore(arange(len(working)))` must reproduce the full deduped arrays.

**`_simplify_error(...) -> dict`** must report the error of the *simplified* curve against the
*original* — same direction as the R² above — and should include at minimum max-abs, max-rel,
RMSE, R², and the compression ratio. Max-**relative** error must guard `y_orig == 0`.

### 5.6 `cluster.py` (L28, L48)

**`detect_allocated_cpus()`** must report the **allocation**, not the machine. Correct precedence:

1. `SLURM_CPUS_PER_TASK`, then `SLURM_CPUS_ON_NODE` / `SLURM_JOB_CPUS_PER_NODE` (the code
   documents an `--emit-jobs` + `sbatch` workflow, so SLURM is a first-class environment);
2. cgroup quota — v2 `/sys/fs/cgroup/cpu.max` (`quota period`, `max` = unlimited), v1
   `cpu.cfs_quota_us` / `cpu.cfs_period_us` (quota `-1` = unlimited);
3. `os.sched_getaffinity(0)` (Linux; honours `taskset` and cpusets);
4. `os.cpu_count()` **last**.

Using `os.cpu_count()` / `multiprocessing.cpu_count()` first is the classic HPC bug: on a 128-core
node with a 4-core allocation it returns 128, the pool oversubscribes 32×, and every worker thrashes.
Must return `≥ 1` in all cases (including when every probe fails).

**`get_optimal_workers()`** must return `≥ 1` and `≤ detect_allocated_cpus()`. It should also be
capped by the number of jobs actually available (spawning 32 workers for 4 sweep combinations
wastes 28 process startups plus 28 copies of whatever module-level state the package holds —
`CLAUDE.md` notes trinity leaks module-level global state, which makes per-process memory
non-trivial). **Returning `0` is fatal**: `multiprocessing.Pool(0)` raises `ValueError`.

### 5.7 `logging_setup.py` (L79–L112)

Out of the physics tier, but two items have correctness consequences:

- **`DedupWarningFilter`** must key on something stable (level + logger name + `record.msg`
  *template* + module/lineno), not on the fully-formatted message. Keying on the formatted string
  means `"Pb negative at t=1.23"` and `"Pb negative at t=1.24"` are distinct and nothing is
  deduped; keying too coarsely suppresses genuinely different warnings. It must apply **only** at
  `≥ min_level` (default WARNING) so INFO/DEBUG are never silently dropped. Critically, **its
  dedup state must not persist across runs within one process** — in a `--workers N` sweep the same
  worker process executes many parameter sets, and a module-global seen-set means run #2 onward
  emit *no* warnings at all. Given `CLAUDE.md`'s explicit note that trinity leaks module-level
  global state in-process, this is a live risk, not a hypothetical.
- **`setup_logging(...)`** must be idempotent — calling it twice must not attach a second handler
  (double-logged lines), and colour codes must be suppressed for the file handler and for
  non-TTY stdout, or the log files fill with `\x1b[` escapes.

### 5.8 `extract_example_snapshots.py` — `PHASES` (L39)

Per **SPEC-010** the integrator's `current_phase` values are exactly
`("energy", "implicit", "transition", "momentum")`. **SPEC-017** is explicit that `collapse` is
constructed in post-processing and is *not* an integrator phase — so if `PHASES` contains
`"collapse"`, `_pick_phase_index` will search for a snapshot that can never exist and must return
`None`, silently producing no snapshot for that label.

---

## 6. Known traps, ranked by how silently they fail

1. **Silent `1.0` for an unrecognised unit string.** A `.param` typo (`Msol`, `parsec`, `pc3`) that
   returns `1.0` instead of raising `UnitConversionError` produces a complete, plausible-looking run
   in the wrong units. Nothing downstream can detect it. Rank 1 because the failure is *invisible*.
2. **cgs value stored where AU is needed (or vice versa).** `C_thermal = 6e-7` (cgs) used directly
   as `M☉ pc Myr⁻³ K⁻⁷ᐟ²` is off by `5.12e-12`; `k_B = 1.380649e-16` used as the AU value is off by
   `1.9e43`. The `1e43`-class errors fail loudly (overflow, immediate NaN). **The dangerous ones
   are the O(1–10³) errors**: `Pb_au2_KcmInv` (4688) and `G_cgs2au`-adjacent quantities can move a
   result by a factor of a few and still produce a run that terminates normally.
3. **Myr vs yr.** `Mdot_au2Msunyr` must be exactly `1e-6`. The two wrong answers are `1/Myr2s`
   (`3.17e-14`, i.e. M☉/s — off by `3.16e7`) and `Myr2s` (off by `3.16e19`). Both fail loudly.
   The **quiet** version is a year that isn't Julian: a 365-day year makes `Myr2s = 3.1536e13`,
   0.069% low — which through the `(L/ρ)^{1/5} t^{3/5}` scaling is a 0.04% radius error nobody
   ever sees, but it also makes `Mdot_au2Msunyr = 1e-6` *inconsistent* with `Myr2s`.
4. **μ per particle vs μ per hydrogen nucleus.** `ρ = μ_H m_H n_H` with `μ_H = 1.4`
   (mass per H nucleus, ionisation-independent) versus `P = ρ k_B T/(μ m_H)` with
   `μ = 14/23 … 14/6` (mass per particle). These are *different constants for the same symbol*.
   Interchanging them is a factor `1.4/0.609 = 2.3` in density or pressure. **SPEC-055 in the
   reference spec already contains this exact slip** (§5.4 above) — it states `μ = 14/22` and
   reports the number for `14/23`. If a spec written specifically to warn about this trap falls
   into it, the code is at real risk.
5. **pc mantissa staleness.** `3.086e18` (−8e-5), `3.08e18` (−1.8e-3), `3.09e18` (+1.1e-3). Through
   `R ∝ (L/ρ)^{1/5}`, a 1e-3 length error is a 5e-3 error in an inferred `L` (SPEC-092 item 3).
   Also: pc-in-**metres** is `3.0857e16`, and a copy-paste from an SI source is a factor-100 error —
   which, unusually, is small enough to run to completion.
6. **`G` with a right exponent and a stale mantissa.** `6.67259e-8` (CODATA 1986) vs `6.67430e-8` is
   2.6e-4 — utterly invisible in any output, and *exactly the same size* as the M☉ convention
   spread, so the two can cancel or compound. The only defence is the internal-consistency check
   in §4.4 (recover M☉ three different ways and demand agreement), which is insensitive to the
   absolute values but catches the *mixture*.
7. **`m_H` vs `m_p`.** `1.6735e-24` vs `1.6726e-24` is 5.4e-4. In `c_s ∝ 1/√(μ m_H)` that is
   2.7e-4 — invisible. It matters only as a consistency signal: if `M_H_CGS` and `M_P_CGS` are
   equal, one of them is wrong by construction.
8. **The `K cm⁻³` boundary.** `PISM` is `P/k_B`. Feeding it to the EOM as a pressure understates
   the external term by `1/k_B = 7.2e15`; feeding a pressure where `P/k_B` is expected overstates
   by the same. Both fail loudly. The **quiet** failure is applying `k_B` in cgs to an AU number
   (or double-applying it once at parse and once at use) — a factor `4688` that still produces a
   finite run.
9. **`erg s⁻¹ cm⁻¹ ≡ dyn s⁻¹` and `erg cm⁻³ ≡ dyn cm⁻²`.** These dimensional identities mean
   `c_therm` and `pdotdot`, and `dudt` and `Pb×Myr2s`, must share values. A module that derives
   them independently can get one right and one wrong, and nothing except the §4.2/§4.3 identity
   checks will notice.
10. **Absolute tolerances in `MONOTONIC_RTOL` / `_DEDUP_TOL_DEFAULT`.** A tolerance compared against
    an absolute difference is a *units* bug wearing a numerics costume: the same array in cgs and
    AU differ by up to `1e43`, so the guard's behaviour depends on which system the caller happened
    to be in. Silent in both directions (spurious `MonotonicError`, or a guard that never fires).

---

## 7. Corrections I am making to the reference spec (declared, so the reconciler can attribute)

These are places where I re-derived a SPEC-091/092/055 number and got something different. In each
case I trust my arithmetic (shown) over the spec's quoted digit, and I flag them so a code/spec
mismatch is not scored against the code.

| Spec | Spec says | I derive | Δ | Note |
|---|---|---|---|---|
| SPEC-091 | `M☉ = 1.98892e33` **and** `G_AU = 4.4985e-3` | those two are inconsistent: `1.98892e33` ⇒ `4.49966e-3`; `4.4985e-3` ⇒ `1.98841e33` | 2.6e-4 | the spec's table mixes two M☉ conventions |
| SPEC-091 | energy `1.90148e43 erg` | `1.9015619e43` (M=1.98892e33) | 4e-5 | spec used a rounded pc |
| SPEC-091 | `v = 0.977781 km/s` | `0.9777922` | 1.1e-5 | same cause |
| SPEC-091 | luminosity `6.0255e29` | `6.0256861e29` | 3e-5 | same cause |
| SPEC-091 | force `6.1623e24 dyn` | `6.1625425e24` | 4e-5 | same cause |
| SPEC-091 | pressure `6.4721e-13` | `6.4723029e-13` | 3e-5 | same cause |
| SPEC-055 | `c_i = 11.7 km/s` at `μ = 14/22` | `11.386 km/s` at `μ = 14/22`; `11.642` at `μ = 14/23` | 2.7% | **the spec's stated μ and its number disagree** — the per-particle/per-shell trap |
| SPEC-050 | `ξ_E = 0.762934` | `(250/(308π))^{1/5} = 0.7628653` | 9e-5 | `250/(308π) = 0.25836862`, spec wrote `0.258364`; `v2` coefficient is `0.4577192`, spec wrote `0.457760` |

None of these change any conclusion; they matter only because the audit's numeric anchors should be
the re-derived values.

---

```json
[
  {
    "id": "S1-C-01",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 257,
    "class": "units",
    "severity": "S1",
    "claim": "Every cgs2au / au2cgs pair must be exact multiplicative inverses, computed as 1.0/x rather than both hand-typed.",
    "evidence": "A unit conversion and its inverse are definitionally reciprocal. If both members are literal constants, an edit to one leaves the other stale and the round-trip cgs->au->cgs no longer returns the input. Computing the inverse as 1.0/x makes the product equal 1 to <=1 ulp; hand-typed pairs land at 1 +/- 1e-6 or worse.",
    "expected": "For every stem X in {cm/pc, s/Myr, g/Msun, ndens, phi, E, L, pdot, pdotdot, G, v_kms, v_cms, F, Pb, k_B, c_therm, dudt, Lambda, tau, gravPhi, grav_force_m}: abs(X_cgs2au * X_au2cgs - 1) < 1e-15. INV_CONV must be generated field-by-field from CONV, not typed.",
    "failure_scenario": "One member of a pair is updated (e.g. Msun refreshed to the IAU value) and its partner is not. A quantity converted out to cgs for a diagnostic and back into AU for the ODE drifts by 2.6e-4 per round trip; over an integration this accumulates as a slow, unattributable energy leak that no test targets.",
    "repro": "for each pair: assert abs(a*b - 1) < 1e-15",
    "confidence": "high"
  },
  {
    "id": "S1-C-02",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 148,
    "class": "units",
    "severity": "S1",
    "claim": "All mass-bearing conversions must be derived from ONE stored Msun2g; recovering M_sun three different ways must give the same number.",
    "evidence": "E_au2cgs = M*pc2cm^2/Myr2s^2, pdot_au2cgs = M*pc2cm/Myr2s^2, G_cgs2au = M*Myr2s^2/pc2cm^3 all contain M linearly. Inverting each recovers M. The two live conventions (1.98892e33 legacy, 1.9884099e33 IAU-nominal) differ by 2.6e-4, so a copy-pasted constant from a source using the other convention is detectable only this way.",
    "expected": "E_au2cgs*Myr2s**2/pc2cm**2 == G_cgs2au*pc2cm**3/Myr2s**2 == pdot_au2cgs*Myr2s**2/pc2cm == Msun2g, all to <1e-12 relative.",
    "failure_scenario": "G_cgs2au copied from a table built on the IAU nominal M_sun while Msun2g holds 1.98892e33. Gravity is then 2.6e-4 inconsistent with every other force term; the force-budget closure test (SPEC-007) fails at a level attributed to integrator tolerance rather than to a constant.",
    "repro": "assert three recovered M_sun values agree to 1e-12",
    "confidence": "high"
  },
  {
    "id": "S1-C-03",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 273,
    "class": "coefficient",
    "severity": "S2",
    "claim": "G_cgs2au must equal Msun2g * Myr2s**2 / pc2cm**3 = 6.74177e4 (legacy Msun) or 6.74004e4 (IAU Msun); G_CGS * G_cgs2au must land at 4.4997e-3 or 4.4985e-3 pc^3 Msun^-1 Myr^-2 respectively.",
    "evidence": "G has dimensions L^3 M^-1 T^-2. Converting cm^3 -> pc^3 divides by pc2cm^3; g^-1 -> Msun^-1 multiplies by Msun2g; s^-2 -> Myr^-2 multiplies by Myr2s^2. Independent cross-check: the textbook G = 4.300917e-3 pc Msun^-1 (km/s)^2 equals GM_sun_nominal/pc/1e10 = 1.3271244e26/3.0856776e18/1e10 exactly; multiplying by (km/s)^2 -> (pc/Myr)^2 = 1.0227122^2 = 1.0459402 gives 4.498502e-3.",
    "expected": "G_cgs2au = 6.7418e4 +/- 0.03%; G in AU = 4.4985e-3 to 4.4997e-3 depending on the Msun convention, and consistent with the module's own Msun2g per S1-C-02.",
    "failure_scenario": "A wrong power of pc (pc^2 instead of pc^3) makes G_cgs2au 3.086e18 too large; gravity dominates from t=0 and every run re-collapses immediately. A wrong Msun makes it quietly 2.6e-4 off.",
    "repro": "assert isclose(G_cgs2au, Msun2g*Myr2s**2/pc2cm**3, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-04",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 290,
    "class": "units",
    "severity": "S2",
    "claim": "k_B_cgs2au must be bit-identical to E_cgs2au, and k_B_au2cgs to E_au2cgs.",
    "evidence": "Temperature is K in both unit systems, so its conversion factor is exactly 1. An energy-per-kelvin therefore converts exactly as an energy. Value: 5.258835e-44 (legacy Msun) / 5.260184e-44 (IAU), giving k_B in AU = 7.2606e-60 / 7.2625e-60 Msun pc^2 Myr^-2 K^-1.",
    "expected": "k_B_cgs2au == E_cgs2au exactly; k_B_au2cgs == E_au2cgs exactly; K_B_CGS * k_B_cgs2au == 7.261e-60 to 4 s.f.",
    "failure_scenario": "k_B_cgs2au derived independently with a stray Myr2s (i.e. treated as a luminosity-per-K) is 3.16e13 off; P = n k T in the bubble is then absurd and the run fails loudly. The quiet variant is a mismatch in the last digits between the two, indicating one was hand-typed.",
    "repro": "assert k_B_cgs2au == E_cgs2au",
    "confidence": "high"
  },
  {
    "id": "S1-C-05",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 281,
    "class": "units",
    "severity": "S2",
    "claim": "F_cgs2au must be bit-identical to pdot_cgs2au (and F_au2cgs to pdot_au2cgs): force and momentum-injection rate are the same dimension.",
    "evidence": "Newton's second law: [F] = [dp/dt] = M L T^-2 = Msun pc Myr^-2 = 6.16254e24 dyn (legacy Msun). SPEC-006 lists F_grav/F_ram/F_HII/F_rad alongside pdot_total, and SPEC-007's closure sums them in one equation, so they must share a scale.",
    "expected": "F_cgs2au == pdot_cgs2au == 1.622707e-25 (legacy) / 1.623123e-25 (IAU); F_au2cgs == pdot_au2cgs == 6.16254e24 / 6.16096e24 dyn.",
    "failure_scenario": "The two are derived separately and differ; the force-budget closure test of SPEC-007 fails by a constant ratio in whichever term used the odd constant, and the stacked force-fraction figures (paper_feedback.py, paper_teaser.py) misattribute the budget.",
    "repro": "assert F_cgs2au == pdot_cgs2au and F_au2cgs == pdot_au2cgs",
    "confidence": "high"
  },
  {
    "id": "S1-C-06",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 292,
    "class": "units",
    "severity": "S2",
    "claim": "c_therm_cgs2au must equal pdotdot_cgs2au (= 5.120873e-12), because erg s^-1 cm^-1 is dimensionally identical to dyn s^-1.",
    "evidence": "SPEC-043 gives C_thermal in erg s^-1 cm^-1 K^-7/2. erg/cm = dyn, so erg s^-1 cm^-1 = dyn s^-1 = Msun pc Myr^-3 in AU. Derivation: E_cgs2au * Myr2s * pc2cm = 5.258835e-44 * 3.15576e13 * 3.0856776e18 = 5.120873e-12, which is also pdot_cgs2au * Myr2s. With C = 6e-7 cgs (SPEC-043) the AU value is 3.07252e-18.",
    "expected": "c_therm_cgs2au == pdotdot_cgs2au == 5.1209e-12; c_therm_au2cgs == pdotdot_au2cgs == 1.95279e11.",
    "failure_scenario": "c_therm_cgs2au derived with pc2cm in the wrong sense (divide instead of multiply) is off by pc2cm^2 = 9.5e36. The conduction closure T_b^{7/2} = a P R^2/(C t) (SPEC-042) then gives a bubble temperature wrong by 9.5e36^(2/7), and delta = (2/7)(2a-b-1) no longer reproduces -6/35.",
    "repro": "assert c_therm_cgs2au == pdotdot_cgs2au",
    "confidence": "high"
  },
  {
    "id": "S1-C-07",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 296,
    "class": "units",
    "severity": "S1",
    "claim": "The cooling closure Lambda_cgs2au * ndens_cgs2au**2 == dudt_cgs2au must hold exactly — it encodes that the volumetric rate is Lambda times TWO density factors.",
    "evidence": "Lambda has units erg cm^3 s^-1 (SPEC-081, a cooling *efficiency*), and du/dt = n_a n_b Lambda has units erg cm^-3 s^-1. Hence Lambda_cgs2au = L_cgs2au/pc2cm^3 = 5.648614e-86 and dudt_cgs2au = Pb_cgs2au*Myr2s = 4.875792e25, and 5.648614e-86 * (2.937999e55)^2 = 4.875792e25.",
    "expected": "abs(Lambda_cgs2au*ndens_cgs2au**2/dudt_cgs2au - 1) < 1e-12. Lambda_cgs2au = 5.6486e-86 (legacy Msun), Lambda_au2cgs = 1.77035e85, dudt_cgs2au = 4.87579e25, dudt_au2cgs = 2.05095e-26.",
    "failure_scenario": "Lambda converted as if it were a plain luminosity (missing the /pc2cm^3) makes L_cool wrong by 2.9e55. If instead the density product is applied with one factor of n rather than two, L_cool scales wrongly with density and the energy->momentum transition time (SPEC-013, the code's headline prediction) moves with cloud density in a way that looks like physics.",
    "repro": "assert isclose(Lambda_cgs2au*ndens_cgs2au**2, dudt_cgs2au, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-08",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 287,
    "class": "units",
    "severity": "S2",
    "claim": "Pb_au2_KcmInv must equal Pb_au2cgs / K_B_CGS = 4687.87 (legacy Msun) / 4686.67 (IAU), and must be used only on the output/diagnostic path.",
    "evidence": "P/k_B in K cm^-3 is the astronomers' pressure unit (SPEC-092 item 4; PISM is declared in it). Pb_au2cgs = Msun2g/(pc2cm*Myr2s^2) = 6.472303e-13 dyn cm^-2 per AU pressure; dividing by k_B = 1.380649e-16 gives 4687.87.",
    "expected": "Pb_au2_KcmInv == Pb_au2cgs / K_B_CGS to 1e-12 relative; value 4.6879e3. Cross-check: PISM = 1e6 K cm^-3 must map to 213.32 AU pressure by both PISM/Pb_au2_KcmInv and PISM*ndens_cgs2au*k_B_AU.",
    "failure_scenario": "Pb_au2_KcmInv hard-coded from a stale k_B or a different Msun drifts from Pb_au2cgs/K_B_CGS. Reported P_b/k_B values are then a few 0.01% off — invisible — but if the same constant is used INBOUND for PISM, the external pressure term in the EOM is off by the same factor and the stall radius shifts.",
    "repro": "assert isclose(Pb_au2_KcmInv, Pb_au2cgs/K_B_CGS, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-09",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 289,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Mdot_au2Msunyr must be exactly 1e-6.",
    "evidence": "Msun/Myr -> Msun/yr: mass cancels, and 1 Myr = 1e6 yr by definition of the prefix, regardless of which year (Julian/tropical) is used, provided the SAME year defines Myr2s. No physical constant enters.",
    "expected": "Mdot_au2Msunyr == 1e-6 exactly. NOT 1/Myr2s (3.1688e-14, which is Msun/s), NOT Myr2s.",
    "failure_scenario": "1/Myr2s stored here reports mass-loss rates 3.16e7 too small; a reader comparing Mdot_SN against SB99 sees ~1e-13 Msun/yr and concludes the wind is negligible. Fails loudly to a careful reader, silently to a plotting script.",
    "repro": "assert Mdot_au2Msunyr == 1e-6",
    "confidence": "high"
  },
  {
    "id": "S1-C-10",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 259,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Myr2s must be exactly 3.15576e13 s (Julian year x 1e6) and pc2cm exactly 3.0856775815e18 cm (IAU definition).",
    "evidence": "Julian year = 365.25*86400 = 3.15576e7 s exactly, the IAU standard for astronomical timekeeping. pc = (648000/pi) AU with AU = 1.495978707e13 cm exactly (IAU 2012); I evaluated 648000/pi = 206264.80624709636 and the product = 3.0856775814913673e18. A 365-day year gives 3.1536e13 (-6.9e-4); '3.15e13' is -1.8e-3. '3.086e18' for pc is -8e-5; '3.08e18' is -1.8e-3.",
    "expected": "Myr2s == 3.15576e13; pc2cm == 3.0856775814913673e18 (or at least 3.0857e18, i.e. correct to 1e-5).",
    "failure_scenario": "A truncated pc propagates through R ~ (L/rho)^{1/5} t^{3/5} into a 5x-amplified error in any inferred L (SPEC-092 item 3), and through pc^3 into ndens_cgs2au with 3x the relative error. Never large enough to notice, always large enough to break a bit-identical regression gate.",
    "repro": "assert Myr2s == 1e6*365.25*86400",
    "confidence": "high"
  },
  {
    "id": "S1-C-11",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 298,
    "class": "units",
    "severity": "S3",
    "claim": "tau_cgs2au/tau_au2cgs must be s2Myr/Myr2s if tau is a timescale; if tau denotes optical depth the pair is dimensionless and both must be exactly 1.0.",
    "evidence": "Optical depth is dimensionless and needs no conversion, so the existence of a conversion pair implies tau is a time (cooling / recombination / conduction timescale). Any other reading has no consistent factor.",
    "expected": "tau_cgs2au == s2Myr == 3.1688088e-14 and tau_au2cgs == Myr2s == 3.15576e13. (If the stored values are 1.0, the constant is an optical depth and the naming is the defect.)",
    "failure_scenario": "If tau is an optical depth but carries the time factor, every optical depth is scaled by 3.17e-14 and f_abs = 1-exp(-tau) collapses to 0 -- direct radiation pressure silently vanishes (SPEC-026). If tau is a time but carries 1.0, cooling times are reported in seconds and compared against Myr dynamical times.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S1-C-12",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 300,
    "class": "units",
    "severity": "S3",
    "claim": "gravPhi_cgs2au must equal v_cms2au**2 = 1.0459402e-10, and grav_force_m_cgs2au must equal v_cms2au*Myr2s = 3.2274341e8.",
    "evidence": "Specific gravitational potential has units of velocity squared (cm^2 s^-2 -> pc^2 Myr^-2), so its conversion is the square of the velocity conversion. Gravitational force per unit mass is an acceleration (cm s^-2 -> pc Myr^-2), so its conversion is velocity-conversion times Myr2s. Both are Msun-independent, which makes them exactly checkable.",
    "expected": "gravPhi_cgs2au == v_cms2au**2 == 1.0459402e-10; gravPhi_au2cgs == v_au2cms**2 == 9.5607763e9; grav_force_m_cgs2au == 3.2274341e8; grav_force_m_au2cgs == 3.0984366e-9.",
    "failure_scenario": "A potential converted with the acceleration factor (or vice versa) is off by pc2cm ~ 3e18; the escape-speed check v_esc = sqrt(2G(M_cl+M_sh)/R2) (SPEC-032) then classifies every run as escaping or none.",
    "repro": "assert gravPhi_cgs2au == v_cms2au**2",
    "confidence": "high"
  },
  {
    "id": "S1-C-13",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 275,
    "class": "coefficient",
    "severity": "S3",
    "claim": "v_kms2au = 1.0227122 and v_au2kms = 0.9777922 (SPEC-091's 0.977781 is wrong in the 6th digit).",
    "evidence": "1 km/s = 1e5 cm/s; in pc/Myr that is 1e5*Myr2s/pc2cm = 3.15576e18/3.0856775815e18 = 1.022712165. Its reciprocal is 0.977792222. I recomputed both to 10 digits.",
    "expected": "v_kms2au == 1.0227122 (7 s.f.), v_au2kms == 0.9777922, v_cms2au == 1.0227122e-5, v_au2cms == 9.7779222e4; and v_kms2au == 1e5*v_cms2au exactly.",
    "failure_scenario": "km/s and pc/Myr differ by only 2.3% (SPEC-092 item 6), so omitting the conversion entirely produces plots that look right and are 2.3% wrong -- the hardest possible error to see by eye.",
    "repro": "assert isclose(v_kms2au, 1e5*Myr2s/pc2cm, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-14",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 236,
    "class": "coefficient",
    "severity": "S3",
    "claim": "The PhysicalConstantsCGS fields must hold current CODATA/exact values in cgs, and the module-level *_CGS names must be the same objects/values as the dataclass fields.",
    "evidence": "K_B_CGS = 1.380649e-16 and H_CGS = 6.62607015e-27 and C_CGS = 2.99792458e10 are EXACT by the 2019 SI. G_CGS = 6.67430e-8 (CODATA 2018/2022). M_P_CGS = 1.67262192369e-24, M_E_CGS = 9.1093837015e-28. SIGMA_SB_CGS = 5.670374419e-5, which I verified equals 2*pi^5*k_B^4/(15 h^3 c^2) = 5.6703744192e-5 from the exact constants.",
    "expected": "SIGMA_SB_CGS == 2*pi**5*K_B_CGS**4/(15*H_CGS**3*C_CGS**2) to 1e-9 relative; M_H_CGS ~= M_P_CGS + M_E_CGS to 1.5e-5; every module-level *_CGS equals the corresponding CGS dataclass field.",
    "failure_scenario": "Two hand-maintained copies (dataclass + module-level) diverge after an update, and which value a call site sees depends on which name it imported. Silent, and unreproducible across refactors.",
    "repro": "assert G_CGS == CGS.G and K_B_CGS == CGS.k_B (etc.)",
    "confidence": "high"
  },
  {
    "id": "S1-C-15",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 238,
    "class": "coefficient",
    "severity": "S3",
    "claim": "M_H_CGS must be the hydrogen ATOM mass (1.673533e-24 g for H-1, or 1.673724e-24 g using the standard atomic weight), not the proton mass.",
    "evidence": "m(H-1) = 1.00782503207 u * 1.66053906660e-24 g/u = 1.6735328e-24 g. Independently, m_p + m_e = 1.6735329e-24 g, and subtracting the 13.6 eV binding mass (2.42e-32 g) gives 1.6735328e-24 -- the three agree to 1.4e-8. The standard-atomic-weight variant (1.00794 u) gives 1.6737237e-24, 1.1e-4 higher; both appear in astro codes.",
    "expected": "M_H_CGS in [1.6735e-24, 1.6738e-24]; M_H_CGS != M_P_CGS; M_H_CGS/(M_P_CGS+M_E_CGS) within 1.2e-4 of 1.",
    "failure_scenario": "M_H_CGS set to the proton mass is 5.4e-4 low; rho = mu_H m_H n_H and c_s = sqrt(gamma k T/(mu m_H)) both shift by ~3-5e-4. Far too small to see, large enough to break any bit-identical gate and to place the code 5e-4 off any published Weaver anchor.",
    "repro": "assert abs(M_H_CGS/(M_P_CGS+M_E_CGS) - 1) < 1.2e-4",
    "confidence": "high"
  },
  {
    "id": "S1-C-16",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 315,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "convert2au must raise UnitConversionError on any unrecognised unit token; it must never fall back to 1.0.",
    "evidence": "A unit string comes from a .param file written by a human. A typo ('Msol', 'parsec', 'pc3', 'ergs/s') that silently yields factor 1.0 produces a numerically complete run whose inputs are in the wrong system, with no diagnostic anywhere in the output. UnitConversionError exists at L310, which establishes that raising is the intended contract. CLAUDE.md lists input validation at trust boundaries as explicitly not-lazy territory, and the .param parser is exactly such a boundary.",
    "expected": "convert2au('bogus') raises UnitConversionError. No dict.get(tok, 1.0), no bare except returning 1.0. convert2au(None) and convert2au('') return 1.0 via an explicit, tested branch.",
    "failure_scenario": "A sweep .param has 'mCloud # UNIT: Msol'. Every run in the sweep treats 1e7 g-equivalents as 1e7 Msun (or the reverse), the grid completes, figures are produced, and nothing in metadata.json records that a unit was never recognised.",
    "repro": "pytest: assert raises(UnitConversionError, convert2au, 'not_a_unit')",
    "confidence": "high"
  },
  {
    "id": "S1-C-17",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 389,
    "class": "exponent",
    "severity": "S1",
    "claim": "split_by_slash must flatten left-associatively: every token after the first slash is inverted ('a/b/c' -> a*b^-1*c^-1), NOT alternating ('a/(b/c)').",
    "evidence": "Scientific unit strings are universally read as flat products of powers. The distinguishing case is the 3-token string: convert2au('cm3/g/s2') must equal G_cgs2au = cm2pc^3 * Msun2g * Myr2s^2 = 6.74177e4. An alternating parser returns cm2pc^3 * Msun2g * Myr2s^-2 = 6.774e-49, off by ~1e53. Two-token strings ('erg/s') cannot distinguish the two implementations, so the bug hides until a three-token unit appears -- and G ('cm3/g/s2') and C_thermal ('erg/s/cm/K3.5', SPEC-043) are exactly such strings.",
    "expected": "convert2au('cm3/g/s2') == G_cgs2au; convert2au('erg/cm3/s') == dudt_cgs2au = 4.87579e25; convert2au('erg/s') == L_cgs2au = 1.65956e-30.",
    "failure_scenario": "Any three-or-more-token unit is parsed with the wrong sign on the third exponent. For G this is a 1e53 error that fails loudly; for a unit whose third token is dimensionless-in-AU (e.g. K), the bug produces a correct answer and lies dormant until a new unit string is added.",
    "repro": "assert isclose(convert2au('cm3/g/s2'), G_cgs2au, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-18",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 431,
    "class": "units",
    "severity": "S2",
    "claim": "convert2au must be multiplicatively composable: convert2au('A/B') == convert2au('A')/convert2au('B') for every A, B in the vocabulary.",
    "evidence": "Unit conversion is a homomorphism from the free abelian group on base units into the positive reals. Any parser that satisfies this property for all pairs is necessarily correct on exponents and slash handling; any parser that violates it for even one pair has a structural bug. This is a property test needing no reference values.",
    "expected": "For the full token vocabulary: isclose(convert2au(f'{a}/{b}'), convert2au(a)/convert2au(b), rel_tol=1e-12); and convert2au('X2') == convert2au('X')**2; and any pure power of K contributes exactly 1.0 (so 'K-3.5' must parse without raising).",
    "failure_scenario": "A special-cased composite (e.g. 'erg/s' handled by a lookup entry rather than by composition) drifts from the primitives after an M_sun update -- the composite keeps the old value while 'erg' and 's' get the new one.",
    "repro": "property test over the token vocabulary",
    "confidence": "high"
  },
  {
    "id": "S1-C-19",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 419,
    "class": "other",
    "severity": "S3",
    "claim": "split_units must correctly lex multi-character bases with trailing digits and explicit signs: 'cm3'->(cm,3), 'cm-3'->(cm,-3), 'g'->(g,1), 'Msun'->(Msun,1), 's2'->(s,2).",
    "evidence": "The base names in this vocabulary include letters only (cm, g, s, K, pc, erg, dyn, Msun, Myr, yr, km), so the lex rule 'longest leading alphabetic run is the base, the remainder is a signed integer exponent, empty remainder means 1' is unambiguous. The hazard is a naive regex that treats 'Msun' as base 'M' or 'cm-3' as base 'cm-'.",
    "expected": "Exponent defaults to 1 when absent; leading '+' and '-' both accepted; no token is silently dropped. An unlexable token routes to UnitConversionError, not to exponent 0.",
    "failure_scenario": "'cm-3' lexed with exponent +3 makes ndens_cgs2au its own reciprocal -- a 8.6e110 error, loud. 'cm-3' lexed as exponent 0 makes it 1.0 -- silent, and the number density is off by 2.9e55.",
    "repro": "assert convert2au('cm-3') == ndens_cgs2au",
    "confidence": "medium"
  },
  {
    "id": "S1-C-20",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 287,
    "class": "units",
    "severity": "S2",
    "claim": "convert2au('K cm-3') must return ndens_cgs2au (2.938e55), not a pressure factor; the k_B multiplication for PISM must happen explicitly at the call site.",
    "evidence": "K is unit-invariant so contributes 1.0; cm^-3 contributes pc2cm^3. PISM is declared in K cm^-3 (SPEC-003, SPEC-092 item 4), i.e. P/k_B, so the parser cannot know it should apply k_B -- doing so would also violate composability (S1-C-18). Numeric check: PISM = 1e6 K cm^-3 -> 1e6*1.380649e-16 = 1.380649e-10 dyn cm^-2 -> 213.32 Msun pc^-1 Myr^-2, which must equal 1e6/Pb_au2_KcmInv = 1e6/4687.87 = 213.32.",
    "expected": "convert2au('K cm-3') == ndens_cgs2au; and the two routes to P_ISM in AU (via k_B, and via 1/Pb_au2_KcmInv) agree to 1e-12.",
    "failure_scenario": "k_B applied twice (once folded into the unit string, once at use) puts the external pressure 7.2e15 too low, so the ISM confinement term vanishes and no run ever stalls; applied zero times it is 7.2e15 too high and every run stalls immediately.",
    "repro": "assert isclose(PISM_au_via_kB, PISM/Pb_au2_KcmInv, rel_tol=1e-12)",
    "confidence": "high"
  },
  {
    "id": "S1-C-21",
    "file": "trinity/_functions/operations.py",
    "line": 30,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "find_nearest_lower must signal (raise or return a checkable sentinel) when value < min(array); returning array[0] -- which is HIGHER than value -- violates the postcondition.",
    "evidence": "The function's contract is 'the largest element <= value'. When no such element exists the correct behaviours are to raise or to return a sentinel. Returning the array minimum silently converts an out-of-table query into a one-sided extrapolation with no diagnostic. This matters because the callers are table lookups (SPS by cluster age, cooling by T/n/Phi), and SPEC-085 notes the bundled tables have restricted coverage.",
    "expected": "find_nearest_lower(a, v) with v < a.min() raises or returns a sentinel the caller must check; the mirror requirement holds for find_nearest_higher with v > a.max(); empty array raises rather than indexing.",
    "failure_scenario": "The bubble temperature or cooling query drifts below the CIE table's 1e4 K floor. The lookup silently clamps to the first row, L_cool is evaluated at the wrong temperature, and the (L_gain-L_loss)/L_gain transition trigger (SPEC-013) fires at the wrong time -- which is the code's headline prediction.",
    "repro": "pytest: find_nearest_lower(array([1.,2.,3.]), 0.5) must not return 1.0 silently",
    "confidence": "high"
  },
  {
    "id": "S1-C-22",
    "file": "trinity/_functions/operations.py",
    "line": 19,
    "class": "numerical",
    "severity": "S3",
    "claim": "find_nearest / _lower / _higher must use a consistent tie-break and a consistent inclusive comparison (<= and >=, not < and >), so that lower <= nearest <= higher always holds and an exact table node brackets to itself.",
    "evidence": "With strict '<' in find_nearest_lower, a query exactly on a table node returns node-1 while find_nearest_higher returns node; the interpolation weight is then 1.0 at the wrong end. Exact nodes are not rare -- the SPS table's own t values are queried directly. Ties in find_nearest (equidistant neighbours) must resolve deterministically; np.argmin returns the lowest index, which on an ascending array is the smaller value.",
    "expected": "find_nearest_lower(a, a[k]) == a[k]; find_nearest_higher(a, a[k]) == a[k]; for any v inside the range, lower(v) <= nearest(v) <= higher(v); the tie-break is documented and identical across the three functions.",
    "failure_scenario": "A silent off-by-one in table bracketing that is exactly zero-error everywhere except at nodes, so it never shows up in a smooth-function test but biases every lookup that lands on a grid point.",
    "repro": "pytest over an ascending array with queries at nodes and midpoints",
    "confidence": "medium"
  },
  {
    "id": "S1-C-23",
    "file": "trinity/_functions/operations.py",
    "line": 68,
    "class": "numerical",
    "severity": "S2",
    "claim": "kindof_increasing/kindof_decreasing/monotonic must be NON-strict (<= / >=); a constant or plateaued sequence, and any sequence of length <= 1, must be monotonic.",
    "evidence": "Strict comparison reports a converged plateau as non-monotonic. Plateaus are routine in this code: a saturated bubble-structure variable, Lmech_SN = 0 before the first supernova, Qi flat over an SPS interval. A guard built on the strict predicate then raises MonotonicError on physically fine data.",
    "expected": "monotonic([1,1,1]) is True; monotonic([]) and monotonic([x]) are True; monotonic([1,2,2,3]) is True; monotonic([3,2,2,1]) is True; monotonic([1,3,2]) is False.",
    "failure_scenario": "Runs abort with MonotonicError in exactly the regimes where a variable has converged -- i.e. preferentially in the well-behaved cases -- and the failure is attributed to the numpy version rather than to the predicate.",
    "repro": "pytest: assert monotonic([1.0,1.0,1.0])",
    "confidence": "high"
  },
  {
    "id": "S1-C-24",
    "file": "trinity/_functions/operations.py",
    "line": 94,
    "class": "units",
    "severity": "S1",
    "claim": "MONOTONIC_RTOL must be applied RELATIVELY (|d| <= rtol*max(|a|,|b|) or against the array's dynamic range), never against an absolute difference.",
    "evidence": "The same physical array differs by up to 1e43 between cgs and AU (energy) or 1e12 (pressure). An absolute threshold is therefore a unit-system-dependent predicate: generous in one system, infinitely strict in the other. The name says RTOL, which fixes the intent; the risk is an implementation that compares the raw difference to the constant.",
    "expected": "The guard's verdict is invariant under scaling the whole array by any positive constant: _is_monotonic_or_tolerable(L) == _is_monotonic_or_tolerable(1e20*L) for every L.",
    "failure_scenario": "A bubble profile in AU passes the guard; the identical profile expressed in cgs (or simply a run with a 1e5x larger cloud) trips it. Diagnosed as a numpy/version problem -- CLAUDE.md already records exactly this symptom -- and 'fixed' by loosening the tolerance, which then disables the guard for small-magnitude arrays.",
    "repro": "pytest: assert guard(L) == guard(1e20*L) for a noisy monotone L",
    "confidence": "high"
  },
  {
    "id": "S1-C-25",
    "file": "trinity/_functions/operations.py",
    "line": 94,
    "class": "numerical",
    "severity": "S3",
    "claim": "MONOTONIC_RTOL must sit between accumulated round-off and physical signal: roughly 1e-12 to 1e-6, with 1e-8 the natural choice.",
    "evidence": "Lower bound: an integration of a few hundred steps accumulates ~N*eps ~ 1e-13 relative, so a tolerance below that rejects clean arithmetic. Upper bound: adjacent samples of a Weaver interior profile (SPEC-040, T ~ (1-x)^{2/5}) differ by O(1e-2) relative, so a tolerance at or above 1e-4 accepts real non-monotonic structure. CLAUDE.md states the guard's purpose is to absorb numpy-version-dependent FP output, which is a round-off-scale effect.",
    "expected": "1e-12 <= MONOTONIC_RTOL <= 1e-6.",
    "failure_scenario": "Too tight: spurious MonotonicError on specific numpy patch versions (the documented symptom). Too loose: a genuinely non-monotonic bubble density or temperature profile is accepted, isobaricity (T9 in SPEC section 11) silently breaks, and P_b becomes r-dependent inside a solver that assumes it is not.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S1-C-26",
    "file": "trinity/_functions/operations.py",
    "line": 95,
    "class": "deadcode",
    "severity": "S2",
    "claim": "BOUNDARY_FRAC must be strictly less than 0.5 (sensibly 0.01-0.1), and MAX_SPIKE_LEN a small positive integer (1-3).",
    "evidence": "BOUNDARY_FRAC excuses violations within that fraction of each end of the array. Two excused regions of fraction f cover the whole array once f >= 0.5, at which point the guard returns True unconditionally and is dead code. MAX_SPIKE_LEN excuses a contiguous run of violating samples; 0 disables the mechanism (making the parameter dead), while a value >~5 admits a genuine local extremum rather than a single-sample glitch.",
    "expected": "0 < BOUNDARY_FRAC < 0.5 (expected 0.01-0.1); 1 <= MAX_SPIKE_LEN <= 3. Also: _is_monotonic_or_tolerable(L, rtol=0, boundary_frac=0, max_spike_len=0) must reduce exactly to monotonic(L).",
    "failure_scenario": "BOUNDARY_FRAC = 0.5 makes the monotonic guard a no-op. Every non-monotonic bubble profile passes; the very failure mode the guard was written to catch (and which CLAUDE.md documents as version-sensitive) goes undetected, and the run produces a profile that violates isobaricity.",
    "repro": "assert 0 < BOUNDARY_FRAC < 0.5 and MAX_SPIKE_LEN >= 1",
    "confidence": "high"
  },
  {
    "id": "S1-C-27",
    "file": "trinity/_functions/operations.py",
    "line": 99,
    "class": "regime",
    "severity": "S3",
    "claim": "_is_monotonic_or_tolerable must be direction-agnostic: it must apply identical tolerance semantics to decreasing sequences as to increasing ones.",
    "evidence": "SPEC-040 gives the Weaver interior as T(r) = T_b (1-x)^{2/5} (decreasing in x... increasing toward the CD depending on parameterisation) and n(r) = n_b (1-x)^{-2/5} (the opposite sense). Both are checked by the same guard. An implementation that special-cases the increasing branch and reuses it for the decreasing branch without negating the tolerance sense will be strict on one profile and permissive on the other.",
    "expected": "_is_monotonic_or_tolerable(L) == _is_monotonic_or_tolerable(-L) for every L; and == _is_monotonic_or_tolerable(L[::-1]).",
    "failure_scenario": "The density profile is guarded and the temperature profile is not (or vice versa), so half the bubble-structure output is unvalidated while the code appears to be checking both.",
    "repro": "pytest: guard(L) == guard(-L) == guard(L[::-1])",
    "confidence": "medium"
  },
  {
    "id": "S1-C-28",
    "file": "trinity/_functions/operations.py",
    "line": 146,
    "class": "state",
    "severity": "S3",
    "claim": "find_nearest_lower and find_nearest_higher must impose the same preconditions -- if one validates monotonicity via the guard, so must the other.",
    "evidence": "The two are exact mirror images and are used as a bracketing pair. The signature layout places find_nearest_lower at L30, before the monotonic-guard block (L94-L143), and find_nearest_higher at L146, immediately after it -- a strong hint that only the latter validates. Asymmetric validation means the same array raises MonotonicError through one entry point and is silently accepted through the other.",
    "expected": "Both functions validate (or both do not). If validation exists, the same array produces the same verdict through either entry point.",
    "failure_scenario": "A corrupted (non-monotonic) cooling or SPS table is caught when bracketed from above and silently mis-bracketed when bracketed from below, so whether the bug is detected depends on which side of the table the query lands.",
    "repro": "pytest: feed a non-monotonic array to both and compare behaviour",
    "confidence": "medium"
  },
  {
    "id": "S1-C-29",
    "file": "trinity/_functions/operations.py",
    "line": 189,
    "class": "units",
    "severity": "S1",
    "claim": "get_soundspeed must return pc/Myr and satisfy c_s = sqrt(gamma/mu * 8.6289090e-3 * T), i.e. (k_B/m_H) in AU = 8.6289090e-3 pc^2 Myr^-2 K^-1.",
    "evidence": "c_s = sqrt(gamma k_B T/(mu m_H)). In AU, (k_B/m_H) = (1.380649e-16/1.6735328e-24) * gravPhi_cgs2au = 8.249435e7 * 1.0459402e-10 = 8.6289090e-3. This ratio is INDEPENDENT of the M_sun convention (both numerator and denominator carry 1/M_sun), so it is exactly checkable. Anchors I computed: isothermal at T=1e4 with mu=14/23 gives 11.642 km/s; with mu=14/22 gives 11.386 km/s; adiabatic (gamma=5/3) at T=1e6 with mu=14/23 gives 150.3 km/s.",
    "expected": "get_soundspeed(1e4, params_ionized)*v_au2kms in [11.3, 11.7] km/s depending on the mu selected; get_soundspeed(T)/sqrt(T) constant; using m_p instead of m_H shifts the coefficient to 8.6336083e-3 (+5.4e-4), which is detectable.",
    "failure_scenario": "A k_B or m_H taken in cgs while T is in K and the result is labelled pc/Myr gives a sound speed off by ~1e21 -- loud. The quiet failure is a mu drawn from the wrong region (SPEC-092 item 2 lists four: 14/23, 14/22, 14/11, 14/6), which is a factor up to 1.9 in c_s and feeds directly into the venting enthalpy flux (SPEC-036) and the D-type clock (SPEC-055).",
    "repro": "pytest: assert isclose(get_soundspeed(1e4,p)**2*mu/(gamma*1e4), 8.6289090e-3, rel_tol=1e-6)",
    "confidence": "high"
  },
  {
    "id": "S1-C-30",
    "file": "trinity/_functions/operations.py",
    "line": 189,
    "class": "regime",
    "severity": "S2",
    "claim": "get_soundspeed must make its gamma explicit: isothermal (gamma=1) for the ionization-front / D-type sound speed, adiabatic (gamma=5/3) for the venting enthalpy flux; one function serving both callers with a single gamma is a ~29% modelling error.",
    "evidence": "sqrt(5/3) = 1.29. SPEC-055 needs the isothermal c_i of photoionized gas for the D-type expansion; SPEC-036 needs c_s = sqrt(gamma P_b/rho_b) with gamma=5/3 for the enthalpy flux through the vent area. These are different sound speeds for different physics.",
    "expected": "The gamma used is either a parameter of the call or documented per call site; gamma_adia = 5/3 comes from default.param and must not be silently applied to the isothermal case.",
    "failure_scenario": "The 29% error lands in L_leak = (1-C_f) 4 pi R2^2 c_s (5/2) P_b. With C_f = 1.0 (the default) the term is off entirely, so the bug is invisible in every fiducial run and only appears in the C_f < 1 runs -- i.e. the ones a user turns on deliberately and trusts.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S1-C-31",
    "file": "trinity/_functions/operations.py",
    "line": 189,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "get_soundspeed must reject T <= 0 rather than returning NaN through sqrt.",
    "evidence": "sqrt of a negative float returns NaN without raising in numpy (with a RuntimeWarning that is routinely filtered). A NaN sound speed propagates into L_leak and into any timestep control that uses it, and NaN comparisons are all False, so downstream guards silently pass.",
    "expected": "T <= 0 raises ValueError (or the caller is required to have validated); the function never returns NaN for finite input.",
    "failure_scenario": "A transient negative bubble temperature during a stiff step yields NaN c_s, which contaminates E_b, and the run terminates with a NaN inventory (SPEC-105) that points at E_b rather than at the sound speed.",
    "repro": "pytest: raises(ValueError, get_soundspeed, -1.0, params)",
    "confidence": "medium"
  },
  {
    "id": "S1-C-32",
    "file": "trinity/_functions/simplify.py",
    "line": 290,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "_simplify must be a subset selection -- every returned (x,y) must be bit-identical to an input pair -- and must always retain the first and last samples.",
    "evidence": "If the simplifier resamples onto a new grid it fabricates points that no physics produced. Every downstream audit test operates on the written output: SPEC-007's force closure (M_sh dv2/dt = sum F) would then be evaluated on interpolated samples that satisfy no equation, and SPEC-017's collapse detection interpolates the v2=0 crossing from these same samples. Dropping the endpoints changes the reported t_end / R2_end / v2_end that the termination block (SPEC-105) records.",
    "expected": "Every returned x is present in the input x (exact float equality); indices 0 and -1 are always retained; output length <= input length; x-order preserved; len(x)!=len(y) raises.",
    "failure_scenario": "Interpolated output points make the recorded force budget fail to close by an amount that scales with the local curvature. The audit attributes the residual to integrator tolerance and searches the solver for a bug that lives in the output writer.",
    "repro": "pytest: assert set(x_simp).issubset(set(x_orig)) and x_simp[0]==x_orig[0] and x_simp[-1]==x_orig[-1]",
    "confidence": "high"
  },
  {
    "id": "S1-C-33",
    "file": "trinity/_functions/simplify.py",
    "line": 290,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The R^2 used by warn_below_r2 must interpolate the SIMPLIFIED curve onto the ORIGINAL x grid, and must handle SS_tot == 0 (constant y) by returning 1.0 rather than NaN.",
    "evidence": "Interpolating the original onto the simplified grid is exact at every retained node by construction and reports R^2 = 1 always -- the check would be vacuous. Separately, constant y gives SS_tot = Sum(y - ybar)^2 = 0 and R^2 = 1 - 0/0 = NaN; NaN < warn_below_r2 evaluates False, so the warning is suppressed precisely for the arrays most likely to indicate a bug. Constant series are common here: Lmech_SN and pdot_SN are identically zero before the first supernova.",
    "expected": "R^2 computed as 1 - SS_res/SS_tot with y_pred = interp(x_orig, x_simp, y_simp); SS_tot == 0 returns 1.0; the warning path is reachable in a test.",
    "failure_scenario": "A simplifier that discards a real feature reports R^2 = 1 (wrong direction) or NaN (constant series), the warning never fires, and the published figure is missing structure that existed in the integration.",
    "repro": "pytest: constant y array must not produce NaN R^2; a deliberately over-decimated array must trip the warning",
    "confidence": "high"
  },
  {
    "id": "S1-C-34",
    "file": "trinity/_functions/simplify.py",
    "line": 113,
    "class": "numerical",
    "severity": "S2",
    "claim": "_rmq must use one consistent interval convention with its callers, and must reject empty ranges rather than returning st[0][lo].",
    "evidence": "For a closed [lo,hi], k = floor(log2(hi-lo+1)) and the answer is reducer(st[k][lo], st[k][hi-2**k+1]); for half-open [lo,hi) it is k = floor(log2(hi-lo)) and reducer(st[k][lo], st[k][hi-2**k]). Mixing the two gives correct answers for most window lengths and wrong answers for windows whose length is exactly a power of two -- a bug that survives random testing. An empty range (hi < lo, which arises when a peak sits at an array boundary) must not silently return the single element st[0][lo].",
    "expected": "_rmq agrees with a brute-force reducer over the same window for all (lo,hi) pairs on a random array of length ~100, including hi-lo+1 equal to 1,2,4,8,16; empty ranges are excluded by the caller or raise.",
    "failure_scenario": "Prominences are computed against a window that is off by one element. Points near features are dropped or retained slightly wrongly; the effect is a small, systematic, direction-dependent bias in which samples survive to the output -- invisible in any single plot.",
    "repro": "pytest: brute-force cross-check of _rmq over all windows of a random array",
    "confidence": "medium"
  },
  {
    "id": "S1-C-35",
    "file": "trinity/_functions/simplify.py",
    "line": 86,
    "class": "other",
    "severity": "S3",
    "claim": "_sparse_table is valid only for idempotent reducers (min/max); it must not be used with sum/mean.",
    "evidence": "The O(1) sparse-table query covers [lo,hi] with two overlapping blocks of length 2^k. Overlap is harmless for idempotent operations (min(a,min(a,b)) = min(a,b)) and silently wrong for additive ones, which would double-count the overlap. The table shape must be (floor(log2 n)+1, n) with st[0] = y and st[k][i] = reducer(st[k-1][i], st[k-1][i+2**(k-1)]).",
    "expected": "Callers pass only np.minimum/np.maximum (or min/max); n=0 and n=1 are handled without indexing past the end.",
    "failure_scenario": "A later change passes a non-idempotent reducer for a new feature; the results are wrong only for windows that are not exact powers of two, so the bug looks like noise.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S1-C-36",
    "file": "trinity/_functions/simplify.py",
    "line": 123,
    "class": "sign",
    "severity": "S3",
    "claim": "_peak_prominences must return values >= 0, using the HIGHER of the two side-minima as the base, with windows bounded by the strictly-greater neighbours.",
    "evidence": "Standard prominence definition: from the peak, extend left and right until the signal strictly exceeds the peak or the array ends; the base is max(min_left, min_right); prominence = y[peak] - base. Using min(min_left, min_right) instead inflates the prominence of every peak on a monotone flank; inverting the subtraction gives negative prominences. Strictness in _prev_next_strict matters for plateaus: with >= a flat top bounds itself and prominence collapses to 0, silently deleting flat-topped features.",
    "expected": "all(prominences >= 0); a peak at the global maximum returns the larger side-minimum as base rather than NaN; a flat-topped peak has non-zero prominence.",
    "failure_scenario": "Flat-topped features (a saturated Lmech between SN episodes, a plateaued Qi) get zero prominence and are the first samples discarded -- so the simplifier preferentially deletes exactly the intervals where a driver is constant and a physics check would be easiest.",
    "repro": "pytest: assert (_peak_prominences(y, idx) >= 0).all() on random and plateaued arrays",
    "confidence": "medium"
  },
  {
    "id": "S1-C-37",
    "file": "trinity/_functions/simplify.py",
    "line": 287,
    "class": "units",
    "severity": "S2",
    "claim": "_DEDUP_TOL_DEFAULT must be a RELATIVE tolerance in the range ~1e-12 to 1e-9.",
    "evidence": "Same argument as MONOTONIC_RTOL: an absolute tolerance is unit-system-dependent (energy differs by 1e43 between cgs and AU) and scale-dependent within a single run (1e-9 absolute is negligible for R2 ~ 100 pc and deletes every distinct sample of a quantity whose scale is 1e-8). The purpose of a dedup tolerance is to collapse samples that are equal to within round-off, which is inherently relative.",
    "expected": "1e-12 <= _DEDUP_TOL_DEFAULT <= 1e-9, applied as |a-b| <= tol*max(|a|,|b|) (or via np.isclose with rtol).",
    "failure_scenario": "An absolute tolerance collapses all samples of a small-magnitude quantity to a single point; the written trajectory for that variable is a constant, and the audit reads it as physics.",
    "repro": "pytest: dedup verdict must be invariant under scaling the array by 1e20",
    "confidence": "medium"
  },
  {
    "id": "S1-C-38",
    "file": "trinity/_functions/simplify.py",
    "line": 246,
    "class": "other",
    "severity": "S3",
    "claim": "_x_uniform_coverage_idx must bin uniformly in x (not in index), return a sorted unique subset of pool_idx, and include the domain endpoints; _COVERAGE_CHUNKS must satisfy 2 <= n <= nmin.",
    "evidence": "The function's name states coverage in x. Binning by index is a no-op, because the sample density in index is already whatever the adaptive solver chose -- the entire purpose is to guarantee that a region where the solver took large steps (e.g. the quiescent late momentum phase) still contributes points. If _COVERAGE_CHUNKS exceeded nmin (default 100) the coverage requirement alone would exhaust the point budget.",
    "expected": "2 <= _COVERAGE_CHUNKS <= 100, plausibly 8-64; returned indices are a sorted unique subset of pool_idx; at least one index per non-empty x-bin; empty pool_idx returns empty without raising.",
    "failure_scenario": "Index-binned coverage leaves the late-time trajectory (where the solver takes long steps) represented by a handful of points; the momentum-phase and collapse portions of every published figure are under-sampled while the early energy phase is over-sampled.",
    "repro": "pytest: on x with a strongly non-uniform sample density, assert every x-bin is represented",
    "confidence": "medium"
  },
  {
    "id": "S1-C-39",
    "file": "trinity/_functions/simplify.py",
    "line": 481,
    "class": "state",
    "severity": "S3",
    "claim": "_restore must be an exact inverse of the dedup index mapping: restoring the full working index range must reproduce the deduped arrays exactly.",
    "evidence": "_simplify operates on a deduped working array; _restore maps working indices back to originals. If the mapping is off by the number of removed duplicates (a cumulative offset rather than a lookup), the error grows along the array and the last points map to the wrong originals -- which are exactly the terminal samples the termination block reports.",
    "expected": "_restore(arange(len(working))) reproduces the deduped (x,y) exactly; _restore of the selected indices yields x values present in the original array (this is the same invariant as S1-C-32 seen from the other side).",
    "failure_scenario": "A cumulative-offset bug shifts late samples by a few indices. R2(t) is written with y values belonging to slightly different t. The trajectory looks smooth and is wrong by a step size; nothing in the output records it.",
    "repro": "pytest: round-trip identity on an array containing duplicates",
    "confidence": "medium"
  },
  {
    "id": "S1-C-40",
    "file": "trinity/_functions/simplify.py",
    "line": 754,
    "class": "other",
    "severity": "S3",
    "claim": "_simplify_error must measure the simplified curve against the original grid (same direction as S1-C-33) and must guard division by zero in the max-relative-error term.",
    "evidence": "Measuring in the opposite direction is exact at retained nodes and reports zero error. Max-relative error against y_orig == 0 is a division by zero; zero values are routine (Lmech_SN, pdot_SN before the first supernova, v2 at the turnaround, F_ram at t=0).",
    "expected": "Returns at least max_abs_error, max_rel_error, rmse, r2, compression ratio; y_pred = interp(x_orig, x_simp, y_simp); relative error uses a guarded denominator (max(|y|, eps) or masked) and never returns inf/NaN for finite input.",
    "failure_scenario": "The error dict reports 0.0 or NaN for every array, the diagnostic is trusted, and a lossy simplification ships unnoticed.",
    "repro": "pytest: y_orig containing exact zeros must not produce inf/NaN",
    "confidence": "medium"
  },
  {
    "id": "S1-C-41",
    "file": "trinity/_functions/cluster.py",
    "line": 28,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "detect_allocated_cpus must consult SLURM env vars, then cgroup quota, then os.sched_getaffinity, and only then os.cpu_count(); it must return >= 1.",
    "evidence": "CLAUDE.md documents an --emit-jobs + sbatch workflow, so SLURM is a first-class environment. os.cpu_count() reports the physical node, not the allocation: on a 128-core node with a 4-core allocation it returns 128. os.sched_getaffinity(0) honours taskset/cpusets; cgroup v2 exposes /sys/fs/cgroup/cpu.max as 'quota period' (or 'max'), v1 as cpu.cfs_quota_us / cpu.cfs_period_us with -1 meaning unlimited.",
    "expected": "SLURM_CPUS_PER_TASK (then SLURM_CPUS_ON_NODE / SLURM_JOB_CPUS_PER_NODE) take precedence; cgroup quota is consulted; os.cpu_count() is the last resort; the return is always >= 1 even when every probe fails.",
    "failure_scenario": "A 4-core SLURM allocation spawns 128 workers. Each thrashes, the sweep runs ~30x slower than serial, and the job is killed by the scheduler for exceeding its time limit -- diagnosed as 'trinity is slow' rather than as an oversubscription bug.",
    "repro": "pytest with SLURM_CPUS_PER_TASK monkeypatched: assert detect_allocated_cpus() == that value",
    "confidence": "medium"
  },
  {
    "id": "S1-C-42",
    "file": "trinity/_functions/cluster.py",
    "line": 48,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "get_optimal_workers must return an int in [1, detect_allocated_cpus()], and should additionally be capped by the number of jobs.",
    "evidence": "multiprocessing.Pool(0) raises ValueError; a negative value raises too. Exceeding the allocation oversubscribes (S1-C-41). Exceeding the job count wastes process startups plus one copy of the package's module-level state per worker -- CLAUDE.md records that trinity leaks module-level global state, so per-process footprint is not negligible.",
    "expected": "1 <= get_optimal_workers() <= detect_allocated_cpus(); never 0 or negative; when a job count is available it is also an upper bound.",
    "failure_scenario": "A single-CPU container where a 'leave one core free' heuristic computes 1-1 = 0; Pool(0) raises and the sweep fails at startup with an error that names multiprocessing rather than the heuristic.",
    "repro": "pytest with detect_allocated_cpus monkeypatched to 1: assert get_optimal_workers() >= 1",
    "confidence": "medium"
  },
  {
    "id": "S1-C-43",
    "file": "trinity/_functions/logging_setup.py",
    "line": 79,
    "class": "state",
    "severity": "S2",
    "claim": "DedupWarningFilter's seen-set must not persist across runs within one process, and must apply only at >= min_level.",
    "evidence": "In a --workers N sweep one worker process executes many parameter sets sequentially. A module-level or instance-level seen-set that is never reset means run #2 onward emit no warnings at all -- every physics warning after the first run is silently dropped. CLAUDE.md explicitly warns that trinity leaks module-level global state in-process, which makes this concrete rather than hypothetical. The min_level default of WARNING must be respected so INFO/DEBUG are never deduped away.",
    "expected": "The filter state is reset per run (or the filter is instantiated per run); records below min_level are always passed through; the dedup key is stable (level + logger + message template + module/lineno), not the fully formatted message.",
    "failure_scenario": "A sweep of 200 combinations emits warnings only for combination #1. 199 runs with suppressed unit/regime/convergence warnings are treated as clean, and the sweep's aggregate result is published.",
    "repro": "pytest: run the logging path twice in one process and assert the second run's warnings are emitted",
    "confidence": "medium"
  },
  {
    "id": "S1-C-44",
    "file": "trinity/_functions/logging_setup.py",
    "line": 112,
    "class": "other",
    "severity": "S4",
    "claim": "setup_logging must be idempotent (no duplicate handlers on repeated calls) and must not emit ANSI colour codes to files or to a non-TTY stream.",
    "evidence": "Repeated setup on the same logger is the standard cause of doubled log lines; in a sweep, setup is plausibly called once per run in the same process. ColoredFormatter escapes belong only on an interactive stdout; written to a file they corrupt any downstream grep/parse of the log.",
    "expected": "Calling setup_logging twice leaves the same number of handlers; use_colors is forced off for the file handler and when stdout is not a TTY.",
    "failure_scenario": "Log files fill with \\x1b[ escapes and every line is duplicated N times for N runs in the worker; log-based diagnostics become unusable at exactly the scale where they matter.",
    "repro": "pytest: call setup_logging twice, assert len(logger.handlers) is unchanged",
    "confidence": "medium"
  },
  {
    "id": "S1-C-45",
    "file": "trinity/_functions/extract_example_snapshots.py",
    "line": 39,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "PHASES must contain only the integrator's real current_phase strings: energy, implicit, transition, momentum -- not 'collapse'.",
    "evidence": "SPEC-010 gives the phase sequence and the literal strings the outputs use. SPEC-017 states explicitly that 'collapse' is constructed in post-processing by splitting the final momentum interval at the interpolated v2=0 crossing; it is not an integrator phase and no snapshot ever carries it.",
    "expected": "PHASES == ('energy','implicit','transition','momentum') (order and membership); _pick_phase_index returns None for any phase absent from the run rather than raising or returning index 0.",
    "failure_scenario": "PHASES includes 'collapse'; _pick_phase_index searches for a value that cannot exist, returns None (or 0), and either no snapshot or the FIRST snapshot of the run is written under the 'collapse' label -- an example file that misrepresents the state it claims to show.",
    "repro": "",
    "confidence": "medium"
  }
]
```
