# S9 cooling — reconciled

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

Reconciliation of three blind lens reports on slice **S9 — Cooling: CIE, non-CIE, and the net
cooling curve**. I read only `S9_cooling_lensA.md`, `S9_cooling_lensB.md`, `S9_cooling_lensC.md`.
No source, no tables, no prose input, no other audit doc.

Raw input: 20 (A) + 32 (B) + 23 (C) = **75 candidates → 21 reconciled findings**. 11 raw candidates
are explicitly **refuted or demoted** by another lens; they are listed in §7 with the reason.

---

## 1. Coverage table

| Sub-area | A (does) | B (claims) | C (should) | Reconcilable? |
|---|---|---|---|---|
| Net-rate expression, all 3 branches | full transcription, line-level | branch structure only | derived spec | **yes** — A is decisive |
| Density factor, CIE branch | `chi_e * ndens**2` (`:164`, `:187`) | **documented zero times** | must be `n_e n_H` | **no** — see §2 |
| Density factor, non-CIE branch | **none applied** | cube unit self-contradictory | cube must be the signed volumetric net | **no** — see §2 |
| Meaning of the `ndens` argument | asserts "total number density" (inference) | "1/pc³ code units" only | SPEC-003: hydrogen nuclei | **contested** — §2.3 |
| `chi_e` value / definition | scalar from `.param`, value unseen | never mentioned | `n_e/n_H` = 1.2 hot, 1.1 in H II | **blind spot in all three** |
| CIE table normalisation | not readable from slice | not stated | G&F2012 ⇒ per `n_e n_H` [recalled] | **blind spot in all three** |
| non-CIE `.dat` column units | inferred volumetric from dimensional balance | `erg cm³/s` (a Λ) vs `erg cm⁻³ s⁻¹` (a rate) — both | CLOUDY `save cooling` is volumetric [recalled] | **blind spot in all three** |
| CIE interpolant construction | **outside slice** — type/bounds policy unknown | admits "does not support extrapolation" | log–log linear or PCHIP, `√T` above top | partial |
| CIE table provenance | — | 4 libraries, 2 with identical text, no CLOUDY version | G&F / S&D normalisations differ | B only |
| non-CIE cube build & fill | full (axes, rounding, exact-equality fill, NaN) | axes "rounded, makes things easier" | monotone axes, `(min,max)` limits | **yes** |
| Cube `.npy` cache | keyed on filename only; `np.save` ragged | "does the cube already exist?" | key on `(Z, rot, path)` | A decisive |
| Seam / blend | linear ramp, endpoints frozen at cutoffs | "interpolate between two Λ values" | cutoffs must partition; step ⇒ normalisation fingerprint | **yes** — §3 |
| Sign convention | `-1 *` in **all three** returns | documented in **1 of 3** branches | same convention both sides | **yes** — §4 |
| Off-grid T / n / Φ / age / Z | 3 different policies, exhaustive | T only; nothing on n or Φ | per-axis spec + record every clamp | **yes** — §5 |
| Metallicity | `get_Lambda` ignores it; non-CIE selects by file | "selects/validates"; auto-pin at `Z==0.15` | exactly one mechanism, never two | **yes** |
| Units / cgs↔AU conversions | verified arithmetic, `n` and `Φ` both converted | docstring internally consistent | 4.876e25, 2.938e55, 2.999e50 | **yes, agree** |
| Call sites of `get_dudt` | out of slice | — | bubble vs shell matters | **blind spot in all three** |

**The three blind spots that matter** — `chi_e`'s value, the CIE table header, and the `.dat` column
scaling — are all *outside* every lens's assigned reading. That, not disagreement, is why §2 cannot
close.

### Lens A: execution-verified vs read-only

A ran repros on **library semantics**, not on trinity. Weight accordingly.

- **Execution-verified (numpy 1.26.4 / scipy 1.17.1):** `RegularGridInterpolator` defaults
  (`bounds_error=True`, no clamp, no fill); RGI returns `NaN` *silently* for any query in a cell
  touching a NaN vertex; RGI raises "must be strictly ascending or descending" on a duplicated axis
  entry; `np.save` of the ragged `[3×1-D, 2×3-D]` list raises `ValueError` on numpy ≥ 1.24;
  `max()`/`min()` on an empty selection raises; the flat-3-list query returns shape-(1,);
  `(3.0857e18)³ = 2.938e55` and `dudt_cgs2au = 4.877e25` reproduce exactly; `get_fileage` on a
  3-digit exponent returns `1e10`.
- **Read-only transcription (high confidence, but unexecuted):** every line reference, the branch
  conditions, the `-1 *` on all three returns, `chi_e * ndens**2`, the absence of any density factor
  on the non-CIE branch, the clamp at `:130-131`.
- **Inference, not evidence:** "`ndens` … is the total number density" (A never saw a call site);
  ".dat `cool`/`heat` must be volumetric" (dimensional necessity, not observation); "cells a decade
  wide in n, T and phi" — **contradicted by B's `(33, 21, 22)` cube shape**, see §7.

---

## 2. Density-factor resolution

### 2.1 Branch by branch

| branch | table normalisation (what the code *assumes*) | factor the code applies | product | verdict |
|---|---|---|---|---|
| **A — non-CIE cube** (`Tmin ≤ log T ≤ Tcut_nCIE`) | must be **volumetric** `erg cm⁻³ s⁻¹`, density folded into the table | **none — no `n²`, no `n_e n_H`, no `chi_e`** (A §1) | table value verbatim | **correct iff the `.dat` columns are volumetric.** If they are a coefficient (`erg cm³ s⁻¹`, B's own docstring), the branch is short by `n²` — up to **10⁸** across a 10⁻²–10⁴ cm⁻³ axis |
| **B — CIE** (`log T ≥ Tcut_CIE`) | must be a coefficient `erg cm³ s⁻¹` | **`chi_e · ndens²`** (`:164`) | `chi_e · n²` | **undecided** — 1.0, 1.2, 4.41 or 5.29 depending on two unseen facts |
| **C — bridge** (`Tcut_nCIE < log T < Tcut_CIE`) | both | `(1−w)·[none] + w·[chi_e · ndens²]` | mixed | inherits branch B's error, **weighted by `w ∈ (0,1)`** — a *ramping* multiplicative error, not a constant one |

The two branches carry the density dependence on **opposite sides of the table boundary**: inside
the table (A) and outside it (B). All three lenses agree this is a real modelling choice, and all
three agree **nothing in the code, and nothing in the prose, states it**.

### 2.2 What the number would be, using Lens C's ratios

Correct volumetric rate (C §1.1): `n_e n_H Λ`, with `n_e = 1.2 n_H`, `n_tot = 2.3 n_H` for fully
ionised cosmic gas (`y = n_He/n_H = 0.1`). The code computes `chi_e · ndens²`. Therefore

```
    error factor  =  chi_e · (ndens/n_H)² / 1.20
```

| if `ndens` is… | and `chi_e` is… | code/correct | C's label |
|---|---|---|---|
| `n_H` | 1.2 (`= n_e/n_H`, ionised) | **1.00** | correct |
| `n_H` | 1.1 (`= n_e/n_H`, H II layer) | 0.917 | ~8 % under-cool |
| `n_H` | 1.0 | 0.833 | T2, 17 % under-cool |
| **`n_tot`** | **1.0** | **4.41** | **T1 — C's headline** |
| **`n_tot`** | **1.2** | **5.29** | T4 |
| `ρ/m_H` (=1.4 `n_H`) | 1.0 | 1.633 | T3 |

If the CIE table is per `n_H²` rather than per `n_e n_H`, multiply every row by 1.2.

### 2.3 Why the three lenses cannot close it, and the one fact that does

- **A** transcribed the factor exactly (`chi_e * ndens**2`) and then *asserted* `ndens` is the total
  number density — but A saw no call site and no `.param` schema, and explicitly wrote "If `chi_e`
  is intended as `n_e/n_tot` then the product is `n_e n_tot`; nothing in this slice pins that down."
  That caveat is the honest reading; **A's "total number density" gloss is an inference and must not
  be treated as observation.**
- **C** reads SPEC-003 as declaring `nCore`/`nISM` to be **hydrogen-nuclei** densities. If the caller
  passes `n_H` and `chi_e = 1.2`, then `chi_e · n_H² = n_e n_H` — **exactly right**. The presence of
  a `chi_e` factor at all, and C's own note that SPEC-029 uses `chi_e_shell = 1.1` (which is
  `n_e/n_H` for singly-ionised He, not `n_e/n_tot`), is positive evidence that the author intended
  `n_e n_H` and got it.
- **B** contributes nothing either way: the multiplier is documented **zero times**. Per the brief,
  that is a documentation gap (S3), not evidence of a bug — but it is precisely why a maintainer
  cannot settle this by reading.

**Verdict: undecided, and it does not need guessing — it needs one lookup.** In priority order:

1. **`chi_e` in `trinity/_input/` (`default.param` + the schema/defaults module): its default value
   and its one-line description.** A value of **1.2 or 1.1** means it is `n_e/n_H` and the code
   intends `n_e n_H`; a value of **~0.52** would mean `n_e/n_tot`; **1.0** means no electron-fraction
   correction at all and the pairing is wrong by 0.833 (if `ndens` is `n_H`) or **4.41** (if it is
   `n_tot`). This single quantity discriminates every row of the table above.
2. **What the caller passes as `ndens`** — the call sites of `get_dudt` in the bubble-structure /
   shell paths: `n_H` or `n_tot`. Combined with (1), the answer is exact.
3. **The header of the bundled CIE file** named by `path_cooling_CIE` under `lib/default/CIE/`
   (default per B: the Gnat & Ferland 2012 library): per `n_e n_H` or per `n_H²`. Worth ×1.2.

And for the **non-CIE** branch, one independent check settles it:

4. **In a bundled `opiate_cooling_*.dat`, hold `(temp, phi)` fixed and scan the `ndens` column.**
   If `cool` scales ≈ `n²`, the column is **volumetric** and the code (which applies no factor) is
   right. If `cool` is ~flat in `n`, the column is a **coefficient** and the non-CIE branch is short
   by `n²` — an error of up to **10⁸** over the axis, which would be the largest defect in the slice.
   B's docstring (`read_cloudy.py:23`, "Cooling rate is in units of [erg cm3 / s]") asserts the
   second; A's dimensional balance and C's recollection of CLOUDY `save cooling` both point to the
   first. **The prose and the arithmetic disagree; the table decides.**

### 2.4 Size of the error if it is wrong

Per C §1.4: at bubble conditions (`n_H = 10⁻²`, `T = 10⁷`) `t_cool ≈ 1.3 Gyr`, so essentially all of
`L_cool` comes from the thin conduction front — a multiplicative normalisation error therefore
rescales *the quantity that fires the energy→momentum transition* (SPEC-013's
`(L_gain − L_loss)/L_gain ≤ 0.05`), not a small correction term. A factor 4.41 or 5.29 moves the
transition time across the entire published grid, silently: every run still completes.

---

## 3. The switch and the blend

**What the code does (A, decisive).** Three branches partitioned by
`Tcut_nCIE = max{grid log T ≤ 5.5}` and `Tcut_CIE = min{CIE log T > 5.5}`, with a bridge between
them. The bridge is `np.interp(log10 T, [Tcut_nCIE, Tcut_CIE], [D_lo, D_hi])` where
`D_lo = NET(n, Tcut_nCIE, Φ)` and `D_hi = chi_e n² Λ_CIE(10^Tcut_CIE)` — **both endpoints frozen at
the cutoff temperatures**. Inside the band, `du/dt` therefore has *no* temperature dependence from
either physical model: it is a straight chord between two fixed numbers.

**Is it defensible?** In form, yes — two independently produced tables need some handover, and
linear-in-log₁₀T is the cheapest one. In substance, three problems, all corroborated:

1. **Nothing checks that the two models agree** (A §5, explicit; B: continuity "never claimed in
   words, only implied"). C requires the ratio at the seam to land in `[0.7, 1.4]` and notes that a
   step of exactly **1.2 / 4.41 / 5.29** is a direct fingerprint of §2. Whatever the mismatch is, the
   ramp absorbs it silently and converts it into a fictitious steep gradient in `du/dt` over a band
   whose width is set by table sampling.
2. **The band width is uncontrolled.** If the CIE grid's first node above 5.5 is at 5.55 the ramp is
   0.05 dex; a coarser CIE library makes it arbitrarily wide. B independently flags that the non-CIE
   cutoff is *table-derived* while the CIE cutoff is anchored to a **hard-coded 5.5**, and that the
   prose claims the two coincide ("exactly what our threshold is") while `Tcut_CIE` is by
   construction strictly greater than 5.5. A commented-out guard `if nonCIE_Tcutoff != CIE_Tcutoff:`
   survives in the prose — the author knew.
3. **The 5.5 split puts the dominant feature on the wrong branch.** C: the main metal-line peak of
   `Λ(T)` sits at `log T = 5.0–5.3`, **below** the split, so the single most important feature for
   bubble energetics is evaluated from a photoionisation-equilibrium CLOUDY cube whose radiation
   field does not apply to shielded collisional gas. The cube must reduce to CIE as `U = Φ/(n c) → 0`
   and **nothing verifies that it does**.

**What the ramp does to photoheating and Φ-dependence** (A, corroborated by B's F3/F8 and C §4.2):

- Branch A returns `cool − heat` (carries the Φ axis, can be **positive**). Branches B and C's CIE
  end return cooling only. **Photoheating is linearly faded to zero across the band and is
  identically absent above `Tcut_CIE`.** Inside the band, `∂(du/dt)/∂Φ` is the cube's Φ-derivative
  scaled by `(1−w)`.
- The **n-dependence changes character**, not just magnitude: whatever CLOUDY tabulated below,
  exactly `chi_e n²` above. The derivative `d(du/dt)/d(log T)` is discontinuous at **both** cutoffs
  (two kinks) even though the value is continuous.
- **Is dropping photoheating above 10^5.5 K defensible?** Partly. C argues that at `T ≳ 10⁶` the
  relevant ions are collisionally stripped, so the omission is physically reasonable *there* — but
  also that the bubble interior is optically thin to LyC with `U ≈ 3×10²`, so it is not automatic at
  `10^5.5–10⁶`. Nothing in the code or the prose states the justification. Net: **defensible in the
  limit, undocumented and unbounded in the band.**

> **Correction to Lens C's cheapest test.** C proposes evaluating `get_dudt` at `T = 10^5.5(1∓ε)`
> and taking the ratio (C §8.2, S9-C-05). Given A's transcription this test **cannot fail**: the
> assembled function is continuous by construction, so the ratio is ≈1 and the test returns a false
> negative. The correct detector compares the **two models at the same temperature**:
> `NET(n, Tcut_nCIE, Φ)` vs `chi_e · n² · Λ_CIE(10^Tcut_nCIE)` at matched `(n, Φ)`, low Φ — exactly
> A's repro for S9-A-10. This is the single cheapest experiment in the whole slice and it tests §2
> and §3 simultaneously.

---

## 4. The sign convention

**Consistent. C's T12 is refuted by A's transcription.**

- A: `-1 *` appears in **all three** return statements (`:156`, `:165`, `:196`). Branch A returns
  `-1 × (cool − heat)`, so heating-dominated cells yield **positive** `du/dt` — exactly C's §4.2
  requirement that the net cube be allowed to be positive and preserve the `T_eq ≈ 10⁴ K` zero
  crossing. Branches B and C's CIE end are `≤ 0` by construction, as they must be for a pure cooling
  coefficient.
- C's feared failure — "CIE branch returns `+n²Λ` alongside a signed net cube, inverting `du/dt`
  above the seam so the transition never fires" — **does not occur**. Drop as a live finding.
- What survives is a **documentation** defect only (B, S9-B-04): the convention is stated in exactly
  one of three branches, in the non-CIE branch, with the typo "convension". Demoted to S4.
- One genuine sign issue remains, in the loader rather than the consumer: the `.dat` column sign
  normalisation inspects **only element `[0]`** and then negates the entire column (A, `:193-198`),
  while the prose claims signs are "forced positive" (B). A first row that is zero or positive while
  the rest are negative passes through unflipped, and `heat` then *adds* to `cool` across the whole
  grid. Retained as R-15.

---

## 5. Off-grid policy table

| axis / edge | **A — code policy** | **B — documented policy** | **C — required policy** | physical cost when exceeded |
|---|---|---|---|---|
| `T` below non-CIE floor | **silent clamp** to `10**Tmin`; no warning, flag or record | clamp documented; justified as "inert on every profiled regime — the bubble ODE never sends T below 3e4 K"; floor implied 10³ K by the word "decade", never written | never silently clamp low-T: hand off to a molecular coolant, return ~0, or raise | **10⁴–10⁶ over-cooling** — clamping returns the near-Lyα-peak value for 100 K gas |
| `T` between the two cutoffs | ramp between frozen endpoints; **no table lookup in T** | "interpolate" (between "two Λ values" — wrong quantity) | cutoffs must partition with no gap | see §3 |
| `T` above CIE table top | **undetermined from the slice** — the CIE interpolant is built upstream; raises if `interp1d` default, silently extrapolates log Λ if a spline / `fill_value='extrapolate'` | explicit admission: "does not support extrapolation. If this happens, implement a function that does that". No CIE table range stated for any of the 4 libraries | extrapolate `Λ(T_max)·(T/T_max)^{1/2}` (bremsstrahlung) | **at most ×3.2 low at 10⁹ K** if clamped — the only T edge where clamping is tolerable. But `T = 1.2×10⁸ K` post-shock at `v_w = 3000 km/s`, and `T_b ∝ t^{−6/35}` diverges as `t → 0`, so **the first timesteps of every run** are here; if it raises instead, the run dies mid-integration |
| `n` outside cube | **no clamp, no guard** → bare `ValueError: One of the requested xi is out of bounds in dimension 0` from inside the ODE RHS (scipy default `bounds_error=True`, A-verified). Also kills the bridge branch, which re-queries the cube | **nothing stated at all** | below min: clamping is *exact* (low-density limit) and benign. Above max: clamping the rate coefficient over-cools | run aborts with a message naming neither cooling nor the variable. `nCore = 10⁵ cm⁻³` is the **default** and the swept shell reaches 10⁶–10⁸ |
| `Φ` outside cube | same — no guard, raises. `Φ = 0` ⇒ `log10(0) = −inf` ⇒ same raise | **nothing stated at all** | `Φ → 0` **must** clamp to the lowest-Φ slice (that *is* the CIE limit, and it avoids `log 0`). Above max: heating grows ~linearly; clamping under-heats ⇒ the H II layer cools below 10⁴ K ⇒ `P_HII` collapses | run aborts. `Φ = Q_i/(4πR₂²) = 8×10¹⁴` at `R₂ = 0.1 pc`: **every run starts above the top**; `Q_i → 0` at late age puts it below the bottom |
| `age` outside file set | **silent clamp** to newest/oldest snapshot; between, linear-in-yr blend of raw cool/heat | documented ("use the max/min instead"); available ages 1e6…1e7 yr with a **5→10 Myr gap**; two contradictory rules ("nearest available age" vs interpolate) | clamp is defensible (spectrum shape varies slowly) **but must be recorded** | heating frozen at the last snapshot forever ⇒ late-time over-heating and mis-timed transition, with nothing in the output saying so |
| `Z` off the {1.0, 0.15} set | **explicit, well-worded `ValueError`** — the one loud case — on exact float equality | prose says "only solar … is considered" (stale); separately an auto-pin at `ZCloud == 0.15` | discrete-Z-by-file-selection is fine; exact float equality is a sweep hazard | loud abort; a sweep-generated `0.15 ± 1 ulp` kills the run rather than mis-selecting |
| NaN cells inside the cube | **silent NaN** — RGI returns NaN for any query in a cell touching a NaN vertex (A-verified) and it propagates into `du/dt` | documented present ("perhaps non-physical"); handling is a **"Future TODO"** | never return NaN into the ODE RHS | NaN in the state vector; solver stalls or truncates far from the cause |

**Three different policies on three axes** (silent clamp on `T`-low and `age`; hard raise on `n` and
`Φ`; explicit domain error on `Z`) with **no stated rationale for the asymmetry**, and the two axes
that raise are the two the documentation never mentions.

---

## 6. Divergence table

| id | finding (short) | class | sev | lenses | divergence | status |
|---|---|---|---|---|---|---|
| R-01 | CIE density pairing `chi_e·ndens²` unverifiable; 1.0 / 1.2 / 4.41 / 5.29 | coefficient | S1 | A,B,C | ABC | contested |
| R-02 | non-CIE branch applies **no** density factor; `.dat` column unit contradicted by its own docstring | units | S1 | A,B,C | AB | corroborated |
| R-03 | `T` below the table floor silently clamped | silent-failure | S2 | A,B,C | AC (reachability) | corroborated |
| R-04 | `n` and `Φ` never bounds-checked ⇒ bare scipy `ValueError` from the ODE RHS; `Φ=0` ⇒ `log 0` | regime | S2 | A,B,C | AC | corroborated |
| R-05 | photoheating + Φ-dependence faded to zero across the seam; models never compared | regime | S2 | A,B,C | ABC | corroborated |
| R-06 | the 10^5.5 split puts the 10^5–10^5.3 cooling peak on the non-CIE cube; low-U limit unvalidated | regime | S2 | A,C | AC | corroborated |
| R-07 | NaN cube cells propagate silently into `du/dt` | silent-failure | S2 | A,B,C | none (agreed defect) | corroborated |
| R-08 | `np.save` of the ragged cube list raises on numpy ≥ 1.24 ⇒ no table can be built without a cache | numerical | S2 | A | none | single-lens |
| R-09 | CIE upper edge unguarded; out-of-range policy undetermined and self-admittedly absent | silent-failure | S2 | A,B,C | BC | corroborated |
| R-10 | `get_Lambda`'s `metallicity` is dead; docstring says it "selects/validates" | deadcode | S3 | A,B | AB | corroborated |
| R-11 | CIE and non-CIE tables at different Z for the same `ZCloud=0.15` (0.1 vs 0.143 Z⊙) | citation | S3 | B | BC | single-lens |
| R-12 | age: silent clamp + linear-in-yr blend of raw rates across a 5→10 Myr gap | numerical | S3 | A,B,C | none | corroborated |
| R-13 | `_CIE_TCUTOFF_CACHE` keyed on `id()`, never evicted, no reference held | state | S3 | A,B,C | ABC | corroborated |
| R-14 | age list harvested from every `.dat` regardless of Z / rotation | other | S3 | A | none | single-lens |
| R-15 | `.dat` sign normalisation decided by element `[0]`, then negates the whole column | sign | S3 | A,B | AB | corroborated |
| R-16 | axes deduped **before** rounding; cube filled by exact float equality | numerical | S3 | A | none | single-lens |
| R-17 | `_cube.npy` cache keyed on filename only, written into `lib/` | silent-failure | S3 | A | none | single-lens |
| R-18 | the bridge ramp is an undeclared third model: linear in `du/dt`, width set by table sampling | numerical | S3 | A,B,C | scope-creep | corroborated |
| R-19 | the density factor is documented **zero times** anywhere in the slice | units | S3 | B | none (absence) | single-lens |
| R-20 | commented-out file-not-found handler at the `.param` trust boundary | silent-failure | S3 | B | none | single-lens |
| R-21 | seam prose bundle: "10e5.5", two-branch description, "exactly what our threshold is", sign convention in 1 of 3 branches | other | S4 | A,B | AB | corroborated |

Class key: **AB** = doc-drift, **AC** = physics, **BC** = mis-cited literature / spec, **ABC** = all
three differ, **scope-creep** = present in code and prose, sanctioned by no spec.

---

## 7. Demoted, refuted, or dropped — with reasons

| raw id | why it does not survive |
|---|---|
| **S9-C-03 / T12** (CIE returns `+n²Λ`, sign inversion above the seam) | **Refuted by A**: `-1 *` is present in all three returns. The structurally-likely bug C predicted did not happen. |
| **S9-C-04 / S9-C-17 / T17** (gap between the tables returning 0.0) | **Refuted by A**: the three branch conditions tile `[Tmin, +∞)` and the bridge covers the seam exactly. No uncovered band exists. |
| **S9-C-16 / T18** (`create_limits` uses `(a[0], a[-1])`; descending axis inverts bounds) | **Refuted by A**: `create_limits` does `round(log10(sort(unique(...))), 3)` — ascending by construction. A's own dedupe-order issue (R-16) is a different, real defect. |
| **S9-C-14 / T13** (log-interpolating the signed cube ⇒ NaN / destroyed `T_eq`) | **Requirement met**: the hot-path interpolator uses the **linear signed** `cool − heat`, exactly as C requires. The two log-valued interpolators that would have caused this are built and never used for the rate. |
| **S9-C-19 / T15** (Φ not converted to cgs ⇒ silent max-photoheating, sign inversion) | **Refuted by A**: `:83` divides Φ by `phi_cgs2au`. A verified the conversion factors independently. The residual `Φ = 0` case is folded into R-04. |
| **S9-C-06/07/08 / T6,T7,T8** (Z double-count, Z applied to `log Λ`, dex-vs-linear Z) | **Refuted by A**: `get_Lambda` never reads `metallicity`. No multiplier exists to double-count, log-scale, or mis-unit. The residual is R-10 (dead argument) and R-11 (cross-branch Z mismatch). |
| **S9-C-09 second half / T9** (linear `T` passed into a `log T` axis ⇒ flattened curve) | **Refuted by A**: `get_Lambda` takes linear `T` and logs it internally (`:60`); the CIE path is log–log, exactly C's requirement. |
| **S9-C-15 / T19** (age bracket and weights formed in different variables) | **Refuted by A**: `cube_lo + (x−a_lo)(cube_hi−cube_lo)/(a_hi−a_lo)`, one variable, weights sum to 1. The residual — linear in **yr** across log-spaced snapshots — is R-12. |
| **S9-C-18 / T20** (`get_fileage` age-unit mismatch) | **Refuted by A**: `:48` converts `t_now` (Myr) × 1e6 → yr and the filenames are in yr. Consistent. (A's 8-character-slice parsing wart is real but S4; not promoted.) |
| **S9-C-20 / T22** (`cooling_boost` applied to the signed net rate) | **Out of scope, and A's full transcription of all three branches contains no boost factor** — the knob is applied downstream, which C itself says makes the expectation vacuous. |
| **S9-C-21 / T23** (CIE `Λ` applied at shell densities above `n_crit`) | **Refuted for the stated scenario**: the CIE branch is only entered above 10^5.5 K, and the dense shell is ~10⁴ K, so it never reaches that branch. The residual (non-equilibrium suppression at the conduction front) is a modelling limitation of the chosen tables, not a code defect. |
| **S9-B-19** (missing `10**` on the CIE interpolant output ⇒ ×10²² error) | **Refuted by A**: `Lambda = 10**(interp(log10 T))` is present and matches the docstring exactly. |
| **S9-B-25** (axis ticks rounded but row keys not ⇒ rows land in the wrong cell) | **Refuted by A**: both the tick array and the lookup key use `np.round(..., 3)`. The real defect is the *order* of dedupe vs rounding (R-16). |
| **S9-B-17** (ulp-off `ZCloud` silently gets the solar CIE curve ⇒ mixed-metallicity run) | **Refuted by A**: the non-CIE path raises an explicit `ValueError` for any `Z ∉ {1.0, 0.15}`, so the run dies loudly *before* any silent mismatch. Residual: legitimate sweep-generated values abort the run (noted, S4, not promoted). |
| **S9-A-03** (linear-value interpolation of the net rate; "cells a decade wide") | **Severity demoted S2 → S3.** C explicitly *requires* linear-in-signed-value for the cube (log of a sign-changing quantity is impossible), so the choice is correct, not incidental. And A's "decade wide" is unsupported: B's `(33, 21, 22)` shape over `[10³, 10^5.5]` implies ≈**0.125 dex** T spacing, where C's own error table gives ~1 %, not "up to ×10". Retained only as the coarser-Φ-axis remark inside R-18. |
| **S9-B-30 / S9-B-32 / S9-A-18 / S9-A-19 / S9-A-13** (unit annotations on log arrays; 3.8 shim; 8-char age slice; `t_now` missing `.value`; unused `age` arg) | Real but cosmetic; kept in the lens reports, not promoted past the S4 bundle (R-21 covers the seam-related prose; `age`-arg deadness is noted inside R-12's evidence). |

---

## 8. Merged ranked findings

```json
[
  {
    "id": "S9-R-01",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 164,
    "class": "coefficient",
    "severity": "S1",
    "claim": "The CIE branch multiplies the cooling coefficient by `chi_e * ndens**2`. Whether that product equals the physically required n_e*n_H cannot be decided from the three lens reports: it is correct (factor 1.00) if `ndens` is the hydrogen-nuclei density and chi_e = n_e/n_H = 1.2, and wrong by 0.833, 1.633, 4.41 or 5.29 for the other pairings. The blend branch inherits the same factor, weighted by w, at line 187.",
    "evidence": "A transcribed `dudt = params_dict['chi_e'].value * ndens**2 * Lambda_CIE` at :164 and the identical form at :187, with ndens converted pc^-3 -> cm^-3 at :82; A explicitly recorded that nothing in the slice pins down whether chi_e is n_e/n_H or n_e/n_tot. C derived n_tot/n_H = 2.3, n_e/n_H = 1.2 for fully ionised cosmic gas, hence n_tot^2/(n_e n_H) = 5.29/1.20 = 4.41, and reads SPEC-003 as declaring nCore/nISM to be hydrogen-nuclei densities. B reports the multiplier is documented nowhere, so the prose provides no cross-check. A's assertion that `ndens` is 'the total number density' is an inference - A saw no call site.",
    "expected": "du/dt = -(n_e n_H) * Lambda_table, with n_e = chi_e * n_H and chi_e matched to the ionisation state, and with the table's published normalisation (per n_e n_H vs per n_H^2) matching the code-side product exactly. The intended definition of both `chi_e` and `ndens` must be stated at the function contract.",
    "failure_scenario": "A clean multiplicative error of up to 5.29x on all cooling above 10^5.5 K - the hot bubble interior whose radiative losses set the energy->momentum transition trigger (L_gain-L_loss)/L_gain <= 0.05. Because C shows the bubble bulk is effectively non-radiative (t_cool ~ 1.3 Gyr at n_H=1e-2, T=1e7) and essentially all L_cool comes from the thin conduction front, this rescales exactly the quantity that fires the transition. Never appears as an instability - only as a shifted transition time across the whole published grid.",
    "repro": "Three lookups, in order: (1) the default value and description of `chi_e` in trinity/_input/ (default.param + schema) - 1.2 or 1.1 means n_e/n_H; (2) what every call site of get_dudt passes as `ndens` (n_H or n_tot); (3) the normalisation stated in the header of the bundled CIE file under lib/default/CIE/ named by path_cooling_CIE. Then evaluate chi_e*(ndens/n_H)^2/1.20; it must be 1.0.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "contested",
    "source_ids": ["S9-A-14", "S9-B-01", "S9-C-01", "S9-C-02"]
  },
  {
    "id": "S9-R-02",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 154,
    "class": "units",
    "severity": "S1",
    "claim": "The non-CIE branch applies NO density factor at all - it returns the interpolated `cool - heat` verbatim - which is correct only if the opiate/CLOUDY .dat columns are volumetric (erg cm^-3 s^-1). The loader's own docstring says the opposite: 'Cooling rate is in units of [erg cm3 / s]', i.e. a coefficient.",
    "evidence": "A: line 154 returns `-1 * netcool_interp([...])[0] * dudt_cgs2au` with no n**2, no n_e n_H and no chi_e anywhere in the branch; the density enters only as the first interpolation axis. A notes the two branches balance dimensionally ONLY under the assumption that the .dat columns are volumetric and the CIE table is a coefficient, and that nothing in the code asserts, converts or checks either. B: read_cloudy.py:23 documents the cube as 'erg cm3 / s' while net_coolingcurve.py:154 and :179 annotate the value read out of it as 'u.erg / u.cm**3 / u.s' - the same quantity given two different units two files apart, with no multiplier documented in between. C: CLOUDY `save cooling` emits the volumetric rate at the model's own density [recalled, medium].",
    "expected": "One documented unit for the CLOUDY cube, asserted at load time. If the columns are volumetric, the branch is correct and the asymmetry against the CIE branch (density inside the table vs outside it) must be stated explicitly, because it is invisible dimensionally once both are cast to erg cm^-3 s^-1.",
    "failure_scenario": "If the columns are in fact a coefficient, every non-CIE rate is short by n^2 - up to 1e8 across a 1e-2..1e4 cm^-3 density axis - and the bridge band at 10^5.5 K linearly blends two incommensurable quantities. This would be the largest defect in the slice, and it is completely silent.",
    "repro": "In a bundled opiate_cooling_*.dat (directory named by path_cooling_nonCIE, lib/default/...), hold (temp, phi) fixed and scan the ndens column: if `cool` scales as n^2 the column is volumetric and the code is right; if `cool` is ~flat in n it is a coefficient and the branch is wrong by n^2.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S9-B-02", "S9-A-14", "S9-C-01"]
  },
  {
    "id": "S9-R-03",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 130,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Temperatures below the non-CIE table floor are silently clamped to the floor (`if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`) with no warning, flag, counter or record; every such T returns the identical rate. The prose defends this as 'inert on every profiled regime', citing docs/dev/magic-numbers for the claim that the bubble ODE never sends T below 3e4 K.",
    "evidence": "A read lines 130-131 directly and notes nothing downstream records that the clamp fired; contrast the ndens and phi axes, which are not clamped at all. B quotes the gate comment verbatim, including its justification and the admission that it 'replaces a hard-coded 1e4 floor that over-floored the whole valid [10**nonCIE_Tmin, 1e4) decade' - the word 'decade' implies the floor is 1e3 K, but the value is never written down. C derives that below 1e4 K, Lambda collapses by 4-6 decades under exp(-1.18e5/T), so clamping returns a near-Lya-peak efficiency for cold gas, and lists T < 1e4 K regimes that occur normally: the neutral/molecular outer shell layer at 10-100 K, the undisturbed cloud, the shell after Q_i collapses, and the whole re-collapse fate.",
    "expected": "Either a recorded/warned clamp the caller can detect, or a proof of unreachability that is an invariant rather than 'the regimes someone happened to profile'. The floor value must be stated numerically in the docstring.",
    "failure_scenario": "Gas below the table floor receives the floor-temperature rate - up to 1e4-1e6 over-cooling per C - with no diagnostic; downstream phase-transition triggers that watch du/dt or the cooling time fire at the wrong epoch and the run completes looking healthy.",
    "repro": "Call get_dudt with T = 10**(nonCIE_Tmin - 2) and T = 10**nonCIE_Tmin and confirm the returns are bit-identical. Then instrument the gate with a counter and run param/simple_cluster.param plus docs/dev/performance/f1edge_{lowdens,hidens}*.param; a non-zero count refutes the 'inert' claim. Separately, enumerate the call sites of get_dudt to settle whether shell/cloud gas reaches it at all.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S9-A-01", "S9-B-06", "S9-C-11", "S9-C-12"]
  },
  {
    "id": "S9-R-04",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 154,
    "class": "regime",
    "severity": "S2",
    "claim": "Density and photon flux are never bounds-checked or clamped: an out-of-grid n or phi raises a bare scipy ValueError from inside the ODE right-hand side, asymmetrically with temperature which is silently clamped. phi = 0 gives log10(0) = -inf and takes the same path. Both axes are legitimately exceeded in normal operation.",
    "evidence": "A verified on scipy 1.17.1 that RegularGridInterpolator defaults to bounds_error=True (read_cloudy.py:136 passes no bounds argument), producing 'ValueError: One of the requested xi is out of bounds in dimension <k>' with no context; net_coolingcurve.py guards only T (:130-131), and line 179 re-queries the cube inside the bridge branch so an out-of-range n or phi kills that branch too. B reports the documentation says nothing whatsoever about out-of-bounds density or phi - the gate comment covers temperature only. C derives that nCore = 1e5 cm^-3 is the DEFAULT and the swept shell reaches 1e6-1e8, that the bubble interior sits at 1e-3..1e-2, that phi = Q_i/(4 pi R2^2) = 8e14 cm^-2 s^-1 at R2 = 0.1 pc so every run starts above the phi axis top, and that Q_i -> 0 at late age produces phi = 0 exactly.",
    "expected": "Consistent, per-axis policy: clamp where the physics genuinely asymptotes (n below min is the exact low-density limit; phi -> 0 is the CIE limit and clamping there is mandatory to avoid log(0)), extrapolate where an asymptote exists, otherwise raise a domain-specific error naming the quantity, its value and the table range - and record every clamp.",
    "failure_scenario": "A dense shell zone, a rarefied bubble interior, a bright young cluster or a late-age Q_i = 0 row aborts the whole run with a scipy message that identifies only an axis index, giving no clue that the cooling table was the cause. Per C, this is not a corner case: several of these are the normal operating envelope, including the first timesteps of every run.",
    "repro": "Call get_dudt with ndens an order of magnitude above the top of log_ndens_arr, and again with phi = 0.0; both must fail loudly today. Then instrument per-axis, per-direction counters and run param/simple_cluster.param.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S9-A-11", "S9-B-29", "S9-C-12", "S9-C-13"]
  },
  {
    "id": "S9-R-05",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 194,
    "class": "regime",
    "severity": "S2",
    "claim": "Photoheating is present below the non-CIE cutoff, linearly faded to zero across the bridge band, and identically absent above the CIE cutoff; the phi-dependence of du/dt therefore vanishes across the switch, and nothing anywhere compares the two physical models at the seam.",
    "evidence": "A: branch A returns the interpolated `cool - heat` (carries the phi axis, may be positive); branch B returns chi_e*n^2*Lambda_CIE with no heat term and no phi dependence; branch C ramps linearly between one value that includes heating and phi and one that includes neither. The assembled function is continuous in VALUE by construction, but the derivative is discontinuous at both cutoffs and nothing compares NET(n, Tcut_nCIE, phi) against chi_e n^2 Lambda_CIE(10**Tcut_nCIE). B independently confirms from the prose that the non-CIE side is cooling-minus-heating while the CIE side is cooling only, that no comment justifies dropping the heating term, and that continuity is never claimed in words. C requires the ratio at the seam to lie in [0.7, 1.4] and notes a step of exactly 1.2, 4.41 or 5.29 is a direct fingerprint of the normalisation question.",
    "expected": "Either a documented and tested statement that photoionisation heating is negligible above 10^5.5 K, or a heating term on the CIE side; plus a startup check that reports the model mismatch at the join instead of absorbing it into the ramp.",
    "failure_scenario": "If the two models disagree at 10^5.5 K (a factor of a few is common between a photoionised CLOUDY net rate and a pure-CIE Lambda), the ramp manufactures a fictitious steep gradient in du/dt over a narrow log-T interval whose width is set by table sampling. Zones in that interval get a rate that depends on grid layout rather than physics; a stiff adaptive solver can chatter on the kinks and a bisection on du/dt = 0 can converge to the kink instead of T_eq.",
    "repro": "Do NOT test by sweeping T across the seam - the assembled function is continuous by construction and the test returns a false negative. Instead evaluate the two models at the SAME temperature: NET(log n, Tcut_nCIE, log phi) versus chi_e*n**2*get_Lambda(10**Tcut_nCIE, ...) at matched (n, phi) with low phi, and print the ratio. It must land in [0.7, 1.4].",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S9-A-10", "S9-B-05", "S9-C-05", "S9-C-23"]
  },
  {
    "id": "S9-R-06",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 43,
    "class": "regime",
    "severity": "S2",
    "claim": "The hardcoded 10^5.5 K split places the dominant metal-line cooling peak (log10 T = 5.0-5.3) on the NON-CIE branch, so the single most important feature of Lambda(T) for bubble energetics is evaluated from a photoionisation-equilibrium CLOUDY cube; nothing verifies that the cube reduces to the CIE curve in its low-ionisation-parameter limit.",
    "evidence": "A transcribed the split as the bare literal 5.5 appearing twice (:43 for max{grid log T <= 5.5}, :53 for min{logT_CIE > 5.5}) with the non-CIE branch covering everything below. C locates the main peak at log10 T = 5.0-5.3 (location high confidence, amplitude (2-4)e-22 recalled medium), notes the cube must reduce to CIE as U = phi/(n c) -> 0, and gives the counterpoint that with the Weaver profile and bubble_xi_Tb = 0.98 the bubble interior is almost entirely above 10^5.5 K, so the cube is exercised mainly by shell/HII gas while the CIE branch carries the bubble.",
    "expected": "A validation that the cube's lowest-phi, lowest-n slice at T = 1e5 K agrees with -(n_e n_H) Lambda_CIE(1e5) from the same normalisation to within ~30%, plus a record of the minimum ionisation parameter the cube's axes actually span.",
    "failure_scenario": "The peak of the cooling curve - which dominates L_cool because T sweeps through it at the conduction front - is systematically wrong under a radiation field that does not apply to shielded collisional gas, shifting the energy->momentum transition time with no diagnostic.",
    "repro": "Compare the cube's lowest-phi slice at T = 1e5 K against an independently normalised CIE value; record min(U) = min(phi)/(max(n) c) over the cube axes; and histogram which branch is taken vs (T, n) over a full run of param/simple_cluster.param.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S9-C-23", "S9-A-20", "S9-C-21"]
  },
  {
    "id": "S9-R-07",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 228,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Cube cells for (n, T, phi) combinations absent from the .dat remain NaN, and RegularGridInterpolator returns NaN silently for any query in a cell touching a NaN vertex; the NaN propagates through the age blend and into du/dt and the integrator with no diagnostic.",
    "evidence": "A: lines 226-228 allocate cool_cube and set it entirely to NaN, then 231-237 scatter-fill only the rows present in the file (same for heat at 244-254), with no post-fill completeness check; A verified that an RGI over a values array with one NaN vertex returns NaN for a query in the containing cell, and that :91's age blend and :98/:100's log10 both propagate it. B confirms from the prose that NaNs are known to be present ('Some are NaN, because they are not available in the cooling table (perhaps non-physical)') and that handling them is an unimplemented 'Future TODO'. C requires that NaN never reach the ODE right-hand side.",
    "expected": "A completeness assertion after the fill loops, or masked/filled cells, or a finiteness check on get_dudt's return. Note the NaN-affected region is larger than the NaN cells: every cell touching a NaN vertex is poisoned.",
    "failure_scenario": "A non-rectangular opiate table - CLOUDY runs that failed to converge for some parameter corners are commonly dropped from the output - yields NaN du/dt in exactly the regime where the physics was hardest. The ODE solver stalls, rejects every step, or carries NaN into the state vector; per CLAUDE.md the bubble-structure monotonic guard may swallow it into a silently truncated run.",
    "repro": "Count non-finite entries in the constructed cool_cube/heat_cube for a bundled table; then delete one row from a copy of an opiate .dat (and its _cube.npy), rebuild, and query netcool_interp at the midpoint of the affected cell.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S9-A-05", "S9-B-08", "S9-C-14"]
  },
  {
    "id": "S9-R-08",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 265,
    "class": "numerical",
    "severity": "S2",
    "claim": "`np.save(cube_filename, [log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube])` raises ValueError on numpy >= 1.24 because the list is inhomogeneous, so create_cubes cannot complete for any table lacking a pre-built `_cube.npy` cache.",
    "evidence": "A read line 265 and verified on numpy 1.26.4 (the installed version; CLAUDE.md pins numpy<2) that np.save on such a list raises 'ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.' The load side at :175 already passes allow_pickle=True, so an object array was intended. The whole parse (lines 183-254) completes and is then discarded by the exception on the last line before the return.",
    "expected": "`np.save(cube_filename, np.array([...], dtype=object))`, or np.savez with named arrays.",
    "failure_scenario": "Any user pointing path_cooling_nonCIE at their own opiate tables, or any run after the bundled *_cube.npy files are deleted or regenerated, crashes in get_coolingStructure with an opaque numpy shape error that names neither cooling nor the table. It also implies the shipped caches were produced under an older numpy and cannot be regenerated on the pinned stack.",
    "repro": "python -c \"import numpy as np; np.save('/tmp/x.npy',[np.arange(3.),np.arange(4.),np.arange(2.),np.zeros((3,4,2)),np.zeros((3,4,2))])\"",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-A-04"]
  },
  {
    "id": "S9-R-09",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 60,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The CIE branch is entered for every T >= CIE_Tcutoff with no bounds check of any kind, and the behaviour above the CIE table's top is undetermined: get_Lambda hands log10(T) straight to an interpolant constructed outside the slice. The prose admits outright that the CIE path 'does not support extrapolation', and no CIE table temperature range is documented for any of the four bundled libraries.",
    "evidence": "A: the entire body is `T = np.log10(T); Lambda = 10**(cooling_CIE_interpolation(T)); return Lambda`, with no guard; the interpolant arrives as params_dict['cStruc_cooling_CIE_interpolation'] so its type, extent and out-of-range policy are not determinable from the slice - it raises if it is interp1d with defaults, and silently extrapolates log Lambda if it is a spline or fill_value='extrapolate'. B quotes the admission 'Might be a problem here because this does not support extrapolation. If this happens, implement a function that does that', and notes the asymmetry: a gate exists for the non-CIE lower edge but none for the CIE upper edge. C derives that T > 1e8 K occurs normally - post-wind-shock T = 3 mu m_H v_w^2/(16 k_B) = 1.2e8 K at v_w = 3000 km/s, and T_b ~ t^{-6/35} diverges as t -> 0 so the first timesteps of every run are the hottest - and that the correct extrapolation is Lambda(T_max)*(T/T_max)^{1/2}.",
    "expected": "A documented CIE table temperature range per library, plus a stated behaviour above the maximum: extrapolate as T^{1/2} (bremsstrahlung, exactly linear in log-log so linear extrapolation is correct there), or clamp and record.",
    "failure_scenario": "A hot early bubble interior above the CIE table maximum either raises out of the ODE right-hand side (run dies mid-integration) or returns an unbounded log-log extrapolation, depending on an interpolant this slice cannot see. Silent clamping would be the mildest outcome at only ~3.2x low at 1e9 K - the one T edge where clamping is tolerable.",
    "repro": "Determine max(logT_CIE) for each of the four bundled CIE libraries and call get_Lambda just above it; record whether it raises, returns NaN, clamps, or extrapolates. Then check get_Lambda(1e8) ~ 2.4e-23 erg cm^3 s^-1 and a log-log slope of +0.5 above 3e7 K.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S9-B-18", "S9-C-11", "S9-C-22", "S9-A-01"]
  },
  {
    "id": "S9-R-10",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "deadcode",
    "severity": "S3",
    "claim": "get_Lambda accepts a `metallicity` argument and never reads it, while its docstring documents that parameter as one that 'selects/validates against the CIE library'. Both call sites pass params_dict['ZCloud'].value. The non-CIE branch does honour Z, via file selection - so the two branches are asymmetric in Z handling.",
    "evidence": "A: the whole body is lines 60-64 and `metallicity` appears only in the signature; net_coolingcurve.py:163 and :186 both pass ZCloud. B quotes the docstring's description of the parameter, a shipped non-solar library, and an auto-pin at ZCloud == 0.15 - i.e. Z selection happens upstream, outside this function. C requires exactly one Z mechanism (per-Z table selection OR a metals-only multiplier OR validation), never two.",
    "expected": "Drop the parameter and correct the docstring, or use it to validate that the loaded CIE table's metallicity matches ZCloud. Note this refutes C's double-count / log-scaling / dex-vs-linear traps: there is no multiplier to get wrong.",
    "failure_scenario": "A reader trusts the docstring and assumes the CIE curve tracks ZCloud through this argument. If the upstream auto-pin were ever removed or bypassed, a Z=0.15 run would silently use the solar CIE curve above 10^5.5 K while using the Z=0.15 CLOUDY cube below it.",
    "repro": "grep for every construction of cStruc_cooling_CIE_interpolation and confirm the CIE library is chosen from ZCloud upstream; then delete the unused parameter and confirm nothing breaks.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S9-A-02", "S9-B-15", "S9-C-07"]
  },
  {
    "id": "S9-R-11",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "citation",
    "severity": "S3",
    "claim": "At ZCloud = 0.15 the two branches are documented to use tables of different metallicity: the CIE side auto-pins to Sutherland & Dopita 1993 at [Fe/H] = -1 (0.1 Zsun) while the non-CIE side uses the '0.15 solar' table at Z = 0.002 against a stated solar Z = 0.014 (0.143 Zsun) - a ~43% difference in metals, applied on opposite sides of the 10^5.5 K seam.",
    "evidence": "B only, from the prose: CIE/read_coolingcurve.py:26-54 (library 4, '[Fe/H] = -1', 'Auto-pinned when ZCloud == 0.15 regardless of path_cooling_CIE') versus read_cloudy.py:295 ('solar, Z = 0.014') and :298 ('0.15 solar, Z = 0.002'). C supplies why it matters: the metal-line component of Lambda is linear in Z, and metals dominate from ~1e5 to ~1e7 K, so the step is real in exactly the band the seam sits in. A saw no CIE library selection at all (it happens outside the slice), so this is unverified.",
    "expected": "Either a CIE table at 0.15 Zsun, or a documented statement that the nearest available CIE metallicity is 0.1 Zsun and that the two branches therefore differ by ~50% in metals across the seam.",
    "failure_scenario": "A low-metallicity run cools with 0.143 Zsun below 10^5.5 K and 0.1 Zsun above it, with a metallicity step exactly at the switch temperature that the bridge band then blends - adding a spurious contribution to the seam mismatch of S9-R-05.",
    "repro": "Run the same config at ZCloud = 0.15 and log which CIE file and which opiate file are actually loaded; compare their nominal metallicities.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "BC",
    "status": "single-lens",
    "source_ids": ["S9-B-16"]
  },
  {
    "id": "S9-R-12",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 91,
    "class": "numerical",
    "severity": "S3",
    "claim": "Cluster ages outside the tabulated set are silently clamped to the first or last snapshot with no warning, and ages between snapshots are blended linearly in years on the raw cool/heat values - across a documented 5e6 -> 1e7 yr gap in the available age set.",
    "evidence": "A: lines 325-329 and 330-334 clamp to max/min with no print and no returned flag (contrast the metallicity path at :300-305, which raises with a clear message); line 91 blends as cubes_low + (x-ages_low)*(cubes_high-cubes_low)/(ages_high-ages_low), applied before the log10 at :98/:100 and before the net at :134. B lists the available ages as 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr and quotes two contradictory documented rules ('find the nearest available age' vs interpolate between neighbours) plus the clamping statement. C confirms the weights are correctly formed in a single variable and sum to 1 - refuting the bracket/weight-variable trap - and accepts clamping as defensible provided it is recorded. A also notes the `age` argument of get_dudt is never referenced, so per-call age re-selection does not happen.",
    "expected": "One documented rule; a warning (the module already imports cprint) when the clamp fires; and a stated maximum age error given the snapshot spacing. Interpolating log-spaced snapshots linearly in years heavily weights the newer snapshot across the 5-10 Myr gap.",
    "failure_scenario": "A simulation integrated past the last snapshot keeps using that snapshot's photoionisation-dependent cooling/heating forever - the heating term stops declining with the ageing cluster, so the bubble is over-heated at late times and the phase-transition timing is wrong, with nothing in the output indicating the table stopped evolving. Runs younger than 1 Myr - the entire early energy-driven phase - silently use the 1 Myr structure.",
    "repro": "Log which cooling file(s) and weights are chosen at t_now = 0.3, 2.3, 7.0 and 15.0 Myr; then blend two adjacent snapshots at the geometric mean age and compare against linear-in-log-age blending of log10(cool).",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S9-A-07", "S9-A-17", "S9-B-22", "S9-C-15", "S9-A-13"]
  },
  {
    "id": "S9-R-13",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 50,
    "class": "state",
    "severity": "S3",
    "claim": "_CIE_TCUTOFF_CACHE is a module-level dict keyed on `id(logT_CIE)` and never evicted, and nothing holds a reference to the keyed array; CPython reuses id values after garbage collection, so a differently-shaped CIE grid allocated at a freed address returns the previous grid's cutoff. The prose defends the id-keying with an assumption ('built once at startup and never replaced') that its sibling cache explicitly rejects.",
    "evidence": "A: line 27 `_CIE_TCUTOFF_CACHE: dict = {}`, lines 50-54 key = id(logT_CIE); contrast _noncie_cutoffs (:40-45), which memoises as an attribute on the cube object and is therefore lifetime-safe. B quotes both justifications side by side: '_cie_tcutoff cached by array id ... its id is stable for the whole run -> no id-reuse hazard' versus '_noncie_cutoffs cached on the cube object (not by id) so it refreshes automatically when the cube is rebuilt'. C predicts exactly this class of bug for sweeps and cites CLAUDE.md's own warning that trinity leaks module-level global state in-process.",
    "expected": "Attribute-based memoisation on the array's owning object, matching the sibling, or a key derived from the array contents.",
    "failure_scenario": "A sweep or any workflow that builds a second cooling structure in-process gets the first structure's CIE_Tcutoff. The switch temperature is then silently wrong, and if nonCIE_Tcutoff >= CIE_Tcutoff results, np.interp's xp is non-increasing and the bridge produces garbage without raising. Results would differ between --workers 1 and --workers N.",
    "repro": "Confirm first whether logT_CIE is ever rebuilt in-process (the ZCloud == 0.15 auto-pin and any per-run re-init are candidates). Then, in one process, build structure A, call get_dudt, drop all references, force gc, build structure B with a different CIE log-T grid, and compare _cie_tcutoff(B.logT) against min(B.logT[B.logT > 5.5]).",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S9-A-06", "S9-B-09", "S9-C-17"]
  },
  {
    "id": "S9-R-14",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 311,
    "class": "other",
    "severity": "S3",
    "claim": "The available-age list is harvested from every *.dat in the cooling directory regardless of that file's metallicity or rotation flag, then a filename is composed from the requested Z/rotation and the harvested age.",
    "evidence": "A: lines 310-317 iterate os.listdir(path2cooling) filtering only on the '.dat' suffix and append get_fileage(files); Z_str and rot_str (computed at 289-305) never enter the filter, while the composed name at :322/:328/:333/:343 uses the requested Z_str and rot_str with an age that may only exist for a different table set. B independently reports that the only documented error handling for a missing file is commented out, so the failure would surface raw.",
    "expected": "Filter the listing by the same `opiate_cooling_{rot_str}_Z{Z_str}_age` prefix the filename is built from.",
    "failure_scenario": "If the Z=1.00 and Z=0.15 (or rot and norot) sets have different age sampling, get_filename returns a name that does not exist and the run dies later in ascii.read with a FileNotFoundError naming a file the user never asked for; or the bracketing branch blends across an interval taken from the wrong table set.",
    "repro": "Place a single extra opiate_cooling_norot_Z0.15_age9.99e+06.dat in the directory and request Z=1.0 at t_now = 9.99 Myr.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-A-08"]
  },
  {
    "id": "S9-R-15",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 193,
    "class": "sign",
    "severity": "S3",
    "claim": "The sign normalisation of the cool and heat columns inspects only element [0] and then negates the entire column, while the documentation describes it as 'make sure signs in heating/cooling column are positive!' - i.e. the prose claims a guarantee the code samples for.",
    "evidence": "A: lines 193-198, `if np.sign(heating_data[0]) == -1: heating_data = -1 * heating_data` and the same for cooling_data - one sample decides for the whole column and the negation is unconditional across all entries. B quotes read_cloudy.py:192 and links it to net = cooling - heating at net_coolingcurve.py:144/:175.",
    "expected": "A whole-column test (e.g. `(heating_data < 0).all()`) or per-element abs(), with an error on genuinely mixed signs, and documentation of which CLOUDY column is emitted negative and why.",
    "failure_scenario": "A table whose first row is zero or positive while the rest are negative passes through unflipped, so heat enters `cool - heat` with the wrong sign and heating is ADDED to cooling over the whole grid - a sign error in the net rate with no warning. Conversely a genuinely mixed-sign column is corrupted by the blanket negation.",
    "repro": "Set the first heat entry of a copy of an opiate table to 0.0 while leaving the rest negative, delete the cache, rebuild and inspect the resulting heat_cube sign; also histogram the raw sign distribution of both columns in a bundled table.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S9-A-09", "S9-B-26"]
  },
  {
    "id": "S9-R-16",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 207,
    "class": "numerical",
    "severity": "S3",
    "claim": "Axis construction dedupes on the LINEAR values before taking log10 and rounding to 3 decimals, so two distinct linear values can collapse to the same axis entry; the cube is then scatter-filled by exact float equality against those rounded values, with `[0][0]` raising IndexError on any non-match.",
    "evidence": "A: create_limits (:204-214) does set(array) -> sort -> log10 -> np.round(..., 3) with no second dedupe after rounding, and the fill loops index by np.where(log_arr == np.round(np.log10(val), 3))[0][0] at :233-235 and :250-252. A verified that a duplicated axis entry makes RegularGridInterpolator raise 'The points in dimension 0 must be strictly ascending or descending'. This also refutes B's related worry: tick and key use the SAME rounding, so rows cannot land in the wrong cell for that reason.",
    "expected": "Round first, then dedupe (np.unique(np.round(np.log10(array), 3))), so the axis and the lookup key derive from the same rounded quantity. Note the interpolation abscissae are the rounded ticks, not the table's true coordinates - a <=0.001 dex displacement, negligible but undocumented.",
    "failure_scenario": "A finely sampled table (spacing below ~0.001 dex, or values differing in the fourth significant figure) fails to build with a scipy message naming no table and no column; a non-positive value in any of the three columns produces -inf/nan and an IndexError instead.",
    "repro": "Construct a two-row opiate-format table with ndens = 1.0 and 1.0005 and run create_cubes on it.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-A-15", "S9-B-25"]
  },
  {
    "id": "S9-R-17",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 174,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The `_cube.npy` cache is keyed only on the .dat filename - no mtime, size or format-version check - and is written into the table directory itself, so a stale cache silently shadows an edited or replaced table.",
    "evidence": "A: lines 172-176 build cube_filename from the .dat stem and return immediately if it exists, skipping the parse at 183+ entirely; line 265 writes the cache back into path2cooling. B corroborates only that persistence exists ('Does the cube already exist?', 'Final step: save into an array to save time in the future') and that a stale TODO claims the feature is still missing.",
    "expected": "Compare the cache mtime against the .dat mtime (or embed a content hash), and write the cache to a user-owned location rather than into what may be a read-only bundled lib/ directory.",
    "failure_scenario": "A user regenerates or edits an opiate table in place and every subsequent run keeps using the old cube with no indication, so the change appears to have no effect. Symmetrically, if path2cooling is read-only the write raises PermissionError only after the full parse has already run.",
    "repro": "Edit a cool value in a .dat whose _cube.npy exists and confirm the interpolated rate is unchanged.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-A-16", "S9-B-31"]
  },
  {
    "id": "S9-R-18",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 168,
    "class": "numerical",
    "severity": "S3",
    "claim": "The bridge band is an undeclared third model: both endpoints are frozen at the cutoff temperatures, so inside the band du/dt has no temperature dependence from either physical model - it is a straight chord in du/dt against log10 T between two fixed numbers - and the band's width is set entirely by where the CIE grid's first node above 5.5 happens to land. The prose describes it as interpolating 'between two Lambda values', which is a different quantity.",
    "evidence": "A: D_lo = NET(log n, Tcut_nCIE, log phi) at :179, D_hi = chi_e*n**2*Lambda_CIE(10**Tcut_CIE) at :186-187, combined by np.interp(log10 T, [Tcut_nCIE, Tcut_CIE], [D_lo, D_hi]) at :194 - linear in log10 T and linear in the VALUE of du/dt. B confirms the design note at :91 says 'interpolate between the two Lambda values' while the branch's own debug print interpolates dudt, and separately confirms the CIE cutoff is anchored to a hardcoded 5.5 while the non-CIE cutoff is table-derived. C requires the two cutoffs to coincide or overlap with a documented, deterministic tie-break; a linear chord in a third band is sanctioned by no spec.",
    "expected": "Either make the cutoffs coincide (no band at all), or declare the blend as a modelling choice with a stated, bounded width and a stated interpolation variable. Blending linearly in du/dt between two values that can differ by orders of magnitude gives a band whose shape is dominated by the larger endpoint.",
    "failure_scenario": "Zones whose temperature lands in the band get a rate produced by neither cooling model, with a width and shape determined by table sampling. Swapping to one of the other three bundled CIE libraries silently widens or narrows the band.",
    "repro": "Print nonCIE_Tcutoff and CIE_Tcutoff for each of the four bundled CIE libraries, record the band width in dex, and confirm the selection min(logT_CIE[logT_CIE > 5.5]) is non-empty in every case.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "scope-creep",
    "status": "corroborated",
    "source_ids": ["S9-A-10", "S9-B-13", "S9-B-10", "S9-C-04"]
  },
  {
    "id": "S9-R-19",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 59,
    "class": "units",
    "severity": "S3",
    "claim": "The density factor that bridges Lambda [erg cm^3 s^-1] and du/dt [erg cm^-3 s^-1] is documented ZERO times anywhere in the slice. This is an absence, not a contradiction - it is the reason no maintainer can check S9-R-01 or S9-R-02 by reading.",
    "evidence": "B: the docstring fixes both ends of the conversion (net_coolingcurve.py:59-76, :163, :154, :179, :187) but no line anywhere writes n^2, n_e*n_H, n_tot^2 or n. The only constraint the prose imposes is dimensional, which cannot distinguish n_tot^2 from n_e n_H - factors that differ by 4.41 in ionised gas per C. A confirms independently that the two branches in fact use DIFFERENT treatments (none vs chi_e*n**2), which is precisely the asymmetry the documentation would need to state.",
    "expected": "The contract docstring states the exact multiplier per branch, e.g. 'du/dt = -(chi_e n_H^2) Lambda with chi_e = n_e/n_H and n_H the hydrogen-nuclei density in cm^-3; the non-CIE cube is already volumetric and takes no factor'. Also state what chi_e is and where the table's normalisation is recorded.",
    "failure_scenario": "Not a runtime failure. It is the audit blocker: the documentation offers no cross-check on the single highest-severity quantity in the slice, so the pairing can only be settled by reading source plus tables.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-B-01", "S9-B-03"]
  },
  {
    "id": "S9-R-20",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 345,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The only error handling for a missing non-CIE cooling table is commented out: '# try:' and '# except: # raise Exception(\"Opiate/CLOUDY file (non-CIE) for cooling curve not found. Make sure to double check parameters ...\")'. This sits at a trust boundary - the path comes from the user's .param.",
    "evidence": "B: read_cloudy.py:287-288 and :345-346. A corroborates indirectly: its transcription of get_filename (:270-344) contains no active try/except, and A separately shows the failure modes that would surface raw - an empty directory raises ValueError from max()/min() on an empty array (:311-317), and a composed-but-absent filename dies later in ascii.read (see S9-R-14).",
    "expected": "An active, actionable error at this trust boundary, or a note explaining why the raw exception is preferred.",
    "failure_scenario": "A user with a wrong path_cooling / metallicity / rotation combination gets an unhandled IndexError, ValueError or FileNotFoundError from deep inside the loader instead of the actionable message that was written for exactly this case.",
    "repro": "Point path_cooling_nonCIE at a directory with no matching .dat and observe the error surfaced to the user.",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S9-B-24"]
  },
  {
    "id": "S9-R-21",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 114,
    "class": "other",
    "severity": "S4",
    "claim": "Prose bundle around the seam and the return convention: '10e5.5 K' where 10^5.5 is meant (an order of magnitude); a two-branch description of a three-branch switch; the claim that the non-CIE table top is 'exactly what our threshold is' when CIE_Tcutoff is by construction strictly greater than 5.5; and the negative-sign convention stated in only one of three return paths.",
    "evidence": "B: :114 vs :88; :116-117 vs :91/:167; :88 vs :49 plus the commented-out guard `if nonCIE_Tcutoff != CIE_Tcutoff:` at :134; :155 (sign comment, non-CIE branch only) with no equivalent at :158-165 or :167-197. A confirms the code side: three branches, the 5.5 literal duplicated at :43 and :53, and `-1 *` present in all three returns (:156, :165, :196) - so the sign convention is CONSISTENT in code and only the documentation is partial.",
    "expected": "'10^5.5 K'; a three-branch description; the sign convention stated once at the function contract rather than in one branch; and either both cutoffs derived from their tables or the 5.5 documented as a magic constant tied to the bundled table set.",
    "failure_scenario": "A reader implementing against the comments picks the wrong decade for the switch or assumes the CIE branch returns an unsigned cooling rate.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S9-B-11", "S9-B-12", "S9-B-10", "S9-B-04", "S9-A-20"]
  }
]
```
