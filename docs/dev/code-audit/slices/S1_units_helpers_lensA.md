# S1 units & helpers — Lens A (what the code does)

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

Slice read with all comments/docstrings blanked. Every claim below is derived from the arithmetic
alone plus physics I recomputed independently; no prose from the files was available to me.

**Coverage.** All seven files, all lines:
`trinity/_functions/unit_conversions.py` (1–604), `operations.py` (1–211), `simplify.py` (1–894),
`cluster.py` (1–61), `logging_setup.py` (1–590), `extract_example_snapshots.py` (1–116),
`__init__.py` (empty). All executable behaviour below was verified by running the code in an
isolated copy (stdlib + numpy only; a stub package tree was used so `operations.py`/`simplify.py`
could import `unit_conversions`). No file in `trinity/` was read or modified.

---

## 1. The internal unit system

The arithmetic is unambiguous: the package works in **M☉ / pc / Myr ("au" = astro units)**, with
**temperature left in kelvin** and **metallicity left in Z☉** (both map to `1.0` in
`convert2au`, `trinity/_functions/unit_conversions.py:377-378`). Every `*_cgs2au` constant is the
multiplicative factor that takes a value **from cgs to that system**, and `convert2au` returns the
same kind of factor (verified direction: `convert2au("cm") = 3.24e-19`, i.e. 1 cm = 3.24e-19 pc —
correct sense, not inverted).

The three base scales implied by the stored constants are:

| base | implied by | value the code uses | independent value | agreement |
|---|---|---|---|---|
| pc | `1/cm2pc` (:74) | 3.0856775814913674e18 cm | (648000/π)·1.495978707e13 = 3.0856775814913674e18 | **exact** (IAU 2015 parsec, IAU 2012 au) |
| Myr | `1/s2Myr` (:78) | 3.15576e13 s | 1e6 × 365.25 × 86400 = 3.15576e13 | **exact** (Julian year) |
| M☉ | `1/g2Msun` (:81) | 1.988409870698051e33 g | GM☉/G = 1.32712440018e26/6.67430e-8 = 1.9884098709677e33 | rel −1.4e-10 (IAU nominal M☉, standard) |

**Every one of the 18 compound constants is exactly derivable from these three to ≤1 ULP.** I
recomputed each from the code's *own* base scales (so the test is internal consistency, not my
choice of M☉) — see the table in §2. There is no constant in this file that is inconsistent with
the M☉/pc/Myr system.

Sanity check on the one composite that has a well-known literature value:
`G_cgs2au × G_cgs = 6.67430e-8 × 67400.3588611473 = 4.4985021514695545e-3` pc³ M☉⁻¹ Myr⁻²; divided
by `v_kms2au²` this is **4.300917e-3 pc M☉⁻¹ (km/s)²**, matching the textbook 4.30091e-3. `G_cgs2au`
is right.

---

## 2. Every unit constant, recomputed

Recomputed column = value derived from first principles using pc = 3.0856775814913674e18 cm,
Myr = 3.15576e13 s, M☉ = 1.988409870698051e33 g and the dimensional formula in the last column.

| line | constant | stored | recomputed | rel. err | dimension it *actually* converts |
|---|---|---|---|---|---|
| 74 | `cm2pc` | 3.240779289444365e-19 | 3.240779289444365e-19 | +0.0e+00 | cm → pc |
| 75 | `km2pc` | 3.240779289444365e-14 | 3.240779289444365e-14 | +0.0e+00 | km → pc |
| 78 | `s2Myr` | 3.168808781402895e-14 | 3.168808781402895e-14 | +0.0e+00 | s → Myr |
| 81 | `g2Msun` | 5.029144215870041e-34 | 5.029144215870041e-34 | +0.0e+00 | g → M☉ |
| 88 | `ndens_cgs2au` | 2.937998946096347e+55 | 2.9379989460963475e+55 | −1.9e-16 | cm⁻³ → pc⁻³ |
| 91 | `phi_cgs2au` | 3.0047272630641653e+50 | 3.0047272630641657e+50 | −1.4e-16 | **cm⁻² s⁻¹ → pc⁻² Myr⁻¹** (number flux, *not* erg s⁻¹ cm⁻²) |
| 94 | `E_cgs2au` | 5.260183968837699e-44 | 5.260183968837697e-44 | +3.8e-16 | erg → M☉ pc² Myr⁻² |
| 97 | `L_cgs2au` | 1.6599878161499254e-30 | 1.6599878161499253e-30 | +0.0e+00 | erg s⁻¹ → M☉ pc² Myr⁻³ |
| 100 | `pdot_cgs2au` | 1.623123174716277e-25 | 1.6231231747162772e-25 | −1.4e-16 | dyne (g cm s⁻²) → M☉ pc Myr⁻² |
| 103 | `pdotdot_cgs2au` | 5.122187189842638e-12 | 5.122187189842638e-12 | +0.0e+00 | g cm s⁻³ → M☉ pc Myr⁻³ |
| 106 | `G_cgs2au` | 67400.3588611473 | 67400.35886114725 | +6.5e-16 | cm³ g⁻¹ s⁻² → pc³ M☉⁻¹ Myr⁻² |
| 109 | `v_kms2au` | 1.022712165045695 | 1.022712165045695 | +0.0e+00 | km s⁻¹ → pc Myr⁻¹ |
| 110 | `v_cms2au` | 1.022712165045695e-05 | 1.0227121650456949e-05 | +1.7e-16 | cm s⁻¹ → pc Myr⁻¹ |
| 113 | `F_cgs2au` | 1.623123174716277e-25 | 1.6231231747162772e-25 | −1.4e-16 | dyne → M☉ pc Myr⁻² (byte-identical literal to `pdot_cgs2au`) |
| 116 | `Pb_cgs2au` | 1545441495671.806 | 1545441495671.806 | +0.0e+00 | dyn cm⁻² → M☉ pc⁻¹ Myr⁻² |
| 119 | `k_B_cgs2au` | 5.260183968837699e-44 | 5.260183968837697e-44 | +3.8e-16 | erg K⁻¹ → M☉ pc² Myr⁻² K⁻¹ (identical literal to `E_cgs2au`; correct, since K→K = 1) |
| 122 | `c_therm_cgs2au` | 5.122187189842638e-12 | 5.122187189842638e-12 | +0.0e+00 | erg s⁻¹ cm⁻¹ K⁻ⁿ → M☉ pc Myr⁻³ K⁻ⁿ (identical literal to `pdotdot_cgs2au`; correct, same dimension) |
| 125 | `dudt_cgs2au` | 4.877042454381257e+25 | 4.877042454381258e+25 | −1.8e-16 | erg cm⁻³ s⁻¹ → M☉ pc⁻¹ Myr⁻³ |
| 128 | `Lambda_cgs2au` | 5.650062667161655e-86 | 5.650062667161653e-86 | +3.8e-16 | erg cm³ s⁻¹ → M☉ pc⁵ Myr⁻³ |
| 131 | `tau_cgs2au` | 4788.452460043275 | 4788.452460043276 | −1.9e-16 | **g cm⁻² → M☉ pc⁻² (a surface mass density)** — see finding S1-A-02 |
| 134 | `gravPhi_cgs2au` | 1.045940172532453e-10 | 1.0459401725324526e-10 | +3.7e-16 | cm² s⁻² → pc² Myr⁻² (specific potential) |
| 137 | `grav_force_m_cgs2au` | 322743414.19646025 | 322743414.19646025 | +0.0e+00 | cm s⁻² → pc Myr⁻² (acceleration = force per mass) |

**Verdict: all 22 conversion constants are numerically correct to floating-point precision.** Three
pairs are deliberately the same literal and are dimensionally identical, so that is correct, not a
copy-paste error: `F_cgs2au ≡ pdot_cgs2au`, `k_B_cgs2au ≡ E_cgs2au`, `c_therm_cgs2au ≡
pdotdot_cgs2au`.

### 2b. CGS physical constants (`PhysicalConstantsCGS`, :202-226)

| line | constant | stored | first-principles value | rel. err |
|---|---|---|---|---|
| 202 | `G` | 6.67430e-8 | CODATA-2018 6.67430e-11 m³kg⁻¹s⁻² ×1e3 = 6.67430e-8 | **0** |
| 205 | `k_B` | 1.380649e-16 | SI-exact 1.380649e-23 J/K ×1e7 | **0** |
| 208 | `m_H` | 1.6735575e-24 | m_p+m_e−13.6 eV/c² = 1.6735328378e-24 | **+1.47e-05** ← see S1-A-01 |
| 211 | `m_p` | 1.67262192e-24 | 1.67262192369e-24 | −2.2e-09 (truncation) |
| 214 | `m_e` | 9.1093837e-28 | 9.1093837015e-28 | −1.6e-10 (truncation) |
| 217 | `c` | 2.99792458e10 | SI exact | **0** |
| 220 | `sigma_SB` | 5.670374e-5 | 5.670374419e-5 | −7.4e-08 (truncation) |
| 223 | `h` | 6.62607015e-27 | SI exact | **0** |
| 226 | `e` | 4.80320425e-10 | 1.602176634e-19 C × 2.99792458e9 = 4.803204713e-10 statC | −9.6e-08 (pre-2019 CODATA value) |

`e`, `sigma_SB`, `h`, `c` are not re-exported at :236-243 (only `G, k_B, m_H, m_p, m_e, c,
sigma_SB, h` are; `e` is reachable only via `CGS.e`).

### 2c. Derived module-level constants

* `Pb_au2_KcmInv = Pb_au2cgs / K_B_CGS` (:287) — dimensionally [erg cm⁻³]/[erg K⁻¹] = **K cm⁻³**.
  Correct; this is P/k_B. Value 4.6879e-4… (i.e. au pressure → K cm⁻³).
* `Mdot_au2Msunyr = 1e-6` (:289) — M☉ Myr⁻¹ → M☉ yr⁻¹. Exactly right, because the code's Myr is
  exactly 1e6 Julian years. **Correct.**
* `unit_map['m_H'] = CGS.m_H * CONV.g2Msun` (:375) = 8.416562021050926e-58 M☉. Carries the
  1.47e-5 `m_H` error into every `.param` value declared in units of `m_H`.

### 2d. Round-tripping

`InverseConversionConstants` (:159-180) stores each inverse as the float `1/x`, so A→B→A is
**not bit-exact** for 5 of the 21: `g2Msun`, `phi_cgs2au`, `pdot_cgs2au`, `F_cgs2au`,
`grav_force_m_cgs2au` all give `x*(1/x) = 1 − 1.11e-16` (1 ULP low). The other 16 round-trip to
exactly 1.0. This is ordinary float behaviour, not a defect, but it means a value that is converted
cgs→au→cgs is not guaranteed to compare equal to itself — relevant if any equality/regression test
relies on byte-identical round-trips.

Two derived-relation checks that are *not* float-exact (again ≤1 ULP, harmless but worth knowing if
anything asserts exact equality): `L_cgs2au ≠ E_cgs2au/s2Myr` (rel −2.1e-16),
`v_kms2au ≠ v_cms2au*1e5` (rel −2.2e-16), `dudt_cgs2au ≠ Pb_cgs2au/s2Myr` (rel −1.8e-16).

---

## 3. `convert2au` — what the parser actually does (`unit_conversions.py:315-477`)

Expression: strips **all** whitespace (:360); `None` or empty → `1.0`; splits on top-level `/`
(paren-aware, :389-411) into one numerator and N denominators (`a/b/c` ⇒ a·b⁻¹·c⁻¹, correct
left-association); splits each part on a single `*` that is neither preceded nor followed by `*`
(:421); parses each token as `name(**exponent)?` (:438); exponent parsed with
`Fraction(str.strip('()'))` so `1/2`, `-3/2`, `2.5` all work; multiplies `base_factor**exponent`,
negating the exponent for denominators. **No `eval`.** Verified outputs:

| input | returns | equals |
|---|---|---|
| `g*cm**2*s**-2` | 5.26018397e-44 | `E_cgs2au` ✓ |
| `g*cm**2*s**-3` | 1.65998782e-30 | `L_cgs2au` ✓ |
| `cm**3/g/s**2` | 67400.35886114728 | `G_cgs2au` (last digit differs, 1 ULP) |
| `erg*cm**3/s` | 5.650062667161655e-86 | `Lambda_cgs2au` ✓ exactly |
| `g*cm**-2` | 4788.452460043275 | `tau_cgs2au` ✓ exactly |
| `cm**-2*s**-1` | 3.0047272630641653e+50 | `phi_cgs2au` ✓ exactly — confirms `phi` is a *number* flux |
| `s**(-3/2)` | 1.77278452e+20 | = (1/s2Myr)^1.5 ✓ |
| `cm**(1/2)` | 5.692784283146838e-10 | = sqrt(cm2pc) ✓ |
| `K**-1`, `Zsun`, `Msun`, `pc`, `Myr`, `""`, `None` | 1.0 | dimensionless-by-convention |

Failure modes (all raise `UnitConversionError`, none silent): `erg/(s*cm**2)` → parentheses are
**only** supported around exponents, never around unit groups, because `split_units` splits the
group's `*` and then `(s` fails the name regex. `km s**-1` (space-separated, the astropy style) →
whitespace is stripped first, producing the bogus token `kms`. `yr`, `Msun/yr`, `dyne`, `eV` →
unknown unit. `cm**1/2` (unparenthesised fractional exponent) → misleading error `Cannot parse
unit: '2'`, because `/` is split before exponents are parsed.

Notable: there is **no `yr` unit** even though `Mdot_au2Msunyr` exists, so a `.param` cannot express
M☉/yr.

---

## 4. `operations.py` — search helpers

### `find_nearest(array, value)` (:19-28)
`np.abs(np.array(array) - value).argmin()`. Returns an **index, not a value**. Ties resolve to the
**first/lowest index** (`argmin` semantics) — verified: `find_nearest([1.,2.], 1.5) → 0`. A 2-D
array returns a **flattened** index (`find_nearest([[1,2],[3,4]], 3.9) → 3`), which the two callers
then use as a 1-D index. An empty array raises `ValueError: attempt to get argmin of an empty
sequence`. A scalar input returns 0.

### `find_nearest_lower(array, value)` (:30-65)
Guard: strict `monotonic()` (non-strict ≤ / ≥, so constant arrays pass) else `MonotonicError`.
Direction from `kindof_increasing(array)`. Then `idx = find_nearest`; if `array[idx] > value`, step
one index toward the lower side; clamp into `[0, len-1]`. Exact ties (`array[idx] == value`) are
**not** stepped — correct.
Verified behaviour on `[1,2,3,4,5]`: v=1.6→idx 0, v=3.0→idx 2, v=5.0→idx 4, v=9.0→idx 4 (all
correct); on `[5,4,3,2,1]`: v=3.5→idx 2 (correct). **Boundary defect:** v=0.5 (below every element)
returns idx 0, whose value 1.0 is *greater* than `value` — the clamp silently converts "no lower
element exists" into "the smallest element", with no signal. Same on the decreasing array.
**Duplicate defect:** on `[1,2,2,3]` with v=2.5 it returns idx 1, not the last index whose value
≤ 2.5 (idx 2). Value-equivalent here; index-inequivalent if the caller uses `idx` to index a
*parallel* array.

### `find_nearest_higher(array, value)` (:146-184)
Mirror image, **but with two deliberate differences from `find_nearest_lower`**: (a) the guard is
the lenient `_is_monotonic_or_tolerable`, not `monotonic`; (b) direction is taken from the endpoint
comparison `array[-1] >= array[0]` rather than `kindof_increasing`. Verified: `[1..5]` v=1.4→idx 1,
v=5.0→idx 4 (correct); v=99→idx 4, value 5.0 < 99 (same silent clamp as above).

### `_is_monotonic_or_tolerable` (:99-143)
`L` finite-checked; `n<2` or strictly monotonic → True. Otherwise `increasing = L[-1] >= L[0]`,
`wrong = step < 0` on the (possibly negated) diffs, then runs of consecutive `wrong` steps are
classified:
* **a run of length 1 is `continue`d unconditionally** (:131-134) — the drop-magnitude test is never
  applied to isolated single-step excursions. Verified: `[1,2,3,0.03,4,…]` (a 99% single-point
  plunge) and `[1,2,3,1e9,4,…]` (a 9-order-of-magnitude spike) are both reported **tolerable**.
* runs of length ≥2 must satisfy `|L[start]−L[end]| / max(|L[start]|,1e-300) ≤ 1e-2` **and**
  (`end ≤ boundary_cut` **or** run length ≤ 2). `boundary_cut = max(1, ceil(0.01·n))`, so for any
  array of ≤100 elements the "boundary" exemption covers only the very first step, and it only
  checks the **low-index** end, never the high-index end.
* Net effect: a **1-step** violation of arbitrary size passes; a **3-step** violation of 0.5% fails.
  Verified both.
* `if not wrong.any(): return True` (:119-120) is **unreachable**: reaching line 117 requires
  `not monotonic(L)`, which guarantees both a strictly-increasing and a strictly-decreasing adjacent
  pair, so `wrong` always has at least one True. Confirmed by 200 000 randomised non-monotonic
  arrays — zero hits.

### `get_soundspeed(T, params)` (:189-211)
`c_s = sqrt(gamma_adia · (k_B[au]·k_B_au2cgs) · T / (mu[au]·Msun2g)) · v_cms2au`, i.e. it converts
k_B and μ **out of** au into cgs, evaluates in cgs (cm/s), and converts the result back to pc/Myr.
Dimensionally exact; verified numerically against sqrt(5/3·k_B·1e6/(0.6 m_H)) = 151.3805 km/s to
all printed digits. μ is selected by a hard `if T > 1e4` (:206), which makes c_s **discontinuous**:
with μ_ion=0.6 m_H and μ_atom=1.27 m_H, c_s jumps 10.405 → 15.138 km/s (+45%) across T = 1e4 K, and
T = 1e4 exactly takes the *atomic* branch. `T` must be a scalar — an array raises
`ValueError: truth value of an array … is ambiguous`.

`MonotonicError` (:186) is defined *after* both of its `raise` sites (:42, :159); legal at runtime.

---

## 5. `simplify.py` — downsampler

`_simplify` is a curve-decimation routine, not physics; but it decides which snapshots survive to
output, so a silent drop is a silent physics loss.

**Curvature (:524-544) is mathematically correct.** With u = (x_i−x_{i-1}, y_i−y_{i-1}) and
w = (x_{i+1}−x_i, y_{i+1}−y_i), the expression `dx1*(dy1+dy2) − dy1*(dx1+dx2)` algebraically reduces
to `dx1·dy2 − dy1·dx2` = u×w, and `kappa = 2|u×w|/(|u||w||u+w|)` is exactly the Menger curvature.
The `+1` at :544 correctly maps the diff index to the vertex index. Coordinates are pre-normalised
by the data ranges, so the default `grad_inc = 1.0` means "radius of curvature < one full data
range". The parameter is named `grad_inc` but is compared against a **curvature**, not a gradient
increment.

**Sparse-table RMQ (:86-120) is correct.** `k_max = floor(log2(n))+1` is exactly sufficient for the
longest query; the `st[k, limit:] = st[k-1, limit:]` tail fill writes entries that no valid query
can read (I proved `lo + 2^k ≤ n` for every query and confirmed by brute force: 0 mismatches over
all (lo,hi) for n ∈ {1,2,3,5,8,17,33}).

**`_prev_next_strict` (:24-83) is asymmetric by construction**: `prev` is *strictly* greater
(`>`), `next` is greater-**or-equal** (`≥`). Verified on `[1,3,3,1]`:
`prev_greater=[-1,-1,-1,2]`, `next_greater=[1,2,4,4]`. This is the standard plateau-disambiguation
trick, but the name says "strict" for both.

**`_peak_prominences` (:123-232) matches scipy for interior peaks** (verified `[0,1,0,3,0,2,0]` →
`[1,3,2]`, identical to `scipy.signal.peak_prominences`) but returns **0** in two cases where a real
prominence exists, because a one-sided search interval degenerates to empty and `np.full(..., inf)`
propagates through `np.maximum` to `y − inf = −inf`, then gets clipped to 0 at :231:
* a peak at index 0 or n−1 (verified: `[5,1,2,0,3]`, idx 0 — the global maximum — gets prominence 0);
* the **leading** point of a plateau maximum (verified: `[0,5,5,0]`, idx `[1,2]` → `[0, 5]`; scipy
  gives `[5,5]`).
Consequence inside `_simplify`: such points fail the `prom ≥ 0.05·y_range` filter (:587) and are not
promoted to the mandatory set. Indices 0 and n−1 are separately mandatory (:631-632, :714), so the
practical exposure is the plateau case.

**Dedup (:466-479) is neighbour-pairwise, not anchor-based.** A point is dropped when *both*
`|Δx| ≤ 1e-6·(x_max−x_min)` and `|Δy| ≤ 1e-6·(y_max−y_min)` **relative to its immediate predecessor**
— so an arbitrarily large cumulative drift built out of small steps is removed. Verified: a smooth
2 000 000-point ramp `x=y=linspace(0,1,2e6)` (step = 5e-7 of range) returns **exactly one point,
`x=[0.], y=[0.]`** — the final point is lost too, because after dedup `x.size == 1` triggers the
early return at :496-497 which bypasses the endpoint-preservation at :631-632. At 200 000 points
(step 5e-6 > tol) the same call correctly returns 100 points. The trip-wire is roughly
n ≳ 1/dedup_tol = 1e6 samples on a smooth curve.

**Other confirmed behaviours of `_simplify`:**
* `nmin` is silently floored to 20 (:502) *after* the `nmin >= x.size` early return. Requesting
  `nmin=5` on 1000 points returned 26.
* Output is returned in **original array order**, not x-sorted (`_restore` :489 sorts the *original
  positions*). Verified: `x=[3,1,2,5,4]` → out `[3,1,2,5]`. Callers that assume monotone x get
  unsorted data. (`_simplify_error` :827-830 defensively re-sorts, so it is safe.)
* `idx_dist` (:625) is off by one relative to the arc-length grid: `bins[j]` is the bin of **point
  j+1**, so `np.where(bins[:-1] != bins[1:])` records point **j**, one index *before* the crossing,
  and can never record the last two indices. In the 5-point example above this is why interior point
  index 3 is dropped even though the output (4 points) is far under the budget (20) — when
  `len(merged) ≤ nmin` there is no top-up path (:655).
* The budget block (:655-728) is otherwise correct: `np.unique(..., return_index=True)` +
  `np.sort(unique_pos)` is the correct first-occurrence-order dedup idiom, and because the mandatory
  block (endpoints, prominent, coverage) is concatenated first, `priority_indices[:budget]` with
  `budget ≥ mandatory_set.size` always retains the mandatory set. The bisection BFS produces
  distinct interior midpoints; `order[count:]` is left uninitialised on the early `break` but is
  never read.

**`_simplify_error` (:754-894):** `np.interp` clamps (does not extrapolate) outside `x_s`. Empty
inputs raise `ValueError: array of sample points is empty` from `np.interp` rather than being
guarded. `r_squared` returns **1.0 (perfect)** whenever `y_o` is constant, regardless of how wrong
the simplification is (:854). Log metrics require all of `y_o, y_s > 0` and otherwise return NaN;
they interpolate log10(y) linearly in x, consistent with their names.

---

## 6. `cluster.py`, `logging_setup.py`, `extract_example_snapshots.py`

`cluster.py:28-45` — env vars `SLURM_CPUS_PER_TASK`/`SLURM_CPUS_ON_NODE` (rejects non-digit and 0),
then `sched_getaffinity`, then `cpu_count()`. `get_optimal_workers` (:48-61) returns
`max(1, cpu_count//2 - 1)` off-SLURM: 1 worker on a 2- or 4-core box, 3 on 8 cores. Note the
affinity/cpu_count detection path in `detect_allocated_cpus` is reachable **only** under
`SLURM_JOB_ID`.

`logging_setup.py` — `ColoredFormatter.format` (:62-76) **mutates the shared `LogRecord`** by
overwriting `record.levelname` and `record.name` with ANSI-wrapped strings. Because the console
handler is registered first (:270) and the file handler second (:302), the file handler formats the
*already-mutated* record. Verified end-to-end with a TTY-reporting stdout: the log file line is
`2026-07-30 … | \x1b[33mWARNING \x1b[0m | \x1b[94mroot\x1b[0m | shell temperature clamped`. **ANSI
escapes are written into the persisted `.log` file.**
Also: `setup_logging` calls `root_logger.handlers.clear()` (:243), wiping any handlers a caller
already installed; `getattr(logging, level.upper())` (:236, :388) raises a bare `AttributeError` on
an unrecognised level name and does not check that the attribute is an int;
`DedupWarningFilter` (:79-109) suppresses **every** repeat of an identical WARNING+ message for the
process lifetime, so a clamp that fires 10 000 times is reported once with no count.

`extract_example_snapshots.py` — `REPO_ROOT = parents[2]` is correct for `trinity/_functions/…`.
Output is hard-wired to `REPO_ROOT/outputs/mockOutput/<folder-name>` (:87). `main` (:107) uses
`__doc__.strip()` and would raise `AttributeError` if run with `python -OO` or if the module
docstring were removed. `_is_terminated` treats a falsy-but-present `SimulationEndReason` (e.g. `""`
or `0`) as not-terminated.

---

## 7. Every bare numeric literal in arithmetic

| file:line | literal | expression it sits in |
|---|---|---|
| unit_conversions.py:289 | `1e-6` | `Mdot_au2Msunyr` (M☉/Myr → M☉/yr) — correct |
| unit_conversions.py:375 | — | `CGS.m_H * CONV.g2Msun` (m_H in M☉) |
| unit_conversions.py:377-381 | `1.0` ×5 | K, Zsun, Msun, pc, Myr treated as dimensionless |
| unit_conversions.py:429, 467 | `1.0` | `total_factor` seed; default exponent |
| unit_conversions.py:508 | `1e-20` | `abs(result-expected) < 1e-20` self-test tolerance — **absolute**, so for `g2Msun` (5e-34) and `cm2pc` (3.2e-19) this test passes no matter how wrong the constant is |
| unit_conversions.py:524, 578 | `1e-10` | relative tolerances (these ones are meaningful) |
| operations.py:94-96 | `1e-2`, `0.01`, `2` | `MONOTONIC_RTOL`, `BOUNDARY_FRAC`, `MAX_SPIKE_LEN` |
| operations.py:137 | `1e-300` | `drop / max(abs(L[start]), 1e-300)` divide-by-zero floor |
| operations.py:206 | `1e4` | `if T > 1e4` ionised/atomic μ switch |
| simplify.py:96 | `1`, `2` | `k_max = max(1, floor(log2(max(n,1)))+1)`; `1 << k` |
| simplify.py:243 | `20` | `_COVERAGE_CHUNKS` |
| simplify.py:278-279 | `0.5` | chunk-centre midpoints |
| simplify.py:287 | `1e-6` | `_DEDUP_TOL_DEFAULT` |
| simplify.py:294-296 | `100`, `1.0`, `0.9` | `nmin`, `grad_inc` (curvature threshold), `warn_below_r2` |
| simplify.py:502 | `20` | `nmin = max(int(nmin), 20)` silent floor |
| simplify.py:521-522, 538 | `1e-30` | range and denominator floors |
| simplify.py:540 | `2.0` | Menger curvature numerator (correct) |
| simplify.py:581 | `0.05` | `prom_thresh_frac` (5% of y-range) |
| simplify.py:619 | — | `maxdist = total_arc / nmin` |
| simplify.py:699 | `0` | `min(_COVERAGE_CHUNKS, max(0, nmin-2))` |
| simplify.py:844 | `1e-30` | `eps` mask for relative error |
| cluster.py:61 | `2`, `1` | `(os.cpu_count() or 1) // 2 - 1` |

---

## 8. Reproduction

All checks were run against verbatim copies of the slice files in
`…/scratchpad/pkg/trinity/_functions/` (a stub package so the intra-package import resolves), e.g.:

```
python3 -c "import uc; print(uc.convert2au('cm**-2*s**-1'), uc.CONV.phi_cgs2au)"
PYTHONPATH=pkg python3 -c "
from trinity._functions.operations import _is_monotonic_or_tolerable as t
print(t([1.,2.,3.,1e9,4.,5.,6.,7.,8.,9.]))"          # -> True
PYTHONPATH=pkg python3 -c "
import numpy as np; from trinity._functions.simplify import _simplify
n=2_000_000; x=np.linspace(0,1,n); print(_simplify(x,x,nmin=100)[0].size)"   # -> 1
```

---

```json
[
  {
    "id": "S1-A-01",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 208,
    "class": "coefficient",
    "severity": "S4 hygiene",
    "claim": "m_H = 1.6735575e-24 g is 1.47e-5 (relative) larger than the hydrogen-atom mass; it is correct only to 5 significant figures.",
    "evidence": "m_p + m_e - 13.6 eV/c^2 = 1.67262192369e-24 + 9.1093837015e-28 - 2.42e-29 = 1.673532838e-24 g. Stored 1.6735575e-24 => rel +1.474e-05. It is not m_p+m_e (1.67353286e-24), not 1.008/N_A (1.673823e-24), not 1.00794/N_A (1.673724e-24), and not 1.00782503 u (1.673533e-24). Stored value minus the file's own m_p is 9.3558e-28, vs m_e = 9.1094e-28.",
    "expected": "1.6735328e-24 g (or an explicit statement of which convention is intended).",
    "failure_scenario": "Every mass-per-particle quantity scales by 1+1.5e-5: mu_ion/mu_atom, hence sound speed by 7.4e-6, and every .param value declared in units of 'm_H' (unit_map at line 375). Far below any physical uncertainty, but it is a systematic bias, not a rounding of the intended value.",
    "repro": "python3 -c \"mp=1.67262192369e-24; me=9.1093837015e-28; Eb=13.605693123*1.602176634e-12/(2.99792458e10)**2; print((1.6735575e-24-(mp+me-Eb))/(mp+me-Eb))\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-02",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 131,
    "class": "units",
    "severity": "S3 misleading",
    "claim": "`tau_cgs2au` holds the conversion for a SURFACE MASS DENSITY (g cm^-2 -> Msun pc^-2), not for an optical depth, which is dimensionless and would convert with factor 1.",
    "evidence": "4788.452460043275 == g2Msun * pc2cm**2 == convert2au('g*cm**-2') exactly (bit-identical). Recomputed: 5.029144215870041e-34 * (3.0856775814913674e18)**2 = 4788.452460043276.",
    "expected": "A name matching the quantity, e.g. sigma_cgs2au / SurfaceDensity; a genuinely dimensionless tau needs factor 1.0.",
    "failure_scenario": "Anyone applying `tau_cgs2au` to an actual optical depth scales a dimensionless number by 4788. Conversely, a reviewer checking 'tau should be 1' would wrongly flag the constant. The stored number itself is arithmetically correct for g/cm^2.",
    "repro": "python3 -c \"import uc; print(uc.convert2au('g*cm**-2') == uc.CONV.tau_cgs2au)\"  # True",
    "confidence": "high"
  },
  {
    "id": "S1-A-03",
    "file": "trinity/_functions/operations.py",
    "line": 131,
    "class": "silent-failure",
    "severity": "S2 latent",
    "claim": "`_is_monotonic_or_tolerable` accepts a single-step monotonicity violation of ARBITRARY magnitude: the `if end - start == 1: continue` branch skips the run before the rtol drop test at line 137 is ever reached.",
    "evidence": "_is_monotonic_or_tolerable([1,2,3,1e9,4,5,6,7,8,9]) -> True (a 9-order-of-magnitude spike). _is_monotonic_or_tolerable([1,2,3,0.03,4,...]) -> True (a 99% plunge). By contrast a 3-step 0.5% wobble at line 137-142 -> False. The MONOTONIC_RTOL=1e-2 gate therefore applies only to runs of length >= 2.",
    "expected": "Either the drop test applies to all runs including length 1, or the single-step exemption is bounded by rtol as the length-2 case is.",
    "failure_scenario": "`find_nearest_higher` (line 157) is the sole consumer. On an array with one large glitch it proceeds, and the +/-1 step logic then returns a neighbouring index without verifying it brackets `value`: find_nearest_higher([1,2,3,100,4,5,6,7,8,9], 50.0) returns idx 9 (value 9.0) even though idx 3 holds 100 >= 50. A cooling/SPS table lookup would silently interpolate off the wrong bracket.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import find_nearest_higher as f; a=[1.,2.,3.,100.,4.,5.,6.,7.,8.,9.]; print(f(a,50.), a[f(a,50.)])\"  # 9 9.0",
    "confidence": "high"
  },
  {
    "id": "S1-A-04",
    "file": "trinity/_functions/operations.py",
    "line": 60,
    "class": "silent-failure",
    "severity": "S2 latent",
    "claim": "`find_nearest_lower` and `find_nearest_higher` clamp out-of-range requests to the nearest end index and return it with no signal, so the returned element can violate the relation the function name promises.",
    "evidence": "find_nearest_lower(np.linspace(1,5,5), 0.1) -> idx 0, value 1.0 (1.0 > 0.1, no element <= 0.1 exists). find_nearest_higher(np.linspace(1,5,5), 99.) -> idx 4, value 5.0 (5.0 < 99). Same on decreasing input. The clamps are lines 60-63 and 179-182.",
    "expected": "Raise, or return a sentinel, when no element satisfies the requested inequality — or document that callers must range-check.",
    "failure_scenario": "A table lookup for a T, n, or age outside the tabulated grid is silently answered with the edge cell, so the run continues on extrapolated-as-constant physics instead of stopping. This is exactly the class of error that produces plausible-looking but wrong trajectories.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.operations import find_nearest_lower as f; a=np.linspace(1.,5.,5); print(f(a,0.1), a[f(a,0.1)])\"  # 0 1.0",
    "confidence": "high"
  },
  {
    "id": "S1-A-05",
    "file": "trinity/_functions/simplify.py",
    "line": 466,
    "class": "silent-failure",
    "severity": "S2 latent",
    "claim": "The dedup filter compares each point to its immediate predecessor rather than to the last kept point, so a curve made of many sub-tolerance steps is erased in full — including its final point, because the collapsed array then hits the `nmin >= x.size` early return that bypasses endpoint preservation.",
    "evidence": "_simplify(np.linspace(0,1,2_000_000), np.linspace(0,1,2_000_000), nmin=100) returns arrays of size 1, x=[0.], y=[0.] — the endpoint (1.0, 1.0) is gone. Mechanism: line 473 keeps point i only if |x[i]-x[i-1]| > 1e-6*range_x OR |y[i]-y[i-1]| > 1e-6*range_y; for n=2e6 the uniform step is 5e-7 of range, so every interior point fails both, leaving x.size==1, which makes line 496 (`nmin >= x.size`) return before mask[0]/mask[-1] at 631-632 are ever set. The same call at n=200000 correctly returns 100 points.",
    "expected": "Anchor the tolerance to the last retained point, and/or force-retain index x.size-1 on the early-return path.",
    "failure_scenario": "Any recorded quantity sampled densely enough (>~1/dedup_tol = 1e6 samples, or a locally stalled integrator segment) is written to output as a single point, losing the entire trajectory silently — no warning fires because the R^2 check at line 736 requires merged.size >= 2.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _simplify; n=2_000_000; x=np.linspace(0,1,n); xs,ys=_simplify(x,x,nmin=100); print(xs.size, xs, ys)\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-06",
    "file": "trinity/_functions/logging_setup.py",
    "line": 68,
    "class": "state",
    "severity": "S3 misleading",
    "claim": "`ColoredFormatter.format` mutates the shared LogRecord in place; since the console handler is added before the file handler, ANSI escape sequences are written into the persisted log file.",
    "evidence": "Lines 68 and 71 assign back to record.levelname / record.name. Handler order: console addHandler at line 270, file addHandler at line 302. With a TTY-reporting stdout and use_colors=True, the written log line is b'2026-07-30 ... | \\x1b[33mWARNING \\x1b[0m | \\x1b[94mroot\\x1b[0m | shell temperature clamped'.",
    "expected": "Format a copy (e.g. build the coloured string locally, or copy.copy(record)) so downstream handlers see the unmodified record; the file should be plain text.",
    "failure_scenario": "Archived run logs contain escape bytes, so grep/diff/parse of trinity_*.log across runs is corrupted and log comparison in a regression workflow gives false differences.",
    "repro": "PYTHONPATH=pkg python3 -c \"import sys,io,os,tempfile,logging; from trinity._functions.logging_setup import setup_logging;\\nclass T(io.StringIO):\\n  def isatty(self): return True\\nsys.stdout=T(); d=tempfile.mkdtemp(); l=setup_logging('INFO',True,True,d,'t.log',True); l.warning('x'); [h.flush() for h in logging.getLogger().handlers]; sys.stdout=sys.__stdout__; print(b'\\\\x1b[' in open(os.path.join(d,'t.log'),'rb').read())\"  # True",
    "confidence": "high"
  },
  {
    "id": "S1-A-07",
    "file": "trinity/_functions/simplify.py",
    "line": 186,
    "class": "numerical",
    "severity": "S3 misleading",
    "claim": "`_peak_prominences` returns 0 for the leading point of a plateau extremum and for extrema at index 0 / n-1, because the empty one-sided search interval leaves `left_min`/`right_min` at +/-inf, which propagates through np.maximum to -inf and is then clipped to 0.",
    "evidence": "_peak_prominences([0,5,5,0], [1,2]) -> [0., 5.]; scipy.signal.peak_prominences gives [5., 5.]. _peak_prominences([5,1,2,0,3], [0]) -> [0.] although index 0 is the global maximum. Interior non-plateau peaks agree with scipy exactly ([0,1,0,3,0,2,0], idx [1,3,5] -> [1,3,2] both). Lines 186-187 seed inf/-inf, line 200/226 combine with maximum/minimum, line 231 clips.",
    "expected": "Treat an empty side as 'no constraint' (use the other side alone), matching scipy semantics.",
    "failure_scenario": "In _simplify, points whose prominence is spuriously 0 fail the `proms >= 0.05*y_range` filter at line 587 and are not promoted to the mandatory set, so the leading edge of a flat-topped feature (e.g. a plateau in shell velocity or bubble temperature) can be dropped from the saved trajectory while its trailing edge is kept.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _peak_prominences as p; print(p(np.array([0.,5.,5.,0.]), np.array([1,2])))\"  # [0. 5.]",
    "confidence": "high"
  },
  {
    "id": "S1-A-08",
    "file": "trinity/_functions/operations.py",
    "line": 119,
    "class": "deadcode",
    "severity": "S4 hygiene",
    "claim": "`if not wrong.any(): return True` at lines 119-120 is unreachable.",
    "evidence": "Reaching line 117 requires n>=2 and `not monotonic(L)`. Not-monotonic means both `not all(x<=y)` and `not all(x>=y)`, i.e. there exists a strictly decreasing adjacent pair AND a strictly increasing one. Whichever branch of line 117 is taken, `step < 0` is therefore True somewhere. A randomised search over 200000 non-monotonic small arrays produced 0 hits on this branch.",
    "expected": "Remove, or note it as defensive.",
    "failure_scenario": "",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np,random; from trinity._functions.operations import monotonic; h=0\\nfor _ in range(200000):\\n  L=np.array([random.choice([0.,1.,2.,3.]) for _ in range(random.randint(2,6))])\\n  if monotonic(L): continue\\n  s=np.diff(L) if L[-1]>=L[0] else -np.diff(L)\\n  h+= (not (s<0).any())\\nprint(h)\"  # 0",
    "confidence": "high"
  },
  {
    "id": "S1-A-09",
    "file": "trinity/_functions/simplify.py",
    "line": 625,
    "class": "numerical",
    "severity": "S4 hygiene",
    "claim": "`idx_dist` is off by one: `bins[j]` is the arc-length bin of point j+1, so `np.where(bins[:-1] != bins[1:])` records index j — the point BEFORE each bin crossing — and can never record the last two indices.",
    "evidence": "s_cum = np.cumsum(ds) has length n-1 with s_cum[j] = arc length from point 0 to point j+1 (line 608). bins = (s_cum/maxdist).astype(int) (line 624) therefore indexes points 1..n-1, but the returned index is the diff position j, i.e. point j. Demonstration: _simplify([3,1,2,5,4],[9,1,4,25,16],nmin=3) returns 4 of 5 points (interior x=4 dropped) even though the effective nmin is 20, because idx_dist yields {0,1,2} instead of {1,2,3}.",
    "expected": "`np.where(bins[:-1] != bins[1:])[0] + 1` if the intent is 'the first point of each new arc-length bin'.",
    "failure_scenario": "Uniform arc-length coverage of the saved trajectory is systematically shifted one sample early and the tail of the curve is under-sampled. Cosmetic for a decimator, but it also means output points are not the ones nominally chosen.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _simplify; x=np.array([3.,1.,2.,5.,4.]); print(_simplify(x,x**2,nmin=3)[0])\"  # [3. 1. 2. 5.]",
    "confidence": "medium"
  },
  {
    "id": "S1-A-10",
    "file": "trinity/_functions/simplify.py",
    "line": 489,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "`_simplify` sorts its working copy by x but `_restore` re-sorts by ORIGINAL position, so the returned arrays are in input order and are not monotone in x when the input was not.",
    "evidence": "_simplify(np.array([3.,1.,2.,5.,4.]), x**2, nmin=3) returns x = [3., 1., 2., 5.] — not sorted. Line 486 maps working indices through dedupe_idx, line 488 through sort_order back to original positions, line 489 sorts those positions.",
    "expected": "Either document that input order is preserved, or return x-sorted output (the internal R^2 check at line 739 already assumes sorted x for np.interp).",
    "failure_scenario": "A consumer that plots or np.interp's the simplified output assumes increasing x and gets a self-crossing curve / wrong interpolation. _simplify_error defends against this (line 827) but external consumers may not.",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np; from trinity._functions.simplify import _simplify; x=np.array([3.,1.,2.,5.,4.]); xs,_=_simplify(x,x**2,nmin=3); print(xs, bool(np.all(np.diff(xs)>=0)))\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-11",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 508,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "The self-test at line 508 uses an ABSOLUTE tolerance of 1e-20 against constants of magnitude 1e-19 to 1e-34, so it passes regardless of the constants' values.",
    "evidence": "`passed = abs(result - expected) < 1e-20` compared against cm2pc = 3.24e-19 (tolerance is 3% of the value), s2Myr = 3.17e-14, g2Msun = 5.03e-34 and km2pc = 3.24e-14 (tolerance is 3e5 times the value — unconditionally true). The compound tests at line 524 and the astropy tests at line 578 correctly use relative tolerances.",
    "expected": "Relative comparison, e.g. abs(result-expected)/expected < 1e-12, as used at lines 524 and 578.",
    "failure_scenario": "The block runs only under `python unit_conversions.py`, so it does not weaken pytest, but it gives a false 'passed' signal to anyone hand-checking the base conversions.",
    "repro": "python3 -c \"print(abs(3.24e-14 - 0.0) < 1e-20)\"  # False, but abs(km2pc - km2pc) trivially passes; substitute a wrong km2pc to see it still pass at 1e-20 absolute for g2Msun",
    "confidence": "high"
  },
  {
    "id": "S1-A-12",
    "file": "trinity/_functions/operations.py",
    "line": 206,
    "class": "regime",
    "severity": "S4 hygiene",
    "claim": "`get_soundspeed` switches the mean molecular weight on a hard `T > 1e4` test, producing a step discontinuity in c_s, and it is scalar-only.",
    "evidence": "With mu_ion = 0.6 m_H and mu_atom = 1.27 m_H, c_s(1e4) = 10.405 km/s (atomic branch, since the test is strict >) and c_s(1.00001e4) = 15.138 km/s — a 45% jump across a single kelvin. An array T raises ValueError: truth value of an array with more than one element is ambiguous. The formula itself is correct: verified c_s(1e6 K) = 151.3805 km/s against sqrt(5/3 k_B T / (0.6 m_H)) = 151.38055 km/s.",
    "expected": "A smoothed or explicitly documented ionisation transition, and np.where for array support if any caller passes arrays.",
    "failure_scenario": "A solver whose residual depends on c_s sees a jump discontinuity if the shell temperature crosses 1e4 K; root-finders and adaptive steppers can stall or chatter at that boundary.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import get_soundspeed as g; import trinity._functions.unit_conversions as c; p={'mu_ion':0.6*c.CGS.m_H*c.g2Msun,'mu_atom':1.27*c.CGS.m_H*c.g2Msun,'gamma_adia':5/3.,'k_B':c.CGS.k_B*c.k_B_cgs2au}; print(g(1e4,p)*c.v_au2kms, g(1.00001e4,p)*c.v_au2kms)\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-13",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 389,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "`convert2au` cannot parse parenthesised unit groups, has no 'yr' unit despite Mdot_au2Msunyr existing, and rejects space-separated units because whitespace is stripped before tokenising.",
    "evidence": "convert2au('erg/(s*cm**2)') -> UnitConversionError: Cannot parse unit: '(s'. The paren-depth tracking in split_by_slash (lines 396-401) is useful only for '/' inside exponents such as 's**(-3/2)' (which works). convert2au('km s**-1') -> Unknown unit 'kms' (line 360 strips whitespace, so the two tokens fuse). convert2au('yr') and convert2au('Msun/yr') -> Unknown unit 'yr'. convert2au('cm**1/2') -> the misleading 'Cannot parse unit: 2' because '/' is split before exponents.",
    "expected": "Either support grouped denominators / a 'yr' entry, or the unit-string forms accepted by .param files must be constrained to the supported subset.",
    "failure_scenario": "All of these raise loudly rather than converting wrongly, so no silent unit error. The exposure is a .param that a user writes in a natural astropy style being rejected at parse time.",
    "repro": "python3 -c \"import uc; \\nfor s in ['erg/(s*cm**2)','km s**-1','Msun/yr','cm**1/2']:\\n  try: print(s, uc.convert2au(s))\\n  except Exception as e: print(s,'->',e)\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-14",
    "file": "trinity/_functions/logging_setup.py",
    "line": 106,
    "class": "silent-failure",
    "severity": "S4 hygiene",
    "claim": "`DedupWarningFilter` suppresses every repeat of an identical WARNING+ message for the lifetime of the process, with no count, so a clamp or fallback that fires thousands of times is reported once.",
    "evidence": "filter() stores (levelno, getMessage()) in self._seen and returns False on any repeat (lines 103-109). Verified: three identical WARNING records -> True, False, False. Attached to both the console (line 269) and file (line 300) handlers.",
    "expected": "Emit a final count, or dedup with a periodic re-emit, so the frequency of a physics clamp is observable.",
    "failure_scenario": "A warning like 'Temperature below minimum, clamping to 1e4 K' appearing once in the log reads as a one-off, when it may have fired at every step of the integration — hiding how much of a run is running on clamped physics.",
    "repro": "PYTHONPATH=pkg python3 -c \"import logging; from trinity._functions.logging_setup import DedupWarningFilter as D; f=D(); r=logging.LogRecord('n',logging.WARNING,'p',1,'same',None,None); print(f.filter(r), f.filter(r), f.filter(r))\"  # True False False",
    "confidence": "high"
  },
  {
    "id": "S1-A-15",
    "file": "trinity/_functions/operations.py",
    "line": 163,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "The two sibling lookups use different monotonicity contracts and different direction tests, so they can disagree about the same array.",
    "evidence": "find_nearest_lower (line 40) requires strict `monotonic()` and takes direction from `kindof_increasing(array)`; find_nearest_higher (lines 157, 163) uses the lenient `_is_monotonic_or_tolerable` and takes direction from the endpoint comparison `array[-1] >= array[0]`. On [1,2,3,100,4,5,6,7,8,9] find_nearest_lower raises MonotonicError while find_nearest_higher returns an answer.",
    "expected": "One contract shared by both, or an explicit reason for the asymmetry.",
    "failure_scenario": "Code that brackets a value by calling both helpers gets an exception from one and a (possibly wrong, see S1-A-03) index from the other on the same array.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.operations import find_nearest_lower as lo, find_nearest_higher as hi; a=[1.,2.,3.,100.,4.,5.,6.,7.,8.,9.]; print(hi(a,4.2));\\ntry: lo(a,4.2)\\nexcept Exception as e: print(type(e).__name__)\"",
    "confidence": "high"
  },
  {
    "id": "S1-A-16",
    "file": "trinity/_functions/simplify.py",
    "line": 854,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "`_simplify_error` reports r_squared = 1.0 (perfect reconstruction) whenever y_o is constant, and raises an unguarded numpy ValueError on empty input.",
    "evidence": "Line 854: `r_squared = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 1.0`. _simplify_error([0,1,2],[5,5,5],[0,2],[5,5]) -> r_squared 1.0. _simplify_error([],[],[],[]) -> ValueError: array of sample points is empty (from np.interp at line 833, before the np.max at line 839). The same ss_tot>0 guard exists at line 738 inside _simplify.",
    "expected": "Return NaN (or 0.0) for an undefined R^2, and guard the empty case explicitly.",
    "failure_scenario": "A quality gate keyed on r_squared >= threshold passes unconditionally for any flat series, so a decimation that lost a constant-but-important channel is scored as perfect.",
    "repro": "PYTHONPATH=pkg python3 -c \"from trinity._functions.simplify import _simplify_error as e; print(e([0.,1.,2.],[5.,5.,5.],[0.,2.],[5.,5.])['r_squared'])\"  # 1.0",
    "confidence": "high"
  },
  {
    "id": "S1-A-17",
    "file": "trinity/_functions/simplify.py",
    "line": 502,
    "class": "other",
    "severity": "S4 hygiene",
    "claim": "`nmin` is silently raised to 20 after the early-return check, so a caller asking for fewer points gets more, without a warning.",
    "evidence": "Line 496 returns everything when nmin >= x.size; line 502 then does nmin = max(int(nmin), 20). _simplify(np.linspace(0,1,1000), np.sin(20*x), nmin=5) returned 26 points. Also note the curvature threshold parameter is named `grad_inc` (line 294) but is compared against the Menger curvature kappa at line 544, not a gradient increment.",
    "expected": "Warn on the clamp, or document the floor of 20 as part of the contract.",
    "failure_scenario": "",
    "repro": "PYTHONPATH=pkg python3 -c \"import numpy as np,warnings; from trinity._functions.simplify import _simplify; x=np.linspace(0,1,1000); warnings.simplefilter('ignore'); print(_simplify(x,np.sin(20*x),nmin=5)[0].size)\"  # 26",
    "confidence": "high"
  },
  {
    "id": "S1-A-18",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 141,
    "class": "deadcode",
    "severity": "S4 hygiene",
    "claim": "`__post_init__` unpacks `__dataclass_fields__.items()` into `field_name, value` but never uses `value` (it re-reads via getattr), so `value` holds a dataclasses.Field object and is dead.",
    "evidence": "Line 141 `for field_name, value in self.__dataclass_fields__.items():` followed by line 142 `field_value = getattr(self, field_name)` and line 143 testing field_value. `value` is unused. Note also that InverseConversionConstants has no equivalent positivity check, and the check itself only enforces `> 0`, not correctness.",
    "expected": "`for field_name in self.__dataclass_fields__:`",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  }
]
```
