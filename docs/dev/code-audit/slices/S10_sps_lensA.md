# S10 SPS feedback — Lens A (what the code does)

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

Slice read (comments/docstrings blanked): `trinity/sps/read_sps.py`, `trinity/sps/sps_columns.py`,
`trinity/sps/update_feedback.py`, `trinity/sps/__init__.py` (empty, one newline).

**Shared read-only exception used:** `trinity/_functions/unit_conversions.py` (S1 slice copy) — needed to
resolve the numeric value and direction of `cvt.s2Myr`, `cvt.L_cgs2au`, `cvt.pdot_cgs2au`, `cvt.g2Msun`,
`cvt.cm2pc`, `cvt.v_kms2au`.

Everything below is read off the arithmetic. No physics judgement is offered.

---

## 0. Shape of the module

Three stages, no class state:

1. `sps_columns.load_user_columns` — sniff file layout, `np.loadtxt`, pull raw float columns by
   integer index or header name.
2. `read_sps._read_sps_user` — de-log, unit-convert to "au", multiply mass-scaled columns by `f_mass`,
   derive the wind/SN quadruples, prepend a `t = 0` row, return an **11-element list**.
3. `read_sps.get_interpolation` — build 10 independent `scipy.interpolate.interp1d(kind='cubic')`
   splines over `t`; `update_feedback.get_current_sps_feedback` evaluates them at a query time and
   packs a 13-field `SPSFeedback`.

The "au" (code/astro) unit system is: mass `Msun`, length `pc`, time `Myr`.

---

## 1. Column mapping — name → index → physical quantity → assumed units

### 1a. How the mapping is established

There are two mapping objects.

**A hardcoded default**, `trinity/sps/sps_columns.py:166-174`:

| canonical | `file_column` | declared units | `log` | physical quantity after conversion |
|---|---|---|---|---|
| `t`           | 0 | `yr`         | linear | time, → Myr (× 1e-6) |
| `Qi`          | 1 | `1/s`        | **log** | ionising photon rate, → 1/Myr (× 1/`s2Myr` = 3.1557e13) |
| `fi`          | 2 | `dimensionless` | **log** | ionising fraction of `Lbol`, dimensionless (× 1) |
| `Lbol`        | 3 | `erg/s`      | **log** | bolometric luminosity, → Msun·pc²/Myr³ (× `L_cgs2au` = 1.65999e-30) |
| `Lmech_total` | 4 | `erg/s`      | **log** | wind **+ SN** mechanical luminosity, → Msun·pc²/Myr³ |
| `pdot_W`      | 5 | `g*cm/s^2`   | **log** | **wind-only** momentum injection rate (a force), → Msun·pc/Myr² (× `pdot_cgs2au` = 1.62312e-25) |
| `Lmech_W`     | 6 | `erg/s`      | **log** | **wind-only** mechanical luminosity, → Msun·pc²/Myr³ |

Note the asymmetry at indices 4–6: index 4 is a **total**, index 5 is **wind-only**, index 6 is
**wind-only**. Columns 4 and 6 are the same physical unit (`erg/s`, log) and adjacent, so an off-by-one
between them is dimensionally undetectable and would flip the sign of `Lmech_SN_raw = Lmech_total −
Lmech_W` (line 201) into "negative → warn → clamp to zero", i.e. **SN feedback silently switched off**
rather than an error.

**`DEFAULT_SPS_COLUMN_MAP` is never referenced anywhere in this slice** (grep over the slice: it is
defined at line 166 and read nowhere). The live path is the user map.

**The live map** is built by `build_user_column_map` (`sps_columns.py:254-275`) from `.param` keys
`sps_col_<canonical>`, each a 3-token string `"<file_column> <units> <log|linear>"` parsed by
`parse_sps_col_value` (`sps_columns.py:213-251`). `file_column` becomes an `int` iff
`file_column_raw.isdigit()` (line 236-237), otherwise it is kept as a **header-name string**.
`read_sps` consumes it via `params['sps_column_map'].value` (`read_sps.py:129`).

Canonicals recognised (13), with the canonical au unit and mass-scaling flag declared at
`sps_columns.py:65-87`:

| canonical | canonical au unit | `mass_scaled` | `required` |
|---|---|---|---|
| `t` | `Myr` | **no** | yes |
| `Qi` | `1/Myr` | yes | yes |
| `fi` | dimensionless | **no** | yes |
| `Lbol` | `Msun*pc^2/Myr^3` | yes | yes |
| `Lmech_W` | `Msun*pc^2/Myr^3` | yes | yes |
| `pdot_W` | `Msun*pc/Myr^2` | yes | yes |
| `Lmech_total` | `Msun*pc^2/Myr^3` | yes | no |
| `Lmech_SN` | `Msun*pc^2/Myr^3` | yes | no |
| `pdot_SN` | `Msun*pc/Myr^2` | yes | no |
| `Mdot_SN` | `Msun/Myr` | yes | no |
| `v_SN` | `pc/Myr` | **no** | no |
| `Li` | `Msun*pc^2/Myr^3` | yes | no |
| `Ln` | `Msun*pc^2/Myr^3` | yes | no |

`validate_user_column_map` (`sps_columns.py:278-311`) requires `{t, Qi, Lbol, Lmech_W, pdot_W}` plus
(`fi` **or** both `Li` and `Ln`) plus (`Lmech_total` **or** `Lmech_SN`). `Li` and `Ln` must be declared
both-or-neither.

### 1b. What validates the mapping against the file's own header — nothing

`_scan_layout` (`sps_columns.py:385-442`) *does* recover a header row into `header_names`, but
`load_user_columns` (`sps_columns.py:472-497`) uses it **only** in the `else` branch, i.e. only when the
spec's `file_column` is a *string*. When `file_column` is an integer, the only check is a bounds check
(`0 <= idx < n_cols`, lines 474-479). **There is no cross-check that `header_names[idx]` has anything to
do with the canonical name.** An integer index that points at the wrong physical column is accepted
silently: the declared unit factor and the declared `log` flag are applied to whatever numbers are
there.

There is also no downstream plausibility check that could catch it — e.g. `velocity_wind = 2·L/ṗ`
(`read_sps.py:215`) has an obvious sanity window (a few hundred to a few thousand km/s) and is never
range-checked. This is finding **S10-A-01**.

### 1c. Header detection is weak in a way that matters

`_scan_layout` finds `data_start` as the first line that is non-blank, does not start with `#`, and all
of whose tokens parse as floats (lines 407-417). It then walks **upward**, `continue`-ing over blank and
`#` lines, and `break`s unconditionally after examining the first line that is neither (lines 431-440).
Consequences:

* A header row written as a **`#` comment** — the SB99/starburst99 convention — is skipped by the
  `continue` and never becomes `header_names`. Header-name column specs then hard-fail with
  "no header row was detected" (lines 482-489). (**S10-A-08**)
* A header row with a **different token count** than the first data row also yields `header_names = []`.
* Delimiter is inferred from the first data row only: `',' if ',' in line else None` (line 411), then
  passed straight to `np.loadtxt` (`delimiter=None` ⇒ whitespace).
* `_can_parse_float` (lines 375-382) catches only `ValueError`, so `nan`/`inf`/`1e5` header tokens would
  make a header row look like data.

`np.loadtxt(filepath, skiprows=data_start, delimiter=delimiter, comments='#', ndmin=2)` (line 464); any
exception is re-raised as `IOError` (line 467), except `FileNotFoundError` which escapes `_scan_layout`
first and is re-raised with a friendlier message in `read_sps.py:160-164`.

---

## 2. Table loading and interpolation — axes

**There is exactly one axis: time.** There is **no metallicity axis and no stellar-mass / IMF axis** in
this module. Metallicity, IMF, and rotation are baked into whichever file `sps_path` names; nothing
interpolates between tables. The cluster-mass dependence is a **single scalar multiplier `f_mass`**
applied to the mass-scaled columns (`read_sps.py:172-173`), i.e. every extensive feedback quantity is
taken to be exactly linear in cluster mass with no re-sampling of the population (**S10-A-18**).

**Linear vs log.** The *file* may store any column in log10 (`log=True`), and
`convert_to_canonical_au` (`sps_columns.py:180-210`) de-logs first and applies the unit factor second:

```
if log: arr = 10.0 ** arr          # line 209
return arr * table[declared_units] # line 210
```

That order is correct (the factor must multiply the linear value, not be added to the exponent).

**After that point everything is linear.** `t` is stored in linear Myr, and every interpolator is built
on **linear `t` against linear `y`**:

```
scipy.interpolate.interp1d(t_Myr, y, kind=ftype)   # read_sps.py:341-354, ftype default 'cubic'
```

Ten splines: `fQi, fLi, fLn, fLbol, fLmech_W, fLmech_SN, fLmech_total, fpdot_W, fpdot_SN, fpdot_total`
(dict at lines 357-368). No `fill_value`, no `bounds_error` override, no `assume_sorted`.

**Query → index.** There is no manual index arithmetic anywhere. A query time is passed to `interp1d`,
which does its own `searchsorted` on `self.x` and evaluates the piecewise not-a-knot cubic. The only
index the code touches directly is `sps_f['fQi'].x[0]` and `.x[-1]` for the range guard
(`update_feedback.py:153-154`).

`t` monotonicity is enforced by `validate_t_monotonic` (`sps_columns.py:334-372`): strictly increasing
required (`diffs <= 0` rejected), with a long diagnostic naming the "`%.2e` collapsed the time column"
failure mode. It is called at `read_sps.py:186` — **after** unit conversion, **before** the `t = 0`
prepend.

---

## 3. Off-grid behaviour

| condition | behaviour |
|---|---|
| `t < t_min` | **raise.** `update_feedback.py:156-159` raises `ValueError("Time t=… outside SPS range …")`. If that guard were bypassed, `interp1d` default `bounds_error` is also True ⇒ `ValueError`. |
| `t > t_max` — **including "the simulation ran past the end of the table"** | **raise.** Same guard. There is **no clamping to the last row, no zeroing, no extrapolation.** A run that outlives the SPS table dies with `ValueError`, it does not silently coast on the final value. |
| `t_min < t < t_max` | **interpolate** — piecewise cubic (`kind='cubic'`, not-a-knot) on linear `y`. Never extrapolation. |
| `t == t_min` or `t == t_max` **exactly** | Passes the guard (it is inclusive, `t_min <= t <= t_max`) — **then crashes anyway** in the derivative, see below. |

**The endpoint crash (S10-A-02).** `update_feedback.py:184-185`:

```
dt = 1e-9
pdotdot_total = (sps_f['fpdot_total'](t + dt) - sps_f['fpdot_total'](t - dt)) / (2.0 * dt)
```

At `t == t_min` the call evaluates `fpdot_total(t_min - 1e-9)`, which is below `interp1d`'s domain with
`bounds_error=True` ⇒ `ValueError: A value in x_new is below the interpolation range`. Symmetrically at
`t == t_max`. And `t_min` is **exactly `0.0`** by construction: `read_sps.py:263-264` guarantees
`t[0] == 0.0`. So a call at simulation time `t = 0.0` — the natural first ODE evaluation — passes the
explicit range check and then raises from the finite difference. The two guards disagree about the
domain: the explicit check uses a closed interval, the derivative needs an open one widened by `1e-9`.

No positivity clamp is applied to any interpolated value.

---

## 4. The feedback quantities, substituted down to table values and `f_mass`

Write `L_W`, `ṗ_W`, `L_tot^tab`, `L_bol`, `Q_i^tab`, `f_i` for the **post-de-log, post-unit-conversion,
post-`f_mass`** column values, and abbreviate the four parameters

* `a ≡ FB_mColdWindFrac`, `θ_W ≡ FB_thermCoeffWind`
* `b ≡ FB_mColdSNFrac`, `θ_S ≡ FB_thermCoeffSN`, `v_SN ≡ FB_vSN` (or the `v_SN` column).

### 4a. Mass scaling — applied exactly once, verified

`read_sps.py:172-173`:

```
if sps_columns.CANONICALS[canonical].mass_scaled:
    arr = arr * f_mass
```

Applied per canonical, exactly once, immediately after unit conversion, before any derivation. Tracing
the exponent of `f_mass` through the wind block:

* `Ṁ_W = ṗ_W²/(2 L_W)` ⇒ `f²/f¹ = f¹` ✔ once
* `v_W = 2 L_W/ṗ_W`     ⇒ `f¹/f¹ = f⁰` ✔ velocity correctly mass-independent
* `ṗ_W^out = Ṁ_W v_W`   ⇒ `f¹` ✔ once
* `L_W^out = ½ Ṁ_W v_W²`⇒ `f¹` ✔ once

and through the SN block: `L_SN^raw = L_tot^tab − L_W` is `f¹`; `v_SN` is `mass_scaled=False`
(`sps_columns.py:81`) whether it comes from the column or from `FB_vSN`; `Ṁ_SN = 2L_SN/v_SN²` is `f¹`.
**I find no double application and no omission.** `Li`/`Ln`, when derived, are `Lbol·f_i` with `Lbol`
scaled and `f_i` unscaled ⇒ `f¹`; when read as columns they are `mass_scaled=True` ⇒ `f¹`. Consistent
either way.

`f_mass` itself is validated only as `finite and > 0` (`read_sps.py:112-115`).

### 4b. Wind — two table reads, everything else derived (`read_sps.py:211-222`)

```
Ṁ_W^raw = ṗ_W² / (2·max(L_W, 1e-100))            # line 214
v_W^raw = 2·L_W / max(ṗ_W, 1e-100)               # line 215
Ṁ_W     = Ṁ_W^raw · (1 + a)                      # line 216
v_W     = v_W^raw · sqrt(θ_W / (1 + a))          # lines 217-220
ṗ_W^out = Ṁ_W · v_W                              # line 221
L_W^out = ½ · Ṁ_W · v_W²                          # line 222
```

Collapsing (the `max()` guards are no-ops for positive input):

* **`Ṁ_W = (1+a)·ṗ_W²/(2 L_W)`** — mass loading multiplies the wind mass flux by `(1+a)`.
* **`v_W = sqrt(θ_W/(1+a))·(2 L_W/ṗ_W)`**
* **`ṗ_W^out = sqrt(θ_W·(1+a)) · ṗ_W`** — mass loading *raises* momentum by `sqrt(1+a)`.
* **`L_W^out = θ_W · L_W`** — exactly. The thermalisation coefficient is a pure multiplier on the wind
  mechanical luminosity, and mass loading has **zero** net effect on it.

Note `v_W^raw` is computed at line 215 from the *unmodified* `L_W`, `ṗ_W`, before line 216 rewrites
`Ṁ_W` — the ordering is correct, not a stale-variable bug.

### 4c. SN — up to four independent table reads with a precedence ladder (`read_sps.py:198-246`)

```
L_SN^raw = cols['Lmech_SN']  if present  else  cols['Lmech_total'] − cols['Lmech_W']   # 198-201
if any(L_SN^raw < 0): warn + np.maximum(L_SN^raw, 0)                                   # 203-208
v_SN^base = cols['v_SN']     if present  else  params['FB_vSN'].value                   # 225-228
Ṁ_SN      = cols['Mdot_SN']  if present  else  2·L_SN^raw / max(v_SN^base², 1e-100)     # 230-233
Ṁ_SN      = Ṁ_SN · (1 + b)                                                              # 234
v_SN^mod  = v_SN^base · sqrt(θ_S / (1 + b))                                             # 236-239
ṗ_SN      = cols['pdot_SN'] if present  else  Ṁ_SN · v_SN^mod                            # 241-244
L_SN^out  = ½ · Ṁ_SN · v_SN^mod²          ← ALWAYS this, never the column               # 246
```

In the default path (`Mdot_SN`, `pdot_SN`, `v_SN` columns all absent), this collapses to

* **`Ṁ_SN = (1+b)·2·(L_tot^tab − L_W)/v_SN²`**
* **`v_SN^mod = sqrt(θ_S/(1+b))·v_SN`**
* **`ṗ_SN = sqrt(θ_S·(1+b))·2·(L_tot^tab − L_W)/v_SN`**
* **`L_SN^out = θ_S·(L_tot^tab − L_W)`** — again an exact pure multiplier, mass loading cancels.

### 4d. Totals and per-time-step outputs

```
L_tot^out = L_SN^out + L_W^out          # read_sps.py:249   (the table's Lmech_total is NOT reused)
ṗ_tot^out = ṗ_SN     + ṗ_W^out          # read_sps.py:250
```

`update_feedback.get_current_sps_feedback` (lines 164-185) evaluates the 10 splines and adds two derived
scalars:

* **`v_mech_total = 2·L_tot(t) / ṗ_tot(t)`** (line 181) — the energy-weighted effective velocity of the
  *combined* wind+SN outflow, **not** the wind velocity and not `v_SN`. **No zero guard** on `ṗ_tot`.
* **`ṗ̈_tot = [ṗ_tot(t+1e-9) − ṗ_tot(t−1e-9)] / 2e-9`** (lines 184-185).

The returned quantities, restated:

| output | expression in table values |
|---|---|
| ionising photon rate `Qi` | spline of `f_mass · 10^{col} · (1/s2Myr)` — the table column, mass-scaled, **not** touched by any feedback parameter |
| bolometric `Lbol` | spline of `f_mass · 10^{col} · L_cgs2au` |
| ionising `Li` | `Lbol · f_i` (line 192) unless a column is declared |
| non-ionising `Ln` | `Lbol · (1 − f_i)` (line 194) unless a column is declared |
| wind mech. lum. `Lmech_W` | `θ_W · L_W` |
| SN mech. lum. `Lmech_SN` | `θ_S · (L_tot^tab − L_W)` (default path) |
| wind momentum rate `pdot_W` | `sqrt(θ_W(1+a)) · ṗ_W` |
| SN momentum rate `pdot_SN` | `sqrt(θ_S(1+b)) · 2(L_tot^tab − L_W)/v_SN` (default path) |
| wind velocity | **not returned**; only recoverable as `2 L/ṗ` |
| mass-loss rates | **not returned at all** — `Mdot_wind` and `Mdot_SN` are local variables, never exported |

---

## 5. Internal consistency — where the four identities are and are not enforced

The exact identities are `L = ½ Ṁ v²`, `ṗ = Ṁ v`, `v = 2L/ṗ`, `Ṁ = ṗ²/(2L)`.

**Wind: fully consistent.** Exactly two independent reads (`Lmech_W`, `pdot_W`); `Ṁ` and `v` are
*derived*, then `ṗ^out` and `L^out` are *rebuilt* from the modified pair. The quadruple satisfies all
four identities exactly at every grid point, by construction. There is no redundant read to disagree
with.

**SN: over-determined, with silent precedence.** Up to four independent inputs
(`Lmech_SN` or `Lmech_total`, `Mdot_SN`, `pdot_SN`, `v_SN`) for a two-degree-of-freedom system.
Resolution:

1. If **`Mdot_SN` is declared**, `L_SN^raw` (built from `Lmech_SN`/`Lmech_total` at lines 198-208) is
   computed, possibly warned about, clamped — **and then never read again**. Line 233 is its only
   consumer and it is inside the `else`. So the `Lmech_SN`/`Lmech_total` column is fully discarded, yet
   `validate_user_column_map` (line 301) still *demands* one of them. The exported `Lmech_SN` becomes
   `½·(1+b)·Ṁ_SN^col·v_SN²·θ_S/(1+b) = ½·θ_S·Ṁ_SN^col·v_SN²`, which need not equal `θ_S·L_SN^raw`.
   (**S10-A-04**)
2. If **`pdot_SN` is declared**, it is taken **raw** (only `f_mass`-scaled) and is **not** multiplied by
   `sqrt(θ_S(1+b))`, while `Ṁ_SN` *is* multiplied by `(1+b)` and `L_SN^out` *is* multiplied by `θ_S`.
   The exported triple then violates `ṗ = Ṁ v`: implied `v = ṗ_SN^col/[(1+b)Ṁ_SN] ≠ v_SN^mod`, and the
   downstream `v_mech_total = 2L/ṗ` inherits the mismatch. (**S10-A-03**)
3. `L_SN^out` is **never** the `Lmech_SN` column value — line 246 always recomputes.
4. `Lmech_total` is likewise always recomputed as the sum at line 249; the table's total column
   survives only as an input to the difference at line 201.

**Interpolation and identities.** `interp1d` with fixed knots is a *linear* operator on the `y` data, so
the **additive** identities survive off-grid to floating-point:
`fLmech_total(t) = fLmech_W(t) + fLmech_SN(t)`, `fpdot_total(t) = fpdot_W(t) + fpdot_SN(t)`,
`fLi(t) + fLn(t) = fLbol(t)`. The **multiplicative/ratio** identities do not: `v_mech_total(t) =
2·fLmech_total(t)/fpdot_total(t)` equals the true grid-point velocity only *at* knots; between knots the
two splines are independent functions and the ratio drifts. Same for any `Ṁ` a caller reconstructs as
`ṗ²/(2L)`.

---

## 6. Dimensions

**Every conversion factor, checked against `unit_conversions.py`.** With `s2Myr = 3.168808781e-14`,
`cm2pc = 3.240779289e-19`, `g2Msun = 5.029144216e-34`, `L_cgs2au = 1.65998782e-30`,
`pdot_cgs2au = 1.62312317e-25`, `v_kms2au = 1.022712165`:

| canonical / declared unit | factor in code | independent check | verdict |
|---|---|---|---|
| `t` / `yr` | `1.0e-6` | yr→Myr | ✔ |
| `t` / `s`, `cgs` | `s2Myr` | s→Myr | ✔ |
| `Qi` / `1/s`, `cgs` | `1/s2Myr` = 3.1557e13 | (1/s)·(s per Myr) | ✔ |
| `L*` / `erg/s`, `cgs` | `L_cgs2au` | `g2Msun·cm2pc²/s2Myr³` = 1.660e-30 | ✔ |
| `L*` / `L_sun` | `3.828e33 · L_cgs2au` | IAU nominal `L_⊙` in erg/s | ✔ |
| `pdot_*` / `g*cm/s^2`, `cgs` | `pdot_cgs2au` | `g2Msun·cm2pc/s2Myr²` = 1.6231e-25 | ✔ |
| `Mdot_SN` / `g/s`, `cgs` | `g2Msun/s2Myr` = 1.587e-20 | (g/s)→(Msun/Myr) | ✔ |
| `v_SN` / `cm/s`, `cgs` | `cm2pc/s2Myr` = 1.0227e-5 | equals `CONV.v_cms2au` exactly | ✔ |
| `v_SN` / `km/s` | `v_kms2au` = 1.02271 | | ✔ |
| `fi` / dimensionless, `cgs` | `1.0` | | ✔ |

**Arithmetic dimensions**, in au (`Msun`, `pc`, `Myr`):

| expression | dimensions | balances |
|---|---|---|
| `ṗ²/(2L)` line 214 | `(M·L/T²)²/(M·L²/T³) = M/T` | ✔ `Msun/Myr` |
| `2L/ṗ` line 215 | `(M·L²/T³)/(M·L/T²) = L/T` | ✔ `pc/Myr` |
| `Ṁ·v` line 221, 244 | `M·L/T²` | ✔ force |
| `½Ṁv²` lines 222, 246 | `M·L²/T³` | ✔ power |
| `2·L_SN/v²` line 233 | `M/T` | ✔ |
| `sqrt(θ/(1+f))` lines 217-220, 236-239 | dimensionless | ✔ (requires `FB_thermCoeff*`, `FB_mColdFrac` be pure numbers) |
| `2L_tot/ṗ_tot` update_feedback:181 | `L/T` | ✔ `pc/Myr` |
| `Δṗ/(2Δt)` update_feedback:185 | `M·L/T³` | ✔ `Msun·pc/Myr³` — consistent with `cvt.pdotdot_cgs2au` existing |

**One asymmetry in the unit tables.** `t`, `Qi`, `Mdot_SN`, `v_SN` each offer a pass-through au option
(`'Myr'`, `'1/Myr'`, `'Msun/Myr'`, `'pc/Myr'` → factor 1.0). `Lbol`, `Lmech_*`, `Li`, `Ln`, `pdot_*` do
**not** — only `erg/s`/`L_sun`/`cgs` and `g*cm/s^2`/`cgs`. A table already written in au units cannot
declare its luminosity or momentum columns (**S10-A-11**).

**One unconverted parameter.** `params['FB_vSN'].value` is consumed raw at `read_sps.py:228` with no
`UNIT_CONVERSIONS` lookup, while the `v_SN` *column* path goes through the full conversion. The two
sources of the same quantity therefore have different unit contracts inside this module
(**S10-A-12**); whether the param layer converts it is outside the slice.

---

## 7. Control flow, branches, clamps, handlers, caching, and every bare constant

### Branches (all of them)

| location | condition | else-path |
|---|---|---|
| `read_sps.py:112` | `not isfinite(f_mass) or f_mass <= 0` → `ValueError` | proceed |
| `read_sps.py:123-125` | any of 7 `required_keys` missing → `KeyError` | proceed |
| `read_sps.py:160` | `FileNotFoundError` → re-raise with hint | — |
| `read_sps.py:172` | `mass_scaled` → `× f_mass` | leave alone |
| `read_sps.py:174` | any non-finite → `ValueError` | proceed |
| `read_sps.py:191`, `:193` | `Li`/`Ln` absent → derive from `Lbol`, `fi` | keep column |
| `read_sps.py:198` | `Lmech_SN` present → use it | `Lmech_total − Lmech_W` |
| `read_sps.py:203` | any `L_SN^raw < 0` → **warn + clamp to 0** | proceed |
| `read_sps.py:225` | `v_SN` column present | `FB_vSN` param |
| `read_sps.py:230` | `Mdot_SN` column present (copied) | derive from `L_SN^raw` |
| `read_sps.py:241` | `pdot_SN` column present (used raw) | `Ṁ_SN · v_SN^mod` |
| `read_sps.py:263` | `len(t) == 0 or t[0] != 0.0` → prepend 0 with constant `y[0]` | proceed |
| `update_feedback.py:156` | `t` outside `[t_min, t_max]` → `ValueError` | proceed |
| `sps_columns.py:197`, `:203` | unknown canonical → `KeyError`; unknown unit → `ValueError` | |
| `sps_columns.py:208` | `log` → `10**arr` | |
| `sps_columns.py:227`, `:239`, `:244` | 3-token check; `log|linear` check; unit-name check | |
| `sps_columns.py:267`, `:272` | key absent, or `'def_unset'`/`None` → skip canonical | |
| `sps_columns.py:303` | any of 4 validation failures → `ValueError` with template | |
| `sps_columns.py:350`, `:353` | `len(t) < 2` → return; no `diffs <= 0` → return | else raise |
| `sps_columns.py:414` | first all-float row → `data_start` | else `ValueError` at 419 |
| `sps_columns.py:437` | token-count match **and** ≥1 non-float → header | else `header_names = []` |
| `sps_columns.py:473` | int index (bounds-checked) | header-name lookup (2 error paths) |

### Clamps

* `np.maximum(Lmech_wind_raw, 1e-100)` (214), `np.maximum(pdot_wind_raw, 1e-100)` (215),
  `np.maximum(velocity_SN_base**2, 1e-100)` (233) — `EPSILON = 1e-100`, `read_sps.py:35`.
  For log-declared columns (the default) `10**x > 0` always, so **all three are unreachable**; they can
  fire only for linear-declared columns containing exact 0, a value that underflows `10**x` to 0, or a
  **negative** value. When they do fire they yield a ~1e100-scale finite number that passes every
  downstream `isfinite` check (**S10-A-06**). Note nothing validates positivity of `Lmech_W`, `pdot_W`,
  `Lbol`, or `Qi` — only `Lmech_SN_raw` is sign-checked.
* `np.maximum(Lmech_SN_raw, 0)` (208) — the only clamp with a log message.
* No clamp anywhere on interpolated output; no clamp at `t_max`.

### Exception handlers

* `read_sps.py:158-164` — `FileNotFoundError` only. `ValueError` from a bad column index, `IOError` from
  `np.loadtxt`, and `KeyError` from a missing canonical all propagate.
* `sps_columns.py:378-382` — `_can_parse_float`, catches `ValueError` only.
* `sps_columns.py:466-467` — bare `except Exception` around `np.loadtxt`, re-raised as `IOError` with
  `from e`.

### Caching / state

* `params['sps_f'].value` (`update_feedback.py:151`) is the only cache — the 10 splines, built once from
  a specific `f_mass`. Nothing in this slice recomputes or invalidates it. If the cluster mass changes
  mid-run, every feedback quantity is silently the old mass's (**S10-A-16**). `t_min`/`t_max` are
  re-derived from `fQi.x` on every call, so they cannot go stale relative to the splines.
* `SPSFeedback` is a plain (mutable, non-frozen) dataclass rebuilt per call — no staleness.

### Every bare numeric constant in arithmetic

| value | location | expression it sits in |
|---|---|---|
| `3.828e33` | `sps_columns.py:33` | `L_sun` → erg/s, used in all 6 luminosity `L_sun` factors |
| `1.0e-6` | `sps_columns.py:116` | yr → Myr |
| `1.0` | `sps_columns.py:117,123,127-128,140,146` | pass-through au factors |
| `10.0` | `sps_columns.py:209` | `10.0 ** arr` de-log |
| `3` | `sps_columns.py:227` | token count of `sps_col_*` |
| `2` | `sps_columns.py:350` | `len(t) < 2` early return |
| `1`, `3`, `5` | `sps_columns.py:358-360` | error-message slicing only |
| `1e-100` | `read_sps.py:35` | `EPSILON`, three `np.maximum` denominators |
| `2` | `read_sps.py:214,215,233` | the `½` in `L = ½Ṁv²`, inverted |
| `1`, `1.0` | `read_sps.py:216,219,234,238,194` | `(1 + mColdFrac)`, `(1 − fi)` |
| `0.5` | `read_sps.py:222,246` | `½Ṁv²` |
| `2` (exponent) | `read_sps.py:214,222,233,246` | squares |
| `0` | `read_sps.py:208` | SN luminosity floor |
| `0.0` | `read_sps.py:263,264` | `t[0] != 0.0` test and the inserted time |
| `13` | `update_feedback.py:95` | hardcoded `__len__` |
| `2.` | `update_feedback.py:181` | `v = 2L/ṗ` |
| `1e-9` | `update_feedback.py:184` | central-difference half-step, in **Myr** |
| `2.0` | `update_feedback.py:185` | central-difference denominator |

### Dead / unreachable code found

* `read_sps.py:263` `len(t) == 0` — if it were ever true, `np.insert(Qi, 0, Qi[0])` on line 265 would
  raise `IndexError` on the empty array. It cannot be true: `_scan_layout` raises unless at least one
  numeric row exists. Broken-if-reached (**S10-A-09**).
* `sps_columns.py:244` — the `canonical in UNIT_CONVERSIONS` half of the `and` is always True on the
  `build_user_column_map` path (it iterates `CANONICAL_NAMES`, and `CANONICALS.keys()` and
  `UNIT_CONVERSIONS.keys()` are the same 13-element set).
* `sps_columns.py:166-174` `DEFAULT_SPS_COLUMN_MAP` — unreferenced in the slice.
* Unused imports: `sys` (`read_sps.py:25`), `cvt` (`read_sps.py:28` — no `cvt.` appears in the file),
  `updateDict` (`update_feedback.py:13`).
* `sps_columns.py:56` `CanonicalSpec.canonical_au_unit` is set for all 13 entries but never read — it is
  documentation-as-data, not used in any conversion.

### Other observations worth recording

* `import scipy` (`read_sps.py:24`) then `scipy.interpolate.interp1d(...)` — `scipy.interpolate` is only
  reachable this way through SciPy's lazy-subpackage `__getattr__`, added in SciPy 1.9. On older SciPy,
  or in an import order where nothing has imported `scipy.interpolate`, this is `AttributeError`.
* `interp1d(..., kind='cubic')` needs ≥ 4 points; a 2- or 3-row SPS table raises from SciPy.
* `sps_f['fQi'](t)[()]` — for an *array* `t`, `arr[()]` returns the array unchanged, so `SPSFeedback`
  silently accepts array-valued fields; nothing enforces scalar.
* `validate_t_monotonic` runs at `read_sps.py:186`, **before** the prepend at 263. A table whose first
  `t` is negative passes validation, then gets `0.0` prepended in front of it; `interp1d`'s default
  `assume_sorted=False` sorts `x` and `y` together, so this is silently reordered rather than caught
  (**S10-A-10**). A first `t` that is tiny-but-nonzero (say `1e-30`) creates a near-duplicate knot.
* The `t = 0` prepend is a **constant (zeroth-order) extension**: `y_new[0] = y[0]`. Followed by a cubic
  spline, the flat-then-steep junction at the first interval is exactly the configuration that produces
  spline ringing.
* `SPSFeedback.__iter__` (82-87) and `__len__` (95, literal `13`) are hand-maintained duplicates of the
  13-field declaration (66-78); `__getitem__` (91) rebuilds `list(self)` on every index access. The
  three are currently consistent — field count is 13 and the iteration order matches `read_sps`'s
  11-element return order for the first 11 entries — but nothing enforces it (**S10-A-13**).
* `_read_sps_user` unconditionally indexes `cols['t']` (186), `cols['Lbol']`/`cols['fi']` (192, 194),
  `cols['Lmech_W']` (201, 211), `cols['pdot_W']` (212). `read_sps` never calls
  `validate_user_column_map`, so if the param layer skips it these are bare `KeyError`s
  (**S10-A-15**).

---

```json
[
  {
    "id": "S10-A-01",
    "file": "trinity/sps/sps_columns.py",
    "line": 473,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "When a column spec uses an integer index, nothing validates it against the file's own header row — only an array-bounds check. A wrong index silently substitutes a different physical quantity.",
    "evidence": "load_user_columns:473-480 takes the int branch and only checks `0 <= spec.file_column < n_cols` before `data[:, spec.file_column]`. `header_names`, recovered by _scan_layout:427-440, is consulted ONLY in the else (string) branch at 490-497. The declared unit factor and `log` flag from sps_col_* are then applied to whatever numbers are at that index. No downstream sanity check exists: read_sps.py:215 computes velocity_wind = 2L/pdot, which has an obvious plausibility window (a few 100 to a few 1000 km/s), and it is never range-checked.",
    "expected": "When header_names is non-empty, cross-check header_names[idx] against the canonical (or an alias table) and raise/warn on mismatch; and/or add a plausibility assertion on the derived wind velocity.",
    "failure_scenario": "DEFAULT_SPS_COLUMN_MAP:171-173 assigns index 4=Lmech_total, 5=pdot_W, 6=Lmech_W. Columns 4 and 6 are both 'erg/s', log, adjacent. Swapping them makes Lmech_SN_raw = Lmech_W - Lmech_total < 0 everywhere, which read_sps.py:203-208 turns into a single WARNING and a clamp to zero — SN feedback silently disabled for the whole run, no exception. Swapping 5 and 6 instead applies pdot_cgs2au to a luminosity, giving a wind velocity wrong by ~1e5 with no error.",
    "repro": "Point sps_col_Lmech_W and sps_col_Lmech_total at each other's indices in a .param, run `python run.py`, observe only the 'Negative SN mechanical luminosity detected; clamping to zero' warning and a completed run.",
    "confidence": "high"
  },
  {
    "id": "S10-A-02",
    "file": "trinity/sps/update_feedback.py",
    "line": 185,
    "class": "numerical",
    "severity": "S2",
    "claim": "The pdotdot_total central difference evaluates the interpolator at t±1e-9, which is outside the spline domain whenever t equals t_min or t_max — exactly the values the range guard three lines earlier admits. t_min is always exactly 0.0.",
    "evidence": "update_feedback.py:156 guards with the CLOSED interval `if not (t_min <= t <= t_max): raise`. Lines 184-185 then call `sps_f['fpdot_total'](t + dt)` and `(t - dt)` with dt=1e-9. The interpolators are built at read_sps.py:341-354 with no fill_value and no bounds_error override, so scipy's default bounds_error=True raises ValueError outside [x[0], x[-1]]. read_sps.py:263-264 guarantees t[0] == 0.0 (either the table already starts at 0.0 or 0.0 is inserted), so t_min == 0.0 exactly.",
    "expected": "Either use a one-sided difference at the endpoints, clamp the stencil into [t_min, t_max], or use the spline's analytic derivative (scipy.interpolate.CubicSpline(...).derivative()), which needs no stencil at all.",
    "failure_scenario": "The first ODE evaluation of a run at t = 0.0 Myr passes the range check on line 156 and then dies on line 185 with 'ValueError: A value in x_new is below the interpolation range.' Symmetrically, a run that reaches exactly t_max dies there.",
    "repro": "sps_f = get_interpolation(read_sps(1.0, params)); get_current_sps_feedback(float(sps_f['fQi'].x[0]), params)",
    "confidence": "high"
  },
  {
    "id": "S10-A-03",
    "file": "trinity/sps/read_sps.py",
    "line": 242,
    "class": "divergence",
    "severity": "S3",
    "claim": "A user-supplied pdot_SN column is exported raw, bypassing FB_thermCoeffSN and FB_mColdSNFrac, while Mdot_SN and Lmech_SN in the same block ARE modified by them. The exported (L, Mdot, pdot, v) quadruple then violates pdot = Mdot*v.",
    "evidence": "read_sps.py:241-242 `if 'pdot_SN' in cols: pdot_SN = cols['pdot_SN']` — used as-is (only f_mass-scaled at line 173). But line 234 applies (1+FB_mColdSNFrac) to Mdot_SN, lines 236-239 apply sqrt(FB_thermCoeffSN/(1+FB_mColdSNFrac)) to the velocity, and line 246 computes Lmech_SN_final = 0.5*Mdot_SN*velocity_SN_modified**2. In the derived branch (line 244) pdot_SN = Mdot_SN*velocity_SN_modified = sqrt(theta_SN*(1+b)) * pdot_raw; in the column branch the sqrt(theta_SN*(1+b)) factor is simply absent.",
    "expected": "Either apply the same sqrt(FB_thermCoeffSN*(1+FB_mColdSNFrac)) factor to the supplied column, or reject the combination, or document that pdot_SN is a full override that also disables the coefficients.",
    "failure_scenario": "With FB_thermCoeffSN = 0.3 and a declared sps_col_pdot_SN, the SN momentum injected is larger than the energy budget implies by 1/sqrt(0.3*(1+b)) ~ 1.8x, and v_mech_total = 2L/pdot (update_feedback.py:181) reports a velocity inconsistent with both Lmech_SN and Mdot_SN.",
    "repro": "Declare sps_col_pdot_SN with FB_thermCoeffSN != 1, and compare pdot_SN against Mdot_SN*velocity_SN_modified.",
    "confidence": "high"
  },
  {
    "id": "S10-A-04",
    "file": "trinity/sps/read_sps.py",
    "line": 233,
    "class": "deadcode",
    "severity": "S3",
    "claim": "If a Mdot_SN column is declared, Lmech_SN_raw — built from the Lmech_SN or Lmech_total column at lines 198-208 — becomes completely unused, yet validate_user_column_map still requires one of those columns to be declared.",
    "evidence": "Lmech_SN_raw is assigned at read_sps.py:199/201 and clamped at 208. Its ONLY consumer is line 233, inside the `else` of `if 'Mdot_SN' in cols:` (line 230). Meanwhile sps_columns.py:299-301 sets sn_input_ok = have_Lmech_total or have_Lmech_SN and line 303 raises if it is False. Lmech_SN_final at line 246 is ALWAYS recomputed as 0.5*Mdot_SN*v_mod**2 and never takes the column value.",
    "expected": "Either derive Lmech_SN from the declared Lmech column when Mdot_SN is also given (and cross-check the two), or relax the validator so Lmech_total/Lmech_SN is not demanded when Mdot_SN is supplied.",
    "failure_scenario": "A user supplies an accurate Mdot_SN table plus a required-by-the-validator Lmech_total column. The Lmech_total values are read, unit-converted, mass-scaled, differenced, possibly warned about and clamped — and then discarded. Exported Lmech_SN = 0.5*theta_SN*Mdot_SN_col*v_SN^2, which can differ from the table's own SN luminosity by any amount, with no message.",
    "repro": "Declare both sps_col_Mdot_SN and sps_col_Lmech_SN with mutually inconsistent values; observe the returned Lmech_SN depends only on Mdot_SN.",
    "confidence": "high"
  },
  {
    "id": "S10-A-05",
    "file": "trinity/sps/update_feedback.py",
    "line": 181,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "v_mech_total = 2*Lmech_total/pdot_total divides with no zero or epsilon guard, unlike the structurally identical divisions in read_sps.py which are all wrapped in np.maximum(..., EPSILON).",
    "evidence": "update_feedback.py:181 `v_mech_total = (2. * Lmech_total / pdot_total)[()]`. Compare read_sps.py:215 `2 * Lmech_wind_raw / np.maximum(pdot_wind_raw, EPSILON)` and :233 `2 * Lmech_SN_raw / np.maximum(velocity_SN_base ** 2, EPSILON)`. pdot_total is a cubic spline of pdot_W + pdot_SN; both cubic interpolation undershoot and a genuinely zero-momentum epoch can drive it to 0 or negative.",
    "expected": "The same EPSILON guard used in read_sps.py, or an explicit check that raises rather than returning inf/nan.",
    "failure_scenario": "pdot_total interpolates to 0 at some t; v_mech_total becomes inf (or nan if Lmech_total is also 0) and is written into SPSFeedback and propagated into the force budget with no warning, poisoning the ODE state.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S10-A-06",
    "file": "trinity/sps/read_sps.py",
    "line": 214,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The EPSILON=1e-100 denominator guards turn a zero or negative table entry into a ~1e100-magnitude finite number that passes every downstream isfinite check, instead of failing. For log-declared columns the guards are unreachable, so they only ever fire on the case they handle worst.",
    "evidence": "read_sps.py:35 EPSILON = 1e-100. Line 214 Mdot_wind = pdot**2/(2*np.maximum(Lmech_wind_raw, EPSILON)); line 215 velocity_wind = 2*Lmech_wind_raw/np.maximum(pdot_wind_raw, EPSILON); line 233 same pattern for v_SN**2. Line 174 only checks np.isfinite, which 1e100-scale values pass. Positivity is checked for exactly one quantity, Lmech_SN_raw (line 203) — never for Lmech_W, pdot_W, Lbol or Qi. When log=True, convert_to_canonical_au:209 gives 10**x > 0 so np.maximum is a no-op; the guards can only fire for linear-declared columns holding 0, an underflowed value, or a negative.",
    "expected": "Validate that Lmech_W, pdot_W, Lbol and Qi are strictly positive after conversion and raise with the offending row index, rather than dividing by 1e-100.",
    "failure_scenario": "A linear-declared pdot_W column with a 0.0 (or a stray negative) at t=0 gives velocity_wind = 2L/1e-100 ~ 1e130 pc/Myr and pdot_wind ~ 1e130, all finite, all silently interpolated and fed to the shell momentum equation.",
    "repro": "Declare sps_col_pdot_W ... linear on a file whose first row has 0 in that column.",
    "confidence": "medium"
  },
  {
    "id": "S10-A-07",
    "file": "trinity/sps/read_sps.py",
    "line": 341,
    "class": "numerical",
    "severity": "S3",
    "claim": "All ten interpolators are cubic splines on LINEAR y over quantities that span many orders of magnitude in time, with no positivity clamp on the output, and the t=0 prepend deliberately creates a flat-then-steep first interval that is the classic spline-ringing configuration.",
    "evidence": "read_sps.py:341-354 build interp1d(t_Myr, y, kind='cubic') on linear arrays; the log10 storage is undone at sps_columns.py:209 before interpolation, so no quantity is interpolated in log space. read_sps.py:263-274 prepend t=0.0 with y_new[0] = y[0] (a zeroth-order constant extension) whenever the table does not already start at 0.0. Nothing anywhere clamps an interpolated value to be >= 0. Qi and Lmech drop by several decades after the first few Myr, so a not-a-knot cubic through those knots can undershoot below zero between them.",
    "expected": "Interpolate in log-y (or use a shape-preserving kind such as 'pchip'), and/or assert positivity of the interpolated Qi/Lbol/Lmech/pdot.",
    "failure_scenario": "Between two widely-separated knots on the post-SN decline, fQi(t) or fLmech_W(t) returns a negative value; downstream a negative ionising rate or negative luminosity propagates into the shell/bubble solve with no diagnostic.",
    "repro": "Evaluate fQi on a dense grid over the steepest decade of the table and check for min < 0.",
    "confidence": "medium"
  },
  {
    "id": "S10-A-08",
    "file": "trinity/sps/sps_columns.py",
    "line": 431,
    "class": "regime",
    "severity": "S4",
    "claim": "Header-row detection skips '#'-prefixed lines, so the standard SB99 convention of a commented header is never recognised — making header-name column specs unusable on exactly the files this module targets.",
    "evidence": "_scan_layout:431-440 walks upward from data_start, `continue`s on blank and '#'-starting lines (line 433), examines only the first line that is neither, and `break`s unconditionally at line 440. A header written as '# time Qi fi Lbol ...' is therefore skipped past. header_names stays [] and load_user_columns:482-489 raises 'no header row was detected'. The token-count equality test at 437 also rejects any header row whose column count differs from the data (e.g. a leading '#' token, or a separate units row).",
    "expected": "Strip a leading '#' from candidate header lines before the token-count and non-numeric tests, or document that only uncommented headers are supported.",
    "failure_scenario": "A user writes sps_col_Qi = 'Q_H  1/s  log' against a normal SB99 file with a '#'-commented header and gets a hard error telling them no header exists, when it visibly does.",
    "repro": "Run load_user_columns on a file whose only header line begins with '#', with any string-valued file_column.",
    "confidence": "high"
  },
  {
    "id": "S10-A-09",
    "file": "trinity/sps/read_sps.py",
    "line": 263,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The `len(t) == 0` branch cannot work: the body immediately indexes Qi[0] on the same empty-length arrays and would raise IndexError.",
    "evidence": "read_sps.py:263 `if len(t) == 0 or t[0] != 0.0:` then line 265 `Qi = np.insert(Qi, 0, Qi[0])`. With len(t)==0 all the sibling arrays are empty too, so Qi[0] is an IndexError. The branch is also unreachable in practice: _scan_layout raises ValueError at sps_columns.py:419-423 unless at least one all-numeric data row exists, so len(t) >= 1 always.",
    "expected": "Drop the `len(t) == 0` disjunct (unreachable), or handle the empty case with an explicit error message.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-A-10",
    "file": "trinity/sps/read_sps.py",
    "line": 186,
    "class": "numerical",
    "severity": "S4",
    "claim": "validate_t_monotonic runs before the t=0.0 prepend, so the prepend can undo the property that was just validated; and it never checks that t[0] >= 0.",
    "evidence": "read_sps.py:186 calls sps_columns.validate_t_monotonic(cols['t'], filepath); the prepend happens 77 lines later at 263-274. validate_t_monotonic:352-354 only rejects diffs <= 0 in the ORIGINAL array. If the table's first time is negative, the check passes, 0.0 is inserted in front of it, and the array becomes non-monotonic. scipy's interp1d defaults to assume_sorted=False and silently sorts x and y together, so nothing is raised. A first time that is tiny but nonzero (e.g. 1e-30 Myr) instead creates a near-duplicate knot and an ill-conditioned first spline interval.",
    "expected": "Re-validate after the prepend, or reject t[0] < 0 up front, or use `t[0] > 0.0` as the prepend condition.",
    "failure_scenario": "A table with a negative first time (or an offset time origin) is silently reordered by scipy rather than rejected, moving the artificial constant-extension row into the interior of the grid.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S10-A-11",
    "file": "trinity/sps/sps_columns.py",
    "line": 130,
    "class": "units",
    "severity": "S4",
    "claim": "The luminosity and momentum-rate canonicals offer no au pass-through unit, while t, Qi, Mdot_SN and v_SN all do. A table already written in code units cannot declare those columns.",
    "evidence": "UNIT_CONVERSIONS entries for Lbol/Lmech_W/Lmech_total/Lmech_SN/Li/Ln (lines 130-135) accept only 'erg/s', 'L_sun', 'cgs'; pdot_W/pdot_SN (136-137) only 'g*cm/s^2', 'cgs'. By contrast 't' has 'Myr': 1.0 (117), 'Qi' has '1/Myr': 1.0 (123), 'Mdot_SN' has 'Msun/Myr': 1.0 (140), 'v_SN' has 'pc/Myr': 1.0 (146). parse_sps_col_value:244-249 rejects any unit not in the table.",
    "expected": "Add 'Msun*pc^2/Myr^3': 1.0 to the six luminosity entries and 'Msun*pc/Myr^2': 1.0 to the two pdot entries, matching the canonical_au_unit strings already declared at lines 69-86.",
    "failure_scenario": "",
    "repro": "parse_sps_col_value('Lbol', '3 Msun*pc^2/Myr^3 linear') raises ValueError.",
    "confidence": "high"
  },
  {
    "id": "S10-A-12",
    "file": "trinity/sps/read_sps.py",
    "line": 228,
    "class": "units",
    "severity": "S4",
    "claim": "FB_vSN is consumed with no unit conversion, while the v_SN column path for the same physical quantity goes through the full UNIT_CONVERSIONS machinery. The two sources have different unit contracts inside this module.",
    "evidence": "read_sps.py:225-228: `if 'v_SN' in cols: velocity_SN_base = cols['v_SN']` (already converted by convert_to_canonical_au via UNIT_CONVERSIONS['v_SN'], sps_columns.py:143-148) `else: velocity_SN_base = params['FB_vSN'].value` — used raw. The value then enters velocity_SN_base**2 at line 233 and velocity_SN_modified at 236, both of which require pc/Myr for the arithmetic to balance (Mdot_SN must come out in Msun/Myr).",
    "expected": "Convert FB_vSN through the same UNIT_CONVERSIONS['v_SN'] table, or assert/document that the param layer delivers it in pc/Myr.",
    "failure_scenario": "If FB_vSN is specified in km/s and the param layer does not convert, Mdot_SN is wrong by v_kms2au**2 = 1.046 and the SN momentum by ~1.02 — small enough to look plausible. If specified in cm/s the error is ~1e10.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S10-A-13",
    "file": "trinity/sps/update_feedback.py",
    "line": 95,
    "class": "state",
    "severity": "S4",
    "claim": "SPSFeedback.__len__ hardcodes the literal 13 and __iter__ hardcodes the field list; both are hand-maintained duplicates of the 13-field dataclass declaration and will drift silently if a field is added or reordered.",
    "evidence": "Fields declared at update_feedback.py:66-78 (13 of them). __iter__ at 82-87 re-lists all 13 by name in a specific order. __len__ at 93-95 `return 13`. __getitem__ at 89-91 rebuilds `list(self)` on every single index access (O(n) per element, O(n^2) for a full positional unpack). All three are currently consistent with each other and with read_sps.py:281-282's 11-element return order, but nothing enforces it.",
    "expected": "Derive all three from dataclasses.fields(self) / dataclasses.astuple(self), so len and iteration order cannot desynchronise from the declaration.",
    "failure_scenario": "A 14th field is added; __len__ still returns 13 and __iter__ omits it, so any caller doing positional unpacking silently drops the new quantity.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-A-14",
    "file": "trinity/sps/read_sps.py",
    "line": 25,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Unused imports and an unreferenced module-level default map.",
    "evidence": "read_sps.py:25 `import sys` — no `sys.` in the file. read_sps.py:28 `import trinity._functions.unit_conversions as cvt` — no `cvt.` in the file (grep over the slice shows cvt. only in sps_columns.py). update_feedback.py:13 `from trinity._input.dictionary import updateDict` — updateDict is never called. sps_columns.py:166-174 DEFAULT_SPS_COLUMN_MAP is never referenced within the slice; the live path is params['sps_column_map'].value (read_sps.py:129). sps_columns.py:56 CanonicalSpec.canonical_au_unit is populated for all 13 canonicals and never read.",
    "expected": "Remove the three dead imports. Confirm whether DEFAULT_SPS_COLUMN_MAP is imported outside the slice before touching it.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-A-15",
    "file": "trinity/sps/read_sps.py",
    "line": 192,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "read_sps never calls validate_user_column_map, yet _read_sps_user unconditionally indexes six canonicals out of cols. If the param layer skips validation the user gets a bare KeyError instead of the carefully written diagnostic that already exists.",
    "evidence": "read_sps.py:117-125 checks only that seven param KEYS exist; it never calls sps_columns.validate_user_column_map (defined at sps_columns.py:278 with the full _format_missing_template message at 314-331). _read_sps_user then does cols['t'] (186), cols['Lbol'] and cols['fi'] (192, 194), cols['Lmech_total'] and cols['Lmech_W'] (201, 211), cols['pdot_W'] (212) with no guard.",
    "expected": "Call validate_user_column_map(column_map, filepath) at the top of _read_sps_user, or assert it was already run.",
    "failure_scenario": "A .param that declares sps_col_t/Qi/Lbol/Lmech_W/pdot_W but forgets sps_col_fi and sps_col_Lmech_total dies with `KeyError: 'fi'` at line 192 rather than the actionable 'missing sps_col_* for [...]' message.",
    "repro": "Call read_sps with a column_map lacking 'fi'.",
    "confidence": "medium"
  },
  {
    "id": "S10-A-16",
    "file": "trinity/sps/update_feedback.py",
    "line": 151,
    "class": "state",
    "severity": "S4",
    "claim": "params['sps_f'] caches ten interpolators baked at one specific f_mass; nothing in this module invalidates or re-derives it, so a change in cluster mass would be silently ignored.",
    "evidence": "read_sps.py:172-173 multiplies every mass_scaled column by f_mass at load time, so f_mass is frozen into the spline y-values. get_current_sps_feedback:151 reads params['sps_f'].value with no key, hash, or f_mass check; t_min/t_max are re-derived per call (153-154) but the mass normalisation is not.",
    "expected": "Store f_mass alongside sps_f and assert it matches the current cluster mass, or rebuild on change.",
    "failure_scenario": "A run where the cluster mass changes (recollapse / second SF event) or an in-process sweep that reuses a params object across configurations would keep using the previous mass's feedback with no warning. Whether either occurs is outside this slice.",
    "repro": "",
    "confidence": "low"
  },
  {
    "id": "S10-A-17",
    "file": "trinity/sps/sps_columns.py",
    "line": 236,
    "class": "other",
    "severity": "S4",
    "claim": "file_column parsing uses str.isdigit(), so '-1' and '+1' are not recognised as integers and fall through to the header-name branch, producing a confusing error.",
    "evidence": "sps_columns.py:236-237 `file_column = (int(file_column_raw) if file_column_raw.isdigit() else file_column_raw)`. '-1'.isdigit() is False, so it becomes the string '-1' and load_user_columns:490-495 reports it as a missing header name. '007'.isdigit() is True and becomes 7. Relatedly, the `canonical in UNIT_CONVERSIONS` half of the guard at line 244 is always True on the build_user_column_map path, since that function iterates CANONICAL_NAMES and CANONICALS.keys() equals UNIT_CONVERSIONS.keys() (both the same 13 names).",
    "expected": "Either accept negative indices explicitly or reject them with a message naming the actual problem.",
    "failure_scenario": "",
    "repro": "parse_sps_col_value('Qi', '-1 1/s log') yields ColumnSpec(file_column='-1', ...).",
    "confidence": "medium"
  },
  {
    "id": "S10-A-18",
    "file": "trinity/sps/read_sps.py",
    "line": 173,
    "class": "regime",
    "severity": "S4",
    "claim": "There is exactly one interpolation axis — time. No metallicity axis and no stellar-mass/IMF axis exist; cluster-mass dependence is a single exact linear factor f_mass on every extensive quantity, applied at load time with no plausibility bound.",
    "evidence": "read_sps.py:172-173 `if CANONICALS[canonical].mass_scaled: arr = arr * f_mass` is the only mass dependence in the module. get_interpolation:341-354 builds ten 1-D interp1d objects over t only. Metallicity and IMF are implicit in whichever single file sps_path names; nothing interpolates between tables. f_mass is checked only for `isfinite and > 0` (line 112).",
    "expected": "N/A — this is a faithful description of the model, recorded because 'axes' was asked for. Worth noting that Qi, Lbol and Lmech are taken as exactly proportional to cluster mass with no stochastic-IMF or small-N correction and no upper/lower bound on the extrapolation in f_mass.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-A-19",
    "file": "trinity/sps/read_sps.py",
    "line": 24,
    "class": "other",
    "severity": "S4",
    "claim": "`import scipy` alone is relied on to make `scipy.interpolate` accessible; this only works via SciPy's lazy-subpackage __getattr__ (SciPy >= 1.9) or if another module happened to import scipy.interpolate first.",
    "evidence": "read_sps.py:24 `import scipy`, then lines 341-354 use `scipy.interpolate.interp1d`. There is no `import scipy.interpolate` or `from scipy.interpolate import interp1d` anywhere in the file. CLAUDE.md pins scipy<2 with no lower bound stated.",
    "expected": "`from scipy.interpolate import interp1d` (or `import scipy.interpolate`), which is version-independent.",
    "failure_scenario": "On SciPy < 1.9, or in any import order where nothing else has pulled in scipy.interpolate, get_interpolation raises AttributeError: module 'scipy' has no attribute 'interpolate'.",
    "repro": "python -c \"import scipy; scipy.interpolate\" on the pinned scipy version.",
    "confidence": "medium"
  }
]
```
