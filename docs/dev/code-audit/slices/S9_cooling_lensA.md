# S9 cooling — Lens A (what the code does)

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

Read with all comments/docstrings blanked. Line numbers are original-file line numbers.
Shared read-only exception **used**: `/tmp/.../phase2/S1_units_helpers/code/_functions/unit_conversions.py`
(only to resolve the numeric values of `ndens_cgs2au`, `phi_cgs2au`, `dudt_cgs2au`).
No `trinity/` source, no `lib/` tables, no `docs/dev/`, no `prose.md`, no other agent's report was read.

Environment used for the behavioural repros below: numpy 1.26.4, scipy 1.17.1.

---

## 1. The net cooling/heating expression, substituted to the inputs

`get_dudt(age, ndens, T, phi, params_dict)` — `trinity/cooling/net_coolingcurve.py:58`.

Input conditioning (the only unit work in the function):

```
net_coolingcurve.py:82   n   [cm^-3]        = ndens_au / 2.937998946096347e+55      (pc^-3 -> cm^-3)
net_coolingcurve.py:83   phi [cm^-2 s^-1]   = phi_au  / 3.0047272630641653e+50      (pc^-2 Myr^-1 -> cm^-2 s^-1)
                          T   [K]            = T, unconverted
                          age                = never referenced inside the function
```

Objects pulled from `params_dict` (99–104): `cStruc_cooling_nonCIE` (a `cube`, used only for its
`.temp` axis), `cStruc_net_nonCIE_interpolation` (`netcool_interp`), `cStruc_cooling_CIE_interpolation`
(`CIE_interp`), `cStruc_cooling_CIE_logT` (`logT_CIE`).

Regime boundaries (121–122, computed by 40–45 and 48–55):

```
Tmin_nCIE  = min(log10T grid of the CLOUDY cube)                     (:43)
Tcut_nCIE  = max{ x in log10T grid of the CLOUDY cube : x <= 5.5 }    (:43)
Tcut_CIE   = min{ x in logT_CIE                       : x >  5.5 }    (:53)
```

Then, with `x = log10(T)` (after the clamp of 130–131) and `Z = params_dict['ZCloud'].value`,
`chi = params_dict['chi_e'].value`, `C = 4.877042454381257e+25` (`dudt_cgs2au`):

**Branch A — non-CIE**, condition `Tmin_nCIE <= x <= Tcut_nCIE` (:138)

```
du/dt |_au  =  -1 * NET(log10 n, x, log10 phi) * C                        (:154, :156)

NET(a,b,c) = trilinear interpolation, on the log10 axes (log_ndens_arr, log_temp_arr, log_phi_arr),
             of the LINEAR-valued array  (cool_cube - heat_cube)          (read_cloudy.py:134-136)
```

There is **no explicit density factor here at all** — no `n**2`, no `n_e n_H`, no `chi_e`. Whatever
`n`-dependence the CLOUDY `cool`/`heat` columns carry is what is used; the density enters only as the
first interpolation axis. The table columns are volumetric rates (see §6), so this branch is
`(Lambda_cool - Gamma_heat)` already integrated over whatever composition CLOUDY assumed.

**Branch B — CIE**, condition `x >= Tcut_CIE` (:159)

```
Lambda_CIE(T) = 10 ** ( CIE_interp( log10 T ) )                  (read_coolingcurve.py:60-62)
du/dt |_au    = -1 * chi_e * n^2 * Lambda_CIE(T) * C              (:163-165)
```

The density factor here is **`chi_e * n_tot^2`** — a single scalar from the parameter file times the
*square of the total number density*. It is **not** `n_e * n_H`, not `n_tot^2` alone, and not `n`.
If `chi_e` is intended as `n_e/n_tot` then the product is `n_e * n_tot`; nothing in this slice pins
that down. Note also `metallicity` is passed to `get_Lambda` and **discarded** (§ finding S9-A-02).

**Branch C — bridge**, condition `Tcut_nCIE < x < Tcut_CIE` (:168)

```
D_lo = NET(log10 n, Tcut_nCIE, log10 phi)                                   (:179)
D_hi = chi_e * n^2 * 10 ** ( CIE_interp( Tcut_CIE ) )                        (:186-187)
w    = (x - Tcut_nCIE) / (Tcut_CIE - Tcut_nCIE)
du/dt|_au = -1 * [ (1-w) * D_lo + w * D_hi ] * C                             (:194, :196)
```

`np.interp` is used with `xp = [Tcut_nCIE, Tcut_CIE]`, so the ramp is **linear in log10 T and linear
in the value of du/dt**.

**Branch D** — `else: raise Exception(...)` (:200–201).

Sign convention: cooling (`NET > 0`, or `Lambda > 0`) yields **negative** `du/dt`. Where the CLOUDY
net is negative (heating dominates) branch A returns a positive `du/dt`. Branches B and C's CIE end
can never be positive.

---

## 2. Table loading and indexing

### 2a. non-CIE (CLOUDY / "opiate") grid — `trinity/cooling/non_CIE/read_cloudy.py`

**File selection** (`get_filename`, :270–344):

```
rot_str = 'rot' if SB99_rotation else 'norot'                    (:289-292)
Z_str   = '1.00' if float(Z)==1.0 else '0.15' if float(Z)==0.15 else -> ValueError   (:294-305)
age     = params['t_now'] * 1e6            [yr]                  (:48)
name    = 'opiate_cooling_' + rot_str + '_Z' + Z_str + '_age' + format(age,'.2e') + '.dat'
```

The available ages are harvested by listing every `*.dat` in `path2cooling` and slicing 8 characters
after the substring `'age'` (`get_fileage`, :349–353) — **regardless of that file's Z or rotation**.

**Column layout** (`create_cubes`, :183–198): `astropy.io.ascii.read` on the `.dat`; columns
`ndens`, `temp`, `phi`, `cool`, `heat`, all **linear**. Sign normalisation at 193–198 inspects only
`heating_data[0]` / `cooling_data[0]` and, if negative, negates the *entire* column.

**Axis construction** (`create_limits`, :204–214):

```
axis = round( log10( sort( unique_linear_values ) ), decimals=3 )
```

so all three axes are **log10, rounded to 3 decimal places**, ascending.

**Cube fill** (:224–237 for cool, :242–254 for heat): the cube is allocated `np.empty` then set to
`np.nan` (:226–228, :244–245) and scatter-filled by *exact float equality* index lookup:

```
i = np.where(log_ndens_arr == np.round(np.log10(ndens_val), 3))[0][0]      (:233, :250)
j = np.where(log_temp_arr  == np.round(np.log10(temp_val ), 3))[0][0]      (:234, :251)
k = np.where(log_phi_arr   == np.round(np.log10(phi_val  ), 3))[0][0]      (:235, :252)
cool_cube[i,j,k] = cooling_val      # linear value, NOT log
```

There is **no interpolation onto the grid** — a row either lands exactly on a grid node or raises
`IndexError` from the `[0][0]`. Any (i,j,k) never visited stays `NaN`.

**Age blending** (:68–94). If `get_filename` returns a 2-element list, both cubes are built and
combined by

```
cube(age) = cube_lo + (age - age_lo) * (cube_hi - cube_lo) / (age_hi - age_lo)      (:87-94)
```

i.e. **linear in age (years, not log age) and linear in the cool/heat values**.

**Interpolators built** (:98–136):

| object | axes | value array | method | used by `get_dudt`? |
|---|---|---|---|---|
| `cooling_interpolation` (:98) | log10 n, log10 T, log10 phi | `np.log10(cool_cube)` | `'linear'` | **no** |
| `heating_interpolation` (:100) | same | `np.log10(heat_cube)` | `'linear'` | **no** |
| `netcooling_interpolation` (:136) | same | `cool_cube - heat_cube` (**linear**) | default `'linear'` | **yes** (:154, :179) |

The two log-valued interpolators are wrapped into `cube` objects (:104–127) and returned; the hot
path (`net_coolingcurve.py:154, :179`) uses only the third, plus `cooling_data.temp` for the axis.

**Query mapping** (:154): `netcool_interp([log10(n_cgs), log10(T), log10(phi_cgs)])` — a flat
3-element list is read by `RegularGridInterpolator` as one point of 3 dimensions, returning a shape-`(1,)`
array; `[0]` extracts the scalar (verified).

**Cache** (:172–176, :265): `<stem>_cube.npy` beside the `.dat`; if present it is loaded with
`allow_pickle=True` and the parse is skipped entirely. Otherwise the parse runs and the result is
written back with `np.save`.

### 2b. CIE curve — `trinity/cooling/CIE/read_coolingcurve.py`

```
get_Lambda(T, cooling_CIE_interpolation, metallicity):
    T      = np.log10(T)                       (:60)   # parameter rebound to its own log
    Lambda = 10 ** ( cooling_CIE_interpolation(T) )    (:62)
    return Lambda                              (:64)
```

That is the whole function. The interpolant maps **log10 T -> log10 Lambda**, and the caller
exponentiates: **log–log interpolation** in the CIE branch. `metallicity` is accepted and never read.
The interpolant object itself is constructed **outside this slice** (it arrives as
`params_dict['cStruc_cooling_CIE_interpolation']`), so its type, its axis extent, and its out-of-range
policy are **not determinable from the files I was given** — see §3.

---

## 3. Off-grid behaviour — exhaustive

`scipy.interpolate.RegularGridInterpolator` defaults (verified on scipy 1.17.1):
`method='linear', bounds_error=True, fill_value=nan`. Lines 98/100 pass only `method`; line 136
passes nothing. So **all three non-CIE interpolators raise on out-of-bounds**, they do not clamp and
do not fill.

### Temperature

| query | behaviour | where |
|---|---|---|
| `log10 T < Tmin_nCIE` | **silently clamped**: `T = 10**Tmin_nCIE`. No warning, no flag, no record. Every temperature below the table floor returns the identical rate. | `net_coolingcurve.py:130-131` |
| `Tmin_nCIE <= log10 T <= Tcut_nCIE` | genuine trilinear **interpolation** in T | `:138`, `:154` |
| `Tcut_nCIE < log10 T < Tcut_CIE` | **not a table lookup in T at all** — the two endpoint values are evaluated at the fixed cutoff temperatures and linearly ramped. `np.interp` would clamp outside `[Tcut_nCIE, Tcut_CIE]`, but the branch guard makes that unreachable. | `:168`, `:179`, `:186`, `:194` |
| `log10 T >= Tcut_CIE` | handed to `CIE_interp` with no bounds check of any kind. **Cannot be determined from this slice**: if the interpolant is `interp1d` with defaults it raises `ValueError`; if it is a spline / `CubicSpline` / `interp1d(fill_value='extrapolate')` it **silently extrapolates log Lambda**, i.e. extrapolates a steep curve in log–log. | `:159`, `read_coolingcurve.py:62` |
| `T` is NaN | `log10(nan)=nan`; the clamp test at :130 is False, and all three branch tests are False -> falls to `else` and raises a generic `Exception` naming `T = nan`. This is the *only* reachable path into the `else`. | `:200-201` |
| `T = +inf` | `x = inf >= Tcut_CIE` -> CIE branch evaluated at `inf`. | `:159` |

`T` above the CLOUDY grid top but below `Tcut_nCIE` is impossible, since `Tcut_nCIE <= max(grid)`.

### Number density and photon flux

**No clamping, no guard, at any point.** `log10(n_cgs)` outside `[log_ndens_arr[0], log_ndens_arr[-1]]`
or `log10(phi_cgs)` outside the phi axis raises

```
ValueError: One of the requested xi is out of bounds in dimension <k>
```

from inside `netcool_interp` (`net_coolingcurve.py:154` and `:179`) — i.e. from inside the ODE
right-hand side, with no context about which quantity or which run state produced it. This is
**asymmetric with the temperature handling**, which clamps silently. Note that :179 evaluates
`netcool_interp` in the *bridge* branch too, so an out-of-range `n` or `phi` also kills branch C.

### NaN holes in the cube

Any (n, T, phi) triple absent from the `.dat` leaves `NaN` in the cube (`read_cloudy.py:228, :245`).
`RegularGridInterpolator` does not detect this: a query inside any cell touching a NaN vertex returns
`NaN` **silently** (verified). `NaN` then propagates: `-1 * nan * 4.877e25 = nan` into the integrator.
The same holds for the age blend at :91 (NaN in either endpoint cube poisons the blend) and for
`np.log10(cool_cube)` at :98/:100 (a zero entry becomes `-inf`).

### Age

| query | behaviour | where |
|---|---|---|
| exact match (float `==` against the parsed file ages) | that file | `:319-323` |
| `age >= max(age_list)` | **silently clamped** to the newest snapshot | `:325-329` |
| `age <= min(age_list)` | **silently clamped** to the oldest snapshot | `:330-334` |
| between | brackets and blends **linearly in age** | `:340-344`, `:87-94` |
| directory has no `.dat` | `max()`/`min()` on an empty array -> `ValueError` | `:311-317` |

### Metallicity

`ZCloud not in {1.0, 0.15}` raises an explicit, well-worded `ValueError` (`:300-305`) — the one
off-grid case in this slice that is handled loudly. The test is exact float equality (`:294, :297`),
so `ZCloud = 0.9999` raises. The CIE branch ignores `ZCloud` entirely (§ S9-A-02).

### Degenerate cutoff selection

`max(t[t <= 5.5])` (`:43`) and `min(logT_CIE[logT_CIE > 5.5])` (`:53`) raise
`ValueError: max() arg is an empty sequence` if a table has no grid point on the relevant side of
5.5 — a bare stdlib error with no mention of cooling tables.

---

## 4. Interpolation in log versus linear space

| interpolant | n axis | T axis | phi axis | age | **value** |
|---|---|---|---|---|---|
| `netcool_interp` — **the one the hot path uses** (`read_cloudy.py:136`) | log10 | log10 | log10 | — | **linear** (`cool - heat`) |
| `cooling_interpolation` (`:98`) — built, unused for the rate | log10 | log10 | log10 | — | log10 |
| `heating_interpolation` (`:100`) — built, unused for the rate | log10 | log10 | log10 | — | log10 |
| age blend `cube_linear_interpolate` (`:87-94`) | — | — | — | **linear (yr)** | **linear** |
| CIE `get_Lambda` (`read_coolingcurve.py:60-62`) | — | log10 | — | — | **log10** (`10 ** interp`) |
| bridge `np.interp` (`net_coolingcurve.py:194`) | — | log10 | — | — | **linear** (in `du/dt`) |

So the code is **log–log for CIE** and **log-axis / linear-value for non-CIE**. The two log-valued
non-CIE interpolators that *would* give log–log behaviour are constructed and then not used for the
rate. Interpolating `cool - heat` linearly is arguably forced (the net can be negative, so `log10`
is unavailable), but the consequence is that a rate spanning orders of magnitude is chorded linearly
across cells that are a decade wide in n, T and phi.

---

## 5. The CIE / non-CIE switch

Exact condition, in evaluation order (`net_coolingcurve.py:130 -> 138 -> 159 -> 168 -> 200`):

```
if   log10 T <  Tmin_nCIE            : T <- 10**Tmin_nCIE          (clamp, then fall through)
if   Tmin_nCIE <= log10 T <= Tcut_nCIE : non-CIE table
elif log10 T >= Tcut_CIE               : CIE curve
elif Tcut_nCIE < log10 T < Tcut_CIE    : linear ramp between the two
else                                   : raise
```

with `Tcut_nCIE = max{grid log T <= 5.5}` and `Tcut_CIE = min{CIE log T > 5.5}`. The hardcoded split
is `5.5` (i.e. 10^5.5 K ~= 3.16e5 K), appearing twice (`:43`, `:53`).

**Do the two agree at the boundary?** The *assembled function* is continuous in value by
construction — branch C reproduces branch A's value at `Tcut_nCIE` and branch B's value at
`Tcut_CIE`. But **nothing checks that the two physical models agree**: the code never compares
`NET(n, Tcut_nCIE, phi)` with `chi_e n^2 Lambda_CIE(10^Tcut_nCIE)`. Whatever the mismatch between
them is, it is absorbed silently into the ramp. Consequences:

- The **derivative** `d(du/dt)/d(log T)` is discontinuous at both cutoffs (two kinks).
- **Heating disappears across the ramp.** Branch A returns `cool - heat`; branches B and C's CIE end
  return cooling only. Photoheating (the `phi`-dependent term) is linearly faded to zero over
  `[Tcut_nCIE, Tcut_CIE]` and is identically absent above `Tcut_CIE`.
- `n` and `phi` dependence changes character across the ramp: below `Tcut_nCIE` the `n`-dependence is
  whatever CLOUDY tabulated and there is `phi` dependence; above `Tcut_CIE` it is exactly `n^2` and
  there is no `phi` dependence at all.
- The ramp width is set entirely by table sampling near 5.5, not by physics: if the CLOUDY grid has a
  node at exactly 5.5 and the CIE grid's first node above 5.5 is at 5.55, the ramp is 0.05 dex wide;
  a coarser CIE grid makes it arbitrarily wide.

**Caching of the cutoffs.** `_noncie_cutoffs` (`:40-45`) memoises on the `cooling_nonCIE` object via a
`_hotpath_cutoffs` attribute — safe, lifetime-tied. `_cie_tcutoff` (`:48-55`) memoises in a
module-level dict keyed on `id(logT_CIE)` (`:27, :50`), never evicted — see S9-A-06.

---

## 6. Dimensions

| term | units, from the arithmetic |
|---|---|
| `ndens` in (au) | pc^-3 |
| `ndens` after `:82` | cm^-3 (÷ 2.937998946096347e55 = (pc/cm)^3, verified: (3.0856775814913673e18)^3 = 2.9379989460963475e55) |
| `phi` in (au) | pc^-2 Myr^-1 |
| `phi` after `:83` | cm^-2 s^-1 (÷ 3.0047272630641653e50 ~= pc^2 * Myr; matches to ~2e-5, a year-definition rounding owned by the units slice) |
| `T` | K, never converted (au T == cgs T assumed) |
| `age` arg of `get_dudt` | unused |
| `age` in `read_cloudy` | **yr** (`t_now` Myr x 1e6, `:48`) |
| `.dat` `cool`, `heat` columns | must be **erg cm^-3 s^-1** (volumetric) for the branches to agree |
| `NET = cool - heat` | erg cm^-3 s^-1 |
| `Lambda_CIE = 10**interp(log10 T)` | must be **erg cm^3 s^-1** (a cooling coefficient) |
| `chi_e` | must be dimensionless |
| `chi_e * n^2 * Lambda_CIE` | 1 x cm^-6 x erg cm^3 s^-1 = erg cm^-3 s^-1 — **balances against NET** |
| `dudt_cgs2au = 4.877042454381257e25` | erg cm^-3 s^-1 -> au. Verified: 5.260183968837699e-44 (erg->E_au) x 2.937998946096347e55 (cm^3/pc^3) x 3.15576e13 (s/Myr) = 4.877042454381258e25 |
| return | au energy-density rate (M_sun pc^-1 Myr^-3 in the code's au system) |

Dimensionally the two branches balance **only under the assumption** that the table's `cool`/`heat`
are volumetric and the CIE table is a coefficient. Nothing in the code asserts, converts, or checks
either. Note the asymmetry: the non-CIE branch carries its density dependence *inside* the table,
the CIE branch carries it *outside* as `chi_e n^2`; that is a real modelling choice, not a units bug,
but it means the two are consistent only if CLOUDY's tabulated `n`-scaling equals `chi_e n^2`.

---

## 7. Numeric literals in arithmetic

**`trinity/cooling/net_coolingcurve.py`**

| line | literal | expression |
|---|---|---|
| 43 | `5.5` | `max(t[t <= 5.5])` — non-CIE upper cutoff, log10 K |
| 43 | — | `min(t)` — non-CIE floor |
| 53 | `5.5` | `min(logT_CIE[logT_CIE > 5.5])` — CIE lower cutoff, log10 K |
| 82 | `cvt.ndens_cgs2au` = 2.937998946096347e+55 | `ndens /= ...` |
| 83 | `cvt.phi_cgs2au` = 3.0047272630641653e+50 | `phi /= ...` |
| 131 | `10**` | `T = 10**nonCIE_Tmin` |
| 154, 179 | `[0]` | scalar extraction from the shape-(1,) RGI result |
| 156, 165, 196 | `-1 *`, `cvt.dudt_cgs2au` = 4.877042454381257e+25 | `return -1 * dudt * C` |
| 164, 187 | `**2` | `chi_e * ndens**2 * Lambda` |
| 186 | `10**` | `CIE.get_Lambda(10**CIE_Tcutoff, ...)` |
| 194 | — | `np.interp(log10 T, [Tcut_nCIE, Tcut_CIE], [d_lo, d_hi])` |

**`trinity/cooling/CIE/read_coolingcurve.py`**

| line | literal | expression |
|---|---|---|
| 60 | — | `T = np.log10(T)` |
| 62 | `10**` | `Lambda = 10**(cooling_CIE_interpolation(T))` |

**`trinity/cooling/non_CIE/read_cloudy.py`**

| line | literal | expression |
|---|---|---|
| 48 | `1e6` | `age = params['t_now'] * 1e6` (Myr -> yr) |
| 91 | — | `cubes_low + (x-ages_low)*(cubes_high-cubes_low)/(ages_high-ages_low)` |
| 172 | `-4` | `filename[:-4]` (strip `.dat`) |
| 193, 196 | `-1`, `[0]` | `np.sign(heating_data[0]) == -1` ; `-1 * heating_data` |
| 213 | `decimals = 3` | `np.round(log10(axis), 3)` |
| 233–235, 250–252 | `decimals = 3`, `[0][0]` | `np.where(axis == np.round(np.log10(val), 3))[0][0]` |
| 294, 296 | `1.0`, `'1.00'` | metallicity -> filename token |
| 297, 299 | `0.15`, `'0.15'` | metallicity -> filename token |
| 313 | `-4`, `'.dat'` | `files[-4:] == '.dat'` |
| 320, 326, 331 | `'.2e'` | `format(age, '.2e')` |
| 353 | `+3`, `+3+8` | `float(filename[i+3 : i+3+8])` — fixed 8-character age field |

Unused imports (no arithmetic effect, noted for completeness): `scipy.interpolate`, `astropy.units`,
`sys` in `net_coolingcurve.py`; `numpy`, `sys`, `scipy`, `astropy.units` are imported in
`read_coolingcurve.py` and only `numpy` is used.

---

## 8. Notable inconsistency: `params['t_now']` vs `params[...].value`

`read_cloudy.py:48` reads `params['t_now'] * 1e6`, with **no `.value`**, while the immediately
following lines 50–52 use `params['path_cooling_nonCIE'].value`, `params['SB99_rotation'].value`,
`params['ZCloud'].value`. Either `t_now` is stored as a bare float while its neighbours are wrapper
objects, or the multiplication is happening on a wrapper. I cannot resolve which from this slice.

---

```json
[
  {
    "id": "S9-A-01",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 130,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Temperatures below the non-CIE table floor are silently clamped to the floor, with no warning, flag, or record; every such T returns the identical cooling rate.",
    "evidence": "Lines 130-131: `if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`, where nonCIE_Tmin = min(cooling_nonCIE.temp) (:43) is the minimum of the CLOUDY log10-T axis. Nothing downstream records that the clamp fired.",
    "expected": "Either an explicit, logged/raised out-of-range condition, or a documented clamp that the caller can detect. Contrast the ndens and phi axes, which are not clamped at all and instead raise from RegularGridInterpolator (bounds_error defaults to True).",
    "failure_scenario": "A shell or bubble zone that cools below the table floor (typically 10^2 K for CLOUDY grids) keeps receiving the floor-temperature cooling rate instead of the much weaker true rate. du/dt is then wrong by whatever the extrapolation error is, with no diagnostic; downstream phase-transition triggers that watch du/dt or the cooling time fire at the wrong epoch.",
    "repro": "Call get_dudt with T = 10**(nonCIE_Tmin - 2) and T = 10**nonCIE_Tmin; the two returns are bit-identical.",
    "confidence": "high"
  },
  {
    "id": "S9-A-02",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "deadcode",
    "severity": "S2",
    "claim": "get_Lambda accepts a `metallicity` argument and never uses it; the CIE branch applies no metallicity scaling at the call site.",
    "evidence": "The entire body is lines 60-64: `T = np.log10(T)`; `Lambda = 10**(cooling_CIE_interpolation(T))`; `return Lambda`. `metallicity` appears only in the signature. Both call sites pass it: net_coolingcurve.py:163 and :186 pass `params_dict['ZCloud'].value`.",
    "expected": "Either the CIE interpolant is built per-metallicity upstream (in which case the parameter is dead and should go, and the CIE table's Z must be verified to match ZCloud), or the Z scaling is genuinely missing. The non-CIE branch does honour Z, via file selection (read_cloudy.py:294-305) - so the two branches are asymmetric in Z handling either way.",
    "failure_scenario": "If the CIE interpolant is built at a fixed (e.g. solar) metallicity, a Z=0.15 run uses solar CIE cooling above ~10^5.5 K while using Z=0.15 CLOUDY cooling below it - a several-times error in the hot-gas cooling rate, and a step in the physical content at the CIE/non-CIE bridge. If Z is instead applied both upstream and (hypothetically) here, it would be applied twice.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S9-A-03",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 136,
    "class": "numerical",
    "severity": "S2",
    "claim": "The interpolator actually used in the hot path interpolates the LINEAR net rate on log10 axes, while the two log-valued interpolators built alongside it (lines 98, 100) are never used for the rate.",
    "evidence": "Line 98 builds `RegularGridInterpolator(axes, np.log10(cool_cube))` and line 100 the same for heat; line 136 builds `RegularGridInterpolator(axes, netcooling)` with `netcooling = cool_cube - heat_cube` (:134), linear values, default method. get_dudt uses only the third (net_coolingcurve.py:154, :179); it touches cooling_nonCIE only for `.temp` (:42).",
    "expected": "A rate spanning many decades chorded linearly across cells a decade wide in n, T and phi. Linear-value interpolation is arguably forced here (the net can be negative, so log10 is unavailable), but the mismatch against the CIE branch - which is fully log-log (read_coolingcurve.py:60-62) - and against the two unused log-valued interpolators should be deliberate and stated, not incidental.",
    "failure_scenario": "Between grid nodes the net rate is systematically overestimated in convex regions of the cooling curve (a chord lies above a convex function), by a factor that grows with cell width. Cells straddling the peak of the cooling curve are worst.",
    "repro": "Compare netcool_interp at a cell midpoint against 10**cooling_interpolation(midpoint) - 10**heating_interpolation(midpoint) for a cell where cool varies by more than ~0.5 dex across the cell.",
    "confidence": "high"
  },
  {
    "id": "S9-A-04",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 265,
    "class": "numerical",
    "severity": "S2",
    "claim": "`np.save` of the ragged list [1-D axis, 1-D axis, 1-D axis, 3-D cube, 3-D cube] raises ValueError on numpy >= 1.24, so create_cubes cannot complete for any table lacking a pre-built `_cube.npy` cache.",
    "evidence": "Line 265: `np.save(cube_filename, [log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube])`. np.save calls np.asanyarray on the list; the shapes are inhomogeneous. Verified on numpy 1.26.4 (the installed version; CLAUDE.md pins numpy<2): `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (5,) + inhomogeneous part.` The load side at :175 already passes allow_pickle=True, so an object array was intended.",
    "expected": "`np.save(cube_filename, np.array([...], dtype=object))`, or np.savez with named arrays. As written the whole parse (lines 183-254) completes and is then thrown away by the exception on the last line before the return.",
    "failure_scenario": "Any user pointing path_cooling_nonCIE at their own opiate tables, or any run after the bundled *_cube.npy files are deleted/regenerated, crashes in get_coolingStructure with an opaque numpy shape error that names neither cooling nor the table.",
    "repro": "python -c \"import numpy as np; np.save('/tmp/x.npy',[np.arange(3.),np.arange(4.),np.arange(2.),np.zeros((3,4,2)),np.zeros((3,4,2))])\"",
    "confidence": "high"
  },
  {
    "id": "S9-A-05",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 228,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Cube cells for (n,T,phi) combinations absent from the .dat file remain NaN, and RegularGridInterpolator returns NaN silently for any query in a cell touching a NaN vertex; the NaN then propagates into du/dt and the integrator with no diagnostic.",
    "evidence": "Lines 226-228 allocate cool_cube and set it entirely to NaN; 231-237 scatter-fill only the rows present in the file. Same for heat at 244-254. There is no post-fill completeness check (no `np.isnan(cool_cube).any()` assert). Verified: an RGI over a values array with one NaN vertex returns nan for a query in the containing cell. net_coolingcurve.py:156 then returns `-1 * nan * 4.877e25`.",
    "expected": "A completeness assertion after the fill loops, or fill_value/nan-aware handling. A NaN cooling rate should be loud.",
    "failure_scenario": "A non-rectangular opiate table (CLOUDY runs that failed to converge for some parameter corners are commonly dropped from the output) yields NaN du/dt in exactly the parameter regime where the physics was hardest. The ODE solver then either stalls, rejects every step, or silently carries NaN into the state vector.",
    "repro": "Delete one row from an opiate .dat (and the corresponding _cube.npy), rebuild, and query netcool_interp at the midpoint of the affected cell.",
    "confidence": "high"
  },
  {
    "id": "S9-A-06",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 50,
    "class": "state",
    "severity": "S3",
    "claim": "_CIE_TCUTOFF_CACHE is a module-level dict keyed on `id(logT_CIE)` and never evicted; CPython reuses id values after garbage collection, so a different array allocated at a freed address returns the previous array's cutoff.",
    "evidence": "Line 27 `_CIE_TCUTOFF_CACHE: dict = {}`; lines 50-54 `key = id(logT_CIE)` ... `_CIE_TCUTOFF_CACHE[key] = cached`. Nothing holds a reference to logT_CIE, so the cache does not keep the keyed object alive. Contrast _noncie_cutoffs (:40-45), which stores the cache as an attribute on the object itself and is therefore lifetime-safe.",
    "expected": "Attribute-based memoisation on the array's owning object (as _noncie_cutoffs does), or a key derived from the array contents.",
    "failure_scenario": "A parameter sweep or any workflow that builds a second cooling structure in-process (a new CIE logT array allocated where the first was freed) gets the first structure's CIE_Tcutoff. The switch temperature is then wrong, silently, and the bridge branch interpolates over the wrong interval - potentially with nonCIE_Tcutoff >= CIE_Tcutoff, which makes np.interp's xp non-increasing and produces garbage without raising.",
    "repro": "In one process, build cooling structure A, call get_dudt, drop all references to A, force gc, build structure B with a different CIE log-T grid, and compare _cie_tcutoff(B.logT) against min(B.logT[B.logT>5.5]).",
    "confidence": "medium"
  },
  {
    "id": "S9-A-07",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 325,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Cluster ages outside the tabulated SB99 age range are silently clamped to the first or last snapshot with no warning.",
    "evidence": "Lines 325-329: `elif age >= max(age_list): age_str = format(max(age_list),'.2e')` -> that filename. Lines 330-334: the mirror image for `age <= min(age_list)`. No print, no warning, no returned flag; contrast the metallicity path at :300-305, which raises with a clear message.",
    "expected": "At least a cpr.WARN print (the module already imports cprint and uses it at :195/:198), or an explicit error, so a run that has outlived its cooling tables is visible.",
    "failure_scenario": "A simulation integrated past the last SB99 snapshot keeps using that snapshot's photoionisation-dependent cooling/heating forever. The heating term in particular stops declining with the ageing cluster, so the bubble is over-heated at late times and the phase-transition timing is wrong - with nothing in the output indicating the table stopped evolving.",
    "repro": "Run with t_now beyond the newest opiate age and check that get_filename returns the newest file rather than signalling.",
    "confidence": "high"
  },
  {
    "id": "S9-A-08",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 311,
    "class": "other",
    "severity": "S3",
    "claim": "The available-age list is harvested from every *.dat in the cooling directory regardless of that file's metallicity or rotation flag, then a filename is composed from the requested Z/rotation and the harvested age.",
    "evidence": "Lines 310-317 iterate `os.listdir(path2cooling)`, filter only on the `.dat` suffix, and append `get_fileage(files)`. Z_str and rot_str (computed at 289-305) never enter the filter. The composed name at :322/:328/:333/:343 uses the requested Z_str and rot_str with an age that may only exist for a different Z or rotation.",
    "expected": "Filter the listing by the same `opiate_cooling_{rot_str}_Z{Z_str}_age` prefix that the filename is built from.",
    "failure_scenario": "If the Z=1.00 and Z=0.15 (or rot and norot) table sets have different age sampling, get_filename returns a name that does not exist and the run dies later in ascii.read with a FileNotFoundError naming a file the user never asked for; or, in the bracketing branch (:340-344), it brackets against ages from the wrong table set and blends across the wrong interval.",
    "repro": "Place a single extra `opiate_cooling_norot_Z0.15_age9.99e+06.dat` in the directory and request Z=1.0 at t_now = 9.99 Myr.",
    "confidence": "high"
  },
  {
    "id": "S9-A-09",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 193,
    "class": "sign",
    "severity": "S3",
    "claim": "The sign normalisation of the cool and heat columns inspects only element [0] and then negates the entire column.",
    "evidence": "Lines 193-198: `if np.sign(heating_data[0]) == -1: heating_data = -1 * heating_data` (and the same for cooling_data). A single sample decides for the whole column, and the negation is unconditional across all entries.",
    "expected": "A whole-column test (e.g. `if (heating_data < 0).all()`), or a per-element `np.abs`, with an error on genuinely mixed signs.",
    "failure_scenario": "A table whose first row happens to be zero or positive while the rest are negative passes through unflipped, so heat enters `cool - heat` (:134) with the wrong sign and heating is added to cooling instead of subtracted - a sign error in the net rate over the whole grid. Conversely a mixed-sign column is corrupted by the blanket negation. Neither case warns.",
    "repro": "Set the first heat entry of a copy of an opiate table to 0.0 while leaving the rest negative, delete the cache, and inspect the resulting heat_cube sign.",
    "confidence": "high"
  },
  {
    "id": "S9-A-10",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 194,
    "class": "regime",
    "severity": "S2",
    "claim": "Photoheating is present below the non-CIE cutoff, linearly faded to zero across the bridge, and identically absent above the CIE cutoff; the phi dependence of du/dt therefore vanishes across the switch and the two models are never checked for agreement at the boundary.",
    "evidence": "Branch A (:154) returns the interpolated `cool - heat`, which carries the phi axis. Branch B (:163-165) returns `chi_e * ndens**2 * Lambda_CIE(T)` - no heat term, no phi dependence. Branch C (:179-194) linearly ramps between one value that includes heating and phi and one that includes neither. Nothing compares NET(n, Tcut_nCIE, phi) with chi_e n^2 Lambda_CIE(10**Tcut_nCIE).",
    "expected": "Either a check/report of the model mismatch at the join, or a documented statement that heating is negligible above 10^5.5 K. As written the size of any discontinuity between the two models is hidden inside the ramp, and du/dt has a kink at both cutoffs.",
    "failure_scenario": "If the two models disagree substantially at 10^5.5 K (a factor of a few is common between a photoionised CLOUDY net rate and a pure-CIE Lambda), the ramp manufactures a fictitious steep gradient in du/dt over a narrow log-T interval. Shell/bubble zones sitting in that interval get a rate that depends on table sampling rather than on physics, and the ramp width itself is set by wherever the CIE grid's first node above 5.5 lands.",
    "repro": "Evaluate NET(log n, Tcut_nCIE, log phi) and chi_e*n**2*get_Lambda(10**Tcut_nCIE, ...) at matched n, phi and print the ratio.",
    "confidence": "high"
  },
  {
    "id": "S9-A-11",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 154,
    "class": "other",
    "severity": "S3",
    "claim": "Density and photon flux are never bounds-checked or clamped; an out-of-grid n or phi raises a bare scipy ValueError from inside the ODE right-hand side, asymmetrically with temperature which is silently clamped.",
    "evidence": "netcool_interp is built at read_cloudy.py:136 with no bounds_error argument, so scipy's default bounds_error=True applies (verified on scipy 1.17.1: `ValueError: One of the requested xi is out of bounds in dimension 0`). net_coolingcurve.py only guards T (:130-131); log10(ndens) and log10(phi) go straight into the interpolator at :154 and :179.",
    "expected": "Consistent treatment across the three axes - either all clamp with a recorded flag, or all raise a domain-specific error naming the quantity, its value, and the table range.",
    "failure_scenario": "A dense or very rarefied shell zone, or a phi outside the tabulated ionising-flux range, aborts the whole run with a scipy message that identifies only the axis index, giving no clue that the cooling table was the cause or which state variable went out of range.",
    "repro": "Call get_dudt with ndens set an order of magnitude above the top of log_ndens_arr.",
    "confidence": "high"
  },
  {
    "id": "S9-A-12",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 200,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The final `else: raise Exception(...)` is unreachable for any finite T; it is reached only when T is NaN, where it becomes an opaque generic Exception rather than a NaN diagnostic.",
    "evidence": "After the clamp at :130-131, log10(T) >= nonCIE_Tmin always. The three branch conditions (:138 `<=Tcut_nCIE and >=Tmin`, :159 `>=Tcut_CIE`, :168 `>Tcut_nCIE and <Tcut_CIE`) tile [Tmin, +inf) whenever Tcut_nCIE < Tcut_CIE, which the 5.5 split at :43/:53 guarantees. For T = NaN every comparison is False (including the clamp test), so control falls through to :201 and prints `T = nan`.",
    "expected": "Either an explicit NaN guard with a message that says so, or acknowledgement that this is the NaN funnel. As written a NaN temperature - the most likely real cause - surfaces as 'Temperature T = nan not understood'.",
    "failure_scenario": "A NaN entering the state vector (e.g. from a NaN cube cell, S9-A-05) is reported as an un-typed Exception from the cooling module rather than as a NaN propagation, sending debugging in the wrong direction.",
    "repro": "get_dudt(age, ndens, float('nan'), phi, params_dict)",
    "confidence": "high"
  },
  {
    "id": "S9-A-13",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The `age` parameter of get_dudt is never referenced in the function body.",
    "evidence": "`def get_dudt(age, ndens, T, phi, params_dict)` at :58; `age` appears nowhere in lines 59-201. The age dependence of the tables is resolved once, upstream, when cStruc_net_nonCIE_interpolation is built (read_cloudy.py:48-94).",
    "expected": "Drop the parameter, or use it to assert that the cached interpolation structure matches the requested age.",
    "failure_scenario": "A caller could reasonably believe get_dudt re-selects the age-appropriate table per call. It does not - the cooling structure is frozen at whatever age it was built for, and a stale structure would be used silently.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-A-14",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 164,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The CIE branch multiplies the cooling coefficient by `chi_e * n_tot**2` (a scalar times total number density squared), while the non-CIE branch applies no density factor at all - the density scaling is baked into the CLOUDY table. The two are only mutually consistent if CLOUDY's tabulated n-scaling equals chi_e * n^2.",
    "evidence": "Line 164 `dudt = params_dict['chi_e'].value * ndens**2 * Lambda_CIE` and the identical form at :187. Branch A (:154) returns the interpolated table value directly with no multiplier. `ndens` at this point is the total number density in cm^-3 (converted at :82), not n_e and not n_H.",
    "expected": "A cooling rate should be n_e * n_H * Lambda (or n_e * n_i). Using chi_e * n_tot^2 is equivalent only if chi_e is defined as (n_e n_H)/n_tot^2 for the composition the CIE table assumes - a single constant scalar, so it cannot track the actual ionisation state, and it must not double-count any electron fraction already folded into the CIE table.",
    "failure_scenario": "A wrong or double-counted chi_e is a fixed multiplicative error on all cooling above 10^5.5 K - exactly the hot-bubble interior whose cooling losses set the energy-driven to momentum-driven transition. Because it is a clean constant factor it never shows up as an instability, only as a shifted transition time.",
    "repro": "Compare chi_e against (n_e n_H / n_tot^2) for the composition assumed by the CIE table, and check whether the CIE table's Lambda is already an n_e n_H-normalised coefficient.",
    "confidence": "medium"
  },
  {
    "id": "S9-A-15",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 207,
    "class": "numerical",
    "severity": "S3",
    "claim": "Axis construction dedupes on the LINEAR values before taking log10 and rounding to 3 decimals, so two distinct linear values can collapse to the same axis entry; the cube is then indexed by exact float equality against those rounded values.",
    "evidence": "create_limits (:204-214): `set(array)` -> sort -> log10 -> `np.round(..., decimals=3)`, with no second dedupe after rounding. The fill loops index by `np.where(log_arr == np.round(np.log10(val), 3))[0][0]` (:233-235, :250-252). A duplicated axis entry makes RegularGridInterpolator raise `ValueError: The points in dimension 0 must be strictly ascending or descending` (verified); if construction somehow survived, `[0][0]` takes the first match and the duplicate slot stays NaN. A value that matches nothing raises IndexError from `[0][0]`.",
    "expected": "Round first, then dedupe (`np.unique(np.round(np.log10(array), 3))`), so the axis and the lookup key are derived from the same rounded quantity.",
    "failure_scenario": "A finely sampled table (grid spacing below ~0.001 dex, or values differing only in the fourth significant figure) fails to build with a scipy message about ascending points that names no table and no column. A non-positive value in any of the three columns produces -inf/nan and an IndexError instead.",
    "repro": "Construct a two-row opiate-format table with ndens = 1.0 and 1.0005 and run create_cubes on it.",
    "confidence": "high"
  },
  {
    "id": "S9-A-16",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 174,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The `_cube.npy` cache is keyed only on the .dat filename - no mtime, size, or format-version check - and is written into the table directory itself, so a stale cache silently shadows an edited or replaced table.",
    "evidence": "Lines 172-176: `_stem = filename[:-4] ...; cube_filename = path2cooling + _stem + '_cube.npy'; if os.path.exists(cube_filename): ... return`. The parse at 183+ is skipped entirely on a hit. Line 265 writes the cache back into `path2cooling`.",
    "expected": "Compare the cache mtime against the .dat mtime (or embed a hash), and write the cache somewhere the user owns rather than into what may be a read-only bundled lib/ directory.",
    "failure_scenario": "A user regenerates or edits an opiate table in place; every subsequent run keeps using the old cube with no indication, so the change appears to have no effect. Symmetrically, if path2cooling is read-only the write at :265 raises PermissionError only after the full parse has already run.",
    "repro": "Edit a cool value in a .dat whose _cube.npy exists and confirm the interpolated rate is unchanged.",
    "confidence": "high"
  },
  {
    "id": "S9-A-17",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 91,
    "class": "numerical",
    "severity": "S3",
    "claim": "The between-snapshot age blend is linear in age (years, not log age) and linear in the raw cool/heat values, with no guard against a zero denominator and no NaN handling.",
    "evidence": "Line 91: `return cubes_low + (x - ages_low) * (cubes_high - cubes_low)/(ages_high - ages_low)`. Applied to cool_cube and heat_cube at :93-94, i.e. before the log10 at :98/:100 and before the net at :134. SB99 snapshot ages are typically log-spaced.",
    "expected": "For log-spaced snapshots, interpolating linearly in age across a decade-wide gap heavily weights the newer snapshot; and interpolating rates linearly rather than in log biases high in convex regions - the same chord-above-curve error as S9-A-03, compounded.",
    "failure_scenario": "In the sparsely sampled late-time part of the SB99 age grid, the effective cooling/heating table lags or leads the true cluster age; the phi-dependent heating in particular is over-weighted toward the younger snapshot. A NaN in either endpoint cube also propagates through the blend into every cell (see S9-A-05).",
    "repro": "Blend two adjacent opiate snapshots at the geometric mean age and compare against linear-in-log-age blending of log10(cool).",
    "confidence": "medium"
  },
  {
    "id": "S9-A-18",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 353,
    "class": "other",
    "severity": "S4",
    "claim": "get_fileage parses the age with a hardcoded 8-character slice after the substring 'age'.",
    "evidence": "Line 353: `return float(filename[age_index_begins+3 : age_index_begins+3+8])`. This assumes exactly the `d.dde+dd` shape produced by `format(age, '.2e')` at :320/:326/:331.",
    "expected": "Parse up to the '.dat' suffix, or use a regex, so that a three-digit exponent or any other age formatting is handled.",
    "failure_scenario": "A filename with a three-digit exponent (age >= 1e100, unrealistic) or any hand-renamed table with a differently formatted age field silently yields a truncated float - e.g. '1.00e+100' parses as 1.00e+10 - which then feeds the bracketing logic at :319-344 and selects the wrong snapshot without error. Also `filename.find('age')` returns -1 if 'age' is absent, giving `filename[2:10]` and a ValueError from float().",
    "repro": "get_fileage('opiate_cooling_norot_Z1.00_age1.00e+100.dat') returns 10000000000.0",
    "confidence": "high"
  },
  {
    "id": "S9-A-19",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 48,
    "class": "state",
    "severity": "S4",
    "claim": "`params['t_now']` is used without `.value` while every neighbouring parameter access in the same function uses `.value`.",
    "evidence": "Line 48 `age = params['t_now'] * 1e6`; lines 50-52 `params['path_cooling_nonCIE'].value`, `params['SB99_rotation'].value`, `params['ZCloud'].value`. net_coolingcurve.py likewise uses `.value` on every params_dict access (:99-104, :163-164, :186-187).",
    "expected": "Consistent accessor. Either t_now is a bare float here (an inconsistency in the params container) or the multiplication is operating on a wrapper object.",
    "failure_scenario": "If t_now is ever wrapped like its neighbours, `wrapper * 1e6` either raises or produces a non-float that then fails the `age in age_list` equality at :319 and silently falls through to a clamp branch (:325/:330), selecting an edge snapshot instead of the correct one.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S9-A-20",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 43,
    "class": "regime",
    "severity": "S4",
    "claim": "The CIE/non-CIE regime split is the bare literal 5.5 (log10 K), duplicated in two functions, and a table with no grid point on one side of it fails with a bare stdlib error.",
    "evidence": "Line 43 `cached = (max(t[t <= 5.5]), min(t))`; line 53 `cached = min(logT_CIE[logT_CIE > 5.5])`. The same magic number appears in two independent helpers with no shared constant. If the non-CIE log-T grid has no point <= 5.5, or the CIE grid none > 5.5, the builtin max()/min() raise `ValueError: max() arg is an empty sequence` (verified) with no mention of cooling tables.",
    "expected": "A single named module constant, and a domain-specific error if either table does not straddle the split.",
    "failure_scenario": "Swapping in a cooling table whose temperature coverage does not straddle 10^5.5 K aborts with 'max() arg is an empty sequence' from a helper whose name gives no hint of the cause.",
    "repro": "Call _noncie_cutoffs on a cube object whose .temp is all > 5.5.",
    "confidence": "high"
  }
]
```
