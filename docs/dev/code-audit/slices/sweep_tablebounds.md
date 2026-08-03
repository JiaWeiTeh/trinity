# Cross-cutting sweep ⑦ — table bounds (static half)

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

## Scope and sources

**Question.** TRINITY interpolates external tables at every ODE step. For each interpolator:
what is the valid domain, what happens outside it, and can the code drive outside it?

**Read (all read-only).** `trinity/**` (all 72 `.py`), the bundled tables under
`lib/default/{CIE,opiate,sps}/`, `run.py`, `param/*.param`, `test/`, `tools/`.
Not read: `docs/dev/code-audit/slices/`, `old_doNotRead/`, `outputs/`, `scratch/`, `tbd/`, `fig/`.

**Method.** Grid extents were read out of the actual data files (`.dat` and cached `_cube.npy`),
not from docstrings. Off-grid behaviour was verified by constructing the *same* interpolator with
the *same* kwargs on the *real* bundled arrays and probing every domain face. Reachability was
estimated analytically (Weaver similarity solution + the bundled SB99 `Qi(t)`/`Lmech(t)`); no
simulation was run — every reachability claim below is flagged as estimate vs. certainty.

Environment: `scipy 1.17.1`, `numpy 1.26.4`.

---

## A. Interpolator inventory

`bounds_error` / `fill_value` are as *constructed*; "off-grid" is the **verified** behaviour.

| # | Site | Object | Axes (query space) | Construction | Off-grid behaviour (verified) |
|---|------|--------|--------------------|--------------|-------------------------------|
| 1 | `trinity/main.py:167` | `interp1d` | `log10 T` → `log10 Λ` (CIE) | `kind='linear'`; `bounds_error` default → **True**; `fill_value=nan`; `assume_sorted=False` (sorts internally) | **raises `ValueError`** below `logT_min` / above `logT_max` |
| 2 | `cooling/non_CIE/read_cloudy.py:98` | `RegularGridInterpolator` | `(log n, log T, log φ)` → `log10(cool)` | `method='linear'`; `bounds_error` default → **True** | **raises `ValueError`** ("out of bounds in dimension k"); **NaN** inside NaN-touching cells |
| 3 | `read_cloudy.py:100` | `RegularGridInterpolator` | same → `log10(heat)` | same | same |
| 4 | `read_cloudy.py:136` | `RegularGridInterpolator` | same → `cool − heat` (**linear values**) | `method` default `'linear'`; `bounds_error` default → **True** | same |
| 5 | `read_cloudy.py:87–94` | hand-rolled `cube_linear_interpolate` | cluster **age** (yr) | linear in age between two cubes | never extrapolates — the caller (#6) guarantees bracketing |
| 6 | `read_cloudy.py:319–334` | hand-rolled file picker `get_filename` | cluster **age** (yr) | `if age >= max: use max`, `if age <= min: use min` | **silently CLAMPS to the age-grid endpoints — no warning, no log** |
| 7 | `cooling/net_coolingcurve.py:194` | `np.interp` | `log10 T` on the 2-point CIE↔non-CIE blend | 2-node table `[nonCIE_Tcutoff, CIE_Tcutoff]` | clamps — but the enclosing `elif` guarantees a strictly-interior query |
| 8 | `cooling/net_coolingcurve.py:130–131` | explicit clamp | `T` | `if log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin` | **deliberate, tested** (`test/test_net_coolingcurve.py`) |
| 9 | `sps/read_sps.py:341–354` | 10 × `interp1d` | `t [Myr]` → feedback | `kind='cubic'`; `bounds_error` default → **True** | **raises `ValueError`** outside `[0, t_max]` |
| 10 | `sps/update_feedback.py:153–159` | explicit guard | `t` | `if not (t_min <= t <= t_max): raise ValueError(...)` | **correct pre-check with a readable message** |
| 11 | `sps/update_feedback.py:185` | `fpdot_total(t ± 1e-9)` | `t` | central difference for `pdotdot_total` | **steps outside the guard of #10 at the exact endpoints** → raw scipy `ValueError` |
| 12 | `_input/fkappa_auto.py:72` | `RegularGridInterpolator` | `(log M, log sfe, log n)` (hard-coded 3×3×7) | `method='linear'`; `bounds_error` default True | caller `fkappa_fire` **`np.clip`s to the hull and logs a WARNING** before querying |
| 13 | `cloud_properties/bonnorEbertSphere.py:270` | `interp1d` | `ξ` → `ρ/ρc` | `kind='cubic'`, `bounds_error=False`, `fill_value=(1.0, rho_rhoc[-1])` | **clamps** to physically correct endpoints (`ρ/ρc→1` at centre) |
| 14 | `bonnorEbertSphere.py:275` | `interp1d` | `ξ` → `m(ξ)` | `kind='cubic'`, `bounds_error=False`, `fill_value=(0.0, m[-1])` | clamps; only ever queried at `ξ ≤ ξ_out ≤ ~6.5 << ξ_max=20` |
| 15 | `bonnorEbertSphere.py:283` | `interp1d` | `ρ/ρc` → `ξ` (inverse) | `kind='cubic'`, `bounds_error=False`, `fill_value=(xi[-1], xi[0])` | clamps; `np.unique` dedups + re-sorts, orientation is correct |
| 16 | `bubble_structure/bubble_luminosity.py:718,720,721` | 3 × `interp1d` | `r` (runtime array) | `kind='linear'/'cubic'/'linear'`; `bounds_error` default True | `brentq` bracket is `[min(r_interp), max(r_interp)]` — query is in-domain by construction |
| 17 | `bubble_luminosity.py:803` (eval at `810`, `875`) | `interp1d` | `r` (2-node) | `kind='linear'`; `bounds_error` default True | line 810 in-domain by construction; **line 875 `fT_interp_interm(bubble_r_Tb)` is not proven in-domain** |
| 18 | `phase0_init/get_InitCloudProp.py:509` | `np.interp` | `r` → `M(r)` | default | clamps; diagnostic-only (`verify_mass_at_rCloud`), `rCloud` is inside `r_arr` |
| 19 | `_output/cloudy/dlaw.py:261` | `np.interp` | `log r` | default | query is strictly between the two reference nodes |
| 20 | `_output/trinity_reader.py:851,879` | `interp1d` | snapshot `t` | `kind='linear'`, `bounds_error=False`, **`fill_value='extrapolate'`** | **silently extrapolates**; post-processing reader only |
| 21 | `_functions/simplify.py:739,833,880` | `np.interp` | `x` | explicit ascending sort first | clamps; simplified curve retains both endpoints → no-op |
| 22 | `_analysis/check_yesno.py:119–120` | `np.interp` | `t` | default | `t_grid` is restricted to the overlapping window first → no-op |
| 23 | `_functions/operations.py:19,30,146` | hand-rolled nearest-index | any array | index clamped to `[0, len-1]` at both ends | **silently returns an endpoint index** when the value is off the array |

---

## B. Measured grid extents (ground truth, read from the files)

### B.1 Non-CIE (OPIATE/CLOUDY) cubes — `lib/default/opiate/`

All 13 bundled `.dat` files and all 8 cached `_cube.npy` share **identical** axes:

| Axis | Stored as | N | Min | Max | Step | Physical span |
|------|-----------|---|-----|-----|------|---------------|
| `ndens` | `log10` (cm⁻³) | 33 | **−4.000** | **+12.000** | 0.5 dex | 1e−4 … 1e12 cm⁻³ |
| `temp` | `log10` (K) | 21 | **+3.500** | **+5.500** | 0.1 dex | 3162 … 3.162e5 K |
| `phi` | `log10` (cm⁻² s⁻¹) | 22 | **+0.000** | **+21.000** | 1.0 dex | 1 … 1e21 cm⁻² s⁻¹ |

Cube shape `(33, 21, 22)` = 15 246 cells; the `.dat` supplies **12 034** rows.
**3 212 cells (21.07 %) are `NaN`** — the table simply has no CLOUDY solution there.
Verified: rows == filled cells for every bundled table (no duplicate-abscissa collisions from
`create_limits`' `np.round(..., 3)`); cached `_cube.npy` == cube freshly rebuilt from the `.dat`;
the `heat` NaN mask is **identical** to the `cool` mask (so `netcooling` inherits it exactly).

**The NaN region is the high-ionisation-parameter corner**, and its boundary is exact:

```
max valid log10(phi)  =  min( floor(log10 n) + K , 21 )
K = 12  for log10 T in [3.5, 3.8]
K = 13  for log10 T in [3.9, 4.8]
K = 14  for log10 T in [4.9, 5.5]
```
Verified with **0 mismatches over 2 079 (n, T) pairs** across `Z1.00_age1e6`, `Z1.00_age1e7`,
`Z0.15_age1e6`. Values: `cool` 7.3e−34 … 3.4e+02, `heat` 1.5e−30 … 2.2e+02, **no zeros or
negatives** (so `np.log10` at read_cloudy:98/100 never produces `-inf`).

Units note: the cube is **volumetric, erg cm⁻³ s⁻¹** (checked: `cool = 3.737e−33` at
`n=1e−4, T=3162, φ=1` ⇒ `Λ = 3.7e−25` erg cm³ s⁻¹, physical). The docstring at
`read_cloudy.py:28` says "erg cm3 / s", which reads as Λ and is misleading.

**Age axis** (from filenames, `lib/default/opiate/`): `1.00e+06, 2.00e+06, 3.00e+06, 4.00e+06,
5.00e+06, 1.00e+07` yr — for **both** `Z1.00` and `Z0.15`. Note the 5 Myr gap between the last
two nodes. Metallicity axis: **{1.00, 0.15} only** — any other `ZCloud` raises a clear
`ValueError` (`read_cloudy.py:301`). Rotation axis: only `rot` files ship.

### B.2 CIE cooling curves — `lib/default/CIE/`

| File | `path_cooling_CIE` | N | `log10 T` min | max | `log10 Λ` min | max | strictly ↑ | `CIE_Tcutoff` = min(logT>5.5) | blend band width |
|------|--------------------|---|---------------|-----|---------------|-----|------------|-------------------------------|------------------|
| `coolingCIE_1_Cloudy.dat` | `1` | 78 | **1.00000** | **9.90000** | −30.000 | −20.725 | yes | 5.60000 | 0.1000 dex |
| `coolingCIE_2_Cloudy_grains.dat` | `2` | 77 | **1.00000** | **9.70000** | −30.000 | −16.250 | yes | 5.60000 | 0.1000 dex |
| `coolingCIE_3_Gnat-Ferland2012.dat` | `3` (**default**) | 31 | **3.99999** | **10.00001** | −22.885 | −21.326 | yes | 5.54810 | 0.0481 dex |
| `coolingCIE_4_Sutherland-Dopita1993.dat` | auto @ `ZCloud=0.15` | 34 | **3.99990** | **10.01000** | −23.310 | −21.470 | yes | 5.51000 | 0.0100 dex |

(N excludes the `# log(T), log(Lambda)` header row.) All four are strictly increasing; none has
duplicate abscissae. Note files 3 and 4 are sparse: file 3 has a **single linear segment spanning
`logT` 8.00 → 10.00** (2 dex).

### B.3 SPS / stellar feedback — `lib/default/sps/starburst99/1e6cluster_default.csv`

| Axis / column | Stored | N | Min | Max |
|---------------|--------|---|-----|-----|
| `t` | linear, **yr** (`ColumnSpec(units='yr', log=False)`) | 1000 | 1.0000e+04 | 9.9910e+07 |
| `t` after conversion + `t=0` prepend | Myr | 1001 | **0.000000** | **99.9100** |
| `Qi` | `log10` (1/s) | | 45.450 | 52.834 |
| `fi` | `log10` | | −5.834 | −0.250 |
| `Lbol` | `log10` (erg/s) | | 40.654 | 42.706 |
| `Lmech_total` | `log10` (erg/s) | | 33.301 | 40.491 |
| `pdot_W` | `log10` (g cm/s²) | | 27.125 | 32.331 |
| `Lmech_W` | `log10` (erg/s) | | 33.301 | 40.326 |

`t` is strictly increasing (min Δt = 1e5 yr); `read_sps.py:186` validates this explicitly.
`sps_refmass = 1e6 Msun`; all mass-scaled columns are multiplied by `f_mass = mCluster/1e6`.

### B.4 `f_kappa` auto-resolution grid (hard-coded, `_input/fkappa_auto.py:40–42`)

| Axis | N | Min | Max |
|------|---|-----|-----|
| `log10 mCloud_input` | 3 | 1e5 | 1e7 Msun |
| `log10 sfe` | 3 | 0.03 | 0.30 |
| `log10 nCore` | 7 | 1e2 | 1e5 cm⁻³ |

### B.5 Lane–Emden table (computed at load, `bonnorEbertSphere.py:91–93`)

`ξ ∈ [1e−7, 20.0]`, 5 000 log-spaced points. `ξ_out` is set by `densBE_Omega < 14.04`
⇒ `ξ_out ≲ 6.5`, so `f_m` is queried at `ξ ≤ ξ_out` and `f_rho_rhoc` at `ξ = ξ_out·(r/rCloud)`.

---

## C. Verified domain-face behaviour of the non-CIE `RegularGridInterpolator`

Constructed exactly as `read_cloudy.py:136` on the real `Z1.00 age 1e6` cube and probed:

| Query | Result |
|-------|--------|
| `logn = −4.0` (min face) | `−2.0872e−30` (finite) |
| `logn = −4.0001` | **`ValueError`: out of bounds in dimension 0** |
| `logn = 12.0` (max face) | `+4.4268e−03` (finite) |
| `logn = 12.0001` | **`ValueError`: dimension 0** |
| `logT = 3.5` (min face) | `+4.1949e−20` (finite) |
| `logT = 3.4999` | **`ValueError`: dimension 1** |
| `logT = 5.5` (max face — the `nonCIE_Tcutoff` query at `net_coolingcurve.py:179`) | `+2.3668e−16` (finite) |
| `logT = 5.5001` | **`ValueError`: dimension 1** |
| `logφ = 0.0` (min face) | `+1.9720e−20` (finite) |
| `logφ = −0.0001` | **`ValueError`: dimension 2** |
| `logφ = −inf` (i.e. `Qi == 0` ⇒ `log10(0)`) | **`ValueError`: dimension 2** |
| `logφ = 21.0` at `logn = 10` | `−5.5816e−03` (finite) |
| `logφ = 21.0001` | **`ValueError`: dimension 2** |
| **`logφ = 16.0` at `logn = 3, logT = 4.0` — the last *tabulated* node** | **`NaN`** |
| `logφ = 15.9` at `logn = 3, logT = 4.0` | `−3.0087e−18` (finite) |

**Key structural fact, verified on toy data and on the real cube:** `RegularGridInterpolator`
linear evaluation multiplies all 2ⁿ cell corners, and `0 * NaN = NaN`. A query landing *exactly on
a valid node* returns `NaN` if any corner of the cell that node opens is `NaN`. Consequence:
21.07 % of nodes are `NaN`, but **22.9 % of cells** yield `NaN`. The usable φ range is the *open*
interval below the last tabulated node, not the closed one.

---

## D. Reachable physics vs. the measured grids

Certainty labels: **[code]** = provable from source; **[est]** = analytic estimate (Weaver
similarity solution + bundled `Qi(t)`, `Lmech(t)`), hand to Phase 6; **[open]** = only a runtime probe settles it.

### D.1 Temperature — CLEARED

- **[code]** The non-CIE cube is queried only where `log10 T ≤ 5.5` (`net_coolingcurve.py:138`,
  and `mask = T_cond < _CIEswitch` at `bubble_luminosity.py:794, 821`) — the exact grid max.
- **[code]** The lower end is protected twice: the bubble ODE anchors at `_T_INIT_BOUNDARY = 3e4`
  (`bubble_luminosity.py:51`) and region 3 floors at `_coolingswitch = 1e4` ⇒ `logT ≥ 4.0 > 3.5`.
  `get_dudt` additionally clamps `T` up to `10**nonCIE_Tmin` (`net_coolingcurve.py:130`).
- **[code]** CIE is queried only at `T ≥ 10^5.5` ⇒ `logT ≥ 5.5`, inside every bundled file.
- **[est]** Upper CIE bound: bubble interiors peak at ~1e7–1e8 K; the *narrowest* bundled ceiling
  is file 2 at `logT = 9.70` (5e9 K), ≈ 2 dex of headroom. Low risk, but see W-4.

### D.2 Density — CLEARED with wide margin

- **[est]** `n = Pb / ((mu_convert/mu_ion)·k_B·T)` with `mu_convert/mu_ion = 2.3`. Over an
  8-config Weaver scan spanning the whole `paperII_grid_sweep` box
  (`mCloud` 1e4–5e9, `sfe` 0.01–0.9, `nCore` 50–1e5, `t` 0.01–15 Myr):
  `log10 n(T=1e4 K) ∈ [0.19, 7.55]`, `log10 n(T=10^5.5 K) ∈ [−1.31, 6.05]`.
  Grid is `[−4, 12]` ⇒ **≥ 2.7 dex of margin at both ends**.
- To hit `logn = −4` at `logT = 5.5` you need `P/k ≈ 32 K cm⁻³`; to hit `logn = 12` at
  `logT = 4` you need `P/k ≈ 1e16 K cm⁻³`. Neither is a GMC-bubble regime.

### D.3 Ionising flux φ — the axis with the least headroom; hand to Phase 6

- **[code]** `phi = Qi / (4πr²)` (`bubble_luminosity.py:427, 783, 811`).
- **[est]** Same 8-config scan, evaluated at `r = R2`: `log10 φ ∈ [3.87, 16.35]`, grid `[0, 21]`.
- **[est]** Distance to the NaN wall: `log10(φ) − log10(n) ∈ [3.22, 9.21]` at `T = 1e4 K`
  (wall at `12–13`) and `∈ [4.72, 10.71]` at `T = 10^5.5 K` (wall at `13–14`).
  **≈ 3 dex of margin**, but the margin *shrinks with radius*: `φ ∝ r⁻²` while `n ∝ Pb ∝ r⁻³`,
  so `log(φ/n)` grows ~1 dex per dex of `R2`.
- **[open]** The conduction band is sampled at `r < R2` (larger φ) and region 3 at `r > R2`; my
  estimate uses `r = R2` and a Weaver `Pb`. The real `Pb` from the `beta/delta` solver can be
  well below Weaver during a stalled/cooled phase, which *raises* `φ/n`. **Only a runtime probe
  settles whether any config crosses the wall.**
- **[est]** Low-φ face (`logφ < 0`): would need `R2 ≳ 5 kpc` even for the faintest tracked
  config. Cleared for the **bundled** SPS; a user SPS file with `Qi → 0` breaks it (`log10(0) = −inf`
  ⇒ `ValueError`, verified in §C).

### D.4 Age — **exceeded by default configs today**

- **[code]** The non-CIE age grid stops at **1e7 yr = 10 Myr**. `stop_t` default is **15 Myr**
  (`registry.py:353`). `get_filename` silently returns the 10 Myr file for any `age ≥ 1e7`,
  with **no warning at any log level**. Verified by direct call:
  `t_now = 10.0 / 12.0 / 15.0 / 30.0 / 99.0 Myr → opiate_cooling_rot_Z1.00_age1.00e+07.dat`.
- **[code]** Also clamped at the bottom: the entire first Myr uses the 1 Myr table.
- **[open]** Whether a given run is still in an energy-driven phase (i.e. still building the
  non-CIE cube) past 10 Myr is config-dependent — Phase 6.

### D.5 SPS time — CLEARED for `stop_t ≤ 99.91 Myr`

- **[code]** `t ∈ [0, 99.91] Myr` after the `t=0` prepend; `stop_t` default 15 Myr; and
  `update_feedback.py:153–159` pre-checks with a readable message. Good.
- **[code]** But `pdotdot_total` at line 185 evaluates `fpdot_total(t ± 1e-9)` *after* that check
  ⇒ **at `t = t_min = 0.0` or `t = t_max` exactly the probe steps off-grid** and scipy raises.
  `t0 = tSF + dt_phase0 > 0` (`get_InitPhaseParam.py:160`), so the low end is not hit in practice;
  the high end fires only if `stop_t ≥ 99.91`.

---

## E. Out-of-domain propagation — does the caller notice?

This is the load-bearing part, and the answer is uncomfortable.

1. **`bounds_error=True` is correct everywhere on the cooling path.** Off-grid `→ ValueError`, not
   a silent extrapolation. That is the right default and it is what the tables get.

2. **But the raise is swallowed.** Every cooling-table query inside the bubble solve runs under
   `get_betadelta.py:436–439` (and `538–549`):
   ```python
   try:
       bubble_props = get_bubbleproperties_pure(params_view)
   except Exception as e:
       logger.warning(f"Bubble properties calculation failed: {_describe_exc(e)}")
       return 100.0, 100.0, None
   ```
   A blanket `except Exception` converts an off-grid `ValueError` into a **penalty residual of
   100.0**. The β/δ solver treats it as a bad trial point and moves on. If the off-grid condition
   is *systematic* (the true root sits in an off-grid region), the solver converges elsewhere or
   reports "no physical root" and hands off to the momentum phase — a *different stopping fate*,
   with only a WARNING line as evidence. The message does include the exception class and the
   deepest frame (`_describe_exc`), so it is greppable — which is exactly what Phase 6 should do.

3. **`NaN` is worse than the raise, because nothing checks for it.** `get_dudt` returns
   `-1 * dudt * cvt.dudt_cgs2au` with no `isfinite` test (`net_coolingcurve.py:156`). A NaN
   propagates into `dTdrr` in `_get_bubble_ODE`, then into `solve_ivp`. There *are* finiteness
   guards downstream (`bubble_luminosity.py:333, 376, 482, 993`) so it is caught eventually —
   but as "solver failure", not as "your cooling table has a hole here".
   `grep -rn "isnan\|isfinite" trinity/cooling/` returns **nothing**.

4. **Clamps that nobody sees.** #6 (age) and #23 (`operations` index clamp) clamp silently. #12
   (`fkappa`) and #13–15 (Bonnor–Ebert) clamp too — but #12 warns, and #13–15 clamp to values
   that are physically correct at those endpoints. #12 is the model to copy.

---

## F. Log-vs-linear axes

**Query space: CLEARED.** Every log-stored axis is queried in log space, with the unit conversion
applied first:
- `net_coolingcurve.py:154, 179` — `[log10(ndens), log10(T), log10(phi)]` after `/= cvt.ndens_cgs2au`, `/= cvt.phi_cgs2au`.
- `bubble_luminosity.py:784–789, 823–828` — `np.log10([n_cond/cvt.ndens_cgs2au, T_cond, phi_cond/cvt.phi_cgs2au])`; `T` is already K.
- `read_coolingcurve.py:60–62` — `T = np.log10(T)`, then `10**interp(T)`.
- `fkappa_auto.py:83` — `np.log10([mCloud_input, sfe, nCore])` after `nCore * cvt.ndens_au2cgs`.
- SPS (`t`) and Bonnor–Ebert (`ξ`) are linear-stored, linear-queried.

No linear-query-against-log-grid was found anywhere.

**Value space: NOT consistent.** The same physical quantity — non-CIE net cooling — is
interpolated in two *different* spaces by two *different* production call paths built from the
same cube in the same function:

| Path | Built at | Interpolated quantity | Consumer |
|------|----------|-----------------------|----------|
| A | `read_cloudy.py:136` | **linear** `cool − heat` | `get_dudt` → the bubble-ODE source term (the hot loop) |
| B | `read_cloudy.py:98,100` | **log10** `cool`, **log10** `heat`, subtracted after `10**` | `L_conduction`, `L_intermediate` in `_bubble_luminosity` |

Measured over 20 000 random mid-cell points in the reachable box
(`logn ∈ [2,7]`, `logT ∈ [4.0, 5.49]`, `logφ ∈ [8,16]`), ratio A/B:

| p1 | p5 | p25 | **p50** | p75 | p95 | p99 | min | max |
|----|----|-----|---------|-----|-----|-----|-----|-----|
| 1.006 | 1.073 | 1.335 | **1.627** | 1.829 | 2.229 | 3.063 | −35.8 | +57.3 |

**93.5 %** of the box differs by > 10 %; **8.7 %** by more than a factor 2; **0.2 %** disagree in
*sign*. Worst sample: `logn=6.021, logT=4.041, logφ=12.492` → A `+1.341e−12` vs B `+2.341e−14`
(**57×**). Path A linearly interpolates a quantity spanning 35 decades on a 0.5-dex/1-dex grid —
that is the larger error of the two, and it drives the ODE. (Path A cannot simply be logged: net
cooling changes sign. The right fix is not obvious; the *inconsistency* is the finding.)

---

## G. Endpoints, monotonicity, ordering

- **Sortedness — CLEARED.** All non-CIE axes strictly ascending (verified `np.all(np.diff>0)`).
  All four CIE tables strictly ascending, min Δ`logT` = 9e−3 (files 1/2), 4.8e−2 (file 3),
  5.0e−2 (file 4). SPS `t` strictly ascending, min Δt = 1e5 yr, validated at load
  (`read_sps.py:186` → `sps_columns.validate_t_monotonic`). No descending-where-ascending-assumed
  cases found. `interp1d` is left at `assume_sorted=False` throughout, so it sorts defensively.
- **Duplicate abscissae — CLEARED.** `create_limits` rounds `log10` to 3 decimals and indexes by
  exact float equality (`read_cloudy.py:233–235`). Verified: for all bundled tables,
  `#rows (12 034) == #filled cells`, i.e. no two distinct raw values collapse onto one index. Two
  raw values that rounded identically would silently overwrite each other — a hazard for
  **user-supplied** tables only.
- **`np.interp` at `net_coolingcurve.py:194` — CLEARED.** `nonCIE_Tcutoff = max(temp[temp≤5.5]) ≤ 5.5`
  and `CIE_Tcutoff = min(logT[logT>5.5]) > 5.5`, so the 2-node reference is *always* ascending.
- **Cached-cube staleness — CLEARED for the bundle.** All 8 `_cube.npy` reproduce bit-for-bit
  from their `.dat`. Note `create_cubes` **writes** new `_cube.npy` into `path2cooling`
  (i.e. into `lib/default/opiate/`) at runtime; a read-only install or a stale cube from an edited
  `.dat` would be a problem (`read_cloudy.py:174–176, 265`).
- **SPS cubic ringing.** `kind='cubic'` on a series with long runs of exact zeros: the bundled
  `Lmech_total − Lmech_W` is 0 at 426 of 1000 nodes and **negative at 47** (so the
  `logger.warning("Negative SN mechanical luminosity … clamping to zero")` at `read_sps.py:204`
  fires on *every default run*). After clamping, the cubic interpolant still rings:
  `fLmech_SN` reaches **−5.27e38 erg/s** and is negative on **23.7 %** of `[0, 99.91] Myr`
  (**12.0 %** of `[0, 15] Myr`). `fLmech_total` / `fpdot_total` stay positive for `t ≤ 15 Myr`
  (min `+8.80e39`, `+1.81e31`) but go negative beyond ~40 Myr, taking
  `v_mech_total = 2L/ṗ` negative with them. `fQi` and `fpdot_W` never go non-positive.
  `Lmech_SN`/`pdot_SN` feed only the reported `F_ram_SN` diagnostic (`grep F_ram_SN` — no ODE
  consumer), so within the default `stop_t` this is an output-fidelity problem, not a physics one.
- **`t_now` staleness at cube rebuild.** `get_coolingStructure(params)` reads `params['t_now']`
  at `run_energy_implicit_phase.py:784`, but `params['t_now'].value = t_now` is only set at
  line 793 — the cube age lags by up to one segment (`DT_SEGMENT_MAX = 5e-2 Myr`). Against a
  1 Myr age grid this is ≤ 5 % of one interval. Real, tiny.
- **Dead data.** Five `opiate_cooling_rot_Z1.00_age*.npy` files (no `_cube` suffix, 122 kB each,
  shape `(33,21,22)` — precomputed net-cooling cubes) are read by nothing:
  `grep -rn "\.npy" trinity/ tools/` matches only the `_cube.npy` path at `read_cloudy.py:173`.

---

## H. Clearances (verified correct — these are results)

| Clearance | Evidence |
|-----------|----------|
| `bounds_error=True` on every cooling-table interpolator (CIE `interp1d`, all three non-CIE `RGI`s) | left at scipy defaults; probed on the real arrays — raises `ValueError`, never extrapolates |
| **Temperature axis fully inside both tables** on both call paths | `_T_INIT_BOUNDARY=3e4`, `_coolingswitch=1e4`, `mask = T < 10**5.5`, and the `net_coolingcurve.py:130` floor gate (already pinned by `test/test_net_coolingcurve.py`) |
| **Density axis has ≥ 2.7 dex margin** at both ends | measured grid `[−4, 12]` vs. estimated reachable `[−1.31, 7.55]` |
| **Every log-stored axis is queried in log space, with the cgs↔AU conversion applied first** | §F; all 5 call sites checked |
| **SPS time domain guarded with a readable pre-check** | `update_feedback.py:153–159`; `stop_t` default 15 ≪ `t_max` 99.91 Myr |
| **`fkappa_auto` clamps to its hull AND logs a WARNING** — the model implementation | `fkappa_auto.py:81–93` |
| **Bonnor–Ebert interpolators clamp to physically correct endpoints** | `fill_value=(1.0, ρ[-1])`, `(0.0, m[-1])`, `(ξ[-1], ξ[0])`; and `ξ_out ≤ 6.5 ≪ ξ_max = 20`, so `f_m` is never off-grid |
| **All table axes strictly ascending; no duplicate abscissae in the bundle** | `np.all(np.diff>0)` on every axis; rows == filled cells for all 3 cubes checked |
| **Cached `_cube.npy` reproduce their `.dat` bit-for-bit** | rebuilt from `.dat` and compared (axes + cube) |
| **No zeros/negatives in `cool`/`heat`** ⇒ `np.log10` at read_cloudy:98/100 never yields `−inf` | min `cool` 7.3e−34, min `heat` 1.5e−30 across all cubes |
| **`np.interp` blend at `net_coolingcurve.py:194` always has an ascending 2-node reference** | `nonCIE_Tcutoff ≤ 5.5 < CIE_Tcutoff` is structural |
| **`ZCloud` outside {1.0, 0.15} raises a clear, actionable error** | `read_cloudy.py:301–305` |
| **`simplify.py` and `check_yesno.py` `np.interp` clamps are provable no-ops** | explicit ascending sort + endpoint retention; `t_grid` restricted to the overlap window |
| **`heat` NaN mask == `cool` NaN mask** ⇒ `netcooling` has no *extra* holes | `np.array_equal(np.isnan(cool), np.isnan(heat))` → True |

---

## I. Phase-6 worklist (named runtime probes)

Instrument every table query and log `(caller, n, T, φ, age, t_now, R2, result)` for any request
that is off-grid, NaN, or clamped. Specifically:

- **W-1 · φ/n NaN-wall crossings.** Log every non-CIE query with
  `log10 φ ≥ floor(log10 n) + K − 0.5` (K per §B.1) and every query returning `NaN`. Static
  analysis gives ~3 dex of margin at `r = R2` under a *Weaver* `Pb`; the real solver `Pb` during a
  stalled/heavily-cooled phase is the unknown. Run the audit-standard edge set:
  `param/simple_cluster.param` + `docs/dev/performance/f1edge_{lowdens,hidens}*.param`, plus the
  extreme corners of `param/paperII_grid_sweep.param` (`mCloud=1e4, sfe=0.9, nCore=1e5` for the
  high-φ end; `mCloud=5e9, sfe=0.01, nCore=50` for the largest `R2`, where `φ/n` grows fastest).
- **W-2 · Age clamp occupancy.** Count segments run with `t_now > 10 Myr` (default `stop_t = 15`)
  and record which phase they are in. If the energy phase is still live there, every cooling
  lookup in that window is using a 10-Myr-old radiation field. Also count segments in the
  5→10 Myr gap, where the cube is a single linear interpolation across 5 Myr.
- **W-3 · Grep the WARNING stream for swallowed bounds errors.**
  `grep "Bubble properties calculation failed" *.log | grep -i "out of bounds\|interpolation range"`.
  This is the cheapest possible probe and it directly answers "does an off-grid request ever
  reach production today?" — because `get_betadelta.py:438` already logs the exception class and
  frame. Do this **first**.
- **W-4 · Peak bubble `T` vs. the CIE ceiling.** Log `max(T_array)` per segment. Cleared by ~2 dex
  under the default file 3 (`logT_max = 10.00001`); the exposure is a user selecting
  `path_cooling_CIE 2` (`logT_max = 9.70`). Also record whether `T_array` ever *fails* to reach
  `10^5.5` (then `index_CIE_switch` clamps to `len-1` and the CIE curve is applied below its
  intended regime — no raise, wrong physics).
- **W-5 · `fT_interp_interm(bubble_r_Tb)` domain.** Log `bubble_r_Tb` vs.
  `[r_array[index_cooling_switch], R2_coolingswitch]` at `bubble_luminosity.py:875`. Statically I
  could not prove `bubble_xi_Tb·R2 ≤ R2_coolingswitch`.
- **W-6 · A vs. B net-cooling divergence in situ.** At each `get_dudt` call also evaluate path B
  (`10**cool_interp − 10**heat_interp`) and log the ratio. §F says median 1.63× on a synthetic
  scan; the question is what it is on the trajectory the solver actually walks, and whether
  `L_conduction` and the ODE source disagree enough to matter for `Θ = Lloss/Lgain`.
- **W-7 · `logφ` floor for faint clusters.** Log `min(log10 φ)` per run for the faintest tracked
  config (`mCloud=1e4, sfe=0.01`). Estimated min 3.87 vs. grid floor 0.0 — confirm, and confirm
  `Qi` never reaches 0.

---

```json
[
  {
    "id": "TBL-01",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 325,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The non-CIE cooling cube is tabulated only for cluster ages 1e6-1e7 yr, but get_filename silently clamps any age >= 1e7 yr to the 1e7 file with no warning at any log level. The default stop_t is 15 Myr, so tracked configs routinely run 50% past the tabulated age range while the cooling/heating rates are frozen at their 10 Myr values.",
    "evidence": "Age grid measured from lib/default/opiate/ filenames: 1.00e+06, 2.00e+06, 3.00e+06, 4.00e+06, 5.00e+06, 1.00e+07 yr (both Z1.00 and Z0.15). Direct call to get_filename: t_now = 10.0 / 12.0 / 15.0 / 30.0 / 99.0 Myr all return opiate_cooling_rot_Z1.00_age1.00e+07.dat. registry.py:353 sets stop_t default '15'. The elif branches at read_cloudy.py:325 and :330 return the endpoint file with no logger call and no print.",
    "expected": "Either a warning when age falls outside [min(age_list), max(age_list)], or an explicit documented decision that clamping is the intended extrapolation. A user cannot currently tell from the logs that 5 of 15 Myr used a frozen cooling table.",
    "failure_scenario": "A low-density / high-SFE cloud that is still energy-driven at t > 10 Myr uses 10-Myr-old CLOUDY heating/cooling for the rest of the run. The ionising field has dropped by ~1 dex between 10 and 15 Myr (Qi 10^49.49 -> 10^47.97 for mCloud=1e7,sfe=0.1), so the photo-heating term is overestimated and net cooling underestimated, biasing the energy->momentum transition time.",
    "repro": "python -c \"from trinity.cooling.non_CIE.read_cloudy import get_filename; print(get_filename(15e6, 1.0, True, 'lib/default/opiate/'))\"  # -> single 1.00e+07 file, no warning",
    "confidence": "high"
  },
  {
    "id": "TBL-02",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 136,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "21.07% of the non-CIE cooling cube is NaN (the high-ionisation-parameter corner where CLOUDY has no solution). RegularGridInterpolator's linear evaluation multiplies all 8 cell corners, and 0*NaN = NaN, so 22.9% of cells return NaN - including queries that land exactly on a valid grid node adjacent to a hole. get_dudt returns that NaN unchecked; there is no isnan/isfinite guard anywhere in trinity/cooling/.",
    "evidence": "Measured on lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06_cube.npy: 3212 NaN of 15246 cells (21.07%); corner-OR over cells gives 3080 of 13440 contaminated (22.9%). Exact wall, 0 mismatches over 2079 (n,T) pairs across three cubes: max valid log10(phi) = min(floor(log10 n) + K, 21) with K=12 for logT in [3.5,3.8], 13 for [3.9,4.8], 14 for [4.9,5.5]. Probe on the real cube: f([3.0, 4.0, 16.0]) -> NaN even though (logn=3, logT=4.0, logphi=16.0) is a filled node; f([3.0, 4.0, 15.9]) -> -3.0087e-18. Toy reproduction: V=np.ones((3,3,3)); V[2,2,2]=nan; RGI((x,y,z),V)([[1.0,1.0,1.0]]) -> nan. grep -rn 'isnan|isfinite' trinity/cooling/ returns nothing.",
    "expected": "Either a NaN check at the get_dudt return (net_coolingcurve.py:156) that raises a named, greppable error naming the offending (n,T,phi), or a table-hole fill/nearest-valid fallback. Silently returning NaN as a cooling rate is the failure mode this sweep exists to find.",
    "failure_scenario": "A conduction-zone query in the high-phi/low-n corner returns NaN -> dudt NaN -> dTdrr NaN in _get_bubble_ODE -> solve_ivp produces a NaN profile -> caught downstream (bubble_luminosity.py:333/376/482) and reported as a generic 'solver failure', so the operator never learns that the cooling table has a hole at that (n,T,phi). The beta-delta solver then converges elsewhere or hands off to momentum with a different stopping fate.",
    "repro": "python -c \"import numpy as np; from scipy.interpolate import RegularGridInterpolator as R; n,t,p,c,h=np.load('lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06_cube.npy',allow_pickle=True); f=R((n,t,p),c-h); print(f([[3.0,4.0,16.0]]), f([[3.0,4.0,15.9]]))\"",
    "confidence": "high"
  },
  {
    "id": "TBL-03",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 437,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "bounds_error=True is correctly left on every cooling-table interpolator, but the resulting ValueError is caught by a blanket `except Exception` and converted into a penalty residual of 100.0. An off-grid table request therefore does not abort the run - it silently becomes a rejected trial point for the beta-delta solver.",
    "evidence": "get_betadelta.py:436-439 `try: bubble_props = get_bubbleproperties_pure(params_view) / except Exception as e: logger.warning(f\"Bubble properties calculation failed: {_describe_exc(e)}\") / return 100.0, 100.0, None`; the same pattern at :538-549 returns ResidualDetails(Edot_residual=100.0, T_residual=100.0, ...). Verified off-grid behaviour of the interpolators involved: RegularGridInterpolator raises ValueError('One of the requested xi is out of bounds in dimension k'), interp1d raises ValueError('A value (x) in x_new is above/below the interpolation range...').",
    "expected": "Off-grid table requests should be distinguishable from genuine solver non-convergence - e.g. a dedicated exception type re-raised or counted separately, so the run report can say 'N trials rejected because the cooling table domain was exceeded' instead of burying it in a generic warning.",
    "failure_scenario": "If the physical root of the beta-delta residual sits in a region whose cooling lookup is off-grid, every trial there is scored 100.0 and the solver converges to a different (in-domain) point, or reports no physical root and hands off to the momentum phase early. The stopping fate changes; the only trace is a WARNING line.",
    "repro": "Run any tracked config and grep the log: grep 'Bubble properties calculation failed' *.log | grep -iE 'out of bounds|interpolation range'  -- this is Phase-6 probe W-3.",
    "confidence": "high"
  },
  {
    "id": "TBL-04",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 98,
    "class": "numerical",
    "severity": "S2",
    "claim": "The same physical quantity - non-CIE net cooling - is interpolated in two different value spaces by two production paths built from the same cube in the same function. read_cloudy.py:136 interpolates the LINEAR difference (cool-heat) and feeds the bubble-ODE source term via get_dudt; read_cloudy.py:98/100 interpolate log10(cool) and log10(heat) and feed L_conduction / L_intermediate. Median disagreement 1.63x, up to 57x, with 0.2% sign flips.",
    "evidence": "20000 random mid-cell samples in the reachable box (logn 2-7, logT 4.0-5.49, logphi 8-16), ratio A/B: p1 1.006, p25 1.335, p50 1.627, p75 1.829, p95 2.229, p99 3.063, min -35.8, max +57.3. 93.5% of samples differ by >10%, 8.7% by >2x, 0.2% differ in sign. Worst sample logn=6.021 logT=4.041 logphi=12.492: A=+1.3408e-12 vs B=+2.3405e-14. Root cause: net cooling spans 35 decades (2.1e-33 to 3.4e+02) and path A interpolates it linearly on a 0.5-dex/1-dex grid.",
    "expected": "One interpolation convention for one physical quantity. Path A cannot simply be logged (net cooling changes sign), but the two paths should at minimum be documented as inconsistent, and the beta-delta residual should not mix a linearly-interpolated source term with a log-interpolated luminosity.",
    "failure_scenario": "The cooling_balance transition trigger compares Lloss (path B) against Lgain, while the bubble temperature profile that produced it was integrated with a path-A source term that is typically 1.6x larger. Theta = Lloss/Lgain therefore fires at a systematically different time than the ODE's own energy budget implies.",
    "repro": "python -c \"import numpy as np; from scipy.interpolate import RegularGridInterpolator as R; n,t,p,c,h=np.load('lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06_cube.npy',allow_pickle=True); A=R((n,t,p),c-h); C=R((n,t,p),np.log10(c)); H=R((n,t,p),np.log10(h)); q=[[5.0,4.5,13.3]]; print(A(q), 10**C(q)-10**H(q))\"",
    "confidence": "high"
  },
  {
    "id": "TBL-05",
    "file": "trinity/sps/update_feedback.py",
    "line": 185,
    "class": "state",
    "severity": "S2",
    "claim": "get_current_sps_feedback validates t against [t_min, t_max] with an inclusive check at line 156, then at line 185 evaluates fpdot_total(t - 1e-9) and fpdot_total(t + 1e-9) for the pdotdot_total central difference. At t == t_min (0.0 Myr) or t == t_max (99.91 Myr) exactly, that probe steps outside the interpolator domain and interp1d raises a cryptic ValueError that the guard was written to prevent.",
    "evidence": "update_feedback.py:156 `if not (t_min <= t <= t_max): raise ValueError(...)` - inclusive on both ends. Line 184-185 `dt = 1e-9` then `(sps_f['fpdot_total'](t + dt)[()] - sps_f['fpdot_total'](t - dt)[()]) / (2.0 * dt)`. read_sps.py:352-354 constructs those interpolators with default bounds_error (True). Measured SPS domain after the t=0 prepend (read_sps.py:263-274): [0.000000, 99.9100] Myr.",
    "expected": "A one-sided difference at the endpoints, or clamping the probe points into [t_min, t_max], so the endpoints of the declared valid domain are actually usable.",
    "failure_scenario": "t = 0.0 is not reachable today because t0 = tSF + dt_phase0 > 0 (get_InitPhaseParam.py:160). The reachable case is stop_t set to the table end: a run configured with stop_t = 99.91 (or any user SPS file whose t_max coincides with a segment boundary) dies at the final step with 'A value in x_new is above the interpolation range's maximum value', which the blanket handlers then report as a solver failure.",
    "repro": "python -c \"import numpy as np,scipy.interpolate as si; f=si.interp1d(np.array([0.,1.,2.]),np.array([1.,2.,3.]),kind='cubic'); f(2.0+1e-9)\"  # ValueError",
    "confidence": "high"
  },
  {
    "id": "TBL-06",
    "file": "trinity/sps/read_sps.py",
    "line": 341,
    "class": "numerical",
    "severity": "S3",
    "claim": "The SPS interpolators are cubic (ftype='cubic') over a series with long runs of exact zeros. The bundled Lmech_SN (= Lmech_total - Lmech_W, clamped at >= 0) is zero at 426 of 1000 nodes with isolated positive spikes, so the cubic spline rings: fLmech_SN reaches -5.27e38 erg/s and is negative on 23.7% of [0, 99.91] Myr and 12.0% of [0, 15] Myr. fLmech_total and fpdot_total stay positive within the default stop_t=15 Myr but go negative beyond ~40 Myr, taking v_mech_total = 2L/pdot negative with them.",
    "evidence": "Measured on lib/default/sps/starburst99/1e6cluster_default.csv with the read_sps pipeline (FB defaults: thermCoeff=1, mColdFrac=0, vSN=1e4 km/s), 400001-point scan: fLmech_SN min -5.2656e+38 (node min 0.0), negfrac 23.66% full domain / 11.97% within 15 Myr; fpdot_SN min -1.0531e+30, negfrac 23.66%; fLmech_total min -4.8915e+38 at t=44.95 Myr (node min +2.0e+33), negfrac 0.397%; fpdot_total min -9.6016e+29, negfrac 0.179%; v_mech_total cubic min -2.30e+11 cm/s. fQi and fpdot_W never go non-positive (min 2.82e+45 and 1.33e+27). Separately, Lmech_total - Lmech_W is negative at 47 nodes in the bundled file, so the logger.warning at read_sps.py:204 fires on every default run.",
    "expected": "A shape-preserving interpolant (PCHIP) or linear for quantities that are physically non-negative and have flat-zero runs, or a post-interpolation clamp at >= 0. A cubic spline through a step function is guaranteed to undershoot.",
    "failure_scenario": "Within the default stop_t, Lmech_SN and pdot_SN feed only the reported F_ram_SN diagnostic (no ODE consumer - verified by grep), so published SN-feedback force columns carry physically impossible negative values on ~12% of the timeline. Raising stop_t past ~40 Myr additionally makes Lmech_total and v_mech_total negative, which does enter the physics.",
    "repro": "python -c \"import numpy as np,scipy.interpolate as si; d=np.genfromtxt('lib/default/sps/starburst99/1e6cluster_default.csv',delimiter=',',names=True); t=np.insert(d['t']/1e6,0,0.); y=np.maximum(10**d['Lmech_total']-10**d['Lmech_W'],0); y=np.insert(y,0,y[0]); f=si.interp1d(t,y,kind='cubic'); q=np.linspace(0,15,200001); print(f(q).min())\"",
    "confidence": "high"
  },
  {
    "id": "TBL-07",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 875,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "fT_interp_interm is a 2-node interp1d over [r_array[index_cooling_switch], R2_coolingswitch] with bounds_error at its default (True). At line 875 it is evaluated at bubble_r_Tb under the guard `bubble_r_Tb > r_array[index_cooling_switch]`, which establishes only the LOWER bound. Nothing proves bubble_r_Tb <= R2_coolingswitch.",
    "evidence": "bubble_luminosity.py:803-807 constructs interp1d(np.array([r_array[index_cooling_switch], R2_coolingswitch]), np.array([T_array[index_cooling_switch], _coolingswitch]), kind='linear') with no bounds_error/fill_value. Line 873 `if bubble_r_Tb > r_array[index_cooling_switch]:` then line 875 `T_rgoal = fT_interp_interm(bubble_r_Tb)`. bubble_r_Tb = bubble_xi_Tb * R2 with bubble_xi_Tb default 0.98 (registry.py:408); R2_coolingswitch = (1e4 - T_array[idx])/dTdR_coolingswitch + r_array[idx] is a linear extrapolation whose magnitude depends on the local dT/dr, so it is not bounded below by 0.98*R2 by construction.",
    "expected": "The upper bound should be asserted or the interpolator given an explicit fill_value, so that a shallow temperature gradient at the cooling switch cannot turn a diagnostic (T at r = xi_Tb*R2) into a solver-aborting ValueError.",
    "failure_scenario": "A shallow dTdR_coolingswitch places R2_coolingswitch only marginally above r2Prime while bubble_r_Tb = 0.98*R2 sits above it; interp1d raises, the blanket handler in get_betadelta scores the trial 100.0, and the beta-delta solve is steered away from a physically valid point.",
    "repro": "Phase-6 probe W-5: log bubble_r_Tb against (r_array[index_cooling_switch], R2_coolingswitch) at bubble_luminosity.py:875 for param/simple_cluster.param and the f1edge configs.",
    "confidence": "medium"
  },
  {
    "id": "TBL-08",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 28,
    "class": "units",
    "severity": "S3",
    "claim": "The get_coolingStructure docstring states 'Cooling rate is in units of [erg cm3 / s]', which reads as the cooling function Lambda. The bundled cubes are volumetric emissivity in erg cm^-3 s^-1 (they already include n^2), which is how every caller uses them.",
    "evidence": "Table row 1 of opiate_cooling_rot_Z1.00_age1.00e+06.dat: ndens=1.0e-4, temp=3162.28, phi=1, cool=3.7372e-33. If that were Lambda it would be 3.7e-33 erg cm^3/s at 3162 K, ~8 orders too small; read as n^2*Lambda it gives Lambda = 3.7e-25 erg cm^3/s, physical. Confirmed by usage: net_coolingcurve.py:154-156 returns the interpolated value directly as dudt (times cvt.dudt_cgs2au) with NO n^2 factor, whereas the CIE branch at :164 explicitly forms chi_e * ndens**2 * Lambda_CIE. Cube max 3.425e+02 = n^2*Lambda at n=1e12.",
    "expected": "Docstring should read 'erg cm^-3 s^-1 (volumetric; already includes n^2)' to match the code and to distinguish it from the CIE Lambda, which really is erg cm^3 s^-1.",
    "failure_scenario": "A future edit that 'fixes' the missing n^2 in the non-CIE branch to match the docstring would multiply the cooling rate by up to 24 orders of magnitude.",
    "repro": "head -2 lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06.dat  and compare net_coolingcurve.py:154 (no n^2) with :164 (explicit n^2).",
    "confidence": "high"
  },
  {
    "id": "TBL-09",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 784,
    "class": "state",
    "severity": "S4",
    "claim": "get_coolingStructure(params) reads params['t_now'] to pick the cooling-table age, but params['t_now'].value = t_now is only assigned nine lines later. The cube is therefore built for the PREVIOUS segment's time.",
    "evidence": "run_energy_implicit_phase.py:783-788 calls non_CIE.get_coolingStructure(params) inside `if abs(params['t_previousCoolingUpdate'].value - t_now) > COOLING_UPDATE_INTERVAL:`; line 793 is `params['t_now'].value = t_now`. read_cloudy.py:48 is `age = params['t_now'] * 1e6`. Segment length is bounded by DT_SEGMENT_MAX = 5e-2 Myr (line 114).",
    "expected": "Set params['t_now'].value = t_now before the cooling-structure update, or pass t_now explicitly to get_coolingStructure.",
    "failure_scenario": "The age used for the cube lags by up to 0.05 Myr against a 1 Myr age grid - at most 5% of one interpolation interval. Real but small; it matters mainly as a correctness trap if the age grid is ever refined.",
    "repro": "Read run_energy_implicit_phase.py:783-793; the same pattern is correct in phase1_energy/run_energy_phase.py:124-130 because t_now is already current there.",
    "confidence": "high"
  },
  {
    "id": "TBL-10",
    "file": "lib/default/opiate",
    "line": 0,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Five bare .npy files in lib/default/opiate/ (opiate_cooling_rot_Z1.00_age{1,2,3,4,5}.00e+06.npy, 122 kB each) are read by no code. create_cubes only ever looks for '<stem>_cube.npy'.",
    "evidence": "grep -rn '\\.npy' trinity/ tools/ --include=*.py returns exactly one match: read_cloudy.py:173 `cube_filename = path2cooling + _stem + '_cube.npy'`. The bare .npy files load as a single (33,21,22) float64 array (a precomputed net-cooling cube), not the 5-element [axes..., cool, heat] list that create_cubes writes and reads.",
    "expected": "Either wire them in or move them to docs/dev/to-be-removed/ per the project rule on file removal. Flagging only - pre-existing, not created by this sweep.",
    "failure_scenario": "None at runtime. They are ~600 kB of repo weight that reads like live cache and could mislead a future session into thinking the netcooling cube is precomputed.",
    "repro": "python -c \"import numpy as np; a=np.load('lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06.npy',allow_pickle=True); print(a.shape, a.dtype)\"  # (33,21,22) float64, not the 5-element list",
    "confidence": "high"
  },
  {
    "id": "TBL-11",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 265,
    "class": "other",
    "severity": "S4",
    "claim": "create_cubes writes the derived _cube.npy back into path2cooling, i.e. into the packaged lib/default/opiate/ directory, at first run. The cache key is the filename only - it does not record the .dat mtime or hash - so an edited .dat is silently ignored in favour of a stale cube. Also, only 8 of the 13 bundled .dat files ship with a cube, so a ZCloud=0.15 run writes new files into lib/default/ at runtime.",
    "evidence": "read_cloudy.py:172-176 loads path2cooling + stem + '_cube.npy' if os.path.exists, with no staleness check; line 265 np.save(cube_filename, ...). ls lib/default/opiate/: 13 .dat but only 8 *_cube.npy (Z0.15 has a cube for age1.00e+06 only). Verified the 8 shipped cubes DO currently reproduce their .dat bit-for-bit (axes and cube compared), so this is a latent hazard rather than a present error.",
    "expected": "Write the cache to a user-writable location, or key it on the .dat mtime/hash. Writing derived data into the installed package directory also breaks read-only installs.",
    "failure_scenario": "A user edits or replaces an opiate .dat to test a different cooling table; the stale _cube.npy is loaded instead and the run silently uses the old table. Or the package is installed read-only and a ZCloud=0.15 run fails on np.save.",
    "repro": "Verified reproduction of the shipped cubes: rebuild from .dat via the create_cubes algorithm and compare - 0 differences for Z1.00_age1e6, Z1.00_age5e6, Z0.15_age1e6.",
    "confidence": "high"
  },
  {
    "id": "TBL-12",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 64,
    "class": "regime",
    "severity": "S4",
    "claim": "The non-CIE/CIE switch temperature is hard-coded as _CIEswitch = 10**5.5 (line 693) and _T_INTERFACE_BAND = 10**5.5 (line 64) in bubble_luminosity, but derived from the table in net_coolingcurve (nonCIE_Tcutoff = max(temp[temp <= 5.5])). They coincide only because every bundled cube happens to top out at exactly logT = 5.5. A user-supplied cube with a lower ceiling makes bubble_luminosity query the cube above its own maximum -> ValueError.",
    "evidence": "Measured: every bundled cube has log_temp max = 5.500 exactly, so nonCIE_Tcutoff = 5.5 = log10(_CIEswitch). bubble_luminosity.py:794 and :821 mask on `T < _CIEswitch` (the hard-coded 10**5.5) before calling cooling_nonCIE.interp, with no reference to the cube's actual .temp max. The coupling is already documented in the comment at bubble_luminosity.py:60-65 ('they coincide on the default bundle; a table swap moves the third'), and pinned by test/test_fA_source_boost.py.",
    "expected": "Derive _CIEswitch from cooling_nonCIE.temp.max() rather than hard-coding it, so a table swap cannot desynchronise the mask from the grid.",
    "failure_scenario": "A user supplies an opiate cube tabulated only to logT = 5.0. bubble_luminosity masks at 10**5.5, queries the cube at logT in (5.0, 5.5), and RegularGridInterpolator raises 'out of bounds in dimension 1'; the blanket handler turns it into a penalty residual.",
    "repro": "python -c \"import numpy as np; n,t,p,c,h=np.load('lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06_cube.npy',allow_pickle=True); print(t.max())\"  # 5.5, equal to log10(_CIEswitch) by coincidence of the bundle",
    "confidence": "medium"
  }
]
```
