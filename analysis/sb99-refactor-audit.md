# SB99 → generic SPS refactor audit

Living document. Updated as the audit goes deeper. End goal: replace the
hardcoded SB99 filename construction with a single `sps_path` (and friends) so
arbitrary stellar-population-synthesis CSVs can be dropped in.

## 1. Current architecture (one paragraph)

`main.start_expansion()` calls `read_SB99.read_SB99(f_mass, params)` which (a)
builds a filename from `SB99_mass / SB99_rotation / ZCloud / SB99_BHCUT`, (b)
loads a fixed 7-column text file, (c) converts log/cgs→AU, and (d) returns 11
numpy arrays. Those go through `get_interpolation()` which produces a dict of 10
scipy `interp1d` callables stored as `params['SB99f'].value`. Every physics
phase imports `get_currentSB99feedback(t, params)` from
`src/sb99/update_feedback.py`, which evaluates those interpolants at `t` and
returns an `SB99Feedback` dataclass while also writing each value back into
`params[…]` (so the rest of the code can read either path).

## 2. Configuration surface

| Param | File:line | Default | Role |
|-------|-----------|---------|------|
| `path_sps` | `src/_input/default.param:306` | `lib/sps/starburst99/` | Directory for SPS data. |
| `SB99_mass` | `default.param:169` | `1e6` | Reference cluster mass; sets `f_mass = mCluster / SB99_mass`. |
| `SB99_rotation` | `default.param:172` | `1` | Becomes `rot` / `norot` in filename. |
| `SB99_BHCUT` | `default.param:176` | `120` | Becomes `BH120` / `BH40` in filename. |
| `ZCloud` | (elsewhere) | — | Becomes `Z0014` / `Z0002` in filename. |
| `FB_mColdWindFrac`, `FB_mColdSNFrac`, `FB_thermCoeffWind`, `FB_thermCoeffSN`, `FB_vSN` | `default.param:178-192` | — | Post-load corrections; orthogonal to file selection. |

The filename comes out as `{1e6cluster}_{rot|norot}_{Z0014|Z0002}_{BH120|BH40}.txt`
(`read_SB99.py:368`).

## 3. Interface every consumer expects

After loading + interpolation, downstream code needs these 10 interpolants on
`params['SB99f'].value` (all in AU units = Msun·pc²/Myr³ except `Qi` in 1/Myr):

```
fQi, fLi, fLn, fLbol,
fLmech_W, fLmech_SN, fLmech_total,
fpdot_W, fpdot_SN, fpdot_total
```

…plus `params['SB99_data'].value` holds the 11-array raw cube (`[t, Qi, Li, Ln,
Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]`).

`get_currentSB99feedback()` then writes these scalars into `params`:
`Lmech_W, Lmech_SN, Lmech_total, v_mech_total, pdot_W, pdot_SN, pdot_total,
pdotdot_total, Qi, Lbol, Ln, Li` (read_param.py:472-486).

## 4. SB99-specific hot spots (what blocks generic CSVs)

| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 1 | `read_SB99.py:285-372` (`get_filename`) | Hardcoded filename grammar; only 2 Z, 2 BH, 2 rotation modes. | Replace with `sps_path` param; delete this function. |
| 2 | `read_SB99.py:144-149` | Validates `>=7` columns by position. | Header-driven column map with SB99 preset as legacy fallback. |
| 3 | `read_SB99.py:158-177` | Hardcoded log-space + cgs assumptions per column. | Per-column unit metadata. |
| 4 | `read_SB99.py:207-245` | Derives `Lmech_SN` and `pdot_SN` from totals + `FB_vSN`. | Allow direct SN columns when present; fall back to derivation. |
| 5 | `read_SB99.py:191` | Ionizing fraction at 13.6 eV hardcoded. | Either require Li/Ln in file, or expose threshold. |
| 6 | `main.py:142` | `f_mass = mCluster / SB99_mass` couples to reference mass. | Add `sps_refmass` param (or read from header). |
| 7 | `update_feedback.py:187` | Hardcoded `Δt=1e-9 Myr` for `pdotdot_total`. | Constant is fine; flag for review only. |

## 5. Refactor phasing (proposed)

1. Add `sps_path` param; if set, skip filename construction. Keep `SB99_*`
   params as a fallback that *constructs* `sps_path` via existing logic. No
   behavior change.
2. Refactor `read_SB99.py` → `read_sps.py` with a column-mapping layer. SB99's
   7-column legacy format becomes one preset.
3. Allow direct SN columns (decouple from `FB_vSN` when not needed).
4. Rename `SB99f` / `SB99_data` → `sps_f` / `sps_data` (touches every consumer;
   do last).
5. Deprecate `SB99_mass/rotation/BHCUT`; remove `get_filename()`.

## 6. Consumer-by-consumer deep dive

For each consumer the audit captures: **(a)** what it actually reads from SB99,
**(b)** whether it touches filename/grammar-specific code, **(c)** what
breaks/changes when SB99 is swapped for a user-supplied CSV behind
`sps_path`. Verified by reading the files (not just grep).

Status: 🔴 needs change · 🟡 rename only · 🟢 transparent.

### Architectural takeaway (the single most important point)

After `start_expansion()` populates `params['SB99f']` and `params['SB99_data']`,
**every downstream consumer goes through exactly one function**:
`get_currentSB99feedback(t, params)` (`src/sb99/update_feedback.py:98`). That
function reads the 10 interpolators from `params['SB99f']`, evaluates them at
`t`, and returns the `SB99Feedback` dataclass; callers then either read fields
off the dataclass or write everything back into `params` via
`updateDict(params, feedback)`. So:

- **No phase-runner or ODE function constructs filenames.**
- **No phase-runner or ODE function indexes raw `SB99_data` columns.**
- The interface contract is: 10 interpolators on `params['SB99f']` with the
  exact keys listed in §3, plus 12 scalar params written by `updateDict` (also §3).

That means **every consumer in §6.3–§6.8 is transparent to the refactor** as
long as the new loader produces the same 10 interpolators under the same keys.
The work concentrates in §6.1, §6.2, §6.10, and the cooling coupling in §6.11.

### 6.1 `src/sb99/read_SB99.py` — 🔴 loader (the main refactor target)

**What it does today.**

| Step | Lines | Action |
|------|-------|--------|
| Validate params | 104-117 | Requires `path_sps, SB99_rotation, ZCloud, SB99_BHCUT, FB_mColdWindFrac, FB_thermCoeffWind, FB_mColdSNFrac, FB_thermCoeffSN, FB_vSN`. |
| Build filename | 125 (→ `get_filename` 285-372) | Hardcoded `{mass}cluster_{rot\|norot}_{Z0014\|Z0002}_{BH120\|BH40}.txt`. Whitelists ZCloud ∈ {1.0, 0.15} and SB99_BHCUT ∈ {120, 40}. |
| Load | 132, validate ≥7 cols at 145 | `np.loadtxt` only — no header. |
| Unit conversion | 158-177 | Columns are positional: 0=t [yr], 1=log Qi, 2=log fi, 3=log Lbol [erg/s], 4=log Lmech_tot [erg/s], 5=log pdot_W [g·cm/s²], 6=log Lmech_W [erg/s]. All log-space except t and fi. |
| Derived (Li, Ln) | 191-192 | `Li = Lbol·fi`, `Ln = Lbol·(1-fi)`; uses hardcoded 13.6 eV threshold (i.e. SB99 column 2 is *defined* as the 13.6 eV ionizing fraction). |
| Derive Lmech_SN | 195 | `Lmech_SN_raw = Lmech_total - Lmech_W` (subtraction, not from file). |
| Wind corrections | 211-224 | `Mdot_W = pdot²/(2·Lmech_W)`; rescale by `FB_mColdWindFrac`, `FB_thermCoeffWind`. |
| SN corrections | 231-245 | `v_SN = FB_vSN.value` (constant); `Mdot_SN = 2·Lmech_SN/v_SN²`; rescale by `FB_mColdSNFrac`, `FB_thermCoeffSN`. |
| Totals | 252-253 | `Lmech_total`, `pdot_total = wind + SN`. |
| Prepend t=0 | 260-273 | All arrays get `[0]` index copy at t=0 to extend interpolation range. |
| Interp factory | 375-460 | Wraps 10 arrays in `scipy.interpolate.interp1d(kind='cubic')`. |

**Breaks with generic CSV.**

- Hardcoded filename: a generic CSV will not match this grammar. (`sps_path` proposal fixes this.)
- Hardcoded column order + log-space + cgs: a generic CSV may have a header, may already be linear-space, may use different units.
- `Lmech_SN` is *derived* from `(Lmech_total − Lmech_W)`. A modern SPS code could provide SN directly; the loader must accept that too.
- `v_SN = FB_vSN` is a user-supplied constant. Generic SPS may carry SN velocity over time → support a column.

**Required changes (refactor target).**

1. Replace the validation block (109-113) with `'sps_path'` instead of `SB99_*`.
2. Delete `get_filename()` (or guard it behind a `_legacy_sb99_grammar()` helper that *constructs* `sps_path` from the legacy params when `sps_path` is unset, for back-compat).
3. Introduce a column-map: either via a CSV header row (auto-detect) or a YAML sidecar. SB99's 7-column positional format remains a preset.
4. Make log-space and units per-column metadata (or document linear cgs as the required CSV format).
5. Allow optional `Lmech_SN`, `pdot_SN`, `Mdot_SN`, `v_SN` columns; fall back to today's derivation when absent.

### 6.2 `src/sb99/update_feedback.py` — 🟡 query function (rename only)

**What it does today.** `get_currentSB99feedback(t, params)` (lines 98-205):
reads 10 interpolators from `params['SB99f'].value`, validates `t ∈ [t_min, t_max]`
using `SB99f['fQi'].x[0/-1]` (156-157), evaluates each, computes
`v_mech_total = 2·Lmech_total/pdot_total` (184) and `pdotdot_total` by central
difference at `dt=1e-9 Myr` (187-188), returns `SB99Feedback` dataclass.

**Breaks with generic CSV.** Nothing functional. The function is interface-pure
relative to the interpolator dict.

**Required changes.**

- 🟡 (Optional, do last.) Rename `SB99f` → `sps_f`, `SB99Feedback` → `SPSFeedback`,
  `get_currentSB99feedback` → `get_current_sps_feedback`. Mechanical and
  touches ~10 phase files (see §6.3-§6.8).
- 🟢 No functional change — the dataclass already abstracts over the source.

### 6.3 `src/phase0_init/get_InitPhaseParam.py` — 🟡

**Direct interpolator access**: lines 88, 111-112 read `params['SB99f'].value`
and call `SB99f['fLmech_W'](tSF)`, `SB99f['fpdot_W'](tSF)` directly (not via
`get_currentSB99feedback`). Computes initial wind terminal velocity for the
free-streaming phase that initializes Weaver.

**Breaks.** None functionally. If we rename `SB99f` → `sps_f`, this is the one
phase file that touches the interpolator dict directly.

**Required changes.** 🟡 Rename only. (Alternatively, refactor to use
`get_currentSB99feedback(tSF, params)` and read `feedback.Lmech_W`,
`feedback.pdot_W` — cleaner.)

### 6.4 `src/phase1_energy/` — 🟢

**Call sites.**
- `run_energy_phase_modified.py:92, 158, 358, 400` — runner segments.
- `energy_phase_ODEs_modified.py:189, 324` — ODE RHS and post-step derived quantities.

**Pattern.** Every call site: `feedback = get_currentSB99feedback(t, params);
updateDict(params, feedback)`. Fields actually read: `Lmech_total`, `v_mech_total`.

**Breaks.** None. Phase only touches the dataclass/params view, not the loader.

**Required changes.** 🟢 None.

### 6.5 `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` — 🟢

**Call sites.** Lines 68 (import), 555 (loop), 917 (post-step), 1001 (final).
Same pattern as §6.4.

**Breaks.** None.

**Required changes.** 🟢 None.

### 6.6 `src/phase1c_transition/run_transition_phase_modified.py` — 🟢

**Call sites.** Lines 57 (import), 472, 770, 854. Same pattern.

**Breaks.** None.

**Required changes.** 🟢 None.

### 6.7 `src/phase2_momentum/run_momentum_phase_modified.py` — 🟢

**Call sites.** Lines 58, 405 (ODE), 552 (loop), 906 (final). Same pattern.

**Breaks.** None.

**Required changes.** 🟢 None.

### 6.8 `src/bubble_structure/bubble_luminosity_modified.py` — 🟢

The `get_currentSB99feedback` import at line 33 is **unused**: bubble structure
reads `params['Lmech_total']`, `params['v_mech_total']`, `params['Qi']`
(lines 104-105, 312, 345, 795) — values already written into `params` by the
caller's `updateDict(params, feedback)`. No direct interpolator access.

**Breaks.** None.

**Required changes.** 🟢 None. (Could drop the dead import in a separate
cleanup.)

### 6.9 `src/_output/cloudy/` and `src/_plots/` — 🟢 (orthogonal)

These are about CLOUDY post-processing and paper plots:

- `snapshot_to_deck.py:166-177` — `age_min_yr/age_max_yr` SB99 age-band check
  used for CLOUDY deck generation. Unrelated to the feedback CSV loader.
- `trinity_to_cloudy.py:71-130, 364, 454-458` — passes a `{{SB99_MOD}}`
  sentinel into a CLOUDY deck template. References the CLOUDY-compiled SB99
  *atmosphere grid*, which is a separate concept from our feedback CSV. Will
  outlive this refactor.
- `_plots/paper_*.py` — references in comments/docstrings only.

**Required changes.** 🟢 None for this refactor. (May want to revisit the
"SB99 atmosphere" terminology once feedback is generalized, for clarity.)

### 6.10 `src/_input/read_param.py` and `default.param` — 🔴 plumbing

**Files / lines.**
- `default.param:163-176` — declares `SB99_mass, SB99_rotation, SB99_BHCUT`.
- `default.param:306` — declares `path_sps`.
- `read_param.py:377-383` — resolves `path_sps` (`def_dir` → `lib/sps/starburst99/`).
- `read_param.py:472-473` — declares `SB99_data, SB99f` runtime containers.
- `read_param.py:474-485` — declares the 12 scalar feedback params that
  `updateDict(params, feedback)` writes into.

**Required changes.**

1. Add a new param `sps_path` (file path) and optionally `sps_refmass` (replaces
   `SB99_mass`'s role at `main.py:142`).
2. Keep `SB99_mass/rotation/BHCUT` as deprecated fallbacks that *construct*
   `sps_path` via the legacy grammar if `sps_path` is unset. Emit a
   `DeprecationWarning` so users see the migration path.
3. Rename the runtime containers `SB99_data → sps_data, SB99f → sps_f`
   (mechanical sweep across consumers in §6.3-§6.8).

### 6.11 `src/cooling/non_CIE/read_cloudy.py` — 🔴 *separate* SB99 coupling

**Newly discovered during deep-dive.** This file *also* uses `SB99_rotation`
and `ZCloud` to construct cooling-table filenames:

- Line 47: reads `params['SB99_rotation'].value`.
- Line 263-331: `get_filename(age, metallicity, SB99_rotation, path2cooling)`
  → `opiate_cooling_{rot|norot}_Z{1.00|0.15}_age{age}.dat`.

These non-CIE cooling cubes were generated by running CLOUDY with SB99-based
SEDs at specific cluster ages. So even if we decouple the feedback CSV from
SB99, **the cooling tables are still tied to the SB99 rotation/Z choice**
because that determined the SED used to generate them.

**Implications.**

- A user supplying a generic SPS CSV with metallicity outside {1.0 Z☉, 0.15 Z☉}
  or with different stellar physics would *also* need matching non-CIE cooling
  tables — and a path-indirection mechanism for those tables analogous to
  `sps_path`.
- For the current refactor scope, document this and keep cooling-table
  selection on the existing SB99-keyed grammar. Optionally introduce a
  `path_cooling_nonCIE_files` knob later.

Also: `cooling/non_CIE/read_cloudy_old.py:287` has the same coupling but is
legacy.

## 7. Per-consumer change matrix (refactor checklist)

| File | Touches filename? | Touches interpolators? | Touches params dict only? | Change needed |
|------|:--:|:--:|:--:|--|
| `src/sb99/read_SB99.py` | ✅ | builds them | — | 🔴 rewrite as `read_sps.py` with column map; delete `get_filename()` |
| `src/sb99/update_feedback.py` | — | ✅ reads | writes 12 scalars | 🟡 rename `SB99f`→`sps_f` (last step) |
| `src/main.py:142-152` | — | calls `read_SB99` | populates containers | 🔴 swap loader; replace `f_mass = mCluster/SB99_mass` |
| `src/_input/read_param.py` | declares containers | — | — | 🔴 add `sps_path`/`sps_refmass`; keep legacy params as fallback |
| `src/_input/default.param` | declares legacy params | — | — | 🔴 add `sps_path`; mark SB99_* deprecated |
| `src/phase0_init/get_InitPhaseParam.py` | — | ✅ reads directly | — | 🟡 rename only |
| `src/phase1_energy/run_energy_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1_energy/energy_phase_ODEs_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1c_transition/run_transition_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase2_momentum/run_momentum_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/bubble_structure/bubble_luminosity_modified.py` | — | — | ✅ (and dead import) | 🟢 (optional: drop import) |
| `src/cooling/non_CIE/read_cloudy.py` | ✅ separate filename | — | — | 🔴 own indirection (out of scope for feedback refactor) |
| `src/_output/cloudy/*` | — | — | — | 🟢 (orthogonal: CLOUDY atmosphere grid) |
| `src/_plots/*` | — | — | — | 🟢 (comments only) |
| `src/_input/dictionary.py:1203-1234` | — | — | demo `__main__` only | 🟢 |

## 8. Minimum-viable refactor (concrete sketch)

The actual code change to "accept self-defined paths" can be done in two
small PRs without touching any phase code:

**PR-A: introduce `sps_path` as the single input.**

1. `default.param`: add `sps_path  def_path` (sentinel). Mark `SB99_mass /
   SB99_rotation / SB99_BHCUT` as deprecated in their `# INFO` lines.
2. `read_param.py`: resolve `sps_path` — if `def_path`, fall through to
   `get_filename(params)` + `path_sps` to *compute* a concrete path. Emit a
   `DeprecationWarning` when the legacy params are used.
3. `read_SB99.read_SB99`: change `filepath = path2sps + get_filename(params)`
   to `filepath = params['sps_path'].value`. Keep everything else the same.
4. `main.py:142`: read `f_mass` from `params['sps_refmass']` (new param,
   defaults to `SB99_mass`'s value for back-compat) instead of `SB99_mass`.

After PR-A, users can drop in any CSV that follows SB99's column convention
just by pointing `sps_path` at it. No phase code changes.

**PR-B: column-mapping layer.**

1. `read_SB99.read_SB99` learns to read a header row (if present), mapping
   column names → indices. SB99's 7-column positional format becomes one
   preset triggered when no header is present.
2. Allow optional columns: `Lmech_SN`, `pdot_SN`, `Mdot_SN`, `v_SN`. When
   present, skip the derivation in steps 4-5. When absent, derive as today.
3. Per-column units (log-vs-linear, cgs-vs-AU) via header tokens.

**PR-C (cosmetic, do last): rename `SB99f → sps_f`, `SB99_data → sps_data`**.

## 9. Working notes (updated 2026-05-12)

- Branch: `claude/sb99-default-parameter-ttQIN` (violates CLAUDE.md
  `feature/bugfix/hotfix/fix` rule — **confirm with user before pushing**).
- `SB99f` and `SB99_data` are flagged `exclude_from_snapshot=True`
  (`read_param.py:472-473`).
- Audit covers 174 SB99-keyword hits across 18 files (2026-05-12). Only 6 are
  real consumers (the rest are comments, docstrings, demo code, or the
  separate non-CIE cooling coupling).
- `lib/sps/starburst99/` is the on-disk default but `path_sps` is the
  indirection.
- Dead import: `bubble_luminosity_modified.py:33` imports
  `get_currentSB99feedback` and never calls it.
- The numerical derivative step `dt = 1e-9 Myr` (`update_feedback.py:187`) is
  unconditionally small; if your interpolator's `t_min` is near 0, calling
  `fpdot_total(t - 1e-9)` near the boundary will raise. Existing SB99 grids
  are coarse enough that this hasn't surfaced; flag for any user-supplied
  CSV whose `t_min > 0`.

## 7. Working notes

- Branch: `claude/sb99-default-parameter-ttQIN` (violates the
  `feature/bugfix/hotfix/fix` rule in CLAUDE.md — confirm with user before
  pushing).
- `SB99f` and `SB99_data` are flagged `exclude_from_snapshot=True`
  (`read_param.py:472-473`).
- `grep -ri sb99 src/` returns 174 hits across 18 files (as of 2026-05-12).
- `lib/sps/starburst99/` is the on-disk default but `path_sps` is the indirection.
