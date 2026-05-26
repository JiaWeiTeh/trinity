# Cooling-table refactor: audit + implementation plan

Single source of truth for decoupling the cooling-table loaders from the
hardcoded SB99/OPIATE/CLOUDY assumptions, in the same shape as
`analysis/sb99-refactor-audit.md`. Combines (Part I) the architectural
audit — *what is* — with (Part II) the phased refactor plan and its
equivalence-test battery — *what to do, in what order, and how to prove
nothing changed*.

End goal: a user can drop in arbitrary CIE cooling tables (2-column
`logT, logΛ` files from any source) and arbitrary non-CIE cooling cubes
(generated from any SPS-driven CLOUDY pipeline, not just SB99-keyed
OPIATE) without modifying code, while every legacy parameter combination
continues to load byte-identically.

## TL;DR

- The cooling module is **structurally cleaner than pre-refactor SPS** — every
  consumer goes through `get_dudt(age, ndens, T, phi, params)`
  (`src/cooling/net_coolingcurve.py:22`), and physics phases only touch
  pre-built interpolators stored on `params['cStruc_*']`. The
  consumer-side abstraction is already correct.
- The **loader-side coupling is the SB99-loader problem twice over**:
  - **Non-CIE** (`src/cooling/non_CIE/read_cloudy.py:266-336`) hardcodes the
    OPIATE filename grammar `opiate_cooling_{rot|norot}_Z{1.00|0.15}_age{age}.dat`,
    reads `params['SB99_rotation']` directly, and whitelists ZCloud ∈
    {1.0, 0.15} with a `NameError` trap if neither matches.
  - **CIE** (`src/_input/read_param.py:424-436`) uses an integer index
    (1/2/3) to select between four hardcoded files, with `ZCloud == 1.0`
    vs `0.15` branching and the same silent fall-through trap if neither
    matches.
  - Both formats hardcode the file's column layout (column names for
    non-CIE; positional `(logT, logΛ)` for CIE), with no per-column
    units / log-or-linear declaration.
- **Four PRs**, each back-compat and independently revertable, mirroring the
  SB99 PR sequence:
  1. `path_cooling_nonCIE` becomes a real path knob; remove
     `SB99_rotation` from `read_cloudy.py`; OPIATE filename grammar moved
     to a legacy-fallback helper in `read_param.py`.
  2. Per-format in-`.param` column maps via `cool_col_cie_<canonical>` and
     `cool_col_nonCIE_<canonical>` (legacy OPIATE + 2-col CIE remain
     permanent presets; strict, no silent fallback).
  3. CIE `path_cooling_CIE` accepts a path directly (integer-index preset
     remains for back-compat); harmonize the `cStruc_*` container set.
  4. Cleanup — drop the unused `metallicity` arg on `CIE.get_Lambda`;
     remove the silent if/elif fall-through bugs. **Legacy params (the
     integer 1/2/3 CIE index, the OPIATE filename grammar, `ZCloud`-keyed
     CIE selection) remain as permanent fallbacks, never removed.**
- **Headline risks:** (a) cube-cache staleness if a user swaps cooling
  files at the same path (the `_cube.npy` cache at `read_cloudy.py:170`
  is keyed only on stem, no axis hash); (b) silent physics-altering
  loaders if a user mis-declares `log` vs `linear` for a cooling-rate
  column. Both are the same class of risk the SB99 refactor flagged.
- **Two equivalence guarantees** enforced on every PR (identical to the
  SB99 refactor's contract):
  1. Bitwise (`np.array_equal`) at loader / interpolator / cube layers.
  2. Tight-tolerance (`rtol=1e-12, atol=0`) snapshot-tree equivalence
     against a golden captured ONCE on main before PR-1.

---

# Part I — Audit (what is)

## 1. Current architecture

Cooling is bifurcated by temperature at `log T = 5.5`:

- **Non-CIE** (`T < 10^5.5 K`): 3D table over `(ndens, T, phi)`.
  Loaded periodically inside the simulation by
  `non_CIE.get_coolingStructure(params)`
  (`src/cooling/non_CIE/read_cloudy.py:22-134`), which:
  (a) consults `params['SB99_rotation']` and `params['ZCloud']` to build
  an OPIATE filename via `get_filename()`
  (`src/cooling/non_CIE/read_cloudy.py:266-336`); (b) reads the table
  via `astropy.io.ascii.read` keyed by hardcoded column names
  `ndens, temp, phi, cool, heat`
  (`src/cooling/non_CIE/read_cloudy.py:179-187`); (c) builds two 3D
  cubes plus a net interpolator
  (`src/cooling/non_CIE/read_cloudy.py:94-132`); (d) caches the rebuilt
  cubes as `<stem>_cube.npy` next to the source file
  (`src/cooling/non_CIE/read_cloudy.py:170-172, 261`).
- **CIE** (`T ≥ 10^5.5 K`): 1D `logΛ(logT)` table loaded once at startup
  by `src/main.py:162-171` via `np.loadtxt(path).T` → `scipy.interp1d`. The
  resolved path is computed at `src/_input/read_param.py:423-436` from
  the integer-index selector `path_cooling_CIE ∈ {1, 2, 3}` (when
  `ZCloud == 1`) or pinned to a single Sutherland-Dopita file (when
  `ZCloud == 0.15`).

`net_coolingcurve.get_dudt(age, ndens, T, phi, params)`
(`src/cooling/net_coolingcurve.py:22-163`) reads the pre-built
structures off `params['cStruc_*']`, dispatches on T against the
hardcoded `10^5.5 K` boundary
(`src/cooling/net_coolingcurve.py:93, 95`), evaluates the appropriate
interpolator (or interpolates linearly across the overlap zone), and
returns `dudt` in AU units.

Non-CIE structures are refreshed periodically during phase 1 / phase
1b:
`src/phase1_energy/run_energy_phase_modified.py:125-130`
(`COOLING_UPDATE_INTERVAL = 5e-2 Myr`,
`src/phase1_energy/run_energy_phase_modified.py:56`) and
`src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:535-540`
(`COOLING_UPDATE_INTERVAL = 5e-3 Myr`,
`src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:108`).

## 2. Configuration surface

| Param | File:line | Default | Role |
|-------|-----------|---------|------|
| `path_cooling_CIE` | `src/_input/default.param:316` | `3` | Integer index into the hardcoded `{1:Cloudy, 2:Cloudy+grains, 3:Gnat-Ferland}` table (when `ZCloud == 1`). Resolved to a path by `read_param.py:430-432`. **Cannot today be a path string** — `int(...)` at `read_param.py:430` will `ValueError`. |
| `path_cooling_nonCIE` | `src/_input/default.param:319` | `def_dir` | Directory of OPIATE cubes; resolves to `lib/default/opiate/` (`read_param.py:417`). Used by `read_cloudy.get_filename()` to build per-age filenames. |
| `ZCloud` | `src/_input/default.param:85` | `1` | Drives CIE file selection AND non-CIE filename's `Z_str`. Whitelisted to `{1.0, 0.15}` in both spots. Also drives dust opacity / other metallicity-keyed physics — **not deprecated by this refactor**. |
| `SB99_rotation` | `src/_input/default.param:180` | `1` | Used by both the SPS loader AND `read_cloudy.get_filename()`'s `{rot|norot}` discriminator. The non-CIE coupling is the §5.1 hot spot. |
| `cool_alpha` / `cool_beta` / `cool_delta` | `src/_input/default.param:304-308` | `0.6 / 0.8 / -6/35` | Cooling-related physics constants for the analytic bubble model; **orthogonal to the table loaders** — flagged only so they aren't accidentally pulled in. |

## 3. Interface every consumer expects

After `non_CIE.get_coolingStructure(params)` (loader) and `main.py:162-171`
(CIE) populate the cooling-structure container set on `params`,
downstream code needs these six `cStruc_*` entries
(`src/_input/read_param.py:626-631`, all marked
`exclude_from_snapshot=True`):

```
cStruc_cooling_nonCIE                  # cube class; has .datacube .interp .ndens .temp .phi
cStruc_heating_nonCIE                  # cube class; same shape
cStruc_net_nonCIE_interpolation        # RegularGridInterpolator over (log_ndens, log_T, log_phi)
cStruc_cooling_CIE_logT                # 1D array, log10(T) sample points
cStruc_cooling_CIE_logLambda           # 1D array, log10(Lambda) at those T
cStruc_cooling_CIE_interpolation       # interp1d over (logT)
```

`get_dudt()` (`src/cooling/net_coolingcurve.py:62-67`) reads:
- `cStruc_cooling_nonCIE.temp` (for the non-CIE side of the `10^5.5 K`
  cutoff)
- `cStruc_net_nonCIE_interpolation([log_ndens, log_T, log_phi])`
- `cStruc_cooling_CIE_logT` (for the CIE side of the cutoff)
- `cStruc_cooling_CIE_interpolation(logT)` (returns `log10(Lambda)`)
- `params['ZCloud'].value` — passed through to `CIE.get_Lambda` at lines
  88 and 148, but **never consulted inside `get_Lambda`** (see §5.2).

## 4. Format-specific hot spots (what blocks generic tables)

| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 1 | `src/cooling/non_CIE/read_cloudy.py:47, 59, 266-336` (`get_filename`) | Hardcoded OPIATE filename grammar; reads `SB99_rotation` and `ZCloud`; whitelists Z ∈ {1.0, 0.15}; if `ZCloud` is neither, `Z_str` is never bound and line 312 hits `NameError`. | PR-1: replace with `path_cooling_nonCIE` as a real path/folder knob; relocate `get_filename()` to `read_param.py` as a legacy fallback (same shape as the SPS `_get_legacy_sb99_filename()` helper at `src/_input/read_param.py:35`). |
| 2 | `src/cooling/non_CIE/read_cloudy.py:182-187` | Hardcoded column names `ndens, temp, phi, cool, heat`. No header-vs-position dispatch; no per-column units / log/linear. | PR-2: `cool_col_nonCIE_<canonical>  <file_column>  <units>  <log\|linear>` declarations, paralleling `sps_col_<canonical>`. |
| 3 | `src/cooling/non_CIE/read_cloudy.py:189-194` | "Wrong sign" auto-flip on cooling/heating columns. Hides a real convention mismatch under a `print` warning. | PR-2: a per-column `<units>` declaration that includes a `cgs_neg` alias for the negative-cgs case. Make the convention explicit, not auto-detected. |
| 4 | `src/cooling/non_CIE/read_cloudy.py:200-210` (`create_limits`) + 227-233 (fill loop) | Axis ticks built by `np.log10(set(col))` then rounded to **3 decimal places**; the fill loop also rounds to 3 dp for the `np.where` lookup. Fragile to any OPIATE-incompatible grid spacing. Note: this exact rounding mismatch caused an `IndexError` previously, fixed in commit c1589fc. | PR-2: optional `<grid_decimals>` declaration per axis, defaulting to 3 (preserves OPIATE bit-exactness); document the requirement that the file is on a regular log-spaced grid. |
| 5 | `src/cooling/non_CIE/read_cloudy.py:170-172, 261` | Cube cache (`<stem>_cube.npy`) keyed only on the source filename's stem. Same path, different file contents → silently uses stale cache. | PR-2: include axis-array hash (`hashlib.md5` of the concatenated tick arrays) in the cache filename, OR check axis-array equality before reusing. |
| 6 | `src/cooling/non_CIE/read_cloudy.py:298-307` (age-discovery loop) + `src/cooling/non_CIE/read_cloudy.py:339-343` (`get_fileage`) | Age list discovered by `os.listdir` filtering for `.dat`, with file age parsed from a **fixed-width slice** `filename[idx+3:idx+11]` (`src/cooling/non_CIE/read_cloudy.py:343`). Brittle to any filename containing `age` outside the OPIATE convention. | PR-1: when `path_cooling_nonCIE` is a single-file path (user mode), skip age discovery entirely — there is no per-age cube set. When it's a folder, allow a configurable file-age regex. |
| 7 | `src/_input/read_param.py:424-436` (CIE-path resolution) | `path_cooling_CIE` is parsed as `int()`; user **cannot** point it at a custom file. If `ZCloud` is neither 1 nor 0.15 the if/elif silently falls through, leaving `params['path_cooling_CIE'].value` as the integer 3 → `main.py:165` calls `np.loadtxt(3, ...)` → cryptic error. | PR-3: detect path-vs-int at `read_param.py:430`; if a string-looking value, use it verbatim; else apply the legacy index table. Raise an explicit `ValueError` instead of falling through. |
| 8 | `src/cooling/net_coolingcurve.py:93, 95` | Hardcoded `5.5` (in log10 K) as the CIE/non-CIE boundary. Currently hand-wired to the OPIATE table's `T_max` and a typical CIE-table `T_min`. | PR-2: derive from the loaded tables — `max(cStruc_cooling_nonCIE.temp)` and `min(cStruc_cooling_CIE_logT)` are already in scope; the magic literal can be replaced with a sanity assertion that the two ranges meet near 5.5. |
| 9 | `src/cooling/net_coolingcurve.py:85` | `if T < 1e4: T = 1e4` — floor hardcoded to the OPIATE non-CIE table's `T_min`. | PR-2: floor to `min(cStruc_cooling_nonCIE.temp)` instead of the constant. |
| 10 | `src/cooling/CIE/read_coolingcurve.py:25` (`get_Lambda(T, interp, metallicity)`) | `metallicity` arg accepted but unused (lines 60-61 commented out). `net_coolingcurve.py:88, 148` pass `params_dict['ZCloud'].value` into it for no functional effect. | PR-4: drop the unused arg; remove the two `ZCloud` reads from `net_coolingcurve.py`. |
| 11 | `src/cooling/non_CIE/read_cloudy.py:130` | `netcooling = cool_cube - heat_cube` is in **linear** space, but the standalone `cooling_interpolation` and `heating_interpolation` interpolate in **log** space (`read_cloudy.py:94-97`). The two consumers don't see the same units. | PR-2: document the unit convention per `cStruc_*` entry in the canonical registry; flag whether each cube is log or linear in the `cool_col_nonCIE_*` declarations. |

## 5. Consumer-by-consumer deep dive

Status: 🔴 needs change · 🟡 cosmetic / cleanup only · 🟢 transparent.

### Architectural takeaway (the single most important point)

After loader-side population, **every downstream consumer goes through
exactly one function**: `net_coolingcurve.get_dudt(age, ndens, T, phi,
params)` (`src/cooling/net_coolingcurve.py:22`). The only direct
loader-imports outside `src/cooling/` are at:

- `src/bubble_structure/bubble_luminosity_modified.py:31` — imports
  `net_coolingcurve` only (no direct loader reach-through).
- `src/phase1_energy/run_energy_phase_modified.py:34` — imports
  `non_CIE.get_coolingStructure` to refresh the cube periodically.
- `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:66`
  — same.

The only `get_dudt` call site is
`src/bubble_structure/bubble_luminosity_modified.py:797`. So every
physics consumer is **transparent to the refactor** as long as the new
loaders produce the same six `cStruc_*` entries under the same keys with
the same internal shapes. The work concentrates in §5.1, §5.2, §5.3,
and §5.7.

### 5.1 `src/cooling/non_CIE/read_cloudy.py` — 🔴 loader (main refactor target)

**What it does today.**

| Step | Lines | Action |
|------|-------|--------|
| Read SB99-coupled params | 47-48 | `SB99_rotation = params['SB99_rotation'].value`; `metallicity = params['ZCloud'].value`. |
| Build filename | 59, 266-336 | `filename = get_filename(age, metallicity, SB99_rotation, path2cooling)`; OPIATE grammar with hardcoded `{rot|norot}`, `Z{1.00|0.15}`, `age{a}` fields. Whitelisted ZCloud values; silent `NameError` if neither matches. |
| Age dispatch | 309-334 | `os.listdir` to discover available ages from `.dat` files; if request age is in the list, single file; if below/above, clamp; else return `[lower, higher]` for linear interpolation. |
| Load table | 179 | `ascii.read(path)`. |
| Read columns | 182-194 | Hardcoded names `ndens / temp / phi / cool / heat`; auto-flip negative signs with a `print` warning. |
| Build axes | 200-215 | `create_limits`: `set → sort → log10 → round(3)`. |
| Fill cubes | 222-250 | Iterate rows, find `np.where(axis == round(log10(val), 3))`, write into `cool_cube` and `heat_cube`. NaN-initialized; sparse coverage stays NaN. |
| Cube cache | 170-172, 261 | Save/load `<stem>_cube.npy`; stem-keyed, no axis hash. |
| Time interpolation between ages | 83-90 | Linear in age between two cubes when request falls between available files. |
| Build interpolators | 94-97, 132 | `RegularGridInterpolator((log_n, log_T, log_phi), np.log10(cube))` for the standalone cubes (log-space) and `(…)` for `cool - heat` (linear-space). |

**Breaks with generic tables.**

- Hardcoded filename grammar: a user's cooling cube won't match.
- Hardcoded column names: a user's file may use `n_H`, `T`, `Phi`,
  `Lambda`, `Gamma` or similar.
- Implicit sign convention auto-flipped.
- Cube-cache silently stale on same-path overwrites.
- Age-discovery and per-age filename parsing tied to OPIATE convention.

### 5.2 `src/cooling/CIE/read_coolingcurve.py` — 🟡 (small)

`get_Lambda(T, cooling_CIE_interpolation, metallicity)`
(`src/cooling/CIE/read_coolingcurve.py:25`) takes a `metallicity` arg
but never uses it (the would-be metallicity branch at lines 60-61 is
commented out). The actual table choice has already been baked into
`cooling_CIE_interpolation` by `main.py:165-167`, so `metallicity` here
is dead weight. The function body is two lines: `T = log10(T); return
10**interp(T)`.

🟡 PR-4: drop the unused arg and the two `ZCloud` reads at
`src/cooling/net_coolingcurve.py:88, 148` that pass it.

### 5.3 `src/_input/read_param.py` and `default.param` — 🔴 plumbing

CIE path resolution (`src/_input/read_param.py:423-436`):

- `path_cooling_CIE` parsed as `int()` (line 430) → user can't set it to
  a path string today; `int('/my/file.dat')` → `ValueError`.
- If `ZCloud == 1`: integer-indexed lookup in the hardcoded `cie_files`
  dict.
- Elif `ZCloud == 0.15`: pinned to Sutherland-Dopita.
- Else: silent fall-through; `params['path_cooling_CIE'].value` remains
  the integer 3 (default) and `main.py:165` later calls
  `np.loadtxt(3, ...)`.
- If `cie_choice not in cie_files` (e.g. user wrote `path_cooling_CIE
  4`): the if-body doesn't execute and the param value is left as the
  raw integer — same trap.

Non-CIE path resolution (`src/_input/read_param.py:415-421`): handles
`def_dir` correctly; user-supplied folder paths are also handled. The
SB99/OPIATE coupling lives inside `read_cloudy.get_filename()`
downstream.

Runtime containers (`src/_input/read_param.py:626-631`) — see §3 for
the six `cStruc_*` entries.

**Required changes.** Add `cool_col_cie_*` and `cool_col_nonCIE_*` key
families; allow path-string for `path_cooling_CIE`; keep the
integer-index table and `ZCloud == 1.0 / 0.15` branches as a permanent
fallback (§9). Replace silent fall-throughs with explicit
`ValueError`s.

### 5.4 `src/main.py:162-173` — 🟢 (one-line touch in PR-3)

```python
cooling_path = params['path_cooling_CIE'].value
logT, logLambda = np.loadtxt(cooling_path, unpack=True)
cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind='linear')
```

🟢 unchanged in PR-1 and PR-2. In PR-3 this block routes through a new
`load_cie_table(params)` helper that respects the column map (so a CIE
file with a header, swapped column order, or non-default units still
works).

### 5.5 `src/cooling/net_coolingcurve.py` — 🟡 (constants, no logic)

166 LOC. The only changes needed:

- `src/cooling/net_coolingcurve.py:85` — floor `T` against
  `min(cStruc_cooling_nonCIE.temp)` instead of literal `1e4`.
- `src/cooling/net_coolingcurve.py:93, 95` — derive the `5.5` cutoff
  from the loaded tables (or sanity-assert the two ranges meet there).
- `src/cooling/net_coolingcurve.py:88, 148` — drop the
  `params_dict['ZCloud'].value` argument when PR-4 lands.

No interpolator-call patterns change. The body that does the CIE/non-CIE
dispatch (`src/cooling/net_coolingcurve.py:102-158`) stays as-is.

### 5.6 Phase consumers — 🟢

- `src/phase1_energy/run_energy_phase_modified.py:125-130` and
  `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:535-540`
  call `non_CIE.get_coolingStructure(params)` periodically and write the
  three non-CIE `cStruc_*` entries. As long as the loader returns the
  same `(cooling_data, heating_data, netcooling_interpolation)` triple
  with the same internal shapes, these are transparent to the refactor.
- `src/bubble_structure/bubble_luminosity_modified.py:797` calls
  `net_coolingcurve.get_dudt(...)` — interface unchanged.

### 5.7 `src/_output/snapshot_to_deck.py` and `src/_plots/*` — 🟢 (orthogonal)

No direct loader access; no `cStruc_*` reads. Unaffected.

## 6. Per-consumer change matrix

| File | Touches filename? | Touches columns? | Touches interpolators / cubes? | Change needed |
|------|:--:|:--:|:--:|--|
| `src/cooling/non_CIE/read_cloudy.py` | ✅ | ✅ hardcoded names | ✅ builds them | 🔴 rewrite with column map; move `get_filename()` to legacy fallback |
| `src/cooling/CIE/read_coolingcurve.py` | — | — | — (reads interp only) | 🟡 PR-4 drop unused `metallicity` arg |
| `src/main.py:162-173` | — | ✅ positional 2-col | builds CIE interp | 🟡 PR-3 swap `np.loadtxt` for `load_cie_table` helper |
| `src/_input/read_param.py:415-436` | resolves both | — | declares containers | 🔴 add path-vs-int dispatch for CIE; add column-map plumbing; relocate OPIATE grammar |
| `src/_input/default.param` | declares legacy params | — | — | 🔴 add `cool_col_cie_*` / `cool_col_nonCIE_*` blocks; keep `path_cooling_CIE  3` and OPIATE-style `path_cooling_nonCIE  def_dir` as permanent fallbacks |
| `src/cooling/net_coolingcurve.py` | — | — | reads them | 🟡 replace literals (`1e4`, `5.5`) with derived limits; PR-4 drops unused `ZCloud` arg |
| `src/phase1_energy/run_energy_phase_modified.py` | — | — | ✅ refreshes cube | 🟢 |
| `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` | — | — | ✅ refreshes cube | 🟢 |
| `src/bubble_structure/bubble_luminosity_modified.py` | — | — | calls `get_dudt` | 🟢 |
| `src/_output/snapshot_to_deck.py` | — | — | — | 🟢 |
| `src/_plots/*` | — | — | — | 🟢 |

---

# Part II — Plan (what to do)

## 7. Goals and non-goals

### Goals

1. Replace the OPIATE filename grammar with a `path_cooling_nonCIE` knob
   that accepts either a folder (legacy multi-age cube set) or a single
   file (user mode, no age dispatch). Remove the `SB99_rotation` read
   from `src/cooling/non_CIE/read_cloudy.py`.
2. Allow `path_cooling_CIE` to be a path string in addition to an
   integer index. Keep the `{1, 2, 3}` integer preset (and the
   `ZCloud == 0.15` auto-pin to Sutherland-Dopita) as a permanent
   back-compat surface.
3. Allow per-column declarations of file-column name/index, units, and
   log/linear convention inside `.param` via:
   - `cool_col_cie_<canonical>` family for the CIE 1D table
     (canonicals: `T`, `Lambda`).
   - `cool_col_nonCIE_<canonical>` family for the non-CIE 3D cube
     (canonicals: `ndens`, `T`, `phi`, `Lambda_cool`, `Lambda_heat`).
4. Replace the literal `5.5` boundary and `1e4` floor in
   `src/cooling/net_coolingcurve.py` with values derived from the loaded
   tables. Keep the dispatch logic identical.
5. Make every step **byte-equivalent** to the current cooling path when
   run with legacy parameters.

### Non-goals

- Not retiring OPIATE/CLOUDY-keyed cooling as a data source. The OPIATE
  filename grammar stays as a permanent fallback, exactly like the SB99
  grammar in the SPS refactor.
- Not rewriting `net_coolingcurve.get_dudt`'s CIE/non-CIE dispatch logic
  or the linear overlap-zone interpolation. The body that does the
  routing (`src/cooling/net_coolingcurve.py:102-158`) is correct as-is.
- Not changing the cooling-update cadence (`COOLING_UPDATE_INTERVAL` at
  `src/phase1_energy/run_energy_phase_modified.py:56` and
  `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:108`).
- Not unifying the CIE and non-CIE loaders into one. They genuinely
  describe different physical objects and the parallel-but-separate
  `cool_col_cie_*` / `cool_col_nonCIE_*` registries are clearer than a
  forced abstraction.

## 8. Invariants (what MUST stay byte-identical under the legacy code path)

Legacy = config sets `path_cooling_CIE` to the integer 1/2/3 (or
defaults to 3), sets `ZCloud ∈ {1.0, 0.15}`, sets `SB99_rotation` and
the OPIATE-style `path_cooling_nonCIE`, and does **not** declare any
`cool_col_*` keys. Sentinel defaults route back to the legacy
behavior.

Under that fallback, the following must remain byte-identical PR-by-PR:

| Invariant | How verified |
|-----------|--------------|
| Resolved CIE path string | string equality |
| Resolved non-CIE filename(s) at each requested age | string equality, full matrix over `{rot, norot} × {1.0, 0.15} × ages` |
| Raw CIE `(logT, logLambda)` arrays from `np.loadtxt` | `np.array_equal` |
| `cStruc_cooling_CIE_interpolation` evaluated at dense logT grid | `np.array_equal` |
| Non-CIE cube triple (`log_ndens`, `log_T`, `log_phi`, `cool_cube`, `heat_cube`) per age | `np.array_equal` element-wise (NaN-aware: `np.array_equal(a, b, equal_nan=True)`) |
| Net non-CIE interpolator evaluated at dense (n, T, phi) grid | `np.array_equal` |
| `get_dudt(age, ndens, T, phi, params)` return at sampled inputs | `np.array_equal` |
| Every JSONL snapshot value in a full trinity run | `np.allclose(rtol=1e-12, atol=0)` per field |

`equal_nan=True` is necessary because the cube fill leaves
unrepresented `(n, T, phi)` triples as `NaN`
(`src/cooling/non_CIE/read_cloudy.py:224, 241`), and NaN ≠ NaN under
plain equality.

## 9. Migration strategy + legacy-as-permanent guarantee

New params, each with a sentinel default that routes back to the
existing behavior:

- `path_cooling_CIE` — already exists; today accepts only an integer
  index. **Extend** to also accept a path string. The integer-index
  preset `{1, 2, 3}` (under `ZCloud == 1`) and the auto-pin to
  Sutherland-Dopita (under `ZCloud == 0.15`) remain as permanent
  back-compat behavior.
- `path_cooling_nonCIE` — already exists; today resolved against the
  OPIATE filename grammar. **Extend** to also accept a single-file path
  (user mode; no age dispatch). When the path resolves to a folder, the
  OPIATE grammar runs as today.
- `cool_col_cie_<canonical>` family — one line per CIE-table canonical
  with positional fields `<file_column>  <units>  <log|linear>`. When
  absent and `path_cooling_CIE` is an integer or a path identified as a
  legacy-format file, use the hardcoded 2-column `(logT, logLambda)`
  preset. Required only when the user wants to override.
- `cool_col_nonCIE_<canonical>` family — same syntax, canonicals
  `ndens, T, phi, Lambda_cool, Lambda_heat`. When absent and
  `path_cooling_nonCIE` is the legacy OPIATE folder, the
  `ndens/temp/phi/cool/heat` preset runs.

### Legacy is permanent, not deprecated

**Hard guarantee:** a `.param` file that sets only the legacy params
(`path_cooling_CIE` integer, `path_cooling_nonCIE def_dir` or any
OPIATE folder, `ZCloud`, `SB99_rotation`) works forever. **None of the
four PRs removes the legacy presets.** Same policy and same wording as
the SB99 refactor's §9.

After all four PRs land:

- `path_cooling_CIE` continues to accept the integer indices `{1, 2, 3}`
  and the `ZCloud == 0.15` Sutherland-Dopita pin.
- `path_cooling_nonCIE = def_dir` continues to resolve to
  `lib/default/opiate/` and the OPIATE filename grammar continues to
  drive per-age file discovery.
- The hardcoded 2-column CIE preset and 5-column OPIATE preset remain
  the loaders' default behavior when no `cool_col_*` keys are declared.
- One startup `logger.info` line names the legacy mechanisms in use —
  informational, not a warning.

## 10. PR sequence

Each PR is independently mergeable. Order matters; do not reorder
without re-running the equivalence battery between every reordering.

### PR-1 — `path_cooling_nonCIE` decoupling + remove SB99 leakage

**Scope.** Stop reading `SB99_rotation` from
`src/cooling/non_CIE/read_cloudy.py`. Relocate the OPIATE filename
grammar to a legacy-fallback helper in `read_param.py` (mirroring
`_get_legacy_sb99_filename` at `src/_input/read_param.py:35-69`). When
`path_cooling_nonCIE` resolves to a single file, bypass age dispatch
entirely and load it directly. The CIE side, all loaders' column
layouts, and `net_coolingcurve.py` are untouched.

**Files touched.**

- `src/_input/read_param.py` — new `_resolve_nonCIE_paths(params,
  age)` helper that either (a) returns the user-supplied single file
  verbatim, or (b) runs the relocated OPIATE grammar over the folder.
- `src/cooling/non_CIE/read_cloudy.py` — drop the `SB99_rotation` and
  `ZCloud` reads from `get_coolingStructure`
  (`src/cooling/non_CIE/read_cloudy.py:47-48`); change `get_filename`
  call at line 59 to `_resolve_nonCIE_paths(params, age)`. Delete
  `get_filename` and `get_fileage` from this module (now live in
  `read_param.py`).
- `src/_input/default.param` — update the `# INFO` block on
  `path_cooling_nonCIE` to document the single-file user mode. Do
  **not** mark anything deprecated.

**Code-level checklist.**

- [ ] `src/cooling/non_CIE/read_cloudy.py` no longer references
      `SB99_rotation` (a `grep -n "SB99" src/cooling/` returns nothing).
- [ ] `_resolve_nonCIE_paths(params, age)` produces string-identical
      filenames to the old `get_filename(age, metallicity,
      SB99_rotation, path2cooling)` for every legacy combination.
- [ ] User mode: when `path_cooling_nonCIE` resolves to a file
      (`os.path.isfile`), the helper returns `[that_file]` regardless
      of `age`, and `get_coolingStructure` loads it without age
      interpolation.
- [ ] Startup `logger.info` line ("Using legacy OPIATE non-CIE cooling
      grammar (rotation=…, Z=…, folder=…)") fires exactly once per run.
- [ ] The silent `NameError` on unsupported ZCloud
      (`src/cooling/non_CIE/read_cloudy.py:290-295`) is replaced by an
      explicit `ValueError` with a fillable mitigation message.
- [ ] No CIE-side changes.

**Tests required to merge.** See §11.2.1 for the full battery. Headline:

1. Path-resolution matrix: 2 rotations × 2 Zs × N ages (covering the
   ages exposed by the OPIATE files in the chosen anchor configs). All
   produce string-identical filenames to a pre-refactor capture.
2. Cube byte-equivalence: load via the new helper, compare cubes to
   pre-refactor pickled goldens.
3. E2E snapshot-tree equivalence for the three anchor configs.
4. E2E equivalence with `path_cooling_nonCIE` explicitly pointed at the
   resolved single OPIATE file → must match the legacy folder mode for
   a config whose age happens to hit an exact file.

### PR-2 — In-`.param` column mapping for the non-CIE 3D cube

**Scope.** Add `cool_col_nonCIE_<canonical>` declarations parsed in
`read_param.py`, consumed by `read_cloudy.create_cubes`. Hardcoded
column names `ndens/temp/phi/cool/heat`
(`src/cooling/non_CIE/read_cloudy.py:182-187`) become the legacy
preset, used when no `cool_col_nonCIE_*` keys are declared OR when
`path_cooling_nonCIE` resolves to the legacy OPIATE folder. Strict
hard-error if a user declares a non-OPIATE path but supplies an
incomplete column map. The 5-key cube-cache mechanism gains an
axis-array hash (§4 issue 5).

**`.param` syntax.** One line per canonical column:

```
cool_col_nonCIE_<canonical>    <file_column>    <units>    <log|linear>
```

Canonicals (all required when user-mode):

| Canonical | Canonical linear unit | Description |
|-----------|------------------------|-------------|
| `ndens`        | `cm^-3`           | Number density (axis 0 of the cube). |
| `T`            | `K`               | Temperature (axis 1 of the cube). |
| `phi`          | `cm^-2 s^-1`      | Ionizing photon flux (axis 2 of the cube). |
| `Lambda_cool`  | `erg cm^3 / s`    | Cooling rate (sign-positive). |
| `Lambda_heat`  | `erg cm^3 / s`    | Heating rate (sign-positive). |

A `cgs_neg` units alias is provided for `Lambda_cool` / `Lambda_heat`
to declare "this column is in cgs but written negative-signed" without
needing the sign-auto-flip logic at lines 189-194.

Worked example:

```
path_cooling_nonCIE    /path/to/my_custom_cooling_cube.csv

cool_col_nonCIE_ndens         n_H              cm^-3            linear
cool_col_nonCIE_T             temperature      K                linear
cool_col_nonCIE_phi           Phi_ion          cm^-2 s^-1       linear
cool_col_nonCIE_Lambda_cool   cooling_rate     erg cm^3 / s     linear
cool_col_nonCIE_Lambda_heat   heating_rate     erg cm^3 / s     linear
```

**Files touched.**

- `src/_input/default.param` — add a commented documentation block for
  the `cool_col_nonCIE_*` family (inactive by default since
  `path_cooling_nonCIE = def_dir` routes to the legacy preset).
- `src/_input/read_param.py` — parse `cool_col_nonCIE_*` into a single
  `DescribedItem` whose value is a dict `{canonical:
  ColumnSpec(file_column, units, log)}`. Validate when in user mode.
- `src/cooling/non_CIE/read_cloudy.py` — refactor `create_cubes` to
  operate on a `column_map`: legacy preset (existing hardcoded
  `ndens/temp/phi/cool/heat` names) when in OPIATE mode; user-defined
  column map when in user mode. Cache filename includes the axis-array
  hash.
- A new `src/cooling/cool_columns.py` module — analog of
  `src/sb99/sps_columns.py` — that owns:
  - `CANONICALS_NONCIE` registry
  - `ColumnSpec` (reuse the SPS one if practical; otherwise mirror it)
  - `UNIT_CONVERSIONS_NONCIE`
  - `LEGACY_OPIATE_COLUMN_MAP` preset
  - `parse_cool_col_value` / `build_user_column_map_nonCIE` /
    `validate_user_column_map_nonCIE`
  - `load_user_cube_file` (the file-IO helper)

**Behaviour matrix.** Identical shape to the SPS PR-2 table.

| `path_cooling_nonCIE` value | `cool_col_nonCIE_*` keys present? | Behavior |
|----------------------------|------------------------------------|----------|
| `def_dir` (sentinel) → legacy folder | n/a (ignored) | Legacy OPIATE fallback. Loader uses the hardcoded preset. Byte-equivalent to PR-1. |
| user-defined folder identified as OPIATE-format | n/a (ignored) | Same legacy preset, OPIATE filename grammar over user's folder. |
| user-defined single file | none | Hard error. Fillable template to stderr; exits non-zero. |
| user-defined single file | partial | Hard error. Names which canonicals are missing. |
| user-defined single file | complete, integer indices only | Works on any layout (header optional). |
| user-defined single file | complete, includes string names | Works iff a header row is detected. |
| user-defined single file | complete | Use the user-declared mapping. |

**Code-level checklist.** (Subset; full list in the parallel
`sb99-refactor-audit.md §10 PR-2` checklist applies in spirit.)

- [ ] `cool_columns.py` exists and is imported by both
      `read_param.py` and `read_cloudy.py`.
- [ ] Legacy fallback path produces byte-identical cubes to PR-1.
- [ ] Cube cache filename incorporates `hashlib.md5(axes).hexdigest()[:8]`;
      stale-cache test passes (overwrite source file at same path →
      regenerated cube, not stale reuse).
- [ ] Sign auto-flip at
      `src/cooling/non_CIE/read_cloudy.py:189-194` removed; the
      `cgs_neg` unit alias makes the convention declarative. Legacy
      preset declares `cgs` (positive), since OPIATE convention.
- [ ] `net_coolingcurve.py:85` and `:93, 95` updated to derive from
      loaded tables; the `5.5` and `1e4` literals deleted.

**Tests required to merge.** Mirror SPS PR-2 tests 1, 2, 3, 4, 5, 6,
6b, 7, 13, 14, 15. The Li/Ln-style derivation tests (8-12) have no
non-CIE analog — cooling has no derived canonicals.

Plus cooling-specific:

- **Cube cache invalidation.** Generate a cube, overwrite the source
  file with different values at the same path, reload → cube reflects
  the new values (no stale reuse).
- **NaN-sparse cube round-trip.** A user file with missing
  `(n, T, phi)` rows produces a cube with NaNs at exactly those
  triples; the netcooling interpolator behaves identically to the
  legacy path.
- **Sign-convention declaration.** A file with negative cooling/heating
  declared `cgs_neg` produces a cube byte-identical to the same file
  with positive values declared `cgs`.

### PR-3 — In-`.param` column mapping for the CIE 1D table + path support

**Scope.** Allow `path_cooling_CIE` to be a path string. Add
`cool_col_cie_<canonical>` declarations for the 1D `(T, Lambda)`
table. Hardcoded `np.loadtxt(path).T → (logT, logLambda)` at
`src/main.py:165` becomes the legacy preset, used when
`path_cooling_CIE` is one of the integer indices `{1, 2, 3}` (under
`ZCloud == 1`) or when `ZCloud == 0.15` (which auto-pins to the
Sutherland-Dopita file regardless of `path_cooling_CIE`).

**Files touched.**

- `src/_input/default.param` — `cool_col_cie_*` documentation block.
- `src/_input/read_param.py:423-436` — detect path-vs-int at line 430;
  if a path-looking string, use it; else apply the legacy index table.
  Replace the silent if/elif fall-through with an explicit
  `ValueError` naming both the unsupported `ZCloud` and the available
  legacy combinations.
- `src/main.py:162-173` — extract the CIE-load block into
  `load_cie_table(params)` that respects the column map.
- `src/cooling/cool_columns.py` — extend with `CANONICALS_CIE`
  (`T`, `Lambda`), `UNIT_CONVERSIONS_CIE`, `LEGACY_CIE_COLUMN_MAP`
  (2-column positional, both in log space).

**Canonicals for CIE:**

| Canonical | Required? | Canonical linear unit |
|-----------|-----------|------------------------|
| `T`       | yes       | `K` |
| `Lambda`  | yes       | `erg cm^3 / s` |

**Tests required to merge.**

1. Legacy integer-index path → byte-identical CIE arrays vs PR-2
   golden.
2. `path_cooling_CIE = /path/to/custom.dat` with `cool_col_cie_*`
   declarations matching the legacy 2-column file → byte-identical
   arrays.
3. `path_cooling_CIE = 5` (out of range) → explicit `ValueError`
   naming the supported indices, not the previous silent
   fall-through.
4. `ZCloud = 0.5` → explicit `ValueError` naming the supported `ZCloud`
   values for legacy CIE, pointing the user at `path_cooling_CIE` as
   the escape hatch.
5. E2E snapshot equivalence across the three anchor configs against
   the PR-2 golden.

### PR-4 — Cleanup (legacy stays)

**Scope.** Cosmetic cleanup only. Does **not** remove any user-facing
surface.

**Files touched.**

- `src/cooling/CIE/read_coolingcurve.py` — drop the unused `metallicity`
  arg on `get_Lambda`
  (`src/cooling/CIE/read_coolingcurve.py:25`); update the docstring at
  lines 35-46 to reference the column map / path knob instead of the
  legacy integer table.
- `src/cooling/net_coolingcurve.py:88, 148` — drop the
  `params_dict['ZCloud'].value` argument to `CIE.get_Lambda`.

**Files explicitly NOT touched.**

- `src/_input/default.param` — `path_cooling_CIE 3`, `path_cooling_nonCIE
  def_dir`, the integer-index preset, and the OPIATE filename grammar
  all remain.
- `src/_input/read_param.py` — legacy resolution branches all remain.
- The startup `logger.info` notifications stay.

**Tests required to merge.**

1. Legacy configs (only integer `path_cooling_CIE` + `def_dir` non-CIE
   + `ZCloud ∈ {1.0, 0.15}` + `SB99_rotation`) produce snapshot trees
   byte-equivalent (within `rtol=1e-12`) to the PR-3 golden.
2. `cool_col_*`-based configs continue to work; full E2E equivalence
   against PR-3 golden.
3. `grep -n "metallicity" src/cooling/CIE/read_coolingcurve.py`
   returns only docstring references.
4. `grep -n "ZCloud" src/cooling/net_coolingcurve.py` returns nothing.

## 11. Test strategy (the substance)

This section is what makes the refactor safe. Everything else is
plumbing.

### 11.1 Golden capture protocol (DO THIS FIRST)

**One-time setup on `main`, before any refactor branch is cut.**

Anchor configs — same three single-runs as the SPS refactor:

- `param/cloud_example_PL.param`
- `param/cloud_example_BE.param`
- `param/cloud_example_homogeneous.param`

```bash
git checkout main
mkdir -p analysis/cooling-refactor-golden
python analysis/cooling_refactor_equivalence.py --capture-golden \
    --configs param/cloud_example_PL.param \
              param/cloud_example_BE.param \
              param/cloud_example_homogeneous.param \
    --golden-dir analysis/cooling-refactor-golden
```

The harness does **three** independent things per anchor:

1. **E2E capture (subprocess).** Copy the `.param` to a temp location,
   rewrite `path2output`, invoke `python run.py <tmp_param>`. Trinity
   writes the JSONL snapshot tree directly into the golden directory.
2. **Loader-layer capture (in-process).** Import
   `src.cooling.non_CIE.read_cloudy`; for each OPIATE age present in
   `lib/default/opiate/` for the resolved
   `(ZCloud, SB99_rotation)` combination, call
   `create_cubes(filename, path2cooling)` and pickle the 5-tuple
   `(log_ndens, log_T, log_phi, cool_cube, heat_cube)`. Also pickle
   the resolved CIE arrays from `np.loadtxt(path)`.
3. **Interpolator-layer capture (in-process).** Build the three
   non-CIE interpolators (cool, heat, net) and the one CIE
   interpolator; evaluate each at a dense grid covering its valid
   range; pickle the input/output pairs.

The golden tree must be **frozen for the entire refactor**. Record the
commit SHA of `main` at capture time in
`analysis/cooling-refactor-golden/MANIFEST.json`.

### 11.2 Per-PR test battery

A single Python script `cooling_refactor_equivalence.py` runs all
tests; each PR's gate is "run the script with `--pr N`, must exit 0".

#### 11.2.1 PR-1 tests

```python
def test_nonCIE_filename_resolution_matrix():
    """Every (rot, Z, age) combo resolves to the same on-disk file under the
    new helper as under the old _read_cloudy.get_filename grammar."""
    combos = [
        (rot, Z, age)
        for rot in (0, 1)
        for Z in (1.0, 0.15)
        for age in [1e6, 2e6, 3e6, 4e6, 5e6, 1e7]  # OPIATE-available ages
    ]
    for rot, Z, age in combos:
        legacy = legacy_opiate_filename(rot, Z, age)            # captured from main
        resolved = resolve_nonCIE_paths_via_fallback(rot, Z, age)  # new code
        assert legacy == resolved

def test_nonCIE_age_interpolation_unchanged():
    """For an age between OPIATE files, the helper returns [lower, higher]
    as before. The two-file branch is the one most likely to drift."""

def test_cube_byte_equivalence(equal_nan=True):
    """The 5-tuple loader output is byte-identical to the pickled golden,
    NaN-aware."""
    golden = pickle.load(open('analysis/cooling-refactor-golden/cubes.pkl', 'rb'))
    for (rot, Z, age), gold_tuple in golden.items():
        cur = create_cubes(*resolve_nonCIE_paths(rot, Z, age))
        for name, g, c in zip(CUBE_FIELDS, gold_tuple, cur):
            assert np.array_equal(g, c, equal_nan=True)

def test_no_SB99_reads_in_read_cloudy():
    """grep-style check: read_cloudy.py contains no 'SB99' reference
    after PR-1. The coupling is gone."""
    src = open('src/cooling/non_CIE/read_cloudy.py').read()
    assert 'SB99' not in src

def test_unsupported_ZCloud_explicit_error():
    """ZCloud = 0.5 under the legacy path raises ValueError, not NameError."""
    with pytest.raises(ValueError, match="ZCloud"):
        resolve_nonCIE_paths(rot=1, Z=0.5, age=2e6)

def test_single_file_user_mode_bypasses_age_dispatch():
    """path_cooling_nonCIE = /path/to/single_file.dat → loader skips
    age discovery and loads that file regardless of params['t_now']."""

def test_e2e_snapshot_equivalence():
    """Full trinity run; snapshot JSONL trees match within rtol=1e-12."""

def test_legacy_path_emits_one_info_log_per_run(caplog):
    """Legacy fallback emits exactly one INFO log naming the OPIATE grammar
    — informational, not a warning."""
```

#### 11.2.2 PR-2 tests

Mirror SPS PR-2 tests 1-15 (substituting `cool_col_nonCIE_*` for
`sps_col_*`), plus cube-cache invalidation, NaN-sparse round-trip, and
sign-convention declaration tests listed in PR-2 above.

#### 11.2.3 PR-3 tests

```python
def test_cie_integer_index_unchanged():
    """path_cooling_CIE = 3 + ZCloud = 1 resolves to the same path
    and same loaded arrays as PR-2 golden."""

def test_cie_path_string_works():
    """path_cooling_CIE = /tmp/cooling.dat (with declared cool_col_cie_*
    matching the legacy format) loads byte-identical arrays."""

def test_cie_out_of_range_index_errors():
    """path_cooling_CIE = 5 + ZCloud = 1 → explicit ValueError listing
    valid indices, not silent fall-through to int 5."""

def test_zcloud_unsupported_for_cie_errors():
    """ZCloud = 0.5 → explicit ValueError pointing at path_cooling_CIE
    as the escape hatch."""

def test_e2e_after_pr3():
    """E2E equivalence across three anchor configs against PR-2 golden."""
```

#### 11.2.4 PR-4 tests

```python
def test_legacy_config_still_works_after_pr4():
    """The 'last resort' guarantee. A config that sets only the legacy
    integer/dir params produces a snapshot tree within rtol=1e-12 of
    the PR-3 golden."""

def test_get_Lambda_signature_dropped_metallicity():
    """src/cooling/CIE/read_coolingcurve.py::get_Lambda has 2 params,
    not 3, after PR-4."""

def test_net_coolingcurve_drops_ZCloud():
    """grep -n 'ZCloud' src/cooling/net_coolingcurve.py returns nothing."""
```

### 11.3 Equivalence tolerance policy

Identical to SPS refactor §11.3 (`sb99-refactor-audit.md`), with one
addition: NaN-equality for the non-CIE cubes
(`np.array_equal(..., equal_nan=True)`). The cubes are NaN-initialized
and sparse-filled (`src/cooling/non_CIE/read_cloudy.py:224, 241`), so
default `np.array_equal` would report all-NaN positions as unequal.

| Layer | Comparator | Rationale |
|-------|------------|-----------|
| Resolved path string | `==` | strings, no ambiguity |
| Raw CIE arrays | `np.array_equal` | deterministic |
| Non-CIE cubes | `np.array_equal(equal_nan=True)` | NaN-sparse fill |
| Interpolators at sample inputs | `np.array_equal` | scipy deterministic on identical inputs |
| `get_dudt()` return | `np.array_equal` | downstream of byte-identical interpolators |
| JSONL snapshot fields | `np.allclose(rtol=1e-12, atol=0)` | I/O roundtrip noise |
| Linear-vs-log unit roundtrip (PR-2 only) | `np.allclose(rtol=4e-15, atol=0)` | `10**log10(x)` is not exactly `x` |

### 11.4 Harness layout

```
test/                                            # (existing pytest dir)
└── test_cooling_refactor_equivalence.py
analysis/
├── cooling-refactor-audit.md                    # this file
└── cooling-refactor-golden/                     # gitignored; pickled goldens + JSONL trees
    ├── MANIFEST.json                            # commit SHA of main at capture
    ├── cubes.pkl                                # 5-tuple per (rot, Z, age)
    ├── cie_arrays.pkl                           # (logT, logLambda) per legacy combo
    ├── interp_samples.pkl                       # dense-grid evaluations
    ├── cloud_example_PL/{1_begin.jsonl, ...}
    ├── cloud_example_BE/...
    └── cloud_example_homogeneous/...
```

Add `analysis/cooling-refactor-golden/` to `.gitignore`.

## 12. Risk register

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| Float drift from changing cube-fill iteration order | Low | High | PR-2 keeps the fill loop as-is; column-map dispatch only changes which file column maps to which axis, not the fill algorithm. Byte-equivalence test catches drift. |
| Path-resolution drift in the relocated OPIATE grammar (mantissa formatter `format(age, '.2e')` at `src/cooling/non_CIE/read_cloudy.py:310, 316, 321`) | Medium | High | Path-resolution matrix test (`test_nonCIE_filename_resolution_matrix`) covers all legal combos. |
| Cube-cache invalidation bug (axis hash misses a real difference) | Low | High | Hash the concatenated axis arrays, not the cube. Axis grids differ if the source file differs. Cube-cache test exercises overwrite-then-reload. |
| User mis-declares `log` vs `linear` on `Lambda_cool` (silent physics-altering bug) | Medium | High | Same class of risk as SPS PR-2; same mitigation — hard-error on unrecognized units, otherwise `.param` review is the line of defense. Open question §14 #4 — add an order-of-magnitude sanity check. |
| `cool_col_*` parser collides with the `cool_alpha/beta/delta` physics knobs (`src/_input/default.param:304-308`) | Low | Medium | Prefix is `cool_col_cie_` / `cool_col_nonCIE_`, not `cool_`. Adding a test that exercises a config setting both `cool_col_*` and `cool_alpha` confirms no collision. |
| Removing the silent if/elif fall-through breaks a config that was relying on the integer 3 staying integer | Low | Low | The fall-through is a bug (it makes `np.loadtxt(3, ...)` fail downstream); no real config can be relying on this. Explicit error is unambiguously better. |
| NaN-sparse cube comparison changes due to floating-point reordering of `cool - heat` | Low | Medium | `cool_cube - heat_cube` (`src/cooling/non_CIE/read_cloudy.py:130`) is element-wise; numpy is deterministic. Test compares NaN-aware. |
| OPIATE-folder mode broken by an empty folder or no `.dat` files | Low | Low | Add explicit `FileNotFoundError` with the folder path in the relocated helper. Already implicit today (line 309 `if age in age_list` on an empty array raises `ValueError`). |
| Cooling-update cadence (`COOLING_UPDATE_INTERVAL` differs between phase1 and phase1b: `5e-2 Myr` vs `5e-3 Myr`) introduces drift if a refactor accidentally moves the call site | Low | Medium | Out of scope, but: golden tree was captured with the existing cadences. Any PR moving the call site triggers the E2E test. |
| `scipy.interpolate.RegularGridInterpolator` future deprecation | Low | Low | Pin scipy in repo deps. Out of scope. |

## 13. Rollout sequence

1. **Capture golden** on `main` (§11.1). Verify the harness can compare
   a freshly captured tree to itself with zero drift (smoke test).
2. **Branch `feature/cooling-nonCIE-decouple`** for PR-1. Run battery; merge
   when green.
3. **Branch `feature/cooling-column-mapping-nonCIE`** for PR-2. Re-run full
   battery (includes cube-cache and NaN-sparse tests).
4. **Branch `feature/cooling-column-mapping-cie`** for PR-3.
5. **Soak.** Let PR-3 run in production for at least one release cycle
   before queueing PR-4. PR-4 is purely cosmetic; soaking catches
   unexpected coupling to the unused `metallicity` arg.
6. **Branch `fix/cooling-drop-unused-metallicity`** for PR-4. Re-run battery
   against the post-PR-3 golden. **Legacy presets remain permanent** —
   see §9.

All branch names use the repo's `feature/` or `fix/` prefix per CLAUDE.md.

## 14. Open questions for the user

Before starting PR-1, please confirm:

1. **Cube-cache invalidation strategy.** Hash on axis arrays (proposed
   here) is cheap but doesn't catch a case where the source file
   changes its data values but not its axes. Stronger options: hash on
   the full file bytes (slow on large files), or hash on the
   `(cool_cube, heat_cube)` themselves before save (correct but
   tautological — the hash IS the cube). Recommendation: axis hash now,
   revisit if the value-only-change case actually bites someone.
2. **CIE path-vs-int detection rule.** Proposed: "if the value can be
   `int()`'d, treat as legacy index; else treat as path." Edge case: a
   user with a file literally named `3.dat` in CWD. Alternative: a
   separate `path_cooling_CIE_path` key, with the integer in
   `path_cooling_CIE`. Recommendation: detection by `os.path.exists` or
   "starts with `.` or `/`" — accept some heuristic, document it.
3. **Anchor configs adequacy.** The three `cloud_example_*` configs
   don't set `ZCloud` explicitly and therefore inherit `ZCloud = 1`
   from `src/_input/default.param:85`. None of them exercise the
   `ZCloud = 0.15` Sutherland-Dopita CIE branch or the `Z0002` OPIATE
   non-CIE branch. We need at least one anchor with `ZCloud = 0.15` to
   cover the subsolar legacy paths. Suggest adding a
   `cloud_example_subsolar.param` derived from one of the three with
   only `ZCloud` flipped (and verifying the requisite `Z0002` OPIATE
   cubes exist in `lib/default/opiate/`).
4. **Order-of-magnitude sniff-test for `Lambda_cool` / `Lambda_heat`?**
   Same recommendation as SPS PR-2 open question #7 — not in PR-2;
   revisit after first real user.
5. **Shared canonical/unit infrastructure.** `src/cooling/cool_columns.py`
   would duplicate ~80% of `src/sb99/sps_columns.py`'s structure
   (`ColumnSpec`, unit-table dispatch, `_scan_layout`, `load_user_columns`).
   Two options: (a) extract a shared `src/_input/column_map_common.py`
   that both modules import; (b) duplicate. Recommendation: (b)
   duplicate now to avoid premature abstraction; extract only after the
   third loader needs it (none currently planned).
6. **`Lmech_total` analog for non-CIE?** SPS PR-2 has derived
   canonicals (`Lmech_total = Lmech_W + Lmech_SN`, etc.). Cooling
   currently has none — `cool` and `heat` are independent file
   columns; `net_cooling` is always derived as `cool - heat`. Should
   `Lambda_net` be addable as a canonical for files that supply it
   directly? Recommendation: not now — the current `cool - heat` model
   is universal and the saving is one column.
7. **Folder vs single-file detection for `path_cooling_nonCIE`.**
   Proposed: `os.path.isdir → legacy folder mode; os.path.isfile →
   user single-file mode`. What if neither? Hard error, naming the
   path. Confirm this is the right dispatch.

---

## Appendix — Working notes

- **Branch.** This audit was drafted on
  `claude/cooling-tables-modularity-P7BNE` (named per session
  provisioning; the actual PRs would each use the `feature/` or `fix/`
  prefix per CLAUDE.md).
- **Snapshot exclusion.** All six `cStruc_*` entries are
  `exclude_from_snapshot=True` (`src/_input/read_param.py:626-631`).
- **Footprint.** Three Python files contain non-trivial loader code:
  - `src/cooling/non_CIE/read_cloudy.py` (361 LOC) — the big one.
  - `src/cooling/CIE/read_coolingcurve.py` (75 LOC) — trivial.
  - `src/cooling/net_coolingcurve.py` (166 LOC) — consumer; touches
    constants only.
- **Default cooling library paths.** `lib/default/opiate/` and
  `lib/default/CIE/*.dat` (only the `lib/default/` subtree is
  un-ignored by `.gitignore`; the rest of `lib/` is git-ignored).
- **Cooling-update cadence asymmetry.** Phase 1 uses `5e-2 Myr`
  (`src/phase1_energy/run_energy_phase_modified.py:56`); phase 1b
  implicit uses `5e-3 Myr`
  (`src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py:108`).
  Intentional or accidental? Flagged for review but out of scope.
- **Logging interface.** `logger_cooling =
  logging.getLogger('src.cooling.net_coolingcurve')` already exists at
  `src/_functions/logging_setup.py:514`. The relocated OPIATE
  resolution helper in `read_param.py` should log under its existing
  `src._input.read_param` logger.
- **The `ZCloud` parameter is shared across SPS, cooling, dust opacity,
  and other physics.** Like in the SPS refactor (§9 of that audit), it
  is **not** affected by this refactor — only the cooling-table-keyed
  uses of it are decoupled.
- **`SB99_rotation`'s home after PR-1.** It stops being read by
  `src/cooling/non_CIE/read_cloudy.py` but is still read by the
  relocated OPIATE-grammar helper in `read_param.py` (legacy fallback).
  It also continues to be read by the SPS loader. The parameter
  remains a first-class config knob.
- **No `read_cloudy_old.py` exists** despite the reference at
  `analysis/sb99-refactor-audit.md:234`. That doc has a stale
  reference; this audit does not reproduce it.
