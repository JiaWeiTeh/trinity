# SB99 → generic SPS refactor: audit + implementation plan

Single source of truth. Combines (Part I) the architectural audit — *what is* —
with (Part II) the phased refactor plan and its equivalence-test battery —
*what to do, in what order, and how to prove nothing changed*.

End goal: replace the hardcoded SB99 filename construction with a single
`sps_path` (and friends) so arbitrary stellar-population-synthesis CSVs can be
dropped in, without shifting a single ULP under the legacy parameter path.

## TL;DR

- After `start_expansion()` populates `params['SB99f']` and
  `params['SB99_data']`, **every downstream consumer goes through exactly one
  function**: `get_currentSB99feedback(t, params)`. No phase code constructs
  filenames or indexes raw columns. So the refactor concentrates in the
  loader (`read_SB99.py`), the param plumbing (`read_param.py` +
  `default.param`), and the one direct interpolator access in
  `phase0_init/get_InitPhaseParam.py`.
- **Four PRs**, each back-compat and independently revertable:
  1. `sps_path` + `sps_refmass` with legacy fallback (zero math changes)
  2. In-`.param` column mapping via `sps_col_<canonical>` (SB99 positional remains the legacy preset; strict, no silent fallback; subsumes explicit SN/Li/Ln overrides as the natural consequence of the column-map design)
  3. Mechanical rename `SB99f → sps_f`
  4. Cleanup — drop dead import; **legacy SB99 params (SB99_mass / SB99_rotation / SB99_BHCUT) remain as permanent fallback, never removed**.
- **Headline risk: silent ULP drift** through the 10 cubic-spline interpolators
  that drive every phase. The whole test battery exists to make that drift
  impossible to ship undetected.
- **Two equivalence guarantees** enforced on every PR:
  1. Bitwise (`np.array_equal`) at loader / interpolator / dataclass layers
  2. Tight-tolerance (`rtol=1e-12, atol=0`) snapshot-tree equivalence against
     a golden captured ONCE on main before PR-1.
- **Cooling-table coupling in `cooling/non_CIE/read_cloudy.py` is out of
  scope.** Even after the feedback CSV is decoupled, the cooling cubes were
  generated from SB99-keyed SEDs. Tracked as follow-up.

---

# Part I — Audit (what is)

## 1. Current architecture

`main.start_expansion()` calls `read_SB99.read_SB99(f_mass, params)` which (a)
builds a filename from `SB99_mass / SB99_rotation / ZCloud / SB99_BHCUT`, (b)
loads a fixed 7-column text file, (c) converts log/cgs→AU, and (d) returns 11
numpy arrays. Those go through `get_interpolation()` which produces a dict of
10 scipy `interp1d` callables stored as `params['SB99f'].value`. Every physics
phase imports `get_currentSB99feedback(t, params)` from
`src/sb99/update_feedback.py`, which evaluates those interpolants at `t` and
returns an `SB99Feedback` dataclass while also writing each value back into
`params[…]` (so the rest of the code can read either path).

## 2. Configuration surface

| Param | File:line | Default | Role |
|-------|-----------|---------|------|
| `path_sps` | `src/_input/default.param:306` | `def_dir` | Directory for SPS data. |
| `SB99_mass` | `default.param:169` | `1e6` | Reference cluster mass; sets `f_mass = mCluster / SB99_mass`. |
| `SB99_rotation` | `default.param:172` | `1` | Becomes `rot` / `norot` in filename. |
| `SB99_BHCUT` | `default.param:176` | `120` | Becomes `BH120` / `BH40` in filename. |
| `ZCloud` | `default.param:85` | `1` | Becomes `Z0014` / `Z0002` in filename. Also drives dust opacity and other metallicity-keyed physics — **not deprecated by this refactor.** |
| `FB_mColdWindFrac`, `FB_mColdSNFrac`, `FB_thermCoeffWind`, `FB_thermCoeffSN`, `FB_vSN` | `default.param:179-192` | `0, 0, 1, 1, 1e4` | Post-load corrections; orthogonal to file selection. |

The filename comes out as `{1e6cluster}_{rot|norot}_{Z0014|Z0002}_{BH120|BH40}.txt`
(`read_SB99.py:368`). Mantissa formatting via nested `format_e()` at
`read_SB99.py:328-333`.

## 3. Interface every consumer expects

After loading + interpolation, downstream code needs these 10 interpolants on
`params['SB99f'].value` (all in AU units = Msun·pc²/Myr³ except `Qi` in 1/Myr):

```
fQi, fLi, fLn, fLbol,
fLmech_W, fLmech_SN, fLmech_total,
fpdot_W, fpdot_SN, fpdot_total
```

…plus `params['SB99_data'].value` holds the 11-array raw cube (`[t, Qi, Li, Ln,
Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]`,
returned at `read_SB99.py:281`).

`get_currentSB99feedback()` (`update_feedback.py:98`) then writes these
12 scalars into `params`: `Lmech_W, Lmech_SN, Lmech_total, v_mech_total,
pdot_W, pdot_SN, pdot_total, pdotdot_total, Qi, Lbol, Ln, Li`
(`read_param.py:474-485`).

## 4. SB99-specific hot spots (what blocks generic CSVs)

| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 1 | `read_SB99.py:285-372` (`get_filename`) | Hardcoded filename grammar; only 2 Z, 2 BH, 2 rotation modes. | Replace with `sps_path` param; move `get_filename` to a legacy-fallback helper in `read_param.py`, then delete in PR-4. |
| 2 | `read_SB99.py:144-149` | Validates `>=7` columns by position. | PR-2: when `sps_path` is user-defined, replace with mandatory `sps_col_<canonical>` declarations in `.param`. Positional ≥7-column check stays for the legacy fallback. |
| 3 | `read_SB99.py:158-177` | Hardcoded log-space + cgs assumptions per column. | PR-2: per-column units and log/linear declared in `.param` via `sps_col_<canonical>  <file_column>  <units>  <log\|linear>`. |
| 4 | `read_SB99.py:229-247` | Derives `Lmech_SN` and `pdot_SN` from totals + `FB_vSN`. | PR-2: optional `Lmech_SN`, `pdot_SN`, `Mdot_SN`, `v_SN` columns in the canonical vocabulary; when supplied via `sps_col_*`, skip the derivation. |
| 5 | `read_SB99.py:191` | Ionizing fraction at 13.6 eV hardcoded (in SB99's `fi` definition; loader does `Li = Lbol·fi`, `Ln = Lbol·(1−fi)`). | PR-2: optional `Li`/`Ln` columns in the canonical vocabulary; when both supplied via `sps_col_*`, skip the derivation and bypass the threshold entirely. |
| 6 | `main.py:142` | `f_mass = mCluster / SB99_mass` couples to reference mass. | Add `sps_refmass` param (defaults to `SB99_mass` for back-compat). |
| 7 | `update_feedback.py:187` | Hardcoded `Δt=1e-9 Myr` for `pdotdot_total`. | Constant is fine; flag for review only. |

## 5. Consumer-by-consumer deep dive

For each consumer the audit captures: **(a)** what it actually reads from
SB99, **(b)** whether it touches filename/grammar-specific code, **(c)** what
breaks/changes when SB99 is swapped for a user-supplied CSV behind
`sps_path`. Verified by reading the files.

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
  exact keys listed in §3, plus 12 scalar params written by `updateDict`.

That means **every consumer in §5.3–§5.8 is transparent to the refactor** as
long as the new loader produces the same 10 interpolators under the same
keys. The work concentrates in §5.1, §5.2, §5.10, and the cooling coupling in
§5.11.

### 5.1 `src/sb99/read_SB99.py` — 🔴 loader (the main refactor target)

**What it does today.**

| Step | Lines | Action |
|------|-------|--------|
| Validate params | 104-118 | Requires `path_sps, SB99_rotation, ZCloud, SB99_BHCUT, FB_mColdWindFrac, FB_thermCoeffWind, FB_mColdSNFrac, FB_thermCoeffSN, FB_vSN`. |
| Build filename | 125-127 (→ `get_filename` 285-372) | `filename = get_filename(params); path2sps = params['path_sps']; filepath = path2sps + filename`. Hardcoded `{mass}cluster_{rot\|norot}_{Z0014\|Z0002}_{BH120\|BH40}.txt`. Whitelists ZCloud ∈ {1.0, 0.15} and SB99_BHCUT ∈ {120, 40}. |
| Load | 132, validate ≥7 cols at 145 | `np.loadtxt` only — no header. |
| Unit conversion | 158-177 | Columns are positional: 0=t [yr], 1=log Qi, 2=log fi, 3=log Lbol [erg/s], 4=log Lmech_tot [erg/s], 5=log pdot_W [g·cm/s²], 6=log Lmech_W [erg/s]. All log-space except t and fi. |
| Derived (Li, Ln) | 191-192 | `Li = Lbol·fi`, `Ln = Lbol·(1-fi)`; uses hardcoded 13.6 eV threshold. |
| Derive Lmech_SN | 195 | `Lmech_SN_raw = Lmech - Lmech_wind_raw` (subtraction, not from file). |
| Wind corrections | 212-224 | `Mdot_W = pdot²/(2·Lmech_W)`; rescale by `FB_mColdWindFrac`, `FB_thermCoeffWind`. |
| SN corrections | 229-247 | `velocity_SN = FB_vSN.value` (constant); `Mdot_SN = 2·Lmech_SN/v_SN²`; rescale by `FB_mColdSNFrac`, `FB_thermCoeffSN`. |
| Totals | 254-255 | `Lmech_total`, `pdot_total = wind + SN`. |
| Prepend t=0 | 262-275 | All arrays get `np.insert(..., 0, arr[0])` so interpolators are defined down to t=0. |
| Interp factory | 375-460 | Wraps 10 arrays in `scipy.interpolate.interp1d(kind='cubic')`. |

**Breaks with generic CSV.**

- Hardcoded filename: a generic CSV will not match this grammar.
- Hardcoded column order + log-space + cgs: a generic CSV may have a header,
  may already be linear-space, may use different units.
- `Lmech_SN` is *derived* from `(Lmech - Lmech_wind_raw)`. A modern SPS code
  could provide SN directly.
- `velocity_SN = FB_vSN` is a user-supplied constant; generic SPS may carry
  SN velocity over time → support a column.

### 5.2 `src/sb99/update_feedback.py` — 🟡 query function (rename only)

`get_currentSB99feedback(t, params)` (lines 98-205) reads 10 interpolators
from `params['SB99f'].value`, validates `t ∈ [t_min, t_max]` (156-157),
evaluates each, computes `v_mech_total = 2·Lmech_total/pdot_total` (184) and
`pdotdot_total` by central difference at `dt=1e-9 Myr` (187-188), returns
`SB99Feedback` dataclass (defined at line 21).

🟡 Optional rename in PR-3: `SB99f → sps_f`, `SB99Feedback → SPSFeedback`,
`get_currentSB99feedback → get_current_sps_feedback`. No functional change.

### 5.3 `src/phase0_init/get_InitPhaseParam.py` — 🟡

Lines 88, 111-112 read `params['SB99f'].value` and call
`SB99f['fLmech_W'](tSF)`, `SB99f['fpdot_W'](tSF)` directly (not via
`get_currentSB99feedback`). The **one** phase file that touches the
interpolator dict directly. 🟡 rename only — or refactor to use
`get_currentSB99feedback`.

### 5.4 `src/phase1_energy/` — 🟢

Call sites: `run_energy_phase_modified.py:92, 158, 358, 400`;
`energy_phase_ODEs_modified.py:189, 324`. Pattern:
`feedback = get_currentSB99feedback(t, params); updateDict(params,
feedback)`. Reads `Lmech_total`, `v_mech_total` off the dataclass.

### 5.5 `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` — 🟢

Lines 68, 555, 917, 1001. Same pattern.

### 5.6 `src/phase1c_transition/run_transition_phase_modified.py` — 🟢

Lines 57, 472, 770, 854. Same pattern.

### 5.7 `src/phase2_momentum/run_momentum_phase_modified.py` — 🟢

Lines 58, 405, 552, 906. Same pattern.

### 5.8 `src/bubble_structure/bubble_luminosity_modified.py` — 🟢

Line 33 imports `get_currentSB99feedback` but **never calls it**. Reads
`params['Lmech_total']`, `params['v_mech_total']`, `params['Qi']` directly
(lines 104-105, 312, 345, 795). No interpolator access. PR-4 drops the dead
import.

### 5.9 `src/_output/cloudy/` and `src/_plots/` — 🟢 (orthogonal)

- `snapshot_to_deck.py:166-177` — SB99 age-band check for CLOUDY deck
  generation. Unrelated to feedback CSV loader.
- `trinity_to_cloudy.py:71-130, 364, 454-458` — `{{SB99_MOD}}` sentinel for
  the CLOUDY-compiled SB99 *atmosphere grid* (separate concept).
- `_plots/paper_*.py` — comments/docstrings only.

### 5.10 `src/_input/read_param.py` and `default.param` — 🔴 plumbing

- `default.param:163-176` — declares `SB99_mass, SB99_rotation, SB99_BHCUT`.
- `default.param:306` — declares `path_sps`.
- `read_param.py:377-383` — resolves `path_sps` (`def_dir` →
  `lib/sps/starburst99/`).
- `read_param.py:472-473` — declares `SB99_data, SB99f` runtime containers.
- `read_param.py:474-485` — declares the 12 scalar feedback params.

**Required changes.** Add `sps_path` (and `sps_refmass`); keep `SB99_*` as
permanent fallback paths (§9); rename runtime containers in PR-3.

### 5.11 `src/cooling/non_CIE/read_cloudy.py` — 🔴 *separate* SB99 coupling

This file *also* uses `SB99_rotation` and `ZCloud` to construct cooling-table
filenames:

- Line 47: reads `params['SB99_rotation'].value`.
- Line 263-335: `get_filename(age, metallicity, SB99_rotation, path2cooling)` → `opiate_cooling_{rot|norot}_Z{1.00|0.15}_age{age}.dat`.

Non-CIE cooling cubes were generated by running CLOUDY with SB99-based SEDs
at specific cluster ages — so they're tied to the SB99 rotation/Z choice. If
a user supplies a generic SPS CSV with different metallicity, the cooling
tables would need their own path indirection. **Out of scope for this
refactor**; documented in §12 with a `UserWarning` mitigation. Same coupling
in legacy `read_cloudy_old.py:287`.

## 6. Per-consumer change matrix

| File | Touches filename? | Touches interpolators? | Touches params dict only? | Change needed |
|------|:--:|:--:|:--:|--|
| `src/sb99/read_SB99.py` | ✅ | builds them | — | 🔴 rewrite with column map; move `get_filename()` to legacy fallback |
| `src/sb99/update_feedback.py` | — | ✅ reads | writes 12 scalars | 🟡 PR-3 rename |
| `src/main.py:142-152` | — | calls `read_SB99` | populates containers | 🔴 swap loader; replace `f_mass = mCluster/SB99_mass` with `sps_refmass` |
| `src/_input/read_param.py` | declares containers | — | — | 🔴 add `sps_path`/`sps_refmass`; keep legacy as fallback |
| `src/_input/default.param` | declares legacy params | — | — | 🔴 add `sps_path`; point INFO line at `sps_path` as alternative. SB99_* params stay (permanent fallback, §9). |
| `src/phase0_init/get_InitPhaseParam.py` | — | ✅ reads directly | — | 🟡 PR-3 rename |
| `src/phase1_energy/run_energy_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1_energy/energy_phase_ODEs_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase1c_transition/run_transition_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/phase2_momentum/run_momentum_phase_modified.py` | — | — | ✅ | 🟢 |
| `src/bubble_structure/bubble_luminosity_modified.py` | — | — | ✅ (and dead import) | 🟢 (PR-4 drops import) |
| `src/cooling/non_CIE/read_cloudy.py` | ✅ separate filename | — | — | 🔴 own indirection (out of scope) |
| `src/_output/cloudy/*` | — | — | — | 🟢 (orthogonal) |
| `src/_plots/*` | — | — | — | 🟢 (comments only) |
| `src/_input/dictionary.py:1203-1234` | — | — | demo `__main__` only | 🟢 |

---

# Part II — Plan (what to do)

## 7. Goals and non-goals

### Goals

1. Replace the hardcoded SB99 filename grammar with a single `sps_path`
   parameter so users can drop in arbitrary SPS CSVs.
2. Decouple `f_mass`'s reference mass from `SB99_mass` (add `sps_refmass`).
3. Allow per-column declarations of file-column name, units, and log/linear
   convention inside `.param` (via `sps_col_<canonical>` keys), so CSV
   column order / units / log convention are no longer hardcoded. **No
   external sidecar file** — the `.param` is the single source of truth.
4. Allow optional explicit SN columns (`Lmech_SN`, `pdot_SN`, `Mdot_SN`,
   `v_SN`) and direct ionizing/non-ionizing splits (`Li`, `Ln`) for SPS
   codes that provide them, removing SB99's hardcoded 13.6 eV ionizing
   threshold.
5. Make every step **byte-equivalent** to the current SB99 path when run with
   legacy parameters.

### Non-goals

- Not retiring SB99 as a data source. SB99 stays the default; this just
  removes the hardcoding around it.
- Not refactoring `update_feedback.py`'s numerical logic. The central
  difference, `v_mech_total` formula, and dataclass shape are unchanged.
- Not touching the cooling-table coupling (`read_cloudy.py`). That is a
  *different* SB99 dependency with its own indirection problem. Mitigation:
  emit a `UserWarning` when `sps_path` is set to a non-default value
  reminding users the cooling cubes are still SB99-keyed at the declared
  rotation/Z.

## 8. Invariants (what MUST stay byte-identical under the legacy code path)

The legacy code path is: user config sets `SB99_mass / SB99_rotation /
SB99_BHCUT / ZCloud` (the existing way), does **not** set `sps_path` or
`sps_refmass`. The refactor adds new params with `def_path` / `def_value`
sentinels that fall through to the legacy grammar.

Under that fallback, the following must remain byte-identical PR-by-PR:

| Invariant | How verified |
|-----------|--------------|
| Resolved on-disk path | string equality |
| Raw 11-array loader output (`[t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]`, `read_SB99.py:281`) | `np.array_equal` element-wise |
| The 10 `params['SB99f']` interpolators evaluated at any t | `np.array_equal` after pickling and reloading |
| `SB99Feedback` dataclass fields from `get_currentSB99feedback(t, params)` | `np.array_equal` per field |
| The 12 scalar params written by `updateDict(params, feedback)` | `np.array_equal` |
| Every JSONL snapshot value in a full trinity run | `np.allclose(rtol=1e-12, atol=0)` per field |

Why `rtol=1e-12` and not bitwise for E2E? Because there's I/O serialization
(`jsonl`) in between. JSONL writes floats via `repr()` or similar; round-trip
through string can introduce ULP-level noise. `1e-12` is well below any
physically meaningful tolerance and well above JSONL round-trip noise.

If a PR breaks any of these invariants, **the PR is broken**, not the tests.

## 9. Migration strategy + legacy-as-last-resort guarantee

Three new params (well, two scalars and one family), each with a sentinel
default that routes back to the legacy SB99 grammar:

- `sps_path` — full path to an SPS CSV. Default `def_path`.
  - When `def_path` and `SB99_mass / SB99_rotation / SB99_BHCUT / ZCloud`
    are set: construct path from legacy grammar (existing `get_filename()`
    logic relocated to `read_param.py`). Emit one `logger.info` line at
    startup naming the legacy params in use — informational, not a
    warning. Legacy users see *zero* friction.
  - When set to anything else: use that path verbatim. Skip legacy
    grammar. The user is now also required to declare the column map via
    the `sps_col_*` family below.
- `sps_refmass` — reference cluster mass for `f_mass = mCluster /
  sps_refmass`. Default `def_value`.
  - When `def_value`: copy `params['SB99_mass'].value`.
  - Else: use directly.
- `sps_col_<canonical>` family — one line per canonical column the user is
  declaring, with positional fields `<file_column>  <units>  <log|linear>`
  (full syntax in §10 PR-2). Required only when `sps_path` is
  user-defined; entirely ignored under the legacy fallback. PR-2
  introduces these keys; subsequent PRs do not touch them. **The `.param`
  is the single source of truth** — no external sidecar file is read.

### Legacy is permanent, not deprecated

**Hard guarantee:** the legacy SB99 calling method — a `.param` file that
declares only `SB99_mass`, `SB99_rotation`, `SB99_BHCUT`, `ZCloud`, and
`path_sps` (no `sps_path`, no `sps_col_*`) — works forever. **None of
the four PRs removes it.** This is the "last resort" path the user
called out: if everything else fails or a user simply doesn't want to
migrate, the existing grammar continues to function identically to its
pre-refactor behavior, byte-for-byte.

Concretely, after all four PRs land:

- `SB99_mass / SB99_rotation / SB99_BHCUT` remain declared in
  `default.param` with their original defaults.
- `read_param.py` continues to translate them into a resolved file path
  via the relocated `get_filename()` helper when `sps_path == def_path`.
- The legacy 7-column positional load path in `read_SB99.py` remains
  the code that runs in that case (byte-equivalent to PR-1's loader).
- The startup `logger.info` notification is informational; no warning,
  no error, no escalation. A user who never sets `sps_path` sees one
  extra line in the log and nothing else changes.

`ZCloud` is **not** affected at all by the SPS refactor (legacy or new).
It controls dust opacity and other metallicity-keyed physics elsewhere
(`read_param.py:280, 321, 363, 372`) and is required regardless of which
SPS path the user takes.

## 10. PR sequence

Each PR is independently mergeable. Order matters; do not reorder without
re-running the equivalence battery between every reordering.

### PR-1 — `sps_path` and `sps_refmass` with legacy fallback

**Scope.** Add the two new params and the resolution logic. The loader,
phases, interpolators, and dataclass are otherwise untouched.

**Files touched.**

- `src/_input/default.param` — add `sps_path  def_path` and
  `sps_refmass  def_value`. Update `# INFO` lines on the three legacy SB99_*
  params (169, 172, 176) to point at `sps_path` as the alternative
  mechanism. Do **not** mark them deprecated — they are permanent (§9).
- `src/_input/read_param.py` — new resolution block. If
  `sps_path == 'def_path'`, construct via the legacy grammar (relocate
  `get_filename` logic here). Emit one `logger.info` line stating that
  the legacy SB99 grammar is in use. Same shape for `sps_refmass`.
- `src/sb99/read_SB99.py` — change `read_SB99.py:125-127` (currently
  `filename = get_filename(params); path2sps = params['path_sps']; filepath
  = path2sps + filename`) to a single `filepath = params['sps_path'].value`.
  Delete `get_filename()` from this module (now lives in `read_param.py`).
  Drop the `path_sps / SB99_rotation / ZCloud / SB99_BHCUT` requirements
  from the validation block (`read_SB99.py:109-118`).
- `src/main.py:142` — change to `f_mass = params['mCluster'] /
  params['sps_refmass']`.

**Code-level checklist.**

- [ ] `sps_path`/`sps_refmass` declared in `default.param` with sentinel
      defaults.
- [ ] Resolution block in `read_param.py` constructs identical path string to
      old `path_sps + get_filename(params)` for every legacy combination.
- [ ] Startup `logger.info` line ("Using legacy SB99 parameter grammar
      …") fires exactly once per run (use a module-level guard, not
      per-call).
- [ ] `read_SB99.read_SB99` reads only `params['sps_path']` and
      `params['FB_*']` after this PR. No `SB99_rotation/ZCloud/BHCUT` reads
      remain in the loader.
- [ ] `main.py:142` uses `sps_refmass`.
- [ ] No phase code changed.
- [ ] `bubble_luminosity_modified.py:33` dead import left alone (PR-4
      cleanup).

**Tests required to merge.** See §11.2.1 for the full battery. Headline:

1. Path-resolution matrix: 2 rotations × 2 Zs × 2 BHCUTs × 4 masses = 32
   legacy combinations. All produce string-identical paths.
2. Loader byte-equivalence against pickled pre-refactor arrays.
3. E2E snapshot-tree equivalence for the canonical configs under legacy
   params.
4. E2E snapshot-tree equivalence with `sps_path` set explicitly to the path
   the legacy fallback would have resolved to. Must equal (3).

### PR-2 — In-`.param` column mapping (strict, no silent fallback)

**Scope.** Add the ability to read a CSV with arbitrary column names and
per-column units when `sps_path` is user-defined, via mandatory
`sps_col_<canonical>` declarations in `.param`. SB99's 7-column
headerless positional layout remains the legacy fallback used when
`sps_path` is the `def_path` sentinel — entirely back-compat.

The contract is **strict-by-default with no silent fallback**: when
`sps_path` is set but required `sps_col_*` keys are missing, the loader
hard-errors with a fillable template printed to stderr. That's what
"shouts and tells the user to edit" — but it's an error, not a warning,
because a warning the run ploughs past is exactly the silent-failure
mode strictness exists to prevent.

**`.param` syntax.** One line per canonical column, three positional
fields after the key:

```
sps_col_<canonical>    <file_column>    <units>    <log|linear>
```

- `<canonical>` is one of the names in the table below.
- `<file_column>` is the column name exactly as it appears in the SPS
  file's header row.
- `<units>` is a string from the recognized set: `yr`, `Myr`, `s`,
  `erg/s`, `L_sun`, `1/s`, `1/Myr`, `g*cm/s^2`, `Msun*pc/Myr^2`, `cm/s`,
  `pc/Myr`, `g/s`, `Msun/Myr`, `dimensionless`. Anything else is a hard
  error.
- `<log|linear>` declares whether the file column is in log10 space.

Worked example for the user-supplied SPS file shown in discussion:

```
sps_path    /path/to/your_sps_file.csv

sps_col_t            time         yr          linear
sps_col_Lbol         l_bol        erg/s       log
sps_col_Lmech_W      l_wind       erg/s       log
sps_col_Lmech_SN     l_sn         erg/s       log
sps_col_Qi           Qilog        1/s         log
sps_col_pdot_W       pd_windmom   g*cm/s^2    log
sps_col_Li           l_ion        erg/s       log
sps_col_Ln           l_non_ion    erg/s       log
```

Note the file's `l_HI`, `l_HeI`, `l_HeII` sub-columns are unmapped and
simply ignored — there is no canonical name for a finer ionization
breakdown.

**Files touched.**

- `src/_input/default.param` — add a commented documentation block
  showing the `sps_col_*` declaration syntax and listing recognized
  units. Inactive by default since `sps_path` defaults to `def_path`.
- `src/_input/read_param.py` — parse `sps_col_*` lines into a structured
  column-mapping dict on `params['sps_column_map']` (a single
  `DescribedItem` whose value is `{canonical_name:
  ColumnSpec(file_column, units, log)}`). Validate that all required
  canonical names are present when `sps_path != def_path`; build and
  emit the error template (see below) if not.
- `src/sb99/read_SB99.py` — refactor the body so it operates on a
  `column_map` regardless of source:
  - Legacy fallback (`sps_path == def_path`) → use a hardcoded SB99
    positional preset `column_map` (constructed once, byte-equivalent
    to PR-1's loader).
  - User-defined `sps_path` → take `column_map` from
    `params['sps_column_map']`. Load the CSV with `np.genfromtxt(...,
    names=True)` (i.e. read the header), select named columns,
    exponentiate log columns, convert units to canonical AU, and apply
    mass scaling (see below) before passing to the existing FB_*
    correction logic.
- (Module/symbol renames deferred to PR-3.)

**Canonical columns.** Required = no derivation fallback (loader errors
without them when `sps_path` is user-defined). Optional = derivation is
the fallback when absent. For `Li`/`Ln`: the `fi`-based derivation from
`Lbol` runs only if both are absent. When both are present the hardcoded
13.6 eV threshold built into SB99's `fi` is bypassed, closing §4
hot-spot #5.

| Canonical | Required? | Canonical linear unit | Mass-scaled? | Derivation if absent |
|-----------|-----------|------------------------|--------------|----------------------|
| `t`           | yes | yr        | no  | — |
| `Lbol`        | yes | erg/s     | yes | — |
| `Lmech_W`     | yes | erg/s     | yes | — |
| `Qi`          | yes | 1/s       | yes | — |
| `pdot_W`      | yes | g·cm/s²   | yes | — |
| `fi`          | yes, unless both `Li` and `Ln` are present | dimensionless | no | — |
| `Lmech_total` | no  | erg/s     | yes | `Lmech_W + Lmech_SN` |
| `Lmech_SN`    | no  | erg/s     | yes | `Lmech_total − Lmech_W` |
| `pdot_SN`     | no  | g·cm/s²   | yes | `Mdot_SN · v_SN` |
| `Mdot_SN`     | no  | g/s       | yes | `2·Lmech_SN/v_SN²` |
| `v_SN`        | no  | cm/s      | no  | `FB_vSN` constant |
| `Li`          | no  | erg/s     | yes | `Lbol · fi` |
| `Ln`          | no  | erg/s     | yes | `Lbol · (1 − fi)` |

Mass scaling (multiply by `f_mass = mCluster / sps_refmass`) is applied
post-load by the loader, hardcoded against the "Mass-scaled?" column
above. The user does **not** declare it per-column — they'd have no way
to know which canonicals are conventionally normalized to the reference
mass.

**Behaviour matrix.**

| `sps_path` value | `sps_col_*` keys present? | Behavior |
|------------------|----------------------------|----------|
| `def_path` (sentinel) | n/a (ignored) | Legacy SB99 fallback. Loader uses the hardcoded 7-column positional preset. Byte-equivalent to PR-1. |
| user-defined         | none                          | Hard error. Prints the fillable template (below) to stderr; exits non-zero. |
| user-defined         | partial (some required missing) | Hard error. Names which required canonicals are missing. Includes the file's actual header columns to make filling-in obvious. |
| user-defined         | `Li` alone or `Ln` alone        | Hard error: "supply both `Li` and `Ln`, or neither". Avoids partial overrides that silently disagree with `fi`. |
| user-defined         | complete                        | Use the user-declared mapping. |

**Error template (printed verbatim on missing-mapping error).**

```
ERROR: sps_path is set to '<resolved-path>' but the column mapping is
       incomplete.

       Missing required canonical columns: <list>

       Add the following lines to your .param file, filling in the file
       column names and unit/log declarations to match your SPS file.
       Each line is:
           sps_col_<canonical>    <file_column>    <units>    <log|linear>

       Required (no derivation fallback):
           sps_col_t            <file_column>     yr                  linear
           sps_col_Lbol         <file_column>     erg/s               log
           sps_col_Lmech_W      <file_column>     erg/s               log
           sps_col_Qi           <file_column>     1/s                 log
           sps_col_pdot_W       <file_column>     g*cm/s^2            log
           sps_col_fi           <file_column>     dimensionless       linear
               (OR supply both sps_col_Li and sps_col_Ln instead)

       Optional (skip derivation if provided):
           sps_col_Lmech_total, sps_col_Lmech_SN, sps_col_pdot_SN,
           sps_col_Mdot_SN, sps_col_v_SN, sps_col_Li, sps_col_Ln

       Recognized units: yr, Myr, s, erg/s, L_sun, 1/s, 1/Myr,
                         g*cm/s^2, Msun*pc/Myr^2, cm/s, pc/Myr,
                         g/s, Msun/Myr, dimensionless

       The SPS file's actual columns (read from its header row):
           <comma-separated column names from the file's header>
```

**Code-level checklist.**

- [ ] `.param` parser recognizes `sps_col_*` keys and consolidates them
      into a single `params['sps_column_map']` `DescribedItem` whose
      value is a dict `{canonical: ColumnSpec(file_column, units,
      log)}`. The container is `exclude_from_snapshot=True`.
- [ ] Each individual `sps_col_<canonical>` key is *also* round-trip
      preserved via `flush()`, so the user's `.param` survives a
      load-flush-reload cycle.
- [ ] `read_param.py` validates the column map when `sps_path !=
      def_path`. Hard-errors with the template above if required
      canonicals are missing, or if `Li`/`Ln` are partially supplied,
      or if a declared `<units>` is not in the recognized set.
- [ ] User-defined `sps_path` files **must** have a header row. The
      loader reads it via `np.genfromtxt(..., names=True)`. If the file
      is detected as headerless (all-numeric first row), hard-error
      with: "user-defined sps_path files must include a header row;
      the columns are matched by name against your sps_col_*
      declarations."
- [ ] Legacy SB99 fallback (`sps_path == def_path`) does **not** require
      a header — the existing `np.loadtxt` path stays.
- [ ] Unit conversion factors live in a single table (logically one
      `dict[str, float]`) keyed by `(declared_unit, canonical_unit)`.
      No per-canonical hardcoding scattered through the loader body.
- [ ] Mass scaling is applied to the canonicals marked "yes" in the
      table above; never to `t`, `fi`, `v_SN`. Hardcoded list, not
      user-declared.
- [ ] `t=0` prepend (currently `read_SB99.py:262-275`) detects an
      existing t=0 row in the loaded data and skips to avoid
      double-application on generic CSVs.
- [ ] No change to FB_* scaling logic.

**Tests required to merge.**

1. **Legacy fallback unchanged.** With `sps_path = def_path` and no
   `sps_col_*` keys, the loader produces byte-identical arrays to PR-1.
2. **User-defined `sps_path` with complete mapping** mirroring SB99's
   column structure (a clone of the legacy SB99 file with a header row
   declaring `time, log_Qi, log_fi, log_Lbol, log_Lmech, log_pdot_W,
   log_Lmech_W` and `sps_col_*` declarations matching exactly) →
   byte-identical arrays to (1). Confirms the new path is loss-free
   when used to replicate the legacy file.
3. **No `sps_col_*` declarations.** User-defined `sps_path`, no
   `sps_col_*` keys at all → loader exits non-zero; stderr contains the
   full template *and* the file's actual header columns.
4. **Partial declarations.** User-defined `sps_path`, missing
   `sps_col_Qi` only → loader exits non-zero; stderr names `Qi` as the
   missing canonical (not "every canonical").
5. **Unknown unit.** `sps_col_Lbol  l_bol  furlongs_per_fortnight  log`
   → loader exits non-zero with the recognized-units list.
6. **Headerless user file.** User-defined `sps_path` pointing at a file
   with no header row → hard error pointing the user at adding one.
7. **Linear-units declaration.** A clone of the SB99 file with values
   pre-exponentiated, declared `log: linear` → arrays within 4 ULP of
   the log-space load (linear→log→linear is not bitwise reversible).
8. **Missing optional columns route through derivation.** Header file
   with the 7 required canonicals and no SN/Li/Ln → arrays match (1).
9. **`Li` + `Ln` both supplied** with values whose ratio differs from
   SB99's `fi` → file values used, `fi`-based derivation is bypassed.
   Closes hot-spot #5 in a verifiable way.
10. **`Li` supplied alone (without `Ln`)** → hard error: "supply both
    `Li` and `Ln`, or neither".
11. **Explicit `Lmech_SN` that doesn't match derivation.** Synthetic CSV
    with `Lmech_SN = 0.5 · (Lmech_total − Lmech_W)` → loader uses the
    file value; resulting `Mdot_SN` and `velocity_SN` differ from the
    derivation path in the expected direction. Sanity test that the
    override path is wired through, not silently overwritten.
12. **`t=0` prepend idempotent** on a CSV that already has a t=0 row.
13. **Mass scaling correctness.** Same file loaded with `mCluster =
    1e6` and `mCluster = 2e6` (with `sps_refmass = 1e6` both times) →
    mass-scaled arrays differ by exactly 2×; non-mass-scaled arrays
    (`t`, `fi`, `v_SN`) are identical.
14. **E2E equivalence** — full trinity run with `sps_path = def_path`
    on the three anchor configs. Snapshot trees match PR-1 goldens at
    `rtol=1e-12`.
15. **E2E with user-defined `sps_path`.** Copy the SB99 file with a
    header row, add a complete `sps_col_*` block to one anchor `.param`,
    run trinity. Snapshot tree matches PR-1's golden for that anchor at
    `rtol=1e-12`. (This is the "did the new path break anything" test.)

### PR-3 — Rename `SB99f` → `sps_f`, `SB99_data` → `sps_data`

**Scope.** Mechanical rename. Every consumer touched. Optional module rename
`read_SB99.py` → `read_sps.py`. Aliased back-compat in `read_param.py` so
external code reading `params['SB99f']` still works for one release.

**Files touched.**

- `src/_input/read_param.py` — rename runtime containers; keep `SB99f`/
  `SB99_data` as alias entries pointing at the same `DescribedItem`. The
  alias is feasible because `DescribedDict.__setitem__` (`dictionary.py:205`)
  just stores the object reference — two keys can point at the same
  `DescribedItem` instance.
- `src/sb99/read_SB99.py` → rename symbols (and optionally file). See open
  question §14 for module-vs-symbol scope.
- `src/sb99/update_feedback.py` — `SB99Feedback` → `SPSFeedback` (currently
  at line 21), `get_currentSB99feedback` → `get_current_sps_feedback`
  (currently at line 98), all `SB99f` reads → `sps_f`.
- All phase files in §5.3–§5.8 — update imports.
- `src/main.py:142-152` — rename container references.

**Code-level checklist.**

- [ ] Single rename PR, no logic changes. `git diff --stat` should be heavy
      on `phase*` files but contain no algorithmic changes.
- [ ] Back-compat alias in `read_param.py`: `params['SB99f'] =
      params['sps_f']` (same underlying object) so external user scripts
      continue to work.
- [ ] `phase0_init/get_InitPhaseParam.py:88, 111-112` updated.
- [ ] `bubble_luminosity_modified.py:33` import updated (still dead; PR-4
      removes).

**Tests required to merge.**

1. Full equivalence battery from PR-2, rerun unchanged.
2. New test: `params['SB99f'] is params['sps_f']` (alias works).
3. New test: importing `get_currentSB99feedback` from `update_feedback`
   raises a clear error pointing at the new name (assuming the old symbol
   is removed; if a transitional alias is kept, this test inverts).

### PR-4 — Cleanup (legacy stays)

**Scope.** Cosmetic cleanup only. Does **not** remove any user-facing
surface. The legacy SB99 calling method (`SB99_mass / SB99_rotation /
SB99_BHCUT`) remains a permanent supported fallback per §9 — PR-4 does
not touch it. This is the "last resort" guarantee.

**Files touched.**

- `src/bubble_structure/bubble_luminosity_modified.py` — delete the
  dead `get_currentSB99feedback` import at line 33 (verified
  zero-call-sites, see §5.8).
- `src/sb99/update_feedback.py` — optionally remove the transitional
  `get_currentSB99feedback` alias kept by PR-3, *only if* you're
  confident no out-of-tree consumer relies on the old name. Otherwise
  leave it in place permanently.
- `src/_input/read_param.py` — optionally remove the `params['SB99f']`
  back-compat alias from PR-3 on the same condition. The
  `params['sps_column_map']` plumbing and the legacy `get_filename()`
  helper stay.

**Files explicitly NOT touched.**

- `src/_input/default.param` — `SB99_mass`, `SB99_rotation`, `SB99_BHCUT`
  remain declared with their original defaults.
- `src/_input/read_param.py` — the legacy fallback resolution
  (`sps_path == def_path` → `get_filename(params)`) remains.
- The startup `logger.info` notification stays.

**Tests required to merge.**

1. Legacy configs that set only `SB99_mass / SB99_rotation / SB99_BHCUT
   / ZCloud / path_sps` (and no `sps_path`) still produce snapshot
   trees byte-equivalent (within `rtol=1e-12`) to the PR-3 golden.
   This is the "last resort" smoke test.
2. Configs that set `sps_path` + `sps_col_*` continue to work; full
   E2E equivalence against PR-3 golden.
3. `bubble_luminosity_modified.py` imports list contains no
   `get_currentSB99feedback` reference.
4. (If the PR-3 transitional aliases are removed) old import paths
   raise `ImportError` with a clear message pointing at the new names.

### Out of scope: cooling-table coupling

`src/cooling/non_CIE/read_cloudy.py:47, 263-335` constructs filenames from
`SB99_rotation` and `ZCloud`. Even after the feedback CSV is decoupled, the
cooling cubes themselves were generated from SB99-keyed SEDs, so swapping in
a different SPS does not magically generalize cooling.

**Mitigation in this refactor.** When `sps_path` is set to a non-default
value, emit a `UserWarning`: "Cooling tables are still keyed by SB99
rotation+Z; results valid only if the SPS source is SB99-compatible at the
declared rotation/Z." Tracked as follow-up: add `path_cooling_nonCIE` knob
analogous to `sps_path`.

## 11. Test strategy (the substance)

This section is what makes the refactor safe. Everything else is plumbing.

### 11.1 Golden capture protocol (DO THIS FIRST)

**One-time setup on `main`, before any refactor branch is cut.**

Anchor configs (single-runs, deterministic, fast — avoid the `_sweep.param`
files because sweeps fan out via `ProcessPoolExecutor` and aren't golden-able
cleanly):

- `param/cloud_example_PL.param` (power-law density profile)
- `param/cloud_example_BE.param` (Bonnor-Ebert profile)
- `param/cloud_example_homogeneous.param` (homogeneous profile)

Trinity's actual CLI is `python run.py <param_file>` (entry at
`/home/user/trinity/run.py`). There is no `--output` flag — the output
directory is set inside the `.param` file via `path2output`. So the harness
must copy each anchor `.param`, rewrite `path2output` to a golden-capture
location, then invoke `run.py` on the copy.

```bash
git checkout main
mkdir -p analysis/sb99-refactor-golden
python analysis/sb99_refactor_equivalence.py --capture-golden \
    --configs param/cloud_example_PL.param \
              param/cloud_example_BE.param \
              param/cloud_example_homogeneous.param \
    --golden-dir analysis/sb99-refactor-golden
```

The harness does two independent things per anchor:

1. **E2E capture (subprocess).** Copy the `.param` to a temp location,
   rewrite `path2output` to point under
   `analysis/sb99-refactor-golden/<stem>/`, then invoke
   `python run.py <tmp_param>` via `subprocess.run`. Trinity writes the
   JSONL snapshot tree directly into the golden directory; nothing inside
   trinity needs to change.
2. **Unit-layer capture (in-process).** Separately, in the harness process,
   `import src.sb99.read_SB99` and call `read_SB99.read_SB99(f_mass=1.0,
   params=mock_params)` to capture the 11-array tuple. Then call
   `get_interpolation(...)` and evaluate each of the 10 interpolators at a
   dense time grid. Pickle both. This lets PR-N's unit tests run without
   re-invoking trinity end-to-end.

The golden snapshot tree must be **frozen for the entire refactor**. Do not
regenerate it on later commits — that would silently mask drift. Record the
commit SHA of `main` at capture time in
`analysis/sb99-refactor-golden/MANIFEST.json`.

If `main` changes between PRs (e.g. an unrelated merge), the golden does not
need to be regenerated for *this* refactor — what matters is that each
refactor PR matches the golden that existed at the moment the refactor
branched from main.

### 11.2 Per-PR test battery

A single Python script `sb99_refactor_equivalence.py` runs all tests; each
PR's gate is "run the script with `--pr N`, must exit 0". Placement of the
script (`test/` vs `analysis/`) is open question §14, but the codebase
already has a tracked `test/` directory (cloudy + simplify + metadata
pytest files), so the harness can plausibly live there.

#### 11.2.1 PR-1 tests

```python
def test_path_resolution_matrix():
    """Every legacy parameter combination resolves to the same on-disk
    path under the new fallback as under the old hardcoded grammar."""
    combos = [
        (rot, Z, BH, mass)
        for rot in (0, 1)
        for Z in (1.0, 0.15)
        for BH in (120, 40)
        for mass in (1e3, 1e4, 1e5, 1e6)
    ]
    for rot, Z, BH, mass in combos:
        legacy = legacy_grammar_path(rot, Z, BH, mass)         # captured from main
        resolved = resolve_sps_path_via_fallback(rot, Z, BH, mass)  # new code
        assert legacy == resolved, f"path drift at {rot, Z, BH, mass}"

def test_loader_byte_equivalence():
    """The 11-array loader output is byte-identical to the pickled golden."""
    golden = pickle.load(open('analysis/sb99-refactor-golden/loader_arrays.pkl', 'rb'))
    arrays = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    for name, gold, cur in zip(ARRAY_NAMES, golden, arrays):
        assert np.array_equal(gold, cur), f"{name} drifted"

def test_interpolator_byte_equivalence(n_samples=1000):
    """All 10 interpolators agree at densely sampled times."""
    golden_interp = pickle.load(open('.../interp_samples.pkl', 'rb'))
    arrays = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    SB99f = read_SB99.get_interpolation(arrays)
    for key, (ts, ys_gold) in golden_interp.items():
        ys_cur = SB99f[key](ts)
        assert np.array_equal(ys_gold, ys_cur), f"interpolator {key} drift"

def test_dataclass_byte_equivalence():
    """get_currentSB99feedback agrees at sampled times."""
    golden = pickle.load(open('.../feedback_samples.pkl', 'rb'))
    params = mock_params_legacy_loaded()
    for t, fb_gold in golden:
        fb_cur = get_currentSB99feedback(t, params)
        for field in fields(fb_gold):
            g, c = getattr(fb_gold, field.name), getattr(fb_cur, field.name)
            assert np.array_equal(g, c), f"feedback.{field.name} drift at t={t}"

def test_e2e_snapshot_equivalence():
    """Full trinity run; snapshot JSONL trees match within rtol=1e-12.

    Invokes the actual CLI: copy anchor param, rewrite path2output, run.
    """
    for cfg in CANONICAL_CONFIGS:
        tmp_cfg = rewrite_path2output(cfg, tmpdir / cfg.stem)
        subprocess.run(['python', 'run.py', str(tmp_cfg)], check=True, cwd=TRINITY_ROOT)
        gold_out = f"analysis/sb99-refactor-golden/{cfg.stem}/"
        diff_snapshot_trees(gold_out, tmpdir / cfg.stem, rtol=1e-12, atol=0)

def test_e2e_with_explicit_sps_path():
    """Setting sps_path explicitly to the legacy file yields the same run."""
    for cfg in CANONICAL_CONFIGS:
        explicit_cfg = inject_sps_path(cfg, legacy_grammar_path_for(cfg))
        tmp_cfg = rewrite_path2output(explicit_cfg, tmpdir / cfg.stem)
        subprocess.run(['python', 'run.py', str(tmp_cfg)], check=True, cwd=TRINITY_ROOT)
        gold_out = f"analysis/sb99-refactor-golden/{cfg.stem}/"
        diff_snapshot_trees(gold_out, tmpdir / cfg.stem, rtol=1e-12, atol=0)

def test_legacy_path_emits_one_info_log_per_run(caplog):
    """Legacy fallback emits exactly one INFO log naming the SB99 grammar
    — informational, not a warning, not an error. The legacy path is a
    permanent supported fallback (§9), not deprecated."""
    with caplog.at_level(logging.INFO):
        load_with_legacy_params()
        load_with_legacy_params()  # second call same process
    matches = [r for r in caplog.records
               if r.levelno == logging.INFO
               and 'legacy SB99' in r.message]
    assert len(matches) == 1
    # And there must be NO warnings of any category about legacy use.
    assert not any('SB99' in r.message and r.levelno >= logging.WARNING
                   for r in caplog.records)
```

#### 11.2.2 PR-2 tests

```python
def test_legacy_fallback_byte_identical_to_pr1():
    """sps_path = def_path, no sps_col_* keys → arrays byte-identical to PR-1."""
    params = mock_params_legacy()
    arrays = read_SB99.read_SB99(f_mass=1.0, params=params)
    golden = pickle.load(open('analysis/sb99-refactor-golden/loader_arrays.pkl', 'rb'))
    for cur, gold in zip(arrays, golden):
        assert np.array_equal(cur, gold)

def test_user_sps_path_mirroring_sb99_byte_identical():
    """Header-equipped clone of SB99 file + complete sps_col_* block →
    byte-identical to legacy load."""
    write_sb99_with_canonical_header(src=LEGACY_FILE, dst=tmpfile)
    params = mock_params_with_sps_path(tmpfile, column_map=SB99_LIKE_MAP)
    arrays_user = read_SB99.read_SB99(f_mass=1.0, params=params)
    arrays_legacy = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    for u, l in zip(arrays_user, arrays_legacy):
        assert np.array_equal(u, l)

def test_no_column_map_hard_errors_with_template():
    """User-defined sps_path with no sps_col_* keys → exit non-zero,
    stderr contains the fillable template AND the file's actual columns."""
    params = mock_params_with_sps_path(USER_FILE, column_map={})
    with pytest.raises(SystemExit) as excinfo:
        read_param.validate_column_map(params)  # called during read_param
    assert excinfo.value.code != 0
    assert 'sps_col_t' in capfd.readouterr().err
    assert 'time, l_bol, l_wind' in capfd.readouterr().err  # file's actual cols

def test_partial_column_map_names_missing_canonicals():
    """User-defined sps_path missing only sps_col_Qi → error names Qi specifically."""

def test_unknown_units_hard_errors():
    """sps_col_Lbol l_bol furlongs_per_fortnight log → error lists recognized units."""

def test_headerless_user_file_hard_errors():
    """User-defined sps_path pointing at a headerless file → error pointing
    at adding a header row."""

def test_linear_units_within_4_ulp_of_log():
    """Pre-exponentiated columns declared 'linear' → arrays within 4 ULP
    of the log-space load. (10**log10(x) is not exactly x.)"""

def test_missing_optional_cols_fall_back():
    """Header file with the 7 required canonicals and no SN/Li/Ln →
    arrays match the legacy load."""

def test_Li_Ln_both_present_bypass_fi_derivation():
    """File supplies Li and Ln with a ratio that differs from SB99's fi.
    Loader uses the file values directly, NOT Lbol·fi."""

def test_Li_alone_hard_errors():
    """sps_col_Li present without sps_col_Ln → error: supply both or neither."""

def test_explicit_Lmech_SN_overrides_derivation():
    """Synthetic CSV with Lmech_SN = 0.5 * (Lmech_total - Lmech_W) → file
    value used. Mdot_SN and velocity_SN differ from the derivation path
    by exactly the expected factor."""

def test_t0_prepend_idempotent():
    """CSV that already has a t=0 row doesn't get a duplicate after prepend."""
    write_csv_with_explicit_t0_row(dst=tmpfile)
    arrays = read_SB99.read_SB99(f_mass=1.0,
                                 params=mock_params_with_sps_path(tmpfile))
    assert arrays[0][0] == 0.0 and arrays[0][1] > 0.0  # no double t=0

def test_mass_scaling_correctness():
    """Same file loaded with mCluster ∈ {1e6, 2e6}, sps_refmass = 1e6:
    mass-scaled arrays differ by exactly 2×; t, fi, v_SN identical."""

def test_e2e_legacy_path_unchanged():
    """Full trinity run with sps_path = def_path matches PR-1 golden at rtol=1e-12."""

def test_e2e_user_sps_path_matches_legacy():
    """Full trinity run with sps_path pointing at header-equipped SB99 clone
    + complete sps_col_* block matches PR-1 golden at rtol=1e-12."""
```

#### 11.2.3 PR-3 tests

```python
def test_all_phases_run_after_rename():
    """E2E equivalence battery."""

def test_back_compat_alias_works():
    """params['SB99f'] is params['sps_f']."""

def test_renamed_imports_resolve():
    """from src.sb99.update_feedback import get_current_sps_feedback works."""
```

#### 11.2.4 PR-4 tests

```python
def test_legacy_config_still_works_after_pr4():
    """The 'last resort' guarantee. A config that sets only SB99_mass /
    SB99_rotation / SB99_BHCUT / ZCloud / path_sps (no sps_path, no
    sps_col_*) produces a snapshot tree within rtol=1e-12 of the PR-3
    golden. Legacy is permanent, not a deprecation timer."""

def test_sps_path_config_still_works_after_pr4():
    """Full E2E equivalence against PR-3 golden for an sps_path-based config."""

def test_bubble_luminosity_imports_clean():
    """No reference to get_currentSB99feedback in
    bubble_luminosity_modified.py after the dead-import drop."""
```

### 11.3 Equivalence tolerance policy

Different stages tolerate different drift. Be explicit:

| Layer | Comparator | Rationale |
|-------|------------|-----------|
| Resolved path string | `==` | strings, no ambiguity |
| Raw loader arrays | `np.array_equal` | same numpy ops, must be deterministic |
| Interpolators at sample times | `np.array_equal` | scipy `interp1d` is deterministic on identical inputs |
| `SB99Feedback` fields | `np.array_equal` | downstream of byte-identical interpolators |
| 12 scalar params | `np.array_equal` | downstream of dataclass |
| JSONL snapshot fields | `np.allclose(rtol=1e-12, atol=0)` | I/O roundtrip introduces ULP-level noise |
| Linear-vs-log unit roundtrip (PR-2 only) | `np.allclose(rtol=4e-15, atol=0)` | `10**log10(x)` is not exactly `x` for arbitrary `x` |

If any test fails with drift below these tolerances, **escalate before
loosening the tolerance**. Tolerance loosening to hide drift is the failure
mode this whole plan exists to prevent.

### 11.4 Harness layout

```
test/                                            # (existing pytest dir)
└── test_sb99_refactor_equivalence.py            # OR analysis/sb99_refactor_equivalence.py
analysis/
├── sb99-refactor-audit.md                       # this file (single source of truth)
└── sb99-refactor-golden/                        # gitignored; pickled goldens + JSONL trees
    ├── MANIFEST.json                            # commit SHA of main at capture time
    ├── loader_arrays.pkl
    ├── interp_samples.pkl
    ├── feedback_samples.pkl
    ├── cloud_example_PL/
    │   ├── 1_begin.jsonl
    │   ├── 2_energy.jsonl
    │   ├── 3_implicit.jsonl
    │   ├── 4_transition.jsonl
    │   ├── 5_momentum.jsonl
    │   ├── 6_final.jsonl
    │   ├── dictionary.jsonl
    │   ├── metadata.json
    │   └── *.param, *.txt, etc.
    ├── cloud_example_BE/
    └── cloud_example_homogeneous/
```

Add `analysis/sb99-refactor-golden/` to `.gitignore`. The pickles and JSONL
trees are too big and ephemeral to commit.

## 12. Risk register

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| Float ULP drift introduced by reordering ops in loader | Medium | High | PR-1 explicitly does NOT touch unit-conversion math. Byte-equivalence test catches it. |
| Path-resolution drift due to subtle formatting in `get_filename` (mantissa formatter `format_e` at `read_SB99.py:328-333`) | Medium | High | Path-resolution matrix test covers all legal combos. |
| User mis-declares `log` vs `linear` or wrong units in `sps_col_*` (silent physics-altering bug) | Medium | High | PR-2 hard-errors on unrecognized unit strings. For declared-but-wrong combinations (e.g. `erg/s log` when the file is actually linear erg/s), no automatic detection — `.param` review by the user is the line of defense. Open question §14 #7 below: add a per-canonical "expected order-of-magnitude" sanity check that warns on grossly out-of-range loaded values? |
| User points `sps_path` at a headerless file expecting it to "just work" | Medium | Low | Hard error directs them to add a header row. Documented in PR-2 error template. |
| `t=0` prepend hack (loader 262-275) double-applied if generic CSV already has t=0 | Medium | Medium | PR-2 loader detects `t[0] == 0` and skips prepend; explicit `test_t0_prepend_idempotent`. |
| Constant-column sniff-test false-fires on legitimate SB99 artifacts (e.g. `l_sn` is constant during the pre-SN regime by design) | Medium | Low | No sniff-test in PR-2 — left for later. If added, it would need an allowlist for known-good constant patterns. |
| Cooling tables silently mismatch generic SPS | High | Medium | Out-of-scope `UserWarning` emitted; tracked separately. |
| `scipy.interp1d` deprecation in future scipy | Low | Low | Pin scipy in repo deps. Out of scope for this refactor. |
| Users have configs that set `SB99_mass` to something other than the canonical 1e6 | Medium | Low | `sps_refmass` defaults to `SB99_mass`, back-compat preserved. Test with mass=2.5e6. |
| PR-3 rename misses a consumer (silently leaves stale `SB99f` reference) | Low | High | `grep -r 'SB99' src/` after PR-3 must show only the back-compat alias declarations and docstrings/comments. |
| Anchor configs (`cloud_example_*`) don't exercise some physics regime that breaks the refactor | Low | Medium | Three profiles (PL, BE, homogeneous) span the density-profile space. If a regression slips through, add a fourth anchor and re-golden. |

## 13. Rollout sequence

1. **Capture golden** on `main` (§11.1). Verify the harness can compare a
   freshly captured tree to itself with zero drift (smoke test).
2. **Branch `feature/sps-path-fallback`** for PR-1. Run battery; merge when
   green.
3. **Branch `feature/sps-column-mapping`** for PR-2. Re-run full battery
   (includes the explicit-SN-column and Li/Ln-override paths).
4. **Branch `feature/sps-rename`** for PR-3. Re-run full battery + new
   alias tests.
5. **Soak.** Let PR-3 run in production for at least one release cycle
   before queueing PR-4. PR-4 is purely cosmetic; nothing user-facing
   breaks, but soaking catches any unexpected coupling to the dead
   import or transitional aliases.
6. **Branch `fix/drop-sb99-dead-import`** for PR-4. Re-run battery
   against the post-PR-3 golden. **Legacy SB99 params remain
   permanent** — see §9.

All branch names use the repo's `feature/` or `fix/` prefix per CLAUDE.md.
The current audit branch (`claude/sb99-default-parameter-ttQIN`) violates
that rule and exists only because the original session was provisioned with
the wrong prefix — confirm with user before pushing PR-1 from a
properly-named branch.

## 14. Open questions for the user

Before starting PR-1, please confirm:

1. **Module rename scope.** In PR-3, keep loader at `src/sb99/read_SB99.py`
   (rename symbols only), or move to `src/sps/read_sps.py`? Affects PR-3
   churn substantially. Recommendation: symbols only, keep module path; SB99
   is the canonical SPS for this codebase.
2. **Legacy persistence (confirmation).** §9 now treats `SB99_mass /
   SB99_rotation / SB99_BHCUT` as a permanent fallback path — none of the
   four PRs removes them. The startup notification is a one-time
   `logger.info`, not a `DeprecationWarning`. Confirm this is the
   intended policy (vs. eventually scheduling removal in a future
   release).
3. **Anchor config selection.** Are `cloud_example_PL.param`,
   `cloud_example_BE.param`, `cloud_example_homogeneous.param` the right
   three for the E2E battery? They're the three tracked single-run example
   configs in `param/`. The `_sweep.param` files would fan out via the
   sweep executor and are awkward to golden cleanly.
4. **Harness placement.** Put `test_sb99_refactor_equivalence.py` under
   the existing tracked `test/` dir (alongside `test_cloudy_*` etc.), or
   keep it as a working artifact under `analysis/` and delete it once the
   refactor lands? CLAUDE.md says "don't reintroduce the deleted /test/
   folder" but a `test/` dir currently exists and is tracked, so the
   guidance is stale.
5. **Cooling coupling timing.** Out of scope here, but: do you want a
   parallel issue/PR opened to address `read_cloudy.py`'s SB99 keying, or
   leave that until the SPS refactor lands?
6. **`sps_format` shortcut?** Should PR-2 accept a single
   `sps_format    sb99_positional` key that lets users opt into the
   7-column legacy preset for a non-default `sps_path` without
   re-declaring every `sps_col_*` line? Convenience for "I'm just
   pointing at a different SB99 file." Recommendation: leave out for
   now — it's a foot-gun if applied to a non-SB99 file, and the explicit
   `sps_col_*` block is short. Add later only if real friction shows up.
7. **Order-of-magnitude sniff-test?** Should PR-2 (or a follow-up) add a
   per-canonical "expected value range" check that warns when a loaded
   column is grossly out-of-range (e.g. `Qi` linear value < 1e30 or
   > 1e60)? Catches `log`-vs-`linear` mis-declarations but adds a
   maintenance burden (the ranges have to be kept honest). Recommendation:
   not in PR-2; revisit after first real user.

---

## Appendix — Working notes

- **Branch.** `claude/sb99-default-parameter-ttQIN` (violates CLAUDE.md
  `feature/bugfix/hotfix/fix` rule — confirm with user before pushing each
  PR from a properly-named branch).
- **Snapshot exclusion.** `SB99f` and `SB99_data` are flagged
  `exclude_from_snapshot=True` (`read_param.py:472-473`).
- **Footprint.** Audit covers 174 SB99-keyword hits across 19 files
  (2026-05-12). Only 6 are real runtime consumers; rest are comments,
  docstrings, demo code (`dictionary.py:1203-1234`), or the separate non-CIE
  cooling coupling.
- **`path_sps` default.** `lib/sps/starburst99/` is the on-disk default;
  `path_sps` is the indirection.
- **Dead import.** `bubble_luminosity_modified.py:33` imports
  `get_currentSB99feedback` and never calls it. PR-4 removes.
- **Boundary edge case.** The numerical derivative step `dt = 1e-9 Myr`
  (`update_feedback.py:187`) is unconditionally small. If a user-supplied
  CSV has `t_min > 0` and queries land near `t_min`, `fpdot_total(t - 1e-9)`
  could go out of bounds. Existing SB99 grids are coarse enough that this
  hasn't surfaced.
- **`test/` directory status.** Tracked, contains 7 pytest files (cloudy,
  simplify, metadata). CLAUDE.md's "don't reintroduce the deleted /test/
  folder" guidance appears to be stale relative to current repo state.
- **`params['path_sps']` accessed without `.value`.** `read_SB99.py:126`
  reads `path2sps = params['path_sps']` (no `.value`). Either
  `DescribedItem.__add__` handles concat, or the storage was upgraded
  in-place earlier. Not a problem for this refactor, but worth a one-line
  check during PR-1.
