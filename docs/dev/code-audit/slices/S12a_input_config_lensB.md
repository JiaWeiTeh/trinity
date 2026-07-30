# S12a input config — Lens B (what the code claims)

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

**This is a prose-only transcription.** Every statement below is a *claim made by comments and
docstrings*, not an observation of behaviour. I have seen **no code** — no function bodies, no
signatures, no literals outside of what the prose itself spells out. My entire input was
`scratchpad/lens/S12a_input_config/prose.md`, which carries the comments and docstrings of seven
files, each tagged with source file and line range:

- `trinity/_input/dictionary.py`
- `trinity/_input/registry.py`
- `trinity/_input/read_param.py`
- `trinity/_input/param_spec.py`
- `trinity/_input/errors.py`
- `trinity/_input/fkappa_auto.py`
- `trinity/_input/__init__.py` (no prose extracted — the slice header lists it, but the prose file
  contains no section for it; that absence is itself recorded below)

Where I write "the prose claims X", X may or may not be what the code does. Line citations point at
the first line of the comment/docstring block carrying the claim.

---

## 1. The declared pipeline: `read_param` steps and their contracts

The prose describes a fixed 10-step load pipeline, with ordering constraints asserted at several
points. Consolidated from `read_param.py` and `registry.py`:

| Step | Claimed job | Claimed driver | Ordering claim |
|---|---|---|---|
| 1 | Read `default.param` with `# INFO:` / `# UNIT:` metadata | inline (`read_param.py:106`) | — |
| 2 | Read user `.param` | inline (`read_param.py:176`) | — |
| 3 | Validate user keys exist in default; enforce companion bundles | `registry.validate_companions` (`registry.py:686`) | must run on the **raw user dict, pre-merge** (`read_param.py:227`) |
| 4 | Build `DescribedDict`, convert units to `[Msun, pc, Myr]` | inline (`read_param.py:250`) | — |
| 5 | Validators | `registry.validate_all` (`registry.py:547`) | runs **before** resolvers (`registry.py:91`) |
| 6 | Derived parameters (composition, masses, model_name) | inline (`read_param.py:299`) | — |
| 7 | Resolve `def_*` sentinels | `registry.resolve_all` (`registry.py:565`) | "Must run after Step 6 (model_name resolved; path2output depends on it)" (`read_param.py:409`) |
| 8 | `active_when` conditional schema | `registry.apply_active_when` (`registry.py:589`) | "Must run after Step 5 … and before Step 9" (`registry.py:589`) |
| 9 | Snapshot-exclusion sweep for constants | inline (`read_param.py:444`) | expects "the final key set" (`registry.py:589`) |
| 10 | Materialize runtime/derived-init params | `registry.materialize_runtime` (`registry.py:625`) | "Must run AFTER Step 9" (`registry.py:625`, echoed `read_param.py:469`) |
| — | Guard: runtime init must not replace `default.param` `DescribedItem`s | inline (`read_param.py:475`) | compares **object identity**, post-Step-10 (`read_param.py:279`) |

`registry.py:1` maps these to "Phase" numbers: Phase 5 = run-const derivation, Phase 6 = validation,
Phase 7 = sentinel resolution, Phase 8 = conditional schema, Phase 9 = runtime materialization,
"Phase 10 will wire Step 6's derived-init resolvers" (stated as still pending, also at
`param_spec.py:1`).

## 2. `.param` grammar — as claimed

**Both files** (`read_param.py:130`, `:146`, `:185`):
- Inline comments are stripped (module docstring example: `"mCloud 1e6 # cloud mass"`,
  `read_param.py:3`).
- Empty lines skipped.
- A parameter line is `key value`, "Split on first whitespace only" (`read_param.py:155`).

**`default.param` only** (`read_param.py:136`, `:142`, `:159`, `:164`, `:169`):
- Lines may be `INFO` or `UNIT` metadata lines; "Remove surrounding brackets if present".
- Storage is `key -> (info, unit, default_value)` (`read_param.py:120`).
- "Skip malformed lines in `default.param`" — silently.
- Metadata is reset after each parameter.
- `default.param` "lives next to this script in `trinity/_input/`, not in the user-facing `param/`
  directory" (`read_param.py:109`).

**Value parsing** (`parse_value`, `read_param.py:73`): stated precedence
`None → boolean → number → fraction → string`, with string as fallback. The inline comment sequence
(`:80` None, `:84` Boolean, `:90` Number float-or-int, `:96` Fraction "e.g., 5/3", `:102` String
fallback) matches the docstring. Fractions like `5/3` are therefore accepted `.param` syntax.

**Merge/precedence** (`read_param.py:214`, `:233`, `:237`, `:241`, `:244`): user keys must exist in
`default.param` or it is an error; user values override defaults; unset keys take the default;
overridden parameters are reported.

**Units** (`read_param.py:256`): conversion to `[Msun, pc, Myr]` via `unit_conversions.py`; "Only
convert numeric values; strings, booleans, and None remain unchanged"; "None values (e.g., for
disabled termination conditions) pass through" (`:259`).

**Errors** (`read_param.py:44`): `ParameterFileError` "If parameter file has formatting errors or
invalid parameters"; `FileNotFoundError` "If default.param cannot be found". `errors.py:1` claims
the exception lives in `errors.py` rather than `read_param` specifically to avoid a
`registry → read_param → registry` import cycle, and that `read_param` re-exports it for back-compat
(also `read_param.py:502`).

## 3. Sentinels, resolvers, `consumed_by`

`param_spec.py:64` enumerates the sentinel vocabulary: `def_dir`, `def_path`, `def_value`,
`def_unset`, all sharing a `SENTINEL_PREFIX`. A sentinel is resolved by **either** its own
`resolver` **or** another spec's, declared via `consumed_by` — "Mutually exclusive … never both"
(`param_spec.py:116`). A test guard is named: `test_every_sentinel_default_has_resolver_or_pointer`
(`param_spec.py:78`), plus `test_consumed_by_targets_exist` (`:128`).

Claimed resolver inventory — **three** specs: `path2output`, `path_cooling_nonCIE`, `sps_path`
(`param_spec.py:120`, `registry.py:565`, `read_param.py:405`).

- `_resolve_path2output` (`registry.py:230`): `'def_dir'` → `<cwd>/outputs/<model_name>`; a user path
  taken as-is; "Either way the directory is created."
- `_resolve_path_cooling_nonCIE` (`registry.py:242`): `'def_dir'` → "the shipped OPIATE cube folder
  under `lib/default/opiate/`"; user path as-is and created.
- `_resolve_sps_bundle` (`registry.py:253`): owns `sps_path` + `sps_refmass` + `sps_column_map`;
  `'def_path'` → `lib/default/sps/starburst99/1e6cluster_default.csv`, described as "an SB99 grid at
  rotation=1, ZCloud=1, mass=1e6 Msun in CSV form with the canonical 7-column SB99 layout
  (DEFAULT_SPS_COLUMN_MAP)". `sps_refmass` `'def_value'` → `1e6` **only** for the bundled file; a
  user `sps_path` "requires an explicit value (silent 1e6 would mis-scale
  `f_mass = mCluster / sps_refmass`)" — repeated at `registry.py:305` and `param_spec.py:73`.
- Resolvers "run unconditionally: each handles BOTH the sentinel (`def_*`) branch and the
  user-supplied branch" (`registry.py:224`), and their "Logic and error messages are lifted verbatim
  from the pre-Phase-7 inline Step-7 block so behavior is byte-identical" (`registry.py:227`).

**Not a sentinel:** `path_cooling_CIE` (`read_param.py:412`) — "an integer-index preset keyed on
ZCloud, so it stays inline". Claimed grammar: "Integer-index preset {1, 2, 3} (under ZCloud == 1)
selects between the bundled CIE tables; ZCloud == 0.15 auto-pins to the Sutherland-Dopita file. All
resolved paths live under `lib/default/CIE/`."

## 4. Validators — declared ranges and coercions

| Parameter | Claimed contract | Cite |
|---|---|---|
| `cooling_boost_fA` (f_A) | "f_A > 0 required"; warn on cross-knob combination with `cooling_boost_mode != none` or an active `cooling_boost_kappa` (double-counts interface cooling); "intended as a SINGLE knob" | `registry.py:118` |
| `cooling_boost_kappa` | at validation time "still its raw value here -- a number or the string 'auto'; both count as 'kappa active'" | `registry.py:118` |
| `cooling_boost_mode` | "'none' default parses to Python None" | `registry.py:118` |
| betadelta solver | `'hybr'` (**default**) = unbounded scipy root-finder with a "physical dMdt>0 acceptance gate"; `'legacy'` = bounded grid + L-BFGS-B | `registry.py:152` |
| `stop_at_rCloud_nSnap` | "Validate AND coerce: whole-number floats (e.g. 5.0 from '5') become ints; fractional floats / negatives / non-numerics raise" | `registry.py:165` |
| `coverFraction` (Cf) | "must be a number in (0, 1]"; Cf=1 = sealed bubble, Cf=0 "unphysical here (would vent the whole wall)" | `registry.py:190` |
| `rCloud_max` | "must be a positive number [pc]"; caps the pre-run GMC check in `trinity.cloud_properties.validate_gmc` | `registry.py:205` |

Contract for validators generally (`registry.py:91`, `:547`): receive `(value, params)`; may raise
`ParameterFileError` **or** normalize in place; "Order follows `SPECS`"; "specs missing from `params`
are skipped so densBE-only / densPL-only keys don't trigger on the other path"; "Error messages are
verbatim from the pre-Phase-6 Step-5 block so existing user diagnostics are preserved."

## 5. Conditional schema and companion rules

`apply_active_when` (`registry.py:589`) asserts an invariant verbatim: *"the spec is in `params` iff
`active_when(params)` returns True"* — active+absent → add a fresh `DescribedItem` with a
**deep-copied** default "so mutable defaults like `[]` aren't shared across runs"; present+inactive
→ `pop`; matching → no-op. The gating value today is "`dens_profile` ∈ {`densBE`, `densPL`}", and
"densBE/densPL profiles are mutually exclusive, so each predicate matches exactly one of the two
profile families" (`registry.py:75`). `read_param.py:434` gives concrete examples: `densPL_alpha`
popped on a densBE run, `densBE_Omega` popped on a densPL run, "the 9 densBE_* runtime params" added
on a densBE run. `param_spec.py:130` claims `active_when` is "Today carried only by the densBE /
densPL profile-conditional family", pinned by `test_active_when_only_on_conditional_specs`.

`CompanionRule` (`registry.py:697`): "If the user .param sets `trigger` to a value present as a key
in `requires`, every name in `requires[value]` must also appear in the same .param file."
Motivation (`registry.py:686`): "setting `dens_profile densPL` in a .param without `densPL_alpha`
silently yields the **default alpha=0 (homogeneous)** -- almost never what a user … actually wanted."
`validate_companions` (`registry.py:716`) "Raises `ParameterFileError` on the first violation,
listing the missing companion keys", and runs pre-merge so it "fires only when the user explicitly
set the trigger, not when the trigger came from default.param."

## 6. Documented defaults (every numeric/string default the prose states)

| Key / quantity | Stated default | Cite | Restated elsewhere? |
|---|---|---|---|
| `sps_refmass` | `1e6` (Msun), bundled file only | `registry.py:253` | yes — `registry.py:305` ("1e6 Msun"), `param_spec.py:73` ("falls back to 1e6") |
| `sps_path` | bundled `lib/default/sps/starburst99/1e6cluster_default.csv` | `registry.py:253` | — |
| `densPL_alpha` | `0` ("homogeneous") | `registry.py:686` | — |
| betadelta solver | `'hybr'` | `registry.py:152` | — |
| `cooling_boost_mode` | `'none'` → Python `None` | `registry.py:118` | — |
| `cooling_boost_kappa` | `1.0` ("the default 1.0 path stays byte-identical") | `fkappa_auto.py:98` | conflicting framing at `registry.py:118` (see F-04) |
| `caseB_alpha` | `2.59e-13 cm^3/s` | `read_param.py:350` | — |
| `mCloud_input`, `mCluster` | "no static default … 0.0 is a placeholder" | `registry.py:409` | — |
| `simplify` `nmin` | `simplify_npoints` from `default.param`, "or to 100 if absent" | `dictionary.py:457` | — |
| `shorten_display` `nshow` | `3` | `dictionary.py:388` | — |
| array-shorten threshold in `__str__` | ">10 elements" | `dictionary.py:408` | — |
| `simplify` R² warn threshold | `0.9` | `dictionary.py:457` | yes — `dictionary.py:516` ("R² < 0.9"), same value, different mechanism (F-05) |
| prominence-mandatory threshold | "≥ 5 % of the y-range" | `dictionary.py:457` | — |
| `reset_keys` fill value | `np.nan` | `dictionary.py:982` | — |
| f_kappa sweep ceiling | `64` | `fkappa_auto.py:1` | yes — `fkappa_auto.py:44` ("Largest f_kappa the sweep tested") |
| `theta` fire threshold | `Lloss/Lgain > 0.95` | `fkappa_auto.py:1` | — |
| composition anchors | `x_He=0.1`, `Z_He=2`, `Z_He_shell=1.0` | `read_param.py:308`, `:309`, `:330` | — |
| historical mu encodings | `14/11`, `14/23`, `14/6`, `1.4` | `read_param.py:305` | yes — `registry.py:1` stores `mu_atom` etc. as `"14/11"` **source strings** (F-02) |
| runtime materialization count | "106 items: 9 … True and 97 … False" | `registry.py:625` | yes — `read_param.py:465`, identical numbers |
| registry size | "fully populated (200 specs)" | `param_spec.py:1` | — |
| fkappa calibration `nISM` | `0.1` | `fkappa_auto.py:1` | — |

`materialize_runtime` also documents the metadata defaults applied to new items:
`info=spec.info`, `ori_units=spec.unit` "(or `\"N/A\"` when unitless)",
`exclude_from_snapshot=spec.exclude_from_snapshot`, value `copy.deepcopy(spec.default)`
(`registry.py:625`).

## 7. File formats and I/O contracts

**`dictionary.jsonl`** (`dictionary.py:765`): "line-delimited JSON", "Line 0: snapshot \"0\" as JSON
object; Line 1: snapshot \"1\" …". `load_snapshots` restates it as an invariant: "Line N in file =
snapshot str(N)" (`dictionary.py:875`). Write behaviour: "If flush_count == 0 and file exists:
overwrite (fresh run) - Else: append new snapshots"; the fresh-run branch "delete[s] existing files
(jsonl AND metadata) so we never end up with a stale metadata.json next to a new simulation's
snapshots" (`:807`). Pending IDs are sorted before writing (`:818`).

**`metadata.json`** (`dictionary.py:765`, `:821`): "one record per run", containing
`trinity._output.run_constants.RUN_CONST_KEYS` plus "a `_metadata_version` field for forward-compat";
written "on the very first flush of the run"; atomicity is "temp+rename" and the JSON is
pretty-printed. Rehydration on load uses `setdefault` semantics — "per-snapshot value wins when both
are present (legacy files keep loading identically)" (`:903`).

**`metadata.json[termination_debug]`** (`dictionary.py:356`): "Mirror the last-2-snapshot debug block";
"Phase 5+ writes a structured block instead of the legacy `termination_debug.txt` file"; formatted by
`python -m trinity._output.show_run`. `read_param.py:494` adds that the legacy
`<model_name>_summary.txt` is "no longer written".

**`debug_snapshot.json`** (`dictionary.py:1023`, `:1129`): "Saves ALL keys without any
cleaning/simplification", "Skips non-serializable objects (interpolators, etc.) gracefully", "Always
OVERWRITES the file", "Can be called from anywhere without `params['path2output']`"; output dir
falls back to `params['path2output']` or the current dir. On load, arrays come back as numpy and
test usage filters keys starting with `'_'`.

**Storage policy** (`dictionary.py:201`): "All data (scalars, arrays) stored inline in JSON"; "Each
snapshot is one line"; "Append-only writes ensure O(1) flush performance (vs O(n²) in old version)";
"Required key before saving: `params[\"path2output\"].value` must exist and point to the output
directory". `_to_json_ready_value` restates "All arrays are inlined as lists (no HDF5)"
(`dictionary.py:554`).

## 8. Snapshot routing — the three independent axes

`param_spec.py:99` states the routing model and one hard invariant:

- `run_const` → written once to `metadata.json` (`RUN_CONST_KEYS`)
- `metadata_exclude` → blocked from `metadata.json` ("paths, loaded tables, empty array placeholders")
- `exclude_from_snapshot` → the live `DescribedItem` flag, omit from the per-snapshot jsonl stream
- "The three axes are independent" and **"run_const ∩ metadata_exclude is always empty (a key is
  written to metadata or blocked from it, never both)."**

`registry.py:1` claims the registry is "the single source of truth for run-const / metadata-exclude
membership" as of Phase 5, with `run_const_keys()` / `metadata_exclude_keys()` as "drop-in
replacement[s]" for the hand-curated lists in `trinity._output.run_constants` (`registry.py:665`,
`:674`). `param_spec.py:28` insists categories are descriptive only and "do NOT drive run-const /
metadata-exclude membership".

Step 9's stated policy (`read_param.py:447`): "Only track time-varying quantities in snapshots /
Exclude initial conditions and constants to save memory", with an explicit carve-out: "Cloud profile
constants — needed for radial profile reconstruction" (`:451`).

## 9. `simplify` — the fullest contract in the slice

`dictionary.py:457` claims downsampling of `y(x)` to `nmin` points **preserving**:
endpoints; "sharp bends (points where the Menger curvature exceeds `grad_inc` on rescaled [0, 1]
axes — unit-free threshold)"; "local extrema (sign-change points of the first derivative)"; points by
cumulative distance in y (uniform arc-length); "topologically persistent extrema — any peak/trough
with prominence ≥ 5 % of the y-range is *mandatory* and never dropped"; "an x-uniform coverage
skeleton — one feature-pool point per equal-width x-chunk is promoted to mandatory".

Explicit I/O contract: input 1-D array-likes of equal length, "ascending, descending, or
non-monotonic in x"; output two `np.ndarray`s of equal length with endpoints preserved, "Values come
back in the caller's original positional order"; "Raises `ValueError` if `len(x_arr) != len(y_arr)`".
Remaining slots filled "in hierarchical-bisection priority, which keeps the subset stable under small
changes in `nmin`". Delegation claim: "Delegates to the standalone `simplify` module (no TRINITY
dependencies)."

## 10. Derived-parameter claims in Step 6

- Composition (`read_param.py:302`): "`x_He` (n_He/n_H) and `Z_He` (helium ionisation state) are the
  **single source of truth** for the gas composition. Exact-rational (Fraction) arithmetic keeps the
  `mu_*` values byte-identical to the historical 14/11, 14/23, 14/6, 1.4 encodings when x_He=0.1,
  Z_He=2 (verified). The 'm_H' unit factor matches what Step 4 applied to the former numeric
  defaults." Per-line unit annotations: "mass per H nucleus [m_H]" (`:310`), "neutral mean
  mass/particle" (`:311`), "ionised" (`:312`), "molecular" (`:313`), "electrons per H nucleus,
  n_e/n_H" (`:314`), "m_H in Msun" (`:315`).
- Shell ionisation (`read_param.py:328`): "Shell / HII region (~1e4 K) is singly ionised
  (`Z_He_shell`), unlike the hot doubly-ionised bubble".
- `caseB_alpha` (`read_param.py:350`): "fixed at its ~1e4 K value and is NOT recomputed from
  `TShell_ion`. Since alpha_B(T) ~ T^-0.7, moving the ionised-shell temperature far from ~1e4 K leaves
  the Stroemgren balance (`n_IF_Str`) and `P_HII`/`F_HII` internally inconsistent unless `caseB_alpha`
  is adjusted to match. Warn once at load."
- Dust (`read_param.py:365`): "Dust cross-section scaling with metallicity" — no formula, no key names.
- `model_name` (`read_param.py:371`): "Use filename as model name if not specified."
- Mass rebinding (`read_param.py:375`), quoted in full because it is the densest contract in the file:
  "NOTE: `params['mCloud']` is rebound here. Upstream of this block — in the .param file and the
  folder name — mCloud is the pre-SFE input GMC mass. Downstream — throughout the simulation, in
  metadata.json, and in every rehydrated snapshot — it is the post-SFE residual cloud mass. The pre-SFE
  input is preserved as `mCloud_input` and the star-formed portion as `mCluster`; **invariant:
  `mCloud_input == mCloud + mCluster`**. Downstream analysis that wants the input value should read
  `mCloud_input`, not back out `mCloud / (1 - sfe)`."
  `registry.py:409` gives the formulas: "`mCloud_input` = input mCloud; `mCluster` = `mCloud_input * sfe`".

## 11. `fkappa_auto` — calibration provenance and stated limits

`fkappa_auto.py:1` is the most heavily cited block in the slice:

- **What `auto` means**: "the smallest Spitzer-conduction multiplier f_kappa that made the
  `cooling_balance` energy->momentum trigger fire (theta = Lloss/Lgain > 0.95, the **Lancaster+2021**
  efficiently-cooled band)".
- **Provenance**: "as MEASURED on the 819-run (mCloud, sfe, nCore) sweep of **2026-06-29**
  (`docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv`, column `f_kappa_fire_measured`; grid
  analysis in `docs/dev/transition/pdv-trigger/data/make_fkappa_theta1_collapse.py`)".
- **Negative result cited**: "The sweep refuted a single-variable f_kappa(n_H) law (spread up to 32x
  across mCloud/sfe at fixed density), so the lookup keeps all three axes: trilinear interpolation in
  (log10 mCloud_input, log10 sfe, log10 nCore) of log10 f_kappa_fire."
- **Axis semantics**: "mCloud axis is the PRE-star-formation input mass (`mCloud_input`) … not the
  post-SFE `mCloud`."
- **Extrapolation**: "Coordinates outside the calibrated hull are clamped to it, with a warning."
- **Censoring**: "Censored cells (the diffuse/high-SFE corner where nothing up to f_kappa=64 fired) are
  filled with the sweep ceiling 64; a resolved value at that ceiling means the calibration could NOT
  demonstrate firing, and the resolver warns accordingly."
- **Regime limit**: "Calibration was measured on flat power-law clouds (densPL, alpha=0), nISM=0.1,
  hybr solver. Other profiles resolve on the same table with no measured guarantee (a warning is
  logged)."

Grid extent per the table comments: mCloud ∈ {1e5, 1e6, 1e7} (`:51`, `:57`, `:63`), sfe ∈ {0.03, 0.1,
0.3} (`:53`, `:54`, `:56`), with censored-cell counts noted at `:56` ("2 censored"), `:62` ("1
censored"), `:66` ("1 censored"), `:68` ("2 censored").

`fkappa_fire` (`:76`): "Pure lookup (no params dict): trilinear in log10 space, coordinates clamped to
the calibrated hull. **Returns a float >= 1.**"
`resolve_fkappa_auto` (`:98`): "Registry resolver for `cooling_boost_kappa` (read_param Step 7).
Numeric values pass through UNTOUCHED (the default 1.0 path stays byte-identical). The string 'auto'
resolves via `fkappa_fire` against `mCloud_input` / `sfe` / `nCore`."
Units note (`:120`): "nCore is already in code units (pc^-3) by Step 7; the grid is cm^-3."

## 12. Citations and external-format references (complete list)

| Reference | Cited for | Cite |
|---|---|---|
| **Lancaster+2021** | the "efficiently-cooled band", i.e. the `theta = Lloss/Lgain > 0.95` threshold | `fkappa_auto.py:1` |
| **Sutherland-Dopita** (CIE cooling file) | "ZCloud == 0.15 auto-pins to the Sutherland-Dopita file" | `read_param.py:415` |
| **SB99 / Starburst99** | bundled SPS grid format: CSV, "canonical 7-column SB99 layout", rotation=1, ZCloud=1, mass=1e6 Msun | `registry.py:253` |
| **OPIATE** cubes | non-CIE cooling directory format under `lib/default/opiate/` | `registry.py:242` |
| **WARPFIELD "Problem 2"** | a velocity-structure diagnostic key group, "diagnostic only" | `dictionary.py:1224` |
| `SOURCE_TERM_DESIGN.md` | definition of f_A as "the interface source-term boost" | `registry.py:118` |
| `docs/dev/archive/sb99-refactor-audit.md` §9 | rationale for the bundled-SPS default and its rejections | `registry.py:253` |
| `docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv` (col `f_kappa_fire_measured`) + `make_fkappa_theta1_collapse.py` | the entire f_kappa lookup table | `fkappa_auto.py:1` |
| `trinity/phase1b_energy_implicit/get_betadelta.py` | the two betadelta solver modes | `registry.py:152` |
| `trinity.cloud_properties.validate_gmc` | consumer of `rCloud_max` | `registry.py:205` |
| `trinity.sps.sps_columns.CanonicalSpec` / `build_user_column_map` | the pattern `ParamSpec` mirrors; the column-map builder | `param_spec.py:1`, `:70` |
| `tools/gen_default_param.py` | "(Phase 3+ regenerates `default.param` from the registry)" | `param_spec.py:1` |
| `test/test_registry.py` + named guards | pinning the spec set, sentinel/resolver pairing, `consumed_by` targets, `active_when` set | `param_spec.py:1`, `:78`, `:128`, `:137` |
| `trinity._output.run_constants` (`RUN_CONST_KEYS`, `METADATA_EXCLUDE`, `DROPPED_IN_V2`) | snapshot/metadata routing | `dictionary.py:578`, `:588`, `param_spec.py:99` |
| `python -m trinity._output.show_run` | human-readable formatter for the metadata blocks | `dictionary.py:356`, `read_param.py:496` |
| branch `hotfix/metadata-excluding` | a past regression where profile data was "silently lost from dictionary.jsonl" | `dictionary.py:61` |

## 13. "must / always / never / guaranteed" statements (verbatim inventory)

- "the snapshot writer **must NEVER strip** these, or their data is silently lost from
  dictionary.jsonl" — `dictionary.py:61`
- "Run-constants … are written to `metadata.json` once per run, and stripped from **every**
  per-snapshot dict here" / "**never** appear in per-snapshot dicts" — `dictionary.py:578`, `:588`
- "the profile arrays in that set **must survive**" — `dictionary.py:594`
- "`params[\"path2output\"].value` **must exist**" — `dictionary.py:201`
- "Endpoints and every high-prominence extremum are **always retained**" / "*mandatory* and **never
  dropped**" — `dictionary.py:457`
- "Always OVERWRITES the file" — `dictionary.py:1023`
- "Keys that don't exist in the dictionary are **silently skipped**" — `dictionary.py:982`
- "Missing keys are **silently skipped**" (dataclass mode) — `dictionary.py:1233`
- "the invariant *\"the spec is in `params` iff `active_when(params)` returns True\"* is restored" —
  `registry.py:589`
- "**Must run after** Step 5 … and **before** Step 9" — `registry.py:589`
- "**Must run AFTER** Step 9" — `registry.py:625`; "Must run after Step 6" — `read_param.py:409`
- "each predicate matches **exactly one** of the two profile families" — `registry.py:75`
- "Mutually exclusive with `consumed_by`: a spec either carries its own resolver OR delegates to
  another spec's, **never both**" — `param_spec.py:116`
- "run_const ∩ metadata_exclude is **always empty** … **never both**" — `param_spec.py:109`
- "**invariant**: `mCloud_input == mCloud + mCluster`" — `read_param.py:382`
- "Later steps (6/8/10) … **must NOT** silently replace any of these" — `read_param.py:279`
- "Returns a float **>= 1**" — `fkappa_auto.py:76`
- "Numeric values pass through **UNTOUCHED**" — `fkappa_auto.py:98`
- "behavior is **byte-identical**" (resolvers) — `registry.py:227`; "**byte-identical** to the
  historical … encodings … (verified)" (mu_*) — `read_param.py:305`
- "this convenience artefact **must never break** the run" — `dictionary.py:334`
- "Does **NOT** cover: kill -9 (SIGKILL) … os._exit()" — `dictionary.py:263`

## 14. Findings — internal contradictions and doc-level defects

Detailed below; the machine-readable list follows.

**Resolver inventory disagrees across files.** `registry.py:565` and `param_spec.py:120` both say
exactly three resolvers exist today, naming them; `fkappa_auto.py:98` declares itself "Registry
resolver for `cooling_boost_kappa` (read_param Step 7)". `read_param.py:405` also lists only the
three. Whichever is right, one of them is stale — and the consequence is not cosmetic: `resolve_all`
leans on the three-resolver inventory to assert "no inter-dependencies in that order", a claim that a
fourth resolver reading `mCloud_input`/`sfe`/`nCore` would invalidate.

**`mu_*` has two claimed provenances.** `registry.py:1` says input specs carry "the *raw source
string* exactly as it would appear in `default.param`" and that "Fraction-encoded constants
(`mu_atom` etc.) are stored as `\"14/11\"` strings"; `param_spec.py:41` lists `mu_*` under
"Input-side (declared in default.param)". But `read_param.py:302` says `x_He`/`Z_He` are "the single
source of truth for the gas composition" and derives the `mu_*` values in Step 6. Both cannot be the
source of truth. The set-once-derived listing at `param_spec.py:43` ("rCloud, nEdge, tSF, mCluster,
mCloud_input, densBE_Teff") omits every quantity `read_param.py:302`–`:365` describes deriving.

**The `mCloud` back-out warning contradicts the stated invariant.** `read_param.py:375` asserts
`mCloud_input == mCloud + mCluster` and `registry.py:409` gives `mCluster = mCloud_input * sfe`.
Those two make `mCloud / (1 - sfe)` *exactly* `mCloud_input`. Yet the same comment warns readers not
to back it out that way. Either the invariant is approximate/conditional (and unstated), or the
warning is stale.

**`cooling_boost_kappa` "active" is self-defeating as written.** `registry.py:118` says a *number*
counts as "kappa active", and `fkappa_auto.py:98` says the default is the number `1.0` (a documented
no-op). Read literally, every run with `f_A` set trips the double-boost warning against a knob that
is off.

**Two mechanisms for one R² warning.** `dictionary.py:457` says `_simplify` "emits a `UserWarning`"
below 0.9; `dictionary.py:516` says it will "log it as a warning regardless of phase or snapshot
count". `warnings.warn` and a logger call are different channels with different suppression
behaviour; only one can be what happens.

**Two complexity claims for `flush`.** `dictionary.py:201`: "Append-only writes ensure O(1) flush
performance"; `dictionary.py:765`: "Performance: O(pending_snapshots)".

**Never-strip vs. strip-defensively.** `dictionary.py:61` says the snapshot writer "must NEVER strip"
the `metadata_exclude`-flagged profile source keys, citing a regression that silently lost data;
`dictionary.py:588` says `METADATA_EXCLUDE` keys *are* "stripped defensively", with a prose carve-out
for exactly those profile arrays. The safety of the whole arrangement rests on a hand-maintained
list matching a hand-maintained set — which is precisely the failure mode the earlier comment
records having already shipped once.

**Silent-drop paths documented as features.** Four places document losing data without failing:
`default.param` malformed lines skipped (`read_param.py:159`) against a module docstring promising
"Robust error handling with line numbers and helpful messages" (`:3`); non-JSON-encodable values
"log a warning and skip the key" during the `metadata.json` build (`dictionary.py:836`); duplicate
snapshots suppressed on `(t_now, R2)` equality alone (`dictionary.py:712`, with detection skipped
entirely if either key is absent, `:730`); `reset_keys` and `updateDict` dataclass mode skipping
unknown keys (`dictionary.py:982`, `:1233`) — a typo in a key name is a no-op.

**Phase numbering is internally inconsistent.** `registry.py:1` maps Step 10 → "Phase 9", but
`materialize_runtime`'s own docstring calls itself the "Phase-8/9 entry point" (`registry.py:625`),
and `registry.py:409` says `mCloud_input`/`mCluster` are "materialised by the derived-init resolver in
Phase 7/10" while the module docstring says Phase 10 is not yet wired.

**Counts that may not add up.** `registry.py:253` and `param_spec.py:70` both say there are 13
`sps_col_*` specs, while the same `registry.py:253` block describes "the canonical 7-column SB99
layout". `param_spec.py:1` claims "200 specs"; `registry.py:625` claims Step 10 materializes exactly
106 items. These are hard numbers in prose with no stated mechanism keeping them current beyond
`test/test_registry.py`.

**Vague or unfalsifiable as written.** `simplify`'s "Output size is **normally** `nmin`" plus two
independent mandatory-promotion rules (`dictionary.py:457`); `caseB_alpha`'s "far from ~1e4 K" with no
trigger condition for the once-at-load warning (`read_param.py:350`); "The default rejects
combinations the bundled cooling tables can't fulfill" with no enumeration (`registry.py:253`); "Dust
cross-section scaling with metallicity" with neither key nor formula (`read_param.py:365`).

**Orphan / unlinked names.** `grad_inc` (the curvature threshold, `dictionary.py:457`) appears
nowhere else in the slice — no default, no spec listing. The trigger name `cooling_balance`
(`fkappa_auto.py:1`) appears nowhere else; `cooling_boost_mode`'s legal value set is never
enumerated beyond `none`. `phaseSwitch_LlossLgain` is listed as an input parameter
(`param_spec.py:42`) but its relation to the hard-coded `theta > 0.95` in `fkappa_auto.py:1` is never
stated. `param_spec.py:60` retains a whole category — "Parsed for back-compat only, never consumed …
(currently none; kept for future use)" — with no members.

**Commented-out key groups.** `COOLING_PHASE_KEYS` (`dictionary.py:1179` onwards) carries nine
commented-out members — `'bubble_Tavg'`, `'bubble_T_r_Tb'`, `'bubble_mass'`, `'bubble_r_Tb'`
(`:1200`–`:1203`) and `'bubble_v_arr'`, `'bubble_T_arr'`, `'bubble_dTdr_arr'`, `'bubble_r_arr'`,
`'bubble_n_arr'` (`:1218`–`:1222`) — with no comment explaining whether their exclusion from the
post-implicit-phase reset is deliberate.

**Round-trip metadata loss.** `load_snapshot` (`dictionary.py:929`) claims to reconstruct "scalars
directly into `DescribedItem(value)`" and "list values back into numpy arrays" — so `info`,
`ori_units` and `exclude_from_snapshot` are not restored, and any genuinely list-valued parameter
returns as an `ndarray`. The prose states the behaviour but never flags the asymmetry with the
`DescribedItem` contract at `dictionary.py:99`.

**Missing prose.** The slice names `trinity/_input/__init__.py` as one of seven files; the prose file
contains no section for it, i.e. the package's public surface is undocumented at the module level.

```json
[
  {
    "id": "S12a-B-01",
    "file": "trinity/_input/registry.py",
    "line": 565,
    "class": "other",
    "severity": "S3",
    "claim": "The registry and param_spec both assert exactly three resolvers exist today, but fkappa_auto declares itself a fourth registry resolver running at the same step.",
    "evidence": "registry.py:565 'the three current resolvers (path2output, path_cooling_nonCIE, sps_path) carry no inter-dependencies in that order'; param_spec.py:120 'Three specs carry resolvers today (path2output, path_cooling_nonCIE, sps_path)'; read_param.py:405 'Path + SPS-bundle sentinels resolve via their registry resolvers (path2output, path_cooling_nonCIE, sps_path)'; fkappa_auto.py:98 'Registry resolver for ``cooling_boost_kappa`` (read_param Step 7).'",
    "expected": "One consistent resolver inventory; if cooling_boost_kappa carries a resolver, resolve_all's and param_spec's inventories must name it.",
    "failure_scenario": "A maintainer trusting 'three resolvers, no inter-dependencies' adds or reorders a spec and breaks the fkappa resolver's dependence on mCloud_input/sfe/nCore without any doc signalling the constraint.",
    "repro": "grep -n 'three current resolvers\\|Three specs carry resolvers' trinity/_input/registry.py trinity/_input/param_spec.py; grep -n 'Registry resolver' trinity/_input/fkappa_auto.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-02",
    "file": "trinity/_input/registry.py",
    "line": 1,
    "class": "state",
    "severity": "S3",
    "claim": "mu_* is documented both as an input spec whose default is a fraction source string in default.param and as a Step-6 derived quantity whose single source of truth is x_He/Z_He.",
    "evidence": "registry.py:1 'Fraction-encoded constants (``mu_atom`` etc.) are stored as ``\"14/11\"`` strings per the agreed representation; they round-trip to the identical float.'; param_spec.py:41 lists '# mu_*, gamma_adia, G, k_B, c_light, dust_*, caseB_alpha' under '---- Input-side (declared in default.param) ----'; read_param.py:302 'x_He (n_He/n_H) and Z_He (helium ionisation state) are the single source of truth for the gas composition.'",
    "expected": "Either mu_* is an input parameter a user can set, or it is derived and any default.param entry is dead; the prose must say which.",
    "failure_scenario": "A user sets mu_ion in their .param expecting it to take effect; Step 6 silently overwrites it from x_He/Z_He — the same class of bug the include_PHII guard at read_param.py:474 was added to catch.",
    "repro": "grep -n 'single source of truth' trinity/_input/read_param.py trinity/_input/registry.py; grep -n 'mu_' trinity/_input/param_spec.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-03",
    "file": "trinity/_input/param_spec.py",
    "line": 43,
    "class": "other",
    "severity": "S4",
    "claim": "The 'set-once derived (Step 6)' category listing omits most of what read_param Step 6 documents itself as deriving.",
    "evidence": "param_spec.py:43 '# ---- Set-once derived (read_param.py Step 6) ---- # rCloud, nEdge, tSF, mCluster, mCloud_input, densBE_Teff'; read_param.py:302-365 documents deriving mu_* , chi_e, the singly-ionised shell mu and electron factor, dust cross-section scaling, and model_name in the same Step 6.",
    "expected": "The category comment should enumerate every Step-6 derived key, or say it is illustrative.",
    "failure_scenario": "A reader auditing which keys are user-settable trusts the list and misclassifies mu_*/chi_e as pure inputs.",
    "repro": "Compare param_spec.py:43-44 against the Step 6 block comments in read_param.py:299-400.",
    "confidence": "high"
  },
  {
    "id": "S12a-B-04",
    "file": "trinity/_input/registry.py",
    "line": 118,
    "class": "other",
    "severity": "S3",
    "claim": "The f_A validator's definition of 'kappa active' includes any number, but the documented default of cooling_boost_kappa is the number 1.0 (an explicit no-op) — so the double-boost warning would fire on the default configuration.",
    "evidence": "registry.py:118 'combining it with cooling_boost_mode != none or an active cooling_boost_kappa double-counts interface cooling. Validators run BEFORE resolvers (read_param Steps 5 vs 7), so cooling_boost_kappa is still its raw value here -- a number or the string \\'auto\\'; both count as \\'kappa active\\'.'; fkappa_auto.py:98 'Numeric values pass through UNTOUCHED (the default 1.0 path stays byte-identical).'",
    "expected": "'kappa active' should be defined as kappa != 1.0 or 'auto', otherwise the warning is unconditional noise for every f_A user.",
    "failure_scenario": "Users see a double-boost warning on a configuration with only one knob set, learn to ignore the warning, and miss the genuine double-boost case.",
    "repro": "grep -n \"kappa active\" trinity/_input/registry.py; grep -n 'default 1.0' trinity/_input/fkappa_auto.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-05",
    "file": "trinity/_input/dictionary.py",
    "line": 457,
    "class": "other",
    "severity": "S4",
    "claim": "The simplify R2 diagnostic is documented as a UserWarning in the docstring and as a log warning in the adjacent comment — two different channels for the same 0.9 threshold.",
    "evidence": "dictionary.py:457 'computes the linear-interpolation R\\u00b2 of the simplified curve against the original grid and emits a ``UserWarning`` if it falls below 0.9'; dictionary.py:516 'If the simplified curve diverges from the original (R\\u00b2 < 0.9) that\\'s a real signal that simplify_npoints is too small — log it as a warning regardless of phase or snapshot count.'",
    "expected": "One documented emission channel; warnings.warn and logger.warning differ in suppression, capture in pytest, and visibility in sweeps.",
    "failure_scenario": "A sweep harness filters UserWarnings expecting to catch degraded simplifications, but the signal only goes to the log (or vice versa), so a too-small simplify_npoints ships unnoticed.",
    "repro": "grep -n 'UserWarning' trinity/_input/dictionary.py; grep -n 'R\\u00b2 < 0.9' trinity/_input/dictionary.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-06",
    "file": "trinity/_input/dictionary.py",
    "line": 201,
    "class": "other",
    "severity": "S4",
    "claim": "flush() is documented with two different complexities in the same file.",
    "evidence": "dictionary.py:201 'Append-only writes ensure O(1) flush performance (vs O(n\\u00b2) in old version)'; dictionary.py:765 'Performance: O(pending_snapshots) - only writes new data, never reads existing file. This is a MASSIVE improvement over the old O(n\\u00b2) behavior.'",
    "expected": "State the cost per flush (O(pending)) and per snapshot (amortised O(1)) unambiguously.",
    "failure_scenario": "None at runtime; a reader sizing the flush interval from the O(1) claim mis-models memory held in the pending buffer.",
    "repro": "grep -n 'O(1)\\|O(pending_snapshots)' trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-07",
    "file": "trinity/_input/dictionary.py",
    "line": 61,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The prose states both that METADATA_EXCLUDE-flagged profile source keys must NEVER be stripped by the snapshot writer, and that METADATA_EXCLUDE keys ARE stripped defensively there — reconciled only by a hand-maintained carve-out list.",
    "evidence": "dictionary.py:61 'Several of them carry ``metadata_exclude`` in the ParamSpec registry ... the snapshot writer must NEVER strip these, or their data is silently lost from dictionary.jsonl (regression fixed in hotfix/metadata-excluding).'; dictionary.py:588 '``METADATA_EXCLUDE`` keys (paths, function tables) are stripped defensively ... but the profile arrays in that set must survive'",
    "expected": "The carve-out should be derived from the registry flags (e.g. a dedicated 'snapshot_required' axis), not from a second literal list that must be kept in sync with METADATA_EXCLUDE.",
    "failure_scenario": "A new metadata_exclude'd profile array is added to the registry but not to the carve-out list; its data vanishes from dictionary.jsonl silently — exactly the regression the comment says already shipped once.",
    "repro": "grep -n 'must NEVER strip\\|stripped defensively\\|must survive' trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-08",
    "file": "trinity/_input/dictionary.py",
    "line": 836,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "During the metadata.json build, values that fail JSON encoding are logged and skipped rather than raising — a run-constant can be absent from metadata.json with only a log line.",
    "evidence": "dictionary.py:836 'Defensive serialization: if the value can\\'t be JSON-encoded (e.g. an unexpected interpolator object snuck into ``params``), log a warning and skip the key rather than crashing the whole flush.'",
    "expected": "A missing RUN_CONST key should be loud, since dictionary.py:578 claims run-constants are stripped from every snapshot and only live in metadata.json.",
    "failure_scenario": "A run-constant is silently dropped from metadata.json; the reader's rehydration (dictionary.py:903) then finds neither a per-snapshot value nor a metadata value, and downstream analysis silently sees a missing key.",
    "repro": "grep -n 'Defensive serialization' trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-09",
    "file": "trinity/_input/dictionary.py",
    "line": 712,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Snapshot de-duplication is keyed on (t_now, R2) only, and is skipped entirely if either key is absent.",
    "evidence": "dictionary.py:712 'Duplicate guard: - If the last saved snapshot has the same t_now and R2, it will not save again.'; dictionary.py:730 '# If t_now/R2 not present, skip duplicate detection'",
    "expected": "Documented rationale for why (t_now, R2) uniquely identifies simulation state, or a broader key.",
    "failure_scenario": "A phase transition or solver retry that changes energy/pressure/force state at unchanged t_now and R2 is silently not recorded, leaving a gap in dictionary.jsonl that no error surfaces.",
    "repro": "grep -n 'Duplicate guard' -A3 trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-10",
    "file": "trinity/_input/dictionary.py",
    "line": 982,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Both bulk helpers document silently ignoring key names that are not in the dictionary.",
    "evidence": "dictionary.py:982 'Notes ----- Keys that don\\'t exist in the dictionary are silently skipped.'; dictionary.py:1233 'When using dataclass mode, only fields that exist in the dictionary are updated. Missing keys are silently skipped.'",
    "expected": "At minimum a debug/warning log naming the skipped keys; a typo in a COOLING_PHASE_KEYS entry or a renamed dataclass field is otherwise undetectable.",
    "failure_scenario": "A dataclass field is renamed (e.g. an SPS feedback field); updateDict silently stops writing it, and the params dict keeps a stale value from the previous timestep for the rest of the run.",
    "repro": "grep -n 'silently skipped' trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-11",
    "file": "trinity/_input/read_param.py",
    "line": 375,
    "class": "state",
    "severity": "S3",
    "claim": "The mCloud rebinding note warns against backing out mCloud_input as mCloud/(1-sfe), but the same block's stated invariant plus the registry's formula make that identity exact.",
    "evidence": "read_param.py:382 'invariant: mCloud_input == mCloud + mCluster. Downstream analysis that wants the input value should read mCloud_input, not back out mCloud / (1 - sfe).'; registry.py:409 'read_param Step 6 computes them from the input mCloud and sfe (mCloud_input = input mCloud; mCluster = mCloud_input * sfe)'",
    "expected": "Either the invariant is conditional (e.g. sfe is itself adjusted, or mCloud is further modified downstream) and that should be stated, or the warning is stale.",
    "failure_scenario": "A reader concludes the invariant is only approximate and stops relying on mCloud_input == mCloud + mCluster in a consistency check; or, conversely, relies on the identity in a regime where sfe was mutated after Step 6.",
    "repro": "grep -n 'not back out\\|mCloud_input ==' trinity/_input/read_param.py; grep -n 'mCluster = mCloud_input' trinity/_input/registry.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-12",
    "file": "trinity/_input/read_param.py",
    "line": 159,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Malformed lines in default.param are silently skipped, contradicting the module's advertised robust, line-numbered error handling.",
    "evidence": "read_param.py:159 '# Skip malformed lines in default.param'; read_param.py:3 'Key features: ... - Robust error handling with line numbers and helpful messages'",
    "expected": "A malformed line in the schema file is a repo defect and should raise (or at minimum warn with the line number), since a dropped key silently removes a parameter from the schema.",
    "failure_scenario": "A typo'd default.param line drops a key from the schema; a user .param that sets it then fails the Step-3 'key must exist in default.param' check with a confusing 'unknown parameter' error, or the parameter silently reverts to the registry/runtime default.",
    "repro": "grep -n 'Skip malformed lines' trinity/_input/read_param.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-13",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 120,
    "class": "units",
    "severity": "S2",
    "claim": "The f_kappa lookup grid is in cm^-3 while the parameter it indexes (nCore) is in code units pc^-3 by the time the resolver runs; the conversion is asserted in a one-line comment only.",
    "evidence": "fkappa_auto.py:120 '# nCore is already in code units (pc^-3) by Step 7; the grid is cm^-3'; fkappa_auto.py:39 '# Sweep grid axes (log10). mCloud is the pre-SFE input mass.'; read_param.py:256 '# Convert units to astronomy units [Msun, pc, Myr]'",
    "expected": "The unit contract for the lookup should be stated on fkappa_fire's docstring (which says only 'trilinear in log10 space'), not just at the single call site.",
    "failure_scenario": "A second caller of fkappa_fire passes nCore in code units without the conversion; a ~3.09e18^3 error in the density coordinate silently clamps to the hull edge and returns a wrong f_kappa with only a hull-clamp warning.",
    "repro": "grep -n 'code units (pc^-3)' trinity/_input/fkappa_auto.py; read fkappa_fire's docstring at trinity/_input/fkappa_auto.py:76",
    "confidence": "high"
  },
  {
    "id": "S12a-B-14",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 1,
    "class": "regime",
    "severity": "S2",
    "claim": "cooling_boost_kappa=auto is calibrated only for densPL with alpha=0, nISM=0.1, and the hybr solver; all other configurations resolve against the same table with an explicitly disclaimed guarantee.",
    "evidence": "fkappa_auto.py:1 '* Calibration was measured on flat power-law clouds (densPL, alpha=0), nISM=0.1, hybr solver. Other profiles resolve on the same table with no measured guarantee (a warning is logged).'",
    "expected": "Documented; but the fallback is silent-by-warning for a value that directly scales conduction, on regimes (densBE, alpha!=0, legacy solver) the code otherwise supports as first-class.",
    "failure_scenario": "A densBE run with cooling_boost_kappa=auto silently receives an f_kappa calibrated on a different profile family, shifting the energy->momentum transition time with only a log warning as evidence.",
    "repro": "grep -n 'no measured guarantee' trinity/_input/fkappa_auto.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-15",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 1,
    "class": "regime",
    "severity": "S2",
    "claim": "Censored grid cells (no firing up to f_kappa=64) are filled with the ceiling 64, so the returned value is indistinguishable from a measured 64 except via a warning.",
    "evidence": "fkappa_auto.py:1 '* Censored cells (the diffuse/high-SFE corner where nothing up to f_kappa=64 fired) are filled with the sweep ceiling 64; a resolved value at that ceiling means the calibration could NOT demonstrate firing, and the resolver warns accordingly.'; fkappa_auto.py:44 '# Largest f_kappa the sweep tested; censored cells (never fired) carry it.'",
    "expected": "A right-censored measurement is a lower bound, not a value; trilinear interpolation across a censored cell mixes bounds with measurements.",
    "failure_scenario": "A run near the diffuse/high-SFE corner interpolates between a real f_kappa and a censoring bound, producing a conduction multiplier that has no calibration behind it; the run proceeds normally.",
    "repro": "grep -n 'censored' trinity/_input/fkappa_auto.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-16",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 51,
    "class": "regime",
    "severity": "S2",
    "claim": "The calibrated hull spans only mCloud in {1e5,1e6,1e7} and sfe in {0.03,0.1,0.3}; anything outside is clamped to the hull edge with a warning.",
    "evidence": "fkappa_auto.py:51 '# mCloud = 1e5', :57 '# mCloud = 1e6', :63 '# mCloud = 1e7'; :53 '# sfe = 0.03', :54 '# sfe = 0.1', :56 '# sfe = 0.3 (2 censored)'; fkappa_auto.py:1 '* Coordinates outside the calibrated hull are clamped to it, with a warning.'",
    "expected": "Documented; the finding is that the module also reports the sweep 'refuted a single-variable f_kappa(n_H) law (spread up to 32x across mCloud/sfe at fixed density)' — i.e. the axes it clamps are exactly the axes with the largest spread.",
    "failure_scenario": "A sweep at mCloud=1e8 or sfe=0.5 clamps to the 1e7 / 0.3 edge and inherits an f_kappa that the module's own 32x-spread finding says cannot be extrapolated.",
    "repro": "grep -n 'clamped to it\\|32x' trinity/_input/fkappa_auto.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-17",
    "file": "trinity/_input/registry.py",
    "line": 565,
    "class": "state",
    "severity": "S2",
    "claim": "resolve_all's ordering guarantee ('no inter-dependencies in that order') is stated only for the three named resolvers, while a fourth documented resolver reads three other params' values.",
    "evidence": "registry.py:565 'the three current resolvers (path2output, path_cooling_nonCIE, sps_path) carry no inter-dependencies in that order — ... so the one cross-key ordering edge (refmass-after-path) lives inside a single resolver rather than across iterations.'; fkappa_auto.py:98 'The string \\'auto\\' resolves via :func:`fkappa_fire` against mCloud_input / sfe / nCore.'",
    "expected": "Any resolver reading other params introduces a cross-iteration ordering edge and must be covered by the ordering statement (or by consumed_by).",
    "failure_scenario": "The fkappa resolver runs before a future resolver that mutates nCore or sfe, and silently resolves auto against pre-resolution values; the SPECS ordering is the only thing preventing it and nothing documents that dependency.",
    "repro": "grep -n 'no inter-dependencies' trinity/_input/registry.py; grep -n 'mCloud_input / sfe / nCore' trinity/_input/fkappa_auto.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-18",
    "file": "trinity/_input/registry.py",
    "line": 253,
    "class": "other",
    "severity": "S4",
    "claim": "The SPS bundle is documented as owning 13 sps_col_* specs while the bundled file is described as having the canonical 7-column SB99 layout, with no explanation of the mismatch.",
    "evidence": "registry.py:253 '``sps_refmass`` and the 13 ``sps_col_*`` specs declare ``consumed_by=\\'sps_path\\'``' and '... in CSV form with the canonical 7-column SB99 layout (DEFAULT_SPS_COLUMN_MAP)'; param_spec.py:70 'the 13 ``sps_col_*`` specs are owned by ``sps_path``\\'s bundle resolver'",
    "expected": "State whether 13 declarable columns map onto a 7-column canonical set (e.g. 6 optional), so a user writing sps_col_* declarations knows how many are required.",
    "failure_scenario": "A user supplying a custom sps_path declares 7 columns and the resolver expects 13 (or vice versa), producing a column-map error whose cause the docs do not explain.",
    "repro": "grep -n '13 ``sps_col_\\*``\\|7-column' trinity/_input/registry.py trinity/_input/param_spec.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-19",
    "file": "trinity/_input/registry.py",
    "line": 1,
    "class": "state",
    "severity": "S2",
    "claim": "Input defaults are stored twice — as raw source strings in the registry and as lines in default.param — with the registry-side parser described in the future tense.",
    "evidence": "registry.py:1 '**Input specs** (``category`` starts with ``input_``): ``default`` is the *raw source string* exactly as it would appear in ``default.param`` ... Phase 10\\'s builder parses it via the same ``parse_value`` path ``read_param`` uses for file content.'; param_spec.py:1 '``tools/gen_default_param.py`` (Phase 3+ regenerates ``default.param`` from the registry)'; read_param.py:106 '# Step 1: Read default.param with INFO and UNIT metadata'",
    "expected": "One authoritative store for input defaults, or an enforced round-trip check that default.param equals the generator output.",
    "failure_scenario": "Someone edits default.param directly (the file read_param actually consumes) without updating the registry default string; the registry's run-const/metadata derivation and any regenerated default.param then disagree with what runs actually loaded.",
    "repro": "grep -n 'raw source string' trinity/_input/registry.py; grep -n 'gen_default_param' trinity/_input/param_spec.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-20",
    "file": "trinity/_input/dictionary.py",
    "line": 457,
    "class": "numerical",
    "severity": "S3",
    "claim": "simplify hardcodes a fallback nmin of 100 duplicating the simplify_npoints schema default, whose value is never stated anywhere in the slice.",
    "evidence": "dictionary.py:457 '``nmin`` defaults to the ``simplify_npoints`` parameter on the dict (loaded from default.param), or to 100 if absent. Pass an explicit ``nmin`` to override per call.'",
    "expected": "Either the fallback equals the documented schema default (and says so), or the absence of simplify_npoints is an error.",
    "failure_scenario": "default.param's simplify_npoints is changed to e.g. 50; any call path where the key is absent (a bare DescribedDict in a test, a debug-snapshot rehydration) silently uses 100 and produces snapshot arrays of a different resolution than production.",
    "repro": "grep -n 'or to 100 if absent' trinity/_input/dictionary.py; grep -n simplify_npoints trinity/_input/",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-21",
    "file": "trinity/_input/dictionary.py",
    "line": 457,
    "class": "other",
    "severity": "S3",
    "claim": "simplify's output-size contract is unfalsifiable as written: 'normally nmin', with two independent rules that can each force extra mandatory points.",
    "evidence": "dictionary.py:457 'Output size is normally ``nmin``. Endpoints and every high-prominence extremum are always retained — for very noisy curves with more than ``nmin`` such extrema, the output may exceed ``nmin`` rather than drop a real feature.' and '... an x-uniform coverage skeleton — one feature-pool point per equal-width x-chunk is promoted to mandatory'",
    "expected": "A stated upper bound (e.g. output <= nmin except when the mandatory set exceeds it, with the mandatory set size bounded), so a consumer sizing dictionary.jsonl can reason about worst case.",
    "failure_scenario": "A noisy bubble profile produces snapshot arrays much larger than simplify_npoints; dictionary.jsonl grows unboundedly with no test able to assert a size contract, because 'normally' admits any counterexample.",
    "repro": "grep -n 'Output size is normally' trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-22",
    "file": "trinity/_input/dictionary.py",
    "line": 457,
    "class": "other",
    "severity": "S4",
    "claim": "grad_inc — the curvature threshold that decides which bends survive downsampling — is named once and appears in no other prose in the slice: no default, no spec, no unit.",
    "evidence": "dictionary.py:457 'sharp bends (points where the Menger curvature exceeds ``grad_inc`` on rescaled [0, 1] axes — unit-free threshold)'",
    "expected": "If grad_inc is a tunable it should have a spec/default like simplify_npoints; if it is a constant of the standalone simplify module, say so.",
    "failure_scenario": "A maintainer tuning snapshot fidelity finds simplify_npoints in default.param but no way to reach grad_inc, and cannot tell whether it is configurable.",
    "repro": "grep -rn 'grad_inc' trinity/_input/",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-23",
    "file": "trinity/_input/read_param.py",
    "line": 412,
    "class": "regime",
    "severity": "S2",
    "claim": "The CIE cooling-table selection is documented only for ZCloud == 1 (integer preset {1,2,3}) and ZCloud == 0.15 (auto-pinned Sutherland-Dopita); behaviour for any other metallicity is unstated.",
    "evidence": "read_param.py:412 'Cooling directory - CIE (NOT a def_* sentinel: an integer-index preset keyed on ZCloud ...). Integer-index preset {1, 2, 3} (under ZCloud == 1) selects between the bundled CIE tables; ZCloud == 0.15 auto-pins to the Sutherland-Dopita file. All resolved paths live under lib/default/CIE/.'",
    "expected": "Documented behaviour (error, nearest-table, or extrapolation) for ZCloud outside {1, 0.15}, since ZCloud is a first-class input (param_spec.py:36).",
    "failure_scenario": "A run at ZCloud=0.5 either silently gets solar-metallicity cooling or fails with an error the docs never mention; the metallicity dependence of the cooling curve is a first-order effect on the bubble energy budget.",
    "repro": "grep -n 'Integer-index preset' -A4 trinity/_input/read_param.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-24",
    "file": "trinity/_input/read_param.py",
    "line": 350,
    "class": "coefficient",
    "severity": "S2",
    "claim": "caseB_alpha is pinned at its ~1e4 K value (2.59e-13 cm^3/s) and is documented as leaving Stroemgren balance and the HII pressure/force internally inconsistent if TShell_ion is moved, with only a load-time warning and no stated trigger threshold.",
    "evidence": "read_param.py:350 'caseB_alpha (the case-B recombination coefficient, default 2.59e-13 cm^3/s) is fixed at its ~1e4 K value and is NOT recomputed from TShell_ion. Since alpha_B(T) ~ T^-0.7, moving the ionised-shell temperature far from ~1e4 K leaves the Stroemgren balance (n_IF_Str) and P_HII/F_HII internally inconsistent unless caseB_alpha is adjusted to match. Warn once at load.'",
    "expected": "Either derive caseB_alpha from TShell_ion via the stated T^-0.7 scaling, or define 'far from ~1e4 K' as a concrete threshold that gates the warning.",
    "failure_scenario": "A user sets TShell_ion to 2e4 K; n_IF_Str, P_HII and F_HII are computed with a recombination coefficient ~1.6x too large for that temperature, biasing the ionised-shell force budget, and the warning either fires on every run or never fires depending on the undocumented threshold.",
    "repro": "grep -n 'caseB_alpha' -A5 trinity/_input/read_param.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-25",
    "file": "trinity/_input/read_param.py",
    "line": 474,
    "class": "state",
    "severity": "S3",
    "claim": "The prose records a shipped bug in which a default.param key was replaced by a runtime DescribedItem, so every run used include_PHII=True regardless of the user's .param.",
    "evidence": "read_param.py:477 'A key from default.param that has been replaced (not just mutated) with a fresh DescribedItem has lost the user\\'s value — the most recent offender was `include_PHII`, which meant every run integrated with include_PHII=True regardless of what the .param file said. Fail loudly so this never ships silently again.'",
    "expected": "Guard is described as in place (identity comparison, read_param.py:279). The finding is that any published results predating the guard carry the include_PHII=True override.",
    "failure_scenario": "Historical outputs/figures generated before the guard have include_PHII=True baked in even where the .param says otherwise; comparisons against them are invalid.",
    "repro": "grep -n 'include_PHII' trinity/_input/read_param.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-26",
    "file": "trinity/_input/dictionary.py",
    "line": 1199,
    "class": "deadcode",
    "severity": "S4",
    "claim": "COOLING_PHASE_KEYS contains nine commented-out members with no explanation of whether their exclusion from the post-implicit-phase reset is deliberate.",
    "evidence": "dictionary.py:1199 '# Bubble temperature/mass' followed by :1200 \"# 'bubble_Tavg',\", :1201 \"# 'bubble_T_r_Tb',\", :1202 \"# 'bubble_mass',\", :1203 \"# 'bubble_r_Tb',\"; dictionary.py:1217 '# Bubble profile arrays' followed by :1218-:1222 \"# 'bubble_v_arr',\" ... \"# 'bubble_n_arr',\"",
    "expected": "Either delete the commented entries or state why these keys must survive the reset (e.g. they are still read after the implicit phase).",
    "failure_scenario": "A maintainer 'restores' the commented entries to reduce memory and clears bubble arrays that later phases still read, or leaves genuinely stale bubble state alive across phases — the comments give no way to decide.",
    "repro": "grep -n \"# 'bubble_\" trinity/_input/dictionary.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-27",
    "file": "trinity/_input/dictionary.py",
    "line": 929,
    "class": "other",
    "severity": "S4",
    "claim": "Snapshot round-trip is documented as lossy for DescribedItem metadata and type-coercing for any list-valued key.",
    "evidence": "dictionary.py:929 'This reconstructs: - scalars directly into DescribedItem(value) - list values back into numpy arrays'; dictionary.py:958 '# Lists are converted back to numpy arrays'; contrast dictionary.py:99 which defines DescribedItem as value plus 'info', 'ori_units', 'exclude_from_snapshot'",
    "expected": "State explicitly that info/ori_units/exclude_from_snapshot are not restored, and that a genuinely list-typed parameter returns as ndarray.",
    "failure_scenario": "Code that resumes from a loaded snapshot reads params[k].ori_units for a unit-sensitive conversion and gets None; or a list-valued config key returns as ndarray and changes truthiness/equality semantics.",
    "repro": "grep -n 'Lists are converted back' trinity/_input/dictionary.py; compare with the DescribedItem docstring at dictionary.py:99",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-28",
    "file": "trinity/_input/registry.py",
    "line": 625,
    "class": "other",
    "severity": "S4",
    "claim": "Phase labels drift within and across the registry docs: Step 10 is Phase 9 in one place, 'Phase-8/9' in another, and the derived-init materialization is 'Phase 7/10' while Phase 10 is stated as unwired.",
    "evidence": "registry.py:1 'Step 10 calls ``materialize_runtime`` (Phase 9). Phase 10 will wire Step 6\\'s derived-init resolvers.'; registry.py:625 'Phase-8/9 entry point for ``read_param`` Step 10.'; registry.py:409 'the value is materialised by the derived-init resolver in Phase 7/10'",
    "expected": "One phase label per entry point.",
    "failure_scenario": "No runtime effect; a reader tracking migration status cannot tell which phases are shipped, which the project's own CLAUDE.md warns is a recurring doc-drift problem.",
    "repro": "grep -n 'Phase' trinity/_input/registry.py | head -40",
    "confidence": "high"
  },
  {
    "id": "S12a-B-29",
    "file": "trinity/_input/param_spec.py",
    "line": 60,
    "class": "deadcode",
    "severity": "S4",
    "claim": "A whole spec category is retained for a membership set that is documented as empty.",
    "evidence": "param_spec.py:60 '# ---- Parsed for back-compat only, never consumed ----' followed by :61 '# back-compat retired specs (currently none; kept for future use)'",
    "expected": "Remove the empty category or note when it is expected to be populated.",
    "failure_scenario": "None; hygiene. Flagged per the project rule that pre-existing dead code is reported, not deleted.",
    "repro": "grep -n 'currently none' trinity/_input/param_spec.py",
    "confidence": "high"
  },
  {
    "id": "S12a-B-30",
    "file": "trinity/_input/param_spec.py",
    "line": 109,
    "class": "other",
    "severity": "S3",
    "claim": "The invariant 'run_const intersect metadata_exclude is always empty' is asserted with no named enforcement, unlike the neighbouring invariants which each cite a test.",
    "evidence": "param_spec.py:109 '# run_const \\u2229 metadata_exclude is always empty (a key is written to metadata or blocked from it, never both).'; contrast param_spec.py:78 'test_every_sentinel_default_has_resolver_or_pointer', :128 'test_consumed_by_targets_exist', :137 'test_active_when_only_on_conditional_specs'",
    "expected": "A named guard, since registry.py:665/:674 say these two projections are the sole source of truth for RUN_CONST_KEYS and METADATA_EXCLUDE.",
    "failure_scenario": "A spec is flagged both run_const and metadata_exclude; depending on which list is consulted first the key is either written to metadata.json or blocked, and the resulting metadata.json contents become order-dependent.",
    "repro": "grep -n 'always empty' trinity/_input/param_spec.py; grep -n 'test_' trinity/_input/param_spec.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-31",
    "file": "trinity/_input/registry.py",
    "line": 118,
    "class": "citation",
    "severity": "S4",
    "claim": "f_A's definition is delegated to a document cited by bare filename with no path, and 'none' is claimed to parse to Python None though parse_value's documented None branch is not described as case-insensitive.",
    "evidence": "registry.py:118 'f_A is the interface source-term boost (SOURCE_TERM_DESIGN.md).' and 'cooling_boost_mode\\'s \\'none\\' default parses to Python None.'; read_param.py:73 'Precedence: None \\u2192 boolean \\u2192 number \\u2192 fraction \\u2192 string'; registry.py:1 gives input default examples as '\"None\"' (capitalised)",
    "expected": "A repo-relative path for SOURCE_TERM_DESIGN.md, and an explicit statement of whether parse_value's None branch is case-insensitive ('none' vs 'None').",
    "failure_scenario": "If parse_value only matches 'None', the default source string 'none' stays the string 'none', and any 'cooling_boost_mode is None' check silently reads it as mode-active — inverting the default.",
    "repro": "grep -rn 'SOURCE_TERM_DESIGN' trinity/ docs/; grep -n \"'none'\" trinity/_input/registry.py",
    "confidence": "medium"
  },
  {
    "id": "S12a-B-32",
    "file": "trinity/_input/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "The slice's package __init__ carries no comments or docstrings at all — the input layer's public surface is undocumented at module level.",
    "evidence": "The prose file lists sections for dictionary.py, registry.py, read_param.py, param_spec.py, errors.py and fkappa_auto.py; there is no '## trinity/_input/__init__.py' section, i.e. zero prose was extracted from it.",
    "expected": "A one-line module docstring naming what the package re-exports (read_param, DescribedDict, ParameterFileError are all referenced by siblings as import targets).",
    "failure_scenario": "None; hygiene. Also means Lens B cannot report on the package's export contract at all.",
    "repro": "Check whether trinity/_input/__init__.py contains any docstring.",
    "confidence": "medium"
  }
]
```
