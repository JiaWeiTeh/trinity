# SB99 → generic SPS refactor: implementation plan

Companion to `analysis/sb99-refactor-audit.md`. The audit is *what is*; this is
*what to do, in what order, and how to prove nothing changed*.

The headline risk for this refactor is **silent numerical drift**. The SB99
loader feeds 10 cubic-spline interpolators that drive every physics phase. A
single mis-ordered multiplication in unit conversion could shift simulation
trajectories by ULPs at first and meaningfully later (cubic-spline
interpolation amplifies ULP errors near knots). The whole plan is structured
around making that drift impossible to ship undetected.

## 0. TL;DR

- **Five PRs**, each independently mergeable and revertable. PRs 1–3 add
  capability behind back-compat. PR 4 renames. PR 5 deletes deprecated
  surface.
- **Two equivalence guarantees** carry across every PR:
  1. **Bitwise** equivalence of the 11 raw loader arrays when the legacy
     fallback resolves to the legacy file. `np.array_equal`, no tolerance.
  2. **Tight-tolerance** (`rtol=1e-12, atol=0`) E2E snapshot-tree equivalence
     against a pre-refactor golden. Anything looser is a regression.
- **Golden capture happens once, on `main`, BEFORE PR-1**. If you skip that
  step the rest of the plan is theatre.
- **Cooling-table coupling in `cooling/non_CIE/read_cloudy.py` is out of
  scope** for this refactor. Tracked separately.

## 1. Goals and non-goals

### Goals

1. Replace the hardcoded SB99 filename grammar with a single `sps_path`
   parameter so users can drop in arbitrary SPS CSVs.
2. Decouple `f_mass`'s reference mass from `SB99_mass` (add `sps_refmass`).
3. Allow header-driven column maps so CSV column order / units / log-vs-linear
   are no longer hardcoded.
4. Allow optional explicit SN columns (`Lmech_SN`, `pdot_SN`, `Mdot_SN`,
   `v_SN`) for SPS codes that provide them directly.
5. Make every step **byte-equivalent** to the current SB99 path when run with
   legacy parameters.

### Non-goals

- Not retiring SB99 as a data source. SB99 stays the default; this just
  removes the hardcoding around it.
- Not refactoring `update_feedback.py`'s numerical logic. The central
  difference, `v_mech_total` formula, and dataclass shape are unchanged.
- Not touching the cooling-table coupling (`read_cloudy.py`). That is a
  *different* SB99 dependency with its own indirection problem. Flagged for a
  follow-up; explicit warning to be emitted if `sps_path` is used with
  rotation/Z outside the legacy {rot/norot, 1/0.15} set.
- Not adding a CI test suite. The equivalence harness is a runnable script
  under `analysis/`, consistent with the repo's "tests aren't kept under
  version control" convention.

## 2. Invariants (what MUST stay byte-identical under the legacy code path)

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
| The 12 scalar params written by `updateDict(params, feedback)` (`read_param.py:474-485`) | `np.array_equal` |
| Every JSONL snapshot value in a full trinity run | `np.allclose(rtol=1e-12, atol=0)` per field |

Why `rtol=1e-12` and not bitwise for E2E? Because there's I/O serialization
(`jsonl`) in between. JSONL writes floats via `repr()` or similar; round-trip
through string can introduce ULP-level noise. `1e-12` is well below any
physically meaningful tolerance and well above JSONL round-trip noise.

If a PR breaks any of these invariants, **the PR is broken**, not the tests.

## 3. Migration / deprecation strategy

Two new params, each with a sentinel default that falls through to legacy
behavior:

- `sps_path` — full path to an SPS CSV. Default `def_path`.
  - When `def_path` and `SB99_mass / SB99_rotation / SB99_BHCUT / ZCloud` are
    set: construct path from legacy grammar (existing `get_filename()` logic
    relocated to `read_param.py`). Emit one `DeprecationWarning` at startup
    listing the migration path.
  - When set to anything else: use that path verbatim. Skip legacy grammar.
- `sps_refmass` — reference cluster mass for `f_mass = mCluster /
  sps_refmass`. Default `def_value`.
  - When `def_value`: copy `params['SB99_mass'].value`.
  - Else: use directly.

Legacy params (`SB99_mass`, `SB99_rotation`, `SB99_BHCUT`) keep working
unchanged through PR-1 through PR-4. PR-5 deletes them; users who have not
migrated by then will see a hard error with a clear message pointing at
`sps_path`.

`ZCloud` is **not** deprecated. It controls dust opacity and other
metallicity-keyed physics elsewhere (`read_param.py:280, 321, 363, 372`); it
still needs to be set even after the SPS file is decoupled.

## 4. PR sequence

Each PR is independently mergeable. Order matters; do not reorder without
re-running the equivalence battery between every reordering.

### PR-1 — `sps_path` and `sps_refmass` with legacy fallback

**Scope.** Add the two new params and the resolution logic. The loader,
phases, interpolators, and dataclass are otherwise untouched.

**Files touched.**

- `src/_input/default.param` — add `sps_path  def_path` and
  `sps_refmass  def_value`. Update `# INFO` lines on the three legacy SB99_*
  params to say "deprecated, use sps_path".
- `src/_input/read_param.py` — new resolution block. If `sps_path == 'def_path'`,
  construct via the legacy grammar (relocate `get_filename` logic here). Emit
  `DeprecationWarning`. Same shape for `sps_refmass`.
- `src/sb99/read_SB99.py` — change `filepath = path2sps +
  get_filename(params)` (line 125) to `filepath = params['sps_path'].value`.
  Delete `get_filename()` from this module (now lives in `read_param.py`).
  Drop the `path_sps / SB99_rotation / ZCloud / SB99_BHCUT` requirements
  from the validation block (`read_SB99.py:104-117`).
- `src/main.py` — change line 142 to `f_mass = params['mCluster'] /
  params['sps_refmass']`.

**Code-level checklist.**

- [ ] `sps_path`/`sps_refmass` declared in `default.param` with sentinel
      defaults.
- [ ] Resolution block in `read_param.py` constructs identical path string to
      old `path_sps + get_filename(params)` for every legacy combination.
- [ ] `DeprecationWarning` fires exactly once per run (use `warnings.warn` +
      module-level guard, not per-call).
- [ ] `read_SB99.read_SB99` reads only `params['sps_path']` and
      `params['FB_*']` after this PR. No `SB99_rotation/ZCloud/BHCUT` reads
      remain in the loader.
- [ ] `main.py:142` uses `sps_refmass`.
- [ ] No phase code changed.
- [ ] `bubble_luminosity_modified.py:33` dead import left alone (PR-5
      cleanup).

**Tests required to merge.** See §5.2.1 for the full battery. Headline:

1. Path-resolution matrix: 2 rotations × 2 Zs × 2 BHCUTs × 4 masses = 32
   legacy combinations. All produce string-identical paths.
2. Loader byte-equivalence against pickled pre-refactor arrays.
3. E2E snapshot-tree equivalence for the canonical config under legacy
   params.
4. E2E snapshot-tree equivalence for the canonical config with `sps_path`
   set explicitly to the path the legacy fallback would have resolved to.
   Must equal (3).

### PR-2 — Header-driven column mapping

**Scope.** Add the ability to read a CSV with a header row that names columns
and declares units. SB99's 7-column headerless positional format remains the
default fallback when no header is detected.

**Files touched.**

- `src/sb99/read_SB99.py` — refactor `read_SB99()` into:
  - `_detect_header(path) -> bool`
  - `_load_with_header(path) -> dict[str, np.ndarray]` (named columns, raw)
  - `_load_positional_sb99(path) -> dict[str, np.ndarray]` (legacy preset)
  - `_apply_units(raw_cols, unit_map) -> dict[str, np.ndarray]` (log →
    linear, cgs → AU)
  - Existing scaling logic (FB_* corrections) operates on the named dict.
- (Optionally rename module to `read_sps.py` with a back-compat shim. Defer
  to PR-4 if it adds churn here.)

**Column-name vocabulary (canonical).**

The header row must use these names; aliases accepted for back-compat:

| Canonical | Aliases | Units (linear) | Required |
|-----------|---------|---------------|----------|
| `t` | `time`, `age` | yr | yes |
| `Qi` | `log_Qi`, `ionizing_photon_rate` | 1/s | yes |
| `fi` | `ionizing_fraction` | dimensionless | yes |
| `Lbol` | `log_Lbol`, `L_bolometric` | erg/s | yes |
| `Lmech_total` | `log_Lmech`, `L_mech` | erg/s | yes |
| `pdot_W` | `log_pdot_wind`, `momentum_rate_wind` | g·cm/s² | yes |
| `Lmech_W` | `log_Lmech_wind`, `L_mech_wind` | erg/s | yes |
| `Lmech_SN` | `L_mech_SN` | erg/s | no (derived if absent) |
| `pdot_SN` | `momentum_rate_SN` | g·cm/s² | no (derived) |
| `v_SN` | `SN_velocity` | cm/s | no (uses `FB_vSN` if absent) |

Log-space columns indicated by `log_` prefix in alias or `log_units=true` in a
sidecar.

**Code-level checklist.**

- [ ] Header detection: first non-comment line; if any field non-numeric,
      treat as header.
- [ ] Positional fallback unchanged when no header (exact same code path as
      PR-1's loader, just refactored into a private function).
- [ ] Per-column unit dispatch (`log` or `linear`, `cgs` or `AU`).
- [ ] Missing optional columns trigger the existing derivation:
      `Lmech_SN = Lmech_total - Lmech_W`, `pdot_SN` etc.
- [ ] No change to FB_* scaling logic.

**Tests required to merge.**

1. Positional path produces byte-identical arrays to PR-1's loader (no
   header → same code).
2. Header-equipped clone of SB99 file (header-declared positions identical
   to legacy) produces byte-identical arrays.
3. Linear-units clone (de-logged via `Lbol_lin = 10**Lbol_log` etc.) with
   `log_units=false` header produces byte-identical arrays.
4. Subset clone with only 7 SB99 columns (no optional SN cols) routes
   through derivation; arrays match (1).
5. E2E snapshot equivalence — same as PR-1 tests but rerun on this branch.

### PR-3 — Optional explicit SN columns

**Scope.** When a CSV provides `Lmech_SN`, `pdot_SN`, or `v_SN` directly,
use them instead of deriving. Falls back to the current derivation when
absent. Affects only the SN scaling block (`read_SB99.py:231-245`).

**Files touched.** `src/sb99/read_SB99.py` only.

**Code-level checklist.**

- [ ] `Lmech_SN` present → skip subtraction path.
- [ ] `pdot_SN` present → skip implicit derivation from `Mdot_SN` and
      `v_SN`.
- [ ] `v_SN` present → use the column; else use `FB_vSN` constant.
- [ ] Document interaction with `FB_mColdSNFrac` / `FB_thermCoeffSN` (they
      still apply on top of explicit columns).

**Tests required to merge.**

1. SB99 file (no SN cols) → arrays identical to PR-2.
2. Synthetic CSV with `Lmech_SN = Lmech_total - Lmech_W` (i.e. SN columns
   that *match* the derivation) → arrays identical to derivation path.
3. Synthetic CSV with `Lmech_SN` set to half of `(Lmech_total - Lmech_W)`
   → loader uses the file value, not derivation (sanity test).
4. E2E equivalence — same battery as PR-2.

### PR-4 — Rename `SB99f` → `sps_f`, `SB99_data` → `sps_data`

**Scope.** Mechanical rename. Every consumer touched. Module rename
`read_SB99.py` → `read_sps.py`. Aliased back-compat in `read_param.py` so
external code reading `params['SB99f']` still works for one release.

**Files touched.**

- `src/_input/read_param.py` — rename runtime containers; keep `SB99f`/
  `SB99_data` as alias entries pointing at the same `DescribedItem`.
- `src/sb99/read_SB99.py` → rename to `src/sps/read_sps.py` (or keep
  `src/sb99/` and rename only files — choose based on whether SB99 is one of
  many SPS sources or *the* SPS source in this codebase).
- `src/sb99/update_feedback.py` — `SB99Feedback` → `SPSFeedback`,
  `get_currentSB99feedback` → `get_current_sps_feedback`, all `SB99f` reads
  → `sps_f`.
- All phase files in §6.3–§6.8 of the audit — update imports.
- `src/main.py:142-152` — rename container references.

**Code-level checklist.**

- [ ] Single rename PR, no logic changes. `git diff --stat` should be heavy
      on `phase*` files but contain no algorithmic changes.
- [ ] Back-compat alias in `read_param.py`: `params['SB99f'] =
      params['sps_f']` (same underlying object) so external user scripts
      continue to work.
- [ ] `phase0_init/get_InitPhaseParam.py:88, 111-112` updated.
- [ ] `bubble_luminosity_modified.py:33` import updated (still dead; PR-5
      removes).

**Tests required to merge.**

1. Full equivalence battery from PR-3, rerun unchanged.
2. New test: `params['SB99f'] is params['sps_f']` (alias works).
3. New test: importing `get_currentSB99feedback` from `update_feedback`
   raises clear error pointing at the new name.

### PR-5 — Cleanup

**Scope.** After at least one release cycle on PR-4. Removes deprecated
surface.

**Files touched.**

- `src/_input/default.param` — remove `SB99_mass`, `SB99_rotation`,
  `SB99_BHCUT` (or convert their `# INFO` line to a hard error message).
- `src/_input/read_param.py` — remove legacy fallback in `sps_path`
  resolution. Remove `SB99f`/`SB99_data` aliases. Remove `DeprecationWarning`.
- `src/sb99/update_feedback.py` — remove `get_currentSB99feedback` alias if
  still present.
- `src/bubble_structure/bubble_luminosity_modified.py:33` — delete dead
  import.

**Tests required to merge.**

1. Configs that don't set `sps_path` now raise a clear error.
2. Configs that set `sps_path` continue to work; full E2E equivalence
   against PR-4 golden.
3. `bubble_luminosity_modified.py` imports list contains no SB99 references.

### Out of scope: cooling-table coupling

`src/cooling/non_CIE/read_cloudy.py:47, 263-331` constructs filenames from
`SB99_rotation` and `ZCloud`. Even after the feedback CSV is decoupled, the
cooling cubes themselves were generated from SB99-keyed SEDs, so swapping in
a different SPS does not magically generalize cooling.

**Mitigation in this refactor.** When `sps_path` is set to a non-default
value, emit a `UserWarning`: "Cooling tables are still keyed by SB99
rotation+Z; results valid only if the SPS source is SB99-compatible at the
declared rotation/Z." Tracked as follow-up: add `path_cooling_nonCIE` knob
analogous to `sps_path`.

## 5. Test strategy (the substance)

This section is what makes the refactor safe. Everything else is plumbing.

### 5.1 Golden capture protocol (DO THIS FIRST)

**One-time setup on `main`, before any refactor branch is cut:**

```bash
git checkout main
# Pick anchor configs covering the relevant physics space:
#  - mockFullrun config (canonical small example)
#  - one paper1 sweep config (production-like)
#  - one rosette config (alternate density profile)
mkdir -p analysis/sb99-refactor-golden
for cfg in param/mockFullrun.param param/trinity_paper1_sweep.param param/rosette_sweep.param; do
    python src/main.py --param "$cfg" --output "analysis/sb99-refactor-golden/$(basename $cfg .param)/"
done
# Pickle the loader's 11-array output and the 10 interpolators evaluated at
# a dense time grid. The harness script below does this.
python analysis/sb99_refactor_equivalence.py --capture-golden
# Stash the golden tree somewhere outside the repo (it's gitignored under
# outputs/, and analysis/ is small).
```

The golden snapshot tree must be **frozen for the entire refactor**. Do not
regenerate it on later commits — that would silently mask drift.

If `main` changes between PRs (e.g. an unrelated merge), the golden does not
need to be regenerated for *this* refactor — what matters is that each
refactor PR matches the golden that existed at the moment the refactor
branched from main. Document the golden's commit SHA in the harness manifest.

### 5.2 Per-PR test battery

A single Python script `analysis/sb99_refactor_equivalence.py` runs all
tests; each PR's CI gate is "run the script with `--pr N`, must exit 0".

The script is not VCS'd long-term per CLAUDE.md ("tests in this repo live
elsewhere or aren't kept under version control"). It lives in `analysis/` as
a working artifact, removed once the refactor lands.

#### 5.2.1 PR-1 tests

```python
# analysis/sb99_refactor_equivalence.py (sketch — full version produced in PR-1)

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
    golden_interp = pickle.load(open('.../interp_samples.pkl', 'rb'))  # {key: (ts, ys)}
    arrays = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    SB99f = read_SB99.get_interpolation(arrays)
    for key, (ts, ys_gold) in golden_interp.items():
        ys_cur = SB99f[key](ts)
        assert np.array_equal(ys_gold, ys_cur), f"interpolator {key} drift"

def test_dataclass_byte_equivalence():
    """get_currentSB99feedback agrees at sampled times."""
    golden = pickle.load(open('.../feedback_samples.pkl', 'rb'))  # list of (t, dataclass)
    params = mock_params_legacy_loaded()
    for t, fb_gold in golden:
        fb_cur = get_currentSB99feedback(t, params)
        for field in fields(fb_gold):
            g, c = getattr(fb_gold, field.name), getattr(fb_cur, field.name)
            assert np.array_equal(g, c), f"feedback.{field.name} drift at t={t}"

def test_e2e_snapshot_equivalence():
    """Full trinity run; snapshot JSONL trees match within rtol=1e-12."""
    for cfg in CANONICAL_CONFIGS:
        new_out = run_trinity(cfg, out_dir=tmpdir)
        gold_out = f"analysis/sb99-refactor-golden/{cfg.stem}/"
        diff_snapshot_trees(gold_out, new_out, rtol=1e-12, atol=0)

def test_e2e_with_explicit_sps_path():
    """Setting sps_path explicitly to the legacy file yields the same run."""
    for cfg in CANONICAL_CONFIGS:
        explicit_cfg = inject_sps_path(cfg, legacy_grammar_path(cfg))
        out = run_trinity(explicit_cfg, out_dir=tmpdir)
        gold = f"analysis/sb99-refactor-golden/{cfg.stem}/"
        diff_snapshot_trees(gold, out, rtol=1e-12, atol=0)

def test_deprecation_warning_fires_once():
    """Legacy fallback emits exactly one DeprecationWarning per run."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_with_legacy_params()
        load_with_legacy_params()  # second call
    depr = [x for x in w if issubclass(x.category, DeprecationWarning)
                          and 'sps_path' in str(x.message)]
    assert len(depr) == 1  # not 0, not 2
```

#### 5.2.2 PR-2 tests

```python
def test_positional_path_byte_identical_to_pr1():
    """No header → loader produces identical arrays to PR-1."""
    arrays = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    # Compare to golden (same byte-equivalence as PR-1 test_loader_byte_equivalence)

def test_header_path_byte_identical_to_positional():
    """Same data with header row produces identical arrays."""
    write_sb99_with_canonical_header(src=LEGACY_FILE, dst=tmpfile)
    arrays_hdr = read_SB99.read_SB99(f_mass=1.0,
                                     params=mock_params_with_path(tmpfile))
    arrays_pos = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    for hdr, pos in zip(arrays_hdr, arrays_pos):
        assert np.array_equal(hdr, pos)

def test_linear_units_byte_identical_to_log_units():
    """Pre-exponentiated columns with log_units=false match log-form load."""
    write_sb99_with_linear_units(src=LEGACY_FILE, dst=tmpfile)
    arrays_lin = read_SB99.read_SB99(f_mass=1.0,
                                     params=mock_params_with_path(tmpfile))
    arrays_log = read_SB99.read_SB99(f_mass=1.0, params=mock_params_legacy())
    for lin, log in zip(arrays_lin, arrays_log):
        # Linear→log→linear roundtrip introduces a ULP or two via 10**log
        # vs the original. Tolerate 4 ULP here, not bitwise.
        assert np.allclose(lin, log, rtol=4e-15, atol=0)

def test_missing_optional_cols_fall_back():
    """7-column header-equipped file (no SN cols) routes through derivation."""
    # Same as positional case; arrays must match.

def test_e2e_equivalence_after_pr2():
    # Same as PR-1 §5.2.1 test_e2e_snapshot_equivalence.
```

#### 5.2.3 PR-3 tests

```python
def test_no_sn_cols_derivation_unchanged():
    """SB99 file without SN cols → arrays match PR-2."""

def test_explicit_sn_cols_matching_derivation():
    """CSV with Lmech_SN that exactly equals (Lmech_total - Lmech_W).
    Loader-using-cols path and derivation path agree bitwise."""

def test_explicit_sn_cols_overriding_derivation():
    """CSV with Lmech_SN = 0.5 * (Lmech_total - Lmech_W). Loader uses the
    file value (not derivation). Verify resulting Mdot_SN, velocity_SN
    differ from derivation in the expected way."""

def test_e2e_equivalence_after_pr3():
    # Same battery.
```

#### 5.2.4 PR-4 tests

```python
def test_all_phases_run_after_rename():
    """E2E equivalence battery."""

def test_back_compat_alias_works():
    """params['SB99f'] is params['sps_f']."""

def test_old_import_path_raises_clearly():
    """from src.sb99.update_feedback import get_currentSB99feedback raises
    ImportError with a message pointing at get_current_sps_feedback."""
```

#### 5.2.5 PR-5 tests

```python
def test_unset_sps_path_raises_clear_error():
    """A config that omits sps_path now hard-errors with migration guidance."""

def test_explicit_sps_path_still_works():
    """Full E2E equivalence against PR-4 golden."""
```

### 5.3 Equivalence tolerance policy

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

### 5.4 Harness layout

```
analysis/
├── sb99-refactor-audit.md
├── sb99-refactor-implementation-plan.md         # this file
├── sb99_refactor_equivalence.py                 # harness script (not VCS'd long term)
└── sb99-refactor-golden/                        # gitignored; pickled goldens + JSONL trees
    ├── MANIFEST.json                            # commit SHA of main at capture time
    ├── loader_arrays.pkl
    ├── interp_samples.pkl
    ├── feedback_samples.pkl
    ├── mockFullrun/
    │   ├── 1_begin.jsonl
    │   ├── 2_energy.jsonl
    │   ├── 3_implicit.jsonl
    │   ├── 4_transition.jsonl
    │   ├── 5_momentum.jsonl
    │   ├── 6_final.jsonl
    │   ├── dictionary.jsonl
    │   ├── metadata.json
    │   └── *.param, *.txt, etc.
    ├── trinity_paper1_sweep/
    └── rosette_sweep/
```

Add `analysis/sb99-refactor-golden/` to `.gitignore` if not already covered by
`outputs/*` exception rules. The pickles and JSONL trees are too big and
ephemeral to commit.

## 6. Risk register

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| Float ULP drift introduced by reordering ops in loader | Medium | High | PR-1 explicitly does NOT touch unit-conversion math. Byte-equivalence test catches it. |
| Path-resolution drift due to subtle formatting in `get_filename` (e.g. mantissa formatter at `read_SB99.py:333`) | Medium | High | Path-resolution matrix test covers all legal combos. |
| Header detection false positive on a numeric-looking SB99 file | Low | Medium | PR-2 detection rule is "first non-comment line has any non-numeric token". Test with synthetic edge case (line starting with `1.0 2.0 ...`). |
| `t=0` prepend hack (loader lines 260-273) double-applied if generic CSV already has t=0 | Medium | Medium | PR-2 loader detects `t[0] == 0` and skips prepend; add explicit test. |
| Cooling tables silently mismatch generic SPS | High | Medium | Out-of-scope warning emitted; tracked separately. |
| `scipy.interp1d` deprecation in future scipy | Low | Low | Pin scipy in repo deps. Out of scope for this refactor. |
| Users have configs that set `SB99_mass` to something other than the canonical 1e6 | Medium | Low | `sps_refmass` defaults to `SB99_mass`, so back-compat preserved. Test with mass=2.5e6 scenario. |
| PR-4 rename misses a consumer (silently leaves stale `SB99f` reference) | Low | High | `grep -r 'SB99' src/` after PR-4 must show only the back-compat alias declarations and docstrings/comments. |

## 7. Rollout sequence

1. **Capture golden** on `main` (§5.1). Verify the harness can compare a
   freshly captured tree to itself with zero drift (smoke test). Commit
   harness if-and-only-if user agrees; otherwise keep it as a working file.
2. **Branch `feature/sps-path-fallback`** for PR-1. Run battery; merge when
   green.
3. **Branch `feature/sps-column-mapping`** for PR-2. Re-run full battery.
4. **Branch `feature/sps-explicit-sn-cols`** for PR-3. Re-run full battery.
5. **Branch `feature/sps-rename`** for PR-4. Re-run full battery + new
   alias tests.
6. **One release cycle** with deprecation warnings in production.
7. **Branch `fix/drop-sb99-deprecated-params`** for PR-5. Re-run battery
   against the *post-PR-4 golden* (configs now use `sps_path` natively).

All branch names use the repo's `feature/` or `fix/` prefix per CLAUDE.md.
The current audit branch (`claude/sb99-default-parameter-ttQIN`) violates
that rule and exists only because the original session was provisioned
with the wrong prefix — confirm with user before pushing PR-1 from a
properly-named branch.

## 8. Open questions for the user

Before starting PR-1, please confirm:

1. **Module location.** Keep loader at `src/sb99/read_SB99.py` (rename only
   the symbols), or move to `src/sps/read_sps.py`? Affects PR-4 churn.
2. **Deprecation window.** How long should `SB99_mass / SB99_rotation /
   SB99_BHCUT` continue working before PR-5 deletes them? One release? Two?
3. **Golden anchor configs.** Are `mockFullrun`, `trinity_paper1_sweep`,
   `rosette_sweep` the right three for the E2E battery, or should the set
   include something else (e.g. a high-mass cluster, a low-Z configuration)?
4. **Harness commit policy.** Commit `analysis/sb99_refactor_equivalence.py`
   to the audit branch for the duration of the refactor, or keep it as an
   untracked working file?
5. **Cooling coupling timing.** Out of scope here, but: do you want a
   parallel issue/PR opened to address `read_cloudy.py`'s SB99 keying, or
   leave that until the SPS refactor lands?
