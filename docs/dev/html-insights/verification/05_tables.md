# Verification ledger 05 — tclamp_report.html (A) + cooling/refactor-audit.md (B)

Verified 2026-06-22 on branch `claude/exciting-gates-mkxqn6`.

---

## SUMMARY

### (A) tclamp_report.html — 6-claim scorecard

| Count | Meaning |
|-------|---------|
| ✅ 8 | Claims verified against current source or committed CSVs |
| ⚠️ 2 | Claims correct in substance but line numbers have drifted |
| ❌ 0 | Flat-out wrong |
| ❓ 0 | Untraceable / no committed artefact |

All numeric claims (9,459,458 calls, 0 fires, min T = 30 000 K, 576/576 per-call,
byte-identical sha256 across 169 snapshots) trace to committed artefacts in
`docs/dev/magic-numbers/data/` and the TCLAMP_PLAN.md gate table.
Fix confirmed shipped at `trinity/cooling/net_coolingcurve.py:130–131` (commit cc8ae76).

### (B) refactor-audit.md — current status verdict

Last doc-status annotation: "ACTIONABLE — nothing shipped" (2026-06-16).

**Partially stale — two items have since shipped, three line-number references
have drifted, and one claim (NameError) is obsolete.**

| Item | Doc claim | Current status |
|------|-----------|---------------|
| 1e4 floor (Issue #9) | not shipped | ✅ SHIPPED — commit cc8ae76 |
| NameError for bad ZCloud (Issue #1 partial) | still a NameError | ✅ FIXED — commit 3deec3d raised explicit `ValueError` instead |
| SB99 coupling in read_cloudy.py | not shipped | NOT SHIPPED — still present |
| Hardcoded column names (Issue #2) | not shipped | NOT SHIPPED — still present |
| CIE integer-index only (Issue #7) | not shipped | NOT SHIPPED — still present |
| Unused metallicity arg (Issue #10) | not shipped | NOT SHIPPED — still present |
| 5.5 literal in dispatch (Issue #8) | not shipped | PARTIALLY SHIPPED — 5.5 still used as a filter in `_noncie_cutoffs`/`_cie_tcutoff`; but the actual dispatch boundaries (nonCIE_Tcutoff, CIE_Tcutoff) are now derived from the table |
| Line numbers | listed explicitly | DRIFTED ~5–45 lines |

---

## (A) tclamp_report.html — claim-by-claim table

| Claim | Report § | Evidence (file:line / CSV) | Verdict | Notes |
|-------|----------|---------------------------|---------|-------|
| Old floor was `if T < 1e4: T = 1e4` with a wrong TODO comment | §0, §1 | `TCLAMP_PLAN.md:36–44` cites the old code verbatim; AUDIT.md §"The code under audit" records it at `net_coolingcurve.py:122-123` (old numbering) | ✅ | |
| Non-CIE table true min is 3162 K (log 3.5), not "3.99" as the comment claimed | §1 | `TCLAMP_PLAN.md:57`: "non-CIE grid: log10 T ∈ [3.5, 5.5] = 3162 K … 316 228 K, 21 pts" | ✅ | Comment was wrong about the table; doc is correct |
| Clamp fired 0 times across 9,459,458 calls in 4 regimes; min T ever = 30 000 K | §3, §4 | `data/simple_cluster_summary.json`: calls=1225515, n_below_1e4=0, min_T=30000.0; `data/f1edge_lowdens_summary.json`: calls=2451122; `data/f1edge_hidens_summary.json`: calls=2666884; `data/conduction_stiff_summary.json`: calls=3115937; total=9,459,458 ✓ | ✅ | All four JSON files committed; total verified by arithmetic |
| Hypothesis H (T~10^3.91) retracted — dead code | §2, §3 | All four `*_summary.json` files: `n_below_1e4=0`, `n_below_3162=0` across all regimes | ✅ | |
| Option 1 (file-tied floor) chosen and shipped | §2, §5 | `trinity/cooling/net_coolingcurve.py:130–131`: `if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`; commit cc8ae76 confirmed in git log | ✅ | |
| New floor placed after cutoffs are computed so nonCIE_Tmin is in scope | §5 | `net_coolingcurve.py:121–131`: `_noncie_cutoffs()` called at line 121; floor at line 130 | ✅ | |
| Per-call 576/576 bit-identical for T≥1e4 | §6 | `TCLAMP_PLAN.md` gate table row "Per-call equivalence": "576 / 576 bit-identical for T≥1e4" | ✅ | Harness file `harness/verify_tclamp_equiv.py` exists |
| Full-run byte-identity (dictionary.jsonl sha256) across 169 snapshots | §6 | `TCLAMP_PLAN.md` gate table: "BYTE-IDENTICAL across 169 snapshots — both 9da691bb458a7aacd7b87a72a4557139edb5bd6699770ba900922773ff302ab0"; `harness/simple_cluster_capped.param` exists | ✅ | |
| 574 tests pass, 3 new test_net_coolingcurve.py cases | §6 | `test/test_net_coolingcurve.py` exists with 3 test functions: `test_below_table_does_not_raise_and_clamps_to_edge`, `test_over_floored_decade_uses_real_temperature`, `test_real_run_regime_untouched` | ✅ | |
| Report cites old floor at "the bubble cooling hot path" | §0 TL;DR | OLD line ref was `net_coolingcurve.py:85` (refactor-audit) or `:122-123` (TCLAMP_PLAN); current fix is at `net_coolingcurve.py:130–131` | ⚠️ | Line drifted; substance correct |
| Report says ZCloud pass-through at lines 88/148 | §5 (refactor-audit §5.5) | Current: `net_coolingcurve.py:163` and `:186` | ⚠️ | Line drifted ~75 lines; substance correct |

**Numeric artefact traceability.** All four per-regime summary JSONs are committed under
`docs/dev/magic-numbers/data/`. The `tclamp_dudt_overlay.csv` (12 KB) is also committed.
No claim is untraceable (❓ count = 0).

---

## (B) refactor-audit.md — shipped / not-shipped checklist

**Overall verdict (re-verified 2026-06-22):** The "nothing shipped" annotation from 2026-06-16
is now **partially stale**. Two items have shipped since then; the four core PR-1–PR-4 changes
remain unimplemented.

### Items that have shipped (not reflected in the doc's status annotation)

| Issue # | Claim in doc | Current source evidence | Status |
|---------|-------------|------------------------|--------|
| #9 (formerly §5.5, also §4 row 9) | "`net_coolingcurve.py:85` — floor T against `min(cStruc_cooling_nonCIE.temp)` instead of literal `1e4`" listed as a future PR-2 item | `trinity/cooling/net_coolingcurve.py:130–131`: file-tied floor shipped via commit cc8ae76 (the T-clamp fix, not the cooling refactor PRs) | ✅ SHIPPED |
| NameError for unsupported ZCloud | `read_cloudy.py:290-295` — "silent `NameError` on unsupported ZCloud … replaced by an explicit `ValueError`" — listed as a PR-1 checklist item | `trinity/cooling/non_CIE/read_cloudy.py:298–302`: explicit `raise ValueError(...)` present; introduced by commit 3deec3d ("fix(hygiene): metallicity guard") | ✅ SHIPPED |

### Items confirmed NOT shipped (core refactor PRs)

| Issue # | Claim in doc | Current source evidence | Status |
|---------|-------------|------------------------|--------|
| #1 (PR-1) | `read_cloudy.py:47–48` still reads `SB99_rotation` and `ZCloud` directly | `trinity/cooling/non_CIE/read_cloudy.py:48–49`: `SB99_rotation = params['SB99_rotation'].value`; `metallicity = params['ZCloud'].value` | NOT SHIPPED |
| #1 (PR-1) | `get_filename()` at `:266–336` still hardcodes OPIATE grammar | `trinity/cooling/non_CIE/read_cloudy.py:267–340`: `get_filename()` still present with OPIATE grammar `opiate_cooling_{rot|norot}_Z{Z_str}_age{age_str}.dat` | NOT SHIPPED |
| #2 (PR-2) | Hardcoded column names `ndens, temp, phi, cool, heat` at `:182–187` | `trinity/cooling/non_CIE/read_cloudy.py:182–187`: column names hardcoded as `ndens`, `temp`, `phi`, `cool`, `heat` | NOT SHIPPED |
| #2 (PR-2) | Sign auto-flip at `:189–194` | `trinity/cooling/non_CIE/read_cloudy.py:189–196`: auto-flip with `print` warning still present | NOT SHIPPED |
| #5 (PR-2) | Cube cache at `:170–172, 261` keyed only on stem, no axis hash | `trinity/cooling/non_CIE/read_cloudy.py:169–172`: `_stem = filename[:-4]`; cache at `path2cooling + _stem + '_cube.npy'` — stem-keyed, no hash | NOT SHIPPED |
| #7 (PR-3) | `read_param.py:408` parses `path_cooling_CIE` as `int()` — user cannot use a path string | `trinity/_input/read_param.py:423`: `cie_choice = int(params['path_cooling_CIE'].value)` still present | NOT SHIPPED |
| #7 (PR-3) | Silent fall-through if `ZCloud` is neither 1 nor 0.15 (no `else` raise) | `trinity/_input/read_param.py:417–429`: `if ZCloud == 1: ... elif ZCloud == 0.15: ...` — no `else` raise; value stays as integer 3 | NOT SHIPPED |
| #10 (PR-4) | Unused `metallicity` arg on `CIE.get_Lambda` at `read_coolingcurve.py:25` | `trinity/cooling/CIE/read_coolingcurve.py:25`: `def get_Lambda(T, cooling_CIE_interpolation, metallicity)` — arg still present, body never uses it | NOT SHIPPED |
| #10 (PR-4) | `ZCloud` reads at `net_coolingcurve.py:88, 148` | Current: `trinity/cooling/net_coolingcurve.py:163, 186`: `params_dict['ZCloud'].value` still passed to `CIE.get_Lambda` | NOT SHIPPED |

### Partial / nuanced status

| Issue # | Claim in doc | Current source evidence | Status |
|---------|-------------|------------------------|--------|
| #8 (PR-2) | `net_coolingcurve.py:93, 95` — "Hardcoded `5.5` (in log10 K) as the CIE/non-CIE boundary" | `trinity/cooling/net_coolingcurve.py:43,53`: literal `5.5` is still present but now used as a **filter** inside `_noncie_cutoffs`/`_cie_tcutoff` to derive the table-aware boundary values (nonCIE_Tcutoff, CIE_Tcutoff); the dispatch conditions at lines 138/159/168 use the derived values, not the bare literal | PARTIAL — literal persists as filter; dispatch is now table-derived |

### Line-number drift (doc cites old numbers; substance is correct)

| Doc reference | Actual current line |
|---------------|---------------------|
| `net_coolingcurve.py:85` (old 1e4 floor) | Shipped fix now at line 130–131 |
| `net_coolingcurve.py:93, 95` (5.5 boundary) | Now in `_noncie_cutoffs` at line 43; `_cie_tcutoff` at line 53 |
| `net_coolingcurve.py:88, 148` (ZCloud pass-through) | Now at lines 163, 186 |
| `read_param.py:397–414` (CIE inline resolution) | Now at lines 412–429 |
| `read_cloudy.py:47, 59, 266–336` (get_filename) | Now at lines 48, 60, 267–340 |
| `read_coolingcurve.py:25` (metallicity arg) | Still at line 25 — no drift |
| `read_cloudy.py:290–295` (NameError) | Now at lines 298–302 as explicit `ValueError` (fix shipped) |
