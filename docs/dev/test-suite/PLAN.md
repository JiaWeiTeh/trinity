# Test-suite remediation — PLAN

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

**Status (2026-07-10):** 🟢 complete — P0 through P5-T6 are complete; no open phase remains in
this plan.

## Provenance & scope

Audit performed 2026-07-06 at commit `70f07532` (clean tree) by four parallel subagent passes:
solver/physics slice (20 files), input/config slice (12 files), output/cloudy/misc slice (16 files),
and a coverage-gap sweep of `trinity/` vs `test/`. Every verdict below was verified against the
source **at that commit** — imports resolved, line references read, suspicious files `git log
--follow`-ed, several files executed. Full per-file verdicts: Appendix A. Coverage matrix:
Appendix B.

**This doc is the frozen output of that audit.** The executor's job is to apply each phase's
CHANGE and clear its GATE — not to re-derive the reasoning. If HEAD has moved past `70f07532`,
re-verify only the specific line references a phase touches (the ⚠️ banner), not the whole audit.

2026-07-09 drift check at `9410212d`: P1's production `solve_R1` non-finite guard is already
present in `trinity/bubble_structure/get_bubbleParams.py`; the remaining P1 work was the duplicate
assertion cleanup in `test/test_energy_collapse_guard.py`. Local `.venv` gate used Python 3.9.6,
numpy 1.26.4, scipy 1.13.1, astropy 6.0.1, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2.
The pre-existing `test_bubble_solver_failures.py::test_rhs_collapse_returns_ok_false` native
LSODA/RHS-exception abort was fixed by catching `BubbleSolverError` inside the Python RHS callbacks
before control returns to LSODA. Targeted P1 tests and full default pytest now pass in the local
`.venv`.

2026-07-09 P2 execution: the four redundant test files were deleted/merged. Count accounting:
751 collected before P2 - 15 removed nodes + 6 relocated/salvaged/merged nodes = 742 collected
after P2, with the same 3 deselected stress tests. Targeted recipient tests and full default
pytest pass in the local `.venv`.

2026-07-10 P3 execution: `test_energy_collapse_snapshot.py::test_energy_collapse_emits_no_negative_Pb`
and `test_simplify.py::TestTiming` are now `stress` tests. Default collection still has 742 total
nodes, with selected tests dropping from 739 to 733 and deselected stress nodes increasing from
3 to 9. Full default pytest wall time dropped from 118.53s at P2 to 77.64s.

2026-07-10 P4 execution: added `densBE_sigma` to the active-when skip coverage in
`test_active_when.py` and `test_materialize_runtime.py`, removed the unused `warnings` import and
dead `_make_params` no-op from `test_metadata.py`, and refreshed the four stale test docstrings
called out below. This was test/prose only; collection stayed at 742 total / 733 selected /
9 deselected.

2026-07-10 P5-T1 drift check: the cited JSONL parsing pattern still lives at
`test/test_betadelta_hybr_stress.py:88-100`, and the cited snapshot-key units still live at
`trinity/_input/registry.py:391-395`. The smoke-test header was stale as predicted and was
updated. Golden capture used `.venv` (Python 3.9.6, numpy 1.26.4, scipy 1.13.1, astropy 6.0.1,
pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2): final row `R2=0.2857315185200479`,
`v2=44.73918438203256`, `Eb=778236.3470566473`. The required mutation check intentionally
changed the `R2` golden to `2.857315185200479` and `pytest test/test_run_smoke.py -q` failed on
the new approximate comparison before the correct golden was restored. Targeted smoke, collection,
docs convention, and full default pytest gates all pass after the restore.

2026-07-10 P5-T2 drift/capture check: the cited stress `_PARAM`, `_GOLDEN`, and `_run` anchors
still live at `test/test_betadelta_hybr_stress.py:46-63` and `:74-85`; `current_phase` is still
registered at `trinity/_input/registry.py:384`; and `main.py:244,278` still stamps `energy` then
`implicit`. A canonical-env shortened `stop_t=0.004` run produced 101 rows, de-duplicated phases
`['energy', 'implicit']`, and 3 implicit rows. The first two `(cool_beta, cool_delta)` pairs
(`0.7592595923944201, -0.03538910029989962` at `t=0.0034103388192113612`; same pair at
`t=0.003807502936573502`) match the stress goldens within `abs=2e-3`, so no dedicated shortened-run
goldens were needed. The required mutation check changed the first beta golden to `7.59260`, and
`pytest test/test_phase_boundary.py -q` failed on the new approximate comparison before the correct
golden was restored.
Targeted boundary, collection, docs convention, and full default pytest gates all pass after the
restore.

2026-07-10 P5-T3 drift check: `trinity/phase_general/phase_events.py` still had no direct tests
before this phase. The event factories currently live at `phase_events.py:99-356`; event-result
checking and application live at `:363-416` and `:588-631`; the same module also has phase-specific
event-list builders at `:423-581`, which remain outside this phase's narrow factory/result scope.
The required mutation check changed the min-radius threshold-zero fixture from `R2=2.0` to `R2=2.1`,
and `pytest test/test_phase_events.py -q` failed on the zero-crossing assertion before the correct
fixture was restored.
Targeted phase-events, collection, docs convention, and full default pytest gates all pass after
the restore.

2026-07-10 P5-T4 drift check: `trinity/cloud_properties/validate_gmc.py` is now 717 lines, not
`~650`; there were still no direct `validate_gmc` tests before this phase. The `rCloud_max`,
edge-density, and mass-consistency reject conditions live at `validate_gmc.py:229-256`; the
`densPL` and `densBE` validator paths live at `:400-466` and `:473-542`. Low-level callers must
pass densities in `pc^-3` and `mu_convert` in `Msun`, matching the conversion convention documented
in `trinity/_input/sweep_runner.py:108-116`. The required mutation check changed the reject test's
`r_max` from `accepted.rCloud * 0.9` to `* 1.1`, and `pytest test/test_validate_gmc.py -q` failed
for both reject cases before the boundary was restored.
Targeted validate-gmc, collection, and full default pytest gates all pass after the restore.

## Executor ground rules

- **Canonical env** = the project requirement: Python ≥ 3.9, pinned stack (`numpy<2`, `scipy<2`).
  Known dev-machine gotchas (2026-07-06): default `python3` is anaconda **3.8**; under the Bash
  sandbox 3 `test_dR2min_magic_number.py` tests fail writing `/tmp/dR2cap` (pass outside sandbox);
  2 `solve_R1` tests fail on scipy 1.10.1 (fixed by P1). **Never capture golden values on the
  py3.8/scipy-1.10 interpreter** — goldens are recorded on the canonical env only, with the
  python/numpy/scipy versions noted in a comment next to the values.
- Phases are ordered by value-per-risk; each is independently landable unless noted. Do them in
  order, update the ledger row when a phase lands, and record actual gate numbers.
- No new dependencies. Match existing test style. `black .` + pre-commit ruff after each phase.
- Baseline at audit time: `pytest --collect-only -q` → **636 collected / 3 deselected (stress)**.

## Ledger (living — update on every phase)

| Phase | What | Status | Gate result (actual) |
|---|---|---|---|
| P0 | Capture baseline counts | [x] DONE | 2026-07-09 working tree based on `9410212d`: `pytest --collect-only -q` → 748/751 collected, 3 deselected; after LSODA callback fix, full default `pytest` → 748 passed, 3 deselected |
| P1 | `solve_R1` non-finite guard + test dedupe | [x] DONE | 2026-07-09 working tree based on `9410212d`: source guard already present; duplicate test assertion removed; `pytest test/test_r1_bracket.py test/test_energy_collapse_guard.py` → 14 passed; full default `pytest` → 748 passed, 3 deselected |
| P2 | Delete/merge 4 test files | [x] DONE | 2026-07-09: removed 15 nodes, relocated/salvaged/merged 6 nodes; `pytest test/test_cloudy_run_loader.py test/test_metadata.py -q` → 87 passed; `pytest --collect-only -q` → 739/742 collected, 3 deselected; full default `pytest` → 739 passed, 3 deselected |
| P3 | Stress-mark 2 slow tests | [x] DONE | 2026-07-10: `pytest --collect-only -q -m stress` → 9/742 collected, 733 deselected, including `test_energy_collapse_snapshot.py::test_energy_collapse_emits_no_negative_Pb` and `test_simplify.py::TestTiming`; default `pytest --collect-only -q` → 733/742 collected, 9 deselected; full default `pytest` → 733 passed, 9 deselected in 77.64s |
| P4 | Cheap batch: coverage holes + stale prose | [x] DONE | 2026-07-10: `pytest test/test_active_when.py test/test_materialize_runtime.py test/test_metadata.py test/test_betadelta_solver_switch.py test/test_residual_resample.py test/test_r1_shadow.py test/test_registry.py -q` → 119 passed; full default `pytest` → 733 passed, 9 deselected in 74.89s |
| P5-T1 | Smoke golden values | [x] DONE | 2026-07-10: captured canonical final-row goldens from `pytest test/test_run_smoke.py -q` after verifying the new test fails when the `R2` golden is intentionally broken; restored goldens; `pytest test/test_run_smoke.py -q` → 1 passed; full default `pytest` → 733 passed, 9 deselected in 93.95s |
| P5-T2 | Default-CI phase-boundary golden | [x] DONE | 2026-07-10: added `test/test_phase_boundary.py`; shortened `stop_t=0.004` capture matches stress `_GOLDEN[:2]`; mutation of first beta golden failed as expected; restored; `pytest test/test_phase_boundary.py -q` → 1 passed; `pytest --collect-only -q` → 734/743 collected, 9 deselected; full default `pytest` → 734 passed, 9 deselected in 191.99s |
| P5-T3 | `phase_events` unit tests | [x] DONE | 2026-07-10: added `test/test_phase_events.py`; mutation of min-radius zero fixture failed as expected; restored; `pytest test/test_phase_events.py -q` → 10 passed; `pytest --collect-only -q` → 744/753 collected, 9 deselected; full default `pytest` → 744 passed, 9 deselected in 199.90s |
| P5-T4 | `validate_gmc` accept/reject | [x] DONE | 2026-07-10: added `test/test_validate_gmc.py`; plausible `densPL`/`densBE` clouds are accepted, and otherwise-identical clouds with `r_max` below computed `rCloud` are rejected; mutation of the `r_max` boundary failed as expected; restored; `pytest test/test_validate_gmc.py -q` → 4 passed; `pytest --collect-only -q` → 748/757 collected, 9 deselected; full default `pytest` → 748 passed, 9 deselected in 181.84s |
| P5-T5 | `read_sps` known-value | [x] DONE | 2026-07-10: added `test/test_read_sps.py`; mutation of the `Qi` first-row golden failed as expected; restored; `pytest test/test_read_sps.py test/test_docs_dev_conventions.py -q` → 110 passed; `pytest --collect-only -q` → 749/758 collected, 9 deselected; full default `pytest` → 749 passed, 9 deselected in 186.80s |
| P5-T6 | Malformed-`.param` trust boundary | [x] DONE | 2026-07-10: extended `test/test_validators.py`; unknown keys and valueless lines are pinned as `ParameterFileError`, duplicate user keys are pinned as current later-value-wins behavior; mutation of the duplicate-key expectation failed as expected; restored; `pytest test/test_validators.py test/test_docs_dev_conventions.py -q` → 135 passed; `pytest --collect-only -q` → 752/761 collected, 9 deselected; full default `pytest` → 752 passed, 9 deselected in 181.68s |

---

## P0 — Capture baseline

**WHY.** P2's gate is count accounting; the "before" must survive the change (CLAUDE.md rule 5).

**DO.** On the canonical env, outside the sandbox:

```
pytest --collect-only -q | tail -2        # expect ~636 collected / 3 deselected at 70f07532
pytest                                    # record passed/failed; known-failure baseline above
```

Record both numbers in the ledger. **DONE-WHEN** the ledger row has real numbers.

---

## P1 — `solve_R1` non-finite guard (the only `trinity/` source change in this plan)

**WHY (frozen — do not re-derive).** Two tests pin "NaN `Eb` on a physical bubble raises":
`test_r1_bracket.py:58-61` and `test_energy_collapse_guard.py:71-73`. But `solve_R1`
(`trinity/bubble_structure/get_bubbleParams.py:414-446`) has **no guard of its own** — the raise
comes from a NaN check *inside newer scipy's `brentq`*. On scipy 1.10.1 (inside the project's
`scipy>=1.7,<2` pin) `brentq` silently returns R1 ≈ 1e-12: a fabricated value, exactly what the
docstring at `:429-430` promises never to happen. Note `get_r1`'s `Ebubble < 1e-30` floor at
`:406` does **not** catch NaN (`NaN < x` is False), so NaN propagates into the sqrt.

**CHANGE.** In `solve_R1`, immediately after the `R2 > 0` check (`:434-435`), add:

```python
if not (np.isfinite(Eb) and np.isfinite(v_mech_total)):
    raise ValueError(
        f"solve_R1: non-finite input for a physical bubble: "
        f"Eb={Eb}, v_mech_total={v_mech_total}"
    )
```

(`np` is already imported in this module.) Guarding `v_mech_total` too is deliberate: it poisons
the same equation the same way; one check covers the class. Then **dedupe**: delete the duplicate
assertion `test_energy_collapse_guard.py:71-73` (the `pytest.raises(ValueError)` block only —
the three `== 0.0` asserts above it stay), leaving a one-line comment pointing to
`test_r1_bracket.py::test_failure_raises_instead_of_fabricating` as the canonical pin.

**SIZING (frozen judgment — saves the executor hours).** This is a precondition `raise` before
`brentq`, adding **zero floating-point operations on the finite path** — bit-identical for all
finite inputs *by construction*. Despite living in `bubble_structure/`, it does NOT need the
full-run equivalence ladder. Gate = unit tests only. Do not run multi-hour sims for this.

**GATE.**
```
pytest test/test_r1_bracket.py test/test_energy_collapse_guard.py    # green
pytest                                                               # green vs P0 baseline
```
If a scipy-1.10 interpreter is available (the dev machine's anaconda py3.8), run the two files
there too — they failed before this change; both must pass after. **DONE-WHEN** both commands
green and the ledger records it.

---

## P2 — Delete / merge four test files (~350 lines)

Every removed test node must land in the table below as either **deleted** (with the frozen
evidence) or **relocated to** `<file>::<test>`. No silent drops.

2026-07-09 accounting:

| Removed node | Disposition |
|---|---|
| `test_tavg_volume.py::test_abs_volume_telescopes_to_full_domain` | Deleted: tautological local-helper arithmetic |
| `test_tavg_volume.py::test_signed_volume_undercounts_due_to_sign_bug` | Deleted: tautological local-helper arithmetic |
| `test_phase4_consumer_migration.py::TestRcloudSmoothingMigration::test_end_reason_resolved_from_v3_termination` | Deleted: strict subset of metadata termination-block coverage |
| `test_phase4_consumer_migration.py::TestRcloudSmoothingMigration::test_end_reason_resolved_from_legacy_text` | Relocated to `test_metadata.py::TestReadSimulationEndMigration::test_v1_metadata_without_termination_falls_back_to_text` |
| `test_phase5_text_drop.py::TestNoLegacyTextWrites::test_no_simulation_end_txt` | Deleted: covered by `test_metadata.py::TestReadSimulationEndMigration::test_reads_from_metadata_block` |
| `test_phase5_text_drop.py::TestNoLegacyTextWrites::test_no_termination_debug_txt` | Deleted: already covered by `test_metadata.py::TestTerminationDebugBlock::test_no_termination_debug_txt_written` |
| `test_phase5_text_drop.py::TestNoLegacyTextWrites::test_no_summary_txt` | Deleted: tautological signature check |
| `test_phase5_text_drop.py::TestNoLegacyTextWrites::test_only_v4_artefacts_remain` | Relocated to `test_metadata.py::TestTerminationDebugBlock::test_only_v4_artefacts_remain` |
| `test_phase5_text_drop.py::TestDeprecationWarnings::test_read_simulation_end_warns_on_text_fallback` | Deleted: subset of `test_metadata.py::TestReadSimulationEndMigration::test_falls_back_to_text_for_legacy_runs` |
| `test_phase5_text_drop.py::TestDeprecationWarnings::test_cloudy_parse_summary_txt_warns` | Relocated to `test_cloudy_run_loader.py::test_parse_summary_txt_warns_on_legacy_text` |
| `test_phase5_text_drop.py::TestDeprecationWarnings::test_cloudy_parse_simulation_end_warns` | Relocated to `test_cloudy_run_loader.py::test_parse_simulation_end_warns_on_legacy_text` |
| `test_phase5_text_drop.py::TestDeprecationWarnings::test_cloudy_load_run_v4_emits_no_warnings` | Relocated to `test_cloudy_run_loader.py::test_load_run_v4_emits_no_deprecation_warnings` |
| `test_cloudy_package_exports.py::test_package_reexports_public_api` | Merged into `test_cloudy_run_loader.py::test_package_reexports_public_api` |
| `test_cloudy_package_exports.py::test_package_all_matches_exports` | Merged into `test_cloudy_run_loader.py::test_package_reexports_public_api` |
| `test_cloudy_package_exports.py::test_package_re_exports_are_the_same_objects` | Merged into `test_cloudy_run_loader.py::test_package_reexports_public_api` |

**2a. `test_tavg_volume.py` — DELETE (tautological).** It never imports `trinity`:
`_vol_signed`/`_vol_abs` at `test:13-22` are local re-implementations, so reverting the real fix
at `bubble_luminosity.py:808-816` (which does use `abs()` — verified) would not fail it. It
asserts arithmetic facts about its own helpers. Rewrite was considered and rejected: exercising
the real Tavg path needs a full profile state; P5-T2's trajectory golden covers Tavg-consuming
output implicitly. Salvage: nothing.

**2b. `test_phase4_consumer_migration.py` — DELETE, salvage one test.** It never imports the
consumer it documents (`paper/methods/figures/paper_rcloud_smoothing.py:190`) — it only *emulates*
the `.get("detail")` pattern at `test:97-106`, so the regression it guards cannot be caught. Its
reader assertions are strict subsets of `test_metadata.py:673-679` (exact termination key-set) and
`:854-880` (legacy text fallback + DeprecationWarning). **Salvage into `test_metadata.py`
TestLegacy:** the one uncovered wrinkle — a legacy run dir with a **v1 `metadata.json` present**
(no `termination` block) *plus* the text file → `read_simulation_end` falls back to text with the
DeprecationWarning. Also drop its unused `import numpy as np`.

**2c. `test_phase5_text_drop.py` — DELETE after relocating survivors.**
Duplicates to drop: `test_no_simulation_end_txt` ⊂ `test_metadata.py:847`;
`test_no_termination_debug_txt` == `test_metadata.py:927-932`;
`test_read_simulation_end_warns_on_text_fallback` ⊂ `test_metadata.py:854-880`.
Tautology to drop: `test_no_summary_txt` (`test:83-97`) asserts a removed kwarg is absent from
`read_param`'s signature — can't meaningfully regress.
**Survivors to relocate:** `test_only_v4_artefacts_remain` (directory-layout check) →
`test_metadata.py`; the three cloudy tests (`_parse_summary_txt`/`_parse_simulation_end`
DeprecationWarning + `load_run` v4 warning-free path) → `test_cloudy_run_loader.py`, which
currently has **zero** `pytest.warns` coverage. Also drop its unused `import numpy as np`.

**2d. `test_cloudy_package_exports.py` — MERGE.** 42 lines / 3 trivial re-export assertions
against `cloudy/__init__.py`; fold into one test in `test_cloudy_run_loader.py`, delete the file.

**GATE — count accounting.**
```
pytest --collect-only -q | tail -2
```
Compute expected count: P0 baseline − (deleted nodes) + (relocated/salvaged nodes); the actual
must match the arithmetic, and the table above must account for every removed node. Then full
`pytest` green. **DONE-WHEN** counts reconcile and the ledger records before/after.

---

## P3 — Markers (runtime hygiene)

**3a. `test_energy_collapse_snapshot.py`** runs a ~1–2 min full `run.py` subprocess with no
marker — the slowest test in the default suite. Add `@pytest.mark.stress`. **Frozen trade:** this
pairs with P5-T2, which *adds* a ~1–2 min boundary-crossing test to the default suite — net
default runtime stays ~flat while default CI's expensive slot switches from the rarer
energy-collapse edge to the canonical phase-boundary path (the collapse edge keeps its fast unit
tests in `test_energy_collapse_guard.py` by default).

**3b. `test_simplify.py` TestTiming (`:775-813`)** asserts wall-clock budgets — flake risk under
CI load. Mark the class `@pytest.mark.stress`.

**GATE.** `pytest --collect-only -q -m stress` lists both; default collect no longer does; note
the default-suite wall-time drop in the ledger.

---

## P4 — Cheap batch: stale coverage holes + prose (test-only, no behavior change)

- `test_active_when.py:29` `_BE_RUNTIME_ADDS` lists 8 names; the densBE branch adds **9** —
  add `densBE_sigma` (`registry.py:491`, added after this test's last edit).
- `test_materialize_runtime.py:57-60` `test_skip_active_when_specs` — add `densBE_sigma` there too.
- `test_metadata.py`: remove unused top-level `import warnings` (`:19`) and the dead no-op
  `_make_params` statement (`:430`).
- Stale docstrings (text only, tests are current): `test_betadelta_solver_switch.py` header still
  says legacy is the default (hybr is, `registry.py:309`); `test_residual_resample.py` header
  describes the "staged P3 patch" as future (it shipped — `_RESIDUAL_NPTS=500` in production);
  `test_r1_shadow.py` says triggers "only feed shadow logging" (non-default sets now drive the
  transition, `registry.py:349`); `test_registry.py` docstring says "187 parameters" — **drop the
  number entirely** rather than updating it to 200, so it can't go stale again.

**GATE.** Full `pytest` green; `git diff` touches only the lines listed. **DONE-WHEN** green.

---

## P5 — New tests (ranked; each is the smallest test that closes its gap)

**Shared gate for every new test:** demonstrate it can fail — mutate the guarded logic once
locally (flip the sign / break the value), watch the test fail, revert. A new test that never
failed is P2a waiting to happen.

### T1 — Smoke golden values (extend `test_run_smoke.py`; zero new runtime)

**Gap (frozen):** the smoke test asserts exit code + file existence, zero values — a silent
physics regression passes the entire default suite. **Change:** after the existing run, parse
`dictionary.jsonl` (pattern in `test_betadelta_hybr_stress.py:88-100`) and assert on the final
row: `R2`, `v2`, `Eb` finite and > 0, and each `== pytest.approx(golden, rel=1e-6)`. Snapshot keys
frozen from `registry.py:391-395`: `t_now` [Myr], `v2` [pc/Myr], `R2` [pc], `Eb` [Msun·pc²/Myr²].
**Golden capture:** run the smoke param once on the canonical env, paste the three values with an
env comment (python/numpy/scipy versions). **Fallback (frozen):** if a future in-pin scipy/numpy
patch shifts last bits past 1e-6, loosen to `rel=1e-4` with a dated comment — never delete the
assertion. Also fix the stale "~2.5 min" header docstring while in the file.

2026-07-10 DONE: `test_run_smoke.py` now parses the existing run's final `dictionary.jsonl`
row and checks `R2`, `v2`, and `Eb` are finite, positive, and equal to canonical-env goldens
at `rel=1e-6`; no new subprocess or test node was added.

### T2 — Default-CI phase-boundary golden (new `test/test_phase_boundary.py`)

**Gap (frozen):** the phase sequence (energy → implicit) and `main.run_expansion` orchestration
never execute in default CI — the only boundary-crossing tests are stress-marked, i.e. deselected
by `pyproject.toml`'s `-m 'not stress'`. **Change:** ONE `run.py` subprocess (borrow `_run` from
`test_betadelta_hybr_stress.py:74-85`), param identical to its `_PARAM` (`:46-53`, boundary fires
~0.0029 Myr) **except `stop_t 0.004`** (roughly half the stress runtime). Assert: exit 0; the
ordered de-duplicated `current_phase` sequence over `dictionary.jsonl` rows `== ['energy',
'implicit']` (key: `registry.py:384`, values set in `main.py:244,278`); ≥ 2 implicit rows with
`betadelta_converged is True` and finite `cool_beta`/`cool_delta`; the first two (beta, delta)
pairs match `_GOLDEN[:2]` from the stress file (t=0.00341, 0.00381 — both < 0.004) at the same
`abs=2e-3`. **Capture check (frozen):** truncating at 0.004 must not change segments before it —
verify once on the canonical env that the two rows match the stress goldens; if dt-boundary
effects shift them, record dedicated goldens for `stop_t=0.004` and note that here, dated.

2026-07-10 DONE: `test/test_phase_boundary.py` now runs one default-suite `run.py` subprocess with
`stop_t=0.004`, asserts the ordered phase sequence is `['energy', 'implicit']`, requires at least
two converged implicit rows with finite beta/delta, and compares the first two beta/delta pairs to
the existing stress goldens at `abs=2e-3`.

### T3 — `phase_events` factories (new `test/test_phase_events.py`)

**Gap:** `trinity/phase_general/phase_events.py` — the event factories that decide phase-ending vs
simulation-ending fates for `solve_ivp` — has zero tests; a sign/threshold slip silently mislabels
the headline result of a run. Pure functions. **Change:** read the module first, then one
parametrized test that each factory's event function crosses zero exactly at its configured
threshold (evaluate at threshold ± ε), plus one test that `check_event_termination` /
`apply_event_result` classify a minimal synthetic `sol` correctly. ~1 test per factory, no more.

2026-07-10 DONE: `test/test_phase_events.py` now checks threshold signs, zero crossing, direction,
terminal status, and run-ending classification for the event factories, plus synthetic
`check_event_termination`/`apply_event_result` handling for both simulation-ending and phase-ending
events.

### T4 — `validate_gmc` accept/reject (new `test/test_validate_gmc.py`)

**Gap:** `cloud_properties/validate_gmc.py` (717 lines as of 2026-07-10; the `rCloud_max`
plausibility trust boundary CLAUDE.md names) had zero direct tests before P5-T4. **Change:** four
cases — a plausible GMC accepted and an implausible one rejected, for each of `densPL` and
`densBE`. Use physically plausible values per project convention (mCloud ~1e5 Msun,
sfe 0.01–0.3, nCore ~1e3 cm⁻³, rCore ~1 pc); read the module's actual reject conditions to build
the implausible cases — don't guess thresholds.

2026-07-10 DONE: `test/test_validate_gmc.py` now covers four direct validator cases: accepted
`densPL` and `densBE` GMCs with physically plausible mass/core-density values, plus targeted
`rCloud_max` rejections for each profile using the same computed clouds with the maximum radius
lowered below `rCloud`. `sfe` is not an input to this validator.

### T5 — `read_sps` known-value (new `test/test_read_sps.py`)

**Gap:** `trinity/sps/read_sps.py` + `trinity/sps/update_feedback.py` load the SB99 tables feeding
every run's source terms (Lw, Qi, pdot); only implicit coverage is the smoke exit code — classic
silent-corruption surface (units are the project's recurring bug class). **Change:** load the
bundled default table from `lib/default/` through `read_sps`, assert the time grid is strictly
monotonic, and one known value per key column at the first grid point (captured once on the
canonical env, committed with an env comment), with units cross-checked via `sps_columns`.

2026-07-10 drift check before coding: the loader entry point is still `read_sps.read_sps`
(`trinity/sps/read_sps.py:38-131`), with conversion/monotonic validation/t=0 prepend in
`_read_sps_user` (`:134-282`). The default canonical unit registry and bundled SB99 column map are
still in `trinity/sps/sps_columns.py:65-174`; monotonic-time validation is at `:334-372`.
`trinity/sps/update_feedback.py:98-185` consumes the interpolators built from these arrays. The
phase remains test-only; no production edit is needed.

2026-07-10 P5-T5 execution: `test/test_read_sps.py` now resolves the bundled default SPS file,
loads it through `read_sps.read_sps`, checks the prepended/strictly increasing time grid, verifies
the default column map/canonical units via `sps_columns`, and pins the first file row for all
returned source-term arrays. The required mutation check changed the captured `Qi` golden by a
factor of 10, and `pytest test/test_read_sps.py -q` failed on the known-value assertion before the
golden was restored.
Targeted read-sps, docs convention, collection, and full default pytest gates all pass after the
restore.

### T6 — Malformed-`.param` trust boundary (extend `test_validators.py`)

**Gap:** `ParameterFileError` paths only cover specific validator messages; structurally malformed
input (unknown key, duplicate key, valueless line) is unpinned — silent dropping would be worse
than a crash. **Change (frozen intent):** this test **pins current behavior, it does not change
it**. First check what `read_param` actually does for each case, then write one parametrized test
asserting exactly that (raise → `pytest.raises(ParameterFileError)`; documented ignore → assert
the ignore). If the discovered behavior looks wrong (e.g. unknown keys silently dropped), flag it
in this doc, dated — fixing it is a separate, user-approved change.

2026-07-10 drift check before coding: user-line parsing still lives in
`trinity/_input/read_param.py:181-206`; valueless user lines raise `ParameterFileError` at
`:198-202`; unknown keys raise `ParameterFileError` at `:214-225`; duplicate user keys currently
overwrite earlier values via `user_dict[key] = value` at `:204-206`. The duplicate-key overwrite is
current behavior, not an endorsement: silently masking an earlier user value looks like a separate
trust-boundary fix candidate, but this phase only pins it.

2026-07-10 P5-T6 execution: `test/test_validators.py` now pins three malformed-user-`.param`
cases: unknown keys raise `ParameterFileError`, valueless lines raise `ParameterFileError`, and
duplicate user keys currently take the later value. The required mutation check inverted the
duplicate-key expectation to first-value-wins, and
`pytest test/test_validators.py::test_read_param_duplicate_user_key_uses_later_value -q` failed
before the correct current-behavior assertion was restored. Targeted validators/docs, collection,
and full default pytest gates all pass after the restore.

---

## Backlog — identified, deliberately NOT scheduled

- **conftest consolidation.** There is no `test/conftest.py`; ~200 duplicated lines:
  `disable_crash_handlers` ×5 (`test_metadata.py:40`, phase4 `:25`, phase5 `:29`,
  `test_registry.py:36`, `test_show_run.py:29` — two die with P2), cooling-cube fixture loader ×3
  (`test_residual_resample.py:55-100`, `test_dR2min_magic_number.py:67-130`,
  `test_net_coolingcurve.py:32-48`), stress subprocess harness ×2, `make_props` ×2,
  `MOCK_FULLRUN` ×3. Consolidate **opportunistically when next editing those files** — not as its
  own churn PR (working code, fixture-scope risk, zero behavior value).
- **CIE cooling-curve edges** (`cooling/CIE/read_coolingcurve.get_Lambda`): tabulated-T
  reproduction + out-of-range contract. (Non-CIE side is tested; CIE isn't.)
- **Density/mass profile consistency:** quadrature of `density_profile` ≈ `mass_profile(rCloud)`
  ≈ `mCloud` within 1%, both profiles.
- **`sweep_runner.generate_param_file` round-trip:** generate one combination → `read_param` it →
  swept keys equal intended values.
- **`shell_structure` driver:** one committed snapshot fixture → `f_absorbed ∈ [0,1]`, mass
  closure. Only pays off after T2 exists (its regressions otherwise show up only as slightly
  wrong trajectories).
- **`test_mu_audit_drift.py` heads-up:** its source-string-count tests fail on any *legitimate*
  refactor of the counted files. When that happens: keep the numeric-formula tests (the durable
  core), re-derive the counts — don't delete the file.

## Appendix A — audit verdict table (48 files, 2026-07-06 @ `70f07532`)

Verdicts: KEEP (verified live), or the P-phase that handles it. Evidence abbreviated; the
slice-audit detail lives in the phase sections above where action is required.

| File | Verdict | Note |
|---|---|---|
| test_active_when.py | KEEP (P4 hole) | `densBE_sigma` missing from `_BE_RUNTIME_ADDS` |
| test_barnes_population.py | KEEP (local-only) | file itself gitignored (`.gitignore:24`), targets gitignored `paper/barnes26/`; CI/fresh clones never see it — deliberate |
| test_betadelta_dt_mitigation.py | KEEP | all constants/functions current |
| test_betadelta_hybr.py | KEEP | rescue-ladder semantics match `get_betadelta.py:649-675` |
| test_betadelta_hybr_stress.py | KEEP | stress-marked golden; healthy path unaffected by rescue ladder |
| test_betadelta_solver.py | KEEP | legacy grid path still live (`betadelta_solver=legacy` + rescue ladder) |
| test_betadelta_solver_switch.py | KEEP (P4 prose) | asserts the flipped hybr default correctly; header stale |
| test_bubble_lsoda_quiet.py | KEEP | `_quiet_lsoda_fortran` live, load-bearing |
| test_bubble_solver_failures.py | KEEP | ok=False/raise contract is live production logic |
| test_bubble_solver_stress.py | KEEP | statistical LSODA crash gate, stress-marked |
| test_cf_leak.py | KEEP | all symbols current |
| test_cloudy_cli.py | KEEP | e2e fixture git-tracked; in-process CLI |
| test_cloudy_dlaw.py | KEEP | pure unit tests, symbols live |
| test_cloudy_package_exports.py | **P2d merge** | 3 trivial re-export asserts |
| test_cloudy_run_loader.py | KEEP | gains phase5 survivors + exports test (P2c/d) |
| test_cloudy_snapshot_to_deck.py | KEEP | symbols live incl. `get_at_time(mode=, quiet=)` |
| test_conventional_units.py | KEEP | display-layer scope, NOT redundant with test_unit_conversions |
| test_cooling_boost.py | KEEP | `effective_Lloss` wired into both energy phases |
| test_dR2min_magic_number.py | KEEP | no-floor invariant live; heaviest non-stress file (cooling cubes rebuilt per test — backlog conftest) |
| test_energy_collapse_guard.py | KEEP (P1 dedupe) | drop duplicate NaN-raise assert only |
| test_energy_collapse_snapshot.py | KEEP (P3a mark) | ~1–2 min unmarked subprocess |
| test_engine_purity.py | KEEP | live invariant; survived two restructures doing its job |
| test_fkappa_auto.py | KEEP | resolver wired at `registry.py:353` |
| test_gen_default_param.py | KEEP | byte-exact gate + documented drift-localizers |
| test_log_stopping_fate.py | KEEP | targets `terminal_prints.py`, not show_run — no merge pair |
| test_materialize_runtime.py | KEEP (P4 hole) | magic counts re-derived and hold (106 adds, 9/97) |
| test_metadata.py | KEEP | canonical writer/reader contract suite; gains P2b/c salvage |
| test_mu_audit_drift.py | KEEP | guards live composition invariant; see backlog heads-up |
| test_net_coolingcurve.py | KEEP | redundant `sys.path.insert` wart only |
| test_operations_monotonic.py | KEEP | tolerant-monotonicity path in use |
| test_phase4_consumer_migration.py | **P2b delete** | emulates its consumer; subset of test_metadata |
| test_phase5_text_drop.py | **P2c delete+relocate** | half duplicates test_metadata; one tautology |
| test_r1_bracket.py | KEEP (P1 fixes) | canonical NaN-raise pin; hollow until P1 lands |
| test_r1_shadow.py | KEEP (P4 prose) | truth-table contract unchanged |
| test_registry.py | KEEP (P4 prose) | byte-identical reconciliation gate for the registry |
| test_residual_resample.py | KEEP (P4 prose) | guards `_RESIDUAL_NPTS` against too-small regression |
| test_resolvers.py | KEEP | no duplicated assertions vs test_registry |
| test_run_smoke.py | KEEP (P5-T1 done) | e2e install check now pins final-row `R2`/`v2`/`Eb` goldens |
| test_shell_overflow_guard.py | KEEP | `_NSHELL_MAX` bounds consistent |
| test_show_run.py | KEEP | CLI semantics verified |
| test_simplify.py | KEEP (P3b mark) | simplify is live in the snapshot-writer hot path; TestTiming unmarked |
| test_sweep_jobs.py | KEEP | asserted sbatch/report strings current (in sync Jul 3) |
| test_sweep_workers.py | KEEP | different modules from sweep_jobs; 5 slow-ish subprocess tests |
| test_tavg_volume.py | **P2a delete** | tautological — never imports trinity |
| test_theta5_harvest.py | KEEP | only test targeting docs/dev scripts; path-coupled to doc reorgs |
| test_unit_conversions.py | KEEP | `convert2au` parser characterization |
| test_validators.py | KEEP (P5-T6 done) | covers validator wiring/companion traps and now pins malformed `.param` unknown-key, valueless-line, and duplicate-key behavior |

`test/data/`: 2 small JSON fixtures, both referenced, none orphaned.

## Appendix B — coverage gaps (from the sweep; scheduled items → P5)

Uncovered at audit time: `phase1_energy/run_energy_phase` + `energy_phase_ODEs`,
`phase1c_transition/run_transition_phase`, `phase2_momentum/run_momentum_phase`,
`phase0_init/get_InitCloudProp`/`get_InitCloudyDens`,
`density_profile`/`mass_profile`/`initial_profile`/
`powerLawSphere` (backlog), `cooling/CIE/read_coolingcurve` (backlog), `_output/header`,
`_functions/logging_setup`/`extract_example_snapshots`,
`_analysis/check_yesno`. Config finding at audit time: `pyproject.toml` `addopts = "-m 'not
stress'"` meant the only boundary-crossing pipeline tests never ran by default; P5-T2 now adds one
default-suite boundary-crossing golden.
