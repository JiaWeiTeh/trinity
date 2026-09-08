# test/ — suite facts

- `pytest` runs the default set (`pyproject.toml` addopts apply `-m 'not stress'`);
  `pytest -m stress` is the opt-in slow set. Single file:
  `pytest test/test_unit_conversions.py`.
- Baseline: **0 failed is the invariant**; pass counts only grow as tests are added. Two numbers
  now, because they legitimately differ (next bullet): **with `docs/dev` present** ~1141 collected;
  **without it** (fresh clone, CI) 841 collected — 821 passed / 20 skipped / 11 deselected, measured
  2026-09-02. More passing tests is fine; any failure is a regression. (Earlier reference, kept for
  trend: 743 passed / 0 failed / 3 skipped on the maintainer's machines 2026-07.)
- **`docs/dev` is untracked** as of `a32b098` (it is in `.gitignore`), so it is absent in any fresh
  clone and in CI. Thirteen test files load harness modules, `.param` configs or CSV fixtures from
  it; each now **skips** when the specific artifact it needs is missing, instead of erroring. Three
  of them used to load at module scope, which failed *collection* and aborted the whole run — if you
  add another test that reads `docs/dev`, guard it the same way or you will take the entire suite
  down in CI. Guard on the exact file, not the directory: checking out across `a32b098` leaves a
  partially-populated `docs/dev` that passes a directory check and then fails on a missing file.
  Cost: ~50 tests skipped in CI. Two of them need only one small `.param` each
  (`base_param` in `test/data/*_fixture.json`, pointing into `docs/dev`); **vendoring those two
  files into `test/data/` would win back 10 tests** — a known, deliberate follow-up, not an oversight.
- The goldens on the **phase-1a exit state** (`test_run_smoke.py`, `test_phase_boundary.py`,
  `test_betadelta_hybr_stress.py`) were re-baselined 2026-08-14 for C3c (`c43a50e`), which removed
  the `P_HII` channel that had been carrying un-ramped pressure past `dt_switchon`. Before/after
  table, mechanism and reproduce commands: `docs/dev/phii-identity/data/g34_golden_rebaseline.csv`.
  These three move together — a change to the 1a exit state moves all of them, which is a signal
  about the change, not three separate regressions.
- A few tempdir-dependent tests flake **only under the Claude Code sandbox** (where `/tmp` is not
  writable) — not a real regression. Re-run with `TMPDIR` pointing at a writable dir before
  treating them as failures; never "fix" the test for the sandbox.
- `test/test_barnes_population.py` is gitignored (local-only) — don't expect it in containers.
- Tests use physically plausible values, not convenient round numbers (rCore ≈ 1 pc, realistic
  GMC masses/densities; `rCloud_max` plausibility must pass). Keep it that way — unphysical
  inputs exercise regimes the code never runs in and hide real regressions.
