# test/ — suite facts

- `pytest` runs the default set (`pyproject.toml` addopts apply `-m 'not stress'`);
  `pytest -m stress` is the opt-in slow set. Single file:
  `pytest test/test_unit_conversions.py`.
- Baseline: **0 failed is the invariant**; pass counts only grow as tests are added. As of
  2026-08-14 the tree collects **1094 (1078 default + 16 stress)**. More passing tests is fine; any
  failure is a regression. (Earlier reference, kept for trend: 743 passed / 0 failed / 3 skipped on
  the maintainer's machines 2026-07, 770 collected 2026-07-12.)
- ⚠️ **The invariant is currently violated and it is not your change.** On `main` at 2026-08-14 the
  default suite is **3 failed / 1075 passed / 16 deselected**: `test_run_smoke.py`,
  `test_phase_boundary.py` (both phase-1a-exit goldens, −1.1%) and
  `test_mu_audit_drift.py::test_phase1_all_eleven_sites_refined_and_no_original_remains` (a site
  count, 11 → 5). All three are the C3c landing (`c43a50e`, PR #738), which shipped without
  re-baselining; `docs/dev/phii-identity/PLAN.md` has the mechanism and the re-baseline authority.
  `test_betadelta_hybr_stress.py` is red for the same reason under `-m stress`. Compare against
  these three before concluding you broke something.
- A few tempdir-dependent tests flake **only under the Claude Code sandbox** (where `/tmp` is not
  writable) — not a real regression. Re-run with `TMPDIR` pointing at a writable dir before
  treating them as failures; never "fix" the test for the sandbox.
- `test/test_barnes_population.py` is gitignored (local-only) — don't expect it in containers.
- Tests use physically plausible values, not convenient round numbers (rCore ≈ 1 pc, realistic
  GMC masses/densities; `rCloud_max` plausibility must pass). Keep it that way — unphysical
  inputs exercise regimes the code never runs in and hide real regressions.
