# test/ — suite facts

- `pytest` runs the default set (`pyproject.toml` addopts apply `-m 'not stress'`);
  `pytest -m stress` is the opt-in slow set (9 tests). Single file:
  `pytest test/test_unit_conversions.py`.
- Baseline: **0 failed is the invariant**; pass counts only grow as tests are added. Full-suite
  reference on the maintainer's machines (2026-07): 743 passed / 0 failed / 3 skipped; as of
  2026-07-12 the tree collects 770 (761 default + 9 stress). More passing tests is fine; any
  failure is a regression.
- A few tempdir-dependent tests flake **only under the Claude Code sandbox** (where `/tmp` is not
  writable) — not a real regression. Re-run with `TMPDIR` pointing at a writable dir before
  treating them as failures; never "fix" the test for the sandbox.
- `test/test_barnes_population.py` is gitignored (local-only) — don't expect it in containers.
- Tests use physically plausible values, not convenient round numbers (rCore ≈ 1 pc, realistic
  GMC masses/densities; `rCloud_max` plausibility must pass). Keep it that way — unphysical
  inputs exercise regimes the code never runs in and hide real regressions.
