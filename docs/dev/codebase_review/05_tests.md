> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

# 05 — Test suite

**Scope:** Static audit of everything under `test/` from the perspective of a stranger who
`git clone`s the public repo, installs deps, and runs `pytest`. I cross-checked every fixture
reference and `from trinity…`/`import …` against `git ls-files` and the on-disk package, verified
the `.gitignore` "Private" claim about `test_barnes_population.py`, checked stress-marker usage
against `pyproject.toml`, and scanned for absolute/personal paths, skips/xfails, `sys.path` hacks,
debug prints, commented-out tests, `pdb`/`breakpoint()`, and TODO/FIXME. I did **not** run pytest
(env lacks numpy/scipy/pytest). 34 test files are tracked. The suite is in good shape: all imports
resolve to real modules, the only fixture data tests consume (`outputs/mockOutput/mockFullrun/`) is
fully committed, and the one "private" test is genuinely absent with no dangling references. The
findings below are all low-severity.

---

### [🟡] Cloudy tests resolve the mockFullrun fixture relative to the current working directory
- **Where:** `test/test_cloudy_run_loader.py:35`, `test/test_cloudy_cli.py:35`,
  `test/test_cloudy_snapshot_to_deck.py:40` — all use `MOCK_FULLRUN = Path("outputs/mockOutput/mockFullrun")`
- **Issue:** The fixture path is a **CWD-relative** string literal, not anchored to the test file.
  Every other path-using test in the suite anchors to the repo via
  `REPO_ROOT = Path(__file__).resolve().parents[1]` (e.g. `test/test_sweep_jobs.py:24`,
  `test/test_sweep_workers.py:24`, `test/test_run_smoke.py:17`) or uses `tmp_path`. These three
  cloudy files are the lone exception, so they only find the fixture when pytest is launched from
  the repo root. There is **no `conftest.py` anywhere** in the repo (`find` and `git ls-files` both
  return zero), so nothing `chdir`s to the root or makes this robust.
- **Impact (git-puller):** With the canonical `pytest` invocation from the repo root (which
  `pyproject.toml`'s `testpaths=["test"]` + `pythonpath=["."]` assume) these pass — pytest does not
  change CWD, so CWD stays the repo root. But a user who runs `cd test && pytest`, runs from an IDE
  with a different working dir, or runs a single file by absolute path from elsewhere will hit
  `RunLoadError`/`FileNotFoundError` on ~40+ assertions. It is a latent foot-gun, not a guaranteed
  failure.
- **Fix:** Anchor like the rest of the suite:
  `MOCK_FULLRUN = Path(__file__).resolve().parents[1] / "outputs" / "mockOutput" / "mockFullrun"`.

### [🟡] `test_run_smoke.py` is an unmarked ~2.5-minute end-to-end test that runs by default
- **Where:** `test/test_run_smoke.py:20` (`def test_quickstart_completes_cleanly`) — no
  `@pytest.mark.stress`; its own docstring at lines 4-5 says "slow integration test (~2.5 min on a
  typical machine)".
- **Issue:** The suite has a `stress` marker for "opt-in slow … tests (deselected by default)"
  (`pyproject.toml`), and two files use it. This smoke test is self-described as slow (it spawns a
  real `python run.py` subprocess with a 600 s timeout, line 43) yet carries **no** marker, so the
  default `pytest` run (`addopts = "-v --tb=short -m 'not stress'"`) executes it.
- **Impact (git-puller):** Functionally fine — it passes if the install is sound (in fact it is a
  *good* install smoke test). The surprise is purely UX: a fresh cloner running `pytest` expecting a
  quick unit pass sits through a multi-minute subprocess run with no progress output. Not a failure.
- **Fix:** Either leave as-is intentionally (it is a valuable install canary) or, if quick feedback
  is the goal, mark it `@pytest.mark.stress` (or a new `slow` marker registered in `pyproject.toml`)
  so it is opt-in like the other heavy tests.

---

## Verified clean (no finding warranted)

- **Private test genuinely absent, no dangling refs.** `.gitignore` lists
  `test/test_barnes_population.py` as "Private: kept local, not tracked".
  `git ls-files --error-unmatch test/test_barnes_population.py` → *"did not match any file"*, and
  `grep -rn "barnes" test/` returns nothing. No other test imports or references it. Correct, no leak.
- **All `from trinity…` / `import trinity…` / `import run` imports resolve.** Cross-checked every
  unique import in `test/` against the on-disk package: `phase1b_energy_implicit/{run_energy_implicit_phase,get_betadelta}.py`,
  `bubble_structure/{bubble_luminosity,get_bubbleParams}.py`, `_output/cloudy/{run_loader,dlaw,snapshot_to_deck,trinity_to_cloudy}.py`,
  `phase0_init/get_InitPhaseParam.py`, `sps/sps_columns.py`, `shell_structure/get_shellODE.py`,
  `_output/{simulation_end,run_constants,trinity_reader}.py`, `_input/{dictionary,registry,errors,read_param}.py`,
  `cloud_properties/bonnorEbertSphere.py`, `_functions/{operations,unit_conversions,cluster}.py` — all present.
  `import run` (`test/test_sweep_workers.py:21`) resolves to the root `run.py` via `pythonpath=["."]`.
  No stale import paths or renamed-symbol references found.
- **The only consumed fixture tree is fully committed.** All 18 fixture references in `test/` point
  at `outputs/mockOutput/mockFullrun/`, which is tracked (`git ls-files` shows
  `dictionary.jsonl`, `metadata.json`, `4e3_sfe001_n5e2_PL0_summary.txt`, `simulationEnd.txt`,
  `4e3_sfe001_n5e2_PL0.param`, `termination_debug.txt`) and survives `.gitignore`'s
  `outputs/*` rule via the `!outputs/mockOutput/` un-ignore. Spot-checked the values the tests
  assert: `dictionary.jsonl` has exactly 178 lines (matches `test_cloudy_run_loader.py:59`),
  summary `ZCloud 1.0` / `dens_profile densPL` / `allowShellDissolution True` and
  metadata `model_name 4e3_sfe001_n5e2_PL0` all match the assertions. `mockCloudyinput/` and
  `mockOnlyphase/` are committed but not read by any test.
- **No absolute/personal paths.** The only `/tmp/...`, `/home`, `~/` hits are the string literal
  `"path_sps   /tmp/lib/sps/starburst99/\n"` fed to a parser in
  `test/test_cloudy_run_loader.py:136-138` (it asserts the parser keeps paths as strings — no file
  is opened). No machine names, no `expanduser`/`Path.home`, no author-specific assumptions.
- **No `sys.path` hacks, `os.chdir`, `breakpoint()`/`pdb`/`set_trace`, debug `print(`,
  commented-out `def test_`, or TODO/FIXME** anywhere in `test/`. (The two "TODO" hits in
  `test/test_cloudy_cli.py:306,348,360` are assertions about a *string the CLI emits*, not test debt.)
- **Stress markers consistent with config.** `@pytest.mark.stress` appears on all tests in
  `test/test_betadelta_hybr_stress.py` (2 markers / 2 tests) and `test/test_bubble_solver_stress.py`
  (1 marker / 1 test); the `stress` marker is registered in `pyproject.toml`'s `[tool.pytest.ini_options].markers`,
  so default runs cleanly deselect them with no "unknown marker" warning. No skipped/xfail tests
  exist; `pytest.importorskip("astropy.units")` in `test/test_conventional_units.py:38-55` is a
  legitimate optional-dep guard (astropy is a hard dependency anyway, so it will not skip on a
  proper install).
- **Cluster/CPU tests are portable.** `test/test_sweep_workers.py` monkeypatches
  `os.sched_getaffinity`/`os.cpu_count` and SLURM env vars rather than hardcoding the host's CPU
  count, so it does not assume a particular machine.

## Counts: 0 high / 0 medium / 2 low
