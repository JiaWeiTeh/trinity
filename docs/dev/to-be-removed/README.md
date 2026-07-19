# to-be-removed — deletion candidates awaiting the maintainer's personal review

Staging area per `docs/dev/CONVENTIONS.md` §Retiring work: nothing here is deleted by an agent;
the maintainer reviews each entry and deletes personally. Exempt from banner checks.

Previous round (per-doc DOC_STATUS ledger + `arms_mock4e3.attempt1.jsonl`) was reviewed and
deleted in commit `9459234a` (2026-07-06).

## Round of 2026-07-19 (orphaned test — broke CI collection)

### Moved here (whole file — imports a module that is not in this repo)

| file | original path | evidence | risk |
|---|---|---|---|
| `test_paper_cf_csv_loader.py` | `test/test_paper_cf_csv_loader.py` | committed by accident in `591e5e4` (a commit whose stated purpose was dropping interim `.gz` scan dicts). It does `from paper.rosette.figures.paper_Cf import load_cf_runs_from_csv, PC_MYR_TO_KM_S`, but **`paper/rosette/` and `paper_Cf` have never existed** in this repo or its git history (`git log --all -S load_cf_runs_from_csv` → only this test; `find . -name paper_Cf*` → nothing). The figs code confirms `paper_Cf` is external — `docs/dev/rosette-cf/figs/make_ssot_bestparam.py:189` calls it "the real paper_Cf on the pulled dicts" living on a local `paper/rosette/plots/` mount. The unresolved import failed collection in `pytest test/`, turning the whole suite red (ModuleNotFoundError: No module named 'paper.rosette'). | none for the repo — the test targets a paper-figure module that lives outside version control. If `paper_Cf` is ever vendored into the repo, restore this test alongside it. |

## Round of 2026-07-06 (solver-audit session)

### Moved here (whole files — zero references anywhere)

| file | original path | evidence | risk |
|---|---|---|---|
| `get_InitCloudyDens.py` | `trinity/phase0_init/get_InitCloudyDens.py` | 56 lines; sole function `create_InitCloudyDens()` has **zero callers** across `*.py`, `*.param`, `*.sh` (grep 2026-07-06 @ `70f07532`); only mentions are historical notes in `docs/dev/CODEBASE_REVIEW.md` + `codebase_review/02` + `misc/backward-compat-audit.md` | none for the code; the WARPFIELD-era cloudy-input writer it fed no longer exists. If cloudy export is ever revived, `trinity/_output/` owns that now. |

### Flagged, NOT moved (function-level or outward-facing — review in place)

| item | location @ `70f07532` | evidence | why not moved |
|---|---|---|---|
| `read_sweep_param()` | `trinity/_input/sweep_parser.py:262-352` (92 lines) | zero references; superseded by `read_sweep_config()` (same interface + tuple mode), which is what `run.py` and `sweep_jobs.py` call | a function inside a live module — delete the lines, don't move the file |
| `compute_minimum_rCore()` | `trinity/cloud_properties/mass_profile.py:566-627` (62 lines) | zero references anywhere; a diagnostic never hooked into the engine | same — line-range deletion inside a live module |
| `.readthedocs.yaml` | repo root; last commit 2025-04-01 | nothing in-repo references it | **outward-facing**: if the ReadTheDocs project is still live, removing it breaks the hosted docs build. Check the RTD dashboard first; delete only if the project is retired or configured elsewhere. |

After deleting a flagged line-range, re-run: `grep -rn "read_sweep_param\|compute_minimum_rCore" --include="*.py" .`
(expect zero hits) and `pytest -q`.
