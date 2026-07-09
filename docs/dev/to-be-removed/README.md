# to-be-removed — deletion candidates awaiting the maintainer's personal review

Staging area per `docs/dev/CONVENTIONS.md` §Retiring work: nothing here is deleted by an agent;
the maintainer reviews each entry and deletes personally. Exempt from banner checks.

Previous round (per-doc DOC_STATUS ledger + `arms_mock4e3.attempt1.jsonl`) was reviewed and
deleted in commit `9459234a` (2026-07-06).

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
