# CLAUDE.md

TRINITY (`trinity-sf`) is a feedback-driven bubble-evolution astrophysics code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile, and ambient medium, it
integrates an expanding feedback bubble — shell radius, velocity, thermal state, force budget —
through its phase transitions and stopping fate. Pure Python (≥3.9), no compile step.

This file holds the project facts to load every session, then the behavioral rules.

## Commands

- Install (dev): `pip install -e ".[dev]"`  — core only: `pip install -r requirements.txt`
- Run one simulation: `python run.py param/simple_cluster.param`
- Parameter sweep (auto-detected from list/tuple syntax in the `.param`):
    - `python run.py param/sweep_example.param --dry-run`    # list combinations, run nothing
    - `python run.py param/sweep_example.param --workers 4`  # run across N workers
    - `python run.py param/sweep_example.param --emit-jobs jobs/`  then `sbatch` for SLURM/HPC
- Tests: `pytest`  (single: `pytest test/test_unit_conversions.py`; slow set: `pytest -m stress`)
- Lint/format: `pre-commit run --all-files` (ruff bug-class) · `black .` (line length 100) · `mypy trinity`
- Docs: `cd docs && make html`
- Paper figures: `python paper/methods/make_figures.py [name]`  (needs LaTeX — plots use `text.usetex`)

## Layout

```
run.py            single entry point — individual runs and sweeps (argparse)
trinity/          the package: solver, evolution phases, bubble/shell/cloud physics, I/O
  _input/         .param parsing, schema/defaults, sweep expansion (default.param lives here)
  _output/        run output, readers, terminal/metadata I/O, cloudy export
  _functions/     shared helpers (units, cluster, operations, logging)
  _analysis/      in-package analysis helpers
  bubble_structure/, cloud_properties/, cooling/   physics modules
param/            .param config files (the tracked ones are worked examples)
lib/default/      bundled SB99 SPS + cooling tables (quickstart runs out of the box)
test/             pytest suite (test_*.py)
tools/            small CLI utilities (param generation, audits, output comparisons)
docs/             Sphinx source (docs/source) built into docs/build
paper/            scripts + committed .npz data that regenerate published figures
```

Generated / scratch — not source, do not tidy or treat as ground truth: `outputs/`, `fig/`,
`scratch/`, `tbd/`, and `old_doNotRead/` (the name is the instruction).

## Conventions & gotchas

- **Do not bump `numpy` past 2.** It is pinned `<2` on purpose: numpy 2.x patches occasionally emit
  floating-point output the bubble-structure integrator's monotonic guard rejects (2.0/2.3 pass,
  2.1/2.2/2.4 fail). `scipy<2`, `astropy<8`, `matplotlib<4`, `pandas<3` are likewise capped.
- Configuration is `.param` files that override only the keys they set; everything else falls back
  to the schema defaults in `trinity/_input/`. Don't hardcode values that belong in a `.param`.
- Style is enforced by pre-commit (ruff `F821/F811/F823/E9`) and black — don't hand-format or
  reformat working code to satisfy style; run the tools. Don't widen the ruff rule set to "clean up."
- Units are a recurring bug class here. Match the surrounding module's unit conventions; see
  `trinity/_functions/unit_conversions.py` and `test/test_conventional_units.py`.

## Working rules

1. **Think first.** State assumptions. If multiple interpretations exist, surface them rather than
   picking silently. If a simpler approach exists, say so. If something is unclear, stop and ask.
2. **Simplicity.** Minimum code that solves the problem — no speculative features, abstractions,
   configurability, or error handling for impossible cases. If 200 lines could be 50, rewrite.
3. **Surgical changes.** Touch only what the request needs. Don't refactor or reformat working code;
   match existing style. Remove only the orphans *your* change creates; flag pre-existing dead code,
   don't delete it. Every changed line should trace to the request.
4. **Verify.** Turn tasks into checks: bug → write a failing test, then fix; feature → test the
   invalid inputs, then pass; refactor → tests green before and after. Run `pytest` before declaring done.

## `analysis/` & `docs/dev/` plan docs are unverified

The `analysis/*.md` files and the plan/skeleton docs under `docs/dev/` are point-in-time
audits/plans, not a maintained spec. They go stale fast — paths, line numbers, and "what shipped"
status drift as the code moves. When reading one: do not treat it as ground truth — flag that it may
be outdated and re-verify every claim, snippet, and line reference against current source. Every such
doc must carry this banner at the top, right under the H1, and any new analysis or `docs/dev/` plan
doc must include it:

```markdown
> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
```

---

Commits: no Claude session links, no "Generated by Claude", no co-author trailers.
