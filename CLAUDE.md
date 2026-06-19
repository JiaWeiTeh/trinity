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
  dev/            internal plan & audit write-ups (incl. CODEBASE_REVIEW.md) — not built, may be stale
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
- Tests and scratch configs use physically plausible values, not convenient round numbers — e.g.
  rCore ≈ 1 pc, realistic GMC masses/densities (and check `rCloud_max` plausibility validation
  passes). Unphysical inputs exercise regimes the code never runs in and hide real regressions.

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
5. **Equivalence depth & honest measurement** (hard-won — full rationale in
   `docs/dev/performance/BUBBLE_LUMINOSITY_PERFORMANCE.md` §Methodology). For a change to an
   **iterative/integrated path** (solver, residual, hot loop), a per-call/single-step equivalence is
   **necessary but NOT sufficient** — clear it only with a **full-run equivalence** test on the
   **stiffest/edge regimes** (not just the easy config), comparing full runs in **separate processes**
   (trinity leaks module-level global state in-process) at **matched simulation time** (runs truncate at
   different `t`). For a "free win", prove **bit-identical**: diff vs `git show HEAD` *and* a byte-identical
   `dictionary.jsonl`. **Measure, don't guess** — retract a hypothesis the moment data contradicts it.
   De-risk with a capture/replay **harness + explicit gates before** touching production, and **persist
   diagnostics** as committed CSVs/figures so a future session need not re-run the hours-long sims.

### Planning protocol — size the change first, then match the depth

Run rule 1 before touching code: state the change, your assumptions, and any competing
interpretations. Then size it — don't gate a typo, don't fast-path a solver edit.

- **Trivial / local** (typo, comment, isolated pure helper, new test): rules 1–4 — make it,
  add/adjust the check, `pytest`, done.
- **Risky / iterative / outward-facing** (anything in a solver/residual/hot loop, a cross-module
  refactor, a perf "win", a default flip, or anything pushed/published): run the ladder in order,
  and don't skip a rung because an earlier one passed —
  1. **Gate first.** Define what "equivalent" means and the pass/fail bar *before* editing
     (rule 5). A doc-worthy effort gets a `docs/dev/<workstream>/` writeup with the three banners.
  2. **Capture a baseline** the edit is measured against (a `git show HEAD` value, a byte hash, a
     saved trajectory) so "before" survives the change.
  3. **Equivalence gate.** Per-call / single-step first (cheap, necessary) — but **NOT sufficient**
     for an iterative path: clear it with a **full-run** equivalence on the **stiffest edge
     regimes**, in **separate processes**, at **matched `t`**. A "free win" ⇒ **bit-identical**
     (value-diff *and* byte-identical `dictionary.jsonl`).
  4. **Apply** the smallest diff that passes the gate (rules 2–3).
  5. **Re-verify** post-apply: the gate again + full `pytest` + ruff F-rules. Measure, don't guess.
  6. **Persist** diagnostics as a committed CSV/figure/harness (not `/tmp` or `scratch/`) with the
     exact config + command, so the next session compares without re-running.

  When unsure which depth applies, treat it as risky. Bubble/solver edges worth testing:
  `param/simple_cluster.param` (energy-driven baseline) + `docs/dev/performance/f1edge_{lowdens,hidens}*.param`
  (span feedback strength × cloud density) — plus the stiffest regime your change could plausibly break.

## Ponytail — lazy senior dev mode

Vendored from [ponytail](https://github.com/DietrichGebert/ponytail) (MIT), instruction-only mode —
committed here so it loads every session (Claude Code and Cowork) without a plugin install. Reinforces
the Working rules above; where it conflicts with project Conventions, the project wins (e.g. tests go
in the `pytest` suite, not ad-hoc self-checks).

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never
written. Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does the standard library already do this? Use it.
3. Does a native platform feature cover it? Use it.
4. Does an already-installed dependency solve it? Use it.
5. Can this be one line? Make it one line.
6. Only then: write the minimum code that works.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size — lazy means less
  code, not the flimsier algorithm.
- Mark intentional simplifications with a `ponytail:` comment. If the shortcut has a known ceiling
  (global lock, O(n²) scan, naive heuristic), the comment names the ceiling and the upgrade path.

Not lazy about: input validation at trust boundaries, error handling that prevents data loss,
security, accessibility, anything explicitly requested. Non-trivial logic leaves a runnable check
behind — the smallest thing that fails if the logic breaks (here, a `test_*.py` case in the `pytest`
suite). Trivial one-liners need no test.

## `docs/dev/` plan & audit docs are unverified

The plan, audit, and write-up docs under `docs/dev/` (the old top-level `analysis/` directory was
folded in here) are point-in-time audits/plans, not a maintained spec. They go stale fast — paths, line numbers, and "what shipped"
status drift as the code moves. When reading one: do not treat it as ground truth — flag that it may
be outdated and re-verify every claim, snippet, and line reference against current source. Every such
doc must carry **all three** banner paragraphs below at the top, right under the H1, and any new
`docs/dev/` doc must include them. The 🔄 paragraph makes these docs *living* —
whoever opens one rechecks it, updates drift, and rethinks the strategy before relying on or
extending it. The 💾 paragraph makes them *durable* — diagnostics worth keeping are committed as
CSV/tables (`docs/dev/data/`) or harnesses/figures in the relevant `docs/dev/<workstream>/` folder, so a future session reproduces or compares
without re-running the expensive sims; leave it better than you found it:

```markdown
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
```

---

Commits: no Claude session links, no "Generated by Claude", no co-author trailers.
