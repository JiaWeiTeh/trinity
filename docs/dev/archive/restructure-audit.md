# Repository restructure: audit + phased plan

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> **Audit status (2026-06-08):** **all three phases (A `_modified` drop, B
> `src/→trinity`, C plotting split) have shipped**, as has the "only remaining"
> integrator follow-up (#659) — this reads as a forward plan but describes
> completed work. The target-layout and design-decision records still hold; the
> blast-radius metrics and the "tracked `scratch/`" outcome do **not** (scratch
> was instead **deleted**, commit `b4e2996`).

Single source of truth for three structural changes raised in the
codebase review, in the same shape as `docs/dev/sb99-refactor-audit.md`
and `docs/dev/cooling-refactor-audit.md`: (Part I) the audit — *what is*
— with measured blast radii, and (Part II) the phased plan and its
equivalence-test battery — *what to do, in what order, and how to prove
nothing changed*.

End goal:

1. Drop the vestigial `_modified` suffix from the 8 physics modules.
2. Rename the importable package `src/` → `trinity/` so the install
   name (`trinity-sf`), the import name, and the brand agree.
3. Separate the 23k-line plotting tree by **audience/lifecycle**: the
   engine (installed), the `make_figures.py` reproduction closure (public
   deliverable, 5 scripts + infra), and 31 personal scripts (tracked
   `scratch/`, quarantined out of `trinity/` and `paper/`).

All three are **behavior-preserving**. Nothing in this plan changes a
single physics result; the verification battery is "the same 358 tests
plus the smoke run still pass, byte-for-byte where applicable."

## TL;DR

- **`_modified` rename is trivial and isolated**: 17 import lines across
  8 files, 3 doc mentions, 3 `.param` *comments*. The original
  non-`_modified` files **do not exist** — confirmed — so the suffix is
  purely vestigial. ~15 min, low risk. Do first.
- **`src` → `trinity` is mechanical but wide**: 105 files, 444 import
  lines, plus `pyproject.toml`, 5 `.rst` docs, and `src/__init__.py`
  examples. A single find/replace + `git mv` + a 2-line pyproject edit,
  fully covered by the test suite. ~30 min, low risk, large diff. Do second.
- **Plot split is the only non-trivial one**: `_plots/` is **47 files /
  23,634 lines = ~43% of all repo Python**. The dependency graph is
  *already clean and one-way* (the engine does not import plots), so this
  is a packaging/audience problem, not a coupling problem. Riskier
  because plots are essentially untested. Do last, incrementally.
- **Do not merge everything public into one folder.** `lib/default/` is
  package-data and must ship with the install; `param/` is user-facing
  examples; `paper/` is the reproduction deliverable. Group by role.

---

# Part I — Audit (what is)

## I.1 The `_modified` suffix

Eight modules carry a `_modified` suffix that is relative to originals
that no longer exist:

| File | Lines | Referencing files |
|------|-------|-------------------|
| `phase1_energy/run_energy_phase_modified.py` | 409 | 1 |
| `phase1_energy/energy_phase_ODEs_modified.py` | — | 4 |
| `phase1b_energy_implicit/run_energy_implicit_phase_modified.py` | 1034 | 1 |
| `phase1b_energy_implicit/get_betadelta_modified.py` | 808 | 2 |
| `phase1c_transition/run_transition_phase_modified.py` | 880 | 1 |
| `phase2_momentum/run_momentum_phase_modified.py` | 923 | 1 |
| `bubble_structure/bubble_luminosity_modified.py` | 832 | 2 |
| `shell_structure/shell_structure_modified.py` | — | 7 |

- **Verified**: for every `*_modified.py`, the sibling `*.py` (e.g.
  `get_betadelta.py`, `bubble_luminosity.py`) **does not exist**. The
  suffix is dead nomenclature, not a marker of parallel versions.
- Total code references: **17 import lines**.
- Non-code references: 3 docs (`docs/source/running.rst`,
  `docs/source/parameters.rst`, `docs/dev/TERMINATION_EVENTS.md`) and 3
  `.param` *comments* (`# use _modified files with solvers` in
  `trinity_fiducial.param`, `trinity_fiducial_yesno.param`,
  `sweep_orion_better_paper.param`) — comments only, no behavior.
- Several docstrings still describe themselves as "improvements over
  `get_betadelta.py`" etc. — references to deleted files.

## I.2 The `src` package name

- `src/__init__.py` is a real package root (`__version__`, docstring with
  `from src._input import read_param` usage examples).
- Blast radius: **105 files** reference `src` as a package across **444
  import lines**.
- Non-code references: `pyproject.toml`
  (`[tool.setuptools.packages.find] include = ["src*"]`, `package-dir`),
  and 5 docs: `architecture.rst`, `running.rst`, `visualization.rst`,
  `parameters.rst`, `trinity_reader.rst`.
- Why it matters: the distribution is named `trinity-sf` but the import
  package is literally `src`. A `pip install` exposes a top-level module
  named `src`, which collides with every other src-layout project and
  reads wrong (`from src._input import ...`).

## I.3 The plotting tree

- **Size**: `src/_plots/` = 47 files, **23,634 lines (~43% of the 54.7k
  Python lines in the repo)**.
- **Dependency direction is clean and one-way**:
  - The engine does **not** import plots. The only physical link is
    `src/_functions/plot_style.py` loading `../_plots/trinity.mplstyle`
    (an asset path), plus two stale *comments* in `show_run.py` and
    `trinity_reader.py`.
  - Plots are downstream consumers of `_output/trinity_reader.py`,
    `_functions/unit_conversions.py`, `_functions/simplify.py`,
    `_output/simulation_end.py`, and a couple of `cloud_properties`
    modules.
  - The plot **infrastructure** (`cli.py`, `plot_base.py`,
    `plot_markers.py`, `force_colors.py`, `grid_template.py`) is imported
    **only by other `_plots` files** — fully self-contained.
- **Composition**: 34 `paper_*`, 2 `pedrini_*`, 2 `diag*`
  (`diag_simplify`, `diagnostic_parameter_changes`), `compare_outputs`,
  plus infra + `trinity.mplstyle`.
- **Existing `paper/` skeleton already present**: `paper/make_figures.py`
  and `paper/data/*.npz` (diagnostics, densityProfile, app_LSODA,
  radiusComparison). The natural home for figure scripts already exists.
- **`lib/default/`**: 13 MB, 29 files, shipped as package-data
  (`[tool.setuptools.package-data]`). Must remain bundled with the
  install — do not relocate into a docs pile.

### Coupling landmines for the plot split

1. **One test imports a plot**: `test/test_phase4_consumer_migration.py`
   imports `pedrini_emergence_timescales`. Its import path changes if
   plots move.
2. **`plot_style.py` appears imported by nobody** (grep finds only its
   own definition). Either dead, or styling is actually applied via
   `plot_base`. Confirm before relocating `trinity.mplstyle`.
3. **34 `paper_*` scripts use intra-package imports**
   (`from src._plots.cli import ...`). Moving out of the package converts
   them to a standalone script tree importing the *installed* `trinity`
   package.
4. **Plots are essentially untested** (only the one import above), so the
   move cannot be fully auto-verified — needs a manual spot-run of
   `run_all.py` against a sample output directory.

---

# Part II — Phased plan

Each phase is an independent, reviewable commit (one logical change, per
`CONTRIBUTING.md`). Use `git mv` everywhere so history follows the files.
Run the full battery (below) after each phase.

## Phase A — Drop the `_modified` suffix  *(smallest, do first)*

1. `git mv` each of the 8 `*_modified.py` → `*.py` (e.g.
   `get_betadelta_modified.py` → `get_betadelta.py`).
2. Update the **17 import lines** at their call sites.
3. Update the 3 doc mentions and 3 `.param` comments.
4. Fix self-referential docstrings ("improvements over `get_betadelta.py`")
   to describe what the module *is*, not what it superseded.
5. **No behavior change.** Verify: full battery.

Risk: **low**. Isolated, mechanical, fully test-covered.

## Phase B — Rename `src/` → `trinity/`  *(mechanical, wide)*

1. `git mv src trinity`.
2. Find/replace across the repo: `from src.` → `from trinity.`,
   `from src ` → `from trinity `, `import src.` → `import trinity.`,
   `import src\b` → `import trinity`.
3. `pyproject.toml`: `include = ["src*"]` → `["trinity*"]`; confirm
   `package-dir` mapping. (`packages.find where = ["."]` stays.)
4. Update the 5 `.rst` docs and the `src/__init__.py` usage examples.
5. Sweep for string-literal references to `"src/..."` paths (e.g. the
   `plot_style.py` mplstyle path, any `__file__`-relative joins).
6. **No behavior change.** Verify: full battery + a fresh
   `pip install -e .` import smoke (`python -c "import trinity"`).

Risk: **low**, but the diff is large (~444 lines). Keep it a *pure*
rename commit — no opportunistic edits — so review and `git blame` stay
clean. Flag to the user: any private notebooks/scripts outside the repo
that do `from src...` will break and must be updated by hand.

Decision to confirm with the user before starting Phase B:
- **Minimal** (recommended): `src/` → `trinity/` at repo root.
- **src-layout proper**: `src/trinity/`. More idiomatic packaging, larger
  move, more churn. Probably overkill here.

## Phase C — Separate the plotting tree by role  *(largest, do last, incrementally)*

> **Restyled per the audience model — see Appendix D (authoritative).**
> Public = the *closure of `paper/make_figures.py`* (5 scripts + 5 infra),
> **not** all 34 `paper_*`. Everything else (31 scripts) is personal and
> goes to a **tracked** `scratch/` (gitignored would not survive this
> ephemeral container). The sketch below is updated to match.

Target layout:

```
trinity/            # engine (installed); lib/default stays as package-data
param/              # example .param inputs (public, user-facing)
paper/              # the reproduction DELIVERABLE
  make_figures.py   #   (already present)
  data/             #   (already present)
  figures/          #   <- the 5 closure scripts (paper_densityProfile,
                    #      paper_teaser, paper_radiusComparison,
                    #      paper_rcloud_smoothing, paper_feedback)
    _lib/           #   <- cli, plot_base, plot_markers, force_colors,
                    #      grid_template, trinity.mplstyle
scratch/            # TRACKED: the 31 personal paper_*/pedrini_*/diag scripts
tools/              # compare_outputs.py (user-facing run-diff)
```

**The step-by-step is now in Appendix D (D.5), restyled around the
`make_figures.py` closure and a tracked `scratch/`.** In brief:

- **C.1** — carve a *tracked* `scratch/`; move the 31 personal scripts
  (29 `paper_*` + 2 `pedrini_*` + `diag_simplify`) there. Safe stop-point.
- **C.2** — move the 5 closure scripts → `paper/figures/`, the 5 infra +
  `trinity.mplstyle` → `paper/figures/_lib/`.
- **C.3** — rewrite imports + `__file__`-depth shims (D.4); repoint
  `make_figures.py`'s `python -m` targets; best-effort-rewrite the personal
  scripts.
- **C.4** — drop `TestPedriniEmergenceMigration`; `compare_outputs.py` →
  `tools/`; delete dead `radial_profile.py` + `plot_style.py`; remove the
  emptied `trinity/_plots/`.
- **C.5** — `pyproject.toml` wheel-excludes + docs.

Risk: **medium**. Untested surface, real import/depth rewrites, plus a
best-effort rewrite of 31 personal scripts. C.1 alone is a safe partial
landing.

---

# Verification battery (every phase)

1. `python -m pytest test/ -q` → **358 passed** (same as baseline).
2. `python run.py param/simple_cluster.param` (or the smoke param) exits 0
   and writes `metadata.json` + `dictionary.jsonl`.
3. `pre-commit run --all-files` clean (ruff F821/F811/F823/E9 — the F821
   undefined-name check is exactly what catches a half-finished rename).
4. After Phase B only: `pip install -e . && python -c "import trinity"`.
5. After Phase C only: manual `run_all.py` render spot-check.

# Why not one big PR

Three independent commits/PRs, smallest-blast-radius first:
**A (_modified) → B (src→trinity) → C (plots)**. Phase A and B are pure
renames with a full test net; Phase C is the only one with an untested
surface, so it goes last and can be split (C.1 then C.2) or deferred
without blocking A and B.

**A separate, behavior-AFFECTING follow-up — sequenced dead last —** is the
flaky `MonotonicError` robustness fix. It is deliberately *not* part of the
A–C restructure (those are byte-preserving; this one changes a runtime code
path), so it lives in its own doc, `docs/dev/bubble-integrator-robustness.md`,
and must land after the structural churn settles. Status: A/B/C all
executed; integrator: planned (separate doc) — the only remaining phase.

# Part III — Final target layout & conventions

The guiding rule (the convention to enshrine): **organize by
audience/lifecycle with a strict one-way dependency.** Everything may
import the installed `trinity` package; the engine imports nothing
downstream (`paper/`, `examples/`, `scratch/`, `tools/`).

```
trinity-repo/
├── trinity/               # ENGINE ONLY (post src-> rename): pure working code,
│                          #   no data, no paper code — kept clean by design
├── tools/                 # maintained utilities: gen_default_param.py, compare_outputs.py
├── paper/                 # reproducibility bundle (public)
│   ├── data/
│   ├── figures/           # the 5 make_figures.py closure scripts only
│   │   └── _lib/          # plot_base, plot_markers, cli, force_colors,
│   │                      #   grid_template, trinity.mplstyle
│   └── make_figures.py
├── lib/default/           # bundled tables, top-level by design (keeps trinity/ pure code)
├── param/                 # canonical parameter-file library
├── examples/              # runnable getting-started scripts (reference param/)
├── scratch/               # TRACKED personal scripts (31 paper_*/pedrini_*/diag)
├── docs/dev/  docs/  tests/
├── pyproject.toml  README.md     # run.py -> console entry point
```

## III.1 What stays in the engine vs. moves out (the audience test)

Two criteria, applied separately (see Appendix D.0):
- **Scripts** split by the **`make_figures.py` closure**: in it → public
  `paper/figures/`; not in it → personal `scratch/` (31 files).
- **Infra** below moves to the public `_lib/` because the closure scripts
  import it. The engine keeps none of it:

| File | Destination | Rationale |
|------|-------------|-----------|
| `plot_base.py` | `paper/figures/_lib/` | docstring: "for paper scripts"; hardcodes `parent.parent.parent` + `fig/` (breaks if moved up); not a public API |
| `plot_markers.py` | `paper/figures/_lib/` | used only by `paper_*` |
| `cli.py` | `paper/figures/_lib/` | docstring: "CLI builder for `_plots` paper scripts" |
| `force_colors.py`, `grid_template.py` | `paper/figures/_lib/` | paper-only |
| `trinity.mplstyle` | `paper/figures/_lib/` | only paper infra loads it |
| `compare_outputs.py` | `tools/` (or `trinity/viz`) | genuinely user-facing: compares two runs via the public `dictionary.jsonl`; run via `python -m` |

Outcome: `trinity/_plots/` ends up empty — delete the folder rather than
leave it named after the thing removed. The engine's visualization surface
should be defined by the public output object (`TrinityOutput`), never by
figure scripts.

## III.2 Directory conventions

- **`param/` vs `examples/`**: `param/` is the canonical parameter-file
  library; `examples/` holds narrative getting-started scripts that
  *reference* `param/` files (no duplicated `.param` content).
- **`scratch/` not `notebooks/`**: reserve `notebooks/` for committed
  tutorials if ever wanted. **In this effort `scratch/` is TRACKED**
  (Appendix D.3): the personal scripts are real work that must survive an
  ephemeral container, so they are version-controlled but quarantined out
  of `trinity/` and `paper/`. (A truly gitignored sandbox is a later option
  on a persistent checkout.) `diag_simplify.py` → `scratch/`.
- **`tests/` not `test/`**: plural is the prevailing pytest convention;
  trivial, only worth folding into the rename pass.

## III.3 Decided: bundled data stays top-level

`lib/default/` **stays a top-level sibling of `trinity/`** — deliberately
*not* moved inside the package. The intent is to keep `trinity/` pure
internal working code (no data, no scripts, no figures), so the package
tree reads as "the engine and nothing else." The bundled tables continue
to ship via the existing `[tool.setuptools.package-data]` glob in
`pyproject.toml`.

Trade-off accepted: this keeps the top-level `package-data` glob rather
than the `importlib.resources` + in-package-data pattern used by
Astropy/scikit-learn. That pattern is more relocatable, but it pulls data
into the package, which conflicts with the "trinity/ is pure code" goal.
The glob already works and the tests pin it, so no change here.

## III.4 Optional follow-up (separate effort, not part of A–C)

- **Console entry point**: add
  `[project.scripts] trinity = "trinity.cli:main"` so users run
  `trinity param/...` instead of `python run.py`; `run.py` becomes a thin
  shim.

## III.5 Enforce the invariant (what makes it stick)

Add a CI guard so the structure does not drift back: a test (or an
`import-linter` contract) asserting the engine package never imports
`paper`, `examples`, or `scratch`. Without enforcement the separation
erodes within a few months.

# Decisions (resolved)

- **Phase B layout**: ~~minimal `trinity/` at root vs. `src/trinity/`?~~
  **root-level `trinity/`.** *(Executed.)*
- **Phase C public surface**: ~~all `paper_*` vs. a subset?~~ **the
  `make_figures.py` closure only** (5 scripts + 5 infra); the other 31
  scripts are personal. *(Appendix D.0.)*
- **Phase C package shape**: ~~importable vs. flat?~~ **importable
  `paper/figures/` package** (`__init__.py` in `figures/` + `_lib/`).
- **`scratch/`**: ~~gitignored?~~ **TRACKED** — ephemeral container would
  lose a gitignored sandbox. *(Appendix D.3.)*
- **pedrini test coupling**: **drop `TestPedriniEmergenceMigration`**
  (it pins a 2-line wrapper over the engine-tested `read_simulation_end()`).
- **Personal-script imports**: **rewrite best-effort** so they still run
  from `scratch/` (no test harness covers them).
- **`plot_style.py` / `radial_profile.py`**: both **dead (0 importers) →
  delete** (re-confirm at execution).

---

# Appendix A — Phase A execution detail (drop `_modified`)

> **STATUS: EXECUTED.** All 8 `*_modified.py` files renamed, the 17 import
> sites + docstrings/docs/param-comments updated. Verified: full battery
> green. (Committed on `feature/reforming-structure`.)

Verified against the real tree on branch `feature/reforming-structure`.
Pure, behavior-preserving rename. Empty `__init__.py` in all six affected
packages — no re-exports — so only the explicit import sites below matter.
All 8 target names are collision-free (no surviving `*.py` sibling).

## A.1 File renames (`git mv`, 8)

| From | To |
|------|----|
| `src/bubble_structure/bubble_luminosity_modified.py` | `bubble_luminosity.py` |
| `src/phase1_energy/energy_phase_ODEs_modified.py` | `energy_phase_ODEs.py` |
| `src/phase1_energy/run_energy_phase_modified.py` | `run_energy_phase.py` |
| `src/phase1b_energy_implicit/get_betadelta_modified.py` | `get_betadelta.py` |
| `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py` | `run_energy_implicit_phase.py` |
| `src/phase1c_transition/run_transition_phase_modified.py` | `run_transition_phase.py` |
| `src/phase2_momentum/run_momentum_phase_modified.py` | `run_momentum_phase.py` |
| `src/shell_structure/shell_structure_modified.py` | `shell_structure.py` |

## A.2 Import-site edits (mandatory — these are what `pytest`/F821 verify)

**`src/main.py`** — 4 imports + 4 call sites:
- L23 `from src.phase1_energy import run_energy_phase_modified` → `run_energy_phase`; L248 call `run_energy_phase_modified.run_energy(...)` → `run_energy.run_energy(...)`
- L24 `... import run_energy_implicit_phase_modified` → `run_energy_implicit_phase`; L283 call
- L25 `... import run_transition_phase_modified` → `run_transition_phase`; L303 call
- L26 `... import run_momentum_phase_modified` → `run_momentum_phase`; L343 call

**`src/phase1_energy/run_energy_phase_modified.py`** (renamed) — 3 `import … as …_modified` aliases + their call sites; drop `_modified` from both alias and usages:
- L30 `import src.shell_structure.shell_structure_modified as shell_structure_modified` → `import src.shell_structure.shell_structure as shell_structure`; usages L188, L334, L373
- L32 `import src.phase1_energy.energy_phase_ODEs_modified as energy_phase_ODEs_modified` → `… energy_phase_ODEs as energy_phase_ODEs`; usages L209, L210, L246, L251, L375, L378
- L33 `import src.bubble_structure.bubble_luminosity_modified as bubble_luminosity_modified` → `… bubble_luminosity as bubble_luminosity`; usage L164

**`src/phase1b_energy_implicit/get_betadelta_modified.py`** (renamed):
- L28 `from src.bubble_structure.bubble_luminosity_modified import (` → `bubble_luminosity`

**`src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py`** (renamed):
- L72 `from src.phase1_energy.energy_phase_ODEs_modified import (` → `energy_phase_ODEs`
- L79 `from src.phase1b_energy_implicit.get_betadelta_modified import (` → `get_betadelta`
- L86 `from src.shell_structure.shell_structure_modified import (` → `shell_structure`

**`src/phase1c_transition/run_transition_phase_modified.py`** (renamed):
- L61 `from src.phase1_energy.energy_phase_ODEs_modified import (` → `energy_phase_ODEs`
- L66 `from src.phase1b_energy_implicit.get_betadelta_modified import (` → `get_betadelta`
- L69 `from src.shell_structure.shell_structure_modified import (` → `shell_structure`

**`src/phase2_momentum/run_momentum_phase_modified.py`** (renamed):
- L62 `from src.shell_structure.shell_structure_modified import (` → `shell_structure`

## A.3 Docstring / comment references to module names or deleted originals (mandatory for accuracy)

These name a `*_modified` module or a deleted original; they become wrong after the rename:
- `get_betadelta(_modified).py` L9 "Key improvements over get_betadelta.py:" (now self-referential — rewrite to describe what it *is*), L14 "Reuses … from bubble_luminosity_modified"
- `run_energy_phase(_modified).py` L7 "replacing the manual Euler integration in run_energy_phase.py", L9 "Key differences from run_energy_phase.py:" (self-referential), L13 "Uses dataclass returns from bubble_luminosity_modified"
- `run_energy_implicit_phase(_modified).py` L33 "The beta-delta solver (get_betadelta_modified.py) uses:"

In other engine files (comments / one f-string that prints the filename):
- `src/_analysis/check_yesno.py` L13 (comment) and **L220 (f-string output)** reference `energy_phase_ODEs_modified.py:253-256` → `energy_phase_ODEs.py:253-256` (line numbers unchanged by rename)
- `src/_plots/pedrini_emergence_timescales.py` L236 comment `shell_structure_modified.py:412` → `shell_structure.py:412`
- `src/_plots/paper_teaser.py` L341 comment `shell_structure_modified.py:411` → `shell_structure.py:411`
- `src/_plots/paper_phii_window.py` L19 docstring `shell_structure_modified` → `shell_structure`

## A.4 Cosmetic (recommended, not required for correctness)

The literal word "Modified" / "modified" appears in titles and log
strings that do **not** name a file and do not affect behavior. Clean for
consistency or leave:
- Head-docstring titles: `bubble_luminosity` L4, `energy_phase_ODEs` L4, `run_energy_phase` L4, `get_betadelta` L4, `run_energy_implicit_phase` L4, `run_transition_phase` L4, `run_momentum_phase` L4, `shell_structure` L4 ("Modified … with …").
- Log strings in `run_energy_phase`: L75 "Starting modified energy phase…", L105 "Energy phase initialization (modified):".
- Comments "Import pure/modified functions" (`run_energy_implicit_phase` L71, `run_transition_phase` L60, `run_momentum_phase` L61).
- **Do NOT touch** `get_betadelta` L360 "Extract needed values from original params (not modified)" — "modified" here means "not mutated", unrelated to the suffix.

## A.5 Docs

- `docs/dev/TERMINATION_EVENTS.md` L20, L35, L52, L68 — section headers naming the four phase-runner files → drop `_modified`.
- `docs/source/parameters.rst` L544 — **needs a real rewrite, not a mechanical rename.** It currently reads: "Use the adaptive ODE solver for the energy-driven phase (`run_energy_phase_modified.py`). If `False`, falls back to the original solver (`run_energy_phase.py`)." But `use_adaptive_solver` is registered as **deprecated and never consumed** (`registry.py:301`: *"changing this flag has no effect"*), and the "original" `run_energy_phase.py` does not exist. Proposed replacement aligned with reality: *"Deprecated. Parsed for backward compatibility but not consumed by any code path — the adaptive solver in `run_energy_phase.py` is the only implementation, so this flag has no effect."*

## A.6 Out of scope / do NOT touch

- `docs/source/running.rst` L176 — `_modified` here is an **output-filename** suffix sibling to `_summary.txt` (255-byte cap), unrelated to these modules.
- `docs/dev/cooling-refactor-audit.md` and `docs/dev/sb99-refactor-audit.md` — multiple `*_modified.py` mentions are **historical point-in-time records** of completed refactors. Recommendation: leave as-is (they document what existed then). Optional: a one-line note that files were later renamed in Phase A.

## A.7 Judgment calls (need a decision before executing)

1. **Param comments (3):** `param/trinity_fiducial.param` L39, `param/trinity_fiducial_yesno.param` L40, `param/sweep_orion_better_paper.param` L38 — the comment `# use _modified files with solvers` sits above a `use_adaptive_solver True` line. The comment references both the renamed files *and* a deprecated no-op param. Options: (a) delete the comment line (recommended — it is doubly stale), (b) reword, (c) leave. These are example input files, so editing them is a content change, not a rename.
2. **`use_adaptive_solver` key itself:** out of scope for Phase A (it is a deprecated-param cleanup, not a rename). The 13+ param files that set it are untouched.

## A.8 Execution order

1. `git mv` all 8 files (A.1).
2. Apply import edits (A.2) — start with `main.py`, then each renamed runner's internal imports.
3. Apply docstring/comment accuracy edits (A.3) + docs (A.5) + cosmetic (A.4 if chosen).
4. Resolve the A.7 param-comment decision.

## A.9 Verification battery

1. `grep -rn "_modified" --include=*.py src run.py` → **only** intentional hits (expected: none; `check_yesno.py`/`_plots` comments fixed in A.3). A non-empty result means a missed reference.
2. `pre-commit run --all-files` → clean; **F821 (undefined name)** is the precise guard against a half-renamed import.
3. `python -m pytest test/ -q` → **358 passed** (unchanged).
4. `python run.py param/simple_cluster.param` exits 0 and writes `metadata.json` + `dictionary.jsonl`.

## A.10 Rollback

Single isolated commit; revert restores the prior state. No data, schema,
or output-format changes — `dictionary.jsonl`/`metadata.json` are
byte-unaffected.

---

# Appendix B — Deprecated-param removal plan

> **STATUS: EXECUTED via Option A (hard break).** The 4 specs were removed
> from the registry; a `.param` still setting one now errors. The empty
> `deprecated` category machinery was kept for future use (the
> `test_deprecated_category_requires_note` probe still guards it), so the
> `RETIRED_KEYS` shim of Option B (§B.3) was not added. Mock outputs (§B.4)
> were left untouched.

Verified against the tree on `feature/reforming-structure`. Four params
carry `category='deprecated'` and are **never consumed** by any code path
(confirmed by grep across `src/` + `run.py`):

| Param | `default.param` | shipped `param/*.param` | tests | docs | mock outputs |
|-------|:---:|:---:|:---:|:---:|:---:|
| `stop_v` | L281 | — | metadata L84, registry L317 | rst L281 | summary.txt L26 |
| `adiabaticOnlyInCore` | L284 | — | metadata L89, registry L317 | rst L532 | summary.txt L54 |
| `immediate_leak` | L287 | — | metadata L90, registry L317 | rst L535 | summary.txt L55 |
| `use_adaptive_solver` | L290 | **16 files** | metadata L88, registry L317 | rst L541 | param L12 + summary L57 |

Only `use_adaptive_solver` appears in user-facing `param/*.param` (16
files). The other three live only in generated/test/doc/mock locations.

## B.1 The blocking constraint (drives the whole approach)

`read_param.py:214-225` rejects any key not in the registry with a hard
`ParameterFileError: Invalid parameter(s)`. So deleting a spec makes every
`.param` that still sets it **fail to load** — exactly the back-compat the
`deprecated` category exists to provide. Two ways forward:

- **Option A — hard break.** Delete the 4 specs; scrub every in-repo
  reference. Any *external* `.param` still setting one now errors out.
  Simplest, smallest diff, but a breaking change for a "Production/Stable"
  release.
- **Option B — graceful retirement (recommended).** Replace the four
  full `ParamSpec` entries with a single `RETIRED_KEYS` set that
  `read_param` strips (with a one-time `logger.warning`) *before* the
  unknown-key check. Old/external files keep loading; the warning tells
  users to delete the now-ignored line. This is the correct next step in
  the deprecation lifecycle (deprecated-in-schema → retired-and-ignored)
  and is non-breaking.

Both options scrub the 16 shipped `param/*.param` files (leaving a retired
key would just emit warnings on every run). Option B additionally shields
external files.

## B.2 Edits — common to both options

1. **`src/_input/registry.py`** — delete the 4 `ParamSpec` lines
   (L271, L298, L299, L301).
2. **`src/_input/param_spec.py`** — the `CATEGORIES` list (L48-61):
   remove the `"deprecated"` entry and its comment (no specs will carry
   it). *Check*: `gen_default_param.is_file_backed` and
   `test_only_file_backed_specs_emitted` reference the literal
   `"deprecated"`; those clauses become dead — simplify them (drop the
   `or cat == "deprecated"`).
3. **`src/_input/default.param`** — regenerate, do not hand-edit:
   `python -m tools.gen_default_param --write`. The 4 lines disappear once
   the specs are gone (the file is a generated artifact;
   `test_default_param_matches` enforces byte-equality with `render(SPECS)`).
4. **Shipped param files** — remove the `use_adaptive_solver` line from
   the 16 `param/*.param` files (grep-verified list).
5. **Docs** — `docs/source/parameters.rst`: delete the four table rows
   (L281 `stop_v`, L532 `adiabaticOnlyInCore`, L535 `immediate_leak`,
   L541 `use_adaptive_solver`, each a `* - ``name``` row + its value/desc
   lines).
6. **Tests**:
   - `test/test_registry.py` L313 `test_deprecated_specs_have_notes` —
     asserts the deprecated set is exactly those 4. Remove this test
     (no deprecated specs remain) **or**, under Option B, repurpose it to
     assert `RETIRED_KEYS` is non-empty and disjoint from live spec names.
   - `test/test_gen_default_param.py` L100 `test_deprecated_text_matches`
     — becomes vacuous (no deprecated keys); remove it.
     `test_only_file_backed_specs_emitted` (L112) — drop the
     `or cat == "deprecated"` clause to match B.2-step-2.
   - `test/test_metadata.py` L84/L88/L89/L90 — remove the four keys from
     the `_scalars` fixture (the fixture is a hand-built `DescribedDict`,
     not registry-validated, but these keys should no longer be presented
     as live `RUN_CONST_KEYS`).

## B.3 Edits — Option B only

7. **`src/_input/read_param.py`** — add near the top:
   ```python
   # Keys retired from the schema but still tolerated in .param files so
   # pre-existing inputs don't hard-error. Stripped (with a warning) before
   # the unknown-key validation below.
   RETIRED_KEYS = {"stop_v", "adiabaticOnlyInCore",
                   "immediate_leak", "use_adaptive_solver"}
   ```
   and, just before the `invalid_keys` check (~L214):
   ```python
   for key in list(user_dict):
       if key in RETIRED_KEYS:
           logger.warning("Parameter '%s' is retired and ignored; "
                          "remove it from your .param file.", key)
           del user_dict[key]
   ```
   Add one test (e.g. in `test_read_param` / `test_registry`) asserting a
   `.param` containing a retired key loads without error and the key is
   absent from the merged dict.

## B.4 Out of scope / leave as-is

- **Mock outputs** (`outputs/mockOutput/mockFullrun/4e3_sfe001_n5e2_PL0.param`
  and `_summary.txt`): these are frozen artifacts of a historical run and
  are **not** routed through `read_param` by any test (the cloudy tests
  parse `dictionary.jsonl` + `_summary.txt` only — verified). Leave them as
  a faithful record. (Under Option A they'd still be inert; under Option B
  they'd load fine anyway.)

## B.5 Verification battery

1. `python -m tools.gen_default_param --check` → exit 0 (committed
   `default.param` matches `render(SPECS)`).
2. `grep -rn "stop_v\|adiabaticOnlyInCore\|immediate_leak\|use_adaptive_solver" src param docs --include=*` → only the
   intended residue (Option B: the `RETIRED_KEYS` set + its test; Option A:
   nothing).
3. `pre-commit run --all-files` → clean.
4. `python -m pytest test/ -q` → all pass (the cloudy/mock tests must stay
   green, proving the mock files were safe to leave).
5. `python run.py param/trinity_fiducial.param`-style smoke: a param file
   that *used to* set `use_adaptive_solver` now loads cleanly (Option B:
   also test a file that still sets it → loads with a warning).

## B.6 Rollback

Single commit; revert restores the specs and the schema. No physics, no
output-format change. `default.param` is regenerated, not hand-edited, so
the codegen gate guarantees consistency.

---

# Appendix C — Phase B execution detail (`src/` → `trinity/`)

> **STATUS: EXECUTED via root-level `trinity/` (recommended option); `test/`
> → `tests/` deferred.** `git mv src trinity` + an anchored find/replace
> (every `src.`/`src/` replacement anchored on the known sub-package names,
> so the local `src` loop/param variables in `test_cloudy_cli.py` and
> `trinity_to_cloudy.py` were left untouched). 156 files, 127 renames,
> 621/621 balanced line swaps. `default.param` regenerated. Verified:
> `pip install -e . && import trinity` OK, `gen_default_param --check` in
> sync, ruff clean, **356/357 tests pass** — the lone failure is the
> *pre-existing* flaky `MonotonicError` in the bubble integrator (numpy
> 1.26.4, passed 2 of 3 identical re-runs; unrelated to the rename). That
> flake is the subject of the final phase — see
> `docs/dev/bubble-integrator-robustness.md`.

Measured on `feature/reforming-structure` after Phase A + the
deprecated-param removal. Pure, behavior-preserving package rename.

## C.1 Layout decision (recommended: minimal `trinity/` at repo root)

`git mv src trinity` at the **repo root**, preserving directory depth.
This matters: 100+ call sites compute the repo root via `__file__`-relative
depth — `Path(__file__).resolve().parents[2]`,
`parent.parent.parent` (`show_run.py:41`, `plot_base.py:23`,
`cli.py:42`, all `paper_*` shims), and `parents[3]`
(`trinity_to_cloudy.py:45`). A root-level `trinity/` keeps every such
shim valid **unchanged**. A `src/trinity/` layout would add one level and
break all of them. → **Recommend root-level `trinity/`.**

`test/` → `tests/` is intentionally **out of scope** here (separate
trivial change: rename dir + `pyproject.toml testpaths`). Keep Phase B a
pure package rename.

## C.2 Blast radius

- **103 files / 446 import lines**: `from src…` / `import src…`.
- **84 additional `.py` lines** with *non-import* package references
  (docstrings, `python -m src._output.show_run` usage examples,
  `:mod:`src._…`` / ``` ``src._…`` ``` cross-refs, `python src/_plots/…`
  invocations). These must change too or the docs/examples go stale.
- **2 path-string literals** (not import lines, easy to miss):
  - `test/test_registry.py:71` — `… / "src" / "_input" / "default.param"`
  - `tools/gen_default_param.py:39` — same pattern
- **`pyproject.toml:74`** — `include = ["src*"]` → `["trinity*"]`
  (`package-dir = {"" = "."}` and `where = ["."]` stay; `pythonpath=["."]`
  and `testpaths=["test"]` stay).
- **`src/__init__.py:8,9,15`** — usage-example docstrings
  (`from src._input import read_param`, …).
- **Docs**: `architecture.rst`, `visualization.rst`, `running.rst`,
  `parameters.rst`, `trinity_reader.rst`, `docs/dev/TERMINATION_EVENTS.md`.

## C.3 Transformation recipe (targeted — avoids the two traps)

Apply over tracked `.py` + `.rst` + `.md` + `pyproject.toml`, **excluding**
the traps in C.4:

1. `from src.`  → `from trinity.`
2. `from src import` → `from trinity import`
3. `import src.` → `import trinity.`
4. `import src` (eol / `as`) → `import trinity`
5. `src._`  → `trinity._`   (covers `:mod:`, backticks, `python -m`)
6. `src/_`  → `trinity/_`    (covers `python src/_plots/…` and path strings)
7. The 2 path-literals in C.2: `"src"` → `"trinity"` (those two lines only)
8. `pyproject.toml`: `["src*"]` → `["trinity*"]`

Then `git mv src trinity`.

## C.4 Do NOT touch (blind-replace traps)

- `test/test_cloudy_cli.py:273` — `for src in MOCK_FULLRUN.iterdir():`
  (`src` is a **loop variable**, not the package).
- `README.md:5` — `<img src="…badge…">` (HTML attribute).

## C.5 Verification battery

1. `grep -rnE "\bsrc\b" --include=* . | grep -v /.git/` → only the two C.4
   lines remain. Anything else is a missed reference.
2. `pip install -e . && python -c "import trinity; import trinity.main"` →
   clean (proves the package is importable under the new name).
3. `python -m tools.gen_default_param --check` → in sync (its path literal
   now points at `trinity/_input/default.param`).
4. `pre-commit run --all-files` → clean (ruff **F821** catches any half-
   renamed import).
5. `python -m pytest test/ -q` → 357 pass.
6. `python run.py param/simple_cluster.param` → exits 0, writes outputs;
   logs now show `trinity.*` module paths.

## C.6 Notes / rollback

- Keep it a **pure rename commit** — no opportunistic edits — so review and
  `git blame` stay clean; `git mv` lets history follow.
- External notebooks/scripts doing `from src…` will break and must be
  hand-updated (call out in the commit/PR).
- Single commit; revert restores the prior layout. No physics/output change.

---

# Appendix D — Phase C execution detail (separate the plotting tree)

> **STATUS: EXECUTED (C.1–C.5).** Done in two commits: C.1 carved the 33
> personal scripts into a tracked `scratch/`; C.2–C.5 moved the 5-script
> `make_figures.py` closure to `paper/figures/` (+ infra in `_lib/`),
> relocated `compare_outputs.py` to `tools/`, deleted dead
> `radial_profile.py`/`plot_style.py`, removed `trinity/_plots/`, and added
> `test_engine_purity` to lock the one-way dependency. Verified: 352 tests +
> purity guard, `import trinity` clean, `make_figures.py` drives the
> relocated scripts (3 latex failures are a missing system binary, not the
> refactor). The audience rule below is what was implemented: **"only what
> `paper/make_figures.py` reproduces is the paper; every other `paper_*`
> script is personal,"** via the measured dependency closure.

## D.0 The audience model (supersedes the I.3 "all plots are the deliverable")

`paper/make_figures.py` is a thin driver that invokes plot scripts by
`python -m`. The **public set is its transitive closure**, nothing more;
**everything else is personal**. Measured (not assumed):

- `make_figures.py` drives **4** scripts (the ones with published bundles
  in `paper/data/`): `paper_densityProfile`, `paper_teaser`,
  `paper_radiusComparison`, `paper_rcloud_smoothing`.
- `paper_teaser` additionally imports **`paper_feedback`** → it joins the
  public set by transitivity. (Closure verified: those 5 scripts import no
  other `paper_*`/`pedrini_*`.)
- The infra those 5 actually import: **`cli`, `plot_base`, `plot_markers`,
  `force_colors`, `grid_template`** (+ `trinity.mplstyle`). Self-contained
  (none of them imports a personal script).

Everything not in that closure — **31 scripts** (29 other `paper_*`, both
`pedrini_*`, plus `diag_simplify`) — is **personal**.

## D.1 Measured blast radius (current tree)

- `trinity/_plots/` = **46 `.py` files, 23,229 lines** (~42% of repo
  Python). Split by the closure above:
  - **Public (10 tracked files)**: 5 scripts + 5 infra + `trinity.mplstyle`.
  - **Personal (31 scripts)**: → `scratch/`.
  - **Dead (delete, 0 importers)**: `radial_profile.py` (grep-confirmed
    unused), and `trinity/_functions/plot_style.py` (the only engine→plots
    link — see D.2).
  - **Relocate**: `compare_outputs.py` → `tools/` (user-facing run-diff,
    not a figure).
  *(`diagnostic_parameter_changes.py` from the I.3 audit no longer exists.)*

## D.2 Dependency graph — one-way, single dead link

- **The only engine→plots reference is dead code.**
  `trinity/_functions/plot_style.py:30` builds
  `…/'_plots'/'trinity.mplstyle'`, and `plot_style.py` is imported by
  nobody. So the engine's lone plot coupling rides on a dead module →
  **delete `plot_style.py`** (re-confirm zero importers at execution).
  Live styling is via `plot_base.py`, which travels to the public `_lib/`
  with `trinity.mplstyle`.
- **Exactly one test imports a plot**:
  `test/test_phase4_consumer_migration.py::TestPedriniEmergenceMigration`
  (4 sites, `from trinity._plots.pedrini_emergence_timescales import
  parse_raw_reason`). But `pedrini_emergence_timescales` is **personal**
  (→ scratch), and `parse_raw_reason` is a 2-line wrapper over the engine's
  already-tested `read_simulation_end()`. **DECIDED: drop that test class**
  (its `_build_*` fixtures too if unused elsewhere) — a tracked test must
  not depend on a personal script, and the underlying engine path keeps its
  own coverage.
- No other `trinity/**` file references `_plots` / `mplstyle`.

## D.3 Resolved decisions

- **`scratch/` is TRACKED, not gitignored.** Rationale: this work happens
  in an *ephemeral* container; a gitignored `scratch/` would not be pushed
  and the 31 scripts (~20k lines) would be lost on teardown, surviving only
  in git history. Tracking `scratch/` keeps them version-controlled and out
  of both `trinity/` and `paper/`. (Revisit later on a persistent
  checkout if a truly-ignored sandbox is wanted.)
- **Personal scripts get their imports rewritten so they still run** from
  `scratch/` — best-effort (no test harness covers them): `from
  trinity._plots.<infra>` → the new public `_lib` path, and intra-personal
  `from trinity._plots.<paper_x>` → the sibling `scratch/` path.
- **Package shape**: `paper/figures/` is an **importable package**
  (`__init__.py` in `figures/` and `figures/_lib/`), so `make_figures.py`'s
  `python -m` targets and the infra imports are clean (no `sys.path` hacks).
- **`compare_outputs.py` → `tools/`**; **`diag_simplify.py` → `scratch/`**.
- **`test/` → `tests/`** stays out of scope (fold in separately).

## D.4 The `__file__`-depth landmines (must rewrite, not just `git mv`)

Phase B was safe because it preserved directory depth; Phase C is **not** —
files change depth, so every repo-root shim in a moved file is re-counted:

1. **`plot_base.py`** finds repo-root via `parent.parent.parent`
   (`trinity/_plots/plot_base.py` → 3 up) and hardcodes a `fig/` sibling.
   At `paper/figures/_lib/plot_base.py` the root is **4** up — the shim
   must change or `fig/` output lands in the wrong place. Same audit for
   `cli.py` and each public script's `__main__` shim.
2. **`run_all.py`** holds a hardcoded path table of
   `"trinity/_plots/paper_X.py"` strings. Since most targets are now
   personal, `run_all.py` either moves to `scratch/` with the bulk of the
   scripts **or** is pared to the public set — decide at execution; do not
   leave it pointing at moved files.

## D.5 Execution order (incremental, each its own revertable commit)

1. **C.1 — carve scratch (tracked)**: create `scratch/`; `git mv` the 31
   personal scripts (29 `paper_*` + 2 `pedrini_*`) + `diag_simplify.py`
   there. Safe stop-point.
2. **C.2 — public surface**: `git mv` the 5 public scripts → `paper/figures/`
   and the 5 infra + `trinity.mplstyle` → `paper/figures/_lib/`; add the
   two `__init__.py`.
3. **C.3 — rewrite imports + shims**: public scripts/infra
   `from trinity._plots.X` → `from figures._lib.X`; engine refs stay the
   installed `from trinity._output… import …`; fix the D.4 depth shims;
   rewrite the 31 personal scripts' imports (D.3, best-effort);
   `make_figures.py`'s `module="trinity._plots.paper_X"` →
   `"figures.paper_X"`.
4. **C.4 — deletes + relocate + test**: drop
   `TestPedriniEmergenceMigration`; `git mv compare_outputs.py tools/`;
   delete dead `radial_profile.py` and `trinity/_functions/plot_style.py`
   (called-out). `trinity/_plots/` is now empty → remove it.
5. **C.5 — packaging + docs**: `pyproject.toml` exclude `paper*` /
   `scratch*` from the wheel (mirror `test*`/`docs*`); update
   `docs/source/visualization.rst` and any `_plots` doc mentions.

## D.6 Verification (the untested-surface caveat)

1. Full battery: `pytest test/ -q` (now with the pedrini class dropped),
   ruff clean, `pip install -e . && import trinity` OK, wheel build
   excludes `paper*`/`scratch*`.
2. **Engine-purity guard** (Part III.5): a test asserting `trinity` imports
   nothing from `paper` / `scratch` — what stops drift.
3. **Public repro**: `python paper/make_figures.py` regenerates the 4
   published figures from `paper/data/*.npz` (this is now the *real*
   acceptance test for the public surface).
4. **Personal scripts** are best-effort only — note explicitly that their
   rewrite is unverified beyond import-resolution (`python -c "import ast"`
   compile-check at most).
5. **Expect the flaky `MonotonicError`** to still surface — orthogonal,
   unfixed until the final phase; not a Phase C regression.

Risk: **medium**. Untested surface + real import/depth rewrites (not a pure
`git mv`) + a best-effort rewrite of 31 personal scripts. The C.1→C.5 split
keeps each commit reviewable and revertable; C.1 alone (carve scratch) is a
safe partial landing.
