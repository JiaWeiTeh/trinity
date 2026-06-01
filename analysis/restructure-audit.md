# Repository restructure: audit + phased plan

Single source of truth for three structural changes raised in the
codebase review, in the same shape as `analysis/sb99-refactor-audit.md`
and `analysis/cooling-refactor-audit.md`: (Part I) the audit — *what is*
— with measured blast radii, and (Part II) the phased plan and its
equivalence-test battery — *what to do, in what order, and how to prove
nothing changed*.

End goal:

1. Drop the vestigial `_modified` suffix from the 8 physics modules.
2. Rename the importable package `src/` → `trinity/` so the install
   name (`trinity-sf`), the import name, and the brand agree.
3. Separate the 23.6k-line plotting tree by **role/lifecycle**: the
   engine (installed), paper-reproduction scripts (public deliverable),
   and personal diagnostics (gitignored scratch).

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

Target layout:

```
trinity/            # engine (installed); lib/default stays as package-data
param/              # example .param inputs (public, user-facing)
paper/              # the reproduction DELIVERABLE
  make_figures.py   #   (already present)
  data/             #   (already present)
  figures/          #   <- paper_*/pedrini_* move here
    _lib/           #   <- cli, plot_base, plot_markers, force_colors,
                    #      grid_template, trinity.mplstyle
scratch/            # gitignored: personal diagnostics / exploratory plots
```

### C.1 — Carve out personal/diagnostic scripts
1. Create a gitignored `scratch/` (add to `.gitignore`).
2. Move `diag_simplify.py`, `diagnostic_parameter_changes.py`, and any
   other exploratory scripts there. Decide per-file with the test:
   *"Would a stranger reproduce a paper figure with this?"* If no →
   scratch.
3. **Landmine**: `test_phase4_consumer_migration.py` imports
   `pedrini_emergence_timescales` — that one is a *consumer-migration*
   pin, keep it on the public side and update its import path in C.2.

### C.2 — Move public paper scripts to `paper/figures/`
1. `git mv src/_plots/{paper_*,pedrini_*}.py paper/figures/`.
2. `git mv` the infra (`cli.py`, `plot_base.py`, `plot_markers.py`,
   `force_colors.py`, `grid_template.py`, `radial_profile.py`,
   `compare_outputs.py`, `run_all.py`, `trinity.mplstyle`) →
   `paper/figures/_lib/`.
3. Rewrite imports: `from src._plots.X` → `from figures._lib.X` (or
   whatever package shape `paper/figures` takes), and engine references
   → the installed `from trinity._output.trinity_reader import ...`.
4. Resolve `plot_style.py`: confirm whether it is dead. If only figures
   use the mplstyle, move `plot_style.py` + `trinity.mplstyle` into
   `paper/figures/_lib/` and drop the engine's `_plots` asset reference.
   If genuinely dead, delete it (separate commit, called out).
5. Update `test_phase4_consumer_migration.py`'s import path (or relocate
   the test alongside the figures).
6. Update `docs/source/visualization.rst` (and any `_plots` mentions).
7. `pyproject.toml`: ensure `paper*` is excluded from the wheel (figures
   are not installed code), matching the existing `test*`/`docs*`
   excludes.

### C.3 — Verify (plots are untested)
- Full battery (the migrated consumer test must pass).
- **Manual**: run `paper/figures/_lib/run_all.py` (default curated set:
  `paper_feedback`, `paper_momentum`, `paper_escapeFraction`) against a
  sample output dir and eyeball the figures render. There is no automated
  net for the other 31 scripts — note that explicitly.

Risk: **medium**. Untested surface, many import rewrites. Splitting C.1
from C.2 keeps each commit reviewable and lets you stop after C.1 if the
full move feels too big.

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
path), so it lives in its own doc, `analysis/bubble-integrator-robustness.md`,
and must land after the structural churn settles. Status of A/B done:
executed; C: planned (Appendix D); integrator: planned (separate doc).

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
│   ├── figures/           # paper_*/pedrini_* move here
│   │   └── _lib/          # plot_base, plot_markers, cli, force_colors,
│   │                      #   grid_template, trinity.mplstyle
│   └── make_figures.py
├── lib/default/           # bundled tables, top-level by design (keeps trinity/ pure code)
├── param/                 # canonical parameter-file library
├── examples/              # runnable getting-started scripts (reference param/)
├── scratch/               # gitignored personal/diagnostic (NOT "notebooks/")
├── analysis/  docs/  tests/
├── pyproject.toml  README.md     # run.py -> console entry point
```

## III.1 What stays in the engine vs. moves out (the audience test)

Criterion: *"would any user run this on their own run output?"* — not
"is it reused across paper scripts." By that test, the `_plots`
infrastructure is paper-only and leaves the engine:

| File | Destination | Rationale |
|------|-------------|-----------|
| `plot_base.py` | `paper/figures/_lib/` | docstring: "for paper scripts"; hardcodes `parent.parent.parent` + `fig/` (breaks if moved up); not a public API |
| `plot_markers.py` | `paper/figures/_lib/` | used only by `paper_*` |
| `cli.py` | `paper/figures/_lib/` | docstring: "CLI builder for `_plots` paper scripts" |
| `force_colors.py`, `grid_template.py` | `paper/figures/_lib/` | paper-only |
| `trinity.mplstyle` | `paper/figures/_lib/` | only paper infra loads it |
| `compare_outputs.py` | `tools/` (or `trinity/viz`) | genuinely user-facing: compares two runs via the public `dictionary.jsonl`; run via `python -m` |

Outcome: `src/_plots/` ends up empty (or holding only `compare_outputs`)
— delete the folder rather than leave it named after the thing removed.
The engine's visualization surface should be defined by the public output
object (`TrinityOutput`), never by figure scripts.

## III.2 Directory conventions

- **`param/` vs `examples/`**: `param/` is the canonical parameter-file
  library; `examples/` holds narrative getting-started scripts that
  *reference* `param/` files (no duplicated `.param` content).
- **`scratch/` not `notebooks/`**: in nearly every project `notebooks/`
  is a *committed* tutorial asset. Use `scratch/` (gitignored) for
  personal/ephemeral work; reserve `notebooks/` for committed tutorials
  if ever wanted. Move `diag_simplify.py` /
  `diagnostic_parameter_changes.py` to `tools/` (if maintained) or
  `scratch/` (if personal).
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

# Open decisions for the user

- **Phase B layout**: ~~minimal `trinity/` at root (recommended) vs.
  `src/trinity/` src-layout proper?~~ **DECIDED: root-level `trinity/`.**
- **Phase C package shape**: should `paper/figures/` be an importable
  package (with `__init__.py`, so `run_all.py` can do
  `from figures._lib import ...`), or a flat script dir run with
  `python paper/figures/paper_feedback.py` and `sys.path` shims? The
  former is cleaner; the latter is closer to the current per-script
  `__main__` style.
- **`plot_style.py`**: confirm dead vs. live before relocating the
  mplstyle.

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
- `analysis/cooling-refactor-audit.md` and `analysis/sb99-refactor-audit.md` — multiple `*_modified.py` mentions are **historical point-in-time records** of completed refactors. Recommendation: leave as-is (they document what existed then). Optional: a one-line note that files were later renamed in Phase A.

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
> `analysis/bubble-integrator-robustness.md`.

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

> **STATUS: PLANNED.** Measured against the post-Phase-B tree on
> `feature/reforming-structure`. This is the one structural phase with an
> *untested* surface, so it is split into independently-revertable
> sub-commits and verified partly by hand.

## D.1 Measured blast radius (current tree)

- `trinity/_plots/` = **46 `.py` files, 23,229 lines** (~42% of repo
  Python). Composition: **34 `paper_*`**, **2 `pedrini_*`**, **1 diag**
  (`diag_simplify.py`), infra (`cli.py`, `plot_base.py`, `plot_markers.py`,
  `force_colors.py`, `grid_template.py`, `radial_profile.py`,
  `run_all.py`), the genuinely-user-facing `compare_outputs.py`, plus
  `trinity.mplstyle` and `__init__.py`.
  *(Note: `diagnostic_parameter_changes.py` from the I.3 audit no longer
  exists — only `diag_simplify.py` remains in the diag bucket.)*

## D.2 Dependency graph — re-confirmed one-way (post-rename)

- **The only engine→plots link is dead code.**
  `trinity/_functions/plot_style.py:30` builds
  `os.path.join(__file__, '..', '_plots', 'trinity.mplstyle')` — and
  `plot_style.py` is **imported by nobody** (grep finds only its own
  definition). So the single asset coupling rides on a dead module.
  → **Decision: confirm-dead then delete `plot_style.py`** in its own
  called-out commit; the live styling path is `plot_base.py` (which the
  paper infra imports), and `trinity.mplstyle` travels with it to
  `paper/figures/_lib/`.
- **Exactly one test imports a plot**:
  `test/test_phase4_consumer_migration.py` (4 sites, all
  `from trinity._plots.pedrini_emergence_timescales import parse_raw_reason`).
  `pedrini_emergence_timescales` therefore stays on the public side and
  this test's import path is rewritten in C.2.
- No other `trinity/**` file references `_plots` / `mplstyle`. The engine
  is clean to amputate from.

## D.3 The two `__file__`-depth landmines (must rewrite, not just `git mv`)

Phase B was safe *because* it preserved directory depth; Phase C does
**not** — files move to a different depth, so every repo-root shim in the
moved files must be re-counted:

1. **`plot_base.py`** computes repo-root via `parent.parent.parent`
   (`trinity/_plots/plot_base.py` → 3 parents = repo root) and hardcodes a
   `fig/` sibling. At `paper/figures/_lib/plot_base.py` the root is **4**
   parents up — the shim must change (`parents[2]`→`parents[3]`) or the
   `fig/` output path silently lands in the wrong place. Same audit for
   `cli.py` and any `paper_*` `__main__` `sys.path` shim.
2. **`run_all.py`** holds a hardcoded path table of
   `"trinity/_plots/paper_X.py"` strings (34 rows, rewritten from
   `src/_plots/…` in Phase B). Every row must be repointed to
   `paper/figures/paper_X.py`, and `run_all.py`'s own root-depth shim
   re-counted.

## D.4 Open decisions (recommended answers)

- **Package shape** → **importable `paper/figures/` package** (add
  `__init__.py` + `paper/figures/_lib/__init__.py`), so `run_all.py` and
  the consumer test use clean `from figures._lib.cli import …` instead of
  per-script `sys.path` hacks. (The flat-script alternative keeps the
  current `__main__` style but multiplies the depth-shim landmines.)
- **`compare_outputs.py`** → **`tools/`** (or `trinity/viz/`): it is
  genuinely user-facing (diffs two runs via the public `dictionary.jsonl`,
  run as `python -m`), not a paper figure. Keep it out of `paper/figures/`.
- **`diag_simplify.py`** → **`scratch/`** (gitignored) unless it is a
  maintained diagnostic, in which case `tools/`. Decide with the audience
  test ("would a stranger reproduce a paper figure with this?" → no).
- **`test/` → `tests/`** stays out of scope (trivial, fold in separately).

## D.5 Execution order (incremental, each its own revertable commit)

1. **C.1** — create gitignored `scratch/`; `git mv diag_simplify.py` there.
   Stop-point: safe to halt here if the full move is too big.
2. **C.2a** — `git mv` the 34 `paper_*` + 2 `pedrini_*` → `paper/figures/`;
   `git mv` infra (`cli`, `plot_base`, `plot_markers`, `force_colors`,
   `grid_template`, `radial_profile`, `run_all`, `trinity.mplstyle`) →
   `paper/figures/_lib/`; add the two `__init__.py`.
3. **C.2b** — rewrite imports: `from trinity._plots.X` → `from figures._lib.X`
   (infra) and leave engine refs as the *installed*
   `from trinity._output… import …`. Fix the D.3 depth shims. Repoint
   `run_all.py`'s path table. Update
   `test/test_phase4_consumer_migration.py`'s import path.
4. **C.2c** — `git mv compare_outputs.py tools/`; confirm-dead and delete
   `trinity/_functions/plot_style.py` (called-out commit).
5. **C.3** — `pyproject.toml`: exclude `paper*` / `scratch*` from the wheel
   (mirror the existing `test*`/`docs*` excludes); `_plots` is gone so drop
   any leftover reference. Update `docs/source/visualization.rst` and any
   `_plots` doc mentions.

## D.6 Verification (the untested-surface caveat)

1. Full battery: `pytest test/ -q` (the migrated consumer test must pass),
   ruff clean, `pip install -e . && import trinity` still OK, wheel build
   excludes `paper*`.
2. **Engine-purity guard** (Part III.5): add a test asserting `trinity`
   imports nothing from `paper`/`scratch` — this is what stops drift.
3. **Manual** (no automated net for 33 of 34 figures): run
   `python -m figures._lib.run_all` (curated default set) against a sample
   output dir and eyeball that figures render. Call out explicitly that the
   other scripts are unverified.
4. **Expect the flaky `MonotonicError`** to still surface here (it is
   orthogonal and unfixed until the final phase) — do not mistake it for a
   Phase C regression.

Risk: **medium**. Untested surface + real import/depth rewrites (not a pure
`git mv`). C.1/C.2 split keeps each commit reviewable and revertable.
