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

- **Phase B layout**: minimal `trinity/` at root (recommended) vs.
  `src/trinity/` src-layout proper?
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
