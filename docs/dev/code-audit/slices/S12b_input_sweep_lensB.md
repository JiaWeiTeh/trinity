# S12b input sweep — Lens B (what the code claims)

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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**This is a prose-only transcription.** Every statement below is a claim *made by comments and
docstrings*, not an observation of behaviour. I read exactly one file — the extracted prose of the
slice — and **have seen no code**: no function bodies, no signatures, no tests, no `.param` files,
no other lens's report. Where I write "the prose claims X", I mean the text asserts X; I cannot and
do not assert that the code does X.

Slice files (prose only):

- `trinity/_input/sweep_parser.py`
- `trinity/_input/sweep_jobs.py`
- `trinity/_input/sweep_runner.py`

Citations are `file:line`, line = first line of the comment/docstring block carrying the claim.
Line numbers are those stamped in the prose extract.

---

## 1. The documented sweep grammar

### 1.1 Value syntaxes claimed to expand into an axis

| Syntax | Claim | Citation |
|---|---|---|
| `key [v1, v2, v3]` | list syntax is *detected* and separates "sweep (varying)" from "base (constant)" params | `sweep_parser.py:3` |
| `mCloud [1e5, 1e7, 1e8]` | the module docstring's canonical example of an axis | `sweep_parser.py:3` |
| scientific notation `[1e5, 1e7, 1e8]` | handled | `sweep_parser.py:97` |
| decimals `[0.01, 0.10, 0.30]` | handled | `sweep_parser.py:97` |
| mixed `[1e5, 100000, 1e8]` | handled | `sweep_parser.py:97` |
| strings `[densPL, densBE]` | handled | `sweep_parser.py:97` |
| booleans `[True, False]` | handled | `sweep_parser.py:97` |
| `tuple(p1, p2, ...) [v1, v2] [v3, v4] ...` | explicit combination list, "instead of Cartesian product" | `sweep_parser.py:3`, `sweep_parser.py:155` |

Claimed parse precedence at top level: **"Precedence: list → boolean → number → fraction → string"**
(`sweep_parser.py:41`). Inside a list the same order minus the list rung, per the inline comments:
boolean (`:130`) → number (`:136`) → fraction (`:140`) → string fallback (`:144`). Fractions such as
`5/3` are claimed to parse (`sweep_parser.py:86`).

### 1.2 Value syntaxes claimed **not** to expand into an axis

- **Single-element lists.** `"# Single-element list is treated as base param"` (`sweep_parser.py:339`).
  This is the only documented demotion.
- **Nested lists.** `"# Parse each item individually (without list detection to avoid recursion)"`
  (`sweep_parser.py:125`) — a list inside a list is claimed not to be re-parsed as a list.
- **Scalars** — anything not bracketed lands in `base_params`, "Single values (constant across all
  runs)" (`sweep_parser.py:263`).
- **Params named inside a `tuple(...)` header** — they vary, but per the module docstring not by
  Cartesian product; and they "must not" also appear as sweep lists (`sweep_parser.py:461`).

### 1.3 Stated escape hatches

Two, and only two, are stated:

1. **Demote an axis to a constant**: write it as a single-element list (`sweep_parser.py:339`) — or
   as a bare scalar.
2. **Escape the Cartesian product** (not list parsing): tuple mode, `"This runs only the specified
   combinations, not the full Cartesian product."` (`sweep_parser.py:3`).

**No escape hatch is documented for list *parsing* itself.** The prose never says how to give a
parameter a value that is literally a bracketed string, nor how to express a parameter that is
genuinely list-valued in the physics. Per the stated rule, any multi-element bracketed value becomes
a sweep axis (finding `S12b-B-21`).

### 1.4 Line-level grammar claims

- `"# Parse parameter line (format: key value)"` (`sweep_parser.py:319`, `:434`).
- `"# Find the first whitespace that separates key from value"` (`sweep_parser.py:321`) — the first
  whitespace is the delimiter, because `"# Handle list values which may contain spaces after commas"`
  (`:320`).
- `"# Remove inline comments"` (`:309`, `:407`) and `"# Skip empty lines"` (`:315`, `:413`).
- Tuple lines: `"# Check if line starts with 'tuple('"` (`:178`) — a stated precondition — and
  `"# Find matching closing parenthesis"` (`:183`). `parse_tuple_line` returns
  `"(param_names, tuple_values) or None if not a tuple line"` (`:155`), with the worked example
  `parse_tuple_line("tuple(sfe, mCloud) [0.01, 1e5] [0.10, 1e7]")` → `(['sfe', 'mCloud'], [[0.01, 100000.0], [0.1, 10000000.0]])`.

---

## 2. Modes and claimed combination ordering

Three modes are claimed (`sweep_parser.py:489`, restated `:856`):

> "- Cartesian mode: Lists of values generate all combinations
>  - Tuple mode: Explicit tuples specify exact combinations
>  - Hybrid mode: Tuple combinations × Cartesian product of sweep params"

Counting contract (`sweep_parser.py:856`): "Cartesian mode: Product of all sweep param lengths ·
Tuple mode: Number of explicit tuples · Hybrid mode: Number of tuples × product of sweep param
lengths". `count_combinations` claims to do this "without generating them" (`:833`).

Mode predicates: `is_tuple_mode` — "Check if this uses tuple syntax (pure tuple or hybrid)."
(`:253`); `is_hybrid_mode` — "Check if this is hybrid mode (tuple + sweep params)." (`:258`).

**Ordering claims, in full:**

- `"# Get keys and values in consistent order"` (`sweep_parser.py:569`) — the *only* statement about
  Cartesian ordering. It does not say what the order is (file order? sorted?) nor which axis varies
  fastest. See finding `S12b-B-11`; this is load-bearing because `runs.tsv` index is claimed to equal
  the SLURM array id.
- `"# Hybrid mode: tuple combinations × sweep params Cartesian product"` (`:510`) — implies tuples
  are the outer loop, sweep params inner.
- `"# Sorted for deterministic ordering regardless of sweep-file key order."` (`:759`) — this *is*
  specified, but only for the ordering of **generic name suffixes**, not for combination ordering.
- `"# Pure tuple mode: use explicit combinations only"` (`:531`); `"# Pure Cartesian mode: use
  existing logic"` (`:539`).
- Degenerate case: `"# No sweep parameters - just return base params with generated name"` (`:564`).

---

## 3. The naming contract (quoted exactly)

Two different forms of the same contract appear in the file.

**Loose form** — module docstring and both generator docstrings:

> "Generate run names following TRINITY convention: {mass}_sfe{sfe}_n{nCore}" (`sweep_parser.py:3`)
> "- output_name: Generated name following convention {mass}_sfe{sfe}_n{nCore}" (`:489`, `:547`)

**Precise form** — `generate_run_name` (`sweep_parser.py:687`):

> "Generate output folder name following existing TRINITY convention.
> Format: {mCloud}_sfe{sfe*100:03d}_n{nCore}[_profile][_PHII][_generic...]
> Examples:
> - 1e7_sfe010_n1e4 (default, no suffixes)
> - 1e5_sfe001_n1e4_PL0 (powerlaw alpha=0)
> - 1e5_sfe001_n1e4_PL-2 (powerlaw alpha=-2)
> - 1e5_sfe001_n1e4_BE14 (Bonnor-Ebert Omega=14.1)
> - 1e7_sfe010_n1e4_yesPHII (include_PHII=True explicit in sweep)
> - 1e7_sfe010_n1e4_noPHII (include_PHII=False explicit in sweep)
> - 1e5_sfe001_n1e4_PL0_noPHII (combined suffixes)
> - 1e5_sfe001_n1e4_Z0p5 (generic suffix for an arbitrary swept key)"

Field-formatting claims:

- mCloud/nCore: `"# Format mCloud (scientific notation)"` (`:724`), `"# Format nCore (scientific
  notation)"` (`:731`), via `format_scientific` — "Always uses scientific notation for values >= 100
  to match TRINITY naming" with examples `1000000.0 -> '1e6'`, `100000.0 -> '1e5'`,
  `5000000.0 -> '5e6'`, `100.0 -> '1e2'`, `0.01 -> '0.01'` (`:781`).
- sfe: `"# Format SFE (multiply by 100, zero-padded to 3 digits)"` (`:727`).
- alpha: `"# Format alpha: 0 -> \"0\", -1 -> \"-1\", -2 -> \"-2\""` (`:741`).
- omega: `"# Format omega: 14.1 -> \"14\", 7.0 -> \"7\""` (`:746`) — rounding vs truncation unstated.
- Density-profile suffix "if present" (`:736`).
- PHII: `"# Add PHII suffix only when the user explicitly sets include_PHII in the sweep file.
  Matches the density-profile convention above: absent key -> no tag (runtime falls back to
  default.param)."` (`:750`). Note the asymmetry: curated suffixes are **presence**-triggered,
  generic suffixes are **swept**-triggered.

**Generic suffix** (`sweep_parser.py:649`): "Build a single ``{key}{value}`` token … snake_case keys
become camelCase, decimal points in floats become ``p`` … and any char outside ``[A-Za-z0-9.+-]`` is
collapsed to ``-`` so spaces, brackets, shell wildcards, and unicode never leak into the folder
name. Minus signs are preserved, matching the existing ``_PL-2`` convention." Examples:
`coll_counter=True -> "collCounterTrue"`, `Z=0.5 -> "Z0p5"`, `kB=3 -> "kB3"`,
`label="my run" -> "labelmy-run"`. Curated keys are exempt: `"# Parameters that already have a
curated, human-readable slot in the run name … Any *other* swept parameter falls through to the
generic suffix logic below."` (`:587`); and the self-test asserts `"# A curated key passed in
swept_keys must NOT be double-encoded"` (`:953`).

**Charset rationale** (`:601`): "Anything outside this set is sanitised to ``-`` in a generic value
token. Keeps alphanumerics plus the printable scaffolding actually needed by the existing convention
(`.` survives the sanitiser then becomes `p`; `+`/`-` stay for scientific notation and negatives like
`_PL-2`; `_` would split tokens visually so we don't allow it through the value)."

**Length budget** (`:596`): "Total length budget for a generated run name. Most filesystems cap a
single path component at 255 bytes; we reserve ~55 for the ``_modified`` tag, ``.param``/
``_summary.txt`` siblings, and OS overhead." The self-test says `"# Length guard: a synthetic huge
value triggers the cap"` (`:998`) — but the prose never says whether hitting the cap **truncates or
raises**.

**Safety rejects** (`sweep_parser.py:610`): "Raise ``ValueError`` if a swept value would be unsafe to
embed in a folder name. Two categories are hard-rejected (no sanitisation): 1. Anything that looks
like a filepath (``/``, ``\``, or ``..``). … Per project policy: sweeping filepath-typed parameters
is not supported — set the path once in the base param file instead. 2. Control characters (ASCII <
32 or 127). … Non-string values bypass this check (numbers/bools can't contain unsafe chars)."
Self-test: `"# Hard-reject cases: unsafe values must raise ValueError"` (`:976`) and `"# Sanitisation
cases: ugly-but-safe chars get collapsed to '-'"` (`:962`).

---

## 4. Output-directory layout and the emitted bundle

**In-process runner** (`sweep_runner.py`):

- `"# Create output directory for this run"` / `"# Each run gets its own subfolder so outputs don't
  overwrite each other"` (`:244`).
- `"# Create parameter file in the output directory"` / `"# path2output points to the run-specific
  folder, not the base sweep folder"` (`:249`).
- `generate_param_file` documents `run_output_dir` as "Output directory for this specific run (e.g.
  outputs/sweep/1e5_sfe001_n1e2/)" (`:160`) — a **relative** example (see `S12b-B-16`).

**Job-array bundle** (`sweep_jobs.py:100`), quoted exactly because downstream tooling depends on it:

> "Writes ``<jobs_dir>/params/<name>.param`` per combination, ``runs.tsv``
> (``param_path<TAB>output_dir``, one per line, index == SLURM array id), a self-describing
> ``manifest.json``, and ``submit_sweep.sbatch``."

with "base_output_dir : str — Absolute base directory; each run lands in ``<base>/<name>``." and
"jobs_dir : str or Path — Where the bundle is written (resolved to absolute)."

Per-run sentinels are claimed to be `.exit_code` and `.duration` files, and the aggregated reports
`sweep_report.txt` / `sweep_report.json` are written "into the sweep's base output directory"
(`sweep_jobs.py:283`). The in-process side documents the same two report artefacts
(`sweep_runner.py:3`, `:481`, `:596`) plus a `PHYSICS OUTCOMES` section read from each run's
`metadata.json` (`sweep_runner.py:532`).

---

## 5. Worker / parallelism semantics (claimed)

- Engine: "Parallel execution engine for parameter sweeps. Features: - Run simulations in parallel
  batches using ProcessPoolExecutor" (`sweep_runner.py:3`).
- Each pooled worker is claimed to shell out: "Execute a single TRINITY simulation. Creates a
  parameter file and runs `python run.py <param_file>`" (`sweep_runner.py:219`).
- **Isolation claims**: separate output subfolder per run (`:244`), per-run `path2output` (`:249`),
  and per-run `.param` written into that folder (`:249`).
- **Thread isolation** (`sweep_runner.py:268`): "Build environment for subprocess: limit BLAS/LAPACK
  threads to 1. Otherwise each sweep worker's NumPy/SciPy would use all cores via OpenMP/MKL/
  OpenBLAS, and N workers x M implicit threads saturates the CPU (thermal throttling on laptops,
  oversubscription on HPC, angry cluster admins). Each simulation is CPU-light; parallelism comes
  from running many simulations, not threading one of them. Users who need BLAS threading inside a
  single run can still override these vars externally before launching." Plus `"# Apple Accelerate"`
  (`:281`).
- **Timeout default**: "timeout_hours : float — Timeout in hours (default: 24)" (`sweep_runner.py:219`).
- **Failure handling**: "Continue on failure, collect and report all errors at end"
  (`sweep_runner.py:3`); `"# Extract last portion of stderr for error message"` (`:306`); report
  shows `"# Last 20 lines"` (`:511`) with `"# Indent error message"` (`:509`).
- **Exit-code behaviour of the in-process sweep is never stated** anywhere in this slice (finding
  `S12b-B-18`). Only the job-array path documents exit codes, via `.exit_code` sentinels and the
  magic value −2.
- Progress: "Progress bar with fallback if tqdm not available. Features: - Completed/Total count -
  Percentage - Elapsed time - ETA - Current simulation name" (`sweep_runner.py:350`), `"# Try to use
  tqdm if available"` (`:371`), `"# Truncate or pad to terminal width"` (`:430`). The module
  docstring lists a shorter feature set ("completed/total, percentage, and ETA", `:3`).

---

## 6. Claims about the emitted job scripts

**Target scheduler**: SLURM, explicitly — "SLURM job-array generation and result collection for
TRINITY sweeps" (`sweep_jobs.py:3`).

**Rationale claim** (`sweep_jobs.py:3`): "The in-process sweep runner (run.py:run_sweep) parallelises
a sweep across the cores of a *single* machine. On an HPC cluster the conventional pattern is instead
a scheduler job array: one array task per combination, so the scheduler packs them across many nodes,
handles fair-share, and restarts failed tasks independently."

**Assumed environment / working directory** (`sweep_jobs.py:3`): "Each array task simply runs
``python run.py <combo>.param`` -- the emitted files contain only scalar values, so they route
through the single-run path (no nested sweep). Inputs are located relative to the package
(_REPO_ROOT-anchored) and each combo's path2output is absolute, so tasks are independent of the
working directory the array runs in." Restated at the script level (`:48`): "One simulation per array
task. Math libraries are pinned to a single thread (parallelism comes from running many tasks, not
from threading one sim), mirroring the in-process runner's per-worker environment. Paths are absolute
so the task is independent of the directory the array runs in."

**Documented example command lines**: exactly one — `python run.py <combo>.param` (`sweep_jobs.py:3`).
The prose contains **no example of the submission command** and **no statement of the `--array`
range**, despite the index↔array-id contract.

**Scheduler flags referenced**: array throttle "`%K` (from ``--workers``); None means no limit"
(`sweep_jobs.py:100`), and `--job-name`: `"# --job-name auto-derived from the sweep param file (falls
back to the jobs dir), sanitised to SLURM-safe characters — e.g. trinity-paperII_grid_sweep."`
(`:204`). "SLURM-safe characters" is not defined.

**Overwrite guard** (`sweep_jobs.py:157`): "Overwrite guard: never clobber an existing bundle. A
submitted array reads runs.tsv by index, so regenerating in place would desync a running job. Require
a fresh directory."

**Validation policy** (`sweep_jobs.py:134`): "Validate up front so the operator sees implausible
combos before queueing doomed tasks. Warn but still emit: a few bad combos shouldn't block the rest
(they fail fast and surface in the collected report), matching run_sweep's behaviour."
`emit_jobs` returns "(n_jobs, n_invalid)" (`:100`); whether `n_jobs` includes the invalid ones is not
stated. `dry_run` is "If True, validate and print a summary but write nothing." (`:100`) — its
interaction with the overwrite guard is unstated.

**Collection** (`sweep_jobs.py:283`): "Aggregate per-task results into a SweepReport. Reads
``<jobs_dir>/manifest.json`` and each run's ``.exit_code`` / ``.duration`` sentinels, then writes
``sweep_report.txt`` / ``sweep_report.json`` into the sweep's base output directory -- identical to
what the in-process runner produces." Write order is justified at `:365`: "JSON first: it needs no
extra I/O, so the machine-readable report is guaranteed even if the (slower) text report's per-run
metadata reads stall or are interrupted on a large sweep."

**Failure triage** (`sweep_jobs.py:250`): "Tally failed runs by each *swept* parameter (and by return
code) so a regime-shaped failure -- e.g. 'small clouds + high sfe + high cooling_boost_kappa' -- is
visible at a glance instead of only a flat list of array indices. Returns the printable block ('' if
none). A 'swept' axis is a param that takes >1 but <n_runs distinct values (so per-run identifiers
like path2output/model_name are excluded). Return code -2 = no sentinel (the task was killed, e.g.
wall-time/OOM, or was still running at collect time), not a sim-level crash." The matching inline
comment reads `"# no sentinel: task did not run or is still running"` (`:324`) — a slightly different
account of the same code (killed vs did-not-run).

---

## 7. Units and physical-constant claims

`_validate_sweep_combination` (`sweep_runner.py:77`): "Sweep parameter values come directly from the
.param file *without* going through ``read_param``, so they are still in their input units (nCore/nISM
in cm⁻³, mu_convert in m_H, mCloud in Msun, rCore in pc). The GMC validator, however, expects values
in TRINITY's astronomy code units (pc⁻³, Msun, pc). Apply the same ``convert2au`` conversions that
``read_param`` applies on the single-run path so the preflight check matches what the actual
simulation will see." It "Returns GMCValidationResult or None if validation cannot be performed",
is "Shared by the in-process sweep runner (run.py:run_sweep) and the job-array generator
(sweep_jobs.emit_jobs)", and "Imports are deferred so importing this module stays cheap on the
single-run path."

Claimed factors (`sweep_runner.py:108`):

| Claim | Independent check |
|---|---|
| "mCloud: [Msun] -> Msun (factor 1)" | consistent |
| "nCore, nISM: [cm**-3] -> pc⁻³ (factor ~2.94e+55)" | ✅ (1 pc)³ = 2.9380e55 cm³ |
| "rCore: [pc] -> pc (factor 1)" | consistent |
| "mu_convert: [m_H] -> Msun (factor ~9.42e-58)" | ❌ m_H/M_⊙ = 8.414e-58 (proton: 8.410e-58). See `S12b-B-01`. |

Also: `"# rCloud_max is in pc (identity conversion), matching the computed rCloud; fall back to the
module default when not overridden."` (`:125`), `"# already in pc"` (`:137`), `"# dimensionless
ratio"` (`:142`), and `"# v2 is stored internally in pc/Myr; show km/s like show_run does."`
(`:541`) — the pc/Myr→km/s factor itself is not stated (it is 0.9778).

---

## 8. Every documented default, precondition, invariant, and "must/always/never/guaranteed"

**Defaults**

- `timeout_hours` "default: 24" (`sweep_runner.py:219`).
- `concurrency` "None means no limit" (`sweep_jobs.py:100`).
- `swept_keys` optional: "When omitted, only the curated base name and suffixes are produced
  (back-compatible with single-run callers)." (`sweep_parser.py:687`).
- "absent key -> no tag (runtime falls back to default.param)" for `include_PHII` and dens-profile
  (`sweep_parser.py:750`).
- `"# Extract key parameters (with defaults if missing)"` (`sweep_parser.py:719`) — **values never
  stated**.
- `"fall back to the module default when not overridden"` for `rCloud_max` (`sweep_runner.py:125`) —
  value never stated.
- `--job-name` "auto-derived from the sweep param file (falls back to the jobs dir)"
  (`sweep_jobs.py:204`).
- Thread env vars pinned to 1 (`sweep_runner.py:268`, `sweep_jobs.py:48`).

**Preconditions**

- Tuple line "starts with 'tuple('" (`sweep_parser.py:178`).
- `emit_jobs` requires "a fresh directory" (`sweep_jobs.py:157`).
- `base_output_dir` is an "Absolute base directory" (`sweep_jobs.py:100`).
- `trinity_root` is "TRINITY root, used to locate ``run.py`` in the sbatch script." (`sweep_jobs.py:100`).
- `params` "containing mCloud, sfe, nCore, and optionally dens_profile, densPL_alpha, densBE_Omega"
  (`sweep_parser.py:687`).

**Postconditions / raises**

- `read_sweep_param` "Raises … ValueError If parameter file has formatting errors; FileNotFoundError
  If file does not exist" (`sweep_parser.py:263`). `read_sweep_config` documents **no** Raises
  (`:355`) despite the overlap validation at `:461`.
- `_reject_unsafe_sweep_value` raises `ValueError` (`sweep_parser.py:610`, self-test `:976`).
- `parse_tuple_line` returns "None if not a tuple line" (`:155`, self-test `:1043`).
- `failure_breakdown` "Returns the printable block ('' if none)" (`sweep_jobs.py:250`).
- `emit_jobs` returns "(n_jobs, n_invalid)" (`sweep_jobs.py:100`).

**Absolute-language statements (verbatim)**

| Word | Statement | Citation |
|---|---|---|
| never | "so distinct combinations **never** collapse onto the same folder name" | `sweep_parser.py:687` |
| never | "spaces, brackets, shell wildcards, and unicode **never** leak into the folder name" | `sweep_parser.py:649` |
| never | "Overwrite guard: **never** clobber an existing bundle" | `sweep_jobs.py:157` |
| never | "the curated suffixes **never** contain ``.`` or ``_``" | `sweep_parser.py:649` |
| always | "**Always** uses scientific notation for values >= 100 to match TRINITY naming" | `sweep_parser.py:781` |
| must not | "tuple params and sweep params **must not** overlap" | `sweep_parser.py:461` |
| must not | "A curated key passed in swept_keys **must NOT** be double-encoded" | `sweep_parser.py:953` |
| must | "unsafe values **must** raise ValueError" | `sweep_parser.py:976` |
| guaranteed | "the machine-readable report is **guaranteed** even if the (slower) text report's per-run metadata reads stall" | `sweep_jobs.py:365` |
| not supported | "Per project policy: sweeping filepath-typed parameters is **not supported**" | `sweep_parser.py:610` |
| identical | "-- **identical** to what the in-process runner produces" | `sweep_jobs.py:283` |
| only | "This runs **only** the specified combinations" | `sweep_parser.py:3` |
| only | "Add PHII suffix **only** when the user explicitly sets include_PHII" | `sweep_parser.py:750` |

---

## 9. External / third-party references cited by the prose

- **SLURM** — job arrays, array id, `%K` throttle, `--job-name`, `submit_sweep.sbatch`, fair-share,
  task restart (`sweep_jobs.py:3`, `:100`, `:157`, `:204`).
- **File formats** — `.param`, `runs.tsv` (`param_path<TAB>output_dir`), `manifest.json`,
  `metadata.json`, `sweep_report.txt`, `sweep_report.json`, `.exit_code`, `.duration`,
  `_summary.txt` (`sweep_jobs.py:100`, `:283`; `sweep_parser.py:596`; `sweep_runner.py:532`).
- **Python stdlib / libs** — `ProcessPoolExecutor` (`sweep_runner.py:3`), `tqdm` as an optional
  dependency (`sweep_runner.py:350`, `:371`), NumPy/SciPy (`sweep_runner.py:268`).
- **Threading backends** — OpenMP, MKL, OpenBLAS, Apple Accelerate (`sweep_runner.py:268`, `:281`).
- **Filesystem/OS** — "Most filesystems cap a single path component at 255 bytes"
  (`sweep_parser.py:596`); ASCII control-char range "< 32 or 127" (`:610`).
- **Intra-project** — `run.py:run_sweep`, `trinity/_input/default.param`, `read_param`, `convert2au`,
  `GMCValidationResult`, `show_run` (`sweep_jobs.py:3`; `sweep_runner.py:77`, `:108`, `:541`).
- **Astrophysics naming** — Bonnor-Ebert Ω, power-law α (`sweep_parser.py:687`).

---

## 10. Internal contradictions within the prose

The sharpest ones, in order of how load-bearing they are:

1. **Tuple mode "fixed" vs hybrid mode.** `sweep_parser.py:3` — "Parameters not in the tuple are
   fixed across all runs." `sweep_parser.py:489` — "Hybrid mode: Tuple combinations × Cartesian
   product of sweep params", and `:258` "Check if this is hybrid mode (tuple + sweep params)". The
   module docstring's contract is false whenever hybrid mode is used, and hybrid mode is not
   mentioned in it at all.
2. **`format_scientific` three-way split.** Docstring (`:781`) "Always uses scientific notation for
   values >= 100" with example "0.01 -> '0.01'"; comment (`:808`) "For values >= 100 **or very small
   numbers**, use scientific notation"; comment (`:825`) "For \"normal\" numbers (1-99), use regular
   formatting". The example value 0.01 satisfies neither comment's regular-format branch.
3. **`failure_breakdown`'s definition of a swept axis excludes the 1-D case.** "A 'swept' axis is a
   param that takes >1 but <n_runs distinct values" (`sweep_jobs.py:250`) — in a single-axis sweep
   the axis has exactly `n_runs` distinct values, so the stated rule drops the only axis there is.
4. **"never collapse" vs a many-to-one sanitiser.** "distinct combinations never collapse onto the
   same folder name" (`:687`) vs "any char outside ``[A-Za-z0-9.+-]`` is collapsed to ``-``" (`:649`)
   — the documented mapping is not injective (`"my run"` and `"my_run"` both → `my-run`).
5. **"never collapse" vs the length cap.** `:687` vs `:596`/`:998`; the cap's overflow behaviour is
   never stated.
6. **`generate_run_name` params: required or defaulted?** Docstring "Parameter dictionary containing
   mCloud, sfe, nCore" (`:687`) vs `"# Extract key parameters (with defaults if missing)"` (`:719`).
7. **Two spellings of the naming contract** — `{mass}_sfe{sfe}_n{nCore}` (`:3`, `:489`, `:547`) vs
   `{mCloud}_sfe{sfe*100:03d}_n{nCore}[_profile][_PHII][_generic...]` (`:687`).
8. **Single-element-list demotion documented in only one of the two readers** — `:339` (in
   `read_sweep_param`) vs `:445` (`read_sweep_config`, bare "Categorize based on whether it's a
   list"), and it contradicts the docstring rule "sweep_params: List values (will generate
   combinations)" (`:263`).
9. **`.` → `p` scope.** Comment `:601` "`.` survives the sanitiser then becomes `p`" (unconditional)
   vs docstring `:649` "decimal points **in floats** become ``p``".
10. **BLAS override.** `sweep_runner.py:268` states the subprocess env pins threads to 1 *and* that
    "Users who need BLAS threading inside a single run can still override these vars externally
    before launching" — no respect-existing-value rule is documented.
11. **`path2output` relative vs absolute.** `sweep_runner.py:160` example "outputs/sweep/1e5_sfe001_n1e2/"
    vs `sweep_jobs.py:3` "each combo's path2output is absolute".
12. **"identical to what the in-process runner produces"** (`sweep_jobs.py:283`) vs the inputs it
    documents (`.exit_code` / `.duration` only) against the in-process report's stderr tail
    (`sweep_runner.py:306`, `:511`) and `metadata.json`-derived physics table (`:532`).
13. **−2 sentinel described two ways** — "the task was killed, e.g. wall-time/OOM, or was still
    running at collect time" (`sweep_jobs.py:250`) vs "task did not run or is still running"
    (`:324`).
14. **Progress-bar feature list** — module docstring (`sweep_runner.py:3`) omits elapsed time and
    current-simulation name that the class docstring claims (`:350`).

---

## 11. Claims that are unfalsifiable or too vague to check as written

- "Get keys and values in consistent order" (`sweep_parser.py:569`) — "consistent" with what?
- "Check if mantissa is close to an integer" (`:814`) — no tolerance given; `:822` "Non-integer
  mantissa - use compact form" gives no format.
- "we reserve ~55" (`:596`) — approximate; the resulting cap is never named.
- "Format omega: 14.1 -> \"14\"" (`:746`) — round or truncate is undetermined by the single example.
- "sanitised to SLURM-safe characters" (`sweep_jobs.py:204`) — set undefined.
- "Use scientific notation for large/small numbers" (`sweep_runner.py:195`) — no thresholds.
- "Each simulation is CPU-light" (`sweep_runner.py:268`) — unquantified.
- "the scheduler … restarts failed tasks independently" (`sweep_jobs.py:3`) — a claim about SLURM's
  behaviour, not this code's; SLURM does not requeue failed array tasks by default.
- "so they route through the single-run path (no nested sweep)" (`sweep_jobs.py:3`) — asserted from
  "the emitted files contain only scalar values", which is itself only asserted.
- "Run simulations in parallel batches" (`sweep_runner.py:3`) — "batches" is never defined.
- "Returns GMCValidationResult or None if validation cannot be performed" (`sweep_runner.py:77`) —
  the conditions under which validation "cannot be performed" are not listed.
- "identical to what the in-process runner produces" (`sweep_jobs.py:283`) — no field-level contract.

## 12. Notable absences (things a reader would expect the prose to state, and it does not)

- Exit code of the in-process sweep when some runs fail.
- Whether `runs.tsv` index is 0- or 1-based, and whether the file has a header.
- The `--array` range written into `submit_sweep.sbatch`, and the submit command itself.
- What `manifest.json` contains beyond being "self-describing".
- Behaviour on hitting the run-name length cap (truncate vs raise).
- Behaviour when a `tuple(...)` line is fed to `read_sweep_param` (which never mentions tuple mode).
- The default values that `:719` says are applied "if missing".
- Retry/resume semantics for a partially-collected job array.

Two `__main__` self-test blocks are documented — `sweep_parser.py:887` ("Testing" / "Quick test of
parsing functions", running through `:1043`) and `sweep_runner.py:629` ("Testing" / "Quick test" /
"Test progress bar") — and per the prose they are the only place several invariants (must-raise,
must-not-double-encode, length cap) are asserted at all.

---

```json
[
  {
    "id": "S12b-B-01",
    "file": "trinity/_input/sweep_runner.py",
    "line": 108,
    "class": "units",
    "severity": "S2",
    "claim": "The preflight validator's documented unit table claims the m_H -> Msun conversion factor for mu_convert is ~9.42e-58. The physical ratio m_H/M_sun is 8.414e-58 (proton mass: 8.410e-58). The documented factor is ~12% high and corresponds to a particle mass of 1.12 m_H. The sibling factor on the same list (cm^-3 -> pc^-3, ~2.94e+55) checks out exactly, which makes the mu_convert value look like a single-digit error (8.42 -> 9.42) rather than a different convention.",
    "evidence": "L108-112: '# Unit conversions matching trinity/_input/default.param unit annotations: # mCloud: [Msun] -> Msun (factor 1) # nCore, nISM: [cm**-3] -> pc⁻³ (factor ~2.94e+55) # rCore: [pc] -> pc (factor 1) # mu_convert: [m_H] -> Msun (factor ~9.42e-58)'",
    "expected": "~8.41e-58 (m_H = 1.6736e-24 g / M_sun = 1.989e33 g). Independently computed: 1.6735575e-24/1.98892e33 = 8.4144e-58; (1 pc)^3 = 2.9380e55 cm^3, confirming the neighbouring factor and the checking method.",
    "failure_scenario": "If the code implements 9.42e-58, every sweep combination's mean molecular mass entering the GMC preflight check is 12% too large, shifting derived quantities (cloud radius / density plausibility) and silently mis-classifying combinations near the rCloud_max validity boundary as valid or invalid. Because the docstring says the point of this code is that 'the preflight check matches what the actual simulation will see', a 12% offset means the preflight and the real run disagree exactly where it matters. If instead the code is correct and only the comment is wrong, the comment misleads the next person auditing the units.",
    "repro": "Compare the constant actually used for mu_convert in _validate_sweep_combination against convert2au's m_H->Msun factor on the read_param single-run path; they are claimed to be 'the same'. Then evaluate the GMC validator for a combination sitting just inside the rCloud_max bound and see whether the two paths agree.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-02",
    "file": "trinity/_input/sweep_parser.py",
    "line": 781,
    "class": "divergence",
    "severity": "S3",
    "claim": "format_scientific is documented three mutually inconsistent ways in the space of 45 lines: the docstring says scientific notation is used for values >= 100 and gives 0.01 -> '0.01' (i.e. NOT scientific); one comment says scientific is also used for 'very small numbers'; another says regular formatting applies to 'normal' numbers (1-99), which excludes 0.01 from the regular branch. The documented example and the documented rule cannot both hold.",
    "evidence": "L781-802: 'Format a number in compact scientific notation. Always uses scientific notation for values >= 100 to match TRINITY naming. Examples: - 1000000.0 -> \\'1e6\\' ... - 100.0 -> \\'1e2\\' - 0.01 -> \\'0.01\\''  vs L808-809: '# For values >= 100 or very small numbers, use scientific notation'  vs L825: '# For \"normal\" numbers (1-99), use regular formatting'",
    "expected": "One stated rule covering the whole real line, with the boundary for 'very small' named, matching the worked examples.",
    "failure_scenario": "This function feeds the mCloud and nCore tokens of the run-name contract. A reader building downstream tooling that parses run names (or a sweep over small nCore/mCloud values) cannot tell from the prose whether e.g. 0.01 renders as '0.01' or '1e-2', so a directory glob or name parser written against the docstring breaks on the sub-1 regime.",
    "repro": "Call format_scientific over 1e-4, 0.01, 0.5, 1, 99, 100, 5e6 and compare against each of the three documented rules.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-03",
    "file": "trinity/_input/sweep_parser.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The module docstring states tuple mode is used 'instead of Cartesian product' and that 'Parameters not in the tuple are fixed across all runs'. Elsewhere the same file documents a hybrid mode in which non-tuple sweep params are NOT fixed but are Cartesian-multiplied with the tuples. The top-level contract is wrong for one of the three supported modes, and hybrid mode is absent from the module docstring entirely.",
    "evidence": "L3-22: 'TUPLE MODE: Specify explicit parameter tuples instead of Cartesian product ... This runs only the specified combinations, not the full Cartesian product. Parameters not in the tuple are fixed across all runs.'  vs L489-507: 'Handles three modes: - Cartesian mode ... - Tuple mode ... - Hybrid mode: Tuple combinations × Cartesian product of sweep params'  and L258: 'Check if this is hybrid mode (tuple + sweep params).'",
    "expected": "The module docstring should document all three modes, or state the 'fixed across all runs' rule as applying to pure tuple mode only.",
    "failure_scenario": "A user reading the module docstring writes a param file with a tuple line plus a list-valued param, expecting the list to be ignored/fixed, and gets len(tuples) x len(list) runs instead - an order-of-magnitude larger sweep than intended, with a correspondingly larger SLURM array.",
    "repro": "Parse a file containing both 'tuple(mCloud, sfe) [1e5, 0.01] [1e7, 0.10]' and 'nCore [1e3, 1e4]', then compare count_combinations_from_config against the module docstring's claim.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-04",
    "file": "trinity/_input/sweep_jobs.py",
    "line": 250,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "failure_breakdown documents its swept-axis detector as 'a param that takes >1 but <n_runs distinct values'. By that rule, a one-dimensional sweep (the most common case) has its single axis excluded, because that axis has exactly n_runs distinct values. The same exclusion hits pure tuple mode whenever the tuples give each param a distinct value per row. The stated rule therefore defeats the stated purpose of the function in exactly the cases it advertises.",
    "evidence": "L250-257: 'Tally failed runs by each *swept* parameter (and by return code) so a regime-shaped failure -- e.g. \\'small clouds + high sfe + high cooling_boost_kappa\\' -- is visible at a glance instead of only a flat list of array indices. ... A \\'swept\\' axis is a param that takes >1 but <n_runs distinct values (so per-run identifiers like path2output/model_name are excluded).'",
    "expected": "The exclusion of per-run identifiers should key on something other than cardinality == n_runs (e.g. an explicit list of identifier keys, or the SweepConfig's own sweep_params/tuple_params, which the module already has access to via manifest.json).",
    "failure_scenario": "Operator runs a 12-point sweep over mCloud alone, 5 tasks fail, and the report shows only the return-code tally with no mCloud breakdown - the regime-shaped failure the function exists to surface is silently invisible. Worse, it degrades exactly as the sweep gets simpler, so it is likely to look fine on the wide multi-axis sweeps used to test it.",
    "repro": "Emit a single-axis sweep of N combinations, force some to fail, run collect_report and check whether the breakdown block names the swept axis.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-05",
    "file": "trinity/_input/sweep_jobs.py",
    "line": 100,
    "class": "other",
    "severity": "S3",
    "claim": "The runs.tsv contract is documented as 'one per line, index == SLURM array id' but the prose never states whether the index is 0-based or 1-based, whether the file carries a header line, or what --array range the emitted sbatch uses. This is the single load-bearing mapping between a scheduler task and a simulation, and it is documented ambiguously.",
    "evidence": "L100-125: 'Writes ``<jobs_dir>/params/<name>.param`` per combination, ``runs.tsv`` (``param_path<TAB>output_dir``, one per line, index == SLURM array id), a self-describing ``manifest.json``, and ``submit_sweep.sbatch``.'  Reinforced at L157-159: 'A submitted array reads runs.tsv by index, so regenerating in place would desync a running job.'",
    "expected": "An explicit statement, e.g. 'line N (1-based, no header) corresponds to SLURM_ARRAY_TASK_ID N' plus the emitted --array range, since sed/awk line addressing is 1-based while SLURM arrays are commonly 0-based.",
    "failure_scenario": "An off-by-one between the emitted --array range and the line-lookup in the sbatch either skips the first combination or runs one task with an empty line (running the wrong param, or nothing, while still writing an .exit_code sentinel). Both fail quietly: the report shows n-1 successes or one odd failure, not a mapping error.",
    "repro": "Read the emitted submit_sweep.sbatch --array range and its runs.tsv line-selection expression, and check both ends of the range against the file's first and last lines.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-06",
    "file": "trinity/_input/sweep_jobs.py",
    "line": 283,
    "class": "divergence",
    "severity": "S3",
    "claim": "collect_report claims the reports it writes are 'identical to what the in-process runner produces', while documenting that its only per-run inputs are the .exit_code and .duration sentinels. The in-process report is documented as containing the tail of each failed run's stderr, which no sentinel file can supply.",
    "evidence": "L283-289 (sweep_jobs.py): 'Reads ``<jobs_dir>/manifest.json`` and each run\\'s ``.exit_code`` / ``.duration`` sentinels, then writes ``sweep_report.txt`` / ``sweep_report.json`` into the sweep\\'s base output directory -- identical to what the in-process runner produces.'  vs sweep_runner.py L306: '# Extract last portion of stderr for error message' and L509-511: '# Indent error message' / '# Last 20 lines'",
    "expected": "Either 'same schema, with error text empty on the job-array path', or a documented mechanism by which stderr reaches the collector.",
    "failure_scenario": "A user debugging a failed HPC sweep expects the same error excerpts the laptop runner gives them, finds empty error fields, and concludes the runs failed without output rather than that the collector never had the stderr. Tooling written against 'identical' that requires the error field breaks on the cluster path only.",
    "repro": "Diff the JSON schema and populated fields of sweep_report.json produced by run_sweep against one produced by collect_report for the same failing sweep.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-07",
    "file": "trinity/_input/sweep_parser.py",
    "line": 687,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "generate_run_name guarantees that 'distinct combinations never collapse onto the same folder name', but the generic-suffix sanitiser it relies on is documented as a many-to-one mapping: every character outside [A-Za-z0-9.+-] is collapsed to '-' and '.' becomes 'p'. Two distinct string values therefore share a token (e.g. 'my run', 'my_run' and 'my-run' all -> 'my-run'), so the guarantee does not follow from the documented sanitiser.",
    "evidence": "L687-718: 'Any swept key without a curated slot in the name (i.e. not in ``_NAMED_RUN_NAME_KEYS``) gets a generic ``_{key}{value}`` suffix so distinct combinations never collapse onto the same folder name.'  vs L649-669: 'any char outside ``[A-Za-z0-9.+-]`` is collapsed to ``-`` so spaces, brackets, shell wildcards, and unicode never leak into the folder name' with example 'label=\"my run\" -> \"labelmy-run\"'  and L601-605: '`.` survives the sanitiser then becomes `p`'",
    "expected": "Either an injective encoding, a collision check across the generated name set, or a weakened claim ('distinct combinations that differ in sanitiser-visible characters').",
    "failure_scenario": "A sweep over a string-valued label with values 'run a' and 'run_a' produces two combinations mapping to one folder. The second run overwrites the first - which is precisely the silent overwrite the comment at L756-758 says this mechanism exists to prevent. Nothing errors; the report shows two successes and the outputs directory holds one result.",
    "repro": "Call generate_run_name for two combinations differing only in a string swept value whose sanitised forms coincide, and compare the returned names.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-08",
    "file": "trinity/_input/sweep_parser.py",
    "line": 596,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "A run-name length cap is documented (255-byte path component minus ~55 reserved) and a self-test says a huge value 'triggers the cap', but the prose never states what triggering the cap does - truncate or raise. If it truncates, the neighbouring 'never collapse onto the same folder name' guarantee is void for long values; if it raises, the exception is undocumented in generate_run_name's docstring, which lists no Raises at all.",
    "evidence": "L596-598: '# Total length budget for a generated run name. Most filesystems cap a single path component at 255 bytes; we reserve ~55 for the ``_modified`` tag, ``.param``/``_summary.txt`` siblings, and OS overhead.'  L998: '# Length guard: a synthetic huge value triggers the cap'  vs L687-718: 'so distinct combinations never collapse onto the same folder name' (and no Raises section)",
    "expected": "generate_run_name's docstring should name the cap value and state the overflow behaviour (truncate-with-hash / raise ValueError).",
    "failure_scenario": "A sweep over a long string parameter whose values share a long prefix generates names truncated to the same string; distinct combinations write into one directory and overwrite each other, with the sweep report showing all runs successful.",
    "repro": "Generate names for two combinations whose generic suffixes differ only past the documented budget, and compare.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-09",
    "file": "trinity/_input/sweep_parser.py",
    "line": 3,
    "class": "divergence",
    "severity": "S3",
    "claim": "The run-name contract - which downstream tooling and every output directory depend on - is stated in two incompatible forms in the same file. The module docstring and both combination generators advertise '{mass}_sfe{sfe}_n{nCore}', omitting the x100 zero-padded sfe encoding and all four suffix slots that generate_run_name documents.",
    "evidence": "L3-22: 'Generate run names following TRINITY convention: {mass}_sfe{sfe}_n{nCore}'; repeated verbatim at L489-507 and L547-562 ('output_name: Generated name following convention {mass}_sfe{sfe}_n{nCore}')  vs L687-718: 'Format: {mCloud}_sfe{sfe*100:03d}_n{nCore}[_profile][_PHII][_generic...]'",
    "expected": "One contract, stated once, with the suffix grammar - since the yields of generate_combinations are exactly the names on disk.",
    "failure_scenario": "A downstream reader/analysis script written against '{mass}_sfe{sfe}_n{nCore}' parses 'sfe010' as sfe=10 rather than 0.10, or fails on any name carrying a _PL-2/_noPHII/_Z0p5 suffix, mis-labelling published figures.",
    "repro": "Compare the names yielded by generate_combinations for a profile+PHII sweep against the module docstring's pattern.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-10",
    "file": "trinity/_input/sweep_parser.py",
    "line": 687,
    "class": "state",
    "severity": "S3",
    "claim": "generate_run_name's docstring states params is a 'Parameter dictionary containing mCloud, sfe, nCore' (a precondition), while the first line of its body comment says defaults are substituted when they are missing. The default values are never stated anywhere in the prose.",
    "evidence": "L687-718: 'params : dict  Parameter dictionary containing mCloud, sfe, nCore, and optionally dens_profile, densPL_alpha, densBE_Omega'  vs L719: '# Extract key parameters (with defaults if missing)'",
    "expected": "Either declare the three keys required and raise on absence, or document the fallback values (they end up in a directory name that is read back as ground truth).",
    "failure_scenario": "A sweep file that omits nCore (relying on default.param at runtime) produces folder names carrying an undocumented placeholder nCore that need not match the value default.param actually supplies at run time. The directory name then misreports the physics of its own contents, and any analysis keyed on the name is wrong.",
    "repro": "Call generate_run_name with a params dict lacking nCore and compare the name's n<...> token against the value default.param supplies on the single-run path.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-11",
    "file": "trinity/_input/sweep_parser.py",
    "line": 569,
    "class": "other",
    "severity": "S3",
    "claim": "The ordering of Cartesian combinations is documented only as 'consistent order'. The prose never says whether keys are taken in file order or sorted, nor which axis varies fastest - yet sweep_jobs pins SLURM array ids to that ordering ('index == SLURM array id'), making the order part of an external contract. By contrast, the ordering of generic name suffixes IS pinned ('Sorted for deterministic ordering'), showing the file knows how to state such a rule when it means it.",
    "evidence": "L569: '# Get keys and values in consistent order'  vs L759: '# Sorted for deterministic ordering regardless of sweep-file key order.'  and sweep_jobs.py L100-125: '``runs.tsv`` (``param_path<TAB>output_dir``, one per line, index == SLURM array id)'",
    "expected": "A stated ordering (e.g. 'sweep keys in sweep-file order, last key varies fastest, stable across Python versions and dict insertion order') because array id -> combination is a published mapping.",
    "failure_scenario": "An operator reruns a subset of a large array by index ('--array=17,22') after regenerating the bundle from an edited param file; if ordering depends on file key order or dict iteration, index 17 is now a different combination and the rerun silently overwrites the wrong output directory.",
    "repro": "Emit bundles from two param files with identical axes listed in different key order and diff runs.tsv line by line.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-12",
    "file": "trinity/_input/sweep_runner.py",
    "line": 268,
    "class": "state",
    "severity": "S3",
    "claim": "The same comment claims the runner pins BLAS/LAPACK thread counts to 1 in every subprocess environment AND that users can still opt out by exporting those variables before launching. No respect-pre-existing-value rule is documented, so the two halves of the sentence are in tension.",
    "evidence": "L268-275: '# Build environment for subprocess: limit BLAS/LAPACK threads to 1. ... # Users who need BLAS threading inside a single run can still # override these vars externally before launching.'",
    "expected": "State the precedence explicitly: either 'the pin is unconditional' or 'existing OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/VECLIB_MAXIMUM_THREADS in the parent environment are preserved'.",
    "failure_scenario": "A user on a big node exports OMP_NUM_THREADS=8 expecting the documented escape hatch, gets single-threaded runs anyway, and concludes the machine or NumPy build is at fault. The failure is a silent performance regime, not an error.",
    "repro": "Export OMP_NUM_THREADS=8, run a two-combination sweep, and inspect the child environment actually passed to the subprocess.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-13",
    "file": "trinity/_input/sweep_parser.py",
    "line": 339,
    "class": "regime",
    "severity": "S3",
    "claim": "The rule that a single-element list is demoted to a base parameter is documented once, as an inline comment inside read_sweep_param. It contradicts that function's own docstring ('sweep_params : dict  Parameters with list values'), and the parallel categorisation step in read_sweep_config carries no such note - so the file documents the same grammar rule for one reader and not the other.",
    "evidence": "L339: '# Single-element list is treated as base param'  vs L263-299: 'Separates parameters into: - base_params: Single values (constant across all runs) - sweep_params: List values (will generate combinations)'  vs L445 (in read_sweep_config): '# Categorize based on whether it\\'s a list' (no single-element note)",
    "expected": "The demotion rule stated in the docstring of every reader that implements it, or removed from whichever reader does not.",
    "failure_scenario": "If the two readers disagree, a file with 'mCloud [1e7]' yields sweep_params={} via one entry point and sweep_params={'mCloud': [1e7]} via the other. The run name then differs (a one-element axis is still a 'swept key', so it would gain a generic suffix under one path and not the other), so the same param file produces differently-named output directories depending on which reader the caller used.",
    "repro": "Feed one file containing a single-element list to both read_sweep_param and read_sweep_config and compare base/sweep partitioning and the resulting run names.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-14",
    "file": "trinity/_input/sweep_parser.py",
    "line": 601,
    "class": "divergence",
    "severity": "S4",
    "claim": "The '.' -> 'p' substitution is scoped differently in two adjacent blocks: the charset comment says any surviving '.' becomes 'p', the function docstring says only decimal points in floats do.",
    "evidence": "L601-605: 'Keeps alphanumerics plus the printable scaffolding actually needed by the existing convention (`.` survives the sanitiser then becomes `p`; ...)'  vs L649-669: 'decimal points in floats become ``p`` (mirroring the fact that the curated suffixes never contain ``.`` or ``_``)'",
    "expected": "One scope. It matters because it decides whether a string value like 'v1.2' becomes 'v1p2' or keeps the dot, and the charset comment explicitly allows '.' through the sanitiser.",
    "failure_scenario": "A folder name retains a '.' in a generic suffix, breaking the stated invariant that curated suffixes never contain '.', and confusing any downstream splitter that treats '.' as an extension boundary (the same comment notes '.param'/'_summary.txt' siblings share the stem).",
    "repro": "Call _generic_suffix_token with a string value containing a dot and with a float, and compare against both statements.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-15",
    "file": "trinity/_input/sweep_parser.py",
    "line": 610,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The stated policy ('sweeping filepath-typed parameters is not supported') is strictly broader than the stated detection rule (values containing '/', '\\\\' or '..'), and the same docstring exempts all non-string values from the check on the grounds that 'numbers/bools can\\'t contain unsafe chars' - a justification that covers numbers and bools but is asserted for every non-str type.",
    "evidence": "L610-625: 'Two categories are hard-rejected (no sanitisation): 1. Anything that looks like a filepath (``/``, ``\\``, or ``..``). The suffix would otherwise inject path separators into the run folder, silently creating nested directories or escaping the sweep root. Per project policy: sweeping filepath-typed parameters is not supported ... Non-string values bypass this check (numbers/bools can\\'t contain unsafe chars).'",
    "expected": "Either reject by parameter identity (the known filepath-typed keys, e.g. path2output / SPS and cooling table paths) rather than by value shape, or narrow the stated policy to match the value-shape test.",
    "failure_scenario": "A bare-filename filepath value ('mytable.dat') passes the check, becomes a run-name token, and the sweep proceeds with a filepath axis the policy says is unsupported - the exact case the policy meant to block, admitted because it happens to contain no separator.",
    "repro": "Call _reject_unsafe_sweep_value with 'mytable.dat', with a pathlib.Path containing a separator (non-str, so claimed to bypass), and with '../x'.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-16",
    "file": "trinity/_input/sweep_runner.py",
    "line": 160,
    "class": "divergence",
    "severity": "S3",
    "claim": "The two sweep back-ends document the path2output contract differently: the in-process generator's worked example is a relative path, while the job-array module asserts every combination's path2output is absolute (and rests its working-directory independence claim on that).",
    "evidence": "sweep_runner.py L160-176: 'run_output_dir : str  Output directory for this specific run (e.g. outputs/sweep/1e5_sfe001_n1e2/)'  vs sweep_jobs.py L3-21: 'Inputs are located relative to the package (_REPO_ROOT-anchored) and each combo\\'s path2output is absolute, so tasks are independent of the working directory the array runs in.'",
    "expected": "One documented rule for the .param that both back-ends emit, since both are described as routing through the same single-run path.",
    "failure_scenario": "If the in-process path really writes a relative path2output, a worker subprocess launched with a different cwd writes its outputs to a second, unexpected tree - and the collector, looking under the absolute base directory, reports the run as having produced nothing.",
    "repro": "Inspect the path2output line of a .param emitted by generate_param_file and of one emitted by emit_jobs for the same combination.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-17",
    "file": "trinity/_input/sweep_runner.py",
    "line": 195,
    "class": "divergence",
    "severity": "S4",
    "claim": "Two independent, differently-documented number formatters act on the same values: format_scientific (with a stated >= 100 rule) produces the run-name tokens, while generate_param_file uses an unspecified 'large/small numbers' rule to write the values into the .param the simulation reads.",
    "evidence": "sweep_runner.py L195: '# Use scientific notation for large/small numbers'  vs sweep_parser.py L781-802: 'Always uses scientific notation for values >= 100 to match TRINITY naming.'",
    "expected": "One documented formatting rule, or an explicit note that the name encoding and the file encoding are deliberately independent.",
    "failure_scenario": "A value that rounds differently under the two rules gives a directory named for one number and a .param containing another (e.g. name says n1e4, file says 10000.0000001 or 1e+04), so provenance from folder name to actual input is no longer exact - and a name-based analysis mis-attributes the run.",
    "repro": "For a sweep value with a non-integer mantissa, compare the n<...> token in the folder name against the nCore line inside that folder's .param.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-18",
    "file": "trinity/_input/sweep_runner.py",
    "line": 3,
    "class": "other",
    "severity": "S4",
    "claim": "Nothing in the slice's prose states the process exit code of an in-process sweep. The only failure-handling contract given is 'Continue on failure, collect and report all errors at end'. The job-array path by contrast documents per-task .exit_code sentinels and a -2 magic value, so the two back-ends are documented at different levels of precision on exactly the property CI and shell scripts branch on.",
    "evidence": "L3-14: 'Features: - Run simulations in parallel batches using ProcessPoolExecutor - Progress bar with completed/total, percentage, and ETA - Continue on failure, collect and report all errors at end - Generate human-readable and JSON reports'  vs sweep_jobs.py L250-257: 'Return code -2 = no sentinel (the task was killed, e.g. wall-time/OOM, or was still running at collect time), not a sim-level crash.'",
    "expected": "A stated rule, e.g. 'exits non-zero if any run failed' or 'always exits 0; consult sweep_report.json'.",
    "failure_scenario": "A CI job or shell wrapper treats a sweep with silent partial failures as a success (or vice versa) because the exit convention had to be guessed.",
    "repro": "Run a two-combination sweep with one combination guaranteed to fail and inspect $?.",
    "confidence": "high"
  },
  {
    "id": "S12b-B-19",
    "file": "trinity/_input/sweep_jobs.py",
    "line": 3,
    "class": "citation",
    "severity": "S4",
    "claim": "The module docstring attributes automatic restart of failed tasks to the scheduler as a property of job arrays. SLURM does not requeue tasks that exit non-zero by default; requeue applies to node failure / preemption or requires an explicit --requeue policy. The claim is about an external tool, is not implemented by this code, and the prose documents no --requeue flag among the flags it does mention (%K, --job-name).",
    "evidence": "L3-21: 'On an HPC cluster the conventional pattern is instead a scheduler job array: one array task per combination, so the scheduler packs them across many nodes, handles fair-share, and restarts failed tasks independently.'",
    "expected": "Either drop the restart claim or document the sbatch flag that provides it; note that the failure path this module actually documents is a -2 sentinel for tasks that never produced one.",
    "failure_scenario": "An operator reads 'restarts failed tasks independently', does not build a resubmission step, and a wall-time-killed subset of a 500-task array is simply missing - surfacing only as -2 return codes at collect time.",
    "repro": "Read the emitted submit_sweep.sbatch header for any --requeue / retry directive.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-20",
    "file": "trinity/_input/sweep_parser.py",
    "line": 263,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The file documents two readers for the same file format. read_sweep_param's docstring describes only the base/sweep split and never mentions tuple syntax, while read_sweep_config documents all modes and is what the combination generators are documented to consume ('config : SweepConfig  Configuration from read_sweep_config()'). read_sweep_param reads as a superseded API whose documented contract is silent about a syntax that can appear in the very files it parses.",
    "evidence": "L263-299: 'Read a sweep-enabled parameter file. Separates parameters into: - base_params ... - sweep_params ...' (no tuple mention)  vs L355-395: 'Read a sweep-enabled parameter file and return a SweepConfig. Supports both modes: - Cartesian mode ... - Tuple mode ...'  and L539: '# Pure Cartesian mode: use existing logic'",
    "expected": "Either document read_sweep_param's behaviour on a tuple-bearing file, or note it as the legacy Cartesian-only entry point.",
    "failure_scenario": "A caller still on read_sweep_param is handed a tuple-mode file; the tuple lines either parse as a parameter named 'tuple(mCloud,' or are skipped, and the sweep silently runs the base params only - one run instead of the intended set, with no error.",
    "repro": "Pass a tuple-mode param file to read_sweep_param and inspect the returned dicts.",
    "confidence": "low"
  },
  {
    "id": "S12b-B-21",
    "file": "trinity/_input/sweep_parser.py",
    "line": 41,
    "class": "other",
    "severity": "S4",
    "claim": "The documented grammar makes bracket syntax unconditionally mean 'sweep axis' (precedence puts list first, ahead of every other type), and documents no escape. There is no stated way to give a parameter a literal bracketed string value, nor to express a parameter that is genuinely list-valued in the physics; the only documented demotion is the single-element case.",
    "evidence": "L41-67: 'Extended to support list syntax: [val1, val2, val3]  Precedence: list → boolean → number → fraction → string'  and L339: '# Single-element list is treated as base param'  (no quoting/escaping rule anywhere in the file)",
    "expected": "Either a documented escape (quotes, a backslash, an explicit sweep: prefix) or an explicit statement that list-valued parameters are unsupported in sweep files - the latter being the current de facto rule per the prose.",
    "failure_scenario": "If any TRINITY parameter is legitimately list-valued, it cannot be set in a sweep file at all: it silently becomes an N-fold axis and multiplies the run count, rather than being passed through. The user sees N runs where they expected one.",
    "repro": "Set a genuinely list-valued parameter in a sweep .param and check count_combinations_from_config.",
    "confidence": "medium"
  },
  {
    "id": "S12b-B-22",
    "file": "trinity/_input/sweep_parser.py",
    "line": 887,
    "class": "other",
    "severity": "S4",
    "claim": "Both modules carry ad-hoc __main__ 'Testing' blocks, and per the prose these are the only place several stated invariants are asserted - the ValueError hard-rejects, the no-double-encoding rule for curated keys, the length cap, and the deterministic suffix ordering. Checks that live in a __main__ block do not run under the project's pytest suite.",
    "evidence": "L886-888: '# =====' / '# Testing' / '# ====='; L891: '# Quick test of parsing functions'; L953: '# A curated key passed in swept_keys must NOT be double-encoded'; L976: '# Hard-reject cases: unsafe values must raise ValueError'; L998: '# Length guard: a synthetic huge value triggers the cap'; L1024: '# Compare with tolerance for floats'; sweep_runner.py L632-635: '# Quick test' / '# Test progress bar'",
    "expected": "Per project convention, non-trivial logic leaves a runnable check behind in the pytest suite (test_*.py), not in a module __main__ block.",
    "failure_scenario": "A regression in the sanitiser or the unsafe-value rejection ships green: nothing in CI executes these assertions, and the failure surfaces later as a path-separator injection or a folder-name collision in a real sweep.",
    "repro": "Check whether equivalent assertions exist in the pytest suite for _reject_unsafe_sweep_value, _generic_suffix_token and the length cap.",
    "confidence": "medium"
  }
]
```
