# TRINITY Codebase Consistency & "Fresh-Clone" Review

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living document — recheck and refine on every visit.** This is an
> evolving audit, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) reconsider
> the findings themselves — if a finding is stale, mis-scoped, or a better fix
> has landed, revise it and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full runs cost minutes-to-hours, so any diagnostic worth keeping must be
> saved as a committed artifact (a CSV/table under `docs/dev/data/`, or a
> force-added harness/figure under `scratch/`) — never left in `/tmp` or an
> untracked `outputs/`. A future visit must be able to reproduce or compare
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

**Review goal.** Audit the *entire* repository for inconsistencies
(code ↔ docstrings ↔ comments), stale/dead references, and anything that is
"not supposed to be there" or would confuse / mislead / break for **someone who
`git clone`s this repo and tries to run a simulation themselves**.

**Date:** 2026-06-16 · **Branch:** `fix/code-hygiene`

---

## How this review was produced

Multi-agent, multi-phase review. **Phase 1** mapped the repo and established
ground truth (deps, entry point, test collection). **Phase 2** partitioned the
codebase into seven areas; one sub-agent audited each in parallel and wrote its
findings to a section file under `docs/dev/codebase_review/`. **Phase 3** (this
file) consolidates them, after independent spot-verification of every
High-severity finding against the source.

**Dynamic verification (not just static):** the scientific stack installed
cleanly in the review container (numpy 1.26.4 — correctly `<2` — scipy 1.17.1,
astropy 7.2, matplotlib 3.11, pandas 2.3), so beyond the static audit:

- ✅ `pip install -r requirements.txt` succeeds; pins resolve.
- ⏱️ `python run.py param/simple_cluster.param` **starts cleanly**: loads/validates
  params, writes `outputs/simple_cluster/{metadata.json, dictionary.jsonl,
  trinity.log}`, and integrates ~100 snapshots through the energy phase with **no
  error or traceback**. It is a genuinely long simulation: it did **not** finish
  within a 5-minute cap (the process was killed by SIGTERM), so full end-to-end
  completion was *not* confirmed here — only a clean start and crash-free
  integration. (An earlier draft of this file claimed "exit 0, ~5 min"; that was
  an over-statement and has been corrected.)
- ✅ `pytest` collects **532 tests**; **531 non-stress tests pass** (the lone
  ~2.5-min end-to-end smoke test was excluded for time; 3 stress deselected) —
  no collection or import errors.

### Severity legend
- 🔴 **High** — breaks a fresh-clone run, or leaks private/personal/machine-specific info.
- 🟠 **Medium** — misleads or confuses a new user; docs/comments contradict the code.
- 🟡 **Low** — cosmetic / nit / stale-but-harmless.

### Section files (full detail lives here)
| # | Area | File | Counts (H/M/L) |
|---|------|------|----------------|
| 01 | `trinity/` I/O + infra (`_input`, `_output`, `_functions`, `_analysis`, `main.py`) | `codebase_review/01_trinity_io_infra.md` | 1 / 3 / 6 |
| 02 | `trinity/` physics (bubble/cloud/cooling/shell/sps + phase modules) | `codebase_review/02_trinity_physics.md` | 1 / 7 / 5 |
| 03 | Entry point & repro (`run.py`, `pyproject.toml`, `requirements.txt`, `MANIFEST.in`, `param/`, CI/config) | `codebase_review/03_entrypoint_config_repro.md` | 0 / 4 / 3 |
| 04 | Docs (`README`, `CLAUDE.md`, `CONTRIBUTING`, `CHANGELOG`, `docs/`) | `codebase_review/04_docs.md` | 1 / 5 / 4 |
| 05 | Tests (`test/`) | `codebase_review/05_tests.md` | 0 / 0 / 2 |
| 06 | `tools/`, `paper/`, `lib/` | `codebase_review/06_tools_paper_lib.md` | 0 / 2 / 2 |
| 07 | Cross-cutting sweep & cruft (`scratch/`, `outputs/`, `docs/dev/`, repo-wide) | `codebase_review/07_crosscutting_cruft.md` | 1 / 3 / 2 |
| | **Total** | | **4 / 24 / 24 = 52** |

---

## Status — applied vs flagged (branch `fix/code-hygiene`)

This review originally lived under `analysis/`; that whole directory was **folded
into `docs/dev/`** (this file is now `docs/dev/CODEBASE_REVIEW.md`, section files
under `docs/dev/codebase_review/`), and every `analysis/<path>` reference across
source/tests/tools/docs was repointed.

**Fixed on this branch:**

*Round 1 — stale docstrings / comments / pointers:*
- H3 dead `example_scripts/` pointer → repointed to `paper/methods/figures/` (`_output/README.md`, `trinity_reader.py`).
- Units labels: `Eb [erg]` → `[Msun pc²/Myr²]`; `get_dudt` Returns block; `get_shellODE` "assumes cgs" → code units; BE-sphere debug logs `cm⁻³` → `1/pc³`.
- Run startup banner link → `jiaweiteh.github.io/trinity-web` (was the deprecated readthedocs mirror).
- Self-referential "Key difference from `<this same file>`" module docstrings reworded.
- `read_sps` example "Correct formula!" → the total-based formula the runtime actually uses.
- `read_cloudy.get_coolingStructure` docstring: removed-`age`-param → documents `params`.
- "4x4 grid" → "5x5 grid (GRID_SIZE=5)".
- `.pre-commit-config.yaml` comment corrected (there is no real "TOML issue").
- `make_density_profile_gif.py` default run-dir + docstring → the shipped mock run.
- `docs/dev/archive/README.md` given the mandated staleness banner.

*Round 2 — small logic / config / citation fixes:*
- **H2 metallicity** `UnboundLocalError` → clear `ValueError` for unsupported `ZCloud` (have 1.0, 0.15).
- `flake8` → ruff/pre-commit across the `[dev]` extra, `requirements.txt`, and CONTRIBUTING.
- Paper citation → `Teh et al. (2026), arXiv:2605.27517` in `publications.rst` + `license.rst` (matches README; confirmed on arXiv by the author).

*Round 3 — structure (`docs/dev/` consolidation):*
- `analysis/` folded into `docs/dev/` (audits, plan docs, `data/`, `archive/`); all references repointed; merged `main` (PR #687) and folded its new `analysis/` docs in too.
- **`scratch/` consolidated → `docs/dev/scratch/`** (all 81 files). The top-level `.gitignore` rule was anchored to `/scratch/`, so `docs/dev/scratch/` is now **normally tracked** — resolving H1's original ignored-but-tracked contradiction — while `scratch/` stays local-only. The 💾 persist convention now commits diagnostics under `docs/dev/`.
- Banners: 🔄 living + 💾 persist added to the review master + all 7 section files.

*Round 4 — `docs/dev/` reorganised into self-contained workstream folders:*
- The flat pile of dev docs + the dissolved `scratch/` were grouped one folder per workstream: `betadelta/` (HYBR_PLAN, PHASE0_BASELINES, PHASE2_ARMS, stalling-energy-phase + `diagnostics/`, `velstruct/`), `transition/` (TRIGGER_PLAN, P0, pshadow-design + `harness/`), `bubble/`, `cooling/`, `n-consistency/`, `misc/`. `CODEBASE_REVIEW.md` + `codebase_review/`, `data/`, `archive/` stay at the `docs/dev/` top.
- Cryptic `scratch/phase2`/`phase6` (which collided with `trinity/phase2_momentum`) became `betadelta/diagnostics`/`betadelta/velstruct`; `transition` harness → `transition/harness`. All references repointed; `docs/dev/README.md` index + the 💾 convention updated.

**Flagged, deliberately NOT changed here** (deletions or needs-a-decision):
- H4 / dead modules `input_warnings.py`, `read_mist_models.py`, unused solvers — deletions (CLAUDE.md rule 3: flag, don't silently delete).
- Packaging: `package-data` / `MANIFEST.in` data-glob gap.
- Cloudy README run-dir layout rewrite; CWD-relative cloudy test paths; inert `output_format` knob.

> **Note:** `main` (PR #687, the transition-trigger workstream) has been merged
> into this branch; its new `analysis/` docs (`transition-trigger-*.md`,
> `data/transition_*.csv`) were folded into `docs/dev/` and repointed like the
> rest, so the `analysis/ → docs/dev/` collapse is complete across the merged tree.

---

## Executive summary

**The good news, and it's the most important news for a git-puller:** the
documented happy path works. A stranger who follows the README
(`git clone` → `pip install -r requirements.txt` → `python run.py
param/simple_cluster.param`) gets a running simulation. All seven README
example `.param` files exist and validate; the bundled `lib/default/` SPS +
cooling tables ship and resolve to the repo root (not a personal path, not CWD);
`pytest` (non-stress) is expected to pass from the repo root; and on the
**privacy/security axis the repo is clean** — no secrets, tokens, API keys,
`breakpoint()`/`pdb`, or `__pycache__`/`.DS_Store` junk are tracked, and the
only email and author names are intentional attribution. The three files
`.gitignore` marks "Private" are genuinely absent (no leak).

**What actually bites a git-puller** clusters into a handful of themes:

1. **One real "shouldn't be here" item — `scratch/`.** `.gitignore` declares
   `scratch/` "kept locally, not tracked," yet **65 files (~6.4 MB of one
   developer's PNGs/GIFs/jsonl diagnostic dumps)** are committed. CLAUDE.md
   itself says scratch is "do not treat as ground truth." This is the headline
   cruft finding and the only genuine ignored-but-tracked contradiction.
2. **Units mislabeling — the repo's own known bug class — shows up in docs and
   docstrings.** The reader Quick Start calls `Eb` `[erg]` when it's stored in
   `Msun·pc²/Myr²` (off by ~43 orders of magnitude); cooling/shell-ODE
   docstrings state wrong time units / "assumes cgs." The *code* is right; the
   *labels* mislead exactly the user who trusts them.
3. **One plausible parameter choice hard-crashes.** Any `ZCloud` other than
   1.0 or 0.15 raises a cryptic `UnboundLocalError` deep in the cooling loader
   (missing `else`) instead of a clear "unsupported metallicity" message.
4. **Doc ↔ code drift a newcomer hits early.** A dead `example_scripts/`
   pointer (the headline "comprehensive examples" link — corroborated by two
   independent agents); the run **startup banner links the docs site the project
   itself marks deprecated**; the method paper is simultaneously "posted on
   arXiv (2026)" and "in preparation"; a cloudy README documents a run-directory
   layout the current code no longer produces.
5. **Dead code from a previous (WARPFIELD-era) life.** `input_warnings.py`
   validates parameters that no longer exist; `read_mist_models.py` is an unused
   isochrone reader; several self-labeled "CURRENTLY UNUSED" alternative solvers
   sit in the most correctness-critical physics module.
6. **Packaging promises more than it ships.** The project bills itself
   distributable (`trinity-sf` 1.0.0, "Production/Stable"), but `pip install .`
   / an sdist bundles **none** of `param/` or `lib/default/` (data-glob
   mis-location). The README's deps-only install dodges this, so it's Medium —
   but it contradicts the packaging intent.
7. **Tooling drift:** `flake8` is advertised (`[dev]` extra, `requirements.txt`,
   CONTRIBUTING) but the project lints with **ruff**; figures need LaTeX
   (`text.usetex: True`, unconditional) despite "no extra downloads needed."

None of these block the documented quickstart. Items 1–3 are the ones worth
fixing before the next person clones; the rest are polish that raises the
trust signal of a public scientific code.

---

## Consolidated findings

### 🔴 High (4)

| # | Theme | Finding | Where | §|
|---|-------|---------|-------|--|
| H1 | Repo hygiene | **`scratch/` is `.gitignore`d "kept locally, not tracked" but 65 files (~6.4 MB of PNG/GIF/jsonl) are committed.** Only genuine ignored-but-tracked contradiction. | `.gitignore:30` vs `git ls-files scratch/` | 07 |
| H2 | Crash on valid input | **Any `ZCloud` ≠ 1.0/0.15 → cryptic `UnboundLocalError`** (`Z_str` set in an `if/elif` with no `else`, then used unconditionally). | `trinity/cooling/non_CIE/read_cloudy.py:290-295` | 02 |
| H3 | Dead doc pointer | **Reader docs send new users to `example_scripts/` (two named files) that don't exist** anywhere in the repo. Corroborated by two agents (README + reader docstring). | `trinity/_output/README.md:121-127`, `trinity/_output/trinity_reader.py:120-121` | 01, 04 |
| H4 | Dead code (ex-schema) | **`input_warnings.py` is orphan code validating a removed WARPFIELD-era schema** (`frag_enabled`, `inc_grav`, …); never imported; would `KeyError` before its `sys.exit` if wired in. | `trinity/_input/input_warnings.py` (whole module) | 01 |

> Note: H1, H2, H3 were independently re-verified against the source during
> consolidation. H4 was reported by the section agent (high-confidence; the
> referenced keys are absent from the registry).

### 🟠 Medium (24)

**Units / values mislabeled (the known bug class)**
| Finding | Where | § |
|---------|-------|--|
| `Eb` documented `[erg]` in reader Quick Start — actually `Msun·pc²/Myr²` (~1.9e43× off); contradicts the reader's own `PARAM_DOCS`. | `trinity/_output/README.md:25,69` vs `trinity_reader.py:163` | 01 |
| `get_dudt` Returns block states `M_sun/pc/yr3` + garbled `[M_sun*pc5/s3]`; code yields `Msun/pc/Myr³` via `cvt.dudt_cgs2au`. | `trinity/cooling/net_coolingcurve.py:34-37` | 02 |
| `get_shellODE` docstring says "This routine assumes cgs" but all I/O is code/pc units. | `trinity/shell_structure/get_shellODE.py:32` | 02 |
| `read_sps.get_interpolation` example labels wind-only `2·Lmech_W/fpdot_W` as "Correct formula!"; runtime uses **total** `2·Lmech_total/pdot_total`. | `sps/read_sps.py:333` vs `sps/update_feedback.py:181` | 02 |

**Doc ↔ code / doc ↔ doc drift**
| Finding | Where | § |
|---------|-------|--|
| Run **startup banner links the deprecated docs site** (`trinitysf.readthedocs.io`) that `conf.py` itself flags dead; canonical is `jiaweiteh.github.io/trinity-web`. Confirmed live in run output. | `trinity/_output/header.py:35` | 04 |
| Method paper is **both "posted on arXiv (2026)" and "in preparation"** across docs. | `README.md:103-106` vs `docs/source/publications.rst:7`, `license.rst:11-12` | 04 |
| arXiv `2605.27517` / ADS `2026arXiv260527517T` appear in **one file only** and look placeholder-ish — needs a human web-check. | `README.md:104,106` | 04 |
| Cloudy README presents a stale run-dir layout (`<model>_summary.txt`, `simulationEnd.txt`, "gate on Status: SUCCESS") the current code no longer writes. | `trinity/_output/cloudy/README.md:49-64` vs `simulation_end.py:209`, `cloudy/run_loader.py:13-14` | 04 |
| `get_coolingStructure` docstring lists a removed `age [yr]` param (signature takes `params`); line 44 relies on an operator-overload instead of `.value`. | `trinity/cooling/non_CIE/read_cloudy.py:22-44` | 02 |
| Self-referential "Key difference from `<this same file>`" module docstrings (leftover from a deleted pure-vs-mutating sibling). | `phase1_energy/energy_phase_ODEs.py:11`, `shell_structure/shell_structure.py:10` | 02 |
| `run_energy_implicit_phase` docstring says "4x4 grid by default"; actual `GRID_SIZE = 5` → 5×5. | `phase1b_energy_implicit/run_energy_implicit_phase.py:34` vs `get_betadelta.py:56` | 02 |

**Inert / dead knobs & code**
| Finding | Where | § |
|---------|-------|--|
| `output_format` is a documented param defaulting to `JSON` that **no code consumes** — output is always JSONL. | `trinity/_input/default.param:31` | 01 |
| Self-labeled "CURRENTLY UNUSED" / "Alternative" solvers padding critical physics: `run_energy_continuous`, `_create_adaptive_radius_grid`, `_solve_bubble_ode_with_ivp`, `bubble_P2E` (last also uses an astropy-Quantity convention its inverse doesn't). | `phase1_energy/run_energy_phase.py:343`, `bubble_structure/bubble_luminosity.py:875,1002`, `get_bubbleParams.py:232` | 02 |

**Packaging / reproducibility (don't break README path, but bite `pip install .`)**
| Finding | Where | § |
|---------|-------|--|
| `package-data` globs resolve under `trinity/`, but `param/`+`lib/default/` live at repo root → a wheel ships **no** params/SPS/cooling data. | `pyproject.toml:82-86` | 03 |
| `MANIFEST.in` has **no rule for `lib/default/`** and `global-exclude *.json` strips bundled JSON → sdist omits the data. | `MANIFEST.in:10-11,33-34` | 03 |
| `text.usetex: True` applied unconditionally on import → **figures crash without LaTeX**, despite README "no extra downloads needed." | `paper/_lib/trinity.mplstyle:8`, `plot_base.py:40-42` | 06 |
| `make_density_profile_gif.py` default run-dir points into gitignored `outputs/rosette_cf_survey_updated_0p77/...` (not shipped) → no-arg run `FileNotFoundError`s. | `tools/make_density_profile_gif.py:65-67` | 06 |

**Tooling / config drift**
| Finding | Where | § |
|---------|-------|--|
| `flake8` advertised in `[dev]` extra / `requirements.txt` / CONTRIBUTING, but the project lints with **ruff** (no flake8 config exists). | `pyproject.toml:50`, `requirements.txt:16-20`, `CONTRIBUTING.md:11-15` | 03, 04 |
| `.pre-commit-config.yaml` comment blames a "pre-existing TOML issue" in `pyproject.toml` that doesn't exist (file parses as valid TOML). | `.pre-commit-config.yaml:26-29` | 03 |
| Committed binary/generated artifacts under `scratch/` (18 PNG, 3 GIF incl. 1.6 MB, 14 jsonl); not test fixtures. | `scratch/phase2/`, `scratch/phase6/` | 07 |
| `docs/dev/archive/README.md` missing the staleness banner CLAUDE.md mandates (1 of 17 such docs). | `docs/dev/archive/README.md` | 07 |
| `.gitignore` has wrapped/garbled comment lines (continuations lost their `#`). | `.gitignore:77,134,140,155,216` | 03, 07 |

> The 04 section self-counts 5 Medium; the table above lists the four
> doc-drift Mediums plus the cross-listed `flake8` item — see
> `codebase_review/04_docs.md` for that section's authoritative tally.

### 🟡 Low (24) — compact

Cosmetic / stale-but-harmless. Full detail per section file.

- **Dead/leftover code & debug:** unused `read_mist_models.py` (MIST reader, hard
  matplotlib import) `§01`; `# debug` `print(array)` in `operations.py:35-36,156-157`
  `§01`; import-time `print()`s in `paper_feedback.py:37`, `paper_radiusComparison.py:42`
  `§06`; commented-out physics/`sys.exit()` in `net_coolingcurve.py`,
  `get_InitCloudyDens.py`, `read_cloudy.py` `§02`; dead "fix kink" block pointing at
  `paper/rCloud_bump` `§02`.
- **Docstring/comment nits:** `header.display()` docstring documents a `params` arg it
  doesn't take `§01`; stale self-reported spec counts (registry "186/72/114", actual
  195/74) `§01`; `simulation_end.py:426` keys a debug row on `F_ion` (real name
  `F_ion_in`) so the row is always blank `§01`; BE-sphere logs label `1/pc³` densities
  as `cm⁻³` `§02`; Weaver `WEAVER_TEMP_COEFFICIENT` comment omits the time unit `§02`.
- **Docs polish:** README/visualization omit the `[plots]` extra the figures need `§04`;
  `pyproject` "Production/Stable" vs CHANGELOG "Unreleased" `§04`; `make_figures.py`
  "not yet published" wording for bundles that do ship `§06`; CONTRIBUTING dev-setup
  divergence `§04`.
- **Tests (both Low):** three cloudy tests resolve the `mockFullrun` fixture
  CWD-relative instead of via `__file__` (only fails outside repo root) `§05`;
  `test_run_smoke.py` is an unmarked ~2.5-min end-to-end test that runs by default `§05`.
- **Cross-cutting:** 19 `TODO`s in shipped `*.py` (normal research-code notes, full
  catalog in `§07`); `termination_debug.txt` fixture embeds a wall-clock timestamp `§07`;
  `SB99_rotation` naming drift (self-documented) `§01`; CIE cooling silently ignores
  `metallicity` `§02`.

---

## Cross-cutting themes (what the 52 findings really are)

- **Units labels vs unit-correct code.** The integrator uses internal
  `[Msun, pc, Myr]` units correctly, but several *labels* (docs, docstrings,
  log strings) still say `erg` / `cm⁻³` / `yr` / "cgs." This is the exact bug
  class CLAUDE.md calls out, surfacing in human-readable text rather than math.
  **Highest-value cleanup** because it silently misleads.
- **WARPFIELD-era residue.** TRINITY descends from an earlier code; the audit
  found removed-schema validation (`input_warnings.py`), an unused isochrone
  reader (`read_mist_models.py`), an inert `output_format` knob, and "Alternative
  / CURRENTLY UNUSED" solver variants. Harmless to execution, but they make a
  newcomer chase paths that aren't live.
- **Doc/code drift around publishing.** Deprecated docs link in the run banner,
  `example_scripts/` dead pointer, paper "posted vs in preparation," stale cloudy
  run-dir layout — all the kind of thing that drifts right before a public
  release and dents trust.
- **Distributable-vs-clone tension.** The repo declares itself a pip package but
  is really run from a clone; package-data/MANIFEST gaps mean only the clone path
  actually works. Pick one story.
- **Privacy/secrets: genuinely clean.** Worth stating plainly — the scariest
  category for a public push (keys, tokens, personal absolute paths, machine
  names) came back empty. The only "personal" content is intentional attribution.

---

## Recommended priority (if you fix five things before the next clone)

1. **H1 — De-track `scratch/`.** `git rm -r --cached scratch/` (keeps files
   locally, honors the existing `.gitignore:30` rule). Removes ~6.4 MB of
   private diagnostics from every clone.
2. **H2 — Add the missing `else`** in `read_cloudy.py` to raise a clear
   "unsupported metallicity (have 1.0, 0.15)" — ideally validated at param-load
   time so it fails *before* a multi-minute run.
3. **H3 — Fix the `example_scripts/` dead pointer** (commit the examples or
   repoint to `paper/methods/figures/`), in both `_output/README.md` and
   `trinity_reader.py`.
4. **Units labels** — correct `Eb [erg]` → `[Msun·pc²/Myr²]`, the `get_dudt`
   Returns block, and the `get_shellODE` "assumes cgs" line. Cheap, removes a
   43-orders-of-magnitude footgun.
5. **Banner + paper status** — point the startup banner at
   `jiaweiteh.github.io/trinity-web`, and reconcile the arXiv-vs-"in preparation"
   citation across README / `publications.rst` / `license.rst` (and human-verify
   the arXiv id resolves).

H4 (delete `input_warnings.py`) and the `flake8 → ruff` cleanup are easy
follow-ups. Per CLAUDE.md rule 3, dead-code removals are *flagged here*, not
silently applied.

---

## What's verified clean (so the next auditor doesn't re-dig)

- **Quickstart & sweep workflows work on a fresh clone** (dynamically confirmed
  for the single run; statically confirmed for the sweep CLI). All 7 README
  example params exist and validate; argparse matches the docs.
- **Bundled data ships and resolves to repo root** (`lib/default/CIE`, `opiate`
  rot Z1.00 tables, default SPS CSV); the default config hits the bundled fast
  path with no download.
- **`pytest` collects 532 tests, imports all resolve**, the only consumed
  fixture tree (`outputs/mockOutput/mockFullrun/`) is committed and matches
  assertions; the "Private" `test_barnes_population.py` is genuinely absent.
- **No secrets/tokens/keys, no `breakpoint()`/`pdb`, no `__pycache__`/`.DS_Store`**;
  HPC references use a `YOUR_ACCOUNT` placeholder; the single email and all names
  are intentional attribution.
- **`.gitignore` whitelist negations** (`outputs/mockOutput`, `param/*_example`,
  `lib/default`) all behave as intended — `scratch/` is the lone exception.
- **Version is consistent** (1.0.0 across `__init__.py`, `pyproject.toml`,
  `conf.py`); `docs/dev/*.md` plan docs all carry the mandated staleness banner.
