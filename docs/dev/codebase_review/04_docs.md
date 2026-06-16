# 04 — Documentation

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

**Scope.** Static audit of repo-level docs from the perspective of a stranger
who `git clone`s the public repo and tries to run a simulation: `README.md`,
`CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, everything under `docs/`
(`docs/source/*.rst`, `docs/source/conf.py`, `docs/dev/*.md`, Makefiles,
`docs/requirements.txt`), plus `trinity/_output/README.md`,
`trinity/_output/cloudy/README.md`, and
`outputs/mockOutput/mockCloudyinput/README.md`. Every doc claim was
cross-checked against the actual code/files (`git ls-files`, `run.py` argparse,
`pyproject.toml`, the param files, the cloudy/output modules). Commands,
file/path references, the layout sections, and badges/URLs/identifiers were all
verified. The headline commands all work and the CLAUDE.md/README layout
sections match reality; the real problems are doc-vs-doc and doc-vs-current-code
contradictions in the output/cloudy docs, a dead `example_scripts/` reference, an
in-startup-banner link to a doc site the repo itself marks deprecated, and an
"in preparation vs posted arXiv id" inconsistency about the paper's status.

---

### [🔴] `trinity/_output/README.md` points to an `example_scripts/` directory that does not exist
- **Where:** `trinity/_output/README.md:122-126` ↔ repo (no such path; `git ls-files | grep example_scripts` → empty)
- **Issue:** The README's "Examples" section says
  > See `example_scripts/` for comprehensive examples:
  > - `example_reader_overview.py`: Full demonstration of reader features
  > - `example_plot_radius_vs_time.py`: Plotting examples

  Neither `example_scripts/` nor either named file is tracked anywhere in the repo (`git ls-files` finds zero matches for `example_scripts`, `example_reader_overview`, `example_plot_radius_vs_time`).
- **Impact (git-puller):** A new user who reads the reader-API doc and goes looking for the promised worked examples finds nothing — a dead pointer to the single most useful onboarding artifact ("comprehensive examples"). It reads as if part of the repo was dropped before publishing.
- **Fix:** Either commit the referenced example scripts, or replace the "Examples" section with the examples that *do* exist (`paper/methods/figures/paper_*.py`, already linked under "See Also") and delete the `example_scripts/` reference.

### [🟠] Startup banner advertises the doc site the project itself marks "deprecated"
- **Where:** `trinity/_output/header.py:35` (printed on every run) ↔ `docs/source/conf.py:51-62` and `README.md:5,14,81`
- **Issue:** `header.display()` (called from `run.py:847` on every single/sweep run) prints:
  > Documentation can be found [here]. → `https://trinitysf.readthedocs.io/en/latest/index.html`

  But `conf.py:57-62` injects a warning into that very site —
  > **This documentation site is deprecated and no longer maintained.** The current TRINITY documentation lives at `jiaweiteh.github.io/trinity-web`

  and README + `conf.py:110` (`html_baseurl`) point everywhere else to `https://jiaweiteh.github.io/trinity-web/`.
- **Impact (git-puller):** The first thing a fresh puller sees when they run `python run.py …` is a link to the deprecated mirror, not the canonical docs the README sends them to. (This is source, not a doc file, so flagged for a human to fix rather than edited here.)
- **Fix:** Change the banner link in `trinity/_output/header.py:35` to `https://jiaweiteh.github.io/trinity-web/` to match README/`conf.py`.

### [🟠] Paper is simultaneously "posted on arXiv (2026)" and "in preparation" across docs
- **Where:** `README.md:103-106` ↔ `docs/source/publications.rst:6-7` and `docs/source/license.rst:11-12`
- **Issue:** README presents the paper as posted:
  > please consider citing the method paper, Teh et al. (2026) (arXiv [2605.27517](https://arxiv.org/abs/2605.27517)). A BibTeX entry is available from [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260527517T/abstract).

  But the Sphinx docs say it is unpublished:
  > `publications.rst:7`: "please cite the TRINITY method paper (Teh et al., in preparation)."
  > `license.rst:11-12`: "please cite the TRINITY method paper (in preparation)".

  And `index.rst:12-13` routes users to `publications` "for the citation" — i.e. straight to the "in preparation" version with no arXiv id.
- **Impact (git-puller):** A user trying to cite TRINITY gets two contradictory instructions depending on which doc they open; one of them is wrong. CHANGELOG.md:14 calls this "First public release," which makes the inconsistency more visible.
- **Fix:** Pick the real status. If the arXiv preprint exists, update `publications.rst` and `license.rst` to the arXiv/ADS citation; if not, drop the concrete id from README until it does.

### [🟠] arXiv / ADS identifier looks like a placeholder — flag for human verification
- **Where:** `README.md:104,106` (the *only* place these ids appear: `git grep` finds them nowhere else)
- **Issue:** README cites arXiv `2605.27517` and ADS `2026arXiv260527517T`. Format `YYMM.NNNNN` parses to May 2026 (`26`/`05`), which is plausibly in the past relative to today (2026-06-16), so not obviously impossible — but it is a suspiciously round-looking sequence number, appears in exactly one file, and conflicts with the "in preparation" wording elsewhere (previous finding). Cannot fetch the web to confirm it resolves.
- **Impact (git-puller):** If this is a stand-in id, every citation and the ADS BibTeX link 404s; readers may cite a non-existent paper.
- **Fix:** Human check: confirm `arxiv.org/abs/2605.27517` and the ADS bibcode resolve to the real TRINITY paper. If they don't, replace with the correct id (and reconcile with `publications.rst`/`license.rst`).

### [🟠] `trinity/_output/cloudy/README.md` presents a stale run-directory layout as current
- **Where:** `trinity/_output/cloudy/README.md:49-64` ↔ `trinity/_output/simulation_end.py:209`, `trinity/_output/cloudy/run_loader.py:13-14`, and `docs/source/running.rst:179-184`
- **Issue:** The cloudy README's "Input" section lists, as the standard TRINITY run directory:
  > `<model>_summary.txt`  # full resolved config (parsed)
  > `simulationEnd.txt`    # success / failure status

  and states (`:63`) "The driver gates on `simulationEnd.txt: Status: SUCCESS`." But current runs no longer write those files: `simulation_end.py:209` — "Phase 5 drop: no longer writes simulationEnd.txt"; `run_loader.py:13-14` — "Legacy runs (pre-Phase-5) **additionally** carried `<model>_summary.txt` and `simulationEnd.txt`" and the loader now reads success/config from `metadata.json[termination]`. `running.rst:179-184` describes the current single-run layout as only `dictionary.jsonl` + `metadata.json` + `trinity.log` — no `_summary.txt`, no `simulationEnd.txt`.
- **Impact (git-puller):** A user who runs a real sim today and then tries the cloudy converter sees a run dir that doesn't match the README's "Input" diagram and a gating rule (`simulationEnd.txt: Status: SUCCESS`) that no longer reflects how success is determined; the doc describes a directory shape they'll never produce. (The committed `outputs/mockOutput/mockFullrun/` *does* still carry the legacy `_summary.txt`/`simulationEnd.txt` files, so the README is internally consistent with that pre-Phase-5 mock — which only deepens the confusion vs a freshly produced run.)
- **Fix:** Update the cloudy README "Input" section to the current `metadata.json`-based layout, note `_summary.txt`/`simulationEnd.txt` as legacy/optional, and restate the success gate in terms of `metadata.json[termination]` (matching `run_loader.py`).

### [🟡] CONTRIBUTING dev-setup diverges from CLAUDE.md/pyproject (`[dev]` extras, ruff vs flake8)
- **Where:** `CONTRIBUTING.md:11-15` ↔ `CLAUDE.md:12,19`, `pyproject.toml:45-52`, `.pre-commit-config.yaml:20-31`
- **Issue:** CONTRIBUTING installs the toolchain ad hoc — `pip install -r requirements.txt` then `pip install pre-commit pytest` — while CLAUDE.md:12 and pyproject advertise the packaged dev extra `pip install -e ".[dev]"`. The `[dev]` extra (`pyproject.toml:46-52`) lists `flake8>=6.0`, but nothing in the repo uses flake8: lint is ruff via `.pre-commit-config.yaml:20-31` (`--select=F821,F811,F823,E9`), and CLAUDE.md:19 itself calls the linter "ruff bug-class." There is no flake8 config anywhere (`git ls-files` finds no `.flake8`/`setup.cfg`/`tox.ini`).
- **Impact (git-puller):** Minor. A contributor following CONTRIBUTING never installs `mypy`/`black`/`ruff`, so they can't reproduce the CLAUDE.md lint/format commands; and the `flake8` listed in `[dev]` is a dead dependency that suggests a linter the project doesn't actually run.
- **Fix:** Point CONTRIBUTING at `pip install -e ".[dev]"`; drop `flake8` from the `[dev]` extra (or swap it for `ruff`, which the pre-commit hook actually needs) so the advertised tools match the ones used.

### [🟡] README "Requirements" and the docs build extras both omit the `[plots]` deps the figures need
- **Where:** `README.md:84-93` / `docs/source/visualization.rst:9-17` ↔ `pyproject.toml:58-61`, `paper/methods/make_figures.py`
- **Issue:** README's "Reproducing the figures" and the visualization doc tell users to run `python paper/methods/make_figures.py`, and README:29-34 lists requirements as just "NumPy, SciPy, Astropy, Matplotlib, pandas … (plus LaTeX)." But the figure scripts pull in extra packages declared only in the `[plots]` optional extra (`pyproject.toml:58-61`: `cmasher`, `Pillow`), which `requirements.txt` (the install path the README quickstart uses) does not include and the docs never mention.
- **Impact (git-puller):** A user who installs via `pip install -r requirements.txt` (as the README quickstart instructs) and then runs `make_figures.py` may hit an `ImportError` for `cmasher`/`Pillow` with no doc pointing them at `pip install -e ".[plots]"`.
- **Fix:** In the "Reproducing the figures" section, note that the figure scripts need the `[plots]` extra (`pip install -e ".[plots]"`), and/or have `make_figures.py` raise a friendly hint.

### [🟡] `pyproject.toml` classifies the project "Production/Stable" while CHANGELOG marks 1.0.0 "Unreleased"
- **Where:** `pyproject.toml:24` ↔ `CHANGELOG.md:8`
- **Issue:** `pyproject.toml:24`: `"Development Status :: 5 - Production/Stable"` and `version = "1.0.0"`, but `CHANGELOG.md:8`: `## [1.0.0] — Unreleased`. Either the version is released (so the changelog heading should not say "Unreleased") or it is not (so the classifier is premature). `conf.py:22-23` independently pins `release = '1.0.0'` / `version = '1.0'`, consistent with pyproject.
- **Impact (git-puller):** Cosmetic/trust signal — a fresh puller comparing the changelog to the package metadata sees the release flagged both shipped and unshipped.
- **Fix:** When 1.0.0 is tagged, change the CHANGELOG heading from "Unreleased" to a date; until then, consider `Development Status :: 4 - Beta`.

### [🟡] `docs/dev/*.md` staleness banners — all three present (no action; positive confirmation)
- **Where:** `docs/dev/archive/betadelta/HYBR_PLAN.md:3-8`, `docs/dev/misc/LEAKING_LUMINOSITIES_SKELETON.md:3-8`, `docs/dev/misc/TERMINATION_EVENTS.md:3-8`
- **Issue:** Per CLAUDE.md:71-87 every `docs/dev/` plan/skeleton doc must carry the "may be out of date" banner directly under the H1. All three do. (Minor wording variant: `LEAKING_LUMINOSITIES_SKELETON.md:4` says "point-in-time plan/skeleton" rather than the canonical "point-in-time analysis/audit" — substantively equivalent, banner intact.)
- **Impact (git-puller):** None — this is a compliance check that passed.
- **Fix:** None required; optionally normalize the LEAKING_LUMINOSITIES wording to the canonical banner text.

---

## Counts: 1 high / 5 medium / 4 low
