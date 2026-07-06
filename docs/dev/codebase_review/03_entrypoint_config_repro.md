# 03 — Entry point, config & reproducibility

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

**Scope.** This section audits the run/install/config surface a stranger meets after
`git clone`: `run.py`, `pyproject.toml`, `requirements.txt`, `MANIFEST.in`, the tracked
`param/*.param` files, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.github/workflows/ci.yml`,
and `.gitignore`. The headline result is good news: the **documented quickstart and sweep
workflows would work on a fresh clone**. Every README-listed param file is tracked and parses
against the schema; `requirements.txt` and `pyproject.toml` dependency pins agree exactly; the
bundled `lib/default/` SPS + cooling data ships and is anchored to the repo root (not CWD, not a
personal path); no hardcoded author/absolute paths, debug prints, or TODO/FIXME leak in scope. The
findings below are about packaging/`pip install .` correctness and a few stale config comments —
none of which break the README's `pip install -r requirements.txt` + `python run.py ...` path.

---

## 🔴 High

*(none — the documented fresh-clone quickstart works as written.)*

---

## 🟠 Medium

### [🟠] `package-data` globs point outside any package, so `pip install .` ships no data/params
- **Where:** `pyproject.toml:82-86` (with `pyproject.toml:72-74`)
- **Issue:** package-data is declared as
  ```toml
  [tool.setuptools.package-data]
  "*" = [
      "param/*.param",
      "lib/default/**/*",
  ]
  ```
  but the only package discovered is `trinity*` (`[tool.setuptools.packages.find] include = ["trinity*"]`, `pyproject.toml:74`). setuptools resolves `package-data` globs *relative to each package's directory*, i.e. relative to `trinity/`. There is no `trinity/param/` or `trinity/lib/`; the real data lives at the repo-root `param/` and `lib/default/` (confirmed: `git ls-files lib/default/...`, `param/...`). So a wheel built from this project will almost certainly bundle **none** of the param examples or the SPS/cooling tables. The package code reads them via `_REPO_ROOT = Path(__file__).resolve().parents[2]` (`trinity/_input/read_param.py:37`, `registry.py:64`), which is the *source tree* root — fine in an editable/clone checkout, broken in a site-packages install where `lib/default/` was never copied.
- **Impact (git-puller):** A user who runs `pip install .` or `pip install trinity-sf` (instead of the documented `pip install -r requirements.txt`) gets an importable package whose default cooling/SPS lookups resolve to a path that doesn't exist next to the installed code. The README quickstart dodges this (it installs deps only and runs from the clone), so this is Medium, not High — but it is a real install-mode foot-gun and contradicts the `[project]` packaging intent (the project declares itself a distributable `trinity-sf` v1.0.0, "Production/Stable").
- **Fix:** Either (a) document that TRINITY is run from a clone and drop the wheel ambition, or (b) make data shippable: move `param/`+`lib/default/` under `trinity/` (or add a `[tool.setuptools.data-files]` / `MANIFEST.in` + `include-package-data` arrangement) and have `_REPO_ROOT`-style lookups fall back to `importlib.resources`. At minimum, build an sdist/wheel and confirm the data is present before claiming installability.

### [🟠] `MANIFEST.in` never includes `lib/default/`, and `global-exclude *.json` strips bundled JSON
- **Where:** `MANIFEST.in:10-11`, `MANIFEST.in:33-34`
- **Issue:** The sdist manifest recursively includes `param *.param` (`MANIFEST.in:11`) but has **no rule for `lib/default/`** at all (`grep -i lib MANIFEST.in` → no match). So the bundled SPS/cooling tables (`lib/default/CIE/*.dat`, `lib/default/opiate/*`, `lib/default/sps/starburst99/1e6cluster_default.csv`) are omitted from any source distribution. Additionally `global-exclude *.json` and `global-exclude *.jsonl` (`MANIFEST.in:33-34`) would strip any `.json` data anywhere in the tree. (The `lib/default/opiate/*.npy` cubes are `.npy`, not `.json`, so they're not hit by that rule — but they're already excluded for lack of any include rule.)
- **Impact (git-puller):** `pip download --no-binary` / a PyPI sdist install would be missing the data that "runs out of the box" per the README (`README.md:22`). Same class of problem as the package-data finding; same mitigation (the documented clone path avoids it).
- **Fix:** Add `recursive-include lib/default *` to `MANIFEST.in`, and audit the `global-exclude *.json` against anything under `lib/default/` you actually need shipped.

### [🟠] `.pre-commit-config.yaml` comment blames a non-existent "TOML issue" in `pyproject.toml`
- **Where:** `.pre-commit-config.yaml:26-29`
- **Issue:** The ruff hook passes `--isolated` with the justification:
  > "ignore pyproject.toml (it has a pre-existing `[tool.setuptools.packages.find]` TOML issue that trips ruff's config loader; unrelated to linting)."
  `pyproject.toml` parses as **valid TOML** (verified with `tomllib.load`). There is no TOML syntax error in `[tool.setuptools.packages.find]`. Whatever motivated `--isolated` (ruff not finding a `[tool.ruff]` table, or the setuptools-config *semantic* issue noted above) is mis-described as a "TOML issue," which will mislead the next maintainer who goes looking for a parse error that isn't there.
- **Impact (git-puller):** Cosmetic/misleading rather than breaking — CI still runs (`--isolated` works). But it sends a contributor on a wild goose chase and obscures the *real* packaging defect (the `package-data` mis-location above).
- **Fix:** Correct the comment to describe the actual reason for `--isolated` (no `[tool.ruff]` config to honor, or the setuptools data-glob semantics), and fix the underlying packaging issue.

### [🟠] `requirements.txt` pins are dev-incomplete vs `pyproject.toml` (dev/docs/plots extras absent)
- **Where:** `requirements.txt:16-20` vs `pyproject.toml:45-61`
- **Issue:** The five **core** runtime pins match exactly between the two files (numpy/scipy/astropy/matplotlib/pandas, same lower+upper bounds — good, including the intentional `numpy<2` cap). But `requirements.txt` only *comments out* the dev tools (`# pytest>=7.0` etc., `requirements.txt:16-20`), and lists `flake8` in the comment while the project actually lints with **ruff** (`.pre-commit-config.yaml:20`) — `flake8` is in neither the pre-commit config nor an extra that's used. There's also no `requirements`-level pointer to the `[dev]`/`[docs]`/`[plots]` extras that hold the real dev deps.
- **Impact (git-puller):** Low friction for *running* (core deps are right), but a contributor following `requirements.txt` to set up tooling gets a stale hint (flake8) for a ruff project. CLAUDE.md/README tell people to `pip install -e ".[dev]"`, so the canonical path is fine; `requirements.txt`'s commented block is just stale.
- **Fix:** Drop the `flake8` mention (or replace with `ruff`/`pre-commit`), and in the comment point at `pip install -e ".[dev]"` as the supported dev install.

---

## 🟡 Low

### [🟡] `.gitignore` has line-wrap corruption in several comment blocks
- **Where:** `.gitignore:76-77`, `:133-134`, `:139-145`, `:149-155`, `:159-169`, `:213-220`
- **Issue:** Multiple comment lines are hard-wrapped mid-sentence so the continuation lands on its own unprefixed line, e.g. lines 76-77:
  ```
  #  before PyInstaller builds the exe, so as to inject date/other infos
  into it.
  ```
  Here `into it.` (no `#`) becomes an actual gitignore **pattern** matching any path named `into it.` / `it.` — harmless in practice (no such file), but it's literally an unintended ignore rule. Same wrap pattern recurs in the pyenv/pipenv/poetry/pdm/PyCharm blocks. This is the boilerplate GitHub Python `.gitignore` that got reflowed by an editor.
- **Impact (git-puller):** None functional (the stray bare words don't match real files). Pure cosmetic cruft that makes the file look broken.
- **Fix:** Re-paste the canonical GitHub Python `.gitignore` comment blocks (or just delete the orphaned continuation lines).

### [🟡] CI installs `pytest` ad-hoc instead of the declared `[dev]` extra
- **Where:** `.github/workflows/ci.yml:23-29`
- **Issue:** The test job runs `pip install -r requirements.txt` then `pip install pytest` (`ci.yml:26-27`), bypassing the `[project.optional-dependencies] dev` set in `pyproject.toml:46-52` (which pins `pytest>=7.0`, `pytest-cov`, etc.). The lint job installs only `pre-commit`. So CI never exercises `pip install -e ".[dev]"`, the exact command the docs tell contributors to use (`CLAUDE.md`).
- **Impact (git-puller):** Low. Tests still run. But CI green doesn't prove the documented dev-install path works, and an unpinned `pip install pytest` can drift from the declared `pytest>=7.0`.
- **Fix:** Replace the manual installs with `pip install -e ".[dev]"` so CI and the docs agree.

### [🟡] README implies SLURM `--account/--partition/--time/--mem` are user-editable in a shipped sbatch; the sbatch is generated, not tracked
- **Where:** `README.md:74-77` vs `trinity/_input/sweep_jobs.py:41,55-60,187-188`
- **Issue:** The README sweep workflow says "edit `jobs/submit_sweep.sbatch`: set --account / --partition / --time / --mem". A first-time reader might look for a tracked template to inspect before running. It doesn't exist in the repo — `submit_sweep.sbatch` is *generated* at `--emit-jobs` time from the in-code `_SBATCH_TEMPLATE` (`sweep_jobs.py`, `SBATCH_NAME='submit_sweep.sbatch'`, written at `:187-188`), and the template already includes commented `#SBATCH --account`/`--partition` lines (`:55-60`). This is actually correct behavior (no stale tracked template to drift), just slightly under-explained.
- **Impact (git-puller):** None — `--emit-jobs` produces the file as documented; verified the template ships the knobs the README mentions. Noting only so a future reader doesn't "fix" a missing-file non-bug.
- **Fix:** Optional: one line in the README clarifying `submit_sweep.sbatch` is generated by `--emit-jobs` (not pre-shipped).

---

## Notes verified clean (fresh-clone happy path)

- **All 7 README-listed params exist and are valid:** `simple_cluster.param`, `cloud_example_{PL,BE,homogeneous}.param`, `sweep_{example,tuple_example,hybrid_example}.param` are all tracked (`git ls-files param/`). Every key they use (`model_name, mCloud, sfe, dens_profile, densPL_alpha, densBE_Omega, nCore, rCore, nISM, stop_at_rCloud_nSnap, path2output`) is in the schema `trinity/_input/default.param`, so `read_param`'s unknown-key rejection (`read_param.py:215-225`) passes. Companion rules are satisfied: PL params supply `densPL_alpha`, BE supplies `densBE_Omega` (`registry.py:662-670`).
- **No tracked param is undocumented:** the tracked set under `param/` is exactly the 7 README examples — no orphan/extra param files (`.gitignore:25-29` whitelists exactly `cloud_*`, `*_example.param`, `simple_cluster.param`).
- **Bundled data ships and resolves to repo root, not CWD/personal path:** SPS default `lib/default/sps/starburst99/1e6cluster_default.csv` and cooling tables `lib/default/{CIE,opiate}/...` are tracked and resolved via `_REPO_ROOT` (`read_param.py:37`, `registry.py:64,255,425`). The default quickstart param (ZCloud=1, SB99_rotation=1) hits the bundled-SPS fast path (`registry.py:239-258`) — no external download needed.
- **No hardcoded personal/absolute paths, no debug prints, no TODO/FIXME** in `run.py` or any in-scope config file (grep for `/Users/`, `/home/<user>/`, `jiawei`, `/scratch/`, `/work/` returned only legitimate GitHub project URLs in `pyproject.toml:64-67`). All `print()` in `run.py` are user-facing CLI output.
- **argparse matches the docs:** `run.py:793-829` implements `--workers/-w`, `--dry-run/-n`, `--yes/-y`, `--verbose/-v`, and the mutually-exclusive `--emit-jobs DIR` / `--collect-report DIR` — exactly what README (`README.md:60-82`) and CLAUDE.md claim. `path2output` defaults via the `def_dir` sentinel to `<cwd>/outputs` (`run.py:137-148`, `registry.py:194-197`) — no author path.
- **Dependency pins agree:** core `numpy<2`/`scipy<2`/`astropy<8`/`matplotlib<4`/`pandas<3` are identical in `requirements.txt:10-14` and `pyproject.toml:38-42`; the `numpy<2` cap is intentional and documented in both. `run.py:49-74` independently warns (without failing) if an installed major exceeds the tested range.
- **`pyproject.toml` is valid TOML** (parsed with `tomllib`); `.readthedocs.yaml` points at `docs/source/conf.py` and `docs/requirements.txt`, both tracked.

---

## Counts: 0 high / 4 medium / 3 low
