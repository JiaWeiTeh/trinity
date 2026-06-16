# 06 — tools/, paper/, lib/

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

**Scope.** Static fresh-clone audit of `tools/`, `paper/`, and `lib/` only —
from the perspective of a stranger who `git clone`s the public repo and tries
to reproduce the paper figures and run the CLI utilities. No code was run
(env lacks numpy/scipy/matplotlib); claims are verified by reading source and
cross-checking `git ls-files`. The good news up front: the paper-figure
reproducibility story is largely intact (all four `.npz` bundles are committed,
the figure short-names match the README, no private `paper/barnes26/` path is
referenced, and the `lib/default/` tables the quickstart needs are all present).
The findings below are the gaps.

---

## Findings

### [🟠] `make_figures.py` figures crash without a LaTeX install (`text.usetex: True`), only partly caveated by README
- **Where:** `paper/_lib/trinity.mplstyle:8`, applied unconditionally at import by `paper/_lib/plot_base.py:40-42`; README `README.md:88` vs `README.md:33-34`.
- **Issue:** Every figure module imports `paper._lib.plot_base`, which on import runs `plt.style.use(_STYLE_PATH)` with `trinity.mplstyle` containing `text.usetex: True` and `text.latex.preamble: \usepackage{amsmath} \usepackage{amssymb}`. There is no try/except or "usetex available?" guard. So `python paper/methods/make_figures.py` fails at render time on any box without a working LaTeX toolchain (latex/dvipng). The Reproducing-the-figures section advertises "no raw simulation output and no extra downloads needed" (`README.md:87-88`) — true for *data*, but a LaTeX install is in fact required. The requirement *is* stated earlier at `README.md:33-34` and in `CLAUDE.md:21`, so this is a documented-but-easy-to-miss footgun, not an undocumented one.
- **Impact (git-puller):** A puller who jumps to the "Reproducing the figures" header (the natural place to start) and lacks LaTeX gets a stack trace, not a figure, despite the local "no extra downloads needed" claim.
- **Fix:** Either (a) gate usetex on availability — fall back to mathtext when `shutil.which("latex")` is None (in `plot_base.py` before `style.use`), or (b) repeat the LaTeX caveat inline in the Reproducing-the-figures block and soften "no extra downloads needed" to "no extra *data* downloads needed".

### [🟠] `make_density_profile_gif.py` default run path points into gitignored `outputs/` — no-arg invocation crashes on a fresh clone
- **Where:** `tools/make_density_profile_gif.py:65-67` (`DEFAULT_RUN = REPO/"outputs"/"rosette_cf_survey_updated_0p77"/"1e5_sfe001_n1e3_PL0_yesPHII"`), docstring `:31` ("`run_dir` defaults to the rosette example"), arg default `:334`.
- **Issue:** The tool's positional `run_dir` defaults to a run folder under `outputs/`, but `.gitignore` tracks only `outputs/mockOutput/` (`outputs/*` is ignored, `!outputs/mockOutput/` re-included). `git ls-files | grep -i rosette` returns nothing — the `rosette_cf_survey_updated_0p77/...` folder is a personal-machine artifact that is not shipped. Running `python tools/make_density_profile_gif.py` with no args hits `load_frames()` → `raise FileNotFoundError(f"no dictionary.jsonl in {run_dir}")` at `:84-85`.
- **Impact (git-puller):** A stranger who runs the tool the way its own docstring suggests (no arg → "the rosette example") gets a FileNotFoundError. The shipped `outputs/mockOutput/mockFullrun/` (which *does* contain a `dictionary.jsonl`) would be the natural default.
- **Fix:** Point `DEFAULT_RUN` at a tracked folder (e.g. `REPO/"outputs"/"mockOutput"/"mockFullrun"`) and update the docstring, or drop the default and make `run_dir` required so the failure mode is an argparse usage message rather than a confusing path-not-found.

### [🟡] Module-import side-effect `print()`s in two paper figure scripts leak into `make_figures.py` output
- **Where:** `paper/methods/figures/paper_feedback.py:37` (`print("...plotting force fractions with ram composition overlay + PISM")`) and `paper/methods/figures/paper_radiusComparison.py:42` (`print("...plotting radius comparison (TRINITY vs WARPFIELD vs Weaver)")`).
- **Issue:** These `print()`s run at *import* time, not inside a function. `make_figures.py` runs each figure in a subprocess (`-m <module>`), and `paper_teaser` additionally imports `paper_feedback` (`paper_teaser.py:57`), so the force-fractions line prints even when only the teaser is requested. Looks like leftover debug scaffolding.
- **Impact (git-puller):** Cosmetic noise in the reproduction console output; can confuse ("why is the teaser printing 'plotting force fractions'?"). No functional harm.
- **Fix:** Move both prints inside their respective `plot_*`/`main` functions, or delete them.

### [🟡] `make_figures.py` reports unmatched names as "skipped — bundle not yet published" even when all four bundles ship
- **Where:** `paper/methods/make_figures.py:107-111`, `:119-122`.
- **Issue:** All four `FIGURES` bundles (`densityProfile.npz`, `diagnostics.npz`, `radiusComparison.npz`, `app_LSODA.npz`) are committed under `paper/methods/data/` (verified via `git ls-files paper/`), so on a clean clone the "Skipped (bundle not yet published)" branch never fires — fine today. But the wording ("not yet published") is a forward-looking placeholder; if a future bundle is genuinely missing the message tells a puller it's an upstream-publishing gap, not "your clone is broken", which is correct, but the message gives no hint that the file should exist after a normal clone. Minor/stale-harmless as-is.
- **Impact (git-puller):** None today (no figure is skipped on a fresh clone). Flagged only so a future missing-bundle regression isn't masked by reassuring wording.
- **Fix:** None required now; if a bundle is ever dropped, distinguish "bundle intentionally unpublished" from "expected file absent from clone".

---

## Verified-clean (no action; recorded so the next auditor doesn't re-dig)

- **All four figure `.npz` bundles exist and are tracked.** `paper/methods/data/{densityProfile,diagnostics,radiusComparison,app_LSODA}.npz` all appear in `git ls-files paper/`, matching the `bundle=` paths in `make_figures.py:45,52,59,66`.
- **Figure short-names match the README.** `FIGURES` names are `density`, `teaser`, `radiusComparison`, `rcloud_smoothing`; README example uses `teaser` (`README.md:92`). Prefix-matching (`_select`) accepts `density`, `rcloud`, etc.
- **No private/author paths in `paper/`.** No reference to `paper/barnes26/` (the `.gitignore`-private dir) or `test/test_barnes_population.py`; grep for `/home`,`/Users`,`/scratch`,`barnes26` over `paper/` is empty. The "WARPFIELD" comparison is the in-repo `_noPHII` TRINITY run plus analytic scaling laws (`paper_radiusComparison.py:3-14`), not an external WARPFIELD dataset — so "no extra downloads" holds for data.
- **`make_figures.py` → `paper_rcloud_smoothing` arg wiring is correct.** `make_figures` passes the bundle positionally and `-o <outdir>/rcloud_smoothing.pdf` as a file path (`:69`); the module's parser takes `source` positionally and `-o/--out` as an output PDF path (`paper_rcloud_smoothing.py:514-520`) — consistent.
- **`lib/default/` is complete for the quickstart/default config.** Defaults resolve to shipped files: CIE `path_cooling_CIE 3` → `lib/default/CIE/coolingCIE_3_Gnat-Ferland2012.dat` (present); non-CIE folder `lib/default/opiate/` with `ZCloud 1`+`SB99_rotation 1` → `opiate_cooling_rot_Z1.00_age*.dat` (all ages 1e6–1e7 present); SPS `def_path` → `lib/default/sps/starburst99/1e6cluster_default.csv` (present). `.gitignore` ignores `lib/*` but re-includes `!lib/default/` + `!lib/default/**`; the full SPS/cooling library is honestly flagged "available on request" (`README.md:95-99`) and is *not* hardcoded as a required path anywhere in the default flow. `default.param:135` correctly warns that `SB99_rotation 0` (norot) needs user-supplied tables since only rot tables ship.
- **`tools/` is clean of personal/secret/path leaks.** Grep for `/home`,`/Users`,`/scratch`,`/mnt`,`teh_jiawei`,`jiawei`,`barnes26` over `tools/` is empty; grep for `TODO`/`FIXME`/`breakpoint`/`pdb` is empty; grep for emails over all three dirs is empty.
- **Tool data dependencies are user-supplied and honestly documented.** `bubble_audit/*` and `tinit_sweep/*` operate on `.pkl` state files that are explicitly "not committed (≈5 MB each)" with regeneration instructions (`tools/tinit_sweep/README.md:36-46`); the README's referenced files (`docs/dev/tinit-sensitivity.md`, `docs/dev/bubble-integrator-robustness.md`) and constants (`_T_INIT_BOUNDARY`, env var `TRINITY_BUBBLE_STATE_DUMP` in `trinity/bubble_structure/bubble_luminosity.py`) all exist. `bubble_audit/audit.py:35` and siblings default their base param to the *tracked* `param/cloud_example_PL.param`. `bubble_conduction_convergence.py` falls back to an inline temp `.param` body (`:212-218`) — fully self-contained. `compare_outputs.py` / `inspect_bubble_diag.py` require explicit user paths and claim no shipped default.
- **`tinit_sweep/README.md` matches the tool.** Documented commands (`run_sweep.py <states_dir> --k 5`, `profile_tinit.py <states_dir>`) match the actual argparse interfaces (`run_sweep.py:22`); the five-gate design description matches the module docstring.

---

## Counts: 0 high / 2 medium / 2 low
