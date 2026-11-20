# paper/ — context for agents working here

- Each paper subfolder's `PLAN.md` is the source of truth for that paper's status and next steps.
  Read it first; reconstruct from it before asking.
- Most subdirs are **gitignored, local-only**: `barnes26/`, `II-survey/`, `shellSSC6/`,
  `pedrini/`, `rosette/`, `orionNeAtHood/`, and `plots/`. They exist on the maintainer's machines
  but **not** in Claude Code web/Cowork containers — check existence before assuming, and if a
  folder is absent say so instead of recreating it.
- Tracked: `_lib/` (shared plot style + helpers) and `methods/` (figure scripts + committed
  `.npz` bundles; `python paper/methods/make_figures.py [name]` regenerates published figures
  into `paper/plots/`).
- `_lib/trajectory_fit.py` is the shared data/χ² layer for the single-object trajectory papers
  (`orionNeAtHood/`, `rosette/`): observational constraints, χ², and sweep loading. It lets those
  folders stay independent of each other — don't reintroduce cross-paper imports. It reads run
  parameters from `metadata.json` **only**; keep it that way. Sweeps predating a metadata field
  (`_metadata_version: 1` has no `sfe`) are a per-folder problem — see the RELIC shim in
  `orionNeAtHood/plotting_scripts/_legacy_v1_metadata.py` — not a reason to add fallbacks here.
- House figure pattern — reduce once, plot from the table: a stdlib-only reduce script walks the
  expensive runs and writes a small `summary.csv` (+ `trajectory.csv`) with provenance
  (`II-survey/reduce_survey.py`, `shellSSC6/harness/reduce_ssc6.py`); figure scripts are pure reads of
  that CSV (or, in `methods/`, of the committed `.npz` bundle) — never re-run sims to plot.
- Figure scripts use `matplotlib.use("Agg")` and `paper/_lib/trinity.mplstyle`
  (`text.usetex=True` with automatic mathtext fallback when `latex` is missing — see
  `_lib/plot_base.py`; `savefig.dpi: 300`). Reduce scripts stay numpy/trinity-free so they run on
  bare cluster login nodes; convert units in the plotting layer with trinity's own constants.

## Folder standard for `paper/<project>/` (ruled 2026-11-20)

One layout for every project folder. The full text, with the ruling behind each line, is
`paper/_structure/STANDARD.md` (local-only, like the project folders).

- **Root:** `PLAN.md` (opens with a status block of at most 40 lines), `sync.sh` if the project
  runs on Helix, and folders. No scripts, `.param` files, PDFs, CSVs, prompts, runbooks, LaTeX
  build products, dated files or dated folders.
- **`draft/`:** `main.tex`, `main.pdf`, `aa.cls`, `aa.bst`, the one `.bib`; only text we write.
  `draft/fig/` holds copies of the figures `main.tex` includes; `draft/edits/` the scripts that edit
  `main.tex`; `draft/to-be-deleted/` the `main_pre_*` backups.
- **Figures and scripts:** `plots/` figures that are not in the draft; `plotting_scripts/` scripts
  whose product is a PDF; `harness/` scripts whose product is a table, a number or PASS/FAIL.
- **Data and runs:** `data/<campaign>/` reduced tables, with pulled run folders in
  `data/<campaign>/big_runs/`; `param/` the `.param` files; `hpc/` sbatch files, helper scripts,
  RUNBOOKs; `sps/` the tables the `.param` files name.
- **Session files:** `prompts/PROMPT_continue.md`, one per stream, overwritten at each hand-off
  after the log entry is written (the previous one stays as `PROMPT_continue.prev.md`);
  `log/chapter<N>.tex` for questions, answers, rulings, numbers and ideas, with a new chapter
  once the PDF reaches 100 pages.
- **Other:** `notes/` prose that is not the plan; `external/<source>/` material received and never
  edited (a collaborator's manuscript goes here); `deliveries/<YYYY-MM-DD>_<to>_<what>/` material
  sent out; `archive/` superseded but kept, frozen; `to-be-deleted/` the only staging name, in
  batches `<YYYYMMDD>_<what>/`.
- **One name per role:** no `figures/`, `figs/`, `plots_*/`, `params/`, `chapter/`,
  `to-be-removed/`, and no folder per task or date.
- **Light folders** (not papers: `proposal/`, `hamed/`, `pineda/`, `janice/`, `website/`,
  `pedrini/`): `README.md`, `src/`, `data/`, `out/`, `to-be-deleted/`, free names for material
  received or sent.
- **Now and later:** from now on, new files follow this layout in every project. The exception is a
  kind of file a project already keeps in a fixed place (the draft's figures where its
  `\graphicspath` points, a campaign's tables where its scripts read them); it stays there until
  that project's reorganisation batch moves it. Existing files move only in that batch, on the
  maintainer's word that the project's session is parked.
- **A batch in progress:** if a project root holds `REORG_IN_PROGRESS.md`, do not write in that
  folder; tell the maintainer. A session opened before a project's batch still has the old paths:
  restart it from the project's `prompts/PROMPT_continue.md`.
- `python tools/check_paper_layout.py [project ...]` lists what in a project does not follow this.
