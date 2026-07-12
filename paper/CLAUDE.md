# paper/ — context for agents working here

- Each paper subfolder's `PLAN.md` is the source of truth for that paper's status and next steps.
  Read it first; reconstruct from it before asking.
- Most subdirs are **gitignored, local-only**: `barnes26/`, `II-survey/`, `shellSSC6/`,
  `pedrini/`, `rosette/`, and `plots/`. They exist on the maintainer's machines but **not** in
  Claude Code web/Cowork containers — check existence before assuming, and if a folder is absent
  say so instead of recreating it.
- Tracked: `_lib/` (shared plot style + helpers) and `methods/` (figure scripts + committed
  `.npz` bundles; `python paper/methods/make_figures.py [name]` regenerates published figures
  into `paper/plots/`).
- House figure pattern — reduce once, plot from the table: a stdlib-only reduce script walks the
  expensive runs and writes a small `summary.csv` (+ `trajectory.csv`) with provenance
  (`II-survey/reduce_survey.py`, `shellSSC6/reduce_ssc6.py`); figure scripts are pure reads of
  that CSV (or, in `methods/`, of the committed `.npz` bundle) — never re-run sims to plot.
- Figure scripts use `matplotlib.use("Agg")` and `paper/_lib/trinity.mplstyle`
  (`text.usetex=True` with automatic mathtext fallback when `latex` is missing — see
  `_lib/plot_base.py`; `savefig.dpi: 300`). Reduce scripts stay numpy/trinity-free so they run on
  bare cluster login nodes; convert units in the plotting layer with trinity's own constants.
