# docs/dev/failed-large-clouds/harness — Eb-collapse matrix harness (+ sibling figures/)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Harness for the massive-cloud `Eb=nan` / `ENERGY_COLLAPSED` investigation (`../PLAN.md`).
Production untouched — `variants.py` monkeypatches V0–V3. All commands run **from the repo root**.

- `python docs/dev/failed-large-clouds/harness/make_params.py` — writes the matrix configs into
  `harness/params/*.param`. Runs output to `$TRINITY_FLC_RUNROOT/<name>` (default `/tmp/flc/<name>`;
  ephemeral by design — the durable artifacts are the committed CSVs in `../data/`).
- `python docs/dev/failed-large-clouds/harness/probe_degeneracy.py` — sim-free probe of the R1→R2
  degeneracy → committed `../data/probe_degeneracy.csv` (instant).
- `timeout 400 python docs/dev/failed-large-clouds/harness/run_variant.py --variant V0
  --param docs/dev/failed-large-clouds/harness/params/fail_repro.param --csv docs/dev/failed-large-clouds/data/eval_V0.csv`
  — one (variant, config) cell per process; appends a CSV row; `--out`/`--stop_t` optional.

Sibling `../figures/` (pure reads of committed `../data/` CSVs — no sim needed):
- `python docs/dev/failed-large-clouds/figures/make_energy_budget_figs.py` → `fig1`–`fig3` PNGs
  (regenerates `../data/budget_*.csv` first if run dicts exist under `$TRINITY_FLC_RUNROOT`).
- `DISC_BATCH=<stamped batch> python docs/dev/failed-large-clouds/figures/make_discriminator.py`
  → `../data/discriminator.csv` + `fig4` (falls back to `$TRINITY_FLC_RUNROOT`; refuses to run on empty data).
- `python docs/dev/failed-large-clouds/figures/make_insights_html.py` → `../insights.html`.
