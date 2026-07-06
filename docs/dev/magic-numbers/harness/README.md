# docs/dev/magic-numbers/harness — cooling T-floor (tclamp) harness

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Harness for the `net_coolingcurve` T-floor story (`../AUDIT.md` finding #1, `../TCLAMP_PLAN.md`).
All commands run **from the repo root**.

- `timeout 180 python docs/dev/magic-numbers/harness/tclamp_instrument.py param/simple_cluster.param
  simple_cluster docs/dev/magic-numbers/data` — non-invasive M1/M2 instrumentation of a real run
  → committed `../data/<tag>_summary.json` + `<tag>_lowT.csv` (SIGTERM-safe; `simple_cluster_capped.param`
  is the bounded stiff config).
- `python docs/dev/magic-numbers/harness/verify_tclamp_equiv.py` — equivalence gate: working-tree
  `get_dudt` vs `git show HEAD:` over a (T, ndens, phi) grid (~20 s; bit-identical for T ≥ 1e4 K).
- `python docs/dev/magic-numbers/harness/make_tclamp_overlay_data.py` — old (1e4) vs new
  (table-edge) floor tabulated once → committed `../data/tclamp_dudt_overlay.csv`.
- `python docs/dev/magic-numbers/harness/make_tclamp_figures.py` — pure read of the committed
  `../data/` artifacts → `../figs/tclamp_*.png`.
- `python docs/dev/magic-numbers/harness/make_tclamp_report.py` — embeds the figs →
  `../tclamp_report.html` (the committed report).

Regeneration chain: instrument/overlay data (committed) → figures → report; the last two steps
re-run without any simulation.
