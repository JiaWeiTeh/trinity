---
name: report
description: Write a detailed, downloadable, self-contained HTML report of a fix or workstream — with LaTeX rendering and plots — that shows the full story end to end: identify the problem, test out the different ideas, work the chosen idea at different configs, performance, solution. Use after wrapping up a chunk of work to get a shareable downloadable narrative. Reuses the existing report-generator machinery; does not reinvent it.
argument-hint: "[fix / workstream / topic to write up]"
disable-model-invocation: true
---

# /report — detailed illustrated HTML write-up of a fix or workstream

Carry this out as if I had asked it directly. `$ARGUMENTS` names the fix / workstream /
topic to write up. If it is empty, infer it from this session: the branch's commits
(`git log`), the changed files (`git diff --stat`), and what we actually did.

## The brief

> Write me a **detailed** report of **$ARGUMENTS** as a **downloadable, self-contained
> HTML**, with **LaTeX rendering** and **plots**, showing the **full story**: identify the
> problem, test out the different ideas, work on one idea at the different configs,
> performance, solution — and the rest of the arc.

## Tell the story in this order (each beat carried by a plot)

1. **Identify the problem.** The concrete *mechanism*, not just the symptom — *why* it
   failed / was slow / was wrong. A schematic here often beats prose.
2. **Test out the different ideas.** The options/approaches considered, as a method matrix:
   what each was, why it was tried, why one won. Don't pretend the path was straight — if an
   idea was wrong and corrected, that arc is part of the story.
3. **Work the chosen idea at the different configs.** Run it across the must-test configs —
   `param/simple_cluster.param` + `docs/dev/performance/f1edge_{lowdens,hidens}*.param`
   (spanning feedback strength × cloud density), plus the stiffest regime in scope — and show
   it holds across regimes, not just the easy one.
4. **Performance.** The measured numbers (speed / accuracy / convergence), one point per
   config, with the gate/threshold line drawn. Real, committed measurements only.
5. **Solution.** The final change — the diff — concisely, and why it is correct.
6. **The rest of the arc.** The validation journey (per-call equivalence is *necessary but not
   sufficient* for an iterative path → full-run equivalence on the stiff edges, in **separate
   processes**, at **matched `t`**; a "free win" ⇒ **bit-identical**), and a one-line takeaway
   a future reader should remember. Open with a short TL;DR box.

## Plots are required — build them first

A report without figures is incomplete. Plan the plots before the prose; aim for **3–5**:
- **result charts** — the payoff across configs (speedup / accuracy / convergence /
  before-vs-after), with error bars or gate lines;
- **explanatory schematic(s)** — the failure mode, the data flow, or the validation ladder;
- **equivalence/overlay plots** — old-vs-new trajectories, matched-`t` relative-diff vs the gate;
- **solver/diagnostic visuals** where they fit — root-finding maps, convergence sweeps, even an
  animated GIF revealed over segments (time).

The durable pattern (matches the AGENTS.md persist rule): **tabulate the expensive physics once
into a committed CSV, then make the figure a pure read of that CSV** — so it regenerates in
seconds with no re-run. Write a committed matplotlib generator under
`docs/dev/<workstream>/harness/` (or `…/plots/`); `matplotlib.use('Agg')`, dpi≈140, PNGs →
`docs/dev/<workstream>/figs/`. **View each PNG and fix layout/overlap before embedding.**

## Build & deliver

Assemble one self-contained HTML by COPYING an existing generator — pick the house style you
prefer: `docs/dev/performance/harness/make_f1_report.py` (dark-accent) or
`docs/dev/shell-solver/make_insights_html.py` (light-mode). Both:
- base64-embed each PNG via an `img(name, alt)` helper so figures render offline;
- load MathJax from CDN for `$…$` / `$$…$$` (needs network for the math — vendor MathJax only
  if fully-offline math is required);
- carry reusable CSS + section scaffolding; replace the `__FIG_*__` placeholders with `img(...)`.

Output `docs/dev/<workstream>/harness/make_<name>_report.py` → `<name>_report.html` (regenerates
with one command). Confirm no leftover `__FIG_*` and that every figure embedded
(`grep -c 'data:image/png;base64'`). Then hand the file to me with **SendUserFile**.

## Rules (from AGENTS.md — do not drift)
- **Honest measurement.** Only numbers we actually measured/committed; cite the CSV/commit.
  Retract a hypothesis the moment data contradicts it — and keep that retraction in the story.
- **No invented results, no fabricated plots.** If a number isn't in a committed artifact, go
  measure it or omit it.
- **Persist (💾).** Generator + figures live under `docs/dev/<workstream>/{harness,figs,plots}/`,
  never `/tmp` or `scratch/`. Any new `docs/dev/` doc carries the three banners.

## Reference exemplars (read before writing — these span workstreams, not just F1)

**Two finished HTML reports to model — read the one whose style you'll copy:**
- `docs/dev/performance/F1_REPORT.html` ← `harness/make_f1_report.py` — the bubble-luminosity /
  LSODA "60k story": odeint uninitialised-memory crash → solve_ivp → F1. Dark-accent template.
- `docs/dev/shell-solver/insights.html` ← `make_insights_html.py` — the shell-solver investigation
  (background → hypotheses → method → 5 diagnostics → shipped mxstep fix). Light-mode template.

**Figure generators (steal these patterns):**
- `docs/dev/performance/harness/make_f1_figures.py` — result charts (speedup/accuracy across configs).
- `docs/dev/shell-solver/plots/make_plots.py` — 5 diagnostics, *each plot answers one question and
  hands off to the next*, all pure reads of committed `data/*.csv` (no re-run).
- `docs/dev/performance/harness/make_odeint_garbage_figure.py` — a pure schematic (no data) of a mechanism.
- `docs/dev/archive/betadelta/diagnostics/{make_rootmap_gif.py,plot_hunt.py}` — hybr/betadelta
  solver visuals: root-finding maps + an animated cage-vs-no-cage GIF revealed over time, both
  pure reads of tabulated CSVs/jsonl (the "tabulate once, render as a pure read" pattern).

**Narrative / methodology conventions to match:**
- `docs/dev/performance/BUBBLE_LUMINOSITY_PERFORMANCE.md` §Methodology — equivalence depth,
  separate-process A/B, matched-`t`, bit-identical; what the "validation journey" beat should reflect.
- `docs/dev/archive/betadelta/stalling-energy-phase.md` (the hybr velocity-contamination hunt) and the
  `docs/dev/transition/harness/harvest.py` + `docs/dev/data/transition_*.csv` — examples of an
  offline harvest → committed data → writeup across other workstreams.
