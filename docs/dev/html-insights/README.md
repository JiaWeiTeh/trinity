# html-insights — storyline books

> ⚠️ **May be out of date — verify before trusting.** These are rendered narratives,
> not a maintained spec; the code moves faster than the prose. Re-check claims and
> `file:line` references against current source before relying on them.
>
> 🔄 **Living collection — recheck and refine on every visit.** When a workstream report
> changes, rebuild the books (below) and update this index. When you add a report, slot it
> into a storyline as a chapter rather than leaving it stranded in its workstream.
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

Each TRINITY investigation renders a self-contained HTML report that lives with its
workstream. This folder **merges** those reports into a few chaptered **storyline books**, so
a reader can follow one narrative arc end to end. Each book is fully self-contained
(base64-embedded figures, one MathJax include), carries the three dev-doc **banners**, and
opens offline in a browser.

## The books

| Book | Storyline | Chapters (source report) |
|---|---|---|
| `storyline_s1.html` | **From the β–δ solver to the transition-trigger problem** | 1 β–δ solver & “Problem 2” (`archive/betadelta/insights_betadelta_illustrated.html`) · 2 speed-ups (`performance/F1_REPORT.html`) · 3 transition trigger — geometric not thermal (`transition/cleanroom/transition_report.html`) · *Postscript* superseded pshadow/P0 |
| `storyline_s2.html` | **The ODE-solver saga — LSODA → solve_ivp** | 1 shell ODE: the LSODA flood & the fix (`shell-solver/insights.html`) |
| `storyline_s3.html` | **Why large clouds failed (helix)** | 1 diagnosis & fix (`failed-large-clouds/insights.html`) |
| `storyline_s4.html` | **Hidden constants & table audits** | 1 cooling-table temperature floor (`magic-numbers/tclamp_report.html`) · 2 cooling-table refactor *(authored bridge)* |

## How they're built

`build_storylines.py` composes each book from the registry at the top of the file. Per
chapter it pulls the report's body, **scopes that report's CSS under a per-chapter id** (so
chapters can't collide), drops the report's `<h1>` (the book assigns the chapter title),
demotes the remaining headings one level, namespaces ids/anchors, and prepends the banners +
a chapter TOC. Figures ride along as inline base64.

```
cd /home/user/trinity
python docs/dev/html-insights/build_storylines.py     # -> storyline_s{1..4}.html
```

**Sources are the source of truth.** To fix a claim, fix the report's *generator* (e.g.
`performance/harness/make_f1_report.py`), re-run it, then re-run `build_storylines.py`. The
only prose authored here is the two short "bridge" chapters (`COOLING_BRIDGE_HTML`,
`PSHADOW_EPILOGUE_HTML`) for material that has no standalone report.

## Verification

Every chapter was checked **line-by-line against current `trinity/` source** (2026-06-22).
The per-report ledgers and the consolidated scorecard + fix-list live in
[`verification/`](verification/) (`SUMMARY.md` first). Notable corrections already applied:
the β–δ Phase-4 **hybr default flip shipped** (the betadelta report said "still legacy"); the
shell §6 now credits the real LSODA-flood fix (`_NSHELL_MAX` clip guard, not `mxstep`); and
two unbacked F1 wall-time numbers are now flagged indicative. **pshadow/P0 are superseded,
not vindicated** — hence a postscript, not a chapter.

## Adding a chapter / storyline

Add an entry to `STORYLINES` in `build_storylines.py` (`{"type": "html", "src": <report>}`,
or `{"type": "inline", "html": ...}` for an authored bridge), re-run it, and add a row above.
