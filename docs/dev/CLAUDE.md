# docs/dev — context for agents working here

Everything under `docs/dev/` is a point-in-time plan/audit, not a maintained spec. Paths, line
numbers, and "what shipped" status go stale fast: treat every claim as unverified and re-check it
against current source before relying on it. The on-tree rules (Status-line format, workstream
folder template, naming, citations, provenance, the banner-exempt list) live in
`docs/dev/CONVENTIONS.md`; `test/test_docs_dev_conventions.py` enforces the mechanical parts.

The canonical banner templates live below — copy them verbatim into new docs, never fork the
wording. Every **active** doc carries all four (⚠️ 🔄 💾 🔗) right under the H1; docs under
`docs/dev/archive/` keep ⚠️ and replace 🔄/💾/🔗 with the single 🧊 banner.

```markdown
> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
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
```

For **archived** docs (`docs/dev/archive/`), the 🔄/💾/🔗 paragraphs are replaced by this one:

```markdown
> 🧊 **Frozen historical record — do not extend.** This workstream shipped or was
> superseded (see the Status line below); the doc is kept as evidence/history. Do
> not update or extend it — new work gets a new doc in an active workstream. The
> ⚠️ caveat above still applies: paths and line references reflect the code as it
> was when this was written.
```

The 🔄 banner makes these docs *living* (whoever opens one rechecks, updates drift, rethinks the
strategy); 💾 makes them *durable* (diagnostics committed as CSVs in `docs/dev/data/` or
harnesses/figures in the workstream folder — reproducible without re-running hours-long sims);
🔗 keeps a multi-doc workstream *mutually consistent* (edit one → reconcile its siblings).
Diagnostic figures here use `matplotlib.use("Agg")`, `text.usetex=False`, dpi≈130–140 and read
only committed CSVs (model: `docs/dev/performance/harness/make_f1_figures.py`).
