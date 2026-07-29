# code-audit reference runs — provenance

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

**Status (2026-07-29):** 🔵 ACTIVE — Phase 0a baselines and the inputs Phase 6 measures against.

Baseline runs for the `trinity/` correctness audit (see `../../PLAN.md`). Each is launched as its
own process, per CLAUDE.md rule 5 — trinity leaks module-level global state in-process, so an
in-process second run is not a valid baseline.

## Runs

| Config | Regime | Artifacts |
|---|---|---|
| `param/simple_cluster.param` | energy-driven baseline | `simple_cluster.{stdout,stderr}` |
| `docs/dev/performance/f1edge_lowdens_himass_hisfe.param` | stiff edge — low density, high SFE | `f1edge_lowdens_himass_hisfe.{stdout,stderr}` |
| `docs/dev/performance/f1edge_hidens_himass_losfe.param` | stiff edge — high density, low SFE | `f1edge_hidens_himass_losfe.{stdout,stderr}` |

The two `f1edge` configs span feedback strength × cloud density, the axes most likely to break a
bubble/solver change; `simple_cluster` is the everyday path.

## Exact command

Reproduced from the repo root. Sequential, one process each, 3 h cap per run:

```bash
R=docs/dev/code-audit/data/runs
for p in param/simple_cluster.param \
         docs/dev/performance/f1edge_lowdens_himass_hisfe.param \
         docs/dev/performance/f1edge_hidens_himass_losfe.param; do
  n=$(basename $p .param)
  timeout 10800 python run.py $p > $R/$n.stdout 2> $R/$n.stderr
done
```

An empty `.stderr` is a result, not a missing file: it records that the run emitted no warning or
traceback on that config.

## What these are for

- **Phase 0a** — the pre-audit state, so any behaviour change introduced later is attributable.
- **Phase 6** — the asymptotic-limit, budget-closure and invariant-scan checks read the
  `dictionary.jsonl` these runs produce. Fitting `R ∝ t^(3/5)` (energy-driven) and `R ∝ t^(1/2)`
  (momentum-driven) against them is what catches a dropped term regardless of what the code says
  about itself.

⚠️ These logs are **stdout/stderr only**. The per-run `dictionary.jsonl` lives under the
gitignored `outputs/` tree and does **not** survive the container. Anything Phase 6 concludes from
it must be reduced to a committed CSV or figure here before the session ends.
