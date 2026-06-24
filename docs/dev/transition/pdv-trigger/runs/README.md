# LIVE cooling-boost edge runs — matched-t, boosted vs `none`, separate processes

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

This folder executes the **NEXT step** the `../PLAN.md` (§Task B) has been pointing at: the
matched-`t` **live** edge-config runs that replace the **frozen-trajectory screen**. The screen
bounded the knob (`f_mix ≈ 1.4–2.8`) but cannot forecast, because boosting cooling lowers
`Pb → PdV → Eb(t),R2(t),v2(t)` and so **moves blowout itself**. Only a live boosted run, compared
to its `none` baseline at matched simulation time in a **separate process**, settles whether a
**constant** `f_mix` fires the handoff near blowout across the density grid, or whether the spread
demands the coupled `θ_target(Da)` / `κ_eff` form.

## Zero-contamination contract (why this is trustworthy)

1. **Separate processes** — every run goes through `../../harness/run_stamped.py`, which shells out
   to `run.py` via `subprocess` and refuses to launch from a dirty tree (records a clean-commit
   `provenance.json` next to each output). trinity leaks module-level globals **in-process**;
   separate processes are the only safe way to compare full runs.
2. **Distinct output dirs** — each `.param` sets a unique `path2output = outputs/pdvlive/<name>`
   (gitignored), so no two runs (or parallel git worktrees) ever write the same file.
3. **Worktree isolation** — the live matrix is driven by background agents, each in its own git
   worktree, so a run's clean-tree check is unaffected by edits in any other worktree.
4. **Matched-`t`** — `compare_live.py` interpolates the boosted trajectory onto the baseline time
   grid and reports divergence only over the shared span (runs truncate at different `t`).

## Run matrix (`make_params.py` → `params/*.param`)

| config | physics | regime | modes |
|---|---|---|---|
| `simple_cluster` | mCloud 1e5, sfe 0.3 | normal, compact | `none`, `mult2` |
| `f1edge_lowdens` | mCloud 1e7, sfe 0.5, nCore 1e2 | normal, diffuse | `none`, `mult2`, `mult3` |
| `f1edge_hidens` | mCloud 1e7, sfe 0.01, nCore 1e6 | normal, dense | `none`, `mult2` |
| `fail_repro` | mCloud 5e9, sfe 0.1, nCore 1e2 | heavy super-critical | `none`, `mult2` |

`mult2`/`mult3` = `cooling_boost_mode multiplier` with `cooling_boost_fmix` 2.0 / 3.0. `none` is the
byte-identical baseline. `lowdens` carries the `mult3` arm because the frozen screen wanted `f≈3.8`
there — the f=2,3 spread reads the constant-vs-coupled question live.

## Commands (exact; each run is its own process)

```bash
# 1. (re)generate the param matrix
python docs/dev/transition/pdv-trigger/runs/make_params.py

# 2. run one config's baseline + boost (separate processes, provenance-stamped)
python docs/dev/transition/harness/run_stamped.py docs/dev/transition/pdv-trigger/runs/params/simple_cluster__none.param
python docs/dev/transition/harness/run_stamped.py docs/dev/transition/pdv-trigger/runs/params/simple_cluster__mult2.param

# 3. matched-t comparison -> one CSV row
python docs/dev/transition/pdv-trigger/runs/compare_live.py \
    outputs/pdvlive/simple_cluster__none outputs/pdvlive/simple_cluster__mult2 \
    --label simple_cluster_mult2 --csv docs/dev/transition/pdv-trigger/runs/data/live_compare.csv

# 4. provenance audit (all outputs share one clean commit)
python docs/dev/transition/harness/run_stamped.py --check outputs/pdvlive/*
```

## What lands here (committed)

- `make_params.py`, `compare_live.py`, `params/*.param` — the reproducible matrix + tools.
- `data/live_compare.csv` — one row per (config × boost): `t_trans`, `blowout`, `Δ`s, whether the
  boost handed off via cooling before blowout, matched-`t` `R2/v2/Eb` divergence, terminal fate.
- Raw run outputs (`dictionary.jsonl`, …) stay in the gitignored `outputs/pdvlive/` — **not**
  committed; only the distilled CSV is.

**Open question this answers:** does the live boosted trajectory still hand off near blowout (frozen
screen ≈ right), or does moving `Pb`/blowout break the constant-`f_mix` story (⇒ `θ_target(Da)`)?
