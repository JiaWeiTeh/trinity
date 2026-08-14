# screen — the multi-config scheme screen

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

**Status (2026-08-05):** 🔵 actionable — harness written and smoke-tested; **no screen has been run
in anger yet**, so it has no committed baseline ledger. Built from `docs/dev/phase1a-init/PLAN.md`
§9, which is where the gap it fills was diagnosed.

## Why this exists

Every end-to-end test in `test/` runs the **same single config** — `mCloud=1e5, sfe=0.3`
(`test_run_smoke.py`, `test_phase_boundary.py`, `test_betadelta_hybr_stress.py`,
`test_bubble_solver_stress.py`; the one outlier, `test_energy_collapse_snapshot.py`, is a 5e9
cloud for the collapse handoff specifically). The genuinely multi-config coverage —
`f1edge_hidens`, `f1edge_lowdens`, the `cal_*` set — exists only as `.param` files driven by HPC
sbatch campaigns, and `test/test_bench7_params.py` validates the *contents of those files*, not
that a run over them behaves.

The consequence, felt directly by the `phase1a-init` workstream: gating that change meant
hand-rolling a worktree, a config matrix, matched-`t` interpolation and a ledger from scratch —
and the first attempt compared *nearest snapshots* instead of matched `t`, which had to be
corrected after the fact. The next scheme change would have hand-rolled it again, differently, and
plausibly repeated the same mistake.

## Run it

```
python docs/dev/screen/screen.py --before <git-ref> [--after <git-ref>] \
    [--configs a,b,c] [--stop-t 0.02] [--bar 5] [--workers 2]
```

- `--before` / `--after` take any git ref. **`WORKTREE`** (the `--after` default) means the
  current working tree, so uncommitted work can be screened without committing it.
- Non-`WORKTREE` refs are materialised with `git worktree add` under `--workdir` and **left in
  place**; the cleanup command is printed. This harness does not delete things for you.
- Exit status is 1 if any config breaches the bar or changes its stopping fate, so it can gate a
  merge.

Default screen set (`CONFIGS` in `screen.py`): `simple_cluster`, `f1edge_lowdens`,
`f1edge_hidens`, `m43_probe`, `gmc_control` — chosen to span the two axes that have actually
broken things here, core density over four decades and both feedback extremes, plus the sub-GMC
scale that no config in `test/` covers.

**Sizing:** a `stop_t = 0.02` Myr arm is roughly 5 min per config on a 4-core container, so the
full 5-config two-arm screen is ~1 hour at `--workers 2`. That is deliberate: it is a gate you run
before landing a scheme change, not something that belongs in `pytest`. Start with
`--configs m43_probe --stop-t 0.002` if you only want to check the harness itself works.

## What it guarantees

- Both arms run in **separate processes** — trinity leaks module-level global state, so an
  in-process A/B is not a comparison.
- Comparison is at **matched simulation time**, by interpolating both arms onto a shared grid.
  Never nearest-snapshot: the arms truncate at different `t` and their snapshot grids differ.
- The grid always includes **the last time both arms share**, which is the "or the end of the run
  if it terminates earlier" clause any bar needs when a config can collapse at 0.04 Myr while
  another runs to 2 Myr.
- **The stopping fate is read from `metadata.json[termination]`** (exit_code + outcome), falling
  back to the snapshot rows' `SimulationEndCode`/`Reason` fields. The first screen run in anger
  (2026-08-06, finding-#3 gate) found the original rows-only read was vacuous on real runs — a
  clean STOPPING_TIME run flushes its last snapshot *before* `main.py` stamps the code, so the
  jsonl tail carries `None`. Pinned by `test_fate_reads_metadata_termination_block`.
- **A run that stops before any stop condition fires** (neither a termination record nor row
  fields) **reports `(no stop condition reached)`**, not `None`. Both arms stopping short still
  compare equal — but the fate check is then vacuous, and the ledger says so rather than
  dressing it up.
- **Stopping fate is checked separately from the radius bar.** A loose radius threshold on its own
  can pass a run that collapses when it should not, by comparing at its own truncated endpoint.

## What it does NOT do

- It does not choose your bar. `--bar` defaults to 5% on `|ΔR2|`, which is the bar the
  `phase1a-init` change was held to — it is a starting point, not a standard. Pre-register
  whatever you actually mean *before* editing code, per CLAUDE.md rule 5.
- It does not check anything but `R2` and the stopping fate. Velocity, energy, phase sequence and
  solver-failure counts are all reasonable extensions; none are implemented.
- It does not know whether a difference is *good*. `phase1a-init` is the cautionary tale: its
  change moved every config in the early window, and the right answer was that the baseline was
  wrong, not the candidate.

## Layout

```
screen.py     the runnable (see its module docstring for the mechanics)
data/         ledger CSVs, one per screen; provenance header in each
```

Harness logic is covered by `test/test_scheme_screen.py` — fast unit tests in the default suite
(config paths resolve, interpolation refuses to extrapolate, the fate check actually fires), plus
an opt-in `-m stress` structural pass over the whole config set.
