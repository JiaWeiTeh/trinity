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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `KAPPA_EFF_SCOPING.md`,
> `RUNGB_SCOPING.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

This folder executes the **NEXT step** the `../PLAN.md` (§Task B) has been pointing at: the
matched-`t` **live** edge-config runs that replace the **frozen-trajectory screen**. The screen
bounded the knob (`f_mix ≈ 1.4–2.8`) but cannot forecast, because boosting cooling lowers
`Pb → PdV → Eb(t),R2(t),v2(t)` and so **moves blowout itself**. Only a live boosted run, compared
to its `none` baseline at matched simulation time in a **separate process**, settles whether a
**constant** `f_mix` fires the handoff near blowout across the density grid, or whether the spread
demands a **density-dependent** enhancement. **Outcome (the merge, 2026-06-26):** no constant fires across
density; the `Da`-coupled form was **refuted**; the answer is **κ_eff** (`cooling_boost_kappa`) as the cooling
*mechanism* with `f_κ(properties)` calibrated to the `θ(n_H)` target (`PLAN.md` ⭐ synthesis).

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

**Probe params beyond the `make_params.py` matrix** (hand-added, not regenerated by `make_params.py`):
`f1edge_hidens__theta9{0,5}` (the `theta_target` straddle validation, `KAPPA_EFF_SCOPING.md` §5) and
`f1edge_hidens__kappa2` (the `κ_eff` Rung-A `cooling_boost_kappa=2` back-reaction probe,
`KAPPA_EFF_SCOPING.md` §6a). Their baseline is the same `f1edge_hidens__none`.

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
screen ≈ right), or does moving `Pb`/blowout break the constant-`f_mix` story (⇒ a density-dependent
enhancement)? *(Answered: yes — the merge's `f_κ(n_H)` calibration, not the refuted `θ_target(Da)`.)*

## Live results (2026-06-25) — 4 of 4 configs; constant f=2 over/under-shoots by density

Ran under the contamination contract (separate processes, provenance-stamped). **Provenance note:** the
run agents' worktrees were mis-forked from `main` (`b40050a`, which lacks this workstream); two agents
corrected to `6642ff4` via `git checkout` (clean tree, no tracked edits — every `provenance.json`
confirms `commit=6642ff4, dirty=False, rc=0`), one stood down. **Env constraint:** a hard ~55–60 min
wall-clock cap SIGTERMs background runs, so `simple_cluster__none` (an energy runaway needing ~90 min
to reach `stop_t=15 Myr`) cannot complete — but its salvaged partial (to t≈10.4 Myr) fully covers the
boosted run's 0.003–0.17 Myr span, so the matched-`t` compare is valid over the overlap. `f1edge_lowdens`
(diffuse) has since been run (×2/×3/none) in a separate clean worktree at `17f9653`; all three arms hit the
1200 s wall but reach blowout (~0.61 Myr) well within it, so their matched-`t` compares are valid. (These
diffuse runs used `run.py` under `timeout` directly, not `run_stamped`, so they carry no `provenance.json`.)

A constant `f_mix=2` lands **differently per density** — the headline (`data/live_compare.csv`):

| config | density | `none` | `mult2` (f=2) | reading |
|---|---|---|---|---|
| `f1edge_hidens` | dense (nCore 1e6) | fires cooling at t=0.031 Myr | fires cooling **at birth** (t=0.0034), before any blowout | f=2 trips the already-cool dense cloud immediately — over-boost unless θ_lit(dense)→~1 (El-Badry catastrophic cooling) |
| `simple_cluster` | compact | **never fires** (energy runaway >10 Myr) | fires handoff at t=0.131, **just after** blowout (0.109); blowout moves **+0.019 Myr later**; matched-`t` Eb −47%, v2 −44%, R2 −15% | f=2 ≈ marginal; large live trajectory shift ⇒ frozen screen insufficient |
| `fail_repro` | heavy 5e9 | ENERGY_COLLAPSED t=0.0034 (phase 1a) | ENERGY_COLLAPSED t=0.0034, ~identical | cooling boost does **not** rescue heavy clouds — control confirmed |
| `f1edge_lowdens` | diffuse (nCore 1e2) | blows out t=0.611, never fires (trunc. at 1200 s) | **does not fire** at f=2 (blowout 0.620, Eb −13%) **or f=3** (blowout 0.639, Eb −24%); boost only trims Eb + delays blowout +9/+28 kyr | even f=3 < the frozen screen's f≈3.8 → diffuse needs the most boost; confirms **live** that no constant fires across density |

**Verdict:** (1) boosting cooling materially moves the trajectory (Eb −47% compact, −24% diffuse at f=3) —
the frozen screen *bounds* but does not *forecast*; (2) a single constant `f_mix` over-boosts the dense end
(fires at birth) and — now confirmed **live** — under-boosts the diffuse end (does not fire even at f=3;
blowout intervenes at ~0.61 Myr) — the spread argues for a **density-dependent** enhancement, **not** a
constant. **Per the merge:** deliver that via **κ_eff** (`cooling_boost_kappa`) as the cooling *mechanism*,
calibrating `f_κ(properties)` to the `θ(n_H)` target (El-Badry `λδv`=κ_eff + Lancaster) — *not* the refuted
`θ_target(Da)`; (3) heavy clouds collapse regardless of cooling. The zero-code
`L_cool/L_mech`-vs-n diagnostic figure (`../theta_vs_density.png`) and the verified write-up
(`../FINDINGS.md`) close the loop — calibrate to θ_lit(n). **Caveat:** the figure's literature band is still
SCHEMATIC pending a PDF digitize (`NOTE_PATCHES.md` Patch 4), so it quantifies **no** gap; the
constant-can't-bridge case rests on the fmix table (f_mix 1.36 dense → 3.81 diffuse), not the band.
