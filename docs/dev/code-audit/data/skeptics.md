# Phase 5 — skeptic panel verdicts

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


**Status (2026-07-30):** 🔵 ACTIVE — 19 verdicts over 7 defects. Ten S1-class defects were never panelled; see `FINDINGS.md` §"raised but never gate-tested".

## Method

Each skeptic is a fresh agent given the **claim only** — never the finder's reasoning, never the other lenses' verdicts — with full source access, an explicit instruction to *refute*, and **"refuted" as the default under uncertainty**. A confirmation is inadmissible without one of: a numeric demonstration, a dimensional proof, or a literature citation with the exact equation and regime. Majority decides.

Lenses vary by defect: `dimensional` / `literature` / `numerical` for physics claims, `mechanism` / `reachability` / `blast-radius` for silent-failure claims. The second triad proved sharper — most splits below are a real mechanism whose *consequence* does not survive.

## Calibration

`S12b-R-01` was panelled as a **control**: a defect the orchestrator had already reproduced live at the terminal. If the "default to REFUTED" instruction biased the panel, it would have refuted a known-true defect. It confirmed 3/3. That is what licenses trusting the refutations here.

## Verdicts

| defect | lens | verdict |
|---|---|---|
| `ClusterC-vd` | leakage | **CONFIRMED** |
| `ClusterC-vd` | mechanism | REFUTED |
| `ClusterC-vd` | reachability | **CONFIRMED** |
| `ClusterD-odeint` | behavioural | REFUTED |
| `ClusterD-odeint` | blastradius | REFUTED |
| `ClusterD-odeint` | reachability | REFUTED |
| `S12b-R-01` | blastradius | **CONFIRMED** |
| `S12b-R-01` | mechanism | **CONFIRMED** |
| `S12b-R-01` | realism | **CONFIRMED** |
| `S4-R-01` | dimensional | REFUTED |
| `S4-R-01` | literature | REFUTED |
| `S4-R-01` | numerical | **CONFIRMED** |
| `S6-R-02` | dimensional | REFUTED |
| `S6-R-02` | literature | REFUTED |
| `S6-R-02` | numerical | REFUTED |
| `SF-002` | mechanism | **CONFIRMED** |
| `SF-002` | reachability | REFUTED |
| `ST-001` | blastradius | REFUTED |
| `ST-001` | mechanism | **CONFIRMED** |

## Outcome by defect

| defect | tally | result |
|---|---|---|
| `S6-R-02` pdot factor of 2 | 0-3 | **REFUTED** — the 2 is present in the `2π`; the proposed fix would have doubled ram pressure |
| `ClusterD` odeint uninitialised memory | 0-3 | **REFUTED as stated** — mechanism real and demonstrated, but the garbage rows are structurally discarded (0/416 reads) |
| `S4-R-01` P_HII unpaid work | 1-2 | **majority refuted, verdict OVERTURNED** — the single confirming lens had measurements the refuters lacked. See below |
| `ST-001` phase-1a stale locals | 1-1 | **split** — mechanism confirmed, S1 consequence refuted → S2 |
| `SF-002` fsolve convergence | 1-1 | **split** — mechanism confirmed, reachability refuted → S2 |
| `ClusterC` vd=-1e8 | 2-1 | **CONFIRMED S1** — since FIXED on main |
| `S12b-R-01` run-name collision | 3-0 | **CONFIRMED S1** (control) |

## Where majority was the wrong rule

`S4-R-01` was recorded REFUTED on a 2-of-3 majority before the third lens finished. The third lens confirmed it **with measurements the other two did not have**, and it was right. All three were correct about different quantities: the refuters proved `P_HII <= params['Pb']` via the `shell_n0` cap; the numerical lens measured that too (bit-for-bit, cap binding 194/194) *and* showed the `max()` compares against the ODE's **ramped** `press_bubble` instead — ratio up to 2.91, selecting `P_HII` at 100 % of accepted phase-1a states.

**Lesson for the method: a majority vote that outvotes the only lens holding data is not a gate.** Where lenses disagree, reconcile the reasoning before counting votes. The correction is recorded in `data/resolutions.md#s4-r-01-correction`.

## Cost

19 skeptics, ~2.25M tokens (mean ~118k). The control batch was the cheapest (89k mean) and caught an orchestrator error — see `data/agent_costs.md`.

Full per-lens reports are not committed (they run to ~100k tokens each); every verdict's evidence is summarised in `data/resolutions.md` under the matching finding id.
