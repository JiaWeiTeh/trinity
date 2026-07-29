# Phase 0e — audit calibration against seeded defects

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

**Status (2026-07-29):** ✅ gate passed — 8/8 seeded defects detected, 0 false accusations against correct code.

## Why this exists

An audit that finds nothing is indistinguishable from an audit that cannot see. Before
spending the pipeline on the real code, it was run against a copy of
`trinity/phase0_init/get_InitPhaseParam.py` seeded with eight defects of exactly the classes
the review is hunting. The pipeline was **not** told that defects existed.

Reproduce with `harness/calibration_mutations.py` (the mutation set is committed there, so a
future session can re-run the gate against a changed pipeline).

## The seeded defects and the result

Two of the eight — **M1** and **M4** — mutate the code **and its comment consistently**. No
amount of reading code-plus-comment can catch those; only Lens C, deriving the physics
independently, can. They are the calibration's real test.

| # | Seeded defect | Catchable by | Detected as | ✓ |
|---|---|---|---|:-:|
| M1 | Weaver thermal fraction `5/11` → `5/7`, **comment changed to match** | Lens C only | R-006, flagged LOUD as `A=B≠C` | ✅ |
| M2 | Temperature exponent `8/35` → `8/25`, comment left stale | A vs B | R-003 | ✅ |
| M3 | `(1 − ξ)^0.4` factor dropped from the `T0` product | A vs B/C | R-005 (also caught that `bubble_xi_Tb` becomes a dead knob) | ✅ |
| M4 | Free-streaming geometry `4π` → `2π`, **comment changed to match** | Lens C only | R-007, flagged LOUD, verdict **"unresolved — needs an independent source"** | ⚠️ |
| M5 | `v0 = 2L/pdot` inverted to `2·pdot/L` | A vs B/C | R-001, with the full dimensional cascade traced through all five return values | ✅ |
| M6 | `cvt.L_au2cgs` conversion dropped from the `T0` luminosity ratio | A dimensional pass | R-004 | ✅ |
| M7 | Density guard `nCore <= 0` weakened to `< 0` | A vs C | R-008 | ✅ |
| M8 | Time exponent sign flipped, `−6/35` → `+6/35` | A vs B | R-002 | ✅ |

**8 of 8 detected.** No entry accused correct code of being wrong: R-001…R-008 map exactly onto
the eight seeds, and the reconciler separately issued an explicit **clearance** (R-016) for the
one slot where the code is right and the comment is ambiguous — the elapsed-vs-absolute-time
trap that Lens C had predicted in advance.

Beyond the seeds the pipeline surfaced ten further discrepancies (R-009…R-018) on
**unmutated** lines — missing `mu_convert` validation, an inclusive `ξ ≤ 1` bound that becomes
live only once M3's dropped factor is restored, the `1e-100` clamping policy, and several
doc-drift items. Those are candidate findings against the *real* file and must be re-derived
from unmutated source before they can be reported; they are not carried over from here.

## What the calibration also measured: the cost of the blocked literature

M4 is the one seed not confirmed outright. Lens C could not reach Rahner's thesis — the
container's egress proxy returns **403 (organisation policy denial)** for every scholarly host
(arxiv.org, ADS, A&A, MPG, Wikipedia; only github.com is reachable). The audit's own rule
forbids ruling against the code on a low-confidence derivation alone, so the reconciler
correctly stopped at "suspicion is high, proof is absent".

That is the right behaviour, and it quantifies the blocker precisely: **a defect that only
primary literature can settle degrades from *confirmed* to *suspected*.** It is still surfaced,
still loud, still in the report — it just cannot be closed in this container.

## Design notes carried into the real run

- The prose ledger originally truncated docstrings at 600 characters. Lens B — which sees
  *only* prose — reported the truncation itself, and the cap was removed before the real run.
  Without that, every long docstring's tail would have gone unread.
- Mutation density was high (8 defects in a 209-line file). A real slice is far sparser, so
  this measures sensitivity, not the false-negative rate at realistic density. Treat 8/8 as
  "the lenses can see these classes", not as "nothing will be missed".
