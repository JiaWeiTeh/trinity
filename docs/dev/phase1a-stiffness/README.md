# phase1a-stiffness — is phase 1a's segment integrator a latent defect?

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

**Status (2026-08-06):** 🟡 partial — **Batches 1-5 done; the C2 fix is in and gated, with one bar
clause open.** It clears P0, P1, P1-free (`dictionary.jsonl` **byte-identical on all five configs**,
where the bar required only `simple_cluster`), P2 (worst 1.007×) and P4 (behavioural test, verified
failing-first); suite 1057/0 and `pre-commit` pass. **P3's mypy clause fails as written** — 144 vs
137 baseline, all +7 of an `attr-defined` idiom the same file already carries 49 of — and is
recorded in `PLAN.md` §2 D5 for a maintainer ruling rather than reinterpreted. **Batch 6 answered
the workstream's motivating question: `dt_switchon` is NOT removable** — ablation still flips the
fate on 3 of 5 configs, including the default published one, so magic-number #2 stays
document-and-pinned (§2 D6). Batch 1: production is ≥4.3e4×
away from the stall (worst segment 4 steps / 0.021 s; the whole phase-1a segment integrator costs
0.2-0.6 s per run), which rules out the LSODA swap on economics. Batch 2: the stall is
**stiffness**, not a singularity — `Eb` collapses 7 decades and pins at 1.6e-6 au on a slow
manifold with dominant λ ≈ −1e13, so one segment would take ~7 days — and **`Eb` never reaches
zero**, so phase 1a's existing `Eb ≤ 0` guard would miss this state even if the segment finished.
The remedy is an in-band, *positive*, scale-relative energy-floor event. Batch 3 built it — a
per-segment event at 1e-3 of the segment's starting `Eb`, a threshold bounded on both sides by
measurement — and the stalling control now ends in **22 s** with the pre-existing
`ENERGY_COLLAPSED` fate instead of grinding — and Batch 4 proved it changes nothing anywhere else,
byte for byte. `PLAN.md` §2 holds the numbers.

## The question

Phase 1a integrates its segments with `solve_ivp(method='RK45')` and **no `min_step`/`max_step`**
(`run_energy_phase.py:309-318` @ `adfc23f`), while phases 1b, 1c and 2 all use **LSODA** —
stiff/non-stiff auto-switching — *with* both step bounds. When the `dt_switchon` ramp is ablated
at `nCore = 1e6`, `Eb` collapses inside a segment and that RK45 call grinds for > 44 minutes on
one segment; the `Eb ≤ 0` guard that would end the run cleanly lives *between* segments, so it
never gets the chance. Measured: `docs/dev/magic-numbers/data/switchon_stall_probe.csv` and
`switchon_stall_stacks.txt`.

So: latent defect, or a curiosity of a configuration production never runs? That is what this
workstream measured before proposing anything. The answer: production is four decades clear of it,
the grind is genuine stiffness on a collapsed-energy slow manifold, and the guard that should have
caught it tests for `Eb ≤ 0` when `Eb` in fact stalls at a small *positive* value. The fix is an
in-band energy-floor event — byte-identical on every production config — and it does **not** make
`dt_switchon` removable.

## Where to start

1. `PLAN.md` §1 — what is actually in the source (verified, with the asymmetry vs 1b/1c/2).
2. `PLAN.md` §2 — the load-bearing unknown: does this bite with the ramp *active*?
3. `PLAN.md` §3 §5 — the pre-registered bars and the decision rule, including the
   stiffness-vs-singularity trap that decides which remedy is even correct.
4. `PLAN.md` §6 — the batches. **All six are done** (D1-D6 in §2). The only open item is the
   mypy-clause ruling in D5.

## Why it exists

Spun out of `docs/dev/magic-numbers/SWEEP2_PLAN.md` §4-5, which closed magic-number audit #2
(`dt_switchon = 1e-3`) as document-and-pin: the ramp is load-bearing, but what it protects turned
out to be this integrator rather than the bubble-structure solve as the earlier write-ups
recorded. Removing that constant is only conceivable once phase 1a can survive an `Eb` collapse
on its own — which was `PLAN.md` Batch 6, deliberately last. **It ran, and the answer is no:** even
with collapse now stopping cleanly, ablating the ramp flips the stopping fate on three of five
configs, `simple_cluster` among them. The constant stays, and the sibling docs that described it
as nearly inert are corrected (§2 D6).
