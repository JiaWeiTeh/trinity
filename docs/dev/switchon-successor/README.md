# switchon-successor — can the fixed 1e-3 Myr clock be made physical?

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

**Status (2026-08-06):** 🔵 actionable — **Batches 1-3 done; no `trinity/` line touched.** The drain
is **PdV work**, not cooling, and the seed violates Weaver's work partition by 4.85× while
satisfying its energy exactly (D1). Two candidates are dead: the physical clock **S1** (fails the
fate bar on 3 of 5, and not in order of window length — the whole "better clock" family is retired,
D2) and the sustainability cap **S2** (first to clear fates on all five, but pins `dEb/dt≈0` so it
lands twice as far from the physics reference as the constant it would replace, D3). Remaining:
**S4** (fix the seed — the root cause) and **S0** (keep the constant, justified by D1-D3). See
`PLAN.md` for the numbers.

## The question — and the one it is NOT

`dt_switchon = 1e-3` Myr ramps the wind termination shock `R1` into the bubble pressure for the
first 1000 yr. **Whether the ramp can be deleted is settled: it cannot** — ablating it flips the
stopping fate on three of five configs, including the default published one
(`docs/dev/phase1a-stiffness/PLAN.md` §2 D6). Do not re-run that.

What is open is its **form**. The ramp resembles a real effect — the termination shock does not
exist until the wind has swept its own mass, the standard free-expansion → energy-driven
transition — but TRINITY already computes that moment as `dt_phase0`, and it is **0.0115-1.96 yr**
across the screen configs, i.e. the fixed 1000-yr window runs 500× to ~87,000× longer than the
physics it imitates. The shape (linear from zero) has no derivation, and no literature was found
for a fixed-duration ramp on `R1`.

## Why it is worth the effort now

Two measurements changed the stakes:

1. The constant decides **whether the bubble survives** on `simple_cluster`, not just at the stiff
   edge — so its form is load-bearing for published results.
2. There is now a **physics yardstick**: Weaver+77 Eq. 20 (`Eb = (5/11)L_w t`, the relation
   TRINITY seeds `E0` from) predicts `Eb/t = (5/11)L_w`. With the ramp, `Eb/t` holds within ~12% of
   that; without it, it falls 154× below. A successor can therefore be judged comparatively rather
   than merely on equivalence — which is bar **N1** in the plan. **Read that bar with `PLAN.md`
   §0.3:** Weaver is wind-only and TRINITY is not (radiation supplies 32-60% of the early drive),
   so N1 asks "no further from the reference than the shipped ramp", never "must match Weaver" —
   and importing Weaver's dimensionless partition as a *target* is ruled out.

## Where to start

1. `PLAN.md` §1 — removal vs replacement, kept separate so the sibling docs stay consistent.
2. `PLAN.md` §2 — what the constant is physically, and where the implementation departs from it.
3. `PLAN.md` §3 — the four candidates, including the root-cause one (the initial condition may be
   inconsistent, and the ramp may be compensating for it).
4. `PLAN.md` §4 — the bars. **N3 forbids trading one absolute constant for another**; §5 Batch 1
   diagnoses *why* the unramped run loses its energy before any candidate is written.
