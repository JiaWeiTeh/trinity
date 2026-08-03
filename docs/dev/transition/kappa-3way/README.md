# kappa-3way — the three-way band-entry calibration, measured fresh

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

**Status (2026-08-02):** 🔵 actionable — **294/294 arms ran; the three-way table is MEASURED**
(`FINDINGS.md`, `report.html`). Headline: **f_κ is the worst of the three knobs** on both metrics and **P1 is falsified**. ⚠️ But
`FINDINGS §11`: Θ_cum is the wrong metric for the knob decision — on the trigger's own instantaneous
criterion f_A and f_mix looked tied, and `§12` (stale-row exclusion) breaks the tie: **f_A is the
best single knob on both axes** (2.71× vs f_mix's 3.70× solved-row spread); `§10` shows the
mechanism ranking is the reverse of the calibration one either way.
Gate **G0 failed 2/11** — traced to truncation (mechanism inferred, not verified — `FINDINGS §1`), which also makes f_A's published 5.39× spread
non-converged. Nothing here changes production. Successor experiment: **`F_AREA_PLAN.md`** (the
combined-knob area construction, pre-registered, 514-arm bench8 design) — **its Phase A0 ran
2026-08-03 and GA0 FAILED, so bench8 was never submitted**: the combined knob reproduces f_κ alone
on Ṁ (f^{2/7}, not the f^{1} area multiplication needs), closing the last loophole in `§10b`'s
"no shipped knob raises mass loading". See `FINDINGS §13` and `F_AREA_PLAN §5a`.

---

## What this workstream is, in one paragraph

TRINITY needs a cooling boost to make realistic GMCs fire the energy→momentum trigger. Three knobs can
supply it — `cooling_boost_fA` (f_A), `cooling_boost_mode='multiplier'` (f_mix), and `cooling_boost_kappa`
(f_κ) — and the way to choose between them is the **Lancaster 2021b Θ_cum band-entry calibration**: for each
knob, what dose brings Θ_cum into the observed band [0.90, 0.99], and **how much does that dose vary across
cloud density?** The knob whose calibrated dose varies least is the better single physical constant. f_A and
f_mix have been through that calibration. **f_κ never has.** This workstream measures the missing third leg
— and, because the earlier harvests are no longer trusted, **re-measures the other two at the same time, in
one campaign, in one code state, with a timestamp on every artifact.**

## Why a new directory

`docs/dev/transition/pdv-trigger/` is the parent workstream: ~25 docs and ~50 committed CSVs accumulated
over five weeks, with a real history of corrections (a metric artifact that published a wrong conclusion in
`§17`/`§18`; a falsified physics claim in `§23`; a re-attributed cause in `§24`). Its evidence is *valuable*
and mostly *correct*, but it is layered — some numbers are measured, some extrapolated, some contaminated,
some superseded, and telling them apart takes a careful read of three registers.

This directory starts clean, with one rule doing the work that register did:

> **The freshness rule.** A number is quotable here only if it comes from an artifact whose own provenance
> stamp is dated **on or after the cutoff, 2026-07-29**. Everything older is `VERIFY` — possibly true,
> not citable until re-measured. See `PROVENANCE.md`.

The parent workstream is not deleted, archived, or contradicted. It is **demoted to a hypothesis source**:
read it for *what to test and why*, never for *what the answer is*.

## Read in this order

1. **`report.html`** — 📌 **THE SOURCE OF TRUTH.** Generated from the committed CSVs, so it cannot drift
   from the data; it prints its own build timestamp and a live freshness roll-up. Rebuild with
   `python docs/dev/transition/kappa-3way/make_report.py`.
2. **`PROVENANCE.md`** — the freshness rule, the quotability ladder, and what is inherited vs re-measured.
   **Read before quoting any number.**
3. **`PLAN.md`** — what happened (the history that produced this campaign), what to do (the run order),
   what we will do (the analysis, the gates, the decision and its pre-registered stopping rule).
4. **`REPRODUCE.md`** — claim → command → artifact.

## What is where

| thing | where | why there |
|---|---|---|
| the record, the rules, the plan, the HTML | **here** (`kappa-3way/`) | starts clean at the cutoff |
| the freshness gate + audit | `data/` here, plus `pdv-trigger/data/make_freshness_audit.py` | the audit walks the parent tree, so it lives where the artifacts are |
| param builders, `.param` files, sbatch, `sync_bench.sh` | `pdv-trigger/runs/` (unchanged) | the cluster paths are baked into `run_bench7.sbatch` and `sync_bench.sh` (`$REPO/docs/dev/transition/pdv-trigger/runs/...`). Moving them buys a tidier tree and risks a silent path break on a one-shot 294-arm reduce. **Deliberately not moved.** |
| the 2026-07-19 and earlier evidence | `pdv-trigger/` (unchanged) | preserved as-is; demoted, not rewritten |

## The campaign, at a glance

**294 arms**, all `stop_t = 5 Myr`, one process each, single-knob by construction.

| campaign | arms | what it measures |
|---|---|---|
| `bench7` — K1 | 54 | **the missing third leg**: f_κ band entry on bench1/2/3 |
| `bench7` — K1b | 20 | dense-end fire map, f_κ ∈ {2,4,8,12,16} |
| `bench7` — K2 | 66 | the whole f_κ fire map for the 6 band configs, re-measured + filled in |
| `bench7` — K3 | 10 | fate determinism (5 flip arms × 2) |
| `bench7` — K4 | 24 | the f_mix ladder, re-measured |
| `bench5r` | 60 | Θ₀ + the f_A ladder ≤16, re-measured |
| `bench6r` | 60 | the f_A ladder 24–128 + the f_mix head-to-head, re-measured |

Run order and the exact commands: `PLAN.md` §4.

## The one honest caveat, stated up front

**No arm has been run.** `report.html`, `PLAN.md` and `REPRODUCE.md` all mark predicted quantities
`PENDING` and measured ones with their source file. If you find a number here presented as a result
without a `# generated` stamp behind it, that is a bug in this workstream — fix it or flag it.
