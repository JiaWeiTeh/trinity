# HANDOFF — where the code-audit stands and what to do next

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

**Status (2026-07-30):** 🔵 ACTIVE — all seven phases pass `check_completeness.py`.
Findings only; **no source has been fixed by the audit**. Written to hand off to a
fresh session.

## Background in one paragraph

`bugfix/code-audit` is a full correctness audit of the `trinity/` package (72 files,
~26k lines), built specifically against the defect classes heavy AI assistance
introduces — plausible-but-wrong coefficients, docstrings that describe intent while
the code does something else, silent fallbacks, copy-paste divergence between phase
runners. The method is blind-lens triangulation: per slice, one agent reads a
**comment-stripped** copy (what the code does), one reads **only** comments and
docstrings (what it claims), one reads the **spec and literature** (what it should
be), and a fourth reconciles their reports **without ever seeing the source**. Then
cross-cutting sweeps, an adversarial test-suite audit, a skeptic gate, and dynamic
verification against real runs. Full method in `PLAN.md`.

## Ground rules that must carry over

1. **This workstream is findings-only.** Finders are read-only; fixes are staged
   separately, by severity, on maintainer approval. Verified as of this handoff:
   `git diff origin/main HEAD -- trinity/ test/ run.py param/` is **empty**.
2. **Max 4 agents per launch, then stop and report** (`PLAN.md` §"Batch size is
   capped"). This session spent **5.27M tokens over 39 agents** in one unbroken run
   before that rule existed. Cost tracks *scope*, not agent count: lenses with an
   explicit 2-8 file list averaged 102k; sweeps told to read `trinity/**` averaged
   272k. Measured spend is in `data/agent_costs.md`.
3. **Ask the checker, never the prose.** `harness/check_completeness.py` is the only
   authority on what is done. It exists because Phase 3 was reported complete at 5/9
   sweeps.
4. **Phases revise each other.** Later evidence gets pushed back onto earlier
   conclusions, and **downgrades are recorded as prominently as upgrades**. The
   register is `data/revisions.csv` (machine) + `data/resolutions.md` (narrative).

## Where to start reading

| file | what it is |
|---|---|
| `FINDINGS.md` | the deliverable — ranked, with repro + fix outline, every severity current |
| `UNVERIFIED.md` | candidates removed by verification, demoted, or never tested. **Do not act on these** |
| `data/resolutions.md` | every lookup and verdict, with evidence and corrections |
| `data/revisions.csv` | machine-readable severity history (birth → current) |
| `data/dynamic_verification.md` | Phase 6 — the run-based probes |
| `PLAN.md` | method, gates, batch cap, revision protocol |

Regenerate counts: `python docs/dev/code-audit/harness/collect_findings.py`

## Current numbers

690 findings from 26/26 sources. After revision: **15 S1**, 197 S2, 250 S3, 220 S4,
plus 2 FIXED, 2 CLEARED, 3 REFUTED, 1 WITHDRAWN.

**Verification removed or demoted more S1s than it confirmed** — that is the headline
about the *method*, and it is why `UNVERIFIED.md` exists as a separate file.

### The 5 verified S1-class findings

| id | finding | evidence |
|---|---|---|
| `NUM-02` / `S11-R-01` / `DD-001` / `ST-002` | `check_event_termination` (`phase_events.py:392`) returns the first event **by list index**, never reads `event.terminal`. `velocity_sign` is index 0 and documented non-terminal; indices 1-3 are all simulation-ending | source-verified; found independently by **4** passes |
| `S12a-R-01` | user-set `mu_convert`/`mu_atom`/`mu_ion`/`mu_mol` silently overwritten (`read_param.py:316-319`); anti-stomp guard compares object identity so it cannot fire | source-verified |
| `S12b-R-01` | `generate_run_name` not injective + no duplicate guard ⇒ a sweep config is never run while the report claims success | gate 3/3; reproduced |
| `S8-R-02` | `n_IF_Str == shell_n0` **bit-identical at every snapshot** ⇒ `P_HII` is a re-expression of `Pb` | Phase 6 dynamic |

### The 10 S1-rated candidates never tested

`S11-R-02` (isCollapse substring — **reaches published figures** via
`paper/_lib/plot_markers.py`), `S11-R-03`, `S6-R-01`, `DD-003`, `DD-004`, `SF-003`,
`SF-004`, `SF-005`. Prior from the 7 that *were* tested: **2 removed outright, 3
demoted**. Expect similar.

## What was fixed (by the maintainer, not the audit)

`hotfix/early-approximations` — the `vd = -1e8` override — merged to `main`. Both
halves gone, zero references remain, with a **property-based** regression test
(`test/test_early_phase_override.py`) that is the pattern the suite's ~105 captured
goldens lack. Full suite green: **1093 passed, 15 deselected**.

## Recommended next steps, in order

1. **Test `S11-R-02` (isCollapse).** Highest value of the untested ten: `isCollapse`
   is set by substring match, `'velocity_runaway_event'` matches nothing, and the
   flag is consumed by `paper/_lib/plot_markers.py` — so a mis-classification reaches
   published figures. Try **dynamically first**: force a runaway-infall termination
   and read the flag. Phase 6 showed dynamic checks are the cheapest verification
   available (one run + an existing harness settled `S8-R-02`).
2. **Settle the `gamma_adia` vs `mu_*` severity question.** `gamma_adia` is
   user-settable (`default.param:251`) and hardcoded `5/3` through the Weaver chain —
   structurally identical to the `mu_*` finding rated **S1**, but sweep ② rated it
   **S2**. A rubric-boundary question, not a facts question. **They must ship rated
   alike.**
3. **Finish the 4 open Phase-6 probes** — `TBL-01` (needs `t > 10` Myr; my attempt
   reached 0.2 %), the `W-3` swallowed-error probe, momentum-phase asymptotics (no run
   yet wrote more than one momentum snapshot), and the *different-config* in-process
   case.
4. **Then, and only on maintainer approval, open the fixing stage** — separate branch,
   by severity, each fix through the CLAUDE.md rule-5 ladder. Start with the event
   dispatch: it is the only verified S1 that changes a recorded physical *fate*.

## Traps a fresh session will otherwise fall into

- **Do not "fix" anything in `UNVERIFIED.md` §A.** Two were born S1. The `pdot`
  factor-of-2 "fix" would have **doubled the ram pressure** — the 2 is already there,
  cancelled into the `2π` of `pRam = Lmech/(2π r² v_mech)`.
- **`calibration_reconciled.md` is not findings.** It is Phase 0e's eight
  *deliberately injected synthetic defects*. `collect_findings.py` excludes it by
  name; do not re-add it.
- **Severity in slice reports is the birth value.** Always read `current_severity`
  from the inventory, which overlays `revisions.csv`.
- **Agreement between lenses is not verification.** All three can share a wrong
  premise; that is how "no channel distinguishes solver failure from physical fate"
  survived until it was checked and found false.
- **Fit scatter before fit value.** A self-similar exponent fitted over a
  non-self-similar window is a confident meaningless number — it nearly shipped here
  (`data/dynamic_verification.md` §3).
