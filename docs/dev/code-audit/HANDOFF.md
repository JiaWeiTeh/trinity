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

**Status (2026-08-08):** 🔵 ACTIVE — all seven phases pass `check_completeness.py`,
and all four remaining Phase-6 probes are closed. **The fixing stage is open on this
branch** (maintainer instruction, 2026-08-08): `NUM-02`, `S11-R-02` and `S12b-R-01`
are fixed in commit `62c1810`, gate PASS. Everything else remains findings-only.

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

1. **Finders are read-only; fixes are deliberate and gated.** This held strictly
   through commit `b05725b` — `git diff origin/main HEAD -- trinity/ test/ run.py
   param/` was **empty**, verified at every commit. ⚠️ **It no longer holds, by
   instruction:** from `62c1810` the fixing stage shares this branch, so that diff is
   *expected* to be non-empty. **Do not "restore" it.** `PLAN.md` still prefers a
   separate branch; the maintainer chose otherwise. Every fix must still clear the
   CLAUDE.md rule-5 ladder with a gate defined *before* editing.
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

690 findings from 26/26 sources. After revision: **13 S1**, 196 S2, 250 S3, 220 S4,
plus **5 FIXED** (2 on `main`, 3 on this branch), 2 CLEARED, 3 REFUTED, 1 WITHDRAWN.

*(2026-08-08, in order: S1 15 → 16 and S2 197 → 196 when `SIGN-01` (`gamma_adia`) was
re-rated S2 → S1; then S1 16 → 13 and FIXED 2 → 5 when `NUM-02`, `S11-R-02` and
`S12b-R-01` were fixed.)*

**Verification removed or demoted more S1s than it confirmed** — that is the headline
about the *method*, and it is why `UNVERIFIED.md` exists as a separate file.

### The 5 verified S1-class findings — ✅ 2 of them now fixed

| id | finding | evidence |
|---|---|---|
| ✅ `NUM-02` / `S11-R-01` / `DD-001` / `ST-002` | `check_event_termination` returned the first event **by list index**, never reading `event.terminal`. `velocity_sign` is index 0 and non-terminal; indices 1-3 all end the run | source-verified; found by **4** passes. **FIXED `62c1810`**, gate PASS |
| `S12a-R-01` | user-set `mu_convert`/`mu_atom`/`mu_ion`/`mu_mol` silently overwritten (`read_param.py:316-319`); anti-stomp guard compares object identity so it cannot fire | source-verified |
| ✅ `S12b-R-01` | `generate_run_name` not injective + no duplicate guard ⇒ a sweep config is never run while the report claims success | gate 3/3; reproduced. **FIXED `62c1810`** — expansion now raises |
| `S8-R-02` | `n_IF_Str == shell_n0` **bit-identical at every snapshot** ⇒ `P_HII` is a re-expression of `Pb` | Phase 6 dynamic |

### The 8 S1-rated candidates never tested

`S11-R-03`, `S5b-R-01`, `S6-R-01`, `DD-003`, `DD-004`, `SF-003`, `SF-004`, `SF-005`.
Prior from the 7 that *were* panelled: **2 removed outright, 3 demoted** — but the two
settled *dynamically* (`S8-R-02`, `S11-R-02`) were both **confirmed**, and `S11-R-02`
came out **larger** than claimed. The shrink prior applies to skeptic panels, not to
run-based checks.

## What was fixed (by the maintainer, not the audit)

`hotfix/early-approximations` — the `vd = -1e8` override — merged to `main`. Both
halves gone, zero references remain, with a **property-based** regression test
(`test/test_early_phase_override.py`) that is the pattern the suite's ~105 captured
goldens lack. Full suite green: **1093 passed, 15 deselected**.

## Recommended next steps, in order

1. ~~**Test `S11-R-02` (isCollapse).**~~ **DONE 2026-08-08 — confirmed and widened.**
   See `FINDINGS.md` §5 and `resolutions.md#s11-r-02`. The claimed false negative is
   real but largely masked in phases 1b/1c/2 by a redundant `v2<0 and R2<R2_prev`
   detector (`run_energy_phase.py` has none, so phase 1a is unmasked). A **second**
   misclassification the claim never mentioned is *not* masked: `large_radius_event`
   contains the substring `radius`, so a shell **expanding** through `stop_r` — a
   clean `LARGE_RADIUS` success — is latched `isCollapse = True`, and nothing ever
   resets the flag. Repro: `harness/probe_iscollapse.py`.
   **Still open:** reachability at the shipped `stop_r = 500` pc (proven only at 3 pc).
2. ~~**Settle the `gamma_adia` vs `mu_*` severity question.**~~ **DONE 2026-08-08 —
   both S1.** `SIGN-01` re-rated S2 → S1; `S12a-R-01` unchanged. Reasoning in
   `resolutions.md`. Short form: neither is guard-masked or cancelling, both are
   documented `default.param` keys, so the rubric's "unreachable in current configs"
   escape does not apply to either — and the asymmetry runs *opposite* to the original
   ratings, since `mu_*` is silently **ignored** (self-consistent run) while
   `gamma_adia` is silently **half-honoured** (internally inconsistent run, 67 %
   pressure imbalance at γ=1.4).
3. ~~**Finish the 4 open Phase-6 probes.**~~ **DONE 2026-08-08 — all four closed.**
   Details in `data/dynamic_verification.md` §4, §8, §9, §10.
   - **`TBL-01`** — CONFIRMED (mechanism). It did *not* need the 10 Myr run: only the
     *frequency* does. The bundled age grid is 1,2,3,4,5,10 Myr; past 1e7 yr
     `get_filename` returns the last-grid file with no warning, so a default
     `stop_t = 15` Myr run spends its **last 33 %** on cooling frozen at 10 Myr.
     *Frequency still open* — no run here has survived to 10 Myr.
   - **`W-3`** — CONFIRMED, and it bears on **`SF-003`** (same code site). Zero
     occurrences across every log, so the mechanism was forced: any exception becomes
     the constant `(100.0, 100.0)` plateau, and a `WARNING` line is the only trace.
   - **Momentum asymptotics** — DONE. `R2 = +0.542` vs ideal `+0.500` at **rms 0.0024
     dex**: the `t^(1/2)` limit is recovered. `v2` (rms 0.0193) is *not* quotable.
     Bonus: it **tests** §3's implicit-phase explanation and confirms it.
   - **Different-config in-process** — **PASS**, byte-identical. A *negative* result:
     the documented in-process leak did not reproduce.

   **New open items these produced:** `TBL-01` frequency; a longer different-config
   pair straddling two cooling-table ages; and the transition phase fitting at
   `rms 0.0000` (§9) — recorded as an observation, not a finding.
4. ~~**Then, and only on maintainer approval, open the fixing stage.**~~ **OPENED
   2026-08-08 on maintainer instruction — on *this* branch, not a separate one.**
   Three fixes have landed (commit `62c1810`); see `data/gate_event_dispatch_fix.md`.

   | finding | fix | gate |
   |---|---|---|
   | `NUM-02` | `check_event_termination` ranks run-ending events above monitoring ones, earliest root within a rank | **PASS** |
   | `S11-R-02` | `isCollapse` keyed on `v2 < 0` at exit, not a `reason_code` substring | **PASS** |
   | `S12b-R-01` | sweep expansion raises on a duplicate run name instead of dropping a cell | tests |

   **Equivalence gate result:** **0 physics columns differ** across 464 snapshots,
   three configs, all four phases, separate processes, row counts unchanged
   (155/195/114). The *only* changed value anywhere is `maxr` row 154,
   `isCollapse` True → False at `v2 = +21.55` pc/Myr — an expanding shell, correct
   to clear. 7 new regression tests, failing before and passing after.

   ⚠️ **The findings-only invariant therefore ends at `b05725b`.** From `62c1810`
   on, `git diff origin/main HEAD -- trinity/ test/ run.py param/` is *expected* to
   be non-empty. Do not "restore" it.

5. **Still unfixed, deliberately** — `mu_*` (`S12a-R-01`) and `gamma_adia`
   (`SIGN-01`), both S1. The evidence was delivered to the maintainer for
   verification and no code was written. Cheapest close for both is **input
   validation**, not reimplementation: refuse a user-set `mu_*` (currently
   overwritten at `read_param.py:316-319`, guard structurally blind) and refuse
   `gamma_adia != 5/3` (honoured by `bubble_E2P`/`get_leak_luminosity`, absent from
   `solve_R1`, so the two disagree by 67 % at γ=1.4). That converts two silent
   wrong-answer paths into loud failures without touching physics.

   ⚠️ **Unverified sub-claim:** sweep ②'s phrasing "the Rahner-A12 pair and the whole
   Weaver structure chain hardcode it" goes beyond what has been checked. The **R1
   half is verified** (`get_bubbleParams.py:408`, `solve_R1` takes no γ); the wider
   claim is not.

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
