# solver-audit — deep critique of the solver core (2026-07-06)

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

**Status (2026-07-06):** 🔵 actionable — F1 fixed same session; F2–F8 open, each with a gate.

## Scope & method

Reviewed: the solver core (~6.4k lines) — phase runners (`phase1_energy/`,
`phase1b_energy_implicit/`, `phase1c_transition/`, `phase2_momentum/`), the β–δ solve
(`get_betadelta.py`), bubble structure (`bubble_structure/`), shared helpers (`_functions/`).
Method: three parallel read-only exploration sweeps (code map, dead code, bug-surfacing
workflow), then hand-verification of every finding below at commit `70f07532` — line numbers
cite that commit. Complements, does not repeat: `docs/dev/CODEBASE_REVIEW.md` (52 findings,
2026-06-16, file-by-file) and `transition/pdv-trigger/FINDINGS.md` (physics). This audit is
about *how the code fails*, not what physics it implements.

Each finding: severity · location · failure scenario · **frozen check** (the executable thing
that pins the conclusion) · fix outline · tier (**[M]** mechanical — any model can execute
against the gate; **[J]** judgment — needs a strong model or the maintainer).

---

## F1 — ✅ FIXED (2026-07-06): `solve_R1` fabricated a root on NaN input under scipy < 1.11

- **Was:** the two pre-existing test failures at HEAD
  (`test_energy_collapse_guard.py::test_solve_R1_returns_zero_for_nonphysical_R2`,
  `test_r1_bracket.py::test_failure_raises_instead_of_fabricating`).
- **Diagnosis (measured, not guessed):** local env is python 3.8.8 + scipy 1.10.1 — *below the
  project's ≥3.9 floor*. On scipy 1.10.1, `brentq` on a NaN-poisoned function **silently
  returns ~1.1e-12** (verified: `brentq(lambda x: np.nan, 0, 10)` → `1.1368683772161603e-12`,
  no raise). Modern scipy (CI, py3.9–3.12) raises ValueError. So
  `solve_R1(R2=10, Eb=np.nan, …)` fabricated a tiny R1 locally — exactly what its docstring
  promises never to do. **Not a code regression; a version-dependent contract.**
- **Fix (applied):** explicit `np.isfinite` guard on `Eb`/`Lmech_total`/`v_mech_total` in
  `trinity/bubble_structure/get_bubbleParams.py::solve_R1` — raises ValueError regardless of
  scipy version. Healthy path (finite inputs) unchanged; non-finite inputs already raised on
  supported scipy, so CI-visible behavior is identical.
- **Frozen check:** the two test files above — now green on scipy 1.10.1 *and* modern scipy
  (14/14, verified 2026-07-06).
- **Follow-up [M]:** the dev machine should get a ≥3.9 env (PLAN item B9) — 3.8 is below the
  floor and will keep producing version-skew ghosts like this one.

## F2 — HIGH: photoionized external pressure silently zeroed on any exception

- **Location:** `phase1b_energy_implicit/run_energy_implicit_phase.py:509-516` @ `70f07532` —
  `compute_forces_pure()`:
  `try: n_r = get_density_profile(...); P_ext = ... except Exception: P_ext = 0.0` — no log,
  broad catch.
- **Failure scenario:** any bug in `get_density_profile` (renamed param key, unit regression,
  shape change) makes `P_ext = 0` for **every segment of every run**, permanently, with zero
  signal anywhere. The force budget is then wrong (missing inward term), trajectories shift,
  and nothing — not the log, not metadata, not termination_debug — records that the except arm
  ever fired. This is the highest-value silent failure found.
- **Frozen check (to write with the fix):** unit test that monkeypatches
  `get_density_profile` to raise and asserts (a) a `logger.warning` fires, (b) `P_ext == 0.0`
  fallback still applies (behavior preserved; visibility added). Full-run gate: byte-identical
  `dictionary.jsonl` on `param/simple_cluster.param` + the two f1edge params (healthy paths
  must never take the except arm — if they do, that is itself a discovery to investigate
  before changing anything).
- **Fix outline [M]:** keep the fallback, narrow the catch to numeric errors
  (`ValueError, FloatingPointError, KeyError` as observed), add a streak-limited
  `logger.warning` (pattern: `BETADELTA_UNCONVERGED_WARN_STREAK` in the same file).
- Same pattern, lower stakes: `run_energy_implicit_phase.py:434-435` — monitor-value
  collection `except Exception: pass` (diagnostics dict silently incomplete). Fix alongside.

## F3 — MEDIUM-HIGH: broad `except` converts code bugs into the "no physical root" physics signal

- **Location:** `phase1b_energy_implicit/get_betadelta.py:435-439` @ `70f07532` —
  `get_residual_pure()` catches `Exception` from `get_bubbleproperties_pure()`, logs a
  warning (it does log — earlier sweep claims of "unlogged" were wrong), returns the
  `(100.0, 100.0, None)` penalty. The hybr gate then reads structure failure as
  `no_physical_root`, and `NO_ROOT_HANDOFF_STREAK=50` (`run_energy_implicit_phase.py:122`)
  routes the run to the momentum phase.
- **Failure scenario:** a refactor introduces an `AttributeError` in the structure solve for
  some corner regime → every solve in that regime returns the penalty → hybr reports no root →
  after 50 segments the run **hands off to momentum and "completes"** with a physically wrong
  fate. A code bug is indistinguishable from real condensation physics in the fate routing;
  only a log-reader would notice.
- **Frozen check (to write with the fix):** unit test injecting a `TypeError` into
  `get_bubbleproperties_pure` and asserting it **propagates** (run dies loudly), plus one
  injecting `ValueError` (numeric) and asserting the penalty path still works.
- **Fix outline [J then M]:** classify which exception types the structure solve legitimately
  raises on stiff-but-physical inputs (judgment: read `bubble_luminosity.py` failure modes;
  the `MonotonicError` class already exists for one of them) — then catch only those; let
  programming errors (`AttributeError, TypeError, NameError, KeyError`) propagate. Full-run
  gate on the f1edge pair + `simple_cluster` (hot-loop change → CLAUDE.md rule 5 depth:
  separate processes, matched t; expected bit-identical since healthy paths never raise).

## F4 — MEDIUM-HIGH: three of four phase runners have no direct tests

- **Facts:** `run_phase_energy()` (~830 lines, `run_energy_implicit_phase.py:631`),
  `run_phase_transition()` (~520 lines, `run_transition_phase.py:367`),
  `run_phase_momentum()` (~471 lines, `run_momentum_phase.py:461`) have **zero direct tests**.
  Coverage is whatever path `test_run_smoke.py`'s single quickstart config happens to take,
  plus the β–δ solver's own excellent suite (6 files) for the 1b *inner* solve only.
- **Failure scenario:** an edit to the 1c sound-crossing fallback or the momentum-phase force
  loop passes the whole pytest suite; the break surfaces days later as a wrong fate in an
  hours-long Helix run — the most expensive possible way to find it.
- **Frozen check = the fix [M, param choice is J-lite]:** one fast full-run regression fixture
  per phase — a committed `.param` that *provably reaches* 1c and 2 quickly (start from
  `transition/cleanroom/configs/` + `docs/dev/performance/f1edge_*.param`; verify phase entry
  in the output before committing), asserting termination outcome + snapshot count + a few
  key finals against committed reference values. This is PLAN item **B3** and deliberately
  sits *before* every other solver-touching item in the queue: it is the gate the rest of the
  queue executes against.

## F5 — MEDIUM: phase-runner duplication — one trio verbatim, one trio silently diverged

- **Verbatim trio (now gated):** `compute_max_dex_change()` exists in 1b/1c/2; logic
  AST-identical (docstrings differ). A fix to one copy is a latent phase-dependent bug.
  **Frozen check (added this session, passing):**
  `test/test_phase_helper_sync.py::test_compute_max_dex_change_copies_identical` — fails the
  moment the copies diverge. Consolidation is `REORG.md` **R1**.
- **Diverged trio (unclassified):** `compute_forces_pure()` (1b:460, 1c:271) and
  `compute_forces_momentum_pure()` (2:206) — measured AST dump lengths 8789 / 9623 / 8942;
  1b ≠ 1c. ~300 lines of near-parallel force logic whose differences are **either intentional
  phase physics or missed fixes — nobody currently knows which**.
- **Failure scenario:** the historical bug class this repo already documents (units, force
  budget) fixed in 1b only; 1c/2 keep the wrong term; energy-phase and momentum-phase
  trajectories disagree for a non-physical reason.
- **Fix outline [J]:** diff the three, classify every hunk `intentional-phase-physics`
  (document why, in-code comment) or `missed-fix` (sync it), then either consolidate
  (`REORG.md` R2) or extend the sync test to the documented-common parts. Do NOT add the
  forces trio to the sync test before classification — it would fail on day one and teach
  people to ignore it.

## F6 — MEDIUM: the in-process global-state leak is documented but not located

- CLAUDE.md rule 5 mandates separate-process comparisons because "trinity leaks module-level
  global state in-process" — but this audit's sweep of the core files found only frozen
  dataclasses (`CONV`, `INV_CONV`, `CGS`) and no module-level mutables there. **The actual
  leaking objects were not located this pass.** Candidates to check: cooling-curve /
  SB99-interpolator caches, `DescribedDict` aliasing, warm-start state dicts, anything
  memoized at import.
- **Why it matters:** every equivalence gate costs 2 process spawns because of an
  unenumerated constraint. Enumerate it once and the constraint becomes checkable.
- **Fix outline [J]:** locate and list every module-level mutable touched during a run
  (import-time diff or `gc`-based sweep); record the list HERE (replace this paragraph);
  optionally add a test that two short back-to-back in-process runs either match or fail with
  a pointer to the list. Until then: separate processes stays the law.

## F7 — LOW (grouped): visibility & hygiene

- `phase2_momentum/run_momentum_phase.py:398` — `R2 = max(R2, 1e-10)` silent clamp inside the
  ODE RHS. Legitimate trial-state guard (same rationale as `solve_R1`'s documented R2≤0 → 0),
  but undocumented. Fix: one comment linking the rationale; no log (RHS-hot). [M]
- ~32 `print()` calls in phase/solver modules bypass the log file (`trinity_*.log`) —
  a crashed run's stdout is lost unless the terminal was captured. `REORG.md` R3. [M]
- dt-strategy asymmetry 1b (fine, `ADAPTIVE_THRESHOLD_DEX=0.05`) vs 1c (coarse, 0.1) is
  acknowledged only by a TODO (`run_energy_implicit_phase.py:109`). Decide intentional-or-not
  once, document at both sites. [J-lite]
- Production runs record **no provenance** (no commit hash, param hash, or command line in
  `metadata.json`; `run_stamped.py` is opt-in research tooling). `REORG.md` R4. [M]

## F8 — LOW: magic-number registry for the solver core

`run_energy_implicit_phase.py:109-181` and `get_betadelta.py:47-74` carry ~20 named constants
(good: named, commented) — but `magic-numbers/AUDIT.md` items #2–#5 remain open and several
tolerances (`RESIDUAL_THRESHOLD=1e-4`, `LBFGSB_FALLBACK_THRESHOLD=5.0`, hybr `eps=3e-4`) have
no recorded justification. Not urgent; folded into PLAN lane C (C2). [J for the justification,
M for the bookkeeping]

---

## HOTSPOT map (where to send the next reviewer first)

1. `run_energy_implicit_phase.py:631` `run_phase_energy` — 830 lines, no direct test, F2+F3 live here.
2. `run_transition_phase.py:367` `run_phase_transition` — 520 lines, zero tests (F4).
3. `get_betadelta.py:678` `_solve_betadelta_legacy` — 191 lines, churn-heavy (rescue ladder 92d22a13).
4. `get_betadelta.py:649` `_rescue_structure_failure` — 3-rung ladder; the re-seed path has no dedicated test.
5. `bubble_luminosity.py:582` `_bubble_luminosity` — 279-line integration; monotonic-guard interaction is the numpy-2.x pin.
6. The forces trio (F5) — pending classification.

*Reproduce this audit's measurements:* AST comparisons — `python3` + `ast.dump` on the named
functions (exact snippet in `test/test_phase_helper_sync.py`); scipy behavior —
`python3 -c "import scipy.optimize,numpy as np; print(scipy.optimize.brentq(lambda x: np.nan, 0, 10))"`
on scipy 1.10.1 vs ≥1.11. No sim runs were needed; nothing here requires Helix time to re-check.
