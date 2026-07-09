# PLAN — the repo-wide execution queue

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

**Status (2026-07-06):** 🔵 ACTIVE — queue seeded from DOC_STATUS open tails + the 2026-07-06
solver audit; recommended order in §3.

## §0 Operating principle (why this doc exists)

Strong-model sessions are the scarce resource. Spend them where the capability gap is largest —
**deep diagnosis, judgment calls, design, critique** — and freeze every conclusion into the repo
as something executable: a test, a gate, a harness, a sequenced plan. Then any later session
(any model) executes against checks instead of re-deriving the thinking.

**Tier tags** on every item:
- **[M] mechanical** — the thinking is done and frozen; execute the smallest diff that passes
  the stated gate. Safe for any model. Do not re-litigate the design; if the gate seems wrong,
  stop and flag rather than improvise.
- **[J] judgment** — diagnosis/classification/design still open; needs a strong model or the
  maintainer. The deliverable of a [J] item is usually a new gate that turns the rest into [M].
- **[X] maintainer decision** — blocked on a human ruling.

Every item lists: why · gate (pass/fail, executable) · owner doc (detail lives THERE, per
`docs/dev/CONVENTIONS.md` — this queue never duplicates it).

## §1 Lanes

### Lane A — physics program (paper-critical path)

Owned by `docs/dev/transition/pdv-trigger/` (INDEX.md §3 is the live thread). Queue view only:

| id | item | tier | gate | owner |
|---|---|---|---|---|
| A1 | f_A source-term **L1**: stiff-fixtures screen | [M] | the L1 pass/fail bar in the design doc | `transition/pdv-trigger/SOURCE_TERM_DESIGN.md` |
| A2 | f_A **L2**: gated param + full 📏 theta5-protocol matrix | [M run, J read-out] | 📏 protocol: 8(9) configs, ≥5 Myr, θ_max, separate processes; hard guardrail: no production change before all configs pass | same + `runs/README.md` |
| A3 | revalidate `cooling_boost_kappa='auto'` (63-cell grid) or keep opt-in-provisional | [M run, X ruling] | re-measured under 📏 protocol | `transition/pdv-trigger/FINDINGS.md` §9 flags |
| A4 | β–δ Phase-5 root fix (mixing-layer cooling/leakage, Eb-peak handoff) | [J] | full-run equivalence on stiff regimes (rule-5 depth) | pdv-trigger `PLAN.md`; history `archive/betadelta/HYBR_PLAN.md` |

### Lane B — correctness & bug-findability (seeded by `solver-audit.md`, 2026-07-06)

| id | item | tier | gate | owner |
|---|---|---|---|---|
| B0 | ✅ DONE 2026-07-06: `solve_R1` non-finite guard (audit F1) | — | `test_r1_bracket.py` + `test_energy_collapse_guard.py` green on scipy 1.10.1 and CI | `solver-audit.md` F1 |
| B1 | ✅ DONE 2026-07-06: duplication sync gate | — | `test/test_phase_helper_sync.py` green | `solver-audit.md` F5 |
| B2 | **Per-phase fast regression fixtures** (audit F4) — *do this before any other solver edit; it is the gate the rest of this lane runs against* | [M; param choice J-lite] | committed fast `.param` per phase, verified to reach 1c and 2; test asserts outcome + snapshot count + key finals vs committed reference | `solver-audit.md` F4 |
| B3 | P_ext silent-zero fix (audit F2) | [M] | monkeypatch test (warning fires, fallback preserved) + byte-identical `dictionary.jsonl` on `simple_cluster` + f1edge pair | `solver-audit.md` F2 |
| B4 | narrow the residual-path `except` so code bugs propagate (audit F3) | [J classify, M apply] | injection tests (TypeError propagates; ValueError → penalty) + full-run equivalence, stiff regimes, separate processes | `solver-audit.md` F3 |
| B5 | classify the forces-trio divergence (audit F5) | [J] | every hunk labeled intentional/missed-fix, in-code comments or synced; only then extend sync test or consolidate (`REORG.md` R2) | `solver-audit.md` F5 |
| B6 | production provenance in `metadata.json` (commit, dirty flag, param sha256, argv) | [M] | metadata test; run still works with no git on PATH | `REORG.md` R4 |
| B7 | `--debug` CLI override for log level; print→logger sweep | [M] | R3/R5 gates | `REORG.md` R3, R5 |
| B8 | locate & enumerate the in-process global-state leak (audit F6) | [J] | the audit's F6 paragraph replaced by the actual object list (+ optional double-run test) | `solver-audit.md` F6 |
| B9 | dev env ≥3.9 (local anaconda is 3.8.8 — below the floor; scipy 1.10.1 caused F1's ghost failures) | [X + M] | `python3 --version` ≥3.9 in the dev shell; full pytest baseline re-recorded | `solver-audit.md` F1 |

### Lane C — long tails carried from `docs/dev/DOC_STATUS.md` (detail lives in the owner docs)

| id | item | tier | owner |
|---|---|---|---|
| C1 | backward-compat cleanup (~95% unexecuted) | [M] | `misc/backward-compat-audit.md` |
| C2 | magic numbers #2–#5 (+ audit F8 tolerances) | [J justify, M fix] | `magic-numbers/AUDIT.md` |
| C3 | HOTPATH §F1-cousin + §F5 | [J] | `performance/HOTPATH_PLAN.md` |
| C4 | leaking luminosities Phase D/F/G + findings #7/#8 | [J] | `misc/LEAKING_LUMINOSITIES_SKELETON.md` |
| C5 | cooling loader refactor PR-1–4 | [M] | `cooling/refactor-audit.md` |
| C6 | tinit recommendation #3 (drop linear L3 patch) | [J] | `misc/tinit-sensitivity.md` |
| C7 | `caseB_alpha` stored in AU (mixed-unit correctness) | [J] | `shell-solver/OVERFLOW_FIX_PLAN.md` |

### Lane D — deletion review

| id | item | tier | owner |
|---|---|---|---|
| D1 | review the 2026-07-06 round in `docs/dev/to-be-removed/` (1 file moved, 3 flagged in place) | [X] | `to-be-removed/README.md` |

## §2 Sequencing logic

1. **B2 first among code work.** Fast per-phase fixtures multiply the safety of every other
   solver-touching item (B3, B4, B5, A4, C3…). Until they exist, every solver edit pays the
   full separate-process manual-gate cost.
2. **Lane A outranks lane B overall** — it is the paper-critical path and A1/A2 are already
   fully specified ([M]) in SOURCE_TERM_DESIGN.md. Physics runs (Helix queue time) and code
   work (local) don't contend, so interleave: submit A runs, do B work while they cook.
3. **B3/B4 ride B2's fixtures.** Cheap diffs, pre-written gates; do them in one sitting.
4. **[J] items (B5, B8, C-lane diagnoses) are strong-model work** — batch them for capable
   sessions; don't let a mechanical session improvise a classification.
5. Lane C stays behind A and B unless one of its items starts blocking (C7 touches
   correctness — pull it forward if shell-solver work resumes).

**Recommended next five, in order: A1 → B2 → B3+B4 → A2 → B6.**

## §3 Session ledger (newest first)

- **2026-07-06** — workstream created. Solver audit ran (`solver-audit.md`); F1 diagnosed
  (scipy-1.10.1 brentq NaN behavior, env below floor) and fixed; sync gate
  `test/test_phase_helper_sync.py` added; deletion round staged in `to-be-removed/`;
  `REORG.md` written for mechanical hand-off. Queue seeded from DOC_STATUS open tails.
