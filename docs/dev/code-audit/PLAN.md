# TRINITY `trinity/` full correctness audit — plan

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

**Status (2026-07-30):** 🔵 ACTIVE — method + gates for the `bugfix/code-audit` review of
`trinity/` (72 files, 26,359 lines). Phases 0–2 complete and machine-verified (8/8 artifacts;
partition asserted; 63/63 lens + reconciler reports). Phase 3 at 5/9 sweeps with ①②⑦⑨ in
flight. Phases 4–7 not started. **Do not trust this line — run
`harness/check_completeness.py`**, which is authoritative and was written after this line
said something false.

## Why this audit exists and what it is designed against

The package has had substantial AI assistance. The failure modes that introduces are
specific, and ordinary code review is structurally bad at catching them: **you read the
code and the comment together, and the comment tells you what to see.** A docstring
asserting `Weaver+77 Eq. 20: E = (5/11) L_w t` makes `5.0/11.0` look right whether or
not it is. Every finding-generation step below exists to break that coupling.

| # | Failure mode | Caught by |
|---|---|---|
| 1 | Plausible-but-wrong coefficient / sign / exponent | Lens C re-derivation + literature pinning |
| 2 | Docstring describes intent, code does something else | Lens A vs Lens B diff |
| 3 | Unit mixing (pc/cm, Myr/s, cgs/astro) — the repo's declared bug class | `units-reviewer` sweep + Lens A dimensional pass |
| 4 | Silent fallback swallowing a physics failure | Silent-failure sweep (§Phase 3) |
| 5 | Copy-paste asymmetry — fix landed in one phase runner, not its twin | Duplicate-divergence sweep |
| 6 | Citation correct, *application* outside its validity regime | Lens C reads the paper, not the code |
| 7 | Stale constant tuned for a code path that no longer exists | Magic-number + `git log -S` provenance sweep |
| 8 | Off-grid interpolation on cooling / SPS tables | Table-bounds sweep + runtime sensor |
| 9 | Tests that lock in the bug (golden values from the same code) | Test-suite adversarial audit (§Phase 4) |
| 10 | Global-state leakage between phases | State-mutation sweep + separate-process determinism check |

## Severity rubric — fixed before any finding exists

- **S1 — results-wrong.** Changes physical output on configurations the code is run on today.
- **S2 — latent.** Wrong, but masked by a guard, unreachable in current configs, or cancelling.
- **S3 — misleading.** Code correct, docstring/comment/citation wrong. Will cause a future error.
- **S4 — hygiene.** Dead code, unused contract, duplicated logic, stale artifact.

## Phase 0 — ground truth before any code is judged

| Step | Artifact |
|---|---|
| 0a Baseline | `pytest` counts, `ruff`/`mypy` state, reference runs in separate processes, determinism check → `data/baseline.md` |
| 0b Physics spec (agent reads **no** implementation) | `reference/PHYSICS_SPEC.md` — numbered `SPEC-nnn` claims from the paper + Weaver+77 / Rahner / Bonnor–Ebert literature |
| 0c Structure map (descriptive only, no judgment) | `reference/STRUCTURE_MAP.md` — call graph, state-key write/read table, solver inventory, zero-caller functions |
| 0d Claims ledger (scripted) | `data/claims_{prose,literals,guards,params}.csv` — every prose assertion, numeric literal, failure-swallowing guard, declared-vs-consumed param. Converts "review the code" into a closed checklist. |
| 0e Calibration | `data/calibration.md` — 8 synthetic defects injected into a scratch copy; the pipeline must find ≥6/8 or the prompts get strengthened *before* the real run |

## Phase 1 — disjoint slices

Slices follow physics coherence, not file size. Every one of the 72 files belongs to exactly one.
★ = highest churn × stiffness, gets an extra verification round.

`S1` units & shared helpers · `S2` cloud properties · `S3` phase0 init · `S4` phase1 energy ·
`S5a` β–δ solve ★ · `S5b` implicit runner ★ · `S6` transition + momentum ★ · `S7` bubble structure ★ ·
`S8` shell structure · `S9` cooling · `S10` SPS/feedback · `S11` orchestration & events ·
`S12a/b` input layer · `S13a/b` output layer · `S14` analysis

## Phase 2 — blind-lens triangulation

Per slice, run in parallel with no knowledge of each other:

- **Lens A — what the code does.** Reads a **comment- and docstring-stripped** copy
  (`harness/strip_comments.py`, line numbers preserved exactly). Writes the mathematics
  actually computed, with dimensions. Cannot be misled by a wrong comment: it never sees one.
- **Lens B — what the code claims.** Reads **only** docstrings, comments and doc prose for the
  same slice. Transcribes claimed formula, units, ranges, citations.
- **Lens C — what it should be.** Reads the Phase-0b spec, the literature, and function
  *signatures* only. Derives the expected expression and its validity regime. (Physics tier.)
- **Reconciler.** Fresh agent, receives A/B/C — **never the source**. A≠B ⇒ doc-drift.
  A≠C ⇒ physics. B≠C ⇒ mis-cited literature. All three agreeing on something the spec does not
  sanction ⇒ scope creep.

## Phase 3 — cross-cutting sweeps

Orthogonal to the slices; these catch what slice-local review structurally cannot.
① units & dimensions (via the repo's `units-reviewer`) · ② signs, factors, exponents ·
③ silent failure · ④ duplicate divergence across phase runners · ⑤ dead code & unused contracts
(flag only — CLAUDE.md rule 3) · ⑥ magic numbers + `git log -S` provenance · ⑦ table bounds ·
⑧ state mutation & aliasing · ⑨ numerical hygiene (tolerances, event flags, float equality).

## Phase 4 — adversarial test-suite audit

Tautology hunt (tests that assert the implementation against itself; mocks that remove the
physics; assertions loose enough to pass under a Phase-2 candidate defect) · golden-value
provenance (independently derived, or generated by a prior run of this same code — the latter
*locks in* the bug) · coverage mapped against the finding list.

## Phase 5 — verification gate

A multi-agent audit's dominant failure is confident nonsense. Nothing reaches `FINDINGS.md`
without surviving:

1. A **fresh skeptic** receiving the *claim only, never the finder's reasoning*, with full
   source access and an explicit instruction to refute (default "refuted" under uncertainty).
   S1/S2 and all ★ slices get **three skeptics with different lenses** — dimensional,
   literature, numerical — and need a majority.
2. **One** of: a failing `pytest` case in the real suite · a numeric/dimensional demonstration ·
   a literature citation with the exact equation and regime.

Unsupported candidates are demoted to `UNVERIFIED.md`, clearly labelled — never mixed in.

## Phase 6 — dynamic verification

Static review cannot tell you the physics is right. These can:

- **Asymptotic limits.** Uniform density, constant mechanical luminosity: energy-driven
  `R ∝ t^(3/5)`, `v ∝ t^(-2/5)`; momentum-driven `R ∝ t^(1/2)`. Fit the exponent from
  `dictionary.jsonl`. A dropped term shows up as a wrong slope regardless of what the code says
  about itself.
- **Budget closure** — do the force/energy terms sum to the reported totals at every snapshot?
- **Invariant scan** — NaN/inf first appearance, monotonicity, phase ordering, negative
  densities/temperatures, across all baseline runs.
- **Table-bounds sensor** — instrumented run logging every off-grid cooling/SPS request.
- **Determinism / global-state probe** — same config in separate processes → byte-identical
  output; and second-in-process vs first, to expose leaked module-level state.

Per CLAUDE.md rule 5: separate processes, matched simulation time `t`, stiff edge regimes
(`param/simple_cluster.param` + `docs/dev/performance/f1edge_{lowdens,hidens}*.param`), and
physically plausible parameter values throughout.

## Cross-contamination controls

1. **Disjoint ownership** — each finder gets an explicit file list, may not read outside it.
   The one shared read-only exception (`unit_conversions.py`) is declared in every report.
2. **No agent reads another agent's output.** Fan-out only; reconcilers see reports *without* source.
3. **No finder reads `docs/dev/`** — stale by its own banners, and would anchor every agent to
   prior conclusions including prior wrong ones. Only the orchestrator reads it, at the end.
4. **Blind lenses** — the stripped-source / prose-only / literature-only split above.
5. **Skeptics get claims, not arguments.**
6. **Fixed JSON finding schema** — `id, file, line, class, severity, claim, evidence, expected,
   failure_scenario, repro, confidence` — so agents cannot editorialize toward agreement.
7. **One output file per agent** under `slices/` — no shared file, full provenance.
8. **Finders are read-only.** All source changes are applied by the orchestrator, after the gate.
9. **Off-limits to everyone:** `old_doNotRead/`, `outputs/`, `scratch/`, `tbd/`, `fig/`.

## Completion is checked, not asserted

**Run `python docs/dev/code-audit/harness/check_completeness.py` before claiming any phase
is done, and again before declaring the audit finished.** Exit status 0 is the only thing
entitled to say a phase is complete. Never conclude "done" from a directory listing.

This rule exists because it was broken. On 2026-07-30 Phase 3 was reported complete and the
audit was about to advance to Phase 4 — with **5 of its 9 sweeps** written. The five that
existed were real and thorough, the listing looked substantial, and the four absent ones
(units, signs/factors/exponents, table bounds, numerical hygiene) were simply not noticed.
Sweep ① is the repo's *declared* recurring bug class; skipping it would have carried a
known hole straight into the verification gate.

The checker enumerates every phase's required artifacts, asserts the slice partition, and
prints what is missing. When a phase gains a deliverable, **add it to `PHASES` first** so
the checklist can never lag the plan. A phase with no file-based artifact still gets an
entry with an explicit check (Phase 1 asserts the partition).

## Phases revise each other — the audit is not append-only

Each phase sees things earlier phases could not, so **later evidence must be pushed back
onto earlier conclusions.** A finding is a live claim with a current state, never a fact
fixed at the moment it was written. Three obligations, on every phase:

1. **On entry, re-open.** Before generating anything new, collect the open questions and
   `low`/`medium`-confidence findings from prior phases that *this* phase's evidence can now
   settle, and settle them. Every reconciler ends with a named "open questions" section
   phrased as one-step lookups precisely so a later phase can consume it as a worklist.
2. **Record every revision, in both directions.** `data/resolutions.md` is the register:
   the original claim, the lookup, the verdict, and the severity change. **Downgrades and
   withdrawals are recorded as prominently as upgrades** — an audit that only ratchets
   upward is measuring its own enthusiasm. Precedent from Phase 2: `S9-R-01` (Lens A's
   inference refuted, code cleared), `S12b-B-01` (S2→S3, comment-only), `S13a-B-05`
   (withdrawn — interval notation misread as a Python list literal), `S12a-R-01` (S1
   upheld, medium→high). A prose-only lens reliably yields *true observations with
   over-severe ratings*, because from the prose side a doc defect and a code defect are
   indistinguishable; every B-only claim therefore needs a lookup before it reaches
   `FINDINGS.md`.
3. **Chase clearances too.** Verifying a suspected defect *away* is a result, and it is
   cheap insurance against a future session re-chasing it. It also finds things: the
   `S13b-R-02` lookup cleared the hypothesised 1.85 dex `ZREL` error and, in passing,
   exposed a real `--z-override` validator bypass that no lens was scoped to see.

**Assumptions are provisional until pinned.** Where a phase reasons from an unverified
premise, it must say so and name the lookup that would settle it. Two Phase-2 examples of
premises that failed: "no channel distinguishes solver failure from physical fate" (false —
every phase runner checks `sol.status`; the claim was true only inside one slice and got
generalised), and "nothing changes numbers on a nominal run of a *tracked* config" (sound
generally, circular for the parameter reader, whose whole job is untracked configs).
Agreement between lenses is **not** verification — all three can share a wrong premise.

## Deliverables

`README.md` (verdict) · `FINDINGS.md` (ranked S1→S4, each with repro + fix outline) ·
`UNVERIFIED.md` (demoted candidates) · `slices/*.md` (raw per-agent reports) ·
`data/resolutions.md` (the revision register) · `harness/` · `data/` · new `pytest` cases
for every confirmed S1/S2, failing first.

**Final gate.** The audit is done when `check_completeness.py` exits 0, `pytest` is green,
and every finding in `FINDINGS.md` carries its current (post-revision) severity — not the
one it was born with.

Fixes are staged **separately** from the audit, by severity, on maintainer approval. Anything
touching a solver, residual or hot loop goes through the CLAUDE.md rule-5 ladder: gate defined
first, baseline captured, full-run equivalence on stiff regimes in separate processes at
matched `t`, apply, re-verify.
