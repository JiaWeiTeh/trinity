# docs/dev — internal development notes

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

**Status (2026-07-06):** 📘 navigation index — rebuilt by the docs/dev housekeeping pass.

## What this is

Internal **plan / audit / diagnostic** write-ups for TRINITY development, grouped one folder per
**workstream**. This tree is **not** part of the rendered documentation (Sphinx builds only
`docs/source/`) and is **not** packaged (`pyproject` excludes `docs*`) — it is a working record of
investigations, not a maintained spec.

**Where things are recorded** (exactly one place each):

- **How to write/maintain docs here** → [`CONVENTIONS.md`](CONVENTIONS.md) — banners (four for
  active docs, ⚠️+🧊 for archived), Status-line format, workstream folder template, naming,
  citation and provenance rules. Canonical banner text is in `docs/dev/CLAUDE.md`;
  `test/test_docs_dev_conventions.py` enforces the mechanical parts.
- **Per-workstream verdicts** → [`DOC_STATUS.md`](DOC_STATUS.md) (one row per workstream).
- **Per-doc status** → each doc's own dated `**Status (…):**` line, under its banners.
- **Navigation** → this README, and nothing else. Adding a workstream? Follow the checklist in
  `CONVENTIONS.md` (README line + DOC_STATUS row + entry-point README).

## Layout

```
docs/dev/
├── README.md              this index (navigation only)
├── CONVENTIONS.md         how to add/maintain docs here — read before writing
├── DOC_STATUS.md          per-workstream status ledger
├── CODEBASE_REVIEW.md + codebase_review/   fresh-clone consistency review (2026-06-16, concluded)
├── roadmap/               repo-wide execution queue + solver audit + reorg spec (🔵 ACTIVE — start: roadmap/README.md)
├── test-suite/            test-suite remediation plan from the 2026-07-06 four-slice audit (🔵 — PLAN.md)
├── transition/            implicit→momentum transition trigger (🔵 ACTIVE — see transition/README.md)
│   ├── kappa-3way/       🔵 THE ACTIVE FRONT — the three-way f_κ/f_A/f_mix band-entry
│   │                     calibration, measured fresh (start: report.html)
│   ├── pdv-trigger/       ⚠️ DEMOTED 2026-07-29 to "verify before use" — the history, the
│   │                     physics reasoning and the HPC tooling (start: INDEX.md)
│   ├── cleanroom/         substrate certification (concluded — the "transition is geometric" verdict)
│   ├── pt4/               hypothesis audits H1–H5 + R1 shadow (concluded, feeds pdv-trigger)
│   └── harness/ + PROVENANCE_PROTOCOL.md    shared run-stamping tooling
├── rosette-cf/            Rosette Cf scan, in-container (🔵 plan + harness + param committed; runs pending)
├── phase1a-init/          early-phase (1a) init at sub-GMC scale — compact probe (🔵 — FINDINGS.md)
├── phase1a-stiffness/     is 1a's RK45 segment integrator a latent defect? (🟡 fix landed on branch — PLAN.md)
├── screen/                multi-config scheme screen: 2 refs x N configs, matched-t ledger (🔵 — README.md)
├── cooling/               cooling-table refactor (🟡 partial)
├── performance/           hot-path cost & conditioning (📘 reference + 🟡 open items)
├── shell-solver/          shell ODE migration + float64 overflow fix (🟡 mixed)
├── magic-numbers/         hardcoded-constant audit (🟡 #1, #4 fixed; #2 measured; #3, #5 open)
├── failed-large-clouds/   1b collapse of large clouds (✅ shipped; 1b routing superseded 2026-07-01)
├── misc/                  standalone audits / notes (🟡 mixed)
├── cluster/               on-cluster plotting workflow guide (📘 operational)
├── html-insights/         📖 storyline books + verification ledgers
├── data/                  legacy transition-era harvest CSVs (see data/README.md)
├── to-be-removed/         deletion candidates staged for the maintainer's personal review
└── archive/               🧊 frozen: shipped/superseded workstreams (see archive/README.md)
```

The top-level `scratch/` (repo root) is separate, git-ignored, local-only.

## Active workstreams

- **`roadmap/`** — the repo-wide **execution queue**: every open item sequenced with pass/fail
  gates and execution-tier tags ([`roadmap/PLAN.md`](roadmap/PLAN.md)); seeded by the
  2026-07-06 solver audit ([`solver-audit.md`](roadmap/solver-audit.md)) with the mechanical
  hand-off spec in [`REORG.md`](roadmap/REORG.md).
- **`transition/`** — the umbrella for the transition-trigger program; start at
  [`transition/README.md`](transition/README.md). The live front is **`kappa-3way/`**
  (entry: [`transition/kappa-3way/report.html`](transition/kappa-3way/report.html) — the generated
  source of truth; rules in `kappa-3way/PROVENANCE.md`). Its parent `pdv-trigger/` was **demoted
  2026-07-29** to "could be true, verify before use": still the place for the history, the physics
  reasoning, the literature imprints and the HPC tooling, but no measured number from it is quotable
  until the 294-arm re-run reproduces it.
- **`cooling/`** — [`refactor-audit.md`](cooling/refactor-audit.md): decouple the cooling-table
  loaders from hardcoded SB99/OPIATE/CLOUDY. Two side items shipped; core PR-1–4 pending.
- **`performance/`** — start at
  [`BUBBLE_LUMINOSITY_PERFORMANCE.md`](performance/BUBBLE_LUMINOSITY_PERFORMANCE.md) (the
  consolidated perf/robustness history + **Methodology**). `HOTPATH_PLAN.md` carries the open
  items (§F1-cousin, §F5); `F1_SUMMARY.md` + `F1_REPORT.html` are the F1 reference;
  `BUBBLE_CONDUCTION_STIFFNESS.md` documents the diagnosed stiffness cause (descoped §F3).
- **`shell-solver/`** — [`OVERFLOW_FIX_PLAN.md`](shell-solver/OVERFLOW_FIX_PLAN.md) (🟢 the real
  fix, implemented) and [`MIGRATION_PLAN.md`](shell-solver/MIGRATION_PLAN.md) (🟠 correction:
  its `mxstep` diagnosis was retracted — read OVERFLOW first).
- **`magic-numbers/`** — [`AUDIT.md`](magic-numbers/AUDIT.md) (✅ audit closed 2026-08-06: #1/#3/#4
  fixed & gated, #2 documented-and-pinned, #5's tail owned by the transition workstream),
  [`SWEEP2_PLAN.md`](magic-numbers/SWEEP2_PLAN.md) (round 2 — pre-registered bars + results for
  #2/#3/#5), [`TCLAMP_PLAN.md`](magic-numbers/TCLAMP_PLAN.md) (✅ #1 fixed & gated) and
  [`SWITCHON_BRIEF.md`](magic-numbers/SWITCHON_BRIEF.md) (✅ #2 resolved as document-and-pin).
- **`phase1a-stiffness/`** — [`PLAN.md`](phase1a-stiffness/PLAN.md): phase 1a integrates on
  `RK45` with no step bounds while 1b/1c/2 use LSODA with both; with #2's ramp ablated, an `Eb`
  collapse inside a segment grinds the solver and the between-segment collapse guard never gets
  to fire. 🔵 **Batches 1-2 done, still no `trinity/` edit** — production measured ≥4.3e4× from
  the stall (worst segment 4 steps / 0.021 s), so the LSODA swap is ruled out on economics; the
  grind is **stiffness** at collapsed `Eb` (pinned at 1.6e-6 au, λ ≈ −1e13, ~7 days/segment), and
  `Eb` never reaches 0 so the existing `Eb ≤ 0` guard would miss it. Batch 3 built the in-band
  *positive*, scale-relative energy-floor event (threshold bounded both sides by measurement) and
  it clears P0 — the stalling control now ends in 22 s with the pre-existing `ENERGY_COLLAPSED`
  fate — and Batch 4's screen came back **byte-identical on all five configs**, so it lands at
  Batch 5.
- **`screen/`** — [`README.md`](screen/README.md): the multi-config scheme screen. Two git refs,
  N configs, both arms in separate processes, compared at matched `t`, ledger + pass/fail out.
  Run it before landing a scheme change; the suite's end-to-end tests all use one config.
- **`failed-large-clouds/`** — [`PLAN.md`](failed-large-clouds/PLAN.md): the 1b
  collapse investigation (✅ fix shipped 2026-06-19; the "permanent fate" framing was superseded
  2026-07-01 — 1b collapses now route to momentum). Data manifest: `data/PROVENANCE.md`.
- **`misc/`** — standalone: `backward-compat-audit.md` (🔵 ~95% pending),
  `tinit-sensitivity.md` (🟡 rec #3 open), `TERMINATION_EVENTS.md` (📘 reference),
  `LEAKING_LUMINOSITIES_SKELETON.md` (🟡 D/F/G open).

## References / operational

- **`cluster/PLOTTING_WORKFLOW.md`** — how to visualize runs on the cluster (📘 guide).
- **`html-insights/`** — chaptered HTML storyline books merged from workstream reports, plus
  `verification/` line-by-line verification ledgers (see its README).
- **`CODEBASE_REVIEW.md` + `codebase_review/01…07`** — the 2026-06-16 fresh-clone consistency
  review (52 findings; concluded, kept as reference).
- **`data/`** — legacy transition-era harvest CSVs; new artifacts belong in
  `<workstream>/data/` (see `data/README.md`).

## Archive & staging

- **`archive/`** — 🧊 frozen history (see [`archive/README.md`](archive/README.md)):
  `betadelta/` (β–δ hybr solver program, ✅ shipped incl. the hybr default flip),
  `bubble/` (integrator robustness ⛔ superseded · conduction convergence ✅),
  `n-consistency/` (n ≡ n_H program, ✅ shipped),
  `transition/` (the superseded trigger trio: `TRIGGER_PLAN.md`, `P0.md`, `pshadow-design.md`),
  `restructure-audit.md`, `sb99-refactor-audit.md`.
- **`to-be-removed/`** — deletion candidates staged for the maintainer's personal review
  (see its README for the why-list). Nothing is deleted directly.
