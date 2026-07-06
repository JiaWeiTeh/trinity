# transition/ — implicit→momentum transition trigger (umbrella workstream)

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

**Status (2026-07-06):** 🔵 ACTIVE — the live front is `pdv-trigger/`.

How this workstream evolved, and where to enter:

- **Precursors** (⛔ superseded, moved to `../archive/transition/`): `TRIGGER_PLAN.md`, `P0.md`,
  `pshadow-design.md` — the F0 cooling-balance trigger program. Its premise ("flat configs cool")
  was falsified by the cleanroom result below; nothing shipped.
- **`cleanroom/`** (✅ concluded 2026-06-22) — substrate certification on 6 configs under the hybr
  default. Verdict in `cleanroom/FINDINGS.md`: **no cooling-balance event fires (0/6); the
  transition is geometric (blowout), not thermal.** Kept in place (not archived) because it is
  live evidence the active work builds on.
- **`pt4/`** (✅ concluded) — four hypothesis audits (H1–H5) + the R1 shadow experiment on the
  trigger stall; entry `pt4/README.md`. Feeds evidence into pdv-trigger.
- **`pdv-trigger/`** (🔵 ACTIVE) — the PdV/f_κ mechanism boost + θ calibration program.
  **Start at `pdv-trigger/INDEX.md`.** Before quoting any number, check
  `pdv-trigger/CONTAMINATION.md` (quotability register).

Shared tooling: `PROVENANCE_PROTOCOL.md` — the stamped-run standard (commit + command + param
hash per run; promoted as the project-wide model, see `../CONVENTIONS.md`) — and `harness/`
(stamped-run + harvest scripts). The legacy P0-era harvest CSVs live one level up in
`docs/dev/data/` (see its README).
