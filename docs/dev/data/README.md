# docs/dev/data/ — legacy transition-era harvest CSVs

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

**Status (2026-07-06):** 📘 legacy data sink — kept in place because existing docs cite these paths.

These CSVs (`transition_*.csv`, `hunt_*.csv`, `stalling_*.csv`) are the P0/stalling-era harvests
owned by the transition and β–δ workstreams — provenance and interpretation live in
`docs/dev/archive/transition/P0.md` and `docs/dev/archive/betadelta/stalling-energy-phase.md`.

Do not add new artifacts here: new data goes in `<workstream>/data/` with a provenance header
(`docs/dev/CONVENTIONS.md` §Provenance).
