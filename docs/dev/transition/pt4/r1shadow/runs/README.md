# r1shadow/runs — committed R1-shadow run outputs

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

These `<config>/` directories are the committed run outputs (each `dictionary.jsonl` + sideline
`shadow_R1_1b.csv`) that back the R1-shadow row harvest; `<config>_row.csv` / `<config>_status.txt`
are the per-config harvested summary and status.

Provenance and interpretation live in `../DATA_NOTE.md` (what the shadow collects, the commit, and
the per-config table). The exact regeneration commands (driver, stop_t/timeout per config) are in
that note's "Exact commands" section.
