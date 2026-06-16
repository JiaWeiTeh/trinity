# docs/dev/archive

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Completed or superseded analysis docs, kept for historical reference only.

The plans in here **have fully shipped** — they read as forward-looking plans
but describe work that is already done, so their paths, line numbers, and
"what to do next" framing are obsolete against current code. Each file keeps
its caution banner; treat everything here as a historical record, not a guide
to the current codebase. Verify against source before relying on anything.

- `restructure-audit.md` — `src/→trinity` rename, `_modified` drop, and the
  plotting/`scratch` split. All shipped (the `scratch/` tree was later removed,
  not kept tracked as the plan proposed).
- `sb99-refactor-audit.md` — SB99 → generic SPS refactor (all four PRs).
  Shipped, then further restructured (ParamSpec registry/resolver,
  auto-generated `default.param`, `SB99f → sps_f`, `read_SB99.py → sps/read_sps.py`).
