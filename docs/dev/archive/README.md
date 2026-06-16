# docs/dev/archive

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
