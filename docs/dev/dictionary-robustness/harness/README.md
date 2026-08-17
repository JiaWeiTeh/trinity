# harness — probe runner for dictionary.py edge cases

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Run from the repo root (no simulation, finishes in seconds):

```bash
python docs/dev/dictionary-robustness/harness/probe_dictionary.py
```

Output: one `PROBE-<id> [F<n>]: <verdict>` line per probe on stdout, mapping row-by-row to the
findings table in `../PLAN.md` §1 (verdicts recorded there from the 2026-08-17 run @ `030b658`);
scratch dirs go to a tempdir and are removed on exit.
