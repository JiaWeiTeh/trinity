# screen data — ledger manifest

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

One ledger CSV per screen run (`screen.py --out`), schema matching
`docs/dev/phase1a-init/data/gate_results.csv`:

```
gate,config,quantity,reference,reference_source,measured,rel_diff,verdict
```

Each file's first line is a provenance header recording the two refs, the
`stop_t`, and the bar the verdicts were judged against.

**Empty until a screen is run in anger.** The harness smoke test writes to a
scratch directory instead, because a run with identical arms on both sides
confirms the plumbing and is not evidence about any change.

| file | before → after | configs | bar |
|---|---|---|---|
| _(none yet)_ | | | |
