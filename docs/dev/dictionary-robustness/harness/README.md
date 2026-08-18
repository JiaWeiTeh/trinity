# harness — probe runner for dictionary.py edge cases

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Two scripts, both run from the repo root.

## `probe_dictionary.py` — reproduce every §1 finding

No simulation, finishes in seconds:

```bash
python docs/dev/dictionary-robustness/harness/probe_dictionary.py
```

Output: one `PROBE-<id> [F<n>]: <verdict>` line per probe on stdout, mapping row-by-row to the
findings table in `../PLAN.md` §1 (verdicts recorded there from the 2026-08-17 run @ `030b658`);
scratch dirs go to a tempdir and are removed on exit.

## `scan_field_record.py` — battery-H invariant scan of a real run

Needs a completed run directory (one containing `dictionary.jsonl`). Appends one CSV row per run
and prints a summary; creates the CSV with a provenance header if absent:

```bash
python run.py param/simple_cluster.param          # produces outputs/simple_cluster/
python docs/dev/dictionary-robustness/harness/scan_field_record.py \
    --label simple_cluster --commit "$(git rev-parse --short HEAD)" \
    --csv docs/dev/dictionary-robustness/data/field_scan.csv \
    outputs/simple_cluster
```

`test/test_dictionary_stress_process.py` imports `scan_run_record` from this file by path, so the
committed CSV and the test suite always use the same invariant logic. Results live in
`../data/field_scan.csv`; invariant IDs are defined in `../PLAN.md` §2.
