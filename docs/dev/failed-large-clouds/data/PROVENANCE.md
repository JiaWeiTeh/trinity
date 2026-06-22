# Data provenance — `failed-large-clouds/data/`

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time manifest; re-check each row against `git log` and the file
> contents before relying on it.
>
> 🔄 **Living manifest — update on every visit.** Any new artifact dropped in
> `data/` must get a row here (file, what it is, code state, source, command).
> No mystery-provenance files.
>
> 💾 **Persist diagnostics — commit, don't re-run.** These artifacts exist so a
> future session reproduces/compares **without** re-running the hours-long sims.
> The `/tmp` sources below are **ephemeral** (gone on container reclaim); only
> what is committed here survives.

## Why this file exists
For a large follow-on fix (the `transition/` workstream) the worry is
**cross-stage contamination**: this `data/` set was accreted across **four
different code states** this session. None of the raw CSVs self-record their
commit, so provenance lived only in filenames + `PLAN.md` prose. This manifest
makes it auditable.

## Key invariant (why the diagnosis is NOT contaminated)
The shipped fix (`G` volume floor + `F` `Eb<=0` stop + phase-1a coverage) is a
**no-op on the pre-collapse trajectory** — the floor branch is dead while the
shell volume is positive, the `Eb<=0` check only fires at the collapse. And the
later `main` LSODA fix (`60fb3626`) is **numerically inert** (its own commit
proves a bit-identical bubble solve: `np.array_equal`, max abs diff 0.0, same
`nfev`). **Verified 2026-06-19:** a full regeneration on the synced commit
`d919ff77` reproduces the **failing** discriminator values **bit-identically**
(`Eb_growth=1.014`), confirming the diagnostic numbers are stage-independent. The
**healthy** values *grew* (≥×39,300 / ≥×94,900 vs the prior ×13,600 / ×37,900) —
not a code-stage effect but a **run-length** one: the prior committed healthy
runs had been **truncated** at `t≈0.32 Myr`, while the healthy energy phase runs
far longer (`Eb` still climbing at `t≈1 Myr`). The regen uses a fixed `t≤1.0 Myr`
window so the metric is reproducible; the healthy figures are lower bounds. This
truncation defect is exactly what the `transition/` provenance contract prevents.

## Manifest

| file | what it is | code state | source run (ephemeral) |
|---|---|---|---|
| `discriminator.csv` | reservoir-growth + PdV discriminator, 5 configs | **regenerated @ `d919ff77`**, header-stamped, `t≤1.0 Myr` window | one batch `/tmp/disc_frozen_d919ff77/*` |
| `budget_fail_repro.csv` | energy budget per snapshot (figs 1–2) | final fix (no-op pre-collapse) | `/tmp/flc_fix3/fail_repro` |
| `budget_small_1e6.csv` | energy budget per snapshot (figs 1–2) | final fix (no-op pre-collapse) | `/tmp/ver/small_1e6` |
| `verify_extended_fix_all_configs.csv` | final fix verification, all configs | **final fix (HEAD)** | extended-fix batch |
| `verify_noop_and_band.csv` | healthy no-op + band check | **final fix (HEAD)** | no-op batch |
| `fixed_fail_repro_clean_termination.csv` | fail_repro clean stop record | **final fix (HEAD)** | `/tmp/flc_fix*/fail_repro` |
| `fail_pism6_clean_termination_log_excerpt.txt` | pism6 clean stop log | **final fix (HEAD)** | pism6 run |
| `biid_byte_diff.txt` | healthy byte-identical no-op proof | **final fix (HEAD)** | bit-identity check |
| `probe_degeneracy.csv` | sim-free analytic degeneracy probe (S0) | **code-state-independent** (math) | `probe_degeneracy.py` |
| `reverify_V0_main_946e860b.csv` | V0 baseline (crash) | **OLD main `946e860b`** (pre-fix) | V0 baseline run |
| `smoke_V3_fail_repro_trajectory.csv` | V3 geometry-guard-only smoke | **monkeypatch, NOT production** | V3 harness variant |
| `fail_helix_phase1a_still_crashes.txt` | helix still crashing | **round-1 fix** (pre phase-1a coverage) | round-1 helix run |
| `fail_helix_trinity_log.txt` | helix run log | helix run (see file header) | helix run |

Legend: **final fix (HEAD)** = current production behaviour, safe to trust as a
reference. **history** (V0/V3/round-1) = deliberately a *different* code state,
kept to document the investigation — **do not reuse as a baseline.**

## Rule for the `transition/` workstream
Treat everything above as **reference/history, not a baseline.** Regenerate
every new measurement in **one batch from a single known commit**, and stamp
each output with `commit + command + param hash` via
`docs/dev/transition/harness/run_stamped.py` (see
`docs/dev/transition/PROVENANCE_PROTOCOL.md`). Nothing in the new workstream
should be mystery-provenance.
