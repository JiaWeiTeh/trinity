# REPRODUCE — claim → command → artifact

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

**Status (2026-07-30):** ✅ all rows runnable — the 294 arms ran, reduced and landed; row 10's
`make_bench7_analysis.py` is written and is the three-way deliverable.

---

All commands run **from the repo root**. Cost: 🟢 seconds (reads committed CSVs) · 🟡 minutes ·
🔴 HPC hours.

| # | claim / artifact | command | produces | cost | state |
|---|---|---|---|---|---|
| 1 | **The source-of-truth page** — regenerated from the committed artifacts; prints its own build time and a live freshness roll-up | `python docs/dev/transition/kappa-3way/make_report.py` | `kappa-3way/report.html` | 🟢 | ✅ |
| 2 | **The 174 bench7 params**, self-gating (G1: GMC plausibility on every arm incl. `densBE`, exact L21b mapping ≤2%, end-to-end `read_param` load-check, count/uniqueness) | `python docs/dev/transition/pdv-trigger/runs/make_kappa_reopen_params.py` | `pdv-trigger/runs/params/bench7/*.param` (174) | 🟢 | ✅ 4/4 |
| 3 | **The params match their builder** — byte-identical regeneration, per-phase counts, single-knob, K3 pairs identical bar their names, each bench arm on its bench5 `__none` cloud | `pytest test/test_bench7_params.py` | 182 passing cases | 🟢 | ✅ |
| 4 | **Gate G0** — Θ₀ and the `§18` band-entry table recompute from the trajectories, at half-last-digit tolerances. Auto-prefers `bench5r`/`bench6r` once they land, turning it into the old-vs-new reproduction gate | `python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py` | `pdv-trigger/data/bench7_gate_g0.csv` (23 rows: 11 G0 + 12 P1) | 🟢 | ✅ 11/11 pre-run |
| 5 | **The freshness receipt** — every committed CSV classified FRESH / OLD / UNSTAMPED against the cutoff, from its own stamp | `python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py [YYYY-MM-DD]` | `pdv-trigger/data/freshness_audit.csv` | 🟢 | ✅ |
| 6 | **Θ₀ + the f_A ladder ≤16, re-measured** | `./sync_bench.sh bench5r submit` → `reduce` → `down` | `runs/data/bench5r_{summary,hashes}.csv` + `bench5r_traj/` (60) | 🔴 | ✅ ran 2026-07-30 |
| 7 | **f_A 24–128 + the f_mix head-to-head, re-measured** | `./sync_bench.sh bench6r submit` → `reduce` → `down` | `runs/data/bench6r_{summary,hashes}.csv` + `bench6r_traj/` (60) | 🔴 | ✅ ran 2026-07-30 |
| 8 | **K1–K4 — the f_κ campaign** (54 + 20 + 66 + 10 + 24) | `./sync_bench.sh bench7 submit` → `reduce` → `down` | `runs/data/bench7_{summary,hashes}.csv` + `bench7_traj/` (174) | 🔴 | ✅ ran 2026-07-30 |
| 9 | **The re-derivation from fresh data** — each builder prints and records a `SOURCES READ:` line | `python .../make_bench5_analysis.py`; `.../make_bench6_analysis.py`; `.../make_bench7_gate_g0.py` | regenerated `bench5_analysis.csv`, `bench6_analysis.csv`, `bench7_gate_g0.csv` | 🟢 | ✅ |
| 10 | **The three-way band-entry table** — the deliverable | `python .../data/make_bench7_analysis.py` | `pdv-trigger/data/bench7_analysis.csv` (234 rows: ARMS/ENTRY/EXPONENT/FIREMAP/DETERM/G6/BACKREACT) + `bench7_entry.png` | 🟢 | ✅ **the deliverable** |

## Rebuild everything runnable today, in one block

```bash
python docs/dev/transition/pdv-trigger/runs/make_kappa_reopen_params.py
python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py
python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py
python docs/dev/transition/kappa-3way/make_report.py
pytest test/test_bench7_params.py
```

## Inputs that are not reproduced here, on purpose

Lancaster 2021b Table 1, the band [0.90, 0.99], λδv ≈ 3, and El-Badry+2019 Eq 47 are **published
literature**, `[V]`-verified 2026-07-12 in `pdv-trigger/LANCASTER_REFERENCE.md §7b` and
`ELBADRY_REFERENCE.md`. A re-run cannot refresh a paper; these are inherited deliberately and are the
only pre-cutoff inputs this workstream quotes without a VERIFY tag. See `PROVENANCE.md` §4.

## Cost note before spending the cluster

294 arms at `--time=1:30:00`. The longest compliant bench5 arm was 64 min under 3-worker contention
(`pdv-trigger/data/bench5_durations.csv`, ⚠️ VERIFY — a pre-cutoff number, used here only for sizing).
Expensive corner: **diffuse × high dose** (bench1 at f_κ ≥ 24), where f_κ enters the structure ODE. A
wall-kill is a recorded **G3 non-compliance**, never a silent drop — resubmit those ids with
`--time=3:00:00` and re-reduce.
