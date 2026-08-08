# Weak-winds — batch runbook

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

**Status (2026-08-08):** 🔵 actionable — batches 0–6 defined, param files and gate script
committed and exercised against the smoke output. No science batch has been run yet.

The science design and its justification live in `PLAN.md`; this file is only the
execution sequence. **Run one batch, check its gate, then decide** — do not fire
the whole ladder blind. All commands run from the repo root.

## The batches at a glance

| # | What | Runs | Command file | Output dir |
|---|---|---|---|---|
| 0 | H0 plumbing gate (knob at default is inert) | 2 short | `batches/batch0_h0_*.param` | `h0_explicit/`, `h0_untouched/` |
| 1 | Control rung, c = 1.0 | 3 | `batches/batch1_c1p0.param` | `c1p0/` |
| 2 | Rung c = 0.3 | 3 | `batches/batch2_c0p3.param` | `c0p3/` |
| 3 | Rung c = 0.1 | 3 | `batches/batch3_c0p1.param` | `c0p1/` |
| 4 | Rung c = 0.03 | 3 | `batches/batch4_c0p03.param` | `c0p03/` |
| 5 | Rung c = 0.01 | 3 | `batches/batch5_c0p01.param` | `c0p01/` |
| 6 | Harvest → CSV → figures → write-up | 0 | — | `data/`, `figures/` |

All batch dirs sit under `outputs/weak_winds_study/`. Each batch has its **own**
subdirectory on purpose: a fixed (non-swept) knob does not appear in the run-folder
name, so batches sharing a directory would silently overwrite each other.

Every batch runs the same three clouds (`PLAN.md` §4): `1e5_sfe030_n1e5` (baseline),
`1e7_sfe050_n1e2` (f1edge_lowdens), `1e7_sfe001_n1e6` (f1edge_hidens).

### Cost

Honest state: **unmeasured at the study's `stop_t 15`.** The only datapoints are the
smoke pair on the baseline cloud at `stop_t 1.5` — 19 min (control, reached the
horizon) and 10 min (c = 0.1, collapsed at 0.28 Myr) on this container. The full
15 Myr horizon and both 1e7 clouds are untested and will be substantially slower.
**Treat batch 1 as the cost calibration**: its `wall` column tells you what the
remaining four batches will cost before you commit a cluster allocation to them.

---

## Batch 0 — H0 plumbing gate (do this first)

Cost: **~15 min per arm** (measured 2026-08-08, 4-core container: 15.2 and
14.6 min, both reaching the 0.5 Myr horizon) — the baseline cloud spends most of
it in the phase-1b implicit integrator. The two arms are independent; run them
concurrently if you have the cores.

**Result on this tree (2026-08-08): PASS, `max |dR2/R2| = 0.000e+00`** over the
full 0.5 Myr — the knob at its default is exactly inert. See `FINDINGS.md` §H0.

Proves that naming the knob at its schema default changes nothing. If this fails,
every later trend is measuring plumbing, not physics.

```bash
python run.py docs/dev/weak-winds/harness/batches/batch0_h0_plumbing.param --yes
python run.py docs/dev/weak-winds/harness/batches/batch0_h0_untouched.param --yes

python docs/dev/weak-winds/harness/check_batch.py --compare \
    outputs/weak_winds_h0/explicit outputs/weak_winds_h0/untouched
```

These two write to `outputs/weak_winds_h0/`, deliberately outside the study
directory: they are short `stop_t 0.5` plumbing arms, and batch 6 harvests
`outputs/weak_winds_study/` whole. (A single run writes straight into its
`path2output` — no per-run subfolder, unlike a sweep.)

**Gate:** `max |dR2/R2| < 1e-9` at matched t (the script's default tolerance;
exit 0 = pass). Not byte equality — see `PLAN.md` §4 on loader ULP jitter.

**If it fails:** stop. Something in the parameter path treats an explicitly-set
default differently from an unset one. Record the deviation and investigate before
running any science batch.

---

## Batch 1 — control rung, c = 1.0 (the reference + the cost calibration)

```bash
python run.py docs/dev/weak-winds/harness/batches/batch1_c1p0.param --workers 3 --yes
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c1p0
```

**Gate:** exit 0 — all three runs completed with a recorded fate and finite
trajectories. Read the `wall` column and decide the budget for batches 2–5.

**If a cloud fails here:** it fails at *full* wind strength, so the failure is
pre-existing and unrelated to this study. Record it, drop that cloud from the
remaining batches, and note the reduced scope in `FINDINGS.md`.

---

## Batches 2–5 — the descent (one rung at a time)

```bash
# Batch 2
python run.py docs/dev/weak-winds/harness/batches/batch2_c0p3.param --workers 3 --yes
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p3

# Batch 3
python run.py docs/dev/weak-winds/harness/batches/batch3_c0p1.param --workers 3 --yes
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p1

# Batch 4
python run.py docs/dev/weak-winds/harness/batches/batch4_c0p03.param --workers 3 --yes
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p03

# Batch 5
python run.py docs/dev/weak-winds/harness/batches/batch5_c0p01.param --workers 3 --yes
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p01
```

**Gate per batch:** `check_batch.py` exits 0.

**Descent rules:**

1. **A crashed run is data, not a discard.** Weak-Lw pushes the phase-1b stiff
   integrator toward untested regimes; the lowest runnable c per cloud is a
   result (`PLAN.md` §8). Record the config, the traceback, and the last good
   snapshot in `FINDINGS.md`.
2. **A failure stops the descent for that cloud only** — the other two continue.
   To drop a cloud, delete its entry from the `tuple(...)` line in the remaining
   batch files and note it in `FINDINGS.md`.
3. **A collapse or dissolution is a normal fate, not a failure.** The gate passes
   because the run terminated for a recorded physical reason; that is the science.
4. **Don't skip a rung.** The trend needs the intermediate points; a gap makes a
   power-law fit unfalsifiable.

### On a cluster instead

Any batch can go out as a SLURM job array rather than local workers:

```bash
python run.py docs/dev/weak-winds/harness/batches/batch2_c0p3.param \
    --emit-jobs jobs/weak_winds_c0p3
sbatch jobs/weak_winds_c0p3/submit_sweep.sbatch
# cap concurrency by adding --workers K to the emit command (appends %K to the
# array), or edit the generated sbatch directly.

# after the array finishes:
python run.py --collect-report jobs/weak_winds_c0p3
python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p3
```

The emitted bundle also carries `manifest.json` and `runs.tsv` (what each array
task runs) and a `logs/` directory — keep them with the batch if you need to
explain a failure later.

---

## Batch 6 — harvest, figures, write-up (no simulations)

```bash
python docs/dev/weak-winds/harness/harvest.py outputs/weak_winds_study \
    --out docs/dev/weak-winds/data/ladder.csv

python docs/dev/weak-winds/harness/make_figures.py docs/dev/weak-winds/data/ladder.csv
```

`harvest.py` recurses, so one call collects every batch that has run so far; the
`run` column keeps the batch subdirectory as a prefix, and `FB_thermCoeffWind`
comes from each run's own `metadata.json`. Partial ladders harvest fine — re-run
this after each batch if you want to watch the trend build.

Then update `FINDINGS.md` against H1–H4 (`PLAN.md` §3) and commit the CSV +
figures. **Commit the artifacts, not the `outputs/` tree** — `outputs/` is
untracked scratch (root `CLAUDE.md`), and the container is ephemeral.

### Comparing any two runs

```bash
python docs/dev/weak-winds/harness/check_batch.py --compare RUN_A RUN_B --tol 1e-9
```

Reports `max |dR2/R2|` over the overlapping time range only — runs truncate at
different `t`, and comparing past the overlap is meaningless.

## Resuming after an interruption

Batches are independent, and re-running one is safe: on a run's first flush
TRINITY deletes any existing `dictionary.jsonl` and `metadata.json` in that run's
output directory (`trinity/_input/dictionary.py` §flush, verified 2026-08-08), so
a re-run replaces its own snapshots rather than appending to them. Other files in
the folder (`trinity.log`, the `.param` copy, `shadow_R1_1b.csv`) are not part of
that reset — delete the run folder if you want a pristine one.

To resume, run `check_batch.py` on each existing batch dir to see what completed,
then continue from the first rung that is missing or failing.
