# TRINITY CLI — preview / cheat-sheet (target design)

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

**Status: TARGET design — NOT yet implemented.** This is the spec the
`feature/helix-implementations` work builds toward (design in `PLAN.md`, this
folder). When a piece ships, port the relevant section into the built docs
(`docs/source/running.rst`) and update this file. Date: 2026-06-23.

---

## 1. Complete command surface

```
python run.py <file.param> --local        # run now, on this machine (single OR sweep — auto)
python run.py <file.param> --submit       # HPC: emit + submit + auto-collect, one command
python run.py <file.param> --emit  DIR    # just write the job bundle, don't submit
python run.py --collect DIR               # aggregate a finished bundle into a report
python run.py --resume  DIR               # re-feed only the not-yet-submitted / failed chunks
```

Exactly one mode is **required** — bare `python run.py x.param` errors with a hint (so a
2640-run sweep can't launch on a laptop by accident).

**Optional overrides** (all have profile/auto defaults — rarely typed):

| flag | applies to | meaning | default |
|---|---|---|---|
| `--workers N` | `--local` | local parallel pool size | auto from cores / SLURM alloc |
| `--throttle N` | `--submit`/`--emit` | `%N` concurrent array tasks | profile `[submit] throttle` |
| `--chunk N` | `--submit`/`--emit` | array tasks per submission | profile `[submit] chunk` |
| `--jobs-dir DIR` | `--submit`/`--emit` | where the bundle goes | `<outbase>/_jobs/<stem>_<ts>/` |
| `--foreground` | `--submit` | watch the feeder instead of backgrounding | background |
| `--no-auto-collect` | `--submit` | skip the dependency collect job | off |
| `--dry-run` | any | show what would happen, run nothing | off |
| `--yes` | `--local` sweep | skip the confirmation prompt | off |

---

## 2. One-time setup (set once, never touch again)

```bash
# ~/.bashrc  (or your cluster module env)
export TRINITY_OUTPUT_DIR=/gpfs/bwfor/work/ws/hd_cq295-trinity
```

```ini
# ~/.config/trinity/cluster.ini   — the "how/where to run" settings, separate from the science
[sbatch]
partition = cpu-single
time      = 02:00:00
mem       = 2G
export    = NONE

[submit]
throttle  = 150        ; %N concurrent tasks  (the old "--workers 150")
chunk     = 880        ; tasks per submission (offsets auto-computed — no more 0/880/1760)

[env]
prologue_file = ~/.config/trinity/helix_prologue.sh
```

```bash
# ~/.config/trinity/helix_prologue.sh   — literally the existing 3 lines
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trinity
```

And the `.param` loses its hardcoded path — it becomes portable (laptop ↔ Helix):

```diff
- path2output    /gpfs/bwfor/work/ws/hd_cq295-trinity/paperII_grid_sweep_new_trigger/
+ path2output    paperII_grid_sweep_new_trigger
```

---

## 3. Simulated sessions

### A) Single run on a laptop
```console
$ python run.py param/simple_cluster.param --local
  TRINITY ──────────────────────────────────────────
  Output: /home/you/trinity-out/simple_cluster        # TRINITY_OUTPUT_DIR + model name
  Running single simulation… done (t_end=8.3 Myr, R2=41.2 pc)
```
```console
$ python run.py param/simple_cluster.param            # forgot the mode
error: choose how to run: --local (here) or --submit (HPC job);
       --emit DIR to just write the job bundle.
```

### B) Local sweep on a workstation
```console
$ python run.py param/sweep_example.param --local --workers 8
  Sweep: 24 combinations | workers: 8 | output: /home/you/trinity-out/sweep_example
  Run 24 simulations with 8 workers? [y/N]: y
  [██████████████████████] 24/24  ✓ 23  ✗ 1
  Report: /home/you/trinity-out/sweep_example/sweep_report.txt
```

### C) The Helix campaign — the whole thing in one command
`paperII_grid_sweep.param` (nCore × PISM × nISM = 24 combos):
```console
$ python run.py param/paperII_grid_sweep.param --submit
  TRINITY ──────────────────────────────────────────
  Sweep:    24 combinations
  Output:   /gpfs/bwfor/work/ws/hd_cq295-trinity/paperII_grid_sweep_new_trigger
  Profile:  ~/.config/trinity/cluster.ini  (partition=cpu-single, time=02:00:00, mem=2G)
  Bundle:   …/paperII_grid_sweep_new_trigger/_jobs/paperII_grid_sweep_20260623_142501/
  job-name: trinity_paperII_grid_sweep      (auto)   logs: <bundle>/logs/   (auto)

  24 ≤ chunk 880 → single submission, throttle %150
    sbatch --array=1-24%150 submit_sweep.sbatch        → job 4815201
  auto-collect (runs when the array finishes):
    sbatch --dependency=afterany:4815201 …             → job 4815202

  ✓ Submitted. Nothing else to do.
    progress:  squeue -j 4815201
    report →   …/paperII_grid_sweep_new_trigger/sweep_report.txt   (written automatically)
```
No sbatch edit, no submit loop, no tmux, no manual collect.

### C′) A grid bigger than the QOS cap (the 2640-run case)
```console
$ python run.py param/paperII_grid_full.param --submit
  Sweep:  2640 combinations
  2640 > chunk 880 → 3 chunks, offsets [0, 880, 1760], throttle %150
  Feeder backgrounded (PID 31822) → log: <bundle>/submit.log
  ✓ You can disconnect. Report appears at <base>/sweep_report.txt when all chunks finish.

# tail -f <bundle>/submit.log
  chunk 1/3  offset 0     → job 4815201   (running %150)
  chunk 2/3  offset 880   → QOSMaxSubmitJobPerUserLimit; waiting 5m… ×3
  chunk 2/3  offset 880   → job 4815640
  chunk 3/3  offset 1760  → QOSMaxSubmitJobPerUserLimit; waiting 5m… ×7
  chunk 3/3  offset 1760  → job 4816012
  auto-collect dep on 4816012 → job 4816013
  feeder done.
```
The offsets, the `%150`, the retry-on-full, and the collect are all handled. (The *waiting*
itself is unavoidable — the cap is on queued jobs — but the loop is no longer hand-written or
babysat.)

### D) Inspect-first, or resume
```console
$ python run.py param/paperII_grid_full.param --emit ~/jobs_paperII   # write, don't submit
  Bundle written to ~/jobs_paperII  (review submit_sweep.sbatch, then --submit or sbatch yourself)

$ python run.py --resume ~/jobs_paperII/_jobs/paperII_grid_full_2026…  # after a crash/disconnect
  3 of 3 chunks already submitted; nothing to resume.   # or: re-feeds the missing ones
```

### E) Reading results (off the env base)
```console
$ python run.py --collect ~/jobs_paperII/_jobs/paperII_grid_full_2026…   # manual collect if --no-auto-collect
$ python -m trinity._output.show_run paperII_grid_sweep_new_trigger/<run_name>
```

---

## 4. No-scheduler fallback
On a machine without `sbatch` (this cloud workspace, a login node without the scheduler),
`--submit` detects the absence and falls back to `--emit` with a printed manual submit line —
it does not crash.

---

## 5. Before → after (the paperII campaign)

| step | today | target |
|---|---|---|
| output path | hardcode `/gpfs/...` in `.param` | `TRINITY_OUTPUT_DIR` + relative name (portable) |
| sbatch | hand-rewrite every run (env, `--export=NONE`, partition, time, mem, job-name, logs) | profile + auto-derived, set once |
| offsets | hand-compute `0 880 1760` | auto from one `chunk` number |
| throttle | bake `%150` into the loop | profile `throttle` |
| submit | hand-written `submit_loop.sh` in tmux, QOS retry | built-in feeder, backgrounded |
| collect | manual `--collect-report` | `afterany` dependency, automatic |
| **commands typed** | edit + emit + edit + write loop + tmux + collect | **`python run.py x.param --submit`** |
