# CLI & path-portability rationalization — plan

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

**Date:** 2026-06-23 · **Branch:** `feature/helix-implementations` · **Status:** plan only, no code yet (per user)

**See also:** [`CLI_PREVIEW.md`](CLI_PREVIEW.md) — the target command surface + simulated user
sessions (the cheat-sheet this plan builds toward).

---

## 0. Why

The user runs TRINITY from an ephemeral cloud workspace and on bwForCluster Helix.
Three friction points surfaced, all about *the seam between the code and where it runs*:

1. **Outputs land in the wrong place / get lost.** The reader already honors a
   `TRINITY_OUTPUT_DIR` env var (`trinity/_output/trinity_reader.py:1291`), but the
   **writer ignores it** — `_resolve_path2output` always uses `<cwd>/outputs`
   (`trinity/_input/registry.py:197-198`). So there is no single switch to redirect
   output to a persistent workspace mount; users hardcode `path2output` into `.param`
   files (which then leak absolute paths when committed).
2. **A machine path leaks into committed artifacts.** `sps_path` resolves to an
   absolute path off `_REPO_ROOT` (`registry.py:254-256`) and is the **one** path field
   missing `metadata_exclude=True` (`registry.py:353`; contrast `path2output` :294,
   both cooling paths :351-352). So `final_state.sps_path = "/home/user/trinity/lib/..."`
   ends up in every committed `metadata.json`. User preference: keep the field but make
   it **relative**, not exclude it.
3. **The `run.py` command surface has accreted cruft.** One overloaded entry point with
   auto-detection, a flag (`--workers`) that means two different things, undocumented
   tools, a dangling reference, and an overloaded "cluster" name.
4. **The HPC submission path is almost entirely manual** (from the user's real paperII run):
   the emitted sbatch *can't run as-is* — it has no env activation (`module`/`conda`) and no
   `--export=NONE`, so the user hand-rewrites it every time (partition, walltime, mem,
   job-name, log dir). Then a hand-written `submit_loop.sh` with **manually computed** offsets
   (`OFFSETS="0 880 1760"`), a baked-in `%150` throttle, and a `QOSMaxSubmitJobPerUserLimit`
   retry loop babysat in tmux. Then a manual `--collect-report`. Nothing is parameterized;
   everything is hardcoded and re-typed per campaign.

This doc plans these as **separate reviewable commits**, low-risk first.

---

## A. Honor `TRINITY_OUTPUT_DIR` on the write side  *(small, safe)*

**Problem.** Read side honors the env var; write side does not. No single switch to
redirect output to a persistent mount — so users hardcode an absolute `path2output` into the
`.param` (e.g. the real paperII grid pins
`/gpfs/bwfor/work/ws/hd_cq295-trinity/paperII_grid_sweep_new_trigger/`), which is
non-portable (breaks on laptop), leaks a machine path, and must be hand-edited per machine.

**Fix — three-way `path2output` resolution.** In the two resolvers
(`_resolve_path2output` `registry.py:193-202`; `resolve_base_output_dir` `run.py:137-148`):

| `path2output` value | resolves to |
|---|---|
| `def_dir` (default) | `$TRINITY_OUTPUT_DIR/<model>` if env set, else `<cwd>/outputs/<model>` |
| **relative** (e.g. `paperII_grid_sweep_new_trigger`) | `$TRINITY_OUTPUT_DIR/<that>` if env set, else `<cwd>/<that>` |
| **absolute** (e.g. `/gpfs/...`) | taken as-is (escape hatch, unchanged) |

So the paperII param drops the `/gpfs/...` line to just
`path2output  paperII_grid_sweep_new_trigger`, and the user sets
`export TRINITY_OUTPUT_DIR=/gpfs/bwfor/work/ws/hd_cq295-trinity` once. The **same .param now
runs on the laptop and on Helix** with no edit; nothing absolute is committed. Behavior is
unchanged when the var is unset *and* the path is `def_dir`/absolute.

**Test.** `test/` cases (monkeypatch env): `def_dir` and a relative path both resolve under
`TRINITY_OUTPUT_DIR` when set; fall back to `<cwd>` when unset; an absolute path is untouched
in all cases.

**Risk.** Low/local. One branch in two functions + tests. (The relative branch is the only
behavior *change* — today a non-`def_dir`, non-absolute value is rare; verify no committed
`.param` relies on a relative `path2output` meaning "cwd-relative".)

---

## B. Path portability — no machine paths in committed artifacts  *(small, safe)*

**What leaks today.** Only `sps_path` (see §0.2). `path2output` and both cooling paths are
already excluded from `final_state`; we are **not** re-adding them.

**The mechanism (the "way" the user asked for).** All bundled assets resolve under
`_REPO_ROOT` (`registry.py:66`). A small helper makes a path portable for serialization:

```python
def portable_path(p: str) -> str:
    """Relative to the repo root if p lives under it (-> 'lib/default/sps/...'),
    else relative to cwd if under it, else p unchanged (a genuinely external
    user path — their call). POSIX separators for cross-platform stability."""
```

Apply it in the `final_state` builder (`trinity/_output/simulation_end.py`, the
`block[key] = val` loop ~:281-306) to path-valued string fields, instead of skipping
`sps_path`. Result: `"sps_path": "lib/default/sps/starburst99/1e6cluster_default.csv"` —
obvious *which* table, zero machine prefix. A user-supplied SPS file outside the repo stays
absolute (we can't relativize it without guessing an anchor, and it's the user's own path).

**One-time scrub of existing committed artifacts.**
- 8 committed `metadata.json` files: replace the `/home/user/trinity/` prefix on `sps_path`
  with the repo-relative form. Mechanical.
- `test/data/residual_resample_fixture.json`: 4 absolute paths (`path2output`,
  `path_cooling_CIE`, `path_cooling_nonCIE`, `sps_path`). **Safe to rewrite** — the consuming
  tests (`test_residual_resample.py:55-60` `_PATH_OVERRIDE_SKIP`, mirrored in
  `test_dR2min_magic_number.py:68-71`) deliberately discard these and re-resolve worktree-local.
  So this is cosmetic/privacy only; cannot affect test outcomes. Make them repo-relative.
- (Optional, lower priority) dev harness `.sh` under `docs/dev/**` that `cd /home/user/trinity`
  — switch to `cd "$(git rev-parse --show-toplevel)"`. Not output artifacts; defer unless asked.

**Test.** Assert a freshly written `metadata.json` (small run, or unit-level on the
`final_state` dict) contains no string starting with the absolute repo root.

**Gate (this is an I/O format touch → outward-facing).** Confirm no reader consumes an
*absolute* `sps_path` from metadata (it's informational; the live run reads `sps_path` from
the `.param`, not from metadata). Grep readers in `trinity/_output/` before landing.

**Risk.** Low, but it changes a committed-artifact format → verify round-trip + no consumer.

---

## C. Command rationalization  *(outward-facing → gated)*

### C.0 Current state (verified `run.py` @ this commit)
Single entry point, auto-detects single vs sweep from `.param` content
(`is_sweep_param_file`, `run.py:81-115`). Flags: `path2param` (positional, optional),
`--workers/-w`, `--dry-run/-n`, `--yes/-y`, `--verbose/-v`, and a mutually-exclusive
`--emit-jobs DIR` / `--collect-report DIR` group (`run.py:817-829`).

**Cruft inventory:**
- **`--workers` is overloaded.** In sweep mode it's the local pool size (`run.py:449`).
  In `--emit-jobs` mode the *same* flag is reused as the SLURM array throttle
  (`concurrency=args.workers`, `run.py:871`). Two unrelated meanings on one flag.
  → **resolved in §C.4** (kept as one "max concurrency" flag — coherent once the mode is explicit).
- **`"cluster"` namespace clash.** `trinity/_functions/cluster.py` is **HPC CPU-allocation
  detection** (`detect_allocated_cpus`, `get_optimal_workers`), and `tools/cluster/` is HPC
  *plotting env* (matplotlibrc, plot_env.sh) — yet everywhere else in TRINITY "cluster" means
  the **stellar** cluster (`mCluster`, `simple_cluster.param`, SPS reference cluster). Reading
  `from trinity._functions.cluster import ...` looks like physics; it's `nproc`. Rename the
  module (e.g. `cpu_allocation.py` / `hpc.py`) to disambiguate; `tools/cluster/` likewise.
- **Undocumented / dangling tools.** `tools/plot_sweep_heatmap.py` exists but isn't mentioned
  in README/running.rst (the "phantom `trinity-plot`" — there is no such alias). `tools/cluster/`
  and `docs/dev/cluster/PLOTTING_WORKFLOW.md` reference the plotting env but it's not in the
  documented workflow. `param/paperII_grid_sweep.param` (+ `_test`) are real sweep examples not
  surfaced in docs. Verify each reference resolves; document or remove.

### C.1 Decided design (2026-06-23) — one spine, one *required* execution mode
The real human choice is *where it runs* (local now vs HPC job), **not** "run vs sweep" —
the `.param`'s list/tuple syntax already determines single vs sweep (auto-detected,
`run.py:81-115`), so the user never picks that. Surface:
```
python run.py x.param --local      # run now, here (single or sweep, auto-detected)
python run.py x.param --submit     # emit bundle + sbatch array + dependent auto-collect
python run.py x.param --emit DIR   # write the bundle only, don't submit (inspect first)
python run.py --collect DIR        # standalone collect (also what --submit chains)
```
- Exactly one of `--local` / `--submit` / `--emit DIR` is **required** with a param
  (argparse mutually-exclusive group, `required=True`). Bare `python run.py x.param` errors:
  *"specify how to run: --local (here) or --submit (HPC job); --emit DIR to write only."*
  **User decision: explicit, no default** — also blocks the accidental-big-sweep-on-laptop foot-gun.
- `--collect DIR` stays a separate no-param action (rename of today's `--collect-report`).

### C.2 Design north star — what other codes do (and where TRINITY lands)
The recurring HPC best practice: **separate "what to compute" (the `.param`, scientific) from
"how/where to run it" (a per-user *site profile*, scheduler).** The user's pain is that
TRINITY has no place for the second, so everything leaks into a hand-edited sbatch + a
hand-written submit loop + hardcoded numbers.

| concern | what they hand-roll today | reference tool | TRINITY plan |
|---|---|---|---|
| scheduler settings (partition, time, mem, account) | edit each sbatch | Snakemake *profile* `config.yaml`; Nextflow `process{}` | **site profile** (§C.3) |
| env activation (`module`/`conda`) | edit each sbatch | Nextflow `beforeScript`; submitit `setup` | profile `prologue` (§C.3) |
| concurrency throttle (`%150`) | bake into the loop | Nextflow `executor.queueSize`; submitit `slurm_array_parallelism` | profile `throttle`, per-run `--throttle` (§C.4) |
| over-cap chunking + offsets (`0 880 1760`) | hand-compute, hand-loop | Nextflow/Snakemake feed the queue for you | auto from one `chunk` number (§C.4) |
| retry on QOS-full | bash `case … sleep 300` loop in tmux | client-side scheduler (Nextflow/Snakemake daemon) | built-in feeder, backgrounded (§C.4) |
| gather/report after | manual `--collect-report` | dependency "reduce" job | `afterany` auto-collect (§C.4) |
| output location | hardcode `/gpfs/...` | `--directory`/work-dir | `TRINITY_OUTPUT_DIR` + relative path (§A) |

Net: `python run.py x.param --submit` becomes the one command; the profile + env are set once.

### C.3 The site profile — set the cluster bits ONCE (the linchpin)
Discovery: `$TRINITY_CLUSTER_PROFILE`, else `~/.config/trinity/cluster.ini` (XDG). Stdlib
`configparser` — **no new dep** (PyYAML is importable but undeclared; py floor is 3.9 so no
`tomllib`). Everything optional, safe defaults; an absent profile reproduces today's generic
template *plus* the two hooks it was missing (`--export=NONE`, prologue).

```ini
[sbatch]
partition = cpu-single
time      = 02:00:00
mem       = 2G
export    = NONE
# account = ...        ; Helix usually doesn't need it -> omit (also reads $SBATCH_ACCOUNT)

[submit]
throttle  = 150        ; %N concurrent array tasks; omit = no cap ("worker 150" lives here, not per-run)
chunk     = 880        ; max array tasks per submission; offsets auto-computed. "auto" = detect cap

[env]
prologue_file = ~/.config/trinity/helix_prologue.sh   ; verbatim shell before `python run.py`
# or inline `prologue = ...` (configparser multi-line)
```

`helix_prologue.sh` is literally the user's existing three lines
(`module load devel/miniforge` / `conda activate trinity`). This one file replaces the manual
sbatch rewrite **every run**.

**Auto-derived per run** (fixes "i need a keyword to set --output/--jobname" — the answer is
*no keyword*, derive it): `--job-name = trinity_<sweep-file-stem>`,
`--output = <bundle>/logs/%A_%a.out`. Array size + `runs.tsv`/`run.py` paths come from the
bundle as today. From the profile: partition/time/mem/account/export + prologue.

### C.4 `--submit` orchestration — chunking + feeder + auto-collect, zero hardcoding
`python run.py paperII_grid_sweep.param --submit` does:
1. **Emit** the bundle into an auto dir `<base>/_jobs/<stem>_<timestamp>/` (or `--jobs-dir`);
   output base from §A env — nothing hardcoded.
2. **Chunk automatically:** total `N`, `chunk=C` → `n_chunks=ceil(N/C)`, offsets `[0,C,2C,…]`
   computed for you. No more hand-written `OFFSETS="0 880 1760"`, no manual `1760`.
3. **Submit with throttle:** per chunk `sbatch --export=OFFSET=<off> --array=1-<size>%<throttle> <sbatch>`,
   throttle from the profile (no per-run `--workers 150`); parse the returned job id.
4. **Feed past the QOS cap:** on `QOSMaxSubmitJobPerUserLimit`, wait + retry — exactly the
   hand-written `submit_loop.sh`, now built-in and parameter-free. *(Honest ceiling, ponytail:
   the cap is on pending+running jobs, so jobs must LEAVE before more enter — a feeder is
   intrinsic; we remove the hand-rolling, not the waiting.)* **Disconnect-resilient:** default
   `setsid`/`nohup` with `<bundle>/submit.log` + printed PID (or `--foreground` to watch) — no
   tmux ritual.
5. **Auto-collect:** after the last chunk, `sbatch --dependency=afterany:<last_id> --wrap
   "python run.py --collect <bundle>"`.
6. Print every array id + the collect id + the `--resume <bundle>` / progress command.

Escape hatches: `--emit DIR` (write only), `--chunk N` / `--throttle N` (override profile),
`--no-auto-collect`, `--foreground`. `--resume <bundle>` re-feeds only the not-yet-submitted /
failed chunks (offsets + per-chunk job ids recorded in the manifest).
**Fallback:** no `sbatch` on PATH (this workspace / a scheduler-less login node) → behaves like
`--emit` + prints the manual submit line; never crashes.

### C.5 Required lockstep — internal callers lose the bare form
"No default" means `python run.py <param>` (no mode) now errors, so **both internal callers
must move to `--local` in the same commit** or sweeps/HPC break (each runs a single-combo
`.param`, so `--local` is correct):
- `trinity/_input/sweep_runner.py:286` → `[python, run.py, param, '--local']`
- `trinity/_input/sweep_jobs.py:81` (sbatch template body) → `python "{run_py}" "$PARAM" --local`
Also rename `--collect-report` → `--collect` (callers: the new dependency job + docs).

### C.6 `--workers`/throttle — one concept, "max concurrent runs"
Local pool size (`--local`) and array throttle (`--submit`) are the same idea: how many at
once. Keep one `--workers` for the local pool; the array throttle's default lives in the
profile (`[submit] throttle`) with a per-run `--throttle` override. No more baking `%150` into
a script.

### C.7 Cruft cleanup (from the C.0 inventory)
Rename the overloaded `cluster` module (`trinity/_functions/cluster.py` = CPU detection, not
stellar) + `tools/cluster/`; document or remove `tools/plot_sweep_heatmap.py`,
`docs/dev/cluster/PLOTTING_WORKFLOW.md`, `param/paperII_grid_sweep*.param`; resolve the phantom
`trinity-plot`.

### C.8 Docs to update
CLAUDE.md:13-17, README.md:42-77, docs/source/running.rst (14/133/136/152), index.rst:35 —
rewrite bare-form examples to `--local`/`--submit`, document `--emit`/`--collect`, the **site
profile + `~/.bashrc` env one-time setup**, and a worked Helix example reproducing the paperII
sweep in a single `--submit`.

### C.9 Gate (outward-facing; fully scheduler-free CI)
1. **Profile parsing:** unit-test an example `cluster.ini` → directives + injected prologue; an
   absent profile → today's template + `--export=NONE`.
2. **Chunk math:** `offsets(N, C)` table test, incl. N<C (single submit, no chunking) and exact
   multiples (the `0/880/1760` case).
3. **Feeder:** fake `sbatch` on PATH that returns `QOSMaxSubmitJobPerUserLimit` then a parseable
   job id; assert it retries then succeeds, records ids in the manifest, and `--resume` skips
   done chunks. (no real SLURM; mock `subprocess` + a tiny stub `sbatch`.)
4. **Equivalence/baseline:** capture the emitted sbatch + manifest for `param/sweep_example.param`
   before; diff after. Every README/running.rst example reproduces.
5. `pytest` green before/after. **Persist** the example profile, the before/after emit diff, and
   the captured `sbatch` command strings as committed artifacts under `docs/dev/cli-rationalization/`.

**Risk.** Medium–high: outward-facing + new profile/submit/collect orchestration. Land after
A+B, own commit(s), run C.9. The profile + feeder are the largest new surface — build behind
the scheduler-free harness (C.9.1-3) *before* wiring into `run.py`.

---

## Sequencing
1. ✅ **A — SHIPPED** (commit `feat(output): honor TRINITY_OUTPUT_DIR…`). Env-driven output
   base incl. relative `path2output` via the shared `resolve_output_path`; full env×sentinel
   test matrix; 598-test suite green. Kills the hardcoded `/gpfs/...` line.
2. ✅ **B — SHIPPED** (commit `feat(metadata): store sps_path repo-relative…`). `portable_path`
   + final_state relativization; 8 metadata.json + fixture scrubbed; 602-test suite green.
3. ✅ **C2a — SHIPPED** (commit `feat(hpc): site profile + profile-driven sbatch generation`).
   `cluster_profile.py` (INI) + `_render_sbatch`; emitted sbatch is now Helix-ready (prologue +
   `--export=NONE`) through the *current* `--emit-jobs`. 10 tests; suite 612.
4. ✅ **C2b — SHIPPED** (commit `feat(hpc): chunking + feeder + auto-collect orchestration`).
   `cluster_submit.py`: `compute_chunks`, `feed_and_collect` (QOS-retry, resume, auto-collect),
   injectable runner/sleep. 15 scheduler-free tests. Inert until wired.
5. ✅ **C1 — SHIPPED** (commit `feat(cli): restructure run.py into explicit run modes`).
   `--local`/`--submit`/`--emit`/`--collect`/`--resume`; hard cutover (bare form errors);
   internal callers pass `--local`; `--submit` wired to `cluster_submit` with detached feeder +
   `--resume`; docs (CLAUDE.md/README/running.rst/index.rst) rewritten. Suite 634.
6. ✅ **C-cruft (high-value part) — SHIPPED**: renamed `trinity/_functions/cluster.py` →
   `cpu_allocation.py` (the import now reads as CPU/allocation, not stellar-cluster physics);
   updated its 2 importers (`run.py`, `test_sweep_workers.py`); fixed stale `--emit-jobs`/
   `--collect-report` references in run.py docstring, sweep_jobs error messages, and
   `tools/plot_sweep_heatmap.py`. Confirmed the phantom `trinity-plot` alias does not exist
   anywhere (nothing to remove). **Optional leftovers** (cosmetic, deferred): rename
   `tools/cluster/` (HPC plot-env dir) and surface `tools/plot_sweep_heatmap.py` /
   `param/paperII_grid_sweep*.param` in the built docs.

## Decisions (2026-06-23)
- **Output:** `TRINITY_OUTPUT_DIR` is the write-side base; `path2output` may be `def_dir`,
  relative (→ under the env base), or absolute (as-is). PaperII param drops its `/gpfs/...` line.
- **Command surface:** required mode flag, no default — `--local`/`--submit`/`--emit DIR` on the
  `python run.py x.param` spine. Bare form errors.
- **HPC = one command:** `--submit` emits + chunks (auto offsets from one `chunk`) + feeds past
  the QOS cap (built-in, backgrounded) + auto-collects (`afterany` dep). No hand-edited sbatch,
  no `submit_loop.sh`, no manual offsets/throttle.
- **Site profile** (`~/.config/trinity/cluster.ini`, stdlib `configparser`): partition / time /
  mem / account / `export` / throttle / chunk / env-prologue — set ONCE. job-name + log dir are
  auto-derived, not configured.

## Open decisions to confirm before coding
1. **Profile format/location:** INI at `~/.config/trinity/cluster.ini` (recommended, stdlib,
   no dep) vs YAML (nicer, but PyYAML is undeclared) vs reuse the `.param` parser. Default: INI.
2. **Disconnect default:** `--submit` backgrounds the feeder (`setsid` + `submit.log`, recommended)
   vs runs foreground and tells you to use tmux. Default: background.
3. **`account` handling:** rely on profile `account` + `$SBATCH_ACCOUNT` (recommended) — confirm
   Helix doesn't *require* an explicit `--account` directive.
4. **§B optional** `.sh` `cd /home/user/trinity` fix in dev harnesses: in scope now, or defer?

## Branch note
Work continues on **`feature/helix-implementations`** (user-directed). The earlier plan commits
landed on `claude/adoring-volta-3nrcmt` (the session's auto-assigned branch); this revision and
all subsequent work go to `feature/helix-implementations`, branched from that HEAD so the prior
plan history carries over.
