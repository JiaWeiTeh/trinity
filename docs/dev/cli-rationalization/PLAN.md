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

**Date:** 2026-06-23 · **Branch:** `claude/adoring-volta-3nrcmt` · **Status:** plan only, no code yet (per user)

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

This doc plans all three as **separate reviewable commits**, low-risk first.

---

## A. Honor `TRINITY_OUTPUT_DIR` on the write side  *(small, safe)*

**Problem.** Read side honors the env var; write side does not. No single switch to
redirect output to a persistent mount.

**Fix.** In the two places that resolve the default output base, prefer
`os.environ['TRINITY_OUTPUT_DIR']` when set, else today's `<cwd>/outputs`:
- single run: `_resolve_path2output` (`registry.py:193-202`) — only the `'def_dir'`
  branch; an explicit user `path2output` is still taken as-is.
- sweep: `resolve_base_output_dir` (`run.py:137-148`) — same `'def_dir'` branch.

Committed `.param` files keep `path2output = def_dir`; the user exports the env var once
(`~/.bashrc`) and nothing absolute is committed. Behavior is unchanged when the var is unset.

**Test.** `test/` case: with `TRINITY_OUTPUT_DIR` set (monkeypatch), `read_param` on a
`def_dir` param resolves `path2output` under that base; unset → falls back to `<cwd>/outputs`.

**Risk.** Low/local. One sentinel branch in two functions + one test.

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

### C.2 `--submit` = one-shot: emit + submit + auto-collect  [user decision]
Sequence inside `--submit`:
1. Build the bundle (reuse `emit_jobs`) into an auto dir
   `<base_output_dir>/_jobs/<sweepname>_<timestamp>/` (or `--jobs-dir DIR` to override) — the
   user names nothing.
2. `sbatch <bundle>/submit_sweep.sbatch` → parse `Submitted batch job <ARRAY_ID>`.
3. `sbatch --dependency=afterany:<ARRAY_ID> --wrap "python run.py --collect <bundle>"` →
   the report is written automatically when the array finishes (`afterany` = even on partial
   failure). No second script file needed (`--wrap`).
4. Print both job IDs.
- **No sbatch editing.** Drop the `--account`/`--partition` directives from the template;
  SLURM reads `SBATCH_ACCOUNT` / `SBATCH_PARTITION` / `SBATCH_TIMELIMIT` from the env natively
  (set once in `~/.bashrc`, like `TRINITY_OUTPUT_DIR`). Keep `--mem`/`--time` defaults overridable.
  *(verify on Helix — a site can disable env propagation; standard SLURM honors it.)*
- **Fallback:** `shutil.which('sbatch') is None` (this cloud workspace, a login node without a
  scheduler) → behave like `--emit` (write bundle + print the manual `sbatch` line) and warn.
  No crash.
- **Known ceiling (ponytail):** one-shot assumes the array fits under the site `MaxSubmitJobs`
  cap. Larger grids keep the documented `OFFSET`-chunking escape hatch (`sweep_jobs.py:68-74`),
  run manually; auto-chunking is out of scope for v1.

### C.3 Required lockstep — internal callers lose the bare form
"No default" means `python run.py <param>` (no mode) now errors, so **both internal callers
must move to `--local` in the same commit** or sweeps/HPC break (each runs a single-combo
`.param`, so `--local` is correct):
- `trinity/_input/sweep_runner.py:286` → `[python, run.py, param, '--local']`
- `trinity/_input/sweep_jobs.py:81` (sbatch template) → `python "{run_py}" "$PARAM" --local`
Also rename `--collect-report` → `--collect` (its callers: the new dependency job + docs).

### C.4 `--workers` — keep one flag, meaning "max concurrent runs"  [recommend]
With an explicit mode the old overload becomes coherent: `--workers N` = concurrency cap,
read as the local pool size under `--local` and the array throttle `%N` under
`--submit`/`--emit`. One flag, one mental model ("how many at once"). *Not* splitting into two
flags unless the user prefers it — simpler wins.

### C.5 Cruft cleanup (from the C.0 inventory)
Rename the overloaded `cluster` module (`trinity/_functions/cluster.py` = CPU detection, not
stellar) + `tools/cluster/`; document or remove `tools/plot_sweep_heatmap.py`,
`docs/dev/cluster/PLOTTING_WORKFLOW.md`, `param/paperII_grid_sweep*.param`; resolve the phantom
`trinity-plot`.

### C.6 Docs to update
CLAUDE.md:13-17, README.md:42-77, docs/source/running.rst (14/133/136/152), index.rst:35 —
rewrite the bare-form examples to `--local` / `--submit`, document `--emit`/`--collect` and the
`SBATCH_*` + `TRINITY_OUTPUT_DIR` env setup.

### C.7 Gate (outward-facing + scheduler-free CI)
1. Equivalence: every README/running.rst example reproduces the same run via the new flags.
2. Baseline: capture the emitted sbatch + manifest for `param/sweep_example.param` before; diff after.
3. `--submit` is testable without SLURM — unit-test (a) the bundle/sbatch text and (b) the exact
   `sbatch` + `sbatch --dependency=afterany:…--wrap` command strings via a mocked `subprocess`
   and a fake `sbatch` on PATH returning a parseable job id; assert the no-`sbatch` fallback.
4. `pytest` green before/after. Persist the before/after emit diff + the captured command
   strings as committed artifacts here.

**Risk.** Medium, outward-facing + new submit/collect orchestration. Land after A+B, own commit, run C.7.

---

## Sequencing
1. **A** (env var write-side) — small, safe, unblocks the workspace setup.
2. **B** (portable paths + scrub) — small, safe, fixes the leak the user flagged.
3. **C** (mode flags + one-shot submit + cruft) — outward-facing; separate commit, run the C.7 gate.

## Decisions (2026-06-23)
- **Command surface:** required mode flag, no default — `--local` / `--submit` / `--emit DIR`
  on the existing `python run.py x.param` spine (§C.1). Bare form errors.
- **HPC flow:** `--submit` is one-shot — emit + sbatch + dependent auto-collect; `SBATCH_*`
  env for account/partition/time so the sbatch is never edited (§C.2).
- **Earlier (workspace/paths):** A honors `TRINITY_OUTPUT_DIR` write-side; B keeps `sps_path`
  but relative; cleanup scope is the full rationalization incl. these mode flags.

## Open decisions to confirm before coding
- §B optional `.sh` `cd /home/user/trinity` fix in dev harnesses: in scope now, or defer?
