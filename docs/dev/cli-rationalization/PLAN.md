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

### C.1 Proposed subcommands
```
python run.py run     <file.param>                 # single run (explicit)
python run.py sweep   <file.param> [--workers N] [--yes] [--dry-run]
python run.py emit    <file.param> <DIR> [--array-throttle K] [--dry-run]
python run.py collect <DIR>
```
This splits the overloaded `--workers`: pool size stays `--workers` on `sweep`; the SLURM
array throttle becomes its own `--array-throttle` on `emit`. `--dry-run` keeps a clear
per-subcommand meaning.

### C.2 OPEN DECISION — backward compatibility (recommend **Option 1**)
- **Option 1 (recommended): keep the bare form as a deprecated alias.** A shim catches
  "first positional is a file / not a known subcommand", routes to today's auto-detect, and
  prints a one-line deprecation nudge. **Nothing breaks.** Slightly more code.
- **Option 2: hard cutover.** `python run.py x.param` → argparse `invalid choice` (exit 2).
  Cleaner, less code, but **breaks two internal call sites in lockstep** and every doc:
  - `trinity/_input/sweep_runner.py:286` — in-process sweep `subprocess.run([python, run.py, param])`
  - `trinity/_input/sweep_jobs.py:81` — sbatch template `python "{run_py}" "$PARAM"` per array task
  These must move to `run.py run <param>` at the same time or sweeps + HPC arrays break.
- **Option 3: add a `trinity` console_script** (`pyproject` entry point) → `trinity run ...`.
  Most discoverable; bigger packaging surface. Orthogonal to 1 vs 2 (can layer on either).

Either way, internal call sites should be updated to the explicit `run` subcommand for clarity;
Option 1 just means they keep working if missed.

### C.3 Docs to update (both options)
CLAUDE.md:13-17, README.md:42-77, docs/source/running.rst (14/133/136/152), index.rst:35.
Option 1 can update progressively; Option 2 must update all at once.

### C.4 Gate (outward-facing CLI change)
1. Define equivalence: every example in README/running.rst produces the same run as before
   (single, sweep `--dry-run`, `emit` bundle contents, `collect`).
2. Baseline: capture `emit` output (sbatch + manifest) on `param/sweep_example.param` *before*
   the change; diff after — array-task command + manifest must be equivalent.
3. `pytest` green before and after (sweep_parser / sweep_jobs / sweep_runner tests).
4. Persist the before/after `emit` bundle diff as a committed note here.

**Risk.** Medium, outward-facing. Land C after A+B, as its own commit/review.

---

## Sequencing
1. **A** (env var write-side) — small, safe, unblocks the workspace setup.
2. **B** (portable paths + scrub) — small, safe, fixes the leak the user flagged.
3. **C** (subcommands + cruft) — outward-facing; separate commit, run the C.4 gate. Needs the
   §C.2 decision confirmed first (recommend Option 1).

## Open decisions to confirm before coding
- §C.2 command surface: Option 1 (recommended) / 2 / 3.
- §B optional `.sh` `cd` fix: in scope now, or defer?
