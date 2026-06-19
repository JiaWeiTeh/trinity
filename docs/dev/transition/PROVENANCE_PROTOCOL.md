# Clean-baseline / provenance protocol — `transition/` workstream

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time protocol; re-check the runner interface and paths against the
> current source before relying on it.
>
> 🔄 **Living protocol — recheck and refine on every visit.** If the runner
> changes or a better contamination guard exists, update this doc and date it.
>
> 💾 **Persist diagnostics — commit, don't re-run.** Every measurement that
> backs a decision must be committed with its provenance stamp; the `/tmp` runs
> it came from are ephemeral.

## Why
The transition fix is large and must start from **zero contamination**. The
sibling `failed-large-clouds/data/` set accreted across four code states this
session with no embedded provenance (see
`../failed-large-clouds/data/PROVENANCE.md`). This protocol guarantees that
**every** artifact in `transition/` records the exact commit + command + param
that produced it, and that a batch can be machine-verified to share one clean
commit.

## The contract
1. **One commit per batch.** All runs feeding a single conclusion come from the
   same commit. No mixing stages.
2. **Clean tree.** Runs come from a committed state. The runner *refuses* a
   dirty tree unless `--allow-dirty`, which records `tree_dirty=true` + the diff
   hash so any taint is logged, never hidden.
3. **Stamp every output.** Each run writes `provenance.json` next to its output
   (`commit`, `command`, `param_sha256`, `wall_seconds`, `returncode`, ...).
4. **Verify before trusting.** Before using a batch, assert it is single-commit
   and clean.

## Usage
```bash
# one run, stamped (refuses a dirty tree)
python docs/dev/transition/harness/run_stamped.py path/to/config.param

# verify a whole batch shares one clean commit (exit!=0 if mixed/tainted/dirty)
python docs/dev/transition/harness/run_stamped.py --check \
    /tmp/tbatch/<cfgA> /tmp/tbatch/<cfgB> ...
```
Each output dir then contains `provenance.json`, e.g.:
```json
{ "commit_short": "d919ff77", "tree_dirty": false,
  "command": "python run.py .../fail_repro.param",
  "param_sha256": "…", "wall_seconds": 90.0, "returncode": 0 }
```

## Phase-0 harvest discipline (applies the contract)
- Pick the commit, confirm `git status` clean, record it at the top of the
  results doc.
- Run **all** configs (normal controls + the verified-SSC probes `fail_repro`,
  `fail_helix`) through `run_stamped.py` into one batch dir.
- `--check` the batch; commit `provenance.json` alongside every harvested CSV.
- Any harvested CSV that is committed gets a header line:
  `# commit=<short> command=<…> param=<sha>` — or a sibling `*.provenance.json`.
- Re-confirm the Part-A line anchors against this commit before reading numbers
  off the run (the live tree drifts; the protocol pins the commit, not the API).

## Note on the existing discriminator
`../failed-large-clouds/data/discriminator.csv` was **regenerated** under this
contract from the synced HEAD and verified bit-identical to the prior values
(the fix is no-op pre-collapse; the LSODA fix is numerically inert). It carries
a provenance header as the worked example.
