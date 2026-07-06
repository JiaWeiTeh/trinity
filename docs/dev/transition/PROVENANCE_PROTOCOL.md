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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (the archived trio
> `docs/dev/archive/transition/{TRIGGER_PLAN,P0,pshadow-design}.md`,
> and the `pdv-trigger/` successor docs in this folder). They drift out of sync *with each
> other* as fast as they drift from the code. Any agent or person editing one MUST, as part of the visit,
> circle back through the siblings and reconcile: if a number, status, claim, or line reference here
> contradicts a sibling — or a sibling has gone stale — fix it (or flag it, dated) so no two docs in the
> workstream disagree. Never update one in isolation.

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
`../failed-large-clouds/data/discriminator.csv` was **regenerated** from a single
stamped batch at commit `d919ff77`, with a `t≤1.0 Myr` window. The **failing**
configs came back **bit-identical** (`Eb_growth=1.014`) — confirming the fix is
no-op pre-collapse and the LSODA fix is numerically inert. The **healthy** values
*grew* (≥×39,300 / ≥×94,900 vs the old ×13,600 / ×37,900): the prior committed
healthy runs had been **truncated** at `t≈0.32 Myr`, so their numbers were an
even-lower lower bound. This is exactly the kind of mystery-provenance defect the
contract prevents going forward. The regenerated CSV carries a provenance header
(`commit`, `batch`, `t_cap_myr`) as the worked example.
