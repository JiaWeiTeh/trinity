# `data-new/` — the trusted-data cut

**Cut commit:** `<paste the output of `git rev-parse --short HEAD` at the moment this directory
was created>`

## The one rule

> Nothing enters `data-new/` without a C-6 provenance stamp whose `code <SHA>` is at or after
> the cut commit.

That is the whole contract. It is deliberately not enforced by tooling: `harness/_stamp.py`
already writes the stamp on every artifact, so the check is `head -1` on any file you are
about to trust.

## Why a directory and not just the stamps

The stamps in `data/` already carry the information — the cut adds a *bright line*, which is
worth more than the information, because auditing twenty stamps by hand is the kind of thing
nobody does twice. A file's presence in `data-new/` is a claim that someone checked.

## What happens to `data/`

**Read-only history. Not deleted.** PLAN.md §0 says never delete history, and it has already
paid: B11.0's third independent route for seam C ran through the `n_cavity` column of
`data/b10_wind_profile.csv`. Old numbers stay quotable *as history*, with their date and SHA,
and must not be mixed into a new comparison (C-1: every comparison names its two SHAs).

## What lives here

| file | produced by | read next to |
|---|---|---|
| `<arm>_manifest.csv` | `harness/make_manifest.py` | **everything else in the arm** |
| `<arm>_alphap.csv` | `harness/alphap_screen.py` | the manifest |
| `<arm>_*.csv` | whichever committed reducer | the manifest |

**Always read the manifest first.** A short reduced CSV next to a full manifest means tasks
failed, not that the effect is small. `make_manifest.py` prints the failure list and flags
every task that never reached the momentum phase, because a run that does not reach the phase
a gate needs is **VOID**, never a confirming null.

## Raw output

`dictionary.jsonl` is ~9 MB per run, so a 1000-task array is ~9 GB. Raw output stays on helix
under `outputs/<arm>_<sha>/` and is **not** synced down. Reducers run there
(`helix.sh reduce`), and only CSVs come back.

Keep the raw until the arm's questions are answered. B11.0's S1 found a real bug in
`layer_density_check.py`'s layer volume *after* the numbers were committed; with raw output in
place that costs a re-reduce, without it a re-run.
