# Cluster plotting workflow (Helix) — plot where the data lives

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

*Written 2026-06-19.*

## The problem

A full sweep (~2600 runs) is ~50 GB. INFO-level `trinity.log` files are small; the
bulk is each run's `dictionary.jsonl`. That data is genuine and **fine to keep on
Helix** — "unavoidable" applies to *storage on the cluster*, **not** to *transfer*.
Don't scp/rsync the jsonl to your laptop: **plot on the cluster, pull only the PDFs.**

## How to plot on an interactive node

```bash
# once per session, on the interactive node:
source tools/cluster/plot_env.sh            # headless: Agg + Computer Modern, no LaTeX
python scratch/paper_radiusEvolution.py -F outputs/my_sweep   # renders straight to PDF

# then, from your laptop — figures only, never the jsonl:
rsync -avz --include='*/' --include='*.pdf' --include='*.png' --exclude='*' \
    helix:/path/to/Trinity/fig/ ./fig/
```

Tip — make it an alias on the cluster (`~/.bashrc`):
```bash
alias trinity-plot='source /path/to/Trinity/tools/cluster/plot_env.sh'
```

For one run you want to poke at interactively, rsync just that single run's directory
(a few MB–hundreds of MB for one run is fine; the 50 GB is the *sum* over 2600 runs).

## Artifacts (committed)

| Path | What it is |
|------|-----------|
| `tools/cluster/matplotlibrc` | Headless render config: `Agg` + Computer Modern mathtext, **no LaTeX toolchain**. |
| `tools/cluster/plot_env.sh` | `source` it on a node to point matplotlib at that rc (+ a writable `MPLCONFIGDIR`). Works in bash and zsh. |

## Headless matplotlib — no display, no LaTeX

`plot_env.sh` exports, before matplotlib imports:

```bash
export MATPLOTLIBRC="$REPO/tools/cluster/matplotlibrc"   # one file → all 54 scripts, zero edits
export MPLBACKEND=Agg
export MPLCONFIGDIR="$SCRATCH/mplcache"                  # writable font-cache dir (Helix /home often read-only at runtime)
```

The rc sets `text.usetex: False` + `mathtext.fontset: cm`, giving the **Computer Modern
LaTeX look with no `latex`/`dvipng`/`ghostscript` dependency** — the fonts ship inside
matplotlib, so it renders on any node. Verified: backend is `agg`, `usetex` is off,
`$R_2$`-style mathtext renders, and under `Agg` **`plt.show()` is a harmless no-op** — so
the 16 scripts that call `show()` need no editing.

**LaTeX split:**
- *Cluster figures* (per-run trajectories, sweep diagnostics) → mathtext `cm`, headless,
  no LaTeX. (The `scratch/` scripts already use **zero** `text.usetex=True`.)
- *Final publication figures* that rely on real LaTeX (`paper/methods`, `\newcommand`,
  `siunitx`, …) → render **locally**, or on a node with `module load texlive`, from a few
  rsynced jsonl — **never** from the 50 GB.

## Why there is no `plt.close()` / figure-leak fix

A figure leak only happens when a script creates a figure **inside a loop over many
runs** and never closes it. Audited (2026-06-19): the `scratch/` scripts that loop over a
whole sweep and create a figure per run (`paper_dominantFeedback.py:934`,
`paper_escapeFraction.py`, `paper_Cf.py`) **already** call `plt.close(fig)`. The scripts
flagged by a naive "savefig + no close" grep (`paper_BEprofile`, `paper_InitialCloudRadius`,
`paper_PISM`, `paper_bubblePhase`, `paper_PPVtest`) each create only **1–2 module-level
figures and exit** — no leak. No edits made.

If you ever write a *driver* that loops over many runs in one long-lived process, prefer
**one subprocess per script/run** anyway: trinity leaks module-level global state
in-process (see `CLAUDE.md`), so a shared process would be *incorrect*, not just
memory-hungry — and process exit frees every figure regardless of `close()`.

## Possible future levers (flagged, not done)

- **Per-run summary table.** Aggregate "one-point-per-run" figures only need a handful of
  scalars per run (final `R2`, `v2`, forces, `escape_fraction`, end reason). A reducer that
  reads each `metadata.json` + the last jsonl line into one small `summary.csv` would let
  you iterate those figures *locally* from a few-MB table. Built and tested in a prior pass
  (32 runs ≈ 700 MB → 48 KB) then removed as not currently needed — resurrect from git
  history if/when aggregate-figure iteration gets painful.
- **Shrinking the jsonl itself** (rows store large per-step arrays, e.g.
  `shell_grav_force_m`) would cut on-cluster *storage*, but that's an output-writer change
  (rule-5 / iterative-path territory). Revisit only if the Helix storage quota binds.
