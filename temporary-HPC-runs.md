# temporary-HPC-runs — things to run on HPC/Helix (delete this file after you've run them)

> Scratch checklist for the maintainer. Branch: `feature/pdv-trigger-pt4b`. Everything else in the
> f_A workstream is being run **in-container** (bench5 Phase 5 — you ruled it in-container, it's
> completing; see `docs/dev/transition/pdv-trigger/FINDINGS.md §15h`). The item(s) below are the
> ones that genuinely need Helix. Run, then `git rm temporary-HPC-runs.md`.

---

## 1. theta5s — Phase-4 confirmation (81-arm f_A matrix)  ⟵ MANDATORY (the only real HPC need)

**Why:** the theta5s 81-arm matrix was run in Claude's ephemeral container (HPC was down), so it is
**PROVISIONAL, not authoritative** (`FINDINGS.md §15e`). Numerical fidelity in-container vs Helix is
unverified. The headline numbers — collapse-law **p = 3.330** (confirms registered p_source ≈ 3.3),
both controls never fire, the 3-class structure, and the dMdt suppression trend — must be reproduced
on Helix before any of them goes in the paper. If Helix disagrees, Helix wins.

**Params are already committed** (`docs/dev/transition/pdv-trigger/runs/params/theta5s/`, 81 arms).
Tooling is ready. From your laptop (repo checked out, `HELIX` ssh alias set):

```bash
# one-shot driver (git pull on Helix + mkdir logs + sbatch the 81-run array):
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh submit
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh watch     # queue + tail newest task log
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh down      # rsync theta5s_summary.csv back
```

or directly on Helix (after `git pull` on the cluster):

```bash
sbatch docs/dev/transition/pdv-trigger/runs/run_theta5s.sbatch    # array 1-81, --time=1:30:00/arm
# after it finishes, harvest:
python docs/dev/transition/pdv-trigger/runs/harvest_theta_max.py "$WS"/outputs/theta5s/* \
    --csv docs/dev/transition/pdv-trigger/runs/data/theta5s_summary.csv
```

**Then (either path), re-run the analysis + re-check every downstream number:**

```bash
python docs/dev/transition/pdv-trigger/data/make_theta5s_analysis.py      # p=3.33, fire map, controls
python docs/dev/transition/pdv-trigger/runs/harvest_dmdt_suppression.py \
    "$WS"/outputs/theta5s/* --csv docs/dev/transition/pdv-trigger/data/theta5s_dmdt_suppression.csv
```

**Compliance gate:** every arm must show `t_final ≥ 5` or a physics termination; re-run any
wall-killed/nonzero-exit arm longer before quoting θ. Report "N/81 compliant".
**On success:** the fresh `theta5s_summary.csv` header replaces the PROVISIONAL banner; update
`FINDINGS.md §15e` from "in-container/ASSUMED" to "HPC-confirmed" (or record the disagreement).

---

## 2. (optional) bench5 — authoritative confirmation, only if you want HPC numbers too

Not required — you ruled bench5 in-container and it's completing there (`§15h`). Listed only so it's
not forgotten: if you later want Helix-grade bench5 numbers, the 60 params are committed
(`runs/params/bench5/`); a bench5 sbatch would mirror `run_theta5s.sbatch` (array 1-60, `--time=2:00:00`).
Ask Claude to write `run_bench5.sbatch` if/when you want this. Otherwise ignore.

---

_Created 2026-07-12 by the f_A in-container session. Delete after running §1._
