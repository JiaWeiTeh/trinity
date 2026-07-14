# temporary-HPC-runs — things to run on HPC/Helix (delete this file after you've run them)

> Scratch checklist for the maintainer. Branch: `feature/pdv-trigger-pt4b`. HPC is back
> (2026-07-13), so this is now the full ordered batch list for the f_A workstream. All params
> and tooling are committed; each batch is one `sbatch` array + one harvest + one `down`.
> Run in the order below (§1 is the only *mandatory* one; §2–§3 are the Phase-6 decision
> inputs). When all three are harvested and analysed, `git rm temporary-HPC-runs.md`.

---

## 1. theta5s — Phase-4 confirmation (81-arm f_A matrix)  ⟵ MANDATORY

**Why:** the theta5s 81-arm matrix ran in Claude's ephemeral container (HPC was down) and is
**PROVISIONAL** (`FINDINGS.md §15e`). The headline numbers — collapse-law **p = 3.330**, both
controls never fire, the 3-class structure, the dMdt suppression trend — must be reproduced on
Helix before any goes in the paper. If Helix disagrees, Helix wins.

```bash
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh submit   # git pull on Helix + sbatch 1-81
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh watch
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh run      # harvest -> theta5s_summary.csv
./docs/dev/transition/pdv-trigger/runs/sync_theta5s.sh down     # rsync into runs/data/
```

**Then analyse + re-check every downstream number:**

```bash
python docs/dev/transition/pdv-trigger/data/make_theta5s_analysis.py      # p=3.33, fire map, controls
python docs/dev/transition/pdv-trigger/runs/harvest_dmdt_suppression.py \
    "$WS"/outputs/theta5s/* --csv docs/dev/transition/pdv-trigger/data/theta5s_dmdt_suppression.csv
```

**Compliance gate:** every arm `t_final ≥ 5` or a physics termination; re-run wall-killed arms
longer before quoting θ. Report "N/81 compliant". On success update `FINDINGS.md §15e`
in-container/ASSUMED → HPC-confirmed (or record the disagreement).

---

## 2. bench5 — Phase-5 HPC confirmation (60-arm L21b matrix)

**Why:** the Phase-5 result (`FINDINGS.md §15h`: fire map 1→4→12→>16→>16; Θ_cum calibration
bench3≈16, bench2>16, bench1≫16) is in-container/PROVISIONAL. This re-run makes it authoritative
AND measures in-container-vs-HPC fidelity for the first time (a standing unknown). Params are
committed (`runs/params/bench5/`, do not re-emit).

```bash
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench5 submit   # sbatch 1-60, 3h/arm
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench5 watch
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench5 run     # -> bench5_summary_hpc.csv + traj
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench5 down    # into runs/data/ (in-container CSV kept)
python docs/dev/transition/pdv-trigger/data/compare_bench5_hpc.py   # per-arm Δθ_max + fire-map flips
```

Known non-critical case: `bench5_m5e5_r2p5__fa16_diag` froze at t=0.037 in-container (stiffness,
not walltime) — if it wall-kills on Helix too, record it, don't chase with longer limits.

---

## 3. bench6 — Phase-6 decision matrix (f_A dose extension + f_mix head-to-head, 60 arms)

**Why:** Phase 5 ended on "no single global f_A reproduces L21b" — but the pre-committed Phase-6
tree (SOURCE_TERM_DESIGN §3 Phase 6) needs two more measurements to pick its row:
(a) **where** (if ever) the diffuse benches enter the band — bench1/bench2 × f_A {24,32,64,128},
bench3 × {24,32} (overshoot check); (b) **the f_mix head-to-head** — all 5 benches ×
`cooling_boost_mode=multiplier`, f_mix {2,3,4,8}, same Θ_cum/blowout-window metric. The decision
metric is band-entry-dose **uniformity across density** per knob (see
`data/make_bench6_analysis.py` docstring; the physical asymmetry — f_A suppresses dMdt in-ODE,
f_mix freezes the structure — is already established and sim-free).

```bash
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench6 submit   # sbatch 1-60, 3h/arm
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench6 watch
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench6 run     # -> bench6_summary.csv + traj
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench6 down
python docs/dev/transition/pdv-trigger/data/make_bench6_analysis.py  # head-to-head + band-entry doses
```

Compliance gate as always: `t_final ≥ 5` or physics termination per arm; "N/60 compliant".

---

## After all three are down

1. `python data/make_bench5_analysis.py` + `make_bench6_analysis.py` + `make_theta5s_analysis.py`.
2. Update `FINDINGS.md` §15e/§15h from PROVISIONAL → HPC-confirmed (or record disagreements —
   HPC wins); write the §15i decision entry mapping the results onto the Phase-6 tree row.
3. Commit the downed CSVs/traj dirs from the laptop (they are the durable diagnostics).
4. `git rm temporary-HPC-runs.md`.

_Created 2026-07-12; expanded 2026-07-13 (HPC restored — batches 2–3 added). Sections 2–3 were
designed by the f_A session; the sbatch/sync mirrors run_theta5s.sbatch/sync_theta5s.sh._
