# REPRODUCE — paper reproducibility manifest for the `pdv-trigger` workstream

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/transition/pdv-trigger/data/`, or a
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

---

**Purpose.** This is the single map from **every result in the storyline** (the narrative is
`pdvtrigger_report.html` / `PLAN.md` / `FINDINGS.md`) to **the exact `.param` you run + the command + the
artifact it produces**. Use it to (a) re-run any piece for a paper, (b) prove the storyline is reproducible,
(c) know what is *cheap* (re-reads a committed CSV in seconds) vs *expensive* (a full sim, minutes–hours).

**Key fact for reproducibility.** `outputs/` is **git-ignored** (TRINITY writes runs there). So the **durable
record is the committed DERIVED CSV** in `data/`, not the raw run. Every figure is a *pure read* of a committed
CSV — so you can **rebuild every figure without running a single sim** (see [§Rebuild all figures](#rebuild-all-figures-no-sims)).
You only re-run the sims when you want to regenerate the underlying CSV from scratch.

## How to run one simulation
```bash
# from repo root. writes outputs/<path2output>/dictionary.jsonl + metadata.json + trinity.log
python run.py docs/dev/transition/pdv-trigger/runs/params/<name>.param
```
Each `.param` overrides only the keys it sets; everything else falls back to `trinity/_input/default.param`.
The `path2output` key inside each `.param` says where its run lands (all the calibration runs land in
`outputs/kcal/`).

---

## The storyline, result by result

Legend — **Sims?**: 🟢 none (reads committed CSV, seconds) · 🟡 a few full runs (minutes) · 🔴 grid/HPC (hours).

| # | Result (claim) | Report § | Input `.param`(s) | Run → Analyze | Artifact (CSV + figure) | Sims? |
|---|---|---|---|---|---|---|
| 1 | Single-count closure (no double-count) | §2 | — (Monte-Carlo) | `python data/make_doublecount_mc.py` | `data/doublecount_mc.csv` | 🟢 |
| 2 | Trigger-convention fix; no constant `f_mix` fires across density (superseded by #28/§10: under θ_max a single f_mix=4 fires the band) | §3 | — (reads frozen) | `python data/make_fmix_table.py` | `data/fmix_table.csv`, `fmix_vs_density.png` | 🟢 |
| 3 | PdV regime split (sub- vs super-critical) | §4 | — (reads frozen) | `python data/make_pdv_regime_table.py` | `data/pdv_regime_budget.csv` | 🟢 |
| 4 | A constant knob is not enough (8-config staged screen) | §5 | — (reads frozen) | `python data/make_closure_test.py && python data/make_closure_plots.py` | `data/closure_test.csv`, `closure_stage*.png` | 🟢 |
| 5 | θ_target(Da) **refuted** (offline proxy) | §6 | — (reads frozen) | `python data/make_da_screen.py` | `data/da_screen.csv`, `da_screen.png` | 🟢 |
| 6 | θ_target(Da) refuted (gate-validated real-Da replay) | §6 | — (replays frozen) | `python data/make_da_replay.py` | `data/da_replay.csv`, `da_replay.png` | 🟢 |
| 7 | Literature anchor θ(n_H) vs TRINITY's resolved loss | §7 | — (reads frozen) | `python data/make_theta_density_plot.py` | `theta_vs_density.png` | 🟢 |
| 8 | Live matched-`t` edge runs (boost vs none) | §9 | `f1edge_{hidens,lowdens}__*`, `simple_cluster__*`, `fail_repro__*` | see [`runs/README.md`](runs/README.md) | `runs/data/live_compare.csv` | 🟡 |
| 9 | **κ_eff Rung A** raises emergent cooling (back-reaction) | §11 | `f1edge_hidens__none.param`, `f1edge_hidens__kappa2.param` | run both (separate processes) → `make_kappa_backreaction.py` | `data/kappa_backreaction.csv`, `kappa_backreaction.png` | 🟡 |
| 10 | **What f_κ is** (Spitzer multiplier; seed law verified) | §13 | — (reads #9's CSV) | `python data/make_fkappa_definition.py` | `fkappa_definition.png` | 🟢 |
| 11 | **f_κ calibration** θ→0.95 (compact/mid/diffuse) ⛔ CONTAMINATION #3 (blowout-θ, <5 Myr) | §13 | `cal_{compact,diffuse}__k{1,2,4}.param`, `cal_mid__ek{1,2,4}.param` | run the 9 → `make_kappa_blowout_calibration.py` | `data/kappa_blowout_calibration.csv`, `kappa_blowout_calibration.png` | 🔴 |
| 12 | PdV is the dominant sink (cool-only vs PdV-incl) | §12 | reuses #11's `cal_*__k{1,2,4}` | `python data/make_pdv_trigger_compare.py` | `data/pdv_trigger_compare.csv`, `pdv_trigger_compare.png` | 🟢¹ |
| 13 | **ebpeak does not fire at f_κ=1** (code-path test) | §12 | `cal_{compact,diffuse}__ebpeak.param`, `cal_mid__ek{1,2,4}.param` (+ #11) | run the ebpeak ones → `make_ebpeak_trigger_test.py` | `data/ebpeak_trigger_test.csv`, `ebpeak_trigger_test.png` | 🔴 |
| 14 | **Holds across 8 configs** (frozen + live overlay) | §12 | — (reads frozen + #13's CSV) | `python data/make_ebpeak_8config_xcheck.py` | `data/ebpeak_8config_xcheck.csv`, `ebpeak_8config_xcheck.png` | 🟢 |
| 15 | Dense-edge stiffness is **not** f_κ (it's extreme density) | PLAN ledger 06-28 | `diag_dense_hybr.param`, `diag_dense_legacy.param` | run both, observe (does not finish at nCore 1e6) | `data/dense_stiffness_diag.csv` | 🟡 |
| 16 | FM1 / FM1b — wrong knobs ruled out (κ_eff confirmed) | §11 | — (offline prototypes) | `python data/make_fm1_rootcheck.py`; `python data/make_fm1b_evapsign.py` | `data/fm1*.csv`, `fm1*.png` | 🟢 |
| 17 | All-ideas scoreboard | hero | — (reads CSVs above) | `python data/make_ideas_comparison.py` | `ideas_comparison.png` | 🟢 |
| 18 | **Controlled f_κ(n_H) calibration** (+ de-conflation test, **RAN on Helix 2026-06-29** — 786/819 ok, 10h17m; `data/sweep_report.txt`) | `F_KAPPA_FUNCTIONAL_FORM.md` §8 | `runs/params/sweep_fkappa_nH.param` (sweep → **819** combos) | `sync.sh submit`→`collect`→`reduce`→`down`, then `make_fkappa_nH_sweep.py` (Block C) | `data/summary.csv` → `data/fkappa_nH_sweep.csv` (committed result), `fkappa_nH_sweep.png` | 🔴 done |
| 19 | **Sweep prediction scorecard** (measured vs pre-registered form) | `F_KAPPA_FUNCTIONAL_FORM.md` §8 | — (reads #18) | `python data/make_fkappa_sweep_analysis.py` | `data/fkappa_sweep_scorecard.csv`, `fkappa_sweep_analysis.png` | 🟢 |
| 20 | **Fan-out anatomy** — catastrophic-cooling cliff + column collapse + metric sanity | `F_KAPPA_FUNCTIONAL_FORM.md` §9–§10 | — (reads `data/summary.csv`) | `python data/make_fkappa_cliff_metric.py` | `data/fkappa_cliff_metric.csv`, `fkappa_cliff_metric.png` | 🟢 |
| 21 | **De-conflation figure** (3-panel, faceted by sfe) | `F_KAPPA_FUNCTIONAL_FORM.md` §8–§9 | `data/summary.csv` (reduced sweep) | `python data/make_fkappa_nH_sweep.py` | `data/fkappa_nH_sweep.csv`, `fkappa_nH_sweep.png` | 🟢 |
| 22 | **Physical-cap reframing** — sign flip + critical column + momentum/energy split | `F_KAPPA_FUNCTIONAL_FORM.md` §11–§12 | — (reads `data/summary.csv`) | `python data/make_fkappa_physical_cap.py` | `data/fkappa_physical_cap.csv`, `fkappa_physical_cap.png` | 🟢 |
| 23 | **Physical prescription derived** — κ_mix(λδv) crossover, scalar-f_κ-can't, the θ* gap | `F_KAPPA_FUNCTIONAL_FORM.md` §13 | constants + `data/summary.csv` | `python data/make_fkappa_physical_derivation.py` | `data/fkappa_physical_derivation.csv`, `fkappa_physical_derivation.png` | 🟢 |
| 24 | **κ_mix offline prototype** — does mixing dominate the cool layer? units-correct, no solver | `KMIX_PROTOTYPE.md` | `runs/data/harvest_*.csv` (Pb time series) | `python data/make_kmix_prototype.py` | `data/kmix_prototype.csv`, `kmix_prototype.png` | 🟢 |
| 25 | **θ₁-collapse + de-conflation verdict** (n_H-only REFUTED; universal leverage p≈0.27; pt3) ⚠️ PROVISIONAL (⛔ #4) | `FINDINGS.md` §9 | — (reads #18's `data/fkappa_nH_sweep.csv`) | `python data/make_fkappa_theta1_collapse.py` | `data/fkappa_theta1_collapse.csv`, `fkappa_theta1_collapse.png` | 🟢 |
| 26 | **`cooling_boost_kappa='auto'` acceptance** — **RAN 2026-07-01** (in-container, ~14 min): auto→12.0, cooling_balance fired t≈0.375, momentum, θ_max=1.061; **4/4 checks PASS** (grid calibration itself stays ⚠️ PROVISIONAL, `FINDINGS.md` §9 flags) | `FINDINGS.md` §9 | `runs/params/fkauto_verify.param` | `python run.py runs/params/fkauto_verify.param`, then `python data/make_fkappa_auto_verify.py` | `data/fkappa_auto_verify.csv` (committed) | 🟡 done |
| 27 | **Kappa stability map** — §8e⇄§9 tension resolved: breakdown non-monotonic in f_κ (17/57 cells; 38 frozen runs; §8e's θ≈0.53 reproduced on Helix) | `FINDINGS.md` §9a | — (reads #18's `data/summary.csv`) | `python data/make_kappa_stability_map.py` | `data/kappa_stability_map.csv` | 🟢 |
| 28 | **📏 theta5 matrix** — **RAN on Helix 2026-07-02** (32/32 compliant): multiplier θ₁-collapse law; **f_mix=4 fires the GMC band incl. diffuse**; route-a = small_1e6/fail_repro; fire-then-recollapse + over-boost Eb-drain flagged | `FINDINGS.md` §10 | `runs/params/theta5/` (32) | `sbatch runs/run_theta5.sbatch` (or `runs/sync_theta5.sh`), then `runs/harvest_theta_max.py` → `runs/make_theta5_calibration.py` | `runs/data/theta5_summary.csv`, `runs/data/theta5_calibration.csv` | 🔴 done |
| 29 | **theta5 publication figures + f_mix candidate scorecard** (F1 arms ladder, F2 θ₁-collapse law, F3 metric correction, F4 target-vs-emergent, F5 knob choice; scorecard = per-config margins for the f_mix pin — **f_mix=4 adopted 2026-07-02**, maintainer recollapse ruling) | `FINDINGS.md` §10 | — (reads #28 + #18 + committed CSVs) | `python data/make_theta5_figures.py` | `theta5_{arms,collapse_law,metric_correction,target_vs_emergent,knob_choice}.png`, `runs/data/theta5_fmix_scorecard.csv` | 🟢 |
| 30 | **theta5b fine bracket + long diffuse arms** — **RAN on Helix 2026-07-02** (43/43): whole-band window **[4, 4.5]**; law out-of-sample rms 0.064 dex; fire-vs-drain race; diffuse f=2 fires at t≈5.04 Myr; dense edge fires at every fine arm | `FINDINGS.md` §11 | `runs/params/theta5b/` (43) | `sbatch runs/run_theta5b.sbatch` (or `runs/sync_theta5b.sh`), then `runs/harvest_theta_max.py` → `python data/make_theta5b_analysis.py` | `runs/data/theta5b_summary.csv`, `data/theta5_fire_map.csv`, `data/theta5_law_check.csv`, `theta5b_{fire_map,law_check}.png` | 🔴 done |
| 31 | **kappa freeze autopsy + mechanism + instrumentation** — §9a re-examined on maintainer challenge: freezes are the evaporation→condensation domain boundary (dMdt eigenvalue goes negative; gate refuses; runner grinds), NOT physics bands; live repro found the solver converging to dMdt=−84.76 at f_κ=8; log-only instrumentation added (freeze-watch trace, streak diagnosis) | `FINDINGS.md` §9b, `KAPPA_FREEZE_MECHANISM.md` | — (reads committed `data/summary.csv`; repro params in session scratchpad, mechanism-only) | `python data/make_kappa_freeze_autopsy.py`; repro: simple_cluster + `cooling_boost_kappa 8` + `log_level DEBUG`, grep `freeze-watch`/`no physical` | `data/kappa_freeze_autopsy.csv` | 🔴 done |
| 32 | **fix #1 (no-root ⇒ momentum handoff) + theta5k matrix** — persistent dMdt<0 streak ends the energy phase as a handoff fate instead of freezing; theta5k = the first rule-compliant kappa validation (needs this branch — pre-fix code freezes) | `FINDINGS.md` §9b, `KAPPA_FREEZE_MECHANISM.md` §7.1 | `runs/params/theta5k/` (56) | verify: `python runs/drive_noroot_handoff_check.py <fk8-param> 3`; run: `sbatch runs/run_theta5k.sbatch`, harvest as theta5b | `runs/data/theta5k_summary.csv` (future) | 🟡 ready |
| 33 | **theta5k RAN + analyzed** — 56/56 proper fates, ZERO freezes (fix #1 at scale); 5 CONDENSE handoffs on the old dead-window cells; fire set non-monotonic (physical race) but θ_max rises monotonically; **no whole-band f_κ** (best 5/6 at k12) vs multiplier [4,4.5] 6/6 | `FINDINGS.md` §12, `KAPPA_FREEZE_MECHANISM.md` §8 | `runs/params/theta5k/` (56) | `sbatch runs/run_theta5k.sbatch`, `runs/sync_theta5k.sh`, then `runs/harvest_theta_max.py` → `python data/make_theta5k_analysis.py` | `runs/data/theta5k_summary.csv`, `data/theta5k_fire_map.csv`, `theta5k_{fire_map,theta_rise}.png` | 🔴 done |
| 34 | **theta5n — the 9th standard config (normal_n1e3: mCloud 1e6, nCore 1e3, sfe 0.01, PL0; M_cluster=1e4), both knobs** — fine multiplier bracket {1,2,2.5,3,3.5,4,4.5,5,8} + kappa {2,4,6,8,12,16}; tests whether f_mix=4 / window [4,4.5] still fires the nine-config band; law predicts f_fire from θ₀ out-of-sample | `FINDINGS.md` §12.6 | `runs/params/theta5n/` (15) | `sbatch runs/run_theta5n.sbatch`, then `runs/harvest_theta_max.py` → fitter/analysis | `runs/data/theta5n_summary.csv`, 9-row `data/theta5_fire_map.csv` + `data/theta5k_fire_map.csv`, 7-point `data/theta5_law_check.csv` | 🔴 done (RAN 2026-07-03: fires NATIVELY at f=1, θ₀=1.047; law resid 0.065 dex; FINDINGS §13) |
| 35 | **dMdt dip figures + trace CSV** — the controlled dense k6-vs-k8 eigenvalue traces (the dip below zero, recovery-vs-second-dive) and the resolution flow diagram; report §16.5 tells the full arc | `KAPPA_FREEZE_MECHANISM.md` §5, `FINDINGS.md` §12.7 | `runs/params/dmdt_trace/dense_k{6,8}.param` | local `python run.py <param>` (DEBUG logs), then `python data/make_dmdt_dip_figures.py` (reads the committed CSV; PARSE_LOGS to re-parse) | `data/dmdt_trace_dense.csv`, `dmdt_dip_traces.png`, `dmdt_tackle_flow.png` | 🔴 done |
| 36 | **f_A interface source-term screen** — the fourth knob corner (source × scalar): boost `dudt` in the T<10^5.5 band inside `_get_bubble_ODE`; 4/4 predictions pass 6/6 (continuous dial, dMdt FALLS = the El-Badry sign, 300/300 stable, no domain-edge cliff) | `SOURCE_TERM_DESIGN.md` §3, `FINDINGS.md` §15 | — (replays frozen C0) | `python data/make_fA_source_boost.py` (env: `FA_LIST`, `N_ROWS`, `CONFIGS`) | `data/fA_source_boost{,_summary}.csv`, `fA_source_boost.png` | 🟢 |
| 37 | **f_A Phase 1** — all-9 offline coverage (2 new committed trajectories + 2 FM1 fixtures) + condensation-edge map. Result: the θ≈1 edge prediction FALSIFIED in the safe direction — NO dMdt≤0 edge to f_A≤128 (0/50 states), and a probe to f_A=512 finds none; the source knob has no reachable condensation edge | `SOURCE_TERM_DESIGN.md` §3 Phase 1, `FINDINGS.md` §15a | `cleanroom/configs/{small_1e6,normal_n1e3}.param` (run to stop_t 1 for the trajectories) | `python data/make_fA_edge_map.py` (env: `N_ROWS`, `FA_COV`, `FA_EDGE`, `CONFIGS`); trajectories via `python run.py docs/dev/transition/cleanroom/configs/<cfg>.param` (stop_t 1, OMP_NUM_THREADS=1) → keep replay columns | `data/fA_edge_map.csv`, `data/fA_coverage9.csv`, `data/traj_{normal_n1e3,small_1e6}.csv`, `fA_edge_map.png` | 🟡 (trajectories) / 🟢 (map) |
| 38 | **f_A Phase 2** — production wiring: `cooling_boost_fA` ParamSpec + validator (`registry.py`), two edit sites in `bubble_luminosity.py` (RHS `dudt` band-multiply; L2/L3 component scaling) behind fA!=1 guards, default 1.0 byte-identical. 9 new unit tests; full pytest 742 green | `SOURCE_TERM_DESIGN.md` §3 Phase 2, `FINDINGS.md` §15b | — (code) | `python -m pytest test/test_fA_source_boost.py`; after any registry edit `python -m tools.gen_default_param --write` | `trinity/_input/registry.py`, `default.param`, `trinity/bubble_structure/bubble_luminosity.py`, `test/test_fA_source_boost.py` | 🟢 |

¹ #12 reads the same `cal_*__k{1,2,4}` runs as #11 — once those exist in `outputs/kcal/`, #12 is a 🟢 re-read.

---

## The two expensive blocks (🔴) — exact commands

### Block A — f_κ calibration grid (results #11, #13)
```bash
# 9 full runs: compact & diffuse at f_kappa in {1,2,4} (cooling_boost_kappa knob), default trigger
for c in compact diffuse; do for k in 1 2 4; do
  python run.py docs/dev/transition/pdv-trigger/runs/params/cal_${c}__k${k}.param
done; done
# 3 full runs: mid at f_kappa in {1,2,4} with ebpeak ACTIVE (cal_mid__ek{1,2,4})
for k in 1 2 4; do
  python run.py docs/dev/transition/pdv-trigger/runs/params/cal_mid__ek${k}.param
done
# 2 full runs: the dedicated ebpeak code-path test (transition_trigger=cooling_balance,ebpeak)
python run.py docs/dev/transition/pdv-trigger/runs/params/cal_compact__ebpeak.param
python run.py docs/dev/transition/pdv-trigger/runs/params/cal_diffuse__ebpeak.param
# then derive the committed CSVs + figures (cheap):
python docs/dev/transition/pdv-trigger/data/make_kappa_blowout_calibration.py
python docs/dev/transition/pdv-trigger/data/make_pdv_trigger_compare.py
python docs/dev/transition/pdv-trigger/data/make_ebpeak_trigger_test.py
```
Each `cal_*` run lands in `outputs/kcal/<model_name>/`. Compact/mid finish in minutes; **diffuse is slow**
(the `cal_diffuse__ebpeak` run goes to `stop_t=2.0`). For a clean single-variable density sweep on HPC, prefer
the array path — see **Block C** below (`runs/sync.sh` + `runs/run_fkappa.sbatch`), the worked Helix example.

### Block B — κ_eff back-reaction (result #9)
```bash
# separate processes + provenance, on the stiff dense edge:
python docs/dev/transition/harness/run_stamped.py docs/dev/transition/pdv-trigger/runs/params/f1edge_hidens__none.param
python docs/dev/transition/harness/run_stamped.py docs/dev/transition/pdv-trigger/runs/params/f1edge_hidens__kappa2.param
python docs/dev/transition/pdv-trigger/data/make_kappa_backreaction.py \
    outputs/pdvlive/f1edge_hidens__none/dictionary.jsonl \
    outputs/pdvlive/f1edge_hidens__kappa2/dictionary.jsonl
```

---

### Block C — controlled f_κ(n_H) calibration sweep (result #18; HPC, **RAN on Helix 2026-06-29** → artifacts committed, `data/fkappa_nH_sweep.csv`)
The clean replacement for the conflated 3-anchor estimate. Sweeps **nCore finely (primary axis) × a fine f_κ
grid** that brackets the firing point at every density, **and also varies mCloud + sfe** so we can test whether
`f_κ_fire` is a clean function of n_H alone or also depends on cloud mass / SFE.
**Grid = 7 nCore × 13 f_κ × 3 mCloud × 3 sfe = 819 combos** (HPC; under the 1000 ceiling).
Run it **reduce-then-plot** (the II-survey pattern): a committed, **pre-patched-for-Helix** array sbatch +
laptop driver (`runs/run_fkappa.sbatch`, `runs/sync.sh`) launch the grid; a **stdlib-only** reducer
(`data/reduce_fkappa_sweep.py`) walks the multi-GB jsonl ONCE on the cluster into a tiny `summary.csv`; only
that CSV crosses the wire, and the figure is fit/drawn on the laptop. The driver emits the bundle from
`/gpfs` so outputs land on the **writable** workspace, not the read-only `/home` repo checkout (the failure
mode a bare `sbatch jobs/submit_sweep.sbatch` from the repo hits).
```bash
# inspect anywhere (no cluster needed):
python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --dry-run     # lists 819 combos

# on Helix, driven from the laptop (code travels by git pull; this folder is TRACKED):
./docs/dev/transition/pdv-trigger/runs/sync.sh submit    # git pull + emit to $WS/jobs_fkappa + sbatch array 1-819
./docs/dev/transition/pdv-trigger/runs/sync.sh watch     # tail the running array (+ squeue)
./docs/dev/transition/pdv-trigger/runs/sync.sh collect   # run.py --collect-report -> sweep_report.{txt,json}
./docs/dev/transition/pdv-trigger/runs/sync.sh reduce    # jsonl -> summary.csv  (ON HPC, stdlib-only, ~minutes)
./docs/dev/transition/pdv-trigger/runs/sync.sh down      # rsync summary.csv -> data/  (the tiny table, not jsonl)

# then on the laptop (no cluster): fit + de-conflation figure from summary.csv
python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py        # reads data/summary.csv
# (self-tests, no data: reduce_fkappa_sweep.py --selftest  · make_fkappa_nH_sweep.py --selftest)
```
Helix conventions baked in (same as II-survey / shellSSC6): `--partition=cpu-single --account=bw22J006
--export=NONE`, `module load devel/miniforge && conda activate trinity`, REPO `/home/hd/hd_hd/hd_cq295/trinity`,
WS `/gpfs/bwfor/work/ws/hd_cq295-trinity`. Validated: `--dry-run` expands to exactly 819 (zero plausibility
warnings); `--emit-jobs` produces a working SLURM array; the diffuse extreme (nCore 1e2) gives rCloud ≈ 39.6 pc
and the whole grid stays < the 200 pc `rCloud_max` ceiling (max is mCloud 1e7 × nCore 1e2 ≈ 70–85 pc). nCore is
**capped at 1e5** on purpose — 1e6 is pathologically stiff/slow (result #15), not f_κ-driven. The harness output
figure overlays the (mCloud, sfe) series: **collapse onto one curve ⇒ f_κ(n_H) is clean; spread ⇒ the
calibration is multi-dimensional.**

## Rebuild all figures (no sims) {#rebuild-all-figures-no-sims}
Every figure is a pure read of a committed CSV, so after a fresh clone you can regenerate the **whole
storyline's figures** without running TRINITY at all:
```bash
cd <repo root>
for h in doublecount_mc fmix_table fmix_spread_plot pdv_regime_table closure_test closure_plots \
         da_screen da_replay theta_density_plot fkappa_definition pdv_trigger_compare \
         ebpeak_trigger_test ebpeak_8config_xcheck fm1_rootcheck fm1b_evapsign \
         kappa_blowout_calibration ideas_comparison; do
  python docs/dev/transition/pdv-trigger/data/make_${h}.py || echo "SKIP $h (needs outputs/kcal — see Block A)"
done
python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py   # rebuild the HTML storyline
```
The ones that need a live run present in `outputs/kcal/` (#11, #13 derivations) will say so; everything else
rebuilds from the committed CSVs.

## Parameter knobs the storyline exercises (all gated, default-off)
| knob | default | sets which result |
|---|---|---|
| `cooling_boost_kappa` (f_κ) | `1.0` | #9, #11, #13 (the El-Badry conduction multiplier) |
| `cooling_boost_mode` / `_fmix` / `_theta` | `none` / `1.0` / `0.0` | #2, #8 (scalar multiplier / Lancaster-θ floor) |
| `transition_trigger` | `cooling_balance` | #13 (`ebpeak` opt-in) |
| `betadelta_solver` | `hybr` | #15 (hybr vs legacy) |

See the **Taxonomy** in `FINDINGS.md` / report §14 for what each knob means physically. **None of these
change a default run** — verified in `PLAN.md` (every experimental knob is off by default).
