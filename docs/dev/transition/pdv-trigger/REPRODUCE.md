# REPRODUCE — paper reproducibility manifest for the `pdv-trigger` workstream

> ---
>
> ⚠️⚠️ **DEMOTED 2026-07-29 — read this as "could be true, verify before use".** The maintainer no
> longer trusts this workstream's measured numbers without re-measurement, and three corrections in
> five days say that is the right call: `§18` (a metric artifact published "f_mix eliminated" for
> eight days across four documents), `§23` (the "wrong El-Badry sign" argument used to retire f_κ was
> false), `§24` (a correct result with a wrong stated cause). None of these were caught by
> `CONTAMINATION.md` — they were **correct data with a wrong reading**, which a per-artifact grade
> cannot detect.
>
> **The active workstream is now [`docs/dev/transition/kappa-3way/`](../kappa-3way/README.md)** — its
> `report.html` is the source of truth, and its rule is: a number is quotable only if its own
> provenance stamp is dated **on or after 2026-07-29**.
>
> **What this doc is still good for:** the history, the physics reasoning, the design rationale, the
> literature imprints (`LANCASTER_REFERENCE.md`, `ELBADRY_REFERENCE.md` — published values, still
> `[V]`), the measurement rules, and the param/HPC tooling under `runs/` (which stays here and is
> actively used). **What it is not good for:** quoting a measured value. Every Θ_cum, band-entry
> dose, spread, fire map and threshold in here is ⚠️ **VERIFY** until the 294-arm re-run reproduces
> it — see [`../kappa-3way/PROVENANCE.md`](../kappa-3way/PROVENANCE.md).
>
> ---


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
| 39 | **f_A Phase 3 gates** — (1) full pytest 742; (2) LITERAL byte-identity: pre-patch worktree vs current at default, +A/A control, identical sha256 dictionary.jsonl; (3) screen reproduces §2; (4) live smoke fA=8 dMdt<fA=1/theta>fA=1 29/29 | `SOURCE_TERM_DESIGN.md` §3 Phase 3, `FINDINGS.md` §15c | `param/simple_cluster.param` (+stop_t 0.03) | `git worktree add <wt> 919feaec`; run pre (in wt) + postA + postB with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `sha256sum */dictionary.jsonl`; screen `python data/make_fA_source_boost.py`; smoke `cooling_boost_fA 8 stop_t 0.03 log_level DEBUG` | local scratch runs (gate, not persisted) | 🟡 |
| 40 | **f_A Phase 4 tooling** — theta5s 81-arm matrix (9 configs x f_A grid), sbatch@6h, sync, analysis (fire map + collapse-law fit p~3.3), dMdt-suppression reducer. Tooling committed & locally validated; **matrix awaiting maintainer HPC run** | `SOURCE_TERM_DESIGN.md` §3 Phase 4, `FINDINGS.md` §15d | `runs/params/theta5s/` (81) | `python runs/make_theta5s_params.py`; `sbatch runs/run_theta5s.sbatch` (maintainer); harvest `runs/harvest_theta_max.py`; `python data/make_theta5s_analysis.py`; `python runs/harvest_dmdt_suppression.py <arms> --csv data/theta5s_dmdt_suppression.csv` | `runs/{make_theta5s_params.py,run_theta5s.sbatch,sync_theta5s.sh,harvest_dmdt_suppression.py}`, `data/make_theta5s_analysis.py` | 🔴 awaiting HPC |
| 41 | **f_A Phase 4 in-container COMPLETE 81/81** (PROVISIONAL, not HPC) — full theta5s matrix run in Claude's ephemeral container (maintainer had no HPC access) over ~11 h across ~dozen restarts. Result: collapse-law **p=3.330 confirms registered p_source≈3.3**; both controls (fail_repro, small_1e6) never fire; 3 classes (normal_n1e3 fires unmodified / 6 configs need f_A, f_fire 4–12 / 2 controls). **ASSUMED — re-run on HPC (row 40) and re-check ALL downstream (§15e mandatory)**. dMdt fidelity (read iii): all 49 quotable ratios <1, falling with f_A — Eq-47 sign matrix-wide | `FINDINGS.md` §15e, `SOURCE_TERM_DESIGN.md` §3 Phase 4 + handoff | `runs/params/theta5s/` (81) | `python runs/run_theta5s_local.py --out $WS/t5s_out --workers 3 --per-arm-timeout 7200 --summary runs/data/theta5s_summary.csv` (+ `runs/autocommit_theta5s.sh $WS/t5s_out`); analyze `python data/make_theta5s_analysis.py`; dMdt `python runs/harvest_dmdt_suppression.py $WS/t5s_out/* --csv data/theta5s_dmdt_suppression.csv` | `runs/{run_theta5s_local.py,autocommit_theta5s.sh,checkpoint_theta5s.py}`, `runs/data/theta5s_summary.csv` (PROVISIONAL header), `data/theta5s_{fire_map,collapse_law,dmdt_suppression}.csv` + `theta5s_{fire_map,theta_rise}.png` | 🟢 in-container 81/81 done, HPC confirmation pending |

| 42 | **f_A Phase 5 pre-step DONE + bench5 params frozen** — L21b Table-1 maintainer-supplied 2026-07-12 and [V]-verified (`LANCASTER_REFERENCE.md §7b`: μ_H=1.4 reproduces R_cl; v_t = α_vir=2 virial velocity ×12 rows; "M_*=5000 fixed" FALSIFIED → ε_* ∈ {0.01,0.1,1}, spec's sfe=0.05 corrected to ε0.1; §7's Eq-10 transcription corrected). 60 params frozen via the EXACT mapping mCloud=M_cl(1+ε), sfe=ε/(1+ε) (5 benches × f_A {1,4,6,8,12,16} × prod/diag arms); emit gates: GMC plausibility + rCloud(gas)=R_cl <2% (the 2% slack is Table-1's own 2.5-pc rounding); end-to-end `read_param` load-check (gas 5.00e5, cluster 5000 exact). **Arms NOT yet run** | `SOURCE_TERM_DESIGN.md` §3 Phase 5, `FINDINGS.md` §15g, `LANCASTER_REFERENCE.md §7b` | `runs/params/bench5/` (60) | `python runs/make_bench5_params.py` (self-gating); later: run arms, harvest `runs/harvest_theta_max.py "$WS"/outputs/bench5/* --csv runs/data/bench5_summary.csv` | `runs/make_bench5_params.py`, `runs/params/bench5/*.param` (60) | 🟡 params frozen; runs pending (in-container vs HPC ruling) |

| 43 | **f_A Phase 5 bench5 in-container run — COMPLETE 60/60, 59 compliant (PROVISIONAL, not HPC)** — maintainer ruled in-container (HPC down 2026-07-12). Campaign via run/harvest/checkpoint_bench5 + autocommit_bench5.sh (adapted from theta5s ops playbook) + send_later heartbeat + hourly cron. Result: 60/60 ran, 59 compliant, 1 dense diag wall-killed (`bench5_fa16_diag`, exit 124, non-critical). FIRE MAP threshold 1→4→12→>16→>16 as n̄ falls 2.28e5→4.42e4→5520→690→43 (bench5 fires UNMODIFIED, bench3 at f_A≥12, bench2/bench1 NOFIRE ≤16). Θ_cum L21b calibration (diagnostic arms, all complete; diffuse benches blow out cleanly = L21b breakout window): bench3 enters band [0.90,0.99] at f_A≈16 (Θ_cum 0.965), bench2/bench1 do NOT reach it even at f_A=16 (max 0.54/0.40) → f_A >16/≫16; dex-vs-El-Badry never below 0.85. RESULT: no single global f_A reproduces L21b across density — required boost climbs steeply toward low density (route-a boundary). Matches registered El-Badry θ_EB (#42-sibling). | `FINDINGS.md` §15h, `SOURCE_TERM_DESIGN.md` §3 Phase 5 + Status/handoff, `data/bench5_{analysis,elbadry_prediction}.csv` | `runs/params/bench5/` (60) | `python runs/run_bench5_local.py --out $WS/bench5_out --workers 3 --per-arm-timeout 7200 --summary runs/data/bench5_summary.csv` (+ `bash runs/autocommit_bench5.sh $WS/bench5_out`); analyze `python data/make_bench5_analysis.py`; El-Badry table `python data/make_bench5_elbadry_prediction.py` | `runs/{run,harvest,checkpoint}_bench5*.py`, `runs/autocommit_bench5.sh`, `runs/data/bench5_summary.csv` + `bench5_traj/` (60), `data/{make_bench5_analysis.py,bench5_analysis.csv,make_bench5_elbadry_prediction.py,bench5_elbadry_prediction.csv}`, `bench5_theta_tracks.png` | 🟢 60/60 done; HPC-confirmed 2026-07-19 (see #45) |
| 44 | **theta5s HPC confirmation (2026-07-19)** — Helix re-run of the 81-arm matrix; harvest replaced `runs/data/theta5s_summary.csv` in place. Analysis reproduces the in-container headlines EXACTLY: p=3.330 (A=1.463, rms 0.0554 dex), whole-band f_A [12,16,24,32], controls never fire, outcomes {FIRED:42, NOFIRE:30, DRAIN:9}. Residue: `data/theta5s_dmdt_suppression.csv` still in-container-derived (re-run reducer on Helix raw arms). | `FINDINGS.md` §15e/§15j | `runs/params/theta5s/` (81) | `./runs/sync_theta5s.sh submit/run/down`; `python data/make_theta5s_analysis.py` | `runs/data/theta5s_summary.csv` (HPC), `data/theta5s_{fire_map,collapse_law}.csv`, `theta5s_{fire_map,theta_rise}.png` | 🟢 HPC-confirmed |
| 45 | **bench5 HPC confirmation + FIRST in-container-vs-HPC fidelity measurement (2026-07-19)** — Helix re-run of the 60-arm Phase-5 matrix. `compare_bench5_hpc.py`: FIDELITY OK — fire map identical (zero flips), 57/60 arms |Δθ_max|<0.05; 3 dense collapse-transient outliers (bench5_fa4_diag 14.35, bench4_fa8_diag 10.34, bench4_fa4 0.39 — excluded from the calibration); fa16_diag freeze REPRODUCED (stiffness, both platforms 59/60). | `FINDINGS.md` §15h/§15j | `runs/params/bench5/` (60) | `./runs/sync_bench.sh bench5 submit/run/down`; `python data/compare_bench5_hpc.py`; `python data/make_bench5_analysis.py` (prefers HPC) | `runs/data/bench5_summary_hpc.csv` + `bench5_traj_hpc/` (60), regenerated `data/bench5_analysis.csv` + `bench5_theta_tracks.png` | 🟢 HPC; authoritative |
| 46 | **bench6 Phase-6 decision matrix (2026-07-19, HPC-only)** — f_A dose extension + f_mix head-to-head, 60 arms. RESULT: f_A band-entry bench3=13.9 / bench2=53.5 / bench1=74.8 (all clean benches reach the L21b band; f_A(n̄)≈315·n̄^−0.335, spread 5.39×; extended fire thresholds 1→4→12→24→64); ~~**f_mix ELIMINATED** — never reaches the band ≤8, wrong-sign dose-response on diffuse benches (bench1 Θ_cum 0.221→0.096 over fm 1→8), fm8 false-fires bench1/2.~~ ⛔ **WITHDRAWN 2026-07-28 (`FINDINGS §18`): metric artifact — the fm numerator omitted the boost. Corrected: monotone rise, bench1 0.221→0.767; band entry bench3 ≈4 measured, bench2/bench1 >8 (extrap ≈8.2/11.9); uniformity spread 2.96× vs f_A's 5.39×. Regenerate with `python data/make_bench6_analysis.py`.** | `FINDINGS.md` §15j (+ §18/§19/§20 corrections), `SOURCE_TERM_DESIGN.md` §3 Phase 6 | `runs/params/bench6/` (60) | `./runs/sync_bench.sh bench6 submit/run/down`; `python data/make_bench6_analysis.py` | `runs/data/bench6_summary.csv` + `bench6_traj/` (60), `data/bench6_analysis.csv` | 🟢 HPC; decision data |

| 47 | **K0 — the f_κ re-read with the "wrong El-Badry sign" argument deleted (2026-07-29, offline)** — Q1: TRINITY's dMdt follows El-Badry Eq 47's C-channel `f^{2/7}` with fitted exponents 0.2819 / 0.2849 vs 0.2857, max error 1.63% / 0.34% over f_κ ∈ [1,64]. Q1b: on a FULL run the same ratio decays −0.12% → −11.30% as E_b drains (per-call ≠ full-run, CLAUDE.md rule 5) — the back-reaction. Q2: the whole-band failure is re-attributed from *reach* to *condensation fallout* — all 6 band configs cross θ=0.95 somewhere; best single dose f_κ=12 fires **5/6**, reproducing §12 exactly. Exposes the gap that f_κ has no L21b band-entry number at all. | `FINDINGS.md` §23/§24, `KAPPA_REOPEN_PLAN.md` §2 | — (reads committed CSVs) | `python data/make_kappa_eq47_check.py` | `data/kappa_eq47_check.csv` | 🟢 |
| 48 | **K1–K4 — the f_κ re-open campaign (174 arms): PARAMS FROZEN, ARMS NOT RUN** — K1 the missing third leg of the band-entry head-to-head (bench1/2/3 × f_κ ∈ {2,3,4,6,8,12,16,24,32} × prod/diag, 54), K1b dense fire-map completeness, f_κ ∈ {2,4,8,12,16} (20), K2 the whole f_κ fire map for the 6 band configs, re-measured + filled in, f_κ ∈ {1,…,16} (66), K3 fate determinism `_a`/`_b` pairs (10), K4 the f_mix **ladder redo** bench1/bench2 × fm {2,3,4,8,12,16} × prod/diag (24). §6.0 ruled 2026-07-29: (a) grid accepted, (b) K1b kept, (c) "no, redo if possible" → the ladder redo, **a reading, not a confirmation** — flip `F_MIX_K4` before submitting if wrong. Predictions P1–P5 + gates G0–G6 pre-registered; **G1 cleared 4/4** (GMC plausibility on all 174 incl. densBE, exact L21b mapping ≤2%, end-to-end `read_param` load-check, count/uniqueness). All five phases share one params dir / one array / one reduce; a phase is a filename prefix. **Under the 2026-07-29 ALL-FRESH ruling this runs alongside `bench5r`/`bench6r` (#50) — 294 arms total — so no bench7 conclusion reads a pre-07-29 CSV.** **§6.2–§6.5 need `ssh helix` — the maintainer's step.** | `KAPPA_REOPEN_PLAN.md` §3–§6, `FINDINGS.md` §25 | `runs/params/bench7/` (174) | `python runs/make_kappa_reopen_params.py` (self-gating); then `./runs/sync_bench.sh bench7 up/submit/watch/reduce/down` | `runs/make_kappa_reopen_params.py`, `runs/params/bench7/*.param` (174), `test/test_bench7_params.py` (182 cases); pending: `runs/data/bench7_{summary,hashes}.csv` + `bench7_traj/` + `data/make_bench7_analysis.py` | 🟡 params frozen; arms 🔴 not run |
| 49 | **Gate G0 — the bench7 pre-HPC baseline check (2026-07-29, offline)** — recomputes Θ₀ and `§18`'s band-entry/spread table **from the 120 committed trajectories** via `make_bench6_analysis`'s own functions (not by re-reading `bench6_analysis.csv`), against tolerances = half the last digit each source quotes. **PASS 11/11**: Θ₀ 0.461806/0.340860/0.220551; f_A entry 13.8834/53.5130/74.8331, spread 5.3901×; f_mix entry 4.00402/8.16293\*/11.8661\* (\* extrapolated), spread 2.96355×. Re-running the three analysis builders leaves their CSVs byte-identical. Also freezes P1's f_κ band-entry prediction (spreads 3.833×/3.427×/2.874× at q = 0.55/0.60/0.70) with `verdict=PENDING` — ⛔ a P1 row is a prediction, never a result. Exits non-zero on any G0 failure. **Doubles as the old-vs-new reproduction gate** (#50): it auto-prefers `bench5r_*`/`bench6r_*` when they land and checks the SAME targets against today's arms, so a PASS then means the 07-19 result reproduced. Names its inputs in a `# SOURCES READ:` header line. | `KAPPA_REOPEN_PLAN.md` §5 (G0) + §3 (P1), `FINDINGS.md` §25 | — (reads committed trajectories) | `python data/make_bench7_gate_g0.py` | `data/make_bench7_gate_g0.py`, `data/bench7_gate_g0.csv` (23 rows) | 🟢 |
| 50 | **The ALL-FRESH re-run — `bench5r` + `bench6r`: TOOLING READY, ARMS NOT RUN** — maintainer ruling 2026-07-29 ("everything I want will be new numerically"). The bench5/bench6 committed params re-run today under fresh landing names (`bench5r_summary.csv` + `bench5r_traj/`, `bench6r_*`), so Θ₀ and the f_A/f_mix ladders are today's numbers and nothing older is overwritten — old-vs-new is a file diff. They also capture bench7's four extra trajectory columns, which the 07-19 harvests never did. Timestamping extended to the **per-arm trajectory CSVs**, `<campaign>_hashes.csv` and the three analysis outputs; the K3 determinism hash now runs over **non-comment lines** so the new stamp cannot fake a P4 failure. `make_bench{5,6}_analysis.py` + `make_bench7_gate_g0.py` auto-prefer the fresh files and print a `# SOURCES READ:` line. **Run order: `KAPPA_REOPEN_PLAN.md §6.2`** (294 arms: bench5r 60 + bench6r 60 + bench7 174). | `KAPPA_REOPEN_PLAN.md` §6.2, `FINDINGS.md` §26 | `runs/params/bench5/` (60) + `runs/params/bench6/` (60), reused unchanged | `./runs/sync_bench.sh bench5r submit/reduce/down`; same for `bench6r`; then `python data/make_bench{5,6}_analysis.py`, `data/make_bench7_gate_g0.py`, `data/make_freshness_audit.py` | `runs/sync_bench.sh` (bench5r/bench6r campaigns), `runs/harvest_bench5.py` (traj stamp); pending: `runs/data/bench{5,6}r_{summary,hashes}.csv` + `bench{5,6}r_traj/` | 🟡 tooling ready; arms 🔴 not run |
| 51 | **Freshness audit (2026-07-29)** — reads every committed CSV's own `# generated …` stamp under `data/` and `runs/data/` and classifies it FRESH / OLD / UNSTAMPED against a cutoff, flagging `+dirty` artifacts as fresh-but-not-reproducible. This is what makes the ALL-FRESH ruling checkable rather than asserted. Baseline before any arm runs: **FRESH 3, OLD 18, UNSTAMPED 262** (the 262 are mostly per-arm trajectory CSVs — the hole the new `harvest_bench5.py` stamp closes on the next reduce). Reports only; never gates. | `FINDINGS.md` §26 | — (reads committed CSVs) | `python data/make_freshness_audit.py [YYYY-MM-DD]` | `data/make_freshness_audit.py`, `data/freshness_audit.csv` (283 rows) | 🟢 |

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
