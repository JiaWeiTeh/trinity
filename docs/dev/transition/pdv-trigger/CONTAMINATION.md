# CONTAMINATION REGISTER — what you may and may not quote from this workstream

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
> several living docs for its workstream (`INDEX.md`, `PLAN.md`, `FINDINGS.md`, `F_KAPPA_FUNCTIONAL_FORM.md`,
> `REPRODUCE.md`, `runs/README.md`, and the other notes in this folder). They drift out of sync *with each
> other* as fast as they drift from the code. Any agent or person editing one MUST, as part of the visit,
> circle back through the siblings and reconcile: if a number, status, claim, or line reference here
> contradicts a sibling — or a sibling has gone stale — fix it (or flag it, dated) so no two docs in the
> workstream disagree. Never update one in isolation.

*Created 2026-07-01, when the pt2 line (merged via PR #717) and the parallel pt3 line
(`feature/transition-trigger-pt3`) were reconciled. Full audit provenance: four independent
re-reads of the docs, the code, and every committed artifact on that date.*

**Version/recency is a separate axis:** this register grades *quotability*; **`MANIFEST.md`** (generated,
`python make_manifest.py`) grades *currency* — per artifact, when it was last regenerated, by which script,
and whether its builder has changed since (⚠️ STALE-RISK). Check both before using any number.

## The rules that define "contaminated"

A result is **CONTAMINATED** if it violates any of (maintainer-authoritative; 📏 boxes in `PLAN.md`,
protocol in `runs/README.md`):

| rule | requirement | violation tag |
|---|---|---|
| **(a)** | every test run reaches **≥5 Myr** (or its natural physics end: recollapse, large_radius) | `<5Myr` |
| **(b)** | θ is reported as **θ_max over the run**, never at blowout | `blowout-θ` |
| **(c)** | θ is harvested from `dictionary.jsonl` accepted rows (`bubble_Lloss/Lmech_total`), never a call-level observer (solver trial (β,δ) points contaminate those — retraction **R6**, bogus θ_max=3.22) | `observer` |
| **(d)** | a calibration fit with one knob is only validated with the **same knob** (`cooling_boost_kappa` ≠ `multiplier` — retraction **R5**) | `knob-mismatch` |
| **(e)** | massive-cloud **fates** from **pre-PR#715** code (dead-stop on Eb≤0) are not quotable as physics | `pre-#715` |

Timeline of rule adoption (everything earlier is *retroactively* graded, i.e. flagged, not deleted):
PR #715 merged 2026-07-01 ~13:04 · rule (a) 2026-06-30 · rule (b) 2026-07-01 ~15:39 (`f125de6`) ·
rules (c)/(d) later on 2026-07-01 (FINDINGS §8d/§8e). The pt3-branch sweep work (2026-06-29 →
folded 2026-07-01) predates all of them.

## ⛔ DO NOT QUOTE (the headline list)

1. **F_KAPPA §14 "live validation" θ_max = 1.334 / 1.006 → "fires YES"** — `knob-mismatch` (fit on
   kappa, run with multiplier, R5) **and** `observer` (R6). Re-harvest from `dictionary.jsonl` before
   quoting anything from those runs.
2. **F_KAPPA §14 calibration table (θ₀, p, f_κ_ideal, θ@f_max, n_routeA boundaries)** — `blowout-θ`
   (θ₀ is a blowout snapshot) + fit on `cooling_boost_kappa` (does not carry over to `multiplier`,
   FINDINGS §8e) + the 6-anchor θ₀ slope (0.41/dex) was **falsified by the 819-sweep** (real 1.13/dex,
   PLAN scorecard P3 ❌). The *shape* of the argument survives; **no number in that table is production-grade**.
   Re-derived (2026-07-02) → `FINDINGS.md §10` / `runs/data/theta5_calibration.csv`.
3. **f_κ-to-fire ≈ 4 / 5–6 / 60 and the closed form f_κ(n_H) ≈ 1.4×10²·n^−0.30** (FINDINGS §6 + the
   06-29 "Calibration target" banner; `data/kappa_blowout_calibration.csv`,
   `data/kappa_calibration_estimate.csv`, `data/fkappa_functional_form.csv`) — `blowout-θ` + `<5Myr`
   (cal runs stop_t 0.3–1.0) + slope refuted by the sweep (−0.60) + superseded by the single-constant
   DECISION. Empirically refuted by theta5 (2026-07-02): diffuse f_fire = **4**, not 60 (the blowout metric
   had under-read diffuse θ by 2× — `FINDINGS.md §10`).
4. **The 63-cell `f_κ_fire` grid and everything derived from it** (`data/fkappa_nH_sweep.csv`,
   `data/summary.csv`, the θ₁-collapse fit `data/fkappa_theta1_collapse.csv`, and the shipped
   **`cooling_boost_kappa='auto'` lookup table** in `trinity/_input/fkappa_auto.py`) — `<5Myr`
   (sweep `stop_t=2`) and "fired-by-2-Myr" is not θ_max (rule b). Status: **PROVISIONAL, opt-in**,
   pending re-measurement under the standard protocol. The *de-conflation verdict* (f_κ ≠ f(n_H) alone)
   and the *θ₁-collapse law shape* are qualitative findings that likely survive; the grid **values** are not
   production-grade.
5. **All pre-PR#715 `fail_repro`/massive-cloud FATES** — `pre-#715`: FINDINGS §8/§8a
   SHELL_COLLAPSED conclusions (🛑-bannered in-doc), `data/shadow_te_fate.csv` fail_repro row,
   `data/sweep_tmax_fate.csv`, `runs/data/harvest_fail_repro__*.csv`, `runs/data/live_compare.csv`
   fail_repro row, `runs/README.md`'s "heavy clouds collapse regardless" verdict. The post-#715 record
   is `data/newcode_default_vs_theta.csv`.
6. **Any θ from the `_fkappa_validation_runner.py` observer** — `observer` (R6). The runner is kept
   as history; its θ numbers are void.

## Full artifact register

Eras: **E1** PdV/f_mix frozen-screen + live edge (06-24→25) · **E2** κ_eff Rung-A/FM probes (06-26→27) ·
**E3** kappa blowout-cal + ebpeak (06-28) · **E4** 819 f_κ(n_H) sweep (06-29, folded 07-01 from pt3) ·
**E5** κ_mix Rung-B (06-29→30, SHELVED) · **E6** impose-El-Badry-θ detour (06-30→07-01, DEMOTED) ·
**E7** PR#715 + direction/knob/θ_max corrections (07-01).

Status legend: **CLEAN** (quotable for its stated question) · **FLAG-x** (usable with the named caveat) ·
**CONTAMINATED-x** (do not quote numbers) · **SUPERSEDED** (kept as history; direction/idea retired).

### `data/`

| artifact | era | knob | status |
|---|---|---|---|
| `fmix_table.csv`, `pdv_combined_trigger.csv`, `closure_test.csv` | E1 | screen (post-hoc mult/θ) | FLAG-(b): blowout-referenced frozen screens — bounds, not forecasts |
| `pdv_regime_budget.csv` | E1 | — | FLAG: frozen, unknown provenance; trust `live_pdv_decomp.csv` where they differ |
| `doublecount_mc.csv` | E1 | — | CLEAN (pure MC) |
| `da_screen.csv`, `da_replay.csv` | E1/E2 | theta_target(Da) | CLEAN as the θ(Da) **refutation**; FLAG-(b) |
| `kappa_backreaction.csv` | E2 | kappa 2 | CLEAN for the f_κ^(2/7) scaling check; FLAG-(a) (t≲0.17) |
| `fkappa_leverage.csv`, `kappa_calibration_estimate.csv` | E2 | kappa (snapshots) | SUPERSEDED (p=0.63 "optimistic" refuted by sweep p≈0.27) |
| `fm1_rootcheck.csv`, `fm1b_evapsign.csv` | E2 | Rung-B probes | CLEAN (negative results; Rung B shelved anyway) |
| `kappa_blowout_calibration.csv` | E3 | kappa 1/2/4 | **CONTAMINATED-(a)+(b)** (stop_t 0.3–1.0, blowout headline) |
| `ebpeak_trigger_test.csv` | E3 | ebpeak | FLAG-(a); finding (never fires at f_κ=1) reconfirmed later |
| `ebpeak_8config_xcheck.csv` | E3 | ebpeak, frozen | CLEAN for "ebpeak doesn't fire"; fail_repro row `pre-#715` |
| `dense_stiffness_diag.csv` | E3 | solver diag | CLEAN (diagnostic); orphan (no script) |
| `summary.csv`, `sweep_report.txt`, `fkappa_nH_sweep.csv`, `fkappa_sweep_scorecard.csv`, `fkappa_cliff_metric.csv`, `fkappa_physical_cap.csv`, `fkappa_theta1_collapse.csv` | E4 | kappa 1–64 | **CONTAMINATED-(a)+(b)** as *calibration values* (stop_t=2, fired-by-2-Myr); de-conflation verdict + θ₁-collapse *shape* + cliff/column *reasoning* survive qualitatively |
| `fkappa_functional_form.csv` | E4 | kappa | **CONTAMINATED-(a)+(b)**; also superseded by single-constant DECISION |
| `fkappa_physical_derivation.csv` | E5 | derivation | CLEAN as analytics; its route-b vehicle (κ_mix) is shelved |
| `kmix_prototype.csv` | E5 | κ_mix offline | **CONTAMINATED-(a)** (Pb anchors from 0.3–1.0 Myr runs) — GO verdict qualitative only |
| `kmix_selfconsistent.csv`, `kmix_theta_trajectory{,_summary}.csv` | E5 | κ_mix patched | SUPERSEDED/SHELVED (+ known kprime −1/T bug, early rows unsolved, "dense low" walked back as port artifact) |
| `elbadry_theta.csv`, `nmap_verify.csv` | E6 | analytic | CLEAN |
| `elbadry_overlay.csv` (cherry-picked from pt1 `3e68143`, 07-01) | E6 | analytic band + resolved θ_1D | band CLEAN (Eq 37/38 PDF-verified; extrapolated beyond n≈10); TRINITY points FLAG-(b) (blowout-epoch θ_1D — θ_max now available (2026-07-02, `runs/data/theta5_summary.csv`), so the re-plot is actionable) |
| `shadow_te_fate.csv` | E6 | imposed θ | SUPERSEDED (impose-θ demoted); trigger-algebra rows OK; fail_repro row `pre-#715`; FLAG-(c) (call-level diag) |
| `sweep_tmax_fate.csv` | E6 | imposed θ | SUPERSEDED + `pre-#715`; orphan (no committed harvester) |
| `live_pdv_decomp.csv` | E7 | none | CLEAN as the dead-stop **bug evidence**; not a fate record |
| `newcode_default_vs_theta.csv` | E7 | default vs imposed θ | **CLEAN — the canonical post-#715 massive-cloud fate record**; orphan (hand-assembled) |
| `gate_prototype.csv` | E7 | gated θ | CLEAN (post-#715 prototype; direction demoted regardless) |
| `fkappa_emergent_calibration.csv` | E7 | prescribes **multiplier** from **kappa**-fit θ₀/p at **blowout** | **CONTAMINATED-(b)+(d)** — committed 33 min before rule (b); superseded by the re-derivation RESULT (§10, 2026-07-02) |
| §8e θ numbers (0.25/0.48/0.53) | E7 | kappa 2/8 | FLAG-(a) (early-time, honestly labelled) + **never persisted as CSV** (💾 violation — FINDINGS-text only) |
| `fkappa_auto_verify.csv` (ran 07-01) | E7 | kappa='auto' (→12) | CLEAN as the **mechanics acceptance** (auto resolves, fires, reaches momentum; θ from dictionary); FLAG-(a) for calibration use (stop_t=2 — inherits the grid's provisional status) |
| `kappa_stability_map.csv` (built 07-01) | E7 | — (re-read of `summary.csv`) | CLEAN re-analysis; resolves ⚡ #1 (FINDINGS §9a) |
| `theta5_*.png` (5 figures) + `runs/data/theta5_fmix_scorecard.csv` (built 07-02, `data/make_theta5_figures.py`) | E8 | — (re-reads of the theta5 CSVs + `summary.csv` + `fkappa_functional_form.csv`) | CLEAN re-analysis (REPRODUCE #29); the scorecard is the f_mix margin table — **f_mix=4 ADOPTED 2026-07-02** (maintainer ruling: momentum-then-recollapse is fine; PLAN ledger); F3 deliberately plots the ⛔ #3 blowout values AS the retired metric being corrected |
| `theta5b_summary.csv` + `theta5b_calibration.csv` (RAN Helix 2026-07-02) | E8 | multiplier fine {2.5,3,3.5,4.5,5} + diffuse stop_t=8 | ✅ CLEAN rule-compliant (43/43 to stop_t or physics end; θ_max from dictionary; stop_t=8 arms EXCEED rule (a)). NB the theta5b-only calibration CSV lacks θ₀ (no f=1 arms) — use the combined analysis |
| `theta5_fire_map.csv`, `theta5_law_check.csv`, `theta5b_{fire_map,law_check}.png` (built 07-02, `data/make_theta5b_analysis.py`) | E8 | — (re-reads both summaries) | CLEAN re-analysis (REPRODUCE #30): window [4,4.5], law rms 0.064 dex, DRAIN class = momentum-without-firing (do NOT count DRAIN as a θ transition) |
| `kappa_freeze_autopsy.csv` (built 07-02, `data/make_kappa_freeze_autopsy.py`) | E8 | — (re-read of `summary.csv`) | CLEAN re-analysis (FINDINGS §9b): θ quoted only to *classify failure modes*, never as calibration (underlying sweep is stop_t=2, FLAG-(a)); supersedes §9a's mechanism claim |
| local freeze-repro runs (07-02, session scratchpad `kappa_repro/`, simple_cluster f_κ∈{1,4,7.5,8,16} + legacy solver + MAX_SEGMENTS=40 driver) | E8 | kappa 1–16 | ⛔ **mechanism diagnosis ONLY** — never quote θ from these (short horizons, monkeypatched driver, local wall-limits); durable record = KAPPA_FREEZE_MECHANISM §4 table |
| `theta5k_summary.csv` (RAN Helix 2026-07-03, post fix #1) | E8 | kappa 1–16 | ✅ CLEAN rule-compliant for FIRE/NO-FIRE classification (56/56 proper fates, stop_t=5, θ_max from dictionary). ⚠️ do NOT quote fired-arm θ_max magnitudes above ~1.2 (structural-boost distortion, e.g. dense 1.99); CONDENSE/DRAIN are fates, never θ transitions |
| `theta5k_fire_map.csv` + `theta5k_{fire_map,theta_rise}.png` (built 07-03, `data/make_theta5k_analysis.py`) | E8 | — (re-read of `theta5k_summary.csv`) | CLEAN re-analysis (FINDINGS §12): no whole-band f_κ; CONDENSE detected via n_impl==50 streak-cap heuristic |
| ⚡ small_1e6 ⇄ large_diffuse early-time degeneracy (found 07-03, FINDINGS §12.6) | — | — | both configs have M_cluster=1e5, nCore=1e2, flat profile → **bit-identical early trajectories** (theta_first identical in all theta5k arm pairs). NOT a harvest bug — but for any early-time (t ≲ 1 Myr) claim they count as ONE check, not two |
| `dmdt_trace_dense.csv` + `dmdt_dip_traces.png` + `dmdt_tackle_flow.png` (built 07-03, `data/make_dmdt_dip_figures.py`; params `runs/params/dmdt_trace/`) | E8 | kappa 6/8, stop_t=0.08 local | ⛔ **mechanism diagnosis ONLY** — per-segment eigenvalue traces; never quote θ or timings from these short local runs as calibration |
| `theta5n_summary.csv` (RAN Helix 2026-07-03) | E8 | multiplier 2–8 + kappa 2–16 + none | ✅ CLEAN rule-compliant (15/15 proper fates, stop_t=5, θ_max from dictionary). The none arm FIRES natively (θ₀=1.047) — quote as route-a evidence; kappa-16 DRAIN is a fate, not a θ transition |
| ⛔ all-NaN θ arms (theta5 dense mult4/mult8 pattern) | — | any | **never a physics outcome**: NaN = `bubble_Lloss` registry default, written because the β–δ solve never succeeded (root at the integrable-domain edge, machine-flippable — FINDINGS §14). Use the finite-θ neighboring arms; never average or interpret NaN arms |

### `runs/data/`

| artifact | status |
|---|---|
| `harvest_simple_cluster__none.csv` | CLEAN (t→10.4 Myr) |
| `harvest_simple_cluster__mult2.csv` | FLAG-(a): ends at handoff t≈0.13 (physics end — fired; θ_max 0.967) |
| `harvest_f1edge_lowdens__{none,mult2,mult3}.csv` | **CONTAMINATED-(a)** (1200 s wall truncation at t≈3.0–3.3) |
| `harvest_f1edge_hidens__{none,mult2}.csv` | FLAG-(a) (birth-fire/handoff-end; hidens mult2 = 1 row) |
| `harvest_fail_repro__{none,mult2}.csv`, `live_compare.csv` fail_repro row | **CONTAMINATED-(e)** |
| `compare_f1edge_hidens_theta9{0,5}.csv` | SUPERSEDED (theta_target demoted) + FLAG-(a) |
| `harvest_cal_*__{k1,ek1}.csv` | **CONTAMINATED-(a)** (stop_t 0.3–1.0 by design) |
| `theta5_summary.csv` + `theta5_calibration.csv` (RAN Helix 2026-07-02) | ✅ **CLEAN — the first fully rule-compliant calibration** (32/32 arms ≥5 Myr or physics end; θ_max from dictionary; same-knob fit+validation; stamped). Two per-arm flags: `small_dense_highsfe__mult{4,8}` carry **NaN loss rows** (dense-edge stiffness — excluded from the fit); `midrange_pl0__mult8` reached momentum via the **Eb≤0 handoff, NOT a cooling fire** (`fired=False` is correct — do not count it as a θ transition). Stamp caveat: `theta5_summary.csv`'s stamp reads **`code 23f623d3+dirty`** — harvested on Helix from a dirty tree (the tree carried only the being-created CSVs, no code edits), noted per the `_stamp.py` doctrine |

### Production code (on this branch)

| item | status |
|---|---|
| `cooling_boost_mode` = `none` (default) / `multiplier` / `theta_target` | shipped, gated, default byte-identical; `theta_target` = documented opt-in override (demoted direction) |
| `cooling_boost_kappa` (numeric) | shipped, gated (×1.0 exact); **structural probe only** — breaks at f_κ=8 (§8e), raises evaporation |
| `cooling_boost_kappa = 'auto'` (pt3) | shipped, gated, opt-in; **PROVISIONAL** — lookup grid is E4-contaminated (see ⛔ #4); revalidate under the standard protocol before relying on it |
| `theta_elbadry` mode | **never merged** — docs/harness only (`THETA_ELBADRY_SPEC.md`) |
| PR #715 Eb≤0→momentum routing; Pb-collapse guard; `_MINT_LOG_TOL` log gate | shipped, behavior-verified |

## ⚡ Open + resolved tensions (while open, neither side is quotable as refuting the other)

1. **FINDINGS §8e vs §9 on `cooling_boost_kappa` — ✅ RESOLVED (2026-07-01, from committed data;
   `FINDINGS.md §9a`, `data/kappa_stability_map.csv`).** Both were right: the knob's breakdown is
   **non-monotonic in f_κ** (firing bands interleave with freeze windows; 17/57 cells non-monotonic,
   38/819 runs froze mid-implicit). The sweep's simple_cluster-analog cell freezes at f_κ=8/12 with
   θ_max=0.5331 — §8e's ~0.53, reproduced on Helix. Solver choice is ruled out (both ran default hybr).
   Fallout: kappa even less shippable; `'auto'` gains an interpolation-into-a-dead-window caveat.
2. **`'auto'` (fire everything, per-cloud f_κ) vs the maintainer decisions** "single physical f_κ
   constant" + route-a ("diffuse clouds may never enter momentum"). 'auto' stands as an opt-in
   convenience, not the production direction.
3. **θ-peak epoch (updated 2026-07-02):** config-dependent, and the diffuse peak sits at the very **EDGE**
   of the 5 Myr window. theta5 (`FINDINGS.md §10`) measured the diffuse baseline peak at **t≈4.9 Myr**
   (large_diffuse θ₀=0.535), and `large_diffuse_lowsfe__mult2` **grazes 0.9552 exactly at stop_t=5** — so
   the 5 Myr window may NOT fully cover the diffuse peak. Earlier anchors (PLAN 📏 rule 1 "~0.4–1 Myr";
   SESSION_HANDOFF §5.2 "~blowout"; measured compact 0.912 @ t≈0.12, diffuse 0.862 @ t≈1.06) stand as
   history. **Open residue: the stop_t=8 diffuse spot-check**, which would bracket the diffuse f_fire
   between 2 and 4.
