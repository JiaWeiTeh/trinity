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
| 2 | Trigger-convention fix; no constant `f_mix` fires across density | §3 | — (reads frozen) | `python data/make_fmix_table.py` | `data/fmix_table.csv`, `fmix_vs_density.png` | 🟢 |
| 3 | PdV regime split (sub- vs super-critical) | §4 | — (reads frozen) | `python data/make_pdv_regime_table.py` | `data/pdv_regime_budget.csv` | 🟢 |
| 4 | A constant knob is not enough (8-config staged screen) | §5 | — (reads frozen) | `python data/make_closure_test.py && python data/make_closure_plots.py` | `data/closure_test.csv`, `closure_stage*.png` | 🟢 |
| 5 | θ_target(Da) **refuted** (offline proxy) | §6 | — (reads frozen) | `python data/make_da_screen.py` | `data/da_screen.csv`, `da_screen.png` | 🟢 |
| 6 | θ_target(Da) refuted (gate-validated real-Da replay) | §6 | — (replays frozen) | `python data/make_da_replay.py` | `data/da_replay.csv`, `da_replay.png` | 🟢 |
| 7 | Literature anchor θ(n_H) vs TRINITY's resolved loss | §7 | — (reads frozen) | `python data/make_theta_density_plot.py` | `theta_vs_density.png` | 🟢 |
| 8 | Live matched-`t` edge runs (boost vs none) | §9 | `f1edge_{hidens,lowdens}__*`, `simple_cluster__*`, `fail_repro__*` | see [`runs/README.md`](runs/README.md) | `runs/data/live_compare.csv` | 🟡 |
| 9 | **κ_eff Rung A** raises emergent cooling (back-reaction) | §11 | `f1edge_hidens__none.param`, `f1edge_hidens__kappa2.param` | run both (separate processes) → `make_kappa_backreaction.py` | `data/kappa_backreaction.csv`, `kappa_backreaction.png` | 🟡 |
| 10 | **What f_κ is** (Spitzer multiplier; seed law verified) | §13 | — (reads #9's CSV) | `python data/make_fkappa_definition.py` | `fkappa_definition.png` | 🟢 |
| 11 | **f_κ calibration** θ→0.95 (compact/mid/diffuse) | §13 | `cal_{compact,diffuse}__k{1,2,4}.param`, `cal_mid__ek{1,2,4}.param` | run the 9 → `make_kappa_blowout_calibration.py` | `data/kappa_blowout_calibration.csv`, `kappa_blowout_calibration.png` | 🔴 |
| 12 | PdV is the dominant sink (cool-only vs PdV-incl) | §12 | reuses #11's `cal_*__k{1,2,4}` | `python data/make_pdv_trigger_compare.py` | `data/pdv_trigger_compare.csv`, `pdv_trigger_compare.png` | 🟢¹ |
| 13 | **ebpeak does not fire at f_κ=1** (code-path test) | §12 | `cal_{compact,diffuse}__ebpeak.param`, `cal_mid__ek{1,2,4}.param` (+ #11) | run the ebpeak ones → `make_ebpeak_trigger_test.py` | `data/ebpeak_trigger_test.csv`, `ebpeak_trigger_test.png` | 🔴 |
| 14 | **Holds across 8 configs** (frozen + live overlay) | §12 | — (reads frozen + #13's CSV) | `python data/make_ebpeak_8config_xcheck.py` | `data/ebpeak_8config_xcheck.csv`, `ebpeak_8config_xcheck.png` | 🟢 |
| 15 | Dense-edge stiffness is **not** f_κ (it's extreme density) | PLAN ledger 06-28 | `diag_dense_hybr.param`, `diag_dense_legacy.param` | run both, observe (does not finish at nCore 1e6) | `data/dense_stiffness_diag.csv` | 🟡 |
| 16 | FM1 / FM1b — wrong knobs ruled out (κ_eff confirmed) | §11 | — (offline prototypes) | `python data/make_fm1_rootcheck.py`; `python data/make_fm1b_evapsign.py` | `data/fm1*.csv`, `fm1*.png` | 🟢 |
| 17 | All-ideas scoreboard | hero | — (reads CSVs above) | `python data/make_ideas_comparison.py` | `ideas_comparison.png` | 🟢 |
| 18 | **Controlled f_κ(n_H) calibration** (clean single-variable, *ready/not-yet-run*) | (next) | `runs/params/sweep_fkappa_nH.param` (sweep → 28 combos) | `--emit-jobs` → `sbatch` → `make_fkappa_nH_sweep.py` | `data/fkappa_nH_sweep.csv`, `fkappa_nH_sweep.png` | 🔴 |

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
`run.py <sweep.param> --emit-jobs jobs/` then `sbatch` (see `python run.py -h`).

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

### Block C — controlled f_κ(n_H) calibration sweep (result #18; HPC, ready, not yet run)
The clean replacement for the conflated 3-anchor estimate: **fix mCloud + sfe, vary only nCore** × a wide
f_κ grid that brackets the firing point at every density. 28 combos; emit as a SLURM array, then harvest.
```bash
# 1. inspect / emit the grid (the .param uses TRINITY's list sweep syntax: nCore [..] x cooling_boost_kappa [..])
python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --dry-run     # lists 28 combos
python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --emit-jobs jobs/
# 2. on the cluster: edit jobs/submit_sweep.sbatch (#SBATCH --account/--partition), then
sbatch jobs/submit_sweep.sbatch               # -> outputs/sweep_fkappa_nH/<run>/
# 3. harvest theta_blowout(nCore, f_kappa), fit f_kappa_fire(nCore):
python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py
# (parser self-test only, no data needed: ... make_fkappa_nH_sweep.py --selftest)
```
Validated: the diffuse extreme (nCore 1e2) gives rCloud ≈ 39.6 pc (< the 200 pc `rCloud_max` ceiling), so all
28 configs are physically plausible. nCore is **capped at 1e5** on purpose — 1e6 is pathologically stiff/slow
(result #15), not f_κ-driven.

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
