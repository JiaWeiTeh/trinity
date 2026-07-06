# Verification ledger 01 — insights_betadelta_illustrated.html

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time
> audit, not a maintained spec; re-check each claim against current source.
>
> 🔄 **Living ledger — recheck and refine on every visit.** Re-run the verdicts when you
> touch the relevant code; tick the fix boxes as they land.
>
> 💾 **Persist diagnostics — commit, don't re-run.** The per-report ledgers (01–05) carry the
> `file:line` evidence so a future visit need not re-derive it.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

Verified 2026-06-22 on branch `claude/exciting-gates-mkxqn6`.

---

## SUMMARY

**Total claims checked:** 46  
**✅ Correct:** 32  
**⚠️ Stale reference:** 6  
**❌ Wrong:** 5  
**❓ Unverifiable:** 3  

**Top fixes needed before this content enters a published storyline book:**

1. **❌ `betadelta_solver` default** (§4, §5, §13): report says "default still legacy" — the current default is `hybr` (default.param:49, registry.py:307). This was flipped after the report was written. Every statement framing hybr as "behind a switch with legacy default" is false in the current tree.
2. **❌ `bubble_luminosity.py:1150`** (§6, §7 problem statement): that line does not exist — the file is 1083 lines. The velocity ODE (`dvdr` term, the `(β+δ)/t` source) is at `bubble_luminosity.py:411`. Must be corrected before any reader tries to verify it.
3. **❌ `bubble_luminosity.py:612/659/677`** (§7 "Problem 2", §11, §13): the cooling integrals `L_bubble`/`L_conduction`/`L_intermediate` are at lines ~698, ~745, ~785 in the current file. Lines 612/659/677 contain unrelated array-processing code. The stale numbers originate in `stalling-energy-phase.md`; both that doc and the HTML report need updating.
4. **❌ β range for `steep·hybr`** (§5 table, line "−2.44→2.82"): the Phase-3 `steep·hybr` 3 Myr run has β∈[0.59, 2.82] (PHASE2_ARMS.md:198); β_min = −2.44 belongs to `sweep_steep` (4 Myr, PHASE2_ARMS.md:223). The two runs are conflated in the table row.
5. **⚠️ File-path glossary** (§14 "Files & glossary"): four paths are wrong — see stale-ref entries below. `scratch/phase2/` and `scratch/phase6/` do not exist in the committed tree; those files live under `docs/dev/archive/betadelta/diagnostics/` and `docs/dev/archive/betadelta/velstruct/` respectively.

---

## Claim-by-claim table

### TL;DR & timeline (§TL;DR)

| Claim (short quote) | Report § | Current-source evidence (file:line / data file) | Verdict | Minimal fix |
|---|---|---|---|---|
| "0→100% solver convergence (legacy→hybr)" | TL;DR | PHASE2_ARMS.md:196–204; sweep rows all 100% | ✅ | — |
| "909 segments swept · 100% converged" | TL;DR | `docs/dev/data/hunt_h*.csv`: 915 lines − 6 headers = 909 segments | ✅ | — |
| "≤0.04% macro impact of deleting the inflow" | TL;DR | stalling-energy-phase.md:328–333 counterfactual table | ✅ | — |
| "M≈1.8e-3 inflow Mach" | TL;DR | stalling-energy-phase.md:179 (`v_min`=−0.62/c_s≈338 → 0.0018) | ✅ | — |
| "Phase 0 · 47.5% unconverged on a worked example" | Timeline | PHASE0_BASELINES.md:88 (52.5% converged → 47.5% unconverged) | ✅ | — |
| "mock 16/50, simple 10/61, steep 8/56 segments" wrong-signed | Timeline | PHASE0_BASELINES.md:65–68 | ✅ | — |
| "hybr shipped behind betadelta_solver (default still legacy)" | Timeline §4 | registry.py:307 default='hybr'; default.param:49 `betadelta_solver hybr` | ❌ | Change "default still legacy" to "default flipped to hybr" everywhere. |

### §1 — Two state variables β, δ

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "β — get_betadelta.py:224 (β), :272 (δ→dT/dt)" | §1 | `cool_beta_to_Ebdot_pure` starts at :182; `delta2dTdt_pure` starts at :272 | ⚠️ | Fix :224 → :182; :272 is correct. |
| "self-consistent runs span β ∈ [−2.44, +4.23], δ ∈ [−1.5, +2]" | §1 | PHASE2_ARMS.md sweep_steep/mock rows; sweep_steep β=[−2.44,3.43] across 4-Myr run | ✅ | — |
| "legacy clamped both into a small box" | §1 | get_betadelta.py:41–44 (BETA_MIN=0.0, BETA_MAX=1.0, DELTA_MIN=−1.0, DELTA_MAX=0.0) | ✅ | — |

### §2 — The self-consistency residual

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "Edot_from_balance — get_betadelta.py:434" | §2 | get_betadelta.py:434 exactly matches | ✅ | — |
| "g-metric denominator L_mech,total — get_betadelta.py:408, arms.py:103" | §2 | get_betadelta.py:408 `Lmech_total = params[...]`; arms.py:82 `self.Lm`; arms.py:103 `gE = ... / self.Lm` | ✅ | — |
| "pole-not-operative — PHASE0_BETADELTA_BASELINES.md (Finding 4)" | §2 | PHASE0_BASELINES.md:103 Finding 4 exists and matches | ✅ | — |
| "f_E has a pole near E_b peak where E˙_{b,β} crosses zero" | §2 | get_betadelta.py:437–440 shows division by Edot_from_beta; the pole is real | ✅ | — |

### §3 — The three defects

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "GRID_EPSILON=0.02 — get_betadelta.py:57, 969–972" | §3 ① | get_betadelta.py:57 (GRID_EPSILON=0.02), :969–972 (usage) | ✅ | — |
| "mock 19 consecutive segments at ±0.02, 0/27 converged" | §3 ① | PHASE0_BASELINES.md, PHASE2_ARMS.md:72 (0/27, A control), HYBR_PLAN.md:62 | ✅ | — |
| "bounds — get_betadelta.py:41–44" | §3 ② | get_betadelta.py:41–44 | ✅ | — |
| "β_adia — BETADELTA_HYBR_PLAN.md:63" | §3 ② | HYBR_PLAN.md line 63 has mock trace prose, not β_adia; β_adia formula is at HYBR_PLAN.md:88 | ⚠️ | Fix :63 → :88. |
| "steep pinning β at BETA_MAX=1 for 10 consecutive segments" | §3 ② | PHASE0_BASELINES.md:68 (14 β=1 hits incl 10 consecutive), HYBR_PLAN.md:97 | ✅ | — |
| "19 of 21 converged hybr roots on mock outside box" | §3 ② | PHASE2_ARMS.md:93 (19 of 21 out-of-box) | ✅ | — |
| "BETADELTA_HYBR_PLAN.md:52, 49–58" for predictor/consistency relation | §3 + §4 foot | HYBR_PLAN.md:52 = Evidence 2 (baseline table); predictor is at :74–83 | ❌ | Fix BETADELTA_HYBR_PLAN.md:52, 49–58 → :74, 74–83. |
| "consistency relation misses solved δ by 0.05–0.14" | §3 | PHASE0_BASELINES.md:122; HYBR_PLAN.md:80–81 | ✅ | — |

### §4 — Four-arm bake-off

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "arm A 0% mock; arm D hybr 78% mock, 80% simple" | §4 | PHASE2_ARMS.md:72–84 | ✅ | — |
| "arms — arms.py:144–196" | §4 | arms.py:144=`_arm_B`; :155=`_arm_C`; :184=`_arm_D`; :196=last line of `_arm_D`; range 144–196 covers B/C/D definitions | ✅ | — |
| "hybr opts (xtol=1e-8, maxfev=30, eps=3e-4) — get_betadelta.py:74" | §4 | get_betadelta.py:74 `HYBR_OPTIONS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4)` | ✅ | — |
| "Gate G2 — BETADELTA_PHASE2_ARMS.md:107–125" | §4 | PHASE2_ARMS.md:107 = Finding 2 middle; Gate G2 section starts at :132; lines 107–125 cover Findings 2–5 | ⚠️ | Fix :107–125 → :132–149 for the Gate G2 section. |
| "Maintainer green-lit widening the bounds; hybr shipped behind betadelta_solver (default still legacy)" | §4 | registry.py:307 default='hybr' | ❌ | Change "default still legacy" to "default hybr". |
| "D promotes (78-pt margin on mock; 20 pts and ~3.7× fewer evals on simple1e5)" | §4 | PHASE2_ARMS.md:148 | ✅ | — |

### §5 — hybr vs legacy (Phase 3)

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "flat (1e6, n=1e5) legacy 0% (0/74), hybr 100% (90/90) →1.63 crosses 0.05 at 0.247 Myr" | §5 | PHASE2_ARMS.md:196–197, 253 | ✅ | — |
| "steep (1e6, α=−2) hybr 100% (113/113) −2.44→2.82 stalls at ratio ≈0.35" | §5 | PHASE2_ARMS.md:198: β_min=0.59, β_max=2.82; −2.44 belongs to sweep_steep (4 Myr run, :223) | ❌ | Change "−2.44→2.82" to "0.59→2.82" for the 3-Myr Phase-3 run; note β reaches −2.44 only in the 4-Myr sweep. |
| "typical (1e6, n=1e3) — 100% (151/151) →4.18 crosses ~2.5 Myr" | §5 | PHASE2_ARMS.md:200 (β_max=4.18, 151/151, t_trans ~2.5) | ✅ | — |
| "mock (4e3) 100% (66/66) −1.04→4.23 energy-driven" | §5 | PHASE2_ARMS.md:222 (sweep_mock β [−1.04, 4.23], 66/66 in mock·hybr at :202) | ✅ | — |
| "Speed: hybr ~18× faster per wall-second" | §5 | PHASE2_ARMS.md:281 "hybr advances ~18× faster"; data at :276–279 (580 s vs 666 s for different sim time) | ✅ | — |
| "580 s for 0.080 Myr vs 666 s for 0.005 Myr; ARMS:247–258" | §5 | PHASE2_ARMS.md:278–279 (data), :272–283 (section); :247 = "Headline comparison 2" (transition timing, not cost) | ⚠️ | Fix ARMS:247–258 → ARMS:272–283 (the cost comparison section). |

### §6 — The cage hides the inflow

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "cage_compare.py · scratch/phase2/README.md" | §6 | cage_compare.py is at `docs/dev/archive/betadelta/diagnostics/cage_compare.py`; `scratch/phase2/` does not exist in the committed tree | ⚠️ | Fix path to `docs/dev/archive/betadelta/diagnostics/cage_compare.py`. |

### §7 — What hybr revealed (Problem 2)

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "bubble_luminosity.py:1150" for dv/dr velocity ODE source | §7 | File has 1083 lines; dv/dr = `dvdr = ((cool_beta + cool_delta)/t_now ...)` is at bubble_luminosity.py:411 | ❌ | Fix :1150 → :411. |
| "(β+δ)/t source term makes inner velocity reverse" | §7 | bubble_luminosity.py:411 (`dvdr`) — confirmed | ✅ | — |
| "legacy cage clamped β≥0, so it could never produce this" | §7 | get_betadelta.py:41 BETA_MIN=0.0 | ✅ | — |

### §8 — The causal chain

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "β −2.10, δ +0.99 → −1.11" at steep 1e6 t≈3.23 Myr | §8 | stalling-energy-phase.md:109 (t=3.228: β=−2.10, δ=+0.99, β+δ=−1.11) | ✅ | — |
| "v_min → −0.62 pc/Myr" | §8 | stalling-energy-phase.md:154 | ✅ | — |
| "negvel_causation.md · dMdt–Lmech Pearson r≈0.95" | §8 | No file `negvel_causation.md` found in the repo. Pearson r≈0.95 is in stalling-energy-phase.md prose but the separate .md file does not exist. | ❓ | Mark as unverifiable (file absent); cite stalling-energy-phase.md instead. |
| "dMdt leads the inflow (surge's dMdt step precedes inflow)" | §8 | stalling-energy-phase.md:293–305 (dMdt jumps +42%, +62% before β+δ goes negative) | ✅ | — |

### §9 — The hunt (6 configs, 909 segments)

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "h1: β+δ min −1.11, v_min −0.62, frac 0.74, cosmetic" | §9 | stalling-energy-phase.md:259; hunt_h1 CSV | ✅ | — |
| "h5 long: min(β+δ) +0.14, no inflow" | §9 | stalling-energy-phase.md:264 | ✅ | — |
| "h4 dense: v_min −1.33" | §9 | stalling-energy-phase.md:263 | ✅ | — |
| "6-config sweep + reject-and-hold counterfactual" | §9 | stalling-energy-phase.md:320–357 | ✅ | — |
| "9.6–42.8% local kick to dMdt moves final R2/v2/Eb by ≤0.04%" | §9 | stalling-energy-phase.md:329–336 (h1 max +0.043%) | ✅ | — |

### §11 — Is it physically real?

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "log10(L_mech,W) rises ~0.25 dex / ~1.8× peaking at t≈3.3 Myr" | §11 | stalling-energy-phase.md:56–62 (Lmech_W 1.95e8→3.54e8 ≈ 1.8×, peaking at ~3.28 Myr) | ✅ | — |
| "SN power exactly zero until t=3.61 Myr" | §11 | stalling-energy-phase.md:62 (Lmech_SN ~1e4–1e6 noise at t≈3.08–3.28, then 1.24e8 at t=3.63) | ✅ | — |
| "M≈1.8e-3, c_s≈338 pc/Myr at T≈4.9×10^6 K" | §11 | stalling-energy-phase.md:178–179 | ✅ | — |
| "v absent from all three cooling integrals — bubble_luminosity.py:612/659/677" | §11 | Actual integrand lines: L_bubble ~:696, L_conduction ~:745, L_intermediate ~:785. Lines 612/659/677 are array-processing code, not integrals. | ❌ | Fix :612/659/677 → ~:696/:745/:785. |

### §13 — Verdict & fix shipped

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "v_neg_frac_thick (registry + COOLING_PHASE_KEYS + compute)" | §13 | registry.py:463; dictionary.py:1225; run_energy_implicit_phase.py:767 | ✅ | — |
| "dMdt>0 + valid structure as the physical acceptance gate" | §13 | get_betadelta.py:797–805 (gate description and _hybr_g_residual) | ✅ | — |
| "β freed to [−2.44, +4.23]" | §13 | PHASE2_ARMS.md:228 (sweep totals) | ✅ | — |

### §14 "Files & glossary"

| Claim (short quote) | Report § | Current-source evidence | Verdict | Minimal fix |
|---|---|---|---|---|
| "analysis/PHASE0_BETADELTA_BASELINES.md" | §14 | Actual path: `docs/dev/archive/betadelta/PHASE0_BASELINES.md` (no `analysis/` prefix; filename differs) | ⚠️ | Fix to `docs/dev/archive/betadelta/PHASE0_BASELINES.md`. |
| "analysis/BETADELTA_PHASE2_ARMS.md" | §14 | Actual path: `docs/dev/archive/betadelta/PHASE2_ARMS.md` | ⚠️ | Fix to `docs/dev/archive/betadelta/PHASE2_ARMS.md`. |
| "analysis/stalling-energy-phase.md" | §14 | Actual path: `docs/dev/archive/betadelta/stalling-energy-phase.md` | ⚠️ | Fix to `docs/dev/archive/betadelta/stalling-energy-phase.md`. |
| "docs/dev/BETADELTA_HYBR_PLAN.md" | §14 | Actual path: `docs/dev/archive/betadelta/HYBR_PLAN.md` | ⚠️ | Fix path. |
| "trinity/phase1b_energy_implicit/get_betadelta.py" | §14 | Path correct | ✅ | — |
| "trinity/bubble_structure/bubble_luminosity.py" | §14 | Path correct | ✅ | — |
| "analysis/data/hunt_h{1..6}_*.csv" | §14 | Actual path: `docs/dev/data/hunt_h*.csv` | ⚠️ | Fix prefix from `analysis/data/` to `docs/dev/data/`. |
| "scratch/phase2/ — arms.py, probe.py, cage_compare.py" | §14 | Actual: `docs/dev/archive/betadelta/diagnostics/`. `scratch/phase2/` does not exist. | ⚠️ | Fix path. |
| "scratch/phase6/ — hunt.py, analyze_hunt.py, compare_hold.py" | §14 | Actual: `docs/dev/archive/betadelta/velstruct/`. `scratch/phase6/` does not exist. | ⚠️ | Fix path. |
| "negvel_causation.md" in §8 sidebar | §8 | File not found in repo | ❓ | Remove cite or replace with stalling-energy-phase.md §causal-chain. |
| "stalling_*.csv" path | §14 | `docs/dev/data/stalling_mock_4e3.csv`, `stalling_steep_1e6_alpha-2.csv` exist | ✅ | — |
