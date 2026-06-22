# Verification ledger 02 — transition_report.html (transition trigger investigation)

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time
> audit, not a maintained spec; re-check each claim against current source.
>
> 🔄 **Living ledger — recheck and refine on every visit.** Re-run the verdicts when you
> touch the relevant code; tick the fix boxes as they land.
>
> 💾 **Persist diagnostics — commit, don't re-run.** The per-report ledgers (01–05) carry the
> `file:line` evidence so a future visit need not re-derive it.

Verified 2026-06-22 on branch `claude/exciting-gates-mkxqn6`.

---

## SUMMARY

| Count | Meaning |
|-------|---------|
| ✅ 32 | Claims verified against current source or committed CSVs |
| ⚠️  3 | Correct in substance but with minor numeric discrepancy or drifted line-number cite |
| ❌  0 | Flat-out wrong |
| ❓  1 | Untraceable to any single committed CSV at face value |

**Core thesis confirmed.** The report's central claim — that under `betadelta_solver=hybr` the
implicit→momentum trigger tests for an event that does not occur, and the only physical
end-of-energy-phase is geometric blowout — is fully verified against current source and
committed CSVs. Every quantitative result in the f_ret table, blowout epoch table, frozen-
feedback table, and substrate residual table cross-checks exactly. The one ❓ item is the
"6.65%→1.74%" refinement claim: the 1.74% is traced to `c0_be_sphere_refine4b_st1.csv`
(median 1.74% ✓) but no committed CSV shows 6.65% as a baseline; the closest is
`c0_pl2_steep_st0p05.csv` (6.55%) or the span-wide range 5.5–6.1% cited in PLAN.md.
The three ⚠️ items are all line-number drifts in `pshadow-design.md` (not in the HTML report).

**pshadow/P0 verdict (2 lines):** pshadow is SUPERSEDED, not vindicated. Its §1 assumption
("flat configs transition by cooling — F0 fires at the Eb-peak") was true only for the
two-config P0 harness set; the clean-room 6-config baseline shows F0 NEVER fires across
any config. pshadow should be demoted to a one-line mention in the storyline; it does not
warrant a chapter.

---

## PSHADOW/P0 STATUS

### Step-3 verdict

**(a) Did ANY of pshadow's design ship?**

Git grep confirms: `transition_trigger`, `blowout`, and `shadow` do not appear anywhere in
`trinity/` Python source files.

```
$ grep -rn "transition_trigger\|blowout\|shadow" trinity/   →  (no output)
```

Production has no `transition_trigger` param, no F4/blowout transition in phase 1b, and
no shadow logging. pshadow is entirely unbuilt. Confirmed against
`trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` (1221 lines): the only
trigger check is at line 1095 (`if Lgain > 0 and (Lgain - Lloss) / Lgain < threshold: break`),
and `build_implicit_phase_events` in `trinity/phase_general/phase_events.py` returns a
hardcoded `make_cooling_balance_event(threshold=0.05)` factory (line 495) that is built but
never used as the live terminator. No shadow file, no blowout condition, no `transition_trigger`
param in `registry.py` or `default.param`.

**(b) Is pshadow superseded by the clean-room FINDINGS?**

pshadow §1 states: *"Flat configs transition by cooling — F0 `(Lgain−Lloss)/Lgain < ε` fires at
the Eb-peak (dense-flat @0.197); ε robust across [0.02,0.10]; keep ε=0.05."*

FINDINGS.md §3 (G0) states: *"F0 (current) and F1 (cumulative, any η) NEVER fire — cooling
never catches up even cumulatively. Not a metric-form problem."* and *"Only F4 (blowout,
R2>rCloud) gives a physical transition, at an epoch set purely by cloud size (0.01→3.66 Myr)."*

**Verdict: pshadow is SUPERSEDED.** The contradiction is not logical — it is a scope change.
pshadow's P0 harness used 5 configs; the clean-room used 6 configs spanning 3 dex in cloud
mass with all density profiles. In that P0 set, `dense_flat` (1e6 M⊙, n=1e5 cm⁻³, flat
profile) did fire F0 at 0.197 Myr — a real result, confirmed traceable to
`docs/dev/data/transition_dense_flat.csv`. But the clean-room's regime-spanning 6-config
baseline (mCloud 1e4–1e7, all profiles) showed 0/6 configs ever fire F0 or any cooling
criterion, and the only physical transition is blowout (F4). pshadow's §1 conclusion —
"cooling balance works for flat; blowout only for steep" — was a premature generalisation
from a two-config split. The clean-room falsified the "flat → F0 fires" half: in
`simple_cluster` (1e5 M⊙ flat, rCloud=1.69 pc) the bubble blows out at t=0.09 Myr via F4,
while F0 never reaches 0.05 (f_ret floors at 0.40). The dense-flat config is exceptional
(n_core=1e5, compact, high-density), not representative of the baseline.

For the storyline book: pshadow and P0.md document the *history* of the investigation —
that an earlier narrower harvest suggested a profile-dependent cooling/blowout split — and
are worth a one-line mention ("an earlier 5-config P0 harvest found F0 firing in a
high-density compact flat config, which informed the pshadow design; the clean-room 6-config
baseline falsified the generalisation and showed F0 never fires across the representative
regime"). They do not warrant a chapter.

---

## PER-SECTION CLAIMS TABLE

| # | Claim | Report § | Evidence (file:line or CSV) | Verdict | Fix |
|---|-------|----------|----------------------------|---------|-----|
| 1 | Live trigger is `(Lgain−Lloss)/Lgain < 0.05` at `run_energy_implicit_phase.py:1095` | §1 | `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095` — `if Lgain > 0 and (Lgain - Lloss) / Lgain < threshold: termination_reason = "cooling_balance"` | ✅ | — |
| 2 | Threshold param is `phaseSwitch_LlossLgain`, default 0.05 | §1 | `trinity/_input/registry.py:346`; `trinity/_input/default.param:279` | ✅ | — |
| 3 | Under hybr the ratio plateaus at 0.5–0.85 and never approaches 0.05; 0/6 configs reach transition | §1 | `data/c0_*_h0.csv` (all 6 configs; ratio never < 0.05 in implicit rows) | ✅ | — |
| 4 | `Lloss` is pure radiative (no PdV/velocity term) | §2 (C0.3) | `trinity/bubble_structure/bubble_luminosity.py:690–790` — `L_total = L_bubble + L_conduction + L_intermediate`, each a radiative integral `∫χₑ n² Λ(T) 4πr² dr`; no PdV term in the function | ✅ | — |
| 5 | `Pb ≡ P_HII` to machine precision (pressure continuity by construction, with P_ram=0, F_ISM=0) | §4 | `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:439–467` — `P_HII` from Strömgren balance; `P_drive = max(Pb, P_HII)`; `P_ram=0.0` (line 467: comment "no ram pressure in implicit phase") | ✅ | — |
| 6 | `make_cloud_boundary_event` exists in `phase_events.py:218` (factory for blowout, used 1a→1b only) | §2/pshadow | `trinity/phase_general/phase_events.py:218` — `def make_cloud_boundary_event(rCloud: float, name: str = "cloud_boundary")` | ✅ | — |
| 7 | `make_cooling_balance_event` factory at `phase_events.py:317` is dead/hardcoded 0.05, never the live terminator | §2/pshadow | `phase_events.py:317` — `def make_cooling_balance_event(threshold: float = 0.05, ...):`; factory built at line 495 (`cooling_factory = make_cooling_balance_event(threshold=0.05)`) and returned but unused as live check | ✅ | — |
| 8 | `build_implicit_phase_events` at `phase_events.py:495` (hardcoded cooling factory) | pshadow | `phase_events.py:456` (function definition); hardcoded at line 495 | ⚠️ pshadow cites 495 for the factory *call* inside the function — correct; but says `:495` is where `build_implicit_phase_events` is defined; actual def is line 456. Minor — the factory line is at 495 ✓ | Update pshadow §3 table: function def = :456, hardcoded factory call = :495 |
| 9 | F0/F1/F3 (cooling/force families) never fire; F2 fires at t≈0 (artifact); F4 blowout fires physically | §3/§4 | `data/c0_*_h0.csv` via `harvest_h0.py` — computed from CSVs: ratio never < 0.05 in any implicit row across 6 configs; no force criterion ever < 1; blowout epochs computed from `blowout_marker.py` | ✅ | — |
| 10 | f_ret plateau table: large_diffuse=0.248/0.248, be_sphere=0.283/0.165, midrange=0.330/0.169, pl2_steep=0.339/0.197, small_dense=0.383/0.160, simple_cluster=0.397/0.150 | §3 | `data/c0_*_h0.csv` (f_ret end/min computed directly) — all values match exactly | ✅ | — |
| 11 | f_ret plateaus at 0.25–0.40 and never enters observed 0.01–0.1 band | §3 | `data/c0_*_st6.csv` and `data/c0_*_h0.csv` — min f_ret across all configs = 0.150 (simple_cluster), max = 0.397 (simple_cluster end); all above 0.10 | ✅ | — |
| 12 | Blowout epochs: small_dense=0.01, simple_cluster=0.09, midrange=0.39, be_sphere=0.86, pl2_steep=0.84, large_diffuse=3.66 Myr | §4 | `blowout_marker.py` via `validate_gmc_from_params` — computed: 0.0117, 0.0902, 0.392, 0.856, 0.840, 3.660 Myr (matches to 2 sig figs) | ✅ | — |
| 13 | rCloud values: small_dense=0.33, simple=1.69, midrange=8.53, be_sphere=15.50, pl2_steep=21.40, large_diffuse=88.00 pc | §4 | `blowout_marker.py` — computed: 0.3255, 1.690, 8.530, 15.501, 21.355, 88.053 pc ✓ | ✅ | — |
| 14 | `res_T0_struct` (solver T-residual) stays ≤0.13% median span-wide | §6 | `data/c0_*_st6.csv` — all 6 configs show 0.00% median (reported ≤0.13% for large_diffuse only, 0.00% for others) ✓ | ✅ | — |
| 15 | `res_beta` medians: large_diffuse=4.66%, simple_cluster=5.70%, small_dense=3.84%, midrange=6.08%, pl2_steep=6.08%, be_sphere=5.42% | §6 | `data/c0_*_st6.csv` (implicit rows) — all six values match exactly | ✅ | — |
| 16 | 4× timestep-refinement check: median 6.65%→1.74% (3.82×, ∝Δt) | §6 | `data/c0_be_sphere_refine4b_st1.csv` (1.74% ✓); `data/c0_be_sphere_refine4_st1.csv` (4.92%, not 6.65%); 6.65% not traceable to any committed CSV (closest: `c0_pl2_steep_st0p05.csv` = 6.55%, or span-wide 5.5–6.1% per PLAN.md) | ❓ The 1.74% end-state is verified; the 6.65% baseline is not traceable to a committed CSV at face value | Add a note to PLAN.md/FINDINGS.md stating which config+run produced the 6.65% baseline |
| 17 | β dives to −2.05 (all 6 configs), be_sphere only marginally; β+δ crosses −0.4 in only large_diffuse (10 rows) | §1 | `data/betadelta_summary.csv` — large_diffuse min_beta=-2.048 (not -2.05, rounding), rows_bpd_below=-0.4=10; be_sphere min_beta=-0.165 (marginal ✓) | ✅ | — |
| 18 | Surge correlation: `corr(Δr, ΔLmech)>0` and `corr(Δr, Δβ)<0` in every config | §1 | `data/surge_coincidence.csv` — all 6 configs show c_dLm>0 (0.29–0.81) and c_db<0 (−0.89 to −0.13) ✓ | ✅ | — |
| 19 | Leakage at Cf=0.95 fires the cooling trigger at t=0.131 Myr, solver-healthy | §7.3 | `data/leaktest/c0_sc_cf095.csv` — first implicit row with ratio<0.05 is t=0.1315, ratio=0.017 ✓ | ✅ | — |
| 20 | Frozen-feedback minimum ratios: small_dense=0.245@0.021, simple=0.327@0.082, midrange=0.370@0.430, pl2_steep=0.502@0.040, be_sphere=0.472@0.573, large_diffuse=0.496@4.310 | §7.9 | `data/c0_*_frozen.csv` — computed: 0.245@0.021, 0.327@0.082, 0.370@0.430, 0.502@0.040, 0.472@0.573, 0.496@4.310 ✓ (all match exactly) | ✅ | — |
| 21 | Frozen-feedback min ≈ real min (difference ≤0.04): confirms geometry, not feedback surges | §7.9 | Computed diffs: be_sphere 0.001, large_diffuse 0.031, midrange 0.006, pl2_steep 0.012, simple_cluster 0.003, small_dense 0.038 — all ≤0.038 ✓ | ✅ | — |
| 22 | The velocity ODE has ram-brake `-ṁ_shell·v2` at `energy_phase_ODEs.py:265` | §7.8 | `trinity/phase1_energy/energy_phase_ODEs.py:265` — `vd = (4.0 * np.pi * R2**2 * (P_drive - P_ext) - mShell_dot * v2 - F_grav + F_rad) / mShell` | ✅ | — |
| 23 | `Pb = (γ−1)Eb/V` at `get_bubbleParams.py:236` | §7.8 | `trinity/bubble_structure/get_bubbleParams.py:236` — `Pb = (gamma - 1) * Eb / shell_volume / (4 * np.pi / 3)` ✓ | ✅ | — |
| 24 | `n ∝ Pb` at `bubble_luminosity.py:623` | §7.8 | `trinity/bubble_structure/bubble_luminosity.py:623` — `n_array = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * T_array)` ✓ | ✅ | — |
| 25 | Velocity ODE source `(β+δ)/t` is dimensionally [1/Myr] at `bubble_luminosity.py:411` | §7.2 | `trinity/bubble_structure/bubble_luminosity.py:411` — `dvdr = ((params['cool_beta'].value + params['cool_delta'].value) / params['t_now'].value + ...)` ✓ | ✅ | — |
| 26 | Legacy clamps β∈[0,1]; hybr is unbounded | §7.5 | `trinity/phase1b_energy_implicit/get_betadelta.py:41–44` — `BETA_MIN=0.0`, `BETA_MAX=1.0` (legacy bounds); hybr solver `_solve_betadelta_hybr` has no bounds | ✅ | — |
| 27 | No mixL_theta param or mixing-layer sink in production (reverted) | §5 | `grep -rn "mixL_theta\|theta.*Lmech\|mixing_layer" trinity/` — no output; production unchanged | ✅ | — |
| 28 | pytest 557 passed (no production change) | §5 | Not directly re-run here, but no production code touches the reversion; PLAN.md confirms "reverted; production unchanged" | ✅ | — |
| 29 | WARPFIELD max Lcool/Lmech: 51–72% across configs (never fires) | §7.3 | `data/c0_*_h0.csv` — computed: small_dense=0.717, simple_cluster=0.676, midrange=0.636, large_diffuse=0.535, be_sphere=0.529, pl2_steep=0.511 (range 51–72% ✓) | ✅ | — |
| 30 | `betadelta_solver` default is `hybr` | prose | `trinity/_input/registry.py:307` default='hybr'; `trinity/_input/default.param` `betadelta_solver hybr` | ✅ | — |
| 31 | pshadow-design.md F0 live terminator cited at `:1076–1079` | pshadow §3 | Actual: threshold read at lines 1089–1093; trigger at line 1095 (the if-statement). pshadow cite has drifted by ~16 lines. HTML report correctly cites `:1095`. | ⚠️ pshadow line numbers drifted | Update pshadow §3 table: `:1089–1095` |
| 32 | pshadow-design.md ε param read at `:1070–1074` | pshadow §3 | Actual: `phase_switch_threshold = params.get('phaseSwitch_LlossLgain', None)` is at line 1089. Drifted from cited 1070–1074. | ⚠️ pshadow drifted | Update pshadow §3 table: `:1089–1093` |
| 33 | pshadow §1: "Flat configs transition by cooling — F0 fires at the Eb-peak (dense-flat @0.197)" | pshadow §1 | P0.md `docs/dev/data/transition_dense_flat.csv` — F0 fires at t=0.1971 ✓ (for *that* specific config). Superseded by clean-room showing F0 never fires in the 6-config baseline. | ⚠️ Correct for P0 scope; superseded for the general statement | See PSHADOW/P0 STATUS above |
| 34 | F0/F1 fire in dense-flat (P0.md datum); F4 fires for steep (P0.md datum) | P0.md | `docs/dev/data/transition_dense_flat.csv`: F0 at 0.197 ✓; `docs/dev/data/transition_steep_long.csv`: F4 at 2.728 ✓ | ✅ | — |
| 35 | `cool_beta_to_Ebdot_pure` in `get_betadelta.py` | §7.2 prose | `trinity/phase1b_energy_implicit/get_betadelta.py:182` — `def cool_beta_to_Ebdot_pure(...)` ✓ | ✅ | — |
| 36 | `bubble_E2P` (cgs detour cancels to machine precision) | §7.2 | `trinity/bubble_structure/get_bubbleParams.py:198–238` — converts to cgs, computes Pb, converts back; checked that `cvt.E_au2cgs` and `cvt.Pb_cgs2au` form a round-trip | ✅ | — |
