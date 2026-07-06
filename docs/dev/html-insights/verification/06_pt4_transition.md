# Verification ledger 06 — pt4_transition_report.html (transition trigger investigation, part 4)

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time
> audit, not a maintained spec; re-check each claim against current source.
>
> 🔄 **Living ledger — recheck and refine on every visit.** Re-run the verdicts when you
> touch the relevant code; tick the fix boxes as they land.
>
> 💾 **Persist diagnostics — commit, don't re-run.** This ledger carries the `file:line`
> and CSV evidence so a future visit need not re-derive it.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

Verified 2026-06-22 against the `fix/transition-trigger-problem-pt4` checkout (ledger committed
in worktree `worktree-agent-a82a2bc9bd998b563`).
**Target:** `docs/dev/transition/pt4/pt4_transition_report.html` (the s1 ch5 source, built by
`docs/dev/transition/pt4/make_pt4_transition_report.py`; merged into `storyline_s1.html` as
ch5 "Re-examining the verdict... (part 4)"). Read-only on everything except this ledger.

---

## SUMMARY

| Count | Meaning |
|-------|---------|
| ✅ 41 | Claims verified against current source, committed CSVs, figures, or git objects |
| ⚠️  2 | Correct in substance but with a minor numeric/rounding inconsistency |
| ❌  0 | Flat-out wrong |
| ❓  0 | Untraceable |

**Core thesis CONFIRMED.** The report's thesis — that part 3's "geometric, not thermal"
verdict survives three good-faith objections (H1: not an `Lcool` bug; H2: rCloud is a clean
boundary whose removal cannot manufacture a cooling event; R1: ship an opt-in transition on
events that *do* occur) — is fully verified. Every quantitative claim traces exactly to a
committed pt4 audit doc / CSV; every `file:line` cite to production is correct against current
source; all 7 figures show what their captions and the surrounding prose claim; MathJax is
balanced and notation is consistent within the chapter and with the neighbor cleanroom chapter.
The R1 production hooks (`evaluate_r1_shadow`, `parse_transition_triggers`,
`r1_transition_decision`, the opt-in `transition_trigger` keyword, the shadow block, the
`Edot_from_balance` formula) all exist exactly as described. The two ⚠️ are cosmetic.

**The two ⚠️ (both minor):**
1. **midrange_pl0 ratio-min shown as 0.364 (H1 table) AND 0.365 (H2 table)** — the same
   underlying value 0.36447 rendered inconsistently. The H2 value rounds *up* incorrectly
   (0.36447 → should be 0.364). Internal inconsistency within one report. Fix in the generator.
2. **bigcloud "5.2×" vs the param-file comment "~4.6×"** — the report's 5.2× is the *correct*
   computed value (8.83/1.69 = 5.22); the stale `~4.6×` lives only in `sc_bigcloud.param`'s
   comment (NOT in the report). Flagged for hygiene; the report is right.

---

## FLOW / NARRATIVE assessment

**Reads as a clean, well-motivated continuation of the neighbor chapter.** The neighbor
(`cleanroom/transition_report.html`) closes on "geometric, not thermal" / "the cooling-balance
criterion tests for an event that does not occur" / "never leave the energy phase" (lines 47,
64, 73, 207, 262, 548). The pt4 §0 recap (`SEC_RECAP`) quotes that verdict verbatim and frames
the chapter as *stress-testing* it before trusting it — the correct, honest move. Structure is
a clean three-act arc (H1 → H2 → R1) mirrored by the flow-step cards at the top, each with a
one-line answer that the body then earns with a figure + a number.

- **Voice/tense:** consistent throughout (present-tense diagnostic, "we audited / we checked"),
  matching the neighbor's register.
- **Continuity of physics:** the R1 §3 Eb-peak equation `\(\dot E_b = L_{gain}-L_{loss}-4\pi
  R_2^2 v_2 P_b \le 0\)` is the *same* candidate the neighbor already floated as the
  "PdV-inclusive" trigger (`cleanroom` line 453: `\((L_{gain}-L_{loss}-\dot W)/L_{gain}` is the
  normalised `\dot E_b/L_{gain}`, fires at the `E_b`-peak). pt4 turns that idea into shipped
  code — a natural, non-contradictory escalation.
- **No internal contradictions.** H1 verdict ("not a bug; β-clamp is the whole legacy/hybr
  split") and H2 verdict ("clean boundary, refuted as a fix") both feed cleanly into R1's
  premise ("no cooling event exists ⇒ build the transition on blowout + Eb-peak").
- **No contradiction with the prior chapter or the source docs.** The recap's `f_ret` 0.25–0.40
  / observed 0.01–0.1 band, the ratio floor 0.28–0.49, and the blowout epochs all agree with
  both the neighbor report and the pt4 audit docs.
- **Honesty / caveats:** §3.2's `over` box is appropriately candid — drive validated only on
  `simple_cluster`; the heavy-cloud Eb-peak→1c hand-off is explicitly future work. The closing
  (`SEC_CLOSE`) ties the full s1 arc (repaired → fast → can't-leave → opt-in exit) without
  overclaiming.

Only nit: the heavy-cloud (5e9) configs appear abruptly in §3.2 / the R1 table without a recap
of what `fail_repro`/`fail_helix` are — a reader who skipped the failed-large-clouds storyline
must infer they are the 5e9 stress configs. Not an error (the report labels them "the two heavy
5e9 clouds"), just a small assumed-context spot. No fix required.

---

## EQUATIONS check

MathJax delimiters: **127 `\(` = 127 `\)` (balanced); 1 `\[` = 1 `\]`** — and that single
display pair is the MathJax *config* (`displayMath` delimiter list), not content; the report
uses inline math only, like the neighbor. No stray `$` content (the 6 `$` are the inline-delim
config). CDN include present; `assert "__FIG_" not in html` in the generator guarantees no
unreplaced placeholders. Verified via a balance script over the rendered HTML.

| # | Equation / inline-math | Where | Verdict | Evidence |
|---|---|---|---|---|
| E1 | cooling-balance trigger `\((L_{gain}-L_{loss})/L_{gain}<0.05\)` | HERO, §0, flow | ✅ | `run_energy_implicit_phase.py:1200` `(Lgain - Lloss) / Lgain < threshold`; threshold = `phaseSwitch_LlossLgain` default 0.05 (`registry.py:346`, `default.param:279`) |
| E2 | `\(L_{cool}\propto n^2V\)` / `\(L_{loss}=\int n^2\Lambda(T)dV\)` | TL;DR, §1, §1.2 | ✅ | `bubble_luminosity.py:696` `integrand_bubble = chi_e * n_bubble**2 * Lambda_bubble * 4πr²`; `n_array = Pb/((mu_convert/mu_ion)·k_B·T)` (:623); three zones summed at `:790` |
| E3 | Eb-peak / net energy `\(\dot E_b = L_{gain}-L_{loss}-4\pi R_2^2 v_2 P_b \le 0\)` | §3 bullet | ✅ | `get_betadelta.py:434` `Edot_from_balance = L_gain - L_loss - 4 * np.pi * R2**2 * v2 * Pb` — character-exact. (Report cites `:434`. The brief said `:426-434`; the assignment is on `:434`, the `L_gain`/`L_loss` setup on `:426-432`.) |
| E4 | `\(\beta=-t\,\partial_t\ln P_b\)` (cool_beta definition) | §1.3 / fig caption | ✅ | Standard cool_beta definition; matches `H1_lcool_audit.md §2` and the figure y-axis label. Consistent with the neighbor's β usage. |
| E5 | `\(n^2V\propto(P_b/T_0)^2R_2^3\)` dilution | §1.2 | ✅ | Dimensionally consistent with E2 + `Pb=(γ−1)Eb/V`; matches `H1_lcool_audit.md §4`. |
| E6 | `\(f_{ret}=E_b/\!\int\!L_{mech}dt\)`, plateau 0.25–0.40 | §0 recap | ✅ | Matches neighbor `cleanroom` line 73 ("`f_ret` plateaus at 0.25–0.40, never the observed 0.01–0.1 band") — same definition, same band. |
| E7 | rCloud `\(\propto m_{cloud}^{1/3}\)`; `\(m_{cluster}=m_{cloud}\cdot sfe\)` | §2.2 | ✅ | `H2_rcloud_audit.md §1/§4`; `powerLawSphere.py:51-74` `rCloud=(3M/4πρ)^{1/3}`; cluster mass = mCloud·sfe. bigcloud param: mCloud 1e7·sfe 0.003 = 3e4 = baseline 1e5·0.3 ✓. |

**Notation consistency.** The pt4 report uses `\(L_{\text{gain}}\)`, `\(L_{\text{loss}}\)`,
`\(L_{\text{cool}}\)`, `\(L_{\text{mech}}\)`, `\(R_2\)`, `\(r_{\text{cloud}}\)`,
`\(f_{\text{ret}}\)`, `\(\beta,\delta\)`, `\(P_b\)`, `\(\dot E_b\)`, `\(\Lambda\)`. The neighbor
cleanroom report uses **the identical `\text{...}`-subscript convention** for every one of these
(grep-confirmed: `L_{\text{gain}}`, `r_{\text{cloud}}`, `R_2`, `f_{\text{ret}}`, `\dot E_b`,
`P_b`, `PdV`). The brief wrote some of these with `\rm` (`L_{\rm gain}`), but **both reports
consistently use `\text{}`, not `\rm`** — internally and cross-chapter consistent. No mismatch.

---

## PLOTS check (all 7 embedded figures Read as PNG)

Every figure referenced in prose is embedded, and every embedded figure is referenced
(generator maps 7 placeholders → 7 `figures/*.png`; `assert` guarantees all replaced). No
wrong/duplicated figure. Each was opened and inspected:

| Figure | Shows what caption+prose claim? | Verdict | Notes |
|---|---|---|---|
| `h1_lloss_surge_collapse.png` | Log-log Lcool=Lloss, 6 hybr configs, peak dots, surge then collapse | ✅ | simple_cluster peaks ~4e8 @ ~0.1 Myr, collapses to ~5e7 @ ~0.8 (the −7.2× claim), recovers. All 6 show the turnover. Title matches caption. |
| `h1_beta_clamp_divergence.png` | Top: cooling ratio legacy crosses 0.05 / hybr recovers; Bottom: β legacy in [0,1] band / hybr → +4 | ✅ | Curves identical to ~0.08 Myr then split; hybr β peaks ~4.2 @ ~0.4–0.5; legacy driven to 0 @ ~0.18. Matches §1.3 + the +3.5/+4.2 prose. (Legacy β briefly pokes just above 1.0 at the split — the edge-root artifact; substantively fine.) |
| `h1_ratio_min_stats.png` | Bars: hybr floors 0.28–0.49 (0/6 fire); legacy ≤0 in 5/6 | ✅ | hybr bars = 0.324/0.283/0.364/0.489/0.471/0.465 (match H1 table). legacy: simple/midrange/be below 0, small_dense ~0.024, large_diffuse ~0.51 (doesn't fire). 0.05 dashed line shown. |
| `h2_ratio_vs_rcloud.png` | Ratio vs R2/rCloud, all 6, R2=rCloud line, recovers past edge | ✅ | Bottoms ~0.3–0.5 near edge then recovers; pl2_steep (dark blue) bottoms far left (~0.06, deep inside). 0.05 trigger dashed, untouched. |
| `h2_matched_r2.png` | Ratio vs absolute R2; baseline+bigcloud overlap; matched R2=0.894, ratio 0.4845 in both | ✅ | Curves overlap exactly in-cloud; annotation "matched R2=0.894 pc, ratio=0.4845 in BOTH (R2/rCloud=0.53 vs 0.10)" at dotted line. CSV-confirmed (both runs ratio=0.4845 @ R2=0.8939). |
| `h2_dip_vs_density_gradient.png` | Top: n(r) per config, dot at ratio bottom (edge for flat, rCore for pl2); Bottom: dip-location vs steepness | ✅ | pl2_steep dot at ~0.06, be_sphere ~0.7, 4 flat at edge ~1.0; correlation panel: flat cluster top, pl2_steep bottom-right (steepest). Matches §2.3. |
| `r1_firing_preview.png` | Per config, ★ blowout (6 normal) / ◆ Eb-peak (2 heavy), grey bar = never-firing current trigger | ✅ | Star epochs match r1_shadow_summary.csv (0.09/0.012/0.39/0.84/0.86/3.66); fail_repro/fail_helix shown as Eb-peak diamonds @ ~1e-3 Myr (1a events). Caption/prose match. |

---

## PER-CLAIM table (RESULTS / NUMBERS + CODE)

| # | Claim | Report § | Evidence (file:line or CSV) | Verdict |
|---|-------|----------|----------------------------|---------|
| 1 | Cooling integral byte-identical across refactor chain `7f08e58/24c6914/4996060/60fb362` | §1.1 | All four are real git commits (`git cat-file -t` = commit). `H1_lcool_audit.md §3` documents the byte-identical diff. | ✅ |
| 2 | Two genuine content changes: `+20% chi_e` (`9222a96`) and conduction dense-output `≤0.18%` (`5f4f229`) | §1.1 | Both commits exist; `9222a96` adds `chi_e` to CIE integrands (`bubble_luminosity.py:696,783`); `H1_lcool_audit.md §3` matches | ✅ |
| 3 | `Lcool` surges ~2× then collapses 4–9× | TL;DR/§1.2 | `H1_lcool_direction_summary.csv` peak vals; trajectory: simple 1.90e8→4.10e8→5.68e7 (−7.2×), pl2 −4.3×, small_dense −8.6× (`H1_lcool_audit.md §4`) | ✅ |
| 4 | simple_cluster: Lloss 1.90e8 → peak 4.10e8 @ t=0.098 → 5.68e7 @ t=0.84 (−7.2×) | §1.2 | `c0_simple_cluster_h0.csv` first finite Lloss=1.896e8 @ t=0.0034; CSV `Lloss_peak_val=4.102e8 @ peak_t=0.0981`. (Summary CSV's `Lloss_first`=2.90e8 is a different filtered first-row; the report uses the trajectory-probe first value 1.90e8 — correct.) | ✅ |
| 5 | hybr β jumps to +3.5 @ t=0.22, +4.2 @ t=0.46 | §1.3 | `h1_beta_clamp_divergence.png` bottom panel: hybr β peaks ~4.2 @ ~0.4–0.5 Myr; `H1_lcool_audit.md §4` | ✅ |
| 6 | hybr ratio floors 0.28–0.49, 0/6 reach 0.05; legacy crosses in 5/6 | §1/§1.3/fig | `H1_lcool_direction_summary.csv`: hybr min 0.283–0.489, all NO; legacy 5/6 YES, large_diffuse min 0.514 NO | ✅ |
| 7 | H1 table (hybr): simple 0.324@0.098/0.764, large_diffuse 0.465@4.86/0.561, small_dense 0.283@0.015/0.695, midrange 0.364@0.432/0.833, pl2 0.489@0.037/0.831, be 0.471@0.556/0.829 | §1.3 table | `H1_lcool_direction_summary.csv` — all ratio_min / ratio_min_t / ratio_final match to 3 dp | ✅ |
| 8 | H1 table (legacy): simple −0.007@0.178, small_dense 0.024@0.024, midrange −0.009@0.82, pl2 −0.001@0.128, be −0.020@1.04 | §1.3 table | `H1_lcool_direction_summary.csv` legacy rows match exactly (cross_t 0.178/0.024/0.822/0.128/1.037) | ✅ |
| 9 | `Lloss = bubble_LTotal` (pure radiative, no PdV in trigger); PdV only in solver `Edot_from_balance` | §1/§3 | `run_energy_implicit_phase.py:1145` Lloss = bubble_LTotal (+Leak); trigger `:1200`; PdV term only at `get_betadelta.py:434` | ✅ |
| 10 | rCloud is derived (`run_const`, `derived_init`), no input knob; `R2>rCloud` is a clean 1a→1b hand-off with `is_simulation_ending=False` | §2.1 | `H2_rcloud_audit.md §1,§3`: `registry.py:393` rCloud derived; `phase_events.py:218-247` cloud_boundary `is_simulation_ending=False`; RCLOUD_BOUNDARY code 3 in clean band | ✅ |
| 11 | Implicit phase carries NO rCloud event; integrates past it; no NaN/error/crash | §2.1 | `H2_rcloud_audit.md §3`; `build_implicit_phase_events` has no rCloud event | ✅ |
| 12 | Matched-R2: ratio 0.4845, Lloss/Lgain 0.5155 at R2=0.894 pc in BOTH baseline and bigcloud | §2.2/fig | `h2_sc_baseline.csv` & `h2_sc_bigcloud.csv`: at R2=0.8939 both give ratio=0.4845, Lloss/Lg=0.5155 (verified directly); curves overlap in-cloud | ✅ |
| 13 | bigcloud rCloud 8.83 pc = 5.2× of 1.69 at fixed feedback (mCloud 1e7/sfe 0.003 vs 1e5/0.3 → mCluster 3e4 both) | §2.2 | `sc_bigcloud.param` mCloud 1e7 sfe 0.003; `sc_baseline.param` 1e5/0.3; 8.83/1.69=5.22 ✓ (param-file comment says "~4.6×" — stale, see ⚠️) | ⚠️ |
| 14 | baseline R2/rCloud=0.53, bigcloud=0.10 at R2=0.894 | §2.2 | 0.894/1.69=0.529, 0.894/8.83=0.101 ✓ | ✅ |
| 15 | H2 table: rCloud 0.326/1.69/8.53/21.35/15.5/88.05 pc | §2.3 table | `h2_crossing_summary.csv` & `h2_rcloud_edge.csv` rCloud_pc — match | ✅ |
| 16 | H2 table t_cross: 0.0117/0.0902/0.392/0.840/0.856/3.66 Myr | §2.3 table | `h2_crossing_summary.csv` t_cross col — match exactly | ✅ |
| 17 | H2 table R2max/rCloud: 525×/147×/34.4×/13.9×/15.2×/1.52× | §2.3 table | `h2_crossing_summary.csv` col `R2max_over_rCloud`: 524.6/146.6/34.4/13.9/15.2/1.52 — match | ✅ |
| 18 | H2 table R2/rCloud @ ratio min: 1.11/1.07/1.06/0.064/0.72/1.22 | §2.3 table | `h2_crossing_summary.csv` col `R2overRc_at_ratio_min`: 1.114/1.073/1.058/0.064/0.724/1.220 — match | ✅ |
| 19 | H2 table ratio min: 0.283/0.324/**0.365**/0.489/0.471/0.465 | §2.3 table | `h2_crossing_summary.csv` ratio_min: 0.283/0.324/**0.36447**/0.489/0.471/0.465. midrange 0.36447 shown as 0.365 here but 0.364 in §1.3 H1 table — same value, inconsistent rounding | ⚠️ |
| 20 | nCore/nEdge density drops per config (1e5/219/1e6/1e4/100/714) | §2.3 / fig | `h2_rcloud_edge.csv` nEdge_over_nISM col: 1e5/219.3/1e6/1e4/100/714.3 — match | ✅ |
| 21 | pl2_steep bottoms at R2/rCloud≈0.06 (rCore), flat configs at ≈1.0 (edge) | §2.3 | `h2_crossing_summary.csv` R2overRc_at_ratio_min: pl2=0.064, flat 1.06–1.22; `h2_dip_vs_density_gradient.png` confirms | ✅ |
| 22 | Eb-peak `\(\dot E_b\)` already computed in production as `Edot_from_balance` at `get_betadelta.py:434`; shadow only reads it | §3 | `get_betadelta.py:434` exact formula; `run_energy_implicit_phase.py:1166` `_edot_bal = betadelta_result.Edot_from_balance` then `evaluate_r1_shadow` | ✅ |
| 23 | R1 helper pair: `evaluate_r1_shadow()` (criteria), `parse_transition_triggers()` / `r1_transition_decision()` (keyword map) | §3 | `run_energy_implicit_phase.py:197` evaluate_r1_shadow, `:216` parse_transition_triggers, `:239` r1_transition_decision | ✅ |
| 24 | blowout = `R2>rCloud`; ebpeak = `edot_balance<=0` | §3 | `run_energy_implicit_phase.py:207-210` `blowout = R2 > k_blowout*rCloud`, `ebpeak = edot_balance <= 0` | ✅ |
| 25 | Shadow always-on/inert: writes sideline `shadow_R1_1b.csv`, never sets termination_reason/breaks ⇒ byte-identical dictionary.jsonl | §3.1 | `run_energy_implicit_phase.py:1160-1183` shadow block (logs only, sideline CSV); G1 gate `GATE_RESULT.txt` sha256 `830b691a…` identical | ✅ |
| 26 | Drive opt-in: `transition_trigger` param, default `cooling_balance`; non-default (blowout/ebpeak/r1/combos) drives R1 | §3.1 | `registry.py:347` default 'cooling_balance'; `default.param:282` cooling_balance; drive at `:1192-1196` (`r1_transition_decision`); `_VALID_TRIGGERS` + 'r1' alias `:213,229` | ✅ |
| 27 | Gates: G1 byte-identical sha `830b691a…`; G2 unit 14/14; G3 regression 588 passed; drive end-to-end on simple_cluster (1c 2.1s → momentum) | §3.1 box | `GATE_RESULT.txt` (sha match); `test/test_r1_shadow.py` = 14 test fns (counted); `R1_FINDINGS.md` gates table | ✅ (588/2.1s per `R1_FINDINGS.md`, not re-run) |
| 28 | Live shadow: blowout fires for all 6 in-cloud configs (0.012–3.66 Myr); Eb-peak never fires in-cloud for normal clouds | §3.2 | `r1_shadow_summary.csv`: all 6 which_fired_first=blowout, blowout_t 0.0117–3.66, ebpeak_t empty | ✅ |
| 29 | R1 table blowout epochs + R2/rCloud (1.02/1.02/1.01/1.05/1.01/1.00) | §3.2 table | `r1_shadow_summary.csv` blowout_R2overRc: 1.022/1.015/1.005/1.050/1.007/1.003 — match to 2 dp | ✅ |
| 30 | Heavy 5e9 clouds (fail_repro/fail_helix): empty 1b shadow; Eb-peak is a 1a event (Eb≤0 collapse precedes the 1b shadow site) | §3.2 | `r1_shadow_summary.csv`: fail_repro/fail_helix n_seg=0, which_fired_first=none; `R1_FINDINGS.md`/`DATA_NOTE.md` explain the 1a placement | ✅ |
| 31 | cooling ratio min across all R1 shadow rows = 0.283 | §3.2 | `DATA_NOTE.md` sanity check 4: min 0.2832 (small_dense); per-config 0.2832–0.4892 | ✅ |
| 32 | Offline blowout epoch matches in-code `blowout_t` to `|Δt|=0` | §3.2 | `cross_validate_result.txt`: simple/pl2/be/large all `|dt|=0.000e+00` | ✅ |
| 33 | R1 caveat: drive validated only on simple_cluster; heavy-cloud Eb-peak→1c hand-off is future work (Path-2 continuity) | §3.2 box | `R1_FINDINGS.md` Conclusion & caveats — verbatim | ✅ |
| 34 | Recap: f_ret 0.25–0.40 (never 0.01–0.1); cooling tests for an event that does not occur; geometric blowout `R2>rCloud` | §0 | Neighbor `cleanroom/transition_report.html` lines 47/64/73/207/262/548 — faithful | ✅ |
| 35 | Six regime-spanning configs same as part 3 (~3 dex mass, all profiles) | §0 | `h2_rcloud_edge.csv` profiles: PL α=0 ×4, PL α=−2, BE; nCore span 1e2–1e6 | ✅ |
| 36 | shadow commits `28611a7` (gate A) / `b71cca6` (committed shadow) are real | §3.1/notes | `git cat-file -t` = commit for both | ✅ |
| 37 | `9222a96`/`5f4f229` are real upstream commits | §1.1 | `git cat-file -t` = commit | ✅ |

---

## ❌ / ⚠️ / ❓ list with locations + recommended fixes

**No ❌. No ❓.** Two ⚠️ (both cosmetic; fix in the generator `make_pt4_transition_report.py`,
then rebuild):

- ⚠️ **[#19] midrange_pl0 ratio-min: 0.364 vs 0.365.** The `SEC_H1` table (generator ~line 254)
  renders midrange as `0.364`; the `SEC_H2` table (generator ~line 339) renders the *same*
  underlying value 0.36447 as `0.365`. **Fix:** in `SEC_H2`'s table row for `midrange_pl0`,
  change `<td>0.365</td>` → `<td>0.364</td>` (0.36447 rounds to 0.364). Optional: also correct
  `H2_rcloud_audit.md §5` table (it likewise shows 0.365) for source consistency. Impact: none
  on the thesis; pure presentation.

- ⚠️ **[#13] bigcloud enlargement factor (hygiene, NOT in the report).** The report's "5.2×" is
  correct (8.83/1.69 = 5.22). The stale estimate `~4.6×` lives only in the comment of
  `sc_bigcloud.param` ("~4.6x larger rCloud"). **Fix (optional, source hygiene):** update the
  `sc_bigcloud.param` comment to "~5.2x". The rendered report needs no change.

**Line-number cite drifts noted (in the audit docs, NOT the report) — no report fix needed:**
- `H2_rcloud_audit.md §3` and `r1shadow/DATA_NOTE.md` cite the `Eb<=0` collapse break at
  `run_energy_implicit_phase.py:1041`; current source has the check at `:1072` and the
  `termination_reason="energy_collapsed"` at `:1079`. The shadow site cited as `:1117` is now
  `:1166`. These are audit-doc drifts; the **HTML report cites no line numbers for these**, so
  the report itself is unaffected.

---

## Recommendation for the parent

The report is publication-clean. The single in-report fix is the one-character ⚠️ #19
(midrange 0.365 → 0.364 in the H2 table of `make_pt4_transition_report.py`, then
`python docs/dev/transition/pt4/make_pt4_transition_report.py` to rebuild, then recompose the
book). Everything else — equations, numbers, code cites, all 7 figures, and the narrative flow
from the "geometric, not thermal" neighbor chapter — verifies.
