# FINDINGS — the fresh record (cutoff 2026-07-29)

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

**Status (2026-07-30):** 🔵 actionable — **294/294 arms ran; the three-way table is MEASURED.** The
headline: **f_κ is the worst of the three knobs**, and P1 is falsified. One gate (G0) failed, and the
failure is understood.

---

## §1. [gate] G0 FAILED 2/11 — and the cause is wall-clock truncation, not physics

**Sources.** `runs/data/{bench5r,bench6r,bench7}_summary.csv` + `*_traj/`, all stamped
2026-07-30T19:01–19:02Z, `code 1056c6d`. 60 + 60 + 174 arms, none missing.

| quantity | pre-registered | fresh | verdict |
|---|---|---|---|
| Θ₀ bench3 / bench2 / bench1 | 0.462 / 0.341 / 0.221 | 0.461806 / 0.340860 / 0.220551 | ✅ PASS ×3 |
| f_A entry bench3 / bench2 | 13.9 / 53.5 | 13.8834 / 53.5130 | ✅ PASS ×2 |
| **f_A entry bench1** | **74.8** | **83.2428** | ❌ **FAIL** |
| **spread f_A** | **5.39×** | **5.99583×** | ❌ **FAIL** |
| f_mix entry bench3 / bench2 / bench1 | 4 / 8.16 / 11.9 | 4.01531 / 8.16293 / 11.8661 | ✅ PASS ×3 |
| spread f_mix | 2.96× | 2.95522× | ✅ PASS |

**The diagnosis, and it is the most important result of the campaign.** Diffing the fresh baselines
against the 2026-07-19 harvest arm by arm:

- **bench5r: 60/60 arms have BIT-IDENTICAL `theta_max`. bench6r: 56/60.** The code is deterministic
  and numerically unchanged. (`git diff 89e802dd HEAD -- trinity/` touches only two **info strings**;
  no numerical code moved.)
- **Every arm that differs is a *truncated* arm** — one that stopped with no `outcome` recorded,
  i.e. ran out of wall-clock mid-solve. Eight such arms; five completed normally in July.
- Of those eight, **four still have bit-identical `theta_max`** — they simply executed **fewer
  implicit steps** before being cut off (e.g. `bench1__fa128_diag`: 212 → 182 steps, θ_max identical
  to the last digit).

**So the G0 failure is a truncation artifact.** `bench1_m5e4_r20__fa128_diag` is the top point of
bench1's f_A ladder; band entry is log-interpolated between fa64 (Θ=0.864, below band) and fa128.
A shorter integration window lowered fa128's Θ_cum from 1.0241 → 0.9592, which slid the interpolated
crossing 74.8 → 83.2 and the spread 5.39 → 6.00.

> **⚠️ The consequence is bigger than the gate.** f_A's bench1 band-entry dose — and therefore f_A's
> **5.39× spread, the number the entire published head-to-head rests on** — is **not a converged
> measurement**. It is a function of where the wall-clock fell. Two runs of identical code on
> identical params give 74.8 and 83.2. Neither is *the* answer; the honest statement is
> **f_A spread ≈ 5.4–6.0×, wall-limited**.

**P4 is the control that makes this airtight** (§5): 5/5 K3 pairs are **bit-identical**, including
the pair that truncated — both truncated at the *identical* step. Within one submission the code is
perfectly reproducible. The July↔July-30 difference is therefore not solver noise; it is how much
wall-clock the stiff arms got, and 294 concurrent arms is a different contention regime from 60.

**G3 (compliance): 21/294 arms truncated (7.1%)**, 12 of them `_diag` arms — each one a poisoned
point in a dose–response track. All 21 are listed in `data/bench7_analysis.csv` (`table=ARMS`,
`truncated=YES`).

---

## §2. [result] THE THREE-WAY TABLE — f_κ is the WORST of the three knobs

`data/bench7_analysis.csv` (`table=ENTRY`), figure `bench7_entry.png`. L21b band [0.90, 0.99],
clean-blowout benches only.

| knob | bench3 (n̄=5520) | bench2 (n̄=690) | bench1 (n̄=43) | **spread** | in-grid? |
|---|---|---|---|---|---|
| **f_mix** | 4.02 | 9.69 | 11.02 | **2.75×** | ✅ all measured |
| **f_A** | 13.88 | 53.51 | 83.24 ⚠️ | **6.00×** ⚠️ | measured, bench1 wall-limited |
| **f_κ** | 10.43 | *169* | *143* | **≥16×** | ❌ **2/3 EXTRAPOLATED** |

*Italic = extrapolated past the grid, not measured.*

**The robust, extrapolation-free statement about f_κ** — this is what should be quoted, because it
needs no model at all:

| bench | Θ_cum at the top of the f_κ grid (f_κ = 32) | reaches the band? |
|---|---|---|
| bench3 (n̄=5520) | 0.913 | ✅ yes — entry **10.43, measured in-grid, clean** |
| bench2 (n̄=690) | **0.890** | ❌ no — and **saturating**: 24→32 moves it 0.889 → 0.890 |
| bench1 (n̄=43) | **0.676** | ❌ no — monotone but far short |

bench2's flattening is *not* a window artifact: its integration window is still **growing** over that
range (t_end 0.563 → 0.606) and every arm completed to `stop_t = 5`. f_κ appears to **asymptote just
below the band** on the intermediate-density cloud.

**Ranking, on measured evidence: f_mix (2.75×) > f_A (6.0×) ≫ f_κ (≥16×).** f_κ is closed as a
single-constant calibration knob — this time on a *measurement*, not on the falsified El-Badry-sign
argument that `§23` deleted. That correction stands regardless; it was always about honesty, never
about promoting f_κ.

**P5 CONFIRMED, and K4 earned its 24 arms.** f_mix now reaches the band **in-grid on all three
benches** (bench1 11.02 via fm12, bench2 9.69 via fm12). `§18`'s 2.96× was ⅔ extrapolated; the
measured value is **2.745×**. The head-to-head inversion is now a measurement, not an estimate.

---

## §3. [falsified] P1 MISSED — because the dose-response exponent is HALF what was assumed

P1 pre-registered `entry = (0.90/Θ₀)^{1/q}` with **q ∈ [0.55, 0.70]**, giving a predicted f_κ spread
of **2.9–3.8×, central 3.4×**. Measured: **≥16×**. A large miss, recorded as a miss.

The cause is not the functional form — it is the exponent. Fitting `Θ_cum ∝ f^q` on the clean points
(`table=EXPONENT`; truncated and early-dissolving arms excluded, since a shrinking window lowers
Θ_cum without any dose-response change):

| knob | bench3 | bench2 | bench1 |
|---|---|---|---|
| f_A | 0.252 | 0.246 | 0.307 |
| f_mix | 0.469 | 0.464 | **0.563** |
| **f_κ** | **0.277** | **0.273** | **0.318** |

**8 of 9 fits fall below P1's bracket.** f_κ's integrated exponent is **q ≈ 0.27–0.32 — roughly half**
the 0.586/0.669 fixed-state L_cool exponents `§24` Q1 measured. Since entry dose goes as `1/q` in the
exponent, halving q roughly *squares* the required dose: that is the whole gap between the predicted
3.4× and the measured ≥16×.

> **This is `CLAUDE.md` rule 5 measured again, in a new place.** A fixed-state, per-call exponent is
> **necessary but not sufficient** — it did not survive the integrated, full-run metric. `§24` Q1
> was not wrong; it was answering a different question than P1 assumed it did.

**A second, quieter result falls out of the same table:** f_mix's exponent (0.46–0.56) is roughly
**double** f_A's and f_κ's (0.25–0.32). That is *why* f_mix is the most uniform knob — it responds
faster per unit dose, so it needs less dose, and less dose means less room for the requirement to
spread across density. The uniformity ranking is a consequence of the exponent ranking, not an
independent fact.

---

## §4. [confirmed] P3 — the condensation squeeze is real, and now resolved to one dose unit

K2's 66-arm fire map, `table=FIREMAP`. **No single f_κ fires all 6 band configs.** Best is **5/6 at
f_κ ∈ {8, 9, 12}** — reproducing `§12`'s "5/6 at f_κ=12" *exactly*, from an independent, fully fresh
grid.

The squeeze itself, which the fine grid was built to resolve:

| config | f_κ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `simple_cluster` | NOFIRE | DRAIN | DRAIN | **FIRED** | **FIRED** | **FIRED** | CONDENSE | CONDENSE | CONDENSE | CONDENSE | CONDENSE |
| `pl2_steep` | NOFIRE | NOFIRE | NOFIRE | NOFIRE | *trunc* | *trunc* | *trunc* | **FIRED** | **FIRED** | **FIRED** | CONDENSE |

`simple_cluster` fires only at f_κ ∈ [4, 6] and condenses from 7 up; `pl2_steep` needs f_κ ≥ 8.
**The two windows do not overlap** — that is the squeeze, and it is now bounded to a single dose
unit. ⚠️ `pl2_steep` at f_κ = 5, 6, 7 truncated, so the exact left edge of its firing window is
unmeasured; it cannot be lower than 5, and the non-overlap holds regardless.

`§24`'s re-attribution is confirmed: the failures are **CONDENSE/DRAIN**, not NOFIRE — the band
breaks at the condensation boundary, not for lack of reach.

---

## §5. [confirmed] P4 — determinism, 5/5 bit-identical

| pair | fate | θ_max | trajectory sha256 |
|---|---|---|---|
| `be_sphere` @ 8 | shell_collapsed | 0.980247 | identical |
| `normal_n1e3` @ 16 | **truncated** | 0.915950 | identical |
| `pl2_steep` @ 16 | shell_collapsed | 0.578173 | identical |
| `simple_cluster` @ 8 | shell_collapsed | 0.532606 | identical |
| `small_dense_highsfe` @ 6 | shell_dissolved | 0.577899 | identical |

The non-monotonic fates are **physical**, not nondeterminism. Note the truncated pair truncated at
the *identical* step — reproducibility holds even for the failure mode, which is what licenses §1's
attribution of the July↔now differences to wall-clock rather than solver noise.

---

## §6. [confirmed] G6 — K4 and bench6r are one measurement

8/8 overlapping f_mix doses reproduce within 2%. The two campaigns may be quoted together; the
cross-campaign stitching risk the redo introduced did not materialise.

---

## §7. [partial] P2 — the back-reaction, extended past `§24`'s horizon

`table=BACKREACT`. Ṁ(f)/Ṁ(1) against Eq 47's `f^{2/7}`, along the full run:

| arm | f^{2/7} | error, first → last sample |
|---|---|---|
| `bench3 fk2` | 1.2190 | −12.6% → −13.7% |
| `bench3 fk4` | 1.4860 | −28.5% → −40.6% |
| `bench3 fk12` | 2.0339 | −60.7% → −95.2% |
| `bench3 fk32` | 2.6918 | −89.3% → −99.6% |

**Direction CONFIRMED and the magnitude is far larger than `§24` could see.** Q1b measured
−0.12% → −11.30% over `t ≲ 2.3×10⁻³ Myr` at f_κ=2; the full run at f_κ=2 lands at −13.7%, a clean
continuation. But the decay **steepens sharply with dose** — at f_κ = 32 the ratio collapses to ~1%
of the Eq-47 prediction. The C-channel is not failing; the bubble is paying for it, and at high dose
it cannot.

⚠️ **Quote this as directional only.** The comparison matches the boosted and unboosted trajectories
by nearest log-t within 0.05 dex, which is coarse where the sampling is sparse; the first-sample
values in particular are not directly comparable with `§24`'s t → 0 limit. A dedicated matched-`t`
harness is the follow-up.

---

## §8. What this changes, and what it does not

**Changes.** f_κ is closed as a single-constant calibration knob on measured evidence. f_mix's
band-entry advantage over f_A is now measured rather than extrapolated. P1 is falsified, and the
reason — the integrated exponent is half the fixed-state one — is itself a reusable result.

**Does not change.** No production behaviour and no default: `cooling_boost_mode='none'`, f_κ = 1.0,
f_A = 1.0. The `§23` Eq-47 correction stands. `§12`'s 5/6 and `§24`'s re-attribution are both
independently reproduced.

**Newly known to be soft.** f_A's bench1 entry and its spread are wall-limited (§1). Any future
quote of "f_A spread = 5.39×" should be **5.4–6.0×, wall-limited**, until the stiff arms are re-run
with enough walltime to complete.

---

## §9. Next

1. **Re-run the 21 truncated arms with `--time=3:00:00`** and re-reduce. That is the only way to turn
   f_A's bench1 entry — and `pl2_steep`'s firing window — from wall-limited into measured.
2. **Extend the f_κ grid past 32 on bench1/bench2**, or accept "does not reach the band by 32" as the
   final answer. Given bench2's saturation at 0.890, the second is defensible and much cheaper.
3. **Settle Q3** (frozen-row Θ_cum convention) — it now visibly moves numbers.
4. A **matched-`t` back-reaction harness** for P2, replacing the coarse log-t nearest-neighbour match.
