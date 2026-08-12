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

**Status (2026-08-08):** 🔵 actionable — **294/294 arms ran, the three-way table is MEASURED, and it
SURVIVED the `main` merge** (`§14`: Θ₀ re-baseline 5 PASS / 1 FAIL, the one failure a window-length
artifact worth 1.3% on bench1 alone).
**f_κ is the worst of the three on both metrics** (it never even reaches the trigger θ = 0.95 on
bench1) and **P1 is falsified**. ⚠️ **`§2`'s ranking of f_mix over f_A is superseded twice:** `§11`
(instantaneous criterion → f_A/f_mix **tied**, 2.71× vs 2.64×) and then `§12` (stale-row exclusion
→ f_mix's solved-row spread degrades to **3.70×** while f_A holds **2.71×** — **f_A is the best
single knob on both axes**). `§10` (mechanism), `§11` (metric) and `§12` (staleness) are the three
sections that change what this campaign concludes; read them before quoting `§2` or `§11` alone.
G0 failed on a truncation artifact (`§1`, bounded in `§1a`).
**2026-08-03 — `§13`: the f_area successor is dead at the screen.** Its Phase A0 ran offline and
**GA0 failed** (A0.1 5/8, A0.2 0/8, A0.3 0/8, A0.5 0/8; only the viability check passed): the
combined knob reproduces f_κ alone on Ṁ (f^{2/7}, not the f^{1} area multiplication requires), so
the 514-arm bench8 campaign was **not submitted** and `§10d`'s two predictions are answered — in the
negative — for ~40 s of container time.
**2026-08-08 — three new sections.** `§14`: merging `main` (`3c090b7`) moved `trinity/` under all 294
arms — `PROVENANCE.md §1` gained a **CODE BASELINE** clause for it, and the Θ₀ re-baseline
**MEASURED 5 PASS / 1 FAIL**: θ(t) overlays on all three arms, and bench1's native failure is its
implicit window closing 3.4% early (matched-window PASS at 3.0e-4). Quote bench1's Θ₀ as moving
1.3%; nothing in `§1`–`§13` changes. `§15`: `§10b`'s mass-loading indictment names the **wrong
channel** — Lancaster's area law multiplies turbulent entrainment, which TRINITY does not represent
at all, while TRINITY's Ṁ is the conductive-evaporation eigenvalue — so **Option 2 is dead on a
second and independent ground**. `§16`: screening Option 3 found a sharper instrument than the
saturated-flux cap — Lancaster **Eq 10**, a closed-form ℓ-free Θ prediction — and it **fails 0/3**:
TRINITY's implied prefactor is 5–50× outside the order-unity bracket *and* drifts 3–6× across the
window. Read the other way, Eq 10 at TRINITY's own Ṙ_b/V_w predicts **Θ = 0.93/0.95/0.97, inside the
L21b band**, against TRINITY's resolved 0.29/0.44/0.58. `§16e` then records the maintainer's
wind-only objection: Lancaster has no radiation pressure, photoionized gas or SNe, which **widens**
the Eq-10 gap rather than explaining it, but devalues the precise multiple — and argues for
reframing Lancaster as a **component benchmark in the wind-only limit** rather than a shell
calibration target. ⚠️ Chasing that surfaced a separate `trinity/` problem in the momentum-phase
force budget, written up in **`docs/dev/momentum-pdrive/`** — it touches no Θ number here, but it
sits upstream of every fate.

---

## §1. [gate] G0 FAILED 2/11 — and the cause is truncation, not physics

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
- **Every arm that differs is a *truncated* arm** — one that stopped with **no `outcome` recorded**.
  Eight such arms; five completed normally in July.
- Of those eight, **four still have bit-identical `theta_max`** — they simply executed **fewer
  implicit steps** before being cut off (e.g. `bench1__fa128_diag`: 212 → 182 steps, θ_max identical
  to the last digit).

**So the G0 failure is a truncation artifact.** `bench1_m5e4_r20__fa128_diag` is the top point of
bench1's f_A ladder; band entry is log-interpolated between fa64 (Θ=0.864, below band) and fa128.
A shorter integration window lowered fa128's Θ_cum from 1.0241 → 0.9592, which slid the interpolated
crossing 74.8 → 83.2 and the spread 5.39 → 6.00.

> **⚠️ The consequence is bigger than the gate.** f_A's bench1 band-entry dose — and therefore f_A's
> **5.39× spread, the number the entire published head-to-head rests on** — is **not a converged
> measurement**. It is a function of where the run happened to stop. Two runs of identical code on
> identical params give 74.8 and 83.2.

### §1a. …but the truncation bias has a SIGN, so the number can be bounded without re-running

Added 2026-07-30 after the maintainer pointed out that `--time=1:30:00` was already the value in the
committed sbatch — i.e. **both runs used the same cap**, and no re-run at that cap changes anything a
deterministic code would do differently. That reframes the question from *"how do we complete this
arm?"* to *"what can we prove without completing it?"* — and the answer turns out to be: enough.

`bench1__fa128_diag` is cut off while **θ = 1.44**, far above its own running mean of Θ_cum = 0.959.
Θ_cum is the L_mech-weighted mean of θ over the window, so extending the window can only **raise**
it. **Every measured Θ_cum on a truncated arm is therefore a LOWER bound on the truth**, and the two
runs order exactly as that predicts: the longer window (July, 212 steps) gave the higher value
(1.024) and the shorter one (07-30, 182 steps) the lower (0.959).

Band entry is log-interpolated between fa64 (Θ = 0.864, **complete**, below the band) and fa128, and
falls monotonically as fa128's Θ_cum rises:

| Θ_cum(fa128) | 0.959 (07-30) | 1.024 (07-19) | 1.20 | 1.44 (θ at cutoff) | → ∞ |
|---|---|---|---|---|---|
| entry | 83.18 | **74.79** | 68.93 | 66.83 | 64.02 |

Since the true Θ_cum ≥ 1.024 and the crossing can never fall below fa64:

> **f_A bench1 entry ∈ (64, 74.8] → f_A spread ∈ (4.61, 5.39×].**

Three consequences. **(i)** The pre-registered G0 target of 74.8 is the correct **upper bound** — the
July number was closer to the truth than today's, and G0 "failed" against a value that was never
wrong, only unconverged. **(ii)** f_A's **Θ_cum** spread is at most 5.39×, so the *Θ_cum* ordering
f_mix > f_A ≫ f_κ holds across the entire bound and does not depend on finishing the arm.
**(iii)** The honest published form is a bound, not a point: **f_A Θ_cum spread ≤ 5.4×**.

> ⚠️ **Superseded in part by `§11` (same day).** Consequence (ii) is a statement about **Θ_cum only**,
> and Θ_cum turns out to be the wrong metric for ranking the knobs — TRINITY's trigger is
> instantaneous and memoryless. On the trigger's own criterion **f_A and f_mix are tied**
> (2.71× vs 2.64×). This whole subsection still stands as written *about Θ_cum*, and the f_κ part of
> the ordering is unaffected; only "f_mix > f_A" does not survive the metric change. The bound
> derivation itself is untouched.

**What "truncated" precisely means, and what is still inferred.** `outcome` is copied from
`metadata.json`'s `termination` block, which `trinity/_output/simulation_end.py` writes **only when
TRINITY itself decides to stop** (`stopping_time`, `shell_collapsed`, `shell_dissolved`, …). An empty
`outcome` therefore means exactly one thing, and it is a statement about the *process*, not the
physics: **the run ended without TRINITY ending it.**

What that does **not** tell us is *why*. Three candidates fit the evidence — a SIGTERM at the
walltime limit, a crash, or an OOM kill — and the downloaded artifacts cannot separate them, because
the exit codes live in `$WS/outputs/<campaign>/<arm>/.exit_code` on gpfs and were never shipped down.
**Walltime is the leading hypothesis, not a verified fact**: these are the slowest, stiffest arms
(`§15h` documented this exact class as a stiffness stall, where steps shrink until the run makes no
useful progress), and 294 concurrent arms is a different contention regime from July's 60. ⚠️ Until
someone reads `.exit_code` (a `124` would confirm it), this attribution is **inferred**. One command
settles it:
`cat $WS/outputs/bench7/*/.exit_code | sort | uniq -c`

**P4 is the control that rules out the alternative** (§5): 5/5 K3 pairs are **bit-identical**,
including the pair that truncated — both truncated at the *identical* step. So within one submission
the code is perfectly reproducible, and the July↔July-30 difference is **not solver nondeterminism**.
Something outside the physics changed how far these arms got.

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
| **f_A** | 13.88 | 53.51 | **(64, 74.8]** | **(4.61, 5.39×]** | bench1 bounded, not point-measured (§1a) |
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

**Ranking, on measured evidence: f_mix (2.75×) > f_A (≤5.39×) ≫ f_κ (≥16×).** The f_A entry is a
bound (§1a), and the ranking holds across the whole of it — no re-run can change the order. f_κ is closed as a
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

**Newly known to be soft.** f_A's bench1 entry is not a point measurement — it is the **bound
(64, 74.8]**, giving **f_A spread ≤ 5.39×** (§1a). Quote the bound. The pre-registered 74.8 is its
correct upper end, so nothing published on the f_A side was *wrong*; it was unconverged and stated
as if converged. `pl2_steep`'s firing-window left edge is the one thing still genuinely unmeasured
(f_κ = 5, 6, 7 all truncated) — but the non-overlap with `simple_cluster` holds regardless, so P3 is
unaffected.

---

## §9. Next

1. **Do NOT re-run the truncated arms to fix f_A — §1a already bounds it**, and the bound is tight
   enough that the ranking cannot change. `--time=1:30:00` was already the committed value for both
   runs, and P4 proves the code is deterministic, so a re-run at the same cap is only worth anything
   if the arms were **wall-killed** *and* node contention (294 arms vs 21) was the binding
   difference. **Read the exit codes first** —
   `cat $WS/outputs/bench7/*/.exit_code | sort | uniq -c`. `124` ⇒ a contention-free retry is worth
   one shot; anything else ⇒ deterministic, and re-running reproduces it exactly. Either way it is
   now a *nice-to-have*, not a blocker.
2. **Extend the f_κ grid past 32 on bench1/bench2**, or accept "does not reach the band by 32" as the
   final answer. Given bench2's saturation at 0.890, the second is defensible and much cheaper.
3. **Settle Q3** (frozen-row Θ_cum convention) — it now visibly moves numbers.
4. A **matched-`t` back-reaction harness** for P2, replacing the coarse log-t nearest-neighbour match.

---

## §10. [physics] Where each knob actually acts, and why NONE of them is the wrinkled-interface knob

Raised by the maintainer 2026-07-30: *"which one is more physically true? what about the mass loading
due to extra surface area from wrinkles (which is the whole motivation of adding the extra cooling)?"*
That question turns out to discriminate the three knobs far more sharply than the Θ_cum calibration
does, and it changes what the campaign's "winner" means. Figure: `bench7_massloading.png`.

### §10a. What the code actually does — verified at the call sites, not from memory

| knob | site | what it multiplies | does the bubble *structure* respond? |
|---|---|---|---|
| **f_mix** | `phase1b_energy_implicit/get_betadelta.py:360` | `L_loss = L_leak + f_mix·L_cool` — applied to the **integrated output** of the structure solve, feeding the β/δ residual and the energy equation | **No.** The structure ODE never sees it. T(r), ρ(r), Ṁ are identical to the unboosted run at the same state. |
| **f_A** | `bubble_structure/bubble_luminosity.py:435` | `dudt = f_A·dudt` **inside the ODE**, and only where `T < _T_INTERFACE_BAND`; plus L2+L3 in the integrals (`:845`) | **Yes** — the radiative source term is perturbed, so T(r) reorganises. The in-code comment is explicit: *"Conduction, ICs and the dMdt seed untouched."* |
| **f_κ** | `bubble_structure/bubble_luminosity.py:441` (also `:304`, `:398`) | `C_thermal` in `dTdrr` — the **Spitzer conduction coefficient** | **Yes**, and deepest: it changes how heat is *transported*, so the whole conduction zone restructures. |

⚠️ **Correction to an earlier statement in this conversation.** It is **f_A**, not f_mix, that acts on
`dudt`. And "f_mix is frozen by construction" was too strong: f_mix *is* frozen at the **structure**
level, but it still enters the global energy equation, so E_b, P_b and R2 do respond. The precise
statement is *structure-frozen, energetics-live*.

### §10b. The wrinkling picture makes a prediction, and it is about Ṁ

The physical motivation for all of this is that turbulent mixing **wrinkles** the contact
discontinuity, so its true area exceeds the 1-D spherical area by some factor `f_area`. In the
thin-layer limit every interface flux is proportional to that area **together**:

- conductive flux through the interface ∝ A
- **evaporative mass flux Ṁ ∝ A**  ← the channel nobody was checking
- radiative loss from the interface layer ∝ A (layer volume at fixed thickness)

So the signature of an area-faithful knob is unambiguous: **Ṁ must RISE with dose.** Measured:

| knob | Ṁ(f)/Ṁ(1) | verdict vs the wrinkle picture |
|---|---|---|
| **f_mix** | ≡ 1 (structure untouched) | ❌ no mass-loading response at all |
| **f_A** | **0.988 → 0.855** over f_A 2→32 (`theta5s_dmdt_suppression.csv`, 9 configs) | ❌ **wrong sign** — more interface cooling ⇒ cooler interface ⇒ *less* evaporation |
| **f_κ** | **1.066 → 1.075 → 1.011 → 0.944 → 0.799 → 0.289** over f_κ 2→3→6→8→12→32 | ⚠️ **right sign only below f_κ ≈ 7** |

**f_κ's sign flips inside the useful range.** At fixed state it follows Eq 47's `f^{2/7}` to within
1.6% (`§24` Q1) — the correct area-like behaviour. But on a **full run** the back-reaction takes over:
the boosted arm radiates more, drains E_b, pressure falls, and evaporation falls with it. The ratio
crosses 1 between f_κ = 6 and 8, and by f_κ = 12 — which is *bench3's own band-entry dose* — mass
loading is already **suppressed by 20%**.

> **So in the dose range where any of these knobs would actually be calibrated, not one of them
> raises mass loading.** The knob whose *whole motivation* is extra interface area produces, at the
> operating point, an interface that evaporates *less*. That is a physics-level indictment that the
> Θ_cum calibration cannot see, because Θ_cum only scores the radiative bookkeeping.

### §10c. Which is "more physically true"

Ranked by faithfulness to the mechanism rather than by calibration score — the reverse of `§2`:

1. **f_κ** — the only one that moves a real transport coefficient, and the only one with the correct
   Ṁ sign anywhere. Its failure is that it moves **only** the conduction channel, so the radiative
   loss rises only indirectly; that is why its Θ_cum exponent is q ≈ 0.28 and why it saturates below
   the band (`§2`, `§3`).
2. **f_A** — in-solve and physically interpretable as an emissivity/efficiency boost in the interface
   band, but it moves the θ-channel **at the cost of** the Ṁ-channel. Under the wrinkle picture those
   two should move together; here they anti-correlate.
3. **f_mix** — a scalar on the integrated answer. It cannot be wrong about the structure because it
   never consults it. **It wins the calibration (`§2`: 2.745× spread, all in-grid) precisely because
   it is unconstrained by the physics it is meant to represent.**

**That inversion is the real result of this campaign.** The measurement ranking (`§2`) and the
mechanism ranking (here) are *opposite*. Reporting either alone would be misleading.

### §10d. The experiment this implies — an area-faithful knob, and it is nearly free

`pdv-trigger/PLAN.md` item 9 already registered "the area-faithful successor" as the discriminating
experiment. `§10b` now supplies the missing quantitative case *and* a concrete construction:

> **f_area applies f_κ and f_A simultaneously with ONE shared constant.** f_κ carries the
> conduction + evaporation channel; f_A carries the radiative channel. Together they are the closest
> thing the current code can express to "multiply the interface area".

Two falsifiable predictions, from the measured single-knob exponents:

- **Ṁ:** f_κ contributes ≈ `f^{+0.28}` and f_A ≈ `f^{-0.05}` (from 0.988→0.855 over a 16× dose), so
  the combination should keep **Ṁ rising** — net ≈ `f^{+0.23}` — instead of crossing below 1 near
  f ≈ 7. **This is the test of whether the code can represent a wrinkled interface at all.**
- **Θ_cum:** both channels push θ up, so the combined exponent should **exceed** either alone
  (f_κ 0.28, f_A 0.25 ⇒ expect ≳ 0.4, possibly approaching f_mix's 0.46–0.56 — but from
  *mechanism* rather than from a scalar on the output).

**It is unmeasured, and deliberately so:** `0 of 174` bench7 arms set more than one knob — the
campaign enforced single-knob by construction so every effect could be attributed. That was right for
attribution and it is exactly why the combination is now the open question.

**Cost: small.** A dual-knob ladder on the three clean benches — f_area ∈ {2,4,8,12,16,24}, prod+diag
— is **36 arms**, ~12% of what this campaign just spent. It needs one change: `make_kappa_reopen_params.py`
would have to emit an arm setting `cooling_boost_kappa` *and* `cooling_boost_fA`, and
`test_bench7_params.py`'s single-knob assertion would need a documented exemption for that phase.

### §10e. What this does NOT change

Every number in `§1`–`§9` stands — this section re-reads them, it does not revise them. f_κ remains
closed *as a single-constant calibration knob* (it does not reach the band; ≥16× spread). The point
here is narrower and sharper: **it was closed on the calibration axis while being the least wrong on
the mechanism axis**, and the knob that won the calibration is the one that models the mechanism
least. Any recommendation that quotes `§2` without `§10` is quoting half the evidence.

---

## §11. [metric] Θ_cum is the wrong metric for the knob decision — and on the right one, f_A and f_mix are TIED

Raised by the maintainer 2026-07-30: *"why are we looking at Θ_cum? Only the instantaneous
L_loss/L_gain matters — the bubble goes into momentum the moment cooling beats gain, it doesn't care
about the history."* That is correct about the code, and acting on it **changes §2's headline.**

### §11a. The trigger has no memory — verified in source

`phase1b_energy_implicit/run_energy_implicit_phase.py:1250` fires on

```
(L_gain − L_loss) / L_gain  ≤  phaseSwitch_LlossLgain   (default 0.05)   ⟺   θ ≥ 0.95
```

evaluated **per step**, on the current state. Nothing integrates. A cloud fires the instant θ first
crosses 0.95, and `θ_max ≥ 0.95` is therefore *exactly* the statement "this cloud fires".

Θ_cum, by contrast, is the L_mech-weighted **mean of θ over the whole blowout window**. It is a
different object, and it can be high while θ never crosses the threshold (a long warm interval), or
low while θ spikes over it briefly (a short sharp peak). **The two metrics can disagree, and here
they do.**

### §11b. Why Θ_cum was used anyway — it is not a mistake, it is a mismatch

Θ_cum is the **right** metric for its actual purpose: Lancaster 2021b measures a *cumulative radiated
fraction* over the breakout window, so reproducing L21b requires an integrated quantity. `§15h`
adopted it for exactly that, correctly.

The error was **carrying a Lancaster-comparison metric into a knob-selection decision.** Those answer
different questions:

| question | right metric |
|---|---|
| does TRINITY reproduce L21b's **energy budget**? | **Θ_cum** — integrated, matches the observable |
| what dose makes a cloud **fire the trigger**? | **θ_max** — instantaneous, matches the code |

The workstream's own `runs/README.md` 📏 **rule 3** already says *"θ is reported as `theta_max` over
the whole run"*. The band-entry calibration drifted off that rule when it borrowed Θ_cum from the L21b
comparison, and nobody re-checked whether the ranking survived the swap. It does not.

### §11c. The instantaneous calibration — `table=TRIGGER` in `bench7_analysis.csv`

Dose at which θ_max first reaches **0.95**, PROD arms (the ones running the live trigger):

| knob | bench3 (n̄=5520) | bench2 (690) | bench1 (43) | **spread** | Θ_cum spread, for contrast |
|---|---|---|---|---|---|
| **f_mix** | 3.00 | 6.41 | 7.93 | **2.64×** | 2.745× |
| **f_A** | 11.3 | 23.5 | 30.8 | **2.71×** | ≤5.39× |
| **f_κ** | 6.53 | 13.8 | **never** | — | ≥16× |

**f_A and f_mix are tied: 2.71× vs 2.64×, a 3% difference.**

> **f_A's apparent 2× disadvantage in `§2` is an artifact of the metric.** On the criterion TRINITY
> actually uses, the two knobs are indistinguishable as single constants. `§2`'s "f_mix > f_A" must
> not be quoted as a physical result — it is a statement about Θ_cum, not about firing.

Three further points, all favouring this metric for this decision:

1. **f_κ's failure is robust and gets *worse*.** Under Θ_cum it "reached the band on bench3 and
   saturated below it elsewhere"; under the trigger it **never fires bench1 at any dose ≤ 32**. The
   `§2` conclusion that f_κ is closed stands — strengthened, not weakened.
2. **The absolute doses differ by ~4× between the two knobs** (f_mix 3–8, f_A 11–31) even though the
   spreads match. Uniformity therefore *cannot* choose between them; only the mechanism (`§10`) or an
   independent constraint can.
3. **It is far less exposed to the truncation problem of `§1`.** θ_max is a maximum over whatever
   ran, so an arm cut short after its peak is unaffected; and the f_A and f_mix trigger tracks contain
   **zero** truncated arms (only f_κ/bench3 has any). The metric that broke G0 is the integrated one.

### §11d. What this revises, and what it does not

**Revised.** `§2`'s ranking of f_mix over f_A. The correct statement is: **on the trigger criterion
f_A ≈ f_mix (2.71× vs 2.64×); on the L21b energy-budget criterion f_mix is better (2.745× vs
≤5.39×).** Both are true, of different questions, and the knob decision should rest on the first.

**Unchanged.** Everything about f_κ (`§2`, `§3`, `§4`) — it loses on both metrics. `§10`'s mechanism
analysis, which was never metric-dependent. P3/P4/P5/G6. `§1`'s truncation diagnosis.

**Newly sharpened.** With f_A and f_mix tied on uniformity, the decision falls entirely to `§10`'s
mechanism axis — where f_A acts in-solve and f_mix does not. **The two axes no longer conflict for
f_A vs f_mix: uniformity says "tied", mechanism says "f_A".** The conflict that remains is only that
neither has the right mass-loading sign, which is what the `f_area` experiment is for.

### §11e. Follow-up

- Report **both** metrics in any published table, labelled by the question each answers. Never one
  alone — that is exactly how this drifted.
- The pre-registered gates G4/G5 should be extended to require the trigger metric alongside Θ_cum;
  they currently only constrain the integrated one.
- `data/make_bench7_analysis.py` now emits `table=TRIGGER` alongside `table=ENTRY`, so both are
  regenerated together and cannot drift apart again.

---

## §12. [metric] "Frozen" is three different things — one is harmless, one is bounded, and one just broke §11's tie

Raised by the maintainer 2026-07-30: *"is the frozen thing doing us more harm than good?"* Answer:
"frozen" names three distinct things in this record, and they deserve three different verdicts. The
third one, measured fresh here, **changes a §11 conclusion**. Artifact:
`data/bench_stale_segments.csv`, regenerated on the ALL-FRESH harvests with two new columns
(`thetamax_row_stale`, `theta_max_solved`).

### §12a. Frozen-state SCREENS (the methodology) — good as falsifiers, harmful as forecasters

The record's receipt is P1: the fixed-state L_cool exponents (0.586/0.669, `§24` Q1) were carried
into a full-run prediction and missed by ~5× in spread (3.4× predicted, ≥16× measured), because the
integrated exponent is half the frozen one (`§3`). The same pattern is older: the 2026-06-25 frozen
f_mix screen "bounds but does not forecast" (`runs/README.md` §9), and blowout-θ was retired for a
2× under-read. **But the frozen screens' falsifications have all held** — P0's edge check, SC-0's
three kills, the Eq-47 sign verification. Verdict: **keep frozen screens, scoped as falsifiers and
sign-checks only; never quote a frozen number as a full-run calibration.** (CLAUDE.md rule 5, now
with three independent confirmations.) The f_area plan's Phase A0 is scoped exactly this way — with
one principled exception: the layer-invariance identity is *itself* a per-call statement, so a
frozen state tests it in its own regime.

### §12b. TRUNCATED arms (wall-clock) — harm bounded

Covered in `§1`/`§1a`: the bias has a sign, the affected number became a bound, no conclusion
depends on it. Closed.

### §12c. Frozen NO-ROOT rows — real harm to Θ_cum, and one material bite on the trigger metric

**What a stale row is.** On a no-root β–δ segment the solver leaves `bubble_props`/`bubble_Lloss`
frozen at the last accepted values but keeps logging rows (`run_energy_implicit_phase.py:893/:929`).
Offline signature: `Lcool` repeats **bit-identically**. Crucially, θ = L_loss/L_mech keeps *moving*
on stale rows — L_mech evolves under the frozen numerator — so staleness is invisible to a
duplicate-θ check and **can inflate θ_max when L_mech falls**.

**Measured fresh (291 arms):**

| quantity | fresh measurement |
|---|---|
| Θ_cum stale share, band-setting diag arms | **30–65%** (bench3 fa16 64.7%, bench2 fa64 61.6%, bench1 fa128 57.5%, k4 fm12 53.9%, k1 fk12 49.5%) |
| Θ_cum stale share, all boosted diag arms | median 9.3%, max 100% |
| Θ_cum stale share, baselines (dose 1) | median 0.0% — staleness is **dose-induced** |
| arms whose θ_max row is itself stale | **76/291** — overwhelmingly dense benches, top-of-grid doses, and K2 band configs |

**Does it change conclusions?** Recomputing every trigger band entry with **solved-only θ_max**:

- 8 of 9 clean-bench entries move in the **4th decimal** — noise.
- **One moves materially: bench1 f_mix 7.93 → 11.11.** Its fm8 bracket arm peaks on a stale row
  (solved-max 0.927 < 0.95), so the crossing slides from the fm4–fm8 bracket to fm8–fm12.
- **K2 fire map: 2/66 FIRED labels are stale-dependent** (`large_diffuse fk16`, `midrange fk4`
  → NOFIRE on solved rows). Neither sits in the best-dose set {8, 9, 12}, so `§4`'s P3 (no
  whole-band f_κ, best 5/6) is **unchanged**.

> **⚠️ Supersedes `§11` in part: the f_A–f_mix "tie" does not survive stale exclusion.**
>
> | trigger-metric spread | stale rows counted (`§11`) | solved rows only |
> |---|---|---|
> | f_mix | 2.64× | **3.70×** (3.00 / 6.41 / 11.11) |
> | f_A | 2.71× | **2.71×** (unchanged — every f_A bracket peaks on a solved row) |
> | f_κ | never fires bench1 | never fires bench1 |
>
> Which convention is *right* depends on the question. The live trigger genuinely evaluates on
> no-root segments (it reads the frozen `bubble_Lloss`), so counting stale rows answers "would the
> code fire" — but that path is also exactly where the `§16` f_mix double-boost bug lives (`§21`
> showed a rosette fm4 fire was bug-dependent), so the stale-row f_mix advantage is entangled with
> a known defect. For **knob comparison**, the physics-only (solved) convention is the defensible
> one — and under it, **f_A is the most uniform knob outright (2.71× vs 3.70×), which now agrees
> with `§10`'s mechanism ranking.** The two axes no longer disagree about f_A vs f_mix at all.

**This also effectively resolves open question Q3** (exclude stale rows vs carry an uncertainty
band): for the *trigger* metric, exclusion is cheap, changes exactly one bracket, and is the
physically defensible convention; for **Θ_cum** the stale share is so large (30–65%) that no
convention rescues it as a knob-selection metric — which independently re-confirms `§11`'s
demotion of Θ_cum to the L21b energy-budget comparison only, always with the stale share printed
beside it.

**Net verdict on "the frozen thing":** the screens are good when used as falsifiers (and the f_area
A0 uses them that way); the truncation is bounded; the no-root staleness was doing quiet,
measurable harm — to Θ_cum massively, and to exactly one trigger bracket, whose correction breaks
the f_A/f_mix tie **in f_A's favour**. Follow-up: fold `theta_max_solved` into
`make_bench7_analysis.py`'s TRIGGER table so both conventions regenerate from one builder.

---

## §13. [falsified] The f_area successor dies at its own screen — TRINITY's Ṁ is a Weaver eigenvalue, and f_A cannot reach it

`§10d` proposed the discriminating experiment: apply f_κ and f_A at **one shared constant** and see
whether the combination raises mass loading — the signature no single knob shows.
`F_AREA_PLAN.md` derived it into an identity (its §2.2: at f_κ = f_A = f the interface layer's
T-profile is invariant and every flux scales by exactly f), pre-registered five offline checks
A0.1–A0.5 and six full-run predictions PA1–PA6, and sized a 514-arm bench8 campaign behind gate GA0.

**A0 ran 2026-08-03 and GA0 FAILED.** Builder `data/make_farea_screen.py` (~40 s in-container, 30
production solves at the two committed captured states, zero `trinity/` changes); artifacts
`data/farea_screen.csv` + `farea_screen.png`.

| check | bar | result |
|---|---|---|
| A0.1 T-profile invariance ranking | combined below κ-only at every f | ❌ **5/8** — stiff 3.2–6.6% below (pass ×4), mild 0.02–0.27% **above** (fail ×3) |
| A0.2 Ṁ superadditivity | combined above κ-only at every f | ❌ **0/8** — combined is 0.02–1.2% **below**, both states |
| A0.3 L_total linearity | within ±30% of 1 + s(f−1) | ❌ **0/8** — measured is **+41% … +446%** above the model |
| A0.4 viability ceiling | combined healthy to f ≥ 8 | ✅ **2/2** — healthy to f = 16, residual ≤ 2.6e-6 |
| A0.5 anchor invariance | combined below κ-only at every f | ❌ **0/8** — and both anchors move as **f^{5/7}** |

**The single measurement behind all four failures.** TRINITY's Ṁ is not set by the interface layer's
local enthalpy balance — it is the **Weaver v(R1) = 0 eigenvalue over the whole R1 → r₂′ domain**,
and it tracks the Eq-33/Eq-47 conduction scaling **f^{2/7}**: fitted q = **0.279–0.285** for the
combined knob against 0.283–0.285 for f_κ alone, versus the f^{1.0} that area multiplication
requires. f_A never reaches it (per-call q = −0.0006 … −0.0013). Because Ṁ ∝ f^{2/7} rather than f,
the Eq-44 front anchor `dR2 ∝ f_κ/Ṁ` grows as **f^{5/7}** — the layer is **7.3× thicker at f = 16**
instead of standing still — so its internal structure cannot be invariant either (max|ΔT/T| ≈ 0.55
at f = 16, within a few per cent of f_κ alone). The f_A/f_κ cancellation in the radiative source
term is real, and is exactly the 3–7% by which combined beats κ-only on the stiff state; it is a
second-order correction riding on the anchor motion, not a cancellation of it.

**So `§10d`'s construction is answered, in the negative and for ~40 s.** The combined knob's Ṁ
ladder *is* rising (×2.17 at f = 16) — but by no more than f_κ alone delivers, so nothing moves the
full-run crossing below 1 at f_κ ≈ 7 that `§10b` measured, and there is no mechanism left by which
the pair could beat the single knob. ⚠️ Per `§12a` that last clause is a *scope* statement, not a
forecast: A0 cannot decide a full-run question, and **PA1–PA6 are recorded NOT SCORED** — A1 was
never submitted, and they are not reinterpreted against A0 numbers.

**A second result falls out, and it sharpens `§10b`.** f_A's Ṁ suppression is **not structural**.
Per call it is **−0.04% at f_A = 2 and −0.35% at f_A = 16** (stiff; −0.01% / −0.16% mild), against
the full-run ratios `§10b` measured — 0.988 at f_A = 2, i.e. **−1.2% at the same dose**, falling to
−14.5% by f_A = 32. An order of magnitude larger, so the full-run suppression must be dominated by
the same **integrated back-reaction** that decays f_κ's Ṁ ratio (`§7`/P2: radiate more → drain E_b
→ P_b falls → evaporation falls), not by an instantaneous restructuring of the interface.
⚠️ The two measurements are on different configs (two captured states here, nine theta5s configs
there), so read the factor as indicative of the mechanism, not as a calibrated ratio. `§10c`'s
ranking is unaffected; its *reason* for placing f_A second is now more precisely attributed.

**Two housekeeping results.** (i) The plan's §9 warm-start hazard — "the seed scales f^{2/7} while
the layer-invariant root scales ≈ f, so it will undershoot" — **does not hold**: the root scales
f^{2/7} too, Ṁ/seed is flat to 1.9% over a 16× dose, and no solve stranded. The one candidate
`trinity/` edit the plan gated behind A0 evidence (multiply the Eq-33 seed by f^{5/7}) is therefore
**withdrawn, not deferred** — it would move the seed ~7× off the root. (ii) The harness's own
f = 1 self-check earned its keep: requiring a profile to deviate from *itself* by exactly zero
exposed a duplicated radius in the production 60k grid at the T = 10^5.5 CIE switch, where
`np.interp` was resolving the tie arbitrarily and putting a spurious 1.2e-3 floor under every
deviation. Fixed in the builder; no verdict changed.

**What this changes.** Nothing in production (`cooling_boost_mode='none'`, f_κ = f_A = 1.0) and
nothing in `§1`–`§12`. What it closes is the last loophole in `§10b`'s indictment: **no combination
of the three shipped knobs raises mass loading**, and the zero-code route to an area-faithful knob
does not exist. The pre-registered TERMINAL clause (`F_AREA_PLAN §7`) is *not* triggered — it is
defined on PA3 + PA1, which are A1 reads. The live decision is `F_AREA_PLAN §11` item 4: stop with
the existing knobs, or carry the area factor explicitly on the evaporative flux itself
(`bubble_luminosity.py:304`/`:398`) — a `trinity/` change, so SC-1 wiring, needing a maintainer nod
and a derived value (`F_AREA_PLAN §3.3`) before it is even a candidate. **Nothing has been started
on it.**

---

## §14. [provenance] The 2026-08-08 `main` merge moved the code under all 294 arms

`main` was merged into this workstream's branch (`feature/threeway-pt2`, merge `3c090b7`, 46
commits), landing the `phase1a-init` fix. Two changes alter full-run trajectories: the
`vd = -1e8` early-phase override is **deleted** (`phase1_energy/energy_phase_ODEs.py`) and phase-1a
segments now scale with bubble age (`run_energy_phase.py`, new `phase1a_segFrac`, default `0.1`).
Every full-run number in `§1`–`§12` was measured at `1056c6d`, before both.

**No date-based rule catches this** — the arms are stamped 2026-07-30 and stay post-cutoff forever.
`PROVENANCE.md §1` therefore gained a **CODE BASELINE** clause, and `§4a` there records the event.

**The expectation, from the sibling workstream's own measurements** (⚠️ `docs/dev/phase1a-init/`,
not re-verified here): the shift is large early — at `t = 3e3 yr`, `simple_cluster` −10.4%,
`f1edge_hidens` −22.8%, GMC control −0.95% — and then converges, GMC ΔR2 reaching −0.002% at 1 Myr
and −0.001% at 2 Myr, with every config inside the adopted `|ΔR2| < 5%` bar (worst +0.44%). This
campaign's benches are GMC-scale at `stop_t = 5 Myr`, so the disturbed window is the first ~0.06% of
the integration and Θ_cum is L_mech-weighted across all of it. The expectation is therefore that the
three-way table survives intact.

**MEASURED 2026-08-08 — 5 PASS / 1 FAIL, and the one failure is a window artifact.** The three
`__none_diag` Θ₀ arms were re-run at `3c090b7`, in separate processes, and scored at G0's own bar
(abs 5e-4). Builder `data/make_merge_rebaseline.py`, artifacts `data/merge_rebaseline.csv` +
`merge_rebaseline.png`. The harness was validated *before* the re-runs by reproducing all three
committed Θ₀ exactly from `runs/data/bench5r_traj/*__none_diag.csv`, so the comparison is
independent of the outcome.

| arm | window | committed | re-run | abs diff | verdict |
|---|---|---|---|---|---|
| bench3_m1e5_r5 | native | 0.461806 | 0.461811 | 5.4e-06 | ✅ PASS |
| bench2_m1e5_r10 | native | 0.340860 | 0.340795 | 6.5e-05 | ✅ PASS |
| **bench1_m5e4_r20** | **native** | **0.220551** | **0.217703** | **2.8e-03** | ❌ **FAIL** |
| bench1_m5e4_r20 | matched | 0.217998 | 0.217703 | 3.0e-04 | ✅ PASS |

**bench1's failure is window length, not physics.** Its implicit phase now closes at t = 0.596 Myr
instead of 0.617 — 3.4% shorter — and Θ_cum is a running L_mech-weighted mean, so a window cut at
the high-θ end reads low. This is the identical mechanism `§1a` characterised for the truncated
arms, arriving here from a code change rather than a wall clock. On a **matched** window the two
trajectories agree to 3.0e-4, inside the bar, and θ(t) overlays visually on all three arms
(`merge_rebaseline.png`, top row).

> **So the trajectories survive the merge — but the number the campaign *quotes* for bench1 moves
> 1.3%.** Band entry goes as `(0.90/Θ₀)^{1/q}`, so at q ≈ 0.6 that propagates to roughly a 2% shift
> in bench1's entry dose. Small, bounded, and now measured rather than assumed. `§2`'s ranking is
> nowhere near that sensitive, so **no conclusion in `§1`–`§13` changes** — but the VERIFY tier
> lifts as a *bound*, not a clean reproduction: quote bench1's Θ₀ as moving under the merge.

⚠️ **A method note that cost a wrong verdict in the first pass.** The matched-window comparison must
**interpolate** its endpoint. A bare `t <= tmax` cut drops one endpoint row whenever the two windows
differ by less than a step, which manufactured a spurious 6.8e-3 FAIL on bench2; interpolating
returns 8.6e-5 PASS. The committed builder interpolates.

**`§13` (f_area A0) is exempt and already re-verified.** It is a per-call screen importing only
`bubble_structure/bubble_luminosity.py`, which the merge does not touch. Re-run post-merge it
returns the identical scorecard — A0.1 5/3, A0.2 0/8, A0.3 0/8, A0.4 2/0, A0.5 0/8, **GA0 FAILED** —
with 37 numeric fields drifting ≤9.2e-16 (~4 ULP) and no verdict moved.

---

## §15. [physics] `§10b`'s mass-loading indictment names the wrong channel — TRINITY has no entrainment at all

`§10b` is the section that condemned all three knobs, and `§13` closed its last loophole. Both
measurements stand. What is wrong is the **premise they are scored against**, and correcting it
changes which f_area option is worth pursuing.

### §15a. The premise imports Lancaster's area law onto channels Lancaster does not model

`§10b` argues that "in the thin-layer limit every interface flux is proportional to that area
**together**" — conductive, evaporative, radiative — and concludes that "the signature of an
area-faithful knob is unambiguous: **Ṁ must RISE with dose**."

Geometrically that is fine. But the area law being invoked is Lancaster's, and (verified against the
papers 2026-08-08):

- **Lancaster+2021a `§4.4` states plainly: *"we ignore thermal conduction."*** The fractal area
  `A_b(R_b; ℓ) = 4π α_A R_b² (R_b/ℓ)^d` multiplies the **turbulent-mixing enthalpy flux** (Eq 12's
  `v_equiv`), and the calibration target Θ = `L_int/Ė_in` is an **energy** ratio throughout.
- **El-Badry+2019 is the other lineage.** It models conduction — and states its cooling contribution
  is not significant. It is where TRINITY's evaporation physics comes from (Eq 47, `§23`).

So the two papers this workstream calibrates against each omit the other's mechanism, and **no
published treatment joins them**. That absence is what `F_AREA_PLAN §2.2`'s equal-dose identity was
quietly standing on, and it is why `§13`'s screen found nothing to stand on.

### §15b. The honest form of the indictment

TRINITY's Ṁ is the **Weaver conductive-evaporation eigenvalue** — `§13` measured exactly that
(`v(R1) = 0` over the whole `R1 → r₂′` domain, tracking `f^{2/7}`). Lancaster's wrinkled interface
loads mass by **turbulent entrainment**. These are different channels, and only one of them is in
the code. So:

> **`§10b` as written:** *no shipped knob raises mass loading at the operating doses.*
> **Corrected:** *TRINITY represents only the conductive-evaporation mass channel; the entrainment
> channel whose area Lancaster's law describes is absent. No knob can be an area knob, because the
> object the area belongs to is not represented.*

That is an indictment of the **framework's representational scope**, not of the knobs' calibration
quality — and it is the same conclusion `§13` reached from measurement, arrived at independently
from the literature. Every number in `§10b` stands; only what they convict changes.

### §15c. It softens `§10c`'s ranking of f_mix, and hardens the case against Option 2

**f_mix.** `§10c` ranks it last — "a scalar on the integrated answer… it wins the calibration
precisely because it is unconstrained by the physics it is meant to represent." Under the corrected
reading that is too harsh. Lancaster's area factor multiplies the interface **cooling** rate, and
his Θ is `L_int/Ė_in`; `f_mix` multiplies `L_cool`. Its *action* is the closest of the three to
Lancaster's *action*. What it misses is the mass entrainment carries — so the precise statement is
**f_mix buys Lancaster's energy consequence of extra area without the mass consequence**, which is a
nameable, bounded limitation rather than a disqualification. ⚠️ It still loses on the axis that
decided the campaign: `§12` measured its solved-row spread degrading to 3.70× against f_A's 2.71×.
This subsection re-reads `§10c`; it does not reorder `§2`, `§11` or `§12`.

**Option 2 is dead.** `F_AREA_PLAN §5a` item 2 — carry `f` explicitly on the evaporative flux at
`bubble_luminosity.py:304`/`:398` — is the option the plan calls "the only one that keeps the area
program alive". It now fails on **two independent grounds**:

1. **No derivable value** (`F_AREA_PLAN §3.3`, pre-existing): Lancaster's own truncation closure
   gives `ℓ_cool ~ 10⁻¹⁵ pc` and `f ~ 10⁹–10²⁴`.
2. **No warrant for the channel** (new, here): the paper supplying the area law explicitly excludes
   conduction. Putting the area factor on the evaporative flux would make Ṁ rise with dose —
   mechanically passing `§10b`'s test — while crediting the area excess to a mechanism the source
   theory does not contain. **Passing a test by construction is not evidence**, and SC-0 exists to
   refuse exactly that.

**Option 3 survives and should be promoted.** The Θ → 1 saturated-flux limit (Lancaster Eq 12's
`v_equiv` capped by Eq 15's `v_hot`) is the one live thread, and its appeal is precisely that it
**sidesteps `§3.3`**: if Θ saturates near 1 regardless of `ℓ`, the truncation scale that killed the
derivation stops mattering. It predicts a **ceiling, not a dose** — a derived quantity with no free
constant, which is what `§15k`/SC-0 ask for. It screens the way `§13` did: a closed-form comparison
at the committed captured states, offline, zero `trinity/` changes.

### §15d. Scope

This section measures nothing. It is a re-reading of `§10`/`§13` against a literature check, and no
number in `§1`–`§14` moves. ⚠️ The Lancaster/El-Badry claims here were verified against the papers
on 2026-08-08 but are **not** reflected in `pdv-trigger/LANCASTER_REFERENCE.md`, which predates them
and does not discuss the mass-channel distinction at all; that doc is in the demoted parent
workstream and was left unedited rather than updated in passing.

---

## §16. [falsified] Lancaster Eq 10 does not describe TRINITY — the ℓ-free prediction lands in the band, TRINITY resolves 3–5× less

`§15c` promoted f_area Option 3 on the argument that it **sidesteps `§3.3`** — the truncation-scale
problem that killed every earlier derivation. Screening it turned up a sharper instrument than the
saturated-flux cap that `F_AREA_PLAN §5a` item 3 names, and it was sitting in the repo's own
reference notes the whole time.

### §16a. Eq 10 is a closed-form Θ prediction with no ℓ in it

`pdv-trigger/LANCASTER_REFERENCE.md:203-204, 339` records Lancaster+2021a Eq 10, verified from the
paper excerpt:

```
1 − Θ = ( ½(1+f_turb)·α_p/α_R + S ) · (Ṙ_b/V_w)          (Eq 10),   S ≈ α_p within 6% (Eq 6)
```

**No `ℓ`. No fractal area. No fitted constant.** Every quantity in the prefactor is an order-unity
constant Lancaster measures — α_p ~ 1.2–4 (`F_KAPPA_FUNCTIONAL_FORM.md:139`), α_R ~ 1, S ≈ α_p,
f_turb the turbulent energy fraction — which brackets **C ∈ [1.8, 12]** across the corners. That
makes the prefactor a *measurable*: invert Eq 10 on TRINITY's own trajectory and ask whether the
implied C is order-unity. TRINITY carries both inputs directly (`v2` = Ṙ_b, `v_mech_total` = V_w =
2L_mech/ṗ, Lancaster's own Eq-1 definition).

### §16b. Measured: 0 PASS / 3 FAIL, by one to two orders of magnitude

Builder `data/make_merge_rebaseline.py` (`table=EQ10`), on the three arms re-run at the merge:

| arm | C median | C range | drift across window | Θ pred at C = 12 | Θ measured | verdict |
|---|---|---|---|---|---|---|
| bench1_m5e4_r20 | **59.3** | 22.3 – 125.2 | 5.6× | 0.932 | 0.288 | ❌ FAIL |
| bench2_m1e5_r10 | **74.8** | 32.0 – 136.8 | 4.3× | 0.950 | 0.436 | ❌ FAIL |
| bench3_m1e5_r5 | **92.8** | 45.9 – 150.4 | 3.3× | 0.965 | 0.579 | ❌ FAIL |

Two failures, and the second is the more damaging. **(i)** The implied C sits **one to two orders of
magnitude** above Lancaster's bracket on every arm and every row — not one point lands inside.
**(ii)** It **drifts 3.3–5.6× across the window**, where Eq 10 with fixed constants predicts a
constant. So the *functional form* misses, not merely the normalisation; no choice of α_p, α_R or
f_turb rescues it.

⚠️ **Quote the order of magnitude, not the multiple.** `§16e` shows the two sides of this comparison
do not contain the same physics (Lancaster is wind-only), which biases C *downward* here — so the
direction and rough size are robust but the precise factor is not. An earlier draft of this section
said "5–50×"; that was over-reading.

**Read the other way round, which is the result worth keeping.** At TRINITY's *own* Ṙ_b/V_w, Eq 10
evaluated at the generous end of Lancaster's bracket predicts **Θ = 0.93 / 0.95 / 0.97** — inside
the L21b calibration band [0.90, 0.99] the whole program is trying to reach — while TRINITY's
resolved structure delivers **0.29 / 0.44 / 0.58**. Lancaster's theory, applied to TRINITY's own
trajectory with no free parameter and no truncation scale, *already lands in the band*. TRINITY
misses it by a factor of 3–5 in Θ, ~2 in (1−Θ).

### §16c. What this does to Option 3 — two live readings, and the screen does not choose

1. **Eq 10 does not describe TRINITY ⇒ Option 3 dies with Option 2.** The prefactor is wrong by
   1–2 orders *and* has the wrong time dependence. On this reading `F_AREA_PLAN §5a` collapses to
   option 1: stop with the existing knobs, and write the campaign up as the negative result that
   the 1-D Weaver framework cannot carry a fractal interface.
2. **Eq 10 is the calibration *target*, and the band-entry program has been fitting around it.**
   The whole three-way campaign asks "what dose brings Θ_cum into [0.90, 0.99]?" — a band. Eq 10
   supplies a **pointwise Θ(t) curve, derived, with no free constant**. That is a strictly stronger
   target and it is the first one in this program that does not need ℓ. ⚠️ But `§16b`'s drift is the
   obstacle: a *scalar* dose cannot convert TRINITY's θ(t) into Eq 10's, because the discrepancy is
   3–6× larger at the end of the window than the start. Any knob that could is state-coupled by
   construction — which is `FA_STATE_COUPLED.md`'s territory, not this campaign's.

**The screen does not decide between them, and is not asked to.** It establishes the measurement
both readings need. ⚠️ Scope, per `§12a`: this is a per-row read of three baseline (f = 1) arms. It
measures no dose, no band entry and no spread, and it says nothing about how a *boosted* run would
track Eq 10.

⚠️ **Mapping caveats, stated rather than buried.** Lancaster's `R_b` is the hot-gas bubble radius
and TRINITY's `R2` is the contact/shell radius; Lancaster's Θ = `L_int/Ė_in` and TRINITY's
θ = `bubble_Lloss/Lmech_total` (`LANCASTER_REFERENCE.md §7b` treats these as the comparable pair,
which is inherited here, not re-derived). Both mappings are order-unity-faithful, not exact, so read
the 5–50× gap as robust and the exact factor as indicative.

### §16d. An unresolved contradiction between two repo docs, flagged not fixed

`F_AREA_PLAN §3.3` describes the saturation cap as *"Eq 12's `v_equiv` may not exceed Eq 15's
`v_hot`"*. `LANCASTER_REFERENCE.md:341` records the same equation as *"Eq 15 (`v_hot(R_b) ≈
V_w/(6α_p−2)`, which cannot exceed Eq 12's `v_equiv`)"* — **the opposite direction**. The saturated-
flux construction only works one way round (`v_equiv ≤ v_hot` is a flux limit; the reverse is a
lower bound and produces no saturation). Both docs are secondary; resolving it needs Lancaster
2021a Eq 15 itself. Not resolved here, and the Eq-10 screen above does not depend on it.

### §16e. ⚠️ Lancaster is WIND-ONLY — what that does to `§16`, and what it does to the whole program

Raised by the maintainer 2026-08-08, and it is the right objection: Lancaster+2021 Paper I/II
simulate **wind only** — no radiation pressure, no photoionized-gas pressure, no supernovae. TRINITY
carries all three. So is the Eq-10 comparison apples-to-oranges?

**Partly, and it cuts the opposite way from what you would guess.**

- **SNe are moot in this window.** Measured on these arms: `Lmech_SN/Lmech_W ≈ 7e-27` at the start,
  and the implicit window closes at 0.27–0.60 Myr — far before first SNe. That difference does not
  touch `§16`.
- **Radiation and photoionized gas make the gap WIDER, not narrower.** Both push the shell, but
  neither enters θ's denominator — `Lmech_total` is *mechanical* (wind+SN) power only. So TRINITY's
  Ṙ_b is faster than a wind-only bubble at the same L_mech. Since C = (1−Θ)/(Ṙ_b/V_w), an inflated
  denominator **deflates** C. The measured C = 59/75/93 is therefore a **lower bound** on what a
  wind-only TRINITY would show.

⚠️ **So the objection does not rescue Eq 10 — but it does devalue the exact factor.** Read `§16b` as
*"order-of-magnitude discrepancy, direction robust"*, **not** as *"the gap is 5–50×"*. Quoting the
precise multiple would be over-reading a comparison whose two sides do not contain the same physics.

**The deeper version of the objection is the one that should reorder the program.** Lancaster's Θ
measures the **wind bubble's** interface cooling. If radiation and photoionized gas dominate TRINITY's
*shell* force budget in this regime, then whether the wind bubble is energy- or momentum-driven is a
second-order determinant of shell evolution — and this campaign has spent 294 arms calibrating a
subdominant term. That is a measurable question, not a rhetorical one.

> **The reframing this argues for.** Stop treating Lancaster as a calibration target for TRINITY's
> **shell**. Treat it as a benchmark for TRINITY's **wind-bubble interface sub-model, in the wind-only
> limit where the benchmark applies** — validate the component in its own regime, then run full
> physics with the validated component. That dissolves the physics mismatch instead of arguing about
> it, and it is a cleaner methods story than *"we tuned a cooling multiplier until Θ_cum entered a
> band measured in a different physics setup"* — which is what the band-entry framing currently
> amounts to. It also offers a reason **why all three knobs failed**: they were being asked to
> reconcile a component model against a whole-system measurement.
>
> `include_PHII` is a real `.param` switch (default `True`), so a photoionization-off bench is one
> line. There is **no** `include_Frad`; killing radiation pressure would need the dust opacity zeroed.
> A wind-only bench is the honest configuration for `§16` and costs ~30 min of runs. **Not done.**

⚠️ **Trying to measure that force budget surfaced a separate, more urgent problem in `trinity/`** —
the momentum-phase `P_HII` equals the wind ram pressure to ≤3.6e-16, so the ODE's
`P_drive = P_HII + P_ram` evaluates to `2 × P_ram`. It is in the integrator, and it is written up in
its own workstream: **`docs/dev/momentum-pdrive/README.md`**. It does **not** touch any Θ number here
(Θ_cum integrates the implicit phase, which ends at the transition), but it does sit upstream of every
fate and stopping outcome, including K3's determinism arm. The force-budget question above cannot be
answered until it is resolved.
