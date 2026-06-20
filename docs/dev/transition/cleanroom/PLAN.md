# Clean-room redo: implicit→momentum transition trigger — certify substrate, then characterize

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
> exists, revise the plan and note what changed and why (date it). Leave the plan
> better than you found it. **Keep all banner paragraphs at the top of every
> plan and analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (2026-06-20):** 🟡 **IN PROGRESS — C0 running.** C0.3 audit ✅; C0.2 harness built + run across the 6-config span (short `stop_t=0.05` done; full `stop_t=6` in flight); the δ↔dT0/dt check resolved as tautological (dropped, replaced by the solver T-residual); C0.1 (adiabatic Weaver null) staged. Emerging `f_ret`/β results are provisional (§2 Progress). No production change; no candidate trigger evaluated yet.
- **Type:** plan + pre-registration — a *clean-room* redo of the implicit→momentum transition-trigger investigation that deliberately does **not** inherit prior numerical results/rankings/verdicts (see §1).
- **Workstream:** `transition/` (clean-room subdir `cleanroom/`).
- **Where it sits:** supersedes-as-entry-point the **quarantined** `docs/dev/transition/{TRIGGER_PLAN,P0,pshadow-design}.md` and the background `docs/dev/archive/betadelta/*`. Those remain on disk as prior art; their *conclusions* are quarantined (§1), their *methodology and candidate menu* are reused.
- **Code it concerns:** the implicit-phase terminator `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095` (the `(Lgain−Lloss)/Lgain < threshold` trigger); the betadelta solver `trinity/phase1b_energy_implicit/get_betadelta.py`; bubble cooling `trinity/bubble_structure/bubble_luminosity.py`; the transition phase `trinity/phase1c_transition/run_transition_phase.py`.
- **Linked files & data:** harness `docs/dev/transition/cleanroom/c0_consistency.py`; data `docs/dev/transition/cleanroom/data/`.

---

## 0. Why a clean-room redo, and the trust boundary

**The actual problem (concrete).** TRINITY integrates a feedback bubble through
four phases: **energy (1a)** — the adiabatic, Weaver-like energy-driven start;
**implicit energy-driven (1b)** — the long energy-driven phase where the hot
shocked interior does PdV work on the shell while radiating, and the β–δ
(beta–delta) structure solver runs every segment; **transition (1c)** — the
hand-off where the interior has lost enough energy that thermal pressure stops
driving the shell; **momentum (2)** — the shell coasts on swept-up momentum until
it dissolves or stalls. A normal run walks energy→implicit→transition→momentum and
ends at a physical fate (dissolution at a few Myr, blowout, or the time cap).

The **implicit→momentum hand-off** is decided by a *single* criterion at
`run_energy_implicit_phase.py:1095`: switch to momentum once radiative cooling has
nearly caught up with feedback heating, operationalized as
`(Lgain − Lloss)/Lgain < 0.05` — the bubble radiates ≥95 % of the *instantaneous*
mechanical power being injected. `Lgain` is the instantaneous SB99 mechanical
luminosity; `Lloss` is the bubble's (pure radiative) cooling.

**The symptom.** Under the new default solver (`betadelta_solver=hybr`), this
trigger **never fires**: across all six test configs the ratio plateaus at
~0.3–0.5 and never approaches 0.05, so every run sits in the implicit phase until
it hits the 15 Myr `stop_t` cap (or `large_radius`) — **0/6 reach the transition or
momentum phases** (the legacy solver reached momentum in 6/6). Because the hybr
runs never complete a physical energy→momentum evolution, their late-time and
stopping-fate outputs are **not trustworthy as-is**.

**Why the trigger stops firing — two views of one effect.** (a) *The metric is
instantaneous.* Each time a new feedback source switches on (Wolf–Rayet winds, then
SNe), the `Lgain` denominator spikes, so the ratio jumps **up and away** from 0.05
exactly when cooling might otherwise catch up — it tests the instantaneous
numerator, not an integrated budget. (b) *hybr finds the true root.* Legacy clamped
the cooling parameter β to [0,1], so `Pb` could only ever decline → the ratio
drifted monotonically to balance and crossed 0.05. hybr is an unbounded root-find
with a physical `dMdt>0` gate; in some regimes it returns **negative β** (`Pb`
*rising* — the bubble re-pressurising under a feedback surge), keeping the interior
net-heating so balance is never reached. The 0.05 threshold *and the rate-ratio
metric itself* were calibrated for the clamped-β legacy solver and are not robust
to the more-correct hybr behaviour.

**Why it matters / the open question.** Two honest readings, both kept open
(see §0.1): the stall is *either* a transition criterion that needs replacing for
the hybr regime (a **trigger** problem), *or* a sign that TRINITY's cooling is
physically incomplete, so the bubble retains too much energy to ever balance (a
**physics** problem). This plan is built to decide which — against external data,
not against TRINITY's own assumptions.

**Why clean-room.** Prior efforts on this exist across branches/docs. Their
*ideas* are valuable; their *numbers, rankings, and verdicts* may be contaminated
(scratch `/tmp` captures, pre-hybr code, stale line refs). To produce results we
can trust and publish, we **re-blind, re-derive, regenerate**: keep the rigor
(shadow diagnostics, pre-registered gates, committed artifacts), discard the
verdicts.

**Trust boundary (maintainer decision, 2026-06-20): _certify, then build_.** We do
not assume the substrate (the hybr solver + the `Lgain`/`Lloss`/`Pb`/`Eb`/β/δ
machinery the trigger reads) is correct. We **certify it with a cheap, independent
gate (C0, §2) first.** If C0 passes, the substrate is trusted ground and the
investigation is scoped to the trigger. If C0 fails, the contamination reaches the
substrate and scope escalates. **Production stays frozen** (default
`instantaneous` trigger unchanged) — all work here is offline harness + docs +
committed CSVs; nothing under `trinity/` changes.

---

## 0.1 Epistemic stance — TRINITY/WARPFIELD are hypotheses, not ground truth

**Do not assume TRINITY, WARPFIELD, or Weaver is correct.** C0 (§2) certifies that
TRINITY correctly *implements its own equations* and reproduces the analytic
**Weaver (1977)** energy-driven limit. That is a bug-catch, **not** a validation of
the physics: Weaver is a model with strong assumptions (quasi-steady,
spherically-symmetric, conduction-mediated interior, *no turbulent mixing*), and
the modern literature argues it is **wrong for real star-forming clouds**. So the
transition criterion must be anchored to *independent* models and observations,
not to TRINITY's or WARPFIELD's internal choices.

**What the external literature says (consulted 2026-06-20; refs §8 — verify each
before citing in the paper):**
- **Efficient cooling ⇒ momentum-driven, low energy retention.** Lancaster et al.
  2021 (I & II) — turbulent mixing at a *fractal* bubble–shell interface radiates
  away the vast majority of wind energy; bubbles are **momentum-driven**, with
  velocities/pressures orders of magnitude below Weaver, and a **retained-energy
  fraction ~0.01–0.1 that decreases with time** — validated by 3D hydro over
  >3 dex in density.
- **Observations agree.** "The classical model in which ~half the wind energy stays
  in a hot bubble is not in agreement with observations" (Geen et al. 2021; Orion
  [CII], Pabst et al. 2020): stored energy ~**1%** of input wind energy; winds
  carry ~10% of the photoionization momentum.
- **SN superbubbles too.** El-Badry et al. 2019 — conduction+cooling at the
  interface remove a large, time-increasing fraction of injected energy.

**Consequence for the "stall" (independent reading — to test, NOT assert).**
TRINITY inherits the **energy-conserving** Weaver/Rahner interior and does **not**
model fractal mixing-layer cooling. If real bubbles radiate 90–99% of their energy
and go momentum-driven early, then a run that retains so much energy it **never
transitions in 15 Myr is physically suspect** — the stall may be a symptom of
**systematic under-cooling (a missing physics term), upstream of the transition
trigger**. You cannot fix missing cooling physics by tuning a threshold. Measure,
don't assume.

**External-anchor diagnostic (added to the harvest, §6).** Alongside the internal
`Eb`-peak oracle, compute TRINITY's **retained-energy fraction**
`f_ret(t) = Eb / ∫Lgain dt` per run and compare to the literature band (~0.01–0.1,
decreasing). Three outcomes, all informative:
1. `f_ret` enters the band and *then* the bubble transitions ⇒ the trigger question
   is well-posed; tune/replace the trigger.
2. `f_ret` stays far **above** the band (retains too much energy) ⇒ the stall is an
   **under-cooling** symptom; the finding is a *physics gap* (mixing-layer cooling,
   cf. Lancaster F5), and the trigger is secondary.
3. `f_ret` collapses far **below** the band ⇒ over-cooling / a different bug.

This anchors "is the model even in the right regime?" to data, not to Weaver.

---

## 1. Contamination policy — ideas in, results out

**Reusable (ideas, cheap, falsifiable — we test them anyway):** the candidate
trigger menu (§5), the physical hypotheses (§5, stated as *open* hypotheses), the
methodology (shadow/pre-registered gates), and the config span (configs are
inputs, not results).

**Quarantined (do NOT inherit; provenance-unknown until regenerated from the
pinned SHA):** every firing-epoch number, ranking, ε-sensitivity result, β/δ
range, "gate passed/failed" verdict, and figure from prior work. Do not cite or
build on any of them.

> **Disclosure (2026-06-20):** during idea-extraction, a subagent leaked a few
> prior *verdicts* (e.g. a per-config retained-energy value, an "F1 retreats under
> surge" claim, a "clock-B is not a macro driver" claim). **These are explicitly
> quarantined and must not seed any gate or expectation.** They are listed here
> only so a reader knows to disregard them. That they slipped through is exactly
> why C0/G0 are pre-registered against an *independent oracle* (§3), not against
> prior conclusions.

**Quarantine inventory (paths only; do not read for their numbers/verdicts):**
- Docs: `docs/dev/transition/P0.md` (all results/verdict sections), `docs/dev/transition/TRIGGER_PLAN.md` (firing claims), `docs/dev/transition/pshadow-design.md` (evidence section); `docs/dev/archive/betadelta/{PHASE2_ARMS,stalling-energy-phase,HYBR_PLAN,PHASE0_BASELINES}.md` (results/verdict sections).
- Prior harness (do not reuse/extend): `docs/dev/transition/harness/` (`harvest.py`, `psens.py`, `heartbeat.py`, `*.param`).
- Data CSV/JSONL: `docs/dev/data/transition_*.csv`, `docs/dev/data/hunt_h*.csv`, `docs/dev/data/stalling_*.csv`, `docs/dev/archive/betadelta/diagnostics/*.{jsonl,csv,csv.gz}`.
- Figures: `docs/dev/archive/betadelta/diagnostics/*.png`, `*.gif`.

Methodology/scope sections of the quarantined docs (the "how", not the "what we
found") are safe to reuse.

---

## 2. C0 — substrate certification gate (PRE-REGISTERED)

Certify the substrate with checks that are independent of the contested trigger
and, where possible, of the numerics. **Bars are pre-registered here, before any
run** (adjustable only with a dated note explaining why, per the 🔄 banner).

### C0.3 — `Lloss` is pure radiative (code audit) — ✅ DONE (2026-06-20)
Verified against current source: `bubble_LTotal = L_bubble + L_conduction +
L_intermediate` (`bubble_luminosity.py:790`), each a radiative integral —
`χ_e·n²·Λ(T)·4πr²` (CIE, `:696`) or `dudt(n,T,φ)·4πr²` (non-CIE, `:743/780`).
**No PdV term, no velocity** in any integrand (`v_array` is computed but never
enters the luminosity). ⇒ the ratio's `Lloss` is a clean cooling-vs-injection
fraction; PdV is carried separately in the energy ODE (next).

### C0.2 — β/δ ↔ trajectory consistency (internal, code-derived) — harness built
**Code-grounded reframing (important).** The energy ODE is **not** a naive
`dEb/dt = Lgain − Lloss − PdV` budget; it is the Rahner-thesis A12 equation
(`get_betadelta.py:182` `cool_beta_to_Ebdot_pure`), where cooling enters
*implicitly via β*, not as an explicit `Lloss` term. So a naive energy-budget
closure would test a relation the code doesn't enforce. The relations the code
*does* enforce by definition are:

- `β ≡ −(t/Pb)(dPb/dt)`  ⇒ predicted `dPb/dt = −β·Pb/t`  (`get_betadelta.py:248`)
- `δ ≡ (t/T)(dT/dt)`     ⇒ predicted `dT0/dt = δ·T0/t`   (`get_betadelta.py:294`)

**Check:** finite-difference the stored `Pb(t)`, `T0(t)` across consecutive
implicit-phase snapshots and compare to the predictions from the stored β, δ.
This certifies the solver's (β,δ) outputs are consistent with the integrated
trajectory — independent of the trigger.

**Pre-registered bars (REVISED 2026-06-20 — the δ↔dT0/dt residual turned out
tautological (see Progress below) and is NOT a gate):**
(i) **`res_beta` (β↔dPb/dt — the genuine, trajectory-level check):** median
relative residual ≤ **5%** over implicit-phase rows, AND it **shrinks under
timestep refinement** (∝ Δt ⇒ finite-difference truncation, not a defect).
(ii) **`res_T0_struct` (= |T0 − bubble_T_r_Tb|/bubble_T_r_Tb — the two sides of the
solver's own `T_residual`) on converged segments:** small (≤ a few %), confirming
the solver drives its temperature residual to zero on the trajectory. Fail either
⇒ the β/δ the trigger depends on are not trajectory-consistent ⇒ escalate scope.

> **Progress (2026-06-20) — δ resolved, first real hybr result; results vary
> sharply by config/phase, so do NOT generalize from one run.**
> - **δ↔dT0/dt is TAUTOLOGICAL — dropped as a gate.** `T0` is a `solve_ivp` state
>   variable whose RHS *is* `δ·T0/t` (`run_energy_implicit_phase.py:507–532,989`),
>   so finite-differencing it trivially matches. Replaced by `res_T0_struct =
>   |T0 − bubble_T_r_Tb|/bubble_T_r_Tb` — the two sides of the solver's own
>   `T_residual` (`get_betadelta.py:449`), meaningful **on converged segments**.
> - **`β↔dPb/dt` is GENUINE** (trajectory-level: `Pb←bubble_E2P(Eb)`, independent
>   of the solver's `β`). Its 5548% mock outlier and the 9% below are real signals
>   to chase, not noise.
> - **First real hybr run** (`small_dense`, reached only t=0.028 Myr in ~10 min ⇒
>   ~35 s/implicit segment): `res_T0_struct` median **0.0%**, max 0.85% ⇒ hybr
>   converges its T-residual tightly **(PASS)**. `res_beta` median **9.2%** (14%→6%
>   early→late within the run) ⇒ **PROVISIONAL**: the early→late decrease looks
>   like finite-difference truncation, not a defect, but the pre-registered bar
>   (ii) **timestep-refinement check** must confirm before `res_beta` counts. No
>   negative β here (the docs' negative-β is a later / steeper-config phenomenon).
> - **Variability (the headline caution):** vs the provenance-unknown legacy mock
>   (`res_beta` 2.5%; `res_T0_struct` 15% from loose legacy convergence; `f_ret`
>   *rising* 0.50→0.62 not falling), the hybr run differs on every axis. **Certify
>   across the full span AND over time/phase — never from one run.**
> - **Cost reality (CORRECTED 2026-06-20):** the ~35 s/segment above was a
>   **DEBUG-logging artifact** — the harness calls `start_expansion` directly,
>   tripping `main.py`'s DEBUG fallback (per-RHS records, a known hot-path cost).
>   With a log handler installed first, full `stop_t=6` runs proceed at normal pace
>   (~30–90 min; t≈3 Myr in ~30 min) — Myr coverage is **feasible**, not "hours".
>   Both short (`stop_t=0.05`) and full (`stop_t=6`) batches run via
>   `run_c0_batch.sh` (3 concurrent on the 4-core box), health-monitored by
>   `heartbeat.sh` (CPU-state liveness; a Monitor pings every 6 min).
>
> **Cross-config short-run certification (2026-06-20, `stop_t=0.05`, EARLY implicit
> only, t≤0.014):** consistent across configs — `res_T0_struct` median 0.27–0.50%,
> max ≤0.92% (**PASS, span-wide**); `res_beta` median **5.5–6.1%** (just over the
> 5% bar, *every* config), all in the early-implicit FD-truncation regime;
> `f_ret`≈**0.42 uniformly** (Weaver half-energy); **no negative β** this early.
> Reading: `res_beta`'s ~6% is most likely early-implicit FD truncation (consistent
> across very different configs + the within-run 14%→6% decay), **not certified
> until** the refinement check + full runs show it drops <5% in mid/late implicit.
> The under-cooling and negative-β questions are **not answerable from short runs** —
> hence the full (`stop_t=6`) runs now in flight. **Logging fix:** the harness
> installs an **INFO** handler before `start_expansion` (bypasses `main.py`'s DEBUG
> fallback — a per-RHS hot-path cost; INFO is the normal `run.py` level — light and
> per-phase insightful), so full runs hit the ~30–90 min regime.
>
> **Full-run progress (2026-06-20, IN FLIGHT — PROVISIONAL, not certified):** with
> 3 of 6 full runs in mid-implicit (t≈2–3 Myr), `f_ret` has fallen from ~0.42 to
> **~0.18–0.27** and sits **above** the literature 0.01–0.1 band (the run past the
> WR surge at t=3.0 is at 0.27, slightly *up* from 0.18 at t≈1.9 — a feedback
> re-energisation). This **leans toward the under-cooling reading** (§0.1 outcome 2)
> but is undecided until t=6 (plateau vs continued descent — the f_ret verdict plot
> settles it). **No negative β** in any of the 3 configs run so far (be_sphere,
> large_diffuse, midrange) even past the WR surge — qualifying the quarantined docs'
> negative-β claim; **watch the steep `pl2_steep`** (wave 2), where it should appear
> if anywhere.
> **Recheck item:** the analyzer reports `betadelta_converged = 0` in the CSVs, so
> the `res_T0_struct` "converged-only" filter is currently falling back to
> all-implicit rows. Verify whether `betadelta_converged` is actually a written
> snapshot key (and its truthiness) so the converged qualification on C0.2 (ii)
> really bites — `res_T0_struct` is tiny regardless (~0.3–0.5%), but the gate's
> wording should match what the filter does.

### C0.1 — analytic adiabatic Weaver null (analytic-limit regression — implementation check, NOT physics validation) — STAGED
**Caveat (§0.1):** matching Weaver certifies the code solves *its own* equations in
a known limit; it does **not** certify the physics — the literature argues the
Weaver energy-conserving limit is wrong for real clouds. Physical plausibility is a
*separate* track (the §0.1 `f_ret` anchor), not part of substrate certification.
The strongest *implementation* check available, numerics-independent: in the
adiabatic limit (`Lloss→0`) the energy-driven bubble has the closed-form
Weaver (1977) solution
(`R₂ ∝ (Lt³/ρ)^{1/5}`, `Eb = (5/11)L_w t` — cf. `get_InitPhaseParam.py:166`; a
constant β; `T` from Weaver Eq. 37).

**Blocker + resolution.** There is **no cooling-off parameter** (registry has only
`cool_alpha/beta/delta` + table paths; `net_coolingcurve.get_dudt` has no disable
path). Rather than add a production flag (would break "frozen"), the harness will
**monkeypatch the cooling lookups to zero inside the offline harness only**
(`bubble_luminosity.py:694` CIE `Lambda_bubble`, `:741/779` non-CIE `dudt`) — the
same runtime-wrap pattern the prior `velstruct/hunt.py` used for the solver.
Production source stays byte-identical (nothing in `trinity/` changes).

**Pre-registered bars:** (i) code `R₂(t),Eb(t),Pb(t),T(t)` match analytic Weaver to
**≤2%** over the energy/implicit phase; (ii) **hard gate** — with `Lloss≡0` the
ratio `(Lgain−Lloss)/Lgain → 1` and the transition trigger **never fires** (any
firing ⇒ mis-normalized substrate, STOP).

**Feasibility risk + fallback.** If the betadelta solver cannot converge with
`Lloss≡0`, fall back to **C0.1b** (below). The hard "never-fires" gate still holds
regardless of convergence (it tests the trigger, not the structure).

### C0.1b — Weaver early-phase scaling (external, no patch) — STAGED, fallback
With cooling on but sub-dominant early, `R₂(t)` in the energy phase should track
the analytic power law `R ∝ t^{3/(5−|α|)}` (cf. `paper/methods/figures/paper_radiusComparison.py`).
Weaker than C0.1 (cooling contaminates; tests phase 1a, a different path) but
needs no patch. Pre-registered bar: power-law exponent recovered to ≤5% over the
early energy phase.

**C0 verdict rule:** substrate is **certified** iff C0.3 ✅ (done) **and** C0.2
passes **and** (C0.1 passes, or C0.1 infeasible **and** C0.1b passes **and** the
C0.1 hard never-fires gate passes). Otherwise scope escalates to substrate
re-verification.

---

## 3. The independent oracle (for the later trigger harvest, not C0)

Every candidate trigger is judged against a reference that depends on **no
candidate and no threshold**: the **PdV-inclusive net-energy zero crossing** — the
first time the bubble stops gaining energy,
`Lgain − Lloss − 4πR₂²·v₂·Pb ≤ 0` (the `Eb`-peak). A good trigger fires at/just
after this peak. Cross-checks: the adiabatic null (§2, must never fire) and the
retained-energy fraction `Eb/∫Lgain dt` at firing (literature sanity band — verify
the range before citing). The oracle is **diagnostic only**; it never drives a run.

---

## 4. Config span (inputs; reused, not inherited results)

The six characterized configs (already pass GMC plausibility), spanning feedback
strength × cloud density × profile, plus one strong-WR/SN-jump config:

| config | mCloud | sfe | nCore | rCore | profile | note |
|---|---|---|---|---|---|---|
| large_diffuse_lowsfe | 1e7 | 0.01 | 1e2 | 1 | densPL α=0 | weak feedback, diffuse |
| simple_cluster | 1e5 | 0.3 | 1e5* | 0.01* | densPL α=0 | baseline |
| small_dense_highsfe | 1e4 | 0.5 | 1e6 | 0.1 | densPL α=0 | strong, dense |
| midrange_pl0 | 1e6 | 0.1 | 1e4 | 0.01* | densPL α=0 | mid |
| **pl2_steep** | 1e6 | 0.1 | 1e5 | 1 | densPL **α=−2** | **crux** (steep halo) |
| be_sphere | 1e6 | 0.05 | 1e4 | 1 | densBE Ω=14 | BE profile |

`*` = schema default. The steep config is the **crux**, not an afterthought: for
r⁻² `Lloss ∝ n²` may collapse before any cooling family fires (§5).

---

## 5. Candidate menu + hypotheses (IDEAS — unverified, to be tested)

**Candidate trigger families** (definitions only; *which fires when* is to be
measured, not inherited):
| id | family | criterion |
|---|---|---|
| F0 | instantaneous rate-ratio (**current**) | `(Lgain−Lloss)/Lgain < ε` |
| F1 | cumulative energy | `∫Lloss dt / ∫Lgain dt > 1−η` |
| F2 | timescale | `t_cool/t_dyn < k` (define `t_cool`,`t_dyn` explicitly; report sensitivity) |
| F3 | force / continuity | `4πR²Pb / (surviving forces) < O(1)` |
| F4 | blowout (geometric) | `R2 > rCloud` |
| F5 | mixing-flux balance (Lancaster+2021) | efficient fractal-interface cooling ⇒ momentum-driven; no sharp 1D transition (cite; a 1D switch is a modeling necessity, not 3D physics) |

**Open hypotheses (to confirm/refute from regenerated data — NOT findings):**
1. The trigger's **instantaneous numerator** resets the ratio *upward* on each
   feedback episode (WR/SN switch-on spikes `Lgain`), so a cooling event can be
   masked. Integrating over episodes (F1) or a sustained-over-`t_cross` rule may
   remove it. *(hypothesis)*
2. For a **steep r⁻² halo**, `Lloss ∝ n²` collapses as the bubble expands, so no
   cooling family (F0–F2) may ever fire; the physical fate is **blowout (F4)** /
   force-subdominance (F3). ⇒ the right trigger may be **profile-dependent**, which
   a single scalar cannot express. *(hypothesis — the crux to settle)*
3. The implicit-phase duration ("clock A", reach-transition) and the 1c
   transition-phase length ("clock B", sound-crossing drain) are **distinct
   clocks**; separate before explaining any "long transition". *(hypothesis)*
4. **Under-cooling hypothesis (§0.1):** the stall is not (only) a mis-tuned trigger
   but a symptom that TRINITY's Weaver/Rahner interior **under-cools** (no fractal
   mixing-layer losses), so the modeled bubble retains far more energy than real
   bubbles (literature `f_ret`~0.01–0.1) and never reaches energy balance.
   Falsifiable: if measured `f_ret` enters the literature band yet the bubble still
   doesn't transition, the trigger is the problem; if `f_ret` stays far above the
   band, the cooling physics is. *(hypothesis — decides whether this is a trigger
   problem or a physics problem; the latter is out of current scope but must be
   named, not silently treated as a threshold-tuning task)*

---

## 6. Downstream (sketched; detail deferred until C0 passes)
- **H0 — harvest (shadow/offline):** from the pinned SHA, per implicit-phase
  segment, log each family's would-fire epoch, the §3 oracle, **and the §0.1
  retained-energy anchor `f_ret = Eb/∫Lgain dt`** (vs the ~0.01–0.1 literature
  band); commit CSVs + harness + exact command.
- **G0 — divergence gate (pre-registered):** which families track the `Eb`-peak
  *without resetting* across the WR/SN jump, across the full span incl. the steep
  crux. "No single scalar works → profile-dependent" is an allowed, publishable
  verdict.
- **Shadow → validate → default decision**, all behind a `transition_trigger`
  param with **default unchanged** (`instantaneous`).

---

## 6.5 Visualization plan (figures + tables)

Build under `cleanroom/figures/`, regenerable from the committed CSVs, using the
repo `trinity.mplstyle` + Wong palette. Each maps to a question; favour figures
that render a verdict by shape alone. **Convention: every time-series figure
shades the four evolution-phase regions (energy / implicit / transition /
momentum) as background bands**, so each plot shows all phases at a glance and the
phase-dependence is never hidden — and **every phase gets a dedicated plot or two**
(below), since the physics differs by phase (rule: don't read one phase as the
whole story).

**Headline / "nail in coffin":**
1. **`f_ret(t)` verdict plot** — all 6 configs, log-y; shade the literature band
   0.01–0.1 (Lancaster/Geen), Weaver ~0.5 dashed; mark WR/SN epochs. Shape = the
   verdict: curves into the band ⇒ **trigger** problem; flat at ~0.2–0.5 never
   reaching it ⇒ **under-cooling physics gap**. *Single most important figure.*
2. **F0 pathology** — 2 shared-x panels: cooling ratio `(Lgain−Lloss)/Lgain` vs the
   0.05 threshold (top), `Lmech_W`/`Lmech_SN` (bottom); arrow each surge showing the
   ratio jump *away* from 0.05. Overlay the Eb-peak oracle + F0/F1/F4 firing epochs.
3. **Legacy-vs-hybr fate bars** — stacked phase durations per config, two bars each;
   hybr's never-transitioned implicit block in red.

**Certification:**
4. `res_beta(t)` & `res_T0_struct(t)` small multiples (truncation vs defect).
5. `res_beta` median vs timestep, log-log (slope ~1 ⇒ truncation; the nail).
6. β(t) with β<0 shaded + `Lmech_total` overlay (re-pressurisation; steep crux vs
   small_dense contrast).

**Per-phase coverage (a plot or two each — the phase-by-phase narrative):**
- **Energy (1a):** *E1* R₂(t), v₂(t) vs the analytic Weaver self-similar law
  `R∝t^{3/(5−|α|)}` (the C0.1b external check — does the energy phase track
  Weaver?); *E2* `f_ret` entry level (~0.42–0.76) per config.
- **Implicit (1b) — the crux, carries the headline set:** the `f_ret` verdict plot
  (1), F0 pathology (2), β(t) re-pressurisation (6), and the res_beta/res_T0_struct
  certification (4–5) all live here.
- **Transition (1c):** *T1* `Eb(t)` & `f_ret(t)` through the sound-crossing drain
  (clock B) — show `f_ret` crashing toward the band (the mock did 0.44→0.012),
  legacy reaching it vs hybr maybe not; *T2* which `min(Ed_energy_balance,
  Ed_soundcrossing)` branch is active over time (does a feedback surge stall the
  drain?).
- **Momentum (2):** *M1* R₂(t)/v₂(t) coast + dissolution and `f_ret`→0 — the
  endgame fate per config (shell_dissolved / large_radius / stopping_time).

**Tables:** A) C0 scorecard (configs × res_beta early/late, res_T0_struct, conv%,
verdict); B) f_ret/fate (configs × f_ret_min, enters-band?, final phase, fate);
C) candidate firing-epoch divergence vs the Eb-peak (the G0 deliverable);
D) per-phase durations (configs × energy/implicit/transition/momentum Myr, legacy
vs hybr — the numeric companion to the fate bars).

## 7. Reproduce / artifacts
- Pinned baseline SHA: recorded in `data/` outputs (see harness `--meta`).
- C0.2: `python docs/dev/transition/cleanroom/c0_consistency.py <param-or-jsonl> [--stop-t T] [--out CSV]`.
- Data lands in `docs/dev/transition/cleanroom/data/` (committed).

---

## 8. External references (independent of TRINITY — verify journal refs before paper)

Consulted 2026-06-20 via web search; treat as the *physical* anchors the trigger is
judged against (TRINITY/WARPFIELD/Weaver are the hypotheses, not these).

- **Weaver, Williams, McCray & Moore 1977** — classical energy-driven wind bubble
  (the analytic limit C0.1 regresses against). *The model the others dispute for
  real clouds.*
- **Lancaster, Ostriker, Kim & Kim 2021a/b**, ApJ 914 — "Efficiently Cooled Stellar
  Wind Bubbles in Turbulent Clouds" I (fractal theory, arXiv:2104.07691) & II (3D
  hydro validation, arXiv:2104.07722). Retained energy ~0.01–0.1, decreasing;
  momentum-driven. *Primary external anchor.*
- **Lancaster et al. 2021c**, ApJL — "Star Formation Regulation and Self-pollution
  by Stellar Wind Feedback" (arXiv:2110.05508).
- **El-Badry, Ostriker, Kim, Quataert & Weisz 2019**, MNRAS — superbubbles with
  conduction & cooling (arXiv:1902.09547).
- **Geen, Pellegrini, Bieri & Klessen 2021**, MNRAS 501, 1352 — wind bubbles in
  photoionised HII regions; stored energy ~1% of input (arXiv:2009.08742).
- **Pabst et al. 2020** — Orion [CII] M42/M43/NGC 1977, expanding wind shells
  (arXiv:2005.03917).
- **Mac Low & McCray 1988; Koo & McKee 1992** — classic cooling/timescale criteria
  (cumulative `t_cool`, not the instantaneous form — see candidate caveats).
- **Rahner et al. 2017/2019** — WARPFIELD (TRINITY's ancestor; the current F0
  energy-retention trigger). *A hypothesis under test, not the reference.*
