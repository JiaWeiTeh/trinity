# PdV-in-the-transition-trigger — argument, evidence, and a plan to test it

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**Date:** 2026-06-23. **Branch:** `feature/PdV-trigger-term`. **Status:** scoping + evidence + an
**offline test of reading B** (§Offline test below) — **no production change yet.** This note answers the
maintainer's question ("add a PdV term to the transition trigger — what was the argument against it, and
is it still valid for larger clusters?"), then a **2026-06-23 maintainer redirect** ("see if
`(Lmech−Lloss−PdV)/Lmech < 0.05` works as a trigger; not sure what the standalone `PdV/Lmech` diagnostic
buys us") which is answered offline in §Offline test, and lays out how to plan/test it. Sibling priors (re-verify per banner): `../pt4/TRANSITION_FIX_SCOPING.md`
(Route 1), `../pt4/r1shadow/R1_FINDINGS.md`, `../../failed-large-clouds/PLAN.md` §6.

---

## The question (maintainer, 2026-06-23)

> "Perhaps it's time to add a PdV term into the transition trigger. What was the argument against it?
> If it was that PdV is always deemed too small and negligible, maybe that's not the case now — or not
> the case for larger clusters."

## TL;DR (answers, with evidence below)

1. **"PdV negligible" was never the real argument.** Measured, PdV is **order-unity** — `PdV/Lmech`
   median **0.43–0.55** for *every* normal cloud (table below), not "way too small." So the premise's
   *stated* reason is false: PdV is not negligible.
2. **The actual argument is two-fold:** (a) PdV is **already in the energy evolution** — `Eb(t)` is
   integrated from `Edot_from_balance = Lmech − Lloss − 4πR2²·v2·Pb` (`get_betadelta.py:434`), which
   *includes* the PdV work term; the `cooling_balance` *trigger* deliberately watches only the
   **radiative** ratio `(Lmech − Lloss)/Lmech` because the modelled transition was hypothesised to be
   cooling-driven. (b) Putting PdV *into the trigger* (= the `ebpeak` criterion `Edot_from_balance ≤ 0`)
   **fires nowhere new for normal clouds**, because they are **sub-critical**: even with PdV included,
   net energy keeps growing (Eb grows 1.5–14×10³ monotonically; shipped shadow fires `ebpeak` **0/6**).
3. **The premise is exactly right for large clusters.** At `mCloud=5e9` the cloud is **super-critical**
   — `PdV/Lmech` median **1.42** (PdV *exceeds* Lmech), net energy goes negative, `Eb` peaks and
   collapses (growth **1.014×**). That super-critical regime is precisely where the heavy-cloud crash /
   `ENERGY_COLLAPSED` lives. So a PdV-inclusive trigger (`ebpeak`) **is** the principled handoff *there*.
4. **The machinery already exists.** `ebpeak` shipped as an opt-in, default-off `transition_trigger`
   token (default `cooling_balance`, byte-identical). So this is **not** "add PdV" from scratch — it is
   **"validate the PdV-inclusive trigger toward becoming the default (or the heavy-cloud handoff), and
   measure where the sub→super-critical boundary sits across the science grid."**

## Where PdV lives today (3 sites — verified against current source 2026-06-23)

| site | formula | PdV included? | role |
|---|---|---|---|
| **Energy evolution** `get_betadelta.py:434` (`Edot_from_balance`), stored `residual_Edot2_guess` (`run_energy_implicit_phase.py:846-847`) | `Lmech − Lloss − 4πR2²·v2·Pb` | **yes** | how `Eb` actually evolves — PdV already drains the reservoir |
| **`cooling_balance` trigger** `run_energy_implicit_phase.py:1200` | `(Lmech − Lloss)/Lmech < 0.05`, `Lloss = bubble_LTotal (+leak)` | **no** | the default energy→momentum handoff; pure radiative |
| **`ebpeak` trigger** (opt-in) `evaluate_r1_shadow` `:208-209`, shadow `:1166-1184`, drive `:1192-1198` | `Edot_from_balance ≤ 0` | **yes** | "PdV in the trigger" — the net-energy turnover; default-off |

⇒ The maintainer's "add PdV to the trigger" **is** the existing `ebpeak` criterion. It is *not* the same
as "add PdV to the 0.05 cooling ratio" — see §Two readings.

## Evidence: PdV magnitude per regime

Persisted: `data/pdv_regime_budget.csv` (derived from the committed `../cleanroom/data/c0_*_h0.csv` and
`../../failed-large-clouds/data/budget_*.csv`; regenerate with `data/make_pdv_regime_table.py`). Startup
rows dropped; ratios in trinity code units (`PdV = 4πR2²·v2·Pb`, same convention as `Edot_from_balance`).

| config | regime | `PdV/Lmech` med | `PdV/Lmech` max | `Eb` growth | `Eb` monotonic | real in-cloud Eb-peak? |
|---|---|---|---|---|---|---|
| simple_cluster | normal | 0.460 | 0.646 | 2405× | yes | **no** |
| small_dense_highsfe | normal | 0.464 | 0.658 | 1951× | yes | **no** |
| midrange_pl0 | normal | 0.461 | 0.702 | 2009× | yes | **no** |
| pl2_steep | normal | 0.429 | 0.701 | 2171× | yes | **no** |
| be_sphere | normal | 0.453 | 0.666 | 1715× | yes | **no** |
| large_diffuse_lowsfe | normal | 0.443 | 0.550 | 1499× | yes | **no** (end-of-run blip only) |
| small_1e6 (ctrl) | normal | 0.554 | 1.102 | 13617× | yes | **no** (end-of-run blip only; max>1) |
| **fail_repro** | **heavy 5e9** | **1.423** | 1.561 | **1.014×** | **no** | **yes (row 5, t≈1.53e-3 Myr)** |

Authoritative cross-check (real shipped shadow, segment-wise, not a CSV reconstruction):
`../pt4/r1shadow/r1_shadow_summary.csv` — all 6 normal configs fire **blowout**, `ebpeak` column **blank
(0/6)**; `fail_repro`/`fail_helix` show `n_seg=0` because they collapse in **phase 1a**, *before* the 1b
shadow ever runs. ⚠️ My offline `net_Edot≤0` reconstruction has edge sensitivity (flags a few
end-of-run / startup blips as "fires"); where it disagrees with the shipped shadow, **trust the shadow**.

**Reading:** `PdV/Lmech` is a smooth control parameter. Normal clouds sit ~0.45 (sub-critical, ≪1
margin to the Eb-peak); the 5e9 cloud sits ~1.4 (super-critical). **The whole behavioural fork is which
side of `PdV/Lmech = 1` the cloud lands on.** "PdV negligible" is false everywhere; "PdV decisive" is
true only past the unity crossing.

## Two readings of "add PdV to the trigger" (pick deliberately)

- **(A) `ebpeak` — net-energy turnover `Lmech − Lloss − PdV ≤ 0`.** Threshold = 0 (energy stops
  growing). Principled (it is the physical Eb-peak), already implemented, fires only super-critically.
  Normal clouds: never in-cloud. Heavy clouds: at birth. **This is the recommended meaning.**
- **(B) PdV inside the 0.05 cooling ratio — `(Lmech − Lloss − PdV)/Lmech < 0.05`.** Keeps the legacy
  threshold but moves the operating point by ~`PdV/Lmech ≈ 0.45`. This is **not** physically grounded
  (0.05 was calibrated for a radiative-only ratio) and would fire for normal clouds at an arbitrary
  epoch set by an un-recalibrated constant. **Not recommended** unless re-derived from a model — record
  it only as the literal interpretation of the request, then steer to (A).

## Offline test of reading B — does `(Lmech−Lloss−PdV)/Lmech < 0.05` fire? (2026-06-23 redirect)

The maintainer asked to **test reading B directly** and questioned the point of the standalone `PdV/Lmech`
diagnostic. Both are answered **offline** from the already-committed per-step CSVs — no sims — by
`data/make_combined_trigger_table.py` (→ `data/pdv_combined_trigger.csv`, `pdv_combined_trigger.png`).
Numbers reproduced by an independent recompute on `large_diffuse_lowsfe`/`simple_cluster`/`small_dense_highsfe`.

**The identity that settles the `PdV/Lmech` question.** Write `cool = (Lmech−Lloss)/Lmech` (the shipped
radiative cooling ratio, *no* PdV) and `coolPdV = (Lmech−Lloss−PdV)/Lmech` (the same ratio *with* PdV =
reading B). The with-PdV ratio is *algebraically* the radiative one minus `PdV/Lmech`:

    coolPdV = (Lmech−Lloss−PdV)/Lmech = (Lmech−Lloss)/Lmech − PdV/Lmech = cool − PdV/Lmech

So **`PdV/Lmech` is exactly the offset between the shipped `cooling_balance` trigger and reading B** — its
only role is to quantify how much folding PdV into the ratio loosens the operating point. It is a
*decomposition* diagnostic, **not** a threshold variable; thresholding it against 1 (old Step 1) chases a
sufficient-but-not-necessary proxy (the real crossing is `(Lloss+PdV)/Lmech`, and `Lloss/Lmech` ≈ 0.17–0.29
is not negligible). Equivalently: **reading B = the shipped trigger run at threshold `0.05 + PdV/Lmech ≈ 0.5`**
— a ~10× looser, un-recalibrated constant.

**Result — first-fire of `coolPdV < 0.05` (sustained), vs the shipped `cool < 0.05`:**

| regime | configs | `cool<0.05` fires | `coolPdV<0.05` fires | where / note |
|---|---|---|---|---|
| normal | 5/6 cleanroom | 0 | **0** | min `coolPdV` only 0.08–0.15 — never reaches 0.05 |
| normal | large_diffuse_lowsfe | 0 (cool≈0.49 there) | **yes, sustained** | t≈4.76 Myr, **86% through** the run — arbitrary epoch |
| heavy 5e9 | fail_repro | 0 | at birth (row 3, t≈1.5e-3) | `coolPdV<0` immediately, stays `<0` for the physical run |
| ctrl | small_1e6 | 0 | row 0 startup blip (not real) | spurious — recovers to ~0.40 |

**Verdict on reading B (threshold 0.05): it does not behave as a usable trigger.** For 5/6 normal clouds it
is silent (the bubble never stops gaining energy — `coolPdV` bottoms at 0.08–0.15 and recovers); for the 6th
it fires at a late, arbitrary epoch fixed by the mis-set constant, where `cool` is still ≈0.49 (no physical
handoff). The only physically-grounded threshold for the PdV-inclusive ratio is **0** (= `ebpeak`/reading A,
net energy stops growing): normal clouds essentially never cross it in-cloud (`large_diffuse` only oscillates
across 0 at the very end, non-sustained), the 5e9 crosses at birth. **The data confirms reading A over B.**

**Corollary — the real handoff for normal clouds is not energy-budget at all.** Sub-critical clouds fire
*neither* `cool` nor `coolPdV`, so what drives their transition is **blowout** (geometric `R2 > rCloud`),
consistent with the shipped 1b shadow (6/6 blowout, `ebpeak` 0/6). An energy-balance trigger — radiative or
PdV-inclusive — is the wrong family for them; it is decisive only super-critically (the 5e9 pathology).
The figure now marks each config's **blowout point** (`R2 = rCloud`, recovered from `r1_shadow_summary.csv`
since the CSVs export `rCloud` as all-NaN; persisted as `blowout_t`/`cool_at_blowout`/`coolPdV_at_blowout`):
**at blowout the cooling ratio is still 0.30–0.75 (no PdV) / 0.12–0.58 (with PdV)** — i.e. the clouds hand
off to momentum while the energy budget is nowhere near the 0.05 band. That gap is the room a cooling-boost
`θ_cool` would have to close — see next section.

## A more promising direction: PdV **+** a cooling-boost `θ_cool` (2026-06-23 maintainer idea)

> "Maybe a combination of PdV in the cooling **and** a cooling-boost factor `θ_cool`, argued from El-Badry,
> Lancaster, Gronke: our 1D model has no turbulent mixing layers, so we under-count interface cooling.
> `θ_cool` could be a constant from those papers — but shouldn't it couple to the bubble physics?"

**Why it's promising (offline screening — frozen-trajectory, see caveat).** Solve for the constant boost that
makes the PdV-inclusive ratio reach the threshold, `(Lmech − θ_cool·Lloss − PdV)/Lmech = 0.05`, on the
*committed (unboosted)* trajectories:

| config | θ_cool to fire **at blowout** (w/ PdV) | θ_cool to fire **anywhere** (w/ PdV) | (no PdV, anywhere) |
|---|---|---|---|
| small_dense_highsfe | 1.10 | 1.04 | 1.33 |
| simple_cluster | 1.12 | 1.06 | 1.41 |
| midrange_pl0 | 1.20 | 1.08 | 1.49 |
| be_sphere | 1.26 | 1.18 | 1.80 |
| pl2_steep | 1.49 | 1.24 | 1.86 |
| large_diffuse_lowsfe | 3.13 | 0.87 (already <1) | 1.78 |

So **PdV + a modest `θ_cool ≈ 1.1–1.5` would fire the energy→momentum handoff right at blowout** for 5/6 normal
clouds — and that boost is *below* the enhancement the mixing-layer literature argues for (El-Badry+19
catastrophic cooling; Lancaster+21 near-complete wind-energy cooling; Gronke & Oh mixing layers). This is the
first candidate that makes a PdV-inclusive trigger physically *and* numerically land where the cloud actually
transitions, instead of never (reading B) or at an arbitrary epoch.

> ⚠️ **Caveat — these θ are a SCREENING estimate, not the answer (rule 5: per-call ≠ full-run).** `θ_cool` is
> "we're missing real cooling," so it belongs in the **energy evolution** (`Edot_from_balance = Lmech −
> θ_cool·Lloss − PdV`), not only in the trigger ratio. Boosting cooling lowers `Pb` → lowers `PdV` (`∝Pb`) →
> changes `Eb(t), R2(t), v2(t)` → **moves blowout itself**. The table freezes the unboosted trajectory, so it
> is necessary-but-not-sufficient. The honest test is a **full run with boosted cooling**, separate processes,
> matched `t`, on the edge configs.

**Constant vs. coupled (the maintainer's question).** Start **constant** — one opt-in float `theta_cool`
(default `1.0`, byte-identical; ponytail-simplest, calibratable, testable). But constant is physically a
placeholder: the mixing-layer luminosity is **not** constant — it scales with the contact-discontinuity area
(`∝R2²`), the shear/turbulent velocity (`∝v2` / hot-gas sound speed), and the mixing-layer cooling function
(Damköhler number; Tan/Oh/Gronke 21, Lancaster fractal-area scaling). The data already argues coupling is
needed: **θ_at_blowout spans 1.1 → 3.1** across configs, so no single constant fires them all at blowout.
Upgrade path: `θ_cool(R2, v2, T)` from the mixing-layer scalings — mark the constant version with a
`ponytail:` comment naming that ceiling.

**Where it plugs in (code map, verified 2026-06-23).** Cooling is computed in
`trinity/bubble_structure/bubble_luminosity.py::_bubble_luminosity()` (three-zone trapezoid integral →
`bubble_LTotal`); **no existing boost knob** (`cool_alpha/beta/delta` are Weaver evolution params, not
efficiency). Add `theta_cool` like `transition_trigger`: `ParamSpec` in `trinity/_input/registry.py` (~:350)
+ a line in `default.param`, then multiply the cooling integrand at the `_bubble_luminosity` site (R2, v[r],
T[r], T_avg, n[r], Pb are all in scope there for a coupled form). Default `1.0` ⇒ byte-identical.

**Recommended sequencing:** (1) opt-in constant `theta_cool` (default 1.0); (2) full-run screening on the edge
configs (`simple_cluster` + `f1edge_{lowdens,hidens}` + a 5e9) to see whether `θ_cool ≈ 1.5–3` makes the
PdV-inclusive trigger fire near blowout *self-consistently*; (3) only then a coupled `θ_cool(R2,v2,T)`.

## Refined plan — unresolved-interface-cooling closure (Paper-II note, 2026-06-24)

Supersedes/sharpens the `θ_cool` sketch above (where my "θ_cool" = the note's **`f_mix`**, a *multiplier*, not a
*fraction*). Driver: the maintainer methods note *"Adding unresolved interface cooling to TRINITY without
double-counting"* + an adversarial physics check (double-count algebra **verified**; `max()` closure **proven
double-count-free** over 5×10⁵ random draws).

**Framework (note §2–6):**
- Distinguish loss **fraction** `θ ≡ Lloss/Lmech ∈ [0,1]` (a target/output) from loss **multiplier**
  `f_mix ≡ Lcool_mix/Lcool_smooth ≥ 1` (a knob on the resolved integral). One symbol must not name two operations.
- **Never double-count.** TRINITY already subtracts the explicit `Lcool`; adding a `(1−θ)Lmech` input-rescale on
  top removes `2θLmech` at consistency (net drive negative for θ>½). The correction must **add only the missing
  part**, never rescale `Lmech`.
- **Boost the LOSS, keep the trigger form.** Note's trigger is `(Lmech − Lloss_eff)/Lmech < 0.05`, with PdV in
  the **ODE only** (not the trigger). Physics: PdV is *reversible* (recoverable as shell momentum), cooling is
  *irreversible* — fire on the irreversible channel. ⇒ **This is distinct from reading B** (which put PdV in the
  trigger); the note instead fixes the cooling *magnitude*.

**Closures under test (default `none` ⇒ byte-identical):**
- `multiplier`: `Lloss_eff = Lleak + f_mix·Lcool` — sweep probe; does *not* change the T-profile or evaporation
  (its ceiling — a scalar can't back-react on the evaporative mass flux).
- `theta_target`: `Lloss_eff = max(Lcool+Lleak, θ_target·Lmech)` — double-count-free **iff** the two terms are
  estimators of the *same* sink (they are: `θ_target·Lmech` is a target on the resolved-cooling fraction). Tops
  up to the target, switches OFF where resolved cooling already exceeds it.
- `kappa_eff` (endgame, out of shadow scope): `κ_eff = max(κ_Spitzer, κ_mix)`, `κ_mix ~ ρ cp D_turb`,
  `D_turb ~ λ δv ~ R2 v2`. The only honest form — couples cooling↔evaporation and can reproduce El-Badry's 3–30×
  evaporation suppression; scalar closures cannot. The scalars are calibration probes that point here.

**Consistency contract (note §Code-level):** one helper feeds the β–δ residual, the energy ODE, *and* the
trigger — the same `Lloss_eff`. Shadow ⇒ reconstruct the trigger ratio only; production ⇒ this is the gate.

**Staged shadow / non-disruptive test — all 8 configs (6 normal + fail_repro + fail_helix):**
- **Stage 1 — Gate audit** (note's "check the gate first"): per-segment {active triggers, Lcool, Lleak, Lmech,
  PdV, β, δ, residual, baseline ratios}. Confirm cooling-balance is *active but never trips* (ratio stays high),
  not a gate bug. Plot: baseline ratio trajectories + blowout markers.
- **Stage 2 — Closure sweep (FROZEN trajectory):** both closures over `f_mix ∈ {1,1.5,2,3,5,10,30}` and
  `θ_target ∈ {0.3,0.5,0.7,0.8,0.9,0.95}` (ceiling **θ_max=0.95** at GMC-core n — the El-Badry density scaling is
  an *extrapolation* there). Per (config × value): does the note `cb` trigger fire? sustained? `t_fire/R2_fire`
  vs blowout? Plots: (a) per-config ratios under increasing boost; (b) fire-vs-blowout heatmap (config × value),
  multiplier and theta_target.
- **Stage 3 — Double-count / consistency check:** instantiate the note's Fig 1 with real per-config `Lcool/Lmech`;
  show the closures stay on the single-count line and never enter the `2θ` region. Plot: that diagram, 8 configs placed.
- **Stage 4 — Which is good:** rank by — fires near blowout for normal clouds (not birth, not never), preserves
  heavy-cloud collapse, double-count-safe, and whether the firing value is ~constant across configs (⇒ a constant
  knob suffices) or spreads (⇒ needs the Da/κ_eff coupling). Recommend a candidate + the gated **Tier-2 full run**
  (disruptive: apply `Lloss_eff` in residual+ODE+trigger, separate processes, matched `t`) as the NEXT step.

**Hard caveat (rule 5 + physics-check §5.1): the shadow only SCREENS.** Boosting cooling lowers Pb → lowers PdV →
moves blowout itself; the unboosted trajectory is *not* the state the boosted ODE visits. Shadow fire-times are a
screen, **not predictions** — the verdict needs Tier-2.

**Coupled upgrade to record (not implement):** `θ_target(state) = θ_max · Da/(1+Da)`,
`Da = (λ/δv)·n_int·Λnet(T)/((3/2)kT)` with `λ~R2, δv~v2`, `n_int` the interface/compressed density. Recovers
El-Badry (high-n, interface-dominated) and Weaver (low-n, energy-driven) limits from one dimensionless ratio.

**Data:** 7/8 offline-reconstructable (6 cleanroom h0 + `budget_fail_repro`); `fail_helix` has only logs (collapses
in phase 1a) → needs the in-solver shadow run. Artifacts: `data/make_closure_test.py`, `data/closure_test.csv`,
`closure_stage{1..4}*.png`.

### Stage results (2026-06-24 — FROZEN-TRAJECTORY SCREEN; bounds the knob, does not forecast)
1. **`cb` trigger (boost loss, no PdV) is the right family for normal clouds:** `f_mix ≈ 1.5–2` brings their cooling
   ratio into the band near the transition. Supersedes reading B (don't put PdV in the trigger; fix the cooling).
2. **A constant knob can't place the transition at blowout across the grid (Stage 2 heatmap).** At `f_mix≈2`,
   compact/dense fire *at* blowout (`simple_cluster −0.02`, `small_dense −0.00` Myr) but diffuse fire *well before*
   (`pl2_steep −0.81`, `large_diffuse −1.3…−3.65`). Density-ordered (dense already cool: `Lcool/Lmech≈0.7` at
   blowout; diffuse `≈0.25`) ⇒ **the data argues for the coupled `θ_target(Da)`/`κ_eff` form, not a constant.**
3. **`theta_target` constant is blunt:** fires nowhere below 0.95, ~at birth at 0.95 — use only via the
   density-dependent `θ_target(n)` model + ceiling. The **multiplier `f_mix` is the better probe.**
4. **Heavy clouds are complementary, not covered by cooling:** `fail_repro` never fires `cb` even at `f_mix=30`
   (`Lcool/Lmech≈0.01`, PdV-dominated) ⇒ heavy clouds need the **PdV/ebpeak handoff**, normal clouds the **cooling
   boost**. Clean sub-/super-critical regime split.
5. **Double-count check (Stage 3):** every config sits on the single-count line; the closures never enter `2θ`.

**Next (gated, disruptive — NOT in the shadow):** wire an opt-in `cooling_boost_mode ∈ {none,multiplier,theta_target}`
feeding the β–δ residual + ODE + trigger *consistently* (note §Code-level), run ≥2 edge configs **live** (separate
processes, matched `t`) to test self-consistency vs the frozen screen; add the in-solver 1a/1b shadow to cover
`fail_helix`. Then decide constant-vs-`θ_target(Da)` from the live spread.

## Plan & test design (rule-5 ladder — this is a risky/iterative/outward-facing change)

The change touches the solver's phase-handoff and the late-time **fate** outputs, and is a
**default-flip** candidate ⇒ full ladder, no rung skipped because an earlier passed.

### Step 0 — Gate first (define "equivalent" before any edit)
- **Hard gate:** any default change must be **byte-identical** (`dictionary.jsonl` sha256) on every
  config that *already* transitions via `cooling_balance`. (Under `hybr` that set is currently empty —
  0/6 fire — but legacy/clamped-β configs and any future-cooled model still use it, so the gate stands.)
- **Continuity gate:** at the handoff, `Eb / R2 / v2 / P_drive` must enter phase 1c no more
  discontinuously than the `cooling_balance` handoff does. **The heavy-cloud Eb-peak is the make-or-break
  case** (reservoir grew only 1.014× → 1c may reject a near-empty bubble).
- Pass/fail bars + `f_ret` targets written here *before* editing.

### Step 1 — Decisive new measurement: combined-ratio first-fire across the science grid
> **2026-06-23 redirect (supersedes the old "map `PdV/Lmech = 1`" framing).** The decision-relevant
> quantity is the **combined ratio** `coolPdV = cool − PdV/Lmech`, not `PdV/Lmech` alone (see §Offline test
> for why `PdV/Lmech=1` is a sufficient-but-not-necessary proxy). The offline first-cut is **done** above;
> the open question is the *in-process, authoritative* version. Still record max/median `PdV/Lmech` per cell,
> but only as the **offset diagnostic** that explains the `coolPdV`–`cool` gap — not as the boundary to map.

The open scientific question behind the maintainer's premise: **does any *realistic* cluster (not just
the 5e9 pathology) approach super-critical?** If the boundary sits far above the science range, the PdV
trigger is an edge-case guard; if real sweeps straddle it, it is a default-relevant correctness fix.
- Sweep `mCloud × sfe × density-profile` (reuse `../../failed-large-clouds/harness/params/` +
  `../cleanroom/configs/`), each run in a **separate process**, with the **shadow active** (default
  trigger ⇒ byte-identical), harvesting `shadow_R1_1b.csv` + per-segment `PdV/Lmech`.
- Record max/median `PdV/Lmech` and first `ebpeak`/`blowout` epoch per cell → a contour of the
  sub→super-critical boundary. Persist as `data/pdv_boundary_grid.csv` + a figure.
- **Note the phase-1a gap:** the 5e9 Eb-peak is a **phase-1a** event invisible to the 1b shadow
  (`r1_shadow_summary.csv` `n_seg=0`). To measure the heavy end, add a **read-only 1a shadow** of
  `Edot_from_balance` (mirror of the 1b shadow; logging only, no break) — itself a gated micro-change.

### Step 2 — Baseline capture
`git show HEAD` trajectories (Eb/R2/v2/P_drive/end-code) on the edge set: `simple_cluster` +
`../../performance/f1edge_{lowdens,hidens}*.param` + a 5e9 point. Saved here so "before" survives.

### Step 3 — Equivalence / behaviour gate
- **Per-call (cheap, necessary, not sufficient):** unit-test `evaluate_r1_shadow` / `r1_transition_decision`
  truth tables already exist (`test/test_r1_shadow.py`, 14/14) — extend with the 1a-shadow analogue.
- **Full-run, stiffest regimes, separate processes, matched `t`:** run `transition_trigger=ebpeak`,
  `blowout`, and `r1` (=both) on all 6 normal configs + ≥1 heavy 5e9, compared against the `cooling_balance`
  baseline at matched simulation time. Check: (i) run completes; (ii) **phase-1c continuity** of the four
  state vars; (iii) terminal fate (momentum, final R2, stop code) is physically defensible.

### Step 4 — Apply the smallest diff that passes
Likely candidates, smallest first: (a) add the **1a Eb-peak shadow** (read-only); (b) if Step 3 shows
the heavy handoff is clean, wire **`ebpeak` as the heavy-cloud handoff** replacing `ENERGY_COLLAPSED`
(opt-in first); (c) only if Step 1 shows science-grid relevance, propose a **default flip** to
`cooling_balance,blowout` (and/or `ebpeak`) — additive, so it never perturbs a run that already fires.

### Step 5 — Re-verify
Continuity + byte-identity gate again, full `pytest` (`-m "not stress"`), ruff F-rules.

### Step 6 — Persist
Boundary grid CSV + figure, baseline/edge trajectories, and the handoff-continuity comparison committed
under `docs/dev/transition/pdv-trigger/` with the exact config + command for each.

## Open questions / risks
- **Heavy handoff viability:** the 5e9 Eb-peak hands off a *stillborn* reservoir (1.014× growth) — does
  phase 1c accept it, or is `ENERGY_COLLAPSED` still the honest end? (pt4 H3/H4 lean toward "needs
  momentum continuation or added cooling," not just a trigger relabel.)
- **Does reading (B) ever make sense?** Only if a recalibrated, model-derived threshold replaces 0.05.
  Default to (A).
- **Boundary location:** if no realistic cluster reaches `PdV/Lmech > 1`, the PdV trigger is a guard for
  the pathological edge, not a science-sweep correctness fix — that changes the priority of a default flip.

### In-solver shadow insertion point (code map, verified 2026-06-23)
If/when the authoritative in-process confirmation of reading B is wanted (the offline reconstruction has the
end-of-run/startup edge sensitivity the §Evidence cross-check warns about), it is a ~5-line **read-only**
add at `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1166`, right after the existing
`ebpeak` shadow eval — `Lgain` (=`Lmech_total`), `Lloss`, `R2`, `v2`, `params['Pb'].value` and
`betadelta_result.Edot_from_balance` are all in scope there. Add `combined_ratio = (Lgain − Lloss −
4πR2²·v2·Pb)/Lgain` (= `edot_balance/Lgain`) and a `combined_ratio` / `combined_ratio_fired` column to the
`shadow_rows` dict (the same block already logs `cooling_ratio` and `edot_balance` → `shadow_R1_1b.csv`).
Byte-identical (logging only); extend the `test/test_r1_shadow.py` truth table (14 tests). **Lower priority
given the offline verdict** — it confirms, it does not change, the reading-B finding.

## Artifacts
- `data/pdv_combined_trigger.csv` (+ `data/make_combined_trigger_table.py`, figure `pdv_combined_trigger.png`)
  — the §Offline-test reading-B first-fire table. Regenerate: `python docs/dev/transition/pdv-trigger/data/make_combined_trigger_table.py`.
- `data/pdv_regime_budget.csv` (+ `data/make_pdv_regime_table.py`) — the §Evidence table.
- Upstream (committed): `../cleanroom/data/c0_*_h0.csv`, `../../failed-large-clouds/data/budget_*.csv`,
  `../pt4/r1shadow/r1_shadow_summary.csv`.
</content>
</invoke>
