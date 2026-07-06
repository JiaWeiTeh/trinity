# TRINITY pdv-trigger — session handoff / summary (2026-07-01)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

> 🚨 **READ THIS FIRST — THIS IS SPECULATION, NOT VERIFIED TRUTH.** Everything below is the *reasoning and
> partial results of one working session*, much of it **single-config, early-time, or interrupted by a
> restart-prone, slow compute environment**. Several claims in this session were **stated confidently and then
> retracted when data contradicted them** (see "Retractions" — that pattern is the point). **Do NOT trust any
> statement here until you re-verify it against the current source and against ≥5 Myr runs.** If you carry this
> to another chat: treat every bullet as a *hypothesis to check*, not a fact.

**Repo:** TRINITY (`trinity-sf`), feedback-bubble astrophysics code. **Branch:** `feature/PdV-trigger-term-pt2`
(50 commits ahead of `origin/main`, contains all of main). **Workstream dir:** `docs/dev/transition/pdv-trigger/`.

---

## 1. One-paragraph context

TRINITY evolves an expanding feedback bubble and switches it from **energy-driven** (hot interior does PdV work
on the shell) to **momentum-driven** when it becomes cooling-dominated, measured by **θ ≡ L_cool/L_mech**
(the `cooling_balance` trigger fires at **θ ≥ 0.95**, i.e. net energy fraction `(L_gain−L_loss)/L_gain < 0.05`).
θ is *identical* in TRINITY, El-Badry+2019, and Lancaster. **The problem:** TRINITY's 1-D cooling **under-cools**
(native θ peaks ~0.66 dense / ~0.17 diffuse) so realistic clouds *never* transition — although El-Badry &
Lancaster say θ should be ~0.9–0.99. This workstream is about **making θ physical and cloud-property-dependent**.

## 2. The production knobs (all gated, default-off, `cooling_boost_mode='none'` = byte-identical)

| knob | what it does | enters structure ODE? | θ is… |
|---|---|---|---|
| `multiplier` (`cooling_boost_fmix`) | `L_loss = L_leak + f_κ·L_cool` (scales resolved L_cool **after** the structure solve) | **No** | emergent L_cool **× scalar** |
| `theta_target` (`cooling_boost_theta`) | `L_loss = max(L_cool+L_leak, θ·L_mech)` (**enforces** θ) | No | **enforced** |
| `cooling_boost_kappa` | multiplies Spitzer conduction `C_thermal` **inside** the structure ODE | **Yes** (`bubble_luminosity.py:291/370/406`) | **fully emergent** (but raises evaporation) |

"θ_elbadry" was a proposed 4th mode that computes `theta_target` from El-Badry's closed form
`θ(n)=A_mix√(λδv·n)/(11/5+A_mix√(λδv·n))` (A_mix=3.5, λδv≈3, θ_max=0.99). **It was demoted this session (see §4).**

## 3. What we actually did this session (chronological, with the decision at each step)

1. **Ran the "θ_elbadry" Stage-A shadow** (impose El-Badry θ as target) on 9 configs via a monkeypatch harness
   (no production edit). Found: the `max()` gate is safe; θ(n) behaves; **n_fire ≈ 48 cm⁻³**; dense clouds fire
   then **SHELL_COLLAPSE**; a θ_max sweep showed the cap is not the lever. → `FINDINGS §8/§8a`.
2. **The maintainer merged PR #715 to main** (high-mass handoff: phase 1b routes a finite `Eb≤0` collapse to the
   momentum phase instead of dead-stopping; phase 1a gets `cooling_balance`). Merged it in and **re-ran**.
   **Result: imposing El-Badry θ REVERSES the fix** — default `fail_repro` expands to 500 pc, but with θ imposed
   it velocity-runs-away inward. **Root cause: the high-mass turnover is PdV/inertia-driven, not radiative;
   El-Badry θ is a radiative ratio, so enforcing `L_loss=θ·L_mech` double-counts the PdV loss.** → `FINDINGS §8b`.
3. **DIRECTION CORRECTION (maintainer steer): θ is an OUTPUT, not an input.** Enforcing θ (`theta_target`
   /θ_elbadry) is the wrong primitive; **boost the cooling MECHANISM and let θ emerge**, using El-Badry/Lancaster
   only to **calibrate** what θ should emerge to. θ_elbadry demoted to an **opt-in override**. Reconciled all
   sibling docs. → `FINDINGS §8c`, `PLAN.md` ⭐⭐.
4. **Calibrated f_κ(n)** so emergent θ hits El-Badry's λδv=3 target (§14). **Caveat found:** θ₀ was measured at
   **blowout** — a bad epoch. → adopted a **standing rule: θ = peak over a ≥5 Myr run (θ_max)**, blowout retired.
5. **Design decision (maintainer): use a SINGLE physical f_κ constant, NOT a steep f_κ(n) formula.** The physical
   enhancement `κ_mix/κ_Spitzer ∝ n` *rises* with density — opposite the "chase-El-Badry" f_κ(n) — so no physical
   f_κ(n) fires every cloud. Density-dependence should **emerge** as a **route-a critical density** (below which
   clouds stay energy-driven by design). → `F_KAPPA §14 DECISION`.
6. **KNOB ERROR discovered:** the §14 calibration (θ₀, leverage p) was fit with **`cooling_boost_kappa`**, but
   the validation runs used **`multiplier`** — *different knobs*. So the §14 validation numbers do not validate
   the calibration. → `PLAN.md` KNOB CORRECTION.
7. **Re-ran the validation with the correct knob (`cooling_boost_kappa`) + fixed the misleading `min_T` DEBUG log
   + documented the single-f_κ decision.** **Result: the structural knob BREAKS DOWN** — at f_κ=8 it drives the
   solver to non-physical dMdt (the evaporation side-effect), θ freezes at ~0.5 and does **not** fire; kappa=2 is
   stable but **too slow** (never clears phase 1a). Emergent θ under kappa is monotonic in density
   (0.25/0.48/0.53 for n=100/1e4/1e5) but all early-time and sub-threshold. → `FINDINGS §8e`.

## 4. Current *tentative* direction (⚠️ unverified — this is where we landed, not proven)

- **θ is an OUTPUT.** Boost the cooling **mechanism**; let the solved bubble produce θ. El-Badry/Lancaster =
  **calibration target**, never enforced.
- **Mechanism knob — tentatively `multiplier`.** `cooling_boost_kappa` (the "fully emergent" one) is fragile
  (breaks at f_κ=8) *and* slow (enters the structure ODE); structural κ_mix (Rung B) is **SHELVED** (unstable);
  so `multiplier` (stable, fast, radiative-only → no PdV double-count) is the pragmatic choice — θ still emerges
  from the structural L_cool, just scaled. **NOT proven to be right; it's the least-bad option so far.**
- **Single physical f_κ constant** (~few), not f_κ(n). Density-dependence = the **route-a critical density**.
- **Accept route-a:** clouds whose emergent θ never reaches 0.95 stay energy-driven **by design** ("diffuse
  clouds may never enter momentum — the physics never allows it").
- **Massive/PdV-dominated clouds** ride the **PR #715 `Eb≤0→momentum` handoff**, untouched by θ.
- **`theta_elbadry`/`theta_target` = documented opt-in override** for users who explicitly want forced cooling.

## 5. Open questions / NEXT STEPS (what we'd do next — none of this is done)

1. **Re-derive the f_κ leverage/θ₀ for the `multiplier` knob** (the one we'd ship). §14's θ₀/p were fit on
   `cooling_boost_kappa` and **do not carry over** — `multiplier`'s crude `8×L_cool` θ (fires easily) ≠ kappa's
   back-reacted θ (~0.5). This is the single most important next task. A small `multiplier` sweep (f_κ ∈ {1,2,4,8}
   × a few configs), θ measured as **θ_max over ≥5 Myr**, harvested from `dictionary.jsonl` `bubble_Lloss/Lmech`
   (NOT the contaminated observer — see Retraction R6).
2. **Decide the mechanism knob definitively:** `multiplier` vs a low-f_κ `cooling_boost_kappa` vs revisiting the
   shelved structural κ_mix. Needs runs that reach the θ **peak** (~blowout, t~0.05–0.1 Myr) and ideally ≥5 Myr.
3. **Compute the route-a critical density** at the chosen physical f_κ, as a **falsifiable** energy→momentum
   split to test against observations.
4. **The diffuse-handoff performance cliff** (`FINDINGS §8d`): diffuse configs crawl at the 1a→1b handoff (small
   R2, fast v2), ~11 h to reach 6 Myr even at f_κ=1. This is a bubble-structure/`dt` **performance** item (out of
   this workstream's scope) but it **blocks** diffuse validation in a normal environment → **use HPC**.
5. **min_T DEBUG-log fix** (`bubble_luminosity.py`, added `_MINT_LOG_TOL`): behaviour-neutral, applied this
   session, `test_run_smoke` passed — candidate to keep/upstream.

## 6. Retractions this session (⚠️ IMPORTANT — do not repeat these mistakes)

- **R1** "Dense clouds recollapse — probably physical" → refined: it was the interaction of θ-imposition + the
  old dead-stop; on merged code, imposing θ *reverses* the PR #715 handoff (§8b).
- **R2** "The diffuse runs STALL/hang" → **wrong: they're just slow** (they progress; beta-delta converges to
  g~1e-15). It's a **performance cliff**, not a stall/convergence bug (§8d).
- **R3** "The `min_T` guard is the bug" → **wrong: benign** (rejection penalty ≈ 1.0); a red-herring DEBUG log
  that misled the investigation (§8d). The guard is correct; only the log was noisy (now fixed).
- **R4** "high f_κ stiffens the ODE" → **wrong: `multiplier` never enters the structure ODE.** It's slower only
  via faster Eb evolution → smaller adaptive `dt` (§8d).
- **R5** "the §14 validation confirms the calibration" → **wrong knob:** calibration fit on `cooling_boost_kappa`,
  validated with `multiplier` (§8e / PLAN KNOB CORRECTION).
- **R6** the θ_max "observer" that records every `effective_Lloss` call is **contaminated** by the solver's
  non-physical trial (β,δ) points (reported a bogus θ_max=3.22). **Use `bubble_Lloss/Lmech_total` from
  `dictionary.jsonl` at accepted segments** for the true emergent θ.

## 7. Workflow / conventions (so the next session/chat stays on the rails)

- **Branch discipline:** develop + push **only** to `feature/PdV-trigger-term-pt2` (`git push -u origin …`).
- **Commit messages:** NO Claude/AI attribution, NO session links, NO co-author trailers, NO model IDs — chat
  only. (Project CLAUDE.md rule.)
- **Hard guardrail:** **nothing touches the production solver** before it is tested (all relevant configs, units
  handled) and gated **default-off byte-identical**. Production physics is unchanged this session **except**: the
  behaviour-neutral `min_T` **log** fix, PR #715 (merged upstream), and an earlier Pb-collapse-guard hygiene fix.
- **Shadow-first:** test ideas via **monkeypatch harnesses** in `docs/dev/transition/pdv-trigger/data/` that
  patch `effective_Lloss_from_params` at runtime — **no `trinity/` edit** — run in **separate processes**
  (trinity leaks module-global state in-process).
- **📏 Standing rules:** (a) run every sim to **≥5 Myr** (θ peaks ~0.4–1 Myr; don't truncate for cheapness);
  (b) **θ is measured as its PEAK over the run (θ_max)**, never at blowout.
- **Compute reality:** the container is **ephemeral and restart-prone**; dense/diffuse runs are **hours-scale**;
  several runs this session were OOM-killed or timed out. **Commit + push frequently.** For anything conclusive,
  **use HPC** and run in separate processes at matched simulation time.
- **Docs discipline:** every `docs/dev/` doc carries the 4 banner paragraphs (stale-warning / living / persist /
  cross-check); when one number changes, **reconcile all sibling docs** so none disagree.

## 8. Key files & artifacts (all committed on the branch)

- **`PLAN.md`** — ⭐⭐ CANONICAL SYNTHESIS + VERDICT (current direction) + KNOB CORRECTION + dated status ledger.
- **`FINDINGS.md`** — §8 Stage-A shadow, §8a θ_max sweep, §8b PR#715 reversal, §8c direction correction, §8d
  diffuse performance-cliff diagnosis, **§8e correct-knob (kappa) validation** (the newest).
- **`F_KAPPA_FUNCTIONAL_FORM.md`** — the emergent-θ calibration; **§14** = current calibration + the single-f_κ
  DECISION. (θ₀/p were fit on `cooling_boost_kappa`.)
- **`ELBADRY_THETA_STORY.html`** — illustrated 9-chapter walkthrough (6 figures) incl. the correction chapter.
- **`ELBADRY_REFERENCE.md` / `LANCASTER_REFERENCE.md`** — the distilled papers (θ, closed form, λδv≈3, route-a).
- **`THETA_ELBADRY_SPEC.md`** — now the **opt-in override** spec (incl. §0.6 "θ applied continuously, not at
  blowout").
- **`KAPPA_VALIDATION_PLAN.md`** — this session's working plan (T1–T5, all done).
- **`INDEX.md`** — master map + staleness audit.
- **Harnesses** (`data/`): `_theta_elbadry_runner.py`, `_theta_elbadry_gated_runner.py`,
  `_fkappa_validation_runner.py`, `_kappa_validation_runner.py`, `_baseline_runner.py`, `harvest_shadow.py`,
  `make_fkappa_emergent_calibration.py`, `make_elbadry_story_figs.py`.
- **Key commits (newest→older):** `54c99c10` (§8e small_1e6) · `d207b2f7`/`fe171b20` (§8e kappa breakdown) ·
  `517c7503` (min_T log fix + single-f_κ decision) · `ca0c8e36` (knob-mislabel correction) · `dd222c92` (§8d
  perf-cliff) · `f125de65` (θ_max rule + validation) · `a5cdf63a` (f_κ calibration) · `e38fa7a4` (DIRECTION
  CORRECTED) · `86950d2d` (PR#715 reversal) · `45cd499c` (merge main/PR#715).

## 9. Reproduce (examples — expect them to be SLOW; use HPC for anything conclusive)

```bash
# emergent-theta calibration figures (no sims, reads committed CSVs):
python docs/dev/transition/pdv-trigger/data/make_fkappa_emergent_calibration.py

# correct-knob validation (cooling_boost_kappa; the one to re-do properly at >=5 Myr):
FK=8 STOP_T=6 OUT_BASE=outputs/kappa_val \
  python docs/dev/transition/pdv-trigger/data/_kappa_validation_runner.py <config.param> <name>

# the knob we'd actually ship (re-derive its f_k here — NEXT STEP #1):
#   set params['cooling_boost_mode']='multiplier', params['cooling_boost_fmix']=f_k
# measure emergent theta from dictionary.jsonl: theta = bubble_Lloss / Lmech_total  (finite, accepted segments)
```

---
*Generated 2026-07-01 for handoff. Branch `feature/PdV-trigger-term-pt2`. Everything above is this session's
working reasoning — **unverified; verify before trusting.***
