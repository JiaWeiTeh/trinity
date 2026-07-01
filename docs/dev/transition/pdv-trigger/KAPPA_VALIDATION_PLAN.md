# Correct-knob (`cooling_boost_kappa`) validation + min_T log fix — working plan (2026-07-01)

> ⚠️ **Point-in-time working plan, not a maintained spec** — re-verify against source before trusting.
> 🔄 **Living** — tick tasks as done, note deviations, date them.
> 💾 **Persist** — every result → committed CSV/figure under `data/` + this folder.
> 🔗 **Siblings** — `PLAN.md` ⭐⭐ (KNOB CORRECTION), `FINDINGS.md §8b–§8d`, `F_KAPPA_FUNCTIONAL_FORM.md §14`.

**Why:** the `2026-07-01` f_κ validation used `cooling_boost_mode='multiplier'`, but the §14 leverage/θ₀ were
fit with **`cooling_boost_kappa`** (the structural conduction boost — θ fully emergent). Different knobs. So the
§14 validation must be re-run with the SAME knob it was calibrated on. Plus a maintainer decision (single f_κ vs
f_κ(n)) and a flagged logging fix.

## Tasks

- [ ] **T1 — min_T log fix (production, behaviour-neutral).** `bubble_luminosity.py:345` logs
  `"Rejected. min T: 29999.99…"` for every *benign* boundary transient (penalty ≈1.0), which misled the §8d
  investigation. Gate the `logger.debug` so it only fires for a *real* sub-floor (min_T meaningfully below
  `_T_INIT_BOUNDARY`), leaving the residual/return value UNCHANGED. Verify byte-identical physics (pytest;
  the guard's return is untouched). *(subagent)*
- [ ] **T2 — record the design decision.** In `F_KAPPA §14` + `PLAN.md`: **use a single physical f_κ constant,
  not a steep f_κ(n) formula.** Rationale: the physical enhancement κ_mix/κ_Spitzer ∝ n *rises* with density,
  opposite to the "chase-El-Badry" f_κ(n) (which rises as n *falls*); so there is no physical f_κ(n) that fires
  every cloud. Set one physical f_κ and let the density-dependence emerge as θ(n) (denser → higher emergent θ →
  fires) and a **route-a critical density**. Match El-Badry where physical (dense); accept undershoot + route-a
  at the diffuse end.
- [ ] **T3 — correct-knob validation sims.** `data/_kappa_validation_runner.py`: production `cooling_boost_kappa`
  (with `cooling_boost_mode='none'`, so θ_emergent = (L_cool+L_leak)/L_mech with L_cool from the boosted
  conduction). Observe θ_max (peak, ≥5 Myr where feasible — the standing rule). Runs at a single physical
  **f_κ=8** on: dense `simple_cluster` (n=1e5, expect fire), boundary `be_sphere` (n=1e4), diffuse `small_1e6`
  (n=100, expect route-a). NB `kappa` enters the structure ODE (+ raises evaporation) so numerics may differ
  from the `multiplier` runs.
- [ ] **T4 — harvest + writeup.** Emergent θ(n) at f_κ=8 (kappa) vs the §14 prediction + the (retracted)
  `multiplier` numbers; the route-a threshold. Land in `FINDINGS §8e`; correct the §14 table's knob.
- [ ] **T5 — commit + push** (min_T fix, docs, runner + CSV/figure).

## Expected (from §14, to be confirmed with the correct knob)
At f_κ=8 the emergent θ should be ~0.99 (n=1e5, fire), ~0.91 (n=1e4, borderline), ~0.55 (n=100, route-a) — but
those were the `multiplier` predictions; `kappa` may land differently because it changes L_cool *through* the
structure (and raises evaporation). The test IS that comparison.
