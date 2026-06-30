# `theta_elbadry` mode — implementation spec (the capstone; consolidates every resolved decision)

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time implementation spec.
> **Re-check each line/site against current source before applying.** Written as a PLAN.
>
> 🔄 **Living plan — recheck and refine on every visit.** Re-verify line refs; update drift; rethink the design
> if a cleaner one exists; date changes. **Keep all banner paragraphs.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The decisions here are backed by committed artifacts:
> `make_elbadry_theta.py`, `make_nmap_verify.py`, `make_kmix_selfconsistent.py` (+ the reference docs).
>
> 🔗 **Cross-check siblings:** `PLAN.md` (⭐⭐ canonical synthesis — the source of these decisions),
> `ELBADRY_REFERENCE.md` (θ, closed form, n-mapping, theta_target verification), `LANCASTER_REFERENCE.md`
> (λδv≈3, route-a). If a number here disagrees with those, reconcile — update the canonical synthesis + §1.5
> staleness audit together.

---

## 0. What this is, status, and the guardrail

The **single implementation-ready design** for wiring El-Badry's analytic θ into TRINITY as a gated trigger
target. It is the capstone of the workstream: every decision below is already settled and evidenced (see the
canonical synthesis). **STATUS: spec only — NO production code is changed by this document.** Under the
maintainer guardrail, nothing ships until it is gated default-off **byte-identical**, tested on all 8 configs,
each run to **≥5 Myr**.

> **🧪 TWO-STAGE PLAN — SHADOW FIRST, then production (maintainer decision 2026-06-30).** Production is **NOT
> finalized** by this spec: two design points (§6 the `max(resolved,target)` choice; §7 the fate pattern) are
> *unresolved-until-data*, and TRINITY has never been run end-to-end with any θ-boosting mode. So:
> - **STAGE A — SHADOW (no production edit):** apply the §3 logic via a runtime **monkeypatch** of
>   `effective_Lloss_from_params` (a dev-only `data/` harness, like every other `make_*.py`), run the 8 configs
>   to ≥5 Myr, harvest θ(t)/firing/fate, and **resolve §6 + §7 from the data.** Nothing in `trinity/` changes.
> - **STAGE B — PRODUCTION (only after Stage A passes):** make the §2/§3 change for real, gated default-off
>   byte-identical, with the now-data-informed §6 choice locked in.
> The spec sections below describe the *final* code (Stage B); Stage A runs the **identical logic** by
> monkeypatch so the shadow result transfers 1:1.

**One-line design:** add a `cooling_boost_mode='theta_elbadry'` that, each step, sets the target loss fraction
to **θ = A_mix·√(λδv·n_amb(R2)) / (11/5 + A_mix·√(λδv·n_amb(R2)))** (capped at θ_max) and feeds it through the
*already-verified* `theta_target` `(1−θ)` budget. No κ_mix port, no structural solve change.

## 1. The formula + the pinned constants

```
X      = A_mix · sqrt( lambda_dv[pc·km/s] · n_amb[cm^-3] )          # El-Badry Eq 37
theta  = min( X / (11/5 + X) , theta_max )                          # Eq 38, capped
A_mix  = 3.5     (El-Badry fit; hard constant)
lambda_dv ≈ 3.0  (CALIBRATED: Lancaster GMC range + El-Badry A_mix fit — LANCASTER_REFERENCE §7)
theta_max = 0.99 (ceiling; else (1-theta)^{1/5}→0 stalls the bubble at GMC-core density)
```
- `n_amb` = **local pre-shock cloud density at the shell**, `get_density_profile(R2, params)` (returns pc⁻³ →
  ×`ndens_au2cgs` = /2.938×10⁵⁵ → cm⁻³). TRINITY's n is already n_H — matches El-Badry (no μ conversion).
- **Firing threshold at λδv=3:** `n_fire ≈ 50 cm⁻³`. GMC cores fire; diffuse clouds (nH≲50) stay energy-driven
  = **fate / route-a** (`make_elbadry_theta.py`).
- θ is the **late-time equilibrium** value (Δt_SNe-independent — `ELBADRY_REFERENCE.md` §8); valid for TRINITY's
  continuous SB99 input.

## 2. Registry params (mirror the `cooling_boost_*` family, `trinity/_input/registry.py:349–352`)

Add **'theta_elbadry'** to the `cooling_boost_mode` token list, and two new params (after `cooling_boost_theta`):

| param | default | meaning |
|---|---|---|
| `cooling_boost_mode` | `'none'` | add token `'theta_elbadry'` = `max(Lcool+Lleak, θ(λδv,n_amb(R2))·Lmech)`, θ from Eq 37/38 |
| `cooling_boost_lambda_dv` | `0.0` | λδv [pc·km/s]. **0 ⇒ θ=0 ⇒ off** (double-safe with mode). Set **3.0** to enable the calibrated model. |
| `cooling_boost_theta_max` | `0.99` | θ ceiling (must be <1) |

All `category='input_solver'`, `run_const=True`, `exclude_from_snapshot=True` — like the rest of the family.
**Default `mode='none'` ⇒ byte-identical**; the extra `λδv=0` default is a second off-switch.

## 3. Code integration — ONE site, reusing the verified budget

The only change is in **`trinity/phase1b_energy_implicit/get_betadelta.py`** at `effective_Lloss_from_params`
(:360) — the single wrapper all three call sites (`:473`, `:577` residual; `run_energy_implicit_phase.py:1154/
1158` trigger) already route through. Compute θ there, then reuse the existing `theta_target` arithmetic:

```python
# trinity/phase1b_energy_implicit/get_betadelta.py
from trinity.cloud_properties.density_profile import get_density_profile   # new import
import trinity._functions.unit_conversions as cvt                          # for ndens_au2cgs

_A_MIX = 3.5   # El-Badry+2019 Eq 37 fit constant (LANCASTER_REFERENCE confirms via λδv≈3)

def effective_Lloss_from_params(params, Lcool, Lleak, Lmech):
    mode = getattr(params.get('cooling_boost_mode', None), 'value', 'none') or 'none'
    if mode == 'none':
        return Lcool + Lleak                                  # BYTE-IDENTICAL
    if mode == 'theta_elbadry':
        ldv = float(getattr(params.get('cooling_boost_lambda_dv', None), 'value', 0.0))
        if ldv <= 0.0:
            return Lcool + Lleak                              # off (second switch)
        n_amb = float(get_density_profile(params['R2'].value, params)) * cvt.ndens_au2cgs  # pc^-3 -> cm^-3
        X = _A_MIX * (ldv * n_amb) ** 0.5
        theta = X / (11.0/5.0 + X)
        theta = min(theta, float(getattr(params.get('cooling_boost_theta_max', None), 'value', 0.99)))
        return max(Lcool + Lleak, theta * Lmech)              # same (1-θ) budget as theta_target
    # existing 'multiplier' / 'theta_target' branches unchanged ...
```

That's it — the `(1−θ)` energy budget, the β-δ residual, and the trigger are **already** verified to consume
this consistently (`ELBADRY_REFERENCE.md` §9). The only new physics is the **per-step density-dependent θ**.

**Units note (the recurring bug class):** λδv stays in pc·km/s and n in cm⁻³ — A_mix=3.5 is the dimensionless
fit for *exactly those units* (El-Badry Eq 37). The one conversion is `get_density_profile` pc⁻³ → cm⁻³.

## 4. Byte-identical-off proof

`mode='none'` (default) returns `Lcool+Lleak` on the first line — the exact pre-change expression ⇒
**bit-identical**. Gate: run the 8 configs with default params before/after the patch; `dictionary.jsonl`
differs only in the 3 nondeterministic ~1e-22 SN-noise terms (the known BLAS non-reproducibility,
`PB_COLLAPSE_GUARD_FIX.md` §5.2), all physics fields identical.

## 5. The trigger pairing (PdV) — set `transition_trigger`, independent of the cooling mode

`cooling_boost_mode` (cooling) and `transition_trigger` (when to fire) are **orthogonal params**. θ is
PdV-**exclusive**; for massive clusters the PdV-inclusive `ebpeak` (`Edot_from_balance = Lmech − Lloss −
4πR2²v2·Pb ≤ 0`) fires earlier and is more physical. **So when enabling `theta_elbadry`, also set
`transition_trigger='cooling_balance,ebpeak'`** (fires on whichever first). Do NOT auto-couple them in code —
keep it a `.param` choice (the boosted Lloss flows into `ebpeak` automatically via `Edot_from_balance`).
**Firing is assessed by FIRST-CROSSING** (the trigger fires the first time θ_eff≥0.95 or Edot_balance≤0) — since
θ tracks n_amb(R2), which peaks early in the dense core, firing is an early event; **never read firing at
blowout** (the trap from `KMIX_SELFCONSISTENT.md` §2b).

## 6. The `max(resolved, target)` subtlety — a gate, not an assumption

`effective_Lloss` uses `max(Lcool+Lleak, θ·Lmech)` (single-count, El-Badry's intent: θ is the *total* cooling
fraction, resolved ≤ total). **Risk:** TRINITY's *resolved* 1D θ has the WRONG density trend (high at diffuse),
so at low density the resolved `Lcool/Lmech` could exceed θ_elbadry and win the max — letting a diffuse cloud
fire that route-a says should not. **Gate (in the test plan):** log, per config, the fraction of steps where
`Lcool > θ_elbadry·Lmech` (resolved wins). If it is non-negligible at the diffuse end, switch the diffuse
behaviour to **direct θ_target** (`return Lleak + theta*Lmech`) so El-Badry's θ governs. Decide from data,
don't assume.

## 7. Test plan (8 configs, each ≥5 Myr — the validation that takes it off paper)

The 8 configs (`INDEX.md` §3): `simple_cluster`, `midrange_pl0`, `be_sphere`, `pl2_steep`,
`large_diffuse_lowsfe`, `small_dense_highsfe`, `fail_repro` (heavy), `small_1e6` (control). **Each run to ≥5 Myr**
(per the standing rule — `PLAN.md`), unless the physics ends it sooner (blowout/collapse).

1. **Unit (fast):** a `pytest` case asserting `effective_Lloss_from_params` returns `Lcool+Lleak` for
   `mode='none'` (byte-identical) and, for `mode='theta_elbadry'` at a known (R2, λδv), the θ matches
   `make_elbadry_theta.theta(λδv, n_amb)` — ties the production path to the validated calculator.
2. **Byte-identical-off (gate):** §4 — the 8 configs, mode off, bit-identical physics.
3. **On-run validation (the deliverable):** the 8 configs with `mode='theta_elbadry'`, `λδv=3`,
   `transition_trigger='cooling_balance,ebpeak'`, to ≥5 Myr. Assert/record per config:
   - θ_eff(t) trajectory; **does it cross 0.95, and WHEN** (first-crossing, in the dense core);
   - the **fate pattern**: GMC cores (nCore≳50) transition, diffuse (nCore≲50) do not — matches `n_fire≈50`;
   - the resolved-wins fraction (§6 gate);
   - momentum / hot-gas / R(t) vs the Lancaster/El-Badry (1−θ) expectations where comparable;
   - `fail_repro` still terminates cleanly (no regression of the Pb-collapse fix).
4. **Equivalence depth (CLAUDE.md rule 5):** this is an iterative/trigger path — clear per-call (test 1) AND
   full-run (test 3) on the stiff edges, in **separate processes**, at **matched t**. Persist the θ(t) and
   fate table as a committed CSV/figure under `data/`.

## 8. Apply order — TWO STAGES (shadow first, production after)

**STAGE A — SHADOW (no `trinity/` edit; the gate before production):**
1. Build `data/make_theta_elbadry_shadow.py`: a runner that **monkeypatches** `effective_Lloss_from_params`
   (both `get_betadelta` and the name imported into `run_energy_implicit_phase`) to the §3 `theta_elbadry`
   logic, then runs each of the 8 configs to **≥5 Myr** via the normal pipeline (λδv=3,
   `transition_trigger='cooling_balance,ebpeak'`). Launch in the background (≤60 min each); monitor.
2. Harvest θ_eff(t), first-crossing firing time, the **fate pattern** (which configs transition vs stay
   energy-driven) and the **§6 resolved-wins fraction** → committed CSV/figure under `data/`.
3. **Resolve the open design points from the data:** fate pattern vs n_fire≈50; max() vs direct-θ_target (§6);
   any θ→1 numerical issues despite the ceiling; λδv fine-tune if needed. Update the spec + canonical synthesis.

**STAGE B — PRODUCTION (only after Stage A is clean):**
4. Make the §2 (3 params) + §3 (one `effective_Lloss_from_params` branch) change for real, with the
   data-informed §6 choice locked in.
5. Test 1 (unit: byte-identical-off + θ matches the calculator) → green; ruff F-rules; full `pytest`.
6. Re-run the 8 configs ≥5 Myr with `mode='theta_elbadry'` (now production) → must reproduce the Stage-A
   shadow result (1:1, since the logic is identical).
7. Commit to `feature/PdV-trigger-term-pt2`; **production default `mode='none'` — no behaviour ships**.
   Reconcile the ⭐⭐ canonical synthesis + `INDEX.md` §1.5 + this spec together.

*If Stage A misbehaves: that is a finding, not a failure — record it, revise the design, re-shadow. The point of
Stage A is that revising a harness is free; revising committed production is not.*

*Written 2026-06-30 on `feature/PdV-trigger-term-pt2`. No production code touched; spec only.*
