# Phase plan — Geometry-set covering-fraction leak (`coverFraction`)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time plan/skeleton, not a maintained spec; the code moves faster than
> these notes (paths, line numbers, and "what shipped" status drift).
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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure under `docs/dev/scratch/` as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> **Note (2026-06-08, not re-audited):** the `coverFraction`/`Cf` feature has
> progressed since this skeleton (commits `#641 Feature/add cf`, `#647`/`#648`
> plot-cf, plus the `rosette_cf*` params). The Status line below predates that —
> re-verify which phases actually shipped before relying on it.

> **Status (v2): Phase A–C drafted on branch `feature/add-Cf`.** The energy leak is wired;
> the **mass sink (Phase D) and X-ray/photon work (Phase G) are NOT implemented** and several
> edge cases are open (§7). Decisions Q1–Q4 are resolved per the leakage spec v2 (its §S5).
> Line numbers drift — locate edits by function and quoted strings, not by number (spec S6).

## 0. Scope & invariants

- **One new input**: `coverFraction` (`Cf`), default `1.0`, validated `0 < Cf ≤ 1`.
- **Hard invariant (merge gate)**: `Cf = 1 ⇒ Lleak ≡ 0`, byte-for-byte identical trajectory.
  Verified: `coverFraction` loads as *exactly* `1.0`, the helper's `Cf ≥ 1` guard returns
  `0.0`, and `Ed = … − 0.0` is bit-identical to the old `… − 0`.
- **Branch**: `feature/add-Cf` (renamed from the session branch; the spec's suggested
  `rosette/cf-leak`/`feature/leaking-luminosities` names were superseded by the user).
- This still edits the core energy equation ⇒ **Joel sign-off before merge**.

## 1. What is implemented in A–C (verified against the code)

| File · symbol | Change |
|---|---|
| `bubble_structure/get_bubbleParams.py` · **`get_leak_luminosity`** (new) | `Lleak = γ/(γ−1)·(1−Cf)·4πR2²·Pb·c_sound`. Returns `0.0` when `Cf≥1`, `Pb≤0`, or `c_sound≤0` (self-limiting; never injects energy). Code units throughout — no conversion. |
| `phase1_energy/energy_phase_ODEs.py` · `ODESnapshot` | added frozen fields `coverFraction`, `c_sound` (hot, from `bubble_Tavg`); populated in `create_ODE_snapshot`. |
| …  · `get_ODE_Edot_pure` | replaced `L_leak = 0` placeholder with a live call (live `press_bubble`,`R2`; frozen `Cf`,`c_sound`). |
| …  · `ODEResult` + `compute_derived_quantities` | new `bubble_Leak` field, computed and returned (pure RHS can't write; this is the recording path). |
| `phase1_energy/run_energy_phase.py` | writes `ode_result.bubble_Leak → params['bubble_Leak']`. |
| `phase1b_energy_implicit/run_energy_implicit_phase.py` | sets `params['bubble_Leak']` from the helper **before** `solve_betadelta_pure` (which already sums it into `L_loss`). 1-step frozen `Pb`/`c_sound`, consistent with the phase's slowly-varying `Lloss` treatment. |
| `_input/registry.py` | new `coverFraction` `ParamSpec` (input_physical, run_const, default `1.0`) + `_validate_coverFraction` (0<Cf≤1). |
| `_input/default.param` | regenerated via `python -m tools.gen_default_param --write`. |
| `test/test_cf_leak.py` (new) | helper correctness, `Cf=1`→0, guards, monotonicity, validator, and the **unit-landing** assertion (`Pb·cs·R²` → `Msun·pc²/Myr³` with no hidden factor). |

`c_sound` provenance: `run_energy_phase.py` sets `params['c_sound'] = get_soundspeed(Tavg, params)`
(Tavg = `bubble_data.bubble_Tavg`) **before** the snapshot is built, so the snapshot's `c_sound`
is the hot-bubble value the spec requires. The implicit phase mirrors this (`get_soundspeed(bubble_Tavg)`).

## 2. Resolved decisions (spec v2 §S5)

- **Q1 — transition double-count: none; keep one `Lleak`.** The transition forms
  `Ėb = min(energy-balance, −Eb/(R2/cs))` — a *selector*, not a sum — so the leak entering the
  energy-balance branch can't double-count with the sound-crossing drain. No phase gate on
  `Lleak`. **Do not** turn the `min` into a sum. (For `Cf~0.9–0.99` the sound-crossing term
  usually wins the `min`; the leak's real effect is earlier — lowering `Eb` in energy/implicit
  and bringing the transition on sooner, at smaller `Rb`.)
- **Q2 — mass sink at the density level (Phase D, not yet done).** Net flux
  `Ṁb = Ṁevap − Ṁleak`. Impose on the retained mass / the density fed to `Lcool` and the X-ray —
  **not** on `bubble_dMdt` (the Weaver conduction BC). Because `Pb` is energy-closed, the mass
  sink does not move `Rb`; it fixes `Lcool`, the X-ray, and the hot-gas mass. Option A
  (density-level reduction) first; Option B (an explicitly tracked `Mb`) only if the X-ray needs
  a handle independent of the energy sink.
- **Q3 — covering-fraction enthalpy leak with the mass sink** (not the constant-θ proxy). ✅ energy half done.
- **Q4 — photon coupling `fleak = 1−Cf`**, same geometry, at the CLOUDY post-processing stage. Not in the ODE.

## 3. Phase status

```
A. Plumbing (param, snapshot, output, helper) ........ DRAFTED
B. Phase diagnosis at stellar age (read-only go/no-go)  PENDING run (needs full trajectory; see §6)
C. Energy leak apply (explicit RHS + implicit) ....... DRAFTED
D. Mass sink (Q2, density level) ..................... NOT STARTED — see open edge cases §7
E. Transition: nothing to do (Q1 = selector) ........ N/A by decision
F. Floor / velocity characterization (S4.5) ......... PENDING (needs Cf<1 runs)
G. Photon budget (Q4) + X-ray calibration (S4.6) .... NOT STARTED (no L_X in code; CLOUDY stage)
```

## 4. Verification status

- `test/test_cf_leak.py`: **16/16 pass**.
- Codegen gate `tools.gen_default_param --check`: **in sync**.
- Full suite: **369 pass**; the one failure is `test_run_smoke.py::test_quickstart_completes_cleanly`,
  a `MonotonicError` in the bubble-structure integrator that occurs **before** any leak code runs
  in the loop. **Confirmed pre-existing flakiness, not a regression**: with *identical* (my) code
  the smoke test gave fail, fail, pass, pass across 4 runs (a thread-pinned run also passed). Same
  code → both outcomes ⇒ environmental (the repo's own `requirements.txt` documents this class of
  non-deterministic breakage). Also consistent with `Cf=1 ⇒ Ėb` bit-identical to baseline.
- Numeric end-to-end check (real functions): `cs(10⁶ K)=150 km/s`; `Cf=1 ⇒ Lleak=0` exactly;
  `t_leak = Eb/Lleak = 13,010 yr` at `Cf=0.9`, matching the analytic `R/(3γ(1−Cf)cs)` and the
  spec's ~10⁴ yr. Confirms the enthalpy coefficient and unit landing in practice.

## 5. Remaining gates to run (S4)

1. **Regression (merge gate)** — `Cf=1` reproduces `Rb(t)`,`Eb(t)`,X-ray. Holds by construction (guard returns 0.0).
2. **Phase diagnosis** — read the phase at the stellar age in an unmodified rosette run; energy/transition ⇒ leak is the right lever.
3. **Monotonicity** — `Cf ∈ {1.0,0.95,0.9}` ⇒ smaller `Rb`,`Eb`; `bubble_Leak` positive.
4. **Energy/mass audit** — needs Phase D for the mass half.
5. **Floor/velocity** — smallest stable `Cf`: `Rb` floors near the momentum-limited radius (>7 pc), `vb` ≪ observed.
6. **X-ray consistency** — needs Phase G.

## 6. Notes for whoever runs the trajectories

- A full rosette trajectory needs SPS + non-CIE cooling tables (`lib/default/opiate/`, `SB99_rotation=1`).
- The bubble-structure integrator is numerically touchy (`MonotonicError`, "excess work" warnings) at
  a knife-edge that multithreaded BLAS can flip run-to-run. If the smoke test is flaky, pin
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to test determinism before blaming a code change.

## 7. Open edge cases / unanswered questions (carry into Phase D and review)

These are **not** handled by the A–C draft and need decisions:

1. **`Mb = 0` (or → 0) with `Ṁb < 0`.** When the mass sink (Phase D) drives the retained hot-gas
   mass to zero while `Ṁleak` still wants to remove mass, the density fed to `Lcool`/X-ray must not
   go negative or divide-by-zero (`n = Pb/(2 k_B T)` already shows `divide by zero` warnings in the
   structure code). Need a floor (`Mb ≥ 0`, leak mass-rate clamped to available mass per step) and a
   decision on what "empty hot bubble" means physically (does the leak then switch off? does `Pb`
   still come from `Eb/V`?). **Open.**
2. **`Eb` depressurising within one RK step at small `Cf`.** `Lleak ∝ Pb ∝ Eb` self-limits, but a
   large `(1−Cf)` with the short hot-gas `cs` (~10⁴ yr leak time) can overshoot `Eb` toward/below 0
   inside a single step and stress the integrator. Current guard zeroes the leak at `Pb ≤ 0`, but the
   step that crosses zero is not substepped. Recommended usable range `Cf ~ 0.9–0.99` (documented on
   the param). **Open: do we need an Eb floor / adaptive substep?**
3. **Evaporation scaling by `Cf`.** Only the *intact* fraction of the wall should evaporate, so
   arguably `Ṁevap → Cf·Ṁevap`. The spec calls this a secondary effect to decide with Joel; the draft
   does **not** touch `bubble_dMdt`. **Open.**
4. **Implicit-phase 1-step staleness.** `bubble_Leak` there uses the previous segment's `Pb`,`c_sound`.
   Fine for slowly-varying conditions; revisit if `Cf` is small enough that the leak changes fast within a segment.
5. **X-ray / retained-pressure calibration target is absent** (no `L_X` in the code). Calibration of
   `Cf` to the Townsley plasma pressure (spec §calib) lands in CLOUDY post-processing — out of scope for A–D.
6. **Transition `min` vs leak interaction at `Cf < ~0.8`.** By Q1 this is correct-by-construction, but
   it has not been exercised numerically; confirm continuity of `Ėb` across the transition entry once Cf<1 runs exist.

### Audit findings (from the verification pass) — to resolve with Joel

7. **Transition-phase leak uses the *effective* pressure, not thermal `Pb`.** `get_ODE_Edot_pure`
   computes `Lleak` from `press_bubble = get_effective_bubble_pressure(...)`, which in the **energy/
   implicit** phases is the thermal `Pb = bubble_E2P` (✅ correct enthalpy-flux pressure), but in the
   **transition** phase is `max(P_thermal, P_ram)`. An enthalpy flux of escaping hot gas should use the
   hot-gas thermal pressure; when `P_ram` dominates late in transition it inflates `Lleak`. The spec
   says "reuse `press_bubble`," and the transition `min()` selector usually picks the sound-crossing
   drain so the inflated value is often masked — but at low `Cf` it is not. **Recommendation:** compute
   the transition leak from thermal `bubble_E2P(Eb,R2,R1,γ)` explicitly (R1 is already in scope), or
   confirm reusing `press_bubble` is intended. Energy-phase behaviour is unaffected either way.
8. **`bubble_Leak` diagnostic is not refreshed in the transition phase.** The leak is *applied* there
   (inside `get_ODE_transition_pure → get_ODE_Edot_pure`), but the transition runner never updates
   `params['bubble_Leak']` (no `compute_derived_quantities` call), so the plotted `bubble_Leak` is stale
   (last implicit value) during transition. Diagnostic-only; the dynamics are unaffected. **Fix:** record
   the applied leak in the transition diagnostic/save path.
9. **Pre-existing (not introduced here):** `operations.get_soundspeed` docstring says "isothermal" but the
   formula `sqrt(γ k_B T/μ)` is the *adiabatic* sound speed — which is what the spec's `cs` wants, so the
   **code is correct** and matches the leak Eq.; only the docstring is stale. Left untouched (surgical).
10. **Phase-D mechanism is under-specified given that density is `Pb`-derived (key design gap).**
    Trinity reconstructs `n(r) = Pb/(2 k_B T)` and `bubble_mass = ∫ n μ 4πr² dr` every step, and `Pb`
    is closed from `Eb`. So the **already-implemented energy leak** (lowering `Eb → Pb → n`) *already*
    lowers the reconstructed density and mass — but by an energy-coupled amount, **not** by the advective
    `Ṁleak = (1−Cf)·4πR²·ρ·cs`. A naive "density-level reduction" (Q2 Option A) layered on top would
    therefore **double-count** the `Pb`-mediated drop. To impose the *advective* mass deficit specifically,
    Phase D likely needs a tracked deficit factor `f_M = M_tracked / M_structure` applied only where density
    feeds `Lcool`/X-ray (effectively Q2 Option B), combined with the `Mb=0 / Ṁb<0` clamp of item 1. The
    exact coupling (and whether any separate mass sink is even needed beyond the `Pb` effect) must be
    settled **before** coding D — this is the crux of the mass half.
```
