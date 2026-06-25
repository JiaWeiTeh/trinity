# κ_eff / mixing-layer interface — feasibility & scoping (the "Option C" endgame)

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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

> **Provenance.** Feasibility mapped against current source 2026-06-25 (file:line below all spot-verified).
> This is the scoping doc for the "endgame" `PLAN.md` ("Outcome & pivot") points at after the cooling-boost
> program concluded that *a scalar magnitude knob cannot fix the transition* (see `FINDINGS.md`: constant-θ
> degenerate with the 0.95 trigger; `θ_target(Da)` refuted by the gate-validated replay).

## 0. Verdict (TL;DR)

**Is Option C possible? Yes — bounded, no hard showstoppers — and *more tractable than the methods note implied.*** It is a 2-rung ladder, not one thing:

- **Rung A — Spitzer-prefactor inflation (`C_thermal → f_κ·C_thermal`, keeping `T^(5/2)`): possible, ~hours.** 1 param + 3 one-line edits. Preserves the `T^(5/2)` form so the ICs and `(β,δ)` solver just re-converge. **Physically incomplete** — it raises evaporation *with* cooling (wrong sign vs El-Badry), so it is a back-reaction *probe*, not the faithful model.
- **Rung B — faithful mixing-layer `κ_eff`: possible, but a genuine re-derivation of ~3 functions in `bubble_luminosity.py` + a new `κ_mix` model + full re-validation.** Multi-day workstream with its own writeup. **No showstoppers**, but the crux (decoupling cooling-up from evaporation-down) must live *inside* the structure solve — and a naive post-hoc version has already been tried and **failed** (see §4).

## 1. Where Spitzer `κ_S = C·T^(5/2)` enters (3 sites, all `trinity/bubble_structure/bubble_luminosity.py`)

`C` is the param **`C_thermal`** (`trinity/_input/registry.py:341`, default `6e-7` cgs, `run_const`; `default.param:255`).

1. **Temperature ODE RHS** — `_get_bubble_ODE` (`:406`): `dTdrr = (Pb / (C_thermal * T**(5/2)) * (...) - ...)`. The `Pb/(C·T^(5/2))` conductive-flux divergence; `T^(5/2)` is Spitzer baked into the 2nd-derivative term (Weaver+77 Eq. 42–43).
2. **Near-front initial conditions** — `_get_bubble_ODE_initial_conditions` (`:364–383`): a **closed-form Spitzer solution** (Weaver Eq. 44) — `dR2 = T_init**(5/2)/(const·dMdt/4πR2²)`, `T = (...)**(2/5)`, `dTdr = -2/5·T/dR2`. The `2/5`, `5/2` powers are consequences of `κ ∝ T^(5/2)`. **This is the hardest piece to replace** (likely no clean closed form for a `max()`-type κ → needs a numerical near-front IC).
3. **`dMdt` initial guess** — `_get_init_dMdt` (`:292–295`, Weaver Eq. 33): `(t·C_thermal/R2²)**(2/7) · Pb**(5/7)`. The `2/7, 5/7` exponents come from inverting `T^(5/2)`.

The **`L_conduction` integral** (`:702–747`) is already κ-independent — it integrates the cooling table over the resolved band; only the *profile* it integrates changes.

## 2. What is NOT baked into Spitzer (lowers the cost vs the methods note)

- **The `(β,δ)` self-similar solver survives** [verified: 0 `T**(5/2)`/`C_thermal` matches in `get_betadelta.py`]. `cool_alpha/beta/delta` are Weaver *evolution* params (`registry.py:351–353`, `get_bubbleParams.py:42/63/111`), found by a root-find that drives an energy-balance + temperature residual to zero (`get_betadelta.py:475/491`). κ enters only because `(β,δ)` are passed *into* the structure ODE and the residual reads back `bubble_LTotal`/`bubble_T_r_Tb`. **So κ_eff shifts the converged `(β,δ)` values but does NOT force replacing the solver** — the note's "κ_S sets the (β,δ) parametrization" is an over-statement. The change is "re-derive ~3 functions," not "rebuild the self-similar machinery."
- **Evaporation does NOT mass-load the shell** [verified: `bubble_dMdt`/`bubble_mass` appear only in output field-lists + `shell_structure.py:103` recording; shell mass = swept-up cloud mass, `run_energy_implicit_phase.py:890`]. So the wrong-sign-evaporation problem bites only through *interior density → cooling luminosity → Eb budget*, **not** a shell mass sink — narrowing what κ_eff must get right.

## 3. The two rungs, concretely

| | Rung A — prefactor inflation | Rung B — faithful mixing-layer κ_eff |
|---|---|---|
| change | `C_thermal → f_κ·C_thermal` (1 param + edits at `:294/:373/:406`) | `T^(5/2)` factor → `κ_eff(T)`; re-derive ICs (`:364–383`, numerical) + `dMdt` seed (`:292–295`); new `κ_mix ~ ρ c_p D_turb`, `D_turb ~ R2·v2` |
| `(β,δ)` solver | unchanged (re-converges) | unchanged (re-converges) — see §2 |
| ICs | preserved (`T^(5/2)` intact) | **re-derived** (numerical near-front) |
| physics | raises cooling **and** evaporation (wrong sign) | decouples cooling-up / evaporation-down (the goal) |
| validation | test string-pins (`test_dR2min_magic_number.py`, `test_metadata.py:118`, `test_mu_audit_drift.py`) | rule-5 ladder + redo cleanroom C0 certification |
| effort | ~hours | multi-day workstream |
| status | **probe only** | **the endgame** |

## 4. The crux — and a documented prior failure

The real obstacle is the **cooling-vs-evaporation decoupling**: a larger scalar κ raises *both*, but El-Badry needs evaporation **suppressed 3–30×** while cooling rises. Only a state-coupled `κ_eff(ρ,T,R2,v2)` *inside* the structure solve achieves that.

**This has already been tried the naive way and failed** [`cleanroom/PLAN.md:484–509`, `cleanroom/FINDINGS.md:98–106`]: a post-hoc `θ·L_mech` bulk sink (`mixcool_whatif.py`) was magnitude-validated (θ≈0.25) but **numerically non-viable** — it drove `dMdt < 0` (no physical evaporation root) and stalled the hybr solver. Their recorded conclusion: *"a proper mixing-layer cooling must be integrated INTO the structure solve."* **That is the trap Rung B must thread: keep `dMdt > 0` self-consistent while raising the interface cooling.**

## 5. Why a scalar can't substitute (the 2026-06-25 validation — two runs)

Two live `theta_target` runs on the dense `f1edge_hidens` (vs its `none` baseline) show a constant θ does
**not** cleanly separate magnitude-correction from triggering:
- **θ=0.95** (= the trigger threshold): fires the cooling trigger **at birth** (t≈0.004 Myr) — the floor
  `0.95·Lmech` *is* the trigger condition. Matched-`t` ΔEb 34%.
- **θ=0.90** (below threshold): **also** fires, at t≈0.011 Myr (ΔEb 78%) — **NOT** the "magnitude-only"
  regime a naive reading predicts. Two reasons: (i) the floor removes energy early, *accelerating* the
  cloud toward the transition; (ii) a **dense** cloud's *resolved* `Lcool/Lmech` rises past 0.95 on its
  own, so `max(Lcool, 0.90·Lmech)=Lcool` trips the trigger regardless of the floor.
- `none`: times out (600 s) still energy-driven (its resolved cooling hadn't crossed 0.95 in-window).

**Corrected picture:** the floor value mostly sets *when* a dense cloud cooling-fires (earlier with higher
θ), not *whether*. The clean "lift magnitude without triggering → blowout triggers" regime exists only for
**diffuse** clouds, where resolved `Lcool/Lmech` stays ~0.25 (well below 0.95) so a `θ<0.95` floor never
reaches the trigger [**predicted, not yet run** — needs a diffuse `theta90` vs `none`]. So the cooling
boost's effect is **density-dependent through the resolved-cooling-vs-0.95 race**, not through the θ value:
dense → cooling-triggered early; diffuse → blowout-triggered late. That is the right physical
density-dependence — and exactly what `θ_target(Da)` tried (and failed, refuted) to impose with a scalar.
**Only κ_eff makes the cooling fraction (and thus the transition timing) emerge per cloud from the
developing mixing layer** — the validation is itself the argument for Rung B. [artifacts:
`runs/data/compare_f1edge_hidens_theta9{0,5}.csv`, `runs/params/f1edge_hidens__theta9{0,5}.param`]

## 6. Proposed plan (if greenlit)

1. **Rung A first as a back-reaction probe** (~hours): add `cooling_boost_kappa` (`f_κ`), apply at `:294/:373/:406`, gate byte-identical when `f_κ=1`. Run the stiff edge configs; confirm it raises conduction-zone cooling through the structure (θ as an *output*) and observe the evaporation back-reaction (expect wrong sign). This de-risks the ODE/IC plumbing without the hard re-derivation.
2. **Rung B (the workstream):** (a) derive a numerical near-front IC for `κ_eff = max(κ_Spitzer, κ_mix)` keeping `dMdt > 0`; (b) replace the ODE RHS factor + `dMdt` seed; (c) build the `κ_mix ~ ρ c_p D_turb` model; (d) full rule-5 ladder — per-call → full-run equivalence on `param/simple_cluster.param` + `f1edge_{lowdens,hidens}` + a 5e9, separate processes, matched `t`; redo the cleanroom C0 substrate certification; (e) its own `docs/dev/` FINDINGS with the three-banner set.
3. **Gate before B:** if Rung A's back-reaction probe shows the structure plumbing can't take a non-Spitzer κ without the IC re-derivation (expected), that confirms B is required, not optional.

## 7. Open questions / risks
- Closed-form vs numerical near-front IC for a `max()` κ — the `dMdt > 0` constraint (the cleanroom failure mode) is the gating risk; solve it first.
- `κ_mix ~ ρ c_p D_turb` with `D_turb ~ λ δv ~ R2·v2` is a *model choice* — validate the form, don't assume it.
- Re-validation cost is real (cleanroom C0 certification + full-run equivalence on stiff regimes). Budget for it.
- Honesty: El-Badry's 3–30× evaporation suppression is the *target*; reproducing it is the success criterion, and a scalar/Spitzer-rescale provably cannot (§4–5).
