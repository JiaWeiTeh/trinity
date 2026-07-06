# κ_eff / mixing-layer interface — feasibility & scoping (the cooling MECHANISM + its optional fidelity bonus)

> **Status (2026-07-02):** the `cooling_boost_kappa` knob shipped here later showed **non-monotonic breakdown
> windows** in f_κ (`FINDINGS.md §8e`/§9a — not shippable as production); the calibration was ultimately done
> on the **`multiplier`** knob (`FINDINGS.md §10`).

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
> **2026-06-26: Rung A executed** — `cooling_boost_kappa` landed gated/byte-identical-off, the crux measured
> (§6a). This is the scoping doc for the "endgame" `PLAN.md` ("Outcome & pivot") points at after the
> cooling-boost program concluded that *a scalar magnitude knob cannot fix the transition* (see `FINDINGS.md`:
> constant-θ degenerate with the 0.95 trigger; `θ_target(Da)` refuted by the gate-validated replay).

## 0. Verdict (TL;DR)

> **⭐ Reframed by "the merge" (2026-06-26) — read this first.** The goal is **enhanced, density-dependent
> cooling matched to obs/3D**, *not* evaporation suppression. **Rung A IS that mechanism** (κ_eff =
> `cooling_boost_kappa` raises the emergent cooling, ×1.23–1.38) — not a mere "probe." The remaining work is
> **calibrating `f_κ(properties)`** so the emergent θ tracks the target `θ(n_H)` (El-Badry `λδv`=κ_eff +
> Lancaster). **Rung B (the faithful evaporation-decoupling re-derivation) is DEMOTED to an optional
> high-fidelity bonus** — the 1D `dMdt` is front-anchored and resists it (`FM1`/`FM1b` in `RUNGB_SCOPING.md`),
> and that suppression is not in the goal. Below, read "the goal/endgame = decoupling" through this demotion.

**Is Option C possible? Yes — bounded, no hard showstoppers — and *more tractable than the methods note implied.*** It is a 2-rung ladder, not one thing:

- **Rung A — Spitzer-prefactor inflation (`C_thermal → f_κ·C_thermal`, keeping `T^(5/2)`): DONE 2026-06-26 — and it IS the cooling mechanism.** 1 param (`cooling_boost_kappa`, default 1.0) + 3 one-line edits; byte-identical when off, 595 tests pass. Preserves the `T^(5/2)` form so the ICs and `(β,δ)` solver just re-converge. Measured (§6a) it **raises the emergent cooling `Lcool` ×1.23–1.38** — the deliverable for the goal. It *also* nudges evaporation up (`dMdt` ×1.08–1.17), a **tolerated side effect** (stays positive/viable; the El-Badry evaporation-*down* coupling is a fidelity bonus, not the goal). Remaining work: **calibrate `f_κ(properties)` to `θ(n_H)`.**
- **Rung B — faithful mixing-layer `κ_eff` (OPTIONAL high-fidelity bonus): possible, but a genuine re-derivation of ~3 functions in `bubble_luminosity.py` + a new `κ_mix` model + full re-validation.** Would make evaporation fall *with* cooling rising (El-Badry fidelity). **Not required for the goal**, and `FM1`/`FM1b` show the 1D front-anchored `dMdt` resists it. Multi-day workstream with its own writeup (`RUNGB_SCOPING.md`); a naive post-hoc version already **failed** (see §4).

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
| change | `C_thermal → f_κ·C_thermal` (1 param + edits at `:291/:370/:406`) — **DONE** | `T^(5/2)` factor → `κ_eff(T)`; re-derive ICs (`:364–383`, numerical) + `dMdt` seed (`_get_init_dMdt`, `:291–294`); new `κ_mix ~ ρ c_p D_turb`, `D_turb ~ R2·v2` |
| `(β,δ)` solver | unchanged (re-converges) | unchanged (re-converges) — see §2 |
| ICs | preserved (`T^(5/2)` intact) | **re-derived** (numerical near-front) |
| physics | raises cooling (**the mechanism**); evaporation side-effect tolerated | decouples cooling-up / evaporation-down (fidelity bonus, *not* the goal) |
| validation | test string-pins (`test_dR2min_magic_number.py` `_scalar_params` patched, `test_metadata.py`, `test_mu_audit_drift.py`) — 595 pass | rule-5 ladder + redo cleanroom C0 certification |
| effort | ~hours (done 2026-06-26) | multi-day workstream |
| status | **the cooling mechanism — DONE, gated; next = f_κ calibration** | **optional bonus — scoped + prototyped offline (`RUNGB_SCOPING.md`): FM1 refuted, FM1b half-pass** |

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

1. **Rung A first as a back-reaction probe** (~hours) — **DONE 2026-06-26.** Added `cooling_boost_kappa` (`f_κ`, default 1.0), applied at `:291/:370/:406`; gated **byte-identical when `f_κ=1`** (sha `acbad31b` over 79 rows of `f1edge_hidens`, diverges when `f_κ=2`), full `pytest` 595 green, ruff F-rules clean. See **§6a** for the measured back-reaction. This de-risked the ODE/IC plumbing without the hard re-derivation.
2. **Rung B (the workstream)** — **scoped in full: `RUNGB_SCOPING.md`**, and **risk #1 already prototyped
   offline.** The first design (sever `dMdt` from the front balance — make it an entrainment-set *input* — and
   shoot `v(R1)=0` on the front gradient `dTdr_front`) was **REFUTED by `make_fm1_rootcheck.py`** (FM1 fired):
   on real captured stiff states, `dMdt` is pinned by `v(R1)=0` and has no replacement eigenvalue — `dTdr_front`
   has no leverage. **Redirect:** keep `dMdt` as the Weaver eigenvalue, add mixing-layer `L_mix` only to the
   **in-structure loss integrand** (~10⁵ K band, κ unchanged), and **measure ΔL_cool vs ΔdMdt** (the new
   make-or-break). **→ RUN 2026-07-06** as the f_A interface source boost
   (`data/make_fA_source_boost.py`, `SOURCE_TERM_DESIGN.md`, `FINDINGS.md §15`): ΔdMdt has the
   El-Badry sign (falls as cooling rises) — the decoupling this section called the crux. Still standing: the mix-branch near-front IC is **numerical** (`p=−1` not front-regular), and
   `κ_mix`'s magnitude needs an efficiency `α_mix≪1` (literal `R2·v2` gives `T_cross~10¹²` K). Remaining steps:
   (a) the second offline prototype (in-structure `L_mix` → ΔdMdt sign); (b) the `v_entrain`/`α_mix` model
   (calibrate to El-Badry/Lancaster); (c) full rule-5 ladder — per-call → full-run equivalence on
   `param/simple_cluster.param` + `f1edge_{lowdens,hidens}` + a 5e9, separate processes, matched `t`; redo the
   cleanroom C0 substrate certification; (d) its own `docs/dev/` FINDINGS with the four-banner set.
3. **On the optional bonus:** Rung A confirmed the structure takes a `κ` knob cleanly and raises emergent cooling — so the **goal** (calibrate `f_κ` to `θ(n_H)`) needs **no** IC re-derivation. The faithful re-derivation is only for the *optional* evaporation-suppression bonus (Rung B), which `FM1`/`FM1b` show the 1D `dMdt` resists.

## 6a. Rung A result — the crux, measured (2026-06-26)

Two separate-process runs on the stiff dense edge `f1edge_hidens` (`mCloud 1e7`, `sfe 0.01`,
`nCore 1e6`), `f_κ=2` vs the `f_κ=1` baseline, compared at **matched simulation time** (the
`f_κ=2` trajectory interpolated onto the `f_κ=1` time grid). Artifacts:
`data/make_kappa_backreaction.py`, `data/kappa_backreaction.csv` (79 matched rows),
`kappa_backreaction.png`. The whole-workstream comparison (all ideas + this result) is
`data/make_ideas_comparison.py` → `ideas_comparison.png`, also embedded in the storyline report
(`pdvtrigger_report.html` §11).

| quantity | `f_κ=2 / f_κ=1` (early→late) | reading |
|---|---|---|
| `Lcool` (`bubble_LTotal`) | **1.38 → 1.23** | conduction-zone cooling rises *through the structure* — θ as an output ✓ |
| `dMdt` (`bubble_dMdt`) | **1.17 → 1.08** | evaporation rises *with* cooling — a **tolerated side effect** (stays >0/viable; only the *optional* Rung-B bonus would suppress it) |
| `Lmech` (`Lmech_total`) | 1.000 | sanity — mechanical input untouched |
| `Eb` | 0.96 → 0.90 | bubble energy drained by the extra cooling |
| `Pb`, `v2` | 0.97→0.91, 1.00→0.96 | mild softening; `R2` essentially unmoved (≤0.4%) |
| loss-ratio proxy `Lcool/Lmech` | **+0.05 → +0.10** (0.41→0.51 vs 0.41 at end) | a **2× κ buys only ~0.05–0.10** toward the 0.95 trigger |

**Conclusions.** (i) The plumbing takes `f_κ` cleanly — cooling genuinely rises through the
structure, vindicating the structural-knob approach over a scalar `Lcool` rescale. (ii) **The crux
is real and quantified:** a flat `f_κ` raises `dMdt` too (≈half the fractional rise of `Lcool`),
exactly the coupling a faithful `κ_eff` must *suppress* (El-Badry: evaporation ÷3–30 while cooling
rises). (iii) `f_κ=2` moves the loss ratio by ~0.05–0.10, so reaching the obs/3D **target** needs a larger,
**calibrated** `f_κ` (the remaining work) — *not* a brute-force toward the 0.95 *trigger* (reaching the
trigger is not the goal; the goal is the cooling magnitude, which Rung A already delivers).
**On the optional bonus:** a *flat* `f_κ` cannot make evaporation *fall* while cooling rises — only a
state-coupled `κ_eff` could (Rung B). But that evaporation suppression is the **fidelity bonus, not the
goal**, and the 1D front-anchored `dMdt` resists it (`FM1`/`FM1b` in `RUNGB_SCOPING.md`).

## 7. Open questions / risks
- Closed-form vs numerical near-front IC for a `max()` κ — the `dMdt > 0` constraint (the cleanroom failure mode) is the gating risk; solve it first.
- `κ_mix ~ ρ c_p D_turb` with `D_turb ~ λ δv ~ R2·v2` is a *model choice* — validate the form, don't assume it.
- Re-validation cost is real (cleanroom C0 certification + full-run equivalence on stiff regimes). Budget for it.
- Honesty: El-Badry's 3–30× evaporation suppression is the success criterion **for the optional Rung-B fidelity bonus**, not for the goal (which is enhanced density-dependent cooling, delivered by Rung-A κ_eff + `f_κ(properties)` calibration). A scalar/Spitzer-rescale provably cannot reproduce the *evaporation* suppression (§4–5) — but it *does* deliver the *cooling* enhancement the goal needs.
