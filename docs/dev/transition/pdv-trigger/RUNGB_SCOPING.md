# Rung B — faithful `κ_eff` / mixing-layer interface: design & IC re-derivation scoping

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
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `KAPPA_EFF_SCOPING.md`,
> `NOTE_PATCHES.md`, `runs/README.md`, and any other notes in the same folder). They drift out of sync
> *with each other* as fast as they drift from the code. Any agent or person editing one MUST, as part of
> the visit, circle back through the siblings and reconcile: if a number, status, claim, or line reference
> here contradicts a sibling — or a sibling has gone stale — fix it (or flag it, dated) so no two docs in
> the workstream disagree. Never update one in isolation.

> **Provenance.** Written 2026-06-26. The IC algebra and the cooling/evaporation-decoupling argument below
> were each **independently re-derived and adversarially checked by a separate verification pass** against
> current source (the front-balance ratio was confirmed numerically to machine precision). This is the
> design doc the `KAPPA_EFF_SCOPING.md` §6.2 ("Rung B, the workstream") and §6.3 gate point at, now that
> Rung A is built and its back-reaction measured (`KAPPA_EFF_SCOPING.md` §6a). All file:line refs below were
> spot-verified 2026-06-26 but **cite function names first** — line numbers drift, the structure does not.

## 0. Verdict (TL;DR)

**The decoupling is the heart of Rung B, not the `κ` swap.** Rung A proved (and the re-derivation below
explains) that in the Weaver+77 structure solve the conductive heat flux **`q = κ·dT/dr` at the evaporation
front is a single quantity read twice**: it fixes the evaporative mass flux `dMdt` (the front enthalpy
balance), *and* it sets the temperature profile whose `dudt`-integral is the radiative loss `L_cool`. So a
scalar `κ` inflation raises **both together** (Rung A: at matched `t`, `L_cool ×1.23–1.38`, `dMdt
×1.08–1.17` — `dMdt`'s fractional rise ≈ half `L_cool`'s). El-Badry/Lancaster need the **opposite** coupling:
cooling **up**, evaporation **down** (3–30×). Therefore Rung B must **sever `dMdt` from the front conductive
balance** — not just substitute a different `κ(T)` into the same IC/ODE.

Concretely, Rung B is **two structurally separate changes**, both inside `bubble_luminosity.py`:

1. **Cooling-up** — a mixing-layer/`κ_eff` enhancement **localized to the ~10⁵ K radiating band** (the
   cooling-curve peak), living *inside* the structure solve so it raises `bubble_LTotal` consistently.
2. **Evaporation-down** — `dMdt` set by a **turbulent entrainment** prescription (`≈ ρ_shell·v_entrain·4πR2²`,
   positive by construction), an **input** to the structure, **not** the root of a front balance.

Three results from the verification that shape the plan:

- **IC re-derivation:** for `κ ∝ T^p` the front profile is `T ∝ (R2−r)^{1/p}`, front-regular **only for
  `p>0`**. Spitzer is `p=5/2 → q=2/5` (the existing closed form). A turbulent `κ_mix ∝ 1/T` is `p=−1 → q=−1`,
  which is **not front-regular** (T diverges at the wall) — the Spitzer "T→0 at the front, integration
  constant →0" construction **does not carry over**. ⇒ **the mix-branch near-front IC must be numerical**, not
  closed-form.
- **`κ_mix` magnitude is the real model.** Taking `D_turb = R2·v2` *literally* gives `T_cross ≈ 10¹² K`
  and `κ_mix/κ_S ≈ 10²⁴` — κ_mix would swamp Spitzer everywhere, which is unphysical. The mixing diffusivity
  **must** carry an entrainment-efficiency `α_mix ≪ 1`; **that factor is the model** (calibrate to
  El-Badry+2019 / Lancaster+2021). The IC re-derivation is *secondary* to getting this magnitude right.
- **`dMdt>0` safety (threads the cleanroom trap):** an entrainment-set `dMdt` is positive by construction and
  is an *input*, so there is no "find the `dMdt` that balances a depressed-`Pb` front" step that can lose its
  positive root — which is exactly the cleanroom §6.6 stall (post-hoc sink → low `Pb` → no `dMdt>0` root → hybr
  spins). Provided the entrainment law is evaluated **inside** the structure solve (so `Pb,β,δ,dMdt` stay
  mutually consistent), not patched on afterward.

The self-similar **`(β,δ)` solver stays untouched** (`get_betadelta.py` carries no explicit conduction-law
dependence) — but its **`dMdt>0` acceptance gate** and its **consumption of `bubble_LTotal`** are the two
coupling surfaces Rung B must respect.

## 1. The coupling, made precise — why Rung A raised both

The near-front IC (`_get_bubble_ODE_initial_conditions`, Weaver+77 Eq 44) encodes **exactly** the front
enthalpy balance:

```
κ · dT/dr |_front  =  −(5/2)(k_B/μ) · T · dMdt/(4πR2²)
```

(verified: rebuilding this from the code's `constant = 25/4·k_B/(μ·C)`, `dR2 = T_init^{5/2}/(constant·jm)`,
`dTdr = −2/5·T/dR2` reproduces the identity to machine precision). The Spitzer `C` *cancels* from
`T(front)=T_init=3e4 K` by construction, but it sets the front thickness `dR2 ∝ 1/(C·dMdt)` and the gradient.
The same `q = κ·dT/dr` is then **read twice**:

- **It fixes `dMdt`.** The front balance is the outer closure; the inner boundary condition `v(R1)=0`
  (`_get_velocity_residuals`, root-found by the `fsolve` in `get_bubbleproperties_pure`) is the inner closure.
  The front `v` carries the evaporative-recoil term `−dMdt/(4πR2²)·k_B T/(μ Pb)`, so `dMdt` directly shapes the
  velocity field the BC acts on. Together the two closures determine `dMdt`.
- **It sets the radiative loss.** `dR2`/`dTdr` seed the inward integration of `_get_bubble_ODE`, whose `T(r)`
  profile is what `net_coolingcurve.get_dudt` is integrated over to form `bubble_LTotal` (the conduction-zone
  trapezoid).

So inflating `κ → f_κ·C` (Rung A) stiffens conduction through one knob: it raises the Eq-33 seed
(`C^{2/7}`), steepens the front (more flux → the balance seeds a larger `dMdt`), and raises the RHS conduction
term (broader radiating zone → more `L_cool`). Because `dMdt` and the radiating profile are both children of
the **same** `q`, both rise together — the measured Rung A back-reaction, and the reason a scalar knob can
never deliver the El-Badry sign.

## 2. The IC re-derivation (the specific ask)

**General law.** For a conductivity `κ(T) = K·T^p`, the front balance `K·T^{p−1} dT = A·dr` integrates to a
local power law `T ∝ (R2−r)^{1/p}`, i.e. **`q = 1/p`**. The Spitzer prefactor `25/4 = (5/2)·(5/2)` generalizes
to `(5/2)·p`. Check: `p=5/2 → q=2/5`, matching the code. ✔

**Front-regularity caveat (the load-bearing point).** Dropping the integration constant ("T→0 at the front")
is valid **only when `q>0` ⇒ `p>0`**: then `T→0` at the wall and the flux `κ·dT/dr` stays finite. For a
turbulent `κ_mix = ρ·c_p·D_turb` with `ρ = Pb·μ/(k_B T)` (ideal gas at bubble pressure) and `D_turb ≈ R2·v2`
(≈ T-independent), `κ_mix ∝ 1/T`, i.e. **`p = −1 → q = −1`**: `T` *diverges* approaching the front, `dT/dr` is
negative-divergent — physically a cold finite-T wall, not a Spitzer conduction front. **The closed-form
Spitzer IC construction does not carry over to the mix branch.**

**Crossover.** `κ_eff = max(κ_S, κ_mix)`; `κ_S = C·T^{5/2}` rises with T, `κ_mix ∝ 1/T` falls with T, so they
cross at `T_cross = (κ0/C)^{2/7}` with `κ0 ≡ κ_mix·T = (Pb·μ/k_B)·c_p·D_turb`. With representative stiff-bubble
numbers (`Pb≈2e11` cgs, `R2≈0.001 pc`, `v2≈3700 km/s ⇒ D_turb≈1.1e24 cm²/s`), `κ0≈6e35`, giving **`T_cross ≈
10¹² K`** and `κ_mix/κ_S ≈ 10²⁴` across the whole radiating zone. That is unphysical — it says literal
`D_turb=R2·v2` makes turbulent transport swamp conduction everywhere. `T_cross ∝ α_mix^{2/7}`, and it would
take `α_mix ~ 10⁻³⁰` to pull `T_cross` down to ~10⁶ K. **Reading:** `D_turb = R2·v2` is a ceiling, not the
value; the physical mixing diffusivity is `α_mix·R2·v2` with `α_mix ≪ 1` set by the entrainment efficiency —
**this magnitude, not the IC algebra, is the central modeling task.**

**Recommendation (IC).** Do **not** try to reuse a closed-form near-front IC for the mix branch. Use a
**numerical near-front integration** from a regularized finite-`T` anchor a small `dR2` inside `R2`. Keep the
Spitzer closed form only where the front anchor genuinely stays in the `p>0` Spitzer branch. **Any** closed
form fails the moment `T_cross` lands *inside* the front/IC region (so `κ_eff` is non-power-law across the
anchor — the exponent switches between `2/5` and `−1`): there, integrate numerically.

## 3. The `dMdt` closure — the decoupling

Sever `dMdt` from the front conductive balance:

- **`_get_init_dMdt`** (Eq-33 / `C^{2/7}` seed) → an **entrainment estimate** `dMdt ≈ ρ_shell·v_entrain·4πR2²`,
  with `v_entrain` from the mixing-layer scaling (El-Badry/Lancaster). Positive by construction.
- **`_get_bubble_ODE_initial_conditions`** (the **critical edit**) → with `dMdt` now exogenous, the front IC
  builds `T, dTdr, v` at `r2′` from the *prescribed* `dMdt` **without** equating `κ·dT/dr` to the enthalpy
  flux. The recoil term in `v` stays (still physical); `dMdt` is no longer the unknown the front fixes.
- **`_get_velocity_residuals` / the `fsolve`** → **the subtlest part, and the gating numerical risk.** Today
  `v(R1)=0` is the equation that *solves for* `dMdt`. Once `dMdt` is entrainment-set, that equation is free: it
  must either be **dropped** (one fewer closure) or **repurposed** to solve for a different free parameter
  (e.g. the front anchor `T_init`/`r2′`, or a `Pb`-consistency variable). Resolving *what `v(R1)=0` now
  determines* is the first thing to settle on paper (§7 risk #1).

## 4. `dMdt>0` safety — how this threads the cleanroom trap

The cleanroom §6.6 stall: a post-hoc `L_mix = θ·Lmech` subtracted from `dEb/dt` **after** the `(β,δ)` solve
depressed `Pb`; the evaporative-recoil `v`-term `−dMdt·k_B T/(μ Pb)` blows up as `Pb` falls, so the front
balance + `v(R1)=0` system has **no positive `dMdt` root** → the `dMdt>0` gate rejects every segment → the
`dt`-shrink guard spins (~zero progress). The energy was removed where the structure equations never saw it.

An **entrainment-set `dMdt`** avoids this because `dMdt` is a positive **input**, not the **root** of a
front balance that can go negative. There is no "find the `dMdt` that balances a depressed-`Pb` front" step.
The radiative loss is then computed *on the structure built from that `dMdt`*, **inside** the same solve —
so `β,δ` are found *with* the enhanced cooling (the §6.6 conclusion: "integrate the cooling INTO the structure
solve"). **Necessary condition:** the entrainment law is evaluated inside the structure solve so
`Pb,β,δ,dMdt` are mutually consistent — not patched on afterward (that would reintroduce the trap).

## 5. The edit map — what changes, what must not

| locus (`bubble_luminosity.py` unless noted) | Rung B change | gate / note |
|---|---|---|
| `_get_init_dMdt` | Eq-33 `C^{2/7}` seed → entrainment `ρ_shell·v_entrain·4πR2²` | positive by construction |
| `_get_bubble_ODE_initial_conditions` | front enthalpy balance → **numerical** IC with `dMdt` as input (**critical**) | mix branch `p=−1` not front-regular (§2) |
| `_get_bubble_ODE` (RHS conduction term) **or** the `dudt`/`L_conduction` integrand | add `κ_eff(T)`/mixing cooling **localized to ~10⁵ K** | the **cooling-up** change; must be in-solve, not post-hoc |
| `_get_velocity_residuals` / `fsolve` target | re-think what `v(R1)=0` solves for once `dMdt` is exogenous | **the gating open risk** (§7 #1) |
| `get_betadelta.py` | **no edits** | respect the `dMdt>0` gate + `bubble_LTotal` consumption (coupling surfaces) |
| `effective_Lloss*` (the `cooling_boost` modes) | **leave alone** | the post-hoc `Lcool` knob §6.6 rejected for this purpose |

## 6. The rule-5 ladder for Rung B (gates before code)

1. **Gate-first.** Define "equivalent": new mixing knobs **default-off → byte-identical** (the `α_mix=0`
   / `cooling_boost_kappa=1` path must reproduce `dictionary.jsonl` bit-for-bit, as Rung A did).
2. **Baseline.** Reuse the captured stiff states + the `git show HEAD` value/hash discipline.
3. **Equivalence gate.** Per-call IC equivalence first (cheap) → **full-run** on `param/simple_cluster.param`
   + `f1edge_{lowdens,hidens}` + a 5e9, **separate processes**, **matched `t`**. A "free win" off ⇒
   bit-identical.
4. **Apply** the smallest diff that passes (the two changes of §5, smallest viable form first).
5. **Re-verify.** Gate again + full `pytest` + ruff F-rules; **redo the cleanroom C0 substrate
   certification** (Rung B changes the structure the cleanroom certified).
6. **Persist.** A `data/` CSV/figure proving cooling↑ **with** evaporation↓ and `dMdt>0` throughout, plus its
   own `FINDINGS`-style writeup carrying the four banners.
   **Success criterion:** reproduce El-Badry's coupling — `L_cool` up *while* `dMdt` is **suppressed** (not
   raised), `dMdt>0` for every segment, and the loss ratio reaching the transition for the clouds it should.

## 7. Open questions / risks (ranked — solve top-down, on paper, before any production edit)

1. **The `fsolve` re-think — what does `v(R1)=0` determine once `dMdt` is exogenous?** The gating numerical
   risk. Removing the front conductive balance removes one equation; the inner BC must be dropped or
   repurposed. Settle this analytically *first*.
2. **`v_entrain` prescription + `α_mix` calibration.** This *is* the model (§2). Validate the form against
   El-Badry+2019 / Lancaster+2021 — do **not** assume `D_turb = R2·v2`; that is a ceiling giving absurd
   `T_cross`.
3. **Numerical near-front IC regularization** for the mix-branch (`p=−1`) front — the offset/anchor and
   stiffness handling (cf. `test_dR2min_magic_number.py`, which pins the *Spitzer* `dR2 ∝ 1/dMdt` law; Rung B
   changes what that test certifies and will need its own coverage).
4. **Localizing `κ_eff`/mixing cooling to the ~10⁵ K band** without re-coupling to the now-exogenous `dMdt` —
   the crossover-shaping is fragile (a κ peaked at 10⁵ K still touches the front gradient through the inward
   integration); prefer adding the cooling to the loss integrand over re-shaping `κ` if the two are equivalent
   in effect.
5. **Re-validation cost** — cleanroom C0 re-certification + full-run equivalence on the stiff regimes is real;
   budget for it.

## 8. Recommended first concrete step (no production edit)

A paper/notebook derivation that **resolves risk #1** (the `fsolve` target) and **#2** (`v_entrain` + `α_mix`
form), plus a **numerical near-front IC prototype tested OFFLINE** on a captured stiff state — the same
capture/replay-with-explicit-gates discipline that de-risked Rung A. Persist it as a `data/` harness +
figure. **Only then** touch `bubble_luminosity.py`. This keeps Rung B in the dev/exploration realm and
production byte-identical until the gated knobs are proven off.

---

**Cross-refs.** `KAPPA_EFF_SCOPING.md` (§3 two-rung ladder, §6.2–6.3 plan/gate, §6a Rung A result),
`PLAN.md` (status ledger; "Outcome & pivot"), `NOTE_PATCHES.md` Patch 7, the cleanroom `PLAN.md` §6.6
(`dMdt<0` failure record), and `data/kappa_backreaction.csv` (the Rung A measurement this design responds to).
