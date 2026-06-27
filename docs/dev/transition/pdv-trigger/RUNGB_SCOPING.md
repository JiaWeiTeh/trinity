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

> **Provenance.** Written 2026-06-26; **risk #1 prototyped offline the same day → the §3a "shoot on
> `dTdr_front`" plan was REFUTED (FM1 fired) and redirected** (`make_fm1_rootcheck.py`). The IC algebra and the
> cooling/evaporation-decoupling argument below were each **independently re-derived and adversarially checked
> by a separate verification pass** against current source (the front-balance ratio was confirmed numerically
> to machine precision); the *eigenvalue-swap recommendation* was then overturned by the prototype. This is the
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
2. **Evaporation-down** — *originally* proposed as `dMdt` set by entrainment as an **input**. **⚠️ Update
   (2026-06-26): the offline FM1 prototype REFUTED imposing `dMdt`** — it is pinned by the `v(R1)=0` boundary
   condition and has no free eigenvalue to replace it (§3a). **Redirect:** keep `dMdt` as the Weaver eigenvalue,
   add the mixing-layer cooling only to the **in-structure loss integrand**, and let any evaporation change
   *emerge* and be **measured** (§3a, §8). The "two separate changes" framing survives; the mechanism for
   change #2 does not — read §3 and change #2 below through this redirect.

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

## 2a. Literature anchor — θ vs λδv vs f_mix vs the 0.95 trigger (verified 2026-06-26)

The three symbols floating around this workstream are **distinct objects**, and conflating them is the notation
cleanup the docs kept flagging. Checked against the sources (search-corroborated; arXiv PDFs 403 through the
proxy, so this is from the abstracts/indexed text, not a direct page read):

| symbol | what it is | set or measured? | TRINITY analog |
|---|---|---|---|
| **θ** (≡ `L_int/Ė` in El-Badry, `Ė_cool/L_w` in Lancaster) | the **loss fraction** | **measured — an OUTPUT** in *both* papers | TRINITY's resolved `L_cool/L_mech` |
| **λδv** (= `κ_eff`, the mixing diffusivity) | the **mixing efficiency** | **set — an INPUT, only in 1D** (El-Badry "treated as an arbitrary parameter, a range of values explored"); Lancaster's 3D resolves the fractal interface and needs **no** such knob | **`κ_mix = ρ c_p D_turb`, `D_turb = λδv`** — i.e. **Rung B's added turbulent conductivity**, NOT the scalar `f_mix` and NOT exactly Rung A's `cooling_boost_kappa` (which *scales Spitzer*, hence couples cooling+evaporation) |
| **0.95** | the energy→momentum trigger | a **threshold on the measured θ** (`(Lgain−Lloss)/Lgain<0.05 ≡ θ>0.95`), not a physics parameter | TRINITY's transition test |

**Consequences for the design.** (i) θ is never something to *impose*; you set the conductivity knob and θ
emerges — which is exactly why the constant-`θ`/`f_mix` knobs were degenerate (`FINDINGS.md`) and why the
genuine El-Badry analog is the **structural `κ_mix` (Rung B)**, not the scalar. (ii) The calibration target is
**density-dependent** — El-Badry's measured `θ(n_H, λδv)` scaling, and Lancaster's parameter-free
`θ ≈ 0.9–0.99` *in their dense-cluster regime* — **not** a flat 0.95 (which would over-cool the diffuse clouds
whose resolved θ at blowout is only ~0.25, and re-introduce the trigger degeneracy). (iii) **El-Badry states
the mechanism our FM1b test measures:** *"most of the energy conducted into the interface is immediately lost to
cooling, **reducing the evaporative mass flux** required to balance conduction"* — i.e. interface cooling and
evaporation **compete for the same conductive-flux budget**, so adding in-structure cooling should drive `dMdt`
**down** (the El-Badry sign). That is an **independent literature prediction of the FM1b sign** (§8), and a clean
falsification: if the prototype shows `dMdt` *up*, the in-structure injection is wrong.

*Caveat:* the claim that El-Badry *prescribe* calibrating `λδv` by matching a **3D** simulation is mildly
anachronistic — El-Badry (2019) predates Lancaster's 3D (2021); confirm their exact wording (likely "calibrate
against more detailed calculations/observations"). The substance — `λδv` is an externally-calibrated free
parameter — holds either way. Sources: El-Badry et al. 2019 (MNRAS 490, 1961; arXiv:1902.09547); Lancaster et
al. 2021 (ApJ 914, 89/90; arXiv:2104.07720 / 2104.07722).

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

## 3a. Risk #1 worked on paper — the closure re-count (what `v(R1)=0` solves for)

*This is the §7 risk #1 resolution — a **design proposal to validate in the offline prototype (§8)**, not yet
proven by a run. It is written down here so a future visit can see the reasoning and check it.*

**The current closure (verified from code).** `get_bubbleproperties_pure` calls
`scipy.optimize.fsolve(velocity_residuals_wrapper, bubble_dMdt, …)` — i.e. **one shooting variable,
`dMdt`** — against the residual `(v[R1] − 0)/(v[front] + ε)` returned by `_get_velocity_residuals` — i.e.
**one target, `v(R1)=0`**. `_get_bubble_ODE_initial_conditions(dMdt, …)` builds the front IC *from* `dMdt`:
the anchor `T(front)=T_init=3e4 K` is **fixed**, and the Spitzer front balance slaves the layer thickness
`dR2 ∝ 1/(C·dMdt)` and the gradient `dTdr = −2/5·T/dR2` to `dMdt`. So the count is **1 eigenvalue (`dMdt`) ↔
2 boundary conditions (`T(front)=T_init`, `v(R1)=0`)**, with the Spitzer balance as the third (algebraic)
relation that closes it. In classical Weaver terms, **`dMdt` is the eigenvalue** that makes the conduction
solution match both ends — evaporation is whatever conduction self-consistency demands.

**Why fixing `dMdt` breaks the count.** Prescribing `dMdt` by entrainment removes the only shooting
variable, leaving `v(R1)=0` with nothing to satisfy it → the two BCs are **over-determined**. A new
eigenvalue is required. Candidates (free **exactly one**; physical meaning; failure mode → §9):

| new eigenvalue | physical meaning | why / why not |
|---|---|---|
| **`dTdr_front` (front T-gradient)** ✅ recommended | local conduction strength at the cold edge | the numerical near-front IC (§2, needed for the κ_mix branch anyway) **already** starts from `(T_init, dTdr_front)`, so this slots straight in; preserves the physically-meaningful `T_init=3e4` anchor and the layer-thinness; isolates `dMdt` to the entrainment law + the recoil term in `v_front` |
| `dR2` / `r2′` (layer thickness) | the conduction layer thickens/thins | viable, but reintroduces the `dR2→0` cancellation regime `test_dR2min_magic_number.py` guards (FM6) |
| `T_init` (anchor temperature) | the cold-edge temperature floats | least physical — `3e4 K` is the recombination/shell edge; letting it drift risks leaving the radiating band |

**The hypothesis I first wrote here** was: demote `dMdt` to an entrainment input and **promote the front
gradient `dTdr_front` to the eigenvalue** `v(R1)=0` shoots on, so the conduction layer absorbs the boundary
mismatch by radiating more/less instead of by changing evaporation — the decoupling expressed in the closure.
The sign argument was: a suppressed (smaller) `dMdt` weakens the recoil term in `v_front`, so the root should be
*easier* to find.

> ### ❌ REFUTED by the offline prototype (FM1 fired) — 2026-06-26
>
> `data/make_fm1_rootcheck.py` replays this exact closure on two **real captured stiff states** (the 5e9/sfe0.01
> flood regime + a mild cluster), sweeping `dTdr_front` over **6 decades** for each suppression
> `s ∈ {1,3,10,30}` (`dMdt := dMdt_Spitzer/s`). Result (`data/fm1_rootcheck.csv`, `fm1_rootcheck.png`):
> - **`s=1` finds the root at `f*≈1.0`, physical** — the built-in correctness check passes (with the full
>   Spitzer `dMdt`, shooting `dTdr_front` recovers the Spitzer solution). So the harness is sound.
> - **`s=3, 10, 30` find NO root, in either state, anywhere in 6 decades of `dTdr_front`.**
>
> **Why** (the diagnostic): the recoil term is numerically tiny — it shifts `v_front` by only ~0.5 out of a
> streaming velocity `cool_alpha·R2/t_now ≈ 2243` — **but the stiff BVP exponentially amplifies `v_front`**:
> that ~0.5 shift moves `v(R1)` by ~2000, while sweeping `dTdr_front` across 6 decades barely moves `v(R1)` at
> all. So **`v(R1)=0` is controlled by `dMdt` (through the recoil in `v_front`), NOT by the conduction
> gradient.** Only `s=1` — where the recoil is exactly the self-consistent Spitzer value — reaches the BC.
> `dTdr_front` has no leverage; promoting it to eigenvalue cannot work. **The sign argument was wrong, in the
> opposite direction.**

**What this teaches (the real redirect).** `dMdt` is **not a free dial** — it is pinned by `v(R1)=0` and the
structure. You therefore **cannot *impose* "evaporation down"**; an El-Badry suppression has to **emerge** from a
*structure change*. So the decoupling must **not** live at the `dMdt` / inner-BC level at all. The viable path:

```
keep dMdt as the Weaver eigenvalue            # v(R1)=0 stays well-posed, dMdt > 0 by the existing root
add mixing-layer cooling to the IN-STRUCTURE  # an explicit L_mix in the dudt / L_conduction integrand,
  radiative-loss integrand (~10^5 K band)     #   localized to the cooling-curve peak, kappa UNCHANGED
re-solve for dMdt (v(R1)=0) WITH it present   # integrated into the solve (threads the cleanroom trap)
MEASURE  Delta L_cool  and  Delta dMdt        # does L_cool rise while dMdt FALLS? that is the real test
```

This keeps the well-posed Weaver closure (so no FM1 stall), puts the cooling exactly where the cleanroom §6.6
said it must go (*inside* the structure solve, not a post-hoc `Eb` sink), and makes the evaporation response an
**output to be measured**, not an input to be imposed. Whether the in-structure `L_mix` actually *lowers* `dMdt`
(El-Badry's sign) or *raises* it (Rung-A's sign) is **the next offline prototype** — see §8. The `(β,δ)` solver
and its `dMdt>0` gate are untouched by either path.

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

1. **The `fsolve` re-think — what does `v(R1)=0` determine once `dMdt` is exogenous?** **RESOLVED — §3a:**
   the "shoot on `dTdr_front`" proposal was **prototyped offline and REFUTED** (FM1 fired — `dMdt` is pinned by
   `v(R1)=0`, the gradient has no leverage). **Redirect:** keep `dMdt` as the Weaver eigenvalue and decouple at
   the in-structure loss integrand instead; the *new* make-or-break is **FM1b** (the ΔdMdt sign), §8.
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

## 8. Where we are, and the next concrete step (no production edit)

Two offline prototypes are now done, and **both rule out a path to El-Badry suppression — for the same reason**:
- **FM1 (`make_fm1_rootcheck.py`): "shoot on `dTdr_front`" REFUTED** — `dMdt` is pinned by `v(R1)=0`, not a free
  dial; it cannot be *imposed* (§3a, §9).
- **FM1b (`make_fm1b_evapsign.py`): "add interior loss-integrand cooling" is a half-pass** — the El-Badry **sign
  is confirmed** (interface cooling lowers `dMdt`) but the **magnitude is negligible** (−0.1% at 5× cooling vs a
  −67…−97% target), because the 10⁵ K cooling is in the interior while `dMdt` lives at the front (§9 box below).

Both fail because **`dMdt` is a *front* quantity**. So the remaining lever is the **front conduction itself**.

**The second offline prototype is DONE (`make_fm1b_evapsign.py`) — the result is a half-pass:**

> ### ◐ FM1b — El-Badry SIGN confirmed, but magnitude negligible (2026-06-26)
>
> Injecting a localized extra cooling into the structure by monkeypatching `net_coolingcurve.get_dudt`
> (`dudt → dudt·(1+A·gaussian(log T; 5, 0.4))`, so more cooling at ~10⁵ K) and running the **full** production
> solve `get_bubbleproperties_pure` (the `v(R1)=0` fsolve for `dMdt`) on the two captured stiff states
> (`data/fm1b_evapsign.csv`, `fm1b_evapsign.png`; `A=0` recovers the converged `dMdt` — correctness check):
> - **Sign: ✅ El-Badry.** `dMdt` decreases monotonically with injected interface cooling, in **both** states,
>   above the fsolve noise floor — interface cooling *does* suppress evaporation, the direction §2a predicted.
>   FM1b is **not falsified.**
> - **Magnitude: ✗ negligible.** Even a **5× cooling boost** at 10⁵ K (`A=4`) moves `dMdt` by only
>   **−0.10% (stiff) / −0.03% (mild)** — against the El-Badry *target* of **−67% to −97%** (3–30× suppression).
>   Three orders of magnitude short.
> - **Why** (the FM1 lesson, again): `dMdt` is anchored at the **3×10⁴ K front** (the recoil/enthalpy
>   balance there), so cooling added in the **10⁵ K interior** — away from the front — barely couples to it.

**What this narrows.** Two paths are now ruled out as routes to El-Badry suppression in the 1D Weaver structure:
imposing `dMdt` (FM1, refuted) **and** adding interior loss-integrand cooling (FM1b, right sign but ~0 leverage).
Both fail for the **same reason**: `dMdt` is a **front quantity**. So the remaining lever is the **front
conduction itself** (the `κ`/boundary balance at ~3×10⁴ K) — which is where Rung A's `κ`-scaling acted, but
coupled cooling *and* evaporation. The honest next question is therefore **how a faithful `κ_eff` modifies the
front so cooling rises while evaporation falls** — i.e. back to the El-Badry `λδv` treatment, but now we know it
must act **at the front**, not as an interior loss term. **Next concrete step:** a third offline prototype that
perturbs the front conduction (e.g. an El-Badry-style `κ_mix` added to `κ_Spitzer` *only near the front*, or a
modified front enthalpy split) and measures ΔL_cool vs Δ`dMdt` — still capture/replay, still no production edit.
Risk #2 (`v_entrain`/`α_mix` calibration) and any gated production change come only after a path clears that.

## 9. Failure-mode ledger — what could go wrong (and how we'd catch it)

*Written down so a future visit can look back and see where this could have broken. Each row: the failure,
how it would show up, and the mitigation/guard. **FM1 was the gating one — and it FIRED** (see below).*

| # | failure mode | how it surfaces | mitigation / guard |
|---|---|---|---|
| **FM1 — 🔴 FIRED** | **The "shoot on `dTdr_front`" closure (§3a) admits NO `v(R1)=0` root once `dMdt` is fixed below Spitzer.** Proven offline (`make_fm1_rootcheck.py`): `s=1` finds the root (correctness check), `s=3/10/30` find none across 6 decades of `dTdr_front`, on both captured states. | the would-be `fsolve` never crosses zero — exactly what the prototype shows. | **the sign argument was WRONG** (the recoil is tiny but exponentially amplified; `v(R1)=0` is set by `dMdt`, not `dTdr_front`). **Resolution: abandon the eigenvalue-swap; keep `dMdt` as the Weaver eigenvalue and decouple at the in-structure loss integrand instead (§3a redirect, §8).** This is why we prototyped offline before touching code. |
| **FM2 — moot** | Over/under-determination of the eigenvalue swap. | — | **superseded:** the redirect (§3a, §8) abandons the eigenvalue swap entirely and keeps the original Spitzer `dMdt` closure, so there is nothing to over/under-determine. |
| **FM1b — ◐ MEASURED (sign ✓, magnitude ✗)** | Whether in-structure `L_mix` lowers `dMdt` (El-Badry) or raises it (Rung-A). | measured offline (`make_fm1b_evapsign.py`, `fm1b_evapsign.csv/png`): **sign is El-Badry** (`dMdt` ↓, monotonic, above the fsolve noise floor, both states) — **not** the Rung-A re-coupling, so the loss-integrand path is *directionally* right. | **but the magnitude is negligible** — `−0.1%` at 5× cooling vs a `−67…−97%` target — because the 10⁵ K cooling is interior and `dMdt` is front-anchored. **Resolution: a loss-integrand `L_mix` is not enough; move the intervention to the front conduction (§8 next step).** |
| **FM3** | **`(β,δ)` non-convergence with the entrainment-set `bubble_LTotal`** — `dMdt>0` gate now passes by construction, but the self-similar solve must still converge with the enhanced cooling. | `get_betadelta` hybr fails / oscillates; `no_physical_root` for a different reason than `dMdt`. | re-run the **cleanroom C0 certification** on the new structure; evaluate the entrainment law **inside** the structure solve so `Pb,β,δ,dMdt` are mutually consistent (the §4 necessary condition). |
| **FM4** | **Entrainment-law coupling oscillation** — `dMdt_entrain ∝ ρ_shell·v_entrain` depends on shell state; a lagged/inconsistent shell snapshot makes the structure↔shell coupling wobble. | non-monotone segment-to-segment convergence; trajectory ripples absent in baseline. | evaluate the entrainment law from the **same** state snapshot the structure solve uses; freeze it within a segment. |
| **FM5** | **`α_mix` mis-set** — too large ⇒ `κ_mix` swamps Spitzer everywhere (the `T_cross~10¹²` K absurdity, §2), turning the whole structure turbulent-diffusion-dominated; too small ⇒ no cooling boost (back to baseline). | loss ratio either jumps to ~1 instantly (too large) or never moves (too small). | calibrate `α_mix` to El-Badry/Lancaster; expose it as a **gated knob, default-off byte-identical**; bracket the physical range with a sweep before believing any single value. |
| **FM6 — relaxed** | Regression of the `dR2` magic-number guarantees (`test_dR2min_magic_number.py` pins the Spitzer `dR2 ∝ 1/dMdt`). | — | the loss-integrand redirect **keeps** the Spitzer `dMdt` closure (and its `dR2` law), so this test stays valid as-is; the added `L_mix` is in the loss integrand, not the front IC. Still gated default-off ⇒ byte-identical. |

**The one-line risk story (updated):** two make-or-breaks are now settled offline, and both point to the same
place. **FM1** fired (`dMdt` is pinned by `v(R1)=0` — can't be imposed); **FM1b** half-passed (interface cooling
lowers `dMdt` — El-Badry sign ✓ — but only ~0.1%, because `dMdt` is front-anchored). The convergent lesson:
**`dMdt` is a front quantity**, so neither imposing it nor adding interior cooling reaches El-Badry suppression —
the next prototype must perturb the **front conduction** itself. The discipline keeps paying off: two wrong/weak
design hypotheses cost two offline harnesses, not a production regression.

---

**Artifacts.** FM1: `data/make_fm1_rootcheck.py` → `data/fm1_rootcheck.csv` + `fm1_rootcheck.png`. FM1b:
`data/make_fm1b_evapsign.py` → `data/fm1b_evapsign.csv` + `fm1b_evapsign.png` (monkeypatches
`net_coolingcurve.get_dudt` to add localized ~10⁵ K cooling, runs the full `get_bubbleproperties_pure`).
Both reproduce from repo root (`python docs/dev/transition/pdv-trigger/data/make_fm1{,b}_*.py`), read the
captured states `test/data/{dR2_stiff_state,residual_resample}_fixture.json`; no production edit, no full run.

**Cross-refs.** `KAPPA_EFF_SCOPING.md` (§3 two-rung ladder, §6.2–6.3 plan/gate, §6a Rung A result),
`PLAN.md` (status ledger; "Outcome & pivot"), `NOTE_PATCHES.md` Patch 7, the cleanroom `PLAN.md` §6.6
(`dMdt<0` failure record), and `data/kappa_backreaction.csv` (the Rung A measurement this design responds to).
