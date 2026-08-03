# Cross-cutting sweep ② — signs, factors, exponents

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

## Scope and sources

Read-only pass over `/home/user/trinity/trinity/**` (the package source), comment-blind: for every
candidate the target expression was re-derived from the physics or the algebra *first*, then compared
against the code and only then against its docstring/comment.

Files read in full: `bubble_structure/get_bubbleParams.py`, `bubble_structure/bubble_luminosity.py`,
`shell_structure/get_shellODE.py`, `shell_structure/shell_structure.py`,
`phase1_energy/energy_phase_ODEs.py`, `phase1b_energy_implicit/get_betadelta.py`,
`phase0_init/get_InitPhaseParam.py`, `cloud_properties/{density_profile,mass_profile,powerLawSphere,
bonnorEbertSphere,initial_profile}.py`, `cooling/net_coolingcurve.py`, `cooling/CIE/read_coolingcurve.py`,
`cooling/non_CIE/read_cloudy.py`, `_functions/{unit_conversions,operations,cluster}.py`,
`sps/update_feedback.py`. Read in part (arithmetic regions): the four phase runners,
`phase_general/phase_events.py`, `phase0_init/get_InitCloudProp.py`, `sps/read_sps.py`,
`_input/{read_param,registry}.py`, `_output/cloudy/dlaw.py`, `_analysis/check_yesno.py`,
`_input/dictionary.py`.

Worklist seed: `docs/dev/code-audit/data/claims_literals.csv` (1282 of 1644 literals flagged
`in_arithmetic`), grouped by file. `docs/dev/code-audit/slices/` deliberately **not** read.

Numerical checks were run against `param/simple_cluster.param` (the tracked energy-driven baseline),
loading only `read_param` → `get_InitCloudProp` → `read_sps` → `get_InitPhaseParam` (no full run).
Every number quoted below is reproducible from the snippets given.

---

## Target 1 — Signs

### 1.1 The four phase ODE right-hand sides share one convention (clear)

All four dynamical RHS are written **outward-positive** and are algebraically identical:

| site | expression |
|---|---|
| `trinity/phase1_energy/energy_phase_ODEs.py:265` | `vd = (4πR2²(P_drive − P_ext) − mShell_dot·v2 − F_grav + F_rad)/mShell` |
| `phase1b_energy_implicit/run_energy_implicit_phase.py:616-623` | delegates to the same function |
| `phase1c_transition/run_transition_phase.py:226-228` | delegates to the same function |
| `phase2_momentum/run_momentum_phase.py:446-450` | `vd = (F_pressure − mShell_dot·v2 − F_grav + F_rad)/mShell` |

Derivation. The swept-up ambient gas is at rest before it joins the shell, so momentum balance for a
shell of mass `m(t)` is `d(mv)/dt = ΣF_out`, i.e. `m·v̇ + ṁ·v = 4πR²ΔP − F_grav + F_rad`, hence
`v̇ = [4πR²ΔP − ṁv − F_grav + F_rad]/m`. Gravity subtracted, radiation added, ram-loading subtracted —
all correct, and the same in all four phases. **Clear.**

`F_grav = G·m_sh/R² · (M_cl + 0.5·m_sh)` (four identical sites). The `0.5` is the thin-shell
self-gravity factor: a mass element in a thin shell sees, on average, half the shell interior to it.
**Clear.**

### 1.2 Energy RHS sign convention (clear)

`energy_phase_ODEs.py:280`: `Ed = (Lmech − L_bubble) − 4πR2²·press_bubble·v2 − L_leak`.
`get_betadelta.py:475`: `Edot_from_balance = Lmech − L_loss − 4πR2²·v2·Pb`, `L_loss = Lcool + Lleak`
(default `cooling_boost_mode='none'`). Identical: gain − radiative loss − PdV work − leak, all losses
negative. **Clear.**

Transition phase adds `Ed_soundcrossing = −Eb/(R2/c_sound)` (`run_transition_phase.py:238`) and takes
`min(Ed_balance, Ed_soundcrossing)` — negative, i.e. the *faster* loss. Sign and timescale correct.

### 1.3 Cooling sign convention is consistent across all three consumers (clear)

`read_cloudy.py:134` builds `netcooling = cool_cube − heat_cube` (positive = net cooling);
`net_coolingcurve.get_dudt` returns `−1 × dudt` on all three branches (lines 156, 165, 196), so
`get_dudt` is a **net energy gain rate**, negative when cooling. That matches
`bubble_luminosity.py:791` `dudt_cond = (heat_cond − cool_cond)` and matches the `− dudt/Pb` term in
the structure ODE (see §2.1). **Clear.**

### 1.4 Shell-structure ODE signs (clear)

`get_shellODE.py:115-122`, ionised branch:

* `dφ/dr = −4πr²·χ_e·α_B·n²/Q_i − n·σ_d·φ`. Photon budget: `dQ/dr = −(recombinations per shell) −
  (dust absorptions)`, `Q = Q_i φ`, `n_e n_p = χ_e n_H²`. Both terms are photon **sinks** ⇒ both
  negative. **Correct.**
* `dτ/dr = +n σ_d f_cover` — optical depth accumulates outward. **Correct** (but see SIGN-05).
* `dn/dr = (μ_p/μ_H)/(k_B T) · [ n σ_d (L_n e^{−τ} + L_i φ)/(4πr²c) + χ_e n² α_B L_i/(Q_i c) ]`.
  Hydrostatic balance against a body force is `dP/dr = f_r`; the radiation force is outward (`+r`), so
  `dP/dr > 0` and the density piles up at the shell's **outer** edge — the code's positive RHS is
  right. The prefactor is exact: with `ρ = μ_H n_H` and `n_tot = ρ/μ_p`, `P = n_H (μ_H/μ_p) k T`, so
  `dn/dr = (μ_p/(μ_H k T)) f_rad`. **Correct.**

### 1.5 All nine `solve_ivp` event functions and their `direction` flags (clear)

`phase_general/phase_events.py`: `R2−min_r` / dir −1; `R2−max_r` / dir +1; `v2+v_max` / dir −1
(collapse); `v_max−v2` / dir −1 (expansion — the function *decreases* as v2 rises, so −1 is right);
`v_max−|v2|` / dir −1; `R2−rCloud` / dir +1; `Eb−floor` / dir −1; `v2` / dir −1; `ratio−threshold` /
dir −1. Every one checked against the crossing it is meant to catch. **All correct.**

---

## Target 2 — Rational coefficients

### 2.1 Weaver+77 Eq. 42–43 bubble-structure ODE — full re-derivation (clear)

`bubble_luminosity.py:441-447`. Derived from scratch (no reference to the comment):

Continuity, `ρ = μP/(kT)` with `P` spatially uniform, and the self-similar Eulerian corrections
`∂P/∂t|_r = −βP/t`, `∂T/∂t|_r = δT/t − (αr/t)(dT/dr)`:

```
dv/dr = −2v/r + β/t + δ/t − (αr/t)(dT/dr)/T + v(dT/dr)/T
      = (β+δ)/t + (v − αr/t)(dT/dr)/T − 2v/r
```
↔ code line 446-447 **exactly**, including `(β+δ)` and `−2v/r`.

Internal energy `∂e/∂t + ∇·(ev) = −P∇·v + ∇·(κ∇T) + dudt`, `e = (3/2)P`:
`(3/2)Ṗ + (5/2)P∇·v = ∇·(κ∇T) + dudt`. Substituting `Ṗ = −βP/t` and the `∇·v` above:

```
LHS = P[ (−3/2)β/t + (5/2)β/t + (5/2)δ/t + (5/2)(v−αr/t)(dT/dr)/T ]
    = P[ β/t + 2.5·δ/t + 2.5·(v−αr/t)(dT/dr)/T ]      ← the β + 2.5δ asymmetry is real
```
and with `κ = C T^{5/2}`,
`∇·(κ∇T) = C T^{5/2}[ T'' + (5/2)T'²/T + (2/r)T' ]`, giving

```
d²T/dr² = P/(C T^{5/2})·[ (β+2.5δ)/t + 2.5(v−αr/t)T'/T − dudt/P ] − 2.5 T'²/T − 2 T'/r
```
↔ code lines 441-444 **exactly**. All four `2.5`s and the `2` are derived, not assumed. The
`− dudt/P` sign matches §1.3. **Clear.**

### 2.2 Weaver+77 Eq. 44 conduction-front boundary layer (clear)

`bubble_luminosity.py:401-409`. Balance conducted heat flux against the enthalpy flux of the
evaporating mass, `ṁ ≡ dMdt/(4πR2²)`, `h = (γ/(γ−1))kT/μ = (5/2)kT/μ`:

```
C T^{5/2} |dT/dr| = (5/2)(k/μ) T ṁ  ⇒  (2/5)C T^{5/2} = (5/2)(k/μ) ṁ s
⇒ T^{5/2} = (25/4)(k/(μC)) ṁ s,   s = R2 − r
```
↔ `constant = 25/4·k_B/mu_ion/C_thermal`, `T = (constant·dMdt·dR2/(4πR2²))^{2/5}`,
`dR2 = T_init^{5/2}/(constant·dMdt/(4πR2²))`. From `T ∝ s^{2/5}`, `dT/dr = −(2/5)T/dR2` ↔ line 408.
`v = αR2/t − (dMdt/4πR2²)·kT/(μ P)` is the gas velocity in the CD frame, `1/ρ = kT/(μP)`. **All clear.**

### 2.3 Rahner thesis Eq. A12 (`cool_beta ↔ Ėb`) — full algebraic verification (clear)

`get_bubbleParams.py:112-137` and the byte-identical copy `get_betadelta.py:248-269`; inverse at
`get_bubbleParams.py:183-189`. Derived independently:

With `E_b = 2π P_b d`, `d = R2³−R1³` (the γ=5/3 form of `E=PV/(γ−1)`), and the pressure-balance
relation `2 E_b R1² = F_ram·d`:

```
Ėb = 2π Ṗb d + 3(Eb/d)(R2² v2 − R1² Ṙ1)
Ṙ1 = [ Ḟ d + 3F R2² v2 − 2Ėb R1² ] / (4 R1 (Eb + c)),   using 3 F R1 = 4c with c = (3/4) F R1
```
Substituting term by term, with `a ≡ (3/2)Ḟ/F`:

* `(3Eb R1)/(4d(Eb+c)) · Ḟ d = (2/3)a c Eb/(Eb+c) = a Eb² R1³/(d(Eb+c))`  (using `c = (3/2)Eb R1³/d`)
* `(3Eb R1)/(4d(Eb+c)) · 3F R2² v2 = 3(Eb/d) R2² v2 · c/(Eb+c)`
* `(3Eb R1)/(4d(Eb+c)) · (−2Ėb R1²) = −Ėb·(3Eb R1³)/(2d(Eb+c)) = −Ėb·c/(Eb+c)`

which rearranges to exactly

```
Ėb (1−c_frac) = 2π Ṗb d + 3(Eb/d)R2²v2(1−c_frac) − a R1³Eb²/(d(Eb+c))
```
↔ the coded numerator/denominator. Every literal (`2π`, `3`, `1.5`, `0.75`) is derived. The inverse
(`Ebdot_to_cool_beta`) is the exact algebraic inversion, and `Pb_dot = −Pb·β/t` ↔
`β = −Ṗb t/Pb` round-trips. **Clear** — but see SIGN-01 for the hidden γ=5/3.

### 2.4 `bubble_E2P`, `pRam`, `get_r1`, `get_leak_luminosity` (clear at γ=5/3)

* `bubble_E2P` (`get_bubbleParams.py:237`): `Pb = (γ−1)Eb/[(R2³−R1³)(4π/3)] = (γ−1)Eb/V`. **Correct**,
  and γ-general.
* `pRam` (line 308): a free wind has `L = ½Ṁv²` ⇒ `ṗ = Ṁv = 2L/v`, so `P_ram = ṗ/(4πr²) = L/(2πr²v)`.
  **Correct**, and consistent with `update_feedback.py:181` `v_mech_total = 2L/ṗ`, which makes
  `pRam` return exactly `ṗ_total/(4πr²)` as the docstring claims. **Clear.**
* `get_r1` (line 409): balancing `ṗ/(4πR1²) = Pb` gives `R1² = 2L(R2³−R1³)/(3(γ−1)vEb)`. For γ=5/3
  `3(γ−1)=2` and this collapses to the coded `R1 = sqrt(L/(v·Eb)·(R2³−R1³))`. **Correct at γ=5/3
  only** → SIGN-01.
* `get_leak_luminosity` (line 284): enthalpy density `γP/(γ−1)`, times open area `(1−Cf)4πR2²`, times
  `c_s`. **Correct**, γ-general.

### 2.5 Weaver energy fraction `5/11` (clear)

`get_InitPhaseParam.py:28`. Derive: for the hot interior alone, `dE/dt = L − P dV/dt`, `E = (3/2)PV`,
`V = (4/3)πR³`, `Ṙ = (3/5)R/t`. Put `E = aLt`; then `P = aLt/(2πR³)` and `dV/dt = (12π/5)R³/t`, so
`P dV/dt = (6/5)aL` and `aL = L − (6/5)aL ⇒ a(11/5) = 1 ⇒ a = 5/11`. **Correct.**

Cross-check: substituting `E_th=(5/11)Lt` into the thin-shell momentum result `P = (7/25)ρR²/t²`
reproduces Weaver Eq. 21's coefficient `R⁵ = (250/308π)Lt³/ρ` exactly — an independent confirmation
that `5/11` (and not, e.g., 7/10) is the right constant for this closure.

### 2.6 Free-streaming duration and wind inversion (clear)

`get_InitPhaseParam.py:130,134,151`: `Ṁ = ṗ²/(2L)`, `v = 2L/ṗ` (inverse of `L=½Ṁv², ṗ=Ṁv`);
`(4/3)π(vt)³ρ_a = Ṁt ⇒ t = sqrt(3Ṁ/(4πρ_a v³))` ↔ code exactly. **Clear.**

### 2.7 Strömgren / recombination coefficients (clear)

* `shell_structure.py:144`: `max_shellRadius = (3Q_i/(4π χ_e α_B n0²))^{1/3} + r_start`. From
  `(4/3)πR_S³ χ_e n² α_B = Q_i`. **Correct.**
* `shell_structure.py:246`: `n_IF_Str = sqrt(3 Q_abs/(4π χ_e α_B ΔV))` with `ΔV = R_IF³ − R2³`.
  Balance `Q_abs = χ_e n² α_B (4π/3)(R_IF³−R2³)`; the `3/(4π)` compensates the cubed-radius
  difference. **Correct** (the local name `_vol_ion` for a cubed-radius difference is misleading but
  the arithmetic is right).
* `shell_structure.py:307`: IF density jump `n_neu = n_ion (μ_atom/μ_ion_shell)(T_ion/T_neu)` from
  pressure continuity `n(μ_H/μ)kT` — **correct**, and the direction (density jumps *up*) is right.
* `P_HII = (μ_H/μ_ion_shell) n_IF_Str k_B T_ion` at five sites
  (`run_energy_phase.py:214`, `run_energy_implicit_phase.py:981,1379`,
  `run_transition_phase.py:564,845`, `run_momentum_phase.py:634`) — identical to
  `energy_phase_ODEs.get_press_ion` (line 54) and the inverse of `shell_structure.py:124`. **Clear.**

### 2.8 Enclosed mass, power law (clear)

`mass_profile.py:332-336` (also duplicated verbatim at `powerLawSphere.py:136-140`,
`get_InitCloudProp.py:263-266`, `validate_gmc.py:435-438`):

`M = 4πρ_c[ r_c³/3 + (r^{3+α} − r_c^{3+α})/((3+α) r_c^α) ]` — integrate `∫₀^{r_c}4πr²ρ_c dr` plus
`4πρ_c r_c^{−α}∫_{r_c}^r r'^{2+α}dr'`. **Correct.** The analytic inversion at
`powerLawSphere.py:152-163` and the fractional-`rCore` variant `g = f³/3 + (1−f^{3+α})/((3+α)f^α)`
are exact rearrangements. `rCore_min = rCloud(nCore/nISM)^{1/α}` and
`nCore_min = nISM(rCloud/rCore)^{−α}` (`get_InitCloudProp.py:188,218`, `mass_profile.py:614`) are the
correct inversions of `nEdge = nCore(rCloud/rCore)^α`. **Clear.**

### 2.9 Bonnor–Ebert / isothermal Lane–Emden (clear)

`bonnorEbertSphere.py`:

* Series ICs (line 216-217). Substituting `u = a₂ξ² + a₄ξ⁴ + a₆ξ⁶` into `u''+(2/ξ)u' = e^{−u}`:
  `6a₂ = 1`, `20a₄ = −a₂`, `42a₆ = a₂²/2 − a₄` ⇒ `a₂=1/6, a₄=−1/120, a₆=1/1890` and
  `u' = ξ/3 − ξ³/30 + ξ⁵/315`. **Both lines exactly correct.**
* `m = ξ² du/dξ` (line 265): the LE equation is `d(ξ²u')/dξ = ξ²e^{−u}`, so
  `M = 4πa³ρ_c∫ξ²e^{−u}dξ = 4πa³ρ_c ξ²u'`. **Correct.**
* `a = c_s/√(4πGρ_c)` (line 418) and `c_s³ = M G^{3/2}√(4πρ_c)/m` (lines 407-408): from
  `M = 4π m ρ_c a³`, `a³ = c_s³/(4πGρ_c)^{3/2}` ⇒ `M = m c_s³/(G^{3/2}√(4πρ_c))`. **Correct.**
* Constants `Ω_crit=14.04`, `ξ_crit=6.451`, `m_crit=15.70`, `m_Bonnor=1.182` are mutually consistent:
  `m_B = ξ²u'√(e^{−u})/√(4π)` ⇒ `15.70·√(1/14.04)/√(4π) = 15.70·0.26688/3.5449 = 1.182`. **Clear.**

### 2.10 SPS mass loading (clear)

`read_sps.py:214-246`. With `Ṁ' = Ṁ(1+f)` and `v' = v√(θ/(1+f))`:
`L' = ½Ṁ'v'² = θL` (thermalisation efficiency, energy-consistent) and
`ṗ' = Ṁ'v' = ṗ√(θ(1+f))` (the standard mass-loading momentum boost). SN branch uses
`Ṁ_SN = 2L_SN/v_SN²` — the same `L=½Ṁv²` inversion. Totals are simple sums. **Clear.**

---

## Target 3 — Exponents

### 3.1 The Weaver self-similar exponent set is internally consistent (clear)

`registry.py:389-391` defaults: `cool_alpha = 0.6`, `cool_beta = 0.8`, `cool_delta = −6/35`.
`α ≡ dlnR/dlnt = 3/5` ✓. `P ∝ ρR²/t² ∝ t^{6/5−2} = t^{−4/5}` ⇒ `β = 4/5 = 0.8` ✓.
`δ = dlnT/dlnt = −6/35` ✓ (derived in §3.2). The runtime updates
`cool_alpha = t_now·v2/R2` (`run_energy_implicit_phase.py:662,798`,
`run_transition_phase.py:399`) match the definition, and `tSF` defaults to 0 so `t_now` *is* the
time since SF — the self-similar time origin is right. **Clear.**

### 3.2 Weaver Eq. 37 exponents 8/35, 2/35, −6/35, 0.4 — derived (clear)

`get_InitPhaseParam.py:172-176`. From the boundary layer (§2.2),
`T^{5/2} ∝ (k/μC)·Ṁ(1−x)/R` and (from §3.3) `Ṁ ∝ C^{2/7}(μ/k)P^{5/7}R^{17/7}t^{−5/7}`, so
`T ∝ C^{−2/7}P^{2/7}R^{4/7}t^{−2/7}(1−x)^{2/5}`. Inserting `P ∝ ρ^{3/5}L^{2/5}t^{−4/5}` and
`R ∝ (L/ρ)^{1/5}t^{3/5}`:

```
T ∝ ρ^{6/35 − 4/35} · L^{4/35 + 4/35} · t^{−8/35 + 12/35 − 10/35}
  = ρ^{2/35} L^{8/35} t^{−6/35}          and the profile factor is (1−x)^{2/5} = (1−x)^{0.4}
```
All four exponents **verified**. Unit handling is also right: `L/1e36` in erg/s via `L_au2cgs`,
`n` in cm⁻³ via `ndens_au2cgs`, and `t6 ≡ t/10⁶ yr = t[Myr]` so passing `dt_phase0` in Myr is
correct. The **argument** of `(1−·)^{0.4}` is not — see SIGN-03.

### 3.3 `_get_init_dMdt` — functional form derived, prefactor within 1.8% (clear, empirical)

`bubble_luminosity.py:305-308`:
`dMdt = (12/75)·1.646^{5/2}·4πR2³/t·(μ/k)·(tC/R2²)^{2/7}·P^{5/7}`.
Collecting powers gives `∝ 4π(μ/k)C^{2/7}P^{5/7}R2^{17/7}t^{−5/7}` — which is exactly what an
independent derivation produces. Doing that derivation (bubble mass
`M_b = 4π(μP/k)R2^{13/5}A^{−1}·B(3,3/5)` with `A^{5/2}=(25/4)(k/μC)(Ṁ/4πR2²)`, closed by
`M_b = Ṁt/(1+s)` and `Ṁ ∝ t^{6/35}` ⇒ `s = 6/35`) gives prefactor
`(41/35)^{5/7}·B(3,3/5)^{5/7}·(25/4)^{−2/7} = 1.1195·0.8537·0.5923 = 0.5661`, versus the coded
`(12/75)·1.646^{5/2} = 0.5561` — **1.8% apart**, i.e. the same constant to the accuracy of Weaver's
own numerical closure (an exact match would need `1.646 → 1.658`). This value is only the **seed**
for the `fsolve` on `dMdt` (`bubble_luminosity.py:261`), so the 1.8% has no effect on the converged
result. `1.646` is empirical (a Weaver fit constant, inherited from WARPFIELD), not derived — noted,
not flagged.

### 3.4 Density-profile exponents (clear)

`density_profile.py:143` `n = nCore(r/rCore)^α`; `mass_profile.py:334` `r^{3+α}` with the `1/(3+α)`
and `rCore^{−α}` normalisations; `α=−3` guarded (`powerLawSphere.py:143`). `compute_rCloud_homogeneous`
uses `(3M/4πρ)^{1/3}`. `dlaw.py:175` `log_ndens_offset = log10(ndens_au2cgs⁻¹) = −55.468` ✓. **Clear.**

---

## Target 4 — Factor-of-2 class

### 4.1 Unit-conversion table — all 18 constants re-derived (clear)

`_functions/unit_conversions.py:74-137`. Each constant recomputed from the three base factors
(`cm2pc`, `s2Myr`, `g2Msun`); every one agrees to ≤ 2.2e-16 relative:

`km2pc`, `ndens_cgs2au = cm2pc⁻³`, `phi_cgs2au = cm2pc⁻²·s2Myr⁻¹`, `E_cgs2au`, `L_cgs2au`,
`pdot_cgs2au`, `pdotdot_cgs2au`, `G_cgs2au`, `v_kms2au`, `v_cms2au`, `Pb_cgs2au`, `k_B_cgs2au`
(= `E_cgs2au` ✓), `c_therm_cgs2au` (= `pdotdot_cgs2au` ✓, same dimensions `g·cm/s³`),
`dudt_cgs2au`, `Lambda_cgs2au`, `tau_cgs2au`, `gravPhi_cgs2au`, `grav_force_m_cgs2au`.
`Pb_au2_KcmInv = Pb_au2cgs/k_B = 4686.7` ✓ matches its comment.

Cross-consistency identity, checked to machine precision:
`ndens_cgs2au² × Lambda_cgs2au = 4.877042454381258e25 = dudt_cgs2au`. This is what makes the two
cooling paths — `net_coolingcurve.get_dudt` (works in cgs, then `× dudt_cgs2au`) and
`bubble_luminosity` L1 (works in AU with `n_au² Λ_au`) — the *same* quantity, with no hidden factor.
**Clear.**

### 4.2 Composition constants (clear)

`read_param.py:308-345`, from `x_He = n_He/n_H` and the He ionisation state, with exact `Fraction`
arithmetic. At `x_He=0.1`, `Z_He=2`, `Z_He_shell=1`:

| quantity | derivation (particles per H nucleus) | value |
|---|---|---|
| `mu_convert` | mass per H nucleus `= 1+4x_He` | 1.4 |
| `mu_atom` | `μ_H/(1+x_He)` (H + He) | 14/11 |
| `mu_mol` | `μ_H/(½+x_He)` (H₂ + He) | 14/6 |
| `mu_ion` | `μ_H/(2+x_He(1+Z_He))` (H⁺,e,He²⁺,2e) | 14/23 |
| `mu_ion_shell` | `μ_H/(2+x_He(1+1))` | 7/11 |
| `chi_e` | `1+Z_He·x_He` | 1.2 |
| `chi_e_shell` | `1+1·x_He` | 1.1 |

All seven **correct** and mutually consistent; the hot-bubble (`mu_ion`, `chi_e`) vs shell
(`mu_ion_shell`, `chi_e_shell`) split is applied at the right call sites (bubble: `bubble_luminosity`
lines 427, 673, 725, 778, 811; shell: `get_shellODE`, `shell_structure`, `get_press_ion`). **Clear.**

Physical constants `C_thermal = 6e-7` (Spitzer, `κ = C T^{5/2}`), `caseB_alpha = 2.59e-13 cm³/s`,
`dust_sigma = 1.5e-21 cm²`, `dust_KappaIR = 4 cm²/g` all match standard literature values.

### 4.3 Geometric factors: `4πr²` vs `4πr²dr` vs `(4/3)πr³` (clear)

Spot-checked every occurrence of the three forms:
`shell_structure.py:176,264,333,368` (mass/gravity, `4πr²·dr`), `bubble_luminosity.py:746,793,830,936`
(luminosity integrand `4πr²`, cumulative mass `4π∫ρr²dr`), `mass_profile.py:224,316,320,327,340,479`
(`(4/3)πr³` for enclosed mass, `4πr²ρv` for `dM/dt`), plus every `4πR2²·P` force. No mis-typed form
found.

`Tavg = 3·Σ∫Tr²dr / Σ|Δ(r³)|` (`bubble_luminosity.py:865,870`): `⟨T⟩ = 4π∫Tr²dr /((4π/3)Δr³)` — the
factor 3 is right and the `4π`s cancel. **Clear.**

### 4.4 Quadrature `dr` conventions inside `shell_structure` (noted, not flagged)

`shell_structure.py:176` accumulates shell mass with a **right-endpoint** sum
(`n[1:]·4πr[1:]²·rShell_step`) while `dr_ion_arr` at line 276 and the `tau_kappa_IR` /
`phi_dust` / `phi_hydrogen` sums at 277-283, 389-395 use a **left-endpoint** sum with actual
differences. Both are `O(dr)`; on a uniform `np.arange` grid the two differ only at `O(dr)`.
Hygiene, not a defect — recorded here so a future session doesn't re-derive it.

---

## Target 5 — Cross-module coefficient divergence

### 5.1 `gamma_adia` is honoured in two places and hardcoded to 5/3 in the rest — **SIGN-01**

`bubble_E2P` (`get_bubbleParams.py:237`) and `get_leak_luminosity` (line 284) take `gamma` as a
parameter. The following are algebraically **locked to γ = 5/3** and take no `gamma`:

| site | hidden γ=5/3 |
|---|---|
| `get_bubbleParams.py:409` (`get_r1`) | `R1² = 2L(R2³−R1³)/(3(γ−1)vEb)`; coded as `L(...)/(vEb)`, i.e. `3(γ−1)=2` |
| `get_bubbleParams.py:123-134`, `get_betadelta.py:251-264` | A12 built on `E_b = 2πP_b d`, i.e. `V/(γ−1)` at γ=5/3; the `1.5` and `0.75` follow from it |
| `bubble_luminosity.py:401` | `25/4 = (5/2)²`, enthalpy `γ/(γ−1) = 5/2` |
| `bubble_luminosity.py:441-444` | `e = (3/2)P`, `2.5 = γ/(γ−1)` |
| `get_InitPhaseParam.py:28` | `5/11` (derived from `E=(3/2)PV`) |

Concrete consequence at γ = 1.4: `solve_R1` returns the γ=5/3 root, at which the wind ram pressure is
`Eb/(2π(R2³−R1³))`, while `bubble_E2P` returns `0.6 ×` that — a **67% pressure imbalance at the
contact discontinuity** the root was supposed to enforce. `gamma_adia` is a documented, user-settable
`.param` key (`registry.py:376`), so this is reachable, not theoretical.

### 5.2 `bubble_xi_Tb` means two different things — **SIGN-03**

* `get_InitPhaseParam.py:176`: `(1.0 − bubble_xi_Tb)**0.4` — this is Weaver Eq. 37's `(1−x)^{2/5}`
  with `x = r/R2`, so it treats ξ as `r/R2`.
* `bubble_luminosity.py:252`: `bubble_r_Tb = R1 + xi_Tb*(R2 − R1)` — treats ξ as a fraction of the
  bubble **thickness** measured from R1.
* `registry.py:408` info: *"The relative radius xi = r/R2"*; `registry.py:501` info for `bubble_r_Tb`:
  *"Radius at bubble_xi_Tb * R2"* — both agree with the first reading, contradicting the code that
  actually computes the radius.

At `r_Tb = R1 + ξ(R2−R1)`, the Weaver profile factor is `(1−r_Tb/R2)^{0.4} = [(1−ξ)(1−R1/R2)]^{0.4}`;
the code drops `(1−R1/R2)^{0.4}`. Measured on `param/simple_cluster.param` at `t0`:
`R1/R2 = 0.8692`, ξ = 0.98 ⇒ code uses `(1−ξ) = 2.0e-2` where the Weaver argument is
`2.617e-3` ⇒ **T0 is 2.256× too high**. `T0` is a state variable carried through phase 1a unchanged
and is the target of the phase-1b δ residual (`get_betadelta.py:491`), so the offset propagates.

### 5.3 The energy-phase `Pb` ramp is applied in the ODE but not in the bubble solver — **SIGN-02**

`get_effective_bubble_pressure` (`get_bubbleParams.py:368-374`) multiplies `R1` by `(t−tSF)/1e-3`
for `t ≤ tSF + 1e-3 Myr` before calling `bubble_E2P`. The ODE and `compute_derived_quantities`
both go through it (`energy_phase_ODEs.py:226, 362`). `bubble_luminosity.get_bubbleproperties_pure`
(line 228) and `get_betadelta.compute_R1_Pb` (line 329) call `bubble_E2P` **without** the ramp, and
it is *their* `Pb` that is written to `params['Pb']`, fed to `shell_structure` (line 104), and used
for the whole bubble structure solve.

Phase 1a runs to `TFINAL_ENERGY_PHASE = 3e-3 Myr`, so the ramp is live for the **first third of the
phase**. Measured on `simple_cluster` at `t0`: `Pb_ODE / Pb_solver = 0.343`, holding ≈0.343 until
`t−tSF ≈ 5e-4` and reaching 0.994 only at `t−tSF = 9.99e-4`. The shell is driven by 34% of the
pressure the bubble/shell modules are simultaneously assuming.

### 5.4 `P_drive` composition rule changes between phases — **SIGN-07**

`max(Pb, P_HII)` in energy/implicit (`energy_phase_ODEs.py:258`,
`run_energy_implicit_phase.py:538`), `max(Pb, P_HII + P_ram)` in transition
(`energy_phase_ODEs.py:255`), and `P_HII + P_ram` (a plain sum) in momentum
(`run_momentum_phase.py:264, 443`). The energy→transition and transition→momentum handoffs happen
where the two forms agree, so the trajectory is continuous; but "max" and "sum" are different
physical closures for the same two pressures, and the `max` makes `include_PHII` a no-op wherever
`Pb` dominates. This is deliberate and documented (`_analysis/check_yesno.py` exists solely to
diagnose it), so it is recorded as a convention divergence, not an error.

### 5.5 Reported force components do not sum to the integrated RHS — **SIGN-04**

`energy_phase_ODEs.compute_derived_quantities` (lines 396-421) reports **both**
`F_ram = Pb·4πR2²` and `F_HII = P_HII·4πR2²`, while the RHS it accompanies uses
`4πR2²(max(Pb,P_HII) − P_ext)`. Summing the published columns
(`F_ram + F_HII − F_ion_in − F_grav + F_rad`) therefore double-counts the driving pressure and does
not reproduce `mShell·vd + mShell_dot·v2`. Same pattern in
`run_energy_implicit_phase.py:534-539` and `run_transition_phase.py:334-338`.

Separately, the *meaning* of the `F_ram` output column changes by phase: `Pb·4πR2²` in
energy/implicit/transition, but a genuine `P_ram·4πR2²` in momentum
(`run_momentum_phase.py:272`). `trinity_reader.py:193` labels it flatly "Ram pressure force".

### 5.6 `dM/dt` uses the smoothed density; `M(r)` uses the sharp analytic integral — **SIGN-06**

`density_profile.py:130` applies a `tanh` bridge of width `0.01·rCloud` to `n(r)`, and
`mass_profile.py:224` computes `dMdt = 4πr²·ρ_smooth·v`. But `M(r)` comes from
`compute_enclosed_mass_powerlaw`, the **un-smoothed** analytic integral, and both are returned from
the same `get_mass_profile(..., return_mdot=True)` call that the ODEs consume
(`energy_phase_ODEs.py:208`). Measured (`simple_cluster`, `rdot = 1`):

| `r/rCloud` | `dMdt` (code) | `dM/dr` of the `M` actually used | ratio |
|---|---|---|---|
| 0.97 | 1.166e5 | 1.169e5 | 0.998 |
| 0.99 | 1.073e5 | 1.218e5 | 0.881 |
| 1.01 | 1.511e4 | 1.268e0 | **1.19e4** |
| 1.03 | 3.273e2 | 1.318e0 | 248 |

The *integrated* discrepancy over the band is small (the `tanh` is antisymmetric, so the excess
outside nearly cancels the deficit inside — the `O(SMOOTH_FRAC²)` claim in the comment at
`density_profile.py:126` is about that integral and is fine). But the *instantaneous* pair
`(mShell, mShell_dot)` handed to the ODE is internally inconsistent by up to four orders of magnitude
inside a ±3% band around `rCloud`, which is exactly where the `-mShell_dot·v2` momentum-loading term
is largest.

### 5.7 `-1e8` hardcoded acceleration override in the phase-1a RHS — **SIGN-08**

`energy_phase_ODEs.py:269-270`:

```python
if snapshot.EarlyPhaseApproximation:
    vd = -1e8
```

`EarlyPhaseApproximation` defaults **True** (`registry.py:423`) and is cleared only *after* the first
`solve_ivp` call (`run_energy_phase.py:342-344`), so this replaces the entire momentum RHS for the
whole first segment (`SEGMENT_DURATION = 3e-5 Myr`). The value is dimensional
(`pc/Myr²`) and scales with nothing.

Measured on `simple_cluster`:

* Free-streaming hand-off: `v0 = 3739 pc/Myr`, `dt_phase0 = 3.388e-7 Myr`, `r0 = 1.267e-3 pc`.
* The **actual** RHS at that state (`4πr²Pb − ṁv − F_grav`)/m = **−1.86e10 pc/Myr²** — the override
  is 186× too weak.
* The self-similar early deceleration `−(2/5)v/t` is `−4.41e9 pc/Myr²` — the override is 44× too weak.
* Integrating the override to the end of segment 1 (`Δt = 3e-5`) gives `R2 = 0.0684 pc`,
  `v2 = 739 pc/Myr`, versus the Weaver solution `R2 = 0.0165 pc`, `v2 = 330 pc/Myr` at the same time
  — **4.1× in radius, 2.2× in velocity**.

The *sign* is right (the shell must decelerate). The magnitude is arbitrary. Whether the following
~100 segments relax back onto the self-similar attractor is not measured here.

### 5.8 `f_cover` applied in one of four places — **SIGN-05**

`get_shellODE.py:122` applies `f_cover` to `dtaudr` in the **ionised** branch; line 144 (neutral
branch) does not; the radiation-force term in `dndr` (lines 116-117, 141) does not; and
`shell_structure.tau_kappa_IR` (lines 389, 395) does not. Currently inert — `shell_structure.py:115`
hardcodes `f_cover = 1` with a `TODO` — but the asymmetry is a trap for whoever wires
fragmentation in.

### 5.9 Registry docstrings drop the normalisation on β and δ — **SIGN-09**

`registry.py:390`: *"beta = - dPb/dt"*; line 391: *"delta = dT/dt"*. The code everywhere uses the
dimensionless logarithmic forms `β = −(t/Pb)dPb/dt` (`get_bubbleParams.py:112,189`) and
`δ = (t/T)dT/dt` (`get_bubbleParams.py:42,63`, `get_betadelta.py:294`). The defaults `0.8` and
`−6/35` are only meaningful under the code's definition. Doc-only.

---

## Clearances

Coefficients / signs / exponents checked and **confirmed correct**, so a future session need not
re-derive them. Derivations are in the sections cited.

| item | site | verdict |
|---|---|---|
| Outward-positive force convention, all 4 phase ODEs | §1.1 | correct & mutually consistent |
| Thin-shell self-gravity `0.5·m_sh` (4 sites) | §1.1 | correct |
| Energy RHS `L − Lcool − PdV − Lleak` (ODE vs β-residual) | §1.2 | identical, correct |
| `get_dudt` net-gain sign convention (3 branches, 3 consumers) | §1.3 | consistent |
| Shell ODE `dn/dr`, `dφ/dr`, `dτ/dr` signs + prefactors | §1.4 | correct |
| All 9 `solve_ivp` event functions + `direction` flags | §1.5 | correct |
| Weaver Eq. 42-43 structure ODE: `β+δ`, `β+2.5δ`, three `2.5`s, `−2v/r`, `−2T'/r` | §2.1 | derived, exact |
| Weaver Eq. 44 boundary layer: `25/4`, `2/5`, `−2/5` | §2.2 | derived, exact |
| Rahner A12 forward + inverse: `2π`, `3`, `1.5`, `0.75`, `c_frac` | §2.3 | derived, exact |
| `bubble_E2P` `(γ−1)/(4π/3)` | §2.4 | correct |
| `pRam = L/(2πr²v)` ↔ `v_mech = 2L/ṗ` ↔ `ṗ/(4πr²)` | §2.4 | correct round trip |
| `get_r1` pressure balance | §2.4 | correct **at γ=5/3** (SIGN-01) |
| `get_leak_luminosity` enthalpy flux `γ/(γ−1)` | §2.4 | correct |
| Weaver `5/11` energy fraction (+ cross-check vs Eq. 21's `250/308π`) | §2.5 | derived, exact |
| `Ṁ=ṗ²/2L`, `v=2L/ṗ`, `dt=√(3Ṁ/4πρv³)` | §2.6 | derived, exact |
| Strömgren radius, `n_IF_Str`, IF density jump, `P_HII` (5 sites) | §2.7 | correct & consistent |
| Power-law enclosed mass + analytic inversion + `rCore_min`/`nCore_min` (4 duplicate sites) | §2.8 | correct, all agree |
| Lane-Emden series `ξ²/6 − ξ⁴/120 + ξ⁶/1890`, `u'` series, `m=ξ²u'`, `a`, `c_s³` | §2.9 | derived, exact |
| BE critical constants 14.04 / 6.451 / 15.70 / 1.182 | §2.9 | mutually consistent |
| SPS mass-loading: `L'=θL`, `ṗ'=ṗ√(θ(1+f))`, `Ṁ_SN=2L/v²` | §2.10 | correct |
| `α=3/5`, `β=4/5`, `δ=−6/35` default set | §3.1 | mutually consistent; `tSF=0` origin OK |
| Weaver Eq. 37 exponents `8/35`, `2/35`, `−6/35`, `0.4` + unit handling | §3.2 | derived, exact |
| `_get_init_dMdt` functional form `∝ (μ/k)C^{2/7}P^{5/7}R^{17/7}t^{−5/7}` | §3.3 | derived; prefactor 1.8% (seed only, empirical `1.646`) |
| 18 unit-conversion constants + `ndens²·Λ = dudt` identity | §4.1 | all ≤2.2e-16 rel. error |
| `mu_atom/mu_ion/mu_mol/mu_convert/mu_ion_shell/chi_e/chi_e_shell` from `x_He, Z_He` | §4.2 | all correct |
| `C_thermal`, `caseB_alpha`, `dust_sigma`, `dust_KappaIR` | §4.2 | standard literature values |
| `4πr²` / `4πr²dr` / `(4/3)πr³` usage across all sites | §4.3 | no mis-typed form |
| `Tavg = 3∫Tr²dr/Δ(r³)` | §4.3 | correct |
| `get_soundspeed` `√(γkT/μ)` + unit chain | `operations.py:203` | correct |
| `dlaw.py` `log_ndens_offset = −55.468` | `dlaw.py:175` | correct |
| BE `T_eff = μ c_s²/(γ k)` (γ in an *isothermal* sphere) | `bonnorEbertSphere.py:431` | γ cancels in the `r↔ξ` round trip via `r2xi`; documented at `registry.py:525`. Not a defect. |

---

```json
[
  {
    "id": "SIGN-01",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 409,
    "class": "divergence",
    "severity": "S2",
    "claim": "gamma_adia is honoured by bubble_E2P and get_leak_luminosity but hardcoded to 5/3 in solve_R1/get_r1, in the Rahner-A12 Ebdot<->beta pair, and in the whole Weaver bubble-structure chain; setting gamma_adia != 5/3 makes R1 and Pb describe different pressure balances.",
    "evidence": "Pressure balance p_dot/(4*pi*R1^2) = Pb with p_dot = 2L/v and Pb = (gamma-1)Eb/((4pi/3)(R2^3-R1^3)) gives R1^2 = 2L(R2^3-R1^3)/(3(gamma-1) v Eb). get_r1:409 codes R1 = sqrt(L/(v*Eb)*(R2^3-R1^3)), which is that expression only when 3(gamma-1)=2, i.e. gamma=5/3. bubble_E2P:237 meanwhile uses the gamma passed in. Companion hardcodes: get_bubbleParams.py:123-134 and get_betadelta.py:251-264 (A12 built on Eb = 2*pi*Pb*d, the gamma=5/3 form of Eb=PV/(gamma-1); the 1.5 and 0.75 follow from it); bubble_luminosity.py:401 (25/4 = (5/2)^2 from enthalpy gamma/(gamma-1)=5/2); bubble_luminosity.py:441-444 (e=(3/2)P, 2.5=gamma/(gamma-1)); get_InitPhaseParam.py:28 (5/11, derived from E=(3/2)PV). gamma_adia is a user-settable .param key, registry.py:376 default '5/3'.",
    "expected": "Either thread gamma through get_r1/A12/the structure ODE, or validate gamma_adia == 5/3 at load and make bubble_E2P/get_leak_luminosity take the same constant, so one setting cannot silently desynchronise two halves of the same pressure balance.",
    "failure_scenario": "A user sets gamma_adia = 1.4 in a .param. solve_R1 returns the gamma=5/3 root; at that R1 the wind ram pressure is Eb/(2*pi*(R2^3-R1^3)) while bubble_E2P returns 0.6x that, a 67% pressure imbalance at the contact discontinuity the root was supposed to enforce. Eb, Pb, the structure solve and the beta residual are then all built on inconsistent thermodynamics, with no warning.",
    "repro": "python3 -c \"import trinity.bubble_structure.get_bubbleParams as g; R2=1.0;Eb=1e3;L=1e4;v=3e3;\\nfor gam in (5/3.,1.4):\\n  R1=g.solve_R1(R2,Eb,L,v); Pb=g.bubble_E2P(Eb,R2,R1,gam); import numpy as np; print(gam, R1, Pb, (2*L/v)/(4*np.pi*R1**2)/Pb)\\\"  # last column is P_ram(R1)/Pb: 1.000 at 5/3, 1.667 at 1.4",
    "confidence": "high"
  },
  {
    "id": "SIGN-02",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 368,
    "class": "divergence",
    "severity": "S2",
    "claim": "The early-phase R1 ramp inside get_effective_bubble_pressure is applied only on the ODE path; the bubble-structure solver and the beta-delta residual compute Pb from the un-ramped R1, so during the first 1e-3 Myr of phase 1a the shell is driven by ~34% of the pressure every other module assumes.",
    "evidence": "get_bubbleParams.py:368-374 scales R1 by (t-tSF)/1e-3 for t <= tSF+1e-3 before calling bubble_E2P. energy_phase_ODEs.py:226 (RHS) and :362 (diagnostics) both route through it. bubble_luminosity.py:228 (get_bubbleproperties_pure) and get_betadelta.py:329 (compute_R1_Pb) call bubble_E2P directly with the true R1, and it is that Pb that is stored to params['Pb'] (run_energy_phase.py:191) and consumed by shell_structure.py:104/124. Phase 1a runs to TFINAL_ENERGY_PHASE=3e-3 Myr (run_energy_phase.py:54), so the ramp covers its first third.",
    "expected": "Either both paths use the ramped Pb, or neither. If the ramp is a numerical softener for the R1->R2 degeneracy at t~tSF, it belongs inside bubble_E2P/solve_R1 so every consumer sees the same number.",
    "failure_scenario": "Every default run today. The momentum RHS accelerates the shell with Pb_ramped while shell_structure sets the shell's inner density n0 from Pb_unramped (a factor ~3 higher) and the bubble structure/cooling solve uses Pb_unramped throughout; the resulting R2(t), Lcool and shell profile over the first third of phase 1a are not a solution of any single consistent pressure.",
    "repro": "simple_cluster.param, after read_param+get_InitCloudProp+read_sps+get_y0: R1=solve_R1(r0,E0,L,v_mech); Pb_true=bubble_E2P(E0,r0,R1,gamma); Pb_ode=get_effective_bubble_pressure('energy',E0,r0,R1,gamma,L,v_mech,t=tSF+dt,tSF=tSF). Ratio Pb_ode/Pb_true = 0.3434 at dt=3.39e-7 and dt=1e-5, 0.3436 at 1e-4, 0.3741 at 5e-4, 0.9943 at 9.99e-4.",
    "confidence": "high"
  },
  {
    "id": "SIGN-03",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 176,
    "class": "coefficient",
    "severity": "S2",
    "claim": "bubble_xi_Tb is consumed under two incompatible definitions: the Weaver Eq. 37 factor (1-xi)**0.4 treats it as x = r/R2, while bubble_r_Tb = R1 + xi*(R2-R1) treats it as a fraction of the bubble thickness. The missing (1-R1/R2)**0.4 makes the initial T0 2.26x too high on the tracked baseline.",
    "evidence": "get_InitPhaseParam.py:176 codes (1.0 - bubble_xi_Tb)**0.4; Weaver Eq. 37's profile factor is (1-x)^{2/5} with x = r/R2 (derived in the report, section 3.2). bubble_luminosity.py:252 computes bubble_r_Tb = R1 + xi_Tb*(R2-R1). For that radius, 1 - r_Tb/R2 = (1-xi)(1-R1/R2), so the correct factor is [(1-xi)(1-R1/R2)]**0.4. registry.py:408 ('The relative radius xi = r/R2') and registry.py:501 ('Radius at bubble_xi_Tb * R2') both document the r/R2 reading, contradicting bubble_luminosity.py:252. T0 is a state variable (run_energy_implicit_phase.py:614, y[3]) carried unchanged through phase 1a and is the target of the delta residual (get_betadelta.py:491).",
    "expected": "Pick one definition. If bubble_r_Tb keeps the thickness form (it must, to satisfy the assert bubble_r_Tb > R1 at bubble_luminosity.py:254), then get_InitPhaseParam.py:176 needs ((1.0 - bubble_xi_Tb)*(1.0 - R1/R2))**0.4, and the two registry info strings need correcting.",
    "failure_scenario": "Every run today. The initial characteristic bubble temperature is over-estimated by a factor that grows as R1/R2 -> 1 (the strong-wind / early-time regime), and the phase-1b delta solver then has to distort delta to reconcile a structure temperature that was never going to match its target.",
    "repro": "simple_cluster.param at t0: R1/R2 = 0.8692, xi_Tb = 0.98. Code argument (1-xi) = 2.000e-2; Weaver argument (1-xi)(1-R1/R2) = 2.617e-3; ratio of the 0.4 powers = 2.256.",
    "confidence": "medium"
  },
  {
    "id": "SIGN-04",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 396,
    "class": "divergence",
    "severity": "S3",
    "claim": "The published force columns double-count the driving pressure: F_ram (= Pb*4piR2^2) and F_HII (= P_HII*4piR2^2) are both written out while the RHS they accompany uses 4piR2^2*max(Pb, P_HII), so the reported forces cannot be summed into the integrated momentum budget. Separately, the F_ram column means Pb-force in three phases and true ram-pressure force in the fourth.",
    "evidence": "energy_phase_ODEs.py:396 F_HII = 4*pi*R2**2*P_HII and :421 F_ram = Pb*4*pi*R2**2, while :258 P_drive = max(press_bubble, P_HII) and :265 uses only P_drive. Same at run_energy_implicit_phase.py:534-539 (with P_drive = max(Pb,P_HII) at :538) and run_transition_phase.py:334-338. run_momentum_phase.py:272 sets F_ram = P_ram*FOUR_PI*R2**2 with P_ram from pRam, a different quantity. trinity_reader.py:193 labels the column flatly 'Ram pressure force'; registry.py:480 says 'Ram pressure force (from Pb-Eb relation)'.",
    "expected": "Publish the force actually used (F_drive = 4piR2^2*P_drive) alongside the components, and either rename the energy-phase F_ram (e.g. F_thermal) or make the momentum phase report its ram force under a distinct key, so a cross-phase plot of F_ram compares like with like.",
    "failure_scenario": "Anyone reconstructing the force budget from dictionary.jsonl -- a paper figure, or an audit checking momentum conservation -- gets a residual equal to min(Pb, P_HII)*4piR2^2 and concludes the integrator is wrong. A cross-phase F_ram(t) plot shows a spurious discontinuity at the transition->momentum handoff because the column's definition changes there.",
    "repro": "Read any dictionary.jsonl row from an energy-phase snapshot and evaluate (F_ram + F_HII - F_ion_in - F_grav + F_rad) vs shell_mass*dv2/dt + shell_massDot*v2; the mismatch is exactly 4*pi*R2**2*min(Pb, P_HII).",
    "confidence": "high"
  },
  {
    "id": "SIGN-05",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 122,
    "class": "divergence",
    "severity": "S4",
    "claim": "f_cover is applied to dtau/dr in the ionised branch only; the neutral branch, both radiation-force terms, and the tau_kappa_IR integral all omit it. Inert today (f_cover is hardcoded to 1) but wired inconsistently for whoever enables fragmentation.",
    "evidence": "get_shellODE.py:122 dtaudr = nShell*sigma_dust*f_cover (ionised) vs :144 dtaudr = nShell*sigma_dust (neutral). The dndr radiation terms at :116-117 and :141 carry no f_cover. shell_structure.py:389,395 build tau_kappa_IR from nShell*dr with no f_cover. shell_structure.py:114-115 hardcodes f_cover = 1 with a '# TODO: Add f_cover from fragmentation mechanics'.",
    "expected": "Decide once whether f_cover attenuates the optical-depth accumulation, the radiation force, or both, and apply it identically in both branches and in tau_kappa_IR. A single module-level helper would prevent the split.",
    "failure_scenario": "The TODO is implemented by setting f_cover < 1 in shell_structure.py:115. The ionised region's dust optical depth is reduced but the neutral region's is not, so the neutral absorption fraction 1-exp(-tau) and the IR-trapping term in F_rad stay at their sealed-shell values -- a partially covered shell absorbs like a full one over half its extent.",
    "repro": "Set f_cover = 0.5 at shell_structure.py:115 and compare shell_fAbsorbedNeu / shell_tauKappaRatio against f_cover = 1: they are unchanged when the shell has a neutral region, while shell_fAbsorbedIon moves.",
    "confidence": "high"
  },
  {
    "id": "SIGN-06",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 224,
    "class": "divergence",
    "severity": "S3",
    "claim": "get_mass_profile(return_mdot=True) returns an M(r) computed from the sharp analytic power law and a dM/dt computed from the tanh-smoothed density, so the pair handed to the ODEs is internally inconsistent by up to ~1e4x within +/-3% of rCloud.",
    "evidence": "density_profile.py:128-130 applies a tanh bridge of width SMOOTH_FRAC=0.01*rCloud to n(r). mass_profile.py:209/224 evaluates rho from that smoothed profile for dMdt = 4*pi*r**2*rho*rdot, while :214 -> compute_enclosed_mass_powerlaw (:332-342) integrates the un-smoothed profile analytically. Both are returned from the same call and consumed together at energy_phase_ODEs.py:208-217 (mShell divides the RHS, mShell_dot is the momentum-loading term). The comment at density_profile.py:126 ('mass conservation holds to O(SMOOTH_FRAC^2)') is about the integral, not the instantaneous derivative.",
    "expected": "Make dM/dt the derivative of the M actually used -- either integrate the smoothed profile numerically for M(r), or drop the smoothing from dM/dt and regularise the ODE some other way.",
    "failure_scenario": "A shell crossing the cloud edge. Just outside rCloud the ODE removes momentum at a rate set by a density up to 1.2e4x the one implied by the shell mass it is dividing by; the two errors have opposite sign either side of rCloud so the net momentum sink over the crossing is only ~O(SMOOTH_FRAC) off, but any per-step diagnostic (shell_massDot, the force budget, a blowout criterion keyed on mass loading) is meaningless inside the band.",
    "repro": "simple_cluster.param: for f in (0.97,0.99,1.01,1.03): r=f*rCloud; M,dMdt = get_mass_profile(r,p,return_mdot=True,rdot=1.0); dMdr = central difference of get_mass_profile(r,p). Ratios dMdt/dMdr = 0.998, 0.881, 1.19e4, 248.",
    "confidence": "high"
  },
  {
    "id": "SIGN-07",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 253,
    "class": "divergence",
    "severity": "S3",
    "claim": "The rule combining bubble and HII pressure into P_drive changes across phases: max(Pb, P_HII) in energy/implicit, max(Pb, P_HII+P_ram) in transition, and a plain sum P_HII+P_ram in momentum. 'max' and 'sum' are different closures for the same two pressures.",
    "evidence": "energy_phase_ODEs.py:253-258 (max(Pb,P_HII) for energy/implicit, max(Pb, P_HII+P_ram) for transition); run_energy_implicit_phase.py:538 P_drive = max(Pb, P_HII); run_momentum_phase.py:264 and :443 P_drive = snapshot.P_HII + P_ram. The consequence is documented: _analysis/check_yesno.py exists specifically to explain why toggling include_PHII changes nothing while Pb dominates the max.",
    "expected": "State the closure once (a module-level helper taking the phase), and say in the parameter docs that include_PHII is a no-op in the energy/implicit phases whenever Pb > P_HII, so the knob is not mistaken for an on/off switch on the HII contribution.",
    "failure_scenario": "A parameter study varying include_PHII or Qi reports 'HII pressure has no effect' for energy-dominated clouds, because the max() silently discards it -- while the same knob is fully additive once the run reaches the momentum phase. The apparent sensitivity of a result to HII pressure then depends on which phase the run spent its time in.",
    "repro": "python -m trinity._analysis.check_yesno -f <folder of paired _yesPHII/_noPHII runs>; the script's own EXPECTED verdict is this divergence.",
    "confidence": "medium"
  },
  {
    "id": "SIGN-08",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 270,
    "class": "numerical",
    "severity": "S2",
    "claim": "vd = -1e8 replaces the entire momentum RHS for the whole first phase-1a segment on every default run. The constant is dimensional (pc/Myr^2), scales with nothing, and is 186x weaker than the RHS it replaces on the tracked baseline.",
    "evidence": "energy_phase_ODEs.py:269-270 overwrites vd unconditionally when snapshot.EarlyPhaseApproximation is true. registry.py:423 defaults EarlyPhaseApproximation to True; run_energy_phase.py:342-344 clears it only after the first solve_ivp returns, and the snapshot is frozen at the top of the loop (energy_phase_ODEs.py:159), so the override holds for the full SEGMENT_DURATION = 3e-5 Myr (run_energy_phase.py:55,141). No derivation or citation accompanies the value.",
    "expected": "Either drop the override (the RHS is finite at t0) or replace it with a scaled form -- the self-similar early deceleration is -(2/5)*v2/(t-tSF), which is dimensionally correct and adapts to L, rho and v0.",
    "failure_scenario": "Every default run. On simple_cluster the true RHS at the hand-off state is -1.86e10 pc/Myr^2 and the self-similar value is -4.41e9; the override's -1e8 keeps the shell near free-streaming ~180x longer than physical, leaving R2 = 0.0684 pc / v2 = 739 pc/Myr at the end of segment 1 where the Weaver solution gives 0.0165 pc / 330 pc/Myr (4.1x and 2.2x). Because the constant is absolute, a cluster with a 10x weaker wind or a 100x lower density gets a completely different-sized error, in the opposite direction.",
    "repro": "simple_cluster.param, after get_y0: R1=solve_R1(r0,E0,L,v_mech); Pb=bubble_E2P(E0,r0,R1,gamma); m,mdot=get_mass_profile(r0,p,return_mdot=True,rdot=v0); vd=(4*pi*r0**2*Pb - mdot*v0 - G*m/r0**2*(mCluster+0.5*m))/m -> -1.8584e10. Weaver at dt=3e-5: R=(250/(308*pi))**0.2*(L*dt**3/rho)**0.2 = 1.651e-2 pc, v = 0.6*R/dt = 330 pc/Myr; override trajectory: R = r0+v0*dt-0.5e8*dt**2 = 6.844e-2 pc, v = v0-1e8*dt = 739 pc/Myr.",
    "confidence": "high"
  },
  {
    "id": "SIGN-09",
    "file": "trinity/_input/registry.py",
    "line": 390,
    "class": "citation",
    "severity": "S4",
    "claim": "The registry info strings for cool_beta and cool_delta omit the logarithmic normalisation, so the documented definitions do not match the code and the defaults 0.8 / -6/35 look dimensionally impossible.",
    "evidence": "registry.py:390 info='Cooling related values. beta = - dPb/dt.' and :391 info='Cooling related values. delta = dT/dt.'. The code uses beta = -(t/Pb)*dPb/dt (get_bubbleParams.py:112 Pb_dot = -Pb*beta/t_now; :189 cool_beta = -Pb_dot*t_now/bubble_P) and delta = (t/T)*dT/dt (get_bubbleParams.py:42,63; get_betadelta.py:294). Both are dimensionless logarithmic derivatives, which is what makes beta=4/5 and delta=-6/35 the Weaver self-similar values (verified in the report, section 3.1).",
    "expected": "info='beta = -(t/Pb) dPb/dt (dimensionless; Weaver self-similar value 4/5)' and 'delta = (t/T) dT/dt (dimensionless; Weaver value -6/35)'. cool_alpha's string (registry.py:389, 'alpha = v2*t_now/R2') is already in the right form and can serve as the template.",
    "failure_scenario": "A user reading the parameter documentation sets cool_beta in physical units of dPb/dt, or concludes the defaults are arbitrary rather than the Weaver exponents, and perturbs them without realising that alpha=3/5, beta=4/5 and delta=-6/35 are a single self-consistent set.",
    "repro": "Compare registry.py:390-391 against get_bubbleParams.py:112 and :189.",
    "confidence": "high"
  }
]
```
