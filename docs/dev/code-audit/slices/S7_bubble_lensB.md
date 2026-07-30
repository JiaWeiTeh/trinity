# S7 bubble structure — Lens B (what the code claims)

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

**Input:** `phase2/S7_bubble/prose.md` only (comments + docstrings, no code).
**Files covered:** `trinity/bubble_structure/bubble_luminosity.py`, `trinity/bubble_structure/get_bubbleParams.py`.
**Nothing below is a statement about what the code does** — every line is a transcription of what the
prose *asserts*, phrased so another lens can test it. Where I say "flag", I mean the prose is
internally inconsistent, too vague to test as written, or admits a defect.

---

## 1. Named constants and the values the prose attaches to them

| Constant (as named in prose) | Claimed value | Claimed meaning | Cite |
|---|---|---|---|
| `_T_INIT_BOUNDARY` (unnamed in prose; "the T_init boundary") | `3e4` K | Outer/cold boundary temperature for the **backward** bubble-structure integration | `bubble_luminosity.py:41`, `:964`, `:665` |
| `_coolingswitch` | `1e4` K | "Temperature at which any lower will have no cooling"; explicitly **NOT** the same quantity as `_T_INIT_BOUNDARY` | `bubble_luminosity.py:49`, `:702` |
| `_CIEswitch` | value never written numerically; implied `10**5.5` K | CIE / non-CIE switch; "Temperatures higher than `_CIEswitch` results in switching on CIE" | `bubble_luminosity.py:704`, `:686`, `:736` |
| f_A interface-band top | "= the non-CIE/CIE switch" (⇒ `10**5.5` K) | Top of the band where `cooling_boost_fA` scales the source term | `bubble_luminosity.py:60` |
| min-T logging threshold | "K below `_T_INIT_BOUNDARY` before the rejection is worth logging" — number not in prose | DEBUG-log gate only | `bubble_luminosity.py:58` |
| solve_ivp structure `rtol` | `1e-8` | "matches odeint's former default accuracy (~1.49e-8)" | `bubble_luminosity.py:86` |
| `_RESIDUAL_RTOL` | `1e-6` | looser rtol for the velocity-residual solve | `bubble_luminosity.py:94` |
| dMdt residual grid | `500` points | coarse `t_eval` for the residual solve (vs the ~60k production grid) | `bubble_luminosity.py:102` |
| `_CONDUCTION_NPTS` | `2000` | dense-output samples across the conduction band for the L/Tavg trapezoids | `bubble_luminosity.py:110` |
| `MIN_SPACING` (`min_relative_spacing`) | `1e-12` | minimum **relative** spacing between consecutive grid radii | `bubble_luminosity.py:571` |
| production radius grid | "60k-point", built from **three** `np.logspace` chunks stitched with `np.insert` | output sampling grid, **decreasing** order | `bubble_luminosity.py:532` |
| fsolve settings (external, quoted) | `xtol=1e-4`, `epsfcn=1e-4` | the dMdt root-find the residual serves | `bubble_luminosity.py:95` |
| diagnostic gates | `TRINITY_BUBBLE_DIAG=1`; `TRINITY_BUBBLE_STATE_DUMP=<N>`; `TRINITY_BUBBLE_STATE_DT` (default `1.0`) | env-var gated, observational only | `bubble_luminosity.py:956`, `:1087`, `:1106` |

---

## 2. Formulas asserted in prose

**F1 — volume-weighted mean temperature** (`bubble_luminosity.py:854`)

    <T> = ∫T dV / ∫dV = 3 × Σ(∫ T r² dr) / Σ | r_outer³ − r_inner³ |

with the stated sign rationale: `r_bubble` and `r_conduction` are **descending** slices,
`r_interm = linspace(r2Prime, R2_coolingswitch)` is **ascending**; `abs()` prevents the intermediate
term from "subtract[ing] its volume", and with `abs()` "the three terms telescope to the true
full-domain volume `R2_coolingswitch³ − R1³`" (`:855`–`:859`).

**F2 — f_A effective luminosity** (`bubble_luminosity.py:839`–`:844`)

    L_eff = L1 + fA * (L2 + L3)

Stated justification: `|∫ f·g| = f·|∫ g|` for constant f. L1 (CIE interior) and `L_leak` are
**deliberately not** scaled ("no mixing interface there / bulk escape"). `fA != 1.0` guard ⇒ default
path byte-identical.

**F3 — cumulative bubble mass** (`bubble_luminosity.py:934`, `:930`)

    M(r) = ∫[0 → r] 4π r'² ρ(r') dr' ,   ρ = mu_H · n_H  [Msun/pc³]

Computed by O(n) cumulative integration "instead of O(n²) loop with simps"; arrays flipped to
monotonically increasing first (`:928`).

**F4 — gravity (DISABLED / commented out)** (`bubble_luminosity.py:939`–`:946`)

    grav_phi     = −4π G ∫ r ρ dr            (scipy.integrate.simpson(r_new*rho_new, x=r_new))   [pc²/Myr²]
    grav_force_m = G · m_cumulative / (r² + 1e-10)                                               [pc/Myr²]

`1e-10` is stated to be there "to avoid division by zero at r=0".

**F5 — Rahner A12, bubble energy rate** (`get_bubbleParams.py:71`–`:110`, verbatim)

    E_b_dot = [ 2π·Pb_dot·d² + 3·E_b·R_b_dot·R_b²·(1 − c/(E_b+c)) − a·R_ts³·E_b²/(E_b + c) ]
              / [ d · (1 − c/(E_b+c)) ]

    a ≡ (3/2) · F_ram_dot / F_ram      [1/time]
    c ≡ (3/4) · F_ram · R_ts           [energy]
    d ≡ R_b³ − R_ts³                   [length³]

Symbol map claimed: `Pb_dot`←dP_b/dt, `Eb`←E_b, `R2,v2`←R_b,Ṙ_b, `R1`←R_ts (termination shock),
`pdot_total`←F_ram, `pdotdot_total`←F_ram_dot, `c_frac`←c/(E_b+c).

**F6 — beta definition** (`get_bubbleParams.py:111`, `:142`–`:168`)

    beta = −(t / P_b) · dP_b/dt          ⇒   cool_beta = − Pb_dot · t_now / P_b

`Ebdot_to_cool_beta` claims to solve F5 for `Pb_dot` and return F6.

**F7 — bubble E→P** (`get_bubbleParams.py:199`–`:238`) — formula not written out; cited as Rahner
thesis "pg71 Eq 6". Inputs `Eb [au]`, `r2 [pc]`, `r1 [pc]`, `gamma`; output `bubble_P [au]`;
internals stated to be cgs (`:219` "Make sure units are in cgs", `:238` "return back in au"). The
denominator is stated to be a shell volume ∝ `(r2³ − r1³)` which is floored (`:230`–`:235`).

**F8 — leak luminosity** (`get_bubbleParams.py:243`–`:279`, verbatim)

    Lleak = gamma/(gamma−1) · (1 − Cf) · 4π R2² · Pb · c_sound

Cf = *closed* fraction; Cf = 1 ⇒ sealed (Weaver) ⇒ **exactly 0**. Physical story: hot gas escapes
through open area `(1−Cf)·4πR2²` at the interior sound speed carrying its enthalpy.

**F9 — ram pressure** (`get_bubbleParams.py:287`–`:307`, verbatim)

    P_ram = L_mech / (2π r² v_mech)

Stated to be called with the **total** mechanical luminosity (winds + SNe) and the corresponding
total mechanical velocity, "e.g. in the momentum and transition phases".

**F10 — R1 root equation endpoints** (`get_bubbleParams.py:415`–`:434`, verbatim)

    at r1 = 0 :   f = sqrt( Lmech / v / Eb · R2³ ) > 0
    at r1 = R2:   f = −R2 < 0
    ⇒ bracket [0, R2] always contains the root when Lmech_total > 0

(The full `get_r1` expression is *not* transcribed; only these two endpoint values and "derived by
balancing pressure" `get_bubbleParams.py:385`.)

**F11 — effective bubble pressure branch rule** (`get_bubbleParams.py:314`–`:367`)

    energy / implicit phase : P = bubble_E2P(Eb, R2, R1, gamma)   [+ early-phase R1 ramp-up if t,tSF given]
    momentum phase          : P = pRam(R2, Lmech_total, v_mech_total)
    transition phase        : P = max(P_thermal, P_ram)

---

## 3. Units and unit conventions asserted

- `bubble_luminosity.py:696` — **"All cooling calculations take in au values, but the inner
  operations and outputs are cgs. The exception is `get_dudt()`, which takes in au and returns in au."**
- `bubble_luminosity.py:698` — "Do[u]ble check units when using interpolation functions; some of
  them also only take log10 or 10^." (sic, typo in source prose)
- Region-by-region unit tags: `:743` cooling rate **[au]** (region 1, CIE) · `:749` "[K pc3]" ·
  `:777` "calculate array [au]" · `:783` cooling rate **[cgs]** (region 2) · `:790` net cooling rate
  **[au]** · `:792` integrand **[au]** · `:794` power loss **[au]**.
- `:930` ρ in **Msun/pc³**; `:940` grav potential **pc²/Myr²**; `:945` grav force/mass **pc/Myr²**.
- `:532`–`:551` `R1 [pc]`, `r2Prime [pc]` where **r2Prime ≡ R2 − dR2**.
- `get_bubbleParams.py:243`–`:279` — code units **[Msun, pc, Myr]**; `Pb [Msun/pc/Myr²]`,
  `c_sound [pc/Myr]`, `R2 [pc]`, result **[Msun·pc²/Myr³]**; explicit claim that
  `Pb·c_sound·R2²` *already* lands in the luminosity unit so **no conversion is applied**
  (asserted in `test/test_cf_leak.py`). *(This one is dimensionally self-consistent as written.)*
- `get_bubbleParams.py:287`–`:307` — `r [pc]`, `Lmech [au]`, `v_mech [pc/Myr]`, `P_ram [au]`.
- `get_bubbleParams.py:385` — `get_r1` "units of au".
- `get_bubbleParams.py:415` — `solve_R1` returns R1 in **[pc]**.
- `get_bubbleParams.py:314`–`:348` — `Eb [au]`, `R2/R1 [pc]`, `t/tSF [Myr]`, returns `press_bubble [au]`.
- `_quiet_lsoda_fortran` — conduction layer thickness quoted as **~1e-10 pc** for a massive-cluster
  wind (`bubble_luminosity.py:121`).

---

## 4. Citations — recorded verbatim

| Where | Citation, verbatim | What is attributed to it |
|---|---|---|
| `bubble_luminosity.py:3` | "Weaver+77 bubble-structure ODE" | whole module |
| `bubble_luminosity.py:238` | "**Equation 33 in Weaver+77**" | estimate for dMdt when not yet computed |
| `bubble_luminosity.py:298` | "Initial guess for dMdt (**Equation 33 in Weaver+77**)." | `_get_init_dMdt` (same eq. as above — consistent) |
| `bubble_luminosity.py:393` | "Get initial conditions for bubble ODE (**Eq 44 in Weaver+77**)." | `_get_bubble_ODE_initial_conditions` |
| `bubble_luminosity.py:415` | "Bubble structure ODE (**Equations 42-43 in Weaver+77**)." | `_get_bubble_ODE` RHS |
| `get_bubbleParams.py:28` | "See **Pg 79, Eq A5**, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf" | `delta2dTdt` |
| `get_bubbleParams.py:48` | "See **Pg 79, Eq A5**, <same URL>" | `dTdt2delta` (inverse of the above — one equation, two functions; benign) |
| `get_bubbleParams.py:71`, `:128` | "**pg 80, Eq A12**", "Main equation (**Rahner thesis A12**)" | `cool_beta_to_Ebdot`, formula F5 |
| `get_bubbleParams.py:142` | "**pg 80, Eq A12**" | `Ebdot_to_cool_beta` (stated inverse of F5 — same eq., two functions; benign) |
| `get_bubbleParams.py:226` | "see <Rahner URL> **pg71 Eq 6**" | bubble pressure from energy (F7) |
| `get_bubbleParams.py:385` | "see **Rahners thesis, eq 1.25**" | the `get_r1` pressure-balance equation |
| `get_bubbleParams.py:243` | "See the **leakage spec, Eq. (leak)**" | F8 — no locatable document or numbered equation |

**Four distinct Weaver+77 numbers are claimed: 33, 42, 43, 44.** Eq 33 is attached to the *same*
quantity in both places it appears, so there is no internal conflict — but nothing in the prose lets
me verify any of the four against the paper. Recorded here verbatim so the literature/code lens can.
The Rahner thesis is cited with **three different numbering styles** ("pg 79, Eq A5", "pg 80, Eq
A12", "pg71 Eq 6", "eq 1.25") — a mixed appendix/chapter numbering that is worth checking is not a
transcription slip.

---

## 5. Contracts, state and ordering claims

**Purity (headline contract).** `bubble_luminosity.py:3`–`:16`, `:200`–`:214`:
"returns them in a BubbleProperties dataclass **instead of mutating the params dict**";
"**No dictionary mutations during calculation**"; "Use `updateDict(params, bubble_data)` **after** the
call returns"; `params : DescribedDict Parameter dictionary (**read-only access**)".

**`BubbleProperties` field inventory** (`:172`–`:196`): total luminosity loss; temperature at
`bubble_r_Tb`; average temperature; bubble mass; L in bubble region (CIE); L in conduction zone
(non-CIE); L in intermediate region; velocity array; temperature array; temperature gradient array;
radius array; density array; mass flux from shell into hot region; inner bubble radius; bubble
pressure; radius at T_goal.

**`_solve_bubble_structure` return contract** (`:454`–`:476`): returns `(psoln, ok, infodict, sol)`.
`psoln` shape `(len(r_array), 3)` = `[v, T, dTdr]`, `psoln[0]` **is the initial condition**;
`ok = sol.success` and **"when False the caller must not consume psoln"**; `infodict` has keys
`message`/`status`/`nfev`/`nst`/`hu`; `sol` is the dense-output object — "the structure path samples
the conduction zone from it; **the velocity-residual solve ignores it**". `t_span` is the *actual*
grid span `(r_array[0], r_array[-1])` "so sampling `r_array` never extrapolates".

**Ordering invariants.** `T_array` is in **increasing** order ⇒ `r_array` in **decreasing** order
(`:271`–`:272`, restated `:692` "Remember r is monotonically decreasing, so temperature increases!",
and `:551` "Cleaned radius array in decreasing order (for backward integration)").
`_get_mass_and_grav` "flip[s] arrays to be monotonically increasing" first (`:928`).

**Failure contract chain.** ODE RHS detects T→0 ⇒ raises `BubbleSolverError` (explicitly *not*
`sys.exit`, "SystemExit is not an Exception", `:419`–`:422`) ⇒ `_solve_bubble_structure` converts to
`ok=False` (`:485`–`:487`); non-finite `y0` makes `solve_ivp` raise `ValueError`, also converted to
`ok=False` (`:477`–`:481`); `sol.sol` may be `None` if the solve failed before any step (`:517`);
`ok=False` ⇒ `BubbleSolverError` or the deterministic fsolve penalty (`:75`–`:78`, `:81`–`:83`).

**Called-in-two-places invariant.** `get_effective_bubble_pressure` "**MUST** be called in both the
ODE and in `compute_derived_quantities` to guarantee consistency between the integrator and
diagnostics" (`get_bubbleParams.py:314`).

**Argument-wrapping contract asymmetry.** `cool_beta_to_Ebdot` — params "Must provide **`.value`**
for: Pb, cool_beta, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total" (`:71`). Its stated inverse
`Ebdot_to_cool_beta` — "my_params : dict-like Must provide t_now, pdot_total, pdotdot_total, R2, v2,
Eb (**plain float values, not `.value`-wrapped**)" (`:142`).

**Three-place lockstep.** The f_A band-top constant, the local `_CIEswitch` in `_bubble_luminosity`,
and the cooling-table-derived `nonCIE_Tcutoff` in `net_coolingcurve._noncie_cutoffs` "**THREE places
must stay in lockstep**"; they only "coincide on the default bundle; **a table swap moves the
third**" — enforced by a pinning test in `test/test_fA_source_boost.py` (`:60`–`:64`). The
`_T_INIT_BOUNDARY` constant likewise plays "**THREE coupled roles, all of which must move
together**" (`:41`–`:48`).

**Observational-only side effects.** `_capture_bubble_integration` "only reads psoln, saves files,
and logs — it **never alters** T_array or the result"; "Never mutates state or raises into the
caller" (`:967`, `:986`). `_dump_bubble_state`: "**Byte-identical to before when unset**; never
mutates state or raises into the caller" (`:1093`); skips the runtime cooling cubes (reconstructed
offline via `read_param` + `get_coolingStructure`), stores `R1, Pb, dMdt, r2Prime,
initial_conditions` + structure arrays. `_quiet_lsoda_fortran` "touches no numerics" (`:121`).

---

## 6. Regimes, assumptions, guards

- **Three cooling regions** (`:685`–`:688`, `:736`, `:752`, `:799`):
  1. bubble / CIE, `T > 10**5.5 K`, "low resolution";
  2. conduction zone / non-CIE, stated at `:752` as `10**4 < T < 10**5.5 K`, "high resolution";
  3. intermediate, "between 1e4 and `T_array[index_cooling_switch]`", built by interpolating
     between `R2_prime` and `R2_1e4` "because the cooling function varies a lot between 1e4 and
     1e5K (R2_prime is above 1e4)"; "tiny (or non-existent)" when `R2_prime` is very close to `R2`.
- **Table bound**: "The non-CIE cooling table is **only defined for T < 10**5.5**, so the band is
  masked to it" (`:765`).
- **Conduction band actually sampled**: `[r2Prime → r(T=10**5.5)]` from the dense-output solution,
  *not* a re-solve (`:758`–`:764`).
- **Stiffness regime**: "The bubble-structure ODE is stiff near small T" (`:70`); LSODA
  step-underflow warnings fire crossing the ~1e-10 pc conduction layer — "a stiff but finite regime,
  **NOT an overflow**"; the solve "still SUCCEEDS and the profile is verified correct (matches Radau
  / tight-LSODA references to ~1e-6)" (`:121`).
- **`r_Tb` sanity**: "r_Tb cannot be smaller than the inner bubble radius R1"; r_Tb is "relative to
  bubble thickness" (`:249`–`:253`).
- **Leak guards** (`get_bubbleParams.py:243`–`:281`): returns 0 when `Cf >= 1` (sealed),
  `Pb <= 0` (depressurised / numerical undershoot), or `c_sound <= 0` (no hot-gas temperature yet)
  — "so the term self-limits and **never injects energy**". `c_sound` must be evaluated at
  `bubble_Tavg`, "**NOT the cold-shell value**". Documented domain `Cf ∈ (0, 1]`.
- **E→P degeneracy guard** (`get_bubbleParams.py:230`–`:235`): under catastrophic cooling Eb
  collapses and R1→R2, so `(r2³ − r1³)` underflows to 0 in float64 → inf/ZeroDivisionError → Eb=nan;
  the shell volume is floored; energy phases detect `Eb<=0` and hand off (phase 1b → momentum,
  phase 1a stops). Claimed "**Bit-identical on every physical bubble (shell_volume > 0)**".
  A second, separate "avoid division by zero" note sits at `:223`.
- **`solve_R1` regimes** (`get_bubbleParams.py:415`–`:437`): `Lmech_total <= 0` ⇒ no termination
  shock ⇒ **returns 0.0**; `R2 <= 0` or NaN (transient ODE excursion during the energy-driven Eb→0
  collapse) ⇒ **returns 0.0**, keeping the RHS finite so the integrator rejects the step instead of
  `sqrt(<0)` → NaN → brentq crash; "**Raises on root-finding failure for a physical bubble instead
  of fabricating a value**"; non-finite Eb/Lmech/v_mech with physical R2 **raise explicitly**
  because "**scipy < 1.11 brentq silently converges on a NaN-poisoned function (returns ~1e-12
  instead of raising)**". Bracket widened from `[1e-3*R2, R2]` (which "missed roots below 1e-3*R2
  and raised") to `[0, R2]`. `get_r1` itself "set[s] minimum energy to avoid zero" (`:405`).
- **Transition-phase handoff rationale** (`get_bubbleParams.py:353`–`:357`): `max(P_thermal, P_ram)`
  "to ensure **smooth** handoff"; as Eb decays on the sound-crossing timescale P_thermal drops while
  P_ram stays roughly constant, so by the time Eb hits the energy floor P_ram dominates and the
  switch is "continuous".

---

## 7. Numerical / convergence claims (each is a measurable assertion)

1. `rtol=1e-8` "sits well inside the regime where the integrated bubble outputs are
   rtol-independent" (`:86`–`:90`).
2. Residual solve at `rtol=1e-6`: "the converged dMdt then shifts by **<= 0.3%** (measured)"
   (`:94`–`:99`).
3. Residual grid: "converged dMdt is insensitive to this in **[200, 2000]** (**rel_dMdt <= 3e-6** vs
   the 60k grid across **6 configs**); 500 is the conservative pick" (`:102`–`:107`).
4. Conduction trapezoid: "converges fast (**~1/K**2**): **K=2000 is within ~7e-5** of the K→infinity
   value at **~1 ms/call**", vs a former "~100-point conduction re-solve" (`:110`–`:115`).
5. Residual definition, stated only in passing: "That residual (**v[-1]/v[0]** plus min-T /
   monotonic checks)" (`:95`); "the residual only needs v at the **endpoints** plus the min_T /
   monotonic guards along the path" (`:329`).
6. **Monotonicity guard**: `find_nearest_higher` raises `MonotonicError`; the gated diagnostic exists
   to disambiguate its **two known triggers** — `"dead_integrator"` (LSODA gives up; T-profile has a
   zero/non-finite tail at the hot/inner end) and `"boundary_transient"` (a small smooth dip at the
   T_init=3e4 outer edge, "confined to the first **~0.1%** of points; the bulk is monotonic")
   (`:960`–`:965`). The diagnostic fires only if the profile is "non-finite, non-monotonic, or has a
   sub-floor tail" (`:986`).
7. **min_T rejection**: profiles whose `min_T` dips below the T_init anchor are penalised so fsolve
   is steered away; "the ~**1e-4 K** floating-point undershoot at the boundary edge is benign
   (**penalty ~1.0**)" and is only *logged* if the dip exceeds a threshold in K (`:46`, `:54`–`:58`).
8. **Deterministic failure residual**: "large and non-zero so fsolve is steered away from the
   infeasible dMdt instead of falsely converging on a garbage (~0) residual" (`:81`–`:83`, `:331`).
9. Grid cleaning: near-duplicates at `np.insert` join boundaries have "differences of **~1e-8 to
   1e-9**"; removal enforces a **minimum relative spacing of 1e-12**; reference magnitude is "the
   average magnitude of consecutive points"; "First point is always kept" (`:571`–`:607`).
10. `bubble_diag` has a "cap on saved events per process" (`:969`); `_dump_bubble_state` optionally
    requires "`t_now` to have grown by `TRINITY_BUBBLE_STATE_DT` between dumps (**default 1.0 = no
    spacing = first-N behavior**)" (`:1106`–`:1108`).

---

## 8. Admissions, known debt, historical scars

- **`get_bubbleParams.py:48`–`:61`** — `dTdt2delta` parameter `T` documented literally as
  "**DESCRIPTION.**" (unfinished docstring placeholder). Neither `delta2dTdt` nor `dTdt2delta`
  transcribes its equation or defines `delta`.
- **`bubble_luminosity.py:916`–`:927`** — "**gravity outputs currently DISABLED**"; returns `None`
  placeholders "so any future consumer that reads them before re-enabling the block **fails loudly**
  instead of silently integrating zero gravity"; the block "is kept commented below so it can be
  restored **verbatim**".
- **`bubble_luminosity.py:627`–`:635`** — "an earlier plan to add a separate primary path with this
  as fallback was **dropped**, which is why this was once suffixed `_legacy`".
- **`bubble_luminosity.py:280`–`:285`, `:636`–`:637`** — an adaptive shock-concentrating grid "was
  **tried and dropped** — it under-sampled the conduction zone".
- **`bubble_luminosity.py:662`–`:667`** — the unphysical-solution check is "**Near-unreachable** on a
  successful solve"; it previously "killed the process via `sys.exit`".
- **`bubble_luminosity.py:70`–`:74`** — the former `odeint` path "returned **UNINITIALISED memory**
  for the un-integrated tail, which made the whole bubble solve **nondeterministic**".
- **`bubble_luminosity.py:1042`** — "**rCloud was a red herring for the spike; kept as reference**"
  (an unresolved "spike" investigation is referenced but not described).
- **`bubble_luminosity.py:571`–`:576`** — grid cleaning downgraded from correctness fix to "**grid
  hygiene**"; kept anyway.
- **`bubble_luminosity.py:698`** — "**Dobule** check units" (typo); `get_bubbleParams.py:3` "grep
  \"Section\" **so** jump between different sections" (typo).
- **`bubble_luminosity.py:36`, `:320`** — two numpy-2.x compatibility shims (`trapz`→`trapezoid`;
  `float(size-1 array)` → `.item()`), in a project whose `CLAUDE.md` pins `numpy<2`.
- **`get_bubbleParams.py:70`, `:141`, `:402`** — three "old code:" renaming notes
  (`beta_to_Edot`/`beta2Edot`, `Edot_to_beta`/`Edot2beta`, `R1_zero`).

---

## 9. Flags (prose vs prose, vague, unit- or contract-inconsistent)

**A. Conduction-band lower bound contradicts the region split and the telescoping volume.**
`:752` labels region 2 as `10**4 < T < 10**5.5 K`. But `:758`–`:765` says the band actually sampled
is `[r2Prime → r(T=10**5.5)]`, and `r2Prime` is by construction the `T_init = 3e4 K` anchor
(`:41`–`:45`); region 3 separately covers `1e4 → r2Prime` (`:799`–`:802`). If region 2 really ran
down to `1e4 K` it would overlap region 3 (double-counted cooling) and the `:858`–`:859` claim that
the three volumes "telescope to … `R2_coolingswitch³ − R1³`" would be false. Related: `:688` calls
region 3 "between 1e4 and `T_array[index_cooling_switch]`" while `:702` defines the cooling switch as
`1e4` — read literally that region is empty. → **S7-B-01**

**B. `abs()` is documented for the volume only; the ∫T r² dr numerator has the same orientation
problem.** `:855`–`:859` explains the sign fix purely in terms of "subtract its volume". Two of the
three numerator terms are integrated over descending r and one over ascending r. → **S7-B-02**

**C. Cooling-rate unit labels differ between regions.** `:696` states a blanket convention
(cooling calls: au in, **cgs** out; only `get_dudt` is au→au). Region 1 then labels its cooling rate
**[au]** (`:743`) with no conversion step visible, while region 2 labels its cooling rate **[cgs]**
(`:783`) and converts to au two lines later (`:790`). Both feed the same summed total (`:850`).
→ **S7-B-03**

**D. Grid-cleaning threshold looks orders of magnitude too small for the defect it names.**
Duplicates are "differences of ~1e-8 to 1e-9" (`:573`); removal threshold is a *relative* spacing of
`1e-12` (`:590`). At r of order 0.1–100 pc an absolute 1e-8–1e-9 gap is a relative 1e-8–1e-11, i.e.
comfortably **above** the 1e-12 cut, so the documented threshold would keep the documented
duplicates. Either the magnitudes are relative (not stated) or the constant does not do what the
docstring says. → **S7-B-04**

**E. f_A band consistency between the ODE and the luminosity split is asserted but not derivable.**
The in-ODE boost applies "in the interface band ONLY" with the top = the CIE switch (`:60`, `:432`)
and no stated bottom; the ODE domain bottoms out at `T_init = 3e4`. The luminosity side scales
`L2 + L3` (`:839`–`:843`) — and L3 is the `1e4 → 3e4 K` band that the ODE never integrates. → **S7-B-05**

**F. "Suppresses only that noise" is broader than described.** The stated mechanism is redirecting
the **C-level stdout/stderr file descriptors to /dev/null for the call** (`:121`–`:134`). That
swallows anything else written to fd 1/2 during the solve (other libraries, other threads, logging
configured to stdout), not just the LSODA banner. → **S7-B-06**

**G. "Bit-identical on every physical bubble (shell_volume > 0)" vs a floor.**
`get_bubbleParams.py:230`–`:235`. A floor on `(r2³ − r1³)` can only be bit-identical for
`shell_volume >= floor`; any positive volume below the floor is altered. Also two separate
divide-by-zero notes (`:223` and `:230`) for what may be one guard. → **S7-B-07**

**H. γ = 5/3 is baked into A12 and the R1 endpoints, but is a parameter elsewhere.** F5's leading
`2π` is exactly `(3/2)·(4π/3)`, i.e. `E_b = 2π P_b d` for γ=5/3, and `cool_beta_to_Ebdot` takes no
gamma. The F10 endpoint `sqrt(Lmech/v/Eb·R2³)` is likewise the γ=5/3 pressure balance. Meanwhile
`bubble_E2P`, `get_leak_luminosity` and `get_effective_bubble_pressure` all take `gamma` explicitly.
→ **S7-B-08**

**I. `M(r) = ∫[0 → r]` but the grid starts at R1.** `:934` states the lower limit as 0; the arrays
are the bubble-structure grid over `[R1, r2Prime]` (`:532`–`:551`). → **S7-B-09**

**J. The "restore verbatim" gravity block would not return what its unit comment claims.**
`grav_phi` as written (`:941`–`:943`) is a single `simpson` over the whole array — a scalar, not a
radial profile — and omits the `−G M(r)/r` interior term of a potential; `grav_force_m` adds `1e-10`
"to avoid division by zero **at r=0**" on a grid whose minimum radius is `R1 > 0`. → **S7-B-10**

**K. Two different stated sources for solver success.** `:454`–`:476`: "`ok` — `sol.success`".
`:1053`: "(No 'ier' key; **success is read from 'status'**)." → **S7-B-11**

**L. The velocity residual is never defined where it lives.** `_get_velocity_residuals`' own
docstring is one line, "Calculate velocity residual for dMdt solver" (`:312`); the only definition is
a parenthetical in a *constant's* comment, "v[-1]/v[0] plus min-T / monotonic checks" (`:95`). How
the min_T penalty composes with that residual (additive? multiplicative? replacement?) is never
stated, and "penalty ~1.0" for a benign undershoot (`:56`) sits awkwardly next to "large and
non-zero" for a failed solve (`:82`) without a stated scale separation. → **S7-B-12**

**M. Admitted lockstep fragility with no code-level enforcement.** Three constants "must stay in
lockstep" but only "coincide on the default bundle", and "a table swap moves the third" (`:60`–`:64`).
The only stated defence is a pinning test. Same shape at `:41`–`:48` (one constant, three roles).
→ **S7-B-13**

**N. `get_effective_bubble_pressure`'s docstring omits a branch and overstates smoothness.**
The docstring documents energy and momentum only (`:314`–`:348`); the transition branch
`max(P_thermal, P_ram)` and the "early-phase R1 ramp-up" exist only as body comments
(`:353`–`:357`, `:366`–`:367`). `max()` is continuous but has a derivative kink, so "smooth handoff"
is not what a `max` provides. → **S7-B-14**

**O. Inverse function pair with opposite argument-wrapping contracts.**
`.value`-wrapped for `cool_beta_to_Ebdot`, plain floats for `Ebdot_to_cool_beta` (`:71`, `:142`).
→ **S7-B-15**

**P. Two citations too vague to check.** "the leakage spec, **Eq. (leak)**" names no document
(`get_bubbleParams.py:243`); "Pg 79, Eq A5" is attached to two functions whose equations are never
written down and one of whose parameters is documented as "DESCRIPTION." (`:28`, `:48`). → **S7-B-16**

**Q. Weaver+77 equation numbers, recorded for cross-check.** Eq 33 (dMdt seed, cited twice,
consistently), Eq 44 (initial conditions), Eqs 42-43 (structure ODE). Internally consistent; not
verifiable from prose. → **S7-B-17**

**R. R1 geometry described backwards.** `get_bubbleParams.py:381`: "R1 = interface separating inner
bubble radius and outer solar wind" — the free (stellar, not solar) wind is *inside* R1 and the hot
shocked bubble *outside*, which is what every other mention says ("R1 : Inner bubble radius (wind
termination shock)", `:199`–`:217`, `:415`). → **S7-B-18**

**S. `TRINITY_BUBBLE_STATE_DT` semantics ambiguous.** "require `t_now` to have **grown by**
`TRINITY_BUBBLE_STATE_DT` between dumps (default **1.0 = no spacing**)" (`:1106`–`:1108`). "Grown by"
reads additive; only a *ratio* test makes 1.0 mean "no spacing". → **S7-B-19**

**T. numpy-2.x compatibility paths under a `numpy<2` pin.** `:36`, `:320`. → **S7-B-20**

**U. Leak guards have no lower bound on `Cf`.** Documented domain is `Cf ∈ (0, 1]` and guards cover
`Cf >= 1`, `Pb <= 0`, `c_sound <= 0` (`:243`–`:279`); a `Cf < 0` would make `(1 − Cf) > 1`, i.e. an
open area exceeding the full sphere. The "self-limits" claim covers energy *injection*, not
over-draining. → **S7-B-21**

**V. The purity contract is stated three times and is the module's headline claim.** "No dictionary
mutations during calculation" / "read-only access" (`:3`–`:16`, `:200`–`:214`), while the gated
diagnostic and state-dump paths both read `params` and claim "never mutates state", and
`_get_init_dMdt` reads a previously-stored dMdt guess (`:244`–`:245`) that must have been written
back by `updateDict`. Worth a direct check that no path writes `params` before return. → **S7-B-22**

**W. Region 1 is declared CIE-only yet imports "two cooling curves".** `:686`/`:736` — "This is the
CIE region … CIE is used" — but `:741` says "import values from two cooling curves", the same comment
used in region 2 (`:780`) and echoed by region 3's explicit "both CIE and non-CIE regimes" (`:813`).
`:765` states the non-CIE table is "only defined for T < 10**5.5". → **S7-B-23**

**Nomenclature gap (noted, not filed):** `r2Prime` is defined (`R2 − dR2`), but `R2_1e4`,
`R2_coolingswitch`, `index_cooling_switch`, `r_cz`, `T_goal`/`rgoal`, `bubble_r_Tb` and `dR2` are
used without definition anywhere in the prose; `:872`–`:882` ("If rgoal is smaller than the radius of
cooling threshold, i.e., larger than the index, looking for the smallest value in r_cz … otherwise,
interpolate") is not testable as written.

---

```json
[
  {
    "id": "S7-B-01",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 752,
    "class": "regime",
    "severity": "S2",
    "claim": "Region 2 (conduction zone, non-CIE) is the band 10**4 K < T < 10**5.5 K.",
    "evidence": ":752 'Conduction zone. High resolution region, 10**4 < T < 10**5.5 K.' contradicted by :758-:765 (band actually sampled is [r2Prime -> r(T=10**5.5)], and r2Prime is the T_init=3e4 K anchor per :41-:45), by :799-:802 (region 3 covers 1e4 -> r2Prime), and by :858-:859 ('the three terms telescope to the true full-domain volume R2_coolingswitch**3 - R1**3'). :688 additionally defines region 3 as 'between 1e4 and T_array[index_cooling_switch]' while :702 sets the cooling switch to 1e4, which reads as an empty interval.",
    "expected": "Region 2 lower bound should be the r2Prime / T_init boundary (3e4 K), not 1e4 K; region 2 and region 3 must partition [R1, R2_coolingswitch] without overlap for the telescoping volume claim to hold.",
    "failure_scenario": "If the code uses 1e4 K as region 2's lower bound, cooling in 1e4-3e4 K is counted in both L_conduction and L_intermediate (inflated total cooling luminosity, hence wrong Eb evolution), and the <T> denominator no longer telescopes to R2_coolingswitch**3 - R1**3.",
    "repro": "Check the region-2 mask bounds against T(r2Prime); assert L1+L2+L3 volumes sum to (R2_coolingswitch**3 - R1**3) on param/simple_cluster.param.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-02",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 854,
    "class": "sign",
    "severity": "S2",
    "claim": "<T> = 3 * SUM(int T r^2 dr) / SUM|r_outer^3 - r_inner^3|; abs() is needed because r_interm is ascending while r_bubble/r_conduction are descending, otherwise the intermediate term 'would carry the wrong sign and (incorrectly) subtract its volume'.",
    "evidence": ":854-:859. The stated sign rationale mentions only the volume (denominator). The numerator terms int T r^2 dr are integrated over the same slices with the same opposite orientations, but no sign handling is documented for them.",
    "expected": "Either all three numerator terms are made consistently signed (flip/abs/ordered integration) or the mixed orientation is explicitly justified; the documented formula has abs() only on the denominator.",
    "failure_scenario": "Descending-r trapezoids return negative integrals; summing them with an ascending-r positive term gives a numerator that partially cancels, producing a bubble_Tavg that is too small (or negative), which then feeds c_sound in get_leak_luminosity and the beta-delta residual.",
    "repro": "Compare bubble_Tavg against a direct 3*int(T r^2 dr)/(R2_cs^3 - R1^3) recomputed on a single ascending concatenated grid.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-03",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 696,
    "class": "units",
    "severity": "S2",
    "claim": "'All cooling calculations take in au values, but the inner operations and outputs are cgs. The exception is get_dudt(), which takes in au and returns in au.'",
    "evidence": "Blanket convention at :696-:697, but region 1 labels its cooling rate '[au]' at :743 with no conversion step in the prose, while region 2 labels its cooling rate '[cgs]' at :783 and only reaches '[au]' at :790. Region 1's L and region 2's L are summed into the total at :850.",
    "expected": "Both region cooling rates should be in the same unit system before summation; the unit tags on the two analogous quantities should match, or the conversion should be visible on both paths.",
    "failure_scenario": "If region 1's cooling rate is genuinely cgs but treated as au (or vice versa), L_bubble is wrong by the cgs<->au luminosity conversion factor, silently biasing the total cooling loss and the energy-phase Eb budget.",
    "repro": "Unit-check L_bubble and L_conduction independently against an analytic constant-Lambda test profile; see test/test_conventional_units.py conventions.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-04",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 571,
    "class": "numerical",
    "severity": "S3",
    "claim": "np.insert join boundaries create near-duplicate points with 'differences of ~1e-8 to 1e-9'; _clean_radius_grid removes them by enforcing a minimum RELATIVE spacing, default 1e-12 (MIN_SPACING), using the average magnitude of consecutive points as reference.",
    "evidence": ":571-:595, :599-:607. A gap of 1e-8..1e-9 at radii of order 0.1-100 pc is a relative gap of ~1e-8..1e-11, i.e. above the stated 1e-12 cut, so the documented threshold would not remove the documented duplicates.",
    "expected": "Either the quoted 1e-8/1e-9 differences are relative (docstring should say so), or MIN_SPACING is too small to do what the docstring claims and the cleaning is a no-op.",
    "failure_scenario": "Cleaning silently removes nothing; harmless today ('grid hygiene only' per :575) but the docstring misleads anyone who re-enables an odeint-style dense-output path or tunes the constant.",
    "repro": "Call _create_radius_grid(R1, r2Prime) for a realistic R1~1e-3 pc, r2Prime~1 pc and compare len() before/after cleaning; check min relative consecutive spacing.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-05",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 839,
    "class": "regime",
    "severity": "S3",
    "claim": "'f_A scales the interface-band losses consistently with the in-ODE source boost (edit site 1)'; L_eff = L1 + fA*(L2+L3); the in-ODE boost scales the net radiative source 'in the interface band ONLY', band top = the CIE switch.",
    "evidence": ":60-:61, :432-:434, :839-:844. The interface band's LOWER bound is never stated. The ODE is integrated only over [R1, r2Prime] where T >= T_init = 3e4 K (:41-:45), yet L3 is by construction the 1e4-3e4 K interpolated region outside that domain (:799-:802).",
    "expected": "Either the interface band explicitly includes the 1e4-3e4 K extrapolated region (and the docstring says so), or scaling L3 by fA has no in-ODE counterpart and the 'consistently' claim is wrong for that component.",
    "failure_scenario": "With fA != 1 the energy accounting double-applies (or mis-applies) the boost to a temperature range the ODE source term never saw, so L_eff and the beta-delta residual disagree with the structure actually integrated.",
    "repro": "pytest test/test_fA_source_boost.py; then compare L3 with fA=1 vs fA!=1 against a re-integrated structure.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-06",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 121,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "_quiet_lsoda_fortran 'suppresses only that noise: it touches no numerics, and genuine solver failures are still caught by the separate sol.success -> BubbleSolverError contract.'",
    "evidence": ":121-:134. The stated mechanism is 'redirecting the C-level stdout/stderr fds to /dev/null for the call' - an unconditional, process-wide fd redirect for the duration of the solve.",
    "expected": "The suppression should be scoped to the LSODA banner (or at minimum the docstring should say that ALL fd-1/fd-2 output during the solve is discarded, including from other libraries, warnings and any concurrent worker output).",
    "failure_scenario": "Under `--workers N` or any library that writes diagnostics to stdout/stderr, real error text emitted during a bubble solve disappears; if the redirect is not exception-safe, a raised BubbleSolverError could leave fds pointing at /dev/null for the rest of the process.",
    "repro": "Write to fd 1 from inside the ODE RHS and confirm it is (a) swallowed and (b) restored after a BubbleSolverError propagates.",
    "confidence": "high"
  },
  {
    "id": "S7-B-07",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 230,
    "class": "numerical",
    "severity": "S3",
    "claim": "The shell volume (r2**3 - r1**3) is floored so the divide stays finite; 'Bit-identical on every physical bubble (shell_volume > 0)'.",
    "evidence": ":230-:235, plus a second divide-by-zero note at :223.",
    "expected": "Bit-identity can only hold for shell_volume >= floor. Any positive shell_volume below the floor is silently replaced, so the claim should be 'bit-identical for shell_volume >= FLOOR'. Also clarify whether :223 and :230 describe one guard or two.",
    "failure_scenario": "Late in a catastrophic-cooling collapse, R1 -> R2 gives a tiny but nonzero shell volume; the floor caps Pb at a finite value instead of letting it diverge, so the phase-handoff trigger (Eb<=0) may fire at a different time than an unfloored run - not bit-identical, and possibly a different stopping fate.",
    "repro": "Evaluate bubble_E2P over a sweep of (r2-r1) spanning the floor and diff against an unfloored reference.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-08",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 71,
    "class": "coefficient",
    "severity": "S3",
    "claim": "E_b_dot = [2*pi*Pb_dot*d^2 + 3*E_b*R_b_dot*R_b^2*(1-c/(E_b+c)) - a*R_ts^3*E_b^2/(E_b+c)] / [d*(1-c/(E_b+c))], with a=(3/2)F_ram_dot/F_ram, c=(3/4)F_ram*R_ts, d=R_b^3-R_ts^3.",
    "evidence": ":71-:110 (and :415-:434, where the r1=0 endpoint is sqrt(Lmech/v/Eb*R2**3)). The leading 2*pi is exactly (3/2)*(4*pi/3), i.e. E_b = 2*pi*P_b*d, which holds only for gamma = 5/3; cool_beta_to_Ebdot takes no gamma argument, and neither does the get_r1 endpoint algebra. bubble_E2P (:199), get_leak_luminosity (:243) and get_effective_bubble_pressure (:314) all take gamma explicitly.",
    "expected": "Either gamma is fixed at 5/3 project-wide (and the gamma parameters elsewhere are decoration), or A12 / get_r1 must carry the same gamma the pressure functions use.",
    "failure_scenario": "A run with gamma != 5/3 makes the beta<->Ebdot conversion and the R1 pressure balance inconsistent with bubble_E2P, so the integrator and the diagnostics disagree about the same bubble.",
    "repro": "grep the .param schema for gamma; run with a non-5/3 gamma and compare bubble_E2P(Eb) against Eb/(2*pi*d).",
    "confidence": "medium"
  },
  {
    "id": "S7-B-09",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 934,
    "class": "other",
    "severity": "S3",
    "claim": "M(r) = integral from 0 to r of 4*pi*r'^2*rho(r') dr'.",
    "evidence": ":934, with the grid documented as spanning [R1, r2Prime] (:532-:551) and arrays flipped to increasing order first (:928).",
    "expected": "The integral's lower limit is R1 (the inner bubble radius), not 0; the docstring/comment should say M(r) = int_{R1}^{r}, or the omitted r<R1 contribution should be justified.",
    "failure_scenario": "A reader (or a future consumer of bubble_mass) treats m_cumulative as total enclosed mass including the free-wind region inside R1, under-counting or double-counting mass in a gravity or column-density calculation.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-B-10",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 939,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The commented-out gravity block 'can be restored verbatim when needed': grav_phi = -4*pi*G*simpson(r_new*rho_new, x=r_new) [pc^2/Myr^2]; grav_force_m = G*m_cumulative/(r_new**2 + 1e-10) [pc/Myr^2], with 1e-10 added 'to avoid division by zero at r=0'.",
    "evidence": ":939-:946, :916-:927 ('gravity outputs currently DISABLED', returns None placeholders 'so any future consumer ... fails loudly').",
    "expected": "Restoring verbatim would yield a single SCALAR for grav_phi (simpson over the whole array, no upper-limit dependence) labelled as a potential, and it omits the -G*M(r)/r interior term; the r=0 epsilon guards a radius that cannot occur on a grid starting at R1 > 0.",
    "failure_scenario": "Anyone re-enabling gravity gets a radius-independent 'potential' and a force law softened by an epsilon chosen for a nonexistent r=0, without any test flagging it (the block is commented out so nothing exercises it).",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-B-11",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 1053,
    "class": "state",
    "severity": "S4",
    "claim": "'(No ier key; success is read from status.)'",
    "evidence": ":1053 vs :454-:476, which states 'ok -- sol.success; when False the caller must not consume psoln'.",
    "expected": "One documented source of truth for solver success (sol.success, or infodict['status']); if the diagnostic re-derives success from 'status' it should agree with the ok flag by construction.",
    "failure_scenario": "The diagnostic classifies an event as healthy/unhealthy using a different success criterion than the one gating consumption of psoln, so captured events and real rejections drift apart.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-B-12",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 312,
    "class": "numerical",
    "severity": "S3",
    "claim": "The dMdt residual is 'v[-1]/v[0] plus min-T / monotonic checks'; a benign ~1e-4 K undershoot at the boundary gives 'penalty ~1.0'; a failed solve returns a residual that is 'large and non-zero'; _T_INIT_BOUNDARY is also 'the scale of that rejection penalty'.",
    "evidence": ":95, :312 (one-line docstring), :46, :54-:58, :81-:83, :329-:332.",
    "expected": "The residual and the penalty composition (additive / multiplicative / replacement) should be defined at the function, with an explicit scale separation between 'benign penalty ~1.0' and the 'large' failure residual, since fsolve sees only the returned number.",
    "failure_scenario": "If the benign penalty (~1.0) is the same order as a legitimate residual value, fsolve can be steered by a floating-point boundary undershoot rather than by the physical boundary condition, shifting the converged dMdt.",
    "repro": "Instrument _get_velocity_residuals and log (residual, min_T, penalty) across an fsolve on docs/dev/performance/f1edge_hidens*.param.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-13",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 60,
    "class": "regime",
    "severity": "S3",
    "claim": "'THREE places must stay in lockstep: this constant, the local _CIEswitch in _bubble_luminosity, and the cooling-table-derived nonCIE_Tcutoff in net_coolingcurve._noncie_cutoffs (they coincide on the default bundle; a table swap moves the third)'.",
    "evidence": ":60-:64 (admitted); mirrored by :41-:48 ('one constant, three roles, all of which must move together'). Only defence named is a pinning test in test/test_fA_source_boost.py.",
    "expected": "Either the three values derive from one source, or the code asserts their equality at load time rather than relying on a test pinned to the default bundle.",
    "failure_scenario": "A user swapping in a different cooling table moves nonCIE_Tcutoff away from 10**5.5; the f_A band top and _CIEswitch stay put, so the CIE/non-CIE split and the boosted band no longer coincide with the table's validity edge - i.e. the non-CIE table gets evaluated outside its documented range (:765).",
    "repro": "pytest test/test_fA_source_boost.py with a non-default cooling table bundle.",
    "confidence": "high"
  },
  {
    "id": "S7-B-14",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 314,
    "class": "regime",
    "severity": "S3",
    "claim": "get_effective_bubble_pressure: 'Energy phase: thermal pressure from hot bubble via bubble_E2P. Momentum phase: ram pressure from freely streaming wind via pRam.' Body: transition phase uses max(P_thermal, P_ram) 'to ensure smooth handoff'.",
    "evidence": ":314-:348 (docstring lists two phases; current_phase documented as \"'energy', 'momentum', etc.\") vs :353-:357 (transition branch) and :366-:367 (early-phase R1 ramp-up gated on t/tSF).",
    "expected": "The docstring should enumerate every branch it dispatches, including the transition max() and the R1 ramp-up; and max() is C0 but not C1, so 'smooth' should read 'continuous'.",
    "failure_scenario": "A caller passing an unlisted phase string, or omitting t/tSF, silently takes a different pressure branch than the docstring implies; the derivative kink at the P_thermal/P_ram crossover can also make the ODE integrator take a rejected step there.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-B-15",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 142,
    "class": "state",
    "severity": "S3",
    "claim": "Ebdot_to_cool_beta's my_params must provide 't_now, pdot_total, pdotdot_total, R2, v2, Eb (plain float values, not .value-wrapped)', while its stated inverse cool_beta_to_Ebdot requires params that 'provide .value for: Pb, cool_beta, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total'.",
    "evidence": ":71-:110 vs :142-:168.",
    "expected": "Two functions documented as exact inverses of the same equation should accept the same parameter container, or the asymmetry should be enforced (type check) rather than only documented.",
    "failure_scenario": "Passing the wrong container raises AttributeError at best; at worst a DescribedDict entry that is itself float-like is consumed as a value, silently producing a wrong beta.",
    "repro": "Round-trip test: beta -> Ebdot -> beta on the same state should be the identity.",
    "confidence": "high"
  },
  {
    "id": "S7-B-16",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 243,
    "class": "citation",
    "severity": "S4",
    "claim": "'See the leakage spec, Eq. (leak).' (get_leak_luminosity); 'See Pg 79, Eq A5, <Rahner thesis URL>' with no equation transcribed and parameter T documented as 'DESCRIPTION.' (delta2dTdt / dTdt2delta).",
    "evidence": ":243-:279; :28-:41; :48-:61.",
    "expected": "A resolvable reference (document path or paper + equation number) for 'the leakage spec, Eq. (leak)', and a written-out definition of delta / the A5 relation so the pair can be checked without the thesis.",
    "failure_scenario": "The leak term's normalisation cannot be audited against any source; delta2dTdt/dTdt2delta cannot be checked as inverses of a specific equation.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-B-17",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 415,
    "class": "citation",
    "severity": "S4",
    "claim": "Weaver+77 equation numbers, verbatim: 'Equation 33 in Weaver+77' = initial guess / estimate for dMdt (:238 and :298); 'Eq 44 in Weaver+77' = bubble-ODE initial conditions (:393); 'Equations 42-43 in Weaver+77' = the bubble-structure ODE RHS (:415).",
    "evidence": ":238, :298, :393, :415. Internally consistent (Eq 33 attached to the same quantity in both places); no other equation numbers appear.",
    "expected": "Cross-check 33 / 42 / 43 / 44 against Weaver, Castor, McCray & Moore 1977 (ApJ 218, 377) - the dMdt estimate, the interior structure ODEs and the boundary conditions must map to those exact numbers.",
    "failure_scenario": "A mis-transcribed equation number is not itself a runtime bug, but it defeats every future attempt to verify the implemented physics against the source.",
    "repro": "",
    "confidence": "low"
  },
  {
    "id": "S7-B-18",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 381,
    "class": "other",
    "severity": "S4",
    "claim": "'R1 = interface separating inner bubble radius and outer solar wind'.",
    "evidence": ":381, contradicted by :199-:217 ('r1 : Inner bubble radius (wind termination shock)') and :415 ('the inner bubble radius R1 (wind termination shock)'). The free wind is inside R1 and the shocked hot bubble outside; also 'solar wind' where the code models a stellar cluster wind.",
    "expected": "'R1 = wind termination shock separating the inner free-streaming stellar wind from the outer shocked hot bubble.'",
    "failure_scenario": "Comment-only; misleads anyone reasoning about which side of R1 carries ram vs thermal pressure (directly relevant to get_r1's pressure balance).",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-B-19",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 1106,
    "class": "numerical",
    "severity": "S4",
    "claim": "'require t_now to have grown by TRINITY_BUBBLE_STATE_DT between dumps (default 1.0 = no spacing = first-N behavior)'.",
    "evidence": ":1106-:1108.",
    "expected": "'Grown by' reads as an additive increment, under which the default 1.0 would impose 1 Myr spacing rather than 'no spacing'; only a multiplicative/ratio test (t_now >= last_t * DT) makes 1.0 a no-op. The docstring should state which.",
    "failure_scenario": "An operator setting TRINITY_BUBBLE_STATE_DT expecting Myr spacing gets a ratio (or vice versa) and captures the wrong states for the offline audit harness.",
    "repro": "TRINITY_BUBBLE_STATE_DUMP=5 TRINITY_BUBBLE_STATE_DT=2 python run.py param/simple_cluster.param and inspect the t_now of the dumps.",
    "confidence": "medium"
  },
  {
    "id": "S7-B-20",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 36,
    "class": "deadcode",
    "severity": "S4",
    "claim": "'NumPy compatibility: trapz was renamed to trapezoid in NumPy 2.0' (:36) and 'numpy 2.x: float(size-1 1-d array) errors, so coerce through .item()' (:320).",
    "evidence": ":36, :320, against CLAUDE.md's deliberate numpy<2 pin.",
    "expected": "If numpy is pinned <2, both shims are unreachable on every supported install; either the pin or the comments overstate the supported range.",
    "failure_scenario": "None at runtime; flagged because the comments assert a supported-version range that conflicts with the project's stated pin, and CLAUDE.md forbids removing the pin.",
    "repro": "",
    "confidence": "low"
  },
  {
    "id": "S7-B-21",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 243,
    "class": "regime",
    "severity": "S4",
    "claim": "coverFraction Cf is documented as 'Cf in (0, 1]'; guards return 0 for Cf >= 1, Pb <= 0, c_sound <= 0, 'so the term self-limits and never injects energy'.",
    "evidence": ":243-:281. No guard is documented for Cf <= 0, which makes (1 - Cf) > 1, i.e. an open area larger than the whole sphere.",
    "expected": "Either Cf is validated at the .param trust boundary to (0,1], or the function clamps (1-Cf) to [0,1]. 'Self-limits' currently only covers the never-injects direction.",
    "failure_scenario": "A negative or zero coverFraction from a sweep .param drains enthalpy faster than the geometric maximum, killing the bubble early with no error raised.",
    "repro": "pytest test/test_cf_leak.py with coverFraction <= 0.",
    "confidence": "low"
  },
  {
    "id": "S7-B-22",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 200,
    "class": "state",
    "severity": "S4",
    "claim": "'returns them in a BubbleProperties dataclass instead of mutating the params dict'; 'No dictionary mutations during calculation'; 'params : DescribedDict Parameter dictionary (read-only access)'; 'Use updateDict(params, bubble_data) after the call returns'.",
    "evidence": ":3-:16, :200-:214. Meanwhile :244-:245 reads a previously-stored dMdt guess ('if value already exist, use previous as current guess'), :257 wraps the residual to use a LOCAL Pb 'instead of params[Pb]', and the two gated paths (:956-:968, :1087-:1093) read params while claiming 'never mutates state'.",
    "expected": "No write to params (or any nested DescribedDict entry) on any path through get_bubbleproperties_pure, including the two env-gated diagnostics and the f_A branch.",
    "failure_scenario": "A single surviving params write makes the 'pure' contract false, reintroducing order-dependence between the bubble solve and the caller's updateDict - exactly the in-process global-state leakage CLAUDE.md rule 5 warns about.",
    "repro": "Snapshot a deep copy of params before get_bubbleproperties_pure and diff after, with TRINITY_BUBBLE_DIAG and TRINITY_BUBBLE_STATE_DUMP both set and unset.",
    "confidence": "low"
  },
  {
    "id": "S7-B-23",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 741,
    "class": "regime",
    "severity": "S3",
    "claim": "Region 1 is 'the CIE region, where T > 10**5.5 K ... CIE is used' (:686, :736), yet the region-1 body comment says 'import values from two cooling curves' (:741) - the identical comment used for the non-CIE region 2 (:780).",
    "evidence": ":686, :736, :741 vs :780 and :813 ('get cooling, taking into account for both CIE and non-CIE regimes' - region 3, where both genuinely apply); :765 states 'The non-CIE cooling table is only defined for T < 10**5.5, so the band is masked to it.'",
    "expected": "In the CIE-only region, only the CIE curve should be evaluated; if the non-CIE table is queried above 10**5.5 K it is being used outside its documented validity bound.",
    "failure_scenario": "Silent out-of-range interpolation/extrapolation of the non-CIE cooling table in the hottest part of the bubble, biasing L_bubble (the dominant luminosity term) with no error raised.",
    "repro": "Instrument the non-CIE table lookup to raise on T > 10**5.5 and run param/simple_cluster.param.",
    "confidence": "medium"
  }
]
```
