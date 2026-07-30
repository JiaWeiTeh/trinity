# S5a beta/delta solve — Lens B (what the code claims)

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

**Scope.** Prose only (docstrings + comments) from
`/tmp/claude-0/-home-user-trinity/75528b15-99c6-5b6c-980a-4aac19bbcd57/scratchpad/phase2/S5a_betadelta/prose.md`,
covering `trinity/phase1b_energy_implicit/get_betadelta.py`. I have not seen the implementation and
make no claim about whether any statement below is true. Every entry is written so another lens can
test it against source. Line numbers are original-file lines as tagged in the prose dump.

---

## 1. Formulas stated in prose

### F1 — `cool_beta_to_Ebdot_pure`, the "Rahner A12" equation (`get_betadelta.py:193`)

Stated verbatim in the docstring:

```
E_b_dot = [ 2*pi * Pb_dot * d^2
            + 3 * E_b * R_b_dot * R_b^2 * (1 - c/(E_b+c))
            - a * R_ts^3 * E_b^2 / (E_b + c) ]
          / [ d * (1 - c/(E_b+c)) ]
```

i.e.

$$\dot E_b=\frac{2\pi\,\dot P_b\,d^{2}+3E_b\dot R_b R_b^{2}\left(1-\tfrac{c}{E_b+c}\right)-a R_{ts}^{3}\dfrac{E_b^{2}}{E_b+c}}{d\left(1-\tfrac{c}{E_b+c}\right)}$$

with the three auxiliary definitions, also stated verbatim:

| symbol | stated definition | stated dimension |
|---|---|---|
| `a` | `(3/2) * F_ram_dot / F_ram` | `[1/time]` |
| `c` | `(3/4) * F_ram * R_ts` | `[energy]` |
| `d` | `R_b^3 - R_ts^3` | `[length^3]` |

Stated code↔symbol mapping (a contract for the reader, checkable against variable names):
`Pb_dot ← dP_b/dt`; `R1, R2, v2 ← R_ts, R_b, R_b_dot`; `pdot_total ← F_ram`;
`pdotdot_total ← F_ram_dot`; `a_coeff ← a`; `c_coeff ← c`; `d_coeff ← d`; `c_frac ← c/(E_b+c)`.

**Self-consistency work (prose-internal, no code seen).** The stated dimensions are consistent:
every numerator term is `energy·length³/time`, the denominator is `length³`, so
`Eb_dot` is `energy/time` — matching the stated return unit `[Msun*pc^2/Myr^3]`. Using
`1 - c/(E_b+c) = E_b/(E_b+c)` the equation collapses to

$$\dot E_b = 2\pi \dot P_b d\,\frac{E_b+c}{E_b} \;+\; \frac{3E_b\dot R_b R_b^{2}}{d} \;-\; \frac{a R_{ts}^{3}E_b}{d}$$

which is exactly $\frac{d}{dt}\left[2\pi P_b d\right]$ **iff** all three of the following hold:
(i) $P_b = E_b/(2\pi d)$, (ii) $F_{ram} = 4\pi R_{ts}^2 P_b$ (ram-pressure balance at the
termination shock), (iii) $\dot R_{ts}/R_{ts} = \tfrac12(\dot F/F - \dot P_b/P_b)$.
Condition (ii) also reproduces the stated `c` exactly: $3\pi P_b R_{ts}^3 = \tfrac34 F_{ram}R_{ts}$ ✓.
So the prose equation is internally coherent — **but** condition (i) is
$P_b=(\gamma-1)E_b/V$ with $V=(4\pi/3)d$ evaluated **only at $\gamma=5/3$**
(general: $P_b = 3(\gamma-1)E_b/(4\pi d)$, so the leading coefficient is
$4\pi/(3(\gamma-1))$, which equals $2\pi$ only when $\gamma=5/3$). See finding **S5a-B-05**,
since `compute_R1_Pb` takes `gamma_adia` as a free parameter.

### F2 — beta definition (`get_betadelta.py:193`, restated `:247`)

`beta = -(t/Pb)(dPb/dt)`  ⇒  $\dot P_b = -\beta P_b / t$.
Stated twice (docstring parameter description and the inline comment at L247); the two agree.

### F3 — delta / `delta2dTdt_pure` (`get_betadelta.py:273`)

**No formula is given.** The docstring supplies only the citation (A5), the inputs
`t [Myr]`, `T [K]` "Temperature at xi = r/R2", `delta` "Cooling parameter", and the output
`dTdt [K/Myr]`. In particular the *sign convention* of delta is never stated, while beta's is
(F2, with an explicit minus). See **S5a-B-08**.

### F4 — `compute_R1_Pb` (`get_betadelta.py:304`)

**No formula is given.** Only signature-level prose: inputs `R2 [pc]`, `Eb [Msun*pc^2/Myr^2]`,
`Lmech_total`, `v_mech_total` ("Mechanical velocity"), `gamma_adia` ("Adiabatic index");
outputs `R1 [pc]` ("Inner bubble radius"), `Pb [au] (code units)`. Note the naming drift:
`R1` is called "Inner bubble radius" here but "Termination shock radius R_ts" in F1.
Whether `Pb` here uses $V=(4\pi/3)R_2^3$ or $V=(4\pi/3)(R_2^3-R_1^3)$ is undocumented, and F1's
`d = R_b^3 - R_ts^3` requires the latter for the two functions to describe the same bubble
(**S5a-B-06**).

### F5 — `effective_Lloss` modes (`get_betadelta.py:335`)

Stated verbatim, four branches:

| mode | stated formula |
|---|---|
| `none` (default) | `Lcool + Lleak` — "default -> byte-identical" |
| `multiplier` | `Lleak + fmix * Lcool`  (stated constraint: `fmix >= 1`; "boosts only the resolved cool") |
| `theta_target` | `max(Lcool + Lleak, theta * Lmech)` |
| anything else | falls back to the resolved loss (`Lcool + Lleak`), silently — "so a typo cannot perturb a run" |

Stated design invariant: the correction is **added** to the resolved integral and is
"never a `(1-theta)*Lmech` rescale of the input on top of subtracting `Lcool` (that would remove the
same energy twice)". For `theta_target`, "the max keeps it single-count — switches OFF where the
resolved loss already exceeds target".

### F6 — energy-balance residual (`get_betadelta.py:400`, `:465`, `:477`)

`Edot_residual` = relative difference between *Edot from beta* (F1) and *Edot from energy balance*,
where the balance is stated at L465 as **`gain - loss - work`**:

$$\dot E_b^{(bal)} = L_{gain} - L_{loss,eff} - W$$

The work term `W` is never described. `L_loss,eff` is `Lcool + Lleak` passed through F5
(L468 "Add leak if available", L472 boost).
`T_residual` = "difference between bubble temperature and target temperature T0" (L400), evaluated
at a "measurement point" (L487) whose $\xi=r/R_2$ value is never stated, against a `T0` whose value
is never stated. Both residuals are labelled "Relative" in the Returns block but the normalisation
is not stated there.

### F7 — the pole-free `g` residual (`get_betadelta.py:880`)

$$g_E=\frac{\dot E_b^{(\beta)}-\dot E_b^{(bal)}}{L_{mech,total}},\qquad
f_E=\frac{\dot E_b^{(\beta)}-\dot E_b^{(bal)}}{\dot E_b^{(\beta)}}$$

Stated rationale: `Lmech_total` is "per-segment-constant", whereas the legacy `f` denominator
`Edot_from_beta` "crosses zero near the E_b peak — the pole". `gT` is stated to be "the temperature
residual, identical in f and g". So the `g` vector mixes an absolute-normalised component (`gE`)
with a relative component (`gT`) — see **S5a-B-15**.

---

## 2. Units and unit conventions claimed

| quantity | stated unit | where |
|---|---|---|
| `t`, `t_now` | Myr | `:193`, `:273` |
| `R1`/`R_ts`, `R2`/`R_b` | pc | `:193`, `:304` |
| `v2` = `R_b_dot` | pc/Myr | `:193` |
| `Eb` | `Msun*pc^2/Myr^2` (code units) | `:193`, `:304` |
| `Eb_dot` (return) | `Msun*pc^2/Myr^3` (code units) | `:193` |
| `T` | K | `:273` |
| `dTdt` (return) | K/Myr | `:273` |
| `Pb` | **`[au]` (code units)** | `:193`, `:304` |
| `pdot_total` (`F_ram`) | *none stated* | `:193` |
| `pdotdot_total` (`F_ram_dot`) | *none stated* | `:193` |
| `Lmech_total`, `v_mech_total` | *none stated* | `:304` |
| `beta`, `delta` | *none stated* (dimensionless by F2's definition) | `:193`, `:273` |
| dimension tags inside F1 | `[1/time]`, `[energy]`, `[length^3]` — generic, not the pc/Myr/Msun set | `:193` |

`[au]` is used for a **pressure** in two places. In the pc/Myr/Msun convention used by every other
quantity in the same docstring, a pressure is `Msun/(pc*Myr^2)`. "au" most commonly reads as
*astronomical unit* (a length). See **S5a-B-07**.

---

## 3. Citations recorded verbatim

| citation as written | attached to | where |
|---|---|---|
| "See pg 80, Eq A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf" | the full `E_b_dot` equation, F1 | `:193` |
| "Main equation (Rahner thesis A12)" | the same equation (inline, at the implementation site) | `:258` |
| "See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf" | `delta -> dT/dt` conversion (formula not reproduced) | `:273` |
| "Paper-II interface-cooling note" | the opt-in unresolved-interface-cooling boost as a whole | `:335` |
| "El-Badry+19, Lancaster+21, Tan/Oh/Gronke 21" | the claim that turbulent mixing layers produce interface cooling that TRINITY's resolved 1D conduction front under-counts | `:335` |
| "the Phase-2.1 transect probe (docs/dev/archive/betadelta/diagnostics)" | the numeric value of the hybr finite-difference step `eps` ("the residual noise floor … not the 1e-4 acceptance threshold") | `:70` |
| "FINDINGS §14" / "the dense-edge all-NaN arms" | the observed all-NaN `bubble_Lloss` dictionary columns caused by a repeating structure-solve failure | `:650` |
| "KAPPA_FREEZE_MECHANISM" | the no-root-streak ⇒ momentum-handoff semantics for genuine `dMdt<=0` | `:650` |
| "Phase 3, plan arm D" | the hybr solver design | `:949` |
| "matching original get_betadelta.py" (×3) | the 5×5 grid size and the grid-search parameters | `:55`, `:56`, `:1017` |

A12 and A5 are each attached to exactly one formula — **no citation is reused for two different
formulas.** The four internal references (`docs/dev/archive/betadelta/diagnostics`, `FINDINGS §14`,
`KAPPA_FREEZE_MECHANISM`, `Phase 3, plan arm D`) are load-bearing justifications that live in
unverified `docs/dev/`-class material per the project's own banner rule (**S5a-B-25**).

---

## 4. Contracts, state, ordering requirements

1. **Purity.** Module docstring (`:3`): "Pure functions that return results instead of mutating the
   params dictionary"; `get_residual_pure` (`:400`) "without mutating params";
   `_solve_betadelta_legacy` (`:684`) "params : dict-like … (not mutated)".
2. **External invariant.** `BubbleParamsView` (`:108`): "Since `get_bubbleproperties_pure()` only
   READS params (never writes), this is safe". A cross-module invariant on `bubble_luminosity`.
3. **Override surface.** `BubbleParamsView` returns overrides for `cool_beta`/`cool_delta` and, when
   `dMdt_guess` is given, for `bubble_dMdt`; everything else passes through to the original params.
   `_MockValue` (`:100`) "Mimics DescribedItem with a `.value` attribute".
4. **Cross-segment state.** `:958` — "seed=None lets `get_residual_pure` use the previous segment's
   accepted dMdt (**carried in params**); the solved dMdt then threads forward". `:108` — "Without it,
   every evaluation in a segment starts from the previous segment's accepted dMdt". So `params`
   carries mutable inter-segment state (`bubble_dMdt`) written by someone (**S5a-B-22**).
5. **Seed hygiene.** `_usable_dMdt` (`:375`): returns the solved dMdt only if the solve **succeeded**
   and the value is **finite and positive**; otherwise `None`. "must not poison the seed chain".
   `_solve_grid` (`:1017`): "Failed points leave the seed untouched"; first point seeded from
   `input_props`.
6. **`_solve_grid` caller contract** (`:1017`): the caller has **already evaluated the input guess**
   (the grid centre); `_solve_grid` skips that point and requires `input_residual`/`input_props` to
   seed best-so-far. Returns `(best_beta, best_delta, best_props, best_residual, n_evals)`.
7. **Candidate tuple** (`:725`): `(beta, delta, residual, method_name, iterations, props)`; `props`
   is `None` when not captured — "e.g. the L-BFGS-B path". `:731` "Always add original input as a
   candidate." `:821` sort by residual, pick best. `:808` if everything failed, "return original
   input with failure status".
8. **Re-solve skip** (`:520`, `:837`): passing the winning candidate's already-solved
   `bubble_props` into `get_residual_detailed` skips the bubble-structure solve; justified by
   "get_bubbleproperties_pure is deterministic in (params, beta, delta), this is equivalent to
   re-solving" (**S5a-B-04**).
9. **Runner handoff** (`:170`, `:933`, `:949`): the `no_physical_root` flag is the signal the runner
   uses to end the implicit phase and hand off to the transition phase.
   "**legacy never sets it**" (`:170`) (**S5a-B-14**).
10. **Rescue routing contract** (`:650`): "the caller only routes 'structure solve failed' reasons
    here" — a condensation root (`'non-physical dMdt=…'`) "is real physics and must NOT be retried
    away". Discrimination is by **reason string** (**S5a-B-16**). `:669` on a failed rescue "keep the
    original diagnosis (incl. its reason string)".
11. **Single point of application** (`:335`, `:361`): `effective_Lloss` is "the single point where
    the opt-in unresolved-interface-cooling boost is applied", fed **consistently** to (a) the
    beta-delta residual, (b) the energy ODE `Edot_from_balance`, (c) the energy→momentum trigger —
    "the three call sites stay one line and identical" (**S5a-B-23**).
12. **Solver dispatch** (`:613`, `:631`, `:645`): key `betadelta_solver`; production default `'hybr'`;
    params predating the key "fall back to the legacy path"; an unknown value is rejected — "The
    param validator guards user input; this guards programmatic misuse."
13. **Ignored parameter** (`:949`): `_solve_betadelta_hybr`'s `method` "is accepted for signature
    parity with the legacy solver and ignored".
14. **Exception type** (`:870`): `_NoPhysicalRoot` is "A BaseException, not Exception, so
    `get_residual_pure`'s `except Exception` plateau handler cannot swallow it while it propagates out
    through the scipy root-finder's internals." This documents, by side effect, that
    `get_residual_pure` contains a broad `except Exception` that converts failures into a **plateau
    penalty value** (**S5a-B-18**).
15. **Result semantics vary by solver** (`:910`): "The f-metric components stay in the result for
    output continuity; g drives acceptance (`total_residual` and `converged` are g quantities under
    this solver)" (**S5a-B-15**).
16. **Diagnostics fields** (`:162`–`:169`, `:505`–`:511`): `residual_Edot1_guess`,
    `residual_Edot2_guess`, `residual_T1_guess`, `residual_T2_guess`, `bubble_Lgain`, `bubble_Lloss`
    — carried on both `BetaDeltaResult` and `ResidualDetails`, "for saving to dictionary" (`:858`).
17. **Log formatting** (`:78`): `_describe_exc` formats as `'ClassName: message at file:line'` using
    the **deepest** traceback frame; motivated by scipy/numpy internals raising with empty messages
    that made the warning print as `"failed: "`.

---

## 5. Numerical claims (thresholds, counts, convergence, fallbacks)

| # | claim | where |
|---|---|---|
| N1 | Bounds exist for beta and delta (values not in prose) | `:40` |
| N2 | "Convergence thresholds" (plural; `RESIDUAL_THRESHOLD` named at `:59`/`:1017`) | `:46` |
| N3 | L-BFGS-B runs **only** if grid residual exceeds a threshold; "If grid gives a reasonable result (**< 5.0**), L-BFGS-B is unlikely to improve much and wastes **~50** expensive function evaluations" | `:50`–`:52`, `:765` |
| N4 | Grid is **5×5** around the guess; `GRID_EPSILON` is the "Search range around guess" | `:55`–`:57`, `:1017` |
| N5 | `GRID_EARLY_EXIT_RESIDUAL` is a margin strictly **inside** the acceptance threshold; scan stops on the first point below it | `:59`, `:1017` |
| N6 | "residuals at the accepted point grow by **roughly 3x per subsequent segment** as beta/delta drift" | `:61` |
| N7 | A deeply converged pick keeps the next segments' input guesses below `RESIDUAL_THRESHOLD` ⇒ "long runs of 1-evaluation short-circuits"; a barely-converged pick "forces a fresh grid search almost immediately" | `:62`–`:64` |
| N8 | Segments with no excellent point evaluate the **full grid** and return the global best, "identical to the original semantics" | `:66`, `:1017` |
| N9 | hybr `eps` = "the residual noise floor measured in the Phase-2.1 transect probe … **not the 1e-4 acceptance threshold**" (so the acceptance threshold is 1e-4) | `:70`–`:71` |
| N10 | hybr `factor=0.1` "keeps Newton steps local so the root-finder does not leap into ODE-failing (beta, delta)"; `maxfev` "caps cost" (value not stated) | `:71`–`:73` |
| N11 | `BubbleParamsView` gives "~**25-100x** speedup per residual evaluation" / "25-100x faster per evaluation" | `:3`, `:108` |
| N12 | Grid scan is **center-out in index space**, "so the ordering stays valid when the linspace is clamped at the parameter bounds"; "Ties broken by (i, j) for determinism" | `:1017`, `:1051`–`:1054` |
| N13 | Centre skip uses a **tolerance** compare, "because linspace's midpoint can differ from the guess in the last ulp"; for a guess near a clamped bound "the shifted grid no longer contains the guess, no point matches, and the full grid runs" | `:1071`–`:1075` |
| N14 | Warm-start justification: "adjacent grid points differ by **at most GRID_EPSILON** in beta/delta, so their dMdt roots are close" | `:1017` |
| N15 | Early-exit safety: "No earlier point was below the margin (the scan would have exited there), so this point is also the current best" | `:1093`–`:1095` |
| N16 | Acceptance gate under hybr: `dMdt > 0` **and** valid bubble structure; plus "the standard g threshold" (value/form not stated) | `:650`, `:880`, `:949`, `:967` |
| N17 | Post-solve re-gating: "Apply the gate to the point actually accepted: reuse the last evaluation if it is that point, otherwise re-evaluate (and re-gate) it" | `:990`–`:991` |
| N18 | Legacy short-circuit: "First check if current guess is already good enough"; hybr mirrors it — "Guess already satisfies the gate and the g threshold: short-circuit" | `:706`, `:967` |
| N19 | Failure handling in the grid: structure failures "score a **plateau value** instead of aborting" (value not stated) | `:650` |
| N20 | Rescue is inert on healthy runs: "Engages only on an already-failed segment: the healthy path is **byte-identical**" | `:650` |
| N21 | `'none'` cooling-boost mode is "default -> **byte-identical**" | `:335`, `:472`, `:576` |
| N22 | `'legacy'` solver is "**byte-identical** to the pre-switch behaviour" | `:631` |

---

## 6. Regimes, validity limits, assumptions

- **A12 applicability.** Nothing in the prose limits F1's validity, yet the `2π` coefficient encodes
  $\gamma=5/3$ and the `a`/`c` definitions encode ram-pressure balance at the termination shock
  ($F_{ram}=4\pi R_{ts}^2P_b$) and $\dot R_{ts}/R_{ts}=\tfrac12(\dot F/F-\dot P_b/P_b)$. These are
  unstated assumptions (§1, F1).
- **Pole regime.** `f_E` "crosses zero near the E_b peak" (`:880`) — the legacy metric is explicitly
  documented as singular in a regime the run passes through.
- **End of regime.** The `dMdt>0` gate rejecting everything means "the energy-driven implicit regime
  has ended" (`:170`); a genuine `dMdt<=0` root is "real physics" — condensation (`:650`).
- **hybr is unbounded** (`:631`, `:949`) while legacy is "bounded" (`:631`, `:650`) with a grid
  clamped at parameter bounds (`:1017`). The production default is therefore the unbounded one
  (**S5a-B-13**).
- **`fmix >= 1`** stated as a constraint on the `multiplier` mode (`:335`); no enforcement described.
- **`theta_target` "switches OFF where the resolved loss already exceeds target"** (`:335`) — a
  regime-dependent, non-smooth (max) term inside the residual that the root-finder differentiates
  across via finite differences (`eps`, `:70`).
- **Settled-regime assumption** behind the center-out scan: "in the settled regime the optimum …
  lies near the previous accepted point, i.e. the grid center" (`:1051`).

---

## 7. Admissions of known debt / defect (verbatim triggers)

No `TODO`, `FIXME`, `XXX` or `HACK` tokens appear anywhere in this slice's prose. The admissions are
phrased as prose:

- **A known output defect, quoted** (`:650`): "because the warm-start guesses only update on
  success, the failure can repeat every segment — **the run then never writes `bubble_Lloss` and every
  dictionary row carries its NaN default** (the dense-edge all-NaN arms, FINDINGS §14)." A shipped,
  observed data-loss mode; `_rescue_structure_failure` is the mitigation, not a root-cause fix.
- **A documented silent config failure** (`:335`): "Any unrecognised `mode` falls back to the
  resolved loss, **so a typo cannot perturb a run**" — framed as a feature.
- **A documented broad-except plateau handler** (`:870`): `get_residual_pure` has an
  `except Exception` "plateau handler" that swallows arbitrary failures into a penalty score.
- **A documented dead parameter** (`:949`): hybr's `method` "is accepted for signature parity … and
  ignored".
- **Prose that admits non-equivalence in the fallback direction only** (`:66`, `:1017`): the full-grid
  path is "identical to the original semantics" — by construction the early-exit path is not.
- **Robustness workaround for missing schema key** (`:613`): "Robust to params that predate the
  `betadelta_solver` key (e.g. the unit-test fixtures), which fall back to the legacy path."
- **Log-quality admission** (`:78`): failures previously printed as `"failed: "` with "no clue about
  what or where".
- **Stale module header** (`:3`): the "Key design choices" list still describes grid-first as the
  default (see **S5a-B-01**).

---

## 8. Flags — contradictions, vagueness, unit/formula tension

### 8.1 Prose contradicting prose in the same slice

- **Default solver.** `:3` "Grid search first (**default**), then L-BFGS-B fallback" vs `:613`/`:631`
  "production default `'hybr'`". → **S5a-B-01**.
- **L-BFGS-B trigger.** `:684` "automatically falls back to `'lbfgsb'` **if grid search fails or
  doesn't converge**" vs `:50`/`:765` "only run L-BFGS-B if grid residual exceeds this … **AND** grid
  residual is bad". Non-convergence alone does not trigger it. → **S5a-B-02**.
- **Determinism vs warm start.** `:520` "get_bubbleproperties_pure is **deterministic in (params,
  beta, delta)**, this is equivalent to re-solving" vs `:108`/`:1017`, where the fsolve seed
  (`bubble_dMdt`) is a *third* input that varies between the stored solve and any re-solve.
  → **S5a-B-04**.
- **Two defaults for one key.** `:631` "production default" is `hybr`, `:613` missing key falls back
  to `legacy`. The default depends on whether the key exists. → **S5a-B-12**.
- **Error philosophy.** `:645` unknown `betadelta_solver` → rejected as programmatic misuse;
  `:335` unknown `cooling_boost` mode → silently ignored. Same file, opposite policy. → **S5a-B-11**.
- **"Bounded" legacy vs always-included input candidate.** `:650` "The legacy grid search is bounded
  and penalty-guarded, **so its optimum is a domain-respecting seed**" vs `:731` "Always add original
  input as a candidate" — in the rescue path that input is precisely the out-of-domain point hybr
  wandered to. → **S5a-B-17**.
- **Two baselines for "identical".** `:631` legacy is "byte-identical to **the pre-switch
  behaviour**"; `:55`/`:1017` the grid matches "**original get_betadelta.py**"; `:66` only the
  no-early-exit path is "identical to the original semantics". Three different reference points for
  one equivalence word. → **S5a-B-20**.
- **`R1` naming.** `:193` "Termination shock radius R_ts"; `:304` "Inner bubble radius". → folded
  into **S5a-B-06**.
- **"Each evaluated point's residual and bubble properties are kept (only the current best is
  held)"** (`:1017`) — the parenthetical negates the clause; readable but self-contradictory as
  written. (S4, not itemised below.)

### 8.2 Claims too vague to check as written

- The **`g` threshold** — named at `:650`, `:967`, `:880` but its value and whether it applies to a
  norm or per-component is never stated (**S5a-B-15**).
- hybr **`eps`** — defined only as "the residual noise floor measured in the Phase-2.1 transect
  probe"; no number (**S5a-B-25**). `maxfev` value likewise absent.
- **`T0` / target temperature** and the **measurement point `xi`** — never given a value
  (**S5a-B-09**).
- The **work term** in `gain - loss - work` (`:465`) — never defined (**S5a-B-19**).
- The **plateau penalty value** (`:650`, `:870`) — never given (**S5a-B-18**).
- **"~25-100x"** (`:3`, `:108`) and **"roughly 3x per subsequent segment"** (`:61`) — quantitative
  empirical claims with no cited measurement or committed diagnostic (**S5a-B-24**).
- `_solve_lbfgsb`'s entire docstring is "L-BFGS-B optimizer solver." (`:1113`) — no objective,
  bounds, tolerance or evaluation budget, though `:52` asserts a "~50 evaluations" cost for it.

### 8.3 Stated unit inconsistent with stated formula

- `Pb : Bubble pressure [au] (code units)` (`:193`, `:304`) against a docstring whose other units are
  pc / Myr / Msun and whose F1 requires `Pb_dot * d^2 / d` to carry energy/time — i.e. `Pb` must be
  `Msun/(pc*Myr^2)`, not any reading of "au". → **S5a-B-07**.

---

## 9. Candidate findings

```json
[
  {
    "id": "S5a-B-01",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "Module docstring 'Key design choices' states the solver is: '3. Grid search first (default), then L-BFGS-B fallback if grid doesn't converge 4. If both fail to converge, picks the best result from grid/L-BFGS-B/original input'.",
    "evidence": "L3-18 module docstring describes grid-first as the default; L613-617 _get_betadelta_solver says 'The configured beta-delta solver (production default 'hybr')' and L631-636 says \"'hybr' (production default) is the unbounded scipy root-finder\", with 'legacy' being the grid+L-BFGS-B path.",
    "expected": "The module header should describe the hybr default and relegate grid/L-BFGS-B to the legacy path, or state that the design-choices list describes 'legacy' only.",
    "failure_scenario": "A reader (or future session) tunes GRID_*/LBFGSB_THRESHOLD believing they affect production runs, when production dispatches to hybr and never touches them.",
    "repro": "Check the schema default for 'betadelta_solver' in trinity/_input/ and the dispatch in solve_betadelta_pure; confirm the grid constants are unreachable when betadelta_solver='hybr'.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-02",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 684,
    "class": "other",
    "severity": "S3",
    "claim": "_solve_betadelta_legacy docstring: \"When method='grid', automatically falls back to 'lbfgsb' if grid search fails or doesn't converge.\"",
    "evidence": "Contradicted in the same file by L50-52 ('Threshold for L-BFGS-B fallback: only run L-BFGS-B if grid residual exceeds this. If grid gives a reasonable result (< 5.0), L-BFGS-B is unlikely to improve much') and L765-766 ('Step 2: If grid didn't converge AND grid residual is bad, try L-BFGS-B').",
    "expected": "Docstring should state the conjunction: fallback runs only when the grid result is both unconverged AND its residual exceeds LBFGSB_THRESHOLD.",
    "failure_scenario": "A caller relying on the docstring assumes an unconverged grid result was always refined by L-BFGS-B; in fact residuals in (RESIDUAL_THRESHOLD, 5.0] are returned unrefined.",
    "repro": "Read the Step-2 condition in _solve_betadelta_legacy and compare with the docstring; construct a segment whose grid residual lands in (1e-4, 5.0) and confirm n_evals shows no L-BFGS-B run.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-03",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 50,
    "class": "numerical",
    "severity": "S2",
    "claim": "LBFGSB_THRESHOLD is 5.0 and a grid residual below it is 'a reasonable result' not worth refining; separately L70-71 states the acceptance threshold is 1e-4.",
    "evidence": "L50-52 'only run L-BFGS-B if grid residual exceeds this / If grid gives a reasonable result (< 5.0), L-BFGS-B is unlikely to improve much and wastes ~50 expensive function evaluations'; L70-71 'not the 1e-4 acceptance threshold'.",
    "expected": "Either the refinement gate is far tighter than 5.0, or the prose should justify why a residual up to ~5e4x the acceptance threshold is 'reasonable' and accepted unrefined.",
    "failure_scenario": "A stiff segment converges the grid to residual ~1 (relative Edot mismatch of order unity), no refinement runs, the point is returned with converged=False, and if the runner does not hard-check `converged` the trajectory advances on a beta/delta that does not satisfy the energy balance.",
    "repro": "Verify the constant's value and the Step-2 comparison operator; then check whether any caller of solve_betadelta_pure/get_beta_delta_wrapper_pure branches on result.converged.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-04",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 520,
    "class": "numerical",
    "severity": "S2",
    "claim": "get_residual_detailed: 'Since get_bubbleproperties_pure is deterministic in (params, beta, delta), this is equivalent to re-solving' — used to justify skipping the bubble-structure solve for the winning candidate.",
    "evidence": "Contradicted by L108-123 (BubbleParamsView's dMdt_guess 'additionally overrides bubble_dMdt — the seed from which get_bubbleproperties_pure starts its fsolve for the mass flux') and L1017-1041 ('The dMdt solved at each successful point warm-starts the next point's fsolve'). The stored props were solved with a threaded per-point seed; a re-solve would use whatever seed params carries. The result is therefore a function of (params, beta, delta, dMdt_seed).",
    "expected": "Either state the equivalence as 'equal to fsolve tolerance, not bit-identical', or make the determinism true by pinning the seed used for the stored props.",
    "failure_scenario": "Reported diagnostics (residual_Edot*/residual_T*/bubble_Lgain/bubble_Lloss written to the dictionary) differ at fsolve-tolerance level from what a re-solve would give, breaking the project's bit-identical equivalence gate for any change that alters seeding order.",
    "repro": "Solve one point twice with different dMdt seeds at identical (params, beta, delta) and diff the returned BubbleProperties; then diff dictionary.jsonl for a run with and without the props-reuse shortcut.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-05",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 193,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The A12 equation is stated with a hard 2*pi coefficient on the Pb_dot term: 'E_b_dot = [ 2*pi * Pb_dot * d^2 + ... ] / [ d * (1 - c/(E_b+c)) ]', with no stated dependence on the adiabatic index.",
    "evidence": "The stated equation reduces to d/dt[2*pi*Pb*d], i.e. it assumes Pb = E_b/(2*pi*d). Generally Pb = 3(gamma-1)E_b/(4*pi*d), so the coefficient is 4*pi/(3(gamma-1)) and equals 2*pi only at gamma=5/3. Meanwhile compute_R1_Pb (L304-326) takes 'gamma_adia : float, Adiabatic index' as an explicit free parameter.",
    "expected": "Either the coefficient is derived from gamma_adia, or the docstring states that A12 is valid only for gamma=5/3 and the code asserts it.",
    "failure_scenario": "A .param setting gamma_adia != 5/3 silently makes Edot_from_beta inconsistent with the Pb used everywhere else, so the beta residual is minimised against a wrong target and the energy trajectory is wrong with no warning.",
    "repro": "Grep for gamma_adia consumers; check whether the 2*pi literal in cool_beta_to_Ebdot_pure is derived or hardcoded, and whether gamma_adia is user-settable in trinity/_input/ schema defaults.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-06",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 304,
    "class": "other",
    "severity": "S2",
    "claim": "compute_R1_Pb documents no formula; R1 is called 'Inner bubble radius' here but 'Termination shock radius R_ts' at L193; the volume convention behind Pb is never stated.",
    "evidence": "L193-246 defines d = R_b^3 - R_ts^3 and (by the reduction in the report) requires Pb = E_b/(2*pi*(R_b^3 - R_ts^3)). L304-326 gives only 'Pb : Bubble pressure [au] (code units)' with inputs R2, Eb, Lmech_total, v_mech_total, gamma_adia — no R1-subtraction is documented, and Pb depends on R1 which the same call is computing.",
    "expected": "Both functions must use the same bubble volume. If compute_R1_Pb uses V = (4/3)pi*R2^3 while A12 uses R_b^3 - R_ts^3, the two Pb definitions disagree by a factor (1 - (R1/R2)^3).",
    "failure_scenario": "Early in the evolution R1/R2 is not small; the Edot residual then compares an Edot built from one Pb convention against an energy balance built from another, biasing the accepted beta.",
    "repro": "Read compute_R1_Pb and check the denominator of Pb (R2^3 vs R2^3 - R1^3) and whether R1 and Pb are solved self-consistently (both depend on each other).",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-07",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 193,
    "class": "units",
    "severity": "S4",
    "claim": "'Pb : float — Bubble pressure [au] (code units)' (repeated at L304-326).",
    "evidence": "Every other unit in the same docstring is explicit and in the pc/Myr/Msun set (R [pc], v2 [pc/Myr], Eb [Msun*pc^2/Myr^2], Eb_dot [Msun*pc^2/Myr^3], t [Myr]). The stated equation requires Pb to be an energy density, i.e. Msun/(pc*Myr^2). '[au]' most naturally reads as astronomical unit, a length.",
    "expected": "Write the actual code unit, e.g. 'Pb : bubble pressure [Msun/(pc*Myr^2)] (code units)'.",
    "failure_scenario": "A future edit converts Pb 'to/from au' or compares it against an astropy quantity in AU, silently rescaling the whole beta residual.",
    "repro": "Check trinity/_functions/unit_conversions.py for the pressure code unit and reconcile with the docstring tag.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-08",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 273,
    "class": "sign",
    "severity": "S3",
    "claim": "delta2dTdt_pure documents only 'Convert delta to dT/dt ... See Pg 79, Eq A5' plus t [Myr], T [K], 'delta : Cooling parameter', returns 'dTdt [K/Myr]'. No formula and no sign convention are given.",
    "evidence": "By contrast beta's convention is stated explicitly and twice, with the sign: 'beta = -(t/Pb)(dPb/dt)' (L193 parameter block and L247 comment). Nothing states whether delta = (t/T)(dT/dt) or -(t/T)(dT/dt).",
    "expected": "State the delta convention inline, as beta's is, so the implemented dT/dt = +/- delta*T/t can be checked against Rahner A5 without opening the thesis.",
    "failure_scenario": "A sign flip in the delta convention makes the T residual chase the wrong branch; because the T residual is the only constraint on delta, this is silent and only visible as a wrong bubble temperature profile.",
    "repro": "Read the one-line body of delta2dTdt_pure and compare with Rahner thesis pg 79 Eq A5; confirm sign consistency with wherever delta is initialised/guessed.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-09",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 400,
    "class": "other",
    "severity": "S4",
    "claim": "'T_residual: difference between bubble temperature and target temperature T0'; L487 'Temperature at measurement point vs target temperature'; L273 'T : Temperature at xi = r/R2'.",
    "evidence": "Neither the numeric value of T0 nor the value of the measurement point xi is stated anywhere in the slice, though both are load-bearing physical choices (in Weaver-type solvers xi is conventionally near 0.9 and T0 near 3e4 K).",
    "expected": "Document the xi and T0 actually used, and where they come from (param key vs hardcoded).",
    "failure_scenario": "The measurement point silently differs from the one the cooling table / T0 was calibrated for, biasing delta.",
    "repro": "Find the xi and T0 sources in get_residual_pure and check they trace to a .param key rather than a literal.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-10",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 335,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "effective_Lloss: 'Any unrecognised `mode` falls back to the resolved loss, so a typo cannot perturb a run.' Reinforced at L357: \"'none' (default) and any unrecognised token -> resolved loss\".",
    "evidence": "The stated behaviour is a silent no-op for an invalid cooling_boost_mode, with no warning documented. The same file takes the opposite stance for the solver key at L645 ('The param validator guards user input; this guards programmatic misuse').",
    "expected": "Either the param validator rejects unknown cooling_boost_mode values at load (in which case the docstring should say so), or the fallback should log a warning.",
    "failure_scenario": "A user sets cooling_boost_mode='mutliplier' (typo) with fmix=3; the run completes normally with no boost applied and no warning, and the published result is attributed to a boosted run.",
    "repro": "Check trinity/_input/ schema for a cooling_boost_mode enum/validator; if absent, run with a deliberately misspelled mode and confirm no warning is emitted.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-11",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 645,
    "class": "other",
    "severity": "S4",
    "claim": "'The param validator guards user input; this guards programmatic misuse' — an unknown betadelta_solver is an error.",
    "evidence": "Directly opposite to L335-352's documented policy for an unknown cooling_boost mode ('a typo cannot perturb a run'). Two config keys read in the same module, two opposite invalid-value policies.",
    "expected": "One policy for unknown enum values in this module (preferably: validate at load, raise on unknown).",
    "failure_scenario": "Inconsistent operator expectations; a typo in one key aborts the run, a typo in the other silently changes the physics.",
    "repro": "Compare the two branches and the corresponding validator entries in trinity/_input/.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-12",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 613,
    "class": "state",
    "severity": "S2",
    "claim": "_get_betadelta_solver: 'The configured beta-delta solver (production default 'hybr'). Robust to params that predate the `betadelta_solver` key (e.g. the unit-test fixtures), which fall back to the legacy path.'",
    "evidence": "Two different defaults are documented for one key: 'hybr' when the key is present (production .param/schema) and 'legacy' when it is absent (unit-test fixtures). The docstring names the unit-test fixtures explicitly as key-less.",
    "expected": "Either the fixtures are updated so tests exercise the production solver, or the test suite explicitly covers both arms.",
    "failure_scenario": "The production code path (hybr: unbounded root-find, _NoPhysicalRoot BaseException, g residual, no_physical_root handoff, _rescue_structure_failure) is not exercised by the unit tests that use those fixtures; regressions land green.",
    "repro": "Grep test/ for 'betadelta_solver'; count tests that reach _solve_betadelta_hybr vs _solve_betadelta_legacy.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-13",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 949,
    "class": "regime",
    "severity": "S2",
    "claim": "_solve_betadelta_hybr is an 'Unbounded scipy hybr root-finder', while L40 declares 'Bounds for beta and delta' and L631/L650 describe legacy as 'bounded'.",
    "evidence": "L631-636 \"'hybr' (production default) is the unbounded scipy root-finder\"; L949-957 'Unbounded scipy hybr root-finder ... gated on physical acceptance (dMdt > 0, valid structure)'. The only stated restraints under hybr are the acceptance gate and factor=0.1 ('keeps Newton steps local so the root-finder does not leap into ODE-failing (beta, delta)', L71-73).",
    "expected": "If BETA/DELTA bounds encode physical validity (e.g. of the Weaver similarity solution), the production solver should respect them or the bounds should be documented as legacy-only heuristics.",
    "failure_scenario": "hybr accepts a (beta, delta) outside the declared bounds that nonetheless passes the dMdt>0 gate; the accepted point is unphysical but indistinguishable in the output, and it seeds the next segment's guess.",
    "repro": "Check whether BETA_MIN/MAX and DELTA_MIN/MAX are referenced anywhere in the hybr path; log accepted (beta, delta) across a full run of param/simple_cluster.param plus the f1edge configs and check for excursions.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-14",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 170,
    "class": "regime",
    "severity": "S2",
    "claim": "'hybr no-physical-root signal: set when the dMdt>0 / valid-structure gate rejects every (beta, delta) the root-finder reaches (the energy-driven implicit regime has ended). The runner uses this to hand off to the transition phase; legacy never sets it.'",
    "evidence": "L170-173 plus L933 ('the runner hands off on the flag') and L949-957. The end-of-implicit-regime detection therefore exists only under betadelta_solver='hybr'.",
    "expected": "Documented consequence: running with betadelta_solver='legacy' has no physical end-of-regime detection and must terminate Phase 1b by some other criterion — that criterion should be named.",
    "failure_scenario": "A comparison/reproduction run set to 'legacy' (or one using key-less params per S5a-B-12) continues integrating the energy-driven implicit phase past the point where no physical root exists, converging on penalty-plateau or bounds-pinned beta/delta instead of handing off to the transition phase.",
    "repro": "Run the same stiff config under both solver settings to the same simulation time and compare the phase-transition time recorded in the output.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-15",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 910,
    "class": "numerical",
    "severity": "S2",
    "claim": "_hybr_result: 'The f-metric components stay in the result for output continuity; g drives acceptance (`total_residual` and `converged` are g quantities under this solver).' Combined with L880-888: gE = (Edot mismatch)/Lmech_total while gT is 'the temperature residual, identical in f and g'.",
    "evidence": "total_residual therefore means a different quantity under 'hybr' (g) than under 'legacy' (f), while the per-component f diagnostics written to the dictionary keep the old meaning. Additionally the g vector mixes an Lmech-normalised component with a relative temperature component, and the threshold applied to it ('the standard g threshold', L650/L967) is never stated as a norm or per-component bound, nor is its value given.",
    "expected": "State the g threshold value and form, and either rescale gT to match gE's normalisation or document why a mixed-scale residual vector is acceptable for hybr's convergence test.",
    "failure_scenario": "(a) Any analysis or regression test comparing total_residual across solver settings compares f against g. (b) If the acceptance threshold is the same 1e-4 constant used for f, then under g the Edot arm is tested as |dE| <= 1e-4 * Lmech — a completely different physical tightness that varies with feedback strength across the sweep.",
    "repro": "Check whether RESIDUAL_THRESHOLD is reused for the g test; evaluate |gE| and |gT| magnitudes at accepted points across param/simple_cluster.param and the f1edge_{lowdens,hidens} configs and see whether one arm dominates the acceptance decision.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-16",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 650,
    "class": "state",
    "severity": "S2",
    "claim": "_rescue_structure_failure must be reached only for 'structure solve failed' no-roots; 'A found *condensation* root (\"non-physical dMdt=…\") is real physics and must NOT be retried away — the caller only routes \"structure solve failed\" reasons here'.",
    "evidence": "L880-888 states _hybr_g_residual 'Raises `_NoPhysicalRoot` if the structure solve fails **or** the resulting dMdt is non-finite / <= 0' — one exception type for both causes. The physically decisive distinction is therefore carried only in a human-readable reason string, matched by the caller.",
    "expected": "Carry the cause as a typed field/enum on the exception or result rather than a message substring, or pin the exact strings with a test.",
    "failure_scenario": "A reworded message (or an equality/substring mismatch, e.g. a message that includes both phrases) routes a genuine condensation root into the grid rescue, retrying away the physics that should trigger the momentum handoff (KAPPA_FREEZE_MECHANISM), or conversely leaves a rescuable wandering failure unrescued and re-creates the all-NaN bubble_Lloss columns.",
    "repro": "Find every construction site of the two reason strings and the caller's comparison; add a test asserting each cause routes to the intended branch.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-17",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 650,
    "class": "regime",
    "severity": "S2",
    "claim": "'The legacy grid search is bounded and penalty-guarded (structure failures score a plateau value instead of aborting), so its optimum is a domain-respecting seed.'",
    "evidence": "Contradicted by L731 'Always add original input as a candidate' and L821 'Sort by residual and pick best' — the legacy solver's returned optimum can be the input guess itself, which in the rescue path is exactly the out-of-domain (beta, delta) hybr wandered into. Also L1017 notes the grid can be clamped at bounds, but says nothing about clamping the input candidate.",
    "expected": "In the rescue path the seed should be restricted to grid points (in-bounds), or the docstring should state that the rescue may hand back the failing point unchanged (which L666 'grid found nothing better than the failing seed' partly implies).",
    "failure_scenario": "The rescue returns the same out-of-domain point it was meant to escape, hybr restarts from it, and the failure repeats every segment — the exact loop the rescue was written to break (all-NaN bubble_Lloss, FINDINGS §14).",
    "repro": "Force a structure failure at a known out-of-bounds (beta, delta) and assert the rescued seed lies inside the declared bounds.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-18",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 870,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "get_residual_pure contains an 'except Exception plateau handler' (documented only obliquely, as the reason _NoPhysicalRoot must be a BaseException); L650 confirms 'structure failures score a plateau value instead of aborting'. The plateau value is never stated.",
    "evidence": "L870-876 '_NoPhysicalRoot ... A BaseException, not Exception, so `get_residual_pure`'s `except Exception` plateau handler cannot swallow it'; L650-663 'bounded and penalty-guarded (structure failures score a plateau value instead of aborting)'.",
    "expected": "The plateau must be strictly larger than any attainable genuine residual (in particular > LBFGSB_THRESHOLD = 5.0 and > any real f/g value), and the handler should log which exception it swallowed (cf. _describe_exc at L78-84).",
    "failure_scenario": "If the plateau value is finite and modest, a segment where every grid point fails returns a plateau-valued 'best' that sorts below a genuinely large real residual, so a failed point wins the candidate sort and is reported as the solution; if the handler is silent, the failure never appears in the log.",
    "repro": "Read the except block: check the plateau constant's value against LBFGSB_THRESHOLD/RESIDUAL_THRESHOLD, and whether it logs via _describe_exc.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-19",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 465,
    "class": "other",
    "severity": "S4",
    "claim": "'Method 2: Edot from energy balance (gain - loss - work)' — the balance used as the second arm of the Edot residual.",
    "evidence": "The 'work' term is never defined anywhere in the slice, although 'gain' and 'loss' are traced in detail (L468 leak, L472 cooling boost, L510-511 bubble_Lgain/bubble_Lloss). Meanwhile the A12 arm (L193-246) already encodes expansion work implicitly through the Pb_dot and R_b_dot terms.",
    "expected": "Name the work term (presumably 4*pi*R2^2*v2*Pb) and confirm the two arms do not double-count or omit it inconsistently — that is precisely what the residual is asserting to be equal.",
    "failure_scenario": "A mismatched work convention between the A12 arm and the balance arm biases the accepted beta systematically rather than randomly, which a residual-based check cannot detect (the residual is driven to zero either way).",
    "repro": "Read the balance expression in get_residual_pure and the same expression in get_residual_detailed (L566-581) and confirm they are identical; then confirm the sign/factor of the work term against the A12 reduction.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-20",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 631,
    "class": "other",
    "severity": "S4",
    "claim": "\"'legacy' is the bounded grid + L-BFGS-B search, byte-identical to the pre-switch behaviour.\"",
    "evidence": "Three different equivalence baselines are used in the file for overlapping claims: 'byte-identical to the pre-switch behaviour' (L631), 'matching original get_betadelta.py' (L55-56, L1017), and 'identical to the original semantics' applied only to the no-early-exit grid path (L66-67, L1017). The grid early-exit (L59-67) by construction can return a different point than the original full-grid global best.",
    "expected": "One named baseline per equivalence claim (a commit, or 'original get_betadelta.py'), and an explicit statement that the early-exit path is behaviourally equivalent but not bit-identical to full-grid best-of.",
    "failure_scenario": "A future session takes 'byte-identical' at face value and skips the project's full-run equivalence gate when touching the legacy path.",
    "repro": "Construct a segment with two grid points below RESIDUAL_THRESHOLD but only one below GRID_EARLY_EXIT_RESIDUAL, where the global best is not the early-exit point, and diff the selected (beta, delta) against a full-grid scan.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-21",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1017,
    "class": "numerical",
    "severity": "S4",
    "claim": "'The dMdt solved at each successful point warm-starts the next point's fsolve (adjacent grid points differ by at most GRID_EPSILON in beta/delta, so their dMdt roots are close)'.",
    "evidence": "The justification is stated in terms of grid adjacency, but the same docstring states the scan order is center-out in index space with ties broken by (i, j) — consecutive *evaluated* points on a 5x5 center-out ordering need not be grid-adjacent (points on the same ring sit on opposite sides of the centre; ring-to-ring transitions jump further, up to the full grid diagonal).",
    "expected": "Either justify the seed hand-off in terms of the actual scan order, or hand off the seed from the nearest already-solved point rather than the previous scanned one.",
    "failure_scenario": "A seed carried across a large jump in (beta, delta) starts fsolve far from its root in a stiff segment, producing a slower or differently-converged dMdt and hence a scan-order-dependent grid result.",
    "repro": "Print the (beta, delta) delta between successive evaluated points in _solve_grid for a full 25-point scan and compare with GRID_EPSILON; then re-run the grid with the seed disabled and diff the selected point.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-22",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 958,
    "class": "state",
    "severity": "S3",
    "claim": "'Warm start: seed=None lets get_residual_pure use the previous segment's accepted dMdt (carried in params); the solved dMdt then threads forward.'",
    "evidence": "The module advertises pure, non-mutating functions (L3-18 'Pure functions that return results instead of mutating the params dictionary'; L400 'without mutating params'; L684 'params : dict-like, Parameter dictionary (not mutated)') and L108-123 asserts get_bubbleproperties_pure 'only READS params (never writes)'. Yet a solved dMdt must be written into params by someone for it to 'thread forward' and be 'carried in params' between segments.",
    "expected": "Name the writer (presumably the runner) so the purity claim is scoped to this module, and document bubble_dMdt as inter-segment mutable state that makes results order-dependent.",
    "failure_scenario": "A test or harness that reuses a params dict across configurations inherits a stale bubble_dMdt seed, making runs non-reproducible in-process — the exact failure mode CLAUDE.md's 'separate processes' rule exists for.",
    "repro": "Grep for assignments to params['bubble_dMdt'] / bubble_dMdt.value across the phase-1b runner and confirm whether get_bubbleproperties_pure truly never writes.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-23",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 335,
    "class": "other",
    "severity": "S2",
    "claim": "effective_Lloss is 'the single point where the opt-in unresolved-interface-cooling boost is applied', fed CONSISTENTLY to the beta-delta residual, the energy ODE (Edot_from_balance) and the energy->momentum trigger; L361-365 'Thin wrapper so the three call sites stay one line and identical.'",
    "evidence": "Only two of the three call sites are visible in this slice (L472 in get_residual_pure and L576 in get_residual_detailed); the energy ODE and the E->p trigger are outside it. The claim is a cross-module invariant asserted from inside one module.",
    "expected": "Exactly three call sites of effective_Lloss_from_params, and no other site that composes Lcool + Lleak by hand.",
    "failure_scenario": "If the energy ODE or the E->p trigger sums Lcool + Lleak directly, then with cooling_boost enabled the residual is solved against a boosted loss while the ODE integrates an unboosted one (or vice versa) — the bubble energy trajectory and the phase-transition time disagree with the beta that was accepted.",
    "repro": "Grep the package for 'effective_Lloss_from_params' and, separately, for every expression combining Lcool with Lleak; confirm the counts match.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-24",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 59,
    "class": "numerical",
    "severity": "S4",
    "claim": "'residuals at the accepted point grow by roughly 3x per subsequent segment as beta/delta drift, so a deeply converged pick keeps the next segments' input guesses below RESIDUAL_THRESHOLD (long runs of 1-evaluation short-circuits), while a barely-converged pick forces a fresh grid search almost immediately.' Also '~25-100x faster per evaluation' (L3, L108) and '~50 expensive function evaluations' (L52).",
    "evidence": "Three quantitative empirical claims (3x/segment growth, 25-100x speedup, ~50 evaluations) with no cited measurement, committed CSV/figure, or config named — the tuning of GRID_EARLY_EXIT_RESIDUAL rests entirely on the 3x figure.",
    "expected": "Per CLAUDE.md rule 5, persist the measurement as a committed diagnostic with the exact config and command, or cite the existing one.",
    "failure_scenario": "The 3x growth rate is regime-dependent; in a regime where it is larger, GRID_EARLY_EXIT_RESIDUAL no longer guarantees the next segment short-circuits and the claimed cost saving inverts (early exit picks a worse point AND triggers a full grid next segment).",
    "repro": "Instrument input-guess residual per segment across param/simple_cluster.param and docs/dev/performance/f1edge_{lowdens,hidens}*.param and fit the growth factor.",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-25",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 70,
    "class": "citation",
    "severity": "S4",
    "claim": "'The finite-difference step eps is the residual noise floor measured in the Phase-2.1 transect probe (docs/dev/archive/betadelta/diagnostics), not the 1e-4 acceptance threshold; factor=0.1 keeps Newton steps local so the root-finder does not leap into ODE-failing (beta, delta); maxfev caps cost.'",
    "evidence": "The numeric values of eps and maxfev are not in the prose; the sole justification for eps points into docs/dev/, which the project's own CLAUDE.md declares unverified and drift-prone ('Treat every claim there as unverified'). Similarly 'FINDINGS §14', 'KAPPA_FREEZE_MECHANISM' (L650) and 'Phase 3, plan arm D' (L949) are internal, unversioned references carrying load-bearing design justifications.",
    "expected": "State the numeric eps in the comment next to the constant (so the code is self-describing), and pin the path/section of any docs/dev reference to a commit or move the essential result inline.",
    "failure_scenario": "eps is left stale after a change to the residual definition (e.g. the f->g switch changed the residual's scale by ~1/Lmech, which changes the noise floor); hybr's Jacobian is then differenced on a step below or far above the true noise floor, and convergence degrades silently.",
    "repro": "Compare the eps constant's value against the g residual's actual noise floor (re-run the transect probe under the g metric, not the f metric it was measured under).",
    "confidence": "medium"
  },
  {
    "id": "S5a-B-26",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 949,
    "class": "deadcode",
    "severity": "S4",
    "claim": "'`method` is accepted for signature parity with the legacy solver and ignored (hybr takes no grid method).'",
    "evidence": "L949-957. A caller passing method='lbfgsb' under the production default solver gets grid-free hybr with no warning.",
    "expected": "Either drop the parameter at the dispatch boundary or warn when a non-default method is passed to hybr.",
    "failure_scenario": "A diagnostic script forces method='lbfgsb' to compare optimisers and silently gets identical hybr results, concluding the method has no effect.",
    "repro": "Check solve_betadelta_pure's dispatch signature and whether any caller passes method explicitly.",
    "confidence": "high"
  },
  {
    "id": "S5a-B-27",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1017,
    "class": "numerical",
    "severity": "S4",
    "claim": "'The caller has already evaluated the input guess itself (the grid center), so that point is skipped here'; skip uses a tolerance compare 'because linspace's midpoint can differ from the guess in the last ulp'; 'For a guess near — but not at — a clamped bound, the shifted grid no longer contains the guess, no point matches, and the full grid runs.'",
    "evidence": "L1017-1041 says the linspace is 'clamped at the parameter bounds' while L1073-1075 says the grid is 'shifted' — these describe different operations. Under a pure np.clip of the linspace values the midpoint would be unchanged and the guess would still match; only clamping the endpoints before building the linspace moves the centre off the guess. Additionally, when the centre is no longer the guess, best-so-far is still seeded with input_residual/input_props for an off-grid point.",
    "expected": "One consistent description of the clamping, and a statement of whether the returned best may be the (possibly out-of-bounds) input guess rather than a grid point — which matters for the 'bounded' claim relied on at L650 (see S5a-B-17).",
    "failure_scenario": "Near a bound the scan silently costs 25 evaluations instead of 24 (benign), but the centre-out ordering justification ('the optimum lies near the grid center') no longer holds, and the returned point may be off-grid.",
    "repro": "Build the grid for a guess at bound - 0.5*GRID_EPSILON and inspect the resulting linspace, its midpoint, and whether the guess is among the points.",
    "confidence": "low"
  }
]
```
