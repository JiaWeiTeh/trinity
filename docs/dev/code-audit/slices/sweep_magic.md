# Sweep: magic numbers and constant provenance

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

Scope: read-only sweep of `/home/user/trinity/trinity/**` against the 1644-row mechanical
literal extract `docs/dev/code-audit/data/claims_literals.csv` (commit `b34025a`).
Priority order was `bubble_structure/`, `cooling/`, `shell_structure/`, `cloud_properties/`,
`phase*/`, `sps/`, `_functions/`, then I/O.

Verification methods used, in order of strength:
1. **Numerical re-derivation** — every conversion factor in `unit_conversions.py` re-computed
   against `astropy.units`; the Bonnor–Ebert critical constants re-computed by integrating the
   isothermal Lane–Emden equation; the Weaver Eq-37 temperature coefficient cross-checked
   against the code's own Eq-33/Eq-44 bubble-structure relations.
2. **Live instrumentation** — `param/simple_cluster.param` run under a monkeypatched RHS to
   measure what a flagged constant actually does to the state vector (used for MN-001).
3. `git log -S` / `git blame` for provenance.
4. Literature search for published coefficients (explicitly marked where **unconfirmed**).

Provenance caveat that applies to nearly every entry below: the physics tree was squashed into
a single commit (`bf50e44 "plotting scripts for runtime"`), so `git log -S` returns that one
commit for most literals and yields **no usable justification history**. Where a constant does
have real provenance I say so; where it does not, "provenance" says so explicitly rather than
inventing one.

---

### MN-001 · A hardcoded `vd = -1e8` overrides the momentum equation for the whole first energy-phase segment, so the shell's initial velocity is set by the integrator's step size, not by physics
- **file:line** — `trinity/phase1_energy/energy_phase_ODEs.py:269-270`
  ```python
      # Early phase approximation
      if snapshot.EarlyPhaseApproximation:
          vd = -1e8
  ```
  and `trinity/phase1_energy/run_energy_phase.py:55`, `:342-343`
  ```python
  SEGMENT_DURATION = 3e-5  # Myr - duration of each integration segment (~30 years)
  ...
          if loop_count == 0 and params['EarlyPhaseApproximation'].value:
              params['EarlyPhaseApproximation'].value = False
  ```
- **class** — tuned-parameter (a numerical constant leaking into a physical result)
- **severity** — **S1 results-wrong**
- **value in code** — `-1e8` [pc/Myr²], with `EarlyPhaseApproximation` defaulting to `True`
  (`trinity/_input/registry.py:423`)
- **expected** — the computed RHS
  `vd = (4πR2²(P_drive − P_ext) − mShell_dot·v2 − F_grav + F_rad)/mShell`, i.e. the line
  immediately above. There is no published "early phase approximation" that replaces the
  momentum equation with a constant deceleration; no reference is cited in the code.
- **units the expression demands vs supplies** — dimensionally an acceleration, so it is at
  least type-correct; but its *magnitude* is unanchored. −1e8 pc/Myr² ≈ −3.1e-6 cm/s², which is
  ~10⁵× the actual `vd` at that moment.
- **provenance** — the flag is flipped to `False` at `run_energy_phase.py:342`, but that line
  runs **after** the first `solve_ivp` call (line 299), so segment 0 integrates entirely under
  the override. `git log -S'-1e8'` returns only `bf50e44` (the squash commit); `git log
  -S'EarlyPhaseApproximation'` returns the same. **No justification exists in history.** The
  name suggests it was meant as a transient stabiliser for the free-streaming→Weaver handoff,
  but it is not a stabiliser: it is applied unconditionally and its effect scales with the
  segment length.
- **failure scenario** — measured live, `param/simple_cluster.param`:
  - phase 1a enters at `t = 3.3882e-07 Myr, R2 = 1.2669e-03 pc, v2 = 3739.2 pc/Myr`;
  - every RHS evaluation in segment 0 returns `vd = -1.0000e+08` (verified: `early=True` on
    every call until `t = 3.0339e-05`);
  - segment 0 exits at `v2 = 739.24 pc/Myr`. The change is **exactly**
    `Δv = −1e8 × SEGMENT_DURATION = −3000.000 pc/Myr` — an **80 % haircut on the initial shell
    velocity, set by a product of two unrelated numerical constants.**
  - Sensitivity proof that this contaminates the physics: halving `SEGMENT_DURATION`
    (3e-5 → 1.5e-5) shifts the **end-of-phase-1a** state at matched `t = 2.910339e-03 Myr`:

    | SEGMENT_DURATION | R2 [pc] | v2 [pc/Myr] | Eb [au] |
    |---|---|---|---|
    | 3e-5 (production) | 2.857315e-01 | 4.473918e+01 | 7.782364e+05 |
    | 1.5e-5 | 2.837668e-01 | 4.497362e+01 | 7.709290e+05 |
    | rel. diff | 0.69 % | 0.52 % | 0.94 % |

    A step-size constant is *supposed* to be the thing the answer is insensitive to.
  - **Sign-reversal regime, reachable with documented parameters.** The haircut is a fixed
    3000 pc/Myr (≈2933 km/s) regardless of `v0`. `v0 = 2·Lmech_W/pdot_W` is the wind terminal
    velocity after the feedback corrections in `read_sps.py:215-221`, i.e.
    `v0 = v_raw·sqrt(FB_thermCoeffWind/(1+FB_mColdWindFrac))`. With the bundled SB99 table
    `v_raw ≈ 3739 pc/Myr`, so setting `FB_mColdWindFrac = 1.0` (a documented `.param` key,
    default 0) gives `v0 ≈ 2644 pc/Myr` and segment 0 ends at **`v2 ≈ −356 pc/Myr` — the shell
    is moving inward and `R2` is shrinking before any physics has been applied.** I observed
    exactly this state (`v2 = −5.04e+02`, `R2` decreasing, `Eb = −3.4e+04`) in a truncated
    variant run. The same happens for any user-supplied SPS table with
    `v_terminal < 2933 km/s`, which is the majority of the realistic range.
- **confidence** — **high** (measured directly; the −3000.000 pc/Myr arithmetic is exact)

---

### MN-002 · The tanh cloud-edge bridge (`SMOOTH_FRAC = 0.01`) is applied to the density but not to the enclosed-mass formula, so `M(r)` and `dM/dt = 4πr²ρ(r)v` disagree by up to a factor 2 exactly at the blowout radius
- **file:line** — `trinity/cloud_properties/density_profile.py:128-130`
  ```python
      SMOOTH_FRAC = 0.01
      delta = SMOOTH_FRAC * rCloud
      w_outside = 0.5 * (1.0 + np.tanh((r_arr - rCloud) / delta))
  ```
  vs `trinity/cloud_properties/mass_profile.py:316` / `:332-342` (sharp analytic `M(r)`, no
  smoothing), consumed together at `mass_profile.py:224`
  `dMdt_arr = 4.0*np.pi*r_arr**2*rho_arr*rdot_arr`.
- **class** — tuned-parameter (with a duplicate-disagreement consequence)
- **severity** — **S2 latent** (fires only for `R2` within ~3 % of `rCloud`, but that is
  precisely the blowout regime the `transition_trigger='blowout'` mode is built to study)
- **value in code** — `0.01` (fractional bridge half-width)
- **expected** — either both `n(r)` and `M(r)` smoothed, or neither. `compute_enclosed_mass_powerlaw`
  returns `M(rCloud) = mCloud` exactly while the smoothed profile integrates to
  0.98973·mCloud inside `rCloud` (computed: 1.027 % deficit for the default
  `nCore/nISM = 1e5`).
- **units the expression demands vs supplies** — n/a (dimensionless fraction of `rCloud`).
- **provenance** — the in-code comment gives the rationale (LSODA stalling on the ~10³ density
  step) and claims *"mass conservation holds to O(SMOOTH_FRAC²)"*. That claim is **correct for
  the total mass over all space** (the leading antisymmetric term cancels, leaving O(δ²/R²)),
  but it is **wrong for the quantity the code actually uses**, `M(<rCloud)`, which is
  O(SMOOTH_FRAC) = 1 % low. This is a numbers-in-a-comment-vs-code mismatch of the kind the
  sweep is asked to hunt.
- **failure scenario** — measured density ratio `n(r)/nCore` across the bridge:

  | r/rCloud | 0.95 | 0.98 | 0.99 | 1.00 | 1.01 |
  |---|---|---|---|---|---|
  | n/nCore | 0.99995 | 0.9820 | 0.8808 | **0.5000** | 0.1192 |

  At `R2 = rCloud` the ODE's shell-sweeping term `−mShell_dot·v2` in
  `energy_phase_ODEs.py:266` is computed from `ρ = (ρ_cloud+ρ_ISM)/2 ≈ ρ_cloud/2`, i.e. the
  ram-loading deceleration is **halved**, while `mShell` (from the sharp analytic branch) is the
  full `mCloud`. The two terms in the same equation are drawn from two different density
  profiles. Concretely, for a run configured with `transition_trigger = 'blowout'`
  (`run_energy_implicit_phase.py:243`, criterion `R2 > rCloud`), the acceleration at the trigger
  radius is systematically too large.
- **confidence** — high (both the code paths and the 1.027 % / 0.50 numbers are computed)

---

### MN-003 · `WEAVER_TEMP_COEFFICIENT = 1.51e6` — units confirmed correct, coefficient value **unconfirmed** against Weaver+77 Eq 37 (secondary sources quote 2.07e6 for what may be the same quantity)
- **file:line** — `trinity/phase0_init/get_InitPhaseParam.py:30-35`, applied at `:172-176`
  ```python
  # Temperature coefficient in Weaver+77, Eq. 37
  # T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4
  WEAVER_TEMP_COEFFICIENT = 1.51e6  # Kelvin
  WEAVER_L_REF = 1e36
  ...
      T0 = WEAVER_TEMP_COEFFICIENT * \
           (Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0) * \
           (nCore * cvt.ndens_au2cgs)**(2.0/35.0) * \
           (dt_phase0)**(-6.0/35.0) * \
           (1.0 - bubble_xi_Tb)**0.4
  ```
- **class** — literature-coefficient
- **severity** — **S4 hygiene** *in current use* (see failure scenario: the value is overwritten
  before it can influence any physics), but it is reported at length because it is the single
  highest-risk misapplied-coefficient candidate in the package and the units question had to be
  settled.
- **value in code** — `1.51e6` K
- **expected** — Weaver, McCray, Castor, Shapiro & Moore 1977, ApJ 218, 377, Eq. (37). I could
  **not** obtain the primary text. Secondary sources consistently quote the bubble temperature
  as `T_b = 2.07 × 10⁶ · L₃₆^(8/35) · n₀^(2/35) · t₆^(−6/35) K` with
  `T(r) = T_b (1 − r/R_s)^(2/5)` — i.e. **2.07e6 in the same slot the code puts 1.51e6**, a
  37 % discrepancy. I cannot rule out that the two coefficients refer to different quantities
  (peak vs. volume-average, or a different `n₀` definition), so this is recorded as
  **unconfirmed, not as an error**. Note the volume average of `(1−x)^(2/5)` is 0.525, which
  does **not** reconcile 1.51 and 2.07 either way — so the difference, if real, is not simply
  averaging.
- **units the expression demands vs supplies** — **this part is confirmed correct**, and it is
  the part most likely to be wrong:
  - `L₃₆` demands erg/s divided by 1e36 — supplied as `Lmech_W * cvt.L_au2cgs / 1e36` ✓
  - `n₀` demands cm⁻³ — supplied as `nCore * cvt.ndens_au2cgs` ✓ (with a residual ~4.6 %
    ambiguity: Weaver's `n₀ = ρ/μm_p` is the *total particle* density while TRINITY's `nCore` is
    the *hydrogen nuclei* density; the ratio 0.435 raised to 2/35 is 0.954)
  - `t₆` demands **10⁶ yr, i.e. Myr** — supplied as `dt_phase0` in Myr ✓. Confirmed two ways:
    (a) the literature definition `t₆ ≡ t/10⁶ yr`; (b) independently, by evaluating the code's
    own Weaver Eq-33 (`_get_init_dMdt`) and Eq-44 (`_get_bubble_ODE_initial_conditions`)
    relations at `L = 1e36`, `n = 1 cm⁻³`, `ξ = 0.98` and comparing:

    | t | T from the code's Eq-33+Eq-44 structure | Eq-37 with t in **Myr** | with t in **yr** | with t in **s** |
    |---|---|---|---|---|
    | 0.1 Myr | 5.731e5 K | 4.686e5 (ratio 1.22) | 4.388e4 (ratio **13.1**) | 2.27e3 (ratio **252**) |
    | 1 Myr | 3.862e5 K | 3.158e5 (ratio 1.22) | 2.957e4 (ratio **13.1**) | 1.53e3 (ratio **252**) |
    | 10 Myr | 2.603e5 K | 2.128e5 (ratio 1.22) | 1.993e4 (ratio **13.1**) | 1.03e3 (ratio **252**) |

    The Myr interpretation is right to 22 %; the years interpretation would be wrong by 13×.
    **The unit handling is sound.**
- **provenance** — inherited from WARPFIELD (Rahner+2017 lineage); no commit history beyond the
  squash commit.
- **failure scenario** — none currently. `T0` from `get_y0` is written to `params['T0']`
  (`main.py:241`) but `get_bubbleproperties_pure` never reads `T0`; the first phase-1a loop
  overwrites it with the solved `bubble_T_r_Tb` (`run_energy_phase.py:186`) before `T0` is
  consumed by anything (its only real consumer is the beta–delta `T_residual` in phase 1b). So a
  wrong coefficient here changes only the `t = t0` snapshot row. **S4.**
- **confidence** — **high** on the units; **low** on the coefficient value (marked unconfirmed).

---

### MN-004 · `1e-100` sentinels in the initial-condition builder convert "no wind at tSF" into a silently garbage initial state instead of an error
- **file:line** — `trinity/phase0_init/get_InitPhaseParam.py:37-40`, used at `:115-138`
  ```python
  MIN_LUMINOSITY = 1e-100  # Prevent div by zero in Mdot calculation
  MIN_MOMENTUM = 1e-100    # Prevent div by zero in velocity calculation
  MIN_VELOCITY = 1e-100    # Prevent div by zero in dt_phase0 calculation
  ```
- **class** — tuned-parameter (sentinel/floor)
- **severity** — **S2 latent**
- **value in code** — `1e-100` in code units (Msun·pc²/Myr³ for luminosity, Msun·pc/Myr² for
  momentum rate, pc/Myr for velocity — three different dimensions, one number)
- **expected** — a raise. These are trust-boundary values read straight out of a user-supplied
  SPS file; a zero wind luminosity at `tSF` means the model has no free-streaming phase to
  integrate, which is a configuration error, not a numerical one.
- **units the expression demands vs supplies** — n/a, but note the same literal is used as a
  floor for three quantities with three different dimensions, so it cannot be "small" in any
  consistent physical sense.
- **provenance** — squash commit only; the comments state the intent ("prevent div by zero")
  but not what value the caller should see when the floor binds.
- **failure scenario** — an SPS table with `Lmech_W(tSF) = 0` (e.g. a user table whose wind
  columns start at a later age, or `FB_thermCoeffWind = 0`): the code logs a `WARNING` and
  continues with `Lmech_W = 1e-100`, giving
  `Mdot0 = pdot²/2e-100` (astronomically large), `E0 = (5/11)·1e-100·dt ≈ 0`, and
  `T0 = 1.51e6·(1e-100·6e29/1e36)^(8/35)·… ≈ 1e-18 K`. The run then proceeds into phase 1a with
  a zero-energy, zero-temperature bubble; the collapse guard at `run_energy_phase.py:368`
  (`Eb <= 0`) may or may not fire depending on rounding. Nothing downstream distinguishes this
  from a physically converged tiny bubble.
- **confidence** — high (the arithmetic is deterministic); medium that a real user table would
  trigger it.

---

### MN-005 · The shipped default `densBE_Omega = 14.1` exceeds `OMEGA_CRITICAL = 14.04` defined in the same package, so the default Bonnor–Ebert cloud is gravitationally unstable by the code's own test
- **file:line** — `trinity/_input/default.param:98` / `trinity/_input/registry.py` spec
  ```
  densBE_Omega    14.1
  ```
  vs `trinity/cloud_properties/bonnorEbertSphere.py:78`
  ```python
  OMEGA_CRITICAL = 14.04      # Critical density contrast ρc/ρsurf
  ```
  consumed at `bonnorEbertSphere.py:357-363`
  ```python
      if Omega > OMEGA_CRITICAL:
          logger.warning(f"Omega={Omega:.2f} > {OMEGA_CRITICAL:.2f} (critical). "
                         f"Sphere will be gravitationally UNSTABLE!")
      is_stable = Omega < OMEGA_CRITICAL
  ```
- **class** — duplicate-disagreement
- **severity** — **S3 misleading** (default config emits a physics warning and reports
  `is_stable = False` for every `dens_profile = densBE` run that does not override `Omega`)
- **value in code** — default `14.1`; critical `14.04`
- **expected** — `OMEGA_CRITICAL = 14.04` and `XI_CRITICAL = 6.451` are both **correct**
  (I re-integrated the isothermal Lane–Emden equation: ρ_c/ρ at ξ=6.451 is 14.0433, and the
  Bonnor dimensionless mass peaks at `m_B = 1.18223` at ξ = 6.4493, Ω = 14.034). So the constant
  is right and the *default* is on the wrong side of it. A default of e.g. 14.0 would be stable;
  14.1 is not.
- **units the expression demands vs supplies** — n/a (dimensionless density contrast).
- **provenance** — squash commit only. The value 14.1 looks like a rounding-up of 14.04, i.e. it
  was probably intended *as* the critical value.
- **failure scenario** — `dens_profile = densBE` without an explicit `densBE_Omega`: every run
  logs `Omega=14.10 > 14.04 (critical). Sphere will be gravitationally UNSTABLE!` and
  `BESphereResult.is_stable = False`. The profile is still computed (ξ_out = 6.4617 vs 6.4504),
  so the numerical difference is ~0.2 % in radius — the harm is the false instability signal,
  not the geometry. Note this is only reachable via a non-default `dens_profile`; the shipped
  `dens_profile` is `densPL`.
- **confidence** — high

---

### MN-006 · `mu: float = 1.4` default arguments in four cloud-property functions are in m_H units while the documented and production convention for that argument is Msun — a factor ~1.2e57
- **file:line** — `trinity/cloud_properties/powerLawSphere.py:51`, `:77`, `:214`;
  `trinity/cloud_properties/bonnorEbertSphere.py:302`
  ```python
  def compute_rCloud_homogeneous(M_cloud, nCore, mu=1.4):
      ...
      mu : float
          Mean molecular weight [Msun] (internal units, converted from m_H)
      ...
      rhoCore = nCore * mu
  ```
- **class** — unit-conversion
- **severity** — **S2 latent** (every production call site passes `mu` explicitly; the default is
  a trap for tests, tools and `paper/` scripts)
- **value in code** — `1.4`
- **expected** — `cvt.convert2au('m_H') * 1.4 = 1.4 · m_H · g2Msun ≈ 1.1786e-57` Msun, i.e.
  the value `params['mu_convert'].value` carries. The literal `1.4` is the value in units of
  `m_H`, which is the `.param` spelling, not the internal spelling.
- **units the expression demands vs supplies** — the expression demands Msun (so that
  `rho = nCore[pc⁻³] · mu[Msun]` lands in Msun/pc³); the default supplies a dimensionless
  m_H-count. Using the default makes `rhoCore` too large by **1/1.1786e-57 ≈ 8.5e56**, and
  `rCloud ∝ ρ^(−1/3)` too small by ~9.5e18.
- **provenance** — squash commit only. `bonnorEbertSphere.py:324` already carries a docstring
  note ("the 1.4 default arg is a placeholder"), showing the hazard is known but was documented
  rather than removed.
- **failure scenario** — any caller that omits `mu` — e.g. the docstring's own example,
  `compute_rCloud_powerlaw(1e5, nCore, alpha=-2)` at `powerLawSphere.py:121`, or
  `compute_consistent_params(...)` which is advertised as "the recommended way to set up test
  parameters" — gets an `rCloud` ~19 orders of magnitude too small, with no error: the internal
  forward mass check at `:166-173` passes, because it re-uses the same wrong `rhoCore`.
- **confidence** — high

---

### MN-007 · `_thr = _thr if _thr else 0.05` silently discards a user-set `phaseSwitch_LlossLgain = 0`, so phase 1a and phase 1b use different transition thresholds
- **file:line** — `trinity/phase1_energy/run_energy_phase.py:280-282`
  ```python
              _thr = params['phaseSwitch_LlossLgain'].value
              _thr = _thr if _thr else 0.05
              if _Lgain > 0 and (_Lgain - _Lloss) / _Lgain < _thr:
  ```
  vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1249-1254`
  ```python
          phase_switch_threshold = params.get('phaseSwitch_LlossLgain', None)
          if phase_switch_threshold and hasattr(phase_switch_threshold, 'value'):
              threshold = phase_switch_threshold.value
          else:
              threshold = 0.05
  ```
- **class** — duplicate-disagreement
- **severity** — **S3 misleading**
- **value in code** — hardcoded fallback `0.05` in both places; registry default is also `0.05`
  (`trinity/_input/registry.py:382`)
- **expected** — the user's value, unconditionally. `0.0` is a legitimate setting ("only switch
  once radiative losses fully balance the mechanical input").
- **units the expression demands vs supplies** — n/a (dimensionless ratio).
- **provenance** — squash commit only. Note the 1b spelling is *correct* — `DescribedItem`
  defines no `__bool__`/`__len__` (checked `trinity/_input/dictionary.py:98-190`), so the object
  is always truthy and `threshold` picks up `0.0` faithfully. The 1a spelling tests the
  **unwrapped float**, so `0.0` is falsy and is replaced.
- **failure scenario** — a `.param` with `phaseSwitch_LlossLgain 0`: phase 1a can break out of
  the energy phase at a 5 % cooling-balance margin (`run_energy_phase.py:287`) while phase 1b
  never fires the same criterion. The run silently ends the early phase early. Reachable only
  with the exact value `0`.
- **confidence** — high on the mechanism; medium on the practical impact (needs a user to set 0).

---

### MN-008 · Chained sentinels `Ebubble = 1e-30` and `shell_volume = 1e-13·r2³` convert a bubble-energy collapse into a huge-but-finite `Pb` rather than an error
- **file:line** — `trinity/bubble_structure/get_bubbleParams.py:406-407` and `:229-236`
  ```python
      # set minimum energy to avoid zero
      if Ebubble < 1e-30:
          Ebubble = 1e-30
  ...
      shell_volume = r2**3 - r1**3
      if shell_volume <= 0:
          shell_volume = 1e-13 * r2**3
      Pb = (gamma - 1) * Eb / shell_volume / (4 * np.pi / 3)
  ```
- **class** — tuned-parameter (sentinel/floor)
- **severity** — **S2 latent**
- **value in code** — `1e-30` [Msun·pc²/Myr² ≈ 1.9e13 erg]; `1e-13` [dimensionless volume
  fraction]; also `r2 += 1e-10` at `:224` (after the pc→cm conversion, so 1e-10 cm ≈ 3.2e-29 pc,
  genuinely negligible — that one is fine)
- **expected** — a raise, or a documented `Pb = 0` handoff. The two floors are *coupled*: the
  `Ebubble` floor in `get_r1` drives the root `r1 → r2`, which is exactly the condition that
  makes `r2³ − r1³` underflow and trips the second floor.
- **units the expression demands vs supplies** — the `1e-30` floor is applied to an energy in
  internal units without comment; 1e-30 au = 1.9e13 erg, which is not obviously "negligible" as
  an absolute energy, only relative to a typical `Eb ~ 1e51 erg = 5e7 au`.
- **provenance** — the `1e-13` floor carries a real, informative in-code justification
  (catastrophic-cooling degeneracy, `Eb ≤ 0` handoff) and claims bit-identity on physical
  bubbles; that claim is sound (`shell_volume > 0` short-circuits it). The `1e-30` floor has a
  bare one-line comment and no provenance.
- **failure scenario** — energy-driven collapse with `0 < Eb ≲ 1e-30`: `get_r1` solves
  `sqrt(Lmech/v/1e-30·(r2³−r1³)) = r1`, forcing `R1 → R2`; `bubble_E2P` then floors the shell
  volume at `1e-13·r2³` and returns a `Pb` up to ~10¹³× the physical value for that geometry.
  That inflated `Pb` is handed to `shell_structure_pure` (`nShell0 ∝ Pb`) and to
  `get_bubbleproperties_pure` (`ndens ∝ Pb`, `dTdrr ∝ Pb`) with no flag. The `Eb <= 0` detector
  at `run_energy_phase.py:368` only catches the case where `Eb` has already gone non-positive,
  which is one float step later.
- **confidence** — medium (the mechanism is clear; I did not construct a run that reaches it)

---

### MN-009 · `MIN_RADIUS_FACTOR = 1.5` makes the actual collapse-termination radius 1.5× the documented `coll_r`, and renders the second, `coll_r`-based collapse check unreachable
- **file:line** — `trinity/phase_general/phase_events.py:71-72`, applied at `:445`, `:485`,
  `:529`, `:568`
  ```python
  MIN_RADIUS_SAFETY = 0.01       # pc - absolute minimum radius
  MIN_RADIUS_FACTOR = 1.5        # Factor above coll_r for early termination
  ...
      min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)
  ```
  vs `trinity/phase1c_transition/run_transition_phase.py:~790`
  ```python
              coll_r = params['coll_r'].value
              if R2 < coll_r:
                  termination_reason = "small_radius"
  ```
- **class** — duplicate-disagreement / tuned-parameter
- **severity** — **S2 latent**
- **value in code** — `1.5`
- **expected** — the parameter's own documentation, `default.param:126`: *"Radius below which
  the cloud is considered completely collapsed"*, default `coll_r = 1` pc. The run actually
  terminates at 1.5 pc.
- **units the expression demands vs supplies** — n/a (dimensionless factor on pc).
- **provenance** — squash commit only; the inline comment states the *what* ("Factor above
  coll_r for early termination") but not what it was tuned against.
- **failure scenario** — measured for the shipped `param/simple_cluster.param`
  (`mCloud 1e5, sfe 0.3`): `rCloud = 1.6900 pc`, `coll_r = 1.0 pc`, so
  `min_r = 1.5 pc = 0.888 × rCloud`. A re-collapsing shell is declared terminally collapsed
  while still at **89 % of the cloud radius** — the run stops before it can show whether the
  collapse actually completes. Separately, because the ODE event (1.5 pc) always fires strictly
  before the manual check (1.0 pc), the `R2 < coll_r` branch in the transition and momentum
  runners is dead code, so the two criteria never agree on what "collapsed" means.
  Boundary behaviour: for any config with `rCloud < 1.5 pc` (dense, low-mass clouds) the
  termination radius exceeds the cloud itself; the event only survives because
  `direction = -1` means it is not armed until `R2` has first crossed 1.5 pc upward.
- **confidence** — high (radii computed from the shipped config)

---

### MN-010 · `dt = 1e-9` Myr in the `pdotdot` central difference evaluates the SPS interpolator outside its own validated domain
- **file:line** — `trinity/sps/update_feedback.py:184-185`
  ```python
      dt = 1e-9  # Myr (small timestep for derivative)
      pdotdot_total = (sps_f['fpdot_total'](t + dt)[()] - sps_f['fpdot_total'](t - dt)[()]) / (2.0 * dt)
  ```
- **class** — tuned-parameter
- **severity** — **S2 latent**
- **value in code** — `1e-9` Myr (≈ 31.6 s)
- **expected** — a step consistent with the domain guard 12 lines above
  (`if not (t_min <= t <= t_max): raise`), or a one-sided difference at the endpoints. The
  interpolators are built with `scipy.interpolate.interp1d(..., kind='cubic')` and **default
  `bounds_error=True`** (`read_sps.py:341-…`), so out-of-range input raises.
- **units the expression demands vs supplies** — Myr on both sides ✓.
- **provenance** — squash commit only.
- **failure scenario** — two boundaries:
  1. `t = t_min` (= 0.0 after the t=0 prepend at `read_sps.py:264`): `fpdot_total(-1e-9)` raises
     `ValueError: A value in x_new is below the interpolation range`. Reached when
     `dt_phase0 < 1e-9 Myr`, i.e. a very dense ambient medium or a very small `Mdot0`
     (`dt_phase0 = sqrt(3·Mdot0/(4π·ρa·v0³))`). For `simple_cluster`, `dt_phase0 = 3.39e-7 Myr`
     — only 2.5 decades of margin.
  2. `t = t_max` exactly: `fpdot_total(t_max + 1e-9)` raises. Reachable when `stop_t` coincides
     with the SPS table's last tabulated age.

  Also note the accuracy cost: at `t ~ 1e-3 Myr` the two evaluations differ in the ~1e-6th
  relative digit, so the central difference retains only ~9-10 significant digits of the
  derivative. `pdotdot_total` feeds `a_coeff = 1.5·pdotdot/pdot` in the Rahner-A12 residual, so
  the noise propagates into `cool_beta`. That is tolerable at the current `RESIDUAL_THRESHOLD =
  1e-4` but leaves no headroom if that threshold is ever tightened.
- **confidence** — high on the mechanism; medium that a shipped config reaches it

---

### MN-011 · `_coolingswitch = 1e4` sits a factor 3 below the ODE's own outer boundary `_T_INIT_BOUNDARY = 3e4`, so the entire L3 "intermediate" luminosity is a **linear** extrapolation across a range where the solution is `T ∝ (R2−r)^(2/5)`
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:52`, `:703`, `:801-809`
  ```python
  _T_INIT_BOUNDARY = 3e4
  ...
      _coolingswitch = 1e4
  ...
      R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch]) / dTdR_coolingswitch + r_array[index_cooling_switch]
      fT_interp_interm = interp1d(
          np.array([r_array[index_cooling_switch], R2_coolingswitch]),
          np.array([T_array[index_cooling_switch], _coolingswitch]),
          kind='linear')
      r_interm = np.linspace(r_array[index_cooling_switch], R2_coolingswitch, num=1000, endpoint=True)
  ```
- **class** — tuned-parameter
- **severity** — **S3 misleading**
- **value in code** — `1e4` K and `3e4` K
- **expected** — the ODE is integrated inward from `T = 3e4 K` at `r2Prime`, so `T_array` never
  goes below 3e4 and `index_cooling_switch` is pinned at 0. The slab from `r2Prime` outward down
  to `T = 1e4` is therefore never integrated; it is reconstructed by extending the boundary slope
  `dTdr = −(2/5)T/dR2` linearly. Over a factor-3 temperature drop the true self-similar profile
  is `T ∝ (R2−r)^(2/5)`, which is strongly convex, so a straight line systematically misplaces
  both the slab thickness and the temperature at which the non-CIE cooling function peaks.
- **units the expression demands vs supplies** — n/a (K, and pc for the extrapolated radius).
- **provenance** — `_T_INIT_BOUNDARY` has an unusually good in-code justification: a named
  constant with a three-role explanation and an explicit statement that it is *"NOT the physical
  'no cooling below' floor — that is `_coolingswitch=1e4`, a deliberately separate quantity."*
  So the *separation* is intentional and documented. What is **not** documented is that the
  separation forces L3 to be an extrapolation rather than a solve. The justification for 3e4
  itself (a boundary-transient discussion) still holds; the consequence for L3 was not part of
  it.
- **failure scenario** — `L_intermediate` (returned as `bubble_L3Intermediate`, summed into
  `bubble_LTotal`, and scaled by `cooling_boost_fA`) is computed on a 1000-point linear ramp
  through the 1e4–3e4 K band where the non-CIE CLOUDY tables vary fastest — the code's own
  comment at `:802` says *"important because the cooling function varies a lot between 1e4 and
  1e5K"*. Raising `_T_INIT_BOUNDARY` shrinks the extrapolated slab and changes `L_total`;
  lowering it toward 1e4 makes the slab vanish. There is no test pinning the L3 fraction against
  a converged reference.
- **confidence** — medium-high (mechanism read off the code; magnitude not measured)

---

### MN-012 · `v_array[0] + 1e-4` regularises a **signed** denominator, moving the residual's pole rather than removing it
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:368`
  ```python
      residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)
  ```
- **class** — tuned-parameter
- **severity** — **S4 hygiene** (cannot currently produce a wrong result — see below — hence
  reported without a promotion to S2)
- **value in code** — `1e-4` pc/Myr
- **expected** — `max(|v[0]|, eps)` with the sign preserved, or an explicit rejection when
  `v[0]` is near zero.
- **units the expression demands vs supplies** — pc/Myr on both ✓.
- **provenance** — squash commit only; no comment.
- **failure scenario** — `v_r2Prime = cool_alpha·R2/t_now − (dMdt/4πR2²)·k_B T/(mu_ion·Pb)` is a
  difference of two comparable positive terms, so it can pass through zero. When it does, the
  additive `+1e-4` does not cancel the pole; it relocates it to `v[0] = −1e-4`, where `fsolve`
  sees a sign-flipped, unbounded residual. In practice `v[0] ~ O(100) pc/Myr` in the profiled
  regimes (I measured `cool_alpha·R2/t ≈ 180 pc/Myr` at `t ~ 1e-3 Myr`, `R2 ~ 0.3 pc`), so the
  term is a 1e-6 relative perturbation and **the pole is not reached** — this is S4, not S2. It
  becomes live only if a regime is found where the two terms in `v_r2Prime` cancel.
- **confidence** — high

---

### MN-013 · Bare unit-conversion literals that bypass `unit_conversions.py`
- **file:line** —
  - `trinity/cloud_properties/bonnorEbertSphere.py:564` — `params['densBE_sigma'].value = result.c_s / 1.0e5  # c_s [cm/s] -> sigma [km/s]`
  - `trinity/phase0_init/get_InitCloudProp.py:349` — `params['densBE_sigma'].value = be_result.c_s / 1.0e5  # cm/s -> km/s`
  - `trinity/cooling/non_CIE/read_cloudy.py:48` — `age = params['t_now'] * 1e6` (Myr → yr)
  - `trinity/sps/sps_columns.py:116` — `'yr': 1.0e-6` (yr → Myr)
- **class** — unit-conversion
- **severity** — **S4 hygiene**
- **value in code** — `1.0e5`, `1e6`, `1.0e-6`
- **expected** — all four are **arithmetically correct**. `unit_conversions.py` exposes
  `v_au2kms`/`v_cms2au` but no cm/s→km/s and no Myr↔yr name (only `Mdot_au2Msunyr = 1e-6`,
  which is the same factor as the `sps_columns` entry but named for a different quantity). So
  these are duplications of conversions that *exist implicitly* rather than contradictions.
- **units the expression demands vs supplies** — consistent in all four cases (verified by hand).
- **provenance** — squash commit only.
- **failure scenario** — none today. The risk is that the two `c_s / 1.0e5` sites are copies of
  each other in different modules and only one would be found by a future edit.
- **confidence** — high

---

### MN-014 · `dMdt_factor = 1.646` in the Weaver Eq-33 seed — **unconfirmed** against the primary source
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:297-308`
  ```python
  def _get_init_dMdt(params, Pb: float) -> float:
      """Initial guess for dMdt (Equation 33 in Weaver+77)."""
      dMdt_factor = 1.646
      ...
      return (12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * R2**3 / t_now
              * mu_ion / k_B
              * (t_now * C_thermal / R2**2)**(2/7)
              * Pb**(5/7))
  ```
- **class** — literature-coefficient
- **severity** — **S4 hygiene**
- **value in code** — `1.646` (dimensionless), plus the rational prefactors `12/75`, `5/2`,
  `2/7`, `5/7`
- **expected** — Weaver+77 Eq. (33). **Unconfirmed**: I could not obtain the primary text, and
  the literature search returned no source quoting 1.646. The value matches the WARPFIELD
  implementation this code descends from, which is corroboration of lineage, not of correctness.
  The exponent structure (`Pb^(5/7)`, `(tC/R²)^(2/7)`) is self-consistent with the conduction
  scaling `Ṁ ∝ C^(2/7)` cited elsewhere in this repo.
- **units the expression demands vs supplies** — **all internal (code) units throughout**:
  `R2` [pc], `t_now` [Myr], `mu_ion` [Msun], `k_B` [Msun·pc²/Myr²/K], `C_thermal` [Msun·pc/Myr³/K^(7/2)],
  `Pb` [Msun/pc/Myr²] → `Msun/Myr`. Dimensionally checked and consistent; no cgs literal is
  mixed in. This is the class of bug the sweep targets and **it is not present here**.
- **provenance** — squash commit only.
- **failure scenario** — none directly: the value is only the **seed** for
  `scipy.optimize.fsolve` at `bubble_luminosity.py:261-267`, and is used only when
  `params['bubble_dMdt']` is NaN (the first call of a run). A wrong seed costs iterations, or in
  the worst case steers `fsolve` (`xtol=1e-4, factor=50, epsfcn=1e-4`) to a different root of a
  residual that is known to have penalty plateaus. Corroborating evidence that it is
  approximately right: feeding this Eq-33 `dMdt` into the code's Eq-44 structure relation
  reproduces the independent Eq-37 temperature law to 22 % (table under MN-003).
- **confidence** — low (marked unconfirmed, as instructed)

---

### MN-015 · Dead / inert constants (aggregate)
- **file:line** —
  - `trinity/shell_structure/shell_structure.py:311` — `tau_max = 100` — assigned, never read.
  - `trinity/phase_general/phase_events.py:74` — `MAX_VELOCITY_EXPANSION = 1000.0` — defined,
    never used; no expansion-runaway event is built by any `build_*_events`. Worth noting that
    the shipped `simple_cluster` run enters phase 1a at `v2 = 3739 pc/Myr`, so this event would
    fire immediately if it were ever wired up.
  - `trinity/phase_general/phase_events.py:497` — `cooling_factory = make_cooling_balance_event(threshold=0.05)`
    — the hardcoded 0.05 here is inert because the returned factory is bound at
    `run_energy_implicit_phase.py:752` and never called.
  - `trinity/shell_structure/get_shellODE.py:122` vs `:144` — `f_cover` multiplies `dtaudr` in
    the ionised branch but **not** in the neutral branch. Currently harmless because
    `f_cover = 1` is hardcoded at `shell_structure.py:115` (with a `TODO`), but it is a
    pre-planted inconsistency for whoever implements fragmentation.
  - `trinity/shell_structure/get_shellODE.py:103`, `:134` — `if tau > 500: neg_exp_tau = 0`
    with the comment *"prevent underflow for very large tau values"*. `np.exp(-500) = 7.1e-218`
    is a perfectly normal float64 (underflow begins near `exp(-745)`), so the comment's stated
    reason does not hold; the guard is harmless (it truncates a 7e-218 term) and does suppress
    warnings for genuinely huge `tau`.
- **class** — hygiene
- **severity** — **S4 hygiene**
- **provenance** — squash commit only. Flagged, not removed (pre-existing dead code per the
  project's working rules).
- **confidence** — high

---

## Counts

### By class

| class | count |
|---|---|
| literature-coefficient | 2 (MN-003, MN-014) |
| tuned-parameter | 6 (MN-001, MN-002, MN-004, MN-008, MN-010, MN-011, MN-012 — MN-002 double-counted below) |
| unit-conversion | 2 (MN-006, MN-013) |
| duplicate-disagreement | 3 (MN-005, MN-007, MN-009) |
| hygiene / dead | 1 (MN-015, aggregating 5 sites) |
| **total findings** | **15** |

(MN-002 is counted under `tuned-parameter`; its consequence is a duplicate-disagreement between
`density_profile.py` and `mass_profile.py`.)

### By severity

| severity | count | IDs |
|---|---|---|
| S1 results-wrong | 1 | MN-001 |
| S2 latent | 6 | MN-002, MN-004, MN-006, MN-008, MN-009, MN-010 |
| S3 misleading | 3 | MN-005, MN-007, MN-011 |
| S4 hygiene | 5 | MN-003, MN-012, MN-013, MN-014, MN-015 |

### Coverage

| | count |
|---|---|
| literals in the extract | 1644 |
| in physics + `_functions` modules (the prioritised set) | 1154 |
| trivially benign in that set (0/1/2/3/4/5/0.5/10/100/−1: array indices, powers, `4π`, `r³`) | 827 |
| individually examined and classified | 218 |
| classified **(e) benign** and not reported individually | 203 |
| reported as findings | 15 |

Benign categories aggregated (not reported individually): array indices and slice bounds;
the exponents and rational prefactors of the ODEs (`4π`, `2.5 = 5/2` for a monatomic gas,
`3/2`, `3/4`, `1/3`, `2/5`, `5/2`, `2/7`, `5/7`, `8/35`, `2/35`, `6/35`); formatting widths
(`"=" * 70`, `.4e`); grid sizes and iteration caps whose only consequence is cost
(`N_POINTS = 5000`, `_CONDUCTION_NPTS = 2000`, `_RESIDUAL_NPTS = 500`, `int(2e4)` × 3,
`num=1000`, `MAX_SEGMENTS`, `GRID_SIZE = 5`, `MAX_ITERATIONS = 15`, `maxfev=30`,
`_BUBBLE_DIAG_MAX = 100`, `_SHELL_ODE_MXSTEP = 50000`); diagnostic-only floors in
`_functions/simplify.py`, `_output/`, `_analysis/check_yesno.py` (`1e-30`, `1e-300`) which never
touch the physics; and the `fkappa_auto.py` lookup table, which is measured sweep data rather
than a magic number.

Note on precision: **no truncated-fraction constants were found.** Every fraction in the physics
path is written as an exact rational (`5.0/11.0`, `8.0/35.0`, `2.0/35.0`, `-6.0/35.0`, `5/3`,
`-6/35`, `2/5`, `2/7`, `5/7`, `12/75`, `25/4`), never as `0.33` or `1.67`. The composition
parameters go further and use `fractions.Fraction` (`read_param.py:302-320`) to keep `mu_atom`,
`mu_ion`, `mu_mol` byte-identical to their `14/11`, `14/23`, `14/6` spellings.

---

## Constants checked and **confirmed correct**

So a later reader knows what was covered rather than skipped.

### Physical constants — all re-verified against `astropy.constants` (CODATA 2018 / IAU 2015)

| constant | file | code value | reference | verdict |
|---|---|---|---|---|
| `G` | `unit_conversions.py:202`, `default.param:263` | 6.67430e-8 | 6.67430e-8 cm³/g/s² | exact |
| `k_B` | `unit_conversions.py:205`, `default.param:267` | 1.380649e-16 | 1.380649e-16 erg/K | exact (SI-exact) |
| `m_H` | `unit_conversions.py:208` | 1.6735575e-24 | 1.6735575e-24 g | exact |
| `m_p` | `unit_conversions.py:211` | 1.67262192e-24 | 1.67262192369e-24 g | 2e-10 rel. — fine |
| `m_e` | `unit_conversions.py:214` | 9.1093837e-28 | 9.1093837015e-28 g | 2e-10 rel. — fine |
| `c` | `unit_conversions.py:217`, `default.param:259` | 29979245800 | 29979245800 cm/s | exact (SI-exact) |
| `sigma_SB` | `unit_conversions.py:220` | 5.670374e-5 | 5.6703744192e-5 | 7e-8 rel. — fine |
| `h` | `unit_conversions.py:223` | 6.62607015e-27 | 6.62607015e-27 erg·s | exact (SI-exact) |
| `e` (esu) | `unit_conversions.py:226` | 4.80320425e-10 | 4.80320425e-10 statC | exact; **unused** in the package |
| `_L_SUN_ERG_S` | `sps/sps_columns.py:33` | 3.828e33 | IAU 2015 nominal L☉ = 3.828e33 erg/s | exact |
| `caseB_alpha` | `default.param:248` | 2.59e-13 | case-B α_B at T = 10⁴ K, cm³/s | standard; and the code **warns** at load if `TShell_ion` leaves 8000–11000 K (`read_param.py:355-362`) |
| `C_thermal` | `default.param:255` | 6e-7 | Spitzer conduction coefficient, erg/s/cm/K^(7/2) | the Weaver+77 / El-Badry+19 value |
| `dust_sigma` | `default.param:236` | 1.5e-21 | dust cross-section per H at Z☉, cm² | WARPFIELD lineage value |

**Critically, no physical constant is written twice with two different values anywhere in the
package.** I grepped for every common spelling of pc→cm, Msun→g, Myr→s, m_H, m_p, k_B, G, L☉,
σ_SB, α_B, C_thermal, σ_dust: each appears exactly once, in either `unit_conversions.py` or the
`registry.py` parameter spec. This is the single most damning pattern the sweep was asked to hunt
and **it is absent**.

### Unit conversions — all 22 re-derived against `astropy.units`

Every field of `ConversionConstants` matched astropy to ≤ 2.2e-16 relative (i.e. float64 exact):
`cm2pc`, `km2pc`, `s2Myr`, `g2Msun`, `ndens_cgs2au`, `phi_cgs2au`, `E_cgs2au`, `L_cgs2au`,
`pdot_cgs2au`, `pdotdot_cgs2au`, `G_cgs2au`, `v_kms2au`, `v_cms2au`, `F_cgs2au`, `Pb_cgs2au`,
`k_B_cgs2au`, `c_therm_cgs2au`, `dudt_cgs2au`, `Lambda_cgs2au`, `tau_cgs2au`, `gravPhi_cgs2au`,
`grav_force_m_cgs2au`. The two derived module-level extras also check out:
`Pb_au2_KcmInv = 4686.6676` (comment says "≈ 4.6867e+03" ✓) and `Mdot_au2Msunyr = 1e-6` ✓.
The `km2pc` comment `# = cm2pc / 1e-5` is arithmetically right.

### Bonnor–Ebert constants — re-derived by integrating the isothermal Lane–Emden equation

| constant | file:line | code | re-computed | verdict |
|---|---|---|---|---|
| `OMEGA_CRITICAL` | `bonnorEbertSphere.py:78` | 14.04 | 14.0433 at ξ = 6.451 | correct |
| `XI_CRITICAL` | `bonnorEbertSphere.py:79` | 6.451 | ξ at Ω=14.04 is 6.4504 | correct |
| `M_DIM_CRITICAL` | `bonnorEbertSphere.py:87` | 15.70 | m(6.451) = 15.7051 | correct |
| `M_BONNOR_CRITICAL` | `bonnorEbertSphere.py:88` | 1.182 | max m_B = 1.18223 at ξ = 6.4493 | correct |
| series ICs `ξ²/6 − ξ⁴/120 + ξ⁶/1890`, `ξ/3 − ξ³/30 + ξ⁵/315` | `:216-217` | — | matches the standard isothermal Lane–Emden expansion term-by-term | correct |
| `c_s³ = M/m · G^1.5 · sqrt(4πρ_c)` | `:407-408` | — | algebraically identical to inverting `M = 4π m ρ_c a³`, `a = c_s/sqrt(4πGρ_c)` | correct |
| `T_eff` ↔ `c_s` round trip | `:431` vs `:606`, `:646` | — | `T = μ c_s²/(γ k_B)` and `c_s = sqrt(γ k_B T/μ)` are exact inverses | correct |

(The only BE problem is the *default* `densBE_Omega = 14.1` — MN-005.)

### Weaver / Rahner physics constants

| constant | file:line | verdict |
|---|---|---|
| `cool_alpha = 0.6`, `cool_beta = 0.8`, `cool_delta = -6/35` | `default.param:300-306` | the Weaver+77 self-similar exponents: `R ∝ t^(3/5)` ⇒ α = v t/R = 3/5; `P_b ∝ t^(-4/5)` ⇒ β = 4/5; δ = −6/35 matches the Eq-37 time exponent. Consistent. |
| `WEAVER_ENERGY_FRACTION = 5/11` | `get_InitPhaseParam.py:28` | Weaver+77 Eq 20, `E_th = (5/11) L_w t` for γ=5/3. Standard; written as an exact rational. |
| `2.5` × 3 in the bubble ODE | `bubble_luminosity.py:442-444` | γ/(γ−1) = 5/2 for a monatomic ideal gas; matches Weaver Eq 42-43 as implemented in WARPFIELD. |
| `constant = 25/4 · k_B/(μ_ion C)` in the ODE ICs | `bubble_luminosity.py:401-409` | I re-derived it: integrating `Ṁ·(5/2)k_BT/μ = 4πR²C T^(5/2) dT/dr` gives exactly `(25/4)(k_B/μC)·Ṁ/(4πR²) = d(T^(5/2))/dr`, hence `T = (const·Ṁ·ΔR/4πR²)^(2/5)` and `dT/dr = −(2/5)T/ΔR`. Self-consistent, and `T(r2Prime)` returns exactly `_T_INIT_BOUNDARY` as intended. |
| `a = 1.5·ṗ̇/ṗ`, `c = 0.75·ṗ·R1` (Rahner thesis A12) | `get_bubbleParams.py:123-124`, `:177-178`, `get_betadelta.py:251-252` | the same two coefficients appear at three sites and **agree exactly**; the forward (`cool_beta_to_Ebdot`) and inverse (`Ebdot_to_cool_beta`) forms are algebraically consistent. |
| `P_ram = L/(2πr²v)` | `get_bubbleParams.py:308` | correct: `Ṁv/(4πr²)` with `Ṁv = 2L/v` gives `L/(2πr²v)`. |
| `L_leak = γ/(γ−1)·(1−C_f)·4πR²·P_b·c_s` | `get_bubbleParams.py:284` | enthalpy flux through the open area; dimensionally lands in Msun·pc²/Myr³ with no conversion, as asserted. |
| `F_grav = G·mShell/R²·(mCluster + 0.5·mShell)` | `energy_phase_ODEs.py:220` | the 0.5 is the standard thin-shell self-gravity factor. |
| `Ṁ = ṗ²/(2L)`, `v = 2L/ṗ` | `get_InitPhaseParam.py:130-134`, `read_sps.py:214-215` | exact inversion of `L = ½Ṁv²`, `ṗ = Ṁv`. |
| `dt_phase0 = sqrt(3Ṁ/(4πρ_a v³))` | `get_InitPhaseParam.py:151` | Rahner thesis Eq 1.15 as cited; dimensionally correct in code units. |
| `MAX_VELOCITY_COLLAPSE = 500.0` and its comment "(~490 km/s)" | `phase_events.py:73` | 500/1.022712 = 488.9 km/s ✓ — comment and code agree. |
| `ADAPTIVE_FACTOR = 10**0.1` and its comment "(~1.26)" | 1b/1c/2 runners | 10^0.1 = 1.2589 ✓. `ADAPTIVE_THRESHOLD_DEX = 0.05` with "10^0.05 ≈ 1.12x" ✓ (1.1220). |

### Composition and thermodynamic-consistency chain (verified end to end)

`x_He = 0.1`, `Z_He = 2`, `Z_He_shell = 1` derive `mu_convert = 1+4x_He = 1.4`,
`mu_atom = mu_H/(1+x_He) = 14/11`, `mu_ion = mu_H/(2+x_He(1+Z_He)) = 14/23`,
`mu_mol = mu_H/(0.5+x_He) = 14/6`, `chi_e = 1+Z_He·x_He = 1.2`, `chi_e_shell = 1.1` — all
algebraically correct for the stated composition, and computed with exact `Fraction` arithmetic.
The ideal-gas conversions that consume them are consistent at **every** site I checked:
`n_H = P·mu_ion/(mu_convert·k_B·T)` (`bubble_luminosity.py:427`, `:673`, `:725`, `:778`, `:811`),
its inverse `P_HII = (mu_convert/mu_ion_shell)·n·k_B·T` (`run_energy_phase.py:214`,
`energy_phase_ODEs.py:54`, `run_momentum_phase.py:~432`), the shell inner-edge density
(`shell_structure.py:124`), and `ρ = n_H·mu_convert` (`bubble_luminosity.py:930`,
`mass_profile.py:126`). `get_soundspeed` (`operations.py`) correctly uses `mu_ion` above 1e4 K
and `mu_atom` below, and correctly round-trips code units → cgs → code units.

### CIE/non-CIE switch temperature `10**5.5`

Appears at four sites — `bubble_luminosity.py:65` (`_T_INTERFACE_BAND`), `:706` (`_CIEswitch`),
`cooling/net_coolingcurve.py:43` (`t[t <= 5.5]`), `:53` (`logT_CIE > 5.5`). **All four carry the
same value**, and the coupling is explicitly documented at `bubble_luminosity.py:60-65` with a
pinning test named (`test/test_fA_source_boost.py`). Not a finding — this is what a
deliberately-shared constant should look like, and it is the counter-example that makes the
absence of MN-005-style disagreement elsewhere credible.

### Not examined in depth (low physics relevance)

`_functions/simplify.py` (117 literals — snapshot array compression), `_input/sweep_parser.py`
(104 — string/filename parsing), `_output/trinity_reader.py`, `_output/show_run.py`,
`_output/cloudy/*` (deck generation), `_functions/logging_setup.py`. I scanned all of these for
floors/sentinels and for duplicated physical constants and found none that reach the physics;
their numerics are formatting, indices, and plotting tolerances.
