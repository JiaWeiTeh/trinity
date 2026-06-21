# Findings: the hybr implicit-phase stall is under-cooling, not a trigger threshold

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.

**Summary.** Under the default `betadelta_solver=hybr`, TRINITY runs stall in the
implicit energy phase and never reach momentum (0/6 of a regime-spanning config
set). This investigation — done clean-room, substrate-certified first, gated
against external physics (never assuming TRINITY/WARPFIELD/Weaver correct) — finds
the stall is **a physics-completeness signal, not a tunable-threshold bug**: the
modelled bubble retains far more energy than real bubbles, so **no cooling-balance
event exists to trigger on**, and the only physical end-of-energy-phase is
**geometric blowout**. The root fix (mixing-layer cooling) is the right direction
but requires integration into the solver, not a bulk energy sink.

Working doc with full method/gates: `PLAN.md`. Data: `data/c0_*_{st6,h0}.csv`.
Figures: `figures/`. All reproducible from committed harnesses.

---

## 1. The problem

The implicit→momentum hand-off fires on a single criterion
(`run_energy_implicit_phase.py:1095`): `(Lgain − Lloss)/Lgain < 0.05` — switch when
radiative cooling has nearly caught up with the *instantaneous* mechanical power.
Under hybr (the more-correct unbounded solver, vs legacy's clamped β∈[0,1]) the
ratio plateaus at ~0.3–0.5 and never approaches 0.05, so every run sits in implicit
to the 15 Myr cap. Late-time / stopping-fate outputs are therefore untrustworthy.

## 2. Method (one paragraph)

Clean-room redo: reuse the candidate menu + methodology, **quarantine all prior
numbers/verdicts**. *Certify the substrate before building on it.* Gate everything
against an independent oracle and external physics. Config span (6, regenerable):
`large_diffuse`, `simple_cluster`, `small_dense`, `midrange`, `pl2_steep` (steep
crux), `be_sphere` — 3 dex in cloud mass, all profiles, all sfe.

## 3. Results

**C0 — substrate is certified (sound).**
- `Lloss` is pure radiative (no PdV/velocity) — audited.
- `res_T0_struct` (solver T-residual) tight span-wide (≤0.13% median).
- `res_beta` (β↔Pb trajectory consistency) is finite-difference **truncation**, not
  a defect — proven by a 4× timestep refinement: median 6.65% → 1.74% (3.82×, ∝Δt).
- The adiabatic-Weaver null is infeasible (solver can't run with `Lloss≡0`), but the
  energy-phase retained fraction lands at ~0.42–0.44 vs Weaver 5/11 — the code
  faithfully reproduces the energy-*conserving* limit.
⇒ hybr introduced no bug; it *exposed* real behaviour.

**Physics — unanimous under-cooling (6/6).** Retained energy
`f_ret = Eb/∫Lmech dt` plateaus at **0.25–0.40** in every config and **never reaches
the observed/3D-sim band 0.01–0.1** (Lancaster+2021; El-Badry+2019; Geen+2021;
Orion [CII] Pabst+2020). TRINITY's energy-conserving Weaver/Rahner interior lacks
the turbulent fractal mixing-layer cooling that dominates real bubbles. Figure
`figures/fret_verdict`. β goes negative (re-pressurisation) at the ~3 Myr surge, but
the *compression/inflow* source `β+δ` rarely reaches its −0.4 trigger (δ offsets β —
see Follow-ups; `figures/beta_repressurization`).

**G0 — no cooling transition exists; only geometric blowout (6/6, unanimous).**
Harvested every candidate trigger vs the Eb-peak oracle (`harvest_h0.py`,
`figures/g0_divergence`):
- **F0 (current) and F1 (cumulative, any η) NEVER fire** — cooling never catches up
  even cumulatively. Not a metric-form problem.
- **F3 (force) never fires**; and `Pb ≡ P_HII` to machine precision (bubble–shell
  pressure continuity by construction) makes pressure-balance criteria degenerate.
- **F2 (instantaneous timescale) fires at t≈0** — an artifact (ignores Lgain
  replenishment).
- **The Eb-peak oracle barely exists** — 5/6 the bubble's Eb grows monotonically to
  t=6.
- **Only F4 (blowout, R2>rCloud) gives a physical transition**, at an epoch set
  purely by cloud size (0.01→3.66 Myr).
⇒ For these under-cooled bubbles the transition **is not a cooling/energy event**;
the F0 trigger tests for an event that does not occur. Figure
`figures/f0_pathology` shows the ratio plateauing far above 0.05 and bumping *up* at
the SN surge.

**Root-fix prototype — direction validated, naive implementation rejected.**
- Offline (`mixcool_whatif.py`): a mixing-layer sink `L_mix = θ·Lmech` at the
  literature θ≈0.25 brings `f_ret` into the observed band in all 6 configs.
- Dynamical bulk-sink injection (subtract θ·Lmech from `dEb/dt`) is **numerically
  non-viable**: it drives the conductive `dMdt<0` (no physical evaporation root), so
  hybr finds no root and the dt-shrink guard spins — the solver stalls. A proper
  mixing-layer cooling must be integrated **into the betadelta structure solve** (so
  β,δ are solved *with* it, keeping `dMdt>0`). The bulk-sink injection was reverted;
  production is unchanged.

**Follow-ups (post-G0).**
- **The dip is geometry, not thermal** (`figures/dip_mechanism`): the early cooling-ratio
  dip is an emission-measure turnover `Lloss ∝ n²V = (Pb/T0)²R2³` (rise = volume growth
  beats dilution; collapse = R2 dilutes n²), **not** gas entering the Λ(T) peak — `T0` stays
  3–8e6 K, far above the 1e5–1e6 K cooling-peak band the whole time. So the under-cooling
  root is that the interior is **too hot to radiate efficiently**; the mixing-layer fix is
  needed to *create* the ~1e5–1e6 K gas.
- **BEFORE/AFTER + legacy-vs-hybr** (`figures/before_after`, `figures/legacy_vs_hybr`): the
  legacy clamped-β solver crosses 0.05 at the first cooling episode (5/6 cross, 0.024–1.037
  Myr) and transitions; hybr's same dip recovers and never crosses. The dip diagnostic on
  legacy vs hybr shows **`T0` is ~identical** — the difference is entirely the β-clamp:
  legacy pinned to [0,1] keeps `Lloss` high (ratio→crossing), hybr's free β swings to +2..+4
  and `Lloss` collapses (ratio→recovery). So legacy's transition was a **constrained
  edge-root artifact of the clamp**, not extra cooling (consistent with C0: hybr finds the
  true root).
- **Leakage makes the cooling trigger fire — viably** (`data/leaktest/`): the WARPFIELD-style
  switch `log Lmech − log Lcool < 0.05` is the same family as F0 and doesn't fire at Cf=1
  (gap 0.145–0.292 dex); but its leakage term `Lcool = Lb + Lleak`, supplied via
  `coverFraction<1`, **does** fire it — at Cf=0.95 (5% leak) the ratio crosses 0.05 @ t=0.131
  and the run transitions, solver-healthy (unlike the bulk-sink). Caveat: leakage *vents* hot
  gas (advective), it does not *create* cool radiating gas — a different lever than mixing.
- **β+δ, not β alone** (`figures/betadelta_portrait`, `data/betadelta_summary.csv`): the
  interior-velocity source is `(β+δ)/t = −t dln n/dt`, inflow trigger `β+δ ≲ −0.4` (**not** β
  alone, **not** β+δ=0). β dives to −1.6 (re-pressurisation, 5/6) but β+δ crosses −0.4 in only
  large_diffuse (δ>0 offsets β); the resulting inflow is "real but cosmetic" (archive). The
  structure provides **no** transition threshold — the stall is a cooling-budget problem.
- **PdV does not belong in the trigger** (`figures/pdv_trigger`): the expansion-work term
  `Ẇ = Pb dV/dt = 4πR2²v Pb` is the *energy-driving mechanism* (reversible work on the shell),
  not a radiative loss; the transition is `Lloss→Lgain` (catastrophic cooling). Adding it gives
  `(Lgain−Lloss−Ẇ)/Lgain = normalised dEb/dt` = the **Eb-peak** criterion. Measured: `Ẇ≈0.43–0.46
  Lgain` (~half the input is *work*), so adding it pulls the ratio ~0.5→0.05–0.15 (nearly fires)
  but it still crosses 0.05 only in large_diffuse (@4.76 Myr, its Eb-peak); the other 5 never peak
  (Eb grows monotonically). It swaps one non-event for another *and* mis-frames the physics — it
  would call the bubble momentum-driven at its energetic peak, mid-drive. **Keep PdV out.**

## 4. Conclusion & recommendations

1. **The stall is physics, not a threshold.** Retuning 0.05→X is futile — there is
   no cooling-balance event in the hybr regime. Do not "fix" it by tuning the trigger.
2. **Pragmatic interim** (if completable runs are needed now): a profile-aware **F4
   blowout** transition (`R2 > rCloud`) — the only candidate that fires physically.
   Caveat: its epoch is geometric (near-instant for compact clouds); consider
   `R2 > k·rCloud` or a sustained criterion.
3. **Root fix — two viable levers for the missing loss.** (a) **Leakage**
   (`coverFraction<1`) is now *demonstrated* to fire the cooling trigger at Cf≈0.95,
   solver-healthy — but it *vents* hot gas (advective). (b) **Mixing-layer cooling**
   (Lancaster/El-Badry) integrated **into the bubble structure solve** (θ≈0.25 magnitude
   confirmed) — it *creates* the ~1e5–1e6 K radiating gas the dip analysis shows is
   missing. Both bring `f_ret` into the band and let a real transition exist; which is
   physically right depends on the true covering fraction. Note θ=const has no state
   dependence and naive `θ·Lmech` double-counts `L_cool`'s smooth interface — a faithful
   `L_mix` ties to interface area / mixing velocity / density.
4. **Paper-worthy as-is:** substrate certified; under-cooling quantified across the
   regime; the transition shown to be geometric, not thermal — earned from data
   against external observations/3D sims, the pre-registered "no single scalar works"
   outcome.

## 5. Artifacts (all committed, regenerable)
- Harnesses: `c0_consistency.py` (certify + f_ret), `harvest_h0.py` (G0 divergence),
  `mixcool_whatif.py` (root-fix calibration), `analyze_c0.py`, `heartbeat*.sh`.
- Data (canonical map in `data/README.md`): `data/c0_*_st6.csv` (C0/f_ret),
  `data/c0_*_h0.csv` (triggers/cooling/forces — the source of truth), `data/c0_*_legacy.csv`
  (BEFORE), `data/leaktest/` (leakage Cf sweep), `data/betadelta_summary.csv`, refinement CSV.
- Figures (13): `fret_verdict`, `beta_repressurization`, `cert_residuals`, `f0_pathology`,
  `g0_divergence`, `dip_drivers`, `betadelta_portrait`, `surge_coincidence`, `blowout_geometric`,
  `mixcool_rootfix`, `dip_mechanism`, `before_after`, `legacy_vs_hybr`, `pdv_trigger` — full HTML
  report in `transition_report.html` (14 figures).
- References (external anchors): Weaver+1977; Lancaster+2021 I/II; El-Badry+2019;
  Geen+2021; Pabst+2020; Mac Low & McCray 1988; Rahner+2017/2019 (see PLAN.md §8).
