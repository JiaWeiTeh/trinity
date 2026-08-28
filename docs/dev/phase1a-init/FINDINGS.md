# Phase-1a initialisation at sub-GMC scale — findings (compact probe)

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

**Status (2026-08-14):** ✅ shipped — artifact diagnosed and quantified; the fix (`0df441f` + `a944727`) is implemented, gated and **merged to `main` 2026-08-06**. The maintainer decision this doc was waiting on is made: the pre-registered G2 bar compared against stock at the instant phase 1a hands off — which this document's own finding says is the wrong reference — so it was re-sited to `|ΔR2| < 5%` at 1 Myr / end of run plus fate unchanged (`PLAN.md` §4), and every config passes it.

Investigation of why a TRINITY run at sub-GMC scale (`mCloud=300`, `sfe=0.01`,
`nCore=8.7e3 cm^-3`; a 0.15 pc / 2.1e4 yr H II region) crosses the observed radius
~30x too early at ~12x the observed velocity. Verdicts per question of the brief; all
run data committed as CSVs in `data/` (see `data/README.md` for the manifest and
`harness/README.md` for the exact commands).

## TL;DR

The **equations are broadly right; the discretisation of the first ~30 years is
wrong at this scale, and it alone produces the entire discrepancy.** The
`vd = -1e8` branch (`energy_phase_ODEs.py:269-270` @ bb94c78) plus the fixed
`SEGMENT_DURATION = 3e-5` Myr first segment injects a fixed, system-independent
momentum (~283 Msun pc/Myr for the probe) that exceeds the wind's cumulative
momentum input at that scale by a factor ~3e5. The compact-probe trajectory afterwards is
pure momentum-coasting on that artifact (p = m·v constant to 0.1% for the next
~3000 yr). At GMC scale the same artifact is real but forgiven within ~400 yr
(the wind re-supplies it quickly) — which is why the published validation never
saw it. A converged run (log-spaced-segment prototype, no hack) reproduces the
observed expansion velocity exactly (5.1 vs 5.0 km/s at the observed age) and
overshoots the observed radius by 28% — within the wind-strength uncertainty of
the effective-cluster substitution. TRINITY-as-discretised cannot currently
model this object; TRINITY-as-formulated essentially can.

## Reference numbers (all reproduced)

The brief's §3 numbers reproduce exactly with stock constants
(`data/m43_probe.csv`): first snapshot t=0.0115 yr, R2=4.2953e-5 pc,
v2=3656.20 km/s; second snapshot (30 yr) R2=6.722e-2 pc, v2=722.82 km/s,
mShell=0.3832 Msun; crosses R2=0.153 pc at t=620 yr with v=61.0 km/s; at
stop_t=1e5 yr, R2=0.6033 pc, v2=2.03 km/s; 128 snapshots. `P_HII == Pb` to all
printed digits at every snapshot (the known `n_IF_Str` min-cap,
`shell_structure.py:251` @ bb94c78 — load-bearing here, see Q4/extras).

## Verdicts

### Q1 — the initial condition is physically sound

`dt_phase0 = sqrt(3 Mdot / (4 pi rho v^3))` is exactly the time at which the
swept-up ambient mass equals the ejected wind mass — the standard end of free
expansion. Verified numerically: M_swept(r0)/M_ejected = 1.000 in both configs.
`r0 = v0 dt_phase0` is the free-streaming front radius at that moment and lands
at 1.135x the Weaver self-similar radius at the same age — consistent to the
accuracy such a seeding can have. `E0 = (5/11) L_w dt_phase0` is Weaver+77
Eq. 20 evaluated at t=dt_phase0, consistent with r0 (same self-similar
solution). `v0 = v_wind = 2L/pdot` is 1.89x the Weaver shell velocity at that
instant (0.6 R/t) — the correct order: at swept-mass = ejected-mass, momentum
balance puts the shell between v_w/2 and v_w. T0 is Weaver Eq. 37 at dt_phase0
(t in Myr, consistent with the coefficient's units).

**Falsifier:** an independent derivation of the free-expansion end giving a
different dt (e.g. swept mass = *momentum-weighted* mass) would change r0/E0 by
O(1) factors — it would not change any conclusion below (the artifact is 5 dex,
not O(1)).

### Q2 — v0 = v_wind not scaling with M* is correct physics

A wind's terminal velocity is a per-star property; the SB99 IMF average makes
it a (mass-independent) population property, ~3656 km/s at t=0. The M*
dependence of the IC is carried where it belongs: dt_phase0 ∝ sqrt(Mdot) ∝
sqrt(M*), r0 ∝ sqrt(M*), E0 ∝ M*^{3/2}. The *shell* velocity handed over,
v0 = v_w, is the free-streaming front velocity — legitimate *at t0*, because at
that instant the swept mass equals the (still fast-moving) wind mass. What is
NOT legitimate is keeping the shell at O(v_w) for 30 years afterwards (see Q4).
The Weaver attractor velocity at the segment-1 boundary is 83 km/s (compact probe) vs
649 km/s (GMC) — v0's mass-independence is fine, the *duration over which the
code lets it persist* is what breaks scale-invariance.

**Falsifier:** none needed — this is arithmetic on the SPS table plus the
free-expansion definition; both verified against `lib/default` values.

### Q3 — SEGMENT_DURATION (and TFINAL) must scale with the system; fixed values cannot converge

The physical timescale of the early energy phase is the expansion time
R/Rdot = (5/3)t (Weaver), whose *starting* value is t0 = dt_phase0. dt_phase0
spans 0.0115 yr (compact probe) to 1.96 yr (GMC control) across the two configs —
it scales as sqrt(M*/rho)/v_w^{3/2}. A fixed 30-yr first segment is therefore
2600 dt_phase0 at sub-GMC scale but only 15 dt_phase0 at GMC scale. Everything the
shell structure/feedback/P_HII snapshot freezes per segment changes on the
timescale ~t, so segments must satisfy dt ≲ eps * t (log-spaced), not a fixed
30 yr. TFINAL_ENERGY_PHASE = 3e-3 Myr is likewise absolute: at GMC scale it is
0.15% of the run; at sub-GMC scale it is 14% of the observed age, and the entire
observed epoch (2.1e4 yr) is handled by phase 1b whose DT_SEGMENT_* floors
(1e-4 Myr = 100 yr) are also absolute. TFINAL itself is a second-order knob:
with TFINAL=3e-4 (handoff at 300 yr instead of 2900) the 1a portion is
identical to baseline at matched t (`data/m43_tfinal3e-4.csv` — the artifact
lives in segment 1, not at the handoff), but the 1b continuation from the
artifact-loaded 300-yr state grinds at the DT floor with repeated
"beta-delta: no physical solution" warnings — the poisoned state also
stresses the 1b bubble solver.

Evidence that the fixed segment is *the* controlling parameter: varying it (and
nothing else) moves the segment-1 exit momentum across a factor >50
(`data/segment1_exit.csv`): SEG=3e-5 → p=283.3; SEG=1e-5 → p=117.9;
SEG=3e-6 → p=5.48; ablated hack (same SEG=3e-5) → p=1300. All four match the
closed-form hack kinematics p(SEG) = (4pi/3) rho R_exit^3 v_exit with
v_exit = v0 - 1e8*SEG and R_exit = r0 + (v0 - 5e7*SEG)*SEG (hack on), or
v_exit = sqrt(P_frozen/rho) (hack off). Tightening rtol/atol 100x changes the
trajectory by at most 1.1e-5 relative in v2 over the whole run
(`data/m43_tol1e-8.csv` vs baseline at identical snapshot times) — the
sensitivity is to the segment structure, not the ODE solver.

The cleanest single statement of the failure: the time at which the model
crosses the observed radius (0.153 pc) is a pure function of the numerical
segment length — 620 yr (SEG=3e-5, at 61 km/s), 1480 yr (SEG=1e-5), 11154 yr
(SEG=3e-6, at 6.2 km/s) — marching toward the physical answer (the adiabatic
Weaver crossing is ~1.2e4 yr at 7.6 km/s; observed 2.1e4 yr at 5.0 km/s) as
the discretisation refines.

**Falsifier:** a run with fixed 30-yr segments whose early trajectory matched a
run with 3-yr segments would refute this; the committed CSVs show the opposite.

### Q4 — the vd = -1e8 branch is a hard-coded relaxation hack, tuned for GMC scale, and is load-bearing

What it does (verified analytically and in `data/`): for exactly the first
segment, the RHS velocity derivative is replaced by -1e8 pc/Myr^2, so the shell
exits segment 1 with v = v0 - 1e8*SEGMENT_DURATION = 3739-3000 = 739 pc/Myr
(723 km/s) at R ≈ 0.067-0.075 pc, *independent of the system*. The true
|vd| at t0 is 3 v0^2/r0 ≈ 1e12 (compact probe) / 5.7e9 (GMC) pc/Myr^2 — the branch is
4-6 dex weaker than the physics it replaces, so it is not a stiffness guard in
any quantitative sense; it is a scripted linear coast-down. Its pairing with
SEGMENT_DURATION is fine-tuned: Δv = 1e8 * 3e-5 = 3000 pc/Myr ≈ 0.8 v0, and
the exit state (723 km/s at 0.075 pc) sits within a factor ~2 of the GMC
Weaver attractor at 30 yr (649 km/s, 0.033 pc) — for the GMC it is a crude but
serviceable "relax onto Weaver in one segment". For the compact probe the same fixed exit
state is 9x the attractor velocity at 16x the attractor radius, i.e. 3.4e4x the
attractor momentum.

It is *load-bearing*: ablating it (EarlyPhaseApproximation=False,
`data/m43_noapprox.csv`) does not recover the physical solution — it exposes a
*worse* artifact. The ODE snapshot freezes P_HII (== Pb via the min-cap) at its
t0 value (1.86e9 in code units, from E0 inside r0^3) for the whole first
segment; the true RHS then rides a pressure-regulated snowplow at
v = sqrt(P_frozen/rho) = 2430 km/s for 30 years (the run shows exactly this
velocity at snapshot 2), exiting with p = 1300 — 4.6x the hack's artifact. The
branch therefore *shields* segment 1 from the frozen-pressure catastrophe while
substituting its own smaller one. Removing it alone changes the probe's final
state at 1e5 yr from 0.603 pc to 0.931 pc (+54%) and is NOT a fix.

Scale sweeps (E3/E4) make the system-independence exact: across mCloud =
3e2..3e6 at fixed sfe (`data/mass_*.csv`) the segment-1 exit state is
(722.8 km/s, 0.067 pc) every time — only t0 and r0 carry the M* dependence
(both ∝ sqrt(M*), as the IC says they should) — and across the nCore range
(`data/ncore_*.csv`) the exit state is again identical while the artifact
momentum scales exactly as the ambient density (p = 283 * n/8.7e3: measured
120.6 at n=3.7e3 vs predicted 120.4). The branch injects hack kinematics, not
physics, at every point in the parameter space; it is only *fatal* where the
wind cannot quickly resupply p_artifact ∝ rho.

Fragility, measured: any SEGMENT_DURATION > v0/1e8 = 3.7e-5 Myr drives v2
*negative* during segment 1. With SEG=1e-4 the probe dies at t=42 yr via the
`velocity_runaway` terminal event (code 50) after writing one snapshot
(`data/m43_seg1e-4.csv`) — the branch and the segment length are a
fine-tuned pair, not independent knobs.

**Falsifier for the "tuned for GMC" reading:** the branch mattering at GMC
scale would refute it — `data/gmc_noapprox.csv` vs `data/gmc_control.csv` at
matched t: ΔR/R = 36% at 100 yr, 10% at 1e3 yr, 1.0% at 1e4 yr, 0.09% at
1e5 yr, 0.02% at 3e5 yr. The branch is irrelevant to published GMC-scale
results beyond ~1e4 yr; at sub-GMC scale it changes the whole run — and neither
variant is right there.

### Q5 — budgets close everywhere except segment 1, where they are violated by 4-5 dex

From `data/m43_probe.csv` (trapezoid impulses over snapshots): at the end of
segment 1 the shell momentum is 283 Msun pc/Myr vs a cumulative wind impulse of
9.8e-4 — a factor 2.9e5. (An energy-driven bubble legitimately amplifies wind
momentum by ~v_w/2Rdot; at this scale and time that factor is ~8 — the Weaver
attractor's own momentum at 30 yr is 8.2e-3 — so the artifact exceeds even the
amplified physical momentum by 3.4e4.) Kinetic energy: 2.0e48 erg vs 3.5e43 erg of
wind mechanical input — factor 5.7e4 (the first segment manufactures a small
supernova's worth of KE from nothing). From segment 2 onward the budget closes:
final p(1e5 yr) = 576 ≈ 283 (artifact) + 311 (pressure impulse) - 21 (gravity
impulse). The thermal energy Eb tracks L_w t minus PdV/cooling correctly
throughout. So the ODE integration is force-faithful; the *only* conservation
break is the scripted first segment — but its single injection dominates the
system's entire momentum budget for the rest of the run (the real forces
supply comparable impulse only by ~1e5 yr).

**Falsifier:** momentum budget failing to close in segments >= 2 would indicate
a second defect; it closes to trapezoid accuracy.

### Q6 — the code finds its own Weaver attractor late; agreement begins at a fixed *absolute* overshoot-decay time, not a fixed fraction of the run

GMC control vs adiabatic Weaver R(t) (`data/gmc_control.csv`): ratio 2.16 at
segment-1 exit (32 yr), 1.4 at 150 yr, 1.13 at 600 yr, 1.05 by 2.7e3 yr —
agreement to ~5% for the remaining 2 Myr. The compact probe *never* reaches its
attractor inside the observed epoch: still 6x the Weaver radius at 620 yr
(where it crosses the observed R), and by the time real impulse catches up
(~1e5 yr) the bubble is far beyond the observed object. The recovery time is
set by p_artifact/pdot_wind ∝ rho R_hack^3 v_hack / M* — a *physical* resupply
time, not a numerical one. It is ~4e-4 Myr for the GMC and ~8.7 Myr for the compact probe:
the artifact is forgiven at exactly the scales the code was validated on and
fatal at sub-GMC scale.

**Falsifier:** if the GMC run agreed with Weaver from t0 onward the overshoot
story would be wrong; the committed CSV shows the factor-2 early overshoot.

### Q7 — the effective-cluster substitution overestimates this object's wind by 1-3 dex, but that is not what broke the run

The Q-matched 3 Msun IMF-averaged population carries the IMF-averaged wind:
Mdot = 8.7e-9 Msun/yr at v_w = 3656 km/s (L_w = 3.7e34 erg/s) — a diluted
O-star wind. The real ionizer is HD 37061 Aa, B0.5V, Teff = 31.1 kK,
M = 16.4 Msun, log L/Lsun = 4.43 (Aschenbrenner & Przybilla 2024, A&A;
Simón-Díaz+ 2011), with *no measured wind*; Vink+2001 predicts ~1e-8.5 Msun/yr
at v_inf ~ 2000 km/s (L_w ~ 1e33), and the "weak-wind problem" for late-O/early-B
dwarfs puts observed rates 1-2 dex below that. So the substitution's L_w is
plausibly 10-1000x too strong and its v_w ~1.5-2.5x too fast. Because
R_Weaver ∝ L_w^{1/5}, even 2 dex of L_w only moves the wind bubble radius by
2.5x — it shifts the converged solution from ~0.21 pc (SB99 wind) to
~0.06-0.13 pc (realistic B-star wind, L_w ~ 1e32-3.7e33 erg/s) at the observed
age, i.e. *brackets* the observed 0.153 pc. None of Q1-Q6 change under any wind choice in this range
(the artifact is 5 dex, the wind uncertainty moves things by <0.5 dex in R).
Also note Q-matching under-represents the bolometric output (system
log L/Lsun ≈ 4.5 vs the effective cluster's ~4.2) — irrelevant here (radiation
pressure is subdominant), but worth remembering for dustier objects.

## Numerics vs physics

Numerics (does NOT survive convergence): the entire early compact-probe trajectory —
R(t), v(t), shell momentum and KE for the first >=1e4 yr — is set by the
segment-1 artifact and changes by factors of 3-200 under purely numerical
knobs (SEGMENT_DURATION, the -1e8 branch). Physics (survives): the IC values
(t0, r0, E0, T0), the Weaver attractor the trajectories relax toward, the
momentum-coasting behaviour once forces are negligible, and the budget closure
from segment 2 on.

The physical prediction TRINITY *does* make at sub-GMC scale when converged
(measured, `data/m43_logseg.csv`): a Weaver-like wind bubble at R2 = 0.196 pc,
v2 = 5.1 km/s at the observed age with the SB99 effective-cluster wind —
velocity exactly observed, radius +28% (and R ∝ L_w^{1/5} puts the observed
radius inside the Q7 wind range). The independent Spitzer D-type solution for
the observed Q and density passes through the observed point too (0.154 pc,
3.7 km/s at 2.1e4 yr for c_i=10 km/s). The compact-probe comparison failing is a
*discretisation* failure, not an equations failure.

## Independent corroboration — code-audit "Cluster C" (2026-08-04)

The `bugfix/code-audit` branch reached the same defect independently
(its Cluster C, `MN-001`/`DD-005`, CONFIRMED S1 in that branch's
`docs/dev/code-audit/data/resolutions.md` — "the audit's largest measured
trajectory error"). Cross-checks, and facts adopted from it (audit line refs
are theirs, not re-verified here):

- **Exact mechanism for the 722.8 km/s invariant** (supersedes the
  phenomenological reading above): with `vd` frozen the R2/v2 subsystem
  decouples and a constant RHS integrates exactly, so
  `v_exit = v0 − 1e8·SEGMENT_DURATION = 3739.2407 − 3000 pc/Myr =`
  **722.82 km/s**, and `v0 = 2L_w/ṗ_w` is mass-scale invariant. Matches
  `data/segment1_exit.csv` (722.8) and the E3/E4 sweeps to 5 figures; the
  audit verified the v0 invariance over f_mass 0.001-10. Corollary of the
  closed form: v_exit is *linear in SEGMENT_DURATION* (why the E1 seg sweep
  moves it) and *independent of solver tolerance* (audit: 4000x step-count
  change moves it 2e-12; my `m43_tol1e-8` ≡ baseline).
- **Displacement A/B reproduced:** their on-vs-off ΔR2 at segment-0 end,
  −10.1%, matches −9.9% from `gmc_control.csv` vs `gmc_noapprox.csv`. Later-t
  magnitudes are config-dependent (theirs on `simple_cluster`: −30.2% peak,
  −19.3% at 1a exit; mine: −21.5% peak, −3.3% at 2.91e-3 Myr, +1.6% by
  8e-3 Myr) — quote neither pair as universal.
- **The hack partially cancels the frozen-driving error.** Audit noted the
  displacement moves the trajectory *toward* Weaver; the converged reference
  proves it: at 3e-4 Myr, logseg R2 = 0.1437 pc vs hack-on 0.1631 vs hack-off
  0.2079. Consistent with E2 (ablating the hack alone is worse). Removing the
  override without fixing the segment schedule is NOT a fix.
- **Leakage half (adopted, not re-verified):** `EarlyPhaseApproximation`
  defaults `True` (`registry.py:423`), is absent from `default.param` (not
  user-disableable), and is cleared at one site
  (`run_energy_phase.py:342-343`), `loop_count == 0`-guarded and placed
  *after* the event check — several segment-0 exits skip the clear, and a
  documented validator-free config (`cooling_boost_mode theta_target` +
  `cooling_boost_theta 0.96`) leaks `vd=-1e8` into 1b/1c, which then writes
  `VELOCITY_RUNAWAY` as the stopping fate. Any fix must clear the flag on
  *all* exit paths.
- **Diagnostics mask the override:** `compute_derived_quantities` has no
  `vd=-1e8` branch, so snapshot force budgets are the physical ones while the
  trajectory is not — which is exactly why the Q5/Q6 budget audit above looks
  clean (it audits the RHS, not the applied override). Nothing in the output
  marks the disagreement.
- **Propagation channel:** the 1a exit state becomes 1b's similarity exponent
  `cool_alpha = t·v2/R2` (`run_energy_implicit_phase.py:662`): 0.4557 (hack
  on) vs 0.3269 (off) — 39.4% apart at the handoff.
- **Provenance:** the constant entered under commit `bf50e44` ("plotting
  scripts for runtime") with no comment, units, reference, or test. Phase 2
  is *not* a consumer (own `MomentumODESnapshot`); consumers are 1a/1b/1c.

## What should change (minimal), and what it costs

**Proposal (prototyped, not implemented in production):** make phase-1a
segments log-spaced — `dt_seg = eps * (t_now - tSF)` with eps ≈ 0.1, from t0
onward — and delete the `vd = -1e8` override (plus the
`EarlyPhaseApproximation` switch that exists only to serve it). No change to
the IC, the RHS, the events, or phase 1b. The per-segment freezing of
P_HII/shell structure then has bounded staleness (a fixed fraction eps of the
expansion time) at every scale, which is exactly the property the fixed 30-yr
segment lacks.

Demonstrated with a zero-production-line prototype (`TRIN_LOGSEG=0.1
TRIN_NO_EARLY_APPROX=1` in `harness/patched_runner.py`, which substitutes an
object for the SEGMENT_DURATION constant):

- **compact probe** (`data/m43_logseg.csv`): tracks the adiabatic Weaver solution
  from the very first segment — R/R_Weaver = 1.25 max during the
  free-streaming relaxation, 1.07 by 1.4 yr, 1.00 by 160 yr, 0.90 by 2.4e4 yr
  (mild sub-adiabatic drift; cooling/PdV). v2 decays smoothly 3656 → 27 km/s
  by 410 yr with no manufactured momentum (p = 0.28 vs the baseline's 283 at
  the same age). The stiff v_w → attractor relaxation integrates with
  **zero solve_ivp failures**. **At the observed age the converged model gives
  R2 = 0.196 pc, v2 = 5.1 km/s** (t=2.1e4 yr; 0.174 pc / 5.6 km/s at the
  1.7e4-yr lower bound) vs observed 0.153 (0.142-0.164) pc and 5.0 (4.5-6.6)
  km/s — the velocity is dead-on and the radius is +28%, well inside the Q7
  wind-strength uncertainty (R ∝ L_w^{1/5}: 28% ≡ a factor 3.4 in L_w). It
  crosses 0.153 pc at 1.35e4 yr (vs the baseline's 620 yr).
- **Cost:** 131 segments / 2m34s for phase 1a vs 97 segments / 1m33s stock
  (same container, contended) — runtime-neutral at the run level. At GMC scale
  log-spacing gives *fewer* segments than stock (ln(3e-3/2e-6)/ln(1.1) ≈ 77).
  **Superseded 2026-08-05 by an uncontended production measurement** (rows
  `perf,*` in `data/gate_results.csv`): on `param/simple_cluster.param` to
  `stop_t=0.1`, each arm run alone on the container, the fix is **faster** —
  12m18s vs 14m37s total (−16%), 2m18s/96 segments vs 2m26s/97 segments in
  phase 1a (−6%), 10m00s vs 12m11s in phase 1b (−18%). The 1b saving is the
  larger one and is not a segment-count effect: the fix hands off to 1b with
  `v2_ODE/v2_alpha = 1.055` where stock hands off at `1.317`, i.e. much closer
  to the self-consistent state 1b then has to iterate towards. A better exit
  state is cheaper to continue from.
- **Large-object equivalence:** `data/gmc_logseg.csv` vs `data/gmc_control.csv`
  at matched t: ΔR/R = -29% at 100 yr (the stock run's overshoot), -3.8% at
  1e3 yr, -0.95% at 3e3 yr, -0.28% at 1e4 yr, **-0.04% at 8e4 yr** — the
  trajectories converge to the same Weaver attractor; differences are confined
  to the early transient where the stock run is the wrong one (its factor-2
  overshoot). The change *improves* the transient and preserves the asymptote.
  Gate for shipping (CLAUDE.md rule 5): this is NOT a "free win" —
  bit-identity is impossible (segment boundaries move) — so the gate is
  full-run trajectory equivalence at matched t on `param/simple_cluster.param`
  + the `docs/dev/performance/f1edge_{lowdens,hidens}` configs, with the
  acceptance bar set by the GMC-attractor agreement above (sub-% beyond the
  first few kyr), plus full `pytest`.

**The gate has now been run (2026-08-04) — full results in `data/gate_results.csv`.**
Headline: the schedule reproduces the converged reference *exactly* where one
exists (production vs prototype at compact-probe and GMC scale: worst rel diff 2.3e-8 in
R2, 1.3e-7 in v2, at identical t), `phase1a_segFrac = 0` reproduces stock
**byte-identically** (G1a) and the committed ablation baselines to 1e-15 (G1b),
and eps convergence passes (0.1 -> 0.03 moves R2 at the observed age by 0.11%).
But the **pre-registered "sub-% beyond the first few kyr" bar is NOT met at
t = 3e3 yr on any config** — measured at matched t: simple_cluster −10.4%,
f1edge_lowdens +1.7%, f1edge_hidens −22.8%, ordering by core density exactly as
the mechanism predicts. All three decay monotonically (lowdens is sub-1% by 1e4
yr, simple_cluster by ~3.5e4 yr, hidens only in the last 3% before it
collapses). Stopping fates are unchanged everywhere, though hidens collapses
28% later than stock and from `transition` rather than `momentum`. **So this
change is not free for the published regime**, and whether to accept the
early-phase shift — or re-site a bar that was written to measure at the instant
phase 1a hands off — is a maintainer decision, recorded rather than
self-approved.

**Resolved 2026-08-05 — the bar was re-sited, by maintainer sign-off.** The
adopted G2 bar (PLAN §4) is `|ΔR2| < 5%` at 1 Myr **or the end of the run if
it terminates earlier**, *and* the stopping fate unchanged. All four configs
pass; the worst is `f1edge_hidens` at +0.44%, 11x inside. (The threshold was
adopted at 10% and tightened to 5% the same day, after the measurements and
without re-running any of them: at 10% the bar sat ~23x above the worst config,
loose enough to let a future regression through.) Two facts that only
appeared once both long configs were run to their **true** natural end —
earlier fixed-arm GMC runs had been SIGTERM-truncated at 8.2e4 yr and
misreported as ending there:

- The trajectories do not merely stay inside a tolerance, they **converge**.
  GMC control ΔR2: −28.8% @100 yr → −0.95% @3e3 → −0.28% @1e4 → −0.037% @8e4 →
  −0.002% @1 Myr → **−0.001% @2 Myr**, with Δv2 +0.014% at 2 Myr.
  `simple_cluster` reaches −0.078% at 1 Myr. The disagreement is confined to
  the early transient, which is the part this document argues stock gets wrong.
- The change is **16% faster** end-to-end (14m37s → 12m18s on
  `simple_cluster` to `stop_t=0.1`, each arm alone on the container), almost
  all of it in phase 1b, which the change does not touch. Stock enters 1b with
  `v2_ODE/v2_alpha = 1.3167`, the fix with `1.0546` — a 1a exit state that is
  already close to α-consistent is cheaper for 1b to continue from.

Note for the record: the GMC control passes the *original* 1%-at-3e3-yr bar
(−0.949%) and is the only config that does — i.e. that bar was met by exactly
the one scale the `vd = -1e8` constant was tuned for.

**Implementation is specified in `PLAN.md`; its §3 decisions were settled on
2026-08-04** — schedule uncapped with `phase1a_segFrac = 0` as the fixed-segment
fallback, override *and* the `EarlyPhaseApproximation` flag deleted outright
(no consumers outside `trinity/`, so one column leaves `dictionary.jsonl`),
landing as two commits so the schedule plumbing can be proven byte-identical
before the physics changes. One refinement to the "not a free win" line above:
the *plumbing* half is provably free (byte-identical at `phase1a_segFrac = 0`) and
the *deletion* half re-derives the committed ablation baselines
(`data/*_noapprox.csv`, 2429 km/s) — only the combined change needs the
full-run gate.

Not part of the minimal change, but required before TRINITY output at the compact probe
scale is *quantitatively* trustworthy (see Extra findings): the
`n_IF_Str`/P_HII min-cap (P_HII == Pb identically) and phase 1b's absolute
DT floors. *(2026-08-14: the first of those two is done — C3c (`c43a50e`) took P_HII off the
capped Strömgren density entirely. Phase 1b's DT floors are still open.)*

## Extra findings (not in the brief's list)

1. **The stale-pressure ratchet.** ✅ **RESOLVED 2026-08-14 by C3c (`c43a50e`) — see the note
   below before using this finding.** `P_drive = max(Pb_live, P_HII_frozen)` with
   P_HII == Pb (min-cap) means every segment's driving pressure cannot fall
   below its segment-start Pb. In any segment where Pb declines (all early
   segments), the shell is driven by the *stale* pressure for the whole
   segment. This is mild at GMC scale (Pb changes ~10%/segment) and
   catastrophic in a compact-probe segment 1 (Pb falls ~7 dex within the segment). The
   min-cap is a separate known issue, but its *interaction with per-segment
   freezing* is what makes the no-hack ablation blow up — worth knowing before
   anyone "fixes" the -1e8 branch by deletion.

   **Resolution.** `phii-identity`'s C3c replaced the capped-Strömgren `P_HII` with a confinement
   regime switch that returns exactly `0.0` on the confined branch, and the confined branch is the
   one that fires on 100% of energy and implicit rows. `P_HII_frozen` is therefore `0.0` and
   `max(Pb_live, 0) == Pb_live`: the driving pressure follows `Pb` down within a segment and the
   ratchet is gone in phases 1a/1b. Per-segment freezing of the *shell structure* is unchanged —
   only the pressure term that made freezing ratchet is. The same C3c change is why two of this
   workstream's goldens moved; `docs/dev/phii-identity/PLAN.md` carries the mechanism.
2. **The SPS cubic-interpolation worry is a non-issue.** Feedback interpolants
   are flat to <1% over 0-0.1 Myr despite the duplicated t=0 row (checked for
   fLmech_W, fpdot_W, fQi, fLbol at compact-probe mass scaling).
3. **Phase 1b's absolute floors are adequate at sub-GMC scale — once 1a hands over
   a sane state.** DT_SEGMENT_MIN = 100 yr and the 3e-3 Myr handoff are
   absolute, and most of the compact-probe observed epoch is integrated by 1b; but the
   log-spaced prototype run (`data/m43_logseg.csv`), which enters 1b on the
   Weaver attractor, stays on it through 1b (R/R_Weaver = 0.95 at 5.7e3 yr) —
   so 1b's resolution is not the binding problem; 1a's segment-1 artifact is.

## Reproduction

Every run: `python docs/dev/phase1a-init/harness/patched_runner.py <param>`
with env overrides per `harness/README.md`, from the repo root. Stock-constant
runs used `python run.py <param>`. Runtime ~5-15 min/run single-core.
