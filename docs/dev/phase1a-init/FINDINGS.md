# Phase-1a initialisation at sub-GMC scale — findings (M43 probe)

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

**Status (2026-08-04):** 🔵 actionable — early-phase artifact diagnosed and quantified; minimal fix proposed, not yet implemented.

Investigation of why a TRINITY run at M43 scale (`mCloud=300`, `sfe=0.01`,
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
momentum input at that scale by a factor ~3e5. The M43 trajectory afterwards is
pure momentum-coasting on that artifact (p = m·v constant to 0.1% for the next
~3000 yr). At GMC scale the same artifact is real but forgiven within ~400 yr
(the wind re-supplies it quickly) — which is why the published validation never
saw it. A converged trajectory (the Weaver solution, which the code's own
attractor matches) passes within ~35% of the M43 observation; a Spitzer D-type
solution passes through it. TRINITY-as-discretised cannot currently model this
object; TRINITY-as-formulated very nearly can.

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
The Weaver attractor velocity at the segment-1 boundary is 83 km/s (M43) vs
649 km/s (GMC) — v0's mass-independence is fine, the *duration over which the
code lets it persist* is what breaks scale-invariance.

**Falsifier:** none needed — this is arithmetic on the SPS table plus the
free-expansion definition; both verified against `lib/default` values.

### Q3 — SEGMENT_DURATION (and TFINAL) must scale with the system; fixed values cannot converge

The physical timescale of the early energy phase is the expansion time
R/Rdot = (5/3)t (Weaver), whose *starting* value is t0 = dt_phase0. dt_phase0
spans 0.0115 yr (M43 probe) to 1.96 yr (GMC control) across the two configs —
it scales as sqrt(M*/rho)/v_w^{3/2}. A fixed 30-yr first segment is therefore
2600 dt_phase0 at M43 scale but only 15 dt_phase0 at GMC scale. Everything the
shell structure/feedback/P_HII snapshot freezes per segment changes on the
timescale ~t, so segments must satisfy dt ≲ eps * t (log-spaced), not a fixed
30 yr. TFINAL_ENERGY_PHASE = 3e-3 Myr is likewise absolute: at GMC scale it is
0.15% of the run; at M43 scale it is 14% of the observed age, and the entire
observed epoch (2.1e4 yr) is handled by phase 1b whose DT_SEGMENT_* floors
(1e-4 Myr = 100 yr) are also absolute.

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

**Falsifier:** a run with fixed 30-yr segments whose early trajectory matched a
run with 3-yr segments would refute this; the committed CSVs show the opposite.

### Q4 — the vd = -1e8 branch is a hard-coded relaxation hack, tuned for GMC scale, and is load-bearing

What it does (verified analytically and in `data/`): for exactly the first
segment, the RHS velocity derivative is replaced by -1e8 pc/Myr^2, so the shell
exits segment 1 with v = v0 - 1e8*SEGMENT_DURATION = 3739-3000 = 739 pc/Myr
(723 km/s) at R ≈ 0.067-0.075 pc, *independent of the system*. The true
|vd| at t0 is 3 v0^2/r0 ≈ 1e12 (M43) / 5.7e9 (GMC) pc/Myr^2 — the branch is
4-6 dex weaker than the physics it replaces, so it is not a stiffness guard in
any quantitative sense; it is a scripted linear coast-down. Its pairing with
SEGMENT_DURATION is fine-tuned: Δv = 1e8 * 3e-5 = 3000 pc/Myr ≈ 0.8 v0, and
the exit state (723 km/s at 0.075 pc) sits within a factor ~2 of the GMC
Weaver attractor at 30 yr (649 km/s, 0.033 pc) — for the GMC it is a crude but
serviceable "relax onto Weaver in one segment". For M43 the same fixed exit
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

Fragility note: any SEGMENT_DURATION > v0/1e8 = 3.7e-5 Myr drives v2 *negative*
during segment 1 [verify with data/m43_seg1e-4.csv].

**Falsifier for the "tuned for GMC" reading:** the branch mattering at GMC
scale would refute it — `data/gmc_noapprox.csv` vs `data/gmc_control.csv` shows
the GMC trajectory with and without the branch converging to the same attractor
within ~3% by 2700 yr.

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
agreement to ~5% for the remaining 2 Myr. The M43 probe *never* reaches its
attractor inside the observed epoch: still 6x the Weaver radius at 620 yr
(where it crosses the observed R), and by the time real impulse catches up
(~1e5 yr) the bubble is far beyond the observed object. The recovery time is
set by p_artifact/pdot_wind ∝ rho R_hack^3 v_hack / M* — a *physical* resupply
time, not a numerical one. It is ~4e-4 Myr for the GMC and ~8.7 Myr for M43:
the artifact is forgiven at exactly the scales the code was validated on and
fatal at M43 scale.

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

Numerics (does NOT survive convergence): the entire early M43 trajectory —
R(t), v(t), shell momentum and KE for the first >=1e4 yr — is set by the
segment-1 artifact and changes by factors of 3-200 under purely numerical
knobs (SEGMENT_DURATION, the -1e8 branch). Physics (survives): the IC values
(t0, r0, E0, T0), the Weaver attractor the trajectories relax toward, the
momentum-coasting behaviour once forces are negligible, and the budget closure
from segment 2 on.

The physical prediction TRINITY *would* make at M43 scale if converged: a
Weaver-like wind bubble reaching ~0.1-0.2 pc at 5-8 km/s at the observed age
(depending on wind strength within the Q7 range), transitioning to
photoionized-pressure (Spitzer D-type) driving — which passes through the
observed (0.153 pc, 5 km/s, 2.1e4 yr) point for the observed density. The M43
comparison failing is a *discretisation* failure, not an equations failure.

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

- **M43 probe** (`data/m43_logseg.csv`): tracks the adiabatic Weaver solution
  from the very first segment — R/R_Weaver = 1.25 max during the
  free-streaming relaxation, 1.10 by 0.5 yr, 1.00 by 160 yr. v2 decays
  smoothly 3656 → 27 km/s by 410 yr with no manufactured momentum
  (p = 0.28 vs the baseline's 283 at the same age). The stiff v_w →
  attractor relaxation integrates with **zero solve_ivp failures**.
- **Cost:** 131 segments / 2m34s for phase 1a vs 97 segments / 1m33s stock
  (same container, contended) — runtime-neutral at the run level. At GMC scale
  log-spacing gives *fewer* segments than stock (ln(3e-3/2e-6)/ln(1.1) ≈ 77).
- **Large-object equivalence:** `data/gmc_logseg.csv` vs `data/gmc_control.csv`
  — the trajectories converge to the same Weaver attractor; differences are
  confined to the first ~2.7e3 yr where the stock run carries its factor-2
  overshoot (i.e. the change *improves* the transient and preserves the
  asymptote). Gate for shipping (CLAUDE.md rule 5): this is NOT a "free win" —
  bit-identity is impossible (segment boundaries move) — so the gate is
  full-run trajectory equivalence at matched t on `param/simple_cluster.param`
  + the `docs/dev/performance/f1edge_{lowdens,hidens}` configs, with the
  acceptance bar set by the GMC-attractor agreement above (sub-% beyond the
  first few kyr), plus full `pytest`.

Not part of the minimal change, but required before TRINITY output at M43
scale is *quantitatively* trustworthy (see Extra findings): the
`n_IF_Str`/P_HII min-cap (P_HII == Pb identically) and phase 1b's absolute
DT floors.

## Extra findings (not in the brief's list)

1. **The stale-pressure ratchet.** `P_drive = max(Pb_live, P_HII_frozen)` with
   P_HII == Pb (min-cap) means every segment's driving pressure cannot fall
   below its segment-start Pb. In any segment where Pb declines (all early
   segments), the shell is driven by the *stale* pressure for the whole
   segment. This is mild at GMC scale (Pb changes ~10%/segment) and
   catastrophic in an M43 segment 1 (Pb falls ~7 dex within the segment). The
   min-cap is a separate known issue, but its *interaction with per-segment
   freezing* is what makes the no-hack ablation blow up — worth knowing before
   anyone "fixes" the -1e8 branch by deletion.
2. **The SPS cubic-interpolation worry is a non-issue.** Feedback interpolants
   are flat to <1% over 0-0.1 Myr despite the duplicated t=0 row (checked for
   fLmech_W, fpdot_W, fQi, fLbol at M43 mass scaling).
3. **Phase 1b's absolute floors bind at M43 scale.** DT_SEGMENT_MIN = 100 yr
   and ODE_MAX_STEP = 20 yr are ~0.5% of a GMC run but O(1)% of the M43
   object's age per step; the 1b handoff at 3e-3 Myr means most of the M43
   observed epoch is integrated by 1b at its floor resolution. [quantify with
   data/m43_1bdt.csv]

## Reproduction

Every run: `python docs/dev/phase1a-init/harness/patched_runner.py <param>`
with env overrides per `harness/README.md`, from the repo root. Stock-constant
runs used `python run.py <param>`. Runtime ~5-15 min/run single-core.
