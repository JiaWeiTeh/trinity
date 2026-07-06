# H2 audit: is "breaking rCloud = fail" real, or an artificial boundary whose removal would let the run continue self-consistently and the cooling trigger fire?

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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Date:** 2026-06-22. Branch `fix/transition-trigger-problem-pt4`. Read-only audit of
production; all line refs verified against current source on this date.

---

## TL;DR verdict

H2 (maintainer's hypothesis) is **partially right and importantly wrong**, and the
correct picture is *more useful* than either framing:

1. **rCloud is DERIVED, not free**, and **cannot be set directly** — it is computed from
   `(mCloud, nCore, alpha/Omega, rCore)`. You enlarge it only by enlarging the cloud's
   *mass budget*. So "make rCloud infinitely large without disturbing other physics" is
   **not literally possible via a knob** — but you CAN enlarge it cleanly by raising
   `mCloud` at fixed `nCore` and lowering `sfe` to hold the cluster (feedback) mass fixed.
2. **"Breaking rCloud" is NOT a failure in code.** `R2 > rCloud` is the **clean
   phase-1a→1b hand-off** (`is_simulation_ending = False`), and the only termination it
   maps to (`RCLOUD_BOUNDARY`) is in the **clean 0–9 exit band**. The prior FINDINGS framing
   "breaking rCloud = fail" is **wrong as stated**: the run does not error/NaN at rCloud and
   the implicit phase deliberately runs *past* it (up to 100–500× rCloud).
3. **The density DOES drop at rCloud** (to `nISM`, a 10²–10⁵× cliff), and the cooling
   collapse IS geometric — but the collapse that *kills the trigger* happens **AFTER
   blowout**, as `R2` runs out to 100s of pc into the rarefied ISM and the shell stops
   sweeping dense gas (Pb falls, interior dilutes, `Lloss ∝ n²` craters). The closest the
   trigger ever gets to 0.05 is **right at the cloud edge** (R2/rCloud ≈ 0.7–1.1), and it
   *recovers* afterwards.
4. **Would enlarging rCloud make the trigger fire?** The empirics say **no, not on its
   own** — even at the in-cloud cooling-ratio *minimum* (shell still inside the dense
   cloud) the ratio floors at **0.28–0.49** across all six configs, ~6–10× above the 0.05
   trigger. Keeping the shell in dense gas longer holds `Lloss/Lgain` near that floor for
   longer, but does not push it through 0.05. **So H2 and the cooling stall are the same
   phenomenon viewed geometrically (correct), but the geometric fix alone does not create a
   cooling-balance event (the under-cooling is intrinsic to the energy-conserving
   interior).** This is consistent with — and sharpens — the cleanroom FINDINGS.

---

## 1. Is rCloud derived or free? What sets it? (file:line)

**Derived, every time.** There is no `rCloud` input key; `rCloud` is a `derived_init`,
`run_const` param (`trinity/_input/registry.py:393` — `ParamSpec(name='rCloud',
default=0, ... category='derived_init', ... run_const=True)`). It is computed at init:

- `trinity/main.py:122` calls `get_InitCloudProp.get_InitCloudProp(params)`, which sets
  `params['rCloud'].value` (`trinity/phase0_init/get_InitCloudProp.py:277, 339`).
- Power-law (`alpha=0`): `rCloud = (3 M / (4π ρ_core))^(1/3)`
  (`powerLawSphere.py:51-74`, `compute_rCloud_homogeneous`).
- Power-law (`alpha≠0`): analytical inversion of the enclosed-mass formula
  (`powerLawSphere.py:77-211`, `compute_rCloud_powerlaw`; Rahner+2018 Eq. 25).
- Bonnor-Ebert: `create_BE_sphere(...).r_out` (`get_InitCloudProp.py:324-333`).

So rCloud is a pure function of `(mCloud, nCore, rCore, alpha)` or `(mCloud, nCore,
Omega)`. **The only way to enlarge it is to change those.**

`rCloud_max` (`default.param:87`, default **200 pc**) is a pre-run plausibility cap
(`validate_gmc.py:111, 230-234`): if the *computed* rCloud exceeds it, the run is rejected
as too diffuse. It is a validation gate, not a physics boundary, and is itself a tunable
`.param` knob (`registry.py:313`).

Authoritative computed rCloud for the six cleanroom configs (via the production pipeline,
`h2_rcloud_compute.py` → `h2_rcloud_edge.csv`; matches the hardcoded table in
`cleanroom/harvest_h0.py:34`):

| config | profile | rCloud [pc] | nCore [cm⁻³] | nEdge [cm⁻³] | nISM [cm⁻³] | nEdge/nISM drop |
|---|---|---|---|---|---|---|
| simple_cluster | PL α=0 | 1.69 | 1e5 | 1e5 | 1 | **1e5** |
| pl2_steep | PL α=−2 | 21.35 | 1e5 | 219 | 1 | **219** |
| small_dense_highsfe | PL α=0 | 0.326 | 1e6 | 1e6 | 1 | **1e6** |
| midrange_pl0 | PL α=0 | 8.53 | 1e4 | 1e4 | 1 | **1e4** |
| large_diffuse_lowsfe | PL α=0 | 88.05 | 1e2 | 1e2 | 1 | **100** |
| be_sphere | BE Ω=14 | 15.5 | 1e4 | 714 | 1 | **714** |

## 2. The density profile at r > rCloud (the crux). Exact code.

`trinity/cloud_properties/density_profile.py` docstring (lines 9-21) and code
(lines 128-164):

```
n(r) = nISM                     for r > rCloud         # PL, all alpha
n(r) = nCore*(r/rCore)^alpha    for rCore < r <= rCloud
```
The cliff is smoothed by a tanh bridge of width `SMOOTH_FRAC = 0.01 * rCloud`
(`density_profile.py:128-130, 148, 164`):
```python
SMOOTH_FRAC = 0.01
delta = SMOOTH_FRAC * rCloud
w_outside = 0.5 * (1.0 + np.tanh((r_arr - rCloud) / delta))
...
n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside
```
So **beyond rCloud the gas density collapses to `nISM` (default 1 cm⁻³)** over ~1% of
rCloud. Measured drop (production `get_density_profile`, sampled 0.97·rCloud → 1.10·rCloud,
`h2_rcloud_edge.csv`): a **factor of 100 to 10⁵** depending on config (table above). The
enclosed-mass profile mirrors this: for `r > rCloud`,
`M(r) = mCloud + (4/3)π·ρ_ISM·(r³ − rCloud³)` (`mass_profile.py:319-322, 339-342, 425-428`)
— i.e. the shell sweeps up only the negligible ISM mass once it leaves the cloud.

The bubble *interior* cooling integrand is explicitly `∝ n²`
(`bubble_luminosity.py:696`): `integrand_bubble = chi_e * n_bubble**2 * Lambda * 4π r²`,
with `n_bubble = Pb / (μ k_B T)` (`bubble_luminosity.py:623`). The interior density is set
by `Pb`, not directly by the cloud profile — but `Pb`, the shell mass, and the ram pressure
the shell drives into all depend on the ambient density the shell is plowing through. So
the cloud→ISM density cliff propagates into `Lloss` indirectly via `Pb`/shell dynamics,
matching the cleanroom "dip is `Lloss ∝ n²V = (Pb/T0)² R2³`" note (FINDINGS §3 follow-ups).

## 3. What "breaking rCloud" actually DOES in code (the headline)

**It is a phase switch, not a stop, and not a failure.**

- The energy phase (1a) carries the `cloud_boundary` event
  (`phase_events.py:218-247`, built in `build_energy_phase_events` :440-449). Its flags:
  `terminal = True`, `direction = 1`, **`is_simulation_ending = False`**,
  `reason_code = "cloud_boundary"`. The module docstring states it plainly
  (`phase_events.py:22-23`): *"cloud_boundary: R2 > rCloud (energy phase -> implicit)"*.
- When 1a ends at the boundary, `main.py:251` returns and the orchestrator advances to
  **phase 1b (implicit)** at `main.py:286` (`run_phase_energy`). The implicit phase events
  (`build_implicit_phase_events` :456-499) **contain NO rCloud event** — it is gone. The
  implicit phase therefore integrates *freely past rCloud*, all the way to `stop_r`
  (default 500 pc) or `stop_t` (default 15 Myr) or collapse.
- The only rCloud-keyed termination is **opt-in**: `stop_at_rCloud_nSnap`
  (`default.param:124`, default **None** = disabled; `main.py:260-272`,
  `SimulationEndCode.RCLOUD_BOUNDARY` = code **3**). Code 3 lives in the **CLEAN 0–9
  band** (`simulation_end.py:69-74, 109-111`) — a clean physical/intentional outcome, the
  same band as `STOPPING_TIME` and `SHELL_DISSOLVED`. It is explicitly *not* an error
  (errors are 10–29; inspection 50–59).
- Crossing rCloud produces **no NaN, no error, no crash**. (The `ENERGY_COLLAPSED`
  stop in the implicit phase — `run_energy_implicit_phase.py:1007-1019` — is triggered by
  `Eb <= 0`, not by rCloud; it fires only for the 5e9-cluster regime per FINDINGS §3.)

⇒ **The prior framing "breaking rCloud = fail" is incorrect against current source.**
Crossing rCloud is the routine, clean 1a→1b transition; the implicit phase is *designed*
to run past it.

## 4. The thought-experiment, made concrete: enlarge rCloud without editing production

You cannot set rCloud directly. The clean lever (homogeneous cloud) is:

- raise `mCloud` (rCloud grows as `mCloud^(1/3)` at fixed `nCore`), and
- lower `sfe` by the same factor so `mCluster = mCloud * sfe` (the feedback driver,
  `read_param.py` Step 6 / `registry.py:367-372`) is **unchanged**.

This keeps the shell's environment identical (same `nCore`, same `n(r)` inside, same
`nISM`, same `Lmech`) but extends the dense cloud to a larger radius. Verified design
(`h2_rcloud_compute.py` on the two params below):

| run | mCloud | sfe | mCluster (feedback) | rCloud | nCore |
|---|---|---|---|---|---|
| `sc_baseline.param` | 1e5 | 0.3 | 3e4 | 1.69 pc | 1e5 |
| `sc_bigcloud.param` | 1e7 | 0.003 | 3e4 (identical) | 8.83 pc (5.2×) | 1e5 |

**What stays self-consistent if rCloud is enlarged this way:** ambient density profile
(`nCore`, `nISM` unchanged); feedback `Lmech` (mCluster fixed); the density cliff at the
(new) edge; `rCloud_max` validation (8.83 < 200, passes). **What changes (unavoidably):**
the cloud's self-gravity / enclosed mass `M(<r)` is ~100× larger inside the cloud
(`mass_profile.py`), so `F_grav` on the shell rises — this is *real* physics, not an
artifact, and is the price of "more cloud". The bubble interior physics, conduction, and
cooling solver are untouched.

## 5. Empirics from committed data: in-cloud dilution vs blowout

Source: `cleanroom/data/c0_*_h0.csv` (hybr, stop_t=6; canonical for triggers/cooling).
The `rCloud` column there is blank (run-const, excluded from snapshots) so rCloud is
injected from §1's authoritative table. Analysis: `h2_analyze.py` → `h2_crossing_summary.csv`;
trajectories: `h2_trajectory.py`.

**Every config crosses rCloud, and R2 runs far past it:**

| config | rCloud | t_cross [Myr] | R2max/rCloud | ratio_min | R2/rCloud @ ratio_min | Lloss peak @ R2/rCloud |
|---|---|---|---|---|---|---|
| small_dense_highsfe | 0.326 | 0.0117 | **525×** | 0.283 | 1.11 | 1.11 |
| simple_cluster | 1.69 | 0.0902 | **147×** | 0.324 | 1.07 | 1.07 |
| midrange_pl0 | 8.53 | 0.392 | 34.4× | 0.365 | 1.06 | 1.13 |
| pl2_steep | 21.35 | 0.840 | 13.9× | 0.489 | 0.064 | 0.064 |
| be_sphere | 15.5 | 0.856 | 15.2× | 0.471 | 0.72 | 0.91 |
| large_diffuse_lowsfe | 88.05 | 3.66 | 1.52× | 0.465 | 1.22 | 1.19 |

Trajectory shape (representative; see `h2_trajectory.py` output, e.g. simple_cluster):

```
 t[Myr]   R2[pc]  R2/rC   ratio   Lloss/Lg   (trigger needs ratio<0.05)
 0.044    1.19    0.71    0.435   0.565      <- inside cloud, Lloss rising
 0.090    1.73    1.02    0.333   0.667      <- CROSSES rCloud; ratio at its minimum
 0.143    2.45    1.45    0.438   0.562      <- past edge, ratio recovering
 1.04    60.0    35.5     0.925   0.075      <- deep in ISM, Lloss collapsed
 5.54   234.5   139       0.775   0.225
```

**Reading:** the cooling ratio reaches its minimum (closest approach to 0.05) **right at
the cloud-edge crossing** (R2/rCloud ≈ 0.7–1.1 in 5/6; pl2_steep is the steep-profile
exception where the in-cloud Lloss peaks much earlier, R2/rCloud≈0.06, because its density
drops steeply *within* the cloud). After crossing into nISM, `Lloss/Lgain` **collapses** as
R2 dilutes (e.g. 0.667 → 0.075), and the ratio **recovers away from the trigger**. So:
- The *trigger-relevant* cooling collapse is **the blowout** (shell entering nISM /
  expanding to 100s of pc), not pure in-cloud dilution — though steep profiles (pl2) show
  significant in-cloud dilution too.
- **Crucially, even the in-cloud minimum (0.28–0.49) never reaches 0.05.** Geometry sets
  *when* the ratio bottoms out (at the edge), but the *floor value* is set by the
  intrinsic under-cooling of the energy-conserving interior — which holding the shell in
  dense gas longer does not lower past 0.05.

## 6. Bounded experiment (enlarged rCloud), if it completed

Params committed: `sc_baseline.param`, `sc_bigcloud.param`. Command (cleanroom harness):
```
OMP_NUM_THREADS=1 python docs/dev/transition/cleanroom/c0_consistency.py \
    docs/dev/transition/pt4/sc_<run>.param --stop-t 0.5 \
    --out docs/dev/transition/pt4/h2_sc_<run>.csv
```
Outputs: `h2_sc_baseline.csv` + `.log` (committed; bigcloud did not finish — see below).

**Baseline run completed cleanly (exit 0, no NaN, no error).** It reached t=0.028 Myr,
R2=0.93 pc (R2/rCloud=0.55 — still inside the cloud), with phases energy(97 segs) +
implicit(31 segs); `f_ret` 0.462→0.257 (DOWN, "under-cooled" — exactly the cleanroom
signature), cooling-ratio minimum so far **0.478** at R2/rCloud=0.55. This is fully
consistent with the committed h0 run of the *same* config (§5: same simple_cluster, carried
to t=6, crosses rCloud at t=0.09, ratio_min=0.324). The run is healthy and behaves
identically to the canonical data — the enlarged-cloud thought-experiment does not break
anything.

**Big-cloud run ALSO completed cleanly (exit 0, no NaN)** — 139 rows to t=0.0546 Myr,
R2max=1.33 pc (R2/rCloud=0.15, deep inside the larger cloud), `f_ret` 0.462→0.221 (DOWN,
under-cooled). Both runs committed: `h2_sc_baseline.csv`, `h2_sc_bigcloud.csv`.

**The decisive matched comparison.** At the *same absolute* shell radius **R2 = 0.894 pc**
— where the baseline shell is at R2/rCloud=0.53 (halfway to its 1.69-pc edge) but the
bigcloud shell is at R2/rCloud=0.10 (deep inside its 8.83-pc cloud) — the cooling state is
**identical**: `Lloss/Lgain = 0.5155`, ratio = `0.4845` in *both* runs. The cooling
behaviour is set by the **local density the shell sits in** (same `nCore` in both), **not
by proximity to rCloud**. This empirically proves: (a) the enlarged-rCloud construction is
clean — at matched R2 inside the cloud the physics is bit-for-bit the same; (b) enlarging
rCloud only changes *where* the eventual density cliff sits, not the in-cloud cooling floor;
(c) that floor (~0.48 at this early epoch; ~0.32 at the edge minimum per §5) is **6–10×
above the 0.05 trigger regardless of cloud size**.

**Caveat — the implicit phase is slow by design (it is the stall under study).** Energy
phase ~1m43s; each implicit segment is a full betadelta solve. Both runs were capped early
(reached t≈0.03–0.05 Myr, R2 < rCloud) — they did not run far enough to watch the bigcloud
shell reach its larger 8.83-pc edge and blow out. **The committed h0 data in §5 (same
configs to t=6, full blowout)** remains the primary evidence for the post-crossing collapse.
Re-run `sc_bigcloud.param` with a multi-hour timeout to watch its (delayed) blowout
directly; the matched comparison above already settles the in-cloud question.

## 7. Verdict (answers to the maintainer's four sub-questions)

**(a) Derived or free; can it be enlarged cleanly?** Derived; no direct knob. Enlarge
cleanly via `mCloud↑` + `sfe↓` (hold mCluster) — feedback and the density profile stay
fixed; only self-gravity (real physics) grows.

**(b) Does density drop at rCloud, and does that drive the Lloss collapse?** Yes — `n` drops
to `nISM`, a **100×–10⁵×** cliff (`density_profile.py:9-13, 148`). It drives the
trigger-relevant `Lloss` collapse **at/after blowout**, as the shell leaves the dense cloud
and expands to 100s of pc (`Lloss ∝ n²V`). In-cloud dilution contributes (dominant only for
steep profiles); the dominant trigger-killer is the post-blowout run into nISM.

**(c) Is "breaking rCloud = fail" real or artificial?** **Artificial as a *failure*** — and
the prior doc's framing is wrong. In code, `R2 > rCloud` is a clean phase switch
(`phase_events.py:218-247`, `is_simulation_ending=False`) and any rCloud-keyed stop is a
**clean exit (code 3, band 0–9)**. The implicit phase runs past it without error. The
boundary is "artificial" only in the sense that the *cloud model* itself ends there (gas →
ISM); it is a real physical regime change (dense cloud → diffuse ISM), not a numerical
failure.

**(d) If rCloud were huge, would the run continue self-consistently AND would the trigger
fire?** **Continue self-consistently: YES** (it already runs past rCloud today; enlarging
the cloud just delays the density cliff, with larger but physical self-gravity). **Trigger
fire: NO, not from geometry alone.** Keeping the shell in dense gas pins `Lloss/Lgain` near
its in-cloud floor of **0.28–0.49** — still ~6–10× above 0.05. The matched-R2 experiment
(§6: identical cooling state at R2=0.894 pc whether at R2/rCloud=0.53 or 0.10) confirms the
floor is set by local density, not by rCloud proximity, so enlarging rCloud cannot lower it.
The under-cooling is
intrinsic to the energy-conserving Weaver/Rahner interior (cleanroom §3: `f_ret` ≈ 0.25–0.40
vs observed 0.01–0.1; interior `T0` ~1e6–8e6 K, too hot to radiate). **So H2's mechanism is
correct (the stall *is* the geometry collapsing `Lloss`), but its remedy is wrong (a bigger
cloud does not manufacture a cooling-balance event).** The real lever is *more cooling* —
mixing-layer cooling integrated into the betadelta solve, or leakage (`coverFraction<1`),
per the cleanroom recommendations — not a bigger rCloud.

**Net:** H2 **refuted as a fix**, **vindicated as a diagnosis**: blowout and the cooling
stall are one geometric phenomenon, and "breaking rCloud" is a clean transition, not a
failure — but removing the boundary does not let the cooling trigger fire.

## 8. Artifacts (all committed under `docs/dev/transition/pt4/`)
- `h2_rcloud_compute.py` → `h2_rcloud_edge.csv` — rCloud + edge density drop via production
  `read_param`+`get_InitCloudProp`+`get_density_profile` (verifies §1, §2 tables).
- `h2_analyze.py` → `h2_crossing_summary.csv` — per-config R2/rCloud crossing, ratio_min,
  Lloss-peak vs crossing (§5 table). Reads `cleanroom/data/c0_*_h0.csv`.
- `h2_trajectory.py` — prints R2/rCloud vs ratio vs Lloss trajectory around the crossing.
- `sc_baseline.param`, `sc_bigcloud.param` — the clean enlarged-rCloud experiment (§6);
  `h2_sc_*.csv`/`.log` if the runs completed.
