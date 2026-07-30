# Orchestrator resolutions of reconciler-flagged decidable questions

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

**Status (2026-07-30):** 🔵 ACTIVE — running log of candidates the orchestrator closed by direct lookup.

## What this file is

Phase-2 reconcilers are forbidden the source: they see only the three lens
reports for their slice. When a reconciler cannot decide a question but *can*
name the exact fact that would decide it, that is a well-posed task for the
orchestrator, who is not blind. This file records those resolutions.

Each entry states the reconciler's open question, the lookup, and the verdict —
including verdicts that **clear** the code, which are the point as much as the
defects are. Nothing here has been through the Phase-5 skeptic gate; a
resolution is evidence, not a ruling.

---

## S9-R-01 — cooling density factor (CIE branch) → **CLEARED**

**Open question.** The S9 reconciler found the CIE branch applies `chi_e · ndens²`
and computed the error factor as `chi_e·(ndens/n_H)²/1.20`: **1.00** if `ndens`
is n_H and `chi_e` = 1.2; **0.833** if `chi_e` = 1.0; **4.41** if `ndens` is
n_tot with `chi_e` = 1.0; **5.29** if n_tot with `chi_e` = 1.2. Lens A had
asserted `ndens` is *total* number density, but flagged it as an inference — A
never saw a call site — and Lens C's spec reading contradicted it. The
reconciler declined to guess and named the deciding facts.

**Lookup.**

- `trinity/_input/registry.py:415` — `chi_e`, default **1.2**, defined as
  *"Electron-per-hydrogen-nucleus factor n_e/n_H = 1 + Z_He*x_He for the HOT
  bubble… Multiplies n_H^2 in the bubble CIE cooling."*
- `trinity/_input/read_param.py:314` — `_chi_e = 1 + _ZHe * _xHe` with the
  inline gloss *"electrons per H nucleus, n_e/n_H"*; at the defaults
  `x_He = 0.1`, `Z_He = 2` this is 1.2.
- Call site `trinity/bubble_structure/bubble_luminosity.py:430`, with `ndens`
  built one line earlier at `:428`:
  `ndens = Pb / ((mu_convert/mu_ion) · k_B · T)`.
- `mu_convert` = μ_H = 1 + 4x_He (mass per **hydrogen nucleus**);
  `mu_ion` = μ_H/(2 + x_He(1+Z_He)) (mean mass **per particle**), both from
  `read_param.py:311-314`. So `mu_convert/mu_ion = 2 + x_He(1+Z_He) = 2.3`,
  which is particles-per-H = n_tot/n_H.
- Ideal gas gives `Pb = n_tot k_B T`, hence
  `ndens = n_tot / (n_tot/n_H) = n_H`.
- Corroborated by `registry.py:366`: *"All densities n in TRINITY are
  hydrogen-nuclei densities n_H."*

**Verdict — the code is correct.** `ndens` is n_H, `chi_e` is n_e/n_H, so the
CIE branch computes `chi_e · n_H² · Λ = n_e n_H Λ` — the error-factor-**1.00**
case, correct for a Λ normalised per `n_e n_H` (the standard CIE convention).

**Lens A's inference was wrong**, and Lens C's headline 4.41× does not apply to
this branch. Recording this as a lens error is deliberate: the reconciler's
refusal to promote an unverified inference to a finding is what prevented a
false S1.

---

## S9-R-02 — non-CIE cube normalisation → **CLEARED (code); doc defect stands**

**Open question.** Lens B's transcription has `trinity/cooling/non_CIE/read_cloudy.py:23`
describing the cube as `[erg cm3 / s]` — i.e. a Λ — while the arithmetic Lens A
transcribed requires `erg cm⁻³ s⁻¹` (volumetric), because the non-CIE branch
applies **no** density factor at all. If the docstring were right, that branch
would be short by n², an error the reconciler bounded at up to 10⁸. The
reconciler's proposed test: scan the `ndens` column of a bundled
`opiate_cooling_*.dat` at fixed (T, Φ).

**Measurement.** Ran on `lib/default/opiate/opiate_cooling_rot_Z1.00_age2.00e+06.dat`
(columns `ID ndens temp phi nedens heat cool`), holding T = 3.16e4 K, Φ = 1e10
fixed and sweeping the 31 densities in that block:

| n_H | cool | cool/n² |
|---|---|---|
| 1e-3 | 3.113e-29 | 3.113e-23 |
| 1e-1 | 6.347e-25 | 6.347e-23 |
| 1e+1 | 4.936e-21 | 4.936e-23 |
| 1e+3 | 7.922e-17 | 7.922e-23 |
| 1e+5 | 8.555e-13 | 8.555e-23 |
| 1e+7 | 8.015e-09 | 8.015e-23 |
| 1e+9 | 7.551e-05 | 7.551e-23 |
| 1e+11 | 7.131e-01 | 7.131e-23 |

`d(log cool)/d(log n) = **2.014**` over 14 decades, and `cool/n²` varies only
3.1e-23 → 8.6e-23 (factor 2.7, consistent with the ionisation state shifting
with density). A Λ would give slope 0.

**Verdict — the cube is volumetric** (`erg cm⁻³ s⁻¹`, n² already included), so
the non-CIE branch applying no density factor is **correct**.

The **docstring is wrong**: `read_cloudy.py:23` labels a volumetric rate as
`[erg cm3 / s]`. That is an **S3** documentation defect, not the S1 the unit
mismatch would otherwise imply. It is exactly the failure mode this audit was
built around — a comment that would make a correct line look wrong (or, read the
other way, license a future "fix" that multiplies by n² and breaks it).

**Reproduce:**

```bash
python3 -c "
import numpy as np
d=np.loadtxt('lib/default/opiate/opiate_cooling_rot_Z1.00_age2.00e+06.dat',skiprows=1)
n,T,phi,cool = d[:,1],d[:,2],d[:,3],d[:,6]
Ts,ps=np.unique(T),np.unique(phi); m=(T==Ts[10])&(phi==ps[10])
print(np.polyfit(np.log10(n[m]),np.log10(cool[m]),1)[0])   # -> 2.014
"
```

---

## Still open in S9 (not resolved here)

- **S9-R-08** — `np.save` of the ragged cube list (`read_cloudy.py:265`) raises
  on the pinned numpy, so no cooling table can be rebuilt from source today.
  Lens A verified this by repro; it needs a fix decision, not a lookup.
- **S9-R-05** — whether the two cooling models actually agree at `Tcut_nCIE`.
  The reconciler corrected Lens C's proposed test (sweeping T across the seam
  returns ≈1 by construction — a false negative); the valid detector compares
  the two models **at the same T**. Not yet run.

---

# S10 SPS — R-01 reachability settled (2026-07-30)

The reconciler flagged its own top finding with an honest caveat: taken
literally, A's mechanism says the first ODE evaluation at `t = 0.0` crashes
every run, which the working quickstart contradicts. It asked for the
reachability lookup rather than promoting the claim. Resolved here.

**Mechanism — confirmed.** `update_feedback.py:184` computes `pdotdot_total`
by central difference at `t ± 1e-9`. The guard at `:155` admits the *closed*
interval `t_min <= t <= t_max`; the stencil needs it open by `1e-9`. The SPS
interpolators are built with `scipy.interpolate.interp1d(...)` at
`read_sps.py:341+` with the default `bounds_error=True`, and `read_sps.py:264`
prepends `t = 0.0`, so `t_min` is exactly `0.0`. Probing just outside the
domain raises:

```
$ python -c "
import scipy.interpolate, numpy as np
f = scipy.interpolate.interp1d(np.array([0.,.1,.2]), np.array([1.,2.,3.]))
f(-1e-9)
"
ValueError: A value (-1e-09) in x_new is below the interpolation range's
minimum value (0.0).
```

**Reachability — refuted for current configs.** Every call site of
`get_current_sps_feedback` is in a *later* phase — `phase1b_energy_implicit`
(`:803`), `phase1c_transition` (`:496`, `:750`, `:834`), `phase2_momentum`
(`:407`, `:577`, `:887`). None is in phase0 or phase1-energy, so `t` has
always advanced past `0.0` before the first call. That is why the quickstart
runs. No clamp of `t` to the SPS `t_max` was found in the implicit or
momentum runners either, so the upper endpoint is not forced exactly.

**Disposition: S1 -> S2 (latent).** The guard/consumer domain contract really
is inconsistent, and it is one root cause, not two (the reconciler was right
to fold the `t=0` and `t=t_max` items together). But nothing reaches the
endpoints today, so it changes no current output. It becomes reachable the
moment a caller is added earlier in the run, or a stopping condition lands on
`t_max` exactly. Fix remains the reconciler's: a one-sided difference at the
endpoints, a clamped stencil, or `CubicSpline(...).derivative()`, which needs
no stencil at all.

**Not settled here:** whether the bundled CSV's columns really are in the
preset's order (R-02/R-03). Neither lens read
`lib/default/sps/starburst99/1e6cluster_default.csv`; the comment-vs-literal
agreement the reconciler found is two artefacts in the same file by the same
author, not independent confirmation. That one needs the table read.

---

# S11 orchestration — R-01 collapse classification (2026-07-30)

The reconciler's headline finding: `apply_event_result` decides whether a run
collapsed by substring-matching the reason code —

```python
# trinity/phase_general/phase_events.py:627
if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():
    params['isCollapse'].value = True
```

It called two consequences: `large_radius_event` wrongly flagged as collapse,
and `velocity_runaway_event` wrongly not flagged. The mechanism is exactly
right. The **reachability is inverted between the two halves** — the one it led
with is dead, the one it listed second is live.

**Half 1 — `large_radius_event` mislabelled: NOT reachable. Demote to S4.**
`make_large_radius_event` (`:139-163`, sets `reason_code = "large_radius_event"`)
is **never constructed**. None of the four event-list builders (`:447`, `:487`,
`:531`, `:569`) includes it. The live stop-radius termination is a plain string
set inline in the three runners — `run_momentum_phase.py:852`,
`run_transition_phase.py:799`, `run_energy_implicit_phase.py:1329` — each of
which assigns `SimulationEndReason`/`SimulationEndCode`/`EndSimulationDirectly`
and `break`s **without calling `apply_event_result`**. So line 627 never sees a
`large_radius` code, and `isCollapse` is not set on that path. The builder is
dead code (flag, don't delete — CLAUDE.md rule 3).

**Half 2 — `velocity_runaway_event` not flagged: LIVE. Confirmed S1.**
`make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse")` is
constructed in **all four** builders (`:450`, `:490`, `:534`, `:572`) — it is
the default direction. Its event is `v2 + v_max`, `direction=-1`, i.e. it fires
when `v2 < -500 pc/Myr`: the most violent infall the code detects. It sets
`terminal = True` and `is_simulation_ending = True`, so it reaches line 627 via
`apply_event_result` (call site e.g. `run_momentum_phase.py:749`). Its
`reason_code` is `"velocity_runaway_event"` — which contains neither `'radius'`
nor `'collapse'`. **`isCollapse` stays False on a genuine collapse.**

This reaches user-facing output: `_output/show_run.py:226-228` prints
"Collapsing: yes/no" from it, and `_output/simulation_end.py:433` records it in
`final_state`. A fate census filtered on `isCollapse` silently undercounts
every velocity-runaway collapse.

**Reproduction (static, no run needed):**

```
$ grep -n 'reason_code = ' trinity/phase_general/phase_events.py
128:    event.reason_code = "small_radius_event"      # -> 'radius'  -> True  (correct)
160:    event.reason_code = "large_radius_event"      # -> 'radius'  -> True  (wrong, but dead)
211:    event.reason_code = "velocity_runaway_event"  # -> no match  -> False (wrong, and live)
$ grep -rn 'make_large_radius_event' trinity/ --include=*.py   # definition only, no caller
```

**Fix outline.** Replace the substring test with an explicit property on the
event — the `SimulationEndCode` enum already exists and is already carried on
every event (`end_code`), so the collapse set can be declared there rather than
inferred from spelling. That also removes the class of bug entirely rather than
patching one member of it.

**Still open in S11 (need the phase-runner reads the lenses were denied):**
R-01's *classification-by-list-index* sibling, plus R-06, R-09, R-22 — all four
rest on the same unread files (`run_energy`, `run_phase_energy`,
`run_phase_transition`, `run_phase_momentum`). The reconciler's Q1+Q2 greps
clear or confirm the whole cluster and are the highest-value pair left.

---

# S11 orchestration — all 14 open questions settled (2026-07-30)

The reconciler could not read the four phase runners (`run_energy_phase`,
`run_energy_implicit_phase`, `run_transition_phase`, `run_momentum_phase`) —
by design, they are outside the slice. It flagged that ~a third of its findings
rested on assumptions all three lenses shared about that unread code, and
listed 14 decisive lookups. All 14 are run below. **Net: one S1 confirmed and
escalated, seven findings demoted, one headline theme refuted.**

## The refutation — Q4

The reconciler's second theme was "no channel distinguishes a solver failure
from a physical fate; `sol.status`/`sol.success` are never read." **That is
false.** Every runner checks it:

| file | line | check |
|---|---|---|
| `phase1_energy/run_energy_phase.py` | 310 | `if not solution.success:` |
| `phase1b_energy_implicit/run_energy_implicit_phase.py` | 1085 | `if not sol.success or len(sol.t) == 0:` -> `termination_reason = f"solver_failed: {sol.message}"` |
| `phase1c_transition/run_transition_phase.py` | 646 | same |
| `phase2_momentum/run_momentum_phase.py` | 728 | same |
| `bubble_structure/bubble_luminosity.py` | 362 | `if not sol.success:` -> `BubbleSolverError` |

The claim is true *within the slice* (`main.py`, `phase_events.py`) and the
reconciler generalised it to the codebase. Exactly the inference-not-fact trap
— and this time all three lenses shared the wrong assumption, which is why
agreement is not proof.

**The narrow half survives:** `run.py:231` calls `main.start_expansion(params)`
without capturing the return value, so the `99` failure code is discarded and
never reaches `sys.exit`. That stays, at S3.

## The escalation — Q7

`isCollapse` is **not** metadata-only. It is consumed by
`paper/_lib/plot_markers.py` — `find_collapse_time` (`:147`) and
`add_collapse_marker` (`:371`). So the `velocity_runaway_event`
mis-classification recorded above (collapse at `v2 < -500 pc/Myr` never sets
`isCollapse`) propagates into **published paper figures**: the collapse marker
is silently omitted. S1 confirmed, and worse than "census undercount".

## Disposition table

| Q | Answer | Moves |
|---|---|---|
| Q1 | `cooling_balance` is **not** spliced into `solve_ivp`. The factory is unpacked at `run_energy_implicit_phase.py:752` and **never used again**; the live decision is the inline ratio at `:1296` (`Lgain > 0 and (Lgain-Lloss)/Lgain < threshold`), as the docstring at `:278` says. | **R-06 S1 -> S3.** New **S4**: `make_cooling_balance_event` + the returned factory are dead (flag, don't delete). |
| Q2 | All four sites pass the **event root**, not the last step: `t_now = event_result.t` is assigned immediately before the call (momentum `:743`->`:749`, transition `:662`->`:669`, implicit `:1105`->`:1117`; energy passes `event_result.t` directly at `:327`). | **R-09 S1 -> S4** (cosmetic: three spell it via `t_now`, one directly). |
| Q3 | `registry.py:420` — `ParamSpec(name='EndSimulationDirectly', default=False, ...)`. Not `None`. | **R-10 hazard 1 S1 -> S4.** |
| Q4 | See above — refuted. | **Theme demoted**; `run.py:231` half stays **S3**. |
| Q5 | `stop_r` default `500` pc (`registry.py:352`), well above any GMC `rCloud`. | **R-04 / R-02 large-radius half stay S4** (and the builder is dead — see the R-01 entry above). |
| Q6 | `current_phase` is consumed by `_output/` only — `show_run`, `simulation_end`, `trinity_reader`, and the cloudy deck exporters. **No hits in `paper/` or `tools/`.** | **R-11 S2 -> S3** (it does reach cloudy decks, so not pure logging). |
| Q7 | See above — escalated. | **R-02 velocity half: S1 confirmed, reaches published figures.** |
| Q8 | `COOLING_PHASE_KEYS` (`dictionary.py:1180+`) holds `residual_*`, `betadelta_*` and `bubble_*` keys. `Lmech_total`/`v_mech_total` are **not** in the block inspected. | **R-15 likely cleared** — marked partial, only the first 20 entries were read. |
| Q9 | `MIN_RADIUS_SAFETY = 0.01` pc, `MIN_RADIUS_FACTOR = 1.5` (`phase_events.py:71-72`), `coll_r` default `1` pc (`registry.py:355`). So `min_r = max(1.5, 0.01) = 1.5` pc; the safety floor binds only for `coll_r < 0.0067` pc, which no plausible GMC config sets. | **R-05**: offset is 0.5 pc. **R-20 confirmed S4** — `MIN_RADIUS_SAFETY` is effectively dead. |
| Q10 | Sweeps use `ProcessPoolExecutor` (`run.py:612`); `start_expansion` runs once per worker process (`:231`). | **R-10 hazard 2 -> S4.** Cross-run flag leakage is not reachable via the sweep path (in-process global leakage remains real — CLAUDE.md says so — but nothing calls it twice per process). |
| Q11 | `stop_at_rCloud_nSnap` appears **only** in `main.py:36-65`, as a conflict *validator* against `stop_r`. A stale `.pyc` matches `run_momentum_phase`, but current source does not. | Confirms the reconciler: the `>= 1` semantics B and C both describe have **no consumer**. **S3.** |
| Q12 | `Lgain` is guarded: the transition test at `:1296` requires `Lgain > 0`, and the diagnostics divide by `max(Lgain, 1e-300)` (`:872`, `:906`). | **R-07 S2 -> S4.** |
| Q13 | Both present — `__format__` at `dictionary.py:152`, `__truediv__` at `:180`. | **R-21 closed, no defect.** |
| Q14 | The integrated state is **phase-dependent**, which is why A/C and B disagreed: momentum `['R2','v2']` (2), transition and phase1-energy `['R2','v2','Eb']` (3), implicit `['R2','v2','Eb','T0']` (4, confirmed by the `y0=` log at `:1062`). | Resolves the A/C-vs-B split. **Leaves open:** `y_index=2` is valid in the 3- and 4-component phases but out of range in momentum — not chased here. |

## Remaining open in S11

- `y_index=2` under the 2-component momentum state (from Q14). One lookup.
- Q8's `COOLING_PHASE_KEYS` tail beyond the first 20 entries.
- R-01's classification-by-list-index sibling, which Q1/Q2 did not reach.

---

## S12b-B-01 — `mu_convert` unit factor → **CLEARED (code); comment defect stands**

**Open question.** S12b Lens B, reading prose only, flagged the comment at
`trinity/_input/sweep_runner.py:108` — `mu_convert: [m_H] -> Msun (factor
~9.42e-58)` — as ~12 % high against the true `m_H/M_sun`, and rated it **S2
(units)**. It could not see whether `9.42e-58` was also a code literal. That
distinction decides the severity, because the block's stated purpose is that
"the preflight check matches what the actual simulation will see": a hardcoded
wrong factor would put the sweep's GMC plausibility screen 12 % off the value
the simulation itself uses. Lens A did not mention the constant at all.

**Lookup.**

- `trinity/_input/sweep_runner.py:104-113` — the comment block is *descriptive*.
  The code immediately below it calls `mu_factor = convert2au('m_H')` and
  `ndens_factor = convert2au('cm**-3')`. **No numeric literal is used**; both
  factors come from the shared conversion table.
- Evaluated directly: `convert2au('m_H')` = **8.416562e-58**.
- Independent value from `astropy.constants`: `m_H/M_sun` =
  (m_p + m_e)/M_sun = **8.4164e-58**. Ratio code/true = **1.00002**.
- The comment's `9.42e-58` is **1.11922 x** the value the code actually uses.
- Sibling factor cross-check: `convert2au('cm**-3')` = **2.937999e+55** against
  the same comment block's `~2.94e+55` — correct, and `(1 pc)^3 = 2.937999e55
  cm^3` confirms it independently.

**Verdict — the code is correct; the comment is wrong.** `S2 -> S3`. The
preflight screen converts `mu_convert` with the same table the simulation uses,
so it does match what the run will see. Only the prose is off, by one digit:
`8.42e-58` -> `9.42e-58`.

Recording the **downgrade** deliberately. Lens B's arithmetic was right and its
cross-check of the neighbouring factor was sound reasoning under a prose-only
view; the S2 rating was the correct call *given what it could see*. The blind
lens produced a true observation and an over-severe rating, and the orchestrator
lookup is what separates them. A finding that gets smaller under verification is
as much a result as one that grows.

---

## S13a-B-05 — `is_successful_run` range test → **CLEARED (false positive)**

**Open question.** S13a Lens B, reading prose only, reported the docstring of
`is_successful_run` as specifying `exit_code in [0, 9]` and flagged it against
`is_clean()`'s documented range 0-9. In Python those differ decisively:
`ec in [0, 9]` is membership of the two-element list `{0, 9}`, so exit codes
**1-8 would be misclassified as failures**. If the code matched the docstring
literally this was a real defect, not doc drift.

**Lookup.**

- `trinity/_output/trinity_reader.py:543` — the code is
  `return 0 <= int(ec) <= 9`. A chained comparison, i.e. a genuine **range**
  test.
- `trinity/_output/simulation_end.py:111` — `is_clean()` is
  `return 0 <= self._code <= 9`. **Identical semantics.**
- The docstring at `:530` reads *"exit code in [0, 9] (clean termination per
  ``SimulationEndCode.is_clean()``)"* — `[0, 9]` here is **mathematical
  interval notation**, not a Python list literal.

**Verdict — the code is correct and consistent with `is_clean()`.** The finding
is withdrawn. Lens B's reading was reasonable under a prose-only view: `[0, 9]`
inside a Python docstring is genuinely ambiguous, and the failure mode it
inferred would have been serious. Only the notation is unfortunate.

A residual **S4** stands on the notation itself — interval brackets in a Python
docstring invite exactly this misreading, and the next reader may be a human
rather than an audit lens. Flagged, not fixed.

This is the second B-only claim in the infra tier to shrink under lookup
(see S12b-B-01 above). The pattern is worth naming: a prose-only lens reliably
produces **true observations with over-severe ratings**, because a doc defect
and a code defect are indistinguishable from the prose side. That asymmetry is
a property of the method, not a failure of it — provided every such claim gets
the lookup before it reaches `FINDINGS.md`.

---

## S13b-R-01 — CLOUDY `dlaw` density: `n` vs `n_H` → **CLEARED**

**Open question.** The S13b reconciler ranked this first in the slice. Both
lenses reached the same hole blind from opposite directions: Lens B found the
`dlaw.py` docstring calling the output column `log10 n [cm^-3]` in prose and
`log10(n_H/cm^-3)` in its format block; Lens A found **no composition factor
anywhere in the code** — the column is a pure pc⁻³→cm⁻³ unit shift. If the
source array `log_shell_n_arr` were *total particle* density rather than
*hydrogen* density, every emitted deck would be wrong by ~0.35 dex in n(H):
decks that run, and produce plausible spectra at the wrong ionisation
parameter. The reconciler named the deciding lookup — trace the producer of
`log_shell_n_arr` and read whether it divides by `μ m_H` or `X_H m_H`.

**Lookup.** The producer is `shell_n_arr` in
`trinity/shell_structure/shell_structure.py:411` (`np.concatenate([nShell_arr_ion,
nShell_arr_neu])`), declared at `:81` as *"Number density through shell
[1/pc^3]"*. Three independent lines of evidence fix which density that is:

- **Global convention.** `trinity/_input/registry.py:366` (`mu_convert` spec):
  *"Use for the n_H -> rho mass conversion (rho = mu_convert * n_H) … **All
  densities n in TRINITY are hydrogen-nuclei densities n_H.**"*
- **Recombination term.** `shell_structure.py:282` computes
  `chi_e_shell * caseB_alpha * nShell_arr_ion**2`, and `registry.py:417` defines
  `chi_e_shell` as the *"Electron-per-hydrogen-nucleus factor n_e/n_H …
  Multiplies n_H^2 in shell recombination"*. The explicit `chi_e_shell` factor
  is present **because** `nShell` is n_H — with n_tot it would be wrong.
  Identical in form to the S9-R-01 CIE result above.
- **Opacity term.** `shell_structure.py:389` builds the mass column as
  `mu_convert * sum(nShell_arr * dr)`, and `mu_convert` is mass per **hydrogen
  nucleus**. Only valid for n_H.

**Verdict — the code is correct, and the missing composition factor is correct.**
`shell_n_arr` is n_H in pc⁻³, so the pure unit shift yields n_H in cm⁻³, which
is exactly what a CLOUDY `dlaw table radius` block expects. There is no
~0.35 dex error and **no factor should be introduced**.

What survives is Lens B's half, at lower severity: the `dlaw.py` docstring
states the column two ways (`n` and `n_H`) in one docstring. **S3** — the code
is right, the prose is ambiguous about the single most consequential quantity
in the export.

Worth recording *why* this looked dangerous: A's "no composition factor" was a
true observation about the code, and B's "documented two ways" was a true
observation about the prose. Both were right, and the alarming conclusion came
from neither lens but from their **conjunction** — which is exactly the
inference the reconciler is supposed to raise and the orchestrator is supposed
to close. The absence of a factor is only a defect if the input is n_tot; it
isn't.

---

## S13b-R-02 — `ZREL` solar-relative convention → **CLEARED (doubly); a different defect found**

**Open question.** The S13b reconciler *promoted* this to R-01's tier and rated
it the larger risk: Lens A marked the solar-relative reading "assumed", Lens B
marked it "not stated", and if the convention were wrong the deck error is
**~1.85 dex** — five times R-01's. Neither lens could settle it because the
declaring spec and the consuming template are both outside the slice.

**Lookup — four links, all outside S13b.**

1. `trinity/_input/registry.py:339` — `ParamSpec(name='ZCloud', default='1',
   info='Cloud metallicity', **unit='Zsun'** …)`. The convention **is**
   declared: solar-relative, linear. Neither lens could see it (S12a's slice).
2. `trinity/_output/cloudy/snapshot_to_deck.py:212` —
   `zrel = float(bundle.summary["ZCloud"])`. Passed through unchanged, emitted
   `f"{zrel:.4f}"` at `:274`.
3. The bundled template, line 20 — `metals and grains {{ZREL}}`. CLOUDY's
   `metals and grains <scale>` takes a **linear solar-relative** scale factor.
   `ZREL=1.0000` is solar. Convention matches end to end.
4. `trinity/_input/registry.py:99-105` — `_validate_ZCloud` **raises**
   `ParameterFileError` unless `value == 1`: *"Metallicity Z=… not implemented.
   Currently only Z=1 (solar) is supported."*

**Verdict — cleared twice over.** The convention is correct *and* the risk is
unreachable through the normal path: no run can start with `ZCloud != 1`, so
`ZREL` is always `1.0000`. The feared 1.85 dex cannot occur via `.param` input.

**But the lookup surfaced a defect neither lens could have found.**
`trinity/_output/cloudy/trinity_to_cloudy.py:140` defines a `--z-override`
CLI flag (`dest="z_override"`), and `snapshot_to_deck.py:206` takes
`zrel = float(z_override)` **in preference to the summary value** — bypassing
`_validate_ZCloud` entirely, since the validator guards `.param` load, not the
export CLI.

So `--z-override 0.2` emits `metals and grains 0.2000` into a deck whose
density structure, shell ionisation and cooling were all integrated at **solar**
metallicity. CLOUDY then computes photoionisation at 0.2 Z⊙ over a
hydrodynamic structure that assumed 1.0 Z⊙. The deck runs, and is internally
inconsistent, with no warning that the trajectory underneath it was not
computed at the metallicity the deck declares.

Rating **S2**: it needs an opt-in flag, and a user passing it may know exactly
what they are doing. But nothing says so, and the flag is the only way to reach
a non-solar deck in a code that otherwise refuses non-solar runs outright —
which makes it likely to be read as supported rather than as an override.
Recommend a warning at the override site, not removal. Flagged, not fixed.

This one is worth noting methodologically: the *cleared* lookup is what exposed
it. Chasing a hypothesised 1.85 dex error through four files found no error on
the audited path, and a real inconsistency on a path no lens was scoped to see.

---

## S12a-R-01 — user-set `mu_*` silently discarded → **CONFIRMED S1** (medium → high confidence)

**Open question.** The S12a reconciler overturned Lens A's blanket no-S1 verdict
for exactly this finding, rating it **S1 at medium confidence** and naming the
one lookup the rating hung on: *are `mu_*` actually declared in `default.param`,
i.e. can a user set them at all?* Neither lens read `default.param` (it is data,
not code, and sits outside the slice's file list). Its discriminator for S1 was:
a user value is **accepted by the schema**, **warns nothing**, and a *different*
value **silently drives the physics**.

**Lookup — both halves confirmed.**

1. **User-settable.** `trinity/_input/default.param` declares all four as
   ordinary editable parameter lines with concrete values:
   `mu_atom 14/11` (`:205`), `mu_ion 14/23` (`:209`), `mu_mol 14/6` (`:213`),
   `mu_convert 1.4` (`:217`). A user setting any of them hits no unknown-key
   error — the key *is* in the schema.
2. **Unconditionally overwritten.** `trinity/_input/read_param.py:316-319`:
   ```python
   params['mu_convert'].value = float(_muH)    * _mH_au
   params['mu_atom'].value    = float(_mu_n)   * _mH_au
   params['mu_ion'].value     = float(_mu_p)   * _mH_au
   params['mu_mol'].value     = float(_mu_mol) * _mH_au
   ```
   In-place `.value` assignment, derived from `x_He`/`Z_He`, with **no test of
   whether the user set them and no warning**. Because the `DescribedItem`
   object is unchanged, the anti-stomp guard — which compares object identity
   (Lens A, A-03) — structurally cannot see it.
3. **Drives the physics.** `mu_convert` is the `n_H -> rho` conversion
   (`rho = mu_convert * n_H`, `registry.py:366`) used in every zone.

**Verdict — S1 confirmed, confidence medium -> high.** A user who sets
`mu_convert 1.6` gets a run integrated at 1.4. No error, no warning, and the
snapshot records the derived value, so the discard is invisible after the fact.

**Mitigation, stated because it is real.** The `INFO:` line directly above each
key in the same file *does* say "Derived at load from x_He" (`:203`, `:207`,
`:211`, `:215`), and the Step-6 comment calls `x_He`/`Z_He` "the single source
of truth". So the behaviour is documented where an attentive user would look.
That lowers discoverability risk but does not change the mechanism: the file
still presents four derived quantities as editable input lines, and the code
still discards user values without a word.

The severity stands. A documented silent discard is still a silent discard —
the rubric asks whether physical output changes on a config run today, not
whether the manual warned you. **Fix direction** (for the maintainer, not
applied here): reject or warn on user-set `mu_*`, or stop advertising them as
inputs. Any of the three closes it.

Recording this as the reconciler's **overturn being upheld**. Lens A's rule —
"nothing changes numbers on a nominal successful run of a tracked config" — is
sound in general and circular for the parameter reader, whose entire job is
untracked configs. A defect there is invisible to tracked configs *by
construction*, which is exactly why the tracked-config test could not see it.
