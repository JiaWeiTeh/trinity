# Failed large clouds — `Eb=nan` / `R1 root finding failed` in the energy phase — fix plan

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

**Status (2026-06-19):** 🟡 **DIAGNOSED + REPRODUCED — root cause confirmed; candidate-fix matrix
running.** Two independent investigations + a sim-free probe + two live repros (local) agree on the
mechanism. The empirical config×idea matrix (robustness + no-op on healthy + science end-state) is the
open item; this doc owns it. **No production code changed yet** — fix direction is gated on the matrix.

---

## 1. Symptom (the user's report — Helix `paperII_grid_sweep`)

On the Heidelberg `Helix` cluster, the Paper II grid sweep produces failed outputs for the
**`mCloud=5e9`, `nCore=1e2`** points (across `sfe`, `PISM`, `nISM`). Representative real log
(`5e9_sfe005_n1e2_PL0_yesPHII_PISM0p0_nISM0p1`):

```
... Initial Weaver phase values: t0=0.000978 Myr, r0=3.66 pc, v0=3739 pc/Myr, E0=2.254e9, T0=4.09e7 K
----- PHASE 1a: Energy-driven phase (constant cooling) -----
  Inner discontinuity (R1): 3.178806e+00 pc
  Initial bubble pressure: 1.000963e+11 K cm⁻³
Switching to no approximation
ERROR | trinity.bubble_structure.get_bubbleParams | R1 root finding failed on [0, R2]:
       R2=7.047540e+00, Eb=nan, Lmech_total=5.070535e+12, v_mech_total=3.739310e+03
Emergency flush: saving 3 pending snapshot(s)...
```

The `R1 root finding failed` line is a **downstream victim**: `solve_R1` is handed `Eb=nan`, so `brentq`
on `[0, R2]` cannot bracket and raises. `Eb` already went `nan` an iteration earlier.

## 2. Root cause (verified 3× — two code traces + sim-free probe + live repro)

A `mCloud=5e9, sfe~0.05-0.1` cloud is a **5×10⁸ M⊙ cluster** with `Lmech ≈ 5×10¹²` (code units), ~500×
a typical `1e6` cluster. At `nCore=1e2` the bubble **radiates faster than the wind resupplies it**, so
`Eb` collapses instead of growing (local repro: `E0=6.4e9 → 4.8e8`, a 13× drop, in 0.003 Myr; the
healthy `1e7` control's `Eb` *grows* `5.7e5 → 2.3e7`).

The inner wind shock `R1` solves `get_bubbleParams.get_r1`:

```
R1 = sqrt( Lmech_total / v_mech_total / Eb * (R2**3 - R1**3) )     # get_bubbleParams.py:400
```

As `Eb → 0` with `Lmech` huge, the only root drives **R1 → R2** (the hot shocked-wind shell collapses to
zero thickness). Then `bubble_E2P` (`get_bubbleParams.py:228`):

```
Pb = (gamma - 1) * Eb / (r2**3 - r1**3) / (4*pi/3)
```

divides by `r2**3 - r1**3`. **The cliff is floating-point, not physical** — see §3. `Pb → inf`
(numpy) or `ZeroDivisionError` (python float); `inf` then yields `nan` downstream (`inf*0` in the cooling
integrand, `inf-inf` in `Ed`), and the next `solve_R1(Eb=nan)` logs the reported error.
The `r2 += 1e-10` guard at `:224` is applied in **cm** (`r2 ≈ 2e19 cm`), so it is numerically
meaningless and does **not** prevent the zero denominator.

**It strikes in either energy sub-phase** (same degeneracy, two call paths, both currently unguarded):
- **Phase 1a** (`run_energy_phase.py:159` → `bubble_luminosity.get_bubbleproperties_pure` →
  `solve_R1` @ `:422`): the real Helix crash. The segment-0 ODE already produced `Eb=nan`; loop-1's
  bubble solve calls `solve_R1(nan)` → raise → **uncaught** → run dies.
- **Phase 1b** (`run_energy_implicit_phase.py:798` → `get_betadelta.compute_R1_Pb` → `bubble_E2P`):
  the local repro. The beta-delta solve fails to find a physical `dMdt>0` root (handled — "Holding last
  physical dMdt"), but `compute_R1_Pb` sits **outside** that guard and divides by zero → run dies.

### Why mass-dependent (the regime boundary)
`mCloud=1e7` (same `nCore`, `sfe`, `PISM`, `nISM`) runs healthy through 1a (95 segments) into 1b — `Eb`
grows. Only the catastrophic-cooling band (high cluster mass → `Lcool > Lmech`) collapses `Eb`. The
matrix (§5) will pin the mass/density threshold.

## 3. Key finding from the sim-free probe (`harness/probe_degeneracy.py` → `data/probe_degeneracy.csv`)

Sweeping `Eb` from `1e9 → 1e-8` at the crash state (`R2=7.0475, Lmech=5.07e12, v_mech=3739`):

| Eb | R1 | R2−R1 | rel shell vol | **Pb (baseline)** | Pb (rel-vol floor 1e-6) |
|---|---|---|---|---|---|
| 1e9 | 6.8101 | 2.37e-1 | 9.8e-2 | 4.65e6 | 4.65e6 |
| 1e6 | 7.04729 | 2.46e-4 | 1.0e-4 | 4.345e6 | 4.345e6 |
| 1e2 | 7.047540 | 2.46e-8 | 1.0e-8 | **4.345e6** | 4.5e4 |
| 1e-2 | 7.047540 | 2.46e-12 | 1.0e-12 | **4.345e6** | 4.5e0 |
| **1e-3** | 7.047540 | **0.0** | **0.0** | **inf** | 4.5e-1 |
| 1e-8 | 7.047540 | 0.0 | 0.0 | **inf** | 4.5e-6 |

Two things the table makes undeniable:

1. **The bubble pressure is analytically finite and ~constant (`~4.345e6`) all the way down.** It does
   *not* diverge as `Eb→0`; `R1` self-adjusts so the shell volume `∝ Eb` and `Eb/vol` stays fixed. The
   `inf` appears only at `Eb≲1e-3`, where `R2−R1` underflows below float64 resolution (~1e-13 relative)
   and `R2³−R1³` — a difference of two nearly-equal ~350 values — **rounds to exactly 0**. This is
   **catastrophic cancellation**, not a real pole.
2. **There is a cancellation-free identity.** At the `get_r1` root, `R1² = Lmech/(v·Eb)·(R2³−R1³)`, i.e.
   `R2³−R1³ = R1²·v_mech·Eb/Lmech` exactly. Substituting into `bubble_E2P`:
   `Pb = (γ−1)/(4π/3) · Lmech/(v_mech·R1²)` — **no subtraction of near-equal numbers, and the constant
   `4.345e6` falls straight out** (matches the table). This is just the wind ram pressure at `R1`.

So the crash has **three** orthogonal fix levers, which the matrix will compare head-to-head.

## 4. Candidate fix ideas (to evaluate as monkeypatched variants — production untouched)

| id | idea | what it changes | scope | hypothesis |
|---|---|---|---|---|
| **V0** | baseline | — | — | crashes on the `5e9/n1e2` band (reference) |
| **V1** | **rel-volume floor / `R1<R2` clamp** in `bubble_E2P` | floor `R2³−R1³ ≥ ε·R2³` (or clamp `R1≤R2(1−ε)`) | ~2 lines | kills the `inf`; below the cliff `Pb∝Eb` (decays) — does the run survive & reach the existing momentum transition? |
| **V2** | **cancellation-free `Pb`** via the `get_r1` identity | compute shell volume as `R1²·v·Eb/Lmech` (no subtraction) | ~3 lines | `Pb` stays at the analytic `~4.345e6`; most faithful to Weaver — but only valid at the `solve_R1` root |
| **V3** | **`isfinite` gate → `BubbleSolverError`** | reject non-finite `Pb`/`T`/profile in the bubble solve; ensure **both** 1a & 1b catch it → clean termination w/ reason | ~5–10 lines | converts crash → handled stop; belt-and-suspenders for *any* nan source (incl. cooling-cube holes) |
| **V4** | **degeneracy → graceful momentum transition** | detect `Eb`-collapse / `R1→R2` / `(Lgain−Lloss)/Lgain<thr` *before* the divide → break to the existing `cooling_balance` exit (→ momentum phase) | medium | the physically-correct end-state: the cloud is momentum-driven; produce *valid* science output, not a stop |
| V5 | `Eb` floor (clamp `Eb≥Eb_min`) | — | ~1 line | likely **rejected** (fabricates energy); included as a negative control |

**Note on V4 vs the existing machinery:** the code *already* has a `cooling_balance` energy→momentum
transition (`run_energy_implicit_phase.py:1072`, `(Lgain−Lloss)/Lgain < 0.05` → break). The crash fires
*before* it can. A central question the matrix answers: **does V1 or V2 alone (just stop the crash) let
these clouds reach that existing transition on their own?** If yes, the minimal numeric guard *is* the
fix and V4 is already implemented — we just need to not crash before it.

## 5. Empirical matrix (config × idea — the hybr-style de-risk)

Each idea is a **monkeypatched variant** (production untouched), run across a regime sweep, scored on
**robustness + no-op-on-healthy + science end-state**.

### Configs (degenerate → healthy)
- **Failing band (must stop crashing, must reach a sane end-state):** `5e9/n1e2` at
  `{sfe=0.05,PISM=0,nISM=0.1}` (the real Helix point), `{sfe=0.1,PISM=1e4,nISM=0.1}` (local repro),
  `{sfe=0.1,PISM=1e6,nISM=1}`.
- **Threshold scan (where does the regime start?):** `nCore=1e2, sfe=0.1` × `mCloud ∈ {1e8,5e8,1e9,5e9}`;
  and `mCloud=5e9, sfe=0.1` × `nCore ∈ {1e2,1e3,1e4}`.
- **Healthy controls (fix MUST be a no-op — target bit-identical):** `mCloud ∈ {1e5,1e6,1e7}` at
  `nCore=1e2, sfe=0.1`.

### Metrics (CSV schema — `data/eval_<idea>.csv`, comparable cells)
`config, variant, crashed(bool), crash_phase, crash_excpt, end_reason, reached_phase, n_seg_1a,
final_t, final_R2, final_v2, final_Eb, runtime_s, healthy_maxreldiff_vs_V0, notes`

### Gates
- **Robustness:** `crashed=False` on the entire failing band and threshold scan.
- **No-op:** on the healthy controls, every saved output column within round-off of V0
  (target `healthy_maxreldiff ≤ 1e-9`; V1/V2 are no-ops by construction when `R1≪R2`).
- **Science:** failing-band runs end in a *defensible* state — either a momentum-phase handoff
  (`reached_phase ≥ 1c/2`) or a clean termination with a recorded `SimulationEndReason` — **never** a
  traceback and never silent `nan` in the outputs.

### Bounded runs (tractability)
Phase 1a is slow (~1–2.5 min/run) because the no-approximation bubble solve runs per segment. Cap each
matrix run with a short `stop_t` (≈`0.05` Myr — these massive clouds evolve fast; the crash is at
`t~3e-3`) so a cell is a few minutes, enough to pass the crash point and observe the end-state.
Parallelise cells across subagents. Record the exact command per CSV.

## 6. Rollout (gated, mirrors the project's S0–S4 pattern)
- **S0 — sim-free probe (DONE).** `harness/probe_degeneracy.py` → `data/probe_degeneracy.csv`. Pins the
  cancellation cliff + the analytic identity. ✅
- **S1 — matrix harness.** `harness/variants.py` (the monkeypatches), `harness/run_variant.py` (drive
  one sim + emit a CSV row), `harness/params/*.param` (the config list). Production untouched.
- **S2 — run the matrix (subagents).** Fill `data/eval_*.csv` across all cells. Commit every CSV (💾).
- **S3 — verdict + implement the winner.** Pick on the gates; add the chosen guard/transition to
  production with a `test_*.py` that (a) reproduces the crash sim-free via the probe state and (b) pins
  the no-op on a healthy config. Update this status block + the §7 verdict.
- **S4 — regression.** `pytest` (+ `-m stress`) green; healthy outputs unchanged.

## 7. Verdict
_TBD — filled after S2/S3. Will record: winning variant, the gate table, and why._

## 8. Key references
- Degeneracy math: `trinity/bubble_structure/get_bubbleParams.py` — `get_r1` `:375-402`, `solve_R1`
  `:405-429`, `bubble_E2P` `:198-230` (the `r2+=1e-10`-in-cm dud guard `:224`, the divide `:228`).
- Call sites (all currently unguarded for `R1→R2`): phase 1a `run_energy_phase.py:95,159,322`;
  energy ODE `phase1_energy/energy_phase_ODEs.py:223,358`; phase 1b
  `phase1b_energy_implicit/get_betadelta.py:327,329` + `run_energy_implicit_phase.py:798`;
  phase 1c `phase1c_transition/run_transition_phase.py:505,747,832`; bubble solve
  `bubble_structure/bubble_luminosity.py:422,428`.
- Existing energy→momentum transition (V4 target): `run_energy_implicit_phase.py:1072` (`cooling_balance`).
- `BubbleSolverError` (V3 target): `bubble_structure/bubble_luminosity.py:98,152,521,536,958`.
- Latent secondary nan source (cooling-cube holes, high-phi/low-n): `cooling/non_CIE/read_cloudy.py:95-97,133`
  (RegularGridInterpolator, default `bounds_error=True`; NaN query → silent NaN). Not the primary trigger
  for the standard high-Pb large cloud, but covered by V3.
- Repro configs: `harness/params/fail.param` (= `/tmp/fail.param`), `harness/params/control.param`.

