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

**Status (2026-06-19):** 🟡 **DIAGNOSED + REPRODUCED; first smoke result in — numeric guard is
necessary but NOT sufficient.** Two independent investigations + a sim-free probe + two live repros agree
on the mechanism. **Smoke test of V3 (both guards) on `fail_repro`:** it *does* stop the divide-by-zero
crash, but the run then drives **`Eb` through zero into NEGATIVE** (`+7.4e8` @ phase-1a end →
`−9.1e8` in phase 1b → `−1.0e12`), producing a negative bubble energy/pressure and grinding (timed out at
320 s, never terminated). Trajectory: `data/smoke_V3_fail_repro_trajectory.csv`. **Implication: clamping
the geometry (V1/V2/V3) only converts a crash into negative-energy garbage — the real fix must detect the
energy collapse (`Eb→0` / cooling-dominated) and *transition to the momentum phase* (V4) before `Eb`
crosses zero.** V4 is now the leading candidate; the numeric guard becomes a safety net. **No production
code changed yet.**

---

## 0. Re-baselined against `main @ 6bdba8de` (2026-06-19, re-merged twice)

This plan was first drafted against an older `main`, then rebased onto `946e860b` (PR #697), then merged
up again to `6bdba8de` (PR #698 + the "info-driven logging" commit `6f3aeab9`) — re-verified line-by-line
each time. **The bug still reproduces** (V0 on `fail_repro`, new code: identical `R1→R2` degeneracy at
`R2≈8.6`, same `Rejected. min T: 29999.99…` grind; `data/reverify_V0_main_946e860b.csv`). Drift + protocol notes
(line numbers below are current as of `6bdba8de`):

- **`get_bubbleParams.py` — refs UNCHANGED & re-confirmed:** `bubble_E2P:198`, `r2+=1e-10` (still in cm,
  still a dud) `:224`, the divide `:228`, `get_r1:375` / its equation `:400`, `solve_R1:405`, error msg
  `:426`. The core mechanism is byte-stable.
- **`bubble_luminosity.py` — REFACTORED (PR: "regroup into logical sections (bit-identical)", "drop
  `_legacy`"), refs MOVED:** the `_get_velocity_residuals` `Rejected. min T` branch `:910-913 → :308-311`;
  `_T_INIT_BOUNDARY=3e4` `:52 → :51`; the solver is now `solve_ivp` (`sol.success`/`sol.y`) but the
  rejection logic is identical (penalty `(3e4/(min_T+0.1))² ≈ 0.999994`, a no-op; early `return :311`
  still shadows the `nan`/`monotonic` checks `:313,:317`).
- **`run_energy_phase.py` — refs shifted +3 by the new INFO logging:** bubble call `:159→:162`,
  `solve_R1 :95→:96`, switch `:293-295→:296-298`, final `solve_R1 :322→:325`.
- **`run_energy_implicit_phase.py` — `compute_R1_Pb :798` unchanged; `cooling_balance :1072-1074→:1077-1078`**
  (the "safety net, NOT a transition trigger" comment moved with it).
- **`get_betadelta.py` (+411 lines) / `run_energy_implicit_phase.py`:** changed materially but the
  `compute_R1_Pb` → `solve_R1`/`bubble_E2P` degeneracy path is intact. Harness monkeypatch targets
  (`get_bubbleParams.solve_R1`/`.bubble_E2P`, reached by module-attribute everywhere) remain valid.
- **NEW (`main @6bdba8de`, "fixed logging verbosity — much more info driven"):** `trinity/_output/terminal_prints.py`
  adds `format_state` (`:163`) / `heartbeat` (`:187`) / `format_end_report` (`:205`), logged at **INFO**
  from the phase runners. `format_state` reports `R2,v2,Eb(erg),Pb,R1` and **renders `nan`/`inf` literally**
  (`:147`) — so for *this* bug the collapse (`Eb→negative/nan`) is now visible at **INFO**, no DEBUG needed.
  This corroborates the logging guidance below (§ Bounded runs): run the matrix at INFO, read the
  structured trajectory, reserve DEBUG for targeted module dives.

**Planning-protocol adoption (new `CLAUDE.md` rule 5 + "size the change first" ladder).** The fix lives
in a solver/iterative path ⇒ **Risky/iterative**, so it must run the full ladder: gate-first, baseline
capture, **full-run equivalence on the stiffest edge regimes in separate processes at matched `t`** (not
just per-call), smallest diff, re-verify (gate + `pytest` + ruff F-rules), persist. The **no-op gate**
here is the strong form: on healthy configs the fix must be **bit-identical** (value-diff vs `git show
HEAD` *and* byte-identical `dictionary.jsonl`) — by construction for the clamp (never active when
`R1≪R2`), to be proven for any transition change. Edge configs to use: `param/simple_cluster.param` +
`docs/dev/performance/f1edge_{lowdens,hidens}*.param`, **plus** the `5e9/n1e2` crash band itself.

**Relationship to the `docs/dev/transition/` workstream (`TRIGGER_PLAN.md`) — important, do not collide.**
That is a *measurement-first* investigation of *when* the **normal** implicit→momentum transition should
fire (candidate families F0–F5; not yet wired to production; default stays F0 `instantaneous`). Three
facts from it bear directly on this fix:
1. **Phase 1a has no transition trigger at all** — it ends only at `TFINAL_ENERGY_PHASE` or geometric
   events; *all* transition logic is in 1b. So a cloud whose `Eb` collapses *during 1a* (the real Helix
   crash) has no escape hatch — that is a robustness gap 1a-side, independent of the trigger study.
2. Its **reference physical event is the PdV-inclusive net-energy zero-crossing**
   `(Lgain − Lloss − 4πR2²·v2·Pb) ≤ 0` — the **`Eb`-peak**. That is *exactly* where my trajectory's `Eb`
   stops growing and starts collapsing. **V4 should detect *this* (energy no longer being gained), not a
   bespoke `Eb≤ε` hack**, so it stays consistent with the trigger work.
3. This bug is the **catastrophic-cooling extreme** of the same transition question (the bubble is
   cooling-dominated essentially from birth, so the `Eb`-peak is immediate). The trigger study optimizes
   *late* firing (the stall); **this is a correctness/robustness bug (crash + NaN/negative `Eb`)** and may
   land its minimal guard now without waiting on the trigger paper — but framed via the same net-energy
   event so the two never disagree. Scope guard: **do not** re-open the F0–F5 trigger choice here.

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

## 4. Candidate fix families & the harness variants

Two layers, kept distinct so labels never collide:

**(a) Numeric-guard variants — what the harness (`harness/variants.py`) actually monkeypatches & ran.**
These probe whether *just stopping the divide-by-zero* is enough.

| id | patch | hypothesis |
|---|---|---|
| **V0** | baseline (no patch) | crashes on the `5e9/n1e2` band (reference) |
| **V1** | clamp `R1 ≤ R2·(1−ε)` in `solve_R1` (ε=1e-6) | kills the `inf`; below the cliff `Pb∝Eb` |
| **V2** | floor the shell volume `R2³−R1³ ≥ ε·R2³` in `bubble_E2P` | same effect via the divide site only |
| **V3** | V1 + V2 | combined guard |

**(b) Fix families (the actual candidates a production fix would pick from):**

| id | family | what it changes | scope | role |
|---|---|---|---|---|
| **G — geometry guard** | = V1/V2/V3 | `R1<R2` / volume floor so the divide can't blow up | ~2 lines | **necessary safety net; proven NOT sufficient alone** (smoke below) |
| **C — cancellation-free `Pb`** | the `get_r1` identity `R2³−R1³ = R1²·v·Eb/Lmech` → `Pb=(γ−1)/(4π/3)·Lmech/(v·R1²)` | removes the catastrophic cancellation at its source | ~3 lines | optional conditioning; only valid at the `solve_R1` root |
| **F — loud-fail** | `isfinite` gate on `Pb`/`T`/profile → `BubbleSolverError`; ensure **both** 1a & 1b catch it → clean termination w/ reason | ~5–10 lines | belt-and-suspenders for *any* nan source (incl. cooling-cube holes) |
| **T — transition (leading)** | detect the **PdV-inclusive net-energy zero-crossing** `(Lgain − Lloss − 4πR2²·v2·Pb) ≤ 0` (the `Eb`-peak) → hand off to the momentum phase | medium | the physically-correct end-state; **aligned with `docs/dev/transition/` (the `Eb`-peak event), NOT a bespoke `Eb≤ε`** |

**Partial empirical answer (2026-06-19 smoke, V3 on `fail_repro`): geometry guard alone is NOT enough.**
With the geometry clamped the energy ODE keeps integrating and `Eb` crosses **zero into negative**
(`+7.4e8 → −9.1e8 → −1.0e12`), giving negative `Pb`; the bubble solve then has no physical solution →
fsolve thrashes → `Rejected. min T` spam → no termination in 320 s (`data/smoke_V3_fail_repro_trajectory.csv`).
The existing `cooling_balance` break (`run_energy_implicit_phase.py:1072-1074`) is *never reached* — the
grind happens earlier in the iteration (the bubble/beta-delta solve `~:798`), not at the end-of-loop
transition check. **So family T must fire the handoff at the net-energy zero-crossing, before `Eb` goes
non-positive** (≈ snapshot 48, `t≈2.8e-3`, `R2≈8.4`, `Eb` still `+7.4e8` but plunging, `R2−R1≈0.09`);
**G stays as the safety net** so the divide can never blow up even if T mis-times. **Open question for the
matrix:** does the existing momentum/transition machinery accept a handoff this early cleanly (continuity
of `Eb`,`R2`,`v2`,`P_drive`), and does T leave the healthy configs **bit-identical**?

**On the `Rejected. min T: 29999.99` noise (re-verified on new code):** benign. The bubble structure
integrates *from* `T=3e4` inward; `min_T=29999.99` is a `1.6e-5 %` dip below `_T_INIT_BOUNDARY=3e4`
(`bubble_luminosity.py:51`) — the documented "boundary_transient" (`:867,:918`). The penalty it returns,
`(3e4/(min_T+0.1))² ≈ 0.999994`, is effectively `1.0` — a *no-op* "rejection" (logs, doesn't steer fsolve).
It is a *symptom* of fsolve thrashing on the negative-`Eb` bubble, not a cause. Two minor *orthogonal*
cleanups (do NOT bundle into the fix): (a) the early `return :311` shadows the `nan`/`monotonic` checks
(`:313,:317`); (b) `min_T < 3e4 − tol` would stop the false trip + the log spam.

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

### Bounded runs (tractability) — corrected 2026-06-19
Phase 1a/1b are slow (the no-approximation bubble solve runs per segment) and **`stop_t` does NOT bound
wall-time** — the energy phases loop on internal `TFINAL_ENERGY_PHASE`/segment constants, not `stop_t`,
and the slowness is per-segment solve cost in the degenerate regime. So **bound each cell with a wall-clock
`timeout`** (the smoke ran 320 s and was SIGTERM'd mid-grind; V0 crashes cleanly in ~110 s). Treat
**three** outcomes as distinct in the CSV: `crashed` (V0), `completed` (clean `end_reason`), and
`timeout`/`SystemExit:143` (no termination — the V3 grind). The harness reads the run's `dictionary.jsonl`
for the final `(t,R2,Eb,Pb,R1,phase)` so a timed-out cell still yields its progress + whether `Eb` went
negative. Parallelise cells across subagents; record the exact command + `timeout` per CSV.

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

## 8. Key references (re-verified against `main @ 946e860b`, 2026-06-19)
- Degeneracy math: `trinity/bubble_structure/get_bubbleParams.py` — `get_r1` `:375-402`, `solve_R1`
  `:405-429`, `bubble_E2P` `:198-230` (the `r2+=1e-10`-in-cm dud guard `:224`, the divide `:228`).
- Call sites (all currently unguarded for `R1→R2`): phase 1a `run_energy_phase.py:96,162,325`;
  energy ODE `phase1_energy/energy_phase_ODEs.py:223,358`; phase 1b
  `phase1b_energy_implicit/get_betadelta.py:297(compute_R1_Pb),327,329` + `run_energy_implicit_phase.py:798`;
  phase 1c `phase1c_transition/run_transition_phase.py:505,750,835`; bubble solve
  `bubble_structure/bubble_luminosity.py:175,181`.
- Existing energy→momentum transition (family T context): `run_energy_implicit_phase.py:1077-1078`
  (`cooling_balance`); the principled `Eb`-peak / net-energy event is owned by `docs/dev/transition/TRIGGER_PLAN.md`.
- `BubbleSolverError` (family F target): `bubble_structure/bubble_luminosity.py:105(class),298,428(except),361,569,584(raise)`.
- The benign `Rejected. min T` branch: `bubble_luminosity.py:308-311`; `_T_INIT_BOUNDARY=3e4 :51`;
  boundary-transient note `:867,:918`.
- Latent secondary nan source (cooling-cube holes, high-phi/low-n): `cooling/non_CIE/read_cloudy.py:95-97,133`
  (RegularGridInterpolator, default `bounds_error=True`; NaN query → silent NaN). Not the primary trigger
  for the standard high-Pb large cloud, but covered by family F.
- Repro configs: `harness/params/fail_repro.param` (sfe0.1/PISM1e4), `harness/params/fail_helix.param`
  (the real Helix sfe0.05/PISM0 point), healthy controls `harness/params/small_1e{5,6,7}.param`.
- Sibling workstream: `docs/dev/transition/TRIGGER_PLAN.md` (+`P0.md`,`pshadow-design.md`) — the normal
  implicit→momentum trigger study (families F0–F5); align family T with its `Eb`-peak event, don't collide.

