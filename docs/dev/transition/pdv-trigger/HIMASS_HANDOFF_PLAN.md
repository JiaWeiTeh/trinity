# High-mass cluster: energy→momentum handoff without `Eb→0` / `ebpeak` — PLAN

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
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

**Status:** PLAN ONLY (2026-06-30). No production code touched. Branch
`bugfix/high-mass-cluster-transition-without-ebpeak`. Siblings: `FINDINGS.md` (the diagnosis data),
`PLAN.md` (the κ_eff/trigger strategy), `RUNGB_SCOPING.md`.

---

## 1. Problem (verified against current source, 2026-06-30)

A massive/dense cloud loses the bubble's thermal energy faster than it can build it up, so `Eb`
turns over and falls through zero **inside the explicit energy phase (1a)**. There, the guard

```python
# The SAME guard exists in BOTH energy phases (identical logic):
#   trinity/phase1_energy/run_energy_phase.py:340            (explicit, phase 1a)
#   trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1074  (implicit, phase 1b)
if not np.isfinite(Eb) or Eb <= 0:
    params['EndSimulationDirectly'].value = True   # SimulationEndCode.ENERGY_COLLAPSED (51)
    ... break
```

sets `EndSimulationDirectly=True`, which **no-ops phases 1c AND 2** (each is gated
`if params['EndSimulationDirectly'].value == False`, `main.py:283,303,343`). The run stops dead at
`ENERGY_COLLAPSED` instead of continuing as a momentum-driven bubble. **That dead-stop is the bug.**

> **Verified fire site (live, `fail_repro`):** the diffuse-massive collapse ran **52 steps in the explicit
> phase with `Eb>0`** (1a's guard never tripped), handed to the **implicit** phase, and the **first implicit
> step drove `Eb`→−9.1e8, tripping the guard at `run_energy_implicit_phase.py:1074`**. So for the failing
> regime the dead-stop that actually fires is the **1b (implicit)** one — that is the **primary fix site**;
> 1a:340 is the secondary (a config could collapse earlier, within 1a).

### What is NOT the mechanism (corrected diagnosis)
- The default `transition_trigger` is **`cooling_balance`** (`default.param:282`), not `ebpeak`.
  `ebpeak` (`edot_balance ≤ 0`, PdV-inclusive) is **opt-in and shadow-only** — logged, never drives the
  switch (`run_energy_implicit_phase.py:199-212,1167-1205`). So "massive clusters cool via `ebpeak`" is
  **false**; the diffuse-massive collapse dies at the `Eb<=0` guard in **1b** (verified: `fail_repro` above),
  because radiative `cooling_balance` can't fire when radiative ≪ Lmech.
- The driver of the turnover is **density-dependent** — and this was settled by a **fresh live run
  against current code (2026-06-30)**, not the committed CSVs. ⚠️ The committed
  `data/pdv_regime_budget.csv` is **post-processing of frozen trajectory CSVs of unknown/old provenance**
  (its harness `make_pdv_regime_table.py` reads `cleanroom/data/c0_*` + `failed-large-clouds/data/budget_*`,
  it runs **no sim**) — treat its magnitudes as stale-risk. Live re-measurement
  (`data/live_pdv_decomp.csv`, harness `data/make_live_pdv_decomp.py`; `PdV = 4πR2²·Pb·v2` exactly as
  `energy_phase_ODEs.py:280`):

  | config (live, **full run**) | mass, nCore | at Eb-peak: PdV / L_bub (·Lmech) | Eb fate | reached momentum? | **dead-stop?** |
  |---|---|---:|---|---|---|
  | fail_repro | 5e9, **1e2** (diffuse) | **0.99 / 0.014** | Eb → **−9.1e8** (strictly negative) | **No** (dies in `implicit`) | **YES — the bug** |
  | f1edge_hidens | 1e7, **1e6** (dense) | 0.09 / **0.92** | Eb floors to 0 via `transition` | **Yes** (`energy→implicit→transition→momentum`) | No (clean) |

  **The key result — the dead-stop bug is regime-specific** (both configs run to completion; see §5 for the
  partial-data correction that this supersedes):
  - **Diffuse-massive** (`5e9, n=1e2`): **PdV dominates** (0.99 vs 0.014·Lmech at the peak; median PdV/Lmech
    1.43). The heavy shell soaks the wind energy as bulk motion; the hot, tenuous interior barely radiates.
    Radiative `cooling_balance` (needs `Lloss/Lgain>0.95`) **can never fire** because radiative is ~1% of
    Lmech — so nothing hands it off, Eb crashes **strictly negative**, and the `Eb<=0` guard dead-stops the
    run in the implicit phase (`reached_momentum=False`). **This is the bug, and it is the PdV-dominated
    regime.**
  - **Dense-massive** (`1e7, n=1e6`): **radiative dominates** at the turnover (0.92 vs 0.09·Lmech at the
    Eb-peak; median L_bub/Lmech 0.36 > PdV 0.28). Radiative `cooling_balance` **fires normally**, so the
    cloud walks through `transition` (Eb floored at `ENERGY_FLOOR=1e3`) into `momentum` — **no dead-stop, no
    bug**. The existing default chain already handles it correctly.
  - **Verdict on the external "it's `L_bubble`, not PdV" claim:** **wrong for the case that actually breaks.**
    The regime that dead-stops (diffuse-massive) is PdV-dominated with radiative negligible; the analysis's
    "radiative drives the collapse" describes the dense case, which **does not break**. So the collapse the
    fix must catch is PdV-driven. PdV is never the negligible self-damping term the analysis claimed, and
    `Pb` is not `∝Eb` (identity below). Original "PdV matters" intuition: **confirmed for the failing case.**
  - **Scope sharpened:** the pressure-crossover trigger (§2) is the **safety net for the PdV-dominated
    regime where `cooling_balance` structurally cannot fire** (radiative ≪ Lmech). It must *not* perturb the
    radiative-dominated dense clouds, which already hand off correctly (→ G0 bit-identical gate covers them).
  - Provenance: the committed `pdv_regime_budget.csv` reproduced for `fail_repro` (1.43 live ≈ 1.42 CSV) but
    is frozen-trajectory post-processing (stale-risk); trust `data/live_pdv_decomp.csv`. `FINDINGS.md` §1/§6a
    flagged accordingly.

### Why `Eb→0` is the wrong trigger point (this part of the external analysis is correct)
At collapse the energy-driven structure is singular: `solve_R1` drives `R1→R2`, so
`shell_volume = R2³−R1³ → 0` and `bubble_E2P = (γ−1)Eb/V → 0/0 → nan`
(`get_bubbleParams.py:229-235,421-426`). Triggering the handoff exactly there asks the model to switch
from its single most ill-conditioned state.

### Key identity (current code, numerically verified 2026-06-30)
Because `R1=solve_R1(R2,Eb,…)` satisfies `R2³−R1³ = R1²·v_mech·Eb/Lmech`, the `Eb` in `bubble_E2P`
**cancels**:

```
Pb  = (γ−1)·Lmech / [(4π/3)·R1²·v_mech]                 (Eb enters only via R1)
PdV/Lmech = 2 (R2/R1)² (v2/v_mech)            [γ=5/3]
```

So `Pb ∝ Eb` holds **only while R1≪R2** (high Eb); as `Eb→0`, `R1→R2` and `Pb` **floors** (it does not
go to zero), and `PdV/Lmech → 2(v2/v_mech)`. The external analysis's premise `press_bubble ∝ Eb` and its
"PdV is a self-damping term that can't push Eb negative" both fail in the collapse regime. Verification
harness to re-run: see §6 (a ~1-second script, no full sim).

## 2. Fix concept — trigger on the pressure crossover, route into the existing 1c→2 chain

The handoff machinery **already exists** and is exactly what the physics wants:
- **Phase 1c (transition):** decays `Eb` using `P_drive = max(Pb, P_HII + P_ram)`
  (`run_transition_phase.py:331`), with a terminal event at `Eb < ENERGY_FLOOR = 1e3` (`:766`).
- **Phase 2 (momentum):** sets `Eb = 0` and drives on ram pressure only (`run_momentum_phase.py:511`).

The energy-driven phase is *done* — physically — when thermal driving has decayed to the momentum
level, i.e. the **pressure crossover**

```
Pb  ≲  P_HII + P_ram                  (θ→ the momentum-driving pressure, seen from the pressure side)
```

At that point `P_drive = max(Pb, P_HII+P_ram)` is continuous by construction, so the velocity ODE does
not jump across the switch. This is well-defined while `Eb` is small-but-finite, so we never enter the
`0/0` region.

### Design (smallest change that removes the dead-stop)
1. **Detect the crossover in the explicit phase (1a).** Add a `solve_ivp` **terminal event**
   `g(t,y) = press_bubble(Eb,R2,R1) − (P_HII + P_ram)` (terminal, `direction=-1`) in
   `run_energy_phase.py`, alongside the existing events. Root-finding on the dense-output interpolant
   locates the crossing inside the step (no reliance on `Eb` landing near zero → no overshoot problem).
2. **Make the RHS finite at an overshot trial point** so the root-finder can bracket: clamp
   `press_bubble = max(0.0, bubble_E2P(...))` and confirm `solve_R1` already returns finite for
   `Eb≤0` / `R2≤0` (it does — `:431-434`). This is the only change to the hot RHS and must clear the
   bit-identical gate on healthy bubbles (the clamp is a no-op when `Pb>0`).
3. **Route, don't stop.** On the crossover event, end phase 1a *normally* (do NOT set
   `EndSimulationDirectly`), carrying `(R2, v2, Eb)` so `main.run_expansion` proceeds into 1c→2 exactly
   as a `cooling_balance` transition does today. Keep `EndSimulationDirectly` + `ENERGY_COLLAPSED` only
   for genuine endings (rCloud, dissolution, stop_t) — never for the energy→momentum transition.
4. **Open question — 1c vs straight-to-momentum.** If the crossover fires essentially at ignition (the
   stillborn massive cloud, `A<0` from birth, PdV-starved), there is no energy epoch for 1c's `Eb`-decay
   to walk out of; routing straight to momentum is the honest representation. If a meaningful energy
   epoch remains, go through 1c. **Decide by measurement** (§5), not assumption — default to the
   lower-risk "always through 1c" unless 1c is shown to be ill-conditioned at near-zero `Eb`.

### Design addendum — TRIGGER PARITY between 1a and 1b (2026-07-01, maintainer point)

Both energy phases carry the identical `Eb<=0` dead-stop (1a:340, 1b:1074), but **only 1b evaluates a
transition criterion** (`cooling_balance`, `run_energy_implicit_phase.py:1207`). 1a (the fixed ~3000-yr
`TFINAL_ENERGY_PHASE=3e-3 Myr` early phase) has **no** transition check, so a cloud that would hand off
*within* the first 3000 yr — a **violently cooling** dense cloud — instead either waits for the 1a→1b
boundary or, if cooling drives `Eb<0` first, **hits 1a's own dead-stop and dies**. Both energy phases
must evaluate the **same two triggers** so neither regime can reach `ENERGY_COLLAPSED`:

- **`cooling_balance` in 1a is cheap and faithful — no betadelta needed.** The 1b ratio is
  `(Lgain − Lloss)/Lgain` with **`Lgain = Lmech_total`** (`get_betadelta.py:466`, runner `:1142`) and
  **`Lloss = bubble_LTotal + bubble_Leak`** (default `effective_Lloss`). All three inputs are already
  computed every 1a step (`bubble_Lgain`/`bubble_Lloss` are nan in 1a, but they are just unfilled
  betadelta *output* slots — the ratio does not use them). Verified on a live `fail_repro` 1a row:
  `(1.014e13 − (1.315e11 + 0))/1.014e13 = 0.987`. Caveat: 1a's `bubble_LTotal` is the direct
  (non-β/δ-resolved) estimate, so its value differs slightly from 1b's resolved one — the *criterion* is
  identical, only the cooling estimate is 1a's cruder one (which is what "constant/early cooling" already
  means). Accept the small inconsistency.
- **Pressure crossover in 1a** (points 1–3 above) catches the **PdV-dominated** collapse that
  `cooling_balance` structurally cannot (there the 1a ratio sits at ~0.99 — radiative ~1%). `Pb`, `P_ram`,
  `P_HII` are all computed in 1a.

So the fix installs **both** the `cooling_balance` check and the pressure-crossover terminal event in
**1a as well as 1b**, each routing into the 1c→2 chain rather than dead-stopping. Whichever fires first
wins (same precedence logic as 1b's `r1_transition_decision`). This subsumes the earlier "primary site =
1b" framing: 1b remains where the *verified* `fail_repro` collapse lands, but parity in 1a is required for
the violent-early-cooling regime. **Still gated by G0** — adding these checks must stay bit-identical on
runs that don't trip them (both are `> threshold` / `Pb > P_ram` no-ops for healthy bubbles).

> Scope guard: this is a **solver-edge** change (rule 5 / planning protocol "risky"). The diff should be
> confined to (a) one terminal event + clamp in `run_energy_phase.py`, (b) the routing branch, and (c) an
> analogous crossover handling in 1b if 1a hands off to 1b first. No new params, no κ_eff coupling, no
> touching `cooling_balance`/`ebpeak`.

## 3. Equivalence gates (define BEFORE editing — rule 5)

- **G0 — bit-identical on healthy bubbles.** On configs whose energy phase does NOT collapse
  (`param/simple_cluster.param`, `docs/dev/performance/f1edge_lowdens*.param`), the new event +
  `max(0.0, Pb)` clamp must be a **no-op**: value-diff vs `git show HEAD` AND byte-identical
  `dictionary.jsonl`, full runs in separate processes at matched `t`. The clamp only changes behaviour
  when `Pb≤0`, which a healthy bubble never reaches.
- **G1 — single-step continuity at the crossover.** At the located event time, `P_drive` and `vd` from
  the energy-phase RHS equal those from the transition-phase RHS to tolerance (necessary, not
  sufficient).
- **G2 — full-run handoff on the stiff edge.** On a genuinely collapsing config (the dense/heavy edge —
  `small_dense_highsfe`, and a heavy-mass case in the `fail_repro`/5e9 family), the run must continue
  past the old `ENERGY_COLLAPSED` point into momentum and reach a physical stopping fate (rCloud /
  dissolution / stop_t), with `R2(t)`, `v2(t)` continuous across the switch. Separate processes, matched
  `t`.
- **G3 — regression.** Full `pytest` green + ruff F-rules. Add a focused `test_*.py`: a collapsing config
  must end on a momentum/stopping fate, NOT `ENERGY_COLLAPSED` — the failing-test-first for this bug.

## 4. Baseline to capture before editing
- `git show HEAD` values + byte hash of `dictionary.jsonl` for the G0 configs.
- The current `ENERGY_COLLAPSED` trajectory (R2,v2,Eb,t at the stop) for the stiff-edge configs, saved as
  a committed CSV under `data/` — this is the "before" the handoff is measured against.

## 5. Measurement to settle the 1c-vs-momentum question (§2.4)
Replay the crossover point on the existing frozen trajectories (no full re-run): at the located
`Pb=P_HII+P_ram` time, report `Eb/Eb_peak`, `R1/R2`, and remaining energy epoch length. If
`Eb/Eb_peak ≲ few %` and `R1/R2→1` at the crossover (stillborn), route straight to momentum; else 1c.
Persist as `data/crossover_epoch.csv`.

**Settled (live, 2026-06-30, `data/live_pdv_decomp.csv` — both runs now COMPLETE):**
- The diffuse-massive `fail_repro` (5e9, n=1e2) is the **stillborn / PdV-dominated** case — 52 explicit
  steps with Eb>0, then the first implicit step drives Eb strictly negative → `ENERGY_COLLAPSED`,
  `reached_momentum=False`. This is the one that needs the fix; route it to momentum.
- The dense-massive `f1edge_hidens` (1e7, n=1e6) **did complete** (126 rows; the earlier "90-row stiff/
  partial" read is superseded). It is **radiative-dominated** (L_bub/Lmech 0.92 vs PdV 0.09 at the Eb-peak)
  and **already transitions correctly**: `energy→implicit→transition→momentum`, Eb floored at
  `ENERGY_FLOOR`, `dead_stop=False`. **It does NOT need the fix and must not be perturbed** (G0).

So the 1c-vs-momentum decision (§2.4) only has to be made for the PdV-dominated collapse, and there the
answer is **straight to momentum** (no meaningful energy epoch remains — Eb barely grew). The `small_dense_highsfe`
(1e4, n=1e6 — dense but *light* shell, isolating radiative from inertial loading) run is still completing;
its row will land in `live_pdv_decomp.csv` and is expected to look like `f1edge_hidens` (radiative-driven,
clean handoff, no dead-stop) — confirming density, not mass, sets the sink balance.

## 6. Reproduce the §1 identity check (no sim, ~1 s)
```
python3 - <<'PY'
import numpy as np, trinity.bubble_structure.get_bubbleParams as gp
R2,Lmech,vmech,g,v2 = 5.0,1e4,3000.0,5/3,30.0
for Eb in [1e6,1e3,1e0,1e-3]:
    R1=gp.solve_R1(R2,Eb,Lmech,vmech); Pb=gp.bubble_E2P(Eb,R2,R1,g)
    print(f'Eb={Eb:.0e} R2/R1={R2/max(R1,1e-9):7.2f} Pb/Eb={Pb/Eb:.3e} PdV/Lmech={4*np.pi*R2**2*Pb*v2/Lmech:.4f}')
PY
```
Expected: `Pb/Eb` constant at high Eb (Pb∝Eb), blows up as Eb→0 (Pb floors); `PdV/Lmech` falls
1200→1.2→0.02. Confirms `Pb` is NOT ∝ Eb near collapse.

## 7. Risks
- Trinity leaks module-level global state in-process → all equivalence runs MUST be separate processes
  at matched `t` (CLAUDE.md rule 5).
- numpy pinned `<2`; the monotonic guard in the bubble integrator is fragile — do not bump.
- If 1a hands to 1b before the crossover, the same crossover handling is needed in
  `run_energy_implicit_phase.py`; check which phase the collapse actually occurs in per config before
  deciding where the event lives.
