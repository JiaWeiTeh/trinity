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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status update (2026-07-01): ⛔ the 1b case below is SUPERSEDED — a finite `Eb<=0` collapse in phase 1b now
ROUTES to the momentum phase instead of ending `ENERGY_COLLAPSED`; only non-finite `Eb` and phase-1a collapses
still end code 51. "Permanent fate" framings below are stale — see the ⛔ note under the table.**

**Status (2026-06-19):** 🟢 **FIX COMPLETE — all three failing configs terminate cleanly; healthy no-op
confirmed (byte-identical); `pytest` 555 passed.** Shipped G+F+ the phase-1a collapse coverage (below).
Verification (`data/verify_extended_fix_all_configs.csv`):

| config | collapse phase | crashed | end | final R2 |
|---|---|---|---|---|
| `fail_repro` (sfe0.1/PISM1e4) | 1b | **False** | `ENERGY_COLLAPSED` (51) | 9.73 |
| `fail_pism6` (sfe0.1/PISM1e6) | 1b | **False** | `ENERGY_COLLAPSED` (51) | 9.73 |
| **`fail_helix` (real Helix: sfe0.05/PISM0)** | **1a** | **False** | `ENERGY_COLLAPSED` (51) | 7.03 |
| `small_1e5`/`small_1e6` (healthy) | — | False | **no-op (byte-identical to pre-fix)** | — |

> **⛔ SUPERSEDED for the 1b case (2026-07-01) — the table above is now STALE.** On branch
> `bugfix/high-mass-cluster-transition-without-ebpeak`, a **finite** `Eb<=0` collapse in **phase 1b** no
> longer ends on `ENERGY_COLLAPSED` — it **ROUTES to the momentum phase** (via 1c), so `fail_repro` now runs
> `energy→implicit→momentum` out to the 500 pc stop radius (not `ENERGY_COLLAPSED`/R2=9.73). Only
> **non-finite** `Eb`, and **phase-1a** collapses (e.g. `fail_helix`, routing deferred), still end
> `ENERGY_COLLAPSED`. This plan's "ENERGY_COLLAPSED is the permanent fate" framing (incl. §"permanent
> behaviour" below) is **no longer true for finite 1b collapses.** Canonical doc:
> `docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md`.

Took **two** rounds: the first (G+F: `bubble_E2P` floor + post-ODE `Eb<=0` check) fixed the 1b configs but
the real Helix point collapses *inside* phase 1a, where the per-segment solves raise *before* the post-ODE
check — `solve_R1`→`get_r1` `sqrt(<0)` at an overshot `R2<0`, then the cooling table out-of-bounds at
`Eb≈0`. Round two added: `solve_R1` returns `0.0` for non-physical `R2<=0`, and the phase-1a bubble solve
is wrapped so any degenerate-collapse exception → `ENERGY_COLLAPSED`. Production diff total: ~93 lines
across 4 files + `test/test_energy_collapse_guard.py` (6 cases).

> **Update (2026-06-23):** the **"family T"** handoff this plan defers (the PdV-inclusive `Eb`-peak →
> momentum transition, §6) has since **partially shipped** in the sibling transition workstream as an
> **opt-in, default-off** `transition_trigger` keyword (`run_energy_implicit_phase.py`; values
> `cooling_balance` (default) / `blowout` / **`ebpeak`**). The `ebpeak` option **is** the net-energy
> zero-crossing `(Lgain − Lloss − 4πR2²·v2·Pb) ≤ 0` named below. The production **default is unchanged**,
> so these clouds still terminate `ENERGY_COLLAPSED` and this plan's verdict holds — but the
> "family T not wired / no production code changed yet" framing is **no longer literally true**. See
> `docs/dev/transition/pt4/r1shadow/R1_FINDINGS.md`. Still unbuilt: the heavy-cloud Eb-peak is a
> **phase-1a** event, whereas the shipped shadow/drive lives in **phase 1b** only.

**Corrections log (claims revised mid-analysis — do not re-trust the originals).** Each was settled by a
specific source/measurement, not opinion:
1. *"The failure is catastrophic cooling"* → **PdV expansion work**, not cooling. Settled by the `dEb/dt`
   budget decomposition (fig1): `L_cool/Lmech ≈ 0.01`; `PdV/Lmech` runs 0.52→1.56, crossing 1 at the `Eb` peak.
2. *"Snapshot 0's `Pb` is a fixed seed/placeholder"* → the **genuine Weaver IC**, computed
   `Pb = bubble_E2P(E0,r0,R1)` (`run_energy_phase.py:97-100`).
3. *"`Pb0` is bit-identical across the two clouds"* → **≈equal to ~6 sig figs** (`2.135768e7` vs `2.135766e7`);
   `Pb0 ∝ nCore` and both share `nCore=1e2`, so it is *near*-equal, not bit-identical.
4. *"The healthy run starts later (a delay)"* → a **plotting artifact** (`reliable_mask` trimmed the `v2`/`Eb`
   state too); both runs' snap 1 sits at the same elapsed `t−t0 ≈ 3e-5 Myr`. Fixed by plotting state at every snapshot.

**Status (2026-06-19, earlier):** 🟡 DIAGNOSED + REPRODUCED; scope DECIDED — minimal robustness fix only.
Two independent investigations + a sim-free probe + two live repros agree on the mechanism. Smoke of V3
(both guards) on `fail_repro`: it stops the divide-by-zero crash but the run then drives `Eb` through zero
into NEGATIVE (`+7.4e8 → −9.1e8 → −1.0e12`) and grinds (timed out 320 s; `data/smoke_V3_fail_repro_trajectory.csv`).

**DECISION (maintainer, 2026-06-19): fix the crash *maximally* without touching transition mechanics.**
Family **T** (the principled `Eb`-peak → momentum handoff) is **DEFERRED** — it requires the physics
investigation owned by `docs/dev/archive/transition/TRIGGER_PLAN.md` and is future work. This branch ships only the
**robustness fix**: **G (geometry guard so the divide can never blow up) + F (detect the non-physical
energy-driven state — non-finite `Pb`/`T` or `Eb`≤0 — and *terminate the run cleanly* with a recorded
`SimulationEndReason`, instead of crashing / emitting NaN / grinding forever).** It must **not** change
*when/how* the normal energy→momentum transition fires, and must be **bit-identical on healthy configs**.
Net effect for the sweep: the `5e9/n1e2` points produce a clean, flagged (collapsed) output instead of a
dead job — the cloud's later momentum-driven evolution is left for the deferred T work. **No production
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

**Relationship to the `docs/dev/transition/` workstream (`docs/dev/archive/transition/TRIGGER_PLAN.md`) — important, do not collide.**
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
3. This bug is the **energy-collapse extreme** of the same transition question (the bubble is
   momentum/PdV-dominated essentially from birth — the energy-driven solution never self-sustains (§3b),
   so the `Eb`-peak is immediate). The trigger study optimizes
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
a typical `1e6` cluster. It launches the shell at **~2000–3700 km/s** (near free-expansion), and the
bubble's **PdV work on that fast shell, `4πR²·Pb·v2`, exceeds `Lmech`** — so `Eb` is drained and collapses
instead of growing (local repro: `E0=6.4e9 → 4.8e8`, a 13× drop, in 0.003 Myr; the healthy `1e7` control's
`Eb` *grows* `5.7e5 → 2.3e7`). **⚠️ Mechanism correction (2026-06-19): radiative cooling is _not_ the
driver** — the measured `L_cool` is only **~1% of `Lmech`**; the energy sink is **PdV expansion work**
(§3b, with figures). "Catastrophic cooling" is a misnomer for this band; keep it only as the historical
label. The crash *mechanics* below (R1→R2, divide-by-zero) are unchanged by this correction.

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
- **Phase 1a** (`run_energy_phase.py:162` → `bubble_luminosity.get_bubbleproperties_pure` →
  `solve_R1` @ `:175`): the real Helix crash. The segment-0 ODE already produced `Eb=nan`; loop-1's
  bubble solve calls `solve_R1(nan)` → raise → **uncaught** → run dies.
- **Phase 1b** (`run_energy_implicit_phase.py:798` → `get_betadelta.compute_R1_Pb` → `bubble_E2P`):
  the local repro. The beta-delta solve fails to find a physical `dMdt>0` root (handled — "Holding last
  physical dMdt"), but `compute_R1_Pb` sits **outside** that guard and divides by zero → run dies.

### Why mass-dependent (the regime boundary)
`mCloud=1e7` (same `nCore`, `sfe`, `PISM`, `nISM`) runs healthy through 1a (95 segments) into 1b — `Eb`
grows. Only the high-cluster-mass band collapses `Eb`: a more massive cluster launches a **faster shell**,
so the **PdV work `4πR²·Pb·v2` exceeds `Lmech`** (`PdV/Lmech > 1`) and the energy-driven solution cannot
self-sustain (§3b). (Originally framed as `Lcool > Lmech`; the budget shows `Lcool/Lmech ≈ 0.01` — it is
the PdV term, not cooling, that crosses unity.) The matrix (§5) will pin the mass/density threshold.

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

## 3b. Energy budget — the collapse is PdV work, not cooling (2026-06-19) — with figures

Decomposing the energy ODE the code actually integrates (`phase1_energy/energy_phase_ODEs.py:280`),
`Ed = (Lmech − L_bubble) − (4π·R2²·press_bubble)·v2 − L_leak`, over the live `fail_repro` trajectory:

Self-consistent snapshots only (the leading free-streaming→Weaver IC-relaxation rows are **excluded** — the
per-snapshot `Pb·v2` proxy doesn't track `dEb/dt` there; see the data-integrity note below for why, and why
snap 0's identical `Pb` across clouds is the *real* IC, not a bug):

| t (×10⁻³ Myr) | Eb | Lmech (in) | **L_cool** (`bubble_LTotal`) | **PdV** = 4πR²·Pb·v2 | PdV/Lmech | v2 (km/s) |
|---|---|---|---|---|---|---|
| 1.41 | 6.20e9 | 1.01e13 | **1.35e11** (1.3%) | **5.25e12** | 0.52 | 723 |
| 1.53 | 6.47e9 | 1.01e13 | **1.39e11** | **1.01e13** | **0.99** (peak Eb) | 1380 |
| 1.74 | 5.87e9 | 1.01e13 | **1.26e11** | **1.48e13** | **1.46** | 2112 |
| 2.28 | 2.94e9 | 1.01e13 | **7.90e10** | **1.49e13** | **1.47** | 2438 |
| 2.82 | 7.40e8 | 1.01e13 | **4.26e10** (0.4%) | **1.32e13** | **1.30** | 2330 |

**The energy sink is the PdV expansion work, not radiative cooling.** `L_cool/Lmech ≈ 0.004–0.014` (~1%)
throughout. `PdV/Lmech` rises from **0.52 → 1.56**, crossing 1 at `t≈1.55e-3` — exactly where `Eb` stops
growing and starts collapsing (`Eb` peaks `6.47e9` at the crossing). So `dEb/dt` flips sign with the PdV
term, not cooling. The driver is the **shell velocity**:
this cluster launches the shell at ~2000–3700 km/s (near free-expansion, `R ≈ v·t`), and `PdV ∝ v2`.
The system is out of the self-similar Weaver equilibrium (where PdV is a fixed fraction of `Lmech` and `Eb`
grows) — physically it is **momentum/free-expansion-dominated from birth**.

**Same mechanism on the real Helix point (`fail_helix`, sfe0.05/PISM0):** `L_cool/Lmech ≈ 0.001–0.012`,
`PdV/Lmech` rises through 1 to ~1.44 — confirmed, not assumed, so the PdV finding holds across the band.

**Data-integrity note (2026-06-19, corrected) — snapshot 0 is the genuine IC, not a placeholder.**
An earlier draft of this note called snap 0 a "seed"; that was wrong. `run_energy_phase.py:97-100` *computes*
the initial `Pb = bubble_E2P(E0, r0, R1)` from the Weaver IC — it is real physics. It is **≈equal (to ~6 sig
figs — `2.135768e7` vs `2.135766e7`, NOT bit-identical) across the `5e9` and `1e6` clouds because they share
`nCore=1e2`** (they differ only in `mCloud`; the ~1e-6 residual is an `mCloud`-dependent correction):
- `Pb0 ∝ nCore` (ambient density). Derivation from `get_InitPhaseParam.py`: with `E0=(5/11)L_w·dt0`,
  `r0=v0·dt0`, `dt0²=3·Ṁ/(4π·ρ_a·v0³)`, `Ṁ=ṗ_w²/(2L_w)` ⇒ `Pb0 ∝ E0/r0³ ∝ L_w²·ρ_a/ṗ_w²`. `L_w,ṗ_w ∝ M_cluster`,
  so `L_w²/ṗ_w²` is mass-independent and `Pb0 ∝ ρ_a = nCore·μ`.
- `v0 = 2L_w/ṗ_w` (wind terminal velocity) — likewise mass-independent (both `∝ M_cluster`). So `v0=3739 pc/Myr`
  for *every* cloud. **Scientifically fine:** intensive IC quantities (`Pb0`, `v0`) are set by `nCore` + the SPS
  wind, not by `mCloud`; only extensive ones (`E0`, `r0`, mass) scale with the cluster. A different `nCore` gives
  a different `Pb0`.

**What the figures trim, and what they don't (corrected — `v2`/`Eb` are never trimmed).** Only the **`PdV` proxy**
has a reliability caveat: the snapshot stores **segment-START** `(R2,Pb,v2)` but the budget `Ed` needs the
**segment-AVERAGE**, so during the fast free-streaming→Weaver relaxation (first few steps) it mis-tracks `dEb/dt`
(`small_1e6` reads `PdV/Lmech>1` at snaps 2–4 while `Eb` is *actually growing* — this was the green "spike";
midpoint-averaging does **not** fix it). So in **panel A** the proxy is **solid where it reconstructs `dEb/dt`**
(`sign(Ed)==sign(forward-diff dEb/dt)`; `fail_repro` from snap 1, `small_1e6` from snap 5) and **dotted/faded through
the IC-relaxation**. `v2` and `Eb` are **stored STATE — plotted at every snapshot** (panels B/C), so both clouds
start together at `t−t0 ≈ 3e-5 Myr`. An earlier draft *trimmed* `small_1e6` to snap 5 in all panels, which made the
green look like it "started late" — that was a plotting artifact, now fixed. **All three figures use elapsed time
`t−t0`** (fig2 log; fig1/fig3 linear) so the same `fail_repro` curve sits at the same x everywhere. Conclusion
unchanged: self-consistent `PdV/Lmech` crosses 1 for the failing band (real max ≈1.56), stays ≤0.95 for healthy.

(NB the early `v2≈739 pc/Myr` is ~equal for both clouds to ~7 sig figs — `v0` is mass-independent and the first
segment is near-self-similar — then diverges from snap 2 as cloud-specific effects enter. Real, not an artifact.)

**Why the two clouds enter phase 1a at very different absolute `t0` (verified, not assumed).** `tSF=0` (logged) so
`t0 = dt_phase0`, the free-streaming duration `= √(3·Ṁ/(4π·ρ_a·v0³))` (`get_InitPhaseParam.py:151`). With `ρ_a`
(same `nCore=1e2`) and `v0=3739 pc/Myr` (same — wind terminal velocity `2L_w/ṗ_w`, mass-independent) equal, only
`Ṁ` differs, and `Ṁ ∝ M_cluster` (logged `Ṁ_wind`: 1.451 vs 2.901e-4 Msun/yr = ratio **5000** = `M_cluster` ratio).
So `dt_phase0 ∝ √M_cluster` and `t0` ratio = √5000 = **70.71** (logged `t0`: 1.383e-3 / 1.956e-5 = 70.71 ✓). The
5e8 M⊙ cluster free-streams ~70× longer before its energy phase begins, then collapses fast — hence the elapsed-time
axis for a fair comparison.

**Decomposition is faithful (validated):** the reconstructed `Ed = Lmech − L_cool − PdV − L_leak` matches a
finite-difference `dEb/dt` over the physical snapshots with **median ratio 1.00** (sign agreement 48/52).

**Healthy vs failing discriminator:** for the healthy `small_1e6`, `PdV/Lmech` stays **< 1** (≤0.95, declining;
Eb grows, classic Weaver) and `v2` decelerates to ~50 km/s; for `fail_repro`, `PdV/Lmech` crosses 1 (peak ≈1.56)
and `v2` stays ~2000+ km/s.

**Figures** (`figures/make_energy_budget_figs.py`, reproducible from the committed CSVs, no re-run needed):
- `figures/fig1_dEbdt_budget.png` — the budget: PdV ≫ L_cool, PdV crosses Lmech (the finding).
- `figures/fig2_healthy_vs_failing.png` — PdV/Lmech, v2, Eb vs **elapsed energy-phase time `t−t0` (log)** for failing vs healthy.
- `figures/fig3_bug_and_fix.png` — Eb→0 collapses R1→R2 (shell vol→0 → 1/0 → NaN); old crash vs new code-51 stop.
- `figures/fig4_energy_driven_discriminator.png` — were they ever energy-driven? (reservoir growth + PdV/Lmech, 5 configs).
- Data: `data/budget_fail_repro.csv`, `data/budget_small_1e6.csv`, `data/discriminator.csv`.

**Were the failing clouds ever genuinely energy-driven? (measured, 5 configs — `data/discriminator.csv`,
regenerated 2026-06-19 from a single stamped batch at commit `d919ff77`, window `t≤1.0 Myr`).**
The cleanest signal is the **reservoir growth** `Eb_peak/Eb_init` (a pure state variable, fully reliable): the
healthy controls build the hot-bubble thermal reservoir **≥×39,300 (`small_1e6`) / ≥×94,900 (`small_1e5`)** —
*lower bounds*, since both are still energy-driven (`Eb` monotonically growing, no peak) at the `t≤1.0 Myr` window
cap; all three failing configs build it **×1.014 (≈1%)**. (The earlier committed values ×13,600/×37,900 came from
**truncated** healthy runs stopped at `t≈0.32 Myr`; the clean regen with a fixed window is larger and the finding
is stronger — failing matches bit-identically.) So the energy-driven reservoir essentially **never forms** in
the failing band. The nuance: it is *not* "PdV>1 from birth" — every config (failing and healthy) starts at the
same self-similar handoff `PdV/Lmech ≈ 0.52–0.60`. The fork is the **direction**: healthy clouds decelerate →
`PdV/Lmech` falls and stays `<1` → the reservoir builds; failing clouds never decelerate → `PdV/Lmech` rises
through 1 within **~7–10%** of the phase and stays above for **~90%** → `Eb` collapses. **Interpretation:** the
failing clouds are "stillborn" energy-driven bubbles — they inherit the energy-driven IC but fail to establish
the self-similar deceleration that *defines* the phase. So a PdV-inclusive transition trigger (family T) would be
detecting "this bubble failed to become energy-driven," i.e. a **regime mismatch**, not a healthy Weaver→momentum
transition — which argues the deeper-correct fix is to recognise these as free-expansion/momentum-dominated
*earlier*, not to bolt a PdV term onto the cooling-based transition test.

This **revises** the "catastrophic cooling" label used in §1–§2 and the early commit messages, and it
*confirms* family **T**'s framing in §4: the principled handoff trigger is the **PdV-inclusive** net-energy
zero-crossing (`Lgain − Lloss − 4πR²·v2·Pb ≤ 0`), i.e. the PdV term is exactly the one that tips it.

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
The existing `cooling_balance` break (`run_energy_implicit_phase.py:1077-1078`) is *never reached* — the
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

## 7. Verdict (2026-06-19)
**Shipped: G (geometry guard) + F (graceful collapse termination).** Family T (momentum handoff) deferred
to `docs/dev/transition/` per the maintainer decision. The implementation:

1. **G — `bubble_E2P` volume floor** (`get_bubbleParams.py:226-235`): `if (r2³−r1³) <= 0: shell_volume =
   1e-13·r2³`. Stops the divide-by-zero at its source. **Bit-identical for every physical bubble** (the
   branch is dead while `vol > 0`) — pinned by `test_bubble_E2P_bit_identical_when_volume_positive`.
2. **F — collapse → clean stop** in **both** energy phases (`run_energy_phase.py` ~`:313`,
   `run_energy_implicit_phase.py` ~`:1006`): when `not isfinite(Eb) or Eb <= 0`, set `EndSimulationDirectly`
   + `SimulationEndReason` + `SimulationEndCode.ENERGY_COLLAPSED` (new, `(51,"energy_collapsed")`, the
   50–59 "inspection required" band) and `break`. `main.py` then skips 1b/1c/2 via the existing
   `EndSimulationDirectly` gate. **Does not touch the `cooling_balance` transition.**

**Gate results:** robustness ✅ (`fail_repro` completes cleanly, code 51); unit ✅ (5/5, incl. the
bit-identity pin); regression ✅ (`pytest -m "not stress"` 554 passed). **Healthy full-run no-op: ✅ confirmed
byte-identical** (see the top status block — this earlier "pending" item is now done). **Deferred (optional):**
the rest of the failing band/threshold sweep (S2 matrix), which by construction no-ops on healthy.

**Why this and not more:** the divide-by-zero is a real correctness bug; G+F make every affected run
*complete with a clear status* instead of crashing/NaN/grinding — the maximal fix that does **not** require
the transition-physics investigation. The cloud's momentum-driven continuation past the collapse is the
deferred T work; until then these points are cleanly flagged `energy_collapsed` for the sweep to filter.

## 8. Key references (re-verified against `main @ 6bdba8de`, 2026-06-19)
- Degeneracy math: `trinity/bubble_structure/get_bubbleParams.py` — `get_r1` `:375-402`, `solve_R1`
  `:405-429`, `bubble_E2P` `:198-230` (the `r2+=1e-10`-in-cm dud guard `:224`, the divide `:228`).
- Call sites (all currently unguarded for `R1→R2`): phase 1a `run_energy_phase.py:96,162,325`;
  energy ODE `phase1_energy/energy_phase_ODEs.py:223,358`; phase 1b
  `phase1b_energy_implicit/get_betadelta.py:297(compute_R1_Pb),327,329` + `run_energy_implicit_phase.py:798`;
  phase 1c `phase1c_transition/run_transition_phase.py:505,750,835`; bubble solve
  `bubble_structure/bubble_luminosity.py:175,181`.
- Existing energy→momentum transition (family T context): `run_energy_implicit_phase.py:1077-1078`
  (`cooling_balance`); the principled `Eb`-peak / net-energy event is owned by `docs/dev/archive/transition/TRIGGER_PLAN.md`.
- `BubbleSolverError` (family F target): `bubble_structure/bubble_luminosity.py:105(class),298,428(except),361,569,584(raise)`.
- The benign `Rejected. min T` branch: `bubble_luminosity.py:308-311`; `_T_INIT_BOUNDARY=3e4 :51`;
  boundary-transient note `:867,:918`.
- Latent secondary nan source (cooling-cube holes, high-phi/low-n): `cooling/non_CIE/read_cloudy.py:95-97,133`
  (RegularGridInterpolator, default `bounds_error=True`; NaN query → silent NaN). Not the primary trigger
  for the standard high-Pb large cloud, but covered by family F.
- Repro configs: `harness/params/fail_repro.param` (sfe0.1/PISM1e4), `harness/params/fail_helix.param`
  (the real Helix sfe0.05/PISM0 point), healthy controls `harness/params/small_1e{5,6,7}.param`.
- Sibling workstream: `docs/dev/archive/transition/TRIGGER_PLAN.md` (+`P0.md`,`pshadow-design.md`) — the normal
  implicit→momentum trigger study (families F0–F5); align family T with its `Eb`-peak event, don't collide.


## 9. Go-forward options (decision pending — 2026-06-19)

The bug fix (`G+F`) is shipped and proven; the diagnosis is complete; `data/discriminator.csv` shows the
failing band are **stillborn energy-driven bubbles** (reservoir ×1.01 vs healthy ×10⁴). What remains is **not an
engineering bug** — it is a physics-modelling decision: *what should these edge clouds do after the energy phase
fails to establish?* Three paths, in detail, so the call can be made on evidence.

### Path 1 — Ship & defer  *(recommended)*
- **What:** open a PR for `bugfix/failed-large-clouds` → main. `G+F` + the diagnostic suite (4 figs, CSVs,
  `insights.html`) land. `ENERGY_COLLAPSED` (code 51) is the documented behaviour for this band.
  *(No longer "permanent": as of 2026-07-01 a finite 1b `Eb<=0` collapse routes to momentum instead —
  see the SUPERSEDED note at the top. `ENERGY_COLLAPSED` now means non-finite `Eb` or a phase-1a collapse.)*
  Paths 2/3 stay pre-scoped, pulled off the shelf only on a concrete need.
- **Remaining work:** a PR description; nothing else.
- **Cost:** ~zero (done).  **Risk:** none technical; the only limitation is scientific (flagged, not continued)
  and it is documented + cheap to lift later.
- **Gates:** already met — robustness ✅, healthy byte-identical ✅, `pytest` 555 ✅.
- **Right when:** no current science needs faithful post-collapse trajectories for the 5e9/sfe0.1 + Helix band.
  Given these are extreme-edge clouds, this is very plausibly the correct *permanent* answer, not just interim.

### Path 2 — `T` probe (gated experiment)
- **What:** add the PdV-inclusive net-energy zero-crossing event (the `Eb`-peak:
  `Lgain − Lloss − 4πR²·v2·Pb ≤ 0`) to phases 1a/1b; at the peak hand off to phase 1c → 2 (momentum) instead of
  force-stopping. Run `fail_repro` + `fail_helix` and **measure** whether 1c→2 yield a physical momentum
  trajectory (R2 grows, v2 decays sensibly, no NaN/grind) or immediately hit 1c's `ENERGY_FLOOR` / degenerate.
- **How:** behind the existing variants harness first — an experiment, **not** a production commit. Persist the
  trajectory CSV; compare against the free-expansion expectation.
- **Cost:** medium (event hook ~10–20 lines; the validation is the work).
- **Risk:** the hand-off state — even at the "peak" the reservoir only grew 1%, so 1c's energy-balance ODE may
  not accept it. A degenerate result is itself informative: it is the evidence that Path 3 is required.
- **Gates (rule 5):** (a) no production change until the probe yields a sane trajectory; (b) if productionised,
  the new event must **not** perturb healthy clouds — they must stay byte-identical (prove the `Eb`-peak event
  never fires before `cooling_balance` on the ×10⁴ runs). This is the make-or-break gate.
- **Right when:** you want to know whether continuation is *feasible* at all, cheaply, before committing. Good
  de-risking; its outcome decides between "T suffices" and "need Path 3."

### Path 3 — Early regime routing
- **What:** detect the free-expansion/momentum regime *early* (sustained `PdV/Lmech > 1`, or shell velocity near
  the wind terminal velocity, or reservoir-not-building over N steps) and route these clouds to the momentum
  model from birth, bypassing the energy phase's self-similar assumptions. The physically-correct treatment per
  the discriminator (they were never energy-driven).
- **Cost:** high — a new regime branch + IC + validation; a physics workstream, not a patch.
- **Risk:** largest surface. Must prove (a) healthy clouds untouched (byte-identical), (b) the early-routing
  criterion separates the bands with no false positives, (c) the momentum trajectories are physical (ideally vs
  an analytic free-expansion / momentum-driven shell solution).
- **Gates:** full rule-5 ladder (gate → baseline → equivalence → full-run on stiff edges → persist).
- **Right when:** the science genuinely needs faithful trajectories for this regime (e.g. the band matters for a
  paper/sweep), justifying the investment.

### Dependency & recommendation
Paths 2 and 3 are both gated on a **science need only the maintainer can assess** — is faithful continuation of
these extreme clouds actually required, or is flag-and-filter scientifically adequate? If continuation is wanted,
the rational order is **Path 2 first (cheap probe) → Path 3 only if Path 2 degenerates** — never start Path 3
blind. **Recommendation: Path 1 now** (ship the correct fix; do not build continuation speculatively — rule 1 /
YAGNI, doubly so when the data shows there is no energy-driven phase to "continue" from), with Path 2 defined and
ready for when/if a sweep needs these cells.
