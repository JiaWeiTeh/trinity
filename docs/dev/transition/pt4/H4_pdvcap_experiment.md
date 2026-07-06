# H4 experiment: a transient PdV-drain cap — survivable early transient, or stillborn?

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

**Date:** 2026-06-22. Branch `fix/transition-trigger-problem-pt4`. DIAGNOSTIC experiment;
production untouched (monkeypatch-only). Line refs verified against current source on this date.

---

## 0. Hypothesis + the energy-injection caveat (READ FIRST)

**Maintainer's idea (H4):** "allow the bubble to expand but CAP the PdV drain at a value for the
first ~1e-3 Myr (and other timescales), so the energy-driven bubble can establish itself." During
`t < t_window`, replace the PdV expansion-work drain in `dEb/dt` with `min(PdV, κ·Lmech)` (κ<1),
while LEAVING the shell-acceleration/drive term intact so the shell still expands. After `t_window`,
no cap (production behavior). The question this experiment answers with data: **does the bubble
survive the capped window and then SELF-SUSTAIN (Eb keeps growing after the cap lifts), or
re-collapse the moment the cap is released (because PdV/Lmech > 1 persists)?** This discriminates a
*survivable early transient* from a cloud that is *genuinely momentum-driven from birth*
("stillborn", per the failed-large-clouds discriminator, `PLAN.md` §3b — treat as unverified).

> **ENERGY-INJECTION CAVEAT.** The PdV term `4π·R²·Pb·v2` is **REAL PHYSICAL expansion work** — for
> these clouds radiative cooling `L_bubble` is only ~1% of `Lmech` (verified §3), so the bubble is
> drained by *doing work on the shell*, not by radiating. **Capping PdV therefore UNDER-DRAINS the
> bubble / INJECTS ENERGY and VIOLATES energy conservation while the cap is active.** This is a
> **DIAGNOSTIC to understand the regime, NOT a production-fix candidate.** A cap that "saves" the
> run does so by manufacturing energy; it cannot ship. (Same caveat class as the H3 Eb-floor sibling,
> `H3_eb_floor_experiment.md` §0 — H3 backstops the symptom, H4 throttles the cause.)

## 1. The collapse picture, verified on current source

- Phase 1a energy ODE RHS `energy_phase_ODEs.get_ODE_Edot_pure` returns `[rd, vd, Ed]`:
  - `Ed = (Lmech_total − L_bubble) − (4π·R²·press_bubble)·v2 − L_leak`  (`energy_phase_ODEs.py:280`).
    The middle term is the PdV drain; `L_bubble` is `snapshot.bubble_LTotal` (cooling), `L_leak` the
    cover-fraction leak (`:277-279`).
  - The **shell acceleration** `vd = (4π·R²·(P_drive − P_ext) − …)/mShell` (`:265-266`) uses
    `P_drive = max(press_bubble, P_HII)` (`:258`), a **separate quantity from the PdV drain term**.
    Capping PdV in `Ed` therefore does NOT touch `vd` — the shell keeps expanding.
- Phase 1b implicit RHS `run_energy_implicit_phase.get_ODE_implicit_pure` returns
  `[rd, vd, Ed_from_beta, Td_from_delta]` (`:532`). Its `rd, vd` come from `get_ODE_Edot_pure`
  (`:526-529`, the same shell drive). The integrated 1b energy derivative is `Ed_from_beta`
  (Rahner-A12 cooling form, `get_betadelta.cool_beta_to_Ebdot_pure`, computed OUTSIDE the ODE at
  `run_energy_implicit_phase.py:854` and passed in at `:929`). **PdV does not appear as a separable
  term in `Ed_from_beta`**; it enters the 1b physics through the energy balance the β-solver matches:
  `Edot_from_balance = L_gain − L_loss − 4π·R²·v2·Pb`  (`get_betadelta.py:434`), with the residual
  `(Ed_from_beta − Edot_from_balance)/Ed_from_beta` driven to ~0 (`:438`).
- The clean stops these collapses hit (code **51 = ENERGY_COLLAPSED**, inspection band 50-59):
  - 1b / `fail_repro`: state check `if not np.isfinite(Eb) or Eb <= 0` (`run_energy_implicit_phase.py:1007`).
  - 1a / `fail_helix`: the **bubble-structure solve degenerates** as Eb→0 (`solve_R1` cannot bracket,
    cooling table OOB) → caught at `run_energy_phase.py:169-177` ("bubble solve degenerate as Eb -> 0").
  - (the plain `Eb<=0` 1a state check is `run_energy_phase.py:340`.)

`fail_repro` collapses in **1b**, `fail_helix` in **1a** — so the cap must work in BOTH phases.

## 2. Cap design, injection point, and proof the DRIVE is untouched

Harness (all under `docs/dev/transition/pt4/`, h4_ prefix; production NOT edited):
`h4_variants.py` (monkeypatch), `h4_run_variant.py` (one-sim/process driver),
`h4_run_matrix.sh` (matrix, ≤3 concurrent), `h4_analyze.py` (tabulate + no-op diff),
`h4_figures.py` (figures). Pattern copied from the H3 sibling `h3_variants.py`.

**Injection point — "add back the capped excess."** A naive `get_effective_bubble_pressure` patch
would throttle `press_bubble`, which the shell drive `P_drive` (`:258,265`) ALSO reads — that would
stop the bubble expanding, which is forbidden. Instead, `PDVCAP` wraps the **ODE RHS** and modifies
**only the energy derivative**:

```
PdV    = 4π·R² · press_bubble · v2        # same Pb,R2,v2 the original RHS used
cap    = κ · Lmech_total
excess = max(0, PdV − cap)                # > 0 only when the drain exceeds the cap
Ed_capped = Ed_orig + excess              # == replacing PdV with min(PdV, cap) in the drain
```

- **Phase 1a** (`_edot_pdvcap`): delegate the whole RHS to the original `get_ODE_Edot_pure` to get
  `[rd, vd, Ed]`; recompute `(press_bubble, Lmech)` with the SAME production helpers the original
  uses (`get_current_sps_feedback` → `solve_R1` → `get_effective_bubble_pressure`, mirroring
  `:195,223,226-231`), form `excess`, return `[rd, vd, Ed + excess]`. **`rd (=v2)` and `vd` are the
  original values, passed through byte-for-byte — the shell drive is never touched.** Patched via the
  module attribute `energy_phase_ODEs.get_ODE_Edot_pure` (call site `run_energy_phase.py:272`).
- **Phase 1b** (`_implicit_pdvcap`): wrap `get_ODE_implicit_pure`; add the same `excess` to the
  incoming `Ed_from_beta`, then delegate to the original. The original computes `rd, vd` from the
  *bare-name* `get_ODE_Edot_pure` in `run_energy_implicit_phase`'s namespace (imported at
  `:73-75`), which the 1a patch does NOT rebind — so the 1b shell drive is also untouched.

**No-op guarantee.** Because `excess = 0` whenever `PdV ≤ cap` OR `t ≥ t_window`, the variant is
**bit-identical to V0** on every step where the cap does not bite. The recomputed `press_bubble`
uses the production `bubble_E2P` (whose shipped shell-volume floor `1e-13·r2³`,
`get_bubbleParams.py:229-235`, keeps the divide finite as Eb→0), so no extra geometry guard is
needed. Telemetry (`ACT`) records how many RHS evals the cap bit (1a/1b), `max PdV/Lmech` in/after
the window, and a cumulative-injected-rate proxy.

Wiring self-test (patch installs on PDVCAP, V0 restores the originals, params plumb through):
re-run with `python -c` against `h4_variants.apply(...)` — passed 2026-06-22.

## 3. Results — full sweep matrix

Run: `bash docs/dev/transition/pt4/h4_run_matrix.sh` (OMP_NUM_THREADS=1, one sim/process, ≤3
concurrent, each cell `timeout`-bounded + small `--stop_t`). Sweep on the collapse configs
(`fail_repro` [1b], `fail_helix` [1a], `mass_1e9` [1b/grind]): `t_window ∈ {1e-3, 3e-3, 1e-2, 1e-1}`
Myr at `κ=0.9`, plus a V0 baseline per config. Control/no-op configs (`small_1e6`, `simple_cluster`,
`pl2_steep`): V0 + `PDVCAP(t_window=1e-2)` — the cap must NEVER activate (PdV<Lmech), so they must be
byte-identical to V0. Per-cell rows in `h4_eval.csv`; trajectories in `traj/h4_traj_<tag>.csv`;
per-cell logs in `h4_logs/`.

Full matrix (21 cells, regenerate the table with `python h4_analyze.py`; `code` blank = run did not
reach a clean end (truncated by `timeout`/`stop_t`); `code 51` = ENERGY_COLLAPSED; `code 1` =
STOPPING_TIME). `rt_s=480` ⇒ the cell ran to the wall-clock timeout (a survivor/grind that kept
integrating), not a hang.

| config | variant | t_win | code | phase | final_t | R2 | v2 | final_Eb | min_Eb | cap | PdV/L_in | PdV/L_after | survived | selfsust | rt_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fail_helix | V0 | 0.001 | 51 | 1a | 0.00255 | 7.03 | 2e+03 | 0.00509 | 0.00509 | False | 0 | 0 | True | False | 60 |
| fail_helix | PDVCAP | 0.001 | 51 | 1a | 0.00264 | 7.21 | 1.96e+03 | 0.00508 | 0.00508 | True | 2.36 | 1.62e+102 | True | False | 69.6 |
| fail_helix | PDVCAP | 0.003 | - | 1b | 0.0103 | 17.1 | 942 | 9.81e+09 | 2.25e+09 | True | 2.36 | 1.21 | True | True | 480 |
| fail_helix | PDVCAP | 0.01 | - | 1b | 0.0133 | 19.8 | 824 | 1.55e+10 | 2.25e+09 | True | 2.36 | 0.668 | True | True | 480 |
| fail_helix | PDVCAP | 0.1 | - | 1b | 0.0133 | 19.8 | 824 | 1.55e+10 | 2.25e+09 | True | 2.36 | 0 | False | - | 480 |
| fail_repro | V0 | 0.001 | 51 | 1b | 0.00341 | 9.73 | 2.16e+03 | -9.14e+08 | -9.14e+08 | False | 0 | 0 | False | False | 81.4 |
| fail_repro | PDVCAP | 0.001 | 51 | 1b | 0.00341 | 9.73 | 2.16e+03 | -9.14e+08 | -9.14e+08 | False | 0 | 2.65 | False | False | 82.9 |
| fail_repro | PDVCAP | 0.003 | 51 | 1b | 0.00515 | 13.2 | 1.68e+03 | -2.82e+07 | -2.82e+07 | True | 2.65 | 1.57 | False | False | 177 |
| fail_repro | PDVCAP | 0.01 | - | 1b | 0.0122 | 22.2 | 1.02e+03 | 2.18e+10 | 6.38e+09 | True | 2.65 | 0.738 | True | True | 480 |
| fail_repro | PDVCAP | 0.1 | - | 1b | 0.0132 | 23.2 | 982 | 2.48e+10 | 6.38e+09 | True | 2.65 | 0 | False | - | 480 |
| mass_1e9 | V0 | 0.001 | - | 1b | 0.03 | 24.8 | 457 | 2.13e+10 | 2.15e+08 | False | 0 | 0 | True | True | 480 |
| mass_1e9 | PDVCAP | 0.001 | - | 1b | 0.03 | 24.8 | 457 | 2.13e+10 | 2.55e+08 | True | 1.65 | 1.66 | True | True | 480 |
| mass_1e9 | PDVCAP | 0.003 | - | 1b | 0.0236 | 21.9 | 509 | 1.61e+10 | 5.70e+08 | True | 1.65 | 0.914 | True | True | 480 |
| mass_1e9 | PDVCAP | 0.01 | - | 1b | 0.0292 | 24.6 | 462 | 2.07e+10 | 5.70e+08 | True | 1.65 | 0.618 | True | True | 480 |
| mass_1e9 | PDVCAP | 0.1 | 1 | 1b | 0.03 | 24.9 | 456 | 2.15e+10 | 5.70e+08 | True | 1.65 | 0 | False | - | 475 |
| pl2_steep | V0 | 0.001 | 1 | 1b | 0.005 | 0.455 | 48.7 | 3.75e+06 | 570 | False | 0 | 0 | True | True | 190 |
| pl2_steep | PDVCAP | 0.01 | 1 | 1b | 0.005 | 0.455 | 48.7 | 3.75e+06 | 570 | True | 0.909 | 0 | False | - | 186 |
| simple_cluster | V0 | 0.001 | 1 | 1b | 0.005 | 0.371 | 37.2 | 1.15e+06 | 93.7 | False | 0 | 0 | True | True | 185 |
| simple_cluster | PDVCAP | 0.01 | 1 | 1b | 0.005 | 0.371 | 37.2 | 1.15e+06 | 93.7 | True | 0.909 | 0 | False | - | 179 |
| small_1e6 | V0 | 0.001 | 1 | 1b | 0.005 | 1.88 | 219 | 4.27e+06 | 1.80e+04 | False | 0 | 0 | True | True | 134 |
| small_1e6 | PDVCAP | 0.01 | 1 | 1b | 0.005 | 1.88 | 219 | 4.27e+06 | 1.80e+04 | True | 0.909 | 0 | False | - | 152 |

**How to read it (flag caveats — flags are heuristics, read them with `final_Eb` / `PdV/L_after`):**
- `survived` = an accepted snapshot exists with `t ≥ t_window` and `Eb > 0`; `selfsust` = `Eb` rose
  between the last two such snapshots. Both are **only meaningful when the run actually integrated
  past `t_window`.**
- **`t_win=0.1` rows always show `survived=False, selfsust=-`** because the run truncates at
  `t≈0.013–0.03 < 0.1`, so *no* snapshot reaches `t_window`. These are the **"cap never released within
  the run" control** — the high, growing `final_Eb` (1.5–2.5e10) is the real read, not the flag.
- **`fail_helix tw=1e-3` shows `survived=True` but `final_Eb≈0.005`** — a *false positive*: the 1a
  bubble-solve degenerated (`PdV/L_after=1.6e102` is the Eb→0 artifact) and the run hit the same
  code-51 stop as V0, just ~1e-4 Myr later. Trust `final_Eb≈0` + `selfsust=False`, not the boolean.
- The control PDVCAP rows (`tw=1e-2`, `stop_t=0.005`) also show `survived=False` for the same
  truncation reason — irrelevant; §5 confirms they are no-ops via the matched-snapshot diff.

## 4. Trajectory evidence — does it self-sustain or re-collapse?

The answer is **window-dependent**, and the two collapse configs differ in *when* PdV crosses 1.
Figures (`figures/h4_*`): `h4_Eb_sweep_<cfg>` (Eb(t), V0 + each window, dotted = cap release),
`h4_pdvratio_sweep_<cfg>` (PdV/Lmech(t), =1 line marked), `h4_summary` (survived/self-sustained vs
t_window). Per-snapshot data in `traj/h4_traj_<tag>.csv`.

**fail_helix (1a, sfe=0.05, mCloud=5e9).** PdV/Lmech is >1 from the phase start (~1.66, declining).
- `tw=1e-3`: too short — the cap lifts at t=1e-3 while PdV/L is still ~2.4; the bubble solve
  degenerates as Eb→0 and the run hits the same code-51 1a stop as V0, only ~1e-4 Myr later
  (final_t 0.00255 → 0.00264, final_Eb ≈ 0.005 both). The `survived_past_window=True` flag here is a
  **false positive** — it only means an accepted snapshot exists past t_window with a *barely*-positive
  degenerate Eb (~0.005); the real signal is `self_sustained=False` and final_Eb≈0. (Read the boolean
  alongside final_Eb, never alone.)
- `tw=3e-3`: **survives.** Eb grows under the cap (2.25e9 → 3.06e9 → 3.21e9), the run crosses into 1b,
  and at release Eb dips (3.21e9 → 2.73e9 → 2.51e9) but PdV/Lmech then falls **through 1**
  (1.09 → 1.00 → 0.94 → 0.88) and **Eb recovers** (2.51e9 → 2.58e9 → 2.71e9, still growing at
  truncation). A genuine *survivable transient*: the cap bought enough time for PdV/Lmech to drop
  below 1, after which the bubble self-sustains on its own.

**fail_repro (1b, sfe=0.1, mCloud=5e9).** PdV/Lmech first exceeds 1 at t~0.0015 Myr (the bubble grows
to ~6.45e9 first, then the drain takes over) and stays >1 until t~0.0045.
- `tw=1e-3`: **the cap never bites** (PdV<Lmech for the whole `t<1e-3` window; activation counters 0,
  `max_pdv_ratio_in_window=0`). The run is **byte-identical to V0** and collapses through zero at
  t=0.0034 (code 51). The maintainer's exact "~1e-3 Myr" is *too early* for this config.
- `tw=3e-3`: cap lifts while PdV/Lmech is still ~1.36. Eb is inflated to 7.9e9 under the cap, dips
  hard at release (7.9e9 → 5.4e9 → 2.4e9 → 6e8), bottoms near zero with a feeble recovery flicker
  (2.6e7 → 5.3e7), then **re-collapses through zero** at t=0.00515 (code 51). PdV/Lmech only crosses
  below 1 at t~0.0046, by which point Eb is already drained — *too little, too late.*
- `tw=1e-2`: cap lifts at t=0.01, by which point a full 1e-2 Myr of injection has built Eb to ~2.2e10
  and PdV/Lmech has fallen to **0.74 (<1)**. Post-release Eb **keeps growing** (15.7e9 → 17.7e9 →
  19.5e9 → 21.8e9): **self-sustains.** But note this is achieved on an Eb reservoir *manufactured* by
  10× the maintainer's window of non-physical energy injection.
- `tw=1e-1`: cap never lifts within `stop_t=0.03` (the whole run is capped); Eb grows monotonically.
  This is the trivial "cap-never-released" control, not a self-sustain test.

**mass_1e9 (1b, mCloud=1e9 — the lighter "collapse" config).** Important caveat: in this matrix
`mass_1e9` **does NOT collapse even at V0** — it reaches the full `stop_t=0.03` (or grinds to it) with
`Eb` *growing* from 5.7e8 to ~2.1e10 (`self_sustained=True` at V0). So mass_1e9 is **not in the
over-drained collapse regime** at all (its `PdV/Lmech` is modest, ~1.65 briefly then <1), and the cap
neither helps nor hurts the qualitative outcome (all windows survive; `tw=1e-1` even closes cleanly at
code 1). It is a useful negative control: a 5× lighter cloud at the same density is *already*
self-sustaining, confirming the collapse is specific to the heaviest (5e9) clouds — consistent with
the failed-large-clouds mass-threshold picture (`PLAN.md`, unverified).

## 5. No-op confirmation — control (healthy/stall) configs

The cap threshold is `κ·Lmech` with κ=0.9, so the strict no-op condition is **`PdV < 0.9·Lmech`**
(not `PdV < Lmech`). All three control configs reach a clean STOPPING_TIME (code 1) under both V0 and
PDVCAP, and their accepted trajectories match V0 to the solver-tolerance level. Matched-snapshot diff
via `python h4_analyze.py --noop` (every common (t,R2,Eb) row, V0 vs PDVCAP tw=1e-2):

| config | V0 code | PDVCAP code | cap_act | max\|ΔR2\| [pc] | max rel\|ΔEb\| | no-op? |
|---|---|---|---|---|---|---|
| small_1e6 | 1 | 1 | True | 2.7e-07 | 2.6e-05 | track-identical (fp) |
| simple_cluster | 1 | 1 | True | 4.9e-10 | 3.0e-08 | track-identical (fp) |
| pl2_steep | 1 | 1 | True | 1.5e-10 | 1.9e-08 | track-identical (fp) |

**Key nuance — `cap_act=True` does NOT mean the physics changed.** Two effects, both benign:
(1) the activation counter tallies *every* RHS eval where `PdV > κ·Lmech`, **including `solve_ivp`'s
rejected trial steps**; (2) with **κ=0.9**, the cap can graze a *healthy* config whose `PdV` rises
into the band `0.9·Lmech < PdV < Lmech` (the control PDVCAP rows all show `PdV/L_in≈0.909`, i.e. they
just touched `0.9·Lmech`). On these healthy runs that graze perturbs the accepted track only at the
**solver-tolerance level** — `simple_cluster`/`pl2_steep` differ at ~1e-8 (a single trial-step
probe); `small_1e6` at ~2.6e-5 (one accepted step grazed the band). Both are negligible vs the physics
(`rtol~1e-6..1e-4`) — **a genuine no-op to ≥4 significant figures.** The cap never engages where the
bubble does not over-drain, exactly the "test on all configs" payoff. (Were κ=1.0 the controls would
be bit-identical; κ=0.9 is the experiment's chosen drain ceiling, and the graze is its honest cost.)

## 6. Verdict

**These massive clouds are NOT instantaneously "stillborn", but they are also not saved by the
maintainer's specific prescription.** The PdV drain stays super-critical (PdV/Lmech > 1) for an
extended epoch (~1.5–4.5e-3 Myr from the phase start), and the outcome of a transient cap depends
entirely on whether the window outlasts that epoch:

- The maintainer's exact **~1e-3 Myr** window is **too short for both configs** — for fail_repro the
  cap doesn't even engage (PdV<Lmech that early); for fail_helix it only delays the 1a collapse by
  ~1e-4 Myr. So as literally proposed, the idea does not establish the bubble.
- A **moderately longer window** (3e-3) **splits the two configs**: fail_helix survives and
  self-sustains (PdV/Lmech crosses below 1 shortly after release); fail_repro re-collapses (PdV/Lmech
  is still >1 at release and the dipped Eb cannot recover in time).
- A **long window** (1e-2) lets even fail_repro self-sustain — but only by injecting ~10× the
  proposed window of non-physical energy, building an Eb reservoir an order of magnitude above its
  natural value, so that PdV/Lmech has dropped below 1 by release.
- **mass_1e9 (1e9 M⊙) is a negative control**, not a collapse: it self-sustains at V0, so the cap is
  immaterial there. The over-drained collapse is specific to the heaviest (5e9 M⊙) clouds.

**Interpretation:** the collapse is a *survivable early transient in principle* — there exists a
PdV-cap window after which the bubble self-sustains — but the required window is **config-dependent
and longer than the natural PdV>1 epoch the real (uncapped) physics imposes.** Under the true
physics the bubble drains before PdV/Lmech falls below 1, so it collapses; the cap "rescues" it only
by manufacturing the energy reservoir that the real expansion work would have removed. This is
**consistent with the failed-large-clouds picture** (`PLAN.md` §3b, unverified): for these clouds the
energy-driven phase is over-drained by *real* PdV expansion work (Lcool is ~1% of Lmech), so the
honest fate is a transition to a momentum-driven continuation — **not** an energy-conservation-
violating cap. H4 confirms the cause (PdV super-criticality over an extended epoch) that H3's Eb-floor
only backstopped as a symptom.

**This is a DIAGNOSTIC result, NOT a production fix.** Every "survives"/"self-sustains" outcome above
is purchased with injected, non-conserved energy (the capped excess `PdV − κ·Lmech`, integrated over
the window). It cannot ship; it tells us the *regime* (extended PdV super-criticality), which is the
input a real momentum-driven-continuation fix needs.

## 7. Artifacts (all committed under `docs/dev/transition/pt4/`)
- `h4_variants.py` — PDVCAP monkeypatch (add-back-the-capped-excess; 1a + 1b) + V0.
- `h4_run_variant.py` — one-sim/process driver; reads `dictionary.jsonl` for trajectory + PdV/Lmech
  per snapshot + cap-activation telemetry + survived/self-sustained flags.
- `h4_run_matrix.sh` — full (config, variant, t_window) matrix, ≤3 concurrent, per-cell timeout+stop_t.
- `h4_analyze.py` — tabulates `h4_eval.csv` + the control no-op diff.
- `h4_figures.py` — Eb(t) sweep, PdV/Lmech(t) sweep, survived/self-sustained summary.
- `h4_eval.csv` — one row per cell: outcome, end_code, final state, cap activation, PdV/Lmech in/after
  window, survived_past_window, self_sustained, runtime.
- `traj/h4_traj_*.csv` — per-snapshot trajectories (t, R2, v2, Eb, Pb, R1, T0, Lmech_total, PdV/Lmech).
- `figures/h4_*.{pdf,png}` — the plotted findings.
- `h4_run_matrix.log`, `h4_logs/*.log` — driver logs (exact per-cell commands + outcomes).
