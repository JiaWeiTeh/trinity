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

<!-- TABLE: filled from h4_eval.csv via `python h4_analyze.py` -->
_(results table inserted below once the matrix completes)_

## 4. Trajectory evidence — does it self-sustain or re-collapse?

<!-- filled from traj CSVs + figures -->

## 5. No-op confirmation — control (healthy/stall) configs

Where PdV never exceeds the cap (`PdV < Lmech < κ·Lmech`… in fact `PdV<Lmech` so `excess=0`), the
`PDVCAP` variant is designed to be identical to V0. Matched-snapshot diff via
`python h4_analyze.py --noop`:

<!-- filled -->

## 6. Verdict

<!-- filled -->

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
