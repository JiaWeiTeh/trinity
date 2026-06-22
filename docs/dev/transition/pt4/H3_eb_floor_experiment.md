# H3 experiment: is "Eb going non-positive" the SOLE failure mode? (Eb-floor diagnostic)

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

## 0. The energy-injection caveat (READ FIRST)

The `EBFLOOR` variant **FLOORS the bubble thermal energy `Eb` to a small positive value so it
can never collapse through zero.** This **INJECTS ENERGY and VIOLATES energy conservation.** It
is a **DIAGNOSTIC experiment, NOT a production fix candidate.** Its sole purpose: isolate whether
"Eb going non-positive" is the **SOLE** failure mode for the massive-cloud collapse regime — i.e.
whether the rest of the physics stays intact (shell keeps expanding, v2 physical, solver healthy)
once `Eb` is forced `> 0`. A floor that "fixes" the run does so by manufacturing energy; it cannot
ship.

## 1. Hypothesis (H3, the maintainer's)

Massive clouds (`mCloud=5e9`) drain `Eb` via PdV expansion work (`4πR²·Pb·v2 > Lmech`), so `Eb`
collapses through zero and the run hits the shipped `ENERGY_COLLAPSED` clean stop
(`docs/dev/failed-large-clouds/PLAN.md`, treat as unverified; verified §3 below). H3 asks: **if we
artificially floor `Eb > 0`, does the bubble keep expanding and the run proceed cleanly — does the
failing cloud then behave like a healthy one** (the "healthy vs failing" diagnostic,
storyline3 §1.4 / `failed-large-clouds/figures/fig2`: healthy ⇒ `Eb` grows, `PdV/Lmech<1`;
failing ⇒ `Eb` collapses)?

## 2. Variant design, injection point, and what it bypasses

Harness: `h3_variants.py` (monkeypatch), `h3_run_variant.py` (one sim/process driver),
`h3_run_matrix.sh` (matrix), `h3_salvage_timeout.py` (recover partial jsonl on a grind/timeout),
`h3_analyze.py` (tabulate). Copied from `docs/dev/failed-large-clouds/harness/{variants,run_variant}.py`;
V0/V1/V2/V3 carried over verbatim, `EBFLOOR` is the new piece. Originals NOT edited.

The shipped collapse picture, verified on current source:
- `Eb` is consumed by `get_bubbleParams.bubble_E2P(Eb,R2,R1)` (→Pb,
  `get_bubbleParams.py:198-238`) and `solve_R1(R2,Eb,…)` (→R1, `:413-446`). As `Eb→0` the wind
  shock `R1→R2`, `(R2³−R1³)→0`, and the divide blows up; the shipped geometry guard floors
  `shell_volume` at `1e-13·r2³` (`:229-235`) and `solve_R1` returns `0.0` for `R2<=0` (`:433`),
  so the divide stays finite.
- The shipped **`ENERGY_COLLAPSED` early-stops read the ODE STATE `Eb`** extracted from
  `solution.y[:,-1]`: **`run_energy_phase.py:340`** (`if not np.isfinite(Eb) or Eb <= 0`) and
  **`run_energy_implicit_phase.py:1007`** (same). Code **51 = `ENERGY_COLLAPSED`**, in the
  **inspection band 50-59** (`simulation_end.py:90`, `is_inspection_required` 50-59) — a clean
  termination, not a crash.

`EBFLOOR` therefore needs TWO coupled pieces (both monkeypatch module attributes):

- **(A) Keep the DRIVE positive.** Clamp `Eb = max(Eb, floor)` at entry to `gbp.bubble_E2P`
  (→ `Pb>0`) and `gbp.solve_R1` (→ `R1` well-defined). Reached everywhere via the module attribute
  (energy RHS `energy_phase_ODEs.py:223,226`, the bubble solve, diagnostics). Includes the V3
  geometry guards internally so the divide can't blow up.
- **(B) Keep the STATE positive — the new piece.** A **reflecting floor on `dEb/dt`**: whenever the
  state `Eb <= floor`, the energy-derivative component returned to `solve_ivp` is clamped `>= 0`, so
  the integrated state can only hold flat or grow at the floor and never crosses below it. Wraps the
  two RHS functions each phase's `solve_ivp` reads:
  - phase 1a: `energy_phase_ODEs.get_ODE_Edot_pure` (module-attr call at `run_energy_phase.py:272`);
  - phase 1b: `run_energy_implicit_phase.get_ODE_implicit_pure` (bare-name call at
    `run_energy_implicit_phase.py:929`; its 3rd return is `Ed_from_beta`, the betadelta cooling
    derivative actually integrated — clamping that controls the 1b state).

  Verified bindings (`h3_variants.py` patches both module namespaces): 1a uses the patchable module
  attribute `energy_phase_ODEs.get_ODE_Edot_pure`; 1b's `get_ODE_implicit_pure` is a bare name in
  `run_energy_implicit_phase`'s namespace.

**WHAT (B) BYPASSES:** by construction the state never reaches `<= 0`, so the shipped
`ENERGY_COLLAPSED` clean stops (`run_energy_phase.py:340`, `run_energy_implicit_phase.py:1007`)
**never fire**. That is the experiment — and exactly why this is not a fix (it manufactures energy).

**Floor choice.** `floor = 1e-3` [au = Msun·pc²/Myr²], a fixed small positive value ~13 orders of
magnitude below the collapse regime's initial `E0 ≈ 6.4e9` au (so the floor only bites once the
bubble has genuinely drained to near-zero). Sensitivity: the activation telemetry (`act_state`,
`min_Eb_seen`) shows whether the floor ever engaged; see §5.

## 3. Results — full all-configs matrix

Run: `bash docs/dev/transition/pt4/h3_run_matrix.sh` (OMP_NUM_THREADS=1, one sim/process, each cell
`timeout`-bounded with a small `--stop_t`). Per-cell rows in `h3_eval.csv`; trajectories in
`traj/h3_traj_<cfg>_<variant>.csv`; driver log `h3_run_matrix.log`. Outcomes: `crashed` /
`completed` (clean `end_reason`) / `timeout` (grind — partial jsonl salvaged by `h3_salvage_timeout.py`).

<!-- TABLE: filled from h3_eval.csv via h3_analyze.py -->
_(results table inserted below once the matrix completes)_

## 4. Trajectory evidence — the collapse configs

<!-- filled from traj CSVs -->

## 5. No-op confirmation — stall + healthy configs

<!-- filled -->

## 6. Verdict

<!-- filled -->

## 7. Artifacts (all committed under `docs/dev/transition/pt4/`)
- `h3_variants.py` — EBFLOOR monkeypatch (drive clamp + reflecting state floor) + V0/V1/V2/V3.
- `h3_run_variant.py` — one-sim/process driver; reads the run's `dictionary.jsonl` for trajectory,
  cooling-balance trigger, floor-activation telemetry.
- `h3_run_matrix.sh` — full (config, variant) matrix with per-cell timeout + stop_t.
- `h3_salvage_timeout.py` — recovers partial jsonl into eval row + trajectory on a grind/timeout.
- `h3_analyze.py` — tabulates `h3_eval.csv` + the no-op check.
- `h3_eval.csv` — one row per (config, variant): outcome, end_code, final state, floor activation,
  trigger, runtime.
- `traj/h3_traj_*.csv` — per-snapshot trajectories (t, R2, v2, Eb, Pb, R1, T0, Lgain, Lloss, ratio).
- `h3_run_matrix.log` — driver log (exact per-cell command lines + outcomes).
