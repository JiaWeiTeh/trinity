# P0 results — transition-trigger harvest (both clocks + candidate divergence)

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
> than you found it. **Keep both banner paragraphs at the top of every plan and
> analysis doc.**

Execution log for **P0** of `docs/dev/TRANSITION_TRIGGER_PLAN.md`. **Status
2026-06-15: IN PROGRESS — harness built and proven; real (hybr) harvest running.**

## Verifications closed before any code (the plan's "Read first")
- **`Lloss` is pure radiative cooling, no PdV** — `bubble_LTotal = L_bubble +
  L_conduction + L_intermediate` (`bubble_luminosity.py:706`, returned `:750–757`),
  each a radiative integral over the interior; PdV (`4πR2²v2·Pb`) is handled
  separately in the betadelta balance. ⇒ the energy ratio is a clean
  cooling-vs-injection fraction. (Plan OPEN item — resolved.)
- **The trigger lives only in phase 1b.** Phase 1a (`run_energy_phase.py`) has no
  cooling trigger (ends at `TFINAL_ENERGY_PHASE`). Live terminator is the inline
  check `run_energy_implicit_phase.py:1076`, honoring `phaseSwitch_LlossLgain`
  (default 0.05); the `make_cooling_balance_event(0.05)` factory is unpacked
  (`:650`) but unused.
- **Output fields available per segment** (`dictionary.jsonl`): `t_now`,
  `current_phase` ∈ {energy, implicit, transition, momentum}, `R2`, `v2`, `Eb`,
  `Pb`, `cool_beta/delta`, `bubble_Lgain/Lloss/LTotal`, `bubble_dMdt`,
  `Lmech_total/_W/_SN`, `F_ram` (= 4πR²Pb), `F_rad`, `F_HII`, `P_HII`,
  `pdot_total`, `betadelta_converged/_total_residual` (new runs). `rCloud` is a
  run constant in `metadata.json`. ⇒ F0/F1/F2/F4 and the F3 force components are
  all harvestable offline; no production change.

## Harness
`scratch/transition/harvest.py <run_dir> [--csv out]` — reads a finished run,
filters implicit rows, evaluates every candidate trigger on the same trajectory,
reports both clocks and each candidate's firing epoch, dumps a per-segment CSV.

## Compute reality (important)
Self-consistent **hybr** runs cost ~1.4e-4 Myr-sim / s-wall (Phase-3 cost gate),
so a steep/flat sweep to `stop_t = 3–4 Myr` is **hours** — not interactive. P0
real harvests are **background/overnight** jobs. The harness is validated on
existing output first; hybr runs are launched in the background and harvested on
completion.

## First data point — mock 4e3 (⚠️ LEGACY trajectory, illustrative only)
Harvested from the committed `outputs/mockOutput/mockFullrun/` — which is the
**legacy** (β-clamped, 0% converged) sample, so the *trajectory* is contaminated
(per the betadelta findings); use this only to prove the harness and the *shape*
of the divergence, **not** for any conclusion. CSV: `analysis/data/transition_mock4e3.csv`.

- **Clock A** (implicit / `t_trans`): 0.0034 → 0.0935 Myr.
- **Clock B** (1c transition phase): 0.0955 → 0.117 Myr, **length 0.022 Myr** —
  short here; the sound-crossing drain is a thin tail for this small bubble.
- **Eb-peak**: at the implicit/transition boundary (Eb still ~rising at handoff).
- **Candidate divergence (the headline the plan predicted — candidates do NOT
  coincide):**
  | candidate | fires at (Myr) | reading |
  |---|---|---|
  | F2 `t_cool/t_dyn < 1` | **0.0041** | ~phase start — fires far too early |
  | F0 `inst < 0.05` (current) | **0.0885** | near the Eb-peak |
  | F1 `cum > 1−η` (η 0.2–0.4) | **never** | only 56 % of injected energy radiated cumulatively by transition |

  So on this run F2 ≪ F0 ≪ F1 — three different epochs spanning the whole phase.
  Even discounting the legacy contamination, this validates that the
  cross-candidate comparison is the right experiment.

## Caveats to pin (feed P-sens)
- **F2 units unverified.** `t_cool = Eb/Lloss` only gives a time if `Eb` and
  `Lloss` are in consistent energy units (`Eb` is bubble energy in code/au;
  `Lloss` au luminosity). The "fires at phase start, t_cool/t_dyn≈0.4 throughout"
  result could be a unit artifact — **verify the unit consistency of Eb/Lloss
  before trusting any F2 epoch.**
- **Legacy contamination.** Re-harvest on hybr trajectories before any reading.

## Next
1. Harvest the **hybr** mock (`tt_mock_hybr`, running) — same config, hybr vs the
   legacy trajectory above (a clean control for the contamination).
2. Background a steep (α=−2) and a dense-flat (n1e5) hybr run; harvest both.
3. Verify F2 units; add F3 (force-ratio) once the surviving-force set is pinned.
4. Build the per-config overlay figure (Eb(t)/ratio(t) with firing epochs marked).
5. Assemble the divergence map → **Gate G0**.
