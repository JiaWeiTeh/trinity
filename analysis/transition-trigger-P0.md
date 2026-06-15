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
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `analysis/data/`, or a force-added
> harness/figure under `scratch/` as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

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

## Second data point — mock 4e3 HYBR (clean control, 2026-06-15)
Fresh `betadelta_solver=hybr` run of the *same* mock config (67 implicit segs,
**energy-driven to stop_t = 0.3, no transition**). CSV:
`analysis/data/transition_mock_hybr.csv`; config: `scratch/transition/mock_hybr.param`.

**The legacy "transition at 0.089 Myr" was a clamp artifact.** On the clean hybr
trajectory the current trigger **F0 never fires** (ratio_F0 ∈ [0.40, 0.85], never
nears 0.05) — the bubble stays energy-driven to stop_t with **no transition phase
at all** (Clock B absent). Legacy's early transition came from the β-clamp's
contaminated `Lloss` crossing 0.05 — exactly the profile-blind story from the
betadelta work. **The trigger's behaviour is solver-dependent.**

| candidate | hybr mock | legacy mock | note |
|---|---|---|---|
| F0 `inst<0.05` | **never** | 0.089 | legacy crossing was a clamp artifact |
| F1 `cum>1−η` | **never** (frac_cum 0.49) | never (0.56) | <60 % radiated either way — energy-driven cloud |
| F2 `tc/td<1` | **0.0065** (phase start) | 0.0041 (phase start) | fires immediately on **both** — units/def. flag |
| F4 `R2>rCloud` | never | never | small bubble |

On the clean trajectory **F0 and F1 agree** (energy-driven, never transition —
physically right for this low-mass cloud), while **F2 fires at phase start on both
trajectories** → the instantaneous `t_cool=Eb/Lloss, t_dyn=R2/v2` form is
mis-scaled or mis-defined (a transition trigger firing at t≈0 is unphysical). Top
P-sens priority.

## Third data point — dense-flat HYBR (transitioning config, 2026-06-15)
`tt_dense_flat` (1e6, n1e5, α=0), 57 implicit + 33 transition segs, transitions
cleanly. CSV: `analysis/data/transition_dense_flat.csv`; config:
`scratch/transition/dense_flat.param`.

- **Clock A** (t_trans): 0.0034 → **0.210 Myr**. **Clock B** (1c): 0.212 → 0.247,
  length **0.036 Myr**. **Eb-peak: t=0.197 Myr, INTERIOR** (the physical reference).
- **F0 fires AT the Eb-peak** (0.197) — the current trigger is well-timed for a
  clean-transition config.
- **F1(η=0.30) coincides with F0/Eb-peak** (0.197); η=0.40 earlier (0.098);
  **η≤0.25 never fires** — `frac_cum` maxes at **0.708**, i.e. only ~71 % of the
  injected energy is radiated by transition. The **retained fraction ≈ 0.29**
  lands squarely in the literature η≈0.25–0.3 range (the Lancaster cross-check
  passes). So cumulative-energy with η≈0.3 ≈ F0 here.
- **F2 fires at phase start (0.0034) AGAIN** — third config in a row. The
  instantaneous `t_cool=Eb/Lloss, t_dyn=R2/v2` form fires at t≈0 universally →
  broken as a transition discriminator (units or definition). Confirmed P-sens
  priority.
- F4 (blowout): never (flat profile, R2 < rCloud).

### Emerging divergence map (3 configs, preliminary)
| config | fate | F0 (current) | F1 (η≈0.3) | F2 inst t_cool/t_dyn | retained @ end |
|---|---|---|---|---|---|
| mock hybr (4e3 flat) | energy-driven | never | never | t≈0 | 0.51 |
| dense-flat (1e6 n1e5) | transitions | **0.197 = Eb-peak** | **0.197** | t≈0 | **0.29** |
| steep (1e6 α−2) | (running) | — | — | — | — |

Early read: **F0 and F1(η≈0.3) agree with each other and with the Eb-peak** on the
configs run so far (fire together when the bubble transitions, both "never" when
it doesn't); **F2 is broken** (fires at t≈0 always). Awaiting steep (the
stall/blowout crux) before G0.

## Caveats to pin (feed P-sens)
- **F2 units unverified.** `t_cool = Eb/Lloss` only gives a time if `Eb` and
  `Lloss` are in consistent energy units (`Eb` is bubble energy in code/au;
  `Lloss` au luminosity). The "fires at phase start, t_cool/t_dyn≈0.4 throughout"
  result could be a unit artifact — **verify the unit consistency of Eb/Lloss
  before trusting any F2 epoch.**
- **Legacy contamination.** Re-harvest on hybr trajectories before any reading.

## Next
1. ~~Harvest the hybr mock~~ **DONE** (above) — F0 never fires on the clean
   trajectory; legacy transition was a clamp artifact.
2. ~~Harvest dense-flat~~ **DONE** (above) — F0 = F1(η0.3) = Eb-peak at 0.197;
   retained ≈0.29 (in lit range). Steep (`tt_steep`) still running.
3. **Verify F2 units** (now the top flag — fires at t≈0 on both mocks); add F3
   (force-ratio) once the surviving-force set is pinned.
4. Build the per-config overlay figure (Eb(t)/ratio(t) with firing epochs marked).
5. Assemble the divergence map → **Gate G0**.
