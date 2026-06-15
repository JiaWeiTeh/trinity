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
physically right for this low-mass cloud), while **F2 fires very early** (0.0065).
*(Earlier I called F2 "broken / mis-scaled / firing at t≈0" — that was wrong; see
the F2 diagnosis below. It is not units, and the mock actually starts adiabatic.)*

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
- **F2 fires early (0.0034)** — dense-flat is *already* radiative (t_cool/t_dyn =
  0.54) at the implicit-phase start, physically right for a dense bubble. See the
  F2 diagnosis below: **not units**, and not "broken" — it marks the
  radiative-interior onset, which precedes the momentum transition by ~60×.
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

## F2 (t_cool/t_dyn) diagnosis — NOT units (2026-06-15, corrects earlier "broken")
The early-firing of F2 is **not** a unit bug and **not** "broken." Raw values
(from the CSVs), `t_cool = Eb/Lloss`, `t_dyn = R2/v2`:

| config | t | t_cool [Myr] | t_dyn [Myr] | t_cool/t_dyn |
|---|---|---|---|---|
| mock (n5e2) | 0.0034 (start) | 0.0123 | 0.0092 | **1.34** (adiabatic) |
| mock | 0.0065 | ~0.013 | ~0.013 | ~1.0 (crosses) |
| mock | 0.30 (end) | 0.107 | 0.538 | 0.20 |
| dense-flat (n1e5) | 0.0034 (start) | 0.0046 | 0.0084 | **0.54** (already radiative) |
| dense-flat | 0.197 (Eb-peak) | ~0.024 | ~0.65 | **~0.04** |

- **Units consistent.** `t_cool` and `t_dyn` are both ~0.01 Myr early — impossible
  if one were erg and the other code units (~10⁴³ mismatch). `Eb` is au-energy
  (`Eb ≈ ⅕·Lgain·t`), so `Eb/Lloss` is genuinely Myr, same as `R2/v2`.
- **F2 is physical but measures the WRONG transition.** `t_cool<t_dyn` marks the
  **radiative-interior onset** — when the bubble interior cools faster than it
  expands. That happens **~60× earlier** than the energy→momentum transition
  (dense-flat: t_cool<t_dyn at 0.003 Myr, F0/Eb-peak at 0.197). The bubble stays
  energy-driven (Eb growing, Lgain>Lloss) long after the interior is locally
  radiative.
- **The threshold k is the whole lever, not units.** k=1 fires at the radiative
  onset; the actual transition sits at **t_cool/t_dyn ≈ 0.04–0.05** (t_cool ~ 1/20
  t_dyn). Tuning k≈0.05 would make F2 ≈ F0 — i.e. F2 becomes a re-parameterization
  of F0/F1, not an independent criterion.
- **Validates the plan's literature caveat.** The instantaneous `Eb/Lloss, R2/v2`
  form is *our* construction; Mac Low & McCray's `t_cool` is the **cumulative**
  balance `∫Lloss dt = Eb` — i.e. essentially **F1 (cumulative-energy)**, the
  candidate already behaving well (η≈0.3 → Eb-peak). So the principled timescale
  criterion collapses onto the cumulative one; the *instantaneous* k=1 form is the
  one to drop (or relabel as a radiative-onset diagnostic, not a transition).

## Caveats to pin (feed P-sens)
- **F2 units — RESOLVED, not a unit issue** (see the F2 diagnosis section). The
  early firing is physical (radiative-interior onset), ~60× before the momentum
  transition; the open P-sens question is now the *threshold/definition*
  (instantaneous k=1 vs the cumulative MLM88 form = F1), not units.
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
