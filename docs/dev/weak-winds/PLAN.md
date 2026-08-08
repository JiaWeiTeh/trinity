# Weak-winds sensitivity study — plan

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

**Status (2026-08-08):** 🔵 actionable — harness + tests shipped and green; the 15-run science
sweep is designed but not yet executed (HPC-sized; see §6).

## 1. Why this study exists

A collaborator asked whether TRINITY can "switch off winds". A strict off switch
(`Lmech_W = pdot_W = 0`) is **not currently runnable**: the free-streaming initial
conditions are built entirely from wind quantities (`get_y0`,
`trinity/phase0_init/get_InitPhaseParam.py:107-176` @ the commit that adds this doc),
`update_feedback` computes `v_mech_total = 2·Lmech_total/pdot_total` on every call —
0/0 = NaN before the first SN — and phases 1a/1b integrate a shocked-wind bubble
that would not exist. A strict-off mode is a separate, larger feature (momentum-phase
entry path + guards).

What **is** runnable today, with zero code changes, is the next-best science
question: **how much do winds matter?** `FB_thermCoeffWind` (default 1, registry
`trinity/_input/registry.py`) models the thermalization efficiency of colliding
winds and scales the wind terminal velocity by √coeff in the loader
(`trinity/sps/read_sps.py:210-222`). Stepping it down a ladder toward zero turns
the wind channel down smoothly while every equation, phase, and solver keeps
operating inside its designed regime. The extrapolation of the trend as coeff → 0
brackets the strict no-winds answer without ever dividing by zero.

## 2. The knob, exactly

With `FB_mColdWindFrac = 0` (default), the loader applies, per table row:

| quantity | scaling | meaning |
|---|---|---|
| `v_wind` | × √c | slower effective wind — energy not thermalized |
| `Lmech_W` | × c | wind power down linearly |
| `pdot_W` | × √c | wind momentum down as √c |
| `Mdot_W` (= pdot²/2L) | invariant | same mass loss — it's an efficiency knob, not a mass knob |
| SN channel (`Lmech_SN`, `pdot_SN`) | untouched | SNe from ~3.6 Myr take over regardless |
| Radiation (`Qi`, `Li`, `Ln`, `Lbol`) | untouched | photoionization + radiation pressure unchanged |

`test/test_weak_winds.py::test_thermcoeff_scaling_contract` pins this table.

Physical sanity floor: at c = 0.01 the wind is still highly supersonic
(v_w ~ 2000 → 200 km/s vs ~10 km/s ionized-gas sound speed), so a termination
shock and hot bubble still form — the Weaver machinery stays valid. Below
c ≈ 3e-4 the wind would go transonic and the model picture itself breaks;
the ladder must not go there.

Free-streaming initial conditions shift but stay finite (derivation in
`test_weak_winds.py::test_get_y0_free_streaming_scalings`, all verified to 1e-10):

- v₀ ∝ √c (slower launch), dt_freestream ∝ c^(-3/4) (longer), r₀ ∝ c^(-1/4),
  E₀ ∝ c^(1/4). Degeneracy happens only at c = 0 exactly — the strict-off limit
  this study deliberately avoids.

## 3. Hypotheses to test (falsifiable, per regime)

- **H1 (early phase):** shell radius/velocity during the energy phase drop with c
  (weaker Weaver driving: R ∝ L^(1/5) t^(3/5) predicts a shallow R ∝ c^(1/5) —
  measurable but small; the force budget should shift from `F_ram`/bubble pressure
  toward `F_HII` + `F_rad`).
- **H2 (HII-dominated regimes):** in the baseline cloud, P_HII + radiation already
  rival winds, so fates (dissolve/collapse/t at rCloud) should barely move until
  c ≲ 0.1 — this is the "winds don't matter much here" outcome the collaborator
  can quote. *(Update 2026-08-08: the smoke pair already refutes this for the
  baseline cloud — at c = 0.1 the fate flips to recollapse at 0.28 Myr; see
  `FINDINGS.md`. The nCore = 1e5 baseline is denser than the H2 intuition
  assumed; H2 remains open only for genuinely diffuse regimes, e.g.
  f1edge_lowdens.)*
- **H3 (dense regime, f1edge_hidens):** weak feedback + dense gas is
  gravity-dominated; reducing c should pull the collapse time earlier
  approximately monotonically.
- **H4 (SN era):** after ~3.6 Myr the totals are SN-dominated; trajectories at
  different c should converge once SNe switch on, unless the early-phase
  divergence has already changed the shell's location/mass enough to matter
  (hysteresis — the interesting outcome).
- **H0 (control):** c = 1.0 must reproduce the untouched baseline for each cloud
  — the equivalence gate before quoting any trend (rule 5, root CLAUDE.md).
  *(Confirmed 2026-08-08 on the baseline cloud: `max |dR2/R2| = 0.000e+00` over
  0.5 Myr — see `FINDINGS.md` §H0. Still unchecked on the two 1e7 clouds, which
  batch 1 covers.)*

## 4. Study design

Grid = 3 cloud regimes × 5-rung ladder = **15 runs**
(`harness/weak_winds_sweep.param`, hybrid tuple × Cartesian sweep):

- Clouds (the documented stiffest solver edges, root CLAUDE.md §Planning protocol):
  - `baseline` 1e5 Msun, sfe 0.30, nCore 1e5 — energy-driven worked example
    (`param/simple_cluster.param` + defaults);
  - `f1edge_lowdens` 1e7 Msun, sfe 0.50, nCore 1e2 — strong feedback, diffuse;
  - `f1edge_hidens` 1e7 Msun, sfe 0.01, nCore 1e6 — weak feedback, dense.
- Ladder: `FB_thermCoeffWind ∈ {1.0, 0.3, 0.1, 0.03, 0.01}` — log-spaced,
  control included; two decades is enough to see any power-law trend and safely
  above the transonic floor (§2).
- `stop_t 15` (default horizon) so fates are comparable across runs.

### Measurements (per run, all already in `dictionary.jsonl`)

1. Trajectories: `R2(t)`, `v2(t)`, `Eb(t)`, `T0(t)`.
2. Force budget vs t: `F_ram_wind`, `F_ram_SN`, `F_HII`, `F_rad`, `F_grav`
   (registry `runtime_force` category) — the direct "which channel drives"
   answer, and the quantity H1/H2 are judged on.
3. Phase chronology: time of each `current_phase` change (energy → implicit →
   transition → momentum).
4. Fate: `SimulationEndCode`/`SimulationEndReason`, collapse/dissolution time,
   max radius, t at R2 = rCloud.
5. Solver health: run completion, termination reason, wall time — rung failures
   are findings (see risks), not discards.

### Comparison protocol

- Compare runs only in **separate processes** at **matched simulation time**
  (root CLAUDE.md rule 5; runs truncate at different t).
- **Do not expect byte-identical outputs across processes**: the SPS loader's
  log-column exponentiation (`10**x` in `sps_columns.convert_to_canonical_au`)
  can wobble by 1 ULP between loads (numpy SIMD/scalar dispatch is
  allocation-alignment-dependent; measured 2026-08-08, ~45/800 rows of the
  derived `Lmech_SN`, all exactly 1 ULP, one row flipping clamped-0 ↔ +1 ULP).
  H0's "reproduce the baseline" gate therefore means: trajectories agree to
  tight tolerance (rel ≤ 1e-9 at matched t), not byte equality. This is a
  pre-existing loader property, not a knob effect — flagged here, not fixed.

## 5. Verification already in place (shipped with this doc)

- `test/test_weak_winds.py` (9 fast + 1 stress):
  - scaling contract of the loader on the full ladder (§2 table);
  - feedback pipeline (`get_current_sps_feedback`) finite on the ladder in both
    eras, wind-only `v_mech_total` scaling √c vs control, radiation untouched;
  - free-streaming ICs (`get_y0`) finite + the four scaling exponents;
  - stress: two end-to-end boots (control vs c = 0.03, M43-scale bounded run
    mirroring `test_early_phase_override`), weak shell must launch < 0.5× the
    control velocity.
- Smoke pair (baseline cloud, c ∈ {1.0, 0.1}, `stop_t 1.5`):
  `harness/weak_winds_smoke.param`; result recorded in `FINDINGS.md`.

## 6. Execution order (ladder-first, not all-at-once)

Weak-Lw pushes `bubble_structure`'s stiff integrator toward untested regimes; run
so that a failure localizes the boundary instead of wasting the sweep. **The
executable sequence — one batch per rung, each with its command and pass/fail
gate — is `RUNBOOK.md`.** In outline:

1. Batch 0: the H0 plumbing gate (knob at default is inert). Minutes.
2. Batch 1: c = 1.0 control across all three clouds — also the cost calibration
   for everything that follows.
3. Batches 2–5: descend one rung at a time (0.3 → 0.1 → 0.03 → 0.01), checking
   the gate between batches. A failed rung is recorded (config + traceback + last
   good snapshot) and the descent stops **for that cloud only**.
4. Batch 6: harvest into `data/` as committed CSVs (provenance header: commit,
   command, param hash — `docs/dev/transition/PROVENANCE_PROTOCOL.md`), figures
   into `figures/` (Agg backend, no usetex, dpi ≈ 130–140, reading only
   committed CSVs — model `docs/dev/performance/harness/make_f1_figures.py`).
   Write up against H1–H4 in `FINDINGS.md`.

`harness/weak_winds_sweep.param` still describes the whole 15-run grid in one
file; it is the design of record, but prefer the batches for actually running —
an all-at-once sweep cannot gate between rungs.

## 7. Risks & known sharp edges

- **Stiff-solver fragility at low c** (energy phase 1b: conduction/dMdt/cooling
  at low Lw). Mitigated by the rung-descent order; failures are boundary data.
- **Interpolator leakage:** the SPS interpolators are global cubics, so post-SN
  knots leak into the wind-only era at the ~1e-10 relative level (measured; see
  test comments). Harmless for science; matters only for over-tight equivalence
  tolerances.
- **Loader ULP jitter** (§4, comparison protocol) — don't gate on byte equality
  across processes.
- **Don't over-read the knob:** thermCoeff reduces momentum as √c too; the study
  measures "weaker winds", not "energy-free winds". Claims about a strict
  no-wind universe must come from the extrapolated trend, clearly labeled.
- **rCloud plausibility:** all three clouds are existing validated configs;
  any new cloud added later must pass `rCloud_max` plausibility (root CLAUDE.md).

## 8. Implications by outcome

- If fates are c-insensitive down to 0.1 (H2 confirmed) in a regime, the
  collaborator's "can we ignore winds?" is a quotable yes *for that regime*, with
  the force-budget plot as the mechanism.
- If trajectories diverge early and never reconverge (H4 hysteresis), winds set
  initial conditions for the SN era and cannot be dropped even where their
  instantaneous force is small.
- If the trend vs c is a clean power law per H1, the c → 0 extrapolation is the
  cheap stand-in for the strict winds-off mode and quantifies what building that
  mode (momentum-entry path + 0/0 guards) would buy.
- Solver-failure boundary (lowest runnable c per cloud) is the hard scope line
  for any future strict-off implementation.
