# Weak-winds — findings

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

**Status (2026-08-08):** 🟡 partial — batch 0 (H0 plumbing gate) PASSES exactly, and the
smoke pair shows a strong signal on the baseline cloud; batches 1–5 (PLAN §4, RUNBOOK)
have not run, so nothing here generalizes beyond that one cloud yet.

## Smoke pair (2026-08-08, in-container)

Config: `harness/weak_winds_smoke.param` — baseline cloud (1e5 Msun, sfe 0.30,
nCore 1e5, rCloud 1.69 pc, mCluster 3e4), `FB_thermCoeffWind ∈ {1.0, 0.1}`,
`stop_t 1.5` Myr. Command:
`python run.py docs/dev/weak-winds/harness/weak_winds_smoke.param --workers 2 --yes`.
Both runs exit SUCCESS. Wall time ~10 min (weak) / ~35 min (control), 2 workers.
Artifacts: `data/smoke_pair.csv` (harvested trajectories + force budget, provenance
header), `figures/smoke_pair_R2.png`, `figures/smoke_pair_forces.png`.

### Headline: at 10× weaker winds the baseline cloud's fate flips

| | control c = 1.0 | weak c = 0.1 |
|---|---|---|
| fate | STOPPING_TIME — still expanding at t = 1.5 Myr, R2 = 91.3 pc, v2 = +58 km/s | **SHELL_COLLAPSED (code 4)** at t = 0.282 Myr, R2 = 0.90 pc, v2 = −9.8 km/s |
| phase path | energy → implicit (still implicit at 1.5 Myr) | energy → implicit (0.003) → transition (0.160) → momentum (0.182) → collapse |
| max R2 | > 91 pc (beyond cloud from t ≈ 0.09) | 1.29 pc at t = 0.19 — never leaves the cloud |

Matched-t states (from `data/smoke_pair.csv`):

| t [Myr] | control R2 [pc] (phase) | weak R2 [pc] (phase) |
|---|---|---|
| 0.05 | 1.236 (implicit) | 0.765 (implicit) |
| 0.16 | 2.737 (implicit) | 1.262 (transition) |
| 0.28 | 6.176 (implicit) | 0.896, v2 = −10 (momentum) |

### Reading

1. **Early phase follows Weaver quantitatively (H1 ✓):** at t = 0.05 Myr the
   radius ratio is 0.765/1.236 = **0.619** vs the adiabatic prediction
   (0.1)^(1/5) = **0.631** — within 2%, before cooling/phase divergence.
2. **The fate flip is phase chronology, not the direct wind force.** At t ≈ 0.1
   the bubble energy is Eb ≈ 1.1e7 (ctrl) vs 8.7e5 au (weak) — ~12× less, i.e.
   the wind-power ratio plus cooling. The weak bubble's Eb collapses on the
   cooling/PdV budget → transition at 0.16 Myr → momentum phase, where
   P_HII + F_rad (F_HII 1.7e6, F_rad 1.1e6 au) cannot beat gravity
   (F_grav 2.6e6 au) in a nCore = 1e5 cloud → recollapse. **H2 is refuted for
   this cloud**: the dense baseline is *not* HII-dominated; winds are decisive.
3. **SNe never got a vote:** collapse at 0.28 Myr precedes the table's SN onset
   (3.61 Myr). Whether SNe could revive a re-collapsed cloud is outside this
   smoke's horizon and involves TRINITY's recollapse machinery — flagged for
   the main sweep, where `stop_t 15` and all three clouds apply (H3/H4 open).
4. **Scope guard:** one cloud, one rung, `stop_t 1.5`. Do not quote beyond
   "in the dense baseline cloud, winds at 1/10 thermalization flip the fate."

## H0 plumbing gate — PASS (2026-08-08, batch 0)

Setting `FB_thermCoeffWind` to its schema default is **exactly inert**, so the
smoke pair's control arm is a valid reference and the divergence above is
physics, not plumbing.

Config: `harness/batches/batch0_h0_{plumbing,untouched}.param` — baseline cloud,
`stop_t 0.5`, one arm naming the knob at `1.0` and one never mentioning it.
Command and gate: `RUNBOOK.md` §Batch 0.

| | explicit `1.0` | untouched |
|---|---|---|
| snapshots | 171 | 171 |
| final state at t = 0.5 | R2 = 17.594194 pc, v2 = 64.2909 | R2 = 17.594194 pc, v2 = 64.2909 |
| wall (4-core container) | 15.2 min | 14.6 min |

`check_batch.py --compare` over 200 matched-t samples: **`max |dR2/R2| = 0.000e+00`**
against a 1e-9 tolerance — not merely within tolerance but identical. Note this
window (t = 0 … 0.5 Myr) fully contains the smoke pair's divergence and the weak
arm's collapse at 0.28 Myr, so the comparison in the previous section rests on a
verified reference.

Two provenance notes, for exactness: (1) `metadata.json` records
`FB_thermCoeffWind = 1.0` for *both* arms, because it stores the resolved value
including the schema default — which is why the harness reads the knob from
metadata rather than from the run-folder name; (2) the explicit arm was executed
from a pre-rename copy of its param file whose only difference was `path2output`
(it landed in `outputs/weak_winds_study/h0_explicit` rather than
`outputs/weak_winds_h0/explicit`). Physics and gate result are unaffected; a
fresh run of the committed file writes to the documented path, as the untouched
arm — which did use the committed file — demonstrates.

## Numerical findings (loader-level, pre-existing)

Both found 2026-08-08 while building `test/test_weak_winds.py`; both are
properties of the current loader, not of the knob:

1. **1-ULP cross-load jitter:** the SB99 log columns pass through `10**x`
   (`sps_columns.convert_to_canonical_au`), and numpy's pow can differ by 1 ULP
   between loads depending on buffer alignment (SIMD vs scalar peel lanes).
   The derived `Lmech_SN = Lmech_total − Lmech_W` inherits it; one late-time row
   can flip between clamped-0 and +1 ULP (~45/800 rows in the incident
   observation, all exactly 1 ULP). Consequence: **never gate cross-process
   comparisons on byte equality of SPS-derived quantities** (PLAN §4).
2. **Global-spline leakage:** the SPS interpolators are global cubics
   (not-a-knot), so post-SN knots leak into the wind-only era at ~2e-10
   relative (measured at c = 0.01). Harmless for science; bounds how tight an
   interpolated-quantity equivalence tolerance can be.
