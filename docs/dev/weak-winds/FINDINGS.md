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

**Status (2026-08-09):** 🟡 partial — batches 0 and 1 PASS. The control rung exposed two
design defects (a degenerate dense arm, a radius-capped diffuse arm) and one
result that reframes the study: **P_HII is identically the bubble pressure while the
Strömgren cap binds**. Batches 2–5 have not run; fix the cloud set before running them.

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

## Batch 1 — control rung c = 1.0 — PASS (2026-08-09)

`harness/batches/batch1_c1p0.param`, 3/3 succeeded, gate PASS. Artifacts:
`data/control_c1p0.csv`, `figures/control_c1p0_{R2,forces}.png`.

| cloud | fate | t_end | R2_end | R2_max | wall |
|---|---|---|---|---|---|
| `1e5_sfe030_n1e5` baseline | stop_t reached | 15.000 | 438.8 pc | 438.8 | 66.6 m |
| `1e7_sfe001_n1e6` hidens | **collapsed** (small radius) | 0.047 | 0.57 pc | 0.57 | 14.2 m |
| `1e7_sfe050_n1e2` lowdens | **stop_r cap** (large radius) | 3.734 | 500.0 pc | 500.0 | 29.0 m |

**Cost calibration (the reason this rung runs first):** ~67 min per rung wall-clock
with `--workers 3` on a 4-core container; the baseline dominates and the two 1e7
clouds self-terminate early. Five rungs ≈ 5.5 h sequential. Earlier projections in
this workstream (~4 h, then ~2 h, for the baseline alone) were both too pessimistic —
the run accelerates sharply mid-way as the shell grows, then slows again after
~10 Myr. Quote the 66.6 min measurement, not the projections.

### Two design defects this rung exposed

1. **`1e7_sfe001_n1e6` is degenerate as a ladder arm.** It collapses at t = 0.047 Myr
   *at full wind strength* — R2 never exceeds 0.57 pc, and it never leaves phase
   1c. Every weaker rung can only collapse at least as fast, so all five rungs
   return the same fate and the arm carries no information about `c`. Replace it
   with a dense-but-viable cloud (higher `sfe`, or `nCore` nearer 1e4–1e5) that
   forms a bubble at c = 1 so weakening the wind has something to change.
   **Resolved 2026-08-12 by the two probes below: the fix is `nCore`, not `sfe`.**
2. **`1e7_sfe050_n1e2` terminates on `stop_r` = 500 pc at t = 3.73 Myr** — the
   default radius cap, not physics, and it fires just as SNe switch on (~3.6 Myr).
   **H4 (SN-era reconvergence) is untestable on this cloud as configured.**
   *(Decision 2026-08-09, maintainer: keep `stop_r` = 500 pc — it is already a
   very large radius. Accepted as a scope limitation, not a defect: for the
   diffuse arm the ladder metric is **"time to reach 500 pc"**, a clean
   monotonic discriminator, and H4 is tested on the baseline cloud, which runs
   the full 15 Myr and does sample the SN era.)*

### P_HII is not an independent driver — it is the bubble pressure, relabelled

The most consequential result of the control rung, and it reframes the whole study.

Sampling the driving terms in the baseline run (`Pb = F_ram / 4πR2²` vs `P_HII`):

| t [Myr] | Pb | P_HII | Pb/P_HII | phase |
|---|---|---|---|---|
| 0.0000 | 3.2004e+09 | 9.6029e+09 | 0.3333 | energy |
| 0.0007 | 1.9482e+07 | 1.9919e+07 | 0.9781 | energy |
| 0.0160 | 1.4781e+06 | 1.4781e+06 | **1.0000000000** | implicit |
| 0.2952 | 1.4402e+04 | 1.4402e+04 | **1.0000000000** | implicit |
| 15.000 | 5.4917e+00 | 5.4542e+00 | 1.0069 | implicit |

Ten-digit identity is not a coincidence; it is algebra
(`trinity/shell_structure/shell_structure.py` @ `054ce6b`):

- `shell_n0 = (mu_ion_shell/mu_convert) / (k_B · TShell_ion) · Pb` — the shell's
  inner density is set by **pressure balance with the bubble** (line ~124);
- `n_IF_Str` is **capped at `shell_n0`** ("pressure equilibrium for thin skins",
  line ~239);
- `P_HII = (mu_convert/mu_ion_shell) · n_IF_Str · k_B · TShell_ion`.

Substituting the cap gives **P_HII ≡ Pb identically**. So whenever the Strömgren
density is cap-limited — which is the entire implicit phase here — the "HII
pressure" channel is the bubble pressure wearing a different name, and the bubble
is wind-powered. P_HII is genuinely independent only when the cap is slack: early
phase 1a (ratio 0.33 → 0.98) and late times (1.0069).

**Why this matters:** the original H2 assumed P_HII was an independent channel
that would hold the shell up as winds weakened, making dense clouds
wind-insensitive. That reasoning is mechanically wrong in the cap-limited regime —
weaken the wind, and Pb falls, and P_HII falls *with it*. This is the mechanism
behind the smoke pair's fate flip, and it predicts the ladder will be more
wind-sensitive than H2 supposed, not less.

**Caveat:** established on the baseline cloud's control run. It should be
re-checked per cloud and per rung — a weaker wind may leave the cap slack, in
which case P_HII decouples and does become an independent floor. That transition,
if it happens, is itself a result worth reporting.

### Harness correction made here

`harvest.py` originally omitted `F_ram` — the shell-facing force from the *bubble*
pressure, which is how the wind actually drives the shell during the energy phase.
Without it, a force-budget read of the CSV attributes 87–99% of the driving to
`F_HII` and 3–16% to the free-streaming `F_ram_wind`, which misses the wind's
entire pathway (a first pass at this analysis made exactly that error). `F_ram`,
`F_ion_in` and `P_drive` are now harvested, and the force figure leads with
`F_ram`. Read that panel as *which term wins* — the ODE drives on
`P_drive = max(Pb, P_HII)`, so the terms compete rather than sum.

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

## Dense-arm replacement probes (2026-08-12) — the blocker is density, not feedback

Batch 1 defect 1 asked for a dense-but-viable arm and offered two levers: raise
`sfe`, or relax `nCore`. Both were probed. **Only `nCore` works, and `sfe` fails in
the direction opposite to the one expected** — worth recording because the intuition
"more feedback ⇒ bigger bubble" is wrong here and would otherwise be retried.

### Probe 1 — raising `sfe` at `nCore` = 1e6 makes it strictly worse

`harness/batches/probe_dense_sfe.param`, `mCloud` = 1e7, c = 1, 3/3 succeeded.
Row 1 is the batch-1 arm, for reference:

Artifact: `data/probe_dense_sfe.csv`.

| `sfe` | fate | t_end [Myr] | R2_max [pc] | v2 at end [km/s] |
|---|---|---|---|---|
| 0.01 (batch-1 arm) | collapsed, small radius | 0.047 | 0.57 | — |
| 0.05 | collapsed, small radius | 0.023 | 0.44 | −14.9 |
| 0.10 | collapsed, small radius | 0.018 | 0.37 | −160.0 |
| 0.30 | **collapse-velocity runaway** | 0.009 | 0.29 | −500.0 |

Every row has `R2_max > R2_end` (the shell turned over) and `v2 < 0` (it is moving
inward), and the infall speed steepens with `sfe` — the collapse is not marginal.

Monotonic over four points: **more star formation ⇒ earlier collapse at a smaller
radius.** `mCluster = sfe·mCloud` enters the shell equation of motion through gravity
(`F_grav ∝ mCluster + mShell/2`), so at this density each increment of `sfe` deepens
the potential well faster than it adds outward push; it also removes gas, shrinking
`rCloud` so the shell starts deeper in. Feedback loses the race. **Do not retry the
`sfe` lever on a dense cloud** — it cannot rescue this arm at any value.

### Probe 2 — relaxing `nCore` at `sfe` = 0.01 works; 1e5 is the pick

`harness/batches/probe_dense_ncore.param`, `mCloud` = 1e7, `sfe` = 0.01, c = 1.
The container was recycled mid-probe, so these are **last-sample-before-kill**, not
final fates — but all three had cleared the "expands rather than collapsing" bar,
which is what the probe was asked to settle:

Artifact: `data/probe_dense_ncore.csv`.

| `nCore` | last t [Myr] | R2 there [pc] | v2 [km/s] | turned over? | phase |
|---|---|---|---|---|---|
| 1e3 | 0.103 | 6.95 | **+39.0** | no (`R2_max = R2_last`) | implicit |
| 1e4 | 0.095 | 3.98 | **+23.8** | no (`R2_max = R2_last`) | implicit |
| **1e5** | 0.044 | **1.47** | **+17.8** | no (`R2_max = R2_last`) | implicit |
| 1e6 (batch-1 arm) | 0.047 | 0.57 → collapsed | negative | **yes** | 1c |

`R2_max = R2_last` exactly, for all three — the radius is strictly monotonic to the
last sample, so none of them had begun to turn over, and all three have `v2 > 0`.
That is a stronger statement than "bigger radius": the sign of the velocity and the
absence of a turning point separate these cleanly from the `sfe` probe above, where
every run had already turned over and was falling inward.

The last two rows are the decisive comparison: at matched t ≈ 0.045 Myr the 1e5 cloud
is at 1.47 pc and still growing, where the 1e6 cloud had already turned around at its
0.57 pc maximum — a 2.6× larger radius on the same clock.

**Recommended replacement arm: `mCloud` = 1e7, `sfe` = 0.01, `nCore` = 1e5.** It keeps
what made the arm valuable (most massive cloud, weakest feedback, densest survivor)
while actually forming a bubble, so weakening `c` has something to act on.
**Caveat to close before relying on it:** 1e5 was only observed to t = 0.044, the
shortest of the three. Re-run it alone to `stop_t` and confirm it does not turn around
later before cutting batches 2–5 against it.

### Probe 3 — confirmation run, still open

`harness/batches/probe_dense_confirm.param` re-runs the 1e5 candidate alone to
`stop_t` = 1.5 to close that caveat. It has been started three times and killed by
container recycling each time; furthest reach so far is **t = 0.0118 Myr, R2 = 0.716 pc,
v2 = +33.7 km/s, no turnover** (`data/probe_dense_confirm.csv`). Consistent with probe 2
and with the arm being viable, but it does **not** yet close the caveat — the run has
not reached the t ≈ 0.047 Myr mark where the old 1e6 arm collapsed, let alone `stop_t`.
Treat `nCore` = 1e5 as the recommended arm *pending confirmation*, not as settled.

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
