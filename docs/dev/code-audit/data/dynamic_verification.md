# Phase 6 — dynamic verification

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

**Status (2026-07-30):** 🔵 ACTIVE — four probes run against **post-hotfix `main`**
(`hotfix/early-approximations` merged). The table-bounds probe is still in flight.

## Why this phase exists

Static review cannot tell you the physics is right. A dropped term shows up as a wrong
slope regardless of what the code says about itself; a leaked global shows up as two
runs disagreeing. These probes test the trajectory, not the source.

**All results below are on the fixed code.** The pre-fix reference is
`data/runs/phase6_short.log`, kept and labelled.

Configs: `harness/phase6_det{A,B}.param` (identical, `stop_t = 0.5`),
`harness/phase6_long.param` (`stop_t = 15`). All are `simple_cluster` physics —
`mCloud 1e5`, `sfe 0.3`, `densPL_alpha 0` (uniform), everything else schema default.

---

## 1. Determinism / global-state probe — **PASS**

Two identical configs, **separate processes** (CLAUDE.md rule 5: trinity leaks
module-level state in-process, so an in-process second run is not a valid baseline).

```
165 complete snapshots, t = 3.4e-7 .. ~0.4 Myr, spanning energy + implicit
byte-compare:  cmp -> identical
md5 (A) = md5 (B) = d9c18ef0aeda50575270dbf93908c665
```

**Byte-identical**, not merely numerically close. The two processes were killed at
different points by a container restart (A reached 171 snapshots, B 166), so the
comparison is over the 165 complete snapshots both wrote — which covers the whole energy
phase and most of the implicit phase. No evidence of run-to-run nondeterminism.

*Not tested:* the second-in-process probe (run twice in one interpreter) that would expose
the module-level global leakage CLAUDE.md documents. That is a separate check and remains
open.

## 2. Invariant scan — **PASS with one flag**

`harness/check_invariants.py` over 150 snapshots.

- **Sign violations: none.** No negative radius, energy, pressure, temperature, mass.
- **Monotonicity violations: none.**
- **Non-finite values: 8 keys are NaN at snapshot 0** — `bubble_Lgain`, `bubble_Lloss`,
  `betadelta_total_residual`, `residual_{Edot1,Edot2,T1,T2}_guess`, `v_neg_frac_thick`.
  All are solver **diagnostics** not yet defined at the first step, so no physics
  depends on them. But they are written into `dictionary.jsonl` as bare `NaN`, which
  is **not valid JSON** — this is the same defect S13a raised for `Infinity` in
  `metadata.json`, now confirmed to occur in the main output too. Python round-trips
  it; `jq`, JS and R do not. **Corroborates S13a; widens its scope from metadata to
  `dictionary.jsonl`.**

## 3. Asymptotic limits — **PASS**

Uniform density (`densPL_alpha 0`) makes the Weaver similarity exponents the correct
expectation: `R ~ t^(3/5)`, `v ~ t^(-2/5)`.

| phase | qty | measured | ideal | rms [dex] | verdict |
|---|---|---:|---:|---:|---|
| energy | R2 | **+0.563** | +0.600 | **0.0018** | on the attractor |
| energy | v2 | **−0.429** | −0.400 | **0.0014** | on the attractor |
| implicit | R2 | +1.094 | +0.600 | **0.0897** | *not a power law — test does not apply* |
| implicit | v2 | +0.727 | −0.400 | **0.1085** | *not a power law — test does not apply* |

**Read the rms column before the exponents.** In the energy phase it is 0.0014–0.0018 dex:
the trajectory is a near-perfect power law, genuinely on an attractor, and a dropped
leading term would not look like this. In the implicit phase it is **50× larger**, so the
exponent there is not an attractor measurement and must not be quoted as one.

Why the implicit phase is not self-similar: `R2` expands **0.283 → 17.59 pc**, a factor of
62, over `t = 0.0035 → 0.5` Myr. The shell leaves the cloud during it, so the swept-mass
law changes and self-similarity is broken by construction. That is physics, not a defect
— but it means **this probe tests the energy phase only**.

> ⚠️ **Correction, recorded because it nearly became a false result.** An earlier draft of
> this file quoted the implicit phase as `R2 +0.557, rms 0.0007` — apparently a *better*
> fit than the energy phase. That came from a truncated run covering only the first 43
> implicit snapshots, early in the phase, where the trajectory still looked self-similar.
> Extending the same run to 74 snapshots moved the exponent to +1.094 and the rms by two
> orders of magnitude. **A self-similar exponent fitted over a window that is not
> self-similar produces a confident, meaningless number.** The rms column exists to catch
> exactly this, and it did.

The exponents sit ~0.037 below ideal, and that deficit was chased rather than assumed:

- **`L_mech` is constant** over the window (slope +0.0000), so declining luminosity is
  *not* the explanation.
- **Radiative loss is, partly.** `bubble_LTotal / L_mech` grows 0.035 → 0.255 across the
  window (slope +0.44). Feeding the measured `d ln(1−f)/d ln t = −0.0564` into
  `R ~ t^((3 + dln(1−f)/dlnt)/5)` predicts **+0.5887** against a measured **+0.5629**.
- **Residual: −0.026.** Unexplained by luminosity evolution or bubble cooling. That is a
  ~4 % effect on the exponent — the scale of the gravity and finite-shell-mass
  corrections the idealised Weaver solution omits, not of a missing leading term.

**Verdict: the energy-driven physics is clear.** The residual is a named follow-up, not
a finding.

### The strongest positive result: the code spans *both* analytic limits

Running the same fit on the stiff `f1edge_hidens` config (`mCloud 1e7`, `nCore 1e6`,
`sfe 0.01` — also uniform density, so the same expectation applies) gives a completely
different exponent, and the right one:

| config | `nCore` | `L_cool/L_mech` | measured | rms | analytic limit |
|---|---:|---|---:|---:|---|
| `simple_cluster` | 1e5 | 0.035 → 0.255 | **+0.563** | 0.0018 | adiabatic Weaver `3/5 = 0.600` |
| `f1edge_hidens` | 1e6 | 0.312 → **0.463** | **+0.287** | **0.0005** | pressure-driven snowplow `2/7 = 0.2857` |

At `nCore = 1e6` cooling removes ~46 % of the mechanical luminosity, and the shell sits
**on the pressure-driven snowplow exponent `2/7` to within 0.5 %**, with rms 0.0005 dex.
At `nCore = 1e5` cooling removes ~25 % and the shell sits between the two limits, near
the adiabatic one.

So the measured exponent moves monotonically from `3/5` toward `2/7` as the cooling
fraction rises, and lands on each analytic limit where that limit applies. **The energy
equation, the cooling coupling and the shell dynamics jointly reproduce two independent
closed-form solutions across a decade in density.** Nothing static in this audit could
have established that, and no plausible dropped or double-counted leading term survives
it.

Recorded as **P6-05**.

## 4. Table bounds (`TBL-01`) — **IN FLIGHT**

`stop_t = 15` Myr against a non-CIE cooling cube whose age grid ends at 1e7 yr. Sweep ⑦
showed `get_filename` clamps silently past that.

**Result: inconclusive — the run reached only `t = 0.0288` Myr of 15 before the container
restarted.** That is 0.2 % of the way to the 10 Myr limit the probe needs. `TBL-01` stays
open, and the honest note is that my affordability estimate below was not borne out in
practice: the geometric segment schedule bounds the *segment count*, but each segment in
the implicit phase carries a full bubble-structure solve, so wall-clock did not follow.

Affordability note for whoever picks this up: `phase1a_segFrac = 0.1` makes segments
**geometric** (`dt = 0.1·(t − tSF)`), so segment count scales with the *log* range, not
with `stop_t`. A 15 Myr run is far cheaper than linear extrapolation suggests.

---

## Findings this phase produced

| # | result | bears on |
|---|---|---|
| **P6-01** | `n_IF_Str == shell_n0` **bit-identical at every snapshot** | **Dynamically confirms `S8-R-02`**, one of the ten S1 candidates that had never been tested. The documented "sole source of `P_HII`" is numerically the same series as the quantity it is derived from. |
| **P6-02** | Bare `NaN` written into `dictionary.jsonl` at snapshot 0 (8 diagnostic keys) | Corroborates and **widens S13a** — the invalid-JSON defect is not confined to `metadata.json`. |
| **P6-03** | `Lmech_W == Lmech_total` and `pdot_W == pdot_total` bit-identical | Expected, not a defect: no SN contribution before ~3 Myr. Recorded so a future run past 3 Myr can check they *separate*, which is a free correctness test of the SN channel. |
| **P6-04** | Expansion exponent residual −0.026 after the radiative-loss correction | Open. Candidates: gravity, finite shell mass, radiation pressure. Small; chase only if a cheap decomposition exists. |

## Not done

- `TBL-01` past 10 Myr (in flight).
- `TBL-03` W-3 probe — needs a run that actually emits
  `"Bubble properties calculation failed"`; none of these runs did.
- Budget closure (do the force terms sum to the reported totals at every snapshot?).
- Momentum-phase asymptotics — no run here reached phase 2.
- **Re-measuring `ST-001`** against the new age-proportional segments, which invalidated
  its ~30 yr magnitude bound.
