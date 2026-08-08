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

## 4. Table bounds (`TBL-01`) — **MECHANISM SETTLED** (2026-08-08); frequency still open

`stop_t = 15` Myr against a non-CIE cooling cube whose age grid ends at 1e7 yr. Sweep ⑦
showed `get_filename` clamps silently past that.

**The earlier attempt was inconclusive** — the run reached only `t = 0.0288` Myr of 15
before the container restarted, 0.2 % of the way to the 10 Myr the probe was framed
around. The affordability estimate below was not borne out: `phase1a_segFrac = 0.1`
makes segments geometric so the *segment count* scales with the log range, but each
implicit-phase segment carries a full bubble-structure solve, so wall-clock did not
follow.

**Reframed and settled without the long run.** The probe was framed as needing
`t > 10` Myr, but only the *frequency* needs a run — the *mechanism* is a pure function
of the shipped table set. `harness/phase6_tbl01_w3.py` reads the bundled grid and calls
`get_filename` directly (`data/phase6_tbl01_w3.csv`):

```
age grid [yr]: 1.00e+06  2.00e+06  3.00e+06  4.00e+06  5.00e+06  1.00e+07
grid max     : 1.000e+07 yr = 10 Myr        default stop_t: 15 Myr  (registry.py:378)

  age=5.000e+06 yr ->          opiate_cooling_rot_Z1.00_age5.00e+06.dat
  age=9.900e+06 yr ->          interp['_age5.00e+06', '_age1.00e+07']
  age=1.000e+07 yr ->          opiate_cooling_rot_Z1.00_age1.00e+07.dat
  age=1.010e+07 yr -> CLAMPED  opiate_cooling_rot_Z1.00_age1.00e+07.dat
  age=1.500e+07 yr -> CLAMPED  opiate_cooling_rot_Z1.00_age1.00e+07.dat
  age=5.000e+07 yr -> CLAMPED  opiate_cooling_rot_Z1.00_age1.00e+07.dat
```

**CONFIRMED.** `read_cloudy.py:325-329` is `elif age >= max(age_list):` → return the
last-grid file. No warning, no exception, no record anywhere in the run. On a
default-length run the clamp covers **t = 10…15 Myr — the last 33 % — with the non-CIE
cooling frozen at the 10 Myr table**, while the SPS feedback it is paired with keeps
evolving (the SPS table runs to 99.91 Myr).

Note also the grid is **sparse at the top**: 1, 2, 3, 4, 5, then a jump straight to
10 Myr. Everything in 5–10 Myr is a two-point interpolation across a 5 Myr gap.

⚠️ **Still open — frequency, not mechanism.** No run in this audit has survived past
10 Myr, so how often real runs enter the clamped window is **unmeasured**. The
mechanism above is config-independent and needs no further run; the frequency needs
one that survives to 10 Myr.

---

## 5. Budget closure — **PASS on the decompositions; two real failures found**

`harness/phase6_budget.py` over three runs (536 snapshots total). No new run needed —
every quantity was already in `dictionary.jsonl`. This was the cheapest probe of the six
and returned the most.

**Closes exactly, 0 violations across all 536 snapshots:**

| identity | result |
|---|---|
| `bubble_LTotal == L1 + L2 + L3` | exact (rel 0.000e+00) |
| `Lmech_total == Lmech_W + Lmech_SN` | exact |
| `pdot_total == pdot_W + pdot_SN` | exact |

**No dropped or double-counted term in the luminosity or momentum-source budget.** That
is the defect class this probe exists for, and it is clear.

### P6-06 — `F_ram` and `P_ram` are mutually inconsistent in the output — **100 % of rows**

`F_ram == P_ram · 4πR2²` fails at **every snapshot of every run** (171/171, 126/126,
239/239). `P_ram` is recorded as **exactly zero** throughout the energy and implicit
phases while `F_ram` is non-zero and grows from 0.45× to 17× `pdot_total`. This is not
staleness — it is systematic.

A consumer reading the published force budget would take `F_ram` as the ram-pressure
force and get a number bearing no relation to the `P_ram` printed beside it.
**Dynamically corroborates sweep ② SIGN-04** ("`F_ram` means two different things by
phase"). By contrast `F_ram_wind == pdot_total` holds — so the momentum *source* is
recorded correctly; it is the `F_ram` column that is untrustworthy.

### P6-07 — the phase-1a reconciliation snapshot breaks `F_HII`'s declared invariant

> ⚠️ **Correction.** An earlier version of this entry claimed P6-07 "dynamically
> confirms ST-001". **That was wrong and is withdrawn.** ST-001 fires on the
> event-`break` path, where the loop exits before the locals update at
> `run_energy_phase.py:345`. All three runs here exit phase 1a on the *time*
> condition, where the locals **are** updated. The mismatch is real, but its cause
> is a different defect — traced below. ST-001 remains **not** dynamically
> confirmed; it needs a run that exits via `cloud_boundary`, and none of these do.

`F_HII == P_HII · 4πR2²` fails at **exactly the last energy-phase row in all three
runs, and nowhere else**:

| run | energy ends | rel error there | violations elsewhere |
|---|---:|---:|---:|
| `phase6_detA` | row 96 of 171 | **3.82e-02** | 0 |
| `f1edge_lowdens` | row 93 of 239 | **1.14e-02** | 0 |
| `f1edge_hidens` | row 97 of 126 | **6.37e-03** | 0 |

**Mechanism, confirmed at source.** `run_energy_phase.py:228-229` computes
`F_HII = 4πR2²·P_HII` *inside the segment loop*. The phase-boundary reconciliation
block (`:394-407`) then recomputes SPS feedback, `R1`, `Pb`, `shell_mass` and the
whole shell structure — including a fresh `P_HII` — and calls `save_snapshot()`
**without re-deriving `F_HII`**. `shell_structure_pure` does not produce any `F_*`
key (`grep F_HII trinity/shell_structure/` is empty), so nothing else can refresh it.

The final energy-phase snapshot therefore pairs a **freshly recomputed `P_HII`** with
an **`F_HII` from the previous segment**.

**It breaks a contract the code itself declares.** `registry.py:509` defines `F_HII`
as *"Outward HII pressure force (= P_HII * 4piR2^2)"*. That identity holds at every
other snapshot of every run and fails at this one.

**Reach is wider than ST-001's**, which is why the mix-up mattered: the
reconciliation block is **unconditional** — it runs after the loop on *every*
phase-1a exit path, not only the event-`break` one. So this occurs in **every run**,
whereas ST-001 needs a `cloud_boundary` exit.

**Severity S2**, on the same reasoning as ST-001: the trajectory is unaffected
(phase 1b recomputes from `params` before integrating); what is wrong is one
published output row, by 0.6-3.8 %, in the force column an analysis would read.

### P6-08 — the final row of a run is also inconsistent

The worst `F_ram_wind == pdot_total` violation in each run is at the **final row**:
row 237/239 in `f1edge_lowdens` at **1.02e-01 (10 %)**, row 125/126 in `f1edge_hidens`
at 1.66e-05. Elsewhere the identity holds to ~1e-9.

This is the **S5b** family — "the final row mixes two times". Worth recording carefully:
S5b proposed testing it via `F_ion_in != press_HII_in · 4πR2²`, and **that test passes
everywhere** (0/171 violations, including the final row, at rel 0.0). The claim family is
real; the repro S5b proposed does not demonstrate it. A different identity does.

## 6. `ST-001` re-measured against the age-proportional schedule — **bound grew 10x, rating survives**

The merged hotfix replaced the fixed `SEGMENT_DURATION` with
`dt = phase1a_segFrac * (t - tSF)` (`phase1a_segFrac = 0.1`). `ST-001`'s S2 rating
rested partly on its staleness window being "one partial segment, ~30 yr", a number
that assumed the old constant. Re-measured on the post-hotfix run:

| | old schedule | new schedule |
|---|---|---|
| window | fixed **3e-5 Myr = 30 yr**, everywhere | `0.1·(t − tSF)`, **grows with t** |
| measured last phase-1a segment | 3e-5 (both pre-fix `f1edge` runs) | **1.01e-4 Myr = 101 yr** (`phase6_detA`) |
| growth across phase 1a | none — constant | **2976x** (3.39e-8 → 1.01e-4 Myr) |
| worst case | 30 yr | **300 yr** |

The worst case is bounded, and that is the point: `ST-001` is a *phase-1a* defect, and
phase 1a is itself capped at `TFINAL_ENERGY_PHASE = 3e-3` Myr
(`run_energy_phase.py:54`). So the window cannot exceed `0.1 × 3e-3 = 3e-4` Myr =
**300 yr — 10x the old bound, not unbounded**.

**Verdict: the S2 rating survives.** The magnitude claim in the original finding is
stale by an order of magnitude and has been corrected, but the reasoning that demoted
it — the trajectory is unaffected because phase 1b recomputes from `params`, and only
one output row is wrong — is untouched by the schedule change.

Worth flagging for anyone who revisits the segment schedule: the 2976x growth across
a single phase means *any* per-segment freezing defect now has a strongly
time-dependent magnitude. A defect measured early in phase 1a will understate itself
near the phase end by up to three orders of magnitude.

## 7. In-process determinism — **PASS**, and the id-reuse hazard is not demonstrable

CLAUDE.md states trinity leaks module-level global state in-process, which is why every
baseline here is launched as its own process. That is a documented hazard nothing had
measured. `harness/phase6_inprocess.py` runs the same config **twice inside one
interpreter** and diffs the outputs.

```
run A: 171 snapshots
run B: 171 snapshots
RESULT: identical — no in-process state leak on this path
```

**The probe is valid, and I checked that before trusting it.** `dictionary.py:857` opens
the output in **append** mode when the file exists, which would have made this test
meaningless — run B would simply have appended to A's rows. It does not, because
`:809-812` deletes the existing `dictionary.jsonl` when `flush_count == 0`, and
`flush_count` is per-`DescribedDict`, so each `read_param` starts a genuinely fresh
file. Both copies are byte-identical at 3 864 080 bytes.

**Scope, stated because it bounds the claim.** This runs the *same* config twice. The
sharper documented hazard is a **different** config second, where a module-level cache
keyed on the first config's data could serve stale values. That case is still untested.

### `ST-003`'s address-reuse claim — not demonstrable

Sweep ⑧ rated `_CIE_TCUTOFF_CACHE` (`net_coolingcurve.py:27`) **S2**: keyed by
`id(logT_CIE)`, "unbounded, never invalidated, and vulnerable to address reuse across
runs". The three sub-claims separate cleanly under test:

| sub-claim | verdict |
|---|---|
| keyed by `id()` | **true** — `_cie_tcutoff` uses `key = id(logT_CIE)` verbatim |
| never invalidated / unbounded | **true, trivially** — nothing anywhere clears the dict |
| vulnerable to address reuse | **not demonstrable** — 200 000 rebuild-and-realloc cycles with matched dtype and shape produced **zero** id collisions |

So the hazard is real in principle and I could not make it fire. The code's own comment
claims immunity for a *different* reason — "logT_CIE is built once at startup and never
replaced, so its id is stable for the whole run" — which holds within one run but is
exactly what an in-process second run would break. Since §7 above shows a second
in-process run reproduces bit-identically, that path is clear too.

**Recommend `ST-003` S2 → S4**: an `id()`-keyed unbounded cache is a real hygiene defect
worth fixing (cache on the object, as the sibling `_noncie_cutoffs` already does), but
the correctness hazard it was rated for is not reachable by any route tested here.

## Findings this phase produced

| # | result | bears on |
|---|---|---|
| **P6-01** | `n_IF_Str == shell_n0` **bit-identical at every snapshot** | **Dynamically confirms `S8-R-02`**, one of the ten S1 candidates that had never been tested. The documented "sole source of `P_HII`" is numerically the same series as the quantity it is derived from. |
| **P6-02** | Bare `NaN` written into `dictionary.jsonl` at snapshot 0 (8 diagnostic keys) | Corroborates and **widens S13a** — the invalid-JSON defect is not confined to `metadata.json`. |
| **P6-03** | `Lmech_W == Lmech_total` and `pdot_W == pdot_total` bit-identical | Expected, not a defect: no SN contribution before ~3 Myr. Recorded so a future run past 3 Myr can check they *separate*, which is a free correctness test of the SN channel. |
| **P6-04** | Expansion exponent residual −0.026 after the radiative-loss correction | Open. Candidates: gravity, finite shell mass, radiation pressure. Small; chase only if a cheap decomposition exists. |

## 8. `W-3` — what survives a swallowed bubble-properties failure — **SETTLED** (2026-08-08)

Sweep ⑦ framed `W-3`/`TBL-03` as "grep the WARNING stream for swallowed bounds errors",
which needs a run that actually emits `"Bubble properties calculation failed"`. **None
does** — grepping every log this audit has produced, including the complete 155-snapshot
`probe_iscollapse_maxr` run, returns zero occurrences, and zero WARNING records of any
kind.

That is a frequency result, not a mechanism result, so the mechanism was tested directly
(`harness/phase6_tbl01_w3.py`) by making `get_bubbleproperties_pure` raise:

```
  returned            : (100.0, 100.0, None)
  exception propagated: no
  log records emitted : ['WARNING']
    WARNING: Bubble properties calculation failed: _Boom: simulated cooling-table bounds error
```

**CONFIRMED, and it bears directly on `SF-003`** (one of the untested S1 candidates —
same code site). `get_betadelta.py:437-439` and `:538-548` are bare `except Exception`
handlers that convert *any* failure into the constant `(100.0, 100.0)` residual plateau.
The **only** trace is a `WARNING` line in `trinity.log`: nothing reaches
`dictionary.jsonl`, `metadata.json`, `SimulationEndReason`, `SimulationEndCode`, or the
process exit code. A run whose bubble solve failed on every call is externally
indistinguishable from a clean one for any consumer that does not read the log text.

**Interaction with `TBL-01` worth stating:** the two silent paths are *different*. An
out-of-range cooling **age** never reaches this handler at all, because §4 shows
`get_filename` clamps rather than raising — so that failure mode produces silently wrong
physics with **not even a warning**.

## Not done

- `TBL-01` **frequency** past 10 Myr — mechanism settled in §4, but no run has survived
  that long.
- Budget closure (do the force terms sum to the reported totals at every snapshot?).
- **Re-measuring `ST-001`** against the new age-proportional segments, which invalidated
  its ~30 yr magnitude bound. *(Done — see §6.)*
