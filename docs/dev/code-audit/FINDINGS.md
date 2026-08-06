# code-audit — findings

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

**Status (2026-07-30):** 🔵 ACTIVE — Phases 0–4 complete, Phase 5 partial (7 of 17
S1-class defects through the gate), Phase 6 barely started. **This is an interim
report, not the final verdict.** Regenerate the counts with
`harness/collect_findings.py`; check phase state with `harness/check_completeness.py`.

## How to read this

689 raw findings were collected from 26 reports into
[`data/findings_inventory.csv`](data/findings_inventory.csv). Raw counts are not
the story: **the audit's own verification removed or demoted more S1s than it
confirmed.** Only findings that cleared a gate appear below as confirmed.

Three tiers of evidence are used, and each finding says which it has:

- **gate** — passed the Phase-5 skeptic panel: three fresh agents, each given the
  *claim only* (never the finder's reasoning), each instructed to refute, with
  "refuted" as the default under uncertainty. Majority required.
- **source** — the orchestrator verified the mechanism directly at the source and
  reproduced it, but no skeptic panel was run.
- **unverified** — a reconciler or sweep raised it; nothing has tried to kill it.
  **Treat these as candidates, not defects.**

Every severity here is the **current** one, after revision. Birth severities and
the full revision history are in [`data/revisions.csv`](data/revisions.csv) and
[`data/resolutions.md`](data/resolutions.md).

| current severity | count |
|---|---:|
| FIXED (on main) | 2 |
| S1 | 16 |
| S2 | 196 |
| S3 | 250 |
| S4 | 219 |
| CLEARED / REFUTED / WITHDRAWN | 6 |

## Coverage — what this report does *not* cover

Stated first, because it bounds everything after it.

- **7 of 17** distinct S1-class defects went through the skeptic gate. Ten did not.
- **0 of 196** S2s were gate-tested. The method calls for it; at the measured
  ~150k tokens per skeptic that is ~88M tokens, which is not proportionate.
- Phase 6 (dynamic verification) produced one partial artifact. The table-bounds
  sensor, determinism probe and asymptotic-limit fits are **not done**.
- The 24 raw S1 mentions de-duplicate to **17 distinct defects**
  ([`data/s1_clusters.md`](data/s1_clusters.md)); four were found independently by
  two to four separate passes.

---

## FIXED on main

### `vd = -1e8` overrode the momentum equation on every run — *gate, now fixed*

**Was:** `energy_phase_ODEs.py` replaced the entire momentum right-hand side with a
hardcoded constant for the first energy-phase segment, on every run.

Fixed by `hotfix/early-approximations`, merged into `main` 2026-07-30: the branch
is deleted, the `EarlyPhaseApproximation` field removed from `ODESnapshot`, its
`ParamSpec` deleted from `registry.py`, and the clear site removed from
`run_energy_phase.py`. `git grep EarlyPhaseApproximation origin/main -- trinity/`
returns zero.

**Kept here because the measurement matters for what comes next.** Separate-process
A/B at matched `t` gave `ΔR2 −30.2 %` at peak and **−19.3 % at phase-1a exit**,
`ΔEb −16.5 %`, and a similarity exponent handed to phase 1b **39.4 %** apart. It
was universal — every run on the bundled SB99 table left segment 0 at the identical
velocity regardless of mass, SFE or density — and **invisible to convergence
testing**: `v2_end` moved 2e-12 across a 4000× change in step count.

> ✅ **Goldens: already re-baselined by the same hotfix** (`0ffa994`, "re-baseline
> the three goldens on the phase-1a exit state"). `test_run_smoke._FINAL_GOLDENS`
> moved `R2` 0.28573 → 0.25956, `v2` 44.739 → 49.226, `Eb` 778236 → 662534.
>
> **That shift independently corroborates this audit's measurement.** The
> separate-process A/B here predicted `R2` down, `v2` up, `Eb` down at phase-1a
> exit (−19.3 % / +12.4 % / −16.5 %); the maintainer's re-baseline moved the same
> three quantities in the same three directions (−9.2 % / +10.0 % / −14.9 %). The
> magnitudes differ because the smoke test stops at `1e-4` Myr rather than at
> phase-1a exit. Two independent routes, same signs.
>
> **Correction to an earlier draft of this file.** It warned that "~105 captured
> goldens are now wrong". They are not — the full suite on merged main is
> **1093 passed, 15 deselected, 0 failed**. Only the three end-to-end goldens
> needed re-baselining and the hotfix did them. The warning was over-stated and is
> withdrawn.
>
> The narrower caution stands: ~105 expectations are still *captured from this
> code's own output* rather than independently derived, so they defend whatever it
> did on the day. Where an analytic check exists — the Weaver similarity solution,
> an asymptotic exponent — prefer it. That is a standing property of the suite,
> not a consequence of this merge.

---

## S1 — confirmed

### 1. A terminal event loses to a monitoring event — *source*

`trinity/phase_general/phase_events.py:392`

`check_event_termination` returns the **first event by list index** and never reads
`event.terminal`. In `build_implicit_phase_events` (`:487`) the list is
`[velocity_sign, min_radius, velocity_runaway, (max_radius)]` — and `velocity_sign`
is documented at `:463` as *"monitoring, non-terminal"*, while indices 1–3 are all
simulation-ending. The runner then `break`s on `triggered` regardless
(`run_energy_implicit_phase.py:1096`).

So whenever `velocity_sign` has fired and a terminal event has *also* fired, the
monitoring event is reported, `is_simulation_ending=False`, and the real fate is
discarded — while the run stops anyway. The collapse detector at
`run_energy_implicit_phase.py:1301` sits after the break and is **unreachable for
the case it was written for**.

**Repro** (demonstrated standalone on scipy 1.17.1): with both events firing, a
terminal `min_radius` genuinely stopped the solve at `R2 = 1.5` pc, but the loop
reported `velocity_sign` at `t = 0.25` with state rewound to `R2 = 3.125`,
`EndSimulationDirectly`/`isCollapse` unset, and the run continuing into phase 1c.

**Fix outline:** select among triggered events by earliest root time, and prefer
`terminal` when roots tie; do not return on first index.

**Corroboration:** found independently by **four** passes — the S11 reconciler
(`S11-R-01`), the duplicate-divergence sweep (`DD-001`), the state sweep
(`ST-002`), and the numerical sweep (`NUM-02`). That is the strongest agreement
this audit produced.

### 2. User-set `mu_*` is silently discarded — *source*

`trinity/_input/read_param.py:316-319`

`default.param:205-217` advertises `mu_atom`, `mu_ion`, `mu_mol` and `mu_convert`
as ordinary editable parameters with concrete values. Step 6 then overwrites all
four **in place** from `x_He`/`Z_He`, with no test of whether the user set them and
no warning. The anti-stomp guard compares **object identity**, so it structurally
cannot see a `.value` mutation. `mu_convert` is the `n_H → ρ` conversion used in
every zone.

**Repro:** set `mu_convert 1.6` in a `.param`; the run integrates at 1.4 and the
snapshot records the derived value, so the discard is invisible afterwards.

**Fix outline:** reject or warn on a user-set `mu_*`, or stop listing them as
inputs. Any of the three closes it.

**Mitigation, stated because it is real:** each key's `INFO:` line does say
"Derived at load from x_He". Documented — but still silently discarded.

### 3. Sweeps silently drop requested configurations — *gate, 3/3*

`trinity/_input/sweep_parser.py:742` · `trinity/_input/sweep_jobs.py:175-188`

`generate_run_name` is not injective, and `emit_jobs` writes `params/<name>.param`
with no duplicate check. A colliding combination's file is overwritten *before
submission*, two array tasks run the identical config into one output directory,
and the report still counts every requested job as a success.

**Repro** (against the real code):

```
densPL_alpha [-2.0, -1.5, -1.0, -0.5, 0.0]  ->  3 distinct names for 5 combinations
  -1.5 -> 1e6_sfe003_n1e3_PL-1  ┐ collide (int() truncates toward zero)
  -1.0 -> 1e6_sfe003_n1e3_PL-1  ┘
```

Worse than first claimed: the manifest *misreports* — `runs[0].params.densPL_alpha`
reads `-1.0` while the file task 1 executes says `-1.5`, so `failure_breakdown`
attributes failures to a config that never ran. And sweeping `densPL_alpha` without
`dens_profile` in the same file drops the `_PL` suffix entirely, collapsing **all**
alphas to one name.

**Scope — the fix must target the right place.** Only `densPL_alpha` and
`densBE_Omega` collide at realistic step sizes. `mCloud`/`nCore` need ~40 points per
decade; `sfe` is injective for every integer-percent sweep. Both colliding keys sit
in `_NAMED_RUN_NAME_KEYS` (`sweep_parser.py:590`), which **suppresses** the generic
disambiguating suffix — the defect is created by the curated special case, not by a
missing feature.

**Fix outline:** drop `densPL_alpha` and `densBE_Omega` from
`_NAMED_RUN_NAME_KEYS` (the generic fallback then yields `densPLAlpha-1p5`), and add
a uniqueness assertion in `emit_jobs` before writing.

**Bound:** **no committed sweep triggers this.** All 10,582 combinations across the
nine tracked `param/` configs and 1,318 across tracked `docs/dev/` configs are
collision-free. It threatens *future* α or Ω sweeps.

---

## S1 — raised but never gate-tested

Ten defects carry an S1 rating from a reconciler or sweep and **have not been
through the skeptic panel**. On the evidence of the seven that were tested — where
two S1s were removed entirely and three demoted — expect a meaningful fraction of
these to shrink. Do not act on them without verification.

| id | claim | file |
|---|---|---|
| `S11-R-02` | `isCollapse` set by substring match: `velocity_runaway_event` matches nothing, so runaway infall is never recorded as collapse | `phase_events.py:627` |
| `S11-R-03` | no solver-failure channel in the slice — **note: the "`sol.status` never read" half is already refuted**; every phase runner checks it. The exit-code propagation half stands | `main.py:211` |
| `S5b-R-01` | a `solve_ivp` failure ends the phase with only a free-text reason, no `SimulationEndCode` | `run_energy_implicit_phase.py` |
| `S6-R-01` | transition phase keeps evaluating `R1`/`Pb` past the energy→momentum boundary | `run_transition_phase.py` |
| `S8-R-02` | `n_IF_Str`, documented as the sole source of `P_HII`, is identically `shell_n0`, itself back-solved from it | `shell_structure.py:251` |
| `DD-003` | momentum ODE freezes shell mass and `dM/dt` per segment; the other phases recompute | `run_momentum_phase.py` |
| `DD-004` | phase 1a reconciles after an `Eb ≤ 0` collapse; phase 1b deliberately skips it | `run_energy_phase.py` |
| `SF-003` | `get_residual_pure` swallows every exception into a fixed `(100, 100)` plateau | `get_betadelta.py:437` |
| `SF-004` | a failed `solve_ivp` in 1b/1c/2 sets only a local string; `main.py` drops it | `main.py` |
| `SF-005` | momentum RHS clamps `R2`/`mShell` to `1e-10`, fabricating outward acceleration during collapse | `run_momentum_phase.py` |

`S11-R-02` is the one I would test first: `isCollapse` is consumed by
`paper/_lib/plot_markers.py`, so a mis-classification reaches published figures.

---

## S2 — the ones worth knowing about

Selected from 196; each is gate-tested or source-verified.

**`P_HII` does unpaid work on the shell, via a pressure inconsistency** — *gate,
corrected*. `energy_phase_ODEs.py:258` drives at `max(press_bubble, P_HII)` while
`:280` debits PdV only at `press_bubble`. The `max` was designed as a no-op — the
`n_IF_Str ≤ shell_n0` cap makes `P_HII ≤ params['Pb']` identically — but the ODE
compares against the **ramped** `press_bubble`, not the `params['Pb']` the cap used.
Measured ratio up to **2.91**, `max()` selecting `P_HII` at **100 %** of accepted
phase-1a states. Cost: 0.80 % of injected energy, `ΔR2` 0.25 %, `Δv2` 0.81 %.
**The energy equation is the canonical Weaver/WARPFIELD/Lancaster form and needs no
new term** — the fix is to make the ODE and the cap agree on `Pb`. Note the
hotfix's own E8b result: *ablating the R1 ramp makes the stiffest config
intractable*, so the ramp is load-bearing and the cap must move, not the ramp.

**`fsolve` convergence is never checked** — *gate, split*. `bubble_luminosity.py:261`
calls `fsolve` without `full_output`, so `ier` is unavailable. A **constant**
residual returns the seed bit-for-bit, and `_get_velocity_residuals` returns literal
constants at four sites. Worse, a penalty plateau beside a steep region returns
`ier=1 "converged"` with `fvec=[1000.]` and no warning — **so an `ier`-only fix does
not close it; the guard must be a residual re-check.** Demoted from S1 because 600
production calls across two configs gave 0 non-convergences, with a ~230× basin
margin. Blast radius if reached: `dMdt` 2333× wrong, returned normally.

**Phase-1a's reconciliation snapshot mixes two states** — *gate, split*. On the
`cloud_boundary` exit the locals are never advanced, so `:391-398` solve the
phase-boundary shell using the **event** `R2` against **pre-event** `Pb`,
`shell_mass`, `Qi` — a state that never existed on the trajectory. Phase 1b's
correct row at the same `(t, R2)` is then suppressed by the duplicate guard. The
trajectory is unaffected (1b recomputes everything before integrating); one wrong
**output row** survives, and under `stop_at_rCloud_nSnap == 0` it is the final row
of the run. ⚠️ Its magnitude bound (~30 yr) assumed the old fixed `SEGMENT_DURATION`
— the merged hotfix makes segments age-proportional, so **this must be re-measured**.

**Non-CIE cooling interpolated two ways** — *sweep ⑦*. The same quantity is
interpolated linearly on the bubble-ODE hot loop and in log-space in
`L_conduction`, from the same cube in the same function. Median **1.63×**
disagreement, up to 57×, 0.2 % sign flips.

**~21 % of the non-CIE cooling cube is NaN** — *sweep ⑦*. `RegularGridInterpolator`
propagates it (`0 × NaN = NaN`), so 22.9 % of cells return NaN including queries
landing on a valid node beside a hole. `grep isnan trinity/cooling/` finds nothing.

**`gamma_adia` is honoured in two places and hardcoded `5/3` everywhere else** —
*sweep ②*. `bubble_E2P` and `get_leak_luminosity` accept it; `get_r1`, the
Rahner-A12 pair and the whole Weaver structure chain hardcode it. At γ=1.4 that is a
**67 %** pressure imbalance at the contact discontinuity. `gamma_adia` is
user-settable (`default.param:251`).
⚠️ **Severity is unresolved and must be settled before this ships as S2.** It has
exactly the shape that justified rating `mu_*` as S1 — schema-accepted, no warning,
different value drives the physics. The two must end up rated alike.

**`--z-override` bypasses the metallicity validator** — *orchestrator*.
`_validate_ZCloud` raises unless `Z == 1`, but `trinity_to_cloudy.py:140`'s
`--z-override` is consumed at `snapshot_to_deck.py:206` ahead of the validated
value, emitting a non-solar CLOUDY deck over a trajectory integrated at solar. No
warning.

---

## Removed by the audit's own gate — do **not** "fix" these

Recorded as prominently as the defects, because acting on them would break working
code. Full evidence in [`data/resolutions.md`](data/resolutions.md).

| claim | verdict | why it matters |
|---|---|---|
| `pdot` missing a factor of 2 in the momentum phase (`S6-R-02`, born S1) | **REFUTED 3/3** | The 2 is present — it cancels one of the 4 in `4π`, leaving `2π`, and `v_mech_total` is *defined* as `2L/pdot`, so `4πr²·pRam ≡ pdot_total` identically (measured ratio 1.00000000). **The proposed fix would have doubled the ram pressure.** |
| `odeint` reads uninitialised memory as physics (`S8-R-01`/`SF-001`, born S1) | **REFUTED 3/3** | Mechanism real and demonstrated (2697/3000 sentinel poison test), but `shell_structure.py:181-188` truncates at the first `phi ≤ 1e-9` row, which across 416 induced failures was **always** before the failure row — 0/416 garbage reads, outputs bit-identical. Residual **S4**: add `full_output` + istate as defence in depth. |
| CIE cooling density factor wrong by up to 5.29× (`S9-R-01`, born S1) | **CLEARED** | `ndens` is n_H and `chi_e` is n_e/n_H, so the branch correctly computes `n_e n_H Λ`. A lens inference was wrong. |
| non-CIE cube short by n² (`S9-R-02`, born S1) | **CLEARED (code)** | The cube is volumetric — measured slope **2.014** over 14 decades. Only the docstring is wrong (**S3**), and it is the kind of docstring that would license a future "fix" that breaks the code. |
| CLOUDY `dlaw` density missing a composition factor (`S13b-R-01`) | **CLEARED** | `shell_n_arr` is already n_H, so the pure pc⁻³→cm⁻³ shift is correct. **No factor should be added.** |
| `ZREL` wrong by ~1.85 dex (`S13b-R-02`) | **CLEARED** | `ZCloud` is declared `unit='Zsun'` and the template consumes it as CLOUDY's linear solar-relative scale; the validator pins `Z == 1` anyway. |

---

## The test suite catches none of it

Phase 4 measured the suite at **851 passed, 9 deselected** and checked it against
six confirmed defects: **0 of 6 were caught**. See
[`slices/test_suite_audit.md`](slices/test_suite_audit.md).

> 🔄 **Updated after the hotfix merge (2026-07-30): now 1 of 6, and the suite is
> 1093 passed / 15 deselected.** `test/test_early_phase_override.py` arrived with
> the fix and closes the `vd` defect properly — four tests that **pin the property,
> not the formula**, so they survive a re-tuning rather than re-blessing an output.
> That is the pattern the rest of the suite's ~105 captured goldens lack, and it is
> worth copying. Its docstring also independently derives the same closed form this
> audit measured (`v_exit = v0 − 1e8·SEGMENT_DURATION = 739.2407` pc/Myr,
> mass-scale invariant) — a third independent arrival at that number.
>
> The other five defects remain uncaught.

- The only test of `check_event_termination` builds `t_events=[array([]), array([0.25])]`.
  Index 0 is *empty*, so `assert result.index == 1` reads as terminality-awareness
  while proving only that the loop skips empty entries.
- `generate_run_name` has **zero enforced coverage** — its 20-case suite sits in
  `sweep_parser.py`'s `__main__` block, which pytest never collects and which
  `print`s `"FAIL"` rather than raising. Eight other modules share the pattern.
- The `mu_*` anti-stomp guard is structurally unable to fire, and is itself untested.
- Provenance: ~55 independent goldens vs **~105 captured** from prior runs.
- 19 `trinity/` modules are imported by no test, including all three phase drivers
  that call the defective dispatcher.

---

## What is not done

1. **Ten S1-class candidates unverified** (table above). Highest value: `S11-R-02`.
2. **Phase 6 dynamic verification** — table-bounds sensor (needs `t > 10` Myr),
   determinism probe, asymptotic-limit fits, budget closure. Sweep ⑦ left a named
   7-probe worklist.
3. **`UNVERIFIED.md`** — the demoted-candidate register the plan calls for is not
   written; the material is currently spread across `resolutions.md`.
4. **The `gamma_adia` vs `mu_*` severity question** — must be settled once, applied
   to both.
5. **Re-measure `ST-001`** against the merged age-proportional segment schedule.
6. **Regenerate the ~105 captured goldens** against the merged hotfix — checking
   them against physics, not re-recording the code's output.
