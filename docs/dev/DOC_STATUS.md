# docs/dev — document status (verified against code)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living index — recheck and refine on every visit.** Re-run the verdicts
> below against current source when you touch the relevant code; move SHIPPED /
> SUPERSEDED docs to `archive/` once their open tails are extracted.
>
> 💾 **Persist diagnostics — commit, don't re-run.** The verification evidence
> (per-doc `trinity/…:line` citations) lives in this file so a future visit need
> not re-derive it.

**Verified:** 2026-06-16 · **Method:** 10 parallel read-only agents, each checking one or
two docs **line-by-line against current `trinity/` source** (HEAD of `fix/code-hygiene`),
cross-checked with `CHANGELOG.md` and `git log`.

### Legend
- ✅ **SHIPPED** — the planned work landed; the doc is now a historical record → **archive candidate**.
- ⛔ **SUPERSEDED** — replaced by a newer doc / a different approach that shipped.
- 🔵 **ACTIONABLE** — the described work has **not** shipped; still relevant/pending.
- 🟡 **PARTIAL** — some shipped, some still open.
- 📘 **REFERENCE** — a current, accurate reference (not a plan); keep as-is.
- ⚠️ **STALE-REFS** — cited paths/symbols/line-numbers have drifted (fix before relying).

## Summary

| Doc | Verdict | Stale refs | Recommendation |
|-----|---------|:---------:|----------------|
| `archive/betadelta/HYBR_PLAN.md` | ✅ SHIPPED (Phases 0–3) + live tail | ⚠️ | Archive after extracting open tail (Phase-4 default flip; Phase-5 study) |
| `archive/betadelta/PHASE0_BASELINES.md` | ✅ SHIPPED (record) | — | **Archive** |
| `archive/betadelta/PHASE2_ARMS.md` | ✅ SHIPPED (arm D / hybr landed) | — | **Archive** (forward item lives in stalling + HYBR Phase 5) |
| `archive/betadelta/stalling-energy-phase.md` | ✅ SHIPPED (settled study) | minor | Fix 1 parenthetical → **archive** |
| `transition/TRIGGER_PLAN.md` | 🔵 ACTIONABLE | accurate | **Keep** (P-shadow unbuilt) |
| `transition/P0.md` | ✅ SHIPPED (results record) | — | Keep alongside the plan (don't archive yet) |
| `transition/pshadow-design.md` | 🔵 ACTIONABLE | ⚠️ (1: `stop_r`) | Fix ref → **keep** (actionable head) |
| `archive/bubble/integrator-robustness.md` | ⛔ SUPERSEDED (by the `solve_ivp` migration) | ⚠️ (heavy) | Add "superseded" banner → **archive** |
| `archive/bubble/conduction-convergence.md` | ✅ SHIPPED (sign-off; switch landed) | minor | **Archive** |
| `cooling/refactor-audit.md` | 🔵 ACTIONABLE (nothing shipped) | ⚠️ (~1–3 lines) | Refresh refs → **keep** |
| `performance/HOTPATH_PLAN.md` | 🟡 PARTIAL (F1 + F2 SHIPPED; F1-cousin + F5 open) | accurate (fresh) | **Keep** |
| `performance/BUBBLE_LUMINOSITY_PERFORMANCE.md` | 📘 REFERENCE (consolidated bubble-perf history A→D + methodology) | fresh (2026-06-19) | **Keep** (canonical) |
| `performance/F1_SUMMARY.md` + `F1_REPORT.html` | 📘 REFERENCE (F1 tables + illustrated report) | fresh | **Keep** |
| `archive/bubble/RESAMPLE_PLAN.md` + `P3_PRODUCTION_PATCH.md` | ✅ SHIPPED (F1, `24c6914`, 2026-06-19) | superseded banner added | **Archived** |
| `archive/n-consistency/audit.md` | ✅ SHIPPED (pinned by `test_mu_audit_drift`) | by-design | **Archive** |
| `archive/n-consistency/implementation-plan.md` | ✅ SHIPPED (all phases landed) | ⚠️ (flat→subdir) | **Archive** |
| `archive/n-consistency/pressure-terms-audit.md` | ⛔ SUPERSEDED (self-declared, confirmed) | historical | **Archive** |
| `misc/backward-compat-audit.md` | 🔵 ACTIONABLE (~95% un-executed) | ⚠️ (heavy) | Refresh refs, mark Tier-4 done → **keep** |
| `misc/tinit-sensitivity.md` | 🟡 PARTIAL (study done; rec #3 open) | — | **Keep** (one open physics rec) |
| `misc/TERMINATION_EVENTS.md` | 📘 REFERENCE (accurate) | — | **Keep** (current reference) |
| `misc/LEAKING_LUMINOSITIES_SKELETON.md` | 🟡 PARTIAL (A–C shipped; D/F/G open) | — | **Keep** |

**Tally:** 7 ✅ shipped · 2 ⛔ superseded · 5 🔵 actionable · 2 🟡 partial · 1 📘 reference · (1 results-record). *(+1 actionable: `performance/HOTPATH_PLAN.md`, added 2026-06-18 — not part of the 2026-06-16 verification pass; its own claims are source-verified inline.)*

> **Acted on (2026-06-16):** the shipped/superseded workstreams were moved to
> `docs/dev/archive/` (`betadelta/`, `bubble/`, `n-consistency/` — writeups +
> harnesses), each doc carries a verified **Status** line, and the two precise
> stale refs (`pshadow-design` `stop_r`, `stalling` velstruct note) were fixed.

## Open items carried forward (from archived docs)

- **β–δ Phase-4 — default-solver flip.** `betadelta_solver` defaults to `legacy`
  (`trinity/_input/default.param:49`); the verified-good `hybr` path exists but the
  flip to `hybr` is a deferred maintainer decision (`archive/betadelta/HYBR_PLAN.md` Phase 4).
- **β–δ Phase-5 — transition criterion.** The "is the fixed 0.05 cooling-balance
  trigger right (esp. steep r⁻²)?" question is now the **active** `transition/`
  workstream (`TRIGGER_PLAN.md` → `pshadow-design.md`), still unbuilt.

## Per-doc detail (decisive evidence)

### `archive/betadelta/HYBR_PLAN.md` — ✅ SHIPPED (Phases 0–3), live tail, ⚠️ stale-refs
Phase 1.1 shared `solve_R1` bracket (`get_bubbleParams.py:433-457`, 6 call sites), Phase 1.2
convergence-flag persistence + dt mitigation (`run_energy_implicit_phase.py:283-316`,
`registry.py:481-482`), Phase 3 hybr behind `betadelta_solver` (`get_betadelta.py:583-601,795-833`;
`default.param:49` default still `legacy`) all **shipped**. **Open tail:** Phase-4 default flip to
`hybr` (deferred maintainer call) and the Phase-5 transition-criterion study. Stale: `bubble_E2P:229`→`:198`;
the §3 `compute_R1_Pb` `[1e-3·R2,R2]` bracket no longer exists (replaced by the Phase-1.1 fix).

### `archive/betadelta/PHASE0_BASELINES.md` — ✅ SHIPPED (record)
Baselined the legacy solver (`GRID_EPSILON=0.02` `get_betadelta.py:57`; bounds `:41-43`); the
`BETADELTA_DT_SHRINK_MAX_STREAK=10` response shipped (`run_energy_implicit_phase.py:129`). Gate-G0 evidence
for a program now through Phase 3. The uncommitted `scratch/phase0/` harness it cites is gone (self-disclosed).

### `archive/betadelta/PHASE2_ARMS.md` — ✅ SHIPPED (arm D promoted)
hybr arm landed: `betadelta_solver` (`registry.py:307`), `_solve_betadelta_hybr` (`get_betadelta.py:874`),
`dMdt>0` acceptance gate (`:170,796`). Harness/data present. §2.3 self-marked "superseded in part" by its
own Phase-3 section; the forward item (steep r⁻² cooling-balance) lives in `stalling-energy-phase.md` + HYBR Phase 5.

### `archive/betadelta/stalling-energy-phase.md` — ✅ SHIPPED (settled "Problem 2" study)
`v_neg_frac_thick` + `_inflow_frac_thickness` shipped (`registry.py:463`,
`run_energy_implicit_phase.py:175-193`); "v absent from cooling integrals" verified in
`bubble_luminosity.py`; reject-and-hold remains harness-only (`velstruct/hunt.py`) by design. One stale
parenthetical: it calls `velstruct/` + data "gitignored scratch" but they are **committed**.

### `transition/TRIGGER_PLAN.md` — 🔵 ACTIONABLE
Clocks A/B and `phaseSwitch_LlossLgain` param verified (`run_energy_implicit_phase.py:1070-1079`,
`registry.py:346`); **no** candidate F0–F5 is wired into production (`git grep transition_trigger|blowout` → empty).
Plan is current and accurate; the next live step (P-shadow) is unbuilt.

### `transition/P0.md` — ✅ SHIPPED (results record)
Offline harvest record; harness (`transition/harness/harvest.py,psens.py`) + 5 `data/transition_*.csv` all present
and match. Complete by its own terms; keep with the plan until P-shadow ships.

### `transition/pshadow-design.md` — 🔵 ACTIONABLE, ⚠️ 1 stale ref
Entirely unbuilt (accurate "awaiting sign-off"): no `transition_trigger` param, no blowout/shadow code.
Routing `main.py:283,303,343` and the F0 terminator verified. Stale: §3 says `stop_r` default `None`; actual `'500'` (`registry.py:316`).

### `archive/bubble/integrator-robustness.md` — ⛔ SUPERSEDED, ⚠️ heavy stale-refs
The `_odeint_checked` wrapper it documents was **removed**; the structure solve is now
`solve_ivp(LSODA, dense_output=True)` via `_solve_bubble_structure` (`bubble_luminosity.py:106-166`, PRs #666/#678).
`BubbleSolverError` + the shape-aware guard shipped; the `sys.exit` it flags as pending is now fixed (`:1129-1136`).
Value is historical.

### `archive/bubble/conduction-convergence.md` — ✅ SHIPPED
The `solve_ivp(dense_output=True)` switch it signs off **landed** (`bubble_luminosity.py:143-151`); conduction band
sampled from dense output at `_CONDUCTION_NPTS=2000` (`:632-641`); tool `tools/bubble_conduction_convergence.py` exists
(its header still says "production odeint" — now stale).

### `cooling/refactor-audit.md` — 🔵 ACTIONABLE, ⚠️ ~1–3 line drift
**Nothing shipped:** the non-CIE loader is still hardcoded (`read_cloudy.py:48-49,267-343,183-188`); no `cool_col_*`
keys; CIE still integer-index with silent fall-through (`read_param.py:417-429`); unused `metallicity` arg
(`read_coolingcurve.py:25`). SPS side did land as `_resolve_sps_bundle` (`registry.py:216`), not the named helper.
Faithful, implementable plan; line numbers drifted +1..+5.

### `archive/n-consistency/audit.md` — ✅ SHIPPED
`n ≡ n_H`, He-aware μ's, `chi_e`/`chi_e_shell`, `mu_ion_shell`, Phase-A bubble-vs-shell split all implemented
(`read_param.py:302-324`, `registry.py:327-375`, `bubble_luminosity.py:539-1160`, `get_shellODE.py:63-66`) and pinned by
`test/test_mu_audit_drift.py`. `get_shellParams.py` removed. The audit's pre-fix line table is intentionally a snapshot.

### `archive/n-consistency/implementation-plan.md` — ✅ SHIPPED, ⚠️ stale-refs
Every phase (0–6 + A) landed (#657); pinned by `test_mu_audit_drift.py` (14 tests). Stale: before/after tables use
**flat filenames** (files now live in `phase1_energy/`, `phase1b_energy_implicit/`, `phase1c_transition/`,
`phase2_momentum/`) and pre-Phase-A line numbers.

### `archive/n-consistency/pressure-terms-audit.md` — ⛔ SUPERSEDED
Self-declared first pass; its `n_tot`-leaning fix is **not** what shipped (`n ≡ n_H` did). Superseded by `audit.md`
+ `implementation-plan.md` (both present). Kept only as a reasoning trail.

### `misc/backward-compat-audit.md` — 🔵 ACTIONABLE, ⚠️ heavy line-drift
Only the Tier-4 `unit_conversions` relabel shipped (`unit_conversions.py:233-254`); **~95% pending** — every Tier-1/2/3/5
shim still present (`_create_adaptive_radius_grid:875`, `_solve_bubble_ode_with_ivp:1002`,
`get_beta_delta_wrapper_pure` `get_betadelta.py:1078`, Phase-6 `DeprecationWarning` shims, `load_output=read` `trinity_reader.py:1095`).
The `feature/remove-backward-compat-codeblocks` branch doesn't exist. Re-anchor lines, mark Tier-4 done, keep.

### `misc/tinit-sensitivity.md` — 🟡 PARTIAL
`_T_INIT_BOUNDARY=3e4` (`bubble_luminosity.py:52`) and the `tools/tinit_sweep/` harness all present; the study
concluded (3e4 conservative). **Open:** recommendation #3 (drop the linear L3 patch over `[1e4,T_init]`, `:665-671`) is
not implemented. No stale refs.

### `misc/TERMINATION_EVENTS.md` — 📘 REFERENCE (accurate)
Every event factory, per-phase list, threshold, and `SimulationEndCode`/`EventResult` field matches
`phase_events.py:69-586` + `simulation_end.py`. A current, accurate reference — keep.

### `misc/LEAKING_LUMINOSITIES_SKELETON.md` — 🟡 PARTIAL
Phases A–C shipped: `coverFraction` param + `_validate_coverFraction` (`registry.py:305,153-165`), `get_leak_luminosity`
(`get_bubbleParams.py:261-303`), `L_leak` wired through energy + implicit phases, `test/test_cf_leak.py`. **Open:** Phase D
(mass sink) / Phase G (photon/X-ray) not started; audit findings #7/#8 (transition leak uses effective pressure + stale
diagnostic) still valid (`run_transition_phase.py` has no `bubble_Leak` refresh).
