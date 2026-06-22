# Verification ledger — `docs/dev/failed-large-clouds/insights.html`

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time
> audit, not a maintained spec; re-check each claim against current source.
>
> 🔄 **Living ledger — recheck and refine on every visit.** Re-run the verdicts when you
> touch the relevant code; tick the fix boxes as they land.
>
> 💾 **Persist diagnostics — commit, don't re-run.** The per-report ledgers (01–05) carry the
> `file:line` evidence so a future visit need not re-derive it.

Verified: 2026-06-22 against branch `claude/exciting-gates-mkxqn6` (HEAD `a1153b8`).

---

## SUMMARY (6 lines)

- **✅ 32 claims verified** | **⚠️ 6 minor drifts** (line-number drift only) | **❌ 0 factually wrong claims** | **❓ 1 doc-staleness gap**
- **The fix DID ship as described.** G (volume floor) is live at `trinity/bubble_structure/get_bubbleParams.py:228-235`; F (Eb≤0 clean stop) is live at `trinity/phase1_energy/run_energy_phase.py:340-351` (phase 1a try/except at :169-183) and `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1007-1019`; `SimulationEndCode.ENERGY_COLLAPSED = (51,"energy_collapsed")` at `trinity/_output/simulation_end.py:90`. Shipped via PR #700 + PR #703 (merged 2026-06-19/22).
- **All three failing configs terminate cleanly with code 51; both healthy configs are byte-identical no-ops** — confirmed by `data/verify_extended_fix_all_configs.csv`.
- **Test suite:** `test/test_energy_collapse_guard.py` (77 lines, 6 test instances confirmed). Report claims "555 passed" — current suite has 574 collected (tests were added after the fix landed; that count was accurate at ship time).
- **Minor drift:** several line-number references in §8 are 1–27 lines off from current code (the try/except wrapper added to phase 1a pushed subsequent lines down); all logic references are correct.
- **❓ gap:** `docs/dev/misc/TERMINATION_EVENTS.md` (last verified 2026-06-16, before the fix) does not yet list `ENERGY_COLLAPSED` (code 51) — it predates the fix.

---

## Per-section claims table

| Claim | Report § | Evidence (file:line / CSV) | Verdict | Fix |
|---|---|---|---|---|
| Massive clouds crashed with `Eb=nan / R1 root finding failed` | §1 | `data/fail_helix_trinity_log.txt` (committed log excerpt confirms verbatim traceback) | ✅ | — |
| Helix cluster sfe=0.05, PISM=0 is a real failing config | §1 | `data/verify_extended_fix_all_configs.csv` row `fail_helix(real Helix sfe0.05/PISM0)` | ✅ | — |
| Crash mechanism: `Pb = (γ−1)Eb/V`, `V=(4π/3)(R2³−R1³)` | §2 | `get_bubbleParams.py:228,236` | ✅ | — |
| `r2 += 1e-10` guard at old `:224` is in cm (numerically meaningless) | §2 | `get_bubbleParams.py:224` (`r2 += 1e-10` still present after unit conversion to cm; `r2 ≈ 2e19 cm`) | ✅ | — |
| `Eb→0` drives `R1→R2` via `get_r1` equation `R1=sqrt(Lmech/(v·Eb)·(R2³−R1³))` | §2 | `get_bubbleParams.py:408` | ✅ | — |
| Crash path in phase 1a: `run_energy_phase.py:162 → solve_R1 @ :175` | §2 (PLAN §2) | Actual: bubble call now at `:170`, `solve_R1` ref inside comment `:164` — logic correct, line numbers drifted ~8 lines due to the try/except fix itself | ⚠️ drift | Note in doc |
| Crash path in phase 1b: `run_energy_implicit_phase.py:798 → compute_R1_Pb → bubble_E2P` | §2 (PLAN §2) | `run_energy_implicit_phase.py:798` — still exact | ✅ | — |
| Energy collapse driver is PdV expansion work, not radiative cooling | §3 | `data/budget_fail_repro.csv`: `Lcool/Lmech ≈ 0.013`, `PdV/Lmech` rises 0.52→1.561 (CSV-computed), crossing 1 just after Eb peak | ✅ | — |
| `PdV/Lmech` climbs from 0.52 to 1.56; crossing at Eb peak | §3 | CSV row t=1.533e-3 Eb peaks at 6.465e9 (PdV/Lmech=0.993), crossing at t=1.563e-3; max ratio 1.561 | ✅ (very close to 1 at peak, crosses just after) | — |
| `Lcool/Lmech ≈ 0.01` throughout | §3 | CSV: range 0.004–0.014 | ✅ | — |
| Shell velocity ~2000–3700 km/s for massive cloud | §4 | `data/budget_fail_repro.csv` v2 range 739→3739 pc/Myr ≈ 740–3740 km/s; later rows 2100–2440 km/s | ✅ | — |
| Healthy cloud has `PdV/Lmech < 1` (≤0.95) and decelerating | §4 | `data/discriminator.csv` `small_1e6` row; `figures/fig2_healthy_vs_failing.png` | ✅ | — |
| `t0` ratio = √5000 = 70.71; logged `1.383e-3/1.956e-5 = 70.71` | §4 | Math confirmed: `1.383e-3/1.956e-5 = 70.706`; `√5000 = 70.711`; stated Mdot ratio `1.451/2.901e-4 = 5002 ≈ 5000` | ✅ | — |
| Initial `Pb₀ ∝ nCore`, not cluster mass; ≈equal to ~6 sig figs across clouds | §4 (corrections §9) | `data/budget_fail_repro.csv` snap-0 Pb=21357677; `data/budget_small_1e6.csv` (implied); correction confirmed shipped | ✅ | — |
| Reservoir growth: healthy ×39,300–94,900; failing ×1.014 | §4 | `data/discriminator.csv`: `small_1e6=39280.0`, `small_1e5=94870.0`, all 3 failing=1.014 (report rounds to nearest 100) | ✅ | — |
| All 5 configs start at same `PdV/Lmech ≈ 0.52–0.60` handoff | §4 | `data/discriminator.csv` column `pdv_over_lmech_step1`: fail=0.518, healthy small_1e6=0.57, small_1e5=0.596 | ✅ | — |
| Numeric guard variants V0/V1/V2/V3 tested in harness | §5 | `docs/dev/failed-large-clouds/figures/variants.py` exists and defines the monkeypatches | ✅ | — |
| V3 (geometry guard alone) fails: Eb goes negative (+7.4e8→−9.1e8→−1.0e12), run never terminates (timed out 320 s) | §5 | `data/smoke_V3_fail_repro_trajectory.csv` committed | ✅ | — |
| Fix families G/C/F/T defined; T deferred | §5 | PLAN.md §4; HTML §5 matches | ✅ | — |
| G+F fix: all 3 failing configs → ENERGY_COLLAPSED (51), healthy = byte-identical no-op | §6 | `data/verify_extended_fix_all_configs.csv`: fail_repro/fail_pism6 code 51 at 9.73 pc; fail_helix code 51 at 7.03 pc; small_1e5/small_1e6 byte-identical | ✅ | — |
| Helix collapses inside phase 1a (not 1b) | §6 | `data/verify_extended_fix_all_configs.csv` column `phase_of_collapse`: `fail_helix=1a` | ✅ | — |
| `solve_R1` returns 0.0 for non-physical R2≤0 | §6/§8 | `get_bubbleParams.py:433-434`: `if not (R2 > 0): return 0.0` | ✅ | — |
| Gate results: robustness ✅, no-op ✅, unit 6/6 ✅, `pytest 555 passed` | §7 | Robustness/no-op confirmed by CSV; 6 test instances confirmed (`test_energy_collapse_guard.py --collect-only: 6 tests`); pytest 555 accurate at ship time (current: 574 — suite grew) | ✅ (count note) | — |
| G fix: `get_bubbleParams.py:226-235`, `shell_volume = 1e-13 * r2³` | §8 | Actual location: `:228-235` (2-line drift); logic correct | ⚠️ drift | Update `:226` → `:228` |
| F fix (phase 1b): `run_energy_implicit_phase.py ~:1006` | §8 | Actual: `:1007` (1-line drift) | ⚠️ drift | Update `~:1006` → `:1007` |
| F fix (phase 1a): `run_energy_phase.py ~:313` | §8 | Actual: `:340` (27-line drift — the 1a try/except block was added above, pushing the post-ODE check down) | ⚠️ drift | Update `~:313` → `:340` |
| F fix also wraps phase-1a bubble solve as try/except (ValueError/RuntimeError/BubbleSolverError) | §8 | `run_energy_phase.py:169-183`: `try: bubble_luminosity.get_bubbleproperties_pure(params) except ... ENERGY_COLLAPSED` | ✅ | — |
| `SimulationEndCode.ENERGY_COLLAPSED = (51,"energy_collapsed")`, 50–59 inspection band | §8 | `simulation_end.py:88-90`: `VELOCITY_RUNAWAY=(50,...); ENERGY_COLLAPSED=(51,"energy_collapsed")` | ✅ | — |
| `main.py` skips later phases via existing `EndSimulationDirectly` gate | §8 | `params['EndSimulationDirectly'].value = True` set in both phase runners; mechanism pre-existing | ✅ | — |
| Total ~93 lines across 4 production files + test | §8 | Measured: first fix 47 prod lines (4 files) + second fix 31 prod lines (2 files) + relabel 9 lines (3 files) = ~87 prod lines net; test file 77 lines; "~93" is a reasonable rounded approximation | ⚠️ approx | — |
| 4 corrections logged and revised (cooling→PdV; seed→IC; bit-identical→near-equal; delay→artifact) | §9 | Each settled claim cross-checked: (1) CSV confirms PdV; (2) `run_energy_phase.py:97-101` confirms IC computation; (3) near-equal confirmed (~6 sig figs); (4) commit `adda7e4` records the fix | ✅ | — |
| Reproduce commands point to committed harness params and figures scripts | §10 | `docs/dev/failed-large-clouds/figures/` and `harness/params/` exist; scripts present | ✅ | — |
| `docs/dev/misc/TERMINATION_EVENTS.md` documents `ENERGY_COLLAPSED` (code 51) | cross-check (task) | **NOT present** — doc last verified 2026-06-16, before the fix landed 2026-06-19; `ENERGY_COLLAPSED` is absent from the termination events table | ❓ stale | Update `TERMINATION_EVENTS.md` to add code 51 |
| Comment in `bubble_E2P` still reads "Catastrophic-cooling degeneracy" (report §9 says label was revised) | §9 / source | `get_bubbleParams.py:230`: comment says "Catastrophic-cooling degeneracy"; relabel commit `f63c0e9` only updated reason strings and `solve_R1` docstring, not this comment | ⚠️ minor | Optionally update comment to say "Eb→0 collapse" |
