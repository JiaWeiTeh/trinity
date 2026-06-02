# TRINITY Test Strategy

This document is the living plan for TRINITY's test suite: what we have, what is
stale or redundant, and the state-of-the-art simulation test cases we want to
build. It is a planning artifact — no tests are implemented by this document.

Companion docs: `TERMINATION_EVENTS.md` (event semantics),
`LEAKING_LUMINOSITIES_SKELETON.md` (an example of a physics feature whose tests,
`test/test_cf_leak.py`, are the model the rest of the engine should follow).

---

## Guiding principle

> Prefer tests that assert an **independent** truth — a known analytic solution,
> a conservation law, or a scaling/symmetry invariant — over tests that
> re-implement the production formula and compare.

A test that recomputes the same right-hand side and checks equality is a
*change-detector*: it passes even when the physics is wrong, and it breaks on
every harmless refactor. The invariant/limit/conservation style catches real
regressions and survives refactors. (For contrast, see
`test_cf_leak.py::test_formula_matches_enthalpy_flux`, which re-derives the leak
expression — fine as a smoke check, but the *invariant* and *unit-landing* tests
in that same file are the higher-value pattern.)

---

## Current state (baseline, June 2026)

| Metric | Value |
|--------|-------|
| Tests | 370 (all passing, 0 skipped, 0 xfail) |
| Overall line coverage | **29 %** |
| Fast suite runtime | 4.3 s (369 tests) |
| Slow suite runtime | +158 s — a single subprocess test (`test_run_smoke.py`) |
| CI | `pytest test/` on Py 3.9–3.12; **no coverage gate**; smoke test runs on every matrix job |

**Coverage is inverted relative to risk.** The plumbing is tested heavily while
the science is barely tested at all:

| Area | Coverage | Notes |
|------|----------|-------|
| `_output/cloudy/*`, `_functions/simplify.py`, `_input/registry` | 88–97 % | I/O & parameter plumbing |
| `phase1_energy`, `phase1b_energy_implicit`, `phase1c_transition`, `phase2_momentum` | **0 %** | the actual physics phases |
| `phase0_init/*`, `bubble_structure/bubble_luminosity.py`, `shell_structure/*`, `cooling/*`, `sps/*`, `phase_events.py`, `main.py` | **0–26 %** | initialization + core engine |

The engine is exercised only by `test_run_smoke.py`, which runs `run.py` as a
subprocess and asserts only that it exited 0 and wrote ≥2 snapshot lines. It
cannot catch a numerical regression, and because it is a subprocess,
`coverage.py` reports 0 % for everything it runs.

---

## Part 1 — Current problems to fix

| # | Problem | Action | Effort |
|---|---------|--------|--------|
| F1 | No `conftest.py`. `disable_crash_handlers` fixture is copy-pasted in 4 files; synthetic-run builders (`_build_v3_run`, `_make_minimal_v4_run`, …) in 5 files, already drifting | Add `test/conftest.py`; hoist the fixture and one parametrized `make_run()` builder. Removes ~150 lines of duplicated setup | S |
| F2 | `test_phase4_consumer_migration.py` and `test_phase5_text_drop.py` are migration scaffolding ("text parser removed in Phase 6"); they overlap `test_metadata.py` (`TestReadSimulationEndMigration`, `TestTerminationBlock`, `TestTerminationDebugBlock`) | Fold the genuine regression guards (e.g. the `.get("reason")`→`None` bug) into `test_metadata.py`; delete the "signature changed" and transitional-`DeprecationWarning` assertions once the metadata migration is declared complete | S |
| F3 | CI has no coverage gate; the 2.5-min smoke test runs on all 4 Python versions; 153 `UserWarning`s (SB99 band) from synthetic fixtures + globally-hidden `DeprecationWarning`s mean the suite is not warning-clean | Mark smoke `@pytest.mark.slow` and run it on one interpreter only; add `--cov` with a floor; feed in-range cluster ages in fixtures; remove the blanket `filterwarnings = ignore::DeprecationWarning` | S |
| F4 | Stale comment in `test_run_smoke.py:53` references `summary.txt` / `simulationEnd.txt`, which Phase 5 stopped writing | One-line fix | XS |

Nothing in the suite is broken or uncollectable; the issues are duplicated
setup, time-boxed migration tests, and CI hygiene.

---

## Part 2 — State-of-the-art simulation test pyramid (full scope)

Proposed directory layout and markers:

```
test/
  unit/         # Tier 1 — pure functions, no data files
  physics/      # Tiers 2–4 — analytic limits, conservation, events
    conftest.py # SPS-mock + constant-feedback fixtures
  regression/   # Tier 5 — golden-master runs
  conftest.py   # shared fixtures (F1)
markers: physics, slow
```

### Key test seams (verified)

- **In-process run entry:** `trinity.main.start_expansion(params)` where
  `params = trinity._input.read_param.read_param(path)` — runs the full engine
  *without a subprocess*, so `coverage.py` sees it. (`trinity/main.py:81`)
- **Feedback mock seam:** every phase ODE calls
  `trinity.sps.update_feedback.get_current_sps_feedback(t, params)`, which
  returns a plain `SPSFeedback` dataclass. `monkeypatch` it to inject constant
  `Lmech` / `ṗ` for analytic-limit tests. *Phase 0 is the exception:
  `get_InitPhaseParam.get_y0` reads the SPS interpolators directly via
  `params['sps_f']['fLmech_W'](tSF)` / `['fpdot_W'](tSF)`, so mock that dict
  instead.*
- **Pure ODE right-hand sides:** `get_ODE_Edot_pure(t, y, snapshot, params)`
  (`phase1_energy/energy_phase_ODEs.py:169`) and
  `get_ODE_momentum_pure(t, y, snapshot, params)`
  (`phase2_momentum/run_momentum_phase.py:371`); state vector `y = [R2, v2, Eb]`.
- **Bundled data:** CIE/non-CIE cooling and SPS tables ship in `lib/default/`,
  so full-physics tests run with real data (the smoke test already proves this).

---

### Tier 1 · Pure-function unit tests  *(fast, no data files)*

Close the 0 % gap on standalone helpers. Direct calls, no mocking.

| Target (`file:line`) | Assertion |
|----------------------|-----------|
| `bubble_E2P` ↔ `bubble_P2E` (`get_bubbleParams.py:197,231`) | Round-trip identity `E → P → E` |
| `pRam` (`get_bubbleParams.py:304`) | `P_ram = ṗ / 4πr²` consistency |
| `get_effective_bubble_pressure` (`get_bubbleParams.py:329`) | Returns thermal / ram / max branch per phase |
| `get_r1` (`get_bubbleParams.py:401`) | Residual ≈ 0 at the returned root (pressure balance) |
| `get_density_profile` (`cloud_properties/density_profile.py:55`) | Power-law slope `n ∝ r^α` |
| `get_mass_profile` (`cloud_properties/mass_profile.py:131`) | Matches closed form `M = 4π/3·ρr³` for uniform ρ |
| `get_shellODE` (`shell_structure/get_shellODE.py:24`) | φ stays in [0,1]; ionizing flux conserved |
| `get_InitPhaseParam.get_y0` (`phase0_init/get_InitPhaseParam.py:44`) | Closed-form Weaver/Rahner init: `E0 = (5/11)·Lw·dt` (Weaver+77 Eq 20), `T0` (Eq 37), free-streaming `dt` (Rahner Eq 1.15); mock `params['sps_f']` interpolators; negative tests on the `tSF≥0` / `nCore>0` / `bubble_xi_Tb∈[0,1]` guards |

*~15–20 tests.*

### Tier 2 · Analytic verification  *(the crown jewels)*

Mock `get_current_sps_feedback` → constant `SPSFeedback`, set initial conditions,
integrate the relevant ODE/phase, fit the trajectory, and check the power-law
exponent to ~1–2 %.

| Limit | Setup | Expected |
|-------|-------|----------|
| **Weaver (1977)** energy-driven | constant `Lmech`, uniform ISM, cooling on | `R2 ∝ (L/ρ)^{1/5} t^{3/5}`, `v ∝ t^{-2/5}` |
| **Sedov–Taylor** | impulsive `Eb`, **cooling off** | `R2 ∝ (E/ρ)^{1/5} t^{2/5}` (validates adiabatic energy ODE independently of Weaver) |
| **Momentum snowplow** (phase 2) | constant `ṗ`, `Eb ≈ 0` | `R2 ∝ (ṗt/ρ)^{1/2}`, `v ∝ t^{-1/2}` |

*~6–8 tests. Highest scientific value — these verify the physics, not just guard it.*

### Tier 3 · Conservation & invariants

| Test | Assertion |
|------|-----------|
| Energy budget (energy-phase RHS) | `dEb/dt ≈ L_mech − L_cool − 4πR2²·Pb·v − L_leak` (residual ≈ 0) |
| **Unit-landing** | Physical inputs land in code units with no hidden factor. *`Eb` and `ENERGY_FLOOR = 1e3` are in **code (AU) units**, not erg — verified in `get_InitPhaseParam.get_y0` (`E0 = (5/11)·Lw·dt`, docstring `[au]`) and `phase1c_transition/run_transition_phase.py:94`. The genuine subtlety worth pinning: `get_y0`'s `T0` formula switches to cgs internally (`L_au2cgs`, `ndens_au2cgs`).* Extend the `test_cf_leak` pattern |
| Scaling invariant | Doubling `Lmech` shifts Weaver `R2(t)` by `2^{1/5}` |
| Sealed-bubble invariant | `Cf = 1` ⇒ leak = 0 (already covered by `test_cf_leak`) |

*~6 tests.*

### Tier 4 · Phase-machine / event tests  *(pure factories — easy, high value)*

The `make_*_event(...)` factories in `phase_general/phase_events.py` return
`event(t, y)` closures; test each fires at its threshold with the right
`terminal`/`direction`, and that the right `SimulationEndCode` is set.

| Event (`phase_events.py`) | Trigger |
|---------------------------|---------|
| `make_cloud_boundary_event` (:218) | zero-crosses at `R2 = rCloud` (energy → implicit) |
| `make_energy_floor_event` (:250) | `Eb < floor` (transition → momentum) |
| `make_cooling_balance_event` (:317) | `(Lgain − Lloss)/Lgain < threshold` (implicit → transition) |
| max-radius / min-radius / velocity-runaway | correct terminal `SimulationEndCode` |

*~10 tests.*

### Tier 5 · Golden-master regression  *(the net under the whole engine)*

Drive **in-process `start_expansion(params)`** for 2–3 short canonical configs
(one per density profile: PL, BE, homogeneous). Store baseline trajectories —
`R2(t)`, `v2(t)`, `Eb(t)`, phase-transition times, exit code — as committed
fixtures and assert against them at a numerical tolerance. Provide a
`--update-golden` regeneration path. This replaces the "did it write a file"
smoke check with "did it produce the *right numbers*."

*~3 tests, marked `slow`.*

### Tier 6 · Convergence  *(optional, advanced)*

Tighten `solve_ivp` `rtol`/`atol` and assert Richardson-style convergence of the
solution — proves numerical, not just physical, correctness. Defer unless wanted.

---

## Part 3 — Rollout sequence

1. **P0** — F1–F4 (infra + CI hygiene) + one Tier 5 golden baseline. *Immediate safety net and clean foundation.*
2. **P1** — Tier 2 (analytic) + Tier 1 (pure units). *The scientific core.*
3. **P2** — Tier 3 (conservation) + Tier 4 (events).
4. **P3** — Tier 6 if desired.

**Target outcome:** engine coverage moves from ~0 % to meaningful; the suite
gains *verification* value (not just regression detection); the test tree
reorganizes into `unit / physics / regression` with shared fixtures, shrinking
the duplicated bulk.
