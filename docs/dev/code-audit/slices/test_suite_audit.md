# Phase 4 — adversarial test-suite audit

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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

## Scope and method

Target: everything under `/home/user/trinity/test/` (48 `test_*.py` files, 10,298 lines,
553 test functions before parametrisation), read against `/home/user/trinity/trinity/**` and
`pyproject.toml`. Off-limits trees (`old_doNotRead/`, `outputs/` — except for git metadata,
`scratch/`, `tbd/`, `fig/`) were not read. This phase is read-only; nothing was modified.

Method:

1. **Static sweep.** AST walk of every test function to find (a) functions with no `assert` and
   no `pytest.raises`/`warns`, (b) functions whose assertions are only shape/length/finiteness,
   (c) every `expected = …` binding, (d) every assertion containing a multi-digit numeric literal
   (160 such lines), (e) every `monkeypatch`/mock site.
2. **Dynamic probes.** Three read-only scripts were run against the real code to *measure* rather
   than assert: the event-dispatch masking repro, the cooling-cube NaN fraction, and the
   default/stress collection counts. Every quantitative claim below marked "measured" comes from
   one of these.
3. **Provenance.** Each hardcoded expected number was traced to its docstring/comment provenance
   and, where useful, `git log -S`. Note: this repo's history is heavily squashed (the whole
   audit tree lands in ~5 commits), so `git log -S` returns the squash commit for almost every
   value and is *not* a usable provenance tool here. Provenance below is established from the
   tests' own stated origins and from cross-checking the value against an independent derivation.

Baseline confirmed by a full run in this container (7m28s):

```
$ python -m pytest --collect-only -q
852/861 tests collected (9 deselected)
$ python -m pytest -m stress --collect-only -q
collected 861 items / 852 deselected / 9 selected
$ python -m pytest -q -rs
========= 851 passed, 9 deselected, 153 warnings in 448.40s (0:07:28) ==========
```

**0 failed, 0 skipped, 0 xfailed.** The headline "847 passing" is this same set, four tests
older. The 9 deselected are the `stress` mark. Nothing in the suite is being hidden behind a
skip marker — that part of the story is clean (see the pytest-config section).

---

## Mandate 1 — Tautology hunt

### 1.1 The hard case: assertions that are algebraic identities in the test's own locals

`test/test_mu_audit_drift.py:108` — `test_phase2_bubble_n_rho_cie_vs_original`. This test is
named as validating the bubble interior refinement (`n -> n_H`, `rho -> mu_H*n`,
`CIE -> chi_e*n^2*Lambda`). **It calls no TRINITY function at all.** It defines both the "original"
and the "refined" formulas as local expressions and then asserts relations between them:

```python
    Pb, T, Lam = 1.234e-3, 1.0e6, 5.0e-7  # arbitrary positive test point

    # ORIGINAL operations (pre-fix 7321fef)
    n_orig = Pb / (2 * kB * T)
    rho_orig = n_orig * mi
    cie_orig = n_orig**2 * Lam

    # REFINED operations (current committed code)
    n_new = Pb / ((mc / mi) * kB * T)
    rho_new = n_new * mc
    cie_new = chi * n_new**2 * Lam

    # Intended factors vs original (no other drift).
    assert np.isclose(n_new / n_orig, 2.0 / (mc / mi), rtol=1e-12)        # ~0.8696
    assert np.isclose(rho_new / rho_orig, 2.0, rtol=1e-12)               # factor-2 fix
    assert np.isclose(cie_new / cie_orig, chi * (2.0 / (mc / mi)) ** 2,  # ~0.9074
                      rtol=1e-12)
```

Every one of those is an algebraic identity in `n_orig`/`n_new` as defined two lines above:
`n_new/n_orig ≡ 2/(mc/mi)` by construction; `rho_new/rho_orig = (n_new·mc)/(n_orig·mi) ≡ 2.0`
exactly, for **any** values of `mc`, `mi`, `Pb`, `T`. The three follow-up assertions
(`n_new == (mi/mc)*ntot`, `rho_new == mi*ntot`, `rho_orig == 0.5*mi*ntot`) are likewise
identities. The function cannot fail no matter what `bubble_luminosity.py` contains. The only
thing actually protecting that refinement is the *source-text grep* in the sibling
`test_phase2_no_original_operations_remain` (see §1.6).

### 1.2 Implementation compared against itself through a shared call

`test/test_sweep_jobs.py:65` — `test_emit_manifest_matches_combinations`:

```python
def test_emit_manifest_matches_combinations(tmp_path) -> None:
    sweep, out, jobs, _n, _i = _emit(tmp_path)
    cfg = read_sweep_config(str(sweep))
    expected = [name for _params, name in generate_combinations_from_config(cfg)]

    manifest = json.loads((jobs / 'manifest.json').read_text())
    assert [r['name'] for r in manifest['runs']] == expected
```

`emit_jobs` produces `manifest['runs']` by calling **the same function**
(`trinity/_input/sweep_jobs.py:129`: `combinations = list(generate_combinations_from_config(config))`,
then `:183`: `'name': name`). The test recomputes the list with the identical call and compares.
The assertion holds by construction. Critically, this is the test that stands where the
**confirmed run-name collision** would surface: if `generate_run_name` maps two distinct
combinations to the same name, that duplicate appears in `manifest['runs']` *and* in `expected`,
so the comparison still passes. See TEST-02.

The neighbouring `test_emit_writes_full_bundle` *does* have the shape that could catch it —
`assert len(list((jobs / 'params').glob('*.param'))) == 4` counts written files, and a collision
overwrites one — but `_make_sweep` only ever builds `mCloud [1e5, 1e7] × sfe [0.01, 0.10]`, four
names that cannot collide. The detector exists and is never pointed at a colliding grid.

### 1.3 Mocks that remove the physics under test

The beta-delta solver is the most heavily tested subsystem in the repo: 53 test functions across
`test_betadelta_solver.py` (19), `test_betadelta_hybr.py` (10), `test_betadelta_solver_switch.py`
(9), `test_betadelta_dt_mitigation.py` (15). **The ~29 in the first two run with the bubble
physics deleted.** `test/test_betadelta_hybr.py:51`:

```python
def install_landscape(monkeypatch, gE, gT, dmdt=lambda b, d: 1.0, edot_beta=None):
    """Patch get_residual_pure + get_residual_detailed to a synthetic landscape."""
    ...
    monkeypatch.setattr(GBD, "get_residual_pure", pure)
    monkeypatch.setattr(GBD, "get_residual_detailed", detailed)
```

and `test/test_betadelta_solver.py:113`:

```python
def forbid_bubble_solve(monkeypatch):
    def bomb(*a, **k):
        raise AssertionError(...)
    monkeypatch.setattr(GBD, "get_bubbleproperties_pure", bomb)
```

This is *legitimate design* — the docstrings say so explicitly, and the goal is the search
algorithm (scan order, early exit, warm-start threading, rescue paths), not the residual. But
the consequence must be stated plainly: **no bubble-structure physics executes in 29 of the 53
beta-delta tests**, and the only tests that drive hybr through real physics
(`test_betadelta_hybr_stress.py`, 2 tests) are `@pytest.mark.stress` and therefore deselected by
default. The residual formula itself has exactly one non-stress guard,
`test_residual_resample.py`, and that one reuses the production helpers (§1.5).

### 1.4 Loose or defaulted tolerances

| Site | Tolerance | Error it would admit |
| --- | --- | --- |
| `test_run_smoke.py:85` `assert value == pytest.approx(expected, rel=1e-6)` | rel 1e-6 on `R2`, `v2`, `Eb` | The only end-to-end numeric gate. Tight — but it fires at `t = 1e-4` Myr (100 yr), so it constrains only the first ~10 snapshots of phase 1. |
| `test_simplify.py:52-59` `assert np.isclose(x_out[0], x_in[0])` (helper `assert_endpoints_preserved`) | numpy defaults `rtol=1e-5, atol=1e-8` | Claims *endpoint preservation*, which should be exact index passthrough. Any endpoint whose magnitude is below `1e-8` passes unconditionally. TRINITY routinely simplifies arrays in code units where that is the physical range. |
| `test_simplify.py:299` `if np.isclose(x[j], xi) and np.isclose(y[j], yi)` | numpy defaults | Same, used to *identify* which input index an output point came from — a near-collision silently maps to the wrong index and the positional-order assertion still passes. |
| `test_cf_leak.py:35` `pytest.approx(expected)` | pytest default `rel=1e-6` | Expected is a restatement of the implementation line (§1.5), so the tolerance is moot. |
| `test_fkappa_auto.py:32-55` `pytest.approx(12.0)` etc. | pytest default `rel=1e-6` | Table-lookup fidelity; values span 1.0→64.0, one decade. Adequate. |
| `test_dR2min_magic_number.py:278-283` | `1e-5` relative on T, dT/dr; `1e-4·|v0|` absolute on v | Cross-solver LSODA-vs-Radau; the bar is stated in `docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md`. Defensible. |

No case was found of `np.allclose`/`pytest.approx` with a defaulted tolerance applied to a
quantity spanning many decades where the looseness is load-bearing — the suite is generally
tight (`rtol=1e-12` is the house style). The tolerance problem here is not looseness; it is
that tight tolerances are applied to *mirrors of the implementation* (§1.5).

### 1.5 Mirror-of-implementation: the dominant pattern

The most common test shape in this suite restates the production formula in the test body and
asserts equality at `rtol=1e-12`. These are not tautologies — they catch a later edit to the
production line — but they are **drift detectors, not correctness checks**: if the production
formula is physically wrong, the mirror is wrong the same way and the test passes at machine
precision. Named instances:

- `test_cf_leak.py:32-35` — `expected = GAMMA/(GAMMA-1.0)*(1.0-Cf)*4.0*np.pi*R2**2*Pb*cs`,
  character-for-character `get_bubbleParams.py:280`.
- `test_mu_audit_drift.py:208` — `dndr_refined = mu_p_shell / mu_H / (kB * tion) * (dust + chi_sh * recomb)`,
  the `get_shellODE` body.
- `test_mu_audit_drift.py:285` — `expect = np.sqrt(p["gamma_adia"].value * (…) * T / mu) * cvt.v_cms2au`,
  the `get_soundspeed` body.
- `test_shell_overflow_guard.py:44` — `expect_dphidr = -4*np.pi*r**2*chi_e*aB*n**2/Qi - n*sd*phi`.
- `test_cooling_boost.py:49` — `assert eff == max(Lcool + Lleak, theta * Lmech)`, the
  `effective_Lloss` `theta_target` branch verbatim.
- `test_net_coolingcurve.py:62` — `_expected_noncie`, whose own docstring says: *"Reproduce
  get_dudt's non-CIE branch arithmetic EXACTLY (same ops, same order, incl. the in-place /=
  round-trip) so equality is bit-for-bit."* It calls the same `RegularGridInterpolator` object
  out of `params`. The test's genuine content is only the T-floor gate; the interpolation itself
  is compared to itself.
- `test_residual_resample.py:100` — `_reference_residual`, docstring: *"Replicates production's
  residual formula and EVERY return branch … using the production helpers"*. The independence is
  purely in the sampling grid (20,000 points vs production's 500). Honest and correctly scoped —
  it is a resample gate, not a residual-physics gate — but it should never be cited as evidence
  that the residual is right.

### 1.6 Source-text greps standing in for behavioural assertions

52 assertions across the suite assert on the *text* of production `.py` files rather than on
behaviour. `test_mu_audit_drift.py` carries 28 of them, e.g.:

```python
    assert bub.count(
        "Pb / ((params['mu_convert'].value / params['mu_ion'].value)"
    ) == 5
    assert s.count("params['mu_convert'].value") == 8   # 7 mass/grav/tau + BC
    assert total == 11, f"expected 11 refined HII-pressure sites, found {total}"
```

These are the *only* protection for several of the audit's Phase-2 refinements (§1.1). They
break on a `black` reformat, on a variable rename, and on any refactor that is behaviour-
preserving; and they pass if the arithmetic inside those 11 sites is wrong. Filed as TEST-12.

### 1.7 Tests that assert only that the code runs

The AST sweep found 21 test functions whose assertions are exclusively existence/length/
finiteness/isinstance. Most are honestly named (`test_parametric_loop_documents_limitation`,
`test_registry_has_unique_names`). Two carry names that promise more than they check:

- `test_energy_collapse_guard.py:52` `test_bubble_E2P_finite_when_shell_collapses` — asserts
  only `np.isfinite(Pb)`. The guard floors `shell_volume = 1e-13 * r2**3`; nothing checks the
  floored value is the intended one, so any floor magnitude passes.
- `test_betadelta_solver.py:196` `test_grid_point_failure_is_skipped` — asserts only
  `np.isfinite(best_beta) and np.isfinite(best_delta)`; it does not check that the *skipped*
  point was excluded from the best-of selection.

### 1.8 A whole test harness that pytest never runs

`trinity/_input/sweep_parser.py:890-1010` contains a 120-line `if __name__ == "__main__":`
self-test block covering `parse_value`, `format_scientific`, and — 20 cases —
`generate_run_name`, including sanitisation, unsafe-value rejection, and the length guard.
**None of it is a test.** `pyproject.toml` sets `testpaths = ["test"]` and
`python_files = ["test_*.py"]`, so pytest never collects it; and it does not assert — it prints:

```python
        status = "PASS" if name == expected else "FAIL"
        print(f"  {status}: generate_run_name({params}) = '{name}' (expected '{expected}')")
```

Even run by hand, `python -m trinity._input.sweep_parser` exits 0 with `FAIL` on stdout. The
function whose non-injectivity is a confirmed, run-dropping defect therefore has **zero**
enforced coverage while appearing, to a reader of the source, to be thoroughly tested. This is
the single most misleading artefact in the repo's testing story. Filed as TEST-04.

`trinity/` has 8 other modules carrying `__main__` self-test blocks
(`get_InitCloudProp.py`, `show_run.py`, `trinity_to_cloudy.py`, `sweep_runner.py`,
`read_param.py`, `dictionary.py`, `unit_conversions.py`, `logging_setup.py`,
`check_yesno.py`) — none collected. `test_unit_conversions.py`'s docstring notes this
explicitly for its own module: *"The parser had no automated coverage (only an eyeball
`__main__` harness)"* — that migration was done for `convert2au` and not for the others.

---

## Mandate 2 — Golden-value provenance

Counting *hardcoded numeric expected values in assertions* (160 assert-lines carry one or more),
the split is roughly **55 independently derived / 105 captured-from-this-code**. The split by
origin class:

### Independently derived (trustworthy as correctness evidence)

| Where | Count | Origin |
| --- | --- | --- |
| `test_conventional_units.py:38-58` | 4 | **astropy** unit algebra — `(1.0*u.M_sun*u.pc**2/u.Myr**2).to(u.erg).value`. A genuinely external oracle. |
| `test_unit_conversions.py:32-67` | 27 params | Composed from base constants (`cvt.CONV.cm2pc` …), never copied. Tests the *parser's routing*; will not catch a wrong base constant, and says so. |
| `test_r1_bracket.py:34,69` | 2 | Analytic asymptote `r1 ≈ sqrt(K·R2³)` for `r1 ≪ R2`, derived in the module docstring from the R1 equation. |
| `test_cf_leak.py:46-65` | 1 | Dimensional cross-check: build the leak in cgs, convert inputs to code units, compare. Independent of the code's own conversion path in the sense that it exercises the round trip. |
| `test_dR2min_magic_number.py:271-283` | 4 | **Cross-solver**: production LSODA vs an independent stiff Radau integration of the same ODE at `rtol=1e-10`. The strongest numerical evidence in the suite. |
| `test_mu_audit_drift.py:40-78, 258-268, 312-325` | ~12 | Exact-`Fraction` composition algebra (`mu_H = 1+4x_He`, `chi_e = 1+Z_He·x_He`, `mu_n → mu_H` at `x_He=0`). Hand-derivable. |
| `test_rosette_cf_harness.py:44-48` | 4 | Hand calculation on a synthetic trajectory: `chi2(t) = (2t-3)² + (1.5t-3)²`, minimum `t* = 21/12.5 = 1.68`, `chi2* = 0.36`. |
| `test_bench_theta_cum.py:91,109` | 2 | `slope_1mtheta` on a synthetic exact power law → analytic `-0.5`. |
| `test_shell_overflow_guard.py:66-67` | 2 | float64 limits: `_NSHELL_MAX < 1.34e154 ≈ sqrt(DBL_MAX)`. |

### Captured from a prior run of this same code

| Where | Count | Provenance (as stated in the test) |
| --- | --- | --- |
| `test_run_smoke.py:23-29` | **3** | *"Captured 2026-07-10 on Python 3.9.6, numpy 1.26.4, scipy 1.13.1…"*. **The only end-to-end numerical golden in the default suite.** Load-bearing: it is the sole gate on the composed integrate→write pipeline. |
| `test_betadelta_hybr_stress.py:57+` | ~10 | *"FILLED FROM A RECORDING RUN"* — 5 `(beta, delta)` pairs. Load-bearing for the hybr solver, and **deselected by default**. |
| `test_fkappa_auto.py:32-55` | 9 | The measured `f_kappa_fire` of an 819-run sweep of this code (`docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv`). Correctly scoped — the claim is lookup/interpolation fidelity, not physics — but the numbers themselves are code output. |
| `test_cloudy_cli.py`, `test_cloudy_run_loader.py`, `test_cloudy_snapshot_to_deck.py`, `test_show_run.py` | ~45 | All derived from `outputs/mockOutput/mockFullrun`, a committed captured run (`t_now_myr == 0.300`, `mCloud_msun ≈ 3.97e3`, `picks[0].index == 177`, `"2.510 pc"`, `AGE_YR ≈ 9.71e4`). `CLAUDE.md` says of `outputs/`: *"not source, do not tidy or treat as ground truth"* — yet ~45 assertions treat it as exactly that. Scope is I/O plumbing, which is the right use; but any physics error baked into that run is now a defended expectation. |
| `test_residual_resample.py`, `test_dR2min_magic_number.py` fixtures | — | `test/data/{residual_resample_fixture,dR2_stiff_state_fixture}.json`, described as *"real bubble solves"* captured by `docs/dev/performance/harness/capture_stiff_dR2_state.py`. Used as *inputs*, not expectations — this is the correct way to use captured data and does not lock in a defect. |
| `test_energy_collapse_guard.py:68,92` | 4 | `solve_R1(-154.0, -4.4e31, 5e12, 3739.0)` and `classify_energy_collapse(-9.143e8)  # the fail_repro collapse value` — captured crash arguments. Again inputs, not expectations. |
| `test_materialize_runtime.py:93-120` | 3 | `106` adds, split `9`/`97`. Captured from a fidelity audit of this code; the comments carry the derivation history. Pure inventory bookkeeping. |
| `test_metadata.py:110-113, 303` | ~5 | `mu_atom == 1.07e-57` etc. — the derived `mu_*` in code units, i.e. `float(Fraction(14,11)) * m_H_in_Msun`. Semi-derived; the same numbers are independently reconstructed in `test_mu_audit_drift.py:45-51`. |

**The load-bearing captured goldens** — the three that, if the code was wrong the day they were
taken, now actively defend the defect — are `test_run_smoke._FINAL_GOLDENS` (3 numbers, the only
integration gate), `test_betadelta_hybr_stress._GOLDEN` (10 numbers, deselected), and the
`mockFullrun` corpus (~45 assertions). Nothing else in the suite pins a physics result.

---

## Mandate 3 — Coverage against the audit's six confirmed findings

| # | Finding | Would the suite catch it? | Why | Smallest failing test |
| --- | --- | --- | --- | --- |
| 1 | **Event dispatch by list index** (`phase_events.py:392`) | **No** | `test_phase_events.py:120` is the only test of `check_event_termination`. It builds `t_events=[np.array([]), np.array([0.25])]` — index 0 **empty**. The loop therefore reaches index 1 for a reason unrelated to terminality, and `assert result.index == 1` reads as "picks the terminal event" while proving nothing of the kind. The companion parametrised test *does* assert `event.terminal is False` for `velocity_sign` — it pins the attribute the dispatcher never reads. | 10 lines: build `[velocity_sign, min_radius]`, give **both** non-empty `t_events`, assert `result.name == "min_radius"`. Verified failing today — measured: returns `name='velocity_sign', index=0, is_simulation_ending=False`. |
| 2 | **Sweep run-name collision** | **No — and the one test that could is a tautology** | `generate_run_name` is never called by any file in `test/`; its 20 cases live in a print-only `__main__` block pytest does not collect (§1.8). `test_emit_manifest_matches_combinations` recomputes `expected` with the same `generate_combinations_from_config` call `emit_jobs` uses, so a duplicate name appears on both sides. `test_emit_writes_full_bundle`'s file count *would* catch it but is only run on a 4-name non-colliding grid. | Two tests. (a) `assert generate_run_name({'mCloud':1e5,'sfe':0.01,'nCore':1e4,'dens_profile':'densPL','densPL_alpha':-1.5}) != generate_run_name({… 'densPL_alpha':-1.9})` (both `int()`-truncate to `_PL-1`). (b) An `emit_jobs` run over that 2-point `densPL_alpha` sweep asserting `len(glob('params/*.param')) == 2` and `n_jobs == 2`. |
| 3 | **User-set `mu_*` discarded** | **No** | No test anywhere writes `mu_ion`/`mu_atom`/`mu_convert`/`mu_mol` into a `.param` and reads it back. `test_mu_audit_drift.py:40` asserts only that the *derived* values match the historical constants — it is satisfied precisely *because* the user value is overwritten. Worse: `read_param.py:475-492` has an anti-stomp guard, but it tests **object identity** (`params[k] is not v_before`), and Step 6 mutates `.value` in place on the same `DescribedItem`, so the guard is structurally incapable of firing here — and the guard itself has **no test at all** (grep for `silently overwrote` / `_stomped` in `test/` → 0 hits). | 4 lines: write a `.param` containing `mu_ion 0.99`, `read_param` it, `assert params['mu_ion'].value == pytest.approx(0.99 * cvt.convert2au('m_H'))` — or, if discard is intentional, `pytest.warns(UserWarning, match="derived from x_He")`. |
| 4 | **`gamma_adia` hardcoded** | **No** | Every occurrence of `gamma` in `test/` is `5.0/3.0`: `test_cf_leak.py:17`, `test_r1_bracket.py:68`, `test_betadelta_solver.py:68`, `test_metadata.py:121`, `test_validate_gmc.py` (`"gamma": 5.0/3.0`). No test varies it, so the inconsistency between the `gamma`-honouring path (`bubble_E2P`, `get_leak_luminosity`) and the `5/3`-assuming path (`get_r1`'s missing `2/(3(γ-1))` factor, the Rahner-A12 pair, the Weaver chain) is invisible: at `γ = 5/3` the factor is exactly 1. | Parametrise `test_compute_R1_Pb_returns_true_small_root` over `gamma_adia ∈ {5/3, 7/5}` and assert `R1 == pytest.approx(sqrt(2/(3*(γ-1)) * Lmech/(v·Eb) · R2³), rel=1e-2)`. Passes at 5/3, fails at 7/5 by a factor `sqrt(5/3) ≈ 1.29`. |
| 5 | **NaN propagation from the cooling cube** | **No** | `trinity/cooling/` contains zero `isnan`/`nan_to_num` (grep: 0 hits). `test_net_coolingcurve.py` is the only cooling-cube test; it probes **six** `(n, T, φ)` points, all at `n = 1e2 cm⁻³`, `φ = 1e10 cm⁻²s⁻¹`, which land in the finite region. **Measured on `param/simple_cluster.param` at `t_now = 0.1`: cool cube `(33, 21, 22)`, `21.07 %` NaN; heat cube `21.07 %`; evaluating `RegularGridInterpolator` at every grid point returns `24.33 %` NaN.** The NaN is spread across *all* 21 log-T slices (18–25 % per slice), not confined to a corner. Incidentally, `assert direct == _expected_noncie(...)` *would* fail on NaN (`nan != nan`) — the test just never samples one. | `test_cooling_cube_has_no_nan`: load the cube via `non_CIE.get_coolingStructure(params)` and `assert not np.isnan(cool_cube.datacube).any()`, or (if NaN is a deliberate mask) `assert np.isfinite(ncc.get_dudt(...))` over the full grid. Fails today at 21 %. |
| 6 | **`np.isclose(times, t, rtol=1e-10)`** (`trinity_reader.py:721`) | **No** | `get_at_time` is called from exactly one test, `test_cloudy_snapshot_to_deck.py:102`, with `mode="closest"` and `t=0.15` against a nearest snapshot at `0.1482` — a gap of `1.8e-3`, four orders above `atol`. The exact-match branch is never exercised near its boundary. The arithmetic: `np.isclose` tests `|a-b| ≤ atol + rtol·|b|` with `atol=1e-8` defaulted; `rtol·t = 1e-10·15 = 1.5e-9 ≪ 1e-8`, so `rtol` never binds below `t = 100` Myr while `default.param:121` caps `stop_t 15`. The effective window is a fixed **1e-8 Myr = 10 yr**, not the intended 1e-10 relative. | 5 lines: write a two-row `dictionary.jsonl` with `t_now = 0.1` and `t_now = 0.1 + 5e-9`, `TrinityOutput.open`, `assert out.get_at_time(0.1 + 5e-9, key='R2') == <second row's R2>`. Returns the first row today. |

**Score: 0 of 6.** Every one of the six confirmed defects survives the full 852-test default suite.

---

## Pytest configuration

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["test"]
python_files = ["test_*.py"]
addopts = "-v --tb=short -m 'not stress'"
markers = ["stress: opt-in slow stress tests (deselected by default; run with `pytest -m stress`)"]
filterwarnings = ["ignore::DeprecationWarning"]
```

Consequences worth naming:

- `testpaths = ["test"]` + `python_files = ["test_*.py"]` is why the `__main__` self-test blocks
  in nine `trinity/` modules are invisible (§1.8). There is no `conftest.py` anywhere in the
  repo and no `--doctest-modules`.
- `filterwarnings = ["ignore::DeprecationWarning"]` is a blanket suppression, not a targeted
  one — a numpy/scipy deprecation that will become a behaviour change in the next pinned
  version cannot surface in CI.
- **`-m 'not stress'` deselects 9 tests, and 5 of those 9 are pure timing budgets**
  (`test_simplify.py::TestTiming::test_runtime_budget[…]` ×4 and `test_subquadratic_scaling`).
  Only **4** carry physics: `test_betadelta_hybr_stress::test_hybr_endtoend_no_crashes`,
  `::test_hybr_implicit_converges_and_matches_golden`,
  `test_bubble_solver_stress::test_smoke_no_bubble_solver_failures`,
  `test_energy_collapse_snapshot::test_energy_collapse_emits_no_negative_Pb`. Those four are
  the **only** tests in the repo that integrate past the energy→implicit boundary. The default
  `pytest` invocation's deepest physics reach is `test_run_smoke`'s `stop_t = 1e-4` Myr — **100
  years of simulated evolution**, phase 1 only.

**Skips and xfails.** There are **no `pytest.mark.skip` and no `pytest.mark.xfail`** anywhere in
`test/`, and the full run reports **0 skipped**. The only conditional skips are four
`pytest.importorskip("astropy.units")` calls at `test_conventional_units.py:38,43,50,55`;
astropy is a hard dependency (`pyproject.toml:40`), so in a correct install they never fire —
they account for `test/CLAUDE.md`'s historical "3 skipped" only on a machine without astropy,
and that is exactly the case where the suite's four genuinely-independent unit-conversion
assertions silently stop running. Every other `skip` string found by grep is a docstring or
test *name* about the code-under-test skipping something, not a pytest skip. **This is a
clearance: nothing is parked behind a marker.**

**Tests that target `docs/dev/` rather than `trinity/`.** `test_bench_theta_cum.py` (6),
`test_theta5_harvest.py` (5), `test_rosette_cf_harness.py` (12) load analysis scripts by path
with `importlib.util.spec_from_file_location` out of `docs/dev/`. 23 of the 852 default tests
therefore exercise no engine code at all. They are well written; they just do not count toward
the engine's coverage.

## Modules with zero test coverage that carry a confirmed finding

Import-graph analysis (which `trinity.*` modules any `test/*.py` imports, directly or as a
parent package). **19 modules are never imported by any test.** Ranked by size, the ones that
matter for this audit:

| Module | Lines | Carries a confirmed finding? |
| --- | --- | --- |
| `trinity/phase2_momentum/run_momentum_phase.py` | 931 | Yes — calls `check_event_termination` at `:735` (finding 1). |
| `trinity/phase1c_transition/run_transition_phase.py` | 886 | Yes — `check_event_termination` at `:653` (finding 1); `gamma_adia` threaded at `:505,751,836` (finding 4). |
| `trinity/phase1_energy/run_energy_phase.py` | 408 | Yes — `check_event_termination` at `:324` (finding 1). |
| `trinity/_input/sweep_runner.py` | 644 | Yes — the in-process sweep driver, the other consumer of the non-injective run name (finding 2). |
| `trinity/shell_structure/shell_structure.py` | 473 | Yes — checked *only* by source-text grep in `test_mu_audit_drift.py:243` (§1.6). |
| `trinity/cloud_properties/mass_profile.py` | 627 | — |
| `trinity/phase0_init/get_InitCloudProp.py` | 666 | Partially — hosts `gamma_adia` handling at `:317` and a `MockParam(5.0/3.0)` at `:628`. |
| `trinity/phase1_energy/energy_phase_ODEs.py` | 431 | Source-text grep only. |
| `trinity/main.py` | 370 | Reached only via `test_run_smoke`'s subprocess. |

`trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` is *nominally* covered, but only
three small pure helpers are imported (`classify_energy_collapse`, `update_unconverged_streak`,
`evaluate_r1_shadow`/`parse_transition_triggers`/`r1_transition_decision`). The 1400-line
integration loop containing the `break`-regardless at `:1096` is never entered by a test.

---

## Clearances — tests a future session can trust

Named because they are genuinely rigorous, with independently-derived expectations:

- **`test/test_conventional_units.py:37-58`** (`test_energy_au_to_cgs`, `test_luminosity_*`,
  `test_time_*`, `test_mdot_*`). Expected values come from **astropy**, an oracle entirely
  outside TRINITY. The strongest unit-correctness evidence in the repo.
- **`test/test_dR2min_magic_number.py:250-283`** (`test_production_matches_independent_radau_reference`
  and its stiff-state twin). Production LSODA vs an independent Radau integration at
  `rtol=1e-10`, on two *real captured* bubble states including the documented stiff
  `5e9/sfe0.01` regime, agreeing to `1e-5` on T, dT/dr and the inner-boundary temperature.
  Independent-method, edge-regime, and it uses captured data correctly (as input, not as the
  expectation).
- **`test/test_r1_bracket.py`** (all 5). The expectation is the analytic small-root asymptote
  `r1 ≈ sqrt(K·R2³)`, derived in the docstring; the test additionally *demonstrates* the old
  bracket raising, so it proves the fix rather than restating it. Caveat: pinned at `γ = 5/3`.
- **`test/test_cf_leak.py:46-65`** (`test_units_land_in_code_luminosity_no_hidden_factor`).
  Builds the leak in cgs, converts the inputs, and requires the code-unit result to reproduce
  the cgs result converted — a real dimensional cross-check, not a formula mirror. (The
  neighbouring `test_formula_matches_enthalpy_flux` is a mirror; trust the units one.)
- **`test/test_unit_conversions.py`** (27 parametrised cases + 3 rejection cases). Expected
  factors composed from base constants, never copied; covers fractional exponents, multiple
  denominators, and the error paths. Correctly scoped by its own docstring — it validates the
  *parser*, not the base constants.
- **`test/test_r1_shadow.py`** (14 tests). A clean, exhaustive truth table over a pure function,
  including the `None`/NaN and `k_blowout` scaling paths. No mocks, no mirrors.
- **`test/test_rosette_cf_harness.py:33-60`**. Synthetic trajectory with a hand-derived
  chi-squared minimum (`t* = 21/12.5 = 1.68`, `chi2* = 0.36`) — an actual hand calculation.
- **`test/test_engine_purity.py`**. An AST-based architectural invariant that cannot be
  satisfied accidentally.
- **`test/test_shell_overflow_guard.py:60-67`** (`test_cap_far_above_physical_density`). The
  bounds are float64 facts (`< sqrt(DBL_MAX)`) and a stated physical scale, not captured values.

Honourable mention with a caveat: **`test/test_residual_resample.py`** and
**`test/test_phase_helper_sync.py`** are both rigorous *for what they claim* — a resample-density
gate and a copy-divergence gate respectively — and both say so explicitly in their docstrings.
Neither is evidence that the underlying residual or `compute_max_dex_change` is *correct*.

---

```json
[
  {
    "id": "TEST-01",
    "file": "test/test_mu_audit_drift.py",
    "line": 108,
    "class": "other",
    "severity": "S3",
    "claim": "test_phase2_bubble_n_rho_cie_vs_original is a pure tautology: every assertion is an algebraic identity in variables the test itself defines, and the test calls no TRINITY function.",
    "evidence": "Lines 117-136 define n_orig = Pb/(2*kB*T), n_new = Pb/((mc/mi)*kB*T), rho_orig = n_orig*mi, rho_new = n_new*mc, then assert np.isclose(n_new/n_orig, 2.0/(mc/mi), rtol=1e-12) and np.isclose(rho_new/rho_orig, 2.0, rtol=1e-12). rho_new/rho_orig = (n_new*mc)/(n_orig*mi) = (2/(mc/mi))*(mc/mi) = 2.0 identically for any mc, mi, Pb, T. The only TRINITY call in the function is _p() to read constants.",
    "expected": "The test should call the production path (e.g. bubble_luminosity's n/rho/CIE construction) and compare its output to the independently-stated refined formula; only then does the assertion have content.",
    "failure_scenario": "The bubble interior n_H / rho / chi_e*n^2 refinement is silently reverted or mis-coefficiented in trinity/bubble_structure/bubble_luminosity.py. This named regression test passes at rtol=1e-12. The only remaining guard is the source-text grep in the sibling test_phase2_no_original_operations_remain, which a black reformat or a rename defeats.",
    "repro": "python -c \"import ast,sys; src=open('test/test_mu_audit_drift.py').read(); print([n.name for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name=='test_phase2_bubble_n_rho_cie_vs_original'])\" then read lines 108-136: no import of bubble_luminosity, no call into it.",
    "confidence": "high"
  },
  {
    "id": "TEST-02",
    "file": "test/test_sweep_jobs.py",
    "line": 65,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "test_emit_manifest_matches_combinations compares emit_jobs' output against the same function emit_jobs used to produce it, and is therefore incapable of detecting the confirmed run-name collision that silently drops sweep combinations.",
    "evidence": "Test line 68: `expected = [name for _params, name in generate_combinations_from_config(cfg)]`; assertion line 71: `assert [r['name'] for r in manifest['runs']] == expected`. Production trinity/_input/sweep_jobs.py:129: `combinations = list(generate_combinations_from_config(config))`, :183: `'name': name`. Identical call, identical output on both sides of ==.",
    "expected": "The manifest test should assert injectivity independently: `names = [r['name'] for r in manifest['runs']]; assert len(names) == len(set(names)) == n_jobs`, and emit_jobs should carry a duplicate guard.",
    "failure_scenario": "A sweep over densPL_alpha [-1.5, -1.9] (both int()-truncate to _PL-1), or over sfe [0.014, 0.0149] (both round to sfe001), or over two mCloud values sharing a 2-sig-fig mantissa, silently collapses to one run name. One .param overwrites the other, one job vanishes, and collect_report reports full success over the reduced set. Nothing in the suite fails.",
    "repro": "python -c \"from trinity._input.sweep_parser import generate_run_name as g; b={'mCloud':1e5,'sfe':0.01,'nCore':1e4,'dens_profile':'densPL'}; print(g({**b,'densPL_alpha':-1.5}), g({**b,'densPL_alpha':-1.9}))\"",
    "confidence": "high"
  },
  {
    "id": "TEST-03",
    "file": "test/test_phase_events.py",
    "line": 120,
    "class": "state",
    "severity": "S1",
    "claim": "The only test of check_event_termination gives event index 0 an EMPTY t_events array, so the confirmed index-order dispatch defect cannot surface; `assert result.index == 1` reads as terminality-awareness but proves only that the loop skips empty entries.",
    "evidence": "Test lines 124-132: `sol = SimpleNamespace(t_events=[np.array([]), np.array([0.25])], ...)`, `result = events.check_event_termination(sol, [phase_end_event, run_end_event])`, `assert result.index == 1`. Production phase_events.py:390-403 iterates `for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events))` and returns on the first non-empty, never reading event.terminal or event.is_simulation_ending as a selector. build_implicit_phase_events (:481) places the non-terminal make_velocity_sign_event() at index 0 ahead of min_radius/velocity_runaway/max_radius.",
    "expected": "A test where two events fire in the same segment and the terminal one is at a higher index must select the terminal event: result.name == 'min_radius', result.is_simulation_ending is True.",
    "failure_scenario": "In the implicit phase, any segment in which v2 crosses zero AND min_radius/max_radius/velocity_runaway terminates returns the velocity_sign result: is_simulation_ending=False, reason_code='velocity_sign', t = the v2 zero-crossing time rather than the terminal time, and y = the state at that crossing. run_energy_implicit_phase.py:1096 then breaks anyway, so the run stops with a wrong reason, a wrong stopping time, and a wrong terminal state written to dictionary.jsonl.",
    "repro": "python -c \"import numpy as np; from types import SimpleNamespace as S; from trinity.phase_general import phase_events as ev; e=[ev.make_velocity_sign_event(), ev.make_min_radius_event(1.0)]; sol=S(t_events=[np.array([0.10]),np.array([0.25])], y_events=[np.array([[3.,0.,4.]]),np.array([[1.,-2.,4.]])]); r=ev.check_event_termination(sol,e); print(r.name, r.index, r.t, r.is_simulation_ending)\"  ->  velocity_sign 0 0.1 False",
    "confidence": "high"
  },
  {
    "id": "TEST-04",
    "file": "trinity/_input/sweep_parser.py",
    "line": 890,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "generate_run_name has ZERO enforced test coverage. Its apparent 20-case test suite lives in an `if __name__ == \"__main__\":` block that pytest never collects and that prints 'FAIL' instead of raising.",
    "evidence": "grep for generate_run_name across test/*.py returns 0 hits. sweep_parser.py:890 opens `if __name__ == \"__main__\":`; :938-940 read `name = generate_run_name(params); status = \"PASS\" if name == expected else \"FAIL\"; print(f\"  {status}: ...\")`. pyproject.toml:114-115 sets testpaths=[\"test\"] and python_files=[\"test_*.py\"], and there is no conftest.py and no --doctest-modules anywhere in the repo.",
    "expected": "The 20 cases in the __main__ block (formatting, generic suffixes, sanitisation, unsafe-value rejection, length guard) should be a parametrised test_sweep_parser.py, converted from print to assert, plus an injectivity case the __main__ block does not cover.",
    "failure_scenario": "Any regression in run-name construction — including the confirmed non-injectivity — ships green. A reader of sweep_parser.py sees an extensive self-test and reasonably concludes the function is covered; a reader of the 852-test count concludes the same. Both are wrong. Eight further trinity/ modules carry the same never-collected pattern (get_InitCloudProp, show_run, trinity_to_cloudy, sweep_runner, read_param, dictionary, unit_conversions, logging_setup, check_yesno).",
    "repro": "grep -rn generate_run_name test/ | wc -l   ->  0 ; python -m trinity._input.sweep_parser | grep -c FAIL   (exits 0 regardless)",
    "confidence": "high"
  },
  {
    "id": "TEST-05",
    "file": "test/test_run_smoke.py",
    "line": 23,
    "class": "numerical",
    "severity": "S3",
    "claim": "_FINAL_GOLDENS is the suite's only end-to-end numerical expectation and is explicitly captured from a run of this same code; it locks in whatever the integrator did on 2026-07-10, at 100 years of simulated evolution.",
    "evidence": "Lines 23-29: `_FINAL_GOLDENS = { # Captured 2026-07-10 on Python 3.9.6, numpy 1.26.4, scipy 1.13.1, astropy 6.0.1, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2.  \"R2\": 0.2857315185200479, \"v2\": 44.73918438203256, \"Eb\": 778236.3470566473}` and line 85 `assert value == pytest.approx(expected, rel=1e-6)`. The .param written at line 43 sets `stop_t 1e-4` Myr.",
    "expected": "A captured golden is a legitimate change-detector, but it should be labelled as such and complemented by at least one physically-derived integration check (e.g. the early-time Weaver similarity solution R2 ∝ t^(3/5) over the first decade, or energy conservation over the segment).",
    "failure_scenario": "If any of the six confirmed defects (or an earlier one) was already active on 2026-07-10, these three numbers now defend it: a future correct fix will trip this test and read as a regression. Separately, the gate covers only 1e-4 Myr of phase 1 — the implicit, transition and momentum phases have no numerical end-to-end expectation at all in the default suite.",
    "repro": "sed -n '23,29p;80,86p' test/test_run_smoke.py",
    "confidence": "high"
  },
  {
    "id": "TEST-06",
    "file": "test/test_net_coolingcurve.py",
    "line": 72,
    "class": "numerical",
    "severity": "S1",
    "claim": "Nothing in the suite asserts finiteness of the non-CIE cooling cube or of get_dudt; the one cooling test samples six (n,T,phi) points that all happen to land in the finite 79% of the cube.",
    "evidence": "grep 'isnan|nan_to_num|np.nan' over trinity/cooling/*.py returns 0 hits. test_net_coolingcurve.py is the only cooling-cube test; _dudt() (line 55) fixes _NDENS_CGS = 1e2 and _PHI_CGS = 1e10 and the three tests probe only T in {1000, 10**nonCIE_Tmin, 5000, 1e4, 3e4, 5e4}. MEASURED on param/simple_cluster.param at t_now=0.1: cool cube shape (33, 21, 22), 21.07% NaN; heat cube 21.07% NaN; evaluating the RegularGridInterpolator at every grid point gives 24.33% NaN, spread across all 21 log-T slices (18-25% each), not confined to a corner.",
    "expected": "A test asserting np.isfinite over the loaded cube (or, if NaN is a deliberate out-of-model mask, asserting that get_dudt raises/clamps rather than returning NaN) at every grid point.",
    "failure_scenario": "A shell or interface state whose (n_H, T, phi) triplet falls in the masked 21% yields dudt = NaN. RegularGridInterpolator spreads NaN to every neighbouring cell, the NaN enters the bubble/shell energy budget, and the run either produces NaN state silently or dies far downstream with an unrelated message. No test fails at any point.",
    "repro": "python -c \"import numpy as np,scipy.interpolate; from trinity._input.read_param import read_param; import trinity.cooling.non_CIE.read_cloudy as n; p=read_param('param/simple_cluster.param'); p['t_now'].value=0.1; c,h,net=n.get_coolingStructure(p); print(c.datacube.shape, np.isnan(c.datacube).mean())\"  ->  (33, 21, 22) 0.2106782106782107",
    "confidence": "high"
  },
  {
    "id": "TEST-07",
    "file": "test/test_cf_leak.py",
    "line": 17,
    "class": "coefficient",
    "severity": "S2",
    "claim": "No test in the repo sets gamma_adia to any value other than 5/3, so the split between the gamma-honouring and gamma-hardcoding code paths is untestable by construction.",
    "evidence": "Every gamma in test/: test_cf_leak.py:17 `GAMMA = 5.0/3.0`; test_r1_bracket.py:68 `gamma_adia=5.0/3.0`; test_betadelta_solver.py:68 `'gamma_adia': 5.0/3.0`; test_metadata.py:121 `\"gamma_adia\": 5.0/3.0`; test_validate_gmc.py `\"gamma\": 5.0/3.0`. Production get_r1 (get_bubbleParams.py:404) solves `sqrt(Lmech/v_mech/Eb*(r2**3-r1**3)) - r1`, which omits the 2/(3*(gamma-1)) factor that equals exactly 1 at gamma=5/3; bubble_E2P (:198) and get_leak_luminosity (:242) take gamma as a parameter and honour it.",
    "expected": "At least one parametrised test over gamma_adia in {5/3, 7/5} on the R1/Pb path, asserting the analytic root r1 = sqrt(2/(3*(gamma-1)) * Lmech/(v*Eb) * R2**3).",
    "failure_scenario": "A user sets gamma_adia to 7/5 (diatomic) in a .param. bubble_E2P and get_leak_luminosity use 1.4; get_r1, the Rahner-A12 pair and the Weaver structure chain silently keep 5/3. R1 comes out a factor sqrt(5/3) ~ 1.29 too large, Pb is inconsistent with the R1 it was computed from, and the whole force budget is wrong. The suite is green.",
    "repro": "grep -rn 'gamma' test/*.py | grep -v '5.0 / 3.0\\|5.0/3.0\\|5/3'   ->  no numeric alternatives",
    "confidence": "high"
  },
  {
    "id": "TEST-08",
    "file": "trinity/_input/read_param.py",
    "line": 475,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The user-set-mu defect is untested twice over: no test writes a mu_* into a .param, AND read_param's own anti-stomp guard checks object identity rather than value, so it is structurally unable to fire on Step 6's in-place mutation — and the guard itself has no test.",
    "evidence": "read_param.py:481-483: `_stomped = [k for k, v_before in _default_items_before.items() if k in params and params[k] is not v_before]`. Step 6 (:316-319) does `params['mu_convert'].value = float(_muH) * _mH_au` etc. — same DescribedItem object, so `params[k] is v_before` stays True and _stomped is empty. grep for 'silently overwrote' or '_stomped' in test/ returns 0 hits. grep for a .param containing mu_ion/mu_atom/mu_convert/mu_mol in test/ returns 0 hits. registry.py:363-366 registers all four with category='input_constants' (user-facing), not 'derived_init' — unlike chi_e and mu_ion_shell (:415-417) which are correctly categorised.",
    "expected": "Either honour the user value, or refuse it loudly. A test writing `mu_ion 0.99` into a .param and asserting either the honoured value or a pytest.warns/raises. Plus a test of the anti-stomp guard itself, extended to compare .value not just identity.",
    "failure_scenario": "A user sets mu_ion in their .param to model a different ionisation state. read_param silently overwrites it from x_He/Z_He. The run proceeds with the default composition, the metadata.json records the overwritten value, and there is no way to tell from the output that the input was ignored. The guard that exists precisely to prevent this class of bug (its comment names include_PHII as the last offender) does not fire.",
    "repro": "grep -rn \"mu_ion\\|mu_atom\\|mu_convert\\|mu_mol\" test/*.py | grep -i \"write_text\\|param\\b\"  ->  0 hits; sed -n '475,492p' trinity/_input/read_param.py",
    "confidence": "high"
  },
  {
    "id": "TEST-09",
    "file": "trinity/_output/trinity_reader.py",
    "line": 721,
    "class": "numerical",
    "severity": "S2",
    "claim": "get_at_time's exact-match branch has no test at its tolerance boundary; the single call site in the suite is 1.8e-3 Myr away from any snapshot, four orders above the effective window.",
    "evidence": "Production line 721: `exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]`. np.isclose defaults atol=1e-8; the predicate is |a-b| <= atol + rtol*|b|. With default.param:121 `stop_t 15`, rtol*t <= 1.5e-9 << 1e-8, so rtol never binds and the window is a fixed 1e-8 Myr (10 yr). The only test call is test_cloudy_snapshot_to_deck.py:102 `bundle.output.get_at_time(0.15, mode=\"closest\", quiet=True)`, and the test's own comment at test_cloudy_cli.py:304 records the nearest snapshot as ~1.83 kyr away.",
    "expected": "A test with two snapshots separated by less than 1e-8 Myr asserting that get_at_time returns the nearer one, and/or an assertion that the tolerance is absolute-by-design (atol given explicitly) rather than an accidental default.",
    "failure_scenario": "Two snapshots within 10 yr of each other — routine in the early energy phase where the smoke config writes ~10 snapshots inside 1e-4 Myr = 100 yr — both satisfy isclose, and get_at_time returns the first by array order rather than the nearest. Downstream cloudy deck export and any analysis that pins a time silently reads the wrong snapshot.",
    "repro": "python -c \"import numpy as np; t=np.array([0.1, 0.1+5e-9]); print(np.where(np.isclose(t, 0.1+5e-9, rtol=1e-10))[0])\"  ->  [0 1], first index wins",
    "confidence": "high"
  },
  {
    "id": "TEST-10",
    "file": "test/test_net_coolingcurve.py",
    "line": 62,
    "class": "other",
    "severity": "S3",
    "claim": "_expected_noncie reimplements get_dudt's non-CIE branch and calls the same interpolator object, so two of the three assertions in the file compare the implementation to itself; only the T-floor gate is genuinely tested.",
    "evidence": "Lines 62-69: `def _expected_noncie(params, T): \"\"\"Reproduce get_dudt's non-CIE branch arithmetic EXACTLY (same ops, same order, incl. the in-place /= round-trip) so equality is bit-for-bit.\"\"\" ... netcool = params[\"cStruc_net_nonCIE_interpolation\"].value; dudt = netcool([np.log10(nd_eff), np.log10(T), np.log10(ph_eff)])[0]; return -1 * dudt * cvt.dudt_cgs2au`. Used at line 96 `assert direct == _expected_noncie(params, T_mid)` and line 104 for T in (1e4, 3e4, 5e4).",
    "expected": "The T-floor assertions (deep == at_edge, direct != _dudt(1e4)) are the real content and are fine. The equality-to-mirror assertions should be replaced by, or supplemented with, a check against an independently loaded cooling table value.",
    "failure_scenario": "The non-CIE interpolation, the log10 argument ordering, the sign convention, or the dudt_cgs2au factor is wrong. Both sides of `direct == _expected_noncie(...)` are wrong identically and the test passes bit-for-bit.",
    "repro": "sed -n '62,69p;95,105p' test/test_net_coolingcurve.py",
    "confidence": "high"
  },
  {
    "id": "TEST-11",
    "file": "test/test_cooling_boost.py",
    "line": 45,
    "class": "other",
    "severity": "S4",
    "claim": "test_theta_target_is_double_count_free asserts the implementation's own expression, so it can only fail if effective_Lloss stops matching a formula the test copied from it.",
    "evidence": "Line 49: `assert eff == max(Lcool + Lleak, theta * Lmech)` where `eff = effective_Lloss(\"theta_target\", 1.0, theta, Lcool=Lcool, Lleak=Lleak, Lmech=Lmech)`. The second assertion, `assert eff < (Lcool + Lleak) + theta * Lmech`, is a real (if weak) property: it excludes the additive double-count. The rest of the file (lines 16-42) uses literal expected numbers and is sound.",
    "expected": "Keep the double-count-free property assertion; drop or relabel the max() mirror.",
    "failure_scenario": "Low impact — the sibling literal-value tests at lines 18-42 pin the same behaviour with independent numbers. Filed for completeness of the mirror inventory.",
    "repro": "sed -n '45,51p' test/test_cooling_boost.py",
    "confidence": "high"
  },
  {
    "id": "TEST-12",
    "file": "test/test_mu_audit_drift.py",
    "line": 81,
    "class": "other",
    "severity": "S3",
    "claim": "52 assertions across the suite check substrings and occurrence counts in production source text rather than behaviour; for several audit refinements they are the ONLY guard (the behavioural sibling being TEST-01's tautology).",
    "evidence": "test_mu_audit_drift.py carries 28: e.g. :102 `assert total == 11, f\"expected 11 refined HII-pressure sites, found {total}\"`; :147 `assert bub.count(\"Pb / ((params['mu_convert'].value / params['mu_ion'].value)\") == 5`; :251 `assert s.count(\"params['mu_convert'].value\") == 8`. test_sweep_jobs.py carries 21 (sbatch text), test_theta5_harvest.py 2, test_energy_collapse_snapshot.py 1.",
    "expected": "Where the claim is 'this coefficient is applied at these sites', a behavioural test that calls each site and checks the value is the right instrument; the text grep is at best a supplement.",
    "failure_scenario": "Two directions. False negative: the arithmetic inside all 11 sites is wrong but the text matches, and the test passes. False positive: `black .` (mandated by CLAUDE.md), a rename, or a behaviour-preserving refactor rewraps one of those expressions and the suite goes red for no physical reason — training future sessions to 'fix' the test.",
    "repro": "grep -c \"_src(\" test/test_mu_audit_drift.py ; grep -n \"\\.count(\" test/test_mu_audit_drift.py",
    "confidence": "high"
  },
  {
    "id": "TEST-13",
    "file": "test/test_betadelta_hybr.py",
    "line": 51,
    "class": "other",
    "severity": "S3",
    "claim": "29 of the 53 beta-delta tests run with the bubble physics monkeypatched out; the residual landscape is a synthetic closed form and get_bubbleproperties_pure is replaced by a function that raises if called.",
    "evidence": "test_betadelta_hybr.py:86-87 `monkeypatch.setattr(GBD, \"get_residual_pure\", pure); monkeypatch.setattr(GBD, \"get_residual_detailed\", detailed)` where pure/detailed are synthetic lambdas over (beta, delta). test_betadelta_solver.py:113-117 `def forbid_bubble_solve(monkeypatch): def bomb(*a, **k): raise AssertionError(...); monkeypatch.setattr(GBD, \"get_bubbleproperties_pure\", bomb)`. The two tests that drive hybr through real physics (test_betadelta_hybr_stress.py) are @pytest.mark.stress and deselected by pyproject.toml's addopts.",
    "expected": "This is a legitimate and well-documented design for testing the search algorithm. What is missing is the complementary non-stress test that runs the real residual on at least one captured state, so the default suite has some physics coverage of the solver it exercises hardest.",
    "failure_scenario": "A regression in get_residual_pure, get_residual_detailed or get_bubbleproperties_pure — the actual bubble physics the beta-delta solve depends on — is invisible to 29 of the 53 tests nominally covering that subsystem, and the 2 that would catch it are deselected by default. The pass count reads as 53 tests of the beta-delta physics; it is 53 tests of the beta-delta search.",
    "repro": "grep -c 'monkeypatch.setattr(GBD' test/test_betadelta_solver.py test/test_betadelta_hybr.py",
    "confidence": "high"
  },
  {
    "id": "TEST-14",
    "file": "test/test_simplify.py",
    "line": 52,
    "class": "numerical",
    "severity": "S4",
    "claim": "assert_endpoints_preserved uses np.isclose with defaulted tolerances (rtol=1e-5, atol=1e-8) to assert exact endpoint passthrough; any endpoint whose magnitude is below 1e-8 passes unconditionally.",
    "evidence": "Lines 52-59: `assert np.isclose(x_out[0], x_in[0]) ... assert np.isclose(y_out[-1], y_in[-1])`. Line 299 uses the same defaults to map an output point back to its input index: `if np.isclose(x[j], xi) and np.isclose(y[j], yi)`. _simplify selects input indices, so the correct assertion is `==`.",
    "expected": "`assert x_out[0] == x_in[0]` (and the three siblings) — exact, since the operation is an index selection. For line 299, an index-based construction rather than a value match.",
    "failure_scenario": "_simplify starts interpolating rather than selecting, or drops/reorders the endpoint by a hair. On any array whose values are below 1e-8 in code units the helper cannot fail. On line 299, two nearly-equal samples map to the wrong input index and the positional-order assertion still passes.",
    "repro": "python -c \"import numpy as np; print(np.isclose(0.0, 9e-9))\"  ->  True",
    "confidence": "medium"
  },
  {
    "id": "TEST-15",
    "file": "pyproject.toml",
    "line": 116,
    "class": "regime",
    "severity": "S3",
    "claim": "addopts deselects 9 tests, 5 of which are pure timing budgets; the remaining 4 are the ONLY tests in the repo that integrate past the energy->implicit boundary. The default suite's deepest physics reach is 100 simulated years.",
    "evidence": "pyproject.toml:116 `addopts = \"-v --tb=short -m 'not stress'\"`. `pytest -m stress --collect-only -q` -> 9 selected: test_simplify.py::TestTiming::test_runtime_budget[1000-0.1|10000-0.3|30000-0.6|100000-2.0] and ::test_subquadratic_scaling (5 timing), plus test_betadelta_hybr_stress::test_hybr_endtoend_no_crashes, ::test_hybr_implicit_converges_and_matches_golden, test_bubble_solver_stress::test_smoke_no_bubble_solver_failures, test_energy_collapse_snapshot::test_energy_collapse_emits_no_negative_Pb. test_run_smoke.py:44 sets `stop_t 1e-4` Myr.",
    "expected": "The 5 timing tests should carry their own mark (e.g. `perf`) so `-m 'not stress'` does not conflate 'slow performance benchmark' with 'the only integration coverage of phases 1b/1c/2'. At least one non-stress test should reach the implicit phase.",
    "failure_scenario": "The transition, implicit and momentum phases have no end-to-end coverage in the default invocation. The confirmed event-dispatch defect lives in exactly those phases. A developer running `pytest`, seeing 852 green, has no coverage of the code paths where three of the six confirmed defects live.",
    "repro": "python -m pytest -m stress --collect-only -q  ->  collected 861 items / 852 deselected / 9 selected",
    "confidence": "high"
  },
  {
    "id": "TEST-16",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 735,
    "class": "regime",
    "severity": "S2",
    "claim": "19 trinity/ modules are never imported by any test, including all three phase drivers that call the defective check_event_termination and the in-process sweep runner that consumes the non-injective run name.",
    "evidence": "Import-graph analysis over test/*.py: run_momentum_phase (931 lines, calls check_event_termination at :735), run_transition_phase (886, :653, plus gamma_adia at :505/:751/:836), run_energy_phase (408, :324), _input/sweep_runner (644), shell_structure/shell_structure (473, guarded only by source-text grep at test_mu_audit_drift.py:243), cloud_properties/mass_profile (627), phase0_init/get_InitCloudProp (666), phase1_energy/energy_phase_ODEs (431), trinity/main (370), plus 10 smaller. run_energy_implicit_phase is nominally covered but only via three small pure helpers (classify_energy_collapse, update_unconverged_streak, evaluate_r1_shadow) — the 1400-line loop containing the unconditional break at :1096 is never entered.",
    "expected": "At minimum, an importable seam per phase driver (the event-result handling is already pure enough to test directly) so the dispatch contract can be asserted without a full run.",
    "failure_scenario": "Three of the six confirmed defects live in code no test imports. The only thing that executes those 3,700 lines is test_run_smoke's subprocess, which runs 1e-4 Myr and asserts three floats.",
    "repro": "python - <<'EOF'\nimport re,pathlib\nimp=set()\nfor f in pathlib.Path('test').glob('test_*.py'):\n    imp|={m.group(1) for m in re.finditer(r'(?:from|import)\\s+(trinity[\\w.]*)', f.read_text())}\nprint('run_momentum_phase covered:', any('run_momentum_phase' in i for i in imp))\nEOF",
    "confidence": "high"
  },
  {
    "id": "TEST-17",
    "file": "test/test_betadelta_hybr_stress.py",
    "line": 57,
    "class": "numerical",
    "severity": "S3",
    "claim": "_GOLDEN is captured from a recording run of this code, by its own admission, and is deselected by default — so the hybr solver's only physics-bearing numerical expectation never runs in CI.",
    "evidence": "Lines 55-57: `# Golden accepted (beta, delta) at the first implicit-phase segments, recorded on the pinned numpy<2 / scipy<2 stack. FILLED FROM A RECORDING RUN.  _GOLDEN: list = [(0.759260, -0.035387), ...]`. The enclosing test carries @pytest.mark.stress (line 125) and pyproject.toml:116 deselects it.",
    "expected": "A captured golden is the right tool here (there is no analytic beta-delta root), but it should run somewhere automatic. Either move this one test out of the stress mark with a smaller stop_t, or document that `pytest -m stress` is a required pre-merge step.",
    "failure_scenario": "Any drift in the hybr solve's beta/delta trajectory ships green under `pytest`. And if the recording run was taken while a defect was active, the golden defends it.",
    "repro": "sed -n '55,60p;123,127p' test/test_betadelta_hybr_stress.py",
    "confidence": "high"
  },
  {
    "id": "TEST-18",
    "file": "test/test_cloudy_cli.py",
    "line": 35,
    "class": "other",
    "severity": "S4",
    "claim": "~45 assertions across four cloudy/reader test files take their expected values from outputs/mockOutput/mockFullrun, a committed captured run that CLAUDE.md explicitly says must not be treated as ground truth.",
    "evidence": "test_cloudy_cli.py:35 `MOCK_FULLRUN = Path(__file__).resolve().parents[1] / \"outputs\" / \"mockOutput\" / \"mockFullrun\"`; .gitignore:10-12 `outputs/*` then `!outputs/mockOutput/` (29 tracked files). Expectations derived from it include test_cloudy_cli.py:112 `assert prefix == \"4e3_sfe001_n5e2_PL0_170_momentum_t0p1482myr\"`, :151 `assert picks[0].index == 177`, test_cloudy_run_loader.py:227 `assert end[\"t_now_myr\"] == 0.300`, :232 `assert end[\"mCloud_msun\"] == pytest.approx(3.97e3)`, test_show_run.py:145 `assert \"2.510 pc\" in out`. CLAUDE.md: \"Generated / scratch — not source, do not tidy or treat as ground truth: outputs/ ...\".",
    "expected": "Correct scope (these test I/O plumbing, formatting and selection logic, not physics) but the fixture's status should be stated in the test docstrings, and no physics conclusion should ever cite these numbers.",
    "failure_scenario": "Low direct risk. The hazard is interpretive: a future session counting green tests over 'real run data' may take mockFullrun's trajectory as validated. It is not — it is one output of this code, produced at an unrecorded commit.",
    "repro": "git ls-files outputs/ | wc -l  ->  29 ; grep -n 'mockOutput' test/test_cloudy_cli.py",
    "confidence": "medium"
  },
  {
    "id": "TEST-19",
    "file": "test/test_cf_leak.py",
    "line": 32,
    "class": "other",
    "severity": "S4",
    "claim": "The suite's dominant pattern is mirror-of-implementation: the production formula is restated in the test body and compared at rtol=1e-12. These detect edits, not errors.",
    "evidence": "Six named instances. test_cf_leak.py:34 `expected = GAMMA/(GAMMA-1.0)*(1.0-Cf)*4.0*np.pi*R2**2*Pb*cs` vs get_bubbleParams.py:280 (identical). test_mu_audit_drift.py:208 `dndr_refined = mu_p_shell/mu_H/(kB*tion)*(dust + chi_sh*recomb)` vs get_shellODE. test_mu_audit_drift.py:285 (get_soundspeed). test_shell_overflow_guard.py:44 `expect_dphidr = -4*np.pi*r**2*chi_e*aB*n**2/Qi - n*sd*phi`. test_cooling_boost.py:49. test_residual_resample.py:100 (_reference_residual, docstring: 'Replicates production's residual formula and EVERY return branch ... using the production helpers').",
    "expected": "Where an independent oracle exists — a paper equation with its own numbers, an analytic limit, a dimensional cross-check, a second solver — use it. test_conventional_units.py (astropy), test_r1_bracket.py (analytic asymptote), test_dR2min_magic_number.py (Radau) and test_cf_leak.py:46 (cgs round-trip) show the house can do this.",
    "failure_scenario": "A physically wrong coefficient or sign in any mirrored formula is invisible: the mirror carries the same error and agrees to machine precision. The suite's rtol=1e-12 house style makes these look like the strongest tests in the repo when they are among the weakest as correctness evidence.",
    "repro": "diff <(sed -n '280p' trinity/bubble_structure/get_bubbleParams.py) <(sed -n '34p' test/test_cf_leak.py)",
    "confidence": "high"
  },
  {
    "id": "TEST-20",
    "file": "pyproject.toml",
    "line": 121,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "filterwarnings blanket-ignores DeprecationWarning, and there are no skips or xfails anywhere in the suite — so the only warning channel that would flag the pinned numpy/scipy stack drifting is closed.",
    "evidence": "pyproject.toml:120-122 `filterwarnings = [\"ignore::DeprecationWarning\"]`. grep for pytest.mark.skip / pytest.mark.xfail across test/*.py returns 0 hits; the only conditional skips are four pytest.importorskip(\"astropy.units\") calls at test_conventional_units.py:38,43,50,55 for a hard dependency.",
    "expected": "Scope the filter to the specific third-party warnings that are noisy, so a deprecation inside trinity/ or one that predicts a numpy 2.x behaviour change still surfaces. CLAUDE.md documents that numpy 2.1/2.2/2.4 already break the monotonic guard — deprecation warnings are the early signal for exactly that class of break.",
    "failure_scenario": "A scipy or numpy API TRINITY depends on is deprecated and then removed or silently changed in a patch release inside the allowed range (scipy<2, numpy<2). The suite runs green through the deprecation window and fails only at removal, with no advance notice. Zero skips/xfails is otherwise a genuinely good sign — nothing is being swept under a marker.",
    "repro": "grep -rn 'pytest.mark.skip\\|pytest.mark.xfail' test/*.py | wc -l  ->  0",
    "confidence": "medium"
  }
]
```
