# Shell-ODE float64 overflow & LSODA `t+h=t` warning flood — fix plan

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

**Status (2026-06-18):** 🔵 **PLANNED, not implemented.** Root cause verified by three
independent investigations. Recommended fix: **evaluate the shell RHS in cgs (exact
identity reconditioning)**. No code change made for this plan. Sibling doc:
`MIGRATION_PLAN.md` (the odeint→solve_ivp migration study) — see the correction note there.

---

## 1. The symptom (the user's report)

`python run.py param/simple_cluster.param` floods stdout with LSODA Fortran warnings:

```
lsoda--  warning..internal t (=r1) and h (=r2) are
       such that in the machine, t + h = t on the next step  (h = step size). solver will continue anyway
```

repeated ~10× per `odeint` call across thousands of calls. **Confirmed present on `main`** after
PR #691 (i.e. with `mxstep=50000` already shipped). The `get_shellODE.py` `overflow encountered
in scalar power/multiply` RuntimeWarning reproduces every run in this container (the Fortran
`t+h=t` lines themselves are not surfaced by this container's SciPy build — verified with a stiff
probe — but the underlying overflow that causes them is).

## 2. Root cause (verified 3×, 2026-06-18)

- The shell-structure ODE (`trinity/shell_structure/get_shellODE.py`) is evaluated in **code units
  `[Msun, pc, Myr]`**, so `nShell` is a number density in **`1/pc³`**. With
  `cvt.ndens_cgs2au ≈ 2.94e55` (`unit_conversions.py:88`), a physical ionisation-front density of
  ~10⁹ cm⁻³ is stored as **`nShell ≈ 10⁶⁴` code units**.
- The recombination terms `± chi_e · nShell² · …` (`get_shellODE.py:97`, `:100`) then form
  `nShell² ≈ 10¹²⁸`, which **overflows float64 (max 1.8e308)** as `nShell` rises past the front
  → `inf`/`nan` derivatives → LSODA forced to machine-precision steps → the `t+h=t` flood.
- **The physics is correct, not a bug.** `dn/dr ∝ +n²` is a *derived* consequence of the
  radiation-force gradient (`docs/dev/archive/n-consistency/implementation-plan.md:268-274`) and is
  pinned by `test/test_mu_audit_drift.py:188-209` to rtol 1e-12. The steep rise is a genuine
  finite-radius pole (`n(r)=n0/(1−A n0 Δr)`); `n≈10⁶⁴` ≙ ~10⁹ cm⁻³ is physically plausible.
- **The overflow is a unit-conditioning artifact**, and it lives **entirely in the discarded
  post-front tail**: `shell_structure.py` truncates each 1000-point slice at the first row where
  `phi <= 1e-9` or cumulative mass ≥ `mShell_end` (`:182`), i.e. at the front (~idx 4–10), while the
  overflow occurs deep in the tail (~idx 26–537) that is thrown away.
- **Regime-specific:** the degenerate `simple_cluster`/`sfe0.6` configs warn on ~100% of
  energy-phase solves; realistic configs (`typical`/`steep`/`dense_flat`/`mock_hybr`) warn ≤20%
  (see `MIGRATION_PLAN.md` §P0-matrix, `data/master_table.csv`).

## 3. Correction to the shipped `mxstep` change (PR #691)

`mxstep=50000` targeted a **different** warning — the Python `ODEintWarning("Excess work done on
this call")` (the mxstep step-count ceiling) — **not** the user's `t+h=t` Fortran step-underflow
flood. It is bit-identical (rel_n=0) but does **not** silence the flood (the flood is driven by the
overflow, which `mxstep` cannot prevent). The matrix harness recorded **0 Fortran `lsoda--` lines in
every sampled cell** because its sampling (first 20 energy + 100 implicit solves) did not cover the
solves that surface the flood. **Therefore the §P0-matrix / `insights.html` / plot-5 claim "warning
fixed for free by mxstep" is inaccurate and must be corrected.** `mxstep` becomes unnecessary once
the root cause (overflow) is fixed.

## 4. Candidate fixes (ranked)

| # | option | fixes the overflow? | used-region change | risk | effort | verdict |
|---|---|---|---|---|---|---|
| **1** | **CGS-rescale the RHS** (the maintainer's idea) | **Yes, at the root** | **none** (exact identity, 1e-16) | LOW–MED | ~½ day | **RECOMMENDED** |
| 2 | **φ>0 front-truncation guard** | indirectly (stops the tail) | fixes a pre-existing overshoot | LOW | ~1–2 lines | **COMPLEMENTARY** |
| 3 | `solve_ivp` + terminal φ-event | yes (never integrates tail) + fixes overshoot | none on prefix | MED | medium | OPTIONAL (robustness; net-slower; insufficient alone) |
| 4 | Log-space `u=ln(nShell)` | **no** (`e^{2u}=n²` returns) | RHS physics touched | MED–HIGH | high | REJECTED |
| 5 | Silence `ODEintWarning` / Fortran stdout | no (hides only) | none | LOW | trivial | band-aid fallback |
| 6 | `mxstep=50000` (shipped) | no | none (bit-identical) | — | done | revert (moot under #1) |

### #1 — CGS-rescale (RECOMMENDED, = the maintainer's out-of-the-box idea)
Evaluate the ionised (and, for symmetry, neutral) RHS in **cgs** so intermediates stay ~10⁰–10¹⁶,
then convert the derivatives back to code units. **Verified to be an exact algebraic identity**
(native-AU vs cgs-then-convert agree to rel ~1e-16; `dphidr` bit-identical). Must convert the
**whole** ionised RHS (the `dndr`/`dphidr` sums mix `n` and `n²` magnitude scales). Keeps `odeint`
(no integrator/structure change), so it's fast and quiet with **zero change to the consumed region**.

**Exact factors** (all verified against `cvt.*`; inputs are stored in AU by `read_param.py:262-263`):

| quantity | AU→cgs factor | note |
|---|---|---|
| `nShell` 1/pc³→1/cm³ | `cvt.ndens_au2cgs` (3.40e-56) | |
| `r` pc→cm | `cvt.pc2cm` (3.09e18) | |
| `phi`,`tau` | 1 | dimensionless |
| `alpha_B` | `1/cvt.convert2au('cm**3*s**-1')` | **already cgs in params**; verify, don't double-convert |
| `sigma_dust` | `1/cvt.convert2au('cm**2')` | handles the `Z<dust_noZ ⇒ σ=0` case fine |
| `c` | `cvt.v_au2cms` | |
| `k_B` | `cvt.k_B_au2cgs` | Kelvin treated as factor 1 (consistent) |
| `Li`,`Ln` | `cvt.L_au2cgs` (6.02e29) | |
| `Qi` | `cvt.s2Myr` (3.17e-14) | **TRAP: `convert2au('1/Myr')` raises** — do NOT use it |
| `mu_p/mu_H` | 1 | dimensionless ratio — pass through unchanged |
| derivative `dndr` cgs→AU | `cvt.convert2au('cm**-4')` (9.07e73) | hoist to module const |
| `dphidr`,`dtaudr` cgs→AU | `cvt.pc2cm` (3.09e18) | |

**Footguns:** the `Qi` `1/Myr` label trap (use `cvt.s2Myr`); precompute all factors as
**module-level constants** (the RHS runs ~10³–10⁴×/solve — don't call `convert2au` per-call). Sketch
and full detail in the CGS sub-agent findings (this session).

### #2 — φ>0 front-truncation guard (COMPLEMENTARY, independent of the overflow)
The integrated `phi` overshoots **negative** at the selected front grid point (e.g. −0.0157); the
in-RHS clamp `phi = max(0,phi)` (`get_shellODE.py:91`) only fixes the derivative, not the stored
state. Today this is **contained** — every consumed scalar is independently clamped
(`f_esc_ion = max(0,…)` `:229`; `n_IF_Str` is analytic Strömgren capped `:242-251`; raw `n_IF` is
diagnostic-only `registry.py:445`; `tau` excludes the front point `:395`). So it is a pre-existing
**cosmetic** wart, not a corruption. A small guard (select the last `phi>0` point, or stop the solve
there) makes that robustness explicit and guarantees no residual tail blow-up. Do it as its own
change, not bundled into the overflow fix.

### #3 — `solve_ivp` + terminal φ-event (OPTIONAL)
Physically the cleanest boundary (stop the ionised solve at `phi=0`); also cures the overshoot and
gives an explicit `sol.success` contract (mirrors `bubble_luminosity.py:106-166,520-522`). **But**:
(a) the matrix shows the φ-event is **net-slower over a realistic run** (energy-phase-only win); and
(b) it is **insufficient alone** — 7/40–39/40 slices are *mass-limited*, where a φ-only event never
fires, so a mass-condition event-state would also be needed. Larger change; defer unless the broader
robustness migration (`MIGRATION_PLAN.md`) is pursued. Must use `t_eval=rShell_arr` (NOT
`dense_output` — it crashes 0/40 on the shell micro-grid) to preserve the uniform-grid invariant.

### #4 — Log-space (REJECTED)
`u=ln(nShell)` tames the state but `du/dr` still contains `e^{2u}=n²` → same overflow; most
invasive; touches the audited RHS. Not worth it.

## 5. Recommendation

1. **Implement #1 (CGS-rescale)** — root-cause fix, exact identity, surgical, keeps `odeint`.
2. **Add #2 (φ>0 guard)** as a separate small commit — fix the overshoot explicitly.
3. **Revert the `mxstep` change** (moot once #1 lands) and **correct** the §P0-matrix / `insights.html`
   / plot-5 overclaim.
4. **Defer #3** (solve_ivp migration) — robustness-only, net-slower, insufficient alone.

## 6. Verification (sim-free where possible)

1. **Unit-factor regression test** (`test/test_unit_conversions.py`): build constants as
   `read_param` does, assert new cgs RHS == current AU RHS to **rtol 1e-12** across non-overflow
   inputs (`n_cgs ∈ {1e2,1e4,1e6}`, several r/φ/τ). Pins every factor; catches the `Qi` trap. Primary
   gate, needs no slow run.
2. **Capture-replay** (`docs/dev/shell-solver/harness/`): current `odeint` vs cgs-RHS on identical
   `(y0, grid, params)` over the used prefix → within round-off; `excess_work`/`underflow` flags flip
   1→0; the overflow tail disappears. Commit `data/cgs_rhs_comparison.csv` (💾).
3. `pytest` full suite (+ `-m stress`) green before/after; `test_conventional_units.py` and
   `test_mu_audit_drift.py` independently guard factors and RHS form.

## 7. Open decisions (maintainer)

- OK to evaluate the RHS in cgs (units = known bug class; mitigated by the 1e-12 factor test)?
- Revert `mxstep`, or keep it + a `warnings` filter as belt-and-braces?
- Land the φ>0 overshoot guard now, or as a separate tracked issue?
- `solve_ivp` migration: defer (recommended) or pursue for robustness?

## 8. Key references

- Overflow site: `trinity/shell_structure/get_shellODE.py:95-100` (`±n²` at 97/100), φ-clamp `:91`.
- Truncation/front: `trinity/shell_structure/shell_structure.py:180-218`; odeint calls `:165-168`
  (ion), `:324-327` (neu); `mxstep` const `:35`.
- Consumed-scalar clamps: `shell_structure.py:229,242-251,395`; P_HII from `n_IF_Str` not raw n:
  `phase1_energy/run_energy_phase.py:188-190`, `phase2_momentum/run_momentum_phase.py:627-636`;
  `n_IF` diagnostic-only `registry.py:445`.
- Units: `trinity/_functions/unit_conversions.py:88` (`ndens_cgs2au`); AU at load
  `read_param.py:262-263`; `caseB_alpha`/`dust_sigma` `read_param.py:350-369`.
- RHS pin test `test/test_mu_audit_drift.py:188-209`; derivation
  `docs/dev/archive/n-consistency/implementation-plan.md:268-274`.
- Bubble precedent (for #3): `trinity/bubble_structure/bubble_luminosity.py:106-166,520-522`.
- Sibling study + matrix: `docs/dev/shell-solver/MIGRATION_PLAN.md`, `data/master_table.csv`,
  earlier overflow capture `data/replay_comparison.csv`.

## 9. Reproduce / this session's confirmations

- `python run.py param/simple_cluster.param` → `get_shellODE.py:97 overflow encountered` (root cause,
  reproduces on `main`).
- Probe (front idx vs overflow idx, monkeypatched `odeint`): front at the φ→0 crossing, overflow
  several steps into the discarded tail (front idx 4 / overflow idx 26 for the captured `y0`).
- This container's SciPy does NOT surface `lsoda--` Fortran lines (stiff-probe → 0), so the flood
  itself is verified via the root-cause overflow, not the Fortran text.
