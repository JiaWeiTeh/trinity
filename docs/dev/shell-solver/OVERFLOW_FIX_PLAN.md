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

**Status (re-verified 2026-06-18 on `main` @ `c8f0d35`):** 🔵 **PLANNED, not implemented.**
Root cause verified by three independent investigations + a captured real solve. Recommended fix:
**evaluate the shell RHS in cgs (exact identity reconditioning)**. No production code changed.

**Where this sits:** this doc **is item §F3 of `docs/dev/performance/HOTPATH_PLAN.md`** (the hot-path
audit's "shell-ODE conditioning" item, which was *descoped to here* on 2026-06-18 and is owned by this
file). HOTPATH §F3 independently re-derived and confirmed the synthesis below — *finite (precision-only)
at the front, real `inf` overflow at the pole, entirely in the discarded tail* — after its own first
"doesn't overflow" take was falsified by `harness/verify_overflow.py`. Sibling: `MIGRATION_PLAN.md`
(the odeint→solve_ivp study) — see its retraction note.

**Units resolved (2026-06-18, was a subagent contradiction):** `caseB_alpha` is stored in **AU**
(`2.782e-55 pc³/Myr`), not cgs — it goes through the blanket `convert2au(unit)` at `read_param.py:261-263`
(unit `cm**3 * s**-1`, `registry.py:339`). So the current RHS is unit-consistent and the cgs-rescale is a
**true identity**; to use `alpha_B` in a cgs RHS, multiply by `1/convert2au('cm**3 * s**-1') ≈ 9.31e41`
(→ `2.59e-13 cm³/s`). The earlier "already cgs in params" note (§4) was wrong and is corrected below.

---

## ⚠️ EMPIRICAL FINDING (2026-06-18) — cgs-rescale does NOT silence the flood; the recommendation flipped

The de-risk matrix (monkeypatched variants on real `simple_cluster`, `harness/get_shellODE_variants.py`
+ `harness/verify_overflow.py`) **falsified the cgs-first recommendation on its first cell:**

| variant | front_idx | ovf_idx | overflow_warns | silences flood? |
|---|---|---|---|---|
| baseline (`odeint`+`mxstep`) | 4 | 26 | 1 | — |
| **cgs-rescale RHS (V1)** | 4 | **26** | **1** | **NO — identical to baseline** |
| **φ>0 freeze-past-front guard (V2)** | 4 | **−1** | **0** | **YES** |
| clip `nShell` in RHS (V4) | 4 | −1 | 0 | yes (crude) |

**Why cgs fails:** it is an *exact identity* (verified 7e-16), so the **state** `nShell` — still integrated
in AU — follows the *same* finite-radius pole to `inf` in the discarded tail. cgs reconditions the
*intermediate* `n²` but cannot stop the integrated state diverging. The overflow is a **pole in the
state**, not a precision loss, so the only fixes that work **stop integrating into the tail**
(freeze/terminate at the front) or cap the state.

**Revised stance:** the root-cause/flood fix is **φ-guard or terminate-at-front (`solve_ivp` φ-event)**,
not cgs. cgs-rescale is demoted to an *optional conditioning/correctness* item (it makes the
mixed-unit `caseB_alpha`·`n²` sites consistent and improves precision) but is **not** the warning fix.
The §4 ranking and §5 recommendation below are updated accordingly. The validation matrix (configs ×
the *flood-fixing* ideas, accuracy + efficiency) is running to pick between φ-guard and terminate.

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
- The recombination terms `± chi_e · nShell² · …` (`get_shellODE.py:97`, `:100`) make `dn/dr ∝ +n²`
  a **finite-radius pole** (`n(r)=n0/(1−A·n0·Δr)`). At the front `nShell² ≈ 9e130` is still *finite*;
  but a few grid steps into the discarded tail `nShell` blows past **~1.3e154**, so `nShell²` (and the
  derivative, further amplified by the `1/(k_B·t_ion) ≈ 10⁵⁵` prefactor) **overflows float64
  (1.8e308) → `inf`/`nan`** → LSODA forced to machine-precision steps → the `t+h=t` flood. (Earlier
  drafts said "`nShell²≈10¹²⁸` overflows float64" — imprecise: `10¹²⁸ < 10³⁰⁸`. The overflow is the
  pole reached ~20 steps past the front, not the front value. Corrected 2026-06-18 from a captured
  real solve — see §9.)
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

*(Updated 2026-06-18 after the empirical finding above — cgs demoted, φ-guard promoted.)*

| # | option | silences flood? (measured) | used-region change | risk | effort | verdict |
|---|---|---|---|---|---|---|
| **1** | **φ>0 freeze-past-front guard** | **YES** (`ovf→−1, warns→0`) | ≤0.8%/≤3.7% front shift (to quantify) | LOW | ~1–3 lines | **RECOMMENDED (flood fix)** |
| 2 | `solve_ivp` + terminal φ-event | **YES** (warns→0 all 6 cfgs; `n` rel ≤1e-8) | none on prefix | MED | medium | STRONG ALT (**net-FASTER on overflow cfgs**: sfe0.3 1.2× / sfe0.6 1.7× blended, 4.2–4.4× energy; net-slower only on no-flood cfgs; mass-limited gap only in no-flood mock — see §#3) |
| 3 | clip `nShell` in RHS | YES (`ovf→−1`) | none if clip > used values | LOW | ~1 line | crude fallback (arbitrary threshold) |
| 4 | **CGS-rescale the RHS** | **NO** (identical to baseline) | none (exact identity 7e-16) | LOW–MED | ~½ day | **NOT the flood fix** — optional conditioning only |
| 5 | Log-space `u=ln(nShell)` | no (`e^{2u}=n²` returns) | RHS physics touched | MED–HIGH | high | REJECTED |
| 6 | Silence `ODEintWarning`/Fortran stdout | hides only | none | LOW | trivial | band-aid fallback |
| 7 | `mxstep=50000` (shipped) | no | none (bit-identical) | — | done | revert (moot) |

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
| `alpha_B` (`caseB_alpha`) | `× 1/cvt.convert2au('cm**3*s**-1')` (≈9.31e41) | **stored in AU** (2.782e-55 pc³/Myr) → cgs 2.59e-13; do NOT treat as already-cgs |
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

**Full code-unit `n²` / `caseB_alpha` site inventory** (the cgs treatment should be consistent across
all of them, but only the first two *overflow* — the rest are finite, precision-only, and do not affect
the consumed result; listed so a future visit sees the whole pattern):

| site | expression | magnitude | overflows? | scope |
|---|---|---|---|---|
| `get_shellODE.py:97` | `+ chi_e·nShell²·alpha_B·Li/Qi/c` (dndr) | pole → `inf` | **YES** | **#1 — must fix** |
| `get_shellODE.py:100` | `−4πr²·chi_e·alpha_B·nShell²/Qi …` (dphidr) | pole → `inf` | **YES** | **#1 — must fix** |
| `shell_structure.py:144` | `max_shellRadius = (3·Qi/(4π·chi_e·caseB_alpha·nShell0²))^⅓` | `n0²`~1e130 | no (finite) | optional consistency |
| `shell_structure.py:248` | `n_IF_Str` denom `4π·chi_e·caseB_alpha·_vol_ion` (Strömgren) | finite | no | optional consistency |
| `shell_structure.py:282` | `phi_hydrogen = Σ(−4πr²/Qi·chi_e·caseB_alpha·nShell²·dr)` (used region) | `n²`~1e130 | no (finite) | optional consistency |

Decision: **#1 (the RHS) is the required fix** — it is the only place that overflows. The three
`shell_structure.py` sites are finite and their consumed results are correct, so they are **not**
required to silence the flood; reconcile them in the same pass *only if* it's clean (they share the
same `caseB_alpha`(au)·`nShell²`(au) form), otherwise leave them and note here. Do **not** let them
expand the surgical scope of the overflow fix.

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
gives an explicit `sol.success` contract (mirrors `bubble_luminosity.py:106-166,520-522`).

**Re-measured 2026-06-18** (`data/eval_terminate.csv`, from the 6-config `*_matrix_*` replays;
`harness/eval_terminate.py`; fresh `simple_cluster` energy-phase repro confirms it):

- **Clears the flood — YES, on every config.** `V_lsoda_event` overflow_warns_total = **0** across all
  6 configs (baseline odeint: 35 sfe0.3 / 54 sfe0.6). The φ-event terminates at the front
  (n_pts_out 4–57 of ~1000), so it never integrates the discarded overflow tail. **It is the EVENT
  that fixes it, not the solver swap:** the no-event drop-in `V_lsoda_teval`
  (`solve_ivp` over the full grid) emits *more* warns than baseline (106 sfe0.3 / 162 sfe0.6).
- **Used-region accuracy:** `n` rel ≤1.0e-8, `phi` rel ≤6.3e-6, `tau` (front-point pair) rel ≤9.4e-9
  vs baseline odeint on the consumed prefix — tight, as the event root-finds the front. (The raw
  `max_rel_diff_tau` column shows ~1e286 — a divide-by-zero artifact at the `tau0==0` IC row, **not**
  a fidelity loss; production excludes the front τ point anyway.) `nonfinite_tail_solves = 0`.
- **NET wall time is CONFIG-DEPENDENT — the prior "net-slower over a realistic run (energy-phase-only
  win)" is WRONG for the overflow configs it is meant to fix.** In the energy phase the event is
  **~4.2–4.4× FASTER** on sfe0.3/sfe0.6 (baseline grinds the overflow tail ~10ms; event ~3ms). Net over
  a 20:100 energy:implicit run: sfe0.3 **1.21×**, sfe0.6 **1.74×** — *faster*. The "net-slower" picture
  holds only for the **non-overflow** configs (probe/steep/dense ≈0.3–0.4× blended, mock_hybr 0.16×),
  where baseline odeint is already sub-ms and `solve_ivp` carries a ~3ms Python/event-overhead floor.
  Caveat: absolute speedup is timing-noisy (baseline ms swings 3–8× with machine load → energy event_x
  measured 3–5× in the committed sweep, 18–31× under a loaded repro); the **sign** (event faster in the
  energy overflow phase) is stable, the magnitude is not.
- **Insufficient ALONE — still true, but only where there's no flood to fix.** A φ-only event never
  fires on *mass-limited* ionised slices (`idx_phi==-1`). Per `eval_terminate.csv` the mass-limited
  fraction of ionised solves is **0% on every overflow config** (sfe0.3/sfe0.6/probe/steep/dense) and
  **63% on mock_hybr** (95% in its energy phase). So on the configs that actually flood, the φ-event
  fires on 100% of ionised slices and suffices; the mass-limited gap matters only for the
  no-overflow mock regime, which would need a 2nd mass-condition event-state for completeness.
- Must use `t_eval=rShell_arr` (NOT `t_eval` + `dense_output` **together** — that bubble-precedent
  config crashes `ValueError: ts must be strictly increasing` on the shell micro-grid, verified;
  step ~1.3e-8 pc collapses LSODA's breakpoints). A standalone `dense_output=True` then `sol.sol(grid)`
  does *not* crash (`V_lsoda_dense` ok=120/120), but adds nothing over `t_eval` and is slower.

Larger change than the φ-guard; defer unless the broader robustness migration (`MIGRATION_PLAN.md`)
is pursued. **Verdict vs φ-guard:** for the flood fix specifically, the φ-event is *more* than the
φ-guard needs — it root-finds the exact front (tighter ~1e-9 accuracy) and is net-faster on the
overflow configs, but it swaps the integrator and leaves a mass-limited gap. The **φ>0 freeze-guard
(#1)** clears the same flood with ~1–3 lines, keeps `odeint`, and has no mass-limited gap (it freezes
on `phi<=0` regardless of why the slice ends), so it stays the recommended minimal fix; #3 is the
strong alternative if the migration happens anyway.

### #4 — Log-space (REJECTED)
`u=ln(nShell)` tames the state but `du/dr` still contains `e^{2u}=n²` → same overflow; most
invasive; touches the audited RHS. Not worth it.

## 5. Recommendation

1. **Flood fix = #1 (φ>0 freeze-past-front guard)** — empirically clears the overflow (`ovf→−1,
   warns→0`), ~1–3 lines, keeps `odeint`. Pick φ-guard vs #2 (terminate-at-front) on the validation
   matrix (accuracy of the used region/downstream + efficiency across configs).
2. **cgs-rescale is NOT the flood fix** (measured). Keep it only as an *optional* conditioning/correctness
   pass for the mixed-unit `caseB_alpha`·`n²` sites — separate, low priority, do not block the flood fix on it.
3. **Revert the `mxstep` change** and **correct** the §P0-matrix / `insights.html` / plot-5 overclaim.
   HOTPATH §F3 explicitly calls for this revert.
4. **Defer the full `solve_ivp` migration** (`MIGRATION_PLAN.md`) unless robustness motivates it.

### Phased rollout (gated — mirrors HOTPATH_PLAN P0–P4)

- **S0 — factor-pin test (sim-free, DO FIRST).** Add to `test/test_unit_conversions.py`: build the
  constants exactly as `read_param` does and assert the new cgs RHS == current AU RHS to **rtol 1e-12**
  across non-overflow inputs (`n_cgs ∈ {1e2,1e4,1e6}`, several r/φ/τ), for **both** ionised and neutral
  branches. **Gate G0:** every factor pinned independently (this is what catches the `Qi`/`caseB_alpha`
  traps). No slow run.
- **S1 — implement #1.** Rewrite `get_shellODE.py` ionised + neutral RHS in cgs with module-level
  factor constants. **Gate G1:** S0 test green; `pytest` (+ `-m stress`) byte-unchanged where it should
  be; `test_mu_audit_drift.py` (RHS-form pin) still green.
- **S2 — equivalence + overflow-gone.** Capture-replay (`harness/`) current `odeint` vs cgs-RHS on
  identical `(y0, grid, params)`. **Gate G2:** used-region `n/φ/τ` within round-off; the overflow row
  (`ovf_idx`) is gone (no non-finite); `overflow_warns → 0`. Commit `data/cgs_rhs_comparison.csv` (💾).
- **S3 — φ>0 guard (#2), separate commit.** **Gate G3:** front selection stops at last `φ>0`; consumed
  scalars unchanged beyond round-off.
- **S4 — cleanup.** Revert `mxstep` (`shell_structure.py:35,167,326`); regenerate `insights.html`
  + plot 5; update `MIGRATION_PLAN.md` and HOTPATH §F3 status. **Gate G4:** no doc still claims
  "mxstep fixes the warning."

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

### Empirical validation matrix (configs × solution-ideas — the hybr-style de-risk)

Before committing to one fix, evaluate each candidate idea as a **monkeypatched variant** (production
untouched) across a regime sweep, measuring **end accuracy AND efficiency** — exactly how
`archive/betadelta/` (hybr) and `MIGRATION_PLAN.md` de-risked their changes. Status: **running
2026-06-18** (subagents).

- **Ideas:** `V0` baseline (current `odeint`+`mxstep`, the reference) · `V1` cgs-rescale RHS ·
  `V2` φ>0 front-truncation guard · `V3` `solve_ivp`+terminal φ-event (re-measured head-to-head).
- **Configs (degenerate → realistic):** `simple_cluster`, `sfe0.6` (overflow-heavy) ·
  `probe_typical_hybr`, `steep`, `dense_flat`, `mock_hybr` (realistic, low/no overflow).
- **Common CSV schema** (`data/eval_<idea>.csv`, so cells are comparable):
  `config, idea, phase, n_solves, used_rel_n_max, used_rel_phi_max, used_rel_tau_max, nIF_rel,
  RIF_rel, fesc_rel, overflow_warns, nonfinite_tail, ms_per_solve, ms_per_solve_V0, total_run_s,
  total_run_s_V0, endtoend_final_maxrel, notes`.
- **Gates:** accuracy — used-region + consumed scalars within round-off of `V0` (target rel ≤1e-9 for
  the identity ideas V1; ≤ the documented ≤0.8%/≤3.7% for V2); robustness — `overflow_warns→0`,
  `nonfinite_tail→0`; efficiency — no per-solve or per-run regression (V1 expected ~neutral; V3 expected
  net-slower per `MIGRATION_PLAN.md`, re-confirmed here). Commit every CSV (💾).

## 7. Open decisions (maintainer)

- OK to evaluate the RHS in cgs (units = known bug class; mitigated by the 1e-12 factor test)?
- Revert `mxstep`, or keep it + a `warnings` filter as belt-and-braces?
- Land the φ>0 overshoot guard now, or as a separate tracked issue?
- `solve_ivp` migration: defer (recommended) or pursue for robustness?

## 8. Key references

- **Parent plan:** `docs/dev/performance/HOTPATH_PLAN.md` §F3 (this doc is that item, descoped here).
- Overflow site: `trinity/shell_structure/get_shellODE.py:95-100` (`±n²` at 97/100), φ-clamp `:91`,
  `alpha_B` local `:69` (misleading `#cm3/s (au)` comment — value is AU).
- Finite (precision-only) code-unit `n²`/`caseB_alpha` sites: `shell_structure.py:144` (`max_shellRadius`),
  `:248` (`n_IF_Str` Strömgren), `:282` (`phi_hydrogen`).
- Truncation/front: `trinity/shell_structure/shell_structure.py:180-218`; odeint calls `:165-168`
  (ion), `:324-327` (neu); `mxstep` const `:35`.
- Consumed-scalar clamps: `shell_structure.py:229,242-251,395`; P_HII from `n_IF_Str` not raw n:
  `phase1_energy/run_energy_phase.py:188-190`, `phase2_momentum/run_momentum_phase.py:627-636`;
  `n_IF` diagnostic-only `registry.py:445`.
- Units: `trinity/_functions/unit_conversions.py:88` (`ndens_cgs2au`); blanket AU conversion at load
  `read_param.py:255-263`; `caseB_alpha` spec (unit `cm**3 * s**-1`, default 2.59e-13) `registry.py:339`
  → stored AU 2.782e-55; `dust_sigma` Z-scaling/σ=0 `read_param.py:366-369`.
- RHS pin test `test/test_mu_audit_drift.py:188-209`; derivation
  `docs/dev/archive/n-consistency/implementation-plan.md:268-274`.
- Bubble precedent (for #3): `trinity/bubble_structure/bubble_luminosity.py:106-166,520-522`.
- Sibling study + matrix: `docs/dev/shell-solver/MIGRATION_PLAN.md`, `data/master_table.csv`,
  earlier overflow capture `data/replay_comparison.csv`.

## 9. Reproduce / this session's confirmations

- `python run.py param/simple_cluster.param` → `get_shellODE.py:97 overflow encountered` (root cause,
  reproduces on `main`).
- **Captured first ionised solve** (`harness/verify_overflow.py`, monkeypatches `odeint`, bails after
  the first ionised slice; ~30 s, verified 2026-06-18):
  | field | value |
  |---|---|
  | `y0_n` (slice start, code units) | `1.337e65` |
  | `nShell` at front | `2.974e65` code-units = **`1.011e10` cm⁻³ physical** |
  | `front_idx` (first `phi<=1e-9`) | **4** |
  | `ovf_idx` (first non-finite row) | **26** |
  | `n_max_finite` before `inf` | `6.65e67` |
  | grid points integrated | 1000 |
  | overflow RuntimeWarnings / solve | 1 |
  Proves: overflow real; `nShell² @ front = 8.8e130` is *finite* (the pole overflows ~20 steps later);
  overflow row (26) is past the consumed front (4) → discarded tail; physical front density ~1e10 cm⁻³.
- This container's SciPy does NOT surface `lsoda--` Fortran lines (stiff-probe → 0), so the flood
  itself is verified via the root-cause overflow, not the Fortran text.
