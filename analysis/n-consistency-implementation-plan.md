# n-consistency implementation plan (detailed, line-by-line)

Companion to `n-consistency-audit.md` (physics rationale) — this is the
**exact edit spec**. Branch `hotfix/mu-audit`. Ground truth = the model paper
(`n ≡ n_H`). Every before-block below was read in source at audit time; line
numbers are as-of-audit and may drift as edits land.

**Ground rules**
- Do not assume code comments are correct; the paper is authoritative.
- Phase 0 must keep μ defaults **byte-identical** (use `Fraction` arithmetic).
- Verify after every phase: `pytest test/` (baseline = 407 passed + 1 *known-flaky*
  `test_run_smoke`), plus 2× smoke re-runs. Commit per phase.
- Atomic phases (2, 3) must land as a single commit — partial application
  breaks mass conservation.

**Canonical factors** (all derived from `x_He`, `Z_He`):
`μ_H=mu_convert`, `μ_n=mu_atom`, `μ_p=mu_ion`; `μ_H/μ_p=mu_convert/mu_ion=2.3`;
`μ_H/μ_n=mu_convert/mu_atom=1.1`; `chi_e≡n_e/n_H=1+Z_He·x_He=1.2`;
bubble `n_H=(μ_p/μ_H)·Pb/(k_B T)=(mu_ion/mu_convert)·Pb/(k_B T)`.

---

## Phase 0 — composition foundation (no physics result change)

**Goal:** introduce `x_He`, `Z_He`; derive `mu_convert/atom/ion/mol` + new
`chi_e` from them with exact `Fraction` arithmetic (byte-identical μ).

**0.1 `registry.py`** — add two input specs (next to the μ block ~L313) and
update the four μ `info` strings to note they are derived:
```python
ParamSpec(name='x_He', default='0.1', info='Helium-to-hydrogen number ratio n_He/n_H. Composition source of truth; mu_* and chi_e derive from this.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
ParamSpec(name='Z_He', default='2',   info='Helium ionisation state in ionised gas (2 = doubly ionised). Sets electron factor chi_e = 1 + Z_He*x_He.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
```
`chi_e` is **created in Step 6** (like `mCluster`), NOT declared in
default.param. If a snapshot/metadata key is wanted, add a sibling spec
`chi_e` (category `input_constants`, unit None) and mutate rather than create.

**0.2 `default.param`** — add under the μ block (~L200):
```
# INFO: Helium-to-hydrogen number ratio n_He/n_H. mu_* and chi_e derive from this.
x_He    0.1
# INFO: Helium ionisation state in ionised gas (2 = doubly ionised).
Z_He    2
```
(Keep the four `mu_*` lines; their values become derived/overwritten — see 0.4.)

**0.3 `read_param.py` Step 6** — insert at the top of the Step-6 block
(after L301), before the dust block:
```python
from fractions import Fraction
# Composition is the single source of truth: derive mu_* and chi_e from
# x_He, Z_He. Exact-rational arithmetic keeps mu defaults byte-identical.
xHe = Fraction(params['x_He'].value).limit_denominator(10**6)   # 0.1 -> 1/10
ZHe = Fraction(params['Z_He'].value).limit_denominator(10**6)   # 2.0 -> 2
muH    = 1 + 4*xHe                          # mass per H nucleus [m_H]
mu_n   = muH / (1 + xHe)                    # neutral mean mass/particle
mu_p   = muH / (2 + xHe*(1 + ZHe))          # ionised mean mass/particle
mu_mol = muH / (Fraction(1, 2) + xHe)       # molecular mean mass/particle
chi_e  = 1 + ZHe*xHe                        # n_e / n_H
mH_au  = cvt.convert2au('m_H')              # m_H in Msun
params['mu_convert'].value = float(muH)    * mH_au
params['mu_atom'].value    = float(mu_n)   * mH_au
params['mu_ion'].value     = float(mu_p)   * mH_au
params['mu_mol'].value     = float(mu_mol) * mH_au
params['chi_e'] = DescribedItem(value=float(chi_e),
    info='Electron-per-H factor n_e/n_H = 1 + Z_He*x_He (derived).',
    ori_units='dimensionless')
```
Mutating `.value` on the four μ is allowed by the anti-stomp guard (only
*replacing* the DescribedItem is forbidden); creating `chi_e` fresh is the
`mCluster` pattern.

**0.4 Open sub-decision (recommend A):**
- **A (low risk):** keep `mu_*` in default.param, overwrite in Step 6 (above);
  add an INFO note that a user-set μ is ignored (or a validator warning).
- **B (cleaner):** remove `mu_*` from default.param, make them `derived_init`;
  touches `gen_default_param.py` + `test_registry`. Defer to follow-up.

**0.5 Verify:** assert `mu_*` post-load == current `float(Fraction('14/11'))`
etc. **bit-for-bit**; `chi_e == 1.2`; full suite green; smoke 2×. No physics
number moves (μ unchanged, `chi_e` not yet consumed).

---

## Phase 1 — ionised-gas pressure prefactor `2.0 → μ_H/μ_p` (11 sites — IMPLEMENTED)

Replace the pure-H factor `2.0` with `mu_convert/mu_ion` (= μ_H/μ_p). Each
site already feeds the correct n_H (`n_r` from the cloud profile, or
`n_IF_Str`). `+= PISM*k_B` untouched.

> **Correction (found during line-by-line implementation):** the audit table
> below listed **6** sites; an exhaustive re-grep found **11**. The 5 missed
> were `P_HII`/`P_HII_f` in the *segment* and *final-value* functions:
> `run_momentum_phase.py:630`, `run_transition_phase.py:562` & `:840`,
> `run_energy_implicit_phase.py:691` & `:1006` — same `2.0 * n_IF_Str(_f) *
> k_B * T_ion` form, same fix, `params` in scope at each. All 11 done;
> `grep` confirms zero `2*n` pressure patterns remain.

| file:line | before | after |
|---|---|---|
| `energy_phase_ODEs.py:55` | `P_ion = 2.0 * n_r * params['k_B'].value * params['TShell_ion'].value` | `P_ion = (params['mu_convert'].value/params['mu_ion'].value) * n_r * params['k_B'].value * params['TShell_ion'].value` |
| `run_energy_phase.py:195` | `P_HII = 2.0 * n_IF_Str * params['k_B'].value * params['TShell_ion'].value` | `P_HII = (params['mu_convert'].value/params['mu_ion'].value) * n_IF_Str * params['k_B'].value * params['TShell_ion'].value` |
| `run_energy_implicit_phase.py:307` | `P_ext = 2.0 * n_r * k_B * TShell_ion` | `P_ext = (mu_convert/mu_ion) * n_r * k_B * TShell_ion` † |
| `run_transition_phase.py:303` | `P_ext = 2.0 * n_r * k_B * TShell_ion` | same † |
| `run_momentum_phase.py:244` | `P_ext = 2.0 * n_r * k_B * TShell_ion` | same † |
| `run_momentum_phase.py:428` | `P_ext = 2.0 * n_r * k_B * snapshot.TShell_ion` | `... (mu_convert/mu_ion) ...` † |

† these functions use local `k_B`/`TShell_ion`; pull `mu_convert`,`mu_ion`
from `params`/`snapshot` in the same scope (verify the local handle before
editing — `snapshot` in the `_pure` momentum fn, `params` elsewhere).

**Effect:** P_HII / P_ext ×(2.3/2.0)=×1.15. **Verify:** full suite; assert the
factor equals `mu_convert/mu_ion` (≈2.3) in a unit test.

---

## Phase 2 — bubble interior `n→n_H`, `ρ→μ_H·n`, CIE cooling `×chi_e` (ATOMIC — VERIFIED against source)

`bubble_luminosity.py` density (5 sites): `Pb/(2*k_B*T)` →
`(mu_ion/mu_convert)*Pb/(k_B*T)`.
```
:338  n_array     = Pb / (2 * params['k_B'].value * T_array)
:390  n_CIEswitch = Pb / (2 * params['k_B'].value * _CIEswitch)
:465  n_cond      = Pb / (2 * params['k_B'].value * T_cond)
:498  n_interm    = Pb / (2 * params['k_B'].value * T_interm)
:948  ndens       = Pb / (2 * params['k_B'].value * T)
```
→ each `Pb / (2 * params['k_B'].value * X)` becomes
`(params['mu_ion'].value/params['mu_convert'].value) * Pb / (params['k_B'].value * X)`.

Mass (1 site): `:970  rho_new = n[::-1] * params['mu_ion'].value` →
`rho_new = n[::-1] * params['mu_convert'].value`. (Also fix the wrong inline
comment `# Mass density`.)

CIE cooling `×chi_e` (rate = n_e n_H Λ = chi_e n_H² Λ; Gnat-Ferland is per
n_e·n_H — confirmed):
```
bubble_luminosity.py:411  integrand_bubble = n_bubble**2 * Lambda_bubble * 4*pi*r**2
bubble_luminosity.py:520  integrand_int    = n_interm[mask]**2 * Lambda_int * 4*pi*r**2
net_coolingcurve.py:126   dudt    = ndens**2 * Lambda_CIE
net_coolingcurve.py:149   dudt_CIE= (ndens**2 * Lambda)
```
→ prepend `params['chi_e'].value *` (bubble) / `params_dict['chi_e'].value *`
(net_coolingcurve). Non-CIE table path is volumetric on n_H and needs **no**
chi_e — Phase-2 n_H fix suffices there. Conduction μ (`:595,:927,:933`) is
already μ_p — **no change**.

**Why atomic:** `n` and `ρ` must flip together (else `ρ=n_H·μ_ion` is ~2×
wrong); CIE `chi_e` must accompany `n=n_H` (else off by chi_e). **Effects:**
`n` ×(2/2.3)=×0.87; `ρ`,`bubble_mass`,self-gravity ×2.0 net; CIE cooling
×(0.87²·1.2)=×0.91. **Verify:** suite; smoke 2×; assert
`ρ == μ_ion·Pb/(k_B T)·…` closed form.

### Phase 2 — source-verification addendum (no assumptions, re-read at commit `0ec91ba`)

**Physics (paper):** the hot bubble is fully ionised, `n ≡ n_H`.
`P_b = n_tot k_B T`, `n_tot = ρ/μ_p`, `ρ = μ_H n_H`
⟹ `n_H = (μ_p/μ_H) P_b/(k_B T) = P_b / ((μ_H/μ_p) k_B T)`. The factor
`μ_H/μ_p = mu_convert/mu_ion = 2.3` is the **same** as Phase 1 (here we
**divide** by it, P→n). Cooling: `dU/dt = −n_e n_H Λ = −chi_e n_H² Λ`
(Gnat-Ferland solar is normalised per `n_e n_H` — your confirmation).

**Exhaustive site list (grep-verified, not the stale audit — Phase 1 taught
that lesson):** 5 P→n (`bubble_luminosity.py:338,390,465,498,948`), 1 ρ
(`:970`), 4 CIE (`:411`, `:520`, `net_coolingcurve.py:126,149`). No others.
`:970` hides behind a trailing `# Mass density` comment (first filter missed
it); it IS present and in scope.

**Scope confirmed by reading the enclosing defs:** the P→n + ρ + CIE-bubble
sites are inside `get_bubbleproperties_pure(params)`,
`_get_bubble_ODE(..., params, ...)`, and `_get_mass_and_grav(n, r, params)`;
`get_dudt` is called at `:951` with the full `params`, so
`params_dict['chi_e']` is reachable in `net_coolingcurve`. Every handle is
`params` (no snapshot) ⟹ `params['chi_e'/'mu_convert'/'mu_ion'].value` all work.

**Exact edit form (P→n):** replace the literal `2` with
`(params['mu_convert'].value / params['mu_ion'].value)` inside the existing
`Pb / ( … * params['k_B'].value * X )` — minimal diff, mirrors Phase 1.

**No-change, re-verified vs paper:** `_get_init_dMdt` (`:591,:595`) and
`_get_bubble_ODE_initial_conditions` (`:923,:927,:933`) use `mu_ion` as the
mean-mass-per-particle μ_p in Weaver Eqs (Mbdot)/(Tprofile)/(vprofile) —
**correct, untouched**. Non-CIE table path (`:471-478`, `:510-516`,
`get_dudt:118`) is volumetric on `n_H` ⟹ fixed by the n→n_H change alone,
**no** chi_e.

**Downstream of changed quantities:** `n_array → bubble_n_arr` (diagnostic
output); `_get_mass_and_grav(n_array) → bubble_mass` → shell gravity base
`mBubble` in `shell_structure.py` (small vs shell mass) + bubble self-gravity;
`get_dudt`/`L_bubble` → bubble energy ODE (Pb, transition timing). **No
pressure term consumes bubble `n`** (P_HII/P_ext use shell `n_IF_Str` / cloud
`n_r`), so Phase 2 is independent of Phase 1.

**Atomic — one commit:** 2A (5) + 2B (1) + 2C (4) together; any intermediate
state is physically wrong (`ρ=n_H·μ_p`, or CIE missing chi_e).

**Expected shifts (sanity targets):** `n_H ×0.87` (2/2.3); `ρ`, `bubble_mass`,
self-grav `×2.0`; CIE `L_cool ×(1.2·0.87²)=×0.91`; non-CIE recomputed at the
correct `n_H`. Bubble T-structure, `Pb`, transition timing move (real, expected).

**Verify:** deterministic suite (407, no golden values) + smoke 2×; closed-form
unit check `n_H=(μ_p/μ_H)Pb/(k_BT)` ⟹ `ρ=μ_H·n_H=μ_p·Pb/(k_BT)`; log
`bubble_mass` (≈×2) and `L_cool` (≈×0.91) deltas to confirm direction.

---

## Phase 3 — shell to true n_H + `chi_e` recombination (ATOMIC)

**`get_shellODE.py`** — add to the unpack block (~L63):
```python
mu_H  = params['mu_convert'].value
chi_e = params['chi_e'].value
```
| line | before | after |
|---|---|---|
| `:93` | `dndr = mu_p/mu_n/(k_B * t_ion) * (` | `dndr = mu_p/mu_H/(k_B * t_ion) * (` |
| `:95` | `+ nShell**2 * alpha_B * Li / Qi / c` | `+ chi_e * nShell**2 * alpha_B * Li / Qi / c` |
| `:98` | `dphidr = - 4*pi*r**2 * alpha_B * nShell**2 / Qi - nShell*sigma_dust*phi` | `dphidr = - 4*pi*r**2 * chi_e * alpha_B * nShell**2 / Qi - nShell*sigma_dust*phi` |
| `:118` | `dndr = 1/(k_B * t_neu) * (` | `dndr = mu_n/mu_H/(k_B * t_neu) * (` |

(`mu_n` stays the existing `params['mu_atom'].value`; `mu_p` stays `mu_ion`.)

**`shell_structure.py`**
| line | before | after |
|---|---|---|
| `:115` | `nShell0 = (params['mu_ion'].value / params['mu_atom'].value / (k_B*T_ion) * Pb)` | `… params['mu_ion'].value / params['mu_convert'].value …` |
| `:135` | `(3*Qi/(4*pi*caseB_alpha*nShell0**2))**(1/3)` | `(3*Qi/(4*pi*chi_e*caseB_alpha*nShell0**2))**(1/3)` ‡ |
| `:167` | `nShell_arr[1:] * params['mu_atom'].value` | `… params['mu_convert'].value` |
| `:237-239` | `np.sqrt(3.0*_Qi_absorbed / (4π*caseB_alpha*_vol_ion))` | `np.sqrt(3.0*_Qi_absorbed / (4π*chi_e*caseB_alpha*_vol_ion))` |
| `:253` | `grav_ion_rho = nShell_arr_ion * params['mu_atom'].value` | `… params['mu_convert'].value` |
| `:273` | `params['caseB_alpha'].value * nShell_arr_ion[:-1]**2` | `params['chi_e'].value * params['caseB_alpha'].value * nShell_arr_ion[:-1]**2` |
| `:298` | `nShell0 * mu_atom/mu_ion * T_ion/T_neu` | **NO CHANGE** (μ_H cancels — pressure-continuity jump, verified) |
| `:324` | `nShell_arr[1:] * params['mu_atom'].value` | `… params['mu_convert'].value` |
| `:357` | `grav_neu_rho = nShell_arr_neu * params['mu_atom'].value` | `… params['mu_convert'].value` |

‡ `chi_e` reference needs `params['chi_e'].value`.

**Why atomic:** BC (`:115`) + ODE prefactors (`get_shellODE :93,:118`) +
mass weights (`:167,:253,:324,:357`) redefine `nShell` from `ρ/μ_n` to
`n_H=ρ/μ_H` — they are one coherent change; and the `chi_e` recombination
terms are only correct once `nShell=n_H`. **Effects:** total shell **mass is
invariant** (variable renormalises); the I-front position and `n_IF_Str`
shift (`n_IF_Str` ×1/√1.2≈0.91). Downstream bonus: `dlaw.py:11` CLOUDY export
(`log10(n_H)`) becomes truthful. **Verify:** suite; smoke 2×; assert shell
total mass unchanged vs Phase-2 baseline within integrator tolerance.

---

## Phase 5 — housekeeping

**5.1 dead `get_shellParams.py:30`** (never imported): either delete the
module/function or fix to the paper BC `params['mu_ion'].value /
params['mu_convert'].value / (k_B*T_ion) * Pb`. Recommend delete (it shadows
the live `shell_structure.py:115`).

**5.2 BE EOS μ → μ_n (mass μ stays μ_H).** In `bonnorEbertSphere.py` each
helper holds **one** `mu = params['mu_convert'].value` used for **both**
`c_s=√(γk_BT/μ)` (EOS) and `rho_core=n·μ` (mass). Split them: EOS μ → `μ_n`
(`mu_atom`), mass μ → `μ_H` (`mu_convert`). Sites (verify exact lines):
`:430` `T_eff = mu*MSUN_TO_G*c_s²/(γ k_B)` (EOS→mu_atom);
`:603` & `:643` `c_s=√(γ K_B_CGS T_eff/(mu*MSUN_TO_G))` (EOS→mu_atom);
`:608` `rho_core=n_core*mu*…` (mass→keep mu_convert). Introduce a second local
`mu_eos = params['mu_atom'].value` rather than swapping the single `mu`.
**Effect:** changes BE-profile physical scale (`c_s↔T_eff↔ξ`) for BE runs only;
power-law runs untouched. Confirm intended before landing.

---

## Dependency / ordering

```
Phase 0 (foundation: chi_e, μ) ─┬─> Phase 1 (pressure prefactor)
                                ├─> Phase 2 (bubble, needs chi_e)   [atomic]
                                ├─> Phase 3 (shell, needs chi_e)     [atomic]
                                └─> Phase 5.2 (BE μ, needs mu_atom split)
Phase 5.1 (dead code) independent.
```
Each phase: own commit, full suite + 2× smoke, record the expected numeric
shift so the diff is explainable. Phases 2 and 3 each land as **one** commit.
