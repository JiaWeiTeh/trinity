# n-consistency implementation plan (detailed, line-by-line)

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (verified 2026-06-16):** ✅ **SHIPPED** (verified 2026-06-16) — all phases (0–6 + A) landed in #657. (Before/after tables use pre-reorg flat filenames / pre-fix line numbers.)
- **Type:** plan — the detailed, line-by-line edit spec (phased, per-site before/after) that applies the `n ≡ n_H` audit; records what shipped.
- **Workstream:** `n-consistency/` — the "every number density n is hydrogen-nuclei n_H" convention & pressure terms.
- **Where it sits:** `audit.md` (physics rationale & finding) → **this** (exact edits, Phases 0–6 + Phase A) → terminal; pinned by `test/test_mu_audit_drift.py`. Shipped in #657.
- **Code it concerns:** `_input/` (`registry.py`, `read_param.py`, composition params), `phase1*`/`phase2*` pressure prefactors, `bubble_structure/bubble_luminosity.py`, `cooling/net_coolingcurve.py`, `shell_structure/`, `cloud_properties/bonnorEbertSphere.py`, `_functions/operations.py`.
- **Linked files & data:** sibling docs `audit.md`, `pressure-terms-audit.md`; code `trinity/_input/{registry,read_param}.py`, `trinity/bubble_structure/bubble_luminosity.py`, `trinity/cooling/net_coolingcurve.py`, `trinity/shell_structure/{get_shellODE,shell_structure}.py`, `trinity/_functions/operations.py`; test `test/test_mu_audit_drift.py`.

> **Audit status (2026-06-08):** **shipped in #657.** The status header correctly
> names the phases, but the **Phase 1 & Phase 3 before/after tables predate the
> Phase A singly-ionised-shell decision** and name the wrong μ/χ (`mu_ion`/
> `chi_e`) at the shell/HII-pressure sites, which now use `mu_ion_shell`/
> `chi_e_shell`. Line numbers throughout have drifted (the doc already disclaims
> this at the top of "Ground rules").

Companion to `n-consistency-audit.md` (physics rationale) — this is the
**exact edit spec**. Branch `hotfix/mu-audit`. Ground truth = the model paper
(`n ≡ n_H`). Every before-block below was read in source at audit time; line
numbers are as-of-audit and may drift as edits land.

> **Status — what actually shipped (`hotfix/mu-audit`):** Phases 0–3 and 5.1 as
> written below. **Phase 6** shipped as *6A* (the `get_soundspeed` docstring fix)
> plus **exposing `densBE_sigma`** and relabelling `densBE_Teff` as an *effective
> (turbulent)* temperature — the `μ_mol`/isothermal "6B" rewrite in §Phase 6 was
> **rejected** (it would only rescale a round-trip-cancelling diagnostic, and
> `μ_mol` makes it hotter, not cooler; see `n-consistency-audit.md`). **Phase A**
> (added after): the ~10⁴ K shell/HII region is now **singly-ionised He**
> (`Z_He_shell=1` → `mu_ion_shell`, `chi_e_shell`); the hot bubble stays doubly
> ionised. All pinned by `test/test_mu_audit_drift.py`.

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

> **Phase A correction (shipped in #657):** every site in this table is the
> **HII / ionised-shell** back-pressure, which is **singly-ionised** He. The
> shipped code therefore uses `params['mu_ion_shell'].value` (not `mu_ion`) at
> all of them — i.e. the "after" cells read
> `(params['mu_convert'].value/params['mu_ion_shell'].value) * …`. The factor
> is `mu_convert/mu_ion_shell = 2.2` (not 2.3), so the effect is ×(2.2/2.0)=×1.10
> (not ×1.15). The bubble's `mu_ion` (2.3) is unchanged and is **not** used at
> these pressure sites.

**Effect:** P_HII / P_ext ×(2.2/2.0)=×1.10 (singly-ionised shell, Phase A).
**Verify:** full suite; assert the factor equals `mu_convert/mu_ion_shell`
(≈2.2) in a unit test.

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

## Phase 3 — shell to true n_H + `chi_e` recombination (ATOMIC — VERIFIED against source @ HEAD 8873cac)

**Re-derived vs paper (no assumptions on sign or code):** the radiation force
`f_rad = −(1/4πr²c)·d/dr(L_n e^{−τ_d}+L_iφ_f)`; substituting `dτ/dr=n_sh σ_d`
and `dφ/dr=−(4πr²/Q_i)·chi_e·α_B n_sh² − n_sh σ_d φ` gives
`f_rad = +(n_sh σ_d/4πr²c)(L_n e^{−τ}+L_iφ) + chi_e·α_B n_sh² L_i/(Q_i c)`
— **both terms positive**, matching the code's `+`-sign `dndr`. So the code
sign/structure are correct; Phase-3 edits are **magnitude-only** (μ prefactor +
`chi_e`), sign-preserving.

**`get_shellODE.py`** — add to the unpack block (after `mu_p` ~L64):
```python
mu_H  = params['mu_convert'].value
chi_e = params['chi_e'].value
```
> **Phase A correction (shipped in #657):** the shell is **singly-ionised**, so
> the shipped unpack binds the *shell* composition to these locals:
> `mu_p = params['mu_ion_shell'].value` and `chi_e = params['chi_e_shell'].value`
> (instead of `mu_ion`/`chi_e`). The ODE expressions below are unchanged — they
> use the local names `mu_p`/`chi_e` — but those now carry `14/22` and `1.1`,
> not `14/23` and `1.2`.
| line | before | after |
|---|---|---|
| `:93` | `dndr = mu_p/mu_n/(k_B * t_ion) * (` | `dndr = mu_p/mu_H/(k_B * t_ion) * (` |
| `:95` | `+ nShell**2 * alpha_B * Li / Qi / c` | `+ chi_e * nShell**2 * alpha_B * Li / Qi / c` |
| `:98` | `dphidr = - 4 * np.pi * r**2 * alpha_B * nShell**2 / Qi - …` | `dphidr = - 4 * np.pi * r**2 * chi_e * alpha_B * nShell**2 / Qi - …` |
| `:118` | `dndr = 1/(k_B * t_neu) * (` | `dndr = mu_n/mu_H/(k_B * t_neu) * (` |

Dust terms (`:94`,`:119`) and `dτ/dr` (`:100`,`:122`) are **unchanged** — they
already use `nShell`, which now *means* n_H (so dust opacity becomes correct).

**`shell_structure.py`** (11 edits; `:298` no-change):
| line | before | after | role |
|---|---|---|---|
| `:115` | `params['mu_ion'].value / params['mu_atom'].value` | `params['mu_ion'].value / params['mu_convert'].value` | BC `nShell0=(μ_p/μ_H)Pb/kT` (Eq nShell0) |
| `:135` | `4*np.pi*params['caseB_alpha'].value*nShell0**2` | `4*np.pi*params['chi_e'].value*params['caseB_alpha'].value*nShell0**2` | Strömgren integration bound |
| `:167` | `nShell_arr[1:] * params['mu_atom'].value` | `nShell_arr[1:] * params['mu_convert'].value` | ion mass |
| `:239` | `4.0*np.pi*params['caseB_alpha'].value*_vol_ion` | `4.0*np.pi*params['chi_e'].value*params['caseB_alpha'].value*_vol_ion` | `n_IF_Str` (Eq nIF_Str) |
| `:253` | `nShell_arr_ion * params['mu_atom'].value` | `nShell_arr_ion * params['mu_convert'].value` | ion grav ρ |
| `:273` | `params['caseB_alpha'].value * nShell_arr_ion[:-1]**2` | `params['chi_e'].value * params['caseB_alpha'].value * nShell_arr_ion[:-1]**2` | φ_hydrogen recomb |
| `:298` | `nShell0 * mu_atom/mu_ion * T_ion/T_neu` | **NO CHANGE** | I-front jump — μ_H cancels (re-derived: `n_neu=n_ion·(μ_n/μ_p)·T_ion/T_neu`) |
| `:324` | `nShell_arr[1:] * params['mu_atom'].value` | `nShell_arr[1:] * params['mu_convert'].value` | neutral mass |
| `:357` | `nShell_arr_neu * params['mu_atom'].value` | `nShell_arr_neu * params['mu_convert'].value` | neutral grav ρ |
| `:380` | `params['mu_atom'].value * np.sum(nShell_arr_ion…)` | `params['mu_convert'].value * …` | τ_IR/κ_IR mass column, ion **[MISSED in earlier audit]** |
| `:381` | `params['mu_atom'].value * np.sum(nShell_arr_neu…)` | `params['mu_convert'].value * …` | τ_IR/κ_IR mass column, neu **[MISSED]** |
| `:386` | `params['mu_atom'].value * np.sum(nShell_arr_ion…)` | `params['mu_convert'].value * …` | τ_IR/κ_IR mass column, ion-only **[MISSED]** |

> **Phase A correction (shipped in #657):** the shell is **singly-ionised**, so
> the shipped `shell_structure.py` uses the *shell* composition wherever the
> table above names `mu_ion`/`chi_e`: `:115` divides by `mu_ion_shell` (BC
> `nShell0=(μ_p,shell/μ_H)Pb/kT`, factor `mu_ion_shell/mu_convert`), and the
> Strömgren/recombination weights at `:135`, `:239`, `:273` use
> `params['chi_e_shell'].value` (= 1.1, not 1.2). The I-front jump `:298` uses
> `mu_ion_shell` (still `NO CHANGE` to the μ-cancelling structure). The mass /
> τ_IR weights (`:167,:253,:324,:357,:380,:381,:386`) use `mu_convert` (μ_H) and
> are **unaffected** by the split.

**Scope confirmed:** all `shell_structure.py` sites are inside
`shell_structure_pure(params)`; `get_shellODE` unpacks from `params`. `chi_e`
(shell: `chi_e_shell`) present at runtime (Phase 0/A). `n_IF_Str` is capped at
`shell_n0` (`:242`) — both n_H after the change; it feeds `P_HII` (already
Phase-1-correct).

**Atomic — ONE commit.** BC + ODE prefactors + all mass/τ weights redefine
`nShell` from `ρ/μ_n` to `n_H=ρ/μ_H`; `chi_e` recomb/Strömgren terms are only
correct once `nShell=n_H`. Any partial application is physically wrong.

**Effects (CORRECTED — NOT mass-invariant):** σ_d is *per H nucleus*, so
`dτ_d/dr=n_sh σ_d` currently runs on `nShell≈1.1 n_H`; converting to true n_H
lowers dust opacity ~10% and **shifts the shell profile, mass, τ_IR and the
I-front by order μ_H/μ_n≈1.1**. Recomb barely moves (old `n_old²≈1.21 n_H²`
vs new `chi_e·n_H²=1.2 n_H²`, ~1%); Strömgren `n_IF_Str ×1/√1.2≈0.913`. (My
earlier "mass invariant" note was **wrong** — the old code was internally
inconsistent, not a clean rescaling. Caught by re-derivation.)
> **Phase A correction (#657):** these numbers assume `chi_e=1.2`. With the
> shipped singly-ionised shell, `chi_e_shell=1.1`, so the recomb term is
> `1.1 n_H²` and the Strömgren scaling is `n_IF_Str ×1/√1.1≈0.953` (not 0.913).
Downstream:
`dlaw.py:11` CLOUDY export (`log10(n_H)`) becomes truthful.

**Verify:** deterministic suite (407) + smoke ×2; closed-form checks
`nShell0=(μ_p/μ_H)Pb/kT` and `ρ_edge=nShell0·μ_H=μ_p·Pb/kT`;
`n_IF_Str ×0.913` vs Phase-2 baseline; log `shell_mass`/`R_IF` deltas
(expect ~10% from the dust correction, direction as above).

---

## Phase 5 — housekeeping (re-verified @ HEAD 149f76a; no assumptions)

**5.1 Delete dead `get_shellParams.py`.** `get_nShell0` is defined there and
**never imported or called** anywhere (grep across `trinity/`+`test/` returns
only its own `def`). Its formula `mu_atom/mu_ion · Pb/(k_B T_ion)` is both the
*reciprocal* of the live BC and on the wrong μ (μ_n vs μ_H) — a pure landmine.
The file contains nothing else. **Action:** `git rm
trinity/shell_structure/get_shellParams.py`. **Verify:** no remaining
references; full suite green.

**5.2 BE EOS μ — CORRECTED: do NOT change to μ_n. (Earlier plan was wrong.)**
Re-derivation from source:
- In `compute_BE_sphere`, `c_s` is derived from `M`, `ρ_core`, `m_dim`
  (`:401-408`); **the EOS μ never enters `c_s`** — only `ρ_core = n_core·mu`
  (`:402`), the *mass* density, which correctly uses `mu_convert = μ_H`.
- `T_eff` is back-computed from `c_s` via μ,γ (`:430`); `r2xi`/`xi2r`
  reconstruct `c_s` from `T_eff` via the **same** μ,γ (`:603`,`:643`). All three
  use `mu_convert`+`gamma_adia`, so **μ and γ cancel exactly** in
  `c_s→T_eff→c_s`. The cloud structure (`ξ=√(4πGρ_core/c_s²)·r`, `n(r)`,
  `r_out`, mass) depends only on `c_s` and `ρ_core` → **independent of the EOS
  μ/γ choice** (`density_profile` reaches it via `be_r2xi`).
- Consequences:
  1. Changing the EOS μ moves **only the reported diagnostic `densBE_Teff`**,
     never any dynamics → **not** a results-correctness fix and **outside the
     n_H audit** (the n→ρ mass conversion already uses μ_H).
  2. `μ_n` is wrong regardless: the paper's cloud is **molecular** (GMC /
     prestellar cores / Barnard 68), so mean mass per particle is
     **`μ_mol=14/6`**, not `μ_n=14/11`.
  3. The paper's BE EOS is **isothermal** `c_s²=k_BT/μ` (no γ); the code carries
     `γ=γ_adia`. A literal temperature would also drop γ.
- **Recommendation:** leave the code as-is (zero result change); optionally add
  a one-line comment that `densBE_Teff` is an *effective* (`mu_convert`,`γ_adia`)
  temperature that cancels in the r↔ξ round-trip — not a literal gas
  temperature. *Only if* a physical `T_eff` diagnostic is wanted: switch all
  three sites (`:430`,`:603`,`:643`) **consistently** to `μ_mol` and drop γ
  (changes the printed `T_eff` only, structure unchanged). **Never use `μ_n`,
  and never change only some sites — that breaks the round-trip and corrupts
  the BE profile.**

---

## Phase 6 — sound-speed cleanup (VERIFIED @ HEAD 1fa10ae; do not assume)

Audited every `c_s`. **Only one dynamical sound speed:** `operations.get_soundspeed`
(`mu_ion` if T>1e4 else `mu_atom` = mean mass per particle; verified numerically
to 1e-12). All dynamical consumers (bubble leak `get_bubbleParams:302`, transition
sound-crossing `run_transition_phase:235,520`, energy-phase leak) route through it.
**Consistent and correct — no dynamical change.** The only μ-outlier is the BE
`c_s↔T_eff` diagnostic round-trip (`mu_convert`), which cancels.

### 6A — `get_soundspeed` docstring only (operations.py:177-197) — ZERO behavior change
The function uses `gamma_adia` (adiabatic) — correct for a hot-bubble sound-crossing —
but the docstring says "isothermal", and "Units: Myr/pc" is inverted
(`v_cms2au = cm/s→pc/Myr`, so c_s is **pc/Myr**). Fixes (comments only):
- `:179` "isothermal soundspeed" → "adiabatic soundspeed".
- `:189` "Units: Myr/pc" → "Units: pc/Myr".
- clarify μ is `mu_ion` (T>1e4) / `mu_atom` (else), i.e. mean mass per particle.
**No code line changes.** Verify: value identical before/after; suite green.

### 6B — REJECTED (μ_mol); replaced by expose-σ + relabel. ✓ IMPLEMENTED (09cdfe6)
**Why μ_mol is wrong (computed, not assumed):** for a real BE param (M=10⁶ M⊙,
n_core=10⁴ cm⁻³) the code gives `c_s = 10.5 km/s`, `densBE_Teff ≈ 1.1×10⁴ K`. The
thermal molecular sound speed at 10 K is 0.19 km/s — the BE `c_s` is **56× thermal**:
it is the **turbulent support velocity dispersion** (Larson-law for a ~16 pc GMC),
not a thermal sound speed. A thermal BE sphere at 10 K has a Jeans/BE mass ~1 M⊙
(Barnard-68 scale); you cannot make a 10⁶ M⊙ BE sphere thermal. So `T_eff` is high
*because the support is turbulent*, and **no μ fixes it** — `μ_mol (2.33) >
μ_convert (1.4)` makes `T_eff` **2.78× hotter** (3.1×10⁴ K), the opposite of the
goal. The earlier "make `T_eff` physical via μ" premise was false.

**What was done instead (no μ/γ/structure change):**
- New derived param **`densBE_sigma = c_s [km/s]`** (mirrors `densBE_Teff`:
  `derived_init`, `active_when=_active_densBE`, `run_const`), set in both BE paths.
- **Relabel** `densBE_Teff` as an *effective (turbulent)* temperature (encodes `c_s`
  for the r↔ξ round-trip), and fix `BESphereResult.c_s` units (`pc/Myr`→`cm/s`).
- `μ`/`γ` untouched ⇒ `densBE_Teff`, `rCloud`, `n(r)` **byte-identical** (verified).
- Tests: `test_registry` active_when set + `test_materialize_runtime` densBE block;
  `test_mu_audit_drift` gains 6A + 6B checks (σ exposed; `densBE_Teff` still on the
  ORIGINAL μ_convert+γ — i.e. μ_mol was *not* applied).
- Validated: 420 suite + 13 anti-drift green; full BE run writes `densBE_sigma` to
  metadata (4.74 km/s, consistent with `densBE_Teff`).

**Open modelling question (author's call, not a code bug):** using the BE *shape*
for whole-GMC masses with turbulent `c_s` is a defensible idealisation but differs
from the paper's dense-*core* motivation. Out of scope for the n-audit.

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
