# S8 shell structure — Lens A (what the code does)

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

**Scope covered.** `trinity/shell_structure/get_shellODE.py` lines 1–153 (complete) and
`trinity/shell_structure/shell_structure.py` lines 1–473 (complete). `__init__.py` is a single
blank line (no re-exports, no side effects) — note that `shell_structure.py:24` does
`from trinity.shell_structure import get_shellODE`, i.e. it binds the *module*, and calls
`get_shellODE.get_shellODE` at :166 and :325.

**Shared-file exception used:** yes — `S1_units_helpers/code/_functions/unit_conversions.py`, read
only to pin down the code's unit system (see §3). All source read with comments/docstrings blanked.

---

## 1. The shell ODE system

Integration variable: **r, the spherical radius, increasing outward** from the bubble's outer
shell boundary `R2`. Every `t`-array handed to `odeint` is built by `np.arange(start, start+slice,
step)` with `step > 0` in the nominal case, so the direction is outward. (See §5 for the one path
where the step can become ≤ 0.)

### 1a. Ionised branch (`is_ionised == True`, `get_shellODE.py:94–125`)

State vector, unpacked at `:96`:

$$\mathbf{y} = \big(n(r),\ \phi(r),\ \tau(r)\big)$$

* `n` — number density of hydrogen nuclei (the ODE's `nShell`); mass density is recovered
  downstream as `ρ = n · mu_convert` (`shell_structure.py:176, 262`), so `n = ρ/µ_H`.
* `φ` — dimensionless ionising-photon survival fraction; initialised to `1` at `shell_structure.py:119`.
* `τ` — dimensionless dust optical depth; initialised to `0` at `:120`.

Two pre-processing steps mutate the *local copies* used to build the RHS (they do **not** feed back
into the integrator state):

* `:100` `n ← min(n, 10^{120})`
* `:111` `φ ← max(0, φ)`
* `:103–106` `E ≡ e^{-\tau}` is replaced by the exact `0` when `τ > 500`.

With `σ_d ≡ dust_sigma`, `µ_p ≡ mu_ion_shell`, `µ_n ≡ mu_atom`, `µ_H ≡ mu_convert`,
`χ_e ≡ chi_e_shell`, `α_B ≡ caseB_alpha`, `T_i ≡ TShell_ion`, `T_n ≡ TShell_neu`:

$$\frac{dn}{dr}=\frac{\mu_p}{\mu_H k_B T_i}\left[\underbrace{\frac{n\,\sigma_d}{4\pi r^2 c}\Big(L_n E+L_i\,\phi\Big)}_{\text{dust momentum}}+\underbrace{\frac{\chi_e\,\alpha_B\,n^2}{c}\frac{L_i}{Q_i}}_{\text{recombination momentum}}\right]$$

$$\frac{d\phi}{dr}=-\frac{4\pi r^2\chi_e\alpha_B n^2}{Q_i}-n\,\sigma_d\,\phi$$

$$\frac{d\tau}{dr}=n\,\sigma_d\,f_{\rm cover}$$

Substituted to inputs, the bracket in `dn/dr` is a radiation-pressure gradient `dP_rad/dr`; the
prefactor `µ_p/(µ_H k_B T_i)` is exactly the same combination used to convert a pressure into a
number density at `shell_structure.py:124–125` (`n = µ_p P /(µ_H k_B T_i)`). So the ODE is
"isothermal shell in radiative-pressure hydrostatic balance": `dn/dr = (µ_p/µ_H k_B T_i)·dP_rad/dr`.

Sign structure: every term of `dn/dr` is **non-negative** (all of `σ_d, L_n, L_i, χ_e, α_B, Q_i, c`
positive, `E ≥ 0`, `φ` clamped ≥ 0), so **n(r) is strictly monotonically increasing outward**. Both
terms of `dφ/dr` are ≤ 0, so **φ decreases monotonically** and has no fixed point at 0 — see §5.

Return at `:125` is the tuple `(dndr, dphidr, dtaudr)` — a Python tuple, not an array; `odeint`
accepts it.

### 1b. Neutral branch (`is_ionised == False`, `get_shellODE.py:129–147`)

$$\mathbf{y}=(n,\tau),\qquad \frac{dn}{dr}=\frac{\mu_n}{\mu_H k_B T_n}\cdot\frac{n\,\sigma_d}{4\pi r^2 c}L_n E,\qquad \frac{d\tau}{dr}=n\,\sigma_d$$

Differences from the ionised branch, all structural:

1. The `L_i φ` term and the recombination-momentum term are dropped entirely (φ is no longer a state).
2. `µ_p → µ_n` and `T_i → T_n`.
3. **`f_cover` is absent from `dτ/dr`** (`:144` vs `:122`) — see finding S8-A-06.
4. **No `min(n, _NSHELL_MAX)` clamp** (`:131` has no analogue of `:100`).

The `τ > 500 ⇒ E = 0` guard is duplicated at `:134–137`.

Return at `:147`: `(dndr, dtaudr)`.

---

## 2. Radiative transfer and ionisation

* **Dust attenuation of the non-ionising band.** `E = e^{-τ}` with `dτ/dr = n σ_d f_cover`. `σ_d`
  is a *per-hydrogen-nucleus* dust cross-section (it multiplies `n`, a number density, to give a
  reciprocal length), units pc² in code units. Hard cut to `E = 0` for `τ > 500`
  (`get_shellODE.py:103, 134`); `exp(-500) = 7.1e-218`, so the discontinuity introduced in the RHS
  is O(1e-218) — negligible in magnitude but it *is* a discontinuity in an adaptive integrator's RHS.
* **Ionising band attenuation** is carried by `φ`, not by `e^{-τ}`. `φ` obeys a two-sink equation:
  dust removal `−n σ_d φ` (same `σ_d` as the neutral band — no wavelength dependence) and
  photoionisation-balancing-recombination removal `−4πr²χ_e α_B n²/Q_i`.
* **Recombination balance.** The volumetric recombination rate is `χ_e α_B n²`
  (`χ_e = n_e/n_H`, `α_B` = case-B coefficient, pc³/Myr). Normalising by `Q_i` (Myr⁻¹) makes the
  sink dimensionless-per-length. Integrating `dφ/dr` with the dust term off and `n ≡ n_0` gives
  `φ(r) = 1 − (4π/3)χ_e α_B n_0² (r³−r_0³)/Q_i`, i.e. the Strömgren condition.
* **Ionisation-front condition.** There is no root-find. The IF is located *by array scan*:
  `phiCondition = phiShell_arr <= 1e-9` (`shell_structure.py:182`), and `idx` is the first index
  where mass-sweep **or** φ-depletion holds (`:183–188`). The IF radius and density are then
  `R_IF = rShell_arr_ion[-1]` and `n_IF = nShell_arr_ion[-1]` (`:224–226`) — i.e. the last collected
  grid point, which is `rShell_arr[idx]` of the final slice (appended at `:213–218`).
* **Maximum shell radius / Strömgren estimate** (`:144`):
  `R_max = [3 Q_i / (4π χ_e α_B n_0²)]^{1/3} + r_{start}` — the uniform-density Strömgren radius
  evaluated with the *inner* density `n_0`, offset by the inner radius. Used **only** to set slice
  sizes (`:148`, `:313`), never as a stopping condition.
* **Independent Strömgren-inversion density** (`:242–253`):
  `n_IF_Str = sqrt( 3 (1−f_esc) Q_i / (4π χ_e α_B (R_IF³ − R2³)) )`, then
  **`n_IF_Str = min(n_IF_Str, shell_n0)`** (`:251`). Guarded by `_vol_ion > 0 and _Qi_absorbed > 0`,
  else `0.0`. See finding S8-A-02: the clamp provably always binds.
* **Escape fraction** `f_esc_ion = max(0, φ(R_IF))` (`:229`), and
  `f_absorbed_ion = 1 − f_esc_ion` (`:398`), `f_absorbed_neu = 1 − e^{−τ_{end}}` (`:399`),
  luminosity-weighted total `f_absorbed = (f_abs,ion·L_i + f_abs,neu·L_n)/(L_i+L_n)` (`:400`).
  Note `τ_end` is the *ionised* dust optical depth when there is no neutral region (`:393`), so
  in that case `f_absorbed_neu` is a dust-only attenuation of the ionised layer.
* **Dust-vs-hydrogen split of the absorbed ionising photons** (`:276–288`): re-integrates the two
  sink terms of `dφ/dr` with a left-endpoint rectangle rule,
  `φ_dust = Σ −n_i σ_d φ_i Δr_i`, `φ_H = Σ −4πr_i²χ_e α_B n_i² Δr_i/Q_i`, then
  `f_ionised_dust = φ_dust/(φ_dust+φ_H)`, with a `== 0.0` guard returning `0.0`.

---

## 3. Dimensions

Code units (confirmed from `unit_conversions.py`: `cm2pc`, `s2Myr`, `g2Msun`, and the derived
`E_cgs2au`, `G_cgs2au = 6.74e4`, `gravPhi_cgs2au = 1.046e-10`, `grav_force_m_cgs2au = 3.227e8`,
`tau_cgs2au = 4788` ≡ g cm⁻² → M⊙ pc⁻²) are **pc, Myr, M⊙**:

| symbol | units |
|---|---|
| `r`, `rShell_step` | pc |
| `n` (`nShell`) | pc⁻³ |
| `σ_d` (`dust_sigma`) | pc² |
| `α_B` (`caseB_alpha`) | pc³ Myr⁻¹ |
| `Q_i` | Myr⁻¹ |
| `L_i`, `L_n` | M⊙ pc² Myr⁻³ |
| `c` | pc Myr⁻¹ |
| `k_B` | M⊙ pc² Myr⁻² K⁻¹ |
| `µ_p, µ_n, µ_H` | M⊙ |
| `Pb` | M⊙ pc⁻¹ Myr⁻² |
| `G` | pc³ M⊙⁻¹ Myr⁻² |
| `φ`, `τ`, `χ_e`, `f_cover` | dimensionless |

Term-by-term:

* `n σ_d /(4π r² c) · L` = `pc⁻³·pc²/(pc²·pc Myr⁻¹)·M⊙ pc² Myr⁻³` = **M⊙ pc⁻² Myr⁻²** = pressure/length ✓
* `χ_e α_B n² L_i/(Q_i c)` = `pc⁻³Myr⁻¹ · M⊙ pc² Myr⁻² · Myr pc⁻¹` = **M⊙ pc⁻² Myr⁻²** ✓ (matches sibling)
* prefactor `µ_p/(µ_H k_B T)` = `Myr² M⊙⁻¹ pc⁻²` ⇒ `dn/dr` = **pc⁻⁴** ✓
* `4πr²χ_e α_B n²/Q_i` = `pc²·pc³Myr⁻¹·pc⁻⁶·Myr` = **pc⁻¹** ✓; `n σ_d φ` = **pc⁻¹** ✓
* `dτ/dr = n σ_d` = **pc⁻¹** ✓
* `R_max`: `[Q_i/(α_B n²)]^{1/3}` = `[Myr⁻¹/(pc³Myr⁻¹ pc⁻⁶)]^{1/3}` = **pc** ✓
* `mShell = n µ_H 4πr² Δr` = **M⊙** ✓; `grav_force_m = G M/r²` = **pc Myr⁻²** (acceleration) ✓
* `grav_phi = −4πG ∫ r ρ dr` = **pc² Myr⁻²** (specific potential) ✓
* `tau_kappa_IR = µ_H Σ n Δr` = **M⊙ pc⁻²** (a column density, i.e. τ/κ) ✓

**No dimensional imbalance found in the ODE or in the derived quantities.** The only dimensional
irregularity is a *bare dimensioned literal*: `np.min([1, …])` at `:148` and `:313` caps the slice
size at `1` — that `1` is 1 pc, with nothing marking it as such (finding S8-A-16).

---

## 4. The integrator call and its return handling

**Two call sites, identical form.**

Ionised, `shell_structure.py:165–171`:

```
sol_ODE = scipy.integrate.odeint(
    get_shellODE.get_shellODE, y0, rShell_arr,
    args=(f_cover, is_ionised, params), mxstep=_SHELL_ODE_MXSTEP
)
nShell_arr   = sol_ODE[:, 0]
phiShell_arr = sol_ODE[:, 1]
tauShell_arr = sol_ODE[:, 2]
```

Neutral, `:324–329`: same, with `y0 = [nShell0, tau0_neu]`, `is_ionised = False`, and
`nShell_arr = sol_ODE[:, 0]`, `tauShell_arr = sol_ODE[:, 1]`.

**Arguments in full:** `func = get_shellODE.get_shellODE`; `y0 = [nShell0, phi0, tau0_ion]`
(or `[nShell0, tau0_neu]`); `t = rShell_arr` (≈1000 points ionised, ≈5000 neutral);
`args = (f_cover, is_ionised, params)`; `mxstep = _SHELL_ODE_MXSTEP = 50000` (`:35`).
**No `Dfun`, no `rtol`/`atol` (LSODA defaults 1.49e-8), no `tcrit`, no `hmax`, and — critically —
no `full_output`.**

**What is done with the return value.** `odeint` with `full_output=0` returns a single
`(len(t), len(y0))` array. The code slices out every column and uses it. **Nothing else is
requested and nothing is inspected**: there is no `full_output=1`, so no `infodict` and no
`message`; there is no `try/except`; there is no `warnings.catch_warnings`. The integrator's
status is therefore *not merely ignored — it is never obtained*.

**What scipy actually returns on failure.** Verified empirically in this environment
(scipy 1.17.1): with `full_output=0`, a failed integration emits a Python
`ODEintWarning("Excess work done on this call…")` — a *warning*, which by default prints once and
lets execution continue — and returns an array in which the rows **after the failure point are
uninitialised heap memory**:

```
array([1.00000000e+000, 4.87323049e+000, 6.90167858e-310, 6.90167857e-310,
       6.90169976e-310, 1.38736274e-315, 6.90167858e-310, ...])
```

(Reproduced with `odeint(lambda y,t: [y[0]**2*1e6], [1.0], np.linspace(0,10,20), mxstep=50)`.)

**Therefore: yes — array elements consumed downstream can be un-integrated, uninitialised memory.**
Because the warning is neither raised nor caught, the failure is completely silent to the caller.

**Exactly which physical quantities are derived from those elements.** The garbage values are
denormals of order `1e-310`. Trace the ionised path:

1. `phiShell_arr` garbage ≈ `1e-310` **satisfies `phiCondition = phi <= 1e-9`** (`:182`). So `idx`
   (`:188`) becomes the *first garbage row*, and `is_phiDepleted = True` (`:192`). A numerical
   integration failure is silently reinterpreted as "the ionisation front is here".
2. `nShell0, phi0, tau0_ion, mShell0, rShell_start` (`:203–207`) are re-seeded from that garbage row.
3. `:213–218` append the garbage row into `nShell_arr_ion`, `phiShell_arr_ion`, `tauShell_arr_ion`,
   `mShell_arr_ion`, `mShell_arr_cum_ion`.
4. Quantities then derived from uninitialised memory, by name:
   * **`n_IF`** and **`n_IF_ODE`** (`:224–225`) — `= nShell_arr_ion[-1]` = garbage.
   * **`R_IF`** (`:226`) — a real radius, but located at a spurious grid index.
   * **`f_esc_ion`** (`:229`) → **`shell_fAbsorbedIon`** (`:398`) → **`shell_fAbsorbedWeightedTotal`** (`:400`).
   * **`n_IF_Str`** (`:246`) via `_vol_ion` and `_Qi_absorbed`.
   * **`shell_grav_r`, `shell_grav_phi`, `shell_grav_force_m`** (`:262–273`) — the gravity arrays are
     built directly from `nShell_arr_ion`, including the garbage tail row.
   * **`shell_fIonisedDust`** (`:277–288`).
   * **`shell_thickness`** (`:392`), **`shell_nMax`** (`:394`), **`shell_tauKappaRatio`** (`:395`),
     **`rShell`** (`:402`), **`shell_n_arr` / `shell_r_arr`** (`:410–414`).
   * **`diss_condition_met`** (`:446`) via `nShell_max`.
5. `has_neutral = is_phiDepleted and not is_allMassSwept` (`:221`) is then `True`, so the neutral
   loop is entered with `nShell0 ≈ 1e-310·(µ_n/µ_p)(T_i/T_n)`. With `n ≈ 0` the mass integral
   `n µ_H 4πr²Δr` never reaches `mShell_end`, `massCondition` is never satisfied, and the
   `while not is_allMassSwept` loop at `:316` **never terminates** — it marches outward in fixed
   slices forever.

That last point is the practical failure mode: a silent LSODA failure in the ionised region
converts into an infinite loop in the neutral region, with no diagnostic emitted beyond a
one-line Python warning.

---

## 5. Control flow that changes the maths

**Clamps and substitutions inside the RHS** (`get_shellODE.py`):

| line | condition | expression used instead |
|---|---|---|
| 100 | always (ionised only) | `n → min(n, 1e120)` — the RHS saturates while the LSODA state keeps growing |
| 103–106 / 134–137 | `τ > 500` | `e^{-τ} → 0` exactly |
| 111 | `φ < 0` | `φ → 0`; then `dφ/dr = −4πr²χ_eα_Bn²/Q_i` only (the dust sink vanishes) and the `L_iφ` term of `dn/dr` vanishes |

Note the consequence of `:111` + `:120`: with `φ` clamped to 0 the remaining term of `dφ/dr` is
**still strictly negative**, so the true state `φ` has no equilibrium at 0 — it is driven
monotonically to −∞ past the front. The negative excursion is only bounded by the array-level
test `phi <= 1e-9`, which lives in the *caller*, not in the integrator.

**Ionised loop** (`shell_structure.py:157–218`), `while not is_allMassSwept and not is_phiDepleted`:

* `:181` `massCondition = mShell_arr_cum >= mShell_end`; `:182` `phiCondition = phiShell_arr <= 1e-9`.
* `:185–188` if neither triggers anywhere in the slice, `idx = len−1` (march another slice);
  otherwise `idx = first index of (massCondition | phiCondition)`.
* `:190` `mShell_arr_cum[idx+1:] = 0.0` — **a no-op** (see S8-A-04): `massCondition` was already
  snapshotted at `:181`, and every later read is `[:idx]` or `[idx]`.
* `:191–192` `is_allMassSwept = any(massCondition)`, `is_phiDepleted = any(phiCondition)` — both
  evaluated over the **entire slice, including indices past `idx`** (S8-A-03). If φ depletes at
  index 10 but the (physically irrelevant) continuation would sweep the full mass at index 500,
  `is_allMassSwept` is set `True`, `has_neutral` becomes `False` at `:221`, and the neutral region
  is **skipped entirely** even though the shell is φ-depleted with mass left to sweep.
* Slice stitching: `[:idx]` is collected, and `rShell_start = rShell_arr[idx]` becomes index 0 of
  the next slice — so no gap and no duplication. Loop termination is guaranteed in exact arithmetic
  because `dφ/dr ≤ −4πr²χ_eα_Bn_0²/Q_i` and `n` is increasing, so φ crosses 0 by `r = R_max`.

**Neutral loop** (`:316–357`), `while not is_allMassSwept`: only the mass condition, no radius bound,
no φ. `sliceSize` and `rShell_step` are recomputed at `:313–314` from
`min(1, (R_max − rShell_start)/10)/5000` where `rShell_start = R_IF`. Because the ionised loop runs
*until* φ depletes and φ is guaranteed to deplete at `r ≤ R_max`, `R_IF → R_max` in the low-dust
limit, driving `sliceSize → 0⁺` — and one grid step of discretisation overshoot makes it
**negative**, which inverts the `np.arange` direction, integrates *inward*, produces negative
`mShell_arr` increments (`:334` multiplies by `rShell_step`), so `massCondition` is never satisfied
and the loop runs inward without bound (S8-A-05). Exactly `sliceSize == 0` gives
`np.arange(a, a, 0.0) → ZeroDivisionError` (verified).

**Top-level branch** `:258 if not is_shellDissolved` / `:416 elif is_shellDissolved` — the `elif`
is an exhaustive `else`; both arms assign every `ShellProperties` field, so there is no
unbound-name path. Note the ionised ODE loop at `:157–218` runs **before** this branch, so a
dissolved shell still pays for (and can still hang or crash in) the full ionised integration whose
results are then discarded (S8-A-10).

**Guarded degeneracies:** `:245` `_vol_ion > 0 and _Qi_absorbed > 0` else `n_IF_Str = 0`;
`:285` `(φ_dust + φ_H) == 0` else `f_ionised_dust = 0`. Both are real guards.
`scipy.integrate.simpson` on the 1-element arrays that arise when `idx == 0` on the first
iteration returns `0.0` (verified) rather than raising.

**Unguarded:** `:400` and `:419` divide by `(Li + Ln)` with no zero test; `get_shellODE.py:117,120`
divide by `Q_i` with no zero test, and `:144`'s `R_max` also divides by `Q_i`-free but by `n_0²`.
`Q_i = 0` (no ionising photons) gives `sliceSize = 0` at `:148` → `ZeroDivisionError` in
`np.arange` before the ODE is even called.

---

## 6. Numeric literals

`get_shellODE.py`

| line | literal | expression |
|---|---|---|
| 32 | `1e120` | `_NSHELL_MAX`, used at `:100` as `min(nShell, _NSHELL_MAX)` |
| 103, 134 | `500` | `if tau > 500: neg_exp_tau = 0` |
| 104, 135 | `0` | replacement value for `e^{-τ}` |
| 111 | `0.0` | `phi = max(0.0, phi)` |
| 115–122 | `4`, `np.pi`, `2` | `4·π·r**2` in the dust-momentum denominator; `nShell**2` |
| 120 | `4`, `np.pi`, `2` | `4πr²…n²` recombination sink |
| 140–141 | `4`, `np.pi`, `2` | same in the neutral branch |

`shell_structure.py`

| line | literal | expression |
|---|---|---|
| 35 | `50000` | `_SHELL_ODE_MXSTEP`, passed as `mxstep` |
| 115 | `1` | `f_cover = 1` (hardcoded; the only value ever passed to the ODE) |
| 119–121 | `1`, `0`, `0` | `phi0`, `tau0_ion`, `mShell0` |
| 144 | `3`, `4`, `1/3` | `(3 Q_i /(4π χ_e α_B n_0²))**(1/3)` |
| 147 | `1e3` | `nsteps` (ionised) |
| 148 | `1`, `10` | `np.min([1, (max_shellRadius − rShell_start)/10])` — the `1` is **1 pc**, dimensioned |
| 176–177 | `4`, `2` | `n µ_H 4π r² Δr` |
| 182 | `1e-9` | `phiCondition = phiShell_arr <= 1e-9` |
| 190 | `0.0` | `mShell_arr_cum[idx+1:] = 0.0` (no-op) |
| 204, 229 | `0.0` | `max(0.0, φ)` clamps |
| 242–248 | `3.0`, `4.0`, `1.0` | Strömgren inversion `sqrt(3 Q_abs /(4π χ_e α_B ΔV))`, `1.0 − f_esc` |
| 253 | `0.0` | fallback `n_IF_Str` |
| 264, 270 | `4`, `2` | `ρ4πr²Δr`, `G M/r²` |
| 266, 370 | `−4` | `−4πG ∫ rρ dr` |
| 277–282 | `−1`(implicit), `4`, `2` | φ-sink re-integration |
| 311 | `100` | `tau_max = 100` — **assigned, never read** |
| 312 | `5e3` | `nsteps` (neutral) |
| 313 | `1`, `10` | same dimensioned `1 pc` cap |
| 399 | `1` | `1 − exp(−τ_end)` |
| 408, 437 | `1`, `−1` | `shell_ion_idx` |

---

## Additional observations (reported, not necessarily defects)

* **`n_IF` and `n_IF_ODE` are set to the identical value** (`:224–225`) and both are returned.
* **`mShell_arr_ion`, `mShell_arr_cum_ion`, `mShell_arr_neu`, `mShell_arr_cum_neu`** are built and
  appended to across both loops and **never read afterwards**.
* **`mShell_arr[0] = mShell0`** (`:175`, `:332`) stores a *cumulative* mass in element 0 of an array
  whose remaining elements are *differential* masses. `cumsum` then makes `mShell_arr_cum` correct,
  but the collected `mShell_arr_ion`/`_neu` arrays contain one cumulative value per slice boundary
  mixed with differentials. (Harmless only because they are unused.)
* **Aliasing:** in the no-neutral case `shell_r_arr` (`:413`), `grav_r`/`shell_grav_r` (`:263, 273`)
  and `rShell_arr_ion` are the *same ndarray object* — three fields of the returned dataclass alias
  one buffer. Same for `shell_n_arr` and `nShell_arr_ion`.
* **`np.arange` point count** is `ceil(sliceSize/rShell_step)` and can be `nsteps` or `nsteps+1`
  depending on floating-point rounding of `sliceSize/nsteps`; nothing depends on the exact count,
  but `mShell_arr[1:]` uses the *nominal* `rShell_step`, not the realised spacing.
* **`grav_ion_m` (`:264`) and `grav_neu_m` (`:368`) also use the nominal `rShell_step`**, which is
  correct only because each concatenated array is uniformly spaced.
* **`logger.debug(f'…{phiShell_arr_ion[:10]}')` (`:379`)** formats eagerly regardless of log level.
* **`any(massCondition)`** uses the Python builtin on a NumPy bool array — element-by-element
  Python iteration over ~1000–5000 items per loop iteration, not `np.any`.
* `f_absorbed_neu` in the no-neutral case (`:393, 399`) uses the **ionised region's** dust optical
  depth, i.e. `1 − e^{−τ_ion}`, as the non-ionising-band absorbed fraction.
* Once dissolved, `nShell_max` is set to exactly `nISM` (`:423`), and `diss_condition_met` tests
  `nShell_max < nISM` strictly (`:446`) — so a dissolved shell always reports
  `diss_condition_met = False`.

---

```json
[
  {
    "id": "S8-A-01",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 165,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "scipy.integrate.odeint is called without full_output and without any exception/warning handling, and every column of its return is consumed unconditionally. On integration failure scipy returns an array whose rows past the failure point are uninitialised heap memory; those rows are read as physical state.",
    "evidence": "shell_structure.py:165-171 and :324-329 call odeint(..., mxstep=_SHELL_ODE_MXSTEP) with no full_output, no try/except, no warnings.catch_warnings, then immediately do sol_ODE[:,0], [:,1], [:,2]. Verified in this environment (scipy 1.17.1): odeint(lambda y,t:[y[0]**2*1e6],[1.0],np.linspace(0,10,20),mxstep=50) emits only an ODEintWarning and returns array([1.0, 4.873, 6.90167858e-310, 6.90167857e-310, 1.387e-315, ...]) - denormal garbage. Those denormals (~1e-310) satisfy phiCondition = phiShell_arr <= 1e-9 at shell_structure.py:182, so idx (:188) lands on the first garbage row and is_phiDepleted becomes True (:192).",
    "expected": "Pass full_output=1 and inspect infodict/istate (or catch ODEintWarning as an error), and abort/retry/flag the timestep on failure instead of consuming the returned buffer.",
    "failure_scenario": "LSODA hits mxstep in the stiff ionised region. The un-integrated tail (uninitialised memory) is treated as 'phi depleted here'. n_IF and n_IF_ODE (:224-225) are read directly from that garbage row; f_esc_ion (:229) -> shell_fAbsorbedIon (:398) -> shell_fAbsorbedWeightedTotal (:400); n_IF_Str (:246); shell_grav_r/_phi/_force_m (:262-273); shell_fIonisedDust (:277-288); shell_thickness (:392); shell_nMax (:394); shell_tauKappaRatio (:395); rShell (:402); shell_r_arr/shell_n_arr (:410-414); diss_condition_met (:446). has_neutral then becomes True with nShell0 ~ 1e-310, so the neutral loop at :316 can never satisfy massCondition and never terminates.",
    "repro": "python -c \"import numpy as np,scipy.integrate; print(scipy.integrate.odeint(lambda y,t:[y[0]**2*1e6],[1.0],np.linspace(0,10,20),mxstep=50).ravel())\"",
    "confidence": "high"
  },
  {
    "id": "S8-A-02",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 251,
    "class": "numerical",
    "severity": "S2",
    "claim": "The clamp n_IF_Str = min(n_IF_Str, shell_n0) provably always binds, so the four-line Strommgren-inversion density computed at :246-249 is always discarded and n_IF_Str is identically shell_n0 whenever the shell exists.",
    "evidence": "In the ionised ODE every term of dn/dr is non-negative (get_shellODE.py:115-118: sigma_dust, Ln, Li, chi_e, alpha_B, Qi, c all positive; exp(-tau)>=0; phi clamped >=0 at :111), so n(r) >= n0 strictly increasing. The raw estimate at shell_structure.py:246 is n_raw^2 = 3*Q_abs/(4*pi*chi_e*alpha_B*(R_IF^3-R2^3)) = Q_abs/(chi_e*alpha_B*dV_geo), while the hydrogen-absorbed photons alone give Q_H = chi_e*alpha_B*<n^2>*dV_geo with <n^2> >= n0^2. Since Q_abs >= Q_H (dust absorbs extra ionising photons), n_raw >= sqrt(<n^2>) >= n0 = shell_n0. Hence min(n_raw, shell_n0) == shell_n0 always.",
    "expected": "If the intent is a sanity ceiling it should be a warning/diagnostic, not a silent replacement; as written the returned n_IF_Str carries no information beyond shell_n0 (which is already returned separately as shell_n0 at :450).",
    "failure_scenario": "Any downstream consumer comparing n_IF against n_IF_Str, or using n_IF_Str as an independent IF-density estimate, is comparing against the inner-boundary density constant, not a Strommgren inversion.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-03",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 191,
    "class": "state",
    "severity": "S2",
    "claim": "is_allMassSwept and is_phiDepleted are evaluated with any() over the ENTIRE integrated slice, including grid points beyond idx that lie past the physical end of the shell, so both termination flags can be set by conditions that occur in the discarded tail.",
    "evidence": "shell_structure.py:181-192: massCondition and phiCondition are computed over the full slice; idx is the first index of their union (:183-188); mShell_arr_cum[idx+1:] = 0.0 at :190 is a no-op because massCondition was already snapshotted at :181 and is a separate bool array; then :191-192 call any() on the full-length arrays. has_neutral = is_phiDepleted and not is_allMassSwept (:221) consumes both.",
    "expected": "The flags should be evaluated at index idx only, e.g. is_allMassSwept = massCondition[idx], is_phiDepleted = phiCondition[idx] - which is presumably what the truncation at :190 was meant to achieve.",
    "failure_scenario": "phi depletes at index 10 while the (unphysical) continuation of the integration accumulates mShell_end by index 500. is_allMassSwept is set True, has_neutral becomes False, and the neutral shell region is never integrated even though the shell is phi-depleted with mass still to sweep. shell_thickness, tau_rEnd, nShell_max, tau_kappa_IR, rShell and the gravity arrays are then all the ionised-only values.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-04",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 190,
    "class": "deadcode",
    "severity": "S4",
    "claim": "mShell_arr_cum[idx + 1:] = 0.0 has no observable effect.",
    "evidence": "massCondition (:181) is an independent boolean array captured before the mutation; every subsequent read of mShell_arr_cum is mShell_arr_cum[:idx] (:196) or mShell_arr_cum[idx] (:206, :214), none of which touch indices > idx. The array is freshly produced by np.cumsum at :178 so no other reference is live. The neutral loop has no analogous statement.",
    "expected": "Either remove it, or move it before :191 and re-derive the flags from the truncated array (see S8-A-03).",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-05",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 313,
    "class": "divergence",
    "severity": "S2",
    "claim": "The neutral-region slice size sliceSize = np.min([1, (max_shellRadius - rShell_start)/10]) has no lower bound; rShell_start is R_IF, which by construction approaches max_shellRadius, so sliceSize can be ~0 (ZeroDivisionError) or negative (inverted integration limits and a non-terminating inward march).",
    "evidence": "max_shellRadius (:144) is the uniform-density Strommgren radius built from n0. dphi/dr <= -4*pi*r^2*chi_e*alpha_B*n0^2/Qi with n monotonically increasing, so phi crosses zero at r <= max_shellRadius; the ionised loop stops at the first grid point past that, hence rShell_start = rShell_arr_ion[-1] (:296) can equal or slightly exceed max_shellRadius after discretisation. np.min([1, x]) does not clamp x from below. Then rShell_step = sliceSize/5e3 (:314) is <= 0 and np.arange(rShell_start, rShell_start+sliceSize, rShell_step) at :320 is decreasing (verified: np.arange(5.0, 4.9, -1e-4) yields 1000 descending points) or raises (verified: np.arange(5.0, 5.0, 0.0) -> ZeroDivisionError).",
    "expected": "Clamp the slice size to a strictly positive floor, or size the neutral slices from the remaining shell mass / a neutral length scale rather than from the ionised Strommgren estimate.",
    "failure_scenario": "Low-dust / dust-poor shell where the IF sits at the Strommgren radius. Either (a) ZeroDivisionError from np.arange, or (b) rShell_step < 0: odeint integrates inward, mShell_arr[1:] = n*mu*4*pi*r^2*rShell_step (:334) is negative, mShell_arr_cum decreases, massCondition is never True, and 'while not is_allMassSwept' (:316) loops forever marching inward toward r=0 (where dn/dr divides by r**2). Or (c) sliceSize tiny-positive: each iteration advances by ~(max_shellRadius-R_IF)/10, so sweeping the full shell mass takes an unbounded number of iterations - a hang.",
    "repro": "python -c \"import numpy as np; print(np.arange(5.0,4.9,-1e-4)[[0,-1]]); np.arange(5.0,5.0,0.0)\"",
    "confidence": "medium"
  },
  {
    "id": "S8-A-06",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 144,
    "class": "other",
    "severity": "S3",
    "claim": "The neutral branch's optical-depth equation omits the f_cover factor that the ionised branch applies, so the two sibling dtau/dr expressions are inconsistent.",
    "evidence": "get_shellODE.py:122 (ionised) is 'dtaudr = nShell * sigma_dust * f_cover'; :144 (neutral) is 'dtaudr = nShell * sigma_dust'. f_cover is a declared parameter of get_shellODE (:39) and is threaded through both odeint calls via args (shell_structure.py:167, :326).",
    "expected": "Both branches should apply the same covering-fraction treatment, or f_cover should be removed from the signature.",
    "failure_scenario": "Currently inert: f_cover is hardcoded to 1 at shell_structure.py:115 and never varied. If f_cover is ever made a real parameter, tau across the neutral region - and hence f_absorbed_neu = 1 - exp(-tau_rEnd) at :399 and shell_fAbsorbedWeightedTotal at :400 - silently uses a different covering assumption from the ionised region.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-07",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 111,
    "class": "numerical",
    "severity": "S3",
    "claim": "phi = max(0.0, phi) clamps only the local copy used to build the RHS; the integrator state is unaffected and dphi/dr remains strictly negative at phi=0, so phi has no equilibrium and runs away to -inf past the ionisation front.",
    "evidence": "get_shellODE.py:111 rebinds the local phi; :116 uses it in (Ln*exp(-tau) + Li*phi) and :120 in '- nShell*sigma_dust*phi'. With phi clamped to 0 the dust sink vanishes but the first term -4*pi*r^2*chi_e*alpha_B*nShell**2/Qi is still strictly negative. Nothing in the ODE or in odeint stops the integration; only the caller's array test phiShell_arr <= 1e-9 (shell_structure.py:182) bounds the excursion, and the caller then clamps again at :204 (phi0 = max(0.0, ...)) and :229 (f_esc_ion = max(0.0, ...)).",
    "expected": "Terminate the ionised integration at the front (e.g. solve_ivp with a terminal event on phi), or make the RHS consistent by zeroing dphi/dr once phi <= 0.",
    "failure_scenario": "The RHS is discontinuous at phi=0 and inconsistent with the state for phi<0, which degrades LSODA's error control exactly where the system is stiffest - increasing the chance of the mxstep failure that S8-A-01 then consumes silently. The three max(0.0, .) clamps also mean an unphysically large negative phi overshoot is reported as f_esc_ion = 0 (i.e. perfect absorption) with no diagnostic.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-08",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 100,
    "class": "numerical",
    "severity": "S4",
    "claim": "nShell = min(nShell, 1e120) saturates the density in the RHS only, decoupling the returned derivative from the integrator state; the clamp is present in the ionised branch and absent in the neutral branch.",
    "evidence": "get_shellODE.py:32 defines _NSHELL_MAX = 1e120; :100 applies it after unpacking y at :96. The neutral unpack at :131 has no analogue. In code units (pc^-3) 1e120 corresponds to ~3e64 cm^-3, so the clamp only engages in a numerical runaway - at which point nShell**2 = 1e240 is still finite but 4*pi*r^2*chi_e*alpha_B*nShell**2/Qi (:120) can overflow to inf.",
    "expected": "Either clamp the state (via a bounded solver / event) or drop the clamp and let the failure surface; a derivative computed from a clamped state that the solver does not see is not a solution of anything.",
    "failure_scenario": "In a runaway the integrator sees a derivative consistent with n=1e120 while its own state exceeds it, so step-size control is driven by a fiction; the resulting mxstep failure feeds S8-A-01.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S8-A-09",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 148,
    "class": "divergence",
    "severity": "S3",
    "claim": "Qi = 0 makes sliceSize = 0 and rShell_step = 0, so np.arange raises ZeroDivisionError before the ODE is entered; Qi = 0 would also divide by zero at get_shellODE.py:117 and :120.",
    "evidence": "shell_structure.py:144 max_shellRadius = (3*Qi/(4*pi*chi_e*alpha_B*nShell0**2))**(1/3) + rShell_start, so Qi=0 gives max_shellRadius == rShell_start; :148 sliceSize = np.min([1, 0/10]) = 0; :149 rShell_step = 0; :161 np.arange(a, a, 0) raises. get_shellODE.py:117 has 'Li/Qi' and :120 has '.../Qi' with no guard.",
    "expected": "Guard the no-ionising-photon regime explicitly (it is a physically reachable state once the massive stars have died) rather than relying on an arange exception.",
    "failure_scenario": "A cluster whose ionising output has switched off (post-SN / late evolution) enters shell_structure_pure and crashes with ZeroDivisionError from np.arange rather than taking the dissolved/neutral path.",
    "repro": "python -c \"import numpy as np; np.arange(5.0,5.0,0.0)\"",
    "confidence": "medium"
  },
  {
    "id": "S8-A-10",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 157,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The full ionised ODE integration loop runs unconditionally even when the shell is already flagged dissolved, and all of its results (including n_IF, n_IF_ODE, R_IF, n_IF_Str) are then overwritten by constants.",
    "evidence": "is_shellDissolved is read at :130-132 but the while loop at :157-218 and the derived quantities at :224-253 execute regardless; the dissolved arm at :416-437 overwrites n_IF, n_IF_ODE, R_IF, n_IF_Str with 0.0 and sets shell_r_arr/shell_n_arr to empty arrays.",
    "expected": "Take the dissolved branch before integrating.",
    "failure_scenario": "A dissolved shell pays the full integration cost and, worse, remains exposed to the failure modes of that integration (S8-A-01, S8-A-09) on a code path whose output is discarded - so a crash or hang can occur in a regime the code intends to short-circuit.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-11",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 195,
    "class": "deadcode",
    "severity": "S4",
    "claim": "mShell_arr_ion, mShell_arr_cum_ion, mShell_arr_neu and mShell_arr_cum_neu are accumulated across both integration loops and never read.",
    "evidence": "mShell_arr_ion appears only at :136 (init), :195 (concat), :213 (append); mShell_arr_cum_ion only at :137, :196, :214; mShell_arr_neu at :291, :347, :359; mShell_arr_cum_neu at :292, :348, :360. None appear in the ShellProperties construction at :449-471 or anywhere else.",
    "expected": "Remove, or return them if a caller needs the mass profile.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-12",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 175,
    "class": "other",
    "severity": "S4",
    "claim": "Element 0 of mShell_arr holds a CUMULATIVE mass (mShell0, carried over from the previous slice) while elements 1: hold DIFFERENTIAL shell masses, so the accumulated mShell_arr_ion / mShell_arr_neu arrays mix the two quantities at every slice boundary.",
    "evidence": "shell_structure.py:175 'mShell_arr[0] = mShell0' with mShell0 set from mShell_arr_cum[idx] at :206 (:355 for the neutral loop), versus :176-177 'mShell_arr[1:] = nShell_arr[1:]*mu_convert*4*pi*rShell_arr[1:]**2*rShell_step'. The cumsum at :178 is correct because of this construction, but the per-cell array collected at :195/:347 is not a differential-mass array.",
    "expected": "Keep the running offset out of the per-cell array (e.g. mShell_arr_cum = mShell0 + np.cumsum(dm)) so the collected array has a single meaning.",
    "failure_scenario": "Currently latent because the arrays are unused (S8-A-11); it becomes a real error the moment anyone reads mShell_arr_ion as a mass profile.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-13",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 311,
    "class": "deadcode",
    "severity": "S4",
    "claim": "tau_max = 100 is assigned and never read; there is no optical-depth stopping condition anywhere in the neutral loop.",
    "evidence": "The identifier tau_max appears exactly once in the file, at :311. The neutral loop at :316 terminates only on 'not is_allMassSwept'.",
    "expected": "Either implement the tau cutoff or delete the assignment.",
    "failure_scenario": "The neutral integration has no optical-depth bound; combined with S8-A-05 the only termination condition is mass sweep-up.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-14",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 225,
    "class": "deadcode",
    "severity": "S4",
    "claim": "n_IF_ODE is an unconditional duplicate of n_IF; both are returned as separate ShellProperties fields with identical values on every code path.",
    "evidence": ":224-225 'n_IF = nShell_arr_ion[-1]; n_IF_ODE = n_IF'; the dissolved arm sets both to 0.0 at :431-432; both are passed through at :465-466.",
    "expected": "One field, or two genuinely different estimates.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-15",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 413,
    "class": "state",
    "severity": "S4",
    "claim": "In the no-neutral case three returned fields alias the same ndarray buffer: shell_r_arr, shell_grav_r and the internal rShell_arr_ion are the same object (likewise shell_n_arr and nShell_arr_ion).",
    "evidence": ":263 grav_ion_r = rShell_arr_ion; :273 grav_r = grav_ion_r; :413-414 shell_r_arr = rShell_arr_ion, shell_n_arr = nShell_arr_ion - none of these copy. Only the has_neutral path (:376-377, :410-411) produces fresh arrays via np.concatenate.",
    "expected": "Copy on return, or document the aliasing, so that a consumer mutating shell_r_arr in place does not silently alter shell_grav_r.",
    "failure_scenario": "Any downstream in-place edit (unit conversion applied with *=, sorting, clipping) to one field silently mutates the other, and only in the no-neutral case - so the bug would be regime-dependent and hard to reproduce.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-16",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 148,
    "class": "units",
    "severity": "S3",
    "claim": "The slice-size cap is a bare dimensioned literal: np.min([1, ...]) caps the radial slice at 1 pc with nothing in the arithmetic marking it as a length.",
    "evidence": ":148 'sliceSize = np.min([1, (max_shellRadius - rShell_start) / 10])' and identically at :313. The code's length unit is pc (unit_conversions.py cm2pc = 3.2408e-19, v_kms2au = 1.0227 pc/Myr), so the 1 is 1 pc while the other argument is a genuine radius difference.",
    "expected": "A named constant with its unit stated, or derive the cap from a physical length scale in the problem.",
    "failure_scenario": "For a compact shell (R_IF - R2 well below 1 pc) the cap never engages and the step is R_St/10000; for a very extended cloud the 1 pc cap forces many outer loop iterations. The literal silently sets the resolution of the whole shell profile and of the mass/gravity quadratures at :176, :264, :277-282, :389-395.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-A-17",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 103,
    "class": "numerical",
    "severity": "S4",
    "claim": "The tau > 500 guard replaces exp(-tau) with exactly 0, introducing a step discontinuity in the ODE right-hand side at tau = 500 even though exp(-500) = 7.1e-218 is perfectly representable (underflow only begins near tau = 745).",
    "evidence": "get_shellODE.py:103-106 and :134-137; np.exp(-500.0) = 7.124576406741286e-218 and np.exp(-745.0) = 5e-324 (both finite, verified). The guard therefore prevents nothing numerically and only adds a discontinuity.",
    "expected": "Either raise the threshold to ~700 where exp genuinely underflows, or drop the branch (np.exp underflows to 0.0 silently anyway).",
    "failure_scenario": "Negligible in magnitude (O(1e-218) jump in dn/dr), but it is a non-smooth RHS crossing that an adaptive stiff solver can waste steps on.",
    "repro": "python -c \"import numpy as np; print(np.exp(-500.0), np.exp(-745.0))\"",
    "confidence": "high"
  },
  {
    "id": "S8-A-18",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 400,
    "class": "divergence",
    "severity": "S4",
    "claim": "The luminosity-weighted absorbed fraction divides by (Li + Ln) with no zero guard, on both the live and the dissolved code path.",
    "evidence": ":400 'f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)' and the identical expression at :419 inside the dissolved arm, where the numerator is identically 0 so the result is 0/0.",
    "expected": "Guard Li + Ln == 0 and return 0 (or NaN with a flag).",
    "failure_scenario": "A cluster with no remaining radiative output (both bands zero) raises ZeroDivisionError - and does so even on the dissolved path, which is precisely the state such a cluster would be in.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S8-A-19",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 446,
    "class": "state",
    "severity": "S3",
    "claim": "Once the shell is flagged dissolved, diss_condition_met can never be True again, because the dissolved arm sets nShell_max exactly equal to nISM and the test is a strict inequality.",
    "evidence": ":423 'nShell_max = params['nISM'].value' in the dissolved arm; :442 'nISM = params['nISM'].value'; :446 'diss_condition_met = bool(allow_dissolution and nShell_max < nISM)'. nISM < nISM is False.",
    "expected": "Either use <= in the dissolved arm's semantics, or set diss_condition_met = True directly when is_shellDissolved is already True, depending on whether the flag means 'the shell just dissolved' or 'the shell is dissolved'.",
    "failure_scenario": "If a downstream latch re-reads diss_condition_met each step to decide whether the shell remains dissolved, it will read False on every step after the first and may un-dissolve the shell. The behaviour depends on caller semantics not visible in this slice.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S8-A-20",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 316,
    "class": "divergence",
    "severity": "S3",
    "claim": "The neutral integration loop has no radius bound, no optical-depth bound and no iteration cap - its only exit is mass sweep-up, so any state with a near-zero neutral density loops forever.",
    "evidence": ":316 'while not is_allMassSwept:' with is_allMassSwept set only from massCondition = mShell_arr_cum >= mShell_end (:337, :345). The declared tau_max at :311 is never used, max_shellRadius is used only for the slice size at :313, and there is no iteration counter. The mass increment n*mu_convert*4*pi*r^2*rShell_step (:334) goes to zero with n. Contrast the ionised loop, which is bounded because dphi/dr <= -4*pi*r^2*chi_e*alpha_B*n0^2/Qi guarantees phi crosses zero by max_shellRadius.",
    "expected": "Add a hard radius / iteration / optical-depth cap with an explicit failure signal.",
    "failure_scenario": "Reached whenever nShell0 entering the neutral loop (:307-308) is ~0 - which is exactly what happens after the silent odeint failure of S8-A-01 (garbage denormals ~1e-310) or after S8-A-05's negative rShell_step makes the increments negative. The run hangs with no diagnostic.",
    "repro": "",
    "confidence": "high"
  }
]
```
