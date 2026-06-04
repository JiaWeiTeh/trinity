# `n` consistency audit against the paper (`n ≡ n_H`)

Single source of truth for making **every** density/pressure/mass/cooling
term in TRINITY consistent with the model paper. Supersedes the convention
discussion in `pressure-terms-audit.md` (that file reached the *opposite*
sign on the bubble before the paper pinned the convention — see §0.1).

> **Paper convention (decisive):** *"All number densities `n` that follow
> denote hydrogen nuclei densities."* Every `n` in the code must therefore be
> `n_H`, with mass via `μ_H` and pressure via the `μ_H/μ` factor.

## 0. Canonical reference (derived from the paper)

Composition: `x_He = 0.1`, helium **doubly ionised** `Z_He = 2`.

| symbol | paper | code param | value |
|---|---|---|---|
| `μ_H`  | mass per H nucleus | `mu_convert` | `1.4 m_H` |
| `μ_n`  | `μ_H/(1+x_He)`, neutral mean mass/particle | `mu_atom` | `14/11 m_H` |
| `μ_p`  | `μ_H/(2+x_He(1+Z_He))`, ionised mean mass/particle | `mu_ion` | `14/23 m_H` |
| `μ_mol`| molecular mean mass/particle | `mu_mol` | `14/6 m_H` |

**Rules every `n`-line must obey (n = n_H):**

| quantity | paper expression | numeric factor |
|---|---|---|
| mass density | `ρ = μ_H · n` | `mu_convert · n` |
| ionised gas pressure | `P = (μ_H/μ_p) n k_B T_ion` | `(mu_convert/mu_ion)·n·k_B·T = 2.3 n k_B T` |
| neutral gas pressure | `P = (μ_H/μ_n) n k_B T_neu` | `(mu_convert/mu_atom)·n·k_B·T = 1.1 n k_B T` |
| electron density (ionised) | `n_e = (1+Z_He x_He) n` | `1.2 n` |
| volumetric cooling | `dU/dt = −n_H n_e Λ_net = −(1+Z_He x_He) n² Λ` | `1.2 n² Λ` (CIE) |
| recombination rate | `(1+Z_He x_He) α_B n²` | `1.2 α_B n²` |
| Strömgren density | `n = [3(1−f_esc)Q_i / (4π(1+Z_He x_He)α_B ΔV)]^{1/2}` | `1/√1.2` in `n` |
| bubble n from pressure | `n_H = (μ_p/μ_H) P_b/(k_B T)` | `(mu_ion/mu_convert)·P_b/(k_B T) = P_b/(2.3 k_B T)` |
| conduction mean mass | `μ = μ_p` | `mu_ion` |
| sound speed | `c_s = (γ k_B T/μ)^{1/2}`, `μ` = mean mass/particle | `mu_ion`/`mu_atom` |

**Key identities:** `μ_H/μ_p = mu_convert/mu_ion = 2.3`,
`μ_H/μ_n = mu_convert/mu_atom = 1.1`, `(1+Z_He x_He) = 1.2`.

### 0.1 Why this reverses `pressure-terms-audit.md`
That earlier audit, lacking the paper, treated `mu_ion = 14/23` (mean mass
per *total* particle) as the mass-conversion weight and concluded the
bubble `/2` should become `/1` (n → n_tot). The paper instead fixes
`n ≡ n_H`: mass uses `μ_H` (`mu_convert`), and the bubble factor becomes
**2.3** (`μ_H/μ_p`), not 1. Net effect on the bubble is the same direction
(density was ~2× low) but the *correct* implementation is the `n_H` form,
which also makes the cooling axis (which is `n_H`) consistent.

---

## 1. Inventory by flow (paper value vs. code)

Legend: ✅ already matches paper · ❌ must change · ⚪ convention-independent / no change.

### Phase 0 — initialisation & cloud (`phase0_init/`, `cloud_properties/`)
| file:line | code | paper | verdict |
|---|---|---|---|
| `get_InitPhaseParam.py:146` | `rhoa = nCore*mu_convert` | `ρ=μ_H n` | ✅ |
| `get_InitPhaseParam.py:172-176` | `T0 … (nCore)^(2/35)` (Eq. Tbubble) | `n=n_H` | ✅ |
| `density_profile.py` | `n(r)` = nCore/nISM-based H density | `n=n_H` | ✅ |
| `mass_profile.py:126,307-308` | `ρ = n*mu_convert` | `ρ=μ_H n` | ✅ |
| `bonnorEbertSphere.py:430,603,643` | `c_s²=γk_BT/μ`, `μ=mu_convert` | `μ` = mean mass/particle (μ_n or μ_mol, **not** μ_H) | ⚠️ §3.5 |

### Phase: bubble structure (`bubble_structure/`, `cooling/`)
| file:line | code | paper | verdict |
|---|---|---|---|
| `bubble_luminosity.py:338,390,465,498,948` | `n = Pb/(2 k_B T)` | `n_H=(μ_p/μ_H)Pb/(k_B T)` | ❌ `2`→`2.3` |
| `bubble_luminosity.py:970` | `rho = n*mu_ion` | `ρ=μ_H n` | ❌ `mu_ion`→`mu_convert` |
| `bubble_luminosity.py:411,520` | `n²·Λ_CIE` | `n_H n_e Λ=(1+Z_He x_He)n²Λ` | ❌ ×1.2 (verify table, §4) |
| `net_coolingcurve.py:126,149` | `ndens²·Λ` (CIE) | same | ❌ ×1.2 (§4) |
| `net_coolingcurve.py:118,141` | non-CIE table `interp(n,T,φ)` | table axis = `n_H` | ✅ once `n=n_H` |
| `bubble_luminosity.py:595,927,933` | conduction `μ=mu_ion` | `μ=μ_p` | ✅ |
| `get_bubbleParams.py:227` | `Pb=(γ−1)Eb/V` | Eq. Pb | ✅ (no n) |

### Phase: shell structure (`shell_structure/`)
| file:line | code | paper | verdict |
|---|---|---|---|
| `get_shellODE.py:93` | ion `dndr` prefactor `mu_p/mu_n` | `μ_p/μ_H` | ❌ `mu_n`→`mu_convert` |
| `get_shellODE.py:95` | recomb `n²·α_B` (in dndr) | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `get_shellODE.py:98` | `dphidr` recomb `α_B n²` | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `get_shellODE.py:118` | neutral `dndr` prefactor `1` | `μ_n/μ_H` | ❌ `1`→`mu_atom/mu_convert` |
| `shell_structure.py:115` | `nShell0=(mu_ion/mu_atom)Pb/(k_B T)` | `(μ_p/μ_H)Pb/(k_B T)` (Eq. nShell0) | ❌ `mu_atom`→`mu_convert` |
| `shell_structure.py:167,253,359` | mass `nShell*mu_atom` | `ρ=μ_H n` | ❌ `mu_atom`→`mu_convert` |
| `shell_structure.py:135` | `max_shellRadius` Strömgren `α_B n²` | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `shell_structure.py:237-239` | `n_IF_Str` (Eq. nIF_Str) | `(1+Z_He x_He)` in denom | ❌ ×1.2 |
| `shell_structure.py:272-273` | `phi_hydrogen` recomb `α_B n²` | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `shell_structure.py:298` | I-front jump `*mu_atom/mu_ion*T_ion/T_neu` | `μ_n/μ_p·T_ion/T_neu` (μ_H cancels) | ⚪ no change |
| `get_shellParams.py:30` | `nShell0=(mu_atom/mu_ion)…` (**dead**) | inverted **and** wrong μ | ❌ §5 (or delete) |

### Phases: energy / implicit / transition / momentum (`phase1*`, `phase2*`)
| file:line | code | paper | verdict |
|---|---|---|---|
| `energy_phase_ODEs.py:55` | `P_ion=2.0 n_r k_B T_ion` | `(μ_H/μ_p)n k_B T_ion` | ❌ `2.0`→`2.3` |
| `run_energy_phase.py:195` | `P_HII=2.0 n_IF_Str k_B T_ion` | Eq. PHII | ❌ `2.0`→`2.3` |
| `run_energy_implicit_phase.py:307` | `P_ext=2.0 n_r k_B T_ion` | Eq. Pext | ❌ `2.0`→`2.3` |
| `run_transition_phase.py:303` | `P_ext=2.0 n_r k_B T_ion` | Eq. Pext | ❌ `2.0`→`2.3` |
| `run_momentum_phase.py:244,428` | `P_ext=2.0 n_r k_B T_ion` | Eq. Pext | ❌ `2.0`→`2.3` |
| (all of the above) | `+= PISM*k_B` | `+P_ISM` | ✅ |
| `operations.py:191-197` | `c_s`, `μ=mu_ion/mu_atom` | mean mass/particle | ✅ |

**Totals:** ~6 pressure-prefactor sites, ~6 bubble sites, ~10 shell sites,
~4 cooling-factor sites, plus 2 flagged (BE μ, dead code).

---

## 2. Phased fix plan (each phase independently testable)

- **Phase 1 — ionised pressure prefactor `2.0 → μ_H/μ_p`.** All `P_HII`/`P_ext`
  sites (energy, implicit, transition, momentum). Self-contained; changes the
  HII/back-pressure by ×1.15. Verify: smoke test + assert factor = `mu_convert/mu_ion`.
- **Phase 2 — bubble interior `n` and `ρ`.** `n = (μ_p/μ_H)Pb/(k_BT)` (5 sites)
  and `ρ = μ_H n` (1 site). Makes `n=n_H` and fixes the ~2× mass/gravity deficit;
  also hands cooling a true `n_H`.
- **Phase 3 — shell to `n_H`.** ODE prefactors (ion `μ_p/μ_H`, neutral `μ_n/μ_H`),
  BC (Eq. nShell0), mass (`μ_H`), and all `(1+Z_He x_He)` recombination/Strömgren
  factors. Mass result is invariant (variable renormalises); the I-front position
  and `n_IF_Str` shift.
- **Phase 4 — cooling He-electron factor (FLAGGED).** Add `(1+Z_He x_He)` to the
  **CIE** `n²Λ` only after confirming the bundled CIE curves are normalised per
  `n_H n_e` (paper Eq. cooling). Non-CIE table already returns volumetric rate at
  `n_H` and needs only Phase 2's `n=n_H`.
- **Phase 5 — housekeeping.** Dead `get_shellParams.get_nShell0` (fix/delete) and
  the BE-sphere `μ` choice (§3.5).

## 3. Notes & open questions

### 3.5 Bonnor–Ebert `μ`
`bonnorEbertSphere.py` uses `mu_convert` (μ_H) as the EOS mean mass per particle
in `c_s²=γk_BT/μ` and `T_eff`. The paper's BE EOS is `c_s=(k_BT/μ)^{1/2}` with
`μ` = mean mass per *particle*; for a molecular/neutral cloud that is `μ_mol`
(or `μ_n`), not `μ_H`. This only rescales the `c_s↔T_eff` mapping (the profile
shape is fixed by `ξ_cl` and total mass), so it is cosmetic for dynamics but
wrong for any quoted cloud temperature. Confirm intended `μ` before changing.

## 4. Cooling normalisation caveat
Paper Eq. (cooling): `dU/dt = −n_H n_e Λ_net`. The code's CIE branch uses `n²Λ`.
Adding `(1+Z_He x_He)` is correct **iff** the bundled CIE `Λ` is tabulated per
`n_H n_e` (equivalently `n_H²` with the He electrons folded into `n_e`). If a
given curve is normalised per `n_tot²` or `n_e²`, the factor differs. The
non-CIE CLOUDY/opiate path is unaffected (it returns a volumetric rate keyed on
`n_H`). Resolve per-curve before applying Phase 4.
