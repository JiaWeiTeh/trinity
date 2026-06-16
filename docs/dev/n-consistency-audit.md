# `n` consistency audit against the paper (`n ≡ n_H`)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> **Audit status (2026-06-08):** **the implementation shipped in #657** — the
> `Implementation: NOT started` / "every ❌ site still open" framing is obsolete.
> The μ physics and values still hold, but the single-composition tables predate
> the **Phase A** split: the ~10⁴ K shell/HII region is now singly-ionised He
> (`mu_ion_shell=14/22`, `chi_e_shell=1.1`), while only the hot bubble stays
> doubly-ionised (`mu_ion=14/23`, `chi_e=1.2`).

Single source of truth for making **every** density/pressure/mass/cooling
term in TRINITY consistent with the model paper. Supersedes the convention
discussion in `pressure-terms-audit.md` (that file reached the *opposite*
sign on the bubble before the paper pinned the convention — see §0.1).

> **Paper convention (decisive):** *"All number densities `n` that follow
> denote hydrogen nuclei densities."* Every `n` in the code must therefore be
> `n_H`, with mass via `μ_H` and pressure via the `μ_H/μ` factor.

## Decisions (locked 2026-06) & verification status

- **Branch:** `hotfix/mu-audit`.
- **Composition is the single source of truth.** `x_He`, `Z_He` become input
  params; **everything else is derived** (μ_H, μ_n, μ_p, μ_mol, and the electron
  factor `chi_e = n_e/n_H = 1+Z_He·x_He`). Chosen over keeping the hardcoded μ.
- **Defaults stay byte-identical.** Naive float division shifts `mu_atom` by 1
  ULP (`float(Fraction('14/11'))=1.2727272727272727` vs `1.4/1.1=…25`), which
  can nudge the FP-sensitive bubble integrator. **Resolution:** do the Phase-0
  derivation in exact `fractions.Fraction` arithmetic (`x_He=Fraction(1,10)`,
  `Z_He=2`) and cast to float once → all four μ reproduce today's values
  exactly (verified), and `chi_e=1.2`. The other three μ are already
  bit-identical under either method.
- **Status:** every ❌ site below was read in the actual source (not grep,
  not comment) at audit time, and has since been fixed in #657. The μ *values*
  are correct; the defects were the **absent `chi_e`** and μ used with the
  **wrong partner**.
- **Do-not-trust-comments evidence:** `bubble_luminosity.py:970`
  (`rho = n*mu_ion  # Mass density`) asserts wrong physics — mass needs `μ_H`;
  the shell stores `nShell=ρ/μ_atom` (≈1.1 n_H) yet `dlaw.py:11` exports it to
  CLOUDY labelled `n_H` (mislabelled ×1.1).
- **Flagged at audit time (no assumption made), since resolved in #657:** §4
  CIE-curve normalisation (chi_e applied to the CIE branch); §3.5 BE EOS μ
  (left on μ_H — see implementation plan Phase 5.2/6B).
- **Implementation:** SHIPPED in #657 (`Hotfix/mu audit`). The fixes below
  landed in Phases 0–3, 5 and 6 of `n-consistency-implementation-plan.md`, with
  a later **Phase A** splitting shell vs bubble ionisation (see note below).
  This section is retained as the audit-time record; the ❌ verdicts mark what
  *was* defective and has since been fixed, not open work.

## 0. Canonical reference (derived from the paper)

Composition: `x_He = 0.1`, helium **doubly ionised** `Z_He = 2`.
Per H nucleus there are `(1+x_He)` particles when neutral, `(2+x_He(1+Z_He))`
when ionised, `(½+x_He)` when molecular, and `(1+Z_He·x_He)` free electrons.

| symbol | paper (derived from x_He, Z_He) | code param | value |
|---|---|---|---|
| `μ_H`  | `(1+4 x_He) m_H`, mass per H nucleus | `mu_convert` | `1.4 m_H` |
| `μ_n`  | `μ_H/(1+x_He)`, neutral mean mass/particle | `mu_atom` | `14/11 m_H` |
| `μ_p`  | `μ_H/(2+x_He(1+Z_He))`, ionised mean mass/particle | `mu_ion` | `14/23 m_H` |
| `μ_mol`| `μ_H/(½+x_He)`, molecular mean mass/particle | `mu_mol` | `14/6 m_H` |
| `chi_e`| `1+Z_He·x_He`, electrons per H nucleus `n_e/n_H` | **(absent)** | `1.2` |

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

> **Phase A correction (shipped in #657, after this table was written):** the
> table above assumes a single ionised composition (doubly-ionised He
> everywhere). The shipped code splits it by region. The **hot bubble** keeps
> these values (doubly-ionised, `Z_He=2` → `mu_ion=14/23`, factor
> `mu_convert/mu_ion=2.3`, `chi_e=1.2`). The **~10⁴ K shell / HII region** is
> singly-ionised (`Z_He_shell=1` → `mu_ion_shell=14/22`, factor
> `mu_convert/mu_ion_shell=2.2`, `chi_e_shell=1.1`). So for the **ionised gas
> pressure**, **electron density**, **volumetric cooling**, **recombination
> rate** and **Strömgren density** rows above, the bubble uses `2.3`/`1.2` but
> the shell/HII uses `2.2`/`1.1` (`mu_ion_shell`/`chi_e_shell`). Neutral-gas and
> mass-density (μ_H) rows are unaffected.

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
| `shell_structure.py:167,253,324,357` | mass/grav `nShell*mu_atom` (ion+neutral) | `ρ=μ_H n` | ❌ `mu_atom`→`mu_convert` |
| `shell_structure.py:135` | `max_shellRadius` Strömgren `α_B n²` | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `shell_structure.py:237-239` | `n_IF_Str` (Eq. nIF_Str) | `(1+Z_He x_He)` in denom | ❌ ×1.2 |
| `shell_structure.py:272-273` | `phi_hydrogen` recomb `α_B n²` | `(1+Z_He x_He)α_B n²` | ❌ ×1.2 |
| `shell_structure.py:298` | I-front jump `*mu_atom/mu_ion*T_ion/T_neu` | `μ_n/μ_p·T_ion/T_neu` (μ_H cancels) | ⚪ no change |
| `get_shellParams.py:30` | `nShell0=(mu_atom/mu_ion)…` (**dead**) | inverted **and** wrong μ | ❌ §5 (or delete) |

> **Phase A (#657):** these shell sites shipped using the singly-ionised shell
> composition, not the bubble `mu_ion`/`chi_e` shown in this audit-time table.
> The fixed code uses `mu_ion_shell` (factor 2.2) for the ionised `nShell0` BC
> and ODE pressure prefactor, and `chi_e_shell = 1.1` for the recombination /
> Strömgren factors (in place of the `×1.2` written above). See
> `get_shellODE.py` (`mu_p=mu_ion_shell`, `chi_e=chi_e_shell`) and
> `shell_structure.py:115,135,239,273,298`.

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

> **Phase A (#657):** the `P_HII`/`P_ext` sites above are the **HII/shell**
> ionised pressure, so the shipped prefactor is `mu_convert/mu_ion_shell = 2.2`
> (singly-ionised), **not** the `2.3` shown (which is the bubble's
> `mu_convert/mu_ion`). The bubble interior `n`/`ρ`/CIE sites in the
> bubble-structure table above are unaffected — they correctly keep `mu_ion`
> (2.3) and `chi_e` (1.2).

---

## 2. Phased fix plan (each phase independently testable)

- **Phase 0 — composition foundation (locked design).** Add `x_He` (1/10),
  `Z_He` (2) as input params. Derive μ_H, μ_n, μ_p, μ_mol and `chi_e` from them
  using exact `Fraction` arithmetic, cast to float once → byte-identical μ
  defaults (verified) + new `chi_e=1.2`. Expose `chi_e` (and the ratios
  `mu_convert/mu_ion`, `mu_convert/mu_atom`) for the later phases. No physics
  result changes in Phase 0 alone (pure plumbing); verify with full suite.
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
