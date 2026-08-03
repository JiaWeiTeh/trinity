# Cross-cutting sweep ① — units & dimensions

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

## Scope, method, and what I did not read

**Read (in full or in the regions that carry units):** the real source under
`/home/user/trinity/trinity/**` — `_functions/unit_conversions.py`, `_input/{registry,read_param,
param_spec,dictionary,fkappa_auto,sweep_runner,sweep_parser}.py`, `_input/default.param`,
`_output/{trinity_reader,show_run,simulation_end,terminal_prints,header,run_constants,_metadata_io}.py`,
`_output/cloudy/{dlaw,snapshot_to_deck,run_loader}.py`, `sps/{read_sps,sps_columns,update_feedback}.py`,
`cooling/net_coolingcurve.py`, `cooling/CIE/read_coolingcurve.py`, `cooling/non_CIE/read_cloudy.py`,
`bubble_structure/{bubble_luminosity,get_bubbleParams}.py`, `shell_structure/{shell_structure,get_shellODE}.py`,
`cloud_properties/{density_profile,mass_profile,powerLawSphere,bonnorEbertSphere,validate_gmc,initial_profile}.py`,
`phase0_init/*`, `phase1_energy/*`, `phase1b_energy_implicit/get_betadelta.py` (unit-bearing regions),
`phase1c_transition/`, `phase2_momentum/`, `phase_general/phase_events.py`, `_functions/{operations,cluster,simplify}.py`,
`main.py`. Consumption-only reads: `test/test_conventional_units.py`, `test/test_mu_audit_drift.py`,
`test/test_net_coolingcurve.py`, `test/test_metadata.py`, `tools/make_density_profile_gif.py`,
`tools/bubble_audit/reference.py`, `paper/methods/figures/paper_densityProfile.py`,
`paper/methods/figures/paper_rcloud_smoothing.py`. Also read two bundled data files to settle a
normalisation question: `lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06.dat` and
`lib/default/CIE/coolingCIE_3_Gnat-Ferland2012.dat`.

**Not read:** `docs/dev/code-audit/slices/` (deliberately — prior agents' conclusions),
`old_doNotRead/`, `outputs/`, `scratch/`, `tbd/`, `fig/`. I did not read
`docs/dev/code-audit/data/claims_prose.csv` either; I worked from the source and the registry
directly, which is a superset of that extract for unit-bearing declarations.

**Method.** Every claim below is arithmetic on quoted code, not on comments. I re-derived all 18
conversion constants against first-principles CGS↔astro definitions (below), then dimensionally
cancelled each formula. Where a conversion could only be settled empirically (the OPIATE cooling
cube's normalisation) I ran the numbers on the shipped table rather than trusting the docstring.
I ran `read_param('param/simple_cluster.param')` once, read-only, to confirm the applied conversions
numerically. **No source file was modified.**

---

## 1. The unit-system contract

`trinity/_functions/unit_conversions.py` is the whole contract. Internal ("astronomy", AU) units are
**Msun / pc / Myr**, with temperature in **K** (unconverted, `unit_map['K'] = 1.0`,
`unit_conversions.py:379`).

The module exposes three layers, all consistent:

* `CONV` (frozen dataclass, `:57`) — CGS→AU factors; flat re-exports `cvt.<name>_cgs2au` at `:257–303`.
* `INV_CONV` (`:155`) — the exact reciprocals, `cvt.<name>_au2cgs`.
* `CGS` (`:192`) — CGS physical constants, flat re-exports `cvt.*_CGS` at `:236–243`.

Two names are *original* (not re-exports), `unit_conversions.py:287,289`:
`Pb_au2_KcmInv = Pb_au2cgs / K_B_CGS` (internal pressure → `K cm⁻³`, i.e. P/k_B) and
`Mdot_au2Msunyr = 1e-6`.

`convert2au(unit_string)` (`:315`) parses a unit string into a multiplicative factor with no `eval`.
Its base map (`:367–382`) is `g→g2Msun`, `s→s2Myr`, `cm→cm2pc`, `km→km2pc`, `erg→E_cgs2au`,
`m_H→CGS.m_H*g2Msun`, and `K/Zsun/Msun/pc/Myr → 1.0`.

**Verification of the constants (independent re-derivation, all agree to ≤2 ulp):**

| constant | value in file | first-principles check |
|---|---|---|
| `cm2pc` | 3.240779289444365e-19 | 1/pc_cm, ratio 1.0000000 |
| `g2Msun` | 5.029144215870041e-34 | 1/Msun_g, ratio 1.0000000 |
| `s2Myr` | 3.168808781402895e-14 | 1/(3.15576e13 s), ratio 1.0000000 |
| `ndens_cgs2au` | 2.937998946096347e+55 | pc_cm³ = 2.9379989460963475e55 |
| `E_cgs2au` | 5.260183968837699e-44 | Myr²/(Msun·pc²) |
| `L_cgs2au` | 1.6599878161499254e-30 | Myr³/(Msun·pc²) |
| `Pb_cgs2au` | 1545441495671.806 | pc·Myr²/Msun |
| `Lambda_cgs2au` | 5.650062667161655e-86 | Myr³/(Msun·pc⁵) |
| `dudt_cgs2au` | 4.877042454381257e+25 | pc·Myr³/Msun |
| `c_therm_cgs2au` = `pdotdot_cgs2au` | 5.122187189842638e-12 | Myr³/(Msun·pc) |
| `F_cgs2au` = `pdot_cgs2au` | 1.623123174716277e-25 | Myr²/(Msun·pc) |
| `G_cgs2au` | 67400.3588611473 | Msun·Myr²/pc³ |
| `tau_cgs2au` | 4788.452460043275 | pc²/Msun |
| `phi_cgs2au` | 3.0047272630641653e+50 | pc²·Myr |
| `gravPhi_cgs2au` | 1.045940172532453e-10 | Myr²/pc² |
| `grav_force_m_cgs2au` | 322743414.19646025 | Myr²/pc |
| `v_kms2au` | 1.022712165045695 | km·Myr/(s·pc) |

`Pb_au2_KcmInv = 4686.6676` is independently reproducible: 1 Msun pc⁻¹ Myr⁻² =
1.988e33/(3.0857e18 × (3.15576e13)²) = 6.469e-13 dyn cm⁻², ÷ k_B = 4686.5 K cm⁻³. ✔

**The key AU/CGS scale factors to keep in mind for the findings below:**
pc⁻³ vs cm⁻³ = **2.938e55**; internal energy unit vs erg = **1.901e43**; internal luminosity vs
erg/s = **6.024e29**.

**One structural fact that matters for §1:** the unit actually *applied* at load does **not** come
from `registry.py`. `read_param` Step 1 (`read_param.py:120–170`) parses `# UNIT:` lines out of
`trinity/_input/default.param`, and Step 4 (`read_param.py:253–274`) multiplies by
`cvt.convert2au(unit)` using *that* string. `registry.py`'s `unit=` field is only ever stored as a
label (`registry.py:617,659` — `ori_units=spec.unit`), never used to convert. I mechanically diffed
all 91 `default.param` parameters against `SPECS`: **zero unit mismatches and zero default-value
mismatches**. The two are in sync today; the hazard is structural, not present (see clearance C1).

**`ori_units` is semantically overloaded, by design.** For `input_*` specs it is the *pre-conversion*
unit (`nCore.ori_units == 'cm**-3'` while `nCore.value` is pc⁻³). For runtime/derived specs it is the
*post-conversion* internal unit (`shell_nMax` → `1/pc**3`). Both conventions are defensible; mixing
them in one field is what makes UNIT-02 easy to introduce.

---

## 2. Declared-vs-applied axis (registry → `convert2au` → consumers)

I loaded `param/simple_cluster.param` and read back every unit-bearing parameter, then checked each
consumer's dimensional expectation. Measured internal values:

```
nCore        2.938e+60 pc^-3   (= 1e5 cm^-3 exactly on round-trip)
nISM         2.938e+55 pc^-3
mu_convert   1.1783e-57 Msun   (= 1.4 m_H exactly)
mu_ion       5.1231e-58 Msun   mu_ion_shell 5.3560e-58   mu_atom 1.0712e-57
k_B          7.2625e-60 Msun pc^2 Myr^-2 K^-1
G            4.4985e-03 pc^3 Msun^-1 Myr^-2   (textbook 4.4986e-3) ✔
c_light      3.0660e+05 pc/Myr  (-> 299792.458 km/s) ✔
FB_vSN       1.0227e+04 pc/Myr  (-> 10000 km/s) ✔
caseB_alpha  2.7820e-55 pc^3/Myr
dust_sigma   1.5754e-58 pc^2    (-> 1.5e-21 cm^2) ✔
dust_KappaIR 8.3534e-04 pc^2/Msun (-> 4.0 cm^2/g) ✔
C_thermal    3.0733e-18 Msun pc Myr^-3 K^-7/2
PISM         0.0        K pc^-3
```

Consumer-side dimensional checks (all pass — see clearances for the full arithmetic):

* `PISM` (declared `K * cm**-3`, `registry.py:381`) → internal `K pc⁻³`. Consumed as
  `P_ext += PISM * k_B` (`energy_phase_ODEs.py:244`, `run_momentum_phase.py:254,438`,
  `run_transition_phase.py:314`, `run_energy_implicit_phase.py:521`).
  `[K pc⁻³] × [Msun pc² Myr⁻² K⁻¹] = Msun pc⁻¹ Myr⁻²` = the internal pressure unit. ✔
  This is the archetype the sweep was told to look for, and it is **correct**.
* `dust_sigma` (`cm**2` → pc²) is used only as `n·σ·dr` (`get_shellODE.py:120,122`,
  `shell_structure.py:278`): `pc⁻³·pc²·pc` = dimensionless optical depth. ✔ Note the Z-scaling at
  `read_param.py:367–370` runs *after* Step 4's conversion and multiplies by a dimensionless `ZCloud`,
  so it cannot disturb the unit.
* `dust_KappaIR` (`cm**2 * g**-1` → pc²/Msun) is used only as
  `shell_tauKappaRatio * dust_KappaIR` (five call sites, e.g. `energy_phase_ODEs.py:135`).
  `shell_tauKappaRatio = mu_convert · Σ n dr` (`shell_structure.py:390,394`) = `Msun·pc⁻³·pc` =
  Msun/pc², matching `registry.py:468`. Product is dimensionless. ✔
* `caseB_alpha` (`cm**3 * s**-1` → pc³/Myr) in `chi_e·n²·α_B` gives `pc⁻⁶·pc³/Myr = pc⁻³/Myr`. ✔
* `C_thermal` (`erg*s**-1*cm**-1*K**(-7/2)`) → `Msun pc Myr⁻³ K⁻⁷ᐟ²`; this is exactly
  `c_therm_cgs2au`, and the ODE term `Pb/(C·T^{5/2})·[1/Myr]` lands in K/pc². ✔
* `FB_vSN` (`km * s**-1` → pc/Myr) feeds `Mdot_SN = 2·Lmech_SN/v²` (`read_sps.py:228,232`):
  `(Msun pc² Myr⁻³)/(pc² Myr⁻²) = Msun/Myr`. ✔
* `TShell_neu` / `TShell_ion` carry `unit=None` (`registry.py:370,371`) — harmless, since `K` maps to
  1.0 anyway, but they are the only physical quantities in the registry without a unit string.

**No declared/applied/consumed mismatch found on this axis.**

---

## 3. Density conventions — testing `registry.py:366`

The claim under test (the `mu_convert` info string, `registry.py:366`): *"All densities n in TRINITY
are hydrogen-nuclei densities n_H."*

I traced every density-bearing site. The claim **holds**, and the composition factors are each
applied exactly once and are the right ones:

* **Mass conversion** always uses `mu_convert` = mass per H nucleus = `(1+4·x_He)·m_H` — verified
  1.4 m_H at x_He=0.1. Sites: `mass_profile.py:126,307,308`, `powerLawSphere.py:71,131`,
  `shell_structure.py:255,333,371,390,394` (shell mass and shell gravity),
  `bubble_luminosity.py:930` (bubble mass), `get_InitPhaseParam.py:146` (`rhoa = nCore*mu_convert`),
  `bonnorEbertSphere.py:403,611,651`. Not once is an ionisation-dependent `mu` used for mass. ✔
* **Pressure ↔ density** always uses the *particles-per-H-nucleus* ratio `mu_convert/mu_<state>`:
  measured `mu_H/mu_ion = 2.300`, `mu_H/mu_ion_shell = 2.200`, `mu_H/mu_atom = 1.100`. These are
  exactly n_tot/n_H for (H⁺+He²⁺+e⁻) = 1+0.1+1.2, (H⁺+He⁺+e⁻) = 1+0.1+1.1, and (H+He) = 1+0.1. ✔
  Sites: `energy_phase_ODEs.py:54`, `bubble_luminosity.py:427,673,725,778,811`,
  `shell_structure.py:124`, `get_shellODE.py:115,140`, and the four `P_HII` sites
  (`run_energy_phase.py:214`, `run_energy_implicit_phase.py:981,1379`,
  `run_transition_phase.py:564,845`, `run_momentum_phase.py:634`).
* **Region-dependent ionisation is honoured.** The hot bubble uses `mu_ion`/`chi_e` (He²⁺, Z_He=2);
  the ~1e4 K shell/HII region uses `mu_ion_shell`/`chi_e_shell` (He⁺, Z_He_shell=1). Derived once at
  `read_param.py:310–348`. I found **no cross-leak**: `mu_ion` never appears in a shell formula and
  `mu_ion_shell` never in a bubble formula.
* **`chi_e = n_e/n_H` is applied exactly once**, only where the cooling/recombination rate is
  ∝ n_e·n_H:
  - CIE branch, `net_coolingcurve.py:164,187` and `bubble_luminosity.py:746,833`:
    `chi_e · n_H² · Λ` = `n_e n_H Λ`, correct **iff** the CIE table is normalised per n_e n_H.
    `lib/default/CIE/coolingCIE_3_Gnat-Ferland2012.dat` is the Gnat & Ferland tabulation, whose Λ is
    defined by (cooling rate) = n_e n_H Λ — so this is right (see confidence note in clearance C6).
  - Non-CIE branch: `chi_e` is **not** applied (`net_coolingcurve.py:154`,
    `bubble_luminosity.py:785–791,824–831`) because the OPIATE cube values are already volumetric
    rates that carry n_e internally (the file has its own `nedens` column). Correct — applying
    `chi_e` there would double-count.
  - `chi_e_shell` multiplies n_H² in recombination and in the Strömgren balance
    (`get_shellODE.py:117,120`, `shell_structure.py:144,248,282`). ✔
* **Strömgren balance is dimensionally exact.** `shell_structure.py:246–249`:
  `n_IF_Str = sqrt(3·Q_abs/(4π·χ_e·α_B·(R_IF³−R2³)))`. Inverting
  `Q = (4π/3)(R³−R2³)·χ_e n_H² α_B` gives precisely that. ✔
* **`shell_nMax < nISM`** (`shell_structure.py:445`) compares an ODE-derived n_H in pc⁻³ against
  `nISM` in pc⁻³. ✔

**One documentation counterexample, not a code counterexample:** `_output/trinity_reader.py`'s
`PARAM_DOCS` labels six of these n_H quantities as `[cm^-3]`, and `registry.py:439` labels one as
`1/cm**3`. See UNIT-01 and UNIT-02.

---

## 4. Pressure, energy, luminosity — sites that convert, and sites that should

**Sites that convert (all verified correct):**

* `bubble_E2P` (`get_bubbleParams.py:198–238`) is the only physics routine that round-trips through
  cgs: `r1,r2 *= pc2cm; Eb *= E_au2cgs`; `Pb = (γ−1)Eb/V/(4π/3)` in erg/cm³ = dyn/cm²; returns
  `Pb * Pb_cgs2au`. ✔ (The `r2 += 1e-10` at `:221` is 1e-10 **cm**, i.e. ~3e-29 relative — inert;
  the real guard is the `shell_volume <= 0` branch at `:227`.)
* `sps_columns.UNIT_CONVERSIONS` (`:113–152`): `yr→Myr = 1e-6` ✔; `1/s→1/Myr = 1/s2Myr` ✔;
  `erg/s→AU = L_cgs2au` ✔; `L_sun = 3.828e33 × L_cgs2au` ✔; `g/s→Msun/Myr = g2Msun/s2Myr` ✔;
  `cm/s→pc/Myr = cm2pc/s2Myr = 1.0227e-5 = v_cms2au` ✔.
* `get_InitPhaseParam.py:170–176` — Weaver Eq 37: `L` converted with `L_au2cgs` and divided by
  `WEAVER_L_REF = 1e36 erg/s`, `n` converted with `ndens_au2cgs` to cm⁻³, and `dt_phase0` left in
  **Myr**, which is Weaver's `t₆` (units of 10⁶ yr). All three arguments are in the units the
  coefficient 1.51e6 K assumes. ✔ This is a subtle one and it is right.
* `get_InitPhaseParam.py:193–195` — the SF-onset summary: `log10(Qi·s2Myr)` [1/s] ✔
  (`s2Myr` has dimensions Myr/s, so `[1/Myr]×[Myr/s] = [1/s]`), `Lbol·L_au2cgs` erg/s ✔,
  `Mdot0·Mdot_au2Msunyr` Msun/yr ✔, `E0·E_au2cgs` erg ✔.
* `cloudy/dlaw.py:174–177` — `log_r_cm = log10(r_pc)+log10(pc2cm)`, `log_n_cm3 = log_n_pc3 +
  log10(ndens_au2cgs)` (offset −55.4681). ✔
* `cloudy/snapshot_to_deck.py:181` — `log_qh = log10(Qi) − log10(Myr2s)`: `[1/Myr]/(s per Myr)` =
  `[1/s]`. ✔
* `operations.get_soundspeed` (`:189–211`) — the only place a *cgs* computation is built from
  au parameters: `mu = params['mu_ion']·Msun2g` [g], `k_B·k_B_au2cgs` [erg/K], result in cm/s, then
  `× v_cms2au` → pc/Myr. ✔ And it correctly uses `mu_ion`/`mu_atom` (per particle), **not**
  `mu_convert`, which would give a sound speed high by √(2.3) = 1.52.
* `fkappa_auto.py:121` — `nCore * ndens_au2cgs` before the cm⁻³-indexed lookup grid, with a comment
  naming the reason. ✔
* `sweep_runner.py:96–127` — the pre-flight GMC validator explicitly re-applies
  `convert2au('cm**-3')` and `convert2au('m_H')` because sweep values bypass `read_param`. Complete
  and correct (mCloud, rCore, rCloud_max are identity; Omega/gamma/alpha dimensionless). ✔

**Sites that should convert and do not:** I found none in the physics path. The two candidates
I chased both turned out fine:
* `get_leak_luminosity` (`get_bubbleParams.py:240–283`) applies **no** conversion, and is right to:
  `Pb·c_sound·R2² = (Msun pc⁻¹ Myr⁻²)(pc Myr⁻¹)(pc²) = Msun pc² Myr⁻³`. ✔
* `pRam` (`get_bubbleParams.py:285–307`): `L/(2πr²v) = (Msun pc² Myr⁻³)/(pc²·pc Myr⁻¹) =
  Msun pc⁻¹ Myr⁻²`. ✔

---

## 5. Cross-module handoffs

### 5a. Phase runners ↔ bubble structure

`_get_bubble_ODE` (`bubble_luminosity.py:409–447`) is the hot loop; every term cancels:

```
ndens = Pb / ((mu_H/mu_ion)·k_B·T)  ->  (Msun pc^-1 Myr^-2)/(Msun pc^2 Myr^-2) = pc^-3   ✔
phi   = Qi/(4 pi r^2)               ->  (1/Myr)/pc^2 = 1/(pc^2 Myr)                      ✔  (matches
                                        get_dudt's declared phi [1/pc2/Myr], line 67)
dTdrr = Pb/(f_kappa*C_thermal*T^{5/2}) * [ (beta+2.5 delta)/t  +  2.5 (v-v_t) dTdr/T  -  dudt/Pb ]
        prefactor: (Msun pc^-1 Myr^-2)/(Msun pc Myr^-3 K^-1) = K Myr / pc^2
        bracket  : every term is 1/Myr   (dudt/Pb = (Msun pc^-1 Myr^-3)/(Msun pc^-1 Myr^-2))
        product  : K/pc^2  ==  d^2T/dr^2                                                  ✔
        tail     : -2.5 dTdr^2/T - 2 dTdr/r  =  (K/pc)^2/K, K/pc/pc  =  K/pc^2            ✔
```

`_get_init_dMdt` (`:297–308`), the Weaver Eq 33 seed, also cancels exactly:
`pc³/Myr · Myr²K/pc² · (Msun pc⁻¹Myr⁻²K⁻⁷ᐟ²)^{2/7} · (Msun pc⁻¹Myr⁻²)^{5/7}` →
pc exponent 3−2−2/7−5/7 = 0, Myr exponent −1+2−4/7−10/7 = −1, Msun exponent 2/7+5/7 = 1 ⇒
**Msun/Myr**. ✔ Likewise `_get_bubble_ODE_initial_conditions` (`:387–407`): `dR2` → pc, `T` → K,
`v` → pc/Myr, `dTdr` → K/pc. ✔

`solve_R1`/`get_r1` (`get_bubbleParams.py:384–411`):
`sqrt(L/v/E·(r2³−r1³)) = sqrt((Msun pc Myr⁻²)/(Msun pc² Myr⁻²)·pc³) = pc`. ✔

### 5b. Cooling tables ↔ luminosity

This is the boundary with the most conversions, and the two independent paths are **exactly**
consistent — a satisfying cancellation worth recording:

* CIE-in-AU path (`bubble_luminosity.py:744–746`):
  `Λ_au = Λ_cgs · Lambda_cgs2au`, integrand `= χ_e·n_au²·Λ_au·4πr_au²`.
* CIE-in-cgs path (`net_coolingcurve.py:163–165`):
  `dudt_cgs = χ_e·n_cgs²·Λ_cgs`, returned `× dudt_cgs2au`.

Equality requires `Lambda_cgs2au · ndens_cgs2au² == dudt_cgs2au`:
`5.650063e-86 × (2.937999e55)² = 5.650063e-86 × 8.63184e110 = 4.87704e25 = dudt_cgs2au`. ✔ Exact.

Non-CIE lookups feed the cube in cgs on all three axes and convert back with `dudt_cgs2au`
(`net_coolingcurve.py:82–83,154,156`; `bubble_luminosity.py:785–791,824–831`), matching the cube's
axis units (`read_cloudy.py:154–159`: log cm⁻³, log K, log cm⁻² s⁻¹). ✔

Integrand → luminosity: `dudt[Msun pc⁻¹ Myr⁻³] · 4πr²[pc²] · dr[pc] = Msun pc² Myr⁻³`. ✔

**What is wrong here is the label, not the code**: `read_cloudy.get_coolingStructure`'s docstring
(`:29`) says *"Cooling rate is in units of [erg cm3 / s]"* — i.e. a Λ. It is not; it is a volumetric
rate. Proof from the shipped table (`opiate_cooling_rot_Z1.00_age1.00e+06.dat`, T=3162 K, φ=1):

```
n_H        n_e        cool        cool/(n_e n_H)   cool/n_H^2
1.000e-04  4.400e-05  3.7372e-33  8.494e-25        3.737e-25
1.000e-03  2.210e-04  1.8680e-31  8.453e-25        1.868e-25
1.000e-02  9.330e-04  1.0941e-29  1.173e-24        1.094e-25
3.162e-02  1.820e-03  9.1258e-29  1.586e-24        9.126e-26
```

`cool/(n_e n_H)` is flat (~8–16e-25) across four decades while `cool` spans five orders; `cool/n_H²`
is not flat. So `cool` is **erg cm⁻³ s⁻¹**, exactly what `dudt_cgs2au` assumes. → UNIT-04.

### 5c. SPS ↔ feedback

`read_sps._read_sps_user` (`:134–288`) converts each column to canonical AU *before* any derivation,
then every derivation is a pure AU identity:
`Mdot_wind = pdot²/(2L) = (Msun pc Myr⁻²)²/(Msun pc² Myr⁻³) = Msun/Myr` ✔;
`v = 2L/pdot = pc/Myr` ✔; `Lmech = ½Mdot v² = Msun pc² Myr⁻³` ✔;
`v_mech_total = 2 Lmech_total/pdot_total` (`update_feedback.py:181`) = pc/Myr ✔.
The mass-scaling factor `f_mass` is dimensionless and applied once per mass-scaled column
(`read_sps.py:174`). The `SPSFeedback` docstring's *"Qi [1/Myr] (× s2Myr → 1/s)"*
(`update_feedback.py:42,125`) is **correct** (s2Myr carries Myr/s). ✔

### 5d. Shell structure ↔ phase runners

`get_shellODE` ionised branch (`:115–122`), full cancellation:

```
term1 = n sigma/(4 pi r^2 c) (Ln e^-tau + Li phi)
      = pc^-3 * pc^2 / (pc^2 * pc Myr^-1) * Msun pc^2 Myr^-3 = Msun pc^-2 Myr^-2
term2 = chi_e n^2 alpha_B Li/(Qi c)
      = pc^-6 * pc^3 Myr^-1 * Msun pc^2 Myr^-3 * Myr * Myr pc^-1 = Msun pc^-2 Myr^-2   (same) ✔
dndr  = (mu_p/mu_H)/(k_B T) * [terms] = (Msun pc^-2 Myr^-2)/(Msun pc^2 Myr^-2) = pc^-4  ✔
dphidr= -4 pi r^2 chi_e alpha_B n^2/Qi - n sigma phi = pc^-1, pc^-1                      ✔
dtaudr= n sigma f_cover = pc^-1                                                          ✔
```

Shell gravity (`shell_structure.py:259–263`): `grav_phi = −4πG∫r ρ dr = pc²/Myr²` (matches
`registry.py:470`); `grav_force_m = G·m/r² = pc/Myr²` (matches `registry.py:471`). ✔ — but see
UNIT-03 for what happens to the latter on the way into the snapshot.

`F_rad = f_abs·Lbol/c·(1+τ/κ·κ_IR)` (`energy_phase_ODEs.py:133–135` and four twins):
`(Msun pc² Myr⁻³)/(pc Myr⁻¹) = Msun pc Myr⁻²` = force. ✔
`F_grav = G·mShell/R2²·(mCluster+½mShell) = Msun pc Myr⁻²`. ✔

### 5e. Physics ↔ `_output`

`metadata.json[final_state]` and every snapshot value are **internal units** — stated explicitly at
`simulation_end.py:12–15,145–147` and corroborated by `show_run.py:136–146`, which re-applies
`INV_CONV.ndens_au2cgs` on read. The display layers all convert correctly (see clearance C5). The
defects are in the *documentation* layer that sits between: `PARAM_DOCS` (UNIT-01) and one registry
entry (UNIT-02).

One naming hazard worth knowing when reading across modules: `phi` means a **dimensionless
attenuation fraction** in `shell_structure`/`get_shellODE`, and an **ionising photon number flux**
[pc⁻² Myr⁻¹] in `bubble_luminosity`/`net_coolingcurve`. Both are internally consistent; there is no
code path that crosses them. Not filed as a finding.

---

## 6. Clearances — boundaries checked and found correct

Recording these so a future session does not re-chase them.

**C1 — registry `unit=` vs `default.param` `# UNIT:`.** Only `default.param` is read by the
converter (`read_param.py:143–146,262`). I diffed all 91 input parameters mechanically: unit strings
and default strings agree **exactly**, zero drift. The registry's "single source of truth" claim
(`registry.py:1–5`) is therefore true in effect today, but is enforced only by a test, not by the
load path — worth knowing if `default.param` is ever hand-edited.

**C2 — `PISM` (`K*cm**-3` → `K pc⁻³`) × `k_B` → internal pressure.** Verified at all five consumer
sites. The exact case the sweep brief flagged as the archetype; it is correct.

**C3 — Weaver Eq 37 mixed-unit call** (`get_InitPhaseParam.py:170–176`): `L` in erg/s over 1e36,
`n` in cm⁻³, `t` in Myr (= Weaver's t₆), `T` out in K. Three different unit systems in one
expression, all right.

**C4 — CIE dual path exactness.** `Lambda_cgs2au × ndens_cgs2au² = dudt_cgs2au` to full precision
(4.877042454381e25 both ways). The AU-space and cgs-space CIE cooling paths cannot disagree.

**C5 — every display conversion in `_output`.**
`terminal_prints._STATE_FIELDS` (`:130–139`): v2 ×`v_au2kms` "km/s" ✔, Eb ×`E_au2cgs` "erg" ✔,
Pb ×`Pb_au2_KcmInv` "K/cm3" ✔ (P/k_B, factor 4686.67 re-derived above), t/R2/R1/Mshell/T0 identity ✔.
`show_run._cloud_section` (`:139–148`) and `_final_state_section` (`:180–208`) ✔ — and they print the
internal value in parentheses, which is exactly the right habit.
`simulation_end.CRITICAL_PARAMS` (`:408–433`) ✔, and it honestly labels the forces `'code'` with
factor 1.0 rather than inventing a unit.
`header.py:91,94` ✔ (`mCloud/(1−sfe)` correctly recovers the pre-SFE input, since Step 6 rebinds
`mCloud` to the post-SFE residual, `read_param.py:381–385`).
`cloudy/dlaw.py`, `cloudy/snapshot_to_deck.py` ✔.
`tools/make_density_profile_gif.py:102,109,141,142` ✔ (correctly un-logs *and* converts).
`paper/methods/figures/paper_densityProfile.py:163,262,460–464` and `paper_rcloud_smoothing.py:365,378`
✔ — the paper figures do convert, so UNIT-01 has **not** propagated into a published figure.

**C6 — `chi_e` applied exactly once.** Verified across CIE (applied), non-CIE (correctly not
applied — the cube carries n_e), recombination and Strömgren (applied via `chi_e_shell`). Confidence
on the *CIE table normalisation* specifically is **medium**: it rests on Gnat & Ferland's published
definition (rate = n_e n_H Λ), which I could not verify from the shipped file — the file is a bare
two-column `log T, log Λ` with no header. If a user swaps in a table normalised per n_H² or per
n_tot², `chi_e` becomes a spurious 1.2× on L1/L3. Worth a header comment in the data file; not filed
as a finding because the shipped default is right.

**C7 — Bonnor–Ebert `mu` choice.** `create_BE_sphere` (`bonnorEbertSphere.py:400–431`) derives
`c_s` from `M` and `ρ_c`, then defines `T_eff = mu_convert·MSUN_TO_G·c_s²/(γ k_B)`; `r2xi`/`xi2r`
(`:606,646`) invert it with the *same* `mu_convert` and `gamma`, so mu and γ cancel exactly in the
r↔ξ round-trip. Using `mu_convert` (mass per H nucleus) rather than `mu_mol` (mass per particle)
means `densBE_Teff` is **not** a gas kinetic temperature — which is stated in `registry.py:525` and
pinned by `test/test_mu_audit_drift.py:289–306`. Deliberate, self-consistent, and documented.
**Not a defect.**

**C8 — sweep pre-flight conversion.** `sweep_runner._validate_sweep_gmc` re-applies exactly the
conversions `read_param` would (`:96–127`), covering nCore, nISM, mu_convert and correctly treating
mCloud/rCore/rCloud_max as identity. The sweep folder-name tag
(`sweep_parser.generate_run_name`, `:718–722`) uses the *raw* `.param` values (cm⁻³, Msun), which is
the documented convention (`registry.py` `mCloud_input` info: "Matches the .param file and the sweep
folder-name tag"). ✔

**C9 — `get_soundspeed` uses per-particle mu.** `operations.py:207–211` uses `mu_ion`/`mu_atom`, not
`mu_convert`. Had it used `mu_convert` the hot-bubble sound speed would be 1.52× too large and
`get_leak_luminosity` and the transition-phase `Ed_soundcrossing` would inherit it. It does not.

**C10 — no drift between `registry` constant defaults and `CGS`.** `G` 6.6743e-08 vs `CGS.G`
6.67430e-8 ✔, `k_B` 1.380649e-16 ✔, `c_light` 29979245800 ✔. Two sources, same numbers.

---

## 7. Method notes and residual uncertainty

* Everything in §§1–5 was settled statically plus one read-only `read_param` call. Nothing here
  required running a simulation.
* The one claim I could not close from source is the CIE table's normalisation convention
  (clearance C6, `medium`).
* `_BUBBLE_ATOL = 1e-10` (`bubble_luminosity.py:93`) is a single scalar absolute tolerance applied to
  a mixed-unit state vector `[v (pc/Myr), T (K), dTdr (K/pc)]`. It is far below every component's
  working magnitude, so it is effectively rtol-only and harmless today; noted here rather than filed
  because it is a numerical-hygiene point, not a unit mismatch.
* I did not audit sign conventions, guard reachability, or coefficient provenance except where a
  unit question forced it.

---

```json
[
  {
    "id": "UNIT-01",
    "file": "trinity/_output/trinity_reader.py",
    "line": 203,
    "class": "units",
    "severity": "S3",
    "claim": "PARAM_DOCS — the public reader's documentation map — labels six density quantities as [cm^-3] while the stored snapshot/metadata values are internal pc^-3, a factor of 2.938e55.",
    "evidence": "trinity_reader.py:203 'shell_n0': '... [cm^-3]', :204 'shell_nMax': '... [cm^-3]', :205 'nEdge': '... [cm^-3]', :209 'nCore': '... [cm^-3]', :210 'nISM': '... [cm^-3]', :221 'initial_cloud_n_arr': '... [cm^-3]'. The stored values are pc^-3: (a) show_run.py:136-146 comments 'nCore/nISM are stored internally in pc-3' and multiplies md['nCore'] by INV_CONV.ndens_au2cgs before display; (b) simulation_end.py:12-15,145-147 states metadata.json[final_state] is 'in INTERNAL units (pc/Myr, pc^-3)'; (c) registry.py:467 shell_nMax unit='1/pc**3', :478 shell_n0 unit='1/pc**3', :437 nEdge unit='1/pc**3'; (d) shell_structure.py:124 derives nShell0 = (mu_ion_shell/mu_convert)/(k_B*TShell_ion)*Pb which is pc^-3 by construction; (e) measured on param/simple_cluster.param, params['nCore'].value = 2.938e60 for an input of 1e5 cm^-3. PARAM_DOCS is surfaced next to the stored value by TrinityOutput.info(verbose=True), trinity_reader.py:1013,1032.",
    "expected": "Labels of '[pc^-3] (internal; x INV_CONV.ndens_au2cgs -> cm^-3)', matching the convention already used for Eb (:162), Qi (:182) and the luminosity block (:174-181).",
    "failure_scenario": "A user or analysis script calls output.info(verbose=True), reads 'Maximum shell number density [cm^-3]', and plots or thresholds snapshot['shell_nMax'] directly as cm^-3. Every derived number is wrong by 2.938e55. The in-repo paper figures happen to convert correctly (paper_densityProfile.py:262,460), so this has not yet corrupted a published figure — but it is the documented contract of the public reader API.",
    "repro": "python -c \"from trinity._output.trinity_reader import PARAM_DOCS; print(PARAM_DOCS['nCore'], '|', PARAM_DOCS['shell_nMax'])\" then compare against: python -c \"from trinity._input.read_param import read_param; import trinity._functions.unit_conversions as cvt; p=read_param('param/simple_cluster.param'); print(p['nCore'].value, p['nCore'].value*cvt.ndens_au2cgs)\"",
    "confidence": "high"
  },
  {
    "id": "UNIT-02",
    "file": "trinity/_input/registry.py",
    "line": 439,
    "class": "units",
    "severity": "S3",
    "claim": "The runtime spec initial_cloud_n_arr declares unit='1/cm**3', but the array it labels is built in internal pc^-3 — inconsistent with every sibling runtime density spec and with the reader's own reconstruction docstring.",
    "evidence": "registry.py:439 ParamSpec(name='initial_cloud_n_arr', ..., unit='1/cm**3'). The value assigned is get_InitCloudProp.py:138 params['initial_cloud_n_arr'].value = props.n_arr, where props.n_arr = get_density_profile(r_arr, params) (get_InitCloudProp.py:288), which returns nCore/nISM-scaled values in pc^-3 (density_profile.py:79 'Number density at radius r [1/pc^3] (code units)', :109-110 reads nISM/nCore straight from params). Sibling specs are all '1/pc**3': registry.py:437 nEdge, :467 shell_nMax, :473 shell_n_arr, :478 shell_n0, :487-490 n_IF/n_IF_ODE/n_IF_Str, :508 bubble_n_arr. The reader's own reconstruction contradicts the registry: trinity_reader.py:589-590 documents the return as 'density [internal pc^-3]'. Runtime spec units are never applied as conversions — registry.py:659 stores them only as ori_units — so this is label-only.",
    "expected": "unit='1/pc**3', matching the seven sibling density specs and trinity_reader.initial_cloud_profile()'s docstring.",
    "failure_scenario": "A consumer that trusts params['initial_cloud_n_arr'].ori_units (or the identical PARAM_DOCS entry, UNIT-01) converts a pc^-3 array as if it were cm^-3 — or, worse, applies ndens_cgs2au 'to convert to internal units' and lands 2.938e55 too high. Latent today because DROPPED_IN_V2 (run_constants.py:88-92) means v2+ runs no longer persist the array.",
    "repro": "python -c \"from trinity._input.registry import SPECS; d={s.name:s.unit for s in SPECS}; print({k:d[k] for k in ['initial_cloud_n_arr','nEdge','shell_n_arr','bubble_n_arr','shell_n0']})\"",
    "confidence": "high"
  },
  {
    "id": "UNIT-03",
    "file": "trinity/_input/dictionary.py",
    "line": 678,
    "class": "units",
    "severity": "S3",
    "claim": "shell_grav_force_m is written into the snapshot as log10(|value|) under its plain key, with no 'log_' prefix — unlike all four sibling profile arrays — while registry.py:471 declares its unit as pc/Myr**2.",
    "evidence": "dictionary.py:678-685: `if key == \"shell_grav_force_m\": ... y_arr = np.log10(np.maximum(np.abs(np.asarray(val)), eps)); new_r, new_y = self.simplify(...); new_dict[key] = ...` — the log array is stored under `key`, not `'log_' + key`. Every sibling does prefix: :645-649 bubble_T_arr/bubble_n_arr -> 'log_'+key; :654-660 bubble_dTdr_arr -> 'log_'+key; :695-700 shell_n_arr -> 'log_'+key. Only bubble_v_arr (:663-670) is stored linear, and it is stored under its own name, correctly. The in-memory value is genuinely pc/Myr^2 (shell_structure.py:263 grav_ion_force_m = G*m_cum/r^2 = pc^3 Msun^-1 Myr^-2 * Msun * pc^-2), matching registry.py:471. run_constants.py:131 lists 'shell_grav_force_m' next to 'log_shell_n_arr' in FINAL_STATE_EXCLUDE_ARRAYS, confirming the snapshot-side key name. test/test_metadata.py:226 already builds a fixture as np.linspace(-3.0,-0.5,50) — i.e. the test author was thinking in log space.",
    "expected": "Either store under 'log_shell_grav_force_m' (consistent with the four siblings), or store linear. Whichever is chosen, registry.py:471's unit and PARAM_DOCS trinity_reader.py:270 should say which.",
    "failure_scenario": "Any future consumer reads snapshot['shell_grav_force_m'], sees registry unit pc/Myr^2, and treats a value of -3.0 as a (negative!) gravitational acceleration instead of 1e-3 pc/Myr^2. No in-repo consumer reads it back today, which is the only reason this is not already producing wrong numbers.",
    "repro": "grep -n \"log_\\\" + key\\|new_dict\\[key\\]\" trinity/_input/dictionary.py | sed -n '1,20p'  # compare the shell_grav_force_m branch (line 683) against the four 'log_'+key branches",
    "confidence": "high"
  },
  {
    "id": "UNIT-04",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 29,
    "class": "units",
    "severity": "S3",
    "claim": "get_coolingStructure's docstring states the cooling cube is in [erg cm3 / s] (a cooling function Lambda). The shipped tables are volumetric rates [erg cm^-3 s^-1]; the code is right and the docstring is wrong.",
    "evidence": "read_cloudy.py:29 'Cooling rate is in units of [erg cm3 / s]'. Consumers treat the cube as volumetric: net_coolingcurve.py:154,156 `dudt = netcool_interp([...])[0]; return -1*dudt*cvt.dudt_cgs2au` where dudt_cgs2au converts erg/cm^3/s (unit_conversions.py:124-125); bubble_luminosity.py:785-791 `dudt_cond = (heat_cond - cool_cond) * cvt.dudt_cgs2au`. Empirical proof from lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06.dat at T=3162.28 K, phi=1: cool/(n_e*n_H) is flat across four density decades (8.494e-25, 8.229e-25, 8.453e-25, 9.458e-25, 1.173e-24, 1.586e-24) while cool itself spans 3.74e-33 -> 9.13e-29 and cool/n_H^2 is not flat (3.74e-25 -> 9.13e-26). A Lambda of 7e-34 erg cm^3/s at 3000 K would be ~8 orders below any physical value.",
    "expected": "'Cooling rate is a volumetric energy-loss rate in [erg cm^-3 s^-1] (already n_e n_H weighted); the CIE branch's Lambda is the [erg cm^3 s^-1] quantity.'",
    "failure_scenario": "A future contributor reads the docstring, concludes the cube is a Lambda, and 'fixes' the call sites by multiplying by n^2 and swapping dudt_cgs2au for Lambda_cgs2au. That inflates non-CIE cooling by n_cgs^2 (~1e10 at GMC-core density) and silently kills every energy-driven bubble. Equally, it would prompt applying chi_e to the non-CIE branch, double-counting the electron factor already baked into the table.",
    "repro": "python -c \"import numpy as np; d=np.genfromtxt('lib/default/opiate/opiate_cooling_rot_Z1.00_age1.00e+06.dat',names=True); m=np.isclose(d['temp'],d['temp'][0])&np.isclose(d['phi'],d['phi'][0]); r=d[m][:6]; print(r['cool']/(r['nedens']*r['ndens'])); print(r['cool']/r['ndens']**2)\"",
    "confidence": "high"
  },
  {
    "id": "UNIT-05",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 62,
    "class": "units",
    "severity": "S4",
    "claim": "The CloudProperties docstring labels nEdge and n_arr as [cm^-3] (they are pc^-3), and the module's __main__ self-test feeds unconverted cgs-style inputs (nCore=1e3, mu_convert=1.4), so its printed cloud radii are physically wrong by 3.43x.",
    "evidence": "get_InitCloudProp.py:61-62 'nEdge : float / Edge number density [cm^-3]' and :65-66 'n_arr ... Number density profile [cm^-3]'; :100-101 repeats it for nCore/nISM. But :288 n_arr = get_density_profile(r_arr, params) returns pc^-3 (density_profile.py:79), and :290 itself prints nEdge*cvt.ndens_au2cgs to get cm^-3 — i.e. the same file both asserts cm^-3 and converts out of pc^-3 two lines apart. The __main__ block at :566-641 builds MockParam dicts with nCore=1e3 and mu_convert=1.4 (production values are 2.938e58 pc^-3 and 1.178e-57 Msun) and then prints '  nEdge = ... cm^-3'. Measured: compute_rCloud_homogeneous(1e5, 1e3, mu=1.4) = 2.574 pc, versus 8.835 pc with correctly converted inputs — ratio 3.4325.",
    "expected": "Dataclass docstring: '[1/pc^3] (internal)'. Demo block: nCore=1e3*cvt.ndens_cgs2au, mu_convert=1.4*cvt.convert2au('m_H').",
    "failure_scenario": "The self-test is the file's own documentation of correct usage; a reader copies its MockParam pattern into a new analysis or test and silently works in a unit system where mu = 1.4 Msun. Its 'verify_mass_at_rCloud' check still passes because the inconsistency is self-consistent within the demo, so nothing flags it.",
    "repro": "python -c \"import trinity._functions.unit_conversions as cvt; from trinity.cloud_properties.powerLawSphere import compute_rCloud_homogeneous as f; print(f(1e5,1e3,mu=1.4), f(1e5,1e3*cvt.ndens_cgs2au,mu=1.4*cvt.convert2au('m_H')))\"",
    "confidence": "high"
  },
  {
    "id": "UNIT-06",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 57,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Five modules import the unit-conversion table but never use it, which reads as 'this module does unit conversion' when it does not.",
    "evidence": "`import trinity._functions.unit_conversions as cvt` with zero `cvt.` references in the file: run_momentum_phase.py:57, phase1c_transition/run_transition_phase.py:57, phase1b_energy_implicit/run_energy_implicit_phase.py:66, shell_structure/get_shellODE.py:16, sps/read_sps.py:28. (validate_gmc.py:56 and bonnorEbertSphere.py:58 also show 0 `cvt.` hits but use `from ... import` names, so they are genuine.) Not caught by pre-commit: the project's ruff set is F821/F811/F823/E9 (CLAUDE.md), which excludes F401.",
    "expected": "Remove the five unused imports, or leave them and accept the noise — but note that CLAUDE.md rule 3 says pre-existing dead code should be flagged, not deleted.",
    "failure_scenario": "A reviewer scanning for 'which modules convert units?' by import gets five false positives, and a future edit inside one of these files may assume a conversion is already being applied somewhere in the module.",
    "repro": "for f in trinity/phase2_momentum/run_momentum_phase.py trinity/phase1c_transition/run_transition_phase.py trinity/phase1b_energy_implicit/run_energy_implicit_phase.py trinity/shell_structure/get_shellODE.py trinity/sps/read_sps.py; do echo \"$f $(grep -c 'cvt\\.' $f)\"; done",
    "confidence": "high"
  },
  {
    "id": "UNIT-07",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 82,
    "class": "state",
    "severity": "S2",
    "claim": "get_dudt converts its ndens and phi arguments in place with `/=`; with a numpy-array argument this would rewrite the caller's array from internal units to cgs.",
    "evidence": "net_coolingcurve.py:82-83: `ndens /= cvt.ndens_cgs2au` and `phi /= cvt.phi_cgs2au`. For a numpy array these are in-place divisions on the caller's buffer. The single production caller passes scalars — bubble_luminosity.py:427-430 computes ndens from the scalar ODE-state T and phi from the scalar radius r inside _get_bubble_ODE — so the rebinding is local and the code is correct today. test/test_net_coolingcurve.py:56 already documents the hazard: 'get_dudt mutates ndens/phi in place (/=), so pass fresh physical scalars.'",
    "expected": "`ndens = ndens / cvt.ndens_cgs2au` (and likewise for phi), which is scalar-identical and array-safe.",
    "failure_scenario": "Anyone vectorising the bubble-structure RHS (an obvious future perf move — the module already vectorises the conduction-band lookups at bubble_luminosity.py:778-791) passes an n array. get_dudt silently rewrites it to cm^-3; the caller's next use of that array, e.g. the n^2 in a CIE integrand, is then wrong by 2.938e55 per factor of n, with no error raised.",
    "repro": "python -c \"import numpy as np; a=np.array([1.0,2.0]); \\ndef f(x): x/=10; return x\\nf(a); print(a)\"  # demonstrates the aliasing; then note bubble_luminosity.py:427-430 currently passes scalars only",
    "confidence": "medium"
  },
  {
    "id": "UNIT-08",
    "file": "trinity/_input/registry.py",
    "line": 526,
    "class": "units",
    "severity": "S4",
    "claim": "densBE_sigma is the only velocity anywhere in params/metadata stored in km/s instead of internal pc/Myr, breaking the 'metadata is internal units' contract stated elsewhere in the same package.",
    "evidence": "registry.py:526 ParamSpec(name='densBE_sigma', ..., unit='km/s', run_const=True). Written as bonnorEbertSphere.py:564 `params['densBE_sigma'].value = result.c_s / 1.0e5  # c_s [cm/s] -> sigma [km/s]` and get_InitCloudProp.py:349 (same). run_const=True puts it in metadata.json via run_constants.RUN_CONST_KEYS (:78). Every other velocity is pc/Myr: registry.py:427 v2, :433 c_sound, :446 v_mech_total, :504 bubble_v_arr. simulation_end.py:12-15 states the metadata convention is 'INTERNAL units (pc/Myr, pc^-3)'. Pinned as intentional by test/test_mu_audit_drift.py:301-302 (asserts ori_units == 'km/s').",
    "expected": "Either store pc/Myr (value * cvt.v_kms2au) and let show_run convert for display like it does for v2, or add an explicit carve-out to the metadata unit-convention statement at simulation_end.py:12-15.",
    "failure_scenario": "A sweep-analysis script that maps 'every velocity-like key in metadata.json' through INV_CONV.v_au2kms (as sweep_runner.py:543-545 does for v2) converts densBE_sigma a second time, producing a value 1.0227x off — small enough to look plausible and never trip a sanity check.",
    "repro": "python -c \"from trinity._input.registry import SPECS; print([(s.name,s.unit) for s in SPECS if s.unit in ('km/s','pc/Myr')])\"",
    "confidence": "medium"
  },
  {
    "id": "UNIT-09",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 51,
    "class": "units",
    "severity": "S4",
    "claim": "compute_rCloud_homogeneous / compute_rCloud_powerlaw default to mu=1.4 and validate_gmc_params documents mu as '[Msun] (code units, typically 1.4)' — but in code units mu is ~1.18e-57 Msun; 1.4 is the m_H-relative number.",
    "evidence": "powerLawSphere.py:51 `def compute_rCloud_homogeneous(M_cloud, nCore, mu=1.4)` and :77 `..., mu=1.4)`, immediately below a module note (:38-44) correctly stating mu is '[Msun] (converted from m_H units via m_H * g2Msun)'. validate_gmc.py:299-300 'mu : float / Mean molecular weight [Msun] (code units, typically 1.4).' and the module docstring example at :29 `mCloud=1e5, nCore=1e3, mu=1.4, nISM=1.0`. Measured production value: params['mu_convert'].value = 1.1783e-57 Msun. Every production caller passes mu explicitly (get_InitCloudProp.py:170,172,194,231,243; validate_gmc.py:409,416,589,592; sweep_runner.py:116), so the defaults are never exercised in a real run.",
    "expected": "Drop the defaults (make mu a required argument) or set them to `1.4 * convert2au('m_H')`; fix the validate_gmc docstring to say 'typically 1.18e-57 Msun (= 1.4 m_H)'.",
    "failure_scenario": "A new caller relies on the default or copies the docstring example, computes rho = n * 1.4 with n in pc^-3, and gets a cloud radius 3.43x too small (measured: 2.574 pc vs 8.835 pc for M=1e5 Msun, n=1e3 cm^-3). Because rCloud then passes the rCloud_max plausibility check comfortably, nothing rejects it.",
    "repro": "python -c \"import inspect, trinity.cloud_properties.powerLawSphere as m; print(inspect.signature(m.compute_rCloud_homogeneous)); import trinity._functions.unit_conversions as cvt; print('production mu =', 1.4*cvt.convert2au('m_H'))\"",
    "confidence": "high"
  }
]
```
